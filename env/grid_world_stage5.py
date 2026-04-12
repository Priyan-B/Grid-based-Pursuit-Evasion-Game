"""
Stage 5 Environment: Adversarial Co-Training.

Both thief and police are LEARNING simultaneously.  No one is frozen.
The environment takes 1 thief action + 2 police actions each step
and returns observations and rewards for all three agents.

Step order each tick:
    1. Traps shuffle (if due)
    2. Traffic moves (if due)
    3. Police move (external actions)
    4. Caught check #1 — police walked onto thief
    5. Thief moves (external action)
    6. CCTV check — update police sightings
    7. Caught check #2 — thief walked into police
    8. Goal / trap / traffic / timeout checks

Thief observation (40-dim, same as stage 3):
    [0:9]   3×3 wall/road/goal    [9:18]  3×3 traps
    [18:27] 3×3 traffic           [27]    goal distance
    [28:37] 3×3 police            [37:40] police distances (3 slots, padded)

Police observation (244-dim, same as stage 4):
    [0:225]   full grid            [225:227] own position
    [227:229] goal position        [229:231] CCTV sighting
    [231]     has_sighting         [232]     staleness
    [233:235] teammate relative    [235:244] 3×3 thief detection

Thief rewards:   +100 goal, -100 caught, -50 trap, -3 traffic, -0.1 step
Police rewards:  +100 catch, +50 teammate, -50 escape, -30 timeout, -0.1 step
"""

import numpy as np
from env.grid_world import generate_city, WALL, ROAD, ACTION_DELTAS
from env.grid_world_stage2 import (
    _split_contiguous, TrafficCar,
    N_TRAPS, N_TRAFFIC, TRAP_SHUFFLE_INTERVAL, TRAFFIC_MOVE_INTERVAL,
    STUN_DURATION, TRAP_REWARD, TRAFFIC_PENALTY, MIN_LANE_LENGTH,
)
from env.grid_world_stage3 import DEFAULT_CCTV_CELLS, CAUGHT_REWARD

# ── constants ──────────────────────────────────────────────────
N_POLICE = 2
THIEF_N_POLICE_SLOTS = 3   # thief obs expects 3 police dist slots (stage 3 compat)

# Police rewards
POLICE_CATCH_REWARD = 100.0
POLICE_TEAM_CATCH_REWARD = 50.0
POLICE_THIEF_ESCAPED = -50.0
POLICE_TIMEOUT = -30.0
POLICE_STEP_PENALTY = -0.1
# ────────────────────────────────────────────────────────────────


class GridWorldStage5:
    """
    Stage 5: Adversarial co-training environment.

    Interface:
        reset()  → thief_obs, [police0_obs, police1_obs]
        step(thief_action, police_actions)
                 → thief_obs, thief_reward,
                   [police0_obs, police1_obs], [police0_reward, police1_reward],
                   done
    """

    THIEF_OBS_DIM = 40
    POLICE_OBS_DIM = 244

    def __init__(self, size=15, max_steps=200, rng_seed=42,
                 n_traps=N_TRAPS, n_traffic=N_TRAFFIC,
                 n_police=N_POLICE, cctv_cells=None):
        self.size = size
        self.max_steps = max_steps
        self.n_traps = n_traps
        self.n_traffic = n_traffic
        self.n_police = n_police

        self.grid = generate_city(size=size, rng_seed=rng_seed)
        self.grid_flat = self.grid.flatten().astype(np.float32)

        self.road_cells = [
            (r, c) for r in range(size) for c in range(size)
            if self.grid[r, c] == ROAD
        ]
        self.road_set = set(self.road_cells)

        target = (size - 2, size - 2)
        self.goal = min(
            self.road_cells,
            key=lambda p: abs(p[0] - target[0]) + abs(p[1] - target[1]),
        )
        self._max_dist = float(2 * (size - 1))
        self._goal_norm = np.array(
            [self.goal[0] / self.size, self.goal[1] / self.size],
            dtype=np.float32,
        )

        self._traffic_lanes = self._build_traffic_lanes()

        if cctv_cells is None:
            self.cctv_cells = set(
                c for c in DEFAULT_CCTV_CELLS if c in self.road_set
            )
        else:
            self.cctv_cells = set(
                c for c in cctv_cells if c in self.road_set
            )

        # Runtime state
        self.agent_pos = None
        self.steps = 0
        self.done = False
        self.traps = set()
        self.traffic_cars = []
        self.police_positions = []
        self.stun_remaining = 0

        # CCTV state
        self.last_cctv_sighting = None
        self.sighting_step = -1

        # Per-episode stats
        self.trap_hits = 0
        self.traffic_hits = 0
        self.caught_by_police = False
        self.catcher_idx = -1
        self.thief_reached_goal = False

    # ─────────────────── lane building ──────────────────────

    def _build_traffic_lanes(self):
        lanes = []
        for c in range(self.size):
            road_rows = sorted(
                r for r in range(self.size) if self.grid[r, c] == ROAD
            )
            for seg in _split_contiguous(road_rows):
                if len(seg) < MIN_LANE_LENGTH:
                    continue
                cells = [(r, c) for r in seg]
                if self.goal in cells:
                    continue
                lanes.append(cells)
        for r in range(self.size):
            road_cols = sorted(
                c for c in range(self.size) if self.grid[r, c] == ROAD
            )
            for seg in _split_contiguous(road_cols):
                if len(seg) < MIN_LANE_LENGTH:
                    continue
                cells = [(r, c) for c in seg]
                if self.goal in cells:
                    continue
                lanes.append(cells)
        return lanes

    # ─────────────────────── helpers ───────────────────────

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _valid_trap_cells(self):
        exclude = {self.goal, self.agent_pos}
        for pos in self.police_positions:
            exclude.add(pos)
        return [c for c in self.road_cells if c not in exclude]

    def _place_traps(self):
        candidates = self._valid_trap_cells()
        n = min(self.n_traps, len(candidates))
        chosen = np.random.choice(len(candidates), size=n, replace=False)
        self.traps = set(candidates[i] for i in chosen)

    def _shuffle_traps(self):
        exclude = {self.goal, self.agent_pos}
        candidates = [c for c in self.road_cells if c not in exclude]
        n = min(self.n_traps, len(candidates))
        chosen = np.random.choice(len(candidates), size=n, replace=False)
        self.traps = set(candidates[i] for i in chosen)

    def _spawn_traffic(self):
        if not self._traffic_lanes:
            self.traffic_cars = []
            return
        n = min(self.n_traffic, len(self._traffic_lanes))
        chosen_idx = np.random.choice(len(self._traffic_lanes), size=n, replace=False)
        self.traffic_cars = []
        for i in chosen_idx:
            lane = self._traffic_lanes[i]
            car = TrafficCar(lane)
            car.idx = np.random.randint(len(lane))
            car.direction = np.random.choice([-1, 1])
            self.traffic_cars.append(car)

    def _spawn_police(self):
        exclude = {self.goal, self.agent_pos}
        candidates = [c for c in self.road_cells if c not in exclude]
        chosen = np.random.choice(len(candidates), size=self.n_police, replace=False)
        self.police_positions = [candidates[i] for i in chosen]

    def _move_police(self, actions):
        for i, action in enumerate(actions):
            r, c = self.police_positions[i]
            dr, dc = ACTION_DELTAS[action]
            nr, nc = r + dr, c + dc
            if (nr, nc) in self.road_set:
                self.police_positions[i] = (nr, nc)

    def _check_caught(self):
        for i, pos in enumerate(self.police_positions):
            if pos == self.agent_pos:
                return i
        return -1

    def _check_cctv(self):
        tr, tc = self.agent_pos
        for (cr, cc) in self.cctv_cells:
            if abs(tr - cr) <= 1 and abs(tc - cc) <= 1:
                self.last_cctv_sighting = self.agent_pos
                self.sighting_step = self.steps
                return

    # ─────────────── thief observation (40-dim) ───────────────

    def get_thief_state(self):
        tr, tc = self.agent_pos
        traffic_positions = set(car.pos for car in self.traffic_cars)
        police_set = set(self.police_positions)

        patch = np.zeros(9, dtype=np.float32)
        trap_layer = np.zeros(9, dtype=np.float32)
        traffic_layer = np.zeros(9, dtype=np.float32)
        police_layer = np.zeros(9, dtype=np.float32)

        idx = 0
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                r, c = tr + dr, tc + dc
                if r < 0 or r >= self.size or c < 0 or c >= self.size:
                    patch[idx] = 1.0
                elif self.grid[r, c] == WALL:
                    patch[idx] = 1.0
                elif (r, c) == self.goal:
                    patch[idx] = 0.5
                else:
                    patch[idx] = 0.0
                if (r, c) in self.traps:
                    trap_layer[idx] = 1.0
                if (r, c) in traffic_positions:
                    traffic_layer[idx] = 1.0
                if (r, c) in police_set:
                    police_layer[idx] = 1.0
                idx += 1

        norm_dist = self._manhattan(self.agent_pos, self.goal) / self._max_dist

        police_dists = sorted(
            self._manhattan(self.agent_pos, pos) for pos in self.police_positions
        )
        while len(police_dists) < THIEF_N_POLICE_SLOTS:
            police_dists.append(self._max_dist)
        police_dists_norm = np.array(
            [d / self._max_dist for d in police_dists[:THIEF_N_POLICE_SLOTS]],
            dtype=np.float32,
        )

        return np.concatenate([
            patch, trap_layer, traffic_layer, [norm_dist],
            police_layer, police_dists_norm,
        ])

    # ─────────────── police observation (244-dim) ─────────────

    def get_police_state(self, police_idx):
        own_pos = self.police_positions[police_idx]
        own_norm = np.array(
            [own_pos[0] / self.size, own_pos[1] / self.size],
            dtype=np.float32,
        )

        if self.last_cctv_sighting is not None:
            sight_norm = np.array([
                self.last_cctv_sighting[0] / self.size,
                self.last_cctv_sighting[1] / self.size,
            ], dtype=np.float32)
            has_sighting = np.array([1.0], dtype=np.float32)
            staleness = np.array([
                (self.steps - self.sighting_step) / self.max_steps
            ], dtype=np.float32)
        else:
            sight_norm = np.array([-1.0, -1.0], dtype=np.float32)
            has_sighting = np.array([0.0], dtype=np.float32)
            staleness = np.array([1.0], dtype=np.float32)

        teammate_idx = 1 - police_idx
        teammate_pos = self.police_positions[teammate_idx]
        rel_teammate = np.array([
            (teammate_pos[0] - own_pos[0]) / self.size,
            (teammate_pos[1] - own_pos[1]) / self.size,
        ], dtype=np.float32)

        pr, pc = own_pos
        thief_local = np.zeros(9, dtype=np.float32)
        idx = 0
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                r, c = pr + dr, pc + dc
                if (r, c) == self.agent_pos:
                    thief_local[idx] = 1.0
                idx += 1

        return np.concatenate([
            self.grid_flat, own_norm, self._goal_norm,
            sight_norm, has_sighting, staleness,
            rel_teammate, thief_local,
        ])

    # ─────────────────────── reset / step ──────────────────

    def reset(self):
        """Returns: thief_obs, [police0_obs, police1_obs]"""
        while True:
            idx = np.random.randint(len(self.road_cells))
            pos = self.road_cells[idx]
            if pos != self.goal:
                break
        self.agent_pos = pos
        self.steps = 0
        self.done = False
        self.stun_remaining = 0
        self.trap_hits = 0
        self.traffic_hits = 0
        self.caught_by_police = False
        self.catcher_idx = -1
        self.thief_reached_goal = False

        self.last_cctv_sighting = None
        self.sighting_step = -1

        self._spawn_police()
        self._place_traps()
        self._spawn_traffic()

        thief_obs = self.get_thief_state()
        police_obs = [self.get_police_state(i) for i in range(self.n_police)]
        return thief_obs, police_obs

    def step(self, thief_action, police_actions):
        """
        Args:
            thief_action:   int (0-4)
            police_actions: list of 2 ints

        Returns:
            thief_obs, thief_reward,
            police_obs_list, police_rewards_list,
            done
        """
        assert not self.done, "Episode finished — call reset()"
        self.steps += 1

        # ── dynamic trap shuffle ──
        if self.steps % TRAP_SHUFFLE_INTERVAL == 0:
            self._shuffle_traps()

        # ── move traffic (slow) ──
        if self.steps % TRAFFIC_MOVE_INTERVAL == 0:
            for car in self.traffic_cars:
                car.step()

        # ── move police ──
        self._move_police(police_actions)

        # ── caught check #1: police walked onto thief ──
        catcher = self._check_caught()
        if catcher >= 0:
            self.caught_by_police = True
            self.catcher_idx = catcher
            self.done = True
            thief_reward = CAUGHT_REWARD
            police_rewards = [
                POLICE_CATCH_REWARD if i == catcher else POLICE_TEAM_CATCH_REWARD
                for i in range(self.n_police)
            ]
            return (self.get_thief_state(), thief_reward,
                    [self.get_police_state(i) for i in range(self.n_police)],
                    police_rewards, self.done)

        # ── move thief (external action) ──
        if self.stun_remaining > 0:
            self.stun_remaining -= 1
            thief_action = 4  # forced stay

        old_dist = self._manhattan(self.agent_pos, self.goal)

        dr, dc = ACTION_DELTAS[thief_action]
        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc
        wall_hit = False
        if (nr, nc) in self.road_set:
            self.agent_pos = (nr, nc)
        else:
            wall_hit = True

        new_dist = self._manhattan(self.agent_pos, self.goal)

        # ── CCTV check ──
        self._check_cctv()

        # ── caught check #2: thief walked into police ──
        catcher = self._check_caught()
        if catcher >= 0:
            self.caught_by_police = True
            self.catcher_idx = catcher
            self.done = True
            thief_reward = CAUGHT_REWARD
            police_rewards = [
                POLICE_CATCH_REWARD if i == catcher else POLICE_TEAM_CATCH_REWARD
                for i in range(self.n_police)
            ]
            return (self.get_thief_state(), thief_reward,
                    [self.get_police_state(i) for i in range(self.n_police)],
                    police_rewards, self.done)

        # ── thief reached goal ──
        if self.agent_pos == self.goal:
            self.thief_reached_goal = True
            self.done = True
            thief_reward = 100.0
            police_rewards = [POLICE_THIEF_ESCAPED] * self.n_police
            return (self.get_thief_state(), thief_reward,
                    [self.get_police_state(i) for i in range(self.n_police)],
                    police_rewards, self.done)

        # ── thief hit trap ──
        if self.agent_pos in self.traps:
            self.trap_hits += 1
            self.done = True
            thief_reward = TRAP_REWARD
            police_rewards = [POLICE_STEP_PENALTY] * self.n_police
            return (self.get_thief_state(), thief_reward,
                    [self.get_police_state(i) for i in range(self.n_police)],
                    police_rewards, self.done)

        # ── thief reward shaping (normal step) ──
        thief_reward = -0.1
        if wall_hit:
            thief_reward -= 1.0
        elif thief_action == 4 and self.stun_remaining == 0:
            thief_reward -= 0.3
        else:
            thief_reward += (old_dist - new_dist) * 1.0

        # ── thief traffic check ──
        traffic_positions = set(car.pos for car in self.traffic_cars)
        if self.agent_pos in traffic_positions:
            thief_reward += TRAFFIC_PENALTY
            self.stun_remaining = STUN_DURATION
            self.traffic_hits += 1

        # ── timeout ──
        if self.steps >= self.max_steps:
            self.done = True
            police_rewards = [POLICE_TIMEOUT] * self.n_police
        else:
            police_rewards = [POLICE_STEP_PENALTY] * self.n_police

        return (self.get_thief_state(), thief_reward,
                [self.get_police_state(i) for i in range(self.n_police)],
                police_rewards, self.done)

    def render(self):
        return self.grid