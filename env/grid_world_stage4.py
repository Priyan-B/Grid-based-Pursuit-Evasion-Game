"""
Stage 4 Environment: Police Training against Frozen Thief.

This is a MULTI-AGENT environment.  The external interface serves
2 police agents.  The thief runs internally using a frozen checkpoint.

Step order each tick:
    1. Traps shuffle (if due)
    2. Traffic moves (if due)
    3. Police move (actions from external agents)
    4. Caught check #1 — police walked onto thief
    5. Thief moves (frozen policy, greedy argmax)
    6. CCTV check — update police sightings if thief spotted
    7. Caught check #2 — thief walked into police
    8. Trap / traffic / goal / timeout checks

Police observation (244-dim):
    [0:225]   — full 15×15 grid flattened (wall=1, road=0)
    [225:227] — own position normalised (row/size, col/size)
    [227:229] — goal position normalised
    [229:231] — last CCTV sighting position normalised (-1,-1 if none)
    [231]     — has_sighting flag (0 or 1)
    [232]     — sighting staleness normalised (steps_ago / max_steps)
    [233:235] — relative position to teammate normalised (dr/size, dc/size)
    [235:244] — 3×3 local thief detection (1.0 if thief in cell, else 0.0)

Police actions: 5  (up, down, left, right, stay)
Police walk through traps and traffic — only walls block them.

Police reward:
    Caught thief (own catch):    +100
    Teammate caught thief:       +50
    Thief reached goal:          -50
    Timeout:                     -30
    Per-step:                    -0.1
"""

import numpy as np
import torch
from env.grid_world import generate_city, WALL, ROAD, ACTION_DELTAS
from env.grid_world_stage2 import (
    _split_contiguous, TrafficCar,
    N_TRAPS, N_TRAFFIC, TRAP_SHUFFLE_INTERVAL, TRAFFIC_MOVE_INTERVAL,
    STUN_DURATION, TRAP_REWARD, TRAFFIC_PENALTY, MIN_LANE_LENGTH,
)
from env.grid_world_stage3 import DEFAULT_CCTV_CELLS

# ── stage 4 constants ──────────────────────────────────────────
N_POLICE = 2

# Police rewards
POLICE_CATCH_REWARD = 100.0      # you caught the thief
POLICE_TEAM_CATCH_REWARD = 50.0  # teammate caught the thief
POLICE_THIEF_ESCAPED = -50.0     # thief reached goal
POLICE_TIMEOUT = -30.0           # nobody won
POLICE_STEP_PENALTY = -0.1

# Thief observation was 40-dim in stage 3 with 3 police
THIEF_OBS_DIM_S3 = 40
THIEF_N_POLICE_S3 = 3   # thief's frozen network expects 3 police distances
# ────────────────────────────────────────────────────────────────


class GridWorldStage4:
    """
    Stage 4: Police learn to catch a frozen thief.

    Multi-agent interface:
        reset()                → list of 2 police observations
        step(police_actions)   → list of 2 obs, list of 2 rewards, done
    """

    POLICE_OBS_DIM = 244   # 225 + 2 + 2 + 2 + 1 + 1 + 2 + 9 (local vision)

    def __init__(self, thief_policy, size=15, max_steps=200, rng_seed=42,
                 n_traps=N_TRAPS, n_traffic=N_TRAFFIC,
                 n_police=N_POLICE, cctv_cells=None):
        self.size = size
        self.max_steps = max_steps
        self.n_traps = n_traps
        self.n_traffic = n_traffic
        self.n_police = n_police

        # Frozen thief policy (runs internally)
        self.thief_policy = thief_policy
        self.thief_policy.eval()

        self.grid = generate_city(size=size, rng_seed=rng_seed)
        self.grid_flat = (self.grid.flatten().astype(np.float32))  # 225-dim

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

        # Traffic lanes
        self._traffic_lanes = self._build_traffic_lanes()

        # CCTV
        if cctv_cells is None:
            self.cctv_cells = set(
                c for c in DEFAULT_CCTV_CELLS if c in self.road_set
            )
        else:
            self.cctv_cells = set(
                c for c in cctv_cells if c in self.road_set
            )

        # Normalised goal position (constant)
        self._goal_norm = np.array(
            [self.goal[0] / self.size, self.goal[1] / self.size],
            dtype=np.float32
        )

        # Runtime state
        self.agent_pos = None         # thief position
        self.steps = 0
        self.done = False
        self.traps = set()
        self.traffic_cars = []
        self.police_positions = []    # list of (r, c) for each police
        self.stun_remaining = 0       # thief stun

        # CCTV state for police
        self.last_cctv_sighting = None   # (r, c) or None
        self.sighting_step = -1          # step when last spotted

        # Per-episode stats
        self.trap_hits = 0
        self.traffic_hits = 0
        self.caught_by_police = False
        self.catcher_idx = -1            # which police made the catch

    # ─────────────────── lane building (from stage 2) ──────────

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

    def _manhattan(self, pos_a, pos_b):
        return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])

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
        chosen_idx = np.random.choice(
            len(self._traffic_lanes), size=n, replace=False
        )
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
        """Move police according to given actions. Only walls block them."""
        for i, action in enumerate(actions):
            r, c = self.police_positions[i]
            dr, dc = ACTION_DELTAS[action]
            nr, nc = r + dr, c + dc
            if (nr, nc) in self.road_set:
                self.police_positions[i] = (nr, nc)

    def _check_caught(self):
        """Check if any police is on the thief's cell. Returns catcher index or -1."""
        for i, pos in enumerate(self.police_positions):
            if pos == self.agent_pos:
                return i
        return -1

    def _check_cctv(self):
        """If thief is within 3×3 of any CCTV, update sighting for police."""
        tr, tc = self.agent_pos
        for (cr, cc) in self.cctv_cells:
            if abs(tr - cr) <= 1 and abs(tc - cc) <= 1:
                self.last_cctv_sighting = self.agent_pos
                self.sighting_step = self.steps
                return

    # ─────────────── thief observation (40-dim, stage 3 compat) ───

    def _get_thief_state(self):
        """
        Build the 40-dim observation the frozen thief expects.
        Same layout as Stage 3, but pad to 3 police distances
        since the checkpoint was trained with 3 police.
        """
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

        # Police distances — pad to 3 for stage 3 checkpoint compat
        police_dists = sorted(
            self._manhattan(self.agent_pos, pos) for pos in self.police_positions
        )
        while len(police_dists) < THIEF_N_POLICE_S3:
            police_dists.append(self._max_dist)
        police_dists_norm = np.array(
            [d / self._max_dist for d in police_dists[:THIEF_N_POLICE_S3]],
            dtype=np.float32,
        )

        return np.concatenate([
            patch, trap_layer, traffic_layer, [norm_dist],
            police_layer, police_dists_norm,
        ])

    def _get_thief_action(self):
        """Run frozen thief policy (greedy argmax)."""
        state = self._get_thief_state()
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.thief_policy(state_t)
        return logits.argmax(dim=-1).item()

    # ─────────────── police observation (244-dim) ─────────────

    def get_police_state(self, police_idx):
        """
        244-dim observation for one police agent:
            [0:225]   full grid (wall=1, road=0)
            [225:227] own position normalised
            [227:229] goal position normalised
            [229:231] last CCTV sighting normalised (-1,-1 if none)
            [231]     has_sighting flag
            [232]     sighting staleness normalised
            [233:235] relative position to teammate
            [235:244] 3×3 local thief detection
        """
        own_pos = self.police_positions[police_idx]
        own_norm = np.array(
            [own_pos[0] / self.size, own_pos[1] / self.size],
            dtype=np.float32,
        )

        # CCTV sighting info
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

        # Teammate relative position
        teammate_idx = 1 - police_idx   # works for 2 police
        teammate_pos = self.police_positions[teammate_idx]
        rel_teammate = np.array([
            (teammate_pos[0] - own_pos[0]) / self.size,
            (teammate_pos[1] - own_pos[1]) / self.size,
        ], dtype=np.float32)

        # 3×3 local thief detection — can the police SEE the thief nearby?
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
            self.grid_flat,     # 225
            own_norm,           # 2
            self._goal_norm,    # 2
            sight_norm,         # 2
            has_sighting,       # 1
            staleness,          # 1
            rel_teammate,       # 2
            thief_local,        # 9
        ])

    # ─────────────────────── reset / step ──────────────────

    def reset(self):
        """Reset and return list of police observations."""
        # Spawn thief
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

        # Reset CCTV state
        self.last_cctv_sighting = None
        self.sighting_step = -1

        self._spawn_police()
        self._place_traps()
        self._spawn_traffic()

        return [self.get_police_state(i) for i in range(self.n_police)]

    def step(self, police_actions):
        """
        Take one env step.

        Args:
            police_actions: list of 2 ints (action per police agent)

        Returns:
            observations: list of 2 numpy arrays (235-dim each)
            rewards:      list of 2 floats
            done:         bool
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

        # ── move police (external actions) ──
        self._move_police(police_actions)

        # ── caught check #1: police walked onto thief ──
        catcher = self._check_caught()
        if catcher >= 0:
            self.caught_by_police = True
            self.catcher_idx = catcher
            self.done = True
            rewards = []
            for i in range(self.n_police):
                if i == catcher:
                    rewards.append(POLICE_CATCH_REWARD)
                else:
                    rewards.append(POLICE_TEAM_CATCH_REWARD)
            obs = [self.get_police_state(i) for i in range(self.n_police)]
            return obs, rewards, self.done

        # ── thief moves (frozen policy) ──
        if self.stun_remaining > 0:
            self.stun_remaining -= 1
            thief_action = 4  # forced stay
        else:
            thief_action = self._get_thief_action()

        old_pos = self.agent_pos
        dr, dc = ACTION_DELTAS[thief_action]
        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc
        if (nr, nc) in self.road_set:
            self.agent_pos = (nr, nc)

        # ── CCTV check (live — updates police sighting) ──
        self._check_cctv()

        # ── caught check #2: thief walked into police ──
        catcher = self._check_caught()
        if catcher >= 0:
            self.caught_by_police = True
            self.catcher_idx = catcher
            self.done = True
            rewards = []
            for i in range(self.n_police):
                if i == catcher:
                    rewards.append(POLICE_CATCH_REWARD)
                else:
                    rewards.append(POLICE_TEAM_CATCH_REWARD)
            obs = [self.get_police_state(i) for i in range(self.n_police)]
            return obs, rewards, self.done

        # ── thief reached goal ──
        if self.agent_pos == self.goal:
            self.done = True
            rewards = [POLICE_THIEF_ESCAPED] * self.n_police
            obs = [self.get_police_state(i) for i in range(self.n_police)]
            return obs, rewards, self.done

        # ── thief hit trap (thief dies — police didn't cause it) ──
        if self.agent_pos in self.traps:
            self.trap_hits += 1
            self.done = True
            # Neutral for police — they didn't catch, thief didn't escape
            rewards = [POLICE_STEP_PENALTY] * self.n_police
            obs = [self.get_police_state(i) for i in range(self.n_police)]
            return obs, rewards, self.done

        # ── thief hit traffic ──
        traffic_positions = set(car.pos for car in self.traffic_cars)
        if self.agent_pos in traffic_positions:
            self.stun_remaining = STUN_DURATION
            self.traffic_hits += 1

        # ── timeout ──
        if self.steps >= self.max_steps:
            self.done = True
            rewards = [POLICE_TIMEOUT] * self.n_police
            obs = [self.get_police_state(i) for i in range(self.n_police)]
            return obs, rewards, self.done

        # ── normal step ──
        rewards = [POLICE_STEP_PENALTY] * self.n_police
        obs = [self.get_police_state(i) for i in range(self.n_police)]
        return obs, rewards, self.done

    def render(self):
        return self.grid