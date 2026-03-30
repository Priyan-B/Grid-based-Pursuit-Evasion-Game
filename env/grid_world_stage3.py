"""
Stage 3 Grid-world: Police + CCTV on top of traps and traffic.

Observation (40-dim):
    [0:9]   — 3×3 wall/road/goal patch   (same as stage 1 & 2)
    [9:18]  — 3×3 trap overlay           (same as stage 2)
    [18:27] — 3×3 traffic overlay        (same as stage 2)
    [27]    — normalised Manhattan dist to goal  (same as stage 1 & 2)
    [28:37] — 3×3 police overlay  (NEW — 1.0 if police in that cell)
    [37:40] — normalised distance to each of 3 police cars,
              sorted nearest-first  (NEW)

    First 28 dims are IDENTICAL to stage 2 → clean transplant.

Actions: 5  (up, down, left, right, stay)  — unchanged

New mechanics (on top of all stage 2 mechanics)
─────────────────────────────────────────────────────────────────
Police (3 cars)
    Move randomly on road cells each step.  They walk through
    traps and traffic without penalty — only walls block them.
    If any police car occupies the same cell as the thief →
        CAUGHT: large negative reward, episode ends immediately.
    Police do NOT learn in this stage.  They are dumb random walkers.
    The thief must learn to see police in its 3×3 window + distance
    features and avoid them while still navigating to the goal.

CCTV (6 cameras)
    Fixed cells on the grid.  Each camera has 3×3 vision — when the
    thief is within 1 cell of a camera (including diagonals), the
    (thief_position, timestep) is recorded.  CCTV is SILENT — the
    thief has no idea which cells are cameras and gets no reward
    signal from them.  Data is collected for later use by police
    in stage 4.
─────────────────────────────────────────────────────────────────
"""

import numpy as np
from env.grid_world import generate_city, WALL, ROAD, ACTION_DELTAS
from env.grid_world_stage2 import (
    _split_contiguous, TrafficCar,
    N_TRAPS, N_TRAFFIC, TRAP_SHUFFLE_INTERVAL, TRAFFIC_MOVE_INTERVAL,
    STUN_DURATION, TRAP_REWARD, TRAFFIC_PENALTY, MIN_LANE_LENGTH,
)

# ── stage 3 constants ──────────────────────────────────────────
N_POLICE = 3
CAUGHT_REWARD = -100.0           # caught by police — worse than traps

# CCTV camera positions — 6 cameras at key intersections
# covering top-left, top-right, center-left, center, mid-right, bottom
DEFAULT_CCTV_CELLS = [
    (2, 5),     # top corridor, before wall gap
    (2, 11),    # top-right area
    (5, 2),     # left corridor
    (5, 8),     # grid center
    (8, 11),    # right side
    (11, 5),    # bottom area, on path toward goal
]
# ────────────────────────────────────────────────────────────────


class PoliceCar:
    """
    A police car that performs a random walk on road cells.
    Walks through traps and traffic freely — only walls block it.
    """

    def __init__(self, pos, road_set, grid, size):
        self.pos = pos
        self._road_set = road_set
        self._grid = grid
        self._size = size

    def step(self):
        """Move to a random adjacent road cell (or stay if boxed in)."""
        r, c = self.pos
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) in self._road_set:
                neighbors.append((nr, nc))
        if neighbors:
            self.pos = neighbors[np.random.randint(len(neighbors))]


class GridWorldStage3:
    """
    Stage 3 environment: city grid + traps + traffic + police + CCTV.

    Inherits all Stage 2 mechanics and adds police and silent CCTV.
    Traps are always dynamic (Phase B) since Stage 2 training is done.
    """

    OBS_DIM = 40   # 28 (stage 2) + 9 (police overlay) + 3 (police distances)

    def __init__(self, size=15, max_steps=200, rng_seed=42,
                 n_traps=N_TRAPS, n_traffic=N_TRAFFIC,
                 n_police=N_POLICE, cctv_cells=None):
        self.size = size
        self.max_steps = max_steps
        self.n_traps = n_traps
        self.n_traffic = n_traffic
        self.n_police = n_police

        # Always dynamic traps in stage 3 (agent graduated from phase B)
        self.trap_mode = "dynamic"

        # Reuse the same deterministic grid
        self.grid = generate_city(size=size, rng_seed=rng_seed)

        self.road_cells = [
            (r, c) for r in range(size) for c in range(size)
            if self.grid[r, c] == ROAD
        ]
        self.road_set = set(self.road_cells)

        # Goal — same logic
        target = (size - 2, size - 2)
        self.goal = min(
            self.road_cells,
            key=lambda p: abs(p[0] - target[0]) + abs(p[1] - target[1]),
        )

        self._max_dist = float(2 * (size - 1))

        # ── traffic lanes (reuse stage 2 logic) ──
        self._traffic_lanes = self._build_traffic_lanes()

        # ── CCTV cells ──
        if cctv_cells is None:
            self.cctv_cells = set(
                c for c in DEFAULT_CCTV_CELLS if c in self.road_set
            )
        else:
            self.cctv_cells = set(
                c for c in cctv_cells if c in self.road_set
            )

        # Runtime state (set in reset)
        self.agent_pos = None
        self.steps = 0
        self.done = False
        self.traps = set()
        self.traffic_cars = []
        self.police_cars = []           # list of PoliceCar
        self.stun_remaining = 0

        # Per-episode stats
        self.trap_hits = 0
        self.traffic_hits = 0
        self.caught_by_police = False

        # CCTV log — list of (thief_pos, timestep) sightings
        self.cctv_log = []
        # Accumulated across all episodes for saving
        self.cctv_all_episodes = []

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

    def _manhattan(self):
        return abs(self.agent_pos[0] - self.goal[0]) + abs(self.agent_pos[1] - self.goal[1])

    def _manhattan_to(self, pos):
        return abs(self.agent_pos[0] - pos[0]) + abs(self.agent_pos[1] - pos[1])

    def _valid_trap_cells(self):
        exclude = {self.goal, self.agent_pos}
        # Also exclude police starting positions
        for pc in self.police_cars:
            exclude.add(pc.pos)
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
        """Place N police cars on random road cells, not on thief or goal."""
        exclude = {self.goal, self.agent_pos}
        candidates = [c for c in self.road_cells if c not in exclude]
        chosen = np.random.choice(len(candidates), size=self.n_police, replace=False)
        self.police_cars = []
        for i in chosen:
            pc = PoliceCar(candidates[i], self.road_set, self.grid, self.size)
            self.police_cars.append(pc)

    def _check_caught(self):
        """Return True if any police car is on the thief's cell."""
        for pc in self.police_cars:
            if pc.pos == self.agent_pos:
                return True
        return False

    def _record_cctv(self):
        """If thief is within 3×3 vision of any CCTV camera, log the sighting."""
        tr, tc = self.agent_pos
        for (cr, cc) in self.cctv_cells:
            if abs(tr - cr) <= 1 and abs(tc - cc) <= 1:
                self.cctv_log.append((self.agent_pos, self.steps))
                return  # one sighting per step, even if in range of multiple cameras

    # ─────────────────────── observation ───────────────────

    def get_state(self):
        """
        40-dim observation:
            [0:9]   3×3 wall/road/goal patch
            [9:18]  3×3 trap overlay
            [18:27] 3×3 traffic overlay
            [27]    normalised Manhattan distance to goal
            [28:37] 3×3 police overlay
            [37:40] normalised distance to each of 3 police (nearest first)
        """
        tr, tc = self.agent_pos
        traffic_positions = set(car.pos for car in self.traffic_cars)
        police_positions = set(pc.pos for pc in self.police_cars)

        patch = np.zeros(9, dtype=np.float32)
        trap_layer = np.zeros(9, dtype=np.float32)
        traffic_layer = np.zeros(9, dtype=np.float32)
        police_layer = np.zeros(9, dtype=np.float32)

        idx = 0
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                r, c = tr + dr, tc + dc
                # Wall / road / goal
                if r < 0 or r >= self.size or c < 0 or c >= self.size:
                    patch[idx] = 1.0
                elif self.grid[r, c] == WALL:
                    patch[idx] = 1.0
                elif (r, c) == self.goal:
                    patch[idx] = 0.5
                else:
                    patch[idx] = 0.0

                # Trap
                if (r, c) in self.traps:
                    trap_layer[idx] = 1.0

                # Traffic
                if (r, c) in traffic_positions:
                    traffic_layer[idx] = 1.0

                # Police
                if (r, c) in police_positions:
                    police_layer[idx] = 1.0

                idx += 1

        # Goal distance
        norm_dist = self._manhattan() / self._max_dist

        # Police distances — sorted nearest first, normalised
        police_dists = sorted(
            self._manhattan_to(pc.pos) for pc in self.police_cars
        )
        # Pad to exactly n_police entries (in case of fewer cars)
        while len(police_dists) < self.n_police:
            police_dists.append(self._max_dist)
        police_dists_norm = np.array(
            [d / self._max_dist for d in police_dists[:self.n_police]],
            dtype=np.float32,
        )

        return np.concatenate([
            patch, trap_layer, traffic_layer, [norm_dist],
            police_layer, police_dists_norm,
        ])

    # ─────────────────────── reset / step ──────────────────

    def reset(self):
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

        # Save previous episode's CCTV log
        if self.cctv_log:
            self.cctv_all_episodes.append(list(self.cctv_log))
        self.cctv_log = []

        self._spawn_police()
        self._place_traps()
        self._spawn_traffic()

        return self.get_state()

    def step(self, action):
        assert not self.done, "Episode finished — call reset()"
        self.steps += 1

        # ── dynamic trap shuffle ──
        if self.steps % TRAP_SHUFFLE_INTERVAL == 0:
            self._shuffle_traps()

        # ── move traffic (slow) ──
        if self.steps % TRAFFIC_MOVE_INTERVAL == 0:
            for car in self.traffic_cars:
                car.step()

        # ── move police (every step, random walk) ──
        for pc in self.police_cars:
            pc.step()

        # ── caught check: police walked onto thief's cell ──
        if self._check_caught():
            reward = CAUGHT_REWARD
            self.caught_by_police = True
            self.done = True
            self._record_cctv()
            return self.get_state(), reward, self.done

        # ── stun check ──
        if self.stun_remaining > 0:
            self.stun_remaining -= 1
            action = 4

        old_dist = self._manhattan()

        # ── move thief ──
        dr, dc = ACTION_DELTAS[action]
        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc

        wall_hit = False
        if (0 <= nr < self.size and 0 <= nc < self.size
                and self.grid[nr, nc] == ROAD):
            self.agent_pos = (nr, nc)
        else:
            wall_hit = True

        new_dist = self._manhattan()

        # ── CCTV recording (silent) ──
        self._record_cctv()

        # ────────── reward shaping ──────────
        reward = -0.1

        # Goal
        if self.agent_pos == self.goal:
            reward = 100.0
            self.done = True
            return self.get_state(), reward, self.done

        # Caught by police — highest penalty
        if self._check_caught():
            reward = CAUGHT_REWARD
            self.caught_by_police = True
            self.done = True
            return self.get_state(), reward, self.done

        # Trap — lethal
        if self.agent_pos in self.traps:
            reward = TRAP_REWARD
            self.trap_hits += 1
            self.done = True
            return self.get_state(), reward, self.done

        # Wall bump
        if wall_hit:
            reward -= 1.0
        elif action == 4 and self.stun_remaining == 0:
            reward -= 0.3
        else:
            reward += (old_dist - new_dist) * 1.0

        # Traffic
        traffic_positions = set(car.pos for car in self.traffic_cars)
        if self.agent_pos in traffic_positions:
            reward += TRAFFIC_PENALTY
            self.stun_remaining = STUN_DURATION
            self.traffic_hits += 1

        # Time limit
        if self.steps >= self.max_steps:
            self.done = True

        return self.get_state(), reward, self.done

    def save_cctv_data(self, path):
        """Save all accumulated CCTV sightings to a .npy file."""
        # Flush current episode's log
        if self.cctv_log:
            self.cctv_all_episodes.append(list(self.cctv_log))
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.cctv_all_episodes, f)

    def render(self):
        return self.grid