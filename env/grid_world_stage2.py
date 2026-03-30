"""
Stage 2 Grid-world: Traps and Traffic on top of the base city grid.

Observation (28-dim):
    [0:9]   — 3×3 local patch: walls / road / goal  (same encoding as stage 1)
    [9:18]  — 3×3 trap overlay: 1.0 if trap present, else 0.0
    [18:27] — 3×3 traffic overlay: 1.0 if traffic car present, else 0.0
    [27]    — normalised Manhattan distance to goal

Actions: 5  (up, down, left, right, stay)  — unchanged

New mechanics
─────────────────────────────────────────────────────────────────
Traps
    Phase A (static):  placed once at episode reset, fixed all episode.
    Phase B (dynamic): teleport every TRAP_SHUFFLE_INTERVAL steps.
    Stepping on a trap → LETHAL.  Episode ends immediately with a
    large negative reward.  The thief must learn to read the trap
    overlay and route around them completely.

Traffic
    N_TRAFFIC cars, a mix of vertical and horizontal.
    Each car patrols a contiguous road segment (lane) — it never
    passes through walls or buildings.  Cars bounce at lane ends.
    Lanes that pass through the safe zone (goal) are excluded.
    Cars move once every TRAFFIC_MOVE_INTERVAL env steps (slow).
    If the thief occupies the same cell as a traffic car →
        penalty + stun for STUN_DURATION steps (thief forced to 'stay').
    Traffic cars pass through each other; only the thief is affected.
─────────────────────────────────────────────────────────────────
"""

import numpy as np
from env.grid_world import generate_city, WALL, ROAD, ACTION_DELTAS

# ── configurable constants ──────────────────────────────────────
N_TRAPS = 6
N_TRAFFIC = 5
TRAP_SHUFFLE_INTERVAL = 50      # steps between dynamic-trap teleports
TRAFFIC_MOVE_INTERVAL = 3       # traffic moves once every N thief steps
STUN_DURATION = 2               # steps the thief is frozen after traffic hit

TRAP_REWARD = -50.0             # lethal — episode ends on trap hit
TRAFFIC_PENALTY = -3.0
MIN_LANE_LENGTH = 3             # minimum cells for a valid traffic lane
# ────────────────────────────────────────────────────────────────


def _split_contiguous(indices):
    """
    Split a sorted list of integers into contiguous runs.
    e.g. [1,2,3, 7,8,9] → [[1,2,3], [7,8,9]]
    """
    if not indices:
        return []
    segments = []
    current = [indices[0]]
    for i in range(1, len(indices)):
        if indices[i] == current[-1] + 1:
            current.append(indices[i])
        else:
            segments.append(current)
            current = [indices[i]]
    segments.append(current)
    return segments


class TrafficCar:
    """
    A car that patrols along a contiguous lane (list of cells),
    bouncing at both ends.  Works for both vertical and horizontal lanes.
    """

    def __init__(self, lane):
        """
        lane : list of (row, col) tuples — ordered, contiguous road cells.
               Vertical lane: same col, consecutive rows.
               Horizontal lane: same row, consecutive cols.
        """
        self.lane = lane                # list of (r, c)
        self.idx = 0                    # index into lane
        self.direction = 1              # +1 = forward, -1 = backward

    @property
    def pos(self):
        return self.lane[self.idx]

    def step(self):
        """Advance one cell; bounce at lane ends."""
        next_idx = self.idx + self.direction
        if next_idx < 0 or next_idx >= len(self.lane):
            self.direction *= -1
            next_idx = self.idx + self.direction
        self.idx = next_idx


class GridWorldStage2:
    """
    Stage 2 environment: base city grid + traps + traffic.

    Modes
    ─────
    trap_mode = "static"   → Phase A (traps fixed per episode)
    trap_mode = "dynamic"  → Phase B (traps teleport every TRAP_SHUFFLE_INTERVAL steps)
    """

    OBS_DIM = 28   # 9 (walls) + 9 (traps) + 9 (traffic) + 1 (distance)

    def __init__(self, size=15, max_steps=200, rng_seed=42,
                 trap_mode="static", n_traps=N_TRAPS, n_traffic=N_TRAFFIC):
        self.size = size
        self.max_steps = max_steps
        self.trap_mode = trap_mode
        self.n_traps = n_traps
        self.n_traffic = n_traffic

        # Reuse the same deterministic grid from stage 1
        self.grid = generate_city(size=size, rng_seed=rng_seed)

        self.road_cells = [
            (r, c) for r in range(size) for c in range(size)
            if self.grid[r, c] == ROAD
        ]
        self.road_set = set(self.road_cells)

        # Goal — same logic as stage 1
        target = (size - 2, size - 2)
        self.goal = min(
            self.road_cells,
            key=lambda p: abs(p[0] - target[0]) + abs(p[1] - target[1]),
        )

        self._max_dist = float(2 * (size - 1))

        # ── build contiguous traffic lanes ──
        self._traffic_lanes = self._build_traffic_lanes()

        # Runtime state (set in reset)
        self.agent_pos = None
        self.steps = 0
        self.done = False
        self.traps = set()              # set of (r, c)
        self.traffic_cars = []          # list of TrafficCar
        self.stun_remaining = 0         # remaining forced-stay steps

        # Per-episode stats
        self.trap_hits = 0
        self.traffic_hits = 0

    # ─────────────────── lane building ────────────────────

    def _build_traffic_lanes(self):
        """
        Build all valid traffic lanes: contiguous road segments along
        columns (vertical) and rows (horizontal).

        A valid lane must:
        - Have at least MIN_LANE_LENGTH cells (room to bounce)
        - NOT pass through the goal cell
        """
        lanes = []

        # ── Vertical lanes (fixed column, varying rows) ──
        for c in range(self.size):
            road_rows = sorted(
                r for r in range(self.size) if self.grid[r, c] == ROAD
            )
            for seg in _split_contiguous(road_rows):
                if len(seg) < MIN_LANE_LENGTH:
                    continue
                cells = [(r, c) for r in seg]
                # Skip if lane passes through goal
                if self.goal in cells:
                    continue
                lanes.append(cells)

        # ── Horizontal lanes (fixed row, varying columns) ──
        for r in range(self.size):
            road_cols = sorted(
                c for c in range(self.size) if self.grid[r, c] == ROAD
            )
            for seg in _split_contiguous(road_cols):
                if len(seg) < MIN_LANE_LENGTH:
                    continue
                cells = [(r, c) for c in seg]
                # Skip if lane passes through goal
                if self.goal in cells:
                    continue
                lanes.append(cells)

        return lanes

    # ─────────────────────── helpers ───────────────────────

    def _manhattan(self):
        return abs(self.agent_pos[0] - self.goal[0]) + abs(self.agent_pos[1] - self.goal[1])

    def _valid_trap_cells(self):
        """Cells where a trap can be placed: road, not goal, not thief start."""
        exclude = {self.goal, self.agent_pos}
        return [c for c in self.road_cells if c not in exclude]

    def _place_traps(self):
        """Randomly place n_traps on valid cells."""
        candidates = self._valid_trap_cells()
        n = min(self.n_traps, len(candidates))
        chosen = np.random.choice(len(candidates), size=n, replace=False)
        self.traps = set(candidates[i] for i in chosen)

    def _shuffle_traps(self):
        """Teleport all traps to new valid positions (Phase B)."""
        exclude = {self.goal, self.agent_pos}
        candidates = [c for c in self.road_cells if c not in exclude]
        n = min(self.n_traps, len(candidates))
        chosen = np.random.choice(len(candidates), size=n, replace=False)
        self.traps = set(candidates[i] for i in chosen)

    def _spawn_traffic(self):
        """Pick n_traffic random lanes and place one car on each."""
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
            # Random starting position within the lane
            car.idx = np.random.randint(len(lane))
            # Random initial direction
            car.direction = np.random.choice([-1, 1])
            self.traffic_cars.append(car)

    # ─────────────────────── observation ───────────────────

    def get_state(self):
        """
        28-dim observation:
            [0:9]   3×3 wall/road/goal patch  (same encoding as stage 1)
            [9:18]  3×3 trap overlay
            [18:27] 3×3 traffic overlay
            [27]    normalised Manhattan distance
        """
        tr, tc = self.agent_pos
        traffic_positions = set(car.pos for car in self.traffic_cars)

        patch = np.zeros(9, dtype=np.float32)
        trap_layer = np.zeros(9, dtype=np.float32)
        traffic_layer = np.zeros(9, dtype=np.float32)

        idx = 0
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                r, c = tr + dr, tc + dc
                # Wall / road / goal layer
                if r < 0 or r >= self.size or c < 0 or c >= self.size:
                    patch[idx] = 1.0          # boundary → wall
                elif self.grid[r, c] == WALL:
                    patch[idx] = 1.0
                elif (r, c) == self.goal:
                    patch[idx] = 0.5
                else:
                    patch[idx] = 0.0

                # Trap layer
                if (r, c) in self.traps:
                    trap_layer[idx] = 1.0

                # Traffic layer
                if (r, c) in traffic_positions:
                    traffic_layer[idx] = 1.0

                idx += 1

        norm_dist = self._manhattan() / self._max_dist
        return np.concatenate([patch, trap_layer, traffic_layer, [norm_dist]])

    # ─────────────────────── reset / step ──────────────────

    def reset(self):
        # Spawn thief on random road cell (not goal)
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

        self._place_traps()
        self._spawn_traffic()

        return self.get_state()

    def step(self, action):
        assert not self.done, "Episode finished — call reset()"
        self.steps += 1

        # ── dynamic trap shuffle (Phase B) ──
        if self.trap_mode == "dynamic" and self.steps % TRAP_SHUFFLE_INTERVAL == 0:
            self._shuffle_traps()

        # ── move traffic cars (slow — only every N steps) ──
        if self.steps % TRAFFIC_MOVE_INTERVAL == 0:
            for car in self.traffic_cars:
                car.step()

        # ── stun check: if stunned, force stay ──
        if self.stun_remaining > 0:
            self.stun_remaining -= 1
            action = 4  # forced stay

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

        # ────────── reward shaping ──────────
        reward = -0.1                             # base step penalty

        # Goal check (highest priority)
        if self.agent_pos == self.goal:
            reward = 100.0
            self.done = True
            return self.get_state(), reward, self.done

        # Trap check — LETHAL: episode ends immediately
        if self.agent_pos in self.traps:
            reward = TRAP_REWARD
            self.trap_hits += 1
            self.done = True
            return self.get_state(), reward, self.done

        # Wall bump
        if wall_hit:
            reward -= 1.0
        elif action == 4 and self.stun_remaining == 0:
            # Voluntary stay (not forced by stun) — small penalty
            reward -= 0.3
        else:
            # Distance shaping
            reward += (old_dist - new_dist) * 1.0

        # Traffic check
        traffic_positions = set(car.pos for car in self.traffic_cars)
        if self.agent_pos in traffic_positions:
            reward += TRAFFIC_PENALTY
            self.stun_remaining = STUN_DURATION
            self.traffic_hits += 1

        # Time limit
        if self.steps >= self.max_steps:
            self.done = True

        return self.get_state(), reward, self.done

    def render(self):
        return self.grid