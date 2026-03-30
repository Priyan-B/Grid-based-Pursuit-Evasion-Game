"""
Grid-world environment for the thief navigation task.

Observation: 3×3 local view (9 floats) + normalised Manhattan distance to goal (1 float)
             → total obs_dim = 10.

The extra distance signal doesn't "give away" the goal location (the agent
still can't see through walls), but it provides a gradient that helps the
policy learn direction even when the goal is outside the 3×3 window.
"""

import numpy as np
from collections import deque

# Cell constants
WALL = 1
ROAD = 0

# Actions: 0=up, 1=down, 2=left, 3=right, 4=stay
ACTION_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}
ACTION_NAMES = {0: "up", 1: "down", 2: "left", 3: "right", 4: "stay"}


def generate_city(size=15, wall_prob=0.25, rng_seed=42):
    """Create a city grid with walls and guaranteed connectivity via BFS.
    Outer ring is always walls. Interior cells are randomly walls or roads,
    then a flood-fill keeps only the largest connected road component and
    converts isolated pockets to walls."""
    rng = np.random.RandomState(rng_seed)
    grid = np.ones((size, size), dtype=np.int8)

    for r in range(1, size - 1):
        for c in range(1, size - 1):
            grid[r, c] = WALL if rng.random() < wall_prob else ROAD

    # Structured corridors
    for r in range(2, size - 2, 3):
        grid[r, 1:size - 1] = ROAD
    for c in range(2, size - 2, 3):
        grid[1:size - 1, c] = ROAD

    # BFS to find largest connected component
    visited = np.zeros_like(grid, dtype=bool)
    components = []
    for r in range(size):
        for c in range(size):
            if grid[r, c] == ROAD and not visited[r, c]:
                queue = deque([(r, c)])
                visited[r, c] = True
                comp = []
                while queue:
                    cr, cc = queue.popleft()
                    comp.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < size and 0 <= nc < size
                                and not visited[nr, nc]
                                and grid[nr, nc] == ROAD):
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                components.append(comp)

    largest = max(components, key=len)
    largest_set = set(largest)
    for r in range(size):
        for c in range(size):
            if grid[r, c] == ROAD and (r, c) not in largest_set:
                grid[r, c] = WALL

    return grid


class GridWorld:
    """
    Thief-in-a-city grid environment.

    Observation space : 10-dim float vector
        [0:9]  — 3×3 local patch (0=road, 1=wall/boundary, 0.5=goal)
        [9]    — normalised Manhattan distance to goal  (0 = at goal, 1 = max)

    Action space : 5  (up, down, left, right, stay)
    """

    OBS_DIM = 10  # 3×3 patch (9) + normalised distance (1)

    def __init__(self, size=15, max_steps=200, rng_seed=42):
        self.size = size
        self.max_steps = max_steps
        self.grid = generate_city(size=size, rng_seed=rng_seed)

        self.road_cells = [
            (r, c) for r in range(size) for c in range(size)
            if self.grid[r, c] == ROAD
        ]

        # Goal near bottom-right corner
        target = (size - 2, size - 2)
        self.goal = min(
            self.road_cells,
            key=lambda p: abs(p[0] - target[0]) + abs(p[1] - target[1]),
        )

        # Maximum possible Manhattan distance (for normalisation)
        self._max_dist = float(2 * (size - 1))

        self.agent_pos = None
        self.steps = 0
        self.done = False

    def reset(self):
        while True:
            idx = np.random.randint(len(self.road_cells))
            pos = self.road_cells[idx]
            if pos != self.goal:
                break
        self.agent_pos = pos
        self.steps = 0
        self.done = False
        return self.get_state()

    def _manhattan(self):
        return abs(self.agent_pos[0] - self.goal[0]) + abs(self.agent_pos[1] - self.goal[1])

    def get_state(self):
        """3×3 local patch + normalised distance."""
        tr, tc = self.agent_pos
        patch = np.zeros(9, dtype=np.float32)

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
                idx += 1

        norm_dist = self._manhattan() / self._max_dist
        return np.append(patch, norm_dist)

    def step(self, action):
        assert not self.done, "Episode finished — call reset()"
        self.steps += 1

        old_dist = self._manhattan()

        dr, dc = ACTION_DELTAS[action]
        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc

        wall_hit = False
        if (0 <= nr < self.size and 0 <= nc < self.size
                and self.grid[nr, nc] == ROAD):
            self.agent_pos = (nr, nc)
        else:
            wall_hit = True

        new_dist = self._manhattan()

        # ---------- reward shaping ----------
        reward = -0.1                            # small step penalty (not too harsh)

        if self.agent_pos == self.goal:
            reward = 100.0                       # reached the safe zone
            self.done = True
        elif wall_hit:
            reward -= 1.0                        # bumped a wall
        elif action == 4:
            reward -= 0.3                        # discourage standing still
        else:
            # Continuous distance-based shaping
            reward += (old_dist - new_dist) * 1.0  # +1 closer, -1 farther

        if self.steps >= self.max_steps:
            self.done = True

        return self.get_state(), reward, self.done

    def render(self):
        return self.grid