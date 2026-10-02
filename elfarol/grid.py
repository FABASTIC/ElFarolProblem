import numpy as np


class SpatialGrid:

    def __init__(self, size=50):
        self.size = size
        self.grid = np.zeros((size, size), dtype=np.int32)
        self.bar_min = (size - 10) // 2
        self.bar_max = self.bar_min + 10

    def place(self, agent_id, x, y):
        if not (0 <= x < self.size and 0 <= y < self.size):
            raise ValueError(f"Position ({x}, {y}) out of bounds")
        if self.grid[y, x] != 0:
            raise ValueError(
                f"Position ({x}, {y}) already occupied by agent {self.grid[y, x]}"
            )
        self.grid[y, x] = agent_id

    def remove(self, agent_id, x, y):
        if self.grid[y, x] != agent_id:
            raise ValueError(f"Agent {agent_id} not at position ({x}, {y})")
        self.grid[y, x] = 0

    def move(self, agent_id, old_x, old_y, new_x, new_y):
        if not (0 <= new_x < self.size and 0 <= new_y < self.size):
            raise ValueError(f"Target position ({new_x}, {new_y}) out of bounds")
        if self.grid[old_y, old_x] != agent_id:
            raise ValueError(f"Agent {agent_id} not at ({old_x}, {old_y})")
        if (old_x, old_y) != (new_x, new_y) and self.grid[new_y, new_x] != 0:
            raise ValueError(f"Target ({new_x}, {new_y}) occupied")
        self.grid[old_y, old_x] = 0
        self.grid[new_y, new_x] = agent_id

    def is_in_bar(self, x, y):
        return self.bar_min <= x < self.bar_max and self.bar_min <= y < self.bar_max

    def nearby_agents(self, x, y, radius=5):
        x_min = max(0, x - radius)
        x_max = min(self.size - 1, x + radius)
        y_min = max(0, y - radius)
        y_max = min(self.size - 1, y + radius)
        result = []
        for dy in range(y_min, y_max + 1):
            for dx in range(x_min, x_max + 1):
                if (dx, dy) != (x, y) and self.grid[dy, dx] != 0:
                    result.append(int(self.grid[dy, dx]))
        return result

    def occupancy(self):
        bar_slice = self.grid[self.bar_min : self.bar_max, self.bar_min : self.bar_max]
        return int(np.count_nonzero(bar_slice))

    def snapshot(self):
        return self.grid.copy()
