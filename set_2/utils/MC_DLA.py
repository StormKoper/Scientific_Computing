import numpy as np
import random
from numba import njit

class MC_DLA():
    """Random Walker DLA implementation"""
    def __init__(self, size, seed: int|None = None):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.gen = np.random.default_rng(seed)

        # Place the seed in the center at the bottom
        self.bottom = self.size - 1
        self.center = self.size // 2

        self.grid[self.bottom, self.center] = 1
        self.particles_count = 1

    def candidate(self, x, y):
        """Check for candidates"""
        neighbours = [(0,1) , (1,0), (0,-1), (-1,0)]

        for dx, dy in neighbours:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:                
                if self.grid[nx, ny] == 1:
                    return True
        return False

    def add_walker(self):
        """Add a random walker"""
        x = 0
        y = random.randint(0, self.size - 1) # Start at a random point at the top

        walking = True
        while walking:
            dx, dy = random.choice([(0,1) , (1,0), (0,-1), (-1,0)])

            new_x = x + dx
            new_y = y + dy

            # If walker goes out of bounds remove it and try again
            if not (0 <= new_x < self.size and 0 <= new_y < self.size):
                return False
            
            # Update positions
            x = new_x
            y = new_y

            # Check if the point neighbours the cluster
            if self.candidate(x,y):
                self.grid[x, y] = 1
                self.particles_count += 1
                return True
    
    def add_walker_ps(self, ps):
        """Add a random walker including a sticking probability"""
        x = 0
        y = random.randint(0, self.size - 1)

        walking = True
        while walking:
            dx, dy = random.choice([(0,1) , (1,0), (0,-1), (-1,0)])

            new_x = x + dx
            new_y = y + dy

            # If walker goes out of bounds remove it and try again
            if not (0 <= new_x < self.size and 0 <= new_y < self.size):
                return False
            
            # Update positions
            x = new_x
            y = new_y

            # Check if the point neighbours the cluster
            if self.candidate(x,y):
                if random.random() < ps:
                    self.grid[x, y] = 1
                    self.particles_count += 1
                    return True
                
    def add_walker_jit(self):
        """High Speed: Calls JIT-compiled standard walker"""
        random.seed(self.gen.bit_generator.state['state']['state']) 
        start_y = random.randint(0, self.size - 1)
        rx, ry, stuck = jit_walker(self.grid, self.size, 0, start_y)
        if stuck:
            self.grid[rx, ry] = 1
            self.particles_count += 1
        return stuck

    def add_walker_ps_jit(self, ps):
        """High Speed: Calls JIT-compiled sticking probability walker"""
        random.seed(self.gen.bit_generator.state['state']['state']) 
        start_y = random.randint(0, self.size - 1)
        rx, ry, stuck = jit_walker_ps(self.grid, self.size, 0, start_y, ps)
        if stuck:
            self.grid[rx, ry] = 1
            self.particles_count += 1
        return stuck

@njit
def jit_walker(grid, size, start_x, start_y):
    """JIT implementation of standard walker procedure"""
    x, y = start_x, start_y
    for _ in range(size * size * 5):
        move = np.random.randint(0, 4)
        dx, dy = [(0,1), (0,-1), (1,0), (-1,0)][move]
        nx, ny = x + dx, y + dy

        if not (0 <= nx < size and 0 <= ny < size): return -1, -1, False
        if grid[nx, ny] == 1: continue
        x, y = nx, ny

        # Sticking logic (check neighbors)
        for dx_s, dy_s in [(0,1), (0,-1), (1,0), (-1,0)]:
            sx, sy = x + dx_s, y + dy_s
            if 0 <= sx < size and 0 <= sy < size:
                if grid[sx, sy] == 1:
                    return x, y, True
    return -1, -1, False

@njit
def jit_walker_ps(grid, size, start_x, start_y, ps):
    """JIT implementation of sticking probability walker procedure"""
    x, y = start_x, start_y
    for _ in range(size * size * 5):
        move = np.random.randint(0, 4)
        dx, dy = [(0,1), (0,-1), (1,0), (-1,0)][move]
        nx, ny = x + dx, y + dy

        if not (0 <= nx < size and 0 <= ny < size): return -1, -1, False
        if grid[nx, ny] == 1: continue
        x, y = nx, ny

        # Sticking probability
        for dx_s, dy_s in [(0,1), (0,-1), (1,0), (-1,0)]:
            sx, sy = x + dx_s, y + dy_s
            if 0 <= sx < size and 0 <= sy < size:
                if grid[sx, sy] == 1:
                    if np.random.random() < ps:
                        return x, y, True
                    break
    return -1, -1, False