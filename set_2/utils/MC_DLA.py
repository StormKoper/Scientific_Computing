from warnings import warn
import numpy as np
from numba import njit

class MC_DLA():
    """Random Walker DLA implementation"""
    def __init__(self, N: int = 100, p_s: float = 1.0, save_every: int = 0, use_jit: bool = False, seed: int|None = None):
        self.N = N
        self.p_s = p_s
        self.grid = np.zeros((N, N), dtype=bool)
        self.gen = np.random.default_rng(seed)

        # Place the seed in the center at the bottom
        self.grid[-1, N // 2] = True
        self.growth_count = 0

        self.start_x = 0 
        self.start_y = self.gen.integers(0, self.N)

        self.grid_arr = None
        self.save_every = save_every
        if save_every:
            self._frames = [self.grid.copy()]

        if use_jit:
            self._setup_jit()

    def candidate(self, x, y):
        """Check for candidates"""
        neighbours = [(0,1) , (1,0), (0,-1), (-1,0)]

        for dx, dy in neighbours:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.N and 0 <= ny < self.N:                
                if self.grid[nx, ny]:
                    return True
        return False

    def add_walker(self):
        """Add a random walker with a sticking probability"""
        x = 0
        y = self.gen.integers(0, self.N - 1) # Start at a random point at the top

        while True:
            dx, dy = self.gen.choice([(0,1) , (1,0), (0,-1), (-1,0)])

            new_x = x + dx
            new_y = (y + dy) % self.N # Enforce periodicity

            # If walker goes out of bounds remove it and try again
            if not (0 <= new_y < self.N):
                return False
            
            # Update positions
            x = new_x
            y = new_y

            # Check if the point neighbours the cluster
            if self.candidate(x,y):
                if self.gen.random() < self.p_s: # Sticking probability
                    self.grid[x, y] = True
                    return True

    def _setup_jit(self):
        jit_seed = int(self.gen.integers(0, 2**31 - 1))
        np.random.seed(jit_seed) # set global seed for JIT
        def jit_wrapper():
            """High Speed: Calls JIT-compiled walker"""
            return self.jit_walker(self.grid, self.N, self.p_s)
        self.add_walker = jit_wrapper

    @staticmethod
    @njit
    def jit_walker(grid, N, ps):
        """JIT implementation of sticking probability walker procedure"""
        x = 0
        y = np.random.randint(0, N - 1) # Start at a random point at the top
        for _ in range(N * N * 5):
            move = np.random.randint(0, 4)
            dx, dy = [(0,1), (0,-1), (1,0), (-1,0)][move]
            nx, ny = x + dx, (y + dy) % N

            if not (0 <= nx < N and 0 <= ny < N): return False
            if grid[nx, ny]: continue
            x, y = nx, ny

            # Sticking probability
            for dx_s, dy_s in [(0,1), (0,-1), (1,0), (-1,0)]:
                sx, sy = x + dx_s, y + dy_s
                if 0 <= sx < N and 0 <= sy < N:
                    if grid[sx, sy]:
                        if np.random.random() < ps:
                            grid[x, y] = True
                            return True
                        break
        return False
    
    def _grow(self):
        stuck = False
        while not stuck:
            stuck = self.add_walker()
        self.growth_count += 1
    
    def run(self, n_growth: int|None = None, grow_until: float|None = None):
        """Run the Monte Carlo DLA simulation for a specified number of growth steps or until a certain growth threshold is reached. The growth threshold is defined as the fraction of the height of the grid that is reached by the highest object"""
        if n_growth is None and grow_until is None:
            raise ValueError("Either n_growth or grow_until should be provided.")
        # set growth threshold
        if n_growth is not None:
            growth_threshold = n_growth
        else:
            growth_threshold = np.inf # effectively no threshold
        # set height threshold
        if grow_until is not None:
            height_threshold = self.N - int(grow_until * self.N)
        else:
            height_threshold = 1 # effectively no threshold
        # run until either growth threshold or height threshold is reached
        while not np.any(self.grid[height_threshold, :]) and self.growth_count < growth_threshold:
            self._grow()
            if self.save_every and self.growth_count % self.save_every == 0:
                self._frames.append(self.grid.copy())
        # save an array of states if save_every is set
        if self.save_every:
            self.x_arr = np.stack(self._frames, axis=-1)