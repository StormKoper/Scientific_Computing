from warnings import warn

import numpy as np
from numba import njit, prange

class DLA():
    """Diffusion limited aggregation class"""
    def __init__(self, x0: np.ndarray, eta: float = 1.0, omega: float = 1.0, save_error: bool = False, use_jit: bool = False):
        self.x = x0
        self.eta = eta
        self.omega = omega

        self._frames = [np.zeros_like(x0, dtype=bool).copy()]
        self.obj_arr = None
        
        self.save_error = save_error
        if save_error: self.error_history = []
        
        self.iter_count = 0

        # initialize an obj_mask with no objects
        self.obj_mask = np.ones_like(x0, dtype=bool)
        # seed the initial object in the middle of the bottom row
        self.obj_mask[-1, x0.shape[1]//2] = False

        if use_jit:
            self._setup_jit()
        else:
            red_mask = (np.indices(self.x.shape).sum(axis=0) % 2) == 0
            black_mask = ~red_mask
            red_mask = red_mask[1:-1, :]
            black_mask = black_mask[1:-1, :]
            red_mask[:, -1] = False # avoid updating the rightmost column to prevent double update
            black_mask[:, -1] = False
            self.red_mask = red_mask
            self.black_mask = black_mask

    def _update_mask(self, mask: np.ndarray) -> float:        
        neighbor_sum = 0.25 * (np.hstack([self.x[1:-1, 1:], self.x[1:-1, 1][:, None]]) \
                       + np.hstack([self.x[1:-1, -2][:, None], self.x[1:-1, :-1]]) \
                       + self.x[2:, :] + self.x[:-2, :])
        
        x_next = self.omega * neighbor_sum + (1 - self.omega) * self.x[1:-1, :]
        error = np.max(np.abs(x_next - self.x[1:-1, :]))
        self.x[1:-1, :][mask & self.obj_mask[1:-1, :]] = x_next[mask & self.obj_mask[1:-1, :]]

        # enforce periodicity condition for the rightmost column
        self.x[1:-1, -1] = self.x[1:-1, 0]
        return error

    def _update_func(self):
        red_error = self._update_mask(self.red_mask)
        black_error = self._update_mask(self.black_mask)
        return max(red_error, black_error)
    
    def _setup_jit(self):
        def jit_wrapper() -> float:
            error = self._update_jit(self.x, self.omega, self.obj_mask)
            return error
        self._update_func = jit_wrapper
    
    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _update_jit(x: np.ndarray, omega: float, obj_mask: np.ndarray) -> float:
        """Optimized SOR iteration using numba JIT compilation and parallelization with red-black ordering."""
        rows, cols = x.shape
        max_diff = 0.0
        
        # red points
        for i in prange(1, rows - 1):
            # boundary
            if i % 2 == 0:
                neighbor_sum = 0.25 * (x[i-1, 0] + x[i+1, 0] + x[i, 1] + x[i, -2])
                next = (1 - omega) * x[i, 0] + omega * neighbor_sum
                diff = abs(next - x[i, 0])
                x[i, 0] = next
                max_diff = max(max_diff, diff)
            
            # interior
            start = 2 if (i % 2 == 0) else 1
            for j in range(start, cols - 1, 2):
                if obj_mask[i, j]:
                    neighbor_sum = 0.25 * (x[i-1, j] + x[i+1, j] + x[i, j-1] + x[i, j+1])
                    next = (1 - omega) * x[i, j] + omega * neighbor_sum
                    diff = abs(next - x[i, j])
                    x[i, j] = next
                    max_diff = max(max_diff, diff)

            # enforce periodicity condition for the rightmost column
            x[i, -1] = x[i, 0]

        # black points
        for i in prange(1, rows - 1):
            # boundary
            if i % 2 != 0:
                neighbor_sum = 0.25 * (x[i-1, 0] + x[i+1, 0] + x[i, 1] + x[i, -2])
                next = (1 - omega) * x[i, 0] + omega * neighbor_sum
                diff = abs(next - x[i, 0])
                x[i, 0] = next
                max_diff = max(max_diff, diff)
            
            # interior
            start = 1 if (i % 2 == 0) else 2
            for j in range(start, cols - 1, 2):
                if obj_mask[i, j]:
                    neighbor_sum = 0.25 * (x[i-1, j] + x[i+1, j] + x[i, j-1] + x[i, j+1])
                    next = (1 - omega) * x[i, j] + omega * neighbor_sum
                    diff = abs(next - x[i, j])
                    x[i, j] = next
                    max_diff = max(max_diff, diff)

            # enforce periodicity condition for the rightmost column
            x[i, -1] = x[i, 0]

        return max_diff
    
    def _step(self):
        """Perform a step and save error to error_history"""
        error = self._update_func()
        self.iter_count += 1

        # error history
        if self.save_error:
            self.error_history.append(error)
        
        return error
    
    def _grow(self):
        """Choose a candidate site for growth and update the obj_mask"""
        # check cells for adjacency to an object
        padded = np.pad(self.obj_mask, pad_width=1, mode='wrap')
        has_obj_neighbor = (
            ~padded[:-2, 1:-1] |  # up
            ~padded[2:, 1:-1]  |  # fown
            ~padded[1:-1, :-2] |  # left
            ~padded[1:-1, 2:]     # right
        )

        # filter out objects
        candidates = self.obj_mask & has_obj_neighbor
        candidates[0, :] = False # prevent growth on the top boundary
        if not np.any(candidates):
            warn("No growth candidates found. Not growing any new sites.")
            return
        # compute the growth probability for each site
        conc_eta = self.x[candidates]**self.eta
        if np.sum(conc_eta) == 0:
            warn("All candidate sites have zero concentration. Not growing any new sites.")
            return
        probabilities = conc_eta / np.sum(conc_eta)
        # choose a candidate site based on the probabilities
        flat_index = np.random.choice(np.flatnonzero(candidates), p=probabilities)
        candidate_index = np.unravel_index(flat_index, self.obj_mask.shape)
        # update the obj_mask to include the new object
        self.obj_mask[candidate_index] = False
    
    def run(self, n_growth: int|None = None, grow_until: float|None = None, epsilon: float = 10e-5):
        """Run the DLA simulation for a specified number of growth steps or until a certain growth threshold is reached. The growth threshold is defined as the fraction of the height of the grid that is reached by the highest object"""
        if n_growth is None and grow_until is None:
            raise ValueError("Either n_growth or grow_until should be provided.")
        elif n_growth is not None and grow_until is not None:
            warn("Both n_growth and grow_until are provided. n_growth will be used as the stopping criterion.")
        if n_growth is not None:
            for _ in range(n_growth):
                error = float('inf')
                while error > epsilon:
                    error = self._step()
                self._grow()
        elif grow_until is not None:
            height_threshold = grow_until * self.x.shape[0]
            while not np.any(~self.obj_mask[-int(height_threshold):, :]):
                error = float('inf')
                while error > epsilon:
                    error = self._step()
                self._grow()
        
        self.obj_arr = np.stack(self._frames, axis=-1)