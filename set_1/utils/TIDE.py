from warnings import warn

import numpy as np
from scipy.signal import convolve2d


class GeneralTIDE(): # TIDE: time-independent diffusion equation
    "Base class for iterative methods"
    def __init__(self, x0: np.ndarray, save_every: int = 0, save_error: bool = False):
        self.x = x0
        self.constants = dict()

        self.save_every = save_every
        if save_every: self.x_arr = x0.astype(np.float16).copy()[..., None]
        
        self.save_error = save_error
        if save_error: self.error_history = []
        
        self.iter_count = 0

        # initialize an obj_mask with no objects
        self.obj_mask = np.full_like(x0, 0.25, dtype=float)

    def _update_func(self) -> float:
        """Should contain the logic to update the x by one step"""
        raise(NotImplementedError)
    
    def _step(self):
        """Perform a step and save error to error_history"""
        error = self._update_func()
        self.iter_count += 1

        # saving logic
        if self.save_every and (self.iter_count % self.save_every == 0):
             new = self.x.copy()[..., None]
             self.x_arr = np.concatenate((self.x_arr, new), axis=-1)

        # error history
        if self.save_error:
            self.error_history.append(error)
        
        return error

    def run(self, n_iters: int|None = None, epsilon: float|None = None):
        if n_iters is None and epsilon is None:
            raise ValueError("Either n_iters or epsilon should be provided.")
        
        if n_iters is not None and epsilon is not None:
            warn("Both n_iters and epsilon are provided. n_iters will be used as the stopping criterion.")
        
        if n_iters is not None:
            for _ in range(n_iters):
                self._step()

        else:
            error = float('inf')
            while error > epsilon:
                error = self._step()

    def objects(self, mask_indices: np.ndarray, insulation: bool = False):
        """Initialize objects in the grid.

        Args:
            - mask_indices (np.ndarray): 2D array with same shape as self.x that has 1s for cells part
                of the object and 0s everywhere else.
            - insulation (bool): whether the object behaves like insulation material - False is sink
        
        """
        # enforce periodicity condition for the rightmost column
        mask_indices[:, -1] = mask_indices[:, 0]
        # clear object in self.x
        self.x[mask_indices] = 0
        # also clear object in self._x_next buffer for Jacobi JIT implementation
        if hasattr(self, '_x_next'):
            self._x_next[mask_indices] = 0

        if insulation:
            kernel = np.array([
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0]
            ])
            # create obj_mask with 1/num_neighbors for all non-object cells, and 0s for object-cells
            neighbor_count = convolve2d(~mask_indices[:, :-1], kernel, mode='same', boundary='wrap')
            # add one column for the rightmost column that is periodic with the leftmost column
            neighbor_count = np.hstack([neighbor_count, neighbor_count[:, 0][:, None]])
            
            # numpy way to do 1/arr only when non-zero value
            obj_mask = np.divide(1.0, neighbor_count, out=np.zeros_like(neighbor_count, dtype=float), where=neighbor_count!=0)

            # explicitly set object to 0 again to enforce good perimiters
            obj_mask[mask_indices] = 0
        else:
            # create obj_mask with 0.25 for all non-object cells, and 0s for object-cells
            obj_mask = ~mask_indices / 4

        self.obj_mask = obj_mask
    
class Jacobi(GeneralTIDE):
    """Jacobi Iteration Function"""
    def __init__(self, x0: np.ndarray, save_every: int = 1, use_jit: bool = False):
        super().__init__(x0, save_every)
        if use_jit:
            self._setup_jit()
    
    def _update_func(self) -> float:
        x_next = self.obj_mask[1:-1, :] * (np.hstack([self.x[1:-1, 1:], self.x[1:-1, 1][:, None]]) \
               + np.hstack([self.x[1:-1, -2][:, None], self.x[1:-1, :-1]]) \
                + self.x[2:, :] + self.x[:-2, :])
        error = np.max(np.abs(x_next - self.x[1:-1, :]))
        self.x[1:-1, :] = x_next

        return error
        
    def _setup_jit(self):
        from .optimized import jacobi_jit
        self._x_next = self.x.copy()
        def jit_wrapper() -> float:
            error = jacobi_jit(self.x, self._x_next, self.obj_mask)
            
            # swap references for next iteration
            old_x = self.x
            self.x = self._x_next
            self._x_next = old_x

            return error
        
        self._update_func = jit_wrapper

class GaussSeidel(GeneralTIDE):
    """Gauss Seidel Function"""
    def __init__(self, x0: np.ndarray, save_every: int = 1, use_jit: bool = False):
        super().__init__(x0, save_every)
        """Update Gauss Seidel using black and red checkerboard"""
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
        x_next = self.obj_mask[1:-1, :] * (np.hstack([self.x[1:-1, 1:], self.x[1:-1, 1][:, None]]) \
               + np.hstack([self.x[1:-1, -2][:, None], self.x[1:-1, :-1]]) \
                + self.x[2:, :] + self.x[:-2, :])
        
        error = np.max(np.abs(x_next - self.x[1:-1, :]))
        self.x[1:-1, :][mask] = x_next[mask]

        # enforce periodicity condition for the rightmost column
        self.x[1:-1, -1] = self.x[1:-1, 0]
        return error
    
    def _update_func(self):
        red_error = self._update_mask(self.red_mask)
        black_error = self._update_mask(self.black_mask)
        return max(red_error, black_error)

    def _setup_jit(self):
        from .optimized import gauss_seidel_jit
        
        def jit_wrapper() -> float:
            error = gauss_seidel_jit(self.x, self.obj_mask)
            return error
            
        self._update_func = jit_wrapper

class SOR(GaussSeidel):
    """SOR function"""
    def __init__(self, x0: np.ndarray, save_every: int = 1, omega = 1.8, use_jit: bool = False):
        super().__init__(x0, save_every, use_jit)
        self.omega = omega
        
    def _update_mask(self, mask: np.ndarray) -> float:
        # Use the omega value defined in __init__
        w = self.omega if self.omega is not None else 1.0
        
        neighbor_sum = self.obj_mask[1:-1, :] * (np.hstack([self.x[1:-1, 1:], self.x[1:-1, 1][:, None]]) \
                       + np.hstack([self.x[1:-1, -2][:, None], self.x[1:-1, :-1]]) \
                       + self.x[2:, :] + self.x[:-2, :])
        
        x_next = w * neighbor_sum + (1 - w) * self.x[1:-1, :]
        error = np.max(np.abs(x_next - self.x[1:-1, :]))
        self.x[1:-1, :][mask] = x_next[mask]

        self.x[-1, :] = 1.0 
        self.x[0, :] = 0.0  

         # enforce periodicity condition for the rightmost column
        self.x[1:-1, -1] = self.x[1:-1, 0]
        return error
    
    def _setup_jit(self):
        from .optimized import sor_jit
        
        def jit_wrapper() -> float:
            error = sor_jit(self.x, self.omega, self.obj_mask)
            return error
            
        self._update_func = jit_wrapper

