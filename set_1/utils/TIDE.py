import numpy as np
from warnings import warn
from scipy.signal import convolve2d

class GeneralTIDE(): # TIDE: time-independent diffusion equation
    "Base class for iterative methods"
    def __init__(self, x0: np.ndarray, save_every: int = 1):
        self.x = x0
        self.constants = dict()

        self.save_every = save_every
        self.x_arr = x0.astype(np.float16).copy()[..., None]
        
        self.iter_count = 0
        self.error_history = [] 

        self.obj_map = np.zeros_like(x0, dtype=int)
        self.obj_val = 0.0

    def _update_func(self):
        """Should contain the logic to update the x by one step"""
        raise(NotImplementedError)
    
    def _step(self):
        """Perform a step and save error to error_history"""
        error = self._update_func()
        self.iter_count += 1
        
        # error history
        self.error_history.append(error)

        # saving logic
        if self.save_every and (self.iter_count % self.save_every == 0):
             new = self.x.copy()[..., None]
             self.x_arr = np.concatenate((self.x_arr, new), axis=-1)
        
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
    
    def _update_func(self):
        x_next = 0.25 * (np.hstack([self.x[1:-1, 1:], self.x[1:-1, 1][:, None]]) \
               + np.hstack([self.x[1:-1, -2][:, None], self.x[1:-1, :-1]]) \
                + self.x[2:, :] + self.x[:-2, :])
        error = np.max(np.abs(x_next - self.x[1:-1, :]))
        self.x[1:-1, :] = x_next

        # For objects
        self.x[self.obj_map == 1] = self.obj_val

        return error
        
    def _setup_jit(self):
        from .optimized import jacobi_jit
        self._x_next = self.x.copy()
        def jit_wrapper() -> float:
            error = jacobi_jit(self.x, self._x_next)
            
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
            self.red_mask = (np.indices(self.x.shape).sum(axis=0) % 2) == 0
            self.black_mask = ~self.red_mask

    def _update_mask(self, mask: np.ndarray) -> None:
        x_next = 0.25 * (np.hstack([self.x[1:-1, 1:], self.x[1:-1, 1][:, None]]) \
               + np.hstack([self.x[1:-1, -2][:, None], self.x[1:-1, :-1]]) \
                + self.x[2:, :] + self.x[:-2, :])
        
        error = np.max(np.abs(x_next - self.x[1:-1, :]))
        inner_mask = mask[1:-1, :]
        # avoid updating the rightmost column to prevent double update
        inner_mask[:, -1] = False
        self.x[1:-1, :][inner_mask] = x_next[inner_mask]

        # For objects
        self.x[self.obj_map == 1] = self.obj_val

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
            error = gauss_seidel_jit(self.x)
            return error
            
        self._update_func = jit_wrapper

class SOR(GaussSeidel):
    """SOR function"""
    def __init__(self, x0: np.ndarray, save_every: int = 1, omega = 1.8, use_jit: bool = False):
        super().__init__(x0, save_every, use_jit)
        self.omega = omega
        
    def _update_mask(self, mask: np.ndarray) -> None:
        w = self.omega
        
        neighbor_sum = 0.25 * (self.x[1:-1, 2:] + self.x[1:-1, :-2] + 
                               self.x[2:, 1:-1] + self.x[:-2, 1:-1])
        
        x_inner = self.x[1:-1, 1:-1]
        x_next = w * neighbor_sum + (1 - w) * x_inner
        
        valid_pixels = ~mask[1:-1, 1:-1]
        x_inner[valid_pixels] = x_next[valid_pixels]

        if self.obj_map is not None:
            self.x[self.obj_map == 1] = self.obj_val

        self.x[-1, :] = 1.0 
        self.x[0, :] = 0.0  

        error = np.max(np.abs(x_next[valid_pixels] - x_inner[valid_pixels]))
        return error
    
    def _setup_jit(self):
        from .optimized import sor_jit
        
        def jit_wrapper() -> float:
            error = sor_jit(self.x, self.omega)
            return error
            
        self._update_func = jit_wrapper

