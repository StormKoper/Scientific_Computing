import numpy as np
from warnings import warn

class GeneralTIDE(): # TIDE: time-independent diffusion equation
    "Base class for iterative methods"
    def __init__(self, x0: np.ndarray, save_every: int = 1):
        self.x = x0
        self.constants = dict()

        self.save_every = save_every
        self.x_arr = x0.astype(np.float16).copy()[..., None]
        
        self.iter_count = 0
        self.error_history = [] 

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

    def run(self, n_iters: int|None, epsilon: float|None = None):
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
        # Use the omega value defined in __init__
        w = self.omega if self.omega is not None else 1.0
        
        neighbor_sum = 0.25 * (np.hstack([self.x[1:-1, 1:], self.x[1:-1, 1][:, None]]) \
                       + np.hstack([self.x[1:-1, -2][:, None], self.x[1:-1, :-1]]) \
                       + self.x[2:, :] + self.x[:-2, :])
        
        x_next = w * neighbor_sum + (1 - w) * self.x[1:-1, :]
        error = np.max(np.abs(x_next - self.x[1:-1, :]))  
        inner_mask = mask[1:-1, :]
        # avoid updating the rightmost column to prevent double update
        inner_mask[:, -1] = False
        self.x[1:-1, :][inner_mask] = x_next[inner_mask]
        # enforce periodicity condition for the rightmost column
        self.x[1:-1, -1] = self.x[1:-1, 0]
        return error
    
    def _setup_jit(self):
        from .optimized import sor_jit
        
        def jit_wrapper() -> float:
            error = sor_jit(self.x, self.omega)
            return error
            
        self._update_func = jit_wrapper