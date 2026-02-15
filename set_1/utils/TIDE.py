import numpy as np


class general_TIDE():
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
       
    def max_error(self, x_old):
        """Stopping criterion for iterative functions"""
        return np.max(np.abs(self.x - x_old))
    
    def _step(self):
        """Perform a step and save error to error_history"""
        x_old = self.x.copy()
        self._update_func()
        self.iter_count += 1
        
        # error history
        error = np.max(np.abs(self.x - x_old))
        self.error_history.append(error)

        # saving logic
        if self.save_every and (self.iter_count % self.save_every == 0):
             new = self.x.copy()[..., None]
             self.x_arr = np.concatenate((self.x_arr, new), axis=-1)
        
        return error

    def run(self, n_iters: int, epsilon: float|None = None): 
        for _ in range(n_iters):
            self._step()
    
class Jacobi(general_TIDE):
    """Jacobi Iteration Function"""
    def __init__(self, x0: np.ndarray, save_every: int = 1):
        super().__init__(x0, save_every)
    
    def _update_func(self):
        x_next = 1/4 * (np.hstack([self.x[1:-1, 1:], self.x[1:-1, 1][:, None]]) \
               + np.hstack([self.x[1:-1, -2][:, None], self.x[1:-1, :-1]]) \
                + self.x[2:, :] + self.x[:-2, :] )
        
        self.x[1:-1, :] = x_next

class Gauss_S(general_TIDE):
    """Gauss Seidel Function"""
    def __init__(self, x0: np.ndarray, save_every: int = 1):
        super().__init__(x0, save_every)
        """Update Gauss Seidel using black and red checkerboard"""
        self.red_mask = (np.indices(self.x.shape).sum(axis=0) % 2) == 0
        self.black_mask = ~self.red_mask

    def _update_mask(self, mask: np.ndarray) -> None:
        x_next = 1/4 * (np.hstack([self.x[1:-1, 1:], self.x[1:-1, 1][:, None]]) \
               + np.hstack([self.x[1:-1, -2][:, None], self.x[1:-1, :-1]]) \
                + self.x[2:, :] + self.x[:-2, :] )
        
        inner_mask = mask[1:-1, :]
        self.x[1:-1, :][inner_mask] = x_next[inner_mask]
    
    def _update_func(self):
        self._update_mask(self.red_mask)
        self._update_mask(self.black_mask)

class SOR(Gauss_S):
    """SOR function"""
    def __init__(self, x0: np.ndarray, save_every: int = 1, omega = 1.8):
        super().__init__(x0, save_every)
        self.omega = omega
        
    def _update_mask(self, mask: np.ndarray) -> None:
        # Use the omega value defined in __init__
        w = self.omega if self.omega is not None else 1.0
        
        neighbor_sum = 1/4 * (np.hstack([self.x[1:-1, 1:], self.x[1:-1, 1][:, None]]) \
                       + np.hstack([self.x[1:-1, -2][:, None], self.x[1:-1, :-1]]) \
                       + self.x[2:, :] + self.x[:-2, :] )
        
        current_interior = self.x[1:-1, :]
        x_next = w * neighbor_sum + (1 - w) * current_interior    
        inner_mask = mask[1:-1, :]
        
        self.x[1:-1, :][inner_mask] = x_next[inner_mask]
