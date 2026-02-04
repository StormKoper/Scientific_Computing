import numpy as np

class GeneralWave():
    """Base class for the wave equation solvers"""
    def __init__(self, x0: np.ndarray, dt: float, dx: float, save_every: int = 1):
        self.x = x0
        self.dt = dt
        self.constants = dict()
        self.save_every = save_every
        self.x_arr = x0.copy()

    def _first_step(self):
        """This completse the first step based on single sided scheme"""
        raise(NotImplementedError)

    def _update_func(self):
        """Should contain the logic to update the x by one step"""
        raise(NotImplementedError)

    def run(self, n_iters: int):
        """Advances the grid by one step"""
        for n in range(n_iters):
            self._update_func()
            if self.save_every and (n % self.save_every == 0):
                self.x_arr = np.vstack((self.x_arr, self.x.copy()))


class Wave1D(GeneralWave):
    """1D wave equation solver using finite difference method"""
    def __init__(self, x0, dt, dx, c: float = 1.0):
        super().__init__(x0, dt, dx)
        self.constants["C^2"] = (c*(dt/dx))**2
        self._first_step()

    def _first_step(self):
        self.x_prev = self.x
        self.x[1:-2] = self.constants['C^2'] \
            * (self.x[0:-3] - 2*self.x[1:-2] + self.x[2:-1]) \
                + self.x[1:-2]
            
    def _update_func(self):
        self.x[1:-2] = self.constants['C^2'] \
            * (self.x[0:-3] - 2*self.x[1:-2] + self.x[2:-1]) \
                - self.x_prev[1:-2] + 2*self.x[1:-2]