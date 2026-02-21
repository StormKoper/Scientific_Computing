from warnings import warn

import numpy as np


class GeneralWave():
    """Base class for the wave equation solvers"""
    def __init__(self, x0: np.ndarray, dt: float, dx: float, save_every: int = 0):
        self.x = x0
        self.dx = dx
        self.dt = dt
        self.constants = dict()
        self.save_every = save_every
        self._frames = [x0.astype(np.float16).copy()] 
        self.x_arr = None

    def _update_func(self):
        """Should contain the logic to update the x by one step"""
        raise(NotImplementedError)

    def run(self, n_iters: int):
        """Advances the grid by one step"""
        for n in range(n_iters):
            self._update_func()
            if self.save_every and (n % self.save_every == 0):
                new = self.x.astype(np.float16).copy()[..., None]
                self._frames.append(self.x.astype(np.float16).copy())
        self.x_arr = np.stack(self._frames, axis=-1)

class Wave1D(GeneralWave):
    """1D wave equation solver using finite difference method"""
    def __init__(self, x0: np.ndarray, dt: float, dx: float, c: float = 1.0, save_every: int = 0, use_jit: bool = False):
        super().__init__(x0, dt, dx, save_every)
        self.constants["C^2"] = (c*(dt/dx))**2
        self.x_prev = self.x.copy()
        self._first_step()
        if use_jit:
            self._setup_jit()

    def _first_step(self):
        self.x[1:-1] = 0.5 * self.constants['C^2'] \
            * (self.x[:-2] - 2*self.x[1:-1] + self.x[2:]) \
                + self.x[1:-1]
    
    def _update_func(self):
        x_next = self.constants['C^2'] \
            * (self.x[:-2] - 2*self.x[1:-1] + self.x[2:]) \
                - self.x_prev[1:-1] + 2*self.x[1:-1]
        self.x_prev = self.x.copy()
        self.x[1:-1] = x_next

    def _setup_jit(self):
        from .optimized import wave_1d_jit
        self._x_next = self.x.copy()
        def jit_wrapper():
            wave_1d_jit(self.x, self.x_prev, self._x_next, self.constants['C^2'])
            
            # swap references for next iteration
            old_prev = self.x_prev
            self.x_prev = self.x
            self.x = self._x_next
            self._x_next = old_prev

        self._update_func = jit_wrapper

class Leapfrog(GeneralWave):
    """1D wave equation solver using leapfrog method"""
    def __init__(self, x0: np.ndarray, dt: float, dx: float, c: float = 1.0, save_every: int = 0, use_jit: bool = False):
        super().__init__(x0, dt, dx, save_every)
        self.constants["cdx"] = (c/dx)**2
        self.a = self.constants['cdx'] * (self.x[:-2] - 2*self.x[1:-1] + self.x[2:])
        self.v = np.zeros_like(self.x)[1:-1]
        if use_jit:
            self._setup_jit()

    def _update_func(self):
        self.v = self.v + 0.5 * self.dt * self.a
        self.x[1:-1] = self.x[1:-1] + self.dt * self.v
        self.a = self.constants['cdx'] * (self.x[:-2] - 2*self.x[1:-1] + self.x[2:])
        self.v = self.v + 0.5 * self.dt * self.a
    
    def _setup_jit(self):
        from .optimized import leapfrog_jit
        def jit_wrapper():
            leapfrog_jit(self.x, self.v, self.a, self.dt, self.constants['cdx'])
            
        self._update_func = jit_wrapper

class Wave2D(GeneralWave):
    """2D wave equation solver using finite difference method"""
    def __init__(self, x0: np.ndarray, dt: float, dx: float, D: float = 1.0, save_every: int = 0, use_jit: bool = False):
        super().__init__(x0, dt, dx, save_every)
        self.constants["d"] = (dt*D) / (dx**2)
        if self.constants["d"] >= 0.25:
            raise ValueError("The scheme is unstable for d >= 0.25. Choose smaller dt. Currently d = " + str(self.constants["d"]))
        elif self.constants["d"] >= 0.2:
            warn("The scheme is close to the stability limit. Consider smaller dt for better results. Currently d = " + str(self.constants["d"]))
        if use_jit:
            self._setup_jit()
    
    def _update_func(self):
        x_next = self.x[1:-1, :] + self.constants['d'] \
            * (np.hstack([self.x[1:-1, 1:], self.x[1:-1, 1][:, None]]) \
               + np.hstack([self.x[1:-1, -2][:, None], self.x[1:-1, :-1]]) \
                + self.x[2:, :] + self.x[:-2, :] - 4*self.x[1:-1, :])
        self.x[1:-1, :] = x_next

    def _setup_jit(self):
        from .optimized import wave_2d_jit
        self._x_next = self.x.copy()
        def jit_wrapper():
            wave_2d_jit(self.x, self._x_next, self.constants['d'])
            
            # swap references for next iteration
            old_x = self.x
            self.x = self._x_next
            self._x_next = old_x

        self._update_func = jit_wrapper