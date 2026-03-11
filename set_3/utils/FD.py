import numpy as np
from numba import njit, prange

class FD:
    def __init__(self, dx: float = .02, dy: float = .02, dt: float = .01, rho: float = 1.0, nu: float = .1):
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.rho = rho
        self.nu = nu
        self.grid = np.zeros(shape=(int(1 / dx), int(1 / dy)),
                             dtype=[("u", "float64"), ("v", "float64")])
        self.next_grid = self.grid.copy()

    def run(self, time: float):
        p = np.zeros_like(self.grid["u"])
        p_prev = np.zeros_like(p)
        source = np.zeros_like(p)
        n_iters = int(time / self.dt)
        inv_dt = 1 / self.dt
        rho_dx_dy = (self.rho * self.dx**2 * self.dy**2) / (2 * (self.dx**2 + self.dy**2))
        for _ in range(n_iters):
            self._p_source(source, rho_dx_dy, inv_dt, self.grid["u"], self.grid["v"], self.dx, self.dy)
            self._pressure(p, p_prev, source)
            self._velocity()

    @staticmethod
    @njit(parallel=True, fastmath=False)
    def _p_source(source: np.ndarray, rho_dx_dy: float, inv_dt: float,
                  u: np.ndarray, v: np.ndarray, dx: float, dy: float):
        for i in prange(1, source.shape[0]-1):
            for j in range(1, source.shape[1]-1):
                source[i, j] = rho_dx_dy * (inv_dt * ((u[i+1, j] - u[i-1, j]) / (2*dx) + (v[i, j+1] - v[i, j-1]) / (2*dy)) \
                 - ((u[i+1, j] - u[i-1, j]) / (2*dx))**2 - ((v[i, j+1] - v[i, j-1]) / (2*dy))**2 \
                  - 2 * ((u[i, j+1] - u[i, j-1]) / (2*dx)) * ((v[i+1, j] - v[i-1, j]) / (2*dy)))

    @staticmethod
    @njit(parallel=True, fastmath=False)
    def _pressure(p: np.ndarray, p_prev: np.ndarray, source: np.ndarray, rho_dx_dy: float,
                  dx: float, dy: float, threshold: float = 1e-4, max_iters: int = 1000):
        iter_count = 0
        error = np.inf
        while error > threshold and iter_count < max_iters:
            error = 0.0
            for i in prange(1, p.shape[0]-1):
                for j in range(1, p.shape[1]-1):
                    p[i, j] = ((p_prev[i, j+1] + p_prev[i, j-1]) * dx**2 \
                               + (p_prev[i+1, j] + p_prev[i-1, j]) * dy**2) \
                                / (2 * (dx**2 + dy**2)) - rho_dx_dy * source[i, j]
                    error = max(error, abs(p[i, j] - p_prev[i, j]))
            p, p_prev = p_prev, p
            iter_count += 1

    @staticmethod
    @njit(parallel=True, fastmath=False)
    def _velocity():
        pass