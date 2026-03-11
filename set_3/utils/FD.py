import numpy as np
from numba import njit, prange
from warnings import warn

class FD:
    def __init__(self, dx: float = .05, dy: float = .05, dt: float = .001, rho: float = 1.0, nu: float = .1):
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.rho = rho
        self.nu = nu
        self.u = np.zeros(shape=(int(1 / dx), int(1 / dy)))
        self.v = np.zeros_like(self.u)
        self.p = np.zeros_like(self.u)
        self.u[:, 0] = 1.0 # inflow from the left
        self.u[:, 1] = 0.1 * (1 + np.sin(2*np.pi*np.linspace(0, 1, self.u.shape[0]))) # small perturbation to trigger turbulence

    def run(self, time: float, p_threshold: float = 1e-4, p_max_iters: int = 1000):
        source = np.zeros_like(self.p)
        u_next = self.u.copy()
        v_next = self.v.copy()
        n_iters = int(time / self.dt)
        inv_dt = 1 / self.dt
        rho_dx_dy = (self.rho * self.dx**2 * self.dy**2) / (2 * (self.dx**2 + self.dy**2))
        for _ in range(n_iters):
            self._p_source(source, rho_dx_dy, inv_dt, self.u, self.v, self.dx, self.dy)
            iter_count, error = self._pressure(self.p, source, rho_dx_dy, self.dx, self.dy, p_threshold, p_max_iters)
            if iter_count >= p_max_iters:
                warn(f"Maximum iterations reached for pressure solver: {iter_count}")
            self._velocity(self.u, self.v, u_next, v_next, self.p, self.rho, self.nu, self.dt, self.dx, self.dy)
            self.u, u_next = u_next, self.u
            self.v, v_next = v_next, self.v
            CFL = (max(np.max(self.u), np.max(self.v)) * self.dt) / min(self.dx, self.dy)
            if CFL > 1:
                warn(f"CFL condition violated: {CFL:.2f} > 1. Consider reducing dt or increasing dx/dy for stability.")

    def plot(self):
        import matplotlib.pyplot as plt
        x = np.linspace(0, 1, self.u.shape[1])
        y = np.linspace(0, 1, self.u.shape[0])
        X, Y = np.meshgrid(x, y)
        plt.contourf(X, Y, self.p, levels=50, cmap='viridis')
        plt.colorbar(label='Pressure')
        plt.streamplot(X, Y, self.u, self.v, color='white')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Pressure and Velocity Field')
        plt.show()

    @staticmethod
    @njit(parallel=True, fastmath=False)
    def _p_source(source: np.ndarray, rho_dx_dy: float, inv_dt: float,
                  u: np.ndarray, v: np.ndarray, dx: float, dy: float):
        """Computes the source term for the pressure Poisson equation based on the velocity field."""
        for i in prange(1, source.shape[0]-1):
            for j in range(1, source.shape[1]-1):
                source[i, j] = rho_dx_dy * (inv_dt * ((u[i+1, j] - u[i-1, j]) / (2*dx) + (v[i, j+1] - v[i, j-1]) / (2*dy)) \
                 - ((u[i+1, j] - u[i-1, j]) / (2*dx))**2 - ((v[i, j+1] - v[i, j-1]) / (2*dy))**2 \
                  - 2 * ((u[i, j+1] - u[i, j-1]) / (2*dx)) * ((v[i+1, j] - v[i-1, j]) / (2*dy)))

    @staticmethod
    @njit(parallel=True, fastmath=False)
    def _pressure(p: np.ndarray, source: np.ndarray, rho_dx_dy: float,
                  dx: float, dy: float, threshold: float = 1e-4, max_iters: int = 1000):
        """Solves the Poisson equation for pressure using the red-black Gauss-Seidel method."""
        # NOTE: updating the error in parallel causes a race condition
        iter_count = 0
        error = np.inf
        while error > threshold and iter_count < max_iters:
            error = 0.0
            # red points
            for i in prange(1, p.shape[0]-1):
                start = 2 - (i % 2)
                for j in range(start, p.shape[1]-1, 2):
                    next = ((p[i, j+1] + p[i, j-1]) * dx**2 \
                               + (p[i+1, j] + p[i-1, j]) * dy**2) \
                                / (2 * (dx**2 + dy**2)) - rho_dx_dy * source[i, j]
                    error = max(error, abs(next - p[i, j]))
                    p[i, j] = next
            # black points
            for i in prange(1, p.shape[0]-1):
                start = 1 + (i % 2)
                for j in range(start, p.shape[1]-1, 2):
                    next = ((p[i, j+1] + p[i, j-1]) * dx**2 \
                               + (p[i+1, j] + p[i-1, j]) * dy**2) \
                                / (2 * (dx**2 + dy**2)) - rho_dx_dy * source[i, j]
                    error = max(error, abs(next - p[i, j]))
                    p[i, j] = next
            iter_count += 1
        return iter_count, error

    @staticmethod
    @njit(parallel=True, fastmath=False)
    def _velocity(u: np.ndarray, v: np.ndarray, u_next: np.ndarray, v_next: np.ndarray, p: np.ndarray,
                  rho: float, nu: float, dt: float, dx: float, dy: float):
        """Updates the velocity field based on the Navier-Stokes equations using combined upwind scheme."""
        for i in prange(1, u.shape[0]-1):
            for j in range(1, u.shape[1]-1):
                # combined upwind terms for advection
                u_adv_dx = 0.5 * ((u[i, j] + abs(u[i, j])) * (u[i, j] - u[i-1, j]) \
                    + (u[i, j] - abs(u[i, j])) * (u[i+1, j] - u[i, j])) * dt/dx
                u_adv_dy = 0.5 * ((v[i, j] + abs(v[i, j])) * (u[i, j] - u[i, j-1]) \
                    + (v[i, j] - abs(v[i, j])) * (u[i, j+1] - u[i, j])) * dt/dy
                v_adv_dx = 0.5 * ((u[i, j] + abs(u[i, j])) * (v[i, j] - v[i-1, j]) \
                    + (u[i, j] - abs(u[i, j])) * (v[i+1, j] - v[i, j])) * dt/dx
                v_adv_dy = 0.5 * ((v[i, j] + abs(v[i, j])) * (v[i, j] - v[i, j-1]) \
                    + (v[i, j] - abs(v[i, j])) * (v[i, j+1] - v[i, j])) * dt/dy
                # velocity update with pressure gradient and diffusion
                u_next[i, j] = u[i, j] - u_adv_dx - u_adv_dy \
                    - dt/(2*rho*dx) * (p[i+1, j] - p[i-1, j]) + nu * dt \
                    * ((u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx**2 \
                    + (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy**2)
                v_next[i, j] = v[i, j] - v_adv_dx - v_adv_dy \
                    - dt/(2*rho*dy) * (p[i, j+1] - p[i, j-1]) + nu * dt \
                    * ((v[i+1, j] - 2*v[i, j] + v[i-1, j]) / dx**2 \
                    + (v[i, j+1] - 2*v[i, j] + v[i, j-1]) / dy**2)