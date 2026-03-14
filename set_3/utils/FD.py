import numpy as np
from numba import njit, prange
from scipy.signal import convolve2d
from warnings import warn

class FD:
    def __init__(self, dx: float = .01, dy: float = .01, dt: float = .001, rho: float = 1.0, nu: float = .1):
        # initialize parameters
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.rho = rho
        self.nu = nu
        pad = 2 # extra padding for boundary conditions
        # initialize velocity and pressure fields with extra padding for boundary conditions
        self.nx = round(2.2 // dx + 2*pad) + 1
        self.ny = round(0.41 // dy + 2*pad) + 1
        self.u = np.zeros(shape=(self.nx, self.ny))
        self.v = np.zeros_like(self.u)
        self.p = np.zeros_like(self.u)
        self.u[:pad, :] = 1.0 # inflow from the left
        # small perturbation to trigger turbulence
        pert = np.sin(2*np.pi*np.linspace(0, 1, self.u.shape[1]-2*pad))
        self.u[pad, pad:-pad] = 0.1 * (1 + pert)
        # set up the cylinder
        x_coords = np.pad(np.linspace(0, 2.2, self.nx-2*pad), (pad, pad), mode='edge')
        y_coords = np.pad(np.linspace(0, 0.41, self.ny-2*pad), (pad, pad), mode='edge')
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        cx, cy, r = 0.2, 0.2, 0.05
        self.mask = (X - cx)**2 + (Y - cy)**2 <= r**2
        self.u[self.mask] = 0.0
        self.v[self.mask] = 0.0
        # add walls to the mask
        self.mask[:pad, :] = True   # inlet
        self.mask[:, :pad] = True   # bottom boundary
        self.mask[:, -pad:] = True  # top boundary
        # calculate x and y neighbors for pressure solver
        x_kernel = np.array([
            [0, 1, 0],
            [0, 0, 0],
            [0, 1, 0]
        ])
        y_kernel = np.array([
            [0, 0, 0],
            [1, 0, 1],
            [0, 0, 0]
        ])
        self.x_neighbors = convolve2d(~self.mask, x_kernel, mode='same', boundary='fill', fillvalue=0)
        self.y_neighbors = convolve2d(~self.mask, y_kernel, mode='same', boundary='fill', fillvalue=0)

        self.Re = 1.0 * 2 * r / nu
        print(f"Initialized FD solver with Re={self.Re:.2f}, grid size=({self.nx-2*pad}, {self.ny-2*pad}), dt={self.dt:.4f}")

    def run(self, time: float, P: int = 2, U: int = 1, p_threshold: float = 1e-4, p_max_iters: int = 1000):
        """Dispatches the run loop according to the pressure correction method and the upwind advection scheme specified by P and U respectively."""
        match P:
            case 0:
                self._P0(time, p_threshold, p_max_iters)
            case 1:
                self._P1(time, U, p_threshold, p_max_iters)
            case 2:
                self._P2(time, U, p_threshold, p_max_iters)
            case _:
                raise ValueError(f"Invalid value for P: {P}. Must be 0 or 1.")

    def _P0(self, time: float, p_threshold: float, p_max_iters: int):
        """Update loop without pressure correction. Incompatible with higher-order advection schemes."""
        source = np.zeros_like(self.p)
        u_next = self.u.copy()
        v_next = self.v.copy()
        n_iters = int(time / self.dt)
        inv_dt = 1 / self.dt
        rho_dx_dy = (self.rho * self.dx**2 * self.dy**2) / (2 * (self.dx**2 + self.dy**2))
        inv_2dx, inv_2dy = 1 / (2*self.dx), 1 / (2*self.dy)
        dx2, dy2 = self.dx**2, self.dy**2
        inv_dx2dy2 = 1 / (dx2 + dy2)
        for _ in range(n_iters):
            # compute source and solve for pressure
            self._full_source(source, self.u, self.v, self.mask, rho_dx_dy, inv_dt, inv_2dx, inv_2dy)
            iter_count, _ = self._pressure(self.p, source, self.mask, self.x_neighbors, self.y_neighbors,
                                               dx2, dy2, inv_dx2dy2, p_threshold, p_max_iters)
            if iter_count >= p_max_iters:
                warn(f"Maximum iterations reached for pressure solver: {iter_count}")
            # update velocity field
            self._1st_order_P(self.u, self.v, u_next, v_next, self.p, self.mask,
                              self.rho, self.nu, self.dt, self.dx, self.dy)
            self.u, u_next = u_next, self.u
            self.v, v_next = v_next, self.v
            # enforce outflow boundary conditions
            self.u[-2:, :] = self.u[-3, :]
            self.v[-2:, :] = self.v[-3, :]
            # check CFL condition for stability
            CFL = (max(np.max(self.u), np.max(self.v)) * self.dt) / min(self.dx, self.dy)
            if CFL > 1:
                warn(f"CFL condition violated: {CFL:.2f} > 1. Consider reducing dt or increasing dx/dy for stability.")

    def _P1(self, time: float, order: int, p_threshold: float, p_max_iters: int):
        match order:
            case 1:
                predict = self._1st_order
            case 3:
                predict = self._3rd_order
            case _:
                raise ValueError(f"Invalid value for upwind order: {order}. Must be 1 or 3.")
        source = np.zeros_like(self.p)
        u_next = self.u.copy()
        v_next = self.v.copy()
        n_iters = int(time / self.dt)
        rho_const = (self.rho * self.dx**2 * self.dy**2) / (2 * self.dt * (self.dx**2 + self.dy**2))
        inv_2dx, inv_2dy = 1 / (2*self.dx), 1 / (2*self.dy)
        dx2, dy2 = self.dx**2, self.dy**2
        inv_dx2dy2 = 1 / (dx2 + dy2)
        dt_over_2rho_dx = self.dt / (2 * self.rho * self.dx)
        dt_over_2rho_dy = self.dt / (2 * self.rho * self.dy)
        for _ in range(n_iters):
            # perform prediction step to get intermediate velocity field
            predict(self.u, self.v, u_next, v_next, self.mask,
                    self.nu, self.dt, self.dx, self.dy)
            self.u, u_next = u_next, self.u
            self.v, v_next = v_next, self.v
            # enforce outflow boundary conditions
            self.u[-2:, :] = self.u[-3, :]
            self.v[-2:, :] = self.v[-3, :]
            # compute source term for pressure Poisson equation
            self._source(source, self.u, self.v, self.mask, rho_const, inv_2dx, inv_2dy)
            # solve for pressure and correct velocity field
            iter_count, _ = self._pressure(self.p, source, self.mask, self.x_neighbors, self.y_neighbors,
            dx2, dy2, inv_dx2dy2, p_threshold, p_max_iters)
            if iter_count >= p_max_iters:
                warn(f"Maximum iterations reached for pressure solver: {iter_count}")
            self._correct(self.u, self.v, self.p, self.mask, dt_over_2rho_dx, dt_over_2rho_dy)
            # enforce outflow boundary conditions again after correction
            self.u[-2:, :] = self.u[-3, :]
            self.v[-2:, :] = self.v[-3, :]
            # check CFL condition for stability
            CFL = (max(np.max(self.u), np.max(self.v)) * self.dt) / min(self.dx, self.dy)
            if CFL > 1:
                warn(f"CFL condition violated: {CFL:.2f} > 1. Consider reducing dt or increasing dx/dy for stability.")

    def _P2(self, time: float, order: int, p_threshold: float, p_max_iters: int):
        match order:
            case 0:
                predict = self._predict
            case 1:
                predict = self._1st_order_P
            case 3:
                predict = self._3rd_order_P
            case _:
                raise ValueError(f"Invalid value for upwind order: {order}. Must be 0, 1 or 3.")
        source = np.zeros_like(self.p)
        dp = np.zeros_like(self.p)
        u_next = self.u.copy()
        v_next = self.v.copy()
        n_iters = int(time / self.dt)
        rho_const = (self.rho * self.dx**2 * self.dy**2) / (2 * self.dt * (self.dx**2 + self.dy**2))
        inv_2dx, inv_2dy = 1 / (2*self.dx), 1 / (2*self.dy)
        dx2, dy2 = self.dx**2, self.dy**2
        inv_dx2dy2 = 1 / (dx2 + dy2)
        dt_over_2rho_dx = self.dt / (2 * self.rho * self.dx)
        dt_over_2rho_dy = self.dt / (2 * self.rho * self.dy)
        for _ in range(n_iters):
            # perform prediction step to get intermediate velocity field
            predict(self.u, self.v, u_next, v_next, self.p, self.mask,
                    self.rho, self.nu, self.dt, self.dx, self.dy)
            self.u, u_next = u_next, self.u
            self.v, v_next = v_next, self.v
            # enforce outflow boundary conditions
            self.u[-2:, :] = self.u[-3, :]
            self.v[-2:, :] = self.v[-3, :]
            # compute source term for pressure Poisson equation
            self._source(source, self.u, self.v, self.mask, rho_const, inv_2dx, inv_2dy)
            # solve for pressure and correct velocity field
            dp.fill(0.0) # reset pressure correction array
            iter_count, _ = self._pressure(dp, source, self.mask, self.x_neighbors, self.y_neighbors,
            dx2, dy2, inv_dx2dy2, p_threshold, p_max_iters)
            self.p += dp # accumulate pressure corrections for better prediction
            if iter_count >= p_max_iters:
                warn(f"Maximum iterations reached for pressure solver: {iter_count}")
            self._correct(self.u, self.v, dp, self.mask, dt_over_2rho_dx, dt_over_2rho_dy)
            # enforce outflow boundary conditions again after correction
            self.u[-2:, :] = self.u[-3, :]
            self.v[-2:, :] = self.v[-3, :]
            # check CFL condition for stability
            CFL = (max(np.max(self.u), np.max(self.v)) * self.dt) / min(self.dx, self.dy)
            if CFL > 1:
                warn(f"CFL condition violated: {CFL:.2f} > 1. Consider reducing dt or increasing dx/dy for stability.")

    def plot(self, show=True, save=False, filename=None):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(14, 0.41*16/2.2), constrained_layout=True)
        x_coords = np.linspace(0, 2.2, self.nx-4)
        y_coords = np.linspace(0, 0.41, self.ny-4)
        X, Y = np.meshgrid(x_coords, y_coords)
        cont = ax.contourf(X, Y, self.p.T[2:-2, 2:-2], levels=50, cmap='viridis')
        fig.colorbar(mappable=cont, label='Pressure')
        ax.streamplot(X, Y, self.u.T[2:-2, 2:-2], self.v.T[2:-2, 2:-2], color='white')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Pressure and Velocity Field: $Re$={self.Re:.0f}')
        ax.set_box_aspect(0.41/2.2)
        if save and filename is not None:
            plt.savefig(filename, dpi=300)
        if show:
            plt.show()

    @staticmethod
    @njit(parallel=True, fastmath=False)
    def _full_source(source: np.ndarray, u: np.ndarray, v: np.ndarray, mask: np.ndarray,
                     rho_dx_dy: float, inv_dt: float, inv_2dx: float, inv_2dy: float):
        """Computes the source term for the pressure Poisson equation based on the velocity field. Includes both the divergence of the velocity and the nonlinear advection terms. Used for P0 since no correction step is performed."""
        for i in prange(2, source.shape[0]-2):
            for j in range(2, source.shape[1]-2):
                if mask[i, j]: continue # skip points inside the cylinder and walls
                source[i, j] = rho_dx_dy * (inv_dt * ((u[i+1, j] - u[i-1, j]) * inv_2dx + (v[i, j+1] - v[i, j-1]) * inv_2dy) \
                    - ((u[i+1, j] - u[i-1, j]) * inv_2dx)**2 - ((v[i, j+1] - v[i, j-1]) * inv_2dy)**2 \
                    - 2 * ((u[i, j+1] - u[i, j-1]) * inv_2dy) * ((v[i+1, j] - v[i-1, j]) * inv_2dx))

    @staticmethod
    @njit(parallel=True, fastmath=False)
    def _source(source: np.ndarray, u: np.ndarray, v: np.ndarray, mask: np.ndarray,
                rho_const: float, inv_2dx: float, inv_2dy: float):
        """Computes the source term for the pressure Poisson equation based on the velocity field. Does not include the nonlinear advection terms since they are handled in the correction step for P1 and P2."""
        for i in prange(2, source.shape[0]-2):
            for j in range(2, source.shape[1]-2):
                if mask[i, j]: continue # skip points inside the cylinder and walls
                source[i, j] = rho_const * ((u[i+1, j] - u[i-1, j]) * inv_2dx + (v[i, j+1] - v[i, j-1]) * inv_2dy)

    @staticmethod
    @njit(parallel=True, fastmath=False)
    def _pressure(p: np.ndarray, source: np.ndarray, mask: np.ndarray, x_neighbors: np.ndarray, y_neighbors: np.ndarray,
                  dx2: float, dy2: float, inv_dx2dy2: float, threshold: float, max_iters: int):
        """Solves the Poisson equation for pressure using the red-black Gauss-Seidel method."""
        # NOTE: updating the error in parallel causes a race condition
        iter_count = 0
        error = np.inf
        while error > threshold and iter_count < max_iters:
            error = 0.0
            # red points
            for i in prange(2, p.shape[0]-2):
                start = 3 - (i % 2)
                for j in range(start, p.shape[1]-2, 2):
                    if mask[i, j]: continue # skip points inside the cylinder and walls
                    next = ((p[i, j+1] + p[i, j-1]) / y_neighbors[i, j] * dx2 \
                        + (p[i+1, j] + p[i-1, j]) / x_neighbors[i, j] * dy2) \
                        * inv_dx2dy2 - source[i, j]
                    error = max(error, abs(next - p[i, j]))
                    p[i, j] = next
            # black points
            for i in prange(2, p.shape[0]-2):
                start = 2 + (i % 2)
                for j in range(start, p.shape[1]-2, 2):
                    if mask[i, j]: continue # skip points inside the cylinder and walls
                    next = ((p[i, j+1] + p[i, j-1]) / y_neighbors[i, j] * dx2 \
                        + (p[i+1, j] + p[i-1, j]) / x_neighbors[i, j] * dy2) \
                        * inv_dx2dy2 - source[i, j]
                    error = max(error, abs(next - p[i, j]))
                    p[i, j] = next
            iter_count += 1
        return iter_count, error

    @staticmethod
    @njit(parallel=True, fastmath=False)
    def _predict(u: np.ndarray, v: np.ndarray, u_next: np.ndarray, v_next: np.ndarray, p: np.ndarray,
                 mask:np.ndarray, rho: float, nu: float, dt: float, dx: float, dy: float):
        """Updates the velocity field based on the Navier-Stokes equations using combined upwind scheme. Includes the pressure gradient term for higher accuracy prediction."""
        for i in prange(2, u.shape[0]-2):
            for j in range(2, u.shape[1]-2):
                if mask[i, j]: continue # skip points inside the cylinder
                # Neumann boundary conditions for pressure at cylinder
                p_E = p[i+1, j] * (1.0 - mask[i+1, j]) + p[i, j] * mask[i+1, j]
                p_W = p[i-1, j] * (1.0 - mask[i-1, j]) + p[i, j] * mask[i-1, j]
                p_N = p[i, j+1] * (1.0 - mask[i, j+1]) + p[i, j] * mask[i, j+1]
                p_S = p[i, j-1] * (1.0 - mask[i, j-1]) + p[i, j] * mask[i, j-1]
                # check if any of third-order points are inside the mask
                safe_E = (1-mask[i+1, j]) * (1-mask[i-1, j]) * (1-mask[i-2, j])
                safe_W = (1-mask[i-1, j]) * (1-mask[i+1, j]) * (1-mask[i+2, j])
                safe_N = (1-mask[i, j+1]) * (1-mask[i, j-1]) * (1-mask[i, j-2])
                safe_S = (1-mask[i, j-1]) * (1-mask[i, j+1]) * (1-mask[i, j+2])
                # first-order upwind terms for advection
                u_E1 = u[i, j] - u[i-1, j]
                u_W1 = u[i+1, j] - u[i, j]
                u_N1 = u[i, j] - u[i, j-1]
                u_S1 = u[i, j+1] - u[i, j]
                v_E1 = v[i, j] - v[i-1, j]
                v_W1 = v[i+1, j] - v[i, j]
                v_N1 = v[i, j] - v[i, j-1]
                v_S1 = v[i, j+1] - v[i, j]
                # third-order upwind terms for advection
                u_E3 = (2*u[i+1, j] + 3*u[i, j] - 6*u[i-1, j] + u[i-2, j]) / 6.0
                u_W3 = (-u[i+2, j] + 6*u[i+1, j] - 3*u[i, j] - 2*u[i-1, j]) / 6.0
                u_N3 = (2*u[i, j+1] + 3*u[i, j] - 6*u[i, j-1] + u[i, j-2]) / 6.0
                u_S3 = (-u[i, j+2] + 6*u[i, j+1] - 3*u[i, j] - 2*u[i, j-1]) / 6.0
                v_E3 = (2*v[i+1, j] + 3*v[i, j] - 6*v[i-1, j] + v[i-2, j]) / 6.0
                v_W3 = (-v[i+2, j] + 6*v[i+1, j] - 3*v[i, j] - 2*v[i-1, j]) / 6.0
                v_N3 = (2*v[i, j+1] + 3*v[i, j] - 6*v[i, j-1] + v[i, j-2]) / 6.0
                v_S3 = (-v[i, j+2] + 6*v[i, j+1] - 3*v[i, j] - 2*v[i, j-1]) / 6.0
                # blend first-order and third-order terms based on safety of third-order points
                u_E  = safe_E * u_E3 + (1 - safe_E) * u_E1
                u_W  = safe_W * u_W3 + (1 - safe_W) * u_W1
                u_N  = safe_N * u_N3 + (1 - safe_N) * u_N1
                u_S  = safe_S * u_S3 + (1 - safe_S) * u_S1
                v_E  = safe_E * v_E3 + (1 - safe_E) * v_E1
                v_W  = safe_W * v_W3 + (1 - safe_W) * v_W1
                v_N  = safe_N * v_N3 + (1 - safe_N) * v_N1
                v_S  = safe_S * v_S3 + (1 - safe_S) * v_S1
                # combine upwind terms for advection
                u_adv_dx = 0.5 * ((u[i, j] + abs(u[i, j])) * u_E + (u[i, j] - abs(u[i, j])) * u_W) * dt/dx
                u_adv_dy = 0.5 * ((v[i, j] + abs(v[i, j])) * u_N + (v[i, j] - abs(v[i, j])) * u_S) * dt/dy
                v_adv_dx = 0.5 * ((u[i, j] + abs(u[i, j])) * v_E + (u[i, j] - abs(u[i, j])) * v_W) * dt/dx
                v_adv_dy = 0.5 * ((v[i, j] + abs(v[i, j])) * v_N + (v[i, j] - abs(v[i, j])) * v_S) * dt/dy
                # velocity update with advection, pressure gradient, and diffusion
                u_next[i, j] = u[i, j] - u_adv_dx - u_adv_dy \
                    - dt/(2*rho*dx) * (p_E - p_W) + nu * dt \
                    * ((u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx**2 \
                    + (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy**2)
                v_next[i, j] = v[i, j] - v_adv_dx - v_adv_dy \
                    - dt/(2*rho*dy) * (p_N - p_S) + nu * dt \
                    * ((v[i+1, j] - 2*v[i, j] + v[i-1, j]) / dx**2 \
                    + (v[i, j+1] - 2*v[i, j] + v[i, j-1]) / dy**2)

    @staticmethod
    @njit(parallel=True, fastmath=False)
    def _correct(u: np.ndarray, v: np.ndarray, p: np.ndarray, mask: np.ndarray,
                 dt_over_2rho_dx: float, dt_over_2rho_dy: float):
        """Corrects the velocity field based on the pressure gradient."""
        for i in prange(2, u.shape[0]-2):
            for j in range(2, u.shape[1]-2):
                if mask[i, j]: continue # skip points inside the cylinder and walls
                # Neumann boundary conditions for pressure at cylinder and walls
                p_E = p[i+1, j] * (1.0 - mask[i+1, j]) + p[i, j] * mask[i+1, j]
                p_W = p[i-1, j] * (1.0 - mask[i-1, j]) + p[i, j] * mask[i-1, j]
                p_N = p[i, j+1] * (1.0 - mask[i, j+1]) + p[i, j] * mask[i, j+1]
                p_S = p[i, j-1] * (1.0 - mask[i, j-1]) + p[i, j] * mask[i, j-1]
                u[i, j] -= dt_over_2rho_dx * (p_E - p_W)
                v[i, j] -= dt_over_2rho_dy * (p_N - p_S)

    @staticmethod
    @njit(parallel=True, fastmath=False)
    def _1st_order_P(u: np.ndarray, v: np.ndarray, u_next: np.ndarray, v_next: np.ndarray, p: np.ndarray,
                        mask:np.ndarray, rho: float, nu: float, dt: float, dx: float, dy: float):
        """Updates the velocity field based on the Navier-Stokes equations using combined first-order upwind scheme. Includes the pressure gradient term for higher accuracy prediction."""
        for i in prange(2, u.shape[0]-2):
            for j in range(2, u.shape[1]-2):
                if mask[i, j]: continue # skip points inside the cylinder
                # Neumann boundary conditions for pressure at cylinder
                p_E = p[i+1, j] * (1.0 - mask[i+1, j]) + p[i, j] * mask[i+1, j]
                p_W = p[i-1, j] * (1.0 - mask[i-1, j]) + p[i, j] * mask[i-1, j]
                p_N = p[i, j+1] * (1.0 - mask[i, j+1]) + p[i, j] * mask[i, j+1]
                p_S = p[i, j-1] * (1.0 - mask[i, j-1]) + p[i, j] * mask[i, j-1]
                # combined upwind terms for advection
                u_adv_dx = 0.5 * ((u[i, j] + abs(u[i, j])) * (u[i, j] - u[i-1, j]) \
                    + (u[i, j] - abs(u[i, j])) * (u[i+1, j] - u[i, j])) * dt/dx
                u_adv_dy = 0.5 * ((v[i, j] + abs(v[i, j])) * (u[i, j] - u[i, j-1]) \
                    + (v[i, j] - abs(v[i, j])) * (u[i, j+1] - u[i, j])) * dt/dy
                v_adv_dx = 0.5 * ((u[i, j] + abs(u[i, j])) * (v[i, j] - v[i-1, j]) \
                    + (u[i, j] - abs(u[i, j])) * (v[i+1, j] - v[i, j])) * dt/dx
                v_adv_dy = 0.5 * ((v[i, j] + abs(v[i, j])) * (v[i, j] - v[i, j-1]) \
                    + (v[i, j] - abs(v[i, j])) * (v[i, j+1] - v[i, j])) * dt/dy
                # velocity update with advection, pressure gradient, and diffusion
                u_next[i, j] = u[i, j] - u_adv_dx - u_adv_dy \
                    - dt/(2*rho*dx) * (p_E - p_W) + nu * dt \
                    * ((u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx**2 \
                    + (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy**2)
                v_next[i, j] = v[i, j] - v_adv_dx - v_adv_dy \
                    - dt/(2*rho*dy) * (p_N - p_S) + nu * dt \
                    * ((v[i+1, j] - 2*v[i, j] + v[i-1, j]) / dx**2 \
                    + (v[i, j+1] - 2*v[i, j] + v[i, j-1]) / dy**2)
                
    @staticmethod
    @njit(parallel=True, fastmath=False)
    def _1st_order(u: np.ndarray, v: np.ndarray, u_next: np.ndarray, v_next: np.ndarray,
                   mask: np.ndarray, nu: float, dt: float, dx: float, dy: float):
        """Updates the velocity field based on the Navier-Stokes equations using combined first-order upwind scheme. Does not include the pressure gradient."""
        for i in prange(2, u.shape[0]-2):
            for j in range(2, u.shape[1]-2):
                if mask[i, j]: continue # skip points inside the cylinder and walls
                # combined upwind terms for advection
                u_adv_dx = 0.5 * ((u[i, j] + abs(u[i, j])) * (u[i, j] - u[i-1, j]) \
                    + (u[i, j] - abs(u[i, j])) * (u[i+1, j] - u[i, j])) * dt/dx
                u_adv_dy = 0.5 * ((v[i, j] + abs(v[i, j])) * (u[i, j] - u[i, j-1]) \
                    + (v[i, j] - abs(v[i, j])) * (u[i, j+1] - u[i, j])) * dt/dy
                v_adv_dx = 0.5 * ((u[i, j] + abs(u[i, j])) * (v[i, j] - v[i-1, j]) \
                    + (u[i, j] - abs(u[i, j])) * (v[i+1, j] - v[i, j])) * dt/dx
                v_adv_dy = 0.5 * ((v[i, j] + abs(v[i, j])) * (v[i, j] - v[i, j-1]) \
                    + (v[i, j] - abs(v[i, j])) * (v[i, j+1] - v[i, j])) * dt/dy
                # velocity update with advection and diffusion
                u_next[i, j] = u[i, j] - u_adv_dx - u_adv_dy \
                    + nu * dt * ((u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx**2 \
                    + (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy**2)
                v_next[i, j] = v[i, j] - v_adv_dx - v_adv_dy \
                    + nu * dt * ((v[i+1, j] - 2*v[i, j] + v[i-1, j]) / dx**2 \
                    + (v[i, j+1] - 2*v[i, j] + v[i, j-1]) / dy**2)
                
    @staticmethod
    @njit(parallel=True, fastmath=False)
    def _3rd_order_P(u: np.ndarray, v: np.ndarray, u_next: np.ndarray, v_next: np.ndarray, p: np.ndarray,
                        mask:np.ndarray, rho: float, nu: float, dt: float, dx: float, dy: float):
        """Updates the velocity field based on the Navier-Stokes equations using combined third-order upwind scheme. Includes the pressure gradient term for higher accuracy prediction."""
        for i in prange(2, u.shape[0]-2):
            for j in range(2, u.shape[1]-2):
                if mask[i, j]: continue # skip points inside the cylinder
                # Neumann boundary conditions for pressure at cylinder
                p_E = p[i+1, j] * (1.0 - mask[i+1, j]) + p[i, j] * mask[i+1, j]
                p_W = p[i-1, j] * (1.0 - mask[i-1, j]) + p[i, j] * mask[i-1, j]
                p_N = p[i, j+1] * (1.0 - mask[i, j+1]) + p[i, j] * mask[i, j+1]
                p_S = p[i, j-1] * (1.0 - mask[i, j-1]) + p[i, j] * mask[i, j-1]
                # combined upwind terms for advection
                u_adv_dx = (1/12) * ((u[i, j] + abs(u[i, j])) * (2*u[i+1, j] + 3*u[i, j] - 6*u[i-1, j] + u[i-2, j]) \
                    + (u[i, j] - abs(u[i, j])) * (-u[i+2, j] + 6*u[i+1, j] - 3*u[i, j] - 2*u[i-1, j])) * dt/dx
                u_adv_dy = (1/12) * ((v[i, j] + abs(v[i, j])) * (2*u[i, j+1] + 3*u[i, j] - 6*u[i, j-1] + u[i, j-2]) \
                    + (v[i, j] - abs(v[i, j])) * (-u[i, j+2] + 6*u[i, j+1] - 3*u[i, j] - 2*u[i, j-1])) * dt/dy
                v_adv_dx = (1/12) * ((u[i, j] + abs(u[i, j])) * (2*v[i+1, j] + 3*v[i, j] - 6*v[i-1, j] + v[i-2, j]) \
                    + (u[i, j] - abs(u[i, j])) * (-v[i+2, j] + 6*v[i+1, j] - 3*v[i, j] - 2*v[i-1, j])) * dt/dx
                v_adv_dy = (1/12) * ((v[i, j] + abs(v[i, j])) * (2*v[i, j+1] + 3*v[i, j] - 6*v[i, j-1] + v[i, j-2]) \
                    + (v[i, j] - abs(v[i, j])) * (-v[i, j+2] + 6*v[i, j+1] - 3*v[i, j] - 2*v[i, j-1])) * dt/dy
                # velocity update with advection, pressure gradient, and diffusion
                u_next[i, j] = u[i, j] - u_adv_dx - u_adv_dy \
                    - dt/(2*rho*dx) * (p_E - p_W) + nu * dt \
                    * ((u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx**2 \
                    + (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy**2)
                v_next[i, j] = v[i, j] - v_adv_dx - v_adv_dy \
                    - dt/(2*rho*dy) * (p_N - p_S) + nu * dt \
                    * ((v[i+1, j] - 2*v[i, j] + v[i-1, j]) / dx**2 \
                    + (v[i, j+1] - 2*v[i, j] + v[i, j-1]) / dy**2)
                
    @staticmethod
    @njit(parallel=True, fastmath=False)
    def _3rd_order(u: np.ndarray, v: np.ndarray, u_next: np.ndarray, v_next: np.ndarray,
                   mask: np.ndarray, nu: float, dt: float, dx: float, dy: float):
        """Updates the velocity field based on the Navier-Stokes equations using combined third-order upwind scheme. Does not include the pressure gradient."""
        for i in prange(2, u.shape[0]-2):
            for j in range(2, u.shape[1]-2):
                if mask[i, j]: continue # skip points inside the cylinder and walls
                # combined upwind terms for advection
                u_adv_dx = (1/12) * ((u[i, j] + abs(u[i, j])) * (2*u[i+1, j] + 3*u[i, j] - 6*u[i-1, j] + u[i-2, j]) \
                    + (u[i, j] - abs(u[i, j])) * (-u[i+2, j] + 6*u[i+1, j] - 3*u[i, j] - 2*u[i-1, j])) * dt/dx
                u_adv_dy = (1/12) * ((v[i, j] + abs(v[i, j])) * (2*u[i, j+1] + 3*u[i, j] - 6*u[i, j-1] + u[i, j-2]) \
                    + (v[i, j] - abs(v[i, j])) * (-u[i, j+2] + 6*u[i, j+1] - 3*u[i, j] - 2*u[i, j-1])) * dt/dy
                v_adv_dx = (1/12) * ((u[i, j] + abs(u[i, j])) * (2*v[i+1, j] + 3*v[i, j] - 6*v[i-1, j] + v[i-2, j]) \
                    + (u[i, j] - abs(u[i, j])) * (-v[i+2, j] + 6*v[i+1, j] - 3*v[i, j] - 2*v[i-1, j])) * dt/dx
                v_adv_dy = (1/12) * ((v[i, j] + abs(v[i, j])) * (2*v[i, j+1] + 3*v[i, j] - 6*v[i, j-1] + v[i, j-2]) \
                    + (v[i, j] - abs(v[i, j])) * (-v[i, j+2] + 6*v[i, j+1] - 3*v[i, j] - 2*v[i, j-1])) * dt/dy
                # velocity update with advection and diffusion
                u_next[i, j] = u[i, j] - u_adv_dx - u_adv_dy \
                    + nu * dt * ((u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx**2 \
                    + (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy**2)
                v_next[i, j] = v[i, j] - v_adv_dx - v_adv_dy \
                    + nu * dt * ((v[i+1, j] - 2*v[i, j] + v[i-1, j]) / dx**2 \
                    + (v[i, j+1] - 2*v[i, j] + v[i, j-1]) / dy**2)