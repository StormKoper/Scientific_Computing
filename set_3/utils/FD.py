import numpy as np
from numba import njit, prange
from scipy.signal import convolve2d
from tqdm import tqdm
from warnings import warn

FASTMATH = True

class FD:
    def __init__(self, dx: float = .01, dy: float = .01, dt: float = .001,
                 rho: float = 1.0, nu: float = .1):
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
        # set up the inflow and cylinder
        x_coords = np.pad(np.linspace(0, 2.2, self.nx-2*pad), (pad, pad), mode='edge')
        y_coords = np.pad(np.linspace(0, 0.41, self.ny-2*pad), (pad, pad), mode='edge')
        # parabolic inflow from the left (in line with DFG 2D-2 Benchmark)
        Um = 1.5
        H = 0.41
        uin_x = 4 * Um * y_coords * (H - y_coords) / (H**2)
        self.u[:pad, :] = uin_x[None, :]
        # cylinder mask for no-slip condition
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
        x_neighbors = convolve2d(~self.mask, x_kernel, mode='same', boundary='fill', fillvalue=0)
        y_neighbors = convolve2d(~self.mask, y_kernel, mode='same', boundary='fill', fillvalue=0)
        self.A_diag = x_neighbors * self.dy**2 + y_neighbors * self.dx**2
        self.inv_A_diag = np.divide(1.0, self.A_diag, out=np.zeros_like(self.A_diag), where=~self.mask)
        # calculate Reynolds number for reference
        self.Re = 1.0 * 2 * r / nu # Re = U*L/nu with L=diameter=2r and U=2/3 Um = 1.0
        print(f"Initialized FD solver with Re={self.Re:.2f}, grid size=({self.nx-2*pad}, {self.ny-2*pad}), dt={self.dt:.4f}")
        # set up probes for calculating Strouhal number
        self.probes_x = [int((cx+5*r)/dx)+pad,   int((cx+5*r)/dx)+pad,   int((cx+7*r)/dx)+pad,   int((cx+7*r)/dx)+pad]
        self.probes_y = [int((cy+0.5*r)/dy)+pad, int((cy-0.5*r)/dy)+pad, int((cy+0.5*r)/dy)+pad, int((cy-0.5*r)/dy)+pad]
        self.strouhal_data = {"u": [], "v": []}

    def run(self, time: float, p_threshold: float = 1e-4, p_max_iters: int = 5000, probe: int = 0):
        # pre-allocate arrays for the solver to avoid repeated allocations inside the loop
        source = np.zeros_like(self.p)
        Ap = np.zeros_like(self.p)
        search_dir = np.zeros_like(self.p)
        dp = np.zeros_like(self.p)
        u_next = self.u.copy()
        v_next = self.v.copy()
        du_prev = np.zeros_like(self.u)
        dv_prev = np.zeros_like(self.v)
        n_iters = int(time / self.dt)
        # precompute constant coefficients for the source term and pressure solver to optimize performance
        half_rho_dx2_dy2_over_dt = 0.5 * (self.rho * self.dx**2 * self.dy**2) / self.dt
        dx2, dy2 = self.dx**2, self.dy**2
        dt_over_2rho_dx = self.dt / (2 * self.rho * self.dx)
        dt_over_2rho_dy = self.dt / (2 * self.rho * self.dy)
        # compute initial source
        self._source(source, self.u, self.v, self.mask, half_rho_dx2_dy2_over_dt, self.dx, self.dy)
        # compute initial pressure
        iter_count, _ = self._pressure(self.p, source, self.mask, self.A_diag, self.inv_A_diag,
                                       Ap, search_dir, dx2, dy2, p_threshold, p_max_iters)
        if iter_count >= p_max_iters:
            warn(f"Maximum iterations reached for pressure solver: {iter_count}")
        # perform prediction step to get intermediate velocity field
        self._predict(self.u, self.v, u_next, v_next, du_prev, dv_prev, self.mask,
                      self.nu, self.dt, self.dx, self.dy, AB=0.0)
        self._correct(u_next, v_next, self.p, self.mask, dt_over_2rho_dx, dt_over_2rho_dy)
        self.u, u_next = u_next, self.u
        self.v, v_next = v_next, self.v
        # enforce outflow boundary conditions
        self.u[-2:, :] = self.u[-3, :]
        self.v[-2:, :] = self.v[-3, :]
        # compute source term for pressure Poisson equation
        self._source(source, self.u, self.v, self.mask, half_rho_dx2_dy2_over_dt, self.dx, self.dy)
        # solve for pressure and correct velocity field
        iter_count, _ = self._pressure(dp, source, self.mask, self.A_diag, self.inv_A_diag,
                                       Ap, search_dir, dx2, dy2, p_threshold, p_max_iters)
        self.p += dp # accumulate pressure corrections for better prediction
        if iter_count >= p_max_iters:
            warn(f"Maximum iterations reached for pressure solver: {iter_count}")
        self._correct(self.u, self.v, dp, self.mask, dt_over_2rho_dx, dt_over_2rho_dy)
        # enforce outflow boundary conditions again after correction
        self.u[-2:, :] = self.u[-3, :]
        self.v[-2:, :] = self.v[-3, :]
        # check CFL condition for stability
        CFL = (max(np.max(self.u), np.max(self.v)) * self.dt) / min(self.dx, self.dy)
        if CFL >= 1:
            warn(f"CFL condition violated: {CFL:.2f} >= 1. Consider reducing dt or increasing dx/dy for stability.")
        if CFL >= 10:
            raise RuntimeError(f"CFL condition severely violated: {CFL:.2f} >> 1. Simulation is likely diverging. Consider significantly reducing dt or increasing dx/dy for stability.")
        if probe:
            self.strouhal_data["u"].extend(self.u[self.probes_x, self.probes_y])
            self.strouhal_data["v"].extend(self.v[self.probes_x, self.probes_y])
        for n in tqdm(range(1, n_iters), desc="Running simulation", leave=False, total=n_iters, initial=1):
            # perform prediction step to get intermediate velocity field
            self._predict(self.u, self.v, u_next, v_next, du_prev, dv_prev, self.mask,
                          self.nu, self.dt, self.dx, self.dy)
            self._correct(u_next, v_next, self.p, self.mask, dt_over_2rho_dx, dt_over_2rho_dy)
            self.u, u_next = u_next, self.u
            self.v, v_next = v_next, self.v
            # enforce outflow boundary conditions
            self.u[-2:, :] = self.u[-3, :]
            self.v[-2:, :] = self.v[-3, :]
            # compute source term for pressure Poisson equation
            self._source(source, self.u, self.v, self.mask, half_rho_dx2_dy2_over_dt, self.dx, self.dy)
            # solve for pressure and correct velocity field
            dp.fill(0.0) # reset pressure correction array
            iter_count, _ = self._pressure(dp, source, self.mask, self.A_diag, self.inv_A_diag,
                                           Ap, search_dir, dx2, dy2, p_threshold, p_max_iters)
            self.p += dp # accumulate pressure corrections for better prediction
            if iter_count >= p_max_iters:
                warn(f"Maximum iterations reached for pressure solver: {iter_count}")
            self._correct(self.u, self.v, dp, self.mask, dt_over_2rho_dx, dt_over_2rho_dy)
            # enforce outflow boundary conditions again after correction
            self.u[-2:, :] = self.u[-3, :]
            self.v[-2:, :] = self.v[-3, :]
            # check CFL condition for stability
            CFL = (max(np.max(self.u), np.max(self.v)) * self.dt) / min(self.dx, self.dy)
            if CFL >= 1:
                warn(f"CFL condition violated: {CFL:.2f} >= 1. Consider reducing dt or increasing dx/dy for stability.")
            if CFL >= 10:
                raise RuntimeError(f"CFL condition severely violated: {CFL:.2f} >> 1. Simulation is likely diverging. Consider significantly reducing dt or increasing dx/dy for stability.")
            if probe and n % probe == 0:
                self.strouhal_data["u"].extend(self.u[self.probes_x, self.probes_y])
                self.strouhal_data["v"].extend(self.v[self.probes_x, self.probes_y])

    def plot(self, show=True, save=False, filename=None):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(14, 0.41*16/2.2), constrained_layout=True)
        x_coords = np.linspace(0, 2.2, self.nx-4)
        y_coords = np.linspace(0, 0.41, self.ny-4)
        X, Y = np.meshgrid(x_coords, y_coords)
        pressure = np.ma.masked_array(self.p[2:-2, 2:-2], mask=self.mask[2:-2, 2:-2])
        cont = ax.contourf(X, Y, pressure.T, levels=50, cmap='viridis')
        fig.colorbar(mappable=cont, label='Pressure')
        ax.streamplot(X, Y, self.u.T[2:-2, 2:-2], self.v.T[2:-2, 2:-2], color='white')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f"Pressure and Velocity Field: $Re={self.Re:.0f}$ \
                     $\\Delta t={self.dt:.4f}$, $\\Delta x={self.dx:.4f}$, $\\Delta y={self.dy:.4f}$")
        ax.set_box_aspect(0.41/2.2)
        if save and filename is not None: plt.savefig(filename, dpi=300)
        if show: plt.show()
    
    def plot_strouhal(self, probe: int, show=True, save=False, filename=None):
        import matplotlib.pyplot as plt
        probes = ["Top left", "Bottom left", "Top right", "Bottom right"]
        u_data = np.array(self.strouhal_data["u"]).reshape(-1, 4)
        v_data = np.array(self.strouhal_data["v"]).reshape(-1, 4)
        signal = v_data - np.mean(v_data, axis=0) # remove mean to focus on oscillations
        freqs, mags = [], []
        f_doms, strouhals = [], []
        for i in range(signal.shape[1]):
            fft_vals = np.fft.rfft(signal[:, i])
            fft_freqs = np.fft.rfftfreq(v_data.shape[0], d=self.dt * probe)
            pos_mask = fft_freqs > 0
            freqs.append(fft_freqs[pos_mask])
            mags.append(np.abs(fft_vals)[pos_mask])
            f_doms.append(freqs[i][np.argmax(mags[i])])
            strouhals.append(f_doms[i] * 2*0.05 / 1.0) # St = f*L/U with L=diameter=0.1 and U=2/3 Um = 1.0
            print(f"{probes[i]} probe")
            print(f"Dominant Shedding Frequency: {f_doms[i]:.2f} Hz")
            print(f"Calculated Strouhal Number:  {strouhals[i]:.4f}")
        fig, ax = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
        fig.suptitle(f"Strouhal Analysis at Probes (every {probe} step{'' if probe == 1 else 's'}), $Re={self.Re:.0f}$")
        # u-component time series
        ax[0].plot(u_data)
        ax[0].set_title("Velocity at Strouhal Probe (u component)")
        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel("Velocity")
        ax[0].legend(probes, loc='upper left')
        ax[0].set_xticks(np.linspace(0, len(u_data)-1, 11), np.round(np.linspace(1, len(u_data), 11) * self.dt * probe, decimals=3))
        # v-component time series
        ax[1].plot(v_data)
        ax[1].set_title("Velocity at Strouhal Probe (v component)")
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("Velocity")
        ax[1].legend(probes, loc='upper left')
        ax[1].set_xticks(np.linspace(0, len(v_data)-1, 11), np.round(np.linspace(1, len(v_data), 11) * self.dt * probe, decimals=3))
        # FFT spectrum of v-component
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        for i in range(signal.shape[1]):
            ax[2].plot(freqs[i], mags[i], label=probes[i], color=colors[i])
            ax[2].axvline(f_doms[i], color=colors[i], linestyle='--', label=f'Peak: {f_doms[i]:.2f} Hz\nSt: {strouhals[i]:.3f}')
        ax[2].set_title("Frequency Spectrum (v component, Top Right Probe)")
        ax[2].set_xlabel("Frequency (Hz)")
        ax[2].set_ylabel("Magnitude")
        ax[2].set_xlim(0, max(10, max(f_doms) * 3)) # zoom in on the relevant low-frequency range
        ax[2].legend()
        if save and filename is not None:
            plt.savefig(filename, dpi=300)
        if show:
            plt.show()

    @staticmethod
    @njit(parallel=True, fastmath=FASTMATH)
    def _source(source: np.ndarray, u: np.ndarray, v: np.ndarray, mask: np.ndarray,
                half_rho_dx2_dy2_over_dt: float, dx: float, dy: float):
        """Computes the source term for the pressure Poisson equation based on the velocity field. Does not include the nonlinear advection terms since they are handled in the correction step for P1 and P2."""
        for i in prange(2, source.shape[0]-2):
            for j in range(2, source.shape[1]-2):
                if mask[i, j]: continue # skip points inside the cylinder and walls
                source[i, j] = half_rho_dx2_dy2_over_dt * ((u[i+1, j] - u[i-1, j]) / dx + (v[i, j+1] - v[i, j-1]) / dy)

    @staticmethod
    @njit(parallel=True, fastmath=FASTMATH)
    def _pressure(p: np.ndarray, source: np.ndarray, mask: np.ndarray, A_diag: np.ndarray, inv_A_diag: np.ndarray,
                  Ap: np.ndarray, search_dir: np.ndarray, dx2: float, dy2: float, threshold: float, max_iters: int):
        """Solves the Poisson equation for pressure using Matrix-Free Preconditioned Conjugate Gradient (Jacobi Preconditioner). NOTE: 'source' is mutated in-place and becomes the residual array (r)."""
        # initialize search direction and compute initial residual
        rz_old = 0.0
        for i in prange(2, p.shape[0]-2):
            for j in range(2, p.shape[1]-2):
                if mask[i, j]: continue
                # A * p
                Ap[i, j] = A_diag[i, j] * p[i, j] - \
                    (p[i+1, j] + p[i-1, j]) * dy2 - \
                    (p[i, j+1] + p[i, j-1]) * dx2
                # r = b - A*p
                source[i, j] = -source[i, j] - Ap[i, j]
                # Jacobi preconditioner (M^-1 * r)
                search_dir[i, j] = inv_A_diag[i, j] * source[i, j]
                rz_old += source[i, j] * search_dir[i, j]
        # main CG loop
        iter_count = 0
        error = np.inf
        while iter_count < max_iters:
            # compute A * search_dir and dot product with search_dir for alpha
            dir_dot_A = 0.0
            for i in prange(2, p.shape[0]-2):
                for j in range(2, p.shape[1]-2):
                    if mask[i, j]: continue
                    Ap[i, j] = A_diag[i, j] * search_dir[i, j] \
                        - (search_dir[i+1, j] + search_dir[i-1, j]) * dy2 \
                        - (search_dir[i, j+1] + search_dir[i, j-1]) * dx2
                    dir_dot_A += search_dir[i, j] * Ap[i, j]
            alpha = rz_old / dir_dot_A
            # update pressure and residual, calculate new rz and error
            rz_new = 0.0
            error = 0.0
            for i in prange(2, p.shape[0]-2):
                for j in range(2, p.shape[1]-2):
                    if mask[i, j]: continue
                    p[i, j] += alpha * search_dir[i, j]
                    source[i, j] -= alpha * Ap[i, j]
                    # apply preconditioner locally again
                    z_val = inv_A_diag[i, j] * source[i, j]
                    rz_new += source[i, j] * z_val
                    error = max(error, abs(z_val)) 
            # check convergence
            if error < threshold:
                break 
            # calculate beta and update search direction
            beta = rz_new / rz_old
            for i in prange(2, p.shape[0]-2):
                for j in range(2, p.shape[1]-2):
                    if mask[i, j]: continue
                    # recompute local z_val one last time for the search direction update
                    z_val = inv_A_diag[i, j] * source[i, j]
                    search_dir[i, j] = z_val + beta * search_dir[i, j]
            rz_old = rz_new
            iter_count += 1
        return iter_count, error
    
    @staticmethod
    @njit(parallel=True, fastmath=FASTMATH)
    def _pressure_SOR(p: np.ndarray, source: np.ndarray, mask: np.ndarray, x_neighbors: np.ndarray, y_neighbors: np.ndarray,
                      omega: float, dx2: float, dy2: float, threshold: float, max_iters: int):
        """Solves the Poisson equation for pressure using the red-black Successive Over-Relaxation method."""
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
                    den = x_neighbors[i, j] * dy2 + y_neighbors[i, j] * dx2
                    next_p = ((p[i+1, j] + p[i-1, j]) * dy2 + (p[i, j+1] + p[i, j-1]) * dx2 - source[i, j]) / den
                    next_val = omega * next_p + (1 - omega) * p[i, j]
                    error = max(error, abs(next_val - p[i, j]))
                    p[i, j] = next_val
            # black points
            for i in prange(2, p.shape[0]-2):
                start = 2 + (i % 2)
                for j in range(start, p.shape[1]-2, 2):
                    if mask[i, j]: continue # skip points inside the cylinder and walls
                    den = x_neighbors[i, j] * dy2 + y_neighbors[i, j] * dx2
                    next_p = ((p[i+1, j] + p[i-1, j]) * dy2 + (p[i, j+1] + p[i, j-1]) * dx2 - source[i, j]) / den
                    next_val = omega * next_p + (1 - omega) * p[i, j]
                    error = max(error, abs(next_val - p[i, j]))
                    p[i, j] = next_val
            iter_count += 1
        return iter_count, error
    
    @staticmethod
    @njit(parallel=True, fastmath=FASTMATH)
    def _predict(u: np.ndarray, v: np.ndarray, u_next: np.ndarray, v_next: np.ndarray,
                 du_prev: np.ndarray, dv_prev: np.ndarray, mask: np.ndarray,
                 nu: float, dt: float, dx: float, dy: float, AB: float = 0.5):
        """Updates the velocity field based on the Navier-Stokes equations using combined third-order upwind scheme."""
        for i in prange(2, u.shape[0]-2):
            for j in range(2, u.shape[1]-2):
                if mask[i, j]: continue # skip points inside the cylinder
                # combined upwind terms for advection
                u_adv_dx = (1/12) * ((u[i, j] + abs(u[i, j])) * (2*u[i+1, j] + 3*u[i, j] - 6*u[i-1, j] + u[i-2, j]) \
                    + (u[i, j] - abs(u[i, j])) * (-u[i+2, j] + 6*u[i+1, j] - 3*u[i, j] - 2*u[i-1, j])) * dt/dx
                u_adv_dy = (1/12) * ((v[i, j] + abs(v[i, j])) * (2*u[i, j+1] + 3*u[i, j] - 6*u[i, j-1] + u[i, j-2]) \
                    + (v[i, j] - abs(v[i, j])) * (-u[i, j+2] + 6*u[i, j+1] - 3*u[i, j] - 2*u[i, j-1])) * dt/dy
                v_adv_dx = (1/12) * ((u[i, j] + abs(u[i, j])) * (2*v[i+1, j] + 3*v[i, j] - 6*v[i-1, j] + v[i-2, j]) \
                    + (u[i, j] - abs(u[i, j])) * (-v[i+2, j] + 6*v[i+1, j] - 3*v[i, j] - 2*v[i-1, j])) * dt/dx
                v_adv_dy = (1/12) * ((v[i, j] + abs(v[i, j])) * (2*v[i, j+1] + 3*v[i, j] - 6*v[i, j-1] + v[i, j-2]) \
                    + (v[i, j] - abs(v[i, j])) * (-v[i, j+2] + 6*v[i, j+1] - 3*v[i, j] - 2*v[i, j-1])) * dt/dy
                # combined advection and diffusion
                du = - u_adv_dx - u_adv_dy + nu * dt \
                    * ((u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx**2 \
                    + (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy**2)
                dv = - v_adv_dx - v_adv_dy + nu * dt \
                    * ((v[i+1, j] - 2*v[i, j] + v[i-1, j]) / dx**2 \
                    + (v[i, j+1] - 2*v[i, j] + v[i, j-1]) / dy**2)
                # use Euler or Adams-Bashforth 2 time-stepping based on whether it's the first step or not
                u_next[i, j] = u[i, j] + (1+AB)*du - AB*du_prev[i, j]
                v_next[i, j] = v[i, j] + (1+AB)*dv - AB*dv_prev[i, j]
                du_prev[i, j] = du
                dv_prev[i, j] = dv

    @staticmethod
    @njit(parallel=True, fastmath=False)
    def _predict_FL(u: np.ndarray, v: np.ndarray, u_next: np.ndarray, v_next: np.ndarray,
                    du_prev: np.ndarray, dv_prev: np.ndarray, mask: np.ndarray,
                    nu: float, dt: float, dx: float, dy: float, AB: float = 0.5):
        """Updates the velocity field based on the Navier-Stokes equations using combined upwind scheme. Dynamically switches between first-order and third-order upwind based on whether the third-order points are inside the cylinder or not as an attempt to limit flux instabilities while maintaining higher-order accuracy where possible."""
        for i in prange(2, u.shape[0]-2):
            for j in range(2, u.shape[1]-2):
                if mask[i, j]: continue # skip points inside the cylinder
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
                # combined advection and diffusion
                du = - u_adv_dx - u_adv_dy + nu * dt \
                    * ((u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx**2 \
                    + (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy**2)
                dv = - v_adv_dx - v_adv_dy + nu * dt \
                    * ((v[i+1, j] - 2*v[i, j] + v[i-1, j]) / dx**2 \
                    + (v[i, j+1] - 2*v[i, j] + v[i, j-1]) / dy**2)
                # use Euler or Adams-Bashforth 2 time-stepping based on whether it's the first step or not
                u_next[i, j] = u[i, j] + (1+AB)*du - AB*du_prev[i, j]
                v_next[i, j] = v[i, j] + (1+AB)*dv - AB*dv_prev[i, j]
                du_prev[i, j] = du
                dv_prev[i, j] = dv

    @staticmethod
    @njit(parallel=True, fastmath=FASTMATH)
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

    def benchmark(self, time: float, p_threshold: float = 1e-4, p_max_iters: int = 1000,
                  show: bool = False, save: bool = False, filename: str|None = None):
        import timeit
        # pre-allocate arrays for the solver to avoid repeated allocations inside the loop
        source = np.zeros_like(self.p)
        Ap = np.zeros_like(self.p)
        search_dir = np.zeros_like(self.p)
        dp = np.zeros_like(self.p)
        u_next = self.u.copy()
        v_next = self.v.copy()
        du_prev = np.zeros_like(self.u)
        dv_prev = np.zeros_like(self.v)
        n_iters = int(time / self.dt)
        # precompute constant coefficients for the source term and pressure solver to optimize performance
        half_rho_dx2_dy2_over_dt = 0.5 * (self.rho * self.dx**2 * self.dy**2) / self.dt
        dx2, dy2 = self.dx**2, self.dy**2
        dt_over_2rho_dx = self.dt / (2 * self.rho * self.dx)
        dt_over_2rho_dy = self.dt / (2 * self.rho * self.dy)
        # ----- initial step to time JIT compilation -----
        # perform source once
        start = timeit.default_timer()
        self._source(source, self.u, self.v, self.mask, half_rho_dx2_dy2_over_dt, self.dx, self.dy)
        end = timeit.default_timer()
        print(f"Initial source term computation took {end - start:.4f} seconds.")
        # perform pressure solve once
        start = timeit.default_timer()
        iter_count, _ = self._pressure(self.p, source, self.mask, self.A_diag, self.inv_A_diag,
                                       Ap, search_dir, dx2, dy2, p_threshold, p_max_iters)
        end = timeit.default_timer()
        print(f"Initial pressure solve took {end - start:.4f} seconds with {iter_count} iterations.")
        # perform prediction step to get intermediate velocity field
        start = timeit.default_timer()
        self._predict(self.u, self.v, u_next, v_next, du_prev, dv_prev, self.mask,
                      self.nu, self.dt, self.dx, self.dy, AB=0.0)
        end = timeit.default_timer()
        print(f"Initial prediction step took {end - start:.4f} seconds.")
        start = timeit.default_timer()
        self._correct(u_next, v_next, self.p, self.mask, dt_over_2rho_dx, dt_over_2rho_dy)
        end = timeit.default_timer()
        print(f"Initial correction step took {end - start:.4f} seconds.")
        self.u, u_next = u_next, self.u
        self.v, v_next = v_next, self.v
        # enforce outflow boundary conditions
        self.u[-2:, :] = self.u[-3, :]
        self.v[-2:, :] = self.v[-3, :]
        # compute source term for pressure Poisson equation
        start = timeit.default_timer()
        self._source(source, self.u, self.v, self.mask, half_rho_dx2_dy2_over_dt, self.dx, self.dy)
        end = timeit.default_timer()
        print(f"Initial source term computation took {end - start:.4f} seconds.")
        # solve for pressure and correct velocity field
        dp.fill(0.0) # reset pressure correction array
        start = timeit.default_timer()
        iter_count, _ = self._pressure(dp, source, self.mask, self.A_diag, self.inv_A_diag,
                                       Ap, search_dir, dx2, dy2, p_threshold, p_max_iters)
        end = timeit.default_timer()
        print(f"Initial pressure solve took {end - start:.4f} seconds with {iter_count} iterations.")
        self.p += dp # accumulate pressure corrections for better prediction
        start = timeit.default_timer()
        self._correct(self.u, self.v, dp, self.mask, dt_over_2rho_dx, dt_over_2rho_dy)
        end = timeit.default_timer()
        print(f"Initial correction step took {end - start:.4f} seconds.")
        # enforce outflow boundary conditions again after correction
        self.u[-2:, :] = self.u[-3, :]
        self.v[-2:, :] = self.v[-3, :]
        # check CFL condition for stability
        CFL = (max(np.max(self.u), np.max(self.v)) * self.dt) / min(self.dx, self.dy)
        if CFL >= 1:
            warn(f"CFL condition violated: {CFL:.2f} >= 1. Consider reducing dt or increasing dx/dy for stability.")
        if CFL >= 10:
            raise RuntimeError(f"CFL condition severely violated: {CFL:.2f} >> 1. Simulation is likely diverging. Consider significantly reducing dt or increasing dx/dy for stability.")
        # ----- benchmark loop -----
        times = np.zeros((n_iters-1, 4))
        iters = np.zeros(n_iters-1, dtype=int)
        div_history = np.zeros(n_iters-1)
        for n in range(n_iters-1):
            # perform prediction step to get intermediate velocity field
            start = timeit.default_timer()
            self._predict(self.u, self.v, u_next, v_next, du_prev, dv_prev, self.mask,
                          self.nu, self.dt, self.dx, self.dy)
            end = timeit.default_timer()
            times[n, 0] = end - start
            start = timeit.default_timer()
            self._correct(u_next, v_next, self.p, self.mask, dt_over_2rho_dx, dt_over_2rho_dy)
            end = timeit.default_timer()
            times[n, 3] = end - start
            self.u, u_next = u_next, self.u
            self.v, v_next = v_next, self.v
            # enforce outflow boundary conditions
            self.u[-2:, :] = self.u[-3, :]
            self.v[-2:, :] = self.v[-3, :]
            # compute source term for pressure Poisson equation
            start = timeit.default_timer()
            self._source(source, self.u, self.v, self.mask, half_rho_dx2_dy2_over_dt, self.dx, self.dy)
            end = timeit.default_timer()
            times[n, 1] = end - start
            # solve for pressure and correct velocity field
            dp.fill(0.0) # reset pressure correction array
            start = timeit.default_timer()
            iter_count, _ = self._pressure(dp, source, self.mask, self.A_diag, self.inv_A_diag,
                                           Ap, search_dir, dx2, dy2, p_threshold, p_max_iters)
            end = timeit.default_timer()
            times[n, 2] = end - start
            iters[n] = iter_count
            self.p += dp # accumulate pressure corrections for better prediction
            start = timeit.default_timer()
            self._correct(self.u, self.v, dp, self.mask, dt_over_2rho_dx, dt_over_2rho_dy)
            end = timeit.default_timer()
            times[n, 3] += end - start
            # enforce outflow boundary conditions again after correction
            self.u[-2:, :] = self.u[-3, :]
            self.v[-2:, :] = self.v[-3, :]
            div_history[n] = self._check_divergence(self.u, self.v, self.mask, self.dx, self.dy)
            # check CFL condition for stability
            CFL = (max(np.max(self.u), np.max(self.v)) * self.dt) / min(self.dx, self.dy)
            if CFL >= 1:
                warn(f"CFL condition violated: {CFL:.2f} >= 1. Consider reducing dt or increasing dx/dy for stability.")
            if CFL >= 10:
                raise RuntimeError(f"CFL condition severely violated: {CFL:.2f} >> 1. Simulation is likely diverging. Consider significantly reducing dt or increasing dx/dy for stability.")
        self._benchmark_data(times, iters, div_history, n_iters, p_max_iters, show=show, save=save, filename=filename)

    def _benchmark_data(self, times: np.ndarray, iters: np.ndarray, div_history: np.ndarray, n_iters: int, p_max_iters: int,
                       show: bool = False, save: bool = False, filename: str|None = None):
        # compute average times and print results
        avg_times = np.mean(times, axis=0)
        std_times = np.std(times, axis=0)
        print(f"Average prediction time over         {n_iters-1} iterations: {avg_times[0]:.4f} seconds (std: {std_times[0]:.4f})")
        print(f"Average source term time over        {n_iters-1} iterations: {avg_times[1]:.4f} seconds (std: {std_times[1]:.4f})")
        print(f"Average pressure solve time over     {n_iters-1} iterations: {avg_times[2]:.4f} seconds (std: {std_times[2]:.4f})")
        print(f"Average correction time over         {n_iters-1} iterations: {avg_times[3]:.4f} seconds (std: {std_times[3]:.4f})")
        # print total time and percentage contributions
        total_time = np.sum(times)
        print(f"\nTotal simulation time: {total_time:.4f} seconds")
        print(f"Percentage of time spent on prediction:     {100 * np.sum(times[:, 0]) / total_time:.4f}%")
        print(f"Percentage of time spent on source term:    {100 * np.sum(times[:, 1]) / total_time:.4f}%")
        print(f"Percentage of time spent on pressure solve: {100 * np.sum(times[:, 2]) / total_time:.4f}%")
        print(f"Percentage of time spent on correction:     {100 * np.sum(times[:, 3]) / total_time:.4f}%")
        # print average pressure solver iterations and plot iterations over time
        print(f"Average pressure solver iterations: {np.mean(iters):.2f} (std: {np.std(iters):.2f})")
        print(f"Average divergence over time: {np.mean(div_history):.2e} (std: {np.std(div_history):.2e})")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
        ax[0].scatter(range(n_iters-1), iters, s=10)
        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel("Pressure Solver Iterations")
        ax[0].set_title("Scatter Plot of Pressure Solver Iterations")
        ax[0].set_xticks(np.linspace(0, n_iters-1, 11), np.round(np.linspace(1, n_iters, 11) * self.dt, decimals=3))
        ax[0].set_yscale("log")
        ax[0].set_ylim(1, p_max_iters + 10) # set y-limits to better visualize log scale
        ax[1].plot(range(n_iters-1), div_history)
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("Divergence")
        ax[1].set_title("Divergence Over Time")
        ax[1].set_xticks(np.linspace(0, n_iters-1, 11), np.round(np.linspace(1, n_iters, 11) * self.dt, decimals=3))
        ax[1].set_yscale("log")
        if save and filename is not None:
            plt.savefig(filename, dpi=300)
        if show:
            plt.show()

    @staticmethod
    @njit(parallel=True, fastmath=FASTMATH)
    def _check_divergence(u: np.ndarray, v: np.ndarray, mask: np.ndarray,
                          dx: float, dy: float):
        """Calculates the maximum absolute divergence of the velocity field."""
        max_div = 0.0
        # NOTE: parallel reductions for max() are safe in modern Numba, 
        # but if you get race conditions, remove parallel=True
        for i in prange(2, u.shape[0]-2):
            for j in range(2, u.shape[1]-2):
                if mask[i, j]: continue # ignore boundaries and cylinder
                
                div = 0.5 * (u[i+1, j] - u[i-1, j]) / dx + 0.5 * (v[i, j+1] - v[i, j-1]) / dy
                # Track the worst offending cell in the grid
                max_div = max(max_div, abs(div))
        return max_div