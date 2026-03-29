import numpy as np
from numba import njit, prange
from tqdm import tqdm

FASTMATH = True

# D2Q9 lattice parameters 
C = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

class LB:
    def __init__(self, dx: float = .01, dt: float = .001, rho: float = 1.0, nu: float = .1):
        dy = dx # square lattice
        self.dx = dx
        self.dy = dy
        self.dt = dt
        pad = 1 # for boundary conditions
        self.nx = round(2.2 / dx + 2*pad) + 1
        self.ny = round(0.41 / dy + 2*pad) + 1
        nu_lb = nu * dt / (dx**2) # convert to lattice units
        tau = 3.0 * nu_lb + 0.5
        self.omega = 1.0 / tau

        # initialize velocity and density fields
        self.u = np.zeros((self.nx, self.ny))
        self.v = np.zeros((self.nx, self.ny))
        self.rho = np.full((self.nx, self.ny), rho)

        # set up the inflow and cylinder
        x = np.pad(np.linspace(0, 2.2, self.nx-2*pad), (pad, pad), mode='edge')
        y = np.pad(np.linspace(0, 0.41, self.ny-2*pad), (pad, pad), mode='edge')
        # parabolic inflow from the left (in line with DFG 2D-2 Benchmark)
        Um = 1.5
        H = 0.41
        u_in = 4 * Um * y * (H - y) / (H**2)
        self.u_in = u_in * dt / dx # scale by dt/dx for consistency with LBM units
        self.u[0, :] = self.u_in
        # cylinder mask for no-slip condition
        cx, cy, r = 0.2, 0.2, 0.05
        X, Y = np.meshgrid(x, y, indexing='ij')
        self.mask = (X - cx)**2 + (Y - cy)**2 <= r**2
        self.u[self.mask] = 0.0
        self.v[self.mask] = 0.0
        # add walls to the mask
        self.mask[:, 0] = True  # bottom boundary
        self.mask[:, -1] = True # top boundary

        # compute initial f based on these fields
        self.f = np.zeros((self.nx, self.ny, 9))
        self._equilibrium(self.f, self.rho, self.u, self.v)
        self.f_inlet = self.f[0, :, :].copy()
        self.f_new = self.f.copy()

        # calculate Reynolds number for reference
        self.Re = 1.0 * 2 * r / nu # Re = U*L/nu with L=diameter=2r and U=2/3 Um = 1.0
        # set up probes for calculating Strouhal number
        self.probes_x = [int((cx+5*r)/dx)+pad,   int((cx+5*r)/dx)+pad,   int((cx+7*r)/dx)+pad,   int((cx+7*r)/dx)+pad]
        self.probes_y = [int((cy+0.5*r)/dy)+pad, int((cy-0.5*r)/dy)+pad, int((cy+0.5*r)/dy)+pad, int((cy-0.5*r)/dy)+pad]
        self.strouhal_data = {"u": [], "v": []}

    def run(self, time: float, probe: int = 0):
        """Runs the simulation for a given time."""
        n_steps = int(time / self.dt)
        for n in tqdm(range(n_steps), desc="Running simulation", leave=False):
            self._lbm_step(self.f, self.f_new, self.rho, self.u, self.v, self.mask, self.omega)
            # swap references for next iteration
            self.f, self.f_new = self.f_new, self.f
            # enforce boundary conditions
            self.f[0, :, :] = self.f_inlet # Dirichlet inlet
            self.f[-1, :, :] = self.f[-2, :, :] # Neumann outlet
            # record probe data for Strouhal number estimation
            if probe and n % probe == 0:
                self.strouhal_data["u"].extend(self.u[self.probes_x, self.probes_y])
                self.strouhal_data["v"].extend(self.v[self.probes_x, self.probes_y])

    def plot(self, show=True, save=False, filename=None):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(14, 0.41*16/2.2), constrained_layout=True)
        x_coords = np.linspace(0, 2.2, self.nx-4)
        y_coords = np.linspace(0, 0.41, self.ny-4)
        X, Y = np.meshgrid(x_coords, y_coords)
        u = self.u * self.dx / self.dt # convert back to physical units for plotting
        v = self.v * self.dy / self.dt # convert back to physical units for plotting
        pressure = (self.rho - 1.0) * (self.dx / self.dt)**2 / 3.0 # convert density to pressure
        pressure = np.ma.masked_array(pressure[1:-1, 1:-1], mask=self.mask[1:-1, 1:-1])
        cont = ax.contourf(X, Y, pressure.T, levels=50, cmap='viridis')
        fig.colorbar(mappable=cont, label='Pressure')
        ax.streamplot(X, Y, u.T[1:-1, 1:-1], v.T[1:-1, 1:-1], color='white')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f"Pressure and Velocity Field: $Re={self.Re:.0f}$ \
                     $\\Delta t={self.dt:.4f}$, $\\Delta x={self.dx:.4f}$, $\\Delta y={self.dy:.4f}$")
        ax.set_box_aspect(0.41/2.2)
        if save and filename is not None: plt.savefig(filename, dpi=300)
        if show: plt.show()

    def strouhal(self, probe: int, show=False, save=False, filename=None) -> float:
        probes = ["Top left", "Bottom left", "Top right", "Bottom right"]
        u_data = np.array(self.strouhal_data["u"]).reshape(-1, 4) * self.dx / self.dt
        v_data = np.array(self.strouhal_data["v"]).reshape(-1, 4) * self.dy / self.dt
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
        if show or save:
            import matplotlib.pyplot as plt
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
        return np.mean(strouhals)

    @staticmethod
    @njit(parallel=True, fastmath=FASTMATH)
    def _equilibrium(feq, rho, u, v):
        for i in prange(feq.shape[0]):
            for j in range(feq.shape[1]):
                uv_sq = u[i, j]**2 + v[i, j]**2
                for k in range(9):
                    cuv = C[k, 0] * u[i, j] + C[k, 1] * v[i, j]
                    feq[i, j, k] = W[k] * rho[i, j] * (1.0 + 3.0 * cuv + 4.5 * cuv**2 - 1.5 * uv_sq)

    @staticmethod
    @njit(parallel=True, fastmath=FASTMATH)
    def _lbm_step(f, f_new, rho, u, v, mask, omega):
        nx = f.shape[0]
        ny = f.shape[1]
        for i in prange(0, nx):
            for j in range(0, ny):
                if mask[i, j]: # cylinder or wall node
                    for k in range(9):
                        # bounce-back: reflect distribution back to opposite direction
                        f_out = f[i, j, OPP[k]]
                        
                        next_i = i + C[k, 0]
                        next_j = j + C[k, 1]
                        if 0 <= next_i < nx and 0 <= next_j < ny:
                            f_new[next_i, next_j, k] = f_out

                else: # fluid node
                    # compute macroscopic properties locally
                    rho_loc = 0.0
                    u_loc = 0.0
                    v_loc = 0.0
                    for k in range(9):
                        val = f[i, j, k]
                        rho_loc += val
                        u_loc += val * C[k, 0]
                        v_loc += val * C[k, 1]
                    
                    # avoid division by zero and ensure physical consistency
                    if rho_loc > 1e-14:
                        u_loc /= rho_loc
                        v_loc /= rho_loc
                    else:
                        rho_loc = 1.0
                        u_loc = 0.0
                        v_loc = 0.0
                        
                    # store them directly to the main arrays
                    rho[i, j] = rho_loc
                    u[i, j] = u_loc
                    v[i, j] = v_loc

                    # equilibrium, collision, and streaming in one step
                    uv_sq = u_loc**2 + v_loc**2
                    for k in range(9):
                        cuv = C[k, 0] * u_loc + C[k, 1] * v_loc
                        feq = W[k] * rho_loc * (1.0 + 3.0 * cuv + 4.5 * cuv**2 - 1.5 * uv_sq)
                        
                        f_out = (1.0 - omega) * f[i, j, k] + omega * feq
                        
                        # push directly to the neighbor in f_new
                        next_i = i + C[k, 0]
                        next_j = j + C[k, 1]
                        if 0 <= next_i < nx and 0 <= next_j < ny:
                            f_new[next_i, next_j, k] = f_out

    @staticmethod
    @njit(parallel=True, fastmath=FASTMATH)
    def max_divergence(u, v, mask, dt):
        max_div = 0.0
        for i in prange(1, u.shape[0] - 1):
            for j in range(1, u.shape[1] - 1):
                # ignore boundaries and cylinder and immediately adjacent points
                if mask[i, j] or mask[i+1, j] or mask[i-1, j] or mask[i, j+1] or mask[i, j-1]: continue
                div = 0.5 * (u[i+1, j] - u[i-1, j] + v[i, j+1] - v[i, j-1]) / dt # in physical units
                max_div = max(max_div, abs(div))
        return max_div