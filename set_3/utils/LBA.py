import numpy as np
from numba import njit

# D2Q9 lattice parameters 
C = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

@njit
def compute_macroscopic(f):
    Nx, Ny, _ = f.shape
    rho_eps = 1e-14
    rho = np.zeros((Nx, Ny))
    ux = np.zeros((Nx, Ny))
    uy = np.zeros((Nx, Ny))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(9):
                rho[i, j] += f[i, j, k]
                ux[i, j] += f[i, j, k] * C[k, 0]
                uy[i, j] += f[i, j, k] * C[k, 1]
            if np.isfinite(rho[i, j]) and rho[i, j] > rho_eps:
                ux[i, j] /= rho[i, j]
                uy[i, j] /= rho[i, j]
            else:
                rho[i, j] = 1.0
                ux[i, j] = 0.0
                uy[i, j] = 0.0
    return rho, ux, uy

@njit
def equilibrium(rho, ux, uy):
    Nx, Ny = rho.shape
    feq = np.zeros((Nx, Ny, 9))
    for i in range(Nx):
        for j in range(Ny):
            u_sqr = ux[i, j]**2 + uy[i, j]**2
            for k in range(9):
                cu = C[k, 0] * ux[i, j] + C[k, 1] * uy[i, j]
                feq[i, j, k] = W[k] * rho[i, j] * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u_sqr)
    return feq

@njit
def lbm_step(f, obstacle, tau, U_inlet):
    Nx, Ny, _ = f.shape

    rho, ux, uy = compute_macroscopic(f)

    feq = equilibrium(rho, ux, uy)
    f_out = f - (f - feq) / tau

    for i in range(Nx):
        for j in range(Ny):
            if obstacle[i, j]:
                for k in range(9):
                    f_out[i, j, k] = f[i, j, OPP[k]]

    f_new = np.zeros_like(f)
    for i in range(Nx):
        for j in range(Ny):
            for k in range(9):
                next_i = i + C[k, 0]
                next_j = (j + C[k, 1]) % Ny
                if 0 <= next_i < Nx:
                    f_new[next_i, next_j, k] = f_out[i, j, k]

    f = f_new

    # Zou-He velocity inlet (left boundary): ux = U_inlet, uy = 0
    for j in range(Ny):
        rho_in = (
            f[0, j, 0] + f[0, j, 2] + f[0, j, 4] + 
            2.0 * (f[0, j, 3] + f[0, j, 6] + f[0, j, 7])
        ) / (1.0 - U_inlet)

        f[0, j, 1] = f[0, j, 3] + (2.0/3.0) * rho_in * U_inlet
        f[0, j, 5] = f[0, j, 7] - 0.5 * (f[0, j, 2] - f[0, j, 4]) + (1.0/6.0) * rho_in * U_inlet
        f[0, j, 8] = f[0, j, 6] + 0.5 * (f[0, j, 2] - f[0, j, 4]) + (1.0/6.0) * rho_in * U_inlet

    # Zou-He velocity outlet (right boundary): ux extrapolated, uy = 0
    for j in range(Ny):
        ux_out = ux[Nx - 2, j]
        if ux_out > 0.25:
            ux_out = 0.25
        elif ux_out < -0.25:
            ux_out = -0.25

        denom = 1.0 + ux_out
        if denom < 1e-8:
            denom = 1e-8

        rho_out = (
            f[Nx - 1, j, 0] + f[Nx - 1, j, 2] + f[Nx - 1, j, 4]
            + 2.0 * (f[Nx - 1, j, 1] + f[Nx - 1, j, 5] + f[Nx - 1, j, 8])
        ) / denom

        f[Nx - 1, j, 3] = f[Nx - 1, j, 1] - (2.0 / 3.0) * rho_out * ux_out
        f[Nx - 1, j, 6] = f[Nx - 1, j, 8] + 0.5 * (f[Nx - 1, j, 4] - f[Nx - 1, j, 2]) - (1.0 / 6.0) * rho_out * ux_out
        f[Nx - 1, j, 7] = f[Nx - 1, j, 5] + 0.5 * (f[Nx - 1, j, 2] - f[Nx - 1, j, 4]) - (1.0 / 6.0) * rho_out * ux_out

    rho, ux, uy = compute_macroscopic(f)
    for i in range(Nx):
        for j in range(Ny):
            if obstacle[i, j]:
                ux[i, j] = 0.0
                uy[i, j] = 0.0

    return f, rho, ux, uy

class LBA:
    def __init__(self, Nx: int = 300, Ny: int = 120, U_inlet: float = 0.12, Re: float = 150):
        self.Nx = Nx
        self.Ny = Ny
        self.U_inlet = U_inlet
        self.Re = Re

        self.cx_cyl = self.Nx // 5
        self.cy_cyl = self.Ny // 2
        self.r_cyl = 8

        self.D = 2 * self.r_cyl
        self.nu = self.U_inlet * self.D / self.Re
        self.tau = 3.0 * self.nu + 0.5

        # Grid and obstacle
        x, y = np.arange(self.Nx), np.arange(self.Ny)
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        self.obstacle = (self.X - self.cx_cyl)**2 + (self.Y - self.cy_cyl)**2 <= self.r_cyl**2
        
        # Initial conditions
        rho = np.ones((self.Nx, self.Ny))
        ux = np.full((self.Nx, self.Ny), self.U_inlet)
        uy = 0.001 * self.U_inlet * np.sin(2.0 * np.pi * self.Y / self.Ny)
        
        ux[self.obstacle] = 0.0
        uy[self.obstacle] = 0.0
        
        # Calculate initial f
        self.f = equilibrium(rho, ux, uy)
        self.rho = rho
        self.ux = ux
        self.uy = uy

    def step(self):
        """Advances the simulation by one timestep using Numba JIT compiled functions."""
        self.f, self.rho, self.ux, self.uy = lbm_step(self.f, self.obstacle, self.tau, self.U_inlet)

    def run(self, n_steps: int = 30000):
        """Runs the simulation for a given number of steps without plotting."""
        for step in range(1, n_steps + 1):
            self.step()
            if step % 1000 == 0:
                print(f"Step {step}/{n_steps}")