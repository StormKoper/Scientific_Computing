import numpy as np
from numba import njit, prange


class GeneralRD():
    def __init__(self, N: int, delta_t: float, delta_x: float, consts: dict, seed: int = 42):
        
        # unpack arguments
        self.N = N
        self.dx = delta_x
        self.dt = delta_t
        for key, value in consts.items():
            if type(value) is not float:
                raise ValueError (f"Constant {key} should be of type float, got {type(value)}")
            setattr(self, key, value)
        self.gen = np.random.default_rng(seed)
        
        # set additional attributies
        self._setup_grid()
        self.iter_count = 0
        self._frames = []
        self.grid_hist = None

    def _setup_grid(self):
        """Create an NxN structured array.

        The dtypes of the grid are given by the 'D*' constants. Practically
        each cell in the NxN grid contains multiple named floats denoting the
        different concentrations.
        """
        conc = []
        for attr in dir(self):
            if attr.startswith("D"):
                conc.append(attr[1:])
        
        self.grid = np.zeros(shape=(self.N, self.N), dtype=[(c, "float64") for c in conc])

        # define next_grid for update_func to change in place
        self.next_grid = self.grid.copy()

    def _update_func(self):
        raise(NotImplementedError)

    def _step(self):
        """Take a single step, swap grid references, and append frame."""
        self.iter_count += 1
        self._update_func()

        # swap references for next iteration
        self.grid, self.next_grid = self.next_grid, self.grid

        # add grid to frames
        self._frames.append(self.grid.astype([(name, "float16") for name in self.grid.dtype.names]).copy()) # type: ignore

    def run(self, n_iters: int):
        """Run the simulation for a given number of iterations.
        
        Args:
            - n_iters (int): Number of iterations to run.
            
        """
        if n_iters is not None:
            for _ in range(n_iters):
                self._step()
        self.grid_hist = np.stack(self._frames, axis=-1)

class GrayScott(GeneralRD):

    def __init__(self, N: int, delta_t: float, delta_x: float, consts: dict, seed: int = 42):
        self._check_req_constants(consts)
        super().__init__(N, delta_t, delta_x, consts, seed=seed)

    def _check_req_constants(self, consts):
        """Check that only the required attributes are given."""
        req = {"Du", "Dv", "f", "k"}
        given = set(consts.keys())

        missing = req - given
        if missing:
            raise ValueError(f"Missing Attribute(s) for Gray-Scott model: {', '.join(missing)}")
        
        extra = given - req
        if extra:
            raise ValueError(f"The following attributes were given, but shouldn't: {', '.join(extra)}")

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _jit_update(grid: np.ndarray, next_grid: np.ndarray, dt: float, dx: float, 
                    Du: float, Dv: float, f: float, k: float) -> None:
        """Calculate and apply one Gray-Scott update step.
        
        The integration step is Forward-Time Central-Space (FTCS).
        
        Args:
            - grid (np.ndarray): NxN structured array containing current u and
                v concentrations.
            - next_grid (np.ndarray): NxN structured array in which the updated
                values will be assigned.
            - dt (float): size of time step.
            - dx (float): size of space step.
            - Du (float): diffusion coefficient of chemical u.
            - Dv (float): diffusion coefficient of chemical v.
            - f (float): denotes the rate at which u is supplied.
            - k (float): k (+ f) denotes the rate at which v decays.
        
        """
        # unpack grid for convenience
        u, v = grid['u'], grid['v']
        u_n, v_n = next_grid['u'], next_grid['v']

        # update concentrations
        rows, cols = u.shape
        for i in prange(rows):
            for j in prange(cols):
                laplace_u = (u[(i+1)%rows, j] + u[(i-1)%rows, j] + u[i, (j+1)%cols] + u[i, (j-1)%cols] - 4*u[i, j]) / dx**2
                u_n[i, j] = u[i, j] + dt * ( Du * laplace_u - (u[i, j] * v[i, j]**2) + f*(1 - u[i, j]) )
        
        for i in prange(rows):
            for j in prange(cols):
                laplace_v = (v[(i+1)%rows, j] + v[(i-1)%rows, j] + v[i, (j+1)%cols] + v[i, (j-1)%cols] - 4*v[i, j]) / dx**2
                v_n[i, j] = v[i, j] + dt * ( Dv * laplace_v + (u[i, j] * v[i, j]**2) - (f + k) * v[i, j] )

    def _update_func(self):
        """Small wrapper to assign object attributes to njit update func."""
        self._jit_update(self.grid, self.next_grid, self.dt, self.dx, 
                   self.Du, self.Dv, self.f, self.k) # type: ignore

if __name__ == "__main__":
    pass
