import time
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from netgen.occ import Circle, OCCGeometry, Rectangle, X, Y, Z, gp_Pnt2d
from ngsolve import *  # noqa: F403

from ..utils.config import *  # noqa: F403


class FE:

    def __init__(self, tau=0.0005, nu=0.001, maxh=0.03):
        self.tau = tau # time step
        self.nu = nu # viscosity
        self.maxh = maxh # maximum element size
        self.is_simulated = False

        self.setup_shape()
        self.setup_mesh()
        self.setup_stokes_system()
        self.setup_IBC()
        self.solve_stokes()
        self.setup_time_stepping()

    def setup_shape(self):
        # outer channel
        rect = Rectangle(2.2, 0.41).Face()
        rect.edges.name = "wall"
        rect.edges.Min(X).name = "inlet"
        rect.edges.Max(X).name = "outlet"
        
        # cylinder
        cyl = Circle(gp_Pnt2d(0.2, 0.2), 0.05).Face()
        cyl.edges.name = "cyl"
        cyl.edges.maxh = 0.005  # force high resolution on the boundary layer
        
        self.shape = rect - cyl
        
    def setup_mesh(self):
        self.mesh = Mesh(OCCGeometry(self.shape, dim=2).GenerateMesh(maxh=self.maxh)).Curve(3)
    
    def setup_stokes_system(self):
        # Finite element spaces for velocity vector field (V) and pressure scalar field (Q)
        self.V = VectorH1(self.mesh, order=3, dirichlet="wall|cyl|inlet")
        self.Q = H1(self.mesh, order=2)

        # mixed space to couple V and Q together into a single system
        self.X = self.V*self.Q

        # Get the unknown velocity (vector) field u, and the unknown pressure (scalar) field p
        self.u, self.p = self.X.TrialFunction()

        # Obtain the weighting functions/variations (v and q)
        self.v, self.q = self.X.TestFunction()

        # This is the weak form of stokes-flow, which models very slow flowing
        # objects (Re << 1), where viscous forces dominate inertial forces
        # I have derived the weak form and it checks out.
        self.stokes = (self.nu*InnerProduct(grad(self.u), grad(self.v))+ \
            div(self.u)*self.q+div(self.v)*self.p - 1e-10*self.p*self.q)*dx

        # create the matrix
        self.A = BilinearForm(self.stokes).Assemble()

        # This creates the b matrix which only depends on the test function v. It is empty, since
        # we dont have any external forces being applied.
        self.b = LinearForm(self.X).Assemble()

        # this is the container that holds the actual numerical values of the solution.
        self.x = GridFunction(self.X)
    
    def setup_IBC(self):
        # parabolic inlet flow in x-direction, in line with DFG 2D-2 Benchmark
        Um = 1.5
        H = 0.41
        uin_x = 4 * Um * y * (H - y) / (H**2)
        uin = CoefficientFunction((uin_x, 0))

        # set the velocity components of the mesh labeled as the inlet to
        # be forced to this boundary condition.
        self.x.components[0].Set(uin, definedon=self.mesh.Boundaries("inlet"))

    def solve_stokes(self):
        # invert the global A matrix which represents the bilinear form of the
        # stokes equation. FreeDofs() makes sure that we dont solve for the boundaries
        # that are already defined
        inv_stokes = self.A.mat.Inverse(self.X.FreeDofs())

        # calc residual (r = b - Ax) which tells us how far is the "current guess" away
        # from a valid physical solution.
        res = self.b.vec - self.A.mat*self.x.vec

        # add the correction (x = A^{-1}b, and, b = r + Ax -> x = A^{-1}(r + Ax) -> x = A^{-1}r + x)
        # thus x_final = x_init + A^{-1}r
        self.x.vec.data += inv_stokes * res
    
    def setup_time_stepping(self):
        """
        Prepare matrices and operators for time integration of the Navier-Stokes equations.
        """

        # Create the main system matrix for each time step. This first variable basically
        # extends the time-independent Stokes Flow equation, and just adds transient behaviour
        # meaning that it combines the mass term (how velocity changes over time) with the physics 
        # (viscosity and pressure constraints).
        self.mstar = BilinearForm(self.u * self.v * dx + self.tau * self.stokes).Assemble()

        # Pre-compute/inverse the matrix, since it never changes
        # 'FreeDofs' ensures we respect the boundary conditions (walls, inlet).
        self.inv = self.mstar.mat.Inverse(self.X.FreeDofs(), inverse="sparsecholesky")

        # Only now do we setup the convection term ((u·∇)u - non-linear) which depends on the
        # current flow field. This can therefore not be precomputed, since it changes every time
        # step.
        self.conv = BilinearForm(self.X, nonassemble=True)
        self.conv += (grad(self.u) * self.u) * self.v * dx
    
    def run(self, t_end = 10, sample_freq=10):
        """
        Simulate the fluid flow from t=0 to t=t_end using semi-implicit time stepping.
        """
        self.is_simulated = False
        t = 0
        i = 0

        # create history
        self.x_hist = GridFunction(self.X, multidim=0)

        # save initial state as first animation frame
        self.vel_hist = GridFunction(self.V, multidim=0)
        self.vel_hist.AddMultiDimComponent(self.x.components[0].vec)  # initial frame

        # probe point for vortex shedding
        self.probe_point = self.mesh(0.4, 0.25)
        self.t_hist = []
        self.vy_hist = []

        with TaskManager():  # Handle parallelization
            while t < t_end:
                
                # Build the right-hand side for this time step.
                # add the non-linear convection term (u·∇)u
                res = self.conv.Apply(self.x.vec)
                # add the pressure from the previous step
                res += self.A.mat*self.x.vec
                
                # Solve for the new flow state using the pre-computed inverse matrix.
                self.x.vec.data -= self.tau * self.inv * res  
                
                # Advance time by one time step
                t = t + self.tau
                i += 1

                # save data for probe point
                self.t_hist.append(t)
                self.vy_hist.append(self.x.components[0](self.probe_point)[1])

                # store frame every sample_every steps
                if (i % sample_freq == 0) or (t >= t_end):
                    self.vel_hist.AddMultiDimComponent(self.x.components[0].vec)
        
        self.is_simulated = True
    
    def draw_sim(self):
        if not self.is_simulated:
            print("Simulation not finished yet. Call run(...) first.")
            return
    
        import netgen.gui  # initialize native GUI only when needed

        playback_gf = GridFunction(self.V)
        
        # Draw the speed magnitude (Norm).
        Draw(Norm(playback_gf), self.mesh, "velocity_mag", autoscale=True)

        print("\n--- Playback Controls ---")
        input("1. Press Enter here in the terminal to start the animation...")
        
        # Loop forward through the saved history
        for frame_vec in self.vel_hist.vecs:
            playback_gf.vec.data = frame_vec
            Redraw()
            time.sleep(0.05)  # 20 frames per second
            
        print("Playback finished.")

    def reynolds_number(self):
        return (1.0 * 2 * 0.05) / self.nu

    def calc_divergence_norm(self):
        return sqrt(Integrate(div(self.x.components[0])**2, self.mesh))

    def get_strouhal_number(self, D=0.1, U=1.0):
        # ignore startup transient. Only sample after t=5.
        steady_start_idx = int(5.0 / self.tau)
        
        # failsafe if the simulation crashed early
        if len(self.vy_hist) <= steady_start_idx:
            return 0.0 
            
        # Slice the array to only look at steady shedding
        vy = np.array(self.vy_hist)[steady_start_idx:]

        # center signal at 0, otherwise fft will find
        # 0hz as max signal
        vy = vy - np.mean(vy)
        
        n = len(vy)
        freqs = np.fft.rfftfreq(n, d=self.tau)
        fft_values = np.abs(np.fft.rfft(vy))
        
        peak_idx = np.argmax(fft_values)
        f = freqs[peak_idx]
        
        St = f * D / U
        return St

if __name__ == "__main__":
    NS = FE()
    print(f"Re = {NS.reynolds_number()}")
    NS.run()
    NS.draw_sim()