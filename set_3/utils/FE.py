import time

from netgen.occ import OCCGeometry, Rectangle, X, Y, Z
from ngsolve import (
    H1,
    BilinearForm,
    CoefficientFunction,
    Draw,
    GridFunction,
    InnerProduct,
    LinearForm,
    Mesh,
    Norm,
    Redraw,
    SetNumThreads,
    TaskManager,
    VectorH1,
    div,
    dx,
    grad,
    y,
)


class FE:

    def __init__(self, tau=0.001, nu=0.001):
        self.tau = tau # time step
        self.nu = nu # viscosity
        self.is_simulated = False

        self.setup_shape()
        self.setup_mesh()
        self.setup_stokes_system()
        self.setup_IBC()
        self.solve_stokes()
        self.setup_time_stepping()

    
    def setup_shape(self):
        shape = Rectangle(2.2,0.41).Circle(0.2,0.2,0.05).Reverse().Face()
        shape.edges.name="wall"
        shape.edges.Min(X).name="inlet"
        shape.edges.Max(X).name="outlet"
        self.shape = shape
    
    def draw_shape(self):
        if self.shape:
            Draw (self.shape)
        
    def setup_mesh(self):
        self.mesh = Mesh(OCCGeometry(self.shape, dim=2).GenerateMesh(maxh=0.07)).Curve(3)
    
    def draw_mesh(self):
        if self.mesh:
            Draw (self.mesh)
    
    def setup_stokes_system(self):
        # Finite element spaces for velocity vector field (V) and pressure scalar field (Q)
        
        # V represents a vector space, since velocity has both magnitude
        # and direction. order=3 means the field is approximated on each element 
        # with a cubic polynomial. dirichlet="wall|cyl|inlet" marks these
        # boundaries for Dirichlet conditions.
        self.V = VectorH1(self.mesh, order=3, dirichlet="wall|cyl|inlet")

        # Q represents a scalar space, since pressure is a scalar. order=2 means 
        # the field is approximated on each element with a quadratic polynomial.
        self.Q = H1(self.mesh, order=2)

        # Mixed space to couple V and Q together into a single system
        self.X = self.V*self.Q

        # Get the unknown velocity (vector) field u, and the unknown pressure (scalar) field p
        self.u, self.p = self.X.TrialFunction()

        # Obtain the weighting functions/variations (v and q). These act as symbolic
        # placeholders for the test functions when assembling the weak form.
        self.v, self.q = self.X.TestFunction()

        # I dont understand jack shit of this, but apparently this is the Navier-Stokes
        # equation, but then without the non-linear (u * nabla)u term and the time derivative.
        # As far as I know it is beneficial to solve this easier equation first, and then combine it
        # later
        self.stokes = (self.nu*InnerProduct(grad(self.u), grad(self.v))+ \
            div(self.u)*self.q+div(self.v)*self.p - 1e-10*self.p*self.q)*dx

        # This creates the A matrix that we have to "invert" which depends both on the trial function
        # u and the test function v
        self.A = BilinearForm(self.stokes).Assemble()

        # This creates the b matrix which only depends on the test function v. It is empty, since
        # we dont have any external forces being applied.
        self.b = LinearForm(self.X).Assemble()

        # this is the container that holds the actual numerical values of the solution.
        self.x = GridFunction(self.X)
    
    def setup_IBC(self):
        # setup the Inlet Boundary Condition.

        # uin defines exactly how the fluid should move as it enters the domain. It is defined
        # as a parabola where the velocity is 0 at y=0 and y=0.41 (bottom and top wall) and in
        # the middle reaches it maximum. It is done this way, since fluid at the walls should
        # move slower due to friction. Notice also (formula, 0), which tells the solver that
        # the fluid is moving in the x-direction.
        uin = CoefficientFunction( (1.5*4*y*(0.41-y)/(0.41*0.41), 0) )

        # set the 0 components (velocity, not pressure) of the mesh labeled as the inlet to
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
        # (viscosity and pressure constraints). We pre-compute this since it never changes.
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

        with TaskManager():  # Handle parallelization/parallelization-related tasks
            while t < t_end:
                
                # Build the right-hand side for this time step.
                # add the non-linear convection term (u·∇)u
                res = self.conv.Apply(self.x.vec)
                # add the pressure from the previous step
                res += self.A.mat*self.x.vec
                
                # Solve for the new flow state using the pre-computed inverse matrix.
                # This single line does the actual time step: x_new = x_old - tau * inv(M*) * res
                # where tau is the time step size, inv(M*) was pre-computed in setup_time_stepping(),
                # and res is the residual above. This is the "implicit-in-diffusion, explicit-in-convection" scheme.
                self.x.vec.data -= self.tau * self.inv * res  
                
                # Advance time by one time step
                t = t + self.tau
                i += 1

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
        
        # Draw the speed magnitude (Norm). This turns it into a CoefficientFunction, 
        # which means it REQUIRES self.mesh as the second argument.
        Draw(Norm(playback_gf), self.mesh, "velocity_mag", autoscale=True)

        print("\n--- Playback Controls ---")
        input("1. Press Enter here in the terminal to start the animation...")
        
        # Loop forward through the saved history
        for frame_vec in self.vel_hist.vecs:
            playback_gf.vec.data = frame_vec
            Redraw()
            time.sleep(0.05)  # 20 frames per second
            
        print("Playback finished.")


if __name__ == "__main__":
    NS = FE()
    NS.run(t_end=10, sample_freq=50)
    NS.draw_sim()
    input("Press Enter to close...")