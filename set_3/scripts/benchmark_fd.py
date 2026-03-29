from argparse import ArgumentParser
from ..utils.config import *  # noqa: F403
from ..utils.FD import FD

def parse_args():
    parser = ArgumentParser(description="Run the finite difference solver for the 2D Navier-Stokes equations.")
    sim = parser.add_argument_group(title="Simulation Parameters")
    sim.add_argument("--dx", type=float, default=0.005, help="Grid spacing in x direction")
    sim.add_argument("--dy", type=float, default=0.005, help="Grid spacing in y direction")
    sim.add_argument("--dt", type=float, default=0.001, help="Time step size")
    sim.add_argument("--rho", type=float, default=1.0, help="Fluid density")
    sim.add_argument("--nu", type=float, default=0.001, help="Kinematic viscosity")
    solve = parser.add_argument_group(title="Solver Parameters")
    solve.add_argument("--time", type=float, default=1.0, help="Total simulation time")
    solve.add_argument("--p_threshold", "-eps", type=float, default=1e-4, help="Pressure solver convergence threshold")
    solve.add_argument("--p_max_iters", "-max", type=int, default=5000, help="Maximum iterations for pressure solver")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(f"Simulation parameters: dx={args.dx}, dy={args.dy}, dt={args.dt}, rho={args.rho}, nu={args.nu}")
    print(f"Solver parameters: time={args.time}s, p_threshold={args.p_threshold}, p_max_iters={args.p_max_iters}")
    print("Benchmarking solver...")
    fd = FD(dx=args.dx, dy=args.dy, dt=args.dt, rho=args.rho, nu=args.nu)
    filename = f"set_3/results/iterations.png"
    fd.benchmark(time=args.time, p_threshold=args.p_threshold, p_max_iters=args.p_max_iters,
                    show=False, save=True, filename=filename)
    filename = f"set_3/results/fields.png"
    fd.plot(show=False, save=True, filename=filename)
    print("")