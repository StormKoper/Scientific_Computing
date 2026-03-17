from argparse import ArgumentParser
from ..utils.config import *  # noqa: F403
from ..utils.FD import benchmark

def parse_args():
    parser = ArgumentParser(description="Run the finite difference solver for the 2D Navier-Stokes equations.")
    sim = parser.add_argument_group(title="Simulation Parameters")
    sim.add_argument("--dx", type=float, default=0.005, help="Grid spacing in x direction")
    sim.add_argument("--dy", type=float, default=0.005, help="Grid spacing in y direction")
    sim.add_argument("--dt", type=float, default=0.001, help="Time step size")
    sim.add_argument("--rho", type=float, default=1.0, help="Fluid density")
    sim.add_argument("--nu", type=float, default=0.001, help="Kinematic viscosity")
    sim.add_argument("--omega", type=float, default=1.5, help="Successive over-relaxation factor for pressure solver")
    solve = parser.add_argument_group(title="Solver Parameters")
    solve.add_argument("--time", type=float, default=1.0, help="Total simulation time")
    solve.add_argument("--P", type=int, nargs="+", default=[2, 3], choices=[2, 3], help="Pressure solver order(s) (2 or 3)")
    solve.add_argument("--p_threshold", "-eps", type=float, default=1e-4, help="Pressure solver convergence threshold")
    solve.add_argument("--p_max_iters", "-max", type=int, default=5000, help="Maximum iterations for pressure solver")
    solve.add_argument("--u0", type=float, default=0.0, help="Initial velocity for benchmarking (default: 1.0)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(f"Running benchmarks with parameters: dx={args.dx}, dy={args.dy}, dt={args.dt}, rho={args.rho}, nu={args.nu}")
    print(f"Total simulation time: {args.time} seconds, velocity initialization: {args.u0}")
    print(f"Pressure solver parameters: omega={args.omega}, p_threshold={args.p_threshold}, p_max_iters={args.p_max_iters}")
    if 2 in args.P:
        print("Benchmarking P2 solver...")
        fd = benchmark(dx=args.dx, dy=args.dy, dt=args.dt, rho=args.rho, nu=args.nu, omega=args.omega, u0=args.u0)
        filename = f"set_3/results/iterations_P2_u{args.u0}.png"
        fd.benchmark_P2(time=args.time, limit_flux=False, p_threshold=args.p_threshold, p_max_iters=args.p_max_iters,
                        show=False, save=True, filename=filename)
        filename = f"set_3/results/fields_P2_u{args.u0}.png"
        fd.plot(show=False, save=True, filename=filename)
        print("")
    if 3 in args.P:
        print("Benchmarking P3 solver...")
        fd = benchmark(dx=args.dx, dy=args.dy, dt=args.dt, rho=args.rho, nu=args.nu, omega=args.omega, u0=args.u0)
        filename = f"set_3/results/iterations_P3_u{args.u0}.png"
        fd.benchmark_P3(time=args.time, limit_flux=False, p_threshold=args.p_threshold, p_max_iters=args.p_max_iters,
                        show=False, save=True, filename=filename)
        filename = f"set_3/results/fields_P3_u{args.u0}.png"
        fd.plot(show=False, save=True, filename=filename)