from argparse import ArgumentParser
from datetime import datetime
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
    solve.add_argument("--time", type=float, default=5.0, help="Total simulation time")
    solve.add_argument("--P", type=int, default=2, choices=[0, 1, 2], help="Pressure solver order (0, 1, or 2)")
    solve.add_argument("--order", type=int, default=3, choices=[0, 1, 3], help="Advection term upwind order (0, 1, or 3)")
    output = parser.add_argument_group(title="Output Options")
    output.add_argument("--warmup", type=float, default=0, help="Time for warm-up simulation (0 to disable)")
    output.add_argument("--probe", type=int, default=0, help="Strouhal number probe data collection frequency (0 to disable)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    fd = FD(dx=args.dx, dy=args.dy, dt=args.dt, rho=args.rho, nu=args.nu)
    if args.warmup:
        print(f"Running warm-up simulation for {args.warmup} seconds...")
        fd.run(time=args.warmup, P=args.P, order=args.order)
    print(f"Running main simulation for {args.time} seconds...")
    fd.run(time=args.time, P=args.P, order=args.order, probe=args.probe)
    filename = f"FD_P{args.P}_order{args.order}_{str(datetime.now()).replace(' ', '_').replace(':', '-')}.png"
    print("Plotting the velocity and pressure fields...")
    fd.plot(show=True, save=True, filename="set_3/results/" + filename)
    if args.probe:
        print("Plotting Strouhal probe data...")
        fd.plot_strouhal(probe=args.probe, show=True, save=True, filename="set_3/results/Strouhal_" + filename)

# uv run -m set_3.scripts.run_fd --time 1 --nu 0.0006666666 --warmup 5 --probe 5