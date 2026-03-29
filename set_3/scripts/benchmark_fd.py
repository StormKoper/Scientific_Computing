from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from ..utils.config import *  # noqa: F403
from ..utils.FD import FD

def parse_args():
    parser = ArgumentParser(description="Run the finite difference solver for the 2D Navier-Stokes equations.")
    sim = parser.add_argument_group(title="Simulation Parameters")
    sim.add_argument("--dx", type=float, default=.01, help="Grid spacing in x direction")
    sim.add_argument("--dy", type=float, default=.01, help="Grid spacing in y direction")
    sim.add_argument("--dt", type=float, default=.001, help="Time step size")
    sim.add_argument("--rho", type=float, default=1.0, help="Fluid density")
    sim.add_argument("--nu", type=float, default=.1, help="Kinematic viscosity")
    solve = parser.add_argument_group(title="Solver Parameters")
    solve.add_argument("--time", type=float, default=5.0, help="Total simulation time")
    solve.add_argument("--warmup", type=float, default=0.0, help="Time for warm-up simulation (0 to disable)")
    solve.add_argument("--probe", type=int, default=0, help="Strouhal number probe data collection frequency (0 to disable)")
    solve.add_argument("--benchmark", action="store_true", help="Run the solver benchmark instead of the main simulation. Disables probe")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    FDIR = Path(__file__).parent.parent / "figures"
    Path.mkdir(FDIR, exist_ok=True)
    time = str(datetime.now()).replace(' ', '_').replace(':', '-')
    fd = FD(dx=args.dx, dy=args.dy, dt=args.dt, rho=args.rho, nu=args.nu)
    if args.warmup:
        print(f"Running warm-up simulation for {args.warmup} seconds...")
        fd.run(time=args.warmup)
    if args.benchmark:
        print(f"Running benchmark for {args.time} seconds...")
        time = str(datetime.now()).replace(' ', '_').replace(':', '-')
        fd.benchmark(time=args.time, p_threshold=1e-4, p_max_iters=5000,
                        show=True, save=True, filename=FDIR / f"FD_iters_{time}.png")
    else:
        print(f"Running main simulation for {args.time} seconds...")
        fd.run(time=args.time, probe=args.probe)
        if args.probe:
            print("Plotting Strouhal probe data...")
            fd.plot_strouhal(probe=args.probe, show=True, save=True, filename=FDIR / f"FD_St_{time}.png")
    print("Plotting the velocity and pressure fields...")
    fd.plot(show=True, save=True, filename=FDIR / f"FD_fields_{time}.png")