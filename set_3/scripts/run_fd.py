import argparse
from ..utils.FD import FD

def parse_args():
    parser = argparse.ArgumentParser(description="Run the finite difference solver for the 2D Navier-Stokes equations.")
    parser.add_argument("--dx", type=float, default=0.05, help="Grid spacing in x direction")
    parser.add_argument("--dy", type=float, default=0.05, help="Grid spacing in y direction")
    parser.add_argument("--dt", type=float, default=0.001, help="Time step size")
    parser.add_argument("--rho", type=float, default=1.0, help="Fluid density")
    parser.add_argument("--nu", type=float, default=0.1, help="Kinematic viscosity")
    parser.add_argument("--time", type=float, default=1.0, help="Total simulation time")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    fd = FD(dx=args.dx, dy=args.dy, dt=args.dt, rho=args.rho, nu=args.nu)
    fd.run(time=args.time)
    fd.plot()