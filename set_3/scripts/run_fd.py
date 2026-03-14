import argparse
import matplotlib.pyplot as plt
from ..utils.config import *  # noqa: F403
from ..utils.FD import FD

def parse_args():
    parser = argparse.ArgumentParser(description="Run the finite difference solver for the 2D Navier-Stokes equations.")
    parser.add_argument_group(title="Simulation Parameters")
    parser.add_argument("--dx", type=float, default=0.005, help="Grid spacing in x direction")
    parser.add_argument("--dy", type=float, default=0.005, help="Grid spacing in y direction")
    parser.add_argument("--dt", type=float, default=0.001, help="Time step size")
    parser.add_argument("--rho", type=float, default=1.0, help="Fluid density")
    parser.add_argument("--nu", type=float, default=0.1, help="Kinematic viscosity")
    parser.add_argument_group(title="Solver Parameters")
    parser.add_argument("--time", type=float, default=1.0, help="Total simulation time")
    parser.add_argument("--P", type=int, default=2, choices=[0, 1, 2], help="Pressure solver order (0, 1, or 2)")
    parser.add_argument("--U", type=int, default=0, choices=[0, 1, 3], help="Advection term upwind order (0, 1, or 3)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    fd = FD(dx=args.dx, dy=args.dy, dt=args.dt, rho=args.rho, nu=args.nu)
    fd.run(time=args.time, P=args.P, U=args.U)
    fd.plot(show=True, save=True, filename=f"set_3/results/FD_P{args.P}_U{args.U}_{args.time}_{fd.Re:.0f}.png")