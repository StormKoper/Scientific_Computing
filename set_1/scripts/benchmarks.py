import argparse
import timeit
import numpy as np
from copy import deepcopy

from ..utils.wave import Wave1D, Wave2D
from ..utils.TIDE import Jacobi, GaussSeidel, SOR

def parse_args() -> argparse.Namespace:
    """Parse the arguments

    Returns:
        - (argparse.Namespace): The parsed arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-method",
        choices=["Wave1D", "Wave2D", "SOR", "GaussSeidel", "Jacobi"],
        help="Class to benchmark ['Wave1D', 'Wave2D', 'SOR', 'GaussSeidel', 'Jacobi']",
        type=str,
        required=True
    )
    parser.add_argument(
        "-n_iters",
        help="Number of iterations to run",
        type=int,
        required=False,
        default=1000
    )
    parser.add_argument(
        "-N",
        type=int,
        default=100,
        help="Grid size (N x N for 2D, length N for 1D)"
    )
    parser.add_argument(
        "--warmup_jit",
        help="Whether to warmup JIT optimization or not",
        action="store_true"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    match args.method:
        case "Wave1D":
            x0 = np.linspace(0, 1, args.N)
            x0 = np.sin(2*np.pi*x0)
            x0[0] = 0
            x0[-1] = 0
            base = Wave1D(x0.copy(), 0.0001, 0.04, save_every=0)
            jit = Wave1D(x0.copy(), 0.0001, 0.04, save_every=0, use_jit=True)
        case "Wave2D":
            x0 = np.zeros((args.N, args.N))
            x0[0, :] = 1 # y=1 is index 0
            base = Wave2D(x0.copy(), 0.0001, 0.04, save_every=0)
            jit = Wave2D(x0.copy(), 0.0001, 0.04, save_every=0, use_jit=True)
        case "Jacobi":
            x0 = np.zeros((args.N, args.N))
            x0[0, :] = 1
            base = Jacobi(x0.copy(), save_every=0)
            jit = Jacobi(x0.copy(), save_every=0, use_jit=True)
        case "GaussSeidel":
            x0 = np.zeros((args.N, args.N))
            x0[0, :] = 1
            base = GaussSeidel(x0.copy(), save_every=0)
            jit = GaussSeidel(x0.copy(), save_every=0, use_jit=True)
        case "SOR":
            x0 = np.zeros((args.N, args.N))
            x0[0, :] = 1
            base = SOR(x0.copy(), save_every=0)
            jit = SOR(x0.copy(), save_every=0, use_jit=True)

    if args.warmup_jit:
        print("Warming up JIT optimization...")
        warmup = deepcopy(jit)
        warmup.run(10)

    print(f"Running base benchmark for {args.method}...")
    start_time = timeit.default_timer()
    base.run(args.n_iters)
    end_time = timeit.default_timer()
    base_time = end_time - start_time

    print(f"Running JIT benchmark for {args.method}...")
    start_time = timeit.default_timer()
    jit.run(args.n_iters)
    end_time = timeit.default_timer()
    jit_time = end_time - start_time

    print(f"{args.method} - Base time: {base_time:.4f} seconds, JIT time: {jit_time:.4f} seconds, Speedup: {base_time/jit_time:.2f}x")