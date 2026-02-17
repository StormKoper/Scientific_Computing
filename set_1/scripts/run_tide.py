import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from ..utils.config import *  # noqa: F403
from ..utils.TIDE import Jacobi, GaussSeidel, SOR


def parse_args() -> argparse.Namespace:
    """Parse the arguments

    Returns:
        - (argparse.Namespace): The parsed arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-question",
        choices=["H", "I", "J"],
        help="For which question you want to create a plot ['H', 'I', 'J']",
        type=str,
        required=True
    )
    return parser.parse_args()

def plot_itermethods_vs_analytical() -> None:
    """Create a figure for showing 3 iteration methods vs Analytical solution."""
    N = 50
    x0 = np.zeros((N, N))
    x0[0, :] = 1

    J = Jacobi(x0)
    G = GaussSeidel(x0)
    S = SOR(x0)

    J.run(500)
    G.run(500)
    S.run(500)

    # init figure and create cmap
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,6), constrained_layout=True)
    mpl_cmap = mpl.colormaps['inferno']

    steps_to_plot = np.arange(0, 501, 50)
    colors = mpl_cmap(np.linspace(0, 1, len(steps_to_plot)))
    custom_cmap = ListedColormap(colors)

    for i, iter in enumerate(steps_to_plot):

        y_vals = np.linspace(0, 1, N, dtype=float)

        J_slice = J.x_arr[::-1, 1, iter]
        ax1.plot(y_vals, J_slice, linestyle="-", c=custom_cmap(i), label=f"Iter: {iter:.0f}")

        G_slice = G.x_arr[::-1, 1, iter]
        ax2.plot(y_vals, G_slice, linestyle="-", c=custom_cmap(i), label=f"Iter: {iter:.0f}")

        S_slice = S.x_arr[::-1, 1, iter]
        ax3.plot(y_vals, S_slice, linestyle="-", c=custom_cmap(i), label=f"Iter: {iter:.0f}")
    
    # solving the time-independent diffusion equation, is just a line
    ax1.plot(y_vals, y_vals, linestyle=":", c='black', label="Analytical Sol")
    ax2.plot(y_vals, y_vals, linestyle=":", c='black', label="Analytical Sol")
    ax3.plot(y_vals, y_vals, linestyle=":", c='black', label="Analytical Sol")
    
    ax1.set_title("Jacobi Iteration")
    ax1.set_xlabel("y-value")
    ax1.set_ylabel("Concentration (C)")
    ax1.legend(fancybox=True, shadow=True)
    
    ax2.set_title("Gauss-Seidel Iteration")
    ax2.set_xlabel("y-value")
    ax2.set_ylabel("Concentration (C)")
    ax2.legend(fancybox=True, shadow=True)

    ax3.set_title("SOR Iteration")
    ax3.set_xlabel("y-value")
    ax3.set_ylabel("Concentration (C)")
    ax3.legend(fancybox=True, shadow=True)

    plt.suptitle(f"3 Different Iteration Methods vs. Analytical Concentration ({N}x{N}-grid)")
    plt.show()

def plot_convergence_measures():
    """Create a figure for showing the convergence speed of 3 iteration methods."""
    N = 50
    x0 = np.zeros((N, N))
    x0[0, :] = 1

    J = Jacobi(x0)
    G = GaussSeidel(x0)

    omegas = [1.0, 1.3, 1.6, 1.8, 1.9]
    Ss = [SOR(x0, save_every=0, omega=omega) for omega in omegas]

    n_steps = 1000

    J.run(n_steps)
    G.run(n_steps)
    [S.run(n_steps) for S in Ss]

    # init figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,6), constrained_layout=True)

    ax1.semilogy(np.arange(1, n_steps+1, 1), J.error_history, c='firebrick')
    ax2.semilogy(np.arange(1, n_steps+1, 1), G.error_history, c="darkcyan")
    
    mpl_cmap = mpl.colormaps['plasma']
    colors = mpl_cmap(np.linspace(0, 1, len(omegas)))
    custom_cmap = ListedColormap(colors)
    for i, (S, omega) in enumerate(zip(Ss, omegas)):
        ax3.semilogy(np.arange(1, n_steps+1, 1), S.error_history, c=custom_cmap(i), label=f"$\\omega={omega}$")
    
    ax1.set_title("Jacobi Iteration")
    ax1.set_xlabel("Iteration Number")
    ax1.set_ylabel("Max Error ($\\epsilon$)")
    
    ax2.set_title("Gauss-Seidel Iteration")
    ax2.set_xlabel("Iteration Number")
    ax2.set_ylabel("Max Error ($\\epsilon$)")

    ax3.set_title("SOR Iteration")
    ax3.set_xlabel("Iteration Number")
    ax3.set_ylabel("Max Error ($\\epsilon$)")
    ax3.legend(fancybox=True, shadow=True)

    plt.suptitle(f"Convergence Speed of 3 Iteration Methods ({N}x{N}-grid)")
    plt.show()

def find_optimal_omega():
    """Find the optimal omega for SOR iteration at different grid sizes."""
    N_values = [10, 20, 50, 100]
    optimal_omegas = []

    for N in N_values:
        x0 = np.zeros((N, N))
        x0[0, :] = 1

        omegas = np.linspace(1.7, 2.0, 11)
        Ss = [SOR(x0, save_every=0, omega=omega) for omega in omegas]

        n_steps = 1000
        [S.run(n_steps) for S in Ss]

        final_errors = [S.error_history[-1] for S in Ss]
        optimal_omega = omegas[np.argmin(final_errors)]
        optimal_omegas.append(optimal_omega)

    plt.plot(N_values, optimal_omegas, marker='o')
    plt.title("Optimal Omega for SOR Iteration vs Grid Size")
    plt.xlabel("Grid Size (N)")
    plt.ylabel("Optimal Omega")
    plt.xscale('log')
    plt.grid()
    plt.show()

def main():
    """Entry point when run as a script.

    CLI arguments:
        -question (str): One of 'H', 'I', or 'J'.
            - H: plot 3 iteration methods vs analytical solution.
            - I: plot convergence speed of 3 iteration methods.
            - J: (!not yet implemented)
    """
    args = parse_args()

    if args.question == 'H':
        plot_itermethods_vs_analytical()
    elif args.question == 'I':
        plot_convergence_measures()
    else:
        pass

if __name__ == "__main__":
    main()


