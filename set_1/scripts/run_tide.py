import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

from ..utils.config import *  # noqa: F403
from ..utils.misc import analytical_concentration
from ..utils.TIDE import SOR, Gauss_S, Jacobi


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
    """Create a figure for showing numerical vs. analytical concentrations at various timesteps.

    Runs a 2D concentration diffusion scheme where top row is initialised and fixed at C=1. Will
    run numeric diffusion and plot concentration of single y-slice at t=[1.0, 0.1, 0.01, 0.001].
    Additionally, since for this specific initialization an analytical solution exists, this will
    be plotted as well.
    """
    N = 50
    x0 = np.zeros((N, N))
    x0[0, :] = 1

    J = Jacobi(x0)
    G = Gauss_S(x0)
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
    
    ax1.plot(y_vals, y_vals, 
                 linestyle=":", c='black', label="Analytical Sol")
    ax2.plot(y_vals, y_vals, 
                 linestyle=":", c='black', label="Analytical Sol")
    ax3.plot(y_vals, y_vals, 
                 linestyle=":", c='black', label="Analytical Sol")
    
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


def main():
    """Entry point when run as a script.

    CLI arguments:
        -question (str): One of 'H', 'I', or 'J'.
            - E: plot numerical vs. analytical concentration profiles.
            - F: plot 2D heatmaps at various time steps.
            - G: animate the 2D diffusion over time.
    """
    args = parse_args()

    if args.question == 'H':
        plot_itermethods_vs_analytical()
    elif args.question == 'I':
        pass
    else:
        pass

if __name__ == "__main__":
    main()


