"""Contains the functions for creating the plots for questions E, F and G of Set 1.

Course: Scientific Computing
Team: 5
"""

import argparse
import itertools

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

from ..utils.config import *  # noqa: F403
from ..utils.misc import analytical_concentration
from ..utils.wave import Wave2D


def parse_args() -> argparse.Namespace:
    """Parse the arguments

    Returns:
        - (argparse.Namespace): The parsed arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-question",
        choices=["E", "F", "G"],
        help="For which question you want to create a plot ['E', 'F', 'G']",
        type=str,
        required=True
    )
    return parser.parse_args()

def plot_concentration() -> None:
    """Create a figure for showing numerical vs. analytical concentrations at various timesteps.

    Runs a 2D concentration diffusion scheme where top row is initialised and fixed at C=1. Will
    run numeric diffusion and plot concentration of single y-slice at t=[1.0, 0.1, 0.01, 0.001].
    Additionally, since for this specific initialization an analytical solution exists, this will
    be plotted as well.
    """
    N = 25
    t_s = [0.001, 0.01, 0.1, 1.0]
    dx = 1.0 / (N-1)
    dt = 0.05 * dx**2 #dt so that d=0.05 for stability

    # init wave
    x0 = np.zeros((N, N))
    x0[0, :] = 1
    wave = Wave2D(x0, dt, dx, D=1.0, save_every=1)

    # run for required steps
    total_steps = int(np.ceil(max(t_s) / wave.dt))
    wave.run(total_steps)

    # init figure and create cmap
    fig, ax = plt.subplots(1, 1, figsize=(10,8), constrained_layout=True)
    mpl_cmap = mpl.colormaps['viridis']

    colors = mpl_cmap(np.linspace(0, 1, len(t_s)))
    custom_cmap = ListedColormap(colors)

    true_ts = []
    for i, t in enumerate(t_s):
        t_index = round(t / wave.dt)
        true_t = t_index * wave.dt
        true_ts.append(true_t)

        concentration_slice = wave.x_arr[::-1, 1, t_index]
        y_vals = np.linspace(0, 1, N, dtype=float)
        ax.plot(y_vals, concentration_slice, linestyle="None", 
                marker='o', c=custom_cmap(i))
        
        ax.plot(y_vals, [analytical_concentration(y, true_t, 1, 1000) for y in y_vals], 
                 linestyle="-", c=custom_cmap(i))

    # custom legend (yes the first variable name was chosen deliberately :D)
    anal_legend = mlines.Line2D([], [], color='black', linestyle='-', label='Analytical')
    num_legend = mlines.Line2D([], [], color='black', marker='o', linestyle='None', label='Numerical')
    t_1 = mlines.Line2D([], [], color=custom_cmap(0), marker='s', linestyle='None', label=f't: {true_ts[0]:.3f}')
    t_2 = mlines.Line2D([], [], color=custom_cmap(1), marker='s', linestyle='None', label=f't: {true_ts[1]:.3f}')
    t_3 = mlines.Line2D([], [], color=custom_cmap(2), marker='s', linestyle='None', label=f't: {true_ts[2]:.3f}')
    t_4 = mlines.Line2D([], [], color=custom_cmap(3), marker='s', linestyle='None', label=f't: {true_ts[3]:.3f}')
    ax.legend(
        handles=[anal_legend, num_legend, t_1, t_2, t_3, t_4], 
        loc='upper left',
        shadow=True,
        fancybox=True
    )

    ax.set_xlabel("y-value", fontsize=14)
    ax.set_ylabel("Concentration (C)", fontsize=14)

    plt.suptitle(f"Numerical vs. Analytical Concentration at Various Time Steps ({N}x{N}-grid)",
                 fontsize=16)
    plt.show()

def plot_states() -> None:
    """Create a figure with heatmaps for 2D diffusion at various time steps.

    Runs a 2D diffusion scheme on an NxN grid with the top row fixed at C=1
    (periodic boundary). Plots heatmaps at t = [0, 0.001, 0.01, 0.1, 1.0].
    """
    N = 25
    t_s = [0, 0.001, 0.01, 0.1, 1.0]
    dx = 1.0 / (N-1)
    dt = 0.05 * dx**2 #dt so that d=0.05 for stability

    # init wave
    x0 = np.zeros((N, N))
    x0[0, :] = 1
    wave = Wave2D(x0, dt, dx, D=1.0, save_every=1)

    # run for required steps
    total_steps = int(np.ceil(max(t_s) / wave.dt))
    wave.run(total_steps)

    fig, axes = plt.subplots(1, 5, figsize=(18, 5), constrained_layout=True)
    
    for ax, t in itertools.zip_longest(axes.flatten(), t_s):
        if t is None:
            ax.set_visible(False)
            continue
        t_index = round(t / wave.dt)
        true_t = t_index * wave.dt

        im = ax.imshow(wave.x_arr[..., t_index])
        ax.set_title(f"t={true_t:.4f}")
        ax.set_ylabel("y")
        ax.set_xlabel("x")
    
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, orientation='horizontal')
    plt.suptitle("2D Heatmap of Concentration Values at Various Time Steps")
    plt.show()

def animate_grid() -> None:
    """Animate the 2D concentration diffusion as a heatmap over time.

    Runs a 2D diffusion scheme on an NxN grid with the top row fixed at C=1
    and animates the result from t=0 to t=1 using ``FuncAnimation``.
    """
    N = 25
    dx = 1.0 / (N-1)
    dt = 0.15 * dx**2 / 1.0 #dt s.t. d=0.15 for stability

    # init wave
    x0 = np.zeros((N, N))
    x0[0, :] = 1
    wave = Wave2D(x0, dt, dx, D=1.0, save_every=1)

    # run to t=1
    total_steps = int(np.ceil(1 / wave.dt))
    wave.run(total_steps)

    fig = plt.figure(constrained_layout=True)
    artist = plt.imshow(wave.x_arr[..., 0])
    ax = plt.gca()
    ax.set_aspect('equal')
    title = ax.set_title("Wave Animation (t = 0.0000)")

    def update(frame_idx: int) -> tuple:
        """Update function that is required by FuncAnimation."""
        artist.set_data(wave.x_arr[..., frame_idx])
        title.set_text(f"Wave Animation (t = {frame_idx * dt:.4f})")
        
        return (artist, title)

    plt.title("Wave Animation")
    plt.xlabel("Space (x)")
    plt.ylabel("Space (y)")

    _ = FuncAnimation(fig, update, frames=wave.x_arr.shape[-1], interval=1, blit=False)
    plt.show()

def main():
    """Entry point when run as a script.

    CLI arguments:
        -question (str): One of 'E', 'F', or 'G'.
            - E: plot numerical vs. analytical concentration profiles.
            - F: plot 2D heatmaps at various time steps.
            - G: animate the 2D diffusion over time.
    """
    args = parse_args()

    if args.question == 'E':
        plot_concentration()
    elif args.question == 'F':
        plot_states()
    else:
        animate_grid()

if __name__ == "__main__":
    main()
