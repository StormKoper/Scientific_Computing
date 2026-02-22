"""Contains the functions for creating the plots for 1.B and 1.C of the asssigmnent

Course: Scientific Computing
Team: 5
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from ..utils.config import *  # noqa: F403
from ..utils.misc import get_fourier_coefficients, analytical_wave1D
from ..utils.wave import Leapfrog, Wave1D


def parse_args() -> argparse.Namespace:
    """Parse the arguments

    Returns:
        - (argparse.Namespace): The parsed arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-b_type",
        help="The initial conditions for the wabe (e.g., i, ii, iii)",
        type=str,
        default='i',
        required=False,
    )
    parser.add_argument(
        "-steps",
        help="How many steps should be taken",
        type=int,
        default=1000,
        required=False,
    )
    parser.add_argument(
        "-save_every",
        help="How often should the wave be saved for plotting",
        type=int,
        default=1,
        required=False,
    )
    parser.add_argument(
        "--animate",
        help="Whether to make an animated plot or just heatmap",
        action="store_true"
    )
    parser.add_argument(
        "--leapfrog",
        help="Whether to use leapfrog method instead of finite difference",
        action="store_true"
    )
    parser.add_argument(
        "--compare",
        help="Whether to plot the comparison between normal and leapfrog method",
        action="store_true"
    )
    return parser.parse_args()

def animate_wave(x_arr: np.ndarray,
                 interval: int = 10,
                 ) -> None:
    """Animate the wave.
    
    Args:
        - x_arr (np.ndarray): a 2D array (x, t), which has chronological amplitude values
            in x stacked horizontally.
        - interval (int) = 10: The interval (in ms) between frames
    
    """
    if x_arr.ndim != 2:
        raise ValueError(f"array must be 2D (x, t), got {x_arr.shape}")

    fig = plt.figure(constrained_layout=True)
    artist = plt.plot(x_arr[..., 0])[0]
    ax = plt.gca()
    ax.set_ylim((-1.2, 1.2))

    def update(frame_idx: int) -> tuple:
        """Update function that is required by FuncAnimation."""
        artist.set_ydata(x_arr[..., frame_idx])
        
        return (artist,)

    plt.title("Wave Animation")
    plt.xlabel("Space (x)")
    plt.ylabel("Amplitude")

    anim = FuncAnimation(fig, update, frames=x_arr.shape[-1], interval=interval, blit=True)
    plt.show()

def plot_normal_vs_leapfrog(steps: int, case: str) -> None:
    dx = 0.001
    dt = 0.001
    c = 1.0
    n_steps = steps
    
    x = np.linspace(0, 1, int(1 / dx) + 1)
    if case == 'i':
        x0 = np.sin(2*np.pi*x)
    elif case == 'ii':
        x0 = np.sin(5*np.pi*x)
    elif case == 'iii':
        x0 = np.where((1/5 < x) & (x < 2/5), np.sin(5*np.pi*x), 0)
    else:
        raise ValueError("Invalid case: case should be one of ['i', 'ii', 'iii']")

    print("Running finite difference scheme...")
    normal = Wave1D(x0.copy(), dt, dx, c=c, save_every=1)
    normal.run(n_steps - 1)

    print("Running leapfrog method...")
    leapfrog = Leapfrog(x0.copy(), dt, dx, c=c, save_every=1)
    leapfrog.run(n_steps - 1)

    print("Calculating analytical solution...")
    t_arr = np.arange(n_steps) * dt
    An = get_fourier_coefficients(n_terms=100)
    analytical = np.array([analytical_wave1D(x, t, case, c=c, An=An) for t in t_arr]).T
    
    plt.figure(figsize=(5, 4))
    plt.plot(t_arr, np.max(np.abs(analytical - normal.x_arr), axis=0),
             c="firebrick", label="Normal Method")
    plt.plot(t_arr, np.max(np.abs(analytical - leapfrog.x_arr), axis=0),
             c="darkcyan", label="Leapfrog Method")
    plt.legend(fancybox=True, shadow=True, loc="upper right")
    plt.title(f"Initial condition {case}")
    plt.xlabel("Time (t)")
    plt.ylabel("Max Absolute Error")
    plt.tight_layout()
    plt.show()

def main():
    """The entry point when run as a script.
    
    Expected CLI arguments:
        - b_type (str): denotes the type of initial conditions for the wave.
        - steps (int): the number of steps to take (delta t = 0.001).
        - save_every (int): how often should the wave be saved for plotting.
        - animate (OPTIONAL): flag to denote if animation should be made instead of heatmap."""
    args = parse_args()

    if args.compare:
        plot_normal_vs_leapfrog(args.steps, args.b_type)
        return

    if args.b_type not in ['i', 'ii', 'iii']:
        raise ValueError("Argument b_type should be one of ['i', 'ii', 'iii']")
    
    x0 = np.linspace(0, 1, 1000)

    if args.b_type == 'i':
        x0 = np.sin(2*np.pi*x0)
    elif args.b_type == 'ii':
        x0 = np.sin(5*np.pi*x0)
    else:
        x0 = np.where((1/5 < x0) & (x0 < 2/5), np.sin(5*np.pi*x0), 0)

    if args.leapfrog:
        wave = Leapfrog(x0, 0.001, 0.001, c=1.0, save_every=args.save_every)
    else:
        wave = Wave1D(x0, 0.001, 0.001, c=1.0, save_every=args.save_every)

    if args.animate:
        wave.run(args.steps)
        animate_wave(wave.x_arr, interval=10)
    else:
        wave.run(args.steps)

        plt.imshow(wave.x_arr, aspect='auto', cmap='viridis', 
                   extent=(0.0,float(args.steps*wave.dt),1.0,0.0))
        plt.colorbar(label='Wave Amplitude')
        plt.ylabel('Space (x)')
        plt.xlabel('Time (t)')
        plt.title(f'Initial Condition: {args.b_type}')
        plt.show()

if __name__ == "__main__":
    main()
