"""Contains the functions for creating the plots for 1.B and 1.C of the asssigmnent

Course: Scientific Computing
Team: 5
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from ..utils.wave import Wave1D


def parse_args() -> argparse.Namespace:
    """Parse the arguments

    Returns:
        - (argparse.Namespace): The parsed arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "b_type",
        help="The initial conditions for the wabe (e.g., i, ii, iii)",
        type=str,
    )
    parser.add_argument(
        "steps",
        help="How many steps should be taken",
        type=int,
    )
    parser.add_argument(
        "-animate",
        help="Whether to make an animated plot or just heatmap",
        action="store_true"
    )
    return parser.parse_args()

def animate_wave(x_s: np.ndarray,
                 interval: int = 50,
                 ) -> None:
    """Animate the wave.
    
    Args:
        - x_s (np.ndarray): a 2D array (t, x), which has chronological amplitude values
            in x stacked vertically.
        - interval (int) = 50: The interval (in ms) between frames
    
    """
    if x_s.ndim != 2:
        raise ValueError(f"array must be 2D (t, x), got {x_s.shape}")

    fig = plt.figure( constrained_layout=True)
    artist = plt.plot(x_s[0])[0]
    ax = plt.gca()
    ax.set_ylim((-1.2, 1.2))

    def update(frame_idx: int) -> tuple:
        """Update function that is required by FuncAnimation."""
        artist.set_ydata(x_s[frame_idx])
        
        return (artist,)

    plt.title("Wave Animation")
    plt.xlabel("Space (x)")
    plt.ylabel("Amplitude")

    anim = FuncAnimation(fig, update, frames=x_s.shape[0], interval=interval, blit=True)
    plt.show()

def main():
    """The entry point when run as a script.
    
    Expected CLI arguments:
        - b_type (str): denotes the type of initial conditions for the wave.
        - steps (int): the number of steps to take (delta t = 0.001).
        - '-animate' (OPTIONAL): flag to denot if animation should be made instead of heatmap."""
    args = parse_args()

    if args.b_type not in ['i', 'ii', 'iii']:
        raise ValueError("Argument b_type should be one of ['i', 'ii', 'iii']")
    
    x0 = np.linspace(0, 1, 1000)

    if args.b_type == 'i':
        x_s = np.sin(2*np.pi*x0)
    elif args.b_type == 'ii':
        x_s = np.sin(5*np.pi*x0)
    else:
        x_s = np.where((1/5 < x0) & (x0 < 2/5), np.sin(5*np.pi*x0), 0)

    wave = Wave1D(x_s, 0.001, 0.001, c=1.0)

    if args.animate:
        wave.run(args.steps)
        animate_wave(wave.x_arr, interval=10)
    
    else:
        wave.run(args.steps)

        plt.imshow(wave.x_arr, aspect='auto', cmap='viridis')
        plt.colorbar(label='Wave Amplitude')
        plt.ylabel('Time (t)')
        plt.xlabel('Space (x)')
        plt.title(f'Initial Condition: {args.b_type}')
        plt.show()

if __name__ == "__main__":
    main()
