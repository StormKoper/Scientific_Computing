"""Contains the functions for creating the plots for 1.B and 1.C of the asssigmnent

Course: Scientific Computing
Team: 5
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from ..utils.wave import Wave2D


def parse_args() -> argparse.Namespace:
    """Parse the arguments

    Returns:
        - (argparse.Namespace): The parsed arguments
    """
    parser = argparse.ArgumentParser()

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
    return parser.parse_args()

def animate_grid(x_arr: np.ndarray,
                 interval: int = 10,
                 ) -> None:
    """Animate the wave.
    
    Args:
        - x_arr (np.ndarray): a 3D array (x, y, t), which has chronological amplitude values
            concatenated along time.
        - interval (int) = 10: The interval (in ms) between frames
    
    """
    if x_arr.ndim != 3:
        raise ValueError(f"array must be 3D (x, y, t), got {x_arr.shape}")

    fig = plt.figure(constrained_layout=True)
    artist = plt.imshow(x_arr[..., 0])
    ax = plt.gca()
    ax.set_aspect('equal')

    def update(frame_idx: int) -> tuple:
        """Update function that is required by FuncAnimation."""
        artist.set_data(x_arr[..., frame_idx])
        
        return (artist,)

    plt.title("Wave Animation")
    plt.xlabel("Space (x)")
    plt.ylabel("Space (y)")

    anim = FuncAnimation(fig, update, frames=x_arr.shape[-1], interval=interval, blit=True)
    plt.show()

def main():
    """The entry point when run as a script.
    
    Expected CLI arguments:
        - steps (int): the number of steps to take (delta t = 0.0001).
        - save_every (int): how often should the wave be saved for plotting.
        - '-animate' (OPTIONAL): flag to denot if animation should be made instead of heatmap."""
    args = parse_args()
    
    x0 = np.zeros((25, 25))
    x0[0, :] = 1

    wave = Wave2D(x0, 0.0001, 0.04, D=1.0, save_every=args.save_every)
    wave.run(args.steps)
    animate_grid(wave.x_arr, interval=10)

if __name__ == "__main__":
    main()
