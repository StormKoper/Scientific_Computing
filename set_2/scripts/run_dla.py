import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from ..utils.config import *  # noqa: F403
from ..utils.DLA import DLA

def save_frames(N: int = 100, n_growth: int = 100, interval: int = 10):
    """Save the frames of the DLA growth"""
    dla = DLA(N=N, eta=0.8, use_jit=True, seed=42)

    for i in range(0, n_growth+1, interval):
        dla.run(n_growth=interval)
        plt.imshow(dla.x_arr[..., i], cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Concentration Field at Frame {i}")
        plt.xlabel("Space (x)")
        plt.ylabel("Space (y)")
        plt.savefig(f"set_2/results/frame_{i:03d}.png")
        plt.close()

def animate_growth():
    """Animate the growth of the DLA cluster"""
    dla = DLA(N=100, eta=0.5, omega=1.0, use_jit=True, seed=42)
    dla.run(n_growth=250)

    fig = plt.figure(constrained_layout=True)
    artist = plt.imshow(dla.x_arr[..., 0], vmin=-1, vmax=1)
    ax = plt.gca()
    ax.set_aspect('equal')
    title = ax.set_title("Growth Animation - Size 0")

    def update(frame_idx: int) -> tuple:
        """Update function that is required by FuncAnimation."""
        artist.set_data(dla.x_arr[..., frame_idx])
        title.set_text(f"Growth Animation - Size {frame_idx}")
        return (artist, title)

    plt.xlabel("Space (x)")
    plt.ylabel("Space (y)")

    _ = FuncAnimation(fig, update, frames=dla.x_arr.shape[-1], interval=10, blit=True)
    plt.show()

if __name__ == "__main__":
    animate_growth()