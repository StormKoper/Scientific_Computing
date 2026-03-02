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

def compare_dla_rw(N: int = 100, n_growth: int = 100, params: int = 10, sims: int = 10):
    """Compare the DLA growth with random walk growth"""
    dlas = np.ndarray((N, N, params, sims), dtype=bool)
    mcs = np.ndarray((N, N, params, sims), dtype=bool)
    seeds = np.random.SeedSequence(42).spawn(2*sims)
    dla_seeds = seeds[sims:]
    mc_seeds = seeds[:sims]
    # DLA growth
    for i, eta in enumerate(np.linspace(0, 1, params)):
        for j, seed in enumerate(dla_seeds):
            dla = DLA(N=N, eta=eta, use_jit=True, seed=seed)
            dla.run(n_growth=n_growth)
            dlas[..., i, j] = dla.obj_mask
    
    # random walk growth
    for i, p_s in enumerate(np.linspace(0.2, 1, params)):
        for j, seed in enumerate(mc_seeds):
            temp = np.random.default_rng(seed).choice([0,1], size=(N, N), p=[0.5, 0.5])
            mcs[..., i, j] = temp

    avg_dla = np.mean(dlas, axis=-1)
    avg_mc = np.mean(mcs, axis=-1)
    msd_grid = np.mean((avg_dla[..., :, None] - avg_mc[..., None, :]) ** 2, axis=[0,1])
    plt.figure(figsize=(8, 6), constrained_layout=True)
    plt.imshow(msd_grid, cmap='viridis')
    plt.colorbar(label="Mean Squared Difference")
    plt.title("Mean Squared Difference between DLA and MC Growth")
    plt.xticks(ticks=np.arange(params), labels=[f"{p:.2f}" for p in np.linspace(0.2, 1, params)])
    plt.yticks(ticks=np.arange(params), labels=[f"{eta:.2f}" for eta in np.linspace(0, 1, params)])
    plt.xlabel("MC $p_s$")
    plt.ylabel("DLA $\\eta$")
    plt.show()

if __name__ == "__main__":
    animate_growth()