import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from ..utils.config import *  # noqa: F403
from ..utils.RD import GrayScott


def parse_args() -> argparse.Namespace:
    """Parse the arguments

    Returns:
        - (argparse.Namespace): The parsed arguments
    """
    desc = "### SIMULATE 2D GRAY-SCOTT REACTION-DIFFUSION SYSTEM ###"
    parser = argparse.ArgumentParser(description="#"*len(desc) + '\n' + desc + '\n' + "#"*len(desc),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    physics = parser.add_argument_group('Physics Parameters')
    physics.add_argument("-dx", help="Size of grid step", 
                         type=float, default=1.0, metavar="")
    physics.add_argument("-dt", help="Size of time step", 
                         type=float, default=1.0, metavar="")
    physics.add_argument("-Du", help="Diffusion Coefficient of u", 
                         type=float, default=0.16, metavar="")
    physics.add_argument("-Dv", help="Diffusion Coefficient of v", 
                         type=float, default=0.08, metavar="")
    physics.add_argument("-f", help="Rate at which u is supplied", 
                         type=float, default=0.035, metavar="")
    physics.add_argument("-k", help="k (+ f) denotes the rate at which v decays", 
                         type=float, default=0.060, metavar="")

    sim = parser.add_argument_group("Simulation Settings")
    sim.add_argument('-N', help="Side length of the square grid", 
                     type=int, default=50, metavar="")
    sim.add_argument('-i', '--iterations', help="Number of time steps to simulate", 
                     type=int, default=5000, metavar="")
    sim.add_argument('--noise', help="Amplitude of random noise added to initial state", 
                     type=float, default=0.0, metavar="")
    
    vis = parser.add_argument_group("Visualization Settings")
    vis.add_argument('-p', '--plot', help="How to visualize the results (choices: animate, static, statspanel)", 
                     type=str, default='animate', choices=['animate', 'static', 'statspanel'], 
                     metavar="")
    
    return parser.parse_args()

def plot_final_conc(GS: GrayScott) -> None:
    """Plot an image of the final state of the simulation.
    
    The concentration of u is assigned to the and R channel, whereas the 
    concentration of v is assigned to the G and B channels. Thus fully red
    means only u within that cell, fully cyan means only v in that cell.
    Anything in between means both chemicals are present.

    Args:
        - GS (GrayScott): The GrayScott reaction diffusion object.
    
    """
    N = GS.grid.shape[0]

    rgb = np.zeros((N, N, 3))
    rgb[..., 0] = GS.grid["u"]
    rgb[..., 1] = GS.grid["v"]
    rgb[..., 2] = GS.grid["v"]
    np.clip(rgb, 0, 1, out=rgb)

    _ = plt.figure(figsize=(8, 8), constrained_layout=True)

    plt.imshow(rgb)
    plt.title(f"Gray-Scott Concentration Field (t={GS.iter_count* GS.dt:.2f})")
    plt.xlabel("Space (x)")
    plt.ylabel("Space (y)")
    plt.show()

def animate_conc(GS: GrayScott) -> None:
    """Animate the concentration fields from the simulation.
    
    The concentration of u is assigned to the and R channel, whereas the 
    concentration of v is assigned to the G and B channels. Thus fully red
    means only u within that cell, fully cyan means only v in that cell.
    Anything in between means both chemicals are present.

    Args:
        - GS (GrayScott): The GrayScott reaction diffusion object.
    
    """
    if GS.grid_hist is None:
        raise ValueError("No simulation history, please call GS.run()")
    
    N = GS.grid.shape[0]
    
    rgb = np.zeros((N, N, 3))
    rgb[..., 0] = GS.grid_hist["u"][..., 0]
    rgb[..., 1] = GS.grid_hist["v"][..., 0]
    rgb[..., 2] = GS.grid_hist["v"][..., 0]
    np.clip(rgb, 0, 1, out=rgb)
    
    fig = plt.figure(figsize=(8,8), constrained_layout=True)

    artist = plt.imshow(rgb)
    ax = plt.gca()
    ax.set_aspect('equal')
    textbox = ax.text(0.05, 0.05, "t: 0.00",
        transform=ax.transAxes,
        verticalalignment="bottom",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    def update(frame_idx: int) -> tuple:
        """Update function that is required by FuncAnimation."""
        rgb[..., 0] = GS.grid_hist["u"][..., frame_idx] # type: ignore
        rgb[..., 1] = GS.grid_hist["v"][..., frame_idx] # type: ignore
        rgb[..., 2] = GS.grid_hist["v"][..., frame_idx] # type: ignore
        np.clip(rgb, 0, 1, out=rgb)
        artist.set_data(rgb)
        textbox.set_text(f"t: {frame_idx * GS.dt:.2f}")
        
        return (artist, textbox)

    plt.title("Gray-Scott Animation")
    plt.xlabel("Space (x)")
    plt.ylabel("Space (y)")

    _ = FuncAnimation(fig, update, frames=GS.grid_hist.shape[-1], interval=1, blit=True)
    plt.show()

def plot_conc_statistics(GS: GrayScott) -> None:
    """Plot concentration statistics from the Gray-Scott simulation.
    
    Figure consists of three panels: 
        1) Mean concentration over time.
        2) Simulation parameters.
        3) 1D cross section of final state

    Args:
        - GS (GrayScott): The GrayScott reaction diffusion object.
    
    """
    if GS.grid_hist is None:
        raise ValueError("No simulation history, please call GS.run()")
    
    fig, ax_dict = plt.subplot_mosaic([["A", "A", "B"], ["C", "C", "C"]],
                                      constrained_layout=True,
                                      figsize=(12,8))
    
    t_range = np.arange(GS.iter_count) * GS.dt
    u_mean = np.mean(GS.grid_hist["u"], axis=(0, 1))
    v_mean = np.mean(GS.grid_hist["v"], axis=(0, 1))

    ax_dict["A"].plot(t_range, u_mean, c="firebrick", label="$u$")
    ax_dict["A"].plot(t_range, v_mean, c="darkcyan", label="$v$")
    ax_dict["A"].set_xlabel("Time (t)")
    ax_dict["A"].set_ylabel("Mean Concentration (c)")
    ax_dict["A"].set_title("Average Concentration of u and v Over Time")
    ax_dict["A"].legend(shadow=True, fancybox=True)

    args = parse_args()
    ax_dict["B"].axis("off")
    parameter_text = (
        "Simulation Parameters:\n"
        "------------------------\n"
        f"$D_u$ = {args.Du}\n"
        f"$D_v$ = {args.Dv}\n"
        f"$f$ = {args.f}\n"
        f"$k$ = {args.k}\n"
        f"Noise = {args.noise}"
    )
    ax_dict["B"].text(0.5, 0.5, parameter_text, transform=ax_dict["B"].transAxes, 
             fontsize=14, va='center', ha='center',
             bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5})

    N = GS.grid.shape[0]
    cntr = N // 2
    cross_sect = GS.grid_hist[cntr, :, -1]

    ax_dict["C"].plot(np.arange(N), cross_sect["u"], c="firebrick", label="$u$")
    ax_dict["C"].plot(np.arange(N), cross_sect["v"], c="darkcyan", label="$v$")
    ax_dict["C"].set_xlabel("Space (x)")
    ax_dict["C"].set_ylabel("Concentration (c)")
    ax_dict["C"].set_title(f"1D Cross-Section of Final State (y = {cntr})")
    ax_dict["C"].legend(shadow=True, fancybox=True)

    plt.show()

def seed_grid(GS: GrayScott, n: float) -> None:
    """Seed the initial grid with concentrations for u and v.

    Chemical u is intialized to be 0.5 in each cell. An center square
    of 20% of the grid is initialized with a concentration of 0.25 of 
    chemical v. A uniform noise mask is applied if n > 0, where n 
    denotes the amplitude of the perturbations.

    Args:
        - GS (GrayScott): The GrayScott reaction diffusion object.
        - n (float): The amplitude of the perturbations
    
    """
    N = GS.grid.shape[0]
    GS.grid["u"] = np.full((N, N), 0.5)

    # inner square of 10%
    start = (N//2) - int(N*0.1)
    end = (N//2) + int(N*0.1)

    GS.grid["v"][start:end, start:end] = 0.25

    if n > 0:
        u_noise = GS.gen.uniform(-n, n, size=(N,N))
        v_noise = GS.gen.uniform(-n, n, size=(N,N))

        GS.grid["u"] += u_noise
        GS.grid["v"] += v_noise

        np.clip(GS.grid["u"], 0, 1, out=GS.grid["u"])
        np.clip(GS.grid["v"], 0, 1, out=GS.grid["v"])

def main():
    args = parse_args()

    consts = {
        "Du": args.Du,
        "Dv": args.Dv,
        "f": args.f,
        "k": args.k
    }
    
    GS = GrayScott(args.N, args.dt, args.dx, consts)
    seed_grid(GS, args.noise)

    GS.run(args.iterations)

    if args.plot == "static":
        plot_final_conc(GS)
    elif args.plot == "animate":
        animate_conc(GS)
    else:
        plot_conc_statistics(GS)

if __name__ == "__main__":
    main()
