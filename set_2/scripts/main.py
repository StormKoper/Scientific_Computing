"""
Main entry for generating the figures used in Assignment 2

Creates plots for
- Part 1: Diffusion-Limited Aggregation (DLA)
- Part 2: Monte Carlo DLA
- Part 3: Gray-Scott Reaction-Diffusion model

!NOTE!
Use the `--animations` flag via CLI to not generate figures, and instead
show some animations for the reaction-diffusion and DLA processes.
"""
import argparse
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt

from set_2.scripts.run_dla import (
    animate_growth,
    benchmark_dla_jit,
    find_optimal_omega,
    plot_5_panel,
    plot_dla_density,
    plot_single,
)
from set_2.scripts.run_mc import plot_5_panel_MC, plot_mc_density
from set_2.scripts.run_rd import (
    GrayScott,
    animate_conc,
    parse_args,
    plot_concs,
    plot_final_conc,
    seed_grid,
)

from ..utils.config import *  # noqa: F403

FDIR = Path(__file__).parent.parent / "figures"
Path.mkdir(FDIR, exist_ok=True)

REPS = 20

def parse_args_main() -> argparse.Namespace:
    """Parse the arguments

    Returns:
        - (argparse.Namespace): The parsed arguments
    """
    parser = argparse.ArgumentParser("Create the Figures in the Report")

    parser.add_argument(
        "--animations",
        help="Whether to ONLY create the animations",
        action="store_true"
    )
    return parser.parse_args()


def figures_part_1():
    """Creating the figures for part 1. of the assignment"""
    print("Creating Figures for part 1 of the assignment")
    print("   Generating 5-panel DLA ...")
    _ = plot_5_panel(sims=REPS)
    plt.savefig(FDIR / "five_panel_DLA.png", dpi=300)
    plt.show()

    print("   Generating DLA density plot ...")
    _ = plot_dla_density(n_runs=REPS)
    plt.savefig(FDIR / "DLA_density.png", dpi=300)
    plt.show()

    print("   Benchmarking Base vs Jit ...\n        ", end="")
    benchmark_dla_jit(reps=REPS)

    print("   Generating Single 200x200 plot ...")
    _ = plot_single(N=200, sims=REPS)
    plt.savefig(FDIR / "single_DLA_N200.png", dpi=300)
    plt.show()

    print("   Generating Optimal Omega plot and Conducting T-tests ...")
    _ = find_optimal_omega(sims=REPS)
    plt.savefig(FDIR / "omega_sweep.png", dpi=300)
    plt.show()

    plt.close()

def figures_part_2():
    """Creating the figures for part 2. of the assignment"""

    print("\nCreating Figures for part 2 of the assignment")
    print("   Generating 5-panel Monte Carlo DLA ...")
    _ = plot_5_panel_MC(sims=REPS)
    plt.savefig(FDIR / "five_panel_MC.png", dpi=300)
    plt.show()

    print("   Generating Monte Carlo DLA density plot ...")
    _ = plot_mc_density(n_runs=REPS)
    plt.savefig(FDIR / "MC_density.png", dpi=300)
    plt.show()

    plt.close()

def setup_and_run_gs() -> GrayScott:
    """Small helper that reads mock arguments, initializes GS, sets up grid and runs"""
    args = parse_args()
    consts = {
        "Du": args.Du,
        "Dv": args.Dv,
        "f": args.f,
        "k": args.k
    }

    GS = GrayScott(args.N, args.dt, args.dx, consts)
    seed_grid(GS, args.A, args.u_init, args.v_init)
    GS.run(args.iterations)

    return GS

def figures_part_3():
    """Creating the figures for part 3. of the assignment.

    Using mock arguments seems a bit convoluted, however since run_rd.py was initially
    created to run with CLI arguments this method was chosen.

    """
    print("\nCreating Figures for part 3 of the assignment")
    print("   Generating Time Evolution of Base Case ...")
    args = ["run_rd.py", "-p", "evolution"]
    with patch("sys.argv", args):
        GS = setup_and_run_gs()

        _ = plot_concs(GS)
        plt.savefig(FDIR / "RD_baseline_evolution.png", dpi=300)
        plt.show()

    print("   Generating Final of Base Case ...")
    args = ["run_rd.py", "-p", "static", "-A", "0.01"]
    with patch("sys.argv", args):
        GS = setup_and_run_gs()

        _ = plot_final_conc(GS)
        plt.savefig(FDIR / "RD_baseline_final_noise.png", dpi=300)
        plt.show()

    print("   Generating Final of Mitosis ...")
    args = ["run_rd.py", "-p", "static", "-A", "0.01", "-f", "0.028", "-k", "0.062", "--u_init", "1"]
    with patch("sys.argv", args):
        GS = setup_and_run_gs()

        _ = plot_final_conc(GS)
        plt.savefig(FDIR / "RD_mitosis_final_noise.png", dpi=300)
        plt.show()

    print("   Generating Final of Solitons ...")
    args = ["run_rd.py", "-p", "static", "-A", "0.01", "-f", "0.03", "-k", "0.06", "--u_init", "1"]
    with patch("sys.argv", args):
        GS = setup_and_run_gs()

        _ = plot_final_conc(GS)
        plt.savefig(FDIR / "RD_solitons_final_noise.png", dpi=300)
        plt.show()

    print("   Generating Final of Flower ...")
    args = ["run_rd.py", "-p", "static", "-A", "0.01", "-f", "0.055", "-k", "0.062", "--u_init", "1"]
    with patch("sys.argv", args):
        GS = setup_and_run_gs()

        _ = plot_final_conc(GS)
        plt.savefig(FDIR / "RD_flower_final_noise.png", dpi=300)
        plt.show()

def plot_animations():
    """If flag '--animations' is given when running this file, only a few animations will
    be plotted."""

    print("Creating (not-required) Animations for the Assignment.")

    print("   Animation of DLA ...")
    _ = animate_growth()
    plt.show()

    print("   Animation of Base Case ...")
    args = ["run_rd.py", "-p", "animate"]
    with patch("sys.argv", args):
        GS = setup_and_run_gs()

        _ = animate_conc(GS)
        plt.show()

    print("   Animation of Mitosis ...")
    args = ["run_rd.py", "-p", "static", "-A", "0.01", "-f", "0.028", "-k", "0.062", "--u_init", "1"]
    with patch("sys.argv", args):
        GS = setup_and_run_gs()

        _ = animate_conc(GS)
        plt.show()

if __name__ == "__main__":
    args = parse_args_main()

    if args.animations:
        plot_animations()
    else:
        figures_part_1()
        figures_part_2()
        figures_part_3()
