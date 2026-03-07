

import argparse
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt

from set_2.scripts.run_dla import (
    animate_growth,
    benchmark_dla_jit,
    plot_5_panel,
    plot_dla_density,
    plot_single,
)
from set_2.scripts.run_mc import plot_5_panel_MC, plot_mc_density, plot_single_MC
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

N_SIMS = 1

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
    _ = plot_5_panel(sims=N_SIMS)
    plt.savefig(FDIR / "five_panel_DLA.png", dpi=300)
    plt.show()

    _ = plot_dla_density(n_runs=N_SIMS)
    plt.savefig(FDIR / "DLA_density.png", dpi=300)
    plt.show()

    benchmark_dla_jit()

    _ = plot_single(N=200, sims=N_SIMS)
    plt.savefig(FDIR / "single_DLA_N200.png", dpi=300)
    plt.show()

    plt.close()

def figures_part_2():
    _ = plot_5_panel_MC(sims=N_SIMS)
    plt.savefig(FDIR / "five_panel_MC.png", dpi=300)
    plt.show()

    _ = plot_mc_density(n_runs=N_SIMS)
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
    args = ["run_rd.py", "-p", "evolution"]
    with patch("sys.argv", args):
        GS = setup_and_run_gs()
        
        _ = plot_concs(GS)
        plt.savefig(FDIR / "RD_baseline_evolution.png", dpi=300)
        plt.show()
    
    args = ["run_rd.py", "-p", "static", "-A", "0.01"]
    with patch("sys.argv", args):
        GS = setup_and_run_gs()

        _ = plot_final_conc(GS)
        plt.savefig(FDIR / "RD_baseline_final_noise.png", dpi=300)
        plt.show()

    args = ["run_rd.py", "-p", "static", "-A", "0.01", "-f", "0.028", "-k", "0.062", "--u_init", "1"]
    with patch("sys.argv", args):
        GS = setup_and_run_gs()

        _ = plot_final_conc(GS)
        plt.savefig(FDIR / "RD_mitosis_final_noise.png", dpi=300)
        plt.show()
    
    args = ["run_rd.py", "-p", "static", "-A", "0.01", "-f", "0.03", "-k", "0.06", "--u_init", "1"]
    with patch("sys.argv", args):
        GS = setup_and_run_gs()

        _ = plot_final_conc(GS)
        plt.savefig(FDIR / "RD_solitons_final_noise.png", dpi=300)
        plt.show()

    args = ["run_rd.py", "-p", "static", "-A", "0.01", "-f", "0.055", "-k", "0.062", "--u_init", "1"]
    with patch("sys.argv", args):
        GS = setup_and_run_gs()

        _ = plot_final_conc(GS)
        plt.savefig(FDIR / "RD_flower_final_noise.png", dpi=300)
        plt.show()

def plot_animations():
    # working on this
    args = ["run_rd.py", "-p", "animate"]
    with patch("sys.argv", args):
        GS = setup_and_run_gs()
        
        _ = animate_conc(GS)
        plt.show()
    
    # doesnt work yet
    # _ = animate_growth()
    # plt.show()

if __name__ == "__main__":
    args = parse_args_main()

    if args.animations:
        plot_animations()
    else:
        figures_part_1()
        figures_part_2()
        figures_part_3()