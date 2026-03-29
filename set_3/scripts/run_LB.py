import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from ..utils.config import *  # noqa: F403
from ..utils.LB import LB

def run_validation_sweep():
    FDIR = Path(__file__).parent.parent / "figures"
    Path.mkdir(FDIR, exist_ok=True)
    target_res = [10, 20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    D = 0.1
    U = 1.0
    
    re_arr = np.full(len(target_res), np.nan)
    div_arr = np.full(len(target_res), np.nan)
    st_arr = np.full(len(target_res), np.nan)

    for i, Re in enumerate(target_res):
        print(f"Running simulation for Re = {Re}")
        start_time = time.time()
        nu = U * D / Re
        lb = LB(dx=.005, dt=.00005, rho=1.0, nu=nu)
        lb.run(time=5)
        lb.run(time=5, probe=100)
        
        div = lb.max_divergence(lb.u, lb.v, lb.mask, lb.dt)
        St = lb.strouhal(probe=100)
        
        re_arr[i] = Re
        if div < 1e3:
            div_arr[i] = div
            st_arr[i] = St
        end_time = time.time()
        print(f"    Re: {Re}, Max-Div: {div:.2e}, St: {St:.3f}, Time: {end_time - start_time:.2f}s")


    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    
    invalid_idcs = np.isnan(div_arr)

    # Subplot 1: divergence
    axes[0].plot(re_arr[~invalid_idcs], div_arr[~invalid_idcs], marker='o', linestyle='-', color='darkcyan', 
                 label="Stable Runs")
    
    # Subplot 2: physical accuracy
    axes[1].scatter(re_arr[~invalid_idcs], st_arr[~invalid_idcs], facecolors='none', edgecolors='firebrick', 
                linewidths=2, s=80, label='Simulated (Parabolic Inlet)', zorder=10)
    
    # markers for diverged runs
    if sum(invalid_idcs) > 0:
        axes[0].scatter(re_arr[invalid_idcs], np.tile(np.min(div_arr[~invalid_idcs]), sum(invalid_idcs)), marker="x", color="tab:gray", label="Diverged")
        axes[1].scatter(re_arr[invalid_idcs], np.tile(0, sum(invalid_idcs)), marker="x", color="tab:gray", label="Diverged")

    # Add the Schäfer-Turek Benchmark Point
    axes[1].scatter([100], [0.300], marker='*', s=200, color='gold', edgecolor='black', 
                label='DFG 2D-2 Benchmark', zorder=5)
    
    axes[0].set_yscale('log')
    axes[0].set_title("Divergence vs. Reynolds Number")
    axes[0].set_xlabel("Reynolds Number (Re)")
    axes[0].set_ylabel("Maximum Divergence")
    axes[0].legend(fancybox=True, shadow=True, loc='upper left')

    axes[1].set_title("Strouhal Number Validation")
    axes[1].set_xlabel("Reynolds Number (Re)")
    axes[1].set_ylabel("Strouhal Number (St)")
    axes[1].legend(fancybox=True, shadow=True)

    plt.suptitle("Divergence and Strouhal Number for LB implementation")
    plt.savefig(FDIR / "challenge_A_LB.png", bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    run_validation_sweep()