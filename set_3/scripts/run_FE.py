import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..utils.config import *  # noqa: F403
from ..utils.FE import FE

FDIR = Path(__file__).parent.parent / "figures"
Path.mkdir(FDIR, exist_ok=True)

def run_validation_sweep():
    target_res = [10, 20, 50, 100, 150, 200]
    D = 0.1
    U = 1.0
    
    re_arr = np.zeros(len(target_res))
    div_arr = np.zeros(len(target_res))
    st_arr = np.zeros(len(target_res))
    
    for i, Re in enumerate(target_res):
        print(f"Running simulation for Re = {Re}")
        start_time = time.time()
        nu = U * D / Re
        ns = FE(tau=0.0005, nu=nu, maxh=0.03)
        ns.run(t_end=10, sample_freq=200) 
        
        div_norm = ns.calc_divergence_norm()
        St = ns.get_strouhal_number(D, U)
        
        re_arr[i] = Re
        div_arr[i] = div_norm
        st_arr[i] = St
        end_time = time.time()
        print(f"    Re: {Re}, L_inf-Div: {div_norm:.2e}, St: {St:.3f}, Time: {end_time - start_time:.2f}s")


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
        axes[0].scatter(re_arr[invalid_idcs], np.min(div_arr[~invalid_idcs]), marker="x", color="tab:gray", label="Diverged")
        axes[1].scatter(re_arr[invalid_idcs], 0, marker="x", color="tab:gray", label="Diverged")

    # Add the Schäfer-Turek Benchmark Point
    axes[1].scatter([100], [0.300], marker='*', s=200, color='gold', edgecolor='black', 
                label='DFG 2D-2 Benchmark', zorder=5)
    
    axes[0].set_yscale('log')
    axes[0].set_title("Divergence vs. Reynolds Number")
    axes[0].set_xlabel("Reynolds Number (Re)")
    axes[0].set_ylabel("$L_{\\infty}$-Norm of Divergence")
    axes[0].legend(fancybox=True, shadow=True, loc='upper left')

    axes[1].set_title("Strouhal Number Validation")
    axes[1].set_xlabel("Reynolds Number (Re)")
    axes[1].set_ylabel("Strouhal Number (St)")
    axes[1].legend(fancybox=True, shadow=True)

    plt.suptitle("Divergence and Strouhal Number for FE implementation")
    
    plt.savefig(FDIR / "challenge_A_FE.png", bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    run_validation_sweep()

