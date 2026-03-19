import time

import matplotlib.pyplot as plt
import numpy as np

from ..utils.config import *  # noqa: F403
from ..utils.FE import FE


def run_validation_sweep():
    target_res = [10, 20, 50, 100, 150, 200]
    D = 0.1
    U = 1.0
    
    re_list = []
    div_list = []
    st_list = []
    
    for Re in target_res:
        print(f"Running simulation for Re = {Re}")
        start_time = time.time()
        nu = U * D / Re
        ns = FE(tau=0.0005, nu=nu, maxh=0.03)
        ns.run(t_end=8, sample_freq=200) 
        
        div_norm = ns.calc_divergence_norm()
        St = ns.get_strouhal_number(D, U)
        
        re_list.append(Re)
        div_list.append(div_norm)
        st_list.append(St)
        end_time = time.time()
        print(f"    Re: {Re}, L2-Div: {div_norm:.2e}, St: {St:.3f}, Time: {end_time - start_time:.2f}s")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Subplot 1: divergence
    axes[0].plot(re_list, div_list, marker='o', linestyle='-', color='tab:blue')
    axes[0].set_yscale('log')
    axes[0].set_title("Divergence vs. Reynolds Number")
    axes[0].set_xlabel("Reynolds Number (Re)")
    axes[0].set_ylabel("L2-Norm of Divergence")
    
    # Subplot 2: physical accuracy
    axes[1].scatter(re_list, st_list, facecolors='none', edgecolors='tab:red', 
                linewidths=2, s=80, label='Simulated (Parabolic Inlet)', zorder=10)
    
    # Add the Schäfer-Turek Benchmark Point
    axes[1].scatter([100], [0.300], marker='*', s=200, color='gold', edgecolor='black', 
                label='DFG 2D-2 Benchmark', zorder=5)
    
    axes[1].set_title("Strouhal Number Validation")
    axes[1].set_xlabel("Reynolds Number (Re)")
    axes[1].set_ylabel("Strouhal Number (St)")
    axes[1].legend(fancybox=True, shadow=True)
    
    axes[1].set_title("Physical Accuracy: Strouhal Number Validation")
    axes[1].set_xlabel("Reynolds Number (Re)")
    axes[1].set_ylabel("Strouhal Number (St)")
    axes[1].legend(fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_validation_sweep()

