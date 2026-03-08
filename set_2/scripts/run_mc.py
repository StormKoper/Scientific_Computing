import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from set_2.utils.MC_DLA import MC_DLA

from ..utils.config import *  # noqa: F403


def calculate_fractal_density(grid):
    """Calculate density based on the bounding box of the fractal."""
    occupied = np.argwhere(grid == 1)
    if len(occupied) == 0:
        return 0.0
    
    min_x, min_y = occupied.min(axis=0)
    max_x, max_y = occupied.max(axis=0)
    
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    bounding_area = width * height
    
    particle_count = len(occupied)
    density = particle_count / bounding_area
    
    return density

def plot_single_MC(N=100, p_s=1.0, use_jit=True, seed=42, sims=10):
    """Run and visualize an averaged DLA simulation."""
    print(f"Plotting Single with {sims} simulations for p_s = {p_s:.2f}...", end="")
    
    seeds = np.random.SeedSequence(seed).spawn(sims)
    avg_mask = np.zeros((N, N))
    densities = []
    
    for i in tqdm(range(sims), desc=f"p_s={p_s:.2f}", leave=False):
        sim = MC_DLA(N, seed=seeds[i], p_s=p_s, use_jit=use_jit)
        sim.run(grow_until=0.8)
        
        mask = sim.grid
        avg_mask += mask
        densities.append(calculate_fractal_density(mask))
        
    avg_mask /= sims
    avg_density = np.mean(densities)
    std_density = np.std(densities)

    print("\r" + " " * 80 + "\r", end="")

    fig = plt.figure(figsize=(6, 5), constrained_layout=True)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(avg_mask, cmap='Blues', interpolation='nearest')
    plt.title(f"Average MC-DLA Cell Occupation (Density={avg_density:.3f}±{std_density:.3f})")
    return fig

def plot_5_panel_MC(N=100, ps_arr=[0.01, 0.25, 0.5, 0.75, 1.0], use_jit=True, seed=42, sims=10):
    """Run and visualize multiple DLA simuations to compare the effect of eta on growth structure."""
    if len(ps_arr) != 5:
        print(f"{len(ps_arr)} are too many/few p_s-values for a 5-panel, please provide 5.")
        return
    n_plots = len(ps_arr)
    fig, axes = plt.subplots(1, n_plots, figsize=(18, 4), constrained_layout=True)

    seeds = np.random.SeedSequence(seed).spawn(n_plots * sims)
    for i, (ax, ps) in enumerate(zip(axes, ps_arr)): 
        
        avg_mask = np.zeros((N, N))
        densities = []
        
        for j in tqdm(range(sims), desc=f"p_s={ps:.2f}", leave=False):
            current_seed = seeds[i * sims + j]
            sim = MC_DLA(N, seed=current_seed, p_s=ps, use_jit=use_jit)
            sim.run(grow_until=0.8)
            
            mask = sim.grid
            avg_mask += mask
            densities.append(calculate_fractal_density(mask))
            
        avg_mask /= sims
        avg_density = np.mean(densities)
        std_density = np.std(densities)

        ax.imshow(avg_mask, cmap='Blues', interpolation='nearest')
        ax.set_title(f"$p_s$ = {ps}\nDensity={avg_density:.3f}±{std_density:.3f}" )
        ax.set_xticks([])
        ax.set_yticks([])

    print("\r" + " " * 80 + "\r", end="")

    plt.suptitle(f"Average MC-DLA Cell Occupation for Different Sticking Probabilities")
    return fig

def plot_mc_density(N=100, ps_values=np.geomspace(0.03, 1.02, 12) - 0.02, n_runs=25, use_jit=True, seed=42):
    """Computes average density for each sticking probability"""
    avg_dvals = []
    std_dvals = []
    
    for ps in ps_values:
        density_runs = []
        for i in tqdm(range(n_runs), desc=f"p_s={ps:.2f}", leave=False):
            # Change seed per run so they aren't identical
            current_seed = seed + i
            
            sim = MC_DLA(N, seed=current_seed, p_s=ps, use_jit=use_jit)
            sim.run(grow_until=0.8)

            density_vals = calculate_fractal_density(sim.grid)
            density_runs.append(density_vals)
        
        avg_dvals.append(np.mean(density_runs))
        std_dvals.append(np.std(density_runs))

    # Convert lists
    avg_dvals = np.array(avg_dvals)
    std_dvals = np.array(std_dvals)

    fig = plt.figure(figsize=(6, 5), constrained_layout=True)
    plt.xlim((0, 1.05))
    plt.plot(ps_values, avg_dvals, marker='o', linestyle='-', color="darkcyan", markersize=4)
    plt.fill_between(ps_values, avg_dvals - std_dvals, avg_dvals + std_dvals, color="darkcyan", alpha=0.3, label='$\\pm 1$ Std Dev')
    plt.xlabel('Sticking Probability ($p_s$)')
    plt.ylabel('Fractal Density')
    plt.title(f'MC-DLA Density vs Sticking Probability')
    return fig