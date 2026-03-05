from set_2.utils.MC_DLA import MC_DLA
import matplotlib.pyplot as plt
import numpy as np

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

def single_sim(N=100, ps = 1.0, use_jit=True, seed=None):
    """Runs a single DLA simulation and plots the result."""
    sim = MC_DLA(N, use_jit=use_jit, seed=seed)

    sim.run(grow_until=0.8)
    density = calculate_fractal_density(sim.grid)

    plt.figure(figsize=(8, 8))
    plt.imshow(sim.grid, cmap='Blues', interpolation='nearest')
    plt.title(f"DLA Cluster Size: {N}, Density: {density:.4f}")
    plt.axis('on')
    plt.show()

def sticking_probabilities_sim(N=100, ps_values=[0.01, 0.1, 0.3, 0.7, 1.0], use_jit=True, seed=None):
    """Runs multiple simulations to compare the effect of sticking probability."""
    n_plots = len(ps_values)
    fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * n_plots, n_plots), constrained_layout=True)

    for ax, ps in zip(axes, ps_values):
        print(f"Running simulations for p_s = {ps:.2f}...") 
        sim = MC_DLA(N, seed=seed, p_s=ps, use_jit=use_jit)
        sim.run(grow_until=0.8)
        
        density = calculate_fractal_density(sim.grid)

        ax.imshow(sim.grid, cmap='Blues', interpolation='nearest')
        ax.set_title(f"$p_s$ = {ps}\nDensity={density:.4f}" )
        ax.axis('on')

    plt.suptitle(f"DLA with sticking probabilities ", fontsize=16)
    plt.show()

def MC_density(N=100, ps_values=np.linspace(0.01, 1, 10), n_runs=25, use_jit=True, seed=None):
    """Computes average density for each sticking probability"""
    avg_dvals = []
    std_dvals = []
    
    for ps in ps_values:
        density_runs = []
        print(f"Running simulations for p_s = {ps:.2f}...") 
        for i in range(n_runs):
            # Change seed per run so they aren't identical
            current_seed = seed + i if seed is not None else None
            sim = MC_DLA(N, seed=current_seed, p_s=ps, use_jit=use_jit)
            sim.run(grow_until=0.8)

            density_vals = calculate_fractal_density(sim.grid)
            density_runs.append(density_vals)
        
        # Average and standard dev over runs
        avg_dvals.append(np.mean(density_runs))
        std_dvals.append(np.std(density_runs))

    # Convert lists
    avg_dvals = np.array(avg_dvals)
    std_dvals = np.array(std_dvals)

    plt.figure(figsize=(8, 8))
    plt.plot(ps_values, avg_dvals, marker='o', linestyle='-', markersize=4)
    plt.fill_between(ps_values, avg_dvals - std_dvals, avg_dvals + std_dvals, alpha=0.3, label='$\pm 1$ Std Dev')
    plt.xlabel('Sticking Probability ($p_s$)')
    plt.ylabel('Fractal Density')
    plt.title(f'Average Fractal Density over {n_runs} runs vs Sticking Probabilities ')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    single_sim(N=100, use_jit=True, seed=42)
    sticking_probabilities_sim(N=100, use_jit=True, seed=42)
    MC_density(N=100, use_jit=True, seed=42)