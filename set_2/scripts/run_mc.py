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
        sim = MC_DLA(N, seed=seed, p_s=ps, use_jit=use_jit)
        sim.run(grow_until=0.8)
        
        density = calculate_fractal_density(sim.grid)

        ax.imshow(sim.grid, cmap='Blues', interpolation='nearest')
        ax.set_title(f"$p_s$ = {ps}\nDensity={density:.4f}" )
        ax.axis('on')

    plt.suptitle(f"DLA with sticking probabilities ", fontsize=16)
    plt.show()

def MC_density(N=100, ps_values = np.linspace(0.1, 1, 50), use_jit=True, seed=None):
    """Computes density for each sticking probability"""
    d_vals = []
    
    for ps in ps_values:
        sim = MC_DLA(N, seed=seed, p_s=ps, use_jit=use_jit)
        sim.run(grow_until=0.8)

        density_vals = calculate_fractal_density(sim.grid)
        d_vals.append(density_vals)

    plt.figure(figsize=(8, 8))
    plt.plot(ps_values, d_vals, marker='o', linestyle='-', markersize=4)
    plt.xlabel('Sticking Probability ($p_s$)')
    plt.ylabel('Fractal Density')
    plt.title('Fractal Density vs Sticking Probability')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    single_sim(N=100, use_jit=True, seed=42)
    sticking_probabilities_sim(N=100, use_jit=True, seed=42)
    MC_density(N=100, use_jit=True, seed=42)