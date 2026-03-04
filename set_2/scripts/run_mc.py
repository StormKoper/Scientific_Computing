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

def single_sim(size=100, max_steps=1000, use_jit=True, seed=None):
    """Runs a single DLA simulation and plots the result."""
    sim = MC_DLA(size, seed=seed)
        
    for _ in range(max_steps):
        if use_jit:
            sim.add_walker_jit()
        else:
            sim.add_walker()

        # Stop if cluster reaches the top
        if sim.grid[0, :].any():
            break
    
    density = calculate_fractal_density(sim.grid)

    plt.figure(figsize=(8, 8))
    plt.imshow(sim.grid, cmap='Blues', interpolation='nearest')
    plt.title(f"DLA Cluster Size: {size}, N: {sim.particles_count}, Density: {sim.particles_count/(size * size)}")
    plt.axis('on')
    plt.show()

def sticking_probabilities_sim(size=100, max_steps=1000, ps_values=[0.1, 0.3, 0.7, 1.0], use_jit=True, seed=None):
    """Runs multiple simulations to compare the effect of sticking probability."""
    n_plots = len(ps_values)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4), constrained_layout=True)

    for ax, ps in zip(axes, ps_values):
        sim = MC_DLA(size, seed=seed)
        
        for _ in range(max_steps):
            if use_jit:
                sim.add_walker_ps_jit(ps)
            else:
                sim.add_walker_ps(ps)

            # Stop if cluster reaches the top
            if sim.grid[0, :].any():
                break
        
        density = calculate_fractal_density(sim.grid)

        ax.imshow(sim.grid, cmap='Blues', interpolation='nearest')
        ax.set_title(f"$p_s$ = {ps}\nN={sim.particles_count} \nDensity={sim.particles_count/(size * size)}" )
        ax.axis('on')

    plt.suptitle(f"DLA sticking probabilities comparison ", fontsize=16)
    plt.show()

if __name__ == "__main__":
    np.random.seed(42) # For JIT implementation
    single_sim(size=100, max_steps=150000, use_jit=True, seed=42)
    sticking_probabilities_sim(size=100, max_steps=150000, use_jit=True, seed=42)