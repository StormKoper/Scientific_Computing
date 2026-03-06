import itertools
import timeit

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotlib.animation import FuncAnimation
from numba import set_num_threads
from tqdm import tqdm

from set_2.scripts.run_mc import calculate_fractal_density

from ..utils.config import *  # noqa: F403
from ..utils.DLA import DLA
from ..utils.MC_DLA import MC_DLA


def plot_single(N=100, eta=0.3, use_jit=True, seed=42, sims=10):
    """Run and visualize an averaged DLA simulation."""
    print(f"Plotting Single with {sims} simulations for eta = {eta:.2f}...", end="")
    
    seeds = np.random.SeedSequence(seed).spawn(sims)
    avg_mask = np.zeros((N, N))
    densities = []
    
    for i in range(sims):
        sim = DLA(N, seed=seeds[i], eta=eta, use_jit=use_jit)
        sim.run(grow_until=0.8)
        
        mask = ~sim.obj_mask
        avg_mask += mask
        densities.append(calculate_fractal_density(mask))
        
    avg_mask /= sims
    avg_density = np.mean(densities)
    std_density = np.std(densities)

    print("\r", end="")


    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(avg_mask, cmap='Greens', interpolation='nearest')
    plt.title(f"Average DLA Cell Occupation (n={sims}, {N}x{N}, 80% Growth)\n$\\eta$ = {eta}, Density={avg_density:.4f}±{std_density:.4f}")
    plt.show()

def plot_5_panel(N=100, etas=[0, 0.25, 0.5, 0.75, 1.0], use_jit=True, seed=42, sims=10):
    """Run and visualize multiple DLA simuations to compare the effect of eta on growth structure."""
    if len(etas) != 5:
        print(f"{len(etas)} are too many/few eta-values for a 5-panel, please use provide 5.")
        return
    n_plots = len(etas)
    _, axes = plt.subplots(1, n_plots, figsize=(18, 5), constrained_layout=True)

    seeds = np.random.SeedSequence(seed).spawn(n_plots * sims)
    for i, (ax, eta) in enumerate(zip(axes, etas)): 
        print(f"\rRunning {sims} simulations for eta = {eta:.2f}...", end="")
        
        avg_mask = np.zeros((N, N))
        densities = []
        
        for j in range(sims):
            current_seed = seeds[i * sims + j]
            sim = DLA(N, seed=current_seed, eta=eta, use_jit=use_jit)
            sim.run(grow_until=0.8)
            
            mask = ~sim.obj_mask
            avg_mask += mask
            densities.append(calculate_fractal_density(mask))
            
        avg_mask /= sims
        avg_density = np.mean(densities)
        std_density = np.std(densities)

        ax.imshow(avg_mask, cmap='Reds', interpolation='nearest')
        ax.set_title(f"$\\eta$ = {eta}\nDensity={avg_density:.4f}±{std_density:.4f}" )
        ax.set_xticks([])
        ax.set_yticks([])

    print("\r", end="")

    plt.suptitle(f"Average DLA Cell Occupation (n={sims}, {N}x{N}, 80% Growth)")
    plt.show()

def benchmark_dla_jit(N: int = 100, grow_until: float = 0.5, reps: int = 5):
    """Benchmarks the DLA JIT optimization against the base implementation."""
    print("Warming up JIT optimization...")
    warmup_dla = DLA(N=N, eta=0.5, use_jit=True, seed=42)
    warmup_dla.run(n_growth=10)

    base_times = np.zeros(reps)
    jit_times = np.zeros(reps)

    print(f"Running benchmark for DLA JIT vs Non-JIT (N={N}, grow_until={grow_until:.1f}, {reps} repeats)...")
    for i in range(reps):
        print(f"Repeat {i+1}/{reps}:")
        
        # Base benchmark
        base_dla = DLA(N=N, eta=0.5, use_jit=False, seed=100+i)
        start_time = timeit.default_timer()
        base_dla.run(grow_until=grow_until)
        base_time = timeit.default_timer() - start_time
        base_times[i] = base_time
        print(f"  Base time : {base_time:.4f} seconds")

        # JIT benchmark
        jit_dla = DLA(N=N, eta=0.5, use_jit=True, seed=100+i)
        start_time = timeit.default_timer()
        jit_dla.run(grow_until=grow_until)
        jit_time = timeit.default_timer() - start_time
        jit_times[i] = jit_time
        print(f"  JIT time  : {jit_time:.4f} seconds")

    if reps > 1:
        print(f"Results - Base: {base_times.mean():.4f} ± {base_times.std():.4f}s, JIT: {jit_times.mean():.4f} ± {jit_times.std():.4f}s, Speedup: {base_times.mean()/jit_times.mean():.2f}x\n")
    else:
        print(f"Results - Base: {base_times[0]:.4f}s, JIT: {jit_times[0]:.4f}s, Speedup: {base_times[0]/jit_times[0]:.2f}x\n")

def save_frames(N: int = 100, n_growth: int = 100, interval: int = 10):
    """Save the frames of the DLA growth"""
    dla = DLA(N=N, eta=0.8, use_jit=True, seed=42)

    cmap = plt.get_cmap('viridis')
    cmap.set_bad(color='white')  # set color for occupied sites

    for i in range(0, n_growth+1, interval):
        dla.run(n_growth=interval)
        plt.imshow(dla.x_arr[..., i], cmap=cmap, vmin=0, vmax=1)
        plt.title(f"Concentration Field at Frame {i}")
        plt.xlabel("Space (x)")
        plt.ylabel("Space (y)")
        plt.savefig(f"set_2/results/frame_{i:03d}.png")
        plt.close()

def animate_growth():
    """Animate the growth of the DLA cluster"""
    dla = DLA(N=100, eta=0.5, omega=1.0, use_jit=True, seed=42)
    dla.run(n_growth=250)

    cmap = plt.get_cmap('viridis')
    cmap.set_bad(color='white')  # set color for occupied sites

    fig = plt.figure(constrained_layout=True)
    artist = plt.imshow(dla.x_arr[..., 0], cmap=cmap, vmin=0, vmax=1)
    ax = plt.gca()
    ax.set_aspect('equal')
    title = ax.set_title("Growth Animation - Size 0")

    def update(frame_idx: int) -> tuple:
        """Update function that is required by FuncAnimation."""
        artist.set_data(dla.x_arr[..., frame_idx])
        title.set_text(f"Growth Animation - Size {frame_idx}")
        return (artist, title)

    plt.xlabel("Space (x)")
    plt.ylabel("Space (y)")

    _ = FuncAnimation(fig, update, frames=dla.x_arr.shape[-1], interval=10, blit=True)
    plt.show()

def compare_dla_rw(N: int = 100, grow_until: float = 0.8, params: int = 10, sims: int = 10):
    """Compare the DLA growth with random walk growth.
    Arguments:
        N: grid size
        grow_until: the fraction of the grid to grow until
        params: the number of different parameters to sweep for both DLA and random walk
        sims: the number of simulations to run for each parameter setting"""
    dlas = np.zeros((N, N, params))
    mcs = np.zeros((N, N, params))
    seeds = np.random.SeedSequence(42).spawn(2*sims)
    dla_seeds = seeds[sims:]
    mc_seeds = seeds[:sims]
    # DLA growth
    for i, eta in enumerate(np.linspace(0, 1, params)):
        for seed in tqdm(dla_seeds, desc=f"Running DLA simulations for eta={eta:.2f}", leave=False):
            dla = DLA(N=N, eta=eta, use_jit=True, seed=seed)
            dla.run(grow_until=grow_until)
            dlas[..., i] += ~dla.obj_mask
        dlas[..., i] /= sims # average over simulations
    
    # Monte Carlo growth
    for i, p_s in enumerate(np.linspace(0.1, 1.0, params)):
        for j, seed in enumerate(tqdm(mc_seeds[i*sims:(i+1)*sims], desc=f"Running MC simulations for p_s={p_s:.2f}", leave=False)):
            mc = MC_DLA(N=N, p_s=p_s, use_jit=True, seed=seed)
            mc.run(grow_until=grow_until)
            mcs[..., i] += mc.grid
        mcs[..., i] /= sims # average over simulations

    # compute mean squared difference between DLA and MC growth
    # done in nested loops to avoid saving array of size (N, N, params, params)
    msd_grid = np.zeros((params, params))
    for i in range(params):
        for j in range(params):
            msd_grid[i, j] = np.mean(np.square(dlas[..., i] - mcs[..., j]))
    plt.figure(figsize=(8, 6), constrained_layout=True)
    plt.imshow(msd_grid, cmap='viridis')
    plt.colorbar(label="Mean Squared Difference")
    plt.title("Mean Squared Difference between DLA and MC Growth")
    plt.xticks(ticks=np.arange(params), labels=[f"{p:.2f}" for p in np.linspace(0.2, 1, params)])
    plt.yticks(ticks=np.arange(params), labels=[f"{eta:.2f}" for eta in np.linspace(0, 1, params)])
    plt.xlabel("MC $p_s$")
    plt.ylabel("DLA $\\eta$")
    plt.show()

def _run_dla_for_omega(N: int, eta: float, omega: float, grow_until: float, bins: int = 1, seed: int|None = None) -> float:
    """Helper function to run a single DLA simulation for a given omega and return the number of iterations to converge."""
    iters = np.zeros(bins)
    growths = np.zeros(bins)
    dla = DLA(N=N, eta=eta, omega=omega, save_error=False, use_jit=True, seed=seed)
    for bin in range(bins):
        if dla.run(grow_until=(bin+1)*grow_until/bins):
            iters[bin] = dla.iter_count - np.sum(iters[:bin]) # iterations for this bin only
            growths[bin] = np.count_nonzero(~dla.obj_mask) - 1 - np.sum(growths[:bin]) # growth for this bin only
        else:
            break # stop if growth failed (e.g. due to NaN values)
    return np.divide(iters, growths, out=np.full_like(iters, 100), where=growths!=0) # return iterations per growth, handle division by zero

def find_optimal_omega(N: int = 100, grow_until: float = 0.8, params: int = 10, sims: int = 10, bins: int = 1) -> np.ndarray:
    """Find the optimal omega for SOR iteration of DLA at different eta.
    Arguments:
        N: grid size
        grow_until: the fraction of the grid to grow until
        params: the number of different eta values to sweep
        sims: the number of simulations to run for each eta value"""
    set_num_threads(1) # set numba to use a single thread to avoid oversubscription with joblib
    etas = np.linspace(0, 1, params)
    min_omega, max_omega = 1.0, 1.95 # safely below divergence TODO: add a check for divergence set max_omega to 1.99
    n_sweep = int((max_omega - min_omega) / 0.05) + 1 # sweep from min_omega to max_omega in steps of 0.05
    omegas = np.linspace(min_omega, max_omega, n_sweep)
    seeds = np.random.SeedSequence(42).spawn(sims)

    tasks = itertools.product(etas, omegas, seeds)
    
    print("Running initial parameter sweep in parallel...")
    results = Parallel(n_jobs=-1)(
        delayed(_run_dla_for_omega)(N, eta, omega, grow_until, bins, seed) 
        for eta, omega, seed in tqdm(tasks, total=params*n_sweep*sims, desc="Parameter Sweep")
    )

    print("Processing results and plotting...")
    results_3d = np.array(results).reshape(params, n_sweep, sims, bins)
    iterations = np.mean(results_3d, axis=2) # average over simulations
    y_min = np.min(iterations) * 0.9
    y_max = np.max(iterations) * 1.1

    fig, axes = plt.subplots(1, bins, figsize=(1+7*bins,6), constrained_layout=True, sharey=True)
    plt.suptitle(f"Parameter Sweep for Optimal $\\omega$ ($N={N}$)")
    axes[0].set_ylabel("Number of Iterations to Converge")
    for bin in range(bins):
        axes[bin].plot(omegas, iterations.T[bin], marker='o')
        axes[bin].set_xlabel("$\\omega$")
        axes[bin].set_title(f"Growth until {((bin+1)*grow_until/bins)*100:.1f}%")
        axes[bin].set_yscale("log")
        axes[bin].set_ylim(y_min, y_max)
        axes[bin].legend([f"$\\eta={eta:.2f}$" for eta in etas], fancybox=True, shadow=True, loc='upper left')
    plt.show()
    
    # golden section search for optimal omega
    iterations = np.mean(iterations, axis=2) # average over bins
    n_golden_section = 5
    omegas_gs = [[] for _ in range(params)]
    iterations_gs = [[] for _ in range(params)]
    invphi = (np.sqrt(5) - 1) / 2 # 1/phi

    # find the two omegas that are closest to the optimal omega found in the sweep
    optimal_omega_ids = np.argmin(iterations, axis=1)
    left_bound_ids = np.maximum(optimal_omega_ids - 1, 0)
    right_bound_ids = np.minimum(optimal_omega_ids + 1, len(omegas) - 1)
    a = omegas[left_bound_ids]
    b = omegas[right_bound_ids]
    
    for i, eta in enumerate(etas):
        omegas_gs[i].extend([a[i], b[i]])
        iterations_gs[i].extend([iterations[i, left_bound_ids[i]], iterations[i, right_bound_ids[i]]])
    
    # iterate until we have done n_golden_section iterations
    for _ in tqdm(range(n_golden_section), desc="Golden Section Iterations"):
        c = b - (b - a) * invphi
        d = a + (b - a) * invphi
        # create a list of tasks for the new omegas to test in parallel
        omegas_to_test = np.column_stack((c, d))
        tasks = [
            (etas[i], omegas_to_test[i, j], seed)
            for i in range(len(etas))
            for j in range(2)  # j=0 is c, j=1 is d
            for seed in seeds
        ]
        # run the new tasks in parallel and collect results
        results = Parallel(n_jobs=-1)(
            delayed(_run_dla_for_omega)(N, eta, omega, grow_until, 1, seed) 
            for eta, omega, seed in tasks
        )
        # reshape results to be (params, [c, d], sims) and average over simulations
        results_3d = np.array(results).reshape(params, 2, sims)
        iterations = np.mean(results_3d, axis=2)
        # update the bounds a and b based on the new iterations
        for i in range(params):
            iters_c = iterations[i, 0]
            iters_d = iterations[i, 1]
            omegas_gs[i].extend([c[i], d[i]])
            iterations_gs[i].extend([iters_c, iters_d])
            if iters_c < iters_d:
                b[i] = d[i]
            else:
                a[i] = c[i]

    # the optimal omega is the midpoint of the final interval [a, b]
    optimal_omegas = (a + b) / 2
    # the best omega is the one with the least iterations in the golden section search
    best_omegas = np.zeros(params)
    best_iterations = np.zeros(params)
    for i in range(params):
        best_omegas[i] = omegas_gs[i][np.argmin(iterations_gs[i])]
        best_iterations[i] = np.min(iterations_gs[i])

    plt.figure(figsize=(8, 6), constrained_layout=True)
    for omega, iters, eta in zip(omegas_gs, iterations_gs, etas):
        plt.plot(omega, iters, marker='o', markersize=3, alpha=0.5, linestyle='--', label=f"$\\eta={eta:.2f}$")
    plt.plot(best_omegas, best_iterations, marker='o', c='black', label="Optimal $\\omega$")
    plt.xlabel("$\\omega$")
    plt.ylabel("Number of Iterations to Converge")
    plt.title(f"Golden Section Search for Optimal $\\omega$ ($N={N}$)")
    plt.legend(fancybox=True, shadow=True, loc='upper left')
    plt.yscale("log")
    plt.show()

    # print a little table in terminal of best omega's and corresponding iterations
    print("\r\033[K" + "-"*59)
    print(f"|{'Etas':^15}|{'Best Omegas':^20}|{'Average Iterations':^20}|")
    print("-"*59)
    for eta, omega, iters in zip(etas, best_omegas, best_iterations):
        print(f"|{eta:^15.3f}|{omega:^20.5f}|{iters:^20.3f}|")
    print("-"*59)

    return optimal_omegas

if __name__ == "__main__":
    #find_optimal_omega(N=100, grow_until=0.45, params=5, sims=3, bins=3)
    #compare_dla_rw(N=100, grow_until=0.8, params=5, sims=3)

    # for part (a) where we have to check effect of eta values on structure
    plot_5_panel()
    
    # for part (b) where we have to compare jit to original on 100x100 grid
    benchmark_dla_jit()

    # for part (b) where they say to try it on a larger gridsize
    plot_single(N=200)
