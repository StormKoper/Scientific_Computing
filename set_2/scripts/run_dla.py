import itertools
import timeit
from warnings import catch_warnings, simplefilter

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotlib.animation import FuncAnimation
from numba import get_num_threads, set_num_threads
from scipy.stats import ttest_rel
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
    
    for i in tqdm(range(sims), desc=f"eta={eta:.2f}", leave=False):
        sim = DLA(N, seed=seeds[i], eta=eta, use_jit=use_jit)
        sim.run(grow_until=0.8)
        
        mask = ~sim.obj_mask
        avg_mask += mask
        densities.append(calculate_fractal_density(mask))
        
    avg_mask /= sims
    avg_density = np.mean(densities)
    std_density = np.std(densities)

    print("\r" + " " * 80 + "\r", end="")

    fig = plt.figure(figsize=(6, 5), constrained_layout=True)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(avg_mask, cmap='Reds', interpolation='nearest')
    plt.title(f"Average DLA Cell Occupation (n={sims}, {N}x{N})\n80% Growth, $\\eta$ = {eta}, Density={avg_density:.4f}±{std_density:.4f}")
    return fig

def plot_5_panel(N=100, etas=[0, 0.25, 1, 2, 5], use_jit=True, seed=42, sims=10):
    """Run and visualize multiple DLA simuations to compare the effect of eta on growth structure."""
    if len(etas) != 5:
        print(f"{len(etas)} are too many/few eta-values for a 5-panel, please use provide 5.")
        return
    n_plots = len(etas)
    fig, axes = plt.subplots(1, n_plots, figsize=(18, 5), constrained_layout=True)

    seeds = np.random.SeedSequence(seed).spawn(n_plots * sims)
    for i, (ax, eta) in enumerate(zip(axes, etas)): 
        print(f"\rRunning {sims} simulations for eta = {eta:.2f}...", end="")
        
        avg_mask = np.zeros((N, N))
        densities = []
        
        for j in tqdm(range(sims), desc=f"eta={eta:.2f}", leave=False):
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

    print("\r" + " " * 80 + "\r", end="")

    plt.suptitle(f"Average DLA Cell Occupation (n={sims}, {N}x{N}, 80% Growth)")
    return fig

def plot_dla_density(N=100, etas=np.geomspace(0.3, 10.3, 20)-0.3, n_runs=25, use_jit=True, seed=42):
    """Computes and plots average density for normal DLA (eta)."""
    dla_avg, dla_std = [], []
    
    for eta in etas:
        dla_runs = []
        for i in tqdm(range(n_runs), desc=f"eta={eta:.2f}", leave=False):
            current_seed = seed + i if seed else None
            
            sim_dla = DLA(N, seed=current_seed, eta=eta, use_jit=use_jit)
            sim_dla.run(grow_until=0.8)
            dla_runs.append(calculate_fractal_density(~sim_dla.obj_mask))
        
        dla_avg.append(np.mean(dla_runs))
        dla_std.append(np.std(dla_runs))

    dla_avg, dla_std = np.array(dla_avg), np.array(dla_std)

    fig = plt.figure(figsize=(6, 5), constrained_layout=True)
    plt.plot(etas, dla_avg, marker='s', linestyle='-', color="firebrick", markersize=4, label='DLA ($\\eta$)')
    plt.fill_between(etas, dla_avg - dla_std, dla_avg + dla_std, color="firebrick", alpha=0.3, label='$\\pm 1$ Std Dev')
    
    plt.xlabel('$\\eta$')
    plt.ylabel('Fractal Density')
    plt.title(f'DLA Density vs $\\eta$ (N={N}, {n_runs} runs)')
    plt.legend(shadow=True, fancybox=True)
    return fig

def benchmark_dla_jit(N: int = 100, grow_until: float = 0.5, reps: int = 5):
    """Benchmarks the DLA JIT optimization against the base implementation."""
    warmup_dla = DLA(N=N, eta=0.5, use_jit=True, seed=42)
    warmup_dla.run(n_growth=10)

    base_times = np.zeros(reps)
    jit_times = np.zeros(reps)

    for i in range(reps):
        
        # Base benchmark
        base_dla = DLA(N=N, eta=0.5, use_jit=False, seed=100+i)
        start_time = timeit.default_timer()
        base_dla.run(grow_until=grow_until)
        base_time = timeit.default_timer() - start_time
        base_times[i] = base_time

        # JIT benchmark
        jit_dla = DLA(N=N, eta=0.5, use_jit=True, seed=100+i)
        start_time = timeit.default_timer()
        jit_dla.run(grow_until=grow_until)
        jit_time = timeit.default_timer() - start_time
        jit_times[i] = jit_time

    if reps > 1:
        print(f"Results ({reps} reps) - Base: {base_times.mean():.4f} ± {base_times.std():.4f}s, JIT: {jit_times.mean():.4f} ± {jit_times.std():.4f}s, Speedup: {base_times.mean()/jit_times.mean():.2f}x")
    else:
        print(f"Results - Base: {base_times[0]:.4f}s, JIT: {jit_times[0]:.4f}s, Speedup: {base_times[0]/jit_times[0]:.2f}x")

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
    dla = DLA(N=100, eta=1, omega=1.0, use_jit=True, seed=42, save_every=True)
    dla.run(grow_until=0.8)

    cmap = plt.get_cmap('viridis')
    cmap.set_bad(color='white')  # set color for occupied sites

    fig = plt.figure(constrained_layout=True, figsize=(8,8))
    artist = plt.imshow(dla.x_arr[..., 0], cmap=cmap, vmin=0, vmax=1)
    ax = plt.gca()
    ax.set_aspect('equal')
    textbox = ax.text(0.95, 0.95, "Size: 1",
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    def update(frame_idx: int) -> tuple:
        """Update function that is required by FuncAnimation."""
        artist.set_data(dla.x_arr[..., frame_idx])
        textbox.set_text(f"Size: {frame_idx + 2:.0f}")
        return (artist, textbox)

    plt.title("DLA Growth Animation ($\\eta = 1$)")
    plt.xlabel("Space (x)")
    plt.ylabel("Space (y)")

    ani = FuncAnimation(fig, update, frames=dla.x_arr.shape[-1], interval=1, blit=True)
    return ani

def _run_dla_for_heatmap(N: int, eta: float, n_growth: int, seed: int|None = None) -> float:
    """Helper function to run a single DLA simulation for a given eta and return the final density."""
    dla = DLA(N=N, eta=eta, use_jit=True, seed=seed)
    dla.run(n_growth=n_growth)
    return ~dla.obj_mask

def _run_mc_for_heatmap(N: int, p_s: float, n_growth: int, seed: int|None = None) -> float:
    """Helper function to run a single MC simulation for a given p_s and return the final density."""
    mc = MC_DLA(N=N, p_s=p_s, use_jit=True, seed=seed)
    mc.run(n_growth=n_growth)
    return mc.grid

def compare_dla_rw(N: int = 100, n_growth: int = 100, params: int = 10, sims: int = 10):
    """Compare the DLA growth with random walk growth.
    Arguments:
        N: grid size
        n_growth: the number of growth steps
        params: the number of different parameters to sweep for both DLA and random walk
        sims: the number of simulations to run for each parameter setting"""
    set_num_threads(1) # set numba to use a single thread to avoid oversubscription with joblib
    seeds = np.random.SeedSequence(42).spawn(2*sims)
    dla_seeds = seeds[sims:]
    mc_seeds = seeds[:sims]

    # DLA growth
    tasks = itertools.product(np.linspace(0.0, 1.0, params), dla_seeds)
    print("Running DLA simulations in parallel...")
    results = Parallel(n_jobs=-1)(
        delayed(_run_dla_for_heatmap)(N, eta, n_growth, seed)
        for eta, seed in tqdm(tasks, total=params*sims, desc="DLA Simulations")
    )
    dlas = np.array(results).reshape(params, sims, N, N).mean(axis=1)  # average over simulations
    
    # Monte Carlo growth
    tasks = itertools.product(np.logspace(-2.0, 0.0, params), mc_seeds)
    print("Running MC simulations in parallel...")
    results = Parallel(n_jobs=-1)(
        delayed(_run_mc_for_heatmap)(N, p_s, n_growth, seed)
        for p_s, seed in tqdm(tasks, total=params*sims, desc="MC Simulations")
    )
    mcs = np.array(results).reshape(params, sims, N, N).mean(axis=1)  # average over simulations

    # compute mean squared difference between DLA and MC growth
    msd_grid = np.mean(np.square(dlas[:, None, ...] - mcs[None, :, ...]), axis=(2, 3))  # shape (params, params)
    plt.figure(figsize=(6, 5), constrained_layout=True)
    plt.imshow(msd_grid, cmap='viridis_r')
    plt.colorbar(label="Mean Squared Difference")
    plt.title("Mean Squared Difference between DLA and MC Growth")
    plt.xticks(ticks=np.arange(params), labels=[f"{p:.2f}" for p in np.linspace(0.1, 1, params)], rotation=90)
    plt.yticks(ticks=np.arange(params), labels=[f"{eta:.2f}" for eta in np.linspace(0, 1, params)])
    plt.xlabel("MC $p_s$")
    plt.ylabel("DLA $\\eta$")
    plt.show()

def _run_dla_for_omega(N: int, eta: float, omega: float, grow_until: float, bins: int = 1, seed: int|None = None) -> float:
    """Helper function to run a single DLA simulation for a given omega and return the number of iterations to converge."""
    with catch_warnings():
        simplefilter("ignore", category=RuntimeWarning) # also ignore warnings in subprocesses
        iters = np.zeros(bins)
        growths = np.zeros(bins)
        dla = DLA(N=N, eta=eta, omega=omega, save_error=False, use_jit=True, seed=seed)
        for bin in range(bins):
            if dla.run(grow_until=(bin+1)*grow_until/bins, max_steps=50000):
                iters[bin] = dla.iter_count - np.sum(iters[:bin]) # iterations for this bin only
                growths[bin] = np.count_nonzero(~dla.obj_mask) - 1 - np.sum(growths[:bin]) # growth for this bin only
            else:
                break # stop if growth failed (e.g. due to NaN values)
    # return iterations per growth, handle division by zero by returning np.nan for divergent cases
    return np.divide(iters, growths, out=np.full_like(iters, np.nan), where=growths!=0)

def find_optimal_omega(N: int = 100, grow_until: float = 0.9, sims: int = 5, bins: int = 3) -> plt.Figure:
    """Find the optimal omega for SOR iteration of DLA at different eta.
    Arguments:
        N: grid size
        grow_until: the fraction of the grid to grow until
        params: the number of different eta values to sweep
        sims: the number of simulations to run for each eta value
        bins: the number of bins to divide the growth into"""
    etas = [0.0, 0.5, 1.0] # test a few characteristic eta values (0 = no bias, 0.5 = moderate bias, 1.0 = strong bias)
    min_omega, max_omega = 1.0, 1.95 # safely below analytical divergence in empty grid at 2.0
    n_sweep = round((max_omega - min_omega) / 0.05) + 1 # sweep from min_omega to max_omega in steps of 0.05
    omegas = np.linspace(min_omega, max_omega, n_sweep)
    seeds = np.random.SeedSequence(42).spawn(sims)

    tasks = itertools.product(etas, omegas, seeds)
    n_threads = get_num_threads()
    set_num_threads(1) # set numba to use a single thread to avoid oversubscription with joblib
    results = Parallel(n_jobs=-1)(
        delayed(_run_dla_for_omega)(N, eta, omega, grow_until, bins, seed) 
        for eta, omega, seed in tqdm(tasks, total=len(etas)*n_sweep*sims, desc="Parameter Sweep", leave=False)
    )
    set_num_threads(n_threads) # set numba to use the original number of threads

    results_3d = np.array(results).reshape(len(etas), n_sweep, sims, bins)
    valid_mask = np.all(~np.isnan(results_3d), axis=2)
    iterations = np.where(valid_mask, np.mean(results_3d, axis=2), np.nan) # average over simulations
    deviations = np.where(valid_mask, np.std(results_3d, axis=2), np.nan) # std dev over simulations
    min_indices = np.nanargmin(iterations, axis=1)
    # statistical comparison of best omega with neighbors for each eta and bin using paired t-test
    for bin in range(bins):
        perc = ((bin+1)*grow_until/bins)*100
        print(f"\nGrowth until {perc:.1f}%:")
        print("-" * 70)
        print(f"|{'Eta':^10}|{'Best Omega':^15}|{'p-val (Left)':^20}|{'p-val (Right)':^20}|")
        print("-" * 70)
        
        for eta_idx, eta in enumerate(etas):
            best_idx = min_indices[eta_idx, bin]
            best_omega = omegas[best_idx]
            best_iters = results_3d[eta_idx, best_idx, :, bin]
            
            # compare with the left neighbor (omega - 0.05)
            p_left = "N/A"
            if best_idx > 0:
                left_iters = results_3d[eta_idx, best_idx - 1, :, bin]
                if np.any(np.isnan(left_iters)):
                    p_left = "Diverged"
                else:
                    res_left = ttest_rel(best_iters, left_iters, alternative='less')
                    p_left = f"{res_left.pvalue:.4f}"
                    
            # compare with the right neighbor (omega + 0.05)
            p_right = "N/A"
            if best_idx < n_sweep - 1:
                right_iters = results_3d[eta_idx, best_idx + 1, :, bin]
                if np.any(np.isnan(right_iters)):
                    p_right = "Diverged"
                else:
                    res_right = ttest_rel(best_iters, right_iters, alternative='less')
                    p_right = f"{res_right.pvalue:.4f}"
                    
            print(f"|{eta:^10.2f}|{best_omega:^15.3f}|{p_left:^20}|{p_right:^20}|")
        print("-" * 70)

    # plot iterations vs omega for each eta and bin
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0, 0.9, bins))
    
    for bin in range(bins):
        color = colors[bin]
        ax.errorbar(omegas, iterations[0, :, bin], yerr=deviations[0, :, bin], marker='o', linestyle='-', color=color, capsize=3)
        ax.errorbar(omegas, iterations[1, :, bin], yerr=deviations[1, :, bin], marker='s', linestyle='--', color=color, capsize=3)
        ax.errorbar(omegas, iterations[2, :, bin], yerr=deviations[2, :, bin], marker='d', linestyle=':', color=color, capsize=3)
        
    ax.set_xlabel("$\\omega$")
    ax.set_ylabel("Average Iterations to Converge")
    ax.set_title(f"Parameter Sweep for Optimal $\\omega$ ($N={N}$)")
    ax.set_yscale("log")
    # increase density of y ticks
    ax.set_yticks([5, 10, 20, 30, 50, 80])
    from matplotlib.ticker import StrMethodFormatter, NullFormatter
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    ax.yaxis.set_minor_formatter(NullFormatter())
    
    # setup legend: first add entries for eta values, then add entries for growth bins
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', marker='o', linestyle='-', label='$\\eta = 0.0$'),
        Line2D([0], [0], color='black', marker='s', linestyle='--', label='$\\eta = 0.5$'),
        Line2D([0], [0], color='black', marker='d', linestyle=':', label='$\\eta = 1.0$')
    ]
    for bin in range(bins):
        perc = ((bin+1)*grow_until/bins)*100
        legend_elements.append(Line2D([0], [0], color=colors[bin], lw=4, label=f'Growth: {perc:.1f}%'))
        
    ax.legend(handles=legend_elements, loc='upper left', shadow=True, fancybox=True)
    return fig