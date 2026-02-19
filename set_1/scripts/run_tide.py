import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

from ..utils.config import *  # noqa: F403
from ..utils.TIDE import SOR, GaussSeidel, Jacobi


def parse_args() -> argparse.Namespace:
    """Parse the arguments

    Returns:
        - (argparse.Namespace): The parsed arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-question",
        choices=["H", "I", "J", "K", "L"],
        help="For which question you want to create a plot ['H', 'I', 'J', 'K', 'L']",
        type=str,
        required=True
    )
    return parser.parse_args()

def plot_itermethods_vs_analytical() -> None:
    """Create a figure for showing 3 iteration methods vs Analytical solution."""
    N = 50
    x0 = np.zeros((N, N))
    x0[0, :] = 1

    J = Jacobi(x0)
    G = GaussSeidel(x0)
    S = SOR(x0)

    J.run(500)
    G.run(500)
    S.run(500)

    # init figure and create cmap
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,6), constrained_layout=True)
    mpl_cmap = mpl.colormaps['inferno']

    steps_to_plot = np.arange(0, 501, 50)
    colors = mpl_cmap(np.linspace(0, 1, len(steps_to_plot)))
    custom_cmap = ListedColormap(colors)

    for i, iter in enumerate(steps_to_plot):

        y_vals = np.linspace(0, 1, N, dtype=float)

        J_slice = J.x_arr[::-1, 1, iter]
        ax1.plot(y_vals, J_slice, linestyle="-", c=custom_cmap(i), label=f"Iter: {iter:.0f}")

        G_slice = G.x_arr[::-1, 1, iter]
        ax2.plot(y_vals, G_slice, linestyle="-", c=custom_cmap(i), label=f"Iter: {iter:.0f}")

        S_slice = S.x_arr[::-1, 1, iter]
        ax3.plot(y_vals, S_slice, linestyle="-", c=custom_cmap(i), label=f"Iter: {iter:.0f}")
    
    # solving the time-independent diffusion equation, is just a line
    ax1.plot(y_vals, y_vals, linestyle=":", c='black', label="Analytical Sol")
    ax2.plot(y_vals, y_vals, linestyle=":", c='black', label="Analytical Sol")
    ax3.plot(y_vals, y_vals, linestyle=":", c='black', label="Analytical Sol")
    
    ax1.set_title("Jacobi Iteration")
    ax1.set_xlabel("y-value")
    ax1.set_ylabel("Concentration (C)")
    ax1.legend(fancybox=True, shadow=True)
    
    ax2.set_title("Gauss-Seidel Iteration")
    ax2.set_xlabel("y-value")
    ax2.set_ylabel("Concentration (C)")
    ax2.legend(fancybox=True, shadow=True)

    ax3.set_title("SOR Iteration")
    ax3.set_xlabel("y-value")
    ax3.set_ylabel("Concentration (C)")
    ax3.legend(fancybox=True, shadow=True)

    plt.suptitle(f"3 Different Iteration Methods vs. Analytical Concentration ({N}x{N}-grid)")
    plt.show()

def plot_convergence_measures():
    """Create a figure for showing the convergence speed of 3 iteration methods."""
    N = 50
    x0 = np.zeros((N, N))
    x0[0, :] = 1

    J = Jacobi(x0)
    G = GaussSeidel(x0)

    omegas = [1.0, 1.3, 1.6, 1.8, 1.9]
    Ss = [SOR(x0, save_every=0, omega=omega) for omega in omegas]

    n_steps = 1000

    J.run(n_steps)
    G.run(n_steps)
    [S.run(n_steps) for S in Ss]

    # init figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,6), constrained_layout=True)

    ax1.semilogy(np.arange(1, n_steps+1, 1), J.error_history, c='firebrick')
    ax2.semilogy(np.arange(1, n_steps+1, 1), G.error_history, c="darkcyan")
    
    mpl_cmap = mpl.colormaps['plasma']
    colors = mpl_cmap(np.linspace(0, 1, len(omegas)))
    custom_cmap = ListedColormap(colors)
    for i, (S, omega) in enumerate(zip(Ss, omegas)):
        ax3.semilogy(np.arange(1, n_steps+1, 1), S.error_history, c=custom_cmap(i), label=f"$\\omega={omega}$")
    
    ax1.set_title("Jacobi Iteration")
    ax1.set_xlabel("Iteration Number")
    ax1.set_ylabel("Max Error ($\\epsilon$)")
    
    ax2.set_title("Gauss-Seidel Iteration")
    ax2.set_xlabel("Iteration Number")
    ax2.set_ylabel("Max Error ($\\epsilon$)")

    ax3.set_title("SOR Iteration")
    ax3.set_xlabel("Iteration Number")
    ax3.set_ylabel("Max Error ($\\epsilon$)")
    ax3.legend(fancybox=True, shadow=True)

    plt.suptitle(f"Convergence Speed of 3 Iteration Methods ({N}x{N}-grid)")
    plt.show()

def find_optimal_omega():
    """Find the optimal omega for SOR iteration at different grid sizes."""
    N_values = [10, 20, 50, 100, 200, 500]
    epsilon = 1e-5
    min_omega = 1.0
    max_omega = 2.0

    # sweep over omegas for each N
    n_sweep = 11
    omegas = np.linspace(min_omega, max_omega, n_sweep)
    omegas[-1] = 1.99  # to avoid divergence at omega=2.0
    iterations = np.zeros((len(N_values), n_sweep))
    for i, N in enumerate(N_values):
        print(f"Running omega sweep for N={N}...")
        x0 = np.zeros((N, N))
        x0[0, :] = 1
        Ss = [SOR(x0.copy(), save_every=0, omega=omega, use_jit=True) for omega in omegas]
        [S.run(epsilon=epsilon) for S in Ss]
        iterations[i, :] = [S.iter_count for S in Ss]
        
    plt.plot(omegas, iterations.T, marker='o')
    plt.xlabel("$\\omega$")
    plt.ylabel("Number of Iterations to Converge")
    plt.title(f"Parameter Sweep for Optimal $\\omega$ ($\\epsilon={epsilon:.0e}$)")
    plt.legend([f"N={N}" for N in N_values], fancybox=True, shadow=True, loc='upper left')
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("set_1/results/omega_sweep.png", dpi=300)
    plt.show()
    
    # golden section search for optimal omega
    n_golden_section = 10
    omegas_gs = [[] for _ in range(len(N_values))]
    iterations_gs = [[] for _ in range(len(N_values))]
    invphi = (np.sqrt(5) - 1) / 2 # 1/phi
    optimal_omegas = np.zeros(len(N_values))
    best_omegas = np.zeros(len(N_values))
    best_iterations = np.zeros(len(N_values))
    for i, N in enumerate(N_values):
        print(f"Running golden section search for N={N}...")
        x0 = np.zeros((N, N))
        x0[0, :] = 1
        # find the two omegas that are closest to the optimal omega found in the sweep
        optimal_omega_idx = np.argmin(iterations[i])
        left_bound_idx = max(optimal_omega_idx - 1, 0)
        right_bound_idx = min(optimal_omega_idx + 1, len(omegas) - 1)
        a = omegas[left_bound_idx]
        b = omegas[right_bound_idx]
        omegas_gs[i].append(a)
        omegas_gs[i].append(b)
        iterations_gs[i].append(iterations[i,left_bound_idx])
        iterations_gs[i].append(iterations[i,right_bound_idx])
        # iterate until we have done n_golden_section iterations
        for _ in range(n_golden_section):
            c = b - (b - a) * invphi
            d = a + (b - a) * invphi

            S_c = SOR(x0.copy(), save_every=0, omega=c, use_jit=True)
            S_d = SOR(x0.copy(), save_every=0, omega=d, use_jit=True)

            S_c.run(epsilon=epsilon)
            S_d.run(epsilon=epsilon)

            omegas_gs[i].append(c)
            omegas_gs[i].append(d)
            iterations_gs[i].append(S_c.iter_count)
            iterations_gs[i].append(S_d.iter_count)

            if S_c.iter_count < S_d.iter_count:
                b = d
            else:
                a = c
        # the optimal omega is the midpoint of the final interval [a, b]
        optimal_omegas[i] = (a + b) / 2
        # the best omega is the one with the least iterations in the golden section search
        best_omegas[i] = omegas_gs[i][np.argmin(iterations_gs[i])]
        best_iterations[i] = np.min(iterations_gs[i])
    for o, i, N in zip(omegas_gs, iterations_gs, N_values):
        plt.plot(o, i, marker='o', markersize=3, alpha=0.5, linestyle='--', label=f"N={N}")
    plt.plot(best_omegas, best_iterations, marker='o', c='black', label="Optimal $\\omega$")
    plt.xlabel("$\\omega$")
    plt.ylabel("Number of Iterations to Converge")
    plt.title(f"Golden Section Search for Optimal $\\omega$ ($\\epsilon={epsilon:.0e}$)")
    plt.legend(fancybox=True, shadow=True, loc='upper left')
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("set_1/results/golden_section.png", dpi=300)
    plt.show()

    return optimal_omegas

def plot_sinks():
    """Plotting convergence for each iterative method including objects in the domain"""
    N = 50
    n_steps=1000
    x0 = np.ones((N, N)) * 10.0 # Initialise with high value
    y, x = np.ogrid[:N, :N]
    center_y, center_x = 25, 25
    radius = 10
    circle_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2 # Circle in domain

    J = Jacobi(x0)
    G = GaussSeidel(x0)

    omegas = [1.0, 1.3, 1.6, 1.8, 1.9]
    Ss = [SOR(x0.copy(), omega=omega) for omega in omegas]

    J.objects(circle_mask)
    G.objects(circle_mask)

    for S in Ss:
        S.objects(circle_mask)

    sol_J = J.run(n_steps)
    sol_G = G.run(n_steps)
    sol_S = [S.run(n_steps) for S in Ss]

    # init figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,6), constrained_layout=True)

    ax1.semilogy(np.arange(1, n_steps+1, 1), J.error_history, c='firebrick')
    ax2.semilogy(np.arange(1, n_steps+1, 1), G.error_history, c="darkcyan")
    
    mpl_cmap = mpl.colormaps['plasma']
    colors = mpl_cmap(np.linspace(0, 1, len(omegas)))
    custom_cmap = ListedColormap(colors)
    for i, (S, omega) in enumerate(zip(Ss, omegas)):
        ax3.semilogy(np.arange(1, n_steps+1, 1), S.error_history, c=custom_cmap(i), label=f"$\\omega={omega}$")
    
    ax1.set_title("Jacobi Iteration")
    ax1.set_xlabel("Iteration Number")
    ax1.set_ylabel("Max Error ($\\epsilon$)")
    
    ax2.set_title("Gauss-Seidel Iteration")
    ax2.set_xlabel("Iteration Number")
    ax2.set_ylabel("Max Error ($\\epsilon$)")

    ax3.set_title("SOR Iteration")
    ax3.set_xlabel("Iteration Number")
    ax3.set_ylabel("Max Error ($\\epsilon$)")
    ax3.legend(fancybox=True, shadow=True)

    plt.suptitle(f"Convergence Speed of 3 Iteration Methods ({N}x{N}-grid)")
    plt.show()

def animate_sinks():
    """Animate the iterative process to see the sink in action."""
    N = 50
    n_steps = 5000 
    
    x0 = np.zeros((N, N)) 
    x0[-1, :] = 1.0 # Top row fixed at C=1
    y, x = np.ogrid[:N, :N]
    center = N // 2
    radius = N // 5
    circle_mask = (x - center)**2 + (y - center)**2 <= radius**2

    solver = Jacobi(x0, save_every = 2)
    solver.objects(circle_mask) 

    solver.run(n_steps)

    fig, ax = plt.subplots(figsize=(6, 6))
    artist = ax.imshow(solver.x_arr[..., 0], origin="lower", cmap='magma', vmin=0, vmax=1)
    
    plt.colorbar(artist, label='Concentration (C)')

    ax.set_title("Diffusion using Jacobi with objects")
    plt.xlabel("Grid X")
    plt.ylabel("Grid Y")

    def update(frame_idx: int):
        artist.set_data(solver.x_arr[..., frame_idx])
        return (artist,)

    anim = FuncAnimation(fig, update, frames=solver.x_arr.shape[-1], interval=20, blit=True)
    
    plt.show()

def plot_isolation():
    N = 50
    n_steps = 5000 
    
    x0 = np.zeros((N, N)) 
    x0[-1, :] = 1.0 # Top row fixed at C=1
    y, x = np.ogrid[:N, :N]
    center = N // 2
    radius = N // 5
    circle_mask = (x - center)**2 + (y - center)**2 <= radius**2

    solver = SOR(x0, use_jit=True, save_every = 2)
    solver.objects(circle_mask, insulation=True)

    solver.run(n_steps)

    fig, ax = plt.subplots(figsize=(6, 6))
    artist = ax.imshow(solver.x_arr[..., -1], origin="lower", cmap='magma', vmin=0, vmax=1)
    plt.show()

def main():
    """Entry point when run as a script.

    CLI arguments:
        -question (str): One of 'H', 'I', or 'J'.
            - H: plot 3 iteration methods vs analytical solution.
            - I: plot convergence speed of 3 iteration methods.
            - J: plot sweep and golden section search for optimal omega.
    """
    args = parse_args()

    if args.question == 'H':
        plot_itermethods_vs_analytical()
    elif args.question == 'I':
        plot_convergence_measures()
    elif args.question == 'J':
        find_optimal_omega()
    elif args.question == 'K':
        plot_sinks()
        animate_sinks()
    elif args.question == 'L':
        plot_isolation()
    else:
        raise ValueError(f"Invalid question choice: {args.question}")

if __name__ == "__main__":
    main()


