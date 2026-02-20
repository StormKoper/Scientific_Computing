import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

from ..utils.config import *  # noqa: F403
from ..utils.misc import load_target_image
from ..utils.TIDE import SOR, GaussSeidel, Jacobi

# getting root directory of set_1
SET1_ROOT = Path(__file__).parent.parent

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

    J = Jacobi(x0.copy(), save_every=1, use_jit=True)
    G = GaussSeidel(x0.copy(), save_every=1, use_jit=True)
    S = SOR(x0.copy(), save_every=1, omega=1.8, use_jit=True)

    n_iter = 5000
    J.run(n_iter)
    G.run(n_iter)
    S.run(n_iter)
    
    _ = plt.figure(figsize=(12,9), constrained_layout=True)

    # solving the time-independent diffusion equation, is just a line
    y_vals = np.linspace(0, 1, N, dtype=float)
    plt.plot(y_vals, y_vals, linestyle="-", linewidth=1, c='black', label="Analytical Sol")

    J_slice = J.x_arr[::-1, 1, -1]
    plt.plot(y_vals, J_slice, marker='o', linestyle='None', c='firebrick', label="Jacobi",
             markevery=(0, 6), markersize=6)

    G_slice = G.x_arr[::-1, 1, -1]
    plt.plot(y_vals, G_slice, marker='x', linestyle='None', c='darkcyan', label="Gauss-Seidel",
             markevery=(2, 6), markersize=6)

    S_slice = S.x_arr[::-1, 1, -1]
    plt.plot(y_vals, S_slice, marker='^', linestyle='None', c='forestgreen', label="SOR ($\\omega = 1.8$)",
             markevery=(4, 6), markersize=6)
    
    plt.xlabel("y-value")
    plt.ylabel("Concentration (C)")
    plt.legend(fancybox=True, shadow=True)

    plt.title(f"Three Different Iteration Methods vs. Analytical Concentration ({N}x{N}-grid, {n_iter} iters)")
    plt.show()

def plot_convergence_measures():
    """Create a figure for showing the convergence speed of 3 iteration methods."""
    N = 50
    x0 = np.zeros((N, N))
    x0[0, :] = 1

    J = Jacobi(x0.copy(), use_jit=True)
    G = GaussSeidel(x0.copy(), use_jit=True)

    omegas = [1.0, 1.3, 1.6, 1.8, 1.9]
    Ss = [SOR(x0.copy(), omega=omega, use_jit=True) for omega in omegas]

    p_s = np.arange(3, 9).astype(int)

    iter_counts = {
            "Jacobi": [],
            "Gauss_Seidel": [],
            "SOR": {omega: [] for omega in omegas},
        }
    for p in p_s:
        epsilon = 10.0**(-p)

        J.run(epsilon=epsilon)
        G.run(epsilon=epsilon)

        iter_counts["Jacobi"].append(J.iter_count)
        iter_counts["Gauss_Seidel"].append(G.iter_count)

        for omega, solver in zip(omegas, Ss):
            solver.run(epsilon=epsilon)
            iter_counts["SOR"][omega].append(solver.iter_count)
    
    # init figure
    _ = plt.figure(figsize=(10,8), constrained_layout=True)

    plt.plot(p_s, iter_counts['Jacobi'], c='orangered', label='Jacobi')
    plt.plot(p_s, iter_counts['Gauss_Seidel'], c='purple', label='Gauss_Seidel')

    mpl_cmap = mpl.colormaps['winter']
    colors = mpl_cmap(np.linspace(0, 1, len(omegas)))
    custom_cmap = ListedColormap(colors)

    for i, (omega, iters) in enumerate(iter_counts['SOR'].items()):
        plt.plot(p_s, iters, c=custom_cmap(i), linestyle="--", label=f'SOR $\\omega = {omega}$')

    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))

    plt.xlabel("p")
    plt.ylabel("Iterations")
    plt.legend(fancybox=True, shadow=True)

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

    J = Jacobi(x0.copy())
    G = GaussSeidel(x0.copy())

    omegas = [1.0, 1.3, 1.6, 1.8, 1.9]
    Ss = [SOR(x0.copy(), omega=omega) for omega in omegas]

    J.objects(circle_mask)
    G.objects(circle_mask)

    for S in Ss:
        S.objects(circle_mask)

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

def animate_sinks():
    """Animate the iterative process to see the sink in action."""
    N = 50
    n_steps = 1000
    
    x0 = np.zeros((N, N)) 
    x0[0, :] = 1.0 # Top row fixed at C=1

    # Introduce objects
    y, x = np.ogrid[:N, :N]
    center = N // 2
    radius = N // 25
    circle_mask = (x - center)**2 + (y - center)**2 <= radius**2

    J = Jacobi(x0, save_every = 1)
    G = GaussSeidel(x0, save_every= 1)
    S = SOR(x0, save_every= 1, omega = 1.8) 

    J.objects(circle_mask)
    G.objects(circle_mask)
    S.objects(circle_mask)

    sol_J = J.run(n_steps)
    sol_G = G.run(n_steps)
    sol_S = S.run(n_steps)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    
    im1 = ax1.imshow(J.x_arr[..., 0], origin="lower", cmap='magma', vmin=0, vmax=1)
    im2 = ax2.imshow(G.x_arr[..., 0], origin="lower", cmap='magma', vmin=0, vmax=1)
    im3 = ax3.imshow(S.x_arr[..., 0], origin="lower", cmap='magma', vmin=0, vmax=1)

    ax1.set_title("Jacobi")
    ax2.set_title("Gauss-Seidel")
    ax3.set_title("SOR ($\\omega=1.8$)")

    def update(i):
        im1.set_data(J.x_arr[..., min(i, J.x_arr.shape[-1]-1)])
        im2.set_data(G.x_arr[..., min(i, G.x_arr.shape[-1]-1)])
        im3.set_data(S.x_arr[..., min(i, S.x_arr.shape[-1]-1)])
        return im1, im2, im3

    max_frames = max(J.x_arr.shape[-1], G.x_arr.shape[-1], S.x_arr.shape[-1])

    ani = FuncAnimation(fig, update, frames=max_frames, interval=20, blit=True)
    plt.suptitle("Diffusion Comparison with Central Sink")
    plt.show()

def complex_mask():
    """Define a complex mask to be used in the concentration field"""
    N = 50
    mask = np.zeros((N, N), dtype=bool)
    y, x =np.ogrid[:N, :N]

    # Circle in bottom left
    circle_mask = (x - 12)**2 + (y - 12)**2 <= 6**2

    # 2. Square in bottom right
    square_mask = (x >= 30) & (x <= 42) & (y >= 8) & (y <= 20)
    
    # 3. Triangle in center top
    triangle_mask = (y >= 30) & (y <= 40) & \
                    (y - 30 <= 2 * (x - 15)) & \
                    (y - 30 <= -2 * (x - 35))

    # Combine all shapes 
    mask = circle_mask | square_mask | triangle_mask
    return mask

def plot_conc_field(mask):
    """Produce a diffusion plot for any shape under the SOR method"""
    Ny, Nx= mask.shape
    n_steps = 2000

    x0 = np.zeros((Ny, Nx)) 
    x0[0, :] = 1.0 # Top row fixed at C=1

    S = SOR(x0.copy(), save_every= 1, omega = 1.8, use_jit=True) # Update for optimal omega level
    S.objects(mask)

    sol_S = S.run(n_steps)

    # Animation plot
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)    
    im = ax.imshow(S.x_arr[..., 0], cmap='magma', vmin=0, vmax=1)

    ax.set_title("SOR Iteration ($\\omega=1.8$) with Sink")
    ax.set_xlabel("Space (x)")
    ax.set_ylabel("Space (y)")
    plt.colorbar(im, label='Concentration (C)')

    def update(i):
        im.set_data(S.x_arr[..., i])
        return (im,)

    frames = S.x_arr.shape[-1]

    ani = FuncAnimation(fig, update, frames=frames, interval=20, blit=True)
    plt.show()

    # Plot at final step
    fig_snap, ax_snap = plt.subplots(figsize=(6, 6))
    
    # Take the last frame 
    final_frame = S.x_arr[..., -1]
    
    im_snap = ax_snap.imshow(final_frame, cmap='magma', vmin=0, vmax=1)
    plt.colorbar(im_snap, label='Concentration (C)')
    
    ax_snap.set_title("Steady-State Solution (Final Frame)")
    ax_snap.set_xlabel("Space (x)")
    ax_snap.set_ylabel("Space (y)")
    
    plt.show()

def plot_insulation(mask):
    Ny, Nx= mask.shape
    n_steps = 500

    x0 = np.zeros((Ny, Nx)) 
    x0[0, :] = 1.0 # Top row fixed at C=1

    solver = SOR(x0, use_jit=True, save_every = 1)
    solver.objects(mask, insulation=True)

    solver.run(n_steps)

    fig, ax = plt.subplots(figsize=(6, 6))
    artist = ax.imshow(solver.x_arr[..., -1], cmap='magma', vmin=0, vmax=1)
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
        img_path = SET1_ROOT / "images/drain.png"
        my_mask = load_target_image(img_path, 50)
        #my_mask = complex_mask()
        final_field = plot_conc_field(my_mask)
    elif args.question == 'L':
        img_path = SET1_ROOT / "images/difficult_objects.png"
        my_mask = load_target_image(img_path, 50)
        # my_mask = complex_mask()
        insul_field = plot_insulation(my_mask)
    else:
        raise ValueError(f"Invalid question choice: {args.question}")

if __name__ == "__main__":
    main()
