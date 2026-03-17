import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Ensure the parent directory is in the path so we can import from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.LBA import LBA


def plot_velocity(ax, lba, step):
    """Plots the velocity magnitude of the simulation."""
    speed = np.sqrt(lba.ux**2 + lba.uy**2)
    speed[lba.obstacle] = np.nan
    ax.clear()
    ax.imshow(speed.T, origin='lower', cmap='jet', vmin=0, vmax=lba.U_inlet * 2.0, aspect='auto', extent=[0, lba.Nx, 0, lba.Ny])
    ax.set_title(f"Velocity magnitude — step {step}")
    plt.pause(0.01)


def plot_vorticity(ax, lba, step):
    """Plots the vorticity field of the simulation."""
    vorticity = (np.roll(lba.uy, -1, axis=0) - np.roll(lba.uy, 1, axis=0) - 
                 np.roll(lba.ux, -1, axis=1) + np.roll(lba.ux, 1, axis=1))
    vorticity[lba.obstacle] = np.nan
    ax.clear()
    ax.imshow(vorticity.T, origin='lower', cmap='RdBu_r', vmin=-0.04, vmax=0.04, aspect='auto', extent=[0, lba.Nx, 0, lba.Ny])
    ax.set_title(f"Vorticity field — step {step}")
    plt.pause(0.01)


def main():
    # 1. Initialize the LBA solver
    print("Initializing LBA solver...")
    solver = LBA(Nx=300, Ny=120, U_inlet=0.12, Re=150)
    
    # 2. Simulation parameters
    n_steps = 30000
    plot_every = 25
    plot_mode = 'velocity'  # Change to 'vorticity' if desired
    
    # 3. Setup real-time plotting
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=120)

    # 4. Main simulation loop
    print("Starting simulation...")
    for step in range(1, n_steps + 1):
        solver.step()  # Advance the simulation by one timestep using JIT
        
        if step % plot_every == 0:
            if plot_mode == 'vorticity':
                plot_vorticity(ax, solver, step)
            else:
                plot_velocity(ax, solver, step)
                
        if step % 1000 == 0:
            print(f"Step {step}/{n_steps}")
            
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()