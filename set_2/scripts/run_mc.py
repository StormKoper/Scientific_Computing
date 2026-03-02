from set_2.utils.MC_DLA import MC_DLA
import matplotlib.pyplot as plt

# Plot the simulation
size = 100
sim = MC_DLA(size)

# Add 500 successful walkers
max_particles = 500

while sim.particles_count < max_particles:
    sim.add_walker()
    
# Plot
plt.figure(figsize=(8, 8))
plt.imshow(sim.grid, cmap='Blues', interpolation='nearest')
plt.title(f"DLA Cluster ({sim.particles_count} particles)")
plt.show()

# Plotting simulation with sticking probability
size = 100
sim = MC_DLA(size)
ps = 0.1

# Add 500 successful walkers
max_particles = 50

while sim.particles_count < max_particles:
    sim.add_walker_ps(ps)
    
# Plot
plt.figure(figsize=(8, 8))
plt.imshow(sim.grid, cmap='Blues', interpolation='nearest')
plt.title(f"DLA Cluster ({sim.particles_count} particles with sticking probability $p_s$ = {ps})")
plt.show()