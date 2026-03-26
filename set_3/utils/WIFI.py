import argparse
import time

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from netgen.occ import Glue, MoveTo, OCCGeometry, Rectangle
from ngsolve import *
from scipy.optimize import differential_evolution

from ..utils.config import *  # noqa: F403


def parse_args() -> argparse.Namespace:
    """Parse the arguments

    Returns:
        - (argparse.Namespace): The parsed arguments
    """
    parser = argparse.ArgumentParser(description="Optimize or evaluate WiFi router placement.")

    parser.add_argument(
        "x",
        help="The x-coordinate for the router position (optional)",
        type=float,
        nargs='?', 
        default=None,
    )
    parser.add_argument(
        "y",
        help="The y-coordinate for the router position (optional)",
        type=float,
        nargs='?',
        default=None,
    )
    parser.add_argument(
        "--GUI",
        help="Use NGSolve GUI for visualization instead of matplotlib",
        action="store_true"
    )
    parser.add_argument(
        "--raw",
        help="Use raw signal values and enforce a 0.5m distance from targets",
        action="store_true"
    )
    return parser.parse_args()

# ============================================================================
# 1. SETUP PARAMETERS & PHYSICAL CONSTANTS
# ============================================================================

# We use a physically smaller wavenumber (0.8 GHz) for faster computation
# and less memory usage, this is also in line with the visualization in the
# assingment, hinting at this transformation. However, this does mean our
# solution will be an approximation of the 'true' optimal solution
frequency = 0.8e9
c = 3e8            # Speed of light
k0 = 2 * pi * frequency / c  # wavenumber (eq 21 in assignment)
wavelength = c / frequency

# source parameters
A = 1e4
sigma = 0.2

# ============================================================================
# 2. GEOMETRY DEFINITION
# ============================================================================

# outer domain
air = Rectangle(10, 8).Face()
air.edges.name = "outer"

# walls
kitchen_top = MoveTo(0.0, 3.0).Rectangle(3.0, 0.15).Face()
kitchen_right = MoveTo(2.5, 0.0).Rectangle(0.15, 2.0).Face()
hall_horiz = MoveTo(4.0, 3.0).Rectangle(2.15, 0.15).Face()
living_vert = MoveTo(6.0, 3.15).Rectangle(0.15, 4.85).Face()
bath_top = MoveTo(7.15, 3.0).Rectangle(2.85, 0.15).Face()
bath_left_bottom = MoveTo(7.0, 0.0).Rectangle(0.15, 1.5).Face()
bath_left_top = MoveTo(7.0, 2.5).Rectangle(0.15, 0.65).Face()

# combine walls
walls = kitchen_top + kitchen_right + hall_horiz + living_vert + bath_top + bath_left_bottom + bath_left_top
walls.edges.name = "inner"

# subtract walls from air to prevent overlaps
air = air - walls

# assign materials
air.mat("air")
walls.mat("wall")

# glue them together so the mesher shares nodes at boundaries
shape = Glue([air, walls])

# generate mesh
geo = OCCGeometry(shape, dim=2)

# dynamically set maximum allowed edge-length to be 1/5 of wavelength.
mesh = Mesh(geo.GenerateMesh(maxh=wavelength / 5))

# ============================================================================
# 3. SETUP FUNCTION SPACE & MATERIAL COEFFICIENTS
# ============================================================================

# setup a complex scalar space with third order polynomials as test functions
fes = H1(mesh, complex=True, order=3)

# u is the trial function and v is the test function
u, v = fes.TnT()

# func to create wall spatial mask, s.t. physics can be easily calculated
def rectangle_mask(x_min, x_max, y_min, y_max):
    return (
        IfPos(x - x_min, 1, 0) * IfPos(x_max - x, 1, 0) * IfPos(y - y_min, 1, 0) * IfPos(y_max - y, 1, 0)
    )

is_wall = (
    rectangle_mask(0.0, 3.0, 3.0, 3.15) +     # Kitchen top
    rectangle_mask(2.5, 2.65, 0.0, 2.0) +     # Kitchen right
    rectangle_mask(4.0, 6.15, 3.0, 3.15) +    # Hall horiz
    rectangle_mask(6.0, 6.15, 3.15, 8.0) +    # Living vert
    rectangle_mask(7.15, 10.0, 3.0, 3.15) +   # Bath top
    rectangle_mask(7.0, 7.15, 0.0, 1.5) +     # Bath left bottom
    rectangle_mask(7.0, 7.15, 2.5, 3.15)      # Bath left top
)

# map the physics: if inside a wall, use 2.5 + 0.5j, otherwise use 1.0 (Air)
n_cf = 1.0 + is_wall * ((2.5 + 0.5j) - 1.0)
k_cf = k0 * n_cf

# ============================================================================
# 4. ASSEMBLE SYSTEM MATRIX (PRE-COMPUTED ONCE)
# ============================================================================

# creating the weak form of the Helmholtz equation
# 'BilinearForm' initializes the object which will store
# the LHS of the equation
a = BilinearForm(fes)
a += grad(u) * grad(v) * dx
a += - (k_cf**2) * u * v * dx
a += - 1j * k0 * u * v * ds("outer") # Impedance BC 
a.Assemble()

# Factorize matrix once, the righthandside is the only part
# that will differ each new router position
inv_matrix = a.mat.Inverse(fes.FreeDofs(), inverse="pardiso")

# this object stores the actual values
gfu = GridFunction(fes)

# These are the measuring points as described by the assignment
# that will determine how well a router position performs.
# we will basically probe 4 positions.
targets = [
    (1.0, 5.0), # Living room 
    (2.0, 1.0), # Kitchen 
    (9.0, 1.0), # Bathroom 
    (9.0, 7.0)  # Bedroom 1 
]
r_meas = 0.05 # 5 cm radius
meas_area = np.pi * (r_meas**2)

# precreate a list of masks for the probe points
masks = [IfPos(r_meas**2 - ((x - xt)**2 + (y - yt)**2), 1, 0) for (xt, yt) in targets]


# ============================================================================
# 5. SIGNAL EVALUATION 
# ============================================================================

def evaluate_router_position(x_r, y_r, use_raw=False):
    """
    Evaluates the wave field for a given router position and computes the 
    average signal strength (in relative dB or raw) at predefined target locations.
    """
    # source is given by the Gaussian pulse, here sigma denotes the std
    # of the gaussian pulse, thus at 1sigma, the amplitude has dropped by
    # about 60%, whereas at 3sigma it has dropped to about 1% etc...
    f_source = A * exp(-((x - x_r)**2 + (y - y_r)**2) / (2 * sigma**2))
    
    # assemble the Linear Form (RHS of equation)
    f = LinearForm(fes)
    f += -f_source * v * dx
    f.Assemble()

    # solve the system with pre-computed factorization
    gfu.vec.data = inv_matrix * f.vec
    
    # magnitude of signal (modulus)
    mod_u = sqrt(gfu.real**2 + gfu.imag**2)
    
    scores = []
    for m in masks:
        # calculate the average value inside probe point radius
        linear_avg = Integrate(mod_u * m, mesh) / meas_area
        
        if use_raw:
            scores.append(linear_avg)
        else:
            # convert to relative db
            db_val = 20 * np.log10((linear_avg + 1e-12) / A)
            scores.append(db_val)
        
    # sum the relative dbs or raw values to get a final score
    total_signal = sum(scores)

    return total_signal

# ============================================================================
# 6. OPTIMIZATION FOR BEST ROUTER PLACEMENT
# ============================================================================

def is_inside_wall(x, y, buffer=0.05):
    """
    Checks if a given coordinate (x, y) is inside any of the walls,
    including a safety buffer (in meters).
    """
    walls_bounds = [
        (0.0, 3.0, 3.0, 3.15),    # Kitchen top
        (2.5, 2.65, 0.0, 2.0),    # Kitchen right
        (4.0, 6.15, 3.0, 3.15),   # Hall horiz
        (6.0, 6.15, 3.15, 8.0),   # Living vert
        (7.15, 10.0, 3.0, 3.15),  # Bath top
        (7.0, 7.15, 0.0, 1.5),    # Bath left bottom
        (7.0, 7.15, 2.5, 3.15)    # Bath left top
    ]
    
    for (xmin, xmax, ymin, ymax) in walls_bounds:
        if (xmin - buffer <= x <= xmax + buffer) and (ymin - buffer <= y <= ymax + buffer):
            return True
            
    return False

def is_close_to_target(x, y):
    for tx, ty in targets:
        if np.sqrt((x - tx)**2 + (y - ty)**2) <= 0.5:
            return True
    return False

def objective(pos, use_raw=False):
    """ Scipy minimizes perfectly, so return negative signal. """
    x_r, y_r = pos

    # in wall check and penalty
    if is_inside_wall(x_r, y_r, buffer=0.05):
        return float('inf')
        
    if is_close_to_target(x_r, y_r):
        return float('inf')

    sig = evaluate_router_position(x_r, y_r, use_raw=use_raw)
    return -sig

def optimize_router(use_gui=False, use_raw=False):
    print("Starting optimization using Differential Evolution (global search)...")
    start_time = time.time()
    
    # we restrict possible router positions inside the 10x8 apartment 
    bounds = [(0.5, 9.5), (0.5, 7.5)]
    
    # differential_evolution to avoid local minima
    result = differential_evolution(
        objective,
        args=(use_raw,),
        bounds=bounds,
        strategy='best1bin', 
        popsize=10,          
        mutation=(0.5, 1.0), 
        recombination=0.7,
        maxiter=30,          
        tol=0.01,            
        disp=True,
        workers=1            
    )
    print("Optimization finished.")
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*50)
    print("               OPTIMIZATION COMPLETE")
    print("="*50)
    print(f"Elapsed Time : {elapsed:.2f} s")
    print(f"Evaluations  : {result.nfev}")
    print(f"Best Position: X = {result.x[0]:.2f} m, Y = {result.x[1]:.2f} m")
    
    unit = "raw" if use_raw else "dB"
    print(f"Best Signal  : {-result.fun:.2f} {unit}")
    print("="*50)
    
    print("Evaluating best position for visualization...")
    draw_field_at_position(result.x[0], result.x[1], use_gui=use_gui, use_raw=use_raw)
    
    return result.x

def draw_field_at_position(x_r, y_r, use_gui=False, use_raw=False):
    """ Evaluate and visualize the field for a given router position. """
    total_signal = evaluate_router_position(x_r, y_r, use_raw=use_raw)
    unit = "raw" if use_raw else "dB"
    print(f"Total Signal  : {total_signal:.2f} {unit}")
    if use_gui:
        open_GUI(x_r, y_r)
        input("Press Enter to close visualization and exit...")
    else:
        draw_matplotlib(x_r, y_r)

# ============================================================================
# VISUALIZATION: EITHER GUI OR MATPLOTLIB
# ============================================================================

def compute_db_field(x_r, y_r):
    """Computes the thresholded magnitude of the field in dB and maps it to a GridFunction."""
    print("Processing data for visualization...")
    
    # use the true complex magnitude (Time-averaged power envelope)
    mod_u_vis = Norm(gfu)

    # convert to relative dB scale (divide by log(10), since ngsolve log is natural)
    db_field = 20 * log((mod_u_vis + 1e-12) / A) / log(10.0)

    # project the symbolic math into a GridFunction
    db_gf = GridFunction(H1(mesh, order=3))
    db_gf.Set(db_field)
    
    # calculate bounds logic for GUI/matplotlib consistent visualization limits
    pts = np.array([v.point for v in mesh.vertices])
    x_coords = pts[:, 0]
    y_coords = pts[:, 1]
    
    # extract the physical dB values at those vertices from the new GridFunction
    z_values = db_gf.vec.FV().NumPy()[:mesh.nv]
    
    # calculate the exact geometric distance from every mesh vertex to the router
    distances = np.sqrt((x_coords - x_r)**2 + (y_coords - y_r)**2)
    
    # filter the array to only include points that are >= 3sigma from router singularity
    far_field_db = z_values[distances >= 3*sigma]
    
    # the max for visualisation is just the maximum decibel in this field without
    # router singularity
    max_far_field = np.max(far_field_db)
    min_far_field = max_far_field - 50 #we treat -50 dB below max as noise
    
    return db_gf, min_far_field, max_far_field

def open_GUI(x_r, y_r):
    import netgen.gui
    db_gf, min_val, max_val = compute_db_field(x_r, y_r)
    Draw(db_gf, mesh, "Signal_Strength_dB", min=min_val, max=max_val)

def draw_matplotlib(x_r, y_r):
    db_gf, min_val, max_val = compute_db_field(x_r, y_r)

    print("Extracting data for Matplotlib rendering...")

    # extract the vertex coordinates from the NGSolve mesh
    pts = np.array([v.point for v in mesh.vertices])
    x_coords = pts[:, 0]
    y_coords = pts[:, 1]
    
    # extract the triangles
    tris = np.array([[v.nr for v in el.vertices] for el in mesh.Elements(VOL)])
    
    # extract the physical dB values at those vertices from the new GridFunction
    z_values = db_gf.vec.FV().NumPy()[:mesh.nv]

    # Create the Matplotlib plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    triangulation = mtri.Triangulation(x_coords, y_coords, tris)
    heatmap = ax.tripcolor(triangulation, z_values, 
                           shading='gouraud', cmap='jet', 
                           vmin=min_val, 
                           vmax=max_val)
    
    # layer the walls on top using Matplotlib Rectangles
    wall_patches = [
        mpatches.Rectangle((0.0, 3.0), 3.0, 0.15, facecolor='black'),    # Kitchen top
        mpatches.Rectangle((2.5, 0.0), 0.15, 2.0, facecolor='black'),    # Kitchen right
        mpatches.Rectangle((4.0, 3.0), 2.15, 0.15, facecolor='black'),   # Hall horiz
        mpatches.Rectangle((6.0, 3.15), 0.15, 4.85, facecolor='black'),  # Living vert
        mpatches.Rectangle((7.15, 3.0), 2.85, 0.15, facecolor='black'),  # Bath top
        mpatches.Rectangle((7.0, 0.0), 0.15, 1.5, facecolor='black'),    # Bath left bottom
        mpatches.Rectangle((7.0, 2.5), 0.15, 0.65, facecolor='black')    # Bath left top
    ]
    
    for wall in wall_patches:
        ax.add_patch(wall)

    # plot marker for router
    ax.plot(x_r, y_r, marker="h", color='gray', markersize=15, 
            markeredgecolor='black', alpha=0.9, linestyle='none', label=f'Router {x_r, y_r}')

    for i, (px, py) in enumerate(targets):
        # tiny crosshair
        if i == 0:
            ax.plot(px, py, marker="P", color='darkviolet', markersize=5, alpha=0.8, linestyle='none', label='Targets', zorder=11)
        else:
            ax.plot(px, py, marker="P", color='darkviolet', markersize=5, alpha=0.8, linestyle='none', zorder=11)

    # formatting the plot
    ax.set_aspect('equal')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title(f"WiFi Signal Coverage at {frequency/1e9:.1f} GHz")
    ax.legend(loc='upper right', framealpha=0.9)
    
    fig.colorbar(heatmap, label="Signal Strength (dB)")
    
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    if args.x is not None and args.y is not None:
        print(f"Evaluating router at ({args.x}, {args.y})...")
        draw_field_at_position(args.x, args.y, use_gui=args.GUI, use_raw=args.raw)
    elif args.x is not None or args.y is not None:
        print("Error: Provide BOTH --x and --y coordinates.")
    else:
        optimize_router(use_gui=args.GUI, use_raw=args.raw)