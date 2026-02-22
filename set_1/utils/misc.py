from collections.abc import Callable
from math import erfc
from scipy.integrate import quad
from pathlib import Path

import numpy as np
from PIL import Image


def analytical_concentration(y: float, t: float, D: float, sum_iters: int = 1000) -> float:
    """Calculate the concentration of time-dependent diffusion.
    
    Args:
        - y (float): The value (vertical) for which to calculate the concentration.
        - t (float): The time at which to calculate the concentration.
        - D (float): Diffusion constant.
        - sum_iters (int) = 1000: The number of items to include in the summation to infinity.
    
    Returns:
        - float: the concentration based on analytical solution formula for 1D case.
    
    """
    total = 0
    for i in range(sum_iters):
        left_erfc_input = (1 - y + 2*i) / (2 * np.sqrt(D*t))
        right_erfc_input = (1 + y + 2*i) / (2 * np.sqrt(D*t))

        total += erfc(left_erfc_input) - erfc(right_erfc_input)
    
    return total

def get_fourier_coefficients(n_terms=100):
    """Calculate the Fourier coefficients for the initial condition in case 'iii' (sin(5*pi*x) if 1/5 < x < 2/5, else 0). Returns an array of the first n_terms coefficients."""
    An = np.zeros(n_terms)
    for n in range(1, n_terms + 1):
        # explicitly handle n=5 to avoid numerical integration oddities
        if n == 5:
            An[n-1] = 0.2  # analytical result for n=5
        else:
            integrand = lambda x_val: np.sin(5 * np.pi * x_val) * np.sin(n * np.pi * x_val)
            integral_val, _ = quad(integrand, 1/5, 2/5)
            An[n-1] = 2 * integral_val
    return An

def analytical_wave1D(x: np.ndarray, t: float, case: str, c: float = 1.0, An: np.ndarray = None) -> np.ndarray:
    """Calculate the analytical amplitude of a 1D wave based on the initial condition case. Assumes x is in [0, 1] and t >= 0. For case 'i' and 'ii', the analytical solution is straightforward. For case 'iii', the solution is given by a Fourier series expansion, and the first n_terms coefficients must be provided in An."""
    # initial condition: sin(2*pi*x)
    if case == "i":
        return np.sin(2*np.pi*x) * np.cos(2*np.pi*c*t)
    
    # initial condition: sin(5*pi*x)
    if case == "ii":
        return np.sin(5*np.pi*x) * np.cos(5*np.pi*c*t)
    
    # initial condition: sin(5*pi*x) if 1/5 < x < 2/5, else 0
    if case == "iii":
        if An is None:
            raise ValueError("An must be provided for case 'iii'")
        # natural number array for the Fourier series terms
        n = np.arange(1, len(An) + 1)
        # 2D array for the spatial term using outer product to efficiently compute sin(pi*n*x) for each n and x
        sin_term = np.sin(np.pi * np.outer(x, n))
        # 1D array for the temporal term for each n
        cos_term = np.cos(np.pi*n*c*t)
        # multiply the coefficients, spatial terms, and temporal terms
        # broadcasting handles the dimensions, then we sum along the 'n' axis (axis=1)
        psi = np.sum(An * sin_term * cos_term, axis=1)
        return psi
    else:
        raise ValueError(f"Invalid case: {case}. Must be one of 'i', 'ii', or 'iii'.")

def load_target_image(image_path: Path, grid_size: int) -> np.ndarray:
    """Load an image, to utilize as a mask for objects

    Args:
        image_path (Path): path to the target image.
        grid_size (int): size of the grid to which the image is
            to be converted.

    Returns:
        np.ndarray: The output mask as a numpy array

    """
    if not image_path.exists():
        error = f"Image not found at {image_path}"
        raise FileNotFoundError(error)

    # load and process
    with Image.open(image_path) as raw_img:
        img = raw_img.convert("L")
    img = img.resize((grid_size, grid_size), Image.Resampling.LANCZOS)

    # convert to numpy and normalize
    object_mask = np.array(img, dtype=np.float32) / 255.0

    # we need black=1.0, and white=0
    object_mask = 1.0 - object_mask

    # force either 0 or 1
    object_mask = (object_mask >= 0.5).astype(bool)

    return object_mask


if __name__ == "__main__":
    from pathlib import Path

    import matplotlib.pyplot as plt
    set1_root = Path(__file__).parent.parent
    images = set1_root / "images"
    path = images / "drain.png"

    object_mask = load_target_image(path, 50)
    
    plt.imshow(object_mask)
    plt.show()