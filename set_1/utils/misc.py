from collections.abc import Callable
from math import erfc
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

# NOT FINALIZED, IDK YET HOW TO ALTER FOR OUR DIRICHELET BOUNDARIES
def analytical_1D_wave(x: np.ndarray, t: float, f: Callable, c: float = 1.0) -> np.ndarray:
    """NOT FINALIZED, IDK YET HOW TO ALTER FOR OUR DIRICHELET BOUNDARIES
    
    Calculate the analytical amplitude of a 1D wave.
    Function is based on solution by d'Alembert ommiting initial velocity component.
    
    Args:
        - x (np.ndarray): Points in 1D space to calculate the amplitude for.
        - t (float): Time at which to calculate amplitude.
        - f (Callable): Function that was used to create initial condition X0.
        - c (float): Propagation velocity of wave.
    
    """
    return 0.5 * (f(x - c*t) + f(x + c*t))

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