from math import erfc

import numpy as np


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


if __name__ == "__main__":
    pass