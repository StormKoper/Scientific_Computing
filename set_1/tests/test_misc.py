import numpy as np
from utils.misc import analytical_concentration


def test_concenctration_t_1():
    """Check that the analytical concentration is close to linear at t=1."""
    y_vals = np.linspace(0, 1, 100)
    c_vals = np.array([analytical_concentration(y, 1, 1, 3000) for y in y_vals])
    assert np.allclose(c_vals, y_vals, atol=1e-2)