import numpy as np
from utils.FD import FD

def test_fd():
    fd = FD(dx=0.05, dy=0.05, dt=0.001, rho=1.0, nu=0.1)
    fd.run(time=1.0)
    assert np.all(fd.u[fd.cylinder_mask] == 0.0), "Velocity inside the cylinder should be zero"
    assert np.all(fd.v[fd.cylinder_mask] == 0.0), "Velocity inside the cylinder should be zero"
    assert np.all(fd.p[fd.cylinder_mask] == 0.0), "Pressure inside the cylinder should be zero"
    assert np.all(fd.u[0, :] == 1.0), "Inflow velocity at the left boundary should be 1.0"
    assert np.all(fd.u[-1, :] == 0.0), "Velocity at the right boundary should be 0.0"
    assert np.all(fd.u[1:, 0] == 0.0), "Velocity at the bottom boundary should be 0.0"
    assert np.all(fd.u[1:, -1] == 0.0), "Velocity at the top boundary should be 0.0"
    assert np.all(fd.u[1:-1, 1] > 0.0), "Velocity near the inflow should be positive due to perturbation"