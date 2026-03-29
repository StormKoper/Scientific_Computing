import numpy as np
from utils.FD import FD
from numba import njit, prange

def test_boundaries():
    fd = FD(dx=0.05, dy=0.05, dt=0.001, rho=1.0, nu=0.1, omega=1.5)
    fd.run(time=0.1)

    # 1. Test Inlet (Dirichlet U=1.0, V=0.0)
    assert np.all(fd.u[:2, :] == 1.0), f"Inlet U velocity should be strictly 1.0"
    assert np.all(fd.v[:2, :] == 0.0), f"Inlet V velocity should be strictly 0.0"

    # 2. Test Top and Bottom Walls (No-slip Dirichlet U=0.0, V=0.0)
    assert np.all(fd.u[2:, :2] == 0.0), f"Bottom wall U should be 0.0 (No-slip)"
    assert np.all(fd.v[:, :2] == 0.0), f"Bottom wall V should be 0.0 (No-slip)"
    assert np.all(fd.u[2:, -2:] == 0.0), f"Top wall U should be 0.0 (No-slip)"
    assert np.all(fd.v[:, -2:] == 0.0), f"Top wall V should be 0.0 (No-slip)"

    # 3. Test Outlet (Zero-gradient Neumann)
    assert np.array_equal(fd.u[-1, :], fd.u[-2, :]), f"Outlet U should match the column next to it"
    assert np.array_equal(fd.v[-1, :], fd.v[-2, :]), f"Outlet V should match the column next to it"
    assert np.array_equal(fd.u[-2, :], fd.u[-3, :]), f"Outlet U should match the column next to it"
    assert np.array_equal(fd.v[-2, :], fd.v[-3, :]), f"Outlet V should match the column next to it"

    # 4. Test Velocity field inside the cylinder (No-slip)
    assert np.all(fd.u[2:-2, 2:-2][fd.mask[2:-2, 2:-2]] == 0.0), f"Cylinder U should be 0.0 (No-slip)"
    assert np.all(fd.v[2:-2, 2:-2][fd.mask[2:-2, 2:-2]] == 0.0), f"Cylinder V should be 0.0 (No-slip)"

    # 5. Test Pressure isolation inside the cylinder
    assert np.all(fd.p[2:-2, 2:-2][fd.mask[2:-2, 2:-2]] == 0.0), f"Pressure inside the cylinder should remain 0.0"

@njit(parallel=True)
def test_race_condition(a: np.ndarray):
    maximum = 0
    for i in prange(a.shape[0]):
        for j in range(a.shape[1]):
            maximum = max(maximum, a[i, j])
    return maximum

if __name__ == "__main__":
    for _ in range(5):
        a = np.random.rand(1000, 200)
        max_value = np.max(a)
        computed = test_race_condition(a)
        print(f"Max value:    {max_value}")
        print(f"Computed max: {computed}")