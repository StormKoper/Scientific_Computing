import numpy as np
from utils.FD import FD

def test_boundaries():
    for P in [0, 1, 2]:
        fd = FD(dx=0.05, dy=0.05, dt=0.001, rho=1.0, nu=0.1)
        fd.run(time=0.1, P=P)

        # 1. Test Inlet (Dirichlet U=1.0, V=0.0)
        assert np.all(fd.u[0, :] == 1.0), f"{P=}: Inlet U velocity should be strictly 1.0"
        assert np.all(fd.v[0, :] == 0.0), f"{P=}: Inlet V velocity should be strictly 0.0"

        # 2. Test Top and Bottom Walls (No-slip Dirichlet U=0.0, V=0.0)
        assert np.all(fd.u[1:, 0] == 0.0), f"{P=}: Bottom wall U should be 0.0 (No-slip)"
        assert np.all(fd.v[:, 0] == 0.0), f"{P=}: Bottom wall V should be 0.0 (No-slip)"
        assert np.all(fd.u[1:, -1] == 0.0), f"{P=}: Top wall U should be 0.0 (No-slip)"
        assert np.all(fd.v[:, -1] == 0.0), f"{P=}: Top wall V should be 0.0 (No-slip)"

        # 3. Test Outlet (Zero-gradient Neumann)
        assert np.array_equal(fd.u[-1, :], fd.u[-2, :]), f"{P=}: Outlet U should match the column next to it"
        assert np.array_equal(fd.v[-1, :], fd.v[-2, :]), f"{P=}: Outlet V should match the column next to it"

        # 4. Test Velocity field inside the cylinder (No-slip)
        assert np.all(fd.u[1:-1, 1:-1][fd.mask[1:-1, 1:-1]] == 0.0), f"{P=}: Cylinder U should be 0.0 (No-slip)"
        assert np.all(fd.v[1:-1, 1:-1][fd.mask[1:-1, 1:-1]] == 0.0), f"{P=}: Cylinder V should be 0.0 (No-slip)"

        # 5. Test Pressure isolation inside the cylinder
        assert np.all(fd.p[1:-1, 1:-1][fd.mask[1:-1, 1:-1]] == 0.0), f"{P=}: Pressure inside the cylinder should remain 0.0"