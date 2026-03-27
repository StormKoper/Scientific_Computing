import numpy as np

from set_3.utils.LBA import LBA, compute_macroscopic, equilibrium
from set_3.scripts.run_LBA import find_max_stable_reynolds, _choose_reynolds_targets


def test_equilibrium_recovers_macroscopic_fields():
    nx, ny = 12, 8
    rho = np.full((nx, ny), 1.2)
    ux = np.full((nx, ny), 0.05)
    uy = np.full((nx, ny), -0.02)

    f = equilibrium(rho, ux, uy)
    rho_out, ux_out, uy_out = compute_macroscopic(f)

    assert np.allclose(rho_out, rho, rtol=1e-12, atol=1e-12)
    assert np.allclose(ux_out, ux, rtol=1e-12, atol=1e-12)
    assert np.allclose(uy_out, uy, rtol=1e-12, atol=1e-12)


def test_lba_short_run_stays_finite():
    solver = LBA(Nx=120, Ny=50, U_inlet=0.12, Re=100)

    for _ in range(400):
        solver.step()

    assert np.isfinite(solver.f).all()
    assert np.isfinite(solver.rho).all()
    assert np.isfinite(solver.ux).all()
    assert np.isfinite(solver.uy).all()


def test_inlet_velocity_is_enforced_in_mean():
    solver = LBA(Nx=120, Ny=50, U_inlet=0.12, Re=100)

    for _ in range(300):
        solver.step()

    inlet_fluid = ~solver.obstacle[0, :]
    ux_inlet_mean = float(np.mean(solver.ux[0, inlet_fluid]))

    # Zou-He inlet is pointwise local and can vary near corners; mean should remain near target.
    assert abs(ux_inlet_mean - solver.U_inlet) < 0.03


def test_obstacle_velocity_remains_no_slip():
    solver = LBA(Nx=120, Ny=50, U_inlet=0.12, Re=100)

    for _ in range(300):
        solver.step()

    assert np.allclose(solver.ux[solver.obstacle], 0.0, atol=1e-14)
    assert np.allclose(solver.uy[solver.obstacle], 0.0, atol=1e-14)


def test_high_re_short_run_no_nan_inf():
    solver = LBA(Nx=120, Ny=50, U_inlet=0.12, Re=200)

    for _ in range(300):
        solver.step()

    assert np.isfinite(solver.rho).all()
    assert np.isfinite(solver.ux).all()
    assert np.isfinite(solver.uy).all()

def test_find_max_stable_reynolds_returns_valid_scan():
    max_re, results = find_max_stable_reynolds(
        Nx=80,
        Ny=40,
        U_inlet=0.10,
        re_values=[10, 20, 30],
        scan_steps=200,
        check_every=50,
    )

    assert len(results) >= 1
    assert all(len(item) == 2 for item in results)
    assert np.isfinite(max_re)
    assert max_re >= 10


def test_choose_reynolds_targets_is_sorted_and_bounded():
    targets = _choose_reynolds_targets(68, min_re=10, n_points=6)

    assert targets[0] == 10
    assert targets[-1] < 68
    assert all(targets[i] < targets[i + 1] for i in range(len(targets) - 1))


def test_choose_reynolds_targets_handles_invalid_max():
    targets = _choose_reynolds_targets(np.nan, min_re=10, n_points=6)
    assert targets == [10]
