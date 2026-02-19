import numpy as np
from utils.TIDE import Jacobi, GaussSeidel, SOR

x_even = np.zeros((20, 20))
x_even[0, :] = 1

x_uneven = np.zeros((25, 25))
x_uneven[0, :] = 1

def test_J():
    for x0 in [x_even, x_uneven]:
        J = Jacobi(x0.copy())
        for n_iters, epsilon in [(10, None), (None, 1e-3)]:
            J.run(n_iters, epsilon)

            assert np.allclose(J.x[0, :], 1) # top boundary condition
            assert np.allclose(J.x[-1, :], 0) # bottom boundary condition
            assert np.allclose(J.x[:, 0], J.x[:, -1]) # periodicity condition

def test_GS():
    for x0 in [x_even, x_uneven]:
        GS = GaussSeidel(x0.copy())
        for n_iters, epsilon in [(10, None), (None, 1e-3)]:
            GS.run(n_iters, epsilon)

            assert np.allclose(GS.x[0, :], 1) # top boundary condition
            assert np.allclose(GS.x[-1, :], 0) # bottom boundary condition
            assert np.allclose(GS.x[:, 0], GS.x[:, -1]) # periodicity condition

def test_SOR():
    for x0 in [x_even, x_uneven]:
        S = SOR(x0.copy())
        for n_iters, epsilon in [(10, None), (None, 1e-3)]:
            S.run(n_iters, epsilon)

            assert np.allclose(S.x[0, :], 1) # top boundary condition
            assert np.allclose(S.x[-1, :], 0) # bottom boundary condition
            assert np.allclose(S.x[:, 0], S.x[:, -1]) # periodicity condition

def test_J_jit():
    for x0 in [x_even, x_uneven]:
        J = Jacobi(x0.copy(), use_jit=True)
        for n_iters, epsilon in [(10, None), (None, 1e-3)]:
            J.run(n_iters, epsilon)

            assert np.allclose(J.x[0, :], 1) # top boundary condition
            assert np.allclose(J.x[-1, :], 0) # bottom boundary condition
            assert np.allclose(J.x[:, 0], J.x[:, -1]) # periodicity condition

def test_GS_jit():
    for x0 in [x_even, x_uneven]:
        GS = GaussSeidel(x0.copy(), use_jit=True)
        for n_iters, epsilon in [(10, None), (None, 1e-3)]:
            GS.run(n_iters, epsilon)

            assert np.allclose(GS.x[0, :], 1) # top boundary condition
            assert np.allclose(GS.x[-1, :], 0) # bottom boundary condition
            assert np.allclose(GS.x[:, 0], GS.x[:, -1]) # periodicity condition

def test_SOR_jit():
    for x0 in [x_even, x_uneven]:
        S = SOR(x0.copy(), use_jit=True)
        for n_iters, epsilon in [(10, None), (None, 1e-3)]:
            S.run(n_iters, epsilon)

            assert np.allclose(S.x[0, :], 1) # top boundary condition
            assert np.allclose(S.x[-1, :], 0) # bottom boundary condition
            assert np.allclose(S.x[:, 0], S.x[:, -1]) # periodicity condition

def test_J_equivalence():
    for x0 in [x_even, x_uneven]:
        J = Jacobi(x0.copy())
        J_jit = Jacobi(x0.copy(), use_jit=True)
        for n_iters, epsilon in [(10, None), (None, 1e-3)]:
            J.run(n_iters, epsilon)
            J_jit.run(n_iters, epsilon)

            assert np.allclose(J.x, J_jit.x)

def test_GS_equivalence():
    for x0 in [x_even, x_uneven]:
        GS = GaussSeidel(x0.copy())
        GS_jit = GaussSeidel(x0.copy(), use_jit=True)
        for n_iters, epsilon in [(10, None), (None, 1e-3)]:
            GS.run(n_iters, epsilon)
            GS_jit.run(n_iters, epsilon)
            
            assert np.allclose(GS.x, GS_jit.x)

def test_SOR_equivalence():
    for x0 in [x_even, x_uneven]:
        S = SOR(x0.copy())
        S_jit = SOR(x0.copy(), use_jit=True)
        for n_iters, epsilon in [(10, None), (None, 1e-3)]:
            S.run(n_iters, epsilon)
            S_jit.run(n_iters, epsilon)
            
            assert np.allclose(S.x, S_jit.x)

def test_obj_circle():
    N = 100
    y, x = np.ogrid[:N, :N]
    center_y, center_x = 50, 50
    radius = 15
    circle_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    x0 = np.zeros((N, N))
    J = Jacobi(x0.copy())
    J.objects(circle_mask)
    J.run(n_iters=10)
    J_jit = Jacobi(x0.copy(), use_jit=True)
    J_jit.objects(circle_mask)
    J_jit.run(n_iters=10)
    assert np.allclose(J.x[circle_mask], 0.0)
    assert np.allclose(J_jit.x[circle_mask], 0.0)
    assert np.allclose(J.x, J_jit.x)

