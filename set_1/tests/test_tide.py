import numpy as np
from utils.TIDE import Jacobi, GaussSeidel, SOR

def test_J():
    x0 = np.zeros((25,25))
    x0[0, :] = 1

    J = Jacobi(x0)
    for n_iters, epsilon in [(10, None), (None, 1e-3)]:
        J.run(n_iters, epsilon)

        assert np.allclose(J.x[0, :], 1) # top boundary condition
        assert np.allclose(J.x[-1, :], 0) # bottom boundary condition
        assert np.allclose(J.x[:, 0], J.x[:, -1]) # periodicity condition

def test_GS():
    x0 = np.zeros((25,25))
    x0[0, :] = 1

    GS = GaussSeidel(x0)
    for n_iters, epsilon in [(10, None), (None, 1e-3)]:
        GS.run(n_iters, epsilon)

        assert np.allclose(GS.x[0, :], 1) # top boundary condition
        assert np.allclose(GS.x[-1, :], 0) # bottom boundary condition
        assert np.allclose(GS.x[:, 0], GS.x[:, -1]) # periodicity condition

def test_SOR():
    x0 = np.zeros((25,25))
    x0[0, :] = 1

    S = SOR(x0)
    for n_iters, epsilon in [(10, None), (None, 1e-3)]:
        S.run(n_iters, epsilon)

        assert np.allclose(S.x[0, :], 1) # top boundary condition
        assert np.allclose(S.x[-1, :], 0) # bottom boundary condition
        assert np.allclose(S.x[:, 0], S.x[:, -1]) # periodicity condition

def test_J_jit():
    x0 = np.zeros((25,25))
    x0[0, :] = 1

    J = Jacobi(x0, use_jit=True)
    for n_iters, epsilon in [(10, None), (None, 1e-3)]:
        J.run(n_iters, epsilon)

        assert np.allclose(J.x[0, :], 1) # top boundary condition
        assert np.allclose(J.x[-1, :], 0) # bottom boundary condition
        assert np.allclose(J.x[:, 0], J.x[:, -1]) # periodicity condition

def test_GS_jit():
    x0 = np.zeros((25,25))
    x0[0, :] = 1

    GS = GaussSeidel(x0, use_jit=True)
    for n_iters, epsilon in [(10, None), (None, 1e-3)]:
        GS.run(n_iters, epsilon)

        assert np.allclose(GS.x[0, :], 1) # top boundary condition
        assert np.allclose(GS.x[-1, :], 0) # bottom boundary condition
        assert np.allclose(GS.x[:, 0], GS.x[:, -1]) # periodicity condition

def test_SOR_jit():
    x0 = np.zeros((25,25))
    x0[0, :] = 1

    S = SOR(x0, use_jit=True)
    for n_iters, epsilon in [(10, None), (None, 1e-3)]:
        S.run(n_iters, epsilon)

        assert np.allclose(S.x[0, :], 1) # top boundary condition
        assert np.allclose(S.x[-1, :], 0) # bottom boundary condition
        assert np.allclose(S.x[:, 0], S.x[:, -1]) # periodicity condition

def test_J_equivalence():
    x0 = np.zeros((25,25))
    x0[0, :] = 1

    J = Jacobi(x0.copy())
    J_jit = Jacobi(x0.copy(), use_jit=True)
    for n_iters, epsilon in [(10, None), (None, 1e-3)]:
        J.run(n_iters, epsilon)
        J_jit.run(n_iters, epsilon)

        assert np.allclose(J.x, J_jit.x)

def test_GS_equivalence():
    x0 = np.zeros((25,25))
    x0[0, :] = 1

    GS = GaussSeidel(x0.copy())
    GS_jit = GaussSeidel(x0.copy(), use_jit=True)
    for n_iters, epsilon in [(10, None), (None, 1e-3)]:
        GS.run(n_iters, epsilon)
        GS_jit.run(n_iters, epsilon)
        
        assert np.allclose(GS.x, GS_jit.x)

def test_SOR_equivalence():
    x0 = np.zeros((25,25))
    x0[0, :] = 1

    S = SOR(x0.copy())
    S_jit = SOR(x0.copy(), use_jit=True)
    for n_iters, epsilon in [(10, None), (None, 1e-3)]:
        S.run(n_iters, epsilon)
        S_jit.run(n_iters, epsilon)
        
        assert np.allclose(S.x, S_jit.x)