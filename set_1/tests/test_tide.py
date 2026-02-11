import numpy as np
from utils.TIDE import Jacobi, Gauss_S, SOR

def test_J():
    x0 = np.zeros((25,25))
    x0[0, :] = 1

    J = Jacobi(x0)
    J.run(10)

    assert np.isclose(J.x[0, :], 1).all() # top boundary condition
    assert np.isclose(J.x[-1, :], 0).all() # bottom boundary condition
    assert np.isclose(J.x[:, 0], J.x[:, -1]).all() # periodicity condition

def test_G():
    x0 = np.zeros((25,25))
    x0[0, :] = 1

    G = Gauss_S(x0)
    G.run(10)

    assert np.isclose(G.x[0, :], 1).all() # top boundary condition
    assert np.isclose(G.x[-1, :], 0).all() # bottom boundary condition
    assert np.isclose(G.x[:, 0], G.x[:, -1]).all() # periodicity condition

def test_SOR():
    x0 = np.zeros((25,25))
    x0[0, :] = 1

    S = SOR(x0)
    S.run(10)

    assert np.isclose(S.x[0, :], 1).all() # top boundary condition
    assert np.isclose(S.x[-1, :], 0).all() # bottom boundary condition
    assert np.isclose(S.x[:, 0], S.x[:, -1]).all() # periodicity condition