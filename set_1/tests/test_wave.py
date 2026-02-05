import numpy as np
from utils.wave import Wave1D, Wave2D

def test1D():
    x0 = np.linspace(0, 1, 25)
    x0 = np.sin(2*np.pi*x0)
    x0[0] = 0
    x0[-1] = 0

    mywave = Wave1D(x0, 0.001, 0.04)
    mywave.run(10)
    assert np.isclose([mywave.x[0], mywave.x[-1]], 0).all() # boundary conditions

def test2D():
    x0 = np.zeros((25, 25))
    x0[:, 0] = 1 # y=1 is index 0

    mywave = Wave2D(x0, 0.0001, 0.04)
    mywave.run(10)
    assert np.isclose(mywave.x[:, 0], 1).all() # top boundary condition
    assert np.isclose(mywave.x[:, -1], 0).all() # bottom boundary condition
    assert np.isclose(mywave.x[0, :], mywave.x[-1, :]).all() # periodicity condition

if __name__ == "__main__":
    test1D()
    test2D()