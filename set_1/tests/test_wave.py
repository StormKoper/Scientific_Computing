import numpy as np
from utils.wave import Wave1D, Wave2D

x1D = np.linspace(0, 1, 25)
x1D = np.sin(2*np.pi*x1D)
x1D[0] = 0
x1D[-1] = 0

x2D = np.zeros((25, 25))
x2D[0, :] = 1 # y=1 is index 0

def test1D():
    mywave = Wave1D(x1D.copy(), 0.001, 0.04)
    mywave.run(10)

    assert np.allclose([mywave.x[0], mywave.x[-1]], 0) # boundary conditions

def test2D():
    mywave = Wave2D(x2D.copy(), 0.0001, 0.04)
    mywave.run(10)

    assert np.allclose(mywave.x[0, :], 1) # top boundary condition
    assert np.allclose(mywave.x[-1, :], 0) # bottom boundary condition
    assert np.allclose(mywave.x[:, 0], mywave.x[:, -1]) # periodicity condition

def test1D_jit():
    mywave = Wave1D(x1D.copy(), 0.001, 0.04, use_jit=True)
    mywave.run(10)

    assert np.allclose([mywave.x[0], mywave.x[-1]], 0) # boundary conditions

def test2D_jit():
    mywave = Wave2D(x2D.copy(), 0.0001, 0.04, use_jit=True)
    mywave.run(10)

    assert np.allclose(mywave.x[0, :], 1) # top boundary condition
    assert np.allclose(mywave.x[-1, :], 0) # bottom boundary condition
    assert np.allclose(mywave.x[:, 0], mywave.x[:, -1]) # periodicity condition

def test1D_equivalence():
    mywave = Wave1D(x1D.copy(), 0.001, 0.04)
    mywave_jit = Wave1D(x1D.copy(), 0.001, 0.04, use_jit=True)
    mywave.run(10)
    mywave_jit.run(10)

    assert np.allclose(mywave.x, mywave_jit.x)

def test2D_equivalence():
    mywave = Wave2D(x2D.copy(), 0.0001, 0.04)
    mywave_jit = Wave2D(x2D.copy(), 0.0001, 0.04, use_jit=True)
    mywave.run(10)
    mywave_jit.run(10)

    assert np.allclose(mywave.x, mywave_jit.x)