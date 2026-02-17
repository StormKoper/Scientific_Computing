import numpy as np
from utils.wave import Wave1D, Wave2D

def test1D():
    x0 = np.linspace(0, 1, 25)
    x0 = np.sin(2*np.pi*x0)
    x0[0] = 0
    x0[-1] = 0

    mywave = Wave1D(x0, 0.001, 0.04)
    mywave.run(10)

    assert np.allclose([mywave.x[0], mywave.x[-1]], 0) # boundary conditions

def test2D():
    x0 = np.zeros((25, 25))
    x0[0, :] = 1 # y=1 is index 0

    mywave = Wave2D(x0, 0.0001, 0.04)
    mywave.run(10)

    assert np.allclose(mywave.x[0, :], 1) # top boundary condition
    assert np.allclose(mywave.x[-1, :], 0) # bottom boundary condition
    assert np.allclose(mywave.x[:, 0], mywave.x[:, -1]) # periodicity condition

def test1D_jit():
    x0 = np.linspace(0, 1, 25)
    x0 = np.sin(2*np.pi*x0)
    x0[0] = 0
    x0[-1] = 0

    mywave = Wave1D(x0, 0.001, 0.04, use_jit=True)
    mywave.run(10)

    assert np.allclose([mywave.x[0], mywave.x[-1]], 0) # boundary conditions

def test2D_jit():
    x0 = np.zeros((25, 25))
    x0[0, :] = 1 # y=1 is index 0

    mywave = Wave2D(x0, 0.0001, 0.04, use_jit=True)
    mywave.run(10)

    assert np.allclose(mywave.x[0, :], 1) # top boundary condition
    assert np.allclose(mywave.x[-1, :], 0) # bottom boundary condition
    assert np.allclose(mywave.x[:, 0], mywave.x[:, -1]) # periodicity condition

def test1D_equivalence():
    x0 = np.linspace(0, 1, 25)
    x0 = np.sin(2*np.pi*x0)
    x0[0] = 0
    x0[-1] = 0

    mywave = Wave1D(x0.copy(), 0.001, 0.04)
    mywave_jit = Wave1D(x0.copy(), 0.001, 0.04, use_jit=True)
    mywave.run(10)
    mywave_jit.run(10)

    assert np.allclose(mywave.x, mywave_jit.x)

def test2D_equivalence():
    x0 = np.zeros((25, 25))
    x0[0, :] = 1 # y=1 is index 0

    mywave = Wave2D(x0.copy(), 0.0001, 0.04)
    mywave_jit = Wave2D(x0.copy(), 0.0001, 0.04, use_jit=True)
    mywave.run(10)
    mywave_jit.run(10)

    assert np.allclose(mywave.x, mywave_jit.x)