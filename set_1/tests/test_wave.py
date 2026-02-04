import numpy as np
from utils.wave import Wave1D

def test1():
    x0 = np.linspace(0, 1, 25)
    x0 = np.sin(2*np.pi*x0)

    mywave = Wave1D(x0, 0.001, 0.04)
    mywave.run(10)
    assert np.isclose([mywave.x[0], mywave.x[-1]], 0).all()

if __name__ == "__main__":
    test1()