import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

from ..utils.config import *  # noqa: F403
from ..utils.misc import analytical_concentration
from ..utils.TIDE import Jacobi, SOR, Gauss_S

if __name__== "__main__":
    x0 = np.zeros((25,25))
    x0[0, :] = 1

    # Jacobi
    J = Jacobi(x0)
    J.run(100)
    print("Jacobi Values")
    print(J.x_arr[..., -1])
    # Gauss Seidel
    G = Gauss_S(x0)
    G.run(100)
    print("Gauss Seidel Values")
    print(G.x_arr[..., -1])
    # SOR
    S = SOR(x0)
    S.run(100)
    print("SOR Values")
    print(S.x_arr[..., -1])


