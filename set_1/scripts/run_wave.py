import numpy as np
import matplotlib.pyplot as plt
from ..utils.wave import Wave1D

def run_wave():
    x0 = np.linspace(0, 1, 100)
    x0 = np.sin(2*np.pi*x0)

    mywave = Wave1D(x0, 0.001, 0.01, c=1.0)
    mywave.run(100)

    return mywave.x_arr

if __name__ == "__main__":
    x_arr = run_wave()
    plt.imshow(x_arr, aspect='auto', cmap='viridis')
    plt.colorbar(label='Wave Amplitude')
    plt.show()