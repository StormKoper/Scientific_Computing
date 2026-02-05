import numpy as np
import matplotlib.pyplot as plt
from ..utils.wave import Wave1D

if __name__ == "__main__":
    # This is part B of the assignment
    x0 = np.linspace(0, 1, 1000)
    x0i = np.sin(2*np.pi*x0)
    x0ii = np.sin(5*np.pi*x0)
    x0iii = np.where((1/5 < x0) & (x0 < 2/5), np.sin(5*np.pi*x0), 0)

    for i, x0 in enumerate([x0i, x0ii, x0iii]):
        mywave = Wave1D(x0, 0.001, 0.001, c=1.0)
        mywave.run(1000)

        plt.imshow(mywave.x_arr, aspect='auto', cmap='viridis')
        plt.colorbar(label='Wave Amplitude')
        plt.title('Initial Condition: ' + (i+1)*'i')
        plt.savefig(f'set_1/results/wave_{(i+1)*"i"}.png')
        plt.show()