import numpy as np


class GeneralWave():

    def __init__(self, x0: np.ndarray, dt: float, dx: float):
        self.x = x0
        self.dt = dt
        self.constants = dict()

    def _first_step(self):
        """This completse the first step based on single sided scheme"""
        raise(NotImplementedError)

    def _update_func(self):
        """Should contain the logic to update the x by one step"""
        raise(NotImplementedError)

    def run(self, n_iters: int):
        """Advances the grid by one step"""
        for _ in range(n_iters):
            self._update_func()


class Wave1D(GeneralWave):

    def __init__(self, x0, dt, dx, c: float = 1.0):
        super().__init__(x0, dt, dx)
        self.constants["C^2"] = ( c*(dt/dx) )**2
        self._first_step()

    def _first_step(self):
        self.x_prev = self.x
        self.x[1:-2] = self.constants['C^2'] \
            * (self.x[0:-3] - 2*self.x[1:-2] + self.x[2:-1]) \
                + self.x[1:-2]
            
    def _update_func(self):
        self.x[1:-2] = self.constants['C^2'] \
            * (self.x[0:-3] - 2*self.x[1:-2] + self.x[2:-1]) \
                - self.x_prev[1:-2] + 2*self.x[1:-2]

def test1():
    x0 = np.linspace(0, 1, 25)
    x0 = np.sin(2*np.pi*x0)

    mywave = Wave1D(x0, 0.001, 0.04)
    mywave.run(10)
    assert np.isclose([mywave.x[0], mywave.x[-1]], 0).all()
    print(mywave.x)


if __name__ == "__main__":
    test1()