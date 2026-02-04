import numpy as np


class GeneralWave():

    def __init__(self, x0: np.ndarray, dt: float):
        self.x = x0
        self.dt = dt

    def _init(self):
        """This completse the first step based on single sided scheme"""
        raise(NotImplementedError)

    def _update_func(self):
        """Should contain the logic to update the x by one step"""
        raise(NotImplementedError)

    def run(self, n_iters: int):
        """Advances the grid by one step"""
        for _ in range(n_iters):
            self._update_func()