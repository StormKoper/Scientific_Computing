import numpy as np
from set_2.utils.MC_DLA import MC_DLA

N = 20
n = 5
f = 0.3

def test_n_growth():
    dla = MC_DLA(N, use_jit=False)
    dla.run(n_growth=n)
    # exact number of sites
    assert np.sum(dla.grid) == n + 1 # account for the initial seed object

def test_grow_until():
    dla = MC_DLA(N, use_jit=False)
    dla.run(grow_until=f)
    # an object at the threshold and none above it
    height_threshold = dla.N - int(f * dla.N)
    assert np.count_nonzero(dla.grid[height_threshold, :]) == 1 # only one object at the threshold
    assert not np.any(dla.grid[:height_threshold, :]) # no objects above the threshold

def test_n_growth_jit():
    dla = MC_DLA(N, use_jit=True)
    dla.run(n_growth=n)
    # exact number of sites
    assert np.sum(dla.grid) == n + 1 # account for the initial seed object

def test_grow_until_jit():
    dla = MC_DLA(N, use_jit=True)
    dla.run(grow_until=f)
    # an object at the threshold and none above it
    height_threshold = dla.N - int(f * dla.N)
    assert np.count_nonzero(dla.grid[height_threshold, :]) == 1 # only one object at the threshold
    assert not np.any(dla.grid[:height_threshold, :]) # no objects above the threshold