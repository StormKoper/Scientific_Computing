import numpy as np

from set_2.utils.DLA import DLA

N = 20
n = 5
f = 0.3

def test_n_growth():
    dla = DLA(N, use_jit=False)
    dla.run(n_growth=n)
    assert np.allclose(dla.x[0, :], 1) # top boundary condition
    assert np.allclose(dla.x[-1, :], 0) # bottom boundary condition
    assert np.allclose(dla.x[:, 0], dla.x[:, -1]) # periodicity condition
    # exact number of sites
    assert np.sum(~dla.obj_mask) == n + 1 # account for the initial seed object

def test_grow_until():
    dla = DLA(N, use_jit=False)
    dla.run(grow_until=f)
    assert np.allclose(dla.x[0, :], 1) # top boundary condition
    assert np.allclose(dla.x[-1, :], 0) # bottom boundary condition
    assert np.allclose(dla.x[:, 0], dla.x[:, -1]) # periodicity condition
    # an object at the threshold and none above it
    height_threshold = dla.x.shape[0] - int(f * dla.x.shape[0])
    assert np.count_nonzero(~dla.obj_mask[height_threshold, :]) == 1 # only one object at the threshold
    assert np.all(dla.obj_mask[:height_threshold, :]) # no objects above the threshold

def test_n_growth_jit():
    dla = DLA(N, use_jit=True)
    dla.run(n_growth=n)
    assert np.allclose(dla.x[0, :], 1) # top boundary condition
    assert np.allclose(dla.x[-1, :], 0) # bottom boundary condition
    assert np.allclose(dla.x[:, 0], dla.x[:, -1]) # periodicity condition
    # exact number of sites
    assert np.sum(~dla.obj_mask) == n + 1 # account for the initial seed object

def test_grow_until_jit():
    dla = DLA(N, use_jit=True)
    dla.run(grow_until=f)
    assert np.allclose(dla.x[0, :], 1) # top boundary condition
    assert np.allclose(dla.x[-1, :], 0) # bottom boundary condition
    assert np.allclose(dla.x[:, 0], dla.x[:, -1]) # periodicity condition
    # an object at the threshold and none above it
    height_threshold = dla.x.shape[0] - int(f * dla.x.shape[0])
    assert np.count_nonzero(~dla.obj_mask[height_threshold, :]) == 1 # only one object at the threshold
    assert np.all(dla.obj_mask[:height_threshold, :]) # no objects above the threshold