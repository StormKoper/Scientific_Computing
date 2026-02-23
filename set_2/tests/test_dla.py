import numpy as np
from utils.DLA import DLA

x0 = np.zeros((20, 20))
x0[0, :] = 1

def test_DLA():
    for n_growth, grow_until in [(10, None), (None, 0.5)]:
        dla = DLA(x0.copy(), eta=1.0, omega=1.5, use_jit=False)
        dla.run(n_growth=n_growth, grow_until=grow_until)
        assert np.allclose(dla.x[0, :], 1) # top boundary condition
        assert np.allclose(dla.x[-1, :], 0) # bottom boundary condition
        assert np.allclose(dla.x[:, 0], dla.x[:, -1]) # periodicity condition
        # exact number of sites
        if n_growth is not None:
            assert np.sum(~dla.obj_mask) == n_growth + 1 # account for the initial seed object
        # an object at the threshold and none above it
        elif grow_until is not None:
            height_threshold = grow_until * dla.x.shape[0]
            assert np.any(~dla.obj_mask[-int(height_threshold):, :])
            assert not np.any(~dla.obj_mask[:-int(height_threshold), :])
    
def test_DLA_jit():
    for n_growth, grow_until in [(10, None), (None, 0.5)]:
        dla = DLA(x0.copy(), eta=1.0, omega=1.5, use_jit=True)
        dla.run(n_growth=n_growth, grow_until=grow_until)
        assert np.allclose(dla.x[0, :], 1) # top boundary condition
        assert np.allclose(dla.x[-1, :], 0) # bottom boundary condition
        assert np.allclose(dla.x[:, 0], dla.x[:, -1]) # periodicity condition
        # exact number of sites
        if n_growth is not None:
            assert np.sum(~dla.obj_mask) == n_growth + 1 # account for the initial seed object
        # an object at the threshold and none above it
        elif grow_until is not None:
            height_threshold = grow_until * dla.x.shape[0]
            assert np.any(~dla.obj_mask[-int(height_threshold):, :])
            assert not np.any(~dla.obj_mask[:-int(height_threshold), :])