import numpy as np
import pytest

from set_2.scripts.run_rd import seed_grid
from set_2.utils.RD import GrayScott

N = 50
DT = 1
DX = 1
CONSTS = {
    "Du": 0.16,
    "Dv": 0.08,
    "f": 0.035,
    "k": 0.060
}

####################################
####### initialization tests #######
####################################

def test_const_to_attr():
    GS = GrayScott(N, DT, DX, CONSTS)
    for c in CONSTS:
        assert c in dir(GS)

def test_non_float_const_err():
    inval_consts = {"Du": 0.16, "Dv": 0.08, "k": "garbage"} 
    with pytest.raises(ValueError):
        _ = GrayScott(N, DT, DX, inval_consts)

def test_incomp_consts_err():
    inval_consts = {"Du": 0.16, "Dv": 0.08, "f": 0.035}
    with pytest.raises(ValueError):
        _ = GrayScott(N, DT, DX, inval_consts)

####################################
########## dynamics tests ##########
####################################

def test_steady_state():
    GS = GrayScott(N, DT, DX, CONSTS)
    
    GS.grid["u"] = 1
    GS.grid["v"] = 0
    init_grid = GS.grid.copy()
    GS.run(10)
    assert np.array_equal(GS.grid, init_grid)

def test_no_negatives():
    GS = GrayScott(N, DT, DX, CONSTS)
    seed_grid(GS, 0)

    GS.run(100)
    assert np.min(GS.grid_hist["u"]) >= 0  # type: ignore

def test_pure_diffusion():
    consts = {"Du": 0.16, "Dv": 0.08, "f": 0.0, "k": 0.0}
    GS = GrayScott(N, DT, DX, consts)
    grid_size = GS.grid.shape[0]
    GS.grid["u"][grid_size//2, grid_size//2] = 0.5
    GS.grid["v"] = 0
    GS.run(50)
    assert all(np.diff(np.max(GS.grid_hist["u"], axis=(0, 1))) < 0) # type: ignore

####################################
######### simulation tests #########
####################################

def test_iteration_count():
    GS = GrayScott(N, DT, DX, CONSTS)
    GS.run(10)
    assert GS.iter_count == 10

def test_grids_swap():
    GS = GrayScott(N, DT, DX, CONSTS)
    init_grid_pointer = id(GS.grid)
    init_next_grid_pointer = id(GS.next_grid)
    GS._step()
    assert id(GS.grid) == init_next_grid_pointer
    assert id(GS.next_grid) == init_grid_pointer
