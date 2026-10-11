"""Stage 10 sensing-dominant dense / cold-only interface tests."""
from pathlib import Path
import numpy as np
from config import load_config
from experiments.run_stage10_tradeoff import load_tradeoff_config, make_group_weights, RESULT_DIR

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs'/'stage10_tradeoff.yaml'

def test_cold_dense_grid():
    grid=load_tradeoff_config(CONFIG).omega_comm
    assert len(grid)==27 and grid[0]==0.0 and grid[-1]==1e-4
    assert len(set(grid))==len(grid)
    assert all(0 <= x <= 1 for x in grid)

def test_noise_rcs_and_tolerances():
    cfg=load_config(CONFIG)
    assert np.isclose(cfg.tol_inner, 1e-4)
    assert np.isclose(cfg.tol_outer, 1e-4)
    assert np.isclose(cfg.sigma_bs2, 10**((-114-30)/10))
    assert np.allclose(cfg.sigma_dl2, 1e-10)
    assert np.allclose(cfg.xi_s, np.sqrt(4*np.pi*10)/cfg.wavelength)

def test_pure_sensing_zero_weight():
    cfg=load_config(CONFIG)
    a,b,c=make_group_weights(cfg,0.0)
    assert np.all(a==0) and np.all(b==0) and np.isclose(c.sum(),1)

def test_results_are_isolated():
    assert RESULT_DIR.name=='cold_dense_tol1e4'
