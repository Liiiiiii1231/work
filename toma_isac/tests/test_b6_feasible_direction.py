"""B6 修改后的位置可行方向基础测试；不需要 cvxpy。"""
from pathlib import Path
import importlib.util
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import ma_baseline as ma
from config import load_config
from baselines import build_fpa_upa_geometry

CFG = load_config(PROJECT_ROOT / 'configs/stage10_tradeoff.yaml')
BASE = build_fpa_upa_geometry(CFG)
TX_C = BASE.tx_positions.mean(axis=0)
RX_C = BASE.rx_positions.mean(axis=0)
D = CFG.wavelength / 2

def test_projection_obeys_active_spacing_constraints():
    # First element of packed 2x2 UPA, neighbors are to its +y and +z.
    proj = ma._project_to_ma_feasible_tangent(
        np.array([1., -2.]), BASE.tx_positions, 0, TX_C, 0.3, D
    )
    assert proj[0] <= 1e-10, proj
    assert proj[1] < -1.9, proj
    # Inward gradient has no feasible single-element ascent direction.
    proj_in = ma._project_to_ma_feasible_tangent(
        np.array([1., 1.]), BASE.tx_positions, 0, TX_C, 0.3, D
    )
    np.testing.assert_allclose(proj_in, [0,0], atol=1e-10)

def test_search_preserves_feasible_geometry_and_improves(monkeypatch):
    seen = []
    def fake_eval(tx, rx, resources, cfg):
        # Constrained optimization may sample an infeasible point only to estimate an
        # unconstrained gradient; accepted position changes must remain feasible.
        seen.append(True)
        # Outward direction from first element increases objective
        score = -float(tx[0,1] + tx[0,2])
        return score, ma._build_ma_geometry(tx,rx,cfg), None

    monkeypatch.setattr(ma,'_evaluate_ma_wsr',fake_eval)
    geometry, channels, wsr, accepted = ma._update_ma_positions_once(
        BASE, None, CFG, TX_C, RX_C, 0.3, D, 0.03
    )
    assert accepted >= 1, accepted
    assert wsr > -float(BASE.tx_positions[0,1] + BASE.tx_positions[0,2])
    assert len(seen) > 0
    assert ma._ma_geometry_is_feasible(
        geometry.tx_positions, geometry.rx_positions, TX_C, RX_C, 0.3, D, CFG
    )

if __name__ == '__main__':
    import pytest
    raise SystemExit(pytest.main([__file__, '-q']))

def test_projected_direction_unblocks_spacing_limited_move(monkeypatch):
    """Gradient points partly inward, but a tangent direction improves WSR."""
    def fake_eval(tx, rx, resources, cfg):
        score = float(tx[0, 1] - tx[0, 2])
        return score, ma._build_ma_geometry(tx, rx, cfg), None

    monkeypatch.setattr(ma, '_evaluate_ma_wsr', fake_eval)
    initial = float(BASE.tx_positions[0, 1] - BASE.tx_positions[0, 2])
    g2, _, final, n_moves = ma._update_ma_positions_once(
        BASE, None, CFG, TX_C, RX_C, .3, D, .001
    )
    assert n_moves >= 1
    assert final > initial + 1e-6
    assert ma._ma_geometry_is_feasible(
        g2.tx_positions, g2.rx_positions, TX_C, RX_C, .3, D, CFG
    )
