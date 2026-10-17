"""Stage 10A warm-start 接口回归测试。"""

from dataclasses import replace
from pathlib import Path

import numpy as np

from algorithm import run_joint_algorithm
from baselines import run_fixed_toma_fp, run_fpa_upa_fp
from config import load_config
from ma_baseline import run_ma_fp
from stage8 import load_stage8_experiment_config, make_scene_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "stage10_tradeoff.yaml"


def _short_cfg():
    cfg = load_config(CONFIG)
    exp = load_stage8_experiment_config(CONFIG)
    scene = make_scene_config(
        cfg,
        scene_seed=8001,
        position_jitter=exp.position_jitter,
        randomize_fading=False,
        fading_model=exp.fading_model,
    )
    return replace(scene, max_inner_iter=3, max_outer_iter=2), exp


def test_toma_and_fpa_warm_start_interfaces():
    cfg, _ = _short_cfg()

    b2 = run_fixed_toma_fp(
        cfg,
        rng=np.random.default_rng(2026),
    )
    b2_warm = run_fixed_toma_fp(
        cfg,
        rng=np.random.default_rng(1),
        initial_endpoints=b2.geometry.endpoints,
        initial_resources=b2.resources,
    )
    assert b2_warm.geometry.endpoints.shape == b2.geometry.endpoints.shape

    b3 = run_fpa_upa_fp(
        cfg,
        rng=np.random.default_rng(2026),
    )
    b3_warm = run_fpa_upa_fp(
        cfg,
        rng=np.random.default_rng(1),
        initial_resources=b3.resources,
    )
    assert b3_warm.resources.q_matrix.shape == b3.resources.q_matrix.shape


def test_proposed_and_ma_warm_start_interfaces():
    cfg, exp = _short_cfg()

    proposed = run_joint_algorithm(
        cfg,
        rng=np.random.default_rng(2026),
        max_outer_iterations=2,
        position_update_mode="fp_aware",
        fp_aware_candidate_iters=1,
    )
    proposed_warm = run_joint_algorithm(
        cfg,
        rng=np.random.default_rng(1),
        max_outer_iterations=2,
        position_update_mode="fp_aware",
        fp_aware_candidate_iters=1,
        initial_endpoints=proposed.geometry.endpoints,
        initial_resources=proposed.resources,
    )
    assert proposed_warm.geometry.endpoints.shape == proposed.geometry.endpoints.shape

    ma = run_ma_fp(
        cfg,
        rng=np.random.default_rng(2026),
        region_size=exp.ma_region_size,
        min_spacing=exp.ma_min_spacing,
        initial_step=exp.ma_initial_step,
        max_outer_iterations=2,
    )
    ma_warm = run_ma_fp(
        cfg,
        rng=np.random.default_rng(1),
        region_size=exp.ma_region_size,
        min_spacing=exp.ma_min_spacing,
        initial_step=exp.ma_initial_step,
        max_outer_iterations=2,
        initial_tx_positions=ma.geometry.tx_positions,
        initial_rx_positions=ma.geometry.rx_positions,
        initial_resources=ma.resources,
    )
    assert ma_warm.geometry.tx_positions.shape == ma.geometry.tx_positions.shape
