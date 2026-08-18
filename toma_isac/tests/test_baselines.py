"""测试 Stage 6 五个 baseline 的接口与公平性条件。"""

from dataclasses import replace
from pathlib import Path

import numpy as np

from baselines import (
    build_fpa_upa_geometry,
    build_traditional_resources,
    run_fixed_toma_fp,
    run_fixed_toma_traditional,
    run_fpa_upa_fp,
    run_fpa_upa_traditional,
    run_rcg_toma_traditional,
)
from channels import build_channels
from config import load_config
from geometry import build_geometry
from initialization import generate_feasible_endpoints
from utils import check_resource_constraints


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "stage6_baselines.yaml"
)


def _short_cfg():
    """缩短迭代，仅用于 pytest 的快速接口检查。"""

    cfg = load_config(CONFIG_PATH)

    return replace(
        cfg,
        max_inner_iter=3,
        max_outer_iter=2,
    )


def _check_traditional_rules(result, cfg) -> None:
    """检查 traditional resource design 的锁定规则。"""

    check_resource_constraints(
        result.resources,
        cfg,
    )

    assert np.allclose(
        result.resources.q_ul,
        cfg.kappa_ul * np.sqrt(cfg.p_ul_max),
    )

    for k in range(cfg.k_dl):
        beam = result.resources.q_matrix[:, k]
        normalized_beam = beam / np.linalg.norm(beam)
        normalized_h = result.channels.h[k] / np.linalg.norm(
            result.channels.h[k]
        )

        assert np.allclose(
            normalized_beam,
            normalized_h,
        )


def test_traditional_resources_follow_locked_rules() -> None:
    """验证 MRT/steering/fixed-UL，而不是 FP。"""

    cfg = _short_cfg()
    rng = np.random.default_rng(cfg.random_seed)

    endpoints = generate_feasible_endpoints(cfg, rng)
    geometry = build_geometry(endpoints, cfg)
    channels = build_channels(geometry, cfg)

    resources = build_traditional_resources(
        geometry,
        channels,
        cfg,
    )

    check_resource_constraints(resources, cfg)

    assert np.allclose(
        resources.q_ul,
        cfg.kappa_ul * np.sqrt(cfg.p_ul_max),
    )

    for k in range(cfg.k_dl):
        beam = resources.q_matrix[:, k]
        normalized_beam = beam / np.linalg.norm(beam)
        normalized_h = channels.h[k] / np.linalg.norm(channels.h[k])

        assert np.allclose(
            normalized_beam,
            normalized_h,
        )


def test_fixed_toma_and_upa_fp_baselines_are_well_formed() -> None:
    """验证 B2 固定 ToMA，B3 使用固定 UPA，并调用现有 FP。"""

    cfg = _short_cfg()

    result_b2 = run_fixed_toma_fp(
        cfg,
        rng=np.random.default_rng(cfg.random_seed),
    )

    assert np.allclose(
        result_b2.geometry.endpoints,
        result_b2.initial_endpoints,
    )
    assert np.isfinite(
        result_b2.performance.weighted_sum_rate
    )
    assert np.all(
        np.diff(result_b2.wsr_history) >= -1e-7
    )
    check_resource_constraints(
        result_b2.resources,
        cfg,
    )

    geometry_upa = build_fpa_upa_geometry(cfg)

    assert geometry_upa.endpoints.shape == (0, 3)
    assert geometry_upa.tx_positions.shape == (cfg.n_tx, 3)
    assert geometry_upa.rx_positions.shape == (cfg.n_rx, 3)

    result_b3 = run_fpa_upa_fp(
        cfg,
        rng=np.random.default_rng(cfg.random_seed),
    )

    assert result_b3.initial_endpoints is None
    assert np.isfinite(
        result_b3.performance.weighted_sum_rate
    )
    assert np.all(
        np.diff(result_b3.wsr_history) >= -1e-7
    )
    check_resource_constraints(
        result_b3.resources,
        cfg,
    )


def test_rcg_toma_traditional_runs_with_same_initial_seed() -> None:
    """验证 B1 与其它 ToMA 方案使用同一初始 seed。"""

    cfg = _short_cfg()

    reference_rng = np.random.default_rng(cfg.random_seed)
    expected_endpoints = generate_feasible_endpoints(
        cfg,
        reference_rng,
    )

    result = run_rcg_toma_traditional(
        cfg,
        rng=np.random.default_rng(cfg.random_seed),
        max_outer_iterations=2,
    )

    assert np.allclose(
        result.initial_endpoints,
        expected_endpoints,
    )

    assert np.isfinite(
        result.performance.weighted_sum_rate
    )

    _check_traditional_rules(
        result,
        cfg,
    )

    assert np.allclose(
        np.linalg.norm(
            result.geometry.endpoints,
            axis=1,
        ),
        cfg.cable_length,
    )


def test_new_traditional_fixed_baselines_are_well_formed() -> None:
    """验证 B4/B5 不运行 FP 或 RCG，并遵守 traditional rules。"""

    cfg = _short_cfg()
    seed = cfg.random_seed

    expected_endpoints = generate_feasible_endpoints(
        cfg,
        np.random.default_rng(seed),
    )

    result_b4 = run_fixed_toma_traditional(
        cfg,
        rng=np.random.default_rng(seed),
    )

    assert np.allclose(
        result_b4.initial_endpoints,
        expected_endpoints,
    )
    assert np.allclose(
        result_b4.geometry.endpoints,
        expected_endpoints,
    )
    assert result_b4.iterations == 0
    assert result_b4.converged
    assert len(result_b4.wsr_history) == 1
    assert np.isfinite(
        result_b4.performance.weighted_sum_rate
    )
    _check_traditional_rules(result_b4, cfg)

    result_b5 = run_fpa_upa_traditional(cfg)

    assert result_b5.initial_endpoints is None
    assert result_b5.geometry.endpoints.shape == (0, 3)
    assert result_b5.iterations == 0
    assert result_b5.converged
    assert len(result_b5.wsr_history) == 1
    assert np.isfinite(
        result_b5.performance.weighted_sum_rate
    )
    _check_traditional_rules(result_b5, cfg)
