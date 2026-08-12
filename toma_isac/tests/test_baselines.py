"""测试 Stage 6 三个 baseline 的接口与公平性条件。"""

from dataclasses import replace
from pathlib import Path

import numpy as np

from baselines import (
    build_fpa_upa_geometry,
    build_traditional_resources,
    run_fixed_toma_fp,
    run_fpa_upa_fp,
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


def test_traditional_resources_follow_locked_rules() -> None:
    """验证 B1 使用 MRT/steering/fixed-UL，而不是 FP。"""

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

    # MRT 只要求方向与 h_k 一致；总功率缩放不会改变方向。
    for k in range(cfg.k_dl):
        beam = resources.q_matrix[:, k]
        normalized_beam = beam / np.linalg.norm(beam)
        normalized_h = channels.h[k] / np.linalg.norm(channels.h[k])

        assert np.allclose(
            normalized_beam,
            normalized_h,
        )


def test_fixed_toma_and_upa_baselines_are_well_formed() -> None:
    """验证 B2 固定 ToMA，B3 使用固定 UPA，并可调用现有 FP。"""

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
    """验证 B1 可执行 RCG，并与其它 ToMA 方案使用同一初始 seed。"""

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

    check_resource_constraints(
        result.resources,
        cfg,
    )

    # 最终端点仍满足缆绳定长。
    assert np.allclose(
        np.linalg.norm(
            result.geometry.endpoints,
            axis=1,
        ),
        cfg.cable_length,
    )
