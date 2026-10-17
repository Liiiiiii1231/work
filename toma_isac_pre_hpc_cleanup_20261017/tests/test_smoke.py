"""第一大里程碑端到端冒烟测试。"""

import numpy as np

from channels import build_channels
from config import load_config
from geometry import build_geometry
from initialization import (
    generate_feasible_endpoints,
    initialize_resources,
)
from metrics import compute_performance
from utils import (
    check_resource_constraints,
)


def test_first_milestone_pipeline() -> None:
    """验证固定 ToMA 下完整系统流程能够正常运行。"""

    cfg = load_config()

    rng = np.random.default_rng(
        cfg.random_seed
    )

    endpoints = (
        generate_feasible_endpoints(
            cfg,
            rng,
        )
    )

    geometry = build_geometry(
        endpoints,
        cfg,
    )

    channels = build_channels(
        geometry,
        cfg,
    )

    resources = (
        initialize_resources(
            geometry,
            channels,
            cfg,
            rng,
        )
    )

    check_resource_constraints(
        resources,
        cfg,
    )

    performance = (
        compute_performance(
            channels,
            resources,
            cfg,
        )
    )

    # ==========================================================
    # GeometryState
    # ==========================================================

    assert (
        geometry.endpoints.shape
        == (
            cfg.m_uav,
            3,
        )
    )

    assert (
        geometry.tx_positions.shape
        == (
            cfg.n_tx,
            3,
        )
    )

    assert (
        geometry.rx_positions.shape
        == (
            cfg.n_rx,
            3,
        )
    )

    # ==========================================================
    # ChannelState
    # ==========================================================

    assert (
        channels.h.shape
        == (
            cfg.k_dl,
            cfg.n_tx,
        )
    )

    assert (
        channels.f.shape
        == (
            cfg.j_ul,
            cfg.n_rx,
        )
    )

    assert (
        channels.g.shape
        == (
            cfg.l_target,
            cfg.n_rx,
            cfg.n_tx,
        )
    )

    assert (
        channels.h_si0.shape
        == (
            cfg.n_rx,
            cfg.n_tx,
        )
    )

    assert (
        channels.h_rsi.shape
        == (
            cfg.n_rx,
            cfg.n_tx,
        )
    )

    # ==========================================================
    # ResourceState
    # ==========================================================

    assert (
        resources.q_matrix.shape
        == (
            cfg.n_tx,
            cfg.d_stream,
        )
    )

    assert (
        resources.q_ul.shape
        == (
            cfg.j_ul,
        )
    )

    assert (
        resources.b_ul.shape
        == (
            cfg.j_ul,
            cfg.n_rx,
        )
    )

    assert (
        resources.u_s.shape
        == (
            cfg.l_target,
            cfg.n_rx,
        )
    )

    # ==========================================================
    # SINR 必须有限且非负
    # ==========================================================

    assert np.all(
        np.isfinite(
            performance.gamma_dl
        )
    )

    assert np.all(
        np.isfinite(
            performance.gamma_ul
        )
    )

    assert np.all(
        np.isfinite(
            performance.gamma_s
        )
    )

    assert np.all(
        performance.gamma_dl
        >= 0.0
    )

    assert np.all(
        performance.gamma_ul
        >= 0.0
    )

    assert np.all(
        performance.gamma_s
        >= 0.0
    )

    # WSR 必须是正常有限数。
    assert np.isfinite(
        performance.weighted_sum_rate
    )