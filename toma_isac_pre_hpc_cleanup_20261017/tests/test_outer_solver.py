"""测试固定资源下的 ToMA 外层位置优化。"""

import numpy as np

from channels import build_channels
from config import load_config
from geometry import (
    build_geometry,
    min_endpoint_distance,
)
from initialization import (
    generate_feasible_endpoints,
    initialize_resources,
)
from inner_solver import solve_inner_problem
from outer_solver import solve_outer_problem
from utils import check_resource_constraints


def test_outer_solver() -> None:
    """检查外层位置优化的几何约束和 WSR 非下降性。"""

    cfg = load_config()

    rng = np.random.default_rng(
        cfg.random_seed
    )

    # 第一阶段：几何与信道
    endpoints = generate_feasible_endpoints(
        cfg,
        rng,
    )

    geometry = build_geometry(
        endpoints,
        cfg,
    )

    channels = build_channels(
        geometry,
        cfg,
    )

    # 初始化资源
    resources = initialize_resources(
        geometry,
        channels,
        cfg,
        rng,
    )

    # 第二阶段：先获得较好的固定资源
    (
        optimized_resources,
        _,
    ) = solve_inner_problem(
        channels,
        resources,
        cfg,
        verbose=False,
    )

    check_resource_constraints(
        optimized_resources,
        cfg,
    )

    # 第三阶段只跑两轮，主要验证程序逻辑。
    (
        optimized_geometry,
        optimized_channels,
        wsr_history,
    ) = solve_outer_problem(
        geometry,
        optimized_resources,
        cfg,
        verbose=False,
        max_sweeps=2,
    )

    # -------------------- 端点尺寸 --------------------

    assert (
        optimized_geometry.endpoints.shape
        ==
        (
            cfg.m_uav,
            3,
        )
    )

    # -------------------- 缆绳定长 --------------------

    endpoint_norms = np.linalg.norm(
        optimized_geometry.endpoints,
        axis=1,
    )

    assert np.allclose(
        endpoint_norms,
        cfg.cable_length,
        atol=1e-8,
    )

    # -------------------- 防碰撞 --------------------

    assert (
        min_endpoint_distance(
            optimized_geometry.endpoints
        )
        >= cfg.min_uav_distance
        - 1e-8
    )

    # -------------------- 信道尺寸 --------------------

    assert (
        optimized_channels.h.shape
        ==
        (
            cfg.k_dl,
            cfg.n_tx,
        )
    )

    assert (
        optimized_channels.f.shape
        ==
        (
            cfg.j_ul,
            cfg.n_rx,
        )
    )

    assert (
        optimized_channels.g.shape
        ==
        (
            cfg.l_target,
            cfg.n_rx,
            cfg.n_tx,
        )
    )

    # -------------------- WSR --------------------

    history = np.asarray(
        wsr_history,
        dtype=float,
    )

    assert len(history) >= 2

    assert np.all(
        np.isfinite(history)
    )

    # Armijo 接受的外层位置应使真实 WSR 非下降。
    assert np.all(
        np.diff(history)
        >= -1e-7
    )

    assert (
        history[-1]
        >= history[0] - 1e-7
    )