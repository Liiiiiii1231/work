"""测试固定 ToMA 下的完整内层资源优化。"""

import numpy as np

from channels import build_channels
from config import load_config
from geometry import build_geometry
from initialization import (
    generate_feasible_endpoints,
    initialize_resources,
)
from inner_solver import solve_inner_problem
from metrics import compute_performance
from utils import check_resource_constraints


def test_inner_solver() -> None:
    """检查内层算法的尺寸、约束和 WSR 收敛性。"""

    cfg = load_config()

    rng = np.random.default_rng(
        cfg.random_seed
    )

    # 与第一阶段相同的初始化
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

    initial_resources = initialize_resources(
        geometry,
        channels,
        cfg,
        rng,
    )

    check_resource_constraints(
        initial_resources,
        cfg,
    )

    initial_performance = compute_performance(
        channels,
        initial_resources,
        cfg,
    )

    initial_wsr = (
        initial_performance.weighted_sum_rate
    )

    # 执行内层优化
    (
        optimized_resources,
        wsr_history,
    ) = solve_inner_problem(
        channels,
        initial_resources,
        cfg,
        verbose=False,
    )

    check_resource_constraints(
        optimized_resources,
        cfg,
    )

    # -------------------- 尺寸 --------------------

    assert (
        optimized_resources.q_matrix.shape
        ==
        (
            cfg.n_tx,
            cfg.d_stream,
        )
    )

    assert (
        optimized_resources.q_ul.shape
        ==
        (cfg.j_ul,)
    )

    assert (
        optimized_resources.b_ul.shape
        ==
        (
            cfg.j_ul,
            cfg.n_rx,
        )
    )

    assert (
        optimized_resources.u_s.shape
        ==
        (
            cfg.l_target,
            cfg.n_rx,
        )
    )

    # -------------------- 功率约束 --------------------

    q_power = (
        np.linalg.norm(
            optimized_resources.q_matrix,
            "fro",
        ) ** 2
    )

    assert (
        q_power
        <= cfg.p_dl_max + 1e-8
    )

    assert np.all(
        optimized_resources.q_ul
        >= -1e-10
    )

    assert np.all(
        optimized_resources.q_ul
        <= np.sqrt(cfg.p_ul_max)
        + 1e-8
    )

    # -------------------- 单位范数 --------------------

    assert np.allclose(
        np.linalg.norm(
            optimized_resources.b_ul,
            axis=1,
        ),
        1.0,
        atol=1e-8,
    )

    assert np.allclose(
        np.linalg.norm(
            optimized_resources.u_s,
            axis=1,
        ),
        1.0,
        atol=1e-8,
    )

    # -------------------- WSR --------------------

    history = np.asarray(
        wsr_history,
        dtype=float,
    )

    assert len(history) >= 2

    assert len(history) <= (
        cfg.max_inner_iter + 1
    )

    assert np.all(
        np.isfinite(history)
    )

    assert np.isclose(
        history[0],
        initial_wsr,
        atol=1e-10,
    )

    # 最终 WSR 不应低于初始化值
    assert (
        history[-1]
        >= history[0] - 1e-7
    )

    # 每轮允许极小浮点误差，但不应明显下降
    assert np.all(
        np.diff(history)
        >= -1e-7
    )