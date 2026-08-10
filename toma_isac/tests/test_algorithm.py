"""测试完整单起点 ToMA-ISAC 联合优化算法。"""

import numpy as np

from algorithm import run_joint_algorithm
from config import load_config
from geometry import min_endpoint_distance
from utils import check_resource_constraints


def test_joint_algorithm() -> None:
    """检查完整算法的尺寸、约束和 WSR 非下降性。"""

    cfg = load_config()

    rng = np.random.default_rng(
        cfg.random_seed
    )

    # 测试只跑两个完整联合迭代，避免 pytest 时间过长。
    result = run_joint_algorithm(
        cfg,
        rng=rng,
        verbose=False,
        max_outer_iterations=2,
    )

    geometry = result.geometry
    channels = result.channels
    resources = result.resources

    # -------------------- 资源约束 --------------------

    check_resource_constraints(
        resources,
        cfg,
    )

    # -------------------- 几何尺寸 --------------------

    assert (
        geometry.endpoints.shape
        ==
        (
            cfg.m_uav,
            3,
        )
    )

    assert (
        geometry.tx_positions.shape
        ==
        (
            cfg.n_tx,
            3,
        )
    )

    assert (
        geometry.rx_positions.shape
        ==
        (
            cfg.n_rx,
            3,
        )
    )

    # -------------------- 缆绳定长 --------------------

    endpoint_norms = np.linalg.norm(
        geometry.endpoints,
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
            geometry.endpoints
        )
        >= cfg.min_uav_distance
        - 1e-8
    )

    # -------------------- 信道尺寸 --------------------

    assert (
        channels.h.shape
        ==
        (
            cfg.k_dl,
            cfg.n_tx,
        )
    )

    assert (
        channels.f.shape
        ==
        (
            cfg.j_ul,
            cfg.n_rx,
        )
    )

    assert (
        channels.g.shape
        ==
        (
            cfg.l_target,
            cfg.n_rx,
            cfg.n_tx,
        )
    )

    # -------------------- 资源尺寸 --------------------

    assert (
        resources.q_matrix.shape
        ==
        (
            cfg.n_tx,
            cfg.d_stream,
        )
    )

    assert (
        resources.q_ul.shape
        ==
        (
            cfg.j_ul,
        )
    )

    assert (
        resources.b_ul.shape
        ==
        (
            cfg.j_ul,
            cfg.n_rx,
        )
    )

    assert (
        resources.u_s.shape
        ==
        (
            cfg.l_target,
            cfg.n_rx,
        )
    )

    # -------------------- 联合 WSR --------------------

    history = np.asarray(
        result.joint_wsr_history,
        dtype=float,
    )

    assert len(history) >= 2

    assert np.all(
        np.isfinite(history)
    )

    # 完整联合迭代的真实 WSR 不应明显下降。
    assert np.all(
        np.diff(history)
        >= -1e-7
    )

    assert (
        history[-1]
        >= history[0] - 1e-7
    )

    # -------------------- 最终性能 --------------------

    assert np.isfinite(
        result.performance
        .weighted_sum_rate
    )

    assert np.all(
        result.performance.gamma_dl
        >= 0.0
    )

    assert np.all(
        result.performance.gamma_ul
        >= 0.0
    )

    assert np.all(
        result.performance.gamma_s
        >= 0.0
    )