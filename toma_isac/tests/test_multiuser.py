"""测试多用户、多目标场景下的完整联合算法。"""

from pathlib import Path

import numpy as np

from algorithm import run_joint_algorithm
from config import load_config
from metrics import build_a_s
from utils import (
    check_resource_constraints,
    split_transmit_matrix,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "stage5_multiuser.yaml"
)


def test_multiuser_joint_algorithm() -> None:
    """验证 K=J=L=2 时的尺寸、干扰项和联合优化。"""

    cfg = load_config(
        CONFIG_PATH
    )

    assert cfg.k_dl == 2
    assert cfg.j_ul == 2
    assert cfg.l_target == 2
    assert cfg.r_sensing == 2

    rng = np.random.default_rng(
        cfg.random_seed
    )

    # pytest 中只跑两轮联合迭代，避免测试时间过长。
    result = run_joint_algorithm(
        cfg,
        rng=rng,
        verbose=False,
        max_outer_iterations=2,
    )

    channels = result.channels
    resources = result.resources
    performance = result.performance

    # ========================================================
    # 1. 多用户 / 多目标尺寸
    # ========================================================

    assert channels.h.shape == (
        cfg.k_dl,
        cfg.n_tx,
    )

    assert channels.f.shape == (
        cfg.j_ul,
        cfg.n_rx,
    )

    assert channels.g.shape == (
        cfg.l_target,
        cfg.n_rx,
        cfg.n_tx,
    )

    assert resources.q_matrix.shape == (
        cfg.n_tx,
        cfg.d_stream,
    )

    assert resources.q_ul.shape == (
        cfg.j_ul,
    )

    assert resources.b_ul.shape == (
        cfg.j_ul,
        cfg.n_rx,
    )

    assert resources.u_s.shape == (
        cfg.l_target,
        cfg.n_rx,
    )

    # ========================================================
    # 2. 多目标 A_S,l
    # ========================================================

    # L=2 时，目标 0 的干扰应包含 G_1 + H_RSI。
    a_s_0 = build_a_s(
        channels,
        0,
    )

    expected_a_s_0 = (
        channels.g[1]
        + channels.h_rsi
    )

    assert np.allclose(
        a_s_0,
        expected_a_s_0,
    )

    # ========================================================
    # 3. 显式验证一个 DL 用户的多用户 SINR
    # ========================================================

    w_matrix, v_matrix = (
        split_transmit_matrix(
            resources.q_matrix,
            cfg,
        )
    )

    h_0 = channels.h[0]

    desired_power = (
        np.abs(
            np.vdot(
                h_0,
                w_matrix[:, 0],
            )
        ) ** 2
    )

    # 用户 1 对用户 0 的 DL 干扰。
    other_user_interference = (
        np.abs(
            np.vdot(
                h_0,
                w_matrix[:, 1],
            )
        ) ** 2
    )

    sensing_interference = (
        np.linalg.norm(
            h_0.conj()
            @ v_matrix
        ) ** 2
    )

    gamma_dl_0_manual = (
        desired_power
        / (
            other_user_interference
            + sensing_interference
            + cfg.sigma_dl2[0]
        )
    )

    assert np.isclose(
        performance.gamma_dl[0],
        gamma_dl_0_manual,
        rtol=1e-8,
        atol=1e-10,
    )

    # ========================================================
    # 4. 性能尺寸
    # ========================================================

    assert performance.gamma_dl.shape == (
        2,
    )

    assert performance.gamma_ul.shape == (
        2,
    )

    assert performance.gamma_s.shape == (
        2,
    )

    assert performance.rate_dl.shape == (
        2,
    )

    assert performance.rate_ul.shape == (
        2,
    )

    assert performance.rate_s.shape == (
        2,
    )

    assert np.all(
        performance.gamma_dl >= 0.0
    )

    assert np.all(
        performance.gamma_ul >= 0.0
    )

    assert np.all(
        performance.gamma_s >= 0.0
    )

    assert np.isfinite(
        performance.weighted_sum_rate
    )

    # ========================================================
    # 5. 资源约束
    # ========================================================

    check_resource_constraints(
        resources,
        cfg,
    )

    assert np.allclose(
        np.linalg.norm(
            resources.b_ul,
            axis=1,
        ),
        1.0,
        atol=1e-8,
    )

    assert np.allclose(
        np.linalg.norm(
            resources.u_s,
            axis=1,
        ),
        1.0,
        atol=1e-8,
    )

    # ========================================================
    # 6. 联合 WSR 非下降
    # ========================================================

    history = np.asarray(
        result.joint_wsr_history,
        dtype=float,
    )

    assert np.all(
        np.isfinite(history)
    )

    assert np.all(
        np.diff(history)
        >= -1e-7
    )