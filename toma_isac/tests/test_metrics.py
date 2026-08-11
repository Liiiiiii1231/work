"""测试真实 DL / UL / sensing SINR、Rate 和 WSR。"""

from dataclasses import replace
from pathlib import Path

import numpy as np

from config import load_config
from metrics import (
    build_a_s,
    build_a_u,
    compute_performance,
)
from state import (
    ChannelState,
    ResourceState,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "stage5_multiuser.yaml"
)


def test_metrics_match_manual_calculation() -> None:
    """构造简单多用户矩阵，手算三类 SINR 与 WSR。"""

    cfg = load_config(
        CONFIG_PATH
    )

    # 为方便手算，把噪声设成 1。
    #
    # 同时放宽功率上限，使下面构造的测试资源
    # 本身也是一个合法 ResourceState。
    cfg = replace(
        cfg,
        p_dl_max=10.0,
        p_ul_max=np.ones(
            cfg.j_ul
        ),
        sigma_dl2=np.ones(
            cfg.k_dl
        ),
        sigma_bs2=1.0,
    )

    # ==========================================================
    # Q = [w_0, w_1, v_0, v_1]
    # ==========================================================

    q_matrix = np.array(
        [
            [1.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.complex128,
    )

    resources = ResourceState(
        q_matrix=q_matrix,

        q_ul=np.array(
            [
                0.5,
                1.0,
            ]
        ),

        # b_0=e_1, b_1=e_2
        b_ul=np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.complex128,
        ),

        # u_0=e_1, u_1=e_2
        u_s=np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.complex128,
        ),
    )

    # ==========================================================
    # 构造两个简单目标回波矩阵。
    # ==========================================================

    g_0 = np.zeros(
        (
            cfg.n_rx,
            cfg.n_tx,
        ),
        dtype=np.complex128,
    )

    g_0[0, 0] = 1.0

    g_1 = np.zeros_like(
        g_0
    )

    g_1[0, 0] = 2.0

    h_rsi = np.zeros_like(
        g_0
    )

    h_rsi[0, 0] = 1.0

    channels = ChannelState(
        h=np.array(
            [
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.complex128,
        ),

        f=np.array(
            [
                [2.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.complex128,
        ),

        g=np.stack(
            [
                g_0,
                g_1,
            ]
        ),

        # 这里不单独使用 H_SI^0，
        # metrics 只需要等效残余 H_RSI。
        h_si0=np.zeros_like(
            g_0
        ),

        h_rsi=h_rsi,
    )

    # ==========================================================
    # 1. 检查 A_U
    #
    # A_U
    # =
    # G_0 + G_1 + H_RSI
    #
    # 第一项因此为 1+2+1=4。
    # ==========================================================

    expected_a_u = np.zeros_like(
        g_0
    )

    expected_a_u[0, 0] = 4.0

    assert np.allclose(
        build_a_u(channels),
        expected_a_u,
    )

    # ==========================================================
    # 2. 检查 A_S,0
    #
    # A_S,0
    # =
    # G_1 + H_RSI
    #
    # 第一项为 2+1=3。
    # ==========================================================

    expected_a_s_0 = np.zeros_like(
        g_0
    )

    expected_a_s_0[0, 0] = 3.0

    assert np.allclose(
        build_a_s(
            channels,
            0,
        ),
        expected_a_s_0,
    )

    # ==========================================================
    # 计算系统真实性能。
    # ==========================================================

    performance = compute_performance(
        channels,
        resources,
        cfg,
    )

    # ==========================================================
    # 3. DL 用户 0
    #
    # desired = 1
    #
    # 用户 1 干扰 = |h_0^H w_1|^2 = 4
    #
    # sensing 干扰
    # = ||h_0^H V||^2
    # = 2
    #
    # noise = 1
    #
    # gamma_D,0 = 1 / 7
    # ==========================================================

    expected_gamma_dl_0 = (
        1.0 / 7.0
    )

    assert np.isclose(
        performance.gamma_dl[0],
        expected_gamma_dl_0,
    )

    # ==========================================================
    # 4. UL 用户 0
    #
    # desired
    # =
    # 0.5^2 * |2|^2
    # =
    # 1
    #
    # other UL interference
    # =
    # 1^2 * |1|^2
    # =
    # 1
    #
    # ||b_0^H A_U Q||^2
    # =
    # 48
    #
    # noise = 1
    #
    # gamma_U,0
    # =
    # 1 / 50
    # ==========================================================

    expected_gamma_ul_0 = (
        1.0 / 50.0
    )

    assert np.isclose(
        performance.gamma_ul[0],
        expected_gamma_ul_0,
    )

    # ==========================================================
    # 5. Sensing 目标 0
    #
    # desired
    # =
    # ||u_0^H G_0 Q||^2
    # =
    # 3
    #
    # UL interference
    # =
    # 1 + 1
    # =
    # 2
    #
    # ||u_0^H A_S,0 Q||^2
    # =
    # 27
    #
    # noise = 1
    #
    # gamma_S,0
    # =
    # 3 / 30
    # =
    # 0.1
    # ==========================================================

    expected_gamma_s_0 = (
        3.0 / 30.0
    )

    assert np.isclose(
        performance.gamma_s[0],
        expected_gamma_s_0,
    )

    # ==========================================================
    # 6. Rate
    #
    # R = log2(1 + gamma)
    # ==========================================================

    assert np.allclose(
        performance.rate_dl,
        np.log2(
            1.0
            + performance.gamma_dl
        ),
    )

    assert np.allclose(
        performance.rate_ul,
        np.log2(
            1.0
            + performance.gamma_ul
        ),
    )

    assert np.allclose(
        performance.rate_s,
        np.log2(
            1.0
            + performance.gamma_s
        ),
    )

    # ==========================================================
    # 7. Weighted Sum Rate
    # ==========================================================

    expected_wsr = float(
        cfg.weight_dl
        @ performance.rate_dl

        + cfg.weight_ul
        @ performance.rate_ul

        + cfg.weight_s
        @ performance.rate_s
    )

    assert np.isclose(
        performance.weighted_sum_rate,
        expected_wsr,
    )