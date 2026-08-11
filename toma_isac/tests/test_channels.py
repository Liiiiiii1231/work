"""测试通信、感知和自干扰信道的数学表达式。"""

import numpy as np

from channels import build_channels
from config import load_config
from geometry import build_geometry


def test_channel_formulas_match_model() -> None:
    """用固定几何直接按公式重算各类信道。"""

    cfg = load_config()

    endpoints = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=float,
    )

    geometry = build_geometry(
        endpoints,
        cfg,
    )

    channels = build_channels(
        geometry,
        cfg,
    )

    # ==========================================================
    # 1. 下行信道
    #
    # h_k = alpha_k^D exp(j k0 d)
    # ==========================================================

    d_dl = np.linalg.norm(
        geometry.tx_positions
        - cfg.q_dl[0],
        axis=1,
    )

    expected_h = (
        cfg.alpha_dl[0]
        * np.exp(
            1j
            * cfg.k0
            * d_dl
        )
    )

    assert np.allclose(
        channels.h[0],
        expected_h,
    )

    # ==========================================================
    # 2. 上行信道
    #
    # f_j = alpha_j^U exp(j k0 d)
    # ==========================================================

    d_ul = np.linalg.norm(
        geometry.rx_positions
        - cfg.q_ul[0],
        axis=1,
    )

    expected_f = (
        cfg.alpha_ul[0]
        * np.exp(
            1j
            * cfg.k0
            * d_ul
        )
    )

    assert np.allclose(
        channels.f[0],
        expected_f,
    )

    # ==========================================================
    # 3. 目标双程回波
    #
    # G_l = alpha_l^S a_R^* a_T^H
    # ==========================================================

    d_s_tx = np.linalg.norm(
        geometry.tx_positions
        - cfg.q_s[0],
        axis=1,
    )

    d_s_rx = np.linalg.norm(
        geometry.rx_positions
        - cfg.q_s[0],
        axis=1,
    )

    alpha_s = (
        cfg.xi_s[0]
        * np.sqrt(
            cfg.beta_t_s[0]
            * cfg.beta_r_s[0]
        )
    )

    expected_g = (
        alpha_s
        * np.outer(
            np.exp(
                -1j
                * cfg.k0
                * d_s_rx
            ),
            np.exp(
                -1j
                * cfg.k0
                * d_s_tx
            ),
        )
    )

    assert np.allclose(
        channels.g[0],
        expected_g,
    )

    # ==========================================================
    # 4. 未消除直接 SI
    #
    # H_SI^0[s,i]
    # =
    # lambda / (4 pi d)
    # * exp(-j k0 d)
    # ==========================================================

    d_si = np.linalg.norm(
        geometry.rx_positions[
            :,
            None,
            :,
        ]
        - geometry.tx_positions[
            None,
            :,
            :,
        ],
        axis=2,
    )

    expected_h_si0 = (
        cfg.wavelength
        / (
            4.0
            * np.pi
            * d_si
        )
        * np.exp(
            -1j
            * cfg.k0
            * d_si
        )
    )

    assert np.allclose(
        channels.h_si0,
        expected_h_si0,
    )

    # ==========================================================
    # 5. 等效残余 SI
    #
    # H_RSI = sqrt(rho_SI) H_SI^0
    # ==========================================================

    assert np.allclose(
        channels.h_rsi,
        np.sqrt(cfg.rho_si)
        * channels.h_si0,
    )