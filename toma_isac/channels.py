"""ToMA 全双工近场 ISAC 信道模型。

本文件根据当前 GeometryState 构造：

h       -> 下行信道 h_k
f       -> 上行信道 f_j
g       -> 目标双程回波矩阵 G_l
h_si0   -> 未消除直接 SI 信道 H_SI^0
h_rsi   -> 等效残余 SI 信道 H_RSI

注意：
这里的变量命名已经锁定。
后续 inner_solver.py / outer_solver.py 不再改名。
"""

import numpy as np

from config import SystemConfig
from state import (
    ChannelState,
    GeometryState,
)


def transmit_array_response(
    tx_positions: np.ndarray,
    q: np.ndarray,
    cfg: SystemConfig,
) -> np.ndarray:
    """计算发射阵列的 USW 球面波响应 a_T。

    数学模型：
        [a_T]_i
        =
        exp(
            j k0 ||p_i^T - q||_2
        )

    返回
    ----------
    np.ndarray
        shape = (N_T,)
    """

    distances = np.linalg.norm(
        tx_positions
        - q[None, :],
        axis=1,
    )

    return np.exp(
        1j
        * cfg.k0
        * distances
    )


def receive_array_response(
    rx_positions: np.ndarray,
    q: np.ndarray,
    cfg: SystemConfig,
) -> np.ndarray:
    """计算接收阵列的 USW 球面波响应 a_R。

    当前严格按照你的论文当前定义：
        [a_R]_s
        =
        exp(
            j k0 ||p_s^R - q||_2
        )

    返回
    ----------
    np.ndarray
        shape = (N_R,)
    """

    distances = np.linalg.norm(
        rx_positions
        - q[None, :],
        axis=1,
    )

    return np.exp(
        1j
        * cfg.k0
        * distances
    )


def build_channels(
    geometry: GeometryState,
    cfg: SystemConfig,
) -> ChannelState:
    """根据当前 ToMA 几何构造所有通信、感知和 SI 信道。"""

    # ==========================================================
    # 创建输出数组。
    # ==========================================================

    # h[k] = h_k
    # shape = (K, N_T)
    h = np.empty(
        (
            cfg.k_dl,
            cfg.n_tx,
        ),
        dtype=np.complex128,
    )

    # f[j] = f_j
    # shape = (J, N_R)
    f = np.empty(
        (
            cfg.j_ul,
            cfg.n_rx,
        ),
        dtype=np.complex128,
    )

    # g[l] = G_l
    # shape = (L, N_R, N_T)
    #
    # 注意：
    # 这里的 g 不是普通向量，
    # 每个 g[l] 都是一个 N_R x N_T 矩阵。
    g = np.empty(
        (
            cfg.l_target,
            cfg.n_rx,
            cfg.n_tx,
        ),
        dtype=np.complex128,
    )

    # ==========================================================
    # 1. 下行信道
    #
    # h_k = alpha_k^D a_T
    # ==========================================================

    for k in range(
        cfg.k_dl
    ):
        a_t = (
            transmit_array_response(
                geometry.tx_positions,
                cfg.q_dl[k],
                cfg,
            )
        )

        h[k] = (
            cfg.alpha_dl[k]
            * a_t
        )

    # ==========================================================
    # 2. 上行信道
    #
    # f_j = alpha_j^U a_R
    #
    # 当前严格保持论文当前相位约定。
    # ==========================================================

    for j in range(
        cfg.j_ul
    ):
        a_r = (
            receive_array_response(
                geometry.rx_positions,
                cfg.q_ul[j],
                cfg,
            )
        )

        f[j] = (
            cfg.alpha_ul[j]
            * a_r
        )

    # ==========================================================
    # 3. 目标双程回波
    #
    # alpha_l^S
    # =
    # xi_l^S sqrt(
    #     beta_T,l^S beta_R,l^S
    # )
    #
    # G_l
    # =
    # alpha_l^S
    # a_R^*
    # a_T^H
    # ==========================================================

    for ell in range(
        cfg.l_target
    ):
        a_t = (
            transmit_array_response(
                geometry.tx_positions,
                cfg.q_s[ell],
                cfg,
            )
        )

        a_r = (
            receive_array_response(
                geometry.rx_positions,
                cfg.q_s[ell],
                cfg,
            )
        )

        alpha_s = (
            cfg.xi_s[ell]
            * np.sqrt(
                cfg.beta_t_s[ell]
                * cfg.beta_r_s[
                    ell
                ]
            )
        )

        # np.outer(a_r.conj(), a_t.conj())
        #
        # 正好对应：
        # a_R^* a_T^H
        #
        # shape = (N_R, N_T)
        g[ell] = (
            alpha_s
            * np.outer(
                a_r.conj(),
                a_t.conj(),
            )
        )

    # ==========================================================
    # 4. 直接 SI 阵元距离矩阵
    #
    # distance[s,i]
    # =
    # ||p_s^R - p_i^T||
    # ==========================================================

    difference = (
        geometry.rx_positions[
            :,
            None,
            :,
        ]
        - geometry.tx_positions[
            None,
            :,
            :,
        ]
    )

    distance = np.linalg.norm(
        difference,
        axis=2,
    )

    if np.min(
        distance
    ) < cfg.min_si_distance:
        raise ValueError(
            "Direct SI distance is too small."
        )

    # ==========================================================
    # 5. 未消除直接 SI
    #
    # H_SI^0[s,i]
    # =
    # lambda / (4 pi d)
    # * exp(-j k0 d)
    # ==========================================================

    h_si0 = (
        cfg.wavelength
        / (
            4.0
            * np.pi
            * distance
        )
        * np.exp(
            -1j
            * cfg.k0
            * distance
        )
    )

    # ==========================================================
    # 6. 等效残余 SI
    #
    # H_RSI
    # =
    # sqrt(rho_SI) H_SI^0
    #
    # rho_SI 是功率比例，
    # 所以信道幅值乘 sqrt(rho_SI)。
    # ==========================================================

    h_rsi = (
        np.sqrt(
            cfg.rho_si
        )
        * h_si0
    )

    channels = ChannelState(
        h=h,
        f=f,
        g=g,
        h_si0=h_si0,
        h_rsi=h_rsi,
    )

    check_channel_shapes(
        channels,
        cfg,
    )

    return channels


def check_channel_shapes(
    channels: ChannelState,
    cfg: SystemConfig,
) -> None:
    """检查所有信道的尺寸以及是否出现 NaN/Inf。"""

    expected_shapes = {
        "h": (
            cfg.k_dl,
            cfg.n_tx,
        ),

        "f": (
            cfg.j_ul,
            cfg.n_rx,
        ),

        "g": (
            cfg.l_target,
            cfg.n_rx,
            cfg.n_tx,
        ),

        "h_si0": (
            cfg.n_rx,
            cfg.n_tx,
        ),

        "h_rsi": (
            cfg.n_rx,
            cfg.n_tx,
        ),
    }

    for (
        name,
        expected_shape,
    ) in expected_shapes.items():

        array = getattr(
            channels,
            name,
        )

        if (
            array.shape
            != expected_shape
        ):
            raise ValueError(
                f"{name}.shape="
                f"{array.shape}, "
                f"expected "
                f"{expected_shape}."
            )

        if not np.all(
            np.isfinite(array)
        ):
            raise ValueError(
                f"{name} contains NaN or Inf."
            )