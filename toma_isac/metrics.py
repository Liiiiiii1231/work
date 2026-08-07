"""计算原始 ToMA-ISAC 问题的真实性能指标。

本文件只计算：

1. DL SINR
2. UL SINR
3. Sensing SINR
4. 三类 Rate
5. Weighted Sum Rate

非常重要：

以后 inner_solver.py 中会出现
LDT/QT 转换后的辅助目标。

那些“辅助目标”不要写进这里。

这里始终只计算原始联合优化问题中的
真实 SINR、Rate 和 WSR。

这样以后：
- 内层收敛检查
- 外层 Armijo 检查
- 最终仿真

都使用同一套真实目标函数。
"""

import numpy as np

from config import SystemConfig
from state import (
    ChannelState,
    PerformanceResult,
    ResourceState,
)
from utils import (
    check_finite,
    split_transmit_matrix,
)


def build_a_u(
    channels: ChannelState,
) -> np.ndarray:
    """构造 UL SINR 中的聚合干扰信道 A_U。

    数学模型：

    A_U
    =
    sum_l G_l
    + H_RSI

    返回：
        shape = (N_R, N_T)
    """

    return (
        np.sum(
            channels.g,
            axis=0,
        )
        + channels.h_rsi
    )


def build_a_s(
    channels: ChannelState,
    ell: int,
) -> np.ndarray:
    """构造目标 ell 的 A_S,l。

    数学模型：

    A_S,l
    =
    sum_{i != l} G_i
    + H_RSI

    返回：
        shape = (N_R, N_T)
    """

    return (
        np.sum(
            channels.g,
            axis=0,
        )
        - channels.g[ell]
        + channels.h_rsi
    )


def compute_downlink_sinr(
    channels: ChannelState,
    resources: ResourceState,
    cfg: SystemConfig,
) -> np.ndarray:
    """计算全部下行用户的 SINR gamma_k^D。"""

    # Q = [W,V]
    w_matrix, v_matrix = (
        split_transmit_matrix(
            resources.q_matrix,
            cfg,
        )
    )

    gamma_dl = np.empty(
        cfg.k_dl,
        dtype=float,
    )

    for k in range(
        cfg.k_dl
    ):
        h_k = channels.h[k]

        # ======================================================
        # 期望 DL 信号功率：
        #
        # |h_k^H w_k|^2
        #
        # np.vdot(a,b)
        # =
        # a^H b
        # ======================================================

        desired_power = (
            np.abs(
                np.vdot(
                    h_k,
                    w_matrix[:, k],
                )
            )
            ** 2
        )

        # ======================================================
        # 其他下行用户干扰：
        #
        # sum_{i != k}
        # |h_k^H w_i|^2
        # ======================================================

        dl_interference = 0.0

        for i in range(
            cfg.k_dl
        ):
            if i == k:
                continue

            dl_interference += (
                np.abs(
                    np.vdot(
                        h_k,
                        w_matrix[:, i],
                    )
                )
                ** 2
            )

        # ======================================================
        # 感知波束对 DL 的干扰：
        #
        # ||h_k^H V||_2^2
        # ======================================================

        sensing_interference = (
            np.linalg.norm(
                h_k.conj()
                @ v_matrix
            )
            ** 2
        )

        denominator = (
            dl_interference
            + sensing_interference
            + cfg.sigma_dl2[k]
        )

        gamma_dl[k] = float(
            np.real(
                desired_power
                / denominator
            )
        )

    return gamma_dl


def compute_uplink_sinr(
    channels: ChannelState,
    resources: ResourceState,
    cfg: SystemConfig,
) -> np.ndarray:
    """计算全部上行用户 SINR gamma_j^U。"""

    gamma_ul = np.empty(
        cfg.j_ul,
        dtype=float,
    )

    # A_U
    a_u = build_a_u(
        channels
    )

    for j in range(
        cfg.j_ul
    ):
        b_j = resources.b_ul[j]

        # ======================================================
        # 期望 UL 信号功率：
        #
        # q_j^2
        # |b_j^H f_j|^2
        #
        # q_ul[j]^2 才是真正 UL 功率。
        # ======================================================

        desired_power = (
            resources.q_ul[j]
            ** 2
            * np.abs(
                np.vdot(
                    b_j,
                    channels.f[j],
                )
            )
            ** 2
        )

        # ======================================================
        # 其他 UL 用户干扰：
        #
        # sum_{i != j}
        # q_i^2
        # |b_j^H f_i|^2
        # ======================================================

        ul_interference = 0.0

        for i in range(
            cfg.j_ul
        ):
            if i == j:
                continue

            ul_interference += (
                resources.q_ul[i]
                ** 2
                * np.abs(
                    np.vdot(
                        b_j,
                        channels.f[i],
                    )
                )
                ** 2
            )

        # ======================================================
        # 感知回波 + RSI：
        #
        # ||b_j^H A_U Q||_2^2
        # ======================================================

        fd_interference = (
            np.linalg.norm(
                b_j.conj()
                @ a_u
                @ resources.q_matrix
            )
            ** 2
        )

        # 因为 ||b_j||=1，
        # 合并后的噪声功率就是 sigma_BS^2。
        denominator = (
            ul_interference
            + fd_interference
            + cfg.sigma_bs2
        )

        gamma_ul[j] = float(
            np.real(
                desired_power
                / denominator
            )
        )

    return gamma_ul


def compute_sensing_sinr(
    channels: ChannelState,
    resources: ResourceState,
    cfg: SystemConfig,
) -> np.ndarray:
    """计算全部感知目标 SINR gamma_l^S。"""

    gamma_s = np.empty(
        cfg.l_target,
        dtype=float,
    )

    for ell in range(
        cfg.l_target
    ):
        u_l = resources.u_s[ell]

        # A_S,l
        a_s = build_a_s(
            channels,
            ell,
        )

        # ======================================================
        # 目标 ell 的期望回波功率：
        #
        # ||u_l^H G_l Q||_2^2
        # ======================================================

        desired_power = (
            np.linalg.norm(
                u_l.conj()
                @ channels.g[ell]
                @ resources.q_matrix
            )
            ** 2
        )

        # ======================================================
        # UL 用户对 sensing 的干扰：
        #
        # sum_j
        # q_j^2
        # |u_l^H f_j|^2
        # ======================================================

        ul_interference = 0.0

        for j in range(
            cfg.j_ul
        ):
            ul_interference += (
                resources.q_ul[j]
                ** 2
                * np.abs(
                    np.vdot(
                        u_l,
                        channels.f[j],
                    )
                )
                ** 2
            )

        # ======================================================
        # 其他目标回波 + RSI：
        #
        # ||u_l^H A_S,l Q||_2^2
        # ======================================================

        other_echo_and_si = (
            np.linalg.norm(
                u_l.conj()
                @ a_s
                @ resources.q_matrix
            )
            ** 2
        )

        denominator = (
            ul_interference
            + other_echo_and_si
            + cfg.sigma_bs2
        )

        gamma_s[ell] = float(
            np.real(
                desired_power
                / denominator
            )
        )

    return gamma_s


def compute_performance(
    channels: ChannelState,
    resources: ResourceState,
    cfg: SystemConfig,
) -> PerformanceResult:
    """统一计算当前系统真实 SINR、Rate 和 WSR。"""

    gamma_dl = (
        compute_downlink_sinr(
            channels,
            resources,
            cfg,
        )
    )

    gamma_ul = (
        compute_uplink_sinr(
            channels,
            resources,
            cfg,
        )
    )

    gamma_s = (
        compute_sensing_sinr(
            channels,
            resources,
            cfg,
        )
    )

    # ==========================================================
    # SINR 理论上均应 >= 0。
    #
    # 极小负数可能来自浮点误差。
    # ==========================================================

    for (
        name,
        gamma,
    ) in (
        ("gamma_dl", gamma_dl),
        ("gamma_ul", gamma_ul),
        ("gamma_s", gamma_s),
    ):
        check_finite(
            gamma,
            name,
        )

        if np.any(
            gamma < -1e-12
        ):
            raise ValueError(
                f"{name} contains "
                "invalid negative values."
            )

    # 仅消除数值级别的极小负数。
    gamma_dl = np.maximum(
        gamma_dl,
        0.0,
    )

    gamma_ul = np.maximum(
        gamma_ul,
        0.0,
    )

    gamma_s = np.maximum(
        gamma_s,
        0.0,
    )

    # ==========================================================
    # R = log2(1 + gamma)
    # ==========================================================

    rate_dl = np.log2(
        1.0 + gamma_dl
    )

    rate_ul = np.log2(
        1.0 + gamma_ul
    )

    rate_s = np.log2(
        1.0 + gamma_s
    )

    # ==========================================================
    # Weighted Sum Rate
    #
    # F =
    # sum_k omega_k^D R_k^D
    # +
    # sum_j omega_j^U R_j^U
    # +
    # sum_l omega_l^S R_l^S
    # ==========================================================

    weighted_sum_rate = float(
        cfg.weight_dl
        @ rate_dl

        + cfg.weight_ul
        @ rate_ul

        + cfg.weight_s
        @ rate_s
    )

    check_finite(
        weighted_sum_rate,
        "weighted_sum_rate",
    )

    return PerformanceResult(
        gamma_dl=gamma_dl,
        gamma_ul=gamma_ul,
        gamma_s=gamma_s,

        rate_dl=rate_dl,
        rate_ul=rate_ul,
        rate_s=rate_s,

        weighted_sum_rate=(
            weighted_sum_rate
        ),
    )