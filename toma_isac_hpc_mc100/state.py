from dataclasses import dataclass

import numpy as np


@dataclass
class GeometryState:
    # c_m
    endpoints: np.ndarray

    # p_i^T
    tx_positions: np.ndarray

    # p_s^R
    rx_positions: np.ndarray


@dataclass
class ChannelState:
    # h_k
    h: np.ndarray

    # f_j
    f: np.ndarray

    # G_l
    g: np.ndarray

    # H_SI^0
    h_si0: np.ndarray

    # H_RSI
    h_rsi: np.ndarray


@dataclass
class ResourceState:
    # Q
    q_matrix: np.ndarray

    # q_j
    q_ul: np.ndarray

    # b_j
    b_ul: np.ndarray

    # u_l
    u_s: np.ndarray


@dataclass
class PerformanceResult:
    gamma_dl: np.ndarray
    gamma_ul: np.ndarray
    gamma_s: np.ndarray

    rate_dl: np.ndarray
    rate_ul: np.ndarray
    rate_s: np.ndarray

    weighted_sum_rate: float

@dataclass
class FPState:
    """内层 Fractional Programming 的辅助变量。

    这里保存的是 LDT 和 QT 为了求解原问题而引入的辅助变量。

   
    """

    # ============================================================
    # LDT 辅助变量
    #
    # eta_k^D = gamma_k^D
    # eta_j^U = gamma_j^U
    # eta_l^S = gamma_l^S
    # ============================================================

    eta_dl: np.ndarray
    # shape = (K,)
    # 对应 {eta_k^D}

    eta_ul: np.ndarray
    # shape = (J,)
    # 对应 {eta_j^U}

    eta_s: np.ndarray
    # shape = (L,)
    # 对应 {eta_l^S}

    # ============================================================
    # QT 辅助变量
    # ============================================================

    phi_dl: np.ndarray
    # shape = (K,)
    # complex
    #
    # 对应标量：
    # phi_k^D

    phi_ul: np.ndarray
    # shape = (J,)
    # complex
    #
    # 对应标量：
    # phi_j^U

    phi_s: np.ndarray
    # shape = (L, D)
    # complex
    #
    # 每一行 phi_s[l]
    # 对应向量：
    # boldsymbol phi_l^S ∈ C^D