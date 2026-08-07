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