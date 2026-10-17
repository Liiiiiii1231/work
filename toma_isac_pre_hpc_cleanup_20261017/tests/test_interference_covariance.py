"""验证无交叉项的回波/残余 SI 协方差模型。"""

from dataclasses import replace

import numpy as np

from channels import build_channels
from config import load_config
from geometry import build_geometry
from initialization import (
    generate_feasible_endpoints,
    initialize_resources,
)
from metrics import (
    build_sensing_interference_covariance,
    build_transmit_interference_gram,
    build_uplink_interference_covariance,
    compute_performance,
)
from state import ChannelState


RHO_VALUES = np.array(
    [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    dtype=float,
)


def _build_fixed_state():
    """生成供协方差与 rho 测试共用的固定几何/资源状态。"""

    cfg = load_config()
    rng = np.random.default_rng(8001)
    endpoints = generate_feasible_endpoints(cfg, rng)
    geometry = build_geometry(endpoints, cfg)
    channels = build_channels(geometry, cfg)
    resources = initialize_resources(
        geometry,
        channels,
        cfg,
        rng,
    )
    return cfg, channels, resources


def test_covariances_match_explicit_component_power_sums() -> None:
    """正式协方差与逐分量模平方之和的误差小于 1e-10。"""

    cfg, channels, resources = _build_fixed_state()
    q_matrix = resources.q_matrix

    c_u = build_uplink_interference_covariance(
        channels,
        q_matrix,
    )

    for b_j in resources.b_ul:
        explicit = sum(
            np.linalg.norm(b_j.conj() @ g_l @ q_matrix) ** 2
            for g_l in channels.g
        )
        explicit += (
            np.linalg.norm(
                b_j.conj() @ channels.h_rsi @ q_matrix
            ) ** 2
        )
        formal = float(np.real(np.vdot(b_j, c_u @ b_j)))
        assert abs(formal - explicit) < 1e-10

        transmit_gram = build_transmit_interference_gram(
            channels,
            b_j,
        )
        q_side = float(
            np.real(
                np.trace(
                    q_matrix.conj().T
                    @ transmit_gram
                    @ q_matrix
                )
            )
        )
        assert abs(q_side - explicit) < 1e-10

    for ell in range(cfg.l_target):
        u_l = resources.u_s[ell]
        c_s_l = build_sensing_interference_covariance(
            channels,
            q_matrix,
            ell,
        )
        explicit = sum(
            np.linalg.norm(u_l.conj() @ g_i @ q_matrix) ** 2
            for index, g_i in enumerate(channels.g)
            if index != ell
        )
        explicit += (
            np.linalg.norm(
                u_l.conj() @ channels.h_rsi @ q_matrix
            ) ** 2
        )
        formal = float(np.real(np.vdot(u_l, c_s_l @ u_l)))
        assert abs(formal - explicit) < 1e-10

        transmit_gram = build_transmit_interference_gram(
            channels,
            u_l,
            excluded_target=ell,
        )
        q_side = float(
            np.real(
                np.trace(
                    q_matrix.conj().T
                    @ transmit_gram
                    @ q_matrix
                )
            )
        )
        assert abs(q_side - explicit) < 1e-10


def test_formal_covariance_excludes_coherent_cross_terms() -> None:
    """构造非零交叉项，确认正式模型采用功率和而非相干和。"""

    channels = ChannelState(
        h=np.zeros((1, 1), dtype=np.complex128),
        f=np.zeros((1, 1), dtype=np.complex128),
        g=np.array([[[1.0]], [[2.0]]], dtype=np.complex128),
        h_si0=np.array([[1.0]], dtype=np.complex128),
        h_rsi=np.array([[1.0]], dtype=np.complex128),
    )
    q_matrix = np.array([[1.0]], dtype=np.complex128)
    combiner = np.array([1.0], dtype=np.complex128)

    coherent_channel = np.sum(channels.g, axis=0) + channels.h_rsi
    old_coherent_interference = (
        np.linalg.norm(combiner.conj() @ coherent_channel @ q_matrix)
        ** 2
    )
    new_power_sum_interference = sum(
        np.linalg.norm(combiner.conj() @ g_l @ q_matrix) ** 2
        for g_l in channels.g
    ) + np.linalg.norm(
        combiner.conj() @ channels.h_rsi @ q_matrix
    ) ** 2

    c_u = build_uplink_interference_covariance(channels, q_matrix)
    formal = float(np.real(np.vdot(combiner, c_u @ combiner)))

    assert old_coherent_interference == 16.0
    assert new_power_sum_interference == 6.0
    assert old_coherent_interference != new_power_sum_interference
    assert abs(formal - new_power_sum_interference) < 1e-12


def test_fixed_state_performance_is_monotone_in_rho_si() -> None:
    """固定几何和资源时，rho 增大不能改善 UL、感知或总 WSR。"""

    cfg, base_channels, resources = _build_fixed_state()
    performances = []

    for rho_si in RHO_VALUES:
        channels = replace(
            base_channels,
            h_rsi=np.sqrt(rho_si) * base_channels.h_si0,
        )
        performances.append(
            compute_performance(channels, resources, cfg)
        )

    gamma_ul = np.stack([item.gamma_ul for item in performances])
    gamma_s = np.stack([item.gamma_s for item in performances])
    rate_dl = np.stack([item.rate_dl for item in performances])
    wsr = np.array(
        [item.weighted_sum_rate for item in performances]
    )

    assert np.all(np.diff(gamma_ul, axis=0) <= 1e-10)
    assert np.all(np.diff(gamma_s, axis=0) <= 1e-10)
    assert np.all(np.diff(wsr) <= 1e-10)
    assert np.allclose(rate_dl, rate_dl[0], atol=1e-12, rtol=0.0)


def test_residual_si_power_scales_linearly_with_rho_si() -> None:
    """H_RSI=sqrt(rho)H_SI0 使残余 SI 功率随 rho 线性缩放。"""

    _, channels, resources = _build_fixed_state()
    b_j = resources.b_ul[0]
    powers = []

    for rho_si in (1e-5, 1e-4):
        h_rsi = np.sqrt(rho_si) * channels.h_si0
        powers.append(
            np.linalg.norm(
                b_j.conj() @ h_rsi @ resources.q_matrix
            ) ** 2
        )

    assert powers[0] > 0.0
    assert np.isclose(powers[1] / powers[0], 10.0, rtol=1e-10)
