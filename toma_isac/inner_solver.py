"""固定 ToMA 几何下的内层资源优化。

更新顺序：
u_l -> b_j -> eta -> phi -> Q -> q_j -> true WSR

真实 SINR、Rate 和 WSR 统一由 metrics.py 计算。
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

from config import SystemConfig
from metrics import (
    build_a_s,
    build_a_u,
    compute_performance,
)
from state import (
    ChannelState,
    FPState,
    ResourceState,
)
from utils import (
    check_resource_constraints,
    hermitianize,
    normalize_vector,
    relative_change,
    split_transmit_matrix,
)


def copy_resources(
    resources: ResourceState,
) -> ResourceState:
    """复制资源，避免直接修改传入的初始 ResourceState。"""

    return ResourceState(
        q_matrix=resources.q_matrix.copy(),
        q_ul=resources.q_ul.copy(),
        b_ul=resources.b_ul.copy(),
        u_s=resources.u_s.copy(),
    )


# ============================================================
# 1. 更新感知接收波束 u_l
# ============================================================

def update_sensing_combiners(
    channels: ChannelState,
    resources: ResourceState,
    cfg: SystemConfig,
) -> np.ndarray:
    """用广义最大-SINR方向更新全部感知接收波束 u_l。"""

    q_q_h = (
        resources.q_matrix
        @ resources.q_matrix.conj().T
    )

    u_new = np.empty(
        (cfg.l_target, cfg.n_rx),
        dtype=np.complex128,
    )

    identity_nr = np.eye(
        cfg.n_rx,
        dtype=np.complex128,
    )

    for ell in range(cfg.l_target):
        g_l = channels.g[ell]
        a_s_l = build_a_s(
            channels,
            ell,
        )

        # 期望目标回波协方差 C_l^S
        c_l = (
            g_l
            @ q_q_h
            @ g_l.conj().T
        )

        # 干扰加噪声协方差 D_l^S
        d_l = (
            a_s_l
            @ q_q_h
            @ a_s_l.conj().T
        )

        # UL 用户对感知接收端的干扰
        for j in range(cfg.j_ul):
            f_j = channels.f[j]

            d_l += (
                resources.q_ul[j] ** 2
                * np.outer(
                    f_j,
                    f_j.conj(),
                )
            )

        d_l += (
            cfg.sigma_bs2
            * identity_nr
        )

        # 消除浮点误差导致的微小非 Hermitian 部分
        c_l = hermitianize(c_l)
        d_l = hermitianize(d_l)

        # 最大广义特征值对应最大-SINR方向
        _, eigenvectors = eigh(
            c_l,
            d_l,
            check_finite=True,
        )

        u_new[ell] = normalize_vector(
            eigenvectors[:, -1]
        )

    return u_new


# ============================================================
# 2. 更新上行接收波束 b_j
# ============================================================

def update_uplink_combiners(
    channels: ChannelState,
    resources: ResourceState,
    cfg: SystemConfig,
) -> np.ndarray:
    """用 MMSE / 最大-SINR方向更新全部上行接收波束 b_j。"""

    q_q_h = (
        resources.q_matrix
        @ resources.q_matrix.conj().T
    )

    a_u = build_a_u(channels)

    b_new = np.empty(
        (cfg.j_ul, cfg.n_rx),
        dtype=np.complex128,
    )

    identity_nr = np.eye(
        cfg.n_rx,
        dtype=np.complex128,
    )

    for j in range(cfg.j_ul):
        # 感知回波和 RSI 产生的干扰协方差
        d_j = (
            a_u
            @ q_q_h
            @ a_u.conj().T
        )

        # 其他 UL 用户干扰
        for i in range(cfg.j_ul):
            if i == j:
                continue

            f_i = channels.f[i]

            d_j += (
                resources.q_ul[i] ** 2
                * np.outer(
                    f_i,
                    f_i.conj(),
                )
            )

        d_j += (
            cfg.sigma_bs2
            * identity_nr
        )

        d_j = hermitianize(d_j)

        # solve(D_j, f_j) 代替显式计算 inv(D_j) @ f_j
        candidate = np.linalg.solve(
            d_j,
            channels.f[j],
        )

        b_new[j] = normalize_vector(
            candidate
        )

    return b_new


# ============================================================
# 3. LDT：更新 eta
# ============================================================

def update_ldt_variables(
    channels: ChannelState,
    resources: ResourceState,
    cfg: SystemConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """更新三类 LDT 辅助变量，最优值为 eta = gamma。"""

    performance = compute_performance(
        channels,
        resources,
        cfg,
    )

    return (
        performance.gamma_dl.copy(),
        performance.gamma_ul.copy(),
        performance.gamma_s.copy(),
    )


# ============================================================
# 4. QT：更新 phi
# ============================================================

def update_qt_variables(
    channels: ChannelState,
    resources: ResourceState,
    cfg: SystemConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """更新 DL、UL 和 sensing 的 QT 辅助变量。"""

    q_matrix = resources.q_matrix

    # Q=[W,V]，QT 的 DL 分子需要 w_k
    w_matrix, _ = split_transmit_matrix(
        q_matrix,
        cfg,
    )

    # -------------------- DL --------------------

    phi_dl = np.empty(
        cfg.k_dl,
        dtype=np.complex128,
    )

    for k in range(cfg.k_dl):
        h_k = channels.h[k]

        # 总接收功率 T_k^D，包含期望信号本身
        t_dl = (
            np.linalg.norm(
                h_k.conj() @ q_matrix
            ) ** 2
            + cfg.sigma_dl2[k]
        )

        phi_dl[k] = (
            np.vdot(
                h_k,
                w_matrix[:, k],
            )
            / t_dl
        )

    # -------------------- UL --------------------

    phi_ul = np.empty(
        cfg.j_ul,
        dtype=np.complex128,
    )

    a_u = build_a_u(channels)

    for j in range(cfg.j_ul):
        b_j = resources.b_ul[j]

        t_ul = 0.0

        # UL 总接收功率中的所有 UL 用户项，包含 i=j
        for i in range(cfg.j_ul):
            t_ul += (
                resources.q_ul[i] ** 2
                * np.abs(
                    np.vdot(
                        b_j,
                        channels.f[i],
                    )
                ) ** 2
            )

        t_ul += (
            np.linalg.norm(
                b_j.conj()
                @ a_u
                @ q_matrix
            ) ** 2
        )

        t_ul += cfg.sigma_bs2

        phi_ul[j] = (
            resources.q_ul[j]
            * np.vdot(
                b_j,
                channels.f[j],
            )
            / t_ul
        )

    # -------------------- Sensing --------------------

    phi_s = np.empty(
        (
            cfg.l_target,
            cfg.d_stream,
        ),
        dtype=np.complex128,
    )

    for ell in range(cfg.l_target):
        u_l = resources.u_s[ell]
        g_l = channels.g[ell]
        a_s_l = build_a_s(
            channels,
            ell,
        )

        # 目标 l 的总接收功率 T_l^S
        t_s = (
            np.linalg.norm(
                u_l.conj()
                @ g_l
                @ q_matrix
            ) ** 2
        )

        for j in range(cfg.j_ul):
            t_s += (
                resources.q_ul[j] ** 2
                * np.abs(
                    np.vdot(
                        u_l,
                        channels.f[j],
                    )
                ) ** 2
            )

        t_s += (
            np.linalg.norm(
                u_l.conj()
                @ a_s_l
                @ q_matrix
            ) ** 2
        )

        t_s += cfg.sigma_bs2

        # Q^H G_l^H u_l，结果是 D 维向量
        numerator = (
            q_matrix.conj().T
            @ g_l.conj().T
            @ u_l
        )

        phi_s[ell] = (
            numerator
            / t_s
        )

    return (
        phi_dl,
        phi_ul,
        phi_s,
    )


# ============================================================
# 5. 构造 A_Q 和 B_Q
# ============================================================

def build_q_matrices(
    channels: ChannelState,
    resources: ResourceState,
    fp_state: FPState,
    cfg: SystemConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """构造 Q 子问题中的 A_Q 和 B_Q。"""

    # log2 转成 ln 后使用的权重
    weight_dl_bar = (
        cfg.weight_dl
        / np.log(2.0)
    )

    weight_ul_bar = (
        cfg.weight_ul
        / np.log(2.0)
    )

    weight_s_bar = (
        cfg.weight_s
        / np.log(2.0)
    )

    a_q = np.zeros(
        (
            cfg.n_tx,
            cfg.n_tx,
        ),
        dtype=np.complex128,
    )

    b_q = np.zeros(
        (
            cfg.n_tx,
            cfg.d_stream,
        ),
        dtype=np.complex128,
    )

    # -------------------- DL --------------------

    for k in range(cfg.k_dl):
        h_k = channels.h[k]

        c_dl = np.outer(
            h_k,
            h_k.conj(),
        )

        common_weight = (
            weight_dl_bar[k]
            * (
                1.0
                + fp_state.eta_dl[k]
            )
        )

        a_q += (
            common_weight
            * np.abs(
                fp_state.phi_dl[k]
            ) ** 2
            * c_dl
        )

        # 只作用于 Q 的第 k 个通信波束列
        b_q[:, k] += (
            common_weight
            * fp_state.phi_dl[k]
            * h_k
        )

    # -------------------- UL --------------------

    a_u = build_a_u(channels)

    for j in range(cfg.j_ul):
        b_j = resources.b_ul[j]

        x_j = (
            a_u.conj().T
            @ b_j
        )

        c_ul = np.outer(
            x_j,
            x_j.conj(),
        )

        common_weight = (
            weight_ul_bar[j]
            * (
                1.0
                + fp_state.eta_ul[j]
            )
        )

        a_q += (
            common_weight
            * np.abs(
                fp_state.phi_ul[j]
            ) ** 2
            * c_ul
        )

    # -------------------- Sensing --------------------

    for ell in range(cfg.l_target):
        u_l = resources.u_s[ell]
        g_l = channels.g[ell]

        a_s_l = build_a_s(
            channels,
            ell,
        )

        g_des = (
            g_l.conj().T
            @ u_l
        )

        g_int = (
            a_s_l.conj().T
            @ u_l
        )

        c_des = np.outer(
            g_des,
            g_des.conj(),
        )

        c_int = np.outer(
            g_int,
            g_int.conj(),
        )

        common_weight = (
            weight_s_bar[ell]
            * (
                1.0
                + fp_state.eta_s[ell]
            )
        )

        phi_norm_sq = (
            np.linalg.norm(
                fp_state.phi_s[ell]
            ) ** 2
        )

        a_q += (
            common_weight
            * phi_norm_sq
            * (
                c_des
                + c_int
            )
        )

        # G_l^H u_l (phi_l^S)^H
        b_q += (
            common_weight
            * np.outer(
                g_des,
                fp_state.phi_s[
                    ell
                ].conj(),
            )
        )

    a_q = hermitianize(a_q)

    if a_q.shape != (
        cfg.n_tx,
        cfg.n_tx,
    ):
        raise ValueError(
            "A_Q dimension is incorrect."
        )

    if b_q.shape != (
        cfg.n_tx,
        cfg.d_stream,
    ):
        raise ValueError(
            "B_Q dimension is incorrect."
        )

    return a_q, b_q


# ============================================================
# 6. 使用拉格朗日乘子更新 Q
# ============================================================

def solve_q_given_lambda(
    a_q: np.ndarray,
    b_q: np.ndarray,
    lambda_p: float,
) -> np.ndarray:
    """给定 lambda_P，计算对应的 Q(lambda_P)。"""

    matrix = (
        a_q
        + lambda_p
        * np.eye(
            a_q.shape[0],
            dtype=np.complex128,
        )
    )

    # 避免显式计算矩阵逆
    return np.linalg.solve(
        matrix,
        b_q,
    )


def update_transmit_matrix(
    a_q: np.ndarray,
    b_q: np.ndarray,
    cfg: SystemConfig,
    range_tol: float = 1e-10,
) -> tuple[
    np.ndarray,
    float,
]:
    """利用 lambda_P 求解满足总功率约束的 Q。"""

    a_q = hermitianize(a_q)

    identity_nt = np.eye(
        cfg.n_tx,
        dtype=np.complex128,
    )

    # ----------------------------------------------------------
    # 先检查 lambda_P = 0 是否可行
    # ----------------------------------------------------------

    a_q_pinv = np.linalg.pinv(a_q)

    q_zero = (
        a_q_pinv
        @ b_q
    )

    # A_Q 奇异时还要检查 B_Q 是否位于 Range(A_Q)
    range_residual = np.linalg.norm(
        (
            identity_nt
            - a_q @ a_q_pinv
        )
        @ b_q,
        "fro",
    )

    range_scale = max(
        1.0,
        np.linalg.norm(
            b_q,
            "fro",
        ),
    )

    in_range = (
        range_residual
        <= range_tol * range_scale
    )

    q_zero_power = (
        np.linalg.norm(
            q_zero,
            "fro",
        ) ** 2
    )

    if (
        in_range
        and q_zero_power
        <= cfg.p_dl_max
        + cfg.tol_bisection
    ):
        return q_zero, 0.0

    # ----------------------------------------------------------
    # lambda_P > 0：先寻找二分搜索右端点
    # ----------------------------------------------------------

    lambda_low = 0.0
    lambda_high = 1.0

    for _ in range(
        cfg.max_bisection_iter
    ):
        q_high = solve_q_given_lambda(
            a_q,
            b_q,
            lambda_high,
        )

        power_high = (
            np.linalg.norm(
                q_high,
                "fro",
            ) ** 2
        )

        if (
            power_high
            <= cfg.p_dl_max
        ):
            break

        lambda_high *= 2.0

    else:
        raise RuntimeError(
            "Failed to bracket lambda_P."
        )

    # ----------------------------------------------------------
    # 二分搜索 lambda_P
    # ----------------------------------------------------------

    q_candidate = q_high
    lambda_candidate = lambda_high

    for _ in range(
        cfg.max_bisection_iter
    ):
        lambda_mid = (
            0.5
            * (
                lambda_low
                + lambda_high
            )
        )

        q_mid = solve_q_given_lambda(
            a_q,
            b_q,
            lambda_mid,
        )

        power_mid = (
            np.linalg.norm(
                q_mid,
                "fro",
            ) ** 2
        )

        q_candidate = q_mid
        lambda_candidate = lambda_mid

        if abs(
            power_mid
            - cfg.p_dl_max
        ) <= cfg.tol_bisection:
            break

        # lambda 越大，Q 的功率越小
        if (
            power_mid
            > cfg.p_dl_max
        ):
            lambda_low = lambda_mid
        else:
            lambda_high = lambda_mid

    final_power = (
        np.linalg.norm(
            q_candidate,
            "fro",
        ) ** 2
    )

    if (
        final_power
        > cfg.p_dl_max
        + 1e-8
    ):
        raise RuntimeError(
            "Q update violates "
            "the BS power constraint."
        )

    return (
        q_candidate,
        float(lambda_candidate),
    )


# ============================================================
# 7. 更新上行发射幅度 q_j
# ============================================================

def update_uplink_amplitudes(
    channels: ChannelState,
    resources: ResourceState,
    fp_state: FPState,
    cfg: SystemConfig,
) -> np.ndarray:
    """利用分段闭式解更新全部 UL 发射幅度 q_j。"""

    weight_ul_bar = (
        cfg.weight_ul
        / np.log(2.0)
    )

    weight_s_bar = (
        cfg.weight_s
        / np.log(2.0)
    )

    q_new = np.empty(
        cfg.j_ul,
        dtype=float,
    )

    for j in range(cfg.j_ul):
        # 线性系数 mu_j
        mu_j = (
            weight_ul_bar[j]
            * (
                1.0
                + fp_state.eta_ul[j]
            )
            * np.real(
                np.conj(
                    fp_state.phi_ul[j]
                )
                * np.vdot(
                    resources.b_ul[j],
                    channels.f[j],
                )
            )
        )

        # 二次系数 nu_j
        nu_j = 0.0

        for i in range(cfg.j_ul):
            nu_j += (
                weight_ul_bar[i]
                * (
                    1.0
                    + fp_state.eta_ul[i]
                )
                * np.abs(
                    fp_state.phi_ul[i]
                ) ** 2
                * np.abs(
                    np.vdot(
                        resources.b_ul[i],
                        channels.f[j],
                    )
                ) ** 2
            )

        for ell in range(cfg.l_target):
            nu_j += (
                weight_s_bar[ell]
                * (
                    1.0
                    + fp_state.eta_s[ell]
                )
                * np.linalg.norm(
                    fp_state.phi_s[ell]
                ) ** 2
                * np.abs(
                    np.vdot(
                        resources.u_s[ell],
                        channels.f[j],
                    )
                ) ** 2
            )

        # 当前统一使用 phi_ul，不再使用旧符号 y_ul
        nu_j = float(
            np.real(nu_j)
        )

        if nu_j < -1e-12:
            raise ValueError(
                "nu_j became negative."
            )

        nu_j = max(
            0.0,
            nu_j,
        )

        q_max = np.sqrt(
            cfg.p_ul_max[j]
        )

        if nu_j <= 1e-14:
            # 退化为线性函数
            q_new[j] = (
                q_max
                if mu_j > 0.0
                else 0.0
            )
        else:
            # 将无约束驻点 mu_j / nu_j 投影到合法区间
            q_new[j] = np.clip(
                mu_j / nu_j,
                0.0,
                q_max,
            )

    return q_new


# ============================================================
# 8. 完整内层求解器
# ============================================================

def solve_inner_problem(
    channels: ChannelState,
    resources: ResourceState,
    cfg: SystemConfig,
    verbose: bool = False,
) -> tuple[
    ResourceState,
    list[float],
]:
    """固定 ToMA 和信道，执行完整内层 FP 资源优化。"""

    optimized_resources = (
        copy_resources(resources)
    )

    check_resource_constraints(
        optimized_resources,
        cfg,
    )

    # 初始真实 WSR
    performance = compute_performance(
        channels,
        optimized_resources,
        cfg,
    )

    previous_wsr = (
        performance.weighted_sum_rate
    )

    wsr_history = [
        previous_wsr
    ]

    if verbose:
        print(
            f"Inner iter 0: "
            f"WSR = {previous_wsr:.10f}"
        )

    for inner_iter in range(
        1,
        cfg.max_inner_iter + 1,
    ):
        # 1. sensing 接收波束
        optimized_resources.u_s = (
            update_sensing_combiners(
                channels,
                optimized_resources,
                cfg,
            )
        )

        # 2. UL 接收波束
        optimized_resources.b_ul = (
            update_uplink_combiners(
                channels,
                optimized_resources,
                cfg,
            )
        )

        # 3. LDT
        (
            eta_dl,
            eta_ul,
            eta_s,
        ) = update_ldt_variables(
            channels,
            optimized_resources,
            cfg,
        )

        # 4. QT
        (
            phi_dl,
            phi_ul,
            phi_s,
        ) = update_qt_variables(
            channels,
            optimized_resources,
            cfg,
        )

        fp_state = FPState(
            eta_dl=eta_dl,
            eta_ul=eta_ul,
            eta_s=eta_s,
            phi_dl=phi_dl,
            phi_ul=phi_ul,
            phi_s=phi_s,
        )

        # 5. 构造 Q 子问题
        a_q, b_q = build_q_matrices(
            channels,
            optimized_resources,
            fp_state,
            cfg,
        )

        # 6. 利用 lambda_P 更新 Q
        (
            q_matrix_new,
            lambda_p,
        ) = update_transmit_matrix(
            a_q,
            b_q,
            cfg,
        )

        optimized_resources.q_matrix = (
            q_matrix_new
        )

        # 7. 更新 UL 发射幅度
        optimized_resources.q_ul = (
            update_uplink_amplitudes(
                channels,
                optimized_resources,
                fp_state,
                cfg,
            )
        )

        check_resource_constraints(
            optimized_resources,
            cfg,
        )

        # 8. 用原问题的真实 WSR 判断收敛
        performance = compute_performance(
            channels,
            optimized_resources,
            cfg,
        )

        current_wsr = (
            performance.weighted_sum_rate
        )

        wsr_history.append(
            current_wsr
        )

        if verbose:
            print(
                f"Inner iter {inner_iter}: "
                f"WSR = {current_wsr:.10f}, "
                f"lambda_P = {lambda_p:.6e}"
            )

        # 精确分块更新理论上应使真实 WSR 非下降
        decrease_tolerance = (
            1e-8
            * max(
                1.0,
                abs(previous_wsr),
            )
        )

        if (
            current_wsr
            < previous_wsr
            - decrease_tolerance
        ):
            raise RuntimeError(
                "Inner WSR decreased "
                "more than numerical tolerance."
            )

        change = relative_change(
            current_wsr,
            previous_wsr,
        )

        if (
            change
            <= cfg.tol_inner
        ):
            if verbose:
                print(
                    "Inner solver converged "
                    f"at iteration {inner_iter}."
                )

            break

        previous_wsr = current_wsr

    else:
        if verbose:
            print(
                "Inner solver reached "
                "max_inner_iter."
            )

    return (
        optimized_resources,
        wsr_history,
    )