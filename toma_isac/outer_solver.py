"""固定通信与感知资源下的 ToMA 外层位置优化。

主要步骤：
中央差分 -> 黎曼梯度 -> PR+ RCG -> 球面回缩 -> Armijo。

位置更新顺序为 Tx ToMA 在前，Rx ToMA 在后。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from channels import build_channels
from config import SystemConfig
from geometry import build_geometry
from metrics import compute_performance
from state import (
    ChannelState,
    GeometryState,
    ResourceState,
)
from utils import relative_change
OuterObjective = Callable[[np.ndarray], float]

@dataclass
class OuterRCGState:
    """保存各 ToMA 端点上一轮的 RCG 信息。"""

    previous_gradients: list[np.ndarray | None]
    previous_directions: list[np.ndarray | None]


def create_outer_rcg_state(
    cfg: SystemConfig,
) -> OuterRCGState:
    """创建空的 RCG 历史状态。"""

    return OuterRCGState(
        previous_gradients=[
            None
            for _ in range(cfg.m_uav)
        ],
        previous_directions=[
            None
            for _ in range(cfg.m_uav)
        ],
    )


# ============================================================
# 1. 固定资源下计算真实 WSR
# ============================================================

def evaluate_fixed_resource_wsr(
    endpoints: np.ndarray,
    resources: ResourceState,
    cfg: SystemConfig,
) -> float:
    """给定端点位置，重新生成信道并计算真实 WSR。"""

    # 中央差分点可能暂时离开球面，因此这里不做几何约束检查。
    geometry = build_geometry(
        endpoints,
        cfg,
        check_constraints=False,
    )

    channels = build_channels(
        geometry,
        cfg,
    )

    performance = compute_performance(
        channels,
        resources,
        cfg,
    )

    return float(
        performance.weighted_sum_rate
    )


def _evaluate_outer_objective(
    endpoints: np.ndarray,
    resources: ResourceState,
    cfg: SystemConfig,
    objective_evaluator: OuterObjective | None,
) -> float:
    """计算外层当前位置对应的目标值。"""

    if objective_evaluator is None:
        return evaluate_fixed_resource_wsr(
            endpoints,
            resources,
            cfg,
        )

    return float(
        objective_evaluator(endpoints)
    )
# ============================================================
# 2. 中央数值差分
# ============================================================

def central_difference_gradient(
    endpoints: np.ndarray,
    endpoint_index: int,
    resources: ResourceState,
    cfg: SystemConfig,
    objective_evaluator: OuterObjective | None = None,
) -> np.ndarray:
    """计算单个 ToMA 端点的三维欧氏数值梯度。"""

    delta = cfg.finite_diff_delta

    if delta <= 0:
        raise ValueError(
            "finite_diff_delta must be positive."
        )

    gradient = np.zeros(
        3,
        dtype=float,
    )

    for axis in range(3):
        plus_endpoints = endpoints.copy()
        minus_endpoints = endpoints.copy()

        plus_endpoints[
            endpoint_index,
            axis,
        ] += delta

        minus_endpoints[
            endpoint_index,
            axis,
        ] -= delta

        f_plus = _evaluate_outer_objective(
            plus_endpoints,
            resources,
            cfg,
            objective_evaluator,
        )

        f_minus = _evaluate_outer_objective(
            minus_endpoints,
            resources,
            cfg,
            objective_evaluator,
        )

        gradient[axis] = (
            f_plus
            - f_minus
        ) / (
            2.0 * delta
        )

    return gradient


# ============================================================
# 3. 球面切空间
# ============================================================

def project_to_tangent(
    point: np.ndarray,
    vector: np.ndarray,
    cfg: SystemConfig,
) -> np.ndarray:
    """把三维向量投影到端点所在球面的切空间。"""

    cable_length_sq = (
        cfg.cable_length ** 2
    )

    return (
        vector
        - point
        * (
            np.dot(
                point,
                vector,
            )
            / cable_length_sq
        )
    )


def vector_transport(
    vector: np.ndarray,
    new_point: np.ndarray,
    cfg: SystemConfig,
) -> np.ndarray:
    """用切空间投影实现 RCG 的向量传输。"""

    return project_to_tangent(
        new_point,
        vector,
        cfg,
    )


# ============================================================
# 4. Polak-Ribiere+ 共轭方向
# ============================================================

def compute_rcg_direction(
    point: np.ndarray,
    gradient: np.ndarray,
    previous_gradient: np.ndarray | None,
    previous_direction: np.ndarray | None,
    cfg: SystemConfig,
) -> tuple[
    np.ndarray,
    float,
]:
    """计算 PR+ 黎曼共轭上升方向。"""

    # 第一次更新没有历史信息，直接使用梯度方向。
    if (
        previous_gradient is None
        or previous_direction is None
    ):
        return gradient.copy(), 0.0

    transported_gradient = vector_transport(
        previous_gradient,
        point,
        cfg,
    )

    transported_direction = vector_transport(
        previous_direction,
        point,
        cfg,
    )

    denominator = np.dot(
        previous_gradient,
        previous_gradient,
    )

    if denominator <= 1e-16:
        return gradient.copy(), 0.0

    beta = np.dot(
        gradient,
        gradient
        - transported_gradient,
    ) / denominator

    # Polak-Ribiere+
    beta = max(
        0.0,
        float(beta),
    )

    direction = (
        gradient
        + beta
        * transported_direction
    )

    direction = project_to_tangent(
        point,
        direction,
        cfg,
    )

    # 如果不再是上升方向，则重新启动为梯度方向。
    if (
        np.dot(
            gradient,
            direction,
        )
        <= 0.0
    ):
        return gradient.copy(), 0.0

    return direction, beta


# ============================================================
# 5. 球面回缩
# ============================================================

def sphere_retraction(
    point: np.ndarray,
    direction: np.ndarray,
    step_size: float,
    cfg: SystemConfig,
) -> np.ndarray:
    """沿切向方向移动，并重新回缩到缆绳定长球面。"""

    trial_point = (
        point
        + step_size
        * direction
    )

    norm = np.linalg.norm(
        trial_point
    )

    if norm <= 1e-15:
        raise ValueError(
            "Invalid sphere retraction."
        )

    return (
        cfg.cable_length
        * trial_point
        / norm
    )


# ============================================================
# 6. 防碰撞检查
# ============================================================

def is_collision_free(
    endpoints: np.ndarray,
    endpoint_index: int,
    candidate: np.ndarray,
    cfg: SystemConfig,
) -> bool:
    """检查候选子无人机与其他端点是否满足最小间距。"""

    for other_index in range(
        cfg.m_uav
    ):
        if (
            other_index
            == endpoint_index
        ):
            continue

        distance = np.linalg.norm(
            candidate
            - endpoints[other_index]
        )

        if (
            distance
            < cfg.min_uav_distance
        ):
            return False

    return True


# ============================================================
# 7. Armijo 回溯
# ============================================================

def armijo_endpoint_update(
    endpoints: np.ndarray,
    endpoint_index: int,
    gradient: np.ndarray,
    direction: np.ndarray,
    current_wsr: float,
    resources: ResourceState,
    cfg: SystemConfig,
    objective_evaluator: OuterObjective | None = None,
) -> tuple[
    np.ndarray,
    float,
    bool,
    float,
]:
    """对一个 ToMA 端点执行防碰撞 + Armijo 回溯。"""

    current_point = endpoints[
        endpoint_index
    ]

    step_size = (
        cfg.initial_outer_step
    )

    directional_derivative = float(
        np.dot(
            gradient,
            direction,
        )
    )

    for _ in range(
        cfg.max_backtracking_iter
    ):
        candidate = sphere_retraction(
            current_point,
            direction,
            step_size,
            cfg,
        )

        # 候选位置首先满足防碰撞要求。
        if not is_collision_free(
            endpoints,
            endpoint_index,
            candidate,
            cfg,
        ):
            step_size *= (
                cfg.backtracking_factor
            )
            continue

        candidate_endpoints = (
            endpoints.copy()
        )

        candidate_endpoints[
            endpoint_index
        ] = candidate

        # 资源保持固定，只重新生成几何、信道和真实 WSR。
        candidate_wsr = _evaluate_outer_objective(
            candidate_endpoints,
            resources,
            cfg,
            objective_evaluator,
        )

        armijo_rhs = (
            current_wsr
            + cfg.armijo_mu
            * step_size
            * directional_derivative
        )

        if (
            candidate_wsr
            >= armijo_rhs
        ):
            return (
                candidate,
                candidate_wsr,
                True,
                step_size,
            )

        step_size *= (
            cfg.backtracking_factor
        )

    # 回溯失败时保持原位置。
    return (
        current_point.copy(),
        current_wsr,
        False,
        0.0,
    )


# ============================================================
# 8. 一次完整 Tx -> Rx ToMA 更新
# ============================================================

def update_positions_once(
    geometry: GeometryState,
    resources: ResourceState,
    cfg: SystemConfig,
    rcg_state: OuterRCGState | None = None,
    verbose: bool = False,
    objective_evaluator: OuterObjective | None = None,
) -> tuple[
    GeometryState,
    ChannelState,
    float,
    OuterRCGState,
    int,
]:
    """按 Tx -> Rx 顺序更新全部 ToMA 端点一次。

    objective_evaluator=None 时保持原 Stage 3/4 行为：
    固定 resources 评价候选位置。
    """

    if rcg_state is None:
        rcg_state = (
            create_outer_rcg_state(
                cfg
            )
        )

    endpoints = (
        geometry.endpoints.copy()
    )

    current_wsr = _evaluate_outer_objective(
    endpoints,
    resources,
    cfg,
    objective_evaluator,
    )

    accepted_count = 0

    # 先 Tx 端点，再 Rx 端点。
    endpoint_order = (
        list(
            range(
                cfg.n_tx_uav
            )
        )
        + list(
            range(
                cfg.n_tx_uav,
                cfg.m_uav,
            )
        )
    )

    for endpoint_index in endpoint_order:
        # 当前端点的欧氏数值梯度
        euclidean_gradient = (
    central_difference_gradient(
        endpoints,
        endpoint_index,
        resources,
        cfg,
        objective_evaluator=objective_evaluator,
    )
    )

        # 投影为黎曼梯度
        grad_m = project_to_tangent(
            endpoints[
                endpoint_index
            ],
            euclidean_gradient,
            cfg,
        )

        # 梯度几乎为零时不再试探。
        if (
            np.linalg.norm(grad_m)
            <= 1e-12
        ):
            rcg_state.previous_gradients[
                endpoint_index
            ] = grad_m.copy()

            rcg_state.previous_directions[
                endpoint_index
            ] = grad_m.copy()

            continue

        direction, beta = (
            compute_rcg_direction(
                endpoints[
                    endpoint_index
                ],
                grad_m,
                rcg_state
                .previous_gradients[
                    endpoint_index
                ],
                rcg_state
                .previous_directions[
                    endpoint_index
                ],
                cfg,
            )
        )

        (
            candidate,
            candidate_wsr,
            accepted,
            step_size,
        ) = armijo_endpoint_update(
            endpoints,
            endpoint_index,
            grad_m,
            direction,
            current_wsr,
            resources,
            cfg,
            objective_evaluator=objective_evaluator,
        )

        # 保存本轮 RCG 信息，供下一轮使用。
        rcg_state.previous_gradients[
            endpoint_index
        ] = grad_m.copy()

        rcg_state.previous_directions[
            endpoint_index
        ] = direction.copy()

        if accepted:
            # Gauss-Seidel：接受后立即更新，后续端点使用新位置。
            endpoints[
                endpoint_index
            ] = candidate

            current_wsr = (
                candidate_wsr
            )

            accepted_count += 1

        if verbose:
            group = (
                "Tx"
                if endpoint_index
                < cfg.n_tx_uav
                else "Rx"
            )

            print(
                f"  {group} endpoint "
                f"{endpoint_index}: "
                f"accepted={accepted}, "
                f"beta={beta:.3e}, "
                f"step={step_size:.3e}, "
                f"WSR={current_wsr:.10f}"
            )

    final_geometry = build_geometry(
        endpoints,
        cfg,
        check_constraints=False,
    )

    final_channels = build_channels(
        final_geometry,
        cfg,
    )

    return (
        final_geometry,
        final_channels,
        current_wsr,
        rcg_state,
        accepted_count,
    )


# ============================================================
# 9. 第三阶段独立测试求解器
# ============================================================

def solve_outer_problem(
    geometry: GeometryState,
    resources: ResourceState,
    cfg: SystemConfig,
    verbose: bool = False,
    max_sweeps: int | None = None,
) -> tuple[
    GeometryState,
    ChannelState,
    list[float],
]:
    """固定资源，多轮执行 ToMA 位置更新，用于第三阶段独立验证。"""

    number_of_sweeps = (
        cfg.max_outer_iter
        if max_sweeps is None
        else max_sweeps
    )

    rcg_state = (
        create_outer_rcg_state(
            cfg
        )
    )

    current_geometry = geometry

    current_channels = build_channels(
        current_geometry,
        cfg,
    )

    performance = compute_performance(
        current_channels,
        resources,
        cfg,
    )

    previous_wsr = float(
        performance.weighted_sum_rate
    )

    wsr_history = [
        previous_wsr
    ]

    consecutive_small_changes = 0

    if verbose:
        print(
            f"Outer sweep 0: "
            f"WSR = {previous_wsr:.10f}"
        )

    for sweep in range(
        1,
        number_of_sweeps + 1,
    ):
        (
            current_geometry,
            current_channels,
            current_wsr,
            rcg_state,
            accepted_count,
        ) = update_positions_once(
            current_geometry,
            resources,
            cfg,
            rcg_state=rcg_state,
            verbose=verbose,
        )

        wsr_history.append(
            current_wsr
        )

        if (
            current_wsr
            < previous_wsr
            - 1e-8
        ):
            raise RuntimeError(
                "Outer WSR decreased "
                "more than numerical tolerance."
            )

        change = relative_change(
            current_wsr,
            previous_wsr,
        )

        if verbose:
            print(
                f"Outer sweep {sweep}: "
                f"WSR = {current_wsr:.10f}, "
                f"accepted = {accepted_count}, "
                f"relative change = {change:.3e}"
            )

        # 外层相对变化连续两次满足条件才停止。
        if (
            change
            <= cfg.tol_outer
        ):
            consecutive_small_changes += 1
        else:
            consecutive_small_changes = 0

        if (
            consecutive_small_changes
            >= 2
        ):
            if verbose:
                print(
                    "Outer solver converged "
                    f"at sweep {sweep}."
                )
            break

        previous_wsr = current_wsr

    return (
        current_geometry,
        current_channels,
        wsr_history,
    )