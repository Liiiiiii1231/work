"""Stage 8B-2：MA + FP 对比方案。

B6: MA + FP Resource Optimization

定义：
- Tx / Rx 都从与 B3 相同的固定 UPA 初始化；
- Tx / Rx 阵元分别位于两个固定 x 平面的二维可移动区域；
- 每个阵元只在 y-z 平面独立移动；
- 同阵列阵元满足最小间距约束；
- Tx/Rx 直接 SI 继续使用真实阵元距离；
- 资源侧完全复用现有 FP；
- 位置侧使用投影梯度 + Armijo 回溯，不使用 ToMA 的球面 RCG。
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy

import numpy as np

from baselines import build_fpa_upa_geometry
from channels import build_channels
from config import SystemConfig
from initialization import initialize_resources
from inner_solver import solve_inner_problem
from metrics import compute_performance
from state import (
    ChannelState,
    GeometryState,
    PerformanceResult,
    ResourceState,
)
from utils import check_resource_constraints, relative_change


@dataclass
class MABaselineResult:
    """保存 MA + FP 单次运行结果。"""

    name: str
    geometry: GeometryState
    channels: ChannelState
    resources: ResourceState
    performance: PerformanceResult
    wsr_history: list[float]
    converged: bool
    iterations: int
    stop_reason: str
    initial_tx_positions: np.ndarray
    initial_rx_positions: np.ndarray


def _build_ma_geometry(
    tx_positions: np.ndarray,
    rx_positions: np.ndarray,
    cfg: SystemConfig,
) -> GeometryState:
    """直接由 MA 阵元坐标构造 GeometryState。"""

    tx_positions = np.asarray(tx_positions, dtype=float)
    rx_positions = np.asarray(rx_positions, dtype=float)

    if tx_positions.shape != (cfg.n_tx, 3):
        raise ValueError(
            f"tx_positions.shape={tx_positions.shape}, "
            f"expected {(cfg.n_tx, 3)}."
        )

    if rx_positions.shape != (cfg.n_rx, 3):
        raise ValueError(
            f"rx_positions.shape={rx_positions.shape}, "
            f"expected {(cfg.n_rx, 3)}."
        )

    si_distances = np.linalg.norm(
        rx_positions[:, None, :]
        - tx_positions[None, :, :],
        axis=2,
    )

    if np.min(si_distances) < cfg.min_si_distance:
        raise ValueError("MA Tx/Rx element distance is too small.")

    return GeometryState(
        endpoints=np.empty((0, 3), dtype=float),
        tx_positions=tx_positions.copy(),
        rx_positions=rx_positions.copy(),
    )


def _minimum_pairwise_distance(
    positions: np.ndarray,
) -> float:
    """返回同一阵列内任意两个不同阵元的最小距离。"""

    if len(positions) <= 1:
        return np.inf

    difference = (
        positions[:, None, :]
        - positions[None, :, :]
    )
    distances = np.linalg.norm(difference, axis=2)
    np.fill_diagonal(distances, np.inf)
    return float(np.min(distances))


def _position_in_region(
    position: np.ndarray,
    center: np.ndarray,
    region_size: float,
    atol: float = 1e-12,
) -> bool:
    """检查一个阵元是否位于指定 y-z 正方形移动区域内。"""

    half = 0.5 * region_size

    return bool(
        np.isclose(position[0], center[0], atol=atol)
        and center[1] - half - atol <= position[1] <= center[1] + half + atol
        and center[2] - half - atol <= position[2] <= center[2] + half + atol
    )


def _ma_geometry_is_feasible(
    tx_positions: np.ndarray,
    rx_positions: np.ndarray,
    tx_center: np.ndarray,
    rx_center: np.ndarray,
    region_size: float,
    min_spacing: float,
    cfg: SystemConfig,
) -> bool:
    """检查 MA 移动区域、阵元间距和直接 SI 数值安全。"""

    if not all(
        _position_in_region(position, tx_center, region_size)
        for position in tx_positions
    ):
        return False

    if not all(
        _position_in_region(position, rx_center, region_size)
        for position in rx_positions
    ):
        return False

    if _minimum_pairwise_distance(tx_positions) < min_spacing - 1e-12:
        return False

    if _minimum_pairwise_distance(rx_positions) < min_spacing - 1e-12:
        return False

    si_distances = np.linalg.norm(
        rx_positions[:, None, :]
        - tx_positions[None, :, :],
        axis=2,
    )

    return bool(np.min(si_distances) >= cfg.min_si_distance)



def _randomize_initial_ma_geometry(
    base_geometry: GeometryState,
    tx_center: np.ndarray,
    rx_center: np.ndarray,
    region_size: float,
    min_spacing: float,
    initial_jitter_fraction: float,
    cfg: SystemConfig,
    rng: np.random.Generator,
) -> GeometryState:
    """在固定 UPA 周围生成一组随机可行 MA 初始位置。

    - Tx/Rx 仍位于各自固定 x 平面；
    - 只对 y-z 坐标加随机扰动；
    - 扰动范围由 initial_jitter_fraction * (region_size/2) 控制；
    - 最终必须满足移动区域、同阵列最小间距和 Tx/Rx SI 数值安全。
    """

    if not (0.0 < initial_jitter_fraction <= 1.0):
        raise ValueError(
            "initial_jitter_fraction must lie in (0, 1]."
        )

    jitter_half = (
        0.5 * float(region_size) * float(initial_jitter_fraction)
    )
    half = 0.5 * float(region_size)

    for _ in range(int(cfg.max_geometry_trials)):
        tx_candidate = base_geometry.tx_positions.copy()
        rx_candidate = base_geometry.rx_positions.copy()

        tx_candidate[:, 1:] += rng.uniform(
            low=-jitter_half,
            high=jitter_half,
            size=(cfg.n_tx, 2),
        )
        rx_candidate[:, 1:] += rng.uniform(
            low=-jitter_half,
            high=jitter_half,
            size=(cfg.n_rx, 2),
        )

        tx_candidate[:, 1] = np.clip(
            tx_candidate[:, 1],
            tx_center[1] - half,
            tx_center[1] + half,
        )
        tx_candidate[:, 2] = np.clip(
            tx_candidate[:, 2],
            tx_center[2] - half,
            tx_center[2] + half,
        )
        rx_candidate[:, 1] = np.clip(
            rx_candidate[:, 1],
            rx_center[1] - half,
            rx_center[1] + half,
        )
        rx_candidate[:, 2] = np.clip(
            rx_candidate[:, 2],
            rx_center[2] - half,
            rx_center[2] + half,
        )

        # 固定两个 MA 平面的 x 坐标。
        tx_candidate[:, 0] = tx_center[0]
        rx_candidate[:, 0] = rx_center[0]

        if _ma_geometry_is_feasible(
            tx_candidate,
            rx_candidate,
            tx_center,
            rx_center,
            region_size,
            min_spacing,
            cfg,
        ):
            return _build_ma_geometry(
                tx_candidate,
                rx_candidate,
                cfg,
            )

    raise RuntimeError(
        "Failed to generate a feasible randomized MA initial geometry."
    )

def _evaluate_ma_wsr(
    tx_positions: np.ndarray,
    rx_positions: np.ndarray,
    resources: ResourceState,
    cfg: SystemConfig,
) -> tuple[float, GeometryState, ChannelState]:
    """固定 FP resources，计算候选 MA 几何的真实 WSR。"""

    geometry = _build_ma_geometry(
        tx_positions,
        rx_positions,
        cfg,
    )
    channels = build_channels(geometry, cfg)
    performance = compute_performance(
        channels,
        resources,
        cfg,
    )

    return (
        float(performance.weighted_sum_rate),
        geometry,
        channels,
    )


def _element_gradient_yz(
    tx_positions: np.ndarray,
    rx_positions: np.ndarray,
    resources: ResourceState,
    cfg: SystemConfig,
    group: str,
    index: int,
    center: np.ndarray,
    region_size: float,
) -> np.ndarray:
    """用中央差分/边界单边差分计算单阵元的 y-z 梯度。"""

    delta = float(cfg.finite_diff_delta)
    half = 0.5 * region_size

    if delta <= 0.0:
        raise ValueError("finite_diff_delta must be positive.")

    positions = tx_positions if group == "tx" else rx_positions

    base_wsr, _, _ = _evaluate_ma_wsr(
        tx_positions,
        rx_positions,
        resources,
        cfg,
    )

    gradient = np.zeros(2, dtype=float)

    for local_axis, global_axis in enumerate((1, 2)):
        lower = center[global_axis] - half
        upper = center[global_axis] + half
        value = positions[index, global_axis]

        plus_ok = value + delta <= upper
        minus_ok = value - delta >= lower

        if plus_ok:
            tx_plus = tx_positions.copy()
            rx_plus = rx_positions.copy()
            target = tx_plus if group == "tx" else rx_plus
            target[index, global_axis] += delta
            f_plus, _, _ = _evaluate_ma_wsr(
                tx_plus,
                rx_plus,
                resources,
                cfg,
            )
        else:
            f_plus = base_wsr

        if minus_ok:
            tx_minus = tx_positions.copy()
            rx_minus = rx_positions.copy()
            target = tx_minus if group == "tx" else rx_minus
            target[index, global_axis] -= delta
            f_minus, _, _ = _evaluate_ma_wsr(
                tx_minus,
                rx_minus,
                resources,
                cfg,
            )
        else:
            f_minus = base_wsr

        if plus_ok and minus_ok:
            gradient[local_axis] = (
                f_plus - f_minus
            ) / (2.0 * delta)
        elif plus_ok:
            gradient[local_axis] = (
                f_plus - base_wsr
            ) / delta
        elif minus_ok:
            gradient[local_axis] = (
                base_wsr - f_minus
            ) / delta

    return gradient


def _update_ma_positions_once(
    geometry: GeometryState,
    resources: ResourceState,
    cfg: SystemConfig,
    tx_center: np.ndarray,
    rx_center: np.ndarray,
    region_size: float,
    min_spacing: float,
    initial_step: float,
) -> tuple[GeometryState, ChannelState, float, int]:
    """固定 resources，对所有 Tx/Rx MA 阵元做一次位置 sweep。"""

    tx_positions = geometry.tx_positions.copy()
    rx_positions = geometry.rx_positions.copy()

    current_wsr, current_geometry, current_channels = _evaluate_ma_wsr(
        tx_positions,
        rx_positions,
        resources,
        cfg,
    )

    accepted_count = 0

    for group, center, n_elements in (
        ("tx", tx_center, cfg.n_tx),
        ("rx", rx_center, cfg.n_rx),
    ):
        for index in range(n_elements):
            gradient = _element_gradient_yz(
                tx_positions,
                rx_positions,
                resources,
                cfg,
                group=group,
                index=index,
                center=center,
                region_size=region_size,
            )

            grad_norm = float(np.linalg.norm(gradient))

            if grad_norm <= 1.0e-12:
                continue

            # 归一化后，step 直接表示实际移动距离的量级（m）。
            direction = gradient / grad_norm
            step = float(initial_step)

            for _ in range(cfg.max_backtracking_iter):
                tx_candidate = tx_positions.copy()
                rx_candidate = rx_positions.copy()
                candidate_positions = (
                    tx_candidate if group == "tx" else rx_candidate
                )

                current_position = candidate_positions[index].copy()
                candidate_position = current_position.copy()
                candidate_position[1:] += step * direction

                half = 0.5 * region_size
                candidate_position[1] = np.clip(
                    candidate_position[1],
                    center[1] - half,
                    center[1] + half,
                )
                candidate_position[2] = np.clip(
                    candidate_position[2],
                    center[2] - half,
                    center[2] + half,
                )
                candidate_position[0] = center[0]

                candidate_positions[index] = candidate_position

                displacement = (
                    candidate_position[1:]
                    - current_position[1:]
                )

                if np.linalg.norm(displacement) <= 1.0e-15:
                    step *= cfg.backtracking_factor
                    continue

                if not _ma_geometry_is_feasible(
                    tx_candidate,
                    rx_candidate,
                    tx_center,
                    rx_center,
                    region_size,
                    min_spacing,
                    cfg,
                ):
                    step *= cfg.backtracking_factor
                    continue

                candidate_wsr, candidate_geometry, candidate_channels = (
                    _evaluate_ma_wsr(
                        tx_candidate,
                        rx_candidate,
                        resources,
                        cfg,
                    )
                )

                armijo_rhs = (
                    current_wsr
                    + cfg.armijo_mu
                    * float(np.dot(gradient, displacement))
                )

                if candidate_wsr >= armijo_rhs:
                    tx_positions = tx_candidate
                    rx_positions = rx_candidate
                    current_wsr = candidate_wsr
                    current_geometry = candidate_geometry
                    current_channels = candidate_channels
                    accepted_count += 1
                    break

                step *= cfg.backtracking_factor

    return (
        current_geometry,
        current_channels,
        current_wsr,
        accepted_count,
    )


def run_ma_fp(
    cfg: SystemConfig,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
    spacing: float | None = None,
    tx_rx_separation: float | None = None,
    region_size: float | None = None,
    min_spacing: float | None = None,
    initial_step: float | None = None,
    max_outer_iterations: int | None = None,
    randomize_initial_geometry: bool = False,
    initial_jitter_fraction: float = 0.5,
    initial_tx_positions: np.ndarray | None = None,
    initial_rx_positions: np.ndarray | None = None,
    initial_resources: ResourceState | None = None,
) -> MABaselineResult:
    """运行 B6：MA + FP Resource Optimization。

    公平性：
    - 与 B3 使用同一 UPA 作为初始阵列；
    - 与 Proposed 使用同一个 FP 内层；
    - MA 只改变阵元位置可行域和位置优化方式。
    """

    if rng is None:
        rng = np.random.default_rng(cfg.random_seed)

    if region_size is None:
        region_size = cfg.cable_length

    if min_spacing is None:
        min_spacing = 0.5 * cfg.wavelength

    if initial_step is None:
        initial_step = 0.1 * region_size

    if region_size <= 0.0:
        raise ValueError("MA region_size must be positive.")

    if min_spacing <= 0.0:
        raise ValueError("MA min_spacing must be positive.")

    if initial_step <= 0.0:
        raise ValueError("MA initial_step must be positive.")

    number_of_outer_iterations = (
        cfg.max_outer_iter
        if max_outer_iterations is None
        else int(max_outer_iterations)
    )

    base_geometry = build_fpa_upa_geometry(
        cfg,
        spacing=spacing,
        tx_rx_separation=tx_rx_separation,
    )

    tx_center = np.mean(base_geometry.tx_positions, axis=0)
    rx_center = np.mean(base_geometry.rx_positions, axis=0)

    if not _ma_geometry_is_feasible(
        base_geometry.tx_positions,
        base_geometry.rx_positions,
        tx_center,
        rx_center,
        region_size,
        min_spacing,
        cfg,
    ):
        raise ValueError(
            "Initial UPA is not feasible for the selected MA region/min_spacing."
        )

    if (initial_tx_positions is None) != (initial_rx_positions is None):
        raise ValueError(
            "initial_tx_positions and initial_rx_positions "
            "must be provided together."
        )

    if initial_tx_positions is not None:
        geometry = _build_ma_geometry(
            initial_tx_positions,
            initial_rx_positions,
            cfg,
        )
        if not _ma_geometry_is_feasible(
            geometry.tx_positions,
            geometry.rx_positions,
            tx_center,
            rx_center,
            region_size,
            min_spacing,
            cfg,
        ):
            raise ValueError(
                "Provided MA warm-start geometry is infeasible."
            )
    elif randomize_initial_geometry:
        geometry = _randomize_initial_ma_geometry(
            base_geometry,
            tx_center=tx_center,
            rx_center=rx_center,
            region_size=region_size,
            min_spacing=min_spacing,
            initial_jitter_fraction=initial_jitter_fraction,
            cfg=cfg,
            rng=rng,
        )
    else:
        geometry = base_geometry

    initial_tx_positions_saved = geometry.tx_positions.copy()
    initial_rx_positions_saved = geometry.rx_positions.copy()

    channels = build_channels(geometry, cfg)
    if initial_resources is None:
        resources = initialize_resources(
            geometry,
            channels,
            cfg,
            rng,
        )
    else:
        resources = deepcopy(initial_resources)

    check_resource_constraints(resources, cfg)

    performance = compute_performance(
        channels,
        resources,
        cfg,
    )
    previous_wsr = float(performance.weighted_sum_rate)
    wsr_history = [previous_wsr]

    consecutive_small_changes = 0
    converged = False
    iterations = 0
    stop_reason = "max_outer_iterations"

    plateau_enabled = (
        cfg.plateau_window >= 2
        and cfg.tol_window > 0.0
    )

    for outer_iter in range(1, number_of_outer_iterations + 1):
        resources, _ = solve_inner_problem(
            channels,
            resources,
            cfg,
            verbose=False,
        )

        geometry, channels, _, accepted_count = _update_ma_positions_once(
            geometry,
            resources,
            cfg,
            tx_center=tx_center,
            rx_center=rx_center,
            region_size=region_size,
            min_spacing=min_spacing,
            initial_step=initial_step,
        )

        performance = compute_performance(
            channels,
            resources,
            cfg,
        )
        current_wsr = float(performance.weighted_sum_rate)
        wsr_history.append(current_wsr)

        change = relative_change(
            current_wsr,
            previous_wsr,
        )

        if verbose:
            print(
                f"MA+FP iteration {outer_iter}: "
                f"accepted={accepted_count}/{cfg.n_tx + cfg.n_rx}, "
                f"WSR={current_wsr:.10f}, "
                f"change={change:.3e}"
            )

        if change <= cfg.tol_outer:
            consecutive_small_changes += 1
        else:
            consecutive_small_changes = 0

        iterations = outer_iter

        if consecutive_small_changes >= 2:
            converged = True
            stop_reason = "outer_tolerance"
            break

        if (
            plateau_enabled
            and len(wsr_history) > cfg.plateau_window
        ):
            old_wsr = wsr_history[-1 - cfg.plateau_window]
            window_gain = (
                current_wsr - old_wsr
            ) / max(1.0, abs(old_wsr))

            if 0.0 <= window_gain <= cfg.tol_window:
                converged = True
                stop_reason = "window_plateau"
                break

        previous_wsr = current_wsr

    check_resource_constraints(resources, cfg)

    return MABaselineResult(
        name="MA + FP Resource Optimization",
        geometry=geometry,
        channels=channels,
        resources=resources,
        performance=performance,
        wsr_history=wsr_history,
        converged=converged,
        iterations=iterations,
        stop_reason=stop_reason,
        initial_tx_positions=initial_tx_positions_saved,
        initial_rx_positions=initial_rx_positions_saved,
    )
