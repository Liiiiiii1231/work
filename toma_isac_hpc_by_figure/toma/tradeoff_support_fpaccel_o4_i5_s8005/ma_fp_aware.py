"""Local FP-aware MA adapter. Shared source files are unchanged.
The existing MA loop owns resources; accepted resources are copied into that
object so geometry and resources advance together. Rejected candidates never
write to the current resource state. Candidates use at most two FP steps.
"""
from dataclasses import replace
from functools import partial
from unittest.mock import patch
import numpy as np
import algorithms as shared
from algorithms import (_evaluate_ma_wsr,_element_gradient_yz,
    _project_to_ma_feasible_tangent,_ma_geometry_is_feasible)
from inner_solver import solve_inner_problem
from system_model import GeometryState,ResourceState,ChannelState,SystemConfig

def update_positions_fp_aware(
    geometry: GeometryState,
    resources: ResourceState,
    cfg: SystemConfig,
    tx_center: np.ndarray,
    rx_center: np.ndarray,
    region_size: float,
    min_spacing: float,
    initial_step: float,
    diagnostics: list | None = None,
) -> tuple[GeometryState, ChannelState, float, int]:
    """FP-aware MA sweep; candidate position and resources are accepted together."""

    tx_positions = geometry.tx_positions.copy()
    rx_positions = geometry.rx_positions.copy()

    current_wsr, current_geometry, current_channels = _evaluate_ma_wsr(
        tx_positions,
        rx_positions,
        resources,
        cfg,
    )

    accepted_count = 0
    fp_calls = 0
    fp_steps = 0
    before_wsr = current_wsr
    old_tx = tx_positions.copy()
    old_rx = rx_positions.copy()
    candidate_cfg = replace(cfg, max_inner_iter=2)

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

            # 原始方向仍保留，用于避免改动现有 MA 基线的有效搜索路径。
            # 额外搜索投影到活动几何约束的可行方向。
            original_direction = gradient / grad_norm
            current_positions = tx_positions if group == "tx" else rx_positions
            feasible_grad = _project_to_ma_feasible_tangent(
                gradient, current_positions, index, center,
                region_size, min_spacing,
            )
            feasible_norm = float(np.linalg.norm(feasible_grad))
            candidate_directions = [original_direction]
            if feasible_norm > 1.0e-12:
                feasible_direction = feasible_grad / feasible_norm
                if not np.allclose(feasible_direction, original_direction, atol=1e-10):
                    candidate_directions.append(feasible_direction)

            # 两条方向从同一状态出发。各自进行原有的 Armijo 回溯，
            # 最后接受本次单阵元搜索中真实 WSR 最高的可行候选。
            best_move = None
            best_wsr = current_wsr
            for direction in candidate_directions:
                step = float(initial_step)
                for _ in range(cfg.max_backtracking_iter):
                    tx_candidate = tx_positions.copy()
                    rx_candidate = rx_positions.copy()
                    candidate_positions = (
                        tx_candidate if group == "tx" else rx_candidate
                    )

                    previous_position = candidate_positions[index].copy()
                    candidate_position = previous_position.copy()
                    candidate_position[1:] += step * direction

                    half = 0.5 * region_size
                    candidate_position[1] = np.clip(
                        candidate_position[1],
                        center[1] - half, center[1] + half,
                    )
                    candidate_position[2] = np.clip(
                        candidate_position[2],
                        center[2] - half, center[2] + half,
                    )
                    candidate_position[0] = center[0]
                    candidate_positions[index] = candidate_position

                    displacement = candidate_position[1:] - previous_position[1:]
                    if np.linalg.norm(displacement) <= 1.0e-15:
                        step *= cfg.backtracking_factor
                        continue

                    if not _ma_geometry_is_feasible(
                        tx_candidate, rx_candidate, tx_center, rx_center,
                        region_size, min_spacing, cfg,
                    ):
                        step *= cfg.backtracking_factor
                        continue

                    candidate_wsr, candidate_geometry, candidate_channels = (
                        _evaluate_ma_wsr(
                            tx_candidate, rx_candidate, resources, cfg,
                        )
                    )
                    candidate_resources, candidate_history = solve_inner_problem(
                        candidate_channels, resources, candidate_cfg, verbose=False,
                    )
                    fp_calls += 1
                    fp_steps += len(candidate_history)-1
                    candidate_wsr = float(candidate_history[-1])
                    armijo_rhs = (
                        current_wsr
                        + cfg.armijo_mu * float(np.dot(gradient, displacement))
                    )
                    if candidate_wsr >= max(current_wsr, armijo_rhs):
                        if candidate_wsr > best_wsr:
                            best_wsr = candidate_wsr
                            best_move = (
                                tx_candidate, rx_candidate,
                                candidate_geometry, candidate_channels, candidate_resources,
                            )
                        break
                    step *= cfg.backtracking_factor

            if best_move is not None:
                tx_positions, rx_positions, current_geometry, current_channels, accepted_resources = best_move
                resources.q_matrix = accepted_resources.q_matrix.copy()
                resources.q_ul = accepted_resources.q_ul.copy()
                resources.b_ul = accepted_resources.b_ul.copy()
                resources.u_s = accepted_resources.u_s.copy()
                current_wsr = best_wsr
                accepted_count += 1

    if diagnostics is not None:
        diagnostics.append(dict(candidate_fp_calls=fp_calls,candidate_fp_iterations=fp_steps,
            accepted_positions=accepted_count,initial_wsr=before_wsr,final_wsr=current_wsr,
            max_displacement_m=max(float(np.linalg.norm(tx_positions-old_tx,axis=1).max()),
                                   float(np.linalg.norm(rx_positions-old_rx,axis=1).max()))))
    if diagnostics is not None and len(diagnostics)%25==0:
        print('MA-AWARE PROGRESS',len(diagnostics),'WSR',current_wsr,'accepted',accepted_count,'candidate_FP_calls',fp_calls,flush=True)
    return (
        current_geometry,
        current_channels,
        current_wsr,
        accepted_count,
    )

def run_ma_fp_aware(cfg, **kwargs):
    diagnostics=[]
    updater=partial(update_positions_fp_aware,diagnostics=diagnostics)
    with patch.object(shared,'_update_ma_positions_once',updater):
        result=shared.run_ma_fp(cfg,**kwargs)
    return result,diagnostics
