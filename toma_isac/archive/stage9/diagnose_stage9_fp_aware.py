"""Stage 9：原始 Armijo 与 FP-aware Armijo 的固定场景 A/B 诊断。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from channels import build_channels
from config import SystemConfig, load_config
from geometry import build_geometry
from initialization import generate_feasible_endpoints, initialize_resources
from inner_solver import solve_inner_problem
from metrics import compute_performance
from outer_solver import (
    EndpointRCGDiagnostic,
    OuterRCGState,
    central_difference_gradient,
    compute_rcg_direction,
    create_outer_rcg_state,
    is_collision_free,
    project_to_tangent,
    sphere_retraction,
    update_positions_once,
)
from stage8 import make_scene_config
from state import ChannelState, GeometryState, ResourceState
from utils import check_resource_constraints, relative_change


CONFIG_PATH = ROOT / "configs" / "stage9_outer_debug.yaml"
RESULT_DIR = ROOT / "results" / "stage9"
DEFAULT_POSITION_JITTER = np.array([1.5, 1.5, 0.25], dtype=float)


@dataclass(frozen=True)
class AwareEndpointDiagnostic:
    endpoint_index: int
    group: str
    grad_norm: float
    beta: float
    accepted: bool
    step_size: float
    backtracking_steps: int
    wsr_after_update: float
    fixed_wsr_at_candidate: float
    aware_wsr_at_candidate: float
    candidate_fp_iterations: int


@dataclass
class DiagnosticRunResult:
    mode: str
    final_geometry: GeometryState
    final_channels: ChannelState
    final_resources: ResourceState
    joint_wsr_history: list[float]
    outer_rows: list[dict[str, float | int | str]]
    endpoint_rows: list[dict[str, float | int | str | bool]]
    converged: bool
    outer_iterations: int
    runtime_s: float

    @property
    def final_wsr(self) -> float:
        return float(self.joint_wsr_history[-1])


def _build_scene_cfg(
    scene_seed: int,
) -> SystemConfig:
    cfg = load_config(CONFIG_PATH)

    if not np.isclose(cfg.rho_si, 1.0e-10):
        raise ValueError("Stage 9 requires rho_si = 1e-10.")
    if cfg.plateau_window != 0 or cfg.tol_window != 0.0:
        raise ValueError("Stage 9 requires window plateau stop to be disabled.")

    return make_scene_config(
        cfg,
        scene_seed=scene_seed,
        position_jitter=DEFAULT_POSITION_JITTER,
        randomize_fading=False,
    )


def _evaluate_wsr(
    endpoints: np.ndarray,
    resources: ResourceState,
    cfg: SystemConfig,
) -> tuple[GeometryState, ChannelState, float]:
    geometry = build_geometry(
        endpoints,
        cfg,
        check_constraints=False,
    )
    channels = build_channels(geometry, cfg)
    performance = compute_performance(channels, resources, cfg)
    return geometry, channels, float(performance.weighted_sum_rate)


def fp_aware_armijo_endpoint_update(
    endpoints: np.ndarray,
    endpoint_index: int,
    gradient: np.ndarray,
    direction: np.ndarray,
    current_wsr: float,
    resources: ResourceState,
    cfg: SystemConfig,
    candidate_fp_iterations: int,
) -> tuple[
    np.ndarray,
    ResourceState,
    float,
    bool,
    float,
    int,
    float,
    float,
    int,
]:
    """只改 Armijo 候选点评价：新位置先 warm-start FP 再判断。"""

    if candidate_fp_iterations <= 0:
        raise ValueError("candidate_fp_iterations must be positive.")

    current_point = endpoints[endpoint_index]
    step_size = cfg.initial_outer_step
    directional_derivative = float(np.dot(gradient, direction))

    candidate_cfg = replace(
        cfg,
        max_inner_iter=candidate_fp_iterations,
    )

    last_fixed_wsr = float("nan")
    last_aware_wsr = float("nan")
    last_fp_iters = 0

    for backtracking_steps in range(cfg.max_backtracking_iter):
        candidate = sphere_retraction(
            current_point,
            direction,
            step_size,
            cfg,
        )

        if not is_collision_free(
            endpoints,
            endpoint_index,
            candidate,
            cfg,
        ):
            step_size *= cfg.backtracking_factor
            continue

        candidate_endpoints = endpoints.copy()
        candidate_endpoints[endpoint_index] = candidate

        candidate_geometry, candidate_channels, fixed_wsr = _evaluate_wsr(
            candidate_endpoints,
            resources,
            cfg,
        )

        candidate_resources, inner_history = solve_inner_problem(
            candidate_channels,
            resources,
            candidate_cfg,
            verbose=False,
        )

        aware_wsr = float(inner_history[-1])
        fp_iters = len(inner_history) - 1

        last_fixed_wsr = fixed_wsr
        last_aware_wsr = aware_wsr
        last_fp_iters = fp_iters

        armijo_rhs = (
            current_wsr
            + cfg.armijo_mu
            * step_size
            * directional_derivative
        )

        if aware_wsr >= armijo_rhs:
            return (
                candidate,
                candidate_resources,
                aware_wsr,
                True,
                step_size,
                backtracking_steps,
                fixed_wsr,
                aware_wsr,
                fp_iters,
            )

        step_size *= cfg.backtracking_factor

    return (
        current_point.copy(),
        resources,
        current_wsr,
        False,
        0.0,
        cfg.max_backtracking_iter,
        last_fixed_wsr,
        last_aware_wsr,
        last_fp_iters,
    )


def update_positions_once_fp_aware(
    geometry: GeometryState,
    resources: ResourceState,
    cfg: SystemConfig,
    rcg_state: OuterRCGState,
    candidate_fp_iterations: int,
    diagnostics: list[AwareEndpointDiagnostic],
) -> tuple[
    GeometryState,
    ChannelState,
    ResourceState,
    float,
    OuterRCGState,
    int,
]:
    """保留原 RCG 梯度/PR+/方向，只把候选点改为 FP-aware 评价。"""

    endpoints = geometry.endpoints.copy()
    current_resources = resources

    _, _, current_wsr = _evaluate_wsr(
        endpoints,
        current_resources,
        cfg,
    )

    accepted_count = 0
    endpoint_order = (
        list(range(cfg.n_tx_uav))
        + list(range(cfg.n_tx_uav, cfg.m_uav))
    )

    for endpoint_index in endpoint_order:
        euclidean_gradient = central_difference_gradient(
            endpoints,
            endpoint_index,
            current_resources,
            cfg,
            objective_evaluator=None,
        )

        grad_m = project_to_tangent(
            endpoints[endpoint_index],
            euclidean_gradient,
            cfg,
        )
        grad_norm = float(np.linalg.norm(grad_m))

        group = "Tx" if endpoint_index < cfg.n_tx_uav else "Rx"

        if grad_norm <= 1e-12:
            rcg_state.previous_gradients[endpoint_index] = grad_m.copy()
            rcg_state.previous_directions[endpoint_index] = grad_m.copy()
            diagnostics.append(
                AwareEndpointDiagnostic(
                    endpoint_index=endpoint_index,
                    group=group,
                    grad_norm=grad_norm,
                    beta=0.0,
                    accepted=False,
                    step_size=0.0,
                    backtracking_steps=0,
                    wsr_after_update=current_wsr,
                    fixed_wsr_at_candidate=float("nan"),
                    aware_wsr_at_candidate=float("nan"),
                    candidate_fp_iterations=0,
                )
            )
            continue

        direction, beta = compute_rcg_direction(
            endpoints[endpoint_index],
            grad_m,
            rcg_state.previous_gradients[endpoint_index],
            rcg_state.previous_directions[endpoint_index],
            cfg,
        )

        (
            candidate,
            candidate_resources,
            candidate_wsr,
            accepted,
            step_size,
            backtracking_steps,
            fixed_wsr,
            aware_wsr,
            fp_iters,
        ) = fp_aware_armijo_endpoint_update(
            endpoints,
            endpoint_index,
            grad_m,
            direction,
            current_wsr,
            current_resources,
            cfg,
            candidate_fp_iterations,
        )

        rcg_state.previous_gradients[endpoint_index] = grad_m.copy()
        rcg_state.previous_directions[endpoint_index] = direction.copy()

        if accepted:
            endpoints[endpoint_index] = candidate
            current_resources = candidate_resources
            current_wsr = candidate_wsr
            accepted_count += 1

        diagnostics.append(
            AwareEndpointDiagnostic(
                endpoint_index=endpoint_index,
                group=group,
                grad_norm=grad_norm,
                beta=float(beta),
                accepted=bool(accepted),
                step_size=float(step_size),
                backtracking_steps=int(backtracking_steps),
                wsr_after_update=float(current_wsr),
                fixed_wsr_at_candidate=float(fixed_wsr),
                aware_wsr_at_candidate=float(aware_wsr),
                candidate_fp_iterations=int(fp_iters),
            )
        )

    final_geometry = build_geometry(
        endpoints,
        cfg,
        check_constraints=False,
    )
    final_channels = build_channels(final_geometry, cfg)

    return (
        final_geometry,
        final_channels,
        current_resources,
        current_wsr,
        rcg_state,
        accepted_count,
    )


def _endpoint_summary(
    diagnostics: list[EndpointRCGDiagnostic | AwareEndpointDiagnostic],
) -> tuple[float, float]:
    accepted = [item for item in diagnostics if item.accepted]
    if not accepted:
        return 0.0, float(np.mean([item.backtracking_steps for item in diagnostics]))

    mean_step = float(np.mean([item.step_size for item in accepted]))
    mean_backtracking = float(
        np.mean([item.backtracking_steps for item in diagnostics])
    )
    return mean_step, mean_backtracking


def run_diagnostic_joint(
    cfg: SystemConfig,
    start_seed: int,
    mode: str,
    candidate_fp_iterations: int,
    max_outer_iterations: int | None = None,
) -> DiagnosticRunResult:
    """用完全相同初始化分别运行 original / fp_aware。"""

    if mode not in {"original", "fp_aware"}:
        raise ValueError("mode must be 'original' or 'fp_aware'.")

    n_outer = cfg.max_outer_iter if max_outer_iterations is None else max_outer_iterations
    rng = np.random.default_rng(start_seed)

    endpoints = generate_feasible_endpoints(cfg, rng)
    geometry = build_geometry(endpoints, cfg)
    channels = build_channels(geometry, cfg)
    resources = initialize_resources(geometry, channels, cfg, rng)
    check_resource_constraints(resources, cfg)

    performance = compute_performance(channels, resources, cfg)
    initial_wsr = float(performance.weighted_sum_rate)

    joint_wsr_history = [initial_wsr]
    outer_rows: list[dict[str, float | int | str]] = []
    endpoint_rows: list[dict[str, float | int | str | bool]] = []

    rcg_state = create_outer_rcg_state(cfg)
    previous_joint_wsr = initial_wsr
    consecutive_small_changes = 0
    converged = False
    outer_iterations = 0

    t0 = perf_counter()

    for outer_iter in range(1, n_outer + 1):
        resources, inner_history = solve_inner_problem(
            channels,
            resources,
            cfg,
            verbose=False,
        )
        wsr_after_inner = float(inner_history[-1])

        if mode == "original":
            endpoint_diag: list[EndpointRCGDiagnostic] = []
            (
                geometry,
                channels,
                wsr_after_positions,
                rcg_state,
                accepted_count,
            ) = update_positions_once(
                geometry,
                resources,
                cfg,
                rcg_state=rcg_state,
                verbose=False,
                diagnostic_records=endpoint_diag,
            )

            for item in endpoint_diag:
                endpoint_rows.append(
                    {
                        "mode": mode,
                        "joint_iteration": outer_iter,
                        "endpoint_index": item.endpoint_index,
                        "group": item.group,
                        "grad_norm": item.grad_norm,
                        "beta": item.beta,
                        "accepted": item.accepted,
                        "step_size": item.step_size,
                        "backtracking_steps": item.backtracking_steps,
                        "wsr_after_update": item.wsr_after_update,
                        "fixed_wsr_at_candidate": float("nan"),
                        "aware_wsr_at_candidate": float("nan"),
                        "candidate_fp_iterations": 0,
                    }
                )

        else:
            aware_diag: list[AwareEndpointDiagnostic] = []
            (
                geometry,
                channels,
                resources,
                wsr_after_positions,
                rcg_state,
                accepted_count,
            ) = update_positions_once_fp_aware(
                geometry,
                resources,
                cfg,
                rcg_state,
                candidate_fp_iterations,
                aware_diag,
            )
            endpoint_diag = aware_diag

            for item in aware_diag:
                endpoint_rows.append(
                    {
                        "mode": mode,
                        "joint_iteration": outer_iter,
                        "endpoint_index": item.endpoint_index,
                        "group": item.group,
                        "grad_norm": item.grad_norm,
                        "beta": item.beta,
                        "accepted": item.accepted,
                        "step_size": item.step_size,
                        "backtracking_steps": item.backtracking_steps,
                        "wsr_after_update": item.wsr_after_update,
                        "fixed_wsr_at_candidate": item.fixed_wsr_at_candidate,
                        "aware_wsr_at_candidate": item.aware_wsr_at_candidate,
                        "candidate_fp_iterations": item.candidate_fp_iterations,
                    }
                )

        performance = compute_performance(channels, resources, cfg)
        current_joint_wsr = float(performance.weighted_sum_rate)

        if not np.isclose(
            current_joint_wsr,
            wsr_after_positions,
            rtol=1e-9,
            atol=1e-10,
        ):
            raise RuntimeError(
                f"{mode}: position WSR mismatch at iteration {outer_iter}."
            )

        decrease_tolerance = 1e-8 * max(1.0, abs(previous_joint_wsr))
        if current_joint_wsr < previous_joint_wsr - decrease_tolerance:
            raise RuntimeError(
                f"{mode}: joint WSR decreased at iteration {outer_iter}."
            )

        change = relative_change(current_joint_wsr, previous_joint_wsr)
        mean_step, mean_backtracking = _endpoint_summary(endpoint_diag)

        outer_rows.append(
            {
                "mode": mode,
                "joint_iteration": outer_iter,
                "wsr_after_inner": wsr_after_inner,
                "wsr_after_positions": current_joint_wsr,
                "relative_change": change,
                "accepted_endpoints": accepted_count,
                "mean_accepted_step": mean_step,
                "mean_backtracking_steps": mean_backtracking,
            }
        )
        joint_wsr_history.append(current_joint_wsr)
        outer_iterations = outer_iter

        print(
            f"[{mode:8s}] iter={outer_iter:3d} "
            f"WSR={current_joint_wsr:.10f} "
            f"accepted={accepted_count} "
            f"mean_step={mean_step:.3e} "
            f"bt={mean_backtracking:.2f}"
        )

        if change <= cfg.tol_outer:
            consecutive_small_changes += 1
        else:
            consecutive_small_changes = 0

        if consecutive_small_changes >= 2:
            converged = True
            break

        previous_joint_wsr = current_joint_wsr

    runtime_s = perf_counter() - t0

    check_resource_constraints(resources, cfg)
    geometry = build_geometry(geometry.endpoints, cfg)
    channels = build_channels(geometry, cfg)

    final_wsr = float(
        compute_performance(channels, resources, cfg).weighted_sum_rate
    )
    if not np.isclose(final_wsr, joint_wsr_history[-1], rtol=1e-9, atol=1e-10):
        raise RuntimeError(f"{mode}: final WSR mismatch.")

    return DiagnosticRunResult(
        mode=mode,
        final_geometry=geometry,
        final_channels=channels,
        final_resources=resources,
        joint_wsr_history=joint_wsr_history,
        outer_rows=outer_rows,
        endpoint_rows=endpoint_rows,
        converged=converged,
        outer_iterations=outer_iterations,
        runtime_s=runtime_s,
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_results(
    original: DiagnosticRunResult,
    aware: DiagnosticRunResult,
    candidate_fp_iterations: int,
) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    _write_csv(
        RESULT_DIR / "stage9_fp_aware_ab_outer.csv",
        original.outer_rows + aware.outer_rows,
    )
    _write_csv(
        RESULT_DIR / "stage9_fp_aware_ab_endpoints.csv",
        original.endpoint_rows + aware.endpoint_rows,
    )

    summary_rows = []
    for result in (original, aware):
        summary_rows.append(
            {
                "mode": result.mode,
                "candidate_fp_iterations": (
                    0 if result.mode == "original" else candidate_fp_iterations
                ),
                "outer_iterations": result.outer_iterations,
                "converged": result.converged,
                "final_wsr": result.final_wsr,
                "runtime_s": result.runtime_s,
                "total_accepted_endpoints": int(
                    sum(int(row["accepted_endpoints"]) for row in result.outer_rows)
                ),
            }
        )

    _write_csv(
        RESULT_DIR / "stage9_fp_aware_ab_summary.csv",
        summary_rows,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-seed", type=int, default=8001)
    parser.add_argument("--start-seed", type=int, default=2029)
    parser.add_argument("--aware-inner-iters", type=int, default=3)
    parser.add_argument("--max-outer", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _build_scene_cfg(args.scene_seed)

    print("Stage 9: Original Armijo vs FP-aware Armijo")
    print(f"  scene seed = {args.scene_seed}")
    print(f"  start seed = {args.start_seed}")
    print(f"  rho_si = {cfg.rho_si:.1e}")
    print(f"  candidate FP max iterations = {args.aware_inner_iters}")
    print("  window plateau stop = OFF")

    original = run_diagnostic_joint(
        cfg,
        start_seed=args.start_seed,
        mode="original",
        candidate_fp_iterations=args.aware_inner_iters,
        max_outer_iterations=args.max_outer,
    )

    aware = run_diagnostic_joint(
        cfg,
        start_seed=args.start_seed,
        mode="fp_aware",
        candidate_fp_iterations=args.aware_inner_iters,
        max_outer_iterations=args.max_outer,
    )

    save_results(original, aware, args.aware_inner_iters)

    print("\nA/B summary")
    print(
        f"  original : iter={original.outer_iterations}, "
        f"WSR={original.final_wsr:.10f}, "
        f"runtime={original.runtime_s:.2f}s"
    )
    print(
        f"  fp_aware : iter={aware.outer_iterations}, "
        f"WSR={aware.final_wsr:.10f}, "
        f"runtime={aware.runtime_s:.2f}s"
    )
    print(f"  CSV -> {RESULT_DIR}")


if __name__ == "__main__":
    main()
