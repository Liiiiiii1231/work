"""Stage 8A-1：诊断 Proposed 的联合外层收敛。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

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
    create_outer_rcg_state,
    update_positions_once,
)
from stage8 import load_stage8_experiment_config, make_scene_config
from utils import check_resource_constraints, relative_change


CONFIG_PATH = ROOT / "configs" / "stage8_monte_carlo.yaml"
RESULT_DIR = ROOT / "results" / "stage8" / "diagnostics"
MONTE_CARLO_CSV = ROOT / "results" / "stage8" / "stage8_monte_carlo.csv"


def _find_previous_best_seed(
    scene_seed: int,
    fallback_seed: int,
) -> int:
    """优先复用已有 Stage 8 中 Proposed 的 best seed。"""

    if not MONTE_CARLO_CSV.exists():
        return fallback_seed

    with MONTE_CARLO_CSV.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            if int(row["scene_seed"]) == scene_seed:
                return int(row["proposed_best_seed"])

    return fallback_seed


def _inner_stop_info(
    history: list[float],
    cfg: SystemConfig,
) -> tuple[int, float, str]:
    """根据内层 WSR history 判断实际停止原因。"""

    iterations = len(history) - 1

    if iterations <= 0:
        return 0, float("nan"), "no_iteration"

    final_change = relative_change(
        history[-1],
        history[-2],
    )

    if final_change <= cfg.tol_inner:
        reason = "tolerance"
    elif iterations >= cfg.max_inner_iter:
        reason = "max_iter"
    else:
        reason = "unknown"

    return iterations, float(final_change), reason


def _summarize_endpoint_records(
    records: list[EndpointRCGDiagnostic],
) -> dict[str, float | int]:
    """把一次 sweep 的端点级诊断压缩成一行。"""

    if not records:
        return {
            "accepted_endpoints": 0,
            "max_grad_norm": float("nan"),
            "mean_grad_norm": float("nan"),
            "mean_accepted_step": 0.0,
            "min_accepted_step": 0.0,
            "max_backtracking_steps": 0,
        }

    grad_norms = np.asarray(
        [record.grad_norm for record in records],
        dtype=float,
    )

    accepted_steps = np.asarray(
        [
            record.step_size
            for record in records
            if record.accepted
        ],
        dtype=float,
    )

    max_backtracking_steps = max(
        record.backtracking_steps
        for record in records
    )

    return {
        "accepted_endpoints": sum(
            int(record.accepted)
            for record in records
        ),
        "max_grad_norm": float(np.max(grad_norms)),
        "mean_grad_norm": float(np.mean(grad_norms)),
        "mean_accepted_step": (
            float(np.mean(accepted_steps))
            if accepted_steps.size > 0
            else 0.0
        ),
        "min_accepted_step": (
            float(np.min(accepted_steps))
            if accepted_steps.size > 0
            else 0.0
        ),
        "max_backtracking_steps": int(max_backtracking_steps),
    }


def run_diagnostic(
    cfg: SystemConfig,
    scene_seed: int,
    algorithm_seed: int,
    position_jitter: np.ndarray,
    max_outer_iterations: int | None = None,
    verbose: bool = True,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """运行单个 scene、单个初始化 seed 的联合外层诊断。"""

    scene_cfg = make_scene_config(
        cfg,
        scene_seed=scene_seed,
        position_jitter=position_jitter,
    )

    number_of_outer_iterations = (
        scene_cfg.max_outer_iter
        if max_outer_iterations is None
        else int(max_outer_iterations)
    )

    if number_of_outer_iterations <= 0:
        raise ValueError("max_outer_iterations must be positive.")

    rng = np.random.default_rng(algorithm_seed)

    endpoints = generate_feasible_endpoints(
        scene_cfg,
        rng,
    )

    geometry = build_geometry(
        endpoints,
        scene_cfg,
    )

    channels = build_channels(
        geometry,
        scene_cfg,
    )

    resources = initialize_resources(
        geometry,
        channels,
        scene_cfg,
        rng,
    )

    check_resource_constraints(
        resources,
        scene_cfg,
    )

    performance = compute_performance(
        channels,
        resources,
        scene_cfg,
    )

    previous_joint_wsr = float(
        performance.weighted_sum_rate
    )

    rcg_state = create_outer_rcg_state(
        scene_cfg
    )

    iteration_rows: list[dict[str, object]] = []
    endpoint_rows: list[dict[str, object]] = []

    consecutive_small_changes = 0

    if verbose:
        print(
            f"scene_seed={scene_seed}, "
            f"algorithm_seed={algorithm_seed}"
        )
        print(
            f"initial WSR={previous_joint_wsr:.10f}"
        )

    for outer_iter in range(
        1,
        number_of_outer_iterations + 1,
    ):
        wsr_before_inner = previous_joint_wsr

        resources, inner_history = solve_inner_problem(
            channels,
            resources,
            scene_cfg,
            verbose=False,
        )

        (
            inner_iterations,
            inner_final_change,
            inner_stop_reason,
        ) = _inner_stop_info(
            inner_history,
            scene_cfg,
        )

        wsr_after_inner = float(
            inner_history[-1]
        )

        endpoint_records: list[
            EndpointRCGDiagnostic
        ] = []

        (
            geometry,
            channels,
            wsr_after_outer,
            rcg_state,
            accepted_count,
        ) = update_positions_once(
            geometry,
            resources,
            scene_cfg,
            rcg_state=rcg_state,
            verbose=False,
            diagnostic_records=endpoint_records,
        )

        performance = compute_performance(
            channels,
            resources,
            scene_cfg,
        )

        current_joint_wsr = float(
            performance.weighted_sum_rate
        )

        if not np.isclose(
            current_joint_wsr,
            wsr_after_outer,
            rtol=1e-9,
            atol=1e-10,
        ):
            raise RuntimeError(
                "Outer WSR does not match recomputed true WSR."
            )

        joint_change = relative_change(
            current_joint_wsr,
            previous_joint_wsr,
        )

        inner_gain = (
            wsr_after_inner
            - wsr_before_inner
        )

        outer_gain = (
            current_joint_wsr
            - wsr_after_inner
        )

        summary = _summarize_endpoint_records(
            endpoint_records
        )

        if accepted_count != summary["accepted_endpoints"]:
            raise RuntimeError(
                "accepted_count does not match diagnostic records."
            )

        outer_small = (
            joint_change <= scene_cfg.tol_outer
        )

        if outer_small:
            consecutive_small_changes += 1
        else:
            consecutive_small_changes = 0

        row = {
            "outer_iteration": outer_iter,
            "wsr_before_inner": wsr_before_inner,
            "inner_iterations": inner_iterations,
            "inner_stop_reason": inner_stop_reason,
            "inner_final_rel_change": inner_final_change,
            "wsr_after_inner": wsr_after_inner,
            "inner_gain": inner_gain,
            "wsr_after_outer": current_joint_wsr,
            "outer_gain": outer_gain,
            "joint_rel_change": joint_change,
            "outer_tol_satisfied": outer_small,
            "consecutive_outer_small_changes": (
                consecutive_small_changes
            ),
            **summary,
        }

        iteration_rows.append(row)

        for record in endpoint_records:
            endpoint_rows.append(
                {
                    "outer_iteration": outer_iter,
                    "endpoint_index": record.endpoint_index,
                    "group": record.group,
                    "grad_norm": record.grad_norm,
                    "beta": record.beta,
                    "accepted": record.accepted,
                    "step_size": record.step_size,
                    "backtracking_steps": (
                        record.backtracking_steps
                    ),
                    "wsr_after_update": (
                        record.wsr_after_update
                    ),
                }
            )

        if verbose:
            print(
                f"Outer {outer_iter:4d} | "
                f"Inner {inner_iterations:4d} "
                f"({inner_stop_reason:9s}) | "
                f"inner_gain={inner_gain:+.3e} | "
                f"outer_gain={outer_gain:+.3e} | "
                f"joint_change={joint_change:.3e} | "
                f"grad_max={summary['max_grad_norm']:.3e} | "
                f"step_mean={summary['mean_accepted_step']:.3e} | "
                f"accepted={accepted_count}/{scene_cfg.m_uav}"
            )

        if consecutive_small_changes >= 2:
            if verbose:
                print(
                    f"Joint convergence criterion reached "
                    f"at outer iteration {outer_iter}."
                )
            break

        previous_joint_wsr = current_joint_wsr

    check_resource_constraints(
        resources,
        scene_cfg,
    )

    build_geometry(
        geometry.endpoints,
        scene_cfg,
    )

    return iteration_rows, endpoint_rows


def _write_csv(
    rows: list[dict[str, object]],
    path: Path,
) -> None:
    """保存字典记录。"""

    if not rows:
        raise ValueError("rows must not be empty.")

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scene-seed",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--algorithm-seed",
        type=int,
        default=None,
        help=(
            "ToMA initialization seed. If omitted, reuse the "
            "existing Proposed best seed for this scene when available."
        ),
    )

    parser.add_argument(
        "--max-outer",
        type=int,
        default=None,
        help="Optional short-run limit for debugging.",
    )

    args = parser.parse_args()

    cfg = load_config(CONFIG_PATH)
    experiment_cfg = load_stage8_experiment_config(
        CONFIG_PATH
    )

    scene_seed = (
        experiment_cfg.scene_seed_start
        if args.scene_seed is None
        else int(args.scene_seed)
    )

    fallback_seed = int(
        experiment_cfg.multi_start_seeds[0]
    )

    algorithm_seed = (
        _find_previous_best_seed(
            scene_seed,
            fallback_seed,
        )
        if args.algorithm_seed is None
        else int(args.algorithm_seed)
    )

    iteration_rows, endpoint_rows = run_diagnostic(
        cfg,
        scene_seed=scene_seed,
        algorithm_seed=algorithm_seed,
        position_jitter=experiment_cfg.position_jitter,
        max_outer_iterations=args.max_outer,
        verbose=True,
    )

    stem = (
        f"outer_diagnostic_scene_{scene_seed}"
        f"_seed_{algorithm_seed}"
    )

    iteration_path = RESULT_DIR / f"{stem}.csv"
    endpoint_path = RESULT_DIR / f"{stem}_endpoints.csv"

    _write_csv(
        iteration_rows,
        iteration_path,
    )

    _write_csv(
        endpoint_rows,
        endpoint_path,
    )

    final = iteration_rows[-1]

    print("\nDiagnostic finished.")
    print(f"iterations: {len(iteration_rows)}")
    print(
        f"final WSR: "
        f"{float(final['wsr_after_outer']):.10f}"
    )
    print(f"iteration CSV: {iteration_path}")
    print(f"endpoint CSV:  {endpoint_path}")


if __name__ == "__main__":
    main()
