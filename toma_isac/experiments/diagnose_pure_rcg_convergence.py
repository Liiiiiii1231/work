"""Stage 8A-2：固定资源后，单独诊断纯 RCG 外层收敛。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from state import ResourceState
from utils import check_resource_constraints, relative_change


CONFIG_PATH = ROOT / "configs" / "stage8_monte_carlo.yaml"
RESULT_DIR = ROOT / "results" / "stage8" / "diagnostics"
FIGURE_DIR = RESULT_DIR / "figures"
MONTE_CARLO_CSV = ROOT / "results" / "stage8" / "stage8_monte_carlo.csv"


def _find_previous_best_seed(scene_seed: int, fallback_seed: int) -> int:
    """优先复用 Stage 8 已保存的 Proposed best seed。"""

    if not MONTE_CARLO_CSV.exists():
        return fallback_seed

    with MONTE_CARLO_CSV.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if int(row["scene_seed"]) == scene_seed:
                return int(row["proposed_best_seed"])

    return fallback_seed


def _summarize_records(
    records: list[EndpointRCGDiagnostic],
) -> dict[str, float | int]:
    """汇总一次纯 RCG sweep 的端点级诊断。"""

    grad_norms = np.asarray([record.grad_norm for record in records], dtype=float)
    accepted_steps = np.asarray(
        [record.step_size for record in records if record.accepted],
        dtype=float,
    )

    if grad_norms.size == 0:
        raise RuntimeError("No endpoint diagnostic records were produced.")

    return {
        "accepted_endpoints": sum(int(record.accepted) for record in records),
        "max_grad_norm": float(np.max(grad_norms)),
        "mean_grad_norm": float(np.mean(grad_norms)),
        "mean_accepted_step": (
            float(np.mean(accepted_steps)) if accepted_steps.size else 0.0
        ),
        "min_accepted_step": (
            float(np.min(accepted_steps)) if accepted_steps.size else 0.0
        ),
        "max_backtracking_steps": max(
            int(record.backtracking_steps) for record in records
        ),
    }


def _copy_resources(resources: ResourceState) -> dict[str, np.ndarray]:
    """保存资源快照，用于确认纯 RCG 阶段资源没有被修改。"""

    return {
        "q_matrix": resources.q_matrix.copy(),
        "q_ul": resources.q_ul.copy(),
        "b_ul": resources.b_ul.copy(),
        "u_s": resources.u_s.copy(),
    }


def _assert_resources_frozen(
    resources: ResourceState,
    snapshot: dict[str, np.ndarray],
) -> None:
    """确认固定资源实验中 FP 变量保持不变。"""

    for name, expected in snapshot.items():
        if not np.array_equal(getattr(resources, name), expected):
            raise RuntimeError(f"Resource {name} changed during pure RCG test.")


def run_pure_rcg_diagnostic(
    cfg: SystemConfig,
    scene_seed: int,
    algorithm_seed: int,
    position_jitter: np.ndarray,
    max_sweeps: int = 50,
    stop_on_wsr_tol: bool = False,
    verbose: bool = True,
) -> tuple[
    float,
    int,
    float,
    list[dict[str, object]],
    list[dict[str, object]],
    int | None,
]:
    """先运行一次完整 FP，然后固定 resources，只重复 RCG sweep。"""

    if max_sweeps <= 0:
        raise ValueError("max_sweeps must be positive.")

    scene_cfg = make_scene_config(
        cfg,
        scene_seed=scene_seed,
        position_jitter=position_jitter,
    )

    rng = np.random.default_rng(algorithm_seed)

    endpoints = generate_feasible_endpoints(scene_cfg, rng)
    geometry = build_geometry(endpoints, scene_cfg)
    channels = build_channels(geometry, scene_cfg)
    resources = initialize_resources(geometry, channels, scene_cfg, rng)

    check_resource_constraints(resources, scene_cfg)

    initial_wsr = float(
        compute_performance(channels, resources, scene_cfg).weighted_sum_rate
    )

    # 只在这里运行一次 FP；之后 resources 完全冻结。
    resources, inner_history = solve_inner_problem(
        channels,
        resources,
        scene_cfg,
        verbose=False,
    )

    inner_iterations = len(inner_history) - 1
    fixed_resource_initial_wsr = float(inner_history[-1])
    resource_snapshot = _copy_resources(resources)

    rcg_state = create_outer_rcg_state(scene_cfg)
    previous_wsr = fixed_resource_initial_wsr
    consecutive_small_changes = 0
    first_wsr_convergence_sweep: int | None = None

    sweep_rows: list[dict[str, object]] = []
    endpoint_rows: list[dict[str, object]] = []

    if verbose:
        print(
            f"scene_seed={scene_seed}, algorithm_seed={algorithm_seed}\n"
            f"initial WSR before FP={initial_wsr:.10f}\n"
            f"one-time FP iterations={inner_iterations}\n"
            f"fixed-resource initial WSR={fixed_resource_initial_wsr:.10f}"
        )

    for sweep in range(1, max_sweeps + 1):
        _assert_resources_frozen(resources, resource_snapshot)

        endpoint_records: list[EndpointRCGDiagnostic] = []

        (
            geometry,
            channels,
            outer_wsr,
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

        current_wsr = float(
            compute_performance(channels, resources, scene_cfg).weighted_sum_rate
        )

        if not np.isclose(
            current_wsr,
            outer_wsr,
            rtol=1e-9,
            atol=1e-10,
        ):
            raise RuntimeError("Outer WSR does not match recomputed true WSR.")

        # Armijo 是上升线搜索；固定资源时 WSR 不应明显下降。
        if current_wsr < previous_wsr - 1e-9:
            raise RuntimeError(
                "Pure RCG WSR decreased under fixed resources: "
                f"{previous_wsr:.12f} -> {current_wsr:.12f}."
            )

        change = float(relative_change(current_wsr, previous_wsr))
        gain = current_wsr - previous_wsr
        summary = _summarize_records(endpoint_records)

        if accepted_count != summary["accepted_endpoints"]:
            raise RuntimeError(
                "accepted_count does not match endpoint diagnostic records."
            )

        wsr_tol_satisfied = change <= scene_cfg.tol_outer
        if wsr_tol_satisfied:
            consecutive_small_changes += 1
        else:
            consecutive_small_changes = 0

        if (
            first_wsr_convergence_sweep is None
            and consecutive_small_changes >= 2
        ):
            first_wsr_convergence_sweep = sweep
            if verbose:
                print(
                    f"WSR-only convergence marker first reached at sweep {sweep}."
                )

        sweep_rows.append(
            {
                "rcg_sweep": sweep,
                "wsr_before": previous_wsr,
                "wsr_after": current_wsr,
                "wsr_gain": gain,
                "relative_change": change,
                "outer_tol_satisfied": wsr_tol_satisfied,
                "consecutive_small_changes": consecutive_small_changes,
                **summary,
            }
        )

        for record in endpoint_records:
            endpoint_rows.append(
                {
                    "rcg_sweep": sweep,
                    "endpoint_index": record.endpoint_index,
                    "group": record.group,
                    "grad_norm": record.grad_norm,
                    "beta": record.beta,
                    "accepted": record.accepted,
                    "step_size": record.step_size,
                    "backtracking_steps": record.backtracking_steps,
                    "wsr_after_update": record.wsr_after_update,
                }
            )

        if verbose:
            print(
                f"RCG {sweep:4d} | "
                f"WSR={current_wsr:.10f} | "
                f"gain={gain:+.3e} | "
                f"change={change:.3e} | "
                f"grad_max={summary['max_grad_norm']:.3e} | "
                f"step_mean={summary['mean_accepted_step']:.3e} | "
                f"backtrack_max={summary['max_backtracking_steps']:2d} | "
                f"accepted={accepted_count}/{scene_cfg.m_uav}"
            )

        previous_wsr = current_wsr

        if stop_on_wsr_tol and consecutive_small_changes >= 2:
            break

    _assert_resources_frozen(resources, resource_snapshot)
    check_resource_constraints(resources, scene_cfg)
    build_geometry(geometry.endpoints, scene_cfg)

    return (
        initial_wsr,
        inner_iterations,
        fixed_resource_initial_wsr,
        sweep_rows,
        endpoint_rows,
        first_wsr_convergence_sweep,
    )


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    """保存诊断 CSV。"""

    if not rows:
        raise ValueError("rows must not be empty.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_figures(
    fixed_resource_initial_wsr: float,
    sweep_rows: list[dict[str, object]],
    stem: str,
) -> tuple[Path, Path]:
    """保存纯 RCG 的 WSR 与梯度诊断图。"""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    sweeps = np.asarray([int(row["rcg_sweep"]) for row in sweep_rows])
    wsr = np.asarray([float(row["wsr_after"]) for row in sweep_rows])
    grad = np.asarray([float(row["max_grad_norm"]) for row in sweep_rows])

    wsr_path = FIGURE_DIR / f"{stem}_wsr.png"
    grad_path = FIGURE_DIR / f"{stem}_grad.png"

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(
        np.concatenate(([0], sweeps)),
        np.concatenate(([fixed_resource_initial_wsr], wsr)),
    )
    ax.set_xlabel("Pure RCG sweep")
    ax.set_ylabel("Fixed-resource WSR")
    ax.set_title("Pure RCG convergence with resources frozen")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(wsr_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    positive = np.maximum(grad, np.finfo(float).tiny)
    ax.semilogy(sweeps, positive)
    ax.set_xlabel("Pure RCG sweep")
    ax.set_ylabel("Max Riemannian gradient norm")
    ax.set_title("Pure RCG gradient diagnostic")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(grad_path, dpi=180)
    plt.close(fig)

    return wsr_path, grad_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-seed", type=int, default=None)
    parser.add_argument("--algorithm-seed", type=int, default=None)
    parser.add_argument(
        "--max-sweeps",
        type=int,
        default=50,
        help="Pure RCG diagnostic sweep budget; default 50.",
    )
    parser.add_argument(
        "--stop-on-wsr-tol",
        action="store_true",
        help=(
            "Stop after two consecutive tol_outer hits. By default the "
            "diagnostic continues to max-sweeps so the gradient trend can be seen."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(CONFIG_PATH)
    experiment_cfg = load_stage8_experiment_config(CONFIG_PATH)

    scene_seed = (
        experiment_cfg.scene_seed_start
        if args.scene_seed is None
        else int(args.scene_seed)
    )

    fallback_seed = int(experiment_cfg.multi_start_seeds[0])
    algorithm_seed = (
        _find_previous_best_seed(scene_seed, fallback_seed)
        if args.algorithm_seed is None
        else int(args.algorithm_seed)
    )

    (
        initial_wsr,
        inner_iterations,
        fixed_resource_initial_wsr,
        sweep_rows,
        endpoint_rows,
        first_wsr_convergence_sweep,
    ) = run_pure_rcg_diagnostic(
        cfg,
        scene_seed=scene_seed,
        algorithm_seed=algorithm_seed,
        position_jitter=experiment_cfg.position_jitter,
        max_sweeps=args.max_sweeps,
        stop_on_wsr_tol=args.stop_on_wsr_tol,
        verbose=True,
    )

    stem = f"pure_rcg_scene_{scene_seed}_seed_{algorithm_seed}"
    sweep_path = RESULT_DIR / f"{stem}.csv"
    endpoint_path = RESULT_DIR / f"{stem}_endpoints.csv"

    _write_csv(sweep_rows, sweep_path)
    _write_csv(endpoint_rows, endpoint_path)
    wsr_figure, grad_figure = _save_figures(
        fixed_resource_initial_wsr,
        sweep_rows,
        stem,
    )

    final = sweep_rows[-1]

    print("\nPure RCG diagnostic finished.")
    print(f"initial WSR before FP: {initial_wsr:.10f}")
    print(f"one-time FP iterations: {inner_iterations}")
    print(f"fixed-resource initial WSR: {fixed_resource_initial_wsr:.10f}")
    print(f"pure RCG sweeps executed: {len(sweep_rows)}")
    print(f"final fixed-resource WSR: {float(final['wsr_after']):.10f}")
    print(f"final max grad norm: {float(final['max_grad_norm']):.6e}")
    print(f"final mean step: {float(final['mean_accepted_step']):.6e}")
    print(f"first WSR-only convergence sweep: {first_wsr_convergence_sweep}")
    print(f"sweep CSV:    {sweep_path}")
    print(f"endpoint CSV: {endpoint_path}")
    print(f"WSR figure:   {wsr_figure}")
    print(f"grad figure:  {grad_figure}")


if __name__ == "__main__":
    main()
