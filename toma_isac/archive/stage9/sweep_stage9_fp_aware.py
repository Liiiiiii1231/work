"""Stage 9D：比较 FP-aware Armijo 的 1/2/3/5-step warm-start FP。"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import diagnose_stage9_fp_aware as fpdiag


RESULT_DIR = ROOT / "results" / "stage9"
DEFAULT_STEPS = (1, 2, 3, 5)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean_accepted_step(endpoint_rows: list[dict[str, Any]]) -> float:
    values = [
        float(row["step_size"])
        for row in endpoint_rows
        if bool(row["accepted"])
    ]
    return float(np.mean(values)) if values else 0.0


def _mean_backtracking(endpoint_rows: list[dict[str, Any]]) -> float:
    if not endpoint_rows:
        return 0.0
    return float(
        np.mean([float(row["backtracking_steps"]) for row in endpoint_rows])
    )


def _run_counted(
    cfg,
    start_seed: int,
    mode: str,
    candidate_fp_max_iterations: int,
    max_outer_iterations: int | None,
    verbose_runs: bool,
):
    """运行一次诊断，并准确统计主 FP 与 Armijo 候选 FP 的总工作量。"""

    original_solver = fpdiag.solve_inner_problem
    counts = {
        "main_fp_calls": 0,
        "main_fp_iterations": 0,
        "candidate_fp_calls": 0,
        "candidate_fp_iterations_total": 0,
    }

    def counting_solver(channels, resources, inner_cfg, verbose=False):
        updated, history = original_solver(
            channels,
            resources,
            inner_cfg,
            verbose=verbose,
        )
        used_iterations = max(0, len(history) - 1)

        is_candidate = (
            mode == "fp_aware"
            and inner_cfg.max_inner_iter == candidate_fp_max_iterations
            and inner_cfg.max_inner_iter != cfg.max_inner_iter
        )

        if is_candidate:
            counts["candidate_fp_calls"] += 1
            counts["candidate_fp_iterations_total"] += used_iterations
        else:
            counts["main_fp_calls"] += 1
            counts["main_fp_iterations"] += used_iterations

        return updated, history

    fpdiag.solve_inner_problem = counting_solver
    try:
        if verbose_runs:
            result = fpdiag.run_diagnostic_joint(
                cfg,
                start_seed=start_seed,
                mode=mode,
                candidate_fp_iterations=candidate_fp_max_iterations,
                max_outer_iterations=max_outer_iterations,
            )
        else:
            with redirect_stdout(io.StringIO()):
                result = fpdiag.run_diagnostic_joint(
                    cfg,
                    start_seed=start_seed,
                    mode=mode,
                    candidate_fp_iterations=candidate_fp_max_iterations,
                    max_outer_iterations=max_outer_iterations,
                )
    finally:
        fpdiag.solve_inner_problem = original_solver

    return result, counts


def _relabel_rows(
    rows: list[dict[str, Any]],
    label: str,
    fp_cap: int,
) -> list[dict[str, Any]]:
    relabeled = []
    for row in rows:
        item = dict(row)
        item["mode"] = label
        item["candidate_fp_max_iterations"] = fp_cap
        relabeled.append(item)
    return relabeled


def _summary_row(
    result,
    counts: dict[str, int],
    label: str,
    fp_cap: int,
) -> dict[str, Any]:
    candidate_calls = counts["candidate_fp_calls"]
    candidate_total = counts["candidate_fp_iterations_total"]

    return {
        "mode": label,
        "candidate_fp_max_iterations": fp_cap,
        "outer_iterations": result.outer_iterations,
        "converged": result.converged,
        "final_wsr": result.final_wsr,
        "runtime_s": result.runtime_s,
        "total_accepted_endpoints": int(
            sum(int(row["accepted_endpoints"]) for row in result.outer_rows)
        ),
        "mean_accepted_step": _mean_accepted_step(result.endpoint_rows),
        "mean_backtracking_steps": _mean_backtracking(result.endpoint_rows),
        "main_fp_calls": counts["main_fp_calls"],
        "main_fp_iterations": counts["main_fp_iterations"],
        "candidate_fp_calls": candidate_calls,
        "candidate_fp_iterations_total": candidate_total,
        "candidate_fp_iterations_per_call": (
            candidate_total / candidate_calls if candidate_calls else 0.0
        ),
        "total_fp_iterations": (
            counts["main_fp_iterations"] + candidate_total
        ),
    }


def _add_relative_metrics(
    summary_rows: list[dict[str, Any]],
) -> None:
    original = next(row for row in summary_rows if row["mode"] == "original")
    original_iters = float(original["outer_iterations"])
    original_wsr = float(original["final_wsr"])
    original_runtime = float(original["runtime_s"])

    for row in summary_rows:
        iters = float(row["outer_iterations"])
        wsr = float(row["final_wsr"])
        runtime = float(row["runtime_s"])

        row["outer_iteration_reduction_pct"] = (
            100.0 * (original_iters - iters) / original_iters
            if original_iters > 0.0
            else 0.0
        )
        row["wsr_gain_vs_original_pct"] = (
            100.0 * (wsr - original_wsr) / abs(original_wsr)
            if abs(original_wsr) > 0.0
            else 0.0
        )
        row["runtime_ratio_vs_original"] = (
            runtime / original_runtime if original_runtime > 0.0 else float("nan")
        )


def _print_summary(summary_rows: list[dict[str, Any]]) -> None:
    print("\nStage 9D summary")
    print(
        "  mode        cap  outer  final_WSR    runtime  "
        "mean_step   mean_bt  cand_FP_iters"
    )
    for row in summary_rows:
        print(
            f"  {str(row['mode']):11s} "
            f"{int(row['candidate_fp_max_iterations']):3d} "
            f"{int(row['outer_iterations']):6d} "
            f"{float(row['final_wsr']):11.8f} "
            f"{float(row['runtime_s']):8.2f}s "
            f"{float(row['mean_accepted_step']):10.3e} "
            f"{float(row['mean_backtracking_steps']):8.2f} "
            f"{int(row['candidate_fp_iterations_total']):13d}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-seed", type=int, default=8001)
    parser.add_argument("--start-seed", type=int, default=2029)
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=list(DEFAULT_STEPS),
        help="Candidate FP maximum iterations to compare (default: 1 2 3 5).",
    )
    parser.add_argument("--max-outer", type=int, default=None)
    parser.add_argument(
        "--verbose-runs",
        action="store_true",
        help="Show every Joint AO iteration instead of only one summary per variant.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    steps = []
    for value in args.steps:
        if value <= 0:
            raise ValueError("All --steps values must be positive integers.")
        if value not in steps:
            steps.append(value)

    cfg = fpdiag._build_scene_cfg(args.scene_seed)

    print("Stage 9D: FP-aware candidate-FP sweep")
    print(f"  scene seed = {args.scene_seed}")
    print(f"  start seed = {args.start_seed}")
    print(f"  rho_si = {cfg.rho_si:.1e}")
    print(f"  candidate FP caps = {steps}")
    print("  window plateau stop = OFF")

    summary_rows: list[dict[str, Any]] = []
    outer_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []

    # Original 只跑一次，避免为每个 FP cap 重复计算同一个对照组。
    print("\nRunning original ...")
    original, counts = _run_counted(
        cfg,
        start_seed=args.start_seed,
        mode="original",
        candidate_fp_max_iterations=steps[0],
        max_outer_iterations=args.max_outer,
        verbose_runs=args.verbose_runs,
    )
    summary_rows.append(_summary_row(original, counts, "original", 0))
    outer_rows.extend(_relabel_rows(original.outer_rows, "original", 0))
    endpoint_rows.extend(_relabel_rows(original.endpoint_rows, "original", 0))
    print(
        f"  done: outer={original.outer_iterations}, "
        f"WSR={original.final_wsr:.10f}, runtime={original.runtime_s:.2f}s"
    )

    for fp_cap in steps:
        label = f"fp_aware_{fp_cap}"
        print(f"\nRunning {label} ...")
        result, counts = _run_counted(
            cfg,
            start_seed=args.start_seed,
            mode="fp_aware",
            candidate_fp_max_iterations=fp_cap,
            max_outer_iterations=args.max_outer,
            verbose_runs=args.verbose_runs,
        )
        summary_rows.append(_summary_row(result, counts, label, fp_cap))
        outer_rows.extend(_relabel_rows(result.outer_rows, label, fp_cap))
        endpoint_rows.extend(_relabel_rows(result.endpoint_rows, label, fp_cap))
        print(
            f"  done: outer={result.outer_iterations}, "
            f"WSR={result.final_wsr:.10f}, runtime={result.runtime_s:.2f}s, "
            f"candidate_FP_iters={counts['candidate_fp_iterations_total']}"
        )

    _add_relative_metrics(summary_rows)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(
        RESULT_DIR / "stage9_fp_aware_sweep_summary.csv",
        summary_rows,
    )
    _write_csv(
        RESULT_DIR / "stage9_fp_aware_sweep_outer.csv",
        outer_rows,
    )
    _write_csv(
        RESULT_DIR / "stage9_fp_aware_sweep_endpoints.csv",
        endpoint_rows,
    )

    _print_summary(summary_rows)
    print(f"\nCSV -> {RESULT_DIR}")


if __name__ == "__main__":
    main()
