"""Stage 9E：Original / FP-aware-2 / FP-aware-3 多场景稳定性验证。"""

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
DEFAULT_SCENE_SEEDS = (8001, 8002, 8003, 8004, 8005)
DEFAULT_FP_CAPS = (2, 3)


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
    """运行一次算法，并统计主 FP 与候选 Armijo FP 的实际迭代量。"""

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
        kwargs = dict(
            cfg=cfg,
            start_seed=start_seed,
            mode=mode,
            candidate_fp_iterations=candidate_fp_max_iterations,
            max_outer_iterations=max_outer_iterations,
        )
        if verbose_runs:
            result = fpdiag.run_diagnostic_joint(**kwargs)
        else:
            with redirect_stdout(io.StringIO()):
                result = fpdiag.run_diagnostic_joint(**kwargs)
    finally:
        fpdiag.solve_inner_problem = original_solver

    return result, counts


def _scene_summary_row(
    scene_seed: int,
    start_seed: int,
    result,
    counts: dict[str, int],
    label: str,
    fp_cap: int,
) -> dict[str, Any]:
    candidate_calls = counts["candidate_fp_calls"]
    candidate_total = counts["candidate_fp_iterations_total"]

    return {
        "scene_seed": scene_seed,
        "start_seed": start_seed,
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
        "total_fp_iterations": counts["main_fp_iterations"] + candidate_total,
    }


def _add_scene_relative_metrics(rows: list[dict[str, Any]]) -> None:
    """逐 scene 以 original 为基准计算相对改善。"""

    by_scene: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_scene.setdefault(int(row["scene_seed"]), []).append(row)

    for scene_rows in by_scene.values():
        original = next(row for row in scene_rows if row["mode"] == "original")
        original_iters = float(original["outer_iterations"])
        original_wsr = float(original["final_wsr"])
        original_runtime = float(original["runtime_s"])

        for row in scene_rows:
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
                runtime / original_runtime
                if original_runtime > 0.0
                else float("nan")
            )


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def _aggregate_rows(scene_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    modes: list[str] = []
    for row in scene_rows:
        label = str(row["mode"])
        if label not in modes:
            modes.append(label)

    aggregate: list[dict[str, Any]] = []
    for label in modes:
        group = [row for row in scene_rows if row["mode"] == label]

        def vals(key: str) -> list[float]:
            return [float(row[key]) for row in group]

        aggregate.append(
            {
                "mode": label,
                "candidate_fp_max_iterations": int(
                    group[0]["candidate_fp_max_iterations"]
                ),
                "n_scenes": len(group),
                "converged_scenes": int(sum(bool(row["converged"]) for row in group)),
                "outer_iterations_mean": _mean(vals("outer_iterations")),
                "outer_iterations_std": _std(vals("outer_iterations")),
                "final_wsr_mean": _mean(vals("final_wsr")),
                "final_wsr_std": _std(vals("final_wsr")),
                "runtime_s_mean": _mean(vals("runtime_s")),
                "runtime_s_std": _std(vals("runtime_s")),
                "mean_accepted_step_mean": _mean(vals("mean_accepted_step")),
                "mean_backtracking_steps_mean": _mean(
                    vals("mean_backtracking_steps")
                ),
                "candidate_fp_iterations_total_mean": _mean(
                    vals("candidate_fp_iterations_total")
                ),
                "total_fp_iterations_mean": _mean(vals("total_fp_iterations")),
                "outer_iteration_reduction_pct_mean": _mean(
                    vals("outer_iteration_reduction_pct")
                ),
                "wsr_gain_vs_original_pct_mean": _mean(
                    vals("wsr_gain_vs_original_pct")
                ),
                "runtime_ratio_vs_original_mean": _mean(
                    vals("runtime_ratio_vs_original")
                ),
            }
        )

    return aggregate


def _relabel_detail_rows(
    rows: list[dict[str, Any]],
    scene_seed: int,
    start_seed: int,
    label: str,
    fp_cap: int,
) -> list[dict[str, Any]]:
    relabeled = []
    for row in rows:
        item = dict(row)
        item["scene_seed"] = scene_seed
        item["start_seed"] = start_seed
        item["mode"] = label
        item["candidate_fp_max_iterations"] = fp_cap
        relabeled.append(item)
    return relabeled


def _print_scene_summary(scene_rows: list[dict[str, Any]], scene_seed: int) -> None:
    rows = [row for row in scene_rows if int(row["scene_seed"]) == scene_seed]
    print(f"\nScene {scene_seed}")
    print("  mode        outer  final_WSR    runtime  mean_step   mean_bt  total_FP")
    for row in rows:
        print(
            f"  {str(row['mode']):11s} "
            f"{int(row['outer_iterations']):6d} "
            f"{float(row['final_wsr']):11.8f} "
            f"{float(row['runtime_s']):8.2f}s "
            f"{float(row['mean_accepted_step']):10.3e} "
            f"{float(row['mean_backtracking_steps']):8.2f} "
            f"{int(row['total_fp_iterations']):8d}"
        )


def _print_aggregate(rows: list[dict[str, Any]]) -> None:
    print("\nStage 9E aggregate summary")
    print(
        "  mode        AO(mean±std)        WSR(mean±std)       "
        "runtime(mean±std)   AO_reduction   WSR_gain"
    )
    for row in rows:
        print(
            f"  {str(row['mode']):11s} "
            f"{float(row['outer_iterations_mean']):7.2f}±"
            f"{float(row['outer_iterations_std']):6.2f}   "
            f"{float(row['final_wsr_mean']):8.5f}±"
            f"{float(row['final_wsr_std']):7.5f}   "
            f"{float(row['runtime_s_mean']):6.2f}±"
            f"{float(row['runtime_s_std']):5.2f}s   "
            f"{float(row['outer_iteration_reduction_pct_mean']):8.2f}%   "
            f"{float(row['wsr_gain_vs_original_pct_mean']):7.2f}%"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SCENE_SEEDS),
        help="Scene seeds (default: 8001 8002 8003 8004 8005).",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=2029,
        help="Use the same initial ToMA seed for all methods/scenes (default: 2029).",
    )
    parser.add_argument(
        "--fp-caps",
        type=int,
        nargs="+",
        default=list(DEFAULT_FP_CAPS),
        help="FP-aware candidate FP caps (default: 2 3).",
    )
    parser.add_argument("--max-outer", type=int, default=None)
    parser.add_argument(
        "--verbose-runs",
        action="store_true",
        help="Show every Joint AO iteration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    scene_seeds: list[int] = []
    for value in args.scene_seeds:
        if value not in scene_seeds:
            scene_seeds.append(value)

    fp_caps: list[int] = []
    for value in args.fp_caps:
        if value <= 0:
            raise ValueError("All --fp-caps values must be positive integers.")
        if value not in fp_caps:
            fp_caps.append(value)

    if not scene_seeds:
        raise ValueError("At least one scene seed is required.")

    print("Stage 9E: Multi-scene FP-aware stability test")
    print(f"  scenes = {scene_seeds}")
    print(f"  common start seed = {args.start_seed}")
    print(f"  FP-aware caps = {fp_caps}")
    print("  fading = OFF (pure LoS Stage 9 diagnostic)")
    print("  window plateau stop = OFF")

    per_scene_rows: list[dict[str, Any]] = []
    outer_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []

    for scene_seed in scene_seeds:
        cfg = fpdiag._build_scene_cfg(scene_seed)

        print(f"\n=== Scene {scene_seed} ===")

        print("Running original ...")
        original, counts = _run_counted(
            cfg,
            start_seed=args.start_seed,
            mode="original",
            candidate_fp_max_iterations=fp_caps[0],
            max_outer_iterations=args.max_outer,
            verbose_runs=args.verbose_runs,
        )
        per_scene_rows.append(
            _scene_summary_row(
                scene_seed,
                args.start_seed,
                original,
                counts,
                "original",
                0,
            )
        )
        outer_rows.extend(
            _relabel_detail_rows(
                original.outer_rows,
                scene_seed,
                args.start_seed,
                "original",
                0,
            )
        )
        endpoint_rows.extend(
            _relabel_detail_rows(
                original.endpoint_rows,
                scene_seed,
                args.start_seed,
                "original",
                0,
            )
        )
        print(
            f"  done: outer={original.outer_iterations}, "
            f"WSR={original.final_wsr:.10f}, runtime={original.runtime_s:.2f}s"
        )

        for fp_cap in fp_caps:
            label = f"fp_aware_{fp_cap}"
            print(f"Running {label} ...")
            result, counts = _run_counted(
                cfg,
                start_seed=args.start_seed,
                mode="fp_aware",
                candidate_fp_max_iterations=fp_cap,
                max_outer_iterations=args.max_outer,
                verbose_runs=args.verbose_runs,
            )
            per_scene_rows.append(
                _scene_summary_row(
                    scene_seed,
                    args.start_seed,
                    result,
                    counts,
                    label,
                    fp_cap,
                )
            )
            outer_rows.extend(
                _relabel_detail_rows(
                    result.outer_rows,
                    scene_seed,
                    args.start_seed,
                    label,
                    fp_cap,
                )
            )
            endpoint_rows.extend(
                _relabel_detail_rows(
                    result.endpoint_rows,
                    scene_seed,
                    args.start_seed,
                    label,
                    fp_cap,
                )
            )
            print(
                f"  done: outer={result.outer_iterations}, "
                f"WSR={result.final_wsr:.10f}, runtime={result.runtime_s:.2f}s, "
                f"candidate_FP_iters={counts['candidate_fp_iterations_total']}"
            )

        _add_scene_relative_metrics(per_scene_rows)
        _print_scene_summary(per_scene_rows, scene_seed)

    _add_scene_relative_metrics(per_scene_rows)
    aggregate_rows = _aggregate_rows(per_scene_rows)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(
        RESULT_DIR / "stage9_fp_aware_multiscene_per_scene.csv",
        per_scene_rows,
    )
    _write_csv(
        RESULT_DIR / "stage9_fp_aware_multiscene_aggregate.csv",
        aggregate_rows,
    )
    _write_csv(
        RESULT_DIR / "stage9_fp_aware_multiscene_outer.csv",
        outer_rows,
    )
    _write_csv(
        RESULT_DIR / "stage9_fp_aware_multiscene_endpoints.csv",
        endpoint_rows,
    )

    _print_aggregate(aggregate_rows)
    print(f"\nCSV -> {RESULT_DIR}")


if __name__ == "__main__":
    main()
