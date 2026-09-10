"""Scene 8001：B6 MA 跨 rho_si 最终解复用诊断。

目的：
1. 在五个 rho_si 点分别完整运行正式 B6 3-start，保存各点最佳 MA 解；
2. 将每个最佳解的 geometry/resources 固定，交叉放到五个 rho_si 下重新评价；
3. 验证固定解 WSR 随 rho_si 增大是否单调不增；
4. 对每个 rho_si 取五个已有解中的最佳复用值，观察原 3-start 非单调是否来自局部最优。

本脚本只做诊断，不修改任何正式算法或 baseline 定义。
"""

from __future__ import annotations

import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from channels import build_channels
from config import load_config
from ma_baseline import run_ma_fp
from metrics import compute_performance
from multistart import run_multi_start
from stage8 import (
    load_stage8_experiment_config,
    make_ma_start_seeds,
    make_scene_config,
)
from utils import check_resource_constraints


CONFIG_PATH = ROOT / "configs" / "stage8_monte_carlo.yaml"
OUTPUT_DIR = ROOT / "results" / "stage8" / "diagnostics"

SCENE_SEED = 8001
RHO_DB_VALUES = (-50, -40, -30, -20, -10)
MONOTONIC_TOL = 1e-10
DIAGONAL_TOL = 1e-9


def _rho_from_db(rho_db: int) -> float:
    """将功率比 dB 转成线性 rho_si。"""

    return 10.0 ** (float(rho_db) / 10.0)


def _run_best_ma_at_rho(scene_cfg, exp_cfg, ma_seeds, rho_db: int):
    """在指定 rho_si 下按正式 3-start 规则运行 B6，并返回最佳结果。"""

    rho_si = _rho_from_db(rho_db)
    cfg = replace(scene_cfg, rho_si=rho_si)
    cfg.validate()

    print(f"\n===== Optimize B6 MA at {rho_db} dB (rho={rho_si:.1e}) =====")

    multi = run_multi_start(
        name=f"B6 MA + FP @ {rho_db} dB",
        seeds=ma_seeds,
        run_single=lambda seed: run_ma_fp(
            cfg,
            rng=np.random.default_rng(seed),
            verbose=False,
            region_size=exp_cfg.ma_region_size,
            min_spacing=exp_cfg.ma_min_spacing,
            initial_step=exp_cfg.ma_initial_step,
            randomize_initial_geometry=(
                int(seed) != int(exp_cfg.fixed_upa_seed)
            ),
            initial_jitter_fraction=exp_cfg.ma_initial_jitter_fraction,
        ),
        iteration_getter=lambda result: result.iterations,
        cfg=cfg,
        require_convergence=True,
        verbose=True,
    )

    best = multi.best_result
    print(
        f"BEST @ {rho_db:>4} dB | "
        f"WSR={best.performance.weighted_sum_rate:.10f} | "
        f"seed={multi.best_seed} | iter={multi.best_start.iterations}"
    )

    return cfg, multi


def _evaluate_fixed_solution(result, eval_cfg):
    """固定 MA geometry/resources，只改变 rho_si 后重新计算真实性能。"""

    check_resource_constraints(result.resources, eval_cfg)

    channels = build_channels(
        result.geometry,
        eval_cfg,
    )

    return compute_performance(
        channels,
        result.resources,
        eval_cfg,
    )


def _is_nonincreasing(values: np.ndarray, tol: float) -> bool:
    """允许数值容差的单调不增检查。"""

    return bool(np.all(np.diff(values) <= tol))


def main() -> None:
    base_cfg = load_config(CONFIG_PATH)
    exp_cfg = load_stage8_experiment_config(CONFIG_PATH)

    # 五个 rho 点共享同一个 scene realization。
    scene_cfg = make_scene_config(
        base_cfg,
        scene_seed=SCENE_SEED,
        position_jitter=exp_cfg.position_jitter,
        randomize_fading=exp_cfg.fading_enabled,
        fading_model=exp_cfg.fading_model,
    )

    ma_seeds = make_ma_start_seeds(
        scene_seed=SCENE_SEED,
        fixed_upa_seed=exp_cfg.fixed_upa_seed,
        n_starts=exp_cfg.ma_n_starts,
    )

    print(f"scene={SCENE_SEED}")
    print(f"MA seeds={ma_seeds}")
    print(f"rho dB={RHO_DB_VALUES}")

    # ------------------------------------------------------------
    # Step 1：五个 rho 点分别独立优化，保存各自最佳 MA 解。
    # ------------------------------------------------------------
    optimized_cfgs = {}
    optimized_runs = {}

    for rho_db in RHO_DB_VALUES:
        cfg, multi = _run_best_ma_at_rho(
            scene_cfg,
            exp_cfg,
            ma_seeds,
            rho_db,
        )
        optimized_cfgs[rho_db] = cfg
        optimized_runs[rho_db] = multi

    # ------------------------------------------------------------
    # Step 2：5 x 5 跨 rho 固定解复用评价。
    # 行：解在哪个 rho 下优化得到；列：拿到哪个 rho 下评价。
    # ------------------------------------------------------------
    n_rho = len(RHO_DB_VALUES)
    wsr_matrix = np.empty((n_rho, n_rho), dtype=float)
    dl_matrix = np.empty_like(wsr_matrix)
    ul_matrix = np.empty_like(wsr_matrix)
    sensing_matrix = np.empty_like(wsr_matrix)

    long_rows: list[dict[str, float | int]] = []

    for source_index, source_db in enumerate(RHO_DB_VALUES):
        result = optimized_runs[source_db].best_result

        for eval_index, eval_db in enumerate(RHO_DB_VALUES):
            eval_cfg = optimized_cfgs[eval_db]
            perf = _evaluate_fixed_solution(result, eval_cfg)

            wsr = float(perf.weighted_sum_rate)
            sum_dl = float(np.sum(perf.rate_dl))
            sum_ul = float(np.sum(perf.rate_ul))
            sum_s = float(np.sum(perf.rate_s))

            wsr_matrix[source_index, eval_index] = wsr
            dl_matrix[source_index, eval_index] = sum_dl
            ul_matrix[source_index, eval_index] = sum_ul
            sensing_matrix[source_index, eval_index] = sum_s

            long_rows.append(
                {
                    "scene_seed": SCENE_SEED,
                    "source_rho_db": source_db,
                    "source_rho_si": _rho_from_db(source_db),
                    "source_best_seed": int(
                        optimized_runs[source_db].best_seed
                    ),
                    "source_iterations": int(
                        optimized_runs[source_db].best_start.iterations
                    ),
                    "eval_rho_db": eval_db,
                    "eval_rho_si": _rho_from_db(eval_db),
                    "wsr": wsr,
                    "sum_dl_rate": sum_dl,
                    "sum_ul_rate": sum_ul,
                    "sum_sensing_rate": sum_s,
                }
            )

    # ------------------------------------------------------------
    # Step 3：理论一致性检查。
    # ------------------------------------------------------------
    print("\n===== Cross-rho fixed-solution WSR matrix =====")
    header = "source\\eval" + "".join(
        f" | {rho_db:>10} dB" for rho_db in RHO_DB_VALUES
    )
    print(header)
    print("-" * len(header))

    row_monotonic_flags = []

    for index, source_db in enumerate(RHO_DB_VALUES):
        values = wsr_matrix[index]
        monotonic = _is_nonincreasing(values, MONOTONIC_TOL)
        row_monotonic_flags.append(monotonic)

        text = f"{source_db:>8} dB" + "".join(
            f" | {value:>13.6f}" for value in values
        )
        text += f" | monotonic={monotonic}"
        print(text)

    # 对角线应该等于各 rho 点原始最佳运行的 WSR。
    diagonal_errors = []
    for index, rho_db in enumerate(RHO_DB_VALUES):
        original_wsr = float(
            optimized_runs[rho_db].best_result.performance.weighted_sum_rate
        )
        error = abs(wsr_matrix[index, index] - original_wsr)
        diagonal_errors.append(error)

    max_diagonal_error = float(max(diagonal_errors))

    # 每个 eval rho，从五个已经求出的 MA 解中选复用 WSR 最高者。
    reused_envelope = np.max(wsr_matrix, axis=0)
    envelope_source_indices = np.argmax(wsr_matrix, axis=0)
    envelope_source_db = [
        RHO_DB_VALUES[int(index)] for index in envelope_source_indices
    ]

    original_diagonal = np.diag(wsr_matrix).copy()

    print("\n===== Original diagonal vs reused-solution envelope =====")
    print("eval rho | original 3-start | reuse envelope | best source")
    print("-" * 64)

    summary_rows: list[dict[str, float | int]] = []

    for index, eval_db in enumerate(RHO_DB_VALUES):
        print(
            f"{eval_db:>7} dB | "
            f"{original_diagonal[index]:>16.6f} | "
            f"{reused_envelope[index]:>14.6f} | "
            f"{envelope_source_db[index]:>6} dB"
        )

        summary_rows.append(
            {
                "scene_seed": SCENE_SEED,
                "eval_rho_db": eval_db,
                "eval_rho_si": _rho_from_db(eval_db),
                "original_ma_wsr": float(original_diagonal[index]),
                "reuse_envelope_wsr": float(reused_envelope[index]),
                "reuse_best_source_rho_db": int(
                    envelope_source_db[index]
                ),
                "improvement": float(
                    reused_envelope[index] - original_diagonal[index]
                ),
            }
        )

    envelope_monotonic = _is_nonincreasing(
        reused_envelope,
        MONOTONIC_TOL,
    )

    print("\n===== Checks =====")
    print(f"all fixed-solution rows non-increasing = {all(row_monotonic_flags)}")
    print(f"reuse envelope non-increasing          = {envelope_monotonic}")
    print(f"max diagonal reconstruction error      = {max_diagonal_error:.3e}")

    if not all(row_monotonic_flags):
        raise RuntimeError(
            "Fixed-solution WSR is not monotonic in rho_si. "
            "Please re-check the covariance interference model."
        )

    if not envelope_monotonic:
        raise RuntimeError(
            "Reuse envelope is not monotonic; this should not happen "
            "if every fixed-solution row is monotonic."
        )

    if max_diagonal_error > DIAGONAL_TOL:
        raise RuntimeError(
            "Diagonal re-evaluation does not reproduce the original "
            f"optimized WSR (max error={max_diagonal_error:.3e})."
        )

    # ------------------------------------------------------------
    # Step 4：保存结果。
    # ------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    long_path = OUTPUT_DIR / "ma_cross_rho_reuse_scene8001.csv"
    matrix_path = OUTPUT_DIR / "ma_cross_rho_reuse_matrix_scene8001.csv"
    summary_path = OUTPUT_DIR / "ma_cross_rho_reuse_summary_scene8001.csv"

    with long_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(long_rows[0].keys()))
        writer.writeheader()
        writer.writerows(long_rows)

    matrix_fieldnames = [
        "source_rho_db",
        *[f"eval_{rho_db}_db" for rho_db in RHO_DB_VALUES],
        "row_nonincreasing",
    ]

    with matrix_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=matrix_fieldnames)
        writer.writeheader()

        for index, source_db in enumerate(RHO_DB_VALUES):
            row = {
                "source_rho_db": source_db,
                "row_nonincreasing": row_monotonic_flags[index],
            }
            for eval_index, eval_db in enumerate(RHO_DB_VALUES):
                row[f"eval_{eval_db}_db"] = float(
                    wsr_matrix[index, eval_index]
                )
            writer.writerow(row)

    with summary_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summary_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nSaved:")
    print(f"  long    = {long_path}")
    print(f"  matrix  = {matrix_path}")
    print(f"  summary = {summary_path}")
    print("\nCross-rho MA solution reuse diagnostic passed.")


if __name__ == "__main__":
    main()
