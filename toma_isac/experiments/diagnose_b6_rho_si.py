"""Stage 8B：B6 (MA + FP) 的 rho_SI 专项诊断。

目的
----
只诊断 B6 在 residual-SI sweep 中出现“rho_SI 越大，最终 WSR 反而升高”的现象。

本脚本固定：
- scene_seed（用户/目标位置 + scene-level fading）
- B6 的 UPA 初始几何
- fixed_upa_seed（资源初始化随机流）
- MA region / min spacing / initial step

只改变：
- rho_si = 1e-3, 1e-2, 1e-1, 1

同时做三类检查：
1. 配置传递检查：实际 H_RSI 功率比例是否等于 rho_si；
2. 初始化公平性检查：各 rho 点的初始 MA geometry / initial resources 是否一致；
3. 局部最优诊断：把每个 rho 点优化得到的最终 MA+FP 解，交叉放到其他 rho 下重新评估 WSR。

如果高-rho 得到的解在低-rho 下也明显优于“低-rho 自己求出来的解”，
通常说明异常趋势主要来自 MA+FP 非凸优化落入不同局部最优，而不是 rho_si 没有传进去。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np

from baselines import build_fpa_upa_geometry
from channels import build_channels
from config import load_config
from initialization import initialize_resources
from ma_baseline import run_ma_fp
from metrics import compute_performance
from stage8 import (
    load_stage8_experiment_config,
    make_scene_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "stage8_monte_carlo.yaml"
OUTPUT_DIR = ROOT / "results" / "stage8" / "diagnostics"

DEFAULT_RHO_VALUES = (1.0e-3, 1.0e-2, 1.0e-1, 1.0)


def _array_checksum(array: np.ndarray) -> str:
    """给数组生成短 checksum，用于确认不同 rho 下初始化是否完全相同。"""

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256(contiguous.view(np.uint8)).hexdigest()
    return digest[:12]


def _resource_checksum(resources) -> str:
    """合并 ResourceState 四个变量的 checksum。"""

    hasher = hashlib.sha256()
    for array in (
        resources.q_matrix,
        resources.q_ul,
        resources.b_ul,
        resources.u_s,
    ):
        contiguous = np.ascontiguousarray(array)
        hasher.update(contiguous.view(np.uint8))
    return hasher.hexdigest()[:12]


def _signal_channel_checksum(channels) -> str:
    """只对 h/f/g 做 checksum；它们理论上不应随 rho_si 改变。"""

    hasher = hashlib.sha256()
    for array in (
        channels.h,
        channels.f,
        channels.g,
    ):
        contiguous = np.ascontiguousarray(array)
        hasher.update(contiguous.view(np.uint8))
    return hasher.hexdigest()[:12]


def _measured_rho(channels) -> float:
    """由 ||H_RSI||_F^2 / ||H_SI0||_F^2 反算实际 residual-SI 功率比例。"""

    denominator = float(np.linalg.norm(channels.h_si0, "fro") ** 2)
    numerator = float(np.linalg.norm(channels.h_rsi, "fro") ** 2)

    if denominator <= 1.0e-30:
        return np.nan

    return numerator / denominator


def _initial_state_for_rho(
    scene_cfg,
    rho_si: float,
    fixed_seed: int,
    tx_rx_separation: float | None,
):
    """重建 B6 在指定 rho 下的确定性初始状态。"""

    cfg_rho = replace(scene_cfg, rho_si=float(rho_si))
    cfg_rho.validate()

    geometry = build_fpa_upa_geometry(
        cfg_rho,
        tx_rx_separation=tx_rx_separation,
    )
    channels = build_channels(geometry, cfg_rho)
    resources = initialize_resources(
        geometry,
        channels,
        cfg_rho,
        np.random.default_rng(int(fixed_seed)),
    )
    performance = compute_performance(channels, resources, cfg_rho)

    return cfg_rho, geometry, channels, resources, performance


def _save_summary_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_cross_eval_csv(
    rho_values: tuple[float, ...],
    matrix: np.ndarray,
    path: Path,
) -> None:
    """行=解在哪个 rho 下优化；列=把该冻结解放到哪个 rho 下评估。"""

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["optimized_at_rho"] + [
        f"eval_rho_{rho:.0e}" for rho in rho_values
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row_index, rho_opt in enumerate(rho_values):
            row: dict[str, float] = {
                "optimized_at_rho": float(rho_opt),
            }
            for col_index, rho_eval in enumerate(rho_values):
                row[f"eval_rho_{rho_eval:.0e}"] = float(
                    matrix[row_index, col_index]
                )
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose the abnormal B6 MA+FP trend across rho_SI."
    )
    parser.add_argument(
        "--scene-seed",
        type=int,
        default=8001,
        help="Fixed Monte-Carlo scene seed (default: 8001).",
    )
    parser.add_argument(
        "--fixed-seed",
        type=int,
        default=None,
        help=(
            "B6 initialization seed. Default: experiment.fixed_upa_seed "
            "from stage8_monte_carlo.yaml."
        ),
    )
    parser.add_argument(
        "--rho",
        type=float,
        nargs="+",
        default=list(DEFAULT_RHO_VALUES),
        help="rho_SI values to diagnose.",
    )
    parser.add_argument(
        "--verbose-ma",
        action="store_true",
        help="Print every MA+FP outer iteration.",
    )
    args = parser.parse_args()

    base_cfg = load_config(CONFIG_PATH)
    experiment_cfg = load_stage8_experiment_config(CONFIG_PATH)

    fixed_seed = (
        experiment_cfg.fixed_upa_seed
        if args.fixed_seed is None
        else int(args.fixed_seed)
    )

    rho_values = tuple(float(value) for value in args.rho)

    if any(value < 0.0 or value > 1.0 for value in rho_values):
        raise ValueError("Every rho_si must lie in [0, 1].")

    # ------------------------------------------------------------------
    # 关键公平性处理：scene 只生成一次。
    # 后面各 rho 点均由同一个 scene_cfg 仅 replace(rho_si=...) 得到。
    # 因而 q_dl/q_ul/q_s 与 random fading realization 完全一致。
    # ------------------------------------------------------------------
    scene_cfg = make_scene_config(
        base_cfg,
        scene_seed=int(args.scene_seed),
        position_jitter=experiment_cfg.position_jitter,
        randomize_fading=experiment_cfg.fading_enabled,
        fading_model=experiment_cfg.fading_model,
    )

    print("=" * 88)
    print("B6 MA + FP residual-SI diagnostic")
    print("=" * 88)
    print(f"scene_seed          = {args.scene_seed}")
    print(f"fixed_upa_seed      = {fixed_seed}")
    print(f"random fading       = {experiment_cfg.fading_enabled}")
    print(f"fading model        = {experiment_cfg.fading_model}")
    print(f"MA region_size      = {experiment_cfg.ma_region_size:.6g} m")
    print(f"MA min_spacing      = {experiment_cfg.ma_min_spacing:.6g} m")
    print(f"MA initial_step     = {experiment_cfg.ma_initial_step:.6g} m")
    print(f"rho values          = {rho_values}")
    print()

    # scene-level fading 只打印 checksum，避免终端过长。
    scene_fading_checksum = hashlib.sha256()
    for array in (scene_cfg.alpha_dl, scene_cfg.alpha_ul, scene_cfg.xi_s):
        scene_fading_checksum.update(np.ascontiguousarray(array).view(np.uint8))
    print(
        "scene fading checksum = "
        f"{scene_fading_checksum.hexdigest()[:12]} "
        "(must stay fixed across all rho values)"
    )
    print()

    initial_geometry_checksums: list[str] = []
    initial_resource_checksums: list[str] = []
    signal_channel_checksums: list[str] = []

    summary_rows: list[dict[str, object]] = []
    final_results = []
    cfg_by_rho = []

    # ------------------------------------------------------------------
    # Part A: 每个 rho 的初始化 + 实际 B6 优化。
    # ------------------------------------------------------------------
    for rho_si in rho_values:
        (
            cfg_rho,
            initial_geometry,
            initial_channels,
            initial_resources,
            initial_performance,
        ) = _initial_state_for_rho(
            scene_cfg,
            rho_si=rho_si,
            fixed_seed=fixed_seed,
            tx_rx_separation=None,
        )

        geometry_checksum = (
            _array_checksum(initial_geometry.tx_positions)
            + "/"
            + _array_checksum(initial_geometry.rx_positions)
        )
        resource_checksum = _resource_checksum(initial_resources)
        signal_checksum = _signal_channel_checksum(initial_channels)
        measured_rho = _measured_rho(initial_channels)

        initial_geometry_checksums.append(geometry_checksum)
        initial_resource_checksums.append(resource_checksum)
        signal_channel_checksums.append(signal_checksum)

        print("-" * 88)
        print(f"rho_si = {rho_si:.3e}  ({10.0 * np.log10(rho_si):.1f} dB)")
        print(f"cfg.rho_si                 = {cfg_rho.rho_si:.6e}")
        print(f"measured RSI power ratio   = {measured_rho:.6e}")
        print(f"initial geometry checksum  = {geometry_checksum}")
        print(f"initial resource checksum  = {resource_checksum}")
        print(f"h/f/g channel checksum     = {signal_checksum}")
        print(
            f"initial WSR                = "
            f"{float(initial_performance.weighted_sum_rate):.10f}"
        )

        result = run_ma_fp(
            cfg_rho,
            rng=np.random.default_rng(fixed_seed),
            verbose=bool(args.verbose_ma),
            tx_rx_separation=None,
            region_size=experiment_cfg.ma_region_size,
            min_spacing=experiment_cfg.ma_min_spacing,
            initial_step=experiment_cfg.ma_initial_step,
        )

        final_wsr = float(result.performance.weighted_sum_rate)
        final_rho_measured = _measured_rho(result.channels)
        tx_displacement = float(
            np.linalg.norm(
                result.geometry.tx_positions - result.initial_tx_positions
            )
        )
        rx_displacement = float(
            np.linalg.norm(
                result.geometry.rx_positions - result.initial_rx_positions
            )
        )

        monotone_violations = int(
            np.sum(np.diff(np.asarray(result.wsr_history, dtype=float)) < -1.0e-10)
        )

        print(f"final WSR                  = {final_wsr:.10f}")
        print(f"iterations                 = {result.iterations}")
        print(f"stop_reason                = {result.stop_reason}")
        print(f"final measured rho         = {final_rho_measured:.6e}")
        print(f"Tx total displacement norm = {tx_displacement:.6e} m")
        print(f"Rx total displacement norm = {rx_displacement:.6e} m")
        print(f"WSR-history decreases      = {monotone_violations}")

        summary_rows.append(
            {
                "scene_seed": int(args.scene_seed),
                "fixed_seed": int(fixed_seed),
                "rho_si": float(rho_si),
                "rho_db": float(10.0 * np.log10(rho_si)),
                "measured_rho_initial": float(measured_rho),
                "measured_rho_final": float(final_rho_measured),
                "initial_wsr": float(initial_performance.weighted_sum_rate),
                "final_wsr": final_wsr,
                "iterations": int(result.iterations),
                "stop_reason": str(result.stop_reason),
                "tx_displacement_norm_m": tx_displacement,
                "rx_displacement_norm_m": rx_displacement,
                "wsr_history_decreases": monotone_violations,
                "initial_geometry_checksum": geometry_checksum,
                "initial_resource_checksum": resource_checksum,
                "signal_channel_checksum": signal_checksum,
            }
        )

        final_results.append(result)
        cfg_by_rho.append(cfg_rho)

    # ------------------------------------------------------------------
    # Part B: 公平性检查。
    # ------------------------------------------------------------------
    same_geometry = len(set(initial_geometry_checksums)) == 1
    same_resources = len(set(initial_resource_checksums)) == 1
    same_signal_channels = len(set(signal_channel_checksums)) == 1

    print("\n" + "=" * 88)
    print("Initialization / configuration fairness checks")
    print("=" * 88)
    print(f"same initial MA geometry across rho : {same_geometry}")
    print(f"same initial resources across rho   : {same_resources}")
    print(f"same h/f/g channels across rho      : {same_signal_channels}")

    rho_match = all(
        np.isclose(row["measured_rho_initial"], row["rho_si"], rtol=1e-10, atol=1e-14)
        and np.isclose(row["measured_rho_final"], row["rho_si"], rtol=1e-10, atol=1e-14)
        for row in summary_rows
    )
    print(f"rho_si actually reaches H_RSI       : {rho_match}")

    # ------------------------------------------------------------------
    # Part C: 交叉评估。
    # 行 i：在 rho_i 下优化得到的最终 geometry + resources。
    # 列 j：冻结该解，只把 rho 改成 rho_j 后重新计算 WSR。
    # ------------------------------------------------------------------
    cross_eval = np.empty((len(rho_values), len(rho_values)), dtype=float)

    for row_index, result in enumerate(final_results):
        for col_index, rho_eval in enumerate(rho_values):
            cfg_eval = replace(scene_cfg, rho_si=float(rho_eval))
            cfg_eval.validate()
            channels_eval = build_channels(result.geometry, cfg_eval)
            performance_eval = compute_performance(
                channels_eval,
                result.resources,
                cfg_eval,
            )
            cross_eval[row_index, col_index] = float(
                performance_eval.weighted_sum_rate
            )

    print("\n" + "=" * 88)
    print("Cross-evaluation matrix")
    print("row = solution optimized at rho; column = frozen solution evaluated at rho")
    print("=" * 88)

    header = "optimized\\eval" + "".join(
        f" | {rho:>11.1e}" for rho in rho_values
    )
    print(header)
    print("-" * len(header))

    for row_index, rho_opt in enumerate(rho_values):
        values = "".join(
            f" | {cross_eval[row_index, col_index]:11.6f}"
            for col_index in range(len(rho_values))
        )
        print(f"{rho_opt:>14.1e}{values}")

    # ------------------------------------------------------------------
    # Part D: 自动判读最重要的局部最优信号。
    # 对每个 eval rho，比较“该 rho 自己跑出的解”和所有其他 rho 跑出的解。
    # 如果其他 rho 的解在这里更好，说明当前单起点 MA+FP 没找到最好局部解。
    # ------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("Local-optimum diagnostic")
    print("=" * 88)

    local_optimum_warning = False

    for col_index, rho_eval in enumerate(rho_values):
        own_wsr = cross_eval[col_index, col_index]
        best_row = int(np.argmax(cross_eval[:, col_index]))
        best_wsr = cross_eval[best_row, col_index]
        gain = best_wsr - own_wsr

        print(
            f"eval rho={rho_eval:.1e}: own-run={own_wsr:.6f}, "
            f"best frozen solution={best_wsr:.6f} "
            f"(optimized at rho={rho_values[best_row]:.1e}), "
            f"gap={gain:+.6f}"
        )

        if gain > 1.0e-5:
            local_optimum_warning = True

    own_final_wsr = np.asarray(
        [cross_eval[i, i] for i in range(len(rho_values))],
        dtype=float,
    )
    own_trend_increases = np.where(np.diff(own_final_wsr) > 1.0e-8)[0]

    print()
    print(f"own B6 final WSR = {own_final_wsr}")
    print(
        "abnormal upward rho steps = "
        f"{own_trend_increases.tolist()} "
        "(empty means non-increasing)"
    )
    print(f"evidence of different local optima = {local_optimum_warning}")

    # ------------------------------------------------------------------
    # 保存诊断 CSV。
    # ------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = (
        OUTPUT_DIR
        / f"b6_rho_si_diagnostic_scene_{args.scene_seed}.csv"
    )
    cross_path = (
        OUTPUT_DIR
        / f"b6_rho_si_cross_eval_scene_{args.scene_seed}.csv"
    )

    _save_summary_csv(summary_rows, summary_path)
    _save_cross_eval_csv(rho_values, cross_eval, cross_path)

    print("\n" + "=" * 88)
    print("Diagnostic finished")
    print("=" * 88)
    print(f"summary CSV   : {summary_path}")
    print(f"cross-eval CSV: {cross_path}")
    print()

    if not rho_match:
        print("[CHECK FIRST] rho_si is not propagated correctly into H_RSI.")
    elif not same_geometry or not same_resources or not same_signal_channels:
        print(
            "[CHECK FIRST] rho sweep is not using identical initialization / "
            "signal channels."
        )
    elif local_optimum_warning:
        print(
            "[LIKELY CAUSE] rho propagation and initialization are consistent, "
            "but different rho runs converge to different MA+FP local optima."
        )
    else:
        print(
            "[NEXT CHECK] rho propagation and initialization look consistent; "
            "no strong local-optimum dominance was detected in this scene."
        )


if __name__ == "__main__":
    main()
