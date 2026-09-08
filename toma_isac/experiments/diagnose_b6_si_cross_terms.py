"""诊断 B6 (MA + FP) 在 residual-SI sweep 中的相干交叉项。

目的：
1. 仅针对一个 scene（默认 8001）运行 B6 的正式 MA 3-start；
2. 对每个 rho_si 选择最终 WSR 最高的收敛起点；
3. 将 UL / sensing 中
       ||echo + RSI||^2
   分解为
       ||echo||^2 + ||RSI||^2 + cross;
4. 同时记录 DL / UL / sensing 的速率与加权 WSR 贡献。

不修改主算法、metrics.py、inner_solver.py 或 multi-start 逻辑。
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from ma_baseline import run_ma_fp
from stage8 import (
    load_stage8_experiment_config,
    make_ma_start_seeds,
    make_scene_config,
)


CONFIG_PATH = ROOT / "configs" / "stage8_monte_carlo.yaml"
OUTPUT_DIR = ROOT / "results" / "stage8" / "diagnostics"


def _coherent_power_parts(
    echo_vec: np.ndarray,
    si_vec: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """返回 echo、SI、cross、total 与重构误差。"""

    echo_power = float(np.vdot(echo_vec, echo_vec).real)
    si_power = float(np.vdot(si_vec, si_vec).real)

    # ||a+b||^2 = ||a||^2 + ||b||^2 + 2 Re{a^H b}
    cross_power = float(
        2.0 * np.real(np.vdot(echo_vec, si_vec))
    )

    total_power = float(
        np.vdot(
            echo_vec + si_vec,
            echo_vec + si_vec,
        ).real
    )

    reconstruction_error = float(
        total_power
        - (echo_power + si_power + cross_power)
    )

    return (
        echo_power,
        si_power,
        cross_power,
        total_power,
        reconstruction_error,
    )


def _diagnose_best_result(
    result,
    cfg,
) -> tuple[dict[str, float], list[dict[str, float | int | str]]]:
    """分解最佳 MA 解中的 UL / sensing 干扰项。"""

    channels = result.channels
    resources = result.resources
    perf = result.performance
    q_matrix = resources.q_matrix

    g_sum = np.sum(channels.g, axis=0)

    detail_rows: list[dict[str, float | int | str]] = []

    ul_echo_sum = 0.0
    ul_si_sum = 0.0
    ul_cross_sum = 0.0
    ul_total_sum = 0.0
    ul_abs_error_max = 0.0

    for j in range(cfg.j_ul):
        b_j = resources.b_ul[j]

        echo_vec = (
            b_j.conj()
            @ g_sum
            @ q_matrix
        )
        si_vec = (
            b_j.conj()
            @ channels.h_rsi
            @ q_matrix
        )

        (
            echo_power,
            si_power,
            cross_power,
            total_power,
            reconstruction_error,
        ) = _coherent_power_parts(
            echo_vec,
            si_vec,
        )

        ul_echo_sum += echo_power
        ul_si_sum += si_power
        ul_cross_sum += cross_power
        ul_total_sum += total_power
        ul_abs_error_max = max(
            ul_abs_error_max,
            abs(reconstruction_error),
        )

        detail_rows.append(
            {
                "block": "UL",
                "index": j,
                "echo_power": echo_power,
                "si_power": si_power,
                "cross_power": cross_power,
                "total_power": total_power,
                "reconstruction_error": reconstruction_error,
            }
        )

    sensing_echo_sum = 0.0
    sensing_si_sum = 0.0
    sensing_cross_sum = 0.0
    sensing_total_sum = 0.0
    sensing_abs_error_max = 0.0

    for ell in range(cfg.l_target):
        u_l = resources.u_s[ell]

        g_other = g_sum - channels.g[ell]

        echo_vec = (
            u_l.conj()
            @ g_other
            @ q_matrix
        )
        si_vec = (
            u_l.conj()
            @ channels.h_rsi
            @ q_matrix
        )

        (
            echo_power,
            si_power,
            cross_power,
            total_power,
            reconstruction_error,
        ) = _coherent_power_parts(
            echo_vec,
            si_vec,
        )

        sensing_echo_sum += echo_power
        sensing_si_sum += si_power
        sensing_cross_sum += cross_power
        sensing_total_sum += total_power
        sensing_abs_error_max = max(
            sensing_abs_error_max,
            abs(reconstruction_error),
        )

        detail_rows.append(
            {
                "block": "Sensing",
                "index": ell,
                "echo_power": echo_power,
                "si_power": si_power,
                "cross_power": cross_power,
                "total_power": total_power,
                "reconstruction_error": reconstruction_error,
            }
        )

    weighted_dl = float(cfg.weight_dl @ perf.rate_dl)
    weighted_ul = float(cfg.weight_ul @ perf.rate_ul)
    weighted_s = float(cfg.weight_s @ perf.rate_s)

    summary = {
        "wsr": float(perf.weighted_sum_rate),
        "sum_rate_dl": float(np.sum(perf.rate_dl)),
        "sum_rate_ul": float(np.sum(perf.rate_ul)),
        "sum_rate_s": float(np.sum(perf.rate_s)),
        "weighted_dl": weighted_dl,
        "weighted_ul": weighted_ul,
        "weighted_s": weighted_s,
        "ul_echo_power": ul_echo_sum,
        "ul_si_power": ul_si_sum,
        "ul_cross_power": ul_cross_sum,
        "ul_total_fd_power": ul_total_sum,
        "sensing_echo_power": sensing_echo_sum,
        "sensing_si_power": sensing_si_sum,
        "sensing_cross_power": sensing_cross_sum,
        "sensing_total_fd_power": sensing_total_sum,
        "max_reconstruction_error": max(
            ul_abs_error_max,
            sensing_abs_error_max,
        ),
    }

    return summary, detail_rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scene",
        type=int,
        default=8001,
        help="Scene seed，默认 8001。",
    )

    parser.add_argument(
        "--rho-values",
        type=float,
        nargs="+",
        default=None,
        help=(
            "可选：手动指定 rho_si。"
            "默认读取 stage8_monte_carlo.yaml 中的正式 rho sweep。"
        ),
    )

    parser.add_argument(
        "--n-starts",
        type=int,
        default=None,
        help="可选：仅用于快速 smoke test；默认使用正式 MA n_starts。",
    )

    parser.add_argument(
        "--max-outer",
        type=int,
        default=None,
        help="可选：覆盖 MA max outer，仅用于快速测试。",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    base_cfg = load_config(CONFIG_PATH)
    exp_cfg = load_stage8_experiment_config(CONFIG_PATH)

    rho_values = (
        tuple(float(x) for x in args.rho_values)
        if args.rho_values is not None
        else exp_cfg.rho_si_values
    )

    n_starts = (
        exp_cfg.ma_n_starts
        if args.n_starts is None
        else int(args.n_starts)
    )

    ma_start_seeds = make_ma_start_seeds(
        scene_seed=args.scene,
        fixed_upa_seed=exp_cfg.fixed_upa_seed,
        n_starts=n_starts,
    )

    print("B6 MA + FP residual-SI cross-term diagnostic")
    print(f"  scene              = {args.scene}")
    print(f"  rho values         = {rho_values}")
    print(f"  MA start seeds     = {ma_start_seeds}")
    print(f"  region_size        = {exp_cfg.ma_region_size:.6f} m")
    print(f"  min_spacing        = {exp_cfg.ma_min_spacing:.6f} m")
    print(f"  initial_step       = {exp_cfg.ma_initial_step:.6f} m")
    print(f"  fading_enabled     = {exp_cfg.fading_enabled}")
    print(
        "  max_outer          = "
        + str(
            base_cfg.max_outer_iter
            if args.max_outer is None
            else args.max_outer
        )
    )

    summary_rows: list[dict[str, float | int | str | bool]] = []
    all_detail_rows: list[dict[str, float | int | str]] = []

    for rho_si in rho_values:
        print("\n" + "=" * 72)
        print(
            f"rho_si={rho_si:.1e}, "
            f"10log10(rho)={10.0 * np.log10(rho_si):.1f} dB"
        )

        cfg_rho = replace(
            base_cfg,
            rho_si=float(rho_si),
        )

        scene_cfg = make_scene_config(
            cfg_rho,
            scene_seed=args.scene,
            position_jitter=exp_cfg.position_jitter,
            randomize_fading=exp_cfg.fading_enabled,
            fading_model=exp_cfg.fading_model,
        )

        runs = []

        for start_index, seed in enumerate(ma_start_seeds, start=1):
            random_initial = (
                int(seed) != int(exp_cfg.fixed_upa_seed)
            )

            t0 = perf_counter()

            result = run_ma_fp(
                scene_cfg,
                rng=np.random.default_rng(seed),
                verbose=False,
                region_size=exp_cfg.ma_region_size,
                min_spacing=exp_cfg.ma_min_spacing,
                initial_step=exp_cfg.ma_initial_step,
                max_outer_iterations=args.max_outer,
                randomize_initial_geometry=random_initial,
                initial_jitter_fraction=(
                    exp_cfg.ma_initial_jitter_fraction
                ),
            )

            runtime_s = perf_counter() - t0

            print(
                f"  start {start_index}/{len(ma_start_seeds)} "
                f"seed={seed}: "
                f"WSR={result.performance.weighted_sum_rate:.10f}, "
                f"iter={result.iterations}, "
                f"conv={result.converged}, "
                f"stop={result.stop_reason}, "
                f"time={runtime_s:.2f}s"
            )

            if not result.converged:
                raise RuntimeError(
                    "B6 diagnostic encountered a non-converged start: "
                    f"rho_si={rho_si:.1e}, scene={args.scene}, seed={seed}."
                )

            runs.append(
                (
                    float(result.performance.weighted_sum_rate),
                    int(seed),
                    int(start_index),
                    float(runtime_s),
                    result,
                )
            )

        best = max(runs, key=lambda item: item[0])
        best_wsr, best_seed, best_index, best_runtime, best_result = best

        diag, detail_rows = _diagnose_best_result(
            best_result,
            scene_cfg,
        )

        summary_row: dict[str, float | int | str | bool] = {
            "scene_seed": int(args.scene),
            "rho_si": float(rho_si),
            "rho_db": float(10.0 * np.log10(rho_si)),
            "best_start_index": int(best_index),
            "best_seed": int(best_seed),
            "iterations": int(best_result.iterations),
            "stop_reason": str(best_result.stop_reason),
            "runtime_s_best_start": float(best_runtime),
            **diag,
        }
        summary_rows.append(summary_row)

        for row in detail_rows:
            all_detail_rows.append(
                {
                    "scene_seed": int(args.scene),
                    "rho_si": float(rho_si),
                    "rho_db": float(10.0 * np.log10(rho_si)),
                    "best_seed": int(best_seed),
                    **row,
                }
            )

        print("  [best-start decomposition]")
        print(f"    best seed          = {best_seed}")
        print(f"    WSR                = {best_wsr:.10f}")
        print(
            "    weighted parts     = "
            f"DL {diag['weighted_dl']:.6f}, "
            f"UL {diag['weighted_ul']:.6f}, "
            f"S {diag['weighted_s']:.6f}"
        )
        print(
            "    sum rates          = "
            f"DL {diag['sum_rate_dl']:.6f}, "
            f"UL {diag['sum_rate_ul']:.6f}, "
            f"S {diag['sum_rate_s']:.6f}"
        )
        print(
            "    UL powers          = "
            f"echo {diag['ul_echo_power']:.3e}, "
            f"SI {diag['ul_si_power']:.3e}, "
            f"cross {diag['ul_cross_power']:.3e}, "
            f"total {diag['ul_total_fd_power']:.3e}"
        )
        print(
            "    Sensing powers     = "
            f"echo {diag['sensing_echo_power']:.3e}, "
            f"SI {diag['sensing_si_power']:.3e}, "
            f"cross {diag['sensing_cross_power']:.3e}, "
            f"total {diag['sensing_total_fd_power']:.3e}"
        )
        print(
            "    max recon. error   = "
            f"{diag['max_reconstruction_error']:.3e}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = (
        OUTPUT_DIR
        / f"b6_si_cross_term_summary_scene_{args.scene}.csv"
    )
    detail_path = (
        OUTPUT_DIR
        / f"b6_si_cross_term_detail_scene_{args.scene}.csv"
    )

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(summary_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(all_detail_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(all_detail_rows)

    print("\n" + "=" * 72)
    print("[Saved]")
    print(f"  summary = {summary_path}")
    print(f"  detail  = {detail_path}")

    print("\n[How to read]")
    print(
        "  1) 若 cross_power 随 rho 增大明显变得更负，"
        "说明相干 echo-SI 相消是 MA WSR 上升的重要来源。"
    )
    print(
        "  2) 若 UL/Sensing weighted part 下降而 DL weighted part 上升，"
        "说明资源重分配也在补偿 SI。"
    )
    print(
        "  3) reconstruction_error 应接近 0，"
        "用于验证分解公式与当前 metrics.py 完全一致。"
    )


if __name__ == "__main__":
    main()
