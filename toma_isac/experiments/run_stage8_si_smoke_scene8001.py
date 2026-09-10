"""Scene 8001：新版协方差干扰模型下的 B6 MA 与 Proposed 五点 rho_si smoke。"""

from __future__ import annotations

import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm import run_joint_algorithm
from config import load_config
from ma_baseline import run_ma_fp
from multistart import run_multi_start
from stage8 import (
    load_stage8_experiment_config,
    make_ma_start_seeds,
    make_scene_config,
)


CONFIG_PATH = ROOT / "configs" / "stage8_monte_carlo.yaml"
OUTPUT_PATH = ROOT / "results" / "stage8" / "stage8_si_smoke_scene8001.csv"

SCENE_SEED = 8001
RHO_DB_VALUES = (-50, -40, -30, -20, -10)


def _sum_rates(performance) -> tuple[float, float, float]:
    """返回 DL / UL / sensing 的总速率。"""

    return (
        float(np.sum(performance.rate_dl)),
        float(np.sum(performance.rate_ul)),
        float(np.sum(performance.rate_s)),
    )


def _weighted_parts(performance, cfg) -> tuple[float, float, float]:
    """返回 WSR 中 DL / UL / sensing 三部分贡献。"""

    return (
        float(np.dot(cfg.weight_dl, performance.rate_dl)),
        float(np.dot(cfg.weight_ul, performance.rate_ul)),
        float(np.dot(cfg.weight_s, performance.rate_s)),
    )


def main() -> None:
    base_cfg = load_config(CONFIG_PATH)
    exp_cfg = load_stage8_experiment_config(CONFIG_PATH)

    # scene 只生成一次，保证五个 rho 点使用完全相同的位置/衰落 realization。
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
    proposed_seeds = exp_cfg.multi_start_seeds

    print(f"scene={SCENE_SEED}")
    print(f"MA seeds={ma_seeds}")
    print(f"Proposed seeds={proposed_seeds}")
    print()

    rows: list[dict[str, float | int]] = []

    for rho_db in RHO_DB_VALUES:
        rho_si = 10.0 ** (rho_db / 10.0)
        cfg = replace(scene_cfg, rho_si=rho_si)
        cfg.validate()

        print(f"===== rho_si={rho_si:.1e} ({rho_db} dB) =====")

        # B6：1 个固定 UPA 起点 + 随机 MA 起点，全部收敛后取最佳 WSR。
        ma_result = run_multi_start(
            name="B6: Multi-start MA + FP Resource Optimization",
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

        # Proposed：FP-aware RCG-ToMA + FP，沿用正式 Stage 8 multi-start。
        proposed_result = run_multi_start(
            name="Proposed: FP-aware RCG-ToMA + FP Resource Optimization",
            seeds=proposed_seeds,
            run_single=lambda seed: run_joint_algorithm(
                cfg,
                rng=np.random.default_rng(seed),
                verbose=False,
                position_update_mode="fp_aware",
                fp_aware_candidate_iters=2,
            ),
            iteration_getter=lambda result: result.outer_iterations,
            cfg=cfg,
            require_convergence=True,
            verbose=True,
        )

        ma_perf = ma_result.best_result.performance
        proposed_perf = proposed_result.best_result.performance

        ma_dl, ma_ul, ma_s = _sum_rates(ma_perf)
        proposed_dl, proposed_ul, proposed_s = _sum_rates(proposed_perf)

        ma_wdl, ma_wul, ma_ws = _weighted_parts(ma_perf, cfg)
        proposed_wdl, proposed_wul, proposed_ws = _weighted_parts(
            proposed_perf,
            cfg,
        )

        row = {
            "scene_seed": SCENE_SEED,
            "rho_db": rho_db,
            "rho_si": rho_si,
            "ma_wsr": float(ma_perf.weighted_sum_rate),
            "ma_sum_dl_rate": ma_dl,
            "ma_sum_ul_rate": ma_ul,
            "ma_sum_sensing_rate": ma_s,
            "ma_weighted_dl": ma_wdl,
            "ma_weighted_ul": ma_wul,
            "ma_weighted_sensing": ma_ws,
            "ma_best_seed": int(ma_result.best_seed),
            "ma_iterations": int(ma_result.best_start.iterations),
            "ma_runtime_s": float(ma_result.total_runtime_s),
            "proposed_wsr": float(proposed_perf.weighted_sum_rate),
            "proposed_sum_dl_rate": proposed_dl,
            "proposed_sum_ul_rate": proposed_ul,
            "proposed_sum_sensing_rate": proposed_s,
            "proposed_weighted_dl": proposed_wdl,
            "proposed_weighted_ul": proposed_wul,
            "proposed_weighted_sensing": proposed_ws,
            "proposed_best_seed": int(proposed_result.best_seed),
            "proposed_iterations": int(
                proposed_result.best_start.iterations
            ),
            "proposed_runtime_s": float(proposed_result.total_runtime_s),
        }
        rows.append(row)

        print(
            f"BEST | MA={row['ma_wsr']:.10f} "
            f"(seed={row['ma_best_seed']}, iter={row['ma_iterations']}) | "
            f"Proposed={row['proposed_wsr']:.10f} "
            f"(seed={row['proposed_best_seed']}, "
            f"iter={row['proposed_iterations']})"
        )
        print(
            "MA rates: "
            f"DL={ma_dl:.6f}, UL={ma_ul:.6f}, S={ma_s:.6f}"
        )
        print(
            "Proposed rates: "
            f"DL={proposed_dl:.6f}, UL={proposed_ul:.6f}, "
            f"S={proposed_s:.6f}\n"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("===== Smoke summary =====")
    for row in rows:
        print(
            f"{int(row['rho_db']):>4} dB | "
            f"MA={row['ma_wsr']:.6f} | "
            f"Proposed={row['proposed_wsr']:.6f}"
        )

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
