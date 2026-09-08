"""B6 MA + FP 在 -10 dB residual-SI 下的 3-start 压力测试。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter
import argparse
import csv
import sys

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
OUTPUT_PATH = ROOT / "results" / "stage8" / "b6_ma_minus10db_pressure.csv"

SCENE_SEED = 8001
RHO_SI = 1.0e-1  # -10 dB


def _relative_change(new: float, old: float) -> float:
    return abs(new - old) / max(1.0, abs(old))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-outer",
        type=int,
        default=None,
        help="Only for a quick smoke test; default uses cfg.max_outer_iter.",
    )
    args = parser.parse_args()

    base_cfg = load_config(CONFIG_PATH)
    experiment_cfg = load_stage8_experiment_config(CONFIG_PATH)

    # 本测试只覆盖 residual SI，其他 nominal 参数保持 Stage 8 正式配置。
    rho_cfg = replace(base_cfg, rho_si=RHO_SI)
    scene_cfg = make_scene_config(
        rho_cfg,
        scene_seed=SCENE_SEED,
        position_jitter=experiment_cfg.position_jitter,
        randomize_fading=experiment_cfg.fading_enabled,
        fading_model=experiment_cfg.fading_model,
    )

    ma_start_seeds = make_ma_start_seeds(
        scene_seed=SCENE_SEED,
        fixed_upa_seed=experiment_cfg.fixed_upa_seed,
        n_starts=experiment_cfg.ma_n_starts,
    )

    print("B6 MA + FP -10 dB pressure test")
    print(f"  scene_seed       = {SCENE_SEED}")
    print(f"  rho_si           = {RHO_SI:.1e} (-10 dB)")
    print(f"  region_size      = {experiment_cfg.ma_region_size:.6f} m")
    print(f"  min_spacing      = {experiment_cfg.ma_min_spacing:.6f} m")
    print(f"  initial_step     = {experiment_cfg.ma_initial_step:.6f} m")
    print(f"  max_inner_iter   = {scene_cfg.max_inner_iter}")
    print(
        "  max_outer_iter   = "
        f"{scene_cfg.max_outer_iter if args.max_outer is None else args.max_outer}"
    )
    print(f"  MA start seeds   = {ma_start_seeds}")

    records: list[dict[str, object]] = []

    for index, seed in enumerate(ma_start_seeds, start=1):
        randomize_initial_geometry = (
            int(seed) != int(experiment_cfg.fixed_upa_seed)
        )

        t0 = perf_counter()
        result = run_ma_fp(
            scene_cfg,
            rng=np.random.default_rng(seed),
            verbose=False,
            region_size=experiment_cfg.ma_region_size,
            min_spacing=experiment_cfg.ma_min_spacing,
            initial_step=experiment_cfg.ma_initial_step,
            max_outer_iterations=args.max_outer,
            randomize_initial_geometry=randomize_initial_geometry,
            initial_jitter_fraction=(
                experiment_cfg.ma_initial_jitter_fraction
            ),
        )
        runtime_s = perf_counter() - t0

        history = np.asarray(result.wsr_history, dtype=float)
        final_change = (
            _relative_change(history[-1], history[-2])
            if history.size >= 2
            else 0.0
        )

        window = min(20, max(0, history.size - 1))
        if window > 0:
            last_window_gain = (
                history[-1] - history[-1 - window]
            ) / max(1.0, abs(history[-1 - window]))
        else:
            last_window_gain = 0.0

        tx_displacement = float(
            np.linalg.norm(
                result.geometry.tx_positions
                - result.initial_tx_positions
            )
        )
        rx_displacement = float(
            np.linalg.norm(
                result.geometry.rx_positions
                - result.initial_rx_positions
            )
        )

        record = {
            "start_index": index,
            "seed": int(seed),
            "random_initial_geometry": randomize_initial_geometry,
            "iterations": int(result.iterations),
            "converged": bool(result.converged),
            "stop_reason": str(result.stop_reason),
            "initial_wsr": float(history[0]),
            "final_wsr": float(history[-1]),
            "final_relative_change": float(final_change),
            "last_20_iter_relative_gain": float(last_window_gain),
            "tx_total_displacement_m": tx_displacement,
            "rx_total_displacement_m": rx_displacement,
            "runtime_s": float(runtime_s),
        }
        records.append(record)

        print(
            f"\nStart {index}/{len(ma_start_seeds)}  seed={seed}"
        )
        print(f"  iterations       = {result.iterations}")
        print(f"  converged        = {result.converged}")
        print(f"  stop_reason      = {result.stop_reason}")
        print(f"  initial WSR      = {history[0]:.10f}")
        print(f"  final WSR        = {history[-1]:.10f}")
        print(f"  final rel change = {final_change:.3e}")
        print(f"  last-20 gain     = {last_window_gain:.3e}")
        print(f"  runtime          = {runtime_s:.2f} s")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    best = max(records, key=lambda row: float(row["final_wsr"]))
    print("\n[Summary]")
    print(
        "  best start = "
        f"{best['start_index']} (seed={best['seed']})"
    )
    print(f"  best WSR   = {float(best['final_wsr']):.10f}")
    print(f"  CSV saved  = {OUTPUT_PATH}")

    hit_limit = [
        row
        for row in records
        if row["stop_reason"] == "max_outer_iterations"
    ]

    if hit_limit:
        print("\n[Decision] At least one start hit max_outer_iterations.")
        print(
            "Check last_20_iter_relative_gain first; "
            "if it is still clearly positive, raise max_outer_iter to 1500."
        )
    else:
        print("\n[Decision] All 3 starts stopped before the outer limit.")
        print("Keep max_outer_iter = 1000.")


if __name__ == "__main__":
    main()
