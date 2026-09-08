"""运行无交叉项协方差模型的 scene 8001 两组 smoke experiments。"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from algorithm import run_joint_algorithm
from channels import build_channels
from config import load_config
from geometry import build_geometry
from initialization import (
    generate_feasible_endpoints,
    initialize_resources,
)
from ma_baseline import run_ma_fp
from metrics import compute_performance
from multistart import run_multi_start
from stage8 import (
    load_stage8_experiment_config,
    make_ma_start_seeds,
    make_scene_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "stage8_monte_carlo.yaml"
OUTPUT_PATH = (
    ROOT
    / "results"
    / "stage8"
    / "covariance_smoke_scene8001.csv"
)
SCENE_SEED = 8001


def _scene_config():
    base_cfg = load_config(CONFIG_PATH)
    experiment_cfg = load_stage8_experiment_config(CONFIG_PATH)
    scene_cfg = make_scene_config(
        base_cfg,
        scene_seed=SCENE_SEED,
        position_jitter=experiment_cfg.position_jitter,
        randomize_fading=experiment_cfg.fading_enabled,
        fading_model=experiment_cfg.fading_model,
    )
    return scene_cfg, experiment_cfg


def run_fixed_state_check(scene_cfg) -> list[dict[str, float]]:
    """固定 scene 8001 的几何与资源，扫描五个 rho 点。"""

    rng = np.random.default_rng(2026)
    endpoints = generate_feasible_endpoints(scene_cfg, rng)
    geometry = build_geometry(endpoints, scene_cfg)
    reference_channels = build_channels(geometry, scene_cfg)
    resources = initialize_resources(
        geometry,
        reference_channels,
        scene_cfg,
        rng,
    )

    rows = []
    for rho_db in (-50, -40, -30, -20, -10):
        rho_si = 10.0 ** (rho_db / 10.0)
        cfg = replace(scene_cfg, rho_si=rho_si)
        channels = build_channels(geometry, cfg)
        performance = compute_performance(channels, resources, cfg)
        rows.append(
            {
                "rho_db": float(rho_db),
                "rho_si": rho_si,
                "wsr": float(performance.weighted_sum_rate),
                "sum_dl_rate": float(np.sum(performance.rate_dl)),
                "sum_ul_rate": float(np.sum(performance.rate_ul)),
                "sum_sensing_rate": float(np.sum(performance.rate_s)),
            }
        )

    wsr = np.array([row["wsr"] for row in rows])
    if not np.all(np.diff(wsr) <= 1e-10):
        raise RuntimeError("Fixed-state WSR is not non-increasing in rho_si.")

    return rows


def run_b6_proposed_sweep(scene_cfg, experiment_cfg):
    """只运行 B6 MA+FP 与 Proposed 的五点 rho sweep。"""

    ma_seeds = make_ma_start_seeds(
        scene_seed=SCENE_SEED,
        fixed_upa_seed=experiment_cfg.fixed_upa_seed,
        n_starts=experiment_cfg.ma_n_starts,
    )
    proposed_seeds = experiment_cfg.multi_start_seeds
    rows = []

    for rho_si in experiment_cfg.rho_si_values:
        cfg = replace(scene_cfg, rho_si=float(rho_si))
        cfg.validate()

        ma_result = run_multi_start(
            name="B6: Multi-start MA + FP Resource Optimization",
            seeds=ma_seeds,
            run_single=lambda seed: run_ma_fp(
                cfg,
                rng=np.random.default_rng(seed),
                verbose=False,
                region_size=experiment_cfg.ma_region_size,
                min_spacing=experiment_cfg.ma_min_spacing,
                initial_step=experiment_cfg.ma_initial_step,
                randomize_initial_geometry=(
                    int(seed) != int(experiment_cfg.fixed_upa_seed)
                ),
                initial_jitter_fraction=(
                    experiment_cfg.ma_initial_jitter_fraction
                ),
            ),
            iteration_getter=lambda result: result.iterations,
            cfg=cfg,
            require_convergence=True,
            verbose=False,
        )

        proposed_result = run_multi_start(
            name="Proposed: RCG-ToMA + FP Resource Optimization",
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
            verbose=False,
        )

        ma_best = ma_result.best_start.result.performance
        proposed_best = proposed_result.best_start.result.performance
        row = {
            "rho_si": float(rho_si),
            "rho_db": float(10.0 * np.log10(rho_si)),
            "ma_wsr": float(ma_best.weighted_sum_rate),
            "proposed_wsr": float(proposed_best.weighted_sum_rate),
            "sum_dl_rate": float(np.sum(proposed_best.rate_dl)),
            "sum_ul_rate": float(np.sum(proposed_best.rate_ul)),
            "sum_sensing_rate": float(np.sum(proposed_best.rate_s)),
        }
        rows.append(row)
        print(row, flush=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    return rows


def main() -> None:
    scene_cfg, experiment_cfg = _scene_config()

    print("Fixed-state scene 8001:")
    for row in run_fixed_state_check(scene_cfg):
        print(row)

    print("B6/Proposed scene 8001 sweep:")
    run_b6_proposed_sweep(scene_cfg, experiment_cfg)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
