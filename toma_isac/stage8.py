"""Stage 8：Monte Carlo、参数扫描与结果保存。

本文件只负责实验编排，不重新实现 FP、RCG、SINR 或 WSR。

Stage 8 中：
- scene seed 只改变 q_dl / q_ul / q_s；
- Multi-start seed 只控制算法初始 ToMA；
- B1/B2/B4/Proposed 使用相同 Multi-start seed 集合；
- B3/B5 为固定 UPA，每个 scene 只运行一次。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

from algorithm import run_joint_algorithm
from baselines import (
    run_fixed_toma_fp,
    run_fixed_toma_traditional,
    run_fpa_upa_fp,
    run_fpa_upa_traditional,
    run_rcg_toma_traditional,
)
from config import SystemConfig
from multistart import MultiStartResult, run_multi_start


SCHEME_ORDER = (
    "b1",
    "b2",
    "b3",
    "b4",
    "b5",
    "proposed",
)

SCHEME_LABELS = {
    "b1": "B1",
    "b2": "B2",
    "b3": "B3",
    "b4": "B4",
    "b5": "B5",
    "proposed": "Proposed",
}


@dataclass(frozen=True)
class Stage8ExperimentConfig:
    """Stage 8 的 Monte Carlo 与 sweep 参数。"""

    n_mc: int
    scene_seed_start: int
    multi_start_seeds: tuple[int, ...]
    fixed_upa_seed: int
    position_jitter: np.ndarray
    rho_si_values: tuple[float, ...]
    p_dl_max_values: tuple[float, ...]
    cable_length_values: tuple[float, ...]

    def validate(self) -> None:
        if self.n_mc <= 0:
            raise ValueError("n_mc must be positive.")

        if len(self.multi_start_seeds) == 0:
            raise ValueError("multi_start_seeds must not be empty.")

        if len(set(self.multi_start_seeds)) != len(self.multi_start_seeds):
            raise ValueError("multi_start_seeds must be unique.")

        if self.position_jitter.shape != (3,):
            raise ValueError("position_jitter must have shape (3,).")

        if np.any(self.position_jitter < 0.0):
            raise ValueError("position_jitter must be nonnegative.")

        if any(value < 0.0 or value > 1.0 for value in self.rho_si_values):
            raise ValueError("rho_si sweep values must lie in [0, 1].")

        if any(value <= 0.0 for value in self.p_dl_max_values):
            raise ValueError("p_dl_max sweep values must be positive.")

        if any(value <= 0.0 for value in self.cable_length_values):
            raise ValueError("cable_length sweep values must be positive.")


@dataclass(frozen=True)
class SchemeSummary:
    """保存一个 scene 下某个方案的轻量结果。"""

    wsr: float
    best_seed: int
    iterations: int
    runtime_s: float
    n_starts: int


@dataclass(frozen=True)
class SceneSummary:
    """保存一个 Monte Carlo scene 的六方案结果。"""

    scene_seed: int
    q_dl: np.ndarray
    q_ul: np.ndarray
    q_s: np.ndarray
    schemes: dict[str, SchemeSummary]


@dataclass(frozen=True)
class SweepRecord:
    """保存一个参数值、一个 scene 的结果。"""

    parameter_name: str
    parameter_value: float
    scene: SceneSummary


def load_stage8_experiment_config(
    path: str | Path,
) -> Stage8ExperimentConfig:
    """读取 Stage 8 专用 experiment 配置。"""

    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    experiment = raw["experiment"]
    sweep = experiment["sweep"]

    cfg = Stage8ExperimentConfig(
        n_mc=int(experiment["n_mc"]),
        scene_seed_start=int(experiment["scene_seed_start"]),
        multi_start_seeds=tuple(
            int(seed) for seed in experiment["multi_start_seeds"]
        ),
        fixed_upa_seed=int(experiment["fixed_upa_seed"]),
        position_jitter=np.asarray(
            experiment["position_jitter"],
            dtype=float,
        ),
        rho_si_values=tuple(
            float(value) for value in sweep["rho_si"]
        ),
        p_dl_max_values=tuple(
            float(value) for value in sweep["p_dl_max"]
        ),
        cable_length_values=tuple(
            float(value) for value in sweep["cable_length"]
        ),
    )

    cfg.validate()
    return cfg


def make_scene_config(
    base_cfg: SystemConfig,
    scene_seed: int,
    position_jitter: np.ndarray,
) -> SystemConfig:
    """由 scene seed 随机扰动用户与目标位置。"""

    jitter = np.asarray(position_jitter, dtype=float)

    if jitter.shape != (3,) or np.any(jitter < 0.0):
        raise ValueError(
            "position_jitter must be a nonnegative array of shape (3,)."
        )

    rng = np.random.default_rng(int(scene_seed))

    def _perturb(points: np.ndarray) -> np.ndarray:
        delta = rng.uniform(
            low=-jitter,
            high=jitter,
            size=points.shape,
        )
        return np.asarray(points + delta, dtype=float)

    q_dl = _perturb(base_cfg.q_dl)
    q_ul = _perturb(base_cfg.q_ul)
    q_s = _perturb(base_cfg.q_s)

    # 当前场景都位于 z > 0；过大的扰动直接报错，不静默截断。
    if (
        np.any(q_dl[:, 2] <= 0.0)
        or np.any(q_ul[:, 2] <= 0.0)
        or np.any(q_s[:, 2] <= 0.0)
    ):
        raise ValueError(
            "Scene perturbation produced a non-positive z coordinate."
        )

    cfg = replace(
        base_cfg,
        q_dl=q_dl,
        q_ul=q_ul,
        q_s=q_s,
    )
    cfg.validate()
    return cfg


def _summarize_multistart(
    result: MultiStartResult,
) -> SchemeSummary:
    """将 MultiStartResult 转成轻量统计。"""

    best = result.best_start

    return SchemeSummary(
        wsr=float(best.final_wsr),
        best_seed=int(best.seed),
        iterations=int(best.iterations),
        runtime_s=float(result.total_runtime_s),
        n_starts=len(result.starts),
    )


def run_stage8_scene(
    cfg: SystemConfig,
    scene_seed: int,
    multi_start_seeds: Sequence[int],
    fixed_upa_seed: int,
    position_jitter: np.ndarray,
    require_convergence: bool = True,
    verbose: bool = False,
    upa_tx_rx_separation: float | None = None,
) -> SceneSummary:
    """运行一个 Monte Carlo scene 的六方案公平对比。"""

    scene_cfg = make_scene_config(
        cfg,
        scene_seed=scene_seed,
        position_jitter=position_jitter,
    )

    seeds = tuple(int(seed) for seed in multi_start_seeds)

    if len(seeds) == 0:
        raise ValueError("multi_start_seeds must not be empty.")

    if len(set(seeds)) != len(seeds):
        raise ValueError("multi_start_seeds must be unique.")

    if verbose:
        print(f"\nScene seed = {scene_seed}")

    results: dict[str, MultiStartResult] = {}

    # ToMA 方案使用完全相同的 Multi-start seed 预算。
    results["b1"] = run_multi_start(
        name="B1: RCG-ToMA + Traditional Resource Design",
        seeds=seeds,
        run_single=lambda seed: run_rcg_toma_traditional(
            scene_cfg,
            rng=np.random.default_rng(seed),
            verbose=False,
        ),
        iteration_getter=lambda result: result.iterations,
        cfg=scene_cfg,
        require_convergence=require_convergence,
        verbose=False,
    )

    results["b2"] = run_multi_start(
        name="B2: Fixed-ToMA + FP Resource Optimization",
        seeds=seeds,
        run_single=lambda seed: run_fixed_toma_fp(
            scene_cfg,
            rng=np.random.default_rng(seed),
            verbose=False,
        ),
        iteration_getter=lambda result: result.iterations,
        cfg=scene_cfg,
        require_convergence=require_convergence,
        verbose=False,
    )

    results["b4"] = run_multi_start(
        name="B4: Fixed-ToMA + Traditional Resource Design",
        seeds=seeds,
        run_single=lambda seed: run_fixed_toma_traditional(
            scene_cfg,
            rng=np.random.default_rng(seed),
        ),
        iteration_getter=lambda result: result.iterations,
        cfg=scene_cfg,
        require_convergence=require_convergence,
        verbose=False,
    )

    results["proposed"] = run_multi_start(
        name="Proposed: RCG-ToMA + FP Resource Optimization",
        seeds=seeds,
        run_single=lambda seed: run_joint_algorithm(
            scene_cfg,
            rng=np.random.default_rng(seed),
            verbose=False,
        ),
        iteration_getter=lambda result: result.outer_iterations,
        cfg=scene_cfg,
        require_convergence=require_convergence,
        verbose=False,
    )

    # 固定 UPA 已在 Stage 7 验证为 seed-insensitive；Stage 8 每个 scene 只跑一次。
    results["b3"] = run_multi_start(
        name="B3: FPA-UPA + FP Resource Optimization",
        seeds=(int(fixed_upa_seed),),
        run_single=lambda seed: run_fpa_upa_fp(
            scene_cfg,
            rng=np.random.default_rng(seed),
            verbose=False,
            tx_rx_separation=upa_tx_rx_separation,
        ),
        iteration_getter=lambda result: result.iterations,
        cfg=scene_cfg,
        require_convergence=require_convergence,
        verbose=False,
    )

    results["b5"] = run_multi_start(
        name="B5: FPA-UPA + Traditional Resource Design",
        seeds=(int(fixed_upa_seed),),
        run_single=lambda seed: run_fpa_upa_traditional(
            scene_cfg,
            tx_rx_separation=upa_tx_rx_separation,
        ),
        iteration_getter=lambda result: result.iterations,
        cfg=scene_cfg,
        require_convergence=require_convergence,
        verbose=False,
    )

    summaries = {
        key: _summarize_multistart(results[key])
        for key in SCHEME_ORDER
    }

    if verbose:
        values = "  ".join(
            f"{SCHEME_LABELS[key]}={summaries[key].wsr:.6f}"
            for key in SCHEME_ORDER
        )
        print(values)

    return SceneSummary(
        scene_seed=int(scene_seed),
        q_dl=scene_cfg.q_dl.copy(),
        q_ul=scene_cfg.q_ul.copy(),
        q_s=scene_cfg.q_s.copy(),
        schemes=summaries,
    )


def run_monte_carlo(
    cfg: SystemConfig,
    experiment_cfg: Stage8ExperimentConfig,
    n_mc: int | None = None,
    require_convergence: bool = True,
    verbose: bool = False,
    upa_tx_rx_separation: float | None = None,
) -> list[SceneSummary]:
    """运行多个独立 scene 的 Monte Carlo 实验。"""

    number_of_scenes = experiment_cfg.n_mc if n_mc is None else int(n_mc)

    if number_of_scenes <= 0:
        raise ValueError("n_mc must be positive.")

    summaries: list[SceneSummary] = []

    for offset in range(number_of_scenes):
        scene_seed = experiment_cfg.scene_seed_start + offset

        summary = run_stage8_scene(
            cfg,
            scene_seed=scene_seed,
            multi_start_seeds=experiment_cfg.multi_start_seeds,
            fixed_upa_seed=experiment_cfg.fixed_upa_seed,
            position_jitter=experiment_cfg.position_jitter,
            require_convergence=require_convergence,
            verbose=verbose,
            upa_tx_rx_separation=upa_tx_rx_separation,
        )
        summaries.append(summary)

    return summaries


def summarize_monte_carlo(
    scenes: Sequence[SceneSummary],
) -> dict[str, tuple[float, float]]:
    """返回六方案 WSR 的 mean/std。"""

    if len(scenes) == 0:
        raise ValueError("scenes must not be empty.")

    summary: dict[str, tuple[float, float]] = {}

    for key in SCHEME_ORDER:
        values = np.asarray(
            [scene.schemes[key].wsr for scene in scenes],
            dtype=float,
        )
        summary[key] = (
            float(np.mean(values)),
            float(np.std(values, ddof=0)),
        )

    return summary


def _scene_csv_fieldnames() -> list[str]:
    fields = ["scene_seed"]

    for key in SCHEME_ORDER:
        fields.extend(
            [
                f"{key}_wsr",
                f"{key}_best_seed",
                f"{key}_iterations",
                f"{key}_runtime_s",
                f"{key}_n_starts",
            ]
        )

    return fields


def save_monte_carlo_results(
    scenes: Sequence[SceneSummary],
    output_dir: str | Path,
    stem: str = "stage8_monte_carlo",
) -> tuple[Path, Path]:
    """将 scene-level 结果保存为 CSV + NPZ。"""

    if len(scenes) == 0:
        raise ValueError("scenes must not be empty.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{stem}.csv"
    npz_path = output_dir / f"{stem}.npz"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=_scene_csv_fieldnames(),
        )
        writer.writeheader()

        for scene in scenes:
            row: dict[str, float | int] = {
                "scene_seed": scene.scene_seed,
            }

            for key in SCHEME_ORDER:
                item = scene.schemes[key]
                row[f"{key}_wsr"] = item.wsr
                row[f"{key}_best_seed"] = item.best_seed
                row[f"{key}_iterations"] = item.iterations
                row[f"{key}_runtime_s"] = item.runtime_s
                row[f"{key}_n_starts"] = item.n_starts

            writer.writerow(row)

    wsr_matrix = np.asarray(
        [
            [scene.schemes[key].wsr for key in SCHEME_ORDER]
            for scene in scenes
        ],
        dtype=float,
    )

    np.savez_compressed(
        npz_path,
        scene_seeds=np.asarray(
            [scene.scene_seed for scene in scenes],
            dtype=int,
        ),
        scheme_order=np.asarray(SCHEME_ORDER),
        q_dl=np.stack([scene.q_dl for scene in scenes]),
        q_ul=np.stack([scene.q_ul for scene in scenes]),
        q_s=np.stack([scene.q_s for scene in scenes]),
        wsr=wsr_matrix,
    )

    return csv_path, npz_path


def _replace_scalar_parameter(
    cfg: SystemConfig,
    parameter_name: str,
    value: float,
) -> SystemConfig:
    """替换 Stage 8 第一版支持的标量扫描参数。"""

    value = float(value)

    if parameter_name == "rho_si":
        new_cfg = replace(cfg, rho_si=value)
    elif parameter_name == "p_dl_max":
        new_cfg = replace(cfg, p_dl_max=value)
    elif parameter_name == "cable_length":
        new_cfg = replace(cfg, cable_length=value)
    else:
        raise ValueError(
            "parameter_name must be one of: "
            "rho_si, p_dl_max, cable_length."
        )

    new_cfg.validate()
    return new_cfg


def run_parameter_sweep(
    cfg: SystemConfig,
    experiment_cfg: Stage8ExperimentConfig,
    parameter_name: str,
    values: Sequence[float],
    n_mc: int | None = None,
    require_convergence: bool = True,
    verbose: bool = False,
) -> list[SweepRecord]:
    """对一个标量参数执行 Monte Carlo sweep。"""

    if len(values) == 0:
        raise ValueError("values must not be empty.")

    records: list[SweepRecord] = []

    # cable-length sweep 时固定 UPA 的 Tx/Rx 中心间距，
    # 避免把 UPA 几何变化混进 ToMA cable-length 效应。
    fixed_upa_separation = (
        cfg.cable_length
        if parameter_name == "cable_length"
        else None
    )

    for value in values:
        if verbose:
            print("\n" + "=" * 80)
            print(f"{parameter_name} = {float(value):.6g}")
            print("=" * 80)

        sweep_cfg = _replace_scalar_parameter(
            cfg,
            parameter_name=parameter_name,
            value=float(value),
        )

        scenes = run_monte_carlo(
            sweep_cfg,
            experiment_cfg,
            n_mc=n_mc,
            require_convergence=require_convergence,
            verbose=verbose,
            upa_tx_rx_separation=fixed_upa_separation,
        )

        records.extend(
            SweepRecord(
                parameter_name=parameter_name,
                parameter_value=float(value),
                scene=scene,
            )
            for scene in scenes
        )

    return records


def save_sweep_results(
    records: Sequence[SweepRecord],
    output_dir: str | Path,
    stem: str,
) -> tuple[Path, Path]:
    """保存 parameter sweep 的 CSV + NPZ。"""

    if len(records) == 0:
        raise ValueError("records must not be empty.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{stem}.csv"
    npz_path = output_dir / f"{stem}.npz"

    fieldnames = [
        "parameter_name",
        "parameter_value",
        "scene_seed",
        *[f"{key}_wsr" for key in SCHEME_ORDER],
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            row: dict[str, str | float | int] = {
                "parameter_name": record.parameter_name,
                "parameter_value": record.parameter_value,
                "scene_seed": record.scene.scene_seed,
            }

            for key in SCHEME_ORDER:
                row[f"{key}_wsr"] = record.scene.schemes[key].wsr

            writer.writerow(row)

    np.savez_compressed(
        npz_path,
        parameter_name=np.asarray(records[0].parameter_name),
        parameter_value=np.asarray(
            [record.parameter_value for record in records],
            dtype=float,
        ),
        scene_seed=np.asarray(
            [record.scene.scene_seed for record in records],
            dtype=int,
        ),
        scheme_order=np.asarray(SCHEME_ORDER),
        wsr=np.asarray(
            [
                [record.scene.schemes[key].wsr for key in SCHEME_ORDER]
                for record in records
            ],
            dtype=float,
        ),
    )

    return csv_path, npz_path
