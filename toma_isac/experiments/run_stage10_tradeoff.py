"""Stage 10：Communication-Sensing trade-off 权重扫描。"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm import run_joint_algorithm
from baselines import run_fixed_toma_fp, run_fpa_upa_fp
from config import SystemConfig, load_config
from ma_baseline import run_ma_fp
from multistart import MultiStartResult, run_multi_start
from stage8 import (
    load_stage8_experiment_config,
    make_ma_start_seeds,
    make_scene_config,
)

CONFIG_PATH = ROOT / "configs" / "stage10_tradeoff.yaml"
RESULT_DIR = ROOT / "results" / "stage10"
CSV_PATH = RESULT_DIR / "stage10_tradeoff.csv"
NPZ_PATH = RESULT_DIR / "stage10_tradeoff.npz"

SCHEME_ORDER = ("b2", "b3", "b6", "proposed")
SCHEME_LABELS = {
    "b2": "Fixed-ToMA + FP",
    "b3": "FPA-UPA + FP",
    "b6": "MA + FP",
    "proposed": "FP-aware RCG-ToMA + FP (Proposed)",
}


@dataclass(frozen=True)
class TradeoffConfig:
    scene_seed: int
    omega_comm: tuple[float, ...]

    def validate(self) -> None:
        if len(self.omega_comm) == 0:
            raise ValueError("omega_comm must not be empty.")
        if len(set(self.omega_comm)) != len(self.omega_comm):
            raise ValueError("omega_comm values must be unique.")
        if any(not (0.0 <= value <= 1.0) for value in self.omega_comm):
            raise ValueError("Each omega_comm must lie in [0, 1].")

@dataclass(frozen=True)
class TradeoffRecord:
    omega_comm: float
    omega_sensing: float
    scheme: str
    scheme_label: str
    sum_rate_dl: float
    sum_rate_ul: float
    sum_rate_comm: float
    sum_rate_sensing: float
    weighted_sum_rate: float
    best_seed: int
    iterations: int
    runtime_s: float
    n_starts: int
    converged: bool


def load_tradeoff_config(path: str | Path) -> TradeoffConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    tradeoff = raw["tradeoff"]
    cfg = TradeoffConfig(
        scene_seed=int(tradeoff["scene_seed"]),
        omega_comm=tuple(float(x) for x in tradeoff["omega_comm"]),
    )
    cfg.validate()
    return cfg


def make_group_weights(
    cfg: SystemConfig,
    omega_comm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """通信总权重平均分给 DL/UL 两组，组内用户等权。"""

    omega_comm = float(omega_comm)
    if not 0.0 <= omega_comm <= 1.0:
        raise ValueError("omega_comm must lie in [0, 1].")

    omega_sensing = 1.0 - omega_comm

    weight_dl = np.full(
        cfg.k_dl,
        omega_comm / (2.0 * cfg.k_dl),
        dtype=float,
    )
    weight_ul = np.full(
        cfg.j_ul,
        omega_comm / (2.0 * cfg.j_ul),
        dtype=float,
    )
    weight_s = np.full(
        cfg.l_target,
        omega_sensing / cfg.l_target,
        dtype=float,
    )

    total = float(weight_dl.sum() + weight_ul.sum() + weight_s.sum())
    if not np.isclose(total, 1.0, atol=1.0e-12):
        raise RuntimeError(f"Trade-off weights sum to {total}, expected 1.")

    return weight_dl, weight_ul, weight_s


def _run_multistart(
    name: str,
    seeds: tuple[int, ...],
    runner: Callable[[int], object],
    iteration_getter: Callable[[object], int],
    cfg: SystemConfig,
    require_convergence: bool,
) -> MultiStartResult:
    return run_multi_start(
        name=name,
        seeds=seeds,
        run_single=runner,
        iteration_getter=iteration_getter,
        cfg=cfg,
        require_convergence=require_convergence,
        verbose=False,
    )


def run_one_scheme(
    scheme: str,
    cfg: SystemConfig,
    experiment_cfg,
    scene_seed: int,
    require_convergence: bool,
    formal_multistart: bool,
) -> MultiStartResult:
    """运行一个 FP 类方案，保持 Stage 8 的 baseline 定义不变。"""

    if formal_multistart:
        toma_seeds = tuple(experiment_cfg.multi_start_seeds)
        ma_n_starts = int(experiment_cfg.ma_n_starts)
    else:
        toma_seeds = (int(experiment_cfg.multi_start_seeds[0]),)
        ma_n_starts = 1

    if scheme == "b2":
        return _run_multistart(
            name="B2: Fixed-ToMA + FP",
            seeds=toma_seeds,
            runner=lambda seed: run_fixed_toma_fp(
                cfg,
                rng=np.random.default_rng(seed),
                verbose=False,
            ),
            iteration_getter=lambda result: result.iterations,
            cfg=cfg,
            require_convergence=require_convergence,
        )

    if scheme == "b3":
        seed = int(experiment_cfg.fixed_upa_seed)
        return _run_multistart(
            name="B3: FPA-UPA + FP",
            seeds=(seed,),
            runner=lambda value: run_fpa_upa_fp(
                cfg,
                rng=np.random.default_rng(value),
                verbose=False,
            ),
            iteration_getter=lambda result: result.iterations,
            cfg=cfg,
            require_convergence=require_convergence,
        )

    if scheme == "b6":
        ma_seeds = make_ma_start_seeds(
            scene_seed=scene_seed,
            fixed_upa_seed=int(experiment_cfg.fixed_upa_seed),
            n_starts=ma_n_starts,
        )
        fixed_seed = int(experiment_cfg.fixed_upa_seed)
        return _run_multistart(
            name="B6: MA + FP",
            seeds=ma_seeds,
            runner=lambda seed: run_ma_fp(
                cfg,
                rng=np.random.default_rng(seed),
                verbose=False,
                region_size=experiment_cfg.ma_region_size,
                min_spacing=experiment_cfg.ma_min_spacing,
                initial_step=experiment_cfg.ma_initial_step,
                randomize_initial_geometry=(int(seed) != fixed_seed),
                initial_jitter_fraction=(
                    experiment_cfg.ma_initial_jitter_fraction
                ),
            ),
            iteration_getter=lambda result: result.iterations,
            cfg=cfg,
            require_convergence=require_convergence,
        )

    if scheme == "proposed":
        return _run_multistart(
            name="Proposed: FP-aware RCG-ToMA + FP",
            seeds=toma_seeds,
            runner=lambda seed: run_joint_algorithm(
                cfg,
                rng=np.random.default_rng(seed),
                verbose=False,
                position_update_mode="fp_aware",
                fp_aware_candidate_iters=2,
            ),
            iteration_getter=lambda result: result.outer_iterations,
            cfg=cfg,
            require_convergence=require_convergence,
        )

    raise ValueError(f"Unknown scheme: {scheme}")


def summarize_result(
    omega_comm: float,
    scheme: str,
    result: MultiStartResult,
) -> TradeoffRecord:
    best = result.best_start
    performance = result.best_result.performance

    sum_rate_dl = float(np.sum(performance.rate_dl))
    sum_rate_ul = float(np.sum(performance.rate_ul))
    sum_rate_comm = sum_rate_dl + sum_rate_ul
    sum_rate_sensing = float(np.sum(performance.rate_s))

    return TradeoffRecord(
        omega_comm=float(omega_comm),
        omega_sensing=float(1.0 - omega_comm),
        scheme=scheme,
        scheme_label=SCHEME_LABELS[scheme],
        sum_rate_dl=sum_rate_dl,
        sum_rate_ul=sum_rate_ul,
        sum_rate_comm=sum_rate_comm,
        sum_rate_sensing=sum_rate_sensing,
        weighted_sum_rate=float(performance.weighted_sum_rate),
        best_seed=int(best.seed),
        iterations=int(best.iterations),
        runtime_s=float(result.total_runtime_s),
        n_starts=len(result.starts),
        converged=bool(best.converged),
    )


def save_results(records: list[TradeoffRecord]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = list(TradeoffRecord.__dataclass_fields__)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)

    np.savez(
        NPZ_PATH,
        omega_comm=np.asarray([r.omega_comm for r in records]),
        omega_sensing=np.asarray([r.omega_sensing for r in records]),
        scheme=np.asarray([r.scheme for r in records]),
        sum_rate_comm=np.asarray([r.sum_rate_comm for r in records]),
        sum_rate_sensing=np.asarray([r.sum_rate_sensing for r in records]),
        weighted_sum_rate=np.asarray([r.weighted_sum_rate for r in records]),
    )

    print(f"\nCSV saved to: {CSV_PATH}")
    print(f"NPZ saved to: {NPZ_PATH}")


def print_monotonicity(records: list[TradeoffRecord]) -> None:
    """只做诊断，不把非单调视为程序错误。"""

    print("\nTrade-off trend diagnostic:")
    for scheme in SCHEME_ORDER:
        rows = sorted(
            (r for r in records if r.scheme == scheme),
            key=lambda r: r.omega_comm,
        )
        comm = np.asarray([r.sum_rate_comm for r in rows])
        sensing = np.asarray([r.sum_rate_sensing for r in rows])

        comm_nondec = bool(np.all(np.diff(comm) >= -1.0e-8))
        sensing_noninc = bool(np.all(np.diff(sensing) <= 1.0e-8))

        print(
            f"  {SCHEME_LABELS[scheme]}: "
            f"comm nondecreasing={comm_nondec}, "
            f"sensing nonincreasing={sensing_noninc}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a short 3-weight, single-start interface check.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check configuration and print planned settings only.",
    )
    args = parser.parse_args()

    base_cfg = load_config(CONFIG_PATH)
    experiment_cfg = load_stage8_experiment_config(CONFIG_PATH)
    tradeoff_cfg = load_tradeoff_config(CONFIG_PATH)

    if (base_cfg.k_dl, base_cfg.j_ul, base_cfg.l_target, base_cfg.r_sensing) != (
        5,
        5,
        5,
        4,
    ):
        raise RuntimeError("Stage 10 formal setting must use K=J=L=5, R_s=4.")

    if experiment_cfg.fading_enabled:
        raise RuntimeError("Stage 10 formal setting must use fading.enabled=false.")

    # 整个权重扫描固定同一个物理 scene。
    physical_scene_cfg = make_scene_config(
        base_cfg,
        scene_seed=tradeoff_cfg.scene_seed,
        position_jitter=experiment_cfg.position_jitter,
        randomize_fading=experiment_cfg.fading_enabled,
        fading_model=experiment_cfg.fading_model,
    )

    if args.smoke:
        omega_values = (0.1, 0.5, 0.9)
        working_scene_cfg = replace(
            physical_scene_cfg,
            max_inner_iter=min(5, physical_scene_cfg.max_inner_iter),
            max_outer_iter=min(3, physical_scene_cfg.max_outer_iter),
        )
        require_convergence = False
        formal_multistart = False
    else:
        omega_values = tradeoff_cfg.omega_comm
        working_scene_cfg = physical_scene_cfg
        require_convergence = True
        formal_multistart = True

    print("Stage 10 Communication-Sensing Trade-off")
    print(
        f"  scene={tradeoff_cfg.scene_seed}, "
        f"K=J=L=5, R_s=4"
    )
    print(f"  omega_C={omega_values}")
    print("  omega_S = 1 - omega_C")
    print("  DL group weight = omega_C / 2")
    print("  UL group weight = omega_C / 2")
    print("  fading = OFF")
    print("  schemes = B2, B3, B6, Proposed")

    if args.dry_run:
        for omega_comm in omega_values:
            wd, wu, ws = make_group_weights(working_scene_cfg, omega_comm)
            print(
                f"  omega_C={omega_comm:.2f}: "
                f"sum(DL)={wd.sum():.3f}, "
                f"sum(UL)={wu.sum():.3f}, "
                f"sum(S)={ws.sum():.3f}"
            )
        print("Dry run passed. No optimization was executed.")
        return

    records: list[TradeoffRecord] = []
    total_start = perf_counter()

    for omega_comm in omega_values:
        weight_dl, weight_ul, weight_s = make_group_weights(
            working_scene_cfg,
            omega_comm,
        )

        cfg = replace(
            working_scene_cfg,
            weight_dl=weight_dl,
            weight_ul=weight_ul,
            weight_s=weight_s,
        )
        cfg.validate()

        print("\n" + "=" * 78)
        print(
            f"omega_C={omega_comm:.2f}, "
            f"omega_S={1.0 - omega_comm:.2f}"
        )
        print("=" * 78)

        for scheme in SCHEME_ORDER:
            result = run_one_scheme(
                scheme=scheme,
                cfg=cfg,
                experiment_cfg=experiment_cfg,
                scene_seed=tradeoff_cfg.scene_seed,
                require_convergence=require_convergence,
                formal_multistart=formal_multistart,
            )
            record = summarize_result(
                omega_comm=omega_comm,
                scheme=scheme,
                result=result,
            )
            records.append(record)

            print(
                f"{SCHEME_LABELS[scheme]}: "
                f"R_C={record.sum_rate_comm:.6f}, "
                f"R_S={record.sum_rate_sensing:.6f}, "
                f"WSR={record.weighted_sum_rate:.6f}, "
                f"best_seed={record.best_seed}"
            )

    save_results(records)
    print_monotonicity(records)

    elapsed = perf_counter() - total_start
    print(f"\nStage 10 completed in {elapsed:.2f} s.")


if __name__ == "__main__":
    main()
