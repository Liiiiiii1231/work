"""Stage 10：20-point independent cold multi-start (no cross-weight warm starts)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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

CONFIG_PATH = ROOT / "configs" / "stage10_tradeoff_cold20.yaml"
RESULT_DIR = ROOT / "results" / "stage10" / "cold_wide20_dl114_tol1e4"
CSV_PATH = RESULT_DIR / "stage10_tradeoff.csv"
NPZ_PATH = RESULT_DIR / "stage10_tradeoff.npz"
FAILURES_CSV_PATH = RESULT_DIR / "stage10_failures.csv"

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
    """运行 multi-start；正式模式下仅在已收敛起点中选择最佳解。

    某一个随机起点未收敛时不立即终止整个 Stage 10。
    若 require_convergence=True，则：
    1. 所有起点都实际运行；
    2. 未收敛起点不参与最佳解选择；
    3. 只有当所有起点均未收敛时才报错。
    """

    # 先允许所有起点完整执行，避免单个起点未收敛就中断整个权重扫描。
    result = run_multi_start(
        name=name,
        seeds=seeds,
        run_single=runner,
        iteration_getter=iteration_getter,
        cfg=cfg,
        require_convergence=False,
        verbose=False,
    )

    if not require_convergence:
        return result

    converged_indices = [
        index
        for index, start in enumerate(result.starts)
        if start.converged
    ]

    if len(converged_indices) == 0:
        details = ", ".join(
            f"seed={start.seed}, iterations={start.iterations}, "
            f"stop_reason={getattr(start.result, 'stop_reason', 'unknown')}"
            for start in result.starts
        )
        raise RuntimeError(
            f"{name}: none of the {len(result.starts)} multi-start runs "
            f"converged. {details}"
        )

    # 只在已收敛的起点中按最终真实 WSR 选择最好的一组。
    best_index = max(
        converged_indices,
        key=lambda index: result.starts[index].final_wsr,
    )

    nonconverged = [
        start
        for start in result.starts
        if not start.converged
    ]

    if nonconverged:
        skipped = ", ".join(
            f"seed={start.seed}"
            f"(iter={start.iterations}, "
            f"reason={getattr(start.result, 'stop_reason', 'unknown')})"
            for start in nonconverged
        )
        print(
            f"  [WARN] {name}: ignoring non-converged start(s): {skipped}"
        )

    return MultiStartResult(
        name=result.name,
        starts=result.starts,
        best_index=best_index,
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

        if len(rows) < 2:
            print(f"  {SCHEME_LABELS[scheme]}: too few converged points to assess trend")
            continue
        comm_nondec = bool(np.all(np.diff(comm) >= -1.0e-8))
        sensing_noninc = bool(np.all(np.diff(sensing) <= 1.0e-8))

        print(
            f"  {SCHEME_LABELS[scheme]}: "
            f"comm nondecreasing={comm_nondec}, "
            f"sensing nonincreasing={sensing_noninc}"
        )



# Backward-compatible reuse: saved 14-point CSV contains converged rates, not optimization state.
# This option NEVER passes states across weights, so it is not warm-starting.
OLD_14_CONFIG_PATH = ROOT / "configs" / "stage10_tradeoff.yaml"
OLD_14_CSV_PATH = ROOT / "results" / "stage10" / "cold_wide14_tol1e4" / "stage10_tradeoff.csv"
MANIFEST_PATH = RESULT_DIR / "cold20_run_metadata.json"
EXPECTED_14 = (0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def _read_rows(path: Path) -> list[TradeoffRecord]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing saved results: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.DictReader(handle))
    items = []
    for r in raw:
        if not (r.get("converged", "").strip().lower() == "true"):
            raise ValueError(f"Cannot reuse an unconverged row in {path}: {r}")
        items.append(TradeoffRecord(
            omega_comm=float(r["omega_comm"]),
            omega_sensing=float(r["omega_sensing"]),
            scheme=r["scheme"],
            scheme_label=r["scheme_label"],
            sum_rate_dl=float(r["sum_rate_dl"]),
            sum_rate_ul=float(r["sum_rate_ul"]),
            sum_rate_comm=float(r["sum_rate_comm"]),
            sum_rate_sensing=float(r["sum_rate_sensing"]),
            weighted_sum_rate=float(r["weighted_sum_rate"]),
            best_seed=int(r["best_seed"]),
            iterations=int(r["iterations"]),
            runtime_s=float(r["runtime_s"]),
            n_starts=int(r["n_starts"]),
            converged=True,
        ))
    return items


def _key(r: TradeoffRecord) -> tuple[float, str]:
    return round(r.omega_comm, 12), r.scheme


def _validate_rows(rows: list[TradeoffRecord], allowed_weights: tuple[float, ...], *, exact=False) -> None:
    allowed = {round(w, 12) for w in allowed_weights}
    keys = set()
    for r in rows:
        if round(r.omega_comm, 12) not in allowed or r.scheme not in SCHEME_ORDER:
            raise ValueError(f"Unexpected result row: {r.omega_comm} {r.scheme}")
        if _key(r) in keys:
            raise ValueError(f"Duplicate result row: {_key(r)}")
        keys.add(_key(r))
        wd = r.omega_comm / 10.0  # K=J=L=5
        ws = (1.0-r.omega_comm) / 5.0
        expected = wd * r.sum_rate_comm + ws * r.sum_rate_sensing
        if abs(expected - r.weighted_sum_rate) > 1e-6 * max(1., abs(expected)):
            raise ValueError(f"WSR inconsistent at {_key(r)}: {r.weighted_sum_rate} != {expected}")
    if exact and len(keys) != len(allowed_weights) * len(SCHEME_ORDER):
        raise ValueError(f"Expected {len(allowed_weights)*len(SCHEME_ORDER)} rows, got {len(keys)}")


def _physical_cfg_dict(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        obj = yaml.safe_load(handle)
    obj = dict(obj)
    obj["tradeoff"] = {"scene_seed": int(obj["tradeoff"]["scene_seed"])}
    return obj


def _ensure_14_reusable() -> list[TradeoffRecord]:
    if not OLD_14_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Cannot verify old configuration: {OLD_14_CONFIG_PATH}")
    if _physical_cfg_dict(CONFIG_PATH) != _physical_cfg_dict(OLD_14_CONFIG_PATH):
        raise ValueError(
            "Old 14-point YAML and new cold20 YAML have different physical/algorithm settings. "
            "Do NOT reuse old results; rerun all 20 points without --reuse-14."
        )
    original = _read_rows(OLD_14_CSV_PATH)
    _validate_rows(original, EXPECTED_14, exact=True)
    if {round(r.omega_comm, 12) for r in original} != {round(w,12) for w in EXPECTED_14}:
        raise ValueError("14-point source CSV does not match expected omega list")
    return original


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_fingerprint(reuse14: bool) -> dict:
    info = {"new_config_sha256": _sha256_file(CONFIG_PATH), "reuse_14": bool(reuse14)}
    if reuse14:
        info["old_config_sha256"] = _sha256_file(OLD_14_CONFIG_PATH)
        info["old_csv_sha256"] = _sha256_file(OLD_14_CSV_PATH)
    return info


def _check_manifest(fingerprint: dict, resume: bool) -> None:
    if MANIFEST_PATH.exists():
        previous = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if previous != fingerprint:
            raise RuntimeError(
                "Output metadata does not match this configuration/input CSV. "
                "Do not mix results from different experiments."
            )
        if not resume:
            raise FileExistsError(
                f"Output already exists at {RESULT_DIR}. Use --resume to continue."
            )
    else:
        if CSV_PATH.exists() or FAILURES_CSV_PATH.exists():
            raise RuntimeError(
                "Found old output without run metadata. Move/backup that folder before running."
            )
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")


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
    parser.add_argument(
        "--reuse-14", action="store_true",
        help="Preserve the 14 converged pure-cold rows and solve only six added weights.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume this experiment's checkpoint without recalculating finished rows.",
    )
    args = parser.parse_args()
    if args.smoke and (args.resume or args.reuse_14):
        parser.error("--smoke cannot be combined with --resume/--reuse-14")

    # The smoke test must not overwrite formal results.
    if args.smoke:
        global RESULT_DIR, CSV_PATH, NPZ_PATH, FAILURES_CSV_PATH
        RESULT_DIR = ROOT / "results" / "stage10" / "cold_wide20_dl114_tol1e4_smoke"
        CSV_PATH = RESULT_DIR / "stage10_tradeoff.csv"
        NPZ_PATH = RESULT_DIR / "stage10_tradeoff.npz"
        FAILURES_CSV_PATH = RESULT_DIR / "stage10_failures.csv"
        global MANIFEST_PATH
        MANIFEST_PATH = RESULT_DIR / "cold20_run_metadata.json"

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
    print("  initialization = cold multi-start only (no warm start)")

    if args.dry_run:
        for omega_comm in omega_values:
            wd, wu, ws = make_group_weights(working_scene_cfg, omega_comm)
            print(
                f"  omega_C={omega_comm:.10g}: "
                f"sum(DL)={wd.sum():.3f}, "
                f"sum(UL)={wu.sum():.3f}, "
                f"sum(S)={ws.sum():.3f}"
            )
        if args.reuse_14:
            original = _ensure_14_reusable()
            missing = sorted(set(round(x,12) for x in omega_values) - set(round(x.omega_comm,12) for x in original))
            print(f"  reuse-14: verified {len(original)} converged rows (14 x 4)")
            print(f"  only NEW omega values will be computed: {missing}")
        else:
            print("  full-cold20: every weight and each scheme will be solved independently")
        print(f"  output folder: {RESULT_DIR}")
        print("Dry run passed. No optimization was executed.")
        return

    records: list[TradeoffRecord] = []
    failures: list[dict[str, object]] = []
    total_start = perf_counter()
    if not args.smoke:
        if args.reuse_14:
            verified_old = _ensure_14_reusable()
        else:
            verified_old = []
        fp = _run_fingerprint(args.reuse_14)
        _check_manifest(fp, args.resume)
        if args.resume and CSV_PATH.exists():
            records = _read_rows(CSV_PATH)
            _validate_rows(records, tradeoff_cfg.omega_comm)
            if FAILURES_CSV_PATH.exists():
                with FAILURES_CSV_PATH.open(encoding="utf-8-sig", newline="") as file:
                    failures = list(csv.DictReader(file))
            print(f"  [RESUME] loaded {len(records)} finished rows and {len(failures)} failures")
        elif verified_old:
            records = verified_old.copy()
            print(f"  [REUSE] 56 existing 14-point cold-only rows loaded")
    done = {_key(r) for r in records}
    failed = {(round(float(r["omega_comm"]),12), r["scheme"]) for r in failures}

    def checkpoint() -> None:
        # Only converged (or explicitly smoke-test) results enter the plot data.
        records.sort(key=lambda r: (r.omega_comm, SCHEME_ORDER.index(r.scheme)))
        save_results(records)
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        with FAILURES_CSV_PATH.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=("omega_comm", "scheme", "scheme_label", "runtime_s", "reason"),
            )
            writer.writeheader()
            writer.writerows(failures)

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
            f"omega_C={omega_comm:.10g}, "
            f"omega_S={1.0 - omega_comm:.10g}"
        )
        print("=" * 78)

        for scheme in SCHEME_ORDER:
            current_key = (round(float(omega_comm),12), scheme)
            if current_key in done:
                print(f"  [REUSE/SKIP] {scheme} at omega_C={omega_comm:g} (already solved)")
                continue
            if current_key in failed:
                print(f"  [SKIP FAILED] {scheme} at omega_C={omega_comm:g} (previous non-convergence)")
                continue
            scheme_start = perf_counter()
            try:
                result = run_one_scheme(
                    scheme=scheme,
                    cfg=cfg,
                    experiment_cfg=experiment_cfg,
                    scene_seed=tradeoff_cfg.scene_seed,
                    require_convergence=require_convergence,
                    formal_multistart=formal_multistart,
                )
            except RuntimeError as exc:
                # Expected non-convergence is a failed measurement, not a valid WSR point.
                # A different RuntimeError is a real code error and must not be hidden.
                if "multi-start runs converged" not in str(exc):
                    checkpoint()
                    raise
                failures.append({
                    "omega_comm": float(omega_comm),
                    "scheme": scheme,
                    "scheme_label": SCHEME_LABELS[scheme],
                    "runtime_s": round(perf_counter() - scheme_start, 3),
                    "reason": str(exc),
                })
                print(f"[NOT CONVERGED, SKIPPED] {SCHEME_LABELS[scheme]}: {exc}")
                checkpoint()
                continue

            record = summarize_result(
                omega_comm=omega_comm,
                scheme=scheme,
                result=result,
            )
            # During formal runs, this should always be converged due to the
            # multi-start filter. Keep the condition as a second safety guard.
            if require_convergence and not record.converged:
                failures.append({
                    "omega_comm": float(omega_comm),
                    "scheme": scheme,
                    "scheme_label": SCHEME_LABELS[scheme],
                    "runtime_s": round(perf_counter() - scheme_start, 3),
                    "reason": "selected record was not converged",
                })
                print(f"[NOT CONVERGED, SKIPPED] {SCHEME_LABELS[scheme]}")
                checkpoint()
                continue
            records.append(record)
            done.add(current_key)
            checkpoint()

            print(
                f"{SCHEME_LABELS[scheme]}: "
                f"R_C={record.sum_rate_comm:.6f}, "
                f"R_S={record.sum_rate_sensing:.6f}, "
                f"WSR={record.weighted_sum_rate:.6f}, "
                f"best_seed={record.best_seed}"
            )

    checkpoint()
    print(f"\nCompleted: {len(records)} converged points, {len(failures)} non-converged cases.")
    print(f"Non-convergence details: {FAILURES_CSV_PATH}")
    if records:
        print_monotonicity(records)
    else:
        print("WARNING: No valid points. Do not use a prior Stage 10 figure as a new result.")

    elapsed = perf_counter() - total_start
    print(f"\nStage 10 completed in {elapsed:.2f} s.")


if __name__ == "__main__":
    main()
