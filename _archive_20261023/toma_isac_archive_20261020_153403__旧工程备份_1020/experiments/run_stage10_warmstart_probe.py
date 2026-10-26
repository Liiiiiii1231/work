r"""Continue the omega_C=0.42, seed=2042 final state at omega_C=0.40.

Place this file in toma_isac/experiments/.
Run: python -u .\experiments\run_stage10_warmstart_probe.py
Only creates a new diagnostic result directory; no formal files are changed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from algorithm import run_joint_algorithm
from channels import build_channels
from config import load_config
from geometry import build_geometry, check_geometry_constraints
from metrics import compute_performance
from stage8 import load_stage8_experiment_config, make_scene_config
from state import ResourceState
from utils import check_resource_constraints, relative_change
from experiments.run_stage10_cold20 import make_group_weights

CONFIG = ROOT / "configs/stage10_tradeoff_cold20.yaml"
PROBE_ROOT = ROOT / "results/stage10/multistart20_probe"
SOURCE_WEIGHT, TARGET_WEIGHT, SOURCE_SEED = 0.42, 0.40, 2042
STATE_NAME = "omega_0.42_seed_2042.npz"


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def select_source(explicit):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Source NPZ not found: {path}")
        return path
    # Use the latest completed non-smoke probe containing the requested state.
    for path in sorted(PROBE_ROOT.glob(f"*/{STATE_NAME}"), reverse=True):
        meta_path, summary_path = path.parent / "metadata.json", path.parent / "summary.csv"
        if not meta_path.exists() or not summary_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("smoke", True):
            continue
        if any(abs(float(row["omega_comm"]) - SOURCE_WEIGHT) < 1e-12
               and row.get("best20_seed") == str(SOURCE_SEED)
               for row in read_rows(summary_path)):
            return path
    raise FileNotFoundError(
        "Cannot find completed 20-start probe with best seed=2042 at omega=0.42. "
        "Keep the full probe output folder, including NPZ and metadata.json, "
        "or specify --source with the absolute NPZ path."
    )


def rates(performance):
    return (float(np.sum(performance.rate_dl) + np.sum(performance.rate_ul)),
            float(np.sum(performance.rate_s)))


def save_state(path, geometry, resources, **extra):
    np.savez_compressed(
        path, endpoints=geometry.endpoints, tx_positions=geometry.tx_positions,
        rx_positions=geometry.rx_positions, q_matrix=resources.q_matrix,
        q_ul=resources.q_ul, b_ul=resources.b_ul, u_s=resources.u_s, **extra,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="Optional source NPZ path; automatic discovery by default.")
    parser.add_argument("--dry-run", action="store_true", help="Read/check/evaluate only; no solving or writing.")
    parser.add_argument("--smoke", action="store_true", help="One outer iteration only; not a formal result.")
    args = parser.parse_args()
    if args.dry_run and args.smoke:
        parser.error("Choose only one of --dry-run and --smoke.")
    source = select_source(args.source)
    metadata_path = source.parent / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("Source folder must contain the original metadata.json.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("smoke", True):
        raise ValueError("Cannot use a smoke-test state.")
    config_sha = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    if config_sha != metadata.get("config_sha256"):
        raise ValueError("Config changed since the 20-start probe; restore it before this controlled experiment.")
    for name, expected in metadata.get("code_sha256", {}).items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Solver code changed since the source run: {name}")

    base = load_config(CONFIG)
    if not np.isclose(base.tol_inner, 1e-4) or not np.isclose(base.tol_outer, 1e-4):
        raise ValueError("This experiment must retain both tolerances at 1e-4.")
    exp = load_stage8_experiment_config(CONFIG)
    if exp.fading_enabled:
        raise ValueError("Expected fading disabled.")
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    scene = make_scene_config(
        base, scene_seed=int(raw["tradeoff"]["scene_seed"]),
        position_jitter=exp.position_jitter, randomize_fading=False,
        fading_model=exp.fading_model,
    )
    wd, wu, ws = make_group_weights(scene, TARGET_WEIGHT)
    cfg = replace(scene, weight_dl=wd, weight_ul=wu, weight_s=ws)
    cfg.validate()
    with np.load(source, allow_pickle=False) as data:
        if not np.isclose(float(data["omega_comm"]), SOURCE_WEIGHT):
            raise ValueError("Expected source omega=0.42.")
        if int(data["seed"]) != SOURCE_SEED or not bool(data["converged"]):
            raise ValueError("Expected converged source seed=2042.")
        endpoints = data["endpoints"].copy()
        resources = ResourceState(**{
            name: data[name].copy() for name in ["q_matrix", "q_ul", "b_ul", "u_s"]
        })
    geometry = build_geometry(endpoints, cfg)
    check_resource_constraints(resources, cfg)
    channels = build_channels(geometry, cfg)
    before = compute_performance(channels, resources, cfg)
    rc0, rs0 = rates(before)
    initial_wsr = float(before.weighted_sum_rate)
    source_summary = read_rows(source.parent / "summary.csv")
    source_row = next(r for r in source_summary if abs(float(r["omega_comm"])-SOURCE_WEIGHT) < 1e-12)
    if source_row["best20_seed"] != str(SOURCE_SEED):
        raise ValueError("Source is not the selected best candidate in summary.csv.")
    if not (np.isclose(rc0, float(source_row["best20_rate_comm"]), rtol=1e-9, atol=1e-9)
            and np.isclose(rs0, float(source_row["best20_rate_sensing"]), rtol=1e-9, atol=1e-9)):
        raise ValueError("NPZ rates do not reproduce the saved summary; check source/config.")
    target_row = next(r for r in source_summary if abs(float(r["omega_comm"])-TARGET_WEIGHT) < 1e-12)
    cold_wsr = float(target_row["best20_wsr"])
    print(f"Source: {source}", flush=True)
    print(f"At omega=0.40: initial WSR={initial_wsr:.12f}, R_C={rc0:.8f}, R_S={rs0:.8f}", flush=True)
    print(f"Cold 20-start best={cold_wsr:.12f}; starting advantage={initial_wsr-cold_wsr:.12f}", flush=True)
    print(f"Tolerances unchanged: inner={cfg.tol_inner:g}, outer={cfg.tol_outer:g}", flush=True)
    if args.dry_run:
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = ROOT / "results/stage10/warmstart_probe" / f"{stamp}_{'smoke' if args.smoke else 'tol1e4'}"
    out.mkdir(parents=True, exist_ok=False)
    run_meta = dict(source_npz=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    source_weight=SOURCE_WEIGHT, target_weight=TARGET_WEIGHT, source_seed=SOURCE_SEED,
                    config=raw, config_sha256=config_sha, code_sha256=metadata.get("code_sha256", {}),
                    position_update_mode="fp_aware", candidate_fp_iters=2,
                    inherit_positions=True, inherit_resources=True,
                    reset_rcg_history=True, smoke=args.smoke)
    (out / "metadata.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    save_state(out / "initial_state.npz", geometry, resources, omega_comm=TARGET_WEIGHT)
    print(f"Output: {out}", flush=True)
    start = perf_counter()
    try:
        result = run_joint_algorithm(
            cfg, rng=np.random.default_rng(SOURCE_SEED), verbose=True,
            initial_endpoints=endpoints.copy(), initial_resources=resources,
            position_update_mode="fp_aware", fp_aware_candidate_iters=2,
            max_outer_iterations=1 if args.smoke else None,
        )
        elapsed = perf_counter()-start
        check_geometry_constraints(result.geometry, cfg)
        check_resource_constraints(result.resources, cfg)
        final = compute_performance(build_channels(result.geometry, cfg), result.resources, cfg)
        final_wsr = float(final.weighted_sum_rate)
        rc1, rs1 = rates(final)
        history = result.joint_wsr_history
        if not np.isclose(history[0], initial_wsr, rtol=1e-10, atol=1e-10):
            raise RuntimeError("Solver did not start from the supplied full state.")
        trace = [dict(joint_iteration=0, wsr=history[0], relative_change="",
                      inner_start_wsr="", inner_end_wsr="", inner_gain="",
                      inner_iterations="", accepted_endpoints="")]
        for index in range(1, len(history)):
            inner = result.inner_wsr_histories[index-1]
            trace.append(dict(joint_iteration=index, wsr=history[index],
                              relative_change=relative_change(history[index], history[index-1]),
                              inner_start_wsr=inner[0], inner_end_wsr=inner[-1],
                              inner_gain=inner[-1]-inner[0], inner_iterations=len(inner)-1,
                              accepted_endpoints=result.accepted_position_counts[index-1]))
        write_csv(out / "joint_history.csv", trace)
        decreased = final_wsr < initial_wsr - 1e-8*max(1., abs(initial_wsr))
        retain_final = bool(np.isfinite(final_wsr) and final_wsr >= initial_wsr)
        if retain_final:
            retained_g, retained_r, retained_wsr = result.geometry, result.resources, final_wsr
        else:
            retained_g, retained_r, retained_wsr = geometry, resources, initial_wsr
        save_state(out / "final_state.npz", result.geometry, result.resources,
                   omega_comm=TARGET_WEIGHT, converged=result.converged,
                   stop_reason=result.stop_reason, joint_wsr_history=np.asarray(history))
        save_state(out / "retained_state.npz", retained_g, retained_r,
                   omega_comm=TARGET_WEIGHT, wsr=retained_wsr,
                   origin="continued" if retain_final else "initial")
        summary = dict(source_weight=SOURCE_WEIGHT, target_weight=TARGET_WEIGHT, source_seed=SOURCE_SEED,
                       initial_wsr=initial_wsr, final_wsr=final_wsr, gain_from_initial=final_wsr-initial_wsr,
                       cold20_wsr=cold_wsr, gain_vs_cold20=final_wsr-cold_wsr,
                       initial_rate_comm=rc0, initial_rate_sensing=rs0,
                       final_rate_comm=rc1, final_rate_sensing=rs1,
                       iterations=result.outer_iterations, converged=bool(result.converged),
                       stop_reason=result.stop_reason, runtime_s=elapsed,
                       first_inner_gain=result.inner_wsr_histories[0][-1]-result.inner_wsr_histories[0][0],
                       accepted_endpoints_total=sum(result.accepted_position_counts),
                       endpoints_change_norm=float(np.linalg.norm(result.geometry.endpoints-endpoints)),
                       significant_decrease=decreased, retained_wsr=retained_wsr, smoke=args.smoke)
        write_csv(out / "warmstart_summary.csv", [summary])
        print(f"DONE: {summary}", flush=True)
        print(f"Files: {out}", flush=True)
        if decreased:
            raise RuntimeError("Final WSR decreased significantly; initial candidate retained for diagnosis.")
    except Exception as exc:
        (out / "error.txt").write_text(f"{type(exc).__name__}: {exc}", encoding="utf-8")
        print(f"ERROR: {exc}; initial state preserved at {out}", flush=True)
        raise


if __name__ == "__main__":
    main()
