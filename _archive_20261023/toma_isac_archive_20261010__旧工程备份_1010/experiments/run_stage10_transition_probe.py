"""Stage 10 transition-zone diagnostic for Proposed and MA+FP.

Two non-destructive phases:
  cold -- seven weights x two schemes x independent seeds, save every result/state.
  pool -- audit all cold candidates at every target weight; refine top foreign states.

Outputs are diagnostic, NOT a smoothed/filtered Pareto frontier. The solver is unchanged.
Requires run_stage10_pool_refinement.py (shipped alongside this script).
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from stage8 import load_stage8_experiment_config, make_scene_config
from experiments import run_stage10_pool_refinement as pool
from experiments.run_stage10_tradeoff import load_tradeoff_config

SCHEMES = ("proposed", "b6")
WEIGHTS = (0.50, 0.52, 0.54, 0.55, 0.56, 0.58, 0.60)
DEFAULT_OUT = ROOT / "results" / "stage10" / "transition_050_060_tol1e4"


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _attempts(out: Path) -> list[dict]:
    result = []
    for path in sorted((out / "attempts").glob("*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        item["attempt_file"] = str(path.relative_to(out))
        result.append(item)
    return result


def _num(row, field):
    return float(row[field])


def _approx_equal(a, b, eps=5e-12):
    return abs(float(a)-float(b)) <= eps


def _candidate_metrics(candidates, scheme):
    """Mark domination among sampled feasible states, not the true Pareto frontier."""
    samples = [x for x in candidates if x["scheme"] == scheme]
    for row in samples:
        c, s = float(row["sum_rate_comm"]), float(row["sum_rate_sensing"])
        dominators = []
        for other in samples:
            if other is row:
                continue
            oc, os = float(other["sum_rate_comm"]), float(other["sum_rate_sensing"])
            if oc >= c - 1e-7 and os >= s - 1e-7 and (oc > c + 1e-7 or os > s + 1e-7):
                dominators.append(other)
        row["dominated_in_sample"] = bool(dominators)
        row["n_dominators_in_sample"] = len(dominators)


def write_analysis(out: Path, scene, weights=WEIGHTS, schemes=SCHEMES):
    """All cold/warm attempts, full reweight matrix, and candidate Pareto sample.

    Nonconverged trajectories are kept in attempt log, but not considered certified
    target-weight converged solutions. Pure feasible cold states ARE reweighted to
    other targets as a diagnostic, not falsely reported as target-converged.
    """
    raw = _attempts(out)
    if not raw:
        return
    attempt_fields = ["method", "scheme", "omega_comm", "origin_omega", "seed", "converged",
                      "status", "iterations", "stop_reason", "sum_rate_dl", "sum_rate_ul",
                      "sum_rate_comm", "sum_rate_sensing", "weighted_sum_rate",
                      "runtime_s", "state_file", "error", "attempt_file"]
    pool.csv_write(out / "all_attempts.csv", raw, attempt_fields)

    successful = [x for x in raw if x.get("status") == "ok" and x.get("converged") is True]
    for candidate in successful:
        # Always verify state authenticity under its own weight.
        cfg = pool.target_cfg(scene, float(candidate["omega_comm"]))
        state_file = out / candidate["state_file"]
        pool.verify_state(state_file, cfg, float(candidate["weighted_sum_rate"]))

    annotated = [dict(x) for x in successful]
    for scheme in schemes:
        _candidate_metrics(annotated, scheme)
    pool.csv_write(out / "all_converged_candidates.csv", annotated,
                   attempt_fields + ["dominated_in_sample", "n_dominators_in_sample"])

    matrix = []
    summaries = []
    targets = _rows(out / "stage10_tradeoff.csv")
    for scheme in schemes:
        foreign_pool = [x for x in successful if x["scheme"] == scheme and x["method"] == "cold"]
        all_scheme = [x for x in successful if x["scheme"] == scheme]
        for omega in weights:
            cfg = pool.target_cfg(scene, omega)
            evaluated = []
            for c in foreign_pool:
                origin = float(c["omega_comm"])
                # Weighted rate from original state; channels and physical scene identical.
                calc = (omega/(2*scene.k_dl)*float(c["sum_rate_dl"]) +
                        omega/(2*scene.j_ul)*float(c["sum_rate_ul"]) +
                        (1-omega)/scene.l_target*float(c["sum_rate_sensing"]))
                evaluated.append((calc, c))
            evaluated.sort(key=lambda entry: entry[0], reverse=True)
            for rank, (value, c) in enumerate(evaluated, 1):
                matrix.append(dict(scheme=scheme, target_omega=omega,
                                   rank_by_target_wsr=rank, origin_omega=c["omega_comm"],
                                   origin_seed=c["seed"], target_reweighted_wsr=value,
                                   source_own_wsr=c["weighted_sum_rate"],
                                   sum_rate_comm=c["sum_rate_comm"],
                                   sum_rate_sensing=c["sum_rate_sensing"],
                                   is_target_converged=_approx_equal(omega, c["omega_comm"]),
                                   state_file=c["state_file"]))
            own = [c for c in foreign_pool if _approx_equal(c["omega_comm"], omega)]
            target_converged = [c for c in all_scheme if _approx_equal(c["omega_comm"], omega)]
            selected = [r for r in targets if r["scheme"] == scheme and _approx_equal(r["omega_comm"], omega)]
            cold_best = max((float(c["weighted_sum_rate"]) for c in own), default=float("nan"))
            feasible_best = evaluated[0][0] if evaluated else float("nan")
            converged_best = max((float(c["weighted_sum_rate"]) for c in target_converged), default=float("nan"))
            summaries.append(dict(scheme=scheme, omega_comm=omega,
                                  n_cold_converged=len(own), n_target_converged=len(target_converged),
                                  cold_best_wsr=cold_best, best_feasible_reweighted_wsr=feasible_best,
                                  best_target_converged_wsr=converged_best,
                                  final_selected_wsr=float(selected[0]["weighted_sum_rate"]) if selected else "",
                                  unresolved_gap=max(0., feasible_best-converged_best) if np.isfinite(converged_best) else "",
                                  best_feasible_origin=evaluated[0][1]["omega_comm"] if evaluated else "",
                                  n_cold_different_rate_pairs=len({(round(float(c["sum_rate_comm"]),3),
                                                                    round(float(c["sum_rate_sensing"]),3)) for c in own})))
    pool.csv_write(out / "all_reweighted_scores.csv", matrix,
                   ["scheme", "target_omega", "rank_by_target_wsr", "origin_omega",
                    "origin_seed", "target_reweighted_wsr", "source_own_wsr",
                    "sum_rate_comm", "sum_rate_sensing", "is_target_converged", "state_file"])
    pool.csv_write(out / "transition_summary.csv", summaries,
                   ["scheme", "omega_comm", "n_cold_converged", "n_target_converged",
                    "cold_best_wsr", "best_feasible_reweighted_wsr", "best_target_converged_wsr",
                    "final_selected_wsr", "unresolved_gap", "best_feasible_origin",
                    "n_cold_different_rate_pairs"])
    print(f"[ANALYSIS] {len(raw)} attempts, {len(successful)} converged, "
          f"{len(matrix)} cross-weight evaluations")
    if (out / "stage10_tradeoff.csv").exists():
        print("[ANALYSIS] Updated selected/target-converged results and remaining gaps")


def _manifest(out: Path, cfg_path: Path, scene, exp, weights, starts, refine_top):
    hash_files = ["algorithm.py", "baselines.py", "ma_baseline.py", "inner_solver.py",
                  "metrics.py", "config.py", "stage8.py", "experiments/run_stage10_tradeoff.py",
                  "experiments/run_stage10_pool_refinement.py", "experiments/run_stage10_transition_probe.py",
                  "configs/stage10_tradeoff.yaml"]
    signature = dict(weights=list(weights), schemes=list(SCHEMES), starts=starts,
                     refine_top=refine_top, physical_scene=repr(scene),
                     ma_params=[exp.ma_region_size, exp.ma_min_spacing,
                                exp.ma_initial_step, exp.ma_initial_jitter_fraction],
                     code_hashes={x: sha256((ROOT / x).read_bytes()).hexdigest() for x in hash_files})
    stamp = sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    path = out / "manifest.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("sha256") != stamp:
            raise RuntimeError("Existing diagnostics were generated under different parameters/source. "
                               f"Choose a new --output folder instead of mixing runs: {out}")
    return path, stamp, signature


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=("cold", "pool", "all"), default="all")
    ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--refine-top", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    if args.starts < 1 or args.refine_top < 1:
        ap.error("--starts and --refine-top must both be positive")
    cfg_path = ROOT / "configs" / "stage10_tradeoff.yaml"
    base = load_config(cfg_path)
    exp = load_stage8_experiment_config(cfg_path)
    tr = load_tradeoff_config(cfg_path)
    if not (np.isclose(base.tol_inner, 1e-4, rtol=1e-8) and
            np.isclose(base.tol_outer, 1e-4, rtol=1e-8)):
        raise RuntimeError("This diagnostic expects tol_inner=tol_outer=1e-4 in configs/stage10_tradeoff.yaml")
    if (base.k_dl, base.j_ul, base.l_target, base.r_sensing) != (5,5,5,4):
        raise RuntimeError("Expected K=J=L=5, R_s=4; do not compare with different system loads")
    if exp.fading_enabled:
        raise RuntimeError("Expected LOS/no randomized fading in current Stage 10 scene")
    scene = make_scene_config(base, scene_seed=tr.scene_seed,
                              position_jitter=exp.position_jitter,
                              randomize_fading=False, fading_model=exp.fading_model)
    out = args.output.resolve()
    # make_seed(3) is deterministic, with B6 and Proposed having different spatial starting schemes.
    for scheme in SCHEMES:
        pool.make_seeds(scheme, exp, tr.scene_seed, args.starts)
    manifest, stamp, signature = _manifest(out, cfg_path, scene, exp, WEIGHTS,
                                           args.starts, args.refine_top)
    print("Stage 10 TRANSITION TEST (NOT final-paper smoothing)")
    print("Schemes:", SCHEMES, "weights:", WEIGHTS)
    print("Starts:", args.starts, "foreign refinements/target:", args.refine_top)
    print("Tolerance inner/outer: 1e-4; fixed physical scene seed:", tr.scene_seed)
    print("Output:", out)
    if args.dry_run:
        print("[OK] Loaded configuration and solver interfaces. No optimization run.")
        return
    pool.json_write(manifest, dict(sha256=stamp, parameters=signature))
    if args.phase in ("cold", "all"):
        pool.run_cold(out, scene, exp, WEIGHTS, SCHEMES, args.starts, tr.scene_seed)
        write_analysis(out, scene)
    if args.phase in ("pool", "all"):
        pool.audit_candidates(out, scene, WEIGHTS, SCHEMES, args.refine_top, exp)
        pool.pareto_audit(out)
        write_analysis(out, scene)
    print("[COMPLETE] Diagnostics available in", out)


if __name__ == "__main__":
    main()
