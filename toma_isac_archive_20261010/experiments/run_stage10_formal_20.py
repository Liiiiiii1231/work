"""Stage 10: reproducible four-scheme, 20-weight WSR sweep with multi-round candidate pool.

Keeps the original FP/RCG/MA solvers and physical configuration unchanged.
Run with --phase cold followed by --phase pool, with identical CLI options.
All saved states are complete (geometry + resources), and only solutions
converged at the TARGET weight can enter the final comparison CSV.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import stage10_20_support as base
from experiments.run_stage10_tradeoff import load_tradeoff_config
from config import load_config
from stage8 import load_stage8_experiment_config, make_scene_config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "stage10" / "formal_20point_dl114_multiround"
SCHEMES = base.SCHEMES
OUTPUT_FIELDS = [
    "omega_comm", "omega_sensing", "scheme", "scheme_label", "sum_rate_dl",
    "sum_rate_ul", "sum_rate_comm", "sum_rate_sensing", "weighted_sum_rate",
    "best_seed", "iterations", "runtime_s", "n_starts", "n_refinements",
    "converged", "selection_kind", "origin_omega", "round", "state_file",
]
AUDIT_FIELDS = [
    "scheme", "omega_comm", "cold_best_wsr", "pool_best_feasible_wsr",
    "selected_converged_wsr", "unresolved_gap", "pool_origin_omega",
    "selection_kind", "cold_candidates", "warm_candidates", "selected_state",
]


def state_score(row: dict, cfg) -> float:
    """Score one weight-independent physical state at a NEW target weight."""
    return float(
        np.sum(cfg.weight_dl) / cfg.k_dl * float(row["sum_rate_dl"])
        + np.sum(cfg.weight_ul) / cfg.j_ul * float(row["sum_rate_ul"])
        + np.sum(cfg.weight_s) / cfg.l_target * float(row["sum_rate_sensing"])
    )


def all_converged(out: Path) -> list[dict]:
    items = []
    for p in sorted((out / "attempts").glob("*.json")):
        entry = json.loads(p.read_text(encoding="utf-8"))
        if entry.get("status") != "ok" or entry.get("converged") is not True:
            continue
        state_rel = entry.get("state_file", "")
        if not state_rel:
            raise RuntimeError(f"Converged attempt lacks full state: {p}")
        if not (out / state_rel).is_file():
            raise FileNotFoundError(f"Missing saved state: {out / state_rel}")
        entry["attempt_file"] = str(p.relative_to(out))
        items.append(entry)
    return items


def ensure_cold(out, weights, schemes):
    records = all_converged(out)
    missing = [(sch, w) for sch in schemes for w in weights if not any(
        row["method"] == "cold" and row["scheme"] == sch
        and np.isclose(float(row["omega_comm"]), w, rtol=0, atol=1e-12)
        for row in records
    )]
    if missing:
        raise RuntimeError(f"Missing converged cold starts: {missing}. Run --phase cold first.")


def best_rows(out: Path, scene_cfg, weights, schemes, starts: int):
    records = all_converged(out)
    csv_results, audits = [], []
    for w in weights:
        cfg = base.target_cfg(scene_cfg, w)
        for scheme in schemes:
            candidates = [r for r in records if r["scheme"] == scheme]
            at_weight = [r for r in candidates if np.isclose(
                float(r["omega_comm"]), w, rtol=0, atol=1e-12)]
            cold = [r for r in at_weight if r["method"] == "cold"]
            if not cold:
                raise RuntimeError(f"Missing converged cold start: {scheme} {w}")
            cold_best = max(cold, key=lambda x: state_score(x, cfg))
            selected = max(at_weight, key=lambda x: state_score(x, cfg))
            reweighted = max(candidates, key=lambda x: state_score(x, cfg))
            wsr = state_score(selected, cfg)
            feasible = state_score(reweighted, cfg)
            # Reject mismatched state/rate metadata; recompute with the true SINR evaluator.
            base.verify_state(out / selected["state_file"], cfg, wsr)
            if feasible > wsr + 1e-6:
                base.verify_state(out / reweighted["state_file"], cfg, feasible)
            csv_results.append(dict(
                omega_comm=w, omega_sensing=1-w, scheme=scheme,
                scheme_label=base.LABELS[scheme],
                sum_rate_dl=selected["sum_rate_dl"], sum_rate_ul=selected["sum_rate_ul"],
                sum_rate_comm=selected["sum_rate_comm"],
                sum_rate_sensing=selected["sum_rate_sensing"],
                weighted_sum_rate=wsr, best_seed=selected["seed"],
                iterations=selected["iterations"], runtime_s=selected.get("runtime_s", 0),
                n_starts=starts, n_refinements=sum(r["method"] != "cold" for r in at_weight),
                converged=True,
                selection_kind="cold" if selected["method"] == "cold" else "warm",
                origin_omega=selected.get("origin_omega", w),
                round=selected.get("round", 0), state_file=selected["state_file"],
            ))
            audits.append(dict(
                scheme=scheme, omega_comm=w,
                cold_best_wsr=state_score(cold_best, cfg),
                pool_best_feasible_wsr=feasible,
                selected_converged_wsr=wsr,
                unresolved_gap=max(0., feasible-wsr),
                pool_origin_omega=reweighted["omega_comm"],
                selection_kind=csv_results[-1]["selection_kind"],
                cold_candidates=len(cold), warm_candidates=len(at_weight)-len(cold),
                selected_state=selected["state_file"],
            ))
    order = {s:i for i,s in enumerate(SCHEMES)}
    csv_results.sort(key=lambda r:(float(r["omega_comm"]),order[r["scheme"]]))
    audits.sort(key=lambda r:(float(r["omega_comm"]),order[r["scheme"]]))
    base.csv_write(out / "stage10_tradeoff.csv", csv_results, OUTPUT_FIELDS)
    base.csv_write(out / "candidate_audit.csv", audits, AUDIT_FIELDS)
    return csv_results, audits


def _attempted_sources(out: Path, scheme: str, target_w: float):
    seen = set()
    for p in (out / "attempts").glob("warm_*.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("scheme") == scheme and np.isclose(
            float(r.get("omega_comm", -9)), target_w, rtol=0, atol=1e-12
        ) and "source_state_file" in r:
            seen.add(r["source_state_file"])
    return seen


def refine_round(out: Path, scene_cfg, exp, weights, schemes, round_id: int,
                 refine_top: int, starts: int):
    # Persist the round plan BEFORE optimization: an interrupted run must use
    # exactly the same candidate sources when resumed.
    catalog = all_converged(out)
    plan_file = out / f"pool_round_{round_id:02d}_plan.json"
    by_file = {r["state_file"]: r for r in catalog}
    jobs = []
    if plan_file.exists():
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
        for record in plan["jobs"]:
            source_path = record["source_state_file"]
            if source_path not in by_file:
                raise RuntimeError(f"Round {round_id} source state not in catalog: {source_path}")
            jobs.append((float(record["omega_comm"]), record["scheme"], by_file[source_path]))
        print(f"[POOL ROUND {round_id}] loaded saved plan", flush=True)
    else:
        for w in weights:
            cfg = base.target_cfg(scene_cfg, w)
            for sch in schemes:
                foreign = [r for r in catalog if r["scheme"] == sch and not np.isclose(
                    float(r["omega_comm"]), w, rtol=0, atol=1e-12)]
                foreign.sort(key=lambda r:state_score(r,cfg), reverse=True)
                attempted = _attempted_sources(out, sch, w)
                selected = []
                distinct_weights = set()
                for candidate in foreign:
                    if candidate["state_file"] in attempted:
                        continue
                    # Spread search over source-weight modes before taking near-duplicates.
                    source_weight = float(candidate["omega_comm"])
                    if source_weight in distinct_weights:
                        continue
                    selected.append(candidate)
                    distinct_weights.add(source_weight)
                    if len(selected) == refine_top:
                        break
                for c in selected:
                    jobs.append((w, sch, c))
        base.json_write(plan_file, {"round":round_id, "jobs":[
            {"omega_comm":w, "scheme":sch, "source_state_file":c["state_file"]}
            for w,sch,c in jobs
        ]})
    print(f"[POOL ROUND {round_id}] {len(jobs)} planned source/target refinements", flush=True)
    for job_i, (w, sch, source) in enumerate(jobs, 1):
        cfg = base.target_cfg(scene_cfg, w)
        srcrel = source["state_file"]
        digest = sha256(srcrel.encode("utf-8")).hexdigest()[:14]
        wi = next(i for i, x in enumerate(weights) if abs(x-w) <= 1e-12)
        key = f"warm_{sch}_w{wi:02d}_r{round_id:02d}_src{digest}"
        attempt_path = out / "attempts" / f"{key}.json"
        if attempt_path.exists():
            old = json.loads(attempt_path.read_text(encoding="utf-8"))
            if old.get("status") in ("ok", "nonconverged"):
                print(f"  [{job_i}/{len(jobs)}] resume {key}", flush=True)
                continue
        print(f"  [{job_i}/{len(jobs)}] {sch} target={w:g} <- "
              f"{source['omega_comm']:g} ({source['method']})", flush=True)
        # Confirm same scene, physical feasibility, and score BEFORE warm start.
        initial_score = state_score(source, cfg)
        base.verify_state(out / srcrel, cfg, initial_score)
        geo, resources, _ = base.read_state(out / srcrel)
        t0 = perf_counter()
        try:
            result = base.run_solver(sch, cfg, exp, int(source["seed"]), warm=(geo, resources))
            meta = base.extract_row(sch, w, int(source["seed"]), result,
                                    "cross_weight_refined", "", float(source["omega_comm"]))
            meta.update(round=round_id, source_state_file=srcrel,
                        initial_feasible_wsr=initial_score,
                        runtime_s=round(perf_counter()-t0, 4),
                        status="ok" if result.converged else "nonconverged")
            if result.converged:
                state_rel = Path("warm_states") / f"{key}.npz"
                base.save_state(out / state_rel, result)
                meta["state_file"] = str(state_rel)
                base.verify_state(out / state_rel, cfg, meta["weighted_sum_rate"])
                if float(meta["weighted_sum_rate"]) + 1e-6 < initial_score:
                    print("  [WARNING] converged WSR BELOW feasible warm-start WSR; "
                          "investigate solver monotonicity", flush=True)
            base.json_write(attempt_path, meta)
        except Exception as exc:
            base.json_write(attempt_path, dict(
                status="error", scheme=sch, omega_comm=w, method="cross_weight_refined",
                source_state_file=srcrel, round=round_id,
                error=f"{type(exc).__name__}: {exc}"))
            raise
        # Save checkpoint table after every completed numerical solve.
        best_rows(out, scene_cfg, weights, schemes, starts)


def report(out, scene_cfg, weights, schemes, starts):
    selected, audits = best_rows(out, scene_cfg, weights, schemes, starts)
    base.pareto_audit(out)
    biggest = sorted(audits, key=lambda r:r["unresolved_gap"], reverse=True)[:5]
    print("Largest remaining feasible-vs-converged gaps:")
    for r in biggest:
        print(f"  {r['scheme']} w={r['omega_comm']:.4g}: gap={r['unresolved_gap']:.6g}")
    print(f"[SUMMARY] {len(selected)} target-converged rows saved under {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("cold", "pool", "report", "all"), default="all")
    p.add_argument("--starts", type=int, default=5)
    p.add_argument("--refine-top", type=int, default=2)
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--schemes", nargs="+", choices=SCHEMES, default=list(SCHEMES))
    p.add_argument("--weights", nargs="+", type=float, help="Optional subset ONLY for testing")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.starts < 1 or args.refine_top < 0 or args.passes < 0:
        p.error("--starts must be >=1; --refine-top and --passes must be >=0")
    yml = ROOT / "configs" / "stage10_tradeoff_20.yaml"
    trade = load_tradeoff_config(yml)
    basecfg = load_config(yml)
    exp = load_stage8_experiment_config(yml)
    weights = tuple(args.weights) if args.weights else trade.omega_comm
    if len(set(weights)) != len(weights) or any(
        not any(np.isclose(w,q,rtol=0,atol=1e-12) for q in trade.omega_comm)
        for w in weights
    ):
        raise ValueError("Weights must be distinct values from configs/stage10_tradeoff.yaml")
    # Guard against accidentally mixing the old -70 dBm scene with the new sweep.
    expected_noise = 3.981071705534969e-15
    if not np.allclose(basecfg.sigma_dl2, expected_noise, rtol=1e-7, atol=0):
        raise RuntimeError("This 20-weight experiment requires DL noise = -114 dBm")
    if not np.isclose(basecfg.sigma_bs2, expected_noise, rtol=1e-7, atol=0):
        raise RuntimeError("This 20-weight experiment requires BS noise = -114 dBm")
    if len(trade.omega_comm) != 20:
        raise RuntimeError("Expected exactly 20 configured communication weights")
    schemes = tuple(args.schemes)
    if len(set(schemes)) != len(schemes):
        raise ValueError("--schemes must not contain duplicates")
    scene = make_scene_config(basecfg, scene_seed=trade.scene_seed,
                              position_jitter=exp.position_jitter,
                              randomize_fading=exp.fading_enabled,
                              fading_model=exp.fading_model)
    seeds = {scheme:list(base.make_seeds(scheme,exp,trade.scene_seed,args.starts)) for scheme in schemes}
    if args.weights or schemes != SCHEMES:
        signature = sha256(json.dumps([weights, schemes]).encode()).hexdigest()[:10]
        output = args.output / f"subset_{signature}"
    else:
        output = args.output
    sources = (
        "algorithm.py", "baselines.py", "ma_baseline.py", "inner_solver.py",
        "outer_solver.py", "metrics.py", "config.py", "channels.py", "state.py",
        "experiments/stage10_20_support.py", "experiments/run_stage10_formal_20.py",
        "configs/stage10_tradeoff_20.yaml",
    )
    params = dict(weights=weights, schemes=schemes, starts=args.starts,
                  refine_top=args.refine_top, passes=args.passes,
                  scene_seed=trade.scene_seed, seeds=seeds,
                  physical_config=repr(scene),
                  source_hashes={fn:sha256((ROOT/fn).read_bytes()).hexdigest() for fn in sources})
    fingerprint = sha256(json.dumps(params,sort_keys=True,default=str).encode()).hexdigest()
    manifest = output / "manifest.json"
    if manifest.exists():
        current = json.loads(manifest.read_text(encoding="utf-8"))
        if current.get("sha256") != fingerprint:
            raise RuntimeError("Experiment parameters/solver files changed. "
                               f"Use a NEW --output directory; do not mix runs in {output}.")
    print("Stage 10 FORMAL 20-WEIGHT DL -114 dBm", "weights=", weights, "schemes=", schemes, flush=True)
    print(f"starts={args.starts} rounds={args.passes} sources/target/round={args.refine_top}")
    print("output:",output)
    if args.dry_run:
        print("[OK] YAML, seeds and source files verified; no optimization executed")
        return
    base.json_write(manifest, dict(sha256=fingerprint, parameters=params))
    if args.phase in ("cold", "all"):
        base.run_cold(output, scene, exp, weights, schemes, args.starts, trade.scene_seed)
    if args.phase in ("pool", "all", "report"):
        ensure_cold(output, weights, schemes)
    if args.phase in ("pool", "all"):
        for k in range(1, args.passes+1):
            refine_round(output, scene, exp, weights, schemes, k, args.refine_top, args.starts)
            report(output, scene, weights, schemes, args.starts)
    elif args.phase == "report":
        report(output, scene, weights, schemes, args.starts)


if __name__ == "__main__":
    main()
