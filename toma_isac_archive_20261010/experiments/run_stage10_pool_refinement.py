"""Stage 10: independent cold starts + within-scheme cross-weight candidate refinement.

Safe by design:
- Does NOT change FP, RCG, geometry constraints, tolerances, or original Stage 10 outputs.
- All 14 weights use one physical scene (scene_seed=8001 from YAML).
- Reweighted feasible states are ONLY an audit; final plotting rows must have
  converged under the target weight (cold or resumed solver result).
- All candidate states (geometry + resource variables) are saved as npz without pickle.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm import run_joint_algorithm
from baselines import run_fixed_toma_fp, run_fpa_upa_fp
from channels import build_channels
from config import load_config
from ma_baseline import run_ma_fp
from metrics import compute_performance
from stage8 import load_stage8_experiment_config, make_ma_start_seeds, make_scene_config
from state import GeometryState, ResourceState
from utils import check_resource_constraints
from experiments.run_stage10_tradeoff import load_tradeoff_config, make_group_weights

SCHEMES = ("b2", "b3", "b6", "proposed")
LABELS = {"b2": "Fixed-ToMA + FP", "b3": "FPA-UPA + FP",
          "b6": "MA + FP", "proposed": "FP-aware RCG-ToMA + FP (Proposed)"}
DEFAULT_OUTPUT = ROOT / "results" / "stage10" / "pool_wide14_tol1e4"


def csv_write(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def json_write(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def snapshot_fields(result):
    g, r, p = result.geometry, result.resources, result.performance
    return dict(
        endpoints=np.array(g.endpoints, copy=True),
        tx_positions=np.array(g.tx_positions, copy=True),
        rx_positions=np.array(g.rx_positions, copy=True),
        q_matrix=np.array(r.q_matrix, copy=True),
        q_ul=np.array(r.q_ul, copy=True),
        b_ul=np.array(r.b_ul, copy=True),
        u_s=np.array(r.u_s, copy=True),
        rate_dl=np.array(p.rate_dl, copy=True),
        rate_ul=np.array(p.rate_ul, copy=True),
        rate_s=np.array(p.rate_s, copy=True),
    )


def save_state(path: Path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **snapshot_fields(result))
    tmp.replace(path)


def read_state(path: Path):
    with np.load(path, allow_pickle=False) as data:
        g = GeometryState(
            endpoints=data["endpoints"].copy(),
            tx_positions=data["tx_positions"].copy(),
            rx_positions=data["rx_positions"].copy(),
        )
        r = ResourceState(
            q_matrix=data["q_matrix"].copy(),
            q_ul=data["q_ul"].copy(),
            b_ul=data["b_ul"].copy(),
            u_s=data["u_s"].copy(),
        )
        rates = (data["rate_dl"].copy(), data["rate_ul"].copy(), data["rate_s"].copy())
    return g, r, rates


def target_cfg(scene_cfg, weight: float):
    wd, wu, ws = make_group_weights(scene_cfg, weight)
    cfg = replace(scene_cfg, weight_dl=wd, weight_ul=wu, weight_s=ws)
    cfg.validate()
    return cfg


def weighted_rates(rate_dl, rate_ul, rate_s, cfg):
    return float(np.dot(cfg.weight_dl, rate_dl)
                 + np.dot(cfg.weight_ul, rate_ul)
                 + np.dot(cfg.weight_s, rate_s))


def verify_state(path: Path, cfg, expected_wsr: float | None = None):
    g, resources, _ = read_state(path)
    check_resource_constraints(resources, cfg)
    # Rebuild channels instead of reusing potentially stale weighted-channel caches.
    performance = compute_performance(build_channels(g, cfg), resources, cfg)
    wsr = float(performance.weighted_sum_rate)
    if not np.isfinite(wsr):
        raise RuntimeError(f"Non-finite WSR for {path}")
    if expected_wsr is not None and not np.isclose(wsr, expected_wsr, rtol=1e-6, atol=2e-7):
        raise RuntimeError(f"State/CSV mismatch {path}: {wsr} != {expected_wsr}")
    return wsr


def make_seeds(scheme, exp, scene_seed, n_starts):
    if scheme == "b6":
        return make_ma_start_seeds(scene_seed, int(exp.fixed_upa_seed), n_starts)
    if scheme == "b3":
        unique = (int(exp.fixed_upa_seed),) + tuple(int(s) for s in exp.multi_start_seeds)
        unique = tuple(dict.fromkeys(unique))
    else:
        unique = tuple(dict.fromkeys(int(s) for s in exp.multi_start_seeds))
    if len(unique) < n_starts:
        raise ValueError(f"Not enough distinct configured seeds for {scheme}: {unique}")
    return unique[:n_starts]


def run_solver(scheme, cfg, exp, seed: int, warm=None):
    rng = np.random.default_rng(int(seed))
    if scheme == "b2":
        kw = {} if warm is None else dict(initial_endpoints=warm[0].endpoints,
                                          initial_resources=warm[1])
        return run_fixed_toma_fp(cfg, rng=rng, verbose=False, **kw)
    if scheme == "b3":
        kw = {} if warm is None else dict(initial_resources=warm[1])
        return run_fpa_upa_fp(cfg, rng=rng, verbose=False, **kw)
    if scheme == "b6":
        kw = {} if warm is None else dict(initial_tx_positions=warm[0].tx_positions,
                                          initial_rx_positions=warm[0].rx_positions,
                                          initial_resources=warm[1])
        return run_ma_fp(
            cfg, rng=rng, verbose=False,
            region_size=exp.ma_region_size,
            min_spacing=exp.ma_min_spacing,
            initial_step=exp.ma_initial_step,
            randomize_initial_geometry=warm is None and int(seed) != int(exp.fixed_upa_seed),
            initial_jitter_fraction=exp.ma_initial_jitter_fraction,
            **kw,
        )
    if scheme == "proposed":
        kw = {} if warm is None else dict(initial_endpoints=warm[0].endpoints,
                                          initial_resources=warm[1])
        return run_joint_algorithm(cfg, rng=rng, verbose=False,
                                   position_update_mode="fp_aware",
                                   fp_aware_candidate_iters=2, **kw)
    raise ValueError(scheme)


def iterations(result):
    return int(getattr(result, "outer_iterations", getattr(result, "iterations", 0)))


def extract_row(scheme, weight, seed, result, method, state_rel, origin_weight):
    p = result.performance
    return dict(scheme=scheme, omega_comm=float(weight), seed=int(seed),
                origin_omega=float(origin_weight), method=method,
                converged=bool(result.converged),
                iterations=iterations(result),
                stop_reason=getattr(result, "stop_reason", "not_recorded"),
                sum_rate_dl=float(np.sum(p.rate_dl)),
                sum_rate_ul=float(np.sum(p.rate_ul)),
                sum_rate_comm=float(np.sum(p.rate_dl) + np.sum(p.rate_ul)),
                sum_rate_sensing=float(np.sum(p.rate_s)),
                weighted_sum_rate=float(p.weighted_sum_rate),
                state_file=str(state_rel) if state_rel else "")


def successful_attempts(out: Path, prefix: str):
    rows = []
    for f in sorted((out / "attempts").glob(prefix + "*.json")):
        item = json.loads(f.read_text(encoding="utf-8"))
        if item.get("status") == "ok" and item.get("converged"):
            state = out / item["state_file"]
            if not state.exists():
                raise FileNotFoundError(f"Checkpoint refers to missing state: {state}")
            rows.append(item)
    return rows


def run_cold(out, scene_cfg, exp, weights, schemes, starts, scene_seed, resume=True):
    start_stamp = perf_counter()
    for wi, omega in enumerate(weights):
        cfg = target_cfg(scene_cfg, omega)
        print(f"\n[COLD] omega_C={omega:g}", flush=True)
        for scheme in schemes:
            seed_values = make_seeds(scheme, exp, scene_seed, starts)
            for seed in seed_values:
                key = f"cold_{scheme}_w{wi:02d}_seed{seed}"
                info_file = out / "attempts" / (key + ".json")
                if info_file.exists() and resume:
                    old = json.loads(info_file.read_text(encoding="utf-8"))
                    if old.get("status") in ("ok", "nonconverged"):
                        print(f"  [resume] {key}", flush=True)
                        continue
                print(f"  {key}", flush=True)
                time0 = perf_counter()
                try:
                    result = run_solver(scheme, cfg, exp, seed)
                    is_converged = bool(result.converged)
                    meta = extract_row(scheme, omega, seed, result, "cold", "", omega)
                    meta["runtime_s"] = round(perf_counter()-time0, 4)
                    meta["status"] = "ok" if is_converged else "nonconverged"
                    if is_converged:
                        state_rel = Path("cold_states") / (key + ".npz")
                        save_state(out / state_rel, result)
                        meta["state_file"] = str(state_rel)
                        verify_state(out / state_rel, cfg, meta["weighted_sum_rate"])
                    json_write(info_file, meta)
                    print(f"    WSR={meta['weighted_sum_rate']:.6f}, iter={meta['iterations']}, converged={is_converged}")
                except Exception as exc:
                    json_write(info_file, dict(status="error", scheme=scheme,
                                               omega_comm=omega, seed=seed,
                                               error=f"{type(exc).__name__}: {exc}"))
                    raise
    write_cold_summary(out, weights, schemes)
    print(f"[COLD] elapsed {perf_counter()-start_stamp:.1f}s")


def write_cold_summary(out: Path, weights, schemes):
    """Keep the unchanged independent cold-start best results for comparison."""
    rows = successful_attempts(out, "cold_")
    results = []
    for omega in weights:
        for scheme in schemes:
            choices = [r for r in rows if r["scheme"] == scheme
                       and np.isclose(float(r["omega_comm"]), omega, atol=1e-13)]
            if not choices:
                continue
            best = max(choices, key=lambda r: float(r["weighted_sum_rate"]))
            results.append(dict(omega_comm=omega, scheme=scheme,
                                n_converged_starts=len(choices),
                                best_seed=best["seed"],
                                weighted_sum_rate=best["weighted_sum_rate"],
                                sum_rate_comm=best["sum_rate_comm"],
                                sum_rate_sensing=best["sum_rate_sensing"],
                                iterations=best["iterations"],
                                state_file=best["state_file"]))
    csv_write(out / "cold_summary.csv", results,
              ["omega_comm", "scheme", "n_converged_starts", "best_seed",
               "weighted_sum_rate", "sum_rate_comm", "sum_rate_sensing", "iterations", "state_file"])


def audit_candidates(out, scene_cfg, weights, schemes, refine_top, exp, resume=True):
    cold = successful_attempts(out, "cold_")
    if not cold:
        raise RuntimeError("No converged cold-state files. Run --phase cold first.")
    # Require every requested scheme/weight to have at least one converged cold start.
    for scheme in schemes:
        for omega in weights:
            if not any(r["scheme"] == scheme and np.isclose(r["omega_comm"], omega, atol=1e-13)
                       for r in cold):
                raise RuntimeError(f"Missing converged cold start: {scheme} omega={omega}. Finish cold phase.")

    audits = []
    for wi, omega in enumerate(weights):
        cfg = target_cfg(scene_cfg, omega)
        print(f"\n[POOL] omega_C={omega:g}", flush=True)
        for scheme in schemes:
            candidates = []
            for item in cold:
                if item["scheme"] != scheme:
                    continue
                # Physical environment is identical across weights; rate vectors are weight-independent.
                # Exact original group weights: 1/2 comm for DL and 1/2 for UL.
                candidate_value = (omega/(2*cfg.k_dl) * float(item["sum_rate_dl"])
                                   + omega/(2*cfg.j_ul) * float(item["sum_rate_ul"])
                                   + (1-omega)/cfg.l_target * float(item["sum_rate_sensing"]))
                candidates.append((candidate_value, item))
            candidates.sort(key=lambda p: p[0], reverse=True)
            same = [pair for pair in candidates if abs(pair[1]["omega_comm"]-omega) <= 1e-13]
            best_cold_score, best_cold = max(same, key=lambda p: p[0])
            best_feasible_score, best_feasible = candidates[0]
            # Verify the best directly reweighted state with the actual SINR/WSR calculator.
            verify_state(out / best_feasible["state_file"], cfg, best_feasible_score)
            chosen = dict(best_cold)
            chosen["selection_kind"] = "cold_converged"
            chosen["target_wsr"] = float(best_cold_score)
            foreign = [pair for pair in candidates if abs(pair[1]["omega_comm"]-omega) > 1e-13]
            # Pick top cross-weight candidates from distinct origin weights.
            selected = []
            seen = set()
            for _, item in foreign:
                if item["omega_comm"] not in seen:
                    selected.append(item)
                    seen.add(item["omega_comm"])
                if len(selected) >= refine_top:
                    break
            for rank, item in enumerate(selected, 1):
                key = f"warm_{scheme}_w{wi:02d}_rank{rank:02d}_from{item['omega_comm']:.7g}_seed{item['seed']}"
                info_file = out / "attempts" / (key + ".json")
                cached = json.loads(info_file.read_text(encoding="utf-8")) if info_file.exists() and resume else None
                if cached is not None and cached.get("status") in ("ok", "nonconverged"):
                    warm_row = cached
                else:
                    print(f"  [REFINE] {scheme}, target={omega:g}, from={item['omega_comm']:g}, seed={item['seed']}", flush=True)
                    g, r, _ = read_state(out / item["state_file"])
                    time0 = perf_counter()
                    try:
                        refined = run_solver(scheme, cfg, exp, int(item["seed"]), warm=(g,r))
                        warm_row = extract_row(scheme, omega, int(item["seed"]), refined,
                                               "cross_weight_refined", "", item["omega_comm"])
                        warm_row["runtime_s"] = round(perf_counter()-time0, 4)
                        warm_row["status"] = "ok" if refined.converged else "nonconverged"
                        if refined.converged:
                            state_rel = Path("warm_states") / (key + ".npz")
                            save_state(out / state_rel, refined)
                            warm_row["state_file"] = str(state_rel)
                            verify_state(out / state_rel, cfg, warm_row["weighted_sum_rate"])
                        json_write(info_file, warm_row)
                    except Exception as exc:
                        json_write(info_file, dict(status="error",scheme=scheme, omega_comm=omega,
                                                   seed=item["seed"], error=str(exc)))
                        raise
                if warm_row.get("converged") and warm_row.get("status") == "ok":
                    if float(warm_row["weighted_sum_rate"]) > float(chosen["target_wsr"]):
                        chosen = dict(warm_row)
                        chosen["target_wsr"] = float(warm_row["weighted_sum_rate"])
                        chosen["selection_kind"] = "warm_converged"
            print(f"  {scheme}: cold={best_cold_score:.6f}, pool-feasible={best_feasible_score:.6f}, final={chosen['target_wsr']:.6f}", flush=True)
            audits.append(dict(
                scheme=scheme, omega_comm=omega,
                cold_best_wsr=best_cold_score,
                pool_best_feasible_wsr=best_feasible_score,
                feasible_candidate_origin=best_feasible["omega_comm"],
                selected_wsr=chosen["target_wsr"],
                chosen=chosen,
            ))
            # Frequent incremental checkpoint, including after interruptible scheme/weight steps.
            if best_feasible_score > float(chosen["target_wsr"]) + 1.0e-5:
                print(f"  [AUDIT WARNING] pool has a feasible candidate {best_feasible_score:.6f} "
                      f"higher than target-converged selected {chosen['target_wsr']:.6f}; "
                      "do not call this weight globally optimized.")
            audits[-1]["cold_starts"] = len(same)
            audits[-1]["refined_starts"] = len(selected)
            write_pool_outputs(out, audits)
    write_pool_outputs(out, audits)


def write_pool_outputs(out, audits):
    final = []
    audit_rows = []
    for data in audits:
        entry = data["chosen"]
        scheme, omega = data["scheme"], data["omega_comm"]
        if not entry["converged"]:
            raise AssertionError("A non-converged target result must never enter the plot.")
        final.append(dict(
            omega_comm=omega,
            omega_sensing=1-omega,
            scheme=scheme,
            scheme_label=LABELS[scheme],
            sum_rate_dl=float(entry["sum_rate_dl"]),
            sum_rate_ul=float(entry["sum_rate_ul"]),
            sum_rate_comm=float(entry["sum_rate_comm"]),
            sum_rate_sensing=float(entry["sum_rate_sensing"]),
            weighted_sum_rate=float(data["selected_wsr"]),
            best_seed=int(entry["seed"]),
            iterations=int(entry["iterations"]),
            runtime_s=float(entry.get("runtime_s", 0.0)),
            n_starts=int(data["cold_starts"]),
            n_refinements=int(data["refined_starts"]),
            converged=True,
            selection_kind=entry["selection_kind"],
            origin_omega=float(entry["origin_omega"]),
            state_file=entry["state_file"],
        ))
        audit_rows.append(dict(scheme=scheme, omega_comm=omega,
                               cold_best_wsr=data["cold_best_wsr"],
                               pool_best_feasible_wsr=data["pool_best_feasible_wsr"],
                               selected_converged_wsr=data["selected_wsr"],
                               unresolved_gap=max(0., data["pool_best_feasible_wsr"]-data["selected_wsr"]),
                               pool_origin_omega=data["feasible_candidate_origin"],
                               selection_kind=entry["selection_kind"]))
    final.sort(key=lambda x: (x["omega_comm"], SCHEMES.index(x["scheme"])))
    fields = ["omega_comm", "omega_sensing", "scheme", "scheme_label",
              "sum_rate_dl", "sum_rate_ul", "sum_rate_comm", "sum_rate_sensing",
              "weighted_sum_rate", "best_seed", "iterations", "runtime_s", "n_starts",
              "n_refinements", "converged", "selection_kind", "origin_omega", "state_file"]
    csv_write(out / "stage10_tradeoff.csv", final, fields)
    csv_write(out / "candidate_audit.csv", audit_rows,
              ["scheme", "omega_comm", "cold_best_wsr", "pool_best_feasible_wsr",
               "selected_converged_wsr", "unresolved_gap", "pool_origin_omega", "selection_kind"])
    if final:
        np.savez(out / "stage10_tradeoff.npz",
                 omega_comm=np.asarray([r["omega_comm"] for r in final]),
                 scheme=np.asarray([r["scheme"] for r in final]),
                 sum_rate_comm=np.asarray([r["sum_rate_comm"] for r in final]),
                 sum_rate_sensing=np.asarray([r["sum_rate_sensing"] for r in final]),
                 weighted_sum_rate=np.asarray([r["weighted_sum_rate"] for r in final]))


def pareto_audit(out, atol=1e-8):
    path = out / "stage10_tradeoff.csv"
    with path.open("r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    count = {}
    for scheme in SCHEMES:
        vals = [r for r in rows if r["scheme"]==scheme]
        dominated = 0
        for i, row in enumerate(vals):
            c, s = float(row["sum_rate_comm"]), float(row["sum_rate_sensing"])
            for j, other in enumerate(vals):
                if i == j: continue
                oc, os = float(other["sum_rate_comm"]), float(other["sum_rate_sensing"])
                if oc >= c-atol and os >= s-atol and (oc>c+atol or os>s+atol):
                    dominated += 1
                    break
        count[scheme] = dominated
    print("Dominated solutions among selected target-converged points:", count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("cold", "pool", "all"), default="all")
    parser.add_argument("--starts", type=int, default=5, help="Cold starts per scheme and weight")
    parser.add_argument("--refine-top", type=int, default=1,
                        help="How many distinct foreign-weight candidate states to refine per target")
    parser.add_argument("--schemes", nargs="+", choices=SCHEMES, default=list(SCHEMES))
    parser.add_argument("--weights", nargs="+", type=float,
                        help="Subset of YAML weights for local validation only")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.starts < 1 or args.refine_top < 0:
        parser.error("--starts must be positive and --refine-top nonnegative")

    path = ROOT / "configs" / "stage10_tradeoff.yaml"
    base = load_config(path)
    exp = load_stage8_experiment_config(path)
    trade = load_tradeoff_config(path)
    if (base.k_dl,base.j_ul,base.l_target,base.r_sensing)!=(5,5,5,4):
        raise RuntimeError("Formal Stage 10 requires K=J=L=5, Rs=4")
    scene = make_scene_config(base, scene_seed=trade.scene_seed,
                              position_jitter=exp.position_jitter,
                              randomize_fading=exp.fading_enabled,
                              fading_model=exp.fading_model)
    weights = trade.omega_comm
    if args.weights is not None:
        weights = tuple(float(w) for w in args.weights)
        if not all(any(abs(w-q)<1e-12 for q in trade.omega_comm) for w in weights):
            raise ValueError("--weights must be a subset of values in stage10_tradeoff.yaml")
    schemes = tuple(args.schemes)
    suffix = "subset_test_" + "_".join(str(w).replace(".", "p") for w in weights)
    output = args.output if args.weights is None else args.output / suffix
    params = dict(weights=list(weights), schemes=list(schemes), starts=args.starts,
                  scene_seed=trade.scene_seed, refine_top=args.refine_top,
                  physical_config=repr(scene),
                  ma_params=[exp.ma_region_size, exp.ma_min_spacing,
                             exp.ma_initial_step, exp.ma_initial_jitter_fraction],
                  cold_seeds={scheme:list(make_seeds(scheme,exp,trade.scene_seed,args.starts)) for scheme in schemes})
    # Refuse to mix checkpoints produced by different solver implementations.
    key_files = ("algorithm.py", "baselines.py", "ma_baseline.py",
                 "inner_solver.py", "metrics.py", "config.py",
                 "experiments/run_stage10_pool_refinement.py",
                 "configs/stage10_tradeoff.yaml")
    params["source_hashes"] = {name: sha256((ROOT/name).read_bytes()).hexdigest() for name in key_files}
    digest = sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
    manifest = output / "manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if old["sha256"] != digest:
            raise RuntimeError(f"Checkpoint settings mismatch in {output}; select another --output")
    print("Stage 10 candidate pool refinement")
    print("Weights:", weights, "Schemes:", schemes)
    print("Cold starts per scheme:", args.starts, "Warm refinements per target:", args.refine_top)
    print("Output:", output)
    if args.dry_run:
        print("[OK] Config/seed checks passed; no optimization executed.")
        return
    json_write(manifest, dict(sha256=digest, parameters=params))
    if args.phase in ("cold", "all"):
        run_cold(output, scene, exp, weights, schemes, args.starts, trade.scene_seed)
    if args.phase in ("pool", "all"):
        audit_candidates(output, scene, weights, schemes, args.refine_top, exp)
        pareto_audit(output)


if __name__ == "__main__":
    main()
