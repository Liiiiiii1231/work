"""Stage 10: independent cold multi-start + forward/backward cross-weight warm-start.

Run from project root:
    python -u experiments/run_stage10_bidirectional.py
    python -u experiments/run_stage10_bidirectional.py --dry-run
    python -u experiments/run_stage10_bidirectional.py --smoke

Original FP/RCG solvers and existing Stage 10 CSVs are left untouched.
"""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Callable

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG_PATH = ROOT / "configs" / "stage10_tradeoff.yaml"
OUTPUT_DIR = ROOT / "results" / "stage10" / "bidirectional_warmstart"
SCHEMES = ("b2", "b3", "b6", "proposed")
LABELS = {
    "b2": "Fixed-ToMA + FP",
    "b3": "FPA-UPA + FP",
    "b6": "MA + FP",
    "proposed": "FP-aware RCG-ToMA + FP (Proposed)",
}
BASE_FIELDS = (
    "omega_comm", "omega_sensing", "scheme", "scheme_label", "sum_rate_dl",
    "sum_rate_ul", "sum_rate_comm", "sum_rate_sensing", "weighted_sum_rate",
    "best_seed", "iterations", "runtime_s", "n_starts", "converged",
)
EXTRA_FIELDS = ("source_type", "donor_omega", "cold_starts", "warm_attempts")
CANDIDATE_FIELDS = (
    "pass_name", "omega_comm", "scheme", "source_type", "donor_omega",
    "seed", "iterations", "converged", "wsr", "rate_comm", "rate_sensing",
    "initial_wsr", "runtime_s", "status",
)


@dataclass
class Candidate:
    result: Any
    scheme: str
    omega: float
    source_type: str
    seed: int
    iterations: int
    runtime_s: float
    initial_wsr: float | None = None
    donor_omega: float | None = None

    @property
    def wsr(self) -> float:
        return float(self.result.performance.weighted_sum_rate)

    @property
    def converged(self) -> bool:
        return bool(self.result.converged)


# Preserve candidate resources and geometry (not only scalar CSV results).
def clone_resources(resources: Any) -> Any:
    return deepcopy(resources)


def run_warm_scheme(scheme: str, cfg: Any, donor: Candidate, exp_cfg: Any) -> Any:
    """Real warm-start for each baseline's OWN feasible geometry, never cross-scheme."""
    from algorithm import run_joint_algorithm
    from baselines import run_fixed_toma_fp, run_fpa_upa_fp
    from ma_baseline import run_ma_fp

    r = donor.result
    resources = clone_resources(r.resources)
    rng = np.random.default_rng(donor.seed)
    if scheme == "b2":
        return run_fixed_toma_fp(
            cfg, rng=rng, verbose=False,
            initial_endpoints=np.array(r.geometry.endpoints, copy=True),
            initial_resources=resources,
        )
    if scheme == "b3":
        return run_fpa_upa_fp(
            cfg, rng=rng, verbose=False, initial_resources=resources,
        )
    if scheme == "b6":
        return run_ma_fp(
            cfg, rng=rng, verbose=False,
            region_size=exp_cfg.ma_region_size,
            min_spacing=exp_cfg.ma_min_spacing,
            initial_step=exp_cfg.ma_initial_step,
            initial_jitter_fraction=exp_cfg.ma_initial_jitter_fraction,
            initial_tx_positions=np.array(r.geometry.tx_positions, copy=True),
            initial_rx_positions=np.array(r.geometry.rx_positions, copy=True),
            initial_resources=resources,
        )
    if scheme == "proposed":
        return run_joint_algorithm(
            cfg, rng=rng, verbose=False,
            position_update_mode="fp_aware", fp_aware_candidate_iters=2,
            initial_endpoints=np.array(r.geometry.endpoints, copy=True),
            initial_resources=resources,
        )
    raise ValueError(f"Unknown scheme: {scheme}")


def iteration_count(scheme: str, result: Any) -> int:
    return int(result.outer_iterations if scheme == "proposed" else result.iterations)


def make_cfg_at_omega(base: Any, omega: float, weight_maker: Callable) -> Any:
    from dataclasses import replace
    wd, wu, ws = weight_maker(base, omega)
    cfg = replace(base, weight_dl=wd, weight_ul=wu, weight_s=ws)
    cfg.validate()
    return cfg


def consider(best: dict, candidate: Candidate) -> bool:
    """Retain only converged finite candidates, comparing TRUE target-weight WSR."""
    if not candidate.converged or not np.isfinite(candidate.wsr):
        return False
    key = (candidate.scheme, candidate.omega)
    old = best.get(key)
    if old is None or candidate.wsr > old.wsr:
        best[key] = candidate
        return True
    return False


def simple_row(candidate: Candidate, cold_starts: int, warm_attempts: int, runtime: float) -> dict:
    p = candidate.result.performance
    dl, ul, sen = map(lambda a: float(np.sum(a)), (p.rate_dl, p.rate_ul, p.rate_s))
    row = {
        "omega_comm": candidate.omega,
        "omega_sensing": 1.0 - candidate.omega,
        "scheme": candidate.scheme,
        "scheme_label": LABELS[candidate.scheme],
        "sum_rate_dl": dl,
        "sum_rate_ul": ul,
        "sum_rate_comm": dl + ul,
        "sum_rate_sensing": sen,
        "weighted_sum_rate": candidate.wsr,
        "best_seed": candidate.seed,
        "iterations": candidate.iterations,
        "runtime_s": runtime,
        "n_starts": cold_starts + warm_attempts,
        "converged": True,
        "source_type": candidate.source_type,
        "donor_omega": candidate.donor_omega if candidate.donor_omega is not None else "",
        "cold_starts": cold_starts,
        "warm_attempts": warm_attempts,
    }
    return row


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def candidate_row(pass_name: str, cand: Candidate, status: str) -> dict:
    p = cand.result.performance
    return {
        "pass_name": pass_name,
        "omega_comm": cand.omega,
        "scheme": cand.scheme,
        "source_type": cand.source_type,
        "donor_omega": "" if cand.donor_omega is None else cand.donor_omega,
        "seed": cand.seed,
        "iterations": cand.iterations,
        "converged": cand.converged,
        "wsr": cand.wsr,
        "rate_comm": float(np.sum(p.rate_dl) + np.sum(p.rate_ul)),
        "rate_sensing": float(np.sum(p.rate_s)),
        "initial_wsr": "" if cand.initial_wsr is None else cand.initial_wsr,
        "runtime_s": cand.runtime_s,
        "status": status,
    }


def print_summary(best: dict, weights: tuple[float, ...]) -> None:
    print("\nFinal coverage and trend checks:")
    for scheme in SCHEMES:
        rows = [best[(scheme, w)] for w in weights if (scheme, w) in best]
        print(f"  {scheme}: {len(rows)}/{len(weights)} converged points", flush=True)
        if len(rows) < 2:
            continue
        comm = np.array([float(np.sum(c.result.performance.rate_dl) + np.sum(c.result.performance.rate_ul)) for c in rows])
        sense = np.array([float(np.sum(c.result.performance.rate_s)) for c in rows])
        if not np.all(np.diff(comm) >= -1e-7) or not np.all(np.diff(sense) <= 1e-7):
            print("    WARNING: rate monotonicity not fully satisfied; check candidate quality.")


def scan(
    weights: tuple[float, ...],
    schemes: tuple[str, ...],
    cfg_factory: Callable[[float], Any],
    cold_runner: Callable[[str, Any, float], list[Candidate]],
    warm_runner: Callable[[str, Any, Candidate], Any],
    evaluate_donor: Callable[[Candidate, Any], float],
    validate: Callable[[Any, Any], None],
    output: Path,
) -> dict:
    """Injectable engine: permits complete logic testing with synthetic solvers."""
    best: dict[tuple[str, float], Candidate] = {}
    logs: list[dict] = []
    errors: list[dict] = []
    stats: dict[tuple[str, float], dict] = {
        (s, w): {"cold": 0, "warm": 0, "runtime": 0.0}
        for s in schemes for w in weights
    }
    cfgs = {w: cfg_factory(w) for w in weights}

    def checkpoint() -> None:
        rows = [simple_row(best[s, w], stats[s, w]["cold"], stats[s, w]["warm"],
                           stats[s, w]["runtime"])
                for w in weights for s in schemes if (s, w) in best]
        write_csv(output / "stage10_tradeoff.csv", BASE_FIELDS + EXTRA_FIELDS, rows)
        write_csv(output / "candidate_attempts.csv", CANDIDATE_FIELDS, logs)
        write_csv(output / "run_errors.csv", ("pass_name", "scheme", "omega_comm", "reason"), errors)
        # npz for compatibility; only converged solutions enter plot data
        np.savez(output / "stage10_tradeoff.npz",
                 omega_comm=np.array([r["omega_comm"] for r in rows]),
                 scheme=np.array([r["scheme"] for r in rows]),
                 sum_rate_comm=np.array([r["sum_rate_comm"] for r in rows]),
                 sum_rate_sensing=np.array([r["sum_rate_sensing"] for r in rows]),
                 weighted_sum_rate=np.array([r["weighted_sum_rate"] for r in rows]))

    checkpoint()
    # Phase 1: original cold starts, all weights, all schemes.
    for w in weights:
        cfg = cfgs[w]
        for s in schemes:
            print(f"[COLD] omega={w:.6g} {s}", flush=True)
            try:
                starts = cold_runner(s, cfg, w)
            except Exception as exc:
                errors.append({"pass_name": "cold", "scheme": s, "omega_comm": w, "reason": repr(exc)})
                checkpoint()
                # Nonconvergence may return empty starts. Never silently mask code errors.
                raise
            for cand in starts:
                validate(cand.result, cfg)
                stats[s, w]["cold"] += 1
                stats[s, w]["runtime"] += cand.runtime_s
                improved = consider(best, cand)
                logs.append(candidate_row("cold", cand, "not_converged" if not cand.converged else ("selected" if improved else "not_selected")))
            if not any((s, w) == key for key in best):
                print(f"[WARN] {s} at {w:g}: no converged cold start", flush=True)
            checkpoint()

    # Phases 2 & 3: always adjacent weight, and within the SAME scheme.
    for phase, traversal in (
        ("forward", tuple(range(1, len(weights)))),
        ("backward", tuple(range(len(weights) - 2, -1, -1))),
    ):
        for idx in traversal:
            w = weights[idx]
            donor_w = weights[idx - 1] if phase == "forward" else weights[idx + 1]
            cfg = cfgs[w]
            for s in schemes:
                donor = best.get((s, donor_w))
                if donor is None:
                    print(f"[SKIP] {phase} {s} omega={w:g}: donor {donor_w:g} missing", flush=True)
                    continue
                print(f"[{phase.upper()}] {s}: {donor_w:g} -> {w:g}", flush=True)
                stats[s, w]["warm"] += 1
                start = perf_counter()
                try:
                    initial_wsr = evaluate_donor(donor, cfg)
                    result = warm_runner(s, cfg, donor)
                    elapsed = perf_counter() - start
                    validate(result, cfg)
                    candidate = Candidate(
                        result=result, scheme=s, omega=w, source_type=phase,
                        seed=donor.seed, iterations=iteration_count(s, result),
                        runtime_s=elapsed, initial_wsr=initial_wsr, donor_omega=donor_w,
                    )
                    stats[s, w]["runtime"] += elapsed
                    improved = consider(best, candidate)
                    logs.append(candidate_row(phase, candidate, "not_converged" if not candidate.converged else ("selected" if improved else "not_selected")))
                    print(f"  init={initial_wsr:.6f}, final={candidate.wsr:.6f}, "
                          f"converged={candidate.converged}, selected={improved}", flush=True)
                    if candidate.wsr < initial_wsr - 1e-7 * max(1.0, abs(initial_wsr)):
                        print("  [WARN] Warm run decreased the target-weight WSR "
                              "from its feasible initial state; investigate solver monotonicity.", flush=True)
                except Exception as exc:
                    errors.append({"pass_name": phase, "scheme": s, "omega_comm": w, "reason": repr(exc)})
                    checkpoint()
                    raise  # actual numerical/program errors should not be hidden
                checkpoint()
    checkpoint()
    print_summary(best, weights)
    missing = [{"omega_comm": w, "scheme": s, "reason": "no converged candidates"}
               for w in weights for s in schemes if (s, w) not in best]
    write_csv(output / "stage10_failures.csv", ("omega_comm", "scheme", "reason"), missing)
    dominated = []
    for s in schemes:
        available = [(w, best[s, w]) for w in weights if (s, w) in best]
        for w1, c1 in available:
            r1 = c1.result.performance
            x1 = float(np.sum(r1.rate_dl) + np.sum(r1.rate_ul))
            y1 = float(np.sum(r1.rate_s))
            for w2, c2 in available:
                if w1 == w2:
                    continue
                r2 = c2.result.performance
                x2 = float(np.sum(r2.rate_dl) + np.sum(r2.rate_ul))
                y2 = float(np.sum(r2.rate_s))
                if (x2 >= x1 - 1e-8 and y2 >= y1 - 1e-8 and
                        (x2 > x1 + 1e-8 or y2 > y1 + 1e-8)):
                    dominated.append({"scheme": s, "omega_comm": w1,
                                      "dominated_by_omega": w2,
                                      "rate_comm": x1, "rate_sensing": y1})
                    break
    write_csv(output / "pareto_dominated_points.csv",
              ("scheme", "omega_comm", "dominated_by_omega", "rate_comm", "rate_sensing"), dominated)
    if dominated:
        print(f"[WARN] {len(dominated)} dominated points remain. "
              "Warm starts do not guarantee global optimality.", flush=True)
    print(f"Completed {len(best)}/{len(schemes)*len(weights)} final points; "
          f"results: {output}", flush=True)
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Validate YAML only, no optimizer")
    parser.add_argument("--smoke", action="store_true", help="3 weights, 1 cold start, short iterations; diagnostic only")
    args = parser.parse_args()
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    values = tuple(float(x) for x in raw["tradeoff"]["omega_comm"])
    if not values or any(not (0 < w < 1) for w in values) or len(set(values)) != len(values):
        raise ValueError("Expected unique 0 < omega_comm < 1 weights")
    weights = tuple(sorted(values))
    print(f"Stage 10 bidirectional warm-start: weights={weights}", flush=True)
    if args.dry_run:
        print(f"RCS m2: {raw['channel'].get('rcs_m2')}; "
              f"BS noise W: {raw['noise']['sigma_bs2']}; "
              f"inner/outer tolerance: {raw['algorithm']['tol_inner']}/"
              f"{raw['algorithm']['tol_outer']}")
        print("Dry-run passed (YAML only, no optimization).")
        return

    from dataclasses import replace
    from config import load_config
    from metrics import compute_performance
    from utils import check_resource_constraints
    from stage8 import make_scene_config, load_stage8_experiment_config
    from experiments.run_stage10_tradeoff import make_group_weights, run_one_scheme

    base_cfg = load_config(CONFIG_PATH)
    exp_cfg = load_stage8_experiment_config(CONFIG_PATH)
    scene_cfg = make_scene_config(
        base_cfg, scene_seed=int(raw["tradeoff"]["scene_seed"]),
        position_jitter=exp_cfg.position_jitter,
        randomize_fading=exp_cfg.fading_enabled,
        fading_model=exp_cfg.fading_model,
    )
    if exp_cfg.fading_enabled:
        raise RuntimeError("This Stage 10 sweep requires a fixed scene, fading off.")
    if args.smoke:
        weights = (0.005, 0.3, 0.9)
        scene_cfg = replace(scene_cfg,
                            max_inner_iter=min(5, scene_cfg.max_inner_iter),
                            max_outer_iter=min(3, scene_cfg.max_outer_iter))
        output = OUTPUT_DIR / "smoke_not_for_paper"
    else:
        output = OUTPUT_DIR

    def cfg_factory(w):
        return make_cfg_at_omega(scene_cfg, w, make_group_weights)

    def cold_runner(s, cfg, omega):
        # Reuse Stage 10 original multi-start rules (B2=5, B3=1, B6=3, Proposed=5).
        ms = run_one_scheme(
            scheme=s, cfg=cfg, experiment_cfg=exp_cfg,
            scene_seed=int(raw["tradeoff"]["scene_seed"]),
            require_convergence=False, formal_multistart=not args.smoke,
        )
        return [Candidate(start.result, s, omega,
                          "cold", start.seed, start.iterations, start.runtime_s)
                for start in ms.starts]

    def evaluate_donor(donor, cfg):
        # Return the true objective on the NEW weight, before any update.
        check_resource_constraints(donor.result.resources, cfg)
        value = float(compute_performance(
            donor.result.channels, donor.result.resources, cfg).weighted_sum_rate)
        if not np.isfinite(value):
            raise ValueError("Transferred-state WSR not finite")
        return value

    def validate(result, cfg):
        check_resource_constraints(result.resources, cfg)
        true_wsr = float(compute_performance(result.channels, result.resources, cfg).weighted_sum_rate)
        if not np.isfinite(true_wsr) or not np.isclose(
            true_wsr, float(result.performance.weighted_sum_rate), rtol=1e-8, atol=1e-9
        ):
            raise ValueError("Returned WSR does not match true objective at target weight")

    print("[INFO] New result directory, original Stage 10 CSV/images unchanged.", flush=True)
    print("[INFO] Cold-start budgets retain original scheme-specific starts.", flush=True)
    start_time = perf_counter()
    best = scan(weights, SCHEMES, cfg_factory, cold_runner,
                lambda s, cfg, donor: run_warm_scheme(s, cfg, donor, exp_cfg),
                evaluate_donor, validate, output)
    print(f"Total time {perf_counter()-start_time:.1f} s", flush=True)
    if not args.smoke:
        print("Plot: python -u experiments/plot_stage10_bidirectional.py", flush=True)


if __name__ == "__main__":
    main()
