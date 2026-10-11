"""Stage 10 tolerance sensitivity: B6 & Proposed, independent cold starts only.

Uses the same physical scene and initialization APIs as run_stage10_cold20.py.
Default fast diagnostic: 8 weights x 2 schemes x 1 historical winning seed,
with tol=1e-5. Neither warm starts nor cross-weight state transfer are used.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from stage8 import load_stage8_experiment_config, make_ma_start_seeds, make_scene_config
from experiments.run_stage10_cold20 import make_group_weights, load_tradeoff_config
from algorithm import run_joint_algorithm
from ma_baseline import run_ma_fp

CFG_PATH = ROOT / 'configs' / 'stage10_tradeoff_cold20.yaml'
BASE_CSV = ROOT / 'results' / 'stage10' / 'cold_wide20_dl114_tol1e4' / 'stage10_tradeoff.csv'
OUT_ROOT = ROOT / 'results' / 'stage10' / 'tolerance_comparison_cold'
SCHEMES = ('b6', 'proposed')
QUICK_WEIGHTS = (0.001, 0.05, 0.1, 0.2, 0.4, 0.5, 0.55, 0.8)


def read_baseline(path: Path, weights: tuple[float, ...]):
    if not path.exists():
        raise FileNotFoundError(f'Original cold20 CSV not found: {path}')
    result = {}
    with path.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            if row['scheme'] not in SCHEMES:
                continue
            k = (round(float(row['omega_comm']), 12), row['scheme'])
            if k in result:
                raise ValueError(f'duplicate original baseline row: {k}')
            if row.get('converged', '').lower() != 'true':
                raise ValueError(f'baseline row not converged: {k}')
            r_c, r_s, w = (float(row[z]) for z in ('sum_rate_comm', 'sum_rate_sensing', 'omega_comm'))
            expected = w*r_c/10 + (1-w)*r_s/5
            if not np.isclose(expected, float(row['weighted_sum_rate']), atol=1e-7, rtol=1e-7):
                raise ValueError(f'baseline WSR inconsistent: {k}')
            result[k] = row
    expected_keys = {(round(w, 12), s) for w in weights for s in SCHEMES}
    if expected_keys != set(result):
        raise RuntimeError(f'Baseline must have 20 x 2 scheme rows. Missing: {sorted(expected_keys - set(result))}')
    return result


def original_seeds(scheme, expcfg, scene_seed):
    if scheme == 'proposed':
        return tuple(int(x) for x in expcfg.multi_start_seeds)
    return make_ma_start_seeds(
        scene_seed=scene_seed,
        fixed_upa_seed=int(expcfg.fixed_upa_seed),
        n_starts=int(expcfg.ma_n_starts),
    )


def run_one(scheme, cfg, expcfg, seed):
    rng = np.random.default_rng(seed)
    if scheme == 'proposed':
        r = run_joint_algorithm(
            cfg, rng=rng, verbose=False,
            position_update_mode='fp_aware', fp_aware_candidate_iters=2,
        )
        n_iter = int(r.outer_iterations)
    elif scheme == 'b6':
        r = run_ma_fp(
            cfg, rng=rng, verbose=False,
            region_size=expcfg.ma_region_size,
            min_spacing=expcfg.ma_min_spacing,
            initial_step=expcfg.ma_initial_step,
            randomize_initial_geometry=(seed != int(expcfg.fixed_upa_seed)),
            initial_jitter_fraction=expcfg.ma_initial_jitter_fraction,
        )
        n_iter = int(r.iterations)
    else:
        raise ValueError(scheme)
    rates = r.performance
    if not np.isfinite(float(rates.weighted_sum_rate)):
        raise FloatingPointError('Optimization returned non-finite WSR')
    return {
        'converged': bool(r.converged),
        'stop_reason': str(getattr(r, 'stop_reason', 'unknown')),
        'iterations': n_iter,
        'sum_rate_dl': float(np.sum(rates.rate_dl)),
        'sum_rate_ul': float(np.sum(rates.rate_ul)),
        'sum_rate_comm': float(np.sum(rates.rate_dl) + np.sum(rates.rate_ul)),
        'sum_rate_sensing': float(np.sum(rates.rate_s)),
        'weighted_sum_rate': float(rates.weighted_sum_rate),
    }


def group_seeds(scheme, w, expcfg, scene_seed, baseline, mode):
    possible = original_seeds(scheme, expcfg, scene_seed)
    if mode == 'best-seed':
        seed = int(baseline[(round(w, 12), scheme)]['best_seed'])
        if seed not in possible:
            raise ValueError(f'Baseline seed {seed} not in known seed list for {scheme}; code/config might differ')
        return (seed,)
    return possible


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        wr = csv.DictWriter(f, fieldnames=columns)
        wr.writeheader()
        wr.writerows(rows)


def key_path(attemptdir, tolerance, omega, scheme, seed):
    return attemptdir / f'tol{tolerance:.0e}_w{omega:.10g}_{scheme}_seed{seed}.json'


def run_and_collect(args):
    cfg = load_config(CFG_PATH)
    expcfg = load_stage8_experiment_config(CFG_PATH)
    tradeoff = load_tradeoff_config(CFG_PATH)
    all_weights = tuple(tradeoff.omega_comm)
    if len(all_weights) != 20:
        raise RuntimeError(f'Expected 20 fixed cold weights, got {len(all_weights)}')
    if args.full_20:
        selected = all_weights
    elif args.weights:
        selected = tuple(float(x) for x in args.weights)
    else:
        selected = QUICK_WEIGHTS
    allowed = {round(w,12) for w in all_weights}
    if len(set(map(lambda w: round(w,12), selected))) != len(selected) or any(round(w,12) not in allowed for w in selected):
        raise ValueError('All selected weights must be distinct and come from the original 20-point scan.')
    selected = tuple(sorted(selected))
    tols = tuple(sorted(set(float(t) for t in args.tols), reverse=True))
    if any(not 0 < t <= 1e-3 for t in tols):
        raise ValueError('Test tolerances must be positive and at most 1e-3.')
    if not np.allclose(cfg.sigma_dl2, 3.981071705534969e-15, rtol=1e-6, atol=0.0) or not np.isclose(cfg.sigma_bs2, 3.981071705534969e-15, rtol=1e-6, atol=0.0):
        raise RuntimeError('Expected DL and BS receiver noise both -114 dBm; stop to prevent mixed experiments.')
    if (cfg.k_dl, cfg.j_ul, cfg.l_target, cfg.r_sensing) != (5, 5, 5, 4):
        raise RuntimeError('Expected K=J=L=5, Rs=4.')
    if expcfg.fading_enabled:
        raise RuntimeError('Expected fixed LoS fading.enabled=false.')
    baseline = read_baseline(BASE_CSV, all_weights)
    output = OUT_ROOT / args.mode
    attemptdir = output / 'attempts'
    plan = [(tol, w, scheme, seed)
            for tol in tols for w in selected for scheme in SCHEMES
            for seed in group_seeds(scheme, w, expcfg, tradeoff.scene_seed, baseline, args.mode)]
    if args.dry_run:
        print(f'Original 1e-4 baseline: {BASE_CSV}')
        print('Initialization: independent cold starts, same original RNG seed(s); no warm start.')
        print(f'Mode: {args.mode}; tolerances={tols}; selected weights={selected}')
        print(f'Planned runs: {len(plan)} (Proposed and B6 only)')
        print(f'Output: {output}')
        return

    # Guard against incompatible reruns with same output path.
    output.mkdir(parents=True, exist_ok=True)
    meta_path = output / 'run_metadata.json'
    meta = {
        'config_sha256': hashlib.sha256(CFG_PATH.read_bytes()).hexdigest(),
        'baseline_sha256': hashlib.sha256(BASE_CSV.read_bytes()).hexdigest(),
        'mode': args.mode,
    }
    if meta_path.exists():
        if json.loads(meta_path.read_text(encoding='utf-8')) != meta:
            raise RuntimeError('Existing output belongs to a different config/baseline. Move it aside before rerunning.')
    else:
        meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    attemptdir.mkdir(exist_ok=True)

    scene = make_scene_config(
        cfg, scene_seed=tradeoff.scene_seed,
        position_jitter=expcfg.position_jitter,
        randomize_fading=expcfg.fading_enabled,
        fading_model=expcfg.fading_model,
    )
    for i,(tol,w,scheme,seed) in enumerate(plan,1):
        fname = key_path(attemptdir,tol,w,scheme,seed)
        if fname.exists():
            print(f'[{i}/{len(plan)}] [CACHED] tol={tol:.0e}, w={w:g}, {scheme}, seed={seed}')
            continue
        wd,wu,ws = make_group_weights(scene,w)
        run_cfg = replace(scene,weight_dl=wd,weight_ul=wu,weight_s=ws,tol_inner=tol,tol_outer=tol)
        run_cfg.validate()
        print(f'[{i}/{len(plan)}] tol={tol:.0e}, w={w:g}, {scheme}, seed={seed} ...', flush=True)
        start = perf_counter()
        # Do not swallow errors or silently claim convergence.
        try:
            data = run_one(scheme,run_cfg,expcfg,seed)
            data['status'] = 'ok'
        except Exception as exc:
            data = {'status':'error','error_type':type(exc).__name__,'error':str(exc),
                    'converged':False,'stop_reason':'exception'}
            print(f'  ERROR {type(exc).__name__}: {exc}')
        data.update({'tol':tol,'omega_comm':w,'scheme':scheme,'seed':seed,'runtime_s':round(perf_counter()-start,3)})
        # Each attempt written once: safe for checkpoint/resume, even if summary CSV is open in Excel.
        fname.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
        if data.get('status')=='ok':
            print('  WSR={:.6f}, iter={}, converged={}, stop={}'.format(data['weighted_sum_rate'],data['iterations'],data['converged'],data['stop_reason']))
    summarize(output, attemptdir, baseline, all_weights, selected, tols, args.mode, expcfg, tradeoff.scene_seed)


def summarize(output, attemptdir, baseline, all_weights, selected, tols, mode, expcfg, scene_seed):
    attempts=[]
    for path in sorted(attemptdir.glob('*.json')):
        try: attempts.append(json.loads(path.read_text(encoding='utf-8')))
        except (ValueError, OSError): continue
    idx={(round(float(a['tol']),12),round(float(a['omega_comm']),12),a['scheme'],int(a['seed'])):a for a in attempts}
    # Keep results from earlier tolerance runs (e.g. 1e-5 first, 1e-6 later).
    all_tols=tuple(sorted({*tols, *(float(a['tol']) for a in attempts)},reverse=True))
    all_selected=tuple(sorted({*selected, *(float(a['omega_comm']) for a in attempts)}))
    summary=[]
    for tol in all_tols:
        for w in all_selected:
            for scheme in SCHEMES:
                base=baseline[(round(w,12),scheme)]
                seeds=group_seeds(scheme,w,expcfg,scene_seed,baseline,mode)
                rows=[idx.get((round(tol,12),round(w,12),scheme,seed)) for seed in seeds]
                finished=[r for r in rows if r is not None]
                successful=[r for r in finished if r.get('status')=='ok' and r.get('converged')]
                best=max(successful,key=lambda r:r['weighted_sum_rate']) if successful else None
                status='complete' if len(finished)==len(seeds) and best is not None else ('no_converged' if len(finished)==len(seeds) else 'partial')
                summary.append({
                    'tol':tol,'omega_comm':w,'scheme':scheme,'mode':mode,'status':status,
                    'expected_starts':len(seeds),'completed_starts':len(finished),
                    'converged_starts':len(successful),'baseline_seed':int(base['best_seed']),
                    'baseline_wsr':float(base['weighted_sum_rate']),
                    'baseline_rc':float(base['sum_rate_comm']),
                    'baseline_rs':float(base['sum_rate_sensing']),
                    'new_seed':(best['seed'] if best else ''),
                    'new_wsr':(best['weighted_sum_rate'] if best else ''),
                    'new_rc':(best['sum_rate_comm'] if best else ''),
                    'new_rs':(best['sum_rate_sensing'] if best else ''),
                    'new_iterations':(best['iterations'] if best else ''),
                    'delta_wsr':(best['weighted_sum_rate']-float(base['weighted_sum_rate']) if best else ''),
                    'new_runtime_s':sum(r.get('runtime_s',0) for r in finished),
                })
    columns=list(summary[0]) if summary else []
    write_csv(output/'comparison.csv',summary,columns)
    print(f'\nComparison CSV: {output / "comparison.csv"}')
    print('Completed / expected groups:',sum(r['status']=='complete' for r in summary),'/',len(summary))
    for t in all_tols:
        for s in SCHEMES:
            items=[r for r in summary if r['tol']==t and r['scheme']==s and r['status']=='complete']
            if items:
                print(f'tol={t:.0e} {s}: {len(items)} points, mean delta WSR={np.mean([x["delta_wsr"] for x in items]):+.6g}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tols',nargs='+',type=float,default=[1e-5],help='One or more values, e.g. 1e-5 1e-6')
    p.add_argument('--weights',nargs='+',type=float,help='Original cold20 weights; default 8 quick representative weights')
    p.add_argument('--full-20',action='store_true',help='Run all 20 original weights (slower)')
    p.add_argument('--mode',choices=['best-seed','all-seeds'],default='best-seed',help='best-seed is fast and isolates tolerance; all-seeds repeats exact original multi-start counts')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    if args.weights and args.full_20: p.error('Choose --weights or --full-20, not both')
    run_and_collect(args)

if __name__=='__main__':
    main()
