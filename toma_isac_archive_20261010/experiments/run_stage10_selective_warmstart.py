"""Stage 10: selective warm-start repair of a few folded points.

This is a DIAGNOSTIC, not a new optimization method or a smoothed Pareto front.
- Reference 14-point CSV is left unchanged.
- Reference CSV contains rates, NOT complete geometry/beamformers.
- Recreates only the necessary source states using each original best seed.
- Replays one cold target seed under CURRENT solver for version-consistent comparison.
- Refines only named targets, verifies full-state feasibility, selects highest
  TARGET-CONVERGED true WSR (or retains historical reference when higher).
- Outputs to results/stage10/selective_warmstart_tol1e4, never overwrites Stage 10.

Run at project root:
  python -u experiments/run_stage10_selective_warmstart.py --dry-run
  python -u experiments/run_stage10_selective_warmstart.py --only b6:0.3 proposed:0.5
  python -u experiments/run_stage10_selective_warmstart.py --all
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from config import load_config
from stage8 import make_scene_config, load_stage8_experiment_config
from experiments import run_stage10_pool_refinement as pool
from experiments.run_stage10_tradeoff import load_tradeoff_config

# Target -> source weights, within the SAME scheme. Explicit: avoids full-grid warm starts.
PLAN = {
    'b2': {0.005: (0.001,), 0.01: (0.001,)},
    'b3': {0.6: (0.5,), 0.8: (0.5,), 0.9: (0.5,)},
    'b6': {0.3: (0.2, 0.005)},
    'proposed': {0.05: (0.005, 0.1), 0.5: (0.005, 0.4), 0.6: (0.3, 0.5)},
}
SCHEME_ORDER = ('b2', 'b3', 'b6', 'proposed')
OUT_DEFAULT = ROOT / 'results' / 'stage10' / 'selective_warmstart_tol1e4'
CONFIG_PATH = ROOT / 'configs' / 'stage10_tradeoff.yaml'
REF_DEFAULT = ROOT / 'results' / 'stage10' / 'cold_wide14_tol1e4' / 'stage10_tradeoff.csv'
REF_FALLBACK = ROOT / 'reference' / 'stage10_tradeoff_14point_reference.csv'
RESULT_FIELDS = ['omega_comm','omega_sensing','scheme','scheme_label','sum_rate_dl',
                 'sum_rate_ul','sum_rate_comm','sum_rate_sensing','weighted_sum_rate',
                 'best_seed','iterations','runtime_s','n_starts','converged',
                 'selected_kind','selected_origin','reference_wsr','reference_solver_verified']
ATTEMPT_FIELDS = ['role','scheme','omega_comm','seed','origin_omega','status','converged',
                  'iterations','stop_reason','sum_rate_dl','sum_rate_ul',
                  'sum_rate_comm','sum_rate_sensing','weighted_sum_rate',
                  'runtime_s','state_file','source_reweight_wsr','difference_vs_reference','error']
AUDIT_FIELDS = ['scheme','omega_comm','historical_wsr','replayed_cold_wsr',
                'best_warm_wsr','best_warm_origin','selected_wsr','selected_kind',
                'improvement_vs_historical','improvement_vs_current_cold',
                'current_cold_converged','warm_converged_count','reference_is_historical']


def same_w(a, b) -> bool:
    return abs(float(a)-float(b)) < 1e-10


def read_csv(path: Path) -> list[dict]:
    with path.open('r', newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def weights_cfg():
    base = load_config(CONFIG_PATH)
    experiment = load_stage8_experiment_config(CONFIG_PATH)
    tradeoff = load_tradeoff_config(CONFIG_PATH)
    if (base.k_dl, base.j_ul, base.l_target, base.r_sensing) != (5, 5, 5, 4):
        raise RuntimeError('Expected K=J=L=5, R_s=4 in current Stage10.')
    if experiment.fading_enabled:
        raise RuntimeError('Diagnostic requires fading off (same physical scene).')
    if not (np.isclose(base.tol_inner,1e-4) and np.isclose(base.tol_outer,1e-4)):
        raise RuntimeError('Keep both tolerances at 1e-4 for this test.')
    scene = make_scene_config(base, scene_seed=tradeoff.scene_seed,
                              position_jitter=experiment.position_jitter,
                              randomize_fading=False,
                              fading_model=experiment.fading_model)
    return tradeoff, scene, experiment


def select_reference(path: Path | None):
    if path is not None:
        chosen=path.resolve()
    else:
        chosen=REF_DEFAULT if REF_DEFAULT.is_file() else REF_FALLBACK
    if not chosen.is_file():
        raise FileNotFoundError('Missing 14-point reference CSV. Pass --reference <path>, or '
                                f'copy provided reference to {REF_FALLBACK}.')
    rows=read_csv(chosen)
    fields={'omega_comm','scheme','sum_rate_dl','sum_rate_ul','sum_rate_comm',
            'sum_rate_sensing','weighted_sum_rate','best_seed','converged'}
    if not rows or not fields.issubset(rows[0]):
        raise ValueError('Reference is not a Stage 10 tradeoff CSV.')
    lookup={}
    for row in rows:
        key=(row['scheme'],round(float(row['omega_comm']),10))
        if key in lookup:
            raise ValueError(f'Duplicate reference row {key}')
        if row['converged'].strip().lower()!='true':
            raise ValueError(f'Reference contains a nonconverged point {key}')
        lookup[key]=row
    for s, plans in PLAN.items():
        for w, sources in plans.items():
            for v in (w, *sources):
                if (s,round(v,10)) not in lookup:
                    raise ValueError(f'Missing reference for {s} omega={v}')
    if len(set(float(r['omega_comm']) for r in rows)) != 14:
        raise ValueError('The input is not the expected 14-weight sweep.')
    return chosen, rows, lookup


def parse_targets(tokens, all_points=False):
    if all_points or not tokens:
        return [(s,w) for s in SCHEME_ORDER for w in PLAN[s]]
    result=[]
    for tok in tokens:
        try:
            s,w=tok.split(':',1); w=float(w)
        except ValueError as exc:
            raise ValueError('Use --only b6:0.3 proposed:0.5 etc.') from exc
        if s not in PLAN or not any(same_w(w,x) for x in PLAN[s]):
            raise ValueError(f'Not a configured repair target: {tok}')
        key=(s,next(x for x in PLAN[s] if same_w(x,w)))
        if key not in result:
            result.append(key)
    return result


def key_for(s, w):
    return (s,round(float(w),10))


def fingerprint(ref: Path, scene, experiment):
    codepaths=['algorithm.py','baselines.py','ma_baseline.py','inner_solver.py',
               'metrics.py','config.py','stage8.py',
               'experiments/run_stage10_tradeoff.py',
               'experiments/run_stage10_pool_refinement.py',
               'experiments/run_stage10_selective_warmstart.py',
               'configs/stage10_tradeoff.yaml']
    spec=dict(ref_sha256=sha256(ref.read_bytes()).hexdigest(),
              physical_scene=repr(scene),
              ma_params=[experiment.ma_region_size, experiment.ma_min_spacing,
                         experiment.ma_initial_step, experiment.ma_initial_jitter_fraction],
              plan={s:{str(w):list(src) for w,src in ps.items()} for s,ps in PLAN.items()},
              code_sha256={name:sha256((ROOT/name).read_bytes()).hexdigest() for name in codepaths})
    return sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest(),spec


def attempt_path(out, role, scheme, omega, source_omega=None):
    label=f'{omega:.4f}'.replace('.','p')
    extra='' if source_omega is None else f'_from{source_omega:.4f}'.replace('.','p')
    return out/'attempts'/f'{role}_{scheme}_w{label}{extra}.json'


def attempt(out, role, scheme, omega, seed, scene, exp, ref_lookup,
            source_path=None, origin_omega=None):
    f=attempt_path(out, role, scheme, omega, origin_omega if role=='warm' else None)
    target=pool.target_cfg(scene,omega)
    if f.is_file():
        record=json.loads(f.read_text(encoding='utf-8'))
        if record['status']=='ok':
            path=out/record['state_file']
            if not path.is_file():
                raise FileNotFoundError(f'Previous checkpoint is missing state: {path}')
            pool.verify_state(path,target,float(record['weighted_sum_rate']))
        print(f'  [RESUME] {role} {scheme} w={omega:g}' +
              (f' from={origin_omega:g}' if origin_omega is not None else ''),flush=True)
        return record
    timer=perf_counter()
    src_score=None
    warm=None
    if source_path is not None:
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        src_score=pool.verify_state(source_path,target)
        geometry,resources,_=pool.read_state(source_path)
        warm=(geometry,resources)
    print(f'  [RUN] {role} {scheme} w={omega:g}, seed={seed}' +
          (f', source={origin_omega:g}, reweighted={src_score:.6f}' if source_path else ''),flush=True)
    try:
        result=pool.run_solver(scheme,target,exp,int(seed),warm=warm)
        data=pool.extract_row(scheme,omega,int(seed),result,role,'',
                              omega if origin_omega is None else origin_omega)
        data['status']='ok' if result.converged else 'not_converged'
        data['role']=role
        data['source_reweight_wsr']=src_score if src_score is not None else ''
        data['difference_vs_reference']=float(data['weighted_sum_rate'])-float(ref_lookup[key_for(scheme,omega)]['weighted_sum_rate'])
        data['runtime_s']=round(perf_counter()-timer,3)
        if result.converged:
            fstate=Path('states')/(f.stem+'.npz')
            pool.save_state(out/fstate,result)
            data['state_file']=str(fstate)
            pool.verify_state(out/fstate,target,float(data['weighted_sum_rate']))
        else:
            data['state_file']=''
        pool.json_write(f,data)
        print(f'    WSR={data["weighted_sum_rate"]:.6f}, conv={result.converged}, '
              f'C={data["sum_rate_comm"]:.4f}, S={data["sum_rate_sensing"]:.4f}',flush=True)
        return data
    except Exception as exc:
        pool.json_write(f,dict(role=role,scheme=scheme,omega_comm=omega,seed=int(seed),
                    origin_omega=origin_omega,status='error',error=f'{type(exc).__name__}: {exc}'))
        raise


def ordered_tasks(targets):
    sources=set(); cold=set(targets); warm=[]
    for s,w in targets:
        for origin in PLAN[s][w]:
            sources.add((s,origin))
            warm.append((s,w,origin))
    return sorted(sources,key=lambda k:(SCHEME_ORDER.index(k[0]),k[1])),cold,warm


def summarize(out, targets, ref_rows):
    results=[]; audit=[]; all_attempts=[]
    for path in sorted((out/'attempts').glob('*.json')):
        record=json.loads(path.read_text(encoding='utf-8'))
        all_attempts.append(record)
    def pick(role,s,w,origin=None):
        path=attempt_path(out,role,s,w,origin if role=='warm' else None)
        if not path.is_file(): return None
        value=json.loads(path.read_text(encoding='utf-8'))
        return value if value.get('status')=='ok' and value.get('converged') else None
    replace_records={}
    for s,w in targets:
        historical=next(r for r in ref_rows if r['scheme']==s and same_w(r['omega_comm'],w))
        old=float(historical['weighted_sum_rate'])
        replay=pick('target',s,w)
        warms=[pick('warm',s,w,src) for src in PLAN[s][w]]
        warms=[x for x in warms if x is not None]
        candidates=([replay] if replay is not None else [])+warms
        best=max(candidates,key=lambda x:float(x['weighted_sum_rate'])) if candidates else None
        bestwarm=max(warms,key=lambda x:float(x['weighted_sum_rate'])) if warms else None
        # Historical CSV has NO state; leave it as historical if current attempts do not beat it.
        selected=best if best is not None and float(best['weighted_sum_rate'])>old+1e-8 else None
        if selected is not None:
            r=dict(historical)
            for field in ('sum_rate_dl','sum_rate_ul','sum_rate_comm','sum_rate_sensing','weighted_sum_rate'):
                r[field]=float(selected[field])
            r['best_seed']=int(selected['seed'])
            r['iterations']=int(selected['iterations'])
            r['converged']='True'
            # Keep n_starts as historical metadata; new trials appear in attempts.csv.
            r['runtime_s']=selected.get('runtime_s',0)
            r['selected_kind']=selected['role']
            r['selected_origin']=selected['origin_omega']
            r['reference_wsr']=old
            r['reference_solver_verified']='False'  # source CSV contains only summary data
            replace_records[key_for(s,w)]=r
        audit.append(dict(scheme=s,omega_comm=w,historical_wsr=old,
                          replayed_cold_wsr=replay['weighted_sum_rate'] if replay else '',
                          best_warm_wsr=bestwarm['weighted_sum_rate'] if bestwarm else '',
                          best_warm_origin=bestwarm['origin_omega'] if bestwarm else '',
                          selected_wsr=selected['weighted_sum_rate'] if selected else old,
                          selected_kind=selected['role'] if selected else 'historical_unchanged',
                          improvement_vs_historical=float(selected['weighted_sum_rate'])-old if selected else 0.0,
                          improvement_vs_current_cold=(float(best['weighted_sum_rate'])-float(replay['weighted_sum_rate']))
                          if best and replay else '',
                          current_cold_converged=bool(replay),warm_converged_count=len(warms),
                          reference_is_historical=True))
    for orig in ref_rows:
        key=key_for(orig['scheme'],orig['omega_comm'])
        row=dict(orig)
        row.update(selected_kind='historical_unchanged',selected_origin='',
                   reference_wsr=orig['weighted_sum_rate'],reference_solver_verified='False')
        results.append(replace_records.get(key,row))
    pool.csv_write(out/'attempts.csv',all_attempts,ATTEMPT_FIELDS)
    pool.csv_write(out/'targeted_comparison.csv',audit,AUDIT_FIELDS)
    pool.csv_write(out/'stage10_tradeoff_targeted.csv',results,RESULT_FIELDS)
    print(f'\n[REPORT] targets={len(audit)}, updated={len(replace_records)}, output={out}',flush=True)
    for r in audit:
        print(f"  {r['scheme']:9} w={r['omega_comm']:.3f}: "
              f"old={r['historical_wsr']:.6f}, warm={r['best_warm_wsr']}, "
              f"chosen={r['selected_wsr']:.6f} ({r['selected_kind']})")
    return audit,results


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference',type=Path,default=None,help='Original 14-point tradeoff CSV')
    ap.add_argument('--output',type=Path,default=OUT_DEFAULT)
    ap.add_argument('--only',nargs='+',default=[],metavar='SCHEME:WEIGHT',
                    help='e.g. --only b6:0.3 proposed:0.5')
    ap.add_argument('--all',action='store_true',help='Run all 9 defined folded targets')
    ap.add_argument('--phase',choices=('sources','targets','warm','report','all'),default='all')
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    try:
        chosen,original,ref_lookup=select_reference(args.reference)
        targets=parse_targets(args.only,args.all)
    except (ValueError,FileNotFoundError) as err:
        ap.error(str(err))
    tradeoff,scene,exp=weights_cfg()
    if not all(any(same_w(w,q) for q in tradeoff.omega_comm) for _,w in targets):
        raise RuntimeError('Target weights not in YAML config')
    out=args.output.resolve()
    stamp,spec=fingerprint(chosen,scene,exp)
    manifest=out/'manifest.json'
    if manifest.is_file() and json.loads(manifest.read_text(encoding='utf-8')).get('sha256')!=stamp:
        raise RuntimeError('The output directory belongs to another code/config/reference revision. '
                           f'Use --output a NEW directory, do not mix states: {out}')
    source_tasks,cold_tasks,warm_tasks=ordered_tasks(targets)
    print(f'Stage10 selective warm-start; original reference: {chosen}')
    print(f'  scene={tradeoff.scene_seed}, tol=1e-4, selected targets={targets}')
    for s,w in targets:
        r=ref_lookup[key_for(s,w)]
        print(f'  {s:9} w={w:g} OLD F={float(r["weighted_sum_rate"]):.6f}, '
              f'origin(s)={PLAN[s][w]}')
    print(f'  {len(source_tasks)} source rebuilds, {len(cold_tasks)} target same-seed replays, '
          f'{len(warm_tasks)} warm-start refinements. All cached when complete.')
    print('  NOTE: old CSV is a historical numerical reference; B6 algorithm has since been updated.')
    if args.dry_run: return
    pool.json_write(manifest,dict(sha256=stamp,parameters=spec))
    if args.phase in ('sources','all'):
        for s,w in source_tasks:
            ref=ref_lookup[key_for(s,w)]
            attempt(out,'source',s,w,int(ref['best_seed']),scene,exp,ref_lookup)
    if args.phase in ('targets','all'):
        for s,w in sorted(cold_tasks,key=lambda k:(SCHEME_ORDER.index(k[0]),k[1])):
            ref=ref_lookup[key_for(s,w)]
            attempt(out,'target',s,w,int(ref['best_seed']),scene,exp,ref_lookup)
    if args.phase in ('warm','all'):
        for s,w,origin in warm_tasks:
            p=attempt_path(out,'source',s,origin)
            if not p.is_file():
                raise FileNotFoundError(f'Missing source {p}, run --phase sources first.')
            meta=json.loads(p.read_text(encoding='utf-8'))
            if meta.get('status')!='ok':
                print(f'[SKIP] source not converged {s} omega={origin}',flush=True)
                continue
            state=out/meta['state_file']
            attempt(out,'warm',s,w,int(meta['seed']),scene,exp,ref_lookup,source_path=state,origin_omega=origin)
    if args.phase in ('report','all'):
        summarize(out,targets,original)


if __name__=='__main__':
    main()
