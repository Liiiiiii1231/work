"""Stage 10: repair four diagnosed cold20 points by targeted cross-weight warm starts.

- Reference is the saved 20-point COLD-START result, never overwritten.
- Rebuilds only 3 source states because the old CSV does not contain antenna/beam data.
- Refines only 4 predeclared targets; never runs a full weight sweep.
- A valid/CONVERGED warm result replaces its target only if its real WSR increases.
- Records the source's direct feasible reweight separately (not counted as converged).
- Supports --dry-run, --phase sources/warm/report and resumable checkpoints.

At project root (Windows CMD):
  python -u .\\experiments\\run_stage10_targeted20.py --dry-run
  python -u .\\experiments\\run_stage10_targeted20.py
  python -u .\\experiments\\plot_stage10_targeted20.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG = ROOT / 'configs' / 'stage10_tradeoff_cold20.yaml'
REFERENCE = ROOT / 'results' / 'stage10' / 'cold_wide20_dl114_tol1e4' / 'stage10_tradeoff.csv'
OUT = ROOT / 'results' / 'stage10' / 'cold20_targeted_warm_dl114'

# Only four targeted repairs; all schemes share the same objective and physical scene.
# (scheme, target weight, source weight)
PLAN = (
    ('proposed', 0.05, 0.001),
    ('proposed', 0.20, 0.001),
    ('b6',       0.20, 0.005),
    ('b2',       0.20, 0.001),
)
SCHEMES = ('b2', 'b3', 'b6', 'proposed')
AUDIT_COLUMNS = [
    'scheme', 'target_weight', 'source_weight', 'historical_wsr',
    'source_historical_wsr', 'replayed_source_wsr', 'source_replay_gap',
    'source_direct_feasible_wsr', 'warm_final_wsr', 'warm_converged',
    'warm_stop_reason', 'warm_iterations', 'selected_wsr', 'selected_kind',
    'improvement_vs_historical', 'remaining_vs_feasible_source',
]
RESULT_COLUMNS = [
    'omega_comm','omega_sensing','scheme','scheme_label','sum_rate_dl',
    'sum_rate_ul','sum_rate_comm','sum_rate_sensing','weighted_sum_rate',
    'best_seed','iterations','runtime_s','n_starts','converged',
    'selected_kind','source_weight','original_wsr',
]
ATTEMPT_COLUMNS = [
    'role','scheme','omega_comm','source_weight','seed','status','converged',
    'iterations','stop_reason','sum_rate_dl','sum_rate_ul','sum_rate_comm',
    'sum_rate_sensing','weighted_sum_rate','runtime_s','state_file',
    'direct_reweight_wsr','error',
]


def key(scheme: str, weight: float):
    return scheme, round(float(weight), 10)


def read_csv(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing CSV: {path}')
    with path.open('r', newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def save_csv(path: Path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a separate file; never replace an existing source/old-result CSV.
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def physical_config():
    if not CONFIG.is_file():
        raise FileNotFoundError(f'Missing config: {CONFIG}. Install the cold20 package first.')
    with CONFIG.open('r', encoding='utf-8') as handle:
        cfg=yaml.safe_load(handle)
    weights=[float(x) for x in cfg['tradeoff']['omega_comm']]
    if len(weights)!=20 or len(set(weights))!=20:
        raise ValueError('Expected exactly 20 distinct communication weights in cold20 YAML.')
    if any(x not in weights for _,target,source in PLAN for x in (target,source)):
        raise ValueError('Required source/target weight missing from YAML.')
    noise=cfg['noise']
    sigma=3.981071705534969e-15
    if not np.allclose(noise['sigma_dl2'],sigma,rtol=1e-7,atol=0):
        raise ValueError('DL noise is not -114 dBm. Do not mix physical scenes.')
    if not np.isclose(float(noise['sigma_bs2']),sigma,rtol=1e-7,atol=0):
        raise ValueError('BS noise is not -114 dBm.')
    it=cfg['algorithm']
    if not np.isclose(it['tol_inner'],1e-4) or not np.isclose(it['tol_outer'],1e-4):
        raise ValueError('Expected tol_inner=tol_outer=1e-4 for this experiment.')
    if (cfg['system']['K'],cfg['system']['J'],cfg['system']['L'])!=(5,5,5):
        raise ValueError('This WSR verification assumes K=J=L=5.')
    if cfg['experiment'].get('fading',{}).get('enabled',False):
        raise ValueError('Expected fading disabled, matching saved cold20 experiment.')
    return cfg,weights


def reference_rows(weights):
    rows=read_csv(REFERENCE)
    expected={key(s,w) for s in SCHEMES for w in weights}
    items={}
    for r in rows:
        k=key(r['scheme'],r['omega_comm'])
        if k in items: raise ValueError(f'Duplicate result: {k}')
        if r['converged'].lower().strip()!='true':
            raise ValueError(f'Unconverged original result: {k}')
        w=float(r['omega_comm'])
        val=w/10*float(r['sum_rate_comm'])+(1-w)/5*float(r['sum_rate_sensing'])
        if not np.isclose(val,float(r['weighted_sum_rate']),rtol=1e-7,atol=1e-7):
            raise ValueError(f'Original CSV WSR formula error: {k}')
        if not np.isclose(float(r['sum_rate_comm']),float(r['sum_rate_dl'])+float(r['sum_rate_ul']),atol=1e-7):
            raise ValueError(f'Communication total does not match DL+UL: {k}')
        items[k]=r
    if set(items)!=expected:
        raise ValueError(f'Expected 80 original results; missing={expected-set(items)}, extra={set(items)-expected}')
    return rows,items


def validate_manifest():
    fingerprint=dict(config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
                     reference_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
                     planned_targets=[list(p) for p in PLAN],
                     solver_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
                         for f in ('algorithm.py','baselines.py','ma_baseline.py','inner_solver.py',
                                   'outer_solver.py','channels.py','metrics.py',
                                   'experiments/run_stage10_targeted20.py')})
    file=OUT/'run_manifest.json'
    if file.exists():
        if read_json(file)!=fingerprint:
            raise RuntimeError('Different YAML / reference CSV in this output folder. Backup it or use another output directory.')
    else:
        save_json(file,fingerprint)


def construct_scene():
    # Solver imports stay in this function: --dry-run works without CVXPY.
    from config import load_config
    from stage8 import load_stage8_experiment_config, make_scene_config
    base=load_config(CONFIG)
    exp=load_stage8_experiment_config(CONFIG)
    tradeoff, _=physical_config()
    scene=make_scene_config(base,scene_seed=int(tradeoff['tradeoff']['scene_seed']),
                            position_jitter=exp.position_jitter,
                            randomize_fading=False,fading_model=exp.fading_model)
    return scene,exp


def cfg_at(scene,weight):
    # Same weight construction as run_stage10_cold20.py.
    from experiments.run_stage10_cold20 import make_group_weights
    wd,wu,ws=make_group_weights(scene,weight)
    return replace(scene,weight_dl=wd,weight_ul=wu,weight_s=ws)


def run_solver(scheme,cfg,exp,seed,warm=None):
    from algorithm import run_joint_algorithm
    from baselines import run_fixed_toma_fp
    from ma_baseline import run_ma_fp
    rng=np.random.default_rng(int(seed))
    if scheme=='proposed':
        kw={} if warm is None else {'initial_endpoints':warm[0].endpoints,'initial_resources':warm[1]}
        return run_joint_algorithm(cfg,rng=rng,verbose=False,
                                   position_update_mode='fp_aware',fp_aware_candidate_iters=2,**kw)
    if scheme=='b2':
        kw={} if warm is None else {'initial_endpoints':warm[0].endpoints,'initial_resources':warm[1]}
        return run_fixed_toma_fp(cfg,rng=rng,verbose=False,**kw)
    if scheme=='b6':
        kw={} if warm is None else {'initial_tx_positions':warm[0].tx_positions,
                                    'initial_rx_positions':warm[0].rx_positions,
                                    'initial_resources':warm[1]}
        return run_ma_fp(cfg,rng=rng,verbose=False,
                         region_size=exp.ma_region_size,min_spacing=exp.ma_min_spacing,
                         initial_step=exp.ma_initial_step,
                         randomize_initial_geometry=(warm is None and seed!=int(exp.fixed_upa_seed)),
                         initial_jitter_fraction=exp.ma_initial_jitter_fraction,**kw)
    raise ValueError(f'This targeted repair does not run {scheme}.')


def iteration_count(result):
    return int(getattr(result,'outer_iterations',getattr(result,'iterations',0)))


def state_path(role,scheme,target,source):
    return OUT/'states'/f'{role}_{scheme}_to{target:.3f}_from{source:.3f}.npz'


def json_path(role,scheme,target,source):
    return OUT/'attempts'/f'{role}_{scheme}_to{target:.3f}_from{source:.3f}.json'


def save_state(path,result):
    path.parent.mkdir(parents=True,exist_ok=True)
    g,r,p=result.geometry,result.resources,result.performance
    temp=path.with_name(path.stem+'.temp.npz')
    np.savez_compressed(temp,
        endpoints=np.array(g.endpoints,copy=True),
        tx_positions=np.array(g.tx_positions,copy=True),
        rx_positions=np.array(g.rx_positions,copy=True),
        q_matrix=np.array(r.q_matrix,copy=True),
        q_ul=np.array(r.q_ul,copy=True),
        b_ul=np.array(r.b_ul,copy=True),
        u_s=np.array(r.u_s,copy=True),
        rate_dl=np.array(p.rate_dl,copy=True),
        rate_ul=np.array(p.rate_ul,copy=True),
        rate_s=np.array(p.rate_s,copy=True))
    temp.replace(path)


def read_state(path):
    from state import GeometryState,ResourceState
    with np.load(path,allow_pickle=False) as z:
        geom=GeometryState(endpoints=z['endpoints'].copy(),tx_positions=z['tx_positions'].copy(),rx_positions=z['rx_positions'].copy())
        res=ResourceState(q_matrix=z['q_matrix'].copy(),q_ul=z['q_ul'].copy(),b_ul=z['b_ul'].copy(),u_s=z['u_s'].copy())
    return geom,res


def verify_state(path,cfg,expected=None):
    from channels import build_channels
    from metrics import compute_performance
    from utils import check_resource_constraints
    geom,res=read_state(path)
    check_resource_constraints(res,cfg)
    perf=compute_performance(build_channels(geom,cfg),res,cfg)
    value=float(perf.weighted_sum_rate)
    if not np.isfinite(value):
        raise RuntimeError(f'Nonfinite WSR from state: {path}')
    if expected is not None and not np.isclose(value,float(expected),rtol=1e-6,atol=3e-7):
        raise RuntimeError(f'WSR differs between saved state and recorded result: {value} / {expected}')
    return value


def rates_record(result):
    perf=result.performance
    a,b,c=float(np.sum(perf.rate_dl)),float(np.sum(perf.rate_ul)),float(np.sum(perf.rate_s))
    return dict(sum_rate_dl=a,sum_rate_ul=b,sum_rate_comm=a+b,sum_rate_sensing=c,
                weighted_sum_rate=float(perf.weighted_sum_rate))


def attempt(role,scheme,target,source,original,scene,exp,source_state=None):
    path=json_path(role,scheme,target,source)
    cfg=cfg_at(scene,target)
    if path.exists():
        row=read_json(path)
        if row.get('status')=='ok':
            saved=OUT/row['state_file']
            if not saved.is_file():raise FileNotFoundError(f'Missing checkpoint state: {saved}')
            verify_state(saved,cfg,row['weighted_sum_rate'])
        print(f'[RESUME] {role:7} {scheme:9} target={target:g} from={source:g}: {row.get("status")}',flush=True)
        return row
    seed=int(original[key(scheme,source)]['best_seed'])
    warm=None
    direct=None
    if source_state is not None:
        if not source_state.is_file():
            raise FileNotFoundError(f'Missing source state: {source_state}')
        direct=verify_state(source_state,cfg)
        warm=read_state(source_state)
    print(f'[RUN] {role:7} {scheme:9} target={target:g} from={source:g} seed={seed}',flush=True)
    t0=perf_counter()
    try:
        result=run_solver(scheme,cfg,exp,seed,warm=warm)
        data=dict(role=role,scheme=scheme,omega_comm=target,source_weight=source,
                  seed=seed,converged=bool(result.converged),
                  status='ok' if result.converged else 'not_converged',
                  iterations=iteration_count(result),
                  stop_reason=getattr(result,'stop_reason','not_recorded'),
                  runtime_s=round(perf_counter()-t0,3),
                  direct_reweight_wsr=direct if direct is not None else '')
        data.update(rates_record(result))
        if result.converged:
            state=state_path(role,scheme,target,source)
            save_state(state,result)
            data['state_file']=str(state.relative_to(OUT))
            verify_state(state,cfg,data['weighted_sum_rate'])
        else:
            data['state_file']=''
        save_json(path,data)
        print(f'   WSR={data["weighted_sum_rate"]:.6f}, conv={result.converged}, iter={data["iterations"]}',flush=True)
        return data
    except Exception as exc:
        # Preserve error for transparent diagnosis but do not silently promote it to a valid result.
        save_json(path,dict(role=role,scheme=scheme,omega_comm=target,source_weight=source,
                            status='error',error=f'{type(exc).__name__}: {exc}'))
        raise


def report(rows,old,scene=None):
    audit=[];replacements={}
    for scheme,target,source in PLAN:
        src=read_json(json_path('source',scheme,source,source)) if json_path('source',scheme,source,source).is_file() else None
        warm=read_json(json_path('warm',scheme,target,source)) if json_path('warm',scheme,target,source).is_file() else None
        historical=float(old[key(scheme,target)]['weighted_sum_rate'])
        # A direct cross-weight score is a feasible candidate, not a converged solution for this target.
        score=float(warm['direct_reweight_wsr']) if warm and warm.get('direct_reweight_wsr') not in (None,'') else None
        choose=(warm is not None and warm.get('status')=='ok' and warm.get('converged')
                and float(warm['weighted_sum_rate'])>historical+1e-8)
        if choose:
            orig=dict(old[key(scheme,target)])
            orig.update({fld:warm[fld] for fld in ('sum_rate_dl','sum_rate_ul','sum_rate_comm','sum_rate_sensing','weighted_sum_rate')})
            orig['best_seed']=warm['seed'];orig['iterations']=warm['iterations'];orig['runtime_s']=warm['runtime_s']
            orig['converged']='True'
            orig['selected_kind']='target_converged_warm'
            orig['source_weight']=source
            orig['original_wsr']=historical
            replacements[key(scheme,target)]=orig
        audit.append(dict(
            scheme=scheme,target_weight=target,source_weight=source,
            historical_wsr=historical,
            source_historical_wsr=float(old[key(scheme,source)]['weighted_sum_rate']),
            replayed_source_wsr=float(src['weighted_sum_rate']) if src and src.get('status')=='ok' else '',
            source_replay_gap=(float(src['weighted_sum_rate'])-float(old[key(scheme,source)]['weighted_sum_rate'])) if src and src.get('status')=='ok' else '',
            source_direct_feasible_wsr=score if score is not None else '',
            warm_final_wsr=float(warm['weighted_sum_rate']) if warm and 'weighted_sum_rate' in warm else '',
            warm_converged=bool(warm and warm.get('status')=='ok' and warm.get('converged')),
            warm_stop_reason=warm.get('stop_reason','') if warm else '',
            warm_iterations=warm.get('iterations','') if warm else '',
            selected_wsr=float(warm['weighted_sum_rate']) if choose else historical,
            selected_kind='target_converged_warm' if choose else 'original_cold_unchanged',
            improvement_vs_historical=float(warm['weighted_sum_rate'])-historical if choose else 0.0,
            remaining_vs_feasible_source=(max(0.0,score-(float(warm['weighted_sum_rate']) if choose else historical))) if score is not None else '',
        ))
    combined=[]
    for row in rows:
        r=dict(row)
        r['selected_kind']='original_cold_unchanged'
        r['source_weight']=''
        r['original_wsr']=r['weighted_sum_rate']
        combined.append(replacements.get(key(r['scheme'],r['omega_comm']),r))
    save_csv(OUT/'targeted_comparison.csv',audit,AUDIT_COLUMNS)
    save_csv(OUT/'stage10_tradeoff_targeted.csv',combined,RESULT_COLUMNS)
    attempts=[read_json(p) for p in sorted((OUT/'attempts').glob('*.json'))]
    save_csv(OUT/'attempts.csv',attempts,ATTEMPT_COLUMNS)
    print(f'\n[REPORT] original results: 80, targeted: {len(PLAN)}, replaced: {len(replacements)}',flush=True)
    for x in audit:
        print(f"  {x['scheme']:9} w={x['target_weight']:.3f} old={x['historical_wsr']:.6f} "
              f"final={x['selected_wsr']:.6f} gain={x['improvement_vs_historical']:+.6f} "
              f"({x['selected_kind']})",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true',help='Validate 20-point CSV and parameters; no solver run.')
    parser.add_argument('--phase',choices=('sources','warm','report','all'),default='all')
    args=parser.parse_args()
    config,weights=physical_config()
    rows,lookup=reference_rows(weights)
    print('[INFO] 20-point COLD reference:',REFERENCE)
    print('[INFO] Targeted repairs:',len(PLAN),' ; separate output:',OUT)
    sources=list(dict.fromkeys((s,w) for s,_,w in PLAN))
    for scheme,target,source in PLAN:
        h=lookup[key(scheme,target)]
        src=lookup[key(scheme,source)]
        rew=target/10*float(src['sum_rate_comm'])+(1-target)/5*float(src['sum_rate_sensing'])
        print(f'  {scheme:9} {target:g} <- {source:g} | cold={float(h["weighted_sum_rate"]):.6f} | source CSV reweight={rew:.6f}')
    print('  source cold reproductions:',len(sources),' ; warm refinements:',len(PLAN))
    if args.dry_run:return
    validate_manifest()
    scene,exp=construct_scene()
    if args.phase in ('sources','all'):
        for s,w in sources:
            attempt('source',s,w,w,lookup,scene,exp)
    if args.phase in ('warm','all'):
        for scheme,target,source in PLAN:
            p=json_path('source',scheme,source,source)
            if not p.is_file():
                raise FileNotFoundError(f'Run --phase sources first: {p}')
            src=read_json(p)
            if src.get('status')!='ok' or not src.get('converged'):
                print(f'[SKIP] Source did not converge: {scheme} {source}',flush=True)
                continue
            delta=float(src['weighted_sum_rate'])-float(lookup[key(scheme,source)]['weighted_sum_rate'])
            if abs(delta)>1e-3:
                print(f'[WARN] Recreated source differs from historical by {delta:+.6g}; '
                      'using the ACTUAL recreated full state, not CSV-only rates.',flush=True)
            attempt('warm',scheme,target,source,lookup,scene,exp,
                    source_state=OUT/src['state_file'])
    if args.phase in ('report','all'):
        report(rows,lookup)

if __name__=='__main__':
    main()
