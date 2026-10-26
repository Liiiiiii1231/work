"""Three MA FP-aware runs, compared against supplied original MA checkpoints."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(1,str(ROOT.parent))
import argparse,json,hashlib,time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import algorithms as shared
import inner_solver
import figure_engine as e
from system_model import SystemConfig
from ma_fp_aware import run_ma_fp_aware

SEEDS=(8001,8005,8010)


def load_case(seed):
    folder=ROOT/'baseline'/f'scene_{seed}'
    d=e.read_json(folder/'record.json')
    for name,key in (('initial.npz','initial_sha256'),(d['state_file'],'state_sha256')):
        if e.sha(folder/name)!=d[key]:raise ValueError('Baseline checksum mismatch')
    raw=d['effective_config'].copy()
    for k,v in list(raw.items()):
        if isinstance(v,list):
            a=np.asarray(v)
            raw[k]=a[:,0]+1j*a[:,1] if k in ('xi_s','fading_dl','fading_ul','fading_s') else a
    cfg=SystemConfig(**raw)
    if e.config_digest(cfg)!=d['cfg_sha256']:raise ValueError('Baseline config mismatch')
    g,r,_=e.load_state(folder/'initial.npz')
    return cfg,g,r,d


def run_one(job):
    seed,out_string,smoke=job;out=Path(out_string);folder=out/f'scene_{seed}'
    cfg,g,r,baseline=load_case(seed)
    if smoke:cfg=replace(cfg,max_outer_iter=2,max_inner_iter=2)
    record=folder/'record.json'
    if record.exists():
        d=e.read_json(record)
        if d['cfg_sha256']!=e.config_digest(cfg) or d['state_sha256']!=e.sha(folder/'final.npz'):
            raise ValueError('Existing result identity mismatch')
        return d['comparison']
    folder.mkdir(parents=True,exist_ok=True)
    e.save_state(folder/'initial.npz',g,r,[])
    p=e.read_json(ROOT/'tradeoff_config.json')['protocol'];proto=p['protocol']
    print('START scene',seed,'pid',os.getpid(),'max_inner',cfg.max_inner_iter,'max_outer',cfg.max_outer_iter,flush=True)
    tick=time.perf_counter()
    with patch.object(shared,'relative_change',e.abs_change),patch.object(inner_solver,'relative_change',e.abs_change):
        result,diagnostics=run_ma_fp_aware(cfg,rng=np.random.default_rng(cfg.random_seed),
            region_size=proto['ma_region_size_m'],min_spacing=proto['ma_min_spacing_m'],
            initial_step=proto['ma_initial_step_m'],initial_tx_positions=g.tx_positions.copy(),
            initial_rx_positions=g.rx_positions.copy(),initial_resources=r,verbose=False)
    elapsed=time.perf_counter()-tick
    h=result.wsr_history
    if np.min(np.diff(h)) < -1e-8 or not np.all(np.isfinite(h)):raise ValueError('Nonmonotone/nonfinite history')
    e.check_state(result.geometry,result.resources,cfg,p,'b6')
    metrics=e.metrics_values(result.geometry,result.resources,cfg)
    if abs(h[-1]-metrics['weighted_sum_rate'])>1e-8:raise ValueError('Metrics mismatch')
    e.save_state(folder/'final.npz',result.geometry,result.resources,h)
    old=baseline['row']
    comparison=dict(scene_seed=seed,weight=.001,smoke_only=smoke,
        original_wsr=old['weighted_sum_rate'],aware_wsr=metrics['weighted_sum_rate'],
        wsr_gain=metrics['weighted_sum_rate']-old['weighted_sum_rate'],
        original_seconds=old['runtime_s'],aware_seconds=elapsed,
        original_iterations=old['iterations'],aware_iterations=result.iterations,
        original_converged=old['converged'],aware_converged=result.converged,
        candidate_fp_calls=sum(x['candidate_fp_calls'] for x in diagnostics))
    e.json_write(folder/'diagnostics.json',diagnostics)
    e.json_write(record,dict(cfg_sha256=e.config_digest(cfg),state_sha256=e.sha(folder/'final.npz'),comparison=comparison,metrics=metrics))
    print('DONE',json.dumps(comparison),flush=True)
    return comparison


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--workers',type=int,choices=(1,2,3),default=3)
    ap.add_argument('--check-only',action='store_true')
    ap.add_argument('--smoke',action='store_true',help='Two outer/inner steps only; separate diagnostic output')
    args=ap.parse_args()
    expected=e.read_json(ROOT/'baseline_shared_hashes.json')
    for name,h in expected.items():
        f=ROOT.parent/name
        if hashlib.sha256(f.read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=h:
            raise ValueError('Shared source differs from reference experiment: '+name)
    for seed in SEEDS:
        cfg,g,r,d=load_case(seed)
        if (cfg.max_inner_iter,cfg.max_outer_iter)!=(1500,2000):raise ValueError('Wrong iteration budget')
        if not np.isfinite(e.metrics_values(g,r,cfg)['weighted_sum_rate']):raise ValueError('Nonfinite initial state')
        protocol=e.read_json(ROOT/'tradeoff_config.json')['protocol']
        e.check_state(g,r,cfg,protocol,'b6')
    if args.check_only:
        print('CHECK PASSED: three baseline identities, common initial states, 1500/2000 budgets and shared-source hashes');return
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    identity=dict(smoke=args.smoke,baseline_hashes=expected,local_sources={f.name:e.sha(f) for f in ROOT.glob('*.py')})
    meta=out/'identity.json'
    if meta.exists() and e.read_json(meta)!=identity:raise ValueError('Use a new output for changed protocol/source')
    e.json_write(meta,identity)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(run_one,[(s,str(out),args.smoke) for s in SEEDS]))
    e.csv_write(out/'comparison.csv',rows,list(rows[0]))
    e.json_write(out/'completion.json',dict(complete=True,all_converged=all(r['aware_converged'] for r in rows),smoke_only=args.smoke,cases=rows))
    aggregate=dict(comparison=rows,smoke_only=args.smoke,protocol=identity,cases={})
    for seed in SEEDS:
        folder=out/f'scene_{seed}'
        case=dict(record=e.read_json(folder/'record.json'),diagnostics=e.read_json(folder/'diagnostics.json'),states={})
        for name in ('initial.npz','final.npz'):
            with np.load(folder/name,allow_pickle=False) as archive:
                case['states'][name]={k:dict(shape=list(archive[k].shape),dtype=str(archive[k].dtype),
                    real=archive[k].real.tolist(),imag=archive[k].imag.tolist()) for k in archive.files}
        aggregate['cases'][str(seed)]=case
    e.json_write(out/'all_data.json',aggregate)
    import shutil
    sources=out/'provenance';sources.mkdir(exist_ok=True)
    for path in ROOT.iterdir():
        if path.is_file() and path.suffix in ('.py','.json','.sh','.md'):shutil.copy2(path,sources/path.name)
    for name in expected:shutil.copy2(ROOT.parent/name,sources/name)
    import zipfile
    with zipfile.ZipFile(out.with_name(out.name+'_bundle.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for f in out.rglob('*'):
            if f.is_file():z.write(f,out.name+'/'+f.relative_to(out).as_posix())
    with zipfile.ZipFile(out.with_name(out.name+'_bundle.zip')) as z:
        if z.testzip() is not None:raise RuntimeError('Output archive verification failed')
    print('All three cases finished:',out,flush=True)


if __name__=='__main__':main()
