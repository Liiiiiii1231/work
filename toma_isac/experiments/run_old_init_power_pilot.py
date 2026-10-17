"""恢复旧 Stage8 初始化/起点预算，仅采用当前绝对停止条件。"""
from __future__ import annotations
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(name,'1')
import argparse, csv, hashlib, json, sys, time
from pathlib import Path
from dataclasses import replace, asdict
from contextlib import ExitStack
from unittest.mock import patch
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SCHEMES=('proposed','b6','b2','b3')
STYLE={'proposed':('Proposed','D','#df2228'),'b6':('MA + FP','^','#279b3b'),'b2':('Fixed-ToMA + FP','o','#e99a21'),'b3':('Fixed UPA + FP','s','#2675c1')}

def serial(x):
    if isinstance(x,np.ndarray):return serial(x.tolist())
    if isinstance(x,np.generic):return serial(x.item())
    if isinstance(x,complex):return [x.real,x.imag]
    if isinstance(x,dict):return {str(k):serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v) for v in x]
    return x

def write_json(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(serial(x),ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)

def save_state(p,g,r,h):
    with p.open('wb') as f:
        np.savez_compressed(f,**{n:getattr(g,n) for n in ('endpoints','tx_positions','rx_positions')},**{n:getattr(r,n) for n in ('q_matrix','q_ul','b_ul','u_s')},wsr_history=np.asarray(h))

def load_state(p):
    from state import GeometryState,ResourceState
    with np.load(p,allow_pickle=False) as z:
        return (GeometryState(**{n:z[n].copy() for n in ('endpoints','tx_positions','rx_positions')}),ResourceState(**{n:z[n].copy() for n in ('q_matrix','q_ul','b_ul','u_s')}),z['wsr_history'].tolist())

def make_cfg(config,scene,dbm,smoke=False):
    from config import load_config
    from stage8 import make_scene_config
    raw=yaml.safe_load(Path(config).read_text(encoding='utf-8'));exp=raw['experiment']
    cfg=replace(load_config(config),p_dl_max=10**((dbm-30)/10),kappa_ul=.5,tol_inner=1e-4,tol_outer=1e-4,plateau_window=0,tol_window=0.,max_inner_iter=2 if smoke else 1200,max_outer_iter=2 if smoke else 1500)
    fading=exp.get('fading',{})
    return make_scene_config(cfg,scene,np.asarray(exp['position_jitter']),randomize_fading=fading.get('enabled',False),fading_model=fading.get('model','scalar_rayleigh'))

def run_one(job):
    import algorithm,baselines,inner_solver,outer_solver,ma_baseline as ma
    from channels import build_channels
    from metrics import compute_performance
    from utils import check_resource_constraints
    from copy import deepcopy
    config,scene,dbm,scheme,seed,out,max_calls,smoke=job
    cfg=make_cfg(config,scene,dbm,smoke)
    exp=yaml.safe_load(Path(config).read_text(encoding='utf-8'))['experiment'];m=exp.get('ma',{})
    folder=Path(out)/f'scene_{scene}'/f'{dbm:g}dBm'/scheme/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    rec=folder/'record.json';g=r=None;h=[];calls=0;elapsed=0.;iterations=0
    if rec.exists():
        d=json.loads(rec.read_text());
        if hashlib.sha256((folder/d['state_file']).read_bytes()).hexdigest()!=d['state_sha256']:raise RuntimeError('Checkpoint state hash mismatch')
        g,r,h=load_state(folder/d['state_file']);calls=d['calls'];elapsed=d['runtime_s'];iterations=d['iterations']
        if d['converged']:return d
    required=2 if scheme in ('proposed','b6') else 1
    tol=cfg.tol_outer if required==2 else cfg.tol_inner
    for _ in range(max_calls):
        calls+=1
        init={}
        module=algorithm if scheme=='proposed' else ma if scheme=='b6' else baselines
        original=module.initialize_resources
        def capture(geometry,channels,config,rng):
            resources=original(geometry,channels,config,rng)
            init.update(g=deepcopy(geometry),r=deepcopy(resources))
            return resources
        kwargs=dict(rng=np.random.default_rng(seed),verbose=False)
        if g is not None:
            kwargs['initial_resources']=deepcopy(r)
            if scheme in ('proposed','b2'):kwargs['initial_endpoints']=g.endpoints.copy()
            elif scheme=='b6':kwargs.update(initial_tx_positions=g.tx_positions.copy(),initial_rx_positions=g.rx_positions.copy())
        tick=time.perf_counter()
        with ExitStack() as stack:
            for mod in (algorithm,baselines,inner_solver,outer_solver,ma):
                stack.enter_context(patch.object(mod,'relative_change',lambda a,b:abs(float(a)-float(b))))
            stack.enter_context(patch.object(module,'initialize_resources',capture))
            if scheme=='proposed':
                result=algorithm.run_joint_algorithm(cfg,position_update_mode='fp_aware',fp_aware_candidate_iters=2,**kwargs)
                new=list(map(float,result.joint_wsr_history));count=result.outer_iterations
            elif scheme=='b6':
                result=ma.run_ma_fp(cfg,region_size=m.get('region_size',.3),min_spacing=m.get('min_spacing',cfg.wavelength/2),initial_step=m.get('initial_step',.03),randomize_initial_geometry=(seed!=int(exp['fixed_upa_seed'])),initial_jitter_fraction=m.get('initial_jitter_fraction',.5),**kwargs)
                new=list(map(float,result.wsr_history));count=result.iterations
            else:
                result=(baselines.run_fixed_toma_fp if scheme=='b2' else baselines.run_fpa_upa_fp)(cfg,**kwargs)
                new=list(map(float,result.wsr_history));count=result.iterations
        elapsed+=time.perf_counter()-tick;iterations+=int(count)
        if init and not (folder/'initial.npz').exists():
            if not np.allclose(abs(init['r'].q_ul)**2,.25*cfg.p_ul_max,rtol=1e-12,atol=1e-14):raise RuntimeError('Initial UL power mismatch')
            save_state(folder/'initial.npz',init['g'],init['r'],[])
        if not (folder/'initial.npz').exists():raise RuntimeError('Initial state was not captured')
        if h and not np.isclose(h[-1],new[0],rtol=1e-9,atol=1e-9):raise RuntimeError('Continuation mismatch')
        h=h+new[1:] if h else new
        g,r=result.geometry,result.resources
        check_resource_constraints(r,cfg)
        if scheme in ('b2','b3'):
            g0,_,_=load_state(folder/'initial.npz')
            for n in ('endpoints','tx_positions','rx_positions'):
                if not np.allclose(getattr(g0,n),getattr(g,n),rtol=0,atol=1e-12):raise RuntimeError('Fixed geometry changed')
        perf=compute_performance(build_channels(g,cfg),r,cfg)
        wsr=float(perf.weighted_sum_rate)
        if not np.isfinite(wsr) or not np.isclose(h[-1],wsr,atol=1e-9,rtol=1e-9):raise RuntimeError('WSR mismatch')
        if np.min(np.diff(h)) < -1e-8*max(1.,max(abs(x) for x in h)):raise RuntimeError('Non-monotone history')
        converged=bool(result.converged)
        if converged and (len(new)<required+1 or max(abs(np.diff(new[-required-1:])))>tol):raise RuntimeError('Stopping check failed')
        state=f'call_{calls:03d}.npz';save_state(folder/state,g,r,h)
        d=dict(scene_seed=scene,p_dl_dbm=dbm,scheme=scheme,seed=seed,weighted_sum_rate=wsr,sum_rate_dl=float(np.sum(perf.rate_dl)),sum_rate_ul=float(np.sum(perf.rate_ul)),sum_rate_sensing=float(np.sum(perf.rate_s)),converged=converged,stop_reason=getattr(result,'stop_reason',''),calls=calls,iterations=iterations,runtime_s=elapsed,state_file=state,effective_config=serial(asdict(cfg)))
        d['state_sha256']=hashlib.sha256((folder/state).read_bytes()).hexdigest()
        write_json(rec,d)
        print(f'{scheme} {dbm:g}dBm seed={seed}: WSR={wsr:.8f}, converged={converged}',flush=True)
        if converged:break
    return d

def csv_write(path,rows):
    if not rows:return
    fields=[k for k in rows[0] if k!='effective_config']
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def main():
    from stage8 import make_ma_start_seeds
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--scene',type=int,default=8001)
    p.add_argument('--config',default='configs/diagnostics/old_init_power_pilot.yaml');p.add_argument('--max-calls',type=int,default=2)
    p.add_argument('--output',default='results/old_init_power_pilot');p.add_argument('--check-only',action='store_true')
    p.add_argument('--smoke',action='store_true',help='仅接口短测；输出不作为正式结果')
    a=p.parse_args();config=(ROOT/a.config).resolve();out=(ROOT/a.output).resolve()
    if a.workers<1 or a.max_calls<1:p.error('workers/max-calls must be positive')
    if a.smoke:out=out.with_name(out.name+'_smoke')
    exp=yaml.safe_load(config.read_text(encoding='utf-8'))['experiment']
    seeds={s:list(map(int,exp['multi_start_seeds'])) for s in ('proposed','b2')}
    seeds['b3']=[int(exp['fixed_upa_seed'])];seeds['b6']=list(make_ma_start_seeds(a.scene,int(exp['fixed_upa_seed']),int(exp.get('ma',{}).get('n_starts',3))))
    files=['algorithm.py','baselines.py','inner_solver.py','outer_solver.py','ma_baseline.py','initialization.py','geometry.py','channels.py','metrics.py','config.py','utils.py','stage8.py','state.py']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    manifest=dict(scene=a.scene,dbm=[20,25,30],seeds=seeds,configuration_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),source_sha256=hashes,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),absolute_tolerance=1e-4,kappa_ul=.5,smoke=a.smoke)
    cfg=make_cfg(config,a.scene,30,a.smoke);cfg.validate()
    if not np.allclose(cfg.p_ul_max,.1):raise RuntimeError('Expected old per-user UL maximum 0.1W; inspect config before proceeding')
    print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)
    jobs=[(str(config),a.scene,float(dbm),sc,seed,str(out),a.max_calls,a.smoke) for dbm in (20,25,30) for sc in SCHEMES for seed in seeds[sc]]
    if a.smoke:jobs=[next(j for j in jobs if j[3]==s) for s in SCHEMES]
    print(f'{len(jobs)} independent starts; no cross-parameter warm starts.',flush=True)
    if a.check_only:return
    out.mkdir(parents=True,exist_ok=True);mp=out/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:raise RuntimeError('Inputs changed; use a new output directory')
    write_json(mp,manifest);rows=[];errors=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(run_one,j):j for j in jobs}
        for f in as_completed(futures):
            try:rows.append(f.result())
            except Exception as e:
                errors.append(dict(job=list(futures[f]),error=repr(e)));print('ERROR',repr(e),flush=True)
    rows.sort(key=lambda d:(d['p_dl_dbm'],d['scheme'],d['seed']));csv_write(out/'all_starts.csv',rows)
    best=[]
    for dbm in (20,25,30):
        for sc in SCHEMES:
            group=[r for r in rows if r['p_dl_dbm']==dbm and r['scheme']==sc]
            if len(group)==len(seeds[sc]) and all(r['converged'] for r in group):best.append(max(group,key=lambda r:r['weighted_sum_rate']))
    csv_write(out/'best_starts.csv',best)
    complete=not a.smoke and not errors and len(rows)==len(jobs) and all(r['converged'] for r in rows)
    write_json(out/'completion.json',dict(complete=complete,smoke=a.smoke,expected_starts=len(jobs),finished_starts=len(rows),converged_starts=sum(r['converged'] for r in rows),errors=errors))
    if complete:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7,4.6))
        for sc in SCHEMES:
            group=sorted([r for r in best if r['scheme']==sc],key=lambda r:r['p_dl_dbm']);label,marker,color=STYLE[sc]
            ax.plot([r['p_dl_dbm'] for r in group],[r['weighted_sum_rate'] for r in group],marker=marker,color=color,label=label)
        reference=Path(__file__).resolve().parent/'old_init_pilot_assets/regular_single_start_scene8001.csv'
        if a.scene==8001 and reference.exists():
            with reference.open(encoding='utf-8-sig') as f:ref=list(csv.DictReader(f))
            comparison=[]
            for sc in SCHEMES:
                group=sorted([r for r in ref if r['scheme']==sc],key=lambda r:float(r['parameter_value']))
                label,marker,color=STYLE[sc]
                ax.plot([10*np.log10(float(r['parameter_value']))+30 for r in group],[float(r['weighted_sum_rate']) for r in group],ls='--',marker=marker,color=color,alpha=.45,label=label+' (current single start)')
                for r in group:
                    dbm=round(10*np.log10(float(r['parameter_value']))+30)
                    new=next(x for x in best if x['scheme']==sc and x['p_dl_dbm']==dbm)
                    comparison.append(dict(scheme=sc,p_dl_dbm=dbm,current_single_start_wsr=float(r['weighted_sum_rate']),restored_old_init_wsr=new['weighted_sum_rate'],difference=new['weighted_sum_rate']-float(r['weighted_sum_rate'])))
            csv_write(out/'comparison.csv',comparison)
        ax.set(xlabel='Maximum downlink power (dBm)',ylabel='WSR (bit/s/Hz)',xticks=[20,25,30],title=f'Scene {a.scene} | old initialization | absolute tolerance 1e-4')
        ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(out/'old_init_power_pilot.png',dpi=220);plt.close(fig)
    if not complete and not a.smoke:raise SystemExit('Incomplete: inspect completion.json; rerun same command to continue.')

if __name__=='__main__':main()
