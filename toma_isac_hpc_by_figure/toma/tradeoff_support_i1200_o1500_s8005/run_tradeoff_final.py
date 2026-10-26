"""Final tradeoff: independent single-pass tasks from convergence-screened geometry."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(1,str(ROOT.parent))
import argparse,hashlib,json,shutil
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import figure_engine as e

SHARED=('algorithms.py','channel_model.py','inner_solver.py','outer_solver.py','system_model.py')

def normalized_hash(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()

def load_protocol(config,selection):
    expected=e.read_json(ROOT/'shared_module_hashes.json')
    for name in SHARED:
        if normalized_hash(ROOT.parent/name)!=expected[name]:
            raise ValueError('Shared source differs from verified screening version: '+name)
    b=e.read_json(config);p=b['protocol'];p['_models']=b['models']
    reference=e.read_json(ROOT/'reference_screening_identity.json')
    if b['models']!=reference['protocol']['_models']:
        raise ValueError('Scene/array models differ from the convergence screening model')
    if p['protocol']!=reference['protocol']['protocol']:
        raise ValueError('Algorithm budget/initialization differs from convergence screening')
    for key in ('scene_seed_start','excluded_scene_seeds','screening'):
        if p.get(key)!=reference['protocol'].get(key):raise ValueError('Scene selection differs from screening protocol: '+key)
    if p['schemes']!=['proposed','b6','b2','b3'] or p['starts']!={s:1 for s in p['schemes']}:
        raise ValueError('Expected four schemes, one frozen start per task')
    if p['n_mc']<1 or not p['tradeoff']['weights'] or len(set(p['tradeoff']['weights']))!=len(p['tradeoff']['weights']):
        raise ValueError('Invalid scene count or duplicate/empty weights')
    if any(not 0<=w<=1 for w in p['tradeoff']['weights']):raise ValueError('Weight outside [0,1]')
    file=Path(selection)/'frozen_initial_geometry.json'
    if not file.is_file():
        raise FileNotFoundError('Wait for convergence job FROZEN GEOMETRY READY; missing '+str(file))
    frozen=e.read_json(file)
    if frozen.get('identity')!=reference:
        raise ValueError('Frozen file is not from the verified 1200/1500 convergence screening')
    if frozen.get('reference')!=reference['protocol']['screening']:
        raise ValueError('Screening reference parameters differ')
    if set(frozen.get('geometry',{}))!={'proposed','b6'}:
        raise ValueError('Both Proposed and MA screened initial geometries are required')
    if set(frozen.get('winners',{}))!={'proposed','b6'}:
        raise ValueError('Missing screening winner metadata')
    p['_frozen_geometry']=frozen['geometry']
    e.STARTS.update({s:1 for s in p['schemes']})
    return p,file

def validate_tasks(p,tasks):
    seen=set()
    for t in tasks:
        if t['scene_seed'] in p.get('excluded_scene_seeds',[]):raise ValueError('Excluded scene in tradeoff task list')
        cfg=e.config_for_task(ROOT,p,t)
        if (cfg.max_inner_iter,cfg.max_outer_iter)!=(1200,1500):raise ValueError('Wrong budget')
        if (cfg.n_tx,cfg.n_rx)!=(16,16):raise ValueError('Wrong array size')
        key=(t['parameter_value'],t['scheme'])
        if key in seen:continue
        seen.add(key)
        g,r=e.pair(cfg,p,t['scheme'],0);e.check_state(g,r,cfg,p,t['scheme'])

def collect(out,p,tasks,smoke):
    rows=[e.read_verified(out,p,t,0) for t in tasks]
    if any(r is None for r in rows):raise RuntimeError('Missing task: partial Monte Carlo means are disabled')
    if any(r['calls']!=1 for r in rows):raise RuntimeError('Only one solver call per task is permitted')
    e.csv_write(out/'tables'/'tradeoff_per_scene.csv',rows,list(rows[0]))
    e.STYLES['b6']['label']='MA + FP-aware'
    e.plot_tradeoff(out,p,tasks,rows)
    stats=[]
    for w in p['tradeoff']['weights']:
        for sc in p['schemes']:
            group=[r for r in rows if r['parameter_value']==w and r['scheme']==sc]
            if len(group)!=p['n_mc']:raise RuntimeError('Wrong Monte Carlo sample count')
            values=np.array([r['weighted_sum_rate'] for r in group])
            stats.append(dict(weight=w,scheme=sc,n_mc=len(group),converged=sum(r['converged'] for r in group),
                capped=sum(not r['converged'] for r in group),mean_wsr=float(values.mean()),
                std_wsr=float(values.std(ddof=1)) if len(group)>1 else 0.,
                mean_seconds=float(np.mean([r['runtime_s'] for r in group]))))
    e.csv_write(out/'tables'/'tradeoff_status.csv',stats,list(stats[0]))
    status=dict(complete=True,figure='tradeoff',smoke_only=smoke,expected_tasks=len(tasks),finished_tasks=len(rows),
        excluded_scene_seeds=p.get('excluded_scene_seeds',[]),scene_selection_note=p.get('scene_selection_note',''),
        converged_tasks=sum(r['converged'] for r in rows),all_converged=all(r['converged'] for r in rows),
        scene_seeds=sorted({t['scene_seed'] for t in tasks}),weights=p['tradeoff']['weights'],
        completion_definition='All independent single-pass tasks finished; capped results included and labelled. No warm starts or continuation.')
    e.json_write(out/'completion.json',status)
    print(json.dumps(status,indent=2),flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=ROOT/'tradeoff_final_config.json')
    ap.add_argument('--selection-dir',type=Path,default=ROOT.parent/'results'/'final_initial_geometry_i1200_o1500_s8005')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=56)
    ap.add_argument('--check-only',action='store_true')
    ap.add_argument('--smoke',action='store_true',help='Diagnostics only: one scene, three weights, two inner/outer iterations')
    a=ap.parse_args()
    if a.workers<1:ap.error('workers must be positive')
    p,file=load_protocol(a.config,a.selection_dir)
    tasks=e.build_tasks(ROOT,p,'tradeoff')
    validate_tasks(p,tasks)
    print('CHECK PASSED:',len(tasks),'tasks; 1200/1500; frozen initial geometry SHA256',e.sha(file),flush=True)
    if a.check_only:return
    if a.smoke:
        p=deepcopy(p);p['n_mc']=1;p['tradeoff']['weights']=[0.,.2,1.]
        p['protocol']['max_inner_iter']=2;p['protocol']['max_outer_iter']=2
        tasks=e.build_tasks(ROOT,p,'tradeoff')
    out=a.output.resolve()
    # Every submission starts a new run; never adopt or continue earlier outputs.
    if out.exists() and any(f.name!='logs' for f in out.iterdir()):
        raise ValueError('Use a fresh output directory; continuation/reuse is disabled')
    out.mkdir(parents=True,exist_ok=True)
    identity=dict(protocol=p,frozen_sha256=e.sha(file),smoke_only=a.smoke,
        sources={f.name:normalized_hash(f) for f in ROOT.glob('*.py')},
        shared_sources={n:normalized_hash(ROOT.parent/n) for n in SHARED})
    e.json_write(out/'metadata'/'identity.json',identity)
    shutil.copy2(file,out/'frozen_initial_geometry.json')
    comparison=file.parent/'screening_comparison.csv'
    if comparison.exists():shutil.copy2(comparison,out/'screening_comparison.csv')
    entry=ROOT.parent/'05_communication_sensing_tradeoff'
    for name in ('submit_tradeoff.sh','run_tradeoff.py'):
        if (entry/name).is_file():
            dest=out/'provenance'/'entry_scripts'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(entry/name,dest)
    print('PLAN',len(tasks),'independent tasks, workers',min(a.workers,len(tasks)),'smoke_only',a.smoke,flush=True)
    errors=[]
    with ProcessPoolExecutor(max_workers=min(a.workers,len(tasks))) as pool:
        futures={pool.submit(e.run_job,(str(out),p,t,0,1)):t for t in tasks}
        for future in as_completed(futures):
            try:
                row=future.result()
                print('TASK FINISHED',e.task_key(row),'converged',row['converged'],flush=True)
            except Exception as exc:
                error=dict(task=futures[future],error=repr(exc));errors.append(error)
                print('FAILED',error,flush=True)
    e.json_write(out/'logs'/'task_errors.json',errors)
    if errors:raise RuntimeError('Tasks failed; see logs/task_errors.json')
    collect(out,p,tasks,a.smoke)

if __name__=='__main__':main()
