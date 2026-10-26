"""Five-node targeted budget pilot using job33885's exact saved initial states."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
from pathlib import Path
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import sys,json,hashlib,time,shutil
import numpy as np

HERE=Path(__file__).resolve().parent;RUNTIME=HERE/'runtime'
sys.path[:0]=[str(RUNTIME),str(HERE)]
import figure_engine as e,algorithms as algorithm,inner_solver as inner
from system_model import SystemConfig
import queue_runtime as queue

SHARED=('algorithms.py','channel_model.py','inner_solver.py','outer_solver.py','system_model.py')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def manifest():return e.read_json(HERE/'pilot_manifest.json')
def decode_cfg(raw):
    raw=deepcopy(raw)
    for k,v in list(raw.items()):
        if isinstance(v,list):
            a=np.asarray(v)
            raw[k]=a[:,0]+1j*a[:,1] if k in ['xi_s','fading_dl','fading_ul','fading_s'] else a
    return SystemConfig(**raw)

def load_input(task,p):
    state=HERE/task['input_state'];record=HERE/task['input_record']
    if sha(state)!=task['state_sha256'] or sha(record)!=task['record_sha256']:raise ValueError('Pilot input checksum differs')
    d=e.read_json(record);cfg=decode_cfg(d['effective_config'])
    if e.config_digest(cfg)!=d['cfg_sha256']:raise ValueError('Original effective configuration digest mismatch')
    if (cfg.tol_inner,cfg.tol_outer,cfg.max_inner_iter,cfg.max_outer_iter)!=(1e-5,1e-4,1200,1500):
        raise ValueError('Unexpected original tolerance or budget')
    budget=manifest()['budget']
    if task['scheme']=='b6':cfg=replace(cfg,max_outer_iter=budget['ma_outer'])
    else:cfg=replace(cfg,max_inner_iter=budget['fixed_inner'])
    if '_pilot_test_budget' in p:
        cfg=replace(cfg,**p['_pilot_test_budget'])
    g,r,h=e.load_state(state)
    if h:raise ValueError('Expected an original INITIAL state, not a terminal checkpoint')
    e.check_state(g,r,cfg,p,task['scheme'])
    return cfg,g,r,d

def run_job(job):
    output,p,task,start,calls=job
    if start!=0 or calls!=1:raise ValueError('Only one fresh solver call per pilot task')
    cfg,g,r,d=load_input(task,p)
    sc=task['scheme'];fp_updates=0;original=inner.solve_inner_problem
    if algorithm.solve_inner_problem is not original:raise RuntimeError('Unexpected resource solver patch')
    original_positions=algorithm._update_ma_positions_once
    def count_fp(channels,resources,cc,**kw):
        nonlocal fp_updates
        rr,hh=original(channels,resources,cc,**kw);fp_updates+=len(hh)-1
        return rr,hh
    actual_task={k:task[k] for k in ['figure','parameter_value','scene_seed','scheme']}
    tick=time.perf_counter()
    with patch.object(algorithm,'solve_inner_problem',count_fp):
        result,history,iterations=e.run_solver_call(cfg,p,actual_task,g,r)
    seconds=time.perf_counter()-tick
    if algorithm._update_ma_positions_once is not original_positions:raise RuntimeError('MA position solver changed')
    if any(n in sys.modules for n in ['ma_fp_aware','fp_acceleration','accelerated_adapter']):
        raise RuntimeError('Pilot must use original MA and ordinary FP only')
    e.check_state(result.geometry,result.resources,cfg,p,sc,g)
    delta=np.diff(history);required=2 if sc=='b6' else 1;tol=cfg.tol_outer if sc=='b6' else cfg.tol_inner
    if len(history)!=iterations+1:raise ValueError('History length and iteration count differ')
    if min(delta)<-1e-7:raise ValueError('Unexpected decreasing objective')
    if result.converged and (len(delta)<required or max(abs(delta[-required:]))>tol):raise ValueError('Stopping/history mismatch')
    v=e.metrics_values(result.geometry,result.resources,cfg);old=d['baseline'];baseline=np.asarray(d['baseline_history'])
    n=min(len(baseline),len(history));prefix_difference=float(np.max(abs(np.asarray(history[:n])-baseline[:n])))
    row=dict(pilot_id=task['_pilot_id'],**actual_task,weighted_sum_rate=v['weighted_sum_rate'],converged=bool(result.converged),
        iterations=int(iterations),stop_reason=result.stop_reason,runtime_s=seconds,total_fp_updates=fp_updates,
        max_inner_iter=cfg.max_inner_iter,max_outer_iter=cfg.max_outer_iter,inner_tolerance=cfg.tol_inner,outer_tolerance=cfg.tol_outer,
        sum_rate_comm=v['sum_rate_comm'],sum_rate_sensing=v['sum_rate_sensing'],
        baseline_converged=bool(old['converged']),baseline_iterations=int(old['iterations']),baseline_runtime_s=float(old['runtime_s']),
        baseline_wsr=float(old['weighted_sum_rate']),baseline_comm=float(old['sum_rate_comm']),baseline_sensing=float(old['sum_rate_sensing']),
        delta_wsr=v['weighted_sum_rate']-float(old['weighted_sum_rate']),
        delta_comm=v['sum_rate_comm']-float(old['sum_rate_comm']),delta_sensing=v['sum_rate_sensing']-float(old['sum_rate_sensing']),
        relative_wsr_change=(v['weighted_sum_rate']-float(old['weighted_sum_rate']))/max(abs(float(old['weighted_sum_rate'])),1e-12),
        last_increment=float(delta[-1]),last100_gain=float(history[-1]-history[max(0,len(history)-101)]),
        original_prefix_max_difference=prefix_difference,local_test=('_pilot_test_budget' in p))
    folder=Path(output)/'tasks'/f"{task['_pilot_id']:03d}";folder.mkdir(parents=True,exist_ok=True)
    e.save_state(folder/'final.npz',result.geometry,result.resources,history)
    e.json_write(folder/'record.json',dict(row=row,initial_sha256=task['state_sha256'],
        original_config_sha256=d['cfg_sha256'],effective_config=e.serial(__import__('dataclasses').asdict(cfg)),
        final_sha256=sha(folder/'final.npz')))
    return row

def collect(output,p,tasks,smoke):
    out=Path(output);rows=[]
    for task in tasks:
        folder=out/'tasks'/f"{task['_pilot_id']:03d}";record=e.read_json(folder/'record.json')
        if sha(folder/'final.npz')!=record['final_sha256']:raise ValueError('Final state checksum mismatch')
        r=record['row']
        if any(r[k]!=task[k] for k in ['scheme','scene_seed','parameter_value']):raise ValueError('Task/record mismatch')
        rows.append(r)
    e.csv_write(out/'tables/pilot_per_task.csv',rows,list(rows[0]))
    summary=[]
    for weight in sorted({r['parameter_value'] for r in rows}):
        for sc in ['b6','b2','b3']:
            group=[r for r in rows if r['parameter_value']==weight and r['scheme']==sc]
            if not group:continue
            if not smoke and len(group)!=10:raise ValueError('Expected all ten selected scenes per weight/scheme')
            summary.append(dict(weight=weight,scheme=sc,n_scenes=len(group),
                baseline_converged=sum(r['baseline_converged'] for r in group),new_converged=sum(r['converged'] for r in group),
                new_capped=sum(not r['converged'] for r in group),
                mean_baseline_wsr=float(np.mean([r['baseline_wsr'] for r in group])),mean_new_wsr=float(np.mean([r['weighted_sum_rate'] for r in group])),
                mean_relative_wsr_change=float(np.mean([r['relative_wsr_change'] for r in group])),
                maximum_absolute_relative_wsr_change=max(abs(r['relative_wsr_change']) for r in group),
                mean_new_iterations=float(np.mean([r['iterations'] for r in group])),max_new_iterations=max(r['iterations'] for r in group),
                baseline_mean_comm=float(np.mean([r['baseline_comm'] for r in group])),new_mean_comm=float(np.mean([r['sum_rate_comm'] for r in group])),
                baseline_mean_sensing=float(np.mean([r['baseline_sensing'] for r in group])),new_mean_sensing=float(np.mean([r['sum_rate_sensing'] for r in group]))))
    e.csv_write(out/'tables/pilot_summary.csv',summary,list(summary[0]))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5.5),layout='constrained')
    for sc,label,color in [('b6','MA + FP','#279b3b'),('b2','Fixed-ToMA + FP','#e99a21'),('b3','Fixed UPA + FP','#2675c1')]:
        group=[r for r in summary if r['scheme']==sc]
        ax.plot([r['baseline_mean_comm'] for r in group],[r['baseline_mean_sensing'] for r in group],'o--',color=color,alpha=.55,label=label+' original budget')
        ax.plot([r['new_mean_comm'] for r in group],[r['new_mean_sensing'] for r in group],'s-',color=color,label=label+' larger budget')
    ax.set(xlabel='Mean communication rate (bit/s/Hz)',ylabel='Mean sensing rate (bit/s/Hz)',
        title=f'Targeted pilot only | {sum(r["converged"] for r in rows)}/{len(rows)} new tasks converged')
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    (out/'figures').mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(out/'figures'/f'pilot_budget_comparison.{ext}',dpi=200)
    plt.close(fig)
    e.json_write(out/'completion.json',dict(complete=True,figure='tradeoff_budget_pilot',expected_tasks=len(tasks),finished_tasks=len(rows),
        baseline_converged=sum(r['baseline_converged'] for r in rows),converged_tasks=sum(r['converged'] for r in rows),
        all_converged=all(r['converged'] for r in rows),local_test=bool(smoke),
        note='Targeted ten-scene diagnostic sample, not a replacement 100-scene paper figure. Capped tasks retained and marked.'))
    shutil.copy2(HERE/'selected_scenes.json',out/'selected_scenes.json')
    archive=Path(shutil.make_archive(str(out)+'_bundle','zip',root_dir=out.parent,base_dir=out.name))
    e.json_write(out/'bundle_ready.json',dict(archive=str(archive),local_test=bool(smoke)))
    print('PILOT RESULT BUNDLE:',archive,flush=True)

def bootstrap(figure):return RUNTIME,e,SimpleNamespace(SHARED=SHARED,collect=collect)

def load_plan(args):
    m=manifest()
    for rel,h in m['files'].items():
        if sha(HERE/rel)!=h:raise ValueError('Runtime source differs from original job33885: '+rel)
    if sha(HERE/'protocol.json')!=m['protocol_sha256']:raise ValueError('Pilot protocol checksum differs')
    p=e.read_json(HERE/'protocol.json');tasks=deepcopy(m['tasks'])
    if args.local_test_fixture:
        if not args.local_test:raise ValueError('Smoke budget requires --local-test')
        fixture=e.read_json(args.local_test_fixture);tasks=[tasks[i] for i in fixture['indices']]
        p['_pilot_test_budget']=fixture['budget']
    elif len(tasks)!=150:raise ValueError('Expected 150 production pilot tasks')
    if len({t['_pilot_id'] for t in tasks})!=len(tasks):raise ValueError('Duplicate pilot tasks')
    for task in tasks:load_input(task,p)
    return RUNTIME,e,bootstrap('tradeoff')[2],p,tasks,None

e.run_job=run_job
queue.HERE=HERE;queue.TOMA=RUNTIME
queue.bootstrap=bootstrap;queue.load_plan=load_plan
if __name__=='__main__':
    if any(a=='--figure' or a.startswith('--figure=') for a in sys.argv[1:]):raise SystemExit('Pilot figure is fixed')
    sys.argv[1:1]=['--figure','tradeoff']
    queue.main()
