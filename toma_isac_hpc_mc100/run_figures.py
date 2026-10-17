"""Five frozen figures: 100 scenes, UL init 25 mW, absolute tolerance 1e-4."""
from __future__ import annotations
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(name,'1')
from pathlib import Path
import sys, argparse, json, hashlib, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
ROOT = Path(__file__).resolve().parent
import figure_engine as engine
import figure_engine as final


def identity(protocol, count):
    files = list(ROOT.glob('*.py'))
    data=dict(protocol=protocol,n_mc=count,sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in sorted(files)})
    return json.loads(json.dumps(data))


def collect(out,p,tasks):
    status = {}
    for figure, tt in tasks.items():
        rows, winners = [], []
        for task in tt:
            group = []
            for start in range(final.n_starts(task)):
                row = final.read_verified(out,p,task,start)
                if row is not None:
                    rows.append(row)
                    group.append(row)
            if len(group)==final.n_starts(task) and all(r['converged'] for r in group):
                winners.append(max(group,key=lambda r:r['weighted_sum_rate']))
        complete = len(winners)==len(tt)
        if rows: engine.csv_write(out/'tables'/f'{figure}_all_starts.csv',rows,list(rows[0]))
        if complete:
            engine.csv_write(out/'tables'/f'{figure}_per_scene.csv',winners,list(winners[0]))
            if figure=='convergence': final.plot_convergence(out,p,tt,winners)
            elif figure=='tradeoff': final.plot_tradeoff(out,p,tt,winners)
            else: engine.plot_sweep(out,p,figure,tt,winners)
        status[figure] = dict(complete=complete,expected_scene_cells=len(tt),completed_scene_cells=len(winners),
            expected_starts=sum(final.n_starts(t) for t in tt),converged_starts=sum(r['converged'] for r in rows))
    engine.json_write(out/'status.json',status)
    complete=all(s['complete'] for s in status.values())
    engine.json_write(out/'completion.json',dict(complete=complete,figures=status,initial_UL_power_per_user_mW=25,
        absolute_tolerance=1e-4,full_five_figures=set(tasks)==set(engine.FIGURES)))
    print(json.dumps(status,indent=2),flush=True)
    return complete


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=Path('simulation_config.json'))
    ap.add_argument('--output',type=Path,default=Path('results/final_mc100_ul25'))
    ap.add_argument('--n-mc',type=int,default=100)
    ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--max-calls',type=int,default=20)
    ap.add_argument('--figures',nargs='+',choices=engine.FIGURES,default=engine.FIGURES)
    ap.add_argument('--scene-index',type=int,help='Slurm array index, zero-based')
    ap.add_argument('--check-only',action='store_true')
    ap.add_argument('--collect-only',action='store_true')
    a=ap.parse_args()
    if min(a.n_mc,a.workers,a.max_calls)<1: ap.error('Positive counts required')
    if len(set(a.figures))!=len(a.figures): ap.error('Duplicate figures')
    if a.scene_index is not None and not 0<=a.scene_index<a.n_mc: ap.error('scene-index out of range')
    if a.collect_only and a.scene_index is not None: ap.error('Collection must include every scene')
    bundle=json.loads((ROOT/a.config).read_text(encoding='utf-8'))
    p=bundle['protocol']
    p['_models']=bundle['models']
    if p['protocol']['initial_ul_power_W']!=.025 or p['protocol']['tol_inner']!=1e-4 or p['protocol']['tol_outer']!=1e-4:
        raise RuntimeError('Frozen protocol requires 25mW and absolute 1e-4')
    tasks={f:engine.build_tasks(ROOT,p,f,a.n_mc) for f in a.figures}
    print('PLAN:',json.dumps({f:dict(scene_cells=len(tt),starts=sum(final.n_starts(t) for t in tt)) for f,tt in tasks.items()}),flush=True)
    if a.check_only:
        # Validate all starts for representative scenes and all sweep settings.
        for tt in tasks.values():
            seen=set()
            for t in tt:
                key=(t['scheme'],t['parameter_value'])
                if key in seen: continue
                seen.add(key)
                cfg=engine.config_for_task(ROOT,p,t)
                for start in range(final.n_starts(t)):
                    g,r=final.pair(cfg,p,t['scheme'],start)
                    engine.check_state(g,r,cfg,p,t['scheme'])
                    if not np.allclose(abs(r.q_ul)**2,.025,rtol=0,atol=1e-12): raise RuntimeError('Wrong UL power')
        tt=next(iter(tasks.values()))
        t=dict(tt[0]);c1=engine.config_for_task(ROOT,p,t);t['scene_seed']+=1;c2=engine.config_for_task(ROOT,p,t)
        if np.array_equal(c1.q_dl,c2.q_dl) or np.array_equal(c1.q_ul,c2.q_ul): raise RuntimeError('Scenes did not change user positions')
        print('CHECK PASSED: frozen solvers, feasible starts, distinct user scenes, 25mW; no optimizer invoked.',flush=True)
        return
    out=(ROOT/a.output).resolve()
    expected=identity(p,a.n_mc)
    # Each array index owns its metadata and task records; collection is a separate job.
    tag='all' if a.scene_index is None else str(a.scene_index)
    meta=out/'metadata'/f'scene_{tag}.json'
    if a.collect_only:
        available=list((out/'metadata').glob('scene_*.json'))
        if not available: raise RuntimeError('No run metadata found')
        if any(engine.read_json(f)!=expected for f in available): raise RuntimeError('Output identity differs')
        if not collect(out,p,tasks): raise SystemExit('Incomplete cells: resume array jobs before collection')
        return
    if out.exists():
        available=list((out/'metadata').glob('scene_*.json'))
        # The Slurm script creates logs before Python initializes run metadata.
        # Concurrent scene processes may also create the metadata directory first.
        # Accept these bootstrap directories, but never adopt orphaned results.
        bootstrap_only=all(entry.name in ('logs','metadata') and entry.is_dir() for entry in out.iterdir())
        if not available and not bootstrap_only: raise RuntimeError('Output has no compatible metadata')
        if any(engine.read_json(f)!=expected for f in available): raise RuntimeError('Output identity differs; use a new output')
    engine.json_write(meta,expected)
    selected=[t for tt in tasks.values() for t in tt if a.scene_index is None or t['scene_seed']==p['scene_seed_start']+a.scene_index]
    jobs=[(str(out),p,t,s,a.max_calls) for t in selected for s in range(final.n_starts(t))]
    errors=[]
    def accept(job,call):
        try:
            row=call()
            if not row['converged']: errors.append(dict(task=job[2],start=job[3],error='Iteration budget exhausted; rerun to resume'))
        except Exception as exc:
            errors.append(dict(task=job[2],start=job[3],error=repr(exc),traceback=traceback.format_exc()))
            print('FAILED:',errors[-1],flush=True)
    if a.workers==1:
        for job in jobs: accept(job,lambda job=job:final.run_job(job))
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            fs={pool.submit(final.run_job,j):j for j in jobs}
            for f in as_completed(fs): accept(fs[f],f.result)
    engine.json_write(out/'logs'/f'scene_{tag}_status.json',dict(complete=not errors,errors=errors))
    if a.scene_index is None:
        done=collect(out,p,tasks)
    else:
        done=not errors
    if errors or not done: raise SystemExit('Incomplete: rerun the same command to resume compatible checkpoints')


if __name__=='__main__': main()
