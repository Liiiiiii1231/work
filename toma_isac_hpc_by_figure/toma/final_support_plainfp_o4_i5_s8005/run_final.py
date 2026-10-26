"""Final figures: screen 20 starts once, then freeze winning initial geometry."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(1,str(ROOT.parent))
import argparse,json,hashlib,shutil
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import figure_engine as e

SHARED=('algorithms.py','channel_model.py','inner_solver.py','outer_solver.py','system_model.py')
FIGURES=('convergence','p_dl_max','rho_si','system_load')


def source_identity(p):
    # Old uploads may leave unused adapters behind; identity follows active code.
    files=[ROOT/n for n in ('run_final.py','figure_engine.py','finalize_run.py','collect_four.py')]+[ROOT.parent/n for n in SHARED]
    return dict(protocol=p,sources={os.path.relpath(f,ROOT).replace('\\','/'):hashlib.sha256(f.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for f in sorted(files)})


def tasks_for(p,figure):
    seeds=p['convergence']['scene_seeds'] if figure=='convergence' else range(p['scene_seed_start'],p['scene_seed_start']+p['n_mc'])
    templates=e.build_tasks(ROOT,p,figure,1)
    tasks=[dict(t,scene_seed=int(s)) for t in templates for s in seeds]
    if any(t['scene_seed'] in p.get('excluded_scene_seeds',[]) for t in tasks):raise ValueError('Excluded scene in final task list')
    return tasks


def run_batch(out,p,jobs,workers):
    metadata=out/'metadata'/'identity.json';expected=source_identity(p)
    if metadata.exists():
        if e.read_json(metadata)!=expected:raise ValueError('Source/config identity differs: use a new output directory')
    elif out.exists() and any(f.name!='logs' for f in out.iterdir()):
        raise ValueError('Existing output lacks identity metadata')
    e.json_write(metadata,expected)
    errors=[]
    with ProcessPoolExecutor(max_workers=min(workers,len(jobs))) as pool:
        futures={pool.submit(e.run_job,j):j for j in jobs}
        for future in as_completed(futures):
            try:
                row=future.result()
                print('TASK FINISHED',e.task_key(row),'start',row['start_id'],'converged',row['converged'],flush=True)
            except Exception as exc:
                job=futures[future];errors.append(dict(task=job[2],start=job[3],error=repr(exc)))
                print('FAILED',errors[-1],flush=True)
    e.json_write(out/'logs'/'task_errors.json',errors)
    if errors:raise RuntimeError('Some tasks failed; inspect task_errors.json. Existing iteration-limit records will not be continued.')


def prepare_selection(p,selection,workers):
    frozen_file=selection/'frozen_initial_geometry.json';expected=source_identity(p)
    if frozen_file.exists():
        d=e.read_json(frozen_file)
        if d['identity']!=expected:raise ValueError('Frozen geometry belongs to different source/config')
        return d
    spec=p['screening']
    if spec['scene_seed'] in p.get('excluded_scene_seeds',[]):raise ValueError('Excluded screening scene')
    taskbase=dict(figure='p_dl_max',parameter_value=spec['p_dl_W'],scene_seed=spec['scene_seed'])
    jobs=[(str(selection),p,dict(taskbase,scheme=sc),s,1) for sc in spec['schemes'] for s in range(spec['candidates_per_scheme'])]
    print('SCREENING',len(jobs),'independent optimization calls',flush=True)
    run_batch(selection,p,jobs,workers)
    winners={};geometries={};all_rows=[]
    for sc in spec['schemes']:
        task=dict(taskbase,scheme=sc)
        rows=[e.read_verified(selection,p,task,s) for s in range(spec['candidates_per_scheme'])]
        if any(r is None for r in rows):raise RuntimeError('Missing screening result')
        winner=max(rows,key=lambda r:(r['weighted_sum_rate'],-r['start_id']))
        folder=selection/f"candidate_{winner['start_id']}"/'tasks'/e.task_key(task)
        initial,resources,_=e.load_state(folder/'initial.npz')
        geometries[sc]={k:getattr(initial,k).tolist() for k in ('endpoints','tx_positions','rx_positions')}
        winners[sc]=dict(start_id=winner['start_id'],reference_wsr=winner['weighted_sum_rate'],reference_converged=winner['converged'],screening_converged=sum(r['converged'] for r in rows),initial_sha256=e.sha(folder/'initial.npz'))
        all_rows.extend(rows)
    e.csv_write(selection/'screening_comparison.csv',all_rows,list(all_rows[0]))
    d=dict(identity=expected,reference=spec,winners=winners,geometry=geometries,
           freeze_definition='Winning INITIAL geometry only. All resource states rebuilt independently for every final task.')
    e.json_write(frozen_file,d)
    print('FROZEN GEOMETRY READY:',frozen_file,flush=True)
    print('The other three figure jobs may now be submitted.',flush=True)
    return d


def collect(out,p,figure,tasks):
    rows=[e.read_verified(out,p,t,0) for t in tasks]
    if any(r is None for r in rows):raise RuntimeError('Missing final task: no partial Monte Carlo averages are allowed')
    e.csv_write(out/'tables'/f'{figure}_per_scene.csv',rows,list(rows[0]))
    if figure=='convergence':e.plot_convergence(out,p,tasks,rows)
    else:e.plot_sweep(out,p,figure,tasks,rows)
    status=dict(complete=True,all_converged=all(r['converged'] for r in rows),figure=figure,
                expected_tasks=len(tasks),finished_tasks=len(rows),converged_tasks=sum(r['converged'] for r in rows),
                excluded_scene_seeds=p.get('excluded_scene_seeds',[]),scene_selection_note=p.get('scene_selection_note',''),
                historically_selected_scenes=(figure=='convergence'),scene_seeds=sorted(set(t['scene_seed'] for t in tasks)),
                completion_definition='All single-pass tasks finished, not necessarily converged')
    e.json_write(out/'completion.json',status);print(json.dumps(status,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--figure',choices=FIGURES,required=True)
    ap.add_argument('--config',type=Path,default=ROOT/'final_config.json')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--selection-dir',type=Path,default=ROOT.parent/'results'/'final_initial_geometry_plainfp_o4_i5_mafp_final_s8005')
    ap.add_argument('--workers',type=int,default=56)
    ap.add_argument('--check-only',action='store_true')
    a=ap.parse_args()
    if a.workers<1:ap.error('workers must be positive')
    expected_shared=e.read_json(ROOT/'shared_module_hashes.json')
    for name,h in expected_shared.items():
        path=ROOT.parent/name
        if not path.is_file() or hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=h:
            raise ValueError('Shared module differs from verified version: '+str(path))
    b=e.read_json(a.config);p=b['protocol'];p['_models']=b['models'];e.STARTS.update({s:1 for s in p['schemes']})
    tasks=tasks_for(p,a.figure)
    print('PLAN',a.figure,'tasks',len(tasks),'workers',a.workers,'scenes',len(set(t['scene_seed'] for t in tasks)),flush=True)
    if a.check_only:
        spec=p['screening']
        for sc in spec['schemes']:
            t=dict(figure='p_dl_max',parameter_value=spec['p_dl_W'],scene_seed=spec['scene_seed'],scheme=sc)
            cfg=e.config_for_task(ROOT,p,t)
            for s in range(spec['candidates_per_scheme']):
                g,r=e.pair(cfg,p,sc,s);e.check_state(g,r,cfg,p,sc)
        seen=set()
        for t in tasks:
            key=(t['parameter_value'],t['scheme'])
            if key in seen:continue
            seen.add(key);cfg=e.config_for_task(ROOT,p,t);g,r=e.pair(cfg,p,t['scheme'],0);e.check_state(g,r,cfg,p,t['scheme'])
            if cfg.n_tx!=16 or cfg.n_rx!=16:raise ValueError('Array size mismatch')
        print('CHECK PASSED: 20 feasible starts per optimized scheme, 16x16 arrays and final task setup',flush=True);return
    selection=a.selection_dir.resolve()
    if a.figure=='convergence':frozen=prepare_selection(p,selection,a.workers)
    else:
        file=selection/'frozen_initial_geometry.json'
        if not file.exists():raise ValueError('Submit convergence first; wait for FROZEN GEOMETRY READY before submitting this figure')
        frozen=e.read_json(file)
        if frozen['identity']!=source_identity(p):raise ValueError('Frozen geometry identity mismatch')
    p=deepcopy(p);p['_frozen_geometry']=frozen['geometry']
    jobs=[(str(a.output.resolve()),p,t,0,1) for t in tasks]
    run_batch(a.output.resolve(),p,jobs,a.workers)
    shutil.copy2(selection/'frozen_initial_geometry.json',a.output/'frozen_initial_geometry.json')
    shutil.copy2(selection/'screening_comparison.csv',a.output/'screening_comparison.csv')
    if a.figure=='convergence':shutil.copytree(selection,a.output/'screening_details',dirs_exist_ok=True)
    collect(a.output.resolve(),p,a.figure,tasks)


if __name__=='__main__':main()
