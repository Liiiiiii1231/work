"""Frozen neighbor selection followed by actual target-weight optimization.

Upload beside run_tradeoff.py. Original runs are read-only; refinement uses a
new output directory and its own explicitly labelled continuation records.
"""
from __future__ import annotations
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import argparse,json,sys,time,hashlib,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np


def engine_at(root):
    sys.path.insert(0,str(root))
    import figure_engine
    return figure_engine


def source_hashes(root):
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py'))}


def verify_refined(e,folder,saved,identity,cfg,p,task,g0):
    if saved['identity']!=identity or saved['cfg_sha256']!=e.config_digest(cfg):raise RuntimeError('Refinement identity/config mismatch')
    if any(saved['row'].get(k)!=v for k,v in task.items()):raise RuntimeError('Refinement task mismatch')
    if e.sha(folder/'selected_seed.npz')!=saved['seed_sha256'] or e.sha(folder/saved['state_file'])!=saved['state_sha256']:
        raise RuntimeError('Refinement state/seed checksum mismatch')
    g,r,h=e.load_state(folder/saved['state_file']);e.check_state(g,r,cfg,p,task['scheme'],g0)
    metrics=e.metrics_values(g,r,cfg)
    if any(not np.isclose(v,saved['row'][k],rtol=1e-9,atol=1e-9) for k,v in metrics.items()):raise RuntimeError('Refinement metrics mismatch')
    if len(h)<2 or not np.all(np.isfinite(h)) or np.min(np.diff(h))<-1e-8*max(1.,max(abs(x) for x in h)):raise RuntimeError('Invalid refinement trajectory')
    if not np.isclose(h[0],saved['row']['selected_seed_wsr'],rtol=1e-9,atol=1e-9) or not np.isclose(h[-1],metrics['weighted_sum_rate'],rtol=1e-9,atol=1e-9):
        raise RuntimeError('Refinement trajectory endpoint mismatch')
    if saved['row']['converged']:
        required=2 if task['scheme'] in ('proposed','b6') else 1
        if len(h)<required+1 or max(abs(np.diff(h)[-required:]))>1e-4:raise RuntimeError('Refinement stopping check failed')
    return g,r,h


def run_job(job):
    root,source,out,p,task,chosen,identity,max_calls=job
    e=engine_at(Path(root));source,out=Path(source),Path(out)
    cfg=e.config_for_task(Path(root),p,task)
    folder=out/'refinement_tasks'/e.task_key(task)
    folder.mkdir(parents=True,exist_ok=True)
    record_file=folder/'refinement_record.json'
    chosen_task=dict(task,parameter_value=chosen['parameter_value'])
    src=source/'candidate_0/tasks'/e.task_key(chosen_task)
    original=source/'candidate_0/tasks'/e.task_key(task)
    for input_task,input_folder in [(chosen_task,src),(task,original)]:
        expected_input=identity['original_input_sha256'][e.task_key(input_task)]
        if e.sha(input_folder/'record.json')!=expected_input['record_sha256'] or e.sha(input_folder/'initial.npz')!=expected_input['initial_sha256']:
            raise RuntimeError('Original input changed after preflight')
    seed_record=e.read_json(src/'record.json')
    if e.sha(src/seed_record['state_file'])!=identity['original_input_sha256'][e.task_key(chosen_task)]['state_sha256']:
        raise RuntimeError('Selected source state checksum changed')
    cfg_src=seed_record['effective_config']
    cfg_ref=e.read_json(original/'record.json')['effective_config']
    if {k:v for k,v in cfg_src.items() if not k.startswith('weight_')}!={k:v for k,v in cfg_ref.items() if not k.startswith('weight_')}:
        raise RuntimeError('Neighbor differs in non-weight configuration')
    g,r,_=e.load_state(src/seed_record['state_file'])
    g0,_,_=e.load_state(original/'initial.npz')
    e.check_state(g,r,cfg,p,task['scheme'],g0)
    initial=e.metrics_values(g,r,cfg)
    baseline=e.read_json(original/'record.json')['row']
    if initial['weighted_sum_rate']+1e-9<baseline['weighted_sum_rate']:
        raise RuntimeError('Selected source worse than original under target weight')
    seed_path=folder/'selected_seed.npz'
    if not seed_path.exists():e.save_state(seed_path,g,r,[])
    seed_g,seed_r,_=e.load_state(seed_path)
    for key in ('endpoints','tx_positions','rx_positions'):
        if not np.array_equal(getattr(seed_g,key),getattr(g,key)):raise RuntimeError('Refinement seed geometry differs')
    for key in ('q_matrix','q_ul','b_ul','u_s'):
        if not np.array_equal(getattr(seed_r,key),getattr(r,key)):raise RuntimeError('Refinement seed resources differ')
    fingerprint=e.config_digest(cfg);history=[];calls=0;iterations=0;runtime=0.
    if record_file.exists():
        saved=e.read_json(record_file)
        if saved['identity']!=identity or saved['cfg_sha256']!=fingerprint or saved['seed_sha256']!=e.sha(seed_path):
            raise RuntimeError('Refinement identity differs; use a new output')
        if e.sha(folder/saved['state_file'])!=saved['state_sha256']:raise RuntimeError('Refinement checkpoint checksum mismatch')
        g,r,history=verify_refined(e,folder,saved,identity,cfg,p,task,g0)
        calls=saved['row']['calls'];iterations=saved['row']['iterations'];runtime=saved['row']['runtime_s']
        required=2 if task['scheme'] in ('proposed','b6') else 1
        if saved['row']['converged']:
            if len(history)<required+1 or max(abs(np.diff(history)[-required:]))>1e-4:raise RuntimeError('Invalid saved stopping condition')
            return saved['row']
    for _ in range(max_calls):
        if source_hashes(Path(root))!=identity['solver_source_sha256']:raise RuntimeError('Source code changed during refinement')
        before=e.metrics_values(g,r,cfg)['weighted_sum_rate'];tick=time.perf_counter()
        print('REFINE_START',e.task_key(task),'call',calls+1,'previous_steps',iterations,flush=True)
        result,new,steps=e.run_solver_call(cfg,p,task,g,r)
        runtime+=time.perf_counter()-tick;calls+=1;iterations+=steps
        if not np.isclose(new[0],before,rtol=1e-9,atol=1e-9):raise RuntimeError('Continuation objective mismatch')
        history=history+new[1:] if history else new
        if np.min(np.diff(history))<-1e-8*max(1.,max(abs(v) for v in history)):raise RuntimeError('Refinement objective decreased')
        g,r=result.geometry,result.resources;e.check_state(g,r,cfg,p,task['scheme'],g0)
        metrics=e.metrics_values(g,r,cfg)
        if not np.isclose(metrics['weighted_sum_rate'],history[-1],rtol=1e-9,atol=1e-9):raise RuntimeError('Final objective mismatch')
        if metrics['weighted_sum_rate']+1e-9<baseline['weighted_sum_rate']:raise RuntimeError('Refined result worse than baseline')
        converged=bool(result.converged);required=2 if task['scheme'] in ('proposed','b6') else 1
        if converged and (len(new)<required+1 or max(abs(np.diff(new)[-required:]))>1e-4):raise RuntimeError('Absolute stopping condition failed')
        state=f'refine_call_{calls:03d}.npz';e.save_state(folder/state,g,r,history)
        row=dict(**task,**metrics,iterations=iterations,runtime_s=runtime,calls=calls,converged=converged,
                 selected_source_weight=chosen['parameter_value'],baseline_wsr=baseline['weighted_sum_rate'],selected_seed_wsr=initial['weighted_sum_rate'])
        e.json_write(record_file,dict(identity=identity,cfg_sha256=fingerprint,seed_sha256=e.sha(seed_path),state_file=state,state_sha256=e.sha(folder/state),row=row))
        print('REFINE_DONE',e.task_key(task),'WSR',metrics['weighted_sum_rate'],'converged',converged,'steps',iterations,flush=True)
        if converged:return row
    return row


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',type=Path,default=Path('/home/lyq00008339/Lihuang_Li/toma'))
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--config',type=Path,default=Path('tradeoff_config.json'))
    ap.add_argument('--scene-seed',type=int,default=8001)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--max-calls',type=int,default=20)
    ap.add_argument('--check-only',action='store_true')
    ap.add_argument('--collect-only',action='store_true')
    a=ap.parse_args()
    if min(a.workers,a.max_calls)<1:ap.error('Positive counts required')
    root=a.project_root.resolve();source=a.source.resolve();out=a.output.resolve();e=engine_at(root)
    if out==source or source in out.parents or out==root:raise ValueError('Use a separate new output directory')
    config_path=(root/a.config).resolve();bundle=e.read_json(config_path);p=bundle['protocol'];p['_models']=bundle['models']
    if p['protocol']['tol_inner']!=1e-4 or p['protocol']['tol_outer']!=1e-4:raise RuntimeError('This control uses the original absolute 1e-4 tolerances')
    weights=sorted(p['tradeoff']['weights']);schemes=p['schemes']
    tasks=[dict(figure='tradeoff',parameter_value=w,scene_seed=a.scene_seed,scheme=sc) for w in weights for sc in schemes]
    rows={};source_inputs={};missing=[]
    for task in tasks:
        folder=source/'candidate_0/tasks'/e.task_key(task)
        if not (folder/'record.json').exists():missing.append(str(folder/'record.json'));continue
        row=e.read_verified(source,p,task,0)
        if row is None or not row['converged']:raise RuntimeError('Original candidate is missing or unconverged')
        rows[(task['scheme'],task['parameter_value'])]=row
        record=e.read_json(folder/'record.json')
        source_inputs[e.task_key(task)]=dict(record_sha256=e.sha(folder/'record.json'),initial_sha256=record['initial_sha256'],state_sha256=record['state_sha256'])
    if missing:raise FileNotFoundError('Missing original checkpoint files:\n'+'\n'.join(missing))
    selected={};selection=[]
    for sc in schemes:
        for index,w in enumerate(weights):
            own=rows[(sc,w)];pool=[rows[(sc,x)] for x in weights[max(0,index-1):min(len(weights),index+2)]]
            score=lambda r:w*r['sum_rate_comm']/10+(1-w)*r['sum_rate_sensing']/5
            candidate=max(pool,key=lambda r:(score(r),r is own));selected[(sc,w)]=candidate
            selection.append(dict(scheme=sc,target_weight=w,source_weight=candidate['parameter_value'],baseline_wsr=score(own),selected_wsr=score(candidate)))
    print(f'CHECK PASSED: {len(tasks)} verified states; frozen immediate-neighbor WSR selection; scene={a.scene_seed}',flush=True)
    if a.check_only:return
    identity=dict(scene=a.scene_seed,source=str(source),weights=weights,schemes=schemes,rule='one-pass WSR-only immediate-neighbor selection then target-weight optimization',
                  solver_source_sha256=source_hashes(root),config_sha256=e.sha(config_path),helper_sha256=e.sha(Path(__file__)),original_input_sha256=source_inputs)
    manifest=out/'refinement_identity.json'
    if out.exists():
        if not manifest.exists() or e.read_json(manifest)!=identity:raise RuntimeError('Output identity differs; use a new refinement directory')
    else:out.mkdir(parents=True)
    e.json_write(manifest,identity);e.csv_write(out/'tables/selection.csv',selection,list(selection[0]))
    errors=[]
    if not a.collect_only:
        jobs=[(str(root),str(source),str(out),p,t,selected[(t['scheme'],t['parameter_value'])],identity,a.max_calls) for t in tasks]
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            pending={pool.submit(run_job,j):j[4] for j in jobs}
            for future in as_completed(pending):
                try:future.result()
                except Exception:errors.append(dict(task=pending[future],error=traceback.format_exc()))
                e.json_write(out/'errors.json',errors)
    refined=[];incomplete=[]
    for task in tasks:
        folder=out/'refinement_tasks'/e.task_key(task);record=folder/'refinement_record.json'
        if not record.exists():incomplete.append(task);continue
        saved=e.read_json(record)
        cfg=e.config_for_task(root,p,task)
        g0,_,_=e.load_state(source/'candidate_0/tasks'/e.task_key(task)/'initial.npz')
        verify_refined(e,folder,saved,identity,cfg,p,task,g0)
        if not saved['row']['converged']:incomplete.append(task);continue
        refined.append(saved['row'])
    e.json_write(out/'refinement_completion.json',dict(complete=not errors and not incomplete,expected=len(tasks),converged=len(refined),errors=errors,incomplete=incomplete))
    if errors or incomplete:raise SystemExit('Refinement incomplete: preserve output and rerun the same command; no partial curve drawn')
    if source_hashes(root)!=identity['solver_source_sha256']:raise RuntimeError('Source code changed during refinement')
    original_rows=[rows[(t['scheme'],t['parameter_value'])] for t in tasks]
    e.csv_write(out/'tables/original_per_scene.csv',original_rows,list(original_rows[0]))
    e.csv_write(out/'tables/refined_per_scene.csv',refined,list(refined[0]))
    plt=e.plotting();fig,axs=plt.subplots(1,3,figsize=(18,5.7),sharex=True,sharey=True)
    for ax,mode,title in zip(axs,['original','selected','refined'],['Original independent solutions','Neighbor selection only','Neighbor selection + optimization']):
        for sc in schemes:
            data=[]
            for w in weights:
                if mode=='original':data.append(rows[(sc,w)])
                elif mode=='selected':data.append(selected[(sc,w)])
                else:data.append(next(r for r in refined if r['scheme']==sc and r['parameter_value']==w))
            style=e.STYLES[sc];ax.plot([r['sum_rate_comm'] for r in data],[r['sum_rate_sensing'] for r in data],color=style['color'],marker=style['marker'],label=style['label'],lw=1.5,ms=4)
        ax.set(xlim=(0,None),ylim=(0,None),xlabel='Communication sum rate (bit/s/Hz)',title=title)
        if hasattr(e,'inside_legend'):e.inside_legend(ax,fontsize=8)
        else:ax.legend(loc='best',fontsize=8)
    axs[0].set_ylabel('Sensing sum rate (bit/s/Hz)')
    xmax=max(ax.get_xlim()[1] for ax in axs);ymax=max(ax.get_ylim()[1] for ax in axs)
    for ax in axs:ax.set_xlim(0,xmax);ax.set_ylim(0,ymax)
    fig.suptitle(f'Scene {a.scene_seed} | absolute tolerance 1e-4 | diagnostic warm-start protocol')
    fig.tight_layout(rect=(0,0,1,.94));e.save_figure(fig,out,'neighbor_refinement_comparison');plt.close(fig)
    print(f'All {len(tasks)} target-weight cases converged. Results: {out}',flush=True)


if __name__=='__main__':main()
