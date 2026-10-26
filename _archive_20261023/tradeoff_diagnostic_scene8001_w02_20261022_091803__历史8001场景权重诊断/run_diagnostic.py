"""Isolated scene8001/w=0.2 controls; never writes the formal project/results."""
from __future__ import annotations
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import sys,json,time,hashlib,traceback,csv
from dataclasses import replace
from unittest.mock import patch
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source_snapshot'))
import figure_engine as engine
import algorithms
from system_model import SystemConfig


class DiagnosticBudgetExceeded(RuntimeError):
    pass


def config_from_record(record):
    data={}
    for key,value in record['effective_config'].items():
        if isinstance(value,list):
            array=np.asarray(value)
            data[key]=array[:,0]+1j*array[:,1] if key in ('xi_s','fading_dl','fading_ul','fading_s') else array
        else: data[key]=value
    return SystemConfig(**data)


def read_input(label,scheme):
    folder=HERE/'inputs'/label/scheme
    record=engine.read_json(folder/'record.json')
    for filename,expected in [('initial.npz',record['initial_sha256']),(record['state_file'],record['state_sha256'])]:
        if engine.sha(folder/filename)!=expected: raise RuntimeError('Input checksum failed')
    if engine.config_digest(config_from_record(record))!=record['cfg_sha256']:
        raise RuntimeError('Input configuration checksum failed')
    g,r,h=engine.load_state(folder/record['state_file'])
    return record,g,r,h


def run_case(job):
    scheme,kind,label,tolerance=job
    folder=HERE/'cases'/f'{scheme}_{kind}'
    folder.mkdir(parents=True,exist_ok=False)
    bundle=engine.read_json(HERE/'source_snapshot/tradeoff_config.json')
    protocol=bundle['protocol'];protocol['_models']=bundle['models']
    reference,_,_,_=read_input('current',scheme)
    source,g,r,_=read_input(label,scheme)
    original=config_from_record(reference)
    a={k:v for k,v in reference['effective_config'].items() if not k.startswith('weight_')}
    b={k:v for k,v in source['effective_config'].items() if not k.startswith('weight_')}
    if a!=b: raise RuntimeError('Candidate uses different non-weight conditions')
    cfg=replace(original,max_outer_iter=100,tol_inner=tolerance,tol_outer=tolerance)
    task=dict(figure='tradeoff',parameter_value=.2,scene_seed=8001,scheme=scheme)
    engine.check_state(g,r,cfg,protocol,scheme)
    initial=engine.metrics_values(g,r,cfg)
    engine.save_state(folder/'diagnostic_initial.npz',g,r,[])
    engine.json_write(folder/'case_protocol.json',dict(task=task,kind=kind,source=label,
        source_record_sha256=engine.sha(HERE/'inputs'/label/scheme/'record.json'),
        tol_inner=tolerance,tol_outer=tolerance,max_outer_iter=100,max_inner_iter=cfg.max_inner_iter,
        max_wall_seconds=300,zero_UL_amplitudes=int(np.sum(r.q_ul==0)),
        intentional_diagnostic_changes=['saved-state starting point','100 additional outer-iteration budget']+
        (['inner and outer absolute tolerance 1e-6'] if tolerance!=1e-4 else []),
        paper_protocol_changed=False,initial_metrics=initial))
    tick=time.perf_counter();progress=[];latest=[g,r];inner_info={}
    original_inner=algorithms.solve_inner_problem
    original_positions=algorithms.update_positions_once_fp_aware if scheme=='proposed' else algorithms._update_ma_positions_once

    def inner(*args,**kwargs):
        value=original_inner(*args,**kwargs)
        history=value[1]
        inner_info.update(iterations=len(history)-1,last_delta=float(history[-1]-history[-2]) if len(history)>1 else None,
                          reached_iteration_limit=len(history)-1>=cfg.max_inner_iter)
        return value

    def positions(*args,**kwargs):
        old_g=args[0]
        value=original_positions(*args,**kwargs)
        if scheme=='proposed': new_g,new_r,accepted=value[0],value[2],value[5]
        else: new_g,new_r,accepted=value[0],args[1],value[3]
        latest[:]=[new_g,new_r]
        metrics=engine.metrics_values(new_g,new_r,cfg)
        prev=progress[-1]['weighted_sum_rate'] if progress else initial['weighted_sum_rate']
        item=dict(iteration=len(progress)+1,elapsed_s=time.perf_counter()-tick,
                  accepted_positions=int(accepted),delta_wsr=metrics['weighted_sum_rate']-prev,
                  max_element_step_m=max(float(np.max(np.linalg.norm(getattr(new_g,k)-getattr(old_g,k),axis=1))) for k in ('tx_positions','rx_positions')),
                  inner=dict(inner_info),**metrics)
        progress.append(item)
        engine.json_write(folder/'progress.json',item)
        if len(progress)%10==0:
            engine.save_state(folder/'diagnostic_latest.npz',new_g,new_r,[initial['weighted_sum_rate']]+[x['weighted_sum_rate'] for x in progress])
        if item['elapsed_s']>=300: raise DiagnosticBudgetExceeded('300-second diagnostic budget reached')
        return value

    patch_name='update_positions_once_fp_aware' if scheme=='proposed' else '_update_ma_positions_once'
    try:
        with patch.object(algorithms,'solve_inner_problem',inner),patch.object(algorithms,patch_name,positions):
            result,history,iterations=engine.run_solver_call(cfg,protocol,task,g,r)
        final_g,final_r=result.geometry,result.resources
        status='converged' if result.converged else 'outer_iteration_budget'
        stop_reason=result.stop_reason
    except DiagnosticBudgetExceeded:
        final_g,final_r=latest
        history=[initial['weighted_sum_rate']]+[x['weighted_sum_rate'] for x in progress]
        iterations=len(progress);status='wall_time_budget';stop_reason=status
    except Exception:
        engine.json_write(folder/'failure.json',dict(error=traceback.format_exc()))
        raise
    engine.check_state(final_g,final_r,cfg,protocol,scheme)
    final=engine.metrics_values(final_g,final_r,cfg)
    if not np.isclose(history[0],initial['weighted_sum_rate'],rtol=1e-9,atol=1e-9):
        raise RuntimeError('Diagnostic initial objective mismatch')
    if np.min(np.diff(history)) < -1e-8*max(1.,max(abs(x) for x in history)):
        raise RuntimeError('Diagnostic objective decreased')
    engine.save_state(folder/'diagnostic_final.npz',final_g,final_r,history)
    engine.json_write(folder/'iterations.json',progress)
    if progress:
        flat=[{k:v for k,v in x.items() if k!='inner'} for x in progress]
        engine.csv_write(folder/'iterations.csv',flat,list(flat[0]))
    summary=dict(scheme=scheme,case=kind,source=label,tolerance=tolerance,status=status,stop_reason=stop_reason,
        additional_outer_iterations=iterations,elapsed_s=time.perf_counter()-tick,
        initial_wsr=initial['weighted_sum_rate'],final_wsr=final['weighted_sum_rate'],
        improvement_from_start=final['weighted_sum_rate']-initial['weighted_sum_rate'],
        improvement_from_original_terminal=final['weighted_sum_rate']-reference['row']['weighted_sum_rate'],
        final_metrics=final,final_UL_power_W=(final_r.q_ul**2).tolist(),
        source_initial_UL_power_W=(r.q_ul**2).tolist(),
        final_state_sha256=engine.sha(folder/'diagnostic_final.npz'),constraint_checks_passed=True)
    engine.json_write(folder/'summary.json',summary)
    return summary


def main():
    jobs=[]
    for kind,tol in [('continue_original_tolerance',1e-4),('continue_tighter_tolerance',1e-6)]:
        jobs.extend((scheme,kind,'current',tol) for scheme in ('proposed','b6'))
    jobs.extend([( 'proposed','alternate_nonzero_weight_seed','mid_weight',1e-4),
                 ('b6','alternate_nonzero_weight_seed','low_weight',1e-4)])
    jobs.extend((scheme,'pure_sensing_seed','pure',1e-4) for scheme in ('proposed','b6'))
    engine.json_write(HERE/'experiment_plan.json',dict(scene=8001,weight=.2,jobs=jobs,workers=2,
        interpretation='Controlled diagnostics, not paper results; best-known candidates are not global certificates'))
    summaries=[];errors=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        pending={pool.submit(run_case,job):job for job in jobs}
        for future in as_completed(pending):
            job=pending[future]
            try:
                result=future.result();summaries.append(result)
                print(json.dumps(result),flush=True)
            except Exception:
                error=dict(job=job,error=traceback.format_exc());errors.append(error);print(json.dumps(error),flush=True)
            engine.json_write(HERE/'summary.json',dict(completed=summaries,errors=errors,planned_cases=len(jobs)))
    if errors: raise SystemExit('Some diagnostic cases failed; see summary.json')


if __name__=='__main__': main()
