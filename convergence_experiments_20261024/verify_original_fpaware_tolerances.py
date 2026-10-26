"""Original Proposed only: plain FP + unchanged FP-aware-2 resource inheritance.

Runs the unchanged joint algorithm once from initial.npz. Wrappers only collect
diagnostics. No extrapolation, curvature preconditioning or direction restarts.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
            'BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,json,zipfile,hashlib,io,time,argparse
from dataclasses import replace,asdict
from copy import deepcopy
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parent
TOMA=ROOT.parent/'toma_isac_hpc_by_figure/toma'
SUPPORT=TOMA/'final_support_i1200_o1500_s8005'
sys.path[:0]=[str(SUPPORT),str(TOMA)]
import algorithms as algo,inner_solver as inner,outer_solver as outer,figure_engine as e
from system_model import SystemConfig,GeometryState,ResourceState
FIELDS=['q_matrix','q_ul','b_ul','u_s']
OUT=ROOT/'original_fpaware_no_extrapolation'

def write(path,data):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)

def load(seed,power):
    with zipfile.ZipFile(Path.home()/'Downloads/final_33774_bundle.zip') as z:
        folder=f'final_33774/candidate_0/tasks/convergence/{power}/scene_{seed}/proposed/'
        record=json.loads(z.read(folder+'record.json'));raw=record['effective_config'].copy()
        for k,v in list(raw.items()):
            if isinstance(v,list):
                a=np.asarray(v);raw[k]=a[:,0]+1j*a[:,1] if k in ['xi_s','fading_dl','fading_ul','fading_s'] else a
        cfg=SystemConfig(**raw);assert e.config_digest(cfg)==record['cfg_sha256']
        data=z.read(folder+'initial.npz');assert hashlib.sha256(data).hexdigest()==record['initial_sha256']
        for name in ['algorithms.py','inner_solver.py','outer_solver.py','channel_model.py','system_model.py']:
            assert (TOMA/name).read_bytes().replace(b'\r\n',b'\n')==z.read('final_33774/provenance/shared_modules/'+name).replace(b'\r\n',b'\n')
        with np.load(io.BytesIO(data)) as a:
            g=GeometryState(**{n:a[n].copy() for n in ['endpoints','tx_positions','rx_positions']})
            r=ResourceState(**{n:a[n].copy() for n in FIELDS})
    return cfg,g,r,record['initial_sha256']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--inner-tolerance',type=float,required=True)
    ap.add_argument('--outer-tolerance',type=float,default=1e-4)
    ap.add_argument('--power',default='1');ap.add_argument('--max-outer',type=int,default=1500)
    args=ap.parse_args();OUT.mkdir(exist_ok=True)
    cfg,g,r,initial_sha=load(args.seed,args.power)
    cfg=replace(cfg,tol_inner=args.inner_tolerance,tol_outer=args.outer_tolerance,max_inner_iter=1200,
        max_outer_iter=args.max_outer,plateau_window=0,tol_window=0.)
    outer_tag='1e-4' if args.outer_tolerance==1e-4 else f'{args.outer_tolerance:g}'
    tag=f's{args.seed}_p{args.power}_outer{outer_tag}_inner{args.inner_tolerance:g}'
    path=OUT/f'{tag}.json'
    original_inner=inner.solve_inner_problem;original_positions=outer.update_positions_once_fp_aware
    assert algo.solve_inner_problem is original_inner and outer.solve_inner_problem is original_inner
    hashes={n:hashlib.sha256((TOMA/n).read_bytes()).hexdigest() for n in
        ['algorithms.py','inner_solver.py','outer_solver.py','channel_model.py','system_model.py']}
    assert 'fp_acceleration' not in sys.modules and 'accelerated_adapter' not in sys.modules
    history=[e.metrics_values(g,r,cfg)['weighted_sum_rate']]
    rows=[];last_main={};expected_resources=None;inheritance_checks=0;tick=time.perf_counter();small=0

    def main_inner(channels,resources,cc,verbose=False):
        nonlocal last_main,inheritance_checks
        if expected_resources is not None:
            assert all(np.array_equal(getattr(resources,n),expected_resources[n]) for n in FIELDS)
            inheritance_checks+=1
        start=time.perf_counter()
        rr,hh=original_inner(channels,resources,cc,verbose=verbose)
        last_main=dict(main_fp_iterations=len(hh)-1,main_seconds=time.perf_counter()-start,
            main_gain=float(hh[-1]-hh[0]),main_last_increment=float(hh[-1]-hh[-2]),
            main_tolerance_met=abs(hh[-1]-hh[-2])<=cc.tol_inner)
        return rr,hh

    def positions(geometry,resources,cc,**kwargs):
        nonlocal expected_resources,small
        diag=[];kwargs['diagnostic_records']=diag
        start=time.perf_counter()
        result=original_positions(geometry,resources,cc,**kwargs)
        gg,ch,rr,score,rcg,accepted=result
        e.check_state(gg,rr,cc,{},'proposed')
        assert np.isfinite(score) and score>=history[-1]-1e-8
        assert all(d.candidate_fp_iterations<=2 for d in diag)
        increment=float(score-history[-1]);small=small+1 if abs(increment)<=cc.tol_outer else 0
        rows.append(dict(iteration=len(rows)+1,wsr=float(score),increment=increment,
            seconds=time.perf_counter()-tick,position_seconds=time.perf_counter()-start,
            accepted_endpoints=accepted,backtracks=sum(d.backtracking_steps for d in diag),
            candidate_fp_calls=sum(d.candidate_fp_calls for d in diag),
            candidate_actual_fp_iterations=sum(d.total_candidate_fp_iterations for d in diag),
            accepted_candidate_fp_iterations=sum(d.candidate_fp_iterations for d in diag if d.accepted),
            **last_main))
        history.append(float(score));expected_resources={n:getattr(rr,n).copy() for n in FIELDS}
        status=dict(algorithm='original_plain_FP_with_FP_aware_2',extrapolation=False,
            scene=args.seed,power_W=float(args.power),inner_tolerance=cc.tol_inner,outer_tolerance=cc.tol_outer,
            outer_consecutive=2,max_outer=args.max_outer,max_inner=1200,
            initial_sha256=initial_sha,source_sha256=hashes,finished=False,
            iterations=len(rows),converged=None,seconds=time.perf_counter()-tick,
            initial_wsr=history[0],final_wsr=history[-1],history=history,rows=rows,
            exact_resource_inheritance_checks=inheritance_checks)
        if len(rows)%10==0 or small>=2:
            write(path,status)
        if len(rows)%25==0 or small>=2:
            print(json.dumps({k:v for k,v in status.items() if k not in ['history','rows','source_sha256']},ensure_ascii=False),flush=True)
        if len(rows)%50==0 or small>=2:
            arrays={n:getattr(rr,n) for n in FIELDS}
            arrays.update(endpoints=gg.endpoints,tx_positions=gg.tx_positions,rx_positions=gg.rx_positions,
                wsr_history=np.asarray(history),
                previous_gradients=np.asarray(rcg.previous_gradients),
                previous_directions=np.asarray(rcg.previous_directions))
            with (OUT/f'{tag}_checkpoint.npz').open('wb') as stream:np.savez_compressed(stream,**arrays)
        return result

    with ExitStack() as stack:
        for mod in [algo,inner,outer]:stack.enter_context(patch.object(mod,'relative_change',e.abs_change))
        stack.enter_context(patch.object(algo,'solve_inner_problem',main_inner))
        stack.enter_context(patch.object(algo,'update_positions_once_fp_aware',positions))
        result=algo.run_joint_algorithm(cfg,rng=np.random.default_rng(cfg.random_seed),verbose=False,
            position_update_mode='fp_aware',fp_aware_candidate_iters=2,
            initial_endpoints=g.endpoints.copy(),initial_resources=deepcopy(r))
    assert result.joint_wsr_history==history
    assert result.outer_iterations==len(rows)
    assert all(hashlib.sha256((TOMA/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
    if result.converged:assert max(abs(np.diff(history[-3:])))<=cfg.tol_outer
    assert inheritance_checks==len(rows)-1
    e.check_state(result.geometry,result.resources,cfg,{},'proposed')
    e.save_state(OUT/f'{tag}_final.npz',result.geometry,result.resources,history)
    status=dict(algorithm='original_plain_FP_with_FP_aware_2',extrapolation=False,
        scene=args.seed,power_W=float(args.power),inner_tolerance=cfg.tol_inner,outer_tolerance=cfg.tol_outer,
        outer_consecutive=2,max_outer=args.max_outer,max_inner=1200,
        initial_sha256=initial_sha,source_sha256=hashes,finished=True,
        iterations=len(rows),converged=result.converged,stop_reason=result.stop_reason,
        seconds=time.perf_counter()-tick,initial_wsr=history[0],final_wsr=history[-1],
        history=history,rows=rows,exact_resource_inheritance_checks=inheritance_checks,
        total_main_fp_iterations=sum(x['main_fp_iterations'] for x in rows),
        total_candidate_fp_iterations=sum(x['candidate_actual_fp_iterations'] for x in rows))
    write(path,status)
    print('FINISHED',json.dumps({k:v for k,v in status.items() if k not in ['history','rows','source_sha256']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
