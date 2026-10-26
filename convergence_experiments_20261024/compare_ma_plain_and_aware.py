"""Paired MA position-search comparison; ordinary FP in both cases."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,time,argparse,hashlib
from dataclasses import replace
from copy import deepcopy
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parent
TOMA=ROOT.parent/'toma_isac_hpc_by_figure/toma'
SUPPORT=TOMA/'final_support_plainfp_o4_i5_s8005'
sys.path[:0]=[str(SUPPORT),str(TOMA)]
import figure_engine as e,algorithms as a,inner_solver as inner,outer_solver as outer
from ma_fp_aware import update_positions_fp_aware

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--scene',type=int,required=True)
    ap.add_argument('--mode',choices=['plain','aware'],required=True);ap.add_argument('--max-outer',type=int,default=400)
    args=ap.parse_args()
    out=ROOT/'ma_plain_vs_fpaware_o4_i5';out.mkdir(exist_ok=True)
    b=e.read_json(SUPPORT/'final_config.json');p=deepcopy(b['protocol']);p['_models']=b['models']
    task=dict(figure='p_dl_max',parameter_value=1.,scene_seed=args.scene,scheme='b6')
    cfg=e.config_for_task(SUPPORT,p,task)
    cfg=replace(cfg,max_outer_iter=args.max_outer,tol_inner=1e-5,tol_outer=1e-4,plateau_window=0,tol_window=0.)
    g,r=e.pair(cfg,p,'b6',0)
    e.check_state(g,r,cfg,p,'b6')
    initial_hash=hashlib.sha256(b''.join(np.ascontiguousarray(v).tobytes() for v in [g.tx_positions,g.rx_positions,r.q_matrix,r.q_ul,r.b_ul,r.u_s])).hexdigest()
    hashes={n:hashlib.sha256((TOMA/n).read_bytes()).hexdigest() for n in ['algorithms.py','inner_solver.py','outer_solver.py','channel_model.py','system_model.py']}
    original_inner=inner.solve_inner_problem
    assert a.solve_inner_problem is original_inner
    assert 'fp_acceleration' not in sys.modules and 'accelerated_adapter' not in sys.modules
    updater=a._update_ma_positions_once if args.mode=='plain' else update_positions_fp_aware
    history=[e.metrics_values(g,r,cfg)['weighted_sum_rate']];rows=[];last={};expected=None;inheritance=0
    tick=time.perf_counter();path=out/f"scene{args.scene}_{args.mode}.json"
    def save(finished=False,result=None):
        data=dict(scene=args.scene,power_W=1.,mode=args.mode,extrapolation=False,
                  initial_geometry='common unscreened start 0; not the future screened winner',
                  outer_tolerance=1e-4,outer_consecutive=2,inner_tolerance=1e-5,max_outer=args.max_outer,max_inner=1200,
                  initial_sha256=initial_hash,shared_source_sha256=hashes,finished=finished,
                  iterations=len(rows),converged=result.converged if result is not None else None,
                  stop_reason=result.stop_reason if result is not None else None,
                  seconds=time.perf_counter()-tick,final_wsr=history[-1],history=history,rows=rows,
                  exact_resource_inheritance_checks=inheritance,
                  total_main_fp_iterations=sum(x['main_fp_iterations'] for x in rows))
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8');os.replace(tmp,path)
        return data
    def main_inner(channels,resources,cc,verbose=False):
        nonlocal last,inheritance
        if expected is not None:
            assert all(np.array_equal(getattr(resources,k),expected[k]) for k in expected)
            inheritance+=1
        t=time.perf_counter();rr,hh=original_inner(channels,resources,cc,verbose=verbose)
        last=dict(main_fp_iterations=len(hh)-1,main_seconds=time.perf_counter()-t,
                  main_tolerance_met=abs(hh[-1]-hh[-2])<=cc.tol_inner)
        return rr,hh
    def positions(geometry,resources,cc,**kw):
        nonlocal expected
        diag=[];t=time.perf_counter()
        result=updater(geometry,resources,cc,**kw,**({'diagnostics':diag} if args.mode=='aware' else {}))
        gg,channels,score,accepted=result
        e.check_state(gg,resources,cc,p,'b6',g)
        assert score>=history[-1]-1e-8
        rows.append(dict(iteration=len(rows)+1,wsr=float(score),increment=float(score-history[-1]),
                         accepted_positions=accepted,position_seconds=time.perf_counter()-t,
                         candidate_fp_iterations=sum(x['candidate_fp_iterations'] for x in diag),**last))
        history.append(float(score));expected={k:getattr(resources,k).copy() for k in ['q_matrix','q_ul','b_ul','u_s']}
        if len(rows)%10==0:save()
        if len(rows)%25==0:print(args.scene,args.mode,'outer',len(rows),'WSR',score,flush=True)
        return result
    q=p['protocol']
    with ExitStack() as stack:
        for mod in [a,inner,outer]:stack.enter_context(patch.object(mod,'relative_change',e.abs_change))
        stack.enter_context(patch.object(a,'solve_inner_problem',main_inner))
        stack.enter_context(patch.object(a,'_update_ma_positions_once',positions))
        result=a.run_ma_fp(cfg,rng=np.random.default_rng(q['resource_seed']),region_size=q['ma_region_size_m'],
                          min_spacing=q['ma_min_spacing_m'],initial_step=q['ma_initial_step_m'],
                          initial_tx_positions=g.tx_positions.copy(),initial_rx_positions=g.rx_positions.copy(),
                          initial_resources=deepcopy(r),verbose=False)
    assert np.allclose(result.wsr_history,history,rtol=0,atol=1e-10)
    assert result.iterations==len(rows) and inheritance==len(rows)-1
    if result.converged:assert max(abs(np.diff(result.wsr_history[-3:])))<=cfg.tol_outer
    assert all(hashlib.sha256((TOMA/k).read_bytes()).hexdigest()==v for k,v in hashes.items())
    e.save_state(out/f'scene{args.scene}_{args.mode}_final.npz',result.geometry,result.resources,result.wsr_history)
    data=save(True,result)
    print('FINISHED',json.dumps({k:v for k,v in data.items() if k not in ['history','rows','shared_source_sha256'] }),flush=True)

if __name__=='__main__':main()
