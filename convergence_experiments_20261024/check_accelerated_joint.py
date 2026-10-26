"""FP extrapolation in the main resource solve only; production positions intact."""
import resource_acceleration as acc
from resource_acceleration import exp, np, source, inner, outer, algo, replace, deepcopy
from controlled_curvature import ROOT, save, Counter
import json,time,zipfile,argparse
from contextlib import ExitStack
from unittest.mock import patch

def run(cfg,g,r,iterations,tag,inner_tolerance=1e-5,outer_tolerance=1e-4,final_probe=False):
    cfg=replace(cfg,tol_inner=inner_tolerance,
        tol_outer=cfg.tol_outer if outer_tolerance is None else outer_tolerance,
        plateau_window=0,tol_window=0.)
    history=[source.e.metrics_values(g,r,cfg)['weighted_sum_rate']]
    rows=[];inner_runs=[];rcg=None;small=0;main_steps=0;candidate_steps=0;tick=time.perf_counter()
    for k in range(iterations):
        ch=acc.build_channels(g,cfg)
        r,stats=acc.accelerated_solve(ch,r,cfg,cfg.max_inner_iter,tolerance=cfg.tol_inner)
        main_steps+=stats['fp_steps']; after_inner=stats['final_score']
        inner_runs.append(stats)
        original=outer.solve_inner_problem;diagnostics=[]
        def counted(ch,rr,cc,verbose=False):
            nonlocal candidate_steps
            result=original(ch,rr,cc,verbose=verbose);candidate_steps+=len(result[1])-1
            return result
        with patch.object(outer,'solve_inner_problem',counted):
            g,ch,r,ff,rcg,accepted=outer.update_positions_once_fp_aware(g,r,cfg,
                rcg_state=rcg,candidate_fp_iterations=2,diagnostic_records=diagnostics)
        source.e.check_state(g,r,cfg,{},'proposed')
        assert ff>=history[-1]-1e-8
        small=small+1 if abs(ff-history[-1])<=cfg.tol_outer else 0
        rows.append(dict(iteration=k+1,wsr=ff,seconds=time.perf_counter()-tick,
            resource_gain=after_inner-history[-1],position_stage_gain=ff-after_inner,
            main_fp_steps=main_steps,candidate_fp_steps=candidate_steps,
            main_fp_steps_this_call=stats['fp_steps'],
            main_after_position_iteration=k if k else None,
            main_inner_tolerance_met=(len(stats['rows'])>=2 and
                max(x['increment'] for x in stats['rows'][-2:])<=cfg.tol_inner),
            accepted_candidate_fp_iterations=sum(d.candidate_fp_iterations for d in diagnostics if d.accepted),
            endpoint_candidate_fp_iterations=[d.candidate_fp_iterations for d in diagnostics if d.accepted],
            accepted_endpoints=accepted,backtracks=sum(d.backtracking_steps for d in diagnostics),
            extrapolation_blocks=len(stats['rows']),
            last_resource_increment=stats['rows'][-1]['increment']))
        history.append(ff)
        result=dict(tag=tag,iterations=k+1,converged=small>=2,
            initial_score=history[0],final_score=ff,gain=ff-history[0],seconds=time.perf_counter()-tick,
            history=history,rows=rows,inner_runs=inner_runs,
            main_fp_steps=main_steps,candidate_fp_steps=candidate_steps,
            inner_tolerance=cfg.tol_inner,outer_tolerance=cfg.tol_outer,
            note='Independent restart; original PR+ and FP-aware-2 positions, accelerated main FP only. Not original RCG continuation.')
        save(ROOT/f'{tag}_accelerated_joint.json',result)
        if (k+1)%5==0 or small>=2:
            print(json.dumps({kk:vv for kk,vv in result.items() if kk not in ('history','rows','inner_runs','note')}),flush=True)
        if small>=2:break
    np.savez(ROOT/f'{tag}_accelerated_joint_final.npz',endpoints=g.endpoints,
        previous_gradients=np.asarray([np.full(3,np.nan) if v is None else v for v in rcg.previous_gradients]),
        previous_directions=np.asarray([np.full(3,np.nan) if v is None else v for v in rcg.previous_directions]),
        **{n:getattr(r,n) for n in ('q_matrix','q_ul','b_ul','u_s')})
    if final_probe:
        # Diagnostic only: the terminal position/resources and convergence result
        # above remain the returned solution. This evaluates the next inner call.
        probe_r,probe=acc.accelerated_solve(acc.build_channels(g,cfg),deepcopy(r),cfg,
            cfg.max_inner_iter,tolerance=cfg.tol_inner)
        source.e.check_state(g,probe_r,cfg,{},'proposed')
        result['post_final_position_probe']=probe
        result['post_final_position_probe']['included_in_solve_runtime']=False
        save(ROOT/f'{tag}_accelerated_joint.json',result)

def main():
    global ROOT
    p=argparse.ArgumentParser();p.add_argument('--power',default='1');p.add_argument('--seed',type=int,default=8003)
    p.add_argument('--iterations',type=int,default=30);p.add_argument('--initial',action='store_true')
    p.add_argument('--inner-tolerance',type=float,default=1e-5)
    p.add_argument('--outer-tolerance',type=float,default=1e-4)
    p.add_argument('--output-subdir')
    p.add_argument('--final-probe',action='store_true')
    args=p.parse_args();hashes=exp.production_hashes()
    if args.output_subdir:
        target=(ROOT/args.output_subdir).resolve()
        if not target.is_relative_to(ROOT.resolve()): raise ValueError('Output must stay within experiment directory')
        ROOT=target;ROOT.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(source.ZIP) as z,ExitStack() as stack:
        for mod in (inner,outer,algo):stack.enter_context(patch.object(mod,'relative_change',source.e.abs_change))
        cfg,g,r=source.load(z,args.power,args.seed)
        if args.initial:
            import io
            from system_model import GeometryState,ResourceState
            folder=f'final_33774/candidate_0/tasks/convergence/{args.power}/scene_{args.seed}/proposed/'
            record=json.loads(z.read(folder+'record.json'));data=z.read(folder+'initial.npz')
            import hashlib
            assert hashlib.sha256(data).hexdigest()==record['initial_sha256']
            with np.load(io.BytesIO(data)) as a:
                g=GeometryState(**{n:a[n].copy() for n in ('endpoints','tx_positions','rx_positions')})
                r=ResourceState(**{n:a[n].copy() for n in ('q_matrix','q_ul','b_ul','u_s')})
        tag=f's{args.seed}_p{args.power}_n{args.iterations}'+('_initial' if args.initial else '')
        run(cfg,g,r,args.iterations,tag,args.inner_tolerance,args.outer_tolerance,args.final_probe)
    assert hashes==exp.production_hashes()
if __name__=='__main__':main()
