"""Finite-difference, tangent curvature and RCG diagnostics on saved final states."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
from dataclasses import replace,asdict
from copy import deepcopy
from contextlib import ExitStack
from unittest.mock import patch
import zipfile,json,time
import numpy as np
import check_larger_step as test
import algorithms as algo,outer_solver as outer,inner_solver as inner
from channel_model import build_channels,compute_performance
from system_model import build_geometry

def relative_error(a,b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12))

def tangent_function(g,r,cfg,index,u,v,fp_budget):
    def evaluate(x,y):
        endpoints=g.endpoints.copy()
        endpoints[index]=outer.sphere_retraction(endpoints[index],x*u+y*v,1.,cfg)
        if fp_budget==0:return outer.evaluate_fixed_resource_wsr(endpoints,r,cfg)
        channels=build_channels(build_geometry(endpoints,cfg,check_constraints=False),cfg)
        # Equal iteration counts are essential for differentiating the candidate map.
        # With early stopping the 0-displacement case can stop after one iteration,
        # while adjacent points execute two and yield a spurious O(1/h^2) Hessian.
        with patch.object(inner,'relative_change',lambda a,b:float('inf')):
            rr,history=inner.solve_inner_problem(channels,deepcopy(r),replace(cfg,max_inner_iter=fp_budget),verbose=False)
        return float(history[-1])
    return evaluate

def derivatives(fun,h):
    f0=fun(0,0);xp=fun(h,0);xm=fun(-h,0);yp=fun(0,h);ym=fun(0,-h)
    grad=np.array([(xp-xm)/(2*h),(yp-ym)/(2*h)])
    cross=(fun(h,h)-fun(h,-h)-fun(-h,h)+fun(-h,-h))/(4*h*h)
    hes=np.array([[(xp-2*f0+xm)/(h*h),cross],[cross,(yp-2*f0+ym)/(h*h)]])
    return grad,hes

def gradients_and_curvature(z):
    rows=[]
    for power,seed in [('0.1',8003),('1',8003),('10',8003),('1',8027),('1',8060)]:
        cfg,g,r=test.load(z,power,seed)
        r,h=inner.solve_inner_problem(build_channels(g,cfg),r,cfg,verbose=False)
        for index in range(cfg.m_uav):
            grads={delta:outer.project_to_tangent(g.endpoints[index],outer.central_difference_gradient(
                g.endpoints,index,r,replace(cfg,finite_diff_delta=delta)),cfg)
                for delta in (1e-4,1e-5,1e-6,1e-7,1e-8)}
            grad=grads[1e-6];u=grad/np.linalg.norm(grad)
            v=np.cross(g.endpoints[index],u);v=v/np.linalg.norm(v)
            fixed=tangent_function(g,r,cfg,index,u,v,0)
            aware=tangent_function(g,r,cfg,index,u,v,2)
            gf,hf=derivatives(fixed,1e-6)
            ga,ha=derivatives(aware,1e-6)
            ga2,ha2=derivatives(aware,3e-6)
            predicted=np.array([float(grad@u),float(grad@v)])
            row=dict(power_W=float(power),seed=seed,endpoint=index,grad_norm=float(np.linalg.norm(grad)),
                delta_relative_errors={str(d):relative_error(x,grads[1e-8]) for d,x in grads.items()},
                tangent_directional_derivative_error=relative_error(predicted,gf),
                fixed_tangent_gradient=gf.tolist(),aware_tangent_gradient=ga.tolist(),
                aware_gradient_relative_change_h=relative_error(ga2,ga),
                cosine_fixed_vs_aware=float(gf@ga/max(np.linalg.norm(gf)*np.linalg.norm(ga),1e-20)),
                fixed_hessian_eigenvalues=np.linalg.eigvalsh(hf).tolist(),
                aware_hessian_eigenvalues=np.linalg.eigvalsh(ha).tolist(),
                aware_hessian_relative_change_h=relative_error(ha2,ha),
                fixed_curvature_along_gradient=float(hf[0,0]),aware_curvature_along_gradient=float(ha[0,0]),
                quadratic_best_alpha_along_raw_gradient=(float(-ga[0]/(ha[0,0]*np.linalg.norm(grad))) if ha[0,0]<0 else None))
            rows.append(row)
        print('GRADIENT/CURVATURE',power,seed,'done',flush=True)
        test.e.json_write(test.OUT/'gradient_curvature_results.json',dict(rows=rows,
            note='Final saved states after one inner solve. Aware derivative probe forces exactly 2 FP iterations at every point, including the center, to remove early-stop discontinuities. Not a global curvature or optimality certificate.'))
    return rows

def compare_rcg(z):
    cfg,g0,r0=test.load(z,'1',8003);runs=[]
    for mode in ('rcg','restart_each_direction'):
        g=deepcopy(g0);r=deepcopy(r0);state=None;history=[];diag=[];tick=time.perf_counter()
        with ExitStack() as local:
            if mode=='restart_each_direction':
                local.enter_context(patch.object(outer,'compute_rcg_direction',lambda point,grad,pg,pd,cfg:(grad.copy(),0.)))
            for iteration in range(1,31):
                before=test.e.metrics_values(g,r,cfg)['weighted_sum_rate']
                r,ih=inner.solve_inner_problem(build_channels(g,cfg),r,cfg,verbose=False)
                endpoint_rows=[]
                # Record angular difference of the actual old-history direction and current gradient.
                original_direction=outer.compute_rcg_direction
                direction_rows=[]
                def record_direction(point,gradient,pg,pd,cfg):
                    direction,beta=original_direction(point,gradient,pg,pd,cfg)
                    direction_rows.append(dict(beta=float(beta),cosine=float(direction@gradient/max(np.linalg.norm(direction)*np.linalg.norm(gradient),1e-20))))
                    return direction,beta
                with patch.object(outer,'compute_rcg_direction',record_direction):
                    ng,ch,nr,score,state,accepted=outer.update_positions_once_fp_aware(g,r,cfg,
                        rcg_state=state,candidate_fp_iterations=2,diagnostic_records=endpoint_rows)
                assert score>=before-1e-8
                test.e.check_state(ng,nr,cfg,{},'proposed')
                history.append(dict(iteration=iteration,before=before,after_inner=ih[-1],after_positions=score,
                    inner_iterations=len(ih)-1,accepted=accepted,
                    displacement_mm=(1000*np.linalg.norm(ng.endpoints-g.endpoints,axis=1)).tolist(),
                    fp_calls=sum(d.candidate_fp_calls for d in endpoint_rows)))
                diag.extend(dict(iteration=iteration,**asdict(d),direction=direction_rows[i]) for i,d in enumerate(endpoint_rows))
                g,r=ng,nr
        run=dict(mode=mode,history=history,diagnostics=diag,seconds=time.perf_counter()-tick,
            gain=history[-1]['after_positions']-history[0]['before'])
        runs.append(run);print('RCG',mode,'gain',run['gain'],'seconds',run['seconds'],flush=True)
        test.e.json_write(test.OUT/'rcg_direction_comparison.json',dict(runs=runs,
            note='Two new 30-iteration runs from identical saved state with no original RCG memory. Restart experiment changes only direction history usage.'))
    return runs

def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--curvature-only',action='store_true');a=ap.parse_args()
    with zipfile.ZipFile(test.ZIP) as z,ExitStack() as stack:
        for mod in (algo,inner,outer):stack.enter_context(patch.object(mod,'relative_change',test.e.abs_change))
        gradients_and_curvature(z)
        if not a.curvature_only:compare_rcg(z)

if __name__=='__main__':main()
