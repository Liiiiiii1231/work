"""Projected squared extrapolation of the resource FP map.

Only main resource solves use this accelerator. Position candidates retain the
original at-most-two-step FP map. The budget counts every actual FP iteration,
including rejected extrapolation trials; history contains accepted block WSRs.
"""
from copy import deepcopy
from dataclasses import replace
import time
import numpy as np
from inner_solver import solve_inner_problem as plain_fp
from channel_model import compute_performance
from system_model import check_resource_constraints

DEFAULT_POLICY = dict(method='projected_squared_extrapolation',max_extrapolation=64.,
    consecutive_small_blocks=2,scope='main_resource_solves_all_schemes',
    candidate_solver='plain_fp_max_2')


def pack(resources):
    q=resources.q_matrix.ravel()
    return np.r_[q.real,q.imag,resources.q_ul]


def project(vector, template, cfg):
    resources=deepcopy(template);n=resources.q_matrix.size
    resources.q_matrix=(vector[:n]+1j*vector[n:2*n]).reshape(resources.q_matrix.shape)
    power=np.linalg.norm(resources.q_matrix)**2
    if power>cfg.p_dl_max:
        resources.q_matrix*=np.sqrt(cfg.p_dl_max/power)
    resources.q_ul=np.clip(vector[2*n:],0,np.sqrt(cfg.p_ul_max))
    check_resource_constraints(resources,cfg)
    return resources


def solve_accelerated(channels, resources, cfg, *, policy=None, verbose=False):
    policy=DEFAULT_POLICY if policy is None else policy
    if policy!=DEFAULT_POLICY:
        raise ValueError('Resource solver policy differs from the validated release')
    if cfg.max_inner_iter<1 or cfg.tol_inner<=0:
        raise ValueError('Positive FP budget and inner tolerance are required')
    resources=deepcopy(resources)
    score=lambda r:float(compute_performance(channels,r,cfg).weighted_sum_rate)
    value=score(resources);history=[value];blocks=[];work=0;small=0
    one=replace(cfg,max_inner_iter=1);tick=time.perf_counter()

    def step(r):
        nonlocal work
        rr,hh=plain_fp(channels,r,one,verbose=False)
        work+=len(hh)-1
        if len(hh)!=2 or not np.isfinite(hh[-1]):
            raise RuntimeError('Expected one finite FP update')
        return rr,float(hh[-1])

    while work<cfg.max_inner_iter:
        before=value;start_work=work
        alpha=None;attempted=None;backtracks=0;accepted=False
        if cfg.max_inner_iter-work>=3:
            x=pack(resources)
            r1,f1=step(resources);r2,f2=step(r1)
            delta=pack(r1)-x;curvature=pack(r2)-2*pack(r1)+x
            alpha=min(64.,max(1.,float(np.linalg.norm(delta)/max(np.linalg.norm(curvature),1e-30))))
            attempted=alpha
            while work<cfg.max_inner_iter:
                trial=project(x+2*alpha*delta+alpha**2*curvature,r2,cfg)
                trial,trial_score=step(trial)
                if trial_score>=f2:
                    resources,value=trial,trial_score;accepted=True;break
                if alpha<=1.00001:break
                alpha=1.+(alpha-1.)*.5;backtracks+=1
            if not accepted:resources,value=r2,f2
        else:
            # A budget shorter than a block degrades to ordinary FP. This also
            # handles the final one/two available evaluations without overruns.
            resources,value=step(resources)
        check_resource_constraints(resources,cfg)
        if not np.isfinite(value) or value<before-1e-8:
            raise RuntimeError('Accelerated resource WSR decreased or is nonfinite')
        history.append(value)
        increment=value-before
        small=small+1 if abs(increment)<=cfg.tol_inner else 0
        blocks.append(dict(block=len(blocks)+1,actual_fp_iterations=work-start_work,
            cumulative_fp_iterations=work,wsr=value,increment=increment,
            alpha=alpha,attempted_alpha=attempted,backtracks=backtracks,
            extrapolation_accepted=accepted))
        if small>=2:break
    if work>cfg.max_inner_iter:raise RuntimeError('Actual FP budget exceeded')
    diagnostics=dict(actual_fp_iterations=work,accepted_blocks=len(blocks),
        converged=small>=2,stop_reason='inner_absolute_tolerance' if small>=2 else 'actual_fp_budget',
        tolerance=cfg.tol_inner,consecutive_small_blocks=2,
        initial_wsr=history[0],final_wsr=history[-1],seconds=time.perf_counter()-tick,
        block_history=blocks)
    if verbose:print('ACCELERATED FP',diagnostics['stop_reason'],'FP steps',work,'WSR',value,flush=True)
    return resources,history,diagnostics
