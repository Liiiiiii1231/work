"""Thirty-iteration same-state comparison; production files remain untouched."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
from pathlib import Path
import zipfile,json,time
from dataclasses import replace
from copy import deepcopy
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
import check_larger_step as test
import algorithms as algo,inner_solver as inner,outer_solver as outer

def main():
    with zipfile.ZipFile(test.ZIP) as z:cfg,g,r=test.load(z,'1',8003)
    rows=[];base=None
    with ExitStack() as stack:
        for mod in (algo,inner,outer):stack.enter_context(patch.object(mod,'relative_change',test.e.abs_change))
        for alpha in [.1,.8,1.]:
            tick=time.perf_counter()
            result=algo.run_joint_algorithm(replace(cfg,initial_outer_step=alpha),
                rng=np.random.default_rng(cfg.random_seed),verbose=False,
                max_outer_iterations=30,position_update_mode='fp_aware',fp_aware_candidate_iters=2,
                initial_endpoints=g.endpoints.copy(),initial_resources=deepcopy(r))
            history=list(map(float,result.joint_wsr_history))
            if base is None:base=history
            test.e.check_state(result.geometry,result.resources,cfg,{},'proposed')
            assert min(np.diff(history))>=-1e-8
            row=dict(initial_alpha=alpha,iterations=result.outer_iterations,converged=result.converged,
                initial_score=history[0],final_score=history[-1],gain=history[-1]-history[0],
                seconds=time.perf_counter()-tick,history=history,
                max_history_difference_from_alpha01=(float(np.max(abs(np.array(history)-base))) if len(history)==len(base) else None))
            rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='history'}),flush=True)
            test.e.json_write(test.OUT/'larger_step_30_iteration_comparison.json',dict(
                note='Same saved 30dBm scene8003 final state; each new 30-iteration run starts with empty RCG memory. Not the original trajectory continued, and not a full convergence-count test.',rows=rows))

if __name__=='__main__':main()
