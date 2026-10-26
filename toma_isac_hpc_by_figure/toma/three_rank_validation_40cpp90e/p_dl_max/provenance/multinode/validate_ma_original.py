"""Check release MA equals the unchanged original MA solver exactly."""
from pathlib import Path
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import sys,json
from copy import deepcopy
from dataclasses import replace
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
HERE=Path(__file__).resolve().parent;TOMA=HERE.parent
sys.path[:0]=[str(TOMA/'final_support_plainfp_o4_i5_s8005'),str(TOMA)]
import algorithms as a,inner_solver as inner,outer_solver as outer,figure_engine as e

def main():
    b=e.read_json(TOMA/'final_support_plainfp_o4_i5_s8005/final_config.json')
    p=deepcopy(b['protocol']);p['_models']=b['models'];q=p['protocol']
    original_positions=a._update_ma_positions_once;original_fp=inner.solve_inner_problem
    rows=[]
    for scene in [8003,8006]:
        task=dict(figure='p_dl_max',parameter_value=1.,scene_seed=scene,scheme='b6')
        cfg=replace(e.config_for_task(TOMA/'final_support_plainfp_o4_i5_s8005',p,task),max_inner_iter=3,max_outer_iter=2)
        g,r=e.pair(cfg,p,'b6',0)
        production,h,n=e.run_solver_call(cfg,p,task,g,r)
        assert a._update_ma_positions_once is original_positions
        assert a.solve_inner_problem is original_fp
        with ExitStack() as stack:
            for mod in [a,inner,outer]:stack.enter_context(patch.object(mod,'relative_change',e.abs_change))
            reference=a.run_ma_fp(cfg,rng=np.random.default_rng(q['resource_seed']),verbose=False,
                region_size=q['ma_region_size_m'],min_spacing=q['ma_min_spacing_m'],initial_step=q['ma_initial_step_m'],
                initial_tx_positions=g.tx_positions.copy(),initial_rx_positions=g.rx_positions.copy(),initial_resources=deepcopy(r))
        assert h==reference.wsr_history and n==reference.iterations
        for k in ['tx_positions','rx_positions']:assert np.array_equal(getattr(production.geometry,k),getattr(reference.geometry,k))
        for k in ['q_matrix','q_ul','b_ul','u_s']:assert np.array_equal(getattr(production.resources,k),getattr(reference.resources,k))
        assert not hasattr(production,'fp_aware_diagnostics')
        rows.append(dict(scene=scene,exact_history_geometry_resources_equal=True))
    assert 'ma_fp_aware' not in sys.modules
    assert 'fp_acceleration' not in sys.modules and 'accelerated_adapter' not in sys.modules
    report=dict(passed=True,ma_solver='original_algorithms.run_ma_fp',ma_position_solver='original_algorithms._update_ma_positions_once',
                candidate_fp_optimization=False,extrapolation=False,paired_cases=rows)
    e.json_write(HERE/'ma_original_validation.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
