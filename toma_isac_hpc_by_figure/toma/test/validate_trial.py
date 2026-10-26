"""Small validation only; does not produce Monte Carlo performance results."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import sys
sys.path.insert(1,str(Path(__file__).resolve().parent.parent))
import figure_engine as e

ROOT = Path(__file__).resolve().parent


def protocol():
    b = e.read_json(ROOT/'tradeoff_config.json')
    p = b['protocol']
    p['_models'] = b['models']
    return p


def short_task(task):
    p = protocol()
    cfg = replace(e.config_for_task(ROOT,p,task),max_inner_iter=2,max_outer_iter=1)
    g,r = e.initial_pair(cfg,p,task['scheme'])
    e.check_state(g,r,cfg,p,task['scheme'])
    result,h,n = e.run_solver_call(cfg,p,task,g,r)
    e.check_state(result.geometry,result.resources,cfg,p,task['scheme'],g)
    assert np.all(np.isfinite(h)) and np.min(np.diff(h)) >= -1e-8
    return dict(task=task,initial_wsr=h[0],final_wsr=h[-1],iterations=n,
                note='Truncated smoke check; not a converged performance result')


def main():
    p = protocol()
    tasks = e.build_tasks(ROOT,p,'tradeoff',10)
    assert len(tasks)==800 and all(e.n_starts(t)==1 for t in tasks)
    r_min,r_max = np.sqrt(1200)-10.,np.sqrt(1200)+10.
    points=[]
    for seed in range(8001,8011):
        reference=None
        for weight in (0.,.2,1.):
            for scheme in p['schemes']:
                cfg=e.config_for_task(ROOT,p,dict(figure='tradeoff',parameter_value=weight,scene_seed=seed,scheme=scheme))
                assert (cfg.m_uav,cfg.n_cable,cfg.n_tx,cfg.n_rx)==(8,4,16,16)
                q=np.vstack((cfg.q_dl,cfg.q_ul,cfg.q_s))
                assert np.all(q[:,2]==-20.)
                radii=np.linalg.norm(q[:,:2],axis=1)
                assert np.all((radii>=r_min-1e-12)&(radii<=r_max+1e-12))
                if reference is None: reference=q
                else: assert np.array_equal(reference,q)
        points.append(reference)
    assert not np.array_equal(points[0],points[1])
    sample=[dict(figure='tradeoff',parameter_value=w,scene_seed=8001,scheme=sc)
            for w in (0.,.2,1.) for sc in p['schemes']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        outcomes=list(pool.map(short_task,sample))
    distances=np.linalg.norm(np.vstack(points),axis=1)
    all_points=np.vstack(points)
    quadrants=(all_points[:,0]>=0).astype(int)+2*(all_points[:,1]>=0).astype(int)
    assert len(np.unique(quadrants))==4
    report=dict(passed=True,planned_tasks=len(tasks),scene_count=10,
                source_geometry=dict(ground_z_m=-20.,horizontal_radii_m=[r_min,r_max]),
                sampled_quadrant_counts=np.bincount(quadrants,minlength=4).tolist(),
                sampled_object_distance_range_m=[float(distances.min()),float(distances.max())],
                cross_weight_and_scheme_positions_identical=True,
                short_solver_checks=outcomes)
    e.json_write(ROOT/'validation_report.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
