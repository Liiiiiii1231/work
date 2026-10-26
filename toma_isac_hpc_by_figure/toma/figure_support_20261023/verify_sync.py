"""Configuration synchronization and small budget-only end-to-end checks."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
import json
import numpy as np
import run_figures as runner
import figure_engine as e


def main():
    root=Path(__file__).resolve().parent
    b=e.read_json(root/'simulation_config.json');p=b['protocol'];p['_models']=b['models'];e.STARTS.update(p['starts'])
    tasks={f:e.build_tasks(root,p,f,10) for f in ('convergence','p_dl_max','rho_si','system_load')}
    ref={};checked=0
    for figure,tt in tasks.items():
        for t in tt:
            cfg=e.config_for_task(root,p,t)
            assert cfg.n_tx==16 and cfg.n_rx==16 and cfg.m_uav==8 and cfg.n_cable==4
            q=np.vstack((cfg.q_dl,cfg.q_ul,cfg.q_s));radii=np.linalg.norm(q[:,:2],axis=1)
            assert np.all(q[:,2]==-20.) and np.all(radii>=np.sqrt(1200)-10-1e-10) and np.all(radii<=np.sqrt(1200)+10+1e-10)
            key=(cfg.k_dl,cfg.j_ul,cfg.l_target,t['scene_seed'])
            if key in ref:assert np.array_equal(ref[key],q)
            else:ref[key]=q
            checked+=1
    assert checked==630
    pp=deepcopy(p);pp['protocol'].update(max_inner_iter=2,max_outer_iter=1)
    pp['sweeps']['p_dl_dbm']=[30];pp['sweeps']['rho_si_db']=[-50];pp['sweeps']['loads']=[3,7]
    smoke={f:e.build_tasks(root,pp,f,2) for f in tasks}
    out=Path(r'D:\Vs codeProjects\research work1\other_figures_sync_validation_20261023')
    jobs=[(str(out),pp,t,s,1) for tt in smoke.values() for t in tt for s in range(e.n_starts(t))]
    with ProcessPoolExecutor(max_workers=2) as pool:rows=list(pool.map(e.run_job,jobs))
    assert len(rows)==38 and all(r['calls']==1 for r in rows)
    assert all(r['position_method']=='fp_aware_2' for r in rows if r['scheme'] in ('proposed','b6'))
    assert runner.collect(out,pp,smoke)
    report=dict(passed=True,verified_full_configuration_tasks=checked,
                full_plan={f:dict(cells=len(tt),starts=sum(e.n_starts(t) for t in tt)) for f,tt in tasks.items()},
                smoke_tasks=len(rows),smoke_only=True,
                note='Two scenes and tiny budgets exercise all four figure exports; not performance results')
    e.json_write(root/'validation_report.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
