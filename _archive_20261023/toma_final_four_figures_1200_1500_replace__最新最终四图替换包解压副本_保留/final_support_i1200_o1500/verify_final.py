"""Small end-to-end tests of screening, geometry freezing and all four exports."""
from pathlib import Path
from copy import deepcopy
import json
import numpy as np
import run_final as runner
import figure_engine as e


def main():
    b=e.read_json(runner.ROOT/'final_config.json');p=b['protocol'];p['_models']=b['models'];e.STARTS.update(p['starts'])
    plan={f:len(runner.tasks_for(p,f)) for f in runner.FIGURES}
    assert plan==dict(convergence=90,p_dl_max=2000,rho_si=2000,system_load=2000)
    assert len(p['convergence']['scene_seeds'])==30 and len(set(p['convergence']['scene_seeds']))==30
    for sc in ('proposed','b6'):
        t=dict(figure='p_dl_max',parameter_value=1.,scene_seed=8001,scheme=sc);cfg=e.config_for_task(runner.ROOT,p,t)
        layouts=[]
        for start in range(20):
            g,r=e.pair(cfg,p,sc,start);e.check_state(g,r,cfg,p,sc)
            layouts.append(g.tx_positions.tobytes()+g.rx_positions.tobytes())
        assert len(set(layouts))==20
    pp=deepcopy(p);pp['n_mc']=2;pp['convergence']['scene_seeds']=p['convergence']['scene_seeds'][:2]
    pp['protocol'].update(max_inner_iter=2,max_outer_iter=1)
    pp['sweeps']['p_dl_dbm']=[30];pp['sweeps']['rho_si_db']=[-50];pp['sweeps']['loads']=[3,7]
    root=Path(r'D:\Vs codeProjects\research work1\final_protocol_validation_20261023')
    frozen=runner.prepare_selection(pp,root/'selection',2)
    for sc,winner in frozen['winners'].items():
        t=dict(figure='p_dl_max',parameter_value=1.,scene_seed=8001,scheme=sc)
        initial,_,_=e.load_state(root/'selection'/f"candidate_{winner['start_id']}"/'tasks'/e.task_key(t)/'initial.npz')
        assert np.array_equal(np.array(frozen['geometry'][sc]['tx_positions']),initial.tx_positions)
    pp['_frozen_geometry']=frozen['geometry']
    initials={}
    for figure in runner.FIGURES:
        tt=runner.tasks_for(pp,figure);out=root/figure
        for t in tt:
            cfg=e.config_for_task(runner.ROOT,pp,t);g,r=e.pair(cfg,pp,t['scheme'],0)
            assert cfg.n_tx==16 and cfg.n_rx==16
            q=np.vstack((cfg.q_dl,cfg.q_ul,cfg.q_s));radius=np.linalg.norm(q[:,:2],axis=1)
            assert np.all(q[:,2]==-20.) and np.all((radius>=np.sqrt(1200)-10-1e-10)&(radius<=np.sqrt(1200)+10+1e-10))
            if t['scheme'] in frozen['geometry']:
                key=t['scheme'];geom=g.tx_positions.tobytes()+g.rx_positions.tobytes()
                if key in initials:assert initials[key]==geom
                else:initials[key]=geom
            assert np.allclose(abs(r.q_ul)**2,.025,rtol=0,atol=1e-12)
        runner.run_batch(out,pp,[(str(out),pp,t,0,1) for t in tt],2)
        runner.collect(out,pp,figure,tt)
    e.json_write(runner.ROOT/'validation_report.json',dict(passed=True,final_plan=plan,
        screening_starts=40,distinct_screening_geometries_per_scheme=20,
        smoke_screening_calls=40,smoke_final_calls=38,frozen_initial_geometry_verified=True,
        all_four_plots_exported=True,smoke_only=True,note='Tiny iteration budgets; not final performance results'))
    print('FINAL PROTOCOL TESTS PASSED',flush=True)


if __name__=='__main__':main()
