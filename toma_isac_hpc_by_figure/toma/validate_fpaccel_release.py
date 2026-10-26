"""Local integration checks and one full known-state regression; no Slurm jobs."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,zipfile,hashlib,io,tempfile
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent
SUPPORT=ROOT/'final_support_fpaccel_o4_i5_s8005'
sys.path[:0]=[str(SUPPORT),str(ROOT)]
import numpy as np
import run_final as runner,figure_engine as e,fp_acceleration as fp,algorithms as algo,outer_solver
from system_model import SystemConfig,GeometryState,ResourceState
from channel_model import build_channels


def load_known():
    with zipfile.ZipFile(Path.home()/'Downloads/final_33774_bundle.zip') as z:
        folder='final_33774/candidate_0/tasks/convergence/1/scene_8003/proposed/'
        d=json.loads(z.read(folder+'record.json'));raw=d['effective_config'].copy()
        for k,v in list(raw.items()):
            if isinstance(v,list):
                a=np.asarray(v);raw[k]=a[:,0]+1j*a[:,1] if k in ['xi_s','fading_dl','fading_ul','fading_s'] else a
        cfg=SystemConfig(**raw);assert e.config_digest(cfg)==d['cfg_sha256']
        data=z.read(folder+'initial.npz');assert hashlib.sha256(data).hexdigest()==d['initial_sha256']
        with np.load(io.BytesIO(data)) as a:
            g=GeometryState(**{k:a[k].copy() for k in ['endpoints','tx_positions','rx_positions']})
            r=ResourceState(**{k:a[k].copy() for k in ['q_matrix','q_ul','b_ul','u_s']})
    return cfg,g,r


def main():
    b=e.read_json(SUPPORT/'final_config.json');p=deepcopy(b['protocol']);p['_models']=b['models']
    assert p['protocol']['tol_inner']==1e-5 and p['protocol']['tol_outer']==1e-4
    assert p['protocol']['resource_solver']==fp.DEFAULT_POLICY
    assert p['protocol']['max_inner_iter']==1200 and p['protocol']['max_outer_iter']==1500
    assert {f:len(runner.tasks_for(p,f)) for f in runner.FIGURES}==dict(convergence=90,p_dl_max=2000,rho_si=2000,system_load=2000)
    trade=e.read_json(ROOT/'tradeoff_support_fpaccel_o4_i5_s8005/tradeoff_final_config.json')
    assert len(trade['protocol']['tradeoff']['weights'])==20
    assert trade['models']==b['models'] and trade['protocol']['protocol']==p['protocol']
    reference=e.read_json(ROOT/'tradeoff_support_fpaccel_o4_i5_s8005/reference_screening_identity.json')
    assert reference==runner.source_identity(p)
    old=e.read_json(ROOT/'final_support_i1200_o1500_s8005/shared_module_hashes.json')
    assert all(hashlib.sha256((ROOT/k).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==v for k,v in old.items())
    cfg,g,r=load_known();ch=build_channels(g,cfg)
    snapshot={k:getattr(r,k).copy() for k in ['q_matrix','q_ul','b_ul','u_s']}
    for budget in [1,2,3,7,12]:
        rr,h,d=fp.solve_accelerated(ch,r,replace(cfg,max_inner_iter=budget,tol_inner=1e-5))
        assert d['actual_fp_iterations']<=budget
        assert sum(x['actual_fp_iterations'] for x in d['block_history'])==d['actual_fp_iterations']
        assert min(np.diff(h))>=-1e-8
        assert not d['converged'] or max(abs(np.diff(h[-3:])))<=1e-5
        e.check_state(g,rr,cfg,p,'proposed')
    with patch.object(fp,'project',lambda vector,template,cc:deepcopy(r)):
        rr,h,d=fp.solve_accelerated(ch,r,replace(cfg,max_inner_iter=7,tol_inner=1e-5))
        assert any(not x['extrapolation_accepted'] for x in d['block_history'])
        assert d['actual_fp_iterations']<=7 and min(np.diff(h))>=-1e-8
    assert all(np.array_equal(getattr(r,k),v) for k,v in snapshot.items())
    # Small end-to-end screening/export checks; never use these fixtures as final data.
    pp=deepcopy(p);pp['n_mc']=1;pp['screening']['candidates_per_scheme']=2
    pp['protocol'].update(max_inner_iter=6,max_outer_iter=2)
    pp['convergence']['scene_seeds']=[8003]
    pp['sweeps']['p_dl_dbm']=[30];pp['sweeps']['rho_si_db']=[-50];pp['sweeps']['loads']=[3,7]
    pp['tradeoff']['weights']=[0.,.2,1.]
    work=Path(tempfile.mkdtemp(prefix='fpaccel_release_',dir=ROOT))
    frozen=runner.prepare_selection(pp,work/'selection',2)
    pp['_frozen_geometry']=frozen['geometry']
    assert frozen['freeze_definition'].startswith('Winning INITIAL geometry')
    tested=0
    for figure in ['convergence','p_dl_max','rho_si','system_load','tradeoff']:
        tasks=runner.tasks_for(pp,figure) if figure!='tradeoff' else e.build_tasks(SUPPORT,pp,'tradeoff')
        target=work/figure
        runner.run_batch(target,pp,[(str(target),pp,t,0,1) for t in tasks],2)
        rows=[e.read_verified(target,pp,t,0) for t in tasks]
        assert all(r is not None for r in rows);tested+=len(rows)
        if figure=='tradeoff':e.plot_tradeoff(target,pp,tasks,rows)
        else:runner.collect(target,pp,figure,tasks)
        for record in target.rglob('record.json'):
            dd=e.read_json(record);hh=e.load_state(record.parent/dd['state_file'])[2]
            diag=e.read_json(record.parent/'accelerated_inner_diagnostics.json')
            assert dd['main_actual_fp_iterations']==sum(x['actual_fp_iterations'] for x in diag)
            assert all(x['actual_fp_iterations']<=6 for x in diag)
            if dd['row']['scheme'] in ['b2','b3']:
                assert dd['row']['iterations']==dd['main_actual_fp_iterations']
            assert min(np.diff(hh))>=-1e-8
    # Do not silently accept old identities at the new freeze path.
    try:runner.prepare_selection(p,work/'selection',1)
    except ValueError:pass
    else:raise AssertionError('Mismatched freeze identity was accepted')
    from finalize_run import export_run
    export_run(work/'tradeoff',SUPPORT,0)
    # Full regression from exactly the previously tested frozen initial state.
    original_main=algo.solve_inner_problem;original_candidates=outer_solver.solve_inner_problem
    cc=replace(cfg,max_outer_iter=120,tol_inner=1e-5,tol_outer=1e-4,plateau_window=0,tol_window=0.)
    task=dict(figure='convergence',parameter_value=1.,scene_seed=8003,scheme='proposed')
    result,history,count=e.run_solver_call(cc,p,task,g,r)
    expected=json.loads((ROOT.parent.parent/'convergence_experiments_20261024/s8003_p1_n120_initial_accelerated_joint.json').read_text(encoding='utf-8'))
    assert result.converged and count==expected['iterations']==81
    assert np.array_equal(np.asarray(history),np.asarray(expected['history']))
    assert algo.solve_inner_problem is original_main and outer_solver.solve_inner_problem is original_candidates
    e.check_state(result.geometry,result.resources,cc,p,'proposed')
    report=dict(passed=True,main_policy=fp.DEFAULT_POLICY,tol_inner=1e-5,tol_outer=1e-4,
        smoke_only=True,smoke_screening_calls=4,smoke_final_calls=tested,
        all_five_plots_exported=True,tradeoff_bundle_verified=True,
        regression_scene=8003,regression_outer_iterations=count,
        regression_history_bitwise_equal=True,actual_fp_budget_verified=True,
        fixed_baseline_iteration_units_verified=True,old_frozen_identity_rejected=True,
        shared_modules_unchanged=True,fixture_output=str(work),
        note='Local integration checks and one known-state regression; not final MC results.')
    for support in [SUPPORT,ROOT/'tradeoff_support_fpaccel_o4_i5_s8005']:
        e.json_write(support/'validation_report.json',report)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
