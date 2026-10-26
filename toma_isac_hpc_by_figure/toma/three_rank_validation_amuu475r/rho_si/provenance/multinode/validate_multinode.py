"""Three local ranks simulate the shared-queue protocol; no Slurm submission."""
from pathlib import Path
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import sys,json,hashlib,subprocess,time,tempfile
from copy import deepcopy
HERE=Path(__file__).resolve().parent
TOMA=HERE.parent
sys.path[:0]=[str(TOMA/'final_support_plainfp_o4_i5_s8005'),str(TOMA)]
import run_final as runner,figure_engine as e


def main():
    reference=e.read_json(TOMA/'tradeoff_support_plainfp_o4_i5_s8005/reference_screening_identity.json')
    b=e.read_json(TOMA/'final_support_plainfp_o4_i5_s8005/final_config.json')
    original=deepcopy(b['protocol']);original['_models']=b['models']
    assert reference==runner.source_identity(original)
    source_before=deepcopy(reference)
    work=Path(tempfile.mkdtemp(prefix='three_rank_validation_',dir=TOMA))
    p=deepcopy(original);p['n_mc']=2;p['screening']['candidates_per_scheme']=2
    p['protocol'].update(max_inner_iter=3,max_outer_iter=1)
    p['sweeps']['p_dl_dbm']=[30];p['sweeps']['rho_si_db']=[-50];p['sweeps']['loads']=[3,7]
    p['tradeoff']['weights']=[0.,.2,1.]
    frozen=runner.prepare_selection(p,work/'selection',2)
    e.json_write(work/'fixture.json',dict(protocol=p))
    rows=[]
    for figure in ['p_dl_max','rho_si','system_load','tradeoff']:
        out=work/figure;token='test_'+figure
        processes=[];streams=[]
        for rank in range(3):
            stream=(work/f'{figure}_rank{rank}.log').open('w',encoding='utf-8');streams.append(stream)
            command=[sys.executable,str(HERE/'run_multinode.py'),'--figure',figure,'--output',str(out),
                '--selection-dir',str(work/'selection'),'--workers-per-node','2','--local-test',
                '--rank',str(rank),'--world-size','3','--run-token',token,
                '--local-test-fixture',str(work/'fixture.json'),'--startup-timeout','60','--finish-timeout','180']
            processes.append(subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT))
        deadline=time.monotonic()+240
        while any(proc.poll() is None for proc in processes):
            if any(proc.poll() not in (None,0) for proc in processes):
                for proc in processes:
                    if proc.poll() is None:proc.terminate()
                raise RuntimeError('Local rank failed; logs in '+str(work))
            if time.monotonic()>deadline:
                for proc in processes:
                    if proc.poll() is None:proc.terminate()
                raise TimeoutError('Local ranks timed out; logs in '+str(work))
            time.sleep(.2)
        for stream in streams:stream.close()
        assert all(proc.returncode==0 for proc in processes)
        summary=e.read_json(out/'metadata/multinode_summary.json')
        complete=e.read_json(out/'completion.json')
        assert summary['world_size']==3 and summary['total_worker_slots']==6
        assert all(node['assigned_tasks']>0 for node in summary['nodes'])
        expected={'p_dl_max':8,'rho_si':8,'system_load':16,'tradeoff':24}[figure]
        assert complete['complete'] and complete['expected_tasks']==expected
        receipts=list((out/'distributed/receipts').glob('*.json'))
        assert len(receipts)==expected and len(list((out/'distributed/claims').iterdir()))==expected
        assert len(list((out/'figures').glob('*.png')))==1
        # Verify identity matching and complete per-task records with original readers.
        pp=deepcopy(p);pp['_frozen_geometry']=frozen['geometry']
        tasks=e.build_tasks(runner.ROOT,pp,figure)
        distributed=[e.read_verified(out,pp,t,0) for t in tasks]
        assert all(r is not None for r in distributed)
        if figure=='p_dl_max':
            # A serial run must produce exactly the same per-task physical metrics.
            serial=work/'serial_control'
            for task,parallel in zip(tasks,distributed):
                one=e.run_job((str(serial),pp,task,0,1))
                for key in ['weighted_sum_rate','sum_rate_comm','sum_rate_sensing','iterations','converged']:
                    assert one[key]==parallel[key],(task,key,one[key],parallel[key])
        row=dict(figure=figure,tasks=expected,assigned_by_rank=[n['assigned_tasks'] for n in summary['nodes']],
            completed_by_rank=[n['finished_tasks'] for n in summary['nodes']],unique_claims=True,full_export=True)
        rows.append(row);print(json.dumps(row),flush=True)
    assert runner.source_identity(original)==source_before
    # A multi-node patch must not alter either original screening support tree.
    for rel,digest in reference['sources'].items():
        file=(TOMA/'final_support_plainfp_o4_i5_s8005'/rel).resolve()
        assert hashlib.sha256(file.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==digest
    report=dict(passed=True,local_multirank_simulation=True,actual_slurm_test_performed=False,
        simulated_ranks=3,workers_per_rank=2,cases=rows,serial_metric_equality=True,
        original_screening_source_identity_unchanged=True,output=str(work),
        note='Local processes simulate three leaders on a shared filesystem. Real three-host Slurm launch remains a platform check.')
    e.json_write(HERE/'validation_report.json',report)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
