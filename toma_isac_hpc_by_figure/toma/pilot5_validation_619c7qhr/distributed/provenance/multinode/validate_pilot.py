"""Local five-rank smoke test; never runs the production 3000/5000 budgets."""
from pathlib import Path
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import sys,json,subprocess,tempfile,time
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent
import run_pilot as pilot

def main():
    work=Path(tempfile.mkdtemp(prefix='pilot5_validation_',dir=HERE.parent))
    fixture=work/'fixture.json'
    indices=[0,1,2,30,31,32,60,61,62,120,121,122,147,148,149]
    pilot.e.json_write(fixture,dict(indices=indices,budget=dict(max_inner_iter=3,max_outer_iter=2)))
    out=work/'distributed';streams=[];processes=[]
    for rank in range(5):
        log=(work/f'rank{rank}.log').open('w',encoding='utf-8');streams.append(log)
        command=[sys.executable,str(HERE/'run_pilot.py'),'--output',str(out),'--local-test','--rank',str(rank),
            '--world-size','5','--workers-per-node','2','--run-token','pilot5_smoke',
            '--local-test-fixture',str(fixture),'--startup-timeout','120','--finish-timeout','180']
        processes.append(subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT))
    deadline=time.monotonic()+240
    while any(p.poll() is None for p in processes):
        if any(p.poll() not in [None,0] for p in processes) or time.monotonic()>deadline:
            for p in processes:
                if p.poll() is None:p.terminate()
            raise RuntimeError('Local rank failed or timed out: '+str(work))
        time.sleep(.2)
    for log in streams:log.close()
    assert all(p.returncode==0 for p in processes)
    completion=pilot.e.read_json(out/'completion.json')
    summary=pilot.e.read_json(out/'metadata/multinode_summary.json')
    assert completion['complete'] and completion['expected_tasks']==15
    assert summary['world_size']==5 and all(n['assigned_tasks']>0 for n in summary['nodes'])
    assert len(list((out/'distributed/receipts').glob('*.json')))==15
    assert len(list((out/'distributed/claims').iterdir()))==15
    assert (out/'tables/pilot_per_task.csv').exists() and (out/'tables/pilot_summary.csv').exists()
    assert (out/'figures/pilot_budget_comparison.png').exists()
    assert out.with_name(out.name+'_bundle.zip').exists()
    args=SimpleNamespace(local_test_fixture=fixture,local_test=True)
    _,_,_,p,tasks,_=pilot.load_plan(args)
    for task in [tasks[0],tasks[1],tasks[2]]:
        serial=pilot.run_job((str(work/'serial'),p,task,0,1))
        parallel=pilot.e.read_json(out/'tasks'/f"{task['_pilot_id']:03d}"/'record.json')['row']
        for k in ['weighted_sum_rate','sum_rate_comm','sum_rate_sensing','iterations','converged','total_fp_updates']:
            assert serial[k]==parallel[k],(task,k,serial[k],parallel[k])
    # Used outputs cannot be adopted or silently continued.
    old_identity=pilot.sha(out/'metadata/identity.json')
    command=[sys.executable,str(HERE/'run_pilot.py'),'--output',str(out),'--local-test','--rank','0',
        '--world-size','5','--workers-per-node','2','--run-token','pilot5_smoke','--local-test-fixture',str(fixture)]
    rejected=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=30)
    assert rejected.returncode!=0 and 'fresh output directory' in rejected.stdout
    assert pilot.sha(out/'metadata/identity.json')==old_identity
    report=dict(passed=True,local_five_rank_simulation=True,workers_per_rank=2,test_tasks=15,
        production_task_count=150,all_five_ranks_used=True,
        assigned_by_rank=[n['assigned_tasks'] for n in summary['nodes']],
        serial_parallel_metrics_exactly_equal=True,all_three_baselines_checked=True,
        unique_claims_verified=True,fresh_output_guard=True,
        actual_hpc_test_performed=False,production_budgets_executed=False,output=str(work))
    pilot.e.json_write(HERE/'validation_report.json',report)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
