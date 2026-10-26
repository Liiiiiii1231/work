"""Three Slurm node leaders, each with a local single-threaded process pool.

Scheduling is separate from the immutable screening/solver support directories.
An atomic mkdir on the shared filesystem claims each independent simulation.
All ranks participate; only rank zero prepares metadata and collects full results.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
            'BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import argparse,hashlib,json,shutil,socket,sys,time,traceback
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from itertools import chain

HERE=Path(__file__).resolve().parent
TOMA=HERE.parent
_CONTEXT=None


def normalized_sha(path):return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()


def bootstrap(figure):
    name='tradeoff_support_plainfp_o4_i5_s8005' if figure=='tradeoff' else 'final_support_plainfp_o4_i5_s8005'
    support=TOMA/name
    sys.path[:0]=[str(support),str(TOMA)]
    import figure_engine as e
    if figure=='tradeoff':import run_tradeoff_final as runner
    else:import run_final as runner
    return support,e,runner


def load_plan(args):
    support,e,runner=bootstrap(args.figure)
    selection=args.selection_dir.resolve()
    if args.figure=='screening':
        if args.local_test_fixture:
            if not args.local_test:raise ValueError('Test fixture requires local-test')
            p=e.read_json(args.local_test_fixture)['protocol']
        else:
            b=e.read_json(support/'final_config.json');p=b['protocol'];p['_models']=b['models']
        for name,digest in e.read_json(support/'shared_module_hashes.json').items():
            if normalized_sha(TOMA/name)!=digest:raise ValueError('Shared source differs: '+name)
        spec=p['screening']
        tasks=[dict(figure='p_dl_max',parameter_value=spec['p_dl_W'],scene_seed=spec['scene_seed'],scheme=sc,_start_id=start)
               for sc in spec['schemes'] for start in range(spec['candidates_per_scheme'])]
        return support,e,runner,p,tasks,None
    if args.local_test_fixture:
        if not args.local_test:raise ValueError('Test fixtures require --local-test')
        fixture=e.read_json(args.local_test_fixture)
        p=fixture['protocol'];file=selection/'frozen_initial_geometry.json'
        frozen=e.read_json(file)
        # Test data still requires its own exact source/config identity.
        if args.figure=='tradeoff':
            sys.path.insert(0,str(TOMA/'final_support_plainfp_o4_i5_s8005'))
            import run_final as screening
        else:screening=runner
        if frozen['identity']!=screening.source_identity(p):raise ValueError('Test freeze identity mismatch')
        p=deepcopy(p);p['_frozen_geometry']=frozen['geometry']
    elif args.figure=='tradeoff':
        p,file=runner.load_protocol(support/'tradeoff_final_config.json',selection)
        frozen=e.read_json(file)
    else:
        expected=e.read_json(support/'shared_module_hashes.json')
        for name,digest in expected.items():
            if normalized_sha(TOMA/name)!=digest:raise ValueError('Shared solver source differs: '+name)
        b=e.read_json(support/'final_config.json');p=b['protocol'];p['_models']=b['models']
        file=selection/'frozen_initial_geometry.json'
        frozen=e.read_json(file)
        if frozen['identity']!=runner.source_identity(p):raise ValueError('Frozen source/config identity mismatch')
        p=deepcopy(p);p['_frozen_geometry']=frozen['geometry']
    e.STARTS.update({s:1 for s in p['schemes']})
    if args.figure=='tradeoff':tasks=e.build_tasks(support,p,'tradeoff')
    else:tasks=runner.tasks_for(p,args.figure)
    if not tasks or len({e.task_key(t) for t in tasks})!=len(tasks):raise ValueError('Empty or duplicate task plan')
    if not args.local_test_fixture:
        if args.figure=='tradeoff':runner.validate_tasks(p,tasks)
        elif (p['protocol']['tol_inner'],p['protocol']['tol_outer'])!=(1e-5,1e-4):
            raise ValueError('Unexpected paper tolerances')
    return support,e,runner,p,tasks,file


def initialize_output(args,world):
    support,e,runner,p,tasks,file=load_plan(args)
    out=args.output.resolve()
    if out.exists() and any(f.name!='logs' for f in out.iterdir()):
        raise ValueError('Use a fresh output directory; multi-node reuse/continuation is disabled')
    out.mkdir(parents=True,exist_ok=True)
    scheduler_sources={f.name:normalized_sha(f) for f in HERE.glob('*.py')}
    identity=dict(figure=args.figure,protocol=p,frozen_sha256=e.sha(file) if file else None,
        solver_sources={f.name:normalized_sha(f) for f in support.glob('*.py')},
        shared_sources={n:normalized_sha(TOMA/n) for n in runner.SHARED},
        scheduler_sources=scheduler_sources,nodes=world,workers_per_node=args.workers_per_node,
        total_worker_slots=world*args.workers_per_node,local_test=args.local_test,
        scheduling='atomic shared-filesystem claims with work stealing; one call per task')
    e.json_write(out/'metadata/identity.json',identity)
    if file is not None:
        shutil.copy2(file,out/'frozen_initial_geometry.json')
        comparison=file.parent/'screening_comparison.csv'
        if comparison.exists():shutil.copy2(comparison,out/'screening_comparison.csv')
        if args.figure=='convergence':shutil.copytree(file.parent,out/'screening_details',dirs_exist_ok=True)
    for src in HERE.iterdir():
        if src.is_file() and src.suffix in ('.py','.md','.json'):
            dest=out/'provenance/multinode'/src.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    directories={'screening':'01_proposed_convergence','convergence':'01_proposed_convergence',
        'p_dl_max':'02_wsr_vs_p_dl_max','rho_si':'03_wsr_vs_rho_si',
        'system_load':'04_wsr_vs_system_load','tradeoff':'05_communication_sensing_tradeoff'}
    for src in (TOMA/directories[args.figure]).glob('*.sh'):
        dest=out/'provenance/entry_scripts'/src.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    for name in ['claims','receipts','nodes']:(out/'distributed'/name).mkdir(parents=True,exist_ok=True)
    plan=dict(figure=args.figure,protocol=p,tasks=tasks,world_size=world,
        workers_per_node=args.workers_per_node,output=str(out),
        identity_sha256=e.sha(out/'metadata/identity.json'))
    e.json_write(out/'distributed/plan.json',plan)
    # The ready marker is written last and tied to this Slurm allocation/run.
    e.json_write(out/'distributed/ready.json',dict(run_token=args.run_token,
        plan_sha256=e.sha(out/'distributed/plan.json')))
    print('MULTINODE PLAN',args.figure,'tasks',len(tasks),'nodes',world,
        'workers/node',args.workers_per_node,'total slots',world*args.workers_per_node,flush=True)


def await_plan(args,world):
    _,e,_=bootstrap(args.figure);out=args.output.resolve()
    end=time.monotonic()+args.startup_timeout
    marker=out/'distributed/ready.json'
    while not marker.exists():
        failed=out/'distributed/initialization_failed.json'
        if failed.exists():raise RuntimeError('Coordinator initialization failed: '+str(failed))
        if time.monotonic()>end:raise TimeoutError('No coordinator ready marker on shared filesystem')
        time.sleep(.2)
    ready=e.read_json(marker)
    if ready['run_token']!=args.run_token:raise ValueError('Output contains another run token')
    file=out/'distributed/plan.json'
    if e.sha(file)!=ready['plan_sha256']:raise ValueError('Distributed plan checksum mismatch')
    plan=e.read_json(file)
    if plan['world_size']!=world or plan['workers_per_node']!=args.workers_per_node:
        raise ValueError('Ranks disagree on node/worker counts')
    if e.sha(out/'metadata/identity.json')!=plan['identity_sha256']:raise ValueError('Run identity changed')
    identity=e.read_json(out/'metadata/identity.json')
    support,_,_=bootstrap(args.figure)
    if any(normalized_sha(support/name)!=digest for name,digest in identity['solver_sources'].items()):
        raise ValueError('Node sees different solver source files')
    if any(normalized_sha(TOMA/name)!=digest for name,digest in identity['shared_sources'].items()):
        raise ValueError('Node sees different shared source files')
    if any(normalized_sha(HERE/name)!=digest for name,digest in identity['scheduler_sources'].items()):
        raise ValueError('Node sees different scheduler source files')
    return plan


def init_worker(plan):
    global _CONTEXT
    _,e,_=bootstrap(plan['figure']);e.STARTS.update({s:1 for s in plan['protocol']['schemes']})
    _CONTEXT=plan


def execute_task(index):
    plan=_CONTEXT;_,e,_=bootstrap(plan['figure']);task=dict(plan['tasks'][index])
    start=task.pop('_start_id',0)
    row=e.run_job((plan['output'],plan['protocol'],task,start,1))
    return row,dict(host=socket.gethostname(),pid=os.getpid())


def run_node(args,rank,world,plan):
    _,e,_=bootstrap(args.figure);out=Path(plan['output']);count=len(plan['tasks'])
    order=chain(range(rank,count,world),(i for i in range(count) if i%world!=rank))
    assigned=0;finished=0;errors=[];tick=time.monotonic()
    host=socket.gethostname()
    affinity=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None
    if not args.local_test and affinity is not None and affinity<args.workers_per_node:
        raise RuntimeError('Slurm CPU affinity exposes fewer CPUs than requested worker processes')
    e.json_write(out/'distributed/nodes'/f'rank_{rank:03d}.started.json',
        dict(rank=rank,host=host,workers=args.workers_per_node,affinity_cpus=affinity))
    barrier_deadline=time.monotonic()+args.startup_timeout
    while not all((out/'distributed/nodes'/f'rank_{r:03d}.started.json').exists() for r in range(world)):
        if time.monotonic()>barrier_deadline:raise TimeoutError('Not all node leaders reached the startup barrier')
        time.sleep(.2)
    print('NODE START rank',rank,'host',host,'workers',args.workers_per_node,'affinity CPUs',affinity,flush=True)

    def claim():
        nonlocal assigned
        for index in order:
            folder=out/'distributed/claims'/f'{index:06d}'
            try:folder.mkdir()
            except FileExistsError:continue
            e.json_write(folder/'owner.json',dict(rank=rank,host=host,index=index,run_token=args.run_token))
            assigned+=1
            return index
        return None

    with ProcessPoolExecutor(max_workers=min(args.workers_per_node,count),initializer=init_worker,initargs=(plan,)) as pool:
        pending={}
        # Give every host its own first batch before allowing work stealing.
        # In particular 40 screening / 90 convergence tasks must use all hosts,
        # even though one host alone has 56 worker slots.
        initial_count=min(args.workers_per_node,len(range(rank,count,world)))
        for _ in range(initial_count):
            index=claim()
            if index is not None:pending[pool.submit(execute_task,index)]=index
        e.json_write(out/'distributed/nodes'/f'rank_{rank:03d}.prefilled.json',dict(rank=rank))
        while not all((out/'distributed/nodes'/f'rank_{r:03d}.prefilled.json').exists() for r in range(world)):
            if time.monotonic()>barrier_deadline:raise TimeoutError('Node leaders did not fill their first batches')
            time.sleep(.2)
        while True:
            while len(pending)<args.workers_per_node:
                index=claim()
                if index is None:break
                pending[pool.submit(execute_task,index)]=index
            if not pending:break
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                index=pending.pop(future);task=plan['tasks'][index]
                try:
                    row,worker=future.result();finished+=1
                    receipt=dict(index=index,rank=rank,task=task,ok=True,worker=worker,
                        wsr=row['weighted_sum_rate'],converged=row['converged'])
                    print('TASK FINISHED rank',rank,'index',index,e.task_key(task),'converged',row['converged'],flush=True)
                except Exception as exc:
                    receipt=dict(index=index,rank=rank,task=task,ok=False,error=repr(exc))
                    errors.append(receipt);print('FAILED',receipt,flush=True)
                e.json_write(out/'distributed/receipts'/f'{index:06d}.json',receipt)
    summary=dict(rank=rank,host=host,workers=args.workers_per_node,affinity_cpus=affinity,
        assigned_tasks=assigned,finished_tasks=finished,failed_tasks=len(errors),
        seconds=time.monotonic()-tick,errors=errors)
    e.json_write(out/'distributed/nodes'/f'rank_{rank:03d}.json',summary)
    print('NODE DONE',summary,flush=True)
    return summary


def collect_all(args,world,plan):
    support,e,runner=bootstrap(args.figure);out=Path(plan['output'])
    deadline=time.monotonic()+args.finish_timeout
    while not all((out/'distributed/nodes'/f'rank_{r:03d}.json').exists() for r in range(world)):
        if time.monotonic()>deadline:raise TimeoutError('Other node leaders did not finish')
        time.sleep(.5)
    nodes=[e.read_json(out/'distributed/nodes'/f'rank_{r:03d}.json') for r in range(world)]
    if not args.local_test and len({n['host'] for n in nodes})!=world:
        raise RuntimeError('Expected one leader on each of three distinct compute hosts')
    receipts=[e.read_json(f) for f in sorted((out/'distributed/receipts').glob('*.json'))]
    errors=[r for r in receipts if not r['ok']]
    e.json_write(out/'logs/task_errors.json',errors)
    if len(receipts)!=len(plan['tasks']) or {r['index'] for r in receipts}!=set(range(len(plan['tasks']))):
        raise RuntimeError('Missing or duplicate distributed task receipts; no partial average is allowed')
    for receipt in receipts:
        if receipt['task']!=plan['tasks'][receipt['index']]:raise RuntimeError('Receipt task identity mismatch')
        owner=e.read_json(out/'distributed/claims'/f"{receipt['index']:06d}"/'owner.json')
        if owner['rank']!=receipt['rank'] or owner['run_token']!=args.run_token:
            raise RuntimeError('Receipt owner identity mismatch')
    if sum(n['assigned_tasks'] for n in nodes)!=len(plan['tasks']) or errors:
        raise RuntimeError('Distributed tasks failed; inspect logs/task_errors.json')
    e.json_write(out/'metadata/multinode_summary.json',dict(nodes=nodes,world_size=world,
        total_worker_slots=world*args.workers_per_node,finished_tasks=len(receipts),
        unique_task_claims_verified=True,local_test=args.local_test))
    if args.figure=='screening':
        p=plan['protocol'];spec=p['screening'];winners={};geometries={};rows=[]
        for sc in spec['schemes']:
            task=dict(figure='p_dl_max',parameter_value=spec['p_dl_W'],scene_seed=spec['scene_seed'],scheme=sc)
            group=[e.read_verified(out,p,task,s) for s in range(spec['candidates_per_scheme'])]
            if any(r is None for r in group):raise RuntimeError('Missing screening candidate')
            winner=max(group,key=lambda r:(r['weighted_sum_rate'],-r['start_id']))
            folder=out/f"candidate_{winner['start_id']}"/'tasks'/e.task_key(task)
            initial,_,_=e.load_state(folder/'initial.npz')
            geometries[sc]={k:getattr(initial,k).tolist() for k in ['endpoints','tx_positions','rx_positions']}
            winners[sc]=dict(start_id=winner['start_id'],reference_wsr=winner['weighted_sum_rate'],
                reference_converged=winner['converged'],screening_converged=sum(r['converged'] for r in group),
                initial_sha256=e.sha(folder/'initial.npz'))
            rows.extend(group)
        e.csv_write(out/'screening_comparison.csv',rows,list(rows[0]))
        frozen=dict(identity=runner.source_identity(p),reference=spec,winners=winners,geometry=geometries,
            freeze_definition='Winning INITIAL geometry only. All resource states rebuilt independently for every final task.')
        e.json_write(out/'frozen_initial_geometry.json',frozen)
        e.json_write(out/'completion.json',dict(complete=True,figure='screening',expected_tasks=len(rows),finished_tasks=len(rows)))
        print('FROZEN GEOMETRY READY:',out/'frozen_initial_geometry.json',flush=True)
    elif args.figure=='tradeoff':runner.collect(out,plan['protocol'],plan['tasks'],args.local_test)
    else:runner.collect(out,plan['protocol'],args.figure,plan['tasks'])
    status=e.read_json(out/'completion.json')
    status.update(nodes=world,workers_per_node=args.workers_per_node,
        total_worker_slots=world*args.workers_per_node,local_multirank_test=args.local_test)
    e.json_write(out/'completion.json',status)
    print('ALL NODES COLLECTED',args.figure,'finished tasks',len(receipts),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--figure',required=True,choices=['screening','convergence','p_dl_max','rho_si','system_load','tradeoff'])
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--selection-dir',type=Path,default=TOMA/'results/final_initial_geometry_plainfp_o4_i5_mafp_s8005')
    ap.add_argument('--workers-per-node',type=int,default=56)
    ap.add_argument('--check-only',action='store_true')
    ap.add_argument('--startup-timeout',type=float,default=900.)
    ap.add_argument('--finish-timeout',type=float,default=172800.)
    ap.add_argument('--local-test',action='store_true',help='Simulate ranks locally, never a final paper run')
    ap.add_argument('--rank',type=int);ap.add_argument('--world-size',type=int,default=3)
    ap.add_argument('--run-token',default=os.environ.get('SLURM_JOB_ID'))
    ap.add_argument('--local-test-fixture',type=Path)
    args=ap.parse_args()
    if args.workers_per_node<1:ap.error('Positive worker count is required')
    if args.check_only:
        _,e,_,p,tasks,file=load_plan(args)
        print('CHECK PASSED',args.figure,'tasks',len(tasks),'nodes 3','workers/node',args.workers_per_node,
            'frozen SHA256',e.sha(file) if file else 'new screening',flush=True);return
    if args.local_test:
        rank=args.rank;world=args.world_size
        if rank is None or not args.run_token:ap.error('Local tests require --rank and --run-token')
    else:
        required=['SLURM_PROCID','SLURM_NTASKS','SLURM_JOB_NUM_NODES','SLURM_CPUS_PER_TASK']
        if any(k not in os.environ for k in required):ap.error('Launch via srun inside the three-node allocation')
        rank=int(os.environ['SLURM_PROCID']);world=int(os.environ['SLURM_NTASKS'])
        if world!=3 or int(os.environ['SLURM_JOB_NUM_NODES'])!=3:
            ap.error('Expected three Slurm tasks on three nodes')
        if args.workers_per_node!=56 or int(os.environ['SLURM_CPUS_PER_TASK'])!=56:
            ap.error('Expected 56 CPUs and 56 workers on each node')
    if not 0<=rank<world:ap.error('Invalid rank')
    _,e,_=bootstrap(args.figure)
    if rank==0:
        try:initialize_output(args,world)
        except Exception:
            # Do not write a failure marker into an already-used output directory.
            print('COORDINATOR INITIALIZATION FAILED',traceback.format_exc(),flush=True)
            raise
    plan=await_plan(args,world)
    run_node(args,rank,world,plan)
    if rank==0:collect_all(args,world,plan)

if __name__=='__main__':main()
