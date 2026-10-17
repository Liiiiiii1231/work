"""Paper figures: 25 mW/user, absolute 1e-4, regular geometry multi-start.

Default is the 10-scene DL-power sweep. The retained legacy functions are not
the CLI entry point. Frozen tradeoff results alone keep their 0.25 mW protocol.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
from pathlib import Path
import sys,argparse,json,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments'))
import paper_figures_unified as engine


def execute_figure(root,out,p,figure,tasks,workers,max_calls,plot_only):
    if figure=='tradeoff':engine.plot_tradeoff(out,p);return dict(complete=True,converged=64,expected=64,reused=64,errors=[])
    errors=[]
    if plot_only:
        rows=engine.collect_rows(out,tasks)
        if len(rows)!=len(tasks) or not all(r['converged'] for r in rows):raise RuntimeError('Plot-only requires ALL expected tasks to have converged; no partial averaging.')
        # run_task returns verified cached results here; guard above prevents new optimization.
        for task in tasks:engine.run_task(str(root),str(out),p,task,max_calls)
    elif workers==1:
        for task in tasks:
            try:engine.run_task(str(root),str(out),p,task,max_calls)
            except Exception as exc:
                errors.append(dict(task=engine.task_key(task),error=str(exc),traceback=traceback.format_exc()))
                print('TASK FAILED:',engine.task_key(task),str(exc),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(engine.run_task,str(root),str(out),p,task,max_calls):task for task in tasks}
            for i,future in enumerate(as_completed(futures),1):
                task=futures[future]
                try:future.result()
                except Exception as exc:
                    errors.append(dict(task=engine.task_key(task),error=str(exc),traceback=traceback.format_exc()))
                    print('TASK FAILED:',engine.task_key(task),str(exc),flush=True)
                print(f'{figure}: {i}/{len(tasks)} tasks returned',flush=True)
    rows=engine.collect_rows(out,tasks);engine.csv_write(out/'tables'/f'{figure}_per_scene.csv',rows,engine.FIELDS)
    done=sum(bool(r['converged']) for r in rows);complete=done==len(tasks) and not errors
    if errors:engine.json_write(out/'logs'/f'{figure}_errors.json',errors)
    if complete:
        if figure=='convergence':engine.plot_convergence(out,p,tasks)
        else:engine.plot_sweep(out,p,figure,tasks,rows)
    return dict(complete=complete,converged=done,expected=len(tasks),errors=errors)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=Path('configs/paper_figures_unified.yaml'))
    ap.add_argument('--output',type=Path,default=Path('results/paper_figures_regular_abs_v2'))
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--max-calls',type=int,default=20)
    ap.add_argument('--n-mc',type=int,help='Optional explicit override; otherwise preserve original Stage8 n_mc.')
    ap.add_argument('--figures',nargs='+',choices=engine.FIGURES,default=engine.FIGURES)
    ap.add_argument('--check-only',action='store_true')
    ap.add_argument('--plot-only',action='store_true')
    args=ap.parse_args()
    if args.workers<1 or args.max_calls<1 or (args.n_mc is not None and args.n_mc<1):ap.error('Positive workers, max-calls and n-mc required.')
    if len(set(args.figures))!=len(args.figures):ap.error('Duplicate figure names.')
    cfgpath=(ROOT/args.config).resolve();out=(ROOT/args.output).resolve();p=engine.load_protocol(cfgpath)
    engine.validate_inputs(ROOT,p,args.figures)
    tasks={name:engine.build_tasks(ROOT,p,name,args.n_mc) for name in args.figures}
    counts={name:(64 if name=='tradeoff' else len(tasks[name])) for name in args.figures}
    print('PLAN:',json.dumps(counts),'(tradeoff64 are all reused; convergence3 are Proposed only)',flush=True)
    if args.check_only:
        # Prepare one representative start for every setting/scheme without invoking solvers.
        for name,jobs in tasks.items():
            seen=set()
            for t in jobs:
                key=(t['parameter_value'],t['scheme'])
                if key in seen:continue
                seen.add(key);cfg=engine.config_for_task(ROOT,p,t);g,r=engine.initial_pair(cfg,p,t['scheme']);engine.check_state(g,r,cfg,p,t['scheme'])
        print('CHECK ONLY PASSED. No optimizer invoked.',flush=True);return
    paths=engine.input_config_paths(ROOT,p)
    identity=dict(protocol_sha256=engine.sha(cfgpath),n_mc_override=args.n_mc,
        script_sha256=engine.sha(Path(__file__)),engine_sha256=engine.sha(ROOT/'experiments/paper_figures_unified.py'),
        assets_sha256=engine.sha(engine.ASSETS/'inputs_manifest.json'),
        original_configs={str(f.relative_to(ROOT)):engine.sha(f) for f in paths if f.exists()})
    if out.exists():
        meta=out/'run_metadata.json'
        if not meta.exists() or engine.read_json(meta)['identity']!=identity:raise RuntimeError('Output identity differs. Choose a NEW --output directory; do not mix old protocols.')
    else:
        out.mkdir(parents=True);engine.json_write(out/'run_metadata.json',dict(identity=identity,protocol=p,
            notes=['Four agreed schemes only; old Stage8 seven-scheme/multistart driver is not used.',
                   'Stage8/load objective weights are preserved from ORIGINAL source configs.',
                   'Tradeoff alone uses its frozen original w*(R_DL+R_UL)/10+(1-w)*R_S/5 objective.',
                   'Power/SI/load curves are raw mean WSR; no nondominance filtering or smoothing.',
                   'Fixed geometry schemes have inner stopping only; joint schemes have two-step outer stopping.']))
        configs=out/'source_configs';configs.mkdir()
        import shutil
        shutil.copy2(cfgpath,configs/'paper_figures_unified.yaml')
        for f in paths:
            if f.exists():shutil.copy2(f,configs/f.name)
    meta=engine.read_json(out/'run_metadata.json');meta['current_styles_sha256']=engine.sha(ROOT/'experiments/paper_figure_styles.py')
    engine.json_write(out/'run_metadata.json',meta)
    import shutil
    shutil.copy2(ROOT/'experiments/paper_figure_styles.py',out/'source_configs/paper_figure_styles.py')
    status=engine.read_json(out/'status.json') if (out/'status.json').exists() else {}
    attempted=[]
    for figure in args.figures:
        started=time.perf_counter();print('\nFIGURE START:',figure,flush=True)
        try:result=execute_figure(ROOT,out,p,figure,tasks[figure],args.workers,args.max_calls,args.plot_only)
        except Exception as exc:
            result=dict(complete=False,errors=[dict(error=str(exc),traceback=traceback.format_exc())])
            engine.json_write(out/'logs'/f'{figure}_failure.json',result);print('FIGURE FAILED:',figure,str(exc),flush=True)
        result['seconds_this_invocation']=time.perf_counter()-started;status[figure]=result;attempted.append(result)
        engine.json_write(out/'status.json',status)
        print('FIGURE END:',figure,'complete=',result['complete'],flush=True)
    allcomplete=all(status.get(f,{}).get('complete',False) for f in engine.FIGURES)
    engine.json_write(out/'completion.json',dict(complete=allcomplete,figures={f:status.get(f,{}).get('complete',False) for f in engine.FIGURES},four_schemes=list(p['schemes']),initial_UL_power_per_user_mW=.25))
    print('OUTPUT:',out,'ALL FIVE COMPLETE:',allcomplete,flush=True)
    if not all(r['complete'] for r in attempted):raise SystemExit(1)

def main25():
    """25 mW multi-start protocol; default next step is the DL-power figure."""
    import yaml
    import shutil
    import numpy as np
    import run_regular25_multistart_pilot as high
    ap=argparse.ArgumentParser(description=main25.__doc__)
    ap.add_argument('--config',type=Path,default=Path('configs/paper_figures_unified.yaml'))
    ap.add_argument('--output',type=Path,default=Path('results/paper_figures_regular25_multistart'))
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--max-calls',type=int,default=2)
    ap.add_argument('--n-mc',type=int,default=10)
    ap.add_argument('--figures',nargs='+',choices=engine.FIGURES,default=['p_dl_max'])
    ap.add_argument('--reuse-base',type=Path,default=Path('results/regular25_multistart_pilot'))
    ap.add_argument('--reuse-extension',type=Path,default=Path('results/regular25_scene8001_20to40'))
    ap.add_argument('--no-reuse',action='store_true')
    ap.add_argument('--check-only',action='store_true')
    ap.add_argument('--plot-only',action='store_true')
    a=ap.parse_args()
    if min(a.workers,a.max_calls,a.n_mc)<1:ap.error('workers/max-calls/n-mc must be positive')
    if len(set(a.figures))!=len(a.figures):ap.error('Duplicate figures')
    cfgpath=(ROOT/a.config).resolve();p=yaml.safe_load(cfgpath.read_text(encoding='utf-8'))
    p['protocol'].update(initial_ul_power_W=.025,tol_inner=1e-4,tol_outer=1e-4,max_inner_iter=1200,max_outer_iter=1500,saved_initialization_authoritative=True)
    starts={'proposed':5,'b6':3,'b2':1,'b3':1}
    if p['schemes']!=list(starts):raise RuntimeError('Expected four agreed schemes')
    engine.validate_inputs(ROOT,p,a.figures)
    tasks={f:engine.build_tasks(ROOT,p,f,a.n_mc) for f in a.figures}
    out=(ROOT/a.output).resolve()
    def folder(base,t,s):return base/f'candidate_{s}'/'tasks'/engine.task_key(t)
    def verified(path,t,s):
        d=engine.read_json(path/'record.json');row=d['row'];cfg=engine.config_for_task(ROOT,p,t)
        if any(row[k]!=v for k,v in t.items()):raise RuntimeError('Checkpoint task differs: '+str(path))
        if d['cfg_sha256']!=engine.config_digest(cfg):raise RuntimeError('Checkpoint config differs: '+str(path))
        if d['initial_sha256']!=engine.sha(path/'initial.npz') or d['state_sha256']!=engine.sha(path/d['state_file']):raise RuntimeError('Checkpoint hash differs: '+str(path))
        g0,r0,_=engine.load_state(path/'initial.npz');g,r,h=engine.load_state(path/d['state_file'])
        expected,_=high.pilot.pair(cfg,p,t['scheme'],s)
        for n in ('endpoints','tx_positions','rx_positions'):
            if not np.allclose(getattr(g0,n),getattr(expected,n),rtol=0,atol=1e-12):raise RuntimeError('Geometry candidate differs: '+str(path))
        engine.check_state(g0,r0,cfg,p,t['scheme']);engine.check_state(g,r,cfg,p,t['scheme'],g0)
        if not np.allclose(abs(r0.q_ul)**2,.025,rtol=0,atol=1e-12):raise RuntimeError('Initial UL power differs: '+str(path))
        if abs(engine.metrics_values(g,r,cfg)['weighted_sum_rate']-row['weighted_sum_rate'])>1e-8:raise RuntimeError('Saved WSR differs')
        if row['converged']:
            required=2 if t['scheme'] in ('proposed','b6') else 1;delta=np.diff(h)
            if len(delta)<required or min(delta)<-1e-8 or max(abs(delta[-required:]))>1e-4:raise RuntimeError('Stopping history differs')
        return dict(row,start_id=s)
    identity=dict(config_sha256=engine.sha(cfgpath),n_mc=a.n_mc,initial_UL_W=.025,starts=starts,
        runner_sha256=engine.sha(Path(__file__)),engine_sha256=engine.sha(Path(engine.__file__)),
        initializer_sha256=engine.sha(Path(high.pilot.__file__)),
        sources={n:engine.sha(ROOT/n) for n in ['algorithm.py','baselines.py','ma_baseline.py','initialization.py','inner_solver.py','outer_solver.py','channels.py','metrics.py','geometry.py','config.py','stage8.py','utils.py','state.py']},
        configs={str(f.relative_to(ROOT)):engine.sha(f) for f in engine.input_config_paths(ROOT,p) if f.exists()})
    meta=out/'run_metadata.json'
    # Only the exact previously delivered runner/initializer can migrate automatically.
    legacy_hashes={'runner_sha256': '9ea0e481129d96c3e5e7bf7111fe17d5d3de73774c4eaa45cc2341e613b5f0b0', 'engine_sha256': '46fc4afbb02cfd567779c2a21adff168e8982ccd291bc58700f588942a633233', 'initializer_sha256': '10fc1ce6f12ed2d6ac19927ea572e7c718408c910705f5c657f1a2859d8e1775'}
    migrating=False
    if out.exists():
        if not meta.exists():raise RuntimeError('Output has no metadata; choose a NEW --output')
        previous=engine.read_json(meta)['identity']
        if previous!=identity:
            migrating=all(previous.get(k)==v for k,v in identity.items() if k not in legacy_hashes) and all(previous.get(k)==v for k,v in legacy_hashes.items())
            if not migrating:raise RuntimeError('Output identity differs beyond the known UL initialization fix; choose a NEW --output')
    bad=[]
    for f,tt in tasks.items():
        for t in tt:
            for start in range(starts[t['scheme']]):
                path=folder(out,t,start)
                initial=path/'initial.npz'
                if not initial.exists():continue
                with np.load(initial) as z:power=np.abs(z['q_ul'])**2
                if np.allclose(power,.025,rtol=0,atol=1e-12):continue
                if not np.allclose(power,.00025,rtol=0,atol=1e-12):raise RuntimeError('Unrecognized initial UL power: '+str(initial))
                bad.append(path)
    if bad:print('WRONG 0.25 mW STARTS:',len(bad),'(will be moved to backup, never relabeled)',flush=True)
    if a.plot_only and bad:raise RuntimeError('Wrong initial power requires optimization; run without --plot-only')
    if not a.check_only and (bad or migrating):
        backup=out/'repair_backup'/time.strftime('%Y%m%d_%H%M%S')
        backup.mkdir(parents=True,exist_ok=False)
        if meta.exists():shutil.copy2(meta,backup/'run_metadata.json')
        for path in bad:
            dest=backup/path.relative_to(out);dest.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(path),str(dest))
        for name in ['tables','figures','tasks','status.json','completion.json']:
            path=out/name
            if path.exists():shutil.move(str(path),str(backup/name))
        engine.json_write(backup/'repair.json',dict(reason='Windows spawn used low module default for perturbed starts',quarantined=[str(x.relative_to(out)) for x in bad]))
        engine.json_write(meta,dict(identity=identity,protocol=p,notes=['UL power now comes from explicit job protocol.']))
        print('REPAIR BACKUP:',backup,flush=True)
    bad_set=set(bad)

    sources=[(ROOT/a.reuse_base).resolve(),(ROOT/a.reuse_extension).resolve()]
    copies=[];jobs={}
    for f,tt in tasks.items():
        jobs[f]=[]
        for t in tt:
            cfg=engine.config_for_task(ROOT,p,t)
            for s in range(starts[t['scheme']]):
                g,r=high.pilot.pair(cfg,p,t['scheme'],s);engine.check_state(g,r,cfg,p,t['scheme'])
                dst=folder(out,t,s)
                if dst not in bad_set and (dst/'record.json').exists() and verified(dst,t,s)['converged']:continue
                if f=='p_dl_max' and t['scene_seed']==8001 and not a.no_reuse:
                    found=False
                    for source in sources:
                        src=folder(source,t,s)
                        if (src/'record.json').exists() and verified(src,t,s)['converged']:
                            copies.append((src,dst));found=True;break
                    if not found:raise FileNotFoundError('Missing scene8001 checkpoint: '+str(dst)+'; set --reuse-base/--reuse-extension or explicitly use --no-reuse')
                else:jobs[f].append((str(out),p,t,s,a.max_calls))
    print('PLAN:',json.dumps({f:len(j) for f,j in jobs.items()}),'new/unfinished starts; reused:',len(copies),flush=True)
    if a.check_only:print('CHECK ONLY PASSED. No optimizer invoked.',flush=True);return
    if a.plot_only and any(jobs.values()):raise RuntimeError('Plot-only requires every start to have converged')
    out.mkdir(parents=True,exist_ok=True)
    engine.json_write(meta,dict(identity=identity,protocol=p,notes=['Best start per scene, then average common scenes.','Tradeoff retains frozen 0.25 mW results.','No cross-parameter warm starts.']))
    for src,dst in copies:
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(src,dst,dirs_exist_ok=True)
    status=engine.read_json(out/'status.json') if (out/'status.json').exists() else {}
    for f,tt in tasks.items():
        if f=='tradeoff':
            engine.plot_tradeoff(out,p);status[f]=dict(complete=True,frozen_initial_UL_mW=.25);engine.json_write(out/'status.json',status);continue
        errors=[]
        if jobs[f]:
            with ProcessPoolExecutor(max_workers=a.workers) as pool:
                pending={pool.submit(high.pilot.run_job,j):j for j in jobs[f]}
                for i,future in enumerate(as_completed(pending),1):
                    try:future.result()
                    except Exception as exc:
                        j=pending[future];errors.append(dict(task=j[2],start_id=j[3],error=repr(exc)));print('FAILED',errors[-1],flush=True)
                    print(f'{f}: {i}/{len(pending)} starts returned',flush=True)
        rows=[];winners=[]
        for t in tt:
            group=[]
            for s in range(starts[t['scheme']]):
                path=folder(out,t,s)
                if (path/'record.json').exists():
                    row=verified(path,t,s);rows.append(row);group.append(row)
            if len(group)==starts[t['scheme']] and all(r['converged'] for r in group):winners.append(max(group,key=lambda r:r['weighted_sum_rate']))
        if rows:engine.csv_write(out/'tables'/f'{f}_all_starts.csv',rows,list(rows[0]))
        complete=not errors and len(winners)==len(tt)
        if complete:
            engine.csv_write(out/'tables'/f'{f}_per_scene.csv',winners,list(winners[0]))
            if f=='convergence':
                for t,r in zip(tt,winners):
                    dst=out/'tasks'/engine.task_key(t);dst.mkdir(parents=True,exist_ok=True)
                    for src in folder(out,t,r['start_id']).iterdir():
                        if src.is_file():shutil.copy2(src,dst/src.name)
                engine.plot_convergence(out,p,tt)
            else:engine.plot_sweep(out,p,f,tt,winners)
        status[f]=dict(complete=complete,expected_starts=sum(starts[t['scheme']] for t in tt),converged_starts=sum(r['converged'] for r in rows),selected_scene_cells=len(winners),errors=errors)
        engine.json_write(out/'status.json',status)
    done=all(status[f]['complete'] for f in a.figures)
    engine.json_write(out/'completion.json',dict(requested_figures_complete=done,figures=status,n_mc=a.n_mc,initial_UL_power_per_user_mW=25))
    print('OUTPUT:',out,'REQUESTED FIGURES COMPLETE:',done,flush=True)
    if not done:raise SystemExit('Incomplete: rerun the same command to continue checkpoints.')


if __name__=='__main__':main25()
