"""Compute ONLY 9 new weights for Proposed+MA and merge with verified prior data.

Plot 20 common weights; preserve all 27 old + 9 new weights in a separate CSV.
No change to initialization, objective, physical constraints or stopping rules.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import numpy as np
import stage10_planar_pm_common as common
from stage10_planar_pm_support import csv_read,csv_write,json_write,sha,FIELDS,values,plot
from planar_pm_initialization import make_initials
from run_stage10_regular_planar_pm import run_one

NEW_WEIGHTS=[.015,.11,.12,.13,.14,.145,.16,.18,.225]
PLOT_WEIGHTS=[.001,.005,.015,.03,.10,.11,.12,.125,.13,.14,
              .145,.15,.16,.175,.18,.20,.225,.25,.42,.46]
SCHEMES=['proposed','b6']
SOURCE=Path(__file__).resolve().parent/'stage10_regular20_source'/'regular_planar_pm_full27.zip'


@contextmanager
def source_state(data):
    with TemporaryDirectory() as temp:
        p=Path(temp)/'state.npz';p.write_bytes(data)
        yield common.load_state(p)


def verify_source(root,source):
    common.verify_files(root)
    scene,opts=common.load_scene(root)
    from channels import build_channels
    from metrics import compute_performance
    from geometry import build_geometry,check_geometry_constraints
    from utils import check_resource_constraints
    import ma_baseline as ma
    import baselines
    expected=make_initials(scene,2042)
    raw_states={};initial_bytes={};initials={};max_error=0.
    with ZipFile(source) as z:
        candidates=[n[:-len('completion.json')] for n in z.namelist() if n=='completion.json' or n.endswith('/completion.json')]
        if len(candidates)!=1:raise RuntimeError('Expected exactly one source result directory.')
        prefix=candidates[0]
        finish=json.loads(z.read(prefix+'completion.json'))
        meta=json.loads(z.read(prefix+'run_metadata.json'))
        protocol=meta['protocol']
        if not finish['complete'] or finish['converged_points']!=54:
            raise RuntimeError('Source must contain the completed regular-planar 54-point experiment.')
        for key,value in {'toma_plane':'x=0 (y-z plane through cable anchor)',
            'ma_initialization':'regular UPA, no random position jitter',
            'tol_inner':1e-4,'tol_outer':1e-4,'inner_consecutive':1,'outer_consecutive':2,
            'max_inner_iter_per_call':1200,'max_outer_iter_per_call':1500,
            'cross_weight_warm_start':False}.items():
            if protocol.get(key)!=value:raise RuntimeError('Source protocol mismatch: '+key)
        for sc in SCHEMES:
            data=z.read(prefix+f'initial_states/{sc}.npz');initial_bytes[sc]=data
            with source_state(data) as state:g,r,_=state
            eg,er=expected[sc]
            # Windows/Linux trig routines may differ by a final floating-point bit.
            # Exact saved bytes are reused; tolerance is only for checking the rule.
            for name in ('endpoints','tx_positions','rx_positions'):
                if not np.allclose(getattr(g,name),getattr(eg,name),rtol=0,atol=1e-12):
                    raise RuntimeError('Source initial geometry differs from regular-planar rule.')
            for name in ('q_matrix','q_ul','b_ul','u_s'):
                if not np.allclose(getattr(r,name),getattr(er,name),rtol=1e-9,atol=1e-10):
                    raise RuntimeError('Source resource initialization differs.')
            check_resource_constraints(r,scene)
            initials[sc]=(g,r)
        rows=csv_read(z.read(prefix+'stage10_tradeoff.csv'))
        keys={(r['scheme'],float(r['omega_comm'])) for r in rows}
        if len(rows)!=54 or len(keys)!=54 or {r['scheme'] for r in rows}!=set(SCHEMES):
            raise RuntimeError('Source must contain 27 unique weights for each of Proposed and MA.')
        old_weights={float(r['omega_comm']) for r in rows}
        if set(NEW_WEIGHTS)&old_weights or not set(PLOT_WEIGHTS)-set(NEW_WEIGHTS)<=old_weights:
            raise RuntimeError('New/reused weight lists disagree with source.')
        for row in rows:
            sc=row['scheme'];w=float(row['omega_comm']);cfg=common.cfg_for(scene,w)
            if str(row['converged']).lower()!='true' or int(row['n_starts'])!=1 or int(row['best_seed'])!=2042:
                raise RuntimeError('Source convergence/initialization mismatch.')
            d=json.loads(z.read(prefix+f'records/{sc}_omega_{w:g}.json'))
            state_data=z.read(prefix+d['state_file'])
            if hashlib.sha256(state_data).hexdigest()!=d['state_sha256']:
                raise RuntimeError('Source state checksum mismatch.')
            if hashlib.sha256(initial_bytes[sc]).hexdigest()!=d['initialization_file_sha256']:
                raise RuntimeError('Source did not start from the shared initial state.')
            with source_state(state_data) as state:g,r,h=state
            check_resource_constraints(r,cfg)
            if sc=='proposed':
                check_geometry_constraints(g,cfg);rebuilt=build_geometry(g.endpoints,cfg)
                if not (np.allclose(g.tx_positions,rebuilt.tx_positions,rtol=0,atol=1e-12) and
                        np.allclose(g.rx_positions,rebuilt.rx_positions,rtol=0,atol=1e-12)):
                    raise RuntimeError('Source ToMA array/endpoints mismatch.')
            else:
                base=baselines.build_fpa_upa_geometry(cfg)
                if not ma._ma_geometry_is_feasible(g.tx_positions,g.rx_positions,
                    np.mean(base.tx_positions,axis=0),np.mean(base.rx_positions,axis=0),
                    opts['region_size'],opts['min_spacing'],cfg):
                    raise RuntimeError('Source MA geometry is infeasible.')
            v=values(compute_performance(build_channels(g,cfg),r,cfg))
            for key,value in v.items():
                err=abs(value-float(row[key]));max_error=max(max_error,err)
                if not np.isclose(value,float(row[key]),rtol=1e-9,atol=1e-9):
                    raise RuntimeError('Source true performance mismatch: '+key)
            ig,ir=initials[sc]
            first=float(compute_performance(build_channels(ig,cfg),ir,cfg).weighted_sum_rate)
            if not np.isclose(first,h[0],rtol=1e-9,atol=1e-9):
                raise RuntimeError('Source initial WSR mismatch.')
            if len(h)!=int(row['iterations'])+1 or len(h)<3 or not np.all(np.abs(np.diff(h[-3:]))<=1e-4):
                raise RuntimeError('Source outer stopping mismatch.')
            if any(b<a-1e-8*max(1,abs(a)) for a,b in zip(h,h[1:])):
                raise RuntimeError('Source WSR decreased.')
            raw_states[f'{sc}_omega_{w:g}.npz']=state_data
    print(f'Verified all 54 source points; maximum true-performance error={max_error:.3e}.',flush=True)
    print('Only new weights:',NEW_WEIGHTS,'; schemes: Proposed and MA.',flush=True)
    return rows,initial_bytes,raw_states,meta


def merge(out,old_rows):
    records=[common.read_json(p)['row'] for p in sorted((out/'records').glob('*.json'))]
    expected={(sc,w) for sc in SCHEMES for w in NEW_WEIGHTS}
    actual={(r['scheme'],float(r['omega_comm'])) for r in records}
    if len(actual)!=len(records) or not actual<=expected:
        raise RuntimeError('Unexpected or duplicate new-result records.')
    done=[r for r in records if r['converged']]
    all_rows=sorted(old_rows+done,key=lambda r:(float(r['omega_comm']),SCHEMES.index(r['scheme'])))
    if len({(r['scheme'],float(r['omega_comm'])) for r in all_rows})!=len(all_rows):
        raise RuntimeError('Duplicate source/new weights.')
    selected=[r for r in all_rows if float(r['omega_comm']) in PLOT_WEIGHTS]
    csv_write(out/'new_points.csv',records,FIELDS)
    csv_write(out/'all_36_weights.csv',all_rows,FIELDS)
    csv_write(out/'stage10_tradeoff.csv',selected,FIELDS)
    complete=len(done)==18
    json_write(out/'completion.json',dict(complete=complete,expected_new_points=18,
        converged_new_points=len(done),source_points_preserved=54,
        plot_source_points=22,plot_expected_points=40,plot_available_points=len(selected),
        all_expected_points=72,all_available_points=len(all_rows),new_weights=NEW_WEIGHTS,
        plot_weights=PLOT_WEIGHTS,schemes=SCHEMES,scene_seed=8001,resource_seed=2042,
        initialization_unchanged=True,cross_weight_warm_start=False))
    if complete:
        plot(selected,out/'figures_20_weights')
        plot(all_rows,out/'figures_all_36_weights')
        print('COMPLETE: 18 new points; 20-weight figure uses 22 reused + 18 new = 40 points.',flush=True)
    else:
        print(f'Progress: {len(done)}/18 new points converged. Final figures wait for all new points.',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--source-zip',type=Path,default=SOURCE)
    ap.add_argument('--output',type=Path,default=Path('results/stage10/regular_planar_pm_refined20'))
    ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--max-calls',type=int,default=20)
    ap.add_argument('--weights',type=float,nargs='+',choices=NEW_WEIGHTS,default=NEW_WEIGHTS)
    ap.add_argument('--check-only',action='store_true')
    ap.add_argument('--plot-only',action='store_true')
    args=ap.parse_args()
    if args.workers<1 or args.max_calls<1:ap.error('--workers and --max-calls must be positive.')
    root=args.project_root.resolve();source=args.source_zip.resolve();out=(root/args.output).resolve()
    old,initial_bytes,states,source_meta=verify_source(root,source)
    if args.check_only:
        print('CHECK ONLY: no optimizer invoked; no new result written.',flush=True);return
    base=Path(__file__).resolve().parent
    identity=dict(entry_sha256=sha(Path(__file__)),source_sha256=sha(source),
        dependencies={n:sha(base/n) for n in ('run_stage10_regular_planar_pm.py',
            'stage10_planar_pm_common.py','stage10_planar_pm_support.py','planar_pm_initialization.py')},
        assets_manifest_sha256=sha(common.ASSETS/'manifest.json'),project_root=str(root),
        plot_weights=PLOT_WEIGHTS,new_weights=NEW_WEIGHTS,schemes=SCHEMES)
    if out.exists():
        if not (out/'run_metadata.json').exists() or common.read_json(out/'run_metadata.json')['identity']!=identity:
            raise RuntimeError('Output identity differs. Choose a new --output directory.')
        for sc,data in initial_bytes.items():
            if (out/'initial_states'/f'{sc}.npz').read_bytes()!=data:
                raise RuntimeError('Saved initialization differs from the verified source.')
    else:
        if args.plot_only:raise RuntimeError('No saved output exists for --plot-only.')
        out.mkdir(parents=True)
        for folder in ('initial_states','reused_states','records','states'):(out/folder).mkdir()
        for sc,data in initial_bytes.items():(out/'initial_states'/f'{sc}.npz').write_bytes(data)
        for name,data in states.items():(out/'reused_states'/name).write_bytes(data)
        csv_write(out/'reused_54_points.csv',old,FIELDS)
        json_write(out/'run_metadata.json',dict(identity=identity,source_protocol=source_meta['protocol'],
            initialization='Exact frozen bytes copied from the completed uniform-planar source',
            original_objective_unchanged=True,physical_constraints_unchanged=True,
            stopping='absolute WSR change 1e-4; inner once, outer twice',
            scene_seed=8001,resource_seed=2042,n_starts_per_scheme=1,
            no_cross_weight_warm_start=True,single_scene_preview=True))
    if args.plot_only:merge(out,old);return
    jobs=[(str(root),str(out),sc,w,args.max_calls) for w in dict.fromkeys(args.weights) for sc in SCHEMES]
    print(f'Requested {len(jobs)} new jobs. Previously converged new jobs will be skipped.',flush=True)
    try:
        if args.workers==1:
            for job in jobs:run_one(*job);merge(out,old)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures=[pool.submit(run_one,*job) for job in jobs]
                for f in as_completed(futures):f.result();merge(out,old)
    finally:merge(out,old)


if __name__=='__main__':main()
