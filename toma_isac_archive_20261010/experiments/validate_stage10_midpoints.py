"""Stage 10: verify physically feasible intermediate communication/sensing states.

Independent *diagnostic* only. No cross-weight warm start, and NO optimization of
intermediate states. Replays only two historical cold-start anchors per scheme;
fixes one anchor's feasible geometry and interpolates feasible resource controls.
Re-evaluates SINR/rates using the original metrics.py.

Run in CMD, project root:
  python -u .\\experiments\\validate_stage10_midpoints.py --dry-run
  python -u .\\experiments\\validate_stage10_midpoints.py
  python -u .\\experiments\\validate_stage10_midpoints.py --schemes proposed b6
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from channels import build_channels
from metrics import compute_performance
from inner_solver import update_sensing_combiners, update_uplink_combiners
from utils import check_resource_constraints
from geometry import check_geometry_constraints
from baselines import build_fpa_upa_geometry
from ma_baseline import run_ma_fp, _ma_geometry_is_feasible
from algorithm import run_joint_algorithm
from stage8 import load_stage8_experiment_config, make_scene_config
from state import GeometryState, ResourceState

CONFIG = ROOT / 'configs' / 'stage10_tradeoff_cold20.yaml'
SOURCE_CSV = ROOT / 'results' / 'stage10' / 'cold_wide20_dl114_tol1e4' / 'stage10_tradeoff.csv'
OUTPUT = ROOT / 'results' / 'stage10' / 'midpoint_feasibility_cold20_dl114'
SCHEME_ANCHORS = {'proposed': (0.001, 0.4), 'b6': (0.001, 0.3)}
EVAL_WEIGHTS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55)
FIELDS = ('scheme','frozen_geometry','source_sensing_weight','source_comm_weight',
          'alpha_comm','sum_rate_dl','sum_rate_ul','sum_rate_comm',
          'sum_rate_sensing','tx_power_w','ul_power_w','W_power_w','V_power_w',
          'middle_window','dominated_by_existing_cold20','receiver_status')
SCORE_FIELDS = ('scheme','target_weight','cold20_selected_wsr',
                'best_reweighted_cold20_wsr','best_reweighted_anchor_wsr',
                'best_constructed_wsr','best_constructed_rc','best_constructed_rs',
                'constructed_minus_cold20','constructed_minus_anchor_envelope',
                'note')


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f'Cannot find: {path}')
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def weights_cfg(base,omega):
    # Matches the original cold20 configuration with K=J=L=5.
    return replace(base,
        weight_dl=np.full(base.k_dl,omega/(2*base.k_dl)),
        weight_ul=np.full(base.j_ul,omega/(2*base.j_ul)),
        weight_s=np.full(base.l_target,(1-omega)/base.l_target))


def wsr_from_rates(rc,rs,w):
    return float(w*rc/10.0 + (1.0-w)*rs/5.0)


def wsr_weighted_performance(perf,cfg):
    return float(np.dot(cfg.weight_dl,perf.rate_dl)
                 + np.dot(cfg.weight_ul,perf.rate_ul)
                 + np.dot(cfg.weight_s,perf.rate_s))


def get_expected(rows,scheme,w):
    matches=[r for r in rows if r['scheme']==scheme
             and abs(float(r['omega_comm'])-w)<1e-10]
    if len(matches)!=1:
        raise RuntimeError(f'Expected exactly one row for {scheme} at {w}; got {len(matches)}')
    row=matches[0]
    if row.get('converged','').strip().lower()!='true':
        raise RuntimeError(f'The original point did not converge: {scheme} {w}')
    return row


def state_path(scheme,w):
    return OUTPUT/'anchors'/f'{scheme}_w{w:.3f}.npz'


def hash_inputs():
    # Fail rather than accidentally mixing cached states generated with old code/config.
    names=[CONFIG, ROOT/'algorithm.py',ROOT/'ma_baseline.py',ROOT/'inner_solver.py',
           ROOT/'metrics.py',ROOT/'channels.py',ROOT/'stage8.py',ROOT/'geometry.py']
    h=sha256()
    for p in names:
        h.update(str(p.relative_to(ROOT)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def snapshot_state(result):
    g,r=result.geometry,result.resources
    return dict(endpoints=np.asarray(g.endpoints),
                tx_positions=np.asarray(g.tx_positions),rx_positions=np.asarray(g.rx_positions),
                q_matrix=np.asarray(r.q_matrix),q_ul=np.asarray(r.q_ul),
                b_ul=np.asarray(r.b_ul),u_s=np.asarray(r.u_s))


def get_state(path):
    with np.load(path,allow_pickle=False) as d:
        g=GeometryState(endpoints=d['endpoints'].copy(),tx_positions=d['tx_positions'].copy(),
                        rx_positions=d['rx_positions'].copy())
        r=ResourceState(q_matrix=d['q_matrix'].copy(),q_ul=d['q_ul'].copy(),
                        b_ul=d['b_ul'].copy(),u_s=d['u_s'].copy())
    return g,r


def check_geometry(scheme,g,cfg,experiment):
    if scheme=='proposed':
        check_geometry_constraints(g,cfg)
    elif scheme=='b6':
        base=build_fpa_upa_geometry(cfg)
        tx_center=np.mean(base.tx_positions,axis=0)
        rx_center=np.mean(base.rx_positions,axis=0)
        if not _ma_geometry_is_feasible(g.tx_positions,g.rx_positions,tx_center,rx_center,
                                        experiment.ma_region_size,experiment.ma_min_spacing,cfg):
            raise ValueError('B6 geometry constraint violation')
    else:
        raise ValueError(scheme)


def validate_state(scheme,g,r,cfg,experiment):
    check_geometry(scheme,g,cfg,experiment)
    check_resource_constraints(r,cfg)
    p=compute_performance(build_channels(g,cfg),r,cfg)
    return p


def cold_replay(scheme,w,seed,cfg,experiment):
    rng=np.random.default_rng(int(seed))
    if scheme=='proposed':
        return run_joint_algorithm(cfg,rng=rng,verbose=False,
                                   position_update_mode='fp_aware',fp_aware_candidate_iters=2)
    if scheme=='b6':
        return run_ma_fp(cfg,rng=rng,verbose=False,
                         region_size=experiment.ma_region_size,
                         min_spacing=experiment.ma_min_spacing,
                         initial_step=experiment.ma_initial_step,
                         randomize_initial_geometry=(int(seed)!=int(experiment.fixed_upa_seed)),
                         initial_jitter_fraction=experiment.ma_initial_jitter_fraction)
    raise ValueError(scheme)


def ensure_anchor(scheme,w,original,base,experiment,fingerprint,allow_drift):
    row=get_expected(original,scheme,w)
    seed=int(row['best_seed'])
    cfg=weights_cfg(base,w)
    path=state_path(scheme,w)
    meta_path=path.with_suffix('.json')
    expected=float(row['weighted_sum_rate'])
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and meta_path.exists():
        meta=json.loads(meta_path.read_text(encoding='utf-8'))
        if (meta.get('fingerprint')!=fingerprint or meta.get('seed')!=seed
                or abs(float(meta.get('omega_comm',-1))-w)>1e-12):
            raise RuntimeError(f'Cached anchor inconsistent: {path}. Do not mix versions; '
                               'rename the midpoint output directory before rerunning.')
        source='cached'
    else:
        print(f'  [COLD REPLAY] {scheme}, omega={w}, seed={seed}',flush=True)
        t0=perf_counter()
        result=cold_replay(scheme,w,seed,cfg,experiment)
        if not result.converged:
            raise RuntimeError(f'Anchor replay did not converge: {scheme} {w}, '
                               f'reason={getattr(result,"stop_reason","unknown")}. '
                               'Do not treat a nonconverged anchor as the original cold20 point.')
        np.savez_compressed(path,**snapshot_state(result))
        meta_path.write_text(json.dumps(dict(scheme=scheme,omega_comm=w,seed=seed,
                              fingerprint=fingerprint,replay_elapsed_s=perf_counter()-t0),
                              indent=2),encoding='utf-8')
        source='replayed'
    g,r=get_state(path)
    p=validate_state(scheme,g,r,cfg,experiment)
    actual=float(p.weighted_sum_rate)
    diff=actual-expected
    drift=abs(diff)>max(2e-3,3e-3*abs(expected))
    print(f'  [ANCHOR {source}] omega={w}: historical={expected:.6f}, '
          f'recomputed={actual:.6f}, delta={diff:+.6f}',flush=True)
    if drift and not allow_drift:
        raise RuntimeError(f'Anchor mismatch {scheme} {w}: {diff:+.6f}. '
                           'Verify YAML/noise/RCS/solver versions. If you intentionally '
                           'accept a different replay, rerun with --allow-replay-drift '
                           'but do NOT compare it as a controlled identical-run experiment.')
    return (g,r),dict(scheme=scheme,source_weight=w,seed=seed,
                      old_wsr=expected,replayed_wsr=actual,
                      replay_abs_diff=abs(diff),replay_drift=drift,
                      sum_rate_comm=float(np.sum(p.rate_dl)+np.sum(p.rate_ul)),
                      sum_rate_sensing=float(np.sum(p.rate_s)),source=source)


def align_q_columns(source_q,target_q):
    """Remove irrelevant independent stream phases before convex interpolation."""
    aligned=target_q.copy()
    for j in range(source_q.shape[1]):
        corr=np.vdot(target_q[:,j],source_q[:,j])
        if abs(corr)>1e-18:
            aligned[:,j] *= np.exp(1j*np.angle(corr))
    return aligned


def make_resource_candidate(low,high,alpha):
    """Q lies in a Frobenius ball and q_ul lies in a box: both convex.

    Geometry is NEVER mixed. b/u combiners are recomputed separately to
    respect unit norm. Result is a valid instantaneous operating point,
    not a time-averaged interpolation of displayed rate coordinates.
    """
    qhigh=align_q_columns(low.q_matrix,high.q_matrix)
    return ResourceState(q_matrix=(1-alpha)*low.q_matrix+alpha*qhigh,
                         q_ul=(1-alpha)*low.q_ul+alpha*high.q_ul,
                         b_ul=low.b_ul.copy(),u_s=low.u_s.copy())


def rates_for_candidate(g,r,cfg,channels):
    receiver_status='optimized'
    try:
        r.u_s=update_sensing_combiners(channels,r,cfg)
        r.b_ul=update_uplink_combiners(channels,r,cfg)
    except (ValueError,np.linalg.LinAlgError) as err:
        # The original valid receiver beams remain available; use them rather
        # than silently dropping a numerically difficult feasible candidate.
        receiver_status=f'fallback:{type(err).__name__}'
    check_resource_constraints(r,cfg)
    p=compute_performance(channels,r,cfg)
    return p,receiver_status


def generate_midpoints(scheme,low,high,original,base,experiment,alphas,comm_min,rs_min):
    cfg=weights_cfg(base,0.5)
    existing=[r for r in original if r['scheme']==scheme]
    result=[]
    for anchor_name,frozen,ra,rb in (
        ('sensing_anchor',low[0],low[1],high[1]),
        ('communication_anchor',high[0],low[1],high[1]),
    ):
        g=frozen
        check_geometry(scheme,g,cfg,experiment)
        channels=build_channels(g,cfg)
        for alpha in alphas:
            resources=make_resource_candidate(ra,rb,float(alpha))
            p,status=rates_for_candidate(g,resources,cfg,channels)
            rc=float(np.sum(p.rate_dl)+np.sum(p.rate_ul))
            rs=float(np.sum(p.rate_s))
            dominated=any(
                float(o['sum_rate_comm'])>=rc-1e-6 and
                float(o['sum_rate_sensing'])>=rs-1e-6 and
                (float(o['sum_rate_comm'])>rc+1e-6 or
                 float(o['sum_rate_sensing'])>rs+1e-6)
                for o in existing)
            wpower=float(np.linalg.norm(resources.q_matrix[:,:cfg.k_dl],'fro')**2)
            vpower=float(np.linalg.norm(resources.q_matrix[:,cfg.k_dl:],'fro')**2)
            result.append(dict(scheme=scheme,frozen_geometry=anchor_name,
                               source_sensing_weight=SCHEME_ANCHORS[scheme][0],
                               source_comm_weight=SCHEME_ANCHORS[scheme][1],
                               alpha_comm=float(alpha),sum_rate_dl=float(np.sum(p.rate_dl)),
                               sum_rate_ul=float(np.sum(p.rate_ul)),
                               sum_rate_comm=rc,sum_rate_sensing=rs,
                               tx_power_w=wpower+vpower,
                               ul_power_w=float(np.sum(resources.q_ul**2)),
                               W_power_w=wpower,V_power_w=vpower,
                               middle_window=bool(rc>=comm_min and rs>=rs_min),
                               dominated_by_existing_cold20=bool(dominated),
                               receiver_status=status))
    return result


def score_candidates(all_generated,original,schemes):
    scores=[]
    for scheme in schemes:
        existing=[r for r in original if r['scheme']==scheme]
        generated=[r for r in all_generated if r['scheme']==scheme]
        for w in EVAL_WEIGHTS:
            cold=get_expected(original,scheme,w)
            cold_wsr=float(cold['weighted_sum_rate'])
            existing_best=max(wsr_from_rates(float(r['sum_rate_comm']),
                                             float(r['sum_rate_sensing']),w)
                              for r in existing)
            pair=SCHEME_ANCHORS[scheme]
            anchor_best=max(wsr_from_rates(float(get_expected(original,scheme,a)['sum_rate_comm']),
                                           float(get_expected(original,scheme,a)['sum_rate_sensing']),w)
                            for a in pair)
            best=max(generated,key=lambda r:wsr_from_rates(float(r['sum_rate_comm']),
                                                 float(r['sum_rate_sensing']),w))
            f=wsr_from_rates(float(best['sum_rate_comm']),
                             float(best['sum_rate_sensing']),w)
            scores.append(dict(scheme=scheme,target_weight=w,
                cold20_selected_wsr=cold_wsr,
                best_reweighted_cold20_wsr=existing_best,
                best_reweighted_anchor_wsr=anchor_best,
                best_constructed_wsr=f,
                best_constructed_rc=best['sum_rate_comm'],
                best_constructed_rs=best['sum_rate_sensing'],
                constructed_minus_cold20=f-cold_wsr,
                constructed_minus_anchor_envelope=f-anchor_best,
                note='Instantaneously feasible; not FP/RCG optimized; not a global optimum bound'))
    return scores


def make_plots(original,midpoints,scores,schemes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,len(schemes),figsize=(7*len(schemes),5.5),squeeze=False)
    for ax,scheme in zip(axes.flat,schemes):
        orig=sorted([r for r in original if r['scheme']==scheme],
                    key=lambda r:float(r['omega_comm']))
        ax.scatter([float(r['sum_rate_comm']) for r in orig],
                   [float(r['sum_rate_sensing']) for r in orig],
                   color='0.55',alpha=.65,s=34,label='Original 20-point cold results')
        for freeze,mark in [('sensing_anchor','o'),('communication_anchor','s')]:
            pts=sorted([r for r in midpoints if r['scheme']==scheme
                        and r['frozen_geometry']==freeze],key=lambda r:float(r['alpha_comm']))
            if pts:
                ax.plot([r['sum_rate_comm'] for r in pts],
                        [r['sum_rate_sensing'] for r in pts],'-',alpha=.7,lw=1.2,
                        label=f'Feasible mix, {freeze.replace("_"," ")}')
                ax.scatter([r['sum_rate_comm'] for r in pts],
                           [r['sum_rate_sensing'] for r in pts],
                           c=[float(r['alpha_comm']) for r in pts],cmap='viridis',
                           vmin=0,vmax=1,marker=mark,s=27,zorder=4)
        ax.set_title('Proposed' if scheme=='proposed' else 'MA + FP (B6)')
        ax.set_xlabel('Communication sum rate (bit/s/Hz)')
        ax.set_ylabel('Sensing sum rate (bit/s/Hz)')
        ax.grid(alpha=.2)
        ax.legend(loc='best',fontsize=8)
    fig.suptitle('Physically feasible intermediate candidates (not optimized Pareto points)')
    fig.tight_layout()
    d=OUTPUT/'figures'; d.mkdir(parents=True,exist_ok=True)
    fig.savefig(d/'midpoint_feasibility.png',dpi=220,bbox_inches='tight')
    fig.savefig(d/'midpoint_feasibility.pdf',bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,len(schemes),figsize=(7*len(schemes),4.8),squeeze=False)
    for ax,scheme in zip(axes.flat,schemes):
        a=[r for r in scores if r['scheme']==scheme]
        xx=[float(r['target_weight']) for r in a]
        ax.plot(xx,[float(r['cold20_selected_wsr']) for r in a],'o-',label='Cold20 selected WSR')
        ax.plot(xx,[float(r['best_reweighted_anchor_wsr']) for r in a],'s--',label='Best of two true anchors (reweighted)')
        ax.plot(xx,[float(r['best_constructed_wsr']) for r in a],'^-',label='Best tested feasible mixture')
        ax.set_title('Proposed' if scheme=='proposed' else 'MA + FP (B6)')
        ax.set_xlabel('Communication weight')
        ax.set_ylabel('WSR (bit/s/Hz)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('WSR comparison at same weights — no optimization of mixtures')
    fig.tight_layout()
    fig.savefig(d/'midpoint_wsr_comparison.png',dpi=220,bbox_inches='tight')
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--schemes',nargs='+',choices=tuple(SCHEME_ANCHORS),default=['proposed'])
    ap.add_argument('--grid-size',type=int,default=21,help='Points along each feasible resource path')
    ap.add_argument('--comm-min',type=float,default=10.,help='Threshold for a middle candidate')
    ap.add_argument('--sensing-min',type=float,default=10.,help='Threshold for a middle candidate')
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--allow-replay-drift',action='store_true')
    args=ap.parse_args()
    if args.grid_size<3 or args.grid_size>1001:
        ap.error('--grid-size must be between 3 and 1001')
    original=read_csv(SOURCE_CSV)
    base_cfg=load_config(CONFIG)
    exp=load_stage8_experiment_config(CONFIG)
    if base_cfg.k_dl!=5 or base_cfg.j_ul!=5 or base_cfg.l_target!=5:
        raise RuntimeError('The diagnostic WSR formula assumes K=J=L=5')
    noise_w=3.981071705534969e-15
    if not (np.allclose(base_cfg.sigma_dl2,noise_w,rtol=1e-9,atol=0)
            and np.isclose(base_cfg.sigma_bs2,noise_w,rtol=1e-9,atol=0)):
        raise RuntimeError('Expected DL and BS noise both -114 dBm. Wrong current YAML!')
    fingerprint=hash_inputs()
    scene_seed=int(exp.scene_seed_start)
    # scene_seed comes from tradeoff config, not a randomly selected MC scene
    import yaml
    scene_seed=int(yaml.safe_load(CONFIG.read_text(encoding='utf-8'))['tradeoff']['scene_seed'])
    scene=make_scene_config(base_cfg,scene_seed,exp.position_jitter,
                            randomize_fading=exp.fading_enabled,
                            fading_model=exp.fading_model)
    print(f'Config: {CONFIG}\nCSV: {SOURCE_CSV}\nScene seed={scene_seed}, '
          f'DL noise={base_cfg.sigma_dl2[0]:.6g} W, BS noise={base_cfg.sigma_bs2:.6g} W')
    print(f'Schemes: {args.schemes}; {args.grid_size} samples per frozen geometry; '
          'no target-weight optimizer calls')
    for scheme in args.schemes:
        for w in SCHEME_ANCHORS[scheme]:
            row=get_expected(original,scheme,w)
            print(f'  anchor {scheme} w={w}: seed={row["best_seed"]}, '
                  f'WSR={float(row["weighted_sum_rate"]):.6f}')
    if args.dry_run:
        print('Dry run passed. No optimization performed.')
        return
    OUTPUT.mkdir(parents=True,exist_ok=True)
    anchor_rows=[]
    midpoint_rows=[]
    alpha_values=np.linspace(0.,1.,args.grid_size)
    for scheme in args.schemes:
        a,b=SCHEME_ANCHORS[scheme]
        low,ra=ensure_anchor(scheme,a,original,scene,exp,fingerprint,args.allow_replay_drift)
        high,rb=ensure_anchor(scheme,b,original,scene,exp,fingerprint,args.allow_replay_drift)
        anchor_rows.extend((ra,rb))
        midpoint_rows.extend(generate_midpoints(scheme,low,high,original,scene,
                        exp,alpha_values,args.comm_min,args.sensing_min))
    score_rows=score_candidates(midpoint_rows,original,args.schemes)
    write_csv(OUTPUT/'anchor_replay.csv',anchor_rows,
              ('scheme','source_weight','seed','old_wsr','replayed_wsr',
               'replay_abs_diff','replay_drift','sum_rate_comm','sum_rate_sensing','source'))
    write_csv(OUTPUT/'feasible_midpoints.csv',midpoint_rows,FIELDS)
    write_csv(OUTPUT/'wsr_at_weights.csv',score_rows,SCORE_FIELDS)
    make_plots(original,midpoint_rows,score_rows,args.schemes)
    for scheme in args.schemes:
        points=[x for x in midpoint_rows if x['scheme']==scheme]
        inside=[x for x in points if x['middle_window']]
        novel=[x for x in inside if not x['dominated_by_existing_cold20']]
        print(f'[{scheme}] tested {len(points)} feasible resource states; '
              f'R_comm>={args.comm_min:g} and R_sens>={args.sensing_min:g}: '
              f'{len(inside)}; not dominated by 20 stored points: {len(novel)}.')
    print(f'Finished. Files: {OUTPUT}')
    print('IMPORTANT: feasible != WSR-optimal. A poor mixture does NOT prove '
          'intermediate optimal solutions do not exist.')

if __name__=='__main__':
    main()
