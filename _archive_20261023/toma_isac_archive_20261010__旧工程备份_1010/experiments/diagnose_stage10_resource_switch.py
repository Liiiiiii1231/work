"""Read-only Stage 10 diagnostics: why Proposed / B6 change modes near omega_C=0.5->0.6.

Place this file in the project's experiments/ directory. Does not run an optimizer
or mutate any existing Stage 10 results. Reads stage10_tradeoff.csv and referenced
NPZ snapshots saved by run_stage10_formal.py.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from stage8 import load_stage8_experiment_config, make_scene_config
from state import GeometryState, ResourceState
from channels import build_channels
from metrics import compute_performance
from utils import check_resource_constraints, split_transmit_matrix
from experiments.run_stage10_tradeoff import load_tradeoff_config, make_group_weights
from dataclasses import replace

DEFAULT_RESULTS = ROOT / 'results' / 'stage10' / 'formal_14point_multiround'


def write_csv(filename: Path, rows: list[dict]):
    if not rows:
        return
    filename.parent.mkdir(parents=True, exist_ok=True)
    # Don't overwrite the user's source CSV: output directory is separate.
    with filename.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def safe_state_path(base: Path, original: str):
    parts = original.replace('\\', '/').split('/')
    if not original or any(p in ('', '.', '..') for p in parts) or Path(original).is_absolute():
        raise ValueError(f'Invalid state_file path: {original!r}')
    path = base.joinpath(*parts)
    if not path.is_file():
        raise FileNotFoundError(f'Missing snapshot: {path}\nKeep the warm_states/ folder from the formal run.')
    return path


def load_snapshot(file: Path):
    with np.load(file, allow_pickle=False) as data:
        names = ('endpoints', 'tx_positions', 'rx_positions', 'q_matrix', 'q_ul', 'b_ul', 'u_s')
        missing = [name for name in names if name not in data.files]
        if missing:
            raise KeyError(f'{file}: missing arrays: {missing}')
        geo = GeometryState(*(np.array(data[name], copy=True) for name in names[:3]))
        r = ResourceState(*(np.array(data[name], copy=True) for name in names[3:]))
    return geo, r


def pow2(x):
    return float(np.real(np.vdot(x, x)))


def as_db(x):
    return 10.0 * np.log10(x) if x > 0 else float('-inf')


def analyze_state(csv_row, base: Path, scene_cfg):
    scheme = csv_row['scheme']
    w = float(csv_row['omega_comm'])
    wd, wu, ws = make_group_weights(scene_cfg, w)
    cfg = replace(scene_cfg, weight_dl=wd, weight_ul=wu, weight_s=ws)
    cfg.validate()

    snapshot = safe_state_path(base, csv_row['state_file'])
    geo, r = load_snapshot(snapshot)
    check_resource_constraints(r, cfg)
    ch = build_channels(geo, cfg)
    perf = compute_performance(ch, r, cfg)

    rates = (float(np.sum(perf.rate_dl)), float(np.sum(perf.rate_ul)), float(np.sum(perf.rate_s)))
    expected = tuple(float(csv_row[name]) for name in ('sum_rate_dl', 'sum_rate_ul', 'sum_rate_sensing'))
    if not np.allclose(rates, expected, atol=2e-5, rtol=1e-5):
        raise RuntimeError(f'NPZ/CSV rate mismatch: scheme={scheme}, w={w} reconstructed={rates}, CSV={expected}. '
                           'Check that the configuration and states are from the same run.')
    if not np.isclose(perf.weighted_sum_rate, float(csv_row['weighted_sum_rate']), rtol=1e-5, atol=2e-5):
        raise RuntimeError(f'NPZ/CSV WSR mismatch: {scheme} w={w}')

    W, V = split_transmit_matrix(r.q_matrix, cfg)
    p_w = np.sum(abs(W)**2, axis=0).real
    p_v = np.sum(abs(V)**2, axis=0).real
    p_ul = np.square(r.q_ul).real
    p_dl_total = pow2(r.q_matrix)
    p_comm, p_sens = float(p_w.sum()), float(p_v.sum())
    uplink_on = int(np.count_nonzero(p_ul > 1e-15))
    ul_exact_zero = int(np.count_nonzero(r.q_ul == 0.0))

    summary = dict(scheme=scheme, omega_comm=w,
        sum_rate_dl=rates[0], sum_rate_ul=rates[1], sum_rate_comm=rates[0]+rates[1],
        sum_rate_sensing=rates[2], weighted_sum_rate=float(perf.weighted_sum_rate),
        wsr_comm_contribution=float(np.dot(cfg.weight_dl, perf.rate_dl) + np.dot(cfg.weight_ul, perf.rate_ul)),
        wsr_sensing_contribution=float(np.dot(cfg.weight_s, perf.rate_s)),
        bs_total_power_w=p_dl_total, bs_communication_beams_w=p_comm,
        bs_sensing_beams_w=p_sens, bs_budget_w=float(cfg.p_dl_max),
        bs_comm_share=p_comm/max(p_dl_total,1e-30), bs_sensing_share=p_sens/max(p_dl_total,1e-30),
        ul_total_power_w=float(p_ul.sum()), ul_power_max_total_w=float(cfg.p_ul_max.sum()),
        ul_positive_users=uplink_on, ul_exact_zero_amplitudes=ul_exact_zero,
        ul_q_amplitudes=';'.join(f'{v:.12g}' for v in r.q_ul),
        ul_user_powers_w=';'.join(f'{v:.12g}' for v in p_ul),
        w_beam_powers_w=';'.join(f'{v:.12g}' for v in p_w),
        v_beam_powers_w=';'.join(f'{v:.12g}' for v in p_v),
        state_file=str(snapshot.relative_to(base)))

    detail_rows=[]
    R = r.q_matrix @ r.q_matrix.conj().T
    for j in range(cfg.j_ul):
        b, f = r.b_ul[j], ch.f[j]
        ul_des = float(p_ul[j] * abs(np.vdot(b,f))**2)
        other_ul = float(sum(p_ul[i] * abs(np.vdot(b,ch.f[i]))**2 for i in range(cfg.j_ul) if i != j))
        echo = float(sum(max(0.0,np.vdot(b, ch.g[l] @ R @ ch.g[l].conj().T @ b).real) for l in range(cfg.l_target)))
        rsi = float(max(0.0,np.vdot(b, ch.h_rsi @ R @ ch.h_rsi.conj().T @ b).real))
        noise=float(cfg.sigma_bs2)
        denom=other_ul+echo+rsi+noise
        recon=ul_des/denom
        if not np.isclose(recon, perf.gamma_ul[j], rtol=1e-5, atol=1e-7):
            raise RuntimeError(f'UL SINR decomposition mismatch j={j} {recon} != {perf.gamma_ul[j]}')
        detail_rows.append(dict(scheme=scheme, omega_comm=w, group='UL',index=j+1,
            amplitude=float(r.q_ul[j]), power_w=float(p_ul[j]), rate=float(perf.rate_ul[j]),
            sinr_linear=float(perf.gamma_ul[j]), sinr_db=as_db(perf.gamma_ul[j]),
            desired_power=ul_des, other_user_interference=other_ul,
            target_echo_interference=echo, residual_si_interference=rsi,
            uplink_interference=0.0, noise=noise, total_denominator=denom,
            ul_interference_share_of_denominator=0.0))
    for l in range(cfg.l_target):
        u, g = r.u_s[l], ch.g[l]
        s_des = float(np.linalg.norm(u.conj() @ g @ r.q_matrix)**2)
        ul_int=float(sum(p_ul[j] * abs(np.vdot(u, ch.f[j]))**2 for j in range(cfg.j_ul)))
        other_echo=float(sum(max(0.0,np.vdot(u, ch.g[i] @ R @ ch.g[i].conj().T @ u).real)
                              for i in range(cfg.l_target) if i != l))
        rsi=float(max(0.0,np.vdot(u,ch.h_rsi @ R @ ch.h_rsi.conj().T @ u).real))
        noise=float(cfg.sigma_bs2)
        den=ul_int+other_echo+rsi+noise
        recon=s_des/den
        if not np.isclose(recon,perf.gamma_s[l],rtol=1e-5,atol=1e-7):
            raise RuntimeError(f'Sensing SINR mismatch l={l} {recon} != {perf.gamma_s[l]}')
        detail_rows.append(dict(scheme=scheme, omega_comm=w, group='Sensing',index=l+1,
            amplitude='',power_w='',rate=float(perf.rate_s[l]),sinr_linear=float(perf.gamma_s[l]),
            sinr_db=as_db(perf.gamma_s[l]),desired_power=s_des,
            other_user_interference=0.0,target_echo_interference=other_echo,
            residual_si_interference=rsi,uplink_interference=ul_int,noise=noise,
            total_denominator=den,ul_interference_share_of_denominator=ul_int/den if den>0 else 0.0))
    return summary, detail_rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=DEFAULT_RESULTS)
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--schemes', nargs='+', default=['proposed','b6'],choices=['proposed','b6','b2','b3'])
    p.add_argument('--weights', nargs='+', type=float,default=[0.5,0.6])
    args=p.parse_args()

    base=args.results.resolve()
    if not base.is_dir():
        p.error(f'Expected results directory: {base}')
    csv_file=base/'stage10_tradeoff.csv'
    if not csv_file.is_file():
        p.error(f'Missing file: {csv_file}')
    with csv_file.open(encoding='utf-8-sig',newline='') as f:
        all_rows=list(csv.DictReader(f))

    yml=ROOT/'configs'/'stage10_tradeoff.yaml'
    conf=load_config(yml)
    exp=load_stage8_experiment_config(yml)
    trade=load_tradeoff_config(yml)
    scene=make_scene_config(conf,scene_seed=trade.scene_seed,
        position_jitter=exp.position_jitter,randomize_fading=exp.fading_enabled,
        fading_model=exp.fading_model)

    summaries,details=[],[]
    for sch in args.schemes:
        for w in args.weights:
            matches=[r for r in all_rows if r['scheme']==sch and abs(float(r['omega_comm'])-w)<=1e-12]
            if len(matches)!=1:
                raise RuntimeError(f'Expected one row for {sch} weight {w}, got {len(matches)}')
            summary, detail=analyze_state(matches[0],base,scene)
            summaries.append(summary);details.extend(detail)
            print(f'[{sch} w={w:g}] BS transmit power={summary["bs_total_power_w"]:.6g} W '
                  f'(W beams={summary["bs_communication_beams_w"]:.6g}, V beams={summary["bs_sensing_beams_w"]:.6g}); '
                  f'UL power={summary["ul_total_power_w"]:.6g} W, '
                  f'UL rate={summary["sum_rate_ul"]:.4f}, sensing rate={summary["sum_rate_sensing"]:.4f}, '
                  f'UL zero amplitudes={summary["ul_exact_zero_amplitudes"]}/{scene.j_ul}')
    out=(args.output if args.output is not None else base/'resource_switch_diagnosis').resolve()
    write_csv(out/'resource_summary.csv',summaries)
    write_csv(out/'per_user_target_sinr.csv',details)
    print(f'\n[OK] Read-only analysis saved: {out}')
    print('  resource_summary.csv: beam, BS, and uplink powers; WSR contributions')
    print('  per_user_target_sinr.csv: each UL user and sensing target SINR, plus interference breakdown')

if __name__=='__main__':
    main()
