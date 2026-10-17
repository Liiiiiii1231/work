"""Frozen 100-scene protocol and deterministic multi-start initialization."""
from pathlib import Path
from unittest.mock import patch
import numpy as np
import paper_figures_unified as engine

ROOT = Path(__file__).resolve().parents[1]
ANCHOR = engine.initial_pair
STARTS = {'proposed': 5, 'b6': 3, 'b2': 1, 'b3': 1}


def n_starts(task):
    return 1 if task['figure'] == 'tradeoff' else STARTS[task['scheme']]


def pair(cfg, protocol, scheme, start):
    """Start 0: uniform planar anchor; remaining starts: feasible perturbations.

    Geometry streams do not depend on scene, weight, or sweep value. All weights
    use exactly the same anchor; channel-dependent MRT is rebuilt per scene.
    """
    from geometry import build_geometry
    from channels import build_channels
    from initialization import initialize_resources
    import ma_baseline as ma
    g, r = ANCHOR(cfg, protocol, scheme)
    if start == 0:
        return g, r
    seeds = [2026, 2027, 2028, 2029] if scheme == 'proposed' else [2026, 2027]
    rng = np.random.default_rng(seeds[start-1])
    if scheme == 'proposed':
        anchor = g.endpoints/cfg.cable_length
        for _ in range(cfg.max_geometry_trials):
            directions = anchor + .15*rng.normal(size=anchor.shape)
            endpoints = cfg.cable_length*directions/np.linalg.norm(directions, axis=1, keepdims=True)
            try:
                g = build_geometry(endpoints, cfg, check_constraints=True)
                break
            except ValueError:
                continue
        else:
            raise RuntimeError('No feasible perturbed ToMA geometry')
    elif scheme == 'b6':
        p = protocol['protocol']
        g = ma._randomize_initial_ma_geometry(g, np.mean(g.tx_positions, axis=0),
            np.mean(g.rx_positions, axis=0), p['ma_region_size_m'], p['ma_min_spacing_m'], .5, cfg, rng)
    else:
        raise ValueError('Fixed arrays have one geometry start')
    r = initialize_resources(g, build_channels(g, cfg), cfg,
                             np.random.default_rng(protocol['protocol']['resource_seed']))
    r.q_ul[:] = np.sqrt(protocol['protocol']['initial_ul_power_W'])
    return g, r


def run_job(job):
    out, protocol, task, start, max_calls = job
    with patch.object(engine, 'initial_pair', lambda cfg, p, sc: pair(cfg, p, sc, start)):
        row = engine.run_task(str(ROOT), str(Path(out)/f'candidate_{start}'),
                              protocol, task, max_calls)
    return dict(row, start_id=start)


def read_verified(out, protocol, task, start):
    folder = Path(out)/f'candidate_{start}'/'tasks'/engine.task_key(task)
    record = folder/'record.json'
    if not record.exists():
        return None
    d = engine.read_json(record)
    cfg = engine.config_for_task(ROOT, protocol, task)
    if any(d['row'].get(k) != v for k, v in task.items()):
        raise RuntimeError('Checkpoint task mismatch')
    if d['cfg_sha256'] != engine.config_digest(cfg):
        raise RuntimeError('Checkpoint config mismatch')
    if d['initial_sha256'] != engine.sha(folder/'initial.npz') or d['state_sha256'] != engine.sha(folder/d['state_file']):
        raise RuntimeError('Checkpoint checksum mismatch')
    g0, r0, _ = engine.load_state(folder/'initial.npz')
    expected, resources = pair(cfg, protocol, task['scheme'], start)
    for name in ('endpoints', 'tx_positions', 'rx_positions'):
        if not np.allclose(getattr(g0, name), getattr(expected, name), rtol=0, atol=1e-12):
            raise RuntimeError('Checkpoint initial geometry mismatch')
    for name in ('q_matrix', 'q_ul', 'b_ul', 'u_s'):
        if not np.allclose(getattr(r0, name), getattr(resources, name), rtol=1e-12, atol=1e-12):
            raise RuntimeError('Checkpoint initial resource mismatch')
    if not np.allclose(abs(r0.q_ul)**2, .025, rtol=0, atol=1e-12):
        raise RuntimeError('Initial UL power must be 25 mW')
    g, r, h = engine.load_state(folder/d['state_file'])
    engine.check_state(g, r, cfg, protocol, task['scheme'], g0)
    if abs(engine.metrics_values(g, r, cfg)['weighted_sum_rate']-d['row']['weighted_sum_rate']) > 1e-8:
        raise RuntimeError('Checkpoint metrics mismatch')
    if d['row']['converged']:
        required = 2 if task['scheme'] in ('proposed', 'b6') else 1
        delta = np.diff(h)
        if len(delta) < required or not np.all(np.isfinite(h)) or np.min(delta) < -1e-8*max(1., max(abs(x) for x in h)) or max(abs(delta[-required:])) > 1e-4:
            raise RuntimeError('Absolute convergence verification failed')
    return dict(d['row'], start_id=start)


def plot_convergence(out, protocol, tasks, winners):
    plt = engine.plotting()
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    histories = []
    for i, power in enumerate(protocol['convergence']['p_dl_W']):
        group = [r for r in winners if r['parameter_value'] == power]
        hs = []
        for row in group:
            task = {k:row[k] for k in ('figure','parameter_value','scene_seed','scheme')}
            folder = Path(out)/f"candidate_{row['start_id']}"/'tasks'/engine.task_key(task)
            d = engine.read_json(folder/'record.json')
            _, _, h = engine.load_state(folder/d['state_file'])
            hs.append(h)
            histories.extend(dict(p_dl_W=power,scene_seed=row['scene_seed'],start_id=row['start_id'],iteration=j,weighted_sum_rate=v) for j,v in enumerate(h))
        # A converged scene keeps its terminal value when other scenes continue.
        matrix = np.asarray([np.pad(h,(0,max(map(len,hs))-len(h)),mode='edge') for h in hs])
        ax.plot(matrix.mean(axis=0),color=engine.POWER_COLORS[i],linestyle=engine.POWER_LINESTYLES[i],label=f'{power:g} W')
    ax.set(xlabel='Joint AO iteration',ylabel='Mean WSR (bit/s/Hz)',title=f"{len(group)} Monte Carlo scenes | best start per scene")
    ax.legend(title='Maximum DL power')
    fig.text(.1,.015,'Stopped scenes retain their terminal WSR when computing the mean trajectory.',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1))
    engine.save_figure(fig,out,'01_proposed_convergence')
    plt.close(fig)
    engine.csv_write(Path(out)/'tables/convergence_histories.csv',histories,list(histories[0]))


def plot_tradeoff(out, protocol, tasks, rows):
    summary = []
    for w in protocol['tradeoff']['weights']:
        for sc in protocol['schemes']:
            group = [r for r in rows if r['parameter_value']==w and r['scheme']==sc]
            summary.append(dict(omega_comm=w,scheme=sc,n_scenes=len(group),
                mean_comm=float(np.mean([r['sum_rate_comm'] for r in group])),
                mean_sensing=float(np.mean([r['sum_rate_sensing'] for r in group]))))
    engine.csv_write(Path(out)/'tables/tradeoff_mean.csv',summary,list(summary[0]))
    plt = engine.plotting()
    fig,ax = plt.subplots(figsize=(8.4,6.1))
    for sc in protocol['schemes']:
        group = [r for r in summary if r['scheme']==sc]
        style = engine.STYLES[sc]
        ax.plot([r['mean_comm'] for r in group],[r['mean_sensing'] for r in group],
            color=style['color'],marker=style['marker'],label=style['label'])
    ax.set(xlabel='Mean communication rate (bit/s/Hz)',ylabel='Mean sensing rate (bit/s/Hz)',
           title=f"{group[0]['n_scenes']} Monte Carlo scenes | common initial geometry across weights")
    ax.legend()
    fig.tight_layout()
    engine.save_figure(fig,out,'05_communication_sensing_tradeoff')
    plt.close(fig)
