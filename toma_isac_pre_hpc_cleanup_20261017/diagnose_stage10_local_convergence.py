"""Stage 10 定点诊断：B6 早停 + Proposed 冷启动解质量。

放在项目根目录运行，不覆盖正式 Stage 10 结果，不修改求解器。
默认复现 B6(omega=0.3, seed=2026)。--compare 另测 B6(0.2) 和 Proposed(0.3/0.5)。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _relative_change(new: float, old: float) -> float:
    return abs(float(new) - float(old)) / max(1., abs(float(old)))


def _weights(base, omega: float):
    # 与 Stage 10 的 make_group_weights 完全一致：通信权重对 DL/UL 两组平分。
    assert 0.0 < omega < 1.0
    cfg = replace(
        base,
        weight_dl=np.full(base.k_dl, omega / (2 * base.k_dl)),
        weight_ul=np.full(base.j_ul, omega / (2 * base.j_ul)),
        weight_s=np.full(base.l_target, (1 - omega) / base.l_target),
    )
    cfg.validate()
    return cfg


def _numeric_summary(res, cfg, name, omega, seed, runtime):
    from utils import check_resource_constraints

    check_resource_constraints(res.resources, cfg)
    p = res.performance
    q = res.resources
    dl = float(np.sum(p.rate_dl))
    ul = float(np.sum(p.rate_ul))
    sensing = float(np.sum(p.rate_s))
    wsr = float(p.weighted_sum_rate)
    expected = omega / 2 * dl / cfg.k_dl + omega / 2 * ul / cfg.j_ul + (1-omega) * sensing / cfg.l_target
    if not np.isclose(wsr, expected, rtol=1e-9, atol=1e-10):
        raise AssertionError(f"WSR mismatch: {wsr} vs {expected}")
    return {
        'scheme': name,
        'omega_comm': omega,
        'seed': int(seed),
        'converged': bool(res.converged),
        'stop_reason': str(res.stop_reason),
        'iterations': int(res.iterations if name=='b6' else res.outer_iterations),
        'runtime_s': round(runtime, 4),
        'sum_rate_dl': dl, 'sum_rate_ul': ul, 'sum_rate_comm': dl + ul,
        'sum_rate_sensing': sensing, 'weighted_sum_rate': wsr,
        'tx_power_W': float(np.linalg.norm(q.q_matrix, 'fro') ** 2),
        'ul_power_W_each': np.square(np.asarray(q.q_ul, dtype=float)).tolist(),
        'zero_ul_count': int(np.sum(np.abs(q.q_ul) < 1e-12)),
    }


def _history_rows(history, tol, inner_info=None, move_info=None):
    inner_info = inner_info or []
    move_info = move_info or []
    rows = []
    consecutive = 0
    for i, value in enumerate(history):
        d = float('nan') if i == 0 else _relative_change(value, history[i-1])
        consecutive = (consecutive+1 if d <= tol else 0) if i else 0
        inner = inner_info[i-1] if 0 < i <= len(inner_info) else {}
        moves = move_info[i-1] if 0 < i <= len(move_info) else {}
        rows.append({
            'outer_iteration': i,
            'wsr': float(value),
            'relative_change': d,
            'consecutive_small_change_count': consecutive,
            'passes_two_consecutive_tolerance': consecutive >= 2,
            'fp_inner_iterations': inner.get('fp_inner_iterations', ''),
            'fp_wsr_before': inner.get('fp_wsr_before', ''),
            'fp_wsr_after': inner.get('fp_wsr_after', ''),
            'fp_wsr_gain': inner.get('fp_wsr_gain', ''),
            'fp_last_relative_change': inner.get('fp_last_relative_change', ''),
            'accepted_ma_moves': moves.get('accepted_ma_moves', ''),
            'max_tx_move_m': moves.get('max_tx_move_m', ''),
            'max_rx_move_m': moves.get('max_rx_move_m', ''),
        })
    return rows


def _write_csv(path: Path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _b6_run(cfg, exp, seed, out):
    import ma_baseline

    inner_events = []
    move_events = []
    original_inner = ma_baseline.solve_inner_problem
    original_move = ma_baseline._update_ma_positions_once

    def recorded_inner(channels, resources, cfg_inner, verbose=False):
        updated, hist = original_inner(channels, resources, cfg_inner, verbose=verbose)
        inner_events.append({
            'fp_inner_iterations': len(hist)-1,
            'fp_wsr_before': float(hist[0]),
            'fp_wsr_after': float(hist[-1]),
            'fp_wsr_gain': float(hist[-1] - hist[0]),
            'fp_last_relative_change': (_relative_change(hist[-1], hist[-2]) if len(hist)>1 else float('nan')),
        })
        return updated, hist

    def recorded_move(geometry, resources, cfg_move, *args, **kwargs):
        before_tx = geometry.tx_positions.copy()
        before_rx = geometry.rx_positions.copy()
        result = original_move(geometry, resources, cfg_move, *args, **kwargs)
        new_geo = result[0]
        move_events.append({
            'accepted_ma_moves': int(result[3]),
            'max_tx_move_m': float(np.max(np.linalg.norm(new_geo.tx_positions-before_tx, axis=1))),
            'max_rx_move_m': float(np.max(np.linalg.norm(new_geo.rx_positions-before_rx, axis=1))),
        })
        return result

    t0 = perf_counter()
    with patch.object(ma_baseline, 'solve_inner_problem', recorded_inner), patch.object(ma_baseline, '_update_ma_positions_once', recorded_move):
        result = ma_baseline.run_ma_fp(
            cfg, rng=np.random.default_rng(seed), verbose=False,
            region_size=exp.ma_region_size,
            min_spacing=exp.ma_min_spacing,
            initial_step=exp.ma_initial_step,
            randomize_initial_geometry=(seed != int(exp.fixed_upa_seed)),
            initial_jitter_fraction=exp.ma_initial_jitter_fraction,
        )
    elapsed = perf_counter()-t0
    summary = _numeric_summary(result, cfg, 'b6', _infer_omega(cfg), seed, elapsed)
    rows = _history_rows(result.wsr_history, cfg.tol_outer, inner_events, move_events)
    prefix = f'b6_omega{_infer_omega(cfg):g}_seed{seed}'
    _write_csv(out / f'{prefix}_outer_history.csv', rows)
    with (out / f'{prefix}_summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'\n[B6] omega={_infer_omega(cfg):g}, seed={seed}, iterations={result.iterations}, stop={result.stop_reason}, WSR={summary["weighted_sum_rate"]:.9f}, time={elapsed:.1f}s', flush=True)
    for r in rows[-min(8,len(rows)):]:
        print('  outer={:3} WSR={:.9f} change={:.3e} small={} FPiters={} FPgain={} accepted={}'.format(
            r['outer_iteration'], r['wsr'], r['relative_change'], r['consecutive_small_change_count'],
            r['fp_inner_iterations'], r['fp_wsr_gain'], r['accepted_ma_moves']), flush=True)
    return result, summary, rows


def _infer_omega(cfg):
    return float(np.sum(cfg.weight_dl) + np.sum(cfg.weight_ul))


def _proposed_run(cfg, seed, out):
    from algorithm import run_joint_algorithm
    t0 = perf_counter()
    result = run_joint_algorithm(cfg, rng=np.random.default_rng(seed), verbose=False,
                                 position_update_mode='fp_aware', fp_aware_candidate_iters=2)
    elapsed = perf_counter() - t0
    omega = _infer_omega(cfg)
    summary = _numeric_summary(result, cfg, 'proposed', omega, seed, elapsed)
    extra = []
    for hist in result.inner_wsr_histories:
        extra.append({'fp_inner_iterations': len(hist)-1,
                      'fp_wsr_before': float(hist[0]), 'fp_wsr_after': float(hist[-1]),
                      'fp_wsr_gain': float(hist[-1]-hist[0]),
                      'fp_last_relative_change': _relative_change(hist[-1],hist[-2]) if len(hist)>1 else float('nan')})
    rows = _history_rows(result.joint_wsr_history, cfg.tol_outer, extra)
    # 只有与 outer iteration 对应时才追加接受的 MA/ToMA 位置更新计数。
    for i in range(1,len(rows)):
        j=i-1
        if j < len(result.accepted_position_counts):
            rows[i]['accepted_ma_moves'] = int(result.accepted_position_counts[j])
    prefix=f'proposed_omega{omega:g}_seed{seed}'
    _write_csv(out / f'{prefix}_outer_history.csv', rows)
    with (out / f'{prefix}_summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'\n[Proposed] omega={omega:g}, seed={seed}, iterations={result.outer_iterations}, stop={result.stop_reason}, WSR={summary["weighted_sum_rate"]:.9f}, time={elapsed:.1f}s',flush=True)
    return result, summary, rows


def _make_plot(traces, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9,5.5))
    for label, rows in traces:
        x = [r['outer_iteration'] for r in rows]
        y = [r['wsr'] for r in rows]
        ax.plot(x,y,marker='o', markersize=3, linewidth=1.5, label=label)
    ax.set_xlabel('Outer iteration')
    ax.set_ylabel('True WSR')
    ax.set_title('Stage 10 local convergence diagnosis')
    ax.grid(alpha=.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / 'diagnostic_wsr_history.png',dpi=190)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description='针对性排查 B6 三次早停，以及 Proposed 不同权重点的局部解质量')
    ap.add_argument('--config', default='configs/stage10_tradeoff.yaml', help='相对项目根目录的 YAML 路径')
    ap.add_argument('--tol',type=float,default=1e-4,help='仅本次诊断内存覆盖 inner/outer 容差，不修改 YAML')
    ap.add_argument('--compare',action='store_true',help='另测 B6 omega=0.2 及 Proposed omega=0.3/0.5')
    ap.add_argument('--all-b6-starts',action='store_true',help='B6 omega=0.3 测全部三个 seed（否则只测 CSV 最佳 seed 2026）')
    ap.add_argument('--dry-run',action='store_true',help='仅验证场景和参数，不运行优化')
    args=ap.parse_args()
    if not (0<args.tol<1):
        ap.error('--tol 应在 (0,1) 内')

    from config import load_config
    from stage8 import load_stage8_experiment_config, make_scene_config, make_ma_start_seeds
    cfg_path = (ROOT/args.config).resolve()
    if not cfg_path.is_file():
        ap.error(f'找不到配置: {cfg_path}')
    base=load_config(cfg_path)
    exp=load_stage8_experiment_config(cfg_path)
    import yaml
    with cfg_path.open(encoding='utf-8') as f:
        raw=yaml.safe_load(f)
    scene_seed=int(raw.get('tradeoff', {}).get('scene_seed', 8001))
    if exp.fading_enabled:
        raise RuntimeError('正式 Stage 10 场景应关闭随机衰落，当前 fading.enabled=True')
    base=make_scene_config(base,scene_seed,exp.position_jitter,
                           randomize_fading=exp.fading_enabled,fading_model=exp.fading_model)
    print(f'场景 scene_seed={scene_seed}, K={base.k_dl}, J={base.j_ul}, L={base.l_target}',flush=True)
    print(f'噪声 sigma_dl2={base.sigma_dl2}, sigma_bs2={base.sigma_bs2:.6e} W',flush=True)
    print(f'散射系数 xi_s={base.xi_s}',flush=True)
    print(f'YAML tol_inner={base.tol_inner:g}, tol_outer={base.tol_outer:g}; 诊断统一使用 {args.tol:g}',flush=True)
    print('只修改本次临时 cfg 的权重和容差，不写回 YAML 或原始结果',flush=True)
    starts = make_ma_start_seeds(scene_seed,exp.fixed_upa_seed,exp.ma_n_starts)
    print(f'B6 seeds={starts}',flush=True)
    if args.dry_run:
        print('Dry run passed, no numerical optimization executed.')
        return

    # 使用项目中实际的求解器，覆盖的只是本次实验的 cfg 容差。
    base=replace(base,tol_inner=args.tol,tol_outer=args.tol)
    target_cfg=_weights(base,0.3)
    timestamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    out=ROOT/'results'/'stage10'/'local_convergence_diagnosis'/timestamp
    out.mkdir(parents=True,exist_ok=True)
    traces=[]
    summaries=[]

    seeds_to_run=starts if args.all_b6_starts else (int(exp.fixed_upa_seed),)
    for seed in seeds_to_run:
        r,s,rows=_b6_run(target_cfg,exp,int(seed),out)
        summaries.append(s)
        traces.append((f'B6 w=0.3 seed={seed}',rows))
    if args.compare:
        # CSV 中 omega=0.2 的 B6 最优 seed；同场景、同参数、同一个具体的冷启动解。
        source_seed=int(starts[-1])  # CSV 的最佳 seed 通常是第 3 个；从列表中明确输出
        src_cfg=_weights(base,0.2)
        old_r, old_s, rows=_b6_run(src_cfg,exp,source_seed,out)
        summaries.append(old_s)
        traces.append((f'B6 w=0.2 seed={source_seed}',rows))
        from metrics import compute_performance
        at_03=compute_performance(old_r.channels,old_r.resources,target_cfg)
        best03=max((s['weighted_sum_rate'] for s in summaries if s['scheme']=='b6' and abs(s['omega_comm']-0.3)<1e-10),default=float('nan'))
        cross=float(at_03.weighted_sum_rate)
        print(f'\n[跨权重可行候选检查] B6(0.2) 最终状态重新按 omega=0.3 计分: {cross:.9f}, B6(0.3) 当前最佳 WSR: {best03:.9f}',flush=True)
        print('若前者更高，说明当前 B6(0.3) 独立冷启动没有找到已有的更优可行解；不等于算法实现错误。',flush=True)
        with (out/'cross_weight_feasible_candidate.json').open('w',encoding='utf-8') as f:
            json.dump({'source_omega':.2,'target_omega':.3,'source_seed':source_seed,'source_state_wsr_at_target':cross,'best_cold_wsr_at_target':best03,'gap':cross-best03},f,indent=2)

        for omega in (.3,.5):
            pr,ps,trace=_proposed_run(_weights(base,omega),2028,out)
            summaries.append(ps)
            traces.append((f'Proposed w={omega:g} seed=2028',trace))

    _write_csv(out/'diagnostic_summary.csv',summaries)
    _make_plot(traces,out)
    print(f'\n完成，输出目录：{out}',flush=True)
    print('优先查看 *_outer_history.csv 中的 relative_change / fp_wsr_gain / accepted_ma_moves，及 diagnostic_summary.csv',flush=True)

if __name__=='__main__':
    main()
