"""Plot and summarize ordinary FP-aware Proposed; no accelerated data used."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent/'original_fpaware_no_extrapolation'
OUT=ROOT/'comparison_400'
OUT.mkdir(exist_ok=True)
data={}
for seed in [8003,8006]:
    for tol in [1e-4,1e-5,1e-6]:
        path=ROOT/f's{seed}_p1_outer1e-4_inner{tol:g}.json'
        if not path.exists():continue
        d=json.loads(path.read_text(encoding='utf-8'))
        assert not d['extrapolation']
        if tol==1e-4:
            if d['iterations']<400:continue
            d['original_recorded_iterations']=d['iterations']
            d['evaluation_note']='Evaluate saved uninterrupted trajectory at user-requested budget 400; raw record retained.'
            d['history']=d['history'][:401]
            d['rows']=d['rows'][:400]
            d.update(iterations=400,max_outer=400,finished=True,converged=False,
                     stop_reason='evaluation_budget_400',seconds=d['rows'][-1]['seconds'],
                     final_wsr=d['history'][-1],exact_resource_inheritance_checks=399)
            assert not any(abs(d['rows'][i]['increment'])<=1e-4 and abs(d['rows'][i-1]['increment'])<=1e-4 for i in range(1,400))
            checkpoint=ROOT/f's{seed}_p1_outer1e-4_inner{tol:g}_checkpoint.npz'
            with np.load(checkpoint) as a:
                assert len(a['wsr_history'])==401
                assert np.array_equal(a['wsr_history'],d['history'])
        elif not d['finished']:continue
        data[seed,tol]=d
        (OUT/path.name).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def plot(tolerances,name,xmax=None):
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,seed in zip(axes,[8003,8006]):
        for tol in tolerances:
            if (seed,tol) not in data:continue
            d=data[seed,tol];h=d['history'];n=d['iterations']
            label=f'Inner {tol:.0e}: '+(f'converged at {n}' if d['converged'] else f'{n}, not converged')
            ax.plot(range(len(h)),h,lw=1.8,label=label)
            ax.plot(n,h[-1],marker='o',ms=5)
        ax.set(title=f'Scene {seed}, 30 dBm',xlabel='Outer iteration',ylabel='Weighted sum rate (bit/s/Hz)')
        ax.grid(alpha=.25);ax.legend(fontsize=8)
        if xmax:ax.set_xlim(0,xmax)
    fig.suptitle('Original Proposed (FP-aware-2); outer absolute tolerance 1e-4, two consecutive rounds')
    fig.savefig(OUT/(name+'.png'),dpi=200)
    fig.savefig(OUT/(name+'.pdf'))
    plt.close(fig)

plot([1e-5],'original_proposed_outer4_inner5')
plot([1e-4,1e-5],'original_proposed_44_vs_45',400)
if all((s,1e-6) in data for s in [8003,8006]):
    plot([1e-4,1e-5,1e-6],'original_proposed_all_tolerances',400)
    plot([1e-5,1e-6],'original_proposed_45_vs_46')
loose={}
for seed in [8003,8006]:
    path=ROOT/f's{seed}_p1_outer0.001_inner1e-05.json'
    if path.exists():
        d=json.loads(path.read_text(encoding='utf-8'))
        if d['finished']:
            assert not d['extrapolation']
            loose[seed]=d
            (OUT/path.name).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
if len(loose)==2:
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,seed in zip(axes,[8003,8006]):
        combos=[(1e-4,1e-4,data[seed,1e-4]),(1e-4,1e-5,data[seed,1e-5]),(1e-3,1e-5,loose[seed])]
        if (seed,1e-6) in data:combos.insert(2,(1e-4,1e-6,data[seed,1e-6]))
        for outer_tol,inner_tol,d in combos:
            n=d['iterations'];label=f'Outer {outer_tol:.0e}, inner {inner_tol:.0e}: {n}'+(' converged' if d['converged'] else ' capped')
            ax.plot(range(len(d['history'])),d['history'],lw=1.6,label=label)
            ax.plot(n,d['final_wsr'],'o',ms=4)
        ax.set(title=f'Scene {seed}, 30 dBm',xlabel='Outer iteration',ylabel='Weighted sum rate (bit/s/Hz)',xlim=(0,400))
        ax.grid(alpha=.25);ax.legend(fontsize=7)
    fig.suptitle('Original Proposed (FP-aware-2), ordinary FP; absolute tolerances')
    fig.savefig(OUT/'original_proposed_four_combinations.png',dpi=200)
    fig.savefig(OUT/'original_proposed_four_combinations.pdf')
    plt.close(fig)
lines=['# 原版 Proposed 容差复测','',
       '普通 FP，保留原有 FP-aware-2 候选位置筛选与资源继承；无外推加速。',
       '场景 8003、8006，30 dBm；每个场景的所有容差组合从相同的归档初态开始。',
       '外层为绝对增量、连续两轮；内层普通 FP 单步绝对增量容差。内层上限 1200。',
       '对照组按用户要求仅评估前 400 轮，采用原连续运行第 400 轮保存的轨迹和状态；原始日志保留。','',
       '|场景|外层容差|内层容差|外层轮数|收敛|耗时(s)|最终 WSR|',
       '|---|---|---|---|---|---|---|']
summary=[]
for d in sorted(list(data.values())+list(loose.values()),key=lambda d:(d['scene'],d['outer_tolerance'],d['inner_tolerance'])):
    lines.append(f"|{d['scene']}|{d['outer_tolerance']:g}|{d['inner_tolerance']:g}|{d['iterations']}|{d['converged']}|{d['seconds']:.2f}|{d['final_wsr']:.9f}|")
    summary.append({k:d[k] for k in ['scene','outer_tolerance','inner_tolerance','iterations','converged','seconds','final_wsr','initial_sha256']})
lines+=['','这些是两个场景的配对复测，不能代替全部场景的收敛统计。',
        '运行时间来自同机并行诊断运行，包含诊断开销，不能直接作为 HPC 加速比。',
        '外层达到停止判据不代表全局最优，也不代表所有历史内层调用均在上限前满足容差。']
(OUT/'复测结论.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
