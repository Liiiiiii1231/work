from pathlib import Path
import json,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'ma_plain_vs_fpaware_o4_i5'
data={}
for scene in [8003,8006]:
    for mode in ['plain','aware']:
        d=json.loads((OUT/f'scene{scene}_{mode}.json').read_text())
        assert d['finished'] and not d['extrapolation']
        if d['converged']:assert max(abs(np.diff(d['history'][-3:])))<=1e-4
        data[scene,mode]=d
    assert data[scene,'plain']['initial_sha256']==data[scene,'aware']['initial_sha256']
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
summary=[]
lines=['# MA 普通位置搜索 vs FP-aware 位置搜索','',
       '两者主内层均为普通 FP，无外推加速。外层绝对1e-4连续两轮，内层绝对1e-5，上限外400/内1200。',
       '场景8003/8006、30dBm；共同第0号未筛选初态，每个场景两种方法的位置与资源逐字节一致。不是未来正式筛选优胜位置，也不是全场景统计。','',
       '|场景|MA+FP轮数|MA+FP-aware轮数|aware减plain|MA+FP最终WSR|MA+FP-aware最终WSR|',
       '|---|---|---|---|---|---|']
for ax,scene in zip(axes,[8003,8006]):
    plain=data[scene,'plain'];aware=data[scene,'aware']
    for mode,label in [('plain','MA + FP'),('aware','MA + FP-aware')]:
        d=data[scene,mode];h=d['history'];n=d['iterations']
        ax.plot(range(len(h)),h,lw=1.8,label=f'{label}: {n} rounds'+(' converged' if d['converged'] else ' capped'))
        ax.plot(n,h[-1],'o',ms=5)
    ax.set(title=f'Scene {scene}, 30 dBm',xlabel='Outer iteration',ylabel='Weighted sum rate (bit/s/Hz)')
    ax.legend(fontsize=8);ax.grid(alpha=.25)
    row=dict(scene=scene,plain_iterations=plain['iterations'],aware_iterations=aware['iterations'],
             plain_converged=plain['converged'],aware_converged=aware['converged'],
             difference_aware_minus_plain=aware['iterations']-plain['iterations'],
             plain_final_wsr=plain['final_wsr'],aware_final_wsr=aware['final_wsr'],
             wsr_gain_percent=100*(aware['final_wsr']/plain['final_wsr']-1),
             plain_seconds=plain['seconds'],aware_seconds=aware['seconds'])
    summary.append(row)
    lines.append(f"|{scene}|{plain['iterations']} ({plain['converged']})|{aware['iterations']} ({aware['converged']})|{row['difference_aware_minus_plain']}|{plain['final_wsr']:.9f}|{aware['final_wsr']:.9f}|")
fig.suptitle('Ordinary FP resource solver; absolute outer 1e-4 / inner 1e-5')
for suffix in ['png','pdf']:fig.savefig(OUT/f'ma_fp_vs_fpaware.{suffix}',dpi=200)
plt.close(fig)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
(OUT/'比较结论.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
with zipfile.ZipFile(OUT/'ma_comparison_results.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.iterdir():
        if p.suffix in ['.json','.md','.png','.pdf','.npz']:z.write(p,p.name)
    for name in ['compare_ma_plain_and_aware.py','summarize_ma_plain_and_aware.py']:z.write(ROOT/name,name)
print(json.dumps(summary,indent=2))
