"""Three separate scientific plots from completed original-Proposed tests."""
from pathlib import Path
import json,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'original_fpaware_no_extrapolation'
OUT=DATA/'three_separate_figures'
OUT.mkdir(exist_ok=True)
cases=[('outer4_inner5','1e-4','1e-05',1e-4,1e-5),
       ('outer4_inner6','1e-4','1e-06',1e-4,1e-6),
       ('outer3_inner5','0.001','1e-05',1e-3,1e-5)]
loaded={}
for name,ot,it,o,i in cases:
    for seed in [8003,8006]:
        d=json.loads((DATA/f's{seed}_p1_outer{ot}_inner{it}.json').read_text(encoding='utf-8'))
        assert d['finished'] and d['converged'] and not d['extrapolation']
        assert max(abs(np.diff(d['history'][-3:])))<=o
        loaded[name,seed]=d
for name,ot,it,o,i in cases:
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,seed in zip(axes,[8003,8006]):
        d=loaded[name,seed];h=np.array(d['history']);n=d['iterations']
        ymin=min(loaded[c[0],seed]['history'][1] for c in cases)
        ymax=max(loaded[c[0],seed]['final_wsr'] for c in cases)
        pad=(ymax-ymin)*.08
        ax.plot(np.arange(1,n+1),h[1:],lw=2,color='#1769aa')
        ax.scatter([n],[h[-1]],color='#d95f02',s=35,zorder=5)
        ax.annotate(f'{n} rounds; WSR {h[-1]:.6f}',(n,h[-1]),xytext=(-8,-24),
                    textcoords='offset points',ha='right',fontsize=10)
        ax.set(title=f'Scene {seed}, 30 dBm',xlabel='Outer iteration',
               ylabel='Weighted sum rate (bit/s/Hz)',xlim=(1,n+2),ylim=(ymin-pad,ymax+pad))
        ax.grid(alpha=.25)
        inset=ax.inset_axes([.54,.13,.4,.33])
        inset.plot(np.arange(n+1),h,lw=1.2,color='#1769aa')
        inset.set_title('Full trace, including round 0',fontsize=8)
        inset.tick_params(labelsize=7)
        inset.grid(alpha=.2)
    fig.suptitle(f'Original Proposed: outer {o:.0e}, inner {i:.0e}\n'
                 'Ordinary FP + FP-aware-2; outer absolute tolerance for two consecutive rounds',fontsize=12)
    for suffix in ['png','pdf']:fig.savefig(OUT/f'{name}.{suffix}',dpi=200)
    plt.close(fig)
with zipfile.ZipFile(OUT/'three_convergence_figures.zip','w',zipfile.ZIP_DEFLATED) as z:
    for f in sorted(OUT.iterdir()):
        if f.suffix in ['.png','.pdf']:z.write(f,f.name)
    z.write(Path(__file__),Path(__file__).name)
for key,d in loaded.items():print(key,d['iterations'],round(d['seconds'],2),round(d['final_wsr'],9))
print(OUT)
