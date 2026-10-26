"""Plot the two controlled reruns, including the stopping increments."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
fig,axes=plt.subplots(1,3,figsize=(14,4.3))
colors=['#2368b4','#db761d']
for seed,color in zip([8003,8006],colors):
    d=json.loads((ROOT/f's{seed}_p1_n120_initial_accelerated_joint.json').read_text(encoding='utf-8'))
    h=np.asarray(d['history']); x=np.arange(len(h)); inc=np.diff(h)
    label=f'Scene {seed} ({d["iterations"]} iterations)'
    axes[0].plot(x,h,'o-',color=color,label=label,markersize=3,linewidth=1.7)
    axes[1].plot(x[1:],h[1:],'o-',color=color,label=label,markersize=3,linewidth=1.7)
    axes[1].scatter(x[-1],h[-1],s=65,facecolors='white',edgecolors=color,zorder=4)
    axes[1].annotate(f'{h[-1]:.5f}',(x[-1],h[-1]),xytext=(-5,8),textcoords='offset points',
        ha='right',color=color,fontsize=9)
    axes[2].semilogy(x[1:],inc,'o-',color=color,label=label,markersize=3,linewidth=1.5)
    axes[2].scatter(x[-2:],inc[-2:],s=60,facecolors='white',edgecolors=color,zorder=4)
axes[0].set(title='Full trajectory, including initialization',xlabel='Outer iteration',ylabel='True WSR')
axes[1].set(title='Zoom after the first outer iteration',xlabel='Outer iteration',ylabel='True WSR')
lo,hi=axes[1].get_ylim()
axes[1].set_ylim(lo,hi+.06)
axes[2].axhline(1e-3,color='#555555',ls='--',lw=1.3,label='Outer tolerance = 1e-3')
axes[2].set(title='Per-iteration WSR increment (log scale)',xlabel='Outer iteration',ylabel='Absolute WSR increment')
for ax in axes:
    ax.grid(alpha=.22)
    ax.set_xlim(left=0)
axes[0].legend(loc='lower right',fontsize=8)
axes[2].legend(loc='upper right',fontsize=8)
fig.suptitle('Controlled Proposed tests: 30 dBm; outer abs tolerance 1e-3 twice; inner tolerance 1e-5',fontsize=12)
fig.tight_layout(rect=(0,0,1,.93))
for ext in ['png','svg']:
    fig.savefig(ROOT/f'convergence_o1e3_i1e5.{ext}',dpi=190,bbox_inches='tight')
print(str(ROOT/'convergence_o1e3_i1e5.png'))
