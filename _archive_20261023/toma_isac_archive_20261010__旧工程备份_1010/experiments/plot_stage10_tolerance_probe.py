"""Plot B6/Proposed tolerance-vs-original comparison, no invented datapoints."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results'/'stage10'/'cold_wide20_dl114_tol1e4'/'stage10_tradeoff.csv'
OUT=ROOT/'results'/'stage10'/'tolerance_comparison_cold'
SCHEMES=('proposed','b6')
LABELS={'proposed':'Proposed (FP-aware RCG-ToMA + FP)','b6':'MA + FP (B6)'}
COLORS={'proposed':'tab:red','b6':'tab:green'}


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))


def plot(mode):
    folder=OUT/mode
    compare=folder/'comparison.csv'
    if not compare.exists(): raise FileNotFoundError(f'Run the tolerance test first: {compare}')
    data=read_csv(compare)
    valid=[r for r in data if r['status']=='complete' and r['new_wsr']!='']
    if not valid: raise RuntimeError('No completed converged groups to plot.')
    base=read_csv(BASE)
    baseline={s: sorted([r for r in base if r['scheme']==s],key=lambda r:float(r['omega_comm'])) for s in SCHEMES}
    tols=sorted({float(r['tol']) for r in valid},reverse=True)
    fig,axs=plt.subplots(1,2,figsize=(12.4,5.4),sharey=True)
    for ax,scheme in zip(axs,SCHEMES):
        b=baseline[scheme]
        ax.plot([float(r['sum_rate_comm']) for r in b], [float(r['sum_rate_sensing']) for r in b],
                color='0.65',linewidth=1.2,marker='.',markersize=5,alpha=.65,label='Original 1e-4 (20 weights)')
        for tol in tols:
            sr=sorted((r for r in valid if r['scheme']==scheme and float(r['tol'])==tol),key=lambda r:float(r['omega_comm']))
            if not sr: continue
            x=[float(r['new_rc']) for r in sr];y=[float(r['new_rs']) for r in sr]
            ax.plot(x,y,'o-',color=COLORS[scheme],markersize=4.5,linewidth=1.5,alpha=0.9 if tol==tols[0] else 0.65,
                    label=f'{tol:.0e} ({len(sr)} computed weights)')
        ax.set_title(LABELS[scheme]);ax.set_xlabel('Sum communication rate (bit/s/Hz)')
        ax.grid(alpha=.22);ax.legend(loc='best',fontsize=8)
    axs[0].set_ylabel('Sum sensing rate (bit/s/Hz)')
    fig.suptitle('Cold-start tolerance comparison: measured points only',fontsize=12)
    fig.tight_layout()
    folder.joinpath('figures').mkdir(exist_ok=True)
    out=folder/'figures'/'tradeoff_tolerance_comparison.png'
    fig.savefig(out,dpi=250,bbox_inches='tight');plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(12.4,4.6),sharey=False)
    for ax,scheme in zip(axs,SCHEMES):
        b=baseline[scheme]
        ax.plot([float(r['omega_comm']) for r in b],[float(r['weighted_sum_rate']) for r in b],
                color='0.65',marker='.',linewidth=1.2,markersize=5,label='Original 1e-4')
        for tol in tols:
            sr=sorted((r for r in valid if r['scheme']==scheme and float(r['tol'])==tol),key=lambda r:float(r['omega_comm']))
            if sr:
                ax.plot([float(r['omega_comm']) for r in sr],[float(r['new_wsr']) for r in sr],
                        'o-',linewidth=1.3,markersize=4,color=COLORS[scheme],alpha=0.9 if tol==tols[0] else 0.65,
                        label=f'{tol:.0e} ({len(sr)} points)')
        ax.set_title(LABELS[scheme]);ax.set_xlabel('Communication weight');ax.set_ylabel('WSR (bit/s/Hz)')
        ax.grid(alpha=.22);ax.legend(fontsize=8)
    fig.tight_layout()
    out2=folder/'figures'/'wsr_tolerance_comparison.png'
    fig.savefig(out2,dpi=250,bbox_inches='tight');plt.close(fig)
    print('Saved:',out,'\nSaved:',out2)
    if any(len([r for r in valid if r['scheme']==s and float(r['tol'])==t])!=20 for s in SCHEMES for t in tols):
        print('NOTE: Some tolerance curves use fewer than 20 points; they are diagnostic previews, not complete 20-point sweeps.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['best-seed','all-seeds'],default='best-seed')
    plot(p.parse_args().mode)
