"""Plot the original 20-point cold trade-off with only four WSR-improving warm repairs."""
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results'/'stage10'/'cold20_targeted_warm_dl114'
SOURCE=BASE/'stage10_tradeoff_targeted.csv'
PLOT_DIR=BASE/'figures'
LABEL={'b2':'Fixed-ToMA + FP','b3':'FPA-UPA + FP','b6':'MA + FP','proposed':'FP-aware RCG-ToMA + FP (Proposed)'}
COLORS={'b2':'#1f77b4','b3':'#ff7f0e','b6':'#2ca02c','proposed':'#d62728'}


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(f'Missing {SOURCE}; run run_stage10_targeted20.py first.')
    with SOURCE.open(newline='',encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=80:
        raise RuntimeError(f'Expected 80 rows, got {len(rows)}')
    plt.rcParams.update({'font.size':11})
    fig,ax=plt.subplots(figsize=(9.5,6.6))
    for scheme in ('b2','b3','b6','proposed'):
        subset=sorted([r for r in rows if r['scheme']==scheme],key=lambda r:float(r['omega_comm']))
        if len(subset)!=20 or any(r['converged'].lower()!='true' for r in subset):
            raise RuntimeError(f'Incomplete or nonconverged result for {scheme}')
        x=np.array([float(r['sum_rate_comm']) for r in subset])
        y=np.array([float(r['sum_rate_sensing']) for r in subset])
        ax.plot(x,y,'-o',label=LABEL[scheme],color=COLORS[scheme],lw=2.4 if scheme=='proposed' else 1.65,ms=5.2)
        modified=[r for r in subset if r.get('selected_kind')=='target_converged_warm']
        if modified:
            ax.scatter([float(r['sum_rate_comm']) for r in modified],
                       [float(r['sum_rate_sensing']) for r in modified],
                       s=115,marker='o',facecolors='none',edgecolors=COLORS[scheme],linewidths=2.0,zorder=6)
    ax.set_xlabel('Sum Communication Rate (bit/s/Hz)')
    ax.set_ylabel('Sum Sensing Rate (bit/s/Hz)')
    ax.grid(alpha=.27)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.015),ncol=2,fontsize=9)
    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True,exist_ok=True)
    fig.savefig(PLOT_DIR/'communication_sensing_tradeoff_targeted.png',dpi=300,bbox_inches='tight')
    fig.savefig(PLOT_DIR/'communication_sensing_tradeoff_targeted.pdf',bbox_inches='tight')
    plt.close(fig)
    print('[PLOT]',PLOT_DIR/'communication_sensing_tradeoff_targeted.png')
    print('Open circles mark valid, improved, converged targeted warm-start results.')

if __name__=='__main__':main()
