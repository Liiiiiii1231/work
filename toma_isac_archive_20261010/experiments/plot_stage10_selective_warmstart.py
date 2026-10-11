"""Overlay original 14-point tradeoff and selectively warm-refined results.

This is a diagnostic comparison, NOT a smoothing or fitted Pareto frontier.
"""
from __future__ import annotations
import csv
from pathlib import Path
import json
from hashlib import sha256
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'stage10'/'selective_warmstart_tol1e4'
SCHEMES=(('b2','Fixed-ToMA + FP','#1f77b4'),
         ('b3','FPA-UPA + FP','#ff7f0e'),
         ('b6','MA + FP','#2ca02c'),
         ('proposed','Proposed','#d62728'))


def read_rows(path):
    if not path.exists():raise FileNotFoundError(path)
    with path.open('r',newline='',encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def plot(out=OUT):
    rows=read_rows(out/'stage10_tradeoff_targeted.csv')
    audit=read_rows(out/'targeted_comparison.csv')
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    target_sha=manifest['parameters']['ref_sha256']
    options=(ROOT/'results'/'stage10'/'cold_wide14_tol1e4'/'stage10_tradeoff.csv',
             ROOT/'reference'/'stage10_tradeoff_14point_reference.csv')
    refpath=next((p for p in options if p.is_file() and
                  sha256(p.read_bytes()).hexdigest()==target_sha),None)
    if refpath is None:
        raise RuntimeError('Cannot find the exact historical reference CSV used for this run.')
    old_rows=read_rows(refpath)
    fig,ax=plt.subplots(figsize=(9.5,6.5))
    for scheme,label,color in SCHEMES:
        old=sorted((x for x in old_rows if x['scheme']==scheme),key=lambda x:float(x['omega_comm']))
        new=sorted((x for x in rows if x['scheme']==scheme),key=lambda x:float(x['omega_comm']))
        ax.plot([float(x['sum_rate_comm']) for x in old],
                [float(x['sum_rate_sensing']) for x in old],ls=':',lw=1.0,color=color,alpha=.33)
        ax.plot([float(x['sum_rate_comm']) for x in new],
                [float(x['sum_rate_sensing']) for x in new],'-o',lw=2.3 if scheme=='proposed' else 1.55,
                ms=4.7,color=color,label=label)
        changed=[x for x in new if x.get('selected_kind')!='historical_unchanged']
        if changed:
            ax.scatter([float(x['sum_rate_comm']) for x in changed],
                       [float(x['sum_rate_sensing']) for x in changed],
                       s=80,marker='o',facecolors='none',edgecolors=color,linewidths=1.4,zorder=6)
    ax.set_xlabel('Sum Communication Rate (bit/s/Hz)')
    ax.set_ylabel('Sum Sensing Rate (bit/s/Hz)')
    ax.set_title('Diagnostic: selected warm-start refinements vs historical 14-point sweep')
    ax.grid(True,alpha=.23)
    ax.legend(ncol=2,loc='lower center',bbox_to_anchor=(.5,1.01),fontsize=9)
    fig.tight_layout()
    folder=out/'figures';folder.mkdir(parents=True,exist_ok=True)
    png=folder/'tradeoff_selective_warmstart.png'
    pdf=folder/'tradeoff_selective_warmstart.pdf'
    fig.savefig(png,dpi=220,bbox_inches='tight');fig.savefig(pdf,bbox_inches='tight');plt.close(fig)
    print('Saved:',png,'and',pdf)
    for r in audit:
        print(r['scheme'],r['omega_comm'],'historical=',r['historical_wsr'],
              'warm=',r['best_warm_wsr'],'selected=',r['selected_wsr'])


if __name__=='__main__':
    plot()
