"""Shared CSV, checksum and two-panel plotting helpers for weight refinement."""
from __future__ import annotations
import argparse
from copy import deepcopy
import csv
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch
from zipfile import ZipFile
import numpy as np
import yaml

FIELDS=['omega_comm','omega_sensing','scheme','scheme_label','sum_rate_dl','sum_rate_ul','sum_rate_comm','sum_rate_sensing',
        'weighted_sum_rate','best_seed','iterations','runtime_s','n_starts','converged','source_seed','provenance']
LABELS={'b2':'Fixed-ToMA + FP','b3':'FPA-UPA + FP','b6':'MA + FP','proposed':'FP-aware RCG-ToMA + FP (Proposed)'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csv_read(data):return list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))
def csv_write(p,rows,fields):
    temp=p.with_suffix('.tmp')
    with temp.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    os.replace(temp,p)
def json_write(p,d):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,p)
def geom_sha(g):
    return hashlib.sha256(np.r_[g.tx_positions.ravel(),g.rx_positions.ravel()].astype('<f8').tobytes()).hexdigest()
def res_sha(r):
    h=hashlib.sha256()
    for n in ('q_matrix','q_ul','b_ul','u_s'):
        a=getattr(r,n);h.update(a.dtype.str.encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()
def values(p):
    dl,ul,rs=map(float,(sum(p.rate_dl),sum(p.rate_ul),sum(p.rate_s)))
    return dict(sum_rate_dl=dl,sum_rate_ul=ul,sum_rate_comm=dl+ul,sum_rate_sensing=rs,weighted_sum_rate=float(p.weighted_sum_rate))
def nondominated(data):
    xy=np.array([[float(r['sum_rate_comm']),float(r['sum_rate_sensing'])] for r in data])
    keep=[i for i,p in enumerate(xy) if not any(np.all(q>=p-1e-10) and np.any(q>p+1e-10) for q in xy)]
    return sorted((data[i] for i in keep),key=lambda r:float(r['sum_rate_comm']))
def plot(rows,folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axs=plt.subplots(1,2,figsize=(13.5,5.8),sharex=True,sharey=True)
    filtered=[]
    for sc,color,marker in [('b6','#279b3b','^'),('proposed','#df2228','D')]:
        a=sorted((r for r in rows if r['scheme']==sc),key=lambda r:float(r['omega_comm']));front=nondominated(a);filtered+=front
        x=[float(r['sum_rate_comm']) for r in a];y=[float(r['sum_rate_sensing']) for r in a]
        axs[0].plot(x,y,color=color,marker=marker,lw=1.7,ms=4.5,mec='white',mew=.5,label=LABELS[sc])
        axs[1].scatter(x,y,color=color,marker=marker,s=22,alpha=.25)
        axs[1].plot([float(r['sum_rate_comm']) for r in front],[float(r['sum_rate_sensing']) for r in front],color=color,marker=marker,lw=1.7,ms=4.5,mec='white',mew=.5)
    for ax in axs:ax.grid(alpha=.22);ax.set_xlabel('Sum communication rate (bit/s/Hz)')
    axs[0].set_ylabel('Sum sensing rate (bit/s/Hz)')
    axs[0].set_title('(a) All points connected in weight order',fontsize=11)
    axs[1].set_title('(b) Nondominated sampled points highlighted',fontsize=11)
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.995),ncol=2,fontsize=9)
    fig.text(.5,.825,f'Scene 8001 | Regular planar initialization | One start per scheme | {len(set(float(r["omega_comm"]) for r in rows))} weights',ha='center',fontsize=10)
    fig.text(.065,.047,'Faint markers retain all original points. Each scheme uses the same nondominance rule. No smoothing.',fontsize=8.5)
    fig.text(.065,.015,'Lines guide the eye; they do not certify intermediate feasible states or a global Pareto boundary.',fontsize=8.5)
    fig.subplots_adjust(left=.065,right=.985,bottom=.17,top=.77,wspace=.12)
    folder.mkdir(exist_ok=True)
    for ext in ('png','pdf','svg'):fig.savefig(folder/f'proposed_ma_tradeoff_two_panels.{ext}',dpi=240,facecolor='white')
    plt.close(fig)
    csv_write(folder/'nondominated_sample_points.csv',filtered,FIELDS)
