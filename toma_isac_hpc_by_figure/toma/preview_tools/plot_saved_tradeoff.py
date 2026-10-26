"""Read-only preview of saved checkpoints, including unconverged results."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();source=a.source.resolve();out=a.output.resolve()
    if out==source or source in out.parents:
        raise ValueError('Preview output must be separate from original run directory')
    if out.exists():raise ValueError('Use a new preview output directory')
    meta=json.loads(next((source/'metadata').glob('scene_*.json')).read_text(encoding='utf-8-sig'))
    p=meta['protocol'];schemes=p['schemes'];seeds=list(range(p['scene_seed_start'],p['scene_seed_start']+meta['n_mc']['tradeoff']))
    rows=[];errors=[]
    for path in sorted((source/'candidate_0'/'tasks'/'tradeoff').glob('*/scene_*/*/record.json')):
        try:
            d=json.loads(path.read_text(encoding='utf-8-sig'))
            for file,h in (('initial.npz',d['initial_sha256']),(d['state_file'],d['state_sha256'])):
                blob=(path.parent/file).read_bytes()
                if hashlib.sha256(blob).hexdigest()!=h:raise ValueError('Checksum mismatch: '+file)
            with np.load(io.BytesIO(blob),allow_pickle=False) as state:
                if abs(float(state['wsr_history'][-1])-d['row']['weighted_sum_rate'])>1e-8:raise ValueError('History/record mismatch')
            r=dict(d['row'],state_file=str(path.parent/d['state_file']))
            if not all(np.isfinite(r[k]) for k in ('sum_rate_comm','sum_rate_sensing','weighted_sum_rate')):raise ValueError('Nonfinite rates')
            rows.append(r)
        except Exception as exc:errors.append(dict(path=str(path),error=str(exc)))
    lookup={(float(r['parameter_value']),r['scene_seed'],r['scheme']):r for r in rows}
    means=[];omitted=[]
    for w in p['tradeoff']['weights']:
        missing=[(s,sc) for s in seeds for sc in schemes if (float(w),s,sc) not in lookup]
        if missing:omitted.append(dict(weight=w,missing_tasks=len(missing)));continue
        for sc in schemes:
            group=[lookup[(float(w),s,sc)] for s in seeds]
            means.append(dict(weight=w,scheme=sc,n_scenes=len(group),n_converged=sum(r['converged'] for r in group),mean_comm=float(np.mean([r['sum_rate_comm'] for r in group])),mean_sensing=float(np.mean([r['sum_rate_sensing'] for r in group]))))
    out.mkdir(parents=True)
    def csv_write(name,items):
        if not items:return
        with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(items[0]));writer.writeheader();writer.writerows(items)
    csv_write('saved_per_scene.csv',rows);csv_write('preview_mean.csv',means)
    report=dict(preview_only=True,source=str(source),saved_tasks=len(rows),converged_tasks=sum(r['converged'] for r in rows),plotted_weights=sorted(set(r['weight'] for r in means)),omitted_weights=omitted,errors=errors,note='All expected scenes and all schemes required per plotted weight; unconverged saved states are explicitly included. Missing tasks are never filled with zero or initial states.')
    (out/'preview_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if means:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8.4,6.1))
        for sc,color,marker in zip(schemes,('#df2228','#279b3b','#e99a21','#2675c1'),('D','^','o','s')):
            group=[r for r in means if r['scheme']==sc]
            label={'proposed':'Proposed','b6':'MA','b2':'Fixed-ToMA','b3':'Fixed UPA'}[sc]
            ax.plot([r['mean_comm'] for r in group],[r['mean_sensing'] for r in group],color=color,marker=marker,label=label)
            for r in group:
                if r['n_converged'] < r['n_scenes']:
                    ax.scatter([r['mean_comm']],[r['mean_sensing']],s=65,marker=marker,facecolors='white',edgecolors=color,zorder=5)
        ax.set(xlabel='Mean communication rate (bit/s/Hz)',ylabel='Mean sensing rate (bit/s/Hz)',title='SAVED-STATE PREVIEW: includes unconverged results')
        ax.set_xlim(left=0);ax.set_ylim(bottom=0);ax.grid(alpha=.2);ax.legend(loc='best');fig.tight_layout()
        ax.text(.02,.02,'Open markers include unconverged states; weights and counts are in preview_mean.csv',transform=ax.transAxes,fontsize=7)
        for ext in ('png','pdf'):fig.savefig(out/('preview_tradeoff.'+ext),dpi=180)
        plt.close(fig)
    archive=out.with_name(out.name+'.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in out.iterdir():z.write(path,out.name+'/'+path.name)
    print(json.dumps(report,indent=2));print('Preview archive:',archive)


if __name__=='__main__':main()
