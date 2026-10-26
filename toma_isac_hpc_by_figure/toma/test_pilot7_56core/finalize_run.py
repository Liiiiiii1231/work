"""Export numerical results to one JSON file, then archive the entire run."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile
import numpy as np


def export_run(out, project, run_exit_code):
    out = Path(out).resolve()
    project = Path(project).resolve()
    if not out.is_dir():
        raise ValueError('Run directory does not exist')
    provenance = out/'provenance'
    provenance.mkdir(exist_ok=True)
    for path in project.iterdir():
        if path.is_file() and path.suffix in ('.py','.json','.sh','.md'):
            shutil.copy2(path,provenance/path.name)
    shared = provenance/'shared_modules'
    shared.mkdir(exist_ok=True)
    for name in ('algorithms.py','channel_model.py','inner_solver.py','outer_solver.py','system_model.py'):
        path=project.parent/name
        if not path.is_file():
            raise RuntimeError('Missing shared module for provenance: '+str(path))
        shutil.copy2(path,shared/name)
    target = out/'all_data.json'
    temporary = out/'all_data.json.tmp'
    files = sorted(p for p in out.rglob('*') if p.is_file() and p not in (target,temporary))
    completion = out/'completion.json'
    complete = bool(json.loads(completion.read_text(encoding='utf-8')).get('complete')) if completion.exists() else False
    header = dict(format='toma-run-export-v1',generated_utc=datetime.now(timezone.utc).isoformat(),
                  run_exit_code=run_exit_code,complete=complete and run_exit_code==0,
                  all_converged=json.loads(completion.read_text(encoding='utf-8')).get('all_converged',False) if completion.exists() else False,
                  completion_definition='All single-pass tasks finished; see all_converged for convergence',
                  note='Partial runs remain partial. Complex NPZ arrays use separate real/imag fields. CSV cells retain original text.')
    counts = dict(json=0,csv=0,npz=0,text=0,binary=0)
    with temporary.open('w',encoding='utf-8') as stream:
        stream.write('{"run":')
        json.dump(header,stream,ensure_ascii=False)
        stream.write(',"files":[')
        for index,path in enumerate(files):
            entry = dict(path=path.relative_to(out).as_posix(),bytes=path.stat().st_size,
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            suffix = path.suffix.lower()
            if suffix=='.json':
                entry.update(kind='json',data=json.loads(path.read_text(encoding='utf-8-sig')))
            elif suffix=='.csv':
                with path.open(encoding='utf-8-sig',newline='') as f:
                    reader=csv.DictReader(f)
                    entry.update(kind='csv',columns=reader.fieldnames,rows=list(reader))
            elif suffix=='.npz':
                arrays={}
                with np.load(path,allow_pickle=False) as archive:
                    for name in archive.files:
                        a=archive[name]
                        data=dict(dtype=str(a.dtype),shape=list(a.shape))
                        if np.iscomplexobj(a):data.update(real=a.real.tolist(),imag=a.imag.tolist())
                        else:data['values']=a.tolist()
                        arrays[name]=data
                entry.update(kind='npz',arrays=arrays)
            elif suffix in ('.txt','.log','.out','.err','.py','.sh','.md'):
                entry.update(kind='text',text=path.read_text(encoding='utf-8',errors='replace'))
            else:
                entry.update(kind='binary',note='Original file is included in the ZIP archive')
            counts[entry['kind']]+=1
            if index:stream.write(',')
            json.dump(entry,stream,ensure_ascii=False,allow_nan=False)
        stream.write('],"file_counts":')
        json.dump(counts,stream)
        stream.write('}\n')
    os.replace(temporary,target)
    archive_path=out.with_name(out.name+'_bundle.zip')
    archive_tmp=archive_path.with_suffix('.zip.tmp')
    with zipfile.ZipFile(archive_tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4,allowZip64=True) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file():z.write(path,out.name+'/'+path.relative_to(out).as_posix())
    with zipfile.ZipFile(archive_tmp) as z:
        bad=z.testzip()
        if bad:raise RuntimeError('Archive verification failed: '+bad)
    os.replace(archive_tmp,archive_path)
    print('Consolidated data:',target,flush=True)
    print('Verified archive:',archive_path,flush=True)
    return target,archive_path


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--run-exit-code',type=int,default=0)
    args=parser.parse_args()
    export_run(args.output,args.project,args.run_exit_code)
