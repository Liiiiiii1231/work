"""Automatically combine the four independently exported final figure runs."""
from pathlib import Path
import argparse,csv,hashlib,json,os,zipfile

DIRECTORIES=('01_proposed_convergence','02_wsr_vs_p_dl_max','03_wsr_vs_rho_si','04_wsr_vs_system_load')


def collect(toma,selection):
    frozen=selection/'frozen_initial_geometry.json'
    if not frozen.exists():return
    fingerprint=hashlib.sha256(frozen.read_bytes()).hexdigest();chosen={}
    for directory in DIRECTORIES:
        ready=[]
        for marker in (toma/directory/'results').glob('final_*/export_ready.json'):
            d=json.loads(marker.read_text(encoding='utf-8'))
            if d.get('frozen_sha256')==fingerprint:ready.append(marker)
        if not ready:
            print('Combined package pending:',directory,flush=True);return
        chosen[directory]=max(ready,key=lambda f:f.stat().st_mtime).parent
    dest=toma/'results';dest.mkdir(exist_ok=True)
    archive=dest/f'final_four_figures_{fingerprint[:12]}.zip';lock=archive.with_suffix('.lock')
    try:handle=lock.open('x')
    except FileExistsError:
        print('Another figure job is already assembling the combined package',flush=True);return
    try:
        handle.close();summary=dict(frozen_sha256=fingerprint,runs={})
        for directory,folder in chosen.items():
            data=dict(path=str(folder),completion=json.loads((folder/'completion.json').read_text(encoding='utf-8')),tables={})
            for table in (folder/'tables').glob('*.csv'):
                with table.open(encoding='utf-8-sig',newline='') as f:data['tables'][table.name]=list(csv.DictReader(f))
            summary['runs'][directory]=data
        summary_path=archive.with_suffix('.json');summary_path.write_text(json.dumps(summary,indent=2),encoding='utf-8')
        tmp=archive.with_suffix('.zip.tmp')
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=4,allowZip64=True) as z:
            z.writestr('all_figures_summary.json',json.dumps(summary,indent=2))
            for directory,folder in chosen.items():
                for path in folder.rglob('*'):
                    if path.is_file() and not path.name.endswith('.tmp'):
                        z.write(path,directory+'/'+folder.name+'/'+path.relative_to(folder).as_posix())
        with zipfile.ZipFile(tmp) as z:
            if z.testzip() is not None:raise RuntimeError('Combined archive verification failed')
        os.replace(tmp,archive);print('ALL FOUR FIGURES PACKAGED:',archive,flush=True)
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--toma-root',type=Path,default=Path(__file__).resolve().parent.parent)
    ap.add_argument('--selection-dir',type=Path,required=True)
    a=ap.parse_args();collect(a.toma_root.resolve(),a.selection_dir.resolve())
