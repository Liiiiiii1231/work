"""Pack upload sources only; never includes local simulation results."""
from pathlib import Path
import json,hashlib,zipfile
ROOT=Path(__file__).resolve().parent

def main():
    manifest=json.loads((ROOT/'UPLOAD_FPACCEL_O4_I5.json').read_text(encoding='utf-8'))
    for rel,expected in manifest['files'].items():
        p=ROOT/rel
        if hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=expected:
            raise RuntimeError('Upload manifest changed: '+rel)
    for support in ['final_support_fpaccel_o4_i5_s8005','tradeoff_support_fpaccel_o4_i5_s8005']:
        report=json.loads((ROOT/support/'validation_report.json').read_text(encoding='utf-8'))
        if not report['passed']:raise RuntimeError('Release validation has not passed')
    archive=ROOT.parent/'toma_fpaccel_o4_i5_upload.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for rel in sorted(manifest['files']):z.write(ROOT/rel,'toma/'+rel)
        z.write(ROOT/'UPLOAD_FPACCEL_O4_I5.json','toma/UPLOAD_FPACCEL_O4_I5.json')
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise RuntimeError('Upload archive CRC check failed')
        assert not any('/results/' in n or '__pycache__' in n for n in z.namelist())
    print(archive)
    print('Source files:',len(manifest['files']),'bytes:',archive.stat().st_size)

if __name__=='__main__':main()
