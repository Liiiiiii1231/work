#!/usr/bin/env bash
#SBATCH --job-name=tradeoff_budget_pilot5
#SBATCH --partition=cpu
#SBATCH --nodes=5
#SBATCH --ntasks=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail

# Upload this launcher and the matching ZIP to toma/. Submit this file only.
UPLOAD_DIR="${PILOT_UPLOAD_DIR:-/home/lyq00008339/Lihuang_Li/toma}"
ZIP_PATH="${PILOT_ZIP:-$UPLOAD_DIR/tradeoff_budget_pilot_5nodes_upload.zip}"
if [[ ! -f "$ZIP_PATH" && -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/tradeoff_budget_pilot_5nodes_upload.zip" ]]; then
    UPLOAD_DIR="$SLURM_SUBMIT_DIR"
    ZIP_PATH="$UPLOAD_DIR/tradeoff_budget_pilot_5nodes_upload.zip"
fi
if [[ ! -f "$ZIP_PATH" ]]; then
    echo "ERROR: missing $ZIP_PATH; upload the matching pilot ZIP" >&2
    exit 1
fi
if [[ "${SLURM_JOB_NUM_NODES:?Submit with sbatch}" != 5 || "${SLURM_NTASKS:?}" != 5 || "${SLURM_CPUS_PER_TASK:?}" != 56 ]]; then
    echo 'ERROR: expected 5 nodes, one Slurm task per node, 56 CPUs per task' >&2
    exit 1
fi
source "${CONDA_SH:-/home/lyq00008339/miniconda3/etc/profile.d/conda.sh}"
conda activate "${TARGET_CONDA_ENV:-JM_CVX}"
PYTHON_BIN="${HPC_PYTHON:-python}"
UNPACK_BASE="${PILOT_UNPACK_BASE:-$UPLOAD_DIR/pilot_unpacked}"
if [[ "$UNPACK_BASE" == /tmp || "$UNPACK_BASE" == /tmp/* ]]; then
    echo 'ERROR: extraction must use shared /home storage, not node-local /tmp' >&2
    exit 1
fi
UNPACK_DIR="$UNPACK_BASE/job_${SLURM_JOB_ID}"
EXPECTED_SHA256="5283a7d6d53eb269ee95c7640e26793b97aaa3f61f3f7ce882b7954343eb4a2e"
"$PYTHON_BIN" - "$ZIP_PATH" "$UNPACK_DIR" "$EXPECTED_SHA256" <<'PY'
from pathlib import Path,PurePosixPath
import hashlib,json,stat,sys,zipfile
archive=Path(sys.argv[1]).resolve()
target=Path(sys.argv[2]).resolve()
if hashlib.sha256(archive.read_bytes()).hexdigest()!=sys.argv[3]:
    raise SystemExit('ZIP checksum differs. Upload the matching tradeoff_budget_pilot_5nodes_upload.zip.')
if target.exists():
    raise SystemExit('Extraction directory already exists. Use a new Slurm job; existing files are preserved.')
with zipfile.ZipFile(archive) as z:
    for entry in z.infolist():
        name=PurePosixPath(entry.filename)
        if name.is_absolute() or '..' in name.parts or '\\' in entry.filename:
            raise SystemExit('Unsafe archive member: '+entry.filename)
        if not name.parts or name.parts[0]!='tradeoff_budget_pilot_5nodes':
            raise SystemExit('Unexpected ZIP layout: '+entry.filename)
        if stat.S_ISLNK(entry.external_attr>>16):
            raise SystemExit('Archive symlinks are not supported')
        try:(target/Path(*name.parts)).resolve().relative_to(target)
        except ValueError:raise SystemExit('Archive member escapes extraction directory')
    target.mkdir(parents=True,exist_ok=False)
    z.extractall(target)
root=target/'tradeoff_budget_pilot_5nodes'
manifest=json.loads((root/'upload_manifest.json').read_text(encoding='utf-8'))
for rel,digest in manifest['files'].items():
    path=(root/rel).resolve()
    try:path.relative_to(root.resolve())
    except ValueError:raise SystemExit('Manifest path escapes pilot directory')
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        raise SystemExit('Extracted file checksum differs: '+rel)
print('PILOT ZIP EXTRACTED AND VERIFIED:',root,flush=True)
print('150 tasks; 5 nodes x 56 CPUs; MA outer 3000 / fixed FP 5000.',flush=True)
PY

export PILOT_ROOT="$UNPACK_DIR/tradeoff_budget_pilot_5nodes"
export PILOT_OUTPUT="${PILOT_OUTPUT:-$UPLOAD_DIR/tradeoff_budget_pilot_results/pilot_budget5_${SLURM_JOB_ID}}"
export PILOT_LOG_DIR="${SLURM_SUBMIT_DIR:-$UPLOAD_DIR}"
# Run in this allocation; do not submit another job.
exec bash "$PILOT_ROOT/submit_pilot.sh"
