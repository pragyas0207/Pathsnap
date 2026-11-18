"""Freeze development-selected model/configuration BEFORE evaluator data import."""
from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import sys
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pathsnap.io import digest


def main():
    artifacts = ROOT / 'artifacts'
    if not (artifacts / 'tuning_summary.json').exists() or not (artifacts / 'route_evaluation.csv').exists():
        raise SystemExit('Complete tuning and student-test decoder evaluation before freezing.')
    folder = artifacts / 'final'; folder.mkdir(exist_ok=True)
    if (folder / 'model_manifest.json').exists():
        raise SystemExit('A frozen manifest already exists. Preserve it; create an explicitly versioned release instead.')
    source = artifacts / 'selected.joblib'
    bundle = joblib.load(source)
    if bundle.get('trained_partition') != 'train':
        raise ValueError('Only training-partition fitted models can be released')
    shutil.copy2(source, folder / 'model.joblib')
    files = ['__init__.py', 'io.py', 'geo.py', 'candidates.py', 'decoder.py',
             'reconstruct.py', 'pipeline.py', 'maps.py', 'api.py', 'reference.py', 'metrics.py']
    manifest = {'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'model_name': bundle['model_name'], 'model_sha256': digest(folder / 'model.joblib'),
        'trained_partition': 'train', 'candidate_config': bundle['candidate_config'],
        'decoder_config': bundle['decoder_config'], 'features': bundle['features'],
        'fit_stride': bundle['fit_stride'], 'seed': bundle['seed'],
        'tuning_summary_sha256': digest(artifacts / 'tuning_summary.json'),
        'student_route_evaluation_sha256': digest(artifacts / 'route_evaluation.csv'),
        'processed_manifest_sha256': bundle['manifest_sha256'],
        'requirements_lock_sha256': digest(ROOT / 'requirements-lock.txt'),
        'selection_scope': 'training-trip CV only; student trips for development diagnostics; no evaluator contents read',
        'inference_source_sha256': {str(Path('src/pathsnap') / name).replace('\\', '/'): digest(ROOT / 'src/pathsnap' / name) for name in files}}
    (folder / 'model_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k:manifest[k] for k in ('frozen_utc','model_name','model_sha256')}, indent=2))


if __name__ == '__main__':
    main()
