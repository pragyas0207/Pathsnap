"""Provision final-only evaluator files, requiring an already frozen model."""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import argparse
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pathsnap.io import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    frozen = ROOT / 'artifacts/final/model_manifest.json'
    manifest = json.loads(frozen.read_text())
    if digest(frozen.with_name('model.joblib')) != manifest['model_sha256']:
        raise ValueError('Frozen model checksum mismatch')
    output = ROOT / 'data/evaluation_server/evaluator'
    if output.exists():
        raise ValueError('Evaluator folder already exists; refusing to overwrite final inputs')
    entries = []
    groups = {}
    with zipfile.ZipFile(args.archive) as source:
        files = [f for f in source.infolist() if not f.is_dir()]
        if sum(f.file_size for f in files) > 2_000_000_000:
            raise ValueError('Archive too large')
        for member in files:
            path = PurePosixPath(member.filename)
            if '..' in path.parts or path.is_absolute() or '\\' in member.filename:
                raise ValueError('Unsafe archive member')
            trip = path.stem
            if not (len(trip) == 8 and trip.isdigit() and path.parent.name == trip
                    and path.suffix in ('.track', '.nodes', '.arcs', '.route')):
                raise ValueError('Unexpected evaluator member: ' + member.filename)
            suffixes = groups.setdefault(trip, set())
            if path.suffix in suffixes:
                raise ValueError('Duplicate evaluator member')
            suffixes.add(path.suffix)
        if len(groups) != 10 or any(v != {'.track', '.nodes', '.arcs', '.route'} for v in groups.values()):
            raise ValueError('Expected ten complete evaluator trips')
        for member in files:
            path = PurePosixPath(member.filename)
            target = output / path.parent.name / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as stream:
                target.write_bytes(stream.read())
            entries.append({'relative_path': str(target.relative_to(output)).replace('\\','/'),
                            'sha256': digest(target), 'bytes': member.file_size})
    receipt = {'imported_utc': datetime.now(timezone.utc).isoformat(), 'archive_sha256': digest(args.archive),
        'model_frozen_utc': manifest['frozen_utc'], 'model_sha256': manifest['model_sha256'],
        'trip_ids': sorted(groups), 'files': entries,
        'scope': 'Final inference and server-only scoring. Excluded from development commands and frontend package.'}
    (output.parent / 'import_receipt.json').write_text(json.dumps(receipt, indent=2))
    print('Provisioned',len(groups),'evaluator trips AFTER model freeze:',','.join(sorted(groups)))


if __name__ == '__main__':
    main()
