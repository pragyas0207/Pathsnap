"""Safely import the supplied ZIPs and preserve a hashed, inspectable inventory."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inspect(archives: list[Path], destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    inventory = []
    for archive in archives:
        target = destination / archive.stem
        target.mkdir(exist_ok=True)
        root = target.resolve()
        entries = []
        with zipfile.ZipFile(archive) as source:
            if sum(x.file_size for x in source.infolist()) > 2_000_000_000:
                raise ValueError('Archive exceeds the 2 GB uncompressed limit')
            for item in source.infolist():
                path = (root / item.filename).resolve()
                if not path.is_relative_to(root):
                    raise ValueError(f'Unsafe archive path: {item.filename}')
                if item.is_dir():
                    path.mkdir(parents=True, exist_ok=True)
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                with source.open(item) as src, path.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
                entries.append({'member': item.filename, 'bytes': item.file_size,
                                'sha256': sha256(path)})
        inventory.append({'archive': archive.name, 'sha256': sha256(archive),
                          'files': entries})
    report = {'archives': inventory}
    (destination / 'inventory.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archives', nargs='+', type=Path)
    parser.add_argument('--out', type=Path, default=Path('data/raw'))
    args = parser.parse_args()
    result = inspect(args.archives, args.out)
    for archive in result['archives']:
        print(archive['archive'], len(archive['files']), 'files', archive['sha256'])
