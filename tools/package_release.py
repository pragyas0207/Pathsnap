"""Package a frontend-only handoff and an explicit, auditable source release.

Never packages raw data, evaluator inputs, environments or kernel/server logs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tree(folder: str, suffixes=None):
    for path in sorted((ROOT / folder).rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or '.ipynb_checkpoints' in path.parts:
            continue
        if suffixes is None or path.suffix in suffixes:
            yield path


def archive(target: Path, files: dict[str, bytes], purpose: str):
    forbidden = {'.track', '.nodes', '.arcs', '.route', '.pyc'}
    for name in files:
        parts = Path(name).parts
        assert not Path(name).is_absolute() and '..' not in parts
        assert not {'data', '.venv', '.git', '.cache', '__pycache__', '.ipynb_checkpoints'}.intersection(parts), name
        assert Path(name).suffix not in forbidden, name
    manifest = {'purpose': purpose, 'members': [
        {'path': name, 'bytes': len(data), 'sha256': sha(data)} for name, data in sorted(files.items())]}
    files['PACKAGE_MANIFEST.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 18, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for item in manifest['members']:
            assert sha(z.read(item['path'])) == item['sha256']
    return {'file': target.relative_to(ROOT).as_posix(), 'bytes': target.stat().st_size,
            'sha256': sha(target.read_bytes()), 'members': len(files)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', default='http://10.1.24.198:8766')
    args = parser.parse_args()
    assert args.backend.startswith('http://') and all(c not in args.backend for c in '\r\n\"\'')
    out = ROOT / 'output'; out.mkdir(exist_ok=True)
    frontend = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in tree('frontend')}
    frontend['frontend/config.js'] = ('window.PATHSNAP_DEFAULT_API = ' + json.dumps(args.backend) + ';\n').encode()
    frontend['START_UI.cmd'] = b'@echo off\r\ncd /d "%~dp0"\r\necho Open http://localhost:5500 in your browser. Keep this window open.\r\npython -m http.server 5500 --bind 127.0.0.1 --directory frontend\r\npause\r\n'
    frontend['README_LAPTOP2.txt'] = (
        'PathSnap frontend laptop\n\n'
        '1. Extract this ZIP. Python 3 must be installed.\n'
        '2. Run START_UI.cmd and keep that window open.\n'
        '3. Open http://localhost:5500 .\n'
        f'4. Backend address: {args.backend} . Click Connect & load trips.\n'
        '5. Choose any of the ten trips and click Match route.\n'
        '6. Match the displayed request ID to the log on the backend laptop.\n\n'
        'Both laptops must share a network that allows device-to-device access.\n'
        'If the backend Wi-Fi IP changes, edit the Backend address field.\n'
        'This package contains no dataset, reference routes or model weights.\n'
        'Leaflet assets are local; optional OpenStreetMap background tiles need internet.\n'
        'Pointwise scores are reference-alignment proxies.\n'
    ).encode()
    assert not any(Path(n).suffix == '.joblib' for n in frontend)
    receipts = [archive(out / 'PathSnap_frontend_laptop2.zip', frontend, 'Frontend only for Laptop 2')]

    source = {}
    for folder in ['src', 'tools', 'scripts', 'tests', 'frontend', 'docs']:
        for path in tree(folder):
            source[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for name in ['README.md', '.gitignore', '.gitattributes', 'pyproject.toml', 'requirements.txt', 'requirements-lock.txt',
                 'requirements-notebook.txt', 'START_BACKEND.cmd', 'OPEN_NOTEBOOK.cmd',
                 'SHOW_OUTPUTS.cmd', 'VIEW_BACKEND_LOGS.cmd']:
        source[name] = (ROOT / name).read_bytes()
    for path in tree('notebooks', {'.ipynb'}):
        source[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    source['PathSnap_outputs.ipynb'] = (ROOT / 'notebooks/PathSnap_final_demo.ipynb').read_bytes()
    (ROOT / 'PathSnap_outputs.ipynb').write_bytes(source['PathSnap_outputs.ipynb'])
    # Root artifacts are generated results/configurations, not input files.
    for path in sorted((ROOT / 'artifacts').iterdir()):
        if path.is_file() and path.suffix in {'.json', '.csv', '.joblib', '.sqlite3'}:
            source[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for folder in ['artifacts/final', 'artifacts/eda', 'artifacts/figures', 'artifacts/maps', 'artifacts/screenshots']:
        for path in tree(folder):
            source[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for name in ['final_tests.log', 'tuning_run.log', 'route_evaluation_run.log', 'evaluator_verification_run.log']:
        source['artifacts/' + name] = (ROOT / 'artifacts' / name).read_bytes()
    report = ROOT / 'output/pdf/PathSnap_technical_report.pdf'
    assert report.is_file(), 'Build the final PDF before packaging.'
    source[report.relative_to(ROOT).as_posix()] = report.read_bytes()
    source['RELEASE_NOTES.txt'] = (
        'PathSnap source release for team review\n\n'
        'Run from the extracted project root. See README.md and docs/BACKEND_HANDOFF.md.\n'
        'Original train/student ZIPs and private evaluator ZIP must be supplied separately.\n'
        'No raw/processed/evaluator input data, virtual environment, credentials or runtime tokens are included.\n'
        'The frozen inference source hashes and final model are preserved.\n'
        'Do not refit on evaluator outcomes. Only load trusted joblib artifacts.\n'
        'Publication destination: https://github.com/pragyas0207/Pathsnap .\n'
        'Physical two-laptop rehearsal, receiving-terminal screenshot and team details remain team tasks.\n'
    ).encode()
    receipts.append(archive(out / 'PathSnap_source_release.zip', source, 'Source, trusted models and measured evidence; excludes input data'))
    receipt = {'packages': receipts, 'backend_url_at_packaging': args.backend,
               'physical_second_laptop_verified': False, 'raw_evaluator_inputs_included': False}
    (out / 'release_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
