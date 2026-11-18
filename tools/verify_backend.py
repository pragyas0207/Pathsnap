"""Exercise all final evaluator trips through the frozen, running backend."""
from pathlib import Path
from time import perf_counter
from datetime import datetime, timezone
import argparse
import json
import urllib.request
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='http://127.0.0.1:8766')
    parser.add_argument('--final-evaluation', action='store_true', required=True,
                        help='Explicitly perform final-only inference, never model selection')
    args = parser.parse_args()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def get(path):
        with opener.open(args.base + path, timeout=60) as response:
            return json.load(response)
    health = get('/health')
    assert health['ready'] and health['frozen'] and health['partition'] == 'evaluator'
    manifest = json.loads((ROOT / 'artifacts/final/model_manifest.json').read_text())
    assert health['model_sha256'] == manifest['model_sha256']
    available = get('/trips')
    imported = json.loads((ROOT / 'data/evaluation_server/import_receipt.json').read_text())
    assert available['trips'] == imported['trip_ids'] and len(available['trips']) == 10
    results = []
    folder = ROOT / 'artifacts/final_evaluation'; folder.mkdir(exist_ok=True)
    for trip_id in available['trips']:
        request = urllib.request.Request(args.base + '/match',
            data=json.dumps({'trip_id': trip_id}).encode(),
            headers={'Content-Type': 'application/json'}, method='POST')
        start = perf_counter()
        with opener.open(request, timeout=600) as response:
            result = json.load(response)
            assert response.status == 200
        seconds = perf_counter() - start
        assert result['model_sha256'] == manifest['model_sha256']
        assert 'labels' not in result and 'reference_arc_ids' not in result
        assert result['scoring_error'] is None
        assert len(result['prediction_arc_ids']) == result['observations']
        stem = ROOT / 'data/evaluation_server/evaluator' / trip_id / trip_id
        arcs = np.loadtxt(stem.with_suffix('.arcs'), dtype=int, ndmin=2)
        track = np.loadtxt(stem.with_suffix('.track'), ndmin=2)
        assert len(track) == result['observations']
        for name in ('prediction_arc_ids', 'classifier_arc_ids'):
            ids = np.array(result[name])
            assert ((ids == -1) | ((ids >= 0) & (ids < len(arcs)))).all()
        for part in result['reconstructed_route_parts']:
            ids = np.array(part, dtype=int)
            assert (arcs[ids[:-1], 1] == arcs[ids[1:], 0]).all()
        with opener.open(args.base + result['map_path'], timeout=120) as response:
            map_html = response.read().decode()
        assert 'Reference route' not in map_html
        assert '/assets/leaflet/leaflet.js' in map_html
        # Prediction outputs are retained; supplied reference/point-label arrays are not copied here.
        (folder / (trip_id + '_prediction.json')).write_text(json.dumps(result))
        results.append({'trip_id': trip_id, 'http_status': 200, 'request_id': result['request_id'],
            'observations': result['observations'], 'http_seconds': seconds,
            'timing': result['timing'], 'metrics': result['metrics'],
            'unmatched': result['unmatched_observations'], 'resets': len(result['decoder_resets'])})
        receipt = {'scope': 'final evaluator inference on this backend laptop; not a physical two-laptop rehearsal',
            'started_after_freeze': True, 'model_sha256': manifest['model_sha256'],
            'completed_utc': datetime.now(timezone.utc).isoformat(), 'base_url': args.base,
            'partition': 'evaluator', 'results': results}
        (ROOT / 'artifacts/evaluator_api_smoke.json').write_text(json.dumps(receipt, indent=2))
        print(trip_id, 'HTTP 200',result['observations'],'observations',f'{seconds:.3f}s',
              'HMM proxy',result['metrics']['hmm']['top1_accuracy_proxy'] if result['metrics'] else None,flush=True)
    for path in ('/ui/', '/assets/leaflet/leaflet.js', '/assets/leaflet/leaflet.css'):
        with opener.open(args.base + path, timeout=30) as response:
            assert response.status == 200
    print('PASS: all ten final evaluator trips, frozen model, valid IDs/lengths/connected parts, reference-free maps.')


if __name__ == '__main__':
    main()
