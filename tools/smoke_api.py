"""Run real HTTP inference on all student-test trips and retain a compact receipt."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='http://127.0.0.1:8000')
    parser.add_argument('--out', type=Path, default=Path('artifacts/api_smoke.json'))
    args = parser.parse_args()
    with urllib.request.urlopen(args.base + '/trips', timeout=20) as response:
        trips = json.load(response)
    if trips['partition'] != 'student_test':
        raise ValueError('Development smoke runner refuses evaluator data')
    results = []
    for trip in trips['trips']:
        request = urllib.request.Request(args.base + '/match',
            data=json.dumps({'trip_id': trip}).encode(), headers={'Content-Type': 'application/json'}, method='POST')
        start = perf_counter()
        with urllib.request.urlopen(request, timeout=600) as response:
            body = json.load(response)
            status = response.status
        assert len(body['prediction_arc_ids']) == body['observations']
        assert 'labels' not in body and 'reference_arc_ids' not in body
        with urllib.request.urlopen(args.base + body['map_path'], timeout=60) as response:
            html = response.read().decode()
        assert 'Reference route' not in html
        row = {'trip_id': trip, 'http_status': status, 'request_id': body['request_id'],
               'observations': body['observations'], 'elapsed_http_and_map_seconds': perf_counter() - start,
               'timing': body['timing'], 'metrics': body['metrics'],
               'unmatched': body['unmatched_observations'], 'resets': len(body['decoder_resets'])}
        results.append(row)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({'scope': 'local loopback HTTP; not a physical two-laptop demo',
                                       'partition': trips['partition'], 'results': results}, indent=2))
        print(trip, status, body['observations'], round(row['elapsed_http_and_map_seconds'], 3), flush=True)


if __name__ == '__main__':
    main()
