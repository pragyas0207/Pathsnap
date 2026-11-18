from pathlib import Path
import json
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pathsnap.io import digest

manifest = json.loads((ROOT / 'artifacts/final/model_manifest.json').read_text())
assert digest(ROOT / 'artifacts/final/model.joblib') == manifest['model_sha256']
for relative, expected in manifest['inference_source_sha256'].items():
    assert digest(ROOT / relative) == expected
receipt = json.loads((ROOT / 'artifacts/evaluator_api_smoke.json').read_text())
assert receipt['model_sha256'] == manifest['model_sha256'] and len(receipt['results']) == 10
rows = []
for result in receipt['results']:
    raw, hmm = result['metrics']['classifier'], result['metrics']['hmm']
    rows.append({'trip_id': result['trip_id'], 'http_status': result['http_status'],
        'gps_observations': result['observations'], 'aligned_observations': hmm['n_reference_aligned'],
        'alignment_coverage': hmm['alignment_coverage'], 'candidate_miss_rate_proxy': hmm['candidate_miss_rate_proxy'],
        'classifier_top1_proxy': raw['top1_accuracy_proxy'], 'hmm_top1_proxy': hmm['top1_accuracy_proxy'],
        'classifier_strict_adjacency': raw['strict_adjacency_rate'], 'hmm_strict_adjacency': hmm['strict_adjacency_rate'],
        'hmm_arc_set_f1': hmm['arc_set_f1'], 'hmm_sampled_edit_similarity': hmm['sampled_arc_edit_similarity'],
        'http_seconds': result['http_seconds'], 'pipeline_seconds': result['timing']['prediction_seconds'],
        'unmatched_observations': result['unmatched'], 'decoder_resets': result['resets'], 'request_id': result['request_id']})
frame = pd.DataFrame(rows)
frame.to_csv(ROOT / 'artifacts/evaluator_results.csv', index=False)
denominator = frame.aligned_observations.sum()
summary = {'frozen_model_sha256': manifest['model_sha256'],
    'frozen_model_name': manifest['model_name'], 'trip_count': len(frame),
    'successful_http_requests': int(frame.http_status.eq(200).sum()),
    'gps_observations': int(frame.gps_observations.sum()), 'aligned_observations': int(denominator),
    'alignment_coverage': float(denominator/frame.gps_observations.sum()),
    'classifier_top1_macro_trip_proxy': float(frame.classifier_top1_proxy.mean()),
    'hmm_top1_macro_trip_proxy': float(frame.hmm_top1_proxy.mean()),
    'classifier_top1_pooled_proxy': float((frame.classifier_top1_proxy*frame.aligned_observations).sum()/denominator),
    'hmm_top1_pooled_proxy': float((frame.hmm_top1_proxy*frame.aligned_observations).sum()/denominator),
    'classifier_strict_adjacency_macro_trip': float(frame.classifier_strict_adjacency.mean()),
    'hmm_strict_adjacency_macro_trip': float(frame.hmm_strict_adjacency.mean()),
    'mean_http_seconds': float(frame.http_seconds.mean()), 'max_http_seconds': float(frame.http_seconds.max()),
    'current_model_and_inference_sources_match_frozen_manifest': True,
    'physical_second_laptop_tested': False, 'final_evaluation_used_for_tuning': False}
(ROOT / 'artifacts/final_summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
