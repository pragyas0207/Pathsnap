"""Check actual development artifacts before any evaluator trip is provisioned."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pathsnap.io import DEFAULT_DATA, load_trip, digest

run = json.loads((ROOT / 'artifacts/experiment_run.json').read_text())
tuning = json.loads((ROOT / 'artifacts/tuning_summary.json').read_text())
assert len(run['models']) == 5 and tuning['trials'] == 15 and tuning['adaptive_tpe_trials'] == 12
splits = json.loads((ROOT / 'artifacts/cv_splits.json').read_text())
held_out = []
for fold in splits:
    assert set(fold['train_trips']).isdisjoint(fold['validation_trips'])
    held_out.extend(fold['validation_trips'])
assert len(held_out) == len(set(held_out)) == 80
frame = pd.read_csv(ROOT / 'artifacts/route_evaluation.csv', dtype={'trip_id': str})
selected = frame[frame.model == 'selected']
assert len(frame) == 80 and len(selected) == 20
for trip_id in sorted(selected.trip_id.unique()):
    trip = load_trip(ROOT / DEFAULT_DATA / 'student_test' / trip_id / (trip_id + '.track'))
    with np.load(ROOT / 'artifacts/predictions' / (trip_id + '.npz')) as prediction:
        for stage in ('classifier', 'hmm'):
            ids = prediction[stage]
            assert len(ids) == len(trip.track)
            assert ((ids == -1) | ((ids >= 0) & (ids < len(trip.arcs)))).all()
    parts = json.loads((ROOT / 'artifacts/predictions' / (trip_id + '_route_parts.json')).read_text())
    for part in parts:
        ids = np.array(part, dtype=int)
        assert ((ids >= 0) & (ids < len(trip.arcs))).all()
        assert (trip.arcs[ids[:-1], 1] == trip.arcs[ids[1:], 0]).all()
summary = selected.groupby('stage')[['top1_accuracy_proxy', 'strict_adjacency_rate', 'arc_set_f1', 'sampled_arc_edit_similarity']].mean()
receipt = {'verified': True, 'five_models': 5, 'whole_trip_cv_folds': len(splits),
    'unique_validation_trips': len(set(held_out)), 'tpe_trials': tuning['trials'],
    'student_test_route_rows': len(frame), 'selected_student_trip_count': len(selected.trip_id.unique()),
    'valid_output_lengths_and_arc_ids': True, 'reconstructed_parts_are_directed_connected': True,
    'macro_trip_student_metrics': summary.to_dict('index'),
    'development_route_csv_sha256': digest(ROOT / 'artifacts/route_evaluation.csv'),
    'evaluator_contents_used': False}
(ROOT / 'artifacts/development_verification.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt, indent=2))
