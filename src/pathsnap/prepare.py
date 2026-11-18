from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from .io import trip_files, load_trip, load_reference, digest
from .geo import RoadNetwork
from .candidates import generate, CandidateConfig, FEATURES
from .reference import align_reference


def prepare(root: Path, output: Path, config=CandidateConfig()):
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'candidate_config': asdict(config), 'features': FEATURES,
                'alignment': {'method': 'monotone_route_position_min_squared_distance', 'max_distance_m': 100},
                'trips': []}
    for partition in ('train', 'student_test'):
        (output / partition).mkdir(exist_ok=True)
        for path in trip_files(root, partition):
            trip = load_trip(path)
            net = RoadNetwork(trip)
            route = load_reference(path, len(trip.arcs))
            c = generate(net, config)
            labels, residual, position = align_reference(net, route)
            target = (c.ids == labels[:, None]) & (labels[:, None] >= 0)
            np.savez(output / partition / (trip.trip_id + '.npz'), X=c.X, ids=c.ids,
                     fraction=c.fraction, valid=c.valid, y=target.astype(np.int8),
                     labels=labels, residual_m=residual, reference_position=position)
            entry = {'partition': partition, 'trip_id': trip.trip_id, 'observations': len(labels),
                     'aligned': int((labels >= 0).sum()), 'candidate_misses': int(((labels >= 0) & ~target.any(axis=1)).sum()),
                     'no_candidates': int((~c.valid.any(axis=1)).sum()),
                     'residual_median_m': float(np.median(residual)), 'residual_p95_m': float(np.quantile(residual, .95)),
                     'input_hashes': {s: digest(path.with_suffix('.' + s)) for s in ('track', 'nodes', 'arcs', 'route')}}
            manifest['trips'].append(entry)
            print('PREPARE', partition, trip.trip_id, entry['observations'], 'aligned', entry['aligned'], flush=True)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest
