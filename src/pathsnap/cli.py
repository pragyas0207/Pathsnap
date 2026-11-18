from __future__ import annotations
import argparse
from pathlib import Path
import json
import joblib
import numpy as np
from .io import DEFAULT_DATA, load_trip, load_reference, trip_files


def main():
    parser = argparse.ArgumentParser(description='Reproducible PathSnap workflow')
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--artifacts', type=Path, default=Path('artifacts'))
    parser.add_argument('--processed', type=Path, default=Path('data/processed'))
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('eda')
    p = sub.add_parser('prepare'); p.add_argument('--radius', type=float, default=60); p.add_argument('--candidates', type=int, default=8)
    p = sub.add_parser('train'); p.add_argument('--folds', type=int, default=3); p.add_argument('--fit-stride', type=int, default=5); p.add_argument('--seed', type=int, default=42); p.add_argument('--resume', action='store_true')
    p = sub.add_parser('tune'); p.add_argument('--trials', type=int, default=15)
    p = sub.add_parser('evaluate'); p.add_argument('--all-models', action='store_true')
    sub.add_parser('compare')
    p = sub.add_parser('map'); p.add_argument('--trip', default='00000000'); p.add_argument('--partition', choices=['train', 'student_test'], default='student_test')
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    if args.command == 'eda':
        from .eda import explore
        explore(args.data, args.artifacts / 'eda')
    elif args.command == 'prepare':
        from .prepare import prepare
        from .candidates import CandidateConfig
        prepare(args.data, args.processed, CandidateConfig(radius_m=args.radius, max_candidates=args.candidates))
    elif args.command == 'train':
        from .experiments import train
        train(args.processed, args.artifacts, args.folds, args.fit_stride, args.seed, args.resume)
    elif args.command == 'tune':
        from .experiments import tune
        tune(args.processed, args.artifacts, args.trials)
    elif args.command == 'evaluate':
        from .experiments import evaluate_routes
        evaluate_routes(args.data, args.processed, args.artifacts, args.all_models)
    elif args.command == 'compare':
        from .experiments import compare_classifiers
        compare_classifiers(args.data, args.processed, args.artifacts)
    elif args.command == 'map':
        from .pipeline import predict
        from .maps import make_map
        paths = {p.stem: p for p in trip_files(args.data, args.partition)}
        if args.trip not in paths:
            parser.error('Unknown trip ID')
        path = paths[args.trip]
        bundle = joblib.load(args.artifacts / 'selected.joblib')
        net, c, raw, hmm, parts, resets, timing = predict(load_trip(path), bundle)
        reference = load_reference(path, len(net.trip.arcs))
        folder = args.artifacts / 'maps'; folder.mkdir(exist_ok=True)
        route = np.array([a for part in parts for a in part], dtype=int)
        make_map(net, route, reference=reference, classifier=raw, expose_reference=True,
                 asset_base='../../frontend/vendor/leaflet' if Path('frontend/vendor/leaflet').exists() else None).save(folder / (args.trip + '.html'))
        print(json.dumps(timing))


if __name__ == '__main__':
    main()
