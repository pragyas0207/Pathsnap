from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
import platform
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from .models import MODEL_NAMES, make_model, search_space
from .metrics import binary_metrics, point_metrics, sequence_metrics
from .candidates import Candidates, candidate_probabilities, independent_predictions, FEATURES, CandidateConfig
from .io import load_trip, load_reference, digest
from .geo import RoadNetwork
from .decoder import viterbi, DecoderConfig
from .reconstruct import reconstruct


def json_write(path, content):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(content, indent=2, default=str), encoding='utf-8')


def load_blocks(processed, partition):
    blocks = []
    for p in sorted((processed / partition).glob('*.npz')):
        with np.load(p) as src:
            block = {key: src[key] for key in src.files}
        block['trip_id'] = p.stem
        blocks.append(block)
    if not blocks:
        raise ValueError(f'No prepared {partition} data; run prepare first')
    return blocks


def assemble(blocks, fit_stride):
    X, y, groups, sampled = [], [], [], []
    for i, b in enumerate(blocks):
        mask = b['valid'] & (b['labels'][:, None] >= 0)
        stride = np.broadcast_to((np.arange(len(mask)) % fit_stride == 0)[:, None], mask.shape)
        X.append(b['X'][mask]); y.append(b['y'][mask])
        groups.append(np.full(mask.sum(), i, dtype=int)); sampled.append(stride[mask])
    return np.concatenate(X), np.concatenate(y), np.concatenate(groups), np.concatenate(sampled)


def score_blocks(model, blocks):
    ys, ps, points = [], [], []
    start = perf_counter()
    for b in blocks:
        c = Candidates(b['ids'], b['fraction'], b['X'], b['valid'])
        probability = candidate_probabilities(model, c)
        prediction = independent_predictions(c, probability)
        mask = b['valid'] & (b['labels'][:, None] >= 0)
        # Binary metrics use raw candidate probabilities; emissions are normalized separately.
        raw = model.predict_proba(b['X'][mask])[:, 1] if mask.any() else np.array([])
        ys.append(b['y'][mask]); ps.append(raw)
        points.append({'trip_id': b['trip_id'], **point_metrics(prediction, b['labels'], b['ids'])})
    result = binary_metrics(np.concatenate(ys), np.concatenate(ps))
    valid = [p for p in points if p['top1_accuracy_proxy'] is not None]
    result['top1_macro_trip_proxy'] = float(np.mean([p['top1_accuracy_proxy'] for p in valid]))
    denominator = sum(p['n_reference_aligned'] for p in points)
    result['top1_pooled_proxy'] = sum(p['top1_accuracy_proxy'] * p['n_reference_aligned'] for p in valid) / max(denominator, 1)
    result['candidate_miss_rate_proxy'] = sum(p['candidate_miss_rate_proxy'] * p['n_reference_aligned'] for p in valid) / max(denominator, 1)
    result['n_reference_aligned'] = denominator
    result['n_observations'] = sum(p['n_observations'] for p in points)
    result['scoring_seconds'] = perf_counter() - start
    return result, points


def cross_validate(name, params, blocks, arrays, folds, seed):
    X, y, groups, sampled = arrays
    results, splits = [], []
    for fold, (fit, val) in enumerate(GroupKFold(n_splits=folds).split(X, y, groups)):
        fit = fit[sampled[fit]]
        train_groups, val_groups = set(groups[fit]), set(groups[val])
        assert train_groups.isdisjoint(val_groups)
        model = make_model(name, seed, params)
        start = perf_counter()
        with threadpool_limits(limits=1):
            model.fit(X[fit], y[fit])
            fit_seconds = perf_counter() - start
            scores, _ = score_blocks(model, [b for i, b in enumerate(blocks) if i in val_groups])
        results.append({'fold': fold, 'fit_seconds': fit_seconds, **scores})
        splits.append({'fold': fold, 'train_trips': [blocks[i]['trip_id'] for i in sorted(train_groups)],
                       'validation_trips': [blocks[i]['trip_id'] for i in sorted(val_groups)],
                       'fit_candidate_rows': len(fit), 'validation_candidate_rows': len(val)})
    return results, splits


def train(processed: Path, output: Path, folds=3, fit_stride=5, seed=42, resume=False):
    if fit_stride < 1:
        raise ValueError('fit_stride must be >= 1')
    output.mkdir(parents=True, exist_ok=True)
    blocks = load_blocks(processed, 'train')
    student = load_blocks(processed, 'student_test')
    manifest = json.loads((processed / 'manifest.json').read_text())
    arrays = assemble(blocks, fit_stride)
    X, y, groups, sampled = arrays
    run = {'started_utc': datetime.now(timezone.utc).isoformat(), 'seed': seed, 'folds': folds,
           'fit_stride': fit_stride, 'training_trips': len(blocks), 'student_test_trips': len(student),
           'fit_candidate_rows': int(sampled.sum()), 'all_candidate_rows': len(y),
           'processed_manifest_sha256': digest(processed / 'manifest.json'),
           'python': platform.python_version(),
           'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'scikit-learn', 'optuna', 'shapely']},
           'primary_selection_metric': 'mean validation top1_macro_trip_proxy', 'models': {}}
    table = []
    if resume and (output / 'experiment_run.json').exists():
        previous = json.loads((output / 'experiment_run.json').read_text())
        for key in ('seed', 'folds', 'fit_stride', 'processed_manifest_sha256'):
            if previous[key] != run[key]:
                raise ValueError('Cannot resume: run configuration/input mismatch: ' + key)
        run = previous
        table = pd.read_csv(output / 'model_comparison.csv').to_dict('records')
    for name in MODEL_NAMES:
        if resume and name in run['models'] and (output / (name + '.joblib')).exists():
            print('RESUME completed', name, flush=True)
            continue
        print('TRAIN', name, flush=True)
        cv, splits = cross_validate(name, {}, blocks, arrays, folds, seed)
        model = make_model(name, seed)
        start = perf_counter()
        with threadpool_limits(limits=1):
            model.fit(X[sampled], y[sampled])
            fit_seconds = perf_counter() - start
            student_scores, per_trip = score_blocks(model, student)
            training_scores, _ = score_blocks(model, blocks)
        entry = {'configuration': model.get_params(), 'cv': cv, 'student_test': student_scores,
                 'training_resubstitution': training_scores, 'refit_seconds': fit_seconds}
        run['models'][name] = entry
        bundle = {'model': model, 'model_name': name, 'features': FEATURES,
                  'candidate_config': manifest['candidate_config'], 'decoder_config': asdict(DecoderConfig()),
                  'seed': seed, 'fit_stride': fit_stride, 'trained_partition': 'train',
                  'manifest_sha256': run['processed_manifest_sha256']}
        joblib.dump(bundle, output / (name + '.joblib'))
        json_write(output / 'cv_splits.json', splits)
        json_write(output / (name + '_student_trips.json'), per_trip)
        row = {'model': name, 'cv_macro_f1': float(np.mean([s['macro_f1'] for s in cv])),
               'cv_top1_macro_trip_proxy': float(np.mean([s['top1_macro_trip_proxy'] for s in cv])),
               'cv_top1_std': float(np.std([s['top1_macro_trip_proxy'] for s in cv])),
               'student_macro_f1': student_scores['macro_f1'], 'student_top1_macro_trip_proxy': student_scores['top1_macro_trip_proxy'],
               'student_top1_pooled_proxy': student_scores['top1_pooled_proxy'], 'student_log_loss': student_scores['log_loss'],
               'refit_seconds': fit_seconds, 'student_scoring_seconds': student_scores['scoring_seconds']}
        table.append(row)
        json_write(output / 'experiment_run.json', run)
        pd.DataFrame(table).to_csv(output / 'model_comparison.csv', index=False)
        print('RESULT', row, flush=True)
    best = max(table, key=lambda x: x['cv_top1_macro_trip_proxy'])['model']
    # Fixed model order breaks exact CV-score ties. Student labels do not select the winner.
    run['selected_untuned_model'] = best
    json_write(output / 'experiment_run.json', run)
    joblib.dump(joblib.load(output / (best + '.joblib')), output / 'selected.joblib')
    return run


def tune(processed: Path, output: Path, trials=15):
    import optuna
    run = json.loads((output / 'experiment_run.json').read_text())
    if trials < 4:
        raise ValueError('Use >=4 trials so search continues beyond the 3 random startup trials')
    name, seed = run['selected_untuned_model'], run['seed']
    blocks = load_blocks(processed, 'train')
    arrays = assemble(blocks, run['fit_stride'])
    storage = 'sqlite:///' + (output / 'optimization.sqlite3').resolve().as_posix()
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=3),
                               storage=storage, study_name='pathsnap_' + name, load_if_exists=False)

    def objective(trial):
        params = search_space(trial, name)
        scores, _ = cross_validate(name, params, blocks, arrays, run['folds'], seed)
        trial.set_user_attr('fold_results', scores)
        return float(np.mean([s['top1_macro_trip_proxy'] for s in scores]))

    start = perf_counter()
    study.optimize(objective, n_trials=trials, n_jobs=1)
    optimization_seconds = perf_counter() - start
    study.trials_dataframe().to_csv(output / 'optimization_trials.csv', index=False)
    X, y, _, sampled = arrays
    model = make_model(name, seed, study.best_params)
    with threadpool_limits(limits=1):
        start = perf_counter()
        model.fit(X[sampled], y[sampled])
        refit_seconds = perf_counter() - start
        student_scores, per_trip = score_blocks(model, load_blocks(processed, 'student_test'))
    bundle = joblib.load(output / (name + '.joblib'))
    bundle.update(model=model, model_name=name + '_tuned', tuning_parameters=study.best_params)
    joblib.dump(bundle, output / 'tuned.joblib')
    tuned = {'model': name, 'best_parameters': study.best_params, 'trials': len(study.trials),
             'startup_trials': 3, 'adaptive_tpe_trials': len(study.trials) - 3,
             'best_cv_top1_macro_trip_proxy': study.best_value,
             'best_fold_results': study.best_trial.user_attrs['fold_results'],
             'student_test': student_scores, 'optimization_seconds': optimization_seconds, 'refit_seconds': refit_seconds}
    json_write(output / 'tuning_summary.json', tuned)
    json_write(output / 'tuned_student_trips.json', per_trip)
    # Retain the baseline if tuning did not improve the CV selection objective.
    baseline_cv = np.mean([f['top1_macro_trip_proxy'] for f in run['models'][name]['cv']])
    if study.best_value > baseline_cv:
        joblib.dump(bundle, output / 'selected.joblib')
    return tuned


def evaluate_routes(root: Path, processed: Path, output: Path, all_models=False):
    models = list(MODEL_NAMES) + ['tuned', 'selected'] if all_models else ['selected']
    rows = []
    for b in load_blocks(processed, 'student_test'):
        path = root / 'student_test' / b['trip_id'] / (b['trip_id'] + '.track')
        net = RoadNetwork(load_trip(path))
        route = load_reference(path, len(net.trip.arcs))
        c = Candidates(b['ids'], b['fraction'], b['X'], b['valid'])
        for name in models:
            if not (output / (name + '.joblib')).exists():
                continue
            bundle = joblib.load(output / (name + '.joblib'))
            start = perf_counter()
            with threadpool_limits(limits=1):
                probabilities = candidate_probabilities(bundle['model'], c)
            raw = independent_predictions(c, probabilities)
            classifier_seconds = perf_counter() - start
            stages = [('classifier', raw, classifier_seconds, [])]
            if name == 'selected':
                start = perf_counter()
                decoded, resets = viterbi(net, c, probabilities, DecoderConfig(**bundle['decoder_config']))
                route_parts = reconstruct(net, decoded, [r['observation'] for r in resets], DecoderConfig(**bundle['decoder_config']))
                stages.append(('hmm', decoded, perf_counter() - start + classifier_seconds, resets))
                folder = output / 'predictions'; folder.mkdir(exist_ok=True)
                np.savez(folder / (b['trip_id'] + '.npz'), classifier=raw, hmm=decoded)
                json_write(folder / (b['trip_id'] + '_resets.json'), resets)
                json_write(folder / (b['trip_id'] + '_route_parts.json'), route_parts)
            for stage, prediction, seconds, resets in stages:
                rows.append({'trip_id': b['trip_id'], 'model': name, 'stage': stage,
                    **point_metrics(prediction, b['labels'], b['ids']),
                    **sequence_metrics(net, prediction, route), 'score_decode_seconds': seconds,
                    'resets': len(resets)})
            print('ROUTES', b['trip_id'], name, flush=True)
    pd.DataFrame(rows).to_csv(output / 'route_evaluation.csv', index=False)
    return rows


def compare_classifiers(root: Path, processed: Path, output: Path):
    """Classifier-only route diagnostics for the model-comparison checkpoint.

    Does not run a decoder, Bayesian search or API inference.
    """
    from types import SimpleNamespace
    rows = []
    bundles = {name: joblib.load(output / (name + '.joblib')) for name in MODEL_NAMES}
    for block in load_blocks(processed, 'student_test'):
        path = root / 'student_test' / block['trip_id'] / (block['trip_id'] + '.track')
        trip = load_trip(path)
        reference = load_reference(path, len(trip.arcs))
        network_view = SimpleNamespace(trip=trip)
        c = Candidates(block['ids'], block['fraction'], block['X'], block['valid'])
        for name, bundle in bundles.items():
            with threadpool_limits(limits=1):
                start = perf_counter()
                probability = candidate_probabilities(bundle['model'], c)
                prediction = independent_predictions(c, probability)
                seconds = perf_counter() - start
            rows.append({'trip_id': block['trip_id'], 'model': name,
                **point_metrics(prediction, block['labels'], block['ids']),
                **sequence_metrics(network_view, prediction, reference),
                'classifier_inference_seconds': seconds})
        print('COMPARE classifiers', block['trip_id'], flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'classifier_sequence_comparison.csv', index=False)
    comparison = pd.read_csv(output / 'model_comparison.csv')
    for column in ('strict_adjacency_rate', 'arc_set_f1', 'sampled_arc_edit_similarity', 'classifier_inference_seconds'):
        means = frame.groupby('model')[column].mean()
        comparison['student_mean_' + column] = comparison.model.map(means)
    comparison.to_csv(output / 'model_comparison.csv', index=False)
    return rows
