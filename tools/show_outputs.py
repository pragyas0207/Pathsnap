"""Present measured PathSnap outputs in a terminal, with optional plot windows."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    'logistic': 'Logistic Regression', 'naive_bayes': 'Gaussian Naive Bayes',
    'decision_tree': 'Decision Tree', 'random_forest': 'Random Forest',
    'hist_boost': 'HistGradientBoosting',
}


def read_artifacts(root=ROOT):
    root = Path(root)
    artifact = root / 'artifacts'
    required = ['model_comparison.csv', 'experiment_run.json', 'eda/summary.json']
    for name in required:
        if not (artifact / name).exists():
            raise FileNotFoundError(f'Missing executed output: {artifact / name}. Run the documented training workflow first.')
    with (artifact / 'model_comparison.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    return (rows, json.loads((artifact / 'experiment_run.json').read_text()),
            json.loads((artifact / 'eda/summary.json').read_text()))


def print_table(headers, rows):
    strings = [[str(value) for value in row] for row in rows]
    widths = [max(len(str(h)), *(len(row[i]) for row in strings)) for i, h in enumerate(headers)]
    def line(values):
        return '  '.join(str(v).ljust(w) for v, w in zip(values, widths))
    print(line(headers))
    print(line(['-' * w for w in widths]))
    for row in strings:
        print(line(row))


def fresh_predictions(trip_id='00000000', root=ROOT):
    """Load real trained classifiers and score prepared features again, without HMM."""
    import joblib
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits
    sys.path.insert(0, str(Path(root) / 'src'))
    from pathsnap.candidates import Candidates, candidate_probabilities, independent_predictions
    from pathsnap.metrics import point_metrics
    if len(trip_id) != 8 or not trip_id.isdigit():
        raise ValueError('Use an eight-digit student-test trip ID')
    path = Path(root) / 'data/processed/student_test' / (trip_id + '.npz')
    if not path.exists():
        raise FileNotFoundError(f'Trip not prepared: {path}')
    with np.load(path) as src:
        block = {key: src[key] for key in src.files}
    candidates = Candidates(block['ids'], block['fraction'], block['X'], block['valid'])
    predictions, summaries = {}, []
    for name in NAMES:
        bundle = joblib.load(Path(root) / 'artifacts' / (name + '.joblib'))
        start = perf_counter()
        with threadpool_limits(limits=1):
            probability = candidate_probabilities(bundle['model'], candidates)
        prediction = independent_predictions(candidates, probability)
        elapsed = perf_counter() - start
        predictions[name] = prediction
        # References are accessed for scoring after the model made its prediction.
        metrics = point_metrics(prediction, block['labels'], block['ids'])
        summaries.append({'model': name, 'observations': len(prediction),
                          'aligned_points': metrics['n_reference_aligned'],
                          'trip_top1_proxy': metrics['top1_accuracy_proxy'],
                          'fresh_classifier_seconds': elapsed})
    preview = pd.DataFrame({'GPS row': np.arange(len(block['labels'])),
                            'reference proxy': block['labels'], **predictions})
    return preview, pd.DataFrame(summaries), block


def comparison_figures(root=ROOT):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    rows, run, _ = read_artifacts(root)
    frame = pd.DataFrame(rows)
    for column in frame.columns[1:]:
        frame[column] = pd.to_numeric(frame[column])
    labels = [NAMES[name].replace(' ', '\n', 1) for name in frame.model]
    x = np.arange(len(frame))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, f1, top1, title in [
        (axes[0], 'cv_macro_f1', 'cv_top1_macro_trip_proxy', '3-fold validation | whole trips held out'),
        (axes[1], 'student_macro_f1', 'student_top1_macro_trip_proxy', 'Student testing | all 10 trips')]:
        for offset, column, color, label in [(-.19, f1, '#196b53', 'Candidate Macro F1'),
                                              (.19, top1, '#2563eb', 'Top-1 proxy (mean trip)')]:
            bars = ax.bar(x + offset, frame[column], width=.38, color=color, label=label)
            ax.bar_label(bars, fmt='%.3f', fontsize=8, padding=3)
        ax.set_xticks(x, labels, fontsize=8)
        ax.set(title=title, ylabel='Score', ylim=(0, 1.06))
        ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
        ax.legend(loc='lower right', fontsize=8)
    fig.suptitle('Five trained models - measured results', fontsize=15)
    fig.tight_layout()
    fig2, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    bars = axes[0].bar(x, frame.student_log_loss, color='#e76f00')
    axes[0].bar_label(bars, fmt='%.4f', fontsize=9, padding=3)
    axes[0].set(title='Student candidate log loss | lower is better', ylabel='Binary log loss', ylim=(0, max(frame.student_log_loss) * 1.25))
    times = frame.student_mean_classifier_inference_seconds
    bars = axes[1].bar(x, times, color='#7c3aed')
    axes[1].bar_label(bars, fmt='%.4f', fontsize=9, padding=3)
    axes[1].set(title='Classifier inference | mean seconds per student trip', ylabel='Seconds (prepared features only)', ylim=(0, max(times) * 1.25))
    for ax in axes:
        ax.set_xticks(x, labels, fontsize=8); ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    fig2.tight_layout()
    return [fig, fig2]


def confusion_figure(root=ROOT):
    import numpy as np
    import matplotlib.pyplot as plt
    _, run, _ = read_artifacts(root)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (name, entry) in zip(axes.flat, run['models'].items()):
        matrix = np.array(entry['student_test']['confusion_matrix'])
        ax.imshow(matrix, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{matrix[i, j]:,}', ha='center', va='center',
                        color='white' if matrix[i, j] > matrix.max() / 2 else '#102a43', fontsize=11)
        ax.set(title=NAMES[name], xlabel='Predicted candidate class', ylabel='True proxy candidate class')
        ax.set_xticks([0, 1], ['Incorrect (0)', 'Correct (1)'])
        ax.set_yticks([0, 1], ['Incorrect (0)', 'Correct (1)'])
    axes.flat[-1].axis('off')
    fig.suptitle('Student-test candidate confusion matrices | counts, not GPS-point classes')
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trip', default='00000000', help='Student-test trip for fresh classifier inference')
    parser.add_argument('--no-predict', action='store_true', help='Display saved measurements only')
    parser.add_argument('--plots', action='store_true', help='Open comparison and confusion-matrix plot windows')
    args = parser.parse_args()
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
    rows, run, eda = read_artifacts()
    print('\nPATHSNAP - DATA AND FIVE-MODEL OUTPUTS\n' + '=' * 76)
    print('Saved experiment measurements below. This command does not retrain models.\n')
    print('1. DATASET OUTPUT')
    print_table(['Partition', 'Trips', 'GPS observations', 'Gaps >30 sec'],
                [[p, s['trips'], f"{s['gps_observations']:,}", s['gaps_gt_30s']] for p, s in eda.items()])
    print(f"\nTraining: {run['folds']} whole-trip CV folds; fit every {run['fit_stride']}th observation.")
    print(f"Fit candidate rows: {run['fit_candidate_rows']:,}. All held-out/student observations scored.\n")
    print('2. FIVE-MODEL COMPARISON')
    print_table(['Model', 'CV F1', 'Test F1', 'Test Top-1*', 'Log loss', 'Adjacency'],
        [[NAMES[r['model']], f"{float(r['cv_macro_f1']):.4f}", f"{float(r['student_macro_f1']):.4f}",
          f"{float(r['student_top1_macro_trip_proxy']):.2%}", f"{float(r['student_log_loss']):.4f}",
          f"{float(r['student_mean_strict_adjacency_rate']):.2%}"] for r in rows])
    print('\n*Top-1 is a mean-trip reference-alignment PROXY, not measured point ground truth.')
    print('Student label coverage: 19,143 / 19,160 points; candidate misses count as errors.')
    print('Provisional CV Top-1 leader: ' + NAMES[run['selected_untuned_model']] + '.')
    print('Random Forest leads CV candidate F1. The two models are very close.\n')
    print('3. SAVED MODEL FILES')
    print_table(['Artifact', 'Size (KiB)'], [[name + '.joblib', f"{(ROOT/'artifacts'/(name+'.joblib')).stat().st_size/1024:.1f}"] for name in NAMES])
    if not args.no_predict:
        print(f'\n4. FRESH CLASSIFIER INFERENCE - STUDENT TRIP {args.trip}')
        preview, scores, block = fresh_predictions(args.trip)
        print('Loaded saved features and trained models; predict_proba ran again now. No HMM.\n')
        print_table(['Model', 'GPS rows', 'Trip Top-1*', 'Fresh seconds'],
            [[NAMES[r.model], r.observations, f'{r.trip_top1_proxy:.2%}', f'{r.fresh_classifier_seconds:.4f}']
             for r in scores.itertuples()])
        print('\nFirst 10 GPS rows: actual predicted arc IDs (-1 means unmatched).')
        print(preview.head(10).to_string(index=False))
        print('\nCandidate feature tensor:', block['X'].shape, '= GPS rows x candidate slots x features')
    figures = comparison_figures()
    figures.append(confusion_figure())
    output = ROOT / 'artifacts/demo_outputs'; output.mkdir(exist_ok=True)
    for filename, figure in zip(['scores.png', 'loss_and_latency.png', 'confusion_matrices.png'], figures):
        figure.savefig(output / filename, dpi=150, bbox_inches='tight')
    print('\nGraphs saved to artifacts/demo_outputs/.')
    print('Bayesian optimisation, HMM evaluation and the physical demo are deferred.')
    if args.plots:
        print('Plot windows are open. Close them to return to the terminal.')
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        print('Cannot show outputs:', error, file=sys.stderr)
        sys.exit(1)
