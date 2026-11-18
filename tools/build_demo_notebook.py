"""Create and execute an output-first walkthrough, without retraining or HMM."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
for name, folder in {'JUPYTER_CONFIG_DIR': 'config', 'JUPYTER_DATA_DIR': 'data',
                     'JUPYTER_RUNTIME_DIR': 'runtime', 'IPYTHONDIR': 'ipython'}.items():
    target = ROOT / '.cache/jupyter' / folder
    target.mkdir(parents=True, exist_ok=True)
    os.environ[name] = str(target)
os.environ['MPLCONFIGDIR'] = str(ROOT / '.cache/matplotlib')


def create():
    import nbformat as nbf
    cells = []
    def md(source):
        cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip()))
    def code(source):
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip()))

    md('''
    # PathSnap: show the actual outputs
    **Open this notebook and scroll through its saved outputs, or use Run All.**
    It loads the completed five-model experiment, displays the data and graphs,
    and runs the trained classifiers again on one student-test trip.
    It does not retrain, tune or run HMM decoding.

    ## Goal
    Show: dataset → raw rows → candidates/features → five-model comparison → actual predictions.
    Pointwise labels are estimated from reference routes, so Top-1 values are **proxies**.
    ''')
    md('''
    ## Setup
    Use the project `.venv` Python kernel. In VS Code, open this notebook, choose
    **Select Kernel → Python Environments → .venv**, then **Run All**.
    The saved figures and tables are already visible without running cells.
    ''')
    code('''
    from pathlib import Path
    import sys, os, json
    root_candidates = [Path.cwd(), Path.cwd().parent]
    ROOT = next(p for p in root_candidates if (p / 'artifacts/model_comparison.csv').exists())
    os.environ['MPLCONFIGDIR'] = str(ROOT / '.cache/matplotlib')
    sys.path.insert(0, str(ROOT / 'src'))
    sys.path.insert(0, str(ROOT / 'tools'))
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display
    from show_outputs import fresh_predictions, comparison_figures, confusion_figure, NAMES
    pd.set_option('display.max_columns', 14)
    pd.set_option('display.width', 150)
    plt.rcParams.update({'figure.dpi': 110, 'font.size': 10})
    TRIP_ID = '00000000'  # Change to any supplied student-test trip.
    print('Ready. Selected student-test trip:', TRIP_ID)
    ''')
    md('''
    ## Steps
    ### 1. Dataset counts and sampling
    Source: the actual audit in `artifacts/eda/summary.json`.
    ''')
    code('''
    summary = json.loads((ROOT / 'artifacts/eda/summary.json').read_text())
    dataset_output = pd.DataFrame([
        {'Partition': name, 'Trips': s['trips'], 'GPS observations': s['gps_observations'],
         'Gaps >30 seconds': s['gaps_gt_30s'], 'Maximum gap (seconds)': s['sampling_quantiles_s']['max'],
         'Reference route breaks': s['reference_breaks']}
        for name, s in summary.items()
    ])
    display(dataset_output)
    ''')
    md('### 2. Show real GPS rows and road arc rows')
    code('''
    from pathsnap.io import DEFAULT_DATA
    trip_path = ROOT / DEFAULT_DATA / 'student_test' / TRIP_ID / (TRIP_ID + '.track')
    gps = pd.read_csv(trip_path, sep=r'\s+', header=None, names=['longitude', 'latitude', 'seconds'])
    arcs = pd.read_csv(trip_path.with_suffix('.arcs'), sep=r'\s+', header=None, names=['from_node_id', 'to_node_id'])
    reference = np.loadtxt(trip_path.with_suffix('.route'), dtype=int, ndmin=1)
    print('GPS observations:', len(gps), '| Directed road arcs:', len(arcs), '| Reference route entries:', len(reference))
    display(gps.head(8))
    display(arcs.head(8).rename_axis('arc_id (zero-based row)'))
    print('First 12 reference route arc IDs:', reference[:12].tolist())
    print('A reference route entry is not a label for the corresponding GPS row.')
    ''')
    md('''
    ### 3. Actual candidate features
    Candidates were generated within 60 metres, capped at eight per GPS row.
    Features contain geometry/motion/network information; no reference labels or IDs are features.
    ''')
    code('''
    from pathsnap.candidates import FEATURES
    with np.load(ROOT / 'data/processed/student_test' / (TRIP_ID + '.npz')) as saved:
        block = {name: saved[name] for name in saved.files}
    print('Feature tensor:', block['X'].shape, '= GPS rows × candidate slots × features')
    observation = 0
    valid = block['valid'][observation]
    candidates_output = pd.DataFrame(block['X'][observation, valid], columns=FEATURES)
    candidates_output.insert(0, 'candidate_arc_id (identifier only)', block['ids'][observation, valid])
    display(candidates_output.round(4))
    ''')
    md('''
    ### 4. Five-model comparison: completed experiment
    Sources: `model_comparison.csv`, `experiment_run.json`, `cv_splits.json`.
    Three group folds hold out entire trips. Fitting uses every fifth training observation;
    validation/student scoring uses all observations. Student Top-1 is averaged equally across trips.
    ''')
    code('''
    comparison = pd.read_csv(ROOT / 'artifacts/model_comparison.csv')
    run = json.loads((ROOT / 'artifacts/experiment_run.json').read_text())
    columns = ['model', 'cv_macro_f1', 'cv_top1_macro_trip_proxy', 'student_macro_f1',
               'student_top1_macro_trip_proxy', 'student_log_loss', 'student_mean_strict_adjacency_rate']
    comparison_output = comparison[columns].copy()
    comparison_output['model'] = comparison_output['model'].map(NAMES)
    comparison_output.columns = ['Model', 'CV candidate Macro F1', 'CV Top-1 proxy',
        'Student candidate Macro F1', 'Student Top-1 proxy', 'Student log loss', 'Student strict adjacency']
    display(comparison_output.round(4))
    print('Fit rows:', f"{run['fit_candidate_rows']:,}", '| Folds:', run['folds'], '| Seed:', run['seed'])
    print('Aligned student GPS points: 19,143 / 19,160. Missing reference candidates count as Top-1 errors.')
    ''')
    md('### 5. Comparison graphs')
    code('''
    score_fig, efficiency_fig = comparison_figures(ROOT)
    display(score_fig)
    display(efficiency_fig)
    plt.close(score_fig)
    plt.close(efficiency_fig)
    ''')
    md('''
    HGB narrowly leads CV Top-1; Random Forest narrowly leads CV candidate Macro F1.
    The differences are small relative to fold variability. Classifier time excludes
    raw loading, candidate generation, maps and network communication.
    ''')
    md('### 6. Actual candidate confusion matrices')
    code('''
    confusion = confusion_figure(ROOT)
    display(confusion)
    plt.close(confusion)
    ''')
    md('''
    These matrices classify **candidate correctness (0/1)** against derived labels.
    They do not classify the GPS points into five model classes.
    ''')
    md('''
    ### 7. Run the saved classifiers now on one trip
    This cell performs a fresh `predict_proba` pass for each trained model on the
    prepared feature tensor. It then displays actual selected arc IDs.
    It does not rerun training or use reference labels as input features.
    ''')
    code('''
    prediction_rows, fresh_scores, prediction_block = fresh_predictions(TRIP_ID, ROOT)
    fresh_output = fresh_scores.copy()
    fresh_output['model'] = fresh_output['model'].map(NAMES)
    display(fresh_output.round(5))
    display(prediction_rows.head(12))
    ''')
    md('''
    ### 8. One observation: candidate probabilities → chosen road
    The selected model's actual candidate scores explain how its highest-scoring
    candidate becomes one predicted road arc for a GPS row.
    ''')
    code('''
    import joblib
    from threadpoolctl import threadpool_limits
    selected = joblib.load(ROOT / 'artifacts/selected.joblib')
    mask = prediction_block['valid'][observation]
    with threadpool_limits(limits=1):
        raw_scores = selected['model'].predict_proba(prediction_block['X'][observation, mask])[:, 1]
    candidate_scores = pd.DataFrame({
        'arc_id': prediction_block['ids'][observation, mask],
        'distance_m': prediction_block['X'][observation, mask, 0],
        'raw_candidate_probability': raw_scores,
        'normalized_score': raw_scores.clip(1e-9) / raw_scores.clip(1e-9).sum()
    }).sort_values('raw_candidate_probability', ascending=False)
    print('Model:', NAMES[run['selected_untuned_model']], '| GPS observation:', observation)
    display(candidate_scores.round(5))
    print('Chosen arc ID:', int(candidate_scores.iloc[0].arc_id))
    ''')
    md('''
    ## Checks
    Verify the five models, whole-trip fold membership, and fresh predictions
    against the saved per-trip measurements.
    ''')
    code('''
    folds = json.loads((ROOT / 'artifacts/cv_splits.json').read_text())
    assert len(comparison) == 5 and len(run['models']) == 5
    for fold in folds:
        assert set(fold['train_trips']).isdisjoint(fold['validation_trips'])
    held_out = [trip for fold in folds for trip in fold['validation_trips']]
    assert len(held_out) == len(set(held_out)) == 80
    recorded = pd.read_csv(ROOT / 'artifacts/classifier_sequence_comparison.csv', dtype={'trip_id': str})
    recorded = recorded[recorded.trip_id == TRIP_ID].set_index('model')
    for row in fresh_scores.itertuples():
        assert np.isclose(row.trip_top1_proxy, recorded.loc[row.model, 'top1_accuracy_proxy'])
        assert row.observations == len(gps)
    print('PASS: five trained models; whole-trip folds; 80 unique held-out trips.')
    print('PASS: fresh predictions match the saved trip scores for all five models.')
    print('No Bayesian optimisation or HMM evaluation has been run.')
    ''')
    md('''
    ## Next steps
    This demonstration stops at model comparison, as requested.
    Change `TRIP_ID` and Run All to show a different student-test trip.
    The original training commands remain in README.md if an evaluator asks you
    to reproduce the experiment; this notebook intentionally avoids retraining
    while presenting outputs.
    ''')
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3 (project .venv)', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11.9'}})
    nbf.validate(notebook)
    return notebook


def main():
    import nbformat
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    target = ROOT / 'notebooks/PathSnap_outputs.ipynb'
    target.parent.mkdir(exist_ok=True)
    notebook = create()
    nbformat.write(notebook, target)
    if args.execute:
        from jupyter_client import KernelManager
        from jupyter_client.kernelspec import KernelSpecManager
        from nbclient import NotebookClient
        kernel_root = ROOT / '.cache/jupyter/kernels'
        kernel_folder = kernel_root / 'pathsnap-demo'; kernel_folder.mkdir(parents=True, exist_ok=True)
        (kernel_folder / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'PathSnap execution', 'language': 'python'}))
        specs = KernelSpecManager(kernel_dirs=[str(kernel_root)], ensure_native_kernel=False)
        manager = KernelManager(kernel_name='pathsnap-demo', kernel_spec_manager=specs)
        client = NotebookClient(notebook, km=manager, timeout=180,
                                resources={'metadata': {'path': str(ROOT)}})
        try:
            client.execute()
        finally:
            nbformat.write(notebook, target)
        nbformat.validate(notebook)
        from nbconvert import HTMLExporter
        page, _ = HTMLExporter(template_name='lab').from_notebook_node(notebook)
        (target.parent / 'PathSnap_outputs.html').write_text(page, encoding='utf-8')
        receipt = {'code_cells_executed': sum(c.cell_type == 'code' for c in notebook.cells),
            'errors': sum(o.output_type == 'error' for c in notebook.cells if c.cell_type == 'code' for o in c.outputs),
            'fresh_classifier_models': 5, 'trip_id': '00000000', 'retrained_models': False, 'hmm_run': False}
        (ROOT / 'artifacts/notebook_execution.json').write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt, indent=2))
    print(target)


if __name__ == '__main__':
    main()
