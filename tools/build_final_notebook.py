"""Execute the final demo notebook using the frozen backend and real artifacts."""
from pathlib import Path
import json
import os
import sys
import textwrap
import re
import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import build_demo_notebook  # also confines Jupyter runtime files to .cache


def main():
    nb = nbformat.read(ROOT / 'notebooks/PathSnap_outputs.ipynb', as_version=4)
    nb.cells[0].source = '''# PathSnap: final backend demonstration
This is an editable Jupyter notebook with actual executed outputs.
The first cells show the original five-model comparison. Later cells show
completed Bayesian optimisation, student-test HMM results, frozen-model
provenance, final evaluator outputs, and a live backend request.
Reference routes remain on this backend laptop. No evaluator data was used
to fit, tune or select the model. Pointwise scores remain alignment proxies.'''
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.execution_count = None
            cell.outputs = []
            cell.source = cell.source.replace("print('No Bayesian optimisation or HMM evaluation has been run.')",
                "print('Original five-model comparison checked. Completed later stages appear below.')")
        elif '## Next steps' in cell.source:
            cell.source = '## Completed optimisation and sequence decoding\nThe following outputs were generated after the original five-model comparison.'
    def md(text):
        nb.cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip()))
    def code(text):
        nb.cells.append(nbformat.v4.new_code_cell(textwrap.dedent(text).strip()))
    md('''
    ### 9. Bayesian optimisation: actual 15-trial search
    Three random startup trials and twelve adaptive TPE trials used the same
    three training-trip folds. No evaluator examples entered the search.
    ''')
    code('''
    tuning = json.loads((ROOT / 'artifacts/tuning_summary.json').read_text())
    trials = pd.read_csv(ROOT / 'artifacts/optimization_trials.csv')
    display(trials[['number', 'value', 'state']])
    print('Best hyperparameters:', tuning['best_parameters'])
    print('Best CV mean-trip Top-1 proxy:', round(tuning['best_cv_top1_macro_trip_proxy'], 6))
    print('Tuned student candidate Macro F1:', round(tuning['student_test']['macro_f1'], 6))
    print('Tuned student mean-trip Top-1 proxy:', round(tuning['student_test']['top1_macro_trip_proxy'], 6))
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(trials.number, trials.value, 'o-', label='Completed trial')
    ax.plot(trials.number, trials.value.cummax(), '--', label='Best so far')
    ax.axvline(2.5, color='#64748b', linestyle=':', label='End of random startup')
    ax.set(xlabel='Trial number (zero-based)', ylabel='Held-out-trip Top-1 proxy', title='Actual Bayesian search history')
    ax.legend(); ax.grid(alpha=.2); plt.show()
    ''')
    md('''
    ### 10. Classifier versus HMM/Viterbi on all student-test trips
    Hidden states are candidate directed road arcs. The decoder combines
    classifier scores with feasible directed-road transitions. Gaps, missing
    candidates and impossible transitions create explicit route breaks.
    ''')
    code('''
    routes = pd.read_csv(ROOT / 'artifacts/route_evaluation.csv', dtype={'trip_id': str})
    selected_routes = routes[routes.model == 'selected']
    summary = selected_routes.groupby('stage')[['top1_accuracy_proxy', 'strict_adjacency_rate', 'arc_set_f1', 'sampled_arc_edit_similarity']].mean()
    display(summary.round(5))
    comparison_by_trip = selected_routes.pivot(index='trip_id', columns='stage', values='top1_accuracy_proxy')
    display(comparison_by_trip.round(5))
    comparison_by_trip.plot.bar(figsize=(11, 4), color=['#e76f00', '#7c3aed'], ylim=(0, 1), rot=35)
    plt.title('Actual classifier/HMM Top-1 proxy on each student-test trip')
    plt.ylabel('Aligned-point Top-1 proxy'); plt.tight_layout(); plt.show()
    ''')
    md('''
    ### 11. Frozen model and independent verification
    The manifest was written before evaluator inputs were imported.
    Runtime checks verify its model and inference-source checksums.
    ''')
    code('''
    frozen = json.loads((ROOT / 'artifacts/final/model_manifest.json').read_text())
    verified = json.loads((ROOT / 'artifacts/development_verification.json').read_text())
    display(pd.DataFrame([{'model': frozen['model_name'], 'frozen_utc': frozen['frozen_utc'],
                           'trained_partition': frozen['trained_partition'], 'model_sha256': frozen['model_sha256']}]))
    print('Directed route-part connectivity verified:', verified['reconstructed_parts_are_directed_connected'])
    print('Whole-trip CV verification:', verified['unique_validation_trips'], 'unique held-out training trips')
    ''')
    md('''
    ### 12. Final evaluator run: all ten trips through HTTP
    These results were obtained only after the model/configuration freeze.
    They were not used to alter the model, thresholds or decoder. Scores are
    reference-alignment proxies, including the weaker result on trip 00000015.
    ''')
    code('''
    final_receipt = json.loads((ROOT / 'artifacts/evaluator_api_smoke.json').read_text())
    final_rows = pd.DataFrame([{'trip_id': r['trip_id'], 'HTTP': r['http_status'],
        'GPS observations': r['observations'], 'classifier Top-1 proxy': r['metrics']['classifier']['top1_accuracy_proxy'],
        'HMM Top-1 proxy': r['metrics']['hmm']['top1_accuracy_proxy'],
        'alignment coverage': r['metrics']['hmm']['alignment_coverage'], 'HTTP seconds': r['http_seconds']}
        for r in final_receipt['results']])
    display(final_rows.round(5))
    assert len(final_rows) == 10 and final_rows.HTTP.eq(200).all()
    assert final_receipt['model_sha256'] == frozen['model_sha256']
    print('PASS: all ten evaluator requests succeeded with the frozen model.')
    ''')
    md('''
    ### 13. Send a fresh request to this backend laptop
    Keep `START_BACKEND.cmd` running. Change `EVALUATOR_TRIP_ID` to any ID
    returned by `/trips`, then rerun this cell. It performs actual inference.
    ''')
    code('''
    import urllib.request
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    API_BASE = 'http://127.0.0.1:8766'
    with opener.open(API_BASE + '/health', timeout=20) as response:
        health = json.load(response)
    with opener.open(API_BASE + '/trips', timeout=20) as response:
        evaluator_ids = json.load(response)['trips']
    EVALUATOR_TRIP_ID = evaluator_ids[0]  # Choose any of the ten IDs here.
    request = urllib.request.Request(API_BASE + '/match',
        data=json.dumps({'trip_id': EVALUATOR_TRIP_ID}).encode(),
        headers={'Content-Type': 'application/json'}, method='POST')
    with opener.open(request, timeout=300) as response:
        live_result = json.load(response)
    print('HTTP 200 | Request:', live_result['request_id'])
    print('Trip:', live_result['trip_id'], '| GPS observations:', live_result['observations'])
    print('Frozen model SHA256:', live_result['model_sha256'])
    display(pd.DataFrame([{'stage': stage, **{key: live_result['metrics'][stage][key]
        for key in ['top1_accuracy_proxy', 'alignment_coverage', 'strict_adjacency_rate', 'arc_set_f1']}}
        for stage in ['classifier', 'hmm']]).round(5))
    display(pd.DataFrame({'GPS row': range(12), 'classifier arc': live_result['classifier_arc_ids'][:12],
                          'HMM arc': live_result['prediction_arc_ids'][:12]}))
    ''')
    md('''
    ### 14. Live Folium map: GPS, roads and predicted route
    This map comes from the backend request above. Reference route labels
    are not included. The GPS/road layers use locally hosted Leaflet assets;
    only the optional OpenStreetMap background tiles need internet.
    ''')
    code('''
    from IPython.display import IFrame
    display(IFrame(src=API_BASE + live_result['map_path'], width='100%', height=580))
    ''')
    md('''
    ### 15. Development Folium map with reference layer
    This separate local view uses the student-test trip, as required for the
    technical report. It is not exposed in the evaluator API/UI.
    ''')
    code('''
    from IPython.display import HTML
    import html
    development_html = (ROOT / 'artifacts/maps/00000000.html').read_text(encoding='utf-8')
    development_html = development_html.replace('../../frontend/vendor/leaflet', API_BASE + '/assets/leaflet')
    display(HTML('<iframe title="Student development Folium map" style="width:100%;height:580px;border:0" srcdoc="'
                 + html.escape(development_html, quote=True) + '"></iframe>'))
    ''')
    md('''
    ## Demonstration handoff
    This machine hosts the model, HMM, evaluator files and API.
    Laptop 2 can open `http://10.1.24.198:8766/ui/` on the same network.
    For a separately served frontend, use the supplied frontend ZIP.
    Backend request IDs and aggregate scores are in `artifacts/backend.log`.
    A physical second-laptop rehearsal and receiving-terminal screenshots
    still need to be captured by the team; local HTTP checks do not prove them.
    ''')
    target = ROOT / 'notebooks/PathSnap_final_demo.ipynb'
    nbformat.write(nb, target)
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient
    kernel_root = ROOT / '.cache/jupyter/kernels'
    manager = KernelManager(kernel_name='pathsnap-demo',
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)], ensure_native_kernel=False))
    try:
        NotebookClient(nb, km=manager, timeout=300,
            resources={'metadata': {'path': str(ROOT)}}).execute()
    finally:
        nbformat.write(nb, target)
    nbformat.validate(nb)
    live_stdout = next((o.text for c in nb.cells if c.cell_type == 'code'
        for o in c.outputs if o.output_type == 'stream' and 'HTTP 200 | Request:' in o.text), '')
    request_match = re.search(r'HTTP 200 \| Request:\s*([0-9a-f]{32})', live_stdout)
    receipt = {'code_cells_present': sum(c.cell_type == 'code' for c in nb.cells),
        'code_cells_executed': sum(c.cell_type == 'code' and c.execution_count is not None for c in nb.cells),
        'errors': sum(o.output_type == 'error' for c in nb.cells if c.cell_type == 'code' for o in c.outputs),
        'fresh_evaluator_request_id': request_match.group(1) if request_match else None,
        'model_sha256': json.loads((ROOT / 'artifacts/final/model_manifest.json').read_text())['model_sha256']}
    (ROOT / 'artifacts/final_notebook_execution.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    print(target)


if __name__ == '__main__':
    main()
