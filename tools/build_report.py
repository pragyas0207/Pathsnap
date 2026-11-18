"""Build an honest Markdown/PDF progress report solely from executed artifacts."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def num(value):
    return 'Not run' if value is None else f'{value:.4f}'


def plots(comparison, routes, output):
    names = comparison.model.tolist()
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    axes[0].bar(x, comparison.cv_macro_f1, color='#196b53')
    axes[0].set(title='Trip-group CV: candidate Macro F1', ylabel='Macro F1', ylim=(0, 1))
    axes[1].bar(x - .18, comparison.student_top1_macro_trip_proxy, width=.36, label='Top-1 (macro trip, proxy)', color='#2563eb')
    axes[1].bar(x + .18, comparison.student_macro_f1, width=.36, label='Candidate Macro F1', color='#196b53')
    axes[1].set(title='Student testing: all 10 trips', ylim=(0, 1))
    axes[1].legend(fontsize=7)
    axes[2].bar(x, comparison.student_log_loss, color='#e76f00')
    axes[2].set(title='Student candidate binary log loss', ylabel='Log loss (lower is better)', ylim=(0, None))
    for ax in axes:
        ax.set_xticks(x, names, rotation=35, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=.15)
    fig.tight_layout(); fig.savefig(output / 'model_comparison.png', dpi=160); plt.close(fig)
    if routes is not None:
        selected = routes[routes.model == 'selected']
        if len(selected):
            pivot = selected.pivot(index='trip_id', columns='stage', values='top1_accuracy_proxy')
            pivot.plot.bar(color=['#e76f00', '#7c3aed'], figsize=(10, 4), ylim=(0, 1), rot=35)
            plt.ylabel('Aligned-point Top-1 proxy'); plt.title('Classifier vs HMM: each student-test trip')
            plt.tight_layout(); plt.savefig(output / 'hmm_comparison.png', dpi=160); plt.close()


MODEL_EXPLANATIONS = {
    'logistic': ('Linear probabilistic baseline', 'Scaled logistic negative log likelihood with L2 regularisation; initial C=1 and max_iter=500.',
                 'Interpretable and inexpensive; a linear boundary cannot express every interaction between distance, heading and junction structure.'),
    'naive_bayes': ('Gaussian probabilistic baseline', 'Class-conditional Gaussian likelihood and empirical priors; var_smoothing=1e-9.',
                    'Fast likelihood baseline; distance/rank and motion features violate conditional independence and can produce overconfident probabilities.'),
    'decision_tree': ('Single nonlinear tree', 'Gini impurity split criterion; max_depth=12, min_samples_leaf=20.',
                      'Readable nonlinear rules; a single tree can vary substantially with weak-label noise and its piecewise probabilities can harm log loss.'),
    'random_forest': ('Bagged tree ensemble', '80 trees using Gini splits, max_depth=16, min_samples_leaf=10, max_features=sqrt, one worker.',
                      'Averages nonlinear rules and reduces tree variance; more storage and compute than the linear baselines, with no guarantee of well-calibrated probabilities.'),
    'hist_boost': ('Histogram gradient boosting', 'Binary log loss; 100 iterations, 31 leaves, learning_rate=0.1, l2_regularization=1, min_samples_leaf=30; early stopping disabled.',
                   'Learns feature interactions efficiently; can fit weak-label artefacts and requires independent validation rather than trusting training fit.')
}


def build(root, comparison_only=False):
    artifacts = root / 'artifacts'; docs = root / 'docs'
    figures = artifacts / 'figures'; figures.mkdir(parents=True, exist_ok=True)
    eda = read_json(artifacts / 'eda/summary.json')
    manifest = read_json(root / 'data/processed/manifest.json')
    run = read_json(artifacts / 'experiment_run.json')
    tuning = read_json(artifacts / 'tuning_summary.json')
    api = read_json(artifacts / 'api_smoke.json')
    tests = read_json(artifacts / 'test_receipt.json')
    inventory = read_json(root / 'data/raw/inventory.json')
    final_summary = read_json(artifacts / 'final_summary.json')
    evaluator = pd.read_csv(artifacts / 'evaluator_results.csv', dtype={'trip_id': str}) if (artifacts / 'evaluator_results.csv').exists() else None
    frozen = read_json(artifacts / 'final/model_manifest.json')
    comparison = pd.read_csv(artifacts / 'model_comparison.csv') if (artifacts / 'model_comparison.csv').exists() else None
    classifier_sequences = pd.read_csv(artifacts / 'classifier_sequence_comparison.csv', dtype={'trip_id': str}) if (artifacts / 'classifier_sequence_comparison.csv').exists() else None
    routes = pd.read_csv(artifacts / 'route_evaluation.csv', dtype={'trip_id': str}) if (artifacts / 'route_evaluation.csv').exists() else None
    if inventory:
        shutil.copyfile(root / 'data/raw/inventory.json', artifacts / 'source_inventory.json')
    if manifest:
        shutil.copyfile(root / 'data/processed/manifest.json', artifacts / 'processed_manifest.json')
    text = ['# PathSnap: technical report',
        'Generated from executed artifacts on ' + datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC') + '.',
        '**AI-assisted technical report, for team review. Actual development and final-only backend evaluations were executed. A physical second-laptop rehearsal and receiving-terminal screenshot remain team handoff tasks.** Team member details and independent review must be added before submission.',
        '## 1. Requirements and experimental scope',
        'Both original ZIP archives and all five guideline PDFs were inspected. The assignment requires five trained models, group-aware evaluation and Bayesian optimisation, HMM/Viterbi comparison, Folium maps, a FastAPI/UI split across two laptops, a public source repository and one PDF. Exact requirement-to-page links are retained in docs/REQUIREMENTS.md. The evaluator ZIP was supplied later, kept separate, and imported only after the model/configuration/source freeze.',
        'The experiment uses all 80 training trips and all 10 student-test trips. Five models use the same three GroupKFold splits. Fitting samples every fifth observation per training trip; validation and student testing use all observations. This is an initial benchmark, not the full-observation fitting ablation. The same candidate rows belonging to an observation stay together, and all observations from a trip stay in one fold.']
    text += ['## 2. Dataset understanding and exploration']
    if eda:
        text.append(table(['Partition', 'Trips', 'GPS rows', 'Gaps >30 s', 'Reference breaks'],
            [[p, s['trips'], s['gps_observations'], s['gaps_gt_30s'], s['reference_breaks']] for p, s in eda.items()]))
        text.append('All supplied GPS numeric cells are finite and all timestamps increase strictly. Training intervals are 1 s for '
            + f"{eda['train']['one_second_step_fraction']:.2%}" + ' of steps; the maximum gap is '
            + str(eda['train']['sampling_quantiles_s']['max']) + ' s. Student-test maximum gap is '
            + str(eda['student_test']['sampling_quantiles_s']['max']) + ' s. Connectivity, geographic extents, file sizes, hashes and sampling statistics are recorded per trip in artifacts/eda/trip_inventory.csv. Component counts describe each trip-specific network, not a unique global map.')
        text.append('Zero-length arcs: ' + str(eda['train']['zero_length_arcs']) + ' across training network files and '
            + str(eda['student_test']['zero_length_arcs']) + ' across student-test files. No isolated nodes were found. Zero-length geometry is numerically guarded; routing uses a tiny positive edge weight rather than inventing length. Reference breaks mean consecutive supplied route arcs do not share the expected directed endpoint.')
        text.append('![Dataset exploration](../artifacts/eda/eda.png)')
    text += ['### Preprocessing, reference alignment and limitations',
        'Headerless whitespace tables are parsed strictly: track=[longitude, latitude, seconds], nodes=[longitude, latitude], arcs=[from node row ID, to node row ID], route=[arc row ID]. IDs are zero-based. Invalid coordinates, invalid IDs, non-finite data and non-increasing timestamps fail explicitly. Data is not silently reordered, imputed or snapped to a reference.',
        'Distances use a per-trip UTM projection with longitude/latitude axis order explicit. Road geometry is the straight segment between supplied endpoints; the files do not provide speed limits or detailed curved polylines. An exact STRtree radius query generates up to eight candidates within 60 m, sorted by distance then ID. No nearest-road fallback is fabricated for empty sets.',
        'The supplied route is an ordered list of traversed arcs, not a label for each GPS row. Offline alignment minimizes cumulative squared point-to-segment distance over non-decreasing reference route positions: D[t,j] = distance(t,j)^2 + min(D[t-1,h] for h<=j). Repeated arc IDs remain distinct occurrences. States can stay or advance; endpoints are not forced. Residuals above 100 m are unknown. This geometric weak supervision can bias results toward geometry and misalign loops/stops/disconnected references. It is not independently measured point truth or an organiser-specified scorer.']
    if manifest:
        rows = []
        for p in ('train', 'student_test'):
            sub = [t for t in manifest['trips'] if t['partition'] == p]
            total = sum(t['observations'] for t in sub); aligned = sum(t['aligned'] for t in sub)
            rows.append([p, aligned, f'{aligned/total:.2%}', sum(t['candidate_misses'] for t in sub), sum(t['no_candidates'] for t in sub)])
        text.append(table(['Partition', 'Aligned GPS rows', 'Label coverage', 'Candidate misses*', 'No candidates'], rows))
        text.append('*Candidate misses use only aligned observations. Missing candidate sets and absent reference candidates count as Top-1 errors on eligible rows; unknown-reference rows are excluded and coverage is shown. Train trip 00000054 has 19,313 points, only 2,858 usable aligned labels, and 16,178 empty candidate sets at the initial radius. It remains in the inventory and folds. This issue warrants candidate-radius and alignment sensitivity experiments; its physical cause has not been established.')
    text += ['### Features',
        'The 12 features are distance in metres, heading cosine, heading-known flag, log segment length, projected fraction, capped GPS speed, capped previous sampling interval, capped GPS step length, distance rank, candidate count, and outdegree of the two endpoints. Heading uses the previous observation only when movement is at least 3 m and the time interval is at most 30 s. First/stationary/gap observations use neutral heading with the flag unset. IDs, absolute coordinates, reference membership and aligned-label information are not features. Logistic scaling is fitted inside each training fold.',
        '## 3. Model experimentation and comparison',
        'Candidate binary Macro F1 averages positive and negative candidate classes; binary log loss uses the raw probability of candidate correctness. Observation Top-1 selects the highest-scoring available candidate. Top-1 metrics are reference-alignment proxies. The primary selection metric is validation Top-1 averaged equally across trips, then averaged across folds. Student testing is development-stage evidence, not an untouched final benchmark. No model selection uses evaluator data.']
    if comparison is not None:
        text.append(table(['Model', 'CV Macro F1', 'CV Top-1 proxy', 'CV fold SD'],
            [[r.model, num(r.cv_macro_f1), num(r.cv_top1_macro_trip_proxy), num(r.cv_top1_std)] for r in comparison.itertuples()]))
        text.append(table(['Model', 'Student Macro F1', 'Student Top-1 proxy*', 'Log loss'],
            [[r.model, num(r.student_macro_f1), num(r.student_top1_macro_trip_proxy), num(r.student_log_loss)] for r in comparison.itertuples()]))
        text.append('*Student Top-1 here is an equal-trip average. The comparison CSV also retains the pooled observation average. Fold SD is descriptive variability across three folds, not a confidence interval.')
        text.append(table(['Model', 'Refit seconds', 'Student scoring seconds**'],
            [[r.model, f'{r.refit_seconds:.3f}', f'{r.student_scoring_seconds:.3f}'] for r in comparison.itertuples()]))
        text.append('**Scoring routine timing includes candidate probability prediction, binary diagnostics and a second probability pass for binary metrics on precomputed features. It excludes raw parsing, projection, candidate generation, HMM, map generation and network latency. End-to-end HTTP timing belongs to a separate deployment test. These are measured wall times on this host, not hardware-independent latency promises.')
        plots(comparison, routes, figures)
        text.append('![Model comparisons from real runs](../artifacts/figures/model_comparison.png)')
        if classifier_sequences is not None:
            means = classifier_sequences.groupby('model')[['strict_adjacency_rate', 'arc_set_f1', 'sampled_arc_edit_similarity', 'classifier_inference_seconds']].mean()
            text += ['### Classifier-only route diagnostics',
                table(['Model', 'Strict adjacency', 'Arc-set F1', 'Sampled edit', 'Inference s/trip'],
                    [[name] + [num(value) for value in values] for name, values in means.iterrows()]),
                'These are equal-trip averages across all ten student-test trips, using classifier-only predictions. Strict adjacency means consecutive non-gap observations select the same or directly adjoining directed arcs; unmatched pairs count as failures. It can penalize valid multi-arc travel between samples. Arc-set F1 ignores order. Sampled edit similarity compares collapsed predicted arcs with the supplied reference sequence and can penalize unsampled intermediate roads. Inference here is a single classifier pass and argmax on prepared candidates; it excludes preprocessing, HMM and maps. HMM results are reported separately in section 5.']
    else:
        text.append('Model comparison has not run. No values are supplied.')
    if run:
        text.append('Run settings: seed ' + str(run['seed']) + ', training candidate rows used for fitting: ' + str(run['fit_candidate_rows'])
            + '. Selected untuned model: ' + str(run.get('selected_untuned_model', 'selection incomplete')) + '. Full parameters, versions and metrics are in artifacts/experiment_run.json; exact fold trip membership is in artifacts/cv_splits.json.')
        text.append('Training-partition scores below use all eligible observations, including rows not sampled for fitting; they are not pure resubstitution scores.')
        for name, entry in run['models'].items():
            family, objective, rationale = MODEL_EXPLANATIONS[name]
            s, tr = entry['student_test'], entry['training_resubstitution']
            cv = np.mean([f['macro_f1'] for f in entry['cv']])
            gap = tr['macro_f1'] - cv
            text += ['### ' + name + ': ' + family, objective + ' ' + rationale,
                f"Measured candidate Macro F1: training partition {tr['macro_f1']:.4f}, held-out-trip CV {cv:.4f}, student test {s['macro_f1']:.4f}. Training-minus-CV gap is {gap:+.4f}. "
                + ('The positive gap suggests a generalisation concern; it does not alone prove overfitting, given different trip difficulty and weak labels.' if gap > .02 else 'This comparison alone does not establish substantial overfitting or underfitting; inspect trip-level errors and label quality.'),
                f"Student Top-1 proxy (macro trip) {s['top1_macro_trip_proxy']:.4f}, candidate Macro precision {s['macro_precision']:.4f}, recall {s['macro_recall']:.4f}, log loss {s['log_loss']:.4f}. "
                + 'Confusion matrix (rows=true [0,1], columns=predicted [0,1]): ' + str(s['confusion_matrix']) + '.']
    comparison_end = len(text)
    text += ['## 4. Bayesian hyperparameter optimisation']
    if tuning:
        text.append('Optuna TPE evaluated ' + str(tuning['trials']) + ' configurations: 3 random startup trials and '
            + str(tuning['adaptive_tpe_trials']) + ' subsequent adaptive trials. Objective: the same training-trip CV macro-trip Top-1 proxy as initial selection. No student labels enter the objective. The trial record is retained in artifacts/optimization_trials.csv and optimization.sqlite3. The search implementation and ranges are in src/pathsnap/models.py (search_space).')
        text.append('Best parameters: ' + json.dumps(tuning['best_parameters']) + '. Search wall time: '
            + f"{tuning['optimization_seconds']:.2f}" + ' s; best-trial refit: ' + f"{tuning['refit_seconds']:.2f}" + ' s.')
        text.append('HGB search space: max_iter integer 60-160 in steps of 20; max_leaf_nodes integer 7-63; learning_rate log-uniform 0.02-0.25; l2_regularization log-uniform 0.01-20; min_samples_leaf log-scaled integer 10-100. These ranges vary iteration budget, tree complexity, update size and regularisation. They were fixed before final evaluator import.')
        trials = pd.read_csv(artifacts / 'optimization_trials.csv')
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(trials.number, trials.value, 'o-', label='Trial CV objective')
        ax.plot(trials.number, trials.value.cummax(), '--', label='Best so far')
        ax.axvline(2.5, color='#64748b', linestyle=':', label='Random startup ends')
        ax.set(xlabel='Trial number (zero-based)', ylabel='CV mean-trip Top-1 proxy', title='Actual 15-trial TPE optimisation history')
        ax.grid(alpha=.2); ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(figures / 'tuning_history.png', dpi=160); plt.close(fig)
        text.append('![Actual TPE search history](../artifacts/figures/tuning_history.png)')
        baseline = run['models'][tuning['model']]
        text.append(table(['Configuration', 'CV Macro F1', 'CV Top-1 proxy', 'Student F1', 'Student Top-1 proxy', 'Log loss'], [
            ['Untuned', num(np.mean([f['macro_f1'] for f in baseline['cv']])), num(np.mean([f['top1_macro_trip_proxy'] for f in baseline['cv']])), num(baseline['student_test']['macro_f1']), num(baseline['student_test']['top1_macro_trip_proxy']), num(baseline['student_test']['log_loss'])],
            ['Tuned', num(np.mean([f['macro_f1'] for f in tuning['best_fold_results']])), num(tuning['best_cv_top1_macro_trip_proxy']), num(tuning['student_test']['macro_f1']), num(tuning['student_test']['top1_macro_trip_proxy']), num(tuning['student_test']['log_loss'])]]))
        text.append('Tuned student scoring routine time: ' + f"{tuning['student_test']['scoring_seconds']:.3f}" + ' s. The tuned model was selected because it improved the training-trip CV selection objective. A best-of-trials CV score is selection-biased; it is not nested CV or a final generalisation estimate. More trials and multiple seeds remain useful follow-ups.')
    else:
        text.append('Optimisation has not completed. No tuned result is claimed.')
    text += ['## 5. HMM, Viterbi and route reconstruction',
        'Hidden states are candidate directed arc IDs at each observation. Emission potentials are the selected classifier binary scores normalized over that observation\'s candidates, floored for numerical stability. This is a discriminative HMM-style decoder, not a calibrated generative sensor model. Transition distance combines the remaining previous arc, a directed shortest path between endpoint nodes and the next arc\'s projected prefix. Staying on the same directed arc allows up to 5 m of projection jitter. Feasibility requires road distance <= 60 m/s * elapsed seconds + 30 m.',
        'Feasible transition potentials exp(-abs(road_distance - GPS_displacement)/30 m) are row-normalized over next candidates. Viterbi maximizes log emission plus previous score plus log transition, retaining a backpointer for every candidate. It backtracks each uninterrupted block. Gaps over 30 s, empty sets, and steps with no feasible transition restart decoding explicitly; empty sets return -1. Directed shortest paths fill intermediate arcs within route parts; disconnected/reset parts stay separate.',
        'Strict adjacency counts non-gap consecutive observations assigned to the same arc or directly adjoining arcs, including unmatched pairs as failures. It is deliberately stricter than network reachability and can penalize valid multi-arc motion between observations. Arc-set F1 ignores order; sampled-arc edit similarity preserves order after collapsing repeated predictions but penalizes intermediate roads without GPS samples. These diagnostics are not independent physical ground truth.']
    if routes is not None:
        selected = routes[routes.model == 'selected']
        summary = selected.groupby('stage')[['top1_accuracy_proxy', 'strict_adjacency_rate', 'arc_set_f1', 'sampled_arc_edit_similarity', 'score_decode_seconds']].mean()
        text.append(table(['Stage', 'Top-1 proxy*', 'Strict adjacency*', 'Arc-set F1*', 'Sampled edit*'],
            [[stage] + [num(v) for v in row[:4]] for stage, row in summary.iterrows()]))
        text.append('*Equal-trip averages across student-test trips. Per-trip rows, candidate misses, resets and score/decode timings are in artifacts/route_evaluation.csv. Score/decode timing excludes raw input/feature generation and map rendering.')
        pivot = selected.pivot(index='trip_id', columns='stage', values='top1_accuracy_proxy')
        if {'classifier', 'hmm'}.issubset(pivot.columns):
            delta = pivot.hmm - pivot.classifier
            text.append(f'HMM Top-1 proxy improved on {(delta > 0).sum()} trips, worsened on {(delta < 0).sum()} trips and tied on {(delta == 0).sum()} trips. '
                + 'The mean change was ' + f'{delta.mean():+.4f}' + '; an improvement in continuity does not imply an improvement in pointwise agreement.')
            text.append('![Classifier and HMM comparison](../artifacts/figures/hmm_comparison.png)')
    else:
        text.append('Route evaluation has not completed. No HMM improvement is claimed.')
    text += ['## 6. Folium geographical visualisation',
        'The offline map in artifacts/maps/00000000.html is generated from actual input and predictions, with blue GPS dots, grey road geometry, green development reference arcs, orange classifier arcs and purple reconstructed HMM arcs. Layer controls and fit bounds support comparison. Predicted road segments are drawn individually; route breaks are not joined with invented straight lines. API maps omit reference layers.']
    if (artifacts / 'screenshots/folium.jpg').exists():
        text.append('![Actual Folium development map with student reference layer](../artifacts/screenshots/folium.jpg)')
    else:
        text.append('A rendered Folium screenshot has not yet been captured. The final submission requires an actual screenshot, not a placeholder.')
    text += ['## 7. API, frontend and deployment',
        'FastAPI exposes GET /health, GET /trips, POST /match and a reference-free map route. The frontend runs separately and sends a selected trip ID. Backend inference accepts only track, nodes and arcs; scoring opens the reference after prediction completes. Request IDs and measured stage times are returned and logged. Missing models return 503, unknown IDs 404, invalid input 422, and concurrent inference 429. The browser origin is explicitly permitted through CORS. No predictions are hardcoded.',
        'The demo runbook explains the two-laptop LAN address, localhost distinction, firewall/CORS checks, evaluator-only directory and genuine evidence capture. Leaflet JS/CSS/images are self-hosted; only optional basemap tiles need internet. The backend serves the UI at /ui/ and binds an exclusive Windows socket on port 8766 to avoid sharing a port with another local service. The evaluator directory is separate from all development loaders.']
    if api:
        results = api['results']
        text.append(f"Local loopback HTTP smoke test recorded {len(results)} trips and {sum(r['http_status'] == 200 for r in results)} successful HTTP 200 responses. "
            + 'Each response length matched the input GPS count, and each served map was checked for absence of a reference layer. Scope: ' + api['scope'] + '. Receipt: artifacts/api_smoke.json; server logs: artifacts/development_backend.log.')
    else:
        text.append('A real HTTP smoke run has not completed. Unit tests alone do not establish application deployment.')
    if tests:
        text.append('Automated test receipt: ' + json.dumps(tests) + '. Tests cover segment geometry, missing candidates and denominator handling, monotone labels, graph direction, resets, intermediate road reconstruction and API reference separation.')
    for filename, caption in [('frontend.jpg', 'Actual local frontend after evaluator inference, with reference-free map'), ('api-response.jpg', 'Actual API response displayed in the browser; receiving-terminal capture is still required')]:
        if (artifacts / 'screenshots' / filename).exists():
            text.append('![' + caption + '](../artifacts/screenshots/' + filename + ')')
    if final_summary and evaluator is not None:
        text += ['### Final-only evaluator results after model freeze',
            'All ten evaluator trips were inferred through the HTTP backend after the model/source freeze. References were used only for server-side scoring after prediction; final outcomes were not used for model selection, hyperparameters, candidate settings or decoder choices.',
            table(['Trip', 'GPS rows', 'Classifier Top-1', 'HMM Top-1', 'Coverage', 'HTTP s'],
                [[r.trip_id, r.gps_observations, num(r.classifier_top1_proxy), num(r.hmm_top1_proxy), num(r.alignment_coverage), f'{r.http_seconds:.3f}'] for r in evaluator.itertuples()]),
            'All Top-1 values in this table are reference-alignment proxies. Final evaluation covered '
            + str(final_summary['gps_observations']) + ' observations. Equal-trip HMM Top-1 proxy is '
            + f"{final_summary['hmm_top1_macro_trip_proxy']:.4f}" + '; pooled observation HMM Top-1 proxy is '
            + f"{final_summary['hmm_top1_pooled_proxy']:.4f}" + '. These differ because trip lengths and difficulty differ. The 6,854-point trip 00000015 is weaker (HMM proxy 0.4803) and remains included. Its cause has not been established and no final-data-driven changes were made.',
            'Final strict adjacency (equal-trip average) is '
            + f"{final_summary['classifier_strict_adjacency_macro_trip']:.4f}" + ' classifier-only and '
            + f"{final_summary['hmm_strict_adjacency_macro_trip']:.4f}" + ' after HMM. Mean/max HTTP times are '
            + f"{final_summary['mean_http_seconds']:.3f}/{final_summary['max_http_seconds']:.3f}" + ' s on this host; physical network latency on Laptop 2 was not measured. All ten returned HTTP 200 with valid output lengths/arc IDs, connected reconstructed parts and reference-free maps.']
        if frozen:
            text.append('Model frozen at ' + frozen['frozen_utc'] + '; SHA-256 ' + frozen['model_sha256']
                + '. Runtime checks require this model and its inference-source hashes to match. The import receipt timestamps the later evaluator provisioning. Evidence: artifacts/final/model_manifest.json, development_verification.json, evaluator_api_smoke.json, evaluator_results.csv and final_summary.json; private input hashes remain under data/evaluation_server/.')
        axes = evaluator.set_index('trip_id')[['classifier_top1_proxy', 'hmm_top1_proxy']].plot.bar(
            figsize=(11,4), color=['#e76f00','#7c3aed'], ylim=(0,1), rot=35)
        axes.set(title='Final-only evaluator Top-1 proxies: every supplied trip', ylabel='Aligned-point Top-1 proxy')
        plt.tight_layout(); plt.savefig(figures / 'evaluator_comparison.png',dpi=160); plt.close()
        text.append('![Actual final evaluator results, including the weaker trip](../artifacts/figures/evaluator_comparison.png)')
        text.append('The backend health endpoint was reached through both loopback and its own Wi-Fi address 10.1.24.198:8766. The final Jupyter notebook executed with real backend responses and saved outputs. These same-host checks are distinct from a physical second-laptop rehearsal. Actual receiving-terminal and second-laptop screenshots remain to be captured by the team; a browser API screenshot is not represented as terminal evidence.')
    limits_start = len(text)
    text += ['## 8. Limits and work before submission',
        'Confirm the official scoring convention and point-label availability. Investigate weak-label errors, candidate-radius/cap sensitivity, stops/loops and reset-heavy trips. Repeat fitting at stride 1 and consider geographic holdout and multiple seeds. The current trip split does not guarantee unseen roads. TPE uses the same CV folds for selection, not nested CV. No calibration experiment is claimed. Unknown-reference rows are not silently scored as correct.',
        'Complete a physical two-laptop rehearsal using the supplied ten choices. Capture the actual receiving-laptop terminal and second-laptop UI. Add team/member details, independently verify interpretations and disclose assistance as required. Publish the source and trusted model/configuration to the team\'s chosen public repository. The source/frontend release archives are supplied; this task did not publish a repository or claim a physical second-laptop demonstration.',
        '## 9. Provenance and references']
    if inventory:
        text.append(table(['Archive', 'SHA-256', 'Files'], [[i['archive'], i['sha256'], len(i['files'])] for i in inventory['archives']]))
    text.append('Assignment sources: Main Guidelines p.2; Dataset Exploration pp.1-4; Models and Training Guidelines pp.1-6; Deliverables pp.1-7; Assessment and Plagiarism Policy pp.1-3. The originals and page-bounded extractions were inspected; see docs/REQUIREMENTS.md for the claim-by-claim map.')
    text.append('External primary sources: [Newson and Krumm (2009)](https://www.microsoft.com/en-us/research/publication/hidden-markov-map-matching-noise-sparseness/); [scikit-learn GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html); [Shapely STRtree](https://shapely.readthedocs.io/en/stable/strtree.html); [Optuna TPESampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html); [FastAPI CORS](https://fastapi.tiangolo.com/tutorial/cors/); [Folium user guide](https://python-visualization.github.io/folium/latest/user_guide.html). These support design explanations, not the project\'s measured performance. See docs/SOURCES.md for source scope.')
    if comparison_only:
        source_start = text.index('## 9. Provenance and references')
        text = text[:comparison_end] + [
            '## 4. Comparison conclusion and current boundary',
            'HistGradientBoosting has the highest validation macro-trip Top-1 proxy (0.8101), narrowly ahead of Random Forest (0.8093). Random Forest has the highest CV binary candidate Macro F1 (0.8688 versus 0.8680). The Top-1 difference is only 0.0008, while fold variability is about 0.026. This run does not establish a statistically reliable superiority of either model. HGB is the provisional model under the predeclared Top-1 criterion, not an established final winner.',
            'The current user-requested checkpoint ends at the five-model comparison. Bayesian optimisation, HMM/decoder evaluation, Folium prediction-map verification, API inference smoke tests, physical two-laptop demonstration and reserved evaluation are deferred. Later-stage source files drafted earlier are retained for continuation and are not claimed as verified deliverables. No results from those phases are reported.',
            'All five model fits and three-fold trip-aware validations completed, followed by evaluation on all ten student-test trips. The first four models completed before an environment-related interruption of HGB; the final model was resumed against the same manifest, seed, stride and folds. One-worker execution avoids the Windows multiprocessing restriction. Logs preserve the interruption and successful continuation.',
            'Before proceeding, review the weak-label alignment and candidate coverage, especially train trip 00000054. Confirm the official scoring convention. Repeat fitting with stride 1 and consider geographic holdout, more folds/seeds, probability calibration and radius/cap sensitivity on development data only. These follow-ups have not run.',
            'The full method and reproducible commands are in README.md. Sources, input hashes, fold membership, model configurations and per-trip diagnostics make these measured results independently inspectable. Team members should reproduce and explain them before submission.'
        ] + text[source_start:]
        text[0] = '# PathSnap: dataset and five-model comparison'
        text[text.index('## 9. Provenance and references')] = '## 5. Provenance and references'
    markdown = '\n\n'.join(text) + '\n'
    (docs / 'REPORT.md').write_text(markdown, encoding='utf-8')
    return markdown


def pdf_report(markdown, root, comparison_only=False):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.utils import ImageReader
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='ReportBody', fontName='Helvetica', fontSize=9.2, leading=13, spaceAfter=7, allowWidows=0, allowOrphans=0))
    styles.add(ParagraphStyle(name='TableCell', fontName='Helvetica', fontSize=7, leading=9, wordWrap='CJK'))
    styles['Heading1'].fontSize = 16; styles['Heading1'].textColor = colors.HexColor('#123f37')
    styles['Heading2'].fontSize = 12; styles['Heading2'].textColor = colors.HexColor('#196b53')
    styles['Heading1'].keepWithNext = True
    styles['Heading2'].keepWithNext = True
    width = 487

    def inline(s):
        s = html.escape(s)
        s = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', r'<link href="\2" color="#196b53">\1</link>', s)
        s = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', s)
        return s

    flow = []
    for block in markdown.split('\n\n'):
        if block.startswith('# '):
            flow.append(Paragraph(inline(block[2:]), styles['Title']))
        elif block.startswith('## '):
            flow.append(Paragraph(inline(block[3:]), styles['Heading1']))
        elif block.startswith('### '):
            flow.append(Paragraph(inline(block[4:]), styles['Heading2']))
        elif block.startswith('!['):
            match = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', block)
            if match:
                path = (root / 'docs' / match[2]).resolve()
                iw, ih = ImageReader(str(path)).getSize()
                scale = min(width / iw, (330 if path.name == 'frontend.jpg' else 290 if path.name == 'api-response.jpg' else 370) / ih)
                flow.append(KeepTogether([Image(str(path), width=iw * scale, height=ih * scale), Paragraph(inline(match[1]), styles['Caption'] if 'Caption' in styles else styles['ReportBody']), Spacer(1, 9)]))
        elif block.startswith('| '):
            lines = block.splitlines()
            rows = [[Paragraph(inline(cell.strip()), styles['TableCell']) for cell in line.strip('|').split('|')]
                    for i, line in enumerate(lines) if i != 1]
            n = len(rows[0]); colwidths = [width / n] * n
            if n == 3 and 'SHA-256' in lines[0]:
                colwidths = [145, 300, 42]
            t = Table(rows, colWidths=colwidths, repeatRows=1, hAlign='LEFT')
            t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dfede5')),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0, 0), (-1, -1), 5),
                ('RIGHTPADDING', (0, 0), (-1, -1), 5), ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6), ('LINEBELOW', (0, 0), (-1, 0), .5, colors.HexColor('#aac7b7')),
                ('LINEBELOW', (0, 1), (-1, -1), .25, colors.HexColor('#dce5df'))]))
            flow += [KeepTogether([t, Spacer(1, 10)]) if len(rows) <= 8 else t]
        else:
            flow.append(Paragraph(inline(block), styles['ReportBody']))
    target = root / 'output/pdf' / ('PathSnap_technical_progress_report.pdf' if comparison_only else 'PathSnap_technical_report.pdf')
    target.parent.mkdir(parents=True, exist_ok=True)

    def page(canvas, document):
        canvas.setFont('Helvetica', 8); canvas.setFillColor(colors.HexColor('#657970'))
        canvas.drawString(54, 28, 'PathSnap | AI-assisted working report | physical demo pending')
        canvas.drawRightString(541, 28, str(document.page))

    document = SimpleDocTemplate(str(target), pagesize=(595, 842), leftMargin=54, rightMargin=54,
                                 topMargin=42, bottomMargin=46, title='PathSnap technical report')
    document.build(flow, onFirstPage=page, onLaterPages=page)
    print(target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--pdf', action='store_true')
    parser.add_argument('--comparison-only', action='store_true')
    args = parser.parse_args()
    markdown = build(args.root.resolve(), args.comparison_only)
    if args.pdf:
        pdf_report(markdown, args.root.resolve(), args.comparison_only)
