# PathSnap - finished backend implementation

This is the backend laptop for GPS map matching. The five-model comparison, 15-trial Bayesian optimisation, student-test HMM/Viterbi evaluation, frozen model release, Folium maps, FastAPI and frontend are implemented and executed. All ten supplied evaluator trips passed real HTTP inference using the frozen model. Pointwise scores are reference-alignment proxies, not independently measured point labels.

Source repository: [pragyas0207/Pathsnap](https://github.com/pragyas0207/Pathsnap). The root [PathSnap_outputs.ipynb](PathSnap_outputs.ipynb) is a copy of the final executed demonstration, preserving the originally shared notebook URL. The original five-model comparison remains under `notebooks/PathSnap_outputs.ipynb`. See [publication notes](docs/GITHUB_PUBLICATION.md) for what is included.

## Use the backend

The running backend is at **http://127.0.0.1:8766/ui/**. Laptop 2 on the same Wi-Fi can open **http://10.1.24.198:8766/ui/**. The Wi-Fi IP can change; confirm it with `ipconfig` if necessary.

- `START_BACKEND.cmd`: restart the frozen evaluator backend after stopping it. Keep its terminal open. A second server on the same port is explicitly rejected.
- `VIEW_BACKEND_LOGS.cmd`: watch real request IDs, timings and classifier/HMM proxy scores.
- `OPEN_NOTEBOOK.cmd`: open the actual executed `notebooks/PathSnap_final_demo.ipynb` in Jupyter.
- `SHOW_OUTPUTS.cmd`: the earlier five-baseline-model terminal demonstration.

See [the backend handoff](docs/BACKEND_HANDOFF.md) for Laptop 2, CORS, firewall/network troubleshooting and evidence capture. Health is `/health`; choices are `/trips`; inference is `POST /match` with `{"trip_id":"00000017"}`; API documentation is `/docs`.

The evaluator inputs are private under `data/evaluation_server/`. They are not part of the frontend package or source release. The interface gets only predicted routes and aggregate scores. References are read for scoring only after the prediction pipeline finishes.

## Actual outputs

- `notebooks/PathSnap_final_demo.ipynb`: editable code, baseline data/features, five-model tables, actual TPE history, HMM comparison, final evaluator results and a fresh backend request/map.
- `artifacts/model_comparison.csv`: original five-model comparison.
- `artifacts/tuning_summary.json`, `optimization_trials.csv`, `optimization.sqlite3`: real 15-trial search with three random startup and twelve adaptive TPE trials.
- `artifacts/route_evaluation.csv`: actual classifier/HMM development comparisons.
- `artifacts/final_summary.json`, `evaluator_results.csv`, `evaluator_api_smoke.json`: actual final-only evaluator results.
- `artifacts/final/model.joblib`, `model_manifest.json`: released model/configuration and checksums, frozen before evaluator import.
- `artifacts/maps/00000000.html`: development Folium map with a reference layer.
- `output/pdf/PathSnap_technical_report.pdf`: consolidated technical report generated from executed artifacts.
- `output/PathSnap_frontend_laptop2.zip`: frontend only for a separately served Laptop 2 UI.
- `output/PathSnap_source_release.zip`: source, trusted model artifacts, configurations and portable results; no raw/evaluator inputs or virtual environment.

The selected model is tuned HistGradientBoosting. Student-test mean-trip Top-1 proxy is 83.06% classifier-only and 87.19% after HMM; strict adjacency is 89.95% and 96.08%. Final evaluator mean-trip HMM Top-1 proxy is 86.89%; pooled observation Top-1 proxy is 75.96%. Trip 00000015 is weaker (48.03%); it is retained in the results and no tuning was performed on final evaluator outcomes.

## Reproduce the development experiment

Python 3.11.9 and locked package versions were used. Run from the project root:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-lock.txt
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$env:MPLCONFIGDIR = (Join-Path (Get-Location) '.cache/matplotlib')
.venv\Scripts\python tools/inspect_sources.py 'C:\path\Student folder.zip' 'C:\path\PathSnap_student_materials.zip'
.venv\Scripts\python -m pathsnap.cli eda
.venv\Scripts\python -m pathsnap.cli prepare
```

To repeat experiments without overwriting the released evidence, use a fresh output directory:

```powershell
.venv\Scripts\python -m pathsnap.cli --artifacts artifacts/reproduction train --folds 3 --fit-stride 5
.venv\Scripts\python -m pathsnap.cli --artifacts artifacts/reproduction compare
.venv\Scripts\python -m pathsnap.cli --artifacts artifacts/reproduction tune --trials 15
.venv\Scripts\python -m pathsnap.cli --artifacts artifacts/reproduction evaluate --all-models
.venv\Scripts\python -m pathsnap.cli --artifacts artifacts/reproduction map --trip 00000000
.venv\Scripts\python -m pytest -q
```

For Jupyter on a new machine, also install `requirements-notebook.txt`. The local notebook launcher selects the final notebook. The notebook loads completed experiments and performs fresh inference; Run All does not retrain the five baselines. Training/tuning can take several minutes. SQLite studies deliberately do not silently resume under a changed search definition; use a new artifact directory.

The actual release used every fifth eligible training observation for fitting; validation and student testing score all observations. All 80 training trips participate in three GroupKFold splits. Do not describe this as fitting every training GPS point. `--fit-stride 1` is an unrun follow-up experiment, not a reported result.

## Method and assumptions

1. Strictly parse headerless `.track`=[longitude,latitude,seconds], `.nodes`=[longitude,latitude], `.arcs`=[from,to] and `.route`=[chronological arc IDs]. IDs are zero-based row indices. Prediction input loading never opens `.route`.
2. Project coordinates into the UTM zone selected from the trip. Build straight directed road segments and a sparse graph. Guard zero-length geometry numerically.
3. Query exact point-to-segment distances with STRtree. Keep the nearest eight within 60 m. Empty sets stay unmatched (`-1`); no distant answer is fabricated.
4. Extract 12 inference-available features: distance, heading cosine/known flag, log segment length, projected fraction, capped speed/interval/step, rank, candidate count, and endpoint outdegrees. IDs and reference-derived attributes are not features.
5. Derive weak point labels offline with a monotone route-occurrence distance DP. Distances are capped at 1000 m in its cost; residuals over 100 m become unknown. Route endpoints are not forced. Coverage and candidate misses are reported. This alignment is not an organiser-provided scorer and can be wrong around loops/stops/reference defects.
6. Train Logistic Regression (fold-local scaling), Gaussian Naive Bayes, Decision Tree, Random Forest and HistGradientBoosting. HGB random-row internal early stopping is disabled. Seed=42. Initial settings and all fold results are recorded.
7. Select/tune by mean validation macro-trip Top-1 proxy on training-trip folds. Candidate binary Macro F1 and log loss are separate diagnostics; binary thresholds use 0.5, road selection uses argmax. Candidate misses count as Top-1 errors on aligned observations. Unknown labels are excluded with coverage shown.
8. Use normalized classifier scores as HMM-style emission potentials. Directed-road transitions compare road distance with GPS displacement, with beta=30 m, speed limit=60 m/s, slack=30 m and gap threshold=30 s. These decoder constants were fixed development choices, not tuned against evaluator results. Viterbi operates in log space and resets at gaps/empty sets/impossible transitions. Shortest paths fill intermediate arcs and preserve loops and route breaks.
9. Freeze the model and inference-source checksums, then import evaluator files with `tools/provision_evaluator.py`. Development commands accept only train/student-test partitions. The evaluator backend refuses a missing/mismatched frozen manifest.

Strict adjacency tests same/direct-next arcs and can penalize valid multi-arc travel between samples. Arc-set F1 ignores order; sampled edit similarity penalizes missing intermediate segments. Neither is physical route truth. Trip-aware validation does not guarantee geographic separation. Probability calibration, feature ablations, multiple seeds and a nearest-road baseline were not measured.

## Repository map

| Folder | Contents |
|---|---|
| `src/pathsnap/` | Parsers, geometry, features, labels, models, experiments, metrics, decoder, reconstruction, maps, API, CLI |
| `tools/` | Archive audit, source freeze, final-only import/HTTP verification, notebook/report builders, package release |
| `frontend/` | Browser UI and pinned self-hosted Leaflet with license/provenance |
| `tests/` | Geometry, data integrity, reference separation, CORS, sequence optimality and loop reconstruction checks |
| `artifacts/` | Measured configurations, folds, trials, results, maps and trusted models |
| `docs/` | Requirement/source audit, handoff, viva guidance and report source |
| `notebooks/` | Original comparison and final executed Jupyter notebooks |

Only load joblib artifacts you generated and trust. The complete raw archive inventory and processed manifest are preserved in portable artifact copies. Frozen source/model hashes are checked at startup; an algorithm change needs a new development release rather than silently modifying the evaluator model.

## Submission boundary

This implementation and same-host HTTP tests are complete. A real Laptop 2 must still connect on the shared network, and the team must capture the actual receiving-terminal/UI screenshots and review the report, team details and assistance disclosure. No remote firewall settings were changed, and same-host LAN reachability is not proof of second-laptop reachability. The existing team repository above is the publication destination. The technical PDF records the earlier pre-publication checkpoint; its GitHub-publication task is superseded by this repository update. This is AI-assisted work with actually executed experiments; team members remain responsible for understanding and defending it.

See [SOURCES.md](docs/SOURCES.md), [REQUIREMENTS.md](docs/REQUIREMENTS.md) and [VIVA_GUIDE.md](docs/VIVA_GUIDE.md).
