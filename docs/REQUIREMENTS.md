# Requirements verified against the original PDFs

The two ZIPs were retrieved through the referenced chat's attachment paths, safely extracted locally, and hashed. This document uses the originals, not the earlier assistant's summary. PDF text extractions are retained in `docs/source_text/`; original PDFs remain in `data/raw/PathSnap_student_materials/`.

| Requirement | Primary source and page | Implementation/evidence |
|---|---|---|
| Two laptops: model/API on 1, UI on 2 | Main Guidelines p.2 | `src/pathsnap/api.py`, `frontend/`, demo runbook; physical demo pending |
| Inspect actual file schemas, counts, intervals, connectivity | Dataset Exploration pp.1-2, 4 | `eda.py`, `artifacts/eda/` |
| Metric distances; verify lon/lat order | Dataset Exploration pp.2-3 | UTM projection in `geo.py` |
| Treat supplied routes as imperfect references | Dataset Exploration p.3 | Explicit weak alignment labels, coverage and residuals |
| 80 train / 10 student development / 10 reserved evaluator trips | Dataset Exploration p.3 | All 80/10/10 observed; evaluator imported only after frozen model release |
| Keep whole trips together; learned preprocessing fitted on training only | Dataset Exploration p.3 | GroupKFold trip groups; fold-local scaler |
| Train and compare >=5 different models | Models and Training Guidelines pp.1-3 | Five actual classifier implementations and run records |
| Retain seeds, configurations, timing and comparison results | Models and Training Guidelines pp.2-4 | `experiment_run.json`, comparison CSV and logs |
| Candidate and route metrics, Top-1 and candidate misses | Models and Training Guidelines p.3 | Binary candidate metrics, aligned-point proxies, coverage, strict adjacency, arc overlap/edit diagnostics |
| Bayesian search after initial model comparison, group-aware CV | Models and Training Guidelines p.4 | Optuna TPE; folds on training trips; portable CSV + SQLite |
| Compare untuned/tuned on the same development partitions | Models and Training Guidelines p.4; Deliverables p.5 | Report generated from baseline and tuning artifacts |
| HMM states/emissions/transitions, gaps, missing candidates, Viterbi; compare to classifier | Models and Training Guidelines pp.4-5 | Decoder and reconstruction implementation; student-route comparison |
| Return a segment for every relevant GPS observation | Models and Training Guidelines p.5 | One output per input row; `-1` explicitly represents an unmatchable row |
| UI dynamically handles all ten reserved choices through API | Models and Training Guidelines pp.5-6 | Directory-derived list; fresh inference; all ten evaluator HTTP requests verified |
| Ground truth stays server-side; display receiving API logs | Models and Training Guidelines p.6 | Prediction before reference loading; sanitized map/response; server logs |
| Public complete GitHub repository + ONE PDF | Deliverables pp.1-2 | Complete source/models/results and consolidated technical PDF prepared for the user-selected pragyas0207/Pathsnap repository; team submission review remains |
| Report: per-model suitability, hyperparameters, objective, student results, fit analysis | Deliverables pp.3-4 | Generated technical report plus viva guide; team must review interpretations |
| Real CV/F1/Top-1/log-loss plots; no invented training curves | Deliverables p.4 | Artifact-driven plots; no synthetic histories |
| Folium GPS/network/reference/prediction layers, controls; screenshot in PDF | Deliverables p.6 | Offline development map; screenshot capture included when available |
| Working UI and receiving-laptop API screenshots | Deliverables pp.6-7 | Actual UI, Folium and browser API screenshots captured; physical Laptop 2 and receiving-terminal screenshots still required |
| Independent work; no inter-team sharing of code/configs/weights/results | Assessment and Plagiarism Policy p.2 | Team review, reproduction and authorship statement required |
| No invented results/citations; team must defend work; disclose assistance as required | Assessment and Plagiarism Policy pp.2-3; Deliverables p.7 | Measured artifacts only; assistance disclosed in project documentation |

The PDFs do **not** prescribe five particular algorithms, a candidate radius, a point-to-route alignment algorithm, exact metric aggregation, decoder hyperparameters, or an exact clock deadline. These are documented engineering choices. Deliverables p.1 states a first-phase evaluation before lunch, without a dated clock time. Confirm the official final scoring convention, especially point labels, with the organisers.
