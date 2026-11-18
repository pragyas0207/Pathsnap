# Sources and provenance

The assignment requirements come from the original supplied PDFs, extracted locally from `PathSnap_student_materials.zip`:

- **Main Guidelines**, 2 pages: challenge and two-laptop architecture.
- **Dataset Exploration**, 4 pages: exact raw formats, zero-based identifiers, data partitions, limitations and EDA expectations.
- **Models and Training Guidelines**, 6 pages: five-model comparison, Bayesian search, metrics, HMM and final demo contract.
- **Deliverables**, 7 pages: public repository, one consolidated PDF, required plots, maps, screenshots and integrity requirements.
- **Assessment and Plagiarism Policy**, 3 pages: independent work, experiment authenticity and viva responsibilities.

Detailed source-to-requirement locations are in `REQUIREMENTS.md`. Text extractions preserve page boundaries under `docs/source_text/`. `data/raw/inventory.json` records the source archive and member checksums. The original dataset is the `Student folder.zip` attachment; every trip's file hash and schema/quality checks are recorded in `artifacts/eda/trip_inventory.csv`.

External primary sources checked on 18 November 2026:

1. Newson, P. and Krumm, J. (2009), [Hidden Markov Map Matching Through Noise and Sparseness](https://www.microsoft.com/en-us/research/publication/hidden-markov-map-matching-noise-sparseness/). Motivation for sequence inference using the road network. Our classifier-based emissions and reset policy are project choices, not a claim of an exact reproduction of that paper. This paper's Seattle dataset is **not** asserted to be the supplied dataset.
2. scikit-learn, [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html) and [grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html). Non-overlapping trip groups protect against same-trip row leakage; they do not guarantee geographic separation.
3. Shapely, [STRtree](https://shapely.readthedocs.io/en/stable/strtree.html). Query results preserve indices into the geometry array, used here as arc IDs; the distance predicate operates in projected coordinate units.
4. Optuna, [TPESampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html). TPE uses completed-trial outcomes to propose configurations after the random startup budget. Our installed version and startup setting are recorded in run artifacts; live documentation may describe a later release.
5. FastAPI, [CORS tutorial](https://fastapi.tiangolo.com/tutorial/cors/). The frontend's scheme/host/port defines its origin; backend and frontend ports on different laptops require explicit permitted origins.
6. Folium, [official user guide](https://python-visualization.github.io/folium/latest/user_guide.html). Map layers and controls for actual GPS/network/prediction geometry.

These references support design explanations. Performance claims derive solely from this repository's executed scripts and artifacts. External papers' accuracy claims are not substituted for project measurements.

The later `evaluator_test.zip` archive was imported for final inference only after the frozen manifest timestamp. Its SHA-256 is `a425b35e6e6f8d3a919690c5df3a9a7df2f7c2de7343a01c50ccd6ecb45178a7`. Private member hashes and the import timestamp remain in `data/evaluation_server/import_receipt.json`; they were not used by the training/tuning workflow.

Leaflet 1.9.3 assets are self-hosted from the pinned [Leaflet distribution](https://unpkg.com/leaflet@1.9.3/dist/leaflet.js). Its BSD-2-Clause license and per-file source URLs/hashes are included under `frontend/vendor/leaflet/`. Optional basemap tiles are credited to OpenStreetMap in the actual maps.
