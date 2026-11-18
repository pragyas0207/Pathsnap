# Backend laptop handoff

This laptop hosts the frozen model, HMM/Viterbi, road inputs, private evaluator references and FastAPI. All ten evaluator trips were tested through the API with HTTP 200 responses. The model was frozen before evaluator files were imported. No evaluator score was used for tuning or model selection.

## Use it now

- Backend UI on this laptop: [http://127.0.0.1:8766/ui/](http://127.0.0.1:8766/ui/)
- Laptop 2, on the same network: [http://10.1.24.198:8766/ui/](http://10.1.24.198:8766/ui/)
- Health: `/health`; trip list: `/trips`; Swagger: `/docs`.
- The actual Jupyter file is `notebooks/PathSnap_final_demo.ipynb`; `OPEN_NOTEBOOK.cmd` opens it.
- `START_BACKEND.cmd` restarts the server after it is stopped. Do not start a second copy while the existing server is running. Exclusive port binding makes a duplicate fail explicitly.
- `VIEW_BACKEND_LOGS.cmd` displays request IDs, timings and proxy scores as requests arrive.

The observed Wi-Fi IPv4 is **10.1.24.198**, not one of the VMware/VirtualBox adapter addresses. If Wi-Fi changes, run `ipconfig` and use the current Wi-Fi IPv4. Port **8766** is intentional: another service was responding on port 8000. The backend binds all IPv4 interfaces but uses an exclusive socket on Windows to prevent ambiguous port sharing.

## Laptop 2

The simplest option is to open the LAN UI URL above in its browser. The interface JavaScript executes on Laptop 2, while all inference/scoring remains on Laptop 1. It automatically connects to the same backend origin.

If the assessor requires the static frontend server itself on Laptop 2, copy `output/PathSnap_frontend_laptop2.zip`, extract it, and run `START_UI.cmd` there (requires Python). Open `http://localhost:5500` and verify the backend field is `http://10.1.24.198:8766`. This ZIP contains frontend assets only: no training data, evaluation references or model artifacts.

Default CORS permits `http://localhost:5500` and `http://127.0.0.1:5500`. If the frontend is served from another origin, pass its exact URL as `-CorsOrigins` to `scripts/run_backend.ps1`. Do not open the frontend as a `file://` page.

## Demonstrate

1. Keep the backend logs visible on Laptop 1.
2. On Laptop 2, connect and show all ten IDs: 00000015, 00000017, 00000023, 00000030, 00000049, 00000059, 00000067, 00000073, 00000087, 00000094.
3. Select any trip and click Match route. Each click recomputes candidates, features, classifier scores and Viterbi; no prerecorded prediction is returned.
4. Show the map, classifier/HMM proxy scores, label coverage and request ID. Match that ID to the receiving-laptop log.
5. Pointwise scores are derived-reference alignment proxies. Trip 00000015 has a weaker HMM proxy (48.03%); the all-trip results retain it. Do not present only the stronger trips as the overall result.
6. Capture the actual receiving-terminal and Laptop 2 UI screenshots for the team submission. A local same-host LAN request is not proof of a physical second-laptop test.

Leaflet JS/CSS/images are self-hosted and work without CDN access. The optional OpenStreetMap basemap tiles need internet. If tiles cannot load, GPS, road-network and prediction geometry still render. To omit remote tiles entirely, set `PATHSNAP_OFFLINE_MAPS=1` before restarting the backend.

If Laptop 2 cannot reach the health URL, confirm both machines use the same network and the current Wi-Fi IP. Check private-network firewall access for Python/TCP 8766 and whether the Wi-Fi isolates devices. No firewall settings were changed by this task; remote reachability must be confirmed on Laptop 2.

## Trusted files and evidence

- Frozen artifact: `artifacts/final/model.joblib`.
- Freeze manifest: `artifacts/final/model_manifest.json`; SHA-256 begins `5cb6500bf65c7edb`.
- Training-only TPE record: `artifacts/optimization_trials.csv` and `optimization.sqlite3`.
- Student decoder results: `artifacts/route_evaluation.csv`.
- Final evaluator summary/table: `artifacts/final_summary.json`, `evaluator_results.csv`.
- Full ten-trip HTTP receipt: `artifacts/evaluator_api_smoke.json`.
- Request logs: `artifacts/backend.log` and `artifacts/server_runs/evaluator_requests.jsonl`.
- Private evaluator inputs and import receipt: `data/evaluation_server/` (excluded from repository/ frontend package).

The runtime verifies the frozen model and inference-source checksums. Changing those files requires an explicitly versioned development release and fresh development validation; do not tune using the final evaluator outcomes.
