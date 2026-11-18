# Final backend status

Completed: original archive/PDF inspection; 80/10 development audit; candidates/features/weak labels; five classifiers with three trip-aware folds; 15 TPE trials; tuned refit; all-ten-student classifier/HMM comparison; directed reconstruction checks; frozen model/source release before evaluator import; all-ten-evaluator HTTP inference; local and same-host LAN health checks; self-hosted Leaflet; executed final notebook.

The evaluator input ZIP contains ten reserved trips. It was structurally inspected by names/sizes before release and imported only after the model freeze. It did not enter fitting, hyperparameter selection or decoder choices. The frozen checksum is recorded in all final receipts.

Fourteen automated tests pass. All ten evaluator requests returned HTTP 200 and one predicted ID per GPS row. Final metrics are in artifacts/final_summary.json and evaluator_results.csv. Scores are alignment proxies; the weak result on trip 00000015 remains included.

Publication destination: https://github.com/pragyas0207/Pathsnap . The repository update includes source, frozen models, final notebook outputs and the PDF; raw/evaluator inputs are excluded. The PDF was generated before this publication update.

Remaining team actions: physical Laptop 2 rehearsal on the shared network; actual receiving-terminal and second-laptop screenshots; member details and independent report/assistance review. These actions were not represented as complete.
