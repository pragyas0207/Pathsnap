# Two-laptop demonstration

Use [BACKEND_HANDOFF.md](BACKEND_HANDOFF.md) for the current setup. This laptop is the model/API server and already contains the private final evaluator inputs.

The running backend port is **8766**. Laptop 2 should open **http://10.1.24.198:8766/ui/** on the shared Wi-Fi, or extract `output/PathSnap_frontend_laptop2.zip` and run its static server. Check the current Wi-Fi IP with `ipconfig` if the network changes.

Keep `START_BACKEND.cmd` open on Laptop 1 when restarting the server. `VIEW_BACKEND_LOGS.cmd` shows the receiving request log. The UI lists all ten evaluator IDs and computes each requested route dynamically. Reference routes remain server-side; only predicted routes and aggregate proxy metrics are returned.

All ten requests passed on the backend laptop, including same-host LAN health access. Physical Laptop 2 reachability, firewall/client-isolation conditions and actual receiving-terminal/UI screenshots must still be confirmed by the team. No final model selection/tuning may use these evaluator outcomes.
