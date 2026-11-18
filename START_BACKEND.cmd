@echo off
cd /d "%~dp0"
set "PYTHONPATH=%CD%\src"
set "PATHSNAP_DATA_DIR=%CD%\data\evaluation_server"
set "PATHSNAP_PARTITION=evaluator"
set "PATHSNAP_MODEL=%CD%\artifacts\final\model.joblib"
set "PATHSNAP_REQUIRE_FROZEN=1"
set "PATHSNAP_CORS_ORIGINS=http://localhost:5500,http://127.0.0.1:5500"
set "PATHSNAP_BACKEND_LOG=%CD%\artifacts\backend.log"
set "MPLCONFIGDIR=%CD%\.cache\matplotlib"
echo PathSnap BACKEND LAPTOP - keep this terminal open.
echo Local UI: http://127.0.0.1:8766/ui/
echo Laptop 2: open http://THIS-LAPTOP-WIFI-IP:8766/ui/
echo Use ipconfig to find the Wi-Fi IPv4 address. Both laptops must share a network.
".venv\Scripts\python.exe" tools/serve_backend.py --host 0.0.0.0 --port 8766
pause
