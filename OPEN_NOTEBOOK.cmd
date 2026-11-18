@echo off
cd /d "%~dp0"
set "JUPYTER_CONFIG_DIR=%CD%\.cache\jupyter\config"
set "JUPYTER_DATA_DIR=%CD%\.cache\jupyter\data"
set "JUPYTER_RUNTIME_DIR=%CD%\.cache\jupyter\runtime"
set "IPYTHONDIR=%CD%\.cache\jupyter\ipython"
set "MPLCONFIGDIR=%CD%\.cache\matplotlib"
if exist "notebooks\PathSnap_final_demo.ipynb" (
  ".venv\Scripts\python.exe" -m notebook "notebooks\PathSnap_final_demo.ipynb" --ip=127.0.0.1
) else (
  ".venv\Scripts\python.exe" -m notebook "notebooks\PathSnap_outputs.ipynb" --ip=127.0.0.1
)
pause
