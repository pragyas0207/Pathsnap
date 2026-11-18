@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Project Python environment is missing. See README.md for setup.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" "tools\show_outputs.py" --plots
pause
