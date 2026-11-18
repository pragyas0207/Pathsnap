@echo off
cd /d "%~dp0"
set "PATHSNAP_LOG=%CD%\artifacts\backend.log"
if not exist "%PATHSNAP_LOG%" (
  echo Start the backend first.
  pause
  exit /b 1
)
powershell -NoProfile -Command "Get-Content -LiteralPath $env:PATHSNAP_LOG -Tail 25 -Wait"
