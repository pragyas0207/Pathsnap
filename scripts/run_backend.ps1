param(
    [ValidateSet('student_test','evaluator')][string]$Partition = 'evaluator',
    [string]$DataDir = '',
    [string]$CorsOrigins = 'http://localhost:5500,http://127.0.0.1:5500',
    [int]$Port = 8766
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:PYTHONPATH = Join-Path $projectRoot 'src'
$env:PATHSNAP_PARTITION = $Partition
$env:PATHSNAP_MODEL = Join-Path $projectRoot 'artifacts/final/model.joblib'
$env:PATHSNAP_REQUIRE_FROZEN = '1'
$env:PATHSNAP_BACKEND_LOG = Join-Path $projectRoot 'artifacts/backend.log'
$env:PATHSNAP_CORS_ORIGINS = $CorsOrigins
$env:MPLCONFIGDIR = Join-Path $projectRoot '.cache/matplotlib'
if ($DataDir) { $env:PATHSNAP_DATA_DIR = $DataDir }
elseif ($Partition -eq 'evaluator') { $env:PATHSNAP_DATA_DIR = Join-Path $projectRoot 'data/evaluation_server' }
else { $env:PATHSNAP_DATA_DIR = Join-Path $projectRoot 'data/raw/Student folder/Student folder' }
& (Join-Path $projectRoot '.venv/Scripts/python.exe') tools/serve_backend.py --host 0.0.0.0 --port $Port
