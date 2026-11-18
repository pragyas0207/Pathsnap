$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:MPLCONFIGDIR = Join-Path $projectRoot '.cache/matplotlib'
$env:JUPYTER_CONFIG_DIR = Join-Path $projectRoot '.cache/jupyter/config'
$env:JUPYTER_DATA_DIR = Join-Path $projectRoot '.cache/jupyter/data'
$env:JUPYTER_RUNTIME_DIR = Join-Path $projectRoot '.cache/jupyter/runtime'
$env:IPYTHONDIR = Join-Path $projectRoot '.cache/jupyter/ipython'
$notebookPath = 'notebooks/PathSnap_outputs.ipynb'
if (Test-Path -LiteralPath 'notebooks/PathSnap_final_demo.ipynb') { $notebookPath = 'notebooks/PathSnap_final_demo.ipynb' }
& (Join-Path $projectRoot '.venv/Scripts/python.exe') -m notebook $notebookPath --ip=127.0.0.1
