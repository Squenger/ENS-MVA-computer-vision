# Point the notebook to local project folders (does not modify main_notebook.ipynb).
# Run this in PowerShell, then start Cursor/Jupyter from the same session,
# or restart the notebook kernel after setting these in your environment.

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:RECVIS_INPUT_DIR  = Join-Path $ProjectRoot 'data'
$env:RECVIS_OUTPUT_DIR = Join-Path $ProjectRoot 'outputs'

Write-Host "RECVIS_INPUT_DIR  = $env:RECVIS_INPUT_DIR"
Write-Host "RECVIS_OUTPUT_DIR = $env:RECVIS_OUTPUT_DIR"
Write-Host "Restart the notebook kernel, then re-run the imports cell."
