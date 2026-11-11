@echo off
REM Point notebook paths to this project (no PowerShell execution policy needed).
set "RECVIS_INPUT_DIR=%~dp0data"
set "RECVIS_OUTPUT_DIR=%~dp0outputs"
echo RECVIS_INPUT_DIR  = %RECVIS_INPUT_DIR%
echo RECVIS_OUTPUT_DIR = %RECVIS_OUTPUT_DIR%
echo Restart the notebook kernel, then re-run the imports cell.
