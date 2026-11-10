@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
rem Manual cleaning UI: IOPaint's web editor (brush a mask, click to remove) running LaMa locally.
rem Usage: double-click, or drag a folder of images onto this file to browse/edit that folder.
rem        Each cleaned image is saved over the original file in that same folder.
if not exist ".venv\Scripts\iopaint.exe" (
    echo Environment not found - running setup first...
    call setup.bat || exit /b 1
)
set "TORCH_HOME=%~dp0models\torch"
set "HF_HOME=%~dp0models\huggingface"

set "SRC=%~1"
if "!SRC!"=="" goto :start
if "!SRC:~-1!"=="\" set "SRC=!SRC:~0,-1!"
echo Editing folder: !SRC!
echo Cleaned images are saved over the originals in that folder.
".venv\Scripts\iopaint.exe" start --model lama --device cuda --port 8080 --model-dir "%~dp0models\torch" --input "!SRC!" --output-dir "!SRC!"
goto :end

:start
echo Starting IOPaint (LaMa) - your browser will open at http://localhost:8080
echo Close this window to stop it.
".venv\Scripts\iopaint.exe" start --model lama --device cuda --port 8080 --model-dir "%~dp0models\torch"

:end
pause
