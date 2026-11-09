@echo off
setlocal
cd /d "%~dp0"
rem Builds a private Python 3.12 venv in .\.venv (SAM 3 needs >=3.12). Uses uv to fetch Python if needed.
where python >nul 2>nul || (echo Install any Python 3 first so uv can be bootstrapped. & exit /b 1)
python -m pip install --quiet uv || exit /b 1
python -m uv venv .venv --python 3.12 || exit /b 1
set UV_LINK_MODE=copy
python -m uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128 || exit /b 1
python -m uv pip install --python .venv\Scripts\python.exe -r requirements.txt || exit /b 1
echo Setup complete. Run run.bat
