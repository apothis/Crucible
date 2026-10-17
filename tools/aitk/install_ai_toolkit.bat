@echo off
REM Ostris AI Toolkit for YuE2 LoRA training (docs/YUE2_TRAINING_PLAN.md).
REM Lives next to ComfyUI: E:\AI\MusicGen\Comfyui\ai-toolkit (inside the box file API's root).
REM Pinned to commit dcebc74 (2026-10-16), fetched as a zip so no system git is needed.
REM The toolkit's own manager then builds everything INSIDE that folder: uv-managed Python 3.12,
REM a .venv with torch 2.13 cu130 for this GPU, portable git, Node.js and FFmpeg.
REM Nothing is installed system-wide and the web UI is NOT started.
REM uv is fetched here with curl so the toolkit's PowerShell fallback is never used.
REM Resumable: re-run this file if a download drops.

setlocal
cd /d "%~dp0.."
set "ROOT=%CD%"
set "AITK=%ROOT%\ai-toolkit"
set COMMIT=dcebc7442df70e5239c68167919fd2909c615769

REM clear anything that could hijack the Python the manager runs on
set PYTHONPATH=
set PYTHONHOME=
set VIRTUAL_ENV=
set CONDA_PREFIX=

echo.
echo === 1 of 4: AI Toolkit source at %COMMIT% ===
if exist "%AITK%\run.py" (
    echo already present: %AITK%
) else (
    curl -L -C - -o "%ROOT%\ai-toolkit-%COMMIT%.zip" "https://github.com/ostris/ai-toolkit/archive/%COMMIT%.zip"
    tar -xf "%ROOT%\ai-toolkit-%COMMIT%.zip" -C "%ROOT%"
    ren "%ROOT%\ai-toolkit-%COMMIT%" ai-toolkit
)
if not exist "%AITK%\run.py" (
    echo ERROR: source not unpacked into %AITK%
    pause
    exit /b 1
)

echo.
echo === 2 of 4: uv (Python and package manager) into ai-toolkit\.uv ===
cd /d "%AITK%"
if not exist ".uv\uv.exe" (
    if not exist ".uv" mkdir ".uv"
    curl -L -C - -o ".uv\uv.zip" "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
    tar -xf ".uv\uv.zip" -C ".uv"
)
if not exist ".uv\uv.exe" (
    echo ERROR: uv.exe not found in %AITK%\.uv
    pause
    exit /b 1
)
set "PATH=%AITK%\.uv;%PATH%"
set "UV_PYTHON_INSTALL_DIR=%AITK%\.uv\python"

echo.
echo === 3 of 4: Python 3.12 (uv-managed, inside ai-toolkit\.uv\python) ===
uv python install 3.12
set "PY="
for /f "delims=" %%P in ('uv python find 3.12') do set "PY=%%P"
if not defined PY (
    echo ERROR: could not find the uv-managed Python 3.12
    pause
    exit /b 1
)

echo.
echo === 4 of 4: the toolkit's environment (torch for this GPU, requirements, git, Node, FFmpeg) ===
"%PY%" -m manager install
if errorlevel 1 (
    echo.
    echo Install failed - see the output above.
    pause
    exit /b 1
)

echo.
echo Done. AI Toolkit is in %AITK% with its own .venv.
echo Nothing else to start: training and captioning run from job files via run_aitk_job.bat.
pause
