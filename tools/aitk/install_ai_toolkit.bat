@echo off
REM Ostris AI Toolkit for YuE2 LoRA training (docs/YUE2_TRAINING_PLAN.md).
REM Sits in CrucibleLorA beside the box helper and installs to <MusicGen>\ai-toolkit, next to
REM AceStep / Comfyui / CrucibleLorA. Started remotely by the helper (POST /aitk/install, which
REM passes "nopause" and logs to CrucibleLorA\aitk_install.log); double-clicking it also works.
REM Pinned to commit dcebc74 (2026-10-16), fetched as a zip so no system git is needed.
REM The toolkit's own manager then builds everything INSIDE that folder: uv-managed Python 3.12,
REM a .venv with torch 2.13 cu130 for this GPU, portable git, Node.js and FFmpeg.
REM Nothing is installed system-wide (uv's and npm's download caches stay inside too) and the web UI
REM is NOT started: jobs run through the helper (POST /aitk/run).
REM uv is fetched here with curl so the toolkit's PowerShell fallback is never used.
REM Resumable: re-run this file if a download drops.

setlocal
set "PAUSE=pause"
if /i "%~1"=="nopause" set "PAUSE=rem"
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
    if exist "%AITK%\run.py" del "%ROOT%\ai-toolkit-%COMMIT%.zip"
)
if not exist "%AITK%\run.py" (
    echo ERROR: source not unpacked into %AITK%
    %PAUSE%
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
    %PAUSE%
    exit /b 1
)
set "PATH=%AITK%\.uv;%PATH%"
set "UV_PYTHON_INSTALL_DIR=%AITK%\.uv\python"
set "UV_CACHE_DIR=%AITK%\.uv\cache"
REM npm (the toolkit installs its web-UI packages even though we never start the UI) caches here, not on C:
set "npm_config_cache=%AITK%\.npm-cache"

echo.
echo === 3 of 4: Python 3.12 (uv-managed, inside ai-toolkit\.uv\python) ===
uv python install 3.12
set "PY="
for /f "delims=" %%P in ('uv python find 3.12') do set "PY=%%P"
if not defined PY (
    echo ERROR: could not find the uv-managed Python 3.12
    %PAUSE%
    exit /b 1
)

echo.
echo === 4 of 4: the toolkit's environment (torch for this GPU, requirements, git, Node, FFmpeg) ===
"%PY%" -m manager install
if errorlevel 1 (
    echo.
    echo Install failed - see the output above.
    %PAUSE%
    exit /b 1
)

echo.
echo Done. AI Toolkit is in %AITK% with its own .venv.
echo Nothing else to start: caption and training jobs run through the helper (POST /aitk/run).
%PAUSE%
