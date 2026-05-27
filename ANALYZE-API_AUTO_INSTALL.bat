@echo off
setlocal enabledelayedexpansion

rem =============================================================================
rem  Crucible - Reference ANALYSIS API installer (Windows GPU box)   RESEARCH §17 P2
rem
rem  Stands up a box-side service that the Mac calls to analyse a reference track:
rem    * allin1 (All-In-One Music Structure Analyzer) -> BPM, beats, downbeats and
rem      FUNCTIONAL section labels (intro/verse/chorus/bridge/outro)
rem    * CLAP zero-shot -> genre/mood/instrument style tags
rem    * librosa -> key/scale
rem  Runs on the 3090; the Mac maps the result into the Song Builder.
rem
rem  *** FULLY SELF-CONTAINED ***  Everything lives under the install folder you pick:
rem  the venv, the pip cache, the HuggingFace cache, the torch-hub cache, and allin1's
rem  demix/spectrogram byproducts. NOTHING is written to %USERPROFILE%, %LOCALAPPDATA%,
rem  or anywhere else on the box. Delete the folder = fully uninstalled.
rem
rem  KEEP analyze_server.py IN THE SAME FOLDER AS THIS .BAT.
rem  Best on Python 3.10 (madmom + NATTEN wheels are most reliable there).
rem =============================================================================

set "PORT=5075"
set "HERE=%~dp0"
rem allin1 is pinned to NATTEN 0.14.6, whose prebuilt wheels are torch 2.0.0 / cu118 — so
rem this venv uses that exact stack (isolated; doesn't affect ComfyUI/ACE on the box).
set "CUDA=cu118"

if not exist "%HERE%analyze_server.py" (
    echo ERROR: analyze_server.py must be in the same folder as this installer.
    pause & exit /b 1
)

:ASK_DIR
echo(
echo Install folder for the analysis service (venv + ~3-4 GB of models live here),
echo e.g.  C:\AI\CrucibleAnalyze
set /p "DEST=Path: "
if "%DEST%"=="" goto ASK_DIR
if not exist "%DEST%" mkdir "%DEST%"

rem ---- CONTAINMENT: pin every cache INSIDE the install dir (install-time) ----
set "DEST_CACHE=%DEST%\.cache"
set "PIP_CACHE_DIR=%DEST_CACHE%\pip"
set "HF_HOME=%DEST_CACHE%\huggingface"
set "HUGGINGFACE_HUB_CACHE=%DEST_CACHE%\huggingface\hub"
set "TORCH_HOME=%DEST_CACHE%\torch"
set "XDG_CACHE_HOME=%DEST_CACHE%"
set "TRANSFORMERS_CACHE=%DEST_CACHE%\huggingface"
for %%D in ("%DEST_CACHE%" "%PIP_CACHE_DIR%" "%HF_HOME%" "%TORCH_HOME%") do if not exist "%%~D" mkdir "%%~D"

rem ---- find a Python (prefer 3.10 for madmom/NATTEN) ----
set "PY="
for %%P in ("py -3.10" "py -3.11" "py -3.12" "python") do (
    %%~P --version >nul 2>&1 && ( set "PY=%%~P" & goto GOTPY )
)
:GOTPY
if "%PY%"=="" ( echo ERROR: need Python 3.10+ on PATH. & pause & exit /b 1 )
echo Using Python: %PY%   (3.10 recommended for madmom + NATTEN)

echo(
echo -------- Creating venv (in the install dir) --------
%PY% -m venv "%DEST%\venv"
set "VPY=%DEST%\venv\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip wheel setuptools

rem =============================================================================
rem  CORE (required) — CLAP tags + key. This alone makes a working analyze service.
rem  torch 2.0.0/cu118 so the (optional) NATTEN 0.14.6 wheel can match; torchvision is
rem  required by laion-clap (timm_model -> torchvision.ops).
rem =============================================================================
echo(
echo -------- [core] torch 2.0.0 (%CUDA%) + CLAP + server deps --------
"%VPY%" -m pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/%CUDA%
"%VPY%" -m pip install "numpy<2" laion-clap librosa soundfile fastapi "uvicorn[standard]" python-multipart
if errorlevel 1 ( echo   [!!] CORE install failed - the service needs these. Fix the error above and re-run. & pause )

echo(
echo -------- [core] Pre-fetching the CLAP checkpoint into the in-dir cache --------
rem ~2 GB now so the first analysis isn't slow; lands under %HF_HOME% (in-dir).
"%VPY%" -c "import laion_clap; m=laion_clap.CLAP_Module(enable_fusion=False); m.load_ckpt(); print('CLAP ckpt ready')" || echo   [!] CLAP prefetch skipped (will download on first use).

rem =============================================================================
rem  OPTIONAL (advanced) — allin1 for BETTER labelled structure (intro/verse/chorus) +
rem  beats. Best-effort: the service works WITHOUT this (it falls back to the Mac's
rem  librosa structure). Needs: git, the MS C++ Build Tools (madmom compiles), and the
rem  prebuilt NATTEN 0.14.6 wheel (often unavailable now — shi-labs cert/host issues; the
rem  PyPI sdist would need the CUDA Toolkit to compile). If any step fails, just skip it.
rem =============================================================================
echo(
echo -------- [optional] allin1 structure (cython - madmom - natten - allin1) --------
"%VPY%" -m pip install "cython>=0.29"
"%VPY%" -m pip install --no-build-isolation git+https://github.com/CPJKU/madmom
if errorlevel 1 echo   [optional] madmom skipped (needs git + MS C++ Build Tools). Structure stays on the Mac.
"%VPY%" -m pip install "natten==0.14.6+torch200cu118" -f https://shi-labs.com/natten/wheels --trusted-host shi-labs.com
if errorlevel 1 echo   [optional] NATTEN 0.14.6 wheel unavailable (shi-labs gone / no prebuilt). allin1 will be skipped.
"%VPY%" -m pip install allin1
if errorlevel 1 echo   [optional] allin1 not installed - the service runs in tags-only mode (Mac handles structure).

echo(
echo -------- Installing the API server --------
copy /Y "%HERE%analyze_server.py" "%DEST%\analyze_server.py" >nul

set "LAUNCH=%DEST%\run_analyze_api.bat"
> "%LAUNCH%" echo @echo off
>> "%LAUNCH%" echo cd /d "%%~dp0"
>> "%LAUNCH%" echo rem --- CONTAINMENT: keep all caches/byproducts inside this folder at runtime ---
>> "%LAUNCH%" echo set "PIP_CACHE_DIR=%%~dp0.cache\pip"
>> "%LAUNCH%" echo set "HF_HOME=%%~dp0.cache\huggingface"
>> "%LAUNCH%" echo set "HUGGINGFACE_HUB_CACHE=%%~dp0.cache\huggingface\hub"
>> "%LAUNCH%" echo set "TRANSFORMERS_CACHE=%%~dp0.cache\huggingface"
>> "%LAUNCH%" echo set "TORCH_HOME=%%~dp0.cache\torch"
>> "%LAUNCH%" echo set "XDG_CACHE_HOME=%%~dp0.cache"
>> "%LAUNCH%" echo set "MG_ANALYZE_WORK=%%~dp0.work"
>> "%LAUNCH%" echo set "MG_ANALYZE_DEVICE=cuda"
>> "%LAUNCH%" echo set "MG_ANALYZE_PORT=%PORT%"
>> "%LAUNCH%" echo venv\Scripts\python.exe analyze_server.py
>> "%LAUNCH%" echo pause

echo(
echo -------------------------------------------------------------
echo   Installed (self-contained in %DEST%).
echo   START THE SERVICE:  %DEST%\run_analyze_api.bat
echo   Reachable:          http://^<this-PC-IP^>:%PORT%/health
echo   On the Mac, set     "analyze_host": "^<this-PC-IP^>:%PORT%"  in app_config.json
echo   Uninstall = delete the folder (no traces left elsewhere).
echo -------------------------------------------------------------
echo Launching the API now (first run may download the allin1 model into the in-dir cache)...
pushd "%DEST%"
call run_analyze_api.bat
popd
pause
exit /b
