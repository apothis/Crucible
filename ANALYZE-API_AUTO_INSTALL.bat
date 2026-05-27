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
rem  Python 3.10 / 3.11 / 3.12 all work (prebuilt NATTEN 0.17.5 wheels exist for each).
rem =============================================================================

set "PORT=5075"
set "HERE=%~dp0"
rem torch 2.6.0 / cu126: runs CLAP AND matches the prebuilt NATTEN 0.17.5 Windows wheel
rem that the all-in-one-fix fork needs — so we get structure too, with no CUDA build.
set "CUDA=cu126"

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

rem ---- CONTAINMENT: pin every cache AND temp INSIDE the install dir (install-time) ----
rem NOTE: PIP_CACHE_DIR only controls the download cache. pip's git-clone + wheel BUILD
rem scratch uses %TMP%/%TEMP% (defaults to C:\Users\<you>\AppData\Local\Temp), so we
rem redirect those too — otherwise building e.g. madmom from git writes outside this folder.
set "DEST_CACHE=%DEST%\.cache"
set "PIP_CACHE_DIR=%DEST_CACHE%\pip"
set "HF_HOME=%DEST_CACHE%\huggingface"
set "HUGGINGFACE_HUB_CACHE=%DEST_CACHE%\huggingface\hub"
set "TORCH_HOME=%DEST_CACHE%\torch"
set "XDG_CACHE_HOME=%DEST_CACHE%"
set "TRANSFORMERS_CACHE=%DEST_CACHE%\huggingface"
set "TMP=%DEST_CACHE%\tmp"
set "TEMP=%DEST_CACHE%\tmp"
for %%D in ("%DEST_CACHE%" "%PIP_CACHE_DIR%" "%HF_HOME%" "%TORCH_HOME%" "%TMP%") do if not exist "%%~D" mkdir "%%~D"

rem ---- find a Python 3.10-3.12 (NATTEN 0.17.5 wheels exist for cp310/cp311/cp312) ----
set "PY="
for %%P in ("py -3.10" "py -3.11" "py -3.12" "python") do (
    %%~P --version >nul 2>&1 && ( set "PY=%%~P" & goto GOTPY )
)
:GOTPY
if "%PY%"=="" ( echo ERROR: need Python 3.10-3.12 on PATH. & pause & exit /b 1 )
echo Using Python: %PY%

echo(
echo -------- Creating venv (in the install dir) --------
%PY% -m venv "%DEST%\venv"
set "VPY=%DEST%\venv\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip wheel setuptools

rem =============================================================================
rem  CLAP tags + key — this is the whole service. Modern CUDA torch; torchvision is
rem  required by laion-clap (timm_model -> torchvision.ops). The Mac handles structure.
rem =============================================================================
echo(
echo -------- torch 2.6.0 (%CUDA%) + CLAP + server deps --------
"%VPY%" -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/%CUDA%
"%VPY%" -m pip install "numpy<2" laion-clap librosa soundfile fastapi "uvicorn[standard]" python-multipart
if errorlevel 1 ( echo   [!!] install failed - the service needs these. Fix the error above and re-run. & pause )

echo(
echo -------- Pre-fetching the CLAP checkpoint into the in-dir cache --------
rem ~2 GB now so the first analysis isn't slow; lands under %HF_HOME% (in-dir).
"%VPY%" -c "import laion_clap; m=laion_clap.CLAP_Module(enable_fusion=False); m.load_ckpt(); print('CLAP ckpt ready')" || echo   [!] CLAP prefetch skipped (will download on first use).

rem =============================================================================
rem  OPTIONAL — better labelled structure via the all-in-one-fix fork. Needs THREE things:
rem    1) a PREBUILT Windows NATTEN 0.17.5 wheel (lldacing) — no CUDA build,
rem    2) madmom (compiles via the MS C++ Build Tools + git — all-in-one-fix imports it
rem       despite not declaring it as a dependency),
rem    3) all-in-one-fix itself.
rem  Best-effort: if any piece is missing the service runs tags-only and the Mac handles
rem  structure. Requires Python 3.10-3.12 (natten wheels) + git + the C++ Build Tools.
rem =============================================================================
echo(
echo -------- [optional] all-in-one-fix structure (prebuilt NATTEN + madmom build) --------
rem NOTE: detect with NO quotes in the Python code — batch for/f ('...') is single-quote
rem delimited, so a python f-string ('cp{..}') would clash and garble PYTAG.
for /f %%T in ('"%VPY%" -c "import sys;print(sys.version_info[1])"') do set "PYMINOR=%%T"
set "PYTAG=cp3%PYMINOR%"
echo   python tag: %PYTAG%  (prebuilt NATTEN 0.17.5 wheels exist ONLY for cp310/cp311/cp312)
if not "%PYTAG%"=="cp310" if not "%PYTAG%"=="cp311" if not "%PYTAG%"=="cp312" (
    echo   [optional] No prebuilt NATTEN wheel for %PYTAG% - SKIPPING allin1 ^(tags-only^).
    echo              For structure, recreate this venv with Python 3.10-3.12 and re-run.
    goto SKIP_ALLIN1
)
"%VPY%" -m pip install "https://huggingface.co/lldacing/NATTEN-windows/resolve/main/natten-0.17.5+torch260cu126-%PYTAG%-%PYTAG%-win_amd64.whl"
echo   building madmom (git + MS C++ Build Tools; all-in-one-fix needs it)...
"%VPY%" -m pip install "cython>=0.29"
"%VPY%" -m pip install --no-build-isolation "git+https://github.com/CPJKU/madmom"
"%VPY%" -m pip install all-in-one-fix
"%VPY%" -m pip install "numpy<2"
"%VPY%" -c "import allin1fix" 2>nul
if errorlevel 1 (
    echo   [optional] allin1 unavailable ^(NATTEN wheel or madmom build failed^) - tags-only mode; Mac handles structure.
) else (
    echo   [optional] allin1fix ready - full labelled structure enabled.
)
:SKIP_ALLIN1

echo(
echo -------- Enforcing numpy<2 (laion-clap requires it; torch/all-in-one-fix can bump it) --------
rem MUST be the last dependency step: pip resolves each install independently, so torch and
rem all-in-one-fix may have pulled numpy 2.x back in, which breaks laion-clap at runtime.
"%VPY%" -m pip install "numpy<2"

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
>> "%LAUNCH%" echo set "TMP=%%~dp0.cache\tmp"
>> "%LAUNCH%" echo set "TEMP=%%~dp0.cache\tmp"
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
