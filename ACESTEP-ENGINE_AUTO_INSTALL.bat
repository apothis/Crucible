@echo off
setlocal enabledelayedexpansion

rem -------------------------------------------------------------
rem  Crucible — OFFICIAL ACE-Step 1.5 engine installer (Windows GPU box)
rem  Installs the real `acestep` inference engine (the one ACE Studio runs)
rem  and its REST API server on port 8001, bound to the LAN so the Mac can
rem  drive it. Full engine: text2music / cover / repaint / lego / extract /
rem  complete, with the proper params (audio_cover_strength etc).
rem
rem  SELF-CONTAINED: uv, the Python it pulls, all package caches, the HF model
rem  cache and the model weights ALL live inside your chosen install folder.
rem  Nothing is written to %USERPROFILE% caches. (Git is the only possible
rem  system-wide piece, and only if you don't already have it.)
rem
rem  THIS TAKES A WHILE: the XL set is ~25-30 GB of weights + a CUDA torch
rem  build. Leave it running; downloads resume if interrupted.
rem  Needs: an NVIDIA GPU w/ CUDA 12.8 drivers (the 3090). Git (auto if missing).
rem -------------------------------------------------------------

set "REPO=https://github.com/ace-step/ACE-Step-1.5.git"
set "PORT=8001"

rem ---------- MODEL SET CHOICE ----------
:CHOOSE_MODEL
echo(
echo Which ACE-Step 1.5 model set?
echo 1) Turbo only  - core repo (~10 GB), fast previews, smallest download   QUICK START
echo 2) XL full     - core + XL base + XL SFT (~30 GB), best quality          RECOMMENDED
set "ACE_SET="
set /p "MODEL_CHOICE=Enter 1 or 2: "
if "%MODEL_CHOICE%"=="1" set "ACE_SET=TURBO"
if "%MODEL_CHOICE%"=="2" set "ACE_SET=XL"
if not defined ACE_SET ( echo Invalid choice. & timeout /t 2 >nul & goto CHOOSE_MODEL )

rem ---------- LM CHOICE ----------
:CHOOSE_LM
echo(
echo Which LM (writes/structures captions, bpm, key)? The core repo already
echo includes the 1.7B LM. The 4B is smarter but ~8 GB more to download.
echo 1) Keep 1.7B (already in core)                                          RECOMMENDED
echo 2) Also fetch 4B (best, larger)
set "GET_LM4B="
set /p "LM_CHOICE=Enter 1 or 2: "
if "%LM_CHOICE%"=="1" set "GET_LM4B=0"
if "%LM_CHOICE%"=="2" set "GET_LM4B=1"
if not defined GET_LM4B ( echo Invalid choice. & timeout /t 2 >nul & goto CHOOSE_LM )

rem ---------- INSTALL FOLDER ----------
:ASK_DIR
echo(
echo Install folder (engine + models + ALL caches live here),
echo e.g.  C:\AI\ACEStep
set /p "DEST=Path: "
if "%DEST%"=="" goto ASK_DIR
if not exist "%DEST%" mkdir "%DEST%"

rem ---------- CONTAINMENT: every cache/runtime pinned inside DEST ----------
set "CKPT=%DEST%\checkpoints"
set "UV_CACHE_DIR=%DEST%\.cache\uv"
set "UV_PYTHON_INSTALL_DIR=%DEST%\.uvpython"
set "UV_INSTALL_DIR=%DEST%\.uvbin"
set "HF_HOME=%DEST%\.cache\huggingface"
set "HUGGINGFACE_HUB_CACHE=%DEST%\.cache\huggingface\hub"
set "PIP_CACHE_DIR=%DEST%\.cache\pip"
for %%D in ("%CKPT%" "%UV_CACHE_DIR%" "%UV_PYTHON_INSTALL_DIR%" "%UV_INSTALL_DIR%" "%HF_HOME%" "%PIP_CACHE_DIR%") do if not exist "%%~D" mkdir "%%~D"

echo(
echo -------- Checking prerequisites --------
call :ensure_git || exit /b 1
call :ensure_uv  || exit /b 1
echo Using uv: %UV%

echo(
echo -------- Getting the ACE-Step engine --------
pushd "%DEST%"
if exist "ACE-Step-1.5\.git" (
    echo   - already cloned - pulling latest
    pushd ACE-Step-1.5 & git pull & popd
) else (
    git clone "%REPO%"
    if errorlevel 1 ( echo Clone failed. & popd & pause & exit /b 1 )
)
pushd "ACE-Step-1.5"
set "ENGINE=%CD%"

echo(
echo -------- Installing the engine + CUDA dependencies (uv sync) --------
echo   (builds the Python env + CUDA torch INSIDE the install dir - several minutes)
"%UV%" sync
if errorlevel 1 ( echo uv sync failed. & popd & popd & pause & exit /b 1 )

echo(
echo -------- Downloading models into "%CKPT%" --------
echo   Selected: model set=[%ACE_SET%]  fetch 4B LM=[%GET_LM4B%]
echo   Each component is listed as it downloads; existing ones are verified/resumed.
echo(
rem Plain %%VAR%% (not !VAR!) so this does not depend on delayed expansion; keep these
rem single-line and parenthesis-free in the descriptions.
call :hfget "ACE-Step/Ace-Step1.5" "%CKPT%" "Core: VAE + text encoder + turbo DiT + 1.7B LM, ~10 GB"
if /i "%ACE_SET%"=="XL" call :hfget "ACE-Step/acestep-v15-xl-base" "%CKPT%\acestep-v15-xl-base" "XL base DiT - 4B, best quality, ~9 GB"
if /i "%ACE_SET%"=="XL" call :hfget "ACE-Step/acestep-v15-xl-sft"  "%CKPT%\acestep-v15-xl-sft"  "XL SFT DiT - 4B, fine-tuned, ~9 GB"
if "%GET_LM4B%"=="1" call :hfget "ACE-Step/acestep-5Hz-lm-4B" "%CKPT%\acestep-5Hz-lm-4B" "4B LM - smartest captioning, ~8 GB"

popd
popd

echo(
echo -------- Creating LAN launcher --------
set "LAUNCH=%DEST%\run_acestep_api.bat"
> "%LAUNCH%" echo @echo off
>> "%LAUNCH%" echo rem Launch ACE-Step API on the LAN. Open http://^<this-PC-IP^>:%PORT%/health from the Mac.
>> "%LAUNCH%" echo set "ROOT=%%~dp0"
>> "%LAUNCH%" echo set "UV_CACHE_DIR=%%ROOT%%.cache\uv"
>> "%LAUNCH%" echo set "UV_PYTHON_INSTALL_DIR=%%ROOT%%.uvpython"
>> "%LAUNCH%" echo set "HF_HOME=%%ROOT%%.cache\huggingface"
>> "%LAUNCH%" echo set "HUGGINGFACE_HUB_CACHE=%%ROOT%%.cache\huggingface\hub"
>> "%LAUNCH%" echo set "ACESTEP_CHECKPOINTS_DIR=%%ROOT%%checkpoints"
>> "%LAUNCH%" echo cd /d "%ENGINE%"
>> "%LAUNCH%" echo "%UV%" run acestep-api --host 0.0.0.0 --port %PORT%
>> "%LAUNCH%" echo pause

echo(
echo -------------------------------------------------------------
echo   Install complete^!  Everything is inside: %DEST%
echo   START THE ENGINE:  %DEST%\run_acestep_api.bat
echo   Reachable:         http://THIS-PC-IP:%PORT%   (check /health)
echo   On the Mac, set    "acestep_host": "THIS-PC-IP:%PORT%"  in app_config.json
echo   (replace THIS-PC-IP with the box's LAN IP - run ipconfig)
echo   First launch loads models into VRAM (the 3090 has plenty) - give it a minute.
echo -------------------------------------------------------------
echo Launching the API now...
pushd "%DEST%"
call run_acestep_api.bat
popd
pause
exit /b


:: ================= helper routines =================

:hfget
rem %1 = HF repo id, %2 = local dir, %3 = description
rem Always call huggingface-cli: it's idempotent (skips complete files, resumes
rem partial ones), so re-running is safe and we never wrongly skip a download.
echo(
echo   ^>^> %~3
if not exist "%~2" mkdir "%~2"
"%UV%" run --with "huggingface_hub[cli]" huggingface-cli download %~1 --local-dir "%~2"
if errorlevel 1 echo      [!] download failed for %~1 - re-run installer to resume, or verify the repo id.
goto :eof

:ensure_git
echo Checking for Git...
git --version >nul 2>&1
if not errorlevel 1 ( echo Git OK. & exit /b 0 )
echo Git not found. Installing with winget (system-wide)...
where winget >nul 2>&1 || ( echo winget unavailable - install Git from https://git-scm.com/download/win then re-run. & pause & exit /b 1 )
winget install -e --id Git.Git --accept-package-agreements --accept-source-agreements
git --version >nul 2>&1 || ( echo Git installed but not on PATH yet - close this window and re-run. & pause & exit /b 1 )
exit /b 0

:ensure_uv
rem Prefer an existing uv (caches are pinned via env vars regardless); else install
rem uv's standalone build INTO the install dir (no winget, fully contained).
set "UV="
uv --version >nul 2>&1 && set "UV=uv"
if defined UV ( echo uv found on PATH. & exit /b 0 )
if exist "%UV_INSTALL_DIR%\uv.exe" ( set "UV=%UV_INSTALL_DIR%\uv.exe" & echo uv found in install dir. & exit /b 0 )
echo Installing uv (standalone, into %UV_INSTALL_DIR%)...
powershell -ExecutionPolicy Bypass -NoProfile -Command "$env:UV_INSTALL_DIR='%UV_INSTALL_DIR%'; $env:UV_UNMANAGED_INSTALL='%UV_INSTALL_DIR%'; irm https://astral.sh/uv/install.ps1 | iex"
if exist "%UV_INSTALL_DIR%\uv.exe" ( set "UV=%UV_INSTALL_DIR%\uv.exe" & echo uv installed. & exit /b 0 )
echo ERROR: uv install failed. Install manually from https://docs.astral.sh/uv/ then re-run.
pause
exit /b 1
