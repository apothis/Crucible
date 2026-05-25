@echo off
setlocal enabledelayedexpansion

rem -------------------------------------------------------------
rem  Crucible — OFFICIAL ACE-Step 1.5 engine installer (Windows GPU box)
rem  Installs the real `acestep` inference engine (the one ACE Studio runs)
rem  and its REST API server on port 8001, bound to the LAN so the Mac can
rem  drive it. This is the full engine: text2music / cover / repaint / lego
rem  / extract / complete, with the proper params (audio_cover_strength etc).
rem
rem  Style/structure adapted from MUSICGEN-COMFYUI_AUTO_INSTALL.bat. Models are
rem  downloaded EXPLICITLY per-component (you'll see each one) via huggingface-cli.
rem
rem  THIS TAKES A WHILE: the XL set is ~25-30 GB of model weights + a CUDA
rem  torch install. Leave it running; downloads resume if interrupted.
rem
rem  Needs: Git + the `uv` package manager (the script installs uv via winget
rem  if missing) + an NVIDIA GPU with CUDA 12.8 drivers (the 3090 is ideal).
rem -------------------------------------------------------------

set "REPO=https://github.com/ace-step/ACE-Step-1.5.git"
set "PORT=8001"

rem ---------- MODEL SET CHOICE ----------
:CHOOSE_MODEL
echo(
echo Which ACE-Step 1.5 model set?
echo 1) Turbo only  - core repo (~10 GB), fast previews, smallest download   QUICK START
echo 2) XL full     - core + XL base + XL SFT (~30 GB), best quality          RECOMMENDED
set /p "MODEL_CHOICE=Enter 1 or 2: "
if "!MODEL_CHOICE!"=="1" (set "ACE_SET=TURBO") ^
else if "!MODEL_CHOICE!"=="2" (set "ACE_SET=XL") ^
else ( echo Invalid choice. & timeout /t 2 >nul & goto CHOOSE_MODEL )

rem ---------- LM CHOICE ----------
:CHOOSE_LM
echo(
echo Which LM (writes/structures captions, bpm, key)? The core repo already
echo includes the 1.7B LM. The 4B is smarter but ~8 GB more to download.
echo 1) Keep 1.7B (already in core)                                          RECOMMENDED
echo 2) Also fetch 4B (best, larger)
set /p "LM_CHOICE=Enter 1 or 2: "
if "!LM_CHOICE!"=="1" (set "GET_LM4B=0") ^
else if "!LM_CHOICE!"=="2" (set "GET_LM4B=1") ^
else ( echo Invalid choice. & timeout /t 2 >nul & goto CHOOSE_LM )

rem ---------- INSTALL FOLDER ----------
:ASK_DIR
echo(
echo Install folder (the engine + a checkpoints folder of models live here),
echo e.g.  C:\AI\ACEStep
set /p "DEST=Path: "
if "%DEST%"=="" goto ASK_DIR
if not exist "%DEST%" mkdir "%DEST%"
set "CKPT=%DEST%\checkpoints"
if not exist "%CKPT%" mkdir "%CKPT%"

echo(
echo -------- Checking prerequisites --------
call :ensure_git || exit /b 1
call :ensure_uv  || exit /b 1

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
echo   (this builds the Python env + CUDA torch - can take several minutes)
uv sync
if errorlevel 1 ( echo uv sync failed. & popd & popd & pause & exit /b 1 )

echo(
echo -------- Downloading models into "%CKPT%" --------
echo   Each component is listed as it downloads; existing ones are skipped.
echo   (progress bars are from huggingface-cli)
echo(

rem core repo (vae + Qwen3-Embedding-0.6B text encoder + turbo DiT + 1.7B LM)
call :hfget "ACE-Step/Ace-Step1.5" "%CKPT%" "Core: VAE + text encoder + turbo DiT + 1.7B LM (~10 GB)"

if "!ACE_SET!"=="XL" (
    call :hfget "ACE-Step/acestep-v15-xl-base" "%CKPT%\acestep-v15-xl-base" "XL base DiT - 4B, best quality (~9 GB)"
    call :hfget "ACE-Step/acestep-v15-xl-sft"  "%CKPT%\acestep-v15-xl-sft"  "XL SFT DiT - 4B, fine-tuned (~9 GB)"
)

if "!GET_LM4B!"=="1" (
    call :hfget "ACE-Step/acestep-5Hz-lm-4B" "%CKPT%\acestep-5Hz-lm-4B" "4B LM - smartest captioning (~8 GB)"
)

popd
popd

echo(
echo -------- Creating LAN launcher --------
set "LAUNCH=%DEST%\run_acestep_api.bat"
> "%LAUNCH%" echo @echo off
>> "%LAUNCH%" echo rem Launch the ACE-Step API server bound to the LAN so the Mac can reach it.
>> "%LAUNCH%" echo rem Open http://^<this-PC-IP^>:%PORT%/health from the Mac to verify (run ipconfig for the IP).
>> "%LAUNCH%" echo cd /d "%ENGINE%"
>> "%LAUNCH%" echo set "ACESTEP_CHECKPOINTS_DIR=%CKPT%"
>> "%LAUNCH%" echo uv run acestep-api --server-name 0.0.0.0 --port %PORT%
>> "%LAUNCH%" echo pause

echo(
echo -------------------------------------------------------------
echo   Install complete^!
echo   START THE ENGINE:  %DEST%\run_acestep_api.bat
echo   Reachable:         http://^<this-PC-IP^>:%PORT%   (check /health)
echo   On the Mac, set    "acestep_host": "^<this-PC-IP^>:%PORT%"  in app_config.json
echo(
echo   First launch loads models into VRAM (the 3090 has plenty) - give it a minute.
echo -------------------------------------------------------------
echo Launching the API now (first run may finish any remaining model setup)...
pushd "%DEST%"
call run_acestep_api.bat
popd
pause
exit /b


:: ================= helper routines =================

:hfget
rem %1 = HF repo id, %2 = local dir, %3 = description
echo(
echo   ^>^> %~3
if exist "%~2\*" (
    echo      already present in "%~2" - skipping
    goto :eof
)
if not exist "%~2" mkdir "%~2"
uv run --with "huggingface_hub[cli]" huggingface-cli download %~1 --local-dir "%~2"
if errorlevel 1 (
    echo      [!] download failed for %~1
    echo          re-run this installer to resume, or check the repo id.
)
goto :eof

:ensure_git
echo Checking for Git...
git --version >nul 2>&1
if not errorlevel 1 ( echo Git OK. & exit /b 0 )
echo Git not found. Installing with winget...
where winget >nul 2>&1 || ( echo winget unavailable - install Git from https://git-scm.com/download/win & pause & exit /b 1 )
winget install -e --id Git.Git --accept-package-agreements --accept-source-agreements
git --version >nul 2>&1 || ( echo Git installed but not on PATH yet - close this window and re-run. & pause & exit /b 1 )
exit /b 0

:ensure_uv
echo Checking for uv...
uv --version >nul 2>&1
if not errorlevel 1 ( echo uv OK. & exit /b 0 )
echo uv not found. Installing with winget...
where winget >nul 2>&1 || ( echo winget unavailable - install uv from https://docs.astral.sh/uv/ & pause & exit /b 1 )
winget install -e --id astral-sh.uv --accept-package-agreements --accept-source-agreements
uv --version >nul 2>&1 || ( echo uv installed but not on PATH yet - close this window and re-run. & pause & exit /b 1 )
exit /b 0
