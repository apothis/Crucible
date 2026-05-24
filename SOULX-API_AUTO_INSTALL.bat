@echo off
setlocal enabledelayedexpansion

rem -------------------------------------------------------------
rem  Crucible - SoulX-Singer API installer (Windows + NVIDIA GPU)
rem  Clones SoulX-Singer, makes a venv, installs deps + models,
rem  drops in soulx_server.py (a /health + /synthesize API the Mac
rem  drives for the AI Vocal Builder), and creates run_soulx_api.bat.
rem
rem  KEEP soulx_server.py IN THE SAME FOLDER AS THIS .BAT.
rem  Run run_soulx_api.bat afterwards (serves on port 5060).
rem  Then set "soulx_host":"<this-PC-ip>:5060" in app_config.json.
rem -------------------------------------------------------------

set "PORT=5060"
set "HERE=%~dp0"
set "REPO=https://github.com/Soul-AILab/SoulX-Singer.git"

if not exist "%HERE%soulx_server.py" (
    echo ERROR: soulx_server.py must be in the same folder as this installer.
    pause
    exit /b 1
)

where git >nul 2>nul
if errorlevel 1 (
    echo ERROR: git was not found on PATH. Install Git for Windows first.
    pause
    exit /b 1
)
where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: python was not found on PATH. Need Python 3.10 or newer.
    pause
    exit /b 1
)

:ASK_DIR
echo(
echo Where should SoulX-Singer be installed? A folder will be created.
echo   e.g.  E:\AI\MusicGen\SoulX-Singer
set /p "DEST=Path: "
if "%DEST%"=="" goto ASK_DIR

if exist "%DEST%\cli\inference.py" goto HAVE_REPO
echo Cloning SoulX-Singer into "%DEST%" ...
git clone "%REPO%" "%DEST%"
if errorlevel 1 (
    echo ERROR: git clone failed.
    pause
    exit /b 1
)
:HAVE_REPO
cd /d "%DEST%"

echo(
echo -------- Creating venv + installing dependencies [large download] --------
if not exist venv\Scripts\python.exe python -m venv venv
if not exist venv\Scripts\python.exe (
    echo ERROR: venv creation failed.
    pause
    exit /b 1
)
set "PY=%DEST%\venv\Scripts\python.exe"
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: requirements install failed. See messages above.
    pause
    exit /b 1
)
echo(
echo -------- Installing CUDA build of torch [requirements ships CPU torch] --------
"%PY%" -m pip install --force-reinstall torch==2.2.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
"%PY%" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available after install'; print('torch', torch.__version__, 'CUDA OK', torch.cuda.get_device_name(0))"
if errorlevel 1 (
    echo ERROR: CUDA torch not working. Check your NVIDIA driver / CUDA support.
    pause
    exit /b 1
)
"%PY%" -m pip install fastapi "uvicorn[standard]" python-multipart huggingface_hub

echo(
echo -------- Fetching nltk data for g2p_en [English phonemes] --------
"%PY%" -c "import nltk; [nltk.download(p) for p in ['averaged_perceptron_tagger','averaged_perceptron_tagger_eng','cmudict']]"

echo(
echo -------- Downloading SoulX-Singer model weights --------
"%PY%" -c "from huggingface_hub import snapshot_download; snapshot_download('Soul-AILab/SoulX-Singer', local_dir='pretrained_models/SoulX-Singer')"
if errorlevel 1 (
    echo ERROR: model download failed.
    pause
    exit /b 1
)
echo Optional: preprocessing models [only needed for user reference clips]...
"%PY%" -c "from huggingface_hub import snapshot_download; snapshot_download('Soul-AILab/SoulX-Singer-Preprocess', local_dir='pretrained_models/SoulX-Singer-Preprocess')"

echo(
echo -------- Installing the API server --------
copy /Y "%HERE%soulx_server.py" "%DEST%\soulx_server.py" >nul

> run_soulx_api.bat echo @echo off
>> run_soulx_api.bat echo setlocal
>> run_soulx_api.bat echo cd /d "%DEST%"
>> run_soulx_api.bat echo set "PYTHONPATH=%DEST%;%%PYTHONPATH%%"
>> run_soulx_api.bat echo set "MG_SOULX_PORT=%PORT%"
>> run_soulx_api.bat echo "%PY%" soulx_server.py
>> run_soulx_api.bat echo pause

echo(
echo ============================================================
echo  Done. Start the server with:   run_soulx_api.bat
echo  It listens on port %PORT%.
echo  On the Mac, set in app_config.json:
echo      "soulx_host": "THIS_PC_IP:%PORT%"
echo  Find THIS_PC_IP with the ipconfig command. Then the Vocal
echo  Builder's SoulX engine will show as ready.
echo(
echo  VRAM: the model loads on demand and unloads after each synth
echo  so it shares the 3090 with ComfyUI/RVC safely. For faster
echo  repeat builds, set MG_SOULX_KEEP_RESIDENT=1 before launching.
echo ============================================================
pause
