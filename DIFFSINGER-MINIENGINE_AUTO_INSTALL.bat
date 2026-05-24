@echo off
setlocal enabledelayedexpansion

rem -------------------------------------------------------------
rem  Crucible - DiffSinger (MiniEngine) installer (Windows + GPU)
rem  Installs openvpi/DiffSingerMiniEngine: a small ONNX inference
rem  server [rhythm + acoustic + NSF-HiFiGAN vocoder] the Mac drives
rem  for the AI Vocal Builder. Lighter than SoulX but needs an
rem  ENGLISH voicebank [no zero-shot cloning].
rem
rem  Serves on port 9266. Then set
rem  "diffsinger_host":"<this-PC-ip>:9266" in app_config.json.
rem -------------------------------------------------------------

set "HERE=%~dp0"
set "REPO=https://github.com/openvpi/DiffSingerMiniEngine.git"

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
echo Where should DiffSingerMiniEngine be installed?
echo   e.g.  E:\AI\MusicGen\DiffSingerMiniEngine
set /p "DEST=Path: "
if "%DEST%"=="" goto ASK_DIR

if exist "%DEST%\server.py" goto HAVE_REPO
git clone "%REPO%" "%DEST%"
if errorlevel 1 (
    echo ERROR: git clone failed.
    pause
    exit /b 1
)
:HAVE_REPO
cd /d "%DEST%"

echo(
echo -------- venv + dependencies --------
if not exist venv\Scripts\python.exe python -m venv venv
if not exist venv\Scripts\python.exe (
    echo ERROR: venv creation failed.
    pause
    exit /b 1
)
set "PY=%DEST%\venv\Scripts\python.exe"
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install onnxruntime-gpu PyYAML soundfile

echo(
echo -------- Model assets you must place manually --------
echo   1. NSF-HiFiGAN ONNX vocoder into  assets\vocoder\
echo        https://github.com/openvpi/vocoders/releases/tag/nsf-hifigan-v1
echo   2. Rhythm predictor ONNX into     assets\rhythmizer\
echo        https://github.com/openvpi/DiffSinger/releases/tag/v1.4.1
echo   3. An ENGLISH acoustic .onnx into assets\acoustic\
echo      plus its English phoneme dictionary into assets\dictionaries\
echo      from community DiffSinger English voicebanks / dsdict-en
echo   Then edit configs\default.yaml: set the dictionary filename,
echo   the acoustic model name, and enable CUDAExecutionProvider.
echo(
echo Opening the asset folder + config for you...
if not exist assets\acoustic   mkdir assets\acoustic
if not exist assets\vocoder     mkdir assets\vocoder
if not exist assets\rhythmizer  mkdir assets\rhythmizer
start "" "%DEST%\assets"
start "" notepad "%DEST%\configs\default.yaml"

> run_diffsinger_api.bat echo @echo off
>> run_diffsinger_api.bat echo cd /d "%DEST%"
>> run_diffsinger_api.bat echo "%PY%" server.py
>> run_diffsinger_api.bat echo pause

echo(
echo ============================================================
echo  After placing the vocoder/rhythmizer/acoustic models and
echo  editing configs\default.yaml, start with run_diffsinger_api.bat
echo  It listens on port 9266. On the Mac set in app_config.json:
echo      "diffsinger_host": "THIS_PC_IP:9266"
echo ============================================================
pause
