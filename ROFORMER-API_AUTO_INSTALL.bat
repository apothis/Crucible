@echo off
setlocal enabledelayedexpansion

rem -------------------------------------------------------------
rem  Crucible — BS-Roformer separation API installer (Windows GPU box)
rem  Creates a dedicated venv, installs CUDA torch + bs-roformer-infer,
rem  downloads the BS-Roformer SW 6-stem model (vocals/bass/drums/
rem  guitar/piano/other), and runs a small REST API on port 5070.
rem
rem  The Mac calls it; the heavy model runs here on the 3090 (running it
rem  on the Mac's MPS hard-crashes the Mac). Model loads per request and
rem  the process frees VRAM after each separation.
rem
rem  KEEP roformer_server.py IN THE SAME FOLDER AS THIS .BAT.
rem  Requires Python 3.10+ on PATH (the `py` launcher or `python`).
rem -------------------------------------------------------------

set "PORT=5070"
set "HERE=%~dp0"
set "CUDA=cu124"

if not exist "%HERE%roformer_server.py" (
    echo ERROR: roformer_server.py must be in the same folder as this installer.
    pause & exit /b 1
)

:ASK_DIR
echo(
echo Install folder for the separation service (a venv + ~3 GB model will live here),
echo e.g.  C:\AI\RoformerSep
set /p "DEST=Path: "
if "%DEST%"=="" goto ASK_DIR
if not exist "%DEST%" mkdir "%DEST%"

rem ---- find a Python 3.10+ ----
set "PY="
for %%P in ("py -3.12" "py -3.11" "py -3.10" "python") do (
    %%~P --version >nul 2>&1 && ( set "PY=%%~P" & goto GOTPY )
)
:GOTPY
if "%PY%"=="" ( echo ERROR: need Python 3.10+ on PATH. & pause & exit /b 1 )
echo Using Python: %PY%

echo(
echo -------- Creating venv --------
%PY% -m venv "%DEST%\venv"
set "VPY=%DEST%\venv\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip wheel

echo(
echo -------- Installing CUDA torch (%CUDA%) --------
"%VPY%" -m pip install torch --index-url https://download.pytorch.org/whl/%CUDA%

echo(
echo -------- Installing bs-roformer-infer + server deps --------
"%VPY%" -m pip install bs-roformer-infer packaging huggingface_hub fastapi "uvicorn[standard]" python-multipart soundfile

echo(
echo -------- Downloading BS-Roformer SW model (6-stem) --------
"%VPY%" -m bs_roformer.download --model roformer-model-bs-roformer-sw-by-jarredou --output-dir "%DEST%\models"

set "MDIR=%DEST%\models\roformer-model-bs-roformer-sw-by-jarredou"
set "CFG=%MDIR%\BS-Rofo-SW-Fixed.yaml"
set "CKPT=%MDIR%\BS-Rofo-SW-Fixed.ckpt"
if not exist "%CKPT%" (
    echo   [!] model checkpoint not found at "%CKPT%"
    echo       check the download step above.
)

echo(
echo -------- Installing the API server --------
copy /Y "%HERE%roformer_server.py" "%DEST%\roformer_server.py" >nul

set "LAUNCH=%DEST%\run_roformer_api.bat"
> "%LAUNCH%" echo @echo off
>> "%LAUNCH%" echo cd /d "%%~dp0"
>> "%LAUNCH%" echo set "MG_ROFORMER_CONFIG=%CFG%"
>> "%LAUNCH%" echo set "MG_ROFORMER_CKPT=%CKPT%"
>> "%LAUNCH%" echo set "MG_ROFORMER_DEVICE=cuda"
>> "%LAUNCH%" echo set "MG_ROFORMER_PORT=%PORT%"
>> "%LAUNCH%" echo venv\Scripts\python.exe roformer_server.py
>> "%LAUNCH%" echo pause

echo(
echo -------------------------------------------------------------
echo   Installed. START THE SERVICE:  %DEST%\run_roformer_api.bat
echo   Reachable: http://^<this-PC-IP^>:%PORT%
echo   On the Mac, set  "roformer_host": "^<this-PC-IP^>:%PORT%"  in app_config.json
echo -------------------------------------------------------------
echo Launching the API now...
pushd "%DEST%"
call run_roformer_api.bat
popd
pause
exit /b
