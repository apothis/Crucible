@echo off
setlocal enabledelayedexpansion

rem =============================================================================
rem  Crucible - LoRA dataset UPLOAD helper installer (Windows GPU box)  §METAL_LORA_PLAN
rem
rem  Tiny box-side bridge so Crucible (on the Mac) can push a prepared LoRA training
rem  dataset (audio + {name}.lyrics.txt + {name}.json) onto the box, where the ACE-Step
rem  engine's /v1/dataset/* + /v1/training/* endpoints read it. NO GPU, NO models -
rem  just FastAPI. Datasets are written under <install>\lora_data\<dataset>\{data,
rem  tensors,adapter}; that path is handed back to the Mac to drive scan/preprocess/train.
rem
rem  *** SELF-CONTAINED *** venv + pip cache live under the install folder. Delete = gone.
rem  KEEP lora_upload_server.py IN THE SAME FOLDER AS THIS .BAT.
rem  Any Python 3.9+ works (pure-Python deps).
rem =============================================================================

set "PORT=5080"
set "HERE=%~dp0"

if not exist "%HERE%lora_upload_server.py" (
    echo ERROR: lora_upload_server.py must be in the same folder as this installer.
    pause & exit /b 1
)

:ASK_DIR
echo(
echo Install folder for the LoRA upload helper (small; datasets you upload also live here),
echo e.g.  C:\AI\CrucibleLoRA
set /p "DEST=Path: "
if "%DEST%"=="" goto ASK_DIR
if not exist "%DEST%" mkdir "%DEST%"

rem ---- containment: pin the pip cache + temp inside the install dir ----
set "DEST_CACHE=%DEST%\.cache"
set "PIP_CACHE_DIR=%DEST_CACHE%\pip"
set "TMP=%DEST_CACHE%\tmp"
set "TEMP=%DEST_CACHE%\tmp"
for %%D in ("%DEST_CACHE%" "%PIP_CACHE_DIR%" "%TMP%") do if not exist "%%~D" mkdir "%%~D"

rem ---- find a Python 3.9+ ----
set "PY="
for %%P in ("py -3.12" "py -3.11" "py -3.10" "py -3.9" "python") do (
    %%~P --version >nul 2>&1 && ( set "PY=%%~P" & goto GOTPY )
)
echo ERROR: no Python found on PATH (need 3.9+).
pause & exit /b 1
:GOTPY
echo Using Python: %PY%

rem ---- venv + deps ----
%PY% -m venv "%DEST%\venv" || ( echo venv create failed & pause & exit /b 1 )
set "VPY=%DEST%\venv\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip
"%VPY%" -m pip install "fastapi" "uvicorn[standard]" "python-multipart" || ( echo pip install failed & pause & exit /b 1 )

rem ---- copy the server in ----
copy /Y "%HERE%lora_upload_server.py" "%DEST%\lora_upload_server.py" >nul

rem ---- write the launcher ----
> "%DEST%\run_lora_upload.bat" (
  echo @echo off
  echo setlocal
  echo set "HERE=%%~dp0"
  echo set "MG_LORA_PORT=%PORT%"
  echo set "MG_LORA_DIR=%%HERE%%lora_data"
  echo "%%HERE%%venv\Scripts\python.exe" "%%HERE%%lora_upload_server.py"
)

echo(
echo ============================================================
echo  Installed. Start the helper with:
echo     "%DEST%\run_lora_upload.bat"
echo  It listens on port %PORT%. In Crucible's app_config.json set:
echo     "lora_upload_host": "^<this-box-ip^>:%PORT%"
echo  Datasets upload to: %DEST%\lora_data\^<dataset^>\
echo ============================================================
pause
