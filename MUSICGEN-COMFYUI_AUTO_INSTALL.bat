@echo off
setlocal enabledelayedexpansion

rem -------------------------------------------------------------
rem  ComfyUI + ACE-Step one-click installer for MusicGen
rem  (rock/metal music + vocals, RTX 3090 24GB)
rem  Style adapted from Aitrepreneur's ERNIE/Z-IMAGE installer
rem -------------------------------------------------------------

:: Version bump zone
set "COMFY_VER=v0.22.0"

:: ---------- MODEL CHOICE ----------
:CHOOSE_MODEL
echo(
echo Which ACE-Step 1.5 model? (music + vocals)
echo 1) Turbo AIO  - single file, fast, easiest          RECOMMENDED to start
echo 2) XL (base)  - best quality, split files, 24GB GPU
set /p "MODEL_CHOICE=Enter 1 or 2: "
if "!MODEL_CHOICE!"=="1" (set "ACE_MODE=AIO") ^
else if "!MODEL_CHOICE!"=="2" (set "ACE_MODE=XL") ^
else (
    echo Invalid choice.
    timeout /t 2 >nul
    goto CHOOSE_MODEL
)

:: ---------- DEMUCS CHOICE ----------
:CHOOSE_DEMUCS
echo(
echo Also install Demucs stem separation? (for base-track restyle)
echo 1) No
echo 2) Yes
set /p "DEMUCS_CHOICE=Enter 1 or 2: "
if "!DEMUCS_CHOICE!"=="1" (set "INSTALL_DEMUCS=0") ^
else if "!DEMUCS_CHOICE!"=="2" (set "INSTALL_DEMUCS=1") ^
else (
    echo Invalid choice.
    timeout /t 2 >nul
    goto CHOOSE_DEMUCS
)

:: ---------- CONSTANTS ----------
set "ACE=https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files/resolve/main"
set "COMFY_RELEASE=https://github.com/comfyanonymous/ComfyUI/releases/download/%COMFY_VER%/ComfyUI_windows_portable_nvidia.7z"

echo(
echo -------- Checking prerequisites --------
call :ensure_7zip || exit /b 1
call :ensure_git  || exit /b 1

echo(
echo -------- Downloading ComfyUI --------
curl -L -o ComfyUI.7z "%COMFY_RELEASE%" --ssl-no-revoke
if errorlevel 1 (
    echo Download failed.
    pause
    exit /b 1
)

echo -------- Extracting ComfyUI --------
"%SEVEN_ZIP_PATH%" x ComfyUI.7z -aoa -o"%CD%" >nul
del ComfyUI.7z
if not exist "ComfyUI_windows_portable" (
    echo Extraction failed.
    pause
    exit /b 1
)

set "ROOT=%CD%"
pushd "ComfyUI_windows_portable"

rem Upstream uses "python_embeded"
set "PY=%CD%\python_embeded\python.exe"

echo(
echo -------- Installing custom nodes --------
pushd ComfyUI\custom_nodes

call :clone https://github.com/ltdrdata/ComfyUI-Manager.git
if exist ComfyUI-Manager\requirements.txt "%PY%" -m pip install -r ComfyUI-Manager\requirements.txt

call :clone https://github.com/rgthree/rgthree-comfy
if exist rgthree-comfy\requirements.txt "%PY%" -m pip install -r rgthree-comfy\requirements.txt

popd

echo(
echo -------- Downloading ACE-Step 1.5 model (!ACE_MODE!) --------
pushd ComfyUI\models

if "!ACE_MODE!"=="AIO" (
    :: --- Single all-in-one checkpoint: uses the "ACE-Step 1.5 AIO" template ---
    call :grab checkpoints\ace_step_1.5_turbo_aio.safetensors ^
         "%ACE%/checkpoints/ace_step_1.5_turbo_aio.safetensors?download=true"
) else (
    :: --- XL split files: uses the "ACE-Step 1.5 (split)" template ---
    :: NOTE: the split template's text encoder is the DUAL qwen_0.6b + qwen_1.7b
    :: pair (per ComfyUI docs). qwen_4b is fetched too for experimentation.
    call :grab diffusion_models\acestep_v1.5_xl_base_bf16.safetensors ^
         "%ACE%/split_files/diffusion_models/acestep_v1.5_xl_base_bf16.safetensors?download=true"
    call :grab text_encoders\qwen_0.6b_ace15.safetensors ^
         "%ACE%/split_files/text_encoders/qwen_0.6b_ace15.safetensors?download=true"
    call :grab text_encoders\qwen_1.7b_ace15.safetensors ^
         "%ACE%/split_files/text_encoders/qwen_1.7b_ace15.safetensors?download=true"
    call :grab text_encoders\qwen_4b_ace15.safetensors ^
         "%ACE%/split_files/text_encoders/qwen_4b_ace15.safetensors?download=true"
    call :grab vae\ace_1.5_vae.safetensors ^
         "%ACE%/split_files/vae/ace_1.5_vae.safetensors?download=true"
)

popd & popd

if "!INSTALL_DEMUCS!"=="1" (
    echo(
    echo -------- Installing Demucs stem separation --------
    "%ROOT%\ComfyUI_windows_portable\python_embeded\python.exe" -m pip install demucs
    if errorlevel 1 echo   [!] Demucs install failed - you can retry later.
)

echo(
echo -------- Creating LAN launcher for Mac access --------
set "LANBAT=%ROOT%\ComfyUI_windows_portable\run_musicgen_lan.bat"
> "%LANBAT%" echo @echo off
>> "%LANBAT%" echo rem Launch ComfyUI bound to the LAN so the Mac app can reach it.
>> "%LANBAT%" echo rem Open http://^<this-PC-IP^>:8188 from the Mac (run ipconfig to find the IP).
>> "%LANBAT%" echo .\python_embeded\python.exe -s ComfyUI\main.py --listen 0.0.0.0 --windows-standalone-build
>> "%LANBAT%" echo pause

echo(
echo -------------------------------------------------------------
echo   Install complete!
echo   - Localhost only:  run_nvidia_gpu.bat
echo   - LAN (for Mac):   run_musicgen_lan.bat   then open
echo                      http://^<this-PC-IP^>:8188  from the Mac
if "!ACE_MODE!"=="AIO" (
    echo   In ComfyUI: Workflow -^> Browse Templates -^> Audio -^>
    echo               "ACE-Step 1.5 Music Generation AIO"
) else (
    echo   In ComfyUI: Workflow -^> Browse Templates -^> Audio -^>
    echo               "ACE-Step 1.5 Music Generation Workflow (split)"
)
echo -------------------------------------------------------------
echo Launching ComfyUI on the LAN now...
pushd "%ROOT%\ComfyUI_windows_portable"
call run_musicgen_lan.bat
popd
echo(
pause
exit /b


:: ================= helper routines =================

:ensure_7zip
rem Try PATH first
set "SEVEN_ZIP_PATH="
for %%I in (7z.exe) do (
    if exist "%%~$PATH:I" (
        set "SEVEN_ZIP_PATH=%%~$PATH:I"
    )
)
if defined SEVEN_ZIP_PATH (
    exit /b 0
)

rem Try common install folders
if exist "%ProgramFiles%\7-Zip\7z.exe" (
    set "SEVEN_ZIP_PATH=%ProgramFiles%\7-Zip\7z.exe"
    exit /b 0
) else if exist "%ProgramFiles(x86)%\7-Zip\7z.exe" (
    set "SEVEN_ZIP_PATH=%ProgramFiles(x86)%\7-Zip\7z.exe"
    exit /b 0
)

echo 7-Zip not found. Trying to install with winget...

where winget >nul 2>&1
if errorlevel 1 (
    echo winget is not available on this system.
    echo Please install 7-Zip manually from:
    echo   https://www.7-zip.org/download.html
    pause
    exit /b 1
)

winget install -e --id 7zip.7zip --accept-package-agreements --accept-source-agreements
if errorlevel 1 (
    echo Failed to install 7-Zip via winget.
    echo Please install 7-Zip manually from:
    echo   https://www.7-zip.org/download.html
    pause
    exit /b 1
)

rem Try again to locate 7z.exe
set "SEVEN_ZIP_PATH="
for %%I in (7z.exe) do (
    if exist "%%~$PATH:I" (
        set "SEVEN_ZIP_PATH=%%~$PATH:I"
    )
)
if not defined SEVEN_ZIP_PATH (
    if exist "%ProgramFiles%\7-Zip\7z.exe" set "SEVEN_ZIP_PATH=%ProgramFiles%\7-Zip\7z.exe"
)
if not defined SEVEN_ZIP_PATH (
    if exist "%ProgramFiles(x86)%\7-Zip\7z.exe" set "SEVEN_ZIP_PATH=%ProgramFiles(x86)%\7-Zip\7z.exe"
)

if defined SEVEN_ZIP_PATH (
    exit /b 0
) else (
    echo 7-Zip seems installed but 7z.exe was not found.
    echo Please check your installation and rerun this script.
    pause
    exit /b 1
)

:ensure_git
echo Checking for Git...
git --version >nul 2>&1
if not errorlevel 1 (
    echo Git is already installed.
    exit /b 0
)

echo Git not found. Trying to install with winget...

where winget >nul 2>&1
if errorlevel 1 (
    echo winget is not available on this system.
    echo Please install Git manually from:
    echo   https://git-scm.com/download/win
    pause
    exit /b 1
)

winget install -e --id Git.Git --accept-package-agreements --accept-source-agreements
if errorlevel 1 (
    echo.
    echo Failed to install Git via winget.
    echo Please install Git manually from:
    echo   https://git-scm.com/download/win
    pause
    exit /b 1
)

echo Git installed successfully. Verifying...
git --version >nul 2>&1
if errorlevel 1 (
    echo Git is installed but not yet available in this terminal session.
    echo Close this window and run the installer again.
    pause
    exit /b 1
)

exit /b 0

:clone
git clone %* >nul 2>&1
if errorlevel 1 echo   [!] Clone failed: %~1
goto :eof

:grab
if not exist "%~dp1" mkdir "%~dp1"
if not exist "%~1" (
    echo   - downloading %~nx1
    curl -L -o "%~1" "%~2" --ssl-no-revoke
    if errorlevel 1 echo     [!] Download failed: %~nx1
) else (
    echo   - %~nx1 already present - skipping
)
goto :eof
