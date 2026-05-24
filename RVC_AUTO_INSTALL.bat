@echo off
setlocal enabledelayedexpansion

rem -------------------------------------------------------------
rem  RVC (Retrieval-based Voice Conversion) one-click installer
rem  for MusicGen vocal timbre conversion - RTX 3090 / Windows
rem  Style adapted from Aitrepreneur's ComfyUI portable installer
rem -------------------------------------------------------------

:: ---------- CONSTANTS ----------
:: Bundled package: includes its own Python runtime, ffmpeg, and all
:: models (hubert, rmvpe, pretrained_v2, uvr5). No pip / no git needed.
:: NOTE: hosted on Hugging Face - GitHub releases are stale.
set "RVC_PKG=RVC20240604Nvidia.7z"
set "RVC_URL=https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/%RVC_PKG%"
set "RVC_DIR=RVC20240604Nvidia"

echo(
echo -------- Checking prerequisites --------
call :ensure_7zip || exit /b 1

echo(
echo -------- Downloading RVC (~5.6 GB, this takes a while) --------
if not exist "%RVC_PKG%" (
    curl -L -o "%RVC_PKG%" "%RVC_URL%" --ssl-no-revoke
    if errorlevel 1 (
        echo Download failed.
        pause
        exit /b 1
    )
) else (
    echo   - %RVC_PKG% already present - skipping download
)

echo -------- Extracting RVC --------
"%SEVEN_ZIP_PATH%" x "%RVC_PKG%" -aoa -o"%CD%" >nul
if not exist "%RVC_DIR%" (
    echo Extraction failed.
    pause
    exit /b 1
)
del "%RVC_PKG%"

set "ROOT=%CD%"

echo(
echo -------------------------------------------------------------
echo   RVC install complete!
echo   Launching the web UI - it opens in your browser on port 7897.
echo   (go-web.bat = normal UI, go-realtime-gui.bat = live changer)
echo -------------------------------------------------------------
pushd "%ROOT%\%RVC_DIR%"
call go-web.bat
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
