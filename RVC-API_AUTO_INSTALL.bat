@echo off
setlocal enabledelayedexpansion

rem -------------------------------------------------------------
rem  MusicGen RVC API installer
rem  Installs a small clean REST API (rvc_server.py) INTO your
rem  existing RVC WebUI package, reusing its working Python env
rem  (no new installs, no fairseq build). Replaces the Gradio UI
rem  with a clean /convert + voice-upload API on port 5050.
rem
rem  KEEP rvc_server.py IN THE SAME FOLDER AS THIS .BAT.
rem -------------------------------------------------------------

set "PORT=5050"
set "HERE=%~dp0"

if not exist "%HERE%rvc_server.py" (
    echo ERROR: rvc_server.py must be in the same folder as this installer.
    pause
    exit /b 1
)

:ASK_PATH
echo(
echo Enter the full path to your RVC folder
echo (the one containing runtime\python.exe and infer-web.py),
echo e.g.  C:\AI\RVC20240604Nvidia
set /p "RVC=Path: "
if not exist "%RVC%\runtime\python.exe" (
    echo   Not found: "%RVC%\runtime\python.exe"  - check the path.
    goto ASK_PATH
)
if not exist "%RVC%\infer-web.py" (
    echo   Warning: infer-web.py not found there - is this the RVC package root?
    goto ASK_PATH
)

echo(
echo -------- Installing API server into the RVC package --------
copy /Y "%HERE%rvc_server.py" "%RVC%\rvc_server.py" >nul
if errorlevel 1 ( echo copy failed & pause & exit /b 1 )

:: ---------- starter voices (optional) ----------
set "FREDDIE_PTH=https://huggingface.co/binant/Freddie_Mercury__RVC_-_700_Epochs_/resolve/main/model.pth?download=true"
set "FREDDIE_IDX=https://huggingface.co/binant/Freddie_Mercury__RVC_-_700_Epochs_/resolve/main/model.index?download=true"
set "HETFIELD_ZIP=https://huggingface.co/Endergamer2121/James_Hetfield/resolve/main/james_hetfield.zip?download=true"

:CHOOSE_VOICES
echo(
echo Download starter metal voices? (James Hetfield + Freddie Mercury, ~450 MB)
echo 1) Yes  2) No
set /p "VOICE_CHOICE=Enter 1 or 2: "
if "!VOICE_CHOICE!"=="1" (set "GET_VOICES=1") ^
else if "!VOICE_CHOICE!"=="2" (set "GET_VOICES=0") ^
else ( echo Invalid choice. & timeout /t 2 >nul & goto CHOOSE_VOICES )

if "!GET_VOICES!"=="1" (
    echo -------- Downloading starter voices --------
    if not exist "%RVC%\assets\weights" mkdir "%RVC%\assets\weights"
    if not exist "%RVC%\logs" mkdir "%RVC%\logs"
    call :grab "%RVC%\assets\weights\FreddieMercury.pth" "%FREDDIE_PTH%"
    call :grab "%RVC%\logs\FreddieMercury.index" "%FREDDIE_IDX%"
    call :grabzip_split "%HETFIELD_ZIP%" "%RVC%"
)

echo(
echo -------- Creating API launcher --------
set "LAUNCH=%RVC%\run_rvc_api.bat"
> "%LAUNCH%" echo @echo off
>> "%LAUNCH%" echo rem MusicGen RVC API on the LAN (port %PORT%). Use this INSTEAD of go-web.bat.
>> "%LAUNCH%" echo cd /d "%%~dp0"
>> "%LAUNCH%" echo set MG_PORT=%PORT%
>> "%LAUNCH%" echo runtime\python.exe rvc_server.py
>> "%LAUNCH%" echo pause

echo(
echo -------------------------------------------------------------
echo   Installed. The clean RVC API replaces the Gradio UI.
echo   START IT:  %RVC%\run_rvc_api.bat
echo   Reachable: http://^<this-PC-IP^>:%PORT%   (rvc_driver auto-detects it)
echo   Voices live in %RVC%\assets\weights ; add more via the app's
echo   "Add voices" helper or by dropping .pth there (+ .index in \logs).
echo -------------------------------------------------------------
echo Launching the API now...
pushd "%RVC%"
call run_rvc_api.bat
popd
echo(
pause
exit /b


:: ================= helper routines =================

:grab
if not exist "%~dp1" mkdir "%~dp1"
if not exist "%~1" (
    echo   - downloading %~nx1
    curl -L -o "%~1" "%~2" --ssl-no-revoke
    if errorlevel 1 echo     [!] download failed: %~nx1
) else (
    echo   - %~nx1 already present - skipping
)
goto :eof

rem download a flat voice zip and split .pth -> assets\weights, .index -> logs
:grabzip_split
set "TMPD=%TEMP%\mg_voice_%RANDOM%"
mkdir "%TMPD%" >nul 2>&1
echo   - downloading voice archive...
curl -L -o "%TMPD%\v.zip" "%~1" --ssl-no-revoke
if errorlevel 1 ( echo     [!] zip download failed & goto :eof )
powershell -NoProfile -Command "Expand-Archive -LiteralPath '%TMPD%\v.zip' -DestinationPath '%TMPD%\x' -Force"
for /r "%TMPD%\x" %%F in (*.pth) do copy /Y "%%F" "%~2\assets\weights\" >nul
for /r "%TMPD%\x" %%F in (*.index) do copy /Y "%%F" "%~2\logs\" >nul
rmdir /s /q "%TMPD%" >nul 2>&1
goto :eof
