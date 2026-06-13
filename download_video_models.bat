@echo off
rem Download the Wan2.2 video pipeline models with ComfyUI's portable python.
rem Run from the ComfyUI_windows_portable root.
rem   download_video_models.bat        -> Phase 1 gate subset (~45-55 GB)
rem   download_video_models.bat full   -> adds the 14B i2v hero pair (full set)
rem All models are ungated (Apache 2.0 / Comfy-Org repackaged). No HF token needed.
setlocal
cd /d "%~dp0"
set "MODE=%~1"
if "%MODE%"=="" set "MODE=min"
"%~dp0python_embeded\python.exe" "%~dp0download_video_models.py" %MODE%
pause
