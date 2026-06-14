@echo off
rem Download the LTX-2.3 video models (~48GB, ungated) with ComfyUI's portable python.
rem Run from the ComfyUI_windows_portable root.
setlocal
cd /d "%~dp0"
"%~dp0python_embeded\python.exe" "%~dp0download_ltx_models.py"
pause
