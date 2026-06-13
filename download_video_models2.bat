@echo off
rem Download the RECOMMENDED music-video pipeline models (GGUF) with ComfyUI's portable
rem python. Run from the ComfyUI_windows_portable root. Companion to download_video_models.bat.
rem All models are ungated (QuantStack / unsloth / Comfy-Org / Kim2091). No HF token needed.
rem ~80GB total. GGUF UNETs load via the installed ComfyUI-GGUF node (UnetLoaderGGUF).
setlocal
cd /d "%~dp0"
"%~dp0python_embeded\python.exe" "%~dp0download_video_models2.py"
pause
