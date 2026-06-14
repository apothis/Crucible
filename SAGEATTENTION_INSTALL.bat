@echo off
rem Install SageAttention (+ triton) into ComfyUI's portable python. SageAttention is the
rem standard accelerator for Wan / video diffusion in ComfyUI: much faster attention and
rem far less VRAM than the default pytorch attention - this is the likely reason past Wan/LTX
rem runs on this GPU were fast. Run from the ComfyUI_windows_portable root, then add the
rem launch flag (below) and RESTART ComfyUI.
setlocal
cd /d "%~dp0"
set "PY=python_embeded\python.exe"

echo == installing triton-windows + sageattention into python_embeded ==
"%PY%" -m pip install -U triton-windows
"%PY%" -m pip install -U sageattention

echo.
echo == verify it imports ==
"%PY%" -c "import sageattention, triton; print('sageattention + triton OK')" || (
  echo [WARN] import failed - likely a triton/torch (2.11 / cu130) version mismatch.
  echo        Tell Claude the error; we may need a specific triton-windows wheel.
)

echo.
echo == NEXT: add --use-sage-attention to your ComfyUI launcher, e.g.:
echo    .\python_embeded\python.exe -s ComfyUI\main.py --listen 0.0.0.0 --use-sage-attention --windows-standalone-build
echo Then restart ComfyUI. (Claude can patch run_musicgen_lan.bat for you once the import above is OK.)
pause
