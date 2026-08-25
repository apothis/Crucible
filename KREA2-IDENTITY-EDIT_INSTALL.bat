@echo off
rem Krea 2 Identity Edit setup (identity-preserving re-staging on Krea2 Turbo).
rem Run from the ComfyUI_windows_portable root (where python_embeded\ and ComfyUI\ live).
rem Installs the ComfyUI-Krea2Edit node pack (github lbouaraba/comfyui-krea2edit).
rem The LoRA itself is a MANUAL download (Civitai needs a login/VPN):
rem   https://civitai.com/models/2761113/krea-2-identity-edit
rem   file: krea2_identity_edit_v1_2.safetensors (1.7 GB, the full fp16 build)
rem   put it in: ComfyUI\models\loras\Krea2\
rem After both: RESTART ComfyUI so the nodes load.
setlocal enabledelayedexpansion
cd /d "%~dp0"
set "CN=%~dp0ComfyUI\custom_nodes"
set "LDIR=%~dp0ComfyUI\models\loras\Krea2"

where git >nul 2>nul
if errorlevel 1 (
  echo [ERROR] git not found in PATH. Install Git for Windows, or add the node via
  echo         ComfyUI-Manager ^(search: comfyui-krea2edit^). Aborting.
  pause
  exit /b 1
)

if exist "%CN%\comfyui-krea2edit" (
  echo [SKIP] comfyui-krea2edit already present - pulling latest...
  pushd "%CN%\comfyui-krea2edit" && git pull && popd
) else (
  git clone https://github.com/lbouaraba/comfyui-krea2edit "%CN%\comfyui-krea2edit"
  if errorlevel 1 ( echo [ERROR] clone failed & pause & exit /b 1 )
)

if not exist "%LDIR%" mkdir "%LDIR%"
if exist "%LDIR%\krea2_identity_edit_v1_2.safetensors" (
  echo [OK] LoRA found: %LDIR%\krea2_identity_edit_v1_2.safetensors
) else (
  echo [TODO] LoRA missing. Download krea2_identity_edit_v1_2.safetensors from
  echo        https://civitai.com/models/2761113/krea-2-identity-edit
  echo        and place it in %LDIR%\
)

echo.
echo == Done. RESTART ComfyUI to load the new nodes. ==
pause
exit /b 0
