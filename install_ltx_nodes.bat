@echo off
rem Install the LTX-2.3 custom nodes into our EXISTING ComfyUI portable (does NOT reinstall
rem ComfyUI). Run from the ComfyUI_windows_portable root, then RESTART ComfyUI.
rem Needs git in PATH. Nodes already present (GGUF, rgthree, KJNodes, VideoHelperSuite,
rem Manager) are skipped/pulled.
setlocal enabledelayedexpansion
cd /d "%~dp0"
set "PY=%~dp0python_embeded\python.exe"
set "CN=%~dp0ComfyUI\custom_nodes"
where git >nul 2>nul || (echo [ERROR] git not found in PATH.& pause & exit /b 1)

call :node "ComfyUI-LTXVideo" "https://github.com/Lightricks/ComfyUI-LTXVideo"
call :node "RES4LYF" "https://github.com/ClownsharkBatwing/RES4LYF"
call :node "ComfyUI-Easy-Use" "https://github.com/yolain/ComfyUI-Easy-Use"
call :node "ComfyMath" "https://github.com/evanspearman/ComfyMath"
call :node "ComfyUI-Custom-Scripts" "https://github.com/pythongosssss/ComfyUI-Custom-Scripts"
call :node "ComfyUI-Impact-Pack" "https://github.com/ltdrdata/ComfyUI-Impact-Pack"
call :node "Comfyui_TTP_Toolset" "https://github.com/TTPlanetPig/Comfyui_TTP_Toolset"
call :node "WhatDreamsCost-ComfyUI" "https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI"
call :node "ComfyUI-WanVideoWrapper" "https://github.com/kijai/ComfyUI-WanVideoWrapper"

echo.
echo == Done. RESTART ComfyUI to load the new nodes. ==
pause
exit /b 0

:node
set "NAME=%~1"
set "URL=%~2"
echo.
echo == %NAME% ==
if exist "%CN%\%NAME%" ( git -C "%CN%\%NAME%" pull --ff-only ) else ( git clone --depth 1 "%URL%" "%CN%\%NAME%" )
if exist "%CN%\%NAME%\requirements.txt" "%PY%" -s -m pip install -r "%CN%\%NAME%\requirements.txt"
exit /b 0
