@echo off
rem Install the video-pipeline custom nodes into this ComfyUI portable.
rem Run from the ComfyUI_windows_portable root (where python_embeded\ and ComfyUI\ live).
rem Needs git in PATH (ComfyUI-Manager already relies on it). After it finishes,
rem RESTART ComfyUI so the new nodes load.
setlocal enabledelayedexpansion
cd /d "%~dp0"
set "PY=%~dp0python_embeded\python.exe"
set "CN=%~dp0ComfyUI\custom_nodes"

where git >nul 2>nul
if errorlevel 1 (
  echo [ERROR] git not found in PATH. Install Git for Windows, or add the nodes via the
  echo         ComfyUI-Manager UI ^(Manager -^> Install Custom Nodes^). Aborting.
  pause
  exit /b 1
)

call :node "ComfyUI-VideoHelperSuite" "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
call :node "ComfyUI-Frame-Interpolation" "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation"
call :node "ComfyUI_Yvann-Nodes" "https://github.com/yvann-ba/ComfyUI_Yvann-Nodes"
call :node "ComfyUI-GGUF" "https://github.com/city96/ComfyUI-GGUF"
call :node "ComfyUI-KJNodes" "https://github.com/kijai/ComfyUI-KJNodes"

echo.
echo == Done. RESTART ComfyUI to load the new nodes. ==
pause
exit /b 0

:node
set "NAME=%~1"
set "URL=%~2"
echo.
echo == %NAME% ==
if exist "%CN%\%NAME%" (
  echo   already present, pulling latest...
  git -C "%CN%\%NAME%" pull --ff-only
) else (
  git clone --depth 1 "%URL%" "%CN%\%NAME%"
)
if exist "%CN%\%NAME%\requirements.txt" (
  echo   installing requirements...
  "%PY%" -s -m pip install -r "%CN%\%NAME%\requirements.txt"
)
exit /b 0
