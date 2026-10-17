@echo off
REM Run one AI Toolkit job file (captioning or training) with the toolkit's own Python.
REM Usage: run_aitk_job.bat <job name>   e.g.  run_aitk_job.bat yue2_battlebeast_caption
REM The job file is ai-toolkit\config\<job name>.yaml (written there by the Mac app).
REM Output goes to ai-toolkit\output\<job name>.log as well as this window, so progress can be
REM read from the Mac. Model downloads (tokenizer head, MERT, captioner) are kept on this drive
REM in ai-toolkit\.hf instead of the user profile on C:.
REM Close ComfyUI renders and the ACE engine first: the job needs the GPU to itself.

setlocal
cd /d "%~dp0..\ai-toolkit"
if "%~1"=="" (
    echo Usage: run_aitk_job.bat ^<job name^>
    pause
    exit /b 1
)
if not exist "config\%~1.yaml" (
    echo No job file config\%~1.yaml
    pause
    exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
    echo AI Toolkit is not installed yet - run install_ai_toolkit.bat first
    pause
    exit /b 1
)
set "HF_HOME=%CD%\.hf"
if not exist output mkdir output

".venv\Scripts\python.exe" run.py "config\%~1.yaml" --log "output\%~1.log"
echo.
echo Job %~1 finished (exit code %errorlevel%).
pause
