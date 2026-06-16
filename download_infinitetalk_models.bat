@echo off
rem ============================================================================
rem InfiniteTalk video-to-video lip-sync models for the Music Video pipeline.
rem Keep the existing walking/motion footage, redrive only the mouth from audio.
rem
rem Pure curl with resume ( -L follow redirects, -C - resume a partial file ).
rem No PowerShell, no extraction, no python. Re-run any time to resume.
rem All files are ungated (Kijai / Comfy-Org public repos). No HF token needed.
rem Total ~24 GB. Windows 10/11 ships curl.exe so this runs as-is.
rem ============================================================================
rem Place this in the ComfyUI_windows_portable folder (same as the other installers) and run it.
setlocal
cd /d "%~dp0"
set MODELS=%~dp0ComfyUI\models

rem curl can write a file but cannot create a missing parent dir - make sure each target exists.
if not exist "%MODELS%\diffusion_models" mkdir "%MODELS%\diffusion_models"
if not exist "%MODELS%\loras" mkdir "%MODELS%\loras"
if not exist "%MODELS%\clip_vision" mkdir "%MODELS%\clip_vision"
if not exist "%MODELS%\wav2vec2" mkdir "%MODELS%\wav2vec2"

echo.
echo [1/5] InfiniteTalk model (5.1 GB) -^> diffusion_models
curl -L -C - -o "%MODELS%\diffusion_models\Wan2_1-InfiniTetalk-Single_fp16.safetensors" "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/InfiniteTalk/Wan2_1-InfiniTetalk-Single_fp16.safetensors"

echo.
echo [2/5] Wan2.1 i2v 14B 480p base fp8 (17 GB) -^> diffusion_models
curl -L -C - -o "%MODELS%\diffusion_models\Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors" "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors"

echo.
echo [3/5] lightx2v i2v 4-step distill LoRA (0.74 GB) -^> loras
curl -L -C - -o "%MODELS%\loras\lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors" "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"

echo.
echo [4/5] CLIP vision H (1.26 GB) -^> clip_vision
curl -L -C - -o "%MODELS%\clip_vision\clip_vision_h.safetensors" "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors"

echo.
echo [5/5] wav2vec2 chinese-base audio encoder (0.19 GB) -^> wav2vec2
curl -L -C - -o "%MODELS%\wav2vec2\wav2vec2-chinese-base_fp16.safetensors" "https://huggingface.co/Kijai/wav2vec2_safetensors/resolve/main/wav2vec2-chinese-base_fp16.safetensors"

echo.
echo Done. All 5 InfiniteTalk files downloaded. Restart ComfyUI to load them.
pause
