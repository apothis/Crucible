@echo off
rem Meridian camera re-shoot for MiniMax H3 (community INT8 build) + MoGe geometry.
rem About 26 GB. Pure curl with resume: re-run to continue an interrupted download.
cd /d "%~dp0"
set MODELS=%~dp0ComfyUI\models
set NYCK=https://huggingface.co/NyckM/Meridian_CameraH3_INT8_build_by_BruxosdoVFX/resolve/main

if not exist "%MODELS%\diffusion_models\H3Camera" mkdir "%MODELS%\diffusion_models\H3Camera"
if not exist "%MODELS%\loras" mkdir "%MODELS%\loras"
if not exist "%MODELS%\text_cond" mkdir "%MODELS%\text_cond"
if not exist "%MODELS%\text_cond\meridian_pt" mkdir "%MODELS%\text_cond\meridian_pt"
if not exist "%MODELS%\geometry_estimation" mkdir "%MODELS%\geometry_estimation"

echo [1/5] Meridian H3 camera model, INT8 pruned (21 GB)
curl -L -C - -o "%MODELS%\diffusion_models\H3Camera\MeridianH3Camera_int8_pruned.safetensors" "%NYCK%/diffusion_models/H3camera/MeridianH3Camera_int8_pruned.safetensors"

echo [2/5] Meridian DMD LoRA (3.8 GB)
curl -L -C - -o "%MODELS%\loras\meridian_dmd_lora_comfyui.safetensors" "%NYCK%/lora/meridian_dmd_lora_comfyui.safetensors"

echo [3/5] H3 3-step LoRA (Kijai, 0.2 GB)
curl -L -C - -o "%MODELS%\loras\minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors" "https://huggingface.co/Kijai/MiniMax-H3_comfy/resolve/main/loras/minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors"

echo [4/5] MoGe 2 geometry model (0.7 GB)
curl -L -C - -o "%MODELS%\geometry_estimation\moge_2_vitl_normal_fp16.safetensors" "https://huggingface.co/Comfy-Org/MoGe/resolve/main/geometry_estimation/moge_2_vitl_normal_fp16.safetensors"

echo [5/5] Meridian frozen text embeddings (tiny)
curl -L -C - -o "%MODELS%\text_cond\meridian_fixed_embed_124.safetensors" "%NYCK%/text_cond/meridian_fixed_embed_124.safetensors"
for %%N in (73 90 107 124 141 158 175 243) do curl -L -C - -o "%MODELS%\text_cond\meridian_pt\fixed_embed_%%N.pt" "https://huggingface.co/Viggle/Meridian/resolve/main/assets/fixed_embed_%%N.pt"

echo.
echo Done. Tell Claude it finished.
pause
