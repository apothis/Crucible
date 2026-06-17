"""ComfyUI video-pipeline graph builders (Phase 1 gate: still -> i2v -> lip-sync).

Wiring mirrors the official ComfyUI templates verified on the box (RESEARCH.md s19a):
  - Z-Image Turbo still: UNETLoader(z_image_turbo) + CLIPLoader(qwen_3_4b, type=lumina2)
    + VAELoader(ae) + ModelSamplingAuraFlow(shift=3) + KSampler(8 steps, cfg 1,
    res_multistep/simple) + SaveImage.
  - Wan2.2 5B TI2V image-to-video: UNETLoader(wan2.2_ti2v_5B) + CLIPLoader(umt5, type=wan)
    + VAELoader(wan2.2_vae) + ModelSamplingSD3(shift=8) + Wan22ImageToVideoLatent(start_image)
    + KSampler(uni_pc/simple) + VAEDecode + CreateVideo + SaveVideo.
  - Wan2.2-S2V single-clip lip-sync: UNETLoader(wan2.2_s2v_14B_fp8) + AudioEncoderLoader(wav2vec2)
    + AudioEncoderEncode + WanSoundImageToVideo(ref_image+audio) + KSampler + CreateVideo(audio).

All builders return (graph_dict, resolved_params) like comfy.build_* do. Output node
filename_prefix routes the result into ComfyUI's output/ folder; app.on_complete fetches it.

Plain ASCII only.
"""
import random

# ---- model files on the box (downloaded by download_video_models.py) ----
Z_IMAGE_UNET = "z_image_turbo_bf16.safetensors"
Z_IMAGE_CLIP = "qwen_3_4b.safetensors"           # CLIPLoader type "lumina2"
Z_IMAGE_VAE = "ae.safetensors"

WAN_TI2V = "wan2.2_ti2v_5B_fp16.safetensors"
WAN_CLIP = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"   # CLIPLoader type "wan"
WAN22_VAE = "wan2.2_vae.safetensors"                  # 5B TI2V VAE
WAN21_VAE = "wan_2.1_vae.safetensors"                 # 14B / S2V VAE

WAN_S2V = "wan2.2_s2v_14B_fp8_scaled.safetensors"
WAV2VEC = "wav2vec2_large_english_fp16.safetensors"

# SeedVR2 (numz/ComfyUI-SeedVR2_VideoUpscaler) diffusion video upscaler. Models auto-download
# on first use into models/SEEDVR2 (no HF token). 3B fp16 fits 24GB with offload none; the 7B
# variants want block-swap. attention sdpa (Ampere has no fp8/flash3; sage stays off per our
# fp8 rule). batch_size MUST be 4n+1 (1,5,9...): larger = better temporal consistency + VRAM.
SEEDVR2_DIT_DEFAULT = "seedvr2_ema_3b_fp16.safetensors"
SEEDVR2_VAE = "ema_vae_fp16.safetensors"
# lightx2v 4-step distillation LoRA (the official S2V template applies the t2v high-noise
# one to the single S2V model): cuts 20 steps -> 4 and CFG 6 -> 1 (~10x fewer DiT evals).
WAN_LIGHTX_HIGH = "wan2.2_t2v_A14b_high_noise_lora_rank64_lightx2v_4step_1217.safetensors"

# Wan2.2-S2V via the Kijai WanVideoWrapper (block-swap) - the path that actually fits the
# 14B on a 24GB 3090. Recipe traced verbatim from the wrapper's own example workflow
# (custom_nodes/ComfyUI-WanVideoWrapper/s2v/wanvideo2_2_S2V_context_window_testing.json):
# WanVideoModelLoader(attention=sageattn, fp8_e4m3fn_scaled, offload_device) - because the
# wrapper casts dtype correctly, sage ENGAGES here (no fp32 fallback like the native nodes);
# block-swap streams blocks to RAM so it fits; lightx2v v2 distill -> 4 steps; wav2vec2 audio.
WAN_S2V_KJ = "Wan2_2-S2V-14B_fp8_e4m3fn_scaled_KJ.safetensors"   # diffusion_models (Kijai repack)
WAN21_VAE_BF16 = "Wan2_1_VAE_bf16.safetensors"                   # vae
UMT5_ENC_BF16 = "umt5-xxl-enc-bf16.safetensors"                  # text_encoders
LIGHTX2V_V2 = "lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors"  # loras

# InfiniteTalk video-to-video lip-sync (keep the source footage's motion/camera/background,
# redrive only the mouth/face from audio). Built on Wan2.1 i2v 14B + the MultiTalk infra that
# the Kijai wrapper already ships (MultiTalkModelLoader auto-detects InfiniteTalk by filename).
# Recipe traced verbatim from the wrapper's wanvideo_2_1_14B_V2V_InfiniteTalk_example_02.json.
INFINITETALK = "Wan2_1-InfiniTetalk-Single_fp16.safetensors"            # diffusion_models
WAN21_I2V_FP8 = "Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors"           # diffusion_models (2.1 base)
LIGHTX2V_I2V_480P = "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"  # loras (4-step)
CLIP_VISION_H = "clip_vision_h.safetensors"                            # clip_vision (Wan2.1 i2v needs it)
WAV2VEC_CN_BASE = "wav2vec2-chinese-base_fp16.safetensors"             # wav2vec2 (768-dim, MultiTalk)

# Qwen-Image-Edit-2511 (GGUF) - reference-driven character consistency, no training.
# Files from download_video_models2.py. GGUF UNET loads via UnetLoaderGGUF (ComfyUI-GGUF).
QWEN_EDIT_GGUF = "qwen-image-edit-2511-Q6_K.gguf"        # legacy GGUF (UnetLoaderGGUF)
# fp8 migration (same rationale as LTX above): clean-unloading scaled fp8 safetensors via
# UNETLoader weight_dtype "default". Keeps the CLIPLoader + VAELoader + edit wiring unchanged.
QWEN_EDIT_FP8 = "qwen_image_edit_2511_fp8mixed.safetensors"  # UNETLoader
QWEN_CLIP = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
QWEN_VAE = "qwen_image_vae.safetensors"

# Wan2.1 VACE 14B (GGUF) - reference-to-VIDEO: holds a subject's identity from a reference
# image straight through motion (the video-side consistency alternative to Qwen stills).
WAN_VACE_GGUF = "Wan2.1_14B_VACE-Q6_K.gguf"

# LTX-2.3 (the fast video backbone: SageAttention runtime, native synced audio, ~12s clips).
# All node signatures + file names verified live on the box /object_info (ComfyUI v0.22.0)
# and against reference/LTX-2-3_ULTRA_WORKFLOW-V2.json (AItrepreneur). LTXDirector is the
# orchestrator: takes model+clip+prompt, emits (model, positive cond, video_latent,
# audio_latent, guide_data, frame_rate, combined_audio) so the prompt-encode + latent-init
# plumbing collapses into one node. Sampling is joint audio+video: concat AV latent ->
# SamplerCustomAdvanced (euler + the distilled 8-step sigma schedule, CFG 1) -> separate ->
# spatio-temporal tiled video decode + audio VAE decode.
# UNet quant is configurable per-request: pass {"quant": "Q5_K_S"} etc. Tiers (AItrepreneur):
# Q4_K_S (<12GB), Q5_K_S (12-16GB; comfortable on 32GB system RAM), Q6_K, Q8_0 (best, 24GB+).
LTX_UNET_TMPL = "ltx-2.3-22b-dev-{quant}.gguf"       # UnetLoaderGGUF
LTX_QUANT_DEFAULT = "Q8_0"
LTX_QUANTS = ("Q4_K_S", "Q5_K_S", "Q6_K", "Q8_0")
LTX_UNET_GGUF = LTX_UNET_TMPL.format(quant=LTX_QUANT_DEFAULT)  # default GGUF file (legacy)
# fp8 migration: clean-unloading scaled fp8 safetensors (no GGUF unpatch segfault, no RAM
# residue leak that thrashed the next model on a switch). Transformer-only: keeps the gemma
# DualCLIP + the two VAELoaderKJ VAEs + distill/detailer LoRAs below. Loads via UNETLoader
# weight_dtype "default" (storage-only fp8, dequant to bf16 in compute; matches our working
# fp8 S2V pattern). NOT _fast: the 3090/Ampere has no fp8 matmul, _fast would mis-engage it.
LTX_UNET_FP8 = "ltx-2.3-22b-dev_transformer_only_fp8_scaled.safetensors"  # UNETLoader
LTX_CLIP1 = "gemma_3_12B_it_fp4_mixed.safetensors"   # DualCLIPLoader clip_name1
LTX_CLIP2 = "ltx-2.3_text_projection_bf16.safetensors"  # clip_name2; type "ltxv"
LTX_VAE_VIDEO = "LTX23_video_vae_bf16.safetensors"   # VAELoaderKJ main_device/bf16
LTX_VAE_AUDIO = "LTX23_audio_vae_bf16.safetensors"   # VAELoaderKJ cpu/bf16
# Default = TenStrip higher-motion-ceiling distill variant (ceil72): less slow-mo than the stock
# distill at the same 8-step speed (user verdict 2026-06-17, "a little less dramatic"). The stock
# distill is ltx-2.3-22b-distilled-lora-384-1.1.safetensors; override per-call via distill_lora.
LTX_LORA_DISTILL = "ltx-2.3-22b-distilled-lora-fro90_ceil72.safetensors"  # few-step distill (req'd for 8-step)
# Stock distill: smoother/tamer gait. MSR walks default to this (ceil72's higher ceiling MAY have caused a
# step hitch in a 20s tracking walk - possibly ephemeral, user verdict 2026-06-17 "stick with original").
LTX_LORA_DISTILL_STOCK = "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
LTX_LORA_DETAILER = "ltx-2-19b-ic-lora-detailer.safetensors"         # texture/detail
LTX_LORA_VBVR = "VBVR-official-comfyui.safetensors"                  # LiconStudio motion-dynamics LoRA
LTX_LORA_MSR = "LTX-2.3\\LTX-2.3-Licon-MSR-V1.safetensors"          # Multiple-Subject-Reference (IC-loader subfolder)
# 8-step distilled sigma schedule (9 values = 8 steps), verbatim from the ULTRA base pass.
LTX_SIGMAS_BASE = "1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"


def _ltx_frames(n, fps):
    """LTX temporal VAE needs (frames-1) divisible by 8. Round to the nearest valid count."""
    n = max(9, int(n))
    k = round((n - 1) / 8)
    return 8 * k + 1

# A readable English negative that steers Wan/Z-Image away from the usual artifacts.
DEFAULT_NEG = ("low quality, blurry, distorted, deformed, bad anatomy, extra fingers, "
               "watermark, text, jpeg artifacts, overexposed, static, plastic skin, cartoon")


def _seed(p):
    s = p.get("seed")
    if s in (None, "", 0, "0"):
        return random.randint(0, 2**31 - 1)
    return int(s)


# ---------------------------------------------------------------- Z-Image still
def build_still(p):
    """Text-to-image photoreal still (Z-Image Turbo). p: {prompt, negative?, seed?,
    width?, height?, steps?, cfg?}. Output: SaveImage -> videogen/still."""
    seed = _seed(p)
    w = int(p.get("width", 1024))
    h = int(p.get("height", 1024))
    steps = int(p.get("steps", 8))
    cfg = float(p.get("cfg", 1.0))
    lora = p.get("lora")                          # optional Z-Image character LoRA (filename)
    lora_strength = float(p.get("lora_strength", 1.0))
    prompt = (p.get("prompt") or "").strip()
    neg = p.get("negative")
    neg = DEFAULT_NEG if neg is None else neg
    model_src = ["1", 0]
    g = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": Z_IMAGE_UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": Z_IMAGE_CLIP, "type": "lumina2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": Z_IMAGE_VAE}},
    }
    if lora:                                       # trained character LoRA -> consistent identity
        g["16"] = {"class_type": "LoraLoaderModelOnly",
                   "inputs": {"model": ["1", 0], "lora_name": lora, "strength_model": lora_strength}}
        model_src = ["16", 0]
    g["4"] = {"class_type": "ModelSamplingAuraFlow",
              "inputs": {"model": model_src, "shift": 3.0}}
    g.update({
        "5": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["2", 0], "text": prompt}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["2", 0], "text": neg}},
        "7": {"class_type": "EmptySD3LatentImage",
              "inputs": {"width": w, "height": h, "batch_size": 1}},
        "8": {"class_type": "KSampler",
              "inputs": {"model": ["4", 0], "seed": seed, "steps": steps, "cfg": cfg,
                         "sampler_name": "res_multistep", "scheduler": "simple",
                         "positive": ["5", 0], "negative": ["6", 0],
                         "latent_image": ["7", 0], "denoise": 1.0}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveImage",
               "inputs": {"images": ["9", 0], "filename_prefix": "videogen/still"}},
    })
    return g, {"seed": seed, "width": w, "height": h, "steps": steps, "cfg": cfg,
               "prompt": prompt, "lora": lora, "kind": "image"}


# ---------------------------------------- Qwen-Image-Edit-2511: consistent character still
def build_qwen_char_still(p, ref_images):
    """Generate a still of a referenced character in a NEW scene/pose, keeping identity
    (Qwen-Image-Edit-2511 GGUF, reference-driven, no training). ref_images = 1-3 uploaded
    image names on ComfyUI (image1 = the primary reference). p: {prompt (the new scene),
    negative?, seed?, steps?, cfg?}. Wiring mirrors the verified ComfyUI 2511 edit template
    (full-quality, non-turbo path). Output: SaveImage -> videogen/charstill."""
    seed = _seed(p)
    steps = int(p.get("steps", 40))
    cfg = float(p.get("cfg", 4.0))
    prompt = (p.get("prompt") or "").strip()
    neg = p.get("negative")
    neg = "" if neg is None else neg          # template uses an empty negative for edit
    refs = [r for r in (ref_images or []) if r][:3]
    if not refs:
        raise ValueError("at least one reference image is required")
    g = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": QWEN_EDIT_FP8, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": QWEN_CLIP, "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN_VAE}},
        "4": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3.1}},
        "5": {"class_type": "CFGNorm", "inputs": {"model": ["4", 0], "strength": 1.0}},
        "6": {"class_type": "LoadImage", "inputs": {"image": refs[0]}},
        "7": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["6", 0]}},
    }
    # positive/negative TextEncodeQwenImageEditPlus: image1 = scaled primary ref; image2/3 raw
    pos_in = {"clip": ["2", 0], "vae": ["3", 0], "image1": ["7", 0], "prompt": prompt}
    neg_in = {"clip": ["2", 0], "vae": ["3", 0], "image1": ["7", 0], "prompt": neg}
    nid = 20
    for i, r in enumerate(refs[1:], start=2):          # extra references -> image2, image3
        g[str(nid)] = {"class_type": "LoadImage", "inputs": {"image": r}}
        pos_in[f"image{i}"] = [str(nid), 0]
        neg_in[f"image{i}"] = [str(nid), 0]
        nid += 1
    g["8"] = {"class_type": "TextEncodeQwenImageEditPlus", "inputs": pos_in}
    g["9"] = {"class_type": "TextEncodeQwenImageEditPlus", "inputs": neg_in}
    g["10"] = {"class_type": "FluxKontextMultiReferenceLatentMethod",
               "inputs": {"conditioning": ["8", 0], "reference_latents_method": "index_timestep_zero"}}
    g["11"] = {"class_type": "FluxKontextMultiReferenceLatentMethod",
               "inputs": {"conditioning": ["9", 0], "reference_latents_method": "index_timestep_zero"}}
    g["12"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["7", 0], "vae": ["3", 0]}}
    # Output canvas: by default the latent (and so the output shape) follows the scaled
    # reference via VAEEncode - that is why square refs yield square stills. If width/height
    # are supplied, generate on a fresh EmptySD3LatentImage canvas at that exact shape instead
    # (EmptySD3LatentImage is the node the Qwen-Image base txt2img template uses, verified on
    # box). denoise stays 1.0 so identity still comes from the reference conditioning (image1/2/3
    # via TextEncodeQwenImageEditPlus); only the output aspect changes.
    cw = int(p.get("width") or 0)
    ch = int(p.get("height") or 0)
    latent_src = ["12", 0]
    if cw > 0 and ch > 0:
        g["16"] = {"class_type": "EmptySD3LatentImage",
                   "inputs": {"width": cw, "height": ch, "batch_size": 1}}
        latent_src = ["16", 0]
    g["13"] = {"class_type": "KSampler",
               "inputs": {"model": ["5", 0], "seed": seed, "steps": steps, "cfg": cfg,
                          "sampler_name": "euler", "scheduler": "simple",
                          "positive": ["10", 0], "negative": ["11", 0],
                          "latent_image": latent_src, "denoise": 1.0}}
    g["14"] = {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["3", 0]}}
    g["15"] = {"class_type": "SaveImage",
               "inputs": {"images": ["14", 0], "filename_prefix": "videogen/charstill"}}
    meta = {"seed": seed, "steps": steps, "cfg": cfg, "prompt": prompt,
            "refs": len(refs), "kind": "image"}
    if cw > 0 and ch > 0:
        meta["width"], meta["height"] = cw, ch
    return g, meta


# --------------------------------- Wan VACE: reference-to-video (identity through motion)
def build_vace_ref2v(p, image_ref):
    """Generate a video of a referenced subject in motion, holding identity from a single
    reference image (Wan2.1 14B VACE GGUF). The video-side consistency alternative: skips
    the still->i2v chain and animates the character directly. control_video/masks are
    optional (omitted = free generation guided by reference + prompt). p: {prompt, negative?,
    seed?, width?, height?, length?, fps?, steps?, cfg?, strength?}. Output: videogen/vace."""
    seed = _seed(p)
    w = int(p.get("width", 768))
    h = int(p.get("height", 768))
    length = int(p.get("length", 81))
    fps = int(p.get("fps", 16))
    steps = int(p.get("steps", 20))
    cfg = float(p.get("cfg", 6.0))
    strength = float(p.get("strength", 1.0))
    prompt = (p.get("prompt") or "").strip()
    neg = p.get("negative")
    neg = DEFAULT_NEG if neg is None else neg
    g = {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": WAN_VACE_GGUF}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": WAN_CLIP, "type": "wan", "device": "cpu"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": WAN21_VAE}},
        "4": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": 8.0}},
        "5": {"class_type": "LoadImage", "inputs": {"image": image_ref}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": neg}},
        "8": {"class_type": "WanVaceToVideo",
              "inputs": {"positive": ["6", 0], "negative": ["7", 0], "vae": ["3", 0],
                         "width": w, "height": h, "length": length, "batch_size": 1,
                         "strength": strength, "reference_image": ["5", 0]}},
        "9": {"class_type": "KSampler",
              "inputs": {"model": ["4", 0], "seed": seed, "steps": steps, "cfg": cfg,
                         "sampler_name": "uni_pc", "scheduler": "simple",
                         "positive": ["8", 0], "negative": ["8", 1],
                         "latent_image": ["8", 2], "denoise": 1.0}},
        "10": {"class_type": "TrimVideoLatent",
               "inputs": {"samples": ["9", 0], "trim_amount": ["8", 3]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
        "12": {"class_type": "CreateVideo", "inputs": {"images": ["11", 0], "fps": fps}},
        "13": {"class_type": "SaveVideo",
               "inputs": {"video": ["12", 0], "filename_prefix": "videogen/vace",
                          "format": "auto", "codec": "auto"}},
    }
    return g, {"seed": seed, "width": w, "height": h, "length": length, "fps": fps,
               "steps": steps, "cfg": cfg, "strength": strength, "prompt": prompt, "kind": "video"}


# ----------------------------------------------------- Wan2.2 5B TI2V (i2v)
def build_i2v(p, image_ref):
    """Image-to-video from a start still (Wan2.2 5B TI2V). image_ref = uploaded image
    name on ComfyUI. p: {prompt, negative?, seed?, width?, height?, length?, fps?,
    steps?, cfg?}. Output: SaveVideo -> videogen/i2v."""
    seed = _seed(p)
    w = int(p.get("width", 832))
    h = int(p.get("height", 480))
    length = int(p.get("length", 81))      # frames; 81 @ ~16fps ~ 5s
    fps = int(p.get("fps", 24))
    steps = int(p.get("steps", 20))
    cfg = float(p.get("cfg", 5.0))
    prompt = (p.get("prompt") or "").strip()
    neg = p.get("negative")
    neg = DEFAULT_NEG if neg is None else neg
    g = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": WAN_TI2V, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": WAN_CLIP, "type": "wan", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": WAN22_VAE}},
        "4": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": 8.0}},
        "5": {"class_type": "LoadImage", "inputs": {"image": image_ref}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": neg}},
        "8": {"class_type": "Wan22ImageToVideoLatent",
              "inputs": {"vae": ["3", 0], "width": w, "height": h, "length": length,
                         "batch_size": 1, "start_image": ["5", 0]}},
        "9": {"class_type": "KSampler",
              "inputs": {"model": ["4", 0], "seed": seed, "steps": steps, "cfg": cfg,
                         "sampler_name": "uni_pc", "scheduler": "simple",
                         "positive": ["6", 0], "negative": ["7", 0],
                         "latent_image": ["8", 0], "denoise": 1.0}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["3", 0]}},
        "11": {"class_type": "CreateVideo", "inputs": {"images": ["10", 0], "fps": fps}},
        "12": {"class_type": "SaveVideo",
               "inputs": {"video": ["11", 0], "filename_prefix": "videogen/i2v",
                          "format": "auto", "codec": "auto"}},
    }
    return g, {"seed": seed, "width": w, "height": h, "length": length, "fps": fps,
               "steps": steps, "cfg": cfg, "prompt": prompt, "kind": "video"}


# ------------------------------------------------- Wan2.2-S2V single-clip lip-sync
def build_s2v(p, image_ref, audio_ref):
    """Audio-driven lip-sync clip from a portrait still + audio (Wan2.2-S2V, single
    ~4.8s clip, no Extend chain). image_ref/audio_ref = uploaded names on ComfyUI.
    p: {prompt?, negative?, seed?, width?, height?, length?, fps?, steps?, cfg?}.
    Output: SaveVideo -> videogen/s2v (carries the song audio via CreateVideo.audio)."""
    seed = _seed(p)
    w = int(p.get("width", 640))
    h = int(p.get("height", 640))
    length = int(p.get("length", 77))      # one S2V chunk = 77 frames ~ 4.8s @16fps
    fps = int(p.get("fps", 16))
    fast = bool(p.get("fast", False))      # opt-in lightx2v 4-step preview; default = full quality
    steps = int(p.get("steps", 4 if fast else 20))
    cfg = float(p.get("cfg", 1.0 if fast else 6.0))
    prompt = (p.get("prompt") or "a person singing into a microphone").strip()
    neg = p.get("negative")
    neg = DEFAULT_NEG if neg is None else neg
    g = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": WAN_S2V, "weight_dtype": "default"}},
        # umt5 on CPU: keeps the 6.7GB text encoder out of VRAM so the 14B S2V fits the
        # 24GB 3090 without spilling to shared RAM. Text encode is a quick one-off.
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": WAN_CLIP, "type": "wan", "device": "cpu"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": WAN21_VAE}},
        "4": {"class_type": "AudioEncoderLoader", "inputs": {"audio_encoder_name": WAV2VEC}},
        "5": {"class_type": "LoadAudio", "inputs": {"audio": audio_ref}},
        "6": {"class_type": "AudioEncoderEncode",
              "inputs": {"audio_encoder": ["4", 0], "audio": ["5", 0]}},
        "7": {"class_type": "LoadImage", "inputs": {"image": image_ref}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": neg}},
        "10": {"class_type": "ModelSamplingSD3",
               "inputs": {"model": (["16", 0] if fast else ["1", 0]), "shift": 8.0}},
        "11": {"class_type": "WanSoundImageToVideo",
               "inputs": {"positive": ["8", 0], "negative": ["9", 0], "vae": ["3", 0],
                          "width": w, "height": h, "length": length, "batch_size": 1,
                          "audio_encoder_output": ["6", 0], "ref_image": ["7", 0]}},
        "12": {"class_type": "KSampler",
               "inputs": {"model": ["10", 0], "seed": seed, "steps": steps, "cfg": cfg,
                          "sampler_name": "uni_pc", "scheduler": "simple",
                          "positive": ["11", 0], "negative": ["11", 1],
                          "latent_image": ["11", 2], "denoise": 1.0}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["3", 0]}},
        "14": {"class_type": "CreateVideo",
               "inputs": {"images": ["13", 0], "fps": fps, "audio": ["5", 0]}},
        "15": {"class_type": "SaveVideo",
               "inputs": {"video": ["14", 0], "filename_prefix": "videogen/s2v",
                          "format": "auto", "codec": "auto"}},
    }
    if fast:
        g["16"] = {"class_type": "LoraLoaderModelOnly",
                   "inputs": {"model": ["1", 0], "lora_name": WAN_LIGHTX_HIGH,
                              "strength_model": 1.0}}
    return g, {"seed": seed, "width": w, "height": h, "length": length, "fps": fps,
               "steps": steps, "cfg": cfg, "fast": fast, "prompt": prompt, "kind": "video"}


# -------------------------------------- Wan2.2-S2V via WanVideoWrapper (block-swap, fits 24GB)
def build_s2v_wrapper(p, image_ref, audio_ref):
    """Audio-driven lip-sync (Wan2.2-S2V) on the Kijai WanVideoWrapper with block-swap, which
    fits the 14B on a 24GB 3090 where the native nodes thrashed. image_ref/audio_ref = uploaded
    names on ComfyUI. p: {prompt?, seed?, width?, height?, frames?, seconds?, context_frames?,
    context_overlap?, fps?, steps?, cfg?, shift?, blocks_to_swap?, lora_strength?, audio_scale?}.
    frames/seconds = TOTAL length; past one window (context_frames, def 81) it adds a
    WanVideoContextOptions node for seamless overlapping-window long-form (Kijai S2V reference).
    Output: SaveVideo -> videogen/s2v.
    Graph traced from the wrapper's own example workflow - settings are its proven defaults."""
    seed = _seed(p)
    w = int(p.get("width", 832))
    h = int(p.get("height", 480))
    fps = int(p.get("fps", 16))                        # S2V native frame rate
    # frames = TOTAL output frames (pass `seconds` for a friendly duration). For long-form we
    # follow Kijai's own S2V context-window reference workflow: keep frame_window_size = total,
    # and add a WanVideoContextOptions node (overlapping windows) into the sampler. The overlap
    # is what makes the window transitions seamless. Default frames=77 (< ctx_frames) = a single
    # clip, no context options = unchanged behavior.
    if p.get("seconds"):
        frames = int(round(float(p["seconds"]) * fps))
    else:
        frames = int(p.get("frames", 77))
    frames = max(5, ((frames - 1) // 4) * 4 + 1)       # Wan latents need 4n+1
    ctx_frames = int(p.get("context_frames", 81))      # per-window size (reference: 81)
    ctx_overlap = int(p.get("context_overlap", 16))    # window overlap for seamless joins (ref: 16)
    long_form = frames > ctx_frames                    # context windowing only kicks in past one window
    steps = int(p.get("steps", 4))                     # lightx2v distill -> 4 steps
    cfg = float(p.get("cfg", 1.0))
    shift = float(p.get("shift", 4.0))
    blocks = int(p.get("blocks_to_swap", 25))          # 25/40 blocks -> fits 24GB (raise if OOM)
    load_device = p.get("load_device") or "offload_device"  # "main_device" = model on GPU (more VRAM, faster)
    lstr = float(p.get("lora_strength", 1.5))
    audio_scale = float(p.get("audio_scale", 1.0))
    # pose-guided combine: a motion video (e.g. an LTX/SVI walk clip) whose body POSE drives S2V
    # while the audio drives the lips -> motion + lip-sync in one shot. Uses framepack mode (the
    # pose path), which is mutually exclusive with the context-window long-form path.
    pose_video = p.get("pose_video")
    prompt = (p.get("prompt") or "a person singing into a microphone, close up").strip()
    neg = p.get("negative") or "blurry, distorted, static, low quality"
    g = {
        # attention=sdpa (NOT sageattn): on a 3090 (Ampere, no native fp8) sage + the fp8 model
        # produces BLACK output (SageAttention issue #221, WanVideoWrapper #1605); sage also
        # silently falls back to pytorch on this card anyway, so we lose nothing real.
        "1": {"class_type": "WanVideoModelLoader",
              "inputs": {"model": WAN_S2V_KJ, "base_precision": "fp16_fast",
                         "quantization": "fp8_e4m3fn_scaled", "load_device": load_device,
                         "attention_mode": "sdpa"}},
        "2": {"class_type": "WanVideoLoraSelectMulti",
              "inputs": {"lora_0": LIGHTX2V_V2, "strength_0": lstr, "lora_1": "none",
                         "strength_1": 1.0, "lora_2": "none", "strength_2": 1.0,
                         "lora_3": "none", "strength_3": 1.0, "lora_4": "none", "strength_4": 1.0,
                         "merge_loras": False}},
        "3": {"class_type": "WanVideoSetLoRAs", "inputs": {"model": ["1", 0], "lora": ["2", 0]}},
        "4": {"class_type": "WanVideoBlockSwap",
              "inputs": {"blocks_to_swap": blocks, "offload_img_emb": False,
                         "offload_txt_emb": False, "use_non_blocking": True}},
        "5": {"class_type": "WanVideoSetBlockSwap",
              "inputs": {"model": ["3", 0], "block_swap_args": ["4", 0]}},
        "6": {"class_type": "WanVideoVAELoader",
              "inputs": {"model_name": WAN21_VAE_BF16, "precision": "bf16"}},
        "7": {"class_type": "WanVideoTextEncodeCached",
              "inputs": {"model_name": UMT5_ENC_BF16, "precision": "bf16",
                         "positive_prompt": prompt, "negative_prompt": neg,
                         "quantization": "disabled", "use_disk_cache": False, "device": "gpu"}},
        "8": {"class_type": "LoadImage", "inputs": {"image": image_ref}},
        "9": {"class_type": "ImageResizeKJv2",
              "inputs": {"image": ["8", 0], "width": w, "height": h, "upscale_method": "lanczos",
                         "keep_proportion": "crop", "pad_color": "0, 0, 0",
                         "crop_position": "center", "divisible_by": 2, "device": "cpu"}},
        "10": {"class_type": "WanVideoEncode",
               "inputs": {"vae": ["6", 0], "image": ["9", 0], "enable_vae_tiling": False,
                          "tile_x": 272, "tile_y": 272, "tile_stride_x": 144, "tile_stride_y": 128}},
        "11": {"class_type": "AudioEncoderLoader", "inputs": {"audio_encoder_name": WAV2VEC}},
        "12": {"class_type": "LoadAudio", "inputs": {"audio": audio_ref}},
        "13": {"class_type": "AudioEncoderEncode",
               "inputs": {"audio_encoder": ["11", 0], "audio": ["12", 0]}},
        "14": {"class_type": "WanVideoEmptyEmbeds",
               "inputs": {"width": ["9", 1], "height": ["9", 2], "num_frames": frames}},
        "15": {"class_type": "WanVideoAddS2VEmbeds",
               "inputs": {"embeds": ["14", 0], "frame_window_size": frames,
                          "audio_scale": audio_scale, "pose_start_percent": 0.0,
                          "pose_end_percent": 1.0, "audio_encoder_output": ["13", 0],
                          "ref_latent": ["10", 0]}},
        "16": {"class_type": "WanVideoSampler",
               "inputs": {"model": ["5", 0], "image_embeds": ["15", 0], "steps": steps,
                          "cfg": cfg, "shift": shift, "seed": seed, "force_offload": True,
                          "scheduler": "dpm++_sde", "riflex_freq_index": 0,
                          "text_embeds": ["7", 0]}},
        "17": {"class_type": "WanVideoDecode",
               "inputs": {"vae": ["6", 0], "samples": ["16", 0], "enable_vae_tiling": False,
                          "tile_x": 272, "tile_y": 272, "tile_stride_x": 144, "tile_stride_y": 128}},
        "18": {"class_type": "CreateVideo",
               "inputs": {"images": ["17", 0], "fps": fps, "audio": ["12", 0]}},
        "19": {"class_type": "SaveVideo",
               "inputs": {"video": ["18", 0], "filename_prefix": "videogen/s2v",
                          "format": "auto", "codec": "auto"}},
    }
    if long_form and not pose_video:
        # Kijai S2V context-window long-form: overlapping windows for seamless joins. Only added
        # past one window so the default single-clip graph is untouched.
        g["20"] = {"class_type": "WanVideoContextOptions",
                   "inputs": {"context_schedule": "uniform_standard", "context_frames": ctx_frames,
                              "context_stride": 4, "context_overlap": ctx_overlap,
                              "freenoise": True, "verbose": False, "fuse_method": "linear"}}
        g["16"]["inputs"]["context_options"] = ["20", 0]
    if pose_video:
        # pose chain: motion video -> DWPose skeleton -> resize -> VAE encode -> pose_latent.
        # The detector auto-downloads its TorchScript models on first use (hr16 HF repos).
        g["30"] = {"class_type": "VHS_LoadVideo",
                   "inputs": {"video": pose_video, "force_rate": float(fps), "custom_width": 0,
                              "custom_height": 0, "frame_load_cap": frames, "skip_first_frames": 0,
                              "select_every_nth": 1}}
        g["31"] = {"class_type": "WanVideoUniAnimateDWPoseDetector",
                   "inputs": {"pose_images": ["30", 0], "score_threshold": 0.3, "stick_width": 4,
                              "draw_body": True, "body_keypoint_size": 4, "draw_feet": True,
                              "draw_hands": True, "hand_keypoint_size": 4, "colorspace": "RGB",
                              "handle_not_detected": "empty", "draw_head": True}}
        g["32"] = {"class_type": "ImageResizeKJv2",
                   "inputs": {"image": ["31", 0], "width": w, "height": h, "upscale_method": "lanczos",
                              "keep_proportion": "crop", "pad_color": "0, 0, 0",
                              "crop_position": "center", "divisible_by": 2, "device": "cpu"}}
        g["33"] = {"class_type": "WanVideoEncode",
                   "inputs": {"vae": ["6", 0], "image": ["32", 0], "enable_vae_tiling": False,
                              "tile_x": 272, "tile_y": 272, "tile_stride_x": 144, "tile_stride_y": 128}}
        g["15"]["inputs"]["pose_latent"] = ["33", 0]
        g["15"]["inputs"]["enable_framepack"] = True
        # framepack mode requires the VAE in the embeds dict (sampler builds ref_motion_image
        # from it); without it the sampler hits `vae.dtype` on None and crashes.
        g["15"]["inputs"]["vae"] = ["6", 0]
        # the frame_packer ALWAYS uses comfy-rope (rope_encode_comfy) internally; with the
        # sampler's "default" rope, rope_embedder.k stays None and rope_riflex crashes on
        # `k > 0`. Forcing comfy rope makes the sampler set k = riflex_freq_index (0).
        g["16"]["inputs"]["rope_function"] = "comfy"
    return g, {"seed": seed, "width": w, "height": h, "frames": frames, "fps": fps,
               "seconds": round(frames / fps, 2), "long_form": long_form and not pose_video,
               "pose_guided": bool(pose_video),
               "context_frames": ctx_frames if (long_form and not pose_video) else None,
               "context_overlap": ctx_overlap if long_form else None,
               "steps": steps, "cfg": cfg, "shift": shift, "blocks_to_swap": blocks,
               "lora_strength": lstr, "prompt": prompt, "kind": "video"}


# ============================================================ InfiniteTalk (video-to-video lip-sync)
def build_infinitetalk_v2v(p, video_ref, audio_ref):
    """Video-to-video lip-sync: take EXISTING footage (e.g. an SVI/LTX walk clip) and redrive only
    the mouth/face from audio, PRESERVING the source's motion, camera and background. This is the
    "walking AND singing" lane that pose-guided S2V can't do (S2V locks to a reference still and
    won't travel). Mechanism: the source video is VAE-encoded and fed as the sampler's init latent
    (`samples`), plus start_image + clip-vision from the same frames, so denoising starts FROM the
    real footage while MultiTalk audio embeds redrive the lips. InfiniteTalk = the Wan2.1 MultiTalk
    variant the Kijai wrapper auto-detects by filename.
    video_ref/audio_ref = uploaded names on ComfyUI. p: {prompt?, negative?, seed?, width?, height?,
    fps?, frames?/seconds?, steps?, cfg?, shift?, blocks_to_swap?, lora_strength?, audio_scale?,
    frame_window_size?, motion_frame?, colormatch?}. Output: SaveVideo -> videogen/infinitetalk.
    Graph traced verbatim from the wrapper's wanvideo_2_1_14B_V2V_InfiniteTalk_example_02.json."""
    seed = _seed(p)
    w = int(p.get("width", 832))
    h = int(p.get("height", 480))
    fps = int(p.get("fps", 25))                            # Wan2.1 / MultiTalk native frame rate
    if p.get("seconds"):
        frames = int(round(float(p["seconds"]) * fps))
    else:
        frames = int(p.get("frames", 81))
    frames = max(5, ((frames - 1) // 4) * 4 + 1)           # Wan latents need 4n+1
    fws = int(p.get("frame_window_size", 81))              # per-window size (reference: 81)
    motion_frame = int(p.get("motion_frame", 9))           # window overlap (reference: 9)
    steps = int(p.get("steps", 4))                         # lightx2v i2v distill -> 4 steps
    cfg = float(p.get("cfg", 1.0))
    shift = float(p.get("shift", 11.0))                    # reference InfiniteTalk shift
    blocks = int(p.get("blocks_to_swap", 20))              # 20 fits the 14B + InfiniteTalk on 24GB
    lstr = float(p.get("lora_strength", 1.0))
    audio_scale = float(p.get("audio_scale", 1.0))
    colormatch = p.get("colormatch", "disabled")           # per-window color matching for long clips
    prompt = (p.get("prompt") or "a person singing, cinematic photoreal").strip()
    neg = p.get("negative") or "blurry, distorted, static, low quality, deformed mouth"
    g = {
        # attention=sdpa (NOT sageattn): Ampere fp8 + sage = black output on this card.
        # the Wan2.1 i2v file is PLAIN fp8 (..._fp8_e4m3fn, not _scaled) -> quantization must be
        # "fp8_e4m3fn" (the loader rejects "_scaled" on a non-scaled file).
        "1": {"class_type": "WanVideoModelLoader",
              "inputs": {"model": WAN21_I2V_FP8, "base_precision": "fp16_fast",
                         "quantization": "fp8_e4m3fn", "load_device": "offload_device",
                         "attention_mode": "sdpa", "block_swap_args": ["2", 0], "lora": ["3", 0],
                         "multitalk_model": ["4", 0]}},
        "2": {"class_type": "WanVideoBlockSwap",
              "inputs": {"blocks_to_swap": blocks, "offload_img_emb": False,
                         "offload_txt_emb": False, "use_non_blocking": True}},
        "3": {"class_type": "WanVideoLoraSelectMulti",
              "inputs": {"lora_0": LIGHTX2V_I2V_480P, "strength_0": lstr, "lora_1": "none",
                         "strength_1": 1.0, "lora_2": "none", "strength_2": 1.0,
                         "lora_3": "none", "strength_3": 1.0, "lora_4": "none", "strength_4": 1.0,
                         "merge_loras": False}},
        "4": {"class_type": "MultiTalkModelLoader",
              "inputs": {"model": INFINITETALK, "base_precision": "fp16"}},
        "5": {"class_type": "WanVideoVAELoader",
              "inputs": {"model_name": WAN21_VAE_BF16, "precision": "bf16"}},
        "6": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": CLIP_VISION_H}},
        "7": {"class_type": "Wav2VecModelLoader",
              "inputs": {"model": WAV2VEC_CN_BASE, "base_precision": "fp16",
                         "load_device": "main_device"}},
        "8": {"class_type": "LoadAudio", "inputs": {"audio": audio_ref}},
        # source footage -> resampled to fps -> resized to the generation canvas
        "9": {"class_type": "VHS_LoadVideo",
              "inputs": {"video": video_ref, "force_rate": float(fps), "custom_width": 0,
                         "custom_height": 0, "frame_load_cap": frames, "skip_first_frames": 0,
                         "select_every_nth": 1}},
        "10": {"class_type": "ImageResizeKJv2",
               "inputs": {"image": ["9", 0], "width": w, "height": h, "upscale_method": "lanczos",
                          "keep_proportion": "crop", "pad_color": "0, 0, 0",
                          "crop_position": "center", "divisible_by": 16, "device": "cpu"}},
        "11": {"class_type": "WanVideoClipVisionEncode",
               "inputs": {"clip_vision": ["6", 0], "image_1": ["10", 0], "strength_1": 1.0,
                          "strength_2": 1.0, "crop": "center", "combine_embeds": "average",
                          "force_offload": True}},
        # VAE-encode the source frames -> init latent the sampler denoises FROM (motion preserved)
        "12": {"class_type": "WanVideoEncode",
               "inputs": {"vae": ["5", 0], "image": ["10", 0], "enable_vae_tiling": False,
                          "tile_x": 272, "tile_y": 272, "tile_stride_x": 144, "tile_stride_y": 128}},
        "13": {"class_type": "MultiTalkWav2VecEmbeds",
               "inputs": {"wav2vec_model": ["7", 0], "audio_1": ["8", 0], "normalize_loudness": True,
                          "num_frames": frames, "fps": float(fps), "audio_scale": audio_scale,
                          "audio_cfg_scale": 1.0, "multi_audio_type": "para"}},
        "14": {"class_type": "WanVideoImageToVideoMultiTalk",
               "inputs": {"vae": ["5", 0], "width": w, "height": h, "frame_window_size": fws,
                          "motion_frame": motion_frame, "force_offload": False,
                          "colormatch": colormatch, "start_image": ["10", 0],
                          "clip_embeds": ["11", 0], "mode": "infinitetalk"}},
        "15": {"class_type": "WanVideoTextEncodeCached",
               "inputs": {"model_name": UMT5_ENC_BF16, "precision": "bf16",
                          "positive_prompt": prompt, "negative_prompt": neg,
                          "quantization": "disabled", "use_disk_cache": False, "device": "gpu"}},
        # MultiTalk needs comfy rope (same reason as S2V framepack); samples = source init latent
        "16": {"class_type": "WanVideoSampler",
               "inputs": {"model": ["1", 0], "image_embeds": ["14", 0], "text_embeds": ["15", 0],
                          "samples": ["12", 0], "multitalk_embeds": ["13", 0], "steps": steps,
                          "cfg": cfg, "shift": shift, "seed": seed, "force_offload": True,
                          "scheduler": "dpm++_sde", "riflex_freq_index": 0, "rope_function": "comfy"}},
        "17": {"class_type": "WanVideoDecode",
               "inputs": {"vae": ["5", 0], "samples": ["16", 0], "enable_vae_tiling": False,
                          "tile_x": 272, "tile_y": 272, "tile_stride_x": 144, "tile_stride_y": 128}},
        "18": {"class_type": "CreateVideo",
               "inputs": {"images": ["17", 0], "fps": fps, "audio": ["8", 0]}},
        "19": {"class_type": "SaveVideo",
               "inputs": {"video": ["18", 0], "filename_prefix": "videogen/infinitetalk",
                          "format": "auto", "codec": "auto"}},
    }
    return g, {"seed": seed, "width": w, "height": h, "frames": frames, "fps": fps,
               "seconds": round(frames / fps, 2), "frame_window_size": fws,
               "motion_frame": motion_frame, "steps": steps, "cfg": cfg, "shift": shift,
               "blocks_to_swap": blocks, "lora_strength": lstr, "audio_scale": audio_scale,
               "prompt": prompt, "kind": "video"}


# ============================================================ LTX-2.3 (fast backbone)
def _build_ltx(p, image_ref=None, lipsync_audio=None):
    """Shared LTX-2.3 t2v/i2v graph. image_ref None = text-to-video; a uploaded image name
    = image-to-video (the keyframe is imprinted onto LTXDirector's video latent as frame 0).
    Joint audio+video sampling with the distilled 8-step schedule; LTX native audio is decoded
    and carried into the file (the music-video assembly muxes the real song over it later).
    lipsync_audio (an uploaded vocal name) = run LatentSync as a mouth-only POST pass on the
    LTX frames so they sync to that vocal - keeps one consistent look (no second video model).
    p: {prompt, seed?, width?, height?, frames?, fps?, cfg?, distill_strength?,
    detailer_strength?, img_strength?, lips_expression?, inference_steps?}.
    Output: SaveVideo -> videogen/ltx."""
    seed = _seed(p)
    w = int(p.get("width", 768))
    h = int(p.get("height", 512))
    fps = int(p.get("fps", 25 if lipsync_audio else 24))   # LatentSync wants 25fps input
    frames = _ltx_frames(p.get("frames", 97), fps)     # default ~4s; valid 8k+1
    secs = round(frames / fps, 3)
    # motion_fps DECOUPLES the rate the model is CONDITIONED on from the playback rate. LTX bakes
    # "how fast time passes" into generation via frame_rate: telling it a LOWER fps than we play
    # back makes it generate more motion per frame, so at the real fps everything moves FASTER -
    # the direct fix for the slow-motion temporal character (independent of distill/cfg/img_strength,
    # which only affect WHAT she does, not the motion RATE). Default = fps (no change).
    motion_fps = float(p.get("motion_fps") or fps)
    motion_secs = round(frames / motion_fps, 3)
    cfg = float(p.get("cfg", 1.0))                     # distilled LoRA -> CFG 1 (negative ignored)
    distill = float(p.get("distill_strength", 0.5))
    distill_lora = p.get("distill_lora") or LTX_LORA_DISTILL   # swap-in higher-ceiling distill (TenStrip)
    detailer = float(p.get("detailer_strength", 0.2))
    img_strength = float(p.get("img_strength", 0.7))   # keyframe imprint strength (i2v)
    lips_expr = float(p.get("lips_expression", 1.5))   # LatentSync lip-movement intensity 1.0-3.0
    lip_steps = int(p.get("inference_steps", 20))      # LatentSync denoise steps
    quant = (p.get("quant") or LTX_QUANT_DEFAULT).strip()  # accepted for back-compat; fp8 is a
    if quant not in LTX_QUANTS:                            # single file so quant no longer selects
        quant = LTX_QUANT_DEFAULT                          # a model (kept only to not break callers)
    prompt = (p.get("prompt") or "").strip()
    g = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": LTX_UNET_FP8, "weight_dtype": "default"}},
        "2": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["1", 0], "lora_name": distill_lora,
                         "strength_model": distill}},
        "3": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["2", 0], "lora_name": LTX_LORA_DETAILER,
                         "strength_model": detailer}},
        "4": {"class_type": "DualCLIPLoader",
              "inputs": {"clip_name1": LTX_CLIP1, "clip_name2": LTX_CLIP2,
                         "type": "ltxv", "device": "default"}},
        "5": {"class_type": "VAELoaderKJ",
              "inputs": {"vae_name": LTX_VAE_VIDEO, "device": "main_device",
                         "weight_dtype": "bf16"}},
        "6": {"class_type": "VAELoaderKJ",
              "inputs": {"vae_name": LTX_VAE_AUDIO, "device": "cpu", "weight_dtype": "bf16"}},
        # LTXDirector: prompt encode + latent init + conditioning, all in one.
        # local_prompts carries the per-segment prompt (split on "|"); >=1 non-empty required.
        # global_prompt is only a shared style overlay. One full-length segment = prompt in
        # local_prompts, segment_lengths empty (auto-covers all frames). Source-verified in
        # the LTXDirector node's _encode_relay (WhatDreamsCost-ComfyUI/ltx_director.py).
        "7": {"class_type": "LTXDirector",
              "inputs": {"model": ["3", 0], "clip": ["4", 0], "global_prompt": "",
                         "duration_frames": frames, "duration_seconds": motion_secs,
                         "timeline_data": "{\"segments\":[],\"audioSegments\":[]}",
                         "local_prompts": prompt, "segment_lengths": "", "epsilon": 0.001,
                         "guide_strength": "", "audio_vae": ["6", 0],
                         "use_custom_audio": False, "frame_rate": motion_fps,
                         "display_mode": "frames", "custom_width": w, "custom_height": h,
                         "resize_method": "maintain aspect ratio", "divisible_by": 32,
                         "img_compression": 18}},
        "8": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["7", 1]}},
        "9": {"class_type": "LTXVConditioning",
              "inputs": {"positive": ["7", 1], "negative": ["8", 0], "frame_rate": ["7", 5]}},
        "10": {"class_type": "LTXVConcatAVLatent",
               "inputs": {"video_latent": ["7", 2], "audio_latent": ["7", 3]}},
        "11": {"class_type": "CFGGuider",
               "inputs": {"model": ["7", 0], "positive": ["9", 0], "negative": ["9", 1],
                          "cfg": cfg}},
        "12": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "13": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "14": {"class_type": "ManualSigmas", "inputs": {"sigmas": LTX_SIGMAS_BASE}},
        "15": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["12", 0], "guider": ["11", 0], "sampler": ["13", 0],
                          "sigmas": ["14", 0], "latent_image": ["10", 0]}},
        "16": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["15", 0]}},
        "17": {"class_type": "LTXVSpatioTemporalTiledVAEDecode",
               "inputs": {"vae": ["5", 0], "latents": ["16", 0], "spatial_tiles": 4,
                          "spatial_overlap": 4, "temporal_tile_length": 48,
                          "temporal_overlap": 8, "last_frame_fix": False,
                          "working_device": "auto", "working_dtype": "auto"}},
        "18": {"class_type": "LTXVAudioVAEDecode",
               "inputs": {"samples": ["16", 1], "audio_vae": ["6", 0]}},
        "19": {"class_type": "CreateVideo",
               "inputs": {"images": ["17", 0], "fps": fps, "audio": ["18", 0]}},
        "20": {"class_type": "SaveVideo",
               "inputs": {"video": ["19", 0], "filename_prefix": "videogen/ltx",
                          "format": "auto", "codec": "auto"}},
    }
    # Opt-in VBVR motion-dynamics LoRA (LiconStudio, trained for ltx-2.3-22b): adds physically
    # plausible movement with natural acceleration - a model-level lever for the slow-mo character.
    # Inserted after the distill+detailer LoRAs; the director (and everything downstream) runs on
    # the VBVR-patched model. vbvr_strength=0 (default) = off.
    vbvr = float(p.get("vbvr_strength", 0) or 0)
    if vbvr > 0:
        g["50"] = {"class_type": "LoraLoaderModelOnly",
                   "inputs": {"model": ["3", 0], "lora_name": LTX_LORA_VBVR, "strength_model": vbvr}}
        g["7"]["inputs"]["model"] = ["50", 0]
    if image_ref is not None:                          # i2v: imprint the keyframe as frame 0
        g["30"] = {"class_type": "LoadImage", "inputs": {"image": image_ref}}
        g["31"] = {"class_type": "LTXVPreprocess",
                   "inputs": {"image": ["30", 0], "img_compression": 18}}
        g["32"] = {"class_type": "LTXVImgToVideoInplace",
                   "inputs": {"vae": ["5", 0], "image": ["31", 0], "latent": ["7", 2],
                              "strength": img_strength, "bypass": False}}
        g["10"]["inputs"]["video_latent"] = ["32", 0]  # concat uses the imprinted latent
    # Opt-in NON-DISTILLED "natural motion" path: the distilled 8-step LoRA bakes in ~1.7x slow
    # motion (gait analysis: she takes ~half the normal steps/20s). Drop the distill LoRA, add a
    # real negative with "slow motion" (active now that cfg>1), more steps via the proper LTX
    # scheduler, and cfg 3 - the levers the LTX docs + the 2.3 Full reference use for full-cadence
    # motion. Heavier per step but at 24fps/481 frames it fits VRAM (no tiling). Mutually exclusive
    # with tiled. Targeted overrides on the existing graph (reuses director + AV sampling + decode).
    if (p.get("full") or p.get("nondistilled")) and not p.get("tiled"):
        g["2"]["inputs"]["strength_model"] = float(p.get("distill_strength", 0.2))
        g["8"] = {"class_type": "CLIPTextEncode",
                  "inputs": {"clip": ["4", 0],
                             "text": (p.get("negative") or
                                      "slow motion, slow, static, motionless, still, frozen, "
                                      "blurry, low quality, distorted, watermark, subtitles")}}
        g["14"] = {"class_type": "LTXVScheduler",
                   "inputs": {"steps": int(p.get("steps", 20)), "max_shift": 2.05,
                              "base_shift": 0.95, "stretch": True, "terminal": 0.1,
                              "latent": ["10", 0]}}
        g["11"]["inputs"]["cfg"] = float(p.get("cfg", 3.0))
    # Opt-in temporal-tiled long-form path (LTXVLoopingSampler): generates the clip in overlapping
    # ~56-frame chunks so peak VRAM is one tile, not the whole latent -> no shared-memory thrash on
    # long/high-fps clips. VIDEO-ONLY (the looping sampler has no audio path; we don't need LTX's
    # throwaway native audio for a walk - the song is muxed later). i2v via optional_cond_images
    # per the node's own doc; reuses our noise/sampler/sigmas/CFGGuider. Wiring + tile values traced
    # from the wrapper's LTX-2_V2V_Detailer.json (which uses a plain CFGGuider, not STGGuiderAdvanced).
    if p.get("tiled") and image_ref is not None:
        g["45"] = {"class_type": "LTXVLoopingSampler",
                   "inputs": {"model": ["7", 0], "vae": ["5", 0], "noise": ["12", 0],
                              "sampler": ["13", 0], "sigmas": ["14", 0], "guider": ["11", 0],
                              "latents": ["7", 2],
                              "temporal_tile_size": int(p.get("temporal_tile_size", 56)),
                              "temporal_overlap": int(p.get("temporal_overlap", 24)),
                              "guiding_strength": 1.0, "temporal_overlap_cond_strength": 0.5,
                              "cond_image_strength": float(p.get("img_strength", 1.0)),
                              "horizontal_tiles": 1, "vertical_tiles": 1, "spatial_overlap": 1,
                              "adain_factor": float(p.get("adain_factor", 0.0)),
                              "guiding_start_step": 0, "guiding_end_step": 1000,
                              "optional_cond_images": ["30", 0],
                              "optional_cond_image_indices": "0"}}
        g["17"]["inputs"]["latents"] = ["45", 0]   # decode the tiled video latent directly
        g["19"]["inputs"].pop("audio", None)       # video-only (no LTX native audio)
        for n in ("10", "15", "16", "18", "32"):   # drop the unused AV-path nodes
            g.pop(n, None)
    if lipsync_audio is not None:                       # LatentSync mouth-only post pass on LTX frames
        g["40"] = {"class_type": "LoadAudio", "inputs": {"audio": lipsync_audio}}
        g["41"] = {"class_type": "LatentSyncNode",
                   "inputs": {"images": ["17", 0], "audio": ["40", 0], "seed": seed,
                              "lips_expression": lips_expr, "inference_steps": lip_steps}}
        g["19"]["inputs"]["images"] = ["41", 0]         # save the lip-synced frames...
        g["19"]["inputs"]["audio"] = ["40", 0]          # ...carrying the vocal (not LTX native audio)
    resolved = {"seed": seed, "width": w, "height": h, "frames": frames, "fps": fps,
                "seconds": secs, "cfg": cfg, "quant": "fp8", "distill_strength": distill,
                "detailer_strength": detailer, "prompt": prompt, "kind": "video"}
    if image_ref is not None:
        resolved["img_strength"] = img_strength
    if lipsync_audio is not None:
        resolved["lipsync"] = True
        resolved["lips_expression"] = lips_expr
        resolved["inference_steps"] = lip_steps
    return g, resolved


def build_ltx_t2v(p):
    """LTX-2.3 text-to-video. p: {prompt, seed?, width?, height?, frames?, fps?, cfg?}.
    Output: SaveVideo -> videogen/ltx."""
    return _build_ltx(p, image_ref=None)


def build_ltx_i2v(p, image_ref):
    """LTX-2.3 image-to-video from a keyframe still (character keyframe -> motion).
    image_ref = uploaded image name on ComfyUI. p as build_ltx_t2v plus {img_strength?}.
    Output: SaveVideo -> videogen/ltx."""
    return _build_ltx(p, image_ref=image_ref)


def build_ltx_lipsync(p, image_ref, audio_ref):
    """LTX-2.3 i2v + LatentSync lip-sync in one graph: animate the keyframe, then re-sync the
    mouth to the supplied vocal (all on LTX footage -> one consistent look, no second video
    model). image_ref/audio_ref = uploaded names on ComfyUI. p as build_ltx_i2v plus
    {lips_expression?, inference_steps?}. Output: SaveVideo -> videogen/ltx."""
    return _build_ltx(p, image_ref=image_ref, lipsync_audio=audio_ref)


def build_ltx_msr(p, subject_refs, background_ref, vocal_ref=None):
    """LTX-2.3 Multiple-Subject-Reference (Licon MSR): hold a character's identity from REFERENCE
    images instead of a keyframe imprint - so there's no keyframe anchor and the motion is fully
    prompt-driven (the fix for the identity-vs-motion tradeoff). subject_refs = 1-4 uploaded
    character-reference image names; background_ref = a scene image name (REQUIRED). Identity comes
    from the references via the MSR IC-LoRA; the walk comes from the prompt. Graph traced from
    ComfyUI-Licon-MSR's LTX-2.3_MSR_sample_workflow_V2.json, grafted onto our fp8-transformer +
    DualCLIP loaders (the sample's full checkpoint won't fit 32GB RAM).
    vocal_ref (optional uploaded vocal name) = NATIVE single-pass lip-sync: instead of the EMPTY
    audio latent, encode the real vocal (LTXVAudioVAEEncode) into the AV latent so LTX drives the
    lips to the song IN THE SAME PASS as the MSR walk (no keyframe anchor, no second video model,
    no LatentSync). p: {prompt (reference description + action), negative?, seed?, width?, height?,
    frames?, fps?, steps?, cfg?, msr_strength?, guide_strength?, ref_frames? (17/25/33/41),
    distill_lora?, distill_strength?}."""
    seed = _seed(p)
    w = int(p.get("width", 832))
    h = int(p.get("height", 480))
    fps = int(p.get("fps", 24))
    frames = _ltx_frames(p.get("frames", 145), fps)
    steps = int(p.get("steps", 8))
    cfg = float(p.get("cfg", 1.0))
    distill = float(p.get("distill_strength", 0.5))
    detailer = float(p.get("detailer_strength", 0.2))
    msr_str = float(p.get("msr_strength", 1.0))
    guide_str = float(p.get("guide_strength", 1.0))
    ref_frames = int(p.get("ref_frames", 17))            # LiconMSR combo: 17/25/33/41
    if ref_frames not in (17, 25, 33, 41):
        ref_frames = 17
    prompt = (p.get("prompt") or "").strip()
    neg = (p.get("negative") or "subtitles, watermark, worst quality, blurry, jittery, distorted, "
           "inconsistent appearance, slow motion")
    subs = [r for r in (subject_refs or []) if r][:4]
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": LTX_UNET_FP8, "weight_dtype": "default"}},
        "2": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["1", 0], "lora_name": (p.get("distill_lora") or LTX_LORA_DISTILL_STOCK),
                         "strength_model": distill}},
        "3": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["2", 0], "lora_name": LTX_LORA_DETAILER, "strength_model": detailer}},
        "4": {"class_type": "LTXICLoRALoaderModelOnly",
              "inputs": {"model": ["3", 0], "lora_name": LTX_LORA_MSR, "strength_model": msr_str}},
        "5": {"class_type": "DualCLIPLoader",
              "inputs": {"clip_name1": LTX_CLIP1, "clip_name2": LTX_CLIP2, "type": "ltxv", "device": "default"}},
        "6": {"class_type": "VAELoaderKJ",
              "inputs": {"vae_name": LTX_VAE_VIDEO, "device": "main_device", "weight_dtype": "bf16"}},
        "7": {"class_type": "VAELoaderKJ",
              "inputs": {"vae_name": LTX_VAE_AUDIO, "device": "cpu", "weight_dtype": "bf16"}},
        "8": {"class_type": "EmptyLTXVLatentVideo",
              "inputs": {"width": w, "height": h, "length": frames, "batch_size": 1}},
        "9": {"class_type": "PromptRelayEncode",
              "inputs": {"model": ["4", 0], "clip": ["5", 0], "latent": ["8", 0], "global_prompt": "",
                         "local_prompts": prompt, "segment_lengths": "", "epsilon": 0.001}},
        "10": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["5", 0], "text": neg}},
        "11": {"class_type": "LTXVConditioning",
               "inputs": {"positive": ["9", 1], "negative": ["10", 0], "frame_rate": float(fps)}},
        "12": {"class_type": "LTX2_NAG",
               "inputs": {"model": ["9", 0], "nag_scale": 11.0, "nag_alpha": 0.25, "nag_tau": 2.5}},
        "13": {"class_type": "LiconMSR",
               "inputs": {"width": w, "height": h, "frame_count": ref_frames, "background": ["35", 0]}},
        "14": {"class_type": "LTXAddVideoICLoRAGuide",
               "inputs": {"positive": ["11", 0], "negative": ["11", 1], "vae": ["6", 0],
                          "latent": ["8", 0], "image": ["13", 0], "frame_idx": 0,
                          "strength": guide_str, "latent_downscale_factor": 1.0, "crop": "center",
                          "use_tiled_encode": False, "tile_size": 256, "tile_overlap": 64}},
        "15": {"class_type": "LTXVEmptyLatentAudio",
               "inputs": {"frames_number": frames, "frame_rate": fps, "batch_size": 1, "audio_vae": ["7", 0]}},
        "16": {"class_type": "LTXVConcatAVLatent",
               "inputs": {"video_latent": ["14", 2], "audio_latent": ["15", 0]}},
        "17": {"class_type": "CFGGuider",
               "inputs": {"model": ["12", 0], "positive": ["14", 0], "negative": ["14", 1], "cfg": cfg}},
        "18": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "19": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "20": {"class_type": "ManualSigmas", "inputs": {"sigmas": LTX_SIGMAS_BASE}},
        "21": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["18", 0], "guider": ["17", 0], "sampler": ["19", 0],
                          "sigmas": ["20", 0], "latent_image": ["16", 0]}},
        "22": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["21", 0]}},
        "23": {"class_type": "LTXVCropGuides",
               "inputs": {"positive": ["14", 0], "negative": ["14", 1], "latent": ["22", 0]}},
        "24": {"class_type": "LTXVSpatioTemporalTiledVAEDecode",
               "inputs": {"vae": ["6", 0], "latents": ["23", 2], "spatial_tiles": 4,
                          "spatial_overlap": 4, "temporal_tile_length": 48, "temporal_overlap": 8,
                          "last_frame_fix": False, "working_device": "auto", "working_dtype": "auto"}},
        "25": {"class_type": "CreateVideo", "inputs": {"images": ["24", 0], "fps": fps}},
        "26": {"class_type": "SaveVideo",
               "inputs": {"video": ["25", 0], "filename_prefix": "videogen/ltxmsr",
                          "format": "auto", "codec": "auto"}},
        # background LoadImage (required) + subject LoadImages wired into LiconMSR
        "35": {"class_type": "LoadImage", "inputs": {"image": background_ref}},
    }
    for i, r in enumerate(subs):                          # subjects -> LiconMSR inputs "1".."4"
        nid = str(30 + i)
        g[nid] = {"class_type": "LoadImage", "inputs": {"image": r}}
        g["13"]["inputs"][str(i + 1)] = [nid, 0]
    if vocal_ref:                                          # native single-pass lip-sync: real vocal
        # encode the supplied vocal and feed it as the AV audio latent instead of the empty one, so
        # the sampler denoises video CONDITIONED on the song -> lips track the singing in one pass.
        # NOTE: the vocal must be trimmed to EXACTLY the clip duration upstream - an over-long audio
        # latent misaligns LTXVConcat/SeparateAVLatent and leaks uncropped MSR reference frames.
        g["27"] = {"class_type": "LoadAudio", "inputs": {"audio": vocal_ref}}
        g["28"] = {"class_type": "LTXVAudioVAEEncode",
                   "inputs": {"audio": ["27", 0], "audio_vae": ["7", 0]}}
        # CRITICAL (from the official LTX ia2v workflow): mask the audio latent with a SolidMask of
        # value 0 -> a noise mask of all zeros = "preserve, do NOT denoise". This holds the vocal
        # FIXED through sampling so the video is generated to MATCH it (real lip-sync). Without this
        # the sampler denoises the audio away -> mouth moves but does not track the words.
        g["36"] = {"class_type": "SolidMask", "inputs": {"value": 0.0, "width": 1024, "height": 1024}}
        g["37"] = {"class_type": "SetLatentNoiseMask",
                   "inputs": {"samples": ["28", 0], "mask": ["36", 0]}}
        g["16"]["inputs"]["audio_latent"] = ["37", 0]     # masked vocal latent (was empty node 15)
        del g["15"]                                        # empty audio no longer used
        g["25"]["inputs"]["audio"] = ["27", 0]            # mux the driving vocal into the output
    return g, {"seed": seed, "width": w, "height": h, "frames": frames, "fps": fps,
               "seconds": round(frames / fps, 2), "steps": steps, "cfg": cfg, "msr_strength": msr_str,
               "guide_strength": guide_str, "ref_frames": ref_frames, "subjects": len(subs),
               "prompt": prompt, "lipsync": bool(vocal_ref), "kind": "video"}


# ------------------------------------------------ SeedVR2 video upscale (post-process)
def build_seedvr2_upscale(p, video_ref, fps):
    """Upscale a finished clip/video with SeedVR2 (diffusion video upscaler, temporal-aware).
    video_ref = uploaded video name on ComfyUI; fps = source fps (probed by the caller so the
    output timing matches). p: {model?, resolution?, batch_size?, color_correction?,
    blocks_to_swap?, offload?, frame_cap?, seed?}. resolution = target SHORT side (e.g. 1080 ->
    1920x1080 for 16:9). Output: SaveVideo -> videogen/upscale (keeps the source audio)."""
    seed = _seed(p)
    model = p.get("model") or SEEDVR2_DIT_DEFAULT
    resolution = int(p.get("resolution", 1440))            # target short side; 1440 = 2x from 720p
    # batch_size MUST be 4n+1 (5, 9, 13, ...). 4K is much heavier (4K/batch5 peaked ~20.8GB) so
    # keep it small there. At 1440, batch 13 rode the ceiling (~23.7GB, held only by aggressive
    # offload) - so default 9 for a ~2-3GB margin on the longer clips, still better coherence
    # than 5. Caller can override either.
    default_batch = 5 if resolution >= 2000 else 9
    batch = int(p.get("batch_size", default_batch))
    if (batch - 1) % 4 != 0:                               # SeedVR2 requires 4n+1
        batch = default_batch
    cc = p.get("color_correction") or "wavelet"           # match upscaled colors to the source
    blocks = int(p.get("blocks_to_swap", 0))              # raise if the DiT OOMs in diffusion (7B/4K)
    # DiT offload to CPU after diffusion so the ~7GB model is NOT resident during VAE decode.
    # "none" left it on-GPU and the decode phase fragmented/OOM'd on a 24GB card (it self-healed
    # via empty_cache per batch, but slowly). cpu = clean headroom for decode.
    offload = p.get("offload") or "cpu"
    # Tile BOTH VAE encode and decode. SeedVR2 resizes the input UP to the target resolution
    # before VAE-encoding it, so at 4K the encode is a 3744x2160 block that OOMs the GroupNorm
    # concat. Both phases are off-by-default in the node; we default them ON (mandatory at 2K/4K).
    tiled = bool(p.get("tiled", True))
    tile = int(p.get("tile_size", 1024))
    cap = int(p.get("frame_cap", 0))                      # 0 = all frames
    g = {
        "1": {"class_type": "VHS_LoadVideo",
              "inputs": {"video": video_ref, "force_rate": 0.0, "custom_width": 0,
                         "custom_height": 0, "frame_load_cap": cap, "skip_first_frames": 0,
                         "select_every_nth": 1}},
        "2": {"class_type": "SeedVR2LoadDiTModel",
              "inputs": {"model": model, "device": "cuda:0", "blocks_to_swap": blocks,
                         "offload_device": offload, "attention_mode": "sdpa"}},
        "3": {"class_type": "SeedVR2LoadVAEModel",
              "inputs": {"model": SEEDVR2_VAE, "device": "cuda:0",
                         "encode_tiled": tiled, "encode_tile_size": tile,
                         "encode_tile_overlap": 128,
                         "decode_tiled": tiled, "decode_tile_size": tile,
                         "decode_tile_overlap": 128}},
        "4": {"class_type": "SeedVR2VideoUpscaler",
              "inputs": {"image": ["1", 0], "dit": ["2", 0], "vae": ["3", 0], "seed": seed,
                         "resolution": resolution, "max_resolution": 0, "batch_size": batch,
                         "uniform_batch_size": False, "color_correction": cc,
                         "offload_device": "cpu"}},
        # frame_count=[1,1], audio=[1,2], video_info=[1,3] are the other VHS_LoadVideo outputs.
        "5": {"class_type": "CreateVideo",
              "inputs": {"images": ["4", 0], "fps": float(fps), "audio": ["1", 2]}},
        "6": {"class_type": "SaveVideo",
              "inputs": {"video": ["5", 0], "filename_prefix": "videogen/upscale",
                         "format": "auto", "codec": "auto"}},
    }
    return g, {"seed": seed, "model": model, "resolution": resolution, "batch_size": batch,
               "color_correction": cc, "blocks_to_swap": blocks, "offload": offload,
               "tiled": tiled, "tile_size": tile, "fps": fps, "kind": "video"}


# ------------------------------------------------ SVI2 Pro: long-form Wan 2.2 A14B i2v
# Two-expert (high/low noise) Wan 2.2 i2v, 4-step lightx2v, extended to long video via the
# Wan22FMLF SVI nodes: each 81-frame segment chains off the previous segment's latent (motion
# continuity) and the decoded segments are overlap-blended into one clip. fp8 native, no GGUF.
# Model/lora chains + sampler split mirror wallen0322's "SVI pro" reference workflow.
SVI_I2V_HIGH = "wan2.2_i2v_A14b_high_noise_scaled_fp8_e4m3_lightx2v_4step_comfyui_1030.safetensors"
SVI_I2V_LOW = "wan2.2_i2v_A14b_low_noise_scaled_fp8_e4m3_lightx2v_4step_comfyui.safetensors"
# Non-distilled bases (same 14B fp8 size, same VRAM) - motion is controllable (more steps + CFG)
# so natural speed instead of the distilled models' slow-motion. model="full" selects these.
SVI_I2V_FULL_HIGH = "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
SVI_I2V_FULL_LOW = "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
SVI_LORA_HIGH = "SVI_v2_PRO_Wan2.2-I2V-A14B_HIGH_lora_rank_128_fp16.safetensors"
SVI_LORA_LOW = "SVI_v2_PRO_Wan2.2-I2V-A14B_LOW_lora_rank_128_fp16.safetensors"
SVI_LIGHTX_HIGH = "wan2.2_t2v_A14b_high_noise_lora_rank64_lightx2v_4step_1217.safetensors"
SVI_LIGHTX_LOW = "wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors"
SVI_CLIP = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
SVI_VAE = "wan_2.1_vae.safetensors"
SVI_SEG_FRAMES = 81          # frames per SVI segment (reference)
SVI_OVERLAP = 5              # overlap-blend frames between segments (reference)


def build_svi_i2v(p, image_ref):
    """SVI2 Pro long-form Wan 2.2 A14B i2v from a keyframe still. Chains N 81-frame segments
    (two-expert high/low, 4-step), each conditioned on the previous segment's latent, then
    overlap-blends the decoded segments into one continuous clip. image_ref = uploaded still.
    p: {prompt?, prompts? (list, one per segment - evolve the motion, keep identity constant),
    negative?, seed? (one seed for all - continuity comes from prev_latent), width?, height?,
    frames?/seconds?/segments?, fps?, overlap?, model? ("distilled"=fast/slow-motion default, or
    "full"=non-distilled), steps?, cfg?, lightx_strength?}. Output: SaveVideo -> videogen/svi.
    Validated 2026-06-16: seam-free joins (same seed + video_frame_offset 0->4 + continue_frames
    1), per-segment prompts, distilled output is uniformly slow -> retime to natural speed via
    /api/video/retime (full mode gives differential subject/scene motion + ping-pong, not used)."""
    seed = _seed(p)
    w = int(p.get("width", 832))
    h = int(p.get("height", 480))
    fps = int(p.get("fps", 16))
    seg = SVI_SEG_FRAMES
    ov = int(p.get("overlap", SVI_OVERLAP))
    # model: "distilled" (4-step lightx baked, fast but slow-motion) or "full" (non-distilled,
    # natural motion via more steps + CFG). Each picks its base files + sensible step/cfg/lightx
    # defaults; all still overridable. "full" needs the non-distilled models on the box.
    full = (p.get("model") or "distilled").lower() in ("full", "natural", "nondistilled", "non-distilled")
    hi_model = SVI_I2V_FULL_HIGH if full else SVI_I2V_HIGH
    lo_model = SVI_I2V_FULL_LOW if full else SVI_I2V_LOW
    steps = int(p.get("steps", 20 if full else 6))     # total; split half high / half low
    split = max(1, steps // 2)
    cfg = float(p.get("cfg", 3.5 if full else 1.0))    # distilled needs cfg 1; full uses real CFG
    lightx = float(p.get("lightx_strength", 0.0 if full else 1.0))  # full: no speed-LoRA (natural motion)
    continue_frames = int(p.get("continue_frames", 1)) # latent continuity frames (reference: 1)
    seg_offset = int(p.get("segment_offset", 4))       # video_frame_offset for segments after the first
    if p.get("segments"):
        nseg = max(1, int(p["segments"]))
    else:
        total = int(round(float(p["seconds"]) * fps)) if p.get("seconds") else int(p.get("frames", seg))
        nseg = max(1, 1 + -(-(max(seg, total) - seg) // (seg - ov)))   # ceil over (seg-overlap) steps
    prompt = (p.get("prompt") or "").strip()
    # Per-segment prompts: pass `prompts` (list) to evolve the motion across the clip - each
    # segment gets its own CLIPTextEncode. Keep the subject identity verbatim-constant across
    # them (only change motion/camera) to avoid drift. Falls back to the single `prompt` for any
    # uncovered segment, and for all segments if no list is given.
    _pl = p.get("prompts")
    if isinstance(_pl, list) and any((x or "").strip() for x in _pl):
        seg_prompts = [(x or "").strip() or prompt for x in _pl]
    else:
        seg_prompts = [prompt]
    neg = p.get("negative")
    neg = DEFAULT_NEG if neg is None else neg
    g = {
        # high-noise expert chain: base -> lightx2v(t2v high) -> SVI HIGH -> ModelSamplingSD3
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": hi_model, "weight_dtype": "default"}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": SVI_LIGHTX_HIGH, "strength_model": lightx}},
        "3": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["2", 0], "lora_name": SVI_LORA_HIGH, "strength_model": 1.0}},
        "4": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["3", 0], "shift": 5.0}},
        # low-noise expert chain: base -> lightx2v(i2v low) -> SVI LOW -> ModelSamplingSD3
        "5": {"class_type": "UNETLoader", "inputs": {"unet_name": lo_model, "weight_dtype": "default"}},
        "6": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["5", 0], "lora_name": SVI_LIGHTX_LOW, "strength_model": lightx}},
        "7": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["6", 0], "lora_name": SVI_LORA_LOW, "strength_model": 1.0}},
        "8": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["7", 0], "shift": 5.0}},
        "9": {"class_type": "CLIPLoader", "inputs": {"clip_name": SVI_CLIP, "type": "wan", "device": "default"}},
        "10": {"class_type": "VAELoader", "inputs": {"vae_name": SVI_VAE}},
        # positive prompt(s) are per-segment (created in the loop); 12 = shared negative
        "12": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["9", 0], "text": neg}},
        "13": {"class_type": "LoadImage", "inputs": {"image": image_ref}},
    }
    prev_low = None          # previous segment's low-noise latent = motion continuity
    decodes = []
    nid = 20
    for s in range(nseg):
        adv, hi, lo, dec = str(nid), str(nid + 1), str(nid + 2), str(nid + 3)
        pos = str(nid + 4)                                   # this segment's positive encode
        seg_prompt = seg_prompts[min(s, len(seg_prompts) - 1)]   # clamp to last if fewer than segments
        g[pos] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["9", 0], "text": seg_prompt}}
        # reference uses ONE seed across all segments (prev_latent gives continuity); varying it
        # per segment caused the look/colour shift at the join. video_frame_offset = 0 first, then
        # seg_offset for continuations (reference: 0, 4, 4...).
        adv_in = {"positive": [pos, 0], "negative": ["12", 0], "vae": ["10", 0],
                  "width": w, "height": h, "length": seg, "batch_size": 1,
                  "mode": "NORMAL", "long_video_mode": "SVI", "start_image": ["13", 0],
                  "video_frame_offset": (0 if s == 0 else seg_offset),
                  "continue_frames_count": continue_frames, "enable_middle_frame": False}
        if prev_low is not None:
            adv_in["prev_latent"] = prev_low
        g[adv] = {"class_type": "WanAdvancedI2V", "inputs": adv_in}
        # WanAdvancedI2V outs: 0 positive_high, 1 positive_low, 2 negative, 3 latent
        g[hi] = {"class_type": "KSamplerAdvanced",
                 "inputs": {"add_noise": "enable", "noise_seed": seed, "steps": steps, "cfg": cfg,
                            "sampler_name": "euler", "scheduler": "simple", "start_at_step": 0,
                            "end_at_step": split, "return_with_leftover_noise": "enable",
                            "model": ["4", 0], "positive": [adv, 0], "negative": [adv, 2],
                            "latent_image": [adv, 3]}}
        g[lo] = {"class_type": "KSamplerAdvanced",
                 "inputs": {"add_noise": "disable", "noise_seed": seed, "steps": steps, "cfg": cfg,
                            "sampler_name": "euler", "scheduler": "simple", "start_at_step": split,
                            "end_at_step": 10000, "return_with_leftover_noise": "disable",
                            "model": ["8", 0], "positive": [adv, 1], "negative": [adv, 2],
                            "latent_image": [hi, 0]}}
        g[dec] = {"class_type": "VAEDecode", "inputs": {"samples": [lo, 0], "vae": ["10", 0]}}
        prev_low = [lo, 0]
        decodes.append(dec)
        nid += 10
    # overlap-blend the decoded segments (ImageBatchExtendWithOverlap, out2 = extended_images)
    if len(decodes) == 1:
        frames_out = [decodes[0], 0]
    else:
        stitch = None
        sid = nid
        for i in range(1, len(decodes)):
            src = stitch if stitch else [decodes[0], 0]
            g[str(sid)] = {"class_type": "ImageBatchExtendWithOverlap",
                           "inputs": {"source_images": src, "new_images": [decodes[i], 0],
                                      "overlap": ov, "overlap_side": "source", "overlap_mode": "linear_blend"}}
            stitch = [str(sid), 2]
            sid += 1
        frames_out = stitch
    g["900"] = {"class_type": "CreateVideo", "inputs": {"images": frames_out, "fps": fps}}
    g["901"] = {"class_type": "SaveVideo",
                "inputs": {"video": ["900", 0], "filename_prefix": "videogen/svi",
                           "format": "auto", "codec": "auto"}}
    total_frames = seg + (nseg - 1) * (seg - ov)
    return g, {"seed": seed, "width": w, "height": h, "segments": nseg, "seg_frames": seg,
               "overlap": ov, "frames": total_frames, "fps": fps,
               "seconds": round(total_frames / fps, 2), "prompt": prompt,
               "segment_prompts": len(seg_prompts) if len(seg_prompts) > 1 else 1,
               "model": "full" if full else "distilled", "steps": steps, "cfg": cfg,
               "lightx_strength": lightx, "continue_frames": continue_frames,
               "segment_offset": seg_offset, "kind": "video"}
