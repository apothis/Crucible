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
# lightx2v 4-step distillation LoRA (the official S2V template applies the t2v high-noise
# one to the single S2V model): cuts 20 steps -> 4 and CFG 6 -> 1 (~10x fewer DiT evals).
WAN_LIGHTX_HIGH = "wan2.2_t2v_A14b_high_noise_lora_rank64_lightx2v_4step_1217.safetensors"

# Qwen-Image-Edit-2511 (GGUF) - reference-driven character consistency, no training.
# Files from download_video_models2.py. GGUF UNET loads via UnetLoaderGGUF (ComfyUI-GGUF).
QWEN_EDIT_GGUF = "qwen-image-edit-2511-Q6_K.gguf"
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
LTX_UNET_GGUF = LTX_UNET_TMPL.format(quant=LTX_QUANT_DEFAULT)  # default file
LTX_CLIP1 = "gemma_3_12B_it_fp4_mixed.safetensors"   # DualCLIPLoader clip_name1
LTX_CLIP2 = "ltx-2.3_text_projection_bf16.safetensors"  # clip_name2; type "ltxv"
LTX_VAE_VIDEO = "LTX23_video_vae_bf16.safetensors"   # VAELoaderKJ main_device/bf16
LTX_VAE_AUDIO = "LTX23_audio_vae_bf16.safetensors"   # VAELoaderKJ cpu/bf16
LTX_LORA_DISTILL = "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"  # few-step distill (req'd for 8-step)
LTX_LORA_DETAILER = "ltx-2-19b-ic-lora-detailer.safetensors"         # texture/detail
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
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": QWEN_EDIT_GGUF}},
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
    g["13"] = {"class_type": "KSampler",
               "inputs": {"model": ["5", 0], "seed": seed, "steps": steps, "cfg": cfg,
                          "sampler_name": "euler", "scheduler": "simple",
                          "positive": ["10", 0], "negative": ["11", 0],
                          "latent_image": ["12", 0], "denoise": 1.0}}
    g["14"] = {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["3", 0]}}
    g["15"] = {"class_type": "SaveImage",
               "inputs": {"images": ["14", 0], "filename_prefix": "videogen/charstill"}}
    return g, {"seed": seed, "steps": steps, "cfg": cfg, "prompt": prompt,
               "refs": len(refs), "kind": "image"}


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
    cfg = float(p.get("cfg", 1.0))                     # distilled LoRA -> CFG 1 (negative ignored)
    distill = float(p.get("distill_strength", 0.5))
    detailer = float(p.get("detailer_strength", 0.2))
    img_strength = float(p.get("img_strength", 0.7))   # keyframe imprint strength (i2v)
    lips_expr = float(p.get("lips_expression", 1.5))   # LatentSync lip-movement intensity 1.0-3.0
    lip_steps = int(p.get("inference_steps", 20))      # LatentSync denoise steps
    quant = (p.get("quant") or LTX_QUANT_DEFAULT).strip()
    if quant not in LTX_QUANTS:
        quant = LTX_QUANT_DEFAULT
    unet = LTX_UNET_TMPL.format(quant=quant)
    prompt = (p.get("prompt") or "").strip()
    g = {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": unet}},
        "2": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["1", 0], "lora_name": LTX_LORA_DISTILL,
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
                         "duration_frames": frames, "duration_seconds": secs,
                         "timeline_data": "{\"segments\":[],\"audioSegments\":[]}",
                         "local_prompts": prompt, "segment_lengths": "", "epsilon": 0.001,
                         "guide_strength": "", "audio_vae": ["6", 0],
                         "use_custom_audio": False, "frame_rate": float(fps),
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
    if image_ref is not None:                          # i2v: imprint the keyframe as frame 0
        g["30"] = {"class_type": "LoadImage", "inputs": {"image": image_ref}}
        g["31"] = {"class_type": "LTXVPreprocess",
                   "inputs": {"image": ["30", 0], "img_compression": 18}}
        g["32"] = {"class_type": "LTXVImgToVideoInplace",
                   "inputs": {"vae": ["5", 0], "image": ["31", 0], "latent": ["7", 2],
                              "strength": img_strength, "bypass": False}}
        g["10"]["inputs"]["video_latent"] = ["32", 0]  # concat uses the imprinted latent
    if lipsync_audio is not None:                       # LatentSync mouth-only post pass on LTX frames
        g["40"] = {"class_type": "LoadAudio", "inputs": {"audio": lipsync_audio}}
        g["41"] = {"class_type": "LatentSyncNode",
                   "inputs": {"images": ["17", 0], "audio": ["40", 0], "seed": seed,
                              "lips_expression": lips_expr, "inference_steps": lip_steps}}
        g["19"]["inputs"]["images"] = ["41", 0]         # save the lip-synced frames...
        g["19"]["inputs"]["audio"] = ["40", 0]          # ...carrying the vocal (not LTX native audio)
    resolved = {"seed": seed, "width": w, "height": h, "frames": frames, "fps": fps,
                "seconds": secs, "cfg": cfg, "quant": quant, "distill_strength": distill,
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
