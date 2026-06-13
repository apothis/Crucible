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
    prompt = (p.get("prompt") or "").strip()
    neg = p.get("negative")
    neg = DEFAULT_NEG if neg is None else neg
    g = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": Z_IMAGE_UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": Z_IMAGE_CLIP, "type": "lumina2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": Z_IMAGE_VAE}},
        "4": {"class_type": "ModelSamplingAuraFlow",
              "inputs": {"model": ["1", 0], "shift": 3.0}},
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
    }
    return g, {"seed": seed, "width": w, "height": h, "steps": steps, "cfg": cfg,
               "prompt": prompt, "kind": "image"}


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
