#!/usr/bin/env python
"""Download the RECOMMENDED music-video pipeline models (Phase B/C) as GGUF where it helps
VRAM, into ComfyUI's models/ tree. Companion to download_video_models.py (the Phase-1 set).

Run on the Windows box with the ComfyUI portable python:
    python_embeded\\python.exe download_video_models2.py

Adds (all UNGATED, no HF token; load GGUF UNETs via the installed ComfyUI-GGUF node):
  - Qwen-Image-Edit-2511 Q6_K GGUF (+ qwen text encoder + qwen vae) - reference-driven
    character consistency, no per-character training. ~16.9GB, fits 24GB.
  - Wan2.2 i2v A14B Q6_K GGUF (high+low noise) - hero image-to-video shots. ~13GB each.
  - Wan2.2-S2V-14B Q6_K GGUF - lip-sync; smaller resident set than fp8 (eases the spill).
  - Wan2.1 14B VACE Q6_K GGUF - reference-to-video identity (subject consistency in motion).
  - 4x-UltraSharpV2 upscaler - final-delivery upscale.

GGUF UNETs land in models/unet (UnetLoaderGGUF reads there). Q6_K = good quality/size on
24GB; swap the QUANT below to Q5_K_M for less VRAM. Already-present files are skipped.
Plain ASCII. No GPU used.
"""
import os
import sys

try:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError
except Exception as e:  # noqa
    print("[ERROR] huggingface_hub not importable in this python:", e)
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "ComfyUI", "models")

QWEN_EDIT_GGUF = "unsloth/Qwen-Image-Edit-2511-GGUF"
QWEN_BASE = "Comfy-Org/Qwen-Image_ComfyUI"
WAN_I2V_GGUF = "QuantStack/Wan2.2-I2V-A14B-GGUF"
WAN_S2V_GGUF = "QuantStack/Wan2.2-S2V-14B-GGUF"
WAN_VACE_GGUF = "QuantStack/Wan2.1_14B_VACE-GGUF"
UPSCALE = "Kim2091/UltraSharpV2"

# (repo_id, in-repo filename, target models subdir)
ITEMS = [
    # --- GGUF UNETs -> models/unet (load via UnetLoaderGGUF) ---
    (QWEN_EDIT_GGUF, "qwen-image-edit-2511-Q6_K.gguf", "unet"),
    (WAN_I2V_GGUF, "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q6_K.gguf", "unet"),
    (WAN_I2V_GGUF, "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q6_K.gguf", "unet"),
    (WAN_S2V_GGUF, "Wan2.2-S2V-14B-Q6_K.gguf", "unet"),
    (WAN_VACE_GGUF, "Wan2.1_14B_VACE-Q6_K.gguf", "unet"),
    # --- Qwen-Image-Edit text encoder + VAE ---
    (QWEN_BASE, "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", "text_encoders"),
    (QWEN_BASE, "split_files/vae/qwen_image_vae.safetensors", "vae"),
    # --- final-delivery upscaler ---
    (UPSCALE, "4x-UltraSharpV2.safetensors", "upscale_models"),
]


def fetch(repo, fname, subdir):
    dest_dir = os.path.join(MODELS, subdir)
    os.makedirs(dest_dir, exist_ok=True)
    base = os.path.basename(fname)
    target = os.path.join(dest_dir, base)
    if os.path.exists(target) and os.path.getsize(target) > 0:
        print("  [skip] already present:", os.path.join(subdir, base))
        return True
    print("  [get ] {}/{}  <-  {}".format(subdir, base, repo))
    try:
        p = hf_hub_download(repo_id=repo, filename=fname, local_dir=dest_dir)
    except HfHubHTTPError as e:
        print("  [HTTP] failed ({}): {}".format(base, e))
        return False
    except Exception as e:  # noqa
        print("  [ERR ] failed ({}): {}".format(base, e))
        return False
    if os.path.realpath(p) != os.path.realpath(target):   # flatten any subfolder path
        os.replace(p, target)
        d = os.path.dirname(p)
        while os.path.realpath(d) != os.path.realpath(dest_dir):
            try:
                os.rmdir(d)
            except OSError:
                break
            d = os.path.dirname(d)
    print("  [done]", os.path.join(subdir, base))
    return True


def main():
    print("== Music-video recommended models (GGUF, ungated) ==")
    print("   target:", MODELS)
    ok = 0
    fail = []
    for repo, fname, subdir in ITEMS:
        if fetch(repo, fname, subdir):
            ok += 1
        else:
            fail.append(os.path.basename(fname))
    print("\n== {} of {} files ready ==".format(ok, len(ITEMS)))
    if fail:
        print("   missing/failed:", ", ".join(fail))
        print("   re-run to retry; it skips what is already done.")
    else:
        print("   all good.")


if __name__ == "__main__":
    main()
