#!/usr/bin/env python
"""Download the Wan2.2 photoreal-video-pipeline models into ComfyUI's models/ tree.

Run on the Windows box with the ComfyUI portable python:
    python_embeded\\python.exe download_video_models.py [min|full]

  min  (default) = the Phase 1 GO/NO-GO gate subset (~45-55 GB): enough to prove
                   photoreal stills -> i2v + S2V lip-sync before committing the full set.
  full           = adds the 14B i2v hero pair (+ optional t2v / s2v-bf16 if uncommented).

Every model here is UNGATED (Apache 2.0 / Comfy-Org repackaged) - NO HuggingFace token
needed. Files land flat in  ComfyUI/models/<subdir>/<basename>  (the split_files/ repo
prefix is stripped). Already-present files are skipped. ASCII only. No GPU used.

Stills model = Z-Image Turbo (best-reported photoreal skin/faces, ~1-2 min/image, fits
16GB, Apache 2.0). See RESEARCH.md s19a + memory [[photoreal-image-models]].
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
# This script sits in the portable root next to ComfyUI\ and python_embeded\.
MODELS = os.path.join(HERE, "ComfyUI", "models")

WAN22 = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
WAN21 = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"
LIGHTX = "lightx2v/Wan2.2-Distill-Loras"
ZIMAGE = "Comfy-Org/z_image_turbo"

# (repo_id, in-repo filename, target models subdir)
MIN_SET = [
    # --- stills (Z-Image Turbo, ungated) ---
    (ZIMAGE, "split_files/diffusion_models/z_image_turbo_bf16.safetensors", "diffusion_models"),
    (ZIMAGE, "split_files/text_encoders/qwen_3_4b.safetensors", "text_encoders"),
    (ZIMAGE, "split_files/vae/ae.safetensors", "vae"),
    # --- Wan2.2 video (fast iter + lip-sync gate) ---
    (WAN22, "split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors", "diffusion_models"),
    (WAN21, "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "text_encoders"),
    (WAN22, "split_files/vae/wan2.2_vae.safetensors", "vae"),
    (WAN22, "split_files/vae/wan_2.1_vae.safetensors", "vae"),
    (LIGHTX, "wan2.2_t2v_A14b_high_noise_lora_rank64_lightx2v_4step_1217.safetensors", "loras"),
    (LIGHTX, "wan2.2_t2v_A14b_low_noise_lora_rank64_lightx2v_4step_1217.safetensors", "loras"),
    (WAN22, "split_files/diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors", "diffusion_models"),
    (WAN22, "split_files/audio_encoders/wav2vec2_large_english_fp16.safetensors", "audio_encoders"),
]

FULL_EXTRA = [
    # 14B i2v HERO pair (fp16; swap for Kijai fp8 to halve disk/VRAM if needed)
    (WAN22, "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors", "diffusion_models"),
    (WAN22, "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors", "diffusion_models"),
    # Optional pure-t2v 14B fp8 pair (uncomment if you want text-to-video with no start still):
    # (WAN22, "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors", "diffusion_models"),
    # (WAN22, "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors", "diffusion_models"),
    # Optional S2V bf16 (higher quality, ~28GB) if fp8 lip-sync looks weak:
    # (WAN22, "split_files/diffusion_models/wan2.2_s2v_14B_bf16.safetensors", "diffusion_models"),
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
    # hf_hub_download with local_dir preserves the repo path under dest_dir; flatten it.
    if os.path.realpath(p) != os.path.realpath(target):
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
    mode = (sys.argv[1] if len(sys.argv) > 1 else "min").lower()
    if mode not in ("min", "full"):
        print("usage: download_video_models.py [min|full]")
        sys.exit(2)
    items = list(MIN_SET) + (FULL_EXTRA if mode == "full" else [])
    print("== Wan2.2 + Z-Image video pipeline model download (mode={}) ==".format(mode))
    print("   target:", MODELS)
    print("   (all ungated; no HF token required)")
    ok = 0
    fail = []
    for repo, fname, subdir in items:
        if fetch(repo, fname, subdir):
            ok += 1
        else:
            fail.append(os.path.basename(fname))
    print("\n== {} of {} files ready ==".format(ok, len(items)))
    if fail:
        print("   missing/failed:", ", ".join(fail))
        print("   re-run to retry; it skips what is already done.")
    else:
        print("   all good.")


if __name__ == "__main__":
    main()
