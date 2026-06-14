#!/usr/bin/env python
"""Download the LTX-2.3 video stack (AItrepreneur's proven set) into ComfyUI's models/ tree.
Run on the box with the portable python:
    python_embeded\\python.exe download_ltx_models.py

All from Aitrepreneur/FLX (public mirror, no token). ~48GB. Q8_0 22B model = best quality
for a 24GB card. GGUF UNET loads via ComfyUI-GGUF; the rest are LTXVideo node loaders.
Plain ASCII. No GPU used.
"""
import os
import sys

try:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError
except Exception as e:  # noqa
    print("[ERROR] huggingface_hub not importable:", e)
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "ComfyUI", "models")
REPO = "Aitrepreneur/FLX"

# (filename, target models subdir)
ITEMS = [
    ("ltx-2.3_text_projection_bf16.safetensors", "text_encoders"),
    ("gemma_3_12B_it_fp4_mixed.safetensors", "text_encoders"),
    ("LTX23_video_vae_bf16.safetensors", "vae"),
    ("LTX23_audio_vae_bf16.safetensors", "vae"),
    ("ltx-2.3-22b-dev-Q8_0.gguf", "unet"),
    ("ltx-2.3-spatial-upscaler-x2-1.1.safetensors", "latent_upscale_models"),
    ("ltx-2.3-22b-distilled-lora-384-1.1.safetensors", "loras"),
    ("ltx-2-19b-ic-lora-detailer.safetensors", "loras"),
]


def fetch(fname, subdir):
    dest_dir = os.path.join(MODELS, subdir)
    os.makedirs(dest_dir, exist_ok=True)
    target = os.path.join(dest_dir, fname)
    if os.path.exists(target) and os.path.getsize(target) > 0:
        print("  [skip] already present:", os.path.join(subdir, fname))
        return True
    print("  [get ] {}/{}".format(subdir, fname))
    try:
        p = hf_hub_download(repo_id=REPO, filename=fname, local_dir=dest_dir)
    except HfHubHTTPError as e:
        print("  [HTTP] failed ({}): {}".format(fname, e))
        return False
    except Exception as e:  # noqa
        print("  [ERR ] failed ({}): {}".format(fname, e))
        return False
    if os.path.realpath(p) != os.path.realpath(target):
        os.replace(p, target)
    print("  [done]", os.path.join(subdir, fname))
    return True


def main():
    print("== LTX-2.3 model download (Aitrepreneur/FLX, ~48GB, ungated) ==")
    print("   target:", MODELS)
    ok = 0
    fail = []
    for fname, subdir in ITEMS:
        if fetch(fname, subdir):
            ok += 1
        else:
            fail.append(fname)
    print("\n== {} of {} files ready ==".format(ok, len(ITEMS)))
    if fail:
        print("   missing/failed:", ", ".join(fail), "- re-run to retry (skips done).")
    else:
        print("   all good.")


if __name__ == "__main__":
    main()
