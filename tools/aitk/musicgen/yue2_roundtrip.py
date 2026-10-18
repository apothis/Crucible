"""YuE2 round trip: a REAL song -> YuE2 music tokens (the community audio tokenizer head, as training
uses) -> YuE2's decoder -> audio. No planning: the decoder is handed the song's own tokens, so what
comes out is the ceiling of what YuE2 can reproduce of that recording. If the singer is recognisable
here, the tokens carry the voice and the planner is what a LoRA must teach; if not, the tokenizer or
the decoder loses it and planner training cannot bring it back.

Runs in AI Toolkit's venv on the box (helper POST /aitk/script {script: yue2_roundtrip, config}), with
the trainer's own model code. Config (ai-toolkit\\config\\<name>.json):
  {"head": "<semantic head 'org/repo/file' or path>", "nar_lora": "<tokenizer's decoder adapter>",
   "checkpoint": "<yue2_3b_bf16.safetensors>", "out_dir": "...", "steps": 32, "seed": 7101,
   "items":    [{"label": "...", "audio": "<mp3>", "style": "...", "lyrics": "..."}],
   "variants": [{"label": "...", "lora": null | "<band LoRA .safetensors>", "strength": 1.0,
                 "only": ["<item label>", ...] (optional: render just these items)}]}
Every item is tokenized once; each variant reloads the base model, merges the tokenizer's decoder adapter
(as training does) and, if given, the band LoRA into BOTH experts (AR = the conditioning the decoder
reads, NAR = the decoder), then renders every item. Output: <out_dir>/<variant label> - <item label>.wav
"""
import json
import os
import sys
import time
import wave

import numpy as np
import torch
import torchaudio
from safetensors.torch import load_file

from extensions_built_in.audio_models.yue2.src.model import YuE2Model, merge_nar_lora
from extensions_built_in.audio_models.yue2.src.pipeline import YuE2Pipeline
from extensions_built_in.audio_models.yue2.src.tokenizer import SemanticTokenizer, YuE2TextTokenizer
from extensions_built_in.audio_models.yue2.src.vae import SAMPLE_RATE, YuE2VAE
from extensions_built_in.audio_models.yue2.yue2_model import resolve_ai_toolkit_file


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def merge_band_lora(model, path, strength):
    """AI Toolkit YuE2 LoRA (peft A/B, alpha = rank): diffusion_model.* = NAR, text_encoders.* = AR."""
    sd = load_file(path, device="cpu")
    done = 0
    for key in sorted(k for k in sd if k.endswith(".lora_A.weight")):
        base = key[: -len(".lora_A.weight")]
        if base.startswith("diffusion_model."):
            name = "nar." + base[len("diffusion_model."):]
        elif base.startswith("text_encoders."):
            name = "ar." + base[len("text_encoders."):]
        else:
            raise ValueError(f"unknown LoRA key {key}")
        a, b = sd[key].float(), sd[base + ".lora_B.weight"].float()
        alpha = sd.get(base + ".alpha")
        scale = strength * (float(alpha) / a.shape[0] if alpha is not None else 1.0)
        w = model.get_submodule(name).weight
        with torch.no_grad():
            w.add_(((b @ a) * scale).to(w.device, w.dtype))
        done += 1
    log(f"band LoRA {os.path.basename(path)} x{strength}: {done} layers patched")
    if done == 0:
        raise ValueError("band LoRA patched nothing")


def save_wav(path, audio):
    x = audio[0].float().clamp(-1, 1).cpu().numpy().T          # [samples, 2]
    pcm = (x * 32767.0).round().astype(np.int16)
    with wave.open(path, "wb") as f:
        f.setnchannels(pcm.shape[1])
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(pcm.tobytes())


def main(cfg_path):
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    device = torch.device("cuda")
    os.makedirs(cfg["out_dir"], exist_ok=True)

    # 1. tokens for every item (the head that trained the LoRA)
    log("loading semantic tokenizer (MERT-v2-FullSong + head)")
    tok = SemanticTokenizer(resolve_ai_toolkit_file(cfg["head"], "yue2 semantic head")).to(device)
    codes = []
    for it in cfg["items"]:
        wav, sr = torchaudio.load(it["audio"])
        if it.get("max_seconds"):
            wav = wav[:, : int(sr * float(it["max_seconds"]))]
        c = tok.tokenize(wav, sr).tolist()
        codes.append(c)
        log(f"tokenized {it['label']}: {len(c)} tokens ({len(c) / 25:.1f} s), {len(set(c))} distinct")
    del tok
    torch.cuda.empty_cache()

    # 2. render each variant
    sd = load_file(cfg["checkpoint"])
    tokenizer = YuE2TextTokenizer(sd.pop("text_encoders.yue2_tokenizer_json").numpy().tobytes())
    vae = YuE2VAE.load_from_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("vae.")}, dtype=torch.float32).to(device)
    for var in cfg["variants"]:
        t0 = time.time()
        model = YuE2Model.load_from_state_dict(sd, dtype=torch.bfloat16)
        merge_nar_lora(model, resolve_ai_toolkit_file(cfg["nar_lora"], "yue2 nar lora"))
        if var.get("lora"):
            merge_band_lora(model, var["lora"], float(var.get("strength", 1.0)))
        model.to(device).eval()
        pipe = YuE2Pipeline(model, vae)
        pipe.do_tiled_decoding = True
        log(f"variant '{var['label']}' loaded in {time.time() - t0:.0f} s")
        for it, codec in zip(cfg["items"], codes):
            if var.get("only") and it["label"] not in var["only"]:
                continue
            t1 = time.time()
            # the off-mode prompt (no score): prefix head + empty ABC block, then the song's tokens
            ids = tokenizer.prefix_head_ids(it["style"], it["lyrics"], cot="off") + tokenizer.abc_tail_ids([])
            with torch.no_grad():
                prefix = model.ar.embed(torch.tensor([ids], device=device))
                latents = pipe.synthesize(prefix, codec, seed=int(cfg.get("seed", 7101)), steps=int(cfg.get("steps", 32)))
                audio = pipe.decode(latents)
            out = os.path.join(cfg["out_dir"], f"{var['label']} - {it['label']}.wav")
            save_wav(out, audio)
            log(f"wrote {out} ({audio.shape[-1] / SAMPLE_RATE:.1f} s, {time.time() - t1:.0f} s)")
        del pipe, model
        torch.cuda.empty_cache()
    log("ROUNDTRIP DONE")


if __name__ == "__main__":
    main(sys.argv[1])
