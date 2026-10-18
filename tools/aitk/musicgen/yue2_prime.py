"""YuE2 priming test: does the planner KEEP a singer's voice once its token stream starts in it?
The planner (AR) is given the song's prompt plus the first <prime_seconds> of the REAL song's music
tokens (community tokenizer head, as in training), then writes the next <gen_seconds> itself. Real +
generated tokens are decoded together, so the take plays the real opening and then the model's own
continuation; the question is whether the voice after the switch still sounds like the singer.
Config = yue2_roundtrip's, plus "prime_seconds", "gen_seconds" (and optional sampling: temperature,
top_p, top_k, repetition_penalty). Output: <out_dir>/<variant label> - <item label>.wav

Voice-primer use (a NEW song): per item, "prime_seconds" (0 = unprimed reference), "lyrics" = the primer's
own sung sections followed by the new song's (the model tracks its place in the lyrics, so the primer's
lines must be there or it skips the new song's opening), "gen_seconds" = the ceiling (the model ends the
song itself after "min_gen_seconds"), "trim": true = the saved audio starts where the primer ends."""
import json
import os
import sys
import time

import torch
import torchaudio
from safetensors.torch import load_file

from extensions_built_in.audio_models.yue2.src.model import CODEC_OFFSET, YuE2Model, merge_nar_lora
from extensions_built_in.audio_models.yue2.src.pipeline import YuE2Pipeline
from extensions_built_in.audio_models.yue2.src.tokenizer import SemanticTokenizer, YuE2TextTokenizer
from extensions_built_in.audio_models.yue2.src.vae import SAMPLE_RATE, YuE2VAE
from extensions_built_in.audio_models.yue2.yue2_model import resolve_ai_toolkit_file
from yue2_roundtrip import log, merge_band_lora, save_wav

FPS = 25


def main(cfg_path):
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    device = torch.device("cuda")
    os.makedirs(cfg["out_dir"], exist_ok=True)
    def per(it, key, default):
        return float(it.get(key, cfg.get(key, default)))

    log("loading semantic tokenizer")
    tok = SemanticTokenizer(resolve_ai_toolkit_file(cfg["head"], "yue2 semantic head")).to(device)
    primes = []
    for it in cfg["items"]:
        n_prime = int(per(it, "prime_seconds", 60) * FPS)
        if n_prime <= 0:
            primes.append([])
            log(f"{it['label']}: unprimed")
            continue
        wav, sr = torchaudio.load(it["audio"])
        wav = wav[:, int(sr * float(it.get("start_seconds", 0))):]
        wav = wav[:, : int(sr * (n_prime / FPS + 1))]
        c = tok.tokenize(wav, sr).tolist()[:n_prime]
        primes.append(c)
        log(f"prime {it['label']}: {len(c)} real tokens ({len(c) / FPS:.1f} s)")
    del tok
    torch.cuda.empty_cache()

    sd = load_file(cfg["checkpoint"])
    tokenizer = YuE2TextTokenizer(sd.pop("text_encoders.yue2_tokenizer_json").numpy().tobytes())
    vae = YuE2VAE.load_from_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("vae.")}, dtype=torch.float32).to(device)
    for var in cfg["variants"]:
        model = YuE2Model.load_from_state_dict(sd, dtype=torch.bfloat16)
        merge_nar_lora(model, resolve_ai_toolkit_file(cfg["nar_lora"], "yue2 nar lora"))
        if var.get("lora"):
            merge_band_lora(model, var["lora"], float(var.get("strength", 1.0)))
        model.to(device).eval()
        pipe = YuE2Pipeline(model, vae)
        pipe.do_tiled_decoding = True
        pipe.use_cuda_graphs = False          # the trainer's sampler crashed under CUDA graphs on this box
        for it, prime in zip(cfg["items"], primes):
            if var.get("only") and it["label"] not in var["only"]:
                continue
            t0 = time.time()
            n_gen = int(per(it, "gen_seconds", 60) * FPS)
            min_gen = int(per(it, "min_gen_seconds", n_gen / FPS) * FPS)
            head = tokenizer.prefix_head_ids(it["style"], it["lyrics"], cot="off") + tokenizer.abc_tail_ids([])
            with torch.no_grad():
                prompt = model.ar.embed(torch.tensor([head], device=device))
                primed = prompt if not prime else torch.cat(
                    [prompt, model.ar.embed(torch.tensor([[c + CODEC_OFFSET for c in prime]], device=device))], dim=1)
                gen = pipe.generate_codec_tokens(
                    primed, max_tokens=n_gen, seed=int(cfg.get("seed", 7101)),
                    temperature=float(cfg.get("temperature", 1.0)), top_p=float(cfg.get("top_p", 0.95)),
                    top_k=int(cfg.get("top_k", 100)), repetition_penalty=float(cfg.get("repetition_penalty", 1.2)),
                    min_tokens=max(1, min(min_gen, n_gen - 1)), legacy_off=True)
                codec = prime + gen
                latents = pipe.synthesize(prompt, codec, seed=int(cfg.get("seed", 7101)), steps=int(cfg.get("steps", 32)))
                audio = pipe.decode(latents)
            if it.get("trim") and prime:
                cut = int(len(prime) / FPS * SAMPLE_RATE)
                save_wav(os.path.join(cfg["out_dir"], f"{var['label']} - {it['label']} (with primer).wav"), audio)
                audio = audio[..., cut:]
            out = os.path.join(cfg["out_dir"], f"{var['label']} - {it['label']}.wav")
            save_wav(out, audio)
            log(f"wrote {out}: {len(prime) / FPS:.0f} s real + {len(gen) / FPS:.0f} s generated "
                f"({len(set(gen))} distinct generated tokens, {time.time() - t0:.0f} s)")
        del pipe, model
        torch.cuda.empty_cache()
    log("ROUNDTRIP DONE")


if __name__ == "__main__":
    main(sys.argv[1])
