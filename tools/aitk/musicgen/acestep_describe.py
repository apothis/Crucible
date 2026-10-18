"""Describe songs with ACE-Step's music captioner ONLY (ACE-Step/acestep-captioner, a Qwen2.5-Omni-7B
music fine-tune), same prompt / greedy decode / repetition penalty as AI Toolkit's AceStepCaptioner.
That job always runs the lyric transcriber too, and with both 7B models on the 3090 it ran out of memory
and took 6-9 min a song; our lyrics come from the online fetch, so only the description is needed.
Config (ai-toolkit\\config\\<name>.json): {"repo": "ACE-Step/acestep-captioner", "dir": "<folder of mp3>",
"ext": "acecap", "qtype": "float8"}. Writes <stem>.<ext> beside each song; existing files are kept."""
import json
import os
import sys
import time

import torch
import torchaudio
from optimum.quanto import freeze
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

from extensions_built_in.captioner.AceStepCaptioner import CAPTION_PROMPT, CAPTIONER_REPETITION_PENALTY, MAX_NEW_TOKENS, TARGET_SAMPLE_RATE
from extensions_built_in.llm_models.src.thinker import prepare_thinker
from toolkit.util.quantize import get_qtype, quantize


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(cfg_path):
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    files = sorted(f for f in os.listdir(cfg["dir"]) if f.lower().endswith(".mp3"))
    todo = [f for f in files if not os.path.exists(os.path.join(cfg["dir"], f[:-4] + "." + cfg["ext"]))]
    log(f"{len(files)} songs, {len(todo)} to describe")
    full = Qwen2_5OmniForConditionalGeneration.from_pretrained(cfg["repo"], dtype=torch.bfloat16, device_map="cpu")
    model = prepare_thinker(full)
    model.to("cuda")
    if cfg.get("qtype"):
        quantize(model, weights=get_qtype(cfg["qtype"]))
        freeze(model)
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(cfg["repo"])
    conversation = [{"role": "user", "content": [{"type": "audio", "audio": "<|audio_bos|><|AUDIO|><|audio_eos|>"},
                                                 {"type": "text", "text": CAPTION_PROMPT}]}]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    for f in todo:
        t0 = time.time()
        wav, sr = torchaudio.load(os.path.join(cfg["dir"], f))
        wav = wav.mean(0, keepdim=True)
        if sr != TARGET_SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, TARGET_SAMPLE_RATE)
        inputs = processor(text=text, audio=[wav.squeeze(0).numpy()], images=None, videos=None, return_tensors="pt",
                           padding=True, sampling_rate=TARGET_SAMPLE_RATE)
        inputs = {k: (v.to("cuda", torch.bfloat16) if v.is_floating_point() else v.to("cuda")) for k, v in inputs.items()}
        n = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, repetition_penalty=CAPTIONER_REPETITION_PENALTY, do_sample=False)
        desc = processor.batch_decode(out[:, n:], skip_special_tokens=True)[0].strip()
        open(os.path.join(cfg["dir"], f[:-4] + "." + cfg["ext"]), "w", encoding="utf-8").write(desc + "\n")
        log(f"{f}: {time.time() - t0:.0f} s, {len(desc)} chars: {desc[:160]}")
        torch.cuda.empty_cache()
    log("DESCRIBE DONE")


if __name__ == "__main__":
    main(sys.argv[1])
