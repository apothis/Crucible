"""Transcribe audio passages to ABC with SheetSage2 (the transcriber YuE2 training and ComfyUI covers
use), in AI Toolkit's venv. Config: {"sheetsage_path": "...", "cot": "full", "items": [{"audio": "...",
"start_seconds": 0, "seconds": 84, "out": "<.abc path>"}]}"""
import json
import sys
import time

import torch
import torchaudio

from extensions_built_in.audio_models.yue2.src.tokenizer import SheetSage2Transcriber

cfg = json.load(open(sys.argv[1], encoding="utf-8"))
tr = SheetSage2Transcriber(cfg["sheetsage_path"]).to("cuda")
for it in cfg["items"]:
    t0 = time.time()
    wav, sr = torchaudio.load(it["audio"])
    a = int(sr * float(it.get("start_seconds", 0)))
    wav = wav[:, a: a + int(sr * float(it["seconds"]))] if it.get("seconds") else wav[:, a:]
    abc = tr.transcribe(wav, sr, cot=cfg.get("cot", "full"))
    open(it["out"], "w", encoding="utf-8").write(abc)
    print(f"[{time.strftime('%H:%M:%S')}] {it['out']}: {len(abc)} chars, {time.time() - t0:.0f} s; warnings {tr.warnings}", flush=True)
    tr.warnings = []
print("ROUNDTRIP DONE", flush=True)
