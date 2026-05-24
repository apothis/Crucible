"""SoulX-Singer API server — runs ON the Windows GPU box INSIDE the SoulX-Singer
repo (use its python; copy this file to the repo root). Exposes the contract the
Mac's backend/voicegen.py expects:

  GET  /health       -> {"ok": true}
  POST /synthesize   (multipart) fields:
        score      : JSON (our melody score: bpm/key/duration/sections[].notes[])
        lyrics     : text (unused; lyrics already in score)
        opts       : JSON {control, pitch_shift, auto_shift, fp16, language}
        reference  : optional WAV (zero-shot target timbre; else bundled prompt)
     -> audio/wav (24 kHz) of the composed melody, sung.

Mirrors cli/inference.py: build_model() once, DataProcessor + model.infer() per
segment. We convert OUR per-syllable score into SoulX's per-WORD score-control
metadata (see RESEARCH.md §5c) with g2p_en for English phonemes.

Verified against the repo interface; CONFIRM on first run: note_type semantics,
VRAM on the 3090, English prompt quality, auto_shift/pitch_shift defaults.
Install via SOULX-API_AUTO_INSTALL.bat.
"""
import io
import json
import os

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, Response

from soulxsinger.utils.file_utils import load_config
from soulxsinger.utils.data_processor import DataProcessor
from cli.inference import build_model

ROOT = os.path.dirname(os.path.abspath(__file__))


def _p(env, default):
    v = os.environ.get(env, "")
    return v if v else os.path.join(ROOT, default)


MODEL_PATH = _p("MG_SOULX_MODEL", "pretrained_models/SoulX-Singer/model.pt")
CONFIG_PATH = _p("MG_SOULX_CONFIG", "soulxsinger/config/soulxsinger.yaml")
PHONESET = _p("MG_SOULX_PHONESET", "soulxsinger/utils/phoneme/phone_set.json")
PROMPT_WAV = _p("MG_SOULX_PROMPT_WAV", "example/audio/en_prompt.mp3")
PROMPT_META = _p("MG_SOULX_PROMPT_META", "example/audio/en_prompt.json")
PORT = int(os.environ.get("MG_SOULX_PORT", "5060"))
DEVICE = os.environ.get("MG_SOULX_DEVICE", "cuda")
# By default the model is loaded ON DEMAND and UNLOADED after each synth so it
# doesn't sit in VRAM next to ComfyUI/RVC (the 3090 is shared). Set
# MG_SOULX_KEEP_RESIDENT=1 to keep it loaded for faster repeated builds.
KEEP_RESIDENT = os.environ.get("MG_SOULX_KEEP_RESIDENT", "0") == "1"
VOICES_DIR = os.environ.get("MG_SOULX_VOICES_DIR", "") or os.path.join(ROOT, "voices")
os.makedirs(VOICES_DIR, exist_ok=True)
SR = 24000

print("Initializing SoulX-Singer server (model loads on first request)…")
CONFIG = load_config(CONFIG_PATH)
PROC = DataProcessor(hop_size=CONFIG.audio.hop_size, sample_rate=CONFIG.audio.sample_rate,
                     phoneset_path=PHONESET, device=DEVICE)  # light; no VRAM until used
from g2p_en import G2p  # noqa: E402
_G2P = G2p()
MODEL = None
MODEL_FP16 = None
print(f"SoulX-Singer server ready (keep_resident={KEEP_RESIDENT}).")


def _get_model(use_fp16=True):
    """Load the model (once) at the requested precision. Reloads if the cached
    model's precision differs — so fp32 vs fp16 can be chosen per request."""
    global MODEL, MODEL_FP16
    if MODEL is not None and MODEL_FP16 != use_fp16:
        del MODEL
        MODEL = None
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    if MODEL is None:
        print(f"Loading SoulX-Singer model into VRAM (fp16={use_fp16})…")
        MODEL = build_model(MODEL_PATH, CONFIG, device=DEVICE, use_fp16=use_fp16)
        MODEL_FP16 = use_fp16
    return MODEL


def _unload():
    global MODEL
    if MODEL is not None and not KEEP_RESIDENT:
        del MODEL
        MODEL = None
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        print("Unloaded SoulX-Singer model (freed VRAM).")


# ---- reference-voice library (zero-shot prompts) ----
_PREP = None


def _voice_paths(name):
    base = os.path.join(VOICES_DIR, os.path.basename(name))
    wav = next((base + e for e in (".wav", ".mp3", ".flac") if os.path.exists(base + e)), None)
    return wav, base + ".json"


def _list_voices():
    out = []
    for f in sorted(os.listdir(VOICES_DIR)):
        if f.lower().endswith((".wav", ".mp3", ".flac")):
            name = os.path.splitext(f)[0]
            _, j = _voice_paths(name)
            out.append({"name": name, "ready": os.path.exists(j)})
    return out


def _get_prep(language):
    global _PREP
    if _PREP is None:
        print("Loading SoulX preprocess pipeline (separation/ASR/note transcription)…")
        from preprocess.pipeline import PreprocessPipeline
        _PREP = PreprocessPipeline(device=DEVICE, language=language,
                                   save_dir=os.path.join(VOICES_DIR, "_work"),
                                   vocal_sep=True, midi_transcribe=True)
    return _PREP


def _unload_prep():
    global _PREP
    if _PREP is not None and not KEEP_RESIDENT:
        del _PREP
        _PREP = None
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        print("Unloaded preprocess pipeline (freed VRAM).")


app = FastAPI(title="SoulX-Singer API")


def _arpabet(word: str):
    phones = [p for p in _G2P(word) if p and p[0].isalpha()]
    return phones or ["AH0"]


def _median(xs):
    s = sorted(xs)
    return s[len(s) // 2]


def _word_groups(section):
    """Collapse our per-syllable notes into per-word notes (SoulX sings one note
    per word). Pitch = MEDIAN of the word's syllable pitches (more representative
    than the first syllable when a word's notes move)."""
    groups = []
    for n in section.get("notes", []):
        wi = n.get("word_idx")
        if groups and groups[-1]["wi"] == wi:
            g = groups[-1]
            g["end"] = max(g["end"], n["start"] + n["dur"])
            g["midis"].append(n["midi"])
        else:
            groups.append({"wi": wi, "word": n.get("word") or n.get("syllable", ""),
                           "start": n["start"], "end": n["start"] + n["dur"],
                           "midis": [n["midi"]]})
    for g in groups:
        g["midi"] = _median(g["midis"])
    return groups


def _segment(section, language="English"):
    """Build one SoulX score-control target segment from a Song section."""
    groups = _word_groups(section)
    if not groups:
        return None
    s0, s1 = section["start"], section["start"] + section["seconds"]
    ph, dur, pitch, ntype, text = [], [], [], [], []

    def add(p, d, mp, t, tx):
        if d <= 0:
            return
        ph.append(p); dur.append(round(d, 4)); pitch.append(mp); ntype.append(t); text.append(tx)

    cursor = s0
    for g in groups:
        add("<SP>", g["start"] - cursor, 0, 1, "<SP>")      # rest before the word
        add("en_" + "-".join(_arpabet(g["word"])), g["end"] - g["start"], int(g["midi"]), 2, g["word"])
        cursor = g["end"]
    add("<SP>", s1 - cursor, 0, 1, "<SP>")                   # trailing rest
    if not pitch:
        return None
    return {
        "index": f"sec_{section.get('role','seg')}_{int(s0*1000)}",
        "language": language,
        "time": [int(s0 * 1000), int(s1 * 1000)],
        "duration": " ".join(str(x) for x in dur),
        "text": " ".join(text),
        "phoneme": " ".join(ph),
        "note_pitch": " ".join(str(x) for x in pitch),
        "note_type": " ".join(str(x) for x in ntype),
    }


@app.get("/health")
def health():
    return {"ok": True, "engine": "soulx", "sr": SR,
            "loaded": MODEL is not None, "keep_resident": KEEP_RESIDENT}


@app.get("/voices")
def voices():
    return {"voices": _list_voices()}


@app.post("/voices/prep")
async def voices_prep(name: str = Form(...), language: str = Form("English"),
                      vocal_sep: bool = Form(True), file: UploadFile = File(...)):
    """Register a reference voice: save the clip, run the preprocess pipeline to
    transcribe it into SoulX prompt metadata (cached as <name>.json). Heavy
    (separation + ASR + note transcription) — done once per voice."""
    safe = os.path.basename(name).replace(" ", "_")
    ext = os.path.splitext(file.filename or "")[1].lower() or ".wav"
    clip = os.path.join(VOICES_DIR, safe + ext)
    with open(clip, "wb") as f:
        f.write(await file.read())
    try:
        prep = _get_prep(language)
        prep.run(clip, vocal_sep=vocal_sep, language=language)  # writes <clip>.json
        produced = os.path.splitext(clip)[0] + ".json"
        if not os.path.exists(produced):
            raise RuntimeError("preprocess produced no metadata")
    except Exception as e:
        import traceback
        return JSONResponse({"error": f"{type(e).__name__}: {e}",
                             "trace": traceback.format_exc().splitlines()[-6:]}, status_code=500)
    finally:
        _unload_prep()
    return {"ok": True, "name": safe}


@app.post("/synthesize")
async def synthesize(score: str = Form(...), lyrics: str = Form(""),
                     opts: str = Form("{}"), reference: UploadFile = File(None)):
    try:
        return await _synthesize(score, lyrics, opts, reference)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        return JSONResponse({"error": f"{type(e).__name__}: {e}", "trace": tb.splitlines()[-6:]},
                            status_code=500)


async def _synthesize(score, lyrics, opts, reference):
    sc = json.loads(score)
    o = json.loads(opts or "{}")
    control = o.get("control", "score")
    language = o.get("language", "English")

    # reference voice priority: a prepared named voice (opts.ref_voice) > a
    # posted clip > the bundled English prompt.
    prompt_wav, prompt_meta_path = PROMPT_WAV, PROMPT_META
    tmp_ref = None
    ref_voice = o.get("ref_voice")
    if ref_voice:
        w, j = _voice_paths(ref_voice)
        if not (w and os.path.exists(j)):
            return JSONResponse({"error": f"voice '{ref_voice}' is not prepared (POST /voices/prep first)"}, status_code=400)
        prompt_wav, prompt_meta_path = w, j
    elif reference is not None:
        tmp_ref = os.path.join(ROOT, "_mg_ref.wav")
        with open(tmp_ref, "wb") as f:
            f.write(await reference.read())
        prompt_wav = tmp_ref  # uses bundled metadata; prefer a prepared voice

    segments = [s for s in (_segment(sec, language) for sec in sc.get("sections", [])) if s]
    if not segments:
        return JSONResponse({"error": "no singable sections in score"}, status_code=400)

    with open(prompt_meta_path, "r", encoding="utf-8") as f:
        prompt_meta = json.load(f)[0]

    fp16 = bool(o.get("fp16", True))
    n_steps = int(o.get("n_steps", CONFIG.infer.n_steps))
    cfg = float(o.get("cfg", CONFIG.infer.cfg))
    try:
        model = _get_model(fp16)
        prompt_data = PROC.process(prompt_meta, prompt_wav)
        total = int(segments[-1]["time"][1] / 1000 * SR)
        out = np.zeros(total, dtype=np.float32)
        for seg in segments:
            start = int(seg["time"][0] / 1000 * SR)
            target_data = PROC.process(dict(seg), None)
            with torch.no_grad():
                audio = model.infer({"prompt": prompt_data, "target": target_data},
                                    auto_shift=o.get("auto_shift", True),
                                    pitch_shift=int(o.get("pitch_shift", 0)),
                                    n_steps=n_steps, cfg=cfg,
                                    control=control, use_fp16=fp16)
            audio = audio.squeeze().cpu().numpy()
            n = min(audio.shape[0], total - start)
            if n > 0:
                out[start:start + n] = audio[:n]
    finally:
        _unload()
        if tmp_ref and os.path.exists(tmp_ref):
            os.remove(tmp_ref)

    buf = io.BytesIO()
    sf.write(buf, out, SR, format="WAV", subtype="PCM_16")
    return Response(content=buf.getvalue(), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
