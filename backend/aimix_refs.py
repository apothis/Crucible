"""AI Mix style presets from reference songs (docs/AI_MIX_PLAN.md, adaptive rules).

A preset is a distribution of measurements over many reference songs of one style: for each
song, the middle EXCERPT_S seconds are split on the box (BS-RoFormer SW, 6 stems), measured with
aimix_features, and only the numbers are kept (stems are discarded, nothing enters the library).

Per-song results are cached as JSON next to the preset, so a stopped build resumes where it left
off and a song is never separated twice.
"""
import io
import json
import os
import subprocess
import threading
import time

import numpy as np

from . import aimix_features as F
from . import roformer_py

EXCERPT_S = 120.0
STATUS = {}                 # preset name -> {state, done, total, current, errors, started}
_LOCK = threading.Lock()


def preset_dir(root):
    d = os.path.join(root, "presets")
    os.makedirs(d, exist_ok=True)
    return d


def list_presets(root):
    d = preset_dir(root)
    out = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".json") and not f.startswith("_"):
            with open(os.path.join(d, f)) as fh:
                p = json.load(fh)
            out.append({"name": p["name"], "label": p.get("label", p["name"]), "n": len(p.get("songs", []))})
    return out


def load_preset(root, name):
    with open(os.path.join(preset_dir(root), f"{name}.json")) as f:
        return json.load(f)


def _excerpt_wav(path):
    """Middle EXCERPT_S seconds of any audio file -> 44.1 kHz stereo 16-bit WAV bytes (ffmpeg)."""
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                               capture_output=True, text=True, check=True).stdout.strip() or 0)
    start = max(0.0, dur / 2 - EXCERPT_S / 2)
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{start:.2f}", "-t", f"{EXCERPT_S:.2f}",
                        "-i", path, "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", "-f", "wav", "-"],
                       capture_output=True, check=True)
    return r.stdout, start, dur


def _wav_arrays(stem_bytes):
    import soundfile as sf
    out, sr = {}, None
    for name, b in stem_bytes.items():
        x, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
        if x.shape[1] == 1:
            x = np.repeat(x, 2, axis=1)
        out[name] = x[:, :2]
    return out, sr


def measure_reference(path, host):
    """Excerpt -> box split -> features. Returns the song's feature dict."""
    data, start, dur = _excerpt_wav(path)
    res = roformer_py.separate(data, "excerpt.wav", host, stems="all")
    arrs, sr = _wav_arrays(res["stems"])
    stems = {k: v for k, v in arrs.items() if k in F.ROLES}      # drop the extra 'instrumental'
    f = F.song_features(stems, sr)
    f["source"] = {"file": os.path.basename(path), "excerpt_start_s": round(start, 1), "duration_s": round(dur, 1)}
    return f


def build(root, name, label, files, host, free_gpu=None):
    """Blocking build (run it in a thread). Writes presets/<name>.json when done; per-song cache in
    presets/_cache_<name>/."""
    cache = os.path.join(preset_dir(root), f"_cache_{name}")
    os.makedirs(cache, exist_ok=True)
    st = {"state": "running", "done": 0, "total": len(files), "current": "", "errors": [], "started": time.time()}
    with _LOCK:
        STATUS[name] = st
    songs = []
    for i, path in enumerate(files):
        key = os.path.splitext(os.path.basename(path))[0]
        cp = os.path.join(cache, f"{key}.json")
        st["current"] = key
        try:
            if os.path.exists(cp):
                with open(cp) as fh:
                    f = json.load(fh)
            else:
                if free_gpu:
                    try:
                        free_gpu("roformer")
                    except Exception:
                        pass
                f = measure_reference(path, host)
                with open(cp, "w") as fh:
                    json.dump(f, fh)
            songs.append(f)
        except Exception as e:
            st["errors"].append(f"{key}: {e}")
        st["done"] = i + 1
    if songs:
        preset = {"name": name, "label": label, "built": time.time(), "excerpt_s": EXCERPT_S,
                  "songs": [s["source"] for s in songs], "stats": F.summarize(songs)}
        with open(os.path.join(preset_dir(root), f"{name}.json"), "w") as fh:
            json.dump(preset, fh, indent=1)
    st["state"] = "done" if songs else "failed"
    st["current"] = ""
    return st


def resummarize(root, name):
    """Rebuild a preset's stats from its per-song cache (after a measurement-rule change),
    without touching the box."""
    p = load_preset(root, name)
    cache = os.path.join(preset_dir(root), f"_cache_{name}")
    songs = []
    for f in sorted(os.listdir(cache)):
        if f.endswith(".json"):
            with open(os.path.join(cache, f)) as fh:
                songs.append(json.load(fh))
    p["stats"] = F.summarize(songs)
    p["songs"] = [s["source"] for s in songs]
    with open(os.path.join(preset_dir(root), f"{name}.json"), "w") as fh:
        json.dump(p, fh, indent=1)
    return p


def start_build(root, name, label, files, host, free_gpu=None):
    with _LOCK:
        if STATUS.get(name, {}).get("state") == "running":
            raise RuntimeError(f"preset '{name}' is already building")
    t = threading.Thread(target=build, args=(root, name, label, files, host, free_gpu), daemon=True)
    t.start()
