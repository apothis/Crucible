"""MusicGen app backend — drives ComfyUI (ACE-Step 1.5) for a thin web UI.

Run:  python -m backend.app   (from the repo root)
"""
import hashlib
import json
import math
import os
import random
import re
import shutil
import sqlite3
import tempfile
import threading
import time
import uuid

import requests
import websocket  # websocket-client
from typing import Any, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import comfy
from . import music3 as music3_mod
from . import video as video_mod
from . import musicvideo as musicvideo_mod
from . import lyricalign as lyricalign_mod
from . import rvc as rvc_mod
from . import rvc_py
from . import roformer_py
from . import acestep_py
from . import lora_runtime
from . import asr as asr_mod
from . import voices as voices_mod
from . import stems as stems_mod
from . import mix as mix_mod
from . import postfx as postfx_mod
from . import master as master_mod
from . import guitar as guitar_mod
from . import sections as sections_mod
from . import analyze as analyze_mod
from . import analyze_py
from . import genres as genres_mod
from . import llm as llm_mod
from . import lyrics as lyrics_mod
from . import melody as melody_mod
from . import solo as solo_mod
from . import kontakt_daemon as kontakt_daemon_mod
from . import wordmark as wordmark_mod
from . import deglitch as deglitch_mod
from . import shape as shape_mod
from . import acestep_train as ace_train
from . import lora_upload_py as lora_up
from . import lora_dataset as lora_ds
from . import lora_eval as lora_eval_mod
from . import metric_validate as metric_val
from . import embed_mert as embed_mert_mod
from . import voicegen as voicegen_mod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WEB_DIST = os.path.join(ROOT, "web", "dist")
FRONTEND = _WEB_DIST if os.path.isdir(_WEB_DIST) else os.path.join(ROOT, "frontend")
LIBRARY = os.path.join(ROOT, "library")
# Whisper transcripts behind the lyric alignment. Derived data, keyed by track + model
# size, so it lives in the scratch tree rather than the library.
LYRIC_CACHE = os.path.join(ROOT, ".mvwork", "lyricalign")
STEMS_DIR = os.path.join(LIBRARY, "stems")
# Curated "gold standard" reference masters for the Master tool's gold mode — the user
# drops a few well-mastered tracks they own here (named by vibe/genre); gitignored.
MASTER_REFS = os.path.join(LIBRARY, "master_refs")
DB = os.path.join(LIBRARY, "library.db")
os.makedirs(LIBRARY, exist_ok=True)
os.makedirs(STEMS_DIR, exist_ok=True)
os.makedirs(MASTER_REFS, exist_ok=True)

_CFG_PATH = os.path.join(ROOT, "app_config.json")
if not os.path.exists(_CFG_PATH):  # fresh clone: fall back to the committed example
    _CFG_PATH = os.path.join(ROOT, "app_config.example.json")
CFG = json.load(open(_CFG_PATH))
HOST = CFG["comfy_host"]
CLIENT_ID = "musicgen-app"
C = comfy.Comfy(HOST)
ROFORMER_HOST = CFG.get("roformer_host", "")
ACESTEP_HOST = CFG.get("acestep_host", "")   # official ACE-Step engine (cover etc.); empty = use ComfyUI
ANALYZE_HOST = CFG.get("analyze_host", "")   # box analyze service (allin1+CLAP); empty = Mac librosa (P1)
LORA_UPLOAD_HOST = CFG.get("lora_upload_host", "")   # box LoRA dataset upload helper (:5080); empty = disabled
# DCW (Differential Correction in Wavelet domain) ships ON-by-default in the engine and
# garbles XL text2music (full from-noise trajectory); it CANNOT be disabled over the HTTP
# API (see HANDOFF). Until DCW is patched off on the box, gate engine text2music on the XL
# models and fall back to ComfyUI so Generate never silently emits garbage. Turbo (short
# trajectory) and cover/repaint (source-anchored) are unaffected. Flip to true once the box
# is patched + verified.
ACESTEP_DCW_OK = bool(CFG.get("acestep_dcw_ok", False))
# Repaint stays on the ComfyUI path by default: the official engine's repaint is weak for
# content regeneration (silence-seeds the region, skips the LM so no audio-code plan, no
# loudness match → quiet/weak regions; feeding external audio codes breaks alignment — all
# verified, see HANDOFF/RESEARCH). ComfyUI's edit guider runs the LM codes in-graph and was
# user-confirmed good. Flip to true only if the engine's repaint improves upstream.
ACESTEP_REPAINT = bool(CFG.get("acestep_repaint", False))
# Add-a-Layer (lego) on the official engine — off by default while we validate it. Lego
# KEEPS the LM on (unlike repaint), so the new part gets an audio-code plan; the engine
# auto-builds the instruction from track_name and returns a full mix with the part baked
# in (duration locks to source, BPM-locked). base/SFT only. Flip true to test the engine path.
ACESTEP_LEGO = bool(CFG.get("acestep_lego", False))


def make_rvc():
    """Choose the RVC driver. 'auto' prefers the clean rvc-python API server,
    falling back to the legacy Gradio WebUI driver if it isn't reachable."""
    driver = CFG.get("rvc_driver", "auto")
    gradio = rvc_mod.RVC(CFG.get("rvc_host", ""), C, CFG.get("comfy_input_dir", ""))
    pyapi = rvc_py.RVCPython(CFG.get("rvc_python_host", ""))
    if driver == "gradio":
        return gradio, "gradio"
    if driver == "rvc_python":
        return pyapi, "rvc_python"
    return (pyapi, "rvc_python") if pyapi.reachable() else (gradio, "gradio")


R, RVC_DRIVER = make_rvc()


def free_gpu(keep=""):
    """Evict every OTHER box GPU service before running `keep`, so the shared 3090
    isn't double-booked. ComfyUI + RVC expose /free; SoulX & RoFormer self-unload
    after each run; ACE-Step runs CPU-offloaded (low idle VRAM, no unload API)."""
    if keep != "comfy":
        try:
            C.free()
        except Exception:
            pass
    if keep != "rvc":
        host = CFG.get("rvc_python_host", "")
        if host:
            try:
                requests.post(f"http://{host}/free", timeout=15)
            except Exception:
                pass


def submit_comfy(graph):
    """Free the other GPU services, then submit a graph to ComfyUI."""
    free_gpu("comfy")
    return C.submit(graph, CLIENT_ID)


# in-memory live job state keyed by prompt_id
JOBS = {}
LOCK = threading.Lock()


# ---------------- SQLite library ----------------
def db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS jobs(
            id TEXT PRIMARY KEY, created REAL, mode TEXT, params TEXT,
            audio TEXT, status TEXT, error TEXT)""")
        cols = [r[1] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()]
        if "bucket" not in cols:  # filing bucket: "" (auto by mode) or "tests"
            conn.execute("ALTER TABLE jobs ADD COLUMN bucket TEXT DEFAULT ''")
        # A "project" = one saved bundle of per-page inputs/settings (the web draft store)
        # plus the song arrangement and current tab. `data` is the JSON the UI serializes.
        conn.execute("""CREATE TABLE IF NOT EXISTS projects(
            id TEXT PRIMARY KEY, name TEXT, created REAL, updated REAL, data TEXT)""")
        # A reusable CHARACTER (cast member), global across projects. `data` JSON holds
        # role, ref_still_ids, lora_name, method (anchor/qwen/vace/lora), notes.
        conn.execute("""CREATE TABLE IF NOT EXISTS characters(
            id TEXT PRIMARY KEY, name TEXT, created REAL, updated REAL, data TEXT)""")


def _next_take(mode, title, conn):
    """Stable per-song take number: max existing 'take' among same-mode rows sharing the
    title (case-insensitive), + 1. Numbers are assigned once at first save and NEVER
    reused, so deleting take 1 leaves 2..N labeled 2..N — positional renumbering made
    library version management impossible (deletes appeared to hit the wrong take)."""
    t = title.strip().lower()
    mx = 0
    for r in conn.execute("SELECT params FROM jobs WHERE mode=?", (mode,)):
        try:
            p = json.loads(r["params"] or "{}")
        except Exception:
            continue
        if str(p.get("title") or "").strip().lower() == t:
            try:
                mx = max(mx, int(p.get("take") or 0))
            except (TypeError, ValueError):
                pass
    return mx + 1


def save_job(pid):
    j = JOBS.get(pid)
    if not j:
        return
    with db() as conn:
        p = j["params"]
        title = str(p.get("title") or "").strip()
        # Numbered on COMPLETION, not submission: failed takes never render in the
        # library, so letting them consume numbers would show gaps for no visible reason.
        if title and j["status"] == "done" and not p.get("take"):
            p["take"] = _next_take(j["mode"], title, conn)
        conn.execute(
            "REPLACE INTO jobs(id,created,mode,params,audio,status,error) VALUES(?,?,?,?,?,?,?)",
            (pid, j["created"], j["mode"], json.dumps(j["params"]),
             j.get("audio_file"), j["status"], j.get("error")))


def _keep_lossless(jid, *src_paths):
    """Finish-chain output rule: flac in, flac out. The processing endpoints all write a lossless
    {jid}.wav; when every source was itself lossless AND at least one was .flac, that wav is
    transcoded (bit-exact) to {jid}.flac so a Music 3 master stays lossless-and-compact through
    deglitch/master/shape/tone/mix. Every other flow keeps its .wav exactly as before - changing
    the default output for existing mp3/wav chains was deliberately avoided."""
    import subprocess
    wav = os.path.join(LIBRARY, f"{jid}.wav")
    exts = [os.path.splitext(sp or "")[1].lower() for sp in src_paths if sp]
    if not exts or ".flac" not in exts or any(e not in (".flac", ".wav") for e in exts):
        return wav
    fl = os.path.join(LIBRARY, f"{jid}.flac")
    try:
        r = subprocess.run(["ffmpeg", "-y", "-i", wav, "-c:a", "flac", fl],
                           capture_output=True, timeout=600)
        if r.returncode == 0 and os.path.exists(fl) and os.path.getsize(fl) > 0:
            os.remove(wav)
            return fl
    except Exception:
        pass
    try:
        os.remove(fl)
    except Exception:
        pass
    return wav


def save_done_row(jid, mode, params, audio_path, bucket=""):
    with db() as conn:
        title = str(params.get("title") or "").strip()
        if title and not params.get("take"):
            params["take"] = _next_take(mode, title, conn)
        conn.execute(
            "REPLACE INTO jobs(id,created,mode,params,audio,status,error,bucket) VALUES(?,?,?,?,?,?,?,?)",
            (jid, time.time(), mode, json.dumps(params), audio_path, "done", None, bucket))


# ---------------- WebSocket progress listener ----------------
VIDEO_MODES = {"videostill", "videoclip", "videolipsync", "musicvideo", "ytvideo"}   # produce image/mp4, not audio
MEDIA_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".webm", ".mkv")
_MEDIA_CT = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
             ".webp": "image/webp", ".gif": "image/gif", ".mp4": "video/mp4",
             ".webm": "video/webm", ".mkv": "video/x-matroska"}


def _write_media_retry(path, data, attempts=5):
    """Write a finished render to the library, retrying transient OS errors. An external SSD enclosure
    intermittently throws EPERM ('Operation not permitted') on writes; a single attempt silently lost
    renders (and left no file/DB row). Write to a .part temp then atomically rename, and retry with
    backoff so a transient failure doesn't drop the render."""
    last = None
    tmp = path + ".part"
    for i in range(attempts):
        try:
            with open(tmp, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)            # atomic: never leaves a half-written final file
            return
        except OSError as e:
            last = e
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
            time.sleep(0.3 * (i + 1))
    raise last


def on_complete_media(pid):
    """Fetch history, download the produced image/video into the library. ComfyUI
    reports SaveImage under 'images' and SaveVideo/VHS under 'images'/'gifs'/'video',
    so we scan every output list for a dict with a media-extension filename."""
    try:
        h = C.history(pid)
        if pid not in h:
            return
        for _, out in h[pid].get("outputs", {}).items():
            for key, items in out.items():
                if not isinstance(items, list):
                    continue
                for it in items:
                    if not isinstance(it, dict) or "filename" not in it:
                        continue
                    fn = it["filename"]
                    ext = os.path.splitext(fn)[1].lower()
                    if ext not in MEDIA_EXTS:
                        continue
                    data = C.view_bytes(fn, it.get("subfolder", ""), it.get("type", "output"))
                    path = os.path.join(LIBRARY, f"{pid}{ext}")
                    with open(path, "wb") as f:
                        f.write(data)
                    with LOCK:
                        JOBS[pid]["audio_file"] = path   # reuse the field; it is just a path
                        JOBS[pid]["status"] = "done"
                    save_job(pid)
                    return
    except Exception as e:
        with LOCK:
            JOBS[pid]["status"] = "error"
            JOBS[pid]["error"] = f"download failed: {e}"
        save_job(pid)


def on_complete(pid):
    """Fetch history, download the produced audio into the library."""
    if JOBS.get(pid, {}).get("mode") in VIDEO_MODES:
        return on_complete_media(pid)
    try:
        h = C.history(pid)
        if pid not in h:
            return
        for _, out in h[pid].get("outputs", {}).items():
            for a in out.get("audio", []):
                data = C.view_bytes(a["filename"], a.get("subfolder", ""), a.get("type", "output"))
                # Keep the format the graph actually produced. Every ACE graph saves MP3, so this
                # is a no-op for them, but Music 3 saves lossless FLAC and hardcoding .mp3 here
                # wrote FLAC bytes into a .mp3 name - which then plays, and silently lies about
                # what it is. /api/export?fmt=mp3 does the 320k conversion on demand instead.
                ext = os.path.splitext(a["filename"])[1].lower() or ".mp3"
                if ext not in (".mp3", ".wav", ".flac", ".opus"):
                    ext = ".mp3"
                path = os.path.join(LIBRARY, f"{pid}{ext}")
                with open(path, "wb") as f:
                    f.write(data)
                # ACE-specific end-burst/clipping fix. Never run it on another engine's output:
                # Music 3 ends its own way and this would be a silent, unasked-for edit.
                if JOBS.get(pid, {}).get("mode") != "music3":
                    try:                              # auto-fix ACE end-burst/clipping (only when needed)
                        fixed = postfx_mod.tidy_ending(path)
                        if fixed and fixed != path:
                            os.remove(path)
                            path = fixed
                    except Exception:
                        pass
                if JOBS.get(pid, {}).get("mode") == "extend":   # close the model's brief seam silence
                    try:
                        postfx_mod.close_seam_gap(path)
                    except Exception:
                        pass
                jp = JOBS.get(pid, {}).get("params", {})
                if JOBS.get(pid, {}).get("mode") == "layerstem" and jp.get("gate"):
                    # Reduce the extracted stem to just the layer's time window.
                    try:
                        import soundfile as _sf
                        wav = os.path.splitext(path)[0] + ".wav"
                        if path != wav:                       # transcode mp3 → wav so gate can rewrite it
                            d, sr = _sf.read(path, always_2d=True)
                            _sf.write(wav, d, sr, subtype="PCM_16")
                            os.remove(path); path = wav
                        postfx_mod.gate_region(path, jp.get("layer_start", 0), jp.get("layer_end", 0))
                    except Exception:
                        pass
                with LOCK:
                    JOBS[pid]["audio_file"] = path
                    JOBS[pid]["status"] = "done"
                save_job(pid)
                return
    except Exception as e:
        with LOCK:
            JOBS[pid]["status"] = "error"
            JOBS[pid]["error"] = f"download failed: {e}"
        save_job(pid)


def handle_msg(d):
    t = d.get("type")
    data = d.get("data", {})
    pid = data.get("prompt_id")
    if t == "progress":
        # progress events may omit prompt_id; apply to the running job in that case
        with LOCK:
            target = pid if pid in JOBS else next(
                (k for k, v in JOBS.items() if v["status"] == "running"), None)
            if target:
                JOBS[target]["progress"] = data.get("value", 0)
                JOBS[target]["max"] = data.get("max", 0)
    elif t == "executing":
        if pid in JOBS:
            with LOCK:
                if data.get("node") is None:
                    if JOBS[pid]["status"] == "running":
                        JOBS[pid]["status"] = "finalizing"
                else:
                    JOBS[pid]["status"] = "running"
            if data.get("node") is None and JOBS[pid]["status"] == "finalizing":
                on_complete(pid)
    elif t == "execution_error" and pid in JOBS:
        with LOCK:
            JOBS[pid]["status"] = "error"
            JOBS[pid]["error"] = data.get("exception_message", "execution error")
        save_job(pid)


def ws_loop():
    url = f"ws://{HOST}/ws?clientId={CLIENT_ID}"
    while True:
        try:
            ws = websocket.create_connection(url, timeout=10)
            ws.settimeout(None)   # block on recv; video jobs go >10s silent during VAE
            # decode + mp4 encode, and a recv-timeout reconnect would MISS the terminal
            # "executing null" completion message (job stuck at running). reconcile_loop
            # is the safety net regardless.
            while True:
                msg = ws.recv()
                if isinstance(msg, (bytes, bytearray)):
                    continue  # binary preview frames
                handle_msg(json.loads(msg))
        except Exception:
            time.sleep(2)


def reconcile_video_job(pid, mode, params):
    """Finalize a video job from ComfyUI /history (independent of the live WS). Pulls the
    produced still/clip into the library. Safe to call repeatedly; idempotent."""
    try:
        entry = C.history(pid).get(pid)
    except Exception:
        return False
    if not entry:
        return False   # not in history yet => still rendering/queued
    if entry.get("status", {}).get("status_str") == "error":
        with LOCK:
            if pid in JOBS:
                JOBS[pid]["status"] = "error"
                JOBS[pid]["error"] = "comfy execution error"
        return True
    for out in entry.get("outputs", {}).values():
        for items in out.values():
            if not isinstance(items, list):
                continue
            for it in items:
                if not isinstance(it, dict) or "filename" not in it:
                    continue
                ext = os.path.splitext(it["filename"])[1].lower()
                if ext not in MEDIA_EXTS:
                    continue
                path = os.path.join(LIBRARY, f"{pid}{ext}")
                if not os.path.exists(path):
                    data = C.view_bytes(it["filename"], it.get("subfolder", ""), it.get("type", "output"))
                    with open(path, "wb") as f:
                        f.write(data)
                with LOCK:
                    if pid in JOBS:
                        JOBS[pid]["audio_file"] = path
                        JOBS[pid]["status"] = "done"
                save_done_row(pid, mode, params, path)
                return True
    return False


def _unfinished_video_jobs():
    """{pid: (mode, params)} for video jobs not yet done/errored — from live memory AND
    the DB (so a finished-but-stuck job recovers even across a backend restart)."""
    out = {}
    with LOCK:
        for pid, j in JOBS.items():
            if j.get("mode") in VIDEO_MODES and j.get("status") in ("pending", "running", "finalizing"):
                out[pid] = (j["mode"], j.get("params", {}))
    try:
        with db() as conn:
            rows = conn.execute(
                "SELECT id, mode, params FROM jobs WHERE mode IN ('videostill','videoclip','videolipsync') "
                "AND status NOT IN ('done','error')").fetchall()
        for r in rows:
            out.setdefault(r["id"], (r["mode"], json.loads(r["params"] or "{}")))
    except Exception:
        pass
    return out


def reconcile_loop():
    """Safety net: poll ComfyUI history for video jobs the WS may have missed (long silent
    VAE/encode tails) and finalize them into the library."""
    while True:
        time.sleep(5)
        try:
            for pid, (mode, params) in _unfinished_video_jobs().items():
                reconcile_video_job(pid, mode, params)
        except Exception:
            pass


# ---------------- API ----------------
app = FastAPI(title="MusicGen")


@app.on_event("startup")
def startup():
    init_db()
    threading.Thread(target=ws_loop, daemon=True).start()
    threading.Thread(target=reconcile_loop, daemon=True).start()


@app.get("/api/config")
def config():
    # unified genre registry (single source) → drives the preset chips + the
    # Guitar riff/solo genre pickers. bpm/key are SUGGESTIONS, not forced.
    genres = [{"id": g["id"], "label": g["label"], "tags": g["tags"],
               "bpm": g["bpm"], "key": g["key"], "scale": g["scale"],
               "lead": bool(g.get("lead")), "parent": g.get("parent")} for g in genres_mod.GENRES]
    return {"comfy_host": HOST, "variants": C.available_variants(),
            "keys": comfy.KEYS, "languages": ["en"], "rvc_driver": RVC_DRIVER,
            "roformer": _roformer_available(),
            "acestep": bool(ACESTEP_HOST),
            "acestep_repaint": ACESTEP_REPAINT,
            "acestep_lego": ACESTEP_LEGO,
            "analyze": analyze_mod.available() or bool(ANALYZE_HOST),
            "analyze_box": bool(ANALYZE_HOST),
            "lora_train": bool(ACESTEP_HOST),       # train/use LoRAs on the official engine
            "lora_upload": bool(LORA_UPLOAD_HOST),  # box dataset upload helper present
            "video": _video_available(),            # Wan2.2 + Z-Image video pipeline models present
            "video_qwen": _qwen_available(),        # Qwen-Image-Edit GGUF present (char consistency)
            "video_vace": _vace_available(),        # Wan VACE GGUF present (reference-to-video)
            "video_ltx": _ltx_available(),          # LTX-2.3 GGUF present (fast video backbone)
            "video_ltx_quants": _ltx_quants_present(),  # which LTX quants are on the box
            "video_msr": _msr_available(),          # LTX MSR renderable (Licon-MSR + PromptRelay nodes)
            "video_h3": _h3_available(),            # MiniMax H3 video+audio backbone present
            "genres": genres}


def _h3_available():
    """True when the MiniMax H3 backbone is renderable: both conditioning nodes registered on the box.
    The model files are checked implicitly (UNETLoader would reject a missing name at submit)."""
    try:
        return C.has_node("MiniMaxH3ImageToVideo") and C.has_node("MiniMaxH3ReferenceToVideo")
    except Exception:
        return False


def _msr_available():
    """True when LTX MSR is renderable: the LTX backbone + the Licon-MSR + PromptRelay
    custom nodes are both registered on the box (the MV Studio spine)."""
    try:
        return _ltx_available() and C.has_node("LiconMSR") and C.has_node("LTXDirector")
    except Exception:
        return False


def _video_available():
    """True when the core video-pipeline models are on the ComfyUI box."""
    try:
        have = set(C.models("diffusion_models"))
        return video_mod.Z_IMAGE_UNET in have and video_mod.WAN_TI2V in have
    except Exception:
        return False


def _qwen_available():
    """True when the Qwen-Image-Edit GGUF (character consistency) is on the box."""
    try:
        return video_mod.QWEN_EDIT_GGUF in set(C.gguf_unets())
    except Exception:
        return False


def _vace_available():
    """True when the Wan VACE GGUF (reference-to-video) is on the box."""
    try:
        return video_mod.WAN_VACE_GGUF in set(C.gguf_unets())
    except Exception:
        return False


def _ltx_quants_present():
    """Which LTX-2.3 22B quants are actually on the box (any of Q4_K_S/Q5_K_S/Q6_K/Q8_0)."""
    try:
        have = set(C.gguf_unets())
        return [q for q in video_mod.LTX_QUANTS
                if video_mod.LTX_UNET_TMPL.format(quant=q) in have]
    except Exception:
        return []


def _ltx_available():
    """True when any LTX-2.3 GGUF quant (fast video backbone) is on the box."""
    return bool(_ltx_quants_present())


@app.get("/api/acestep/info")
def acestep_info():
    """Health + available models from the official ACE-Step engine (for verifying
    the box install). Returns reachable=False if acestep_host isn't set/up."""
    if not ACESTEP_HOST:
        return {"reachable": False, "reason": "acestep_host not set in app_config.json"}
    try:
        return {"reachable": True, "host": ACESTEP_HOST,
                "health": acestep_py.health(ACESTEP_HOST),
                "models": acestep_py.models(ACESTEP_HOST)}
    except Exception as e:
        return {"reachable": False, "host": ACESTEP_HOST, "error": str(e)}


def _new_job(resolved, mode):
    return {"created": time.time(), "mode": mode, "status": "pending",
            "progress": 0, "max": 0, "params": {k: v for k, v in resolved.items()
                                                 if not k.startswith("_")}}


@app.post("/api/generate")
def generate(p: dict):
    if not (p.get("tags") or "").strip():
        raise HTTPException(400, "style tags are required (an empty prompt produces noise)")

    # Song-Builder round-trip: when the request carries the song recipe, tag the row as a
    # `song` (not a bare generate), store its title + full recipe (arrangement/key/bpm/…)
    # so the card can show the name and re-open in the builder for a new version.
    song_meta = p.get("song_meta")
    title = (p.get("title") or "").strip()
    from_builder = bool(p.get("from_builder") or song_meta)
    gmode = "song" if from_builder else "generate"

    def _decorate(resolved):
        if from_builder:
            resolved["from_builder"] = True
        if title:
            resolved["title"] = title
        if song_meta:
            resolved["song_meta"] = song_meta
        return resolved

    eng_model = p.get("model") or "acestep-v15-xl-sft"
    eng_is_turbo = "turbo" in eng_model.lower()
    # Use the engine only when it won't hit the DCW garble bug: turbo is DCW-safe, and XL
    # models are allowed once the box's DCW default is patched off (acestep_dcw_ok).
    use_engine = bool(ACESTEP_HOST) and (eng_is_turbo or ACESTEP_DCW_OK)
    if use_engine:                                    # ----- official ACE-Step engine path -----
        seed = int(p.get("seed") or random.randint(1, 2**31 - 1))
        instrumental = bool(p.get("instrumental"))
        # cfg→guidance_scale (base/non-turbo only); steps→inference_steps; sampler→infer_method
        # (drop ComfyUI sampler/scheduler names + APG — the engine's cfg_interval is its high-cfg
        # control). negative_tags has NO engine equivalent (only lm_negative_prompt) → dropped.
        fields = {
            "task_type": "text2music",
            "prompt": p.get("tags", ""),
            "lyrics": "" if instrumental else p.get("lyrics", ""),
            "instrumental": instrumental,
            "duration": float(p.get("duration", 40.0)),
            "bpm": int(p.get("bpm", 120)),
            "keyscale": p.get("keyscale", "E minor"),
            "vocal_language": p.get("language", "en"),
            # Defaults = the documented best-quality recipe (A/B-verified): guidance 8,
            # 64 steps + ADG for base/sft (turbo wants ~8 steps, ignores guidance/ADG).
            "guidance_scale": float(p.get("cfg") if p.get("cfg") not in (None, "") else 8.0),
            "inference_steps": int(p.get("steps") or (8 if eng_is_turbo else 64)),
            "shift": float(p.get("shift", 3.0)),
            "infer_method": p.get("infer_method", "ode"),
            "cfg_interval_start": float(p.get("cfg_interval_start", 0.0)),
            "cfg_interval_end": float(p.get("cfg_interval_end", 0.95)),
            "use_adg": bool(p.get("use_adg", not eng_is_turbo)),  # Adaptive Dual Guidance (quality; base/sft only)
            "thinking": bool(p.get("thinking", True)),          # 4B LM audio codes (ComfyUI parity)
            "use_cot_caption": bool(p.get("use_cot_caption", True)),   # LM auto-expands the tags
            "use_cot_language": bool(p.get("use_cot_language", True)),
            "batch_size": 1,                                    # 1 take/request → matches the count loop
            "use_random_seed": False,
            "seed": seed,
            "audio_format": "wav",
            "model": p.get("model", "acestep-v15-xl-sft"),
        }
        try:
            free_gpu("ace")                            # free ComfyUI + RVC; keep ACE resident
        except Exception:
            pass
        _acestep_ensure_model(fields["model"])         # /release_task won't auto-load — swap if the picked model isn't the loaded one
        # Per-generation LoRA reconcile: when the request carries a `loras` list
        # (the LoRA picker always sends one, even empty), set the engine to
        # EXACTLY that adapter set + scales, verified, before submitting — so a
        # take provably uses only the selected adapters. Absent `loras` keeps the
        # legacy global behavior (the old toggle/scale control) unchanged.
        applied_loras = None
        loras_req = p.get("loras")
        if loras_req is not None:
            # Name each engine adapter slot after the picker's label (sanitized,
            # unique within the request) so the engine status AND the library
            # card show WHICH adapter ran, not a generic slot0.
            specs, seen = [], {}
            for i, l in enumerate(loras_req):
                label = (l.get("label") or "").strip()
                base = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")[:48] or f"slot{i}"
                if base in seen:
                    seen[base] += 1
                    name = f"{base}_{seen[base]}"
                else:
                    seen[base] = 0
                    name = base
                specs.append({"path": l.get("path"), "scale": float(l.get("scale", 1.0)),
                              "name": name, "label": label or name})
            try:
                lora_runtime.reconcile(ACESTEP_HOST, specs)
            except Exception as e:
                raise HTTPException(500, f"lora reconcile failed: {e}")
            # Record display-friendly + recoverable info on the job.
            applied_loras = [{"label": s["label"], "scale": s["scale"], "path": s["path"]} for s in specs]
        try:
            task_id = acestep_py.submit(ACESTEP_HOST, fields)
        except Exception as e:
            raise HTTPException(500, f"acestep submit failed: {e}")
        pid = uuid.uuid4().hex
        resolved = _decorate({"engine": "acestep", "tags": p.get("tags", ""), "seed": seed,
                    "model": fields["model"], "guidance_scale": fields["guidance_scale"],
                    "steps": fields["inference_steps"], "duration": fields["duration"]})
        if applied_loras is not None:
            resolved["loras"] = applied_loras       # record exactly what ran, for the library card
        with LOCK:
            JOBS[pid] = _new_job(resolved, gmode)
            JOBS[pid]["status"] = "running"
        save_job(pid)
        phases = 2 if fields.get("thinking") else 1     # LM 'thinking' stage + DiT = two ramps
        threading.Thread(target=_acestep_poll, args=(pid, task_id, gmode, phases), daemon=True).start()
        return {"job_id": pid, "seed": seed}

    # ----- ComfyUI text2music (fallback; also the safe path when the engine's DCW bug
    # would garble XL text2music — see ACESTEP_DCW_OK) -----
    try:
        graph, resolved = comfy.build_t2m(p)
        res = submit_comfy(graph)
    except Exception as e:
        raise HTTPException(500, f"submit failed: {e}")
    if res.get("node_errors"):
        raise HTTPException(400, f"node errors: {res['node_errors']}")
    pid = res["prompt_id"]
    with LOCK:
        JOBS[pid] = _new_job(_decorate(resolved), gmode)
    save_job(pid)
    return {"job_id": pid, "seed": resolved["seed"]}


# ---------------- Video pipeline (Phase 1 gate: still -> i2v -> lip-sync) ----------------
def _lib_image_path(pid):
    """Absolute path to a library item's image file (by id)."""
    pid = os.path.basename(pid or "")
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = os.path.join(LIBRARY, f"{pid}{ext}")
        if os.path.exists(p):
            return p
    return None


def _trim_audio_window(path, start, dur):
    """Return WAV bytes for [start, start+dur] of an audio file (mp3/wav). Mac CPU only."""
    import io
    import librosa
    import soundfile as sf
    y, sr = librosa.load(path, sr=None, mono=True, offset=max(0.0, start), duration=dur)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _audio_name(prefix, b):
    """Upload filename DERIVED FROM THE AUDIO CONTENT (md5). ComfyUI's /upload/image overwrites by
    filename and LoadAudio caches by filename, so a FIXED name makes batched renders clobber/cache-
    collide onto ONE audio clip (the per-shot audio_start is right, but every queued render reads the
    same file). Hashing the bytes makes the name unique per distinct audio (no collision) and identical
    per identical audio (clean cache). NEVER use a constant audio upload filename."""
    return f"{prefix}_{hashlib.md5(b).hexdigest()[:16]}.wav"


# Last model signature submitted to ComfyUI. We only force a model-free when the NEXT graph uses a
# DIFFERENT big model - back-to-back renders of the same model (we're now anchored on LTX) stay warm
# instead of reloading the 22B transformer (~170s) every shot. A manual free
# (POST /api/video/free_models) resets this so the next render reloads cleanly.
_LAST_MODEL_SIG = None


def _graph_model_sig(graph):
    """The set of model-loader filenames in a graph - identifies which big model(s) it loads."""
    sig = []
    for n in graph.values():
        if not isinstance(n, dict):
            continue
        if n.get("class_type") in ("UNETLoader", "UnetLoaderGGUF", "CheckpointLoaderSimple"):
            ins = n.get("inputs", {})
            sig.append(ins.get("unet_name") or ins.get("ckpt_name") or "")
    return tuple(sorted(s for s in sig if s))


def _submit_video(graph, resolved, mode, free=None):
    # ComfyUI keeps the last model resident, so consecutive renders of the SAME model are warm. Only
    # force a /free when the model CHANGES (free=None -> auto by model signature): avoids the ~170s
    # reload of the same 22B LTX transformer on every shot, while still clearing VRAM when we switch
    # model families (Qwen still-gen <-> LTX video) to dodge offload thrash / OOM. free=True/False
    # forces it; manual free: POST /api/video/free_models.
    global _LAST_MODEL_SIG
    sig = _graph_model_sig(graph)
    if free is None:
        free = (sig != _LAST_MODEL_SIG)
    if free:
        try:
            # NOTE: do NOT run an in-graph cleanup node (easy cleanGpuUsed / LevelPixel) before GGUF
            # jobs - forcing unload_all_models on a GGUF-patched model segfaults ComfyUI (access
            # violation in ComfyUI-GGUF unpatch_model -> torch .to, 2026-06-15). Plain /free is safe.
            C.free(unload_models=True, free_memory=True)
        except Exception:
            pass
    _LAST_MODEL_SIG = sig
    res = submit_comfy(graph)
    if res.get("node_errors"):
        raise HTTPException(400, f"node errors: {res['node_errors']}")
    pid = res["prompt_id"]
    with LOCK:
        JOBS[pid] = _new_job(resolved, mode)
    save_job(pid)
    return {"job_id": pid, "seed": resolved.get("seed"), "media_url": f"/api/media/{pid}"}


@app.api_route("/api/comfy/{path:path}", methods=["GET", "POST"])
async def comfy_proxy(path: str, request: Request):
    """Same-origin proxy to the box ComfyUI for the embedded LTXDirector timeline editor's file ops
    (/view, /upload/image, /ltx_director_*). The editor runs in our React app; hitting the box directly
    is cross-origin (CORS-blocked), so it calls /api/comfy/<path> and we forward to ComfyUI."""
    url = f"http://{HOST}/{path}"
    params = dict(request.query_params)
    try:
        if request.method == "GET":
            r = requests.get(url, params=params, timeout=120)
        else:
            body = await request.body()
            r = requests.post(url, params=params, data=body,
                              headers={"Content-Type": request.headers.get("content-type", "application/octet-stream")},
                              timeout=600)
    except Exception as e:
        raise HTTPException(502, f"comfy proxy failed: {e}")
    return Response(content=r.content, status_code=r.status_code,
                    media_type=r.headers.get("Content-Type"))


@app.post("/api/video/free_models")
def video_free_models():
    """Manually evict ComfyUI's resident GPU models and reset the auto-free tracker so the next render
    reloads cleanly. We no longer free on every render (back-to-back same-model shots stay warm) - hit
    this when switching model families or to reclaim VRAM on demand."""
    global _LAST_MODEL_SIG
    try:
        C.free(unload_models=True, free_memory=True)
    except Exception as e:
        raise HTTPException(500, f"free failed: {e}")
    _LAST_MODEL_SIG = None
    return {"freed": True}


@app.post("/api/video/still")
def video_still(p: dict):
    """Generate a photoreal still (Z-Image Turbo). p: {prompt, negative?, seed?, width?,
    height?, steps?, cfg?}. Krea2 also accepts `layout` (regional bbox prompting via the
    Ideogram4PromptBuilderKJ node) instead of a plain prompt."""
    if not (p.get("prompt") or "").strip() and not p.get("layout"):
        raise HTTPException(400, "a prompt (or a krea2 layout) is required")
    # Engine: explicit request wins; else the app_config default (still_engine), else Z-Image.
    if not (p.get("engine") or "").strip():
        p["engine"] = CFG.get("still_engine", "zimage")
    # Krea2 levers default from app_config unless the request set them. Both ship ON to match how
    # the AItrepreneur workflow is actually run (enhancer for prompt adherence/quality; seed-variance
    # for image variety). They need their custom nodes installed on the box (ComfyUI-Krea2T-Enhancer,
    # ComfyUI-RBG-SmartSeedVariance) - turn them off in Settings if the box doesn't have them.
    if (p.get("engine") or "").lower() == "krea2":
        if "enhancer" not in p:
            p["enhancer"] = bool(CFG.get("still_krea2_enhancer", True))
        if "seed_variance" not in p:
            p["seed_variance"] = bool(CFG.get("still_krea2_seed_variance", True))
    # init_still_id (krea2 only) = run the workflow's IMAGE TO IMAGE path over an existing library
    # still instead of generating from noise. img2img on ONE image at denoise 0.40: it re-renders
    # what it is given, so it polishes or restyles a still - it cannot place referenced people.
    init_name = None
    if p.get("init_still_id"):
        ip = _lib_image_path(p["init_still_id"])
        if not ip:
            raise HTTPException(400, "init_still_id must reference a generated still in the library")
        if (p.get("engine") or "").lower() != "krea2":
            raise HTTPException(400, "init_still_id is a krea2 path - pass engine='krea2'")
        with open(ip, "rb") as f:
            init_name = C.upload_audio(f.read(), os.path.basename(ip))
    try:
        graph, resolved = video_mod.build_still(p, init_image=init_name)
    except Exception as e:
        raise HTTPException(500, f"build failed: {e}")
    if p.get("init_still_id"):
        resolved["init_still_id"] = os.path.basename(str(p["init_still_id"]))
    return _submit_video(graph, resolved, "videostill")


@app.post("/api/video/i2v")
def video_i2v(p: dict):
    """Animate a library still (Wan2.2 5B TI2V). p: {still_id, prompt, ...}."""
    still = _lib_image_path(p.get("still_id"))
    if not still:
        raise HTTPException(400, "still_id must reference a generated still in the library")
    try:
        with open(still, "rb") as f:
            ref = C.upload_audio(f.read(), os.path.basename(still))   # /upload/image helper
        graph, resolved = video_mod.build_i2v(p, ref)
    except Exception as e:
        raise HTTPException(500, f"build failed: {e}")
    resolved["still_id"] = os.path.basename(p.get("still_id"))
    return _submit_video(graph, resolved, "videoclip")


@app.post("/api/video/lipsync")
def video_lipsync(p: dict):
    """Lip-sync a portrait still to a song (Wan2.2-S2V, single ~4.8s clip).
    p: {still_id, audio_id, prompt?, ...}."""
    still = _lib_image_path(p.get("still_id"))
    if not still:
        raise HTTPException(400, "still_id must reference a generated still in the library")
    audio = _lib_source_path(p.get("audio_id"))
    if not audio:
        raise HTTPException(400, "audio_id must reference a library track")
    # S2V renders only ~4.8s (one 77-frame chunk) from the START of the supplied audio.
    # Trim to a window beginning at audio_start so the clip lands on a vocal passage
    # (otherwise an intro with no singing = nothing to lip-sync).
    start = max(0.0, float(p.get("audio_start") or 0))
    win = float(p.get("length", 77)) / float(p.get("fps", 16)) + 1.5
    try:
        with open(still, "rb") as f:
            img_ref = C.upload_audio(f.read(), os.path.basename(still))
        aud_bytes = _trim_audio_window(audio, start, win)
        aud_ref = C.upload_audio(aud_bytes, _audio_name("s2v_clip", aud_bytes))
        graph, resolved = video_mod.build_s2v(p, img_ref, aud_ref)
    except Exception as e:
        raise HTTPException(500, f"build failed: {e}")
    resolved["audio_start"] = start
    resolved["still_id"] = os.path.basename(p.get("still_id"))
    resolved["audio_id"] = os.path.basename(p.get("audio_id"))
    return _submit_video(graph, resolved, "videolipsync")


@app.post("/api/video/char_still")
def video_char_still(p: dict):
    """Generate a consistent character still in a NEW scene from reference still(s)
    (Qwen-Image-Edit-2511). p: {ref_ids: [library still id, ...] (1-3), prompt, ...}."""
    refs = [_lib_image_path(x) for x in (p.get("ref_ids") or [])]
    refs = [r for r in refs if r]
    if not refs:
        raise HTTPException(400, "ref_ids must reference 1-3 generated stills in the library")
    try:
        uploaded = []
        for r in refs[:3]:
            with open(r, "rb") as f:
                uploaded.append(C.upload_audio(f.read(), os.path.basename(r)))
        graph, resolved = video_mod.build_qwen_char_still(p, uploaded)
    except Exception as e:
        raise HTTPException(500, f"build failed: {e}")
    resolved["ref_ids"] = [os.path.basename(x) for x in (p.get("ref_ids") or [])][:3]
    return _submit_video(graph, resolved, "videostill")


def _krea2_edit_lora():
    """The Identity Edit LoRA's name as ComfyUI sees it (any folder), or None. The shipped
    workflow files it under loras/Krea2/, but accept it anywhere in the loras tree."""
    try:
        for n in C.models("loras"):
            if "krea2_identity_edit" in n.lower():
                return n
    except Exception:
        pass
    return None


@app.post("/api/video/krea2_edit")
def video_krea2_edit(p: dict):
    """Identity-preserving re-staging (Krea 2 Identity Edit): put the person from a
    reference still into a new scene, keeping face/body, with clothing and setting from
    the instruction prompt. p: {ref_id (library still), prompt, seed?, width?, height?,
    ref_boost?, grounding_px?}. Needs the comfyui-krea2edit node pack + the LoRA on the
    box (KREA2-IDENTITY-EDIT_INSTALL.bat)."""
    ref = _lib_image_path(p.get("ref_id") or "")
    if not ref:
        raise HTTPException(400, "ref_id must reference a generated still in the library")
    lora = _krea2_edit_lora()
    if not lora:
        raise HTTPException(400, "krea2_identity_edit LoRA not found on the box - download it "
                                 "from civitai model 2761113 into ComfyUI/models/loras/Krea2/ "
                                 "and run KREA2-IDENTITY-EDIT_INSTALL.bat")
    try:
        with open(ref, "rb") as f:
            up = C.upload_audio(f.read(), os.path.basename(ref))
        graph, resolved = video_mod.build_krea2_identity_still(p, up, lora)
    except Exception as e:
        raise HTTPException(500, f"build failed: {e}")
    resolved["ref_id"] = os.path.basename(str(p.get("ref_id")))
    return _submit_video(graph, resolved, "videostill")


@app.get("/api/video/loras")
def video_loras():
    """LoRAs available on the box (for the character-LoRA picker). Excludes the pipeline's
    own speed LoRAs so only user/character LoRAs show."""
    try:
        skip = ("lightx2v", "causvid", "lightning")
        return [n for n in C.models("loras") if not any(s in n.lower() for s in skip)]
    except Exception:
        return []


@app.post("/api/video/vace")
def video_vace(p: dict):
    """Reference-to-video: animate a referenced character directly (Wan VACE), holding
    identity through motion. p: {still_id, prompt, ...}."""
    still = _lib_image_path(p.get("still_id"))
    if not still:
        raise HTTPException(400, "still_id must reference a generated still in the library")
    try:
        with open(still, "rb") as f:
            ref = C.upload_audio(f.read(), os.path.basename(still))
        graph, resolved = video_mod.build_vace_ref2v(p, ref)
    except Exception as e:
        raise HTTPException(500, f"build failed: {e}")
    resolved["still_id"] = os.path.basename(p.get("still_id"))
    return _submit_video(graph, resolved, "videoclip")


@app.post("/api/video/ltx_t2v")
def video_ltx_t2v(p: dict):
    """LTX-2.3 text-to-video (fast backbone, native synced audio). p: {prompt, seed?,
    width?, height?, frames?, fps?, cfg?}."""
    if not (p.get("prompt") or "").strip():
        raise HTTPException(400, "a prompt is required")
    try:
        graph, resolved = video_mod.build_ltx_t2v(p)
    except Exception as e:
        raise HTTPException(500, f"build failed: {e}")
    return _submit_video(graph, resolved, "videoclip")


def _h3_dispatch(p, build, *args):
    """Shared hunt/finish dispatch for every MiniMax H3 lane.

    SEED HUNT (design decided 2026-08-02 after measuring resolution transfer - see
    docs/MINIMAX_H3_PLAN.md 0a). Drafts render on the TURBO recipe at the SAME resolution as the
    finish, NOT at a cheaper tier: a pinned seed does NOT survive a resolution change (measured
    composition correlation 0.46, below even the different-scene reference of 0.77) but it DOES
    survive the recipe change (0.95). So turbo drafts predict their base finish; low-res drafts
    would not. mode="hunt" -> N drafts at base_seed, +1, +2... (same shape as /api/video/ltx_fflf);
    mode="finish" -> one render on the base recipe at the seed you picked."""
    mode = (p.get("mode") or "finish").strip().lower()
    if mode == "hunt":
        base = video_mod._seed(p)
        n = max(1, min(int(p.get("drafts") or 3), 6))
        out = []
        for i in range(n):
            bp = dict(p)
            bp["seed"] = base + i
            bp.setdefault("turbo", True)              # drafts default to the fast recipe
            bp.pop("mode", None)
            try:
                graph, resolved = build(bp, *args)
            except Exception as e:
                raise HTTPException(500, f"build failed: {e}")
            resolved["hunt_index"] = i
            out.append(_submit_video(graph, resolved, "videoclip"))
        return {"mode": "hunt", "base_seed": base, "drafts": out}
    try:
        graph, resolved = build(p, *args)             # finish: base recipe unless turbo is asked for
    except Exception as e:
        raise HTTPException(500, f"build failed: {e}")
    return _submit_video(graph, resolved, "videoclip")


@app.post("/api/video/h3_t2v")
def video_h3_t2v(p: dict):
    """MiniMax H3 TEXT to video+audio (Phase 0 feasibility gate; see docs/MINIMAX_H3_PLAN.md).
    Faithful port of the reference workflow's TEXT TO VIDEO lane: FL2VA model, res_multistep/simple,
    20 steps, cfg-free BasicGuider (so there is NO negative prompt), joint AV latent decoded through
    both VAEs, VHS_VideoCombine at 24fps. p: {prompt (the author's SECTIONED format), seed?,
    width?/height? or megapixels? (default 0.9 = 1280x736), frames? or seconds? (snapped to 17k+5),
    steps? (20), spectrum? (opt-in SPEEDUP group), turbo? (the author's atomic 4-step recipe),
    mode? ("finish" [default] | "hunt"), drafts? (hunt only, default 3, max 6)}.
    HUNT renders N turbo drafts at the SAME resolution as the finish and returns
    {mode, base_seed, drafts:[...]}; pick a seed, then re-submit it with mode="finish"."""
    if not (p.get("prompt") or "").strip():
        raise HTTPException(400, "a prompt is required")
    return _h3_dispatch(p, video_mod.build_h3_t2v)


@app.post("/api/video/h3_i2v")
def video_h3_i2v(p: dict):
    """MiniMax H3 IMAGE to video+audio (reference workflow's IMAGE TO VIDEO lane). One endpoint,
    three of the node's modes depending on which ids you pass:
      still_id                -> image-to-video
      still_id + last_id      -> first-last-frame interpolation
      last_id only            -> last-frame-only (converge onto a still)
    Output size is derived from the FIRST still (preserving its aspect ratio) unless you pass explicit
    width+height. The prompt must be the sectioned format and, for i2v, must open with the author's
    exact "<Picture 1> ... is fully referenced." line - see docs/MINIMAX_H3_PLAN.md section 4.
    Also supports mode="hunt"/"finish" + drafts?, exactly like /api/video/h3_t2v."""
    if not (p.get("prompt") or "").strip():
        raise HTTPException(400, "a prompt is required")
    if not (p.get("still_id") or p.get("last_id")):
        raise HTTPException(400, "provide still_id (first frame) and/or last_id (last frame)")
    up = {}
    try:
        for key, slot in (("still_id", "first"), ("last_id", "last")):
            if not p.get(key):
                continue
            path = _lib_image_path(p.get(key))
            if not path:
                raise HTTPException(400, f"{key} must reference a generated still in the library")
            with open(path, "rb") as f:
                up[slot] = C.upload_audio(f.read(), os.path.basename(path))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"still upload failed: {e}")

    def _build(q):
        graph, resolved = video_mod.build_h3_i2v(q, up.get("first"), up.get("last"))
        if p.get("still_id"):
            resolved["still_id"] = os.path.basename(p["still_id"])
        if p.get("last_id"):
            resolved["last_id"] = os.path.basename(p["last_id"])
        return graph, resolved

    return _h3_dispatch(p, lambda q: _build(q))


@app.post("/api/video/h3_ref2v")
def video_h3_ref2v(p: dict):
    """MiniMax H3 FULL REFERENCES to video+audio (reference workflow's ref lane, REF2VA model) -
    the identity / MSR-replacement lane. References, all from the library:
      ref_still_ids:  up to 9 generated stills   -> <Picture 1..> in prompt order
      ref_video_ids:  up to 3 library videos     -> <Video 1..>; entries are ids or
                      {"id":..., "use_audio": true} to also reference that video's own soundtrack
                      (its <Audio j> label lands just BEFORE its <Video k>)
      ref_audio_ids:  up to 3 library tracks     -> standalone <Audio j> (voice timbre / lip-sync);
                      entries are ids or {"id":..., "start": sec, "seconds": sec} to reference just
                      a window of the track (e.g. the sung line for this shot)
    Standalone-audio numbering continues AFTER any video soundtracks - the node emits labels in
    fixed order (images, then videos with their soundtracks, then standalone audio).
    The prompt must be the FULL REFERENCES sectioned format (subject_definitions / summary /
    retention_analysis / detailed_description / overall_soundscape / non_diegetic_music) using the
    same <Picture/Video/Audio/Subject N> tags - see docs/MINIMAX_H3_PLAN.md section 4.
    Other p keys as /api/video/h3_t2v: seed?, width?/height? or megapixels?(0.9), frames? or
    seconds?, turbo?, spectrum?, ref_image_size? ("match" [default] | "max" - max = better identity,
    several times slower), mode? ("finish" | "hunt"), drafts?."""
    if not (p.get("prompt") or "").strip():
        raise HTTPException(400, "a prompt is required")
    stills = p.get("ref_still_ids") or []
    vids = p.get("ref_video_ids") or []
    auds = p.get("ref_audio_ids") or []
    if not (stills or vids or auds):
        raise HTTPException(400, "provide at least one reference (ref_still_ids / ref_video_ids / "
                                 "ref_audio_ids); use /api/video/h3_t2v for pure text-to-video")
    ref_images, ref_videos, ref_audios = [], [], []
    try:
        for sid in stills[:9]:
            path = _lib_image_path(sid)
            if not path:
                raise HTTPException(400, f"ref still {sid} not found in the library")
            with open(path, "rb") as f:
                ref_images.append(C.upload_audio(f.read(), os.path.basename(path)))
        for ent in vids[:3]:
            vid = ent.get("id") if isinstance(ent, dict) else ent
            path = _lib_video_path(vid)
            if not path:
                raise HTTPException(400, f"ref video {vid} not found in the library")
            with open(path, "rb") as f:
                ref = C.upload_audio(f.read(), os.path.basename(path))
            ref_videos.append({"video": ref,
                               "use_audio": bool(isinstance(ent, dict) and ent.get("use_audio"))})
        for ent in auds[:3]:
            aid = ent.get("id") if isinstance(ent, dict) else ent
            path = _lib_source_path(aid)
            if not path:
                raise HTTPException(400, f"ref audio {aid} not found in the library")
            if isinstance(ent, dict) and (ent.get("start") is not None or ent.get("seconds")):
                start = max(0.0, float(ent.get("start") or 0))
                win = float(ent.get("seconds") or 15.0)
                aud_bytes = _trim_audio_window(path, start, win)
            else:
                with open(path, "rb") as f:
                    aud_bytes = f.read()
            ref_audios.append(C.upload_audio(aud_bytes, _audio_name("h3_ref", aud_bytes)))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"reference upload failed: {e}")

    def _build(q):
        graph, resolved = video_mod.build_h3_ref2v(q, ref_images, ref_videos, ref_audios)
        resolved["ref_still_ids"] = [os.path.basename(s) for s in stills[:9]]
        resolved["ref_video_ids"] = [os.path.basename(e.get("id") if isinstance(e, dict) else e)
                                     for e in vids[:3]]
        resolved["ref_audio_ids"] = [os.path.basename(e.get("id") if isinstance(e, dict) else e)
                                     for e in auds[:3]]
        return graph, resolved

    return _h3_dispatch(p, lambda q: _build(q))


@app.post("/api/video/ltx_i2v")
def video_ltx_i2v(p: dict):
    """LTX-2.3 image-to-video: animate a library keyframe still (Z-Image/Qwen-Edit still ->
    motion). p: {still_id, prompt, seed?, width?, height?, frames?, fps?, cfg?, img_strength?}."""
    still = _lib_image_path(p.get("still_id"))
    if not still:
        raise HTTPException(400, "still_id must reference a generated still in the library")
    try:
        with open(still, "rb") as f:
            ref = C.upload_audio(f.read(), os.path.basename(still))
        graph, resolved = video_mod.build_ltx_i2v(p, ref)
    except Exception as e:
        raise HTTPException(500, f"build failed: {e}")
    resolved["still_id"] = os.path.basename(p.get("still_id"))
    return _submit_video(graph, resolved, "videoclip")


def _isolate_vocal_bytes(audio_path, start, win, p):
    """Trim a [start, start+win] window of a library track and isolate the vocal for lip-sync
    driving (RoFormer default, Demucs fallback, raw mix if both fail). Returns (bytes, engine_used).
    Same recipe as /api/video/infinitetalk - RoFormer vocal is smeary for STEMS but fine to DRIVE
    lips. Caller is responsible for the temp dir."""
    work = tempfile.mkdtemp(prefix="msr_voc_")
    iso_used = None
    try:
        aud_bytes = _trim_audio_window(audio_path, start, win)
        if p.get("isolate_vocal", True):
            clip_path = os.path.join(work, "clip.wav")
            with open(clip_path, "wb") as f:
                f.write(aud_bytes)
            voc = None
            if bool(ROFORMER_HOST) and p.get("isolate_engine") != "demucs":
                try:
                    stems = _separate(clip_path, work, engine="roformer", stems="all")
                    voc = next((pp for (name, pp) in stems if name == "vocals"), None)
                    if voc:
                        iso_used = "roformer"
                except Exception:
                    voc = None
            if not voc:
                try:
                    files = stems_mod.separate(clip_path, work, mode="vocals")
                    voc = next((f for f in files if os.path.basename(f).startswith("vocals")), None)
                    if voc:
                        iso_used = "demucs"
                except Exception:
                    voc = None
            if voc and os.path.isfile(voc):
                with open(voc, "rb") as f:
                    aud_bytes = f.read()
        return aud_bytes, iso_used
    finally:
        shutil.rmtree(work, ignore_errors=True)


@app.post("/api/video/ltx_msr")
def video_ltx_msr(p: dict):
    """LTX-2.3 Multiple-Subject-Reference: identity from reference image(s) (no keyframe anchor),
    motion prompt-driven. p: {subject_ids: [1-4 library still ids of the character], background_id
    (a scene still), prompt (reference description + action), seed?, width?, height?, frames?, fps?,
    msr_strength?, guide_strength?, ref_frames?, audio_id? (drive NATIVE single-pass lip-sync from a
    library track's isolated vocal - walk AND sing in one LTX pass), audio_start?, isolate_vocal?}."""
    subs = [_lib_image_path(x) for x in (p.get("subject_ids") or [])]
    subs = [s for s in subs if s]
    if not subs:
        raise HTTPException(400, "subject_ids must reference 1-4 generated stills (the character)")
    bg = _lib_image_path(p.get("background_id"))
    if not bg:
        raise HTTPException(400, "background_id must reference a generated still (scene/background)")
    audio = _lib_source_path(p.get("audio_id")) if p.get("audio_id") else None
    start = max(0.0, float(p.get("audio_start") or 0))
    iso_used = None
    try:
        up_subs = []
        for s in subs[:4]:
            with open(s, "rb") as f:
                up_subs.append(C.upload_audio(f.read(), os.path.basename(s)))
        with open(bg, "rb") as f:
            up_bg = C.upload_audio(f.read(), os.path.basename(bg))
        vocal_ref = None
        if audio:
            fps = int(p.get("fps", 24))
            frames = video_mod._ltx_frames(p.get("frames", 145), fps)
            win = frames / fps                # EXACT clip duration: an over-long vocal misaligns the
                                              # AV latent and leaks uncropped MSR reference frames
            aud_bytes, iso_used = _isolate_vocal_bytes(audio, start, win, p)
            vocal_ref = C.upload_audio(aud_bytes, _audio_name("msr_vocal", aud_bytes))
        # EXPERIMENT: optional keyframe pin(s) on the MSR graph (keyframe_first_id / keyframe_last_id =
        # library still ids). Uploaded + passed through so build_ltx_msr can splice an LTXVAddGuide.
        for src_key, dst_key in (("keyframe_first_id", "first_keyframe"), ("keyframe_last_id", "last_keyframe")):
            if p.get(src_key):
                kpath = _lib_image_path(p.get(src_key))
                if not kpath:
                    raise HTTPException(400, f"{src_key} must reference a generated still")
                with open(kpath, "rb") as f:
                    p[dst_key] = C.upload_audio(f.read(), os.path.basename(kpath))
        # editor keyframes are ALREADY in ComfyUI input (uploaded by the timeline editor via /api/comfy);
        # pass their raw filenames straight through so the seed-hunt can drive MSR off the editor timeline.
        if p.get("keyframe_first_name"):
            p["first_keyframe"] = p["keyframe_first_name"]
        if p.get("keyframe_last_name"):
            p["last_keyframe"] = p["keyframe_last_name"]
        graph, resolved = video_mod.build_ltx_msr(p, up_subs, up_bg, vocal_ref)
    except Exception as e:
        raise HTTPException(500, f"build failed: {e}")
    resolved["subject_ids"] = [os.path.basename(x) for x in (p.get("subject_ids") or [])][:4]
    resolved["background_id"] = os.path.basename(p.get("background_id"))
    if audio:
        resolved["audio_id"] = os.path.basename(p.get("audio_id"))
        resolved["audio_start"] = start
        resolved["vocal_isolated"] = bool(iso_used)
        resolved["isolate_engine"] = iso_used
    return _submit_video(graph, resolved, "videoclip")


@app.post("/api/video/ltx_flf")
def video_ltx_flf(p: dict):
    """LTX-2.3 First-Last-Frame: pin the clip's first/last frame to keyframe stills, model interpolates
    between (no IC-LoRA guide -> no head-leak; camera move = the framing difference of the two stills).
    p: {first_id (library still), last_id? (defaults to first_id = STATIC shot), prompt, negative?, seed?,
    width?, height?, frames?, fps?, cfg?, first_strength?, last_strength?, audio_id?, audio_start?,
    isolate_vocal?}."""
    first = _lib_image_path(p.get("first_id"))
    if not first:
        raise HTTPException(400, "first_id must reference a generated still (the opening keyframe)")
    last = _lib_image_path(p.get("last_id")) if p.get("last_id") else first   # default static (first==last)
    audio = _lib_source_path(p.get("audio_id")) if p.get("audio_id") else None
    start = max(0.0, float(p.get("audio_start") or 0))
    iso_used = None
    try:
        with open(first, "rb") as f:
            up_first = C.upload_audio(f.read(), os.path.basename(first))
        up_last = up_first if last == first else C.upload_audio(open(last, "rb").read(), os.path.basename(last))
        vocal_ref = None
        if audio:
            fps = int(p.get("fps", 24))
            frames = video_mod._ltx_frames(p.get("frames", 121), fps)
            aud_bytes, iso_used = _isolate_vocal_bytes(audio, start, frames / fps, p)
            vocal_ref = C.upload_audio(aud_bytes, _audio_name("flf_vocal", aud_bytes))
        graph, resolved = video_mod.build_ltx_flf(p, up_first, up_last, vocal_ref)
    except Exception as e:
        raise HTTPException(500, f"build failed: {e}")
    resolved["first_id"] = os.path.basename(p.get("first_id"))
    resolved["last_id"] = os.path.basename(p.get("last_id") or p.get("first_id"))
    if audio:
        resolved["audio_id"] = os.path.basename(p.get("audio_id"))
        resolved["audio_start"] = start
        resolved["vocal_isolated"] = bool(iso_used)
    return _submit_video(graph, resolved, "videoclip")


@app.post("/api/video/ltx_keyframe")
def video_ltx_keyframe(p: dict):
    """LTX-2.3 KEYFRAME mode: faithful port of the WhatDreamsCost LTX Director 2-stage example workflow
    (base sample -> spatial-x2 latent-upscale REFINE) driving LTXDirector -> LTXDirectorGuide with no MSR
    IC-LoRA. Place 1-N library stills at absolute frame positions and interpolate, with an optional
    per-segment PROMPT timeline. The LTXDirector-native successor to /api/video/ltx_flf. No lip-sync (the
    keyframe/B-roll use case pins identity via the stills). p: {keyframes: [{still_id (library still),
    start? (frame), length? (frames), isEndFrame? (place at the END of its window), guide_strength?
    (0-1)}], prompt, negative?, global_prompt?, local_prompts?, segment_lengths?, epsilon?, seed?, width?,
    height? (TARGET output res when base_scale=0.5), frames?, fps?, cfg?, distill_strength?, base_scale?
    (0.5 = output==target res [default], 1.0 = output 2x via the x2 upsampler), base_steps?, refine_steps?,
    refine_denoise?}. OR pass timeline_data (+ local_prompts/segment_lengths/guide_strength/global_prompt)
    straight from the Shot Studio LTXDirector editor — its media is already uploaded to ComfyUI input via
    /api/comfy, so it renders directly with no keyframes[]."""
    if (p.get("timeline_data") or "").strip():
        # Shot Studio editor passthrough: render the editor's authored timeline through the Director graph.
        try:
            graph, resolved = video_mod.build_ltx_keyframe(p, [])
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"build failed: {e}")
        resolved["from_editor"] = True
        return _submit_video(graph, resolved, "videoclip")
    raw_kfs = p.get("keyframes")
    if not raw_kfs and p.get("keyframe_ids"):              # convenience: bare id list -> keyframes
        raw_kfs = [{"still_id": x} for x in p.get("keyframe_ids")]
    if not isinstance(raw_kfs, list) or not raw_kfs:
        raise HTTPException(400, "keyframes must be a non-empty list of {still_id, ...}")
    resolved_kfs = []
    for k in raw_kfs:
        if not isinstance(k, dict):
            raise HTTPException(400, "each keyframe must be an object with a still_id")
        path = _lib_image_path(k.get("still_id"))
        if not path:
            raise HTTPException(400, f"keyframe still_id {k.get('still_id')!r} must reference a generated still")
        resolved_kfs.append({"_path": path, "_still_id": os.path.basename(k.get("still_id")),
                             "start": k.get("start"), "length": k.get("length", 1),
                             "isEndFrame": bool(k.get("isEndFrame", False)),
                             "guide_strength": float(k.get("guide_strength", 1.0))})
    try:
        for k in resolved_kfs:                              # upload each still to ComfyUI input
            with open(k["_path"], "rb") as f:
                k["imageFile"] = C.upload_audio(f.read(), os.path.basename(k["_path"]))
        graph, resolved = video_mod.build_ltx_keyframe(p, resolved_kfs)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"build failed: {e}")
    resolved["keyframe_ids"] = [k["_still_id"] for k in resolved_kfs]
    return _submit_video(graph, resolved, "videoclip")


@app.post("/api/video/ltx_fflf")
def video_ltx_fflf(p: dict):
    """LTX-2.3 FFLF Seed-Hunter / Multiroll (faithful foxydits port, stock LTXVAddGuide - NOT LTXDirector).
    Pin the clip's first + last frame and interpolate; EACH anchor is a library STILL or a library CLIP
    (a clip's tail/head carries boundary motion -> the basis for continuous-take chaining; a still cannot
    show a subject entering frame). Two stages with DECOUPLED seeds:
      mode="hunt"   -> 3 HALF-RES drafts (base_seed, +1, +2) to eyeball and pick a golden seed (cheap).
      mode="finish" -> spatial-x2 upscale REFINE of the chosen stage-1 seed; re-roll stage2_seed = multiroll.
    Identity comes from the anchors (FFLF can't share a graph with MSR) - author on-model anchors upstream
    (/api/video/qwen_char_still) or feed a prior MSR clip's tail as a video anchor for entrances.
    p: {first_id (library still OR clip), last_id? (defaults to first_id = static push), first_kind?/
    last_kind? ("image"|"video"; auto-detected from the library id if omitted), first_frames?/first_skip?,
    last_frames?/last_skip? (video anchors: frame_load_cap / skip_first_frames), prompt, negative?, mode?
    ("finish" [default] | "hunt"), seed?/stage1_seed?, stage2_seed?, width?, height?, frames?, fps?, cfg?,
    first_strength?(0.7), last_strength?(0.7), nag_scale?(50), char_lora?, omni_lora?, audio_id?,
    audio_start?, isolate_vocal? (FINISH-only masked-audio lip-sync, identity still from the anchors)}."""
    if not (p.get("prompt") or "").strip():
        raise HTTPException(400, "a prompt is required (what happens between the anchors)")

    def _anchor(idval, want_kind, frames, skip):
        """Resolve a library id to an FFLF anchor spec (uploaded to ComfyUI). Auto-detects still vs clip
        unless want_kind forces it. Returns (spec, label) or (None, None)."""
        if not idval:
            return None, None
        img = _lib_image_path(idval)
        vid = _lib_video_path(idval)
        if want_kind == "video":
            img = None
        elif want_kind == "image":
            vid = None
        path = vid or img
        if not path:
            raise HTTPException(400, f"anchor {idval!r} must reference a generated still or clip")
        kind = "video" if vid else "image"
        with open(path, "rb") as f:
            name = C.upload_audio(f.read(), os.path.basename(path))   # /upload/image helper (any file)
        spec = {"kind": kind, "name": name}
        if kind == "video":
            spec["frames"] = int(frames or 9)
            spec["skip"] = int(skip or 0)
        return spec, os.path.basename(path)

    def _named(name):
        """An anchor already uploaded to ComfyUI input (the timeline editor's image keyframe). Used as-is,
        no re-upload. Image-only (the timeline holds stills)."""
        name = (name or "").strip()
        return ({"kind": "image", "name": name}, name) if name else (None, None)

    mode = (p.get("mode") or "finish").strip().lower()
    try:
        # Anchors prefer an explicit ComfyUI image name (from the timeline editor's keyframes); fall back to
        # a library id (still or video tail, e.g. an extend) resolved + uploaded by _anchor.
        if p.get("first_name"):
            first_src, first_lbl = _named(p.get("first_name"))
        else:
            first_src, first_lbl = _anchor(p.get("first_id"), (p.get("first_kind") or "").lower(),
                                           p.get("first_frames"), p.get("first_skip"))
        if not first_src:
            raise HTTPException(400, "an opening anchor is required (timeline image keyframe or first_id)")
        if p.get("last_name"):
            last_src, last_lbl = _named(p.get("last_name"))
        elif p.get("last_id"):
            last_src, last_lbl = _anchor(p.get("last_id"), (p.get("last_kind") or "").lower(),
                                         p.get("last_frames"), p.get("last_skip"))
        else:
            last_src, last_lbl = first_src, first_lbl    # default last == first (a slow static push)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"anchor prep failed: {e}")

    # FINISH-only masked-audio lip-sync (identity stays with the anchors; drafts skip audio for speed)
    audio = _lib_source_path(p.get("audio_id")) if p.get("audio_id") else None
    start = max(0.0, float(p.get("audio_start") or 0))
    iso_used = None
    vocal_ref = None
    if audio and mode != "hunt":
        try:
            fps = int(p.get("fps", 24))
            frames = video_mod._ltx_frames(p.get("frames", 97), fps)
            aud_bytes, iso_used = _isolate_vocal_bytes(audio, start, frames / fps, p)
            vocal_ref = C.upload_audio(aud_bytes, _audio_name("fflf_vocal", aud_bytes))
        except Exception as e:
            raise HTTPException(500, f"vocal prep failed: {e}")

    if mode == "hunt":
        base = video_mod._seed(p)                        # honour a given seed, else random base
        drafts = []
        for i in range(3):                               # 3 half-res drafts: base, base+1, base+2
            bp = dict(p); bp["mode"] = "hunt"; bp["stage1_seed"] = base + i
            try:
                graph, resolved = video_mod.build_ltx_fflf(bp, first_src, last_src, None)
            except Exception as e:
                raise HTTPException(500, f"build failed: {e}")
            resolved["first_id"] = first_lbl; resolved["last_id"] = last_lbl; resolved["hunt_index"] = i
            drafts.append(_submit_video(graph, resolved, "videoclip"))  # auto-free only if the model changed
        return {"mode": "hunt", "base_seed": base, "drafts": drafts}

    try:
        graph, resolved = video_mod.build_ltx_fflf(p, first_src, last_src, vocal_ref)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"build failed: {e}")
    resolved["first_id"] = first_lbl; resolved["last_id"] = last_lbl
    if audio:
        resolved["audio_id"] = os.path.basename(p.get("audio_id"))
        resolved["audio_start"] = start
        resolved["vocal_isolated"] = bool(iso_used)
    return _submit_video(graph, resolved, "videoclip")


@app.post("/api/video/crop_still")
def video_crop_still(p: dict):
    """Center-crop a library still by `keep` (0.3-0.95 of each side) and register the crop as a NEW
    library videostill. Used by the Shot Editor's B-roll push-in: the chosen background is the FFLF
    opening anchor and this center-crop is the closing anchor, so the shot dollies IN between two
    person-free pinned frames (no hallucinated figures, controlled speed). GPU-free (Pillow).
    p: {still_id (a library videostill), keep? (0.72)}. Returns {id, media_url}."""
    from PIL import Image
    src = _lib_image_path(p.get("still_id"))
    if not src:
        raise HTTPException(400, "still_id must reference a generated still")
    keep = max(0.3, min(0.95, float(p.get("keep") or 0.72)))
    img = Image.open(src).convert("RGB")
    w, h = img.size
    cw, ch = round(w * keep), round(h * keep)
    sx, sy = (w - cw) // 2, (h - ch) // 2
    out = img.crop((sx, sy, sx + cw, sy + ch)).resize((w, h), Image.LANCZOS)   # back to source res = matches the first anchor
    jid = uuid.uuid4().hex
    path = os.path.join(LIBRARY, f"{jid}.png")
    out.save(path)
    save_done_row(jid, "videostill", {"source": "crop_still", "still_id": os.path.basename(p.get("still_id")), "keep": keep}, path)
    return {"id": jid, "media_url": f"/api/media/{jid}"}


@app.post("/api/video/ltx_retake")
def video_ltx_retake(p: dict):
    """LTX-2.3 RETAKE: re-render only a time slice of an EXISTING clip (LTXDirectorGuide retake_mode),
    freezing the rest. For fixing a glitchy second or two without re-rolling the whole shot. The clip's
    own resolution/length/fps are probed so the rebuilt latent lines up frame-for-frame with the source.
    SINGING clips: pass audio_id + audio_start (the SAME ones the clip was rendered with) so the retaken
    slice is regenerated against the real vocal and the lips stay in sync; without it the slice desyncs.
    p: {clip_id (a rendered library clip), retake_start (seconds), retake_length (seconds), prompt
    (what the slice should be), retake_strength? (0-1, default 1.0; lower = stay closer to the original),
    audio_id? (the track the clip lip-syncs to), audio_start? (offset into that track for this clip),
    isolate_vocal?, negative?, seed?, cfg?}."""
    vid = _lib_video_path(p.get("clip_id") or p.get("video_id"))
    if not vid:
        raise HTTPException(400, "clip_id must reference a generated clip in the library")
    if not (p.get("prompt") or "").strip():
        raise HTTPException(400, "a prompt is required (what the retaken slice should show)")
    fps = _probe_fps(vid) or 24.0
    try:
        import subprocess
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                              "-show_entries", "stream=width,height,nb_read_frames", "-of", "csv=p=0", vid],
                             capture_output=True, text=True, timeout=120).stdout.strip().split(",")
        w = int(out[0]); h = int(out[1]); frames = int(out[2]) if len(out) > 2 and out[2].isdigit() else 0
    except Exception as e:
        raise HTTPException(500, f"could not probe the clip: {e}")
    if not (w and h and frames):
        raise HTTPException(500, "could not read the clip's dimensions/length")
    rs = max(0.0, float(p.get("retake_start") or 0))
    rl = max(0.1, float(p.get("retake_length") or 1))
    fps_i = int(round(fps))
    bp = dict(p)
    bp.update({"width": w, "height": h, "frames": frames, "fps": fps_i,
               "retake_start": int(round(rs * fps)), "retake_length": int(round(rl * fps))})
    # SINGING retake: isolate the SAME vocal window the clip was rendered with so the regenerated slice
    # lip-syncs (a music video is mostly singing - this is the default for any clip that has audio).
    audio = _lib_source_path(p.get("audio_id")) if p.get("audio_id") else None
    astart = max(0.0, float(p.get("audio_start") or 0))
    iso_used = None
    try:
        vocal_ref = None
        if audio:
            win = frames / float(fps_i)                  # full clip duration, aligned at audio_start
            aud_bytes, iso_used = _isolate_vocal_bytes(audio, astart, win, p)
            vocal_ref = C.upload_audio(aud_bytes, _audio_name("retake_vocal", aud_bytes))
        with open(vid, "rb") as f:
            base = C.upload_audio(f.read(), os.path.basename(vid))
        graph, resolved = video_mod.build_ltx_retake(bp, base, vocal_ref)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"build failed: {e}")
    resolved["clip_id"] = os.path.basename(p.get("clip_id") or p.get("video_id"))
    if audio:
        resolved["audio_id"] = os.path.basename(p.get("audio_id"))
        resolved["audio_start"] = astart
        resolved["vocal_isolated"] = bool(iso_used)
    return _submit_video(graph, resolved, "videoclip")


@app.post("/api/video/svi_i2v")
def video_svi_i2v(p: dict):
    """SVI2 Pro long-form Wan 2.2 A14B i2v: animate a library still into a long continuous clip
    via chained 81-frame segments (two-expert, fp8). p: {still_id, prompt?, seconds?/frames?/
    segments?, width?, height?, fps?, overlap?, seed?}."""
    still = _lib_image_path(p.get("still_id"))
    if not still:
        raise HTTPException(400, "still_id must reference a generated still in the library")
    try:
        with open(still, "rb") as f:
            ref = C.upload_audio(f.read(), os.path.basename(still))
        graph, resolved = video_mod.build_svi_i2v(p, ref)
    except Exception as e:
        raise HTTPException(500, f"build failed: {e}")
    resolved["still_id"] = os.path.basename(p.get("still_id"))
    return _submit_video(graph, resolved, "videoclip")


@app.post("/api/video/retime")
def video_retime(p: dict):
    """Speed up/down a finished library clip (GPU-free ffmpeg) - the fix for uniform slow-motion
    (e.g. distilled SVI output). p: {video_id, speed (>1 = faster, default 1.7), fps?}."""
    vid = _lib_video_path(p.get("video_id"))
    if not vid:
        raise HTTPException(400, "video_id must reference a generated clip in the library")
    speed = max(0.1, float(p.get("speed") or 1.7))
    fps = int(p.get("fps") or 24)
    jid = uuid.uuid4().hex
    out = os.path.join(LIBRARY, f"{jid}.mp4")
    try:
        musicvideo_mod.retime(vid, out, speed, fps)
    except Exception as e:
        raise HTTPException(500, f"retime failed: {e}")
    save_done_row(jid, "videoclip", {"source": os.path.basename(p.get("video_id")), "speed": speed}, out)
    return {"job_id": jid, "media_url": f"/api/media/{jid}", "status": "done"}


def _lib_video_path(vid):
    """Absolute path to a library video (clip or assembled music video) by id."""
    vid = os.path.basename(vid or "")
    path = os.path.join(LIBRARY, f"{vid}.mp4")
    return path if vid and os.path.exists(path) else None


def _probe_nframes(path):
    """Count video frames (so a pose-guided S2V clip matches the motion video's length)."""
    import subprocess
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                              "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
                             capture_output=True, text=True, timeout=40).stdout.strip()
        return int(out) if out.isdigit() else 0
    except Exception:
        return 0


def _probe_fps(path, default=24.0):
    """Source video fps (so the upscaled output keeps the same timing)."""
    import subprocess
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                              "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", path],
                             capture_output=True, text=True, timeout=15).stdout.strip()
        num, den = (out.split("/") + ["1"])[:2] if "/" in out else (out, "1")
        return round(float(num) / float(den), 3) if float(den) else default
    except Exception:
        return default


@app.post("/api/video/upscale")
def video_upscale(p: dict):
    """SeedVR2 diffusion upscale of a finished library clip/video (temporal-aware). p:
    {video_id, resolution? (target short side, default 1080), model?, batch_size? (4n+1),
    color_correction?, blocks_to_swap?, offload?, frame_cap?, seed?}. First run auto-downloads
    the SeedVR2 model into models/SEEDVR2."""
    vid = _lib_video_path(p.get("video_id"))
    if not vid:
        raise HTTPException(400, "video_id must reference a generated clip/video in the library")
    fps = _probe_fps(vid)
    try:
        with open(vid, "rb") as f:
            ref = C.upload_audio(f.read(), os.path.basename(vid))
        graph, resolved = video_mod.build_seedvr2_upscale(p, ref, fps)
    except Exception as e:
        raise HTTPException(500, f"build failed: {e}")
    resolved["video_id"] = os.path.basename(p.get("video_id"))
    return _submit_video(graph, resolved, "videoclip")


@app.post("/api/video/flashvsr")
def video_flashvsr(p: dict):
    """FlashVSR diffusion upscale (naxci1 Stable node) of a finished library clip - alternative to
    SeedVR2. p: {video_id, scale? (2/4, default 2), mode? (tiny|tiny-long|full), vae_model?,
    tiled_vae?, tiled_dit?, attention_mode?, frame_chunk_size?, seed?}. Models in
    models/FlashVSR-v1.1 (download_flashvsr_models.bat); node installed via Manager."""
    vid = _lib_video_path(p.get("video_id"))
    if not vid:
        raise HTTPException(400, "video_id must reference a generated clip/video in the library")
    fps = _probe_fps(vid)
    # AUTO-CHUNK long clips: a whole upscaled clip is accumulated in system RAM before encode, so a
    # ~20s shot (480f @720p) OOMs the box's 32GB RAM in one pass while 6s (145f) clips are fine.
    # frame_chunk_size caps the working buffer. Only apply it ABOVE a threshold so short clips stay
    # unchunked (no temporal seam at chunk boundaries). User can override by passing frame_chunk_size.
    if "frame_chunk_size" not in p:
        try:
            import subprocess
            n = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                                "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", vid],
                               capture_output=True, text=True, timeout=30).stdout.strip()
            nframes = int(n) if n.isdigit() else 0
        except Exception:
            nframes = 0
        cap = int(p.get("frame_cap") or 0)
        eff = min(nframes, cap) if cap else nframes
        if eff > 200:                                       # ~8s+ -> chunk; below that, clean unchunked
            p["frame_chunk_size"] = 48
    try:
        with open(vid, "rb") as f:
            ref = C.upload_audio(f.read(), os.path.basename(vid))
        graph, resolved = video_mod.build_flashvsr_upscale(p, ref, fps)
    except Exception as e:
        raise HTTPException(500, f"build failed: {e}")
    resolved["video_id"] = os.path.basename(p.get("video_id"))
    return _submit_video(graph, resolved, "videoclip")


@app.post("/api/video/regrade")
def video_regrade(p: dict):
    """AI/colour-science grade of a finished library clip, run in ComfyUI (replaces the ffmpeg
    looks; applied per-clip before assembly). p: {video_id, look_source: "darkroom"|"vcg"|
    "colormatch", film_stock?, grain?, halation?, lut_id?, ref_still_id?, cm_method?}. SCAFFOLD:
    node class names/keys are verified against /object_info after the box install (see
    docs/MV_AI_GRADING_PLAN.md). Mirrors video_flashvsr."""
    vid = _lib_video_path(p.get("video_id"))
    if not vid:
        raise HTTPException(400, "video_id must reference a generated clip/video in the library")
    fps = _probe_fps(vid)
    src = (p.get("look_source") or "darkroom").lower()
    try:
        with open(vid, "rb") as f:
            vref = C.upload_audio(f.read(), os.path.basename(vid))
        if src == "vcg":
            lut = _lib_image_path(p.get("lut_id"))           # a saved VCG LUT artifact
            if not lut:
                raise HTTPException(400, "vcg look requires a lut_id (generate one via /api/mv/generate_lut)")
            with open(lut, "rb") as f:
                lref = C.upload_audio(f.read(), os.path.basename(lut))
            graph, resolved = video_mod.build_vcg_apply(p, vref, lref, fps)
        else:                                                # darkroom (default); colormatch TBD
            graph, resolved = video_mod.build_darkroom_grade(p, vref, fps)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"build failed: {e}")
    resolved["video_id"] = os.path.basename(p.get("video_id"))
    return _submit_video(graph, resolved, "videoclip")


@app.post("/api/mv/generate_lut")
def mv_generate_lut(p: dict):
    """Generate a reusable VCG 3D LUT from a reference still (the 4GB diffusion model loads only
    here; fired ONCE per look). Body: {ref_still_id}. Returns a job whose saved output is the LUT
    artifact, reused across all clips via /api/video/regrade look_source="vcg"."""
    ref = _lib_image_path(p.get("ref_still_id"))
    if not ref:
        raise HTTPException(400, "ref_still_id must reference a still in the library")
    try:
        with open(ref, "rb") as f:
            rref = C.upload_audio(f.read(), os.path.basename(ref))
        graph, resolved = video_mod.build_vcg_lut(p, rref)
    except Exception as e:
        raise HTTPException(500, f"build failed: {e}")
    return _submit_video(graph, resolved, "videostill")


@app.get("/api/mv/look_library")
def mv_look_library():
    """Look sources for the grade picker (data-driven, like /api/mv/grades). SCAFFOLD: the
    darkroom_stocks list should be read from /object_info once Darkroom is installed; refs are the
    curated public-domain starter stills under library/grade_refs/."""
    refs = []
    refdir = os.path.join(LIBRARY, "grade_refs")
    man = os.path.join(refdir, "manifest.json")
    if os.path.exists(man):
        try:
            with open(man) as f:
                refs = json.load(f)
        except Exception:
            refs = []
    return {"darkroom_stocks": [], "luts": [], "refs": refs,
            "ffmpeg_looks": musicvideo_mod.grade_names()}   # ffmpeg looks kept as no-box fallback


@app.post("/api/video/ltx_lipsync")
def video_ltx_lipsync(p: dict):
    """LTX-2.3 i2v + LatentSync lip-sync: animate a keyframe still, then sync its mouth to a
    vocal window from a library track - all on LTX footage (one consistent look). p: {still_id,
    audio_id, audio_start?, prompt?, frames?, lips_expression?, inference_steps?}."""
    still = _lib_image_path(p.get("still_id"))
    if not still:
        raise HTTPException(400, "still_id must reference a generated still in the library")
    audio = _lib_source_path(p.get("audio_id"))
    if not audio:
        raise HTTPException(400, "audio_id must reference a library track")
    start = max(0.0, float(p.get("audio_start") or 0))
    fps = 25                                            # LatentSync input rate
    frames = int(p.get("frames", 97))
    win = frames / fps + 1.0                            # vocal window >= clip length
    try:
        with open(still, "rb") as f:
            img_ref = C.upload_audio(f.read(), os.path.basename(still))
        aud_bytes = _trim_audio_window(audio, start, win)
        aud_ref = C.upload_audio(aud_bytes, _audio_name("ltx_vocal", aud_bytes))
        graph, resolved = video_mod.build_ltx_lipsync(p, img_ref, aud_ref)
    except Exception as e:
        raise HTTPException(500, f"build failed: {e}")
    resolved["audio_start"] = start
    resolved["still_id"] = os.path.basename(p.get("still_id"))
    resolved["audio_id"] = os.path.basename(p.get("audio_id"))
    return _submit_video(graph, resolved, "videolipsync")


@app.post("/api/video/s2v_wrapper")
def video_s2v_wrapper(p: dict):
    """Wan2.2-S2V lip-sync via the WanVideoWrapper (block-swap, fits 24GB). Animate a portrait
    still to a vocal window from a library track. p: {still_id, audio_id, audio_start?, prompt?,
    frames?, width?, height?, blocks_to_swap?, steps?}."""
    still = _lib_image_path(p.get("still_id"))
    if not still:
        raise HTTPException(400, "still_id must reference a generated still in the library")
    audio = _lib_source_path(p.get("audio_id"))
    if not audio:
        raise HTTPException(400, "audio_id must reference a library track")
    start = max(0.0, float(p.get("audio_start") or 0))
    fps = int(p.get("fps", 16))
    # pose-guided combine: a motion clip's body pose drives S2V while the audio drives the lips.
    # Match the clip length to the pose video (its frame count wins over seconds), and upload it.
    pose_id = p.get("pose_video")
    pose_path = None
    if pose_id:
        pose_path = _lib_video_path(pose_id)
        if not pose_path:
            raise HTTPException(400, "pose_video must reference a generated clip in the library")
        p["frames"] = _probe_nframes(pose_path) or int(p.get("frames", 77))
        p.pop("seconds", None)
    # Trim enough audio to cover the TOTAL clip (seconds, or frames). For multi-window long S2V
    # the window count is derived from the audio length, so the audio must span the whole clip.
    total_frames = int(round(float(p["seconds"]) * fps)) if p.get("seconds") else int(p.get("frames", 77))
    win = total_frames / fps + 1.5
    try:
        with open(still, "rb") as f:
            img_ref = C.upload_audio(f.read(), os.path.basename(still))
        aud_bytes = _trim_audio_window(audio, start, win)
        aud_ref = C.upload_audio(aud_bytes, _audio_name("s2v_clip", aud_bytes))
        if pose_path:
            with open(pose_path, "rb") as f:
                p["pose_video"] = C.upload_audio(f.read(), os.path.basename(pose_path))   # name on ComfyUI
        graph, resolved = video_mod.build_s2v_wrapper(p, img_ref, aud_ref)
    except Exception as e:
        raise HTTPException(500, f"build failed: {e}")
    resolved["audio_start"] = start
    resolved["still_id"] = os.path.basename(p.get("still_id"))
    resolved["audio_id"] = os.path.basename(p.get("audio_id"))
    if pose_id:
        resolved["pose_video"] = os.path.basename(pose_id)
    return _submit_video(graph, resolved, "videolipsync")


@app.post("/api/video/infinitetalk")
def video_infinitetalk(p: dict):
    """InfiniteTalk video-to-video lip-sync: keep an EXISTING clip's motion/camera/background and
    redrive only the mouth/face from a vocal window. The 'walking AND singing' lane that pose-S2V
    can't do. p: {video_id (source motion clip in the library), audio_id, audio_start?, prompt?,
    width?, height?, steps?, cfg?, shift?, blocks_to_swap?, audio_scale?}."""
    src = _lib_video_path(p.get("video_id"))
    if not src:
        raise HTTPException(400, "video_id must reference a generated clip in the library")
    audio = _lib_source_path(p.get("audio_id"))
    if not audio:
        raise HTTPException(400, "audio_id must reference a library track")
    start = max(0.0, float(p.get("audio_start") or 0))
    fps = int(p.get("fps", 25))
    # Length follows the SOURCE clip (it provides the motion). Probe its duration -> seconds, so the
    # generated clip spans the whole walk; the audio window is trimmed to match.
    sfps = _probe_fps(src) or 25.0
    snf = _probe_nframes(src) or 0
    if snf and not p.get("seconds") and not p.get("frames"):
        p["seconds"] = snf / sfps
    total_frames = int(round(float(p["seconds"]) * fps)) if p.get("seconds") else int(p.get("frames", 81))
    # The MultiTalk windowed sampler rounds the frame count UP to a whole number of windows
    # (motion_frame + k*(frame_window_size - motion_frame)). If the audio embeds are computed for the
    # un-rounded count, the rounded-up tail has no audio and the lips freeze for the last ~1-2s.
    # Snap to the window boundary HERE so the audio embeds + trimmed audio cover exactly what's
    # generated. (k*(fws-mf)+mf is always 4n+1, so it stays a valid Wan latent length.)
    fws = int(p.get("frame_window_size", 81))
    mf = int(p.get("motion_frame", 9))
    step = max(1, fws - mf)
    windows = max(1, math.ceil((total_frames - mf) / step))
    total_frames = mf + windows * step
    p["frames"] = total_frames          # builder uses this verbatim for num_frames
    p.pop("seconds", None)              # frames is now authoritative
    win = total_frames / fps + 1.0      # trim enough vocal to cover the whole clip
    # wav2vec drives the lips from PHONETIC content, so a full music mix (drums/guitars) muddies
    # the sync. Isolate the vocal first (default on for this route) -> much better singing lip-sync.
    # roformer (box GPU, SOTA) if configured + requested, else Demucs (Mac MPS). Falls back to the
    # raw mix if separation fails. We trim the short window first, then separate just that (fast).
    isolate = p.get("isolate_vocal", True)
    # default = BS-RoFormer (the box service, SOTA). Demucs is only a fallback if RoFormer is
    # unset or fails - it's documented-inadequate, so never the default. `isolate_engine:"demucs"`
    # forces Demucs if ever wanted.
    want_roformer = bool(ROFORMER_HOST) and p.get("isolate_engine") != "demucs"
    work = tempfile.mkdtemp(prefix="it_voc_")
    iso_used = None
    try:
        with open(src, "rb") as f:
            vid_ref = C.upload_audio(f.read(), os.path.basename(src))   # name on ComfyUI
        aud_bytes = _trim_audio_window(audio, start, win)
        if isolate:
            clip_path = os.path.join(work, "clip.wav")
            with open(clip_path, "wb") as f:
                f.write(aud_bytes)
            voc = None
            if want_roformer:
                try:
                    stems = _separate(clip_path, work, engine="roformer", stems="all")
                    voc = next((pp for (name, pp) in stems if name == "vocals"), None)
                    if voc:
                        iso_used = "roformer"
                except Exception:
                    voc = None                                         # RoFormer down -> try Demucs
            if not voc:
                try:
                    files = stems_mod.separate(clip_path, work, mode="vocals")   # Demucs, Mac MPS
                    voc = next((f for f in files if os.path.basename(f).startswith("vocals")), None)
                    if voc:
                        iso_used = "demucs"
                except Exception:
                    voc = None                                         # both failed -> full mix
            if voc and os.path.isfile(voc):
                with open(voc, "rb") as f:
                    aud_bytes = f.read()
        aud_ref = C.upload_audio(aud_bytes, _audio_name("infinitetalk_clip", aud_bytes))
        graph, resolved = video_mod.build_infinitetalk_v2v(p, vid_ref, aud_ref)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"build failed: {e}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    resolved["audio_start"] = start
    resolved["video_id"] = os.path.basename(p.get("video_id"))
    resolved["audio_id"] = os.path.basename(p.get("audio_id"))
    resolved["vocal_isolated"] = bool(iso_used)
    resolved["isolate_engine"] = iso_used
    return _submit_video(graph, resolved, "videolipsync")


@app.post("/api/video/assemble_chain")
def video_assemble_chain(p: dict):
    """Concat a Shot Studio chain (base shot + FFLF extends) into ONE continuous library clip, GPU-free
    (ffmpeg). Each extend was driven by the previous clip's last `tail` frames, so it re-depicts them at
    its head; we trim `tail` frames off every piece after the first so the join is seamless (matches the
    UI's totalSecs math: base + (N-1)*(frames-tail)). p: {clips:[ids in order], frames (per-piece), fps?,
    tail?(33), width?, height?}. Returns {id, media_url}."""
    clips = [os.path.basename(str(c)) for c in (p.get("clips") or []) if c]
    if len(clips) < 2:
        raise HTTPException(400, "need 2+ ordered clip ids to assemble a chain")
    fps = float(p.get("fps") or 24)
    frames = int(p.get("frames") or 0)
    tail = int(p.get("tail") or 33)
    if frames <= 0:
        raise HTTPException(400, "frames (per-piece length) is required")
    full, skip = frames / fps, tail / fps
    segs = []
    for i, cid in enumerate(clips):
        path = os.path.join(LIBRARY, f"{cid}.mp4")
        if not os.path.exists(path):
            raise HTTPException(404, f"clip {cid} not found")
        segs.append({"path": path, "dur": full, "cid": cid} if i == 0
                    else {"path": path, "dur": max(0.1, full - skip), "ss": skip, "cid": cid})
    jid = uuid.uuid4().hex
    out = os.path.join(LIBRARY, f"{jid}.mp4")
    # transition (seconds) = optional crossfade blended at each seam; 0 = hard cut (default). A small
    # value (~0.2-0.5s) smooths any hitch at the trim joins.
    transition = max(0.0, float(p.get("transition") or 0))
    try:
        musicvideo_mod.assemble(segs, None, out, width=int(p.get("width") or 1280),
                                height=int(p.get("height") or 720), fps=int(fps), transition=transition)
    except Exception as e:
        raise HTTPException(500, f"chain assemble failed: {e}")
    save_done_row(jid, "videoclip", {"kind": "video", "source": "chain assemble", "pieces": len(clips)}, out)
    return {"id": jid, "media_url": f"/api/media/{jid}", "status": "done"}


@app.post("/api/mv/script")
def mv_script(body: dict):
    """Generate an editable music-video shot list from a song. Body: {project? (key) OR
    song (normalized view with sections), cast?: [{name, role}], provider?, model?, shots?}."""
    song = body.get("song")
    if not song and body.get("project"):
        r = _resolve_project(body["project"])
        if not r:
            raise HTTPException(404, "project not found")
        song = _project_song_view(json.loads(r["data"] or "{}"))
    if not song or not song.get("sections"):
        raise HTTPException(400, "provide a song with sections, or a project key with a Song arrangement")
    provider = body.get("provider") or llm_mod.best_provider()
    # Default the script model to Sonnet 5 on the Claude paths (was Sonnet 4.6): same speed class at
    # --effort low, stronger long structured-JSON output, which is what the writer emits. Overridable
    # per-call - the UI picker also offers Opus 5 and Sonnet 4.6.
    model = body.get("model") or ""
    if not model and provider in ("claude_sub", "claude_code", "claude"):
        model = "claude-sonnet-5"
    claude_model = CFG.get("claude_model", "claude-3-5-sonnet-latest")
    cast = body.get("cast") or []
    n_shots = int(body.get("shots") or 0)
    # STRUCTURE-DRIVEN: analyze the actual audio (allin1 on the box) and build a deterministic shot grid
    # from the real segment boundaries + downbeats, so every cut lands ON the song structure. The LLM
    # then only fills each fixed window's content (it never chooses timing). Falls back to free-timing
    # generation if there's no audio / analyze is down.
    grid = None
    seg_count = 0
    audio_id = body.get("audio_id")
    if ANALYZE_HOST and audio_id:
        ap = _lib_source_path(audio_id)
        if ap:
            try:
                a = analyze_py.analyze(ANALYZE_HOST, ap, with_tags=False, with_key=False)
                segs = a.get("segments") or []
                if segs:
                    total = max((float(s.get("end") or 0) for s in segs), default=0.0)
                    grid = musicvideo_mod.build_shot_grid(segs, a.get("downbeats") or [], total)
                    seg_count = len(segs)
            except Exception as e:
                print(f"[mv/script] structure analyze failed, falling back to free timing: {e}")
    try:
        if grid:
            shots = musicvideo_mod.generate_script_grid(song, cast, provider, model, claude_model, grid)
        else:
            shots = musicvideo_mod.generate_script(song, cast, provider, model, claude_model, n_shots)
    except Exception as e:
        raise HTTPException(500, f"script generation failed: {e}")
    return {"shots": shots, "song_title": song.get("title"),
            "structure_driven": bool(grid), "audio_segments": seg_count, "shots_count": len(shots),
            "duration": sum(int(s.get("seconds") or 0) for s in song.get("sections", []))}


@app.post("/api/mv/h3_script")
def mv_h3_script(body: dict):
    """MiniMax H3 HYBRID script (docs/MINIMAX_H3_PLAN.md Phase 3/4): the audio-structure grid is
    merged into render SEGMENTS on H3's frame grid (5.17-15.08s, never crossing a section
    boundary); the writer LLM picks "single" (one shot - always for lip-sync) or "scene" (2-4
    timestamped cuts in one render) per segment and fills creative content; code compiles each
    segment's six-section full-references prompt (subject-swap, pace anchors, sky-pin, doubled
    framing, direct-audio-reuse on lip-sync). STRUCTURE-DRIVEN ONLY: requires audio_id (the H3
    segment grid is built from the real audio structure; there is no free-timing fallback).
    Body: {song? OR project?, cast?: [{name, look, ...}], audio_id, provider?, model?}.
    Returns segments each carrying: window (start/end/seconds), render_seconds/frames, kind,
    shots, lipsync, compiled `prompt`, `picture_map` (character -> <Picture N>) and `env_picture`
    - the dispatcher uploads refs in exactly that order and, for lip-sync segments, passes
    ref_audio_ids [{id: audio_id, start: seg.start, seconds: seg.render_seconds}]."""
    song = body.get("song")
    if not song and body.get("project"):
        r = _resolve_project(body["project"])
        if not r:
            raise HTTPException(404, "project not found")
        song = _project_song_view(json.loads(r["data"] or "{}"))
    if not song or not song.get("sections"):
        raise HTTPException(400, "provide a song with sections, or a project key with a Song arrangement")
    audio_id = body.get("audio_id")
    ap = _lib_source_path(audio_id) if audio_id else None
    if not (ANALYZE_HOST and ap):
        raise HTTPException(400, "audio_id (a library track) is required - H3 segments are built "
                                 "from the real audio structure")
    provider = body.get("provider") or llm_mod.best_provider()
    model = body.get("model") or ""
    if not model and provider in ("claude_sub", "claude_code", "claude"):
        model = "claude-sonnet-5"          # see /api/mv/script for why Sonnet 5 over 4.6
    claude_model = CFG.get("claude_model", "claude-3-5-sonnet-latest")
    cast = body.get("cast") or []
    try:
        a = analyze_py.analyze(ANALYZE_HOST, ap, with_tags=False, with_key=False)
        segs = a.get("segments") or []
        if not segs:
            raise ValueError("audio analysis returned no segments")
        total = max((float(s.get("end") or 0) for s in segs), default=0.0)
        # H3-tuned grid: ~4.8s windows (vs the LTX default 7s). With 7s windows NO pair fits
        # inside the measured 10.5s segment ceiling, so every segment degenerates to a single
        # window and the writer never gets to choose "scene" - the hybrid collapses (observed on
        # the first real script: 35/35 singles). ~4.8s windows let two-cut scenes (~9.6s) merge.
        grid = musicvideo_mod.build_shot_grid(segs, a.get("downbeats") or [], total,
                                              target=4.8, min_shot=2.8)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"audio structure analysis failed: {e}")
    # WHERE THE LYRICS ACTUALLY ARE. The Song tab's block seconds are what ACE-Step was asked for,
    # not what it delivered - measured drift of up to 7s on "Dream of Me" - and everything the
    # writer does with lyrics hangs off them. Measured once per track and cached; a failure here is
    # never fatal, the writer just falls back to mapping by section label and says so.
    align, align_note = None, "not attempted"
    try:
        align = lyricalign_mod.align_song(ap, song.get("sections") or [], cache_dir=LYRIC_CACHE)
        align_note = (f"measured ({align['cover'] * 100:.0f}% of lyric words placed)"
                      if align.get("cover", 0) >= musicvideo_mod.H3_ALIGN_MIN_COVER
                      else f"too thin to use ({align.get('cover', 0) * 100:.0f}% matched) - "
                           f"falling back to section labels")
    except Exception as e:
        align_note = f"failed ({e}) - falling back to section labels"
        print(f"[mv/h3_script] lyric alignment {align_note}")
    try:
        segments = musicvideo_mod.generate_h3_script_grid(song, cast, provider, model, claude_model,
                                                          grid, align=align)
    except Exception as e:
        raise HTTPException(500, f"h3 script generation failed: {e}")
    return {"segments": segments, "song_title": song.get("title"), "audio_id": audio_id,
            "segments_count": len(segments),
            "singles": sum(1 for s in segments if s["kind"] == "single"),
            "scenes": sum(1 for s in segments if s["kind"] == "scene"),
            # lip-sync shots recast to the voice the song's section markers call for
            "voice_fixes": sum(len(s.get("voice_fixed") or []) for s in segments),
            # whether the timeline came from MEASURING the vocals or from guessing by section label
            "lyric_alignment": align_note,
            "lyric_cover": (align or {}).get("cover")}


@app.post("/api/mv/h3_compile")
def mv_h3_compile(body: dict):
    """Recompile ONE H3 segment's full-references prompt after the user edits its structured
    fields (scene/action/framing/camera/costume/lipsync per shot, soundscape). Keeps every
    compiler-enforced rule intact (duration anchors, sky-pin, subject-swap, doubled framing,
    audio-reuse) - the safe alternative to hand-editing the compiled text.
    Body: {segment: <the segment object with edited shots>, cast: [<the same cast payload the
    script writer got>]}. Returns {prompt, picture_map, outfit_map, env_picture, lipsync}."""
    seg = body.get("segment") or {}
    cast = body.get("cast") or []
    shots_in = [s for s in (seg.get("shots") or []) if isinstance(s, dict)]
    if not shots_in or not seg.get("cuts"):
        raise HTTPException(400, "segment needs shots and cuts")
    # normalize enums the same way the parser does, so hand edits cannot smuggle bad values
    shots = []
    for s in shots_in[:4]:
        framing = str(s.get("framing") or "").strip().lower()
        if framing not in ("close", "medium", "wide"):
            framing = "medium"
        lipsync = bool(s.get("lipsync"))
        if lipsync and framing == "wide":
            framing = "medium"
        stype = s.get("type") or "broll"
        # instrument-performance-from-afar, same coercion as the script parser
        if stype == "performance" and not lipsync:
            framing = "wide" if "solo" in str(seg.get("section") or "").lower() else \
                      ("medium" if framing == "close" else framing)
        camera = str(s.get("camera") or "static").strip().lower()
        if camera not in musicvideo_mod.H3_CAMERA_MOVES:
            camera = "static"
        shots.append({"type": stype, "framing": framing, "lipsync": lipsync,
                      "camera": camera, "location": str(s.get("location") or "").strip(),
                      "scene": str(s.get("scene") or "").strip(),
                      "action": str(s.get("action") or "").strip(),
                      "costume": str(s.get("costume") or "").strip(),
                      # carried through on purpose: this whitelist rebuilds the shot, and dropping
                      # the lock here would hand an "unlocked" shot to enforce_voice_casting, which
                      # would then undo the user's hand casting on every recompile
                      "cast_locked": bool(s.get("cast_locked")),
                      # who performs the vocal, when it is not simply "everyone with a singer role"
                      "singers": [str(x).strip() for x in (s.get("singers") or []) if str(x).strip()],
                      "characters": [str(x).strip() for x in (s.get("characters") or []) if str(x).strip()]})
    lipsync_any = any(s["lipsync"] for s in shots)
    # lip-sync cuts are allowed inside scenes (2026-08-09); only one-window segments are single
    seg2 = {"start": float(seg.get("start") or 0), "end": float(seg.get("end") or 0),
            "cuts": seg.get("cuts"), "kind": seg.get("kind") or "single",
            "shots": shots, "lipsync": lipsync_any,
            "soundscape": str(seg.get("soundscape") or "").strip()}
    if seg2["kind"] == "single" or len(seg.get("cuts") or []) == 1:
        seg2["kind"] = "single"
        seg2["cuts"] = [{"start": seg2["start"], "end": seg2["end"]}]
        keep = next((s for s in shots if s["lipsync"]), shots[0])
        seg2["shots"] = [keep]
    miss = _missing_cast([seg2], cast)
    if miss:
        raise HTTPException(400, "not recompiling: these characters are in the shots but missing from "
                                 f"the cast payload - {', '.join(miss)}. Recompiling now would strip "
                                 "their identity references.")
    # voice-matched casting, same rule the script path enforces (needs the song's section markers -
    # the UI sends them; without a song payload the writer's/editor's casting stands)
    # section_grid = the whole video's [{start, end, section}], needed to map the arrangement's
    # nominal times onto the real audio; one segment cannot supply those anchors
    voice_fixes = musicvideo_mod.enforce_voice_casting([seg2], body.get("song") or {}, cast,
                                                      grid=body.get("section_grid") or None)
    try:
        prompt, refs = musicvideo_mod.compile_h3_prompt(seg2, cast, audio_ref=lipsync_any)
    except Exception as e:
        raise HTTPException(500, f"compile failed: {e}")
    return {"prompt": prompt, "picture_map": refs["sheets"], "outfit_map": refs["outfits"],
            "prop_map": refs["props"], "env_map": refs["envs"], "env_picture": refs["env"],
            "lipsync": lipsync_any, "shots": seg2["shots"], "kind": seg2["kind"], "cuts": seg2["cuts"],
            "voice_fixes": voice_fixes,
            # length against MiniMax's 7000-char ceiling - this endpoint rebuilds its own response,
            # so these have to be forwarded explicitly or the guard never reaches the caller
            "chars": refs.get("chars"), "over_limit": refs.get("over_limit")}


def _missing_cast(segments, cast):
    """Characters the shots name but the cast payload does not carry. Compiling in that state
    silently drops their identity references (a UI that fires before the character library has
    loaded once stripped every subject from 30 segments), so callers must refuse instead."""
    have = {c.get("name") for c in (cast or []) if c.get("name")}
    miss = []
    for s in segments or []:
        for sh in s.get("shots") or []:
            for n in sh.get("characters") or []:
                if n and n not in have and n not in miss:
                    miss.append(n)
    return miss


@app.post("/api/mv/h3_audio_check")
def mv_h3_audio_check(body: dict):
    """Score how closely each rendered take's own audio follows the song window it was given.

    H3 re-renders the audio rather than pasting the reference in, and on a bad seed it drifts, with
    the lip-sync following the drift. Assembly then lays the master song over the top and the mouths
    no longer match - which only shows up once the whole video is cut together, after the GPU time
    is already spent. Scoring it here makes a bad seed rejectable at the draft stage.

    Body: {clip_ids: [...], audio_id, start, seconds}. Returns {scores: {clip_id: 0..1 | null}}.
    Clean takes on this project score 0.87-1.00 and drifted ones 0.36-0.40; H3_AUDIO_MIN is the
    line between them. GPU-free, a second or so per clip."""
    ids = [os.path.basename(str(c)) for c in (body.get("clip_ids") or []) if c]
    if not ids:
        raise HTTPException(400, "clip_ids are required")
    song = _lib_source_path(body.get("audio_id"))
    if not song:
        raise HTTPException(400, "audio_id must be the song track this video is cut to")
    start = float(body.get("start") or 0)
    seconds = float(body.get("seconds") or 0) or None
    scores = {}
    for cid in ids[:12]:
        path = os.path.join(LIBRARY, f"{cid}.mp4")
        scores[cid] = (musicvideo_mod.audio_match(path, song, start, seconds)
                       if os.path.exists(path) else None)
    return {"scores": scores, "min_ok": musicvideo_mod.H3_AUDIO_MIN}


@app.post("/api/mv/h3_recompile")
def mv_h3_recompile(body: dict):
    """Recompile every segment whose stored prompt no longer matches what the compiler would emit
    today - the repair for a script written before a compiler rule changed (the no-singing clause,
    the reference budget, the band fill). Cheap and deterministic: no writer run, no GPU.

    HAND-EDITED prompts are left untouched and reported, since a bulk action must never silently
    throw away text the user wrote. Voice-matched casting is enforced first, so a refreshed prompt
    cannot be compiled around stale casting.
    Body: {segments, cast, song?, section_grid?}. Returns {segments, changed, skipped, voice_fixes}."""
    segs = body.get("segments") or []
    cast = body.get("cast") or []
    song = body.get("song") or {}
    if not segs:
        raise HTTPException(400, "segments are required")
    miss = _missing_cast(segs, cast)
    if miss:
        raise HTTPException(400, "not recompiling: these characters are in the shots but missing from "
                                 f"the cast payload - {', '.join(miss)}. Recompiling now would strip "
                                 "their identity references. If the page has just loaded, give the "
                                 "character library a moment and try again.")
    grid = body.get("section_grid") or segs
    voice_fixes = (musicvideo_mod.enforce_voice_casting(segs, song, cast, grid=grid)
                   if song.get("sections") else [])
    changed, skipped = [], []
    for i, seg in enumerate(segs, 1):
        if seg.get("handEdited"):
            skipped.append(i)
            continue
        try:
            prompt, refs = musicvideo_mod.compile_h3_prompt(seg, cast, audio_ref=seg.get("lipsync"))
        except Exception as e:
            raise HTTPException(500, f"segment {i} failed to compile: {e}")
        if prompt.strip() != (seg.get("prompt") or "").strip():
            seg["prompt"] = prompt
            seg["picture_map"], seg["outfit_map"] = refs["sheets"], refs["outfits"]
            seg["prop_map"], seg["env_map"], seg["env_picture"] = refs["props"], refs["envs"], refs["env"]
            changed.append(i)
    # a bulk recompile is exactly where an over-length prompt would slip through unnoticed
    over = [i for i, seg in enumerate(segs, 1)
            if len(seg.get("prompt") or "") > musicvideo_mod.H3_PROMPT_MAX]
    return {"segments": segs, "changed": changed, "skipped": skipped, "voice_fixes": voice_fixes,
            "over_limit": over}


@app.post("/api/mv/h3_snap_edges")
def mv_h3_snap_edges(body: dict):
    """Move segment boundaries onto vocal handovers that sit too close to an edge to cut, and
    recompile the segments whose window moved. This is the repair for "the singer changes half a
    second before the segment ends": a third shot there would be under a second, so the boundary
    moves instead and the neighbour absorbs the difference.

    A moved window invalidates that segment's render, so its clipId is cleared (kept in
    clipVariants, so the take is still reachable) rather than left pointing at a clip of the wrong
    length. Body: {segments, cast, song, section_grid?}. Returns {segments, moved, recompiled}."""
    segs = body.get("segments") or []
    cast = body.get("cast") or []
    song = body.get("song") or {}
    if not segs:
        raise HTTPException(400, "segments are required")
    if not song.get("sections"):
        raise HTTPException(400, "the song's sections are required - they say where the voice changes")
    miss = _missing_cast(segs, cast)
    if miss:
        raise HTTPException(400, "not snapping: these characters are in the shots but missing from the "
                                 f"cast payload - {', '.join(miss)}. The recompile that follows would "
                                 "strip their identity references.")
    # anchors come from the section labels, so take the grid BEFORE anything moves
    grid = body.get("section_grid") or [{"start": s.get("start"), "end": s.get("end"),
                                        "section": s.get("section")} for s in segs]
    moved = musicvideo_mod.snap_segment_edges_to_handovers(segs, song, grid=grid)
    recompiled = []
    for i, seg in enumerate(segs, 1):
        if not seg.pop("edge_snapped", False):
            continue
        if seg.get("clipId"):
            seg["clipVariants"] = list(dict.fromkeys((seg.get("clipVariants") or []) + [seg["clipId"]]))
            seg["clipId"] = None
            seg["staleClip"] = True
        try:
            prompt, refs = musicvideo_mod.compile_h3_prompt(seg, cast, audio_ref=seg.get("lipsync"))
        except Exception as e:
            raise HTTPException(500, f"segment {i} failed to compile after snapping: {e}")
        seg["prompt"] = prompt
        seg["picture_map"], seg["outfit_map"] = refs["sheets"], refs["outfits"]
        seg["prop_map"], seg["env_map"], seg["env_picture"] = refs["props"], refs["envs"], refs["env"]
        recompiled.append(i)
    return {"segments": segs, "moved": moved, "recompiled": recompiled}


def _lyric_align(audio_id, song):
    """Measured lyric timeline for a track, or None. Cached per track+model, so only the first call
    per song pays the transcription (~100s for 4 minutes on CPU). Never raises: an alignment we
    could not compute just means the caller maps by section label instead."""
    if not audio_id:
        return None
    ap = _lib_source_path(audio_id)
    if not ap:
        return None
    try:
        return lyricalign_mod.align_song(ap, (song or {}).get("sections") or [], cache_dir=LYRIC_CACHE)
    except Exception as e:
        print(f"[mv] lyric alignment unavailable: {e}")
        return None


@app.post("/api/mv/h3_voicemap")
def mv_h3_voicemap(body: dict):
    """Where each VOICE sings, on the real audio timeline - what the per-segment timeline strip in
    the editor draws so boundaries can be checked and nudged by ear.
    Pass `audio_id` to place the windows from the MEASURED vocals rather than by matching section
    labels - the strip then draws the handovers the writer actually used.
    Body: {song, section_grid?, audio_id?}. Returns {windows, anchors, source}."""
    song = body.get("song") or {}
    grid = body.get("section_grid") or None
    if not song.get("sections"):
        raise HTTPException(400, "the song's sections (with their style markers) are required")
    align = _lyric_align(body.get("audio_id"), song)
    anchors = musicvideo_mod.align_anchors(song, align)
    return {"windows": musicvideo_mod._voice_windows(song, grid, align),
            "anchors": anchors or (musicvideo_mod._time_anchors(song, grid) if grid else []),
            # so the UI can say which timeline it is drawing rather than implying measurement
            "source": "measured" if anchors else ("labels" if grid else "nominal"),
            "cover": (align or {}).get("cover")}


@app.post("/api/mv/h3_voicefix")
def mv_h3_voicefix(body: dict):
    """Recast every lip-sync shot in an EXISTING script to the voice its song section calls for,
    and recompile the prompts of the segments that changed. The deterministic repair for a script
    the writer miscast (observed: the two leads swapped for the first two verses) - seconds, versus
    a full rewrite. Body: {segments, song, cast}. Returns {segments, fixes, fixed_segments}."""
    segs = body.get("segments") or []
    cast = body.get("cast") or []
    song = body.get("song") or {}
    if not segs:
        raise HTTPException(400, "segments are required")
    if not (song.get("sections")):
        raise HTTPException(400, "the song's sections (with their style markers) are required - "
                                 "they are what says whose voice sings when")
    fixes = musicvideo_mod.enforce_voice_casting(segs, song, cast)
    fixed = 0
    for seg in segs:
        if not seg.get("voice_fixed"):
            continue
        fixed += 1
        try:
            seg["prompt"], refs = musicvideo_mod.compile_h3_prompt(seg, cast, audio_ref=seg.get("lipsync"))
            seg["picture_map"], seg["outfit_map"] = refs["sheets"], refs["outfits"]
            seg["prop_map"], seg["env_map"], seg["env_picture"] = refs["props"], refs["envs"], refs["env"]
            seg["handEdited"] = False
        except Exception as e:
            raise HTTPException(500, f"recompile after recasting failed: {e}")
    return {"segments": segs, "fixes": fixes, "fixed_segments": fixed}


@app.get("/api/mv/grades")
def mv_grades():
    """Color-grade looks available for assembly (the Music Video tab picker)."""
    return {"grades": musicvideo_mod.grade_names()}


@app.post("/api/mv/assemble")
def mv_assemble(body: dict):
    """Stitch generated shot clips into one MP4 + the song audio (GPU-free, ffmpeg on the
    Mac). Body: {shots: [{clip_id, start, end}], audio_id?, title?, grade?, width?, height?}.
    width/height set the output canvas (default 1280x720; pass 2560x1440 for upscaled clips)."""
    segs = []
    for s in body.get("shots") or []:
        cid = os.path.basename(str(s.get("clip_id") or ""))
        clip = os.path.join(LIBRARY, f"{cid}.mp4")
        if cid and os.path.exists(clip):
            dur = float(s.get("end") or 0) - float(s.get("start") or 0)
            segs.append({"path": clip, "dur": dur if dur > 0 else 5.0, "cid": cid})
    if not segs:
        raise HTTPException(400, "no rendered shot clips found - generate the shots first")
    audio = _lib_source_path(body.get("audio_id"))
    grade = str(body.get("grade") or "none")
    width = int(body.get("width") or 1280)
    height = int(body.get("height") or 720)
    transition = float(body.get("transition") or 0)
    # optional intro pre-roll: a clip (the opening shot, rendered with LTX-native wind audio) plays
    # first with the song silent, then the song enters and the intro audio crossfades out.
    intro = None
    icid = os.path.basename(str(body.get("intro_clip_id") or ""))
    ipath = os.path.join(LIBRARY, f"{icid}.mp4")
    if icid and os.path.exists(ipath) and float(body.get("intro_dur") or 0) > 0:
        intro = {"path": ipath, "dur": float(body["intro_dur"]), "xfade": float(body.get("intro_xfade") or 1.5)}
        # pre-roll SOUND: an explicit library audio (intro_audio_id), else the bundled wind clip; the
        # assemble step level-matches it to the song before the crossfade.
        wind = _lib_source_path(body.get("intro_audio_id")) if body.get("intro_audio_id") else None
        if not wind:
            cand = os.path.join(LIBRARY, "wind_howling_ccby.mp3")
            wind = cand if os.path.exists(cand) else None
        if wind:
            intro["audio"] = wind
        # if the intro IS the opening shot's clip (rendered long for the wind head), the body's first
        # shot must CONTINUE from where the intro stopped, not replay the head -> offset it by the
        # intro duration. This is the fix for the restart-jump at the wind-fade.
        if segs and segs[0].get("cid") == icid:
            segs[0]["ss"] = float(body["intro_dur"])
    jid = uuid.uuid4().hex
    out = os.path.join(LIBRARY, f"{jid}.mp4")
    try:
        musicvideo_mod.assemble(segs, audio, out, width=width, height=height, grade=grade,
                                transition=transition, intro=intro,
                                # "crop" fills the canvas instead of padding to it - what you want
                                # for YouTube, which tells you never to upload baked-in bars
                                fit=("crop" if str(body.get("fit") or "") == "crop" else "pad"))
    except Exception as e:
        raise HTTPException(500, f"assembly failed: {e}")
    save_done_row(jid, "musicvideo", {"shots": len(segs), "title": (body.get("title") or "music video"), "grade": grade, "transition": transition}, out)
    return {"job_id": jid, "media_url": f"/api/media/{jid}", "status": "done"}


# ---------------- YouTube static video (cover art + song -> upload-ready MP4) ----------------
# The YouTube tab: Krea2 renders cover-art candidates WITH the song title as literal text in the
# image (the Ideogram4 builder's type="text" elements), the user picks one, and ffmpeg muxes the
# pick with the full track into a static video ready to upload (musicvideo.still_video).
YT_COVER_W, YT_COVER_H = 1920, 1080


@app.post("/api/youtube/concepts")
def youtube_concepts(body: dict):
    """Cover-art concepts authored by the LLM from the song itself (title + tags + lyrics).
    Body: {song?: {title,tags,sections[{lyrics}]}, OR project? (key), title?, n?, provider?,
    model?}. Returns {concepts: [{name, overview, background, aesthetics, lighting,
    palette[], title_style}], provider} - each concept plugs straight into /api/youtube/cover."""
    song = body.get("song")
    if not song and body.get("project"):
        r = _resolve_project(body["project"])
        if r:
            song = _project_song_view(json.loads(r["data"] or "{}"))
    song = song or {}
    title = str(body.get("title") or song.get("title") or "").strip() or "Untitled"
    tags = str(song.get("tags") or body.get("tags") or "").strip()
    lyrics = "\n".join((s.get("lyrics") or "").strip()
                       for s in (song.get("sections") or []) if (s.get("lyrics") or "").strip())[:1500]
    # optional freeform visual direction (the tab's "visual notes" + the Music 3 song brief)
    notes = str(body.get("notes") or "").strip()[:1200]
    n = max(2, min(6, int(body.get("n") or 4)))
    provider = body.get("provider") or llm_mod.best_provider()
    model = body.get("model") or ""
    if not model and provider in ("claude_sub", "claude_code", "claude"):
        model = "claude-sonnet-5"          # see /api/mv/script for why Sonnet 5
    system = ("You are an art director designing ALBUM COVER ART for a static YouTube music "
              "video. It must be a gorgeous, richly detailed cinematic image at full size - "
              "the viewer stares at it for the whole song - AND work as a dramatic hook "
              "that makes people click when it appears as the thumbnail: one bold subject, "
              "high stakes or scale, strong contrast, a clear focal point that reads "
              "instantly. Drama is the emphasis, not a small-size design constraint. "
              "Output STRICT JSON ONLY (no prose, no markdown fences). The art is rendered "
              "by a photoreal image model from rich prose fields; the song title is typeset "
              "into the image separately, so describe the SCENE, not the lettering "
              "(lettering style goes only in title_style).")
    prompt = f"""Song title: "{title}"
Genre/mood tags: {tags or "(none given)"}
Lyrics (may be empty):
{lyrics or "(none)"}
Visual direction from the user (if present, weigh it heavily - it describes what the art
should show):
{notes or "(none)"}

Write {n} DISTINCT cover-art concepts for this song, each a different visual direction
(e.g. one literal scene from the lyrics, one symbolic/iconic object, one landscape/mood,
one character-driven). EVERY concept must be a DRAMATIC CLICK-HOOK: a frozen peak moment
(mid-strike, mid-collapse, mid-ignition), extreme scale or peril, weather and light doing
something violent or uncanny - never a calm generic mood piece. One dominant subject with
empty-ish space in the upper band where the title will sit. Photoreal, cinematic, no text
or letters described in the scene itself.
WHERE IT FITS THE SONG - a female singer (see the tags), a female character or persona in
the lyrics, or a romantic/seductive/tragic-heroine theme - make SOME of the concepts (not
all) center a strikingly beautiful, alluring woman as the dominant subject: gothic glamour,
sensual and provocative in the album-cover tradition of the genre (bared shoulders, clinging
or torn couture, wind-caught hair, commanding or smoldering gaze), while keeping it
dramatic and tasteful - seductive power, never explicit. Songs with no such angle keep
their concepts subject-appropriate instead.
Return ONLY a JSON array of {n} objects, each:
{{"name": "<2-4 word label>",
  "overview": "<2-4 flowing sentences describing the whole cover image - subject, scene, composition, camera/lens, atmosphere. Rich, concrete prose.>",
  "background": "<1-3 sentences on the setting/backdrop with real materials, depth layers, believable light falloff>",
  "aesthetics": "<comma phrases: photoreal/cinematic/texture cues; avoid 8k/masterpiece gloss>",
  "lighting": "<one sentence of motivated lighting>",
  "palette": ["#RRGGBB", 3-5 hex colors for the whole image],
  "title_style": "<one sentence: the title lettering's typography, material and color so it stays legible over this scene>",
  "features_woman": <true when the concept's dominant subject is a woman, else false>}}"""
    claude_model = CFG.get("claude_model", "claude-3-5-sonnet-latest")
    try:
        text = llm_mod.complete(provider, model, system, prompt, claude_model, timeout=300)
        m = re.search(r"\[.*\]", text, re.S)
        raw = json.loads(m.group(0) if m else text)
    except Exception as e:
        raise HTTPException(500, f"concept generation failed ({provider}): {e}")
    concepts = []
    for c in raw if isinstance(raw, list) else []:
        if not isinstance(c, dict):
            continue
        concepts.append({
            "name": str(c.get("name") or f"Concept {len(concepts) + 1}").strip(),
            "overview": str(c.get("overview") or "").strip(),
            "background": str(c.get("background") or "").strip(),
            "aesthetics": str(c.get("aesthetics") or "").strip(),
            "lighting": str(c.get("lighting") or "").strip(),
            "palette": [str(x) for x in (c.get("palette") or []) if str(x).startswith("#")][:5],
            "title_style": str(c.get("title_style") or "").strip(),
            "features_woman": bool(c.get("features_woman")),
        })
    concepts = [c for c in concepts if c["overview"]][:n]
    if not concepts:
        raise HTTPException(500, "the LLM returned no usable concepts - try again")
    return {"concepts": concepts, "provider": provider}


@app.post("/api/youtube/cover")
def youtube_cover(p: dict):
    """Render ONE cover-art candidate with the song title typeset INTO the image.
    Krea2-only: the title is a type="text" element on the Ideogram4 builder path (the
    layout param), which is also the only Krea2 path proven safe on this stack - so the
    engine is forced to krea2 regardless of the still_engine default. p: {title,
    artist?, concept: {overview, background?, aesthetics?, lighting?, palette?,
    title_style?}, seed?, width?, height?, two_pass?}. Returns the async still job
    ({job_id, seed, media_url}) - poll /api/job/{job_id}."""
    title = str(p.get("title") or "").strip()
    if not title:
        raise HTTPException(400, "title is required (it is rendered into the artwork)")
    c = p.get("concept") or {}
    overview = str(c.get("overview") or "").strip()
    if not overview:
        raise HTTPException(400, "concept.overview is required (Krea2 needs rich global fields)")
    # ref_still_id = the COVER MODEL: render via Krea 2 Identity Edit so the woman in the
    # art is the same woman every time (face/body from the reference; clothing, pose and
    # scene from the concept). Only meaningful for concepts that feature her.
    if p.get("ref_still_id"):
        instruction = (f"Re-stage the woman from the reference into a completely different scene: "
                       f"{overview} {str(c.get('background') or '').strip()} "
                       f"Change her clothing, pose and surroundings to fit that scene; keep her face, "
                       f"hair and body exactly as they are in the reference. "
                       f"{str(c.get('lighting') or '').strip()} "
                       f"Leave the upper band of the frame free of any large object.")
        # two_pass = the 4-step refine stage, DEFAULT ON to match the plain-cover combo path.
        # A/B on a held seed [MEASURED 2026-08-25]: refine keeps identity AND composition while
        # sharpening detail; the Krea2T enhancer restructures the scene on this path, so it
        # stays off here permanently.
        req = {"ref_id": p["ref_still_id"], "prompt": " ".join(instruction.split()),
               "two_pass": bool(p.get("two_pass", True))}
        if p.get("seed") is not None:
            req["seed"] = int(p["seed"])
        return video_krea2_edit(req)
    background = str(c.get("background") or "").strip() or overview
    tstyle = (str(c.get("title_style") or "").strip()
              or "bold weathered metal album lettering with high contrast against the scene")
    regions = [
        # full-canvas scene region (the verified rich-caption recipe, see H3Studio.envRequest)
        {"type": "obj", "text": "", "palette": [],
         "desc": background + " Rendered with true-to-life photographic detail.",
         "x": 0.02, "y": 0.02, "w": 0.96, "h": 0.96},
    ]
    # omit_title = the title will be STAMPED afterwards (deterministic typography, always
    # spelled right - in-model lettering held for short titles but misspelled the 5-word
    # "Angel of the Shattered Sky" in 6 of 6 renders [OBSERVED 2026-08-24]). The scene
    # concepts already reserve the upper band, so no text region is placed at all.
    if not p.get("omit_title"):
        # the title as LITERAL text: the Ideogram4 builder's type="text" element renders the
        # `text` string; `desc` carries the typography
        regions.append({"type": "text", "text": title, "palette": [],
                        "desc": f"the song title in large {tstyle}, centered, fully legible, correctly spelled",
                        "x": 0.10, "y": 0.07, "w": 0.80, "h": 0.16})
    # The band name is deliberately NOT rendered in-model: Krea2 misspelled the invented
    # band name in 5 of 6 renders (small text + non-dictionary word = the model "corrects"
    # it). It is stamped deterministically afterwards - see /api/youtube/stamp.
    layout = {
        "overview": overview,
        "background": background,
        "photo_style": "",
        "aesthetics": (str(c.get("aesthetics") or "").strip()
                       or "photorealistic, cinematic, dramatic high-contrast album-cover composition, "
                          "one bold focal subject, true-to-life detail, atmospheric"),
        "lighting": (str(c.get("lighting") or "").strip()
                     or "dramatic motivated lighting with deep soft shadows and a strong key on the subject"),
        "medium": "photograph",
        "palette": [str(x) for x in (c.get("palette") or [])][:5],
        "regions": regions,
    }
    req = {"engine": "krea2", "two_pass": bool(p.get("two_pass", True)), "layout": layout,
           "width": int(p.get("width") or YT_COVER_W), "height": int(p.get("height") or YT_COVER_H)}
    if p.get("seed") is not None:
        req["seed"] = int(p["seed"])
    return video_still(req)


@app.post("/api/youtube/render")
def youtube_render(body: dict):
    """Mux a picked cover still + a library track into an upload-ready static MP4
    (GPU-free ffmpeg on the Mac: 1 fps x264 stillimage + AAC 320k + faststart).
    Body: {image_id, audio_id, title?, res?: "1080p"|"4k"}. Synchronous - returns
    {job_id, media_url, status: "done"}; download via GET /api/export/{job_id}."""
    img = _lib_image_path(body.get("image_id") or "")
    if not img:
        raise HTTPException(400, "image_id must reference a generated still in the library")
    audio = _lib_source_path(os.path.basename(str(body.get("audio_id") or "")))
    if not audio:
        raise HTTPException(400, "audio_id must reference a library track")
    res = str(body.get("res") or "1080p").lower()
    w, h = (3840, 2160) if res in ("4k", "2160p") else (1920, 1080)
    title = str(body.get("title") or "").strip()
    jid = uuid.uuid4().hex
    out = os.path.join(LIBRARY, f"{jid}.mp4")
    try:
        musicvideo_mod.still_video(img, audio, out, width=w, height=h, title=title,
                                   artist=str(_yt_wordmark_cfg().get("text") or ""))
    except Exception as e:
        raise HTTPException(500, f"render failed: {e}")
    save_done_row(jid, "ytvideo", {"title": title or "youtube video", "res": res,
                                   "image_id": os.path.basename(str(body.get("image_id") or "")),
                                   "audio_id": os.path.basename(str(body.get("audio_id") or ""))}, out)
    return {"job_id": jid, "media_url": f"/api/media/{jid}", "status": "done"}


@app.post("/api/youtube/metadata")
def youtube_metadata(body: dict):
    """The upload text package for a finished video: title line, description (strong
    keyword-bearing first lines, credits, hashtags at the end) and a short tags list.
    Body: {title, artist?, song?: {title,tags,sections[{lyrics}]}, notes?, provider?, model?}.
    Returns {video_title, description, tags:[...]} - all paste-ready for YouTube Studio."""
    song = body.get("song") or {}
    title = str(body.get("title") or song.get("title") or "").strip()
    if not title:
        raise HTTPException(400, "title is required")
    artist = str(body.get("artist") or _yt_wordmark_cfg().get("text") or "").strip()
    tags = str(song.get("tags") or "").strip()
    lyrics = "\n".join((s.get("lyrics") or "").strip()
                       for s in (song.get("sections") or []) if (s.get("lyrics") or "").strip())[:1200]
    provider = body.get("provider") or llm_mod.best_provider()
    model = body.get("model") or ""
    if not model and provider in ("claude_sub", "claude_code", "claude"):
        model = "claude-sonnet-5"
    system = ("You write YouTube upload metadata for a band's official-audio music uploads. "
              "Output STRICT JSON ONLY (no prose, no markdown fences). You know how YouTube "
              "search works: the title carries 'Artist - Song (Official Audio)'; the first "
              "two description lines are indexed hardest and shown above the fold, so they "
              "carry the genre words, mood words and a 'for fans of X, Y' line naming 2-3 "
              "real bands the song genuinely resembles; hashtags go at the END of the "
              "description (the first three show above the video title - lead with the "
              "genre ones); the separate tags field mainly catches misspellings.")
    prompt = f"""Song: "{title}" by {artist or "an independent band"}.
Genre/mood tags: {tags or "(none given)"}
Lyrics excerpt (may be empty):
{lyrics or "(none)"}
Extra notes: {str(body.get("notes") or "").strip() or "(none)"}

Return ONLY a JSON object:
{{"video_title": "<{artist + ' - ' if artist else ''}{title} (Official Audio)> or a better variant, under 100 chars",
  "description": "<the full description: 2 strong opening lines with genre/mood keywords and a 'For fans of ...' line; a blank line; 1-2 short lines about the song's story drawn from the lyrics; a blank line; a credits line naming {artist or 'the band'}; then ONE final line of 4-6 hashtags, genre first (e.g. #symphonicmetal), no spaces inside a tag>",
  "tags": [<8-12 short strings for YouTube's tags field: the artist name, the song title, genre phrases, and 2-3 plausible misspellings of the artist name>]}}"""
    claude_model = CFG.get("claude_model", "claude-3-5-sonnet-latest")
    try:
        text = llm_mod.complete(provider, model, system, prompt, claude_model, timeout=240)
        m = re.search(r"\{.*\}", text, re.S)
        parsed = json.loads(m.group(0) if m else text)
    except Exception as e:
        raise HTTPException(500, f"metadata generation failed ({provider}): {e}")
    return {"video_title": str(parsed.get("video_title") or f"{artist} - {title} (Official Audio)").strip()[:100],
            "description": str(parsed.get("description") or "").strip(),
            "tags": [str(t).strip() for t in (parsed.get("tags") or []) if str(t).strip()][:15],
            "provider": provider}


# ---- band wordmark: deterministic Pillow lettering, never model-rendered (see backend/wordmark.py)
def _yt_wordmark_cfg():
    saved = CFG.get("yt_wordmark") or {}
    return {**wordmark_mod.DEFAULTS, **{k: v for k, v in saved.items() if v not in (None, "")}}


@app.get("/api/youtube/wordmark_options")
def youtube_wordmark_options():
    """The picker's vocabulary: available fonts (existence-checked), treatments, and the
    saved global wordmark choice (app_config.json `yt_wordmark`, merged over defaults)."""
    return {"fonts": wordmark_mod.available_fonts(),
            "treatments": list(wordmark_mod.TREATMENTS.keys()),
            "positions": wordmark_mod.POSITIONS,
            "current": _yt_wordmark_cfg()}


@app.get("/api/youtube/wordmark_preview")
def youtube_wordmark_preview(text: str = "", font: str = "luminari", treatment: str = "bone"):
    """One picker tile: the wordmark rendered on a dark card, returned as PNG directly
    (stateless - no library rows; the picker grid is just <img> tags on this URL)."""
    try:
        png = wordmark_mod.preview_png(text.strip() or _yt_wordmark_cfg()["text"], font, treatment)
    except Exception as e:
        raise HTTPException(500, f"wordmark preview failed: {e}")
    return Response(content=png, media_type="image/png")


@app.put("/api/youtube/wordmark")
def youtube_wordmark_save(body: dict):
    """Persist the chosen wordmark (text/font/treatment/position/scale) GLOBALLY in
    app_config.json - the band logo is choose-once, reused on every cover. Preserves
    unrelated config keys (same pattern as PUT /api/settings) and refreshes CFG so it
    applies without a restart."""
    allowed = {"text": str, "font": str, "treatment": str, "position": str, "scale": float,
               "title_font": str, "title_treatment": str, "title_position": str, "title_scale": float,
               "cover_model_still": str}   # the canonical cover-model reference still (library id)
    wm = _yt_wordmark_cfg()
    for k, cast in allowed.items():
        if k in body and body[k] is not None:
            try:
                wm[k] = cast(body[k])
            except (TypeError, ValueError):
                raise HTTPException(400, f"bad value for {k}")
    cur = {}
    if os.path.exists(_CFG_PATH):
        try:
            with open(_CFG_PATH) as f:
                cur = json.load(f)
        except Exception:
            cur = {}
    cur["yt_wordmark"] = wm
    with open(_CFG_PATH, "w") as f:
        json.dump(cur, f, indent=2)
    CFG["yt_wordmark"] = wm
    return {"ok": True, "wordmark": wm}


@app.post("/api/youtube/stamp")
def youtube_stamp(body: dict):
    """Stamp text layers onto a cover still -> a NEW library still (the original is kept).
    Always stamps the band wordmark; when `title` is given, the song TITLE is stamped too
    (large, header band by default) - the deterministic alternative to in-model lettering,
    which misspells long titles. Body: {image_id, title?, text?, font?, treatment?,
    position?, scale?, title_font?, title_treatment?, title_position?, title_scale?} -
    omitted fields fall back to the saved global config. Synchronous (Pillow on the Mac)."""
    img = _lib_image_path(body.get("image_id") or "")
    if not img:
        raise HTTPException(400, "image_id must reference a generated still in the library")
    wm = _yt_wordmark_cfg()
    for k in ("text", "font", "treatment", "position", "scale",
              "title_font", "title_treatment", "title_position", "title_scale"):
        if body.get(k) not in (None, ""):
            wm[k] = body[k]
    if not str(wm.get("text") or "").strip():
        raise HTTPException(400, "wordmark text is empty - set the band name")
    layers = []
    song_title = str(body.get("title") or "").strip()
    if song_title:
        layers.append({"text": song_title, "font": wm.get("title_font") or "unifraktur",
                       "treatment": wm.get("title_treatment") or "ember",
                       "position": wm.get("title_position") or "header",
                       "scale": float(wm.get("title_scale") or 0.72), "max_h": 0.22})
    layers.append({"text": str(wm["text"]).strip(), "font": str(wm["font"]),
                   "treatment": str(wm["treatment"]),
                   "position": str(wm.get("position") or "bottom"),
                   "scale": float(wm.get("scale") or 0.40)})
    jid = uuid.uuid4().hex
    out = os.path.join(LIBRARY, f"{jid}.png")
    try:
        wordmark_mod.stamp_layers(img, out, layers)
    except Exception as e:
        raise HTTPException(500, f"wordmark stamp failed: {e}")
    save_done_row(jid, "videostill", {"prompt": f"stamped: {(song_title + ' / ') if song_title else ''}{wm['text']}",
                                      "source": os.path.basename(str(body.get("image_id") or "")),
                                      "wordmark": wm}, out)
    return {"job_id": jid, "media_url": f"/api/media/{jid}", "status": "done"}


@app.get("/api/media/{pid}")
def media(pid: str):
    """Serve a generated still/video by id with the right content-type."""
    pid = os.path.basename(pid)
    for ext, ct in _MEDIA_CT.items():
        path = os.path.join(LIBRARY, f"{pid}{ext}")
        if os.path.exists(path):
            return FileResponse(path, media_type=ct)
    raise HTTPException(404, "no media")


@app.post("/api/restyle")
async def restyle(file: UploadFile = File(None), params: str = Form(...), job_id: str = Form(None)):
    p = json.loads(params)
    if not (p.get("tags") or "").strip():
        raise HTTPException(400, "style tags are required to restyle toward a target style")
    ref, label, dur = await _resolve_edit_source(file, job_id)   # upload OR a library track (e.g. an import)
    try:
        graph, resolved = comfy.build_restyle(p, ref)
        res = submit_comfy(graph)
    except Exception as e:
        raise HTTPException(500, f"submit failed: {e}")
    if res.get("node_errors"):
        raise HTTPException(400, f"node errors: {res['node_errors']}")
    pid = res["prompt_id"]
    with LOCK:
        JOBS[pid] = _new_job(resolved, "restyle")
        JOBS[pid]["params"]["source"] = label
    save_job(pid)
    return {"job_id": pid, "seed": resolved["seed"]}


def _save_engine_audio(jid, file_ref):
    """Download one engine output into the library as <jid>.wav (+ tidy_ending). Returns path."""
    data = acestep_py.download(ACESTEP_HOST, file_ref)
    out = os.path.join(LIBRARY, f"{jid}.wav")
    with open(out, "wb") as f:
        f.write(data)
    try:
        fixed = postfx_mod.tidy_ending(out)
        if fixed and fixed != out:
            os.remove(out)
            out = fixed
    except Exception:
        pass
    return out


def _acestep_ensure_model(model):
    """Make sure the engine has `model` loaded before submitting. `/release_task` does
    NOT auto-load a requested model — if it isn't in a loaded slot the engine silently
    falls back to the primary (ace-step/ACE-Step-1.5 job_model_selection.py). So when the
    loaded DiT differs, swap it via POST /v1/init (keeps the LM). Best-effort: on failure
    we log and proceed (the engine will use whatever is loaded)."""
    if not (ACESTEP_HOST and model):
        return
    try:
        cur = (acestep_py.health(ACESTEP_HOST).get("data") or {}).get("loaded_model", "")
    except Exception:
        return
    if cur == model:
        return
    base = ACESTEP_HOST if ACESTEP_HOST.startswith("http") else f"http://{ACESTEP_HOST}"
    try:
        requests.post(base + "/v1/init", json={"model": model, "init_llm": False}, timeout=1800)
    except Exception as e:
        print(f"[acestep] model swap to {model} failed ({e}); using {cur}")
        return
    for _ in range(180):                       # poll health until the swap completes
        try:
            if (acestep_py.health(ACESTEP_HOST).get("data") or {}).get("loaded_model", "") == model:
                return
        except Exception:
            pass
        time.sleep(5)


def _acestep_poll(pid, task_id, mode, phases=1):
    """Background: poll the official ACE-Step engine until the task finishes, then
    download EVERY batch take — the first becomes the tracked job (pid), the rest are
    saved as their own library rows ('take 2', 'take 3'…) under the same `mode` so they
    can be compared. Mirrors the ComfyUI job UX so the frontend's pollJob works unchanged.

    `phases` = how many 0→100% progress ramps the engine emits for this task (e.g. 2 when
    the 5Hz LM 'thinking' stage runs before DiT diffusion). We fold them into one monotonic
    bar (each phase = a 1/phases segment) so it climbs once instead of resetting per stage."""
    start = time.time()
    prog = {"phase": 0, "lastRaw": 0.0, "shown": 0.0}   # closure state for monotonic mapping

    def _on_progress(task):
        # Live progress for the bar. The engine emits a fresh "N/M" per stage, so detect a
        # reset (the parsed fraction dropping back) → advance to the next phase segment, and
        # never let the displayed value go backwards. JOBS progress/max → /api/job → frontend.
        raw = acestep_py.progress_fraction(task)
        if raw is not None:
            if raw + 0.15 < prog["lastRaw"] and prog["phase"] < phases - 1:
                prog["phase"] += 1                        # a stage finished and the next reset to ~0
            prog["lastRaw"] = raw
            cand = (prog["phase"] + raw) / max(1, phases)
        else:
            cand = min(0.92, (time.time() - start) / 75.0)  # no log yet → gentle time estimate
        prog["shown"] = min(0.98, max(prog["shown"], cand))  # monotonic, capped below 100
        with LOCK:
            if pid in JOBS:
                JOBS[pid]["progress"] = prog["shown"]
                JOBS[pid]["max"] = 1.0

    try:
        files = acestep_py.wait(ACESTEP_HOST, task_id, poll=3.0, on_progress=_on_progress)  # list, one ref per take
        out = _save_engine_audio(pid, files[0])
        with LOCK:
            base = dict(JOBS[pid]["params"])
            # 'take' is the song-lifetime take number assigned by save_job/save_done_row
            # (stable, never reused). The extras must NOT inherit the primary's number.
            base.pop("take", None)
            JOBS[pid]["audio_file"] = out
            JOBS[pid]["status"] = "done"
        save_job(pid)
        for i, fr in enumerate(files[1:], start=2):        # extra takes → their own library items
            try:
                jid = uuid.uuid4().hex
                o2 = _save_engine_audio(jid, fr)
                p2 = dict(base)
                save_done_row(jid, mode, p2, o2)
            except Exception:
                pass
    except Exception as e:
        with LOCK:
            JOBS[pid]["status"] = "error"
            JOBS[pid]["error"] = f"acestep {mode} failed: {e}"
        save_job(pid)


@app.post("/api/cover")
async def cover(file: UploadFile = File(None), params: str = Form(...),
                job_id: str = Form(None), timbre: UploadFile = File(None)):
    """Structure-preserving COVER (ACE-Step `cover` task): keep the source's
    structure/melody, change timbre/genre via tags (+ optional timbre clip).

    Routes to the OFFICIAL ACE-Step engine when `acestep_host` is set (proper
    `audio_cover_strength` — RESEARCH §13); otherwise falls back to the ComfyUI
    cover guider (weaker, no strength knob)."""
    p = json.loads(params)
    if not (p.get("tags") or "").strip():
        raise HTTPException(400, "style tags are required (describe the target sound/genre)")

    if ACESTEP_HOST and not p.get("force_comfy"):     # ----- official engine path -----
        if file is not None:
            data = await file.read()
            label = file.filename or "upload"
        elif job_id:
            src = next((os.path.join(LIBRARY, job_id + e) for e in (".wav", ".mp3", ".flac")
                        if os.path.exists(os.path.join(LIBRARY, job_id + e))), None)
            if not src:
                raise HTTPException(404, "source track not found")
            with open(src, "rb") as f:
                data = f.read()
            label = job_id
        else:
            raise HTTPException(400, "provide a source (library job_id or upload)")
        ctx = None
        if timbre is not None:
            tname = timbre.filename or "timbre.wav"
            ctx = (await timbre.read(), tname)
        # Param mapping mirrors /api/generate (cfg→guidance_scale, steps→inference_steps,
        # model/use_adg/infer_method/cfg_interval from the engine tuning UI). The LM is
        # auto-skipped for cover, so thinking/CoT don't apply here. Reimagine and Cover both
        # ride this `cover` task — reimagine just passes a lower audio_cover_strength.
        eng_model = p.get("model") or "acestep-v15-xl-sft"   # sft = the validated cover basis (melody retention)
        is_turbo = "turbo" in eng_model.lower()
        result_mode = p.get("result_mode", "cover")    # "cover" or "restyle" (reimagine) — library label
        fields = {
            "task_type": "cover",
            "prompt": p.get("tags", ""),
            "lyrics": "" if p.get("instrumental") else p.get("lyrics", ""),
            "audio_cover_strength": float(p.get("cover_strength", 0.5)),
            # Melody retention: 0 = pure style transfer (loses the tune), 0.1–0.25 = keep the
            # melody while changing style (validated by ear on Baby One More Time → metal).
            "cover_noise_strength": float(p.get("cover_noise_strength", 0.2)),
            "guidance_scale": float(p.get("cfg") if p.get("cfg") not in (None, "") else 8.0),
            "inference_steps": int(p.get("steps") or (8 if is_turbo else 32)),
            "shift": float(p.get("shift", 3.0)),
            "cfg_interval_start": float(p.get("cfg_interval_start", 0.0)),
            "cfg_interval_end": float(p.get("cfg_interval_end", 0.95)),
            "infer_method": p.get("infer_method", "ode"),
            "use_adg": bool(p.get("use_adg", not is_turbo)),
            "batch_size": int(p.get("batch_size", 2)),  # engine default is 2 takes/request
            "audio_format": "wav",
            "model": eng_model,
        }
        if p.get("seed"):
            fields["seed"] = int(p["seed"])
            fields["use_random_seed"] = False         # else the engine ignores `seed` and rolls its own
        try:
            free_gpu("ace")                           # free ComfyUI + RVC before the (offloaded) ACE engine generates
        except Exception:
            pass
        _acestep_ensure_model(eng_model)              # /release_task won't auto-load; swap if needed so the picked model is honored
        try:
            task_id = acestep_py.submit(ACESTEP_HOST, fields, src_audio=(data, label), ctx_audio=ctx)
        except Exception as e:
            raise HTTPException(500, f"acestep submit failed: {e}")
        pid = uuid.uuid4().hex
        resolved = {"engine": "acestep", "source": label, "tags": p.get("tags", ""),
                    "cover_strength": fields["audio_cover_strength"],
                    "cover_noise_strength": fields["cover_noise_strength"], "model": fields["model"],
                    "guidance_scale": fields["guidance_scale"], "steps": fields["inference_steps"]}
        with LOCK:
            JOBS[pid] = _new_job(resolved, result_mode)
            JOBS[pid]["status"] = "running"
        save_job(pid)
        threading.Thread(target=_acestep_poll, args=(pid, task_id, result_mode), daemon=True).start()
        return {"job_id": pid}

    # ----- fallback: ComfyUI cover guider -----
    ref, label, dur = await _resolve_edit_source(file, job_id)
    if dur:
        p["duration"] = dur                         # cover auto-locks length to the source (per ACE docs)
    timbre_ref = None
    if timbre is not None:
        tdata = await timbre.read()
        tname = timbre.filename if (timbre.filename or "").endswith(".wav") else ((timbre.filename or "timbre") + ".mp3")
        timbre_ref = C.upload_audio(tdata, tname)
    try:
        graph, resolved = comfy.build_cover(p, ref, timbre_ref=timbre_ref)
        res = submit_comfy(graph)
    except Exception as e:
        raise HTTPException(500, f"submit failed: {e}")
    if res.get("node_errors"):
        raise HTTPException(400, f"node errors: {res['node_errors']}")
    pid = res["prompt_id"]
    with LOCK:
        JOBS[pid] = _new_job(resolved, "cover")
        JOBS[pid]["params"]["source"] = label
    save_job(pid)
    return {"job_id": pid, "seed": resolved["seed"]}


async def _resolve_edit_source(file, job_id, trim_tail=False):
    """Source audio for repaint/extend: an uploaded file OR a library job_id.
    Uploads it to ComfyUI and returns (ref, label, duration_seconds).

    `trim_tail` (extend): strip the source's trailing fade-out / near-silence first,
    so new content attaches to LIVE music instead of after a fade (otherwise the join
    is "fade → silent gap → new content"). Our generations end with a tidy fade, so
    extending them without this leaves an audible gap at the seam."""
    if file is not None:
        data = await file.read()
        label = file.filename
    elif job_id:
        src = next((os.path.join(LIBRARY, job_id + e) for e in (".wav", ".mp3", ".flac")
                    if os.path.exists(os.path.join(LIBRARY, job_id + e))), None)
        if not src:
            raise HTTPException(404, "source track not found in the library")
        with open(src, "rb") as f:
            data = f.read()
        label = job_id
    else:
        raise HTTPException(400, "provide a source track (library job_id or upload)")
    import io
    import soundfile as _sf
    dur = 0.0
    try:
        if trim_tail:
            import numpy as _np
            x, sr = _sf.read(io.BytesIO(data), always_2d=True)
            env = _np.abs(x).max(axis=1)
            pk = float(env.max()) or 1.0
            above = _np.where(env > pk * 0.04)[0]          # last sample of real signal
            if above.size:
                end = min(len(x), int(above[-1]) + int(0.03 * sr))   # keep a tiny tail
                x = x[:end]
            buf = io.BytesIO()
            _sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
            data = buf.getvalue()
            dur = len(x) / float(sr)
            label = (label or "edit_src").rsplit(".", 1)[0] + ".wav"
        else:
            dur = float(_sf.info(io.BytesIO(data)).duration)
    except Exception:
        try:
            dur = float(_sf.info(io.BytesIO(data)).duration)
        except Exception:
            dur = 0.0
    fname = label if label and label.endswith(".wav") else ((label or "edit_src") + ".mp3")
    ref = C.upload_audio(data, fname)
    return ref, label, dur


async def _source_bytes(file, job_id):
    """Raw bytes + label for an edit source (upload or library job_id) — for engine
    tasks that take a multipart src_audio (cover/repaint), vs the ComfyUI upload path."""
    if file is not None:
        return await file.read(), (file.filename or "src.wav")
    if job_id:
        src = next((os.path.join(LIBRARY, job_id + e) for e in (".wav", ".mp3", ".flac")
                    if os.path.exists(os.path.join(LIBRARY, job_id + e))), None)
        if not src:
            raise HTTPException(404, "source track not found")
        with open(src, "rb") as f:
            return f.read(), job_id
    raise HTTPException(400, "provide a source (library job_id or upload)")


def _submit_edit(p, ref, mode, label):
    if not (p.get("tags") or "").strip():
        raise HTTPException(400, "style tags are required (describe the new content)")
    try:
        graph, resolved = comfy.build_edit(p, ref)
        res = submit_comfy(graph)
    except Exception as e:
        raise HTTPException(500, f"submit failed: {e}")
    if res.get("node_errors"):
        raise HTTPException(400, f"node errors: {res['node_errors']}")
    pid = res["prompt_id"]
    with LOCK:
        JOBS[pid] = _new_job(resolved, mode)
        JOBS[pid]["params"]["source"] = label
    save_job(pid)
    return {"job_id": pid, "seed": resolved["seed"]}


@app.post("/api/repaint")
async def repaint(file: UploadFile = File(None), params: str = Form(...), job_id: str = Form(None)):
    """Regenerate a time range of an existing track, preserving the rest
    (ACEStep15NativeEditGuider). params JSON carries repaint_start/repaint_end (sec),
    tags/lyrics for the new content, + tuning."""
    p = json.loads(params)
    if not (p.get("tags") or "").strip():
        raise HTTPException(400, "style tags are required (describe the new content)")

    if ACESTEP_HOST and ACESTEP_REPAINT and not p.get("force_comfy"):  # ----- official engine repaint (off by default; engine repaint is weak) -----
        # Native `repaint` works on base/sft (no turbo-forcing, unlike the ComfyUI guider).
        # LM is auto-skipped for repaint, so thinking/CoT don't apply. Region preserved
        # outside [start,end]; new content follows tags/lyrics.
        data, label = await _source_bytes(file, job_id)
        eng_model = p.get("model") or "acestep-v15-xl-sft"
        is_turbo = "turbo" in eng_model.lower()
        fields = {
            "task_type": "repaint",
            "prompt": p.get("tags", ""),
            "lyrics": "" if p.get("instrumental") else p.get("lyrics", ""),
            "repainting_start": float(p.get("repaint_start", 0.0)),
            "repainting_end": float(p.get("repaint_end", -1)),   # -1 = to end of source
            # how much of the region to regenerate: higher = follows the prompt more / keeps
            # less of the original (engine default 0.5 "balanced" keeps a lot of the source).
            "repaint_strength": float(p.get("repaint_strength", 0.5)),
            "repaint_mode": p.get("repaint_mode", "balanced"),
            "guidance_scale": float(p.get("cfg") if p.get("cfg") not in (None, "") else 8.0),
            "inference_steps": int(p.get("steps") or (8 if is_turbo else 32)),
            "shift": float(p.get("shift", 3.0)),
            "cfg_interval_start": float(p.get("cfg_interval_start", 0.0)),
            "cfg_interval_end": float(p.get("cfg_interval_end", 0.95)),
            "infer_method": p.get("infer_method", "ode"),
            "use_adg": bool(p.get("use_adg", not is_turbo)),
            "batch_size": int(p.get("batch_size", 2)),
            "audio_format": "wav",
            "model": eng_model,
        }
        if p.get("seed"):
            fields["seed"] = int(p["seed"]); fields["use_random_seed"] = False
        try:
            free_gpu("ace")
        except Exception:
            pass
        _acestep_ensure_model(fields["model"])         # honor the picked model (no auto-load on /release_task)
        try:
            task_id = acestep_py.submit(ACESTEP_HOST, fields, src_audio=(data, label))
        except Exception as e:
            raise HTTPException(500, f"acestep submit failed: {e}")
        pid = uuid.uuid4().hex
        resolved = {"engine": "acestep", "source": label, "tags": p.get("tags", ""),
                    "repaint_start": fields["repainting_start"], "repaint_end": fields["repainting_end"],
                    "model": eng_model, "guidance_scale": fields["guidance_scale"], "steps": fields["inference_steps"]}
        with LOCK:
            JOBS[pid] = _new_job(resolved, "repaint")
            JOBS[pid]["status"] = "running"
        save_job(pid)
        threading.Thread(target=_acestep_poll, args=(pid, task_id, "repaint"), daemon=True).start()
        return {"job_id": pid}

    # ----- ComfyUI fallback (turbo-forced edit guider) -----
    ref, label, dur = await _resolve_edit_source(file, job_id)
    if dur and not p.get("duration"):
        p["duration"] = dur                         # repaint keeps the original length
    p.pop("extend_left", None); p.pop("extend_right", None)
    return _submit_edit(p, ref, "repaint", label)


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(None), params: str = Form("{}"), job_id: str = Form(None)):
    """Transcribe a source song's lyrics locally (Whisper on the Mac). Isolates the
    vocal first (Demucs) by default so dense/metal mixes transcribe cleanly. Returns
    {text, language, duration}. Used to auto-fill the cover/vocal lyrics field —
    transcribes the user's own file (not reproducing copyrighted lyrics from the web)."""
    p = json.loads(params or "{}")
    size = p.get("model_size", "small")
    isolate = p.get("isolate_vocal", True)
    iso_engine = "roformer" if p.get("isolate_engine") == "roformer" else "demucs"
    language = p.get("language") or None
    sid = uuid.uuid4().hex
    work = os.path.join(STEMS_DIR, sid)
    os.makedirs(work, exist_ok=True)
    try:
        if file is not None:
            ext = os.path.splitext(file.filename or "")[1] or ".wav"
            src = os.path.join(work, "src" + ext)
            with open(src, "wb") as f:
                f.write(await file.read())
        elif job_id:
            src = next((os.path.join(LIBRARY, job_id + e) for e in (".wav", ".mp3", ".flac")
                        if os.path.exists(os.path.join(LIBRARY, job_id + e))), None)
            if not src:
                raise HTTPException(404, "source track not found")
        else:
            raise HTTPException(400, "provide a file or job_id")
        target = src
        if isolate:
            try:
                if iso_engine == "roformer" and ROFORMER_HOST:
                    # cleaner vocal (box GPU) → better lyrics on dense mixes. NB: shares the
                    # 3090 with the ACE-Step engine; _separate frees ComfyUI but not the ACE
                    # engine, so this can be tight on VRAM — Demucs (Mac) is the safe default.
                    stem_files = _separate(src, work, engine="roformer", stems="all")
                    voc = next((pp for (name, pp) in stem_files if name == "vocals"), None)
                else:
                    files = stems_mod.separate(src, work, mode="vocals")   # Demucs vocal, Mac MPS
                    voc = next((f for f in files if os.path.basename(f).startswith("vocals")), None)
                if voc:
                    target = voc
            except Exception:
                pass                                                   # fall back to the full mix
        try:
            return asr_mod.transcribe(target, size=size, language=language)
        except Exception as e:
            raise HTTPException(500, f"transcription failed: {e}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


# NOTE: /api/extend (append-to-track) removed — "extend" is not a native ACE-Step
# task (official tasks: text2music/remix/repaint/lego/extract/complete; no append),
# so the community guider hack had unfixable seam/beat/length artifacts. Use Generate
# at a longer duration or the Song Constructor for longer songs. See RESEARCH.md §10j.
# (comfy.build_edit + _resolve_edit_source + postfx.close_seam_gap remain for Repaint.)


async def _layer_backing_bytes(file, job_id, track, clean_bed, bed_engine):
    """Backing audio as raw bytes for the engine lego path. With clean_bed, strips the
    layer's own instrument first (separate → recombine the rest) so the added part is the
    only instance of that instrument. Returns (bytes, label, stripped_stem_or_None)."""
    if not clean_bed:
        data, label = await _source_bytes(file, job_id)
        return data, label, None
    work = os.path.join(STEMS_DIR, uuid.uuid4().hex)
    os.makedirs(work, exist_ok=True)
    try:
        data, label = await _source_bytes(file, job_id)
        srcp = os.path.join(work, "src" + (".wav" if label.endswith(".wav") else ".mp3"))
        with open(srcp, "wb") as f:
            f.write(data)
        want = DEMUCS_STEM.get(track, "other")
        stem_files = _separate(srcp, work, engine=bed_engine, stems="all", demucs_mode="6stem")
        others = [pp for (name, pp) in stem_files if name not in (want, "instrumental")]
        if not others:
            raise HTTPException(500, f"clean-bed: no stems left after removing '{want}'")
        return postfx_mod.recombine(others, normalize=True), "cleanbed.wav", want
    finally:
        shutil.rmtree(work, ignore_errors=True)


async def _engine_lego(p, file, job_id, timbre, track):
    """Official-engine `lego`: add a named track over the backing. The engine auto-builds
    the instruction from track_name, keeps the LM on (thinking) so the new part gets an
    audio-code plan, and returns a full mix with the part baked in (duration locks to
    source). base/SFT only. Off by default (ACESTEP_LEGO)."""
    clean_bed = bool(p.get("clean_bed"))
    bed_engine = "roformer" if p.get("clean_bed_engine") == "roformer" else "demucs"
    backing_bytes, label, stripped = await _layer_backing_bytes(file, job_id, track, clean_bed, bed_engine)
    if stripped:
        p["clean_bed_stripped"] = stripped
    import io as _io
    import soundfile as _sf
    try:
        src_dur = float(_sf.info(_io.BytesIO(backing_bytes)).duration)   # lock output to the backing length
    except Exception:
        src_dur = float(p.get("duration") or 0)
    is_vocal = track in ("vocals", "backing_vocals")
    eng_model = p.get("model") or "acestep-v15-xl-base"   # lego needs a BASE DiT + the LM
    lstart = float(p.get("layer_start", 0.0))
    lend = float(p.get("layer_end") or -1)
    # Regional layer → "explicit" mask (0/1 from the repaint range, confines the part to
    # the window, keeps the rest); whole-track layer → "auto" (model decides).
    is_region = lstart > 0.05 or (lend > 0 and src_dur and lend < src_dur - 0.05)
    fields = {
        "task_type": "lego",
        "track_name": track,                              # engine builds "Generate the {TRACK} track…"
        "prompt": p.get("tags", ""),
        "global_caption": p.get("global_caption", "") or p.get("tags", ""),  # Global: full-song desc
        "lyrics": p.get("lyrics", "") if is_vocal else "",
        "instrumental": not is_vocal,
        "repainting_start": lstart,
        "repainting_end": lend,                              # -1 = full backing
        "chunk_mask_mode": "explicit" if is_region else "auto",
        "duration": src_dur if src_dur > 0 else None,        # lock output length to the backing
        "guidance_scale": float(p.get("cfg") if p.get("cfg") not in (None, "") else 8.0),
        "inference_steps": int(p.get("steps") or 32),
        "shift": float(p.get("shift", 3.0)),
        "infer_method": p.get("infer_method", "ode"),
        "cfg_interval_start": float(p.get("cfg_interval_start", 0.0)),
        "cfg_interval_end": float(p.get("cfg_interval_end", 0.95)),
        "use_adg": bool(p.get("use_adg", True)),
        "thinking": True,                                  # lego uses the LM (the key advantage)
        "audio_format": "wav",
        "model": eng_model,
        "batch_size": int(p.get("batch_size", 2)),
    }
    if p.get("global_caption"):
        fields["global_caption"] = p["global_caption"]
    if p.get("seed"):
        fields["seed"] = int(p["seed"]); fields["use_random_seed"] = False
    ref = None
    if timbre is not None:
        ref = (await timbre.read(), timbre.filename or "timbre.wav")
    try:
        free_gpu("ace")
    except Exception:
        pass
    _acestep_ensure_model(fields["model"])             # honor the picked model (no auto-load on /release_task)
    try:
        task_id = acestep_py.submit(ACESTEP_HOST, fields, src_audio=(backing_bytes, label), ref_audio=ref)
    except Exception as e:
        raise HTTPException(500, f"acestep lego submit failed: {e}")
    pid = uuid.uuid4().hex
    resolved = {"engine": "acestep", "track_name": track, "tags": p.get("tags", ""), "source": label,
                "model": eng_model, "clean_bed_stripped": p.get("clean_bed_stripped")}
    with LOCK:
        JOBS[pid] = _new_job(resolved, "layer")
        JOBS[pid]["status"] = "running"
    save_job(pid)
    phases = 2 if fields.get("thinking") else 1          # lego runs the LM stage too
    threading.Thread(target=_acestep_poll, args=(pid, task_id, "layer", phases), daemon=True).start()
    return {"job_id": pid}


@app.post("/api/layer")
async def layer(file: UploadFile = File(None), params: str = Form(...),
                job_id: str = Form(None), timbre: UploadFile = File(None)):
    """Add-a-Layer (ACE-Step `lego` task): generate a new named track
    (track_name = vocals/drums/bass/guitar/keyboard/strings/…) into a time region of
    an existing backing, preserving the rest. params JSON carries track_name,
    layer_start/layer_end (sec), tags (+lyrics for vocal layers), optional timbre.
    Base/SFT only (defaults xl_sft) — see RESEARCH §11a / §10j."""
    p = json.loads(params)
    if not (p.get("tags") or "").strip():
        raise HTTPException(400, "style tags are required (describe the layer to add)")
    track = p.get("track_name", "vocals")
    clean_bed = bool(p.get("clean_bed"))

    if ACESTEP_HOST and ACESTEP_LEGO and not p.get("force_comfy"):   # engine lego (off by default)
        return await _engine_lego(p, file, job_id, timbre, track)

    if clean_bed:
        # "By construction" clean layer: strip the layer's own instrument from the
        # backing FIRST, so after lego the added part is the ONLY instance of that
        # instrument → it isolates cleanly (post-hoc separation of a lego mix that
        # already contains the same instrument is unreliable — RESEARCH §12c).
        bed_engine = "roformer" if p.get("clean_bed_engine") == "roformer" else "demucs"
        work = os.path.join(STEMS_DIR, uuid.uuid4().hex)
        os.makedirs(work, exist_ok=True)
        if file is not None:
            data = await file.read()
            label = file.filename or "upload"
            srcp = os.path.join(work, "src" + (os.path.splitext(label)[1] or ".wav"))
            with open(srcp, "wb") as f:
                f.write(data)
        elif job_id:
            s = next((os.path.join(LIBRARY, job_id + e) for e in (".wav", ".mp3", ".flac")
                      if os.path.exists(os.path.join(LIBRARY, job_id + e))), None)
            if not s:
                raise HTTPException(404, "backing track not found")
            srcp = os.path.join(work, job_id + os.path.splitext(s)[1])
            shutil.copy(s, srcp)
            label = job_id
        else:
            raise HTTPException(400, "provide a backing (library job_id or upload)")
        import soundfile as _sf
        try:
            dur = float(_sf.info(srcp).duration)
        except Exception:
            dur = 0.0
        want = DEMUCS_STEM.get(track, "other")
        stem_files = _separate(srcp, work, engine=bed_engine, stems="all", demucs_mode="6stem")
        # recombine everything EXCEPT the layer's instrument (and the bonus full-mix
        # 'instrumental' stem) → a bed missing that instrument.
        others = [pp for (name, pp) in stem_files if name not in (want, "instrumental")]
        if not others:
            raise HTTPException(500, f"clean-bed: no stems left after removing '{want}'")
        try:
            bed_bytes = postfx_mod.recombine(others, normalize=True)
        except Exception as e:
            raise HTTPException(500, f"clean-bed recombine failed: {e}")
        ref = C.upload_audio(bed_bytes, "cleanbed.wav")
        shutil.rmtree(work, ignore_errors=True)
        p["clean_bed_stripped"] = want
    else:
        ref, label, dur = await _resolve_edit_source(file, job_id)
    if dur and not p.get("duration"):
        p["duration"] = dur                         # layer over the backing's full length
    if not p.get("layer_end"):
        p["layer_end"] = p.get("duration", dur or 60.0)
    timbre_ref = None
    if timbre is not None:
        tdata = await timbre.read()
        tname = timbre.filename if timbre.filename.endswith(".wav") else (timbre.filename + ".mp3")
        timbre_ref = C.upload_audio(tdata, tname)
    try:
        graph, resolved = comfy.build_lego(p, ref, timbre_ref=timbre_ref)
        res = submit_comfy(graph)
    except Exception as e:
        raise HTTPException(500, f"submit failed: {e}")
    if res.get("node_errors"):
        raise HTTPException(400, f"node errors: {res['node_errors']}")
    pid = res["prompt_id"]
    with LOCK:
        JOBS[pid] = _new_job(resolved, "layer")
        JOBS[pid]["params"]["source"] = label
    save_job(pid)
    return {"job_id": pid, "seed": resolved["seed"]}


# Map the 12 lego/extract track names onto the separators' stem sets.
# htdemucs_6s and BS-Roformer SW share the same 6 stems (vocals/bass/drums/
# guitar/piano|keyboard/other), so one map serves both.
DEMUCS_STEM = {"vocals": "vocals", "backing_vocals": "vocals", "drums": "drums",
               "percussion": "drums", "bass": "bass", "guitar": "guitar",
               "keyboard": "piano", "synth": "other", "strings": "other",
               "brass": "other", "woodwinds": "other", "fx": "other"}


def _roformer_available():
    return bool(ROFORMER_HOST)


def _separate(input_path, work, engine="demucs", stems="all", demucs_mode="6stem"):
    """Separate `input_path` into stems, returning [(stem_name, wav_path), ...].

    engine='demucs'  → Mac-side htdemucs / htdemucs_6s (fast, no GPU).
    engine='roformer'→ box-side BS-Roformer SW on the 3090 (SOTA, 6 stems). We free
                       ComfyUI's VRAM first so the separator isn't fighting the
                       ACE-Step models for the 3090's memory. `stems` may narrow the
                       returned set (e.g. 'guitar') to save bandwidth."""
    if engine == "roformer":
        if not ROFORMER_HOST:
            raise HTTPException(400, "roformer_host not set in app_config.json")
        try:
            free_gpu("roformer")           # free ComfyUI + RVC before the separator loads
        except Exception:
            pass
        send_path = input_path
        if not input_path.lower().endswith(".wav"):   # box separator needs WAV
            import soundfile as _sf
            x, sr = _sf.read(input_path, always_2d=True)
            send_path = os.path.join(work, "ssrc.wav")
            _sf.write(send_path, x, sr, subtype="PCM_16")
        with open(send_path, "rb") as f:
            data = f.read()
        try:
            res = roformer_py.separate(data, os.path.basename(send_path), ROFORMER_HOST, stems=stems)
        except Exception as e:
            raise HTTPException(502, f"roformer separation failed (is the box service running?): {e}")
        out = []
        for name, wav in res["stems"].items():
            p = os.path.join(work, f"{name}.wav")
            with open(p, "wb") as fh:
                fh.write(wav)
            out.append((name, p))
        return out
    files = stems_mod.separate(input_path, work, mode=demucs_mode)
    return [(os.path.splitext(os.path.basename(f))[0], f) for f in files]


@app.post("/api/layer/isolate")
async def layer_isolate(file: UploadFile = File(None), params: str = Form(...), job_id: str = Form(None)):
    """Isolate a named track from a mix (typically a layer render) as a standalone stem.
    method='demucs' (Mac-side htdemucs_6s, synchronous) or 'extract' (GPU, native ACE
    extract task). Optionally region-gate to the layer window [layer_start, layer_end].
    Saves a library item with mode 'layerstem'."""
    p = json.loads(params)
    track = p.get("track_name", "vocals")
    method = p.get("method", "demucs")
    start = float(p.get("layer_start", 0) or 0)
    end = float(p.get("layer_end", 0) or 0)
    gate = bool(p.get("gate", True))
    # The lego re-render scatters a distorted instrument across its own stem AND
    # 'other' (verified: a lego lead lands almost entirely in 'other'). When the
    # backing was cleaned of this instrument first (clean_bed), summing target+other
    # robustly recovers the whole added part regardless of which bucket it fell into.
    combine_other = bool(p.get("combine_other"))

    if method == "extract":                          # GPU: native ACE extract task
        ref, label, dur = await _resolve_edit_source(file, job_id)
        if dur and not p.get("duration"):
            p["duration"] = dur
        try:
            graph, resolved = comfy.build_extract(p, ref)
            res = submit_comfy(graph)
        except Exception as e:
            raise HTTPException(500, f"submit failed: {e}")
        if res.get("node_errors"):
            raise HTTPException(400, f"node errors: {res['node_errors']}")
        pid = res["prompt_id"]
        with LOCK:
            JOBS[pid] = _new_job(resolved, "layerstem")
            JOBS[pid]["params"].update({"source": label, "track_name": track,
                "gate": gate, "layer_start": start, "layer_end": end, "method": "extract"})
        save_job(pid)
        return {"job_id": pid, "seed": resolved["seed"]}

    # method in ('demucs','roformer'): synchronous separation (Mac or box GPU)
    engine = "roformer" if method == "roformer" else "demucs"
    sid = uuid.uuid4().hex
    work = os.path.join(STEMS_DIR, sid)
    os.makedirs(work, exist_ok=True)
    if file is not None:
        ext = os.path.splitext(file.filename)[1] or ".wav"
        inp = os.path.join(work, "input" + ext)
        with open(inp, "wb") as f:
            f.write(await file.read())
        src_label = file.filename
    elif job_id:
        src = next((os.path.join(LIBRARY, job_id + e) for e in (".wav", ".mp3", ".flac")
                    if os.path.exists(os.path.join(LIBRARY, job_id + e))), None)
        if not src:
            raise HTTPException(404, "source track not found")
        inp = os.path.join(work, job_id + os.path.splitext(src)[1])
        shutil.copy(src, inp)
        src_label = job_id
    else:
        raise HTTPException(400, "provide a file or job_id")
    want = DEMUCS_STEM.get(track, "other")
    stem_files = _separate(inp, work, engine=engine, stems="all")
    by = {name: pth for (name, pth) in stem_files}
    jid = uuid.uuid4().hex
    out = os.path.join(LIBRARY, f"{jid}.wav")
    if combine_other:
        sel = [by[k] for k in dict.fromkeys([want, "other"]) if k in by]  # dedup if want=='other'
        if not sel:
            raise HTTPException(500, f"{engine}: no '{want}'/'other' stems produced")
        if len(sel) == 1:
            shutil.copy(sel[0], out)
        else:
            with open(out, "wb") as fh:
                fh.write(postfx_mod.recombine(sel, normalize=False))
        stem_label = f"{want}+other"
    else:
        match = by.get(want) or by.get("other")
        if not match:
            raise HTTPException(500, f"{engine} stem '{want}' not produced")
        shutil.copy(match, out)
        stem_label = want
    if gate and end > start:
        try:
            postfx_mod.gate_region(out, start, end)
        except Exception:
            pass
    save_done_row(jid, "layerstem", {"source": src_label, "track_name": track,
        "method": engine, "stem": stem_label, "layer_start": start, "layer_end": end}, out)
    shutil.rmtree(work, ignore_errors=True)
    return {"job_id": jid, "audio_url": f"/api/audio/{jid}", "status": "done"}


@app.get("/api/job/{pid}")
def job(pid: str):
    with LOCK:
        j = JOBS.get(pid)
        if j:
            is_vid = j.get("mode") in VIDEO_MODES
            done_file = bool(j.get("audio_file"))
            return {"id": pid, "status": j["status"], "progress": j.get("progress", 0),
                    "max": j.get("max", 0), "error": j.get("error"),
                    "audio_url": (f"/api/audio/{pid}" if done_file and not is_vid else None),
                    "media_url": (f"/api/media/{pid}" if done_file and is_vid else None),
                    "kind": j.get("params", {}).get("kind")}
    # fall back to persisted library row (e.g. after a restart)
    with db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (pid,)).fetchone()
    if not row:
        raise HTTPException(404, "unknown job")
    is_vid = row["mode"] in VIDEO_MODES
    return {"id": pid, "status": row["status"], "error": row["error"],
            "audio_url": (f"/api/audio/{pid}" if row["audio"] and not is_vid else None),
            "media_url": (f"/api/media/{pid}" if row["audio"] and is_vid else None)}


@app.post("/api/cancel")
def cancel():
    C.interrupt()
    return {"ok": True}


@app.get("/api/library")
def library():
    # Return recent AUDIO and VIDEO separately so neither starves the other. A flat "LIMIT 500 over
    # everything" let a large video library (1000s of clips/stills) crowd ALL audio tracks off the list,
    # which broke the MV Studio song picker (the song couldn't be found -> no song -> no timeline/media).
    vmodes = ",".join("'%s'" % m for m in VIDEO_MODES)
    with db() as conn:
        vids = conn.execute(
            f"SELECT * FROM jobs WHERE status='done' AND mode IN ({vmodes}) ORDER BY created DESC LIMIT 600").fetchall()
        auds = conn.execute(
            f"SELECT * FROM jobs WHERE status='done' AND mode NOT IN ({vmodes}) ORDER BY created DESC LIMIT 600").fetchall()
    rows = sorted([*vids, *auds], key=lambda r: r["created"], reverse=True)
    out = []
    for r in rows:
        is_vid = r["mode"] in VIDEO_MODES
        out.append({"id": r["id"], "created": r["created"], "mode": r["mode"],
                    "params": json.loads(r["params"]),
                    "audio_url": (None if is_vid else f"/api/audio/{r['id']}"),
                    "media_url": (f"/api/media/{r['id']}" if is_vid else None),
                    "bucket": (r["bucket"] if "bucket" in r.keys() else "") or ""})
    return out


@app.post("/api/library/{jid}/bucket")
def set_library_bucket(jid: str, body: dict):
    """Move an item into a filing bucket ('' = auto by type, 'tests' = Tests)."""
    with db() as conn:
        conn.execute("UPDATE jobs SET bucket=? WHERE id=?", (body.get("bucket", ""), os.path.basename(jid)))
    return {"ok": True}


@app.delete("/api/library/{jid}")
def delete_library_item(jid: str):
    jid = os.path.basename(jid)  # guard against path traversal
    removed = []
    for ext in (".mp3", ".wav", ".flac") + MEDIA_EXTS:
        p = os.path.join(LIBRARY, jid + ext)
        if os.path.exists(p):
            os.remove(p)
            removed.append(ext)
    with db() as conn:
        conn.execute("DELETE FROM jobs WHERE id=?", (jid,))
    with LOCK:
        JOBS.pop(jid, None)
    return {"ok": True, "removed": removed}


# ---------------- Projects (saved per-page state bundles) ----------------
@app.get("/api/projects")
def projects_list():
    """List saved projects (metadata only — no data payload)."""
    with db() as conn:
        rows = conn.execute(
            "SELECT id,name,created,updated FROM projects ORDER BY updated DESC").fetchall()
    return [{"id": r["id"], "name": r["name"], "created": r["created"], "updated": r["updated"]}
            for r in rows]


@app.post("/api/projects")
def projects_create(body: dict):
    """Create a new project. Body: {name, data}. Returns the new id."""
    pid = uuid.uuid4().hex
    name = (body.get("name") or "Untitled project").strip() or "Untitled project"
    data = json.dumps(body.get("data") or {})
    now = time.time()
    with db() as conn:
        conn.execute("INSERT INTO projects(id,name,created,updated,data) VALUES(?,?,?,?,?)",
                     (pid, name, now, now, data))
    return {"id": pid, "name": name, "created": now, "updated": now}


@app.get("/api/projects/{pid}")
def projects_get(pid: str):
    pid = os.path.basename(pid)
    with db() as conn:
        r = conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
    if not r:
        raise HTTPException(404, "project not found")
    return {"id": r["id"], "name": r["name"], "created": r["created"],
            "updated": r["updated"], "data": json.loads(r["data"] or "{}")}


@app.put("/api/projects/{pid}")
def projects_save(pid: str, body: dict):
    """Save (overwrite) a project's data, optionally its name too."""
    pid = os.path.basename(pid)
    now = time.time()
    with db() as conn:
        r = conn.execute("SELECT name FROM projects WHERE id=?", (pid,)).fetchone()
        if not r:
            raise HTTPException(404, "project not found")
        name = (body.get("name") or r["name"]).strip() or r["name"]
        data = json.dumps(body.get("data") or {})
        conn.execute("UPDATE projects SET name=?,data=?,updated=? WHERE id=?", (name, data, now, pid))
    return {"id": pid, "name": name, "updated": now}


@app.patch("/api/projects/{pid}")
def projects_rename(pid: str, body: dict):
    pid = os.path.basename(pid)
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "name required")
    with db() as conn:
        cur = conn.execute("UPDATE projects SET name=?,updated=? WHERE id=?", (name, time.time(), pid))
        if cur.rowcount == 0:
            raise HTTPException(404, "project not found")
    return {"id": pid, "name": name}


@app.delete("/api/projects/{pid}")
def projects_delete(pid: str):
    pid = os.path.basename(pid)
    with db() as conn:
        conn.execute("DELETE FROM projects WHERE id=?", (pid,))
    return {"ok": True}


# ---- Programmatic song read/patch (so lyrics/tags/tempo can be wired into a
# project's Song-builder arrangement without a bespoke DB script each time).
# The Song builder restores from data.drafts on Open, so these read/write
# data.drafts.song (+ song.tuning) and mirror into the lifted data.song. ----
def _resolve_project(key: str):
    """Find a project by id OR (case-insensitive) name; newest wins on name."""
    key = os.path.basename(key or "")
    with db() as conn:
        r = conn.execute("SELECT * FROM projects WHERE id=?", (key,)).fetchone()
        if not r:
            r = conn.execute("SELECT * FROM projects WHERE lower(name)=lower(?) "
                             "ORDER BY updated DESC LIMIT 1", (key,)).fetchone()
    return r


def _project_song_view(data: dict) -> dict:
    """Normalized view of a project's song: tags/bpm/keyscale + numbered sections."""
    dr = data.get("drafts") or {}
    ds = dr.get("song") or {}
    tuning = dr.get("song.tuning") or {}
    lifted = data.get("song") or {}
    blocks = ds.get("blocks") or lifted.get("blocks") or []
    return {
        "title": ds.get("title") or lifted.get("title"),
        "tags": ds.get("tags") or lifted.get("tags") or "",
        "bpm": tuning.get("bpm") or lifted.get("bpm"),
        "keyscale": tuning.get("keyscale") or lifted.get("key"),
        "instrumental": bool(ds.get("instrumental")),
        "drive": ds.get("drive") or "compile",
        "sections": [{"index": i, "type": b.get("type"), "seconds": b.get("seconds"),
                      "lyrics": b.get("lyrics") or "", "style": b.get("style") or ""}
                     for i, b in enumerate(blocks)],
    }


@app.get("/api/projects/{key}/song")
def project_song_get(key: str):
    """Read a project's Song-builder arrangement (by id or name) so a caller can
    see the section structure before writing lyrics into it."""
    r = _resolve_project(key)
    if not r:
        raise HTTPException(404, "project not found")
    return {"id": r["id"], "name": r["name"], "song": _project_song_view(json.loads(r["data"] or "{}"))}


@app.post("/api/projects/{key}/song")
def project_song_patch(key: str, body: dict):
    """Patch a project's Song-builder arrangement (by id or name). Body (all
    optional): tags, bpm, keyscale, instrumental, drive, and `sections` = a list
    of {index, lyrics?, style?, type?, seconds?} patches (index = section number
    from the GET). Writes into data.drafts.song so the change loads when the user
    re-opens the project. Returns the updated normalized view."""
    r = _resolve_project(key)
    if not r:
        raise HTTPException(404, "project not found")
    data = json.loads(r["data"] or "{}")
    dr = data.setdefault("drafts", {})
    ds = dr.setdefault("song", {})
    lifted = data.setdefault("song", {})
    # Ensure drafts.song.blocks exists (the UI's source of truth) - seed from the
    # lifted song or leave as-is - and give every block an id + locked flag.
    blocks = ds.get("blocks")
    if not blocks:
        blocks = [{"type": b.get("type"), "seconds": b.get("seconds"),
                   "lyrics": b.get("lyrics") or "", "style": b.get("style") or ""}
                  for b in (lifted.get("blocks") or [])]
        ds["blocks"] = blocks
    for i, b in enumerate(blocks):
        b.setdefault("id", f"blk{i}")
        b.setdefault("locked", False)
        b.setdefault("style", "")
        b.setdefault("lyrics", "")
    for patch in (body.get("sections") or []):
        idx = patch.get("index", patch.get("i"))
        if idx is None or not (0 <= int(idx) < len(blocks)):
            continue
        b = blocks[int(idx)]
        if "lyrics" in patch:  b["lyrics"] = patch["lyrics"] or ""
        if "style" in patch:   b["style"] = patch["style"] or ""
        if patch.get("type"):  b["type"] = patch["type"]
        if patch.get("seconds"): b["seconds"] = patch["seconds"]
    if "tags" in body:
        ds["tags"] = body["tags"]; lifted["tags"] = body["tags"]
    if "instrumental" in body:
        ds["instrumental"] = bool(body["instrumental"])
    if "drive" in body:
        ds["drive"] = body["drive"]
    tuning = dr.setdefault("song.tuning", {})
    if body.get("bpm") is not None:
        tuning["bpm"] = str(body["bpm"]); lifted["bpm"] = body["bpm"]
    if body.get("keyscale"):
        tuning["keyscale"] = body["keyscale"]; lifted["key"] = body["keyscale"]
    lifted["blocks"] = [{"type": b["type"], "seconds": b["seconds"], "lyrics": b.get("lyrics") or ""}
                        for b in blocks]
    now = time.time()
    with db() as conn:
        conn.execute("UPDATE projects SET data=?, updated=? WHERE id=?",
                     (json.dumps(data), now, r["id"]))
    return {"id": r["id"], "name": r["name"], "updated": now, "song": _project_song_view(data)}


# ---- Programmatic music-video script read/patch (so I can INJECT shots/cast into a
# project's Music Video tab, marked to time, and the user edits them). The tab restores
# from data.drafts.musicvideo on Open, same as the Song builder. ----
def _normalize_shots(shots):
    out = []
    for s in shots or []:
        if not isinstance(s, dict):
            continue
        t = s.get("type")
        item = {"section": str(s.get("section") or ""),
                "start": int(float(s.get("start") or 0)),
                "end": int(float(s.get("end") or 0)),
                "type": t if t in ("performance", "narrative", "broll") else "broll",
                "scene": str(s.get("scene") or "").strip(),
                "action": str(s.get("action") or s.get("motion") or "").strip(),
                "costume": str(s.get("costume") or "").strip(),
                "characters": [str(x) for x in (s.get("characters") or []) if x],
                "lipsync": bool(s.get("lipsync"))}
        if s.get("clipId"):
            item["clipId"] = s["clipId"]
        out.append(item)
    out.sort(key=lambda x: (x["start"], x["end"]))
    for i, s in enumerate(out):
        s["idx"] = i
    return out


def _normalize_blocks(blocks):
    """Normalize MSR timeline blocks (the LTX-MSR-native editor unit). Permissive: coerces
    timing, keeps the ref/timeline/audio/render structures verbatim (the editor owns the
    block schema, so unknown fields are preserved for forward-compat)."""
    out = []
    for b in blocks or []:
        if not isinstance(b, dict):
            continue
        item = dict(b)
        item["start"] = float(b.get("start") or 0)
        item["end"] = float(b.get("end") or 0)
        item["kind"] = str(b.get("kind") or "msr")
        if not item.get("id"):
            item["id"] = uuid.uuid4().hex
        out.append(item)
    out.sort(key=lambda x: (x["start"], x["end"]))
    for i, b in enumerate(out):
        b["idx"] = i
    return out


def _project_video_view(data: dict) -> dict:
    mv = (data.get("drafts") or {}).get("musicvideo") or {}
    return {"cast": mv.get("cast") or [], "castIds": mv.get("castIds") or [],
            "shots": mv.get("shots") or [], "blocks": mv.get("blocks") or [],
            "audioId": mv.get("audioId"), "grade": mv.get("grade") or "none",
            "method": mv.get("method") or "auto"}


@app.get("/api/characters")
def characters_list():
    """The reusable character library (cast members), newest first."""
    with db() as conn:
        rows = conn.execute("SELECT id,name,data,updated FROM characters ORDER BY updated DESC").fetchall()
    out = []
    for r in rows:
        c = json.loads(r["data"] or "{}")
        c.update({"id": r["id"], "name": r["name"], "updated": r["updated"]})
        out.append(c)
    return out


@app.post("/api/characters")
def characters_upsert(body: dict):
    """Create or update a character. Body: {id?, name, role?, kind?, appearance?, refStillId?,
    refStillIds?, loraName?, method?, notes?, identity?, wardrobes?}. appearance = the free-text
    look description (drives still generation). identity = the clothing-agnostic core
    {faceRefId?, bodyRefId?, notes?}; wardrobes = per-video looks [{id,name,outfitPrompt,
    faceRefId,bodyRefId,sheetId?}] (each = an MSR ref pair). Returns the saved record."""
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "character name required")
    cid = os.path.basename(str(body.get("id") or "")) or uuid.uuid4().hex
    data = {k: body.get(k) for k in ("kind", "role", "gender", "refStillId", "refStillIds", "loraName",
            "method", "notes", "identity", "wardrobes", "appearance",
            # H3-era root+costume model (docs/MINIMAX_H3_PLAN.md "Outfit-layering test"):
            # sheetId = the canonical identity sheet (the ONLY face source, never re-rendered);
            # costumes = person-free garment stills layered at render time via H3's outfit Subject;
            # props = person-free instrument/object stills (same mechanic - pins e.g. the bassist's
            # bass to ONE design across renders instead of a fresh invention per render)
            "sheetId", "costumes", "props", "style")
            if body.get(k) is not None}
    now = time.time()
    with db() as conn:
        ex = conn.execute("SELECT created FROM characters WHERE id=?", (cid,)).fetchone()
        created = ex["created"] if ex else now
        conn.execute("REPLACE INTO characters(id,name,created,updated,data) VALUES(?,?,?,?,?)",
                     (cid, name, created, now, json.dumps(data)))
    return {"id": cid, "name": name, "updated": now, **data}


@app.delete("/api/characters/{cid}")
def characters_delete(cid: str):
    cid = os.path.basename(cid)
    with db() as conn:
        conn.execute("DELETE FROM characters WHERE id=?", (cid,))
    return {"ok": True}


def _character(cid):
    with db() as conn:
        r = conn.execute("SELECT id,name,data FROM characters WHERE id=?", (os.path.basename(cid),)).fetchone()
    if not r:
        return None
    c = json.loads(r["data"] or "{}")
    c.update({"id": r["id"], "name": r["name"]})
    return c


# ---- H3-era character asset generation (docs/MINIMAX_H3_PLAN.md "Outfit-layering test").
# Both endpoints wrap the VERIFIED Krea2 regional-layout templates so the UI never re-invents
# prompts: the identity SHEET is the Sheet-D 4-region recipe (identity block repeated in every
# region => panels cannot drift), the COSTUME still is the person-free dress-form recipe (no
# face is ever rendered => zero identity risk). Library still id == job id, so refs attach
# immediately and the image lands when the render completes.

def _char_sheet_request(identity, base_wear, seed):
    ident = identity.rstrip(". ")
    return {
        "engine": "krea2", "two_pass": True, "enhancer": True,
        "width": 1920, "height": 1088, "seed": seed,
        "layout": {
            "overview": ("A professional photoreal character model sheet of one single person shown in "
                         "four separate panels on one page, every panel showing the same person with an "
                         "identical face, identical hair and identical wardrobe"),
            "background": ("A seamless plain light-grey studio backdrop covering the whole page, even soft "
                           "shadowless studio lighting, clean model-sheet layout, the four panels clearly "
                           "separated by empty grey margins, no text and no logos anywhere"),
            "photo_style": ("85mm studio reference photography, sharp focus in every panel, uniform even "
                            "lighting across the whole page, deep depth of field, true-to-life skin texture"),
            "aesthetics": ("photorealistic, ultra-realistic, true-to-life detail, professional character "
                           "reference sheet, consistent identity across all panels"),
            "lighting": "even soft diffused studio light, identical in every panel, no dramatic shadows",
            "medium": "photograph",
            "regions": [
                {"desc": f"Full-body front view: {ident}, wearing {base_wear}. Standing straight facing "
                         f"the camera, arms relaxed at the sides, the whole figure visible from head to foot.",
                 "x": 0.02, "y": 0.03, "w": 0.28, "h": 0.94},
                {"desc": f"Head-and-shoulders close-up portrait only, cropped at mid-chest, the face large "
                         f"and filling most of the panel: the same person - {ident} - turned slightly to a "
                         f"three-quarter view, sharp focus on the face, neutral calm expression.",
                 "x": 0.345, "y": 0.03, "w": 0.31, "h": 0.58},
                {"desc": f"Full side profile head-and-shoulders view of the same person - {ident} - facing "
                         f"left, the profile line of the face sharply resolved.",
                 "x": 0.345, "y": 0.65, "w": 0.31, "h": 0.32},
                {"desc": f"Full-body left side profile view of the same person - {ident} - standing "
                         f"straight, the same height and posture, arms relaxed, seen from the side.",
                 "x": 0.70, "y": 0.03, "w": 0.28, "h": 0.94},
            ],
        },
    }


def _costume_request(desc, seed):
    # THREE-VIEW garment sheet (front / back / side), one render: a single front view leaves the
    # garment's back and sides unconstrained, so H3 invents them the moment the character turns.
    # Within-one-render panel consistency is the proven property of the regional layout (identity
    # sheets never contradict themselves across panels) - same mechanism, applied to the garment.
    d = desc.rstrip(". ")
    def panel(view):
        return (f"{view} view of the SAME single outfit - {d} - displayed on a headless tailor's "
                f"dress form, the identical garment in every panel, every seam, fold and fabric "
                f"texture sharply resolved. No person, no face, no mannequin head.")
    return {
        "engine": "krea2", "two_pass": True, "enhancer": True,
        "width": 1920, "height": 1088, "seed": seed,
        "layout": {
            "overview": (f"A professional photoreal garment reference sheet showing ONE single outfit - "
                         f"{d} - in three separate panels on one page: a front view, a back view and a "
                         f"side view, the identical garment on a headless dress form in each panel, "
                         f"no person present anywhere"),
            "background": ("A seamless plain warm-grey studio backdrop covering the whole page, even soft "
                           "diffused studio lighting, a clean fashion-catalogue layout with the three "
                           "panels clearly separated by empty margins, no text, no logos, no person"),
            "photo_style": "",
            "aesthetics": ("photorealistic, true-to-life fabric detail, professional garment reference "
                           "sheet, catalogue clarity, the same garment identical in all three panels"),
            "lighting": "even soft diffused studio light, identical in every panel, no dramatic shadows",
            "medium": "photograph",
            "regions": [
                {"desc": "Full-length FRONT " + panel("front"), "x": 0.02, "y": 0.03, "w": 0.30, "h": 0.94},
                {"desc": "Full-length BACK " + panel("back"),   "x": 0.35, "y": 0.03, "w": 0.30, "h": 0.94},
                {"desc": "Full-length SIDE " + panel("side profile"), "x": 0.68, "y": 0.03, "w": 0.30, "h": 0.94},
            ],
        },
    }


@app.post("/api/characters/{cid}/sheet")
def character_sheet(cid: str, body: dict):
    """Generate canonical IDENTITY SHEET candidates for a character (the Sheet-D recipe).
    Body: {identity? (defaults to the character's appearance text), base_wear?, drafts? (4),
    seed?}. Returns {drafts: [{job_id, seed}]} - the UI picks one and saves it as `sheetId`.
    The picked sheet becomes the character's ONLY face source; it is never re-rendered."""
    c = _character(cid)
    if not c:
        raise HTTPException(404, "character not found")
    identity = (body.get("identity") or c.get("appearance") or "").strip()
    if not identity:
        raise HTTPException(400, "describe the character's identity first (appearance)")
    base_wear = (body.get("base_wear") or "a plain fitted black long-sleeved top and simple "
                 "black trousers").strip()
    n = max(1, min(int(body.get("drafts") or 4), 6))
    base = int(body.get("seed") or 0) or random.randint(0, 2**31 - 1)
    drafts = []
    for i in range(n):
        r = video_still(dict(_char_sheet_request(identity, base_wear, base + i)))
        drafts.append({"job_id": r["job_id"], "seed": base + i})
    return {"drafts": drafts, "base_seed": base}


def _prop_request(desc, seed):
    # person-free product shot of an instrument/object on a stand - the canonical prop reference
    # (verified pattern: outfit test 2026-08-09; instruments are H3 "object" subjects)
    d = desc.rstrip(". ")
    return {
        "engine": "krea2", "two_pass": True, "enhancer": True,
        "width": 1024, "height": 1536, "seed": seed,
        "layout": {
            "overview": (f"A professional photoreal studio product photograph of {d}, displayed on a "
                         f"suitable black stand, no person present"),
            "background": ("A seamless plain warm-grey studio backdrop, even soft diffused studio "
                           "lighting, a clean catalogue presentation with the object centered and "
                           "nothing else in frame, no text, no logos, no person anywhere"),
            "photo_style": "",
            "aesthetics": ("photorealistic, true-to-life material detail, professional reference "
                           "photograph, catalogue clarity"),
            "lighting": "even soft diffused studio light, gentle falloff, no dramatic shadows",
            "medium": "photograph",
            "regions": [
                {"desc": f"{d}, fully visible, every detail of its shape, materials, hardware and "
                         f"finish sharply resolved. No person, no hands.",
                 "x": 0.10, "y": 0.03, "w": 0.80, "h": 0.94},
            ],
        },
    }


@app.post("/api/characters/{cid}/prop")
def character_prop(cid: str, body: dict):
    """Add a PROP (instrument, weapon, signature object) to a character: renders a person-free
    product still and attaches the prop entry immediately (stillId = first draft job id).
    Body: {name, desc, drafts? (2), seed?}. Same flow as /costume; the UI repicks between the
    drafts via the normal character save (prop.stillId)."""
    c = _character(cid)
    if not c:
        raise HTTPException(404, "character not found")
    name = (body.get("name") or "").strip()
    desc = (body.get("desc") or "").strip()
    if not desc:
        raise HTTPException(400, "describe the prop/instrument")
    n = max(1, min(int(body.get("drafts") or 2), 4))
    base = int(body.get("seed") or 0) or random.randint(0, 2**31 - 1)
    drafts = []
    for i in range(n):
        r = video_still(dict(_prop_request(desc, base + i)))
        drafts.append({"job_id": r["job_id"], "seed": base + i})
    entry = {"id": uuid.uuid4().hex, "name": name or "New prop", "desc": desc,
             "stillId": drafts[0]["job_id"], "created": time.time()}
    props = (c.get("props") or []) + [entry]
    with db() as conn:
        row = conn.execute("SELECT data FROM characters WHERE id=?", (c["id"],)).fetchone()
        data = json.loads(row["data"] or "{}")
        data["props"] = props
        conn.execute("UPDATE characters SET data=?, updated=? WHERE id=?",
                     (json.dumps(data), time.time(), c["id"]))
    return {"prop": entry, "drafts": drafts, "base_seed": base}


@app.post("/api/characters/{cid}/costume")
def character_costume(cid: str, body: dict):
    """Add a COSTUME to a character: renders the person-free garment still (headless dress form -
    no face involved, zero identity risk) and attaches the costume entry immediately (stillId =
    job id; the image lands when the render finishes). Body: {name, desc, drafts? (2), seed?}.
    Returns the costume entry with its draft jobs - the UI picks which still to keep via
    the normal character save (costume.stillId)."""
    c = _character(cid)
    if not c:
        raise HTTPException(404, "character not found")
    name = (body.get("name") or "").strip()
    desc = (body.get("desc") or "").strip()
    if not desc:
        raise HTTPException(400, "describe the outfit")
    n = max(1, min(int(body.get("drafts") or 2), 4))
    base = int(body.get("seed") or 0) or random.randint(0, 2**31 - 1)
    drafts = []
    for i in range(n):
        r = video_still(dict(_costume_request(desc, base + i)))
        drafts.append({"job_id": r["job_id"], "seed": base + i})
    entry = {"id": uuid.uuid4().hex, "name": name or "New outfit", "desc": desc,
             "stillId": drafts[0]["job_id"], "created": time.time()}
    costumes = (c.get("costumes") or []) + [entry]
    with db() as conn:
        row = conn.execute("SELECT data FROM characters WHERE id=?", (c["id"],)).fetchone()
        data = json.loads(row["data"] or "{}")
        data["costumes"] = costumes
        conn.execute("UPDATE characters SET data=?, updated=? WHERE id=?",
                     (json.dumps(data), time.time(), c["id"]))
    return {"costume": entry, "drafts": drafts, "base_seed": base}


@app.get("/api/projects/{key}/video")
def project_video_get(key: str):
    """Read a project's Music Video script (cast + timed shots)."""
    r = _resolve_project(key)
    if not r:
        raise HTTPException(404, "project not found")
    return {"id": r["id"], "name": r["name"], "video": _project_video_view(json.loads(r["data"] or "{}"))}


@app.post("/api/projects/{key}/video")
def project_video_patch(key: str, body: dict):
    """Write/inject a project's Music Video script. Body (all optional): cast, castIds, audioId,
    method, grade, blocks (REPLACE the MSR-block timeline - the LTX-MSR-native editor unit),
    shots (REPLACE the legacy shot list), add_shots (APPEND new shots). Blocks + shots are
    normalized + sorted by start time. The change loads when the user opens the project."""
    r = _resolve_project(key)
    if not r:
        raise HTTPException(404, "project not found")
    data = json.loads(r["data"] or "{}")
    mv = data.setdefault("drafts", {}).setdefault("musicvideo", {})
    for k in ("cast", "castIds", "audioId", "method", "grade"):
        if k in body:
            mv[k] = body[k]
    if "blocks" in body:
        mv["blocks"] = _normalize_blocks(body["blocks"])
    shots = body["shots"] if "shots" in body else (mv.get("shots") or [])
    if body.get("add_shots"):
        shots = list(shots) + list(body["add_shots"])
    mv["shots"] = _normalize_shots(shots)
    with db() as conn:
        conn.execute("UPDATE projects SET data=?,updated=? WHERE id=?",
                     (json.dumps(data), time.time(), r["id"]))
    return {"id": r["id"], "name": r["name"], "video": _project_video_view(data)}


@app.get("/api/rvc/voices")
def rvc_voices():
    return R.voices()


@app.get("/api/voices/search")
def voices_search(q: str, sort: str = "likes", limit: int = 25):
    try:
        return {"results": voices_mod.search(q, sort, min(limit, 50))}
    except Exception as e:
        raise HTTPException(502, f"search failed: {e}")


@app.get("/api/voices/repo")
def voices_repo(id: str):
    try:
        return {"voices": voices_mod.repo_voices(id)}
    except Exception as e:
        raise HTTPException(502, f"repo lookup failed: {e}")


@app.post("/api/voices/install")
def voices_install(body: dict):
    base = f"http://{CFG.get('rvc_python_host', '')}" if CFG.get("rvc_python_host") else ""
    if not base:
        raise HTTPException(400, "rvc-python host not configured (voice install requires rvc-python)")
    try:
        if body.get("url"):
            res = voices_mod.install_from_url(base, body["url"], body.get("name"))
        elif body.get("zip"):
            res = voices_mod.install_from_hf(base, body["repo"], zip=body["zip"], name=body.get("name"))
        else:
            res = voices_mod.install_from_hf(base, body["repo"], pth=body["pth"],
                                             index=body.get("index"), name=body.get("name"))
    except Exception as e:
        raise HTTPException(500, f"install failed (is the rvc-python server running on the PC?): {e}")
    return res


@app.post("/api/rvc/convert")
async def rvc_convert(file: UploadFile = File(...), params: str = Form(...)):
    p = json.loads(params)
    if not p.get("voice"):
        raise HTTPException(400, "no voice selected")
    free_gpu("rvc")                       # free ComfyUI before RVC runs (shared 3090)
    try:
        wav = R.convert(
            await file.read(), file.filename, p["voice"],
            transpose=int(p.get("transpose", 0)),
            f0_method=p.get("f0_method", "rmvpe"),
            index_rate=float(p.get("index_rate", 0.75)),
            rms_mix_rate=float(p.get("rms_mix_rate", 0.25)),
            protect=float(p.get("protect", 0.33)))
    except Exception as e:
        raise HTTPException(500, f"RVC convert failed: {e}")
    jid = uuid.uuid4().hex
    path = os.path.join(LIBRARY, f"{jid}.wav")
    with open(path, "wb") as f:
        f.write(wav)
    save_done_row(jid, "vocal", {"voice": p["voice"], "transpose": p.get("transpose", 0),
                                 "index_rate": p.get("index_rate", 0.75),
                                 "source": file.filename}, path)
    return {"job_id": jid, "audio_url": f"/api/audio/{jid}", "status": "done"}


@app.post("/api/stems/separate")
async def stems_separate(mode: str = Form("vocals"),
                         engine: str = Form("demucs"),
                         job_id: str = Form(None),
                         file: UploadFile = File(None)):
    """Split into stems. engine='demucs' (Mac, modes vocals/all/6stem) or
    'roformer' (box GPU, BS-Roformer SW — always its 6 stems)."""
    sid = uuid.uuid4().hex
    work = os.path.join(STEMS_DIR, sid)
    os.makedirs(work, exist_ok=True)
    if file is not None:
        ext = os.path.splitext(file.filename)[1] or ".wav"
        inp = os.path.join(work, "input" + ext)
        with open(inp, "wb") as f:
            f.write(await file.read())
    elif job_id:
        src = next((os.path.join(LIBRARY, job_id + e) for e in (".mp3", ".wav", ".flac")
                    if os.path.exists(os.path.join(LIBRARY, job_id + e))), None)
        if not src:
            raise HTTPException(404, "library track not found")
        inp = os.path.join(work, job_id + os.path.splitext(src)[1])
        shutil.copy(src, inp)
    else:
        raise HTTPException(400, "provide a file or job_id")
    try:
        produced = _separate(inp, work, engine=("roformer" if engine == "roformer" else "demucs"),
                             stems="all", demucs_mode=mode)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"separation failed: {e}")
    try:
        os.remove(inp)
    except Exception:
        pass
    files = [p for (_n, p) in produced]
    stems = [{"name": os.path.splitext(os.path.basename(f))[0],
              "url": f"/api/stem/{sid}/{os.path.basename(f)}"} for f in files]
    for f in files:  # register each stem in the library (Stems section)
        save_done_row(uuid.uuid4().hex, "stem",
                      {"source": job_id or (file.filename if file else "upload"),
                       "engine": engine,
                       "kind": os.path.splitext(os.path.basename(f))[0]}, f)
    return {"id": sid, "stems": stems}


CFG["helix_presets_dir"] = os.path.join(LIBRARY, "helix_presets")
_SF_DEFAULT = os.path.join(ROOT, "soundfonts", "eguitar_clean.sf2")
if not CFG.get("guitar_soundfont") and os.path.exists(_SF_DEFAULT):
    CFG["guitar_soundfont"] = _SF_DEFAULT
# Kontakt (for Shreddage DI): auto-detect the VST3; state captured via the editor.
if not CFG.get("kontakt_vst3_path"):
    for _k in ("/Library/Audio/Plug-Ins/VST3/Kontakt 7.vst3",
               "/Library/Audio/Plug-Ins/VST3/Kontakt 8.vst3"):
        if os.path.exists(_k):
            CFG["kontakt_vst3_path"] = _k
            break
KONTAKT_STATE = os.path.join(ROOT, "soundfonts", "kontakt_guitar.state")


def _kontakt_ready():
    return bool(CFG.get("kontakt_vst3_path") and os.path.exists(CFG["kontakt_vst3_path"])
                and os.path.exists(KONTAKT_STATE))


@app.get("/api/tone/presets")
def tone_presets():
    """Guitar-tone presets + engine availability (pedalboard, Helix Native, DI engines)."""
    d = postfx_mod.presets(CFG)
    d["guitar_soundfont"] = bool(CFG.get("guitar_soundfont") and os.path.exists(CFG["guitar_soundfont"]))
    d["kontakt_available"] = bool(CFG.get("kontakt_vst3_path") and os.path.exists(CFG["kontakt_vst3_path"]))
    d["kontakt_ready"] = _kontakt_ready()
    d["riff_genres"] = [{"id": k, "label": v["label"]} for k, v in guitar_mod.RIFF_GENRES.items()]
    d["align_available"] = sections_mod.available()
    return d


@app.post("/api/guitar/kontakt/capture")
def kontakt_capture():
    """Open Kontakt's editor (BLOCKS until you close it) — load Shreddage 3
    Stratus FREE + pick a patch, then close — and save its state for the DI
    render engine. Mac-only (needs the GUI). One-time setup."""
    kp = CFG.get("kontakt_vst3_path")
    if not (kp and os.path.exists(kp)):
        raise HTTPException(400, "Kontakt VST3 not found")
    os.makedirs(os.path.dirname(KONTAKT_STATE), exist_ok=True)
    try:
        # run in a separate process so show_editor() is on the main thread (it
        # fails from this FastAPI worker thread). Blocks until the user closes it.
        import subprocess
        import sys as _sys
        r = subprocess.run([_sys.executable, "-m", "backend.plugin_capture", kp, KONTAKT_STATE],
                           cwd=ROOT, capture_output=True, text=True, timeout=1800)
    except Exception as e:
        raise HTTPException(500, f"Kontakt capture failed: {e}")
    if not os.path.exists(KONTAKT_STATE):
        raise HTTPException(500, f"capture produced no state: {(r.stderr or r.stdout or '')[-400:]}")
    return {"ready": _kontakt_ready()}


@app.post("/api/helix/capture")
def helix_capture(body: dict):
    """Open Helix Native's editor (blocks until you close it), then save its
    full state as a named, reusable tone. Mac-only (needs the GUI)."""
    import re
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "provide a name for the tone")
    hp = postfx_mod._helix_path(CFG)
    if not hp:
        raise HTTPException(400, "Helix Native not configured (helix_vst3_path)")
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "tone"
    out = os.path.join(CFG["helix_presets_dir"], name + ".helixstate")
    os.makedirs(CFG["helix_presets_dir"], exist_ok=True)
    try:
        # separate process → show_editor() on its main thread (fails from a worker thread)
        import subprocess
        import sys as _sys
        r = subprocess.run([_sys.executable, "-m", "backend.plugin_capture", hp, out],
                           cwd=ROOT, capture_output=True, text=True, timeout=1800)
    except Exception as e:
        raise HTTPException(500, f"capture failed: {e}")
    if not os.path.exists(out):
        raise HTTPException(500, f"capture produced no state: {(r.stderr or r.stdout or '')[-400:]}")
    return {"saved": name, "states": postfx_mod.list_helix(CFG)}


@app.post("/api/tone/apply")
async def tone_apply(preset: str = Form("tighten_highgain"),
                     job_id: str = Form(None),
                     file: UploadFile = File(None),
                     normalize: bool = Form(True)):
    """Reshape the distorted guitar in a track: 6-stem split (Mac/Demucs) →
    tone chain on the guitar stem (pedalboard / Helix Native) → recombine."""
    if not postfx_mod.available():
        raise HTTPException(500, "pedalboard not installed on the Mac (pip install pedalboard)")
    sid = uuid.uuid4().hex
    work = os.path.join(STEMS_DIR, sid)
    os.makedirs(work, exist_ok=True)
    src_label = "upload"
    if file is not None:
        ext = os.path.splitext(file.filename)[1] or ".wav"
        inp = os.path.join(work, "input" + ext)
        with open(inp, "wb") as f:
            f.write(await file.read())
        src_label = file.filename
    elif job_id:
        src = next((os.path.join(LIBRARY, job_id + e) for e in (".mp3", ".wav", ".flac")
                    if os.path.exists(os.path.join(LIBRARY, job_id + e))), None)
        if not src:
            raise HTTPException(404, "library track not found")
        inp = os.path.join(work, job_id + os.path.splitext(src)[1])
        shutil.copy(src, inp)
        src_label = job_id
    else:
        raise HTTPException(400, "provide a file or job_id")

    try:
        stem_files = stems_mod.separate(inp, work, mode="guitar")
    except Exception as e:
        raise HTTPException(500, f"separation failed: {e}")
    by_name = {os.path.splitext(os.path.basename(f))[0]: f for f in stem_files}
    guitar = by_name.get("guitar")
    if not guitar:
        raise HTTPException(500, "no guitar stem produced (htdemucs_6s)")

    processed = os.path.join(work, "guitar_tone.wav")
    try:
        postfx_mod.process_stem(guitar, processed, preset, cfg=CFG)
    except Exception as e:
        raise HTTPException(500, f"tone processing failed: {e}")

    # Layer only the guitar tone-delta onto the pristine original mix, so the
    # drums/bass keep their original quality (no Demucs re-sum artifacts).
    try:
        wav = postfx_mod.recombine_delta(inp, guitar, processed, normalize=normalize)
    except Exception as e:
        raise HTTPException(500, f"recombine failed: {e}")

    try:
        os.remove(inp)
    except Exception:
        pass
    jid = uuid.uuid4().hex
    with open(os.path.join(LIBRARY, f"{jid}.wav"), "wb") as f:
        f.write(wav)
    save_done_row(jid, "tone", {"source": src_label, "preset": preset}, _keep_lossless(jid, src))
    return {"job_id": jid, "audio_url": f"/api/audio/{jid}",
            "guitar_url": f"/api/stem/{sid}/guitar_tone.wav", "preset": preset}


def _stash_input(work, label, file_bytes=None, filename=None, job_id=None):
    """Write an uploaded file or copy a library track into `work`, returning the
    local path. Used by tone/master endpoints."""
    if file_bytes is not None:
        ext = os.path.splitext(filename or "")[1] or ".wav"
        p = os.path.join(work, label + ext)
        with open(p, "wb") as f:
            f.write(file_bytes)
        return p, (filename or "upload")
    if job_id:
        src = next((os.path.join(LIBRARY, job_id + e) for e in (".mp3", ".wav", ".flac")
                    if os.path.exists(os.path.join(LIBRARY, job_id + e))), None)
        if not src:
            raise HTTPException(404, f"library track not found: {job_id}")
        p = os.path.join(work, label + os.path.splitext(src)[1])
        shutil.copy(src, p)
        return p, job_id
    raise HTTPException(400, f"provide a file or job_id for {label}")


_AUDIO_EXT = (".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg")


def _gold_refs():
    """Curated gold-standard reference masters (filename stem → path)."""
    out = {}
    if os.path.isdir(MASTER_REFS):
        for fn in sorted(os.listdir(MASTER_REFS)):
            if fn.lower().endswith(_AUDIO_EXT):
                out[os.path.splitext(fn)[0]] = os.path.join(MASTER_REFS, fn)
    return out


@app.get("/api/master/options")
def master_options():
    """What the Master tool can do: which modes are available + their choices."""
    return {
        "reference": master_mod.available(),                 # match to a user reference (Matchering)
        "gold": master_mod.available() and bool(_gold_refs()),  # match to a curated built-in reference
        "auto": master_mod.auto_available(),                 # reference-free DSP chain to a LUFS target
        "gold_refs": list(_gold_refs().keys()),
        "tones": list(master_mod.TONE_CURVES.keys()),
        "lufs_presets": master_mod.LUFS_PRESETS,
    }


def _source_master_title(job_id, tgt_label):
    """Readable title for a mastered output: '<source name> (master)'. Source name
    is the library track's title when mastering a library item, else the uploaded
    file's stem; falls back to 'track'. Keeps the master named after its song so it
    isn't an unnamed row in the library."""
    base = None
    if job_id:
        try:
            with db() as conn:
                r = conn.execute("SELECT params FROM jobs WHERE id=?", (job_id,)).fetchone()
            if r and r[0]:
                base = (json.loads(r[0]).get("title") or "").strip() or None
        except Exception:
            base = None
    if not base and tgt_label and tgt_label != job_id:
        base = os.path.splitext(os.path.basename(str(tgt_label)))[0].strip() or None
    return f"{base or 'track'} (master)"


@app.post("/api/master/apply")
async def master_apply(job_id: str = Form(None),
                       ref_job_id: str = Form(None),
                       file: UploadFile = File(None),
                       ref_file: UploadFile = File(None),
                       bit_depth: int = Form(16),
                       mode: str = Form("reference"),       # reference | gold | auto
                       target_lufs: float = Form(-12.0),    # auto: integrated LUFS target
                       tone: str = Form("balanced"),        # auto: tonal curve
                       width: float = Form(1.0),            # auto: M/S stereo width (1.0 = unchanged)
                       bass_mono_hz: float = Form(0.0),     # auto: keep Side mono below this Hz (0 = off)
                       warmth: float = Form(0.0),           # auto: harmonic saturation amount 0..1
                       ref_name: str = Form(None),          # gold: which curated reference
                       tone_only: bool = Form(False),       # reference: match TONE + keep dynamics
                       match_amount: float = Form(1.0),     # reference tone-only: 0..1 tonal match strength
                       drive: float = Form(0.0)):           # reference tone-only: 0..1 loudness push toward the ref
    """Master a target track. Three modes:
    • reference — match to a user-supplied reference master (Matchering).
    • gold — match to a curated built-in 'gold standard' reference (Matchering, no per-track ref).
    • auto — reference-free DSP chain (tone + glue + limit) to a target loudness (pedalboard)."""
    work = os.path.join(STEMS_DIR, uuid.uuid4().hex)
    os.makedirs(work, exist_ok=True)
    tgt, tgt_label = _stash_input(work, "target",
                                  await file.read() if file else None,
                                  file.filename if file else None, job_id)
    jid = uuid.uuid4().hex
    out = os.path.join(LIBRARY, f"{jid}.wav")
    try:
        if mode == "auto":
            if not master_mod.auto_available():
                raise HTTPException(500, "auto mastering needs pedalboard + pyloudnorm on the Mac")
            _, lufs = master_mod.master_auto(tgt, out, target_lufs=float(target_lufs),
                                             tone=tone, bit_depth=int(bit_depth),
                                             width=float(width), bass_mono_hz=float(bass_mono_hz),
                                             warmth=float(warmth))
            extras = []
            if float(width) != 1.0: extras.append(f"width {float(width):.2f}")
            if float(bass_mono_hz) > 0: extras.append(f"bass-mono <{round(float(bass_mono_hz))}Hz")
            if float(warmth) > 0: extras.append(f"warmth {round(float(warmth) * 100)}%")
            note = f"auto master · {tone} · {round(float(target_lufs))} LUFS" + (" · " + " · ".join(extras) if extras else "")
            params = {"source": tgt_label, "mode": "auto", "tone": tone,
                      "target_lufs": float(target_lufs), "lufs": lufs,
                      "width": float(width), "bass_mono_hz": float(bass_mono_hz),
                      "warmth": float(warmth), "note": note}
        elif mode == "gold":
            if not master_mod.available():
                raise HTTPException(500, "matchering not installed on the Mac (pip install matchering)")
            refs = _gold_refs()
            ref = refs.get(ref_name) or (next(iter(refs.values()), None))
            if not ref:
                raise HTTPException(400, "no gold-standard references — add audio files to library/master_refs/")
            picked = ref_name if ref_name in refs else next(iter(refs.keys()))
            master_mod.master(tgt, ref, out, bit_depth=int(bit_depth))
            params = {"source": tgt_label, "mode": "gold", "reference": picked,
                      "note": f"gold master · {picked}"}
        else:  # reference
            ref, _ = _stash_input(work, "reference",
                                  await ref_file.read() if ref_file else None,
                                  ref_file.filename if ref_file else None, ref_job_id)
            if tone_only:
                # Match the reference's TONE, then push loudness toward it by `drive` through the
                # transparent glue+saturation+true-peak-limit chain (NOT Matchering's brute RMS-match,
                # which slams a loud reference). drive=0 = tone only; drive up = louder but punchier.
                _, achieved = master_mod.reference_match(tgt, ref, out, match_amount=float(match_amount),
                                                         drive=float(drive), bit_depth=int(bit_depth))
                note = "reference tone-match (dynamics preserved)" if float(drive) <= 0 else \
                       f"reference tone + loudness drive {round(float(drive), 2)}" + \
                       (f" -> {round(achieved, 1)} LUFS" if achieved is not None else "")
                params = {"source": tgt_label, "mode": "reference-tone", "reference": ref_job_id or "upload",
                          "match_amount": float(match_amount), "drive": float(drive),
                          "lufs": achieved, "note": note}
            else:
                if not master_mod.available():
                    raise HTTPException(500, "matchering not installed on the Mac (pip install matchering)")
                master_mod.master(tgt, ref, out, bit_depth=int(bit_depth))
                params = {"source": tgt_label, "mode": "reference", "reference": ref_job_id or "upload"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"mastering failed: {e}")
    shutil.rmtree(work, ignore_errors=True)
    params["title"] = _source_master_title(job_id, tgt_label)   # name it after the song
    out = _keep_lossless(jid, tgt)
    save_done_row(jid, "master", params, out)
    return {"job_id": jid, "audio_url": f"/api/audio/{jid}"}


@app.post("/api/backing/strip-guitar")
async def backing_strip_guitar(job_id: str = Form(None),
                               file: UploadFile = File(None),
                               engine: str = Form("demucs"),
                               normalize: bool = Form(True)):
    """6-stem split → recombine every stem EXCEPT guitar → a guitar-less backing.
    Foundation for the symbolic→DI→amp guitar route: drop the model's distorted
    guitar so our own clean-DI-then-amped guitar can sit on a clean backing.
    engine='demucs' (Mac) or 'roformer' (box GPU, cleaner guitar removal)."""
    work = os.path.join(STEMS_DIR, uuid.uuid4().hex)
    os.makedirs(work, exist_ok=True)
    inp, src_label = _stash_input(work, "input",
                                  await file.read() if file else None,
                                  file.filename if file else None, job_id)
    try:
        produced = _separate(inp, work, engine=("roformer" if engine == "roformer" else "demucs"),
                            stems="all", demucs_mode="guitar")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"separation failed: {e}")
    others = [p for (name, p) in produced if name != "guitar"]
    if not others:
        raise HTTPException(500, "no non-guitar stems produced")
    try:
        wav = postfx_mod.recombine(others, normalize=normalize)
    except Exception as e:
        raise HTTPException(500, f"recombine failed: {e}")
    jid = uuid.uuid4().hex
    with open(os.path.join(LIBRARY, f"{jid}.wav"), "wb") as f:
        f.write(wav)
    save_done_row(jid, "backing", {"source": src_label}, os.path.join(LIBRARY, f"{jid}.wav"))
    return {"job_id": jid, "audio_url": f"/api/audio/{jid}"}


@app.post("/api/guitar/render-amp")
async def guitar_render_amp(midi: UploadFile = File(None),
                            riff: bool = Form(False),
                            key: str = Form("E minor"),
                            bpm: int = Form(160),
                            bars: int = Form(8),
                            style: str = Form("gallop"),
                            riff_brain: str = Form("algorithmic"),
                            genre: str = Form(""),
                            part: str = Form("riff"),
                            sections: str = Form(None),
                            align_backing: bool = Form(False),
                            di_engine: str = Form("ks"),
                            duration: float = Form(None),
                            seed: int = Form(None),
                            preset: str = Form("helix"),
                            backing_job_id: str = Form(None),
                            guitar_gain_db: float = Form(0.0)):
    """Symbolic-route render step: MIDI guitar → clean DI (Karplus-Strong) →
    tone/amp (Helix on a *clean* DI is finally valid) → optionally mix onto a
    guitar-less backing. Either upload `midi` or set `riff=true` (music-theory
    metal riff). When mixing onto a backing, the riff auto-matches its length."""
    if not postfx_mod.available():
        raise HTTPException(500, "pedalboard not installed on the Mac")
    sid = uuid.uuid4().hex
    work = os.path.join(STEMS_DIR, sid)
    os.makedirs(work, exist_ok=True)

    # if mixing onto a backing and no explicit duration, match the backing length
    backing_path = None
    if backing_job_id:
        backing_path = next((os.path.join(LIBRARY, backing_job_id + e) for e in (".wav", ".mp3", ".flac")
                             if os.path.exists(os.path.join(LIBRARY, backing_job_id + e))), None)
        if backing_path and not duration:
            try:
                import soundfile as _sf
                duration = float(_sf.info(backing_path).duration)
            except Exception:
                pass

    if midi is not None:
        mp = os.path.join(work, "in.mid")
        with open(mp, "wb") as f:
            f.write(await midi.read())
        try:
            notes = guitar_mod.midi_to_notes(mp)
        except Exception as e:
            raise HTTPException(400, f"could not read MIDI: {e}")
        src_label = midi.filename
    else:
        # riff brain → (brain, provider): algorithmic | auto/local/claude (LLM)
        brain, prov = "algorithmic", ""
        if riff_brain in ("auto", "local", "claude"):
            brain = "llm"
            prov = {"local": "ollama", "claude": "claude"}.get(riff_brain, "")
        btag = "" if brain == "algorithmic" else f" · {riff_brain} AI"
        # Optionally align section timing to the backing's REAL section boundaries
        # (librosa) instead of the arrangement's nominal lengths.
        aligned = False
        blocks = None
        if sections:
            try:
                blocks = json.loads(sections)
            except Exception:
                raise HTTPException(400, "sections must be JSON [{type,seconds}]")
            if align_backing and backing_path and sections_mod.available():
                try:
                    blocks = sections_mod.align_blocks(blocks, backing_path)
                    aligned = True
                except Exception:
                    pass                                  # detection failed → keep nominal
        elif align_backing and backing_path and sections_mod.available():
            try:
                blocks = sections_mod.auto_blocks(backing_path)   # arrange from the backing itself
                aligned = True
            except Exception:
                blocks = None
        if blocks is not None:
            notes = guitar_mod.generate_riff_arrangement(blocks, key, int(bpm), seed,
                                                         brain=brain, provider=prov, genre=genre)
            gtag = f" {genre}" if (brain == "llm" and genre) else ""
            atag = " · aligned to backing" if aligned else ""
            src_label = f"arrangement riff{gtag} {key} {bpm}bpm ({len(blocks)} sections){atag}{btag}"
        elif riff:
            notes = guitar_mod.compose_riff(brain, key, int(bpm), duration_s=duration,
                                            bars=int(bars), style=style, genre=genre,
                                            part=part, provider=prov, seed=seed)
            gtag = genre if (brain == "llm" and genre) else style
            src_label = f"{gtag} {part} {key} {bpm}bpm{btag}"
        else:
            raise HTTPException(400, "upload a MIDI file, set riff=true, or pass sections")
    if not notes:
        raise HTTPException(400, "no notes found")

    di = os.path.join(work, "di.wav")
    guitar_mod.render_di_file(notes, di, engine=di_engine,
                              sf2_path=CFG.get("guitar_soundfont"),
                              kontakt_path=CFG.get("kontakt_vst3_path"),
                              kontakt_state=KONTAKT_STATE)
    amped = os.path.join(work, "amped.wav")
    try:
        postfx_mod.process_stem(di, amped, preset, cfg=CFG)
    except Exception as e:
        raise HTTPException(500, f"amp failed: {e}")

    def _save(mode, params, src_path):
        j = uuid.uuid4().hex
        shutil.copy(src_path, os.path.join(LIBRARY, f"{j}.wav"))
        save_done_row(j, mode, params, os.path.join(LIBRARY, f"{j}.wav"))
        return j
    di_j = _save("guitardi", {"source": src_label, "kind": "clean DI"}, di)
    amp_j = _save("guitar", {"source": src_label, "preset": preset}, amped)
    out = {"di_url": f"/api/audio/{di_j}", "amped_url": f"/api/audio/{amp_j}", "preset": preset}

    if backing_job_id:
        tracks = [{"src": f"/api/audio/{amp_j}", "gain_db": float(guitar_gain_db)},
                  {"src": f"/api/audio/{backing_job_id}", "gain_db": 0.0}]
        try:
            wav = mix_mod.mix(tracks, LIBRARY, STEMS_DIR, normalize=True)
        except Exception as e:
            raise HTTPException(500, f"mix onto backing failed: {e}")
        mj = uuid.uuid4().hex
        with open(os.path.join(LIBRARY, f"{mj}.wav"), "wb") as f:
            f.write(wav)
        save_done_row(mj, "song", {"tags": f"symbolic guitar ({preset}) + backing", "source": src_label},
                      os.path.join(LIBRARY, f"{mj}.wav"))
        out["mix_url"] = f"/api/audio/{mj}"
    return out


# ==================== Add-a-Solo (compose a lead for a region, overlay it) ====================
@app.get("/api/solo/options")
def solo_options():
    """Genres, DI engines, and amp presets for the Solo tab (no GPU)."""
    d = postfx_mod.presets(CFG)
    di = [{"id": "ks", "label": "Karplus-Strong (synth, no setup)"}]
    if CFG.get("guitar_soundfont") and os.path.exists(CFG.get("guitar_soundfont", "")):
        di.append({"id": "soundfont", "label": "SoundFont (FreePats clean)"})
    if _kontakt_ready():
        di.append({"id": "kontakt", "label": "Kontakt / Shreddage"})
    return {
        "amp_presets": d["presets"],
        "helix_available": d["helix_available"],
        "pedalboard": d["pedalboard"],
        "di_engines": di,
        "kontakt_available": bool(CFG.get("kontakt_vst3_path") and os.path.exists(CFG["kontakt_vst3_path"])),
        "kontakt_ready": _kontakt_ready(),
        "genres": [{"id": k, "label": v["label"]} for k, v in guitar_mod.RIFF_GENRES.items()],
    }


def _proc_rss_mb(pid):
    """Resident memory of a pid in MB (mac/linux `ps`), or None."""
    if not pid:
        return None
    try:
        import subprocess
        r = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=3)
        s = r.stdout.strip()
        return round(int(s) / 1024) if s else None
    except Exception:
        return None


@app.get("/api/plugins/status")
def plugins_status():
    """Whether the Kontakt/Shreddage daemon is resident (holding samples in RAM) + its
    footprint, and the current idle auto-unload setting. Helix is loaded per-render so
    it isn't held between renders."""
    pid = kontakt_daemon_mod.daemon_pid()
    return {"kontakt_loaded": bool(pid), "kontakt_rss_mb": _proc_rss_mb(pid),
            "idle_sec": kontakt_daemon_mod.IDLE_SEC}


@app.post("/api/plugins/unload")
def plugins_unload(body: dict = None):
    """Free plugin RAM on demand: kill the Kontakt daemon (releases Shreddage) and drop
    the in-process Helix/VST thread + GC. Both reload lazily on the next render."""
    kontakt_daemon_mod.shutdown()
    postfx_mod.unload()
    import gc
    gc.collect()
    return {"ok": True, "kontakt_loaded": kontakt_daemon_mod.is_loaded()}


@app.post("/api/plugins/idle")
def plugins_idle(body: dict):
    """Set the Kontakt idle auto-unload timeout in seconds (0 = off). The daemon frees
    itself after that long with no renders; the next render respawns it."""
    sec = kontakt_daemon_mod.set_idle(int(body.get("seconds") or 0))
    return {"ok": True, "idle_sec": sec}


def _extract_caption(task):
    """Pull the prose caption out of a /query_result full_analysis_only task."""
    res = task.get("result")
    if isinstance(res, str):
        try:
            res = json.loads(res)
        except Exception:
            return res.strip()[:600]
    if isinstance(res, list):
        res = res[0] if res else {}
    if isinstance(res, dict):
        for k in ("caption", "prompt", "description"):
            v = res.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()[:600]
        meta = res.get("metas") or res.get("meta")
        if isinstance(meta, dict):
            c = meta.get("caption") or meta.get("prompt")
            if c:
                return str(c).strip()[:600]
        return json.dumps(res)[:600]
    return ""


def _release_caption(wav):
    """POST /release_task {full_analysis_only:true} + region audio -> poll /query_result
    -> caption prose. The LM is effectively always resident (verified 2026-06-01: a region
    captions in ~6s with NO forced load; the engine's loaded_lm_model/is_loaded fields LIE
    -- see [[engine-lm-captioning]]), so we do NOT force-load it routinely."""
    tid = acestep_py.submit(ACESTEP_HOST, {"full_analysis_only": True},
                            src_audio=(wav, "region.wav"))
    deadline = time.time() + 240
    while time.time() < deadline:
        t = acestep_py.query(ACESTEP_HOST, tid)
        st = t.get("status")
        if st == 1 or st == "succeeded":
            return _extract_caption(t)
        if st == 2 or (isinstance(st, str) and st.lower() in ("failed", "error")):
            raise RuntimeError(t.get("message") or t.get("error") or f"caption task failed: {t}")
        time.sleep(1.5)
    raise RuntimeError("LM caption timed out")


def _caption_region_via_lm(src_path, start, end):
    """Caption a [start,end] region with the ACE engine's LM (audio->text) -- the 'ears'
    for the Listen brain. Same understand-audio call the dataset autolabel runs; the LM's
    caption prompt is fixed/non-steerable, so we use it purely as audio-grounded context
    for the note-writer. The LM's own bpm/key are unreliable -- ignore them (we use librosa)."""
    _lora_require_engine()
    wav = solo_mod.slice_region_wav(src_path, start, end)
    try:
        return _release_caption(wav)            # fast path: resident LM, ~6s, no load
    except Exception:
        # Cold-engine fallback only: if the very first call fails, ensure the LM
        # subsystem is up (one /v1/init) and retry once. Avoids a routine leaky init.
        free_gpu("acestep")
        _ensure_labeling_ready(ACESTEP_HOST)
        return _release_caption(wav)


@app.post("/api/solo/compose")
def solo_compose(body: dict):
    """Compose (or re-roll) a solo for a region of a library track. Detects the
    region's bpm/key (unless overridden) and returns a piano-roll score + the
    used bpm/key. No render/mix yet -- this is the preview/re-roll step.

    brain='listen' first has the ACE LM 'listen' to the region (audio->text caption)
    and feeds that as context to the LLM note-writer -- a theme-grounded solo. Other
    brains: 'algorithmic' (instant) or 'llm' (provider-driven)."""
    pid = body.get("job_id")
    if not pid:
        raise HTTPException(400, "job_id required")
    src = _lib_source_path(pid)
    if not src:
        raise HTTPException(404, "track not found")
    start = float(body.get("start") or 0.0)
    end = float(body.get("end") or 0.0)
    if not (end > start):
        raise HTTPException(400, "region end must be after start")
    bpm = body.get("bpm")
    key = body.get("key")
    if not bpm or not key:
        det = solo_mod.analyze_region(src, start, end)
        bpm = bpm or det["bpm"]
        key = key or det["key"]
    brain = body.get("brain", "algorithmic")
    # `context` = a (possibly user-edited) description from /api/solo/listen. When the
    # Listen brain is used WITH a context, we feed it straight to the note-writer and
    # skip re-captioning (lets the user review/edit the description first).
    context = (body.get("context") or "").strip()
    heard = context
    if brain == "listen" and not context:
        heard = _caption_region_via_lm(src, start, end)  # not pre-listened: caption now
    compose_brain = "llm" if brain == "listen" else brain
    notes, score = solo_mod.compose(
        key=key, bpm=float(bpm), duration_s=end - start,
        genre=body.get("genre", ""), brain=compose_brain,
        provider=body.get("provider", ""), seed=body.get("seed"), context=heard)
    return {"score": score, "bpm": float(bpm), "key": key,
            "duration": round(end - start, 3), "heard": heard}


@app.post("/api/solo/listen")
def solo_listen(body: dict):
    """ACE LM 'listens' to the region and returns an editable description (no compose).
    The user can review/edit it, then /api/solo/compose (brain='listen', context=<edited>)
    feeds it to the note-writer. Audio->text only; ~6s on the resident LM."""
    pid = body.get("job_id")
    if not pid:
        raise HTTPException(400, "job_id required")
    src = _lib_source_path(pid)
    if not src:
        raise HTTPException(404, "track not found")
    start = float(body.get("start") or 0.0)
    end = float(body.get("end") or 0.0)
    if not (end > start):
        raise HTTPException(400, "region end must be after start")
    return {"heard": _caption_region_via_lm(src, start, end)}


@app.post("/api/solo/render")
def solo_render(body: dict):
    """Render a composed solo (DI -> amp), overlay it onto the original track at the
    region offset (level-matched, edge-faded, optional duck), and save the result.
    All Mac-side DSP -- no GPU, no source separation."""
    pid = body.get("job_id")
    if not pid:
        raise HTTPException(400, "job_id required")
    src = _lib_source_path(pid)
    if not src:
        raise HTTPException(404, "track not found")
    start = float(body.get("start") or 0.0)
    end = float(body.get("end") or 0.0)
    if not (end > start):
        raise HTTPException(400, "region end must be after start")
    region_len = end - start
    # Notes: prefer the previewed score (so the render matches the piano-roll); else compose.
    score = body.get("score")
    if score and score.get("notes"):
        notes = solo_mod.score_to_notes(score)
    else:
        bpm = body.get("bpm") or solo_mod.analyze_region(src, start, end)["bpm"]
        key = body.get("key") or solo_mod.analyze_region(src, start, end)["key"]
        notes, _ = solo_mod.compose(key=key, bpm=float(bpm), duration_s=region_len,
                                    genre=body.get("genre", ""), brain=body.get("brain", "algorithmic"),
                                    provider=body.get("provider", ""), seed=body.get("seed"))
    di_engine = body.get("di_engine", "ks")
    amp_preset = body.get("amp_preset", "")
    mix = body.get("mix") or {}
    fade_ms = float(mix.get("fade_ms", 120))

    solo_jid = uuid.uuid4().hex
    solo_path = os.path.join(LIBRARY, f"{solo_jid}.wav")
    try:
        solo_mod.render_clip(notes, solo_path, region_len, engine=di_engine,
                             sf2_path=CFG.get("guitar_soundfont"),
                             kontakt_path=CFG.get("kontakt_vst3_path"),
                             kontakt_state=KONTAKT_STATE, amp_preset=amp_preset,
                             cfg=CFG, fade_ms=min(fade_ms, 60))
    except Exception as e:
        raise HTTPException(500, f"solo render failed: {e}")
    try:
        os.remove(solo_path + ".di.wav")
    except OSError:
        pass

    mix_jid = uuid.uuid4().hex
    mix_path = os.path.join(LIBRARY, f"{mix_jid}.wav")
    try:
        solo_mod.overlay(src, solo_path, start, mix_path,
                         solo_gain_db=float(mix.get("solo_gain_db", 0.0)),
                         auto_match=bool(mix.get("auto_match", True)),
                         duck_db=float(mix.get("duck_db", 0.0)),
                         fade_ms=fade_ms,
                         highpass_hz=float(mix.get("highpass_hz", 0.0)),
                         normalize=True)
    except Exception as e:
        raise HTTPException(500, f"solo overlay/mix failed: {e}")

    with db() as conn:
        row = conn.execute("SELECT params FROM jobs WHERE id=?", (pid,)).fetchone()
    orig_params = json.loads(row["params"]) if row and row["params"] else {}
    orig_title = orig_params.get("title") or "track"
    region_str = f"{round(start, 1)}-{round(end, 1)}s"
    save_done_row(solo_jid, "guitar",
                  {"preset": amp_preset or di_engine, "part": "solo", "source": orig_title,
                   "region": region_str}, solo_path)
    save_done_row(mix_jid, "song",
                  {"title": f"{orig_title} (with solo)", "source": orig_title, "solo": True,
                   "region": region_str, "genre": body.get("genre", ""),
                   "key": body.get("key"), "bpm": body.get("bpm")}, mix_path)
    return {"audio_url": f"/api/audio/{mix_jid}", "job_id": mix_jid,
            "solo_url": f"/api/audio/{solo_jid}", "region": region_str}


def _solo_orig_title(pid):
    with db() as conn:
        row = conn.execute("SELECT params FROM jobs WHERE id=?", (pid,)).fetchone()
    p = json.loads(row["params"]) if row and row["params"] else {}
    return p.get("title") or "track"


def _solo_region(body):
    """(src_path, start, end, region_len, region_str) from a request, or raise."""
    pid = body.get("job_id")
    if not pid:
        raise HTTPException(400, "job_id required")
    src = _lib_source_path(pid)
    if not src:
        raise HTTPException(404, "track not found")
    start = float(body.get("start") or 0.0)
    end = float(body.get("end") or 0.0)
    if not (end > start):
        raise HTTPException(400, "region end must be after start")
    return src, start, end, end - start, f"{round(start, 1)}-{round(end, 1)}s"


@app.post("/api/solo/di")
def solo_di(body: dict):
    """STAGE A of the staged Add-Solo flow: render the composed score to a CLEAN DI
    clip (no amp), gated to the region length, and save it as a playable library
    intermediate so the raw composed solo can be auditioned + re-rolled on its own."""
    src, start, end, region_len, region_str = _solo_region(body)
    score = body.get("score")
    if score and score.get("notes"):
        notes = solo_mod.score_to_notes(score)
    else:
        bpm = body.get("bpm") or solo_mod.analyze_region(src, start, end)["bpm"]
        key = body.get("key") or solo_mod.analyze_region(src, start, end)["key"]
        notes, _ = solo_mod.compose(key=key, bpm=float(bpm), duration_s=region_len,
                                    genre=body.get("genre", ""), brain=body.get("brain", "algorithmic"),
                                    provider=body.get("provider", ""), seed=body.get("seed"))
    di_engine = body.get("di_engine", "ks")
    di_jid = uuid.uuid4().hex
    di_path = os.path.join(LIBRARY, f"{di_jid}.wav")
    try:
        solo_mod.render_di_clip(notes, di_path, region_len, engine=di_engine,
                                sf2_path=CFG.get("guitar_soundfont"),
                                kontakt_path=CFG.get("kontakt_vst3_path"),
                                kontakt_state=KONTAKT_STATE, fade_ms=40)
    except Exception as e:
        raise HTTPException(500, f"DI render failed: {e}")
    save_done_row(di_jid, "guitar",
                  {"preset": di_engine, "part": "solo-di", "source": _solo_orig_title(body["job_id"]),
                   "region": region_str}, di_path)
    return {"di_job_id": di_jid, "di_url": f"/api/audio/{di_jid}", "region": region_str}


@app.post("/api/solo/clip")
def solo_clip(body: dict):
    """STAGE B: take the cached DI clip (di_job_id) -> apply the amp/tone -> a dry,
    region-length amped solo CLIP, saved to the library. Re-runs without recomposing
    or re-rendering the DI when only the amp choice changed."""
    src, start, end, region_len, region_str = _solo_region(body)
    di_job_id = body.get("di_job_id")
    if not di_job_id:
        raise HTTPException(400, "di_job_id required (run /api/solo/di first)")
    di_src = _lib_source_path(di_job_id)
    if not di_src:
        raise HTTPException(404, "DI clip not found")
    amp_preset = body.get("amp_preset", "")
    clip_jid = uuid.uuid4().hex
    clip_path = os.path.join(LIBRARY, f"{clip_jid}.wav")
    try:
        solo_mod.amp_clip(di_src, clip_path, region_len, amp_preset=amp_preset,
                          cfg=CFG, fade_ms=40)
    except Exception as e:
        raise HTTPException(500, f"amp render failed: {e}")
    save_done_row(clip_jid, "guitar",
                  {"preset": amp_preset or "clean DI", "part": "solo-amp",
                   "source": _solo_orig_title(body["job_id"]), "region": region_str}, clip_path)
    return {"clip_job_id": clip_jid, "clip_url": f"/api/audio/{clip_jid}", "region": region_str}


@app.post("/api/solo/mix")
def solo_mix(body: dict):
    """STAGE C: overlay the cached amped clip (clip_job_id) onto the original track at
    the region offset (level-matched, edge-faded, optional duck/HPF) and save the final
    combined result. Re-runs without redoing compose/DI/amp when only mix knobs changed."""
    src, start, end, region_len, region_str = _solo_region(body)
    clip_job_id = body.get("clip_job_id")
    if not clip_job_id:
        raise HTTPException(400, "clip_job_id required (run /api/solo/clip first)")
    clip_src = _lib_source_path(clip_job_id)
    if not clip_src:
        raise HTTPException(404, "solo clip not found")
    mix = body.get("mix") or {}
    mix_jid = uuid.uuid4().hex
    mix_path = os.path.join(LIBRARY, f"{mix_jid}.wav")
    try:
        solo_mod.overlay(src, clip_src, start, mix_path,
                         solo_gain_db=float(mix.get("solo_gain_db", 0.0)),
                         auto_match=bool(mix.get("auto_match", True)),
                         duck_db=float(mix.get("duck_db", 0.0)),
                         fade_ms=float(mix.get("fade_ms", 120)),
                         highpass_hz=float(mix.get("highpass_hz", 0.0)),
                         normalize=True)
    except Exception as e:
        raise HTTPException(500, f"solo overlay/mix failed: {e}")
    orig_title = _solo_orig_title(body["job_id"])
    save_done_row(mix_jid, "song",
                  {"title": f"{orig_title} (with solo)", "source": orig_title, "solo": True,
                   "region": region_str, "genre": body.get("genre", ""),
                   "key": body.get("key"), "bpm": body.get("bpm")}, mix_path)
    return {"audio_url": f"/api/audio/{mix_jid}", "job_id": mix_jid, "region": region_str}


@app.post("/api/shape")
async def shape_track(params: str = Form(...), ref_file: UploadFile = File(None)):
    """Tone + dynamics shaping (de-harsh / multiband dynamics / transient / tonal feel-match)
    over a library track. Multipart: `params` (JSON) + optional `ref_file` for the tonal
    match reference (else `ref_job_id` in params = a library track). Each module runs only if
    its key is present in params. Pure Mac DSP; chain before or after Master."""
    body = json.loads(params)
    pid = body.get("job_id")
    if not pid:
        raise HTTPException(400, "job_id required")
    src = _lib_source_path(pid)
    if not src:
        raise HTTPException(404, "track not found")
    config = {}
    for mod in ("deharsh", "dynamics", "transient", "match"):
        if mod in body and body[mod] is not None:
            config[mod] = body[mod] or {}
    if not config:
        raise HTTPException(400, "enable at least one shaping module")
    work = os.path.join(STEMS_DIR, uuid.uuid4().hex)
    ref_path = None
    if "match" in config:
        if ref_file is not None:
            os.makedirs(work, exist_ok=True)
            ref_path, _ = _stash_input(work, "ref", await ref_file.read(), ref_file.filename)
        elif body.get("ref_job_id"):
            ref_path = _lib_source_path(body["ref_job_id"])
        if not ref_path:
            raise HTTPException(400, "tonal feel-match needs a reference track or uploaded file")
    out_jid = uuid.uuid4().hex
    out_path = os.path.join(LIBRARY, f"{out_jid}.wav")
    try:
        report = shape_mod.process_file(src, out_path, config, ref_path=ref_path)
    except Exception as e:
        raise HTTPException(500, f"shape failed: {e}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    out_path = _keep_lossless(out_jid, src)
    orig_title = _solo_orig_title(pid)
    save_done_row(out_jid, "master",
                  {"title": f"{orig_title} (shaped)", "source": orig_title,
                   "shape": report.get("applied", [])}, out_path)
    return {"audio_url": f"/api/audio/{out_jid}", "job_id": out_jid, "report": report}


@app.post("/api/deglitch")
def deglitch_track(body: dict):
    """Detect + repair short impulsive glitches (clicks/pops/zipper bursts) baked into a
    generated track, by LPC-residual detection + AR interpolation (Mac DSP, no GPU). Saves
    a cleaned version and returns a report of exactly what was repaired. `analyze_only`
    skips the write and just reports detections so you can preview before committing.
    `threshold` (higher = more conservative), `max_click_ms`, `repair` (ar|spline)."""
    pid = body.get("job_id")
    if not pid:
        raise HTTPException(400, "job_id required")
    src = _lib_source_path(pid)
    if not src:
        raise HTTPException(404, "track not found")
    threshold = float(body.get("threshold", 14.0))
    max_click_ms = float(body.get("max_click_ms", 2.0))
    repair = body.get("repair", "ar")
    if body.get("analyze_only"):
        import soundfile as sf
        import numpy as _np
        data, sr = sf.read(src, dtype="float32", always_2d=True)
        clicks_total, bursts_total, regs = 0, 0, []
        for c in range(data.shape[1]):
            cl, br = deglitch_mod.detect(data[:, c], sr, threshold=threshold, max_click_ms=max_click_ms)
            clicks_total += len(cl); bursts_total += len(br)
            for s, e in cl:
                regs.append({"channel": c, "start_s": round(s / sr, 3),
                             "dur_ms": round((e - s) * 1000.0 / sr, 2)})
        return {"analyze_only": True, "clicks_found": clicks_total,
                "bursts_flagged": bursts_total, "regions": regs[:200],
                "duration_s": round(len(data) / sr, 2)}
    out_jid = uuid.uuid4().hex
    out_path = os.path.join(LIBRARY, f"{out_jid}.wav")
    try:
        report = deglitch_mod.deglitch_file(src, out_path, threshold=threshold,
                                            max_click_ms=max_click_ms, repair=repair)
    except Exception as e:
        raise HTTPException(500, f"de-glitch failed: {e}")
    out_path = _keep_lossless(out_jid, src)
    orig_title = _solo_orig_title(pid)
    save_done_row(out_jid, "master",
                  {"title": f"{orig_title} (de-glitched)", "source": orig_title,
                   "deglitch": True, "clicks_repaired": report["clicks_repaired"],
                   "repaired_ms": report["repaired_ms"], "bursts_flagged": report["bursts_flagged"]},
                  out_path)
    return {"audio_url": f"/api/audio/{out_jid}", "job_id": out_jid, "report": report}


@app.post("/api/import/upload")
async def import_upload(file: UploadFile = File(...), title: str = Form(None)):
    """Import a local audio file INTO the library as a reusable `source` track
    (usable by Cover/Restyle/Stems/Backing/Layer/Master). mp3/wav saved as-is;
    anything else (flac/m4a/ogg…) transcoded to mp3 via ffmpeg so /api/audio and
    every feature's source-resolver can find it at library/<id>.<ext>."""
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    iid = uuid.uuid4().hex
    ext = os.path.splitext(file.filename or "")[1].lower()
    # .flac passes through untouched: transcoding a lossless import to mp3 threw away
    # exactly the quality the finish chain is built to preserve
    if ext in (".mp3", ".wav", ".flac"):
        out = os.path.join(LIBRARY, iid + ext)
        with open(out, "wb") as f:
            f.write(data)
    else:                                            # transcode to mp3 via ffmpeg
        import shutil as _sh
        import subprocess
        ff = _sh.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
        if not os.path.exists(ff):
            raise HTTPException(500, "ffmpeg not found (brew install ffmpeg) — needed to import this format")
        tmp = os.path.join(LIBRARY, iid + (ext or ".bin"))
        with open(tmp, "wb") as f:
            f.write(data)
        out = os.path.join(LIBRARY, iid + ".mp3")
        r = subprocess.run([ff, "-y", "-i", tmp, "-vn", "-b:a", "192k", out],
                           capture_output=True, text=True)
        try:
            os.remove(tmp)
        except Exception:
            pass
        if r.returncode != 0 or not os.path.exists(out):
            raise HTTPException(500, f"transcode failed: {r.stderr[-400:]}")
    name = (title or os.path.splitext(file.filename or "")[0] or "imported audio")[:80]
    save_done_row(iid, "source", {"source": name, "imported": True}, out)
    return {"job_id": iid, "import_id": iid, "audio_url": f"/api/audio/{iid}", "title": name}


@app.post("/api/import/fetch")
def import_fetch(body: dict):
    """Download audio from a URL (YouTube etc.) via yt-dlp → mp3 in the library
    (no DB row; scratch). Returns a playable URL + an import_id for /extract.
    Personal-use: respects the same rights caveat as the cloning features."""
    url = (body.get("url") or "").strip()
    if not url:
        raise HTTPException(400, "provide a URL")
    import shutil as _sh
    try:
        import yt_dlp
    except Exception:
        raise HTTPException(500, "yt-dlp not installed on the Mac (pip install yt-dlp)")
    iid = uuid.uuid4().hex
    opts = {"format": "bestaudio/best", "noplaylist": True, "quiet": True, "no_warnings": True,
            "outtmpl": os.path.join(LIBRARY, iid + ".%(ext)s"),
            # alternate player clients help dodge YouTube 403s (no cookies needed)
            "extractor_args": {"youtube": {"player_client": ["tv", "ios", "web_safari", "mweb", "android"]}},
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3",
                                "preferredquality": "192"}]}
    ffloc = _sh.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    if os.path.exists(ffloc):
        opts["ffmpeg_location"] = ffloc
    # use the user's YouTube login (cookies) to dodge bot-detection 403s
    if CFG.get("ytdlp_cookies_file"):
        opts["cookiefile"] = CFG["ytdlp_cookies_file"]
    elif CFG.get("ytdlp_cookies_from_browser"):
        opts["cookiesfrombrowser"] = (CFG["ytdlp_cookies_from_browser"],)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as e:
        raise HTTPException(502, f"download failed: {e}")
    if not os.path.exists(os.path.join(LIBRARY, iid + ".mp3")):
        raise HTTPException(500, "no audio produced — is ffmpeg installed? (brew install ffmpeg)")
    title = (info.get("title") or "")[:80]
    save_done_row(iid, "source", {"source": title or "url-import", "url": url[:200]},
                  os.path.join(LIBRARY, iid + ".mp3"))
    return {"import_id": iid, "audio_url": f"/api/audio/{iid}", "title": title}


@app.get("/api/archive/search")
def archive_search(q: str, rows: int = 24):
    """Search the Internet Archive for audio items (free, no auth)."""
    try:
        r = requests.get("https://archive.org/advancedsearch.php", params={
            "q": f"({q}) AND mediatype:audio", "rows": min(rows, 50), "output": "json",
            "fl[]": ["identifier", "title", "creator", "year"]}, timeout=20)
        docs = r.json().get("response", {}).get("docs", [])
    except Exception as e:
        raise HTTPException(502, f"archive search failed: {e}")

    def one(v):
        return (v[0] if isinstance(v, list) and v else (v or ""))
    return {"results": [{"identifier": d.get("identifier"),
                         "title": str(one(d.get("title")))[:90],
                         "creator": str(one(d.get("creator")))[:60],
                         "year": one(d.get("year"))} for d in docs if d.get("identifier")]}


@app.get("/api/archive/item")
def archive_item(id: str):
    """List an Archive item's audio files, grouped per track with the available
    formats/qualities (FLAC > WAV > AAC > MP3 > Ogg) + sizes + download URLs."""
    import urllib.parse
    try:
        m = requests.get(f"https://archive.org/metadata/{os.path.basename(id)}", timeout=20).json()
    except Exception as e:
        raise HTTPException(502, f"archive metadata failed: {e}")
    AUDIO = ("mp3", "flac", "ogg", "vorbis", "wav", "aac", "aiff", "m4a")
    order = {"flac": 0, "wav": 1, "aiff": 1, "aac": 2, "m4a": 2, "mp3": 3, "ogg": 4, "vorbis": 4}
    groups = {}
    for f in m.get("files", []):
        name = f.get("name", "")
        fmt = f.get("format") or ""
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if not (any(k in fmt.lower() for k in AUDIO) or ext in AUDIO):
            continue
        base = name.rsplit(".", 1)[0]
        groups.setdefault(base, []).append({
            "format": fmt or ext.upper(), "size": int(f.get("size", 0) or 0), "name": name,
            "url": f"https://archive.org/download/{os.path.basename(id)}/" + urllib.parse.quote(name)})
    tracks = [{"title": b, "files": sorted(fs, key=lambda x: order.get(x["name"].rsplit(".", 1)[-1].lower(), 9))}
              for b, fs in groups.items()]
    tracks.sort(key=lambda t: t["title"])
    md = m.get("metadata", {})
    title = md.get("title", "")
    return {"id": id, "title": str(title[0] if isinstance(title, list) else title)[:90], "tracks": tracks}


@app.post("/api/import/extract")
async def import_extract(file: UploadFile = File(None), import_id: str = Form(None),
                         start: float = Form(0.0), end: float = Form(0.0)):
    """Trim a [start,end] region of a song (uploaded OR previously fetched by URL)
    and extract its vocal with Demucs (Mac MPS). Returns the isolated vocal URL."""
    import soundfile as sf
    sid = uuid.uuid4().hex
    work = os.path.join(STEMS_DIR, sid)
    os.makedirs(work, exist_ok=True)
    if file is not None:
        ext = os.path.splitext(file.filename or "")[1] or ".wav"
        src_id = uuid.uuid4().hex
        raw = os.path.join(LIBRARY, src_id + ext)  # persist uploaded song as a source
        with open(raw, "wb") as f:
            f.write(await file.read())
        save_done_row(src_id, "source", {"source": file.filename or "upload"}, raw)
    elif import_id:
        raw = next((os.path.join(LIBRARY, import_id + e) for e in (".mp3", ".wav", ".flac")
                    if os.path.exists(os.path.join(LIBRARY, import_id + e))), None)
        if not raw:
            raise HTTPException(404, "fetched audio not found (re-fetch the URL)")
    else:
        raise HTTPException(400, "provide a file or import_id (fetch a URL first)")
    try:
        data, sr = sf.read(raw, dtype="float32", always_2d=True)
    except Exception as ex:
        raise HTTPException(400, f"could not read audio: {ex}")
    s0 = max(0, int(start * sr))
    s1 = int(end * sr) if end > start else len(data)
    s1 = min(s1, len(data))
    if s1 - s0 < int(0.5 * sr):
        raise HTTPException(400, "selection too short (need at least ~0.5s)")
    trimmed = os.path.join(work, "trimmed.wav")
    sf.write(trimmed, data[s0:s1], sr)
    try:
        files = stems_mod.separate(trimmed, work, mode="vocals")
    except Exception as ex:
        raise HTTPException(500, f"vocal extraction failed: {ex}")
    voc = next((f for f in files if os.path.basename(f).startswith("vocals")), None)
    if not voc:
        raise HTTPException(500, "no vocal stem produced")
    stem_id = uuid.uuid4().hex
    save_done_row(stem_id, "stem", {"source": "import-extract", "kind": "vocals"}, voc)
    return {"sid": sid, "vocal_url": f"/api/stem/{sid}/{os.path.basename(voc)}",
            "duration": round((s1 - s0) / sr, 2)}


@app.get("/api/llm/providers")
def llm_providers():
    return {"ollama": llm_mod.ollama_models(),
            "claude": bool(os.environ.get("ANTHROPIC_API_KEY")),   # API key (per-token)
            "claude_sub": llm_mod.claude_code_authed()}            # Claude subscription - only if the CLI can actually auth


@app.post("/api/llm")
def llm_chat(body: dict):
    try:
        text = llm_mod.chat(body.get("provider", "ollama"), body.get("model", ""),
                            body.get("task", "ideas"), body.get("input", ""),
                            CFG.get("claude_model", "claude-3-5-sonnet-latest"))
    except Exception as e:
        raise HTTPException(500, f"LLM failed: {e}")
    return {"text": text}


@app.post("/api/lyrics/song")
def lyrics_song(body: dict):
    """Structure-aware lyrics for a Song-Constructor arrangement: distinct verses,
    one repeated chorus hook, optional pre-chorus/bridge; instrumental sections
    left wordless. Returns lyrics keyed by block index. Runs on the Mac (Ollama)."""
    blocks = body.get("blocks") or []
    if not blocks:
        raise HTTPException(400, "provide the arrangement blocks")
    try:
        return lyrics_mod.write_song_lyrics(
            blocks, body.get("theme", ""), body.get("style", ""),
            body.get("provider", ""), body.get("model", ""),
            CFG.get("claude_model", "claude-3-5-sonnet-latest"),
            extra_sung=body.get("extra_sung"))
    except Exception as e:
        raise HTTPException(500, f"lyric generation failed: {e}")


@app.get("/api/sources")
def sources():
    """All mixable audio: library items + stem outputs (for the mixer dropdowns)."""
    out = []
    with db() as conn:
        rows = conn.execute(
            "SELECT id,mode,params FROM jobs WHERE status='done' ORDER BY created DESC LIMIT 100").fetchall()
    for r in rows:
        p = json.loads(r["params"])
        label = f"{r['mode']}: " + (p.get("tags") or p.get("voice") or p.get("source") or "")[:36]
        out.append({"label": label, "url": f"/api/audio/{r['id']}"})
    if os.path.isdir(STEMS_DIR):
        for sid in sorted(os.listdir(STEMS_DIR), reverse=True)[:30]:
            d = os.path.join(STEMS_DIR, sid)
            if os.path.isdir(d):
                for f in sorted(os.listdir(d)):
                    if f.endswith(".wav"):
                        out.append({"label": f"stem {f[:-4]} ({sid[:6]})",
                                    "url": f"/api/stem/{sid}/{f}"})
    return out


@app.post("/api/mix")
def mix_tracks(body: dict):
    tracks = body.get("tracks", [])
    if not tracks:
        raise HTTPException(400, "add at least one track")
    try:
        wav = mix_mod.mix(tracks, LIBRARY, STEMS_DIR, normalize=body.get("normalize", True))
    except Exception as e:
        raise HTTPException(500, f"mix failed: {e}")
    jid = uuid.uuid4().hex
    with open(os.path.join(LIBRARY, f"{jid}.wav"), "wb") as f:
        f.write(wav)
    save_done_row(jid, "mix", {"sources": [t.get("src") for t in tracks]},
                  _keep_lossless(jid, *[_lib_source_path(str(t.get("src") or "")) for t in tracks]))
    return {"job_id": jid, "audio_url": f"/api/audio/{jid}"}


@app.post("/api/reference/analyze")
async def reference_analyze(file: UploadFile = File(None), params: str = Form("{}"), job_id: str = Form(None)):
    """Analyse a reference track (Mac/librosa) → a Song-Constructor spec (bpm, key,
    labelled sections, optional lyrics) to seed the Song Builder for a fresh text2music
    generation. RESEARCH §17 P1. (P2 will route structure/tags to the box allin1+CLAP.)"""
    if not analyze_mod.available():
        raise HTTPException(500, "analysis needs librosa + scikit-learn on the Mac")
    p = json.loads(params or "{}")
    work = os.path.join(STEMS_DIR, uuid.uuid4().hex)
    os.makedirs(work, exist_ok=True)
    try:
        src, _ = _stash_input(work, "ref",
                              await file.read() if file else None,
                              file.filename if file else None, job_id)
        # Structure / BPM / key / lyrics are computed on the Mac (P1) — always, no box deps.
        try:
            res = analyze_mod.analyze_reference(src, with_lyrics=bool(p.get("with_lyrics")),
                                                asr_size=p.get("model_size", "small"))
            res["engine"] = "librosa"
        except Exception as e:
            raise HTTPException(500, f"analysis failed: {e}")
        # If the box analyze service is set, fetch CLAP style tags (the GPU value-add) and
        # merge them in. If the box also happens to have allin1 (optional), prefer its
        # labelled structure too. Box failure → keep the Mac result.
        if ANALYZE_HOST and not p.get("force_local"):
            try:
                free_gpu("")                          # free ComfyUI + RVC before the box uses the 3090
            except Exception:
                pass
            try:
                genre_labels = [g["label"] for g in genres_mod.GENRES if not g.get("parent")]
                b = analyze_py.analyze(ANALYZE_HOST, src, labels=genre_labels, with_tags=True, with_key=True)
                if isinstance(b.get("tags"), list) and b["tags"]:
                    res["tags"] = b["tags"]
                # allin1 present on the box → use its labelled structure + bpm/key, and
                # remap the Mac transcript onto the allin1 sections so lyrics don't cost us
                # the better structure.
                boxblocks = analyze_mod.blocks_from_allin1(b.get("segments"))
                if boxblocks:
                    musical = [s for s in (b.get("segments") or [])
                               if str(s.get("label", "")).lower() not in ("start", "end", "silence", "")]
                    if p.get("with_lyrics") and res.get("lyric_segments") and len(musical) == len(boxblocks):
                        analyze_mod.map_lyric_segments(
                            list(zip(boxblocks, [float(s["start"]) for s in musical], [float(s["end"]) for s in musical])),
                            res["lyric_segments"])
                    res["blocks"] = boxblocks
                    res["duration"] = round(float(sum(x["seconds"] for x in boxblocks)), 1)
                    if b.get("bpm"):
                        res["bpm"] = b["bpm"]
                    if b.get("key"):
                        res["key"] = b["key"]
                    res["engine"] = "allin1"
            except Exception as e:
                print(f"[analyze] box tags unavailable ({e}); using Mac analysis only")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return res


@app.post("/api/stitch")
def stitch_tracks(body: dict):
    """Concatenate per-block clips into one song with a crossfade (Song Constructor)."""
    tracks = body.get("tracks", [])
    if not tracks:
        raise HTTPException(400, "no tracks to stitch")
    try:
        wav = mix_mod.stitch(tracks, LIBRARY, STEMS_DIR,
                             crossfade_s=float(body.get("crossfade_s", 1.0)),
                             bpm=body.get("bpm"), beat_align=bool(body.get("beat_align", False)))
    except Exception as e:
        raise HTTPException(500, f"stitch failed: {e}")
    jid = uuid.uuid4().hex
    with open(os.path.join(LIBRARY, f"{jid}.wav"), "wb") as f:
        f.write(wav)
    params = {"tags": body.get("tags", ""),
              "sections": body.get("sections", ""),
              "crossfade_s": body.get("crossfade_s", 1.0),
              "blocks": len(tracks)}
    if body.get("title"):
        params["title"] = str(body["title"]).strip()
    if body.get("song_meta"):
        params["song_meta"] = body["song_meta"]; params["from_builder"] = True
    save_done_row(jid, "song", params, os.path.join(LIBRARY, f"{jid}.wav"))
    return {"job_id": jid, "audio_url": f"/api/audio/{jid}"}


# ---------------- Vocal Builder (AI melody → singing → re-timbre → mix) ----------------
@app.get("/api/vocal/engines")
def vocal_engines():
    return {"engines": voicegen_mod.engines(CFG)}


@app.post("/api/melody/compose")
def melody_compose(body: dict):
    """Compose a note-per-syllable melody from a Song Constructor song
    (blocks + key + bpm). Returns the score for the piano-roll + synthesis."""
    if not body.get("blocks"):
        raise HTTPException(400, "no sections to compose for")
    if not any((b.get("lyrics") or "").strip() for b in body["blocks"]):
        raise HTTPException(400, "add lyrics to at least one section to compose a melody")
    try:
        score = melody_mod.compose(
            body, provider=body.get("provider", ""), model=body.get("model", ""),
            claude_model=CFG.get("claude_model", "claude-3-5-sonnet-latest"),
            seed=body.get("seed"))
    except Exception as e:
        raise HTTPException(500, f"compose failed: {e}")
    return {"score": score}


@app.post("/api/melody/midi")
def melody_midi(body: dict):
    score = body.get("score")
    if not score:
        raise HTTPException(400, "no score")
    data = melody_mod.to_midi(score)
    return Response(content=data, media_type="audio/midi",
                    headers={"Content-Disposition": "attachment; filename=melody.mid"})


@app.post("/api/vocal/build")
def vocal_build(body: dict):
    """Synthesize the composed melody into a sung vocal via the chosen engine,
    optionally re-timbre it through RVC, and save it to the library."""
    score = body.get("score")
    engine = body.get("engine", "guide")
    if not score or not score.get("notes"):
        raise HTTPException(400, "compose a melody first (no notes in score)")
    ref = None
    if body.get("reference_src"):
        try:
            ref = open(mix_mod._resolve(body["reference_src"], LIBRARY, STEMS_DIR), "rb").read()
        except Exception as e:
            raise HTTPException(400, f"reference clip not found: {e}")
    # Quality defaults for SoulX (the sweep winner: fp32 + more steps clears the
    # crackle and tightens pitch); still overridable per request via body.opts.
    opts = dict(body.get("opts") or {})
    if engine == "soulx":
        opts = {"n_steps": 200, "fp16": False, "cfg": 3, **opts}
    # Shared 3090: if this build will use the GPU (a host engine or RVC
    # re-timbre), free ComfyUI's VRAM first so models don't collide.
    if engine != "guide" or body.get("retimbre"):
        free_gpu("soulx")
    try:
        wav = voicegen_mod.synthesize(engine, score, CFG, reference=ref, opts=opts)
    except Exception as e:
        raise HTTPException(500, f"synthesis failed ({engine}): {e}")
    voice = body.get("voice")
    if body.get("retimbre") and voice:
        try:
            wav = R.convert(wav, "vocal.wav", voice,
                            transpose=int(body.get("transpose", 0)),
                            f0_method=body.get("f0_method", "rmvpe"),
                            index_rate=float(body.get("index_rate", 0.75)),
                            protect=float(body.get("protect", 0.33)))
        except Exception as e:
            raise HTTPException(500, f"RVC re-timbre failed (is the RVC API running?): {e}")
    jid = uuid.uuid4().hex
    path = os.path.join(LIBRARY, f"{jid}.wav")
    with open(path, "wb") as f:
        f.write(wav)
    save_done_row(jid, "vocal", {"source": "vocal-builder", "engine": engine,
                                 "voice": voice if body.get("retimbre") else None,
                                 "key": score.get("key"), "bpm": score.get("bpm"),
                                 "notes": len(score.get("notes", [])),
                                 "opts": opts}, path)
    return {"job_id": jid, "audio_url": f"/api/audio/{jid}", "status": "done"}


@app.get("/api/vocal/soulx/voices")
def soulx_voices():
    host = CFG.get("soulx_host", "")
    if not host:
        raise HTTPException(400, "soulx_host not configured")
    try:
        return requests.get(f"http://{host}/voices", timeout=10).json()
    except Exception as e:
        raise HTTPException(502, f"SoulX unreachable: {e}")


@app.post("/api/vocal/soulx/prep")
async def soulx_prep(name: str = Form(...), language: str = Form("English"),
                     vocal_sep: bool = Form(True), job_id: str = Form(None),
                     stem_src: str = Form(None), file: UploadFile = File(None)):
    """Register a SoulX reference voice from an uploaded clip, a library track,
    or an already-extracted vocal stem (forwards to the SoulX preprocess
    pipeline). Heavy / one-time."""
    host = CFG.get("soulx_host", "")
    if not host:
        raise HTTPException(400, "soulx_host not configured")
    if file is not None:
        data = await file.read()
        fname = file.filename or "ref.wav"
    elif stem_src:
        try:
            data = open(mix_mod._resolve(stem_src, LIBRARY, STEMS_DIR), "rb").read()
        except Exception as ex:
            raise HTTPException(404, f"stem not found: {ex}")
        fname = "vocal.wav"
    elif job_id:
        src = next((os.path.join(LIBRARY, job_id + e) for e in (".wav", ".mp3", ".flac")
                    if os.path.exists(os.path.join(LIBRARY, job_id + e))), None)
        if not src:
            raise HTTPException(404, "library track not found")
        data = open(src, "rb").read()
        fname = os.path.basename(src)
    else:
        raise HTTPException(400, "provide a file or job_id")
    free_gpu("soulx")  # free ComfyUI + RVC; preprocess is GPU-heavy
    try:
        r = requests.post(f"http://{host}/voices/prep",
                          data={"name": name, "language": language,
                                "vocal_sep": str(vocal_sep).lower()},
                          files={"file": (fname, data, "application/octet-stream")},
                          timeout=1800)
        d = r.json()
    except Exception as e:
        raise HTTPException(502, f"prep failed: {e}")
    if not r.ok:
        raise HTTPException(500, f"preprocess failed: {d.get('error', 'unknown')}")
    return d


@app.post("/api/voiceswap")
async def voiceswap(voice: str = Form(...), job_id: str = Form(None),
                    file: UploadFile = File(None), transpose: int = Form(0),
                    vocal_gain: float = Form(0.0), instr_gain: float = Form(0.0),
                    f0_method: str = Form("rmvpe"), index_rate: float = Form(0.75),
                    protect: float = Form(0.33)):
    """One-click: split a vocal song → re-timbre the vocal via RVC → remix over
    its own instrumental. Everything stays in sync (all from one source)."""
    sid = uuid.uuid4().hex
    work = os.path.join(STEMS_DIR, sid)
    os.makedirs(work, exist_ok=True)
    # resolve source song
    if file is not None:
        inp = os.path.join(work, "input" + (os.path.splitext(file.filename)[1] or ".wav"))
        with open(inp, "wb") as f:
            f.write(await file.read())
    elif job_id:
        src = next((os.path.join(LIBRARY, job_id + e) for e in (".mp3", ".wav", ".flac")
                    if os.path.exists(os.path.join(LIBRARY, job_id + e))), None)
        if not src:
            raise HTTPException(404, "source song not found")
        inp = os.path.join(work, "input" + os.path.splitext(src)[1])
        shutil.copy(src, inp)
    else:
        raise HTTPException(400, "provide a song (job_id or file)")

    # 1) split into vocals + no_vocals (Mac MPS)
    try:
        stem_files = stems_mod.separate(inp, work, mode="vocals")
    except Exception as e:
        raise HTTPException(500, f"split failed: {e}")
    voc = next((f for f in stem_files if os.path.basename(f).startswith("vocals")), None)
    inst = next((f for f in stem_files if "no_vocals" in os.path.basename(f)), None)
    if not voc or not inst:
        raise HTTPException(500, "split did not produce vocal + instrumental stems")

    # 2) re-timbre the vocal via RVC
    free_gpu("rvc")                       # free ComfyUI before RVC runs (shared 3090)
    try:
        with open(voc, "rb") as f:
            conv = R.convert(f.read(), "vocals.wav", voice, transpose=transpose,
                             f0_method=f0_method, index_rate=index_rate, protect=protect)
    except Exception as e:
        raise HTTPException(500, f"voice conversion failed (is the RVC API running?): {e}")
    with open(os.path.join(work, "vocals_converted.wav"), "wb") as f:
        f.write(conv)

    # 3) remix converted vocal over the original instrumental
    try:
        tracks = [{"src": f"/api/stem/{sid}/{os.path.basename(inst)}", "gain_db": instr_gain},
                  {"src": f"/api/stem/{sid}/vocals_converted.wav", "gain_db": vocal_gain}]
        mixed = mix_mod.mix(tracks, LIBRARY, STEMS_DIR, normalize=True)
    except Exception as e:
        raise HTTPException(500, f"remix failed: {e}")

    jid = uuid.uuid4().hex
    with open(os.path.join(LIBRARY, f"{jid}.wav"), "wb") as f:
        f.write(mixed)
    save_done_row(jid, "voiceswap", {"source": job_id or (file.filename if file else ""),
                                     "voice": voice, "transpose": transpose},
                  os.path.join(LIBRARY, f"{jid}.wav"))
    return {"job_id": jid, "audio_url": f"/api/audio/{jid}",
            "stems": {"converted_vocal": f"/api/stem/{sid}/vocals_converted.wav",
                      "instrumental": f"/api/stem/{sid}/{os.path.basename(inst)}"}}


@app.get("/api/stem/{sid}/{name}")
def get_stem(sid: str, name: str):
    path = os.path.join(STEMS_DIR, os.path.basename(sid), os.path.basename(name))
    if not os.path.exists(path):
        raise HTTPException(404, "no stem")
    return FileResponse(path, media_type="audio/wav")


@app.get("/api/audio/{pid}")
def audio(pid: str):
    for ext, mt in ((".mp3", "audio/mpeg"), (".wav", "audio/wav"), (".flac", "audio/flac")):
        path = os.path.join(LIBRARY, f"{pid}{ext}")
        if os.path.exists(path):
            return FileResponse(path, media_type=mt)
    # fall back to the stored path (e.g. stems registered from STEMS_DIR)
    with db() as conn:
        row = conn.execute("SELECT audio FROM jobs WHERE id=?", (pid,)).fetchone()
    if row and row["audio"] and os.path.exists(row["audio"]):
        mt = {".wav": "audio/wav", ".flac": "audio/flac"}.get(
            os.path.splitext(row["audio"])[1].lower(), "audio/mpeg")
        return FileResponse(row["audio"], media_type=mt)
    raise HTTPException(404, "no audio")


def _detect_beats(path):
    """librosa beat tracking (Mac CPU, no GPU). Returns {bpm, beats[], duration} for the
    waveform region-selector's beat markers + snapping."""
    import librosa
    import numpy as _np
    y, sr = librosa.load(path, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    bpm = float(_np.atleast_1d(tempo)[0])
    return {"bpm": round(bpm, 1),
            "beats": [round(float(t), 4) for t in beat_times],
            "duration": round(float(librosa.get_duration(y=y, sr=sr)), 3)}


@app.get("/api/beats/{pid}")
def beats(pid: str):
    src = _lib_source_path(pid)
    if not src:
        raise HTTPException(404, "no audio")
    try:
        return _detect_beats(src)
    except Exception as e:
        raise HTTPException(500, f"beat detection failed: {e}")


@app.post("/api/beats")
async def beats_upload(file: UploadFile = File(...)):
    """Beat detection for an uploaded (not-yet-in-library) repaint source."""
    work = os.path.join(STEMS_DIR, "beats_" + uuid.uuid4().hex)
    os.makedirs(work, exist_ok=True)
    try:
        ext = os.path.splitext(file.filename or "")[1] or ".wav"
        p = os.path.join(work, "src" + ext)
        with open(p, "wb") as f:
            f.write(await file.read())
        return _detect_beats(p)
    except Exception as e:
        raise HTTPException(500, f"beat detection failed: {e}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _lib_source_path(pid):
    """Absolute path to a library item's audio (by id, or the stored DB path)."""
    for ext in (".mp3", ".wav", ".flac"):
        p = os.path.join(LIBRARY, f"{pid}{ext}")
        if os.path.exists(p):
            return p
    with db() as conn:
        row = conn.execute("SELECT audio FROM jobs WHERE id=?", (pid,)).fetchone()
    if row and row["audio"] and os.path.exists(row["audio"]):
        return row["audio"]
    return None


def _has_audio_stream(path):
    """Does this file carry an audio track? A silent video handed to the mp3 path produced a 500
    with a wall of ffmpeg output instead of a usable message."""
    import subprocess
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                              "stream=codec_type", "-of", "csv=p=0", path],
                             capture_output=True, text=True, timeout=30)
        return "audio" in (out.stdout or "")
    except Exception:
        return True          # can't tell - let the export try rather than refuse


def _export_name(pid):
    """Friendly download name: the user-given song title wins (else note/tags/source), with a
    "-v2" suffix when the item is one of several versions of that name."""
    try:
        with db() as conn:
            row = conn.execute("SELECT params,mode,created FROM jobs WHERE id=?", (pid,)).fetchone()
        if row and row["params"]:
            pp = json.loads(row["params"])
            title = str(pp.get("title") or "").strip()
            raw = title or pp.get("note") or pp.get("tags") or pp.get("source") or pid
            slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(raw)).strip("_")[:48] or pid
            if title:                       # version among same mode + same title (oldest=v1)
                with db() as conn:
                    sibs = conn.execute(
                        "SELECT id,params FROM jobs WHERE mode=? AND status='done' ORDER BY created ASC",
                        (row["mode"],)).fetchall()
                vers = [s["id"] for s in sibs
                        if str((json.loads(s["params"]) if s["params"] else {}).get("title") or "").strip().lower() == title.lower()]
                if len(vers) > 1 and pid in vers:
                    slug = f"{slug}-v{vers.index(pid) + 1}"
            return slug
    except Exception:
        pass
    return pid


@app.get("/api/export/{pid}")
def export_audio(pid: str, fmt: str = "mp3"):
    """Download a library item. Video/image items come back as themselves; audio as MP3 (320k),
    already-MP3 passing through and WAVs transcoded. Filename uses the item's note/label.

    Video used to fall through to the audio path: the DB row's stored output IS the .mp4, so
    ffmpeg was asked to make an MP3 out of a silent clip and the download button 500'd on every
    video card [2026-08-16]."""
    pid = os.path.basename(pid)
    # a rendered clip, still or assembled video - hand back the actual file
    for ext, ct in _MEDIA_CT.items():
        mp = os.path.join(LIBRARY, f"{pid}{ext}")
        if os.path.exists(mp):
            return FileResponse(mp, media_type=ct, filename=f"{_export_name(pid)}{ext}")
    src = _lib_source_path(pid)
    if not src:
        raise HTTPException(404, "nothing to export for that id")
    if not _has_audio_stream(src):
        raise HTTPException(400, "that item has no audio track to export")
    # fmt=original hands back the master untouched. Music 3 masters are lossless FLAC, and
    # transcoding one to MP3 just to download it would throw away the reason for keeping it.
    if fmt in ("original", "source", "flac", "wav"):
        ext = os.path.splitext(src)[1].lower()
        ct = {".flac": "audio/flac", ".wav": "audio/wav", ".mp3": "audio/mpeg"}.get(ext, "application/octet-stream")
        return FileResponse(src, media_type=ct, filename=f"{_export_name(pid)}{ext}")
    fname = f"{_export_name(pid)}.mp3"
    headers = {"Content-Disposition": f'attachment; filename="{fname}"'}
    if src.lower().endswith(".mp3"):
        return FileResponse(src, media_type="audio/mpeg", filename=fname)
    import subprocess
    try:
        out = subprocess.run(["ffmpeg", "-y", "-i", src, "-f", "mp3", "-b:a", "320k", "pipe:1"],
                             capture_output=True, timeout=120)
        if out.returncode != 0 or not out.stdout:
            raise RuntimeError((out.stderr or b"")[-300:].decode("utf-8", "ignore"))
    except Exception as e:
        raise HTTPException(500, f"mp3 export failed: {e}")
    return Response(content=out.stdout, media_type="audio/mpeg", headers=headers)


# ==================== MiniMax Music 3 (second generation engine) ====================
# Deliberately separate from /api/generate. Music 3 wants a 4000+ character structured caption
# where ACE-Step wants ~10-12 tag phrases, so sharing a form would produce bad prompts for both.

@app.get("/api/music3/available")
def music3_available():
    """Whether the box can actually run this: the nodes ship with ComfyUI 0.33+, but the 14.3GB of
    weights are a separate download and their absence otherwise surfaces as a validation error
    several seconds after pressing Generate."""
    try:
        nodes = C.has_node("MiniMaxMusic3TextEncode") and C.has_node("EmptyMiniMaxMusic3LatentAudio")
    except Exception as e:
        return {"available": False, "reason": f"ComfyUI unreachable: {e}"}
    if not nodes:
        return {"available": False, "reason": "ComfyUI has no MiniMax Music 3 nodes (needs 0.33.0+)"}
    try:
        unets = set(C.models("diffusion_models"))
        clips = set(C.models("text_encoders"))
        vaes = set(C.models("vae"))
    except Exception as e:
        return {"available": False, "reason": f"could not list models: {e}"}
    missing = [n for n, have in ((music3_mod.MODELS["unet"], unets),
                                 (music3_mod.MODELS["clip"], clips),
                                 (music3_mod.MODELS["vae"], vaes))
               if n not in have]
    if missing:
        return {"available": False, "reason": "missing weights: " + ", ".join(missing)}
    return {"available": True}


@app.get("/api/music3/schema")
def music3_schema():
    """Caption field list, per-field guidance and the documented section tags. Served rather than
    duplicated in the UI so the guidance has one home."""
    return {
        "fields": [{"key": k, "group": g, "help": h} for k, g, h in music3_mod.CAPTION_FIELDS],
        "groups": music3_mod.GROUP_ORDER,
        "tags": music3_mod.SECTION_TAGS,
        "defaults": music3_mod.DEFAULTS,
        "max_seconds": music3_mod.MAX_SECONDS,
    }


@app.post("/api/music3/from_song")
def music3_from_song(p: dict):
    """Convert a Song tab arrangement into caption fields + bare-tag lyrics.

    The per-block style strings are compiled into the caption's progression fields, NOT into the
    lyric tags - measured: anything after the section name inside brackets gets sung aloud."""
    song = p.get("song") or {}
    if not (song.get("blocks") or []):
        raise HTTPException(400, "that project has no song arrangement to import")
    return {"fields": music3_mod.song_to_fields(song), "lyrics": music3_mod.song_to_lyrics(song)}


@app.post("/api/music3/to_song")
def music3_to_song(body: dict):
    """The REVERSE of /api/music3/from_song: a Music 3 caption + bare-tag lyrics -> a Song tab
    (ACE-Step) arrangement. Deterministic where fidelity matters (lyrics kept verbatim per
    section; bpm/key/genre parsed straight out of Basic Attributes); the LLM only does the
    semantic compression ACE needs - a 4000-char caption boiled down to 10-12 dense tag
    phrases, plus a 1-3 word delivery style and a duration per section.
    Body: {fields, lyrics, title?, seconds? (total), provider?, model?}.
    Returns {title, tags, bpm, keyscale, instrumental, blocks:[{type,seconds,lyrics,style}]}."""
    fields = body.get("fields") or {}
    sections = music3_mod.lyrics_to_sections(body.get("lyrics") or "")
    if not sections:
        raise HTTPException(400, "no [Section] markers found in the Music 3 lyrics")
    caption = music3_mod.assemble_caption(fields)
    if not caption.strip():
        raise HTTPException(400, "the Music 3 caption fields are empty - nothing to translate")
    bpm, keyscale, genre = music3_mod.parse_basic_attributes(fields)
    total = int(float(body.get("seconds") or 0)) or None
    provider = body.get("provider") or llm_mod.best_provider()
    model = body.get("model") or ""
    if not model and provider in ("claude_sub", "claude_code", "claude"):
        model = "claude-sonnet-5"
    listing = "\n".join(
        f"  {i + 1}. {s['type']}" + (f' - starts "{s["lyrics"].splitlines()[0][:60]}"' if s["lyrics"] else " (no lyrics)")
        for i, s in enumerate(sections))
    system = ("You translate a MiniMax Music 3 caption into an ACE-Step song-builder setup. "
              "The two engines want OPPOSITE prompt shapes: Music 3 is a 4000-character prose "
              "caption; ACE-Step wants 10-12 DENSE comma-separated tag phrases TOTAL - more than "
              "that measurably produces disjointed takes. Output STRICT JSON ONLY (no prose, no "
              "markdown fences).")
    prompt = f"""The Music 3 caption:
---
{caption[:6000]}
---
The song's sections, in order:
{listing}
{f"Total song length target: about {total} seconds." if total else "No total length given - use sensible section lengths."}

Return ONLY a JSON object:
{{"tags": "<10-12 dense comma-separated tag phrases for ACE-Step. The GENRE comes first{f" (the caption says: {genre})" if genre else ""}. Compress the caption's essence: core instruments, production feel, vocal character, tempo feel. No sentences, no negatives, no section names.>",
  "sections": [<EXACTLY {len(sections)} objects, one per section above, in order:
    {{"style": "<1-3 word delivery/feel cue for that section (e.g. anthemic, whispered, half-time, gang vocals, soaring) drawn from the caption's per-section direction; "" if nothing notable>",
     "seconds": <int, the section's length: intros/outros ~8-12, verses/choruses ~16-28, bridge/solo ~12-24{", scaled so the total is about " + str(total) if total else ""}>}}>]}}"""
    claude_model = CFG.get("claude_model", "claude-3-5-sonnet-latest")
    try:
        text = llm_mod.complete(provider, model, system, prompt, claude_model, timeout=300)
        m = re.search(r"\{.*\}", text, re.S)
        parsed = json.loads(m.group(0) if m else text)
    except Exception as e:
        raise HTTPException(500, f"translation failed ({provider}): {e}")
    tags = ", ".join(t.strip() for t in str(parsed.get("tags") or "").split(",") if t.strip())[:400]
    per = parsed.get("sections") or []
    blocks = []
    for i, s in enumerate(sections):
        p = per[i] if i < len(per) and isinstance(per[i], dict) else {}
        # cap the style at 3 words WITHOUT leaving a dangling connective ("soaring wall of")
        words = str(p.get("style") or "").split()[:3]
        while words and words[-1].lower() in ("of", "the", "a", "an", "and", "with", "into"):
            words.pop()
        try:
            secs = max(4, min(40, int(p.get("seconds") or 0)))
        except (TypeError, ValueError):
            secs = 0
        blocks.append({"type": s["type"], "seconds": secs or 16,
                       "lyrics": s["lyrics"], "style": " ".join(words)})
    # The LLM routinely misses the total-length target; ACE timing is explicit, so honor the
    # requested total deterministically: scale every section proportionally when off by >10%.
    if total:
        cur = sum(b["seconds"] for b in blocks)
        if cur and abs(cur - total) > total * 0.10:
            f = total / cur
            for b in blocks:
                b["seconds"] = max(4, min(60, round(b["seconds"] * f)))
    return {"title": str(body.get("title") or "").strip(),
            "tags": tags, "bpm": bpm, "keyscale": keyscale,
            "instrumental": not any(b["lyrics"] for b in blocks),
            "blocks": blocks, "provider": provider}


@app.post("/api/music3/preview")
def music3_preview(p: dict):
    """Assemble the caption and report its size, so the 5000-token ceiling is visible while
    authoring rather than as a failure at submit time."""
    caption = (music3_mod.assemble_caption(p["fields"]) if p.get("fields")
               else (p.get("caption") or ""))
    lyrics = p.get("lyrics") or ""
    # Rough: the real count needs the model's tokenizer, which lives on the box. ~4 chars/token is
    # close enough to warn on, and the backend re-checks nothing the box will not catch itself.
    approx = (len(caption) + len(lyrics)) // 4
    return {"caption": caption, "chars": len(caption), "approx_tokens": approx,
            "token_limit": music3_mod.MAX_PROMPT_TOKENS,
            "over_limit": approx > music3_mod.MAX_PROMPT_TOKENS}


@app.post("/api/music3/write")
def music3_write(p: dict):
    """Author or rewrite the caption fields. writer="ours" uses what we measured on this box;
    writer="skill" runs MiniMax's vendored music-caption-rewriter by its own method.

    Returns the NEW fields alongside the ones sent in, so the UI can show a per-field before/after
    and accept a rewrite field by field rather than all or nothing."""
    from . import music3_writer
    writer = (p.get("writer") or "ours").lower()
    before = p.get("fields") or {}
    args = dict(brief=(p.get("brief") or ""), fields=before, lyrics=(p.get("lyrics") or ""),
                provider=p.get("provider") or llm_mod.best_provider(),
                model=p.get("model") or "",
                claude_model=CFG.get("claude_model", "claude-3-5-sonnet-latest"))
    try:
        if writer == "skill":
            r = music3_writer.write_skill(**args, pick_family=(p.get("family") or ""),
                                          pick_templates=(p.get("templates") or None))
        else:
            r = music3_writer.write_ours(**args)
    except Exception as e:
        raise HTTPException(500, f"{writer} writer failed: {e}")
    changed = [k for k, v in r["fields"].items() if (v or "").strip() != (before.get(k) or "").strip()]
    out = {**r, "before": before, "changed": changed}
    # Lyrics are drafted ONLY when the box arrived empty - populated lyrics are never rewritten
    # (same contract as the ACE writer). Shared step for both writer paths: the skill's own rules
    # forbid lyric content in its output, so lyric writing is its own call either way.
    if not (p.get("lyrics") or "").strip() and p.get("write_lyrics", True):
        try:
            out["lyrics"] = music3_writer.write_lyrics(
                p.get("brief") or "", r["fields"],
                provider=args["provider"], model=args["model"], claude_model=args["claude_model"])
        except Exception:
            pass   # lyrics are a bonus on top of the caption; their failure should not sink it
    return out


@app.get("/api/music3/writer_status")
def music3_writer_status():
    from . import music3_writer
    return {"skill_installed": music3_writer.available(), "provider": llm_mod.best_provider()}


@app.get("/api/music3/references")
def music3_references(family: str = ""):
    """The skill's style families, or the cards inside one. Pinning references by hand is the only
    way to make a skill rewrite reproducible: auto-routing chose a different template trio on two
    consecutive runs of the same brief."""
    from . import music3_writer
    if not music3_writer.available():
        raise HTTPException(400, "the caption-rewriter skill is not installed")
    try:
        return {"cards": music3_writer.cards(family)} if family else {"families": music3_writer.families()}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/music3/encode_ref")
async def music3_encode_ref(request: Request):
    """Encode audio into a Music 3 flow latent, ON THE BOX, staged in ComfyUI's input dir.

    Source: JSON {src: library id} or multipart {file: upload}. The audio is uploaded to
    ComfyUI's input dir, then the :5080 helper's /dav/encode runs the encode on the 3090 in a
    SUBPROCESS that exits when done - so every byte of RAM and VRAM it used is released by
    process termination, nothing stays resident. A few seconds of GPU; the helper's first call
    fetches the 292MB encoder weights onto the box. There is deliberately no Mac fallback: a
    full-length CPU encode measured 9m54s for a 4-minute track, and without the box there is
    nothing to render the latent with anyway. The returned `latent` name goes into /generate's
    `audio_ref`; one encode is reusable across any number of takes and strengths."""
    if not LORA_UPLOAD_HOST:
        raise HTTPException(400, "lora_upload_host not set - the box helper (:5080) does the encoding")
    path, cleanup = None, None
    max_seconds = 0.0
    ctype = (request.headers.get("content-type") or "")
    if ctype.startswith("multipart/"):
        form = await request.form()
        up = form.get("file")
        if up is None:
            raise HTTPException(400, "multipart needs a `file`")
        suffix = os.path.splitext(getattr(up, "filename", "") or "ref")[1] or ".bin"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(await up.read())
            path = cleanup = f.name
        max_seconds = float(form.get("max_seconds") or 0)
    else:
        p = await request.json()
        max_seconds = float(p.get("max_seconds") or 0)
        src = (p.get("src") or "").strip()
        path = _lib_source_path(src) if src else None
    if not path or not os.path.isfile(path):
        raise HTTPException(404, "source track not found")

    try:
        with open(path, "rb") as f:
            audio = f.read()
        stem = f"m3ref_{hashlib.md5(audio).hexdigest()[:16]}"
        try:
            in_name = C.upload_audio(audio, stem + os.path.splitext(path)[1].lower())
        except Exception as e:
            raise HTTPException(502, f"uploading the source audio to the box failed: {e}")
        try:
            r = requests.post(f"http://{LORA_UPLOAD_HOST}/dav/encode",
                              json={"input_name": in_name, "out_name": stem + ".latent",
                                    "max_seconds": max_seconds}, timeout=1800)
        except Exception as e:
            raise HTTPException(502, f"box helper unreachable at {LORA_UPLOAD_HOST}: {e}")
        if r.status_code == 404:
            raise HTTPException(502, "the box helper predates /dav/encode - restart it "
                                     "(the run_lora_upload.bat window) to pick up the update")
        if not r.ok:
            raise HTTPException(502, f"box encode failed: {r.text[:400]}")
        info = r.json()
        return {"latent": info.pop("latent", stem + ".latent"), "where": "box", **info}
    finally:
        if cleanup:
            try:
                os.unlink(cleanup)
            except OSError:
                pass


@app.post("/api/music3/fetch_lyrics")
def music3_fetch_lyrics(p: dict):
    """Online lyrics for a known song (LRCLIB then lyrics.ovh - the exact chain LoRA training
    captioning uses; never transcription). Artist/title explicit, or resolved from a library
    track's tags/filename via `src`."""
    from . import lyrics_fetch
    artist = (p.get("artist") or "").strip()
    title = (p.get("title") or "").strip()
    path = _lib_source_path((p.get("src") or "").strip()) if p.get("src") else None
    artist, title = lyrics_fetch.resolve_artist_title(path=path, artist=artist or None,
                                                      title=title or None)
    if not (artist and title):
        raise HTTPException(400, "need an artist and title (could not resolve them from the track)")
    ly, source = lyrics_fetch.fetch_online(artist, title)
    if not ly:
        raise HTTPException(404, f"no lyrics found online for {artist} - {title}")
    return {"lyrics": ly, "source": source, "artist": artist, "title": title}


@app.post("/api/music3/generate")
def music3_generate(p: dict):
    caption = (music3_mod.assemble_caption(p["fields"]) if p.get("fields")
               else (p.get("caption") or ""))
    if not caption.strip():
        raise HTTPException(400, "a caption is required (Music 3 has nothing else to go on)")
    seed = int(p.get("seed") or random.randint(1, 2**31 - 1))
    try:
        graph, resolved = music3_mod.build_graph({**p, "caption": caption, "seed": seed})
    except ValueError as e:
        raise HTTPException(400, str(e))
    resolved["title"] = (p.get("title") or "").strip()
    resolved["note"] = (p.get("note") or "").strip()
    resolved["fields"] = p.get("fields") or {}
    try:
        res = submit_comfy(graph)
    except Exception as e:
        raise HTTPException(500, f"submit failed: {e}")
    if res.get("node_errors"):
        raise HTTPException(400, f"node errors: {res['node_errors']}")
    pid = res["prompt_id"]
    with LOCK:
        JOBS[pid] = _new_job(resolved, "music3")
    save_job(pid)
    return {"job_id": pid, "seed": seed}


# ==================== Metal LoRA training (drives the box engine pipeline) ====================
# Mac orchestration over the box's HTTP pipeline (METAL_LORA_PLAN): enrich+upload a dataset
# (Mac), then scan -> review -> auto-label -> save -> preprocess -> train -> export -> load.
# The box upload helper returns all box-side paths; we pass them to the engine verbatim.

def _lora_require_engine():
    if not ACESTEP_HOST:
        raise HTTPException(400, "acestep_host not set — the official ACE-Step engine is needed for LoRA training")

def _lora_require_upload():
    if not LORA_UPLOAD_HOST:
        raise HTTPException(400, "lora_upload_host not set — install/run the box LoRA upload helper (LORA-UPLOAD_AUTO_INSTALL.bat)")

def _lora_paths(dataset):
    _lora_require_upload()
    return lora_up.dataset_paths(LORA_UPLOAD_HOST, dataset)


# Canonical LoRA training/labeling models. Pinned, NOT read from engine live state —
# if the engine was restarted (or another flow swapped models), we must put it back
# into the known-good config rather than preserve whatever happens to be loaded.
# - xl-BASE is the labeling DiT now (changed 2026-06-09): we train on xl-base
#   [[lora-train-on-xl-base]], so label on it too for consistency (the DiT barely
#   matters for captioning - the 4B LM does the work - but keep one model across the run).
#   Only used by _ensure_labeling_ready (labeling path); training inits xl-base explicitly.
# - 4B LM produces the rich prose captions the merge layer is designed around;
#   0.6B (the engine's default when lm_model_path is omitted) gives weaker tags.
LORA_DIT_MODEL = "acestep-v15-xl-base"
LORA_LM_MODEL  = "acestep-5Hz-lm-4B"


def _ensure_training_ready(host, dit_model=None):
    """Prep engine for training via /v1/reinitialize.

    The engine's start_lokr/start handlers do their OWN component management at
    training start (`RuntimeComponentManager.unload_llm()` + decoder-to-GPU + VAE
    offload) — verified in acestep/api/train_api_lokr_start_route.py. So we do NOT
    need to call /v1/init {init_llm: false} ourselves. In fact, doing so is harmful:

        acestep/core/generation/handler/init_service_orchestrator.py `initialize_service`
        REASSIGNS self.model/self.vae/self.text_encoder when loading new ones, but
        only sets the OLD ones to None on FAILURE paths. On success it never calls
        gc.collect() or torch.cuda.empty_cache(). Each /v1/init leaks ~4 GB of
        orphaned CUDA tensors. Over a session of many init calls (e.g. our model-
        pinning fix-up loop) the leaks compound until VRAM saturates and training
        goes bandwidth-bound (the symptom: 99% GPU util + 24 GB VRAM + silent fan
        + ~7x slower steps; observed 2026-05-29).

    /v1/reinitialize is the documented fix — the engine's own start_lokr error path
    literally says "reload the model via /v1/reinitialize before training." It runs
    gc.collect() + torch.cuda.empty_cache() and moves the decoder to GPU. It will
    reload the LM if previously unloaded (so the next training start can properly
    unload+empty_cache it) but skips it if already loaded. Cheap, idempotent.

    `dit_model` is accepted for API compat but unused — the engine's reinitialize
    uses last_init_params, so the currently-loaded DiT (set via /v1/init earlier
    by _ensure_labeling_ready) is preserved."""
    _ = dit_model  # api compat
    try:
        ace_train.reinitialize(host, timeout=600)
    except Exception as e:
        print(f"[lora] _ensure_training_ready /v1/reinitialize warning: {e}")


def _ensure_labeling_ready(host, lm_model_path=None, dit_model=None):
    """Make sure the engine has pinned DiT + LM loaded for labeling/preprocess.

    Idempotent: skips the /v1/init call entirely when the engine is already in the
    target state. Avoiding /v1/init when not needed dodges the leak documented in
    _ensure_training_ready (orphaned CUDA tensors per init). When the state is
    wrong (e.g. 0.6B LM loaded instead of 4B, or wrong DiT), we DO init — the cost
    of one leaked re-init is worth the correctness."""
    want_dit = dit_model or LORA_DIT_MODEL
    want_lm = lm_model_path or LORA_LM_MODEL
    try:
        health = (acestep_py.health(host).get("data") or {})
        if (health.get("loaded_model") == want_dit
                and health.get("llm_initialized")
                and health.get("loaded_lm_model") == want_lm):
            return  # already correct — skip the leak-prone init
    except Exception:
        pass  # fall through to the explicit init
    try:
        ace_train.init(host, model=want_dit, init_llm=True,
                       lm_model_path=want_lm, timeout=600)
    except Exception as e:
        print(f"[lora] _ensure_labeling_ready warning: {e}")


@app.get("/api/lora/status")
def lora_status(dataset: str = "crucible_metal"):
    """Preflight for the Training tab: which box services are reachable + live state.

    `dataset` query param routes the training-status fallback to the right
    persisted snapshot when the engine state is empty (post-restart). Default
    preserves existing UI behavior (was crucible_metal before per-dataset
    snapshots existed)."""
    out = {"train_available": bool(ACESTEP_HOST), "upload_available": bool(LORA_UPLOAD_HOST)}
    if ACESTEP_HOST:
        try: out["engine"] = acestep_py.health(ACESTEP_HOST)
        except Exception as e: out["engine_error"] = str(e)
        try:
            # Use the with-fallback helper so a completed run still shows
            # progress+loss+config after an engine restart.
            tr = _lora_training_status_with_fallback(dataset)
            # Engine sets tensorboard_url to http://localhost:6006 — that's loopback
            # on the BOX, useless from any other machine. Rewrite to the box's actual
            # host so the UI link works from the Mac browser. Best-effort: only swap
            # the host part, preserve the port + path.
            try:
                tb = (tr or {}).get("tensorboard_url") or ""
                if tb and ("localhost" in tb or "127.0.0.1" in tb):
                    from urllib.parse import urlparse
                    # ACESTEP_HOST may be stored without a scheme (e.g. "192.168.1.201:8001");
                    # urlparse needs a scheme to populate .hostname, so add one if missing.
                    host_str = ACESTEP_HOST if "://" in ACESTEP_HOST else f"http://{ACESTEP_HOST}"
                    box_host = urlparse(host_str).hostname or ""
                    if box_host:
                        tr["tensorboard_url"] = tb.replace("localhost", box_host).replace("127.0.0.1", box_host)
            except Exception:
                pass
            out["training"] = tr
        except Exception: pass
        # Preprocess lives in its own engine task pool — include it here so a single
        # status poll covers the whole Save → Preprocess → Train chain. Otherwise the
        # encode-to-tensors step (the long pause between clicking the train button and
        # seeing epoch 1) is invisible.
        try: out["preprocess"] = ace_train.preprocess_status(ACESTEP_HOST)
        except Exception: pass
        try: out["lora"] = ace_train.lora_status(ACESTEP_HOST)
        except Exception: pass
    if LORA_UPLOAD_HOST:
        try: out["upload"] = lora_up.health(LORA_UPLOAD_HOST)
        except Exception as e: out["upload_error"] = str(e)
    return out


@app.post("/api/lora/dataset/add")
async def lora_dataset_add(file: UploadFile = File(...), dataset: str = Form("crucible_metal"),
                           instrumental: bool = Form(False), artist: str = Form(None),
                           title: str = Form(None), allow_online: bool = Form(True),
                           whisper_size: str = Form("small")):
    """Enrich ONE picked track on the Mac (librosa bpm/key + online/whisper lyrics) and
    upload the audio + {name}.json + {name}.lyrics.txt bundle to the box dataset folder.
    Returns the per-track preview (bpm/key/lyrics/source) for the review table."""
    _lora_require_upload()
    data = await file.read()
    files, info = lora_ds.bundle_for_track(
        data, file.filename, instrumental=instrumental, artist=(artist or None),
        title=(title or None), allow_online=allow_online, whisper_size=whisper_size,
        analyze_host=ANALYZE_HOST, lastfm_key=CFG.get("lastfm_key", ""),
        acoustid_key=CFG.get("acoustid_key", ""))
    lora_up.dataset_new(LORA_UPLOAD_HOST, dataset)
    up = lora_up.upload(LORA_UPLOAD_HOST, dataset, files)
    info["uploaded"] = up.get("count")
    info["lyrics"] = next((d.decode("utf-8", "ignore") for (n, d) in files if n.endswith(".lyrics.txt")), "")
    return info


@app.post("/api/lora/dataset/scan")
def lora_dataset_scan(body: dict):
    """Load the uploaded dataset into the engine for review/training."""
    _lora_require_engine()
    dataset = body.get("dataset", "crucible_metal")
    p = _lora_paths(dataset)
    free_gpu("acestep")
    return ace_train.dataset_scan(ACESTEP_HOST, p["data_dir"], dataset_name=dataset,
                                  all_instrumental=bool(body.get("instrumental", False)))


@app.get("/api/lora/dataset/samples")
def lora_dataset_samples(dataset: str = "crucible_metal"):
    """Engine's loaded-samples list. Auto-loads the dataset.json on demand if
    the engine has no dataset in memory — engine state is wiped on every
    restart, but the on-disk dataset.json is durable. This keeps the UI
    sample table populated across engine restarts without making the user
    re-click Step 2's 'Load dataset into engine' button."""
    _lora_require_engine()
    out = ace_train.dataset_samples(ACESTEP_HOST)
    # Empty / missing samples → try to load the dataset.json on the box if it
    # exists, then re-query. Best effort; ignore errors to keep the route
    # behavior compatible with no-dataset-configured callers.
    samples = (out.get("data", out).get("samples", []) if isinstance(out, dict) else []) or []
    if not samples and dataset:
        try:
            if LORA_UPLOAD_HOST:
                p = _lora_paths(dataset)
                eng = ACESTEP_HOST if ACESTEP_HOST.startswith("http") else f"http://{ACESTEP_HOST}"
                # Use raw requests — ace_train doesn't expose dataset_load yet.
                import requests as _r
                r = _r.post(f"{eng}/v1/dataset/load",
                            json={"dataset_path": p["dataset_json"]}, timeout=120)
                if r.status_code == 200:
                    out = ace_train.dataset_samples(ACESTEP_HOST)
        except Exception:
            pass
    return out


@app.put("/api/lora/dataset/sample/{idx}")
def lora_dataset_sample_put(idx: int, body: dict):
    """Save a corrected entry (lyrics/caption/bpm/keyscale) — the review step."""
    _lora_require_engine()
    return ace_train.dataset_put_sample(ACESTEP_HOST, idx, body)


# Per-dataset cache of caption seeds (Crucible enrichment) captured RIGHT BEFORE the LM
# runs, so we can merge them with the LM's output afterward (preserves MB/Last.fm/CLAP
# signal while still letting the LM contribute on every track).
_AUTOLABEL_SEED: dict = {}


def _samples_list(host):
    """Normalize the engine's samples response into a list of dicts."""
    s = ace_train.dataset_samples(host) or []
    if isinstance(s, dict):
        s = s.get("samples") or s.get("data") or []
    return list(s) if s else []


def _sample_idx(s, i):
    v = s.get("sample_idx")
    if v is None:
        v = s.get("index", i)
    return v


@app.post("/api/lora/dataset/autolabel")
def lora_dataset_autolabel(body: dict):
    """Caption the dataset via the box LM (writes only the 'caption' field; lyrics/bpm/
    key/etc untouched). Ensures the LM is loaded first (a prior training run will have
    dropped it via _ensure_training_ready). Returns the task; poll status.

    Default behavior is **merge=True**: caption seeds (our MB / Last.fm / CLAP signal)
    are cached BEFORE the LM runs, and only_unlabeled is forced to False so the LM
    captions every track. Call /autolabel_merge after the task completes to combine
    seed + LM tags back into each track's caption."""
    _lora_require_engine()
    dataset = body.get("dataset", "crucible_metal")
    merge = bool(body.get("merge", True))
    lm_model_path = body.get("lm_model_path") or LORA_LM_MODEL
    free_gpu("acestep")
    _ensure_labeling_ready(ACESTEP_HOST, lm_model_path=lm_model_path,
                           dit_model=body.get("dit_model"))
    if merge:
        try:
            samples = _samples_list(ACESTEP_HOST)
            _AUTOLABEL_SEED[dataset] = {_sample_idx(s, i): (s.get("caption") or "")
                                        for i, s in enumerate(samples)}
        except Exception as e:
            print(f"[lora] autolabel seed-cache failed: {e}")
            _AUTOLABEL_SEED.pop(dataset, None)
    only_unlabeled = False if merge else bool(body.get("only_unlabeled", True))
    return ace_train.dataset_auto_label_async(ACESTEP_HOST,
                                              only_unlabeled=only_unlabeled,
                                              transcribe_lyrics=False, format_lyrics=False,
                                              lm_model_path=lm_model_path)


@app.post("/api/lora/dataset/autolabel_merge")
def lora_dataset_autolabel_merge(body: dict):
    """After /autolabel (merge=True) completes, combine each track's cached seed
    caption with the LM-produced caption (de-duped, specific subgenres first via
    caption_fetch._merge), and PUT the merged caption back to the engine. Preserves
    MB / Last.fm / CLAP signal while keeping the LM's contribution."""
    _lora_require_engine()
    from . import caption_fetch
    dataset = body.get("dataset", "crucible_metal")
    seeds = _AUTOLABEL_SEED.pop(dataset, None)
    if not seeds:
        raise HTTPException(400, "no seed cache for this dataset — call /autolabel with merge=true first")
    samples = _samples_list(ACESTEP_HOST)
    results = []
    for i, s in enumerate(samples):
        idx = _sample_idx(s, i)
        seed = (seeds.get(idx) or "").strip()
        lm_cap = (s.get("caption") or "").strip()
        # The LM produces prose (verified by probe — long descriptive sentences); naive
        # comma-split dedup would drop big chunks. merge_seed_with_lm picks the right
        # strategy per detected format.
        merged = caption_fetch.merge_seed_with_lm(seed, lm_cap)
        if merged and merged != lm_cap:
            put_body = {"sample_idx": idx, "caption": merged}
            for k in ("lyrics", "bpm", "keyscale", "timesignature", "language",
                      "is_instrumental", "genre", "prompt_override"):
                if s.get(k) is not None:
                    put_body[k] = s.get(k)
            try:
                ace_train.dataset_put_sample(ACESTEP_HOST, idx, put_body)
                results.append({"idx": idx, "seed": seed, "lm": lm_cap, "merged": merged})
            except Exception as e:
                results.append({"idx": idx, "error": str(e)})
        else:
            results.append({"idx": idx, "skipped": True, "caption": merged or lm_cap})
    # CRITICAL: persist the merged captions to dataset.json on the box.  Without
    # this, the engine's in-memory caption state is correct but a subsequent
    # engine restart (required before training per [[engine-fresh-boot-for-lora]])
    # reloads dataset.json — which still has the pre-merge seed-only captions —
    # so the UI shows stale data and ANY re-preprocess after restart would bake
    # those stale captions into the tensors. Preprocess that ran BEFORE the
    # restart is unaffected (tensors already have merged captions baked in).
    # Bug discovered 2026-05-30 during the 8-track Nightwish discrete retrain.
    try:
        p = _lora_paths(dataset)
        ace_train.dataset_save(ACESTEP_HOST, p["dataset_json"], dataset_name=dataset)
        persisted = True
    except Exception as e:
        persisted = False
        print(f"[lora] autolabel_merge: dataset save warning: {e}")
    return {"ok": True, "merged_count": sum(1 for r in results if r.get("merged")),
            "persisted_to_disk": persisted, "results": results}


@app.get("/api/lora/dataset/autolabel/status")
def lora_dataset_autolabel_status():
    _lora_require_engine()
    return ace_train.auto_label_status(ACESTEP_HOST)


@app.post("/api/lora/dataset/save")
def lora_dataset_save(body: dict):
    _lora_require_engine()
    dataset = body.get("dataset", "crucible_metal")
    p = _lora_paths(dataset)
    return ace_train.dataset_save(ACESTEP_HOST, p["dataset_json"], dataset_name=dataset)


@app.post("/api/lora/dataset/preprocess")
def lora_dataset_preprocess(body: dict):
    """Encode the dataset audio to latents/tensors on the box GPU (async).
    Pinned LM + xl-sft DiT BEFORE preprocess: the engine's preprocess pipeline uses
    the LM to encode captions into the training tensors, then offloads it during the
    GPU-heavy audio encode (engine log: 'LLM was temporarily unloaded during
    preprocessing and restored afterward'). Without an LM at start the engine bails
    with '❌ Model not initialized' and writes 0 tensors. Train (the next step) is
    what needs the LM DROPPED — handled separately on /api/lora/train."""
    _lora_require_engine()
    dataset = body.get("dataset", "crucible_metal")
    p = _lora_paths(dataset)
    free_gpu("acestep")
    _ensure_labeling_ready(ACESTEP_HOST, dit_model=body.get("dit_model"))
    return ace_train.dataset_preprocess_async(ACESTEP_HOST, p["tensor_dir"],
                                              skip_existing=bool(body.get("skip_existing", False)))


@app.get("/api/lora/dataset/preprocess/status")
def lora_dataset_preprocess_status():
    _lora_require_engine()
    return ace_train.preprocess_status(ACESTEP_HOST)


# ==================== LoRA training history capture ====================
# The engine's /v1/training/status route doesn't expose plot_best_step or the
# per-epoch val curve (even though Patch D writes them into training_state
# server-side). Without that we can't tell what epoch the best checkpoint
# landed at — making it impossible to rationally tune train_epochs for the
# NEXT run. Fix: poll /v1/training/status fast enough to catch the 🏆 / 🧪
# status strings (which my Patch D yields on every val + best save), parse +
# dedupe them, persist on training end. No engine patch required.
_LORA_TRAIN_HISTORY: Dict[str, Dict[str, Any]] = {}
_LORA_TRAIN_HISTORY_LOCK = threading.Lock()
_LORA_TRAIN_HISTORY_DIR = os.path.join(LIBRARY, "lora_train_history")
_BEST_STATUS_RE = re.compile(r"New best val_loss ([\d.]+) at epoch (\d+)")
_VAL_STATUS_RE = re.compile(r"Val loss @ epoch (\d+):\s*([\d.]+)")


def _lora_history_path(dataset: str) -> str:
    return os.path.join(_LORA_TRAIN_HISTORY_DIR, f"{dataset}.json")


def _lora_history_load(dataset: str) -> Dict[str, Any]:
    path = _lora_history_path(dataset)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _lora_history_persist(dataset: str) -> None:
    with _LORA_TRAIN_HISTORY_LOCK:
        snap = dict(_LORA_TRAIN_HISTORY.get(dataset, {}))
    os.makedirs(_LORA_TRAIN_HISTORY_DIR, exist_ok=True)
    try:
        with open(_lora_history_path(dataset), "w") as f:
            json.dump(snap, f, indent=2)
    except Exception as e:
        print(f"[lora] history persist warning ({dataset}): {e}")


def _lora_train_history_poller(dataset: str, host: str, started_at: float) -> None:
    """Watch engine training status for Patch D's 🏆/🧪 messages.

    Polls every ~2s because engine yields fly fast (~5-7 per epoch at ~38s/ep
    → status string changes every ~5-6s). Misses are tolerable for the val
    curve (sparse data is still useful) but we really want to catch the
    🏆-best messages — those are how we pick `train_epochs` going forward.
    """
    seen_best: set = set()
    seen_val: set = set()
    consecutive_idle = 0
    while True:
        try:
            res = ace_train.training_status(host)
            d = res.get("data", res) if isinstance(res, dict) else {}
            status = str(d.get("status", "") or "")
            is_training = bool(d.get("is_training", False))

            for m in _BEST_STATUS_RE.finditer(status):
                val_loss = float(m.group(1)); epoch = int(m.group(2))
                key = (epoch, val_loss)
                if key in seen_best:
                    continue
                seen_best.add(key)
                with _LORA_TRAIN_HISTORY_LOCK:
                    h = _LORA_TRAIN_HISTORY.setdefault(dataset, {
                        "started_at": started_at, "best": [], "val": [],
                    })
                    h["best"].append({
                        "epoch": epoch, "val_loss": val_loss, "ts": time.time(),
                    })
                _lora_history_persist(dataset)

            for m in _VAL_STATUS_RE.finditer(status):
                epoch = int(m.group(1)); val_loss = float(m.group(2))
                key = (epoch, val_loss)
                if key in seen_val:
                    continue
                seen_val.add(key)
                with _LORA_TRAIN_HISTORY_LOCK:
                    h = _LORA_TRAIN_HISTORY.setdefault(dataset, {
                        "started_at": started_at, "best": [], "val": [],
                    })
                    h["val"].append({
                        "epoch": epoch, "val_loss": val_loss, "ts": time.time(),
                    })

            if not is_training:
                consecutive_idle += 1
                # Give the engine a couple of poll cycles to finalize, in case
                # there's a trailing 🏆 status between is_training flipping
                # False and the very last yield being read.
                if consecutive_idle >= 2:
                    with _LORA_TRAIN_HISTORY_LOCK:
                        h = _LORA_TRAIN_HISTORY.setdefault(dataset, {
                            "started_at": started_at, "best": [], "val": [],
                        })
                        h["completed_at"] = time.time()
                        h["final_status"] = status
                        h["final_epoch"] = int(d.get("current_epoch", 0) or 0)
                        h["final_step"] = int(d.get("current_step", 0) or 0)
                        # NEW 2026-05-30: capture the FULL engine training_state at
                        # completion so the UI can re-render the run's progress
                        # block after an engine restart (engine state is in-memory
                        # only — lost on restart, but we want the UI to still
                        # show the completed run).
                        h["final_loss_history"] = list(d.get("loss_history", []) or [])
                        h["final_config"] = dict(d.get("config", {}) or {})
                        h["final_tensorboard_url"] = d.get("tensorboard_url")
                        h["final_tensorboard_logdir"] = d.get("tensorboard_logdir")
                    _lora_history_persist(dataset)
                    return
            else:
                consecutive_idle = 0
        except Exception as e:
            print(f"[lora] history poller warning ({dataset}): {e}")
        time.sleep(2)


@app.get("/api/lora/train/best_history")
def lora_train_best_history(dataset: str = "crucible_metal"):
    """Return per-dataset training history: best-ckpt updates + val curve.

    Sourced from the Mac's status poller (see _lora_train_history_poller). The
    engine itself doesn't expose plot_best_step / plot_val_loss — this is the
    only way to know what epoch the val-minimum landed at, which we need to
    tune train_epochs rationally on the next run."""
    with _LORA_TRAIN_HISTORY_LOCK:
        h = dict(_LORA_TRAIN_HISTORY.get(dataset, {}))
    if not h:
        h = _lora_history_load(dataset)
    return h


@app.post("/api/lora/evaluate")
def lora_evaluate(body: dict):
    """Plan 1 — post-training perceptual fitness scoring (see METAL_LORA_PLAN §11).

    Loops over caller-supplied checkpoints × scales, generates one take per
    (ckpt, scale) at fixed prompt + seed, scores each via the box analyze
    service's CLAP zero-shot tags. Persists results to
    library/lora_train_history/<dataset>_fitness.json.

    Body shape (all fields optional except dataset + ckpts):
      dataset: str — required, names the persisted curve
      ckpts:   [{label, lora_path, epoch?}] — required, list of adapter files
               to score. For now caller supplies these explicitly (e.g. best/,
               final/). Auto-enumeration of train/checkpoints/epoch_N/ is a v2 add.
      scales:  [0.3, 0.5] by default
      prompt + lyrics + seed + duration + bpm + keyscale + model — gen config
      target_tags + negative_tags — fitness vocabulary (defaults to power-metal)
      top_k — analyze top-K considered for tag presence (default 10)

    GPU-exclusive — caller is responsible for ensuring the engine is in
    inference-mode (LM + DiT loaded) before invoking. Engine restart per
    [[engine-fresh-boot-for-lora]] recommended before a long eval run.

    Synchronous — runs the full grid before returning. Per-iteration progress
    is persisted as we go so a long run isn't lost on crash.
    """
    _lora_require_engine()
    dataset = body.get("dataset") or "crucible_metal"
    ckpts = body.get("ckpts") or []
    if not ckpts:
        raise HTTPException(400, "ckpts list required: [{label, lora_path, epoch?}, ...]")
    scales = body.get("scales") or [0.3, 0.5]
    free_gpu("acestep")
    return lora_eval_mod.evaluate_dataset(
        mac_base=f"http://127.0.0.1:{int(CFG.get('server_port', 8000))}",
        engine_host=ACESTEP_HOST,
        analyze_host=ANALYZE_HOST,
        library_dir=LIBRARY,
        dataset=dataset,
        ckpts=ckpts,
        scales=[float(s) for s in scales],
        prompt=body.get("prompt") or body.get("tags") or "",
        lyrics=body.get("lyrics") or "",
        seed=int(body.get("seed", 42)),
        duration=int(body.get("duration", 40)),
        bpm=int(body.get("bpm", 132)),
        keyscale=body.get("keyscale") or "D minor",
        model=body.get("model") or "acestep-v15-xl-sft",
        target_tags=body.get("target_tags"),
        negative_tags=body.get("negative_tags"),
        top_k=int(body.get("top_k", 10)),
    )


@app.get("/api/lora/evaluate/results")
def lora_evaluate_results(dataset: str = "crucible_metal"):
    """Return the persisted fitness curve for a dataset, or {} if none exists."""
    path = os.path.join(LIBRARY, "lora_train_history", f"{dataset}_fitness.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


@app.get("/api/lora/evaluate/derive_targets")
def lora_evaluate_derive_targets(dataset: str = "crucible_metal",
                                  min_occurrences: int = 2,
                                  max_tags: int = 16):
    """Auto-build a STARTING target tag vocabulary from a dataset's seed captions.

    Per [[optional-additions]] and the experiment-tunability principle: target
    tags are experiment-specific. This endpoint inspects what the LoRA was
    trained against and returns the most common tag-like prefixes from the
    seed captions. Eyeball the result, drop wrong-genre tags, then pass the
    curated list to /api/lora/evaluate as target_tags.

    Example: a Nightwish dataset returns "symphonic metal, finnish metal,
    gothic metal, progressive metal" — but probably also some noise like
    "black metal" or "groove metal" because CLAP zero-shot mis-tagged a
    couple training samples. Drop those before evaluating.
    """
    _lora_require_engine()
    p = _lora_paths(dataset)
    try:
        tags = lora_eval_mod.derive_target_tags_from_dataset(
            ACESTEP_HOST, p["dataset_json"],
            min_occurrences=int(min_occurrences),
            max_tags=int(max_tags))
        return {"dataset": dataset, "suggested_target_tags": tags,
                "dataset_json": p["dataset_json"]}
    except Exception as e:
        raise HTTPException(500, f"derive failed: {e}")


def _metric_embedder(body: dict):
    """Resolve the embedder from body['embedder'] -> (embed_fn, metric_name). 'mert' = local
    MERT on the Mac (default, deterministic, no box); 'clap' = box CLAP via analyze /embed."""
    which = (body.get("embedder") or "mert").lower()
    if which == "clap":
        if not ANALYZE_HOST:
            raise HTTPException(400, "analyze_host not configured (box CLAP service)")
        return (lambda p: analyze_py.embed(ANALYZE_HOST, p, timeout=600)), "clap_centroid_cosine"
    if which == "mert":
        mw = body.get("max_windows")
        return (lambda p: embed_mert_mod.embed(p, max_windows=mw)), "mert_centroid_cosine"
    raise HTTPException(400, f"unknown embedder '{which}' (use 'mert' or 'clap')")


@app.post("/api/metric/validate")
def metric_validate_run(body: dict):
    """Validate a fitness metric against ground truth BEFORE trusting it (CLAP-centroid failed,
    METAL_LORA_PLAN §13d; MERT is the music-trained retry). embedder='mert' (default, Mac-local)
    or 'clap' (box). MERT = no box GPU; CLAP = serialize vs the engine.

    Body: {embedder?, artist_dir, bucket_dirs:{label:dir}, holdout_frac?, expected_order?,
           ear_pairs?:[{a,b,winner}], per_bucket_limit?, max_windows?}. artist_dir = the
           artist's own tracks (split into a centroid set + a held-out 'artist' bucket so
           similarity isn't trivially 1.0).
    Returns the validity report (ordering / test-retest / AUC / ear-agreement / verdict),
    saved to library/lora_train_history/metric_validation_<metric>.json."""
    artist_dir = (body.get("artist_dir") or "").strip()
    if not artist_dir:
        raise HTTPException(400, "artist_dir required")
    embed_fn, metric_name = _metric_embedder(body)
    try:
        return metric_val.run_validation(
            embed_fn=embed_fn, metric_name=metric_name, artist_dir=artist_dir,
            bucket_dirs=body.get("bucket_dirs") or {}, library_dir=LIBRARY,
            holdout_frac=float(body.get("holdout_frac", 0.4)),
            expected_order=body.get("expected_order"),
            ear_pairs=body.get("ear_pairs"),
            per_bucket_limit=body.get("per_bucket_limit"))
    except Exception as e:
        raise HTTPException(500, f"metric validation failed: {e}")


@app.post("/api/metric/centroid")
def metric_centroid_build(body: dict):
    """Build + save an artist 'sound centroid' (mean embedding) from a folder of the artist's
    tracks, for later centroid-distance scoring. embedder='mert' (default) or 'clap'."""
    artist_dir = (body.get("artist_dir") or "").strip()
    name = (body.get("name") or os.path.basename(artist_dir.rstrip("/")) or "centroid").strip()
    if not artist_dir:
        raise HTTPException(400, "artist_dir required")
    embed_fn, metric_name = _metric_embedder(body)
    try:
        cen = metric_val.build_centroid(embed_fn, metric_val.list_audio(artist_dir))
        out_dir = os.path.join(LIBRARY, "lora_train_history")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"centroid_{name}.json")
        with open(out_path, "w") as f:
            json.dump({"name": name, "metric": metric_name, "artist_dir": artist_dir, **cen}, f)
        return {"name": name, "metric": metric_name, "n": cen["n"], "dim": cen["dim"], "saved_to": out_path}
    except Exception as e:
        raise HTTPException(500, f"centroid build failed: {e}")


@app.post("/api/lora/train")
def lora_train(body: dict):
    """Start LoRA or LoKr training on the box from the preprocessed tensors. GPU-exclusive.

    Engine default-mismatch guardrails (LoKr method):
    - Engine ships `lokr_weight_decompose=True` (DoRA on) + `learning_rate=0.03`.
      These two are catastrophically incompatible: DoRA's dora_scale param needs
      lr~1e-4..1e-3, while 0.03 blows it up (verified 2026-05-29: 100-epoch run
      produced dora_scale max=19.18 → garbled output even at toggle-off, since
      set_multiplier(0) only zeros the delta, not the magnitude scaling).
    - We default this route to plain LoKr (`lokr_weight_decompose=False`) + a safer
      `learning_rate=0.01`. Callers can re-enable DoRA explicitly when paired with
      a low lr (e.g. 0.001) — that's the documented sweet spot for higher quality.

    `_ensure_training_ready` clears any residual CUDA tensors via /v1/reinitialize
    (it does NOT call /v1/init, which leaks ~4 GB orphans per call — see that fn)."""
    _lora_require_engine()
    dataset = body.get("dataset", "crucible_metal")
    p = _lora_paths(dataset)
    method = (body.get("method") or "lokr").lower()
    free_gpu("acestep")
    _ensure_training_ready(ACESTEP_HOST, dit_model=body.get("dit_model"))
    # LoKr-specific keys we plumb through; defaults applied below for safety.
    lokr_keys = ("lokr_linear_dim", "lokr_linear_alpha", "lokr_factor",
                 "lokr_decompose_both", "lokr_use_tucker", "lokr_use_scalar",
                 "lokr_weight_decompose")
    common = {k: body[k] for k in ("train_epochs", "train_batch_size", "gradient_accumulation",
                                   "save_every_n_epochs", "learning_rate", "training_seed",
                                   "gradient_checkpointing", "timestep_sampling_mode") + lokr_keys if k in body}
    if method == "lora":
        return ace_train.train_lora(ACESTEP_HOST, p["tensor_dir"], p["train_dir"],
                                    use_fp8=bool(body.get("use_fp8", False)),
                                    lora_rank=int(body.get("lora_rank", 64)),
                                    lora_alpha=int(body.get("lora_alpha", 128)),
                                    lora_dropout=float(body.get("lora_dropout", 0.1)), **common)
    # LoKr safety defaults — engine's defaults are broken-together (see docstring).
    common.setdefault("lokr_weight_decompose", False)
    common.setdefault("learning_rate", 0.01)
    # val_split: opt-in. Requires the 2026-05-29 engine patches (see
    # METAL_LORA_PLAN §7a). Default 0.1 when not specified — small enough to
    # train mostly on the corpus, large enough to give the best-checkpoint
    # tracking a signal on small (6-track) datasets.
    common["val_split"] = float(body["val_split"]) if "val_split" in body else 0.1
    # target_modules: which DiT module suffixes LyCORIS wraps. Opt-in; requires the
    # 2026-06-06 engine patch (StartLoKRTrainingRequest.target_modules + plumb into
    # LoKRConfig). Engine default is attention-only (q/k/v/o); adding the Qwen3MLP
    # names (gate/up/down_proj) = "attn+mlp" for richer style (METAL_LORA_PLAN §13b).
    # Unpatched engines silently ignore it (Pydantic drops unknown fields).
    if body.get("target_modules"):
        common["target_modules"] = list(body["target_modules"])
    # Crucible patch (2026-06-07): forward LoKr LyCORIS dropouts when provided
    # (engine patch StartLoKRTrainingRequest.lora_dropout/rank_dropout/module_dropout,
    # METAL_LORA_PLAN §13g Tier 2). Opt-in only; absent -> 0.0 server-side, so
    # behavior is unchanged for existing callers and unpatched engines.
    for _dk in ("lora_dropout", "rank_dropout", "module_dropout"):
        if body.get(_dk) is not None:
            common[_dk] = float(body[_dk])
    # Per-run output dir: never overwrite a prior adapter. Format encodes the
    # config so we can identify the run later by name alone.
    # Pattern: <dataset>/train_<YYYYMMDD-HHMMSS>__<adapter>_<epochs>ep_<sampling>
    # Example: crucible_nightwish/train_20260530-150524__lokr_150ep_continuous
    import datetime as _dt
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Encode the key varying knobs (lr + alpha) into the run-dir name so adapters are
    # self-describing + distinguishable in the picker (e.g. two discrete runs at
    # different lr no longer collide). Compact, folder-safe lr token: 0.001 -> "1e-3".
    try:
        _lrtok = format(float(common.get("learning_rate", 0.01)), ".0e").replace("e-0", "e-").replace("e+0", "e").replace("e+", "e")
    except Exception:
        _lrtok = str(common.get("learning_rate", "?"))
    suffix = (
        f"train_{ts}__{method}_{common.get('train_epochs','?')}ep_"
        f"{common.get('timestep_sampling_mode','discrete')}"
        f"_lr{_lrtok}_a{common.get('lokr_linear_alpha','?')}"
    )
    # train_dir is `<dataset_dir>/train` per the upload helper; replace last
    # path segment with our timestamped subdir. Uses forward slashes — the
    # engine on Windows accepts them.
    base_train = p["train_dir"]
    sep = "\\" if "\\" in base_train else "/"
    parts = base_train.rstrip(sep).split(sep)
    parts[-1] = suffix
    per_run_train_dir = sep.join(parts)
    # Reset history for this dataset's new run, then spawn background poller.
    # See _lora_train_history_poller for why this exists (engine doesn't expose
    # plot_best_step). Daemon thread → dies with the process; persisted JSON
    # at library/lora_train_history/<dataset>.json survives restarts.
    with _LORA_TRAIN_HISTORY_LOCK:
        _LORA_TRAIN_HISTORY[dataset] = {
            "started_at": time.time(), "best": [], "val": [],
            "output_dir": per_run_train_dir,
            "run_label": suffix,
            "config_at_start": dict(common),
        }
    res = ace_train.train_lokr(ACESTEP_HOST, p["tensor_dir"], per_run_train_dir, **common)
    threading.Thread(
        target=_lora_train_history_poller,
        args=(dataset, ACESTEP_HOST, time.time()),
        daemon=True,
    ).start()
    # Return the per-run dir so the caller knows where best/ and final/ landed
    if isinstance(res, dict):
        res["output_dir"] = per_run_train_dir
        res["run_label"] = suffix
    return res


@app.post("/api/lora/train_v2")
def lora_train_v2(body: dict):
    """Start LoKr training via the engine's training_v2 trainer (Crucible engine patch
    2026-06-08, route /v1/training/start_lokr_v2). Mirrors /api/lora/train (per-run output
    dir + history poller + xl-base reuse via /v1/reinitialize) but unlocks optimizer_type
    (incl. prodigy), scheduler_type, cfg_ratio, attention_type. Requires the 2026-06-08
    engine patch deployed; an unpatched engine returns 404 from start_lokr_v2.

    The DiT must be inited to xl-base BEFORE calling this (same as /api/lora/train):
    _ensure_training_ready only /v1/reinitialize's from last_init_params; it does not
    load a DiT [[lora-train-on-xl-base]]."""
    _lora_require_engine()
    dataset = body.get("dataset", "crucible_metal")
    p = _lora_paths(dataset)
    free_gpu("acestep")
    _ensure_training_ready(ACESTEP_HOST, dit_model=body.get("dit_model"))
    # Keys plumbed to train_lokr_v2 (only those present in body are forwarded).
    v2_keys = ("train_epochs", "train_batch_size", "gradient_accumulation",
               "save_every_n_epochs", "learning_rate", "training_seed",
               "gradient_checkpointing", "lokr_linear_dim", "lokr_linear_alpha",
               "lokr_factor", "lokr_decompose_both", "lokr_use_tucker", "lokr_use_scalar",
               "lokr_weight_decompose", "optimizer_type", "scheduler_type", "cfg_ratio",
               "attention_type", "dropout")
    common = {k: body[k] for k in v2_keys if k in body}
    if body.get("target_modules"):
        common["target_modules"] = list(body["target_modules"])
    # Per-run output dir (never overwrite a prior adapter); encode the v2-distinguishing
    # knobs (optimizer + lr) into the run name so it is self-describing in the picker.
    import datetime as _dt
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    try:
        _lrtok = format(float(common.get("learning_rate", 1e-4)), ".0e").replace("e-0", "e-").replace("e+0", "e").replace("e+", "e")
    except Exception:
        _lrtok = str(common.get("learning_rate", "?"))
    _opt = common.get("optimizer_type", "adamw")
    suffix = (
        f"train_{ts}__lokrv2_{common.get('train_epochs','?')}ep_"
        f"{_opt}_lr{_lrtok}_a{common.get('lokr_linear_alpha','?')}"
    )
    base_train = p["train_dir"]
    sep = "\\" if "\\" in base_train else "/"
    parts = base_train.rstrip(sep).split(sep)
    parts[-1] = suffix
    per_run_train_dir = sep.join(parts)
    with _LORA_TRAIN_HISTORY_LOCK:
        _LORA_TRAIN_HISTORY[dataset] = {
            "started_at": time.time(), "best": [], "val": [],
            "output_dir": per_run_train_dir,
            "run_label": suffix,
            "config_at_start": dict(common),
        }
    res = ace_train.train_lokr_v2(ACESTEP_HOST, p["tensor_dir"], per_run_train_dir, **common)
    threading.Thread(
        target=_lora_train_history_poller,
        args=(dataset, ACESTEP_HOST, time.time()),
        daemon=True,
    ).start()
    if isinstance(res, dict):
        res["output_dir"] = per_run_train_dir
        res["run_label"] = suffix
    return res


@app.get("/api/lora/adapters")
def lora_adapters_list(dataset: str = "crucible_metal"):
    """Enumerate archived adapters for a dataset.

    Without this, every training run overwrote the previous adapter (we lost
    the 2026-05-30 6-track Nightwish discrete one this way). Now each run
    writes to a unique `<dataset>/train_<timestamp>__<config>/` and this
    endpoint lists them so you can load any of them by run_label.

    For now we infer the list from on-Mac history (started_at + output_dir
    recorded by the per-run poller). Box-side enumeration via a directory
    listing would be a follow-up — for runs predating this commit, history
    points at the legacy `train/` dir which has been overwritten.
    """
    out = []
    # Pull from in-memory + persisted history
    with _LORA_TRAIN_HISTORY_LOCK:
        h = dict(_LORA_TRAIN_HISTORY.get(dataset, {}))
    if not h:
        h = _lora_history_load(dataset)
    if h and h.get("output_dir"):
        out.append({
            "run_label": h.get("run_label") or "current",
            "output_dir": h.get("output_dir"),
            "started_at": h.get("started_at"),
            "completed_at": h.get("completed_at"),
            "config": h.get("config_at_start") or {},
            "best_lora_path": (
                (h.get("output_dir") or "") +
                ("\\" if "\\" in (h.get("output_dir") or "") else "/")
                + "checkpoints"
                + ("\\" if "\\" in (h.get("output_dir") or "") else "/")
                + "best"
                + ("\\" if "\\" in (h.get("output_dir") or "") else "/")
                + "lokr_weights.safetensors"
            ),
            "final_lora_path": (
                (h.get("output_dir") or "") +
                ("\\" if "\\" in (h.get("output_dir") or "") else "/")
                + "final"
                + ("\\" if "\\" in (h.get("output_dir") or "") else "/")
                + "lokr_weights.safetensors"
            ),
        })
    # TODO v2: scan box-side via upload helper for additional run_TIMESTAMP
    # subfolders. Requires adding a directory-listing endpoint to the helper.
    return {"dataset": dataset, "adapters": out}


@app.get("/api/lora/adapters/all")
def lora_adapters_all():
    """Enumerate selectable adapters across ALL datasets, for the LoRA picker.

    Primary source is BOX-SIDE enumeration via the upload helper's /adapters
    route, which walks lora_data on disk and lists EVERY run (legacy `train/` +
    all per-run dirs, e.g. continuous vs discrete) that has a best/final
    checkpoint. Falls back to the Mac's per-run training history (latest run per
    dataset) when the helper is unreachable or predates the /adapters route.
    The picker also accepts a manual path for anything not enumerated."""
    # Prefer box-side enumeration: it sees every run that actually exists.
    if LORA_UPLOAD_HOST:
        try:
            res = lora_up.adapters(LORA_UPLOAD_HOST)
            ads = res.get("adapters") if isinstance(res, dict) else None
            if ads:
                return {"adapters": ads, "source": "box"}
        except Exception:
            pass  # helper down or old build -> history fallback below
    out = []
    histdir = os.path.join(LIBRARY, "lora_train_history")
    if os.path.isdir(histdir):
        for fn in sorted(os.listdir(histdir)):
            if not fn.endswith(".json"):
                continue
            ds = fn[:-5]
            h = _lora_history_load(ds) or {}
            od = h.get("output_dir")
            if not od:
                continue
            sep = "\\" if "\\" in od else "/"
            best = sep.join([od, "checkpoints", "best", "lokr_weights.safetensors"])
            final = sep.join([od, "final", "lokr_weights.safetensors"])
            out.append({
                "dataset": ds,
                "run_label": h.get("run_label") or ds,
                "best_path": best,
                "final_path": final,
                "completed_at": h.get("completed_at"),
            })
    return {"adapters": out, "source": "history"}


def _lora_training_status_with_fallback(dataset: str) -> Dict[str, Any]:
    """Engine training_state proxy + persisted-history fallback.

    Engine's /v1/training/status is in-memory only — on engine restart, the
    completed-run progress block disappears from the UI. We persist the full
    final snapshot to disk on completion (see _lora_train_history_poller). When
    the engine reports idle + empty AND we have a persisted snapshot for this
    dataset, we synthesize a status response from the saved data so the UI can
    re-render the run's progress block + sparkline + final config.

    Live training always wins over the cached snapshot — no risk of stale data
    shadowing an active run."""
    live = ace_train.training_status(ACESTEP_HOST) or {}
    # Some engines wrap responses in {"data": ...}, others return raw. Normalize.
    payload = live.get("data", live) if isinstance(live, dict) and "data" in live else live
    if not isinstance(payload, dict):
        payload = {}
    engine_is_training = bool(payload.get("is_training", False))
    engine_epoch = int(payload.get("current_epoch", 0) or 0)
    engine_has_state = engine_is_training or engine_epoch > 0 or bool(payload.get("loss_history"))
    if engine_has_state:
        return payload
    # Engine idle + empty → fall back to persisted history for this dataset
    snap = _lora_history_load(dataset) if dataset else {}
    if not snap or "final_epoch" not in snap:
        return payload  # genuine fresh state, nothing to restore
    epochs_total = int((snap.get("final_config") or {}).get("epochs") or snap.get("final_epoch") or 0)
    synthesized = {
        "is_training": False,
        "current_epoch": snap.get("final_epoch", 0),
        "current_step": snap.get("final_step", 0),
        "current_loss": None,
        "status": snap.get("final_status") or "(persisted from previous run)",
        "loss_history": snap.get("final_loss_history", []) or [],
        "config": snap.get("final_config", {}) or {},
        "tensorboard_url": snap.get("final_tensorboard_url"),
        "tensorboard_logdir": snap.get("final_tensorboard_logdir"),
        "start_time": snap.get("started_at"),
        "_persisted_snapshot": True,
        "_persisted_dataset": dataset,
    }
    if epochs_total and not synthesized["config"].get("epochs"):
        synthesized["config"]["epochs"] = epochs_total
    return synthesized


@app.get("/api/lora/train/status")
def lora_train_status(dataset: str = "crucible_metal"):
    """Engine training_state proxy + persisted-history fallback. See
    _lora_training_status_with_fallback for the architecture."""
    _lora_require_engine()
    return _lora_training_status_with_fallback(dataset)


@app.post("/api/lora/train/stop")
def lora_train_stop():
    _lora_require_engine()
    return ace_train.training_stop(ACESTEP_HOST)


@app.post("/api/lora/export")
def lora_export(body: dict):
    """Export the trained adapter to its .safetensors and (optionally) load it."""
    _lora_require_engine()
    dataset = body.get("dataset", "crucible_metal")
    p = _lora_paths(dataset)
    res = ace_train.training_export(ACESTEP_HOST, p["adapter_file"], p["train_dir"])
    if body.get("load"):
        try:
            ace_train.lora_load(ACESTEP_HOST, p["adapter_file"], adapter_name=dataset)
        except Exception as e:
            res = {"export": res, "load_error": str(e)}
    return res


# inference-time LoRA control (Phase 5 wires these into Generate's tuning UI)
@app.post("/api/lora/load")
def lora_load(body: dict):
    _lora_require_engine()
    dataset = body.get("dataset")
    path = body.get("lora_path") or (_lora_paths(dataset)["adapter_file"] if dataset else None)
    if not path:
        raise HTTPException(400, "lora_path or dataset required")
    return ace_train.lora_load(ACESTEP_HOST, path, adapter_name=body.get("adapter_name") or dataset)


@app.post("/api/lora/scale")
def lora_scale(body: dict):
    _lora_require_engine()
    return ace_train.lora_scale(ACESTEP_HOST, float(body.get("scale", 1.0)), adapter_name=body.get("adapter_name"))


@app.post("/api/lora/toggle")
def lora_toggle(body: dict):
    _lora_require_engine()
    return ace_train.lora_toggle(ACESTEP_HOST, bool(body.get("use_lora", True)))


@app.post("/api/lora/unload")
def lora_unload():
    _lora_require_engine()
    return ace_train.lora_unload(ACESTEP_HOST)


# ==================== Settings panel (curated app_config.json editor) ====================
# A whitelist + per-field metadata so the UI can render a self-documenting form for
# the values the user actually changes. Saving writes app_config.json *preserving*
# any non-whitelisted keys. Most changes need a ./run.sh restart to take effect.
SETTINGS_FIELDS = [
    # Box services (hosts) -------------------------------------------------------
    {"group": "Box services", "key": "comfy_host", "type": "host",
     "label": "ComfyUI", "hint": "default :8188 · ACE-Step generation via ComfyUI (fallback path)"},
    {"group": "Box services", "key": "acestep_host", "type": "host",
     "label": "ACE-Step engine", "hint": "default :8001 · Generate / Cover / LoRA training (the official engine)"},
    {"group": "Box services", "key": "analyze_host", "type": "host",
     "label": "Analyze service", "hint": "default :5075 · allin1 structure + CLAP tags; also seeds LoRA captions (§17a)"},
    {"group": "Box services", "key": "lora_upload_host", "type": "host",
     "label": "LoRA upload helper", "hint": "default :5080 · run LORA-UPLOAD_AUTO_INSTALL.bat on the box"},
    {"group": "Box services", "key": "roformer_host", "type": "host",
     "label": "BS-Roformer separator", "hint": "default :5070 · SOTA 6-stem separation"},
    {"group": "Box services", "key": "soulx_host", "type": "host",
     "label": "SoulX-Singer", "hint": "default :5060 · zero-shot vocal synthesis"},
    {"group": "Box services", "key": "rvc_python_host", "type": "host",
     "label": "RVC API server", "hint": "default :5050 · voice conversion"},
    # Engine feature flags ------------------------------------------------------
    {"group": "Engine flags", "key": "acestep_dcw_ok", "type": "bool",
     "label": "ACE DCW patched on box",
     "hint": "true when xl-base/sft DCW is patched OFF on the box; otherwise Generate falls back to ComfyUI (HANDOFF)"},
    {"group": "Engine flags", "key": "acestep_repaint", "type": "bool",
     "label": "Use engine for Repaint",
     "hint": "OFF (recommended) → ComfyUI · engine repaint is weak (silence-seed, no LM, no loudness match)"},
    {"group": "Engine flags", "key": "acestep_lego", "type": "bool",
     "label": "Use engine for Add-a-Layer",
     "hint": "OFF (recommended) → ComfyUI · engine lego garbles regions in current testing"},
    {"group": "Engine flags", "key": "still_engine", "type": "select:zimage,krea2",
     "label": "Still image engine",
     "hint": "Default model for photoreal stills · zimage = Z-Image Turbo (default) · krea2 = Krea 2 Ultra (needs the Krea2 models on the box). Takes effect immediately."},
    {"group": "Engine flags", "key": "still_krea2_enhancer", "type": "bool", "default": True,
     "label": "Krea2: prompt-adherence enhancer",
     "hint": "ON by default (matches the workflow). ComfyUI-Krea2T-Enhancer (model patch) — stronger prompt adherence + unfilter. Turn OFF only if the ComfyUI-Krea2T-Enhancer node isn't installed on the box."},
    {"group": "Engine flags", "key": "still_krea2_seed_variance", "type": "bool", "default": True,
     "label": "Krea2: seed variance",
     "hint": "ON by default (matches the workflow). RBG_Smart_Seed_Variance (conditioning noise) for image variety. Turn OFF only if the ComfyUI-RBG-SmartSeedVariance node isn't installed on the box."},
    # API keys ------------------------------------------------------------------
    {"group": "API keys", "key": "lastfm_key", "type": "secret",
     "label": "Last.fm API key",
     "hint": "Richer LoRA caption tags for known songs · free key at https://www.last.fm/api/account/create"},
    {"group": "API keys", "key": "acoustid_key", "type": "secret",
     "label": "AcoustID API key",
     "hint": "Identify untagged audio by fingerprint · needs `brew install chromaprint` for fpcalc · key at https://acoustid.org/api-key"},
    # Mac server ----------------------------------------------------------------
    {"group": "Mac server", "key": "server_host", "type": "host",
     "label": "Mac listen address",
     "hint": "127.0.0.1 = local only · 0.0.0.0 = reachable on the LAN at http://<this-mac-ip>:8000 (no auth — trusted networks only)"},
    {"group": "Mac server", "key": "rvc_driver", "type": "select:auto,rvc_python,gradio",
     "label": "RVC driver",
     "hint": "auto picks the clean API server, falls back to the legacy WebUI"},
]
SETTINGS_KEYS = {f["key"] for f in SETTINGS_FIELDS}
SETTINGS_TYPES = {f["key"]: f["type"] for f in SETTINGS_FIELDS}


@app.get("/api/settings")
def get_settings():
    """Return the curated, self-documenting field list with current values."""
    return {"fields": [{**f, "value": CFG.get(f["key"], f.get("default", ""))} for f in SETTINGS_FIELDS],
            "config_path": _CFG_PATH}


@app.put("/api/settings")
def put_settings(body: dict):
    """Write whitelisted keys to app_config.json, preserving any others.
    Most changes need a ./run.sh restart to take effect (module-level vars are
    read at startup); the response flags whether any changes actually landed."""
    if not isinstance(body, dict):
        raise HTTPException(400, "body must be an object")
    # load the current file (preserve unrelated keys)
    cur = {}
    if os.path.exists(_CFG_PATH):
        try:
            with open(_CFG_PATH) as f:
                cur = json.load(f)
        except Exception:
            cur = {}
    changes = []
    for k, v in body.items():
        if k not in SETTINGS_KEYS:
            continue
        t = SETTINGS_TYPES[k]
        if t == "bool":
            v = bool(v)
        elif t.startswith("select:"):
            choices = t.split(":", 1)[1].split(",")
            if str(v) not in choices:
                continue
            v = str(v)
        else:  # host / secret / string
            v = str(v).strip()
        if cur.get(k) != v:
            cur[k] = v
            changes.append(k)
    if changes:
        with open(_CFG_PATH, "w") as f:
            json.dump(cur, f, indent=2)
        CFG.update({k: cur[k] for k in changes})   # refresh in-memory dict so request-time CFG.get() reads see it
    # still_* keys are read per-request (CFG.get in /api/video/still), so they apply without a
    # restart; flag the rest (module-level vars captured at startup).
    restart_keys = [k for k in changes if not k.startswith("still_")]
    return {"ok": True, "changes": changes, "restart_required": bool(restart_keys)}


# static frontend at root (registered last so /api/* wins)
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    # server_host: "127.0.0.1" = this Mac only (default); "0.0.0.0" = reachable from
    # other machines on the LAN at http://<this-Mac-IP>:<port>. The API has no auth,
    # so only use 0.0.0.0 on a trusted network.
    uvicorn.run(app, host=CFG.get("server_host", "127.0.0.1"), port=CFG.get("server_port", 8000))
