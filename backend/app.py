"""MusicGen app backend — drives ComfyUI (ACE-Step 1.5) for a thin web UI.

Run:  python -m backend.app   (from the repo root)
"""
import json
import os
import random
import shutil
import sqlite3
import threading
import time
import uuid

import requests
import websocket  # websocket-client
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import comfy
from . import rvc as rvc_mod
from . import rvc_py
from . import roformer_py
from . import acestep_py
from . import asr as asr_mod
from . import voices as voices_mod
from . import stems as stems_mod
from . import mix as mix_mod
from . import postfx as postfx_mod
from . import master as master_mod
from . import guitar as guitar_mod
from . import sections as sections_mod
from . import genres as genres_mod
from . import llm as llm_mod
from . import lyrics as lyrics_mod
from . import melody as melody_mod
from . import voicegen as voicegen_mod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WEB_DIST = os.path.join(ROOT, "web", "dist")
FRONTEND = _WEB_DIST if os.path.isdir(_WEB_DIST) else os.path.join(ROOT, "frontend")
LIBRARY = os.path.join(ROOT, "library")
STEMS_DIR = os.path.join(LIBRARY, "stems")
DB = os.path.join(LIBRARY, "library.db")
os.makedirs(LIBRARY, exist_ok=True)
os.makedirs(STEMS_DIR, exist_ok=True)

_CFG_PATH = os.path.join(ROOT, "app_config.json")
if not os.path.exists(_CFG_PATH):  # fresh clone: fall back to the committed example
    _CFG_PATH = os.path.join(ROOT, "app_config.example.json")
CFG = json.load(open(_CFG_PATH))
HOST = CFG["comfy_host"]
CLIENT_ID = "musicgen-app"
C = comfy.Comfy(HOST)
ROFORMER_HOST = CFG.get("roformer_host", "")
ACESTEP_HOST = CFG.get("acestep_host", "")   # official ACE-Step engine (cover etc.); empty = use ComfyUI
# DCW (Differential Correction in Wavelet domain) ships ON-by-default in the engine and
# garbles XL text2music (full from-noise trajectory); it CANNOT be disabled over the HTTP
# API (see HANDOFF). Until DCW is patched off on the box, gate engine text2music on the XL
# models and fall back to ComfyUI so Generate never silently emits garbage. Turbo (short
# trajectory) and cover/repaint (source-anchored) are unaffected. Flip to true once the box
# is patched + verified.
ACESTEP_DCW_OK = bool(CFG.get("acestep_dcw_ok", False))


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


def save_job(pid):
    j = JOBS.get(pid)
    if not j:
        return
    with db() as conn:
        conn.execute(
            "REPLACE INTO jobs(id,created,mode,params,audio,status,error) VALUES(?,?,?,?,?,?,?)",
            (pid, j["created"], j["mode"], json.dumps(j["params"]),
             j.get("audio_file"), j["status"], j.get("error")))


def save_done_row(jid, mode, params, audio_path, bucket=""):
    with db() as conn:
        conn.execute(
            "REPLACE INTO jobs(id,created,mode,params,audio,status,error,bucket) VALUES(?,?,?,?,?,?,?,?)",
            (jid, time.time(), mode, json.dumps(params), audio_path, "done", None, bucket))


# ---------------- WebSocket progress listener ----------------
def on_complete(pid):
    """Fetch history, download the produced audio into the library."""
    try:
        h = C.history(pid)
        if pid not in h:
            return
        for _, out in h[pid].get("outputs", {}).items():
            for a in out.get("audio", []):
                data = C.view_bytes(a["filename"], a.get("subfolder", ""), a.get("type", "output"))
                path = os.path.join(LIBRARY, f"{pid}.mp3")
                with open(path, "wb") as f:
                    f.write(data)
                try:                                  # auto-fix ACE end-burst/clipping (only when needed)
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
            while True:
                msg = ws.recv()
                if isinstance(msg, (bytes, bytearray)):
                    continue  # binary preview frames
                handle_msg(json.loads(msg))
        except Exception:
            time.sleep(2)


# ---------------- API ----------------
app = FastAPI(title="MusicGen")


@app.on_event("startup")
def startup():
    init_db()
    threading.Thread(target=ws_loop, daemon=True).start()


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
            "genres": genres}


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
            "guidance_scale": float(p.get("cfg") if p.get("cfg") not in (None, "") else 6.0),
            "inference_steps": int(p.get("steps") or 32),
            "shift": float(p.get("shift", 3.0)),
            "infer_method": p.get("infer_method", "ode"),
            "cfg_interval_start": float(p.get("cfg_interval_start", 0.0)),
            "cfg_interval_end": float(p.get("cfg_interval_end", 0.95)),
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
        try:
            task_id = acestep_py.submit(ACESTEP_HOST, fields)
        except Exception as e:
            raise HTTPException(500, f"acestep submit failed: {e}")
        pid = uuid.uuid4().hex
        resolved = {"engine": "acestep", "tags": p.get("tags", ""), "seed": seed,
                    "model": fields["model"], "guidance_scale": fields["guidance_scale"],
                    "steps": fields["inference_steps"], "duration": fields["duration"]}
        with LOCK:
            JOBS[pid] = _new_job(resolved, "generate")
            JOBS[pid]["status"] = "running"
        save_job(pid)
        threading.Thread(target=_acestep_poll, args=(pid, task_id, "generate"), daemon=True).start()
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
        JOBS[pid] = _new_job(resolved, "generate")
    save_job(pid)
    return {"job_id": pid, "seed": resolved["seed"]}


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


def _acestep_poll(pid, task_id, mode):
    """Background: poll the official ACE-Step engine until the task finishes, then
    download EVERY batch take — the first becomes the tracked job (pid), the rest are
    saved as their own library rows ('take 2', 'take 3'…) under the same `mode` so they
    can be compared. Mirrors the ComfyUI job UX so the frontend's pollJob works unchanged."""
    try:
        files = acestep_py.wait(ACESTEP_HOST, task_id)     # list, one ref per take
        out = _save_engine_audio(pid, files[0])
        with LOCK:
            base = dict(JOBS[pid]["params"])
            JOBS[pid]["params"]["take"] = 1
            JOBS[pid]["audio_file"] = out
            JOBS[pid]["status"] = "done"
        save_job(pid)
        for i, fr in enumerate(files[1:], start=2):        # extra takes → their own library items
            try:
                jid = uuid.uuid4().hex
                o2 = _save_engine_audio(jid, fr)
                p2 = dict(base); p2["take"] = i
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

    if ACESTEP_HOST:                                  # ----- official engine path -----
        if file is not None:
            data = await file.read()
            label = file.filename or "upload"
        elif job_id:
            src = next((os.path.join(LIBRARY, job_id + e) for e in (".wav", ".mp3")
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
        # Field names/model id follow the docs (RESEARCH §13) — verify vs /v1/models.
        fields = {
            "task_type": "cover",
            "prompt": p.get("tags", ""),
            "lyrics": "" if p.get("instrumental") else p.get("lyrics", ""),
            "audio_cover_strength": float(p.get("cover_strength", 0.5)),
            "guidance_scale": float(p.get("guidance_scale", 7.0)),
            "inference_steps": int(p.get("steps", 32)),
            "shift": float(p.get("shift", 3.0)),
            "cfg_interval_start": float(p.get("cfg_interval_start", 0.0)),
            "cfg_interval_end": float(p.get("cfg_interval_end", 0.95)),
            "infer_method": p.get("infer_method", "ode"),
            "audio_format": "wav",
            "model": p.get("model", "acestep-v15-xl-base"),
        }
        if p.get("seed"):
            fields["seed"] = int(p["seed"])
            fields["use_random_seed"] = False         # else the engine ignores `seed` and rolls its own
        try:
            free_gpu("ace")                           # free ComfyUI + RVC before the (offloaded) ACE engine generates
        except Exception:
            pass
        try:
            task_id = acestep_py.submit(ACESTEP_HOST, fields, src_audio=(data, label), ctx_audio=ctx)
        except Exception as e:
            raise HTTPException(500, f"acestep submit failed: {e}")
        pid = uuid.uuid4().hex
        resolved = {"engine": "acestep", "source": label, "tags": p.get("tags", ""),
                    "cover_strength": fields["audio_cover_strength"], "model": fields["model"],
                    "guidance_scale": fields["guidance_scale"], "steps": fields["inference_steps"]}
        with LOCK:
            JOBS[pid] = _new_job(resolved, "cover")
            JOBS[pid]["status"] = "running"
        save_job(pid)
        threading.Thread(target=_acestep_poll, args=(pid, task_id, "cover"), daemon=True).start()
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
        src = next((os.path.join(LIBRARY, job_id + e) for e in (".wav", ".mp3")
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
            src = next((os.path.join(LIBRARY, job_id + e) for e in (".wav", ".mp3")
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
            s = next((os.path.join(LIBRARY, job_id + e) for e in (".wav", ".mp3")
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
        src = next((os.path.join(LIBRARY, job_id + e) for e in (".wav", ".mp3")
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
            return {"id": pid, "status": j["status"], "progress": j.get("progress", 0),
                    "max": j.get("max", 0), "error": j.get("error"),
                    "audio_url": f"/api/audio/{pid}" if j.get("audio_file") else None}
    # fall back to persisted library row (e.g. after a restart)
    with db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (pid,)).fetchone()
    if not row:
        raise HTTPException(404, "unknown job")
    return {"id": pid, "status": row["status"], "error": row["error"],
            "audio_url": f"/api/audio/{pid}" if row["audio"] else None}


@app.post("/api/cancel")
def cancel():
    C.interrupt()
    return {"ok": True}


@app.get("/api/library")
def library():
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE status='done' ORDER BY created DESC LIMIT 500").fetchall()
    return [{"id": r["id"], "created": r["created"], "mode": r["mode"],
             "params": json.loads(r["params"]), "audio_url": f"/api/audio/{r['id']}",
             "bucket": (r["bucket"] if "bucket" in r.keys() else "") or ""}
            for r in rows]


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
    for ext in (".mp3", ".wav"):
        p = os.path.join(LIBRARY, jid + ext)
        if os.path.exists(p):
            os.remove(p)
            removed.append(ext)
    with db() as conn:
        conn.execute("DELETE FROM jobs WHERE id=?", (jid,))
    with LOCK:
        JOBS.pop(jid, None)
    return {"ok": True, "removed": removed}


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
        src = next((os.path.join(LIBRARY, job_id + e) for e in (".mp3", ".wav")
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
        src = next((os.path.join(LIBRARY, job_id + e) for e in (".mp3", ".wav")
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
    save_done_row(jid, "tone", {"source": src_label, "preset": preset}, os.path.join(LIBRARY, f"{jid}.wav"))
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
        src = next((os.path.join(LIBRARY, job_id + e) for e in (".mp3", ".wav")
                    if os.path.exists(os.path.join(LIBRARY, job_id + e))), None)
        if not src:
            raise HTTPException(404, f"library track not found: {job_id}")
        p = os.path.join(work, label + os.path.splitext(src)[1])
        shutil.copy(src, p)
        return p, job_id
    raise HTTPException(400, f"provide a file or job_id for {label}")


@app.post("/api/master/apply")
async def master_apply(job_id: str = Form(None),
                       ref_job_id: str = Form(None),
                       file: UploadFile = File(None),
                       ref_file: UploadFile = File(None),
                       bit_depth: int = Form(16)):
    """Reference-master a target track toward a reference master (Matchering, Mac)."""
    if not master_mod.available():
        raise HTTPException(500, "matchering not installed on the Mac (pip install matchering)")
    work = os.path.join(STEMS_DIR, uuid.uuid4().hex)
    os.makedirs(work, exist_ok=True)
    tgt, tgt_label = _stash_input(work, "target",
                                  await file.read() if file else None,
                                  file.filename if file else None, job_id)
    ref, _ = _stash_input(work, "reference",
                          await ref_file.read() if ref_file else None,
                          ref_file.filename if ref_file else None, ref_job_id)
    jid = uuid.uuid4().hex
    out = os.path.join(LIBRARY, f"{jid}.wav")
    try:
        master_mod.master(tgt, ref, out, bit_depth=int(bit_depth))
    except Exception as e:
        raise HTTPException(500, f"mastering failed: {e}")
    shutil.rmtree(work, ignore_errors=True)
    save_done_row(jid, "master", {"source": tgt_label, "reference": ref_job_id or "upload"}, out)
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
        backing_path = next((os.path.join(LIBRARY, backing_job_id + e) for e in (".wav", ".mp3")
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
    if ext in (".mp3", ".wav"):
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
        raw = next((os.path.join(LIBRARY, import_id + e) for e in (".mp3", ".wav")
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
    return {"ollama": llm_mod.ollama_models(), "claude": bool(os.environ.get("ANTHROPIC_API_KEY"))}


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
                  os.path.join(LIBRARY, f"{jid}.wav"))
    return {"job_id": jid, "audio_url": f"/api/audio/{jid}"}


@app.post("/api/stitch")
def stitch_tracks(body: dict):
    """Concatenate per-block clips into one song with a crossfade (Song Constructor)."""
    tracks = body.get("tracks", [])
    if not tracks:
        raise HTTPException(400, "no tracks to stitch")
    try:
        wav = mix_mod.stitch(tracks, LIBRARY, STEMS_DIR,
                             crossfade_s=float(body.get("crossfade_s", 1.0)))
    except Exception as e:
        raise HTTPException(500, f"stitch failed: {e}")
    jid = uuid.uuid4().hex
    with open(os.path.join(LIBRARY, f"{jid}.wav"), "wb") as f:
        f.write(wav)
    save_done_row(jid, "song", {"tags": body.get("tags", ""),
                                "sections": body.get("sections", ""),
                                "crossfade_s": body.get("crossfade_s", 1.0),
                                "blocks": len(tracks)},
                  os.path.join(LIBRARY, f"{jid}.wav"))
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
        src = next((os.path.join(LIBRARY, job_id + e) for e in (".wav", ".mp3")
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
        src = next((os.path.join(LIBRARY, job_id + e) for e in (".mp3", ".wav")
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
    for ext, mt in ((".mp3", "audio/mpeg"), (".wav", "audio/wav")):
        path = os.path.join(LIBRARY, f"{pid}{ext}")
        if os.path.exists(path):
            return FileResponse(path, media_type=mt)
    # fall back to the stored path (e.g. stems registered from STEMS_DIR)
    with db() as conn:
        row = conn.execute("SELECT audio FROM jobs WHERE id=?", (pid,)).fetchone()
    if row and row["audio"] and os.path.exists(row["audio"]):
        mt = "audio/wav" if row["audio"].endswith(".wav") else "audio/mpeg"
        return FileResponse(row["audio"], media_type=mt)
    raise HTTPException(404, "no audio")


# static frontend at root (registered last so /api/* wins)
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    # server_host: "127.0.0.1" = this Mac only (default); "0.0.0.0" = reachable from
    # other machines on the LAN at http://<this-Mac-IP>:<port>. The API has no auth,
    # so only use 0.0.0.0 on a trusted network.
    uvicorn.run(app, host=CFG.get("server_host", "127.0.0.1"), port=CFG.get("server_port", 8000))
