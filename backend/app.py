"""MusicGen app backend — drives ComfyUI (ACE-Step 1.5) for a thin web UI.

Run:  python -m backend.app   (from the repo root)
"""
import json
import os
import random
import re
import shutil
import sqlite3
import threading
import time
import uuid

import requests
import websocket  # websocket-client
from typing import Any, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import comfy
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
from . import acestep_train as ace_train
from . import lora_upload_py as lora_up
from . import lora_dataset as lora_ds
from . import lora_eval as lora_eval_mod
from . import voicegen as voicegen_mod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WEB_DIST = os.path.join(ROOT, "web", "dist")
FRONTEND = _WEB_DIST if os.path.isdir(_WEB_DIST) else os.path.join(ROOT, "frontend")
LIBRARY = os.path.join(ROOT, "library")
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
            "acestep_repaint": ACESTEP_REPAINT,
            "acestep_lego": ACESTEP_LEGO,
            "analyze": analyze_mod.available() or bool(ANALYZE_HOST),
            "analyze_box": bool(ANALYZE_HOST),
            "lora_train": bool(ACESTEP_HOST),       # train/use LoRAs on the official engine
            "lora_upload": bool(LORA_UPLOAD_HOST),  # box dataset upload helper present
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

    if ACESTEP_HOST and not p.get("force_comfy"):     # ----- official engine path -----
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


async def _source_bytes(file, job_id):
    """Raw bytes + label for an edit source (upload or library job_id) — for engine
    tasks that take a multipart src_audio (cover/repaint), vs the ComfyUI upload path."""
    if file is not None:
        return await file.read(), (file.filename or "src.wav")
    if job_id:
        src = next((os.path.join(LIBRARY, job_id + e) for e in (".wav", ".mp3")
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
                       ref_name: str = Form(None)):         # gold: which curated reference
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
            if not master_mod.available():
                raise HTTPException(500, "matchering not installed on the Mac (pip install matchering)")
            ref, _ = _stash_input(work, "reference",
                                  await ref_file.read() if ref_file else None,
                                  ref_file.filename if ref_file else None, ref_job_id)
            master_mod.master(tgt, ref, out, bit_depth=int(bit_depth))
            params = {"source": tgt_label, "mode": "reference", "reference": ref_job_id or "upload"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"mastering failed: {e}")
    shutil.rmtree(work, ignore_errors=True)
    params["title"] = _source_master_title(job_id, tgt_label)   # name it after the song
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
    for ext in (".mp3", ".wav"):
        p = os.path.join(LIBRARY, f"{pid}{ext}")
        if os.path.exists(p):
            return p
    with db() as conn:
        row = conn.execute("SELECT audio FROM jobs WHERE id=?", (pid,)).fetchone()
    if row and row["audio"] and os.path.exists(row["audio"]):
        return row["audio"]
    return None


@app.get("/api/export/{pid}")
def export_audio(pid: str, fmt: str = "mp3"):
    """Download a library track as MP3 (320k). Already-MP3 files pass through; WAVs are
    transcoded with ffmpeg. Filename uses the item's note/label when available."""
    src = _lib_source_path(pid)
    if not src:
        raise HTTPException(404, "no audio")
    # friendly download name: the user-given song title wins (else note/tags/source),
    # with a "-v2" suffix when the track is one of several versions of that name.
    name = pid
    try:
        with db() as conn:
            row = conn.execute("SELECT params,mode,created FROM jobs WHERE id=?", (pid,)).fetchone()
        if row and row["params"]:
            pp = json.loads(row["params"])
            title = str(pp.get("title") or "").strip()
            raw = title or pp.get("note") or pp.get("tags") or pp.get("source") or pid
            slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(raw)).strip("_")[:48] or pid
            if title:  # version among same mode + same title (oldest=v1)
                with db() as conn:
                    sibs = conn.execute(
                        "SELECT id,params FROM jobs WHERE mode=? AND status='done' ORDER BY created ASC",
                        (row["mode"],)).fetchall()
                vers = [s["id"] for s in sibs
                        if str((json.loads(s["params"]) if s["params"] else {}).get("title") or "").strip().lower() == title.lower()]
                if len(vers) > 1 and pid in vers:
                    slug = f"{slug}-v{vers.index(pid) + 1}"
            name = slug
    except Exception:
        pass
    fname = f"{name}.mp3"
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
# - xl-sft is the project default per [[base-sft-default]]; we train against it so
#   the LoRA matches what Generate uses.
# - 4B LM produces the rich prose captions the merge layer is designed around;
#   0.6B (the engine's default when lm_model_path is omitted) gives weaker tags.
LORA_DIT_MODEL = "acestep-v15-xl-sft"
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
    # Per-run output dir: never overwrite a prior adapter. Format encodes the
    # config so we can identify the run later by name alone.
    # Pattern: <dataset>/train_<YYYYMMDD-HHMMSS>__<adapter>_<epochs>ep_<sampling>
    # Example: crucible_nightwish/train_20260530-150524__lokr_150ep_continuous
    import datetime as _dt
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = (
        f"train_{ts}__{method}_{common.get('train_epochs','?')}ep_"
        f"{common.get('timestep_sampling_mode','discrete')}"
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
    return {"fields": [{**f, "value": CFG.get(f["key"], "")} for f in SETTINGS_FIELDS],
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
    return {"ok": True, "changes": changes, "restart_required": bool(changes)}


# static frontend at root (registered last so /api/* wins)
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    # server_host: "127.0.0.1" = this Mac only (default); "0.0.0.0" = reachable from
    # other machines on the LAN at http://<this-Mac-IP>:<port>. The API has no auth,
    # so only use 0.0.0.0 on a trusted network.
    uvicorn.run(app, host=CFG.get("server_host", "127.0.0.1"), port=CFG.get("server_port", 8000))
