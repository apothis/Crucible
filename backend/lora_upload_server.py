"""Box-side LoRA dataset upload helper (runs ON the Windows GPU box, tiny, no GPU).

The ACE-Step engine's training pipeline reads dataset/tensor/output folders from
BOX-side paths, but Crucible + its library live on the Mac. This service is the bridge:
the Mac uploads the enriched dataset (audio + {name}.lyrics.txt + {name}.json) and this
writes it into a folder the engine can `/v1/dataset/scan`. It also hands back the
sibling tensor/output paths so the Mac can drive preprocess/train/export.

Self-contained: everything lives under MG_LORA_DIR (default ./lora_data beside this
script) -> `{base}/{dataset}/data` (audio+labels), `{base}/{dataset}/tensors`,
`{base}/{dataset}/adapter`. Delete the folder = clean uninstall.

Dataset endpoints (all JSON unless noted):
  GET  /health                         -> {ok, base_dir, fs_roots}
  POST /dataset/new      {name}        -> {data_dir, tensor_dir, adapter_dir} (created)
  POST /dataset/upload   multipart     -> writes files into {dataset}/data; {written, data_dir}
  POST /dataset/clear    {name}        -> empties {dataset}/data
  POST /dataset/delete   {name}        -> removes the whole {dataset} folder
  GET  /adapters                       -> every trained adapter on disk (best/final paths)
  GET  /dataset/paths?name=            -> {data_dir, tensor_dir, adapter_dir, exists}
  GET  /dataset/list?name=             -> {files: [...]}

Filesystem API (sandboxed to MG_FS_ROOTS; lets the Mac/agent read+enumerate+patch box
files without manual round-trips - e.g. deploy engine patches, read engine logs, check
tensor timestamps, find checkpoint folders, move adapters):
  GET  /fs/roots                       -> {roots: [...]} (the allowed sandbox roots)
  GET  /fs/list?path=&glob=            -> {entries:[{name,path,is_dir,size,mtime_iso}]}
  GET  /fs/stat?path=                  -> {exists,is_dir,size,mtime_iso}
  GET  /fs/read?path=&max_bytes=&b64=  -> {content|b64, size, truncated}
  GET  /fs/tail?path=&lines=           -> {content} (last N lines; for logs)
  GET  /fs/find?root=&pattern=&max=    -> {matches:[...]} (recursive glob; checkpoint discovery)
  GET  /fs/du?path=                    -> {bytes, files, human} (recursive size; find cleanup targets)
  POST /fs/write  {path, content|b64, backup?}  -> {bytes, mtime_iso, backup?}  (creates parent dirs)
  POST /fs/mkdir  {path}               -> {ok, path}
  POST /fs/move   {src, dst}           -> {ok}
  POST /fs/copy   {src, dst}           -> {ok}
  POST /fs/delete {path, confirm:true} -> {ok, removed}
  POST /fs/pycompile {path}            -> {ok} | {ok:false, error}  (py_compile on the box)

SAFETY: every fs path is realpath-resolved and must sit inside one of the allowed roots
(default: BASE + the auto-detected engine install dir [contains acestep/ AND .venv/, so
`.venv/.../lycoris` patches work] + its parent for the HF `.cache` + any auto-detected
sibling ComfyUI install for the video pipeline). MG_FS_ROOTS full-overrides that set;
MG_FS_EXTRA_ROOTS adds to it. Paths outside -> 403.
If MG_FS_TOKEN is set, mutating ops (write/mkdir/move/copy/delete) require a matching
`token` in the body. /fs/delete additionally requires `confirm:true`.
Deliberately NO arbitrary command/exec endpoint in this cut - ask if you want one (opt-in).

AI Toolkit (YuE2 LoRA training, docs/YUE2_TRAINING_PLAN.md) - fixed actions only, no free exec:
  GET  /aitk/status                    -> {dir, installed, kind, job, pid, alive, log, log_tail}
  POST /aitk/install {token?}          -> runs install_ai_toolkit.bat (beside this script) detached
  POST /aitk/run  {job, token?, free_gpu?=true} -> ai-toolkit/config/<job>.yaml via its own venv
  POST /aitk/stop {token?}             -> kills ONLY the process tree this helper started
  GET  /aitk/log?lines=                -> tail of the current/last job's log
  POST /helper/restart {token?}        -> relaunch this helper from run_lora_upload.bat (picks up
                                          fs/write-deployed updates with nobody at the box)
The AI Toolkit folder (MG_AITK_DIR, default <MusicGen>/ai-toolkit) is always an fs root.
"""
import os
import re
import base64
import datetime
import fnmatch
import json
import py_compile
import shutil
import subprocess
import time
import urllib.request

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse

PORT = int(os.environ.get("MG_LORA_PORT", "5080"))
BASE = os.environ.get("MG_LORA_DIR",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "lora_data"))
os.makedirs(BASE, exist_ok=True)

# Optional shared token gating the MUTATING fs ops (write/mkdir/move/copy/delete).
# Unset -> no auth (matches the existing dataset endpoints' trusted-LAN model).
FS_TOKEN = os.environ.get("MG_FS_TOKEN", "").strip()

# AI Toolkit lives next to AceStep / Comfyui / CrucibleLorA under the MusicGen folder
HELPER_DIR = os.path.dirname(os.path.abspath(__file__))
AITK_DIR = os.environ.get("MG_AITK_DIR", "").strip() or os.path.join(os.path.dirname(HELPER_DIR), "ai-toolkit")

app = FastAPI()


# ----------------------------- fs sandbox helpers -----------------------------
def _engine_root():
    """Walk UP from BASE to find the engine install dir = the ancestor that contains
    `acestep/` (engine source) or `.venv/` (so .venv/Lib/site-packages/lycoris patches are
    reachable). Robust regardless of how deep MG_LORA_DIR is nested under the engine."""
    d = os.path.realpath(BASE)
    for _ in range(8):
        if os.path.isdir(os.path.join(d, "acestep")) or os.path.isdir(os.path.join(d, ".venv")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def _comfy_roots():
    """Auto-detect a ComfyUI install sitting next to (or one level above) the engine, so
    the fs API can place custom_nodes + model files for the video pipeline. Matches dir
    names like `Comfyui`, `ComfyUI`, `ComfyUI_windows_portable` (case/separator-insensitive)."""
    er = _engine_root()
    bases = [os.path.dirname(er), os.path.dirname(os.path.dirname(er))] if er else [os.path.dirname(BASE)]
    found = []
    for b in bases:
        try:
            for name in os.listdir(b):
                norm = name.lower().replace("_", "").replace("-", "").replace(" ", "")
                p = os.path.join(b, name)
                if norm.startswith("comfyui") and os.path.isdir(p):
                    found.append(p)
        except OSError:
            pass
    return found


def _fs_roots():
    """Allowed roots for the filesystem API.
    - MG_FS_ROOTS (os.pathsep-separated) FULL-OVERRIDES the auto-detected default set.
    - MG_FS_EXTRA_ROOTS (os.pathsep-separated) is ADDED to whatever set is in effect.
    Default = BASE + the detected engine install dir (covers acestep source AND
    `.venv/.../lycoris` for site-packages patches) + that dir's parent (the HF `.cache`
    modeling files) + any auto-detected sibling ComfyUI install (custom_nodes + models for
    the video pipeline). Everything else on disk stays 403."""
    env = os.environ.get("MG_FS_ROOTS", "").strip()
    if env:
        cands = [p for p in env.split(os.pathsep) if p.strip()]
    else:
        cands = [BASE]
        er = _engine_root()
        if er:
            cands += [er, os.path.dirname(er)]          # engine install (acestep + .venv) + parent (.cache)
        else:
            cands += [os.path.dirname(BASE), os.path.dirname(os.path.dirname(BASE))]  # fallback
        cands += _comfy_roots()                          # sibling ComfyUI (video pipeline deploys)
    extra = os.environ.get("MG_FS_EXTRA_ROOTS", "").strip()
    if extra:                                            # additive, honored even under a full override
        cands += [p for p in extra.split(os.pathsep) if p.strip()]
    cands.append(AITK_DIR)                               # AI Toolkit: job files, datasets, patches, logs
    roots = []
    for p in cands:
        try:
            rp = os.path.normcase(os.path.realpath(p))
            if rp and os.path.isdir(rp) and rp not in roots:
                roots.append(rp)
        except OSError:
            pass
    return roots


def _human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def _resolve(path: str) -> str:
    """Realpath-resolve `path` and confirm it sits inside an allowed root, else 403.
    Works for not-yet-existing paths (fs/write to a new file)."""
    if not path:
        raise HTTPException(400, "path required")
    real = os.path.realpath(path)
    nc = os.path.normcase(real)
    for root in _fs_roots():
        if nc == root or nc.startswith(root + os.sep):
            return real
    raise HTTPException(403, f"path outside allowed roots: {real}")


def _auth(body: dict):
    if FS_TOKEN and (body or {}).get("token") != FS_TOKEN:
        raise HTTPException(401, "bad or missing token")


def _iso(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def _entry(full: str) -> dict:
    st = os.stat(full)
    return {"name": os.path.basename(full), "path": full, "is_dir": os.path.isdir(full),
            "size": st.st_size, "mtime": st.st_mtime, "mtime_iso": _iso(st.st_mtime)}


def _safe(name: str) -> str:
    """Sanitize a dataset name into a single safe folder segment (no traversal)."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip()).strip("._-")
    return s[:64] or "dataset"


def _paths(name: str):
    """All box-side paths the ACE engine's pipeline needs, computed here (correct
    Windows separators) so the Mac passes them through verbatim."""
    root = os.path.join(BASE, _safe(name))
    return {
        "dataset_dir": root,
        "data_dir": os.path.join(root, "data"),            # audio + .lyrics.txt + .json (scan)
        "dataset_json": os.path.join(root, "dataset.json"),  # saved dataset (save)
        "tensor_dir": os.path.join(root, "tensors"),        # preprocess output / train input
        "train_dir": os.path.join(root, "train"),           # training run output (checkpoints/logs)
        "adapter_dir": os.path.join(root, "adapter"),
        "adapter_file": os.path.join(root, "adapter", "adapter_model.safetensors"),  # export target
    }


@app.get("/health")
def health():
    return {"ok": True, "base_dir": BASE, "fs_roots": _fs_roots(), "fs_token_required": bool(FS_TOKEN)}


@app.post("/dataset/new")
def dataset_new(name: str = Form(...)):
    p = _paths(name)
    for key in ("dataset_dir", "data_dir", "tensor_dir", "train_dir", "adapter_dir"):
        os.makedirs(p[key], exist_ok=True)
    return p


@app.post("/dataset/upload")
async def dataset_upload(name: str = Form(...), files: list[UploadFile] = File(...)):
    p = _paths(name)
    os.makedirs(p["data_dir"], exist_ok=True)
    written = []
    for f in files:
        fn = os.path.basename(f.filename or "").strip()
        if not fn:
            continue
        dest = os.path.join(p["data_dir"], fn)
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        written.append(fn)
    return {"written": written, "count": len(written), "data_dir": p["data_dir"]}


@app.post("/dataset/clear")
def dataset_clear(name: str = Form(...)):
    p = _paths(name)
    if os.path.isdir(p["data_dir"]):
        shutil.rmtree(p["data_dir"], ignore_errors=True)
    os.makedirs(p["data_dir"], exist_ok=True)
    return {"ok": True, "data_dir": p["data_dir"]}


@app.post("/dataset/delete")
def dataset_delete(name: str = Form(...)):
    """Remove the entire dataset folder (data + tensors + train + adapters). Useful for
    cleaning up smoke tests or abandoned runs without box-side rmdir gymnastics."""
    p = _paths(name)
    removed = []
    if os.path.isdir(p["dataset_dir"]):
        shutil.rmtree(p["dataset_dir"], ignore_errors=True)
        removed.append(p["dataset_dir"])
    return {"ok": True, "removed": removed}


@app.get("/adapters")
def adapters():
    """Enumerate every trained LoRA/LoKr adapter that actually exists on disk.

    Walks <BASE>/<dataset>/<run>/ for `checkpoints/best/lokr_weights.safetensors`
    and `final/lokr_weights.safetensors`. Includes the legacy `train/` run and all
    per-run `train_<timestamp>__<config>/` dirs. Returns absolute box-side paths
    (correct Windows separators) the engine can load verbatim. This is what lets
    the LoRA picker list ALL runs (e.g. continuous vs discrete), not just the
    latest recorded in the Mac's training history."""
    out = []
    try:
        datasets = [d for d in sorted(os.listdir(BASE)) if os.path.isdir(os.path.join(BASE, d))]
    except Exception:
        datasets = []
    for ds in datasets:
        droot = os.path.join(BASE, ds)
        try:
            runs = [r for r in sorted(os.listdir(droot))
                    if os.path.isdir(os.path.join(droot, r)) and (r == "train" or r.startswith("train_"))]
        except Exception:
            runs = []
        for run in runs:
            rroot = os.path.join(droot, run)
            best = os.path.join(rroot, "checkpoints", "best", "lokr_weights.safetensors")
            final = os.path.join(rroot, "final", "lokr_weights.safetensors")
            entry = {"dataset": ds, "run_label": run}
            ok = False
            if os.path.isfile(best):
                entry["best_path"] = best
                ok = True
            if os.path.isfile(final):
                entry["final_path"] = final
                ok = True
            if ok:
                out.append(entry)
    return {"base_dir": BASE, "adapters": out}


@app.get("/dataset/paths")
def dataset_paths(name: str):
    p = _paths(name)
    p["exists"] = os.path.isdir(p["data_dir"])
    return p


@app.get("/dataset/list")
def dataset_list(name: str):
    p = _paths(name)
    if not os.path.isdir(p["data_dir"]):
        return JSONResponse({"files": []})
    return {"files": sorted(os.listdir(p["data_dir"]))}


# ============================ Filesystem API ============================
@app.get("/fs/roots")
def fs_roots():
    return {"roots": _fs_roots()}


@app.get("/fs/list")
def fs_list(path: str, glob: str = None):
    rp = _resolve(path)
    if not os.path.isdir(rp):
        raise HTTPException(404, "not a directory")
    entries = []
    for n in sorted(os.listdir(rp)):
        if glob and not fnmatch.fnmatch(n, glob):
            continue
        try:
            entries.append(_entry(os.path.join(rp, n)))
        except OSError:
            pass
    return {"path": rp, "count": len(entries), "entries": entries}


@app.get("/fs/stat")
def fs_stat(path: str):
    rp = _resolve(path)
    if not os.path.exists(rp):
        return {"exists": False, "path": rp}
    e = _entry(rp)
    e["exists"] = True
    return e


@app.get("/fs/read")
def fs_read(path: str, max_bytes: int = 2_000_000, b64: bool = False):
    rp = _resolve(path)
    if not os.path.isfile(rp):
        raise HTTPException(404, "not a file")
    size = os.path.getsize(rp)
    with open(rp, "rb") as f:
        raw = f.read(max_bytes + 1)
    truncated = len(raw) > max_bytes
    raw = raw[:max_bytes]
    if b64:
        return {"path": rp, "size": size, "truncated": truncated, "b64": base64.b64encode(raw).decode()}
    try:
        return {"path": rp, "size": size, "truncated": truncated, "content": raw.decode("utf-8")}
    except UnicodeDecodeError:
        return {"path": rp, "size": size, "truncated": truncated,
                "b64": base64.b64encode(raw).decode(), "note": "binary; returned base64"}


@app.get("/fs/tail")
def fs_tail(path: str, lines: int = 200):
    rp = _resolve(path)
    if not os.path.isfile(rp):
        raise HTTPException(404, "not a file")
    with open(rp, "rb") as f:
        f.seek(0, os.SEEK_END)
        end = f.tell()
        pos = end
        data = b""
        while pos > 0 and data.count(b"\n") <= lines:
            step = min(65536, pos)
            pos -= step
            f.seek(pos)
            data = f.read(step) + data
    text = data.decode("utf-8", errors="replace")
    tail = "\n".join(text.splitlines()[-lines:])
    return {"path": rp, "size": end, "lines": min(lines, tail.count("\n") + 1), "content": tail}


@app.get("/fs/find")
def fs_find(root: str, pattern: str, max: int = 1000):
    rp = _resolve(root)
    if not os.path.isdir(rp):
        raise HTTPException(404, "not a directory")
    matches = []
    for dirpath, dirs, files in os.walk(rp):
        for n in list(dirs) + list(files):
            if fnmatch.fnmatch(n, pattern):
                try:
                    matches.append(_entry(os.path.join(dirpath, n)))
                except OSError:
                    pass
                if len(matches) >= max:
                    return {"root": rp, "pattern": pattern, "matches": matches, "truncated": True}
    return {"root": rp, "pattern": pattern, "matches": matches, "truncated": False}


@app.get("/fs/du")
def fs_du(path: str):
    """Recursive size of a file or directory tree - for finding which run/checkpoint
    folders are eating disk before cleaning them up."""
    rp = _resolve(path)
    if not os.path.exists(rp):
        raise HTTPException(404, "not found")
    if os.path.isfile(rp):
        sz = os.path.getsize(rp)
        return {"path": rp, "bytes": sz, "files": 1, "human": _human(sz)}
    total = 0
    files = 0
    for dp, _dirs, fns in os.walk(rp):
        for n in fns:
            try:
                total += os.path.getsize(os.path.join(dp, n))
                files += 1
            except OSError:
                pass
    return {"path": rp, "bytes": total, "files": files, "human": _human(total)}


@app.post("/fs/write")
def fs_write(body: dict = Body(...)):
    _auth(body)
    rp = _resolve(body.get("path"))
    if "b64" in body and body["b64"] is not None:
        raw = base64.b64decode(body["b64"])
    elif "content" in body and body["content"] is not None:
        raw = body["content"].encode("utf-8")
    else:
        raise HTTPException(400, "content or b64 required")
    parent = os.path.dirname(rp)
    if parent:
        os.makedirs(parent, exist_ok=True)
    backup = None
    if body.get("backup") and os.path.isfile(rp):
        backup = rp + ".bak"
        shutil.copy2(rp, backup)
    with open(rp, "wb") as f:
        f.write(raw)
    st = os.stat(rp)
    return {"path": rp, "bytes": len(raw), "mtime_iso": _iso(st.st_mtime), "backup": backup}


@app.post("/fs/mkdir")
def fs_mkdir(body: dict = Body(...)):
    _auth(body)
    rp = _resolve(body.get("path"))
    os.makedirs(rp, exist_ok=True)
    return {"ok": True, "path": rp}


@app.post("/fs/move")
def fs_move(body: dict = Body(...)):
    _auth(body)
    src = _resolve(body.get("src"))
    dst = _resolve(body.get("dst"))
    if not os.path.exists(src):
        raise HTTPException(404, "src not found")
    parent = os.path.dirname(dst)
    if parent:
        os.makedirs(parent, exist_ok=True)
    shutil.move(src, dst)
    return {"ok": True, "src": src, "dst": dst}


@app.post("/fs/copy")
def fs_copy(body: dict = Body(...)):
    _auth(body)
    src = _resolve(body.get("src"))
    dst = _resolve(body.get("dst"))
    if not os.path.exists(src):
        raise HTTPException(404, "src not found")
    parent = os.path.dirname(dst)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
    return {"ok": True, "src": src, "dst": dst}


@app.post("/fs/delete")
def fs_delete(body: dict = Body(...)):
    _auth(body)
    rp = _resolve(body.get("path"))
    if not body.get("confirm"):
        raise HTTPException(400, "set confirm:true to delete")
    removed = None
    if os.path.isdir(rp):
        shutil.rmtree(rp, ignore_errors=True)
        removed = rp
    elif os.path.isfile(rp):
        os.remove(rp)
        removed = rp
    return {"ok": True, "removed": removed}


@app.post("/fs/pycompile")
def fs_pycompile(body: dict = Body(...)):
    """Compile a .py file with the box's own Python to catch syntax errors BEFORE an
    engine restart (verify a patch landed clean)."""
    rp = _resolve(body.get("path"))
    if not os.path.isfile(rp):
        raise HTTPException(404, "not a file")
    try:
        py_compile.compile(rp, doraise=True)
        return {"ok": True, "path": rp}
    except py_compile.PyCompileError as e:
        return {"ok": False, "path": rp, "error": str(e)}


# ----------------------------- ComfyUI process control -----------------------------
# Opt-in kill+restart for ComfyUI, so the caller can fully clear VRAM/RAM between jobs
# (GGUF models hold allocations that only a process restart frees). Windows-only (the box).
COMFY_PORT = int(os.environ.get("MG_COMFY_PORT", "8188"))


def _comfy_launcher():
    """Locate the ComfyUI portable dir + its LAN launcher .bat. MG_COMFY_LAUNCHER overrides
    the .bat name; MG_COMFY_DIR overrides the working dir. Default = the auto-detected
    ComfyUI_windows_portable folder + run_musicgen_lan.bat (the --listen + sage launcher)."""
    bat = os.environ.get("MG_COMFY_LAUNCHER", "run_musicgen_lan.bat")
    d = os.environ.get("MG_COMFY_DIR", "").strip()
    if not d:
        for r in _comfy_roots():
            cand = r if os.path.isfile(os.path.join(r, bat)) else os.path.join(r, "ComfyUI_windows_portable")
            if os.path.isfile(os.path.join(cand, bat)):
                d = cand
                break
    return (d or None), os.path.join(d, bat) if d else None


def _listening_pids(port: int):
    """PIDs LISTENING on `port` (Windows netstat). Scoped to the exact local port + LISTENING
    state so we never touch client connections or unrelated processes."""
    pids = set()
    try:
        out = subprocess.run(["netstat", "-ano", "-p", "tcp"], capture_output=True,
                             text=True, timeout=15).stdout
    except Exception:
        return pids
    for line in out.splitlines():
        p = line.split()
        if len(p) >= 5 and p[0].upper() == "TCP" and p[3].upper() == "LISTENING" \
                and p[1].endswith(":" + str(port)):
            if p[4].isdigit() and p[4] != "0":
                pids.add(p[4])
    return pids


def _pid_image(pid: str):
    """Image name for a PID (e.g. 'python.exe'), so we can confirm a kill target is actually
    ComfyUI's python and not whatever else might transiently hold the port."""
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        m = re.match(r'"([^"]+)"', out)
        return m.group(1) if m else None
    except Exception:
        return None


def _comfy_up(port: int, timeout: float = 2.0) -> bool:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/system_stats", timeout=timeout)
        return True
    except Exception:
        return False


@app.get("/comfy/status")
def comfy_status():
    cwd, bat = _comfy_launcher()
    return {"port": COMFY_PORT, "up": _comfy_up(COMFY_PORT),
            "listening_pids": sorted(_listening_pids(COMFY_PORT)),
            "launcher": bat, "launcher_found": bool(bat and os.path.isfile(bat))}


@app.post("/comfy/restart")
def comfy_restart(body: dict = Body(default={})):
    """Kill ComfyUI (only the PID LISTENING on its port - never by image name, so this can't
    hit the helper or other python) and relaunch its LAN .bat in a detached console, then wait
    for it to answer again. Body: {token?, wait?=true, timeout?=180, launch_only?=false}."""
    _auth(body or {})
    cwd, bat = _comfy_launcher()
    if not bat or not os.path.isfile(bat):
        raise HTTPException(404, f"ComfyUI launcher not found (looked for {bat!r}); set MG_COMFY_DIR/MG_COMFY_LAUNCHER")

    killed, skipped = [], []
    if not body.get("launch_only"):
        for pid in _listening_pids(COMFY_PORT):
            img = _pid_image(pid) or ""
            # ONLY kill if this exact PID is python - never an unrelated process that might
            # hold the port, and never by image name (which could hit the helper/other python).
            if not img.lower().startswith("python"):
                skipped.append({"pid": pid, "image": img})
                continue
            try:
                subprocess.run(["taskkill", "/F", "/T", "/PID", pid], capture_output=True, timeout=20)
                killed.append({"pid": pid, "image": img})
            except Exception:
                pass
        # wait for the port to actually free before relaunching (avoid bind clash)
        for _ in range(20):
            if not _listening_pids(COMFY_PORT):
                break
            time.sleep(0.5)

    # relaunch in a VISIBLE titled console so the user can watch ComfyUI's output. cmd's "start"
    # builtin opens a new window and returns immediately (so it outlives this request); the .bat's
    # own trailing `pause` keeps the window up if ComfyUI ever exits.
    batname = os.path.basename(bat)
    try:
        subprocess.Popen(f'start "ComfyUI" /D "{cwd}" "{batname}"', cwd=cwd, shell=True)
    except Exception as e:
        raise HTTPException(500, f"relaunch failed: {e}")

    waited = None
    if body.get("wait", True):
        timeout = float(body.get("timeout", 180))
        t0 = time.time()
        while time.time() - t0 < timeout:
            if _comfy_up(COMFY_PORT):
                waited = round(time.time() - t0, 1)
                break
            time.sleep(2)
    return {"killed": killed, "skipped_non_python": skipped, "launcher": bat,
            "relaunched": True, "up": _comfy_up(COMFY_PORT), "waited_seconds": waited}


# ----------------------------- AI Toolkit job control -----------------------------
# Fixed actions only (install / run a job file / stop / status) so training is driven from the
# Mac with nothing done by hand on the box. Processes start DETACHED in their own process group,
# so they survive a helper restart; their PID is kept in aitk_state.json and /aitk/stop kills
# only that tree (never by image name).
AITK_INSTALLER = os.path.join(HELPER_DIR, "install_ai_toolkit.bat")
AITK_STATE = os.path.join(HELPER_DIR, "aitk_state.json")
AITK_INSTALL_LOG = os.path.join(HELPER_DIR, "aitk_install.log")
ENGINE_PORT = int(os.environ.get("MG_ENGINE_PORT", "8001"))
_DETACH = 0x00000008 | 0x00000200          # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
_AITK_PROCS = {}                           # pid -> Popen, for exit codes while this helper lives


def _aitk_python():
    return os.path.join(AITK_DIR, ".venv", "Scripts", "python.exe")


def _aitk_state():
    try:
        with open(AITK_STATE) as f:
            return json.load(f)
    except Exception:
        return {}


def _aitk_alive(st):
    pid = st.get("pid")
    if not pid:
        return False
    proc = _AITK_PROCS.get(pid)
    if proc is not None:
        return proc.poll() is None
    img = (_pid_image(str(pid)) or "").lower()       # helper restarted: fall back to the PID's image
    return img in ("python.exe", "cmd.exe")


def _aitk_tail(path, lines=40):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 200_000))
            text = f.read().decode("utf-8", "replace")
        return "\n".join(text.replace("\r", "\n").splitlines()[-lines:])
    except Exception:
        return ""


def _aitk_spawn(args, cwd, log_path, env=None, **state):
    if _aitk_alive(_aitk_state()):
        raise HTTPException(409, "an AI Toolkit process is already running (POST /aitk/stop first)")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logf = open(log_path, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(args, cwd=cwd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, creationflags=_DETACH)
    _AITK_PROCS[proc.pid] = proc
    st = {"pid": proc.pid, "started": _iso(time.time()), "log": log_path, **state}
    with open(AITK_STATE, "w") as f:
        json.dump(st, f)
    return st


def _free_gpu():
    """Ask ComfyUI and the ACE engine to drop their models so training gets the 3090. A
    running ComfyUI render is never interrupted: the caller refuses to start instead."""
    out = {}
    for name, url, body in (("comfy", f"http://127.0.0.1:{COMFY_PORT}/free", {"unload_models": True, "free_memory": True}),
                            ("engine", f"http://127.0.0.1:{ENGINE_PORT}/v1/free", {})):
        try:
            req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=30)
            out[name] = "freed"
        except Exception as e:
            out[name] = f"skipped ({type(e).__name__})"
    return out


def _comfy_busy():
    try:
        q = json.load(urllib.request.urlopen(f"http://127.0.0.1:{COMFY_PORT}/queue", timeout=5))
        return bool(q.get("queue_running") or q.get("queue_pending"))
    except Exception:
        return False


@app.get("/aitk/status")
def aitk_status(lines: int = 30):
    st = _aitk_state()
    alive = _aitk_alive(st)
    proc = _AITK_PROCS.get(st.get("pid"))
    return {"dir": AITK_DIR, "installed": os.path.isfile(_aitk_python()),
            "installer_present": os.path.isfile(AITK_INSTALLER), **st, "alive": alive,
            "exit_code": (proc.poll() if proc is not None else None),
            "log_tail": _aitk_tail(st["log"], lines) if st.get("log") else ""}


@app.get("/aitk/log")
def aitk_log(lines: int = 200):
    st = _aitk_state()
    return {"log": st.get("log"), "content": _aitk_tail(st["log"], lines) if st.get("log") else ""}


@app.post("/aitk/install")
def aitk_install(body: dict = Body(default={})):
    _auth(body or {})
    if not os.path.isfile(AITK_INSTALLER):
        raise HTTPException(404, f"installer not deployed at {AITK_INSTALLER} (fs/write it there)")
    return _aitk_spawn(["cmd", "/c", AITK_INSTALLER, "nopause"], HELPER_DIR, AITK_INSTALL_LOG, kind="install")


@app.post("/aitk/run")
def aitk_run(body: dict = Body(...)):
    """Run ai-toolkit\\config\\<job>.yaml (a caption or training job written by the Mac) with the
    toolkit's own venv. Body: {job, token?, free_gpu?=true}. Model downloads go to ai-toolkit\\.hf."""
    _auth(body or {})
    job = _safe(str(body.get("job") or ""))
    cfg = os.path.join(AITK_DIR, "config", job + ".yaml")
    if not os.path.isfile(_aitk_python()):
        raise HTTPException(409, "AI Toolkit is not installed (POST /aitk/install)")
    if not os.path.isfile(cfg):
        raise HTTPException(404, f"no job file {cfg}")
    if _comfy_busy():
        raise HTTPException(409, "ComfyUI has a render queued or running; try again when it is idle")
    freed = _free_gpu() if body.get("free_gpu", True) else {}
    env = dict(os.environ, HF_HOME=os.path.join(AITK_DIR, ".hf"), PYTHONUNBUFFERED="1")
    for k in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX"):
        env.pop(k, None)
    log = os.path.join(AITK_DIR, "output", job + ".log")
    st = _aitk_spawn([_aitk_python(), "run.py", cfg, "--log", log + ".run"], AITK_DIR, log,
                     env=env, kind="job", job=job)
    return {**st, "freed": freed}


@app.post("/aitk/script")
def aitk_script(body: dict = Body(...)):
    """Run one of the app's own scripts (ai-toolkit\\musicgen\\<script>.py, written by the Mac) in the
    toolkit's venv with its config (ai-toolkit\\config\\<config>.json) - for model tests that are not
    training jobs, e.g. the YuE2 round trip. Same GPU rules as /aitk/run (one process at a time,
    never over a ComfyUI render). Body: {script, config, token?, free_gpu?=true}."""
    _auth(body or {})
    script = os.path.join(AITK_DIR, "musicgen", _safe(str(body.get("script") or "")) + ".py")
    cfg_name = _safe(str(body.get("config") or ""))
    cfg = os.path.join(AITK_DIR, "config", cfg_name + ".json")
    if not os.path.isfile(_aitk_python()):
        raise HTTPException(409, "AI Toolkit is not installed (POST /aitk/install)")
    for p in (script, cfg):
        if not os.path.isfile(p):
            raise HTTPException(404, f"no file {p}")
    if _comfy_busy():
        raise HTTPException(409, "ComfyUI has a render queued or running; try again when it is idle")
    freed = _free_gpu() if body.get("free_gpu", True) else {}
    env = dict(os.environ, HF_HOME=os.path.join(AITK_DIR, ".hf"), PYTHONUNBUFFERED="1", PYTHONPATH=AITK_DIR)
    for k in ("PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX"):
        env.pop(k, None)
    log = os.path.join(AITK_DIR, "output", cfg_name + ".log")
    st = _aitk_spawn([_aitk_python(), script, cfg], AITK_DIR, log, env=env, kind="script", job=cfg_name)
    return {**st, "freed": freed}


@app.post("/aitk/stop")
def aitk_stop(body: dict = Body(default={})):
    _auth(body or {})
    st = _aitk_state()
    if not _aitk_alive(st):
        return {"stopped": False, "reason": "nothing running", **st}
    r = subprocess.run(["taskkill", "/F", "/T", "/PID", str(st["pid"])], capture_output=True, text=True, timeout=30)
    return {"stopped": r.returncode == 0, "pid": st["pid"], "output": (r.stdout or r.stderr).strip()[-300:]}


@app.post("/helper/restart")
def helper_restart(body: dict = Body(default={})):
    """Relaunch this helper so an fs/write-deployed update takes effect remotely: a detached cmd
    waits for this process to exit (freeing the port), then starts run_lora_upload.bat in a new
    titled console; this process exits right after answering. Running AI Toolkit jobs are
    detached and keep going."""
    _auth(body or {})
    launcher = os.path.join(HELPER_DIR, "run_lora_upload.bat")
    if not os.path.isfile(launcher):
        raise HTTPException(404, f"launcher not found: {launcher}")
    subprocess.Popen(f'cmd /c "timeout /t 4 /nobreak >nul & start "lora_upload" /D "{HELPER_DIR}" run_lora_upload.bat"',
                     cwd=HELPER_DIR, shell=False, creationflags=_DETACH,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import threading
    threading.Timer(1.0, lambda: os._exit(0)).start()
    return {"restarting": True, "launcher": launcher}


DAV_ENCODER_FILE = "minimax_music3_dav_encoder.safetensors"
DAV_ENCODER_URL = ("https://huggingface.co/SimpleTuner/MiniMax-Music-3-Encoder/resolve/main/"
                   "audio_vae/diffusion_pytorch_model.safetensors")


def _comfy_app():
    """(portable_dir, app_dir, embedded_python) for the ComfyUI install, or (None, None, None)."""
    d, _bat = _comfy_launcher()
    if not d:
        return None, None, None
    app_dir = os.path.join(d, "ComfyUI")
    py = os.path.join(d, "python_embeded", "python.exe")
    if not (os.path.isfile(os.path.join(app_dir, "main.py")) and os.path.isfile(py)):
        return None, None, None
    return d, app_dir, py


@app.post("/dav/encode")
def dav_encode(body: dict = Body(...)):
    """Encode an audio file already in ComfyUI's input dir into a Music 3 .latent, on this
    box's GPU, writing the .latent back into the same input dir where the stock LoadLatent
    node reads it. Body: {token?, input_name, out_name?, max_seconds?}.

    Runs backend-deployed dav_encode_box.py under ComfyUI's own embedded python (torch cu130,
    av, torchaudio - nothing to install). First call fetches the 292MB encoder weights into
    ComfyUI models\\vae via curl. The encode itself is chunked, a few seconds of GPU."""
    _auth(body or {})
    portable, app_dir, py = _comfy_app()
    if not py:
        raise HTTPException(404, "ComfyUI portable install not found (set MG_COMFY_DIR)")
    script = os.path.join(portable, "dav_encode_box.py")
    if not os.path.isfile(script):
        raise HTTPException(404, f"dav_encode_box.py not deployed at {script} (fs/write it there)")

    name = os.path.basename(str(body.get("input_name") or ""))
    src = os.path.join(app_dir, "input", name)
    if not (name and os.path.isfile(src)):
        raise HTTPException(404, f"input audio not found: {src}")
    out_name = os.path.basename(str(body.get("out_name") or "")) or (os.path.splitext(name)[0] + ".latent")
    out = os.path.join(app_dir, "input", out_name)

    ckpt = os.path.join(app_dir, "models", "vae", DAV_ENCODER_FILE)
    if not os.path.isfile(ckpt):
        tmp = ckpt + ".part"
        r = subprocess.run(["curl", "-L", "-C", "-", "-o", tmp, DAV_ENCODER_URL],
                           capture_output=True, text=True, timeout=1800)
        if r.returncode != 0 or not os.path.isfile(tmp):
            raise HTTPException(502, f"encoder download failed: {(r.stderr or '')[-300:]}")
        os.replace(tmp, ckpt)

    args = [py, script, src, out, ckpt]
    if body.get("max_seconds"):
        args.append(str(float(body["max_seconds"])))
    r = subprocess.run(args, capture_output=True, text=True, timeout=900)
    lines = [ln for ln in (r.stdout or "").splitlines() if ln.strip()]
    if r.returncode != 0 or not lines:
        raise HTTPException(500, f"encode failed: {(r.stderr or r.stdout or '')[-500:]}")
    try:
        return json.loads(lines[-1])
    except Exception:
        raise HTTPException(500, f"encode output unparseable: {lines[-1][:300]}")


if __name__ == "__main__":
    import uvicorn
    print(f"[lora-upload] base dir: {BASE}")
    print(f"[lora-upload] fs roots: {_fs_roots()}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
