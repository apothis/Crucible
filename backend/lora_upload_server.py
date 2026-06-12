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
  POST /fs/write  {path, content|b64, backup?}  -> {bytes, mtime_iso, backup?}  (creates parent dirs)
  POST /fs/mkdir  {path}               -> {ok, path}
  POST /fs/move   {src, dst}           -> {ok}
  POST /fs/copy   {src, dst}           -> {ok}
  POST /fs/delete {path, confirm:true} -> {ok, removed}
  POST /fs/pycompile {path}            -> {ok} | {ok:false, error}  (py_compile on the box)

SAFETY: every fs path is realpath-resolved and must sit inside one of MG_FS_ROOTS
(default: the engine install tree = two levels up from BASE, plus BASE). Paths outside
-> 403. If MG_FS_TOKEN is set, mutating ops (write/mkdir/move/copy/delete) require a
matching `token` in the body. /fs/delete additionally requires `confirm:true`.
Deliberately NO arbitrary command/exec endpoint in this cut - ask if you want one (opt-in).
"""
import os
import re
import base64
import datetime
import fnmatch
import py_compile
import shutil

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse

PORT = int(os.environ.get("MG_LORA_PORT", "5080"))
BASE = os.environ.get("MG_LORA_DIR",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "lora_data"))
os.makedirs(BASE, exist_ok=True)

# Optional shared token gating the MUTATING fs ops (write/mkdir/move/copy/delete).
# Unset -> no auth (matches the existing dataset endpoints' trusted-LAN model).
FS_TOKEN = os.environ.get("MG_FS_TOKEN", "").strip()

app = FastAPI()


# ----------------------------- fs sandbox helpers -----------------------------
def _fs_roots():
    """Allowed roots for the filesystem API. MG_FS_ROOTS (os.pathsep-separated) overrides;
    default = BASE + the engine install tree (1 and 2 dirs up from BASE), so reads/patches
    of acestep source, the .venv site-packages, lora_data and the HF .cache all work, while
    the rest of the disk stays out of reach."""
    env = os.environ.get("MG_FS_ROOTS", "").strip()
    if env:
        cands = [p for p in env.split(os.pathsep) if p.strip()]
    else:
        cands = [BASE, os.path.dirname(BASE), os.path.dirname(os.path.dirname(BASE))]
    roots = []
    for p in cands:
        try:
            rp = os.path.normcase(os.path.realpath(p))
            if rp and os.path.isdir(rp) and rp not in roots:
                roots.append(rp)
        except OSError:
            pass
    return roots


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


if __name__ == "__main__":
    import uvicorn
    print(f"[lora-upload] base dir: {BASE}")
    print(f"[lora-upload] fs roots: {_fs_roots()}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
