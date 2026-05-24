"""Voice-model discovery + install helper.

Search is powered by the Hugging Face API (voice-models.com has no public API,
so for that site users paste a download URL). Install pushes the model to the
rvc-python server's /upload_model endpoint, so voices land on the Windows box
where RVC runs — the Mac never needs the Windows filesystem.

rvc-python's /upload_model does `zip.extractall(models_dir)` and then scans
`models_dir/<subfolder>/*.pth`. So whatever we upload MUST be a zip whose top
level is `<name>/<model>.pth` (+ optional `.index`). Community voices come in
all shapes (flat zip, nested folder, bare .pth), so we ALWAYS normalize on the
backend: extract, locate the .pth/.index, and repackage cleanly. Transparent.
"""
import io
import os
import posixpath
import re
import zipfile

import requests

HF = "https://huggingface.co"


def _sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s or "").strip("._-") or "voice"


def search(q: str, sort: str = "likes", limit: int = 25):
    params = {"search": q, "sort": sort, "direction": -1, "limit": limit, "full": "false"}
    r = requests.get(f"{HF}/api/models", params=params, timeout=15)
    r.raise_for_status()
    return [{"id": m["id"], "likes": m.get("likes", 0), "downloads": m.get("downloads", 0)}
            for m in r.json()]


def repo_voices(repo_id: str):
    """Installable items in an HF repo: each .pth (paired with same-folder .index),
    plus any .zip archives (normalized on install)."""
    r = requests.get(f"{HF}/api/models/{repo_id}", timeout=20)
    r.raise_for_status()
    sib = [s["rfilename"] for s in r.json().get("siblings", [])]
    idxs = [f for f in sib if f.lower().endswith(".index")]
    out = []
    for p in sib:
        if p.lower().endswith(".pth"):
            folder = posixpath.dirname(p)
            stem = posixpath.splitext(posixpath.basename(p))[0]
            idx = next((i for i in idxs if posixpath.dirname(i) == folder), "")
            out.append({"name": posixpath.basename(folder) or stem, "pth": p, "index": idx})
    for z in sib:
        if z.lower().endswith(".zip"):
            out.append({"name": posixpath.splitext(posixpath.basename(z))[0], "zip": z})
    return out


def _dl(url: str) -> bytes:
    r = requests.get(url, timeout=600)
    r.raise_for_status()
    return r.content


def _zip_model(name, pth_bytes, pth_name, index_bytes=None, index_name=None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        z.writestr(f"{name}/{pth_name}", pth_bytes)
        if index_bytes is not None:
            z.writestr(f"{name}/{index_name}", index_bytes)
    return buf.getvalue()


def _normalize_zip(raw: bytes, name=None):
    """Extract an arbitrary voice zip and repackage as <name>/<pth>(+<index>)."""
    zin = zipfile.ZipFile(io.BytesIO(raw))
    names = [n for n in zin.namelist() if not n.endswith("/") and "__MACOSX" not in n]
    pths = [n for n in names if n.lower().endswith(".pth")]
    idxs = [n for n in names if n.lower().endswith(".index")]
    if not pths:
        raise ValueError("zip contains no .pth model file")
    fold = posixpath.dirname
    pth = next((p for p in pths if any(fold(i) == fold(p) for i in idxs)), pths[0])
    idx = next((i for i in idxs if fold(i) == fold(pth)), idxs[0] if idxs else None)
    nm = _sanitize(name or posixpath.splitext(posixpath.basename(pth))[0])
    pb = zin.read(pth)
    ib = zin.read(idx) if idx else None
    return nm, _zip_model(nm, pb, posixpath.basename(pth), ib,
                          posixpath.basename(idx) if idx else None)


def _upload(rvc_base: str, name: str, zipbytes: bytes):
    if not rvc_base:
        raise RuntimeError("rvc-python server not configured")
    files = {"file": (f"{name}.zip", zipbytes, "application/zip")}
    r = requests.post(f"{rvc_base}/upload_model", files=files, timeout=600)
    r.raise_for_status()
    try:
        msg = r.json()
    except Exception:
        msg = r.text
    return {"ok": True, "name": name, "message": msg}


def _install_bytes(rvc_base, data: bytes, fname: str, name=None):
    """Handle any downloaded payload: zip (by extension OR magic bytes) or bare .pth."""
    low = fname.lower()
    is_zip = low.endswith(".zip") or data[:2] == b"PK"
    if is_zip and not low.endswith(".pth"):
        nm, z = _normalize_zip(data, name)
        return _upload(rvc_base, nm, z)
    if low.endswith(".pth") or data[:1] == b"P":  # torch .pth is itself a zip; ext decides
        nm = _sanitize(name or os.path.splitext(os.path.basename(fname))[0])
        return _upload(rvc_base, nm, _zip_model(nm, data, os.path.basename(fname) or f"{nm}.pth"))
    raise ValueError("unrecognized voice file (expected .zip or .pth)")


def install_from_url(rvc_base, url, name=None):
    data = _dl(url)
    fname = url.split("/")[-1].split("?")[0] or "voice"
    return _install_bytes(rvc_base, data, fname, name)


def install_from_hf(rvc_base, repo_id, pth=None, index=None, zip=None, name=None):
    if zip:
        nm, z = _normalize_zip(_dl(f"{HF}/{repo_id}/resolve/main/{zip}"), name)
        return _upload(rvc_base, nm, z)
    nm = _sanitize(name or posixpath.splitext(posixpath.basename(pth))[0])
    pb = _dl(f"{HF}/{repo_id}/resolve/main/{pth}")
    ib = _dl(f"{HF}/{repo_id}/resolve/main/{index}") if index else None
    return _upload(rvc_base, nm, _zip_model(nm, pb, posixpath.basename(pth), ib,
                                            posixpath.basename(index) if index else None))
