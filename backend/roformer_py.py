"""Mac-side client for the box's BS-Roformer separation service (roformer_server.py
on the Windows GPU box). HTTP only — the heavy model runs on the 3090 (running it on
the Mac's MPS hard-crashes the machine: too much unified-memory pressure).

  health(host)                         -> dict (ok/model/device) or raises
  separate(audio_bytes, filename, host, stems="all")
        -> {"sr": int, "stems": {name: wav_bytes}}

The caller (backend/app.py) frees ComfyUI's VRAM (POST /free) before calling, so the
3090 isn't holding the ACE-Step models when the separator loads.
"""
import base64
import requests

TIMEOUT = 1200   # separation can take a while on a long track


def _base(host):
    host = (host or "").strip()
    if not host:
        raise RuntimeError("roformer_host not configured")
    return host if host.startswith("http") else f"http://{host}"


def health(host):
    r = requests.get(_base(host) + "/health", timeout=10)
    r.raise_for_status()
    return r.json()


def separate(audio_bytes, filename, host, stems="all"):
    files = {"audio": (filename or "mix.wav", audio_bytes, "application/octet-stream")}
    data = {"stems": stems or "all"}
    r = requests.post(_base(host) + "/separate", files=files, data=data, timeout=TIMEOUT)
    if r.status_code != 200:
        try:
            j = r.json()
            raise RuntimeError(j.get("error") or j.get("stderr") or r.text[:500])
        except ValueError:
            raise RuntimeError(f"separation failed: {r.text[:500]}")
    j = r.json()
    return {"sr": j.get("sr", 44100),
            "stems": {k: base64.b64decode(v) for k, v in j.get("stems", {}).items()}}
