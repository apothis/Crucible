"""Mac-side client for the box ANALYZE service (allin1 + CLAP + key), RESEARCH §17 P2.

Posts an audio file to the box; returns {bpm, key, tags, segments:[{start,end,label}],
beats, downbeats}. The box runs allin1/CLAP on the 3090. See analyze_server.py.
"""
import requests


def _base(host):
    host = (host or "").strip()
    if not host:
        raise RuntimeError("analyze_host not configured")
    return host if host.startswith("http") else f"http://{host}"


def health(host):
    r = requests.get(_base(host) + "/health", timeout=10)
    r.raise_for_status()
    return r.json()


def analyze(host, path, labels=None, with_tags=True, with_key=True, with_scores=False,
            timeout=600):
    """POST /analyze with the audio file. `labels` = optional style vocabulary (list)
    for CLAP zero-shot tagging. with_scores=True also returns `tag_scores` (raw cosine
    magnitudes, not just ranks). Returns the parsed JSON dict."""
    import json as _json
    import os
    with open(path, "rb") as f:
        files = {"file": (os.path.basename(path), f.read(), "application/octet-stream")}
    data = {"with_tags": "true" if with_tags else "false",
            "with_key": "true" if with_key else "false",
            "with_scores": "true" if with_scores else "false"}
    if labels:
        data["labels"] = _json.dumps(list(labels))
    r = requests.post(_base(host) + "/analyze", files=files, data=data, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    if isinstance(j, dict) and j.get("error"):
        raise RuntimeError(j["error"])
    return j


def embed(host, path, timeout=600):
    """POST /embed -> the L2-normalized CLAP audio embedding (list of floats) for one file.
    The building block for centroid-distance: cosine(this, mean(artist corpus))."""
    import os
    with open(path, "rb") as f:
        files = {"file": (os.path.basename(path), f.read(), "application/octet-stream")}
    r = requests.post(_base(host) + "/embed", files=files, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    if isinstance(j, dict) and j.get("error"):
        raise RuntimeError(j["error"])
    return j["embedding"]
