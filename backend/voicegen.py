"""Engine-agnostic singing-synthesis layer for the Vocal Builder.

A composed melody score (from melody.py) is turned into a sung vocal by a
pluggable ENGINE. Mirrors the app's other abstractions (llm.py providers, the
RVC driver switch) so more engines can be added without touching callers.

Engines (see RESEARCH.md §5b):
  - "guide"     : Mac-only synthetic guide vocal (always available). Pitch-
                  accurate "ah" melody; re-timbre with RVC for a real voice.
  - "soulx"     : SoulX-Singer (zero-shot SVS) over a Windows API host.
  - "diffsinger": DiffSinger/NNSVS voicebank over a Windows API host.
Host engines speak a simple contract: POST http://<host>/synthesize with the
score JSON (+ optional reference clip) -> audio/wav bytes. They light up in the
UI once their host is configured in app_config.json and reachable.
"""
import requests

from . import melody as melody_mod

ENGINES = [
    {"id": "guide", "label": "Synth guide (Mac)", "host_key": None,
     "sings_words": False,
     "desc": "Instant synthetic melody on the Mac — pitch-accurate 'ah' vocal. "
             "Pair with RVC re-timbre for a real voice (wordless)."},
    {"id": "soulx", "label": "SoulX-Singer (zero-shot)", "host_key": "soulx_host",
     "health_path": "/health", "sings_words": True,
     "desc": "Sings the melody with the actual lyrics; clones a target timbre "
             "from a short reference clip (no voicebank training). Needs the "
             "SoulX API on the Windows GPU."},
    {"id": "diffsinger", "label": "DiffSinger voicebank", "host_key": "diffsinger_host",
     "health_path": "/version", "sings_words": True,
     "desc": "Sings the lyrics with a trained DiffSinger voicebank. Needs the "
             "DiffSinger MiniEngine on the Windows GPU."},
]


def _reachable(host: str, path: str) -> bool:
    if not host:
        return False
    try:
        r = requests.get(f"http://{host}{path}", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def engines(cfg: dict):
    """List engines with availability resolved from config/hosts."""
    out = []
    for e in ENGINES:
        host = cfg.get(e["host_key"], "") if e["host_key"] else ""
        available = True if e["host_key"] is None else _reachable(host, e.get("health_path", "/health"))
        out.append({**e, "host": host, "available": available})
    return out


def _host_synth(host: str, score: dict, lyrics: str, reference: bytes = None,
                opts: dict = None) -> bytes:
    """Generic client for a host-based SVS engine (SoulX/DiffSinger).
    POSTs the score; returns WAV bytes."""
    if not host:
        raise RuntimeError("engine host not configured in app_config.json")
    import json as _json
    data = {"score": _json.dumps(score), "lyrics": lyrics,
            "opts": _json.dumps(opts or {})}
    files = {"reference": ("ref.wav", reference, "audio/wav")} if reference else None
    r = requests.post(f"http://{host}/synthesize", data=data, files=files, timeout=600)
    if not r.ok:
        detail = ""
        try:
            detail = r.json().get("error", "")
        except Exception:
            detail = r.text[:300]
        raise RuntimeError(detail or f"{r.status_code} {r.reason}")
    return r.content


_G2P = None


def _phones(text: str):
    """English word/syllable → lowercased, stress-stripped ARPABET phones
    (the common English DiffSinger dictionary convention). Approximate; the
    exact set must match the installed voicebank's dictionary."""
    global _G2P
    if _G2P is None:
        from g2p_en import G2p
        _G2P = G2p()
    out = [p.lower().rstrip("0123") for p in _G2P(text) if p and p[0].isalpha()]
    return out or ["sp"]


def _diffsinger_synth(host: str, score: dict, opts: dict) -> bytes:
    """Drive openvpi DiffSingerMiniEngine (native API: /models, /rhythm,
    /submit, /query, /download). Pitch comes from an explicit f0 curve we
    build from the melody (MIDI-less acoustic model). See RESEARCH.md §5d."""
    import time as _time
    import requests
    if not host:
        raise RuntimeError("diffsinger_host not configured")
    base = f"http://{host}"
    models = requests.get(f"{base}/models", timeout=10).json().get("models", [])
    model = (opts or {}).get("model") or (models[0] if models else None)
    if not model:
        raise RuntimeError("no DiffSinger acoustic voicebank installed on the host")

    notes, cursor = [], 0.0
    for n in sorted(score.get("notes", []), key=lambda x: x["start"]):
        gap = n["start"] - cursor
        if gap > 0.05:
            notes.append({"key": 0, "duration": round(gap, 3), "slur": False, "phonemes": ["SP"]})
        notes.append({"key": int(n["midi"]), "duration": round(n["dur"], 3),
                      "slur": False, "phonemes": _phones(n.get("syllable", ""))})
        cursor = n["start"] + n["dur"]
    if not notes:
        raise RuntimeError("score has no notes")

    ph = requests.post(f"{base}/rhythm", json={"notes": notes}, timeout=120).json()["phonemes"]
    step = 0.01
    steps = max(1, int(sum(x["duration"] for x in notes) / step))
    f0 = [0.0] * steps
    t = 0.0
    for nt in notes:
        i0, i1 = int(t / step), int((t + nt["duration"]) / step)
        hz = 0.0 if nt["key"] == 0 else 440.0 * 2 ** ((nt["key"] - 69) / 12.0)
        for i in range(i0, min(i1, steps)):
            f0[i] = hz
        t += nt["duration"]

    sub = requests.post(f"{base}/submit", json={
        "model": model, "phonemes": ph, "f0": {"timestep": step, "values": f0},
        "speedup": (opts or {}).get("speedup", 20)}, timeout=30).json()
    token = sub["token"]
    for _ in range(600):
        q = requests.post(f"{base}/query", json={"token": token}, timeout=10).json()
        st = q.get("status")
        if st in ("FINISHED", "HIT_CACHE"):
            break
        if st in ("FAILED", "CANCELLED"):
            raise RuntimeError(f"diffsinger {st}: {q.get('message', '')}")
        _time.sleep(1)
    return requests.get(f"{base}/download", params={"token": token}, timeout=60).content


def synthesize(engine: str, score: dict, cfg: dict, reference: bytes = None,
               opts: dict = None) -> bytes:
    """Produce a sung-vocal WAV for the score using the chosen engine."""
    lyrics = "\n".join(s["lyrics"] for s in score.get("sections", []) if s.get("lyrics"))
    if engine == "guide":
        return melody_mod.render_guide(score)
    spec = next((e for e in ENGINES if e["id"] == engine), None)
    if not spec:
        raise RuntimeError(f"unknown engine: {engine}")
    host = cfg.get(spec["host_key"], "") if spec["host_key"] else ""
    if engine == "diffsinger":
        return _diffsinger_synth(host, score, opts or {})
    return _host_synth(host, score, lyrics, reference=reference, opts=opts)
