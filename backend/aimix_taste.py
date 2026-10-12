"""AI Mix taste (docs/AI_MIX_PLAN.md): targets learned from the versions the user marks preferred.

Style presets describe what reference songs do; the user's ears can prefer something else (on
Dream Of Me the preferred vocal was denser, wetter and further over the backing than any
symphonic reference [M]). When a version is starred, its vocal treatment is measured from the
render's mix-doc snapshot and folded into taste.json (mean over all starred versions). The
adaptive rules then use taste targets for the vocal moves they cover, and the style for the rest.
"""
import json
import os
import time

import numpy as np

from . import aimix
from . import aimix_features as F

TASTE_KEYS = ("vocal_phrase_range_db", "vocal_dip_db", "lead_over_backing_db", "vocal_over_rest_db")


def path(root):
    return os.path.join(root, "taste.json")


def load(root):
    p = path(root)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _excerpt(x, sr, s=120.0):
    n, h = x.shape[0], int(s * sr / 2)
    return x if n <= 2 * h else x[n // 2 - h:n // 2 + h]


def render_vocal(doc, with_verb=True, vocals_only=False):
    """Processed + landed vocal-group stems of a mix doc (middle excerpt), optionally with the
    vocal share of the reverb/delay buses added back. Returns (lead, backing, vocal_total, rest, sr)."""
    import pedalboard as pb
    import pyloudnorm as pyln
    lead = back = rest = verb = dly = None
    sr = None
    for n, st in doc["stems"].items():
        is_voc = (st.get("group") or "") == "vocals" or st.get("role") in ("vocals", "backing")
        if st.get("mute") or (vocals_only and not is_voc):
            continue
        x, sr = aimix._read(st["file"])
        y = x.T.copy()
        y = aimix._apply_chain(st["chain"], y, sr) if st["chain"] else y
        y = aimix._width_lowmono(y, sr, float(st.get("width", 1.0)), float(st.get("low_mono_hz") or 0))
        cur = pyln.Meter(sr).integrated_loudness(y.T)
        g = (st["target_lufs"] - cur if np.isfinite(cur) else -30.0) + float(st.get("trim_db", 0.0))
        y = _excerpt((y * 10 ** (g / 20.0)).T, sr)
        if is_voc:
            if st.get("role") == "backing":
                back = y if back is None else back + y
            else:
                lead = y if lead is None else lead + y
            if with_verb:
                sends = st.get("sends", {})
                if "verb" in sends:
                    v = y * 10 ** (sends["verb"] / 20.0)
                    verb = v if verb is None else verb + v
                if "delay" in sends:
                    v = y * 10 ** (sends["delay"] / 20.0)
                    dly = v if dly is None else dly + v
        else:
            rest = y if rest is None else rest[:len(y)] + y[:len(rest)]
    voc = (lead if lead is not None else 0) + (back if back is not None else 0)
    if with_verb and verb is not None:
        vb = doc["buses"]["verb"]
        voc = voc + pb.Pedalboard([
            pb.HighpassFilter(cutoff_frequency_hz=float(vb.get("hpf", 200))),
            pb.Delay(delay_seconds=float(vb.get("predelay_ms", 30)) / 1000.0, feedback=0.0, mix=1.0),
            pb.Reverb(room_size=float(vb.get("room_size", 0.35)), damping=float(vb.get("damping", 0.5)), wet_level=1.0, dry_level=0.0),
            pb.LowpassFilter(cutoff_frequency_hz=float(vb.get("lpf", 8000))),
        ])(verb.T.copy(), sr).T
    if with_verb and dly is not None:
        db_ = doc["buses"]["delay"]
        voc = voc + pb.Pedalboard([
            pb.HighpassFilter(cutoff_frequency_hz=float(db_.get("hpf", 300))),
            pb.Delay(delay_seconds=float(db_.get("ms", 500)) / 1000.0, feedback=float(db_.get("feedback", 0.25)), mix=1.0),
            pb.LowpassFilter(cutoff_frequency_hz=float(db_.get("lpf", 6000))),
        ])(dly.T.copy(), sr).T
    return lead, back, voc, rest, sr


def measure(doc):
    """The vocal treatment of a mix doc, in taste terms."""
    lead, back, voc, rest, sr = render_vocal(doc)
    m = voc.mean(1)
    n = min(len(voc), len(rest)) if rest is not None else 0
    out = {"vocal_phrase_range_db": F.dynamics(m, sr)["phrase_range_db"],
           "vocal_dip_db": F.wetness(m, sr)["dip_db"]}
    if lead is not None and back is not None:
        out["lead_over_backing_db"] = float(10 * np.log10(np.mean(lead ** 2) / max(np.mean(back ** 2), 1e-20)))
    if n:
        out["vocal_over_rest_db"] = float(10 * np.log10(np.mean(voc[:n] ** 2) / max(np.mean(rest[:n] ** 2), 1e-20)))
    return out


def add_preferred(root, doc, info):
    """Fold a starred version into taste.json (plain mean over starred versions)."""
    t = load(root) or {"picks": []}
    m = measure(doc)
    t["picks"] = [p for p in t["picks"] if p.get("job_id") != info.get("job_id")]
    t["picks"].append({**info, "measured": m, "at": time.time()})
    t["vocals"] = {}
    for k in TASTE_KEYS:
        vals = [p["measured"][k] for p in t["picks"] if p["measured"].get(k) is not None]
        if vals:
            t["vocals"][k] = float(np.mean(vals))
    with open(path(root), "w") as f:
        json.dump(t, f, indent=1)
    return t


def remove_preferred(root, job_id):
    t = load(root)
    if not t:
        return None
    t["picks"] = [p for p in t["picks"] if p.get("job_id") != job_id]
    t["vocals"] = {}
    for k in TASTE_KEYS:
        vals = [p["measured"][k] for p in t["picks"] if p["measured"].get(k) is not None]
        if vals:
            t["vocals"][k] = float(np.mean(vals))
    with open(path(root), "w") as f:
        json.dump(t, f, indent=1)
    return t
