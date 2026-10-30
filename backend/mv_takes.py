"""Read side of the Music Video tab (docs/MV_UI_DESIGN.md, phase 1): the saved scripts of a song,
the takes rendered for a scene, and the on-demand checks on a take (song-envelope floor +
opening_jump + contact sheet). GPU-free, Mac only. Plain ASCII.

Checks are cached under .mvwork/mv_take_checks/<job_id>/ (check.json + sheet.png) so the tab can
show them without re-measuring; a take never changes once rendered.
"""
import json
import os
import subprocess
import time

import numpy as np

from . import mv_take_check


def scripts_dir(library_dir):
    return os.path.join(os.path.dirname(library_dir), ".mvwork", "mv_scripts")


def checks_dir(library_dir):
    return os.path.join(os.path.dirname(library_dir), ".mvwork", "mv_take_checks")


def list_scripts(library_dir, audio_id=None):
    """Saved scripts (newest first) with the counts the tab shows: scenes, clips, stills filed,
    takes locked. `audio_id` narrows to one song."""
    d = scripts_dir(library_dir)
    out = []
    if not os.path.isdir(d):
        return out
    for fn in os.listdir(d):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(d, fn)
        try:
            with open(path) as fh:
                s = json.load(fh)
        except Exception:
            continue
        if not isinstance(s, dict) or "scenes" not in s:
            continue
        if audio_id and str(s.get("audio_id")) != str(audio_id):
            continue
        scenes = s.get("scenes") or []
        out.append({"name": fn, "title": s.get("title"), "audio_id": s.get("audio_id"),
                    "mtime": os.path.getmtime(path), "duration": s.get("duration"),
                    "scenes": len(scenes),
                    "clips": sum(len(sc.get("clips") or []) for sc in scenes),
                    "stills": sum(1 for sc in scenes if sc.get("opening_still_id")),
                    "locks": sum(1 for sc in scenes if sc.get("locked")),
                    "problems": sum(len(sc.get("scene_problems") or []) for sc in scenes)})
    out.sort(key=lambda x: -x["mtime"])
    return out


def _duration(path):
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return round(float(out), 3)
    except Exception:
        return None


def cached_check(library_dir, job_id):
    p = os.path.join(checks_dir(library_dir), job_id, "check.json")
    if os.path.exists(p):
        try:
            with open(p) as fh:
                return json.load(fh)
        except Exception:
            return None
    return None


TAKE_PARAM_KEYS = ("seed", "width", "height", "turbo", "lane", "first_frame_id", "anchor_first", "distant",
                   "free_av", "scene_start", "scene_end", "scene_title", "scene_index", "frames",
                   "context_frames", "location_ref_id", "upscale_width", "upscale_height")


def takes_for_script(rows, script_name, library_dir, scene=None):
    """Library rows (sqlite dicts with params JSON) -> {scene_index: [take...]} for one script.
    Running jobs are passed in the same shape with status != done."""
    out = {}
    for r in rows:
        try:
            prm = r["params"] if isinstance(r["params"], dict) else json.loads(r["params"])
        except Exception:
            continue
        if os.path.basename(str(prm.get("script") or "")) != script_name:
            continue
        idx = prm.get("scene_index")
        if idx is None or (scene is not None and int(idx) != int(scene)):
            continue
        jid = r["id"]
        path = os.path.join(library_dir, f"{jid}.mp4")
        done = r.get("status") == "done" and os.path.exists(path)
        take = {"id": jid, "created": r.get("created"), "status": r.get("status"),
                "progress": r.get("progress", 0), "max": r.get("max", 0),
                "media_url": f"/api/media/{jid}" if done else None,
                "seconds": _duration(path) if done else None,
                "params": {k: prm.get(k) for k in TAKE_PARAM_KEYS if k in prm},
                "check": cached_check(library_dir, jid) if done else None}
        out.setdefault(int(idx), []).append(take)
    for k in out:
        out[k].sort(key=lambda t: -(t.get("created") or 0))
    return out


# ---------- the floor check (port of .mvwork/g1_refs/envelope_check.py) ----------

def _pcm(path, start=0.0, dur=None, sr=16000):
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{start:.6f}"]
    if dur:
        cmd += ["-t", f"{dur:.6f}"]
    cmd += ["-i", path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"]
    out = subprocess.run(cmd, capture_output=True, check=True, timeout=120).stdout
    return np.frombuffer(out, dtype=np.float32), sr


def _envelope(x, sr, hop_hz=100):
    hop = sr // hop_hz
    n = len(x) // hop
    if n < 2:
        return np.zeros(2, dtype=np.float32)
    e = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-10)
    e = np.log10(e)
    return (e - e.mean()) / (e.std() + 1e-9)


def envelope_r(clip, song, song_start):
    """Pearson r of the clip's log-RMS envelope against the song window at the best lag within
    +-0.6 s. Clean song-locked takes measured 0.87-1.00; drifted 0.36-0.40."""
    clip_dur = _duration(clip) or 0.0
    c, sr = _pcm(clip)
    s, _ = _pcm(song, start=float(song_start), dur=clip_dur)
    ec, es = _envelope(c, sr), _envelope(s, sr)
    n = min(len(ec), len(es))
    ec, es = ec[:n], es[:n]
    best = (-2.0, 0)
    if n < 120:
        return None, None, clip_dur
    for lag in range(-60, 61):
        if lag >= 0:
            a, b = ec[lag:], es[: n - lag]
        else:
            a, b = ec[: n + lag], es[-lag:]
        if len(a) < 10:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if r > best[0]:
            best = (r, lag)
    return round(best[0], 3), round(best[1] / 100, 2), clip_dur


FLOOR_CLEAN = 0.87


def run_check(library_dir, job_id, song_path, song_start, planned_seconds=None):
    """Floor check + opening_jump + contact sheet for one take; cached. Returns the check dict."""
    clip = os.path.join(library_dir, f"{job_id}.mp4")
    if not os.path.exists(clip):
        raise FileNotFoundError(clip)
    out_dir = os.path.join(checks_dir(library_dir), job_id)
    os.makedirs(out_dir, exist_ok=True)
    r, lag, clip_dur = envelope_r(clip, song_path, song_start) if song_path and os.path.exists(song_path) else (None, None, _duration(clip))
    ratio, first = mv_take_check.opening_jump(clip)
    sheet_png = os.path.join(out_dir, "sheet.png")
    mv_take_check.sheet(clip, sheet_png)
    check = {"job_id": job_id, "checked": time.time(), "seconds": clip_dur, "planned_seconds": planned_seconds,
             "song_start": song_start,
             "floor_r": r, "floor_lag": lag, "floor_ok": (r is not None and r >= FLOOR_CLEAN),
             "opening_jump": None if ratio is None else round(ratio, 2),
             "opening_flag": bool(ratio is not None and ratio >= mv_take_check.JUMP_FLAG),
             "sheet_url": f"/api/mv/take_sheet/{job_id}"}
    with open(os.path.join(out_dir, "check.json"), "w") as fh:
        json.dump(check, fh, indent=1)
    return check
