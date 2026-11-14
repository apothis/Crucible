"""Read side of the Music Video tab (docs/MV_UI_DESIGN.md, phase 1): the saved scripts of a song,
the takes rendered for a scene, and the on-demand checks on a take (song-envelope floor +
opening_jump + contact sheet). GPU-free, Mac only. Plain ASCII.

Checks are cached under .mvwork/mv_take_checks/<job_id>/ (check.json + sheet.png) so the tab can
show them without re-measuring; a take never changes once rendered.
"""
import json
import os
import re
import subprocess
import time

import numpy as np

from . import mv_take_check


def scripts_dir(library_dir):
    return os.path.join(os.path.dirname(library_dir), ".mvwork", "mv_scripts")


def song_ref_stills(library_dir, audio_id, title, exists=None):
    """The singer's pictures a NEW script of a song starts with: ref_stills of the newest saved
    script of the same track, else of the newest one with the same title (another master of the
    song). `exists(id)` filters out pictures no longer in the library. Returns (ref_stills, name)
    or ({}, None). A fresh write used to start with none, so the tab fell back to its built-in
    default pictures from another song (user caught it, 2026-11-14)."""
    d = scripts_dir(library_dir)
    if not os.path.isdir(d):
        return {}, None
    key = re.sub(r"[^a-z0-9]+", " ", str(title or "").lower()).strip()
    by_track, by_title = [], []
    for fn in os.listdir(d):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(d, fn)
        try:
            with open(path) as fh:
                s = json.load(fh)
        except Exception:
            continue
        rs = s.get("ref_stills") if isinstance(s, dict) else None
        if not isinstance(rs, dict):
            continue
        rs = {k: v for k, v in rs.items() if v and (exists is None or exists(v))}
        if not rs:
            continue
        row = (os.path.getmtime(path), fn, rs)
        if audio_id and s.get("audio_id") == audio_id:
            by_track.append(row)
        elif key and re.sub(r"[^a-z0-9]+", " ", str(s.get("title") or "").lower()).strip() == key:
            by_title.append(row)
    for rows in (by_track, by_title):
        if rows:
            _, fn, rs = max(rows)
            return dict(rs), fn
    return {}, None


def checks_dir(library_dir):
    return os.path.join(os.path.dirname(library_dir), ".mvwork", "mv_take_checks")


def list_scripts(library_dir, audio_id=None):
    """Saved scripts (newest first) with the counts the tab shows: scenes, clips, stills filed,
    takes locked, and the full videos cut from each (videos, last_video = the newest cut's stamp;
    revisions of one song share a title, so this is what tells the working script apart).
    `audio_id` narrows to one song."""
    d = scripts_dir(library_dir)
    out = []
    if not os.path.isdir(d):
        return out
    cuts = {}
    ad = assembly_dir(library_dir)
    for stamp in (os.listdir(ad) if os.path.isdir(ad) else []):
        rp = os.path.join(ad, stamp, "assembly.json")
        try:
            with open(rp) as fh:
                rec = json.load(fh)
        except Exception:
            continue
        name = os.path.basename(str((rec.get("body") or {}).get("script") or ""))
        if name and not (rec.get("result") or {}).get("partial") and not (rec.get("body") or {}).get("partial"):
            cuts.setdefault(name, []).append(stamp)
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
                    "problems": sum(len(sc.get("scene_problems") or []) for sc in scenes),
                    "videos": len(cuts.get(fn, [])), "last_video": max(cuts[fn]) if cuts.get(fn) else None})
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
                   "context_frames", "location_ref_id", "upscale_width", "upscale_height", "upscale_refiner",
                   "end_still_id", "rail_frames", "variant", "engine", "meridian_of", "move2d_of", "move", "amount",
                   "steps", "sampler", "scheduler")


def lineage_base(script_name):
    """Scripts chain by appending suffixes (_rw3_<stamp>, _sp0_<stamp>, _mc1_<stamp>, _facts_<stamp>),
    so every ancestor's name is a prefix of the current one. The shared root is the part before
    the first timestamp."""
    return re.split(r"_\d{8}_\d{6}", script_name, 1)[0]


def is_ancestor(name, current, ancestors=None):
    """`name` is `current` or one of its ancestors: by the suffix chain, or (once a long chain was
    shortened, see child_name) by the script's own recorded ancestor list."""
    strip = lambda x: x[:-5] if x.endswith(".json") else x
    a, c = strip(name), strip(current)
    return a == c or c.startswith(a + "_") or any(x == a or x.startswith(a + "_") for x in map(strip, ancestors or []))


MAX_NAME = 160     # a child name longer than this restarts from the first script's name (255 is the FS limit)


def child_name(parent, tag, stamp, script=None):
    """The basename (no extension) of a script derived from `parent` by an edit `tag` (rw3, sp0,
    mc1, facts): parent + _tag_stamp, so ancestors stay prefixes. A chain too long for the file
    system restarts from the FIRST script's name (root + its timestamp) plus a short hash of the
    parent, so that script's takes still match by prefix; `script` (the child) records every
    ancestor in script['ancestors'] so the takes of the shortened-away ones still count."""
    import hashlib
    base = re.sub(r"\.json$", "", os.path.basename(str(parent)))
    if script is not None:
        script["ancestors"] = list(dict.fromkeys(list(script.get("ancestors") or []) + [base]))
    name = f"{base}_{tag}_{stamp}"
    if len(name) > MAX_NAME:
        m = re.match(r"(.*?_\d{8}_\d{6})", base)
        origin = m.group(1) if m else base[:60]
        name = f"{origin}_x{hashlib.md5(base.encode()).hexdigest()[:6]}_{tag}_{stamp}"
    return name


def takes_for_script(rows, script_name, library_dir, scene=None, scenes=None, ancestors=None):
    """Library rows (sqlite dicts with params JSON) -> {scene_index: [take...]} for one script.
    Running jobs are passed in the same shape with status != done. With `scenes` (the script's
    scene list) a take rendered under an ANCESTOR script counts too when its song window equals
    the scene's (the SSBS cut of 2026-10-30 was assembled from takes across six script
    generations); such a take carries from_script = the name it was rendered under."""
    out = {}
    windows, locked_ids, after = {}, {}, {}
    if scenes:
        for sc in scenes:
            if sc.get("takes_after"):
                after[int(sc["index"])] = float(sc["takes_after"])
            windows[int(sc["index"])] = (round(float(sc["start"]), 2), round(float(sc["end"]), 2))
            if sc.get("locked") and sc["locked"].get("job_id"):
                locked_ids[sc["locked"]["job_id"]] = int(sc["index"])
    for r in rows:
        try:
            prm = r["params"] if isinstance(r["params"], dict) else json.loads(r["params"])
        except Exception:
            continue
        rendered_under = os.path.basename(str(prm.get("script") or ""))
        idx = prm.get("scene_index")
        if idx is None:
            continue
        from_script = None
        if r["id"] in locked_ids:
            # the scene's locked take: listed under its scene whatever it was rendered as (a lock
            # may trim the take's window, so the window match below would miss it)
            idx = locked_ids[r["id"]]
            from_script = rendered_under if rendered_under != script_name else None
        elif rendered_under != script_name:
            if not windows or not is_ancestor(rendered_under, script_name, ancestors):
                continue
            # the ancestor's scene index may differ (splits renumber): match by the song window
            try:
                win = (round(float(prm.get("scene_start")), 2), round(float(prm.get("scene_end")), 2))
            except (TypeError, ValueError):
                continue
            hits = [i for i, w in windows.items() if w == win]
            if len(hits) != 1:
                continue
            idx, from_script = hits[0], rendered_under
        if scene is not None and int(idx) != int(scene):
            continue
        jid = r["id"]
        path = os.path.join(library_dir, f"{jid}.mp4")
        done = r.get("status") == "done" and os.path.exists(path)
        take = {"id": jid, "created": r.get("created"), "status": r.get("status"),
                "progress": r.get("progress", 0), "max": r.get("max", 0),
                "media_url": f"/api/media/{jid}" if done else None,
                "seconds": _duration(path) if done else None,
                "params": {k: prm.get(k) for k in TAKE_PARAM_KEYS if k in prm},
                "from_script": from_script,
                # rendered before the scene was rewritten or marked for a re-render: an old shot
                # (still listed, but the scene counts as unrendered for a batch)
                "stale": bool(after.get(int(idx)) and float(r.get("created") or 0) < after[int(idx)]),
                "check": cached_check(library_dir, jid) if done else None}
        out.setdefault(int(idx), []).append(take)
    for k in out:
        out[k].sort(key=lambda t: -(t.get("created") or 0))
    return out


def _engine(prm):
    """Upscales queued before SeedVR2 came back carry flashvsr_plus or no engine: all FlashVSR+.
    open_h3 = the Open-H3-Regenerate-2K refiner run on a finished take (mv_h3.build_h3_regen);
    rtx_vsr = NVIDIA RTX Video Super Resolution (video.build_rtx_vsr_upscale)."""
    e = prm.get("engine")
    return e if e in ("seedvr2", "open_h3", "rtx_vsr") else "flashvsr_plus"


def attach_upscales(takes, rows, library_dir):
    """Link each take to its upscales (library or running jobs whose params carry upscale_of =
    the take id), per engine: take["upscales"] = {engine: the newest finished one, else the
    running or failed one}, so the tab can show progress or the error; take["upscaled"] = the
    FlashVSR+ one (what the tab showed before SeedVR2). Each upscale carries its settings
    (temporal_compression 1 = the FlashVSR+ "fast motion" run) and, when a newer job of the same
    engine is still queued or running behind a finished one, that job as newer. Mutates and returns takes."""
    by_take = {}
    for r in rows:
        try:
            prm = r["params"] if isinstance(r["params"], dict) else json.loads(r["params"])
        except Exception:
            continue
        src = prm.get("upscale_of")
        if not src:
            continue
        path = os.path.join(library_dir, f"{r['id']}.mp4")
        done = r.get("status") == "done" and os.path.exists(path)
        if r.get("status") == "done" and not done:
            continue
        by_take.setdefault(src, []).append({
            "id": r["id"], "created": r.get("created") or 0, "status": r.get("status"),
            "progress": r.get("progress", 0), "max": r.get("max", 0), "error": r.get("error"),
            "media_url": f"/api/media/{r['id']}" if done else None,
            "engine": _engine(prm), "scale": prm.get("scale"), "resolution": prm.get("resolution"),
            "temporal_compression": prm.get("temporal_compression"), "mode": prm.get("mode"),
            "quality": prm.get("quality"),
            "fps": prm.get("fps_out"), "fps_fixed": prm.get("fps_fixed")})
    for lst in takes.values():
        for t in lst:
            ups = sorted(by_take.get(t["id"], []), key=lambda u: -u["created"])
            t["upscales"] = {}
            for eng in ("flashvsr_plus", "seedvr2", "open_h3", "rtx_vsr"):
                mine = [u for u in ups if u["engine"] == eng]
                pick = next((u for u in mine if u["status"] == "done"), None) or (mine[0] if mine else None)
                if pick:
                    # a newer job of the same engine still queued or running behind a finished one (a
                    # re-run, e.g. FlashVSR+ fast motion over a plain 2x): the tab shows its progress
                    newer = mine[0] if mine and mine[0] is not pick and mine[0]["status"] not in ("done", "error") else None
                    t["upscales"][eng] = dict(pick, newer=newer) if newer else pick
            t["upscaled"] = t["upscales"].get("flashvsr_plus")
    return takes


def upscale_path(rows, take_id, library_dir, engine="flashvsr_plus"):
    """The newest finished upscale file of a take from one engine, or None (rows as for attach_upscales)."""
    best = None
    for r in rows:
        try:
            prm = r["params"] if isinstance(r["params"], dict) else json.loads(r["params"])
        except Exception:
            continue
        if prm.get("upscale_of") != take_id or r.get("status") != "done" or _engine(prm) != engine:
            continue
        path = os.path.join(library_dir, f"{r['id']}.mp4")
        if os.path.exists(path) and os.path.getsize(path) > 0 and (best is None or (r.get("created") or 0) > best[0]):
            best = (r.get("created") or 0, r["id"], path)
    return (best[1], best[2]) if best else (None, None)


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


# ---------- assemblies ----------

def assembly_dir(library_dir):
    return os.path.join(os.path.dirname(library_dir), ".mvwork", "mv_assembly")


def record_assembly(out_dir, body, result):
    """Sidecar next to a cut: the request (which take per scene) and the result, so the tab can
    list earlier cuts and reuse their take maps."""
    safe = {k: v for k, v in (body or {}).items() if k not in ("async",)}
    with open(os.path.join(out_dir, "assembly.json"), "w") as fh:
        json.dump({"body": safe, "result": result, "recorded": time.time()}, fh, indent=1)


def list_assemblies(library_dir, script_name=None):
    """Cuts under .mvwork/mv_assembly, newest first: stamp, mp4s, and (when recorded) the script,
    the take map and the result. script_name narrows to the script's lineage."""
    d = assembly_dir(library_dir)
    out = []
    if not os.path.isdir(d):
        return out
    base = lineage_base(script_name) if script_name else None
    for stamp in sorted(os.listdir(d), reverse=True):
        sd = os.path.join(d, stamp)
        if not os.path.isdir(sd):
            continue
        rec = None
        rp = os.path.join(sd, "assembly.json")
        if os.path.exists(rp):
            try:
                with open(rp) as fh:
                    rec = json.load(fh)
            except Exception:
                rec = None
        mp4s = [f for f in os.listdir(sd) if f.endswith(".mp4") and not f.startswith("scene_") and f != "video_only.mp4"]
        if not mp4s:
            continue
        rec_script = os.path.basename(str(((rec or {}).get("body") or {}).get("script") or ""))
        if base and rec_script and lineage_base(rec_script) != base:
            continue
        if base and not rec_script:
            continue          # an unrecorded cut cannot be tied to a song
        # the finished file (ident + end card) when there is one: at 1440p the plain cut beside it
        # is the bigger file, so size alone picked the wrong one
        finals = [f for f in mp4s if f.endswith("_final.mp4")]
        final = finals[0] if finals else max(mp4s, key=lambda f: os.path.getsize(os.path.join(sd, f)))
        res = (rec or {}).get("result") or {}
        out.append({"stamp": stamp, "dir": sd, "file": final, "path": os.path.join(sd, final),
                    "seconds": res.get("seconds") or _duration(os.path.join(sd, final)),
                    "scenes": res.get("scenes"), "partial": bool(res.get("partial")),
                    "script": rec_script or None,
                    "jobs": {str(k): v for k, v in (res.get("jobs") or ((rec or {}).get("body") or {}).get("jobs") or {}).items()},
                    "intro": bool(res.get("intro")), "end_card": res.get("end_card"),
                    "library_id": res.get("library_id"), "media_url": res.get("media_url"),
                    "mtime": os.path.getmtime(os.path.join(sd, final))})
    return out
