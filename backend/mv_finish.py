"""Finishing a music-video cut (docs/MV_UI_DESIGN.md, phase 2): the channel ident in front and
the YouTube end-card plate over the last seconds. Mac only, ffmpeg + the Remotion ident; no GPU.
Plain ASCII.

- ident_render(): `npx remotion render` of the Ident composition in tools/loosh_remotion with the
  song title as a prop, cached under .mvwork/ident/cache/<hash>.mp4 (the hash covers the props
  and the composition source, so an edited Ident.tsx re-renders, a repeated title does not).
- finish_cut(): one ffmpeg pass: [ident] + [cut with the end plate overlaid for t >= dur - secs]
  -> final mp4. The cut's own audio (the song, already muxed by assemble) is kept; the ident's
  audio plays before it. Everything is re-encoded once at the output size.
"""
import hashlib
import json
import os
import shutil
import subprocess

FPS = 24


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _npx():
    return shutil.which("npx") or ("/opt/homebrew/bin/npx" if os.path.exists("/opt/homebrew/bin/npx") else "npx")


def ident_props(title, sub=None, edge=None, audio="audio/ident_sound.wav"):
    return {"title": str(title or "").strip(), "sub": str(sub if sub is not None else "Official Video"),
            "audio": audio, "edge": str(edge or "none")}


def ident_cache_dir(work_dir):
    return os.path.join(work_dir, "ident", "cache")


def ident_render(work_dir, title, sub=None, edge=None, timeout=600):
    """Render (or reuse) the ident for one song title. Returns {path, hash, cached, seconds}."""
    root = _repo_root()
    proj = os.path.join(root, "tools", "loosh_remotion")
    src = os.path.join(proj, "src", "Ident.tsx")
    props = ident_props(title, sub, edge)
    with open(src, "rb") as fh:
        src_hash = hashlib.sha1(fh.read()).hexdigest()[:8]
    key = hashlib.sha1((json.dumps(props, sort_keys=True) + src_hash).encode()).hexdigest()[:16]
    out_dir = ident_cache_dir(work_dir)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{key}.mp4")
    cached = os.path.exists(out) and os.path.getsize(out) > 0
    if not cached:
        env = dict(os.environ)
        env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + env.get("PATH", "")
        cmd = [_npx(), "remotion", "render", "src/index.ts", "Ident", out, "--codec", "h264", "--crf", "16",
               f"--props={json.dumps(props)}"]
        r = subprocess.run(cmd, cwd=proj, capture_output=True, text=True, timeout=timeout, env=env)
        if r.returncode != 0 or not os.path.exists(out):
            raise RuntimeError(f"ident render failed: {(r.stderr or r.stdout)[-600:]}")
    return {"path": out, "hash": key, "cached": cached, "seconds": _duration(out), "props": props}


def _duration(path):
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return round(float(out), 3)
    except Exception:
        return None


def finish_cut(cut_path, out_path, width, height, ident_path=None, endcard_plate=None, endcard_secs=0.0,
               crf=18, preset="medium"):
    """[ident +] cut [+ end plate over the last endcard_secs] -> out_path (h264 yuv420p, aac 320k,
    faststart). Returns {seconds, intro_seconds, endcard_at}."""
    cut_dur = _duration(cut_path) or 0.0
    inputs, fc = [], []
    n = 0
    if ident_path:
        inputs += ["-i", ident_path]
        fc.append(f"[{n}:v]scale={width}:{height}:flags=lanczos,fps={FPS},format=yuv420p,setsar=1[v{n}]")
        fc.append(f"[{n}:a]aformat=sample_rates=48000:channel_layouts=stereo[a{n}]")
        n += 1
    inputs += ["-i", cut_path]
    cut_i = n
    n += 1
    ec_at = None
    if endcard_plate and endcard_secs and cut_dur > float(endcard_secs) + 5:
        inputs += ["-i", endcard_plate]
        ec_at = max(0.0, cut_dur - float(endcard_secs))
        fc.append(f"[{cut_i}:v]scale={width}:{height}:flags=lanczos,fps={FPS},format=yuv420p,setsar=1[c0]")
        fc.append(f"[c0][{n}:v]overlay=0:0:enable='gte(t,{ec_at:.3f})',format=yuv420p[v{cut_i}]")
        n += 1
    else:
        fc.append(f"[{cut_i}:v]scale={width}:{height}:flags=lanczos,fps={FPS},format=yuv420p,setsar=1[v{cut_i}]")
    fc.append(f"[{cut_i}:a]aformat=sample_rates=48000:channel_layouts=stereo[a{cut_i}]")
    if ident_path:
        fc.append(f"[v0][a0][v{cut_i}][a{cut_i}]concat=n=2:v=1:a=1[v][a]")
    else:
        fc.append(f"[v{cut_i}]null[v]")
        fc.append(f"[a{cut_i}]anull[a]")
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(fc),
           "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
           "-c:a", "aac", "-b:a", "320k", "-movflags", "+faststart", out_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"finish failed: {r.stderr[-600:]}")
    return {"seconds": _duration(out_path), "intro_seconds": _duration(ident_path) if ident_path else 0.0,
            "endcard_at": ec_at}
