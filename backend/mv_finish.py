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


def _video_rate(path):
    """(avg fps, video packet count, stream seconds) of the first video stream, or Nones."""
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                              "-show_entries", "stream=avg_frame_rate,nb_read_packets,duration", "-of", "json", path],
                             capture_output=True, text=True, timeout=60).stdout
        st = (json.loads(out).get("streams") or [{}])[0]
        num, den = (str(st.get("avg_frame_rate") or "0/1").split("/") + ["1"])[:2]
        fps = float(num) / float(den) if float(den) else None
        n = int(st["nb_read_packets"]) if str(st.get("nb_read_packets") or "").isdigit() else None
        secs = float(st["duration"]) if st.get("duration") not in (None, "N/A") else None
        return fps, n, secs
    except Exception:
        return None, None, None


def fix_fps(path, expect, tol=0.01):
    """Make an upscaled clip play at the source's rate without touching a frame. The stock
    FlashVSR+ VIDEO node writes a 24 fps input at 24.4167 fps (it reads the rate off a temp .mkv,
    which keeps 1 ms timestamps); the box copy is patched, but a node update would bring the bug
    back, so every downloaded upscale is checked here. A wrong rate is fixed by rescaling the video
    timestamps (-itsscale, stream copy: all frames kept; a raw .h264 remux with -r dropped two and
    shifted the picture), with the audio taken unscaled from the same file. Returns
    {fps_in, fps, frames, fixed}."""
    fps_in, frames, secs = _video_rate(path)
    if not fps_in or not expect or abs(fps_in - float(expect)) <= tol:
        return {"fps_in": fps_in, "fps": fps_in, "frames": frames, "fixed": False}
    if not frames or not secs:
        raise RuntimeError(f"fps fix: cannot read the frame count or length of {os.path.basename(path)}")
    tmp = path + ".fps.mp4"
    # scale so the video lasts exactly frames / expect seconds
    factor = frames / (float(expect) * secs)
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-itsscale", f"{factor:.9f}",
                        "-i", path, "-i", path, "-map", "0:v:0", "-map", "1:a:0?", "-c", "copy",
                        "-movflags", "+faststart", tmp], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"fps fix failed: {r.stderr[-300:]}")
    fps_out, frames_out, _ = _video_rate(tmp)
    if not fps_out or abs(fps_out - float(expect)) > tol or (frames and frames_out != frames):
        os.remove(tmp)
        raise RuntimeError(f"fps fix did not hold: {fps_out} fps, {frames_out} of {frames} frames")
    os.replace(tmp, path)
    return {"fps_in": round(fps_in, 4), "fps": round(fps_out, 4), "frames": frames_out, "fixed": True}


def source_audio(path, source):
    """Put the source clip's own audio back on an upscale, both streams copied. FlashVSR+
    re-encodes the audio it keeps: on SSBS scene 6 it came back 21 ms longer and its envelope sat
    +0.02 s late against the song (r 0.975 vs the take's 0.998 at 0). The take's audio is the
    song-locked one. A source without audio leaves the file as it is. Returns True if replaced."""
    try:
        has = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=index",
                              "-of", "csv=p=0", source], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        has = ""
    if not has:
        return False
    tmp = path + ".aud.mp4"
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", path, "-i", source, "-map", "0:v:0",
                        "-map", "1:a:0", "-c", "copy", "-movflags", "+faststart", tmp], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"audio swap failed: {r.stderr[-300:]}")
    os.replace(tmp, path)
    return True


def trim_with_audio(path, source, frames, crf=16):
    """A Meridian re-shoot (s.20.56) comes back at Meridian's own length with no audio: cut it to
    the source take's `frames` (frame-accurate, re-encoded) and lay the source take's audio
    under it (the take's audio is the song-locked one). Returns the frame count written."""
    tmp = path + ".trim.mp4"
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", path]
    has_audio = False
    if source:
        try:
            has_audio = bool(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                                             "stream=index", "-of", "csv=p=0", source],
                                            capture_output=True, text=True, timeout=30).stdout.strip())
        except Exception:
            has_audio = False
    if has_audio:
        cmd += ["-i", source, "-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy"]
    else:
        cmd += ["-map", "0:v:0"]
    cmd += ["-frames:v", str(int(frames)), "-c:v", "libx264", "-crf", str(crf), "-preset", "slow",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"trim failed: {r.stderr[-300:]}")
    os.replace(tmp, path)
    return int(frames)


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
