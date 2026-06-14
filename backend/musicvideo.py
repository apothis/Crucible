"""Music-video pipeline (Phase A): turn a song into an editable shot list via the LLM.

A "script" = an ordered list of shots, each timed to a song section, with a photoreal scene
prompt, a motion cue, which characters appear, and whether it is a lip-sync performance shot.
The user edits it, then each shot is rendered with the existing video builders (still -> i2v
or S2V) anchored on each character's reference still. Plain ASCII.
"""
import json
import os
import re
import shutil
import subprocess
import tempfile

from . import llm as llm_mod


def _ffmpeg():
    return shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"


def assemble(segments, audio_path, out_path, width=1280, height=720, fps=24):
    """Stitch shot clips into one MP4 (GPU-free, ffmpeg). segments = [{path, dur}]: each clip
    is scaled+padded to width x height, set to a common fps, and fitted to exactly `dur`
    seconds (long clips trimmed; short clips hold their last frame). Concatenated in order,
    then the full song audio is muxed over the result (replacing per-clip audio)."""
    ff = _ffmpeg()
    work = tempfile.mkdtemp(prefix="mvasm_")
    try:
        norm = []
        for i, seg in enumerate(segments):
            o = os.path.join(work, f"seg{i:03d}.mp4")
            vf = (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                  f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},"
                  f"tpad=stop_mode=clone:stop_duration=3600")
            subprocess.run([ff, "-y", "-loglevel", "error", "-i", seg["path"], "-vf", vf,
                            "-t", f"{max(0.1, seg['dur']):.3f}", "-an", "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-r", str(fps), o], check=True)
            norm.append(o)
        listf = os.path.join(work, "list.txt")
        with open(listf, "w") as f:
            for o in norm:
                f.write("file '%s'\n" % o)
        concat = os.path.join(work, "concat.mp4")
        subprocess.run([ff, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                        "-i", listf, "-c", "copy", concat], check=True)
        total = sum(max(0.1, s["dur"]) for s in segments)
        if audio_path:
            subprocess.run([ff, "-y", "-loglevel", "error", "-i", concat, "-i", audio_path,
                            "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
                            "-t", f"{total:.3f}", out_path], check=True)
        else:
            shutil.move(concat, out_path)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return out_path

SHOT_TYPES = ("performance", "narrative", "broll")


def _song_summary(song):
    """Human-readable song brief for the LLM + total duration (sec). Lyrics are shown line by
    line under each section's time window so shots can be anchored to the words sung then."""
    title = song.get("title") or "Untitled"
    lines = [f"Title: {title}",
             f"Genre/mood tags: {song.get('tags') or ''}"]
    if song.get("bpm"):
        lines.append(f"BPM: {song['bpm']}")
    if song.get("keyscale"):
        lines.append(f"Key: {song['keyscale']}")
    secs = song.get("sections") or []
    t = 0
    body = []
    for s in secs:
        dur = int(s.get("seconds") or 0)
        lyr = (s.get("lyrics") or "").strip()
        head = f"  [{t}-{t + dur}s] {s.get('type') or 'section'}:"
        if lyr:
            body.append(head)
            for ln in lyr.splitlines():
                ln = ln.strip()
                if ln:
                    body.append(f'      "{ln}"')
        else:
            body.append(head + " (instrumental - no lyrics)")
        t += dur
    lines.append(f"Approx duration: {t}s across {len(secs)} sections. Lyrics with their time windows:")
    lines.extend(body)
    return "\n".join(lines), t


def build_prompt(song, cast, n_shots):
    summary, total = _song_summary(song)
    title = song.get("title") or "Untitled"
    cast_txt = "\n".join(f"  - {c.get('name')} ({c.get('role', 'character')})"
                         for c in cast if c.get("name")) or "  (no fixed cast - lean scenic/atmospheric)"
    target = n_shots or max(12, min(30, round((total or 180) / 8)))   # ~1 shot / 8s
    system = ("You are a music video director. Output a SHOT LIST as STRICT JSON ONLY (no prose, no "
              "markdown fences). Direct the video from TWO anchors: (1) the SONG TITLE as the "
              "overarching visual concept and story, and (2) the LYRICS sung in each shot's time "
              "window - every shot should literally depict or evoke the exact words sung then. Cut "
              "on the beat and keep named characters visually consistent.")
    prompt = f"""{summary}

DIRECTION (follow both):
- The song is titled "{title}". Make the TITLE the central visual theme of the WHOLE video: the
  recurring imagery, setting, world, and story should embody what the title means.
- For every shot, read the lyric lines in that shot's time window above and make the scene VISUALLY
  ILLUSTRATE those exact words - their imagery, story beat, or emotion. (e.g. a shot over a line
  about "blades of lightning in our hands" should literally show that, not a generic stage.)
  Instrumental windows have no lyrics: there, advance the title's story or show atmosphere.

Characters (keep each visually consistent wherever they appear):
{cast_txt}

Write about {target} shots covering the whole song IN ORDER, 0 to {total}s with no gaps.
Return ONLY a JSON array. Each element is an object:
{{"section": "<section type>", "start": <sec int>, "end": <sec int>,
  "type": "performance" | "narrative" | "broll",
  "scene": "<SCENE description = the static look of the frame: setting/environment, subject, wardrobe, lighting, framing and lens. Photoreal, grounded in the lyrics at this time + the title theme>",
  "action": "<ACTION description = what happens over ~5s: what the subject DOES and how the camera moves>",
  "characters": [<names of any named characters present; [] if none>],
  "lipsync": <true ONLY for close performance shots where a named singer sings these lyrics>}}

Rules: performance/lipsync shots only on SUNG sections; scenic/broll on instrumental sections;
vary shot types and framing (close-up, wide, tracking); photoreal live-action metal video (not
animation). Every scene must connect to BOTH the title theme AND the specific lyrics in its window."""
    return system, prompt


def parse_shots(text):
    m = re.search(r"\[.*\]", text, re.S)
    raw = m.group(0) if m else text
    shots = json.loads(raw)
    out = []
    for i, s in enumerate(shots):
        if not isinstance(s, dict):
            continue
        t = s.get("type")
        out.append({
            "idx": i,
            "section": str(s.get("section") or ""),
            "start": int(float(s.get("start") or 0)),
            "end": int(float(s.get("end") or 0)),
            "type": t if t in SHOT_TYPES else "broll",
            "scene": str(s.get("scene") or "").strip(),
            "action": str(s.get("action") or s.get("motion") or "").strip(),
            "characters": [str(x) for x in (s.get("characters") or []) if x],
            "lipsync": bool(s.get("lipsync")),
        })
    return out


def generate_script(song, cast, provider, model, claude_model, n_shots=0):
    """Returns the parsed shot list (raises on LLM / JSON failure)."""
    system, prompt = build_prompt(song, cast, n_shots)
    text = llm_mod.complete(provider, model, system, prompt, claude_model)
    return parse_shots(text)
