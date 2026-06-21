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


# Color-grade "looks": ffmpeg filter chains applied identically to every segment at assemble
# time. One grade across all clips is what unifies the separately-generated shots into a single
# consistent feel. GPU-free; opt-in (default "none"). Add LUT-based looks later via lut3d.
GRADES = {
    "none": "",
    "cold_gothic": ("eq=saturation=0.72:contrast=1.12,"
                    "curves=all='0/0 0.25/0.18 1/0.96',colorbalance=rs=-0.05:bs=0.07:bm=0.05"),
    "teal_orange": ("colorbalance=rs=-0.08:bs=0.08:rh=0.10:bh=-0.08,"
                    "eq=contrast=1.10:saturation=1.06"),
    "warm_film": ("colortemperature=temperature=8200,curves=all='0/0.06 1/0.97',"
                  "eq=saturation=0.92:contrast=1.05"),
    "noir": ("eq=saturation=0.12:contrast=1.35:brightness=-0.02,"
             "curves=all='0/0 0.2/0.12 0.8/0.9 1/1'"),
    "bleach_bypass": "eq=saturation=0.42:contrast=1.34:brightness=0.03,curves=all='0/0.02 1/0.98'",
    "vibrant": "eq=saturation=1.18:contrast=1.08,vibrance=intensity=0.25",
    # Film / director / art-style inspired looks (all verified to render distinctly on footage).
    "fincher_thriller": ("curves=all='0/0.02 0.5/0.45 1/0.95',eq=saturation=0.62:contrast=1.18,"
                         "colorbalance=rs=-0.06:gs=0.04:bs=0.05:gm=0.03:gh=-0.03"),
    "wes_anderson": ("eq=saturation=1.05:contrast=0.90:brightness=0.05,"
                     "colorbalance=rm=0.06:gm=0.04:rh=0.05:bh=-0.05,colortemperature=temperature=7400"),
    "blade_runner": ("colorbalance=rs=-0.10:bs=0.12:rh=0.10:gh=-0.04:bh=0.04,"
                     "eq=saturation=1.15:contrast=1.12"),
    "dune_amber": ("colorbalance=rm=0.12:gm=0.04:bm=-0.08:rh=0.12:bh=-0.10,"
                   "eq=saturation=0.80:contrast=1.10,colortemperature=temperature=9200"),
    "matrix_green": ("colorbalance=gm=0.10:gs=0.06:gh=0.08:rm=-0.03:bm=-0.03,"
                     "eq=saturation=0.85:contrast=1.10"),
    "kodachrome_70s": "curves=preset=vintage,eq=saturation=0.95:contrast=1.04",
    "technicolor": "eq=saturation=1.40:contrast=1.15,vibrance=intensity=0.30",
    "moonlight_teal": ("colorbalance=bs=0.09:gs=0.03:rh=0.05:bh=-0.02,"
                       "eq=saturation=1.10:contrast=1.08"),
    "sepia": ("colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131,"
              "eq=contrast=1.05"),
    "fury_road": ("colorbalance=rs=-0.12:bs=0.12:rh=0.15:bh=-0.12,"
                  "eq=saturation=1.30:contrast=1.20"),
    "cross_process": "curves=preset=cross_process,eq=saturation=1.10",
    "amelie_gold": ("colorbalance=rm=0.05:gm=0.06:bm=-0.05:gh=0.04:bh=-0.06,"
                    "eq=saturation=1.12:contrast=1.05,colortemperature=temperature=8000"),
}


def grade_names():
    """The available grade looks, 'none' first (for the UI picker)."""
    return ["none"] + [k for k in GRADES if k != "none"]


def assemble(segments, audio_path, out_path, width=1280, height=720, fps=24, grade="none",
             transition=0.0, intro=None):
    """Stitch shot clips into one MP4 (GPU-free, ffmpeg). segments = [{path, dur}]: each clip
    is scaled+padded to width x height, set to a common fps, and fitted to exactly `dur`
    seconds (long clips trimmed; short clips hold their last frame). Concatenated in order,
    then the full song audio is muxed over the result (replacing per-clip audio). `grade` =
    a key in GRADES, applied identically to every segment for a consistent look. `transition`
    (seconds, 0 = hard cut) = crossfade duration blended between consecutive clips (ffmpeg
    xfade); opt-in, default 0 keeps the original hard-cut concat path verbatim."""
    ff = _ffmpeg()
    grade_chain = GRADES.get(grade or "none", "")
    work = tempfile.mkdtemp(prefix="mvasm_")
    try:
        durs = [max(0.1, s["dur"]) for s in segments]
        norm = []
        for i, seg in enumerate(segments):
            o = os.path.join(work, f"seg{i:03d}.mp4")
            vf = (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                  f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},"
                  f"tpad=stop_mode=clone:stop_duration=3600")
            if grade_chain:
                vf += "," + grade_chain
            subprocess.run([ff, "-y", "-loglevel", "error", "-i", seg["path"], "-vf", vf,
                            "-t", f"{durs[i]:.3f}", "-an", "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-r", str(fps), o], check=True)
            norm.append(o)
        concat = os.path.join(work, "concat.mp4")
        # crossfade chain (xfade) when a transition is requested AND there are >= 2 clips;
        # else the original lossless concat-demuxer path. xfade offset for each join = the
        # accumulated duration so far minus the overlap; total shrinks by `t` per join.
        t = float(transition or 0)
        if t > 0 and len(norm) > 1:
            t = max(0.05, min(t, min(durs) - 0.1))       # overlap must fit the shortest clip
            inputs = []
            for o in norm:
                inputs += ["-i", o]
            acc, label, parts = durs[0], "[0:v]", []
            for i in range(1, len(norm)):
                off = max(0.0, acc - t)
                out = f"[v{i}]"
                parts.append(f"{label}[{i}:v]xfade=transition=fade:duration={t:.3f}:offset={off:.3f}{out}")
                acc = acc + durs[i] - t
                label = out
            subprocess.run([ff, "-y", "-loglevel", "error"] + inputs +
                           ["-filter_complex", ";".join(parts), "-map", label,
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps), concat], check=True)
            total = acc
        else:
            listf = os.path.join(work, "list.txt")
            with open(listf, "w") as f:
                for o in norm:
                    f.write("file '%s'\n" % o)
            subprocess.run([ff, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                            "-i", listf, "-c", "copy", concat], check=True)
            total = sum(durs)
        # optional INTRO PRE-ROLL: a clip (e.g. the opening shot rendered with LTX-native wind audio)
        # plays first with the song SILENT, then the song enters and the intro's own audio crossfades
        # out. Lip-sync stays intact - the song still aligns to the body's t=0, just offset by `dur` in
        # the final. intro = {"path", "dur" (pre-roll seconds), "xfade" (intro->song crossfade seconds)}.
        if intro and intro.get("path") and os.path.isfile(intro["path"]):
            P = max(0.1, float(intro.get("dur") or 3))
            X = max(0.1, min(float(intro.get("xfade") or 1.5), P))
            # pre-roll VIDEO = the opening clip (its own audio is dropped; the wind comes from a
            # dedicated track if given, else the clip's audio).
            introf = os.path.join(work, "intro.mp4")
            ivf = (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                   f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},"
                   f"tpad=stop_mode=clone:stop_duration=3600")
            if grade_chain:
                ivf += "," + grade_chain
            subprocess.run([ff, "-y", "-loglevel", "error", "-i", intro["path"], "-vf", ivf,
                            "-t", f"{P:.3f}", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-r", str(fps), introf], check=True)
            full = os.path.join(work, "full.mp4")
            subprocess.run([ff, "-y", "-loglevel", "error", "-i", introf, "-i", concat,
                            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps), full], check=True)
            ftotal = P + total
            wind_src = intro["audio"] if (intro.get("audio") and os.path.isfile(intro["audio"])) else intro["path"]
            if audio_path:
                # level-match the wind to the SONG's opening so the crossfade has no jump (the user
                # asked for this) - measure both mean volumes, scale the wind by the difference.
                def _mean_db(path, dur):
                    try:
                        out = subprocess.run([ff, "-hide_banner", "-t", f"{dur:.2f}", "-i", path,
                                              "-af", "volumedetect", "-f", "null", "-"],
                                             capture_output=True, text=True).stderr
                        m = re.search(r"mean_volume:\s*(-?\d+\.?\d*) dB", out)
                        return float(m.group(1)) if m else -20.0
                    except Exception:
                        return -20.0
                gain = max(-30.0, min(6.0, _mean_db(audio_path, P + X) - _mean_db(wind_src, P)))
                af = (f"[1:a]atrim=0:{P:.3f},volume={gain:.1f}dB,afade=t=out:st={max(0.0, P - X):.3f}:d={X:.3f}[wind];"
                      f"[2:a]adelay={int(P * 1000)}|{int(P * 1000)},afade=t=in:st={P:.3f}:d={X:.3f}[song];"
                      f"[wind][song]amix=inputs=2:duration=longest:dropout_transition=0[a]")
                subprocess.run([ff, "-y", "-loglevel", "error", "-i", full, "-i", wind_src,
                                "-i", audio_path, "-filter_complex", af, "-map", "0:v", "-map", "[a]",
                                "-c:v", "copy", "-c:a", "aac", "-t", f"{ftotal:.3f}", out_path], check=True)
            else:
                shutil.move(full, out_path)
        elif audio_path:
            subprocess.run([ff, "-y", "-loglevel", "error", "-i", concat, "-i", audio_path,
                            "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
                            "-t", f"{total:.3f}", out_path], check=True)
        else:
            shutil.move(concat, out_path)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return out_path


def retime(in_path, out_path, speed, fps=24):
    """Speed up (or down) a clip, GPU-free. speed > 1 = faster/shorter (fixes uniform slow-motion
    by scaling the whole clip back to natural speed). Re-encodes at `fps`; drops any per-clip
    audio (the master is added at assemble)."""
    ff = _ffmpeg()
    speed = max(0.1, float(speed))
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", in_path,
                    "-vf", f"setpts=PTS/{speed:.4f},fps={fps}", "-an",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path], check=True)
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
    named = [c for c in cast if c.get("name")]
    musicians = [c for c in named if c.get("kind") != "actor"]   # default = musician/band
    actors = [c for c in named if c.get("kind") == "actor"]

    def _fmt(lst):
        return "\n".join(f"  - {c['name']} ({c.get('role') or 'character'})" for c in lst)
    cast_parts = []
    if musicians:
        cast_parts.append("Band / musicians (primarily PERFORMANCE shots playing/singing - the "
                          "lead singer lip-syncs; but they MAY ALSO act in NARRATIVE scenes, in "
                          "costume):\n" + _fmt(musicians))
    if actors:
        cast_parts.append("Actors (appear in NARRATIVE / story shots, NOT performing music):\n"
                          + _fmt(actors))
    cast_txt = "\n\n".join(cast_parts) or "  (no fixed cast - lean scenic/atmospheric)"
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
  "render": "msr" | "keyframe",
  "scene": "<SCENE = the look of the frame: setting/environment, subject, lighting, and FRAMING (close-up / medium / wide). Photoreal, grounded in the lyrics at this time + the title theme>",
  "action": "<what the SUBJECT does over the shot - performance, gesture, expression, movement. Describe the PERSON, not the camera>",
  "camera": "static" | "slow push-in" | "slow pull-back",
  "costume": "<what the named characters WEAR in this shot - lets the same person change outfits between scenes; '' if not notable or on-stage performance wear>",
  "characters": [<names of any named characters present; prefer ONE per shot; [] if none>],
  "lipsync": <true for ANY shot where a singer is singing the lyrics ON CAMERA - close OR wide, alone OR with the band, standing OR moving through the scene. This is a music video: if the lead singer is visible while words are being sung, her lips must sync, so set true. Set false ONLY when no one is singing on screen: instrumental sections, pure scenic B-roll with no people, or a shot of non-singing actors>,
  "segments": [<OPTIONAL per-shot timeline; each {{"seconds": <number>, "action": "<the subject's motion / gesture / expression during this slice>"}}; [] for one continuous action>]}}

DURATIONS (important): CHOOSE each shot's length to fit its content - do NOT make them all the same
length. Each shot must be between 2 and 20 seconds (a fast/energetic or punchy cut can be 2-4s; a held
performance, emotional, or atmospheric shot 8-20s). Set start/end so durations VARY with the pacing of
the music and lyrics. Never exceed 20s for a single shot. Cut more often through busy sections, hold
longer through sparse ones.

RENDER MODE (set "render" per shot - this picks the engine):
- "msr" is the DEFAULT - use it for almost everything. It animates a PERSON with prompt-driven motion and
  is the ONLY mode that can LIP-SYNC the singer and SYNC the band's playing to the music. Use "msr" for
  every performance / singing / instrument-playing shot, any shot where a character moves, gestures, or
  emotes in time with the music or lyrics, and any narrative shot with a person acting. If unsure, use "msr".
- "keyframe" is ONLY for pure B-ROLL / scenery / establishing shots that contain NO performer and NOTHING
  that must sync to the music, beat, or lyrics - e.g. a landscape, an empty set/stage, an object, weather,
  a slow scenic camera move over a still environment. It interpolates between fixed still frames, so it
  CANNOT lip-sync and CANNOT sync any motion to the music. NEVER use "keyframe" for a shot with a singing
  or playing band member, a dancer, or any music-/beat-synced action.
HARD RULE: if "lipsync" is true, or "type" is "performance", or any band member is playing/singing in the
shot, then "render" MUST be "msr". Only scenic/atmospheric shots with no people performing may be "keyframe".

CUT ON THE SONG STRUCTURE (important): the bracketed [start-end s] markers above are the song's actual
sections (intro, verse, chorus, bridge, etc.). START A NEW SHOT exactly at each section boundary - a single
shot must NEVER span a section change (e.g. never let one shot run from the verse into the chorus). At every
big musical transition - verse->chorus, a drop, the bridge, the final chorus - cut to a fresh shot and a
new look. Within a long section, place additional cuts on the lyric lines / phrase boundaries. Make each
shot's start and end line up with these section boundaries and lyric lines, not arbitrary times.

CAMERA (this engine is finicky - obey exactly): set "camera" to ONLY one of "static", "slow push-in"
(gentle zoom toward the subject) or "slow pull-back" (gentle zoom out). Those are the only moves that
render cleanly. NEVER call for pans, trucks, tracking, dolly left/right, orbits, handheld, or any
sideways/rotating motion - they either do nothing or warp the image. One move per shot; never combine
moves or jump framing mid-shot (it morphs the picture). Default most shots to "static"; use a slow
push-in for an emotional build, a slow pull-back for a reveal. Keep camera language OUT of scene/action.

CASTING & FRAMING (keeps identity stable - obey exactly): prefer ONE named character per shot - solo
shots hold a face and outfit, while crowded frames make secondary people drift and SWAP attributes (hair,
clothing, tattoos bleed between characters). Give each important character at least one MEDIUM or CLOSE
solo shot. Use multi-character frames sparingly: at most ONE wide "whole band/group" establishing shot in
the whole video, and never frame two strongly-contrasting looks (e.g. bare tattooed arms vs fully covered)
tightly together. Favor CLOSE and MEDIUM framings; lip-sync looks wrong and wides read empty when the
singer's face is small.

TIMELINE (per-shot micro-direction): a single shot can carry an ORDERED timeline of sub-actions via
"segments" - use it when the subject's action, expression, or energy CHANGES across the shot, or to hit
successive lyric lines within one held shot. Each segment is {{"seconds": <duration>, "action": "<the
subject's motion / gesture / expression during that slice>"}}. Keep the SCENE and FRAMING constant across
the shot (segments change only what the PERSON does, never the camera or setting). Use 2-4 segments, each
at least ~1.5s, roughly summing to the shot's length. Example for an 8s held close-up over a rising chorus
line: [{{"seconds":4,"action":"eyes down, swaying gently, holding back"}},{{"seconds":4,"action":"lifts her
head and belts the line, both arms rising"}}]. Leave "segments" empty ([]) for shots with one continuous
action. This is the single most powerful tool for making a shot feel alive - use it on most held shots.

Rules: performance/lipsync shots only on SUNG sections and feature the BAND/MUSICIANS (the lead
singer lip-syncs); NARRATIVE shots tell the title's story and feature the ACTORS; scenic/broll on
instrumental sections. LIP-SYNC RULE: during any SUNG section, EVERY shot that shows the lead singer -
wide, close, walking, or on stage with the band - must set "lipsync": true (she is singing those words
on camera). Do not reserve lip-sync for close-ups. Put the right people in each shot's "characters" by
name. Photoreal live-action
metal video (not animation). Every scene must connect to BOTH the title theme AND the specific lyrics
in its window."""
    return system, prompt


# The ONLY camera moves that render cleanly on our LTX-2.3 MSR pipeline (proven 2026-06-19): a locked
# frame, a gentle prompt-driven zoom in, or a gentle prompt-driven zoom out. Lateral/orbit/tracking moves
# do nothing or warp the frame, and the camera-control LoRAs morph when stacked on the MSR IC-LoRA - so the
# move is carried as PROMPT TEXT, never a LoRA. CAMERA_PHRASE maps each to the cue appended to the render
# prompt. See memory project_ltx-camera-lora.
CAMERA_MOVES = ("static", "slow push-in", "slow pull-back")
CAMERA_PHRASE = {
    "static": "the camera is locked off and still",
    "slow push-in": "the camera slowly and smoothly zooms in toward the subject",
    "slow pull-back": "the camera slowly and smoothly pulls back from the subject",
}


def _camera(v):
    v = str(v or "").strip().lower()
    return v if v in CAMERA_MOVES else "static"


def parse_shots(text):
    m = re.search(r"\[.*\]", text, re.S)
    raw = m.group(0) if m else text
    shots = json.loads(raw)
    out = []
    for i, s in enumerate(shots):
        if not isinstance(s, dict):
            continue
        t = s.get("type")
        # OPTIONAL per-shot timeline: ordered {seconds, action} slices -> Block.segs (relay timeline)
        segs = []
        for sg in (s.get("segments") or []):
            if not isinstance(sg, dict):
                continue
            act = str(sg.get("action") or sg.get("prompt") or "").strip()
            if not act:
                continue
            try:
                secs = float(sg.get("seconds") or sg.get("len") or 0)
            except (TypeError, ValueError):
                secs = 0.0
            segs.append({"seconds": round(secs, 2), "action": act})
        stype = t if t in SHOT_TYPES else "broll"
        lipsync = bool(s.get("lipsync"))
        chars = [str(x).strip() for x in (s.get("characters") or [])
                 if x and str(x).strip() not in ("[]", "none", "None", "-", "")]
        # RENDER MODE: msr (default, performer/music-synced) vs keyframe (pure scenic B-roll). Hard-enforce
        # that lip-sync / performance / character-present shots can NEVER be keyframe (keyframe interpolates
        # stills and cannot lip-sync or sync motion to the music). See backend/video.build_ltx_keyframe.
        render = str(s.get("render") or "").strip().lower()
        if render not in ("msr", "keyframe"):
            render = "msr"
        if render == "keyframe" and (lipsync or stype == "performance" or chars):
            render = "msr"
        out.append({
            "idx": i,
            "section": str(s.get("section") or ""),
            "start": int(float(s.get("start") or 0)),
            "end": int(float(s.get("end") or 0)),
            "type": stype,
            "render": render,
            "scene": str(s.get("scene") or "").strip(),
            "action": str(s.get("action") or s.get("motion") or "").strip(),
            "costume": str(s.get("costume") or "").strip(),
            "characters": chars,
            "lipsync": lipsync,
            "camera": _camera(s.get("camera")),
            "segments": segs,
        })
    return out


def generate_script(song, cast, provider, model, claude_model, n_shots=0):
    """Returns the parsed shot list (raises on LLM / JSON failure)."""
    system, prompt = build_prompt(song, cast, n_shots)
    text = llm_mod.complete(provider, model, system, prompt, claude_model)
    return parse_shots(text)


def build_shot_grid(segments, downbeats, total, target=7.0, min_shot=3.0):
    """Build a deterministic shot grid from the song's ACTUAL audio structure (allin1). Every section
    boundary is a cut; long sections are subdivided into ~`target`s shots placed on DOWNBEATS (bar
    lines) so every cut lands on a real structural or musical boundary - not nudged, placed there.
    segments=[{start,end,label}], downbeats=[sec]. Returns [{start,end,section}] in order, gapless.
    The LLM later fills CONTENT per window; it never chooses timing (the thing it does badly)."""
    dbs = sorted(float(d) for d in (downbeats or []))
    grid = []
    for seg in segments or []:
        s, e = float(seg.get("start", 0)), float(seg.get("end", 0))
        label = seg.get("label") or "section"
        dur = e - s
        if dur <= target * 1.4:
            grid.append({"start": round(s, 2), "end": round(e, 2), "section": label})
            continue
        n = max(2, int(round(dur / target)))
        inner = [d for d in dbs if s + min_shot < d < e - min_shot]
        cuts = [s]
        for k in range(1, n):
            ideal = s + dur * k / n
            nd = min(inner, key=lambda d: abs(d - ideal)) if inner else ideal
            if cuts[-1] + min_shot < nd < e - min_shot:   # else fall back to the even-split time
                cuts.append(round(nd, 2))
            elif cuts[-1] + min_shot < ideal < e - min_shot:
                cuts.append(round(ideal, 2))
        cuts.append(e)
        cuts = sorted(set(round(c, 2) for c in cuts))
        for i in range(len(cuts) - 1):
            grid.append({"start": cuts[i], "end": cuts[i + 1], "section": label})
    return grid


def build_grid_prompt(song, cast, grid):
    """Prompt variant for the structure-driven path: the shots are ALREADY cut to the audio structure;
    the LLM only fills each fixed window's CONTENT (no timing)."""
    summary, total = _song_summary(song)
    title = song.get("title") or "Untitled"
    named = [c for c in cast if c.get("name")]
    musicians = [c for c in named if c.get("kind") != "actor"]
    actors = [c for c in named if c.get("kind") == "actor"]

    def _fmt(lst):
        return "\n".join(f"  - {c['name']} ({c.get('role') or 'character'})" for c in lst)
    cast_parts = []
    if musicians:
        cast_parts.append("Band / musicians (PERFORMANCE shots playing/singing - the lead singer "
                          "lip-syncs; may also act in NARRATIVE scenes):\n" + _fmt(musicians))
    if actors:
        cast_parts.append("Actors (NARRATIVE / story shots, not performing music):\n" + _fmt(actors))
    cast_txt = "\n\n".join(cast_parts) or "  (no fixed cast - lean scenic/atmospheric)"

    windows = "\n".join(f"  Shot {i + 1}: [{g['start']}-{g['end']}s]  ({g['section']})"
                        for i, g in enumerate(grid))
    system = ("You are a music video director. The video is ALREADY cut into shots on the song's actual "
              "structure (section changes + downbeats). Your job is to fill in the CONTENT of each shot - "
              "NOT the timing. Output STRICT JSON ONLY (no prose, no markdown).")
    prompt = f"""{summary}

DIRECTION:
- The song is titled "{title}". Make the TITLE the central visual theme of the whole video.
- For each shot, read the lyric lines sung in its time window above and make the scene VISUALLY
  ILLUSTRATE those exact words. Instrumental windows: advance the story or show atmosphere.

Characters (keep each visually consistent wherever they appear):
{cast_txt}

The video is cut into {len(grid)} shots, IN ORDER, already aligned to the song structure:
{windows}

Return ONLY a JSON array of EXACTLY {len(grid)} objects, one per shot above, IN THE SAME ORDER.
Do NOT include start/end - the timing is fixed by the list above. Each object:
{{"type": "performance" | "narrative" | "broll",
  "render": "msr" | "keyframe",
  "scene": "<the look of the frame: setting, subject, lighting, FRAMING (close/medium/wide), photoreal>",
  "action": "<what the SUBJECT does - performance, gesture, expression, movement; describe the PERSON>",
  "camera": "static" | "slow push-in" | "slow pull-back",
  "costume": "<what the named characters WEAR; '' if on-stage performance wear or not notable>",
  "characters": [<names of any named characters present; prefer ONE per shot; [] if none>],
  "lipsync": <true for ANY shot where a singer is singing the lyrics ON CAMERA - close OR wide, alone OR
    with the band, standing OR moving. If the lead singer is visible while words are sung, set true.
    false ONLY when no one is singing on screen (instrumental, scenic B-roll, non-singing actor)>,
  "segments": [<OPTIONAL within-shot timeline; each {{"seconds": <num>, "action": "<motion/expression>"}}; [] if one continuous action>]}}

RENDER MODE: "msr" = the DEFAULT, the only mode that animates a person and lip-syncs - use for every
performance / singing / playing shot and any shot with a character. "keyframe" = ONLY pure scenic
B-roll with NO performer and nothing synced to the music. If "lipsync" is true or the lead singer is
present, "render" MUST be "msr".

CAMERA: only "static", "slow push-in", or "slow pull-back". No pans/tracking/orbits. Keep camera
language OUT of scene/action.

CASTING: prefer ONE named character per shot (crowded frames make people drift/swap attributes). Give
each character solo MEDIUM/CLOSE shots. At most ONE wide whole-band shot in the video.

LIP-SYNC RULE: during any SUNG section, EVERY shot showing the lead singer - wide, close, walking, or on
stage with the band - must set "lipsync": true. Do not reserve lip-sync for close-ups.

Photoreal live-action metal video. Every shot connects to BOTH the title theme and the lyrics in its window."""
    return system, prompt


def generate_script_grid(song, cast, provider, model, claude_model, grid):
    """Structure-driven script: the LLM fills CONTENT for each fixed grid window; we attach the locked
    audio-aligned times (the LLM never sets timing). Returns the parsed shot list, gapless on-structure."""
    system, prompt = build_grid_prompt(song, cast, grid)
    text = llm_mod.complete(provider, model, system, prompt, claude_model)
    content = parse_shots(text)                       # parsed content (its start/end are ignored)
    out = []
    for i, g in enumerate(grid):
        c = dict(content[i]) if i < len(content) else {}
        c["idx"] = i
        c["start"] = g["start"]
        c["end"] = g["end"]
        c["section"] = c.get("section") or g.get("section") or ""
        if not c.get("scene") and not c.get("action"):
            c.setdefault("type", "broll")
            c.setdefault("render", "keyframe")
            c["scene"] = f"{g.get('section', 'scene')} - atmospheric shot consistent with the song"
        out.append(c)
    return out


def snap_shots_to_structure(shots, seg_bounds, downbeats):
    """Align shot cuts to the song's ACTUAL audio structure. seg_bounds + downbeats are second-times
    detected from the rendered audio by allin1 (NOT the planned arrangement, which drifts). Each cut
    between consecutive shots is nudged to the nearest segment boundary (structural, wider tolerance)
    else the nearest downbeat (musical, tight), keeping shots contiguous and a sane minimum length.
    Only the boundaries are used - allin1's section labels are not trusted. The first start and last
    end are left as-is. Returns the shots with start/end adjusted (idempotent if already aligned)."""
    if len(shots) < 2:
        return shots
    SEG_TOL, DB_TOL, MIN = 3.5, 0.4, 1.5
    segs = sorted(float(x) for x in (seg_bounds or []))
    dbs = sorted(float(x) for x in (downbeats or []))

    def nearest(v, arr):
        if not arr:
            return None
        best = min(arr, key=lambda x: abs(x - v))
        return best, abs(best - v)

    bounds = [float(shots[0].get("start", 0))] + [float(s.get("end", 0)) for s in shots]
    for i in range(1, len(bounds) - 1):
        v = bounds[i]
        seg = nearest(v, segs)
        db = nearest(v, dbs)
        nv = v
        if seg and seg[1] <= SEG_TOL:
            nv = seg[0]
        elif db and db[1] <= DB_TOL:
            nv = db[0]
        if bounds[i - 1] + MIN < nv < bounds[i + 1] - MIN:
            bounds[i] = round(nv, 2)
    for i, s in enumerate(shots):
        s["start"] = bounds[i]
        s["end"] = bounds[i + 1]
    return shots
