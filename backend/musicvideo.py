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
            # optional per-segment start offset (`ss`): play the clip from `ss` seconds in, not
            # frame 0. Used when the opening shot was rendered LONG and its head is consumed by the
            # intro pre-roll - the body must CONTINUE from where the intro stopped, not replay the
            # head (replaying it is the "restart jump" at the wind-fade). Input seek before -i.
            cmd = [ff, "-y", "-loglevel", "error"]
            ss = float(seg.get("ss") or 0)
            if ss > 0:
                cmd += ["-ss", f"{ss:.3f}"]
            cmd += ["-i", seg["path"], "-vf", vf, "-t", f"{durs[i]:.3f}", "-an", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-r", str(fps), o]
            subprocess.run(cmd, check=True)
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


def _song_summary(song, grid=None):
    """Human-readable song brief for the LLM + total duration (sec). Lyrics are shown line by
    line under each section's time window so shots can be anchored to the words sung then.

    With `grid` (the analysed segments) the windows are stated on the REAL audio timeline instead of
    the arrangement's nominal one. That matters: the drift reached 12s on a real song, and the writer
    was anchoring shots to words that are sung several seconds earlier or later than it was told."""
    title = song.get("title") or "Untitled"
    lines = [f"Title: {title}",
             f"Genre/mood tags: {song.get('tags') or ''}"]
    if song.get("bpm"):
        lines.append(f"BPM: {song['bpm']}")
    if song.get("keyscale"):
        lines.append(f"Key: {song['keyscale']}")
    secs = song.get("sections") or []
    anchors = _time_anchors(song, grid) if grid else []
    t = 0
    body = []
    for s in secs:
        dur = int(s.get("seconds") or 0)
        lyr = (s.get("lyrics") or "").strip()
        # the section STYLE carries the vocal-part marker ("tender female bell canto", "warm male",
        # "duet swell") - essential for voice-matched casting, so surface it to the writer
        style = (s.get("style") or "").strip()
        stag = f" ({style})" if style else ""
        w0 = round(_map_time(anchors, t)) if anchors else t
        w1 = round(_map_time(anchors, t + dur)) if anchors else t + dur
        head = f"  [{w0}-{w1}s] {s.get('type') or 'section'}{stag}:"
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


def _fmt_cast(lst):
    """One line per character: name (role) - appearance. The appearance carries each member's LOOK
    incl. gender, so the LLM can name them correctly in a band tableau (e.g. 'female guitarist on the
    left, male bassist on the right') instead of guessing."""
    out = []
    for c in lst:
        look = (c.get("appearance") or "").strip().replace("\n", " ")
        if len(look) > 160:
            look = look[:160].rsplit(" ", 1)[0] + "..."
        gender = (c.get("gender") or "").strip()
        role = c.get("role") or "character"
        tag = f"{gender} {role}".strip() if gender else role
        out.append(f"  - {c['name']} ({tag})" + (f" - {look}" if look else ""))
    return "\n".join(out)


def build_prompt(song, cast, n_shots):
    summary, total = _song_summary(song)
    title = song.get("title") or "Untitled"
    named = [c for c in cast if c.get("name")]
    musicians = [c for c in named if c.get("kind") != "actor"]   # default = musician/band
    actors = [c for c in named if c.get("kind") == "actor"]

    def _fmt(lst):
        return _fmt_cast(lst)
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
  "scene": "<SETTING / ENVIRONMENT ONLY - location, backdrop, lighting, mood. NO people, NO face, NO performer, NO framing words. This generates the empty background the character is composited into, so it must contain no person. e.g. 'a ruined ashen concert stage, ember haze' NOT 'close-up on Selene'.>",
  "framing": "close" | "medium" | "wide",
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
        band_in_scene = [str(x).strip() for x in (s.get("band_in_scene") or [])
                         if x and str(x).strip() not in ("[]", "none", "None", "-", "")]
        band_in_scene = [b for b in band_in_scene if b.lower() not in [c.lower() for c in chars]]
        # RENDER MODE: msr (default, performer/music-synced) vs keyframe (pure scenic B-roll). Hard-enforce
        # that lip-sync / performance / character-present shots can NEVER be keyframe (keyframe interpolates
        # stills and cannot lip-sync or sync motion to the music). See backend/video.build_ltx_keyframe.
        render = str(s.get("render") or "").strip().lower()
        if render not in ("msr", "keyframe"):
            render = "msr"
        if render == "keyframe" and (lipsync or stype == "performance" or chars):
            render = "msr"
        framing = str(s.get("framing") or "").strip().lower()
        if framing not in ("close", "medium", "wide"):
            framing = "medium"
        if lipsync and framing == "wide":                 # lip-sync needs the face big enough
            framing = "medium"
        out.append({
            "idx": i,
            "section": str(s.get("section") or ""),
            "start": int(float(s.get("start") or 0)),
            "end": int(float(s.get("end") or 0)),
            "type": stype,
            "render": render,
            "framing": framing,
            "scene": str(s.get("scene") or "").strip(),
            "action": str(s.get("action") or s.get("motion") or "").strip(),
            "costume": str(s.get("costume") or "").strip(),
            "characters": chars,
            "band_in_scene": band_in_scene,
            "lipsync": lipsync,
            "camera": _camera(s.get("camera")),
            "segments": segs,
        })
    return out


def generate_script(song, cast, provider, model, claude_model, n_shots=0):
    """Returns the parsed shot list (raises on LLM / JSON failure)."""
    system, prompt = build_prompt(song, cast, n_shots)
    text = llm_mod.complete(provider, model, system, prompt, claude_model, timeout=600)
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

    # Coalesce any sub-`min_shot` window into a neighbour. allin1 can emit a tiny leading "start"
    # segment (e.g. 0.0-0.42s); left in, the frontend's per-shot minimum expands it and it OVERLAPS the
    # next shot (a 2s shot at 0-2 plus a 7s shot at 0.42-7.47). Merge keeps the grid gapless + on-structure.
    merged = []
    for w in grid:
        w = dict(w)
        short = (w["end"] - w["start"]) < min_shot
        prev_short = merged and (merged[-1]["end"] - merged[-1]["start"]) < min_shot
        if merged and (short or prev_short):
            merged[-1]["end"] = w["end"]                                   # absorb (grow the previous window)
            merged[-1]["section"] = merged[-1].get("section") or w.get("section")
        else:
            merged.append(w)
    return merged


def build_grid_prompt(song, cast, grid):
    """Prompt variant for the structure-driven path: the shots are ALREADY cut to the audio structure;
    the LLM only fills each fixed window's CONTENT (no timing)."""
    summary, total = _song_summary(song)
    title = song.get("title") or "Untitled"
    named = [c for c in cast if c.get("name")]
    musicians = [c for c in named if c.get("kind") != "actor"]
    actors = [c for c in named if c.get("kind") == "actor"]

    def _fmt(lst):
        return _fmt_cast(lst)
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
  "scene": "<the SETTING / ENVIRONMENT ONLY - the location, backdrop, lighting and mood. Describe NO people, NO face, NO performer, and NO framing words (no 'close-up', 'wide', etc.). This text generates the empty background the character is composited into, so it must contain no person. e.g. 'a ruined ashen concert stage, ember haze, dark sky' - NOT 'close-up on Selene singing'.>",
  "framing": "close" | "medium" | "wide",
  "action": "<what the SUBJECT does - performance, gesture, expression, movement; describe the PERSON>",
  "camera": "static" | "slow push-in" | "slow pull-back",
  "costume": "<what the named characters WEAR; '' if on-stage performance wear or not notable>",
  "characters": [<the ONE MSR-anchored performer this shot is about (the lead singer for a singing shot, the soloist for a solo). NEVER list more than one - extra MSR subjects blend/swap identities. [] if none>],
  "band_in_scene": [<OTHER named band members playing in this shot - composited into the BACKGROUND still (shown playing their instruments), NOT MSR-anchored. Name them in "action" by look + instrument + side. [] if none / not a band shot>],
  "lipsync": <true for ANY shot where a singer is singing the lyrics ON CAMERA (frame it CLOSE or MEDIUM
    per FRAMING below). false ONLY when no one is singing on screen (instrumental, scenic B-roll, non-singing actor)>,
  "segments": [<OPTIONAL within-shot timeline; each {{"seconds": <num>, "action": "<motion/expression>"}}; [] if one continuous action>]}}

STYLE - CRITICAL FOR IDENTITY: pick ONE consistent visual grade/palette for the WHOLE video (from the
song's mood + title) and apply that SAME look to every shot's "scene" - one palette, one consistent soft
light. Do NOT give individual shots their own dramatic or coloured lighting (no "deep crimson/gold light
cutting across her face", no harsh coloured side-light on people): strong coloured directional light
re-shades a character's face and hair and BREAKS the identity reference. Keep light on faces soft and
even, and the palette identical shot to shot.

RENDER MODE: "msr" = the DEFAULT, the only mode that animates a person and lip-syncs - use for every
performance / singing / playing shot and any shot with a character. "keyframe" = ONLY pure scenic
B-roll with NO performer and nothing synced to the music. If "lipsync" is true or the lead singer is
present, "render" MUST be "msr".

BAND / LIVE PERFORMANCE shots: ONLY the LEAD SINGER goes in "characters" (she is the MSR-anchored,
animated, lip-syncing subject). The OTHER on-stage band members go in "band_in_scene" - they are
composited into the BACKGROUND still (which DOES show them playing), NOT MSR-anchored. Do NOT put more
than the lead singer in "characters": anchoring two+ people via MSR makes their identities blend/swap.
For a band shot, write the "action" as the whole tableau, naming each member by their LOOK (incl. gender
from the cast list above), instrument and a fixed side - e.g. "Selene sings centre into the mic; the
female guitarist plays on the left; the male bassist plays on the right; a drummer at a full kit behind
them". EVERY band member present (the featured one in "characters" AND everyone in "band_in_scene") MUST
be named in the "action" - this is what the background composite is built from AND what the video render
keeps; a member NOT named in the action gets DROPPED by the render (it dissolves the unmentioned figure).
So always name all of them. A live band shot ALWAYS has a drum kit with a drummer at the back (the drummer
is generic - never named as a character / never given a reference, but always mention "a drummer at a full
kit behind them" so the kit + drummer appear).

SETTING (critical - avoids empty, repetitive shots): a CONCERT / PERFORMANCE STAGE is ONLY for FULL-BAND
shots. If you set a shot on a stage you MUST populate "band_in_scene" with the other members - never put
the lead singer alone on a bare stage (it looks empty and wrong, and a stage implies a band).
- SOLO lead-singer shots (band_in_scene = []) must be set in a VARIED, evocative location from the WORLD
  of this song - tie it to the title theme and to the lyric sung in that window (e.g. the ashen garden,
  ruined halls, a windswept plain, a flooded crypt, an overgrown interior). Make each interesting and
  DIFFERENT from the others; do NOT reuse the same location or default everything to one backdrop.
- Across the whole video, vary the scenery shot to shot. Only the genuine band shots share the stage.
- SETTING SCALE MUST MATCH FRAMING: a CLOSE or MEDIUM shot needs the subject CLOSE to camera, so the
  setting must give her a NEAR FOREGROUND ANCHOR to stand at - a textured surface, foliage, rubble, a wall
  / charred trunk / arch right behind her, OR a near edge she stands at (a pool's edge, a ledge, a doorway).
  The render rescales the scene around her, so even a larger setting works IF there is a clear foreground
  anchor. Do NOT set a close/medium shot in an OPEN VISTA with nothing in the near foreground (an overlook,
  ridgeline, open plain to the horizon) - there she renders tiny in the landscape. Open vistas = WIDE only.

LIP-SYNC ACTION (critical): on any lip-sync shot the "action" MUST be the singer PERFORMING the vocal TO
CAMERA - facing the camera and singing, mouth visible. NEVER write a pose that hides or turns the face:
no kneeling, no gazing away or down, no looking at a reflection, no back-to-camera, no silhouette, and do
not compose the shot around anything other than her singing. If she isn't visibly singing to camera, the
lip-sync has nothing to drive and the shot fails.

CAMERA: only "static", "slow push-in", or "slow pull-back". No pans/tracking/orbits. A push-in/pull-back
must stay GENTLE - never pull back far enough to shrink the subject. Keep camera language OUT of scene/action.
NO-RETURN RULE: band_in_scene members are a static background composite - if a shot has band_in_scene, the
camera must NOT move off them and then back onto them (they would look frozen/wrong). Prefer static, or a
gentle push-in toward the lead singer (moving away from the background band), never back toward them.

FRAMING (hard rule): any lip-sync shot MUST be CLOSE or MEDIUM so the singer's face is large enough to
lip-sync - NEVER wide, full-body, or establishing, and never a pull-back that shrinks her face. Reserve
WIDE / establishing framings for B-roll or non-singing moments only.

CASTING: prefer ONE named character per shot (crowded frames make people drift/swap attributes). Give
each character solo MEDIUM/CLOSE shots. At most ONE wide whole-band shot, and it must NOT be a lip-sync shot.

LIP-SYNC RULE: during any SUNG section, EVERY shot showing the lead singer must set "lipsync": true - and
per FRAMING keep it close/medium. Do not reserve lip-sync for close-ups, but never make a singing shot wide.

Photoreal live-action metal video. Every shot connects to BOTH the title theme and the lyrics in its window."""
    return system, prompt


def generate_script_grid(song, cast, provider, model, claude_model, grid):
    """Structure-driven script: the LLM fills CONTENT for each fixed grid window; we attach the locked
    audio-aligned times (the LLM never sets timing). Returns the parsed shot list, gapless on-structure."""
    system, prompt = build_grid_prompt(song, cast, grid)
    text = llm_mod.complete(provider, model, system, prompt, claude_model, timeout=600)
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


# ========================================================= MiniMax H3 hybrid segment pipeline (Phase 3)
# The MV spine on the H3 reference lane (docs/MINIMAX_H3_PLAN.md). Division of labor, mirroring the
# proven grid flow: CODE owns timing and the compiled prompt; the LLM owns creative content and the
# single-vs-scene choice (user-decided hybrid shot model, 2026-08-09).
#
# A SEGMENT is the render unit (one /api/video/h3_ref2v call): either
#   kind "single" - one shot (mandatory for lip-sync; keeps the expensive lane re-rollable), or
#   kind "scene"  - 2-4 scripted cuts inside one generation via H3's native [Shot N] At MM:SS.mmm
#                   timestamps (B-roll / narrative runs; intrinsic cross-cut continuity).
# All timing lives on H3's frame grid: frames % 17 == 5 at 24 fps, trained range 124-362 frames
# (5.17 - 15.08 s). Segments are built from the audio-structure shot grid and never cross a section
# boundary. Renders are snapped UP to the grid and trimmed at assembly (same pattern as LTX).

H3_SEG_FPS = 24
H3_SEG_MIN_S = 124 / 24.0        # 5.17s - shortest trained duration
H3_PROMPT_MAX = 7000     # MiniMax's documented prompt ceiling (1-7000 chars). Verified-clean band
#   renders sat near 4,800; a three-cut two-person segment reached 6,938 before the 2026-08-12
#   duet-staging wording, which pushed it to 7,154. Compile reports over_limit rather than letting
#   an over-length prompt go out silently.
H3_SEG_MAX_S = 10.5              # practical ceiling MEASURED 2026-08-09: at the trained max
#   (15.08s) a heavy-reference render (5 pictures + audio) collapses from the tail - audio
#   diverges from ~10s (envelope r 0.415 vs 0.975) and the image degrades to noise by ~14s;
#   the IDENTICAL payload at 10.1s is clean end to end. 11-14s is unexplored territory.
#   362 frames stays as the hard clamp in h3_seg_seconds for explicit long renders.
H3_CAMERA_MOVES = {
    # Writer vocabulary -> the author's exact phrasing. TWO TIERS:
    #  - GENTLE (small amplitude / slow speed) is the pace-safe envelope measured in Phase 1.
    #  - ASSERTIVE was added 2026-08-10 because the first videos read as static and timid: 27 of 54
    #    shots were locked off and the rest were nearly all a small push-in. These carry real
    #    movement while every shot still keeps the duration anchor and sky-pin that actually prevent
    #    the timelapse failure - amplitude is a different axis from scene pace. [UNVERIFIED: the
    #    measured envelope covered the gentle tier only, so treat the assertive moves as needing a
    #    look before a batch of renders leans on them.]
    "static": "The camera holds still",
    "push in": "The camera pushes in with small amplitude at slow speed",
    "pull back": "The camera pulls back with small amplitude at slow speed",
    "truck left": "The camera trucks left with small amplitude at slow speed",
    "truck right": "The camera trucks right with small amplitude at slow speed",
    "arc left": "The camera arcs left around the subject with small amplitude at slow speed",
    "arc right": "The camera arcs right around the subject with small amplitude at slow speed",
    "tilt up": "The camera tilts up with small amplitude at slow speed",
    "crane up": "The camera cranes up with small amplitude at slow speed",
    # --- assertive tier ---
    "push in strong": "The camera pushes in decisively on the subject, closing the distance at a "
                      "steady moderate speed, the framing tightening as it goes",
    "pull back reveal": "The camera pulls steadily back at a moderate speed, opening out to reveal "
                        "the space around the subject",
    "orbit left": "The camera orbits left around the subject at a steady moderate speed, with clear "
                  "parallax as nearer and farther elements shift against each other",
    "orbit right": "The camera orbits right around the subject at a steady moderate speed, with clear "
                   "parallax as nearer and farther elements shift against each other",
    "handheld drift": "The camera floats on a subtle handheld drift, breathing slightly around the "
                      "subject, never locked off",
    "steadicam follow": "The camera moves with the subject in a smooth steadicam follow, holding them "
                        "in frame as they go",
    "crane down": "The camera cranes down at a steady moderate speed, settling toward the subject",
    "tilt down": "The camera tilts down at a steady moderate speed",
    "rack focus": "The camera holds position while the focus racks onto the subject, the foreground "
                  "and background falling soft around them",
}
_H3_SKY_RE = re.compile(r"\b(sky|skies|cloud|clouds|stars?|starlit|moon|moonlit|milky way|sunset|"
                        r"sunrise|dawn|dusk|horizon|aurora|night sky)\b", re.I)
_H3_STAGE_RE = re.compile(r"\bstage\b", re.I)
H3_MAX_PEOPLE = 4
#   Anchored PEOPLE per render - a safety valve, not a hard model limit. Provenance, checked
#   2026-08-10 against docs + community usage rather than assumed:
#   - MiniMax's own ref-prompt guide sets NO people limit (its worked example uses 4 subjects);
#     the input caps are 9 ref images / 3 videos / 3 audios.
#   - Community prompt collections show multi-person casts working (a family ensemble; a 7-member
#     group), and their successful prompts carry explicit "no extra people, no duplicated
#     characters" restrictions - duplication is a KNOWN failure mode countered in prompt text, not
#     by capping the cast.
#   - Our own data: 3 people verified clean once; identity merges observed AT 3 in both segment-5
#     drafts (bassist rendered twice playing bass AND guitar; the guitarist duplicated in place of
#     Selene wearing Selene's outfit - the close-up cut stayed clean both times). Those shots said
#     only "the band plays together"; the verified-clean render STATIONED everyone ("sings lead at
#     the center microphone / plays <Subject 5> on the left / ..."). So the working fix is the
#     station assignment + uniqueness restrictions compile_h3_prompt now emits, and 4 covers the
#     full band without changing content. Beyond it, extras become the text-only background band.
H3_MAX_REFS = 9
#   Reference pictures per render. NINE is MiniMax's documented input cap for ref_images (checked
#   2026-08-10), so this is the model's own ceiling rather than a guess. It was 8 - one past our
#   verified-clean 7 - and that cost us: segment 8's band shot lost the BASS picture to shedding,
#   leaving the bassist with no instrument while the guitarist held the only one, and H3 duly mixed
#   the two women up and put Selene's outfit on the guitarist [OBSERVED 2026-08-10]. A full band shot
#   needs exactly 9 (4 people + outfit + 2 instruments + 2 locations), and instruments are a big part
#   of what tells the players apart, so shedding them is worse than carrying the ninth picture.
#   Over the cap, compile_h3_prompt still sheds PROP pictures first (the instrument stays described
#   in text, and the station line names it), then the second environment. Characters are never shed.
#   Load is still real: 15.08s at 5 pictures collapsed from the tail, hence H3_SEG_MAX_S.


def h3_seg_seconds(dur):
    """Snap a duration UP to the H3 frame grid (frames % 17 == 5), clamped to the trained range.
    Returns (render_seconds, frames). Render slightly long + trim at assembly."""
    frames = max(124, int(round(float(dur) * H3_SEG_FPS)))
    while frames % 17 != 5:
        frames += 1
    frames = min(frames, 362)
    return round(frames / H3_SEG_FPS, 3), frames


def build_h3_segments(grid):
    """Merge the audio-structure shot grid ([{start,end,section}]) into SEGMENT windows for the H3
    hybrid model: greedy within-section merge up to H3_SEG_MAX_S; a window that cannot reach
    H3_SEG_MIN_S on its own stays (the render is snapped up and trimmed). Each segment carries its
    member windows as `cuts` so a "scene" segment can place H3 timestamps on the real grid cuts.
    Returns [{start, end, seconds, render_seconds, frames, section, cuts:[{start,end}...]}]."""
    segs = []
    for w in grid or []:
        s, e, label = float(w["start"]), float(w["end"]), w.get("section") or "section"
        if (segs and segs[-1]["section"] == label
                and (e - segs[-1]["start"]) <= H3_SEG_MAX_S):
            segs[-1]["end"] = e
            segs[-1]["cuts"].append({"start": s, "end": e})
        else:
            segs.append({"start": s, "end": e, "section": label, "cuts": [{"start": s, "end": e}]})
    for g in segs:
        g["seconds"] = round(g["end"] - g["start"], 2)
        g["render_seconds"], g["frames"] = h3_seg_seconds(g["seconds"])
    return segs


H3_MIN_CUT_S = 1.5       # a cut shorter than this is not a usable shot, so never create one
H3_CUT_SNAP_S = 0.6      # a handover this close to an existing boundary is already aligned


def snap_segment_edges_to_handovers(segments, song, grid=None):
    """Move a SEGMENT boundary onto a vocal handover that lands too close to it to cut.

    A handover inside a segment normally gets its own cut. When it sits within H3_MIN_CUT_S of the
    segment's own start or end, a cut there would be a sub-1.5s shot - so the segment edge itself
    moves onto the handover and the neighbour absorbs the difference. That is what segment 5 needed:
    the handover landed 0.37s before its end, leaving Selene lip-syncing over Bob's first words with
    no room for a third shot. Renders are re-snapped (render_seconds/frames) for both segments, so
    any clip already rendered for them is invalidated and must be re-rendered.

    Returns notes describing every boundary moved."""
    wins = _voice_windows(song, grid or segments)
    if not wins or len(segments) < 2:
        return []
    times = sorted({round(w[k], 3) for w in wins for k in ("start", "end")})
    notes = []

    def resize(seg):
        seg["seconds"] = round(seg["end"] - seg["start"], 2)
        seg["render_seconds"], seg["frames"] = h3_seg_seconds(seg["seconds"])

    # The neighbour absorbing the difference gets LONGER, and length is the one axis where H3 is
    # known to fail (a 15.08s render collapsed from the tail; 10.1s is the longest verified clean).
    # Frames come in steps of 17, so absorbing a fraction of a second can jump a whole step: allow
    # at most one step past the measured ceiling, and say so in the note rather than drift silently.
    grow_cap = H3_SEG_MAX_S + 17 / H3_SEG_FPS

    def prospective(a, b):
        return h3_seg_seconds(round(b - a, 2))[0]

    for i, seg in enumerate(segments):
        for ht in times:
            # a handover just before this segment's END: hand the tail to the next segment
            if (i + 1 < len(segments) and seg["end"] - H3_MIN_CUT_S < ht < seg["end"]
                    and ht > seg["start"] + H3_MIN_CUT_S):
                nxt = segments[i + 1]
                rs = prospective(ht, nxt["end"])
                if rs > grow_cap:
                    continue                       # would push the neighbour past the render ceiling
                over = f", segment {i + 2} now renders {rs:.2f}s (past the {H3_SEG_MAX_S}s measured ceiling)" \
                       if rs > H3_SEG_MAX_S else ""
                notes.append(f"segment {i + 1} end {seg['end']:.2f}s -> {ht:.2f}s (voice change){over}")
                seg["end"] = ht
                seg["cuts"][-1]["end"] = ht
                nxt["start"] = ht
                nxt["cuts"][0]["start"] = ht
                seg["edge_snapped"] = nxt["edge_snapped"] = True
                resize(seg)
                resize(nxt)
            # a handover just after this segment's START: hand the head to the previous segment
            elif (i > 0 and seg["start"] < ht < seg["start"] + H3_MIN_CUT_S
                    and ht < seg["end"] - H3_MIN_CUT_S):
                prv = segments[i - 1]
                rs = prospective(prv["start"], ht)
                if rs > grow_cap:
                    continue
                over = f", segment {i} now renders {rs:.2f}s (past the {H3_SEG_MAX_S}s measured ceiling)" \
                       if rs > H3_SEG_MAX_S else ""
                notes.append(f"segment {i + 1} start {seg['start']:.2f}s -> {ht:.2f}s (voice change){over}")
                seg["start"] = ht
                seg["cuts"][0]["start"] = ht
                prv["end"] = ht
                prv["cuts"][-1]["end"] = ht
                seg["edge_snapped"] = prv["edge_snapped"] = True
                resize(seg)
                resize(prv)
    return notes


def split_cuts_at_voice_handovers(segments, song):
    """Add a cut boundary wherever the singing VOICE changes inside a segment.

    The grid's cuts come from downbeat windows, which know nothing about who is singing: on a real
    song the female-to-male handover landed 2.9s after segment 5's only internal cut, so the singer
    on camera changed ~3s before the voice did. A boundary at the handover lets each cut sit inside
    ONE voice window, which is what makes per-cut casting correct.

    Only timestamps are touched - the segment's own start/end, render_seconds and frame count are
    untouched, so the frames%17==5 grid is unaffected. Splits are skipped when they would make a cut
    shorter than H3_MIN_CUT_S, when the boundary already exists (within H3_CUT_SNAP_S), or when the
    segment already carries the 4 cuts the writer schema allows. A segment that gains a split is
    marked `voice_split` so it cannot later collapse to a single shot spanning both voices."""
    wins = _voice_windows(song, segments)
    if not wins:
        return []
    times = sorted({round(w[k], 3) for w in wins for k in ("start", "end")})
    notes = []
    for seg in segments:
        for ht in times:
            cuts = seg.get("cuts") or []
            if len(cuts) >= 4:
                break
            if not (seg["start"] + H3_MIN_CUT_S <= ht <= seg["end"] - H3_MIN_CUT_S):
                continue
            if any(abs(ht - c["start"]) <= H3_CUT_SNAP_S or abs(ht - c["end"]) <= H3_CUT_SNAP_S for c in cuts):
                continue
            for idx, c in enumerate(cuts):
                if c["start"] + H3_MIN_CUT_S <= ht <= c["end"] - H3_MIN_CUT_S:
                    seg["cuts"] = (cuts[:idx] + [{"start": c["start"], "end": ht},
                                                 {"start": ht, "end": c["end"]}] + cuts[idx + 1:])
                    seg["voice_split"] = True
                    notes.append(f"segment at {seg['start']:.1f}s: cut added at {ht:.1f}s "
                                 f"({_voice_at(wins, c['start'], ht) or 'none'} -> "
                                 f"{_voice_at(wins, ht, c['end']) or 'none'})")
                    break
    return notes


def build_h3_grid_prompt(song, cast, segments, grid=None):
    """Writer prompt for the hybrid model: per SEGMENT the LLM chooses "single" or "scene" and fills
    creative content only (timing is fixed by the segment windows; scene cut times come from the
    segment's own audio-grid cuts). The environment may now CONTAIN people (H3's subject-swap
    replaces them - verified); it must still be environment-LED prose, not a person portrait.

    `grid` is the whole video's segments; it puts the lyric windows in the brief on the real audio
    timeline rather than the arrangement's nominal one (`segments` here is only this batch)."""
    summary, total = _song_summary(song, grid or segments)
    title = song.get("title") or "Untitled"
    named = [c for c in cast if c.get("name")]
    cast_txt = _fmt_cast(named) if named else "  (no fixed cast - lean scenic/atmospheric)"

    wins_all = _voice_windows(song, grid or segments)

    def _win(i, g):
        cuts = ""
        if len(g["cuts"]) > 1:
            cuts = "  internal cuts at " + ", ".join(f"{c['start']:.2f}s" for c in g["cuts"][1:])
        # spell out the handover: this segment's cut is ALREADY placed on it, and the writer has to
        # cast each side to the voice singing there
        hand = ""
        if g.get("voice_split"):
            for c in g["cuts"][1:]:
                a = _voice_at(wins_all, g["start"], c["start"])
                b = _voice_at(wins_all, c["start"], g["end"])
                if a != b:
                    hand = (f"  <<< THE VOICE CHANGES at {c['start']:.2f}s: {a or 'instrumental'} before, "
                            f"{b or 'instrumental'} after. Cast each cut to the voice singing in it; "
                            f"this segment must be a \"scene\", never one shot.")
                    break
        return f"  Segment {i + 1}: [{g['start']:.2f}-{g['end']:.2f}s] ({g['section']}){cuts}{hand}"
    windows = "\n".join(_win(i, g) for i, g in enumerate(segments))

    system = ("You are a music video director working with the MiniMax H3 video model. The video is "
              "ALREADY cut into render segments on the song's actual structure. Fill in the CONTENT of "
              "each segment - never the timing. Output STRICT JSON ONLY (no prose, no markdown).")
    prompt = f"""{summary}

DIRECTION:
- The song is titled "{title}". Make the TITLE the central visual theme of the whole video.
- For each segment, read the lyric lines sung in its window and make the visuals ILLUSTRATE those
  exact words. Instrumental windows: advance the story or show atmosphere.

Characters (keep each visually consistent wherever they appear):
{cast_txt}

The video is cut into {len(segments)} SEGMENTS, IN ORDER, aligned to the song structure:
{windows}

Return ONLY a JSON array of EXACTLY {len(segments)} objects, one per segment, IN ORDER:
{{"kind": "single" | "scene",
  "shots": [<"single" = exactly ONE shot object; "scene" = one shot object PER internal cut listed
             for that segment (2-4), in order>],
  "soundscape": "<the segment's ambient/physical sounds - wind, room tone, cloth, footsteps; short>"}}
Each shot object:
{{"type": "performance" | "narrative" | "broll",
  "location": "<a SHORT stable name for the place, reused EXACTLY (same spelling) every time the
    video returns there - e.g. 'lighthouse lantern room', 'clifftop edge'. Each named location gets
    ONE environment reference shared by all its shots, so reuse builds continuity. '' if abstract>",
  "scene": "<the ENVIRONMENT, described as rich flowing prose: location, surfaces, materials, light
    sources, weather, depth layers. Environment-LED (not a person portrait), but people ARE allowed
    in it when the story wants them - the render replaces/controls who appears.>",
  "framing": "close" | "medium" | "wide",
  "action": "<what the subject does: EXACTLY ONE continuous motion or performance for the whole shot,
    described plainly. No compound sequences, no 'then'.>",
  "camera": gentle: "static" | "push in" | "pull back" | "truck left" | "truck right" |
            "arc left" | "arc right" | "tilt up" | "crane up"
            assertive: "push in strong" | "pull back reveal" | "orbit left" | "orbit right" |
            "handheld drift" | "steadicam follow" | "crane down" | "tilt down" | "rack focus",
  "costume": "<what named characters wear; '' if their reference wardrobe>",
  "characters": [<named cast in this shot; [] if none>],
  "lipsync": <true when the lead singer sings the lyrics ON CAMERA in this shot>}}

HARD RULES:
- VOICE-MATCHED CASTING: the singer ON CAMERA must be the singer whose VOICE is actually
  singing in that window. The section markers in the song brief say whose voice it is (e.g.
  "female bell canto" = the female singer, "warm male" = the male singer, "duet" = both).
  NEVER show a singer mouthing a part sung in the other voice. On duet windows both singers
  may appear - together in one shot or intercut - each singing their own lines.
  The MARKER decides, never the lyric's point of view. If the first verse is marked female and
  its words address a departed man, the WOMAN is on camera singing it - do not put the man on
  camera because the lyric is about him. Check every lip-sync shot against the marker for the
  window it sits in; getting this backwards makes the video look dubbed.
- SUNG SECTIONS focus on the SINGERS: B-roll IS allowed during singing - imagery tied to the
  lyric being sung breathes between singing shots - but never let it dominate a sung section.
  Instrumental windows are where B-roll and narrative belong.
- NO MIMED SINGING: any shot that SHOWS a singer singing MUST set "lipsync": true - there is no
  such thing as an unsynced singing shot (the mouth would visibly not match the vocal). Singers
  may also appear NOT singing - walking, gazing, reaching - never mouthing words.
- LIP-SYNC SHOTS may be single segments OR cuts inside a "scene" (e.g. [band wide -> the singer
  sings medium] is ONE scene). Every lip-sync shot is CLOSE or MEDIUM, the singer performing TO
  CAMERA, mouth clearly visible - never turned away, never silhouette, never wide.
- SINGING SHOTS ARE THE PRIORITY: while lyrics are sung, the video is carried by the correct
  singer's face SINGING - MOST segments in a sung section must contain a lip-sync shot, and a
  good share of them CLOSE-UPS of the singing face. Singing shots outrank band/stage shots:
  use band shots as seasoning between them, never instead of them.
- AT MOST TWO locations per scene segment (intercutting two threads - e.g. performance vs
  story - is good); name both. Each named location keeps ONE consistent environment.
- A STAGE location implies the BAND on it (the named musicians playing, plus a drummer at a
  full kit behind). Never put a lone singer on an empty stage - solo moments belong in the
  story locations instead.
- INSTRUMENT PERFORMANCE is shot FROM AFAR: solos and any non-singing playing shots (guitar,
  bass, drums) must be WIDE - full figures in the environment, never close-ups of hands,
  fretboards or drum sticks. Generated finger movement cannot match the actual notes, and a
  close-up makes that obvious; distance hides it. (The opposite of the lip-sync rule.)
- LOCATION CONTINUITY AND VARIETY: the whole video lives in ABOUT {H3_LOCATIONS} named locations
  ({H3_LOCATIONS - 1}-{H3_LOCATIONS + 1} is the target; 4 across four minutes read as repetitive).
  Enough places that the eye keeps moving, few enough that it stays one world. Returning to a
  location = the same "location" string verbatim, and each named location gets one shared
  environment reference. Introduce a NEW named location when the lyric moves somewhere the
  established ones cannot carry, until about {H3_LOCATIONS} exist; after that return to the
  established set rather than inventing more.
- "scene" gives one shot per listed internal cut, visually connected (an evolving viewpoint or
  a two-thread intercut). Its cuts may include lip-sync shots.
- ONE motion per shot. The action is a single continuous thing a real person/scene does at
  real-world speed. Never describe multiple movements, weather changes or time passing.
- CAMERA: give the video some MOVEMENT. The first cut of this video came back 27 shots locked off
  out of 54, nearly all the rest a small push-in, and it read as flat. So:
  * Vary the move from shot to shot. Never more than TWO static shots in a row, and do not make
    "push in" the answer to everything.
  * Match the energy. Choruses, band shots and the emotional peak take the assertive moves
    ("orbit left/right", "push in strong", "pull back reveal", "steadicam follow", "crane down").
    Verses and intimate singing take the quieter ones ("handheld drift", "push in", "rack focus",
    "tilt up"). "handheld drift" suits a hand-held documentary feel; "rack focus" suits a close-up
    where you want the eye pulled onto the face without the frame travelling.
  * "static" is a deliberate choice for a held, still moment - not a default.
  * Still ONE move per shot: never combine two, and never describe the camera in "action".
- Open-sky wide shots are motion-prone: prefer a near foreground anchor in every scene, and keep
  sky/stars/clouds INCIDENTAL, not the subject of motion.
- Framing scale must match the scene: close/medium needs a near foreground anchor for the subject;
  open vistas are wide-only.
- Pick ONE consistent visual grade/palette for the WHOLE video from the song's mood + title, and
  keep light on faces soft and even in every segment.
- Prefer ONE named character per shot. Vary locations across segments; tie each to the title theme
  and its lyric window.

Photoreal live-action music video. Every segment connects to BOTH the title theme and the lyrics
in its window."""
    return system, prompt


def _extract_json_array(text):
    """Pull the LARGEST valid JSON array of objects out of arbitrary LLM output. Robust against
    markdown fences, prose before/after, bracketed asides like "[Shot 1]", and trailing notes -
    the greedy-regex approach failed on exactly those (a real 8-minute writer run died with
    "Extra data" at the final parse, 2026-08-09). raw_decode at every '[' and keep the best."""
    dec = json.JSONDecoder()
    best, best_score, best_seg_like = None, (-1, -1), 0
    for m in re.finditer(r"\[", text):
        try:
            val, _end = dec.raw_decode(text, m.start())
        except ValueError:
            continue
        if isinstance(val, list) and val and all(isinstance(x, dict) for x in val):
            # prefer arrays whose elements ARE segments ("kind"/"shots" keys) - plain size picks
            # a nested shots array over a short outer segment array (caught by test 2026-08-09)
            seg_like = sum(1 for x in val if "shots" in x or "kind" in x)
            score = (seg_like, len(val))
            if score > best_score:
                best, best_score, best_seg_like = val, score, seg_like
    # A NON-segment-shaped array (e.g. an inner "shots" array, when the outer array is corrupt or
    # unterminated) must NOT win - otherwise salvage never runs and one shot is mistaken for the
    # whole script (caught by test 2026-08-09).
    fallback = best if (best is not None and best_seg_like == 0) else None
    if fallback is not None:
        best = None
    if best is None:
        # SALVAGE: no complete array parsed (a reply corrupted mid-stream, or a fragment whose
        # head was lost). Collect whatever complete top-level OBJECTS the text does contain -
        # partial output beats discarding a multi-minute run. MEASURED failure shapes 2026-08-09:
        # a 47KB reply that went malformed at object 27 (`,"",""` inside an object), and a
        # 919-byte fragment that began mid-object.
        objs = []
        pos = 0
        while pos < len(text):
            nxt = text.find("{", pos)
            if nxt < 0:
                break
            try:
                val, end = dec.raw_decode(text, nxt)
            except ValueError:
                pos = nxt + 1
                continue
            if isinstance(val, dict) and ("shots" in val or "kind" in val):
                objs.append(val)
            pos = end
        if objs:
            return objs
        if fallback is not None:
            return fallback
    if best is None:
        # keep the evidence: a failed 8-minute LLM run must be diagnosable, not vanished
        try:
            os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     ".mvwork"), exist_ok=True)
            fp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              ".mvwork", "h3_writer_failed.txt")
            with open(fp, "w") as f:
                f.write(text)
            raise ValueError(f"no JSON segment array found in the writer output (raw saved to {fp})")
        except OSError:
            raise ValueError("no JSON segment array found in the writer output")
    return best


def parse_h3_segments(text, segments):
    """Validate the writer's JSON against the fixed segment windows. Enforces (hard, in code):
    lipsync => kind single + close/medium; scene shot-count == internal cut count (clamped 1-4);
    camera vocabulary; one shot minimum. Returns the segment list with content attached."""
    data = _extract_json_array(text)
    out = []
    for i, g in enumerate(segments):
        seg = dict(g)
        content = data[i] if i < len(data) and isinstance(data[i], dict) else {}
        kind = str(content.get("kind") or "single").strip().lower()
        shots_in = [s for s in (content.get("shots") or []) if isinstance(s, dict)]
        shots = []
        for s in shots_in[:4]:
            framing = str(s.get("framing") or "").strip().lower()
            if framing not in ("close", "medium", "wide"):
                framing = "medium"
            lipsync = bool(s.get("lipsync"))
            stype = s.get("type") if s.get("type") in SHOT_TYPES else "broll"
            # NO-MIMED-SINGING safety net: a shot whose action shows someone singing MUST be
            # lip-synced (the 2026-08-09 rewrite hid singing inside scene segments with
            # lipsync=false - mouths flapping unsynced to the vocal). Forcing lipsync here also
            # collapses the segment to a synced single via the existing rule below.
            if (not lipsync and s.get("characters")
                    and re.search(r"\bsing(s|ing)?\b", str(s.get("action") or ""), re.I)):
                lipsync = True
            if lipsync and framing == "wide":
                framing = "medium"
            # INSTRUMENT-PERFORMANCE-FROM-AFAR (hard rule, mirrors the writer guidance): generated
            # finger/stick movement cannot match the actual notes, so non-singing performance shots
            # never render close - solo-section ones go fully wide
            if stype == "performance" and not lipsync:
                framing = "wide" if "solo" in str(g.get("section") or "").lower() else \
                          ("medium" if framing == "close" else framing)
            camera = str(s.get("camera") or "static").strip().lower()
            if camera not in H3_CAMERA_MOVES:
                camera = "static"
            chars = [str(x).strip() for x in (s.get("characters") or [])
                     if x and str(x).strip() not in ("[]", "none", "None", "-", "")]
            shots.append({"type": stype, "framing": framing, "lipsync": lipsync, "camera": camera,
                          "location": str(s.get("location") or "").strip(),
                          "scene": str(s.get("scene") or "").strip(),
                          "action": str(s.get("action") or "").strip(),
                          "costume": str(s.get("costume") or "").strip(),
                          "characters": chars})
        if not shots:
            shots = [{"type": "broll", "framing": "wide", "lipsync": False, "camera": "static",
                      "scene": "", "action": "", "costume": "", "characters": []}]
        lipsync_any = any(s["lipsync"] for s in shots)
        # lip-sync no longer forces a single segment (design change 2026-08-09): a scene may
        # carry lip-sync CUTS - the segment's audio window drives them all. Only a one-window
        # segment is structurally single.
        if len(seg["cuts"]) == 1:
            kind = "single"
        # a segment split at a vocal handover CANNOT be one shot: a single shot spanning the
        # boundary would have one person mouthing both voices
        elif seg.get("voice_split"):
            kind = "scene"
        if kind == "single":
            # keep the SINGING shot if the writer supplied several for a one-cut window
            keep = next((s for s in shots if s["lipsync"]), shots[0])
            shots = [keep]
            # one shot spans the WHOLE segment - collapse the merged windows so the compiled
            # duration anchor and any timestamps cover the full render, not just the first cut
            seg["cuts"] = [{"start": seg["start"], "end": seg["end"]}]
        else:
            # a scene segment carries one shot per internal cut; pad/trim to the real cut count
            n = max(2, min(len(seg["cuts"]), 4))
            while len(shots) < n:
                shots.append(dict(shots[-1]))
            shots = shots[:n]
        seg["kind"] = kind
        seg["shots"] = shots
        seg["lipsync"] = lipsync_any
        seg["soundscape"] = str(content.get("soundscape") or "").strip()
        out.append(seg)
    return out


def compile_h3_prompt(seg, cast, audio_ref=False):
    """Compile ONE segment into the six-section FULL REFERENCES prompt (verified format,
    docs/MINIMAX_H3_PLAN.md section 4 + the Phase 2 test prompts verbatim where proven).

    Picture numbering contract (the dispatcher MUST upload refs in this order):
      <Picture 1..N>    = the character sheets of `chars` (order = `picture_map`),
      <Picture N+1..M>  = the OUTFIT stills of characters carrying a `costume`
                          ({"name","desc","still_id"} on the cast entry; order = `outfit_map`),
      <Picture M+1>     = the segment's ENVIRONMENT still (`env_picture`).
      <Audio 1>         = the song window (only when audio_ref=True, i.e. a lip-sync segment).
    A character with a costume is dressed via H3's native outfit Subject [VERIFIED 2026-08-09]:
    the sheet's wardrobe is declared not-used and the outfit declared fully_preserved as worn.
    Enforces (in the emitted text): the subject-swap/exclusion clause on the environment still
    [verified Test B], direct-audio-reuse retention [verified lip-sync test], ONE motion + a
    real-world duration anchor per shot, the sky-pin clause when sky words appear, framing stated
    in BOTH summary and detailed_description [SING-2 drift lesson], and the environment-hold line.
    Also enforced here (so BOTH the script path and the per-segment recompile get them):
      - the H3_MAX_REFS reference budget (props shed to text first, then the second environment),
      - the no-lone-singer-on-a-bare-stage fill (background band, text only).
    Returns (prompt_text, picture_map)."""
    by_name = {c.get("name"): c for c in cast if c.get("name")}
    chars = []
    for s in seg["shots"]:
        for n in s["characters"]:
            if n in by_name and n not in chars:
                chars.append(n)
    # PEOPLE CAP (see H3_MAX_PEOPLE): keep the singers first, then instrument holders, then order of
    # appearance. Everyone dropped is still in the shot as the unresolved background band, so the
    # frame stays full without giving H3 a fourth identity to confuse.
    dropped_people = []
    if len(chars) > H3_MAX_PEOPLE:
        def _prio(n):
            sings = any(sh["lipsync"] and n in sh["characters"] for sh in seg["shots"])
            holds = bool((by_name[n].get("prop") or {}).get("still_id"))
            return (0 if sings else 1, 0 if holds else 1, chars.index(n))
        keep = set(sorted(chars, key=_prio)[:H3_MAX_PEOPLE])
        dropped_people = [n for n in chars if n not in keep]
        chars = [n for n in chars if n in keep]
    outfits = [n for n in chars if (by_name[n].get("costume") or {}).get("still_id")]
    outfit_pic = {n: len(chars) + 1 + i for i, n in enumerate(outfits)}
    # props (instruments/objects) follow the outfits in picture order - same verified mechanic
    props = [n for n in chars if (by_name[n].get("prop") or {}).get("still_id")]
    # ENVIRONMENTS, one per distinct LOCATION among the cuts (max 2 - the intercut pattern:
    # performance thread vs story thread). Keys are lowercased location names so the dispatcher
    # can match its location-shared stills; "" = an unnamed per-segment environment.
    env_locs = []
    for s in seg["shots"]:
        k = (s.get("location") or "").strip().lower()
        if k not in env_locs:
            env_locs.append(k)
    env_locs = env_locs[:2]
    if not env_locs:
        env_locs = [""]
    # REFERENCE BUDGET (see H3_MAX_REFS): shed props first - the instrument still stays described in
    # the character's definition, it just is not identity-locked to a picture - then the second
    # environment. Characters are never shed: they ARE the shot.
    text_props = []
    while len(chars) + len(outfits) + len(props) + len(env_locs) > H3_MAX_REFS and props:
        text_props.insert(0, props.pop())
    if len(chars) + len(outfits) + len(props) + len(env_locs) > H3_MAX_REFS and len(env_locs) > 1:
        env_locs = env_locs[:1]
    prop_pic = {n: len(chars) + len(outfits) + 1 + i for i, n in enumerate(props)}
    env_base = len(chars) + len(outfits) + len(props)
    env_map = {k: env_base + 1 + i for i, k in enumerate(env_locs)}

    def env_pic_of(s):
        return env_map.get((s.get("location") or "").strip().lower(), env_map[env_locs[0]])
    env_pic = env_map[env_locs[0]]                    # the primary environment (first cut's)
    lead = chars[0] if chars else None

    # PER-SHOT PRESENCE SCOPING (precaution): retention names the shots a person is in instead of
    # claiming "throughout", and each shot says who is absent. NOTE the observed segment-5 failures
    # were NOT in the close-up (it stayed clean both times) - they were identity merges INSIDE the
    # multi-person wide shot; the fix for those is the STATION assignment below. This scoping just
    # removes a standing contradiction (a subject declared preserved "throughout" a scene they are
    # absent from for a cut).
    def _shots_of(n):
        return [k + 1 for k, sh in enumerate(seg["shots"]) if n in sh["characters"]]

    def _when(n):
        ks = _shots_of(n)
        if len(seg["shots"]) == 1 or len(ks) == len(seg["shots"]):
            return "throughout"
        return "whenever they are on screen (only " + \
               " and ".join(f"[Shot {k}]" for k in ks) + ")"

    defs, keeps = [], []
    # WORDING IS DELIBERATELY TERSE. The one band render that came back clean was a 4,772-char
    # prompt; ours had grown to 9,251 - past MiniMax's documented 7,000-char ceiling - because every
    # fix added another sentence, and the drafts that failed were the long ones. The verified
    # mechanics all survive here (wardrobe-not-used + outfit fully_preserved, the environment
    # subject-swap clause, direct audio reuse); they are just said once and said short.
    for i, n in enumerate(chars):
        c = by_name[n]
        look = (c.get("look") or c.get("description") or "").strip()
        looktxt = f" - {look}" if look else ""
        tp = ""
        if n in text_props:      # instrument shed by the reference budget: keep it named, in words
            pr = by_name[n].get("prop") or {}
            tp = f" Plays {(pr.get('desc') or pr.get('name') or 'their instrument').rstrip('. ')}."
        if n in outfit_pic:
            co = by_name[n]["costume"]
            op = outfit_pic[n]
            defs.append(f"<Subject {i + 1}> is {n} from <Picture {i + 1}>{looktxt}. Keep the face and "
                        f"hair exactly; the clothes come from <Subject {op}>, NOT from "
                        f"<Picture {i + 1}>.{tp}")
            defs.append(f"<Subject {op}> is the OUTFIT from <Picture {op}>: "
                        f"{(co.get('desc') or 'the outfit').rstrip('. ')}. Shown on a dress form; only "
                        f"the garment is referenced, worn by <Subject {i + 1}>.")
            keeps.append(f"<Subject {i + 1}>: fully_preserved - face and hair exactly as "
                         f"<Picture {i + 1}> {_when(n)}; clothes from <Picture {op}>.")
            keeps.append(f"<Subject {op}>: fully_preserved - cut, colour and fabric as <Picture {op}>, "
                         f"worn by <Subject {i + 1}>.")
        else:
            costume = next((s["costume"] for s in seg["shots"] if s["costume"] and n in s["characters"]), "")
            wear = f", wearing {costume}" if costume else ""
            defs.append(f"<Subject {i + 1}> is {n} from <Picture {i + 1}>{looktxt}{wear}. Keep the "
                        f"face, hair and wardrobe exactly.{tp}")
            keeps.append(f"<Subject {i + 1}>: fully_preserved - face, hair and wardrobe exactly as "
                         f"<Picture {i + 1}> {_when(n)}.")
    for n in props:
        pr = by_name[n]["prop"]
        pp = prop_pic[n]
        pi = chars.index(n) + 1
        defs.append(f"<Subject {pp}> is the {(pr.get('name') or 'PROP').upper()} from <Picture {pp}>: "
                    f"{(pr.get('desc') or 'the object').rstrip('. ')}. Held and played by "
                    f"<Subject {pi}> ONLY.")
        keeps.append(f"<Subject {pp}>: fully_preserved - shape, materials, colours and hardware as "
                     f"<Picture {pp}>, played by <Subject {pi}>.")
    for k in env_locs:
        p = env_map[k]
        shots_here = [s for s in seg["shots"]
                      if (s.get("location") or "").strip().lower() == k] or seg["shots"]
        scene_txt = next((s["scene"] for s in shots_here if s["scene"]),
                         "the environment").rstrip(". ") + "."
        # the PICTURE carries the place; this prose only reinforces it, so keep it to the opening
        # sentences. Two full scene descriptions were ~800 chars of a prompt that has a 7,000 ceiling.
        if len(scene_txt) > 300:
            cut = scene_txt[:300].rsplit(". ", 1)
            scene_txt = (cut[0] + "." if len(cut) > 1 and len(cut[0]) > 80 else scene_txt[:300].rstrip(", ") + ".")
        disp = next(((s.get("location") or "").strip() for s in shots_here
                     if (s.get("location") or "").strip()), "")
        name = f" '{disp}'" if disp else ""
        # the subject-swap exclusion is a VERIFIED mechanic (Test B) - shortened, never dropped
        defs.append(f"<Subject {p}> is the ENVIRONMENT{name} from <Picture {p}>: {scene_txt} Only the "
                    f"place and its objects are referenced; anyone visible in <Picture {p}> is NOT in "
                    f"the target video.")
        keeps.append(f"<Subject {p}>: reference - architecture, objects and lighting recognisable from "
                     f"<Picture {p}>; its people do not appear.")
    if audio_ref and lead:
        defs.append(f"<Audio 1> is the song that <Subject 1> (S1) performs in the target video.")
        keeps.append("<Audio 1>: fully_preserved - the target video's soundtrack IS <Audio 1>, "
                     "reused directly, and the singer's lip movements, phrasing and breaths are "
                     "synchronized to its vocal line.")

    first = seg["shots"][0]
    who = (f"<Subject 1>" if chars else "the scene")
    if lead and lead in outfit_pic:
        who = f"<Subject 1> wearing <Subject {outfit_pic[lead]}>"
    # NAME the cast. This sentence caps the extras, but it used to say "They are the only people in
    # the video" straight after naming <Subject 1> alone - a plural with no antecedent, in the one
    # sentence that describes the whole segment, while a second person the later shots need was not
    # mentioned anywhere in it. Listing them removes the contradiction and introduces everyone the
    # segment actually uses before the shots start.
    only = ((" Only " + " and ".join(f"<Subject {chars.index(n) + 1}>" for n in chars)
             + " appear; nobody else.") if len(chars) > 1 else
            (" This is the only person in the video." if chars else ""))
    mode_tag = "[reference generation + audio reference]" if (audio_ref and lead) else "[reference generation]"
    sings = (f", singing <Audio 1> directly to the camera in a {first['framing']} framing, lips and "
             f"breathing synchronized to the vocal" if (audio_ref and lead) else "")
    def _stag(n):
        base_tag = f"<Subject {chars.index(n) + 1}>"
        return f"{base_tag} wearing <Subject {outfit_pic[n]}>" if n in outfit_pic else base_tag

    def _btag(n):
        """Bare subject tag. Used where the same sentence group has already said what they wear -
        the outfit is bound in the definitions, the retention list and the shot header, and each
        extra "wearing <Subject N>" is ~24 chars against a 7,000 ceiling."""
        return f"<Subject {chars.index(n) + 1}>"

    def _is_singer(n):
        return "singer" in str(by_name[n].get("role") or "").lower()

    def _vocalists(s):
        """Who actually carries the vocal in this shot: an explicit per-shot pick if there is one,
        else the cast members whose role is a singer, falling back to the first named character
        when no roles are set.

        The explicit pick exists because ROLE alone cannot express "both singers are in frame but
        only one of them is performing this line" - both of them have a singer role, so both were
        named as singing and both lip-synced, whatever the action prose said [segment 30]."""
        here = [n for n in s["characters"] if n in chars]
        pick = [n for n in (s.get("singers") or []) if n in here]
        if pick:
            return pick
        return [n for n in here if _is_singer(n)] or here[:1]

    def _distinguish(n):
        """A short "the woman with long straight black hair" phrase, for shots holding two people of
        the same sex. Their reference sheets differ, but the shot line is where H3 decides who is
        who, and two slim women with long dark hair swapped roles when it had only <Subject N> to go
        on. Hair first (the most visible separator), then eyes."""
        look = str((by_name[n].get("look") or by_name[n].get("appearance") or ""))
        g = _gender(by_name[n])
        who = "the woman" if g == "female" else ("the man" if g == "male" else "the one")
        hair = re.search(r"((?:long|short|shoulder-length|cropped|chin-length)[^,.;]*hair)", look, re.I)
        eyes = re.search(r"(\b\w+ eyes)", look, re.I)
        if hair:
            return f"{who} with {hair.group(1).strip()}"
        if eyes:
            return f"{who} with {eyes.group(1).strip()}"
        return ""

    def _stations(s):
        """Explicit per-person arrangement for a multi-person shot - the verified band render's key
        ingredient. "The band plays together" left the layout to H3, which duplicated players and
        moved an outfit between subjects (both segment-5 drafts, 2026-08-10); the one clean band
        render instead stationed everyone: singer at the mic, each player with their instrument on
        their own side, drummer behind. Same wording here."""
        here = [n for n in s["characters"] if n in chars]
        if len(here) < 2:
            return ""
        # EVERY person gets a DISTINCT place. Two people sharing "is beside them" is how segment 8
        # failed: an unplaced man next to an unplaced woman, so H3 chose for itself and merged them.
        # The spot order alternates SIDES (left, right, back-right, back-left) so that when the queue
        # below alternates genders, two similar-looking people never end up next to each other -
        # putting the guitarist "on the left" and Selene "at the back on the left" swapped the two
        # women outright: Selene played the Les Paul and the guitarist stood idle [OBSERVED 08-10].
        spots = ["at the center of the frame", "on the left", "on the right",
                 "at the back on the right", "at the back on the left", "furthest back"]
        # WHO HOLDS THE CENTRE.
        #  - Both singers in one shot: they take it TOGETHER, side by side, even when only one of
        #    them is singing this window (user rule 2026-08-10). That also keeps the two look-alike
        #    women apart, since the guitarist ends up out on a flank.
        #  - Otherwise the vocalist takes it, lip-syncing here or not: a band shot with the guitarist
        #    centre and the lead singer off to one side reads as the wrong band (segment 8).
        pair = [n for n in here if _is_singer(n)]
        centre, singer = [], None
        if len(pair) > 1:
            if s["lipsync"]:
                lead = next((n for n in _vocalists(s) if n in pair), pair[0])
                pair = [lead] + [n for n in pair if n != lead]
            a, b = pair[0], pair[1]
            # DO NOT stage the second one as "stands" when they are BOTH carrying the vocal here.
            # The lip-sync line right below names every vocalist as singing, so the old wording put
            # two contradictory instructions in one paragraph - "<Subject 2> stands SIDE BY SIDE"
            # then "<Subject 1> and <Subject 2> sing <Audio 1>" - and H3 followed the concrete
            # staging verb: the second singer just stood there through the duet [segment 22,
            # OBSERVED 2026-08-12]. Duet phrasing when both sing, the standing form only when one
            # of the two genuinely is not a vocalist in this shot.
            duet = s["lipsync"] and a in _vocalists(s) and b in _vocalists(s)
            if duet:
                # EACH gets their own side. Saying both are "at the centre of the frame" gives the
                # two of them one shared position, which leaves the anti-duplication tail ("each in
                # that one place") with nothing to pin: segment 22 came back with Bob rendered twice,
                # once on either side of Selene [OBSERVED 2026-08-12]. Everywhere else in this
                # staging block a person gets exactly one named spot, and that is what stops the
                # doubling - the pair still reads as centre-frame because they are shoulder to
                # shoulder and nobody else is there.
                duo = (f". Exactly one {_gender(by_name[a]) or 'person'} and one "
                       f"{_gender(by_name[b]) or 'person'} in frame, two people in total"
                       if len(here) == 2 else "")
                centre = [f"{_btag(a)} just LEFT of centre and {_btag(b)} just RIGHT of centre, "
                          f"shoulder to shoulder at one microphone, BOTH singing to camera, both "
                          f"mouths moving with the vocal{duo}"]
            else:
                verb = "sings at the microphone" if s["lipsync"] else "stands at the microphone"
                centre = [f"{_stag(a)} {verb} and {_stag(b)} stands SIDE BY SIDE immediately beside them, "
                          f"the two of them together at the centre of the frame, both facing the camera"]
            rest = [n for n in here if n not in (a, b)]
        else:
            # the VOCALIST takes the mic, not simply the first name listed: on segment 5 the writer
            # listed the guitarist first, which stationed her at the microphone singing while the
            # actual singer stood off to the side - contradicting the lip-sync line right after it
            singer = (_vocalists(s)[0] if s["lipsync"] and _vocalists(s)
                      else next((n for n in here if _is_singer(n) and n not in prop_pic), None))
            rest = [n for n in here if n != singer]
        rest.sort(key=lambda n: (0 if (n in prop_pic or (by_name[n].get("prop") or {}).get("desc")) else 1,
                                 here.index(n)))
        # INTERLEAVE GENDERS so consecutive spots (which alternate sides) separate look-alikes
        by_g = {}
        for n in rest:
            by_g.setdefault(_gender(by_name[n]) or "?", []).append(n)
        # Look-alikes are counted across EVERYONE in the shot, not just the flanking players: the
        # observed swap was between Selene (centre, beside Bob) and the guitarist (flank), and
        # computing this over `rest` alone left the two women never compared, so no warning was
        # emitted for the exact pair that kept trading places.
        all_g = {}
        for n in here:
            all_g.setdefault(_gender(by_name[n]) or "?", []).append(n)
        same = {n for lst in all_g.values() if len(lst) > 1 for n in lst}
        if len(by_g) > 1:
            queues = sorted((list(v) for v in by_g.values()), key=len, reverse=True)
            woven = []
            while any(queues):
                for q in queues:
                    if q:
                        woven.append(q.pop(0))
            rest = woven
        role_of = {}
        parts, si = list(centre), (1 if centre else 0)
        if centre:
            role_of[pair[0]] = "at the microphone"
            role_of[pair[1]] = "beside them at the microphone, holding nothing"
        if singer:
            parts.append(f"{_stag(singer)} " + ("sings at the center microphone" if s["lipsync"]
                                                else "stands at the center microphone"))
            role_of[singer] = "at the microphone"
            si = 1
        for n in rest:
            spot = spots[min(si, len(spots) - 1)]
            si += 1
            pr = by_name[n].get("prop") or {}
            tag = _stag(n) + (f", {_distinguish(n)}," if n in same and _distinguish(n) else "")
            if n in prop_pic:
                parts.append(f"{tag} plays <Subject {prop_pic[n]}> {spot}")
                role_of[n] = f"holding <Subject {prop_pic[n]}> {spot}"
            elif pr.get("desc") or pr.get("name"):
                # picture shed by the reference budget - still say WHICH instrument, so the person
                # is identified by what they hold rather than left interchangeable
                what = (pr.get("name") or "instrument").lower()
                parts.append(f"{tag} plays {'their ' + what if what else 'an instrument'} {spot}")
                role_of[n] = f"holding their {what} {spot}"
            else:
                parts.append(f"{tag} stands {spot}")
                role_of[n] = f"standing {spot}, holding nothing"
        out = " In this shot " + ", ".join(parts) + ", each in that one place."
        # NO-SWAP, spelled out for look-alikes. H3 kept each woman's identity AND wardrobe intact but
        # exchanged their ROLES - Selene ended up playing the guitar on the flank while the guitarist
        # stood at the microphone [OBSERVED 2026-08-10, both drafts]. Naming each one by her visible
        # difference alongside the job she is doing is the thing that was missing.
        pairs = [n for n in same if n in role_of and _distinguish(n)]
        if len(pairs) > 1:
            bits = [f"{_stag(n)} = {_distinguish(n)}, {role_of[n]}" for n in pairs]
            out += " Do NOT swap them: " + "; ".join(bits) + "."
        return out

    arrangement = _stations(first).replace(" In this shot ", " In the opening shot ", 1)
    summary = (f"{mode_tag} The target video shows {who} inside <Subject {env_pic}>{sings}, "
               f"framed {first['framing']} at the start.{only}{arrangement}")

    fr_txt = {"close": "a close-up", "medium": "a medium shot", "wide": "a wide shot"}
    lines = ["The target video uses a cinematic, photorealistic style with natural film-like exposure."]
    if len(chars) > 1:
        # Both segment-5 drafts failed exactly this way: a person rendered twice, and one subject
        # wearing another's outfit. State the constraint instead of hoping it is implied.
        lines.append("Each defined subject appears EXACTLY ONCE: nobody is duplicated, no face is reused "
                     "for another character, and no outfit or instrument moves to another person.")
    if chars:
        # a draft cut to Bob's reference sheet - grid, grey backdrop and all - for several seconds
        lines.append("No reference picture is ever shown: no character sheet, no panel grid or split "
                     "screen, no studio backdrop. Every frame is the live scene described below.")
    t0 = seg["start"]
    for i, (s, cut) in enumerate(zip(seg["shots"], seg["cuts"])):
        cut_dur = round(cut["end"] - cut["start"], 2)
        head = "[Shot 1]" if i == 0 else \
               f"[Shot {i + 1}] At {int((cut['start'] - t0) // 60):02d}:{(cut['start'] - t0) % 60:06.3f}, the camera cuts to"
        subj = " and ".join(_stag(n) for n in s["characters"] if n in chars) or "the scene"
        act = s["action"].rstrip(". ")
        anchor = (f" One continuous movement at natural real-world speed, filling the full "
                  f"{cut_dur:.0f} seconds.")
        cam = f" {H3_CAMERA_MOVES[s['camera']]}." if s["camera"] != "static" else \
              " The camera holds still."
        sing = ""
        if s["lipsync"] and audio_ref:
            # ONLY the vocalists sing. This used to name every character in the shot, so a 3-person
            # lip-sync shot told H3 that the guitarist and bassist were singing the lead too - an
            # open invitation to put the vocal on the wrong face.
            voc = _vocalists(s)
            # bare subject tags here, no "wearing <Subject N>": the outfit is already bound in the
            # definitions, the retention list and the shot header, and repeating it a fourth time in
            # the same paragraph cost ~120 chars a shot against a 7,000 ceiling one segment had
            # already crossed
            vtag = " and ".join(f"<Subject {chars.index(n) + 1}>" for n in voc) or subj
            quiet = [n for n in s["characters"] if n in chars and n not in voc]
            # plural agreement matters on a duet: "<Subject 1> and <Subject 2> (S1) sings" reads as
            # one of them singing, which is half of why the second singer stayed still
            sing = (f" {vtag} (S1) {'sing' if len(voc) > 1 else 'sings'} <Audio 1>, "
                    f"{'their mouth shapes' if len(voc) > 1 else 'mouth shapes'}, phrasing and breaths "
                    f"matching the vocal exactly, "
                    f"{'both faces' if len(voc) > 1 else 'face'} to camera in {fr_txt[s['framing']]}.")
            if quiet:
                sing += (" " + " and ".join(_stag(n) for n in quiet) +
                         (" does" if len(quiet) == 1 else " do") + " NOT sing: lips closed.")
        elif audio_ref and s["characters"]:
            # The segment carries <Audio 1>, so H3 will happily animate ANYONE on screen as the
            # singer: segment 5's band-wide cut had Bob mouthing a female vocal purely because he
            # was in frame. A cut that is not the lip-sync cut has to say so out loud.
            sing = (" Nobody in this shot is singing: lips stay closed and still, faces calm. The "
                    "song is only heard here, not performed to camera.")
        # NO-LONE-SINGER-ON-A-BARE-STAGE net (standing user rule; the first full script put Bob alone
        # on the stage for 40s across five consecutive cuts). A stage shot wide enough to show the
        # stage, carrying at most one named character, gets the rest of the band filled in as
        # BACKGROUND - text only, no reference pictures, so it costs nothing against H3_MAX_REFS and
        # no identity needs locking for players whose faces never resolve. Close-ups are exempt: the
        # empty stage is not in frame. A shot that already names 2+ characters is left alone.
        band = ""
        if (s["framing"] in ("wide", "medium") and len(s["characters"]) < 2
                and _H3_STAGE_RE.search((s.get("location") or "") + " " + s["scene"])):
            # never fill in a role the named character ALREADY plays. This asked for "a guitarist"
            # in the background of a shot whose only subject IS the guitarist, so her solo came back
            # with extra people playing guitar and bass behind her [segment 25, OBSERVED 2026-08-12].
            here_roles = " ".join(str(by_name[n].get("role") or "").lower()
                                  for n in s["characters"] if n in by_name)
            fill = [w for k, w in (("guitar", "a guitarist"), ("bass", "a bassist"))
                    if k not in here_roles]
            fill.append("a drummer at a kit")     # the kit is never the named subject's instrument
            band = (f" The rest of the band plays behind on the same stage - {' and '.join(fill)} "
                    "- set back in the dimmer depth of the frame and softly out of focus, their "
                    "faces never resolving. The stage is never empty behind the performer.")
        elif any(n in dropped_people for n in s["characters"]):
            # players squeezed out by the people cap stay in frame as the unresolved background band
            band = (" The other band members play their instruments further back in the frame, "
                    "softly out of focus, their faces never resolving.")
        stations = _stations(s)
        # who is NOT here: without this, a scene's other subjects bleed into every cut (the
        # segment-5 identity merges happened in the multi-person shot; this scoping is the
        # complementary guard for the other cuts). Named when a cut carries a subset.
        absent = [n for n in chars if n not in s["characters"]]
        gone = ""
        if absent and s["characters"]:
            gone = (" " + " and ".join(f"<Subject {chars.index(n) + 1}>" for n in absent) +
                    (" are" if len(absent) > 1 else " is") + " NOT in this shot and must not appear in it.")
        line = (f"{head} {fr_txt[s['framing']].capitalize()} of {subj} in <Subject {env_pic_of(s)}>: "
                f"{act}.{stations}{sing}{gone}{anchor}{cam} The environment behind holds completely still.{band}")
        if _H3_SKY_RE.search(s["scene"] + " " + s["action"]):
            line += " The sky holds fixed - no drifting stars, moon or clouds."
        lines.append(line)
    detailed = "\n\n".join(lines)

    sound = seg.get("soundscape") or "Quiet natural room tone and the soft ambience of the location."
    if audio_ref and lead:
        sound = f"The song from <Audio 1> carries the scene; beneath it only {sound[0].lower() + sound[1:]}"

    prompt = (f"subject_definitions:\n" + "\n".join(defs) + "\n\n"
              f"summary:\n{summary}\n\n"
              f"retention_analysis:\n" + "\n".join(keeps) + "\n\n"
              f"detailed_description:\n{detailed}\n\n"
              f"overall_soundscape:\n{sound}\n\n"
              f"non_diegetic_music:\nN/A")
    picture_map = {n: i + 1 for i, n in enumerate(chars)}
    return prompt, {"sheets": picture_map, "outfits": outfit_pic, "props": prop_pic,
                    "envs": env_map, "env": env_pic, "chars": len(prompt),
                    # nothing was checking this before, and a wording change quietly took one
                    # segment to 7,154 - past the documented ceiling, where the request is
                    # truncated or refused with no sign of it in the app
                    "over_limit": len(prompt) > H3_PROMPT_MAX}


# ---- voice-matched casting, enforced in CODE ----------------------------------------------
# The writer is told whose voice each section is (the section style markers are in its brief), and
# it still got the first two verses exactly backwards on a real script (2026-08-09): Bob mouthing
# the "tender female bell canto" verse and Selene the "warm male" one, apparently reasoning from the
# lyric's point of view (verse 1 addresses the departed, verse 2 answers) instead of the marker.
# Who sings is not a judgement call - the song says so - so it is decided here, not by the LLM.
# NOTE: word boundaries matter. "female" CONTAINS "male", so substring tests silently read every
# female marker as male too.
_VOICE_FEM = re.compile(r"\bfemale\b|\bwoman\b|\bsoprano\b|\bmezzo\b|\balto\b", re.I)
_VOICE_MALE = re.compile(r"\bmale\b|\bman\b|\btenor\b|\bbaritone\b|\bbass vocal\b", re.I)
_VOICE_DUET = re.compile(r"\bduet\b|\bboth\b|\bharmon|\btogether\b|\bunison\b", re.I)


def _voice_of(style):
    """The voice a section style marker calls for: "female" | "male" | "duet" | ""."""
    s = str(style or "")
    if _VOICE_DUET.search(s):
        return "duet"
    if _VOICE_FEM.search(s):
        return "female"
    if _VOICE_MALE.search(s):
        return "male"
    return ""


# ---- reconciling the two timelines -------------------------------------------------------------
# The arrangement's section durations (the Song tab) are a PLAN; ACE-Step does not render them
# literally, so the real track's sections sit somewhere else - measured on a real song, between
# -1.9s and +12.3s away. The segment grid comes from analysing the REAL audio, while the lyrics and
# the vocal markers come from the plan, and nothing reconciled the two: a handover the plan puts at
# 42.0s actually lands at ~40.2s, which is why segment 5's singer changed at the wrong moment.
# Fix: anchor the two timelines wherever they agree on a section label, then interpolate between
# anchors. Only agreeing labels become anchors, so a plan section the analysis cannot see (our two
# consecutive verses arrive as ONE "verse") is positioned by interpolation instead of being trusted.
_SEC_ALIAS = {"start": "intro", "end": "outro", "pre-chorus": "prechorus", "pre chorus": "prechorus"}


def _norm_section(name):
    n = str(name or "").strip().lower()
    return _SEC_ALIAS.get(n, n)


def _audio_sections(segments):
    """The REAL section spans, merged from the analysis label each segment carries."""
    out = []
    for s in segments or []:
        lab = _norm_section(s.get("section"))
        if out and out[-1][2] == lab:
            out[-1][1] = float(s.get("end") or 0)
        else:
            out.append([float(s.get("start") or 0), float(s.get("end") or 0), lab])
    return out


def _time_anchors(song, segments):
    """[(nominal_t, real_t)] where the plan and the analysis agree on a section boundary.

    Uses a longest-common-subsequence alignment of the two LABEL sequences, not a forward walk. A
    forward walk mis-pairs badly here: consecutive plan sections of the same kind arrive from the
    analysis as ONE section (our two verses come back as a single 14.1-80.3s "verse"), so a greedy
    matcher consumes that verse on the first one and pairs the second with the NEXT audio verse a
    minute later - which mapped a 42.0s handover to 103.5s. Runs of the same label are therefore
    collapsed on the plan side first, and sections the analysis never saw (pre-chorus, bridge) are
    simply left unmatched and positioned by interpolation."""
    groups, t = [], 0.0
    for s in (song or {}).get("sections") or []:
        lab = _norm_section(s.get("type"))
        if groups and groups[-1][1] == lab:
            pass                                  # same kind as the previous: one group, one anchor
        else:
            groups.append((t, lab))
        t += float(s.get("seconds") or 0)
    nom_total = t
    audio = _audio_sections(segments)
    if not (groups and audio):
        return []
    nl = [g[1] for g in groups]
    al = [a[2] for a in audio]
    # LCS table over the label sequences
    m, n = len(nl), len(al)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            dp[i][j] = 1 + dp[i + 1][j + 1] if nl[i] == al[j] else max(dp[i + 1][j], dp[i][j + 1])
    # Seeded with the origin, and every later pair must advance on BOTH axes. Without the seed a
    # first matched pair like (10.0, 0.0) survived and then prepending the origin left two anchors
    # sharing a real time - a flat span that mapped a whole plan section onto one instant.
    anchors, i, j = [(0.0, 0.0)], 0, 0
    while i < m and j < n:
        if nl[i] == al[j]:
            nt, at = groups[i][0], audio[j][0]
            if nt > anchors[-1][0] and at > anchors[-1][1]:
                anchors.append((nt, at))
            i, j = i + 1, j + 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    aud_total = audio[-1][1]
    if nom_total > anchors[-1][0] and aud_total > anchors[-1][1]:
        anchors.append((nom_total, aud_total))
    return anchors


def _map_time(anchors, t):
    """Nominal time -> real-audio time, piecewise linear between anchors."""
    if not anchors:
        return t
    if t <= anchors[0][0]:
        return anchors[0][1]
    for (n0, a0), (n1, a1) in zip(anchors, anchors[1:]):
        if t <= n1:
            span = n1 - n0
            return a0 + (a1 - a0) * ((t - n0) / span if span else 0.0)
    return anchors[-1][1] + (t - anchors[-1][0])


def _voice_windows(song, segments=None):
    """[{start, end, voice}] for every SUNG section. With `segments` (which carry the analysis
    labels) the windows are mapped onto the REAL audio timeline; without them they stay nominal."""
    anchors = _time_anchors(song, segments) if segments else []
    out, t = [], 0.0
    for s in (song or {}).get("sections") or []:
        dur = float(s.get("seconds") or 0)
        lyr = str(s.get("lyrics") or "").strip()
        sung = bool(lyr) and lyr.lower() != "instrumental"
        v = _voice_of(s.get("style"))
        if sung and v:
            out.append({"start": _map_time(anchors, t) if anchors else t,
                        "end": _map_time(anchors, t + dur) if anchors else t + dur,
                        "voice": v})
        t += dur
    return out


def _voice_at(windows, start, end):
    """The voice of whichever sung window overlaps [start, end] most; empty if none does."""
    best, best_ov = "", 0.0
    for w in windows:
        ov = min(end, w["end"]) - max(start, w["start"])
        if ov > best_ov:
            best, best_ov = w["voice"], ov
    return best


def _gender(c):
    g = str((c or {}).get("gender") or "").strip().lower()
    return "female" if g.startswith("f") else ("male" if g.startswith("m") else "")


# Recasting a shot has to carry the PRONOUNS with the name, or the prompt reads "Selene sings as he
# holds the railing" and the render gets a contradictory gender cue. Longest forms first; \b already
# protects "himself"/"herself" from the short patterns. "her" is both object and possessive: followed
# by a word it is possessive ("her face" -> "his face"), otherwise object ("toward her" -> "him").
_PRON = {
    ("male", "female"): [(r"\bhimself\b", "herself"), (r"\bhis\b", "her"), (r"\bhim\b", "her"), (r"\bhe\b", "she")],
    ("female", "male"): [(r"\bherself\b", "himself"), (r"\bhers\b", "his"),
                         (r"\bher\b(?=\s+[a-z])", "his"), (r"\bher\b", "him"), (r"\bshe\b", "he")],
}


def _swap_pronouns(text, frm, to):
    """Rewrite third-person pronouns from one gender to the other, preserving capitalization."""
    for pat, rep in _PRON.get((frm, to), []):
        text = re.sub(pat, lambda m, r=rep: r.capitalize() if m.group(0)[0].isupper() else r, text)
    return text


def enforce_voice_casting(segments, song, cast, grid=None):
    """Recast any lip-sync shot whose singer is the wrong voice for the section being sung, and
    fix the shot's prose so the names match. Duet windows and unmarked sections are left alone,
    as is any shot whose gender has no cast member to swap in. Records what changed on the
    segment as `voice_fixed` (the UI surfaces the count). Returns the list of fixes.

    `grid` is the WHOLE video's segments (or any [{start, end, section}] list) and is what maps the
    arrangement's nominal times onto the real audio - pass it whenever `segments` is only a slice,
    such as one writer batch or a single segment being recompiled, since section anchors cannot be
    derived from a fragment."""
    wins = _voice_windows(song, grid or segments)
    if not wins:
        return []
    singers, named = {}, []
    for c in sorted(cast or [], key=lambda x: 0 if "singer" in str(x.get("role") or "").lower() else 1):
        g, n = _gender(c), (c.get("name") or "").strip()
        if not (g and n):
            continue
        named.append(n)
        singers.setdefault(g, []).append(n)     # singers sort first, so [0] is that voice's lead
    fixes = []
    for seg in segments:
        cuts = seg.get("cuts") or [{"start": seg.get("start", 0), "end": seg.get("end", 0)}]
        for j, sh in enumerate(seg.get("shots") or []):
            if not sh.get("lipsync"):
                continue
            if sh.get("cast_locked"):
                # Hand-cast by the user: their ear outranks the inferred window. Both inputs here
                # are approximations (the section markers are nominal times mapped onto the real
                # audio), so when a person says "this shot is Bob" and the map says female, the
                # person wins. Without this the swap was undone on every recompile, silently.
                continue
            cut = cuts[j] if j < len(cuts) else cuts[0]
            need = _voice_at(wins, float(cut.get("start", 0)), float(cut.get("end", 0)))
            if need not in ("female", "male"):
                continue                       # duet or unmarked: the writer's choice stands
            right = (singers.get(need) or [None])[0]
            if not right:
                continue                       # nobody of that voice in the cast
            wrong = [n for n in sh.get("characters") or []
                     if n in named and n not in (singers.get(need) or [])]
            if not wrong or right in (sh.get("characters") or []):
                continue
            # pronouns only when ONE cast member is in the shot - otherwise a "she" might belong to
            # somebody else in frame and swapping it would misgender them instead
            solo = len([n for n in sh.get("characters") or [] if n in named]) == 1
            for w in wrong:
                sh["characters"] = [right if n == w else n for n in sh["characters"]]
                for k in ("action", "scene"):
                    if sh.get(k):
                        sh[k] = re.sub(rf"\b{re.escape(w)}\b", right, sh[k])
                        if solo:
                            sh[k] = _swap_pronouns(sh[k], "female" if need == "male" else "male", need)
                fixes.append(f"{cut.get('start', 0):.0f}s {w}->{right} ({need} part)")
            seen, uniq = set(), []
            for n in sh["characters"]:
                if n not in seen:
                    seen.add(n)
                    uniq.append(n)
            sh["characters"] = uniq
            seg["voice_fixed"] = (seg.get("voice_fixed") or []) + fixes[-len(wrong):]
    return fixes


H3_BATCH = 8            # segments per writer call (see generate_h3_script_grid)
H3_LOCATIONS = 6        # named locations targeted across a whole video (user call 2026-08-09: the
#   first script used 4 over four minutes and read as repetitive). Enforced as guidance in the
#   writer rules plus a per-batch budget in generate_h3_script_grid - not clamped in code, since
#   which places a song needs is a creative call, not an arithmetic one.


def generate_h3_script_grid(song, cast, provider, model, claude_model, grid, batch=H3_BATCH):
    """H3 hybrid script: segments from the audio grid -> writer fills content -> each segment gets
    its compiled full-references prompt attached (prompt, picture_map, env picture index).

    WRITTEN IN BATCHES (2026-08-09). A whole-video single call produces a ~30KB JSON reply, and
    that reply was arriving TRUNCATED through the CLI transport - two 8-minute runs died at the
    final parse ("Extra data", then a 919-byte fragment that started mid-object). Batching
    caps each reply at a few KB, and a failed batch is retried ONCE on its own instead of
    discarding the whole video. Continuity is preserved by telling each batch what came before
    (locations already established + the previous segment's last shot)."""
    segments = build_h3_segments(grid)
    # before writing: line the segments up with the vocals. Edges first (a handover too close to a
    # boundary to cut moves the boundary instead), then a cut on every handover left inside one.
    snap_segment_edges_to_handovers(segments, song)
    split_cuts_at_voice_handovers(segments, song)
    done, established = [], []
    for start in range(0, len(segments), max(1, batch)):
        chunk = segments[start:start + max(1, batch)]
        prev = done[-1] if done else None
        note = ""
        if established:
            note += ("\nLOCATIONS ALREADY ESTABLISHED (reuse these names verbatim where the story "
                     "returns to them): " + ", ".join(sorted(set(established))) + ".")
        # LOCATION BUDGET, paced across the batches. Without it a batch cannot know how much of the
        # whole-video allowance is left: the first script (written in one pass) settled into 4
        # locations for four minutes. The allowance ramps with position so the world builds up
        # instead of all six places existing by the end of batch 1.
        have = len(set(established))
        want_by_now = max(2, round(H3_LOCATIONS * (start + len(chunk)) / max(1, len(segments))))
        may_add = max(0, want_by_now - have)
        # A FLOOR, not just a ceiling. First cut of this said "introduce at most N new locations",
        # which the writer satisfied by introducing none: the whole-video "about 6" lost to the
        # per-batch continuity pressure and the script came back with 4 locations again, exactly the
        # repetition this was meant to fix. Stating the number as a quota is what actually moves it.
        note += (f"\nLOCATION BUDGET: {have} named location(s) established so far; about "
                 f"{H3_LOCATIONS} across the whole video, and about {want_by_now} should exist by "
                 f"the end of THIS batch. " + (
                     f"So introduce EXACTLY {may_add} NEW named location(s) here - this is a floor "
                     f"as well as a ceiling, not an option - and set every other shot in "
                     + ("them." if not have else "the locations already established.")
                     if may_add else
                     "That target is already met, so introduce NO new locations in this batch - "
                     "reuse the established names."))
        if prev:
            last = prev["shots"][-1]
            note += (f"\nCONTINUING FROM: segment {start} ended at {prev['end']:.1f}s in "
                     f"'{last.get('location') or 'an unnamed place'}' - {last.get('action','')[:120]}")
        note += (f"\nThese are segments {start + 1}-{start + len(chunk)} of {len(segments)} for the "
                 f"whole video; number your JSON array from 1 for THIS batch only "
                 f"({len(chunk)} objects).")
        system, prompt = build_h3_grid_prompt(song, cast, chunk, grid=segments)
        prompt += note
        part = None
        for attempt in (1, 2):
            try:
                text = llm_mod.complete(provider, model, system, prompt, claude_model, timeout=600)
                cand = parse_h3_segments(text, chunk)
                # a SHORT result means the reply was cut off / salvaged - retry once for a full
                # batch, but keep the salvaged content rather than failing if the retry is no better
                filled = sum(1 for s in cand if s["shots"][0]["scene"])
                if filled >= len(chunk) or attempt == 2:
                    part = cand if (part is None
                                    or filled >= sum(1 for s in part if s["shots"][0]["scene"])) else part
                    break
                part = cand
            except Exception:
                if attempt == 2 and part is None:
                    raise
        # per batch, not at the end: the next batch's continuity note quotes the previous segment's
        # last shot, so it should quote the CORRECTED singer
        # `segments` (the whole video) supplies the section anchors - a single batch cannot
        enforce_voice_casting(part, song, cast, grid=segments)
        done.extend(part)
        for s in part:
            for sh in s["shots"]:
                if (sh.get("location") or "").strip():
                    established.append(sh["location"].strip())
    segs = done
    for seg in segs:
        seg["prompt"], refs = compile_h3_prompt(seg, cast, audio_ref=seg["lipsync"])
        seg["picture_map"] = refs["sheets"]
        seg["outfit_map"] = refs["outfits"]
        seg["prop_map"] = refs["props"]
        seg["env_map"] = refs["envs"]
        seg["env_picture"] = refs["env"]
    return segs
