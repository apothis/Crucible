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
        # the section STYLE carries the vocal-part marker ("tender female bell canto", "warm male",
        # "duet swell") - essential for voice-matched casting, so surface it to the writer
        style = (s.get("style") or "").strip()
        stag = f" ({style})" if style else ""
        head = f"  [{t}-{t + dur}s] {s.get('type') or 'section'}{stag}:"
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
H3_SEG_MAX_S = 10.5              # practical ceiling MEASURED 2026-08-09: at the trained max
#   (15.08s) a heavy-reference render (5 pictures + audio) collapses from the tail - audio
#   diverges from ~10s (envelope r 0.415 vs 0.975) and the image degrades to noise by ~14s;
#   the IDENTICAL payload at 10.1s is clean end to end. 11-14s is unexplored territory.
#   362 frames stays as the hard clamp in h3_seg_seconds for explicit long renders.
H3_CAMERA_MOVES = {
    # writer vocabulary -> the author's exact phrasing (small amplitude / slow speed is the pace-safe
    # envelope measured in Phase 1; see the pace rules)
    "static": "The camera holds still",
    "push in": "The camera pushes in with small amplitude at slow speed",
    "pull back": "The camera pulls back with small amplitude at slow speed",
    "truck left": "The camera trucks left with small amplitude at slow speed",
    "truck right": "The camera trucks right with small amplitude at slow speed",
    "arc left": "The camera arcs left around the subject with small amplitude at slow speed",
    "arc right": "The camera arcs right around the subject with small amplitude at slow speed",
    "tilt up": "The camera tilts up with small amplitude at slow speed",
    "crane up": "The camera cranes up with small amplitude at slow speed",
}
_H3_SKY_RE = re.compile(r"\b(sky|skies|cloud|clouds|stars?|starlit|moon|moonlit|milky way|sunset|"
                        r"sunrise|dawn|dusk|horizon|aurora|night sky)\b", re.I)


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


def build_h3_grid_prompt(song, cast, segments):
    """Writer prompt for the hybrid model: per SEGMENT the LLM chooses "single" or "scene" and fills
    creative content only (timing is fixed by the segment windows; scene cut times come from the
    segment's own audio-grid cuts). The environment may now CONTAIN people (H3's subject-swap
    replaces them - verified); it must still be environment-LED prose, not a person portrait."""
    summary, total = _song_summary(song)
    title = song.get("title") or "Untitled"
    named = [c for c in cast if c.get("name")]
    cast_txt = _fmt_cast(named) if named else "  (no fixed cast - lean scenic/atmospheric)"

    def _win(i, g):
        cuts = ""
        if len(g["cuts"]) > 1:
            cuts = "  internal cuts at " + ", ".join(f"{c['start']:.2f}s" for c in g["cuts"][1:])
        return f"  Segment {i + 1}: [{g['start']:.2f}-{g['end']:.2f}s] ({g['section']}){cuts}"
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
  "scene": "<the ENVIRONMENT, described as rich flowing prose: location, surfaces, materials, light
    sources, weather, depth layers. Environment-LED (not a person portrait), but people ARE allowed
    in it when the story wants them - the render replaces/controls who appears.>",
  "framing": "close" | "medium" | "wide",
  "action": "<what the subject does: EXACTLY ONE continuous motion or performance for the whole shot,
    described plainly. No compound sequences, no 'then'.>",
  "camera": "static" | "push in" | "pull back" | "truck left" | "truck right" | "arc left" |
            "arc right" | "tilt up" | "crane up",
  "costume": "<what named characters wear; '' if their reference wardrobe>",
  "characters": [<named cast in this shot; [] if none>],
  "lipsync": <true when the lead singer sings the lyrics ON CAMERA in this shot>}}

HARD RULES:
- VOICE-MATCHED CASTING: the singer ON CAMERA must be the singer whose VOICE is actually
  singing in that window. The section markers in the song brief say whose voice it is (e.g.
  "female bell canto" = the female singer, "warm male" = the male singer, "duet" = both).
  NEVER show a singer mouthing a part sung in the other voice. On duet windows both singers
  may appear - together in one shot or intercut - each singing their own lines.
- SUNG SECTIONS focus on the LEAD SINGER: while lyrics are being sung, most segments should be
  the singer performing (lip-sync singles). B-roll IS allowed during singing - use an occasional
  "scene" segment to breathe (imagery tied to the lyric being sung) - but the singer carries
  sung sections; never let B-roll dominate a sung section. Instrumental windows are where
  B-roll and narrative belong.
- LIP-SYNC segments are ALWAYS "kind": "single" (one shot, close or medium framing, the singer
  performing TO CAMERA, mouth visible - never turned away, never silhouette, never wide).
- "scene" is for connected NON-vocal cuts (B-roll runs, narrative beats, establishing sequences):
  give it one shot per listed internal cut, visually connected (same world, evolving viewpoint).
- ONE motion per shot. The action is a single continuous thing a real person/scene does at
  real-world speed. Never describe multiple movements, weather changes or time passing.
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


def parse_h3_segments(text, segments):
    """Validate the writer's JSON against the fixed segment windows. Enforces (hard, in code):
    lipsync => kind single + close/medium; scene shot-count == internal cut count (clamped 1-4);
    camera vocabulary; one shot minimum. Returns the segment list with content attached."""
    m = re.search(r"\[.*\]", text, re.S)
    raw = m.group(0) if m else text
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("writer output is not a JSON array")
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
            if lipsync and framing == "wide":
                framing = "medium"
            camera = str(s.get("camera") or "static").strip().lower()
            if camera not in H3_CAMERA_MOVES:
                camera = "static"
            stype = s.get("type") if s.get("type") in SHOT_TYPES else "broll"
            chars = [str(x).strip() for x in (s.get("characters") or [])
                     if x and str(x).strip() not in ("[]", "none", "None", "-", "")]
            shots.append({"type": stype, "framing": framing, "lipsync": lipsync, "camera": camera,
                          "scene": str(s.get("scene") or "").strip(),
                          "action": str(s.get("action") or "").strip(),
                          "costume": str(s.get("costume") or "").strip(),
                          "characters": chars})
        if not shots:
            shots = [{"type": "broll", "framing": "wide", "lipsync": False, "camera": "static",
                      "scene": "", "action": "", "costume": "", "characters": []}]
        lipsync_any = any(s["lipsync"] for s in shots)
        if lipsync_any or len(seg["cuts"]) == 1:
            kind = "single"
        if kind == "single":
            shots = shots[:1]
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
    Returns (prompt_text, picture_map)."""
    by_name = {c.get("name"): c for c in cast if c.get("name")}
    chars = []
    for s in seg["shots"]:
        for n in s["characters"]:
            if n in by_name and n not in chars:
                chars.append(n)
    outfits = [n for n in chars if (by_name[n].get("costume") or {}).get("still_id")]
    outfit_pic = {n: len(chars) + 1 + i for i, n in enumerate(outfits)}
    # props (instruments/objects) follow the outfits in picture order - same verified mechanic
    props = [n for n in chars if (by_name[n].get("prop") or {}).get("still_id")]
    prop_pic = {n: len(chars) + len(outfits) + 1 + i for i, n in enumerate(props)}
    env_pic = len(chars) + len(outfits) + len(props) + 1
    lead = chars[0] if chars else None

    defs, keeps = [], []
    for i, n in enumerate(chars):
        c = by_name[n]
        look = (c.get("look") or c.get("description") or "").strip()
        looktxt = f" - {look}" if look else ""
        if n in outfit_pic:
            # dressed via the outfit Subject: the sheet is FACE/HAIR ONLY, wardrobe comes from
            # the costume still (the verified wardrobe-not-used + fully_preserved-outfit pair)
            co = by_name[n]["costume"]
            op = outfit_pic[n]
            defs.append(f"<Subject {i + 1}> is {n}, the person from <Picture {i + 1}>{looktxt}. "
                        f"<Picture {i + 1}> is a character reference sheet showing {n} from several "
                        f"views; preserve the face and hair exactly. {n}'s wardrobe in "
                        f"<Picture {i + 1}> is NOT used in the target video.")
            defs.append(f"<Subject {op}> is the OUTFIT from <Picture {op}>: "
                        f"{(co.get('desc') or 'the outfit').rstrip('. ')}. <Picture {op}> shows it "
                        f"on a headless dress form; only the garment is referenced.")
            keeps.append(f"<Subject {i + 1}>: fully_preserved - {n}'s identity, face and hairstyle "
                         f"remain exactly consistent with <Picture {i + 1}> throughout; the clothing "
                         f"comes from <Picture {op}>, not from <Picture {i + 1}>.")
            keeps.append(f"<Subject {op}>: fully_preserved - the outfit's cut, color and fabric "
                         f"remain exactly consistent with <Picture {op}>, now worn by "
                         f"<Subject {i + 1}> and fitting naturally.")
        else:
            costume = next((s["costume"] for s in seg["shots"] if s["costume"] and n in s["characters"]), "")
            wear = f", wearing {costume}" if costume else ""
            defs.append(f"<Subject {i + 1}> is {n}, the person from <Picture {i + 1}>{looktxt}{wear}. "
                        f"<Picture {i + 1}> is a character reference sheet showing {n} from several views; "
                        f"preserve the face, hair and wardrobe exactly.")
            keeps.append(f"<Subject {i + 1}>: fully_preserved - {n}'s identity, face, hairstyle and "
                         f"wardrobe remain exactly consistent with <Picture {i + 1}> throughout.")
    for n in props:
        pr = by_name[n]["prop"]
        pp = prop_pic[n]
        pi = chars.index(n) + 1
        defs.append(f"<Subject {pp}> is the {(pr.get('name') or 'PROP').upper()} from <Picture {pp}>: "
                    f"{(pr.get('desc') or 'the object').rstrip('. ')}. <Picture {pp}> shows it alone; "
                    f"only the object is referenced. <Subject {pi}> plays and holds it.")
        keeps.append(f"<Subject {pp}>: fully_preserved - its shape, materials, colors and hardware "
                     f"remain exactly consistent with <Picture {pp}>, played by <Subject {pi}>.")
    env_scene = next((s["scene"] for s in seg["shots"] if s["scene"]), "the environment").rstrip(". ") + "."
    defs.append(f"<Subject {env_pic}> is the ENVIRONMENT from <Picture {env_pic}>: {env_scene} "
                f"Only the environment and its objects are referenced from <Picture {env_pic}>. "
                f"Any person who appears in <Picture {env_pic}> is NOT part of the target video "
                f"and does not appear in it.")
    keeps.append(f"<Subject {env_pic}>: reference - the environment, its architecture, objects and "
                 f"lighting remain recognizable from <Picture {env_pic}>; any person shown in "
                 f"<Picture {env_pic}> is replaced by the defined subjects and does not appear.")
    if audio_ref and lead:
        defs.append(f"<Audio 1> is the song that <Subject 1> (S1) performs in the target video.")
        keeps.append("<Audio 1>: fully_preserved - the target video's soundtrack IS <Audio 1>, "
                     "reused directly, and the singer's lip movements, phrasing and breaths are "
                     "synchronized to its vocal line.")

    first = seg["shots"][0]
    who = (f"<Subject 1>" if chars else "the scene")
    if lead and lead in outfit_pic:
        who = f"<Subject 1> wearing <Subject {outfit_pic[lead]}>"
    only = " They are the only people in the video." if len(chars) > 1 else \
           (" This is the only person in the video." if chars else "")
    mode_tag = "[reference generation + audio reference]" if (audio_ref and lead) else "[reference generation]"
    sings = (f", singing <Audio 1> directly to the camera in a {first['framing']} framing, lips and "
             f"breathing synchronized to the vocal" if (audio_ref and lead) else "")
    summary = (f"{mode_tag} The target video shows {who} inside <Subject {env_pic}>{sings}, "
               f"framed {first['framing']} at the start.{only}")

    fr_txt = {"close": "a close-up", "medium": "a medium shot", "wide": "a wide shot"}
    lines = ["The target video uses a cinematic, photorealistic style with natural film-like exposure."]
    t0 = seg["start"]
    for i, (s, cut) in enumerate(zip(seg["shots"], seg["cuts"])):
        cut_dur = round(cut["end"] - cut["start"], 2)
        head = "[Shot 1]" if i == 0 else \
               f"[Shot {i + 1}] At {int((cut['start'] - t0) // 60):02d}:{(cut['start'] - t0) % 60:06.3f}, the camera cuts to"
        def _stag(n):
            base_tag = f"<Subject {chars.index(n) + 1}>"
            return f"{base_tag} wearing <Subject {outfit_pic[n]}>" if n in outfit_pic else base_tag
        subj = " and ".join(_stag(n) for n in s["characters"] if n in chars) or "the scene"
        act = s["action"].rstrip(". ")
        anchor = (f" The movement unfolds at a natural real-world pace, taking the full "
                  f"{cut_dur:.0f} seconds, exactly as it would in real time.")
        cam = f" {H3_CAMERA_MOVES[s['camera']]}." if s["camera"] != "static" else \
              " The camera holds still."
        sing = ""
        if s["lipsync"] and audio_ref:
            sing = (f" {subj} (S1) sings <Audio 1> with genuine expression, mouth shapes, phrasing "
                    f"and breaths matching the vocal exactly, face to camera and clearly visible in "
                    f"{fr_txt[s['framing']]}.")
        line = (f"{head} {fr_txt[s['framing']].capitalize()} of {subj} in <Subject {env_pic}>: "
                f"{act}.{sing}{anchor}{cam} The environment behind holds completely still.")
        if _H3_SKY_RE.search(s["scene"] + " " + s["action"]):
            line += (" The sky holds fixed: stars, moon and clouds keep their positions with no "
                     "drift at all, and there is no cloud movement anywhere.")
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
    return prompt, {"sheets": picture_map, "outfits": outfit_pic, "props": prop_pic, "env": env_pic}


def generate_h3_script_grid(song, cast, provider, model, claude_model, grid):
    """H3 hybrid script: segments from the audio grid -> writer fills content -> each segment gets
    its compiled full-references prompt attached (prompt, picture_map, env picture index)."""
    segments = build_h3_segments(grid)
    system, prompt = build_h3_grid_prompt(song, cast, segments)
    text = llm_mod.complete(provider, model, system, prompt, claude_model, timeout=600)
    segs = parse_h3_segments(text, segments)
    for seg in segs:
        seg["prompt"], refs = compile_h3_prompt(seg, cast, audio_ref=seg["lipsync"])
        seg["picture_map"] = refs["sheets"]
        seg["outfit_map"] = refs["outfits"]
        seg["prop_map"] = refs["props"]
        seg["env_picture"] = refs["env"]
    return segs
