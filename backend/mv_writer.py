"""Claude music-video script writer for the H3 scene-chain pipeline (docs/MV_H3_REBUILD_PLAN.md 3.3).

Division of labour, the same as every writer that has held up here: CODE owns timing and the
compiled prompt; CLAUDE owns the creative content. Two passes:

  pass 1 (scenes): Claude reads the MEASURED section/line table (lyricalign on the actual take,
          never the nominal arrangement) and returns the whole video's plan: concept, look, one
          wardrobe, and scenes = contiguous runs of sections, each with a setting, a performance
          mode (sing to camera / sing away / no singing), a camera, framing, lighting, mood.
          B-roll scenes are asked for explicitly - a music video is not a lip-sync reel.
  code:   turns each scene's window into a chain plan on the 17k+5 grid (mv_h3.chain_plan) and
          lists the lyric lines audible inside every clip window, with their offsets.
  pass 2 (clips): per scene, Claude writes each clip's prose (summary, detailed_description,
          overall_soundscape, non_diegetic_music) from its window and words, continuing clip N
          from clip N-1. Code wraps it in the fixed six-section Ref2VA boilerplate (subject
          definitions, retention), adds the style opening, and checks the hard rules.

Rules baked in from the measured work (see the plan and memory): no <Audio 1> - the song is
locked in the latent; one described motion per shot with a real-world duration anchor; pin the
sky when it is visible; pin the hands; readable mouth on the vocal phrases that matter; a
generation seam is not a cut; references for one subject must agree, so the prompt never names
a detail only one reference shows. One lead (the singer) and no band; other people only where a
scene lists them in `others` (or `anchor_figure`), as anonymous prose figures (user 2026-10-28).

Plain ASCII only.
"""
import json
import math
import re

from . import llm as llm_mod
from . import lyricalign
from .mv_h3 import CHAIN_CONTEXT, CHAIN_FRAMES, chain_plan
from .video import H3_FPS, _h3_frames

WRITER_MODEL = "claude-opus-5-5"      # user 2026-10-27: the latest Opus
WRITER_EFFORT = "medium"              # user 2026-10-27
WRITER_TIMEOUT = 900
MIN_SINGLE_FRAMES = 124               # a scene shorter than one chain clip renders as one clip of >= 5.17 s
GAP_IS_INSTRUMENTAL = 2.0             # an unsung gap longer than this becomes its own section


# ----------------------------------------------------------------------------- song facts

def parse_lyrics(raw):
    """Tagged lyrics (Suno / Song-tab style) -> blocks [{type, seconds: 0, lyrics}] for lyricalign.
    A line made only of [tags] opens a section; its first tag (minus a trailing number) is the
    type. Inline tags such as [Female Vocal] are stripped from lyric lines."""
    blocks, cur = [], None
    for line in (raw or "").split("\n"):
        s = line.strip()
        if not s:
            continue
        tags = re.findall(r"\[([^\]]+)\]", s)
        bare = re.sub(r"\[[^\]]+\]", "", s).strip()
        if tags and not bare:
            cur = {"type": re.sub(r"\s*\d+$", "", tags[0]).strip(), "seconds": 0, "lyrics": ""}
            blocks.append(cur)
            continue
        if cur is None:
            cur = {"type": "Verse", "seconds": 0, "lyrics": ""}
            blocks.append(cur)
        cur["lyrics"] += bare + "\n"
    return blocks


def song_facts(audio_path, lyrics_raw, duration, cache_dir=None, align_path=None):
    """Measured facts about THIS take: the aligned lines (real seconds, with cover) and a
    contiguous section table covering 0..duration. Sung sections come from the alignment;
    unsung gaps longer than GAP_IS_INSTRUMENTAL become instrumental sections (intro, solo,
    outro, breaks). Small gaps are absorbed into the preceding section."""
    blocks = parse_lyrics(lyrics_raw)
    # align_path: the isolated vocal stem when one is cached (dense mixes bury the words for
    # Whisper; the stem does not). Its cache key is its own basename, so both stay cached.
    align = lyricalign.align_song(align_path or audio_path, blocks, cache_dir=cache_dir)
    sung = []
    prev_end = 0.0
    all_lines = lyricalign._lyric_stream(blocks)[2]
    for sec in align["sections"]:
        if sec.get("start") is None:
            continue
        lines = [{"start": l["start"], "end": l["end"], "text": l["text"], "cover": l["cover"],
                  **({"echo": l["echo"]} if l.get("echo") else {}),
                  **({"word_starts": l["word_starts"]} if l.get("word_starts") else {})}
                 for l in sec["lines"] if l.get("start") is not None]
        if align_path and sec["lines"]:
            # the aligner drops unmatched lines from its sections; the block's own lines before
            # the first matched one are the opening lines Whisper missed
            first_idx = min(l["index"] for l in sec["lines"])
            missing = [l for l in all_lines if l["block"] == sec["index"] and l["index"] < first_idx]
            lead = stem_timed_lead(align_path, missing, min(l["start"] for l in sec["lines"]), prev_end)
            if lead:
                lines = lead + lines
                sec = dict(sec, start=lead[0]["start"])
        prev_end = sec["end"]
        sung.append({"type": sec["type"], "start": max(0.0, sec["start"] - 0.5),
                     "end": min(duration, sec["end"] + 0.3), "lines": lines, "vocal": True})
    table, t = [], 0.0
    for s in sung:
        if s["start"] - t > GAP_IS_INSTRUMENTAL:
            kind = "Intro" if not table else "Instrumental"
            table.append({"type": kind, "start": round(t, 3), "end": round(s["start"], 3),
                          "lines": [], "vocal": False})
        elif table:
            table[-1]["end"] = round(s["start"], 3)
        else:
            s["start"] = 0.0
        table.append(s)
        t = s["end"]
    if duration - t > GAP_IS_INSTRUMENTAL:
        table.append({"type": "Outro", "start": round(t, 3), "end": round(duration, 3),
                      "lines": [], "vocal": False})
    elif table:
        table[-1]["end"] = round(duration, 3)
    for i, s in enumerate(table):
        s["index"] = i
        s["start"], s["end"] = round(float(s["start"]), 3), round(float(s["end"]), 3)
    song_levels(audio_path, table)
    return {"duration": duration, "cover": align.get("cover"), "words_matched": align.get("words_matched"),
            "words_total": align.get("words_total"), "sections": table, "blocks": blocks,
            "aligned_on": "vocals" if align_path else "mix"}


STEM_HOP = 0.25          # seconds per vocal-stem frame
STEM_ON_DB = 30.0        # a frame is voiced within this many dB of the stem's loudest frame
STEM_BRIDGE = 0.75       # unvoiced runs up to this long (breaths) do not end a vocal run


def stem_timed_lead(stem_path, lead, first, floor=0.0):
    """Whisper can miss a section's OPENING lines (SSBS verse 1, 2026-10-29: the first two lines
    unmatched, so the verse 'began' 5 s after her voice did and the cut to her came late). For
    the unmatched opening lines `lead` of a section whose first matched line starts at `first`,
    the vocal stem says where the voice really comes in: walk back from `first` while the stem
    stays voiced (breaths bridged), not before `floor` (the previous section's end). The missing
    lines are spread evenly from there to `first`, cover 0, timed 'stem'. Returns those lines, or
    [] when nothing is missing, nothing earlier is voiced, or the stem cannot be read."""
    if not lead:
        return []
    first = float(first)
    try:
        import numpy as np
        import soundfile as sf
        x, sr = sf.read(stem_path, always_2d=True)
    except Exception:
        return []
    x = x.mean(axis=1)
    hop = int(STEM_HOP * sr)
    n = len(x) // hop
    if n == 0:
        return []
    db = 20 * np.log10(np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1)) + 1e-9)
    voiced = db > db.max() - STEM_ON_DB
    i = min(n - 1, int(first / STEM_HOP))
    lo = int(max(0.0, floor) / STEM_HOP)
    gap = 0
    entry = None
    while i > lo:
        i -= 1
        if voiced[i]:
            entry, gap = i, 0
        else:
            gap += 1
            if gap * STEM_HOP > STEM_BRIDGE:
                break
    if entry is None or first - entry * STEM_HOP < 1.0:
        return []
    t0 = entry * STEM_HOP
    step = (first - t0) / len(lead)
    return [{"start": round(t0 + k * step, 3), "end": round(t0 + (k + 1) * step, 3), "text": l["text"],
             "cover": 0.0, "timed": "stem"} for k, l in enumerate(lead)]


LEVEL_HOP = 0.5          # seconds per loudness frame
LOUD_BAND = 2.5          # a section this close to the song's median dB (or above) is loud
LEVEL_BAND = 5.0         # this far below the median, a section is quiet
QUIET_DROP = 6.0         # dB below the median, held QUIET_MIN seconds, is a quiet span inside a section
QUIET_MIN = 2.0


def song_levels(audio_path, sections):
    """Measured loudness per section, from the audio (not the style text: measured 2026-10-29, a
    style that said 'hushed opening' had the full band from 2.5 s). Adds to each section
    level_db (mean RMS dB), level (loud | mid | quiet against the song's median) and
    quiet_spans [[a, b], ...]. Plain numpy on the Mac; on any read error the sections are left
    as they were."""
    try:
        import numpy as np
        import soundfile as sf
        x, sr = sf.read(audio_path, always_2d=True)
        x = x.mean(axis=1)
        hop = int(sr * LEVEL_HOP)
        n = len(x) // hop
        db = 20 * np.log10(np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(axis=1)) + 1e-9)
    except Exception:
        return sections
    med = float(np.median(db))
    k = max(1, int(round(2.0 / LEVEL_HOP)))
    smooth = np.convolve(db, np.ones(k) / k, mode="same")
    for s in sections:
        i0, i1 = int(s["start"] / LEVEL_HOP), max(int(s["start"] / LEVEL_HOP) + 1, int(s["end"] / LEVEL_HOP))
        seg = db[i0:i1]
        if not len(seg):
            continue
        lvl = float(seg.mean())
        s["level_db"] = round(lvl, 1)
        s["level"] = "loud" if lvl >= med - LOUD_BAND else ("quiet" if lvl < med - LEVEL_BAND else "mid")
        spans, run = [], None
        for i in range(i0, min(i1, len(smooth))):
            low = smooth[i] < med - QUIET_DROP
            if low and run is None:
                run = i
            if (not low or i == min(i1, len(smooth)) - 1) and run is not None:
                if (i - run) * LEVEL_HOP >= QUIET_MIN:
                    spans.append([round(run * LEVEL_HOP, 1), round(i * LEVEL_HOP, 1)])
                run = None
        s["quiet_spans"] = spans
    return sections


# ----------------------------------------------------------------------------- pass 1: scenes

SCENE_SYSTEM = """You are the director of a music video for one song, writing for a video model (MiniMax H3) that renders each scene as one continuous take of up to 15 seconds, chained into longer takes when needed, with the real song locked to the picture. You return STRICT JSON only - no prose before or after it, no markdown fences.

Fixed facts about this production:
- The singer is the one lead and the only person the references define. There is no band: never a band, never other musicians, never a performance crowd by default. Other people are allowed where the story needs them (a landlord, a fisherman on the far quay, a child at a window, mourners at a grave): list them in the scene's "others" field, anonymous and few, each with a place in frame and their own ordinary business, and write null when the scene needs nobody. People never act in unison (measured: three men written as "one slow movement together" rendered as a synchronised puppet show); each does a different thing, staggered in time. Extras do SMALL ordinary things only: walking, standing, hauling, drinking, talking, coiling a rope. Never a large body transition for an extra: no climbing, jumping, stepping down into or out of a boat, mounting, falling; and only actions the location's geometry allows (the harbour's boats are moored off the quay on their ropes, so nobody can step into one). Measured: "one steps down into the nearest boat" rendered as a man flying through the air onto a boat that was too far away to reach. Most scenes need nobody; the video is about her.
- Her identity and wardrobe come from two reference photographs described to you. One wardrobe for the whole video; you may add weathering (rain, dust, wind in the hair) but not change garments.
- She lip-syncs where it serves the video, not in every scene. A good music video for this kind of song mixes performance with cinematic B-roll: environment, symbolic imagery, her moving through a place without singing, details, wide shots. You decide the mix from the song's content and its genre.
- Each scene is ONE continuous take (no cuts inside a scene). Scene changes are hard cuts. Any take SHE is in (singing or not) runs at most 15 seconds, one rendered clip (measured: 4 of 5 scenes of hers that chained a second clip re-invented the shot inside it: a second copy of her, a dissolve to a close-up, a ghost building, a jump to a run; her one-clip scenes held); an environment-only B-roll take (no person) may run up to 45 seconds; nothing shorter than 5 seconds. A long passage she is in is therefore two or more scenes, cut between them.
- SHOT PALETTE, by what has rendered reliably (measured over two songs); build the video from the reliable types:
  (a) RELIABLE: her singing to camera at medium close-up or close-up (lip-sync and identity hold); her standing at medium or medium wide while wind, hair, a head turn or a small camera move carries the motion; her small in a wide or extreme wide shot ONLY if she stays in place (stands, turns, lifts her arms) for one clip; environment-only plates of a place, where LARGE camera moves are safe (push, truck, crane along the still's own lines); townspeople and extras in their places doing ordinary things.
  (b) UNRELIABLE, never write: her walking any distance along a path or road (she approaches the lens or doubles), her travelling between places inside a scene, a scene of hers longer than one clip.
- MOTION comes from the camera on the B-roll plates, from the cutting, from weather and light, and from wind, hair and small gestures on her; not from her travelling. A static camera is a choice for a held moment, never the default: a video of locked-off shots is dull.
- PACE FOLLOWS THE MEASURED LEVEL of each section (listed with the sections; it is measured from the audio, so trust it over the style text): a loud section is cut faster, with shorter scenes (5-9 seconds) and more setups, and its B-roll uses large camera moves; a quiet section or quiet span holds longer scenes with small moves. A loud intro is not a slow establishing shot: it is several plates cut to the music.
- SECTION ROLES (adapt to the song, do not force): intro = plates of the world cut to its energy, one shot revealing her, ending on the first-line setup; verse = her performance with cutaways to what the lyric names; pre-chorus = the world reacting (people, places); chorus = performance-heavy, the fastest cutting, two or three setups; a long instrumental or solo = a montage of plates and people that carries a story beat, never one long take of her; bridge = a contrast in time, place or light; outro = aftermath plates and a last shot of her or a return to the opening image.
- AN ARC AND PAYOFFS: light, weather or time of day progress across the sections and the video says how (the `arc` field). Objects the lyrics name (a coat, a fiddle, a boat, a letter) are shown early and shown again near the end, changed (empty, left behind, gone): those are the payoff shots.
- FRAME 0 MATCHES THE SCENE: an environment plate starts from its location's still unless the scene's light, weather, time of day or people differ from that still, in which case the scene needs its own opening still (measured: a storm still under a calm-evening scene made the model dissolve the whole world within the first second). Mark it with matches_location_still. Stills are cheap; a new one is fine wherever a scene needs it.
- HER SCALE IS FIXED INSIDE A SCENE (measured on two takes, 2026-10-29: a face that starts tiny and grows as she approaches resolves badly all the way in). Never write her walking toward the camera from a distance, never "grows larger in frame", never an approach that changes how big she is. Her distance from the camera at the end of a scene is what it was at the start: she moves across the frame, away, or in place. A change of scale (far to near, wide to close) is always a CUT to a new scene with its own framing, never a move inside one scene. Push-ins on her are small amplitude only. Her walking direction is named relative to the CAMERA, never to the frame edge: "away from the camera, deeper into the frame" or "across the frame on the near ground". A road or path that runs from the far ground toward the camera is a walk toward the lens whichever side it is on (measured 2026-10-29: "to the right along the ridge, never toward the camera" doubled her size in 18 seconds because the road led to the camera), so on such a path she walks away, toward the far end, or stands. A verse or a chorus is therefore covered by two or three SETUPS in the same location, cut between them: a change of angle or framing (chair front-on, then three-quarter profile from the lamp side; wide, then detail), each its own scene with its own camera. The location reference keeps the place consistent across those cuts.
- Scene boundaries fall on CUT POINTS: a section edge, or one of the listed line cut points (half a second before a sung line begins), or the listed FIRST-LINE SETUP point (two seconds before the song's first sung line: the lip-sync scene for the first word starts there). Choose cuts where the words break.
- Inside a scene, the camera does ONE thing, written in the video model's own grammar (measured: it obeys this and ignores metric distances): "static", or "<verb> with <small|large> amplitude at <slow|fast> speed toward/along X throughout the take". Verbs: push in, pull out, pan left/right, truck left/right, tilt up/down, pedestal up/down, arc, tracking. Performance takes use small amplitude at slow speed; large amplitude only on B-roll. A scene has ONE framing (the one it starts in); never write a second framing the move should reach (measured: "from a medium shot to a medium close-up" became a hidden cut rendered as a dissolve, with the wardrobe changing across it). Bound a move only by what must stay in frame to the end (the lamp, the window, her shoulders). No whip pans, no handheld chaos, no orbit-plus-zoom, never centimetres or metres.
- Lip-sync scenes need her mouth readable: medium close-up or close-up, face toward camera. Wider performance is fine when the words matter less. The converse is a hard rule (user, 2026-10-29): a medium close-up or anything closer while a vocal line is audible is ALWAYS sing_to_camera. A close shot of her silent face under a sung line reads as a mistake, so no_singing_her and sing_away are written at medium and wider only; a silent close-up belongs to instrumental passages.
- THE FIRST LINE LANDS ON HER FACE (user, 2026-10-29): the scene that holds the first sung line of the song is a lip-sync setup (sing_to_camera, medium close-up or closer) that STARTS AT LEAST 2 SECONDS BEFORE that line, so she is already in frame, facing the lens, when the first word arrives. The intro therefore ends on a cut point two or more seconds ahead of the first word, not on the section edge half a second before it.
- Settings must be concrete and filmable in one place, described without people. If sky, clouds or stars are visible, say so (the renderer must pin them).
- Environment-only B-roll (no person in frame) moves ONE element only (the sea, a curtain, the rain, a flame), slowly, with everything else still. A second moving element, or any rhythm ("every few seconds", "with each gust"), makes the model run the shot as a timelapse (measured). Sounds belong to the soundscape, not to the picture.
- LOCATIONS: every scene happens in one named location. Use enough places that the video travels (user, 2026-11-08: four places for 39 scenes felt flat): about one location per five scenes (a full-length song of 35-45 scenes has 7-9), no location holds more than a quarter of the running time, and never more than three scenes in a row in one location. Take places from the images the lyric names (each verse that names a new place, event or object is a reason for a new location); a location may still be returned to, especially for the chorus. For each location you write a person-free still prompt for an image model; that still is given to the video model as a reference picture of the place, so every scene set there renders the same room or landscape. A location is one physical place at one camera distance class (a room; a detail of that room counts as its own location). Weather and time of day may change between scenes in the same location; walls, furniture, window placement and materials may not. The parlour the reference photographs show is a location. When the brief lists KNOWN LOCATIONS (places already photographed for earlier scripts or other videos, their stills on file), prefer them wherever one serves THIS video and refer to each by its exact id; invent a new location only where no known one fits the song. Never redecorate a known place: its fixed elements are what its description says, and your scene settings there must match that description.
- NOBODY POPS IN OR OUT AT A SAME-ANGLE CUT (user, 2026-11-08): two scenes in a row in the same location are fine. What is wrong is a cut that keeps the SAME ANGLE (the same camera position and nearly the same framing) while a person appears or vanishes: an empty plate, then her on the same view, makes her appear out of nowhere; her, then the same view empty, makes her vanish. Every scene names its "angle", the camera position in that location ("the location view" for the still's own position; otherwise a short name such as "reverse from the far bank", "low beside the hearth"). When the people in frame change between two scenes in a row in one location, the second changes the angle: another camera position, or a clearly different framing (wide to medium close-up, not wide to extreme wide).
- Timing is NOT yours beyond choosing cut points: the system computes clip windows and which words fall where.

Write for the song's genre: its pace, weight and mood should be visible in the choice of places, light, weather, movement and the proportion of performance to B-roll."""


LINE_LEAD = 0.5          # a line's cut point sits this far before the line begins
FIRST_LINE_LEAD = 2.0    # the lip-sync setup for the song's FIRST sung line starts this far ahead of it (user rule, 2026-10-29)


def cut_points(sections):
    """Every allowed scene boundary, in order: section edges and, inside sung sections, a point
    LINE_LEAD before each line (never before the previous line ends). Returns sorted floats."""
    pts = set()
    for s in sections:
        pts.add(round(s["start"], 3))
        pts.add(round(s["end"], 3))
        prev_end = s["start"]
        for l in s["lines"]:
            c = max(l["start"] - LINE_LEAD, prev_end)
            if s["start"] < c < s["end"]:
                pts.add(round(c, 3))
            prev_end = l["end"]
        for c in s.get("voice_cuts") or []:                   # two-singer songs: a measured hand-over
            pts.add(round(c, 3))                               # from one voice to the other
    first = first_sung_line(sections)
    if first and first["start"] - FIRST_LINE_LEAD > 0:        # the lip-sync setup for the first word
        pts.add(round(first["start"] - FIRST_LINE_LEAD, 3))   # starts this far ahead of it (user rule)
    return sorted(pts)


def first_sung_line(sections):
    """The song's first sung line (dict with start/end/text), or None for an instrumental."""
    for s in sections:
        for l in s.get("lines") or []:
            return l
    return None


CLOSE_FRAMINGS = ("medium close-up", "medium close up", "close-up", "close up", "extreme close-up", "extreme close up", "mcu", "cu")


def is_close(framing):
    """Medium close-up or closer: the framings where her mouth is readable."""
    fr = str(framing or "").strip().lower()
    return any(fr.startswith(c) for c in CLOSE_FRAMINGS)


def check_scene(sc, sections):
    """Scene-level hard rules (user, 2026-10-29), mechanically; a list of problems, empty = clean.
    1. A medium close-up or closer while a sung line is audible is sing_to_camera: never her
       silent face, close, under words. 2. The scene holding the song's first sung line is a
       lip-sync setup that starts FIRST_LINE_LEAD or more before that line."""
    problems = []
    perf = str(sc.get("performance") or "")
    a, b = float(sc["start"]), float(sc["end"])
    audible = [l for s in sections for l in (s.get("lines") or []) if l["end"] > a and l["start"] < b]
    if audible and is_close(sc.get("framing")) and perf != "sing_to_camera":
        problems.append(f"close framing '{sc.get('framing')}' with performance {perf or '(none)'} while "
                        f"{len(audible)} sung line(s) are audible: a close shot under words is sing_to_camera")
    first = first_sung_line(sections)
    if first and a <= first["start"] < b:
        if perf != "sing_to_camera" or not is_close(sc.get("framing")):
            problems.append(f"holds the first sung line ({first['start']:.2f}s) but is {perf or '(none)'} "
                            f"'{sc.get('framing')}': the first line lands on her face (sing_to_camera, medium close-up or closer)")
        if first["start"] - a < FIRST_LINE_LEAD - 0.05:
            problems.append(f"starts {first['start'] - a:.2f}s before the first sung line; the lip-sync setup "
                            f"starts at least {FIRST_LINE_LEAD:.0f}s ahead of it (cut point {first['start'] - FIRST_LINE_LEAD:.2f}s)")
    if is_duet(sections):
        problems += check_duet(sc, sections)
    return problems


def snap_cut(t, points):
    return min(points, key=lambda x: abs(x - t))


def _section_text(sections):
    out = []
    first = first_sung_line(sections)
    setup = round(first["start"] - FIRST_LINE_LEAD, 3) if first and first["start"] - FIRST_LINE_LEAD > 0 else None
    for s in sections:
        head = f"[{s['index']}] {s['type']}  {s['start']:.2f}-{s['end']:.2f}s  ({s['end'] - s['start']:.1f}s)"
        if s.get("level"):
            head += f"  level {s['level']} ({s['level_db']:.0f} dB)"
            if s.get("quiet_spans"):
                head += ", quiet " + ", ".join(f"{a:.1f}-{b:.1f}s" for a, b in s["quiet_spans"])
        out.append(head if s["vocal"] else head + "  instrumental, no vocal")
        if setup is not None and s["start"] <= setup < s["end"]:
            out.append(f"      cut {setup:6.2f} | FIRST-LINE SETUP: the lip-sync scene for the first word starts here")
        if s["vocal"]:
            prev_end = s["start"]
            vcuts = sorted(s.get("voice_cuts") or [])
            for l in s["lines"]:
                c = max(l["start"] - LINE_LEAD, prev_end)
                while vcuts and vcuts[0] < l["start"]:
                    out.append(f"      cut {vcuts.pop(0):6.2f} | VOICE CHANGE (measured)")
                tag = f"cut {c:6.2f} |" if s["start"] < c < s["end"] else "            |"
                ans = "".join(f", answered by {VOICE_LABEL.get(x['voice'], x['voice'])} at {x['start']:.2f}s"
                              for x in l.get("voice_answers") or [])
                who = (f"[{VOICE_LABEL.get(l['voice'], l['voice'])}{ans}{', UNSURE: B-roll, no lip-sync' if l.get('voice_unsure') else ''}] "
                       if l.get("voice") else "")
                out.append(f"      {tag} {l['start']:7.2f}-{l['end']:6.2f}  {who}{l['text']}")
                prev_end = l["end"]
                while vcuts and vcuts[0] < l["end"]:
                    out.append(f"      cut {vcuts.pop(0):6.2f} | VOICE CHANGE (measured, inside the line above)")
            for vc in vcuts:
                out.append(f"      cut {vc:6.2f} | VOICE CHANGE (measured)")
    return "\n".join(out)


VOICE_LABEL = {"her": "HER", "him": "HIM", "her>him": "SHARED (her then him): show HER", "him>her": "SHARED (him then her): show HER",
               "unknown": "voice unmeasured"}


def is_shared(voice):
    """A line both singers sing (handed over partway): it always shows HER (user rule, 2026-11-07)."""
    return ">" in str(voice or "")


def is_duet(sections):
    """True when the section table carries measured voices (a two-singer script)."""
    return any(l.get("voice") for s in sections for l in (s.get("lines") or []))


def voice_seconds_in(sections, a, b):
    """Measured seconds of each voice singing a lyric line inside [a, b): the section's voice runs
    intersected with its lines (so a line handed over partway counts each voice where it sings).
    A SHARED line counts as hers throughout: shared lines always show her (user rule, 2026-11-07)."""
    out = {"her": 0.0, "him": 0.0}
    for s in sections:
        runs = s.get("voice_runs") or []
        for l in s.get("lines") or []:
            lo, hi = max(a, l["start"]), min(b, l["end"])
            if hi <= lo:
                continue
            if is_shared(l.get("voice")) and a <= l["start"] < b:   # the scene the shared line starts in shows her;
                out["her"] += hi - lo                                # a tail spilling past a cut is counted as sung
                continue
            for r in runs:
                o = min(hi, r["end"]) - max(lo, r["start"])
                if o > 0:
                    out[r["voice"]] = out.get(r["voice"], 0.0) + o
    return out


VOICE_OWN = {"her": "her", "him": "his"}
DUET_OTHER_MIN = 1.0     # the other singer's voice this long inside a lip-sync scene is a problem


def check_duet(sc, sections):
    """Two-singer rules: every scene names its singer; a lip-sync scene shows the voice that is
    heard (the measured timeline is definitive). Sets sc['singer'] for a lip-sync scene from the
    measurement when the writer named the other one. Returns problems."""
    problems = []
    perf = str(sc.get("performance") or "")
    if perf == "no_singing_environment":
        sc["singer"] = "none"
        return problems
    who = str(sc.get("singer") or "").strip().lower()
    if who == "both" and perf.startswith("sing"):
        problems.append("both singers in a lip-sync scene (untested): set to her")
        who = "her"
    if who == "both":                  # a silent two-shot (opt-in, 2026-11-08 test): her and him in frame
        sc["singer"] = who
        return problems
    if who not in ("her", "him"):
        problems.append(f"names no singer ('{sc.get('singer')}'): set to her")
        who = "her"
    if perf.startswith("sing"):
        sec = voice_seconds_in(sections, float(sc["start"]), float(sc["end"]))
        if sum(sec.values()) > 0:
            heard = max(sec, key=sec.get)
            if heard != who:
                problems.append(f"lip-sync scene written for {who} but the measured voice here is {heard} "
                                f"({sec[heard]:.1f}s vs {sec[who]:.1f}s): singer set to {heard}; its opening "
                                f"still prompt may still describe {who}")
                who = heard
            a, b = float(sc["start"]), float(sc["end"])
            unsure = [l for s in sections for l in (s.get("lines") or [])
                      if l.get("voice_unsure") and min(b, l["end"]) - max(a, l["start"]) >= 0.5]
            if unsure:
                problems.append("lip-sync over a line the measurement cannot place (" + "; ".join(
                    f"{l['start']:.1f}s '{l['text'][:32]}'" for l in unsure) + "): B-roll there, no lip-sync")
            other = "him" if who == "her" else "her"
            if sec[other] >= DUET_OTHER_MIN:
                problems.append(f"{VOICE_OWN[other]} voice is heard for {sec[other]:.1f}s inside {VOICE_OWN[who]} "
                                f"lip-sync scene: cut at the voice change")
    sc["singer"] = who
    return problems


def solo_mute_ranges(script, scene, a, b):
    """Two singers: the song-second stretches inside [a, b) where the OTHER singer is heard (their
    measured runs and their answers), for a lip-sync scene of this singer. Muted from what the
    render hears, the singer on screen is not driven by the other's words (2026-11-07: his lips
    took her answering "Vanadis" too). Shared lines are left in: they show her, lip-syncing all
    of it (user rule). [] on a one-singer script or a scene with no singer."""
    d = script.get("duet") or {}
    who = singer_of(script, scene)
    if not d or who not in ("her", "him"):
        return []
    other = "him" if who == "her" else "her"
    shared = [(l["start"], l["end"]) for s in script["facts"]["sections"] for l in s.get("lines") or []
              if is_shared(l.get("voice"))]
    out = []
    for r in list(d.get("runs") or []) + list(d.get("answers") or []):
        if r["voice"] != other:
            continue
        lo, hi = max(a, float(r["start"])), min(b, float(r["end"]))
        for s0, e0 in shared:            # cut the shared lines out of the range
            if hi > s0 and lo < e0:
                if lo < s0:
                    out.append([round(lo, 2), round(s0, 2)])
                lo = max(lo, e0)
        if hi - lo > 0.05:
            out.append([round(lo, 2), round(hi, 2)])
    return sorted(out)


FRAMING_LADDER = (("extreme wide", 0), ("very wide", 0), ("establishing", 0), ("medium wide", 2), ("medium close", 4),
                  ("extreme close", 6), ("close", 5), ("medium", 3), ("wide", 1), ("full", 2), ("detail", 6), ("insert", 6))


def framing_step(framing):
    """A framing's place on the ladder extreme wide (0) .. extreme close-up / detail (6); None if unknown."""
    fr = str(framing or "").strip().lower().replace("-", " ")
    for k, v in FRAMING_LADDER:
        if fr.startswith(k.replace("-", " ")):
            return v
    return None


def same_angle(a, b):
    """Two scenes in one location shot from the same angle: the same named camera position (scenes
    with no angle field, from before 2026-11-08, count as one position) and framings at most one
    step apart (wide and extreme wide: the same view; wide and medium close-up: a different shot)."""
    na = str(a.get("angle") or "").strip().lower()
    nb = str(b.get("angle") or "").strip().lower()
    if na and nb and na != nb:
        return False
    fa, fb = framing_step(a.get("framing")), framing_step(b.get("framing"))
    return fa is None or fb is None or abs(fa - fb) <= 1


def continuity_problems(script):
    """Same-angle cuts where a person appears or vanishes (user, 2026-11-08: the same place twice in
    a row is fine; the same ANGLE with someone popping in or out is not). Who is in frame =
    singer_of (her / him / none). Returns [(index of the later scene, problem)]."""
    out = []
    sc = script.get("scenes") or []
    for a, b in zip(sc, sc[1:]):
        if not a.get("location") or a.get("location") != b.get("location"):
            continue
        wa, wb = singer_of(script, a), singer_of(script, b)
        if wa == wb or not same_angle(a, b):
            continue
        who = {"none": "nobody"}
        out.append((b["index"], f"cut from scene {a['index']} keeps the same angle in `{b['location']}` "
                                f"({a.get('framing')} -> {b.get('framing')}) while who is in frame changes "
                                f"({who.get(wa, wa)} -> {who.get(wb, wb)}): change the angle or framing of one of the two"))
    return out


def mark_continuity(script):
    """Refresh the continuity problems on every scene (drops old ones first, so it can rerun after edits)."""
    for sc in script.get("scenes") or []:
        sc["scene_problems"] = [p for p in (sc.get("scene_problems") or []) if not p.startswith("cut from scene ")]
    for i, msg in continuity_problems(script):
        script["scenes"][i]["scene_problems"].append(msg)
    return script


def locks_text(locks):
    """The ALREADY SHOT block: one line per locked take, in song order."""
    out = []
    for L in sorted(locks, key=lambda x: x["start"]):
        s = L["scene"]
        out.append(f"- {L['start']:.2f}-{L['end']:.2f}s  \"{s.get('title')}\": {s.get('framing')}, {s.get('performance')}, "
                   f"location `{s.get('location')}`, camera {s.get('camera')}. {s.get('action') or ''}".strip())
    return "\n".join(out)


DUET_RULES = """

TWO SINGERS. This song is a duet, and this block overrides "the singer is the one lead" above:
- Two people are defined by references: SHE (the lead, her two pictures) and HE (the male singer, his two pictures). Every rule above about the singer applies to whichever singer a scene shows: his scale is fixed inside a scene, he never walks toward the camera, his lip-sync needs his mouth readable at medium close-up or closer, a close shot of his silent face under words is as wrong as one of hers, and the first sung line lands on the face of whoever sings it.
- WHO SINGS WHEN IS MEASURED from this recording: every line below carries its voice ([HER], [HIM], or [HER then HIM] for a line handed over partway). The lyric tags may say otherwise; the measurement wins.
- Every scene names its "singer": "her" or "him" for the one person the references define in that scene, "none" for an environment-only plate. A lip-sync scene (sing_to_camera, sing_away) shows the singer whose voice is heard, never her lip-syncing his words or him lip-syncing hers. Where the voice changes inside a section, cut on the listed VOICE CHANGE point so each lip-sync scene holds one voice.
- Each scene shows ONE of the two (two-shots of both singers are untested on this model and are not written). A SHARED line (both voices in it) always shows HER, lip-syncing it.
- A line marked UNSURE is one the measurement cannot place: cover it with B-roll (an environment plate, or a silent shot of either singer at medium or wider), never a lip-sync scene. If in doubt about who sings, B-roll.
- A line ANSWERED by the other singer (a short call and answer, measured: "[HIM, answered by HER at 91.10s]") belongs to the line's singer: one lip-sync scene of that singer holds it, and the answer is heard off screen (the render mutes the answer from what the singer on screen lip-syncs). Never cut to the other singer for an answer this short.
- Either singer may appear silent (no_singing_her, written with singer him for his silent shots) under the other's voice at medium or wider.
- A change of singer at a cut is the same jump as a person appearing when the angle stays: when the next scene in the same location shows the other singer, change the angle (another camera position, or a clearly different framing).
- He has one outfit for the whole video (wardrobe2), as his references show; his opening_still describes him in the setting, hers describes her.
- His vocal delivery: {voice}. Write his performance for it."""


def scene_prompt(title, style, lyrics_raw, facts, refs, concept=None, known=None, locks=None, duet=None):
    """known: library rows (mv_locations) the model may reuse by id; None/[] = none offered.
    locks: takes already shot and passed ({start, end, scene}); the model writes only the gaps.
    duet: {"voice": how he sings} when the song has a second (male) singer: the duet rules join
    the system prompt and every scene names its singer (refs then describes all four pictures)."""
    schema = {
        "concept": "2-3 sentences: what the video is about and how it moves (string)",
        "look": "one sentence of visual style for every prompt: film stock feel, light QUALITY, contrast, colour grade, camera weight ONLY. Never name an object or a light source in it (no candles, lamps, fires, torches, windows): the sentence rides on every exterior prompt too, and a prop named here gets hallucinated into shots that cannot hold it (measured: 'candle and lamp interiors' put lit candles over a cliff road). Interior warmth belongs in that scene's lighting field (string)",
        "wardrobe": "the singer's one outfit, as it appears in the references, in one sentence (string)",
        **({"wardrobe2": "the male singer's one outfit, as his references show it, in one sentence (string)"} if duet else {}),
        "arc": "how light, weather and time of day progress across the song's sections, and the payoff objects from the lyrics (where each is first shown and where it returns changed), 2-4 sentences (string)",
        "locations": [{
            "id": "snake_case id (a KNOWN location's exact id when reusing one)",
            "reuse": "true only when id is a KNOWN location; then omit name, description and still_prompt (boolean)",
            "name": "short name, e.g. the parlour",
            "description": "one sentence of what a reference photograph of this place shows: layout, key furniture or landforms, materials, where the light comes from; no people",
            "still_prompt": "a prompt for a photoreal 16:9 still of this place with NO people, one paragraph: camera position and height, every fixed element and where it sits in frame, materials, the light and weather of its first scene, the video's look; the elements must match this location's scene settings word for word"
        }],
        "scenes": [{
            "from": "seconds where the scene starts: a section edge or a listed cut point (number)",
            "to": "seconds where it ends: the next scene's from (number)",
            "title": "short label",
            "location": "id of one of the locations",
            "setting": "the place, concrete and filmable, no people, 2-3 sentences; name any visible sky",
            "performance": "one of: sing_to_camera | sing_away | no_singing_her | no_singing_environment",
            **({"singer": "her | him | none: the one singer this scene shows (none for no_singing_environment); a lip-sync scene shows the measured voice"} if duet else {}),
            "framing": "e.g. medium close-up, medium, wide, extreme wide, detail",
            "angle": "the camera position in this location: 'the location view' (the location still's own position) or a short name of another position, e.g. 'reverse from the far bank'; reuse the same name for the same position",
            "camera": "'static', or ONE move in the model's grammar: '<verb> with small amplitude at slow speed toward X throughout the take' (one framing only, never a second framing to reach, no distances)",
            "keep_in_frame": "one or two fixed things that must stay in frame until the end of the take, e.g. 'the oil lamp and the window' (string)",
            "lighting": "one sentence",
            "mood": "three words",
            "action": "what she does, 1-3 sentences, one main motion; for environment B-roll exactly ONE moving element, slow, and what stays still",
            "others": "null, or one sentence: the other people this scene needs (who, how many, where in frame, what they do), anonymous; no band",
            "opening_still": "a prompt for a photoreal 16:9 still of the FIRST FRAME of this scene, one paragraph: the location still's camera position, framing, fixed elements and light, with she in the setting where the scene has her, and with every person listed in others ALREADY IN PLACE in their opening pose (each caught mid-activity, each different, faces turned away or softened by shadow, small in frame). On the image-to-video lane this still IS frame 0, so anyone missing from it has to fade in (measured)",
            "matches_location_still": "true when this scene's light, weather, time of day and people are what the location's still shows (that still can be frame 0); false when the scene needs its own opening still (boolean)",
            "why": "one sentence: why this scene belongs at this point of the song"
        }],
    }
    user = (f"SONG: {title}\nSTYLE / GENRE: {style}\n"
            f"DURATION: {facts['duration']:.1f}s. Lyric alignment cover {facts.get('cover')}.\n\n"
            + (f"REFERENCES (the two singers):\n{refs}\n\n" if duet else f"REFERENCES (the only person in the video):\n{refs}\n\n")
            + (f"USER CONCEPT (follow it): {concept}\n\n" if concept else "")
            + (("KNOWN LOCATIONS (already photographed; a still of each is on file). Reuse one by its exact id "
                "with {\"id\": ..., \"reuse\": true}; a new place needs a full entry with a still prompt:\n"
                + known_text(known) + "\n\n") if known else "")
            + "MEASURED SECTIONS (real seconds on this recording; use these indices):\n"
            + _section_text(facts["sections"]) + "\n\nLYRICS:\n" + (lyrics_raw or "").strip()
            + ("\n\nALREADY SHOT (the user has passed these takes; they are fixed, with their places, framings and "
               "light. Do NOT write scenes for these windows. Write scenes only for the GAPS between them, and plan "
               "the gaps so the cut flows into and out of each one; the arc must fit them):\n" + locks_text(locks)
               if locks else "")
            + ("\n\nYour scenes tile every gap in order: within a gap the first scene starts at the gap's start, "
               "each `to` is the next scene's `from`, the last ends at the gap's end. Every from/to is a section "
               "edge, a listed cut point or a locked take's edge; inside a lineless span (intro, solo, outro) any "
               "second is allowed. List only your own scenes, in song order. "
               if locks else
               "\n\nThe scenes tile the whole song in order with no gaps: the first starts at 0, each `to` is the "
               "next scene's `from`, the last ends at the song's end. Every from/to is a section edge or a listed "
               "cut point; inside a lineless span (intro, solo, outro) any second is allowed. ")
            + "Return JSON with exactly this shape:\n" + json.dumps(schema, indent=1))
    system = SCENE_SYSTEM + (DUET_RULES.replace("{voice}", duet.get("voice") or "as heard on the recording") if duet else "")
    return system, user


def _keep_raw(cache_dir, name, text):
    """The model's raw answer, kept beside the lyric cache so a parse failure can be read back."""
    if not cache_dir:
        return
    try:
        import os
        os.makedirs(os.path.join(cache_dir, "writer_raw"), exist_ok=True)
        with open(os.path.join(cache_dir, "writer_raw", name), "w") as f:
            f.write(text or "")
    except Exception:
        pass


def _extract_json(text):
    text = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0] or [0])
    return json.loads(text[start:])


def _finish_scene(sc, sections):
    """start/end -> covered section range, types, lipsync flag."""
    idx = [k for k, x in enumerate(sections) if x["end"] > sc["start"] and x["start"] < sc["end"]]
    if not idx:
        idx = [min(range(len(sections)), key=lambda k: abs(sections[k]["start"] - sc["start"]))]
    sc["sections"] = [idx[0], idx[-1]]
    sc["section_types"] = [sections[k]["type"] for k in idx]
    sc["lipsync"] = str(sc.get("performance", "")).startswith("sing")
    sc["scene_problems"] = check_scene(sc, sections)
    return sc


FREE_CUT_MARGIN = 2.5    # inside a lineless section a cut may fall anywhere this far from its edges


def snap_scene_cut(t, sections, points):
    """A scene boundary: anywhere inside a lineless (instrumental) section, at least
    FREE_CUT_MARGIN from its edges (a solo or an intro is cut to its music, not to words);
    elsewhere the nearest cut point."""
    for s in sections:
        if not s["vocal"] and s["start"] + FREE_CUT_MARGIN <= t <= s["end"] - FREE_CUT_MARGIN:
            return round(t, 2)
    return snap_cut(t, points)


def normalize_scenes(plan, sections, locks=None):
    """Make the scene list tile the song in order. Scenes given as from/to seconds are snapped to
    cut points (section edges and line cut points; any second inside a lineless section) and
    forced contiguous; scenes given as a `sections` range (older answers) take those sections'
    edges. Empty scenes are dropped. `locks` ({start, end, scene, locked}) are takes already shot:
    they are placed at their windows as given and the model's scenes fill the gaps between them
    (a model scene reaching into a lock is cut at the lock's start; a gap the model left empty
    becomes a flagged placeholder). Returns the scenes with index/start/end/sections/
    section_types/lipsync filled in."""
    n = len(sections)
    locks = sorted(locks or [], key=lambda L: L["start"])
    pts = sorted(set(cut_points(sections)) | {round(x, 3) for L in locks for x in (L["start"], L["end"])})
    song_end = sections[-1]["end"]
    raw = plan.get("scenes") or []
    scenes, cursor, li = [], 0.0, 0

    def _place_locks():
        nonlocal cursor, li
        while li < len(locks) and locks[li]["start"] <= cursor + 1e-3:
            L = locks[li]
            sc = {k: v for k, v in L["scene"].items() if k not in ("index", "sections", "section_types", "lipsync", "scene_problems")}
            sc["start"], sc["end"] = round(L["start"], 3), round(L["end"], 3)
            sc["locked"] = L["locked"]
            scenes.append(sc)
            cursor = max(cursor, L["end"])
            li += 1

    def _gap_end():
        return locks[li]["start"] if li < len(locks) else song_end

    if locks:
        _place_locks()
        for sc in raw:
            if not isinstance(sc, dict) or cursor >= song_end:
                continue
            try:
                b = snap_scene_cut(float(sc["to"]), sections, pts)
            except (KeyError, TypeError, ValueError):
                continue
            b = min(b, _gap_end())
            if b <= cursor + 1e-3:
                continue
            sc = {k: v for k, v in sc.items() if k not in ("from", "to")}
            sc["start"], sc["end"] = round(cursor, 3), round(b, 3)
            scenes.append(sc)
            cursor = b
            _place_locks()
        while cursor < song_end - 1e-3:                   # a gap the model did not fill
            end = _gap_end()
            if scenes and not scenes[-1].get("locked") and abs(scenes[-1]["end"] - cursor) < 1e-3:
                scenes[-1]["end"] = round(end, 3)
            else:
                scenes.append({"title": "(gap not written)", "start": round(cursor, 3), "end": round(end, 3),
                               "performance": "no_singing_environment", "gap": True})
            cursor = end
            _place_locks()
        for i, sc in enumerate(scenes):
            sc["index"] = i
            _finish_scene(sc, sections)
            if sc.get("gap"):
                sc["scene_problems"].append("the writer left this gap empty")
        return scenes
    for sc in raw:
        if not isinstance(sc, dict):
            continue
        a = b = None
        if sc.get("from") is not None and sc.get("to") is not None:
            try:
                a, b = float(sc["from"]), float(sc["to"])
            except (TypeError, ValueError):
                a = b = None
        if a is None:
            rng = sc.get("sections")
            if isinstance(rng, str):                  # "[3, 4]", "3-4", "3" all seen from the model
                rng = [int(x) for x in re.findall(r"\d+", rng)]
            elif isinstance(rng, (int, float)):
                rng = [int(rng)]
            try:
                i0, i1 = int(rng[0]), int(rng[-1])
            except Exception:
                continue
            i0, i1 = max(0, min(i0, n - 1)), max(0, min(max(i1, i0), n - 1))
            a, b = sections[i0]["start"], sections[i1]["end"]
        a = cursor                                    # contiguity: every scene starts where the last ended
        b = snap_scene_cut(b, sections, pts)
        if b <= a:
            continue
        sc = dict(sc)
        sc.pop("from", None)
        sc.pop("to", None)
        sc["start"], sc["end"] = round(a, 3), round(b, 3)
        scenes.append(sc)
        cursor = b
        if cursor >= song_end:
            break
    if not scenes:
        raise ValueError("the writer returned no usable scenes")
    if scenes[-1]["end"] < song_end:
        scenes[-1]["end"] = song_end
    for i, sc in enumerate(scenes):
        sc["index"] = i
        _finish_scene(sc, sections)
    return scenes


def _slug(x):
    return re.sub(r"[^a-z0-9]+", "_", str(x or "").lower()).strip("_")


def known_text(rows):
    """The block the writer shows the model: one line per known place, id first."""
    return "\n".join(f"- id `{r['id']}` ({r.get('name')}): {r.get('description') or '(no description)'}"
                     + (f" Its still shows: {still_state(r.get('still_prompt'))}" if still_state(r.get("still_prompt")) else "")
                     for r in rows)


STATE_WORDS = re.compile(r"(?i)\b(sky|cloud|storm|rain|sun|sunset|dusk|dawn|evening|night|morning|light|lit|weather|fog|mist|candle|lamp)\w*")


def still_state(prompt):
    """The sentences of a still prompt that fix its light, weather and time of day: what frame 0
    of a plate from that still will show, so the writer can tell when a scene needs its own."""
    sents = re.split(r"(?<=[.!?])\s+", (prompt or "").strip())
    return " ".join(s for s in sents if STATE_WORDS.search(s))[:400]


def _known_row(r):
    return {"id": r["id"], "name": (r.get("name") or r["id"].replace("_", " ")).strip(),
            "description": (r.get("description") or "").strip(),
            "still_prompt": (r.get("still_prompt") or "").strip(),
            "still_id": r.get("still_id"), "known": True}


def normalize_locations(plan, scenes, known=None):
    """The script's location table, and every scene pointing at one of its rows. A location whose
    id (or name) matches a KNOWN library row is that row: the library's description, still prompt
    and still id win, marked `known` (the still exists; nothing to generate). A scene whose
    `location` names nothing in the table gets a row of its own built from its setting (marked
    derived, with the scene's opening still as a stand-in prompt - that one may show her)."""
    kn = {r["id"]: r for r in (known or []) if r.get("id")}
    by_name = {_slug(r.get("name")): r["id"] for r in kn.values() if _slug(r.get("name"))}
    locs, seen, alias = [], {}, {}

    def _add_known(lid):
        seen[lid] = len(locs)
        locs.append(_known_row(kn[lid]))

    for L in plan.get("locations") or []:
        if not isinstance(L, dict):
            continue
        lid = _slug(L.get("id") or L.get("name"))
        if not lid or lid in seen:
            continue
        # a known place under its own id, or under a new id with a known place's name
        hit = lid if lid in kn else (by_name.get(_slug(L.get("name"))) or by_name.get(lid))
        if hit:
            if hit != lid:
                alias[lid] = hit
            if hit not in seen:
                _add_known(hit)
            continue
        seen[lid] = len(locs)
        locs.append({"id": lid, "name": (L.get("name") or lid.replace("_", " ")).strip(),
                     "description": (L.get("description") or "").strip(),
                     "still_prompt": (L.get("still_prompt") or "").strip()})
    for sc in scenes:
        lid = _slug(sc.get("location"))
        lid = alias.get(lid, lid)
        if lid not in seen and lid in kn:
            _add_known(lid)
        if lid not in seen:
            lid = _slug(sc.get("title")) or f"scene_{sc['index']}"
            if lid not in seen:
                seen[lid] = len(locs)
                locs.append({"id": lid, "name": sc.get("title") or lid, "description": sc.get("setting", ""),
                             "still_prompt": sc.get("opening_still", ""), "derived": True})
        sc["location"] = lid
    return locs


def location_of(script, scene):
    """The location row a scene points at, or None (scripts written before locations existed)."""
    lid = scene.get("location")
    for L in script.get("locations") or []:
        if L.get("id") == lid:
            return L
    return None


# ----------------------------------------------------------------------------- code: clip windows

def plan_scene_clips(scene, sections, fps=H3_FPS, frames=CHAIN_FRAMES, context=CHAIN_CONTEXT):
    """A scene's chain: 15.08 s clips advancing 13.46 s until the scene window is covered (the
    tail past the scene end is trimmed at assembly; the audio is the master so the cut is exact).
    A scene shorter than one clip gets ONE clip on the smallest grid length that covers it."""
    window = scene["end"] - scene["start"]
    clip_s = frames / fps
    if window <= clip_s:
        f = max(MIN_SINGLE_FRAMES, _h3_frames(math.ceil(window * fps)))
        plan = chain_plan(scene["start"], 1, f, context, fps)
    else:
        adv = (frames - context) / fps
        n = 1 + math.ceil((window - clip_s) / adv)
        plan = chain_plan(scene["start"], n, frames, context, fps)
    clips = []
    for c in plan["clips"]:
        lines = []
        for sec in sections:
            for l in sec["lines"]:
                if l["end"] <= c["song_start"] or l["start"] >= c["song_end"]:
                    continue
                v = l.get("voice")
                sw = l.get("voice_switch")
                if v and is_shared(v) and sw and c["song_start"] >= float(sw) - 0.05:
                    v = v.split(">")[-1]      # the clip starts after the hand-over: the rest is the second voice's
                text, extra = l["text"], {}
                ans = [x for x in l.get("voice_answers") or [] if v and x["voice"] != v.split(">")[-1]]
                if ans:
                    # the other singer's answer is heard, not sung on screen: out of the sung text
                    text = re.sub(r"\s*\([^)]*\)", "", text).strip()
                    extra = {"answered_off_screen": [{"text": " ".join(re.findall(r"\(([^)]*)\)", l["text"])),
                                                      "by": x["voice"], "song_start": x["start"],
                                                      "song_end": x["end"]} for x in ans]}
                ws = l.get("word_starts") or []
                sung = text.split()
                if ws and len(ws) == len(sung) and (l["start"] < c["song_start"] or l["end"] > c["song_end"]):
                    # every sung word has a measured start: name exactly the words starting inside the part
                    # of the clip that is kept (the tail past the scene end is trimmed at assembly), so the
                    # writer does not guess where a cut falls in the line (2026-11-08: it put 'ever' on both
                    # sides of a cut at 'pay')
                    hi = min(c["song_end"], scene["end"])
                    words = [w for w, t in zip(sung, ws) if c["song_start"] - 0.05 <= t < hi]
                    if words:                  # none: a note held over the edge, written as before
                        extra["audible_words"] = " ".join(words)
                lines.append({"text": text, "song_start": l["start"], "song_end": l["end"],
                              **extra,
                              **({"voice": v} if v else {}),
                              "in_clip_start": round(max(0.0, l["start"] - c["song_start"]), 2),
                              "in_clip_end": round(min(plan["seconds"], l["end"] - c["song_start"]), 2),
                              "starts_before_clip": l["start"] < c["song_start"],
                              "ends_after_clip": l["end"] > c["song_end"]})
        clips.append({**c, "seconds": plan["seconds"], "frames": plan["frames"], "lines": lines,
                      "vocal": bool(lines)})
    return plan, clips


# ----------------------------------------------------------------------------- pass 2: clip prose

CLIP_SYSTEM = """You write the per-clip text of MiniMax H3 reference-mode prompts for ONE scene of a music video, as STRICT JSON only (no prose, no markdown fences). The system wraps your text in the fixed sections (subject definitions, retention) itself; you write four fields per clip:
  summary, detailed_description, overall_soundscape, non_diegetic_music.

Hard rules (they come from measured failures; do not bend them):
- The song is already locked into the clip's audio. NEVER write <Audio 1>, never call the song a reference, never ask the model to generate or match music. Write non_diegetic_music as a description of the current song section itself and say the visible performance stays synchronized to it.
- The singer is <Subject 1>. When she sings, write: <Subject 1> (S1) sings <d>[English] exact words</d>. Only the words audible inside THIS clip's window go inside <d>; a line that starts before the clip begins mid-phrase; a line that ends after the clip is cut mid-phrase. When a line carries audible_words (measured), those are exactly its words inside <d>, no word more or fewer. Instrumental passages have no <d>. Lips closed and relaxed when no vocal is audible; one open vowel shape held through sustained notes.
- detailed_description starts with "[Shot 1]" and is ONE continuous shot. No [Shot 2], no cuts, no <scenetrans>.
- Exactly ONE camera motion for the scene, as given, repeated in every clip in the model's own grammar: "The camera pushes in with small amplitude at slow speed toward X throughout the take, one continuous shot with no cut and no dissolve." Name what must stay in frame ("The oil lamp and the window stay in frame to the end.") and never name a second framing for the move to reach (that reads as a cut). Never metric distances (centimetres mean nothing to the model). If static: "The camera is static, locked off for the full clip."
- Her scale is fixed for the whole scene: she never walks toward the camera, never comes closer, never grows in frame; the distance between her and the camera at the end of the clip is what it was at the start (she moves across the frame, away, or in place). If the scene's framing is wide or extreme wide she stays small to the last frame and her face is never shown closer; a change of scale is the next scene's job, not this clip's. (Measured 2026-10-29: an approach from the far end of a road rendered her face broken all the way in.) Name her direction relative to the CAMERA ("away from the camera, deeper into the frame", "across the frame on the near ground"), never as "to the right" or "to the left": a path that runs from the far ground toward the camera is a walk toward the lens whichever side it lies on, and the model follows the path, not the negation (measured 2026-10-29: "to the right along the ridge, never toward the camera" doubled her size in 18 seconds). On such a path she walks away toward the far end, or stands.
- One main physical motion for her per clip, with a duration anchor (how long it takes, in seconds). Hands: say what they do or that they stay still. Hair: say whether it moves. If sky/clouds/stars are visible: say they hold fixed positions with no drift and no cloud movement.
- Environment-only clips (performance no_singing_environment) have no person to anchor real time, so the discipline is stricter: exactly ONE moving element in the whole description (the boats on the swell, OR the shutters, OR the rain), named once, with a whole-clip anchor stated as a count ("over the whole 6 seconds the boats lift and settle once, slowly"). The sky is never that element: if sky, cloud or stars are visible they hold fixed positions with no drift and no cloud movement (sky motion is the strongest timelapse cue; a cloud bank rolling in over 9 seconds rendered as a timelapse, 2026-10-29). Every other element is said to be still, by name. No second moving element, no rhythm or frequency ("every two seconds", "with each gust", "again and again"), never water plus wind plus weather together. The camera move is the only other motion. Sounds of motion (creak, rattle, slap) go in overall_soundscape, never in detailed_description. (Measured: a harbour clip written with boats, ropes, shutters and water and a two-second rhythm rendered as a timelapse.)
- Clip 2 and later CONTINUE: the first lines say that the incoming motion, camera, framing, lighting and state continue uninterrupted from the protected video context, then say what that ongoing motion becomes. Never restart an action already in progress, never re-establish the shot, never return to the reference pose.
- Never change the wardrobe. Other people appear ONLY when the scene lists them (its others field, or anchor_figure: true): write exactly those, anonymous, never a face that competes with hers for the lens, and never anyone the scene did not list; never a band. People are NOT the environment's one moving element and the one-motion rule does not apply to them: real people never act in unison. Give each person their own ordinary business, staggered in time (one drinks while another talks and the third leans back; a laugh starts with one man and the others follow a second or two later), with the small continuous idle movement of living people (weight shifts, a hand turning a tankard, a head turn, a glance). Say what each is doing at the start of the clip (it must match the opening still) and give the group's rhythm across the clip in seconds without choreographing them together. Extras keep to small ordinary actions (walk, stand, haul, drink, talk, coil); never write a large body transition for an extra (climbing, jumping, stepping down into or out of a boat, mounting, falling), and nothing the still's geometry makes impossible (a boat moored off the quay cannot be stepped into): the model renders the impossible as a leap (measured). (Measured: "over the whole 6 seconds they make one slow movement together" rendered three men raising tankards and throwing their heads back in perfect sync, which read as unnatural.) The anchor_figure case is one distant anonymous figure (a fisherman in oilskins, a woman with a basket), small in frame, back to camera or face unreadable, doing ONE slow everyday action at a natural human pace for the whole clip (walking the far quay, hauling a rope hand over hand, sweeping a step). It is the real-time anchor of the shot: say how many steps or pulls it makes over the clip's seconds. Never near the camera, never a second figure, never interacting with anything the singer owns.
- The scene's location is also a referenced subject (a person-free photograph of the place is handed to the model). Call the place by its name (the parlour, the quarry); the system numbers subjects and pictures. Do not redesign the place: its fixed elements are those in the scene setting; you may add weather, time of day and the state changes the scene calls for.
- Lip-sync clips: mouth readable (face toward camera, medium close-up or closer, nothing covering the mouth).
- 150-320 words of detailed_description per clip; concrete, visual, in order of time."""


def singer_of(script, scene):
    """'her' | 'him' | 'none': who <Subject 1> is in this scene (always her on a one-singer script)."""
    if not script.get("duet"):
        return "none" if scene.get("performance") == "no_singing_environment" else "her"
    return scene.get("singer") or ("none" if scene.get("performance") == "no_singing_environment" else "her")


def clip_prompt(script, scene, clips):
    loc = location_of(script, scene)
    who = singer_of(script, scene)
    user = {
        "song": script["title"], "look": script["look"],
        "wardrobe": script.get("wardrobe2") if who == "him" else script["wardrobe"],
        "concept": script["concept"],
        "scene": {k: scene.get(k) for k in ("title", "setting", "performance", "framing", "camera", "keep_in_frame",
                                            "lighting", "mood", "action", "section_types", "start", "end", "anchor_figure", "others")},
        "location": ({k: loc.get(k) for k in ("name", "description")} if loc else None),
        "clips": [{"index": c["index"], "seconds": c["seconds"], "song_start": c["song_start"],
                   "song_end": c["song_end"], "continues_previous": c["index"] > 0,
                   "protected_prefix_seconds": round(CHAIN_CONTEXT / H3_FPS, 3) if c["index"] > 0 else 0,
                   "lines_audible": c["lines"]} for c in clips],
        "return": {"clips": [{"index": 0, "summary": "...", "detailed_description": "[Shot 1] ...",
                              "overall_soundscape": "...", "non_diegetic_music": "..."}]},
    }
    if script.get("duet") and who == "both":
        user["wardrobe"] = {"her (Subject 1)": script["wardrobe"], "him (Subject 2)": script.get("wardrobe2")}
        user["subjects"] = (
            "BOTH singers are in this shot, silent: <Subject 1> is the female lead (she, her), written <Subject 1> (S1); "
            "<Subject 2> is the male singer (he, him), written <Subject 2> (S2). Name each by subject every time "
            "they act. No <d> for either. Every rule about the singer (fixed scale, no walking toward the lens, "
            "one main motion) applies to each of them; their motions are staggered in time, never in unison. "
            "Say where each stands in frame and what each does at the start of the clip (it must match the opening still)")
    elif script.get("duet") and who != "none":
        voice = (script["duet"].get("voice") or "").strip()
        user["subject_1"] = (
            "THE MALE SINGER (he, him), not the female lead: every rule written about her applies to him in "
            "this scene; write he/his, never she/her" + (f". His vocal delivery: {voice}" if voice else "")
            if who == "him" else "the female lead (she, her)")
        user["voices"] = ("each audible line carries its MEASURED voice; only lines in this scene's singer's "
                          "voice go inside <d>. A line in the other singer's voice is heard but not sung on screen: "
                          "no <d> for it, and the singer's lips stay closed through it. A line's "
                          "answered_off_screen words are the other singer answering from off screen: never inside <d>, "
                          "lips closed through them")
    return CLIP_SYSTEM, json.dumps(user, indent=1)


def is_distant(scene):
    """She is a small figure: an extreme wide, or a wide shot where she is not singing to camera.
    Such shots get her BODY picture only with a partial retention (measured 2026-10-29, scene 0:
    a face picture with 'fully_preserved ... face' on a 20 px figure made the model superimpose
    the portrait over the landscape mid-chain). Not 'medium wide'."""
    fr = (scene.get("framing") or "").strip().lower()
    perf = scene.get("performance") or ""
    if perf == "no_singing_environment":
        return False
    return fr.startswith("extreme wide") or (fr.startswith("wide") and perf != "sing_to_camera")


def compile_prompt(script, scene, clip, text, env=None, cam_video=False, distant=False):
    """The six-section Ref2VA prompt for one clip: fixed boilerplate from the script's references
    and wardrobe + Claude's four fields. The style opening is the script's look sentence.

    env: the scene's location row when a person-free still of the place goes into the chain as
    the last reference picture (<Picture 3> behind her two, or <Picture 1> alone in an
    environment-only scene); None = no location picture (her two pictures only, or none at all
    for an environment-only scene on the FL2VA weights). Picture numbering here must match the
    order the route uploads them in: face, body, location.
    cam_video: a manufactured camera-pace clip rides along as <Video 1>; it is declared as pace
    and movement only (the official guide's camera-reference role, weak_reference).
    distant: she is a small figure (is_distant): the route uploads her BODY picture only, so it
    is <Picture 1> and the location still <Picture 2>; the retention is partial and says her face
    is not readable at this distance."""
    refs = script["refs"]
    env_only = scene.get("performance") == "no_singing_environment"
    who = singer_of(script, scene)
    him = who == "him"
    if who == "both" and not env_only:
        # a silent two-shot (opt-in): her two pictures, his two, then the place; the route uploads
        # them in this order (face, body, his face, his body, location)
        r2 = script["duet"]["refs2"]
        singer_def = ("<Subject 1> is the female singer: her facial identity comes from <Picture 1>; her full-body "
                      f"appearance, body proportions and wardrobe come from <Picture 2>. {refs['shared']} "
                      f"Wardrobe: {script['wardrobe']}\n"
                      "<Subject 2> is the male singer: his facial identity comes from <Picture 3>; his full-body "
                      f"appearance, body proportions and wardrobe come from <Picture 4>. {r2['shared']} "
                      f"Wardrobe: {script.get('wardrobe2') or ''}\n")
        singer_ret = ("<Subject 1> (appears throughout [Shot 1]): fully_preserved - retain her referenced face, eyes, "
                      "hair, skin, body proportions and wardrobe exactly, while pose, expression, framing, lighting and "
                      "movement follow the description below.\n"
                      "<Subject 2> (appears throughout [Shot 1]): fully_preserved - retain his referenced face, eyes, "
                      "hair, beard, skin, body proportions and wardrobe exactly, while pose, expression, framing, "
                      "lighting and movement follow the description below. The two are different people and keep "
                      "their own faces and clothes.\n")
        loc_pic, n_loc = 5, 3
    else:
        n_loc = 2
        if him:
            refs = script["duet"]["refs2"]
            wardrobe = script.get("wardrobe2") or ""
        else:
            wardrobe = script["wardrobe"]
    if who == "both" and not env_only:
        pass                                   # defined above
    elif distant and him:
        singer_def = ("<Subject 1> is the male singer, seen at a distance: his full-body appearance, build, hair and "
                      f"wardrobe come from <Picture 1>. Wardrobe: {wardrobe}\n")
        singer_ret = ("<Subject 1> (appears throughout [Shot 1], a small figure in a wide frame): partially_preserved - "
                      "retain his silhouette, hair, build and the wardrobe; his face is not readable at this "
                      "distance and is never shown closer; the framing stays as described below and no close view of "
                      "him is inserted.\n")
        loc_pic = 2
    elif distant:
        singer_def = ("<Subject 1> is the singer, seen at a distance: her full-body appearance, build, hair and "
                      f"wardrobe come from <Picture 1>. Wardrobe: {wardrobe}\n")
        singer_ret = ("<Subject 1> (appears throughout [Shot 1], a small figure in a wide frame): partially_preserved - "
                      "retain her silhouette, long dark hair, build and the wardrobe; her face is not readable at this "
                      "distance and is never shown closer; the framing stays as described below and no close view of "
                      "her is inserted.\n")
        loc_pic = 2
    elif him:
        singer_def = ("<Subject 1> is the male singer: his facial identity comes from <Picture 1>; "
                      "his full-body appearance, body proportions and wardrobe come from <Picture 2>. "
                      f"{refs['shared']} Wardrobe: {wardrobe}\n")
        singer_ret = ("<Subject 1> (appears throughout [Shot 1]): fully_preserved - retain the "
                      "referenced face, eyes, hair, beard, skin, body proportions and the wardrobe exactly, while pose, "
                      "expression, framing, lighting and movement follow the performance described below.\n")
        loc_pic = 3
    else:
        singer_def = ("<Subject 1> is the singer: her facial identity comes from <Picture 1>; "
                      "her full-body appearance, body proportions and wardrobe come from <Picture 2>. "
                      f"{refs['shared']} Wardrobe: {wardrobe}\n")
        singer_ret = ("<Subject 1> (appears throughout [Shot 1]): fully_preserved - retain the "
                      "referenced face, eyes, hair, skin, body proportions and the wardrobe exactly, while pose, "
                      "expression, framing, lighting and movement follow the performance described below.\n")
        loc_pic = 3

    def env_def(n, pic):
        return (f"<Subject {n}> is the location, {env['name']}, shown in <Picture {pic}>: "
                f"{env.get('description', '').strip()}\n")

    def env_ret(n, pic):
        return (f"<Subject {n}> (the setting of [Shot 1]): fully_preserved - retain the layout, architecture, "
                f"furniture, objects, materials and light direction of <Picture {pic}>, so this is recognisably the "
                "same place; camera position, framing, weather, time of day and the state changes described "
                "below are free to differ from the photograph.\n")

    cam_def = ("<Video 1> is the camera reference: a slow, even push-in filmed in this same place. It supplies "
               "only the camera movement, its amplitude and its pace; nothing of its content, framing or timing of "
               "action carries over.\n") if cam_video else ""
    cam_ret = ("<Video 1> (camera movement and pace): weak_reference - follow its push-in amplitude and speed; "
               "the picture, subjects, performance and sound come from the description below.\n") if cam_video else ""
    if env_only and env:
        subject = "subject_definitions:\n" + env_def(1, 1)
        retention = "retention_analysis:\n" + env_ret(1, 1)
    elif env_only:
        subject = "subject_definitions:\nNo referenced subjects: this clip is environment only, with no people on screen.\n"
        retention = "retention_analysis:\nNothing to retain from references; the setting, style and palette follow the description below.\n"
    elif env:
        subject = "subject_definitions:\n" + singer_def + env_def(n_loc, loc_pic)
        retention = "retention_analysis:\n" + singer_ret + env_ret(n_loc, loc_pic)
    else:
        subject = "subject_definitions:\n" + singer_def
        retention = "retention_analysis:\n" + singer_ret
    subject += cam_def
    retention += cam_ret
    desc = (text.get("detailed_description") or "").strip()
    if not desc.startswith("[Shot 1]"):
        desc = "[Shot 1] " + desc
    summary = (text.get("summary") or "").strip()
    if not summary.startswith("[reference generation]"):
        summary = "[reference generation] " + summary
    soundscape = (text.get("overall_soundscape") or "Near-silent ambience; no other sound.").strip()
    music = (text.get("non_diegetic_music") or "").strip()
    prompt = (subject + "\nsummary:\n" + summary + "\n\n" + retention + "\ndetailed_description:\n"
              + script["look"].strip() + "\n" + desc + "\n\noverall_soundscape:\n" + soundscape
              + "\n\nnon_diegetic_music:\n" + music + "\n")
    return prompt


I2V_OPENING = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."


def compile_prompt_i2v(script, scene, clip, text):
    """The image-to-video prompt for one clip (FL2VA weights, the location still as the FIRST
    FRAME, no reference pictures): the official three-section format behind the author's exact
    opening line (docs/MINIMAX_H3_PLAN.md s.4). Claude's four fields are reused UNCHANGED so the
    lane is the only variable against the ref2va plate of the same clip; the summary and the
    retention lines have no place in this format and are dropped."""
    desc = (text.get("detailed_description") or "").strip()
    if desc.startswith("[Shot 1]"):
        desc = desc[len("[Shot 1]"):].strip()
    # her scenes on this lane (opt-in): no subject is declared, she is the woman in frame 0
    person = "the man" if singer_of(script, scene) == "him" else "the woman"
    desc = re.sub(r"<Subject 1>\s*\(S1\)|<Subject 1>|\bS1\b", person, desc)
    desc = re.sub(r"(^|[.!?]\s+)" + person, lambda m: m.group(1) + person.capitalize(), desc)
    soundscape = (text.get("overall_soundscape") or "Near-silent ambience; no other sound.").strip()
    music = (text.get("non_diegetic_music") or "N/A").strip()
    return (I2V_OPENING + "\n\n"
            + "integrated_multimodal_description: [Shot 1] " + script["look"].strip() + " " + desc + "\n"
            + "overall_soundscape: " + soundscape + "\n"
            + "non_diegetic_music: " + music + "\n")


def opening_still_prompt(script, scene, location_row=None):
    """The prompt for a scene's OPENING still on the image-to-video lane: the frame-0 picture.
    Scenes written under the current rule carry it in `opening_still` with the people already
    placed; for older scripts (or when the scene lists `others` the stored prompt does not
    mention) this composes one mechanically from the location still prompt plus the listed
    people, so the first frame contains them (measured 2026-10-28: people absent from frame 0
    ghost in over the first second). Returns (prompt, source) with source "scene" or "composed"."""
    others = (scene.get("others") or "").strip()
    own = (scene.get("opening_still") or "").strip()
    if not others:
        return own, "scene"
    first_word = re.sub(r"[^a-z]", "", others.split()[0].lower()) if others.split() else ""
    if own and ("others" in own.lower() or (first_word and first_word in own.lower() and "people" in own.lower())):
        return own, "scene"
    row = location_row or {}
    base = (row.get("still_prompt") or "").strip()
    if not base:
        loc = location_of(script, scene) or {}
        base = (loc.get("still_prompt") or loc.get("description") or scene.get("setting") or "").strip()
    base = re.sub(r"(?i)\b(with )?no people\b[,.]?\s*", "", base).strip()
    base = re.sub(r"(?i)\bperson-free\b[,.]?\s*", "", base).strip()
    people = others.rstrip(".")
    # the clip pass states each person's opening pose ("At the start, ..."): frame 0 must match it
    poses = ""
    clips = scene.get("clips") or []
    desc = ((clips[0].get("text") or {}).get("detailed_description") or "") if clips else ""
    sents = re.split(r"(?<=[.!?])\s+", desc)
    for i, sent in enumerate(sents):
        if sent.lower().startswith("at the start"):
            take = [sent]
            for nxt in sents[i + 1:]:
                if re.match(r"(?i)(during|around|over|then|after|by |their |they |each |nobody|the camera|he (takes|walks|hauls)|she )", nxt):
                    break
                take.append(nxt)
            poses = " ".join(take).strip()
            break
    # THIS scene's weather, time of day and light override the location still's (measured
    # 2026-10-28, ENV14: a storm-lit opening still under a "clear evening, pale gold sky" scene
    # made the model dissolve the whole world into the scene's weather inside the first second)
    state = " ".join(x for x in ((scene.get("setting") or "").strip(), (scene.get("lighting") or "").strip()) if x)
    state_line = (f"THIS SCENE'S STATE, which overrides any weather, time of day, light or open/closed state "
                  f"described above: {state} ") if state else ""
    prompt = (f"{base} {state_line}In this frame the people of the scene are already in place: {people}. "
              + (f"Opening poses, exactly: {poses} " if poses else "")
              + "Each is caught mid-activity in a different pose, as in an unposed documentary frame, none of them in unison; "
              "faces turned away from camera or softened by distance and shadow, small in frame, dressed for "
              "the period and the place. Same camera position, framing, fixed elements, light and film look as "
              "the location still.")
    return re.sub(r"\s+", " ", prompt).strip(), "composed"


def compile_scene(script, scene, with_location=True, cam_video=False, lane=None):
    """Recompile every clip prompt of a scene from its stored text (the writer keeps the four
    fields per clip), with or without the location picture. Clips with no stored text keep the
    prompt they have. Returns the list of prompts in clip order.
    lane="i2v": the image-to-video prompt instead (first-frame lane, no references)."""
    env = location_of(script, scene) if with_location else None
    distant = is_distant(scene)
    out = []
    for c in scene["clips"]:
        if c.get("text") and lane == "i2v":
            out.append(compile_prompt_i2v(script, scene, c, c["text"]))
        elif c.get("text"):
            out.append(compile_prompt(script, scene, c, c["text"], env=env, cam_video=cam_video, distant=distant))
        else:
            out.append(c["prompt"])
    return out


def check_prompt(prompt, clip, performance="sing_to_camera"):
    """The hard rules, mechanically. Returns a list of problems (empty = clean). A no-singing
    scene may have vocal audible in its window (the corridor walk under the verse) - that is
    the point of B-roll, so the <d> requirement only applies to singing scenes."""
    problems = []
    sings = str(performance or "").startswith("sing")
    if "<Audio" in prompt:
        problems.append("names <Audio N> (the song is locked in the latent)")
    if "[Shot 2]" in prompt or "<scenetrans>" in prompt:
        problems.append("more than one shot in a chain clip")
    if sings and clip["vocal"] and "<d>[English]" not in prompt:
        problems.append("vocal clip in a singing scene without a <d>[English] ...</d> line")
    if not sings and "<d>" in prompt:
        problems.append("no-singing scene carries a <d> lyric line")
    if not clip["vocal"] and "<d>" in prompt:
        problems.append("instrumental clip carries a <d> line")
    for sec in ("subject_definitions:", "summary:", "retention_analysis:", "detailed_description:",
                "overall_soundscape:", "non_diegetic_music:"):
        if sec not in prompt:
            problems.append(f"missing section {sec}")
    problems += pace_problems(prompt, performance)
    return problems


SKY_WORDS = re.compile(r"(?i)\b(sky|skies|cloud\w*|overcast|stars?|milky way)\b")
SKY_MOVES = re.compile(r"(?i)\b(roll\w*|drift\w*|creep\w*|rac(e|es|ing)|scud\w*|stream\w*|sweep\w*|swirl\w*|churn\w*"
                       r"|boil\w*|tumbl\w*|gather\w*|mov(e|es|ing)|pass(es|ing)?|advanc\w*|spread\w*|surg\w*|tear(s|ing)"
                       r"|sweep\w*|blow\w*)\b")
SKY_PINNED = re.compile(r"(?i)\bno (cloud )?(drift|movement|cloud movement)\b|\bhold\w* (fixed|still)\b"
                        r"|\b(do|does) not (move|drift)\b|\bstays? (still|fixed)\b|\bmotionless\b")


def pace_problems(prompt, performance=None):
    """Pace rules that read as a timelapse when broken, checked on the compiled prompt of either
    lane. Run at write time (check_prompt) and again in the render dry run, so a scene that
    breaks them never reaches the GPU unseen. A list of problems, empty = clean."""
    desc = prompt
    for head in ("detailed_description:", "integrated_multimodal_description:"):
        if head in desc:
            desc = desc.split(head, 1)[1]
    desc = desc.split("overall_soundscape:", 1)[0]
    problems = []
    if str(performance or "") == "no_singing_environment":
        # measured 2026-10-28: a rhythm or frequency phrase in an environment-only description
        # (boats "once every two seconds", shutters "with each gust") rendered as a timelapse
        m = re.search(r"\b(every|each) (one|two|three|four|five|few|couple of|\d+) seconds?\b"
                      r"|\bwith each\b|\bagain and again\b|\bevery few\b", desc, re.I)
        if m:
            problems.append(f"environment clip carries a rhythm phrase ('{m.group(0)}'); one slow motion, counted over the whole clip")
    # rulebook s.8 rule 2: unmentioned sky gets animated and sky motion is the strongest timelapse
    # cue; 2026-10-29 scene 0 made the cloud bank its one moving element and rendered as a timelapse
    # the look line ("grey-white skies" in the grade) and "no sky shows" are not a visible sky
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", desc)
                 if not re.search(r"(?i)\b(film|grade|grain|palette)\b|\bno (sky|skies)\b", s)]
    for s in sentences:
        if SKY_WORDS.search(s) and SKY_MOVES.search(s) and not SKY_PINNED.search(s):
            problems.append(f"the sky moves ('{s.strip()[:120]}'); sky motion reads as a timelapse: pin it, move something else")
            break
    if any(SKY_WORDS.search(s) for s in sentences) and not any(SKY_WORDS.search(s) and SKY_PINNED.search(s) for s in sentences):
        problems.append("sky or cloud is visible but never pinned ('holds fixed positions with no drift and no cloud movement')")
    return problems


# ----------------------------------------------------------------------------- the whole thing

def write_script(audio_path, audio_id, title, style, lyrics_raw, duration, refs, concept=None,
                 provider="claude_sub", model=WRITER_MODEL, effort=WRITER_EFFORT, claude_model="",
                 cache_dir=None, log=print, align_path=None, known_locations=None, locks=None, duet=None):
    """Song -> full script JSON (scenes with chain plans and compiled per-clip prompts).
    `locks` = takes the user passed, kept as they are ({start, end, scene: the source scene with
    its clips, locked: {job_id, take_start, script}}): the writer fills only the gaps and pass 2
    skips them.
    `refs` = {"face": one-line description of <Picture 1>, "body": of <Picture 2>,
              "shared": a sentence naming what BOTH show (the details the prompt may mention)}.
    `known_locations` = library rows (mv_locations) the model may reuse by id instead of
    inventing a new place; a reused row carries its still id into the script.
    `duet` (optional, two-singer songs) = {"refs2": {face, body, shared}: his two pictures, described
    like `refs`; "voice": how he sings; "measure": sections -> the measured voice timeline
    (mv_voices.timeline)}. The timeline tags every aligned line with its voice; each scene then
    names its singer and the checker holds lip-sync scenes to the voice that is heard."""
    facts = song_facts(audio_path, lyrics_raw, duration, cache_dir=cache_dir, align_path=align_path)
    log(f"[mvw] facts: {len(facts['sections'])} sections, cover {facts.get('cover')}, aligned on {facts['aligned_on']}")
    duet_rec = None
    if duet:
        from . import mv_voices
        tl = duet["measure"](facts["sections"])
        mv_voices.apply(facts["sections"], tl)
        duet_rec = {"refs2": duet["refs2"], "voice": duet.get("voice") or "", "how": tl["how"], "swap": tl["swap"],
                    "separation": tl["separation"], "runs": tl["runs"], "answers": tl.get("answers") or [],
                    "sung_seconds": tl["sung_seconds"]}
        log(f"[mvw] voices: her {tl['sung_seconds']['her']}s, him {tl['sung_seconds']['him']}s ({tl['how']})")
    refs_text = (f"<Picture 1> (face / identity): {refs['face']}\n<Picture 2> (full body / wardrobe): {refs['body']}\n"
                 f"Both show: {refs['shared']}")
    if duet:
        r2 = duet["refs2"]
        refs_text = ("SHE, the female lead:\n" + refs_text + "\n\nHE, the male singer (his own two pictures):\n"
                     f"face / identity: {r2['face']}\nfull body / wardrobe: {r2['body']}\nBoth show: {r2['shared']}")
    system, user = scene_prompt(title, style, lyrics_raw, facts, refs_text, concept, known=known_locations, locks=locks,
                                duet=duet_rec)
    raw = llm_mod.complete(provider, model, system, user, claude_model, timeout=WRITER_TIMEOUT, effort=effort)
    _keep_raw(cache_dir, f"{audio_id}_pass1.txt", raw)
    plan = _extract_json(raw)
    scenes = normalize_scenes(plan, facts["sections"], locks=locks)
    locations = normalize_locations(plan, scenes, known=known_locations)
    log(f"[mvw] pass 1: {len(scenes)} scenes, {len(locations)} locations "
        f"({sum(1 for L in locations if L.get('known'))} known, {len(known_locations or [])} offered)")
    script = {"title": title, "audio_id": audio_id, "style": style, "duration": duration,
              "concept": plan.get("concept", ""), "look": plan.get("look", ""),
              "wardrobe": plan.get("wardrobe", ""), "arc": plan.get("arc", ""), "refs": refs, "facts": facts,
              **({"duet": duet_rec, "wardrobe2": plan.get("wardrobe2", "")} if duet_rec else {}),
              "locations": locations, "known_offered": [r["id"] for r in (known_locations or [])],
              "writer": {"provider": provider, "model": model, "effort": effort}, "scenes": []}
    for sc in scenes:
        if not sc.get("locked") and not sc.get("gap"):
            write_scene_clips(script, sc, provider, model, effort, claude_model, cache_dir, log)
        sc.setdefault("clips", [])
        script["scenes"].append(sc)
    script["clip_count"] = sum(len(s["clips"]) for s in script["scenes"])
    mark_continuity(script)
    return script


def write_scene_clips(script, sc, provider="claude_sub", model=WRITER_MODEL, effort=WRITER_EFFORT,
                      claude_model="", cache_dir=None, log=print):
    """Pass 2 for ONE scene: plan its clips, ask Claude for the four fields per clip, compile and
    check. Mutates and returns the scene (plan, clips)."""
    cplan, clips = plan_scene_clips(sc, script["facts"]["sections"])
    system2, user2 = clip_prompt(script, sc, clips)
    raw2 = llm_mod.complete(provider, model, system2, user2, claude_model, timeout=WRITER_TIMEOUT, effort=effort)
    _keep_raw(cache_dir, f"{script['audio_id']}_pass2_scene{sc['index']}_{_slug(sc.get('title'))[:24]}.txt", raw2)
    texts = {int(t.get("index", i)): t for i, t in enumerate((_extract_json(raw2) or {}).get("clips") or [])}
    for c in clips:
        t = texts.get(c["index"]) or {}
        c["text"] = {k: t.get(k, "") for k in ("summary", "detailed_description", "overall_soundscape", "non_diegetic_music")}
        c["prompt"] = compile_prompt(script, sc, c, c["text"], env=location_of(script, sc))
        c["problems"] = check_prompt(c["prompt"], c, sc.get("performance"))
    sc["plan"] = cplan
    sc["clips"] = clips
    log(f"[mvw] pass 2: scene {sc['index']} '{sc.get('title')}' {sc['start']}-{sc['end']}s -> {len(clips)} clips, "
        f"problems {sum(len(c['problems']) for c in clips)}")
    return sc


SPLIT_SNAP_TOL = 1.0     # a split snaps to a cut point this close; further away (an instrumental span) it stays put


def split_scene(script, index, at, second):
    """Cut scene `index` in two at `at` seconds (snapped to a cut point when one lies within
    SPLIT_SNAP_TOL; inside a lineless span, an intro or a solo, the cut stays where asked). The
    first half keeps the scene's fields (override with second['first'] if given) including its
    filed opening still; the second half takes `second`'s fields over a copy and starts without
    an opening still (it is a new setup, so it needs its own). Both halves lose their clips (run
    write_scene_clips on each). Scenes are renumbered. Returns (first, second)."""
    sections = script["facts"]["sections"]
    at = float(at)
    near = snap_cut(at, cut_points(sections))
    if abs(near - at) <= SPLIT_SNAP_TOL:
        at = near
    sc = script["scenes"][index]
    if not sc["start"] < at < sc["end"]:
        raise ValueError(f"cut {at}s is not inside scene {index} ({sc['start']}-{sc['end']}s)")
    keep = {k: v for k, v in sc.items() if k not in ("plan", "clips", "index", "sections", "section_types", "lipsync")}
    a = dict(keep, **(second.get("first") or {}))
    b = dict({k: v for k, v in keep.items() if k not in ("opening_still_id", "locked")},
             **{k: v for k, v in second.items() if k != "first"})
    if a.get("locked"):
        if second.get("first"):
            a.pop("locked")                 # a changed first half is a new shot: it re-renders
        else:
            a["plan"], a["clips"] = sc.get("plan"), sc.get("clips")   # the passed take, trimmed to the first half
    a["start"], a["end"] = sc["start"], at
    b["start"], b["end"] = at, sc["end"]
    script["scenes"][index:index + 1] = [a, b]
    for i, x in enumerate(script["scenes"]):
        x["index"] = i
    _finish_scene(a, sections)
    _finish_scene(b, sections)
    return a, b


def move_cut(script, index, to, first=None, second=None):
    """Move the boundary between scene `index` and scene `index + 1` to `to` seconds, which must
    lie within SPLIT_SNAP_TOL of a cut point (a boundary is always a cut point; the first-line
    setup point counts). `first` / `second` are field overrides for the two scenes. A scene whose
    framing or performance changes loses its filed opening still (it was made for the old shot).
    Both scenes lose their clips (run write_scene_clips on each). Returns (first, second)."""
    sections = script["facts"]["sections"]
    to = float(to)
    # two singers: a shared line's measured hand-over is also a cut point here (the user's edit
    # "he finishes the line", 2026-11-08); the writer itself keeps shared lines on her
    handovers = [l["voice_switch"] for s in sections for l in s.get("lines") or [] if l.get("voice_switch")]
    near = snap_scene_cut(to, sections, sorted(cut_points(sections) + handovers))   # free inside a lineless section
    if abs(near - to) > SPLIT_SNAP_TOL:
        raise ValueError(f"{to}s is not within {SPLIT_SNAP_TOL}s of a cut point (nearest {near}s)")
    to = near
    if not 0 <= index < len(script["scenes"]) - 1:
        raise ValueError(f"scene {index} has no scene after it")
    a, b = script["scenes"][index], script["scenes"][index + 1]
    if not a["start"] < to < b["end"]:
        raise ValueError(f"cut {to}s must fall inside {a['start']}-{b['end']}s (scenes {index} and {index + 1})")
    # a passed take keeps its lock only when its scene merely shrinks inside it, unchanged;
    # otherwise the scene is a new shot and re-renders
    a_keeps = bool(a.get("locked")) and not first and to < a["end"]
    b_keeps = bool(b.get("locked")) and not second and to > b["start"]
    for sc, ch, keeps in ((a, first, a_keeps), (b, second, b_keeps)):
        ch = dict(ch or {})
        if ("framing" in ch and ch["framing"] != sc.get("framing")) or \
           ("performance" in ch and ch["performance"] != sc.get("performance")):
            sc.pop("opening_still_id", None)
        sc.update(ch)
        if not keeps:
            sc.pop("locked", None)
            sc.pop("plan", None)
            sc.pop("clips", None)
    a["end"], b["start"] = to, to
    _finish_scene(a, sections)
    _finish_scene(b, sections)
    return a, b


def script_markdown(script):
    """A readable rendering of a script for review (the user reads this, not the JSON)."""
    out = [f"# {script['title']} - music video script", "",
           f"Writer: {script['writer']['model']} ({script['writer']['effort']} effort). "
           f"Song {script['duration']:.1f}s, lyric cover {script['facts'].get('cover')}. "
           f"{len(script['scenes'])} scenes, {script['clip_count']} clips.", "",
           f"**Concept.** {script['concept']}", "", f"**Look.** {script['look']}", "",
           f"**Wardrobe.** {script['wardrobe']}", ""]
    if script.get("duet"):
        d = script["duet"]
        out += [f"**His wardrobe.** {script.get('wardrobe2')}", "",
                f"**Voices (measured).** her {d['sung_seconds']['her']}s, him {d['sung_seconds']['him']}s; {d['how']}.", ""]
    if script.get("arc"):
        out += [f"**Arc.** {script['arc']}", ""]
    # only plates: her scenes render on the reference lane, where a still is not frame 0
    need = [sc for sc in script["scenes"] if not sc.get("locked") and sc.get("matches_location_still") is False
            and sc.get("performance") == "no_singing_environment" and not sc.get("opening_still_id")]
    if need:
        out += ["**Opening stills needed** (the scene's light, weather or people differ from its location still): "
                + ", ".join(f"scene {sc['index'] + 1} {sc.get('title')}" for sc in need), ""]
    if script.get("locations"):
        out += ["## Locations (one person-free still each, handed to the chain as the place reference)", ""]
        for L in script["locations"]:
            used = [str(sc["index"] + 1) for sc in script["scenes"] if sc.get("location") == L["id"]]
            out += [f"### {L['name']}  (`{L['id']}`, scenes {', '.join(used) or '-'}"
                    + (", derived from a scene setting" if L.get("derived") else "")
                    + (", KNOWN: still on file" if L.get("known") else "") + ")", "",
                    f"- shows: {L.get('description')}", ""]
            if L.get("known"):
                out += [f"- still: library {L.get('still_id') or '(none yet)'}; nothing to generate", ""]
            else:
                out += ["Still prompt:", "", "```", (L.get("still_prompt") or "").strip(), "```", ""]
    for sc in script["scenes"]:
        out += [f"## Scene {sc['index'] + 1}: {sc.get('title', '')}  ({sc['start']:.1f}-{sc['end']:.1f}s, "
                f"{', '.join(sc['section_types'])})"
                + (f"  LOCKED: passed take, from {sc['locked'].get('take_start', sc['start']):.2f}s of its render"
                   if sc.get("locked") else ""), "",
                f"- performance: {sc.get('performance')}" + (f" | singer: {sc.get('singer')}" if script.get("duet") else "")
                + f" | framing: {sc.get('framing')} | camera: {sc.get('camera')}"
                + (f" | keep in frame: {sc.get('keep_in_frame')}" if sc.get("keep_in_frame") else "")
                + (f" | location: `{sc.get('location')}`" if sc.get("location") else ""),
                f"- setting: {sc.get('setting')}", f"- lighting: {sc.get('lighting')} | mood: {sc.get('mood')}",
                f"- action: {sc.get('action')}", f"- why: {sc.get('why')}",
                f"- opening still prompt: {sc.get('opening_still')}", ""]
        if sc.get("scene_problems"):
            out += ["SCENE PROBLEMS: " + "; ".join(sc["scene_problems"]), ""]
        for c in sc["clips"]:
            words = " / ".join(l["text"] for l in c["lines"]) or "(instrumental)"
            out += [f"### Clip {c['index'] + 1}: song {c['song_start']:.2f}-{c['song_end']:.2f}s, {c['frames']}f", "",
                    f"Lines audible: {words}", ""]
            if c["problems"]:
                out += ["PROBLEMS: " + "; ".join(c["problems"]), ""]
            out += ["```", c["prompt"].rstrip(), "```", ""]
    return "\n".join(out)
