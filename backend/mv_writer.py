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
    for sec in align["sections"]:
        if sec.get("start") is None:
            continue
        lines = [{"start": l["start"], "end": l["end"], "text": l["text"], "cover": l["cover"]}
                 for l in sec["lines"] if l.get("start") is not None]
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
    return {"duration": duration, "cover": align.get("cover"), "words_matched": align.get("words_matched"),
            "words_total": align.get("words_total"), "sections": table, "blocks": blocks,
            "aligned_on": "vocals" if align_path else "mix"}


# ----------------------------------------------------------------------------- pass 1: scenes

SCENE_SYSTEM = """You are the director of a music video for one song, writing for a video model (MiniMax H3) that renders each scene as one continuous take of up to 15 seconds, chained into longer takes when needed, with the real song locked to the picture. You return STRICT JSON only - no prose before or after it, no markdown fences.

Fixed facts about this production:
- The singer is the one lead and the only person the references define. There is no band: never a band, never other musicians, never a performance crowd by default. Other people are allowed where the story needs them (a landlord, a fisherman on the far quay, a child at a window, mourners at a grave): list them in the scene's "others" field, anonymous and few, each with a place in frame and their own ordinary business, and write null when the scene needs nobody. People never act in unison (measured: three men written as "one slow movement together" rendered as a synchronised puppet show); each does a different thing, staggered in time. Extras do SMALL ordinary things only: walking, standing, hauling, drinking, talking, coiling a rope. Never a large body transition for an extra: no climbing, jumping, stepping down into or out of a boat, mounting, falling; and only actions the location's geometry allows (the harbour's boats are moored off the quay on their ropes, so nobody can step into one). Measured: "one steps down into the nearest boat" rendered as a man flying through the air onto a boat that was too far away to reach. Most scenes need nobody; the video is about her.
- Her identity and wardrobe come from two reference photographs described to you. One wardrobe for the whole video; you may add weathering (rain, dust, wind in the hair) but not change garments.
- She lip-syncs where it serves the video, not in every scene. A good music video for this kind of song mixes performance with cinematic B-roll: environment, symbolic imagery, her moving through a place without singing, details, wide shots. You decide the mix from the song's content and its genre.
- Each scene is ONE continuous take (no cuts inside a scene). Scene changes are hard cuts. A performance (lip-sync) take runs 8-15 seconds by default and never more than 28 seconds; a B-roll take may run up to 45 seconds; nothing shorter than 5 seconds. A verse or a chorus is therefore covered by two or three SETUPS in the same location, cut between them: a change of angle or framing (chair front-on, then three-quarter profile from the lamp side; wide, then detail), each its own scene with its own camera. The location reference keeps the place consistent across those cuts.
- Scene boundaries fall on CUT POINTS: a section edge, or one of the listed line cut points (half a second before a sung line begins). Choose cuts where the words break.
- Inside a scene, the camera does ONE thing, written in the video model's own grammar (measured: it obeys this and ignores metric distances): "static", or "<verb> with <small|large> amplitude at <slow|fast> speed toward/along X throughout the take". Verbs: push in, pull out, pan left/right, truck left/right, tilt up/down, pedestal up/down, arc, tracking. Performance takes use small amplitude at slow speed; large amplitude only on B-roll. A scene has ONE framing (the one it starts in); never write a second framing the move should reach (measured: "from a medium shot to a medium close-up" became a hidden cut rendered as a dissolve, with the wardrobe changing across it). Bound a move only by what must stay in frame to the end (the lamp, the window, her shoulders). No whip pans, no handheld chaos, no orbit-plus-zoom, never centimetres or metres.
- Lip-sync scenes need her mouth readable: medium close-up or close-up, face toward camera. Wider performance is fine when the words matter less.
- Settings must be concrete and filmable in one place, described without people. If sky, clouds or stars are visible, say so (the renderer must pin them).
- Environment-only B-roll (no person in frame) moves ONE element only (the sea, a curtain, the rain, a flame), slowly, with everything else still. A second moving element, or any rhythm ("every few seconds", "with each gust"), makes the model run the shot as a timelapse (measured). Sounds belong to the soundscape, not to the picture.
- LOCATIONS: every scene happens in one named location, and the video uses FEW locations (typically 3-6), each returned to by several scenes. For each location you write a person-free still prompt for an image model; that still is given to the video model as a reference picture of the place, so every scene set there renders the same room or landscape. A location is one physical place at one camera distance class (a room; a detail of that room counts as its own location). Weather and time of day may change between scenes in the same location; walls, furniture, window placement and materials may not. The parlour the reference photographs show is a location. When the brief lists KNOWN LOCATIONS (places already photographed for earlier scripts or other videos, their stills on file), prefer them wherever one serves THIS video and refer to each by its exact id; invent a new location only where no known one fits the song. Never redecorate a known place: its fixed elements are what its description says, and your scene settings there must match that description.
- Timing is NOT yours beyond choosing cut points: the system computes clip windows and which words fall where.

Write for the song's genre: its pace, weight and mood should be visible in the choice of places, light, weather, movement and the proportion of performance to B-roll."""


LINE_LEAD = 0.5          # a line's cut point sits this far before the line begins


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
    return sorted(pts)


def snap_cut(t, points):
    return min(points, key=lambda x: abs(x - t))


def _section_text(sections):
    out = []
    for s in sections:
        head = f"[{s['index']}] {s['type']}  {s['start']:.2f}-{s['end']:.2f}s  ({s['end'] - s['start']:.1f}s)"
        if s["vocal"]:
            out.append(head)
            prev_end = s["start"]
            for l in s["lines"]:
                c = max(l["start"] - LINE_LEAD, prev_end)
                tag = f"cut {c:6.2f} |" if s["start"] < c < s["end"] else "            |"
                out.append(f"      {tag} {l['start']:7.2f}-{l['end']:6.2f}  {l['text']}")
                prev_end = l["end"]
        else:
            out.append(head + "  instrumental, no vocal")
    return "\n".join(out)


def scene_prompt(title, style, lyrics_raw, facts, refs, concept=None, known=None):
    """known: library rows (mv_locations) the model may reuse by id; None/[] = none offered."""
    schema = {
        "concept": "2-3 sentences: what the video is about and how it moves (string)",
        "look": "one sentence of visual style for every prompt: film stock feel, light, contrast, colour, camera weight (string)",
        "wardrobe": "the singer's one outfit, as it appears in the references, in one sentence (string)",
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
            "framing": "e.g. medium close-up, medium, wide, extreme wide, detail",
            "camera": "'static', or ONE move in the model's grammar: '<verb> with small amplitude at slow speed toward X throughout the take' (one framing only, never a second framing to reach, no distances)",
            "keep_in_frame": "one or two fixed things that must stay in frame until the end of the take, e.g. 'the oil lamp and the window' (string)",
            "lighting": "one sentence",
            "mood": "three words",
            "action": "what she does, 1-3 sentences, one main motion; for environment B-roll exactly ONE moving element, slow, and what stays still",
            "others": "null, or one sentence: the other people this scene needs (who, how many, where in frame, what they do), anonymous; no band",
            "opening_still": "a prompt for a photoreal 16:9 still of the FIRST FRAME of this scene, one paragraph: the location still's camera position, framing, fixed elements and light, with she in the setting where the scene has her, and with every person listed in others ALREADY IN PLACE in their opening pose (each caught mid-activity, each different, faces turned away or softened by shadow, small in frame). On the image-to-video lane this still IS frame 0, so anyone missing from it has to fade in (measured)",
            "why": "one sentence: why this scene belongs at this point of the song"
        }],
    }
    user = (f"SONG: {title}\nSTYLE / GENRE: {style}\n"
            f"DURATION: {facts['duration']:.1f}s. Lyric alignment cover {facts.get('cover')}.\n\n"
            f"REFERENCES (the only person in the video):\n{refs}\n\n"
            + (f"USER CONCEPT (follow it): {concept}\n\n" if concept else "")
            + (("KNOWN LOCATIONS (already photographed; a still of each is on file). Reuse one by its exact id "
                "with {\"id\": ..., \"reuse\": true}; a new place needs a full entry with a still prompt:\n"
                + known_text(known) + "\n\n") if known else "")
            + "MEASURED SECTIONS (real seconds on this recording; use these indices):\n"
            + _section_text(facts["sections"]) + "\n\nLYRICS:\n" + (lyrics_raw or "").strip()
            + "\n\nThe scenes tile the whole song in order with no gaps: the first starts at 0, each `to` is the "
            "next scene's `from`, the last ends at the song's end. Every from/to is a section edge or a listed "
            "cut point. Return JSON with exactly this shape:\n" + json.dumps(schema, indent=1))
    return SCENE_SYSTEM, user


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
    return sc


def normalize_scenes(plan, sections):
    """Make the scene list tile the song in order. Scenes given as from/to seconds are snapped to
    cut points (section edges and line cut points) and forced contiguous; scenes given as a
    `sections` range (older answers) take those sections' edges. Empty scenes are dropped. Returns
    the scenes with index/start/end/sections/section_types/lipsync filled in."""
    n = len(sections)
    pts = cut_points(sections)
    song_end = sections[-1]["end"]
    raw = plan.get("scenes") or []
    scenes, cursor = [], 0.0
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
        b = snap_cut(b, pts)
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
    return "\n".join(f"- id `{r['id']}` ({r.get('name')}): {r.get('description') or '(no description)'}" for r in rows)


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
                lines.append({"text": l["text"], "song_start": l["start"], "song_end": l["end"],
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
- The singer is <Subject 1>. When she sings, write: <Subject 1> (S1) sings <d>[English] exact words</d>. Only the words audible inside THIS clip's window go inside <d>; a line that starts before the clip begins mid-phrase; a line that ends after the clip is cut mid-phrase. Instrumental passages have no <d>. Lips closed and relaxed when no vocal is audible; one open vowel shape held through sustained notes.
- detailed_description starts with "[Shot 1]" and is ONE continuous shot. No [Shot 2], no cuts, no <scenetrans>.
- Exactly ONE camera motion for the scene, as given, repeated in every clip in the model's own grammar: "The camera pushes in with small amplitude at slow speed toward X throughout the take, one continuous shot with no cut and no dissolve." Name what must stay in frame ("The oil lamp and the window stay in frame to the end.") and never name a second framing for the move to reach (that reads as a cut). Never metric distances (centimetres mean nothing to the model). If static: "The camera is static, locked off for the full clip."
- One main physical motion for her per clip, with a duration anchor (how long it takes, in seconds). Hands: say what they do or that they stay still. Hair: say whether it moves. If sky/clouds/stars are visible: say they hold fixed positions with no drift and no cloud movement.
- Environment-only clips (performance no_singing_environment) have no person to anchor real time, so the discipline is stricter: exactly ONE moving element in the whole description (the boats on the swell, OR the shutters, OR the rain), named once, with a whole-clip anchor stated as a count ("over the whole 6 seconds the boats lift and settle once, slowly"). Every other element is said to be still, by name. No second moving element, no rhythm or frequency ("every two seconds", "with each gust", "again and again"), never water plus wind plus weather together. The camera move is the only other motion. Sounds of motion (creak, rattle, slap) go in overall_soundscape, never in detailed_description. (Measured: a harbour clip written with boats, ropes, shutters and water and a two-second rhythm rendered as a timelapse.)
- Clip 2 and later CONTINUE: the first lines say that the incoming motion, camera, framing, lighting and state continue uninterrupted from the protected video context, then say what that ongoing motion becomes. Never restart an action already in progress, never re-establish the shot, never return to the reference pose.
- Never change the wardrobe. Other people appear ONLY when the scene lists them (its others field, or anchor_figure: true): write exactly those, anonymous, never a face that competes with hers for the lens, and never anyone the scene did not list; never a band. People are NOT the environment's one moving element and the one-motion rule does not apply to them: real people never act in unison. Give each person their own ordinary business, staggered in time (one drinks while another talks and the third leans back; a laugh starts with one man and the others follow a second or two later), with the small continuous idle movement of living people (weight shifts, a hand turning a tankard, a head turn, a glance). Say what each is doing at the start of the clip (it must match the opening still) and give the group's rhythm across the clip in seconds without choreographing them together. Extras keep to small ordinary actions (walk, stand, haul, drink, talk, coil); never write a large body transition for an extra (climbing, jumping, stepping down into or out of a boat, mounting, falling), and nothing the still's geometry makes impossible (a boat moored off the quay cannot be stepped into): the model renders the impossible as a leap (measured). (Measured: "over the whole 6 seconds they make one slow movement together" rendered three men raising tankards and throwing their heads back in perfect sync, which read as unnatural.) The anchor_figure case is one distant anonymous figure (a fisherman in oilskins, a woman with a basket), small in frame, back to camera or face unreadable, doing ONE slow everyday action at a natural human pace for the whole clip (walking the far quay, hauling a rope hand over hand, sweeping a step). It is the real-time anchor of the shot: say how many steps or pulls it makes over the clip's seconds. Never near the camera, never a second figure, never interacting with anything the singer owns.
- The scene's location is also a referenced subject (a person-free photograph of the place is handed to the model). Call the place by its name (the parlour, the quarry); the system numbers subjects and pictures. Do not redesign the place: its fixed elements are those in the scene setting; you may add weather, time of day and the state changes the scene calls for.
- Lip-sync clips: mouth readable (face toward camera, medium close-up or closer, nothing covering the mouth).
- 150-320 words of detailed_description per clip; concrete, visual, in order of time."""


def clip_prompt(script, scene, clips):
    loc = location_of(script, scene)
    user = {
        "song": script["title"], "look": script["look"], "wardrobe": script["wardrobe"],
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
    return CLIP_SYSTEM, json.dumps(user, indent=1)


def compile_prompt(script, scene, clip, text, env=None, cam_video=False):
    """The six-section Ref2VA prompt for one clip: fixed boilerplate from the script's references
    and wardrobe + Claude's four fields. The style opening is the script's look sentence.

    env: the scene's location row when a person-free still of the place goes into the chain as
    the last reference picture (<Picture 3> behind her two, or <Picture 1> alone in an
    environment-only scene); None = no location picture (her two pictures only, or none at all
    for an environment-only scene on the FL2VA weights). Picture numbering here must match the
    order the route uploads them in: face, body, location.
    cam_video: a manufactured camera-pace clip rides along as <Video 1>; it is declared as pace
    and movement only (the official guide's camera-reference role, weak_reference)."""
    refs = script["refs"]
    env_only = scene.get("performance") == "no_singing_environment"
    singer_def = ("<Subject 1> is the singer: her facial identity comes from <Picture 1>; "
                  "her full-body appearance, body proportions and wardrobe come from <Picture 2>. "
                  f"{refs['shared']} Wardrobe: {script['wardrobe']}\n")
    singer_ret = ("<Subject 1> (appears throughout [Shot 1]): fully_preserved - retain the "
                  "referenced face, eyes, hair, skin, body proportions and the wardrobe exactly, while pose, "
                  "expression, framing, lighting and movement follow the performance described below.\n")

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
        subject = "subject_definitions:\n" + singer_def + env_def(2, 3)
        retention = "retention_analysis:\n" + singer_ret + env_ret(2, 3)
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
    out = []
    for c in scene["clips"]:
        if c.get("text") and lane == "i2v":
            out.append(compile_prompt_i2v(script, scene, c, c["text"]))
        elif c.get("text"):
            out.append(compile_prompt(script, scene, c, c["text"], env=env, cam_video=cam_video))
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
    if str(performance or "") == "no_singing_environment":
        # measured 2026-10-28: a rhythm or frequency phrase in an environment-only description
        # (boats "once every two seconds", shutters "with each gust") rendered as a timelapse
        desc = prompt.split("detailed_description:", 1)[-1].split("overall_soundscape:", 1)[0]
        m = re.search(r"\b(every|each) (one|two|three|four|five|few|couple of|\d+) seconds?\b"
                      r"|\bwith each\b|\bagain and again\b|\bevery few\b", desc, re.I)
        if m:
            problems.append(f"environment clip carries a rhythm phrase ('{m.group(0)}'); one slow motion, counted over the whole clip")
    return problems


# ----------------------------------------------------------------------------- the whole thing

def write_script(audio_path, audio_id, title, style, lyrics_raw, duration, refs, concept=None,
                 provider="claude_sub", model=WRITER_MODEL, effort=WRITER_EFFORT, claude_model="",
                 cache_dir=None, log=print, align_path=None, known_locations=None):
    """Song -> full script JSON (scenes with chain plans and compiled per-clip prompts).
    `refs` = {"face": one-line description of <Picture 1>, "body": of <Picture 2>,
              "shared": a sentence naming what BOTH show (the details the prompt may mention)}.
    `known_locations` = library rows (mv_locations) the model may reuse by id instead of
    inventing a new place; a reused row carries its still id into the script."""
    facts = song_facts(audio_path, lyrics_raw, duration, cache_dir=cache_dir, align_path=align_path)
    log(f"[mvw] facts: {len(facts['sections'])} sections, cover {facts.get('cover')}, aligned on {facts['aligned_on']}")
    refs_text = (f"<Picture 1> (face / identity): {refs['face']}\n<Picture 2> (full body / wardrobe): {refs['body']}\n"
                 f"Both show: {refs['shared']}")
    system, user = scene_prompt(title, style, lyrics_raw, facts, refs_text, concept, known=known_locations)
    raw = llm_mod.complete(provider, model, system, user, claude_model, timeout=WRITER_TIMEOUT, effort=effort)
    _keep_raw(cache_dir, f"{audio_id}_pass1.txt", raw)
    plan = _extract_json(raw)
    scenes = normalize_scenes(plan, facts["sections"])
    locations = normalize_locations(plan, scenes, known=known_locations)
    log(f"[mvw] pass 1: {len(scenes)} scenes, {len(locations)} locations "
        f"({sum(1 for L in locations if L.get('known'))} known, {len(known_locations or [])} offered)")
    script = {"title": title, "audio_id": audio_id, "style": style, "duration": duration,
              "concept": plan.get("concept", ""), "look": plan.get("look", ""),
              "wardrobe": plan.get("wardrobe", ""), "refs": refs, "facts": facts, "locations": locations,
              "known_offered": [r["id"] for r in (known_locations or [])],
              "writer": {"provider": provider, "model": model, "effort": effort}, "scenes": []}
    for sc in scenes:
        write_scene_clips(script, sc, provider, model, effort, claude_model, cache_dir, log)
        script["scenes"].append(sc)
    script["clip_count"] = sum(len(s["clips"]) for s in script["scenes"])
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


def split_scene(script, index, at, second):
    """Cut scene `index` in two at `at` seconds (snapped to a cut point). The first half keeps the
    scene's fields (override with second['first'] if given); the second half takes `second`'s
    fields over a copy. Both halves lose their clips (run write_scene_clips on each). Scenes are
    renumbered. Returns (first, second)."""
    sections = script["facts"]["sections"]
    at = snap_cut(float(at), cut_points(sections))
    sc = script["scenes"][index]
    if not sc["start"] < at < sc["end"]:
        raise ValueError(f"cut {at}s is not inside scene {index} ({sc['start']}-{sc['end']}s)")
    keep = {k: v for k, v in sc.items() if k not in ("plan", "clips", "index", "sections", "section_types", "lipsync")}
    a = dict(keep, **(second.get("first") or {}))
    b = dict(keep, **{k: v for k, v in second.items() if k != "first"})
    a["start"], a["end"] = sc["start"], at
    b["start"], b["end"] = at, sc["end"]
    script["scenes"][index:index + 1] = [a, b]
    for i, x in enumerate(script["scenes"]):
        x["index"] = i
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
                f"{', '.join(sc['section_types'])})", "",
                f"- performance: {sc.get('performance')} | framing: {sc.get('framing')} | camera: {sc.get('camera')}"
                + (f" | keep in frame: {sc.get('keep_in_frame')}" if sc.get("keep_in_frame") else "")
                + (f" | location: `{sc.get('location')}`" if sc.get("location") else ""),
                f"- setting: {sc.get('setting')}", f"- lighting: {sc.get('lighting')} | mood: {sc.get('mood')}",
                f"- action: {sc.get('action')}", f"- why: {sc.get('why')}",
                f"- opening still prompt: {sc.get('opening_still')}", ""]
        for c in sc["clips"]:
            words = " / ".join(l["text"] for l in c["lines"]) or "(instrumental)"
            out += [f"### Clip {c['index'] + 1}: song {c['song_start']:.2f}-{c['song_end']:.2f}s, {c['frames']}f", "",
                    f"Lines audible: {words}", ""]
            if c["problems"]:
                out += ["PROBLEMS: " + "; ".join(c["problems"]), ""]
            out += ["```", c["prompt"].rstrip(), "```", ""]
    return "\n".join(out)
