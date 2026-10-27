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
a detail only one reference shows. Solo singer only - no other people, ever.

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


def song_facts(audio_path, lyrics_raw, duration, cache_dir=None):
    """Measured facts about THIS take: the aligned lines (real seconds, with cover) and a
    contiguous section table covering 0..duration. Sung sections come from the alignment;
    unsung gaps longer than GAP_IS_INSTRUMENTAL become instrumental sections (intro, solo,
    outro, breaks). Small gaps are absorbed into the preceding section."""
    blocks = parse_lyrics(lyrics_raw)
    align = lyricalign.align_song(audio_path, blocks, cache_dir=cache_dir)
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
            "words_total": align.get("words_total"), "sections": table, "blocks": blocks}


# ----------------------------------------------------------------------------- pass 1: scenes

SCENE_SYSTEM = """You are the director of a music video for one song, writing for a video model (MiniMax H3) that renders each scene as one continuous take of up to 15 seconds, chained into longer takes when needed, with the real song locked to the picture. You return STRICT JSON only - no prose before or after it, no markdown fences.

Fixed facts about this production:
- There is exactly ONE person in the whole video: the singer. No band, no extras, no crowd, no hands of others, no silhouettes of other people. Ever.
- Her identity and wardrobe come from two reference photographs described to you. One wardrobe for the whole video; you may add weathering (rain, dust, wind in the hair) but not change garments.
- She lip-syncs where it serves the video, not in every scene. A good music video for this kind of song mixes performance with cinematic B-roll: environment, symbolic imagery, her moving through a place without singing, details, wide shots. You decide the mix from the song's content and its genre.
- Each scene is a contiguous run of the song's measured sections and is rendered as ONE continuous take (no cuts inside a scene). Scene changes are hard cuts. Prefer scenes of 10-45 seconds; never shorter than 5 seconds.
- Inside a scene, the camera does ONE thing: static, or one slow move you can name with a real-world duration (a push-in that covers an arm's length over the whole scene). No whip pans, no handheld chaos, no orbit-plus-zoom.
- Lip-sync scenes need her mouth readable: medium close-up or close-up, face toward camera. Wider performance is fine when the words matter less.
- Settings must be concrete and filmable in one place, described without people. If sky, clouds or stars are visible, say so (the renderer must pin them).
- Timing is NOT yours: you choose which sections form a scene; the system computes clip windows and which words fall where.

Write for the song's genre: its pace, weight and mood should be visible in the choice of places, light, weather, movement and the proportion of performance to B-roll."""


def _section_text(sections):
    out = []
    for s in sections:
        head = f"[{s['index']}] {s['type']}  {s['start']:.2f}-{s['end']:.2f}s  ({s['end'] - s['start']:.1f}s)"
        if s["vocal"]:
            out.append(head)
            for l in s["lines"]:
                out.append(f"      {l['start']:7.2f}-{l['end']:6.2f}  {l['text']}")
        else:
            out.append(head + "  instrumental, no vocal")
    return "\n".join(out)


def scene_prompt(title, style, lyrics_raw, facts, refs, concept=None):
    schema = {
        "concept": "2-3 sentences: what the video is about and how it moves (string)",
        "look": "one sentence of visual style for every prompt: film stock feel, light, contrast, colour, camera weight (string)",
        "wardrobe": "the singer's one outfit, as it appears in the references, in one sentence (string)",
        "scenes": [{
            "sections": [0, 1],
            "title": "short label",
            "setting": "the place, concrete and filmable, no people, 2-3 sentences; name any visible sky",
            "performance": "one of: sing_to_camera | sing_away | no_singing_her | no_singing_environment",
            "framing": "e.g. medium close-up, medium, wide, extreme wide, detail",
            "camera": "static, or ONE slow move with a real-world distance and 'over the whole scene'",
            "lighting": "one sentence",
            "mood": "three words",
            "action": "what she does (or what moves, for environment B-roll), 1-3 sentences, one main motion",
            "opening_still": "a prompt for a photoreal still of the first frame of this scene (she in the setting, or the environment alone), one paragraph",
            "why": "one sentence: why this scene belongs at this point of the song"
        }],
    }
    user = (f"SONG: {title}\nSTYLE / GENRE: {style}\n"
            f"DURATION: {facts['duration']:.1f}s. Lyric alignment cover {facts.get('cover')}.\n\n"
            f"REFERENCES (the only person in the video):\n{refs}\n\n"
            + (f"USER CONCEPT (follow it): {concept}\n\n" if concept else "")
            + "MEASURED SECTIONS (real seconds on this recording; use these indices):\n"
            + _section_text(facts["sections"]) + "\n\nLYRICS:\n" + (lyrics_raw or "").strip()
            + "\n\nCover EVERY section index exactly once, in order, across the scenes; `sections` is a JSON "
            "array of two integers [first_index, last_index] (the same index twice for a one-section scene). "
            "Return JSON with exactly this shape:\n" + json.dumps(schema, indent=1))
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


def normalize_scenes(plan, sections):
    """Make the scene list cover every section exactly once, in order. Claude's ranges are kept
    where valid; gaps are attached to the previous scene, overlaps trimmed, and anything beyond
    the table dropped. Returns the scenes with start/end seconds filled in."""
    n = len(sections)
    raw = plan.get("scenes") or []
    scenes, cursor = [], 0
    for sc in raw:
        rng = sc.get("sections")
        if isinstance(rng, str):                      # "[3, 4]", "3-4", "3" all seen from the model
            rng = [int(x) for x in re.findall(r"\d+", rng)]
        elif isinstance(rng, (int, float)):
            rng = [int(rng)]
        try:
            a, b = int(rng[0]), int(rng[-1])
        except Exception:
            continue
        a = max(a, cursor)
        b = min(max(b, a), n - 1)
        if a > b or a >= n:
            continue
        if a > cursor and scenes:
            scenes[-1]["sections"] = [scenes[-1]["sections"][0], a - 1]
        elif a > cursor:
            a = cursor
        sc = dict(sc)
        sc["sections"] = [a, b]
        scenes.append(sc)
        cursor = b + 1
    if not scenes:
        raise ValueError("the writer returned no usable scenes")
    if cursor < n:
        scenes[-1]["sections"] = [scenes[-1]["sections"][0], n - 1]
    for i, sc in enumerate(scenes):
        a, b = sc["sections"]
        sc["index"] = i
        sc["start"] = sections[a]["start"]
        sc["end"] = sections[b]["end"]
        sc["section_types"] = [sections[k]["type"] for k in range(a, b + 1)]
        sc["lipsync"] = str(sc.get("performance", "")).startswith("sing")
    return scenes


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
- Exactly ONE camera motion for the scene, as given, with its real-world distance and duration anchor repeated in every clip; or "the camera is static" if static.
- One main physical motion for her per clip, with a duration anchor (how long it takes, in seconds). Hands: say what they do or that they stay still. Hair: say whether it moves. If sky/clouds/stars are visible: say they hold fixed positions with no drift and no cloud movement.
- Clip 2 and later CONTINUE: the first lines say that the incoming motion, camera, framing, lighting and state continue uninterrupted from the protected video context, then say what that ongoing motion becomes. Never restart an action already in progress, never re-establish the shot, never return to the reference pose.
- Never describe other people. Never change the wardrobe.
- Lip-sync clips: mouth readable (face toward camera, medium close-up or closer, nothing covering the mouth).
- 150-320 words of detailed_description per clip; concrete, visual, in order of time."""


def clip_prompt(script, scene, clips):
    user = {
        "song": script["title"], "look": script["look"], "wardrobe": script["wardrobe"],
        "concept": script["concept"],
        "scene": {k: scene.get(k) for k in ("title", "setting", "performance", "framing", "camera",
                                            "lighting", "mood", "action", "section_types", "start", "end")},
        "clips": [{"index": c["index"], "seconds": c["seconds"], "song_start": c["song_start"],
                   "song_end": c["song_end"], "continues_previous": c["index"] > 0,
                   "protected_prefix_seconds": round(CHAIN_CONTEXT / H3_FPS, 3) if c["index"] > 0 else 0,
                   "lines_audible": c["lines"]} for c in clips],
        "return": {"clips": [{"index": 0, "summary": "...", "detailed_description": "[Shot 1] ...",
                              "overall_soundscape": "...", "non_diegetic_music": "..."}]},
    }
    return CLIP_SYSTEM, json.dumps(user, indent=1)


def compile_prompt(script, scene, clip, text):
    """The six-section Ref2VA prompt for one clip: fixed boilerplate from the script's references
    and wardrobe + Claude's four fields. The style opening is the script's look sentence."""
    refs = script["refs"]
    subject = ("subject_definitions:\n<Subject 1> is the singer: her facial identity comes from <Picture 1>; "
               "her full-body appearance, body proportions and wardrobe come from <Picture 2>. "
               f"{refs['shared']} Wardrobe: {script['wardrobe']}\n")
    retention = ("retention_analysis:\n<Subject 1> (appears throughout [Shot 1]): fully_preserved - retain the "
                 "referenced face, eyes, hair, skin, body proportions and the wardrobe exactly, while pose, "
                 "expression, framing, lighting and movement follow the performance described below.\n")
    if scene.get("performance") == "no_singing_environment":
        retention = ("retention_analysis:\n<Subject 1> (not on screen in [Shot 1]): weak_reference - the singer "
                     "does not appear in this clip; only the video's style and palette carry over.\n")
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


def check_prompt(prompt, clip):
    """The hard rules, mechanically. Returns a list of problems (empty = clean)."""
    problems = []
    if "<Audio" in prompt:
        problems.append("names <Audio N> (the song is locked in the latent)")
    if "[Shot 2]" in prompt or "<scenetrans>" in prompt:
        problems.append("more than one shot in a chain clip")
    if clip["vocal"] and "<d>[English]" not in prompt:
        problems.append("vocal clip without a <d>[English] ...</d> line")
    if not clip["vocal"] and "<d>" in prompt:
        problems.append("instrumental clip carries a <d> line")
    for sec in ("subject_definitions:", "summary:", "retention_analysis:", "detailed_description:",
                "overall_soundscape:", "non_diegetic_music:"):
        if sec not in prompt:
            problems.append(f"missing section {sec}")
    return problems


# ----------------------------------------------------------------------------- the whole thing

def write_script(audio_path, audio_id, title, style, lyrics_raw, duration, refs, concept=None,
                 provider="claude_sub", model=WRITER_MODEL, effort=WRITER_EFFORT, claude_model="",
                 cache_dir=None, log=print):
    """Song -> full script JSON (scenes with chain plans and compiled per-clip prompts).
    `refs` = {"face": one-line description of <Picture 1>, "body": of <Picture 2>,
              "shared": a sentence naming what BOTH show (the details the prompt may mention)}."""
    facts = song_facts(audio_path, lyrics_raw, duration, cache_dir=cache_dir)
    log(f"[mvw] facts: {len(facts['sections'])} sections, cover {facts.get('cover')}")
    refs_text = (f"<Picture 1> (face / identity): {refs['face']}\n<Picture 2> (full body / wardrobe): {refs['body']}\n"
                 f"Both show: {refs['shared']}")
    system, user = scene_prompt(title, style, lyrics_raw, facts, refs_text, concept)
    raw = llm_mod.complete(provider, model, system, user, claude_model, timeout=WRITER_TIMEOUT, effort=effort)
    _keep_raw(cache_dir, f"{audio_id}_pass1.txt", raw)
    plan = _extract_json(raw)
    scenes = normalize_scenes(plan, facts["sections"])
    log(f"[mvw] pass 1: {len(scenes)} scenes")
    script = {"title": title, "audio_id": audio_id, "style": style, "duration": duration,
              "concept": plan.get("concept", ""), "look": plan.get("look", ""),
              "wardrobe": plan.get("wardrobe", ""), "refs": refs, "facts": facts,
              "writer": {"provider": provider, "model": model, "effort": effort}, "scenes": []}
    for sc in scenes:
        cplan, clips = plan_scene_clips(sc, facts["sections"])
        system2, user2 = clip_prompt(script, sc, clips)
        raw2 = llm_mod.complete(provider, model, system2, user2, claude_model, timeout=WRITER_TIMEOUT, effort=effort)
        _keep_raw(cache_dir, f"{audio_id}_pass2_scene{sc['index']}.txt", raw2)
        texts = {int(t.get("index", i)): t for i, t in enumerate((_extract_json(raw2) or {}).get("clips") or [])}
        for c in clips:
            t = texts.get(c["index"]) or {}
            c["text"] = {k: t.get(k, "") for k in ("summary", "detailed_description", "overall_soundscape", "non_diegetic_music")}
            c["prompt"] = compile_prompt(script, sc, c, c["text"])
            c["problems"] = check_prompt(c["prompt"], c)
        sc["plan"] = cplan
        sc["clips"] = clips
        script["scenes"].append(sc)
        log(f"[mvw] pass 2: scene {sc['index']} '{sc.get('title')}' {sc['start']}-{sc['end']}s -> {len(clips)} clips, "
            f"problems {sum(len(c['problems']) for c in clips)}")
    script["clip_count"] = sum(len(s["clips"]) for s in script["scenes"])
    return script


def script_markdown(script):
    """A readable rendering of a script for review (the user reads this, not the JSON)."""
    out = [f"# {script['title']} - music video script", "",
           f"Writer: {script['writer']['model']} ({script['writer']['effort']} effort). "
           f"Song {script['duration']:.1f}s, lyric cover {script['facts'].get('cover')}. "
           f"{len(script['scenes'])} scenes, {script['clip_count']} clips.", "",
           f"**Concept.** {script['concept']}", "", f"**Look.** {script['look']}", "",
           f"**Wardrobe.** {script['wardrobe']}", ""]
    for sc in script["scenes"]:
        out += [f"## Scene {sc['index'] + 1}: {sc.get('title', '')}  ({sc['start']:.1f}-{sc['end']:.1f}s, "
                f"{', '.join(sc['section_types'])})", "",
                f"- performance: {sc.get('performance')} | framing: {sc.get('framing')} | camera: {sc.get('camera')}",
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
