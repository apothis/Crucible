"""YuE2-3B: prompt assembly, ABC score helpers, and the ComfyUI graphs.

A THIRD generation engine, alongside ACE-Step and MiniMax Music 3, with its own tab. Nothing here
is shared with the other two on purpose: each engine wants a different prompt.

  ACE-Step   ~10-12 dense tag phrases.
  Music 3    a 4000+ character structured caption in THEIR field schema.
  YuE2       one COMPACT style line (language, one genre, instruments, vocal character, tempo,
             mood) plus lyrics with bare Title-case section tags. Upstream's own guidance: "more
             adjectives do not automatically create more control".

What makes YuE2 different is the plan: before rendering audio it writes an editable ABC score
(melody, and with mode "full" the chords too), then sings and plays that score. The score is a
first-class artifact here: it is kept with every take, can be generated on its own (fast, no
acoustic stage), edited, and rendered again; and a real recording can be transcribed into one
(SheetSage2) and re-rendered in a new style = the cover mode.

Everything about the graph is lifted from the reference workflow attached to the ComfyUI core
PR that added the nodes (Comfy-Org/ComfyUI#16250, merged 2026-09-11), not invented here. Node
input names and defaults were read from comfy_extras/nodes_yue2.py at that commit.

Measured on our 3090 (2026-09-15): 224 s wall for 265 s of audio (first run, model load
included); 48 kHz stereo FLAC at about -12.7 LUFS with peaks at 0 dBFS. Two seeds of the same
prompt both planned D# minor at 85-86 bpm against a style line asking for A major at 84, so the
key/tempo words in the style are NOT a reliable lever; the score header is (see set_header).
"""
import re

MODELS = {
    "ckpt": "yue2_3b_bf16.safetensors",          # models/checkpoints  (MODEL + CLIP + VAE in one)
    "sheetsage": "sheetsage2_bf16.safetensors",  # models/audio_encoders, cover-mode transcriber
}

# 25 semantic tokens per second; the node allows up to 900 s but the 24576-token context has to
# hold style + lyrics + score + music, so 360 s is the honest ceiling (a 4-minute song is 6000).
MAX_SECONDS = 360.0
FRAMES_PER_SECOND = 25

# Reference-workflow values. Two samplers: the ABC planner and the semantic (music-token) stage
# are both AR decoders inside the "CLIP"; the KSampler is the acoustic flow-matching stage.
DEFAULTS = {
    "mode": "full",                # off | melody | full  (plan mode; "off" = no score at all)
    "max_seconds": 300.0,          # ceiling; the model ends the song itself, cap != target
    # acoustic stage
    "steps": 32, "cfg": 1.0, "sampler": "dpm_2", "scheduler": "sgm_uniform", "tiled_decode": False,
    # ABC planner sampling
    "abc_temperature": 0.7, "abc_top_p": 0.9, "abc_top_k": 30, "abc_repetition_penalty": 1.005,
    "abc_penalty_window": 100, "max_abc_tokens": 8192,
    # semantic-stage sampling
    "temperature": 1.0, "top_p": 0.95, "top_k": 100, "repetition_penalty": 1.2,
}

# Bare, Title-case: upstream's own examples use [Verse] [Pre-Chorus] [Chorus] [Bridge] [Interlude]
# [Outro] [Intro]; numbered [Verse 2] and [Guitar Solo] are common in the demo-site prompts. There
# is no tag parser (the lyrics are raw text to the model), so the name is what matters and
# anything else inside the brackets is sung. An empty body after every tag = instrumental.
SECTION_TAGS = ["Intro", "Verse", "Pre-Chorus", "Chorus", "Bridge", "Guitar Solo", "Interlude", "Outro"]

# Style-line guidance, served to the UI so it has one home. Order matters less than compactness.
STYLE_GUIDE = [
    ("language", "English"),
    ("genre", "ONE genre, e.g. symphonic metal / modern hard rock power ballad"),
    ("instruments", "the ones that define the sound, and where: 'distorted rhythm guitars panned wide, double-kick drums, orchestra behind the guitars'"),
    ("vocal", "gender + register + timbre + delivery: 'female vocal, operatic soprano, bel canto vibrato, soaring in the choruses'"),
    ("tempo", "'140 BPM': a soft prior, honoured about half the time on the demo set (half/double time drift); Q: in the score header is the hard lever"),
    ("key", "not controllable from text (no demo case honours one); K: in the score is. Soft prompts without one tend to plan D# minor at 85"),
    ("mood", "two or three words: 'romantic and haunting'"),
    ("negatives", "short and absolute, upstream uses them itself: 'no guitar', 'no growls'"),
    ("instrumental", "omit language and singer from the style, add 'instrumental', and leave every section tag's body empty (works about half the time upstream)"),
]


# ---------------- prompt assembly from a Song tab arrangement ----------------

_TAG_MAP = {"pre-chorus": "Pre-Chorus", "prechorus": "Pre-Chorus", "post-chorus": "Chorus",
            "postchorus": "Chorus", "solo": "Guitar Solo", "guitar solo": "Guitar Solo",
            "outro": "Outro", "intro": "Intro", "bridge": "Bridge", "chorus": "Chorus",
            "verse": "Verse", "instrumental": "Interlude", "interlude": "Interlude",
            "break": "Interlude", "breakdown": "Interlude"}


def _tag_name(raw):
    """'chorus - anthemic' / 'VERSE 2' -> 'Chorus' / 'Verse 2': the name only, Title-cased,
    a trailing number kept (numbered verses are common upstream)."""
    # split on a comma, a colon, or a SPACED hyphen: a bare hyphen is part of "Pre-Chorus"
    name = re.split(r"[,:]|\s-\s", raw or "")[0].strip()
    m = re.match(r"^(.*?)\s*(\d+)?$", name)
    base, num = (m.group(1).strip().lower() if m else name.lower()), (m.group(2) if m else None)
    tag = _TAG_MAP.get(base) or " ".join(w.capitalize() for w in base.split()) or "Verse"
    return f"{tag} {num}" if num else tag


def song_to_lyrics(song):
    """SongDraft blocks -> lyrics with bare section tags, one blank line between."""
    out = []
    for b in (song or {}).get("blocks") or []:
        tag = _tag_name(b.get("type") or "Verse")
        lyr = (b.get("lyrics") or "").strip()
        out.append(f"[{tag}]" + (f"\n{lyr}" if lyr else ""))
    return "\n\n".join(out)


def song_to_style(song):
    """SongDraft tags + key/bpm -> one compact style line. The ACE tag list is already a comma
    list of descriptors, so it is kept as-is (trimmed, deduplicated); key and bpm are added at
    the end because that is where upstream's example puts tempo."""
    song = song or {}
    tags = [t.strip() for t in (song.get("tags") or "").split(",") if t.strip()]
    seen, parts = set(), []
    for t in tags:
        k = t.lower()
        if k not in seen:
            seen.add(k)
            parts.append(t)
    if not any(p.lower().startswith("english") for p in parts):
        parts.insert(0, "English")
    bpm = song.get("bpm")
    if bpm:
        parts.append(f"{int(bpm)} BPM")
    key = (song.get("key") or "").strip()
    if key:
        parts.append(key)
    return ", ".join(parts)


def lyrics_to_sections(lyrics):
    """Bare-tag lyrics -> ordered [{tag, lyrics}], text verbatim. Anything after a '-' or ',' in
    a tag is dropped (an ACE-style loaded tag pasted in) and the name is Title-cased."""
    out = []
    parts = re.split(r"\[([^\]\n]{1,40})\]", lyrics or "")
    pre = (parts[0] or "").strip()
    if pre:
        out.append({"tag": "Verse", "lyrics": pre})
    for i in range(1, len(parts), 2):
        body = (parts[i + 1] if i + 1 < len(parts) else "").strip()
        out.append({"tag": _tag_name(parts[i]), "lyrics": body})
    return out


def normalize_lyrics(lyrics):
    """Strip anything loaded into the section tags and Title-case them; lyric lines untouched."""
    return "\n\n".join(f"[{s['tag']}]" + (f"\n{s['lyrics']}" if s["lyrics"] else "")
                       for s in lyrics_to_sections(lyrics))


def lint(style, lyrics, max_seconds):
    """Cheap pre-flight checks surfaced in the UI; none of them block a render."""
    notes = []
    tags = re.findall(r"\[([^\]\n]+)\]", lyrics or "")
    loaded = [t for t in tags if re.search(r"[,:]|\s-\s", t)]
    if loaded:
        notes.append("loaded tags (only the section name is safe inside brackets): " + ", ".join(loaded))
    words = len(re.findall(r"\S+", re.sub(r"\[[^\]]*\]", "", lyrics or "")))
    # ~2.3 words/second of singing is a slow-ballad rate; above that the song runs past the cap
    if max_seconds and words > max_seconds * 2.3:
        notes.append(f"{words} lyric words is a lot for {int(max_seconds)} s; the cap is a hard "
                     "mid-phrase cut, so raise it or trim")
    if not (style or "").strip():
        notes.append("empty style line")
    elif len(style) > 600:
        notes.append(f"style line is {len(style)} chars; upstream advises compact prompts")
    if re.search(r"\([^)]*(guitar|drum|solo|riff|choir|enters?|fade|swell|piano|strings)[^)]*\)", lyrics or "", re.I):
        notes.append("parenthesised stage directions in the lyrics may be sung")
    return notes


# ---------------- ABC score helpers ----------------
# The planner emits: X:1 / T: / M:4/4 / L:1/32 / Q:1/4=NN / two V: declarations (Vocal, Ins) /
# K:<key> / then alternating "V: Vocal" and "V: Ins" bar groups with "% section" comments and
# chord symbols in quotes. These helpers only touch the HEADER and read structure; note-level
# edits (transposition, melody changes) are done in the UI editor on the text itself.

_HEADER_KEYS = ("X", "T", "M", "L", "Q", "K")


def parse_header(abc):
    """-> {X, T, M, L, Q, K, bpm, voices:[...], sections:[{name, line}]}"""
    h = {"voices": [], "sections": []}
    for i, line in enumerate((abc or "").splitlines()):
        s = line.strip()
        m = re.match(r"^([A-Z]):\s*(.*)$", s)
        if m and m.group(1) in _HEADER_KEYS and m.group(1) not in h:
            h[m.group(1)] = m.group(2).strip()
        elif m and m.group(1) == "V":
            vm = re.match(r'^(\S+)(?:.*?name="([^"]*)")?', m.group(2))
            if vm and vm.group(1) not in [v["id"] for v in h["voices"]]:
                h["voices"].append({"id": vm.group(1), "name": vm.group(2) or vm.group(1)})
        elif s.startswith("%"):
            h["sections"].append({"name": s.lstrip("% ").strip(), "line": i})
    qm = re.match(r"^(\d+)/(\d+)\s*=\s*(\d+)", h.get("Q", ""))
    h["bpm"] = int(qm.group(3)) if qm else None
    h["bars"] = sum(line.count("|") for line in (abc or "").splitlines()
                    if line.strip() and not re.match(r"^[A-Za-z]:", line.strip()) and not line.strip().startswith("%"))
    return h


def set_header(abc, key=None, bpm=None, title=None):
    """Rewrite K: / Q: / T: in place. Changing K: alone does NOT transpose the notes - it
    re-spells them - so key changes that should move the pitch go through the editor's
    transpose (note-level). Q: is the honoured tempo lever: the planner ignored the style's BPM
    on both seeds we measured, and Q: is what the semantic stage reads."""
    out = []
    for line in (abc or "").splitlines():
        s = line.strip()
        if key and re.match(r"^K:", s):
            line = f"K:{key}"
        elif bpm and re.match(r"^Q:", s):
            unit = re.match(r"^Q:\s*(\d+/\d+)", s)
            line = f"Q:{unit.group(1) if unit else '1/4'}={int(bpm)}"
        elif title is not None and re.match(r"^T:", s):
            line = f"T:{title}"
        out.append(line)
    return "\n".join(out) + ("\n" if (abc or "").endswith("\n") else "")


def validate_abc(abc):
    """Structural sanity for an edited score before it is sent: header present, a K: line, at
    least one voice body. Returns a list of problems (empty = fine)."""
    problems = []
    if not (abc or "").strip():
        return ["empty score"]
    h = parse_header(abc)
    for k in ("M", "L", "K"):
        if k not in h:
            problems.append(f"missing {k}: header line")
    if "|" not in (abc or ""):
        problems.append("no bar lines found")
    if "Vocal" not in (abc or "") and not h["voices"]:
        problems.append("no voice declared (expected V: Vocal / V: Ins)")
    return problems


# ---------------- graphs ----------------

def _f(p, k):
    v = p.get(k)
    return DEFAULTS[k] if v in (None, "") else v


def _music_node(p, abc_input, mode):
    node = {"class_type": "YuE2GenerateMusic", "inputs": {
        "clip": ["1", 1], "style": p["style"], "lyrics": p["lyrics"], "abc": abc_input,
        "seed": int(p["seed"]), "mode": "melody" if mode == "melody" else "full",
        "max_duration": float(p["max_seconds"]),
        "temperature": float(_f(p, "temperature")), "top_p": float(_f(p, "top_p")),
        "top_k": int(_f(p, "top_k")), "repetition_penalty": float(_f(p, "repetition_penalty"))}}
    # semantic-stage CFG (upstream: 1.0 = off in full/melody, 1.01 in off mode; 1.2 = "stronger
    # text guidance", the negative branch keeps the score and drops style + lyrics). Only sent
    # when the user sets it, so the node keeps its own per-mode default otherwise.
    if p.get("cfg_scale") is not None:
        node["inputs"]["cfg_scale"] = float(p["cfg_scale"])
    return node


def _abc_node(p, mode, seed):
    return {"class_type": "YuE2GenerateABC", "inputs": {
        "clip": ["1", 1], "style": p["style"], "lyrics": p["lyrics"], "seed": int(seed),
        "mode": "melody" if mode == "melody" else "full",
        "max_abc_tokens": int(_f(p, "max_abc_tokens")),
        "temperature": float(_f(p, "abc_temperature")), "top_p": float(_f(p, "abc_top_p")),
        "top_k": int(_f(p, "abc_top_k")),
        "repetition_penalty": float(_f(p, "abc_repetition_penalty")),
        "penalty_window": int(_f(p, "abc_penalty_window"))}}


def _resolve(p):
    style = (p.get("style") or "").strip()
    lyrics = normalize_lyrics(p.get("lyrics") or "")
    if not style:
        raise ValueError("a style line is required")
    mode = (p.get("mode") or DEFAULTS["mode"]).strip().lower()
    if mode not in ("off", "melody", "full"):
        raise ValueError(f"unknown plan mode {mode!r} (off, melody or full)")
    abc = (p.get("abc") or "").strip()
    if abc and mode == "off":
        mode = "full"     # a supplied score implies planning; the node itself ignores mode when abc is empty
    r = {
        "style": style, "lyrics": lyrics, "mode": mode, "abc": abc,
        "seed": int(p.get("seed") or 0),
        "max_seconds": max(1.0, min(float(p.get("max_seconds") or DEFAULTS["max_seconds"]), MAX_SECONDS)),
        "steps": int(_f(p, "steps")), "cfg": float(_f(p, "cfg")),
        "sampler": p.get("sampler") or DEFAULTS["sampler"],
        "scheduler": p.get("scheduler") or DEFAULTS["scheduler"],
        "tiled_decode": bool(p.get("tiled_decode", DEFAULTS["tiled_decode"])),
        # the plan seed defaults to the render seed; splitting them lets one score be rendered
        # with several performances, or several scores be tried on one render seed
        "abc_seed": int(p.get("abc_seed") or 0) or int(p.get("seed") or 0),
        "cfg_scale": (float(p["cfg_scale"]) if p.get("cfg_scale") not in (None, "") else None),
    }
    for k in ("temperature", "top_p", "top_k", "repetition_penalty", "abc_temperature",
              "abc_top_p", "abc_top_k", "abc_repetition_penalty", "abc_penalty_window",
              "max_abc_tokens"):
        r[k] = _f(p, k)
    if abc:
        probs = validate_abc(abc)
        if probs:
            raise ValueError("score problems: " + "; ".join(probs))
    return r


def build_graph(p):
    """Full render: (optional plan ->) music tokens -> acoustic -> FLAC. Returns (graph, resolved).
    When the score is generated in-graph, a PreviewAny node puts it in the job history as text
    so the completion worker can keep it with the take."""
    r = _resolve(p)
    g = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": MODELS["ckpt"]}}}
    if r["mode"] == "off":
        abc_input = ""
    elif r["abc"]:
        abc_input = r["abc"]
    else:
        g["2"] = _abc_node(r, r["mode"], r["abc_seed"])
        g["10"] = {"class_type": "PreviewAny", "inputs": {"source": ["2", 0]}}
        abc_input = ["2", 0]
    g["3"] = _music_node(r, abc_input, r["mode"])
    g["4"] = {"class_type": "EmptyYuE2LatentAudio", "inputs": {"seconds": ["3", 1], "batch_size": 1}}
    # positive == negative is how the reference workflow wires it (cfg 1.0 makes it moot)
    g["5"] = {"class_type": "KSampler", "inputs": {
        "model": ["1", 0], "positive": ["3", 0], "negative": ["3", 0], "latent_image": ["4", 0],
        "seed": r["seed"], "steps": r["steps"], "cfg": r["cfg"],
        "sampler_name": r["sampler"], "scheduler": r["scheduler"], "denoise": 1.0}}
    g["6"] = ({"class_type": "VAEDecodeAudioTiled",
               "inputs": {"samples": ["5", 0], "vae": ["1", 2], "tile_size": 1920, "overlap": 128}}
              if r["tiled_decode"] else
              {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}})
    g["7"] = {"class_type": "SaveAudio", "inputs": {"audio": ["6", 0], "filename_prefix": "audio/yue2"}}
    return g, r


def build_plan_graph(p):
    """Score only: the AR planner, no music tokens, no acoustic stage. Seconds of GPU rather than
    minutes, which is what makes a plan batch (several seeds, pick one) practical."""
    r = _resolve({**p, "mode": (p.get("mode") or "full"), "abc": ""})
    if r["mode"] == "off":
        raise ValueError("plan mode off produces no score; choose melody or full")
    g = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": MODELS["ckpt"]}},
         "2": _abc_node(r, r["mode"], r["abc_seed"]),
         "10": {"class_type": "PreviewAny", "inputs": {"source": ["2", 0]}}}
    return g, r


def build_transcribe_graph(audio_name, mode="melody"):
    """Cover mode step 1: a real recording (already uploaded to ComfyUI's input dir) -> ABC via
    SheetSage2. melody = melody only (recommended for style changes: the accompaniment is free);
    full = melody + chord symbols (keeps the harmony)."""
    mode = "full" if (mode or "").lower() == "full" else "melody"
    g = {"1": {"class_type": "AudioEncoderLoader", "inputs": {"audio_encoder_name": MODELS["sheetsage"]}},
         "2": {"class_type": "LoadAudio", "inputs": {"audio": audio_name}},
         "3": {"class_type": "SheetSage2AudioToABC", "inputs": {"audio_encoder": ["1", 0], "audio": ["2", 0], "mode": mode}},
         "10": {"class_type": "PreviewAny", "inputs": {"source": ["3", 0]}}}
    return g, {"audio": audio_name, "mode": mode}


def text_outputs(history_entry):
    """The PreviewAny text(s) from a finished job's history entry, joined. ComfyUI reports the
    node's ui output as {"text": [..]}; the score is the first string."""
    texts = []
    for _nid, out in (history_entry.get("outputs") or {}).items():
        for t in out.get("text", []) or []:
            texts.append(t if isinstance(t, str) else str(t))
    return "\n".join(texts).strip()


# ---------------- LLM helpers: the style writer and the agentic score editor ----------------
# Both go through backend/llm.py like every other writer in the app (claude_sub by default) and
# return PROPOSALS: the tab shows before/after and the user accepts, nothing is applied blind.

_STYLE_SYSTEM = (
    "You write the style line for the YuE2 music model. Output STRICT JSON ONLY: "
    "{\"style\": \"...\", \"lyrics\": \"...\"}. No prose, no markdown.")

_STYLE_RULES = """Rules for "style" (upstream's guidance: "more adjectives do not automatically create more control"):
- ONE compact comma-separated line, 8-14 descriptors, under 60 words, strongest identity first:
  language, ONE genre, the defining instruments and where they sit in the mix, the vocal (gender,
  register, timbre, delivery), tempo as "NNN BPM" when the brief implies one, a two-word mood, and
  short absolute negatives ("no growls", "no synthesizers") - upstream uses "no guitar" itself.
- Upstream's own examples, for register: "English, warm piano pop, expressive female voice,
  acoustic piano, rounded bass and light drums, lyrical memorable melody, unhurried phrasing, 88 BPM"
  and "Heavy metal rock, palm-muted riffs, distorted guitars, driving bass, double-kick drums,
  powerful English vocal, 160 BPM".
- Describe audible qualities; never name a living artist. Nothing about song sections or timing.
- For an instrumental: omit language and singer, add "instrumental".
- If existing style text is given, IMPROVE it in place (keep its genre, instruments and voice
  unless the brief says otherwise) rather than starting over.
Rules for "lyrics": ONLY when no lyrics are provided, draft original lyrics structured with bare
Title-case section tags on their own lines ([Intro] [Verse] [Pre-Chorus] [Chorus] [Bridge]
[Guitar Solo] [Outro]; [Verse 2] style numbering is fine), nothing else inside the brackets,
verses 4-8 lines, a chantable chorus, plain ASCII, no parentheses. If lyrics ARE provided return
"" for lyrics."""


def write_style(brief, style, lyrics, provider, model, claude_model):
    from . import llm
    prompt = (f"{_STYLE_RULES}\n\nBRIEF: {(brief or '').strip() or '(none - improve the existing style line)'}\n\n"
              f"EXISTING STYLE LINE: {(style or '').strip() or '(empty)'}\n\n"
              f"LYRICS: {(lyrics or '').strip()[:4000] or '(none)'}\n\nReturn the JSON now.")
    text = llm.complete(provider, model, _STYLE_SYSTEM, prompt, claude_model, timeout=240)
    m = re.search(r"\{.*\}", text or "", re.S)
    import json as _json
    data = _json.loads(m.group(0) if m else text)
    out_style = str(data.get("style") or "").strip()
    out_lyrics = str(data.get("lyrics") or "").strip() if not (lyrics or "").strip() else ""
    if not out_style:
        raise ValueError("the writer returned no style line")
    return {"style": out_style[:800], "lyrics": normalize_lyrics(out_lyrics) if out_lyrics else "",
            "provider": provider}


_EDIT_SYSTEM = (
    "You are a music editor working on an ABC score for the YuE2 music model. Output STRICT JSON "
    "ONLY: {\"abc\": \"...\", \"summary\": \"...\"}. The abc value is the COMPLETE edited score "
    "as one JSON string with \\n newlines. No prose outside the JSON, no markdown fences.")

_EDIT_RULES = """The YuE2 ABC dialect (anything outside it is rejected by the renderer):
- Header lines, in order: X:1, T:, M:, L: (keep the given L, usually 1/32 or 1/16), Q:1/4=NN,
  the two V: declarations exactly as given, K:<key>.
- Body: bar groups alternate "V: Vocal" and "V: Ins" blocks of 1-4 bars each, bars end with |.
  Sections are marked by a comment line "% verse" / "% chorus" / "% pre-chorus" / "% bridge" /
  "% intro" / "% interlude" / "% outro" placed before the first block of the section.
- Chord symbols are quoted, in the Vocal voice, at the start of the beat they change on: "Am7".
- Note durations are multiples of L: with L:1/32 use only 1,2,3,4,6,8,12,16,24,32,48. Rests are
  z<dur>; a whole silent bar is z32 (L:1/32) or z16 (L:1/16); Z, Z2, Z4 = 1/2/4 silent bars.
- Ties with -, accidentals ^ _ = before the note letter, octaves with , and '.
- NOT allowed: tuplets, grace notes, chord stacks [CE], repeats |: :|, slurs (), w: lyric lines,
  decorations, inline field changes.
- Keep every bar the right length for M:. Keep the section order consistent with the lyric
  sheet's sections unless the instruction changes the form, in which case say so in the summary
  so the lyrics can be changed to match.
- Make ONLY the change asked for; leave everything else byte-identical. Say in "summary" (one or
  two sentences) exactly what changed, in musical terms (bars, section, interval)."""


def edit_score(abc, style, lyrics, instruction, provider, model, claude_model):
    from . import llm
    import json as _json
    if not (abc or "").strip():
        raise ValueError("no score to edit")
    if not (instruction or "").strip():
        raise ValueError("say what to change")
    prompt = (f"{_EDIT_RULES}\n\nSTYLE LINE: {(style or '').strip()}\n\nLYRIC SHEET (sections):\n"
              f"{(lyrics or '').strip()[:3000]}\n\nTHE SCORE:\n{abc.strip()}\n\n"
              f"INSTRUCTION: {instruction.strip()}\n\nReturn the JSON now.")
    text = llm.complete(provider, model, _EDIT_SYSTEM, prompt, claude_model, timeout=300)
    m = re.search(r"\{.*\}", text or "", re.S)
    data = _json.loads(m.group(0) if m else text)
    out = str(data.get("abc") or "").strip()
    if not out:
        raise ValueError("the editor returned no score")
    if not out.endswith("\n"):
        out += "\n"
    before, after = parse_header(abc), parse_header(out)
    return {"abc": out, "summary": str(data.get("summary") or "").strip(),
            "problems": validate_abc(out),
            "before": {"K": before.get("K"), "bpm": before.get("bpm"), "bars": before["bars"],
                       "sections": [s["name"] for s in before["sections"]]},
            "after": {"K": after.get("K"), "bpm": after.get("bpm"), "bars": after["bars"],
                      "sections": [s["name"] for s in after["sections"]]},
            "provider": provider}


# ---------------- sectioning fetched / pasted lyrics ----------------
# Online lyric sources come untagged (LRCLIB/lyrics.ovh give plain stanzas, sometimes one block).
# YuE2 paces the song from the section tags, so untagged words would be sung as one long verse.
# The score's own "% section" markers (from a SheetSage2 transcription in cover mode, or a plan)
# are the best guide to the form, so they steer the tagging when a score is loaded.

_SECTION_SYSTEM = (
    "You split song lyrics into tagged sections for a music model. Output STRICT JSON ONLY: "
    "{\"sections\": [{\"tag\": \"Verse\", \"first\": 1, \"last\": 4}, ...]}. No prose, no markdown.")

_SECTION_RULES = """Rules:
- The lyric lines are numbered. Return sections in order; each covers a contiguous range of line
  numbers (first..last inclusive). Every line must be in exactly one section; no gaps, no overlaps,
  never reorder or rewrite lines.
- Tags come from: Intro, Verse, Verse 2, Verse 3, Pre-Chorus, Chorus, Bridge, Guitar Solo, Interlude,
  Outro. Number verses in order. A stanza that repeats (same or nearly the same words) is the Chorus.
  A short lifting passage right before a chorus is the Pre-Chorus. A one-off passage after the
  second chorus is the Bridge.
- Sections with no lyrics (Intro, Guitar Solo, Interlude, Outro) may be included with "first": 0
  and "last": 0 where the song's form has them - use the SCORE SECTIONS below, which come from the
  actual recording or plan, as the form to follow when they are given. When they are given, match
  their order as closely as the words allow (one sung section per score section, in order)."""


def _stanzas(text):
    lines = [l.rstrip() for l in (text or "").replace("\r", "").split("\n")]
    # LRC timestamps and empty decorations
    lines = [re.sub(r"^\[\d{1,2}:\d{2}(?:\.\d{1,3})?\]\s*", "", l) for l in lines]
    out, cur = [], []
    for l in lines:
        if l.strip():
            cur.append(l.strip())
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def _fallback_sections(stanzas):
    """No LLM: repeated stanzas are choruses, the rest verses in order, a unique stanza after the
    second chorus is the bridge. One undivided block is cut into 4-line verses."""
    if len(stanzas) == 1 and len(stanzas[0]) > 6:
        block = stanzas[0]
        stanzas = [block[i:i + 4] for i in range(0, len(block), 4)]
    # near-duplicates count as repeats: choruses often vary a word, so key on the first line
    norm = [re.sub(r"[^a-z ]", "", s[0].lower()).strip() for s in stanzas]
    counts = {n: norm.count(n) for n in norm}
    out, verse_n, chorus_seen, bridged = [], 0, 0, False
    for s, n in zip(stanzas, norm):
        if counts[n] > 1:
            chorus_seen += 1
            out.append(("Chorus", s))
        elif chorus_seen >= 2 and not bridged and len(s) <= 6:
            bridged = True
            out.append(("Bridge", s))
        else:
            verse_n += 1
            out.append((f"Verse {verse_n}" if verse_n > 1 else "Verse", s))
    return out


def section_lyrics(text, abc="", provider="", model="", claude_model=""):
    """Plain (untagged) lyrics -> lyrics with bare section tags. Returns (lyrics, how)."""
    raw = (text or "").strip()
    if not raw:
        return "", "empty"
    if len(re.findall(r"^\s*\[[^\]\n]{1,40}\]\s*$", raw, re.M)) >= 2:
        return normalize_lyrics(raw), "source tags"
    stanzas = _stanzas(raw)
    lines = [l for s in stanzas for l in s]
    score_sections = [s["name"] for s in parse_header(abc)["sections"]] if (abc or "").strip() else []
    if provider:
        try:
            from . import llm
            import json as _json
            numbered = []
            n = 0
            for s in stanzas:
                for l in s:
                    n += 1
                    numbered.append(f"{n}. {l}")
                numbered.append("")
            prompt = (f"{_SECTION_RULES}\n\nSCORE SECTIONS (in order): "
                      f"{', '.join(score_sections) if score_sections else '(none given)'}\n\n"
                      f"LYRIC LINES:\n" + "\n".join(numbered) + "\n\nReturn the JSON now.")
            txt = llm.complete(provider, model, _SECTION_SYSTEM, prompt, claude_model, timeout=240)
            m = re.search(r"\{.*\}", txt or "", re.S)
            data = _json.loads(m.group(0) if m else txt)
            secs = data.get("sections") or []
            covered, out, expect = [], [], 1
            for s in secs:
                tag = _tag_name(str(s.get("tag") or "Verse"))
                a, b = int(s.get("first") or 0), int(s.get("last") or 0)
                if a == 0 and b == 0:
                    out.append(f"[{tag}]")
                    continue
                if b < a or b > len(lines):
                    raise ValueError(f"bad line range {a}-{b}")
                if a == expect:
                    expect = b + 1
                elif b < expect:
                    pass        # a REPEAT of lines already placed (a chorus the source wrote once)
                else:
                    raise ValueError(f"bad line range {a}-{b} (expected to start at {expect})")
                out.append(f"[{tag}]\n" + "\n".join(lines[a - 1:b]))
                covered.extend(range(a, b + 1))
            if expect != len(lines) + 1:
                raise ValueError("not every line was placed")
            return "\n\n".join(out), f"llm ({provider})"
        except Exception as e:
            how = f"fallback ({e})"
    else:
        how = "fallback"
    parts = _fallback_sections(stanzas)
    out = []
    if "intro" in score_sections:
        out.append("[Intro]")
    out += [f"[{tag}]\n" + "\n".join(s) for tag, s in parts]
    if "outro" in score_sections:
        out.append("[Outro]")
    return "\n\n".join(out), how
