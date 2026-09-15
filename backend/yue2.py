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
    return {"class_type": "YuE2GenerateMusic", "inputs": {
        "clip": ["1", 1], "style": p["style"], "lyrics": p["lyrics"], "abc": abc_input,
        "seed": int(p["seed"]), "mode": "melody" if mode == "melody" else "full",
        "max_duration": float(p["max_seconds"]),
        "temperature": float(_f(p, "temperature")), "top_p": float(_f(p, "top_p")),
        "top_k": int(_f(p, "top_k")), "repetition_penalty": float(_f(p, "repetition_penalty"))}}


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
