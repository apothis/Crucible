"""MiniMax Music 3: caption authoring + ComfyUI graph.

A SECOND generation engine alongside ACE-Step, deliberately not wired into the Song tab. The two
want opposite things from a prompt and mixing them produces bad output for both:

  ACE-Step   ~10-12 dense tag phrases; more than that and takes come out disjointed.
  Music 3    a 4000+ character structured caption in THEIR field schema. Freeform prose loses the
             genre outright - measured on "Garden of Ashes", same seed, caption the only variable:
             prose gave cinematic orchestral pop, the schema gave symphonic metal.

Everything in CAPTION_FIELDS below is lifted from MiniMax's own caption-rewriter templates
(`skills/music-caption-rewriter/templates/*.txt` in MiniMax-AI/MiniMax-Music3), not invented here.

Two findings that shape this module, both measured rather than assumed:

  - Per-section style cues must NOT go in the lyric tags. `prompt.py`'s tag regex accepts any
    bracketed text, so `[chorus - anthemic, soaring]` looks legal, but the model SINGS everything
    after the section name: "enthemic soaring in the garden of ashes", "half-time emotional when the
    night was endless". Tags stay bare. Per-section guidance goes in the CAPTION, addressed by
    section name, which is what the canonical templates do and what `song_to_fields` compiles.
  - There are no timing tokens at all. Section ORDER and CONTENT are controllable; section LENGTH
    is not, at any price. `max_duration` is a ceiling and the model routinely stops short of it.
"""
import re

# 25 acoustic frames/sec, 9000 frames -> 360s. Mirrors comfy/ldm/minimax_music/ar.py.
MAX_SECONDS = 360.0
# The tokenizer hard-fails above this; the check is cheap and the failure is otherwise opaque.
MAX_PROMPT_TOKENS = 5000

# Sampler defaults are the shipped ComfyUI template's, not ours. Deviating from them is a change
# worth making deliberately, so they live in one place. Deliberate deviations so far:
#   seconds 360 (template 210)   - it is a ceiling, not a target; the model stops when the song
#                                  ends, so the max just stops truncating longer songs.
#   tiled_decode False (template True) - tiling is an OOM guard for small cards; the 3090's 24GB
#                                  decodes full-length untiled fine, and untiled has no seams.
DEFAULTS = {
    "seconds": 360.0, "cfg_scale": 1.7, "top_k": 50, "steps": 30,
    "sampler": "euler", "scheduler": "simple", "tiled_decode": False,
    # schedule "template" = the shipped KSampler graph exactly as before. "shift5" = AIPLAY
    # Studio's measured alternative: euler over a shift-5 flow schedule at 15 steps, which they
    # measured ~2x closer to the converged solution than euler/simple@30 in half the time.
    # Switchable per render; "template" stays the default until our own ears agree.
    "schedule": "template", "shift": 5.0, "shift5_steps": 15,
    # Audio-reference strength (AIPLAY's measured band, at 15 steps / shift 5):
    # 0.60 a copy, 0.80 a variation of the same song, 0.85 a genuine remix, 0.90+ ignored.
    "ref_denoise": 0.85,
}
MODELS = {
    "unet": "minimax_music3_dit_fp16.safetensors",
    "clip": "minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
    "vae": "minimax_music3_dav.safetensors",
}

# The full documented set, from ComfyUI's Music 3 tutorial (the README lists only a subset, which
# is why [Solo] and [Post-Chorus] look undocumented if you only read the repo). The tag regex
# accepts anything at all, but an undocumented tag is an experiment rather than a feature, so the
# picker offers these and lets anything else be typed by hand.
SECTION_TAGS = ["Intro", "Verse", "Pre-Chorus", "Chorus", "Post-Chorus",
                "Bridge", "Instrumental", "Solo", "Outro"]

# (key, group, guidance). Order IS the caption order; groups become the bare header lines.
CAPTION_FIELDS = [
    ("Basic Attributes", "Global Metadata",
     "Exact canonical phrasing: 'bpm is 160. key is C, and scale is major. Symphonic Metal / "
     "Power Metal.' The genre belongs HERE, at the end. Naming it anywhere else is the single "
     "most common way to lose it."),
    ("Global Emotional Progression", "Global Metadata",
     "The arc, section by section by NAME (the intro..., the first verse..., the bridge..., the "
     "final chorus...). Naming sections here is what makes the arrangement follow: a caption "
     "saying 'the intro has no drums whatsoever' measured 16 dB less kick than one that did not."),
    ("Application Scenarios & Imagery", "Global Metadata",
     "Where this music would be heard or what it depicts. Cheap to write and it colours the whole "
     "take: 'epic fantasy battle sequences, a rallying army marching from a burned city'."),
    ("Sonics & Production Profile", "Global Metadata",
     "Soundstage, frequency balance, dynamics. Match the genre's real production. For metal that "
     "means wall of sound, guitars panned hard left and right, HEAVILY COMPRESSED. Asking for "
     "'preserved dynamic range' here steers straight out of metal and into cinematic pop."),
    ("Vocal Gender & Timbre", "Vocal Details",
     "Prose, not a tag list. The corpus frame (770/1000 official templates): 'Singer A (Female). "
     "The vocalist possesses a powerful, clear soprano timbre with a bright, resonant quality "
     "capable of soaring over dense instrumentation. Her tone shifts from X in the verses to Y in "
     "the choruses.' Anchor on a register noun (soprano/mezzo/alto/tenor/baritone). 'bel canto', "
     "'coloratura', 'classically trained' appear in ZERO templates and derail the voice."),
    ("Vocal Style", "Vocal Details",
     "Delivery per named section, and vibrato lives HERE ('utilizing controlled vibrato', 'with "
     "strong vibrato'). Use 'operatic' as an adjective on a behaviour: 'a powerful, operatic "
     "projection in the choruses'. State the negatives too: 'clean singing throughout, no growls "
     "and no screams' actually holds."),
    ("Harmony/Backing Vocals", "Vocal Details",
     "Stacks, choir, gang vocals, and WHERE they appear. 'A full mixed choir joins on the final "
     "chorus only' is the kind of instruction that lands. To keep a lead unstacked: 'No harmony "
     "or backing vocals are present; the track relies entirely on the solo lead vocal.'"),
    ("Vocal FX", "Vocal Details",
     "Reverb, delay, compression, saturation on the voice. Never vibrato here (it is a Vocal "
     "Style word, 0/1000 templates put it in FX)."),
    ("Primary", "Arrangement",
     "The genre-defining instrument, and nothing else. Put the guitars here for rock and metal. "
     "Whatever is described first becomes the centre of the mix: describing the orchestra first "
     "is what produced an orchestral track with guitars buried in it."),
    ("Secondary", "Arrangement",
     "Supporting instruments. Orchestra, piano, bass, synths belong here, under the Primary."),
    ("Tertiary", "Arrangement",
     "Background texture and atmosphere."),
    ("Groove & Foundation Progression", "Arrangement",
     "The rhythm section, section by section by name. The most reliable per-section lever there "
     "is: 'the intro has no drums whatsoever', 'the bridge cuts to half-time'."),
    ("Embellishments, Textures & Spatial FX", "Arrangement",
     "Risers, fills, solos, one-off moments, again per named section."),
]
GROUP_ORDER = ["Global Metadata", "Vocal Details", "Arrangement"]
# Printed bare inside Arrangement, above Primary, exactly as the templates have it.
_ARRANGEMENT_HEADER = "Instrument Lifecycle Description (Primary/Secondary Layering):"

_FIELD_KEYS = [k for k, _g, _h in CAPTION_FIELDS]
_GROUP_OF = {k: g for k, g, _h in CAPTION_FIELDS}


def assemble_caption(fields):
    """Ordered field dict -> the caption text actually sent. Empty fields are dropped rather than
    emitted blank, so a half-filled form still produces a valid caption."""
    out, seen = [], set()
    for key in _FIELD_KEYS:
        val = (fields.get(key) or "").strip()
        if not val:
            continue
        group = _GROUP_OF[key]
        if group not in seen:
            out.append(group)
            seen.add(group)
            if group == "Arrangement":
                out.append(_ARRANGEMENT_HEADER)
        out.append(f"{key}: {val}")
    return "\n".join(out)


def parse_caption(text):
    """Caption text -> field dict. Lets a caption pasted from one of MiniMax's own templates be
    loaded straight into the editor."""
    fields, cur = {}, None
    for line in (text or "").split("\n"):
        key = line.split(":", 1)[0].strip() if ":" in line else None
        if key in _GROUP_OF:
            cur = key
            fields[cur] = line.split(":", 1)[1].strip()
        elif cur and line.strip() and line.strip() not in GROUP_ORDER and line.strip() != _ARRANGEMENT_HEADER:
            fields[cur] = (fields[cur] + " " + line.strip()).strip()
    return fields


# ---------------- importing an existing ACE song ----------------

_ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]
# ACE block types that carry no lyrics; Music 3 spells the played ones [Instrumental].
# [Solo] is documented, so an ACE "Solo" block keeps its own name; only the tags with no Music 3
# equivalent collapse to [Instrumental].
_INSTRUMENTAL = {"instrumental", "interlude", "break"}


def _section_names(blocks):
    """Human names for each block, numbered only where the type repeats, and with the LAST of a
    repeated type called 'final' - so a caption can say 'the final chorus' and mean the right one.
    That distinction is exactly what the Song tab encodes as separate per-block style strings."""
    types = [(b.get("type") or "Section").strip().lower() for b in blocks]
    counts, seen, names = {}, {}, []
    for t in types:
        counts[t] = counts.get(t, 0) + 1
    for i, t in enumerate(types):
        if counts[t] == 1:
            names.append(f"the {t}")
            continue
        seen[t] = seen.get(t, 0) + 1
        # "the final chorus" only earns its place from three occurrences up; with two, "the second"
        # is just as unambiguous and reads as normal English, which is what the caption is made of.
        if seen[t] == counts[t] and counts[t] >= 3:
            names.append(f"the final {t}")
        else:
            ordinal = _ORDINALS[seen[t] - 1] if seen[t] <= len(_ORDINALS) else str(seen[t])
            names.append(f"the {ordinal} {t}")
    return names


def song_to_lyrics(song):
    """SongDraft -> lyrics with BARE section tags.

    Bare is not a style choice. Anything after the section name inside the brackets gets sung: a
    test with the Song tab's own style strings in the tags produced "driving fuller, a scar story
    carved in stone" and wrecked the final chorus. The style strings go to song_to_fields instead.
    """
    out = []
    for b in (song or {}).get("blocks") or []:
        typ = (b.get("type") or "Section").strip()
        lyr = (b.get("lyrics") or "").strip()
        tag = "Instrumental" if typ.lower() in _INSTRUMENTAL else typ
        out.append(f"[{tag}]" + (f"\n{lyr}" if lyr else ""))
    return "\n\n".join(out)


def _keyscale(song):
    """'C major' / 'Eb minor' -> the canonical 'key is C, and scale is major' halves."""
    raw = ((song or {}).get("key") or "").strip()
    m = re.match(r"^([A-Ga-g][#b]?)\s*(major|minor|maj|min)?", raw)
    if not m:
        return None, None
    scale = (m.group(2) or "major").lower()
    return m.group(1).upper().replace("B", "b") if len(m.group(1)) > 1 else m.group(1).upper(), \
        {"maj": "major", "min": "minor"}.get(scale, scale)


def song_to_fields(song):
    """SongDraft -> prefilled caption fields.

    The per-block `style` strings are compiled into the three progression fields as prose addressed
    to named sections. That is the placement the canonical templates use, and the one measured to
    steer the arrangement - as opposed to the lyric tags, where the same strings get sung.
    """
    song = song or {}
    blocks = song.get("blocks") or []
    names = _section_names(blocks)
    tags = (song.get("tags") or "").strip()
    root, scale = _keyscale(song)
    bpm = song.get("bpm")

    basic = []
    if bpm:
        basic.append(f"bpm is {int(bpm)}.")
    if root:
        basic.append(f"key is {root}, and scale is {scale}.")
    # The Song tab's first tag phrase is usually the genre; the rest is texture that belongs in
    # Sonics. Left for the user to correct - guessing wrongly here is worse than leaving it obvious.
    genre = tags.split(",")[0].strip() if tags else ""
    if genre:
        basic.append(f"{genre.title()}.")

    def per_section(pick):
        parts = []
        for name, b in zip(names, blocks):
            cue = (b.get("style") or "").strip()
            if cue:
                parts.append(f"{name.capitalize()} is {cue}." if pick == "arc"
                             else f"{name.capitalize()}: {cue}.")
        return " ".join(parts)

    arc = per_section("arc")
    return {
        "Basic Attributes": " ".join(basic),
        "Global Emotional Progression": arc,
        "Application Scenarios & Imagery": "",
        "Sonics & Production Profile": ", ".join(t.strip() for t in tags.split(",")[1:]).strip(),
        "Vocal Gender & Timbre": "" if song.get("instrumental") else "Singer A (Female). ",
        "Vocal Style": "",
        "Harmony/Backing Vocals": "",
        "Vocal FX": "",
        "Primary": "",
        "Secondary": "",
        "Tertiary": "",
        "Groove & Foundation Progression": per_section("groove"),
        "Embellishments, Textures & Spatial FX": "",
    }


# ---------------- reverse translation: Music 3 -> Song tab (ACE-Step) ----------------
# The deterministic half of /api/music3/to_song. Lyrics keep VERBATIM fidelity here; only
# tags / per-section styles / durations - the parts needing semantic compression from a
# 4000-char caption down to ACE's 10-12 dense phrases - go through the LLM in app.py.

# Music 3 section tag -> Song tab SECTION_TYPES (web/src/forms.tsx). Post-Chorus and
# Instrumental have no Song-tab twin: Post-Chorus reads as a chorus variant; a bare
# Instrumental is closest to Breakdown (Solo stays Solo).
_TAG_TO_SONG = {"intro": "Intro", "verse": "Verse", "pre-chorus": "Pre-Chorus",
                "prechorus": "Pre-Chorus", "chorus": "Chorus", "post-chorus": "Chorus",
                "postchorus": "Chorus", "bridge": "Bridge", "instrumental": "Breakdown",
                "interlude": "Breakdown", "break": "Breakdown", "breakdown": "Breakdown",
                "solo": "Solo", "outro": "Outro"}


def lyrics_to_sections(lyrics):
    """Music 3 lyrics (bare [Tag] markers) -> ordered [{tag, type, lyrics}], lyrics verbatim.
    Tolerates ACE-style '[chorus - anthemic]' input by keeping only the name before '-'.
    Text before the first tag becomes a Verse (someone pasted lyrics without markers)."""
    out = []
    parts = re.split(r"\[([^\]\n]{1,40})\]", lyrics or "")
    pre = (parts[0] or "").strip()
    if pre:
        out.append({"tag": "Verse", "type": "Verse", "lyrics": pre})
    for i in range(1, len(parts), 2):
        tag = parts[i].strip()
        body = (parts[i + 1] if i + 1 < len(parts) else "").strip()
        base = tag.split("-")[0].strip().lower()
        out.append({"tag": tag, "type": _TAG_TO_SONG.get(base, "Verse"), "lyrics": body})
    return out


def parse_basic_attributes(fields):
    """Reverse of song_to_fields' Basic Attributes line -> (bpm, 'C major' keyscale, genre).
    Tolerant of hand-written variants; every piece is optional."""
    t = str((fields or {}).get("Basic Attributes") or "")
    bpm = None
    m = re.search(r"bpm\s*(?:is|[:=])?\s*(\d{2,3})", t, re.I)
    if m:
        bpm = int(m.group(1))
    keyscale = None
    km = re.search(r"key\s*(?:is|[:=])?\s*([A-Ga-g][#b♭♯]?)", t, re.I)
    sm = re.search(r"scale\s*(?:is|[:=])?\s*(major|minor)", t, re.I)
    if km:
        root = km.group(1)[0].upper() + km.group(1)[1:].replace("♭", "b").replace("♯", "#")
        keyscale = f"{root} {(sm.group(1).lower() if sm else 'major')}"
    # genre = whatever sentence(s) carry neither bpm nor key/scale
    genre = " ".join(s.strip() for s in re.split(r"(?<=[.!])\s+", t)
                     if s.strip() and not re.search(r"\b(bpm|key|scale)\b", s, re.I)).strip(" .")
    return bpm, keyscale, genre


# ---------------- the graph ----------------

def shift_sigmas(steps, shift):
    """The shift-N flow schedule, AIPLAY Studio's shiftSigmas verbatim: sigma(t) = s*t/(1+(s-1)*t)
    over a linear t from 1 to 0. Fed through the stock ManualSigmas node."""
    out = []
    for i in range(steps + 1):
        t = 1 - i / steps
        out.append(0.0 if i == steps else (shift * t) / (1 + (shift - 1) * t))
    return out


def build_graph(p):
    """Params -> (ComfyUI API graph, resolved params). Mirrors the shipped template
    audio_minimax_music_3.json; the template's main node is a subgraph, which a raw /prompt post
    cannot carry, so it is expanded here.

    Optional deviations, each defaulting to the template's exact behaviour:
      mix_seed  - noise seed for the sampler only. The encoder seed picks the COMPOSITION (the
                  AR token trajectory); this picks the RENDER of it. Because ComfyUI caches node
                  outputs in its long-lived process, holding seed and changing only mix_seed
                  skips the whole AR stage: AIPLAY measured re-rolls at 15 s vs 50 s.
      flow_cfg  - sampler guidance only (cfg_scale keeps steering the composition). One dial
                  driving both only ever samples the diagonal of that plane.
      schedule  - "template" (KSampler euler/simple) or "shift5" (euler over shift-5 sigmas,
                  default 15 steps).
      audio_ref - a .latent filename in ComfyUI's input dir (see dav_encoder.py). The sampler
                  then starts from that real recording instead of zeros, keeping only the tail
                  of the schedule (ref_denoise of it). Forces the shift5 path, where the
                  strength band was measured. The latent's length sets the duration.
    """
    caption = (p.get("caption") or "").strip()
    lyrics = (p.get("lyrics") or "").strip()
    if not caption:
        raise ValueError("a caption is required (Music 3 has nothing else to go on)")

    seconds = max(1.0, min(float(p.get("seconds") or DEFAULTS["seconds"]), MAX_SECONDS))
    seed = int(p.get("seed") or 0)
    audio_ref = (p.get("audio_ref") or "").strip()
    schedule = (p.get("schedule") or DEFAULTS["schedule"]).strip().lower()
    if audio_ref:
        schedule = "shift5"
    if schedule not in ("template", "shift5"):
        raise ValueError(f"unknown schedule {schedule!r} (template or shift5)")
    default_steps = DEFAULTS["shift5_steps"] if schedule == "shift5" else DEFAULTS["steps"]
    r = {
        "caption": caption, "lyrics": lyrics, "seconds": seconds, "seed": seed,
        "mix_seed": int(p.get("mix_seed") or 0) or seed,
        "cfg_scale": float(p.get("cfg_scale") or DEFAULTS["cfg_scale"]),
        "top_k": int(p.get("top_k") or DEFAULTS["top_k"]),
        "steps": int(p.get("steps") or default_steps),
        "sampler": p.get("sampler") or DEFAULTS["sampler"],
        "scheduler": p.get("scheduler") or DEFAULTS["scheduler"],
        "tiled_decode": bool(p.get("tiled_decode", DEFAULTS["tiled_decode"])),
        "schedule": schedule,
    }
    r["flow_cfg"] = float(p.get("flow_cfg") or 0) or r["cfg_scale"]
    if audio_ref:
        r["audio_ref"] = audio_ref
        r["ref_denoise"] = min(0.99, max(0.05, float(p.get("ref_denoise") or DEFAULTS["ref_denoise"])))

    decode = ({"class_type": "VAEDecodeAudioTiled",
               "inputs": {"samples": ["7", 0], "vae": ["3", 0], "tile_size": 1536, "overlap": 64}}
              if r["tiled_decode"] else
              {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}})

    graph = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": MODELS["unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": MODELS["clip"], "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": MODELS["vae"]}},
        "4": {"class_type": "MiniMaxMusic3TextEncode",
              "inputs": {"clip": ["2", 0], "caption": caption, "lyrics": lyrics, "seed": seed,
                         "max_duration": seconds, "cfg_scale": r["cfg_scale"], "top_k": r["top_k"]}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "8": decode,
        # FLAC, not MP3. The ACE graphs all save MP3 and the library historically assumed that, but
        # Music 3 decodes to genuine 44.1kHz (verified: no brick wall at 16kHz, so the bandwidth is
        # real rather than upsampled from the 32kHz the model card claims). Keeping the master
        # lossless costs ~5MB/minute; /api/export still hands out 320k MP3 on demand.
        "9": {"class_type": "SaveAudio",
              "inputs": {"audio": ["8", 0], "filename_prefix": "audio/music3"}},
    }

    # The starting latent. Normally empty, with seconds LINKED from the encoder rather than set:
    # the model decides its own length inside the ceiling, and the latent has to match the
    # conditioning it actually made. With an audio reference, the encoded latent of a real song -
    # its length then sets the duration, which is what you want: a remix runs as long as its source.
    if audio_ref:
        graph["6"] = {"class_type": "LoadLatent", "inputs": {"latent": audio_ref}}
    else:
        graph["6"] = {"class_type": "EmptyMiniMaxMusic3LatentAudio",
                      "inputs": {"seconds": ["4", 1], "batch_size": 1}}

    if schedule == "template":
        # The shipped template's sampler, byte-identical to before when mix_seed/flow_cfg are
        # left to follow seed/cfg_scale - which keeps ComfyUI's node cache valid across the change.
        graph["7"] = {"class_type": "KSampler",
                      "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0],
                                 "latent_image": ["6", 0], "seed": r["mix_seed"],
                                 "steps": r["steps"], "cfg": r["flow_cfg"],
                                 "sampler_name": r["sampler"], "scheduler": r["scheduler"],
                                 "denoise": 1.0}}
    else:
        sig = shift_sigmas(r["steps"], DEFAULTS["shift"])
        if audio_ref:
            # Keep only the TAIL of the schedule, so the flow starts partway down and less of
            # the reference is destroyed. Done with arithmetic here rather than a custom node.
            keep = max(1, round(r["steps"] * r["ref_denoise"]))
            sig = sig[-(keep + 1):]
        graph["10"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": r["sampler"]}}
        graph["11"] = {"class_type": "ManualSigmas",
                       "inputs": {"sigmas": ", ".join(f"{v:.6f}" for v in sig)}}
        graph["7"] = {"class_type": "SamplerCustom",
                      "inputs": {"model": ["1", 0], "add_noise": True,
                                 "noise_seed": r["mix_seed"], "cfg": r["flow_cfg"],
                                 "positive": ["4", 0], "negative": ["5", 0],
                                 "sampler": ["10", 0], "sigmas": ["11", 0],
                                 "latent_image": ["6", 0]}}
    return graph, r
