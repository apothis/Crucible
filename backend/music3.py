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
# worth making deliberately, so they live in one place.
DEFAULTS = {
    "seconds": 210.0, "cfg_scale": 1.7, "top_k": 50, "steps": 30,
    "sampler": "euler", "scheduler": "simple", "tiled_decode": True,
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
     "Open with 'Singer A (Female).' or 'Singer A (Male).' then the timbre. Add 'Singer B (...)' "
     "for a duet."),
    ("Vocal Style", "Vocal Details",
     "Delivery and how it changes across the song. State the negatives too: 'clean singing "
     "throughout, no growls and no screams' actually holds."),
    ("Harmony/Backing Vocals", "Vocal Details",
     "Stacks, choir, gang vocals, and WHERE they appear. 'A full mixed choir joins on the final "
     "chorus only' is the kind of instruction that lands."),
    ("Vocal FX", "Vocal Details",
     "Reverb, delay, compression, saturation on the voice."),
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


# ---------------- the graph ----------------

def build_graph(p):
    """Params -> (ComfyUI API graph, resolved params). Mirrors the shipped template
    audio_minimax_music_3.json; the template's main node is a subgraph, which a raw /prompt post
    cannot carry, so it is expanded here."""
    caption = (p.get("caption") or "").strip()
    lyrics = (p.get("lyrics") or "").strip()
    if not caption:
        raise ValueError("a caption is required (Music 3 has nothing else to go on)")

    seconds = max(1.0, min(float(p.get("seconds") or DEFAULTS["seconds"]), MAX_SECONDS))
    seed = int(p.get("seed") or 0)
    r = {
        "caption": caption, "lyrics": lyrics, "seconds": seconds, "seed": seed,
        "cfg_scale": float(p.get("cfg_scale") or DEFAULTS["cfg_scale"]),
        "top_k": int(p.get("top_k") or DEFAULTS["top_k"]),
        "steps": int(p.get("steps") or DEFAULTS["steps"]),
        "sampler": p.get("sampler") or DEFAULTS["sampler"],
        "scheduler": p.get("scheduler") or DEFAULTS["scheduler"],
        "tiled_decode": bool(p.get("tiled_decode", DEFAULTS["tiled_decode"])),
    }

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
        # seconds is LINKED from the encoder rather than set to `seconds`: the model decides its own
        # length inside the ceiling, and the latent has to match the conditioning it actually made.
        "6": {"class_type": "EmptyMiniMaxMusic3LatentAudio",
              "inputs": {"seconds": ["4", 1], "batch_size": 1}},
        "7": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0],
                         "latent_image": ["6", 0], "seed": seed, "steps": r["steps"],
                         "cfg": r["cfg_scale"], "sampler_name": r["sampler"],
                         "scheduler": r["scheduler"], "denoise": 1.0}},
        "8": decode,
        # FLAC, not MP3. The ACE graphs all save MP3 and the library historically assumed that, but
        # Music 3 decodes to genuine 44.1kHz (verified: no brick wall at 16kHz, so the bandwidth is
        # real rather than upsampled from the 32kHz the model card claims). Keeping the master
        # lossless costs ~5MB/minute; /api/export still hands out 320k MP3 on demand.
        "9": {"class_type": "SaveAudio",
              "inputs": {"audio": ["8", 0], "filename_prefix": "audio/music3"}},
    }
    return graph, r
