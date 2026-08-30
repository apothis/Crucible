"""Compile a Music 3 caption + lyric sheet into a Suno Custom Mode prompt.

The caption stays the master document; this produces the three things Suno's Custom Mode
wants, per docs/SUNO_PROMPTING.md (researched 2026-08-31):

  style    the Style of Music field: ~6-7 front-loaded descriptors, under ~700 chars.
           More descriptors measurably mushes the mix (community consensus mirroring our
           own ACE tag-bloat finding), so compression is the whole game - which is why the
           compile is an LLM call rather than sentence-plucking regexes.
  exclude  the Exclude Styles field (Suno's reliable negative control; negatives written
           into the style prompt are ignored often enough to be useless).
  lyrics   the sheet passed through UNCHANGED except deterministic tag mapping:
           [Solo] -> [Guitar Solo] ([Guitar Solo] in the lyrics is what actually produces
           a solo there; the style field alone is unreliable for it).

The LLM path degrades gracefully: on any failure a deterministic fallback assembles a
serviceable style line from the structured parts of the caption, so the button always
returns something usable.
"""
import json
import re

from . import llm

_SYSTEM = (
    "You compress a structured music-generation caption into a Suno Custom Mode prompt. "
    "Output STRICT JSON ONLY: {\"style\": \"...\", \"exclude\": \"...\", \"tags\": {...}}. "
    "No prose, no markdown.")

# The PROVEN tag vocabulary (community-tested, docs/SUNO_PROMPTING.md section 3a).
# Anything outside it is a gamble; imagery words ("Ash Outro") are noise at best.
_TAG_VOCAB = """PROVEN Suno tag vocabulary - use ONLY this:
- Core structure tags (most reliable, use verbatim): Intro, Verse, Verse 1, Verse 2,
  Pre-Chorus, Chorus, Hook, Bridge, Break, Interlude, Instrumental, Instrumental Break,
  Build-Up, Breakdown, Drop, Guitar Solo, Outro, End.
- A core tag MAY take ONE qualifier, either an instrument/arrangement word (Orchestral,
  Piano, Drum, Riff, Acoustic, Choir, Chant, A Cappella, Half-time, Heavy, Final) or a
  proven delivery word (Whispered, Spoken, Belted, Falsetto, Harmonized, Gang Vocal):
  "Orchestral Intro", "Drum Intro", "Whispered Bridge", "Final Chorus", "Choir Interlude".
- Delivery may instead be STACKED as a second bracket tag on the same line - the
  documented pattern: "[Chorus] [Belted]", "[Verse] [Whispered]".
- MULTIPLE SINGERS: cast parts by stacking vocal tags per section - "[Verse] [Female
  Vocal]", "[Verse 2] [Male Vocal]", "[Chorus] [Duet]". Both voices must ALSO be
  described distinctly in the style field (register + weight + grit each) or they
  converge into one voice. Turn-taking by section is reliable; simultaneous different
  lines are not - shared moments use [Duet]/[Harmonized] and parenthesized echo lines.
- NEVER use the song's imagery or theme words in a tag ("Ash Outro", "Storm Bridge" are
  wrong; "Fading Outro", "Quiet Bridge" are right). If a word would not appear on a
  session musician's chart, it does not belong in a tag.
- Intro/outro tags should NAME the featured element from the caption ("Drum Intro",
  "Orchestral Intro", "A Cappella Intro"): a vague instrumental intro invites Suno's
  default lead-guitar noodling. Always end a sheet with [Outro] (and [End] last) so the
  song closes instead of cutting off."""

_TAG_RULES = _TAG_VOCAB + """

Rules for "tags" (enriched section tags for the lyric sheet):
- You are given the sheet's section tags as a NUMBERED list. Return {"<number>": "<enriched
  tag text>"} ONLY for sections where the caption's per-section direction earns an
  enrichment; omit a number to keep its original tag.
- 1-3 words from the proven vocabulary above, Title Case, NO brackets in the value, no
  punctuation.
- Derive them from the caption's Groove/Embellishments/Harmony/Vocal Style fields (e.g.
  "the intro has no drums, only massed stomps and claps" -> "A Cappella Intro"; "a full
  choir joins the final chorus" -> "Final Chorus").
- A [Solo] section becomes a solo tag CONTAINING "Guitar Solo", with at most one style
  qualifier drawn from the caption: "Melodic Guitar Solo", "Shred Guitar Solo",
  "Blues Guitar Solo", "Harmonized Guitar Solo". Plain "Guitar Solo" when unsure."""

_RULES = """Rules for "style" (Suno's Style of Music field):
- UNDER 700 characters. Aim for 6-7 descriptors TOTAL - more turns the mix to mush.
- FRONT-LOAD: start with "<bpm> bpm, <key> <scale>, <one precise subgenre>". Precision beats
  piling; never list three overlapping genre names.
- Then 2-3 instruments WITH adjectives (guitars first for rock/metal - include tuning words
  like "downtuned" and a drum-feel word like "double-kick" / "half-time stomp" when the
  caption implies them; these are what hold metal on-genre).
- Then ONE vocal phrase translating the caption's vocal timbre (register + weight + grit +
  delivery). Operatic vocabulary (operatic, bel canto, dramatic vibrato, classically
  trained) is GOOD here when the caption calls for it. NEVER name a real artist.
- Then 1-2 production words and 1-2 mood words.
- Plain comma-separated phrases, no sentences, no section-by-section direction (that lives
  in the lyric tags, not the style field).

Rules for "exclude" (Suno's Exclude Styles field):
- A short comma-separated list of what must NOT appear, derived from the caption's explicit
  negatives plus the genre's natural drift risks (e.g. metal: "pop, EDM, country, acoustic,
  trap"). Single words or two-word phrases only. No sentences."""


def map_lyrics(lyrics: str) -> str:
    """Deterministic tag mapping only - the words are the user's and are never rewritten."""
    out = (lyrics or "").strip()
    out = re.sub(r"(?mi)^\[solo\]\s*$", "[Guitar Solo]", out)
    out = re.sub(r"(?mi)^\[instrumental\]\s*$", "[Instrumental Break]", out)
    return out


def _basic_bits(fields: dict):
    basic = str(fields.get("Basic Attributes") or "")
    mb = re.search(r"bpm is (\d+)", basic)
    mk = re.search(r"key is ([A-G][#b]?),? and scale is (\w+)", basic)
    tail = [t.strip() for t in basic.split(".") if t.strip()]
    genre = tail[-1] if tail else ""
    return (mb.group(1) if mb else None,
            f"{mk.group(1)} {mk.group(2)}" if mk else None,
            genre)


def fallback_style(fields: dict) -> str:
    """No-LLM compile from the caption's structured parts. Plainer than the LLM's version
    but always available and always on-format."""
    bpm, keyscale, genre = _basic_bits(fields)
    bits = []
    if bpm:
        bits.append(f"{bpm} bpm")
    if keyscale:
        bits.append(keyscale)
    if genre:
        bits.append(genre.replace(" / ", ", ").lower())
    v = str(fields.get("Vocal Gender & Timbre") or "")
    gender = "female" if "(Female)" in v else "male" if "(Male)" in v else ""
    m = re.search(r"possesses an? ([^.]+?) timbre", v)
    if m:
        bits.append((f"{gender} " if gender else "") + m.group(1).strip() + " vocals")
    p = str(fields.get("Primary") or "").split(".")[0].strip()
    if p:
        bits.append(p[:140].lower().lstrip("electric guitars:").strip() or p[:140].lower())
    return ", ".join(bits)[:700]


def fallback_exclude(fields: dict) -> str:
    cap = " ".join(str(v) for v in fields.values()).lower()
    out = ["pop", "EDM", "trap"]
    for word, tag in (("no country", "country"), ("no acoustic", "acoustic"),
                      ("no synthesizers", "synth"), ("no rapping", "rap"),
                      ("no screams", "harsh vocals")):
        if word in cap:
            out.append(tag)
    return ", ".join(dict.fromkeys(out))


_TAG_LINE = re.compile(r"^\[(.+?)\]\s*$")

# Deterministic whitelist backing the prompt rules: the LLM still leaks caption imagery
# into tags occasionally ("[Mountain-Heartbeat Pre-Chorus]" observed with the rules in
# place), so every emitted tag is validated mechanically. A tag is valid when it is a
# proven CORE tag, optionally prefixed by ONE proven qualifier.
_CORE_TAGS = {"intro", "verse", "verse 1", "verse 2", "pre-chorus", "chorus", "hook",
              "bridge", "break", "interlude", "instrumental", "instrumental break",
              "build-up", "breakdown", "drop", "guitar solo", "outro", "end"}
_TAG_QUALIFIERS = {"orchestral", "piano", "drum", "riff", "acoustic", "choir", "chant",
                   "a cappella", "half-time", "heavy", "final", "whispered", "spoken",
                   "belted", "falsetto", "harmonized", "gang vocal", "drone", "fading",
                   "quiet", "melodic", "shred", "blues", "emotional", "neoclassical",
                   "duet"}
# Standalone vocal-assignment tags (proven): stack under a section tag to cast the part.
_VOCAL_TAGS = {"male vocal", "female vocal", "duet", "harmonized", "whispered", "spoken",
               "belted", "falsetto", "gang vocals", "choir"}


def _valid_tag(text: str) -> bool:
    t = re.sub(r"\s+", " ", text.strip().lower())
    if t in _CORE_TAGS or t in _VOCAL_TAGS:
        return True
    for core in _CORE_TAGS:
        if t.endswith(" " + core) and t[: -len(core) - 1] in _TAG_QUALIFIERS:
            return True
    return False


def _nearest_core(text: str) -> str:
    """Salvage an invalid tag: keep its core structure word if one is present
    ("Mountain-Heartbeat Pre-Chorus" -> "Pre-Chorus"), else drop the enrichment."""
    t = re.sub(r"\s+", " ", text.strip().lower())
    for core in sorted(_CORE_TAGS, key=len, reverse=True):
        if t == core or t.endswith(" " + core):
            return core.title().replace("-c", "-C")   # Pre-Chorus / Build-Up casing
    return ""


def sanitize_tags(lyrics: str) -> str:
    """Rules-check every bracket tag line in a sheet (single tags and stacked pairs).
    Invalid tags are reduced to their core structure tag when one is recognizable,
    otherwise left alone (a human wrote them or there is nothing safe to reduce to)."""
    out = []
    for line in (lyrics or "").split("\n"):
        stripped = line.strip()
        parts = re.findall(r"\[([^\[\]]+)\]", stripped)
        if parts and re.fullmatch(r"(?:\[[^\[\]]+\]\s*)+", stripped):
            fixed = []
            for ptxt in parts:
                if _valid_tag(ptxt):
                    fixed.append(f"[{ptxt.strip()}]")
                else:
                    core = _nearest_core(ptxt)
                    fixed.append(f"[{core}]" if core else f"[{ptxt.strip()}]")
            out.append(" ".join(fixed))
        else:
            out.append(line)
    return "\n".join(out)


def _section_tags(lyrics: str):
    """(line_index, tag_text) for every bare section-tag line, in order."""
    out = []
    for i, line in enumerate((lyrics or "").split("\n")):
        m = _TAG_LINE.match(line.strip())
        if m:
            out.append((i, m.group(1).strip()))
    return out


def apply_tag_enrichment(lyrics: str, enriched: dict) -> tuple:
    """Replace section-tag LINES by their 1-based occurrence number. The lyric words are
    untouched by construction - only lines that are already bare tags can be replaced.
    Values are validated hard (<=3 words, no brackets) because they come from a model."""
    lines = (lyrics or "").split("\n")
    tags = _section_tags(lyrics)
    applied = 0
    for n, (idx, _orig) in enumerate(tags, 1):
        val = str(enriched.get(str(n)) or enriched.get(n) or "").strip()
        if not val or "[" in val or "]" in val or len(val.split()) > 3 or len(val) > 30:
            continue
        if not _valid_tag(val):
            continue                      # imagery/off-vocabulary tag: keep the original
        lines[idx] = f"[{val}]"
        applied += 1
    return "\n".join(lines), applied


def compile_suno(fields: dict, lyrics: str, title: str,
                 provider: str, model: str, claude_model: str) -> dict:
    caption = "\n".join(f"{k}: {v}" for k, v in fields.items() if str(v).strip())
    tag_list = "\n".join(f"  {n}. [{t}]" for n, (_i, t) in enumerate(_section_tags(lyrics), 1))
    prompt = (f"{_RULES}\n\n{_TAG_RULES}\n\nSong title: {title or 'Untitled'}\n\n"
              f"The caption to compress:\n{caption}\n\n"
              f"The lyric sheet's section tags, numbered in order:\n{tag_list or '  (none)'}\n\n"
              "Return the JSON now.")
    style, exclude, source, out_lyrics, enriched_n = "", "", "llm", map_lyrics(lyrics), 0
    try:
        text = llm.complete(provider, model, _SYSTEM, prompt, claude_model, timeout=240)
        mjson = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(mjson.group(0) if mjson else text)
        style = str(data.get("style") or "").strip()[:1000]
        exclude = str(data.get("exclude") or "").strip()[:500]
        if not style:
            raise ValueError("empty style from LLM")
        enriched = data.get("tags") or {}
        if isinstance(enriched, dict) and enriched:
            out_lyrics, enriched_n = apply_tag_enrichment(lyrics, enriched)
            out_lyrics = map_lyrics(out_lyrics)   # [Solo] safety net if the model skipped it
    except Exception as e:
        style, exclude, source = fallback_style(fields), fallback_exclude(fields), f"fallback ({e})"
    return {"style": style, "exclude": exclude, "lyrics": out_lyrics,
            "tags_enriched": enriched_n, "source": source}


_WRITE_SYSTEM = (
    "You write prompts for the Suno music generator (Custom Mode). Output STRICT JSON "
    "ONLY: {\"title\": \"...\", \"style\": \"...\", \"exclude\": \"...\", \"lyrics\": \"...\"}. "
    "No prose, no markdown.")

_LYRIC_RULES = _TAG_VOCAB + """

Rules for "lyrics" (only when no lyrics are provided):
- Original lyrics, never quoting any existing song. Structure with bracket tags on their
  own lines: [Intro] [Verse] [Pre-Chorus] [Chorus] [Bridge] [Guitar Solo] [Outro].
- Tags come ONLY from the proven vocabulary above (core tags, one qualifier, or a
  stacked delivery tag). A solo section is a tag CONTAINING "Guitar Solo", optionally
  with ONE style qualifier ("[Melodic Guitar Solo]", "[Shred Guitar Solo]"); the house
  default is melodic and fast. End the sheet with [Outro], then [End].
- DELIVERY DIRECTION GOES IN BRACKET TAGS ONLY, never in parentheses: on Suno,
  parenthesized text is SUNG as a backing/echo vocal. So never write "(whispered)" or
  "(softly)" - write "[Whispered Verse]" as the section tag instead. Parentheses are
  allowed ONLY for actual words a backing vocal should sing, e.g. "(down we go)".
- Verses 4-8 lines, chantable choruses, a bridge that shifts the energy. Plain ASCII.
If lyrics ARE provided, return them EXACTLY as given (they are context for the style
fields; the caller keeps its own copy and ignores yours)."""


def write_suno(brief: str, title: str, style: str, exclude: str, lyrics: str,
               provider: str, model: str, claude_model: str, solo_style: str = "") -> dict:
    """Brief -> a full Suno Custom Mode prompt. Existing field values are given to the
    model as the starting point so a short brief ("more aggressive", "make it a ballad")
    edits rather than restarts."""
    ctx = []
    if title.strip():
        ctx.append(f"Current title: {title.strip()}")
    if style.strip():
        ctx.append(f"Current style field: {style.strip()}")
    if exclude.strip():
        ctx.append(f"Current exclude field: {exclude.strip()}")
    if lyrics.strip():
        ctx.append(f"Current lyrics (context only - do not rewrite):\n{lyrics.strip()}")
    solo_rule = (f"\nThe guitar solo tag must be exactly \"[{solo_style}]\".\n"
                 if solo_style.strip() else "")
    prompt = (f"{_RULES}\n\n{_LYRIC_RULES}{solo_rule}\n\n"
              + ("\n".join(ctx) + "\n\n" if ctx else "")
              + f"The brief, in the author's own words:\n{brief.strip() or '(none - use the current fields)'}\n\n"
              "Return the JSON now.")
    text = llm.complete(provider, model, _WRITE_SYSTEM, prompt, claude_model, timeout=240)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    data = json.loads(m.group(0) if m else text)
    return {"title": str(data.get("title") or "").strip()[:120],
            "style": str(data.get("style") or "").strip()[:1000],
            "exclude": str(data.get("exclude") or "").strip()[:500],
            "lyrics": sanitize_tags(
                _scrub_delivery_parens(str(data.get("lyrics") or "").strip()[:8000]))}


# Delivery words that must never appear parenthesized in a Suno lyric sheet: parens are
# SUNG there (backing/echo vocal), so "(whispered)" comes out as a sung word. Scrubbed
# ONLY from writer-drafted lyrics - user-typed lyrics are never modified. Real backing
# lines ("(down we go)") don't match: the scrub hits single delivery words only.
_DELIVERY_WORDS = ("whispered", "whisper", "whispering", "softly", "quietly", "spoken",
                   "shouted", "screamed", "screaming", "growled", "breathy", "hushed",
                   "belted", "gently", "loudly")
_DELIVERY_PAREN = re.compile(r"\(\s*(?:%s)\s*\)" % "|".join(_DELIVERY_WORDS), re.IGNORECASE)


def _scrub_delivery_parens(lyrics: str) -> str:
    lines = []
    for line in lyrics.split("\n"):
        cleaned = re.sub(r"  +", " ", _DELIVERY_PAREN.sub("", line)).rstrip()
        # a line that WAS only a delivery paren disappears entirely
        if line.strip() and not cleaned.strip():
            continue
        lines.append(cleaned if cleaned.strip() else line)
    return "\n".join(lines)
