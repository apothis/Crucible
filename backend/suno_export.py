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
  "Piano Intro", "A Cappella Intro"): a vague instrumental intro invites Suno's
  default lead-guitar noodling. Always end a sheet with [Outro] (and [End] last) so the
  song closes instead of cutting off.
- DYNAMICS TAGS give energy changes a sanctioned place: [Crescendo] on its own line
  marks where a build happens ([Build-Up] similar; [Pause] a held silence). After a
  gentle intro, place [Crescendo] where the caption says the band enters - without a
  sanctioned growth point, Suno inflates the intro itself.
- THE INTRO TAG MUST MATCH WHAT THE CAPTION SAYS THE INTRO IS - the generic
  "Orchestral Intro" on a metal song collapses every song to the same brass fanfare and
  crashing tutti. Be specific in the caption's own direction: a heroic opening earns
  "Fanfare Intro" or "Brass Intro"; a gentle one earns the SMALL element it names
  ("Piano Intro", "Strings Intro", "Quiet Intro", "Soft Cello Intro"); a rhythmic one
  "Drum Intro"; a chanted one "A Cappella Intro". Specific either way, generic never."""

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
- SWAPPING VOICES: when the caption defines TWO singers (Singer A and Singer B), cast
  EVERY sung section with a stacked vocal tag, written in the value as a slash pair:
  "Verse / Male Vocal", "Verse / Female Vocal", "Chorus / Duet", "Bridge / Male Vocal".
  Read the caption's Vocal Style field for who takes which section and cast accordingly;
  shared sections get "/ Duet". A section that alternates line-by-line still gets ONE
  cast (turn-taking by section is what Suno does reliably). Never leave a duet song's
  sections uncast - uncast sections converge to a single voice.
- A [Solo] section becomes a solo tag CONTAINING "Guitar Solo", with at most one style
  qualifier drawn from the caption: "Melodic Guitar Solo", "Shred Guitar Solo",
  "Blues Guitar Solo", "Harmonized Guitar Solo". Plain "Guitar Solo" when unsure."""

HOUSE_ENDING = "Complete every lyric before a short instrumental outro and final hard stop."
STYLE_LIMIT = 1000

_RULES = """Rules for "style" (Suno's Style of Music field) - the HOUSE FOUR-FIELD FORMAT
(measured 2026-10: labelled prose fields beat a comma list for vocal placement and arc):
- Exactly four labelled fields, each on its own line, in this order and with these labels:
  "Vocals:", "Style:", "Instruments:", "Production:". Plain prose sentences inside each
  field. CHARACTER BUDGETS, count before answering: Vocals about 380, Style about 110,
  Instruments about 250, Production about 120 - roughly 860 in all, never over 950 (Suno's
  hard limit is 1000; the app trims any field over its budget by whole sentences from its
  end, so an over-long Vocals field LOSES its chorus and backing-singer sentences). Over
  budget, cut adjectives, not sentences.
- Vocals: (the longest field) ONE singer unless the brief/caption defines two. State:
  gender + register + timbre + texture ("husky low register and natural rasp", "clear
  bright soprano", "bel canto" is fine - NEVER a real artist name); PLACEMENT in the mix
  in these words: "full and close, centred and forward in the mix"; the texture
  negatives as prose ("never thin, breathy or operatic" / "no rasp, rattle or
  harshness") - whichever the voice is NOT; a per-section ARC when lyrics exist
  ("Verses are low and restrained ... Pre-chorus rises ... Chorus opens into a huge
  sustained belt"); and BACKING SINGERS as their own voices with character and where
  they enter ("gritty female rock voices, dark harmonies rising around the lead in the
  final chorus"). If one voice only, add "female vocals only" (or male).
- Style: short - "<bpm> bpm, <key> <scale>, <one precise subgenre>" when known, then
  three to five feel words, e.g. "dark, aching and defiant, slow building, anthemic
  chorus".
- Instruments: the named instruments WITH adjectives (guitars first for rock/metal,
  tuning + drum-feel words) and the BUILD across the song: what plays alone at the
  open, what enters at the pre-chorus, what crashes in on the chorus, the solo. Measured
  rules for the opening: (a) name an opener instrument OUTSIDE the song's big ensembles
  (piano, tin whistle, music box, solo acoustic; NOT "strings"/"cello" when an orchestra
  is in the prompt - naming a section member puts the whole section on stage from bar
  one); (b) no energy words in the opening ("lone", "hushed", "distant", not "soaring",
  "epic"); (c) genre tokens carry intro priors, so a gentle open on a symphonic song
  needs (a)+(b) worked harder. End the field with "Instruments sit behind the vocal with
  clear separation."
- Production: a NAMED studio character ("Polished modern hard rock studio sound",
  "Polished Nashville studio sound"), the vocal's prominence again ("Warm prominent lead
  vocal"), width/separation/reverb words, the chorus lift "without burying the vocal",
  and ALWAYS this exact final sentence: "Complete every lyric before a short instrumental outro and final hard stop."
- No section-by-section lyric instructions ("the bridge says X"): the arc describes
  energy and voice, the lyric tags carry the words.

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


def _split_fields(text: str):
    """House four-field text -> ordered [(label, body)]; anything before the first label (or a
    plain comma list) comes back as [("", text)]."""
    parts = re.split(r"(?m)^(Vocals|Style|Instruments|Production):[ \t]*", text)
    if len(parts) == 1:
        return [("", text.strip())]
    out = [("", parts[0].strip())] if parts[0].strip() else []
    for i in range(1, len(parts), 2):
        out.append((parts[i], " ".join(parts[i + 1].split())))
    return out


# Per-field character budgets (labels + the ending sentence bring the total to ~980).
FIELD_BUDGET = {"Vocals": 380, "Style": 110, "Instruments": 250, "Production": 120, "": 60}


def _drop_last_sentence(body: str) -> str:
    """Drop one sentence from the end of a field. A short closing sentence (the house
    boilerplate: "Female vocals only.", "Instruments sit behind the vocal with clear
    separation.") is kept and the sentence before it goes instead."""
    sents = [x for x in re.split(r"(?<=[.!?])\s+", body.strip()) if x]
    if len(sents) < 2:
        return body
    if len(sents[-1]) <= 60 and len(sents) >= 3:
        return " ".join(sents[:-2] + sents[-1:])
    return " ".join(sents[:-1])


def finish_style(style: str) -> str:
    """House finishing for EVERY Suno style field the app writes: the ending sentence is
    present exactly once (at the end of Production), and the whole text fits Suno's
    1000-character limit. Over the limit, the LONGEST field loses its last sentence,
    repeatedly, so no field is ever dropped whole or cut mid-phrase."""
    t = (style or "").strip()
    if not t:
        return t
    t = t.replace(HOUSE_ENDING, "")
    fields = [(k, " ".join(v.split()).strip()) for k, v in _split_fields(t)]
    fields = [(k, v) for k, v in fields if v or k]
    if not any(k == "Production" for k, _ in fields):
        fields.append(("Production", ""))

    def render():
        lines = []
        for k, v in fields:
            if k == "Production":
                v = (v.rstrip() + (" " if v.rstrip().endswith((".", "!", "?")) else ". ") if v.strip() else "") + HOUSE_ENDING
            if k:
                lines.append(f"{k}: {v}".rstrip())
            elif v:
                lines.append(v)
        return "\n".join(lines)

    out = render()
    guard = 0
    while len(out) > STYLE_LIMIT and guard < 40:
        guard += 1
        # victim = the field furthest over its budget; if none is over, the rules' own cut
        # order (Instruments, Production, Style, and Vocals only as a last resort)
        over = [(len(v) - FIELD_BUDGET.get(k, 60), j) for j, (k, v) in enumerate(fields)]
        worst = max(over)
        if worst[0] > 0:
            i = worst[1]
        else:
            prio = {"Instruments": 0, "Production": 1, "Style": 2, "": 3, "Vocals": 4}
            cands = [j for j in range(len(fields)) if len(fields[j][1]) > 60] or list(range(len(fields)))
            i = min(cands, key=lambda j: (prio.get(fields[j][0], 3), -len(fields[j][1])))
        k, v = fields[i]
        nv = _drop_last_sentence(v)
        if nv == v:                                     # single sentence: hard-trim it at a word
            nv = v[: max(40, len(v) - 60)].rsplit(" ", 1)[0].rstrip(",;: ") + "."
        fields[i] = (k, nv)
        out = render()
    return out[:STYLE_LIMIT]


def _clip(text: str, n: int) -> str:
    """Cut at a word boundary, no trailing punctuation."""
    t = " ".join(str(text).split())
    if len(t) <= n:
        return t.rstrip(".,;: ")
    return t[:n].rsplit(" ", 1)[0].rstrip(".,;: ")


def fallback_style(fields: dict) -> str:
    """No-LLM compile from the caption's structured parts, in the house four-field format.
    Plainer than the LLM's version but always available and always on-format."""
    bpm, keyscale, genre = _basic_bits(fields)
    v = str(fields.get("Vocal Gender & Timbre") or "")
    genders = [g.lower() for g in re.findall(r"\((Female|Male)\)", v)]
    timbres = re.findall(r"possesses an? ([^.]+?) timbre", v)
    voices = []
    for i, t in enumerate(timbres):
        g = genders[i] if i < len(genders) else ""
        voices.append(((g + " ") if g else "") + t.strip() + " vocal")
    if not voices:
        voices = [(genders[0] + " vocal") if genders else "lead vocal"]
    vs = str(fields.get("Vocal Style") or "").split(".")[0].strip()
    vocals = (voices[0][0].upper() + voices[0][1:]) + ", full and close, centred and forward in the mix."
    if vs:
        vocals += " " + _clip(vs, 200) + "."
    if len(voices) > 1:
        vocals += " A duet: " + "; ".join(voices[1:]) + ", each voice distinct in register and grit."
    else:
        vocals += f" {'Female' if genders and genders[0] == 'female' else 'Male' if genders else 'Lead'} vocals only."
    hb = str(fields.get("Harmony/Backing Vocals") or "").split(".")[0].strip()
    if hb:
        vocals += f" Backing singers: {_clip(hb, 160)}."
    style_bits = [b for b in (f"{bpm} bpm" if bpm else "", keyscale or "",
                              genre.replace(" / ", ", ").lower() if genre else "") if b]
    mood = str(fields.get("Global Emotional Progression") or "").split(".")[0].strip()
    style_line = ", ".join(style_bits) + (f", {mood[:90].lower().rstrip('.')}" if mood else "")
    inst = []
    for key in ("Primary", "Secondary", "Groove & Foundation Progression"):
        t = str(fields.get(key) or "").split(".")[0].strip()
        if t:
            inst.append(re.sub(r"^electric guitars:\s*", "", _clip(t, 160).lower()))
    instruments = ("; ".join(inst).rstrip(".") + ". " if inst else "") + "Instruments sit behind the vocal with clear separation."
    prod = str(fields.get("Sonics & Production Profile") or "").split(".")[0].strip()
    production = (_clip(prod, 160) + ". " if prod else "Polished studio sound. ") + \
        "Warm prominent lead vocal, clear separation, chorus lift without burying the vocal."
    out = (f"Vocals: {vocals}\nStyle: {style_line or 'as the caption describes'}.\n"
           f"Instruments: {instruments}\nProduction: {production}")
    return finish_style(out)


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
              "build-up", "breakdown", "drop", "guitar solo", "outro", "end",
              "crescendo", "pause"}
_TAG_QUALIFIERS = {"orchestral", "piano", "drum", "riff", "acoustic", "choir", "chant",
                   "a cappella", "half-time", "heavy", "final", "whispered", "spoken",
                   "belted", "falsetto", "harmonized", "gang vocal", "drone", "fading",
                   "quiet", "melodic", "shred", "blues", "emotional", "neoclassical",
                   "duet", "soft", "gentle", "ambient", "strings", "cello", "harp",
                   "brass", "horn", "fanfare"}
# Standalone vocal-assignment tags (proven): stack under a section tag to cast the part.
_VOCAL_TAGS = {"male vocal", "female vocal", "duet", "harmonized", "whispered", "spoken",
               "belted", "falsetto", "gang vocals", "choir"}


def _valid_tag(text: str) -> bool:
    t = re.sub(r"\s+", " ", text.strip().lower())
    if t in _CORE_TAGS or t in _VOCAL_TAGS:
        return True
    for core in _CORE_TAGS:
        if not t.endswith(" " + core):
            continue
        prefix = t[: -len(core) - 1]
        if prefix in _TAG_QUALIFIERS:
            return True
        # two stacked qualifiers ("Soft Cello Intro") - both halves must be proven
        for i in range(1, len(prefix)):
            if prefix[:i].rstrip() in _TAG_QUALIFIERS and prefix[i:].lstrip() in _TAG_QUALIFIERS:
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


def _render_enriched(val: str) -> str:
    """One enriched value -> a tag line, or "" when invalid. A plain value becomes one
    tag; a slash pair ("Chorus / Duet") becomes the documented STACKED form
    "[Chorus] [Duet]" - the voice-casting mechanism for duet/swapping-voice songs.
    Every part is whitelisted independently."""
    parts = [p.strip() for p in val.split("/") if p.strip()]
    if not 1 <= len(parts) <= 2:
        return ""
    for p in parts:
        if "[" in p or "]" in p or len(p.split()) > 3 or len(p) > 30 or not _valid_tag(p):
            return ""
    return " ".join(f"[{p}]" for p in parts)


def apply_tag_enrichment(lyrics: str, enriched: dict) -> tuple:
    """Replace section-tag LINES by their 1-based occurrence number. The lyric words are
    untouched by construction - only lines that are already bare tags can be replaced.
    Values are validated hard because they come from a model; "A / B" values render as
    stacked tags ("[Verse] [Male Vocal]") to cast voices per section."""
    lines = (lyrics or "").split("\n")
    tags = _section_tags(lyrics)
    applied = 0
    for n, (idx, _orig) in enumerate(tags, 1):
        val = str(enriched.get(str(n)) or enriched.get(n) or "").strip()
        rendered = _render_enriched(val) if val else ""
        if not rendered:
            continue                      # imagery/off-vocabulary tag: keep the original
        lines[idx] = rendered
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
        style = finish_style(str(data.get("style") or ""))
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
            "style": finish_style(str(data.get("style") or "")),
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
