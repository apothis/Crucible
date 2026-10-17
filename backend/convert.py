"""Cross-engine song conversions: YuE2 <-> Music 3 <-> Suno.

Each engine wants a different prompt shape, so a song moves between tabs through a translation,
not a copy:

  Music 3  13 prose caption fields (4000+ chars)         backend/music3.py
  YuE2     ONE compact style line                        backend/yue2.py
  Suno     Style of Music (~700 chars) + Exclude Styles  backend/suno_export.py

Lyrics are the user's words and are never rewritten: only the bracket tags are mapped between
the engines' conventions (Suno's enriched/stacked tags, Music 3's [Solo]/[Instrumental], YuE2's
bare Title-case names). Where a translation needs judgement (expanding a line into a caption,
compressing a caption into a line) it is an LLM call with a deterministic fallback, the same
contract the Music 3 -> Suno export already has.

The user's working direction (2026-09-15): YuE2 follows the section theming better, Suno sounds
better, so a likely workflow is write and arrange in YuE2, then cover the take on Suno with the
same lyrics and a matching style. yue2_to_suno is that handoff.
"""
import json
import re

from . import llm, suno_export, music3 as music3_mod, yue2 as yue2_mod


def llm_args(p, cfg):
    provider = (p or {}).get("provider") or llm.best_provider()
    model = (p or {}).get("model") or ""
    if not model and provider in ("claude_sub", "claude_code", "claude"):
        model = "claude-sonnet-5"
    return provider, model, cfg.get("claude_model", "claude-3-5-sonnet-latest")


def _json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    return json.loads(m.group(0) if m else text)


# ---------------- lyric tag mapping ----------------

# Suno core tags that are not a section name in the other engines.
_SUNO_TO_BARE = {"hook": "Chorus", "break": "Interlude", "instrumental": "Interlude",
                 "instrumental break": "Interlude", "build-up": "Interlude", "breakdown": "Interlude",
                 "drop": "Interlude", "guitar solo": "Guitar Solo", "end": None, "crescendo": None,
                 "pause": None}


def suno_lyrics_to_bare(lyrics):
    """Suno sheet ([Acoustic Guitar Intro], [Chorus] [Belted], [End], ...) -> bare section tags.
    A tag line keeps the first part that names a section; delivery-only lines ([Belted] alone)
    and [End]/[Crescendo]/[Pause] lines are dropped. Lyric lines untouched."""
    out = []
    for line in (lyrics or "").split("\n"):
        s = line.strip()
        parts = re.findall(r"\[([^\[\]]+)\]", s)
        if not (parts and re.fullmatch(r"(?:\[[^\[\]]+\]\s*)+", s)):
            out.append(line)
            continue
        kept = None
        for ptxt in parts:
            # the number is split off first: the Suno core list only knows "verse 1"/"verse 2",
            # so "[Verse 3]" used to lose its tag and its words were glued onto the section before
            m = re.match(r"^(.*?)\s*(\d+)$", ptxt.strip())
            base, num = (m.group(1), m.group(2)) if m and m.group(1) else (ptxt, None)
            core = suno_export._nearest_core(base).lower()   # "" when no section word in it
            if not core:
                continue
            if core in _SUNO_TO_BARE:
                mapped = _SUNO_TO_BARE[core]
                if mapped is None:
                    continue
                kept = mapped
            else:
                kept = core.title().replace("-c", "-C")      # Intro, Verse, Pre-Chorus, Outro
            if num:
                kept = f"{kept} {num}"
            break
        if kept:
            out.append(f"[{kept}]")
    # an empty section repeating the one before it ("[Outro] words / [Instrumental Outro]" both
    # reduce to Outro) is one section, not two
    secs = []
    for sec in yue2_mod.lyrics_to_sections("\n".join(out)):
        if secs and not sec["lyrics"] and sec["tag"] == secs[-1]["tag"]:
            continue
        secs.append(sec)
    return "\n\n".join(f"[{x['tag']}]" + (f"\n{x['lyrics']}" if x["lyrics"] else "") for x in secs)


def bare_lyrics_to_music3(lyrics):
    """YuE2/bare tags -> Music 3's documented set: [Guitar Solo] -> [Solo], [Interlude] ->
    [Instrumental], numbered verses lose their number (Music 3 names sections in the caption)."""
    out = []
    for line in (lyrics or "").split("\n"):
        m = re.fullmatch(r"\[([^\]]+)\]\s*", line.strip())
        if not m:
            out.append(line)
            continue
        name = re.sub(r"\s+\d+$", "", m.group(1).strip())
        low = name.lower()
        name = {"guitar solo": "Solo", "interlude": "Instrumental", "break": "Instrumental"}.get(low, name)
        out.append(f"[{name}]")
    return "\n".join(out)


def _negatives(text):
    """'no growls, no synthesizers' style clauses out of prose -> ['growls', 'synthesizers']."""
    found = re.findall(r"\b(?:no|without|never)\s+([a-z][a-z \-]{2,40}?)(?=[,.;)]|\s+(?:no|and)\b|$)",
                       text or "", re.I)
    seen, out = set(), []
    for f in found:
        f = f.strip().lower()
        if f and f not in seen:
            seen.add(f)
            out.append(f)
    return out


# ---------------- YuE2 -> Suno ----------------

_Y2S_SYSTEM = (
    "You turn a compact music-style line into a Suno Custom Mode prompt. Output STRICT JSON ONLY: "
    "{\"style\": \"...\", \"exclude\": \"...\", \"tags\": {...}}. No prose, no markdown.")
# The house Suno rules (docs/SUNO_PROMPTING.md): the SAME style/exclude text and the SAME proven
# tag vocabulary + enrichment rules the Music 3 -> Suno export uses, plus what is specific to
# starting from a YuE2 line rather than a Music 3 caption.
_Y2S_RULES = suno_export._RULES + """

Starting from a YuE2 style line (not a Music 3 caption):
- Keep the line's genre, instruments and singer exactly; add nothing it does not name, except the
  bpm / key / subgenre header the rules require. bpm: the line's "NNN BPM" if present, else the
  score tempo given below, else omit. Key: the score key given below, else omit.
- The line names ONE singer unless it says duet: then ONE vocal phrase (register + timbre +
  delivery), never invent a second voice.
- Negatives in the line ("no male vocals") never go in "style"; put them in "exclude".
- Write the four labelled fields exactly as the rules above describe; the line's genre goes in
  Style:, its instruments (with the build) in Instruments:, its singer in Vocals:. COUNT the
  characters before answering: under 1000 total. Over it, cut adjectives from Instruments first,
  never the Vocals placement sentence or the Production ending sentence.

""" + suno_export._TAG_RULES + """
- Here the "caption" is the style line: derive the enrichments from the instruments and the
  singer it names and from the section flow (an intro tag names the line's lead melodic
  instrument when it is in the proven vocabulary, else "Quiet Intro"/"Drum Intro" as the line
  implies; a chanted chorus earns "Chorus / Gang Vocals"; the last chorus "Final Chorus"; an
  interlude between choruses "Instrumental Break" or "Guitar Solo" as the line implies; the
  closing section "Outro"). One singer = no vocal casting tags."""


def yue2_to_suno(style, lyrics, title, provider, model, claude_model, abc=""):
    hdr = yue2_mod.parse_header(abc) if (abc or "").strip() else {}
    score = ""
    if hdr.get("K") or hdr.get("bpm"):
        score = (f"\nFrom the score header: key {hdr.get('K') or 'unknown'}, tempo "
                 f"{hdr.get('bpm') or 'unknown'} bpm, meter {hdr.get('M') or 'unknown'}.")
    base_lyrics = yue2_mod.normalize_lyrics(lyrics)                       # Title-case bare tags
    tag_list = "\n".join(f"  {n}. [{t}]" for n, (_i, t) in enumerate(suno_export._section_tags(base_lyrics), 1))
    prompt = (f"{_Y2S_RULES}\n\nSong title: {title or 'Untitled'}\nThe style line:\n{style}{score}\n\n"
              f"The lyric sheet's section tags, numbered in order:\n{tag_list or '  (none)'}\n\n"
              "Return the JSON now.")

    def finish(text):
        # YuE2's [Interlude] -> Suno's name, then the [Solo]/[Instrumental] safety net; the words are untouched
        out = re.sub(r"(?mi)^\[interlude\]\s*$", "[Instrumental Break]", text)
        out = suno_export.map_lyrics(out)
        if not re.search(r"(?mi)^\[end\]\s*$", out):
            out = out.rstrip() + "\n[End]"                                  # the sheet always closes
        return out

    try:
        data = _json(llm.complete(provider, model, _Y2S_SYSTEM, prompt, claude_model, timeout=240))
        s = suno_export.finish_style(str(data.get("style") or ""))
        x = str(data.get("exclude") or "").strip().rstrip(",. ")[:500]
        if not s:
            raise ValueError("empty style from LLM")
        out_lyrics, enriched_n = base_lyrics, 0
        enriched = data.get("tags") or {}
        if isinstance(enriched, dict) and enriched:
            out_lyrics, enriched_n = suno_export.apply_tag_enrichment(base_lyrics, enriched)
        return {"style": s, "exclude": x, "lyrics": finish(out_lyrics), "tags_enriched": enriched_n, "source": "llm"}
    except Exception as e:
        negs = _negatives(style)
        kept = [t.strip() for t in re.split(r",", style) if t.strip() and not re.match(r"^(no|without|never)\b", t.strip(), re.I)]
        kept = [t for t in kept if not re.match(r"^english$", t, re.I)]
        is_voice = lambda t: re.search(r"vocal|voice|singer|soprano|mezzo|tenor|baritone|belt|falsetto", t, re.I) and len(t.split()) <= 10
        voices = [t for t in kept if is_voice(t)]
        rest = [t for t in kept if not is_voice(t)]
        bpm = next((t for t in rest if re.search(r"\d+\s*bpm", t, re.I)), "")
        rest = [t for t in rest if t != bpm]
        genre = rest[0] if rest else "as the style line describes"
        inst = rest[1:] if len(rest) > 1 else []
        vocals = (", ".join(voices) if voices else "lead vocal")
        vocals = vocals[0].upper() + vocals[1:] + ", full and close, centred and forward in the mix."
        fb = (f"Vocals: {vocals}\nStyle: " + ", ".join(b for b in (bpm, genre) if b) + ".\n"
              f"Instruments: {', '.join(inst) + '. ' if inst else ''}Instruments sit behind the vocal with clear separation.\n"
              "Production: Polished studio sound. Warm prominent lead vocal, clear separation, chorus lift without burying the vocal.")
        return {"style": suno_export.finish_style(fb), "exclude": ", ".join(negs)[:500],
                "lyrics": finish(base_lyrics), "tags_enriched": 0, "source": f"fallback ({e})"}


# ---------------- Suno -> YuE2 ----------------

def _suno_fields(style):
    """The house four-field Suno style ("Vocals: ...\nStyle: ...") -> {label: body}; a plain
    comma-list style (pre-2026-10 projects) comes back as {} so callers fall through."""
    out = {}
    for m in re.finditer(r"(?ms)^(Vocals|Style|Instruments|Production):\s*(.*?)(?=^\w+:|\Z)", style or ""):
        out[m.group(1)] = " ".join(m.group(2).split())
    return out


def suno_to_yue2(style, exclude, lyrics):
    """Deterministic: a Suno style line -> a compact YuE2 style line. The house four-field
    format is flattened first (Style genre + Instruments + the Vocals' first sentence, the
    mix-placement and Production boilerplate dropped - YuE2 wants a short comma line); a
    plain comma list passes through. The exclude field becomes short 'no X' clauses (the form
    upstream itself uses), capped so the line stays compact; enriched tags collapse to bare names."""
    f = _suno_fields(style)
    if f:
        drop = re.compile(r"centred and forward|in the mix|full and close|sit behind the vocal|clear separation|"
                          r"burying the vocal|instrumental outro|hard stop|vocals only", re.I)
        chunks = []
        for body in (f.get("Style", ""), f.get("Instruments", ""), (f.get("Vocals", "").split(". ")[0])):
            for c in re.split(r"[,;.]\s*", body):
                c = c.strip()
                if c and not drop.search(c) and c.lower() not in [x.lower() for x in chunks]:
                    chunks.append(c)
        style = ", ".join(chunks)
    parts = [t.strip() for t in re.split(r",", style or "") if t.strip()]
    if parts and not any(p.lower().startswith("english") for p in parts):
        parts.insert(0, "English")
    negs = [t.strip() for t in re.split(r",", exclude or "") if t.strip()][:4]
    parts += [f"no {n}" for n in negs]
    return {"style": ", ".join(parts), "lyrics": suno_lyrics_to_bare(lyrics)}


# ---------------- Music 3 -> YuE2 ----------------

_M2Y_SYSTEM = (
    "You compress a long structured music caption into ONE compact style line for the YuE2 music "
    "model. Output STRICT JSON ONLY: {\"style\": \"...\"}. No prose, no markdown.")
_M2Y_RULES = """Rules for "style" (upstream guidance: more adjectives do not create more control):
- One comma-separated line, at most 60 words, strongest identity first: language, ONE genre (the
  caption's Basic Attributes genre, verbatim), the defining instruments and where they sit (from
  Primary/Secondary), the vocal in one clause (gender, register, timbre, delivery: from Vocal
  Gender & Timbre and Vocal Style), then, if the caption describes the vocal or energy changing by
  section, a 2-3 phrase contrast clause ("intimate breathy verses, belted choruses, hushed final
  verse"), a two-word mood, at most two explicit negatives as short "no X" clauses, and tempo as
  "NNN BPM" LAST if the caption gives one.
- No sentences, no section-by-section arrangement narration (the contrast clause is the only
  per-section wording), no key or time signature, no living artist names."""


def music3_to_yue2(fields, lyrics, provider, model, claude_model):
    caption = music3_mod.assemble_caption(fields or {})
    out_lyrics = yue2_mod.normalize_lyrics(lyrics)
    if not caption.strip():
        return {"style": "", "lyrics": out_lyrics, "source": "empty caption"}
    prompt = f"{_M2Y_RULES}\n\nThe caption:\n{caption[:6000]}\n\nReturn the JSON now."
    try:
        data = _json(llm.complete(provider, model, _M2Y_SYSTEM, prompt, claude_model, timeout=240))
        s = str(data.get("style") or "").strip()
        if not s:
            raise ValueError("empty style from LLM")
        return {"style": s[:800], "lyrics": out_lyrics, "source": "llm"}
    except Exception as e:
        bpm, keyscale, genre = music3_mod.parse_basic_attributes(fields or {})
        f = fields or {}
        first = lambda k: re.split(r"(?<=[.!])\s+", str(f.get(k) or "").strip())[0].strip(" .") if f.get(k) else ""
        vocal = re.sub(r"^Singer [A-Z] \((\w+)\)\.\s*", lambda m: f"{m.group(1).lower()} vocal, ", first("Vocal Gender & Timbre"))
        vocal = re.sub(r"^The vocalist possesses ", "", vocal)
        parts = ["English", genre, first("Primary"), vocal, f"{bpm} BPM" if bpm else "", keyscale or ""]
        negs = _negatives(" ".join(str(v) for v in f.values()))[:4]
        parts += [f"no {n}" for n in negs]
        return {"style": ", ".join(p for p in parts if p)[:800], "lyrics": out_lyrics, "source": f"fallback ({e})"}


# ---------------- YuE2 / Suno -> Music 3 ----------------

def to_music3(brief_style, lyrics, title, provider, model, claude_model, negatives=""):
    """Expand a compact style (YuE2 line, or Suno style + exclude) into the 13 Music 3 caption
    fields with the Music 3 tab's own writer, so the result carries everything measured on this
    box (guitars in Primary, genre tail, absolute negatives)."""
    from . import music3_writer
    brief = (f"Song title: {title}. " if title else "") + \
        f"Keep EXACTLY this genre, these instruments and this voice: {brief_style}." + \
        (f" Must not contain: {negatives}." if negatives else "") + \
        " Fill every field; name the sections in the progression fields as the lyric sheet has them."
    m3_lyrics = bare_lyrics_to_music3(yue2_mod.normalize_lyrics(lyrics))
    r = music3_writer.write_ours(brief, fields=None, lyrics=m3_lyrics, provider=provider,
                                 model=model, claude_model=claude_model)
    return {"fields": r["fields"], "lyrics": m3_lyrics, "provider": r.get("provider")}
