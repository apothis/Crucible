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
            core = suno_export._nearest_core(ptxt).lower()   # "" when no section word in it
            if not core:
                continue
            if core in _SUNO_TO_BARE:
                mapped = _SUNO_TO_BARE[core]
                if mapped is None:
                    continue
                kept = mapped
            else:
                kept = core.title().replace("-c", "-C")      # Intro, Verse 2, Pre-Chorus, Outro
            break
        if kept:
            out.append(f"[{kept}]")
    return yue2_mod.normalize_lyrics("\n".join(out))


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
    "{\"style\": \"...\", \"exclude\": \"...\"}. No prose, no markdown.")
_Y2S_RULES = """Rules:
- "style": Suno's Style of Music field. 6-7 front-loaded comma-separated descriptors, under 700
  characters: genre first, then instruments/production, then the vocal (gender, register, timbre,
  delivery), tempo feel, mood. Keep the ORIGINAL line's genre, instruments and voice exactly; do
  not add new instruments. Drop negatives ("no ...") from the style: they go in "exclude".
- "exclude": Suno's Exclude Styles field: the line's own negatives plus the genre's natural drift
  risks (metal: "pop, EDM, country, acoustic, trap, rap"). Single words or two-word phrases, comma
  separated, no sentences."""


def yue2_to_suno(style, lyrics, title, provider, model, claude_model):
    prompt = f"{_Y2S_RULES}\n\nSong title: {title or 'Untitled'}\nThe style line:\n{style}\n\nReturn the JSON now."
    out_lyrics = suno_export.map_lyrics(yue2_mod.normalize_lyrics(lyrics))   # Title-case bare tags, then Suno's names
    try:
        data = _json(llm.complete(provider, model, _Y2S_SYSTEM, prompt, claude_model, timeout=240))
        s, x = str(data.get("style") or "").strip()[:1000], str(data.get("exclude") or "").strip()[:500]
        if not s:
            raise ValueError("empty style from LLM")
        return {"style": s, "exclude": x, "lyrics": out_lyrics, "source": "llm"}
    except Exception as e:
        negs = _negatives(style)
        kept = [t.strip() for t in re.split(r",", style) if t.strip() and not re.match(r"^(no|without|never)\b", t.strip(), re.I)]
        return {"style": ", ".join(kept)[:1000], "exclude": ", ".join(negs)[:500],
                "lyrics": out_lyrics, "source": f"fallback ({e})"}


# ---------------- Suno -> YuE2 ----------------

def suno_to_yue2(style, exclude, lyrics):
    """Deterministic: Suno's descriptor list already reads as a YuE2 style line. The exclude
    field becomes short 'no X' clauses (the form upstream itself uses), capped so the line stays
    compact; the sheet's enriched tags collapse to bare section names."""
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
  Gender & Timbre and Vocal Style), tempo as "NNN BPM" if the caption gives one, a two-word mood,
  and the caption's explicit negatives as short "no X" clauses.
- No sentences, no section-by-section narration, no living artist names, nothing about
  timing."""


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
