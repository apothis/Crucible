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
    "Output STRICT JSON ONLY: {\"style\": \"...\", \"exclude\": \"...\"}. No prose, no markdown.")

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


def compile_suno(fields: dict, lyrics: str, title: str,
                 provider: str, model: str, claude_model: str) -> dict:
    caption = "\n".join(f"{k}: {v}" for k, v in fields.items() if str(v).strip())
    prompt = (f"{_RULES}\n\nSong title: {title or 'Untitled'}\n\n"
              f"The caption to compress:\n{caption}\n\n"
              "Return the JSON now.")
    style, exclude, source = "", "", "llm"
    try:
        text = llm.complete(provider, model, _SYSTEM, prompt, claude_model, timeout=240)
        mjson = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(mjson.group(0) if mjson else text)
        style = str(data.get("style") or "").strip()[:1000]
        exclude = str(data.get("exclude") or "").strip()[:500]
        if not style:
            raise ValueError("empty style from LLM")
    except Exception as e:
        style, exclude, source = fallback_style(fields), fallback_exclude(fields), f"fallback ({e})"
    return {"style": style, "exclude": exclude, "lyrics": map_lyrics(lyrics),
            "source": source}
