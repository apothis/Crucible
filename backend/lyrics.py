"""Structure-aware song-lyric generation (LLM).

Writes lyrics that FIT a Song-Constructor arrangement instead of a generic block
of text: distinct VERSES that advance the story, ONE repeated CHORUS hook (reused
for every chorus, like a real song), and an optional PRE-CHORUS and BRIDGE.
Instrumental sections (intro / solo / breakdown / outro / interlude) are left
wordless. Runs through the shared provider layer (`llm.py`) — local Gemma on the
Mac (default) or Claude — so it's Mac-side, no 3090 contention.

`write_song_lyrics` returns lyrics keyed by the block's index in the arrangement,
so the UI can drop each lyric onto the right section block.
"""
import re

from . import llm as llm_mod

# Section roles that carry sung lyrics. Everything else (intro, solo, breakdown,
# outro, interlude, …) is treated as instrumental and left wordless.
_SUNG = {"verse", "chorus", "prechorus", "bridge", "refrain", "hook"}


def _norm(t):
    """'Pre-Chorus' → 'prechorus', 'Verse' → 'verse'."""
    return re.sub(r"[\s_-]+", "", str(t or "").strip().lower())


def is_sung(role):
    return _norm(role) in _SUNG


def _parse(text):
    """Parse the LLM's labeled output into {LABEL: lyric}. Labels normalised to
    VERSE1/VERSE2/…/CHORUS/PRECHORUS/BRIDGE. Tolerant of inline or own-line
    headers and stray markdown."""
    text = re.sub(r"```[a-z]*", "", text or "")
    pat = re.compile(r"(VERSE\s*\d+|CHORUS|PRE[\s-]?CHORUS|BRIDGE|INTRO|OUTRO)\s*:?", re.I)
    matches = list(pat.finditer(text))
    out = {}
    for j, m in enumerate(matches):
        label = re.sub(r"[\s-]+", "", m.group(1).upper())          # 'PRE-CHORUS' → 'PRECHORUS'
        start = m.end()
        end = matches[j + 1].start() if j + 1 < len(matches) else len(text)
        body = text[start:end].strip().strip("-—").strip()
        if body:
            out[label] = body
    return out


def write_song_lyrics(blocks, theme, style="", provider="", model="",
                      claude_model="claude-3-5-sonnet-latest", extra_sung=None):
    """Generate lyrics fitting the arrangement. `blocks` = [{type, …}] in order.
    Returns {"sections": [{"index", "type", "lyrics"}], "raw": <llm text>} with one
    entry per SUNG block (instrumental blocks omitted). Choruses share one hook;
    verses are distinct and ordered. `extra_sung` (e.g. ["intro","outro"]) opts those
    normally-instrumental roles into getting lyrics too."""
    extra = {_norm(r) for r in (extra_sung or [])}
    roles = [_norm(b.get("type")) for b in blocks]
    verse_idx = [i for i, r in enumerate(roles) if r == "verse"]
    chorus_idx = [i for i, r in enumerate(roles) if r in ("chorus", "refrain", "hook")]
    pre_idx = [i for i, r in enumerate(roles) if r == "prechorus"]
    bridge_idx = [i for i, r in enumerate(roles) if r == "bridge"]
    intro_idx = [i for i, r in enumerate(roles) if r == "intro"] if "intro" in extra else []
    outro_idx = [i for i, r in enumerate(roles) if r == "outro"] if "outro" in extra else []
    if not (verse_idx or chorus_idx or pre_idx or bridge_idx or intro_idx or outro_idx):
        return {"sections": [], "raw": ""}              # nothing sung in this arrangement

    provider = provider or llm_mod.best_provider()
    nv = len(verse_idx)
    wants, fmt = [], []
    if intro_idx:
        wants.append("a short INTRO — 1-2 atmospheric opening lines")
        fmt.append("INTRO:\n<1-2 lines>")
    if nv:
        wants.append(f"{nv} VERSE section(s), each DIFFERENT and advancing the story/imagery")
        fmt += [f"VERSE {k + 1}:\n<2-4 lines>" for k in range(nv)]
    if chorus_idx:
        wants.append("1 CHORUS — a memorable, anthemic hook (it repeats every chorus)")
        fmt.append("CHORUS:\n<2-4 punchy lines>")
    if pre_idx:
        wants.append("1 PRE-CHORUS that builds tension into the chorus")
        fmt.append("PRE-CHORUS:\n<1-2 lines>")
    if bridge_idx:
        wants.append("1 BRIDGE — a contrasting lyrical turn")
        fmt.append("BRIDGE:\n<2-3 lines>")
    if outro_idx:
        wants.append("an OUTRO — a closing refrain or fading final lines")
        fmt.append("OUTRO:\n<1-2 lines>")

    system = (
        "You are a skilled rock/metal lyricist. Write vivid, singable lyrics that fit "
        "the requested arrangement and match the given style and mood. Verses tell a "
        "developing story; the chorus is the emotional hook. Keep lines tight and "
        "singable. Output ONLY the labelled sections below, in this exact format and "
        "order, with no commentary, numbering of lines, or extra text:\n" + "\n".join(fmt)
    )
    prompt = (f"Theme: {theme or 'an epic, heroic metal song'}.\n"
              f"Style/mood: {style or 'heavy metal'}.\n"
              f"Write: " + "; ".join(wants) + ".")
    text = llm_mod.complete(provider, model, system, prompt, claude_model)
    res = _parse(text)

    sections = []

    def add(idx, typ, lyric):
        if lyric:
            sections.append({"index": idx, "type": typ, "lyrics": lyric})

    for i in intro_idx:
        add(i, "Intro", res.get("INTRO"))
    # verses: VERSE 1..n in order; if a slot is missing reuse any available verse
    any_verse = next((res[f"VERSE{k + 1}"] for k in range(nv) if res.get(f"VERSE{k + 1}")), "")
    for n, i in enumerate(verse_idx):
        add(i, "Verse", res.get(f"VERSE{n + 1}") or any_verse)
    for i in chorus_idx:
        add(i, "Chorus", res.get("CHORUS"))
    for i in pre_idx:
        add(i, "Pre-Chorus", res.get("PRECHORUS"))
    for i in bridge_idx:
        add(i, "Bridge", res.get("BRIDGE"))
    for i in outro_idx:
        add(i, "Outro", res.get("OUTRO"))

    # Fallback: if nothing parsed but the model produced text, drop it on the first
    # sung block so the user still gets something to edit.
    if not sections and (text or "").strip():
        first = (verse_idx or chorus_idx or pre_idx or bridge_idx or intro_idx or outro_idx)[0]
        sections = [{"index": first, "type": str(blocks[first].get("type", "Verse")),
                     "lyrics": text.strip()}]

    sections.sort(key=lambda s: s["index"])
    return {"sections": sections, "raw": text}
