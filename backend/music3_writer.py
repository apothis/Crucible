"""Two ways to author a Music 3 caption, both returning the SAME field dict so they can be diffed.

  "ours"  - one LLM call carrying what we measured on this box (see music3.py).
  "skill" - MiniMax's own `music-caption-rewriter`, vendored under backend/skills/, run the way its
            SKILL.md says to: route to a family, compare compact cards, read only the templates
            those cards name. Three small calls rather than one enormous one; the library is 1000
            templates and the whole point of the router is never to load them all.

Neither writer is trusted blindly. Both hand back the fields plus the reasoning inputs (which
family, which templates), and the UI shows the before/after per field so a rewrite can be accepted
one field at a time. The skill is very good at genre vocabulary and knows nothing about what we
measured here; our writer is the opposite. Keeping both, visibly, is the point.

One deliberate deviation from SKILL.md: its output contract is three prose headings, but our editor
is 13 discrete fields (the sub-labels its own templates use), so we ask for the fields directly.
Everything else about its method is followed.
"""
import json
import os
import re

from . import llm
from .music3 import CAPTION_FIELDS

SKILL_DIR = os.path.join(os.path.dirname(__file__), "skills", "music-caption-rewriter")
_KEYS = [k for k, _g, _h in CAPTION_FIELDS]


def available():
    return os.path.isdir(os.path.join(SKILL_DIR, "templates"))


def _read(*parts):
    """Read a file from inside the skill directory. The path is resolved and containment-checked
    because template ids come back from a model, and a model is an untrusted source of filenames."""
    path = os.path.realpath(os.path.join(SKILL_DIR, *parts))
    if not path.startswith(os.path.realpath(SKILL_DIR) + os.sep):
        raise ValueError(f"path escapes the skill directory: {parts}")
    with open(path, encoding="utf-8") as f:
        return f.read()


def _json(text, want_list=False):
    """Pull the first JSON value out of a model reply. Fenced blocks and a sentence of preamble are
    both common enough that failing on them would just mean retrying by hand."""
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    open_c, close_c = ("[", "]") if want_list else ("{", "}")
    i, j = t.find(open_c), t.rfind(close_c)
    if i < 0 or j <= i:
        raise ValueError(f"no JSON in model reply: {t[:200]}")
    return json.loads(t[i:j + 1])


def _brief(brief, song_fields, lyrics):
    """What both writers get told about the song."""
    out = [f"BRIEF: {brief.strip()}" if brief and brief.strip() else ""]
    if song_fields:
        have = {k: v for k, v in song_fields.items() if (v or "").strip()}
        if have:
            out.append("EXISTING CAPTION FIELDS (improve these, do not discard what is right):\n"
                       + json.dumps(have, indent=1))
    if lyrics and lyrics.strip():
        # SKILL.md is explicit that lyric text informs emotional context only and must never be
        # quoted back. The section tags are the part that carries structure, so send just those.
        tags = re.findall(r"\[[^\]]+\]", lyrics)
        out.append(f"SECTION TAGS IN ORDER ({len(tags)} sections): {' '.join(tags)}")
    return "\n\n".join(x for x in out if x)


_FIELD_SPEC = ("Return ONLY a JSON object with exactly these keys, every value a plain string:\n"
               + json.dumps(_KEYS, indent=1))


# ---------------- our writer ----------------

_OURS_SYSTEM = """You write MiniMax Music 3 captions for a metal-focused music studio.

The caption is the ONLY control surface. Section tags in the lyrics stay bare and carry nothing but
a section name, so every instruction has to live in these fields.

What has been MEASURED on this setup, which you must follow:

1. The genre goes at the END of Basic Attributes, in this exact phrasing:
   "bpm is 160. key is C, and scale is major. Symphonic Metal / Power Metal."
   Naming the genre anywhere else loses it.
2. Whatever is described in Primary becomes the centre of the mix. For rock and metal that is the
   guitars. Describing the orchestra first produced an orchestral track with the guitars buried.
3. Sonics & Production Profile must match how the genre is really produced. For metal: wall of
   sound, guitars panned hard left and right, heavily compressed. Asking for preserved dynamic
   range there steers straight out of metal into cinematic pop.
4. Address sections BY NAME in Global Emotional Progression, Groove & Foundation Progression and
   Embellishments ("the intro", "the first verse", "the bridge", "the final chorus"). A caption
   saying "the intro has no drums whatsoever" measured 16 dB less kick than one that did not.
   This is the only per-section control that exists.
5. Where several singers are wanted, name them "Singer A (Female)" and "Singer B (Male)" in
   Vocal Gender & Timbre, then assign them per named section in the arrangement fields.
6. Guitar solos: describe the solo in Embellishments against the named section that holds it.
7. There is NO per-section timing. Never write durations, bar counts or timestamps.
8. Aim for 250 to 450 words across all fields combined. Concrete musical changes, not adjectives.

Write in English. Do not invent an exact BPM or key if the brief does not imply one; leave that part
out rather than guessing. Never quote the lyrics."""


def write_ours(brief, fields=None, lyrics="", provider="", model="", claude_model=""):
    prompt = _brief(brief, fields, lyrics) + "\n\n" + _FIELD_SPEC
    raw = llm.complete(provider or llm.best_provider(), model, _OURS_SYSTEM, prompt,
                       claude_model, timeout=240)
    out = _json(raw)
    return {"fields": {k: str(out.get(k, "") or "").strip() for k in _KEYS},
            "writer": "ours", "provider": provider or llm.best_provider()}


# ---------------- MiniMax's skill ----------------

_ROUTE_SYSTEM = """You are routing a music brief to a style family, following the routing contract
in the genre router given below. Choose one primary family, and a secondary ONLY for an explicit
fusion. Treat "ballad", "emotional", "epic", "modern", "dark" and "cinematic" as modifiers, not
genres. Reply with ONLY a JSON array of the index filenames you chose, at most two."""

_PICK_SYSTEM = """You are selecting reference templates from style cards, following the selection
priority in the skill: genre and subgenre compatibility first, then explicit requirements, groove
and tempo, vocal configuration, instrumentation, mood, production. Apply a strong penalty to direct
conflicts, and prefer a close musical family over a card that merely shares mood words. Choose up to
three cards with DIFFERENT roles (Foundation, Modifier, Arrangement); choose fewer if the request is
simple. Reply with ONLY a JSON array of template filenames, e.g. ["power-metal-symphonic-metal_0001.txt"]."""


def _families(brief_text, provider, model, claude_model):
    router = _read("references", "genre-router.md")
    raw = llm.complete(provider, model, _ROUTE_SYSTEM,
                       f"{router}\n\n---\n\n{brief_text}", claude_model, timeout=180)
    names = [str(n) for n in _json(raw, want_list=True)]
    out = []
    for n in names[:2]:
        n = os.path.basename(n.strip())
        if not n.endswith(".md"):
            n += ".md"
        if os.path.exists(os.path.join(SKILL_DIR, "references", n)):
            out.append(n)
    # The router is the whole reason this stays cheap, but a bad route should degrade to a usable
    # caption rather than an error, so fall back to the broadest family.
    return out or ["index-general-pop-ballad.md"]


def _templates(brief_text, families, provider, model, claude_model):
    cards = "\n\n".join(_read("references", f) for f in families)
    raw = llm.complete(provider, model, _PICK_SYSTEM,
                       f"{cards}\n\n---\n\n{brief_text}", claude_model, timeout=180)
    picked = []
    for n in [str(x) for x in _json(raw, want_list=True)][:3]:
        n = os.path.basename(n.strip())
        if not n.endswith(".txt"):
            n += ".txt"
        if os.path.exists(os.path.join(SKILL_DIR, "templates", n)):
            picked.append(n)
    return picked


def write_skill(brief, fields=None, lyrics="", provider="", model="", claude_model=""):
    if not available():
        raise RuntimeError("the music-caption-rewriter skill is not installed under backend/skills/")
    provider = provider or llm.best_provider()
    brief_text = _brief(brief, fields, lyrics)

    families = _families(brief_text, provider, model, claude_model)
    templates = _templates(brief_text, families, provider, model, claude_model)
    refs = "\n\n=====\n\n".join(f"TEMPLATE {t}\n{_read('templates', t)}" for t in templates)

    system = (_read("SKILL.md")
              + "\n\n---\nOUTPUT OVERRIDE: this caller edits the caption as discrete fields, which "
                "are the sub-labels your own templates use. Follow every rule above, then express "
                "the finished caption as those fields instead of three prose headings. Use the "
                "references for musical identity, arrangement logic and vocabulary only: do not "
                "copy their sentences, key, BPM, instruments or section order.")
    prompt = f"{refs}\n\n---\n\n{brief_text}\n\n{_FIELD_SPEC}"
    raw = llm.complete(provider, model, system, prompt, claude_model, timeout=300)
    out = _json(raw)
    return {"fields": {k: str(out.get(k, "") or "").strip() for k in _KEYS},
            "writer": "skill", "provider": provider,
            "families": families, "templates": templates}
