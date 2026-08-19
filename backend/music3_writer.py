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


def families():
    """The style families, with the count of cards in each. Named from the index filenames rather
    than the router table so the list cannot drift from what is actually on disk."""
    out = []
    for f in sorted(os.listdir(os.path.join(SKILL_DIR, "references"))):
        if not f.startswith("index-") or not f.endswith(".md"):
            continue
        text = _read("references", f)
        title = next((l.lstrip("# ").strip() for l in text.split("\n") if l.startswith("#")), f)
        out.append({"file": f, "label": title, "cards": len(_cards_from(text))})
    return out


def _cards_from(text):
    """Parse an index's markdown table. Columns, in order:
    ID | Style | Secondary routes | Tempo / key | Mood arc | Vocal cue | Core palette | Template"""
    cards = []
    for line in text.split("\n"):
        if not line.startswith("|"):
            continue
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cols) < 8 or cols[0] in ("ID", "---") or set(cols[0]) <= {"-", ":"}:
            continue
        tmpl = re.search(r"templates/([A-Za-z0-9._-]+\.txt)", cols[7])
        cards.append({
            "id": cols[0].strip("`"), "style": cols[1], "secondary": cols[2],
            "tempo_key": cols[3], "mood": cols[4], "vocal": cols[5], "palette": cols[6],
            "template": tmpl.group(1) if tmpl else "",
        })
    return [c for c in cards if c["template"]]


def cards(family):
    family = os.path.basename(family)
    if not (family.startswith("index-") and family.endswith(".md")):
        raise ValueError("not a family index filename")
    return _cards_from(_read("references", family))


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


# NOT JSON. The field values are prose, and the guidance asks the model to write phrases like
# "no drums whatsoever" in quotes, which it then echoes into the values - unescaped double quotes
# inside a JSON string, which is a hard parse failure with no useful recovery. A line-delimiter
# format cannot be broken by any punctuation the prose contains.
_FIELD_SPEC = (
    "Return the caption as delimited blocks and NOTHING else. One block per field, in this order, "
    "each opening with a line that is exactly '### ' followed by the field name, then the field's "
    "text on the following lines. Use no other headings, no JSON, no commentary, no markdown "
    "emphasis. Leave a field's text empty if it genuinely has nothing to say.\n\n"
    + "\n".join(f"### {k}" for k in _KEYS))


def _blocks(text):
    """Parse the '### Field name' format back into a field dict. Unknown headings are ignored rather
    than guessed at, and a missing field simply stays empty."""
    out, cur = {}, None
    for line in (text or "").replace("\r\n", "\n").split("\n"):
        m = re.match(r"^\s{0,3}#{2,4}\s*(.+?)\s*:?\s*$", line)
        key = m.group(1).strip() if m else None
        if key and key in _KEYS:
            cur = key
            out[cur] = []
        elif cur is not None:
            out[cur].append(line)
    parsed = {k: "\n".join(v).strip() for k, v in out.items()}
    if not parsed:
        raise ValueError(f"no '### Field' blocks in model reply: {(text or '')[:300]}")
    return parsed


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
5. State absolute negatives absolutely: "no drums whatsoever", "no guitars at all", "the drums drop
   out entirely" rather than "piano-led" or "sparse". NOT PROVEN, unlike the rest of this list. It
   is house style on the reasoning that an implied absence is weaker than a stated one, and a
   caption that stated it did get a bare intro - but a caption using the same phrase got a busy
   one, so the phrasing is clearly not sufficient on its own. Do not treat this as a measured rule.
6. Where several singers are wanted, name them "Singer A (Female)" and "Singer B (Male)" in
   Vocal Gender & Timbre, then assign them per named section in Vocal Style prose ("Singer B
   enters in the bridge", "both vocalists layer their parts in the final chorus"). Casting can be
   reinforced from the lyrics side with a short bracketed vocal attribution line (see the lyric
   rules), but the caption assignment comes first.
7. VOCAL GRAMMAR. Measured against the official 1000-template caption corpus this model was
   aligned to (backend/skills/music-caption-rewriter/templates/):
     - the vocal fields are PROSE SENTENCES, never a comma list of tags. The corpus frame, used
       in 770 of 1000 templates, is: 'Singer A (Female). The vocalist possesses a <two or three
       adjectives> <register noun> timbre with a <quality> capable of <function in the mix>.
       Her tone shifts from <X> in the <softer sections> to <Y> in the <heavier sections>.';
     - anchor the voice on a register noun (soprano, mezzo-soprano, alto, tenor, baritone,
       bass): 929 of 1000 corpus timbre lines contain one;
     - adjectives the corpus actually uses: clear, resonant, breathy, powerful, smooth, bright,
       warm, gritty, raspy, rich;
     - these phrases appear in ZERO corpus templates and are known here to derail the voice:
       "bel canto", "coloratura", "classically trained", "operatic soprano". Use "operatic"
       only as an adjective on a section-anchored behaviour: "a powerful, operatic projection
       in the choruses", "an operatic belt";
     - vibrato belongs in Vocal Style (279 corpus uses), not Timbre (13) and never Vocal FX (0):
       "with strong vibrato", "utilizing controlled vibrato";
     - to keep a lead unstacked, use the corpus form in Harmony/Backing Vocals: "No harmony or
       backing vocals are present; the track relies entirely on the solo lead vocal.";
     - the model has no speaker conditioning, so the exact voice character stays partly a seed
       lottery however well the fields are written. Write the grammar right, expect rerolls.
8. GUITAR SOLOS. Measured across 18 takes (6 caption variants x seeds 7/8/9). What actually works:
     - the section itself was NEVER the problem: every variant produced a 13-39s instrumental
       stretch, including ones that barely mentioned a solo;
     - the reliable trigger for the model to PLAY something soloistic there is the lyric tag
       [Solo] rather than [Instrumental]. That is one word and it worked 3 times out of 3;
     - in Embellishments, use the corpus's HANDOVER framing rather than describing technique:
       "an extended guitar solo section where the lead electric guitar takes the lead outright,
       trading phrases with the orchestral strings before the full ensemble returns". This was
       the best-sounding variant and held a clean structure at every seed;
     - name the lead guitar in Primary as taking the melodic lead in the instrumental break, not
       only as the riff instrument.
   What did NOT work, so do not bother: explicitly forbidding a keyboard or synth solo. A caption
   that said "there is no keyboard solo, no synth solo and no organ solo anywhere in this song"
   scored exactly the same as one that did not. The remaining failure is TIMBRE - the model plays
   the solo but often on a synth voice - and it stays at roughly one take in three whatever the
   caption says. Write the solo well and expect to hunt seeds for the tone.
9. There is NO per-section timing. Never write durations, bar counts or timestamps.
10. Aim for 250 to 450 words across all fields combined. Concrete musical changes, not adjectives.

Write in English. Do not invent an exact BPM or key if the brief does not imply one; leave that part
out rather than guessing. Never quote the lyrics."""


def write_ours(brief, fields=None, lyrics="", provider="", model="", claude_model=""):
    prompt = _brief(brief, fields, lyrics) + "\n\n" + _FIELD_SPEC
    raw = llm.complete(provider or llm.best_provider(), model, _OURS_SYSTEM, prompt,
                       claude_model, timeout=240)
    out = _blocks(raw)
    return {"fields": {k: str(out.get(k, "") or "").strip() for k in _KEYS},
            "writer": "ours", "provider": provider or llm.best_provider()}


# ---------------- lyrics (both writer paths) ----------------

_LYRICS_SYSTEM = """You write song lyrics for MiniMax Music 3 in a metal-focused studio.

Hard rules, all measured on this setup:
- Structure the song with BARE section tags, one per line, chosen from: [Intro], [Verse],
  [Pre-Chorus], [Chorus], [Post-Chorus], [Bridge], [Instrumental], [Solo], [Outro].
  A tag carries NOTHING but the section name: anything after it inside the brackets gets SUNG.
- ONE exception: a section tag may be followed on ITS OWN NEXT LINE by a short bracketed vocal
  attribution of one or two words, e.g. [female vocal] or [male vocal]. This works as a casting
  lever. Use it only when the caption names more than one singer, or when a section must switch
  voice; never more than two words, never anything but the voice.
- Parentheses are UNRELIABLE in both directions: sometimes performed as sung backing-vocal
  lines, sometimes silently obeyed as direction. So never put a stage direction in parentheses
  ("(guitar solo begins)" may be sung), and never rely on parentheses to guarantee a sung
  backing line either. Use them only for words that are acceptable to hear sung.
- [Intro], [Instrumental], [Solo] and [Outro] normally carry no lyric lines. Use [Solo] for a
  guitar solo section - it is the reliable trigger for one.
- Follow the section order implied by the caption fields when they describe one; otherwise use a
  conventional shape (intro, verse, pre-chorus, chorus, verse, pre-chorus, chorus, solo, bridge,
  final chorus, outro).
- Write in English unless the brief says otherwise. No titles, no commentary, no markdown:
  return ONLY the lyrics, starting with the first section tag."""


def write_lyrics(brief, fields=None, provider="", model="", claude_model=""):
    """Draft full lyrics from the brief plus whatever caption fields exist. Callers only invoke
    this when the lyrics box is EMPTY - existing lyrics are never rewritten, matching the ACE
    writer's contract."""
    ctx = [f"BRIEF: {(brief or '').strip()}"]
    for k in ("Basic Attributes", "Global Emotional Progression", "Vocal Gender & Timbre"):
        v = ((fields or {}).get(k) or "").strip()
        if v:
            ctx.append(f"{k}: {v}")
    raw = llm.complete(provider or llm.best_provider(), model, _LYRICS_SYSTEM,
                       "\n\n".join(ctx), claude_model, timeout=240)
    t = re.sub(r"^```[a-z]*|```$", "", (raw or "").strip(), flags=re.M).strip()
    # anything before the first tag is preamble the model was told not to write
    i = t.find("[")
    return t[i:] if i > 0 else t


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


def write_skill(brief, fields=None, lyrics="", provider="", model="", claude_model="",
                pick_family="", pick_templates=None):
    """`pick_family` / `pick_templates` override the routing and selection calls. Auto-routing is
    convenient but NOT reproducible - the same brief picked a different template trio on two
    consecutive runs - so pinning the references is the only way to hold them steady across an A/B."""
    if not available():
        raise RuntimeError("the music-caption-rewriter skill is not installed under backend/skills/")
    provider = provider or llm.best_provider()
    brief_text = _brief(brief, fields, lyrics)

    if pick_templates:
        chosen, fams, routed = [], [], False
        for n in pick_templates[:3]:
            n = os.path.basename(str(n).strip())
            if not n.endswith(".txt"):
                n += ".txt"
            if os.path.exists(os.path.join(SKILL_DIR, "templates", n)):
                chosen.append(n)
        if not chosen:
            raise ValueError("none of the chosen templates exist")
        templates = chosen
        fams = [os.path.basename(pick_family)] if pick_family else []
    elif pick_family:
        fams = [os.path.basename(pick_family)]
        templates = _templates(brief_text, fams, provider, model, claude_model)
        routed = False
    else:
        fams = _families(brief_text, provider, model, claude_model)
        templates = _templates(brief_text, fams, provider, model, claude_model)
        routed = True
    families = fams
    refs = "\n\n=====\n\n".join(f"TEMPLATE {t}\n{_read('templates', t)}" for t in templates)

    system = (_read("SKILL.md")
              + "\n\n---\nOUTPUT OVERRIDE: this caller edits the caption as discrete fields, which "
                "are the sub-labels your own templates use. Follow every rule above, then express "
                "the finished caption as those fields instead of three prose headings. Use the "
                "references for musical identity, arrangement logic and vocabulary only: do not "
                "copy their sentences, key, BPM, instruments or section order.")
    prompt = f"{refs}\n\n---\n\n{brief_text}\n\n{_FIELD_SPEC}"
    raw = llm.complete(provider, model, system, prompt, claude_model, timeout=300)
    out = _blocks(raw)
    return {"fields": {k: str(out.get(k, "") or "").strip() for k in _KEYS},
            "writer": "skill", "provider": provider, "auto_routed": routed,
            "families": families, "templates": templates}
