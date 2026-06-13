"""Music-video pipeline (Phase A): turn a song into an editable shot list via the LLM.

A "script" = an ordered list of shots, each timed to a song section, with a photoreal scene
prompt, a motion cue, which characters appear, and whether it is a lip-sync performance shot.
The user edits it, then each shot is rendered with the existing video builders (still -> i2v
or S2V) anchored on each character's reference still. Plain ASCII.
"""
import json
import re

from . import llm as llm_mod

SHOT_TYPES = ("performance", "narrative", "broll")


def _song_summary(song):
    """Human-readable song brief for the LLM + total duration (sec)."""
    lines = [f"Title: {song.get('title') or 'Untitled'}",
             f"Genre/mood tags: {song.get('tags') or ''}"]
    if song.get("bpm"):
        lines.append(f"BPM: {song['bpm']}")
    if song.get("keyscale"):
        lines.append(f"Key: {song['keyscale']}")
    secs = song.get("sections") or []
    t = 0
    body = []
    for s in secs:
        dur = int(s.get("seconds") or 0)
        lyr = (s.get("lyrics") or "").strip().replace("\n", " / ")
        body.append(f"  [{t}-{t + dur}s] {s.get('type') or 'section'}: "
                    + (lyr[:160] if lyr else "(instrumental)"))
        t += dur
    lines.append(f"Approx duration: {t}s across {len(secs)} sections:")
    lines.extend(body)
    return "\n".join(lines), t


def build_prompt(song, cast, n_shots):
    summary, total = _song_summary(song)
    cast_txt = "\n".join(f"  - {c.get('name')} ({c.get('role', 'character')})"
                         for c in cast if c.get("name")) or "  (no fixed cast - lean scenic/atmospheric)"
    target = n_shots or max(12, min(30, round((total or 180) / 8)))   # ~1 shot / 8s
    system = ("You are a music video director and editor. Output a SHOT LIST as STRICT JSON ONLY "
              "(no prose, no markdown fences). The video matches the song's energy and lyrics, cuts "
              "on the beat, and keeps the named characters visually consistent across their shots.")
    prompt = f"""{summary}

Characters (keep each visually consistent wherever they appear):
{cast_txt}

Write about {target} shots covering the whole song IN ORDER, 0 to {total}s with no gaps.
Return ONLY a JSON array. Each element is an object:
{{"section": "<section type>", "start": <sec int>, "end": <sec int>,
  "type": "performance" | "narrative" | "broll",
  "scene": "<vivid PHOTOREAL image prompt: setting, subject, wardrobe, lighting, camera/lens>",
  "motion": "<how the shot moves over ~5s>",
  "characters": [<names of any named characters present; [] if none>],
  "lipsync": <true ONLY for close performance shots where a named singer sings these lyrics>}}

Rules: performance/lipsync shots only on SUNG sections; scenic/broll on instrumental sections;
vary shot types and framing (close-up, wide, tracking); scene prompts must be concrete and
photoreal (this is a live-action style metal video, not animation)."""
    return system, prompt


def parse_shots(text):
    m = re.search(r"\[.*\]", text, re.S)
    raw = m.group(0) if m else text
    shots = json.loads(raw)
    out = []
    for i, s in enumerate(shots):
        if not isinstance(s, dict):
            continue
        t = s.get("type")
        out.append({
            "idx": i,
            "section": str(s.get("section") or ""),
            "start": int(float(s.get("start") or 0)),
            "end": int(float(s.get("end") or 0)),
            "type": t if t in SHOT_TYPES else "broll",
            "scene": str(s.get("scene") or "").strip(),
            "motion": str(s.get("motion") or "").strip(),
            "characters": [str(x) for x in (s.get("characters") or []) if x],
            "lipsync": bool(s.get("lipsync")),
        })
    return out


def generate_script(song, cast, provider, model, claude_model, n_shots=0):
    """Returns the parsed shot list (raises on LLM / JSON failure)."""
    system, prompt = build_prompt(song, cast, n_shots)
    text = llm_mod.complete(provider, model, system, prompt, claude_model)
    return parse_shots(text)
