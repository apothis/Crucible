"""AI Melody Composer — builds a note-per-syllable vocal melody from a Song
Constructor song (section structure + key + BPM + per-section lyrics).

Design (see RESEARCH.md §5b): an LLM proposes the melodic *intent* as scale
degrees (small integers, inherently in-key and trivial to parse) and a
deterministic music-theory layer realizes them — degrees → MIDI pitches in a
vocal range, snapped to the key's scale, laid into each section's time window,
with cadences at phrase ends. If the LLM is unavailable or unparseable, an
algorithmic composer guarantees output. Outputs a score (notes with absolute
times + syllables) that the synthesis engines and the piano-roll UI consume.
"""
import io
import random
import re

import pyphen

from . import llm as llm_mod

_DIC = pyphen.Pyphen(lang="en_US")

NOTE_PC = {"C": 0, "C#": 1, "D": 2, "Eb": 3, "E": 4, "F": 5,
           "F#": 6, "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11}
MAJOR = [0, 2, 4, 5, 7, 9, 11]
MINOR = [0, 2, 3, 5, 7, 8, 10]  # natural minor

VLO, VHI = 52, 79          # singable MIDI range (E3 .. G5)
ROOT_BASE = 55             # place the tonic near G3 so verses sit low, choruses lift

# Per section-role melodic character: degree range the composer draws from
# (higher = more emotional lift) and whether the phrase should peak.
ROLE_DEG = {
    "Verse": (1, 5), "Pre-Chorus": (3, 8), "Chorus": (3, 10),
    "Bridge": (2, 7), "Intro": (1, 5), "Solo": (1, 8),
    "Breakdown": (1, 6), "Outro": (1, 5),
}
STABLE = {1, 3, 5, 8, 10, 12}  # scale degrees that make restful phrase endings


def parse_key(key: str):
    parts = (key or "E minor").strip().rsplit(" ", 1)
    name = parts[0] if parts else "E"
    mode = parts[1].lower() if len(parts) > 1 else "minor"
    root_pc = NOTE_PC.get(name, 4)
    intervals = MAJOR if mode.startswith("maj") else MINOR
    return root_pc, intervals, mode


def syllabify_words(line: str):
    """Split a lyric line into words, each with its singable syllables:
    [(word, [syl, ...]), ...]. Word grouping is kept so downstream engines
    (e.g. SoulX) can phonemize whole words."""
    out = []
    for word in re.findall(r"[A-Za-z']+", line):
        parts = [p for p in _DIC.inserted(word).split("-") if p]
        out.append((word, parts or [word]))
    return out


def syllabify(line: str):
    """Flat list of syllables (one note each)."""
    return [s for _, syls in syllabify_words(line) for s in syls]


def degree_to_midi(deg: int, root_pc: int, intervals):
    root0 = ROOT_BASE + ((root_pc - ROOT_BASE) % 12)
    octave, idx = divmod(deg - 1, 7)
    pitch = root0 + 12 * octave + intervals[idx]
    while pitch > VHI:
        pitch -= 12
    while pitch < VLO:
        pitch += 12
    return pitch


def _snap_cadence(deg: int):
    """Pull a phrase-ending degree to the nearest restful scale degree."""
    if deg in STABLE:
        return deg
    return min(STABLE, key=lambda s: abs(s - deg))


# ---------------- composer brains ----------------
def _algorithmic_degrees(role: str, syl_lines):
    """Deterministic fallback: stepwise random walk within the role's range,
    ending each line on a stable (restful) degree."""
    lo, hi = ROLE_DEG.get(role, (1, 6))
    out = []
    for syls in syl_lines:
        n = len(syls)
        line = []
        cur = random.randint(lo, min(hi, lo + 2))
        for j in range(n):
            line.append(cur)
            step = random.choice([-2, -1, -1, 1, 1, 2])
            cur = max(lo, min(hi, cur + step))
        if line:
            line[-1] = _snap_cadence(max(lo, min(hi, line[-1])))
        out.append(line)
    return out


def _llm_degrees(role, syl_lines, key, provider, model, claude_model):
    """Ask the LLM for a melody as scale degrees (one integer per syllable).
    In-key by construction; cadence/range repaired by the theory layer."""
    lo, hi = ROLE_DEG.get(role, (1, 6))
    numbered = "\n".join(f"L{i+1} ({len(s)} syllables): {' '.join(s)}"
                         for i, s in enumerate(syl_lines))
    system = (
        "You compose memorable METAL vocal melodies as scale degrees. "
        "1 = the key's tonic, 2..7 ascend the scale, 8 = the octave; you may use "
        "1..12 and dip to -2. Output ONE line per lyric line: space-separated "
        "integers, EXACTLY one per syllable, no words, no commentary."
    )
    prompt = (
        f"Key: {key}. Section: {role} (use roughly degrees {lo}-{hi}; "
        f"{'soar to peaks and land on a strong hook' if role == 'Chorus' else 'mostly stepwise and singable'}). "
        f"End each line on a restful degree (1, 3 or 5).\n\nLyric lines:\n{numbered}\n\n"
        f"Return exactly {len(syl_lines)} line(s) of integers."
    )
    text = llm_mod.complete(provider, model, system, prompt, claude_model)
    text = re.sub(r"```[a-z]*", "", text)
    rows = [ln for ln in text.splitlines() if re.search(r"-?\d", ln)]
    out = []
    for i, syls in enumerate(syl_lines):
        nums = [int(x) for x in re.findall(r"-?\d+", rows[i])] if i < len(rows) else []
        if not nums:
            raise ValueError("empty degree line")
        # align to syllable count (pad by repeating last, or truncate)
        while len(nums) < len(syls):
            nums.append(nums[-1])
        nums = nums[:len(syls)]
        nums[-1] = _snap_cadence(nums[-1])
        out.append(nums)
    return out


def compose_degrees(role, syl_lines, key, provider, model, claude_model):
    if not syl_lines:
        return []
    try:
        return _llm_degrees(role, syl_lines, key, provider, model, claude_model)
    except Exception:
        return _algorithmic_degrees(role, syl_lines)


# ---------------- layout ----------------
def _lay_notes(syl_lines, word_lines, degrees, start, secs, root_pc, intervals, sect_idx, wcount):
    """Place each line's syllables into an equal slice of the section window,
    each line a phrase with a held final note and a short breath gap.
    `word_lines` (parallel to syl_lines) tags each syllable with its (word, idx)
    so per-word engines can regroup; `wcount` is a 1-element list = running
    global word counter."""
    notes = []
    if not syl_lines:
        return notes
    line_win = secs / len(syl_lines)
    lead = min(0.2, line_win * 0.12)      # small attack lead-in so a singing
    for li, syls in enumerate(syl_lines):  # engine doesn't clip the first onset
        if not syls:
            continue
        line_start = start + li * line_win + lead
        sing = (line_win - lead) * 0.9    # leave a breath at the line end
        n = len(syls)
        step = sing / n
        degs = degrees[li] if li < len(degrees) else [1] * n
        wtags = word_lines[li] if li < len(word_lines) else [(0, "")] * n
        local_to_global = {}
        for si, syl in enumerate(syls):
            deg = degs[si] if si < len(degs) else degs[-1]
            dur = step * (1.6 if si == n - 1 else 0.95)  # hold the last syllable
            local_wi, word = wtags[si] if si < len(wtags) else (0, "")
            if local_wi not in local_to_global:
                local_to_global[local_wi] = wcount[0]
                wcount[0] += 1
            notes.append({
                "midi": degree_to_midi(int(deg), root_pc, intervals),
                "start": round(line_start + si * step, 4),
                "dur": round(dur, 4),
                "syllable": syl,
                "word": word,
                "word_idx": local_to_global[local_wi],
                "section": sect_idx,
            })
    return notes


def compose(song: dict, provider="", model="", claude_model="claude-3-5-sonnet-latest", seed=None):
    """Compose a melody for a Song Constructor song.
    `song` = {bpm, key, blocks:[{type, seconds, lyrics}]}. Sections without
    lyrics (intros/solos/etc.) become instrumental rests (no notes).
    `seed` makes the (algorithmic) melody reproducible for A/B tuning."""
    if seed is not None:
        random.seed(int(seed))
    bpm = int(song.get("bpm") or 120)
    key = song.get("key") or "E minor"
    root_pc, intervals, _ = parse_key(key)
    provider = provider or llm_mod.best_provider()

    sections, flat, cursor, wcount = [], [], 0.0, [0]
    for i, b in enumerate(song.get("blocks", [])):
        role = b.get("type", "Verse")
        secs = float(b.get("seconds") or 0)
        lyr = (b.get("lyrics") or "").strip()
        lines = [ln for ln in lyr.splitlines() if ln.strip()]
        sec = {"role": role, "start": round(cursor, 4), "seconds": secs,
               "lyrics": lyr, "notes": []}
        if lines and secs > 0:
            syl_lines, word_lines = [], []
            for ln in lines:
                words = syllabify_words(ln)
                syls = [s for _, syls in words for s in syls]
                if not syls:
                    continue
                tags = [(wi, w) for wi, (w, syls) in enumerate(words) for _ in syls]
                syl_lines.append(syls)
                word_lines.append(tags)
            if syl_lines:
                degrees = compose_degrees(role, syl_lines, key, provider, model, claude_model)
                sec["notes"] = _lay_notes(syl_lines, word_lines, degrees, cursor, secs,
                                          root_pc, intervals, i, wcount)
                flat.extend(sec["notes"])
        sections.append(sec)
        cursor += secs
    return {"bpm": bpm, "key": key, "duration": round(cursor, 4),
            "provider": provider, "sections": sections, "notes": flat}


# ---------------- exports ----------------
def to_midi(score: dict) -> bytes:
    import mido
    tpb = 480
    beat = 60.0 / max(1, score.get("bpm", 120))
    mid = mido.MidiFile(ticks_per_beat=tpb)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(score.get("bpm", 120)), time=0))
    tr.append(mido.Message("program_change", program=53, time=0))  # Voice Oohs
    events = []
    for n in score.get("notes", []):
        on = int(round(n["start"] / beat * tpb))
        off = int(round((n["start"] + n["dur"]) / beat * tpb))
        events.append((on, "note_on", n["midi"]))
        events.append((max(off, on + 1), "note_off", n["midi"]))
    events.sort(key=lambda e: (e[0], e[1] == "note_on"))
    last = 0
    for tick, kind, midi in events:
        tr.append(mido.Message(kind, note=midi, velocity=90 if kind == "note_on" else 0,
                               time=max(0, tick - last)))
        last = tick
    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def render_guide(score: dict, sr=44100) -> bytes:
    """Render the melody to a simple synthetic 'ah' guide vocal (harmonics +
    vibrato + envelope). Pitch-accurate so RVC can re-timbre it into a real
    voice. Mono WAV bytes."""
    import numpy as np
    import soundfile as sf

    total = float(score.get("duration", 0)) + 0.5
    buf = np.zeros(int(total * sr) + 1, dtype=np.float32)
    for n in score.get("notes", []):
        f = 440.0 * 2 ** ((n["midi"] - 69) / 12.0)
        dur = max(0.05, n["dur"])
        t = np.arange(int(dur * sr)) / sr
        vib = 1.0 + 0.006 * np.sin(2 * np.pi * 5.5 * t)        # gentle vibrato
        phase = 2 * np.pi * f * np.cumsum(vib) / sr
        wave = (np.sin(phase) + 0.5 * np.sin(2 * phase)
                + 0.25 * np.sin(3 * phase) + 0.12 * np.sin(4 * phase))
        env = np.ones_like(t)
        a = int(0.02 * sr); r = int(0.06 * sr)
        if a:
            env[:a] = np.linspace(0, 1, a)
        if r and len(env) > r:
            env[-r:] = np.linspace(1, 0, r)
        seg = (wave * env * 0.2).astype(np.float32)
        i0 = int(n["start"] * sr)
        i1 = min(len(buf), i0 + len(seg))
        buf[i0:i1] += seg[:i1 - i0]
    peak = float(np.max(np.abs(buf)) or 1.0)
    if peak > 0.97:
        buf *= 0.97 / peak
    out = io.BytesIO()
    sf.write(out, buf, sr, format="WAV", subtype="PCM_16")
    return out.getvalue()
