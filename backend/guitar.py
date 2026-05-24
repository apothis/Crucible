"""MIDI → clean guitar DI rendering (the symbolic-route render step).

Renders a MIDI guitar part to a CLEAN direct-input (DI) signal that downstream
amp-sims (Helix Native, NAM) can legitimately process — the missing piece that
prompting ACE-Step / de-amping couldn't give us (RESEARCH.md §10g/§10h).

Default renderer = Karplus-Strong plucked string (pure numpy/scipy, no install,
no soundfont) — the classic guitar synthesis method; gives a real clean plucked
DI. Pluggable: a fluidsynth+soundfont or a VST instrument (Shreddage 3 Stratus
FREE via pedalboard) can slot in later for higher fidelity.
"""
import io


def _midi_to_freq(m):
    return 440.0 * (2.0 ** ((m - 69) / 12.0))


def karplus_strong(freq, dur_s, sr=44100, decay=0.996, brightness=0.5, pick=0.5, seed=None):
    """One plucked-string note (Extended Karplus-Strong). `brightness` (0..1)
    weights the loop lowpass (bright→darker), `pick` adds attack transient."""
    import numpy as np
    from scipy.signal import lfilter
    n = max(2, int(round(sr / freq)))
    total = max(n + 1, int(dur_s * sr))
    rng = np.random.default_rng(seed)
    # excitation: noise lowpassed toward darker for low brightness, + a sharp
    # pick transient at the onset for string/pick definition.
    exc = rng.uniform(-1.0, 1.0, n)
    lp = float(np.clip(0.15 + 0.7 * (1.0 - brightness), 0.1, 0.95))   # one-pole LP
    exc = lfilter([1 - lp], [1.0, -lp], exc)
    x = np.zeros(total, dtype="float64")
    x[:n] = exc
    x[0] += pick * 0.8                            # pick attack click
    # loop: y[k] = x[k] + decay*((1-s)*y[k-n] + s*y[k-n-1]); s controls damping
    s = 0.5 * (1.0 - float(np.clip(brightness, 0.0, 1.0)))
    a = np.zeros(n + 2); a[0] = 1.0
    a[n] = -decay * (1.0 - s); a[n + 1] = -decay * s
    y = lfilter([1.0], a, x)
    env = np.ones(total)
    atk = min(int(0.003 * sr), total)
    env[:atk] = np.linspace(0, 1, atk)
    rel = min(int(0.02 * sr), total // 4)         # short release to avoid clicks
    if rel > 0:
        env[-rel:] = np.linspace(1, 0, rel)
    return (y * env).astype("float32")


def midi_to_notes(midi_path):
    """Parse a MIDI file → [(midi_pitch, start_s, dur_s, velocity)]."""
    import mido
    mid = mido.MidiFile(midi_path)
    notes, on = [], {}
    t = 0.0
    for msg in mid:                              # mido iterates in real seconds
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            on[msg.note] = (t, msg.velocity)
        elif msg.type in ("note_off",) or (msg.type == "note_on" and msg.velocity == 0):
            if msg.note in on:
                st, vel = on.pop(msg.note)
                notes.append((msg.note, st, max(0.05, t - st), vel))
    return notes


def notes_to_midi(notes, path, program=0, tpqn=480):
    """Write notes [(pitch,start_s,dur_s,vel)] to a Standard MIDI File (fixed
    120bpm grid; seconds → ticks)."""
    import mido
    mid = mido.MidiFile(ticks_per_beat=tpqn)
    tr = mido.MidiTrack(); mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=500000))      # 120 bpm
    tr.append(mido.Message("program_change", program=int(program), time=0))
    ev = []
    for pitch, st, dur, vel in notes:
        p = int(max(0, min(127, pitch)))
        ev.append((st, 1, p, int(max(1, min(127, vel)))))       # note_on
        ev.append((st + max(0.03, dur), 0, p, 0))               # note_off
    ev.sort(key=lambda e: (e[0], e[1]))                          # offs before ons at same t
    tk = lambda s: int(round(s * tpqn * 2))                      # 120bpm: 960 ticks/sec
    last = 0
    for t, on, p, v in ev:
        now = tk(t); dt = max(0, now - last); last = now
        tr.append(mido.Message("note_on" if on else "note_off", note=p, velocity=v, time=dt))
    mid.save(path)
    return path


def render_di_soundfont(notes, sf2_path, sr=44100, program=0):
    """Higher-fidelity DI: render notes via fluidsynth + a SoundFont (CLI through
    midi2audio — robust on macOS vs the pyfluidsynth dylib). Returns (frames,2)."""
    import os
    import shutil
    import subprocess
    import tempfile
    import numpy as np
    import soundfile as sf
    d = tempfile.mkdtemp()
    mp, wp = os.path.join(d, "r.mid"), os.path.join(d, "r.wav")
    notes_to_midi(notes, mp, program=program)
    fs = shutil.which("fluidsynth") or "/opt/homebrew/bin/fluidsynth"
    # options BEFORE positional soundfont/midi (fluidsynth 2.5 is strict); -F = fast
    # file render (auto-disables the audio driver), -ni = no shell / no midi driver.
    r = subprocess.run([fs, "-ni", "-g", "1.0", "-F", wp, "-r", str(sr), sf2_path, mp],
                       capture_output=True, text=True)
    if not os.path.exists(wp):
        raise RuntimeError("fluidsynth render failed: " + (r.stderr or r.stdout)[-300:])
    data, _ = sf.read(wp, dtype="float32", always_2d=True)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 0.99:
        data = data * (0.99 / peak)
    return data


_KONTAKT = {}  # cache the (slow ~25s to load) Kontakt plugin instance by path


def render_di_kontakt(notes, kontakt_path, state_path, sr=44100):
    """Render notes through Kontakt (e.g. Shreddage) → sampled DI. Requires a
    captured Kontakt `raw_state` that already has the instrument loaded (set up
    once via its editor). Instance is cached (Kontakt is slow to load)."""
    import os
    import mido
    import numpy as np
    import pedalboard as pb
    k = _KONTAKT.get(kontakt_path)
    if k is None:
        k = pb.load_plugin(kontakt_path)
        _KONTAKT[kontakt_path] = k
    if state_path and os.path.exists(state_path):
        with open(state_path, "rb") as f:
            k.raw_state = f.read()
    msgs = []
    for pitch, st, dur, vel in notes:
        p = int(max(0, min(127, pitch)))
        msgs.append(mido.Message("note_on", note=p, velocity=int(max(1, min(127, vel))), time=float(st)))
        msgs.append(mido.Message("note_off", note=p, velocity=0, time=float(st + max(0.05, dur))))
    msgs.sort(key=lambda m: m.time)
    total = (max(m.time for m in msgs) + 0.5) if msgs else 1.0
    audio = np.asarray(k(msgs, duration=total, sample_rate=sr))   # (channels, samples)
    if audio.ndim == 2 and audio.shape[0] <= 2:
        audio = audio.T                                            # → (samples, channels)
    elif audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.99:
        audio = audio * (0.99 / peak)
    return audio.astype("float32")


def render_di_file(notes, path, sr=44100, decay=0.996, engine="ks",
                   sf2_path=None, program=0, kontakt_path=None, kontakt_state=None):
    """Render notes to a clean DI WAV. engine 'ks' = Karplus-Strong (default,
    no deps); 'soundfont' = fluidsynth + sf2; 'kontakt' = Kontakt/Shreddage."""
    import soundfile as sf
    if engine == "kontakt" and kontakt_path and kontakt_state:
        di = render_di_kontakt(notes, kontakt_path, kontakt_state, sr=sr)
    elif engine == "soundfont" and sf2_path:
        di = render_di_soundfont(notes, sf2_path, sr=sr, program=program)
    else:
        di = render_di(notes, sr=sr, decay=decay)
    sf.write(path, di, sr, subtype="PCM_16")
    return path


def render_di(notes, sr=44100, decay=0.996, seed=0, humanize=True):
    """Render a list of (pitch, start_s, dur_s, velocity) notes to a stereo DI.
    Velocity drives brightness/pick; slight per-note detune + timing jitter keep
    stacked chords/repeats from phase-locking (more natural)."""
    import numpy as np
    if not notes:
        raise ValueError("no notes to render")
    rng = np.random.default_rng(seed)
    end = max(st + d for _, st, d, _ in notes) + 0.4
    out = np.zeros(int(end * sr) + 1, dtype="float32")
    for pitch, st, dur, vel in notes:
        v = vel / 127.0
        cents = rng.uniform(-6, 6) if humanize else 0.0          # slight detune
        freq = _midi_to_freq(pitch) * (2.0 ** (cents / 1200.0))
        bright = float(np.clip(0.35 + 0.5 * v, 0.1, 0.95))       # louder = brighter
        note = karplus_strong(freq, dur + 0.3, sr=sr, decay=decay,
                              brightness=bright, pick=0.4 + 0.4 * v,
                              seed=int(rng.integers(1, 2**31)))
        note = note * v
        jit = rng.uniform(-0.003, 0.003) if humanize else 0.0    # ±3ms timing
        s = max(0, int((st + jit) * sr))
        e = min(len(out), s + len(note))
        out[s:e] += note[:e - s]
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.99:
        out = out * (0.99 / peak)
    return np.stack([out, out], axis=1)          # (frames, 2)


# ---- in-key test riff (until the symbolic generator is wired) ----
_MAJOR = [0, 2, 4, 5, 7, 9, 11]
_MINOR = [0, 2, 3, 5, 7, 8, 10]
_ROOTS = {"C": 0, "C#": 1, "D": 2, "Eb": 3, "E": 4, "F": 5, "F#": 6,
          "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11}


RIFF_STYLES = ("chug", "gallop", "powerchords", "pedal")
# i–VI–VII–i style degree roots (scale-index, 0-based) for a minor metal feel
_PROG = {"minor": [0, 5, 6, 0], "major": [0, 3, 4, 0]}

# Scales the LLM riff can realize degrees against (genre-dependent modal flavour).
SCALES = {
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "major": [0, 2, 4, 5, 7, 9, 11],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],          # dark/thrash/death (b2)
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],     # neoclassical/black (maj7)
    "dorian": [0, 2, 3, 5, 7, 9, 10],             # folk/prog (nat6)
    "pentatonic_minor": [0, 3, 5, 7, 10],         # hard/southern rock, AC/DC (5-note)
    "blues": [0, 3, 5, 6, 7, 10],                 # bluesy southern rock (b5)
    "phrygian_dominant": [0, 1, 4, 5, 7, 8, 10],   # 5th mode of harmonic minor (neoclassical/exotic)
}

# Genre definitions live in the unified registry (backend/genres.py) — the single
# source of truth shared with the generation chips. Keyed by id for riff/solo use.
from .genres import BY_ID as RIFF_GENRES  # noqa: E402


def _powerchord(root, vel, t, dur, drop=True):
    """Root + fifth (+ octave) — the metal power chord."""
    ns = [(root, t, dur, vel), (root + 7, t, dur, int(vel * 0.9))]
    if drop:
        ns.append((root + 12, t, dur, int(vel * 0.7)))
    return ns


def generate_riff(key="E minor", bpm=160, duration_s=None, bars=8, style="gallop", seed=None):
    """Music-theory metal riff generator (in-key, low register) → notes
    [(pitch,start,dur,vel)]. Styles: chug (palm-mute root chugs), gallop
    (1/8+1/16+1/16 on root w/ scale accents), powerchords (held chords following
    a i–VI–VII progression), pedal (low root pedal alternating with scale tones).
    Fills `duration_s` if given, else `bars` bars (4/4)."""
    import numpy as np
    parts = key.split()
    root_pc = _ROOTS.get(parts[0], 4)
    is_minor = not (len(parts) > 1 and parts[1].lower().startswith("maj"))
    scale = _MINOR if is_minor else _MAJOR
    base = 28 + root_pc                            # low/drop register (~E1–E2)
    prog = _PROG["minor" if is_minor else "major"]
    spb = 60.0 / float(bpm)
    bar = spb * 4.0
    eighth, sixt = spb / 2.0, spb / 4.0
    total = float(duration_s) if duration_s else bars * bar
    rng = np.random.default_rng(seed)
    notes, t, b = [], 0.0, 0

    def chord_root(bar_idx):
        deg = prog[bar_idx % len(prog)]
        return base + scale[deg % len(scale)]

    while t < total - 1e-3:
        r = chord_root(b)
        if style == "powerchords":
            dur = bar * 0.98
            notes += _powerchord(r, 112, t, min(dur, total - t))
            t += bar
        elif style == "chug":
            steps = int(round(bar / sixt))
            for i in range(steps):
                if t >= total:
                    break
                # mostly muted root chugs; occasional scale accent
                acc = rng.random() < 0.18
                p = r + (scale[int(rng.integers(1, len(scale)))] if acc else 0)
                notes += _powerchord(p, 118 if acc else 100, t, sixt * 0.85, drop=False)
                t += sixt
        elif style == "pedal":
            steps = int(round(bar / eighth))
            for i in range(steps):
                if t >= total:
                    break
                if i % 2 == 0:                     # low pedal root
                    notes += _powerchord(r, 110, t, eighth * 0.9, drop=False)
                else:                              # alternate scale tone up top
                    deg = scale[int(rng.integers(2, len(scale)))]
                    notes.append((base + 12 + deg, t, eighth * 0.8, 96))
                t += eighth
        else:  # gallop: 1/8 + 1/16 + 1/16 per beat
            for _beat in range(4):
                if t >= total:
                    break
                notes += _powerchord(r, 112, t, eighth * 0.8, drop=False)
                t += eighth
                for _s in range(2):
                    if t >= total:
                        break
                    notes += _powerchord(r, 104, t, sixt * 0.8, drop=False)
                    t += sixt
        b += 1
    return notes


# Per-section riff profile: section role → (style, velocity scale, octave shift).
# Drives dynamics across a Song Constructor arrangement (verse chuggy, chorus big,
# breakdown low+heavy, intro/outro sparse).
SECTION_PROFILE = {
    "intro":      ("powerchords", 0.70, 0),
    "verse":      ("chug",        0.85, 0),
    "prechorus":  ("gallop",      0.95, 0),
    "chorus":     ("powerchords", 1.10, 0),
    "bridge":     ("pedal",       0.90, 0),
    "solo":       ("powerchords", 0.80, 0),   # sustained rhythm bed under a lead
    "breakdown":  ("chug",        1.05, -12),  # drop an octave, heavy
    "outro":      ("powerchords", 0.70, 0),
}


def _degree_to_pitch(v, base, scale):
    """Scale degree (1=root, then up the scale; one octave past the top wraps) →
    MIDI in the low register. Handles 5-note (pentatonic) and 7-note scales."""
    n = len(scale)
    octave, idx = divmod(int(v) - 1, n)
    return base + 12 * octave + scale[idx % n]


def llm_riff(key="E minor", bpm=160, duration_s=None, bars=8, style="gallop", genre="",
             part="riff", provider="", model="", claude_model="claude-3-5-sonnet-latest", seed=None):
    """LLM-guided riff OR solo. The model writes integers on a 16th grid
    (0=rest, 1=tonic, higher=up the scale). part='riff' → a repeatable 2-bar
    pattern realized as power chords (or single notes if the genre is a lead);
    part='solo' → a longer through-composed SINGLE-NOTE lead line in a high
    register, using the genre's solo style. Raises on failure → caller falls back."""
    import re
    from . import llm as llm_mod
    parts = key.split()
    root_pc = _ROOTS.get(parts[0], 4)
    is_minor = not (len(parts) > 1 and parts[1].lower().startswith("maj"))
    g = RIFF_GENRES.get(genre or "")
    scale = SCALES.get(g["scale"]) if g else (_MINOR if is_minor else _MAJOR)
    solo = (part == "solo")
    lead = solo or bool(g and g.get("lead"))        # single-note line (shred/solo) vs power chords
    top = len(scale) + 1                            # octave degree number for this scale
    reg = (g.get("reg", 0) if g else 0)
    if solo:
        reg = max(reg, 24)                          # solos sit in a high lead register
    base = 28 + root_pc + reg
    spb = 60.0 / float(bpm)
    sixt = spb / 4.0
    total = float(duration_s) if duration_s else bars * spb * 4
    nsteps = max(8, int(round(total / sixt)))
    pat_len = 64 if solo else 32                    # solos = longer phrase, less repetition
    provider = provider or llm_mod.best_provider()
    if solo:
        system = (
            "You compose expressive virtuoso lead GUITAR SOLOS on a sixteenth-note grid as integers. "
            f"0 = rest (phrasing space). 1 = the tonic; higher integers ascend the scale ({top} = an "
            "octave up; go to 12-15 for high climaxes). Write flowing SINGLE-NOTE phrases — runs, "
            "leaps, pedal points and held peaks — that build and breathe (use some rests between "
            "phrases). Output ONLY space-separated integers — no words, no commentary."
        )
    elif lead:
        system = (
            "You compose virtuosic SHRED lead lines on a sixteenth-note grid as integers. "
            f"0 = rest. 1 = the tonic; higher integers ascend the scale ({top} = one octave up, "
            "and you may go to 12-15 for high runs). Write FAST, mostly-continuous SINGLE-NOTE "
            "scalar runs and pedal-point lines with very few rests. "
            "Output ONLY space-separated integers — no words, no commentary."
        )
    else:
        system = (
            "You compose heavy guitar riffs on a sixteenth-note grid as integers. "
            f"0 = rest (silence/gap). 1 = low root chug (palm-muted power chord on the tonic). "
            f"2-{len(scale)} = scale degrees above the root, {top} = octave. Real riffs are TIGHT and "
            "REPETITIVE: a root-anchored pattern with rests for groove and a few melodic moves. "
            "Output ONLY space-separated integers — no words, no commentary."
        )
    feel = f"{g['label']} — {g['solo' if solo else 'riff']}" if g else (
        style + ": " + ("galloping root chugs with accents" if style == "gallop"
        else "driving palm-muted chugs" if style == "chug"
        else "sustained power chords that change every bar or two" if style == "powerchords"
        else "a low root pedal alternating with scale tones"))
    bars_word = f"{pat_len // 16}-BAR" + (" solo phrase" if solo else " riff")
    prompt = (f"Key root: {key.split()[0]}. Style: {feel}. Write a {bars_word} = EXACTLY {pat_len} "
              f"integers on a 16th-note grid that captures that style." +
              ("" if solo else " Make it a memorable, repeatable pattern."))
    text = llm_mod.complete(provider, model, system, prompt, claude_model)
    vals = [int(x) for x in re.findall(r"-?\d+", re.sub(r"```[a-z]*", "", text))]
    vals = [v if v >= 0 else 0 for v in vals]      # treat negatives as rests
    if not any(v > 0 for v in vals):
        raise ValueError("no playable steps from LLM")
    pattern = vals[:pat_len] if len(vals) >= pat_len else vals
    tiled = (pattern * ((nsteps // len(pattern)) + 1))[:nsteps]
    notes = []
    for i, v in enumerate(tiled):
        if v <= 0:
            continue
        t = i * sixt
        if t >= total:
            break
        pitch = _degree_to_pitch(v, base, scale)
        vel = 116 if v != 1 else 100
        if lead:
            notes.append((pitch, t, sixt * 0.95, vel))          # single-note shred line
        else:
            notes += _powerchord(pitch, vel, t, sixt * 0.9, drop=False)
    if not notes:
        raise ValueError("LLM riff produced no notes")
    return notes


def _algorithmic_solo(key, bpm, duration_s, genre="", seed=None):
    """Deterministic fallback lead: a meandering single-note scalar line in a
    high register (used if the LLM is unavailable for a solo)."""
    import numpy as np
    parts = key.split()
    root_pc = _ROOTS.get(parts[0], 4)
    g = RIFF_GENRES.get(genre or "")
    scale = SCALES.get(g["scale"]) if g else _MINOR
    base = 28 + root_pc + 24
    spb = 60.0 / float(bpm); sixt = spb / 4.0
    n = max(8, int(round(float(duration_s or 8) / sixt)))
    rng = np.random.default_rng(seed)
    notes, deg = [], 1
    for i in range(n):
        if rng.random() < 0.15:                     # occasional phrasing rest
            deg = max(1, deg)
        else:
            notes.append((_degree_to_pitch(deg, base, scale), i * sixt, sixt * 0.95, 108))
        deg = max(1, min(len(scale) * 2, deg + int(rng.integers(-2, 3))))
    return notes or [(_degree_to_pitch(1, base, scale), 0, sixt, 100)]


def compose_riff(brain, key="E minor", bpm=160, duration_s=None, bars=8, style="gallop",
                 genre="", part="riff", provider="", model="", seed=None):
    """Pick the riff brain: 'algorithmic' (default, instant) or LLM-guided
    (provider set by the caller). `genre` (LLM only) sets feel + modal scale;
    `part` = 'riff' or 'solo' (lead line). LLM failures fall back to algorithmic."""
    if brain and brain != "algorithmic":
        try:
            return llm_riff(key, bpm, duration_s=duration_s, bars=bars, style=style,
                            genre=genre, part=part, provider=provider, model=model, seed=seed)
        except Exception:
            pass
    if part == "solo":
        return _algorithmic_solo(key, bpm, duration_s, genre=genre, seed=seed)
    return generate_riff(key, bpm, duration_s=duration_s, bars=bars, style=style, seed=seed)


def generate_riff_arrangement(blocks, key="E minor", bpm=160, seed=None,
                              brain="algorithmic", provider="", model="", genre=""):
    """Generate a full-song riff that varies per section (Song Constructor).
    `blocks` = [{type, seconds}]. Each section gets a style/intensity/register
    from SECTION_PROFILE; segments are laid end-to-end. `brain` chooses the
    algorithmic or LLM-guided generator per section."""
    notes, t0 = [], 0.0
    for i, b in enumerate(blocks or []):
        secs = float(b.get("seconds", 8) or 8)
        typ = str(b.get("type", "verse")).lower().replace(" ", "").replace("-", "")
        style, vscale, octv = SECTION_PROFILE.get(typ, ("gallop", 1.0, 0))
        seg = compose_riff(brain, key, bpm, duration_s=secs, style=style, genre=genre,
                           provider=provider, model=model,
                           seed=(None if seed is None else seed + i))
        for (p, st, d, v) in seg:
            notes.append((p + octv, t0 + st, d, max(1, min(127, int(v * vscale)))))
        t0 += secs
    if not notes:
        raise ValueError("no sections to generate from")
    return notes


def test_riff(key="E minor", bpm=160, bars=4, seed=None):
    """Back-compat shim → the gallop generator."""
    return generate_riff(key=key, bpm=bpm, bars=bars, style="gallop", seed=seed)
