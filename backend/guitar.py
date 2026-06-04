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


# Notes are (pitch, start_s, dur_s, velocity) tuples with an OPTIONAL 5th element:
# an articulation code ('b' bend, 's' slide, 'v' vibrato, 'h' hammer/legato,
# '~' let-ring, '.' staccato, '' none). Most consumers only need the first four;
# only the Karplus-Strong DI render realizes the articulation.
def _artic(n):
    return n[4] if len(n) > 4 else ""


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
    for n in notes:
        pitch, st, dur, vel = n[0], n[1], n[2], n[3]
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


def render_di_kontakt(notes, kontakt_path, state_path, sr=44100):
    """Render notes through Kontakt (e.g. Shreddage) → sampled DI. Requires a
    captured Kontakt `raw_state` that already has the instrument loaded (set up
    once via its editor).

    Kontakt's VST3 can only be loaded/rendered on a process's MAIN thread, but
    FastAPI serves us on a worker threadpool -- so we delegate to a persistent
    daemon subprocess (backend/kontakt_daemon.py) that owns the plugin on its own
    main thread. The daemon writes the WAV; we read it back. See that module's
    docstring for the why. Returns (frames, 2) float32."""
    import os
    import tempfile
    import numpy as np
    import soundfile as sf
    from . import kontakt_daemon
    d = tempfile.mkdtemp()
    wp = os.path.join(d, "kontakt_di.wav")
    try:
        kontakt_daemon.render(notes, wp, kontakt_path, state_path, sr=sr)
        audio, _ = sf.read(wp, dtype="float32", always_2d=True)
    finally:
        try:
            os.remove(wp)
            os.rmdir(d)
        except OSError:
            pass
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
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


def _pitch_mod(sig, sr, semitone_env):
    """Re-pitch a note by a per-sample semitone envelope (phase-accumulation
    resample). Realizes bends/slides/vibrato on the synthesized Karplus note.
    Positive = up. Output length follows the warped read position."""
    import numpy as np
    n = len(sig)
    if n < 8:
        return sig
    ratio = 2.0 ** (np.asarray(semitone_env, dtype="float64") / 12.0)
    read = np.cumsum(ratio) - ratio[0]                  # read position into sig per output sample
    read = read[read < (n - 1)]
    return np.interp(read, np.arange(n), sig).astype("float32")


def _artic_env(artic, length, sr, prev_semitones=0.0):
    """Per-sample semitone offset for an articulation, or None for a plain pick.
    bend = whole-step rise into the target; slide = glide from the previous pitch;
    vibrato = delayed pitch shake; others = no pitch motion."""
    import numpy as np
    if length < 8:
        return None
    t = np.arange(length) / sr
    if artic == "b":                                    # bend up a whole step into pitch
        gl = max(1, int(length * 0.35))
        env = np.zeros(length); env[:gl] = np.linspace(-2.0, 0.0, gl)
        return env
    if artic == "s" and abs(prev_semitones) > 0.1:      # slide from the previous note's pitch
        gl = max(1, int(length * 0.30))
        env = np.zeros(length); env[:gl] = np.linspace(prev_semitones, 0.0, gl)
        return env
    if artic == "v":                                    # vibrato: shake after a short onset delay
        onset = min(int(0.08 * sr), length // 3)
        depth = 0.28                                    # ~28 cents
        env = depth * np.sin(2 * np.pi * 5.6 * t)
        env[:onset] *= np.linspace(0, 1, onset) if onset else 1.0
        return env
    return None


def render_di(notes, sr=44100, decay=0.996, seed=0, humanize=True):
    """Render (pitch, start_s, dur_s, velocity[, articulation]) notes to a stereo DI.
    Velocity drives brightness/pick; slight per-note detune + timing jitter keep
    stacked chords/repeats from phase-locking. Articulations (5th field) are
    realized: bends/slides/vibrato via pitch modulation, hammer/legato softens the
    pick attack, let-ring extends sustain, staccato is already shortened upstream."""
    import numpy as np
    if not notes:
        raise ValueError("no notes to render")
    rng = np.random.default_rng(seed)
    end = max(n[1] + n[2] for n in notes) + 0.4
    out = np.zeros(int(end * sr) + 1, dtype="float32")
    prev_pitch = None
    for n in notes:
        pitch, st, dur, vel = n[0], n[1], n[2], n[3]
        art = _artic(n)
        v = vel / 127.0
        cents = rng.uniform(-6, 6) if humanize else 0.0          # slight detune
        freq = _midi_to_freq(pitch) * (2.0 ** (cents / 1200.0))
        bright = float(np.clip(0.35 + 0.5 * v, 0.1, 0.95))       # louder = brighter
        pick = 0.4 + 0.4 * v
        ndecay = decay
        if art == "h":                                           # hammer-on/legato: soft, no hard pick
            pick *= 0.25; bright = float(np.clip(bright - 0.1, 0.1, 0.95))
        elif art == "~":                                         # let ring: longer sustain
            ndecay = min(0.9992, decay + 0.003)
        tail = 0.6 if art == "~" else 0.3
        note = karplus_strong(freq, dur + tail, sr=sr, decay=ndecay,
                              brightness=bright, pick=pick,
                              seed=int(rng.integers(1, 2**31)))
        prev_semi = (prev_pitch - pitch) if (prev_pitch is not None) else 0.0
        env = _artic_env(art, len(note), sr, prev_semitones=prev_semi)
        if env is not None:
            note = _pitch_mod(note, sr, env)
        note = note * v
        jit = rng.uniform(-0.003, 0.003) if humanize else 0.0    # ±3ms timing
        s = max(0, int((st + jit) * sr))
        e = min(len(out), s + len(note))
        out[s:e] += note[:e - s]
        prev_pitch = pitch
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
    "solo":       ("powerchords", 0.80, 0),   # lead line over a quieter bed (see generate_riff_arrangement)
    "breakdown":  ("chug",        1.05, -12),  # drop an octave, heavy
    "outro":      ("powerchords", 0.70, 0),
}


def _degree_to_pitch(v, base, scale):
    """Scale degree (1=root, then up the scale; one octave past the top wraps) →
    MIDI in the low register. Handles 5-note (pentatonic) and 7-note scales."""
    n = len(scale)
    octave, idx = divmod(int(v) - 1, n)
    return base + 12 * octave + scale[idx % n]


def _parse_note_dsl(text, max_deg=15):
    """Parse the LLM's note-token DSL into events [(degree, dur16, artic)]. A token is
    DEGREE[:DURATION][artic] where the articulation may be appended to the duration
    ('8:4v') or colon-separated ('8:4:v'); bare integers (legacy) are accepted as 16ths.
    DEGREE 0 = rest. DURATION in sixteenth-notes (default 1). ARTIC in b/s/v/h/~/."""
    import re
    text = re.sub(r"```[a-z]*", "", text or "")
    events = []
    for tok in re.split(r"[\s,]+", text.strip()):
        m = re.match(r"^(-?\d+)(?::(\d+))?:?([bsvhBSVH~.]?)", tok)
        if not m:
            continue
        deg = min(max(int(m.group(1)), 0), max_deg)
        dur = max(1, min(16, int(m.group(2)))) if m.group(2) else 1
        artic = (m.group(3) or "").lower()
        if artic not in ("b", "s", "v", "h", "~", "."):
            artic = ""
        events.append((deg, dur, artic))
    return events


def _lay_events(events, base, scale, sixt, total, lead):
    """Lay parsed (degree, dur16, artic) events onto the timeline, repeating the
    phrase until `total` seconds are filled (motif return for solos / loop for
    riffs). lead=True → single articulated notes; else power chords. Returns note
    tuples ((pitch,start,dur,vel) or (pitch,start,dur,vel,artic) for lead notes)."""
    notes = []
    if not events:
        return notes
    t = 0.0
    guard = 0
    while t < total - 1e-3 and guard < 20000:
        for deg, dur16, artic in events:
            if t >= total - 1e-3:
                break
            seg = dur16 * sixt
            if deg > 0:
                pitch = _degree_to_pitch(deg, base, scale)
                if lead:
                    accent = artic in ("b", "v", "~") or dur16 >= 4
                    vel = 118 if accent else 104
                    gate = 1.2 if artic == "~" else (0.5 if artic == "." else 0.95)
                    notes.append((pitch, t, max(0.05, seg * gate), vel, artic))
                else:
                    vel = 116 if deg != 1 else 100
                    notes += _powerchord(pitch, vel, t, max(0.05, seg * 0.9), drop=False)
            t += seg
            guard += 1
    return notes


def llm_riff(key="E minor", bpm=160, duration_s=None, bars=8, style="gallop", genre="",
             part="riff", provider="", model="", claude_model="claude-3-5-sonnet-latest", seed=None,
             context=""):
    """LLM-guided riff OR solo via a compact note-token DSL: each token is
    DEGREE:DURATION:ARTICULATION (0=rest, 1=tonic ascending the scale; duration in
    16th units; articulation b/s/v/h/~/.). part='riff' → a repeatable pattern (power
    chords, or single notes for a lead genre); part='solo' → an expressive,
    motif-driven single-note lead in a high register. Raises on failure → fallback."""
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
    pat_len = 64 if solo else 32                    # target phrase length in 16th units
    provider = provider or llm_mod.best_provider()
    dsl = (
        "Output a sequence of NOTE TOKENS separated by spaces. Each token is "
        "DEGREE:DURATION:ARTICULATION.\n"
        f"- DEGREE: 0 = rest. 1 = the tonic; 2..{top} ascend the scale ({top} = one octave up). "
        "Numbers up to 15 reach higher octaves for climaxes.\n"
        "- DURATION in sixteenth-notes: 1=16th, 2=8th, 3=dotted-8th, 4=quarter, 6=dotted-quarter, "
        "8=half. Omit to mean a 16th.\n"
        "- ARTICULATION (optional): b=bend up into the note, s=slide from the previous note, "
        "v=vibrato (held, shaking), h=hammer-on/legato, ~=let it ring (sustain), .=staccato (short). "
        "Omit for a normal picked note.\n"
        "Write ONLY the tokens, nothing else."
    )
    if solo:
        system = (
            "You are a virtuoso lead guitarist composing an expressive SOLO. " + dsl +
            "\nPHRASING: think in short MOTIFS of 2-4 notes, then answer and vary them (call and "
            "response). Leave RESTS to breathe between phrases. Land longer notes on chord tones "
            "(degrees 1, 3, 5) on strong beats. Put bends and vibrato on held peak notes. Build the "
            "contour upward to a high climax, then resolve back down."
        )
        example = "1:2 3:1 5:1 8:4v 0:2 8:1 7:1b 5:2~ 3:1 1:1 3:1 5:1h 7:4v"
    elif lead:
        system = (
            "You compose virtuosic SHRED lead lines. " + dsl +
            "\nWrite fast, mostly-continuous scalar runs and pedal-point lines with few rests, "
            "occasional bends/vibrato on accents, resolving onto chord tones."
        )
        example = "1:1 2:1 3:1 5:1 7:1 8:1 7:1 5:1 8:2b 7:1 5:1 3:2v"
    else:
        system = (
            "You compose heavy, TIGHT guitar RIFFS. " + dsl +
            "\nReal riffs are repetitive and root-anchored: low chugs on the tonic (degree 1) with "
            "rests for groove and a few melodic moves. Favour short durations and a palm-muted feel."
        )
        example = "1:1 1:1 0:1 1:1 3:1 1:1 0:1 1:2 1:1 0:1 1:1 6:2"
    feel = f"{g['label']} — {g['solo' if solo else 'riff']}" if g else (
        style + ": " + ("galloping root chugs with accents" if style == "gallop"
        else "driving palm-muted chugs" if style == "chug"
        else "sustained power chords that change every bar or two" if style == "powerchords"
        else "a low root pedal alternating with scale tones"))
    bars_word = f"{pat_len // 16}-bar " + ("solo phrase" if solo else "riff")
    prompt = (f"Key root: {key.split()[0]}. Style: {feel}. Write a {bars_word} (roughly "
              f"{pat_len // 2} to {pat_len} tokens) that captures that style. "
              f"Example of the FORMAT only (do not copy these notes): {example}." +
              ("" if solo else " Make it a memorable, repeatable pattern."))
    if context:
        # Audio-grounded description of the actual section (from the ACE LM "ears").
        prompt += (f" The section it plays over actually sounds like this — match its energy, "
                   f"mood and instrumentation: {str(context).strip()[:600]}")
    text = llm_mod.complete(provider, model, system, prompt, claude_model)
    events = _parse_note_dsl(text, max_deg=15)
    if solo:
        phrase = events                              # through-composed; repeats only if short
    else:
        phrase, acc = [], 0                          # tight loop: cap the pattern length
        for ev in events:
            phrase.append(ev); acc += ev[1]
            if acc >= pat_len:
                break
    if not any(d > 0 for d, _, _ in phrase):
        raise ValueError("no playable steps from LLM")
    notes = _lay_events(phrase, base, scale, sixt, total, lead)
    if not notes:
        raise ValueError("LLM riff produced no notes")
    return notes


def _algorithmic_solo(key, bpm, duration_s, genre="", seed=None):
    """Deterministic fallback lead (used when the LLM is unavailable): a MOTIF-based
    single-note line. States a short melodic cell built from chord tones + passing
    notes, then repeats it with transposition/variation and rests (call and response),
    landing held notes with vibrato/bend/let-ring. Density shaped by tempo. Uses the
    genre's scale + register, and is realized through the same articulation path."""
    import numpy as np
    parts = key.split()
    root_pc = _ROOTS.get(parts[0], 4)
    g = RIFF_GENRES.get(genre or "")
    scale = SCALES.get(g["scale"]) if g else _MINOR
    reg = max((g.get("reg", 0) if g else 0), 24)         # solos sit in a high register
    base = 28 + root_pc + reg
    spb = 60.0 / float(bpm); sixt = spb / 4.0
    total = float(duration_s or 8)
    rng = np.random.default_rng(seed)
    busy = float(np.clip((bpm - 80) / 100.0, 0.0, 1.0))          # 0 slow/spacious .. 1 fast/busy
    span = len(scale) * 2
    chord_tones = [1, 3, 5]

    def make_motif():
        cell, deg = [], int(rng.choice(chord_tones))
        steps = 3 + int(rng.integers(0, 4))
        for k in range(steps):
            if k == steps - 1:                                   # land + hold the phrase end
                dur = int(rng.choice([4, 6])) if busy < 0.6 else 2
                artic = str(rng.choice(["v", "~", "b"]))
            else:
                dur = 1 if busy > 0.5 else int(rng.choice([1, 2, 2]))
                artic = "h" if (busy > 0.6 and rng.random() < 0.4) else ""
            cell.append((max(1, deg), dur, artic))
            deg = int(np.clip(deg + int(rng.choice([-2, -1, 1, 1, 2, 3])), 1, span))
        return cell

    motif, events = make_motif(), []
    while sum(d for _, d, _ in events) * sixt < total - 1e-3 and len(events) < 4000:
        shift = int(rng.choice([0, 0, 2, -2, 3]))                # answer phrase, transposed
        events += [(int(np.clip(d + shift, 1, span)), du, a) for (d, du, a) in motif]
        events.append((0, 2, ""))                                # breathe between phrases
        if rng.random() < 0.4:
            motif = make_motif()                                 # occasionally restate a new idea
    notes = _lay_events(events, base, scale, sixt, total, lead=True)
    return notes or [(_degree_to_pitch(1, base, scale), 0.0, sixt, 100, "")]


def compose_riff(brain, key="E minor", bpm=160, duration_s=None, bars=8, style="gallop",
                 genre="", part="riff", provider="", model="", seed=None, context=""):
    """Pick the riff brain: 'algorithmic' (default, instant) or LLM-guided
    (provider set by the caller). `genre` (LLM only) sets feel + modal scale;
    `part` = 'riff' or 'solo' (lead line). `context` (LLM only) = an audio-grounded
    description of the section to ground the line in. LLM failures fall back to algorithmic."""
    if brain and brain != "algorithmic":
        try:
            return llm_riff(key, bpm, duration_s=duration_s, bars=bars, style=style,
                            genre=genre, part=part, provider=provider, model=model, seed=seed,
                            context=context)
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
        if typ == "solo":
            # A real solo section = a high-register LEAD line over a quieter
            # rhythm bed (both in one DI; the renderer sums overlapping notes).
            bed = generate_riff(key, bpm, duration_s=secs, style="powerchords",
                                seed=(None if seed is None else seed + i))
            lead = compose_riff(brain, key, bpm, duration_s=secs, part="solo", genre=genre,
                                provider=provider, model=model,
                                seed=(None if seed is None else seed + 1009 + i))
            seg = [(p, st, d, int(v * 0.55)) for (p, st, d, v) in bed] + list(lead)
        else:
            seg = compose_riff(brain, key, bpm, duration_s=secs, style=style, genre=genre,
                               provider=provider, model=model,
                               seed=(None if seed is None else seed + i))
        for n in seg:                                # tolerate optional 5th (articulation) field
            p, st, d, v = n[0], n[1], n[2], n[3]
            shifted = (p + octv, t0 + st, d, max(1, min(127, int(v * vscale))))
            notes.append(shifted + (n[4],) if len(n) > 4 else shifted)
        t0 += secs
    if not notes:
        raise ValueError("no sections to generate from")
    return notes


def test_riff(key="E minor", bpm=160, bars=4, seed=None):
    """Back-compat shim → the gallop generator."""
    return generate_riff(key=key, bpm=bpm, bars=bars, style="gallop", seed=seed)
