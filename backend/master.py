"""Reference-based mastering via Matchering 2.0 — runs LOCALLY on the Mac.

Matches a target track's loudness, frequency response, peak and stereo width to
a *reference* master the user supplies (a pro track they own — e.g. a Halestorm
/ AC-DC / Bon Jovi song). See RESEARCH.md §10a/§10e #2.

Personal-use: the reference is only analysed, never redistributed.
"""


def available():
    try:
        import matchering  # noqa: F401
        return True
    except Exception:
        return False


def master(target_path: str, reference_path: str, out_path: str, bit_depth: int = 16):
    """Master `target_path` toward `reference_path`, writing a WAV to out_path."""
    import matchering as mg

    # keep matchering quiet-ish in the backend log (no crashes on info spam)
    try:
        mg.log(warning_handler=print, info_handler=lambda *a, **k: None,
               show_codename=False)
    except Exception:
        pass

    result = mg.pcm24(out_path) if bit_depth == 24 else mg.pcm16(out_path)
    mg.process(target=target_path, reference=reference_path, results=[result])
    return out_path


# ---------------- Reference-FREE "auto" mastering (gold-standard, no reference) ----------------
# A deterministic master-bus chain on the Mac via Spotify `pedalboard` + `pyloudnorm`:
#   tonal-curve EQ → glue compression → loudness-normalize to a target LUFS → true-peak limit.
# Gives the "one-click gold standard" master with no reference track, and a loudness target
# (streaming −14 / Apple −16 / loud −9). See RESEARCH.md §16 option ②.

# Gentle, broad master-bus tone curves (subtractive-leaning to stay clean). Each = list of
# (pedalboard filter name, kwargs); resolved against pedalboard at call time.
TONE_CURVES = {
    "balanced": [("HighpassFilter", {"cutoff_frequency_hz": 28}),
                 ("LowShelfFilter", {"cutoff_frequency_hz": 90, "gain_db": 1.0, "q": 0.7}),
                 ("PeakFilter", {"cutoff_frequency_hz": 320, "gain_db": -1.0, "q": 1.0}),
                 ("HighShelfFilter", {"cutoff_frequency_hz": 10000, "gain_db": 1.5, "q": 0.7})],
    "metal":    [("HighpassFilter", {"cutoff_frequency_hz": 32}),
                 ("PeakFilter", {"cutoff_frequency_hz": 400, "gain_db": -2.0, "q": 1.1}),    # tame low-mid mud
                 ("PeakFilter", {"cutoff_frequency_hz": 3500, "gain_db": 1.5, "q": 0.9}),    # attack/presence
                 ("HighShelfFilter", {"cutoff_frequency_hz": 9000, "gain_db": 2.0, "q": 0.7})],  # air/sizzle
    "warm":     [("HighpassFilter", {"cutoff_frequency_hz": 25}),
                 ("LowShelfFilter", {"cutoff_frequency_hz": 110, "gain_db": 1.5, "q": 0.7}),
                 ("HighShelfFilter", {"cutoff_frequency_hz": 12000, "gain_db": -1.0, "q": 0.7})],
    "bright":   [("HighpassFilter", {"cutoff_frequency_hz": 30}),
                 ("PeakFilter", {"cutoff_frequency_hz": 250, "gain_db": -1.5, "q": 1.0}),
                 ("HighShelfFilter", {"cutoff_frequency_hz": 8000, "gain_db": 3.0, "q": 0.7})],
    "flat":     [("HighpassFilter", {"cutoff_frequency_hz": 20})],   # loudness/limit only, no tone shaping
}

# Loudness presets (integrated LUFS). −14 = streaming-safe; −9 = loud modern master.
LUFS_PRESETS = {"apple": -16.0, "streaming": -14.0, "balanced": -12.0, "loud": -9.0}


def auto_available():
    try:
        import pedalboard  # noqa: F401
        import pyloudnorm  # noqa: F401
        import soundfile   # noqa: F401
        return True
    except Exception:
        return False


def _stereo_width(x, sr, width: float, bass_mono_hz: float):
    """Mid/Side widener. `width` scales the Side channel (1.0 = unchanged, >1 wider,
    <1 narrower). `bass_mono_hz` (>0) high-passes the Side so lows stay centred — the
    standard 'keep the bass mono' trick that widens without smearing the low end."""
    import numpy as np
    import pedalboard as pb
    L, R = x[0], x[1]
    mid = (L + R) * 0.5
    side = (L - R) * 0.5 * float(width)
    if bass_mono_hz and bass_mono_hz > 0:
        hp = pb.Pedalboard([pb.HighpassFilter(cutoff_frequency_hz=float(bass_mono_hz))])
        side = hp(side.reshape(1, -1).astype("float32"), sr).reshape(-1)
    return np.stack([mid + side, mid - side], axis=0)


def _saturate(x, amount: float):
    """Gentle harmonic 'warmth' via soft (tanh) saturation, dry/wet-blended by `amount`
    (0 = bypass, 1 = max). Adds analog-style glue/density without a hard distortion edge."""
    import numpy as np
    a = max(0.0, min(1.0, float(amount)))
    if a <= 0:
        return x
    drive = 1.0 + 3.0 * a
    wet = np.tanh(x * drive) / np.tanh(drive)
    return (1.0 - a) * x + a * wet


def master_auto(target_path: str, out_path: str, target_lufs: float = -12.0,
                tone: str = "balanced", ceiling_db: float = -1.0, bit_depth: int = 16,
                width: float = 1.0, bass_mono_hz: float = 0.0, warmth: float = 0.0):
    """Reference-free master: tone curve → glue comp → (stereo width) → (warmth) →
    normalize to target_lufs → limit. Writes a WAV to out_path.
    Returns (out_path, measured_lufs_after)."""
    import numpy as np
    import soundfile as sf
    import pedalboard as pb
    import pyloudnorm as pyln

    data, sr = sf.read(target_path, dtype="float32", always_2d=True)   # (frames, ch)
    x = data.T                                                          # (ch, frames)
    if x.shape[0] == 1:
        x = np.repeat(x, 2, axis=0)                                     # mono → stereo
    elif x.shape[0] > 2:
        x = x[:2]

    moves = TONE_CURVES.get(tone, TONE_CURVES["balanced"])
    plugins = [getattr(pb, name)(**kw) for name, kw in moves]
    plugins.append(pb.Compressor(threshold_db=-18.0, ratio=2.0, attack_ms=15.0, release_ms=180.0))  # gentle glue
    pre = pb.Pedalboard(plugins)(x, sr)                                 # (ch, frames) — tone-shaped, pre-loudness

    # Stereo + harmonic enhancement BEFORE the loudness/limit stage, so the limiter
    # catches any peak lift these add (width=1 + bass_mono=0 + warmth=0 = exact no-op).
    if float(width) != 1.0 or (bass_mono_hz and bass_mono_hz > 0):
        pre = _stereo_width(pre, sr, width, bass_mono_hz)
    pre = _saturate(pre, warmth)

    meter = pyln.Meter(sr)
    limiter = pb.Pedalboard([pb.Limiter(threshold_db=ceiling_db, release_ms=120.0)])
    ceil_lin = 10.0 ** (ceiling_db / 20.0)

    # The limiter both controls peaks AND raises integrated loudness by an amount that
    # depends on how hard we drive it — so we can't just normalize-then-limit. Instead
    # iterate the PRE-limiter drive gain, re-measuring the limited+ceilinged result each
    # pass until integrated LUFS lands on target (the "re-measure loop", RESEARCH §16).
    base = meter.integrated_loudness(pre.T)                             # pyloudnorm wants (frames, ch)
    if not np.isfinite(base):
        y, after = limiter(pre, sr), None
    else:
        gain_db = target_lufs - base                                   # first guess
        y, after = pre, base
        for _ in range(6):
            cand = limiter(pre * (10.0 ** (gain_db / 20.0)), sr)
            peak = float(np.max(np.abs(cand))) if cand.size else 0.0
            if peak > ceil_lin and peak > 0:                           # enforce the true-peak ceiling
                cand = cand * (ceil_lin / peak)
            cur = meter.integrated_loudness(cand.T)
            y = cand
            if not np.isfinite(cur):
                after = None
                break
            after = cur
            if abs(target_lufs - cur) <= 0.3:                          # converged
                break
            gain_db += (target_lufs - cur)                             # nudge drive by the error

    sf.write(out_path, y.T, sr, format="WAV", subtype=("PCM_24" if bit_depth == 24 else "PCM_16"))
    return out_path, (float(after) if after is not None else None)


def _lufs(path):
    """Integrated LUFS of an audio file (stereo-aware)."""
    import soundfile as sf
    import pyloudnorm as pyln
    y, sr = sf.read(path, always_2d=True)
    return float(pyln.Meter(sr).integrated_loudness(y))


def reference_match(target_path, ref_path, out_path, match_amount=1.0, drive=0.7,
                    ceiling_db=-1.0, bit_depth=16):
    """Reference master that keeps dynamics: match the reference's TONE (shape.tonal_match,
    no loudness/limiting), then push loudness TOWARD the reference's by `drive` (0..1) through
    the transparent auto chain (glue + soft-saturation + true-peak limit) instead of Matchering's
    brute RMS-match. drive=0 = tone only (most dynamic); drive=1 = the reference's full loudness
    but via clip+limit (punchier than Matchering). Returns (out_path, achieved_lufs|None)."""
    import os
    import shutil
    import tempfile
    from . import shape
    work = tempfile.mkdtemp()
    toned = os.path.join(work, "toned.wav")
    try:
        shape.process_file(target_path, toned, {"match": {"amount": float(match_amount)}}, ref_path=ref_path)
        if float(drive) <= 0 or not auto_available():
            shutil.copy(toned, out_path)               # tone only, dynamics fully intact
            return out_path, None
        toned_l, ref_l = _lufs(toned), _lufs(ref_path)
        target_lufs = toned_l + float(drive) * (ref_l - toned_l)   # partial loudness toward the reference
        return master_auto(toned, out_path, target_lufs=target_lufs, tone="flat",
                           warmth=0.12, ceiling_db=float(ceiling_db), bit_depth=int(bit_depth))
    finally:
        shutil.rmtree(work, ignore_errors=True)
