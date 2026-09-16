"""Naturalize an AI-generated track — runs LOCALLY on the Mac (pedalboard + numpy).

Reshapes audio so it reads less "machine-perfect" to naive spectral/AI detectors, by
adding back the small imperfections a purely synthetic render lacks. This is the honest
subset of what tools like undetectr.com sell, MINUS the watermark/SynthID theatre (our
generators embed no SynthID/C2PA, so there's nothing to strip).

What it does, in order:
  0. Level-drift correction (optional) - Suno v6 takes tend to creep louder across a song
                                       (measured 2026-09-16, Your Fill of Blood: +0.83 dB/min,
                                       ~4 dB first verse to end). A slow gain ride flattens
                                       that trend against the opening minute while sections
                                       shorter than the window keep their contrast. Runs FIRST
                                       so it sits before Master's normalise + limit.
  1. Subtle harmonic saturation      — natural even/odd harmonics (reuse of the master.py trick).
  2. High-frequency taming           — gentle top-end shelf to soften brittle vocoder fizz.
  3. Micro wow/flutter (optional)    — tiny slow time modulation so timing isn't grid-perfect.
  4. Lossy round-trip (optional)     — encode to MP3/AAC and back; smears the frequency-domain
                                       upsampling signature better than any EQ. The strongest step.
  5. Noise-floor injection           — a very low-level shaped noise bed so the "digital black"
                                       silence between transients isn't suspiciously perfect;
                                       added LAST so a lossy codec can't quantise it away. Also
                                       acts as dither.

HONEST CAVEAT (surface this in the UI): this is cosmetic and unverifiable. It makes audio
look less machine-perfect to naive detectors; it will NOT fool a classifier trained on the
specific generator, and no one can prove a given track now "passes". Cat and mouse.

Pure Mac DSP, no GPU, no box. Chainable before/after Master and Shape.
"""
import os
import shutil
import subprocess
import tempfile


def available():
    """True if the core DSP deps are importable (round-trip additionally needs ffmpeg)."""
    try:
        import numpy      # noqa: F401
        import soundfile  # noqa: F401
        import pedalboard  # noqa: F401
        return True
    except Exception:
        return False


def ffmpeg_available():
    return shutil.which("ffmpeg") is not None


def _saturate(x, amount: float):
    """Gentle harmonic 'warmth' via soft (tanh) saturation, dry/wet-blended by `amount`
    (0 = bypass, 1 = max). Same curve as master.py so the two tools stay consistent."""
    import numpy as np
    a = max(0.0, min(1.0, float(amount)))
    if a <= 0:
        return x
    drive = 1.0 + 3.0 * a
    wet = np.tanh(x * drive) / np.tanh(drive)
    return (1.0 - a) * x + a * wet


def _wow_flutter(x, sr, depth: float):
    """Tiny slow time modulation (analog wow/flutter). `depth` 0..1 maps to a max timing
    deviation of ~0..6 ms via two slow LFOs (~0.6 Hz wow + ~6 Hz flutter). Fractional-sample
    resample per channel with np.interp so it stays click-free. depth=0 is an exact no-op."""
    import numpy as np
    d = max(0.0, min(1.0, float(depth)))
    if d <= 0:
        return x
    n = x.shape[1]
    t = np.arange(n, dtype=np.float64)
    max_dev = 0.006 * d * sr                                   # peak deviation in samples
    lfo = (0.7 * np.sin(2 * np.pi * 0.6 * t / sr)             # slow wow
           + 0.3 * np.sin(2 * np.pi * 6.0 * t / sr))          # faster flutter
    warp = t + max_dev * lfo
    np.clip(warp, 0, n - 1, out=warp)
    out = np.empty_like(x)
    for c in range(x.shape[0]):
        out[c] = np.interp(warp, t, x[c]).astype(x.dtype, copy=False)
    return out


def _add_noise_floor(x, sr, floor_db: float):
    """Add a very low-level, slightly-pink noise bed at `floor_db` dBFS (peak-referenced).
    Lowpass tilt so it sits as a warm hiss, not bright white. Doubles as dither."""
    import numpy as np
    import pedalboard as pb
    if floor_db is None or floor_db >= 0:
        return x
    amp = 10.0 ** (float(floor_db) / 20.0)
    rng = np.random.default_rng()                              # non-repro noise is fine here
    noise = rng.standard_normal(x.shape).astype(np.float32)
    tilt = pb.Pedalboard([pb.LowpassFilter(cutoff_frequency_hz=6000.0)])  # darken (vectorised)
    noise = tilt(noise, sr)
    noise /= (float(np.max(np.abs(noise))) or 1.0)            # renormalise after filtering
    return x + noise * amp


def _short_term_lufs(x, sr, win_s=3.0, hop_s=0.5):
    """Short-term loudness curve (EBU R128 style: K-weighted, 3 s windows) via pyloudnorm's
    K-weighting filters. Returns (times, lufs) with times = window END in seconds. Silent
    windows come back as -inf."""
    import numpy as np
    import pyloudnorm as pyln
    meter = pyln.Meter(sr)
    ms = np.zeros(x.shape[1], dtype=np.float64)
    for c in range(x.shape[0]):
        y = x[c].astype(np.float64)
        for f in meter._filters.values():
            y = f.apply_filter(y)
        ms += y * y                       # channel weights are 1.0 for L/R
    win, hop = int(win_s * sr), int(hop_s * sr)
    if x.shape[1] < win:
        win = x.shape[1]
    cs = np.concatenate([[0.0], np.cumsum(ms)])
    ends = np.arange(win, x.shape[1] + 1, hop)
    mean_sq = (cs[ends] - cs[ends - win]) / win
    with np.errstate(divide="ignore"):
        lufs = -0.691 + 10.0 * np.log10(mean_sq)
    return ends / sr, lufs


def _level_drift(x, sr, strength: float, window_s: float = 40.0, anchor_s: float = 60.0,
                 max_cut_db: float = 9.0):
    """Flatten the SLOW loudness trend of a song toward its opening minute.

    Measures short-term loudness, smooths it with a `window_s` running mean (so a bridge or a
    solo shorter than the window keeps its contrast while a creep across the whole song is
    caught), takes the mean of the first `anchor_s` of that trend as the reference level, and
    rides the gain by `strength` x (reference - trend). Gain is clamped to (-max_cut_db, 0]:
    it only ever pulls a climb back, never boosts a quiet ending. Silent windows (intro/outro
    gaps) are ignored when forming the trend. `strength` 0 = exact no-op.
    Returns (x, stats) with stats = {drift_db, max_cut_db, ref_lufs}."""
    import numpy as np
    k = max(0.0, min(1.0, float(strength)))
    if k <= 0 or x.shape[1] < int(10 * sr):
        return x, None
    t, lufs = _short_term_lufs(x, sr)
    valid = np.isfinite(lufs)
    if valid.sum() < 10:
        return x, None
    gate = max(np.max(lufs[valid]) - 30.0, -50.0)         # ignore near-silence
    valid &= lufs > gate
    if valid.sum() < 10:
        return x, None
    tv, lv = t[valid], lufs[valid]
    # running mean over window_s of the gated curve (normalised by the valid count per window)
    hop = float(t[1] - t[0]) if len(t) > 1 else 0.5
    n = max(1, int(round(window_s / hop)))
    kern = np.ones(n)
    num = np.convolve(lv, kern, mode="same")
    den = np.convolve(np.ones_like(lv), kern, mode="same")
    trend = num / den
    ref = float(np.mean(trend[tv <= tv[0] + anchor_s]))
    gain_db = k * (ref - trend)
    gain_db = np.clip(gain_db, -abs(float(max_cut_db)), 0.0)
    # per-sample gain: interpolate the (sparse, already-smooth) curve; hold the ends
    samples = np.arange(x.shape[1]) / sr
    g = np.interp(samples, tv, gain_db, left=gain_db[0], right=gain_db[-1])
    y = (x * (10.0 ** (g / 20.0))[None, :]).astype(x.dtype, copy=False)
    last = trend[tv >= tv[-1] - (tv[-1] - tv[0]) / 3.0]
    stats = {"drift_db": round(float(np.mean(last) - ref), 2),
             "max_cut_db": round(float(-gain_db.min()), 2),
             "ref_lufs": round(ref, 1)}
    return y, stats


def _roundtrip(x, sr, codec: str, bitrate: str):
    """Encode → decode through a lossy codec to smear the frequency-domain synthesis signature,
    then return the decoded PCM (resampled back to `sr`, same channel count). Needs ffmpeg;
    returns x unchanged if ffmpeg is missing or the round-trip fails."""
    import numpy as np
    import soundfile as sf
    if codec in (None, "none", "") or not ffmpeg_available():
        return x, False
    ext = {"mp3": ".mp3", "aac": ".m4a"}.get(codec)
    enc = {"mp3": "libmp3lame", "aac": "aac"}.get(codec)
    if not ext:
        return x, False
    work = tempfile.mkdtemp()
    try:
        src = os.path.join(work, "in.wav")
        comp = os.path.join(work, "c" + ext)
        back = os.path.join(work, "out.wav")
        sf.write(src, x.T, sr, subtype="PCM_24")
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-i", src, "-c:a", enc, "-b:a", str(bitrate), comp],
                       check=True)
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-i", comp, "-ar", str(sr), back],
                       check=True)
        y, _ = sf.read(back, dtype="float32", always_2d=True)
        y = y.T
        if y.shape[0] == 1 and x.shape[0] == 2:
            y = np.repeat(y, 2, axis=0)
        elif y.shape[0] > x.shape[0]:
            y = y[: x.shape[0]]
        # codec delay/padding can shift length by a few samples — trim/pad to match
        if y.shape[1] > x.shape[1]:
            y = y[:, : x.shape[1]]
        elif y.shape[1] < x.shape[1]:
            y = np.pad(y, ((0, 0), (0, x.shape[1] - y.shape[1])))
        return y, True
    except Exception:
        return x, False
    finally:
        shutil.rmtree(work, ignore_errors=True)


def naturalize(target_path: str, out_path: str, warmth: float = 0.12,
               hf_tame_db: float = -1.5, hf_tame_hz: float = 12000.0,
               wow_flutter: float = 0.0, roundtrip: str = "none",
               roundtrip_bitrate: str = "320k", noise_floor_db: float = -65.0,
               ceiling_db: float = -1.0, bit_depth: int = 16,
               level_drift: float = 0.0, drift_window_s: float = 40.0):
    """Naturalize `target_path`, writing a WAV to out_path. All stages are individually
    de-fanged by their neutral value (level_drift 0, warmth 0, hf_tame_db 0, wow_flutter 0,
    roundtrip 'none', noise_floor_db >= 0), so any subset can be run. Returns (out_path, applied) where
    `applied` is a list of short strings describing which stages actually ran."""
    import numpy as np
    import soundfile as sf
    import pedalboard as pb

    data, sr = sf.read(target_path, dtype="float32", always_2d=True)   # (frames, ch)
    x = data.T                                                          # (ch, frames)
    if x.shape[0] == 1:
        x = np.repeat(x, 2, axis=0)
    elif x.shape[0] > 2:
        x = x[:2]

    applied = []

    if float(level_drift) > 0:                                          # FIRST: before any colour
        x, drift = _level_drift(x, sr, level_drift, window_s=float(drift_window_s))
        if drift:
            applied.append(f"level drift {round(float(level_drift) * 100)}% "
                           f"(climb {drift['drift_db']:+.1f}dB, cut up to {drift['max_cut_db']:.1f}dB)")

    x = _saturate(x, warmth)
    if float(warmth) > 0:
        applied.append(f"warmth {round(float(warmth) * 100)}%")

    if float(hf_tame_db) < 0:                                           # gentle top-end shelf
        shelf = pb.Pedalboard([pb.HighShelfFilter(cutoff_frequency_hz=float(hf_tame_hz),
                                                  gain_db=float(hf_tame_db), q=0.7)])
        x = shelf(x, sr)
        applied.append(f"HF tame {hf_tame_db:g}dB>{round(float(hf_tame_hz) / 1000, 1)}k")

    if float(wow_flutter) > 0:
        x = _wow_flutter(x, sr, wow_flutter)
        applied.append(f"wow/flutter {round(float(wow_flutter) * 100)}%")

    x, did_rt = _roundtrip(x, sr, roundtrip, roundtrip_bitrate)
    if did_rt:
        applied.append(f"{roundtrip} round-trip @{roundtrip_bitrate}")

    if float(noise_floor_db) < 0:
        x = _add_noise_floor(x, sr, noise_floor_db)
        applied.append(f"noise floor {round(float(noise_floor_db))}dB")

    # true-peak safety: keep the added noise/saturation from nudging over the ceiling
    ceil_lin = 10.0 ** (float(ceiling_db) / 20.0)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > ceil_lin and peak > 0:
        x = x * (ceil_lin / peak)

    sf.write(out_path, x.T, sr, format="WAV",
             subtype=("PCM_24" if int(bit_depth) == 24 else "PCM_16"))
    return out_path, applied
