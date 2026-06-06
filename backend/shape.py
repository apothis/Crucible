"""Shape - a tone + dynamics shaping stage (de-harsh, multiband dynamics, transient
shaping, gentle tonal feel-match). Distinct from Master (Matchering = static EQ + loudness
match): Shape acts DYNAMICALLY and surgically, and is meant to be chained BEFORE and/or
AFTER a master. Pure Mac DSP (scipy/numpy), no GPU. Each module is opt-in.

Modules:
  deharsh  - dynamic resonance/fizz suppression (a Soothe-lite): in the STFT domain, pull
             down per-bin energy that sticks out above a smoothed spectral envelope (harsh
             resonances) AND dynamically duck a high band when it gets hot (broadband fizz),
             with optional 'air' restore at the very top. Tames HF only when it spikes.
  dynamics - multiband compression over Linkwitz-Riley bands (control dynamics per band).
  transient- broadband transient shaper (fast vs slow envelope -> emphasize/soften attacks).
  match    - gentle tonal feel-match to a reference's average spectrum, by an adjustable
             amount (partial, smoothed) - lighter/more flexible than Matchering's full match.
"""
import numpy as np


# ----------------------------- de-harsh -----------------------------
def hf_denoise(x, sr, freq=5000.0, strength=0.8, smooth_hz=600.0, smooth_ms=60.0,
               n_std=1.6):
    """Band-limited spectral-gate de-noise: split at `freq`, run stationary spectral
    gating (noisereduce) on ONLY the high band to scrub steady broadband HF hash/fizz,
    then recombine - mids/lows are untouched. `strength` (0..1) = prop_decrease;
    `smooth_hz`/`smooth_ms` trade depth against 'musical noise' (higher = smoother)."""
    import noisereduce as nr
    lo, hi = _lr_split(x.astype("float64"), sr, [float(freq)])     # 2-way crossover
    try:
        hi_dn = nr.reduce_noise(y=hi.astype("float32"), sr=sr, stationary=True,
                                prop_decrease=float(np.clip(strength, 0, 1)),
                                freq_mask_smooth_hz=float(smooth_hz),
                                time_mask_smooth_ms=float(smooth_ms),
                                n_std_thresh_stationary=float(n_std), n_fft=1024)
    except Exception:
        return x.astype("float32")
    n = min(len(lo), len(hi_dn))
    return (lo[:n] + hi_dn[:n]).astype("float32")


def deharsh(x, sr, smooth=0.5, freq=6000.0, resonance=0.3, sharpness=0.5, air_db=0.0,
            denoise=0.0, denoise_smooth=0.6, nfft=2048, hop=512):
    """De-fizz / de-harsh a mono channel in the STFT domain.
    `smooth` (0..1) tames broadband HF fizz = per-bin spectral subtraction of the steady
    noise floor PLUS dynamic ducking of HF spikes; `resonance` (0..1) ducks harsh spectral
    peaks that flare up (temporally gated so steady musical harmonics are spared);
    `freq` = where HF taming starts; `air_db` = gentle top-end restore shelf.
    `denoise` (0..1) = STRONG mode: a band-limited stationary spectral-gate de-noise of
    the HF band (best for constant 'digital' fizz/hash); runs first, then smooth/resonance
    refine. `denoise_smooth` raises gate smoothing to trade depth vs musical-noise."""
    if denoise > 0:                                              # strong spectral-gate de-fizz first
        x = hf_denoise(x, sr, freq=freq, strength=denoise,
                       smooth_hz=400.0 + 800.0 * float(denoise_smooth),
                       smooth_ms=40.0 + 80.0 * float(denoise_smooth))
    from scipy.signal import stft, istft
    from scipy.ndimage import median_filter, uniform_filter1d
    f, t, Z = stft(x, fs=sr, nperseg=nfft, noverlap=nfft - hop, window="hann", boundary="zeros")
    mag = np.abs(Z); phase = np.angle(Z)
    band = f >= float(freq)
    gain = np.ones_like(mag)

    # 0) spectral subtraction of STEADY HF fizz: estimate each band-bin's noise floor (a low
    # percentile over time) and subtract a fraction. Constant hiss (mag ~ floor) is reduced
    # hard; loud musical HF (mag >> floor) is barely touched -> de-fizz without dulling.
    if smooth > 0 and band.any():
        nf = np.percentile(mag[band, :], 15, axis=1, keepdims=True)
        sub = (mag[band, :] - float(smooth) * 1.5 * nf) / (mag[band, :] + 1e-9)
        floor_g = 10.0 ** (-(float(smooth) * 12.0) / 20.0)       # max steady-HF cut ~ smooth*12 dB
        gain[band, :] *= np.clip(sub, floor_g, 1.0)

    # 1) resonance suppression: bins that exceed a freq-smoothed envelope AND are
    # temporally HOT (building up) - the temporal gate protects steady musical harmonics
    # (which are spectral peaks too) so we only duck resonances/harshness that flare up.
    if resonance > 0 and band.any():
        logm = np.log(mag + 1e-9)
        width = int(np.interp(np.clip(sharpness, 0, 1), [0, 1], [41, 7])) | 1
        env = median_filter(logm, size=(width, 1))
        excess = np.clip(logm - env, 0, None) * 8.686            # dB above local spectral envelope
        tavg = uniform_filter1d(mag, size=max(3, int(0.15 * sr / hop)) | 1, axis=1)
        tgate = np.clip(mag / (tavg + 1e-9) - 1.0, 0.0, 1.0)     # ~0 for steady tones, >0 when flaring
        gr_db = -float(resonance) * excess * tgate
        gr_db[~band, :] = 0.0
        gain *= 10.0 ** (gr_db / 20.0)

    # 2) dynamic HF ducking: duck the band when its energy spikes above its running median.
    if smooth > 0 and band.any():
        be = np.sqrt(np.mean(mag[band, :] ** 2, axis=0)) + 1e-9   # band energy per frame
        ref = median_filter(be, size=max(3, int(0.4 * sr / hop)) | 1)
        over = np.clip(be / (ref + 1e-9) - 1.0, 0, 4.0)
        cut_db = -float(smooth) * 4.0 * over
        cut_db = uniform_filter1d(cut_db, size=max(1, int(0.02 * sr / hop)))
        gain[band, :] *= (10.0 ** (cut_db / 20.0))[None, :]

    # 3) air restore: gentle static high shelf above ~12 kHz.
    if abs(air_db) > 0.01:
        hi = f >= 12000.0
        gain[hi, :] *= 10.0 ** (float(air_db) / 20.0)

    Z2 = mag * gain * np.exp(1j * phase)
    _, y = istft(Z2, fs=sr, nperseg=nfft, noverlap=nfft - hop, window="hann", boundary=True)
    return y[:len(x)].astype("float32")


# ----------------------------- multiband dynamics -----------------------------
def _lr_split(x, sr, crossovers):
    """Split into bands at the given crossover freqs using cascaded Butterworth (~LR4)."""
    from scipy.signal import butter, sosfilt
    bands, rem = [], x
    for fc in crossovers:
        fc = min(max(fc, 20.0), sr / 2 - 100)
        lo = butter(2, fc / (sr / 2), "low", output="sos")
        hi = butter(2, fc / (sr / 2), "high", output="sos")
        bands.append(sosfilt(lo, sosfilt(lo, rem)))
        rem = sosfilt(hi, sosfilt(hi, rem))
    bands.append(rem)
    return bands


def _compress(x, sr, threshold_db=-24.0, ratio=2.0, attack_ms=10.0, release_ms=120.0,
              makeup_db=0.0):
    """Simple feed-forward compressor with an attack/release-smoothed envelope."""
    eps = 1e-9
    env = np.abs(x) + eps
    aa = np.exp(-1.0 / (max(1e-4, attack_ms / 1000.0) * sr))
    ar = np.exp(-1.0 / (max(1e-4, release_ms / 1000.0) * sr))
    sm = np.empty_like(env); prev = env[0]
    for i in range(len(env)):                                    # one-pole, attack vs release
        coef = aa if env[i] > prev else ar
        prev = coef * prev + (1 - coef) * env[i]
        sm[i] = prev
    db = 20.0 * np.log10(sm)
    over = np.maximum(0.0, db - threshold_db)
    gr_db = -over * (1.0 - 1.0 / max(1.0, ratio))
    g = 10.0 ** ((gr_db + makeup_db) / 20.0)
    return (x * g).astype("float32")


def multiband(x, sr, amount=0.5, crossovers=(180.0, 2000.0, 8000.0), bands_cfg=None):
    """Multiband compression. `amount` (0..1) = one-knob glue (sets threshold/ratio/makeup
    across all bands); or pass explicit per-band `bands_cfg` (threshold_db/ratio/attack_ms/
    release_ms/makeup_db) for full control."""
    bands = _lr_split(x, sr, list(crossovers))
    if bands_cfg is None:
        a = float(np.clip(amount, 0, 1))
        thr = -18.0 - 12.0 * a; ratio = 1.5 + 2.5 * a; mk = 1.5 * a
        bands_cfg = [dict(threshold_db=thr, ratio=ratio, makeup_db=mk) for _ in bands]
    out = np.zeros_like(bands[0])
    for b, cfg in zip(bands, bands_cfg):
        if cfg is None:
            out = out + b
        else:
            out = out + _compress(b, sr, **{**dict(threshold_db=-26.0, ratio=2.0,
                                                    attack_ms=15.0, release_ms=150.0), **cfg})
    return out.astype("float32")


# ----------------------------- transient shaper -----------------------------
def transient(x, sr, attack=0.0, sustain=0.0):
    """Emphasize (attack>0) or soften (<0) transients; sustain shapes the tail.
    Uses a fast vs slow envelope differential to find attacks."""
    if abs(attack) < 1e-3 and abs(sustain) < 1e-3:
        return x.astype("float32")
    eps = 1e-9
    e = np.abs(x) + eps
    def env(ms):
        c = np.exp(-1.0 / (max(1e-4, ms / 1000.0) * sr))
        o = np.empty_like(e); p = e[0]
        for i in range(len(e)):
            p = c * p + (1 - c) * e[i]; o[i] = p
        return o
    fast, slow = env(2.0), env(60.0)
    trans = np.clip((fast - slow) / (slow + eps), -1.0, 1.0)      # >0 at attacks
    tail = np.clip((slow - fast) / (slow + eps), -1.0, 1.0)       # >0 in sustain
    g = 1.0 + float(attack) * trans + float(sustain) * tail
    g = np.clip(g, 0.1, 4.0)
    return (x * g).astype("float32")


# ----------------------------- tonal feel-match -----------------------------
def tonal_match(x, ref, sr, amount=0.5, n=8192):
    """Gently match x's average magnitude spectrum toward ref's, by `amount` (0..1),
    applied as a smoothed zero-phase correction. Lighter than a full Matchering match."""
    from scipy.signal import welch
    if ref is None or len(ref) < n:
        return x.astype("float32")
    fx, px = welch(x, fs=sr, nperseg=n)
    fr, pr = welch(ref, fs=sr, nperseg=n)
    px = px + 1e-12; pr = pr + 1e-12
    corr = np.sqrt(pr / px)                                       # magnitude correction
    # smooth in log-frequency (octave-ish) to avoid combing
    from scipy.ndimage import uniform_filter1d
    corr = np.exp(uniform_filter1d(np.log(corr), size=9))
    corr = corr ** float(np.clip(amount, 0, 1))                   # partial match
    corr = np.clip(corr, 0.25, 4.0)
    # apply via FFT bin interpolation (zero-phase)
    X = np.fft.rfft(x)
    fb = np.fft.rfftfreq(len(x), 1.0 / sr)
    g = np.interp(fb, fx, corr)
    y = np.fft.irfft(X * g, n=len(x))
    return y.astype("float32")


# ----------------------------- orchestrator -----------------------------
_ORDER = ("deharsh", "dynamics", "transient", "match")


def process_file(in_path, out_path, config, ref_path=None):
    """Run the enabled Shape modules (in `config`) over a file. config = {module: params|None};
    a module runs only if present with a truthy dict. Returns a small report."""
    import soundfile as sf
    data, sr = sf.read(in_path, dtype="float32", always_2d=True)
    out_subtype = "FLOAT"
    try:
        if in_path.lower().endswith(".wav"):
            out_subtype = sf.info(in_path).subtype or "FLOAT"
    except Exception:
        out_subtype = "FLOAT"
    ref = None
    if ref_path:
        try:
            r, rsr = sf.read(ref_path, dtype="float32", always_2d=True)
            ref = r.mean(axis=1)
        except Exception:                               # uploaded MP3/M4A/etc. -> robust decode
            import librosa
            ref, rsr = librosa.load(ref_path, sr=None, mono=True)
        if rsr != sr:
            import librosa
            ref = librosa.resample(ref, orig_sr=rsr, target_sr=sr)

    applied = []
    y = data.copy()
    for mod in _ORDER:
        if mod not in config or config.get(mod) is None:
            continue                                            # module enabled iff its key is present (dict, may be empty=defaults)
        cfg = config.get(mod) or {}
        applied.append(mod)
        for c in range(y.shape[1]):
            ch = y[:, c]
            if mod == "deharsh":
                y[:, c] = deharsh(ch, sr, **cfg)
            elif mod == "dynamics":
                y[:, c] = multiband(ch, sr, **cfg)
            elif mod == "transient":
                y[:, c] = transient(ch, sr, **cfg)
            elif mod == "match":
                y[:, c] = tonal_match(ch, ref, sr, **cfg)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.999:                                             # guard only if a stage overshot
        y = y * (0.999 / peak)
    sf.write(out_path, y, sr, subtype=out_subtype)
    return {"applied": applied, "sr": sr, "duration_s": round(len(data) / sr, 2),
            "peak": round(peak, 3)}
