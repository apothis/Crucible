"""Automatic de-glitch / de-click for generated tracks.

ACE-Step sometimes bakes short impulsive glitches (clicks/pops/zipper bursts) into an
otherwise-good take. This detects them and repairs each one by interpolation, without
needing a clean reference and without harming real transients.

Approach (the classic audio-restoration pipeline, implemented in numpy/scipy/librosa
since Essentia has no Apple-Silicon wheel):

  DETECT  - whiten the signal with a short per-block LPC inverse filter so the music's
            tonal/percussive energy is flattened and only the impulsive "innovation"
            (clicks) stands out, then threshold the residual against a robust noise
            floor (MAD). A click is a VERY short over-threshold run (<= max_run samples);
            longer over-threshold runs are real percussive onsets (snare/kick) -> ignored.
  REPAIR  - replace each click's samples with autoregressive (LPC) interpolation from the
            surrounding clean context (Janssen/Vaseghi-style least-squares), the standard
            high-quality fill for tonal audio; cubic-spline fallback if the AR solve is
            unstable. Tiny edge blend avoids introducing a new discontinuity.
  REPORT  - returns exactly what was repaired (location, length, channel) so the caller
            can confirm it only touched real glitches.

Longer "garbled" bursts (tens-hundreds of ms) are detected and FLAGGED but not
interpolation-repaired here (they need diffusion inpainting) - see deglitch() report.
"""
import numpy as np


def _global_residual(x, order):
    """Whiten with a SINGLE continuous LPC inverse filter (no per-block boundary
    artifacts). Imperfect whitening of non-stationary music is fine - clicks are still
    massive outliers, and detection normalizes by a LOCAL scale afterwards."""
    import librosa
    from scipy.signal import lfilter
    xf = x.astype("float64")
    if float(np.max(np.abs(xf))) < 1e-7:
        return np.zeros_like(xf)
    try:
        a = librosa.lpc(xf, order=order)               # a[0] = 1
    except Exception:
        return np.zeros_like(xf)
    r = lfilter(a, [1.0], xf)
    r[:order] = 0.0                                     # drop the filter-startup transient
    return r


def _runs(mask):
    """Yield (start, end) for each contiguous True run in a boolean array."""
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            yield i, j
            i = j
        else:
            i += 1


def detect(x, sr, order=24, threshold=14.0, max_click_ms=2.0, max_burst_ms=120.0,
           dilate=2):
    """Return (clicks, bursts): sample regions. A click = a spike whose whitened residual
    exceeds `threshold` x the LOCAL robust noise level for only a very short run
    (<= max_click_ms); longer over-threshold runs are real transients/glitch bursts and
    are FLAGGED (bursts) but not interpolation-repaired. Higher `threshold` = more
    conservative (default errs toward leaving good audio untouched)."""
    from scipy.ndimage import median_filter
    e = np.abs(_global_residual(x, order))
    if not np.any(e):
        return [], []
    # local robust scale (median of |e| over ~50 ms, robust to the spikes themselves),
    # computed on a decimated signal for speed then interpolated back.
    win = max(8, int(sr * 0.05))
    dec = max(1, win // 64)
    ed = e[::dec]
    k = max(3, (win // dec) | 1)
    scale = median_filter(ed, size=k)
    scale = np.interp(np.arange(len(e)), np.arange(len(ed)) * dec, scale)
    floor = float(np.median(e[e > 0])) if np.any(e > 0) else 1e-9
    scale = np.maximum(scale, floor * 0.25) + 1e-12
    over = (e / scale) > float(threshold)
    max_click = max(1, int(sr * max_click_ms / 1000.0))
    max_burst = max(max_click, int(sr * max_burst_ms / 1000.0))
    merge_gap = max(1, int(sr * 2.0 / 1000.0))         # join spikes within 2 ms FIRST so a
    # dense transient (drum/noise burst = many spikes) collapses into ONE long span and is
    # classified as a burst, while a truly isolated click stays short.
    spans = []
    for s, e2 in _runs(over):
        if spans and s - spans[-1][1] <= merge_gap:
            spans[-1][1] = e2
        else:
            spans.append([s, e2])
    clicks, bursts = [], []
    for s, e2 in spans:
        run = e2 - s
        if run <= max_click:
            clicks.append((max(0, s - dilate), min(len(x), e2 + dilate)))
        elif run <= max_burst:
            bursts.append((s, e2))
    return clicks, bursts


def _spline_fill(x, s, e, ctx=24):
    from scipy.interpolate import CubicSpline
    a0, b1 = max(0, s - ctx), min(len(x), e + ctx)
    xs = np.concatenate([np.arange(a0, s), np.arange(e, b1)])
    ys = np.concatenate([x[a0:s], x[e:b1]])
    if len(xs) < 4:
        lo = x[max(0, s - 1)]; hi = x[min(len(x) - 1, e)]
        return np.linspace(lo, hi, e - s)
    return CubicSpline(xs, ys)(np.arange(s, e)).astype(x.dtype)


def _ar_fill(x, s, e, order=16, ctx=256):
    """Least-squares AR interpolation of x[s:e] from surrounding context."""
    import librosa
    a0, b1 = max(0, s - ctx), min(len(x), e + ctx)
    left, right = x[a0:s].astype("float64"), x[e:b1].astype("float64")
    if len(left) < order + 1 or len(right) < order + 1:
        return _spline_fill(x, s, e)
    try:
        a = librosa.lpc(np.concatenate([left, right]), order=order)   # len order+1
    except Exception:
        return _spline_fill(x, s, e)
    p = order
    w0, w1 = max(0, s - p), min(len(x), e + p)
    y = x[w0:w1].astype("float64").copy()
    L = len(y)
    if L <= p + 1:
        return _spline_fill(x, s, e)
    gap = np.arange(s - w0, e - w0)
    rows = L - p
    A = np.zeros((rows, L))
    arev = a[::-1]
    for t in range(p, L):
        A[t - p, t - p:t + 1] = arev               # a[k]*y[t-k], k=0..p
    known = np.ones(L, bool); known[gap] = False
    Au, Ak = A[:, gap], A[:, known]
    try:
        with np.errstate(all="ignore"):
            sol, *_ = np.linalg.lstsq(Au, -Ak @ y[known], rcond=None)
    except Exception:
        return _spline_fill(x, s, e)
    ctxpeak = float(np.max(np.abs(np.concatenate([left, right])))) + 1e-9
    if not np.all(np.isfinite(sol)) or float(np.max(np.abs(sol))) > 1.6 * ctxpeak:
        return _spline_fill(x, s, e)               # AR unstable -> safe fallback
    return sol.astype(x.dtype)


def _blend_edges(x, s, e, n=2):
    """Tiny cosine crossfade at the repair boundaries to avoid a new step."""
    if s - n >= 0 and n > 0:
        w = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))
        x[s:s + n] = x[s - 1] * (1 - w) + x[s:s + n] * w
    if e + n <= len(x) and n > 0:
        w = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))
        x[e - n:e] = x[e - n:e] * (1 - w[::-1]) + x[min(len(x) - 1, e)] * w[::-1]


def deglitch_channel(x, sr, order=24, threshold=14.0, max_click_ms=2.0,
                     max_burst_ms=120.0, repair="ar"):
    """De-glitch one mono channel. Returns (repaired, clicks, bursts)."""
    clicks, bursts = detect(x, sr, order=order, threshold=threshold,
                            max_click_ms=max_click_ms, max_burst_ms=max_burst_ms)
    y = x.astype("float64").copy()
    for s, e in clicks:
        vals = _ar_fill(y, s, e, order=min(order, 16)) if repair == "ar" else _spline_fill(y, s, e)
        y[s:e] = vals
        _blend_edges(y, s, e)
    return y.astype("float32"), clicks, bursts


def deglitch_file(in_path, out_path, threshold=14.0, max_click_ms=2.0,
                  max_burst_ms=120.0, order=24, repair="ar"):
    """Detect + repair clicks per channel, write the cleaned WAV. Returns a report dict."""
    import soundfile as sf
    data, sr = sf.read(in_path, dtype="float32", always_2d=True)     # (frames, ch)
    # preserve the source bit depth so non-repaired audio stays bit-identical (a float
    # WAV must NOT be quantized to PCM16 across the whole track); float for lossy sources.
    out_subtype = "FLOAT"
    try:
        if in_path.lower().endswith((".wav", ".flac")):
            out_subtype = sf.info(in_path).subtype or "FLOAT"
    except Exception:
        out_subtype = "FLOAT"
    out = data.copy()
    regions, bursts = [], []
    for c in range(data.shape[1]):
        y, cl, br = deglitch_channel(data[:, c], sr, order=order, threshold=threshold,
                                     max_click_ms=max_click_ms, max_burst_ms=max_burst_ms,
                                     repair=repair)
        out[:, c] = y
        for s, e in cl:
            regions.append({"channel": c, "start_s": round(s / sr, 3),
                            "dur_ms": round((e - s) * 1000.0 / sr, 2)})
        for s, e in br:
            bursts.append({"channel": c, "start_s": round(s / sr, 3),
                           "dur_ms": round((e - s) * 1000.0 / sr, 2)})
    # Surgical: leave every non-repaired sample bit-identical (no global re-normalize).
    # A plain clip guards only the rare case where a repair overshoots full scale.
    np.clip(out, -1.0, 1.0, out=out)
    sf.write(out_path, out, sr, subtype=out_subtype)
    total_ms = round(sum(r["dur_ms"] for r in regions), 1)
    return {"clicks_repaired": len(regions), "repaired_ms": total_ms,
            "bursts_flagged": len(bursts), "regions": regions[:200], "bursts": bursts[:50],
            "sr": sr, "duration_s": round(len(data) / sr, 2)}
