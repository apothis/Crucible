"""The song's bar grid, measured from the audio, for the music-video writer's opt-in bar-grid
cutting (scene boundaries on bar lines, one bar minimum, varied lengths; user rule 2026-11-13).

Measured, per track, nothing assumed:
- beats: librosa's beat tracker (tightness 200, hop 128 = 5.8 ms resolution);
- tempo: a straight-line fit of beat time against beat number. librosa's own tempo readout is
  quantised to its tempogram bins (it read 152 bpm on a song that fits 155.000 bpm with 6 ms
  residual, Nobody Stays Dead, 2026-11-13), so it is only the tracker's starting point;
  half/double tempo is folded into 90-180 bpm;
- downbeat: of the beats_per_bar phases, the one where the harmony changes most (chroma novelty)
  and the kick is loudest (onsets under 120 Hz), each scored as a z-score across the phases.

A steady song (fit residual under STEADY_RMS) gets a constant grid; otherwise the bar lines are
every beats_per_bar-th tracked beat, so a drifting tempo is followed (and the result says so).
"""

import hashlib
import json
import os

import numpy as np

SR = 22050
HOP = 128
TEMPO_LO, TEMPO_HI = 90.0, 180.0
STEADY_RMS = 0.025            # seconds: inlier residual of a constant tempo
STEADY_INLIERS = 0.85         # share of tracked beats within INLIER_TOL of the constant grid
INLIER_TOL = 0.12             # of a beat: a tracked beat further off the grid is a tracker slip
PHASE_WINDOW = 16             # bars per window when checking the downbeat holds through the song
GRID_VERSION = 2


def _cache_key(path, beats_per_bar, bpm_hint):
    st = os.stat(path)
    raw = f"{os.path.basename(path)}|{st.st_size}|{int(st.st_mtime)}|{beats_per_bar}|{bpm_hint}|v{GRID_VERSION}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _coherence(ts, lo, hi, n=4000):
    """The period in [lo, hi] whose grid the beat times agree with most (circular mean length R,
    1 = every beat on one grid). A beat the tracker inserted or slipped only lowers R a little,
    where counting beats would shift every bar line after it (Second Skin, 2026-11-13)."""
    P = np.linspace(lo, hi, n)
    z = np.exp(2j * np.pi * ts[None, :] / P[:, None]).mean(1)
    i = int(np.argmax(np.abs(z)))
    return float(P[i]), float(np.abs(z[i])), float(np.angle(z[i]) / (2 * np.pi) * P[i]) % float(P[i])


def _coherent_grid(bt):
    """One constant tempo: (period, offset, R, inlier share, inlier rms seconds)."""
    med = float(np.median(np.diff(bt)))
    p, R, off = _coherence(bt, med * 0.97, med * 1.03)
    p, R, off = _coherence(bt, p * 0.999, p * 1.001)          # refine
    res = (bt - off + p / 2) % p - p / 2
    inl = np.abs(res) < INLIER_TOL * p
    rms = float(np.sqrt(np.mean(res[inl] ** 2))) if inl.any() else 1.0
    return p, off, R, float(inl.mean()), rms


def _drifting_grid(bt, dur):
    """A tempo that moves (Never Yours drifts 128 -> 132 bpm): number the tracked beats with the
    local beat length, fit a smooth curve time(beat number) with tracker slips rejected, and read
    every beat off the curve, extended to the song's ends at the end tempos. Returns (beat times,
    rms seconds of the kept beats against the curve)."""
    from scipy.interpolate import LSQUnivariateSpline
    from scipy.ndimage import median_filter

    d = np.diff(bt)
    lp = median_filter(d, size=17, mode="nearest")
    n = np.concatenate([[0.0], np.cumsum(np.maximum(1.0, np.round(d / lp)))])
    keep = np.ones(len(bt), bool)
    for _ in range(3):
        knots = np.arange(n[keep][0] + 32, n[keep][-1] - 32, 32)
        sp = LSQUnivariateSpline(n[keep], bt[keep], knots, k=3)
        res = bt - sp(n)
        keep = np.abs(res) < INLIER_TOL * np.median(lp)
    rms = float(np.sqrt(np.mean(res[keep] ** 2)))
    nn = np.arange(n[0], n[-1] + 1)
    beats = sp(nn)
    p0, p1 = float(np.median(np.diff(beats[:16]))), float(np.median(np.diff(beats[-16:])))
    head = beats[0] - p0 * np.arange(1, int(beats[0] / p0) + 1)[::-1]
    tail = beats[-1] + p1 * np.arange(1, int((dur - beats[-1]) / p1) + 1)
    return np.concatenate([head, beats, tail]), rms


def measure_grid(audio_path, beats_per_bar=4, bpm_hint=None, cache_dir=None):
    """audio -> {bpm, beat, bar (median seconds), beats_per_bar, downbeat (first bar line), steady,
    fit_rms_ms, beats, coherence, inliers, phase_scores, phase_margin, phase_windows,
    phase_agreement, duration, bar_lines: [seconds of every bar line], how}. Cached per file.
    bpm_hint: the user's tempo, the tracker's start when its own guess is wrong (half or double)."""
    bpb = int(beats_per_bar or 4)
    if not 2 <= bpb <= 7:
        raise ValueError("beats_per_bar must be 2-7")
    cpath = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cpath = os.path.join(cache_dir, f"bargrid_{_cache_key(audio_path, bpb, bpm_hint)}.json")
        if os.path.exists(cpath):
            with open(cpath) as f:
                return json.load(f)
    import librosa

    y, sr = librosa.load(audio_path, sr=SR, mono=True)
    dur = len(y) / sr
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    if bpm_hint:
        start = float(bpm_hint)
    else:
        t0, _ = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=HOP)
        start = float(np.atleast_1d(t0)[0])
        while start < TEMPO_LO:
            start *= 2
        while start > TEMPO_HI:
            start /= 2
    _, frames = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=HOP, start_bpm=start, tightness=200)
    bt = librosa.frames_to_time(frames, sr=sr, hop_length=HOP)
    if len(bt) < 4 * bpb:
        raise ValueError(f"only {len(bt)} beats found: no bar grid on this track")
    period, off, R, inl, rms = _coherent_grid(bt)
    steady = inl >= STEADY_INLIERS and rms <= STEADY_RMS
    if steady:
        first = off % period
        beats = first + period * np.arange(int((dur - first) / period) + 1)
    else:
        beats, rms = _drifting_grid(bt, dur)
        period = float(np.median(np.diff(beats)))

    # downbeat phase: harmony changes and kick hits land on bar lines
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    kick = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S[(freqs >= 20) & (freqs < 120)]), sr=sr, hop_length=HOP)
    kf = np.clip(librosa.time_to_frames(beats, sr=sr, hop_length=HOP), 0, len(kick) - 1)
    chop = 512
    C = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=chop)
    cf = np.clip(librosa.time_to_frames(beats, sr=sr, hop_length=chop), 0, C.shape[1] - 1)
    w = max(2, int(round(period * sr / chop)))          # one beat either side
    nov = np.array([np.linalg.norm(C[:, min(c + w, C.shape[1] - 1)] - C[:, max(c - w, 0)]) for c in cf])

    def _z(v):
        v = np.asarray(v, float)
        return (v - v.mean()) / (v.std() or 1.0)

    def _scores(lo, hi):
        """harmony and kick z-scores of each beat phase over beats lo..hi"""
        return (_z([nov[lo:hi][p::bpb].mean() for p in range(bpb)]),
                _z([kick[kf[lo:hi][p::bpb]].mean() for p in range(bpb)]))

    harm, kik = _scores(0, len(beats))
    score = harm + kik
    order = np.argsort(score)[::-1]
    phase = int(order[0])
    # the same vote per PHASE_WINDOW bars: does that downbeat hold through the song?
    span = PHASE_WINDOW * bpb
    wins = []
    for lo in range(0, len(beats) - span // 2, span):
        h, k_ = _scores(lo, min(lo + span, len(beats)))
        wins.append({"start": round(float(beats[lo]), 2), "phase": (int(np.argmax(h + k_)) + lo) % bpb})
    agree = sum(1 for w_ in wins if w_["phase"] == phase) / max(1, len(wins))
    bars = [float(t) for t in beats[phase::bpb]]
    local = 60.0 / np.diff(beats)
    tempo = (f"{60.0 / period:.2f} bpm constant" if steady else
             f"{np.percentile(local, 5):.1f}-{np.percentile(local, 95):.1f} bpm, tempo drifts (curve fit)")
    out = {"bpm": round(60.0 / period, 3), "beat": round(period, 5), "bar": round(period * bpb, 5),
           "beats_per_bar": bpb, "downbeat": round(bars[0], 4), "steady": bool(steady),
           "fit_rms_ms": round(1000 * rms, 1), "beats": int(len(bt)), "coherence": round(R, 3),
           "inliers": round(inl, 3), "duration": round(dur, 3),
           "phase_scores": {"harmony": [round(float(x), 2) for x in harm], "kick": [round(float(x), 2) for x in kik]},
           "phase_margin": round(float(score[order[0]] - score[order[1]]), 2),
           "phase_windows": wins, "phase_agreement": round(agree, 2),
           "bar_lines": [round(t, 4) for t in bars],
           "how": (f"{tempo}, fit {1000 * rms:.0f} ms; downbeat = beat phase {phase} of {bpb} (harmony + kick, "
                   f"margin {float(score[order[0]] - score[order[1]]):.2f}; holds in {round(100 * agree)}% of "
                   f"{PHASE_WINDOW}-bar windows)"),
           "audio": os.path.basename(audio_path)}
    if cpath:
        with open(cpath, "w") as f:
            json.dump(out, f)
    return out
