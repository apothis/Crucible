"""AI Mix measurement layer (docs/AI_MIX_PLAN.md, adaptive rules).

One feature extractor, used identically on (a) stems separated from REFERENCE songs (to build
style presets as distributions of data points) and (b) a Suno song's stems grouped into the same
six roles. Processors then act only where a Suno stem sits outside the preset's range.

Roles follow the box separator (BS-RoFormer SW): vocals, drums, bass, guitar, piano, other.
Suno's 12 stems are grouped onto those (SUNO_GROUP) so both sides are measured alike.

Every feature is in dB and documented where computed. Wetness and punch are proxies [H]: they
are validated only by whether the resulting moves sound right.
"""
import numpy as np

ROLES = ("vocals", "drums", "bass", "guitar", "piano", "other")

# Suno stem role (aimix.ROLE_PATTERNS) -> separator role
SUNO_GROUP = {
    "vocals": "vocals", "backing": "vocals",
    "drums": "drums", "percussion": "drums",
    "bass": "bass",
    "guitar": "guitar",
    "keys": "piano",
    "synth": "other", "strings": "other", "brass": "other", "woodwinds": "other", "fx": "other",
    "other": "other",
}

# third-octave centres 31.5 Hz .. 16 kHz
THIRD_OCT = [31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250,
             1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000]


def _db(x):
    return 10.0 * np.log10(np.maximum(x, 1e-20))


def _frames(m, sr, ms):
    hop = max(1, int(sr * ms / 1000.0))
    n = len(m) // hop
    return m[:n * hop].reshape(n, hop), hop


def ltas(mono, sr):
    """Third-octave long-term spectrum, dB, normalised so the bands' power sums to 0 dB.
    Shape only (level is a separate feature)."""
    from scipy.signal import welch
    f, p = welch(mono, sr, nperseg=8192)
    out = []
    for c in THIRD_OCT:
        lo, hi = c / 2 ** (1 / 6), c * 2 ** (1 / 6)
        out.append(p[(f >= lo) & (f < hi)].sum())
    out = np.array(out)
    return (_db(out) - _db(out.sum())).tolist()


def dynamics(mono, sr):
    """range_db: p95-p10 of 400 ms loudness over active frames (a loudness-range proxy).
    crest_db: median over active 100 ms frames of peak/RMS (how spiky inside a beat)."""
    seg, _ = _frames(mono, sr, 400)
    l4 = _db((seg ** 2).mean(1))
    act4 = l4 > l4.max() - 40
    seg1, _ = _frames(mono, sr, 100)
    rms1 = np.sqrt((seg1 ** 2).mean(1) + 1e-20)
    act1 = 20 * np.log10(rms1) > 20 * np.log10(rms1.max()) - 40
    crest = 20 * np.log10(np.abs(seg1).max(1)[act1] / rms1[act1]) if act1.any() else np.array([0.0])
    # within-phrase range: the same p95-p10, but inside 10 s windows (25 x 400 ms), median over
    # active windows. Arrangement builds (quiet intro, loud chorus) inflate range_db but not this,
    # so compression keys off how unevenly a part is PLAYED, not how the song is arranged.
    w = 25
    k = len(l4) // w
    pr = []
    for i in range(k):
        L, A = l4[i * w:(i + 1) * w], act4[i * w:(i + 1) * w]
        if A.sum() >= w // 2:
            pr.append(np.percentile(L[A], 95) - np.percentile(L[A], 10))
    return {"range_db": float(np.percentile(l4[act4], 95) - np.percentile(l4[act4], 10)) if act4.any() else 0.0,
            "phrase_range_db": float(np.median(pr)) if pr else None,
            "crest_db": float(np.median(crest)),
            "active_frac": float(act4.mean())}


def punch(mono, sr):
    """Attack-over-body at detected hits. A low-band (<250 Hz) detector finds hits (fast envelope
    rising 4 dB over a slow one, 80 ms refractory); per hit: peak level 0-15 ms vs RMS 30-120 ms.
    punch_db = median over hits; hits_per_s for context. Full-band levels, low-band detection."""
    from scipy.signal import butter, sosfilt
    det = sosfilt(butter(4, 250, "lowpass", fs=sr, output="sos"), mono)
    b = int(sr / 1000)
    n = len(mono) // b
    pk = np.abs(det[:n * b]).reshape(n, b).max(1)

    def follow(att, rel):
        a, r = np.exp(-1 / att), np.exp(-1 / rel)
        out = np.empty(n)
        e = 0.0
        for i in range(n):
            v = pk[i]
            e = a * e + (1 - a) * v if v > e else r * e + (1 - r) * v
            out[i] = e
        return out

    fast, slow = follow(0.5, 10), follow(30, 150)
    floor = np.percentile(pk, 60)
    on = np.where((fast > slow * 10 ** (4 / 20)) & (fast > floor))[0]
    hits, last = [], -1000
    for i in on:
        if i - last >= 80:
            hits.append(i)
            last = i
    full = np.abs(mono[:n * b]).reshape(n, b)
    vals = []
    for i in hits:
        if i + 120 >= n:
            break
        att = full[i:i + 15].max()
        body = np.sqrt((full[i + 30:i + 120] ** 2).mean())
        if body > 0:
            vals.append(20 * np.log10(att / body))
    return {"punch_db": float(np.median(vals)) if vals else None,
            "hits_per_s": float(len(hits) / (len(mono) / sr))}


def sibilance(mono, sr):
    """5-9 kHz bursts: p99 minus median of the 5-9 kHz share of energy in 20 ms frames (dB).
    High = spitty esses / harsh bursts."""
    from scipy.signal import stft
    f, _, Z = stft(mono, sr, nperseg=1024, noverlap=0, boundary=None, padded=False)
    P = np.abs(Z) ** 2
    hi = P[(f >= 5000) & (f < 9000)].sum(0)
    tot = P.sum(0) + 1e-20
    act = _db(tot) > _db(tot).max() - 40
    share = _db(hi[act] / tot[act])
    return {"sib_db": float(np.percentile(share, 99) - np.median(share)) if act.any() else 0.0}


def stereo(x, sr):
    """Side-minus-mid dB overall and below 150 Hz (low-end width)."""
    from scipy.signal import butter, sosfiltfilt
    mid, side = (x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2
    sos = butter(4, 150, "lowpass", fs=sr, output="sos")
    lm, ls = sosfiltfilt(sos, mid), sosfiltfilt(sos, side)
    return {"width_db": float(_db(np.mean(side ** 2)) - _db(np.mean(mid ** 2))),
            "low_width_db": float(_db(np.mean(ls ** 2)) - _db(np.mean(lm ** 2)))}


def wetness(mono, sr):
    """Dip depth inside active passages: p90 minus p10 of 20 ms levels within active 1 s windows,
    median over windows. Dry, gated parts dip deep between notes; reverb fills the dips -> lower
    = wetter. [H] proxy; separation bleed also fills dips, see the calibration offsets."""
    seg, _ = _frames(mono, sr, 20)
    l = _db((seg ** 2).mean(1))
    w = 50
    k = len(l) // w
    if k == 0:
        return {"dip_db": 0.0}
    L = l[:k * w].reshape(k, w)
    act = L.mean(1) > l.max() - 30
    if not act.any():
        return {"dip_db": 0.0}
    d = np.percentile(L[act], 90, axis=1) - np.percentile(L[act], 10, axis=1)
    return {"dip_db": float(np.median(d))}


def stem_features(x, sr, mix_lufs, meter):
    """All features for one stem (frames, 2) float."""
    mono = x.mean(1)
    lufs = meter.integrated_loudness(x)
    out = {"rel_lufs": float(lufs - mix_lufs) if np.isfinite(lufs) else None, "ltas": ltas(mono, sr)}
    out.update(dynamics(mono, sr))
    out.update(punch(mono, sr))
    out.update(sibilance(mono, sr))
    out.update(stereo(x, sr))
    out.update(wetness(mono, sr))
    return out


def presence_windows(voc, rest, sr, win_s=10.0):
    """presence_ratio per win_s window where the vocal is active (list of dB)."""
    n = int(win_s * sr)
    out = []
    for i in range(len(voc) // n):
        v, r = voc[i * n:(i + 1) * n], rest[i * n:(i + 1) * n]
        if np.sqrt(np.mean(v ** 2)) < 1e-4:
            continue
        p = presence_ratio(v, r, sr)
        if p is not None:
            out.append(p)
    return out


def presence_frames(voc, rest, sr):
    """Vocal-over-rest in 1-5 kHz per vocal-active 100 ms frame (dB array, may be empty)."""
    from scipy.signal import butter, sosfilt
    sos = butter(4, [1000, 5000], "bandpass", fs=sr, output="sos")
    v, r = sosfilt(sos, voc), sosfilt(sos, rest)
    sv, _ = _frames(v, sr, 100)
    sr_, _ = _frames(r, sr, 100)
    lv, lr = _db((sv ** 2).mean(1)), _db((sr_ ** 2).mean(1))
    act = lv > np.percentile(lv, 40)
    return lv[act] - lr[act]


def presence_ratio(voc, rest, sr):
    """Vocal over everything else in 1-5 kHz, median over vocal-active 100 ms frames (dB)."""
    d = presence_frames(voc, rest, sr)
    return float(np.median(d)) if len(d) else None


def presence_spread(voc, rest, sr):
    """Median minus 10th percentile of the per-frame vocal-over-rest (dB): how far the vocal's
    weak moments (soft syllables, tails, breaths) drop under the band relative to its typical
    level. Vocal compression narrows exactly this; it is independent of overall vocal level."""
    d = presence_frames(voc, rest, sr)
    return float(np.median(d) - np.percentile(d, 10)) if len(d) > 20 else None


def song_features(stems: dict, sr: int) -> dict:
    """stems: role -> (frames, 2) float arrays, all the same length (a reference song's separated
    stems, or a Suno song's grouped stems). Returns {mix: {...}, roles: {role: {...}}}."""
    import pyloudnorm as pyln
    meter = pyln.Meter(sr)
    n = min(a.shape[0] for a in stems.values())
    stems = {k: v[:n] for k, v in stems.items()}
    mix = sum(stems.values())
    mix_lufs = meter.integrated_loudness(mix)
    out = {"mix": {"lufs": float(mix_lufs), "ltas": ltas(mix.mean(1), sr)}, "roles": {}}
    out["mix"].update(dynamics(mix.mean(1), sr))
    out["mix"].update(stereo(mix, sr))
    for role, x in stems.items():
        if np.sqrt(np.mean(x ** 2)) < 1e-5:
            continue
        out["roles"][role] = stem_features(x, sr, mix_lufs, meter)
    if "vocals" in stems and "vocals" in out["roles"]:
        rest = (mix - stems["vocals"]).mean(1)
        out["roles"]["vocals"]["presence_db"] = presence_ratio(stems["vocals"].mean(1), rest, sr)
        out["roles"]["vocals"]["presence_spread_db"] = presence_spread(stems["vocals"].mean(1), rest, sr)
    return out


ABSENT_REL_DB = -24.0      # a role this far below the mix is separation residue, not a part


def present(role_feat) -> bool:
    r = role_feat.get("rel_lufs")
    return r is not None and r > ABSENT_REL_DB


def summarize(songs: list) -> dict:
    """Preset = per role, per scalar feature: median / p25 / p75 / n; LTAS as per-band median
    and IQR. Songs missing a role simply don't count for it."""
    def agg(vals):
        v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
        if not len(v):
            return None
        return {"med": float(np.median(v)), "p25": float(np.percentile(v, 25)),
                "p75": float(np.percentile(v, 75)), "n": int(len(v))}

    def agg_vec(vecs):
        A = np.array(vecs, float)
        return {"med": np.median(A, 0).tolist(), "p25": np.percentile(A, 25, 0).tolist(),
                "p75": np.percentile(A, 75, 0).tolist(), "n": int(len(A))}

    out = {"mix": {}, "roles": {}}
    keys = [k for k in songs[0]["mix"] if k != "ltas"]
    out["mix"] = {k: agg([s["mix"].get(k) for s in songs]) for k in keys}
    out["mix"]["ltas"] = agg_vec([s["mix"]["ltas"] for s in songs])
    for role in ROLES:
        rs = [s["roles"][role] for s in songs if role in s["roles"] and present(s["roles"][role])]
        if not rs:
            continue
        ks = sorted({k for r in rs for k in r if k != "ltas"})
        out["roles"][role] = {k: agg([r.get(k) for r in rs]) for k in ks}
        out["roles"][role]["ltas"] = agg_vec([r["ltas"] for r in rs])
    return out
