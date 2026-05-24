"""Detect a track's real section boundaries (librosa structural segmentation)
so a symbolic guitar arrangement lands on where the music ACTUALLY changes,
instead of filling by the arrangement's nominal block lengths in order.

The gap this closes (RESEARCH.md §10h / HANDOFF "Open"): `generate_riff_arrangement`
lays each Song-Constructor block by its `seconds` end-to-end, assuming the
arrangement matches the backing. A generated/imported backing has its own
section timing — this module finds those boundaries and re-times the blocks to
them (`align_blocks`, labels/order preserved) or builds an arrangement from
scratch when there is none (`auto_blocks`, roles inferred from energy).

Mac-side, no GPU (runs alongside Demucs like the rest of the audio tools).
"""
HOP = 512
SR = 22050


def available():
    try:
        import librosa  # noqa: F401
        import sklearn   # noqa: F401  (agglomerative clustering backend)
        return True
    except Exception:
        return False


def _default_k(dur_s):
    """Pick a segment count for auto-arrangement from track length (~one section
    per ~18s), clamped to a musically sane 3–8."""
    return int(max(3, min(8, round(dur_s / 18.0))))


def detect_segments(path, k=None, sr=SR):
    """Structural segmentation → a list of `{start, end, energy}` segments
    (seconds; energy = mean RMS in the segment, 0..~1). Uses timbre (MFCC) +
    harmony (chroma) features and agglomerative clustering into `k` segments
    (k defaults from duration). Returns fewer than `k` only for very short clips.
    """
    import librosa
    import numpy as np
    y, sr = librosa.load(path, sr=sr, mono=True)
    dur = len(y) / float(sr)
    if y.size < sr:                                   # < 1s — nothing to segment
        return [{"start": 0.0, "end": dur, "energy": float(np.sqrt(np.mean(y ** 2)) if y.size else 0.0)}]

    S = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=HOP, n_mels=128)
    logS = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=logS, n_mfcc=13)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    feat = np.vstack([librosa.util.normalize(mfcc, axis=1),
                      librosa.util.normalize(chroma, axis=1)])
    nframes = feat.shape[1]
    kk = int(k) if k else _default_k(dur)
    kk = max(1, min(kk, nframes))
    if kk <= 1:
        return [{"start": 0.0, "end": dur, "energy": float(np.sqrt(np.mean(y ** 2)))}]

    bound_frames = librosa.segment.agglomerative(feat, kk)   # k segment-start frames (incl. 0)
    bound_frames = sorted(set(int(b) for b in bound_frames))
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]

    segs = []
    for i, f0 in enumerate(bound_frames):
        f1 = bound_frames[i + 1] if i + 1 < len(bound_frames) else nframes
        t0 = float(librosa.frames_to_time(f0, sr=sr, hop_length=HOP))
        t1 = float(librosa.frames_to_time(f1, sr=sr, hop_length=HOP)) if f1 < nframes else dur
        e = float(np.mean(rms[f0:f1])) if f1 > f0 else 0.0
        segs.append({"start": round(t0, 3), "end": round(t1, 3), "energy": round(e, 5)})
    segs[-1]["end"] = round(dur, 3)
    return segs


def align_blocks(blocks, path):
    """Re-time an arrangement's labeled blocks onto the backing's REAL section
    boundaries. Detects exactly `len(blocks)` segments and maps them 1:1 in order
    — each block keeps its `type` (so verse/chorus/breakdown dynamics still
    apply) but takes the detected segment's duration. Returns new
    `[{type, seconds}]` whose total == the backing length.
    """
    blocks = list(blocks or [])
    if not blocks:
        return blocks
    segs = detect_segments(path, k=len(blocks))
    if len(segs) != len(blocks):                      # short clip / degenerate — scale nominal
        return _scale_to(blocks, segs[-1]["end"] if segs else None)
    return [{"type": b.get("type", "verse"), "seconds": round(s["end"] - s["start"], 3)}
            for b, s in zip(blocks, segs)]


def auto_blocks(path):
    """Build an arrangement from scratch by detecting the backing's sections and
    inferring a role per segment from position + energy (no Song needed):
    first→intro, last→outro, loudest middles→chorus, the rest→verse. Returns
    `[{type, seconds}]`."""
    segs = detect_segments(path)
    n = len(segs)
    if n == 1:
        return [{"type": "verse", "seconds": round(segs[0]["end"] - segs[0]["start"], 3)}]
    energies = [s["energy"] for s in segs]
    mid = sorted(energies[1:-1]) if n > 2 else []
    thresh = mid[len(mid) // 2] if mid else 0.0       # median energy of the middle sections
    out = []
    for i, s in enumerate(segs):
        if i == 0:
            typ = "intro"
        elif i == n - 1:
            typ = "outro"
        else:
            typ = "chorus" if s["energy"] >= thresh else "verse"
        out.append({"type": typ, "seconds": round(s["end"] - s["start"], 3)})
    return out


def _scale_to(blocks, total):
    """Fallback: scale nominal block seconds proportionally to `total` (the
    detected track length) so they still fill the backing when segmentation
    can't yield one segment per block."""
    blocks = [{"type": b.get("type", "verse"), "seconds": float(b.get("seconds", 8) or 8)} for b in blocks]
    if not total:
        return blocks
    nom = sum(b["seconds"] for b in blocks) or 1.0
    f = float(total) / nom
    return [{"type": b["type"], "seconds": round(b["seconds"] * f, 3)} for b in blocks]
