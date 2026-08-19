"""In-app mixer — layer multiple tracks (e.g. instrumental + converted vocal)
with per-track gain and timing offset, then bounce a stereo WAV. Pure Mac-side
(soundfile + torchaudio for resampling); no external tools.

Sources are referenced by their in-app URL so the frontend can pass what it
already has: `/api/audio/<id>` (library) or `/api/stem/<sid>/<name>` (stems).
"""
import io
import os


def _resolve(url: str, library: str, stems_dir: str) -> str:
    parts = url.strip("/").split("/")
    if url.startswith("/api/audio/"):
        jid = parts[-1]
        for ext in (".wav", ".mp3", ".flac"):
            p = os.path.join(library, jid + ext)
            if os.path.exists(p):
                return p
    elif url.startswith("/api/stem/"):
        p = os.path.join(stems_dir, os.path.basename(parts[-2]), os.path.basename(parts[-1]))
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"cannot resolve source: {url}")


def _load_stereo(path, target_sr):
    import soundfile as sf
    import torch
    import torchaudio
    data, sr = sf.read(path, dtype="float32", always_2d=True)  # (frames, ch)
    x = torch.from_numpy(data.T)  # (ch, frames)
    if x.shape[0] == 1:
        x = x.repeat(2, 1)
    elif x.shape[0] > 2:
        x = x[:2]
    if sr != target_sr:
        x = torchaudio.functional.resample(x, sr, target_sr)
    return x


def mix(tracks, library, stems_dir, target_sr=44100, normalize=True) -> bytes:
    import torch
    import torch.nn.functional as F
    import soundfile as sf

    loaded = []
    for t in tracks:
        x = _load_stereo(_resolve(t["src"], library, stems_dir), target_sr)
        x = x * (10.0 ** (float(t.get("gain_db", 0.0)) / 20.0))  # gain in dB
        off = int(float(t.get("offset", 0.0)) * target_sr)        # start offset (s)
        if off > 0:
            x = F.pad(x, (off, 0))
        loaded.append(x)
    if not loaded:
        raise ValueError("no tracks to mix")

    length = max(x.shape[1] for x in loaded)
    out = torch.zeros(2, length)
    for x in loaded:
        out[:, :x.shape[1]] += x

    if normalize:
        peak = out.abs().max().item()
        if peak > 0.97:
            out = out * (0.97 / peak)  # prevent clipping, keep relative levels

    buf = io.BytesIO()
    sf.write(buf, out.T.numpy(), target_sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _first_onset_sample(seg, sr, max_trim_s=0.7):
    """First strong onset in a clip (samples), capped — used to trim the arbitrary
    pre-roll so a clip starts on a beat. Returns 0 if detection fails."""
    try:
        import librosa
        mono = seg.mean(0).numpy().astype("float32")
        on = librosa.onset.onset_detect(y=mono, sr=sr, units="samples", backtrack=True)
        if len(on):
            return int(min(int(on[0]), int(max_trim_s * sr)))
    except Exception:
        pass
    return 0


def _trim_for_stitch(seg, sr, max_lead_s=0.7, tail_pad_s=0.25):
    """Trim a clip's dead lead-in (to its first onset) and trailing near-silence so
    sections connect with continuous music instead of dead air. ACE-Step often ends
    its phrase a few seconds before the requested duration and pads the rest quiet;
    this removes that pad (keeping a short decay tail). Returns the trimmed clip."""
    n = seg.shape[1]
    if n <= int(1.0 * sr):
        return seg
    start = _first_onset_sample(seg, sr, max_lead_s)
    try:
        import numpy as np
        m = seg.abs().amax(dim=0).numpy()
        body = float(m[: int(n * 0.85)].max()) if int(n * 0.85) else float(m.max())
        end = n
        if body > 0:
            above = np.where(m > body * 0.02)[0]
            if above.size:
                end = min(n, int(above[-1]) + int(tail_pad_s * sr))
        return seg[:, start:end] if end > start + int(0.1 * sr) else seg
    except Exception:
        return seg[:, start:] if start > 0 else seg


def _beat_samples(seg, sr):
    """Beat positions (samples) of a clip via librosa; [] on failure."""
    try:
        import librosa
        mono = seg.mean(0).numpy().astype("float32")
        _, beats = librosa.beat.beat_track(y=mono, sr=sr, units="samples")
        return [int(b) for b in beats]
    except Exception:
        return []


def stitch(tracks, library, stems_dir, crossfade_s=1.0, target_sr=44100, normalize=True,
           bpm=None, beat_align=False) -> bytes:
    """Concatenate tracks end-to-end (one after another) with an equal-power
    crossfade between adjacent segments. `tracks` is a list of in-app URLs in
    play order. Used by the Song Constructor per-block + stitch mode.

    When `beat_align` is set (and `bpm` is known), each clip's pre-roll is trimmed
    to its first onset so it starts on a beat, the crossfade is snapped to ~one beat,
    and the splice is anchored to the outgoing clip's nearest detected beat — so the
    rhythmic grid carries across section seams instead of phasing.
    """
    import math
    import io as _io
    import torch
    import soundfile as sf

    segs = [_load_stereo(_resolve(u, library, stems_dir), target_sr) for u in tracks]
    if not segs:
        raise ValueError("no tracks to stitch")

    beat = (60.0 / float(bpm)) if (beat_align and bpm and float(bpm) > 0) else None
    if beat_align:
        # start each clip on a beat AND drop the trailing near-silence ACE pads on, so
        # sections butt up with continuous music instead of dead air at every seam.
        segs = [_trim_for_stitch(s, target_sr) for s in segs]

    out = segs[0]
    default_xf = max(0, int(float(crossfade_s) * target_sr))
    for nxt in segs[1:]:
        if beat:
            want = max(1, round(float(crossfade_s) / beat)) if crossfade_s > 0 else 1
            xf = int(want * beat * target_sr)
            # anchor the overlap start to the outgoing clip's nearest beat, but only if
            # that beat is within ±half a crossfade of the target (else keep the musical length).
            tgt = out.shape[1] - xf
            bs = [b for b in _beat_samples(out, target_sr) if 0 < b < out.shape[1]]
            if bs:
                pos = min(bs, key=lambda b: abs(b - tgt))
                if abs(pos - tgt) <= xf // 2:
                    xf = out.shape[1] - pos
            xf = min(xf, out.shape[1], nxt.shape[1])
        else:
            xf = min(default_xf, out.shape[1], nxt.shape[1])
        n = max(0, xf)
        if n > 0:
            t = torch.linspace(0.0, 1.0, n)
            fade_out = torch.cos(t * math.pi / 2)  # 1 -> 0
            fade_in = torch.sin(t * math.pi / 2)   # 0 -> 1
            blend = out[:, -n:] * fade_out + nxt[:, :n] * fade_in
            out = torch.cat([out[:, :-n], blend, nxt[:, n:]], dim=1)
        else:
            out = torch.cat([out, nxt], dim=1)

    if normalize:
        peak = out.abs().max().item()
        if peak > 0.97:
            out = out * (0.97 / peak)

    buf = _io.BytesIO()
    sf.write(buf, out.T.numpy(), target_sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()
