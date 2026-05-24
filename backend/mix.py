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
        for ext in (".wav", ".mp3"):
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


def stitch(tracks, library, stems_dir, crossfade_s=1.0, target_sr=44100, normalize=True) -> bytes:
    """Concatenate tracks end-to-end (one after another) with an equal-power
    crossfade between adjacent segments. `tracks` is a list of in-app URLs in
    play order. Used by the Song Constructor per-block + stitch mode.
    """
    import math
    import io as _io
    import torch
    import soundfile as sf

    segs = [_load_stereo(_resolve(u, library, stems_dir), target_sr) for u in tracks]
    if not segs:
        raise ValueError("no tracks to stitch")

    out = segs[0]
    xf = max(0, int(float(crossfade_s) * target_sr))
    for nxt in segs[1:]:
        n = min(xf, out.shape[1], nxt.shape[1])
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
