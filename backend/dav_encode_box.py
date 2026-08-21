"""Box-side DAV encoder: audio file -> MiniMax Music 3 .latent, on the 3090.

Deployed by the Mac (fs/write) into ComfyUI_windows_portable\\ and run by the :5080 helper's
/dav/encode endpoint under ComfyUI's own embedded python (torch cu130 + av + torchaudio all
present). Standalone on purpose: no imports from ComfyUI or the helper.

    python dav_encode_box.py IN_AUDIO OUT_LATENT CKPT [MAX_SECONDS]

The LAST line of stdout is one JSON object; prose goes to stderr (the helper parses that line).

Model definition adapted from AIPLAY Studio's scripts/dav_encode.py (Apache-2.0), itself the
MiniMax/HuggingFace Apache-2.0 architecture from SimpleTuner with the diffusers base classes
stripped. The audio loader follows AIPLAY's _audio_io: PyAV PACKED stereo arrives from
to_ndarray() as (1, N*ch) interleaved, and reading it as mono silently doubles the length -
their changelog records that bug round-tripping at 14 dB, so the layout handling here is
copied from their fixed version, and we still report round-trip SI-SDR as the tripwire.

Encoding is CHUNKED (60s cores, 128-latent-frame overlap trimmed at interior seams) so peak
activation memory stays ~2-3 GB however long the track is - this must never OOM a box that
may be holding the music stack. Interior frames get a full receptive field (the encoder's is
far under the 65536-sample overlap), so chunked output matches a full pass.
"""
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

HOP = 512
SR = 44100
PAD = 65536              # samples of context each side of a chunk core (128 latent frames)
CORE = 5168 * HOP        # ~60s per chunk core, hop-aligned


def log(*a):
    print(*a, file=sys.stderr, flush=True)


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return x + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * x).pow(2)


class ResidualUnit(nn.Module):
    def __init__(self, dim, dilation):
        super().__init__()
        pad = 3 * dilation
        self.block = nn.Sequential(
            Snake1d(dim), weight_norm(nn.Conv1d(dim, dim, 7, dilation=dilation, padding=pad)),
            Snake1d(dim), weight_norm(nn.Conv1d(dim, dim, 1)))

    def forward(self, x):
        r = self.block(x)
        if r.shape[-1] != x.shape[-1]:
            p = (x.shape[-1] - r.shape[-1]) // 2
            x = x[..., p: x.shape[-1] - p]
        return x + r


class EncoderBlock(nn.Module):
    def __init__(self, dim, stride):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, 1), ResidualUnit(dim // 2, 3), ResidualUnit(dim // 2, 9),
            Snake1d(dim // 2),
            weight_norm(nn.Conv1d(dim // 2, dim, 2 * stride, stride=stride,
                                  padding=math.ceil(stride / 2))))

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, encoder_dim=64, encoder_rates=(2, 4, 8, 8), latent_dim=1024):
        super().__init__()
        blk = [weight_norm(nn.Conv1d(1, encoder_dim, 7, padding=3))]
        d = encoder_dim
        for s in encoder_rates:
            d *= 2
            blk.append(EncoderBlock(d, stride=s))
        blk += [Snake1d(d), weight_norm(nn.Conv1d(d, latent_dim, 3, padding=1))]
        self.block = nn.Sequential(*blk)

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, ic, oc, stride):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(ic),
            weight_norm(nn.ConvTranspose1d(ic, oc, 2 * stride, stride=stride,
                                           padding=math.ceil(stride / 2))),
            ResidualUnit(oc, 1), ResidualUnit(oc, 3), ResidualUnit(oc, 9))

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=1536, upsampling_ratios=(8, 8, 4, 2)):
        super().__init__()
        layers = [weight_norm(nn.Conv1d(input_dim, hidden_dim, 7, padding=3))]
        out = hidden_dim
        for i, s in enumerate(upsampling_ratios):
            ic = hidden_dim // (2 ** i)
            out = hidden_dim // (2 ** (i + 1))
            layers.append(DecoderBlock(ic, out, stride=s))
        layers += [Snake1d(out), weight_norm(nn.Conv1d(out, 1, 7, padding=3)), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DAV(nn.Module):
    """128 latent channels = 64 per audio channel x 2; L/R encoded as independent mono
    streams through mean_proj alone (not mean+logvar)."""
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.mean_proj = nn.Conv1d(1024, 64, 1)
        self.logs_proj = nn.Conv1d(1024, 64, 1)
        self.dec_in_proj = nn.Conv1d(64, 1024, 1)
        self.decoder = Decoder()

    @torch.no_grad()
    def encode(self, w):                      # w: (1, 2, N), N a multiple of HOP
        b = w.shape[0]
        h = self.encoder(w.reshape(b * 2, 1, -1))
        return self.mean_proj(h).reshape(b, 128, -1)

    @torch.no_grad()
    def decode(self, z):
        b, _, t = z.shape
        h = z.reshape(b * 2, 64, t)
        return self.decoder(self.dec_in_proj(h)).reshape(b, 2, -1)


def load_stereo_44k(path):
    """(1, 2, N) float32 at 44.1 kHz via PyAV, planar/packed handled explicitly."""
    import av
    c = av.open(path)
    st = c.streams.audio[0]
    ch = st.codec_context.channels
    sr = st.codec_context.sample_rate
    parts = []
    for f in c.decode(audio=0):
        a = f.to_ndarray().astype(np.float32)
        if f.format.is_planar:
            a = a if a.ndim == 2 else a.reshape(1, -1)   # already (ch, N)
        else:
            a = a.reshape(-1, ch).T                      # (1, N*ch) interleaved -> (ch, N)
        parts.append(a)
    c.close()
    a = np.concatenate(parts, axis=-1)
    if np.abs(a).max() > 1.5:
        a = a / 32768.0
    if a.shape[0] == 1:
        a = np.repeat(a, 2, 0)
    elif a.shape[0] > 2:
        a = a[:2]
    w = torch.from_numpy(np.ascontiguousarray(a))[None]
    if sr != SR:
        import torchaudio
        w = torchaudio.functional.resample(w[0], sr, SR)[None]
        log(f"resampled {sr} -> {SR}")
    return w


def encode_chunked(m, wav, device):
    """Chunked encode with overlap-trim; equivalent to a full pass but bounded memory."""
    n = wav.shape[-1]
    zs = []
    start = 0
    while start < n:
        end = min(n, start + CORE)
        s0 = max(0, start - PAD)
        s1 = min(n, end + PAD)
        seg = wav[..., s0:s1].to(device)
        rem = seg.shape[-1] % HOP
        if rem:
            seg = torch.nn.functional.pad(seg, (0, HOP - rem))
        z = m.encode(seg).cpu()
        lead = (start - s0) // HOP                       # context frames to drop at the front
        want = math.ceil((end - start) / HOP)            # frames this chunk's core owns
        zs.append(z[..., lead: lead + want])
        start = end
    return torch.cat(zs, dim=-1)


def main():
    src, out, ckpt = sys.argv[1], sys.argv[2], sys.argv[3]
    max_seconds = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device: {device}")

    from safetensors.torch import load_file, save_file
    m = DAV()
    inc = m.load_state_dict(load_file(ckpt), strict=False)
    unexpected = [k for k in inc.unexpected_keys if not k.startswith("flow.")]
    if inc.missing_keys or unexpected:
        raise SystemExit(f"state dict mismatch: missing {inc.missing_keys[:3]} unexpected {unexpected[:3]}")
    m = m.to(device).eval()

    wav = load_stereo_44k(src)
    if max_seconds > 0:
        wav = wav[..., : int(max_seconds * SR)]
    log(f"input: {tuple(wav.shape)} ({wav.shape[-1]/SR:.1f}s)")

    z = encode_chunked(m, wav, device)
    log(f"latent: {tuple(z.shape)}")

    # Round-trip tripwire on the first 15s only (a full decode allocates multi-GB for a
    # number a window measures just as well).
    cf = min(z.shape[-1], int(15 * SR / HOP))
    rec = m.decode(z[..., :cf].to(device)).cpu()
    ref = wav[0, 0, : cf * HOP].numpy()
    est = rec[0, 0, : len(ref)].numpy()
    ref = ref - ref.mean(); est = est - est.mean()
    a = float(np.dot(est, ref) / (np.dot(ref, ref) + 1e-12))
    noise = est - a * ref
    si_sdr = float(10 * np.log10((np.dot(a * ref, a * ref) + 1e-12)
                                 / (np.dot(noise, noise) + 1e-12)))
    log(f"round-trip SI-SDR: {si_sdr:.2f} dB (>15 = trustworthy)")

    # ComfyUI's own .latent format; latent_format_version_0 must be present or the stock
    # LoadLatent node rescales by 1/0.18215.
    save_file({"latent_tensor": z.contiguous(),
               "latent_format_version_0": torch.tensor([])}, out)
    print(json.dumps({"ok": True, "latent": os.path.basename(out), "frames": int(z.shape[-1]),
                      "seconds": round(z.shape[-1] * HOP / SR, 3),
                      "si_sdr_db": round(si_sdr, 2), "encode_ok": si_sdr > 15.0,
                      "device": device}))


if __name__ == "__main__":
    main()
