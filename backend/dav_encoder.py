"""Encode audio into a MiniMax Music 3 flow latent - the step ComfyUI refuses to do.

ComfyUI ships the DAV DECODER only (comfy/sd.py hard-raises "MiniMax Music3 DAV cannot
encode audio"), but the encoder weights are public (SimpleTuner/MiniMax-Music-3-Encoder,
ungated) and their 121 decoder tensors are bit-identical to the minimax_music3_dav.safetensors
ComfyUI loads - same latent space. So a latent encoded HERE, on the Mac, is one the box's
sampler already understands: write it into ComfyUI's input dir and swap
EmptyMiniMaxMusic3LatentAudio for a stock LoadLatent.

Model definition adapted from AIPLAY Studio's scripts/dav_encode.py (Apache-2.0,
https://github.com/Senzube4n/AIPLAY-Studio), itself the MiniMax/HuggingFace Apache-2.0
architecture from SimpleTuner with the diffusers base classes stripped. Their measured
round trip: +26.26 dB SI-SDR, r=0.999.

Shape contract: EmptyMiniMaxMusic3LatentAudio produces (batch, 128, T) at one latent frame
per 512 samples. The encoder produces exactly that: 64 latent dims per audio channel x 2
channels = 128, downsample 2*4*8*8 = 512. L/R are encoded as two independent mono streams
through mean_proj alone (not mean+logvar).

The round-trip SI-SDR self-check is kept on purpose: AIPLAY's changelog records a packed-
stereo interleaving bug that still round-tripped at 14 dB - encoder and decoder agree on
garbage - so we sidestep the whole class by decoding through ffmpeg to a known layout, and
still report SI-SDR so the UI can warn when a reference encodes badly.
"""
import math
import os
import subprocess
import tempfile

import numpy as np

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
WEIGHTS = os.path.join(WEIGHTS_DIR, "minimax_music3_dav_encoder.safetensors")
WEIGHTS_URL = ("https://huggingface.co/SimpleTuner/MiniMax-Music-3-Encoder/resolve/main/"
               "audio_vae/diffusion_pytorch_model.safetensors")
SAMPLE_RATE = 44100
HOP = 512


def weights_present():
    return os.path.isfile(WEIGHTS)


def download_weights():
    """Fetch the encoder (about 292 MB) from HuggingFace. Ungated, plain HTTPS."""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    tmp = WEIGHTS + ".part"
    subprocess.run(["curl", "-L", "-C", "-", "-o", tmp, WEIGHTS_URL],
                   check=True, timeout=1800)
    os.replace(tmp, WEIGHTS)


# ---------------- model (DAC-style encoder + decoder for the self-check) ----------------

def _build():
    import torch
    import torch.nn as nn
    from torch.nn.utils import weight_norm

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
                Snake1d(dim),
                weight_norm(nn.Conv1d(dim, dim, 7, dilation=dilation, padding=pad)),
                Snake1d(dim),
                weight_norm(nn.Conv1d(dim, dim, 1)),
            )

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
                                      padding=math.ceil(stride / 2))),
            )

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
                ResidualUnit(oc, 1), ResidualUnit(oc, 3), ResidualUnit(oc, 9),
            )

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
        def __init__(self):
            super().__init__()
            self.hop_length = HOP
            self.latent_channels = 128
            self.channel_latent_channels = 64
            self.encoder = Encoder()
            self.mean_proj = nn.Conv1d(1024, 64, 1)
            self.logs_proj = nn.Conv1d(1024, 64, 1)
            self.dec_in_proj = nn.Conv1d(64, 1024, 1)
            self.decoder = Decoder()

        def _prep(self, w):
            if w.ndim == 1:
                w = w.unsqueeze(0).unsqueeze(0)
            elif w.ndim == 2:
                w = w.unsqueeze(0)
            if w.shape[1] == 1:
                w = w.repeat(1, 2, 1)
            rem = w.shape[-1] % self.hop_length
            if rem:
                w = torch.nn.functional.pad(w, (0, self.hop_length - rem))
            return w

        @torch.no_grad()
        def encode(self, w):
            w = self._prep(w)
            b = w.shape[0]
            h = self.encoder(w.reshape(b * 2, 1, -1))
            return self.mean_proj(h).reshape(b, self.latent_channels, -1)

        @torch.no_grad()
        def decode(self, z):
            b, _, t = z.shape
            h = z.reshape(b * 2, self.channel_latent_channels, t)
            return self.decoder(self.dec_in_proj(h)).reshape(b, 2, -1)

    return DAV()


_MODEL = None  # loaded once per process; ~1.2 GB fp32 on the Mac's RAM


def _model():
    global _MODEL
    if _MODEL is None:
        import torch
        from safetensors.torch import load_file
        if not weights_present():
            download_weights()
        m = _build()
        sd = load_file(WEIGHTS)
        inc = m.load_state_dict(sd, strict=False)
        unexpected = [k for k in inc.unexpected_keys if not k.startswith("flow.")]
        if inc.missing_keys or unexpected:
            raise RuntimeError(f"DAV encoder state dict mismatch: missing {inc.missing_keys[:3]} "
                               f"unexpected {unexpected[:3]}")
        _MODEL = m.eval()
    return _MODEL


def _load_stereo_44k(path):
    """Decode ANY audio to (1, 2, N) float32 at 44.1 kHz via ffmpeg.

    ffmpeg does the resample and the channel layout, so the packed-vs-planar stereo trap
    that produced two wrong conclusions in AIPLAY's changelog cannot occur here.
    """
    import soundfile as sf
    import torch
    mvwork = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".mvwork")
    with tempfile.TemporaryDirectory(dir=mvwork if os.path.isdir(mvwork) else None) as td:
        wav = os.path.join(td, "ref.wav")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                        "-ar", str(SAMPLE_RATE), "-ac", "2", "-c:a", "pcm_f32le", wav],
                       check=True, timeout=300)
        a, sr = sf.read(wav, dtype="float32", always_2d=True)   # (N, 2)
    return torch.from_numpy(np.ascontiguousarray(a.T))[None], sr


def encode_to_latent(src_path, max_seconds=0.0):
    """Encode an audio file; returns (latent_file_bytes, info dict).

    The bytes are a ComfyUI .latent (safetensors) the stock LoadLatent node reads directly;
    `latent_format_version_0` must be present or LoadLatent rescales by 1/0.18215.
    """
    import torch
    from safetensors.torch import save_file

    m = _model()
    wav, sr = _load_stereo_44k(src_path)
    if max_seconds and max_seconds > 0:
        wav = wav[..., : int(max_seconds * sr)]

    z = m.encode(wav)

    # Round-trip self-check on a short window only: decoding minutes of stereo allocates
    # multi-GB activations for a number that a 15 s window measures just as well.
    check_frames = min(z.shape[-1], int(15 * sr / HOP))
    rec = m.decode(z[..., :check_frames])
    ref = wav[0, 0, : check_frames * HOP].numpy()
    est = rec[0, 0].numpy()
    ref = ref - ref.mean()
    est = est[: len(ref)] - est[: len(ref)].mean()
    a = float(np.dot(est, ref) / (np.dot(ref, ref) + 1e-12))
    noise = est - a * ref
    si_sdr = float(10 * np.log10((np.dot(a * ref, a * ref) + 1e-12)
                                 / (np.dot(noise, noise) + 1e-12)))

    with tempfile.NamedTemporaryFile(suffix=".latent", delete=False) as f:
        tmp = f.name
    try:
        save_file({"latent_tensor": z.contiguous(),
                   "latent_format_version_0": torch.tensor([])}, tmp)
        with open(tmp, "rb") as f:
            blob = f.read()
    finally:
        os.unlink(tmp)

    return blob, {
        "frames": int(z.shape[-1]),
        "seconds": round(z.shape[-1] * HOP / sr, 3),
        "si_sdr_db": round(si_sdr, 2),
        # >15 dB means the encode is trustworthy (AIPLAY's threshold); below it the
        # reference will steer the render toward a degraded version of itself.
        "encode_ok": si_sdr > 15.0,
    }
