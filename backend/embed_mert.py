"""Mac-side MERT music embedding (m-a-p/MERT-v1-95M) for artist-fidelity scoring.

Why: CLAP-centroid failed validation (METAL_LORA_PLAN §13d) on two counts -- it could not
separate the artist from same-genre (AUC 0.54), and its embedding was non-deterministic
(laion_clap random-crops a window -> retest self-cosine 0.58-0.88). MERT is a music-trained
SSL encoder (HuBERT-style) and should have the timbre/artist resolution CLAP lacks. This
module also embeds DETERMINISTICALLY: fixed, evenly-spaced full-track windows, mean-pooled
over time and layers -> one L2-normalized vector. Same file in => same vector out.

Runs on the Mac (MPS, CPU fallback). Light: 95M params, ~200-400 MB. NOT a heavy separation
model (those crash MPS) -- this is smaller than the Demucs we already run here.

Caches: relies on HF_HOME/TORCH_HOME being pinned to SSD1 by run.sh. We also hard-set them
to the repo .caches if unset, so nothing ever lands on the system disk even if called
outside run.sh.
"""
from __future__ import annotations

import os

# Belt-and-braces: pin HF/torch caches to SSD1 (repo .caches) BEFORE importing transformers,
# in case this module is imported outside run.sh's environment.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHES = os.path.join(_ROOT, ".caches")
os.environ.setdefault("HF_HOME", os.path.join(_CACHES, "hf"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(_CACHES, "hf", "hub"))
os.environ.setdefault("TORCH_HOME", os.path.join(_CACHES, "torch"))
os.environ.setdefault("TMPDIR", os.path.join(_CACHES, "tmp"))
for _d in (os.environ["HF_HOME"], os.environ["HUGGINGFACE_HUB_CACHE"],
           os.environ["TORCH_HOME"], os.environ["TMPDIR"]):
    os.makedirs(_d, exist_ok=True)

MODEL_ID = os.environ.get("MG_MERT_MODEL", "m-a-p/MERT-v1-95M")
TARGET_SR = 24000
WIN_SECONDS = 5.0          # MERT pre-training context
MAX_WINDOWS = 12           # evenly spaced across the track -> deterministic + representative

_model = None
_processor = None
_device = None


def _pick_device():
    import torch
    forced = os.environ.get("MG_MERT_DEVICE", "").strip()
    if forced:
        return forced
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _ensure_model():
    global _model, _processor, _device
    if _model is not None:
        return
    import torch
    from transformers import AutoModel, Wav2Vec2FeatureExtractor
    _processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID, trust_remote_code=True)
    _model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    _model.eval()
    _device = _pick_device()
    try:
        _model.to(_device)
    except Exception as e:                       # MPS op gap -> fall back to CPU
        print(f"[embed_mert] {_device} load failed ({e}); falling back to cpu")
        _device = "cpu"
        _model.to("cpu")


def _windows(y, sr):
    """Fixed-length, evenly-spaced windows (deterministic). Returns a list of equal-length
    float32 arrays. Short tracks -> a single zero-padded window."""
    import numpy as np
    win = int(WIN_SECONDS * sr)
    n = len(y)
    if n <= win:
        out = np.zeros(win, dtype=np.float32)
        out[:n] = y
        return [out]
    if n < win * MAX_WINDOWS:
        k = max(1, n // win)                      # non-overlapping for short-ish tracks
    else:
        k = MAX_WINDOWS
    # evenly spaced start offsets so the last window ends at the track end
    starts = np.linspace(0, n - win, k).astype(int)
    return [y[s:s + win].astype(np.float32) for s in starts]


def embed(path, max_windows=None):
    """Deterministic L2-normalized MERT embedding (list of floats) for one audio file.

    Mean over time, mean over all hidden-state layers, mean over evenly-spaced windows.
    Same file -> same vector (no random cropping)."""
    import numpy as np
    import torch
    import librosa
    _ensure_model()
    global MAX_WINDOWS
    if max_windows:
        _old, MAX_WINDOWS = MAX_WINDOWS, int(max_windows)
    try:
        y, _ = librosa.load(path, sr=TARGET_SR, mono=True)
        wins = _windows(np.asarray(y, dtype=np.float32), TARGET_SR)
        inputs = _processor(wins, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        inputs = {k: v.to(_device) for k, v in inputs.items()}
        with torch.no_grad():
            out = _model(**inputs, output_hidden_states=True)
        # hidden_states: tuple(L) of [B, T, H] -> [L, B, T, H]
        hs = torch.stack(out.hidden_states, dim=0).float().cpu()
        vec = hs.mean(dim=2).mean(dim=0).mean(dim=0)      # time -> layers -> windows
        v = vec.numpy().astype(np.float64)
        v = v / (np.linalg.norm(v) + 1e-9)
        return [float(x) for x in v]
    finally:
        if max_windows:
            MAX_WINDOWS = _old


def unload():
    """Drop the model + free MPS/CPU memory (mirror the box services' tidiness)."""
    global _model, _processor
    _model = None
    _processor = None
    try:
        import gc
        import torch
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def info():
    return {"model": MODEL_ID, "device": _device, "loaded": _model is not None,
            "win_seconds": WIN_SECONDS, "max_windows": MAX_WINDOWS,
            "hf_home": os.environ.get("HF_HOME")}
