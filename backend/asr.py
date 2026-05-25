"""Local lyric transcription (Mac CPU, faster-whisper).

Transcribes the USER'S OWN audio file into text — used to pull lyrics from a source
song for the cover / vocal flows. This is software transcription of a file the user
possesses (not reproducing copyrighted lyrics from the web), and runs on the Mac CPU
(int8) — light, not one of the heavy GPU-only models.

For dense mixes (metal!), transcribe the Demucs-isolated VOCAL, not the raw mix —
the caller (app.py) handles isolation; here we just transcribe whatever path we're given.
"""
import threading

_MODELS = {}
_LOCK = threading.Lock()


def _get_model(size="small"):
    """Cache one WhisperModel per size (load once, reuse)."""
    with _LOCK:
        if size not in _MODELS:
            from faster_whisper import WhisperModel
            _MODELS[size] = WhisperModel(size, device="cpu", compute_type="int8")
        return _MODELS[size]


def transcribe(audio_path, size="small", language=None):
    """Transcribe an audio file to text. Returns {text, language, duration}.
    `size`: tiny/base/small/medium/large-v3 (small = good speed/quality on CPU)."""
    model = _get_model(size)
    segments, info = model.transcribe(audio_path, language=language, vad_filter=True)
    lines = []
    for s in segments:
        t = s.text.strip()
        if t:
            lines.append(t)
    return {"text": "\n".join(lines).strip(),
            "language": getattr(info, "language", None),
            "duration": float(getattr(info, "duration", 0.0) or 0.0)}
