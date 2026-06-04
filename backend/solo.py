"""Compose + render a guitar SOLO onto a chosen region of an existing track.

No source separation is used (the user can't reliably demux full mixes). Instead we
compose a CLEAN lead line for the region, render it to DI, amp it, and OVERLAY it on
the original at the region's time offset -- level-matched, edge-faded, with optional
ducking of the backing under the solo.

Reuses: guitar.compose_riff(part="solo") + render_di_file (compose/DI), postfx.process_stem
(amp), analyze._detect_key + librosa (region bpm/key). Only the orchestration, the
region analysis slice, and the level-matched overlay are new here.
"""

import os
import numpy as np
import soundfile as sf

from . import guitar as guitar_mod
from . import postfx as postfx_mod
from . import analyze as analyze_mod


# ---------------- audio helpers ----------------
def _load_stereo(path, sr_target=44100):
    """Load as float32 stereo. sr_target=None keeps the file's native sample rate
    (used when overlaying onto the original so we don't resample the user's track)."""
    x, sr = sf.read(path, always_2d=True, dtype="float32")
    if sr_target and sr != sr_target:
        import librosa
        x = np.stack([librosa.resample(x[:, c], orig_sr=sr, target_sr=sr_target)
                      for c in range(x.shape[1])], axis=1)
        sr = sr_target
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    elif x.shape[1] > 2:
        x = x[:, :2]
    return x.astype(np.float32), sr


def _highpass(x, sr, hz):
    from scipy.signal import butter, sosfilt
    sos = butter(2, hz / (sr / 2.0), btype="high", output="sos")
    return sosfilt(sos, x, axis=0).astype(np.float32)


def _edge_fade(x, sr, fade_ms):
    f = max(1, int(fade_ms / 1000.0 * sr))
    if 2 * f >= len(x):
        return x
    ramp = np.sqrt(np.linspace(0, 1, f))[:, None]   # equal-power
    x[:f] *= ramp
    x[-f:] *= ramp[::-1]
    return x


# ---------------- region analysis ----------------
def analyze_region(path, start, end):
    """Detect bpm + key for [start, end]. BPM comes from the WHOLE track (short
    regions give unreliable tempo); key comes from the region slice."""
    import librosa
    y, sr = librosa.load(path, mono=True)
    n = len(y)
    a = max(0, int(float(start) * sr))
    b = min(n, int(float(end) * sr)) if end and float(end) > 0 else n
    region = y[a:b] if b > a else y
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    try:
        key = analyze_mod._detect_key(region, sr)
    except Exception:
        key = "E minor"
    return {"bpm": round(bpm, 1), "key": key, "duration": round(len(region) / sr, 3)}


# ---------------- region slice (for the ACE LM "ears") ----------------
def slice_region_wav(path, start, end):
    """Slice [start, end] of a track to in-memory WAV bytes (stereo) for uploading
    to the engine's audio-understanding endpoint. The engine re-normalizes SR/channels."""
    import io
    x, sr = _load_stereo(path)
    a = max(0, int(float(start) * sr))
    b = min(len(x), int(float(end) * sr)) if end and float(end) > 0 else len(x)
    seg = x[a:b] if b > a else x
    buf = io.BytesIO()
    sf.write(buf, seg, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ---------------- compose ----------------
def compose(key, bpm, duration_s, genre="", brain="algorithmic",
            provider="", model="", seed=None, context=""):
    """Compose a solo. `context` = an audio-grounded section description (ACE LM ears),
    only used by the LLM brains. Returns (notes_tuples, score_dict)."""
    notes = guitar_mod.compose_riff(brain, key=key, bpm=float(bpm),
                                    duration_s=float(duration_s), genre=genre,
                                    part="solo", provider=provider, model=model, seed=seed,
                                    context=context)
    score = notes_to_score(notes, bpm, key, duration_s, brain, provider)
    return notes, score


def notes_to_score(notes, bpm, key, duration_s, brain="", provider=""):
    """Convert guitar.py (pitch,start,dur,vel) tuples into the Vocal-Builder score
    shape so the existing SVG piano-roll renders it unchanged (one 'Solo' section,
    empty syllables)."""
    out = []
    for n in notes:
        pitch, start, dur, vel = n[0], n[1], n[2], n[3]
        out.append({"midi": int(pitch), "start": round(float(start), 4),
                    "dur": round(float(dur), 4), "vel": int(vel),
                    "artic": (n[4] if len(n) > 4 else ""),
                    "section": 0, "syllable": ""})
    dur = float(duration_s) or max((n["start"] + n["dur"] for n in out), default=0.0)
    dur = round(dur, 3)
    return {"bpm": bpm, "key": key, "duration": dur,
            "provider": provider or brain or "algorithmic",
            "sections": [{"role": "Solo", "start": 0.0, "seconds": dur,
                          "lyrics": "", "notes": out}],
            "notes": out}


def score_to_notes(score):
    out = []
    for n in (score or {}).get("notes", []):
        tup = (int(n["midi"]), float(n["start"]), float(n["dur"]), int(n.get("vel", 100)))
        artic = n.get("artic", "")
        out.append(tup + (artic,) if artic else tup)
    return out


# ---------------- render ----------------
def _fit_region(src_path, out_path, region_len, fade_ms):
    """Load `src_path`, trim/pad to region_len seconds, edge-fade, write to out_path."""
    x, sr = _load_stereo(src_path)
    target = int(float(region_len) * sr)
    if target <= 0:
        target = len(x)
    if len(x) < target:
        x = np.pad(x, ((0, target - len(x)), (0, 0)))
    else:
        x = x[:target]
    x = _edge_fade(x, sr, fade_ms)
    sf.write(out_path, x, sr, subtype="PCM_16")
    return out_path


def render_di_clip(notes, out_path, region_len, engine="ks", sf2_path=None,
                   kontakt_path=None, kontakt_state=None, fade_ms=40):
    """STAGE A: notes -> clean DI (NO amp) -> trim/pad to region_len -> edge-fade.
    Writes a standalone, region-length stereo 44.1k WAV (the raw composed solo)."""
    di_raw = out_path + ".diraw.wav"
    guitar_mod.render_di_file(notes, di_raw, engine=engine, sf2_path=sf2_path,
                              kontakt_path=kontakt_path, kontakt_state=kontakt_state)
    try:
        _fit_region(di_raw, out_path, region_len, fade_ms)
    finally:
        try:
            os.remove(di_raw)
        except OSError:
            pass
    return out_path


def amp_clip(di_path, out_path, region_len, amp_preset="", cfg=None, fade_ms=40):
    """STAGE B: take a (region-length) DI clip -> apply amp/tone -> re-fit to
    region_len -> edge-fade. With no amp this is a faded passthrough of the DI so
    the stage still yields a consistent, auditionable clip."""
    src = di_path
    if amp_preset and amp_preset not in ("none", "") and postfx_mod.available():
        amped = out_path + ".amp.wav"
        try:
            postfx_mod.process_stem(di_path, amped, amp_preset, cfg=cfg)
            src = amped
        except Exception as e:
            # Don't silently masquerade clean DI as an amped clip (that hid a real
            # Helix-on-worker-thread failure); surface it so the stage shows the error.
            raise RuntimeError(f"amp '{amp_preset}' failed: {e}") from e
    try:
        _fit_region(src, out_path, region_len, fade_ms)
    finally:
        if src != di_path:
            try:
                os.remove(src)
            except OSError:
                pass
    return out_path


def render_clip(notes, out_path, region_len, engine="ks", sf2_path=None,
                kontakt_path=None, kontakt_state=None, amp_preset="", cfg=None,
                fade_ms=40):
    """Do-all convenience: notes -> DI -> (optional amp) -> region-length clip.
    Composes the two staged helpers. Returns (out_path, di_path)."""
    di_path = out_path + ".di.wav"
    render_di_clip(notes, di_path, region_len, engine=engine, sf2_path=sf2_path,
                   kontakt_path=kontakt_path, kontakt_state=kontakt_state, fade_ms=fade_ms)
    amp_clip(di_path, out_path, region_len, amp_preset=amp_preset, cfg=cfg, fade_ms=fade_ms)
    return out_path, di_path


# ---------------- overlay onto the original ----------------
def overlay(original_path, solo_path, region_start, out_path,
            solo_gain_db=0.0, auto_match=True, duck_db=0.0, fade_ms=120,
            highpass_hz=0.0, normalize=True):
    """Place the (region-length, edge-faded) solo clip onto the original at
    region_start seconds. auto_match scales the solo to the region's RMS (a lead
    should sit ABOVE the backing, so solo_gain_db is a boost on top of the match);
    duck_db dips the backing under the solo within the region; highpass_hz cleans
    solo mud. The original's native sample rate is preserved (no downsample)."""
    orig, sr = _load_stereo(original_path, sr_target=None)   # keep the track's native SR
    solo, _ = _load_stereo(solo_path, sr_target=sr)          # bring the solo clip to it
    a = max(0, int(float(region_start) * sr))
    b = min(len(orig), a + len(solo))
    if b <= a:
        sf.write(out_path, orig, sr, subtype="PCM_16")
        return out_path
    seg = solo[: b - a].copy()
    if highpass_hz and float(highpass_hz) > 0:
        seg = _highpass(seg, sr, float(highpass_hz))
    gain = 10.0 ** (float(solo_gain_db) / 20.0)
    if auto_match:
        ref = orig[a:b]
        r_ref = float(np.sqrt(np.mean(ref ** 2))) if len(ref) else 0.0
        r_solo = float(np.sqrt(np.mean(seg ** 2))) or 1e-9
        if r_ref > 0:
            gain *= (r_ref / r_solo)
    seg = seg * gain
    out = orig.copy()
    if duck_db and float(duck_db) < 0:
        duck = 10.0 ** (float(duck_db) / 20.0)
        f = max(1, int(float(fade_ms) / 1000.0 * sr))
        f = min(f, (b - a) // 2 or 1)
        env = np.full(b - a, duck, dtype=np.float32)
        env[:f] = np.linspace(1.0, duck, f)
        env[-f:] = np.linspace(duck, 1.0, f)
        out[a:b] *= env[:, None]
    out[a:b] += seg
    if normalize:
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > 0.999:
            out = out * (0.999 / peak)
    sf.write(out_path, out.astype(np.float32), sr, subtype="PCM_16")
    return out_path
