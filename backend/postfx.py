"""Guitar-stem tone shaping + recombine — the post-processing "Guitar Tone" stage.

Runs LOCALLY on the Mac (Spotify `pedalboard`), in parallel with the 3090.
The realistic job here is to *reshape* an already-distorted guitar stem (from
Demucs `htdemucs_6s`), not to re-amp from a clean DI — see RESEARCH.md §10a/§10f:
- native presets = surgical EQ (tighten lows, tame fizz) + saturation + glue comp,
  optionally an IR cab convolution.
- optional Line 6 **Helix Native** insert (VST3/AU via pedalboard) when configured
  (`helix_vst3_path`); use a cab/EQ/comp preset with the amp block BYPASSED.

Helix amp blocks (and NAM) want a clean DI, which a generated/separated stem
isn't — so the default chains deliberately do not "add an amp".
"""
import io
import os


def tidy_ending(in_path):
    """Clean up an ACE-Step end-of-song artifact (a loud clipping burst right at
    the end of the musical content — pronounced on xl_sft). Applied ONLY when the
    take actually clips or has an end-burst, so clean takes pass through untouched.

    When triggered: trim the trailing near-silence, fade out the last ~0.4 s (so a
    terminal spike can't pop), then peak-limit to −1 dBFS. Writes a WAV beside the
    input and returns its path; returns None if no fix was needed (no rewrite).
    Pure numpy/soundfile — no pedalboard.
    """
    import numpy as np
    import soundfile as sf
    x, sr = sf.read(in_path, always_2d=True)
    n = len(x)
    if n < sr:                                   # < 1 s — leave it
        return None
    aenv = np.abs(x).max(axis=1)
    peak = float(aenv.max())
    ploc = int(np.argmax(aenv))
    body_peak = float(aenv[:int(n * 0.85)].max()) if int(n * 0.85) else peak
    clipping = peak >= 0.985
    end_burst = ploc >= n * 0.80 and peak > body_peak * 1.25
    if not (clipping or end_burst):
        return None
    thr = max(1e-3, body_peak * 0.02)            # trim trailing near-silence
    above = np.where(aenv > thr)[0]
    if above.size:
        x = x[: int(above[-1]) + 1]
    f = min(int(0.4 * sr), len(x))               # fade out the tail (tames a terminal spike)
    if f > 0:
        x[-f:] = x[-f:] * np.linspace(1.0, 0.0, f)[:, None]
    pk = float(np.max(np.abs(x))) if x.size else 0.0
    if pk > 0.891:                               # peak-limit to ~ -1 dBFS
        x = x * (0.891 / pk)
    out = os.path.splitext(in_path)[0] + ".wav"
    sf.write(out, x, sr, subtype="PCM_16")
    return out


def gate_region(in_path, start, end, fade_ms=40):
    """Keep only [start, end] seconds of a track and silence the rest, with short
    equal-power fades at the two boundaries so the kept window doesn't click. Used to
    reduce an isolated stem to just the layer's time window (the layer only exists
    inside the lego region; outside it the source == the preserved backing). Rewrites
    the file in place as a WAV. Returns the path. numpy/soundfile only."""
    import numpy as np
    import soundfile as sf
    x, sr = sf.read(in_path, always_2d=True)
    n = len(x)
    a = max(0, int(float(start) * sr))
    b = min(n, int(float(end) * sr)) if end and float(end) > 0 else n
    if b <= a:
        return in_path
    out = np.zeros_like(x)
    out[a:b] = x[a:b]
    f = max(1, int(fade_ms / 1000.0 * sr))
    ramp = np.sqrt(np.linspace(0, 1, f))[:, None]        # equal-power
    if a + f <= b:
        out[a:a + f] *= ramp
    if b - f >= a:
        out[b - f:b] *= ramp[::-1]
    sf.write(in_path, out, sr, subtype="PCM_16")
    return in_path


def close_seam_gap(in_path, min_ms=60, max_ms=900, xfade_ms=60):
    """Remove the brief near-silent gap the ACE edit guider inserts at the seam of
    an EXTEND (model starts the extended region with a moment of silence). Finds the
    largest short interior near-silent stretch (not the leading/trailing edge) and
    splices it out with a short equal-power crossfade so the join is continuous.
    Returns the path if it changed anything, else None. numpy/soundfile only."""
    import numpy as np
    import soundfile as sf
    x, sr = sf.read(in_path, always_2d=True)
    n = len(x)
    if n < sr:
        return None
    w = max(1, int(0.02 * sr))                       # 20 ms peak-envelope frames
    env = np.array([np.abs(x[i:i + w]).max() for i in range(0, n - w, w)])
    if env.size == 0:
        return None
    pk = float(env.max()) or 1.0
    thr = pk * 0.05
    edge = int(1.0 / 0.02)                            # ignore the first/last ~1 s (lead-in / fade-out)
    runs, i = [], 0
    while i < len(env):
        if env[i] < thr:
            j = i
            while j < len(env) and env[j] < thr:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    cand = [(i, j) for (i, j) in runs
            if min_ms / 1000 <= (j - i) * 0.02 <= max_ms / 1000 and i > edge and j < len(env) - edge]
    if not cand:
        return None
    i, j = max(cand, key=lambda r: r[1] - r[0])
    s0, s1 = i * w, j * w                             # silent span [s0:s1] to remove
    xf = min(int(xfade_ms / 1000 * sr), s0, n - s1)
    if xf > 0:
        a, b = x[:s0], x[s1:]
        ramp = np.linspace(0.0, np.pi / 2, xf)[:, None]
        blend = a[-xf:] * np.cos(ramp) + b[:xf] * np.sin(ramp)
        joined = np.concatenate([a[:-xf], blend, b[xf:]], axis=0)
    else:
        joined = np.concatenate([x[:s0], x[s1:]], axis=0)
    sf.write(in_path, joined, sr, subtype="PCM_16")
    return in_path


# Native pedalboard presets. Each is a list of (effect_name, kwargs). Effects are
# resolved against pedalboard at build time so importing this module never
# requires pedalboard to be installed.
PRESETS = {
    "tighten_highgain": {
        "label": "Tighten (modern high-gain)",
        "desc": "HPF the flub, notch the 3–5 kHz fizz, tame the digital top, glue comp. Good default for metal rhythm.",
        "chain": [
            ("HighpassFilter", {"cutoff_frequency_hz": 95}),
            ("PeakFilter", {"cutoff_frequency_hz": 4200, "gain_db": -4.5, "q": 2.2}),
            ("PeakFilter", {"cutoff_frequency_hz": 250, "gain_db": -2.0, "q": 1.2}),
            ("HighShelfFilter", {"cutoff_frequency_hz": 9000, "gain_db": -5.0, "q": 0.7}),
            ("Compressor", {"threshold_db": -18, "ratio": 3, "attack_ms": 6, "release_ms": 120}),
            ("Gain", {"gain_db": 2.0}),
        ],
    },
    "accdc_crunch": {
        "label": "AC/DC crunch",
        "desc": "Open low end, presence lift, only a touch of glue — for swinging hard-rock crunch.",
        "chain": [
            ("HighpassFilter", {"cutoff_frequency_hz": 80}),
            ("PeakFilter", {"cutoff_frequency_hz": 2500, "gain_db": 2.5, "q": 1.0}),
            ("HighShelfFilter", {"cutoff_frequency_hz": 10500, "gain_db": -3.0, "q": 0.7}),
            ("Compressor", {"threshold_db": -14, "ratio": 2, "attack_ms": 10, "release_ms": 150}),
            ("Gain", {"gain_db": 1.0}),
        ],
    },
    "scooped_metal": {
        "label": "Scooped metal",
        "desc": "Mid scoop + presence + tight lows for chuggy, modern scooped rhythm tone.",
        "chain": [
            ("HighpassFilter", {"cutoff_frequency_hz": 90}),
            ("PeakFilter", {"cutoff_frequency_hz": 700, "gain_db": -3.5, "q": 1.0}),
            ("PeakFilter", {"cutoff_frequency_hz": 3000, "gain_db": 2.0, "q": 1.5}),
            ("PeakFilter", {"cutoff_frequency_hz": 4500, "gain_db": -3.0, "q": 2.5}),
            ("HighShelfFilter", {"cutoff_frequency_hz": 9000, "gain_db": -4.0, "q": 0.7}),
            ("Compressor", {"threshold_db": -18, "ratio": 3, "attack_ms": 5, "release_ms": 110}),
            ("Gain", {"gain_db": 2.0}),
        ],
    },
    "warm_smooth": {
        "label": "Warm & smooth",
        "desc": "Gentle de-fizz + warmth, light saturation — softens harsh AI guitar without big EQ moves.",
        "chain": [
            ("HighpassFilter", {"cutoff_frequency_hz": 70}),
            ("HighShelfFilter", {"cutoff_frequency_hz": 7000, "gain_db": -4.0, "q": 0.7}),
            ("Distortion", {"drive_db": 6.0}),
            ("Compressor", {"threshold_db": -16, "ratio": 2, "attack_ms": 12, "release_ms": 160}),
            ("Gain", {"gain_db": 0.0}),
        ],
    },
}


def available():
    try:
        import pedalboard  # noqa: F401
        return True
    except Exception:
        return False


def _helix_path(cfg):
    return ((cfg or {}).get("helix_vst3_path") or "").strip()


def _helix_dir(cfg):
    return ((cfg or {}).get("helix_presets_dir") or "").strip()


def list_helix(cfg):
    """Names of captured Helix tones (raw VST3 state saved to <name>.helixstate)."""
    import glob
    d = _helix_dir(cfg)
    if not d or not os.path.isdir(d):
        return []
    return sorted(os.path.splitext(os.path.basename(f))[0]
                  for f in glob.glob(os.path.join(d, "*.helixstate")))


def capture_helix_state(cfg, name):
    """Open Helix Native's editor (BLOCKS until the window is closed), then save
    its full VST3 state under `name`. Dial any tone — amps/cabs/effects — and it
    can be re-applied later with no UI. Mac-only (needs the GUI session)."""
    import re
    import pedalboard as pb
    helix = _helix_path(cfg)
    if not (helix and os.path.exists(helix)):
        raise RuntimeError("Helix Native not configured")
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "tone"
    vst = pb.load_plugin(helix)
    vst.show_editor()                      # blocks until the user closes it
    d = _helix_dir(cfg)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name + ".helixstate"), "wb") as f:
        f.write(vst.raw_state)
    return name


def presets(cfg=None):
    """List tone presets for the UI, plus engine availability."""
    cfg = cfg or {}
    helix = _helix_path(cfg)
    out = [{"id": k, "label": v["label"], "desc": v["desc"]} for k, v in PRESETS.items()]
    helix_ok = bool(helix and os.path.exists(helix))
    if helix_ok:
        # captured tones first (each = a dialed Helix preset, reused via raw_state)
        for n in reversed(list_helix(cfg)):
            out.insert(0, {"id": f"helix:{n}", "label": f"Helix · {n}",
                           "desc": f"Your captured Helix tone '{n}' (full amp/cab/FX state)."})
        out.insert(0, {"id": "helix", "label": "Helix Native (current/default state)",
                       "desc": "Helix Native with its last-loaded state. Capture a named tone to reuse a specific preset."})
    return {"presets": out, "pedalboard": available(),
            "helix_available": helix_ok, "helix_states": list_helix(cfg)}


def _build_board(preset, cfg, opts):
    """Construct a pedalboard.Pedalboard for the chosen preset.

    `helix` preset → load the Helix Native VST3/AU (optionally a .vstpreset).
    Otherwise build the named native chain. An IR cab (cfg['guitar_ir_path'] or
    opts['ir_path']) is convolved in when present (mix = opts['ir_mix']).
    """
    import pedalboard as pb
    cfg = cfg or {}
    opts = opts or {}
    plugins = []

    if preset == "helix" or preset.startswith("helix:"):
        helix = _helix_path(cfg)
        if not (helix and os.path.exists(helix)):
            raise RuntimeError("Helix Native not configured — set helix_vst3_path in app_config.json")
        vst = pb.load_plugin(helix)
        # a captured named tone ("helix:<name>") → restore its full VST3 state
        if preset.startswith("helix:"):
            sp = os.path.join(_helix_dir(cfg), preset.split(":", 1)[1] + ".helixstate")
            if os.path.exists(sp):
                with open(sp, "rb") as f:
                    vst.raw_state = f.read()
        else:
            vpreset = (opts.get("helix_preset") or cfg.get("helix_default_preset") or "").strip()
            if vpreset and os.path.exists(vpreset):
                try:
                    vst.load_preset(vpreset)  # .vstpreset/.fxp; .hlx not guaranteed
                except Exception:
                    pass
        plugins.append(vst)
    else:
        spec = PRESETS.get(preset)
        if not spec:
            raise RuntimeError(f"unknown tone preset: {preset}")
        for name, kw in spec["chain"]:
            plugins.append(getattr(pb, name)(**kw))

    ir = (opts.get("ir_path") or cfg.get("guitar_ir_path") or "").strip()
    if ir and os.path.exists(ir):
        plugins.append(pb.Convolution(ir, float(opts.get("ir_mix", 1.0))))

    return pb.Pedalboard(plugins)


def process_stem(in_path: str, out_path: str, preset: str, cfg=None, opts=None):
    """Apply the tone chain to one guitar stem, writing a 44.1k WAV."""
    import numpy as np
    import soundfile as sf
    data, sr = sf.read(in_path, dtype="float32", always_2d=True)  # (frames, ch)
    x = data.T  # (ch, frames)
    if x.shape[0] == 1:        # upmix mono → stereo (Helix/VST plugins expect 2ch)
        x = np.repeat(x, 2, axis=0)
    elif x.shape[0] > 2:
        x = x[:2]
    board = _build_board(preset, cfg, opts)
    y = board(x, sr)
    # guard against rare overs introduced by EQ/comp make-up gain
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.999:
        y = y * (0.999 / peak)
    sf.write(out_path, y.T, sr, format="WAV", subtype="PCM_16")
    return out_path


def _load(p, target_sr):
    import numpy as np
    import soundfile as sf
    d, sr = sf.read(p, dtype="float32", always_2d=True)
    if d.shape[1] == 1:
        d = np.repeat(d, 2, axis=1)
    elif d.shape[1] > 2:
        d = d[:, :2]
    if sr != target_sr:
        import torch
        import torchaudio
        d = torchaudio.functional.resample(torch.from_numpy(d.T), sr, target_sr).numpy().T
    return d


def _detect_lag(proc_mono, raw_mono, maxlag=4096):
    """Samples `proc` lags `raw` by (plugin latency), via cross-correlation on a
    central window. Returns an int lag to shift `proc` back by so it re-aligns."""
    import numpy as np
    n = min(len(proc_mono), len(raw_mono), 200000)
    if n < 2048:
        return 0
    a = proc_mono[:n] - proc_mono[:n].mean()
    b = raw_mono[:n] - raw_mono[:n].mean()
    c = np.correlate(a, b, "full")
    mid = len(c) // 2
    seg = c[mid - maxlag: mid + maxlag]
    return int(np.argmax(seg) - maxlag)


def recombine_delta(original_path, raw_guitar_path, processed_guitar_path,
                    target_sr=44100, normalize=True, gain_match=True) -> bytes:
    """Replace the guitar in the pristine original mix with the processed guitar:
        out = original - raw_guitar + processed_guitar
    so drums/bass/etc. keep their original quality (no Demucs re-sum artifacts)
    and only the guitar's TONE changes — not the mix balance.

    Safeguards so a loud/latent plugin (e.g. a Helix amp) can't dominate:
      - latency-align: shift `processed` back by its plugin latency so `-raw`
        actually cancels the original guitar (no comb-filter residue);
      - gain-match: scale `processed` to the raw guitar's RMS, so the guitar
        stays at its original level regardless of preset make-up gain.
    (Tone character still comes through; only the level is held constant.)"""
    import numpy as np
    import soundfile as sf

    orig = _load(original_path, target_sr)
    raw = _load(raw_guitar_path, target_sr)
    proc = _load(processed_guitar_path, target_sr)

    lag = _detect_lag(proc.mean(axis=1), raw.mean(axis=1))
    if lag > 0:               # proc is delayed → drop the leading `lag` samples
        proc = proc[lag:]
    elif lag < 0:
        proc = np.pad(proc, ((-lag, 0), (0, 0)))

    if gain_match:
        rr = float(np.sqrt(np.mean(raw ** 2)))
        pr = float(np.sqrt(np.mean(proc ** 2)))
        if pr > 1e-6:
            proc = proc * (rr / pr)

    n = max(len(orig), len(raw), len(proc))

    def pad(a):
        return np.pad(a, ((0, n - len(a)), (0, 0))) if len(a) < n else a[:n]

    out = pad(orig) - pad(raw) + pad(proc)
    if normalize:
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > 0.97:
            out = out * (0.97 / peak)
    buf = io.BytesIO()
    sf.write(buf, out, target_sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def recombine(stem_paths, target_sr=44100, normalize=True) -> bytes:
    """Sum stem WAVs (already 44.1k stereo from Demucs) into one stereo bounce.
    Pass the PROCESSED guitar stem in place of the raw one."""
    import numpy as np
    import soundfile as sf

    loaded = []
    for p in stem_paths:
        data, sr = sf.read(p, dtype="float32", always_2d=True)  # (frames, ch)
        a = data
        if a.shape[1] == 1:
            a = np.repeat(a, 2, axis=1)
        elif a.shape[1] > 2:
            a = a[:, :2]
        if sr != target_sr:  # demucs is 44.1k, but be safe
            import torch
            import torchaudio
            a = torchaudio.functional.resample(torch.from_numpy(a.T), sr, target_sr).numpy().T
        loaded.append(a)
    if not loaded:
        raise ValueError("no stems to recombine")

    length = max(a.shape[0] for a in loaded)
    out = np.zeros((length, 2), dtype="float32")
    for a in loaded:
        out[:a.shape[0]] += a

    if normalize:
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > 0.97:
            out = out * (0.97 / peak)

    buf = io.BytesIO()
    sf.write(buf, out, target_sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()
