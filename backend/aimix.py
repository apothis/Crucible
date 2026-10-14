"""AI Mix, Phase 1 (docs/AI_MIX_PLAN.md): mix a Suno stem download from its STEMS.

Pure Mac DSP (numpy/scipy/pedalboard/pyloudnorm), no GPU, no ML. Stages:

  ingest   - a Suno zip or folder -> role-mapped stems + the full mix (reference only)
  analyse  - per-stem loudness/crest/bands/width, a Suno-balance gain fit, a masking matrix
  rules    - analysis -> a MIX DOC (plain JSON parameters, editable, re-renderable)
  render   - mix doc -> stereo WAV via pedalboard, one stem at a time (16 GB Mac)

Suno's stems are regenerated, not separated (Gate 0, measured): they do not null against the
mix and arrive ~7 dB hot. Their LOUDNESS ENVELOPES do track the mix (r ~0.9), so the balance
Suno intended is recovered by fitting per-stem gains to the mix's band-energy envelopes (NNLS
on power: incoherent stems add in power). The full mix is never part of the output.

Rule constants marked [H] are first guesses to be judged by ear in the Phase 1 A/B.
"""
import json
import os
import re
import zipfile

import numpy as np

HOP_S = 0.1                                       # analysis frame hop (100 ms)
BAND_EDGES = [20, 60, 120, 250, 500, 1000, 2000, 4000, 8000, 12000, 20000]
VOCAL_BANDS = (5, 6, 7)                           # 1-2k, 2-4k, 4-8k: where the lead must win

# file-name -> role. Order matters (backing before vocals).
ROLE_PATTERNS = [
    ("backing", r"backing|bgv|back.?vocal|harmon"),
    ("vocals", r"vocal|vox|lead"),
    ("drums", r"drum"),
    ("bass", r"bass"),
    ("guitar", r"guitar|gtr"),
    ("keys", r"key|piano|organ"),
    ("synth", r"synth|pad"),
    ("strings", r"string|violin|cello|orch"),
    ("brass", r"brass|horn|trumpet|trombone"),
    ("woodwinds", r"woodwind|flute|clarinet|sax|oboe"),
    ("percussion", r"perc|shaker|tamb|conga"),
    ("fx", r"^fx$|sfx|effect|riser|ambien"),
    ("other", r"\bother\b"),
]

# Per-role defaults [H]. hpf = high-pass Hz (lowered automatically if the stem has real content
# below it); comp = (ratio, gain-reduction dB at the stem's loud moments, attack ms, release ms)
# or None; width = side gain above low_mono_hz; sends in dB (None = no send); punch = transient-
# shaper attack boost in dB (0 = off). No reverb on vocals: Suno vocal stems arrive wet (user, 2026-10-11).
ROLE_RULES = {
    "vocals":     dict(hpf=80,  comp=(3.0, 5.0, 8, 120),   width=1.0,  low_mono_hz=150, verb=None, delay=-26),
    "backing":    dict(hpf=140, comp=(2.5, 4.0, 10, 150),  width=1.15, low_mono_hz=150, verb=None, delay=None),
    "drums":      dict(hpf=30,  comp=(2.0, 3.0, 25, 120),  width=1.0,  low_mono_hz=120, verb=None, delay=None, punch=6.0),
    "bass":       dict(hpf=30,  comp=(4.0, 5.0, 15, 150),  width=1.0,  low_mono_hz=200, verb=None, delay=None),
    "guitar":     dict(hpf=70,  comp=None,                 width=1.1,  low_mono_hz=150, verb=None, delay=None),
    "keys":       dict(hpf=100, comp=(2.0, 3.0, 15, 150),  width=1.1,  low_mono_hz=150, verb=-24, delay=None),
    "synth":      dict(hpf=100, comp=(2.0, 3.0, 15, 150),  width=1.1,  low_mono_hz=150, verb=None, delay=None),
    "strings":    dict(hpf=120, comp=None,                 width=1.1,  low_mono_hz=150, verb=-20, delay=None),
    "brass":      dict(hpf=120, comp=(2.0, 3.0, 15, 150),  width=1.0,  low_mono_hz=150, verb=-22, delay=None),
    "woodwinds":  dict(hpf=150, comp=None,                 width=1.0,  low_mono_hz=150, verb=-20, delay=None),
    "percussion": dict(hpf=150, comp=None,                 width=1.2,  low_mono_hz=150, verb=-24, delay=None),
    "fx":         dict(hpf=150, comp=None,                 width=1.0,  low_mono_hz=150, verb=None, delay=None),
    "other":      dict(hpf=100, comp=None,                 width=1.0,  low_mono_hz=150, verb=None, delay=None),
}

# Level offsets (dB) applied on top of the Suno-fitted balance [H].
PROFILES = {
    "suno": {"label": "Suno balance (no level changes)", "offsets": {}},
    "rock": {"label": "Rock/metal: vocal + drums forward",
             "offsets": {"vocals": 1.5, "drums": 1.0, "backing": -1.0, "fx": -2.0, "percussion": -1.0}},
    "vocal": {"label": "Vocal-forward",
              "offsets": {"vocals": 3.0, "backing": -1.0, "guitar": -1.0, "synth": -1.0, "keys": -1.0}},
}

MASK_START = 0.30          # masking score above which a masker gets cut in a vocal band [H]
MASK_MAX_CUT_DB = 4.0      # deepest masking cut [H]


def db(x):
    return 10.0 * np.log10(np.maximum(x, 1e-20))


# ---------------------------------------------------------------- ingest

def role_of(name: str) -> str:
    n = re.sub(r"^\d+\s*", "", name).strip().lower().replace("_", " ")
    for role, pat in ROLE_PATTERNS:
        if re.search(pat, n):
            return role
    return ""


def _extract_zip(zpath, src):
    with zipfile.ZipFile(zpath) as z:
        for m in z.infolist():
            base = os.path.basename(m.filename)
            if m.is_dir() or base.startswith(".") or os.path.splitext(base)[1].lower() not in AUDIO_EXT:
                continue
            with z.open(m) as fi, open(os.path.join(src, base), "wb") as fo:
                while True:
                    chunk = fi.read(1 << 20)
                    if not chunk:
                        break
                    fo.write(chunk)


AUDIO_EXT = (".wav", ".flac")


def ingest(path: str, work_dir: str) -> dict:
    """Locate stems + (optional) full mix in a Suno zip, a folder, or a folder holding a zip plus
    loose files (e.g. the stems zip + the full song chosen together). Zip members are extracted
    (audio only, flattened) into work_dir/src. Returns {title, mix: path|None, stems: [...]}.

    Both Suno download kinds are handled: Studio "Get Stems" numbers the full mix "0 <title>"
    and the stems 1..12; the regular library download has NO full mix and numbers stems from
    "0 Lead Vocals" incl. an "Other" stem. A file only counts as the full mix if it is the one
    file whose name is not a stem name; an unknown stem name is never mistaken for the mix when
    there are several (Gleipnir, 2026-10-14: "9 Other" was taken for the mix)."""
    title = None
    if os.path.isfile(path) and path.lower().endswith(".zip"):
        src = os.path.join(work_dir, "src")
        os.makedirs(src, exist_ok=True)
        _extract_zip(path, src)
        title = os.path.splitext(os.path.basename(path))[0]
    elif os.path.isdir(path):
        zips = [f for f in os.listdir(path) if f.lower().endswith(".zip") and not f.startswith(".")]
        if zips:
            src = os.path.join(work_dir, "src")
            os.makedirs(src, exist_ok=True)
            for zf in zips:
                _extract_zip(os.path.join(path, zf), src)
            for f in os.listdir(path):                      # loose files chosen alongside the zip
                if os.path.splitext(f)[1].lower() in AUDIO_EXT and not f.startswith("."):
                    import shutil
                    shutil.copy2(os.path.join(path, f), os.path.join(src, f))
            title = os.path.splitext(zips[0])[0]
        else:
            # link (same volume) or copy the stems into the session, so the session never depends
            # on the chosen folder staying where it is
            import shutil
            src = os.path.join(work_dir, "src")
            os.makedirs(src, exist_ok=True)
            for f in os.listdir(path):
                if os.path.splitext(f)[1].lower() in AUDIO_EXT and not f.startswith("."):
                    dst = os.path.join(src, f)
                    try:
                        os.link(os.path.join(path, f), dst)
                    except OSError:
                        shutil.copy2(os.path.join(path, f), dst)
            base = os.path.basename(os.path.normpath(path))
            title = None if re.fullmatch(r"[0-9a-f]{32}", base) else base
    else:
        raise ValueError(f"not a zip or folder: {path}")

    files = sorted(f for f in os.listdir(src) if os.path.splitext(f)[1].lower() in AUDIO_EXT and not f.startswith("."))
    stems, unmatched = [], []
    for f in files:
        name = os.path.splitext(f)[0]
        role = role_of(name)
        (stems if role else unmatched).append({"name": re.sub(r"^\d+\s*", "", name), "role": role,
                                               "path": os.path.join(src, f)})
    mix = None
    if len(unmatched) == 1:
        mix = unmatched[0]["path"]
    elif len(unmatched) > 1:
        # several unknown names: only a "0 <x>" file or one named like the zip is the mix
        cand = [u for u in unmatched if re.match(r"^0\s", os.path.basename(u["path"]))
                or (title and u["name"].lower() == re.sub(r"\s*\(.*\)$", "", title).lower())]
        if len(cand) == 1:
            mix = cand[0]["path"]
    for u in unmatched:
        if u["path"] != mix:
            stems.append({**u, "role": "other"})
    if mix and not title:
        title = re.sub(r"^\d+\s*", "", os.path.splitext(os.path.basename(mix))[0])
    if title:
        title = re.sub(r"\s*[-_(]*\s*stems?\s*\)?\s*$", "", title, flags=re.I).strip() or title
    if not stems:
        raise ValueError("no stems found")
    return {"title": title or "Untitled", "mix": mix, "stems": stems}


# ---------------------------------------------------------------- analysis

def _read(path, dtype="float32"):
    import soundfile as sf
    x, sr = sf.read(path, dtype=dtype, always_2d=True)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    return x[:, :2], sr


def _band_env(mono, sr):
    """(bands, frames) linear power per analysis frame."""
    from scipy.signal import stft
    hop = int(HOP_S * sr)
    nfft = 4096
    f, _, Z = stft(mono, sr, nperseg=nfft, noverlap=nfft - hop, boundary=None, padded=False)
    P = (np.abs(Z) ** 2).astype(np.float64)
    out = np.zeros((len(BAND_EDGES) - 1, P.shape[1]))
    for k in range(len(BAND_EDGES) - 1):
        sel = (f >= BAND_EDGES[k]) & (f < BAND_EDGES[k + 1])
        out[k] = P[sel].sum(0)
    return out


def _stem_stats(x, sr, meter):
    from scipy.signal import butter, sosfiltfilt
    mono = x.mean(1)
    hop = int(HOP_S * sr)
    n = len(mono) // hop
    fr = np.sqrt((mono[:n * hop].reshape(n, hop) ** 2).mean(1) + 1e-20)
    fr_db = 20 * np.log10(fr)
    active = fr_db > -60
    lufs = float(meter.integrated_loudness(x)) if active.mean() > 0.01 else float("-inf")
    peak_db = float(20 * np.log10(max(np.abs(x).max(), 1e-10)))
    act_rms_db = float(20 * np.log10(np.sqrt(np.mean(fr[active] ** 2)))) if active.any() else -120.0
    mid, side = (x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2
    pm, ps = float(np.mean(mid ** 2)), float(np.mean(side ** 2))
    sos = butter(4, 150, "lowpass", fs=sr, output="sos")
    lm, ls = sosfiltfilt(sos, mid), sosfiltfilt(sos, side)
    env = _band_env(mono, sr)
    spec = env.sum(1)
    cum = np.cumsum(spec) / max(spec.sum(), 1e-20)
    return {
        "lufs": lufs, "peak_db": peak_db, "active_rms_db": act_rms_db,
        "crest_db": peak_db - act_rms_db, "active_frac": float(active.mean()),
        "p95_frame_db": float(np.percentile(fr_db[active], 95)) if active.any() else -120.0,
        "width_side_db": float(db(ps) - db(pm)) if pm > 0 else 0.0,
        "low_side_db": float(db(np.mean(ls ** 2)) - db(np.mean(lm ** 2))) if np.mean(lm ** 2) > 0 else 0.0,
        "band_db": [float(v) for v in db(spec / max(env.shape[1], 1))],
        "energy_2pct_hz": float(next((BAND_EDGES[k + 1] for k in range(len(cum)) if cum[k] >= 0.02), 20)),
    }, env


def analyse(session: dict) -> dict:
    """Per-stem stats + band envelopes -> Suno-balance gains (NNLS vs the mix) + masking."""
    import pyloudnorm as pyln
    from scipy.optimize import nnls

    stems = session["stems"]
    stats, envs, sr0 = {}, {}, None
    for s in stems:
        x, sr = _read(s["path"])
        sr0 = sr0 or sr
        if sr != sr0:
            raise ValueError(f"sample-rate mismatch: {s['name']} {sr} vs {sr0}")
        st, env = _stem_stats(x, sr, pyln.Meter(sr))
        stats[s["name"]], envs[s["name"]] = st, env
        del x
    live = [s for s in stems if stats[s["name"]]["active_frac"] > 0.01 and np.isfinite(stats[s["name"]]["lufs"])]
    silent = [s["name"] for s in stems if s not in live]

    rep = {"sr": sr0, "stats": stats, "silent": silent, "gains_db": {}, "fit": None, "bpm": None}
    names = [s["name"] for s in live]

    # Suno balance: mix band power ~ sum_i w_i * stem_i band power, w_i = g_i^2 >= 0
    if session.get("mix"):
        mx, sr = _read(session["mix"])
        rep["mix"] = _stem_stats(mx, sr, pyln.Meter(sr))[0]
        menv = _band_env(mx.mean(1), sr)
        del mx
        nf = min(menv.shape[1], *(envs[n].shape[1] for n in names))
        scale = menv[:, :nf].mean(1, keepdims=True) + 1e-20          # equal weight per band
        A = np.stack([(envs[n][:, :nf] / scale).ravel() for n in names], 1)
        b = (menv[:, :nf] / scale).ravel()
        with np.errstate(all="ignore"):               # Accelerate matmul raises spurious FP flags
            w, _ = nnls(A, b)
            pred = A @ w
        lp, lb = np.log(pred + 1e-9), np.log(b + 1e-9)
        rep["fit"] = {"log_env_r": float(np.corrcoef(lp, lb)[0, 1])}
        for n, wi in zip(names, w):
            rep["gains_db"][n] = float(np.clip(10 * np.log10(wi), -30, 12)) if wi > 0 else -30.0
    else:
        # No full mix (Suno's regular library download): start from the stems' own delivered
        # levels. Those stems sum to a plausible finished level (Gleipnir: -13.4 LUFS) so they
        # look mix-balanced [H, unverifiable without the mix]; the adaptive level solve then
        # pulls any group outside the style's range back in. (The old -24 LUFS-each fallback
        # made every stem equally loud.)
        for n in names:
            rep["gains_db"][n] = 0.0
        rep["fit"] = {"no_mix": True}

    # masking of the lead vocal by every other stem, in the vocal bands (after Suno balance)
    voc = next((s["name"] for s in live if s["role"] == "vocals"), None)
    rep["masking"] = {}
    if voc:
        g = {n: 10 ** (rep["gains_db"][n] / 10) for n in names}
        V = envs[voc] * g[voc]
        for n in names:
            if n == voc:
                continue
            M = envs[n] * g[n]
            nf = min(V.shape[1], M.shape[1])
            row = {}
            for k in VOCAL_BANDS:
                v, m = V[k, :nf], M[k, :nf]
                on = v > np.percentile(v, 40)                         # frames where the vocal is present
                if on.sum() < 10:
                    continue
                d = db(m[on]) - db(v[on])
                row[k] = float(np.mean(np.clip((d + 6.0) / 12.0, 0.0, 1.0)))
            rep["masking"][n] = row

    # tempo for the delay bus
    try:
        import librosa
        src = next((s for s in live if s["role"] == "drums"), None)
        p = src["path"] if src else session.get("mix")
        if p:
            y, _ = librosa.load(p, sr=22050, mono=True, duration=90.0, offset=30.0)
            t, _ = librosa.beat.beat_track(y=y, sr=22050)
            rep["bpm"] = float(np.atleast_1d(t)[0])
    except Exception:
        rep["bpm"] = None
    return rep


# ---------------------------------------------------------------- rules -> mix doc

def _band_center(k):
    return float(np.sqrt(BAND_EDGES[k] * BAND_EDGES[k + 1]))


def build_mixdoc(session: dict, analysis: dict, profile: str = "rock") -> dict:
    prof = PROFILES.get(profile, PROFILES["rock"])
    stems_doc = {}
    for s in session["stems"]:
        n, role = s["name"], s["role"] or "other"
        if n in analysis["silent"]:
            continue
        st = analysis["stats"][n]
        r = ROLE_RULES.get(role, ROLE_RULES["other"])
        chain = []
        # high-pass, never above where the stem's own content starts (2% energy point)
        hpf = min(r["hpf"], max(20.0, st["energy_2pct_hz"] * 0.7))
        chain.append({"fx": "highpass", "hz": round(hpf)})
        # masking-aware cuts: the masker gives way to the lead in the vocal bands
        cuts = []
        for k, m in (analysis.get("masking", {}).get(n) or {}).items():
            if m > MASK_START:
                cut = -min(MASK_MAX_CUT_DB, round(8.0 * (m - MASK_START), 1))
                cuts.append({"hz": round(_band_center(int(k))), "q": 1.0, "db": cut})
        if cuts:
            chain.append({"fx": "peq", "bands": cuts, "why": "unmask lead vocal"})
        if r.get("punch"):
            chain.append({"fx": "transient", "attack_db": r["punch"], "sustain_db": 0.0})
        if r["comp"]:
            ratio, gr, att, rel = r["comp"]
            thr = st["p95_frame_db"] - gr / (1.0 - 1.0 / ratio)
            chain.append({"fx": "compressor", "threshold_db": round(thr, 1), "ratio": ratio,
                          "attack_ms": att, "release_ms": rel})
        target_lufs = st["lufs"] + analysis["gains_db"][n] + prof["offsets"].get(role, 0.0)
        stems_doc[n] = {
            "file": s["path"], "role": role, "mute": False,
            "target_lufs": round(target_lufs, 2),      # post-chain loudness the renderer lands on
            "suno_lufs": round(st["lufs"] + analysis["gains_db"][n], 2),   # Suno-balance level (plain sum)
            "trim_db": 0.0,                            # user fader on top of the target
            "width": r["width"], "low_mono_hz": r["low_mono_hz"],
            "chain": chain,
            "sends": {k: v for k, v in (("verb", r["verb"]), ("delay", r["delay"])) if v is not None},
        }
    bpm = analysis.get("bpm") or 120.0
    return {
        "version": 1, "title": session["title"], "mode": "remix", "source": "suno_stems",
        "profile": profile, "sample_rate": analysis["sr"],
        "stems": stems_doc,
        "buses": {
            "verb": {"fx": "reverb", "room_size": 0.35, "damping": 0.5, "predelay_ms": 30,
                     "hpf": 200, "lpf": 8000},
            "delay": {"fx": "delay", "ms": round(60000.0 / bpm, 1), "feedback": 0.25,
                      "hpf": 300, "lpf": 6000},
        },
        "master": {"tone": "flat", "glue": False, "target_lufs": (analysis.get("mix") or {}).get("lufs"),
                   "ceiling_db": -1.0},
        "provenance": [{"layer": "rules", "profile": profile}],
    }


# ---------------------------------------------------------------- render

def _width_lowmono(x, sr, width, low_hz):
    """x (ch, frames). Side above low_hz scaled by width; side below low_hz removed. Zero-phase
    split so the bands are exact complements."""
    if width == 1.0 and not low_hz:
        return x
    from scipy.signal import butter, sosfiltfilt
    mid, side = (x[0] + x[1]) / 2, (x[0] - x[1]) / 2
    if low_hz:
        side = side - sosfiltfilt(butter(4, low_hz, "lowpass", fs=sr, output="sos"), side)
    side = side * width
    return np.stack([mid + side, mid - side]).astype(np.float32)


def _transient(x, sr, attack_db, sustain_db=0.0, block_ms=1.0):
    """Differential-envelope transient shaper on x (ch, frames). A fast and a slow peak follower
    run on 1 ms blocks of a LOW-BAND (<250 Hz, kick/snare body) detector, so dense hats don't
    keep it engaged; where the fast one leads (an attack) the gain rises by up to attack_db, where
    it trails (the tail) by sustain_db. The gain hits the full band, interpolated to sample rate
    (no zipper, no latency). Timings picked as the most attack-focused of four tried on Dream Of
    Me (boost active ~21% of the time vs 35% for a full-band detector) [M]; the amount is by ear."""
    if not attack_db and not sustain_db:
        return x
    b = max(1, int(sr * block_ms / 1000.0))
    n = x.shape[1] // b
    from scipy.signal import butter, sosfilt
    det = sosfilt(butter(4, 250, "lowpass", fs=sr, output="sos"), x.mean(0))
    pk = np.abs(det[:n * b]).reshape(n, b).max(1)

    def follow(att_ms, rel_ms):
        a = np.exp(-block_ms / att_ms)
        r = np.exp(-block_ms / rel_ms)
        out = np.empty(n)
        e = 0.0
        for i in range(n):
            v = pk[i]
            e = a * e + (1 - a) * v if v > e else r * e + (1 - r) * v
            out[i] = e
        return out

    fast, slow = follow(0.1, 8.0), follow(20.0, 120.0)
    diff = 20 * np.log10((fast + 1e-9) / (slow + 1e-9))       # + on attacks, - in tails
    g_db = np.where(diff > 0, attack_db * np.clip(diff / 6.0, 0, 1),
                    sustain_db * np.clip(-diff / 6.0, 0, 1))
    t = (np.arange(n) + 0.5) * b
    g = 10 ** (np.interp(np.arange(x.shape[1]), t, g_db) / 20.0)
    return (x * g).astype(np.float32)


def _apply_chain(chain, x, sr):
    """Run a chain: consecutive pedalboard-native steps as one board, custom DSP in between."""
    run = []
    for c in chain + [None]:
        if c is not None and c["fx"] not in ("transient", "deess"):
            run.append(c)
            continue
        if run:
            x = _board(run)(x, sr)
            run = []
        if c is not None and c["fx"] == "transient":
            x = _transient(x, sr, float(c.get("attack_db", 0)), float(c.get("sustain_db", 0)))
        elif c is not None and c["fx"] == "deess":
            from .aimix_adaptive import deess
            x = deess(x, sr, float(c.get("range_db", 0)))
    return x


def _board(chain):
    import pedalboard as pb
    plugins = []
    for c in chain:
        if c["fx"] == "highpass":
            plugins.append(pb.HighpassFilter(cutoff_frequency_hz=float(c["hz"])))
        elif c["fx"] == "lowpass":
            plugins.append(pb.LowpassFilter(cutoff_frequency_hz=float(c["hz"])))
        elif c["fx"] == "eq":
            for m in c["moves"]:
                if m["type"] == "lowshelf":
                    plugins.append(pb.LowShelfFilter(cutoff_frequency_hz=float(m["hz"]), gain_db=float(m["db"]), q=0.7))
                elif m["type"] == "highshelf":
                    plugins.append(pb.HighShelfFilter(cutoff_frequency_hz=float(m["hz"]), gain_db=float(m["db"]), q=0.7))
                else:
                    plugins.append(pb.PeakFilter(cutoff_frequency_hz=float(m["hz"]), gain_db=float(m["db"]),
                                                 q=float(m.get("q", 1.4))))
        elif c["fx"] == "peq":
            for b in c["bands"]:
                plugins.append(pb.PeakFilter(cutoff_frequency_hz=float(b["hz"]), gain_db=float(b["db"]),
                                             q=float(b.get("q", 1.0))))
        elif c["fx"] == "compressor":
            plugins.append(pb.Compressor(threshold_db=float(c["threshold_db"]), ratio=float(c["ratio"]),
                                         attack_ms=float(c["attack_ms"]), release_ms=float(c["release_ms"])))
    return pb.Pedalboard(plugins)


def render(doc: dict, out_path: str, processed: bool = True) -> dict:
    """Render a mix doc to a float WAV at out_path (headroom-normalised to -6 dBFS peak, no
    master). processed=False = the Suno-balance plain sum (gains only, no chain/width/sends)."""
    import pedalboard as pb
    import pyloudnorm as pyln
    import soundfile as sf

    sr = int(doc["sample_rate"])
    meter = pyln.Meter(sr)
    mix = verb = dly = None
    report = {}
    for n, s in doc["stems"].items():
        if s.get("mute"):
            continue
        x, srx = _read(s["file"])
        x = x.T.copy()                                        # (ch, frames)
        if processed:
            y = _apply_chain(s["chain"], x, sr) if s["chain"] else x
            y = _width_lowmono(y, sr, float(s.get("width", 1.0)), float(s.get("low_mono_hz") or 0))
        else:
            y = x
        cur = meter.integrated_loudness(y.T)
        tgt = s["target_lufs"] if processed else s.get("suno_lufs", s["target_lufs"])
        g_db = (tgt - cur) if np.isfinite(cur) else -30.0
        g_db += float(s.get("trim_db", 0.0))
        y = (y * (10 ** (g_db / 20.0))).astype(np.float32)
        report[n] = {"applied_gain_db": round(float(g_db), 2)}
        if mix is None:
            mix = np.zeros_like(y)
            verb = np.zeros_like(y)
            dly = np.zeros_like(y)
        L = min(mix.shape[1], y.shape[1])
        mix[:, :L] += y[:, :L]
        if processed:
            if "verb" in s.get("sends", {}):
                verb[:, :L] += y[:, :L] * (10 ** (s["sends"]["verb"] / 20.0))
            if "delay" in s.get("sends", {}):
                dly[:, :L] += y[:, :L] * (10 ** (s["sends"]["delay"] / 20.0))
        del x, y
    if mix is None:
        raise ValueError("nothing to render (all stems muted)")
    if processed:
        vb, db_ = doc["buses"]["verb"], doc["buses"]["delay"]
        if np.any(verb):
            mix += pb.Pedalboard([
                pb.HighpassFilter(cutoff_frequency_hz=float(vb["hpf"])),
                pb.Delay(delay_seconds=vb["predelay_ms"] / 1000.0, feedback=0.0, mix=1.0),
                pb.Reverb(room_size=float(vb["room_size"]), damping=float(vb["damping"]),
                          wet_level=1.0, dry_level=0.0),
                pb.LowpassFilter(cutoff_frequency_hz=float(vb["lpf"])),
            ])(verb, sr)
        if np.any(dly):
            mix += pb.Pedalboard([
                pb.HighpassFilter(cutoff_frequency_hz=float(db_["hpf"])),
                pb.Delay(delay_seconds=db_["ms"] / 1000.0, feedback=float(db_["feedback"]), mix=1.0),
                pb.LowpassFilter(cutoff_frequency_hz=float(db_["lpf"])),
            ])(dly, sr)
    peak = float(np.abs(mix).max())
    if peak > 0:
        mix *= (10 ** (-6 / 20.0)) / peak
    sf.write(out_path, mix.T, sr, subtype="FLOAT")
    return report


def match_macro(pre_path, ref_path, max_db=6.0, win_s=3.0, hop_s=0.5, smooth_s=4.0):
    """Section-level match: give our render the same SLOW loudness shape as Suno's own mix.
    Suno's regular-download stems are pre-master and exaggerate section changes vs its mastered
    mix (Gleipnir: stems-sum minus mix ran +0.8..+4.9 dB across the song [M]), which sounds
    jumpy. Both loudness curves (3 s windows, 100 Hz high-passed so bass doesn't dominate) are
    median-normalised; the difference, gaussian-smoothed over smooth_s and clamped +-max_db,
    becomes a gain curve. Only changes over several seconds are touched, never hits or phrases.
    Rewrites pre_path in place; returns stats."""
    import soundfile as sf
    from scipy.signal import butter, sosfilt, resample_poly
    from scipy.ndimage import gaussian_filter1d
    x, sr = sf.read(pre_path, dtype="float32", always_2d=True)
    r, srr = _read(ref_path)
    if srr != sr:
        r = resample_poly(r, sr, srr, axis=0).astype(np.float32)
    hp = butter(2, 100, "highpass", fs=sr, output="sos")
    w, hop = int(win_s * sr), int(hop_s * sr)

    def curve(a):
        m = sosfilt(hp, a.mean(1))
        cs = np.concatenate([[0.0], np.cumsum(m.astype(np.float64) ** 2)])
        idx = np.arange(0, max(1, len(m) - w), hop)
        return idx + w // 2, 10 * np.log10((cs[np.minimum(idx + w, len(m))] - cs[idx]) / w + 1e-12)

    t, co = curve(x)
    _, cr = curve(r[:len(x)] if len(r) >= len(x) else np.pad(r, ((0, len(x) - len(r)), (0, 0))))
    n = min(len(co), len(cr))
    t, co, cr = t[:n], co[:n], cr[:n]
    live = (cr > cr.max() - 35) & (co > co.max() - 35)
    if live.sum() < 10:
        return {"applied": False}
    g = np.zeros(n)
    g[live] = (cr[live] - np.median(cr[live])) - (co[live] - np.median(co[live]))
    if (~live).any():                                    # silence/tails: carry the neighbours' gain
        g[~live] = np.interp(np.where(~live)[0], np.where(live)[0], g[live])
    g = np.clip(gaussian_filter1d(g, smooth_s / hop_s), -max_db, max_db)
    gs = np.interp(np.arange(len(x)), t, g)
    x = (x * (10 ** (gs / 20.0))[:, None]).astype(np.float32)
    peak = float(np.abs(x).max())
    if peak > 0:
        x *= (10 ** (-6 / 20.0)) / peak
    sf.write(pre_path, x, sr, subtype="FLOAT")
    return {"applied": True, "max_db": round(float(np.abs(g).max()), 1),
            "rms_db": round(float(np.sqrt(np.mean(g ** 2))), 1),
            "shape_diff_before_db": round(float(np.std((cr - co)[live])), 1)}


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)
