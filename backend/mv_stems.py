"""Vocal stems for the music-video pipeline (docs/MV_H3_REBUILD_PLAN.md section 13).

Two uses, both optional:
  alignment  - Whisper on the isolated vocal instead of the full mix (dense metal buries the
               words; the stem does not), so lyric timing and cut points come out right;
  the latent lock - the chain can lock a different audio in the clip latents from the one it
               muxes underneath: "mix" (default; the whole song steers the picture, beats and
               all), "vocals" (the stem alone), or "vocal_forward" (the mix with the stem added
               on top at +VOCAL_FORWARD_DB, a middle ground for dense songs).

Separation runs ON THE BOX (BS-RoFormer SW, :5070) through the app's `_separate`; nothing here
runs a model on the Mac. Stems are cached under .mvwork/mv_stems/<audio_id>.vocals.wav.
"""
import os
import subprocess

VOCAL_FORWARD_DB = 6.0
LOCKS = ("mix", "vocals", "vocal_forward")


def stem_dir(library_dir):
    d = os.path.join(os.path.dirname(os.path.abspath(library_dir)), ".mvwork", "mv_stems")
    os.makedirs(d, exist_ok=True)
    return d


def vocal_stem_path(library_dir, audio_id):
    """Cached vocal stem for a library song, or None."""
    p = os.path.join(stem_dir(library_dir), f"{os.path.basename(audio_id)}.vocals.wav")
    return p if os.path.exists(p) else None


def separate_vocals(library_dir, audio_id, src_path, separate_fn):
    """Run the box separator on the song and cache its vocal stem. `separate_fn(src, work,
    engine, stems)` is the app's _separate (RoFormer on the box, frees ComfyUI first).
    Returns the stem path."""
    d = stem_dir(library_dir)
    work = os.path.join(d, f"work_{os.path.basename(audio_id)}")
    os.makedirs(work, exist_ok=True)
    produced = separate_fn(src_path, work, engine="roformer", stems="vocals")
    voc = None
    for name, path in produced:
        if name == "vocals":
            voc = path
    if not voc:
        raise RuntimeError(f"separator returned no vocals stem (got {[n for n, _ in produced]})")
    out = os.path.join(d, f"{os.path.basename(audio_id)}.vocals.wav")
    os.replace(voc, out)
    return out


def vocal_forward_path(library_dir, audio_id, mix_path, stem_path, gain_db=VOCAL_FORWARD_DB):
    """The mix with the vocal stem added on top at +gain_db, limited to avoid clipping; cached
    next to the stem. This is a LOCK source only (what the latent hears), never what is muxed."""
    out = os.path.join(stem_dir(library_dir), f"{os.path.basename(audio_id)}.vocal_forward_{gain_db:g}db.wav")
    if os.path.exists(out):
        return out
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", mix_path, "-i", stem_path,
           "-filter_complex", f"[1:a]volume={gain_db}dB[v];[0:a][v]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.97[o]",
           "-map", "[o]", "-ac", "2", "-c:a", "pcm_s16le", out]
    subprocess.run(cmd, check=True)
    return out


def muted_path(library_dir, audio_id, base_path, stem_path, ranges):
    """`base_path` (the lock source: mix, stem or vocal-forward) with the vocal stem SUBTRACTED
    inside `ranges` ([[start, end], ...] song seconds): the other singer's stretches removed from
    what a lip-sync render hears, so the singer on screen is not driven by them (two singers,
    2026-11-07). The stem comes from this mix, so the subtraction cancels the voice and leaves the
    band. A LOCK source only; the mux stays the mix. Cached by the ranges."""
    import hashlib
    rs = sorted([round(float(a), 2), round(float(b), 2)] for a, b in ranges if float(b) > float(a))
    tag = hashlib.md5((os.path.basename(base_path) + repr(rs)).encode()).hexdigest()[:12]
    out = os.path.join(stem_dir(library_dir), f"{os.path.basename(audio_id)}.muted_{tag}.wav")
    if os.path.exists(out):
        return out
    inside = "+".join(f"between(t,{a},{b})" for a, b in rs) or "0"
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", base_path, "-i", stem_path,
           "-filter_complex", f"[1:a]aformat=channel_layouts=stereo,volume=0:enable='not({inside})',aeval='-val(0)|-val(1)'[v];"
                              f"[0:a]aformat=channel_layouts=stereo[m];[m][v]amix=inputs=2:duration=first:normalize=0[o]",
           "-map", "[o]", "-ac", "2", "-c:a", "pcm_s16le", out]
    subprocess.run(cmd, check=True)
    return out


def lock_source(library_dir, audio_id, mix_path, lock):
    """The file the chain should cut its LOCK window from. Raises if a stem lock is asked for and
    no stem is cached (run POST /api/mv/stems first)."""
    lock = (lock or "mix").strip()
    if lock not in LOCKS:
        raise ValueError(f"lock must be one of {LOCKS}")
    if lock == "mix":
        return mix_path
    stem = vocal_stem_path(library_dir, audio_id)
    if not stem:
        raise FileNotFoundError("no vocal stem cached for this song; POST /api/mv/stems first")
    if lock == "vocals":
        return stem
    return vocal_forward_path(library_dir, audio_id, mix_path, stem)
