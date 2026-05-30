"""Phase 2 — Mac-side LoRA dataset builder (no GPU).

Given audio the user picked, produce the per-track training labels the ACE-Step engine
expects (METAL_LORA_PLAN §5): a `{name}.json` (bpm / keyscale / timesignature / language)
and a `{name}.lyrics.txt`. BPM+key are computed locally with librosa (the engine's LM
hallucinates them — tutorial); lyrics via our faster-whisper (`asr.py`). Captions are
left to the box LM auto-label step (or hand-entered in the review UI).

This module only BUILDS the bundle (audio + .lyrics.txt + .json) as (filename, bytes)
tuples ready for `lora_upload_py.upload()`. Orchestration (upload → scan → preprocess →
train) + the review/edit UI are Phase 4.
"""
import io
import json
import os

# Krumhansl–Schmuckler key profiles + pitch names (match comfy.KEYS / the box analyzer,
# so the detected keyscale is a valid Song-Builder/engine option, e.g. "E minor").
PITCHES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def detect_bpm_key(path):
    """Return (bpm:int, keyscale:str) for an audio file via librosa. Mac CPU, no GPU."""
    import numpy as np
    import librosa
    y, sr = librosa.load(path, mono=True)
    # tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = int(round(float(np.atleast_1d(tempo)[0])))
    # key: correlate the mean chroma against each rotated major/minor profile
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    prof = chroma.mean(axis=1)
    maj = np.asarray(_MAJOR); minr = np.asarray(_MINOR)
    best = None
    for i in range(12):
        for mode, p in (("major", np.roll(maj, i)), ("minor", np.roll(minr, i))):
            score = float(np.corrcoef(p, prof)[0, 1])
            if best is None or score > best[2]:
                best = (PITCHES[i], mode, score)
    keyscale = f"{best[0]} {best[1]}"
    return bpm, keyscale


def get_lyrics(path, *, filename=None, artist=None, title=None, duration=None,
               allow_online=True, whisper_size="small", language="en"):
    """Lyrics for a track: online lyrics DB for a known song (LRCLIB → lyrics.ovh),
    falling back to faster-whisper. Returns {lyrics, source, artist, title}."""
    try:
        from . import lyrics_fetch
        return lyrics_fetch.get_lyrics(path, filename=filename, artist=artist, title=title,
                                       duration=duration, allow_online=allow_online,
                                       whisper_size=whisper_size, language=language)
    except Exception:
        return {"lyrics": "", "source": "", "artist": artist, "title": title}


def _resolve_artist_title_full(path, *, filename=None, artist=None, title=None,
                               acoustid_key=""):
    """Resolve (artist, title) with full priority chain. Higher entries win:
        1. explicit args (caller passed them in)
        2. ID3 tags via mutagen (we trust the file's own metadata)
        3. AcoustID fingerprint (only when tags don't yield both)
        4. "Artist - Title" filename parse

    Returns (artist, title, source) where source ∈ {'explicit','tags','acoustid',
    'filename',''}. Promoting AcoustID above filename-parse means tag-less tracks
    still feed an accurate artist into LRCLIB so the lyric stage can succeed
    without falling back to whisper. ID3 wins over AcoustID per user preference
    — the file's own tags are higher signal than fingerprint guesswork."""
    src = ""
    try:
        from . import lyrics_fetch
        # Tier 1+2: explicit + ID3 tags (lyrics_fetch.resolve_artist_title handles both)
        a, t = lyrics_fetch.resolve_artist_title(path=path, filename=None,
                                                  artist=artist, title=title)
        if a and t:
            src = "explicit" if (artist and title) else "tags"
            return a, t, src
        # Tier 3: AcoustID fingerprint (only if we still need it)
        if acoustid_key:
            try:
                from . import caption_fetch
                ident = caption_fetch.acoustid_identify(path, acoustid_key)
                if ident:
                    ai, ti = ident
                    a = a or ai
                    t = t or ti
                    if a and t:
                        return a, t, "acoustid"
            except Exception:
                pass
        # Tier 4: "Artist - Title" filename parse (last resort, often noisy)
        a2, t2 = lyrics_fetch.parse_filename(filename or path or "")
        if (a2 and not a) or (t2 and not t):
            src = "filename"
        return (a or a2), (t or t2), src
    except Exception:
        return artist, title, src


def build_labels(path, *, instrumental=False, want_lyrics=True, caption=None,
                 language="en", timesignature="4", whisper_size="small",
                 filename=None, artist=None, title=None, allow_online=True,
                 want_caption=True, analyze_host="", lastfm_key="", acoustid_key=""):
    """Compute the label payload for one track. Returns
    {bpm, keyscale, lyrics, lyrics_source, caption, caption_sources, meta} where
    `meta` is the {name}.json dict. Lyrics + caption both prefer online sources for
    known songs (LRCLIB / MusicBrainz / Last.fm) and fall back to local/audio-based
    analysis (whisper / CLAP). The user edits the seeded caption in the review step.

    artist+title are pre-resolved here (explicit → ID3 → AcoustID → filename) so
    BOTH the lyric stage and the caption stage start with the best-available
    identifiers, and AcoustID fires only once per track (caption_fetch's own
    AcoustID block is guarded by `if not (artist and title)` so it skips when
    we've already resolved)."""
    bpm, keyscale = detect_bpm_key(path)
    artist, title, _id_source = _resolve_artist_title_full(
        path, filename=filename, artist=artist, title=title,
        acoustid_key=acoustid_key)
    lyrics, lyrics_source = "", ""
    if not instrumental and want_lyrics:
        try:
            import soundfile as sf
            dur = float(sf.info(path).duration)
        except Exception:
            dur = None
        got = get_lyrics(path, filename=filename or path, artist=artist, title=title,
                         duration=dur, allow_online=allow_online, whisper_size=whisper_size,
                         language=language)
        lyrics, lyrics_source = got["lyrics"], got["source"]
        # let lyrics resolution upgrade artist/title for the caption fetcher below
        artist = artist or got.get("artist")
        title = title or got.get("title")

    caption_sources = []
    if want_caption and not caption:
        try:
            from . import caption_fetch
            cap = caption_fetch.get_caption(audio_path=path, artist=artist, title=title,
                                            lastfm_key=lastfm_key, acoustid_key=acoustid_key,
                                            analyze_host=analyze_host, allow_clap=allow_online)
            caption = cap.get("caption") or ""
            caption_sources = cap.get("sources") or []
            artist = artist or cap.get("artist")
            title = title or cap.get("title")
        except Exception:
            caption = caption or ""

    meta = {"bpm": bpm, "keyscale": keyscale, "timesignature": str(timesignature),
            "language": language}
    if caption:
        meta["caption"] = caption
    # Persist lyrics_source so the UI can surface the LRCLIB/lyrics.ovh/whisper
    # badge on page reload (engine passes meta fields through /v1/dataset/samples,
    # see UI's tracks-from-samples rehydration in LoraTraining.tsx).
    if lyrics_source:
        meta["lyrics_source"] = lyrics_source
    return {"bpm": bpm, "keyscale": keyscale, "lyrics": lyrics, "lyrics_source": lyrics_source,
            "caption": caption or "", "caption_sources": caption_sources, "meta": meta}


def _stem(filename):
    return os.path.splitext(os.path.basename(filename))[0]


def bundle_for_track(audio_bytes, filename, *, instrumental=False, want_lyrics=True,
                     caption=None, language="en", timesignature="4", whisper_size="small",
                     artist=None, title=None, allow_online=True,
                     analyze_host="", lastfm_key="", acoustid_key=""):
    """Build the upload bundle for ONE track from its audio bytes + original filename.
    Writes the audio to a temp file to run librosa/whisper/online-lookup, then returns
    (files, info) where `files` = [(name, bytes), ...] ready for lora_upload_py.upload()
    and `info` = {name, bpm, keyscale, has_lyrics, lyrics_source, caption}.

    Lyrics prefer an online DB for a known song (artist/title from args, embedded tags,
    or a 'Artist - Title' filename) and fall back to whisper — see lyrics_fetch."""
    import tempfile
    suffix = os.path.splitext(filename)[1] or ".wav"
    name = _stem(filename)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
        tf.write(audio_bytes)
        tmp = tf.name
    try:
        lab = build_labels(tmp, instrumental=instrumental, want_lyrics=want_lyrics,
                           caption=caption, language=language, timesignature=timesignature,
                           whisper_size=whisper_size, filename=filename, artist=artist,
                           title=title, allow_online=allow_online, want_caption=True,
                           analyze_host=analyze_host, lastfm_key=lastfm_key,
                           acoustid_key=acoustid_key)
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass
    files = [(f"{name}{suffix}", audio_bytes),
             (f"{name}.json", json.dumps(lab["meta"], ensure_ascii=False, indent=2).encode("utf-8"))]
    if lab["lyrics"]:
        files.append((f"{name}.lyrics.txt", lab["lyrics"].encode("utf-8")))
    info = {"name": name, "bpm": lab["bpm"], "keyscale": lab["keyscale"],
            "has_lyrics": bool(lab["lyrics"]), "lyrics_source": lab["lyrics_source"],
            "caption": lab.get("caption", ""), "caption_sources": lab.get("caption_sources", [])}
    return files, info
