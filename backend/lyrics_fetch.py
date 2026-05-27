"""Fetch lyrics for a KNOWN song from open online sources, falling back to local
whisper transcription when the song isn't found.

Why: whisper is hit-and-miss on metal (strong accents, screams/growls), so for a track
we can identify by artist+title we'd rather pull accurate lyrics from a database. Primary
source is LRCLIB (lrclib.net) — free, no API key, community DB, good metal coverage,
returns `plainLyrics` directly; secondary is lyrics.ovh; last resort is whisper (asr.py).

Personal-use note: these lyrics are used as LoRA *training labels* (not redistributed).
LRCLIB is a crowdsourced lyrics database; we identify our client via User-Agent.

Artist/title resolution order: explicit args → embedded tags (mutagen, if present) →
"Artist - Title" filename parse.
"""
import os
import re

import requests

UA = "Crucible/0.1 (https://github.com/apothis/Crucible)"
_LRCLIB = "https://lrclib.net/api"


def _clean(txt):
    return (txt or "").strip()


def lrclib_get(artist, title, album=None, duration=None, timeout=10):
    """Exact lookup. Returns plainLyrics or None (None on 404/instrumental/no-plain)."""
    params = {"artist_name": artist, "track_name": title}
    if album:
        params["album_name"] = album
    if duration:
        params["duration"] = int(round(float(duration)))
    try:
        r = requests.get(_LRCLIB + "/get", params=params, headers={"User-Agent": UA}, timeout=timeout)
        if r.status_code != 200:
            return None
        d = r.json()
        if d.get("instrumental"):
            return None
        return _clean(d.get("plainLyrics")) or None
    except Exception:
        return None


def lrclib_search(artist, title, timeout=10):
    """Fuzzy fallback when /get 404s. Picks the first candidate with plain lyrics whose
    artist/title loosely match. Returns plainLyrics or None."""
    try:
        r = requests.get(_LRCLIB + "/search", params={"artist_name": artist, "track_name": title},
                         headers={"User-Agent": UA}, timeout=timeout)
        if r.status_code != 200:
            return None
        for cand in r.json():
            if cand.get("instrumental") or not cand.get("plainLyrics"):
                continue
            return _clean(cand["plainLyrics"])
    except Exception:
        pass
    return None


def lyrics_ovh(artist, title, timeout=8):
    """Secondary free source (often flaky). Returns lyrics or None."""
    try:
        r = requests.get(f"https://api.lyrics.ovh/v1/{requests.utils.quote(artist)}/{requests.utils.quote(title)}",
                         timeout=timeout)
        if r.status_code != 200:
            return None
        return _clean(r.json().get("lyrics")) or None
    except Exception:
        return None


def read_tags(path):
    """(artist, title) from embedded tags via mutagen, or (None, None)."""
    try:
        from mutagen import File as MFile
        m = MFile(path, easy=True)
        if not m:
            return None, None
        a = (m.get("artist") or [None])[0]
        t = (m.get("title") or [None])[0]
        return (_clean(a) or None), (_clean(t) or None)
    except Exception:
        return None, None


def parse_filename(filename):
    """(artist, title) from a 'Artist - Title' style filename, else (None, None)."""
    stem = os.path.splitext(os.path.basename(filename or ""))[0]
    stem = re.sub(r"^\s*\d+\s*[-.]\s*", "", stem)   # drop a leading track number
    if " - " in stem:
        a, t = stem.split(" - ", 1)
        return _clean(a) or None, _clean(t) or None
    return None, None


def resolve_artist_title(path=None, filename=None, artist=None, title=None):
    """Best-effort artist+title: explicit → tags → filename."""
    if artist and title:
        return _clean(artist), _clean(title)
    if path:
        a, t = read_tags(path)
        artist = artist or a
        title = title or t
    if not (artist and title):
        a, t = parse_filename(filename or path or "")
        artist = artist or a
        title = title or t
    return (_clean(artist) or None), (_clean(title) or None)


def fetch_online(artist, title, album=None, duration=None):
    """Try the online sources in order. Returns (lyrics, source) or (None, None)."""
    if not (artist and title):
        return None, None
    ly = lrclib_get(artist, title, album, duration)
    if ly:
        return ly, "lrclib"
    ly = lrclib_search(artist, title)
    if ly:
        return ly, "lrclib"
    ly = lyrics_ovh(artist, title)
    if ly:
        return ly, "lyrics.ovh"
    return None, None


def get_lyrics(path=None, *, filename=None, artist=None, title=None, album=None,
               duration=None, allow_online=True, allow_whisper=True, whisper_size="small",
               language="en"):
    """Best lyrics for a track. Tries online (for a known song) then falls back to
    whisper transcription of the local audio. Returns {lyrics, source, artist, title}
    where source ∈ {'lrclib','lyrics.ovh','whisper',''}."""
    artist, title = resolve_artist_title(path=path, filename=filename, artist=artist, title=title)
    if allow_online:
        ly, src = fetch_online(artist, title, album, duration)
        if ly:
            return {"lyrics": ly, "source": src, "artist": artist, "title": title}
    if allow_whisper and path:
        try:
            from . import asr as asr_mod
            txt = _clean(asr_mod.transcribe(path, size=whisper_size, language=language).get("text"))
            if txt:
                return {"lyrics": txt, "source": "whisper", "artist": artist, "title": title}
        except Exception:
            pass
    return {"lyrics": "", "source": "", "artist": artist, "title": title}
