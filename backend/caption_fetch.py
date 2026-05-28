"""Auto-build a CAPTION seed for a LoRA training track from multiple sources.

Layered, all optional (graceful degradation):
  1. **MusicBrainz** (free, no key) — recording-level tags + artist disambiguation
     ("English heavy metal band") for a known artist/title.
  2. **Last.fm** (free key, optional) — `track.getTopTags` for richer subgenre tags.
  3. **CLAP via the box analyze service** (audio-based, optional) — reuses the §17a
     service to get style tags directly from the audio (works even when the song
     is unknown).
  4. **AcoustID + fpcalc** (optional) — audio fingerprint → artist+title for files
     that lack tags/filename metadata, then loops back into (1)/(2).

Returns {caption, tags, sources, artist, title} ready to drop into `{name}.json` as
the caption seed; the user reviews/edits in the Training-tab Step 2 (METAL_LORA_PLAN
§10.4 — descriptive style captions are what makes the LoRA work).
"""
import os
import re
import subprocess

import requests

UA = "Crucible/0.1 (https://github.com/apothis/Crucible)"
_MB = "https://musicbrainz.org/ws/2"
_LFM = "http://ws.audioscrobbler.com/2.0/"
_AID = "https://api.acoustid.org/v2/lookup"


def _clean(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip().lower())


# Tags that don't add useful style information on their own.
_GENERIC = {"music", "song", "track", "instrumental", "vocal", "vocals", "good", "favourite",
            "favourites", "favorite", "favorites", "seen live", "albums i own", "spotify"}
# Tokens that mark a tag as a specific genre/style — used to put the *specific* ones first.
_SPECIFIC = {"thrash", "doom", "black", "death", "power", "symphonic", "speed", "groove",
             "djent", "industrial", "viking", "pirate", "folk", "melodic", "progressive",
             "gothic", "stoner", "sludge", "hardcore", "blackened", "technical",
             "atmospheric", "epic", "nwobhm", "deathcore", "metalcore", "grindcore"}


def _rank(tag: str) -> int:
    t = tag.lower()
    if any(s in t for s in _SPECIFIC):
        return 0          # most useful — specific subgenres lead
    if "metal" in t:
        return 2
    if "rock" in t:
        return 3
    return 5


def _merge(tag_lists, limit=10):
    """Union + de-dupe + sort by specificity."""
    seen, out = set(), []
    for src in tag_lists:
        for t in src:
            t = _clean(t)
            if not t or t in seen or t in _GENERIC or len(t) > 40:
                continue
            seen.add(t)
            out.append(t)
    return sorted(out, key=_rank)[:limit]


def musicbrainz_tags(artist, title, timeout=10):
    """Recording-level tags + artist disambiguation. Free, no key.
    Returns (tags:list, artist_blurb:str)."""
    if not (artist and title):
        return [], ""
    try:
        q = f'recording:"{title}" AND artist:"{artist}"'
        r = requests.get(f"{_MB}/recording", params={"query": q, "fmt": "json", "limit": 3},
                         headers={"User-Agent": UA, "Accept": "application/json"}, timeout=timeout)
        if r.status_code != 200:
            return [], ""
        recs = r.json().get("recordings") or []
        if not recs:
            return [], ""
        tags = [t["name"] for t in (recs[0].get("tags") or []) if t.get("name")]
        ac = (recs[0].get("artist-credit") or [{}])[0]
        blurb = _clean(((ac.get("artist") or {}).get("disambiguation") or ""))
        return tags, blurb
    except Exception:
        return [], ""


def lastfm_tags(artist, title, key, timeout=10):
    """track.getTopTags via Last.fm (requires free key). Empty list when key missing."""
    if not (artist and title and key):
        return []
    try:
        r = requests.get(_LFM, params={"method": "track.getTopTags", "artist": artist,
                                       "track": title, "api_key": key, "format": "json",
                                       "autocorrect": 1}, timeout=timeout)
        if r.status_code != 200:
            return []
        tags = (r.json().get("toptags") or {}).get("tag") or []
        return [t["name"] for t in tags[:10] if t.get("name")]
    except Exception:
        return []


def acoustid_identify(audio_path, key, fpcalc_bin="fpcalc", timeout=25):
    """Audio fingerprint → AcoustID → (artist, title) or None. Needs free key + fpcalc."""
    if not (key and audio_path and os.path.exists(audio_path)):
        return None
    try:
        r = subprocess.run([fpcalc_bin, "-json", audio_path], capture_output=True, timeout=timeout)
        if r.returncode != 0:
            return None
        import json as _json
        f = _json.loads(r.stdout.decode("utf-8", "ignore"))
        dur, fp = f.get("duration"), f.get("fingerprint")
        if not (dur and fp):
            return None
        rr = requests.get(_AID, params={"client": key, "duration": int(dur),
                                        "fingerprint": fp, "meta": "recordings"}, timeout=timeout)
        if rr.status_code != 200:
            return None
        for res in rr.json().get("results", []):
            for rec in res.get("recordings", []) or []:
                title = rec.get("title")
                artists = [a.get("name") for a in (rec.get("artists") or []) if a.get("name")]
                if title and artists:
                    return artists[0], title
    except Exception:
        return None
    return None


def clap_tags(audio_path, analyze_host, timeout=180):
    """Reuse the §17a box analyze service: CLAP zero-shot style tags from the audio.
    Empty list when analyze_host isn't set or the call fails."""
    if not (audio_path and analyze_host and os.path.exists(audio_path)):
        return []
    try:
        from . import analyze_py
        out = analyze_py.analyze(analyze_host, audio_path, with_tags=True, with_key=False, timeout=timeout)
        return list(out.get("tags") or [])
    except Exception:
        return []


def get_caption(audio_path=None, *, artist=None, title=None,
                lastfm_key="", acoustid_key="", analyze_host="",
                allow_clap=True, max_tags=8):
    """Build a caption seed by layering the available sources. All sources are
    optional; missing keys/services just skip that layer. The user edits the result
    in the review step. Returns:
        {caption, tags, sources, artist, title, artist_info}"""
    sources = []
    # 1. Identify untagged audio if we have AcoustID set up
    if (not (artist and title)) and audio_path and acoustid_key:
        ident = acoustid_identify(audio_path, acoustid_key)
        if ident:
            artist, title = ident
            sources.append("acoustid")
    # 2. MusicBrainz tags + artist disambiguation
    mb_tags, artist_info = ([], "")
    if artist and title:
        mb_tags, artist_info = musicbrainz_tags(artist, title)
        if mb_tags or artist_info:
            sources.append("musicbrainz")
    # 3. Last.fm tags (richer; opt-in)
    lfm = lastfm_tags(artist, title, lastfm_key) if (artist and title and lastfm_key) else []
    if lfm:
        sources.append("lastfm")
    # 4. CLAP audio-based tags from the box (works without artist/title)
    clap = clap_tags(audio_path, analyze_host) if (allow_clap and analyze_host and audio_path) else []
    if clap:
        sources.append("clap")

    tags = _merge([mb_tags, lfm, clap], limit=max_tags)
    pieces = []
    if artist_info:    # e.g. "english heavy metal band"
        pieces.append(artist_info)
    pieces.extend(tags)
    caption = ", ".join(pieces)
    return {"caption": caption, "tags": tags, "sources": sources,
            "artist": artist, "title": title, "artist_info": artist_info}
