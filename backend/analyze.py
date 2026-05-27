"""Reference-track analysis (Mac, librosa) → a Song-Constructor arrangement spec.

Upload/pick a reference song → detect BPM, key, and a labelled section structure
(+ optional lyrics) → seed the Song Builder so the user can generate a SIMILAR song via
normal text2music (NOT the cover task). See RESEARCH.md §17 (this = P1, all Mac-side; P2
swaps structure/bpm/tags to the box `allin1`+CLAP service for higher quality).

Note: ACE-Step text2music has no melody input, so the generated song matches
bpm/key/structure/(lyrics), not the literal tune — by design.
"""
HOP = 512
SR = 22050

# Pitch-class names matching comfy.KEYS exactly (mixed sharps/flats) so the detected key
# is a valid Song-Builder key option.
PITCHES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
# Krumhansl–Schmuckler key profiles (major / minor).
_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

# auto_blocks role strings → Song-Constructor section types (forms.tsx SECTION_TYPES).
_ROLE_MAP = {"intro": "Intro", "verse": "Verse", "prechorus": "Pre-Chorus", "pre-chorus": "Pre-Chorus",
             "chorus": "Chorus", "bridge": "Bridge", "solo": "Solo", "breakdown": "Breakdown", "outro": "Outro"}

# allin1 (Harmonix) segment labels → Song-Constructor section types (for the P2 box path).
_ALLIN1_MAP = {"intro": "Intro", "verse": "Verse", "chorus": "Chorus", "bridge": "Bridge",
               "inst": "Solo", "solo": "Solo", "break": "Breakdown", "breakdown": "Breakdown",
               "prechorus": "Pre-Chorus", "outro": "Outro"}


def blocks_from_allin1(segments):
    """Map allin1 segments [{start,end,label}] → Song-Constructor blocks
    [{type,seconds,lyrics}], dropping non-musical markers (start/end/silence)."""
    blocks = []
    for s in segments or []:
        lab = str(s.get("label", "")).lower().strip()
        if lab in ("start", "end", "silence", ""):
            continue
        secs = max(1, int(round(float(s.get("end", 0)) - float(s.get("start", 0)))))
        blocks.append({"type": _ALLIN1_MAP.get(lab, "Verse"), "seconds": secs, "lyrics": ""})
    return blocks


def map_lyric_segments(blocks_with_times, tsegs):
    """Fill per-block lyrics from transcript segments by timestamp midpoint.
    `blocks_with_times` = [(block_dict, start, end)]."""
    for blk, t0, t1 in blocks_with_times:
        txt = " ".join(t["text"] for t in (tsegs or [])
                       if t0 <= (t["start"] + t["end"]) / 2.0 < t1).strip()
        if txt:
            blk["lyrics"] = txt


def available():
    try:
        import librosa  # noqa: F401
        import numpy     # noqa: F401
        import sklearn   # noqa: F401  (sections.py agglomerative backend)
        return True
    except Exception:
        return False


def _detect_key(y, sr):
    import numpy as np
    import librosa
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP).mean(axis=1)
    if not np.isfinite(chroma).all() or chroma.sum() <= 0:
        return "E minor"
    best = None
    for mode, prof in (("major", _MAJOR), ("minor", _MINOR)):
        p = np.array(prof)
        for i in range(12):
            corr = np.corrcoef(chroma, np.roll(p, i))[0, 1]
            if np.isfinite(corr) and (best is None or corr > best[0]):
                best = (float(corr), f"{PITCHES[i]} {mode}")
    return best[1] if best else "E minor"


def _roles_from_segments(segs):
    """Infer a section role per segment from position + energy (mirrors
    sections.auto_blocks): first→intro, last→outro, loud middles→chorus, else verse."""
    n = len(segs)
    if n == 1:
        return ["Verse"]
    energies = [s["energy"] for s in segs]
    mid = sorted(energies[1:-1]) if n > 2 else []
    thresh = mid[len(mid) // 2] if mid else 0.0
    roles = []
    for i, s in enumerate(segs):
        if i == 0:
            roles.append("Intro")
        elif i == n - 1:
            roles.append("Outro")
        else:
            roles.append("Chorus" if s["energy"] >= thresh else "Verse")
    return roles


def analyze_reference(path, with_lyrics=False, asr_size="small"):
    """Analyse `path` → {bpm, key, duration, blocks:[{type,seconds,lyrics}], has_vocals,
    lyrics_text}. Blocks form a Song-Constructor arrangement ready to load into the builder."""
    import numpy as np
    import librosa
    from . import sections as sections_mod

    y, sr = librosa.load(path, sr=SR, mono=True)
    dur = len(y) / float(sr) if y.size else 0.0

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = int(round(float(np.atleast_1d(tempo)[0]))) if np.atleast_1d(tempo).size else 0
    bpm = bpm or 120

    key = _detect_key(y, sr)

    segs = sections_mod.detect_segments(path)               # [{start,end,energy}]
    roles = _roles_from_segments(segs)
    blocks = [{"type": roles[i], "seconds": max(1, int(round(s["end"] - s["start"]))), "lyrics": ""}
              for i, s in enumerate(segs)]

    lyrics_text = ""
    has_vocals = False
    tsegs = []
    if with_lyrics:
        try:
            from . import asr as asr_mod
            r = asr_mod.transcribe(path, size=asr_size, with_segments=True)
            lyrics_text = (r.get("text") or "").strip()
            has_vocals = bool(lyrics_text)
            # map each transcript segment to the section that contains its midpoint
            tsegs = r.get("segments") or []
            if tsegs:
                for i, s in enumerate(segs):
                    txt = " ".join(t["text"] for t in tsegs
                                   if s["start"] <= (t["start"] + t["end"]) / 2.0 < s["end"]).strip()
                    if txt:
                        blocks[i]["lyrics"] = txt
        except Exception:
            pass

    return {"bpm": bpm, "key": key, "duration": round(dur, 1),
            "blocks": blocks, "has_vocals": has_vocals, "lyrics_text": lyrics_text,
            "lyric_segments": tsegs}   # raw transcript (start/end/text) to remap onto box sections
