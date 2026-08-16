"""Where the lyrics ACTUALLY land in the finished audio (Mac, CPU, no GPU).

The Song tab's block seconds are what we ASK ACE-Step for, not what it delivers: it fits the lyrics
and sections where it likes. Everything the video writer does with lyrics - which words are sung
under a shot, where a voice hands over - was derived from those nominal seconds via a section-label
mapping, which is why casting kept landing a beat or two out. MEASURED on "Dream of Me"
(2026-08-16): sections drift -1.1s to +7.0s against nominal, growing through the song.

METHOD: transcribe the mix with Whisper (word timestamps), then align that word stream against the
lyrics we already have. We are not asking Whisper to be right about the words - only to put words we
already know onto the clock - so its mistakes cost us nothing as long as enough of the stream
matches, and the match RATE is a genuine confidence signal.

Two things here were measured, not assumed, and both went against expectation:

  - The RAW MIX beats a Demucs vocal stem, 3-5x, on every section tried (0.22-0.65 vs 0.07-0.13
    under CTC alignment). Standard karaoke advice is to separate first; on our generated material it
    made things much worse, so we align the mix and skip the separation pass entirely.
  - Whisper + fuzzy match beats torchaudio's MMS_FA forced aligner outright: 95% of lyric words
    placed with every line in order, against MMS_FA's 0.22-0.65 confidence and a total collapse
    (0.02 everywhere) when run over a whole song in one pass. MMS_FA also relies on
    torchaudio.functional.forced_align, which is deprecated and removed in torchaudio 2.9.

VAD IS OFF on purpose: the voice-activity filter is tuned for speech and drops sustained sung vowels
and quiet phrase tails - exactly the material we need timed.
"""
import json
import os
import re

# a lyric line matching below this fraction of its words is not trustworthy enough to move a cut to
LINE_MIN_COVER = 0.5


def normalize(line):
    """Words reduced to lowercase a-z plus the apostrophe, so the lyric stream and the transcript
    are comparable. Punctuation, digits and section markers drop out."""
    out = []
    for w in re.split(r"\s+", (line or "").strip().lower()):
        w = re.sub(r"[^a-z']", "", w).strip("'")
        if w:
            out.append(w)
    return out


def _lyric_stream(blocks):
    """The lyrics as one word stream, each word tagged with (block index, line index), plus the
    line list. One stream for the whole song keeps the match monotonic, so a repeated chorus cannot
    be matched to the wrong repeat."""
    words, owner, lines = [], [], []
    for bi, b in enumerate(blocks or []):
        for ln in (b.get("lyrics") or "").split("\n"):
            if not ln.strip():
                continue
            lines.append({"index": len(lines), "block": bi, "text": ln.strip()})
            for tok in normalize(ln):
                words.append(tok)
                owner.append(len(lines) - 1)
    return words, owner, lines


def transcribe_words(audio_path, size="small", cache_dir=None):
    """Whisper word timestamps for a track, cached per file+model (the expensive step: ~100s for a
    4-minute song at `small` on CPU)."""
    from . import asr
    cache = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        key = os.path.splitext(os.path.basename(audio_path))[0]
        cache = os.path.join(cache_dir, f"{key}.whisper-{size}.json")
        if os.path.exists(cache):
            with open(cache) as f:
                return json.load(f)
    r = asr.transcribe(audio_path, size=size, language="en", with_words=True, vad=False)
    if cache:
        with open(cache, "w") as f:
            json.dump(r, f)
    return r


def align_song(audio_path, blocks, size="small", cache_dir=None):
    """Measure where every lyric line and section actually lands.

    `blocks` = the Song tab arrangement [{type, seconds, lyrics, style}, ...]. Returns
    {lines, sections, cover, words_matched, words_total}, all times in REAL seconds.

    `cover` (0..1) is the share of lyric words placed - the confidence gate. Callers should fall
    back to the nominal timeline, and SAY they have, rather than trust a low-cover result.
    Instrumental blocks carry no measured span: they are the gaps between sung ones, and the
    audio-structure grid already covers those."""
    import difflib

    lyr, owner, lines = _lyric_stream(blocks)
    if not lyr:
        return {"lines": [], "sections": [], "cover": 0.0, "words_matched": 0, "words_total": 0}

    tr = transcribe_words(audio_path, size=size, cache_dir=cache_dir)
    heard, times = [], []
    for seg in (tr.get("segments") or []):
        for w in (seg.get("words") or []):
            for tok in normalize(w.get("word")):
                heard.append(tok)
                times.append((float(w["start"]), float(w["end"])))

    # longest-common-subsequence style match of the two word streams. Whisper mishearing a word
    # just leaves that word unplaced; the surrounding matches still carry the timing.
    at = {}
    if heard:
        for blk in difflib.SequenceMatcher(a=lyr, b=heard, autojunk=False).get_matching_blocks():
            for k in range(blk.size):
                at[blk.a + k] = times[blk.b + k]

    for ln in lines:
        idx = [i for i, o in enumerate(owner) if o == ln["index"]]
        hit = [at[i] for i in idx if i in at]
        ln["cover"] = round(len(hit) / len(idx), 3) if idx else 0.0
        ln["start"] = round(min(h[0] for h in hit), 3) if hit else None
        ln["end"] = round(max(h[1] for h in hit), 3) if hit else None
        ln["ok"] = bool(hit) and ln["cover"] >= LINE_MIN_COVER

    nominal, t = [], 0.0
    for b in (blocks or []):
        nominal.append(t)
        t += float(b.get("seconds") or 0)

    sections = []
    for bi, b in enumerate(blocks or []):
        mine = [l for l in lines if l["block"] == bi and l["start"] is not None]
        good = [l for l in mine if l["ok"]]
        sections.append({
            "index": bi, "type": b.get("type"), "style": b.get("style") or "",
            "nominal_start": round(nominal[bi], 3),
            "nominal_end": round(nominal[bi] + float(b.get("seconds") or 0), 3),
            "start": round(min(l["start"] for l in good), 3) if good else None,
            "end": round(max(l["end"] for l in good), 3) if good else None,
            "cover": round(sum(l["cover"] for l in mine) / len(mine), 3) if mine else None,
            "lines": mine,
        })

    matched = len(at)
    return {"lines": lines, "sections": sections,
            "cover": round(matched / len(lyr), 3), "words_matched": matched,
            "words_total": len(lyr), "duration": tr.get("duration")}
