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
            lines.append({"index": len(lines), "block": bi, "text": ln.strip(), "_toks": []})
            for part, echo in _bracket_parts(ln):
                for tok in normalize(part):
                    lines[-1]["_toks"].append((len(words), tok, echo))
                    words.append(tok)
                    owner.append(len(lines) - 1)
    return words, owner, lines


def _bracket_parts(line):
    """A lyric line split into (text, bracketed) parts: '(...)' marks an echo or an answer, the
    lyric convention for a backing voice repeating the call ("Vanadis (Vanadis)")."""
    out, pos = [], 0
    for m in re.finditer(r"\(([^)]*)\)", line):
        if m.start() > pos:
            out.append((line[pos:m.start()], False))
        out.append((m.group(1), True))
        pos = m.end()
    if pos < len(line):
        out.append((line[pos:], False))
    return out


ECHO_MIN = 0.3     # an echo span shorter than this is not kept


def _echo_spans(ln, at, next_start):
    """Song-second spans of a line's bracketed words. Placed by the transcript when Whisper heard
    them; otherwise (measured on Vanadis 2026-11-08: Whisper never transcribes the answer, it
    leaves a gap or stretches the next word over it) timed from the call it repeats: the answer
    starts where the call ends and lasts as long as the call, ending before the next sung word.
    `next_start`/(start, end) of the first placed word after the bracket, or None."""
    toks = ln["_toks"]
    out = []
    i = 0
    while i < len(toks):
        if not toks[i][2]:
            i += 1
            continue
        j = i
        while j < len(toks) and toks[j][2]:
            j += 1
        run = toks[i:j]
        hit = [at[k] for k, _, _ in run if k in at]
        before = toks[max(0, i - len(run)):i]
        repeat = [t for _, t, e in before if not e] == [t for _, t, _ in run]
        if hit and repeat and not any(k in at for k, _, _ in before):
            # the matcher gave the one heard word to the repeat; Whisper hears the call, never
            # the answer (measured), so the placed word is the call
            for (kb, _, _), (kr, _, _) in zip(before, run):
                if kr in at:
                    at[kb] = at.pop(kr)
            hit = []
        if hit:
            out.append([round(min(h[0] for h in hit), 3), round(max(h[1] for h in hit), 3)])
        else:
            call = [at[k] for k, _, e in before if not e and k in at]
            if call:
                a = max(h[1] for h in call)
                b = a + (a - min(h[0] for h in call))
                nxt = next((at[k] for k, _, _ in toks[j:] if k in at), None) or next_start(run[-1][0])
                if nxt:
                    b = min(b, nxt[0] if nxt[0] >= a + ECHO_MIN else nxt[1] - 0.15)
                if b - a >= ECHO_MIN:
                    out.append([round(a, 3), round(b, 3)])
        i = j
    return out


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


REPASS_MIN = 2.0     # seconds: a window shorter than this is not re-transcribed


def transcribe_window(audio_path, a, b, size="small", cache_dir=None):
    """Whisper words for one window of the track on its own, earlier text NOT carried forward
    (cached per file + window). Returns [(word, start, end)] in song seconds."""
    from . import asr
    cache = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        key = os.path.splitext(os.path.basename(audio_path))[0]
        cache = os.path.join(cache_dir, f"{key}.whisper-{size}.window_{a:.2f}_{b:.2f}.json")
        if os.path.exists(cache):
            with open(cache) as f:
                return [tuple(x) for x in json.load(f)]
    r = asr.transcribe(audio_path, size=size, language="en", with_words=True, vad=False,
                       clip=(a, b), condition_on_previous_text=False)
    out = [(w["word"], float(w["start"]), float(w["end"]))
           for seg in (r.get("segments") or []) for w in (seg.get("words") or [])
           if a - 0.05 <= float(w["start"]) < b]
    if cache:
        with open(cache, "w") as f:
            json.dump(out, f)
    return out


def _empty_block_windows(lines, at, duration):
    """Song windows of runs of lyric blocks that have sung (non-bracketed) words and fewer than
    LINE_MIN_COVER of them placed (a skipped passage still picks up stray matches: the bridge's
    'the', 'and', 'you' landed on the repeated verse): [(a, b, [block indices])]. A window runs
    from the last placed word before the run to the first placed word after it; wordless blocks
    (a solo) inside it do not break the run."""
    by_block = {}
    for ln in lines:
        by_block.setdefault(ln["block"], []).extend(ln["_toks"])
    runs, cur = [], []
    for bi in sorted(by_block):
        toks = by_block[bi]
        sung = [k for k, _, e in toks if not e]
        if sung and sum(1 for k in sung if k in at) < LINE_MIN_COVER * len(sung):
            cur.append(bi)
        elif sung:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    line_of = {k: ln["index"] for ln in lines for k, _, _ in ln["_toks"]}
    out = []
    for run in runs:
        first = min(k for bi in run for k, _, _ in by_block[bi])
        last = max(k for bi in run for k, _, _ in by_block[bi])
        before = [k for k in at if k < first]
        after = [k for k in at if k > last]
        # the window takes in the whole placed line on each side: the first pass's stamps next
        # to a skipped passage are damaged too (the bridge's next line had three words on one
        # instant, 1.3 s late), so those lines are heard again with it
        if before:
            ln = line_of[max(before)]
            a = min(at[k][0] for k in before if line_of[k] == ln)
        else:
            a = 0.0
        if after:
            ln = line_of[min(after)]
            b = max(at[k][1] for k in after if line_of[k] == ln) + 0.1
        else:
            b = float(duration or 0.0)
        if b - a >= REPASS_MIN:
            out.append((round(a, 3), round(b, 3), run))
    return out


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
    def _match(heard, times):
        at = {}
        if heard:
            for blk in difflib.SequenceMatcher(a=lyr, b=heard, autojunk=False).get_matching_blocks():
                for k in range(blk.size):
                    at[blk.a + k] = times[blk.b + k]
        return at

    at = _match(heard, times)

    # SECOND PASS: a block of sung lines with under half its words placed is a passage Whisper skipped
    # (measured, Nobody Stays Dead 2026-11-13: it heard nothing over the bridge, then repeated
    # verse 2 with every word stamped on one instant). That window alone, transcribed again
    # without earlier text carried forward, heard all four bridge lines; its words replace the
    # first pass's inside the window and the whole song is matched again (one monotonic match).
    # Blocks that are only bracketed chants are left out: they were never placed before either.
    repass = []
    if heard:
        for a, b, run in _empty_block_windows(lines, at, tr.get("duration")):
            got = transcribe_window(audio_path, a, b, size=size, cache_dir=cache_dir)
            words = [(tok, (s0, e0)) for w, s0, e0 in got for tok in normalize(w)]
            keep = [(h, t) for h, t in zip(heard, times) if not a <= t[0] < b]
            merged = sorted(keep + words, key=lambda x: x[1][0])
            heard, times = [h for h, _ in merged], [t for _, t in merged]
            repass.append({"window": [a, b], "blocks": [blocks[bi].get("type") for bi in run],
                           "words_heard": len(words)})
        if repass:
            at = _match(heard, times)

    def next_start(k):
        for m in range(k + 1, len(lyr)):
            if m in at:
                return at[m]
        return None

    for ln in lines:
        ln["echo"] = _echo_spans(ln, at, next_start)
        ln["word_starts"] = [round(at[k][0], 3) for k, _, e in ln["_toks"] if not e and k in at]
        ln.pop("_toks", None)
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
            "words_total": len(lyr), "duration": tr.get("duration"), "repass": repass}
