"""Who sings when, measured from the vocal stem: the voice timeline for two-singer music videos.

The audio decides, not the lyric tags. Speaker fingerprints of every aligned line and of 1.5 s
windows (0.75 s hop, the WeSpeaker diarisation defaults) over the voiced stretches of the stem
come from the box helper's POST /voice/timeline (WeSpeaker ResNet34, CPU, a few seconds). Here,
on the Mac, the windows are sorted into THREE groups with spherical k-means: the two singers and
the instrument bleed the separator leaves in the stem (measured on Vanadis 2026-11-07: with two
groups the bleed of the intro and the solos merged into the second voice; with three it took a
group of its own, and the verse voice and the chorus voice split cleanly, which the user
confirmed by ear).

Which group is which:
  bleed = the group with the largest share of its windows outside any sung line;
  him   = of the two voices, the one the lyric tags call male ([Male Vocal], [Male], [Growl]...)
          more often. The tags only NAME the groups; the times come from the audio. With no
          gender tags, the voice with less sung time is him (she is the lead). `swap` flips it.
Pitch is not used: a growled male voice reads as 340-370 Hz to pyin (measured, Vanadis).
"""
import hashlib
import json
import os
import re
import subprocess
import wave

import numpy as np
import requests

from . import lyricalign

VERSION = 2
SR = 16000
WIN, HOP = 1.5, 0.75          # window and hop, seconds
VOICED_DB = 30.0              # a 50 ms frame is voiced within this many dB of the stem's loudest
VOICED_FRAC = 0.6             # a window is measured when this share of it is voiced
MIN_RUN = 1.2                 # a voice run shorter than this merges into its neighbour
MIXED_MIN = 1.0               # a line is "mixed" when the other voice holds this many seconds of it
MIXED_FRAC = 0.25             # ... and this share of its length
UNSURE_MARGIN = 0.10          # a line whose own fingerprint is this close to both voices is unsure
UNSURE_FLOOR = 0.40           # ... or this far from both (a choir, a layered shout): B-roll, not lip-sync
# Vanadis 2026-11-07: clean lines sit 0.2-0.5 apart; the unsure ones were 2:35 (0.04), 3:49 (0.05) and
# 4:08 (0.37 / 0.26), the same lines the window strips showed as borderline
# ANSWERS: a short call and answer inside one line ("Vanadis (Vanadis)": he calls, she answers,
# under a second each) is too short for the 1.5 s windows (measured 2026-11-07: every window over
# the Vanadis choruses read him, the user heard her answers). A fine pass of short windows over the
# sung stretches finds stretches of the OTHER voice inside a run; they stay off the runs (the line
# is not a shared line) and are listed as answers, which a lip-sync render can mute from what the
# singer on screen is driven by.
FINE_WIN, FINE_HOP = 0.6, 0.2  # fine windows, seconds
FINE_PAD = (0.3, 1.6)          # fine windows cover each line from this before its start to this after its end
FINE_MARGIN = 0.08             # a fine window names a voice when nearer to it than to the other by this
ANSWER_MIN_WINS = 2            # ... and an answer is this many consecutive such windows
MALE_TAG = re.compile(r"(?i)\b(male|man|men|boy|baritone|bass|tenor|growl\w*|harsh|scream\w*|death)\b")
FEMALE_TAG = re.compile(r"(?i)\b(female|woman|women|girl|soprano|alto|mezzo)\b")


def cache_path(library, audio_id):
    d = os.path.join(os.path.dirname(library), ".mvwork", "mv_voices")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{os.path.basename(audio_id)}.json")


def _stem16k(stem_path, out_path):
    if not os.path.exists(out_path) or os.path.getmtime(out_path) < os.path.getmtime(stem_path):
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", stem_path, "-ac", "1",
                        "-ar", str(SR), "-c:a", "pcm_s16le", out_path], check=True)
    with wave.open(out_path) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32)
    return x


def _segments(x, sections):
    hop = int(0.05 * SR)
    n = len(x) // hop
    rms = np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-9)
    db = 20 * np.log10(rms)
    on = db > db.max() - VOICED_DB
    lines = [[float(l["start"]), float(l["end"])] for s in sections for l in s.get("lines") or []]
    wins, t = [], 0.0
    while t + WIN <= len(x) / SR:
        a, b = int(t / 0.05), int((t + WIN) / 0.05)
        if on[a:b].mean() >= VOICED_FRAC:
            wins.append([round(t, 2), round(t + WIN, 2)])
        t += HOP
    spans = []
    for a, b in sorted(lines):
        a, b = max(0.0, a - FINE_PAD[0]), b + FINE_PAD[1]
        if spans and a <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    fine = []
    for a, b in spans:
        t = a
        while t + FINE_WIN <= min(b, len(x) / SR):
            i, j = int(t / 0.05), int((t + FINE_WIN) / 0.05)
            if on[i:j].mean() >= VOICED_FRAC:
                fine.append([round(t, 2), round(t + FINE_WIN, 2)])
            t += FINE_HOP
    return lines, wins, fine


def _kmeans(X, k=3, inits=40, iters=60, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(inits):
        C = X[rng.choice(len(X), k, replace=False)]
        for _ in range(iters):
            lab = np.argmax(X @ C.T, 1)
            C = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
            C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
        lab = np.argmax(X @ C.T, 1)
        score = float((X @ C.T)[np.arange(len(X)), lab].sum())
        if best is None or score > best[0]:
            best = (score, C, lab)
    return best[1], best[2]


def fingerprint(stem_path, sections, helper_host, upload, cache_file, log=print):
    """Embeddings for every aligned line and voiced window of the stem, cached in cache_file
    (keyed on the stem's size+mtime and the line times). upload(bytes, name) -> ComfyUI input
    name. Returns {lines, windows, line_emb, win_emb}."""
    x = None
    key_src = f"{VERSION}|{os.path.getsize(stem_path)}|{int(os.path.getmtime(stem_path))}|" + json.dumps(
        [[l["start"], l["end"]] for s in sections for l in s.get("lines") or []])
    key = hashlib.md5(key_src.encode()).hexdigest()
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                c = json.load(f)
            if c.get("key") == key:
                return c
        except Exception:
            pass
    wav16 = os.path.splitext(cache_file)[0] + ".16k.wav"
    x = _stem16k(stem_path, wav16)
    lines, wins, fine = _segments(x, sections)
    with open(wav16, "rb") as f:
        data = f.read()
    name = upload(data, f"voice_{hashlib.md5(data).hexdigest()[:16]}.wav")
    log(f"[voices] fingerprinting {len(lines)} lines + {len(wins)} windows + {len(fine)} fine windows on the box")
    r = requests.post(f"http://{helper_host}/voice/timeline",
                      json={"input_name": name, "segments": lines + wins + fine}, timeout=1800)
    if r.status_code == 404 and "Not Found" in r.text:
        raise RuntimeError("the box helper predates /voice/timeline: redeploy lora_upload_server.py and restart it")
    r.raise_for_status()
    res = r.json()
    emb = res["embeddings"]
    n1, n2 = len(lines), len(lines) + len(wins)
    c = {"key": key, "lines": lines, "windows": wins, "fine": fine, "line_emb": emb[:n1],
         "win_emb": emb[n1:n2], "fine_emb": emb[n2:], "provider": res.get("provider"), "summary": res.get("summary")}
    with open(cache_file, "w") as f:
        json.dump(c, f)
    return c


def _tag_genders(lyrics_raw):
    """normalized lyric line -> 'him' | 'her' from the block headers' vocal tags (or absent)."""
    out, cur = {}, None
    for raw in (lyrics_raw or "").splitlines():
        s = raw.strip()
        if s.startswith("["):
            tags = " ".join(re.findall(r"\[([^\]]*)\]", s))
            m, f = bool(MALE_TAG.search(tags)), bool(FEMALE_TAG.search(tags))
            cur = "him" if m and not f else "her" if f and not m else None
            continue
        if s and cur:
            out.setdefault(" ".join(lyricalign.normalize(s)), cur)
    return out


def _runs(wins, labels, him, her):
    """Window labels -> contiguous voice runs [{start, end, voice}] over the voiced windows; runs
    shorter than MIN_RUN merge into the previous run (or the next one at the start)."""
    pts = [((a + b) / 2, "him" if l == him else "her" if l == her else None) for (a, b), l in zip(wins, labels)]
    pts = [p for p in pts if p[1]]
    runs = []
    for i, (t, v) in enumerate(pts):
        lo = (pts[i - 1][0] + t) / 2 if i and t - pts[i - 1][0] <= 2 * HOP + 1e-6 else t - WIN / 2
        hi = (t + pts[i + 1][0]) / 2 if i + 1 < len(pts) and pts[i + 1][0] - t <= 2 * HOP + 1e-6 else t + WIN / 2
        if runs and runs[-1]["voice"] == v and lo - runs[-1]["end"] < 1e-6:
            runs[-1]["end"] = hi
        else:
            runs.append({"start": lo, "end": hi, "voice": v})
    merged = []
    for r in runs:
        if merged and (r["end"] - r["start"] < MIN_RUN) and abs(r["start"] - merged[-1]["end"]) < 1e-6:
            merged[-1]["end"] = r["end"]
        elif merged and merged[-1]["voice"] == r["voice"] and r["start"] - merged[-1]["end"] < WIN:
            merged[-1]["end"] = r["end"]
        else:
            merged.append(dict(r))
    if len(merged) > 1 and merged[0]["end"] - merged[0]["start"] < MIN_RUN:
        merged[1]["start"] = merged[0]["start"]
        merged.pop(0)
    for r in merged:
        r["start"], r["end"] = round(r["start"], 2), round(r["end"], 2)
    return merged


def _answers(fp, C, her, him, bleed, runs):
    """Short stretches of the other voice inside a run (a call and answer), from the fine windows:
    [{start, end, voice}]. A fine window names a voice when it is FINE_MARGIN nearer to it than to
    the other and nearer to it than to the bleed; ANSWER_MIN_WINS consecutive windows naming the
    voice the run does not hold make an answer."""
    F = np.array(fp.get("fine_emb") or [], float)
    if not len(F):
        return []
    sim = F @ C.T
    out, cur = [], None
    for (a, b), s in zip(fp["fine"], sim):
        t = (a + b) / 2
        base = next((r["voice"] for r in runs if r["start"] <= t < r["end"]), None)
        v = None
        if s[her] - s[him] >= FINE_MARGIN and s[her] > s[bleed]:
            v = "her"
        elif s[him] - s[her] >= FINE_MARGIN and s[him] > s[bleed]:
            v = "him"
        hit = v if base and v and v != base else None
        if hit and cur and cur["voice"] == hit and a - cur["last"] <= FINE_HOP + 1e-6:
            cur["end"], cur["last"], cur["n"] = b, a, cur["n"] + 1
            continue
        if cur and cur["n"] >= ANSWER_MIN_WINS:
            out.append(cur)
        cur = {"start": a, "end": b, "last": a, "n": 1, "voice": hit} if hit else None
    if cur and cur["n"] >= ANSWER_MIN_WINS:
        out.append(cur)
    return [{"start": round(x["start"], 2), "end": round(x["end"], 2), "voice": x["voice"]} for x in out]


def voice_seconds(runs, a, b):
    """Seconds of each voice inside [a, b)."""
    out = {"her": 0.0, "him": 0.0}
    for r in runs:
        o = min(b, r["end"]) - max(a, r["start"])
        if o > 0:
            out[r["voice"]] += o
    return out


def timeline(fp, sections, lyrics_raw, swap=False):
    """The labelled voice timeline from the fingerprints: {runs, lines: [{start, end, text, voice,
    seconds}], assign, how, separation}."""
    W = np.array(fp["win_emb"], float)
    wins = fp["windows"]
    C, lab = _kmeans(W, 3)
    line_spans = [(l["start"], l["end"]) for s in sections for l in s.get("lines") or []]

    def in_line(a, b):
        return any(b > s and a < e for s, e in line_spans)
    outside = [np.mean([not in_line(*wins[i]) for i in np.where(lab == j)[0]]) if (lab == j).any() else 1.0
               for j in range(3)]
    bleed = int(np.argmax(outside))
    voices = [j for j in range(3) if j != bleed]
    # bleed windows inside sung lines take the nearer of the two voices
    sim = W @ C.T
    lab = lab.copy()
    for i, (a, b) in enumerate(wins):
        if lab[i] == bleed and in_line(a, b):
            lab[i] = voices[int(np.argmax(sim[i, voices]))]
    tags = _tag_genders(lyrics_raw)
    score = {j: 0 for j in voices}
    sung = {j: 0.0 for j in voices}
    all_lines = [l for s in sections for l in s.get("lines") or []]
    for l in all_lines:
        for i, (a, b) in enumerate(wins):
            o = min(b, l["end"]) - max(a, l["start"])
            if o > 0 and lab[i] in score:
                sung[lab[i]] += o
                g = tags.get(" ".join(lyricalign.normalize(l["text"])))
                score[lab[i]] += o if g == "him" else -o if g == "her" else 0
    if any(abs(v) > 1e-6 for v in score.values()) and score[voices[0]] != score[voices[1]]:
        him = max(voices, key=lambda j: score[j])
        how = "lyric tags name the groups"
    else:
        him = min(voices, key=lambda j: sung[j])
        how = "no gender tags: the voice with less sung time is him"
    her = voices[0] if voices[1] == him else voices[1]
    if swap:
        him, her = her, him
        how += "; swapped by the user"
    runs = _runs(wins, lab, him, her)
    answers = _answers(fp, C, her, him, bleed, runs)
    L = {(round(a, 2), round(b, 2)): np.array(e, float) for (a, b), e in zip(fp["lines"], fp["line_emb"]) if e}
    lines = []
    for l in all_lines:
        sec = voice_seconds(runs, l["start"], l["end"])
        tot = sum(sec.values())
        if tot <= 0:
            v = "unknown"
        else:
            major = max(sec, key=sec.get)
            minor = "her" if major == "him" else "him"
            v = (f"{major}>{minor}" if _first_voice(runs, l) == major else f"{minor}>{major}") \
                if sec[minor] >= MIXED_MIN and sec[minor] >= MIXED_FRAC * tot else major
        e = L.get((round(l["start"], 2), round(l["end"], 2)))
        cos = {"her": float(e @ C[her]), "him": float(e @ C[him])} if e is not None else None
        # unsure = the measurement cannot say who sings it: no voiced windows, or the line's own
        # fingerprint sits between the two voices or far from both. A shared hand-over is measured,
        # not unsure (it shows her, user rule); B-roll covers an unsure line (user rule, 2026-11-07)
        unsure = v == "unknown" or (">" not in v and cos is not None and (
            abs(cos["him"] - cos["her"]) < UNSURE_MARGIN or max(cos.values()) < UNSURE_FLOOR))
        ans = [x for x in answers if x["end"] > l["start"] and x["start"] < l["end"] + FINE_PAD[1]]
        lines.append({"start": l["start"], "end": l["end"], "text": l["text"], "voice": v, "unsure": bool(unsure),
                      "answers": ans,
                      "cos": {k: round(x, 3) for k, x in cos.items()} if cos else None,
                      "seconds": {k: round(x, 2) for k, x in sec.items()}})
    return {"version": VERSION, "runs": runs, "answers": answers, "lines": lines, "how": how, "swap": bool(swap),
            "separation": round(float(C[voices[0]] @ C[voices[1]]), 3),
            "sung_seconds": {"her": round(sum(r["end"] - r["start"] for r in runs if r["voice"] == "her"), 1),
                             "him": round(sum(r["end"] - r["start"] for r in runs if r["voice"] == "him"), 1)}}


def _first_voice(runs, l):
    for r in runs:
        if r["end"] > l["start"] and r["start"] < l["end"]:
            return r["voice"]
    return None


def line_voice(v):
    """'her', 'him', 'her>him' -> the voice that holds most of the line."""
    if not v or v == "unknown":
        return None
    if ">" in v:
        return None
    return v


def apply(sections, tl):
    """Write each aligned line's measured voice into the section table in place (line['voice'])
    and the voice-change times inside sung sections (section['voice_cuts']), for the writer."""
    by = {(round(l["start"], 2), round(l["end"], 2)): l for l in tl["lines"]}
    for s in sections:
        for l in s.get("lines") or []:
            m = by.get((round(l["start"], 2), round(l["end"], 2)))
            l["voice"] = m["voice"] if m else "unknown"
            l["voice_seconds"] = m["seconds"] if m else {"her": 0.0, "him": 0.0}
            l["voice_unsure"] = bool(m["unsure"]) if m else True
            l["voice_answers"] = m.get("answers") or [] if m else []
        cuts = []
        for a, b in zip(tl["runs"], tl["runs"][1:]):
            # a change straight from one voice to the other (a change across a silence is
            # already covered by the next line's cut point)
            if a["voice"] == b["voice"] or b["start"] - a["end"] >= WIN:
                continue
            t = round((a["end"] + b["start"]) / 2, 2)
            near_line_cut = any(-0.6 < (l["start"] - 0.5) - t < 0.6 for l in s.get("lines") or [])
            # a hand-over inside a shared line is no cut: shared lines show her (user rule, 2026-11-07)
            in_shared = any(">" in str(l.get("voice")) and l["start"] < t < l["end"] for l in s.get("lines") or [])
            if in_shared:
                continue
            if s["start"] + 0.5 < t < s["end"] - 0.5 and s.get("vocal") and not near_line_cut:
                cuts.append(t)
        s["voice_cuts"] = cuts
        s["voice_answers"] = [x for x in tl.get("answers") or [] if x["end"] > s["start"] and x["start"] < s["end"]]
        s["voice_runs"] = [{"start": max(r["start"], s["start"]), "end": min(r["end"], s["end"]), "voice": r["voice"]}
                           for r in tl["runs"] if r["end"] > s["start"] and r["start"] < s["end"]]
    return sections


def summary_text(tl, sections):
    """One line per sung section: who sings it, line by line, for the tab and the log."""
    out = []
    for s in sections:
        if not s.get("lines"):
            continue
        vs = [l.get("voice", "?") for l in s["lines"]]
        out.append(f"{s['type']} {s['start']:.1f}-{s['end']:.1f}s: " + ", ".join(vs))
    return "\n".join(out)
