"""AI Mix adaptive rules (docs/AI_MIX_PLAN.md): every move is measured, compared with a style
preset's range, and sized to close only the gap. Nothing here is a fixed per-song setting; the
constants are LIMITS (max cut, max boost) and loop tolerances.

Inputs: an ingested session, its aimix.analyse() result (Suno-balance gains, per-stem stats,
masking) and a preset from aimix_refs (distributions per separator role). The Suno stems are
grouped onto the separator roles (aimix_features.SUNO_GROUP) and measured over the same middle
excerpt length as the references.

Range rule everywhere: target = clip(current, p25, p75) of the preset. Inside the band = no move.
"""
import numpy as np

from . import aimix
from . import aimix_features as F

EXCERPT_S = 120.0
MAX_LEVEL_DB = 6.0
MAX_EQ_DB = 3.0
MAX_BANDS = 4
EQ_MIN_DB = 1.0            # ignore tonal gaps smaller than this after smoothing
MAX_PUNCH_DB = 12.0
MAX_DEESS_DB = 8.0
WIDTH_MIN_GAP_DB = 2.0     # separated stems measure narrower than generated ones (calibration), so
                           # small width gaps are inside the measurement bias and are left alone


def _clip_target(v, st):
    if v is None or st is None:
        return v
    return float(min(max(v, st["p25"]), st["p75"]))


def _excerpt(x, sr):
    n = x.shape[0]
    h = int(EXCERPT_S * sr / 2)
    if n <= 2 * h:
        return x
    return x[n // 2 - h:n // 2 + h]


def suno_groups(session, analysis, gains_db=None):
    """Role -> summed excerpt (frames, 2) at the given per-stem gains (default Suno balance),
    plus role -> member stem names."""
    gains_db = gains_db or analysis["gains_db"]
    groups, members, sr = {}, {}, None
    for s in session["stems"]:
        n = s["name"]
        if n in analysis["silent"]:
            continue
        x, sr = aimix._read(s["path"])
        x = _excerpt(x, sr) * (10 ** (gains_db[n] / 20.0))
        r = F.SUNO_GROUP.get(s["role"] or "other", "other")
        groups[r] = groups[r] + x[:groups[r].shape[0]] if r in groups else x
        members.setdefault(r, []).append(n)
    return groups, members, sr


def mix_features(path):
    """Whole-mix shape features of a full mix (middle excerpt), comparable to preset['stats']['mix']."""
    x, sr = aimix._read(path)
    x = _excerpt(x, sr)
    out = {"ltas": F.ltas(x.mean(1), sr)}
    out.update(F.dynamics(x.mean(1), sr))
    out.update(F.stereo(x, sr))
    return out


def pick_preset(mix_feat, presets):
    """Nearest preset by whole-mix shape: LTAS gap normalised by each preset's IQR, plus crest and
    width. Returns (name, scores)."""
    scores = {}
    for name, p in presets.items():
        m = p["stats"]["mix"]
        lt = m["ltas"]
        iqr = np.maximum(np.array(lt["p75"]) - np.array(lt["p25"]), 0.5)
        d = np.mean(np.abs((np.array(mix_feat["ltas"]) - np.array(lt["med"])) / iqr))
        for k in ("crest_db", "width_db", "range_db"):
            st = m.get(k)
            if st:
                d += abs(mix_feat[k] - st["med"]) / max(st["p75"] - st["p25"], 0.5) / 3.0
        scores[name] = float(d)
    return min(scores, key=scores.get), scores


# ------------------------------------------------------------------ levels

def solve_levels(feat, preset):
    """Per-role gain offsets so each role's share of the mix (rel_lufs) lands inside the preset
    band, solved jointly in power (moving one role moves every share). Vocals use presence_db
    (vocal over the rest in 1-5 kHz) instead of rel_lufs when both are available."""
    roles = [r for r in feat["roles"] if F.present(feat["roles"][r])]
    rel = {r: feat["roles"][r]["rel_lufs"] for r in roles}
    lufs = {r: rel[r] for r in roles}                      # relative to the mix: fine for shares
    g = {r: 0.0 for r in roles}
    pst = preset["stats"]["roles"]
    voc_pres = feat["roles"].get("vocals", {}).get("presence_db")
    for _ in range(8):
        tot = 10 * np.log10(sum(10 ** ((lufs[r] + g[r]) / 10) for r in roles))
        for r in roles:
            cur = lufs[r] + g[r] - tot
            if r == "vocals" and voc_pres is not None and pst.get("vocals", {}).get("presence_db"):
                # presence moves ~dB for dB with vocal gain; the rest moves a little the other way.
                # The lead is the priority element: raised toward the band, never cut below Suno's
                # own placement (user, 2026-10-12: a style-driven vocal cut buried the verses).
                cur_p = voc_pres + g[r] - (tot - 10 * np.log10(sum(10 ** ((lufs[q]) / 10) for q in roles)))
                want = max(cur_p, pst["vocals"]["presence_db"]["p25"])
                g[r] = max(0.0, g[r] + 0.5 * (want - cur_p))
            elif pst.get(r, {}).get("rel_lufs"):
                want = _clip_target(cur, pst[r]["rel_lufs"])
                g[r] += 0.5 * (want - cur)
            g[r] = float(np.clip(g[r], -MAX_LEVEL_DB, MAX_LEVEL_DB))
    return g


def _vocal_section_guard(session, analysis, members, gains, pst, notes):
    """Whole-song check: vocal presence (vocal over the rest, 1-5 kHz) per 10 s vocal window at
    the solved levels. If the weakest sections (20th percentile) fall below the style band's
    p25, lift the vocal by the deficit (max +4 dB). Catches sparse intros/verses that a
    song-wide number misses. (A per-section ride was tried on Dream Of Me and made the verse
    weaker: it spent its lift on breath/bleed windows; uniform guard matched the user's preferred
    verse level within 0.1 dB [M].)"""
    st = pst.get("vocals", {}).get("presence_db")
    if not st or "vocals" not in members:
        return 0.0
    voc = rest = None
    sr = None
    by_name = {s["name"]: s for s in session["stems"]}
    for role, names in members.items():
        for n in names:
            x, sr = aimix._read(by_name[n]["path"])
            x = x.mean(1) * (10 ** ((analysis["gains_db"][n] + gains.get(role, 0.0)) / 20.0))
            if role == "vocals":
                voc = x if voc is None else voc[:len(x)] + x[:len(voc)]
            else:
                rest = x if rest is None else rest[:len(x)] + x[:len(rest)]
    if voc is None or rest is None:
        return 0.0
    n = min(len(voc), len(rest))
    w = F.presence_windows(voc[:n], rest[:n], sr)
    if len(w) < 3:
        return 0.0
    weak = float(np.percentile(w, 20))
    if weak >= st["p25"]:
        notes.append(f"vocals: weakest sections at {weak:.1f} dB presence, inside style (>= {st['p25']:.1f}): no lift")
        return 0.0
    lift = float(min(4.0, st["p25"] - weak))
    notes.append(f"vocals: +{lift:.1f} dB section guard (weakest sections {weak:.1f} dB presence < style {st['p25']:.1f})")
    return lift


# ------------------------------------------------------------------ tonal EQ

def eq_moves(ltas, st):
    """Third-octave gap to the preset band -> up to MAX_BANDS peaking moves (+ shelves at the ends).
    Gap is smoothed over 3 bands so single-band noise doesn't become a notch."""
    x = np.array(ltas)
    lo, hi = np.array(st["p25"]), np.array(st["p75"])
    gap = np.clip(x, lo, hi) - x
    gap = np.convolve(np.pad(gap, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    moves, used = [], np.zeros(len(gap), bool)
    for _ in range(MAX_BANDS):
        k = int(np.argmax(np.abs(gap) * ~used))
        if used[k] or abs(gap[k]) < EQ_MIN_DB:
            break
        db = float(np.clip(gap[k], -MAX_EQ_DB, MAX_EQ_DB))
        hz = F.THIRD_OCT[k]
        if k <= 2:
            moves.append({"type": "lowshelf", "hz": 63.0, "db": round(db, 1)})
        elif k >= len(gap) - 2:
            moves.append({"type": "highshelf", "hz": 12500.0, "db": round(db, 1)})
        else:
            moves.append({"type": "bell", "hz": float(hz), "q": 1.4, "db": round(db, 1)})
        used[max(0, k - 2):k + 3] = True
    return moves


# ------------------------------------------------------------------ closed loops

def _search(fn, lo, hi, target, tol=0.3, steps=7):
    """Bisection on a monotone-increasing fn(param) -> measured value, toward target."""
    a, b = lo, hi
    fa = fn(a)
    if fa >= target:
        return a, fa
    fb = fn(b)
    if fb <= target:
        return b, fb
    for _ in range(steps):
        m = (a + b) / 2
        fm = fn(m)
        if abs(fm - target) <= tol:
            return m, fm
        if fm < target:
            a = m
        else:
            b = m
    return (a + b) / 2, fn((a + b) / 2)


def deess(x, sr, range_db, lo=5000.0, hi=9000.0):
    """Split-band de-esser on x (ch, frames): where the 5-9 kHz share of a 10 ms frame rises above
    its median, that band is turned down by up to range_db (zero-phase band split, so the rest of
    the signal is untouched)."""
    if range_db <= 0:
        return x
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, [lo, hi], "bandpass", fs=sr, output="sos")
    band = sosfiltfilt(sos, x, axis=1)
    b = int(sr / 100)
    n = x.shape[1] // b
    pb_ = (band[:, :n * b] ** 2).mean(0).reshape(n, b).mean(1)
    pt = (x[:, :n * b] ** 2).mean(0).reshape(n, b).mean(1) + 1e-20
    share = 10 * np.log10(pb_ / pt + 1e-20)
    over = np.clip(share - np.median(share) - 3.0, 0, None)       # 3 dB above typical = a burst
    red = -np.clip(over, 0, range_db)
    t = (np.arange(n) + 0.5) * b
    g = 10 ** (np.interp(np.arange(x.shape[1]), t, red) / 20.0)
    return (x - band * (1 - g)).astype(np.float32)


VOCAL_COMP = {"ratio": 3.0, "attack_ms": 8, "release_ms": 120}
MAX_VOCAL_GR_DB = 10.0


def _vocal_comp_loop(session, analysis, members, groups, gains, pst, stems_doc, stem_by_name, sr, notes):
    """Closed loop: compress the vocal stems (same gain reduction on lead and backing), restore
    their loudness (as the renderer's level landing does), and re-measure presence_spread_db
    (median minus p10 of vocal-over-rest, 1-5 kHz) against the rest of the mix at the solved
    levels. Search the gain reduction until the spread is inside the style band. Off when the
    preset lacks the measure or the vocal is already even enough."""
    import pyloudnorm as pyln
    st = pst.get("vocals", {}).get("presence_spread_db")
    names = members.get("vocals", [])
    if not st or not names:
        return
    meter = pyln.Meter(sr)
    rest = None
    for role, x in groups.items():
        if role != "vocals":
            y = x * (10 ** (gains.get(role, 0.0) / 20.0))
            rest = y if rest is None else rest[:len(y)] + y[:len(rest)]
    if rest is None:
        return
    raw = {}
    for n in names:
        x, _ = aimix._read(stem_by_name[n]["path"])
        raw[n] = _excerpt(x, sr).T.copy()
    gv = gains.get("vocals", 0.0)

    def spread(gr):
        voc = None
        for n, x in raw.items():
            y = x
            if gr > 0:
                thr = analysis["stats"][n]["p95_frame_db"] - gr / (1.0 - 1.0 / VOCAL_COMP["ratio"])
                y = aimix._board([{"fx": "compressor", "threshold_db": thr, **VOCAL_COMP}])(x, sr)
                a, b = meter.integrated_loudness(x.T), meter.integrated_loudness(y.T)
                if np.isfinite(a) and np.isfinite(b):
                    y = y * (10 ** ((a - b) / 20.0))           # level landing
            y = y.T * (10 ** ((analysis["gains_db"][n] + gv) / 20.0))
            voc = y if voc is None else voc[:len(y)] + y[:len(voc)]
        m = min(len(voc), len(rest))
        return F.presence_spread(voc[:m].mean(1), rest[:m].mean(1), sr)

    base = spread(0.0)
    if base is None:
        return
    if base <= st["p75"]:
        notes.append(f"vocals: spread {base:.1f} dB already inside style (<= {st['p75']:.1f}): no vocal compression")
        return
    gr, got = _search(lambda g: -(spread(g) or 0.0), 0.0, MAX_VOCAL_GR_DB, -st["p75"])
    if gr < 0.5:
        return
    for n in names:
        thr = analysis["stats"][n]["p95_frame_db"] - gr / (1.0 - 1.0 / VOCAL_COMP["ratio"])
        stems_doc[n]["chain"].append({"fx": "compressor", "threshold_db": round(thr, 1), **VOCAL_COMP,
                                      "why": f"vocal spread {base:.1f} -> {-got:.1f} dB, style <= {st['p75']:.1f}"})
    notes.append(f"vocals: compress ~{gr:.1f} dB (weak moments drop {base:.1f} dB under typical vs the band; "
                 f"-> {-got:.1f}, style <= {st['p75']:.1f})")


def _taste_vocals(doc, analysis, taste, notes):
    """Closed loops toward the user's starred versions, on the vocal group only: backing level
    (lead over backing), vocal compression (within-phrase range of the dry vocal), vocal reverb
    send (dip depth of vocal + its reverb). Each loop measures the actual rendered vocal."""
    import copy
    from . import aimix_taste as T
    tv, k = taste["vocals"], len(taste.get("picks", []))
    who = f"your taste ({k} starred version{'s' if k != 1 else ''})"
    voc = {n: s for n, s in doc["stems"].items() if s.get("group") == "vocals"}
    lead = [n for n, s in voc.items() if s["role"] == "vocals"]
    back = [n for n, s in voc.items() if s["role"] == "backing"]

    # 1. lead over backing
    want = tv.get("lead_over_backing_db")
    if want is not None and lead and back:
        lv, bv, _, _, _ = T.render_vocal(doc, with_verb=False, vocals_only=True)
        now = float(10 * np.log10(np.mean(lv ** 2) / max(np.mean(bv ** 2), 1e-20)))
        delta = now - want
        if abs(delta) >= 0.5:
            for n in back:
                doc["stems"][n]["target_lufs"] = round(doc["stems"][n]["target_lufs"] + delta, 2)
            notes.append(f"vocals: backing {delta:+.1f} dB (lead over backing {now:.1f} -> {want:.1f} dB, {who})")

    # 2. compression toward the taste's within-phrase range (dry vocal)
    want = tv.get("vocal_phrase_range_db")
    if want is not None:
        def pr(gr, d=doc):
            dd = copy.deepcopy(d)
            if gr > 0:
                for n in voc:
                    thr = analysis["stats"][n]["p95_frame_db"] - gr / (1.0 - 1.0 / VOCAL_COMP["ratio"])
                    dd["stems"][n]["chain"] = dd["stems"][n]["chain"] + [{"fx": "compressor", "threshold_db": thr, **VOCAL_COMP}]
            _, _, v, _, sr = T.render_vocal(dd, with_verb=False, vocals_only=True)
            return F.dynamics(v.mean(1), sr)["phrase_range_db"] or 0.0
        base = pr(0.0)
        if base > want + 0.3:
            gr, got = _search(lambda g: -pr(g), 0.0, 12.0, -want)
            if gr >= 0.5:
                for n in voc:
                    thr = analysis["stats"][n]["p95_frame_db"] - gr / (1.0 - 1.0 / VOCAL_COMP["ratio"])
                    doc["stems"][n]["chain"].append({"fx": "compressor", "threshold_db": round(thr, 1), **VOCAL_COMP,
                                                     "why": f"within-phrase range {base:.1f} -> {-got:.1f} dB, {who} {want:.1f}"})
                notes.append(f"vocals: compress ~{gr:.1f} dB (within-phrase range {base:.1f} -> {-got:.1f} dB, {who} {want:.1f})")
        else:
            notes.append(f"vocals: already as dense as {who} ({base:.1f} dB within-phrase range): no compression")

    # 3. reverb send toward the taste's wetness (vocal + its reverb return)
    want = tv.get("vocal_dip_db")
    if want is not None:
        def dip(send):
            dd = copy.deepcopy(doc)
            for n in voc:
                if send <= -40:
                    dd["stems"][n]["sends"].pop("verb", None)
                else:
                    dd["stems"][n]["sends"]["verb"] = send
            _, _, v, _, sr = T.render_vocal(dd, with_verb=True, vocals_only=True)
            return F.wetness(v.mean(1), sr)["dip_db"]
        base = dip(-41)
        if base > want + 0.3:
            send, got = _search(lambda s: -dip(s), -40.0, -12.0, -want, tol=0.3, steps=6)
            if send > -40:
                for n in voc:
                    doc["stems"][n]["sends"]["verb"] = round(send)
                notes.append(f"vocals: reverb send {round(send)} dB (gaps {base:.1f} -> {-got:.1f} dB, {who} {want:.1f})")
        else:
            notes.append(f"vocals: already as wet as {who} (gaps {base:.1f} dB): no reverb")


# ------------------------------------------------------------------ the mix doc

def build(session, analysis, preset, calib=None, taste=None):
    """Adaptive mix doc. calib: optional role -> feature -> offset (separated minus generated),
    subtracted from preset bands so the comparison happens in the Suno-stem domain."""
    groups, members, sr = suno_groups(session, analysis)
    feat = F.song_features(groups, sr)
    pst = {r: dict(v) for r, v in preset["stats"]["roles"].items()}
    if calib:
        for r, offs in calib.items():
            for k, off in offs.items():
                st = pst.get(r, {}).get(k)
                if st and off is not None:
                    pst[r][k] = {**st, "med": st["med"] - off, "p25": st["p25"] - off, "p75": st["p75"] - off}
    preset_c = {**preset, "stats": {**preset["stats"], "roles": pst}}

    gains = solve_levels(feat, preset_c)
    stems_doc, notes = {}, []
    gains["vocals"] = gains.get("vocals", 0.0) + _vocal_section_guard(session, analysis, members, gains, pst, notes)
    stem_by_name = {s["name"]: s for s in session["stems"]}
    for role, names in members.items():
        rf, rs = feat["roles"].get(role), pst.get(role)
        if rf is None or rs is None or not F.present(rf):
            # nothing to compare against: pass the stems through at Suno balance, HPF only
            notes.append(f"{role}: no comparison ({'absent from this song' if rf is None or not F.present(rf) else 'not in preset'}), left at Suno balance")
            rf, rs = None, {}
        g = gains.get(role, 0.0) if rf else 0.0
        if rf and abs(g) >= 0.5:
            notes.append(f"{role}: level {g:+.1f} dB (share of mix {rf['rel_lufs']:.1f} dB vs style "
                         f"{rs['rel_lufs']['p25']:.1f}..{rs['rel_lufs']['p75']:.1f})")
        eq = eq_moves(rf["ltas"], rs["ltas"]) if rf and rs.get("ltas") else []
        if role == "vocals" and eq:
            # no air/presence cuts on vocals unless the vocal is measured sibilant beyond the style
            # (a top-band cut there boxed the vocal in, user 2026-10-12); brightness is fixed on
            # the instruments instead
            sib_st = rs.get("sib_db")
            sibilant = bool(sib_st and rf.get("sib_db") is not None and rf["sib_db"] > sib_st["p75"])
            eq = [m for m in eq if not (m["hz"] >= 5000 and m["db"] < 0 and not sibilant)]
        if role == "other":                                       # catch-all: half-strength
            eq = [{**m, "db": round(m["db"] / 2, 1)} for m in eq if abs(m["db"] / 2) >= EQ_MIN_DB]
        if eq:
            notes.append(f"{role}: tonal EQ " + ", ".join(f"{m['db']:+.1f} dB @ {m['hz']:.0f}" for m in eq))
        # compression: only if the loudness swings exceed the style band
        # compression keys off WITHIN-PHRASE range (how unevenly it is played), never the whole-song
        # range (an arrangement build would get flattened and lift sparse intros over the vocal).
        # Presets measured before phrase_range_db existed get no range compression.
        comp = None
        rng = rs.get("phrase_range_db") if role != "vocals" else None   # vocals: spread loop below
        if rf and rng and rf.get("phrase_range_db") is not None and rf["phrase_range_db"] > rng["p75"]:
            excess = rf["phrase_range_db"] - rng["p75"]
            comp = {"ratio": 3.0, "gr_db": float(min(excess, 8.0))}
            notes.append(f"{role}: compress ~{comp['gr_db']:.1f} dB (within-phrase range {rf['phrase_range_db']:.1f} dB "
                         f"> style {rng['p75']:.1f})")
        # width
        wst, width = rs.get("width_db"), 1.0
        if rf and wst:
            want = _clip_target(rf["width_db"], wst)
            if abs(want - rf["width_db"]) >= WIDTH_MIN_GAP_DB:
                width = float(np.clip(10 ** ((want - rf["width_db"]) / 20.0), 0.5, 1.6))
                notes.append(f"{role}: width x{width:.2f} (side {rf['width_db']:.1f} dB vs style "
                             f"{wst['p25']:.1f}..{wst['p75']:.1f})")
        lw, low_mono = rs.get("low_width_db"), 0
        if rf and lw and rf["low_width_db"] > lw["p75"] + 1.0:
            low_mono = 150
            notes.append(f"{role}: mono below 150 Hz (low side {rf['low_width_db']:.1f} dB > style {lw['p75']:.1f})")
        # wetness: send only when drier than the style band (bigger dip = drier)
        dst, send = rs.get("dip_db"), None
        if rf and dst and rf["dip_db"] > dst["p75"]:
            gap = rf["dip_db"] - dst["p75"]
            send = float(np.clip(-28 + 2.0 * gap, -28, -16))
            notes.append(f"{role}: reverb send {send:.0f} dB (dips {rf['dip_db']:.1f} dB, drier than style "
                         f"{dst['p75']:.1f})")

        for n in names:
            s = stem_by_name[n]
            st = analysis["stats"][n]
            chain = []
            hpf = min(aimix.ROLE_RULES.get(s["role"] or "other", aimix.ROLE_RULES["other"])["hpf"],
                      max(20.0, st["energy_2pct_hz"] * 0.7))
            chain.append({"fx": "highpass", "hz": round(hpf)})
            if eq:
                chain.append({"fx": "eq", "moves": eq, "why": "tonal gap to style"})
            cuts = []
            for k, m in (analysis.get("masking", {}).get(n) or {}).items():
                if m > aimix.MASK_START:
                    cuts.append({"hz": round(aimix._band_center(int(k))), "q": 1.0,
                                 "db": -min(aimix.MASK_MAX_CUT_DB, round(8.0 * (m - aimix.MASK_START), 1))})
            if cuts:
                chain.append({"fx": "peq", "bands": cuts, "why": "unmask lead vocal"})
            if comp:
                thr = st["p95_frame_db"] - comp["gr_db"] / (1.0 - 1.0 / comp["ratio"])
                chain.append({"fx": "compressor", "threshold_db": round(thr, 1), "ratio": comp["ratio"],
                              "attack_ms": 15, "release_ms": 150})
            stems_doc[n] = {
                "file": s["path"], "role": s["role"] or "other", "group": role, "mute": False,
                "target_lufs": round(st["lufs"] + analysis["gains_db"][n] + g, 2),
                "suno_lufs": round(st["lufs"] + analysis["gains_db"][n], 2),
                "trim_db": 0.0, "width": round(width, 2), "low_mono_hz": low_mono,
                "chain": chain, "sends": ({"verb": round(send)} if send is not None else {}),
            }

    # closed loops on the stems where the measure lives
    dr = [n for n in members.get("drums", []) if stem_by_name[n]["role"] == "drums"]
    pst_d = pst.get("drums", {}).get("punch_db")
    if dr and pst_d and feat["roles"].get("drums", {}).get("punch_db") is not None:
        n = dr[0]
        x, _ = aimix._read(stem_by_name[n]["path"])
        x = _excerpt(x, sr).T.copy()
        base = F.punch(x.mean(0), sr)["punch_db"]
        if base is not None and base < pst_d["p25"]:
            amt, got = _search(lambda a: F.punch(aimix._transient(x, sr, a).mean(0), sr)["punch_db"] or 0,
                               0.0, MAX_PUNCH_DB, pst_d["p25"])
            if amt >= 0.5:
                stems_doc[n]["chain"].insert(-1 if stems_doc[n]["chain"][-1]["fx"] == "compressor" else len(stems_doc[n]["chain"]),
                                             {"fx": "transient", "attack_db": round(amt, 1), "sustain_db": 0.0})
                notes.append(f"drums: punch +{amt:.1f} dB attack (measured {base:.1f} dB -> {got:.1f}, style "
                             f"{pst_d['p25']:.1f}..{pst_d['p75']:.1f})")
        else:
            notes.append(f"drums: punch already in style range ({base:.1f} dB, style {pst_d['p25']:.1f}..{pst_d['p75']:.1f}): no shaper")
    if not (taste and (taste.get("vocals") or {}).get("vocal_phrase_range_db") is not None):
        _vocal_comp_loop(session, analysis, members, groups, gains, pst, stems_doc, stem_by_name, sr, notes)
    pst_s = pst.get("vocals", {}).get("sib_db")
    if pst_s:
        for n in members.get("vocals", []):
            x, _ = aimix._read(stem_by_name[n]["path"])
            x = _excerpt(x, sr).T.copy()
            base = F.sibilance(x.mean(0), sr)["sib_db"]
            if base > pst_s["p75"]:
                amt, got = _search(lambda a: -F.sibilance(deess(x, sr, a).mean(0), sr)["sib_db"],
                                   0.0, MAX_DEESS_DB, -pst_s["p75"])
                if amt >= 0.5:
                    stems_doc[n]["chain"].append({"fx": "deess", "range_db": round(amt, 1)})
                    notes.append(f"{n}: de-ess up to {amt:.1f} dB (bursts {base:.1f} dB -> {-got:.1f}, style <= {pst_s['p75']:.1f})")

    mix_lufs = (analysis.get("mix") or {}).get("lufs")
    doc = {
        "version": 2, "title": session["title"], "mode": "remix", "source": "suno_stems",
        "preset": preset["name"], "preset_label": preset.get("label"), "sample_rate": analysis["sr"],
        "calibrated": bool(calib), "notes": notes, "measured": feat,
        "stems": stems_doc,
        "buses": {
            "verb": {"fx": "reverb", "room_size": 0.35, "damping": 0.5, "predelay_ms": 30, "hpf": 200, "lpf": 8000},
            "delay": {"fx": "delay", "ms": round(60000.0 / (analysis.get("bpm") or 120.0), 1), "feedback": 0.25,
                      "hpf": 300, "lpf": 6000},
        },
        "master": {"tone": "flat", "glue": False, "target_lufs": mix_lufs, "ceiling_db": -1.0},
        "provenance": [{"layer": "adaptive", "preset": preset["name"]}],
    }
    if taste and taste.get("vocals"):
        _taste_vocals(doc, analysis, taste, notes)
        doc["provenance"].append({"layer": "taste", "picks": len(taste.get("picks", []))})
    return doc
