"""Validate a LoRA-fitness METRIC before we trust it to pick winners.

Context: the user does not trust CLAP zero-shot tag-presence as a judge (see
METAL_LORA_PLAN §13c / memory feedback_clap-scoring-unproven). It scores rank-only
against a crowded ~40-label vocab and, in our one stored run, surfaced the artist's
defining tags only on the GARBLED take. So before any automated number is allowed to
rank training runs, it has to EARN trust against ground truth.

The metric we actually want to validate here = CLAP CENTROID DISTANCE: cosine between a
take's CLAP audio embedding and the MEAN embedding of the artist's own corpus. That
measures "how close to this artist" directly, not via lossy text labels. The box now
exposes raw audio embeddings via analyze_server /embed (analyze_py.embed).

Three validity criteria a metric must pass (METAL_LORA_PLAN §13c):
  (1) MONOTONIC RANKING of known references:
      artist (held-out) > same-genre-other-artist > other-genre > noise.
  (2) TEST-RETEST stability: re-embedding the same file gives ~the same score.
  (3) AGREEMENT with the user's blind-A/B EAR verdicts on known pairs.
A metric that fails (1) is useless; one that passes (1)+(2) but fails (3) is a genre
guard, not a fidelity meter. Only a metric passing all three becomes an advisory
pre-filter (never the sole judge -- ears stay the judge, [[wait-for-feedback]]).

GPU: every embed() call runs CLAP on the shared 3090. Serialize vs the engine
[[no-concurrent-clap-engine]]. This whole module is read-only on audio; it does NOT
generate. But it DOES use the box GPU, so it is USER-initiated, not fired automatically.

All math is plain numpy. Embeddings come from the box; nothing heavy runs on the Mac.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import analyze_py

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac")
DEFAULT_EXPECTED_ORDER = ["artist", "same_genre", "other_genre", "noise"]


def list_audio(folder: str, limit: Optional[int] = None) -> List[str]:
    """Audio files directly under `folder` (sorted, deterministic)."""
    if not folder or not os.path.isdir(folder):
        return []
    out = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                 if f.lower().endswith(AUDIO_EXTS))
    return out[:limit] if limit else out


def _embed_many(host: str, paths: List[str], timeout: float = 600.0
                ) -> Tuple[np.ndarray, List[str]]:
    """Embed each file; return (matrix [n,d], paths_that_worked). Skips failures so one
    bad file can't abort a long validation run."""
    vecs, ok = [], []
    for p in paths:
        try:
            vecs.append(np.asarray(analyze_py.embed(host, p, timeout=timeout), dtype=np.float64))
            ok.append(p)
        except Exception as e:
            print(f"[metric_validate] embed failed for {os.path.basename(p)}: {e}")
    if not vecs:
        return np.zeros((0, 0)), []
    return np.vstack(vecs), ok


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def build_centroid(host: str, audio_paths: List[str], timeout: float = 600.0
                   ) -> Dict[str, Any]:
    """Mean (L2-normalized) CLAP embedding of an audio set = the artist's 'sound centroid'."""
    mat, ok = _embed_many(host, audio_paths, timeout)
    if mat.shape[0] == 0:
        raise RuntimeError("no embeddings produced for centroid")
    centroid = _unit(mat.mean(axis=0))
    return {"centroid": [float(x) for x in centroid], "n": len(ok),
            "files": [os.path.basename(p) for p in ok], "dim": int(centroid.shape[0])}


def _cos_to_centroid(mat: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Cosine of each row to the centroid (rows already unit-norm from /embed)."""
    if mat.shape[0] == 0:
        return np.zeros((0,))
    return mat @ _unit(centroid)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardized separation between two score groups (higher = cleaner separation)."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / (sp + 1e-9))


def _pair_auc(hi: np.ndarray, lo: np.ndarray) -> float:
    """Fraction of (hi, lo) pairs where hi scores above lo. 1.0 = perfect separation,
    0.5 = chance. This is the Mann-Whitney U / AUC, the cleanest 'can it rank these?' number."""
    if len(hi) == 0 or len(lo) == 0:
        return float("nan")
    wins = sum(1.0 if h > l else 0.5 if h == l else 0.0 for h in hi for l in lo)
    return wins / (len(hi) * len(lo))


def run_validation(
    *,
    analyze_host: str,
    artist_dir: str,
    bucket_dirs: Dict[str, str],
    library_dir: str,
    holdout_frac: float = 0.4,
    expected_order: Optional[List[str]] = None,
    ear_pairs: Optional[List[Dict[str, str]]] = None,
    per_bucket_limit: Optional[int] = None,
    timeout: float = 600.0,
) -> Dict[str, Any]:
    """Full validity test of the CLAP-centroid metric.

    artist_dir: folder of the artist's OWN tracks. Split into a centroid set and a
        held-out 'artist' scoring bucket (so artist similarity isn't trivially 1.0).
    bucket_dirs: {label: folder} for the other reference buckets, e.g.
        {"same_genre": ".../other_symphonic_metal", "other_genre": ".../pop",
         "noise": ".../noise"}. ('artist' is added automatically from the holdout.)
    holdout_frac: fraction of artist tracks held out for scoring vs building the centroid.
    expected_order: best->worst label ranking the metric SHOULD reproduce.
    ear_pairs: [{"a": path, "b": path, "winner": "a"|"b"}] known ear verdicts.

    Returns a report dict (also persisted to library/lora_train_history/metric_validation.json)
    with per-bucket score stats, the ordering check, pairwise AUC + Cohen's d vs artist,
    test-retest delta, and ear-agreement -- plus an overall verdict string.
    """
    expected_order = expected_order or DEFAULT_EXPECTED_ORDER
    artist_paths = list_audio(artist_dir)
    if len(artist_paths) < 3:
        raise RuntimeError(f"need >=3 artist tracks to split, found {len(artist_paths)} in {artist_dir}")

    # Deterministic holdout split (sorted order, every Nth held out) -- no RNG (repeatable).
    step = max(2, int(round(1.0 / max(0.05, holdout_frac))))
    holdout = [p for i, p in enumerate(artist_paths) if i % step == 0]
    centroid_set = [p for i, p in enumerate(artist_paths) if i % step != 0]
    if len(centroid_set) < 2 or len(holdout) < 2:   # tiny corpus fallback: still split something
        holdout, centroid_set = artist_paths[:1], artist_paths[1:]

    cen = build_centroid(analyze_host, centroid_set, timeout)
    centroid = np.asarray(cen["centroid"], dtype=np.float64)

    # Score every bucket (artist holdout + the supplied reference buckets).
    buckets: Dict[str, List[str]] = {"artist": holdout}
    for label, d in (bucket_dirs or {}).items():
        buckets[label] = list_audio(d, per_bucket_limit)

    scores: Dict[str, List[float]] = {}
    files: Dict[str, List[str]] = {}
    for label, paths in buckets.items():
        mat, ok = _embed_many(analyze_host, paths, timeout)
        s = _cos_to_centroid(mat, centroid)
        scores[label] = [float(x) for x in s]
        files[label] = [os.path.basename(p) for p in ok]

    stats = {label: {
        "n": len(s),
        "mean": float(np.mean(s)) if s else None,
        "std": float(np.std(s, ddof=1)) if len(s) > 1 else None,
        "min": float(np.min(s)) if s else None,
        "max": float(np.max(s)) if s else None,
    } for label, s in scores.items()}

    # (1) Ordering check: observed mean ranking vs expected.
    present = [l for l in expected_order if scores.get(l)]
    observed = sorted(present, key=lambda l: stats[l]["mean"], reverse=True)
    ordering_ok = observed == present

    # Separation of artist vs every other bucket: AUC + Cohen's d.
    art = np.asarray(scores.get("artist", []))
    separation = {}
    for label, s in scores.items():
        if label == "artist" or not s:
            continue
        separation[label] = {
            "auc_artist_over": round(_pair_auc(art, np.asarray(s)), 3),
            "cohens_d": round(_cohens_d(art, np.asarray(s)), 3),
        }

    # (2) Test-retest: re-embed up to 3 artist holdout files; delta should be ~0.
    retest = []
    for p in holdout[:3]:
        try:
            v1 = _unit(np.asarray(analyze_py.embed(analyze_host, p, timeout=timeout)))
            v2 = _unit(np.asarray(analyze_py.embed(analyze_host, p, timeout=timeout)))
            retest.append({"file": os.path.basename(p),
                           "self_cosine": round(float(v1 @ v2), 6)})
        except Exception as e:
            retest.append({"file": os.path.basename(p), "error": str(e)})
    retest_ok = all(r.get("self_cosine", 0) > 0.999 for r in retest if "self_cosine" in r) and bool(retest)

    # (3) Ear-verdict agreement.
    ear = {"pairs": [], "n": 0, "agree": 0, "agreement_rate": None}
    for pr in (ear_pairs or []):
        try:
            sa = float(_unit(np.asarray(analyze_py.embed(analyze_host, pr["a"], timeout=timeout))) @ _unit(centroid))
            sb = float(_unit(np.asarray(analyze_py.embed(analyze_host, pr["b"], timeout=timeout))) @ _unit(centroid))
            metric_winner = "a" if sa > sb else "b"
            agree = (metric_winner == pr.get("winner"))
            ear["pairs"].append({"a": os.path.basename(pr["a"]), "b": os.path.basename(pr["b"]),
                                 "ear_winner": pr.get("winner"), "metric_winner": metric_winner,
                                 "score_a": round(sa, 4), "score_b": round(sb, 4), "agree": agree})
            ear["n"] += 1
            ear["agree"] += int(agree)
        except Exception as e:
            ear["pairs"].append({"a": pr.get("a"), "b": pr.get("b"), "error": str(e)})
    if ear["n"]:
        ear["agreement_rate"] = round(ear["agree"] / ear["n"], 3)

    # Overall verdict.
    auc_other = separation.get("other_genre", {}).get("auc_artist_over")
    verdict_bits = []
    verdict_bits.append("ORDERING " + ("ok" if ordering_ok else f"FAIL ({observed} vs {present})"))
    verdict_bits.append("RETEST " + ("ok" if retest_ok else "FAIL"))
    if auc_other is not None:
        verdict_bits.append(f"AUC(artist>other_genre)={auc_other}")
    if ear["agreement_rate"] is not None:
        verdict_bits.append(f"ear_agreement={ear['agreement_rate']}")
    trustworthy = bool(ordering_ok and retest_ok and (auc_other is None or auc_other >= 0.8)
                       and (ear["agreement_rate"] is None or ear["agreement_rate"] >= 0.7))

    report = {
        "metric": "clap_centroid_cosine",
        "created_at": time.time(),
        "analyze_host": analyze_host,
        "artist_dir": artist_dir,
        "centroid": {"n": cen["n"], "files": cen["files"]},
        "holdout_files": [os.path.basename(p) for p in holdout],
        "expected_order": present,
        "observed_order": observed,
        "ordering_ok": ordering_ok,
        "stats": stats,
        "scores": scores,
        "files": files,
        "separation_vs_artist": separation,
        "test_retest": retest,
        "retest_ok": retest_ok,
        "ear": ear,
        "verdict": "; ".join(verdict_bits),
        "trustworthy_as_prefilter": trustworthy,
        "note": ("Passing => usable as an advisory PRE-FILTER only; ears remain the judge. "
                 "Failing ordering => metric is useless for artist fidelity."),
    }

    out_dir = os.path.join(library_dir, "lora_train_history")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "metric_validation.json")
    try:
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        report["saved_to"] = out_path
    except Exception as e:
        print(f"[metric_validate] persist warning: {e}")
    return report
