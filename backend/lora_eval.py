"""Plan 1 — post-training LoRA evaluation (perceptual fitness via CLAP).

Why this exists: val_loss is misleading at our scale (verified 2026-05-30 — the
35-track best had LOWER val than the 6-track best yet sounded WORSE). We need a
perceptual signal that actually correlates with what the user hears. See
METAL_LORA_PLAN §11 for the full spec + §13a.6 for the val-loss finding.

What this does: for each adapter checkpoint supplied, loads it on the engine,
fires N generations with fixed prompts × seeds, posts each generated audio to
the box analyze service (CLAP zero-shot, port 5075), counts how often our
target style tags land in the top-K. The resulting *fitness curve* tells us
which checkpoint produces the most on-target output, independent of val_loss.

VRAM safety: this is POST-training only. Engine must be in inference-mode
(LM + DiT loaded, training state cleared). Caller is responsible for the
engine restart per [[engine-fresh-boot-for-lora]] before invoking the loop.

Architecture:
- evaluate_adapter() — single (ckpt, scale) → generate + score
- evaluate_dataset() — loop over a list of (ckpt, scale) → persist a fitness
  curve to library/lora_train_history/<dataset>_fitness.json
- Default target_tags = power-metal/symphonic-metal vocabulary; caller can
  override for any subgenre.

v2 changes (2026-05-30 afternoon, after partial-12-take smoke):
- **Position-weighted scoring**: rank-1 hit counts much more than rank-10 hit.
  Linear decay: weight = (K - rank) / K. Binary presence still reported for
  compatibility, but `fitness_weighted` is the recommended signal — finer
  granularity (0.0-N continuous) and rewards confident matches.
- **Retry logic for analyze service**: smoke run had 6/12 takes time out when
  autolabel hammered the LM on the shared 3090. Now wraps each analyze call
  with 3 retries + exponential backoff (15s, 30s, 60s) to ride out GPU
  contention. Per-track timeout dropped to 90s so retries kick in faster.

Limitations (still v2):
- Caller supplies the list of checkpoint paths; no automatic enumeration of
  train/checkpoints/epoch_N/ folders yet. Future patch via a box-side
  directory-listing endpoint.
- CLAP centroid distance (METAL_LORA_PLAN §11.3.4) not implemented yet —
  analyze service doesn't expose raw embeddings. Future patch.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
from typing import Any, Dict, List, Optional

import requests


# ─── Target / negative vocabularies ──────────────────────────────────────────
# IMPORTANT: target_tags are EXPERIMENT-SPECIFIC. What "good output" means for
# a Nightwish-trained LoRA (symphonic, soprano, orchestral) is different from
# what it means for a Manowar-trained LoRA (traditional heavy metal, male
# vocals, gallop riffs) which is different from a DragonForce LoRA (speed
# metal, virtuosic solos, double bass), etc. The caller of evaluate_adapter()
# / evaluate_dataset() MUST pass the target_tags appropriate to what THIS run
# was trained to produce — otherwise the fitness score is meaningless.
#
# The constants below are SUGGESTED STARTING POINTS for common subgenres.
# Use derive_target_tags_from_dataset() to auto-build a starting list from
# the training data's own seed captions, then hand-tune.
DEFAULT_TARGET_TAGS_POWER_METAL = [
    "power metal", "symphonic metal", "heavy metal", "anthemic",
    "harmonized lead guitars", "double bass drums", "soaring vocals",
    "epic chorus",
]
DEFAULT_TARGET_TAGS_SYMPHONIC_METAL = [
    "symphonic metal", "operatic vocals", "female vocals", "soprano",
    "orchestral", "choir", "cinematic", "epic", "dramatic",
]

# Negative tags — if these land high in top-K it's a red flag (model drifted
# off-genre). These are broad enough to apply across most metal experiments
# but the caller can override per-experiment too (e.g. for an electronic-
# metal hybrid LoRA you'd remove "electronic dance music" from the negatives).
DEFAULT_NEGATIVE_TAGS = ["pop music", "hip hop", "electronic dance music",
                          "country music", "jazz"]


def derive_target_tags_from_dataset(engine_host: str,
                                     dataset_path: str,
                                     min_occurrences: int = 2,
                                     max_tags: int = 16) -> List[str]:
    """Auto-build a starting target vocab from a dataset's seed captions.

    Each training track has a caption that begins with comma-separated style
    tags (from the MB / Last.fm / CLAP seed layer) before any prose. These
    tags are literally "what the LoRA was trained to produce." Aggregating
    them across the dataset gives a sensible default target vocabulary that
    actually matches THIS experiment.

    Args:
        engine_host: e.g. "192.168.1.201:8001"
        dataset_path: absolute path to dataset.json ON THE BOX
        min_occurrences: tag must appear in ≥N tracks to count (filters noise)
        max_tags: cap on returned list

    Returns: sorted list of tag strings, most-common first.

    Tip: call this once at experiment-start, eyeball the result, drop tags
    that look wrong, then pass the curated list to evaluate_dataset(). Don't
    use the raw output blindly — CLAP seeds can include wrong-genre tags
    (we saw "black metal, groove metal, thrash metal" on Nightwish tracks).
    """
    from collections import Counter
    base = engine_host if engine_host.startswith("http") else f"http://{engine_host}"
    # Make sure the right dataset is loaded
    requests.post(f"{base}/v1/dataset/load",
                  json={"dataset_path": dataset_path}, timeout=120)
    r = requests.get(f"{base}/v1/dataset/samples", timeout=120)
    r.raise_for_status()
    d = r.json()
    samples = d.get("data", d).get("samples", []) if isinstance(d, dict) else []
    counter: Counter[str] = Counter()
    for s in samples:
        cap = (s.get("caption") or "").strip()
        # First sentence / first 200 chars is where the tag list lives
        head = cap.split(".", 1)[0][:240].lower()
        for raw in head.split(","):
            tag = raw.strip()
            # Filter trivial / single-word noise tags
            if 3 <= len(tag) <= 40 and not tag.isdigit():
                counter[tag] += 1
    return [t for t, n in counter.most_common(max_tags) if n >= min_occurrences]


def _post(host: str, path: str, body: Dict[str, Any], timeout: float = 60.0) -> Dict[str, Any]:
    r = requests.post(host.rstrip("/") + path, json=body, timeout=timeout)
    r.raise_for_status()
    return r.json() if r.headers.get("content-type", "").startswith("application/json") else {}


def _wait_for_job(mac_base: str, job_id: str, poll_s: float = 5.0,
                   timeout_s: float = 600.0) -> Dict[str, Any]:
    """Poll Mac /api/job/<id> until status terminal. Returns the final job dict."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = requests.get(f"{mac_base.rstrip('/')}/api/job/{job_id}", timeout=20)
        r.raise_for_status()
        d = r.json()
        if d.get("status") in ("done", "failed", "error"):
            return d
        time.sleep(poll_s)
    raise TimeoutError(f"job {job_id} did not finish within {timeout_s}s")


def _position_weighted_score(tags_list: List[str], top_k_tags: List[str],
                              k: int) -> Dict[str, float]:
    """Linear-decay position weight: rank-1 = 1.0, rank-K = ~0.0, miss = 0.0.

    Why: the v1 smoke (2026-05-30) showed that broad-genre CLAP tags ("pop
    music", "country music") often land at high ranks for our generations,
    even when the audio is clearly metal. Treating presence-in-top-K as
    binary inflates these false-positives. Position weighting preserves the
    information while letting rank-1 hits dominate the score.

    Returns {tag: weight} where weight in [0.0, 1.0]. Mean weight = aggregate
    fitness for the tag list.
    """
    out: Dict[str, float] = {}
    top_lower = [t.lower().strip() for t in top_k_tags]
    for tag in tags_list:
        try:
            rank = top_lower.index(tag.lower().strip())
            out[tag] = max(0.0, (k - rank) / k)
        except ValueError:
            out[tag] = 0.0
    return out


def _analyze_with_retry(analyze_host: str, audio_path: str,
                         labels: List[str], timeout: float = 90.0,
                         max_attempts: int = 3) -> Dict[str, Any]:
    """Call the box analyze service with bounded retries + backoff.

    The shared-3090 box can starve the analyze service when the engine is
    busy (autolabel / training). Without retry we lose the score for any
    take that hits a transient timeout — and we generated it already, so
    that's a wasted ~80s of GPU work. Backoff sequence: 15s, 30s, 60s.
    """
    from . import analyze_py
    delays = [15, 30, 60]
    last_err: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            return analyze_py.analyze(analyze_host, audio_path, labels=labels,
                                       with_tags=True, with_key=False,
                                       timeout=timeout)
        except Exception as exc:
            last_err = exc
            if attempt < max_attempts - 1:
                wait = delays[min(attempt, len(delays) - 1)]
                print(f"[lora_eval] analyze attempt {attempt+1}/{max_attempts} "
                      f"failed ({exc}); retrying in {wait}s")
                time.sleep(wait)
    raise RuntimeError(f"analyze failed after {max_attempts} attempts: {last_err}")


def _set_adapter_state(engine_host: str, lora_path: str, adapter_name: str,
                       scale: float) -> None:
    """Unload all + load just this adapter + set its scale. Keeps state clean
    between checkpoints so we never score with two adapters compounding."""
    eng = engine_host if engine_host.startswith("http") else f"http://{engine_host}"
    requests.post(f"{eng}/v1/lora/unload", timeout=30)
    requests.post(f"{eng}/v1/lora/load",
                  json={"lora_path": lora_path, "adapter_name": adapter_name},
                  timeout=120).raise_for_status()
    requests.post(f"{eng}/v1/lora/scale",
                  json={"scale": float(scale), "adapter_name": adapter_name},
                  timeout=30).raise_for_status()


def evaluate_adapter(
    *,
    mac_base: str,
    engine_host: str,
    analyze_host: str,
    library_dir: str,
    lora_path: str,
    adapter_name: str,
    scale: float,
    prompt: str,
    lyrics: str,
    seed: int,
    duration: int,
    bpm: int,
    keyscale: str,
    model: str,
    target_tags: List[str],
    negative_tags: List[str],
    top_k: int = 10,
    label_prefix: str = "Eval",
) -> Dict[str, Any]:
    """Generate one take + score it via CLAP. Returns:
        {ckpt, scale, target_hits: {tag: bool}, negative_hits: {tag: bool},
         top_tags: [..], fitness: int (0-len(target_tags)),
         negative_fitness: int (0-len(negative_tags)),
         audio_path: str | None, job_id: str, error: str | None}
    """
    title = f"{label_prefix} — {adapter_name} @ {scale:.2f}"
    result: Dict[str, Any] = {
        "lora_path": lora_path, "adapter_name": adapter_name, "scale": scale,
        "target_tags": list(target_tags), "top_tags": [],
        # binary presence-in-top-K (v1, kept for back-compat)
        "target_hits": {}, "negative_hits": {},
        "fitness": 0, "negative_fitness": 0,
        # position-weighted (v2, recommended)
        "target_weights": {}, "negative_weights": {},
        "fitness_weighted": 0.0, "negative_fitness_weighted": 0.0,
        "audio_path": None, "job_id": None, "title": title, "error": None,
    }
    try:
        # 1. Engine state: only this adapter loaded at this scale
        _set_adapter_state(engine_host, lora_path, adapter_name, scale)
        # 2. Generate via Mac
        gen_body = {
            "tags": prompt, "lyrics": lyrics, "instrumental": False,
            "duration": duration, "bpm": bpm, "keyscale": keyscale,
            "seed": seed, "model": model, "title": title,
        }
        gr = requests.post(f"{mac_base.rstrip('/')}/api/generate",
                            json=gen_body, timeout=60)
        gr.raise_for_status()
        job_id = gr.json().get("job_id")
        result["job_id"] = job_id
        if not job_id:
            result["error"] = "no job_id from /api/generate"
            return result
        # 3. Wait for job
        job = _wait_for_job(mac_base, job_id)
        if job.get("status") != "done":
            result["error"] = f"job ended {job.get('status')}: {job.get('error')}"
            return result
        # 4. Locate audio on Mac disk
        candidate = os.path.join(library_dir, f"{job_id}.wav")
        if not os.path.exists(candidate):
            candidate = os.path.join(library_dir, f"{job_id}.mp3")
        if not os.path.exists(candidate):
            result["error"] = f"audio not found at library/{job_id}.{{wav,mp3}}"
            return result
        result["audio_path"] = candidate
        # 5. Send to analyze with merged label vocabulary (with retry —
        # see _analyze_with_retry for why)
        labels = list(target_tags) + list(negative_tags)
        an = _analyze_with_retry(analyze_host, candidate, labels=labels,
                                  timeout=90.0, max_attempts=3)
        top_tags = [t.lower().strip() for t in (an.get("tags") or [])][:top_k]
        result["top_tags"] = top_tags
        # 6a. v1 binary scoring (back-compat)
        target_hits = {t: (t.lower() in top_tags) for t in target_tags}
        neg_hits = {t: (t.lower() in top_tags) for t in negative_tags}
        result["target_hits"] = target_hits
        result["negative_hits"] = neg_hits
        result["fitness"] = sum(1 for v in target_hits.values() if v)
        result["negative_fitness"] = sum(1 for v in neg_hits.values() if v)
        # 6b. v2 position-weighted scoring (recommended signal)
        target_w = _position_weighted_score(list(target_tags), top_tags, top_k)
        neg_w = _position_weighted_score(list(negative_tags), top_tags, top_k)
        result["target_weights"] = target_w
        result["negative_weights"] = neg_w
        result["fitness_weighted"] = round(sum(target_w.values()), 4)
        result["negative_fitness_weighted"] = round(sum(neg_w.values()), 4)
        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


def _persist(library_dir: str, dataset: str, curve: Dict[str, Any]) -> str:
    out_dir = os.path.join(library_dir, "lora_train_history")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset}_fitness.json")
    try:
        with open(out_path, "w") as f:
            json.dump(curve, f, indent=2)
    except Exception as e:
        print(f"[lora_eval] persist warning: {e}")
    return out_path


def evaluate_dataset(
    *,
    mac_base: str,
    engine_host: str,
    analyze_host: str,
    library_dir: str,
    dataset: str,
    ckpts: List[Dict[str, str]],
    scales: List[float],
    prompt: str,
    lyrics: str,
    seed: int = 42,
    duration: int = 40,
    bpm: int = 132,
    keyscale: str = "D minor",
    model: str = "acestep-v15-xl-sft",
    target_tags: Optional[List[str]] = None,
    negative_tags: Optional[List[str]] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Run the full eval grid: each (ckpt, scale) pair gets a take + a score.
    Persists curve to library/lora_train_history/<dataset>_fitness.json.

    ckpts: list of {"label": str (e.g. "epoch_50_loss_0.6919"),
                     "lora_path": str (full box path to lokr_weights.safetensors),
                     "epoch": int | None}
    scales: list of floats to test per checkpoint, e.g. [0.3, 0.5].
    """
    target_tags = target_tags or list(DEFAULT_TARGET_TAGS_POWER_METAL)
    negative_tags = negative_tags or list(DEFAULT_NEGATIVE_TAGS)
    curve: Dict[str, Any] = {
        "dataset": dataset, "started_at": time.time(),
        "prompt": prompt, "seed": seed, "duration": duration,
        "target_tags": target_tags, "negative_tags": negative_tags,
        "scales": scales, "ckpts_evaluated": [], "results": [],
    }
    for ck in ckpts:
        label = ck.get("label") or os.path.basename(ck["lora_path"])
        adapter_name = f"eval_{label}".replace("/", "_").replace("\\", "_")[:60]
        for scale in scales:
            print(f"[lora_eval] {label} @ {scale} — generating + scoring …")
            r = evaluate_adapter(
                mac_base=mac_base, engine_host=engine_host,
                analyze_host=analyze_host, library_dir=library_dir,
                lora_path=ck["lora_path"], adapter_name=adapter_name,
                scale=scale, prompt=prompt, lyrics=lyrics, seed=seed,
                duration=duration, bpm=bpm, keyscale=keyscale, model=model,
                target_tags=target_tags, negative_tags=negative_tags,
                top_k=top_k, label_prefix=f"Eval {dataset}/{label}",
            )
            r["ckpt_label"] = label
            r["ckpt_epoch"] = ck.get("epoch")
            curve["results"].append(r)
            # Persist as we go so a long run isn't lost on crash
            _persist(library_dir, dataset, curve)
        curve["ckpts_evaluated"].append(label)
    curve["completed_at"] = time.time()
    _persist(library_dir, dataset, curve)
    return curve
