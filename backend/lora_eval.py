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

Limitations (v1):
- Tag *presence* in top-K, not raw confidence scores (the analyze service
  exposes only sorted top-K). Position-weighted scoring TBD if v1 signal
  isn't sensitive enough.
- Caller supplies the list of checkpoint paths; no automatic enumeration of
  train/checkpoints/epoch_N/ folders yet. v2 adds enumeration via a box-side
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


# Default target tags for power-metal / symphonic-metal LoRAs.  Caller can
# override per dataset / per experiment.  Curated against the engine analyze
# service's existing metal vocabulary so they're scoreable.
DEFAULT_TARGET_TAGS_POWER_METAL = [
    "power metal", "symphonic metal", "heavy metal", "anthemic",
    "harmonized lead guitars", "double bass drums", "soaring vocals",
    "epic chorus",
]
DEFAULT_TARGET_TAGS_SYMPHONIC_METAL = [
    "symphonic metal", "operatic vocals", "female vocals", "soprano",
    "orchestral", "choir", "cinematic", "epic", "dramatic",
]

# Negative tags — if these score in top-K it's a red flag (model drifted off
# target genre). Caller can override.
DEFAULT_NEGATIVE_TAGS = ["pop music", "hip hop", "electronic dance music",
                          "country music", "jazz"]


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
        "target_tags": list(target_tags), "top_tags": [], "target_hits": {},
        "negative_hits": {}, "fitness": 0, "negative_fitness": 0,
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
        # 5. Send to analyze with merged label vocabulary
        labels = list(target_tags) + list(negative_tags)
        from . import analyze_py
        an = analyze_py.analyze(analyze_host, candidate, labels=labels,
                                with_tags=True, with_key=False, timeout=300)
        top_tags = [t.lower().strip() for t in (an.get("tags") or [])][:top_k]
        result["top_tags"] = top_tags
        # 6. Score: how many target tags landed in top-K? Same for negatives.
        target_set = {t.lower().strip() for t in target_tags}
        neg_set = {t.lower().strip() for t in negative_tags}
        target_hits = {t: (t.lower() in top_tags) for t in target_tags}
        neg_hits = {t: (t.lower() in top_tags) for t in negative_tags}
        result["target_hits"] = target_hits
        result["negative_hits"] = neg_hits
        result["fitness"] = sum(1 for v in target_hits.values() if v)
        result["negative_fitness"] = sum(1 for v in neg_hits.values() if v)
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
