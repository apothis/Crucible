"""Runtime LoRA reconciliation for the ACE-Step engine.

Sets the engine's loaded adapters to EXACTLY a requested set before a
generation, with verification. This makes per-generation adapter state
deterministic: a take provably uses only the selected adapters at the
selected scales, with no residual/global drift (the class of bug that
caused a lot of confusion on 2026-05-31).

Design notes:
- The engine wraps responses in {"data": {...}} and reports loaded LoKr
  adapters under data.scales = {adapter_name: scale}. data.adapters is
  always [] for LoKr (discovery quirk) -- see [[engine-synthetic-default-mode]].
- The engine's no-arg /v1/lora/unload only clears the active adapter in some
  builds, so clear() falls back to per-name unloads and verifies empty.
- Modular by intent: app.py routes call reconcile()/clear(); if the LoRA
  backend changes (e.g. ComfyUI LoRA), only this file changes, not the routes
  or the picker UI.
"""
import requests

from .acestep_py import _base

TIMEOUT = 60
LOAD_TIMEOUT = 300


def _post(host, path, body, timeout=TIMEOUT):
    r = requests.post(_base(host) + path, json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()


def scales(host):
    """Return the engine's currently-loaded adapters as {name: scale}."""
    r = requests.get(_base(host) + "/v1/lora/status", timeout=15)
    r.raise_for_status()
    data = (r.json() or {}).get("data", {}) or {}
    return dict(data.get("scales") or {})


def clear(host):
    """Unload every adapter; verify the engine is back to pure base.

    no-arg unload first, then per-name fallback, then assert empty.
    Raises RuntimeError if adapters remain."""
    try:
        _post(host, "/v1/lora/unload", {})
    except Exception:
        pass
    cur = scales(host)
    if cur:
        for name in list(cur):
            try:
                _post(host, "/v1/lora/unload", {"adapter_name": name})
            except Exception:
                pass
        cur = scales(host)
    if cur:
        raise RuntimeError(f"could not clear adapters, still loaded: {sorted(cur)}")


def reconcile(host, loras):
    """Set the engine to EXACTLY `loras`, verified.

    `loras`: list of {"path": <engine-side safetensors path>, "scale": float,
             "name": optional stable name}. Empty/None => pure base.

    Returns the applied {name: scale} map. Raises if the final engine state
    does not match the request (so a caller never generates with the wrong
    adapters silently)."""
    clear(host)
    applied = {}
    for i, spec in enumerate(loras or []):
        path = (spec or {}).get("path")
        if not path:
            continue
        name = spec.get("name") or f"slot{i}"
        scale = float(spec.get("scale", 1.0))
        _post(host, "/v1/lora/load", {"lora_path": path, "adapter_name": name}, timeout=LOAD_TIMEOUT)
        _post(host, "/v1/lora/scale", {"scale": scale, "adapter_name": name})
        applied[name] = scale
    cur = scales(host)
    if set(cur) != set(applied):
        raise RuntimeError(
            f"adapter set mismatch after reconcile: requested {sorted(applied)}, engine has {sorted(cur)}")
    return applied
