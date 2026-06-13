"""On-demand VRAM free route - Crucible patch 2026-06-13.

Exposes a programmatic GPU-memory reclaim over HTTP so the Mac/agent can drop the
persistent serving model (DiT/VAE/text-encoder) and/or the vLLM LM mid-session WITHOUT
an OS-level engine restart. This just makes the engine's OWN free idiom callable on demand:
the DiT loader already does `del self.model; torch.cuda.empty_cache()` before every reload
(core/generation/handler/init_service_loader.py), and LLMHandler.unload() best-effort
releases the LM. So a fresh-boot is NOT required to reclaim VRAM - this route reclaims it
in-process.

Pairs with /v1/reinitialize: after /v1/free nulls handler.model, /v1/reinitialize reloads
it from handler.last_init_params (it reloads whenever handler.model is None). Sequence:
    /v1/free            -> VRAM back toward idle (DiT + LM dropped)
    /v1/training/estimate (or any heavy op that wants a clean GPU)
    /v1/reinitialize    -> serving model back, NO OS restart

CAVEAT: vLLM may retain a residual pool that a Python-level unload cannot fully reclaim
(unload() skips destroy_model_parallel). For a guaranteed 100% LM teardown an OS restart is
still the only certainty. The DiT/VAE/text-encoder path frees cleanly.

Refuses while training is in progress (the trainer holds the decoder).
"""
from __future__ import annotations

import gc
from typing import Any, Callable, Dict, Optional

import torch
from fastapi import Depends, FastAPI
from loguru import logger
from pydantic import BaseModel, Field

from acestep.api.train_api_models import initialize_training_state


class FreeRequest(BaseModel):
    free_dit: bool = Field(
        True,
        description="Drop the persistent DiT/VAE/text-encoder (handler.model). Reload later via /v1/reinitialize.",
    )
    free_llm: bool = Field(
        True,
        description="Unload the vLLM LM (best-effort; may leave a residual pool).",
    )


def _vram_snapshot() -> Dict[str, float]:
    """Return current CUDA allocator stats (proof the free engaged)."""
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / 1024 ** 3, 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1024 ** 3, 3),
    }


def register_free_route(
    app: FastAPI,
    verify_api_key: Callable[..., Any],
    wrap_response: Callable[[Any, int, Optional[str]], Dict[str, Any]],
) -> None:
    """Register POST /v1/free. Wire in register_training_api_routes."""

    @app.post("/v1/free")
    def free_vram(request: FreeRequest, _: None = Depends(verify_api_key)):
        # Refuse mid-training: the trainer holds a live ref to the decoder, so nulling
        # handler.model would both break the run AND fail to free (ref still held).
        try:
            initialize_training_state(app)
            if app.state.training_state.get("is_training", False):
                return wrap_response(
                    None, code=409,
                    error="Training in progress; refusing to free VRAM. Stop training first.",
                )
        except Exception:
            pass

        before = _vram_snapshot()
        freed = []

        if request.free_llm:
            llm = getattr(app.state, "llm_handler", None)
            if llm is not None and getattr(llm, "llm_initialized", False):
                try:
                    llm.unload()
                    app.state._llm_initialized = False
                    app.state._llm_init_error = None
                    freed.append("LLM")
                except Exception:
                    logger.exception("[free] LLM unload failed")

        if request.free_dit:
            # Drop any lingering post-run trainer ref so `del` actually frees the model
            # (only safe when idle - guarded above).
            try:
                ts = app.state.training_state
                if not ts.get("is_training", False):
                    ts.pop("trainer", None)
                    ts.pop("_component_manager", None)
            except Exception:
                pass

            handler = getattr(app.state, "handler", None)
            if handler is not None and getattr(handler, "model", None) is not None:
                try:
                    handler.model = None
                    handler.vae = None
                    handler.text_encoder = None
                    handler.text_tokenizer = None
                    handler.config = None
                    handler.silence_latent = None
                    freed.append("DiT/VAE/TextEncoder")
                except Exception:
                    logger.exception("[free] DiT free failed")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        after = _vram_snapshot()
        reclaimed = None
        if before and after:
            reclaimed = round(before.get("reserved_gb", 0.0) - after.get("reserved_gb", 0.0), 3)

        return wrap_response({
            "freed": freed,
            "vram_before": before,
            "vram_after": after,
            "reclaimed_gb": reclaimed,
            "note": "Reload the serving model with /v1/reinitialize (no OS restart). vLLM may keep a residual pool.",
        })
