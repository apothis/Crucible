"""LoKr gradient-sensitivity ("Fisher") estimation route - Crucible patch 2026-06-12.

Exposes acestep.training_v2.estimate.run_estimation over HTTP so the Mac/agent can rank which
decoder modules a dataset is most loss-sensitive to, then pass the top-K as target_modules to
/v1/training/start_lokr_v2 (targeted-capacity LoRA). run_estimation self-loads + unloads its own
decoder, so call this on a FRESH boot BEFORE /v1/init (a model already loaded would double-occupy
VRAM -> OOM on 24GB). SYNC route (FastAPI threadpool); ~1-3 min for num_batches passes.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from fastapi import Depends, FastAPI
from loguru import logger
from pydantic import BaseModel, Field


class EstimateRequest(BaseModel):
    dataset_dir: str = Field(..., description="Directory with preprocessed .pt tensors")
    checkpoint_dir: str = Field("E:/AI/MusicGen/AceStep/checkpoints",
                                description="Root dir containing the model variant subdir")
    variant: str = Field("xl_base", description="Model variant alias (xl_base/xl_sft/...)")
    num_batches: int = Field(15, ge=1, le=200, description="Forward/backward passes for the estimate")
    top_k: int = Field(64, ge=1, le=2000, description="How many top modules to return")
    granularity: str = Field("module", description="'module' (q/k/v/o + mlp projections) or 'layer'")
    cfg_ratio: float = Field(0.1, ge=0.0, le=1.0, description="CFG dropout to match training")


def register_estimate_route(
    app: FastAPI,
    verify_api_key: Callable[..., Any],
    wrap_response: Callable[[Any, int, Optional[str]], Dict[str, Any]],
) -> None:
    """Register POST /v1/training/estimate. Wire in register_training_api_routes."""

    @app.post("/v1/training/estimate")
    def estimate_sensitivity(request: EstimateRequest, _: None = Depends(verify_api_key)):
        # SYNC def on purpose: run_estimation blocks for minutes; FastAPI runs sync routes in a
        # threadpool so the event loop stays responsive.
        try:
            from acestep.training_v2.estimate import run_estimation
            results = run_estimation(
                checkpoint_dir=request.checkpoint_dir,
                variant=request.variant,
                dataset_dir=request.dataset_dir,
                num_batches=request.num_batches,
                batch_size=1,
                top_k=request.top_k,
                granularity=request.granularity,
                cfg_ratio=request.cfg_ratio,
            )
            return wrap_response({
                "modules": results,
                "count": len(results),
                "checkpoint_dir": request.checkpoint_dir,
                "variant": request.variant,
                "dataset_dir": request.dataset_dir,
                "num_batches": request.num_batches,
                "granularity": request.granularity,
            })
        except Exception as exc:  # noqa: BLE001
            logger.exception("[estimate] gradient-sensitivity estimation failed")
            return wrap_response(None, code=500, error=f"estimation failed: {exc}")
