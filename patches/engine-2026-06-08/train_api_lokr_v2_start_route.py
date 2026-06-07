"""LoKR *v2* training start route - runs the training_v2 FixedLoRATrainer IN-PROCESS.

Crucible patch (2026-06-08, METAL_LORA_PLAN §13a Path B adoption). The engine only
exposes the v1 trainer over HTTP (`/v1/training/start_lokr`). v2 (`acestep/training_v2`)
is CLI-only but is more advanced: optimizer choice (incl. Prodigy), scheduler choice,
CFG dropout (`cfg_ratio`), native continuous timestep, Fisher-info ranks. This route
makes v2 drivable over HTTP WITHOUT a subprocess (which would double-load the model into
VRAM and hit the "VRAM never fully frees" bug) by reusing the ALREADY-LOADED engine model,
exactly like the v1 `start_lokr` route does.

================================ DEPLOY-TIME VERIFICATION ============================
This file was written against GitHub `main`. Before the first REAL run, smoke-test it
(README "Verify"). The following are GitHub-main-derived and MUST match the box's
installed engine - each is isolated so a mismatch fails LOUDLY at request time (HTTP 500
with the exception), never mid-run:
  [V1] import path  acestep.training_v2.configs : LoKRConfigV2, TrainingConfigV2
  [V2] import path  acestep.training_v2.trainer_fixed : FixedLoRATrainer
  [V3] LoKRConfigV2 / TrainingConfigV2 constructor field names (see _build_v2_configs)
  [V4] FixedLoRATrainer(model, adapter_cfg, train_cfg) signature + .train() is a generator
       yielding objects with .step/.loss/.msg/.epoch (TrainingUpdate)
  [V5] DATASET-DIR TENSOR COMPATIBILITY: v2 reads via acestep.training.data_module
       (same as v1), so our existing v1 `tensors/` SHOULD load. If v2 needs a key v1
       didn't write (e.g. context_latents for cfg), the smoke run fails at data load -
       then re-preprocess once via v2 (--preprocess) and point dataset_dir there.
=====================================================================================
"""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from acestep.api.train_api_models import initialize_training_state
from acestep.api.train_api_runtime import RuntimeComponentManager, unwrap_module
from acestep.handler import AceStepHandler


class StartLoKRV2TrainingRequest(BaseModel):
    """Request payload for starting LoKr training via the training_v2 trainer."""

    tensor_dir: str = Field(..., description="Directory with preprocessed tensors (v1 tensors/ dir is expected to be compatible - see [V5])")
    output_dir: str = Field(default="./lokr_v2_output", description="Output directory (writes lokr_weights.safetensors, same as v1)")
    # LoKr capacity (same meaning as v1)
    lokr_linear_dim: int = Field(default=64, ge=1, le=256)
    lokr_linear_alpha: int = Field(default=128, ge=1, le=512)
    lokr_factor: int = Field(default=-1)
    lokr_decompose_both: bool = Field(default=False)
    lokr_use_tucker: bool = Field(default=False)
    lokr_use_scalar: bool = Field(default=False)
    lokr_weight_decompose: bool = Field(default=False, description="DoRA mode (pair with low lr)")
    target_modules: Optional[List[str]] = Field(default=None, description="None = v2 default (attention-only)")
    # v2 EXTRAS - the reason this route exists
    optimizer_type: str = Field(default="adamw", description="adamw | adamw8bit | adafactor | prodigy")
    scheduler_type: str = Field(default="cosine", description="cosine | cosine_restarts | linear | constant | constant_with_warmup")
    cfg_ratio: float = Field(default=0.15, ge=0.0, le=1.0, description="CFG/caption dropout during training")
    attention_type: str = Field(default="both", description="self | cross | both")
    dropout: float = Field(default=0.0, ge=0.0, le=1.0, description="adapter dropout")
    # Training schedule
    learning_rate: float = Field(default=1e-4, gt=0.0)
    train_epochs: int = Field(default=100, ge=1)
    train_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation: int = Field(default=4, ge=1)
    save_every_n_epochs: int = Field(default=10, ge=1)
    training_shift: float = Field(default=3.0, ge=0.0)
    training_seed: int = Field(default=42)
    gradient_checkpointing: bool = Field(default=True)


def _build_v2_configs(request: "StartLoKRV2TrainingRequest"):
    """Construct v2 adapter_cfg + train_cfg directly (no CLI parser, no checkpoint_dir
    path validation - the trainer receives the already-loaded model so it never needs
    to load one). Field names are [V3] - verify against the box's training_v2/configs.py.
    """
    from acestep.training_v2.configs import LoKRConfigV2, TrainingConfigV2  # [V1]

    target_modules = list(request.target_modules) if request.target_modules else [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ]
    adapter_cfg = LoKRConfigV2(
        linear_dim=request.lokr_linear_dim,
        linear_alpha=request.lokr_linear_alpha,
        factor=request.lokr_factor,
        decompose_both=request.lokr_decompose_both,
        use_tucker=request.lokr_use_tucker,
        use_scalar=request.lokr_use_scalar,
        weight_decompose=request.lokr_weight_decompose,
        target_modules=target_modules,
        attention_type=request.attention_type,
    )
    train_cfg = TrainingConfigV2(
        shift=request.training_shift,
        learning_rate=request.learning_rate,
        batch_size=request.train_batch_size,
        gradient_accumulation_steps=request.gradient_accumulation,
        max_epochs=request.train_epochs,
        save_every_n_epochs=request.save_every_n_epochs,
        seed=request.training_seed,
        output_dir=request.output_dir,
        dataset_dir=request.tensor_dir,
        gradient_checkpointing=request.gradient_checkpointing,
        adapter_type="lokr",
        optimizer_type=request.optimizer_type,
        scheduler_type=request.scheduler_type,
        cfg_ratio=request.cfg_ratio,
    )
    return adapter_cfg, train_cfg


def register_lokr_v2_training_start_route(
    app: FastAPI,
    verify_api_key: Callable[..., Any],
    wrap_response: Callable[[Any, int, Optional[str]], Dict[str, Any]],
    start_tensorboard: Callable[[FastAPI, str], Optional[str]],
) -> None:
    """Register the `/v1/training/start_lokr_v2` route. Wire this in
    train_api_service.register_training_api_routes right after the v1 call (see README)."""

    @app.post("/v1/training/start_lokr_v2")
    async def start_lokr_v2_training(request: StartLoKRV2TrainingRequest, _: None = Depends(verify_api_key)):
        """Start LoKr training via the training_v2 FixedLoRATrainer, in-process."""

        initialize_training_state(app)
        training_state = app.state.training_state
        if training_state.get("is_training", False):
            raise HTTPException(status_code=400, detail="Training already in progress")

        handler: AceStepHandler = app.state.handler
        if handler is None or handler.model is None:
            raise HTTPException(status_code=500, detail="Model not initialized")
        if not hasattr(handler.model, "decoder") or handler.model.decoder is None:
            raise HTTPException(
                status_code=500,
                detail="Decoder not found. Please reload the model via /v1/reinitialize before training.",
            )

        # Same VRAM management as the v1 start_lokr route: decoder to GPU, everything
        # else off, LM unloaded. Proven to fit on the 3090 (it's how v1 trains today).
        handler.model.decoder = unwrap_module(handler.model.decoder)
        mgr = RuntimeComponentManager(handler=handler, llm=app.state.llm_handler, app_state=app.state)
        mgr.move_decoder_to(str(handler.device))
        mgr.offload_vae_to_cpu()
        mgr.offload_text_encoder_to_cpu()
        mgr.offload_model_encoder_to_cpu()
        mgr.unload_llm()

        try:
            from acestep.training_v2.trainer_fixed import FixedLoRATrainer  # [V2]
            adapter_cfg, train_cfg = _build_v2_configs(request)
            # [V4] trainer receives the ALREADY-LOADED full model (it accesses
            # self.model.decoder internally); it does NOT load its own.
            trainer = FixedLoRATrainer(handler.model, adapter_cfg, train_cfg)
        except Exception as exc:
            training_state["is_training"] = False
            mgr.restore()
            return wrap_response(None, code=500, error=f"Failed to start LoKR v2 training: {exc}")

        tensorboard_logdir = os.path.join(request.output_dir, "logs")
        os.makedirs(tensorboard_logdir, exist_ok=True)

        run_id = str(uuid4())
        training_state.update(
            {
                "is_training": True,
                "should_stop": False,
                "run_id": run_id,
                "trainer": trainer,
                "tensor_dir": request.tensor_dir,
                "tensorboard_logdir": tensorboard_logdir,
                "current_step": 0,
                "current_loss": None,
                "status": "Starting (v2)...",
                "loss_history": [],
                "training_log": "Starting v2 trainer...",
                "start_time": time.time(),
                "current_epoch": 0,
                "last_step_time": time.time(),
                "steps_per_second": 0.0,
                "estimated_time_remaining": 0.0,
                "error": None,
                "config": {
                    "adapter_type": "lokr",
                    "trainer": "v2",
                    "lokr_linear_dim": request.lokr_linear_dim,
                    "lokr_linear_alpha": request.lokr_linear_alpha,
                    "lokr_factor": request.lokr_factor,
                    "lokr_weight_decompose": request.lokr_weight_decompose,
                    "learning_rate": request.learning_rate,
                    "epochs": request.train_epochs,
                    "optimizer_type": request.optimizer_type,
                    "scheduler_type": request.scheduler_type,
                    "cfg_ratio": request.cfg_ratio,
                    "attention_type": request.attention_type,
                    "target_modules": list(request.target_modules or []),
                },
                "_component_manager": mgr,
            }
        )
        training_state["tensorboard_url"] = start_tensorboard(app, tensorboard_logdir)

        def _runner() -> None:
            local_run_id = run_id
            try:
                # [V4] train() yields TrainingUpdate(step, loss, msg, kind, epoch=, ...)
                for update in trainer.train():
                    if training_state.get("run_id") != local_run_id:
                        break
                    step = getattr(update, "step", 0)
                    loss = getattr(update, "loss", None)
                    msg = getattr(update, "msg", "") or str(update)
                    epoch = getattr(update, "epoch", None)
                    training_state["current_step"] = step
                    training_state["current_loss"] = loss
                    # Keep the v1 "Epoch N/M" shape in status so the Mac poller's
                    # existing regex still extracts epoch progress.
                    training_state["status"] = msg
                    if epoch is not None:
                        training_state["current_epoch"] = int(epoch)
                    else:
                        match = re.search(r"Epoch (\d+)/(\d+)", str(msg))
                        if match:
                            training_state["current_epoch"] = int(match.group(1))
                    if loss is not None and loss == loss and step and step > 0:
                        history = training_state.get("loss_history", [])
                        history.append({"step": step, "loss": float(loss)})
                        training_state["loss_history"] = history[-1000:]
                    if getattr(update, "kind", "") == "fail":
                        training_state["error"] = str(msg)
                    if training_state.get("should_stop", False):
                        break
            except Exception as exc:
                training_state["error"] = str(exc)
            finally:
                training_state["is_training"] = False
                try:
                    if handler.model is not None and getattr(handler.model, "decoder", None) is not None:
                        handler.model.decoder = unwrap_module(handler.model.decoder)
                        handler.model.decoder.eval()
                except Exception:
                    logger.exception("Failed to restore decoder wrapper state after LoKR v2 training")
                cm = training_state.pop("_component_manager", None)
                if cm is not None:
                    cm.restore()

        threading.Thread(target=_runner, daemon=True).start()
        return wrap_response(
            {
                "message": "LoKR v2 training started",
                "tensor_dir": request.tensor_dir,
                "output_dir": request.output_dir,
                "config": training_state["config"],
            }
        )
