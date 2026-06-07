"""LoKR training start route registration."""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Any, Callable, Dict, Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
from loguru import logger

from acestep.api.train_api_models import StartLoKRTrainingRequest, initialize_training_state
from acestep.api.train_api_runtime import RuntimeComponentManager, unwrap_module
from acestep.handler import AceStepHandler


def register_lokr_training_start_route(
    app: FastAPI,
    verify_api_key: Callable[..., Any],
    wrap_response: Callable[[Any, int, Optional[str]], Dict[str, Any]],
    start_tensorboard: Callable[[FastAPI, str], Optional[str]],
) -> None:
    """Register the `/v1/training/start_lokr` route."""

    @app.post("/v1/training/start_lokr")
    async def start_lokr_training(request: StartLoKRTrainingRequest, _: None = Depends(verify_api_key)):
        """Start LoKr training from preprocessed tensors."""

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

        handler.model.decoder = unwrap_module(handler.model.decoder)
        mgr = RuntimeComponentManager(handler=handler, llm=app.state.llm_handler, app_state=app.state)
        mgr.move_decoder_to(str(handler.device))
        mgr.offload_vae_to_cpu()
        mgr.offload_text_encoder_to_cpu()
        mgr.offload_model_encoder_to_cpu()
        mgr.unload_llm()

        try:
            from acestep.training.configs import LoKRConfig as LoKRConfigClass, TrainingConfig
            from acestep.training.trainer import LoKRTrainer

            factor = request.lokr_factor
            if factor != -1:
                factor = int(factor)
                if factor == 0:
                    factor = 1
                factor = min(factor, 8)

            lokr_kwargs = dict(
                linear_dim=request.lokr_linear_dim,
                linear_alpha=request.lokr_linear_alpha,
                factor=factor,
                decompose_both=request.lokr_decompose_both,
                use_tucker=request.lokr_use_tucker,
                use_scalar=request.lokr_use_scalar,
                weight_decompose=request.lokr_weight_decompose,
            )
            # Crucible patch (2026-06-06): honor caller-chosen target_modules so we
            # can wrap the Qwen3MLP layers (gate/up/down_proj) on top of attention
            # for richer-style LoRAs (METAL_LORA_PLAN §13b). None -> leave LoKRConfig's
            # default (attention-only) untouched. lokr_utils.inject_lokr_into_dit
            # already reads lokr_config.target_modules + logs the enabled/disabled
            # module counts, so a wrong suffix is visible in the engine log.
            if request.target_modules:
                lokr_kwargs["target_modules"] = list(request.target_modules)
            # Crucible patch (2026-06-07): plumb LyCORIS dropouts (METAL_LORA_PLAN
            # §13g Tier 2) into LoKRConfig ONLY when > 0, so the default-0.0 path
            # is byte-identical to prior runs. request.lora_dropout maps to the
            # LyCORIS `dropout` field; rank/module dropout pass through by name.
            if request.lora_dropout:
                lokr_kwargs["dropout"] = request.lora_dropout
            if request.rank_dropout:
                lokr_kwargs["rank_dropout"] = request.rank_dropout
            if request.module_dropout:
                lokr_kwargs["module_dropout"] = request.module_dropout
            lokr_config = LoKRConfigClass(**lokr_kwargs)
            training_config = TrainingConfig(
                shift=request.training_shift,
                learning_rate=request.learning_rate,
                batch_size=request.train_batch_size,
                gradient_accumulation_steps=request.gradient_accumulation,
                max_epochs=request.train_epochs,
                save_every_n_epochs=request.save_every_n_epochs,
                seed=request.training_seed,
                output_dir=request.output_dir,
                gradient_checkpointing=request.gradient_checkpointing,
                # Crucible patch (2026-05-29): pass val_split through so the
                # LoKr trainer's val_loader (+ best-checkpoint tracking) actually
                # gets samples. Without this it stayed at the dataclass default
                # of 0.0 -> no val_loader -> no checkpoints/best/.
                val_split=request.val_split,
                # Crucible patch (2026-05-30): plumb timestep sampling mode
                # through (METAL_LORA_PLAN §12). Trainer reads this in its
                # training_step to pick discrete vs continuous sampling.
                timestep_sampling_mode=request.timestep_sampling_mode,
            )
            trainer = LoKRTrainer(dit_handler=handler, lokr_config=lokr_config, training_config=training_config)
        except Exception as exc:
            training_state["is_training"] = False
            mgr.restore()
            return wrap_response(None, code=500, error=f"Failed to start LoKR training: {exc}")

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
                "status": "Starting...",
                "loss_history": [],
                "training_log": "Starting...",
                "start_time": time.time(),
                "current_epoch": 0,
                "last_step_time": time.time(),
                "steps_per_second": 0.0,
                "estimated_time_remaining": 0.0,
                "error": None,
                "config": {
                    "adapter_type": "lokr",
                    "lokr_linear_dim": request.lokr_linear_dim,
                    "lokr_linear_alpha": request.lokr_linear_alpha,
                    "lokr_factor": request.lokr_factor,
                    "lokr_decompose_both": request.lokr_decompose_both,
                    "lokr_use_tucker": request.lokr_use_tucker,
                    "lokr_use_scalar": request.lokr_use_scalar,
                    "lokr_weight_decompose": request.lokr_weight_decompose,
                    "learning_rate": request.learning_rate,
                    "epochs": request.train_epochs,
                    "val_split": request.val_split,
                    "timestep_sampling_mode": request.timestep_sampling_mode,
                    # Crucible patch (2026-06-06): surface the resolved target set
                    # (request override or the LoKRConfig default) for visibility.
                    "target_modules": list(getattr(lokr_config, "target_modules", []) or []),
                    # Crucible patch (2026-06-07): surface resolved dropouts so the
                    # run's regularization shows in /v1/training/status + history.
                    "lora_dropout": getattr(lokr_config, "dropout", 0.0),
                    "rank_dropout": getattr(lokr_config, "rank_dropout", 0.0),
                    "module_dropout": getattr(lokr_config, "module_dropout", 0.0),
                },
                "_component_manager": mgr,
            }
        )
        training_state["tensorboard_url"] = start_tensorboard(app, tensorboard_logdir)

        def _runner() -> None:
            local_run_id = run_id
            try:
                for step, loss, status in trainer.train_from_preprocessed(request.tensor_dir, training_state):
                    if training_state.get("run_id") != local_run_id:
                        break
                    training_state["current_step"] = step
                    training_state["current_loss"] = loss
                    training_state["status"] = status
                    text = str(status)
                    match = re.search(r"Epoch (\d+)/(\d+)", text)
                    if match:
                        training_state["current_epoch"] = int(match.group(1))
                    if loss is not None and loss == loss and step > 0:
                        history = training_state.get("loss_history", [])
                        history.append({"step": step, "loss": float(loss)})
                        training_state["loss_history"] = history[-1000:]
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
                    logger.exception("Failed to restore decoder wrapper state after LoKR training")
                cm = training_state.pop("_component_manager", None)
                if cm is not None:
                    cm.restore()

        threading.Thread(target=_runner, daemon=True).start()
        return wrap_response(
            {
                "message": "LoKR training started",
                "tensor_dir": request.tensor_dir,
                "output_dir": request.output_dir,
                "config": training_state["config"],
            }
        )
