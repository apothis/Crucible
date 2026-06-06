# Engine patch 2026-06-06 - LoKr `target_modules` (attention + MLP targeting)

INCREMENTAL on top of `patches/engine-2026-05-30/` (all of those patches are still
required - DCW off, auto-label, LoKr preset honor, val_split, val loop, continuous
timestep). These two files SUPERSEDE their 2026-05-30 versions.

## Why
The engine's `LoKRConfig.target_modules` defaults to attention-only
(`q_proj,k_proj,v_proj,o_proj`) and `StartLoKRTrainingRequest` never exposed it, so
every LoKr we trained wrapped attention only. For STYLE LoRAs the feed-forward / MLP
layers carry the artist's timbre - the highest-value capacity to add
(METAL_LORA_PLAN §13b). ACE-Step's DiT MLP is `Qwen3MLP` with linear names
`gate_proj,up_proj,down_proj` (verified from
acestep/models/xl_base/modeling_acestep_v15_xl_base.py). So "attn+mlp" =
`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`.

## What changed
- `train_api_models.py` - add `target_modules: Optional[List[str]] = None` to
  `StartLoKRTrainingRequest` (+ `List` import). None = leave the engine default.
- `train_api_lokr_start_route.py` - pass `target_modules` into `LoKRConfig` ONLY when
  the request provides it; surface the resolved set in `training_state["config"]`.
  `lokr_utils.inject_lokr_into_dit` (Patch 3, 2026-05-29) already reads
  `lokr_config.target_modules` and logs `"LoKr target filter: enabled N (disabled M)
  for targets=[...]"` - use that log line to CONFIRM the MLP suffixes matched.

## Deploy (box)
Copy over the engine source, then OS-restart `run_acestep_api.bat` (a fresh boot is the
LoRA routine anyway):
- `train_api_models.py`            -> `<engine>/acestep/api/train_api_models.py`
- `train_api_lokr_start_route.py`  -> `<engine>/acestep/api/train_api_lokr_start_route.py`
Reverted by any engine `git pull` / re-running ACESTEP-ENGINE_AUTO_INSTALL.bat -
re-apply after engine updates (see METAL_LORA_PLAN §7a, [[engine-patches]]).

## Verify
Start a LoKr run from the Train tab with Targets = "Attention + MLP (style)". In the
engine log, the Patch-3 line should show a HIGHER enabled count than an attention-only
run (MLP modules now wrapped). If enabled count is unchanged, the suffixes didn't match
- check the model's actual Linear names.

## Mac side (already shipped, no deploy)
- `backend/acestep_train.py` `train_lokr(..., target_modules=None)` sends it when set.
- `backend/app.py` `/api/lora/train` forwards `body["target_modules"]` (lokr path only).
- `web/src/LoraTraining.tsx` Advanced -> "Targets" select + "Style preset" button.
Unpatched engines silently ignore the field (Pydantic drops unknowns) -> safe to send.
