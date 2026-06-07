# Engine patch 2026-06-07 - LoKr dropout (lora_dropout / rank_dropout / module_dropout)

INCREMENTAL on top of `patches/engine-2026-05-30/` AND `patches/engine-2026-06-06/`
(all prior patches still required - DCW off, auto-label, LoKr preset honor, val_split,
val loop, continuous timestep, target_modules). The four files here SUPERSEDE their
earlier versions:
- `configs.py`                     supersedes 2026-05-30
- `lokr_utils.py`                  supersedes 2026-05-30
- `train_api_models.py`            supersedes 2026-06-06
- `train_api_lokr_start_route.py`  supersedes 2026-06-06

## Why
Tier 2 of the noisiness fix (METAL_LORA_PLAN §13g). The lr-cooldown (Tier 1) reduced
frying but undertrained the style; the next lever to tame an over-geared / "fried"
adapter without losing capacity is REGULARIZATION via LyCORIS dropout. The LoRA path
(`StartTrainingRequest`) already exposes `lora_dropout`, but the LoKr path
(`StartLoKRTrainingRequest`) never did, and `LoKRConfig` had no dropout field, so it was
impossible to request. This wires it end to end.

## What changed
- `train_api_models.py` - add `lora_dropout`, `rank_dropout`, `module_dropout`
  (all `float`, default `0.0`, range [0,1]) to `StartLoKRTrainingRequest`.
- `configs.py` - add `dropout`, `rank_dropout`, `module_dropout` (default `0.0`) to
  `LoKRConfig` + its `to_dict()`.
- `train_api_lokr_start_route.py` - map `request.lora_dropout -> lokr_kwargs["dropout"]`
  (+ rank/module by name) ONLY when > 0; surface the resolved values in
  `training_state["config"]` (so they show in `/v1/training/status` + the Mac history).
- `lokr_utils.py` - add the dropout values to the `create_lycoris(...)` kwargs (both the
  plain and the `dora_wd=True` calls) ONLY when > 0; log the resolved dropout config.

## No-regression guarantee
ALL three default to `0.0` and are passed to LyCORIS ONLY when > 0. With no dropout
requested, every code path is byte-identical to the 2026-06-06 stack - existing runs are
unaffected, and older LyCORIS builds that don't accept these kwargs are never handed them.
`lora_dropout` maps to LyCORIS `dropout`; the name matches the LoRA request + Mac client.

## Deploy (box) - HELD until user finishes testing the 250ep lr-1e-3 run
Copy over the engine source, then OS-restart `run_acestep_api.bat` (fresh boot = the LoRA
routine anyway; the restart is the USER's action [[engine-restart-is-user-only]]):
- `train_api_models.py`            -> `<engine>/acestep/api/train_api_models.py`
- `train_api_lokr_start_route.py`  -> `<engine>/acestep/api/train_api_lokr_start_route.py`
- `configs.py`                     -> `<engine>/acestep/training/configs.py`
- `lokr_utils.py`                  -> `<engine>/acestep/training/lokr_utils.py`
Reverted by any engine `git pull` / re-running ACESTEP-ENGINE_AUTO_INSTALL.bat - re-apply
after engine updates (METAL_LORA_PLAN §7a, [[engine-patches]]).

## Verify after deploy
Start a LoKr run with `lora_dropout: 0.1`. In `/v1/training/status` the `config` block
should show `lora_dropout: 0.1`, and the engine log should print
`LoKr dropout config: dropout=0.1 rank_dropout=0.0 module_dropout=0.0`. If LyCORIS errors
on the kwarg, the build is too old - upgrade lycoris-lora or drop the unsupported knob.

## Mac side (shipped in this commit, NOT yet active)
- `backend/acestep_train.py` `train_lokr(..., lora_dropout=None, rank_dropout=None,
  module_dropout=None)` sends each only when set.
- `backend/app.py` `/api/lora/train` forwards `body["lora_dropout"]` etc. (lokr path).
Needs a backend restart to activate the SEND side (do it AFTER the current 250ep run so the
val-curve poller isn't interrupted). Until the ENGINE patch is also deployed, sending
`lora_dropout` is a harmless no-op (unpatched engine drops the unknown field).

## Suggested Tier 2 experiment (once deployed)
Same body as the §13g Tier 1 run + `"lora_dropout": 0.1`. Single variable = dropout.
Judge by ear: does it fry less at usable strength while keeping Tarja character?
