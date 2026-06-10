# Engine patch 2026-06-08 - training_v2 over HTTP (in-process LoKr v2 trainer)

INCREMENTAL, ADDITIVE. Does not supersede any prior patch (2026-05-30 / 2026-06-06 /
2026-06-07 all still required). Adds ONE new route file + a 2-line wiring edit.

## FIX 2026-06-10b - `dropout` (normal) is a NO-OP for LoKr; use rank_dropout/module_dropout
After the 06-10a fix below got dropout=0.1 all the way to LyCORIS, the engine console showed
`[WARN]LoHa/LoKr haven't implemented normal dropout yet.` x352 (once per module) + applied nothing.
VERIFIED in `lycoris/modules/lokr.py`: `dropout` (normal) is stored + warned + IGNORED (line ~200);
only `rank_dropout` (line ~375, drops rank components each step) and `module_dropout` (line ~544,
skips whole modules) are IMPLEMENTED for LoKr. FIX: the v2 route now also declares + setattrs +
echoes `rank_dropout` and `module_dropout` (inject_lokr_into_dit already reads all three via getattr,
and configs.py/lokr_utils.py are already deployed). Only THIS route file needs re-copying. For LoKr
regularization send **`rank_dropout`** (the dropout analog), NOT `dropout`. VERIFY after restart:
console shows NO "[WARN]...normal dropout" spam AND `LoKr dropout config: ... rank_dropout=0.1`.

## FIX 2026-06-10a - adapter `dropout` was declared but DROPPED ON THE FLOOR (REDEPLOY REQUIRED)
The `StartLoKRV2TrainingRequest.dropout` field existed but `_build_v2_configs` never passed it
into `LoKRConfigV2(...)`, so any `dropout` value was silently ignored = a no-op (caught when a
`dropout:0.1` run showed no dropout in `/v1/training/status` config + no "LoKr dropout config:" log
line). FIX (in `train_api_lokr_v2_start_route.py`): (1) set dropout via `setattr(adapter_cfg, "dropout",
request.dropout)` AFTER building LoKRConfigV2 - NOT a constructor kwarg (a constructor kwarg HARD-
CRASHES with "unexpected keyword argument 'dropout'" if LoKRConfig lacks the field; setattr +
`inject_lokr_into_dit`'s `getattr` read works regardless); (2) add `"dropout"` to the status `config`
echo so engagement is verifiable.
DISCOVERY 2026-06-10: the 2026-06-07 dropout patch was NOT live on the box (v1
`StartLoKRTrainingRequest` had no dropout fields + `LoKRConfigV2(dropout=)` threw). CAUSE = it was
NEVER DEPLOYED (NOT a git-pull reversion - user confirmed no git pull): the 06-07 README marked
"Deploy HELD until the 250ep run finishes", then we pivoted to the v2/Prodigy path (06-08) and never
circled back. Unnoticed because dropout was never used until now. The dropout machinery (LoKRConfig
field + `inject_lokr_into_dit` read) lives in the 06-07 `configs.py` + `lokr_utils.py`, so those MUST
be deployed (FIRST TIME) alongside this route fix. Box-diff-checked both vs
current upstream main 2026-06-10 = clean (upstream + our additions only). REDEPLOY: 06-07 `configs.py`
+ `lokr_utils.py` AND this route file, then OS-restart. VERIFY: a `dropout:0.1` run shows
`dropout: 0.1` in `/v1/training/status` `config` AND the engine log prints `LoKr dropout config:
dropout=0.1 ...` - gate on BOTH before trusting the run.

## Why
The engine only exposes the v1 trainer over HTTP. v2 (`acestep/training_v2`) is CLI-only
but more advanced (METAL_LORA_PLAN §13a Path B): **optimizer choice incl. Prodigy**,
scheduler choice, **CFG dropout (`cfg_ratio`)**, native continuous timestep, Fisher-info
ranks. Instead of hand-porting each v2 feature onto v1 one patch at a time (we already did
continuous timestep, target_modules, dropout), this exposes v2 itself over HTTP.

## Design (verified from GitHub main source)
- v2's `FixedLoRATrainer(model, adapter_cfg, train_cfg)` takes an **already-loaded model**
  and `train()` is a **generator** yielding `TrainingUpdate(step, loss, msg, kind, epoch=)`.
- So the new route runs v2 **in-process**, reusing the engine's loaded decoder (same VRAM
  management as v1 `start_lokr`) - NO subprocess, NO second model in VRAM.
- v2 reads data via the **same `acestep.training.data_module`** v1 uses -> our existing
  v1 `tensors/` should feed it directly.
- v2 LoKr saves the **same `lokr_weights.safetensors`** (lycoris) -> loads in our picker
  unchanged.

## Files (both are full files - copy over, no hand-editing)
- `train_api_lokr_v2_start_route.py` (NEW) -> `<engine>/acestep/api/train_api_lokr_v2_start_route.py`
  Registers `POST /v1/training/start_lokr_v2`. Self-contained request model.
- `train_api_service.py` (FULLY PATCHED) -> `<engine>/acestep/api/train_api_service.py`
  Adds the v2 import + `register_lokr_v2_training_start_route(...)` call. Built from GitHub
  `main`; verified to differ from upstream by ONLY those two additions (import line + the
  registration block right after the v1 `register_lokr_training_start_route(...)` call).

### IMPORTANT - verify train_api_service.py before swapping (high blast radius)
This file registers ALL training routes and we have NEVER captured the box's own copy (it
is the one file in this patch built purely from GitHub main). If the box engine is a
different commit, a blind full-file replace could drop/alter routes. So before swapping:
```
# on the box, back up + diff our version against the live one
copy acestep\api\train_api_service.py acestep\api\train_api_service.py.bak
fc acestep\api\train_api_service.py <our train_api_service.py>     # Windows diff
```
PASS = the only differences are the 2 Crucible additions above -> safe to copy ours over.
If there are OTHER differences, the box is a different version: do NOT use our full file -
instead hand-add just the import + the registration call to the box's own file (the 2
additions are small and self-contained).

## PREREQ - install Prodigy in the engine venv (else it silently uses AdamW)
v2 `optim.py` does `from prodigyopt import Prodigy` and SILENTLY falls back to AdamW on
ImportError (only a WARNING line `[Side-Step] prodigyopt not installed -- falling back to
AdamW`; the success path is a suppressed INFO). The engine is a uv project, so install from
the ENGINE dir (verified 2026-06-08 via ACESTEP-ENGINE_AUTO_INSTALL.bat = uv-managed):
```
cd E:\AI\MusicGen\AceStep\ACE-Step-1.5
uv add "prodigyopt>=1.1.2"      # or:  & "<DEST>\.uvbin\uv.exe" add "prodigyopt>=1.1.2"
```
Verify (from the ENGINE dir, not the parent): `uv run python -c "import prodigyopt"` returns
no ImportError. adamw8bit needs `bitsandbytes`, adafactor needs `transformers` - same pattern.

## Deploy (box) - HELD until user finishes testing + the smoke test passes
Copy both files over (after the train_api_service.py diff check above), then USER OS-restarts
`run_acestep_api.bat` ([[engine-restart-is-user-only]]). Reverted by any engine `git pull` -
re-apply ([[engine-patches]]).

## Known fix applied during bring-up (2026-06-08)
The in-process path skips the CLI's `auto` device/precision resolution, so `TrainingConfigV2`
defaulted `device="auto"` -> `torch.device("auto")` threw at `trainer_fixed.py:126`. Fixed in
`train_api_lokr_v2_start_route.py`: pass concrete `device=str(handler.device)` + precision.

## DEPLOY-TIME VERIFICATION (do BEFORE any real multi-hour run)
The route file is GitHub-main-derived; these are isolated so a mismatch returns HTTP 500
with the exception (fails at request time, never wastes a GPU run). Confirm against the
box's installed engine:
- [V1] `from acestep.training_v2.configs import LoKRConfigV2, TrainingConfigV2` resolves.
- [V2] `from acestep.training_v2.trainer_fixed import FixedLoRATrainer` resolves.
- [V3] LoKRConfigV2 / TrainingConfigV2 accept the constructor field names in
  `_build_v2_configs` (esp. `gradient_accumulation_steps`, `max_epochs`,
  `save_every_n_epochs`, `dataset_dir`, `optimizer_type`, `scheduler_type`, `cfg_ratio`,
  `attention_type`). If a name differs, fix `_build_v2_configs` only.
- [V4] `FixedLoRATrainer(model, adapter_cfg, train_cfg)` + `.train()` generator yields
  objects with `.step/.loss/.msg/.epoch`.
- [V5] **Tensor compatibility (the showstopper):** v2 reads via `acestep.training.data_module`
  (same as v1) so v1 `tensors/` SHOULD load. If the smoke run errors at data load (missing
  key, e.g. `context_latents`), re-preprocess once via v2
  (`python -m acestep.training_v2.cli.train_fixed --preprocess --dataset-json <dataset.json>
  --tensor-output <new tensors dir>`) and point `tensor_dir` there.

### Smoke test (cheap, catches V1-V5 before a long run)
A 2-epoch run on the existing tensors:
```
curl -s -X POST http://192.168.1.201:8001/v1/training/start_lokr_v2 \
  -H 'Content-Type: application/json' -d '{
    "tensor_dir":"E:\\AI\\MusicGen\\AceStep\\ACE-Step-1.5\\lora_data\\crucible_nightwish_tarja\\tensors",
    "output_dir":"E:\\AI\\MusicGen\\AceStep\\ACE-Step-1.5\\lora_data\\crucible_nightwish_tarja\\train_v2_smoke",
    "lokr_factor":8,"lokr_linear_dim":64,"lokr_linear_alpha":64,"lokr_weight_decompose":false,
    "learning_rate":0.001,"train_epochs":2,"save_every_n_epochs":1,
    "optimizer_type":"prodigy","scheduler_type":"constant","cfg_ratio":0.1,
    "target_modules":["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]}'
```
Then poll `/v1/training/status`. PASS = it reaches "Epoch 1/2" + writes `train_v2_smoke/.../lokr_weights.safetensors`.
NOTE: Prodigy wants `lr` semantics ~1.0 (it auto-estimates). For a first Prodigy run set
`learning_rate: 1.0` + `scheduler_type: constant` (METAL_LORA_PLAN §13g optimizer notes);
the 0.001 above is just to exercise the path.

## Mac side (shipped in this commit, NOT active until backend restart)
- `backend/acestep_train.py` `train_lokr_v2(...)` -> POSTs `/v1/training/start_lokr_v2`.
- `backend/app.py` `POST /api/lora/train_v2` -> builds body + per-run dir + history poller,
  same as `/api/lora/train`. Needs a backend restart to activate (do it AFTER the current
  250ep run so the poller isn't interrupted).
- Existing `/api/lora/train/status` + best-history poller work unchanged (status keeps the
  "Epoch N/M" shape). v2 may not emit v1-style val/best messages -> the val curve may show
  a gap; we judge by ear anyway ([[clap-scoring-unproven]]).
