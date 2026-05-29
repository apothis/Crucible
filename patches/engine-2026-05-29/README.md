# ACE-Step engine patches — 2026-05-29

Drop-in replacements for 4 engine source files. Fixes (a) `target_modules`
being silently widened to "full" by LyCORIS, and (b) `val_split` being
hardcoded to 0 → `checkpoints/best/` never written for LoKr runs.

**Engine root on the box** (per `ACESTEP-ENGINE_AUTO_INSTALL.bat`):
`E:\AI\MusicGen\AceStep\ACE-Step-1.5\`

## File-to-destination map

Copy each file in this folder over its counterpart under `<engine-root>\acestep\`:

| Patch file | Destination on the box |
| --- | --- |
| `lokr_utils.py` | `<engine>\acestep\training\lokr_utils.py` |
| `trainer.py` | `<engine>\acestep\training\trainer.py` |
| `train_api_models.py` | `<engine>\acestep\api\train_api_models.py` |
| `train_api_lokr_start_route.py` | `<engine>\acestep\api\train_api_lokr_start_route.py` |

After copying: **restart** `run_acestep_api.bat` (close the console window
and relaunch). The patches need a fresh Python import + a fresh decoder boot
anyway, per [[engine-fresh-boot-for-lora]].

## What changes

### `lokr_utils.py` — target_modules now actually filters wrapping
Pass `preset={target_module, target_name, unet_target_name}` directly into
`create_lycoris(...)` so LyCORIS doesn't overwrite it with its default
`PRESET["full"]`. Effect: time_embed / embed_tokens / norm layers no longer
get wrapped at all → smaller adapter files, less VRAM, fewer all-zero w2
tensors in the saved safetensors.

### `train_api_models.py` — `val_split` field added
Adds `val_split: float = 0.0` to `StartLoKRTrainingRequest`. Default keeps
current behavior (no val). Mac route sends 0.1 explicitly per
`backend/app.py:lora_train` defaults (next commit).

### `train_api_lokr_start_route.py` — pass val_split into `TrainingConfig`
One line: `val_split=request.val_split` on the `TrainingConfig(...)` call.
Plus exposes it on the `training_state["config"]` dict so the Mac progress
UI can read it.

### `trainer.py` — validation loop + best-checkpoint save for LoKr
Two changes inside the LoKr-only code:
- `PreprocessedLoKRModule.training_step(batch, record_loss=True)`: now
  accepts `record_loss` like the LoRA module so val passes don't pollute
  `training_losses`.
- `LoKRTrainer._train_with_fabric`: fetches `val_loader` from the
  data_module, tracks `best_val_loss`, runs a per-epoch eval pass when
  `val_split > 0`, and saves `checkpoints/best/` via
  `save_lokr_training_checkpoint` (LoKr format, with optimizer + scheduler
  state for resume).

When `val_split=0` (default), the whole block is inert — no behavior change.

## Verification after applying

After the engine restart:

```bash
# From the Mac
curl -s http://<box>:8001/v1/training/status | jq
# should still report idle / config: {}
```

Then kick a short LoKr run with `val_split=0.1` from the Train LoRA tab and
verify on the box:

```
<output_dir>/checkpoints/best/lokr_weights.safetensors
<output_dir>/checkpoints/best/training_state.pt
```

both exist by the end of training (or earlier — best is overwritten each
time val_loss improves).

## Reversal

Any engine `git pull` reverts these. Re-copy after every engine update.
Cataloged in `[[engine-patches]]` memory + METAL_LORA_PLAN §7a.
