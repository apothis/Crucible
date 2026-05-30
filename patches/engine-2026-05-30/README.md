# ACE-Step engine patches — 2026-05-30 (cumulative)

**This batch supersedes `engine-2026-05-29/`.** Copy these 5 files; don't
re-apply the earlier batch separately.

Includes everything from 2026-05-29 (target_modules narrowing, val_split
exposure, LoKr val-loop + best-checkpoint save) **plus**:

- **NEW Patch 7: opt-in continuous timestep sampling** (METAL_LORA_PLAN §12).
  Adds `timestep_sampling_mode: "discrete" | "continuous"` field to the
  LoKr training request. **Default = `"discrete"`** — preserves all prior
  behavior. `"continuous"` is the documented opt-in alternative per
  [[optional-additions]] / METAL_LORA_PLAN §12.NO-REGRESSION.

## File-to-destination map

| Patch file | Destination on the box |
| --- | --- |
| `lokr_utils.py` | `<engine>\acestep\training\lokr_utils.py` |
| `trainer.py` | `<engine>\acestep\training\trainer.py` |
| `configs.py` | `<engine>\acestep\training\configs.py` |
| `train_api_models.py` | `<engine>\acestep\api\train_api_models.py` |
| `train_api_lokr_start_route.py` | `<engine>\acestep\api\train_api_lokr_start_route.py` |

Engine root: `E:\AI\MusicGen\AceStep\ACE-Step-1.5\` (per
`ACESTEP-ENGINE_AUTO_INSTALL.bat`).

After copying: **OS-level restart** of `run_acestep_api.bat` (close
console + relaunch) per [[engine-fresh-boot-for-lora]].

## What Patch 7 changes

### `configs.py` — add config fields to `TrainingConfig`
```python
timestep_sampling_mode: str = "discrete"  # "discrete" | "continuous"
timestep_mu: float = -0.4                  # used when mode=continuous
timestep_sigma: float = 1.0                # used when mode=continuous
```
`__post_init__` validates the mode string. `to_dict` includes the new fields.

### `train_api_models.py` — request schema
`StartLoKRTrainingRequest` gets:
```python
timestep_sampling_mode: str = Field(default="discrete", ...)
```
Default keeps existing callers behavior-identical.

### `train_api_lokr_start_route.py` — plumbing
Passes `request.timestep_sampling_mode` into `TrainingConfig(...)` + adds
it to the `training_state["config"]` dict for visibility on the status
endpoint.

### `trainer.py` — the actual sampling change
Both `PreprocessedLoRAModule.training_step` (LoRA path) and
`PreprocessedLoKRModule.training_step` (LoKr path, what Crucible uses) now
dispatch:

```python
if mode == "continuous":
    from acestep.training_v2.timestep_sampling import sample_timesteps
    t, _ = sample_timesteps(batch_size=bsz, device=self.device,
                             dtype=self.dtype, data_proportion=0.0,
                             timestep_mu=..., timestep_sigma=...,
                             use_meanflow=False)
else:
    t, _ = sample_discrete_timestep(bsz, self.timesteps_tensor)
```

We **import from `acestep/training_v2/timestep_sampling.py`** (already
in-tree, faithful reimplementation of the model's own `sample_t_r()`).
Single source of truth, no drift.

## Verification

After engine restart, `GET /v1/info` should still show the engine up.
Then a request body like:

```json
{
  "tensor_dir": "...",
  ...
  "timestep_sampling_mode": "continuous"
}
```

should be accepted, and `GET /v1/training/status` `.data.config` should
echo `"timestep_sampling_mode": "continuous"`. Default-call body without
the field stays discrete.

## Reversal

Any engine `git pull` reverts these. Re-copy after every engine update.
Catalogued in `[[engine-patches]]` memory + METAL_LORA_PLAN §7a.
