# Engine patch 2026-06-13 - /v1/free on-demand VRAM reclaim

Adds a programmatic GPU-memory free over HTTP so VRAM can be reclaimed mid-session
WITHOUT an OS-level engine restart. Built after source-verifying (on the box) that the
engine already frees VRAM cleanly in-process; the "needs a fresh boot to free VRAM" belief
was overgeneralized. See METAL_LORA_PLAN.md and the corrected memory engine-fresh-boot-for-lora.

## What it does

`POST /v1/free` with body `{"free_dit": true, "free_llm": true}` (both default true):
- `free_llm`: `app.state.llm_handler.unload()` (best-effort vLLM teardown).
- `free_dit`: nulls `handler.model/vae/text_encoder/text_tokenizer/config/silence_latent`
  (mirrors the loader's own free idiom), and drops any post-run trainer ref from
  `training_state` so `del` actually releases the model.
- Then `gc.collect()` + `torch.cuda.empty_cache()` + `torch.cuda.synchronize()`.
- Returns `{freed, vram_before, vram_after, reclaimed_gb, note}` (allocator stats prove it engaged).
- Refuses with HTTP 409 while training is in progress.

Reload the serving model afterward with `POST /v1/reinitialize` (reloads from
`handler.last_init_params` whenever `handler.model is None`) - no OS restart.

CAVEAT: vLLM may retain a residual pool a Python-level unload cannot fully reclaim
(`unload()` skips `destroy_model_parallel`). For a guaranteed 100% LM teardown an OS restart
is still the only certainty. The DiT/VAE/text-encoder path frees cleanly.

## Files (full patched copies, per [[ship-full-patched-files]])

- `train_api_free_route.py` - NEW file -> `<engine>/acestep/api/train_api_free_route.py`
- `train_api_service.py` - full patched copy of `<engine>/acestep/api/train_api_service.py`
  (adds the import + `register_free_route(...)` call right after the estimate registration).

## Deploy (already done on the box 2026-06-13 via the :5080 fs API)

Written via `fs/write` + verified via `fs/pycompile`. Reverts on any engine `git pull`
(third-party tree) - re-apply with the two fs/write calls. The route is INERT until the
USER restarts `run_acestep_api.bat` (engine restart is user-only).

## Smoke test (after the engine restart)

```
curl -s http://192.168.1.201:8001/openapi.json | grep -o '/v1/free'   # route present
curl -s -X POST http://192.168.1.201:8001/v1/free -H 'Content-Type: application/json' \
     -d '{"free_dit":true,"free_llm":true}'
# expect reclaimed_gb > 0 and freed listing DiT/VAE/TextEncoder (+ LLM if it was loaded)
curl -s -X POST http://192.168.1.201:8001/v1/reinitialize   # serving model back
```
