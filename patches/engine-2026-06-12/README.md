# Engine patch 2026-06-12 - Fisher / gradient-sensitivity estimation over HTTP

Wires the engine's existing-but-unused `training_v2/estimate.py::run_estimation` (gradient-norm
per-module sensitivity = empirical-Fisher proxy) into an HTTP route, so the targeted-capacity
("put rank where THIS artist lives") lever from METAL_LORA_PLAN §13p is usable from our pipeline.
Deployed live via the box fs API (fs/write + fs/pycompile) 2026-06-12; all three compile clean.

## Files (deploy to the box engine tree)
- `train_api_estimate_route.py` (NEW) -> `<engine>/acestep/api/train_api_estimate_route.py`
  - `POST /v1/training/estimate` {dataset_dir, checkpoint_dir="E:/AI/MusicGen/AceStep/checkpoints",
    variant="xl_base", num_batches=15, top_k=64, granularity="module", cfg_ratio=0.1}
    -> {modules:[{module, sensitivity}], ...} ranked desc.
  - SYNC route (FastAPI threadpool); blocks ~1-3 min. run_estimation SELF-LOADS + unloads its own
    decoder, so CALL ON A FRESH BOOT BEFORE /v1/init (a loaded model -> double VRAM -> OOM on 24GB).
- `train_api_service.py` (PATCHED) -> import + `register_estimate_route(...)` in
  `register_training_api_routes` (right before `register_training_dataset_routes`).
- `estimate.py` (PATCHED) -> `_find_attention_modules` now also matches `gate_proj/up_proj/down_proj`
  (MLP/FFN), not just attention q/k/v/o, so the sensitivity pool covers what our champion targets.

## Workflow
fresh boot (NO init) -> POST /v1/training/estimate {dataset_dir: <tensors>} -> take the top-K
`modules` (full names like `decoder...q_proj`) -> POST /v1/training/start_lokr_v2 with
target_modules = those names (full-name targeting verified OK in lokr_utils + LyCORIS) -> train.
Cheapest first step = just run estimate as a DIAGNOSTIC to see which modules an artist lights up.

## Deploy / re-apply
Re-copy these 3 files to the box (or re-run the fs/write deploy) + OS-restart the engine. Reverts on
any engine git pull ([[engine-patches]]). Caveat: HONEST EXPECTATIONS in §13p - targeted ~= uniform at
equal budget (modest, mostly diagnostic); unlikely to fix the artist-voice ceiling.

## Verify after restart
`curl .../openapi.json | grep estimate` shows the route; a small run (num_batches=3) returns a
ranked `modules` list with sensitivity floats.
