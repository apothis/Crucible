# Metal LoRA Training — Plan & Working Doc

_Crucible feature: train a custom metal LoRA (or LoKr) for the official ACE-Step engine and use it at generation time. This is the long-term payoff of keeping the engine over ComfyUI (RESEARCH §15, §18)._

Last updated: 2026-05-27. Status: **research complete + verified on the box; implementation starting (Phase 1).**

---

## 1. Why
ACE-Step's weak spot is exactly metal (sustained high-gain rhythm walls, aggressive vocals/mix). No off-the-shelf model or LoRA nails it. A LoRA trained on real metal teaches the **DiT music model** that character — the single highest-ceiling lever we have. We train on **`xl_sft`** to match Crucible's generation default (LoRAs are bound to the base they trained on).

## 2. The big discovery (verified live, 2026-05-27)
The engine on the box (`192.168.1.201:8001`, xl-sft + 4B LM loaded) exposes the **entire data→train→use pipeline over HTTP** — not just training. So Crucible drives it all remotely from the Mac backend; **no Gradio on the box needed.** Verified via `openapi.json` + status probes.

### Verified API contract (live request schemas)
| Endpoint | Purpose | Key body fields (defaults) |
|---|---|---|
| `POST /v1/dataset/scan` | scan a box folder of audio | `audio_dir*`, `dataset_name=my_lora_dataset`, `custom_tag`, `tag_position=replace`, `all_instrumental=true` |
| `POST /v1/dataset/auto_label[_async]` | caption via the box LM | `skip_metas`, `format_lyrics`, `transcribe_lyrics`, `only_unlabeled`, `lm_model_path`, `save_path`, `chunk_size=16`, `batch_size=1` |
| `GET /v1/dataset/auto_label_status[/{task_id}]` | label progress | — |
| `GET/PUT /v1/dataset/sample/{idx}`, `GET /v1/dataset/samples` | review/edit entries | — |
| `POST /v1/dataset/save` | write dataset JSON | `save_path*`, `dataset_name`, `custom_tag`, `tag_position`, `all_instrumental`, `genre_ratio(0–100)` |
| `POST /v1/dataset/preprocess[_async]` | encode audio→latents (GPU) | `output_dir*`, `skip_existing=false` |
| `GET /v1/dataset/preprocess_status[/{task_id}]` | preprocess progress | — |
| `POST /v1/training/start` | train LoRA | `tensor_dir*`, `lora_rank=64`, `lora_alpha=128`, `lora_dropout=0.1`, `learning_rate=1e-4`, `train_epochs=10`⚠, `train_batch_size=1`, `gradient_accumulation=4`, `save_every_n_epochs=5`, `training_shift=3.0`, `training_seed=42`, `lora_output_dir`, `use_fp8=false`, `gradient_checkpointing=false` |
| `POST /v1/training/start_lokr` | train LoKr (~10× faster) | `tensor_dir*`, `lokr_linear_dim=64`, `lokr_linear_alpha=128`, `lokr_factor=-1(auto)`, `lokr_weight_decompose=true(DoRA)`, `learning_rate=0.03`, `train_epochs=500`, … shared fields |
| `GET /v1/training/status` | live status + loss history | → `{is_training, status, current_epoch, current_step, current_loss, loss_history[], steps_per_second, estimated_time_remaining, tensorboard_*}` |
| `POST /v1/training/stop` | stop training | — |
| `POST /v1/training/export` | export adapter file | `export_path*`, `lora_output_dir*` |
| `POST /v1/lora/load` | load adapter for inference | `lora_path*`, `adapter_name?` |
| `POST /v1/lora/scale` | set adapter strength | `scale*`, `adapter_name?` |
| `POST /v1/lora/toggle` | enable/disable | `use_lora*` |
| `POST /v1/lora/unload`, `GET /v1/lora/status` | unload / state | → `{lora_loaded, use_lora, lora_scale, adapters[], scales{}, active_adapter}` |

⚠ `train_epochs=10` is a placeholder default — use **500–800** (tutorial: ~100 songs→500, 10–20 songs→800).

**Loading sequence quirk:** auto_label needs the **LM loaded**; preprocess/train need the **DiT + decoder** (the tutorial restarts Gradio between stages to free VRAM). Over the API the equivalent is `/v1/init` (with LM) for labeling → `/v1/reinitialize` (DiT+decoder, no LM) before preprocess/train. Training rejects if no decoder. Confirm the exact init payloads on first run.

## 3. Hardware / VRAM (verify empirically on the 3090)
16 GB min / 20 GB rec / ~17 GB typical (tutorial; not split by 2B vs XL/4B). 3090 = 24 GB → fits, but **XL/4B headroom + epoch-time are the empirical unknowns.** Levers: **LoKr** (≈5 min vs ≈1 h), `gradient_checkpointing`, `use_fp8`, smaller rank. **Training is GPU-exclusive** — free/stop ComfyUI/RVC/SoulX/RoFormer (our `free_gpu` pattern) and never generate concurrently.

## 4. LoRA vs LoKr (which to use)
Same preprocessed tensors feed both; switching is cheap.
- **LoKr** — Kronecker decomposition + DoRA; **~10× faster** (5-min cycles), smaller files, comparable/better quality, lr 0.03. **Use for the experimentation phase** (iterate dataset/captions/epochs fast). Some loader limitations.
- **LoRA** — classic low-rank; universal compatibility, lr 1e-4. **Use for the final/portable adapter** or if LoKr quality disappoints.
- **Recommendation: start with LoKr.**

## 5. Data pipeline (what each track needs)
Per track, co-located in a folder **on the box**: audio (`.mp3/.wav/.flac/.ogg/.opus`) + `{name}.lyrics.txt` + `{name}.json` (`caption`, `bpm`, `keyscale`, `timesignature`, `language`). Instrumental sets: `all_instrumental=true`, no lyrics.

**Labeling — reuse what Crucible already has (don't hand-label):**
- **Lyrics** → **online lyrics DB first, whisper fallback** (`backend/lyrics_fetch.py`, built 2026-05-28): for a *known* song, **LRCLIB** (lrclib.net — free, no key, community DB, good metal coverage, returns `plainLyrics`; `/api/get` exact → `/api/search` fuzzy) then **lyrics.ovh**; if not found, **faster-whisper** (`asr.py`). Whisper is hit-and-miss on accented/screamed metal, so DB lyrics are far better when available. Artist/title resolved from explicit args → embedded tags (**mutagen**) → "Artist - Title" filename. Lyrics still hand-correctable in the review step; `lyrics_source` is surfaced so whisper-sourced ones can be flagged for extra review. _Personal-use: lyrics are training labels, not redistributed._
- **BPM/Key** → our **librosa** (`sections.py`/`/api/beats`) or the **box analyze service** (§17a). NOT the LM (hallucinates).
- **Caption** → the box **`/v1/dataset/auto_label`** (5Hz-LM) is the cheapest; align vocabulary to `backend/genres.py` + the §17a CLAP metal vocab.

**Dataset size:** a few dozen well-labeled metal tracks the **user owns/generates** is enough (13→500 epochs in the demo). Rights matter — own works only.

## 6. The transfer problem (key design decision)
All box endpoints take **box-side paths** (`audio_dir`, `tensor_dir`, `lora_path`…). The dataset audio must live **on the box**, but Crucible's backend + library are on the **Mac**. Options:
1. **Tiny box-side upload helper** (our `*_server.py` + `*_AUTO_INSTALL.bat` pattern, e.g. `lora_upload_server.py`): POST files → writes them into a dataset folder the engine can scan. Cleanest full integration. _(preferred)_
2. **LAN share / manual copy:** user drops audio into a box folder; Crucible just points the engine at it. Zero-build fallback for the first real train.
3. **Reuse ComfyUI input dir** (`comfy_input_dir`) as a staging area — already wired for transfers, but semantically odd.

**DECISION (2026-05-27): option (1), the box upload helper — built.** `backend/lora_upload_server.py` (box, :5080, no GPU) + `LORA-UPLOAD_AUTO_INSTALL.bat` + Mac client `backend/lora_upload_py.py` + `lora_upload_host` config. UX: a Crucible file picker → Crucible enriches each track (faster-whisper lyrics + librosa bpm/key + json) → uploads the bundle → the helper writes `{base}/{dataset}/{data,tensors,adapter}` and returns those box paths for `/v1/dataset/scan` → preprocess → train → export. Smoke-tested locally on the Mac (health/new/upload/list all pass). The installer prompts for a **dataset root** (the server's `MG_LORA_DIR`) so datasets/tensors/adapters can live on a big/separate drive, not under the install folder. _To activate: run the installer on the box (pick install dir + dataset root), start `run_lora_upload.bat`, set `lora_upload_host`._

## 7. Phased plan

> **Build-order note:** phases are numbered by pipeline position, NOT the order built. Phase 3 (upload helper) was built before Phase 2 because it was requested directly. Status is marked per phase below.
- **Phase 0 — Research + verify box** ✅ (RESEARCH §18; routes live-verified).
- **Phase 1 — Box-driver client (Mac, no GPU)** ◀ _starting._ Extend `backend/acestep_py.py` (or new `acestep_train.py`) with thin helpers for every endpoint in §2 + poller for the `_status` endpoints. Verify against live `GET …/status`.
- **Phase 2 — Dataset builder (Mac, no GPU)** ✅ _built._ `backend/lora_dataset.py`: `detect_bpm_key` (librosa beat-track + Krumhansl–Schmuckler key → valid `comfy.KEYS` string), `transcribe_lyrics` (faster-whisper via `asr.py`), `build_labels`, and `bundle_for_track(audio_bytes, filename, …)` → `[(name.ext,bytes),(name.json,bytes),(name.lyrics.txt,bytes)]` ready for `lora_upload_py.upload()`. Verified on a real track (bpm 161 / E major / lyrics + bundle). Caption left to the box LM / review UI. (Optional vocal-isolation-before-whisper is a Phase 4 toggle.)
- **Phase 3 — Transfer (box helper §6.1)** ✅ _built (needs install on the box)._ `lora_upload_server.py` (:5080) + installer + Mac client + config; smoke-tested locally.
- **Phase 4 — Orchestration + UI.** Backend `/api/lora/*` endpoints chaining scan→auto_label→save→preprocess→train→export→status; a **Training tab**: pick files → **review & correct labels** (REQUIRED step — whisper mis-hears metal vocals, captions are rough; edit lyrics/caption/bpm/key per track via the engine's `GET/PUT /v1/dataset/sample/{idx}`) → LoRA/LoKr + params → progress + loss curve → export → register. Serialize the GPU (free others first).
- **Phase 5 — Inference integration.** `/v1/lora/load|scale|toggle|unload` wired into generate; a **"Metal LoRA" toggle + strength slider** in the engine tuning UI; adapters registered in config and shown in the ACE header chip.
- **Phase 6 — First real train (GPU, flag first).** LoKr smoke-test on a handful of tracks → measure VRAM/epoch-time on the 3090 → then a full metal LoRA. Heavy GPU run → confirm with the user before kicking off.
- **Phase 7 — Evaluation/iteration loop.** A/B the trained adapter: same prompt with LoRA off vs on at a few strengths, judge by ear, then decide add-data / change-epochs / retrain. Closes the quality loop (user works by ear).

## 8. Risks / open questions
- XL/4B **training VRAM + epoch time** on the 3090 (empirical; Phase 6).
- Exact `/v1/init` vs `/v1/reinitialize` payloads for the label→preprocess VRAM swap (confirm on first run).
- `export` output format/filename + how `lora/load` expects the path (per-run check).
- Long songs may OOM in preprocess (tutorial note) — may need to cap/segment clip length.
- Dataset transfer size (full songs × dozens) over LAN — fine, but note it.
- Adapter is bound to `xl_sft`; if the default model ever changes, retrain.

## 9. Sources
RESEARCH §18 (+ its sources): ACE-Step-1.5 `docs/en/LoRA_Training_Tutorial.md`, `train.py`, `acestep/training_v2/cli/args.py`, `acestep/api/train_api_models.py`, training/lora route files, `scripts/lora_data_prepare/`, Side-Step toolkit. Live verification: `192.168.1.201:8001/openapi.json` + status probes (2026-05-27).
