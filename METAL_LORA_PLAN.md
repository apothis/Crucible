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
- **Caption** → **`backend/caption_fetch.py`** (built 2026-05-28) layers sources, all optional with graceful degradation:
  1. **MusicBrainz** (free, no key) — recording tags + artist disambiguation ("english heavy metal band") for known artist/title. Verified live: Iron Maiden / The Trooper → `heavy metal, metal, classic metal, rock, hard rock`.
  2. **Last.fm** (free key, opt-in) — `track.getTopTags` richer subgenre tags. Set `lastfm_key` in `app_config.json`. Get one at https://www.last.fm/api/account/create.
  3. **CLAP** via the box `analyze_host` service (§17a) — audio-based tags, works for unknown songs. Reuses what's already deployed.
  4. **AcoustID + fpcalc** (free key + 1.2 MB binary, opt-in) — audio fingerprint → identify artist/title for *untagged* files, then loops back into (1)/(2). `acoustid_key` config + `brew install chromaprint`. Get a key at https://acoustid.org/api-key.
  5. **Engine `/v1/dataset/auto_label`** — the box LM, exposed as a button in Training-tab Step 2 for the "fill in the blanks" pass.
  Results get merged + de-duped + sorted with specific subgenres first; sources tagged per track in the Step 1 table so you know what to trust. **Then you edit in the review step** (caption is what makes the LoRA work — §10.4).

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
- **Phase 4 — Orchestration + UI** 🟡 _mostly built._ ✅ Backend `/api/lora/*` (4a): status preflight, dataset/add (Mac enrich+upload), scan, samples GET + sample PUT, autolabel, save, preprocess, train (LoRA/LoKr), status/stop, export(+load), inference load/scale/toggle/unload. ✅ Training tab (4b, `web/src/LoraTraining.tsx`, under **Lab**): explainer + live box-service preflight + 4 guided steps (add tracks w/ auto-label + lyrics-source badges; load+autocaption; train w/ params + live loss; export+load). ✅ **Review/edit table built** — Step 2 scans → lists samples → per-track editable caption + lyrics (whisper-flagged) + Save → `PUT sample/{idx}` (engine `UpdateSampleRequest` fields). ⬜ Box-dependent actions unrun (helper not installed + GPU) — review table needs live-schema confirmation on first real dataset. Serialize the GPU (free others first).
- **Phase 5 — Inference integration** ✅ _built (UI)._ `MetalLoraControl` on the Generate form (always visible when `cfg.lora_train`, not expert-gated): when an adapter is loaded → toggle (`/api/lora/toggle`) + strength slider (`/api/lora/scale`, commits on release); when none → a hint pointing to Lab → Train LoRA. Engine-global (applies to the next generation). Verified: renders the "none loaded" hint live. ⬜ Live toggle/scale needs a loaded adapter; header ACE chip could later show the active adapter.
- **Phase 6 — First real train (GPU, flag first).** LoKr smoke-test on a handful of tracks → measure VRAM/epoch-time on the 3090 → then a full metal LoRA. Heavy GPU run → confirm with the user before kicking off.
- **Phase 7 — Evaluation/iteration loop.** A/B the trained adapter: same prompt with LoRA off vs on at a few strengths, judge by ear, then decide add-data / change-epochs / retrain. Closes the quality loop (user works by ear).

## 7a. Engine patches required on the box (re-apply on every engine update)

Like the DCW patch (HANDOFF), the box-side `ACE-Step-1.5` source has bugs we work around with one-line patches. These get reverted by any `git pull` (i.e. re-running `ACESTEP-ENGINE_AUTO_INSTALL.bat`).

| Patch | File | Fix | Why |
|---|---|---|---|
| DCW off (XL gen) | `acestep/inference.py` ~L148 | `dcw_enabled = False` | XL text2music garbled with DCW=True (HANDOFF) |
| Auto-label kwargs | `acestep/training/dataset_builder_modules/label_all.py` ~L17 | Add `**_ignored,` to `label_all_samples` signature before `)` | Engine route refactor (commit `d19c2f3`, 2026-03-05) passes `chunk_size=…, batch_size=…` to a method that doesn't accept them; method last updated 2026-02-06. Verified live: probe failed with `LabelAllMixin.label_all_samples() got an unexpected keyword argument 'chunk_size'`. Pydantic fills the default (16) when the body omits it, so we can't fix this client-side. The `**_ignored` accept-and-discards the bogus kwargs. |
| LoKr target_modules preset | `acestep/training/lokr_utils.py` ~L72 | Pass `preset=lokr_preset` to `create_lycoris(...)` (both calls) | Without it, LyCORIS's default `preset="full"` overwrites our class-level `apply_preset(...)` → every Linear gets wrapped (incl. time_embed/embed_tokens). Post-hoc `requires_grad=False` filter freezes non-target modules, so trainable set IS correct — but wrapped-then-frozen modules still consume VRAM and ship in the saved safetensors at init state (w2 = zeros). Verified 2026-05-29 against upstream `main`. |
| LoKr val_split — request field | `acestep/api/train_api_models.py` ~L52 | Add `val_split: float = Field(default=0.0, ge=0.0, lt=1.0, ...)` to `StartLoKRTrainingRequest` | `TrainingConfig` has `val_split` but the request never exposed it → stuck at 0.0 → no val_loader → `checkpoints/best/` never written. |
| LoKr val_split — route plumbing | `acestep/api/train_api_lokr_start_route.py` ~L84 | Pass `val_split=request.val_split` into `TrainingConfig(...)`; also surface in `training_state["config"]` | Without this the request field is decorative — the dataclass default still wins. |
| LoKr trainer val loop + best ckpt | `acestep/training/trainer.py` (L1363 + L1637 + L1712 + L1888) | (a) `PreprocessedLoKRModule.training_step` accepts `record_loss=True`; (b) `LoKRTrainer._train_with_fabric` fetches `val_loader`, runs per-epoch eval when present, tracks `best_val_loss`, saves `checkpoints/best/` via `save_lokr_training_checkpoint`. | The LoRA trainer has this; the LoKr trainer never did. Even with val_split plumbed, the LoKr fabric loop ignored the val_loader entirely → no best ckpt. ~60 lines, mirrors LoRA's L1097-1127. |
| Patch 7 — continuous timestep sampling toggle (2026-05-30) | `acestep/training/configs.py` (TrainingConfig fields) + `acestep/api/train_api_models.py` (StartLoKRTrainingRequest field) + `acestep/api/train_api_lokr_start_route.py` (plumbing) + `acestep/training/trainer.py` (L508 + L1402) | Adds `timestep_sampling_mode: "discrete" | "continuous"` (default `"discrete"`). When `"continuous"`, trainer imports `acestep.training_v2.timestep_sampling.sample_timesteps` and uses logit-normal sampling (μ=-0.4, σ=1.0) matching the model's original `sample_t_r`. Per [[optional-additions]] + §12.NO-REGRESSION, discrete stays the permanent default. | Tests the hypothesis in §13a.3 that the discrete-8-timestep training distribution doesn't match xl-sft's 32-64 continuous-step inference path, contributing to guitar smear / val-loss plateau. |

Patched copies live in `patches/engine-2026-05-30/` (cumulative — supersedes `2026-05-29/` set). Drop-in over the box's `<engine>\acestep\...` paths and restart `run_acestep_api.bat`. README in that folder has the file-to-destination map. **Don't upstream as PRs yet** — applying locally first to see if they actually help adapter quality (per [[dont-propagate-guesses]]: only PR what we've measured working).

## 8. Risks / open questions
- XL/4B **training VRAM + epoch time** on the 3090 (empirical; Phase 6).
- Exact `/v1/init` vs `/v1/reinitialize` payloads for the label→preprocess VRAM swap (confirm on first run).
- `export` output format/filename + how `lora/load` expects the path (per-run check).
- Long songs may OOM in preprocess (tutorial note) — may need to cap/segment clip length.
- Dataset transfer size (full songs × dozens) over LAN — fine, but note it.
- Adapter is bound to `xl_sft`; if the default model ever changes, retrain.

## 10. Strategy: scoping LoRAs (broad vs subgenre vs single-band)

_A LoRA's strength comes from focus. Bigger isn't better — narrower, well-scoped sets that capture a specific style usually beat a single "do-everything" LoRA, especially since the engine supports loading multiple adapters and blending them at inference time. Synthesized from the ACE-Step 1.5 docs + community Side-Step guide + general PEFT/LoRA practice. Sources at the end of this section._

### 10.1 The four scopes

**(a) Broad metal LoRA — your baseline sound.** Trained on a varied corpus across subgenres (heavy / power / thrash / doom / melodeath, etc.). Captures the production/mix/feel you generally want from Crucible — guitar character, drum tightness, overall heaviness — without being specific to any subgenre. Generalizes well; doesn't excel anywhere. Best as a low-strength "always on" base layer.

**(b) Subgenre LoRA — the sweet spot.** Trained tightly on one subgenre (e.g. *thrash*, *funeral doom*, *Gothenburg melodeath*, *symphonic power*, *djent*). These styles sit far apart in feature space (tempo, mix bright/dark, vocal delivery, instrumentation density), so they benefit from dedicated adapters. This is where focused LoRAs reliably out-do a generic one.

**(c) Band-cluster LoRA — recognizable era/scene without one-artist targeting.** A small cluster of 3–5 stylistically tight bands (e.g. *Big-4 thrash* = Metallica + Megadeth + Anthrax + Slayer; *NWOBHM* = Maiden + Priest + Saxon; *Gothenburg* = In Flames + Dark Tranquillity + At The Gates). Gets you most of the recognizable-style benefit of a single-band LoRA with materially less overfitting risk and far less ethical fog.

**(d) Single-band LoRA — strongest fidelity, real downsides.** Maximum stylistic capture, BUT two genuine problems:
- **Overfitting / memorization.** With ~10–15 tracks the model can memorize specific riffs/melodies instead of learning *style* — outputs may sound suspiciously close to source tracks rather than "in the style of." Image-LoRA community sees the same pattern: subject LoRAs with tiny sets memorize hard ([Apatero](https://apatero.com/blog/lora-training-parameters-subject-vs-style-guide-2025), [SeaArt](https://docs.seaart.ai/guide-1/3-advanced-guide/3-2-lora-training-advance)).
- **Ethics / copyright.** Training on tracks you legally own for purely personal experimentation is broadly permissible, but a LoRA *named and aimed* at one artist crosses into identity-cloning territory. Recent RIAA-vs-Suno/Udio litigation specifically targets training-on-copyrighted-music products; private personal use is a different category but the closer your LoRA gets to "in the style of X," the more it raises eyebrows. **Don't share/redistribute single-artist LoRAs.**

### 10.2 The recommended pyramid

Start with the broad baseline so something useful is loaded by default, then add focused adapters as needs arise. Numbers below are good starting points; tune to loss-curve behavior on actual runs.

| LoRA | Tracks | Method | Epochs (corrected/Side-Step) | Epochs (engine built-in) | Notes |
|---|---|---|---|---|---|
| Broad metal | 50–100 | LoKr / LoRA | ~50–100 | ~500 | wide variety; the everyday baseline |
| Subgenre | 20–40 | **LoKr (fast iteration)** | ~100–200 | ~500–700 | the sweet spot — narrow & focused |
| Band-cluster (3–5 bands) | 15–30 | LoKr / LoRA | ~150–250 | ~700–800 | recognizable era/scene |
| Single band | 10–15 | LoRA, low rank | ~200–500 | ~800+ | accept overfit risk; keep private |

**Why two epoch columns?** ACE-Step's built-in trainer and the community Side-Step trainer use different timestep-sampling schemes. Side-Step's **corrected mode** (continuous logit-normal + 15% CFG dropout) converges roughly an order of magnitude faster than the engine's default — *"1–10 songs: 200–500 epochs · 10–50: 100–200 · 50+: 50–100"* ([Side-Step Training Guide](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/sidestep/Training%20Guide.md)). The engine tutorial's *"~100 songs → 500 epochs · 10–20 → 800"* is the built-in numbers ([LoRA tutorial](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/LoRA_Training_Tutorial.md)). Crucible's HTTP path currently uses the built-in.

### 10.3 Adapter stacking — the killer feature

The engine exposes `/v1/lora/load` (with `adapter_name`), `/v1/lora/scale` (per-adapter `scale`), and the status response carries `adapters: []` + `scales: {}` — i.e. multi-adapter is a first-class concept, not a hack. Underneath, ACE-Step is built on PEFT/diffusers, where loaded adapters are blended by **weighted concatenation of their low-rank matrices**, weights set by `set_adapters(["a","b","c"], adapter_weights=[wa,wb,wc])` ([HF: Merge LoRAs](https://huggingface.co/docs/diffusers/main/en/using-diffusers/merge_loras)).

What that gives you in Crucible:
- **Layered styles.** Load `metal_broad@0.4` + `thrash@0.6` for "metal but thrash-leaning"; swap thrash for `doom@0.7` to flip the mood. Strength sliders let you dial each independently.
- **Genre-blending.** `symphonic@0.5` + `thrash@0.6` = symphonic thrash even without a dedicated LoRA for that subgenre.
- **One small LoRA can patch another.** Train a tight `gang_vocals@0.3` style LoRA and layer it over any base.

Caveats worth knowing:
- **Additive interference.** Diffusers' simple `set_adapters` does weighted-sum stacking → too many strong adapters cause oversaturation/conflicts; sum of effective scales ≲ ~1.0–1.5 is a reasonable safety band. PEFT's smarter merges — **TIES** (trim/sign-resolve/elect) and **DARE** (drop+rescale) — explicitly fix this ([HF PEFT merging](https://huggingface.co/blog/peft_merging)), but they're not exposed via `set_adapters`; if blending issues bite, that's the upstream lever to ask for.
- **Rank-matching only matters for *merging* (fuse_lora), not stacking.** You can run adapters with different ranks side-by-side at inference; they only need identical ranks if you bake them into a single merged file.
- **Personal-use guidance:** the engine has no per-prompt LoRA control (it's engine-global) — the Generate-tab Metal-LoRA toggle/scale we built sets the *current* engine state.

### 10.4 Captioning is what makes the scope work

The single biggest lever after dataset choice. The official tutorial is emphatic: keep the LM-generated *genre* descriptions ratio at 0, captions do the work. The community consensus across image+music LoRAs: **descriptive style captions generalize; band-name / trigger-word captions memorize**.

For metal training data:
- **Do** describe what the LoRA should *learn* to associate with the audio: genre + subgenre + instrumentation + vocal delivery + mix character + tempo feel.  
  e.g. *"melodic death metal, fast tempo, blast beats, twin-guitar harmonies, tremolo picking, growled lead vocals with clean choruses, dense mix, dark/cold production"*
- **Do** keep captions consistent across a subgenre LoRA's tracks — that's what the model latches onto.
- **Don't** put the band name in the caption for a style/subgenre LoRA. That biases toward memorization and makes the LoRA only fire when you also prompt the band name.
- **Don't** mix contradictory genre tags (`ambient, blackened thrash`) — confuses the model ([ACE-Step prompt guide](https://www.ambienceai.com/tutorials/ace-step-music-prompting-guide)).
- **Captions under ~512 chars**; 3–7 well-chosen descriptors beats long lists ([RunComfy ACE-Step training](https://www.runcomfy.com/trainer/ai-toolkit/ace-step-1-5-lora-training)).

In Crucible's flow this maps to: let the box LM Auto-Label generate the first-pass captions, **then edit them in the review step** to add the specific descriptors that define your scope. The review step is doing real work — don't skip it.

### 10.4a Measured numbers (this box: RTX 3090 24 GB, xl-sft 4B, 2026-05-28)

A/B calibration on 6 power-metal tracks, LoKr (rank 64, lr 0.03, batch 1, grad-accum 4):

| | grad_ckpt **ON** | grad_ckpt **OFF** |
|---|---|---|
| Epoch 1 (incl. warmup) | 61 s | 127 s |
| Post-warmup epochs (avg) | **~42 s/epoch** | **~104 s/epoch** |
| 5-epoch total | 227 s | 544 s |

**Surprise: grad_ckpt=ON is ~2.5× FASTER here**, opposite the textbook prior. On this card, the 4B XL decoder backward without checkpointing saturates VRAM bandwidth and the GPU stalls waiting on memory (bandwidth-bound). With checkpointing it's compute-bound and SMs stay hot (audible via the GPU fan during A vs near-silent during B). **Make `gradient_checkpointing=True` the default for XL/4B LoKr training.**

Real-world planning numbers on this rig:
- **50-epoch run ≈ 35 minutes** (epoch 1 ~60 s + 49 × ~42 s)
- **500-epoch run ≈ 5.8 hours**
- Preprocess: ~25 s for 6 tracks → ~4 s/track

**Clean engine state before training matters — bake it in.** The very first attempt was stuck at ~170 s/step for 5 minutes (CPU-bound, GPU idle). Root cause: the engine was in default *inference* state (DiT + LM both loaded for captioning) when training started; Lightning Fabric setup against that saturated state ran slow. The fix is what the tutorial calls "restart and do NOT select the LM model" — via API that's:

```
POST /v1/init {"model": "<current>", "init_llm": false}
```

before `dataset/preprocess_async` or `training/start[_lokr]`. The backend's `/api/lora/dataset/preprocess` and `/api/lora/train` now do this automatically (`_ensure_training_ready`).

### 10.5 Hyperparameter intuition by scope

Same defaults as Side-Step's *"good default rank 64, alpha 128"* mostly hold, with these nudges:

- **Style LoRAs (broad/subgenre)** — *style imprints fast.* Rank **32** is often enough; alpha ≈ rank. LR `1e-4`. Watch for early plateau on the loss curve and stop early — the best checkpoint is usually well before the final.
- **Cluster / single-band** — closer to "subject" training. Rank **64** (the default); alpha = 2×rank (128). Higher epoch counts as in the table; expect more overfit risk.
- **High-quality but small dataset** — drop rank to **16** (Side-Step's "low capacity, very small datasets"); reduces overfit pressure.
- **Loss curve as ground truth.** "Should decrease over time. Spikes are normal but persistent increase means overfitting" ([Side-Step Guide](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/sidestep/Training%20Guide.md)). The default `save_every_n_epochs=5` means you can A/B mid-training checkpoints; **the best one is rarely the last one.**
- **Audio quality matters more than people think.** Train on **WAV/FLAC at ≥44.1 kHz**, not transcoded MP3s; the model learns spectral artifacts otherwise ([RunComfy](https://www.runcomfy.com/trainer/ai-toolkit/ace-step-1-5-lora-training)).
- **LoKr caveat.** ACE-Step's own tutorial loves LoKr (10× speed); the more-conservative Side-Step guide currently marks LoKr "Experimental — may have rough edges." So: **use LoKr to iterate cheaply**, but if a final adapter feels off and you can't tell why, retrain that one as a classic LoRA.

### 10.6 Detecting overfitting (qualitatively, by ear)

The loss curve is necessary-but-not-sufficient. Definitive signs *by ear*:
- **Holdout failure.** Hold back one or two tracks from each set you *didn't* train on. If the LoRA outputs sound nothing like that style without the trained tracks "leaking in," it learned the style. If they don't sound like the style at all, undertrained; if they sound suspiciously like *specific* trained tracks, overtrained.
- **Memorized motifs.** A recognizable riff/melody/lyric phrase from a training track turning up in unrelated prompts ⇒ overfit, pull the LoRA back to an earlier checkpoint.
- **Prompt-insensitivity.** If outputs ignore your prompt and converge to one sound regardless ⇒ the LoRA is steamrolling the base; lower its scale at inference (the strength slider) and/or retrain with lower epochs.
- **Style-strength inversion.** Outputs at LoRA strength 0.2 sound good; at 1.0 they're worse → the LoRA's signal is strong; use it as a *bias*, not a *replacement*.

### 10.7 Concrete starting plan for Crucible

A pragmatic sequence that avoids wasted GPU time:
1. **Smoke test first.** LoKr, 5–8 tracks, low epochs (~50). Goal: confirm VRAM/timing on the 3090 + the whole pipeline runs end-to-end. Throw the adapter away after.
2. **Subgenre v1 (your favorite first).** ~25 tracks, LoKr, ~150 epochs. Review labels carefully. A/B by-ear vs. the base model. This is your real first usable adapter.
3. **Build out 2–3 more subgenre adapters** for the styles you'll actually want to swap between. By this point you'll know your dataset/epoch sweet spot empirically.
4. **Then a broad baseline.** ~60 tracks pulling from across the subgenres, ~75 epochs. Always-loaded at low scale (e.g. 0.3) as a baseline-color layer; stack subgenres on top.
5. **Band-cluster only if a specific era keeps escaping you** even with a good subgenre LoRA. Keep private; don't share.
6. **Skip single-band LoRAs** unless you've genuinely exhausted the cluster approach.

Throughout: caption review is the highest-leverage step. Five extra minutes editing the auto-labels per track is worth more than another 200 epochs.

### 10.8 Sources for §10

- [ACE-Step 1.5 LoRA Training Tutorial](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/LoRA_Training_Tutorial.md) (official; dataset/epoch guidance)
- [Side-Step Training Guide](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/sidestep/Training%20Guide.md) (corrected mode + rank/epoch tables + overfit notes)
- [ACE-Step Tutorials & Best Practices — DeepWiki](https://deepwiki.com/ace-step/ACE-Step-1.5/10-tutorials-and-best-practices)
- [RunComfy: ACE-Step 1.5 LoRA training](https://www.runcomfy.com/trainer/ai-toolkit/ace-step-1-5-lora-training) (audio format + caption length)
- [Apatero: LoRA Training Parameters — Subject vs Style 2025](https://apatero.com/blog/lora-training-parameters-subject-vs-style-guide-2025) (dataset-size / epoch heuristics by training type)
- [SeaArt LoRA Training Advanced Guide](https://docs.seaart.ai/guide-1/3-advanced-guide/3-2-lora-training-advance) (subject vs style epochs/overfitting)
- [Kohya LoRA Training Settings — PropelRC](https://www.propelrc.com/kohya-lora-training-settings-explained/) (rank/alpha guidance)
- [HF Diffusers: Merge LoRAs](https://huggingface.co/docs/diffusers/main/en/using-diffusers/merge_loras) (`set_adapters` + adapter_weights mechanics)
- [HF PEFT: new merging methods (TIES, DARE)](https://huggingface.co/blog/peft_merging) (smarter multi-adapter merges)
- [HF PEFT for inference (LoRA)](https://huggingface.co/docs/diffusers/en/tutorials/using_peft_for_inference)
- [Ambience AI: ACE-Step Prompt Guide](https://www.ambienceai.com/tutorials/ace-step-music-prompting-guide) (caption do's/don'ts)
- [RIAA vs Suno / Udio (2024) — Wikipedia overview](https://en.wikipedia.org/wiki/Suno_(company)) (the legal context for music-AI training data — for the single-band ethics caveat)

## 11. Plan 1 — Improved evaluation (post-hoc fitness, no VRAM risk)

_Status (2026-05-30): authored, not shipped. Tracked as task #10._

### 11.0 Why we need it
Val_loss is necessary but coarse — MSE in latent space, only loosely correlated with perceptual quality. The 2026-05-29/30 overnight 200-ep / 35-track run made this concrete: the "best" val checkpoint (epoch 105, val 0.6919) is statistically tied with epoch 27 (val 0.697) — a 0.005 gap inside a 0.20 val noise band. Picking between them on val_loss alone is essentially a coin flip. We need a perceptual fitness signal that actually correlates with "sounds metal" / "sounds coherent."

### 11.1 Architectural constraint — VRAM safety during training
**In-training generation is unsafe and not part of this plan.** During a training run the engine holds:
- DiT decoder on GPU with LoKr wrappers (gradient flow active)
- AdamW optimizer state (~2× param memory)
- Gradient buffers + activations (reduced by grad ckpt but still present)
- VAE / text encoder offloaded to CPU
- LM unloaded

Adding generation mid-training would require swapping VAE/TE/LM back to GPU, running inference, then unloading again — peak VRAM during this window would be far higher than steady-state training. With our 22–24 GB working set, a single sizing miscalculation = CUDA OOM = whole multi-hour training crashes. Not acceptable as routine behavior.

**Plan 1 routes around this**: everything that runs *during* training is CPU-only weight introspection. Everything *perceptual* runs *after* training against the on-disk checkpoints, with the engine in a clean inference-mode state.

### 11.2 During-training surface (CPU only, zero GPU)
A watcher thread spawned from `/api/lora/train` (parallels the existing best-history poller, commit `9f6b613`). Watches `<output_dir>/checkpoints/` for new `epoch_NN_loss_X.XXXX/` directories.

For each new directory:
1. Load `lokr_weights.safetensors` via the `safetensors` lib in CPU mode (no torch CUDA init).
2. Per LoKr module, compute:
   - **Mean abs weight magnitude** — growth across consecutive checkpoints = still learning; flat = converged.
   - **L2 norm of update vs previous checkpoint** — low + persistent = converged.
   - **`dora_scale` distribution** (if present) — alarm threshold at `abs(max) > 3.0`. Prevents the Run-1 DoRA blowup pattern from going unnoticed.
   - **Singular value spectrum of `w1 @ w2`** — alarm if degenerate (top-1 SV ≫ rest = mode collapse).
3. Train/val gap from existing `loss_history` + `plot_val_loss` — alarm when val climbs N epochs in a row while train drops = overfitting signal.
4. Persist per-checkpoint stats to `library/lora_train_history/<dataset>_weights.json`.

Surface via new `GET /api/lora/train/health?dataset=X` returning `{weight_norm_trend, dora_scale_max, sv_collapse_flag, train_val_gap, overfit_alarm}`. Wire into a small "Training health" panel in the Train LoRA tab.

### 11.3 Post-training surface (heavy, GPU, isolated)
Triggered explicitly by the user after training completes + a clean engine restart per [[engine-fresh-boot-for-lora]].

For each on-disk checkpoint (`epoch_10/`, `epoch_20/`, ..., `checkpoints/best/`, `final/`):
1. **Adapter swap**: unload current, load this checkpoint at fixed scale (default 0.5).
2. **Generation**: N samples with fixed prompts × rotating seeds. Default: 4 power-metal prompts × 3 seeds = 12 takes per checkpoint.
3. **CLAP zero-shot scoring** (box analyze service, port 5075): for each take, get confidence on target tags (`power metal`, `double bass drums`, `harmonized lead guitars`, `soaring clean vocals`, `epic chorus`). Mean + std across the 12 takes.
4. **Centroid distance**: pre-compute once per dataset — CLAP embeddings of all training mp3s → mean = "metal centroid". Per-generation cosine similarity to that centroid.
5. Persist to `library/lora_train_history/<dataset>_fitness.json`.

Cost: ~12 takes × ~80s/take ≈ 16 min/checkpoint. A 200-ep run with save_every_n_epochs=10 = 20 ckpts + best + final = 22 ckpts ≈ 6 h post-eval. Feasible overnight #2.

### 11.4 Aggregation + winner pick
After all per-checkpoint scores are in:
- Plot 5 overlaid curves on epoch axis: train_loss, val_loss, weight_norm, CLAP tag fitness (mean of target tags), centroid distance.
- **Winner**: argmax(CLAP fitness), not argmin(val_loss).
- **Disagreement between curves is itself diagnostic** (e.g. val drops but CLAP fitness flat = adapter is memorising training distribution but not generalising to "metal-ness" as humans recognise it).

### 11.5 Checks along the way (avoid regressions)
- **Check #1A** — verify weight introspection works on last night's checkpoints (epoch_10/.../epoch_200/, best/, final/) before shipping the watcher. Load, compute all stats, sanity-check numbers. If anything blows up, fix before wiring to live training.
- **Check #1B** — single-checkpoint dry run of the post-eval pipeline end-to-end. Pick one ckpt from last night, run §11.3 against it, validate scores complete in expected time (~16 min) and look sensible.
- **Check #1C** — after the first 3 checkpoints of a real eval run, verify fitness scores are monotonic-ish (early ckpts should score lower than mid-training). If they're random noise the CLAP signal isn't sensitive enough — abort the per-ckpt loop, fall back to weight-only, and consider larger N or different prompts.

### 11.6 Deliverables
- `backend/lora_eval.py` — weight introspection (§11.2) + CLAP fitness pipeline (§11.3)
- `backend/app.py` — `GET /api/lora/train/health` (during) + `POST /api/lora/evaluate` (post)
- Watcher thread spawn from `/api/lora/train` (alongside existing history poller)
- UI: "Training health" strip in Train LoRA tab + post-training evaluation card

### 11.7 Stop conditions
- After check #1A fails → abort. Investigate safetensors lib + lokr_weights structure.
- After check #1B takes 3× longer than expected → redesign for fewer takes/ckpt.
- After check #1C → fitness signal isn't useful → ship weight introspection only, defer perceptual to a later round (maybe try CLAP centroid alone, or move to FAD via PANN).

---

## 12. Plan 2 — Improved training (continuous timestep sampling A/B)

_Status (2026-05-30): authored, not shipped. Tracked as task #11. Depends on Plan 1 shipping first._

### 12.NO-REGRESSION — additions only, no removals (per [[optional-additions]])

**The current training stack stays available unchanged.** Plain LoKr + lr=0.01 + discrete-8-timestep sampling has produced usable adapters (6-track Nightwish 150-ep is currently our best-sounding baseline). We are NOT removing it to try the new method. The patches add a *toggleable opt-in alternative*; the default remains the current behaviour. This applies through every decision branch in §12.5 — even a "win" outcome makes the new path available, it doesn't take the old one away.

### 12.0 The hypothesis
**Cited from arxiv 2506.00045 (ACE-Step paper):** the model was originally trained with continuous logit-normal timestep sampling, `μ=0, σ=1, shift=3.0`.

**Read directly from `acestep/training/configs.py:78-82` (the v1 trainer the engine HTTP API uses):** *"Discrete timesteps from turbo shift=3.0 schedule (8 steps). Randomly samples one of 8 timesteps per training step: [1.0, 0.9545, 0.9, 0.8333, 0.75, 0.6429, 0.5, 0.3]"*.

**Read directly from `acestep/training_v2/configs.py:155-163` (the in-tree v2 trainer, CLI-only):** continuous logit-normal with `timestep_mu=-0.4, timestep_sigma=1.0, data_proportion=0.5`.

**Hypothesis (mine, not yet measured on our setup, per [[dont-propagate-guesses]]):** training on the discrete 8-step grid means the adapter optimises corrections at noise levels the 32–64-step xl-sft inference path mostly skips. The distribution mismatch could explain (a) val_loss plateauing despite weights still drifting and (b) the "smeary mix" symptom user has reported. If true, switching to continuous sampling is a structural fix, not a hyperparam tweak.

### 12.1 Pre-flight 2.0 — reference takes (no training)
Before touching the trainer, capture the A side of the planned A/B while the box is in a known-good state:
- 4 fixed power-metal prompts × 3 seeds = **12 reference takes** off `crucible_metal/checkpoints/best/` (epoch 105) at strength 0.5.
- Stored as a dedicated Library project named "Plan 2 baseline". Becomes the comparison point for every subsequent A/B in this section.
- _Why now:_ we have the current best ckpt loaded and known; capturing baseline before patching means there's no risk of accidentally comparing apples-to-oranges later.

### 12.2 Phase 2.1 — author Patch 7
Single-file engine patch in `acestep/training/trainer.py` (~40 lines):
- Locate the `sample_discrete_timestep(bsz, self.timesteps_tensor)` call in `PreprocessedLoRAModule.training_step` and `PreprocessedLoKRModule.training_step` (need to read both before patching — same shape changes).
- Add a `_sample_continuous_timestep(bsz)` helper that draws `ε ~ N(μ, σ²)`, applies `t = sigmoid(ε)`, applies the shift transform — same returned tensor shape/dtype as the discrete path.
- Defaults: `μ=-0.4, σ=1.0` from v2's `recommended.json` (cited, not measured for our flow).
- Add `timestep_sampling_mode: "discrete" | "continuous"` field to `StartLoKRTrainingRequest` and `StartTrainingRequest`. **Default = `"discrete"`** — preserves current behavior. Continuous is opt-in.
- Mac route `/api/lora/train` accepts the new field; `acestep_train.train_lokr` plumbs it through (only sent when caller provides → unpatched engines ignore).
- Drop the patched files into `patches/engine-2026-05-30/` with a README matching the 2026-05-29 batch's format.
- Add a row to METAL_LORA_PLAN §7a and a Patch 7 entry to `[[engine-patches]]` memory.

### 12.3 Phase 2.2 — calibrated 50-epoch A/B run
Same 35 tracks, same lr=0.01 plain LoKr, same gradient_checkpointing=true, same val_split=0.15, same seed=42. **Only** changes vs the overnight 200-ep run:
- `timestep_sampling_mode="continuous"`
- `train_epochs=50` — calibrated from last night's measured data: useful learning happened in the first ~30 epochs, so 50 is enough to see whether continuous sampling is doing something different + leaves headroom. ~50 × 157s ≈ 1h 20m, not overnight. Cheap enough to iterate.
- Engine fresh boot before kicking off (per [[engine-fresh-boot-for-lora]]).

### 12.4 Phase 2.3 — A/B evaluation
After 12.2 training completes:
1. Engine fresh boot for clean inference state.
2. Apply Plan 1's post-training eval pipeline (§11.3) to the new run's checkpoints. **Plan 1 is a hard dependency for Plan 2** — without it we can't quantify "win" vs "tie" beyond listening tests, which are noisy.
3. Generate the same 4 prompts × 3 seeds = 12 takes off:
   - `continuous-50ep/checkpoints/best/`
   - `continuous-50ep/final/`
4. User by-ear A/B vs Phase 2.0 baseline takes.
5. Compare CLAP fitness curves: continuous-50ep vs discrete-200ep.

### 12.5 Check #2 — decision criteria (concrete, before we start)
- **Check #2A — patch sanity**: stand up the patched engine, fire a 3-epoch toy training (save_every=1, val_split=0.15, same 35-track dataset). Confirm: no NaN/Inf loss, training completes, adapter loads cleanly, generation with the toy adapter doesn't garble. If any fail → abort, re-read the paper's training section before continuing.
- **Check #2B — training health during the 50-ep run**: use Plan 1's `/api/lora/train/health` surface (§11.2). Verify: val curve differs in shape vs last night's, no dora_scale or singular-value pathology, train/val gap stable. If the curve looks identical to last night's, the patch isn't actually changing the training distribution — debug before continuing.
- **Check #2C — decision after Phase 2.3**:
  - **Win**: continuous-50ep best ckpt sounds at-least-as-good as discrete-200ep best by ear, AND CLAP fitness curve is comparable or higher. → Add the continuous mode as a **documented opt-in** in the Train LoRA UI and Mac route (NOT the default per §12.NO-REGRESSION). Update [[engine-patches]] memory describing both methods. Run a proper 100-ep production run on the same data using the new mode. _Current discrete-sampling code path stays in place and remains the default for any caller that doesn't explicitly opt in._
  - **Tie**: no audible difference, CLAP curves indistinguishable. → Keep the patch toggleable as opt-in; default stays discrete; flag for revisit on a future dataset (bigger or different subgenre) where divergence might emerge.
  - **Loss**: continuous is measurably worse. → Document the negative result. The toggle stays in the codebase (the patch shipped, the field is harmless when defaulted to discrete) but is documented as "tested, regressed at our scale" rather than recommended. Look at v2's `cfg_ratio` (CFG dropout) and Fisher-info adaptive ranks as the next lever; or shift to Side-Step trial. _Old behaviour is unchanged regardless._

### 12.6 Phase 2.4 — optional follow-ups (only if 2.3 wins)
Each gets its own one-variable A/B vs the post-2.3 baseline:
- **CFG dropout (`cfg_ratio=0.15`)** — classifier-free guidance dropout during training, improves prompt adherence per diffusion literature. ~10-line additional patch.
- **Fisher information adaptive ranks** — measure per-module gradient sensitivity on a held-out subset, allocate higher LoRA rank to high-sensitivity modules. Material implementation work (`training_v2/estimate.py` is non-trivial) but potentially highest-impact for small datasets.
- **Per-variant timestep schedule selection** — different μ/σ for base vs sft vs turbo (Side-Step's "variant-aware" approach).

### 12.7 Risks + mitigations
- **Engine patch surface area growing** — at the time of writing Crucible maintains 5 active patches (DCW, auto-label, lokr_utils, val_split, trainer val-loop). Patch 7 makes 6. Maintenance burden on every engine update. _Mitigated by_ the `patches/` folder convention; single-copy drop-in.
- **Patched trainer behavior different enough to hide a bug** — the discrete-sampling version is "working" right now (we have a usable, if imperfect, adapter). A bad continuous patch could silently regress. _Mitigated by_ default-off toggle + checks #2A and #2B catching breakage before we waste a full training run.
- **CLAP signal isn't perceptive enough to discriminate at our quality level** — covered by check #1C in Plan 1. If CLAP gives us noise, the A/B falls back to listening-only and we lose the quantitative win.
- **Continuous sampling actually requires a different optimizer schedule** — if v2's continuous trainer also tunes warmup_steps / scheduler differently and we don't match, we could be measuring schedule effects, not sampling effects. _Mitigation:_ read v2's full training_step before phase 2.1 to confirm sampling is the ONLY difference we need to port.

### 12.8 Deliverables
- `patches/engine-2026-05-30/trainer.py` — continuous timestep toggle
- `patches/engine-2026-05-30/train_api_models.py` — `timestep_sampling_mode` field
- `patches/engine-2026-05-30/train_api_lokr_start_route.py` — plumb the field into `TrainingConfig` (mirrors the val_split patch from 2026-05-29 batch)
- `patches/engine-2026-05-30/train_api_lora_start_route.py` — same for the LoRA path
- `patches/engine-2026-05-30/README.md` — file-to-destination map
- Mac: `backend/acestep_train.py` accepts `timestep_sampling_mode` kwarg (only sent when provided)
- Mac: `backend/app.py /api/lora/train` accepts + forwards
- METAL_LORA_PLAN §7a — Patch 7 row
- `[[engine-patches]]` memory — Patch 7 entry

---

## 13. 200-epoch / 35-track baseline (measured 2026-05-29 → 2026-05-30 overnight)

_The first run on our setup at this scale. Numbers below are MEASURED on our box, not cited from upstream._

### 13.1 Run summary
- **Hardware**: Windows + RTX 3090, 24 GB VRAM, engine `acestep-v15-xl-sft` + 4B LM stack
- **Dataset**: 35 power-metal tracks (Sabaton, DragonForce, Manowar, Helloween, Rhapsody, Nightwish, Stratovarius, Blind Guardian, HammerFall, Sonata Arctica, Gloryhammer, Avantasia, Edguy, Beast in Black mix), all with online lyrics + LM-merged captions, all preprocessed to latents.
- **Hyperparams**: plain LoKr (no DoRA), lr=0.01, gradient_checkpointing=true, batch_size=1, gradient_accumulation=4, val_split=0.15 (= 5 val samples on 30 train), train_epochs=200, save_every_n_epochs=10, seed=42.
- **Engine patches active**: 2026-05-29 batch (target_modules narrowing, val_split exposure, LoKr val-loop + best-checkpoint save).
- **Wall clock**: 23:14:24 → 07:58:46 = **8h 44m**. **~157 s/epoch**, 1600 steps total.

### 13.2 Best-history (captured by Mac-side poller from commit `9f6b613`)
5 best updates total:

| Epoch | Val loss |
|---|---|
| 1 | 0.8083 |
| 4 | 0.7408 |
| 16 | 0.6995 |
| 27 | 0.6970 |
| **105** | **0.6919** |

### 13.3 Val curve shape (every-10-epoch slice)
```
ep   1: 0.808   ep 110: 0.818
ep  10: 0.757   ep 120: 0.774
ep  20: 0.704   ep 130: 0.851
ep  27: 0.697   (best of first half)
ep  30: 0.798   ep 140: 0.814
ep  40: 0.866   ep 150: 0.774
ep  50: 0.789   ep 160: 0.839
ep  60: 0.793   ep 170: 0.804
ep  70: 0.716   ep 180: 0.788
ep  80: 0.712   ep 190: 0.763
ep  90: 0.896   ep 200: 0.778
ep 100: 0.786   (final)
ep 105: 0.692   (overall best)
```

### 13.4 Calibration takeaways
- **Useful learning happens in the first ~30 epochs.** Val dropped 0.808→0.697 by epoch 27. After that, val oscillates in [0.69, 0.90] with no clear downward trend.
- **The "best at epoch 105" is statistically a tie with epoch 27.** Δ = 0.005 inside a noise band of ~0.20.
- **Future runs at this dataset size: 30–50 epochs is sufficient.** 200 was an overshoot — ~6 of the 8.7 hours was wasted GPU.
- **5-sample val is much better than 1-sample** but still not enough for reliable best-ckpt picking. Plan 1's perceptual fitness signal is the real fix; multi-noise val reduction (val_passes>1) is a cheaper interim.
- **5× per-epoch cost vs 6-track runs is expected** — 8 grad-update steps/epoch vs 2 + 5-sample val vs 1. Matches the ~4-5× scaling.

### 13.5 A/B listening artifacts (fired 2026-05-30 morning)
4 takes captured in the Library under the "Generate" section, same prompt + lyrics + seed=42:
- "Crucible 35-track 200ep — BEST @ 0.3"
- "Crucible 35-track 200ep — BEST @ 0.5"
- "Crucible 35-track 200ep — FINAL @ 0.3"
- "Crucible 35-track 200ep — FINAL @ 0.5"

User listening feedback to be appended here.

---

## 13a. Research catalog — training paths, adapter algos, quality metrics (2026-05-30 deep dive)

_Surfaced here so a cold session can read this doc + HANDOFF and pick up where we left off. Numbers cited from upstream / repo source are labelled. Hypotheses are labelled. Measured-on-our-box findings are labelled. Per [[dont-propagate-guesses]]._

### 13a.1 The training paths that exist (verified by reading the repo 2026-05-30)

**Path A — `acestep/training/` HTTP API** (what Crucible uses today)
- Exposed via `POST /v1/training/start_lokr` and `POST /v1/training/start` on the engine
- We drive it from `backend/acestep_train.py` → Mac `/api/lora/train` route
- Config defaults read from `acestep/training/configs.py`: `learning_rate=0.03` for LoKr, `max_epochs=100`, `warmup_steps=100`, `mixed_precision="bf16"`, `val_split=0.0` (we patched to expose it)
- Timestep sampling: **discrete, 8-element turbo grid** `[1.0, 0.9545, 0.9, 0.8333, 0.75, 0.6429, 0.5, 0.3]` per the configs.py docstring at lines 78-82
- Loss objective: MSE on flow-matching prediction in latent space (`F.mse_loss(decoder_outputs[0], flow)` in `trainer.py:1410`)
- Crucible patches active: target_modules narrowing, val_split exposure, val-loop + best-ckpt save for LoKr path

**Path B — `acestep/training_v2/` CLI** (in-tree, not exposed over HTTP)
- Standalone Python CLI: `python -m acestep.training_v2.cli.train_fixed` or `train_vanilla`
- 7 preset configs ship in `acestep/training_v2/presets/`: `quick_test.json`, `recommended.json`, `high_quality.json`, `vram_8gb.json`, `vram_12gb.json`, `vram_16gb.json`, `vram_24gb_plus.json`
- Key v1→v2 deltas (read from `acestep/training_v2/configs.py` 2026-05-30):
  - **Continuous logit-normal timestep sampling** with `timestep_mu=-0.4, timestep_sigma=1.0, data_proportion=0.5`
  - **CFG dropout** (`cfg_ratio=0.15`) — classifier-free guidance dropout during training
  - **Explicit `attention_type`** field (`self`/`cross`/`both`) — vs v1's implicit-via-target_modules
  - **Optimiser choice**: adamw, adamw8bit, adafactor, prodigy (vs v1's adamw-only)
  - **Scheduler choice**: cosine, cosine_restarts, linear, constant, constant_with_warmup
  - **VRAM profile auto-detection**: `vram_profile: auto|comfortable|standard|tight|minimal`
  - **Sample generation during training**: `sample_every_n_epochs` (we don't use — VRAM risk per §11.1)
  - **Fisher Information per-module gradient estimation**: `acestep/training_v2/estimate.py` writes a `fisher_map.json` that subsequent training auto-detects and uses to allocate higher rank to high-sensitivity modules
  - **First-class checkpoint resume** via `resume_from`
- _recommended.json_ (cited as v2 default starting point):
  ```
  rank=64, alpha=128, dropout=0.1, target=q_proj k_proj v_proj o_proj,
  attention_type=both, learning_rate=1e-4, batch_size=1, gradient_accumulation=4,
  epochs=100, save_every=10, gradient_checkpointing=true, cfg_ratio=0.15
  ```
- _high_quality.json_: rank=128, alpha=256, epochs=1000, save_every=250. So upstream's "high quality" recipe is **10×** our 100-ep runs.

**Path C — Side-Step (community)** — `koda-dernet/Side-Step` on GitHub, one-click via Pinokio
- Effectively "training_v2 with more adapter types + GUI + CLI + wizard + AI captioning"
- **Adapter zoo**: LoRA, DoRA, LoKr, **LoHA, OFT** (Crucible currently uses LoRA/LoKr only)
- Auto-detects model variant (base/sft/turbo), picks "scientifically correct" timestep schedule per variant
- **Fisher Information adaptive ranks** (same idea as v2's estimate)
- Local Qwen2.5-Omni / Google Gemini / OpenAI / Music Flamingo captioning options
- VRAM down to 8 GB via 8-bit optimisers + encoder offload + adaptive batch sizing
- Multi-provider AI captioning + Genius lyric scraping

**Path D — Raw `diffusers` + PEFT** (bypass everything)
- Model weights on HF at `ACE-Step/ACE-Step-v1-3.5B` and `ACE-Step/Ace-Step1.5`
- Full control over training loop, sampler, evaluation; lose all engine integration
- Estimate: 1-2 weeks of work to reach feature parity with what we have
- _Only consider after exhausting A/B/C and if there's a specific feature we need that none of them offer_

### 13a.2 Existing HF LoRA adapters (cataloged, not directly usable)
- [`woctordho/ACE-Step-v1-LoRA-collection`](https://huggingface.co/woctordho/ACE-Step-v1-LoRA-collection) — musician-style LoRAs, **PEFT format targeting the 2B base/turbo**, NOT compatible with our XL 4B without conversion (and ComfyUI #9753 / #12638 indicate the key-format conversion is non-trivial)
- [`David-A-Amoo/ACE-Step-1.5-Naija-Legacy-Rhythms-LoRA-v1`](https://huggingface.co/David-A-Amoo/ACE-Step-1.5-Naija-Legacy-Rhythms-LoRA-v1) — Nigerian-music LoRA, format details unclear from HF page
- **No metal-specific community LoRAs found** as of 2026-05-30. Our hand-rolled adapter remains the only metal LoRA targeting our config.

### 13a.3 Timestep sampling — the critical distribution-mismatch hypothesis

**Cited from the ACE-Step paper ([arxiv 2506.00045](https://arxiv.org/pdf/2506.00045))**: the model was trained with **continuous logit-normal timestep sampling**, `μ=0, σ=1, shift=3.0`. Flow-matching objective is continuous-time by design.

**Read from `acestep/training/configs.py:78-82`** (v1 trainer = what Crucible uses): trains by **randomly sampling one of 8 discrete timesteps** per training step — the turbo inference grid.

**Read from `acestep/training_v2/configs.py:155-163`** (v2 trainer = in-tree, CLI-only): uses continuous logit-normal with `μ=-0.4, σ=1.0, data_proportion=0.5`.

**Inference reality**: xl-sft at our documented best-quality settings uses **32-64 continuous timesteps** in the denoising trajectory.

**Hypothesis (mine, NOT measured on our setup)**: training on discrete-8 = adapter learns to fix denoising errors at 8 specific noise levels, while inference traverses 32-64 different noise levels mostly disjoint from those 8. The mismatch could plausibly cause the "smear / lack of coherence" symptom + the val-loss plateau we observe with discrete-8 training. Plan 2 (§12) tests this hypothesis with a single-variable A/B.

Related research: [Curriculum Sampling: A Two-Phase Curriculum for Efficient Training of Flow Matching (arxiv 2603.12517)](https://arxiv.org/pdf/2603.12517) — discusses how "Logit-Normal and other middle-biased sampling distributions accelerate early convergence but yield worse asymptotic fidelity than Uniform sampling, suggesting that corrected or curriculum-based sampling approaches may improve final model quality." Worth re-reading if Plan 2's continuous A/B itself shows pathology (e.g. fast early gains but worse late fidelity).

### 13a.4 Adapter algorithm landscape
Ordered by parameter efficiency / expressiveness trade-off:

| Algorithm | Mechanism | Engine support | Side-Step | Notes |
|---|---|---|---|---|
| **LoRA** | `W + α(BA)` low-rank product | yes (PEFT) | yes | Classic. Universal compatibility. lr=1e-4 typical. |
| **DoRA** | LoRA + magnitude/direction split | yes (LyCORIS `weight_decompose=true`) | yes | More expressive per param. **Needs `lr~1e-3`** — engine ships `lr=0.03` default which blows it up. Memory note [[engine-lokr-defaults]] confirms this. Worth testing as Plan 3 with `lr=0.001` after Plan 2 resolves. "Better extraction from small data" — relevant given our small-dataset territory. |
| **LoKr** (our current) | Kronecker product decomposition | yes (LyCORIS) | yes | **~10× faster** training than LoRA per upstream tutorial (cited, not measured by us). Smaller files. Some loader limitations. Compatible with DoRA via `weight_decompose=true`. |
| **LoHA** | Hadamard product of two low-rank | no (engine doesn't ship) | yes | More expressive than LoRA at same parameter count. **Sometimes better for style transfer** (community claim, not measured). Worth a Side-Step trial if Plan 2 + Plan 3 don't move the needle. |
| **OFT** | Orthogonal Fine-Tuning — update is constrained to rotations of base weights | no | yes | Tends to produce cleaner adapters because the model's structure is preserved by construction. **Hypothesis**: could be the right fit for "augment metal character without adding noise". |
| **BOFT** | Block-diagonal OFT | no | yes | More parameter-efficient OFT variant. |

Engine `start_lokr` exposes via `lokr_weight_decompose`, `lokr_decompose_both`, `lokr_use_tucker`, `lokr_use_scalar`, `lokr_factor`, etc. — fine-grained control once we're inside LyCORIS-land.

### 13a.5 Quality measurement landscape (why we picked CLAP for Plan 1)

We evaluated 7 approaches before picking the Plan 1 set:

1. **Val_loss alone** (current default) — MSE in latent space. Necessary but coarse. **Measured 2026-05-29/30**: 35-track best had val 0.6919 vs 6-track best ~higher, yet 35-track sounded WORSE. **Val_loss correlation with perceptual quality is broken at our scale.** This finding alone is sufficient justification for moving beyond val_loss as a primary signal.

2. **Multi-noise val passes** — same val sample, multiple noise rolls per epoch, average loss. Reduces variance ~√N. Cheap improvement (just more forward passes). Implementable in our Patch D val loop as a `val_passes` config. Doesn't change the fundamental "MSE-vs-perception" issue.

3. **Train/val gap as overfit canary** — derived from existing history, no new infra. Alarms when val rises N epochs in a row while train falls. Standard ML hygiene.

4. **Adapter weight introspection** — load saved safetensors on CPU, compute mean abs weight magnitude per module (growth = learning; flat = converged), L2 norm of update vs previous ckpt, dora_scale distribution (alarm at `abs(max) > 3.0` per the Run-1 blowup pattern verified in [[engine-lokr-defaults]] memory), singular value spectrum of `w1 @ w2` (alarm if degenerate = mode collapse). **Zero GPU**, fast, safe to run during training. **In Plan 1 §11.2**.

5. **CLAP zero-shot tag scoring** — generate samples per checkpoint, run through box analyze service (port 5075, `analyze_py.analyze`), get confidence on target tags. **We already have CLAP installed**. Yields a "metal-ness" curve over training. Cost: ~80s per generation, ~12 takes per checkpoint = ~16 min/ckpt. **Plan 1 §11.3 picks this as primary perceptual signal.**

6. **Centroid-distance via CLAP embeddings** — pre-compute mean CLAP embedding of training corpus = "metal centroid." Per generated sample, cosine sim to centroid. More robust than tag matching because it's distribution-based not category-based. **Plan 1 §11.3.4 includes this as secondary signal.**

7. **True FAD (Fréchet Audio Distance)** — gold standard in music-gen literature. Uses VGGish or PANN features → distribution distance between generated and reference sets. Requires installing PANN inference + storing a reference distribution. **Skipped for Plan 1**: marginal benefit over (6) given the extra install + storage cost. Revisit if (5)+(6) prove inadequate.

8. **MERT / MULE / encodec feature similarity** — same family as (6), different audio encoder. **Skipped**: requires installing a separate audio model we don't currently use.

### 13a.6 The val_loss-is-misleading finding (key empirical observation, 2026-05-30)

**Measured on our box**: 6-track 150-epoch run (yesterday) sounded BETTER than 35-track 200-epoch run (overnight), despite the 35-track run having LOWER absolute val_loss (0.692 vs ~higher). User listening tests on 4 takes (best/final × 0.3/0.5) confirmed the regression.

Implications:
- Val_loss as currently computed (5-sample val × discrete-8-timestep MSE) is **not a reliable proxy for perceptual quality** at this scale.
- "More tracks → better LoRA" is not validated on our setup — at least not without scope discipline (the 35-track run mixed 14 bands across multiple production eras + vocal styles).
- Plan 1's perceptual fitness signal is not optional polish; it's the only way to know which direction is up.
- The Nightwish single-band 6-track experiment (task #12) tests whether scope discipline alone explains the regression, or if the training process itself is part of the problem.

### 13a.7 DoRA discussion (deferred from this session)

Engine ships LoKr with `lokr_weight_decompose=True` (DoRA on) + `learning_rate=0.03` as defaults. Per [[engine-lokr-defaults]] memory + Run-1 verification: this combo **catastrophically blows up `dora_scale`** (max=19.18 after 100 ep). Mac route safeguard defaults to plain LoKr (`weight_decompose=False`) + `lr=0.01`.

DoRA done right = `weight_decompose=True` + `lr~1e-3` + ~150+ ep. Memory note claims "better extraction from small data" — relevant to our small-dataset territory. Worth a Plan 3 A/B *after* Plan 2 (continuous sampling) resolves. Don't change two variables simultaneously.

Plan 3 spec to be authored after Plan 2 lands — its baseline is whichever sampling mode wins Plan 2's §12.5 decision.

### 13a.8 Key takeaways for a cold session

1. **The trainer we use is one of three in-tree options + a community option + a raw-PEFT option.** Path A (HTTP API) is convenient; Paths B and C may produce materially better adapters via continuous sampling + CFG dropout + more adapter types. Plan 2 tests this.

2. **Val_loss is misleading.** Plan 1 ships perceptual scoring via CLAP that's much more correlated with what the user actually hears. **Plan 1 is a hard dependency for any A/B that wants to be quantitative.**

3. **Adapter algorithm choice matters and we've only tried 2 of 6.** LoHA + OFT (via Side-Step) are unmeasured opportunities. Plan 3+ territory.

4. **Dataset scope matters more than dataset size at our scale.** 6-track focused beat 35-track mixed. Future runs should probably default to subgenre or single-band scope until we have evidence broader works.

5. **Engine patch surface is growing.** 5 active patches + Plan 2's Patch 7. Maintain the `patches/engine-YYYY-MM-DD/` convention + the row in §7a + memory entry. Single-file copy-over on engine updates.

---

## 13b. Style-fidelity research: carry more of the artist (2026-06-06)

Goal: make a trained LoKr carry MORE of the artist's sonic character (timbre, production,
vocal style) into generations. Driven by the user's recall of three levers: "use more parts
of the model", "use something other than AdamW", "and other things". Four parallel research
agents (3 web + 1 doc-harvest) converged; full agent transcripts not stored, conclusions below.
Constraint accepted: dataset is MP3 only (no WAV available).

### 13b.1 Root cause of weak / "perturbation-like" capture (high confidence)
Three compounding causes, each independently capable of thinning style:
1. **lokr_factor = -1 is the SMALLEST, lowest-capacity LoKr.** This is the key new finding.
   LyCORIS docs: factor=-1 = full Kronecker = the deliberately-tiny variant ("Small LoKr").
   For strong style you want the "Large LoKr": a LOW positive factor (4-8) + larger dim.
   We have been training at factor=-1 the whole time. This is a FREE fix (API already passes it).
2. **alpha/dim = 128/64 = scale 2.0 stacked on a hot LR = double-driving.** Keep alpha = dim
   (scale 1.0) and let one knob drive.
3. **DoRA + lr 0.03** (engine default) blows up dora_scale (already known, [[engine-lokr-defaults]]).

### 13b.2 Lever 1 - "use more parts of the model" = TARGET MODULES (corrects prior note)
PRIOR internal note (HANDOFF / §7a Patch 3 rationale) said tighten to "just q/k/v/o, skip MLP".
**That is WRONG for a STYLE LoRA.** LyCORIS Guidelines + DoRA/style-LoRA consensus:
- INCLUDE: self-attn q/k/v/o AND **feed-forward / MLP (fc1/fc2 or gate/up/down)**. The FFN is
  where the model stores "what this timbre sounds like" - it is the single highest-value addition
  for richer timbre transfer. Optionally cross-attn k/v (binds style to prompt/tags).
- EXCLUDE: time/timestep-embed (learns nothing - matches our measured 29% all-zero w2 finding),
  and AdaLN/modulation (shared adaLN-single -> high-risk global garble; add only as a last,
  risk-accepted expansion).
- LyCORIS preset that gives exactly attn+FFN = **"attn-mlp"**. So Patch 3 should force preset
  "attn-mlp", NOT attn-only. Expansion order for more reach: attn -> +FFN -> +cross-attn-kv ->
  (last, risky) +adaLN.

### 13b.3 Lever 2 - "something other than AdamW" = OPTIMIZER + LR
- Our v1 HTTP path = AdamW only; the engine's training_v2 exposes prodigy / adafactor / adamw8bit.
- **Prodigy** (auto-LR, de-facto SDXL/diffusion-LoRA default) directly targets the "messy/
  perturbation" symptom. Community recipe: lr=1.0 (it is a multiplier; Prodigy estimates the real
  LR), d_coef=1 (->2 if d won't climb on our low steps/epoch), weight_decay=0.01,
  betas=(0.9,0.99), use_bias_correction=True, safeguard_warmup=True, scheduler=CONSTANT, warmup 0.
  Control overfit via epochs, not LR. ~2x optimizer memory (trivial for a LoKr adapter).
- Cheap diagnostic FIRST (no patch): one run at AdamW lr ~1e-4 (vs our 0.01). Flow-matching
  (SD3/Flux) trains at much lower absolute LR than DDPM; if "messy" largely resolves, the LR was
  the bug. CAVEAT: LoKr's Kronecker factors are tiny so its effective LR differs from full LoRA;
  the engine ships LoKr lr=0.03. So treat lr as a SWEEP (1e-4 / 1e-3 / 1e-2), not a settled fact.
- LR schedule: with Prodigy = constant. With manual AdamW = cosine (single cycle) + ~3-5% warmup.

### 13b.4 Lever 3 - "other things"
- **Continuous logit-normal timestep sampling - ALREADY TRIED, inconclusive (do not re-list as a
  fresh lever).** Patch 7 shipped (commit 1ea6465) and we ran a CLAP A/B on Nightwish ~2026-05-30
  (`library/lora_train_history/crucible_nightwish_continuous_vs_discrete.json`). At the usable
  strength 0.3 the two were ~tied (discrete BEST net 1.9 vs continuous FINAL net 2.1; continuous
  BEST 1.6); at 0.5 both garbled to pop/country. CONFOUNDED: the continuous run was 8 tracks vs the
  discrete 6, so not a clean single-variable test. CLAP signal was weak (barely tagged the
  symphonic/operatic/female character at all). Net: no clear win -> default stayed discrete and ALL
  later real LoRAs (battlebeast, beastinblack, avantasia) trained discrete. By-ear verdict not
  recorded - ask the user. Only worth re-touching as a CLEAN single-variable re-test, ideally paired
  with the genuinely-untried **CFG/caption dropout ~10-15%** (v2 cfg_ratio) which was NOT in that A/B.
- **Captions for STYLE** (high leverage, data-side): DROP the artist/band name and any trigger
  token (ACE-Step's trigger tag has "limited effect" per the official tutorial; names cause
  memorization). Describe only what VARIES across the set (structure, instrumentation present,
  tempo/feel, generic vocal type, lyric theme) and DELIBERATELY OMIT the constant production/timbre
  fingerprint you want the adapter to absorb. Keep the LM prose (the high-value audio-grounded part);
  replace the noisy Last.fm crowd-tag prefix with a small CONSISTENT curated tag set. <=256 tokens.
- **DoRA done right** (second pass, after LoKr capacity is correct): weight_decompose=True,
  lr ~1e-3, and if a param-group LR is exposed set the magnitude/dora_scale group to ~0.1x base.
  DoRA often matches LoRA at half the rank.
- **MP3 (accepted constraint): second-order vs the above.** Use highest-bitrate source (>=256,
  ideally 320), keep bitrate CONSISTENT across the set, never transcode/re-encode, drop silent
  tails (codec noise floor). Only if you HEAR HF fizz/birdies: a gentle ~18-19 kHz low-pass on
  training audio caps the model's HF expectation. Do NOT MP3->WAV "restore" for training (adds no
  real detail). The engine band-limits through its own latent anyway, so high-bitrate MP3 is OK.
- **Side-Step Fisher-information adaptive per-module ranks** (bigger lift): assigns rank to the
  layers that most carry THIS artist - a direct "absorb more character" lever; needs adopting the
  Side-Step trainer. Parked as a later experiment.
- **OFT/BOFT** report higher subject fidelity than LoRA (orthogonal finetuning preserves
  pretrained structure); only available via Side-Step. Aspirational upgrade.

### 13b.5 Recommended experiment plan (one variable discipline; train is USER-fired GPU)
Baseline target = crucible_avantasia (50 tracks, already preprocessed). Must measure with CLAP
fitness (Plan 1, §11) not val_loss - val_loss is misleading at our scale (§13a.6). Serialize
CLAP(:5075) and engine(:8001) [[no-concurrent-clap-engine]]; fresh engine boot before training
[[engine-fresh-boot-for-lora]].

Tier A - FREE (API only, no patch), do first, biggest leverage-per-effort:
  - lokr_factor 8 (was -1), lokr_linear_dim 64-128, lokr_linear_alpha = dim (scale 1.0),
    weight_decompose False, lr SWEEP {1e-2, 1e-3, 1e-4}, val_split 0.1, save_every 5, ~80-120 ep.
Tier B - PATCH/CLIENT (genuinely untried; Patch 3 = the high-value one):
  - Patch 3 (lokr_utils.py honors a preset) IS applied, but the train_lokr client never SENDS a
    preset, so targeting is still effectively default/full. Add a preset/target param to the client
    + Mac route and send "attn-mlp" (add FFN, drop time_embed/adaLN). This is the fresh lever.
  - Continuous timestep already tested (above) - re-test only as a clean single-variable run paired
    with CFG/caption dropout ~10-15% (the untried part), not as a default change.
Tier C - OPTIMIZER (untried):
  - Prodigy (lr=1.0, constant) via a v1 trainer patch or by routing through training_v2.
Tier D - DATA:
  - Caption rewrite (drop band name, omit production fingerprint, curated consistent tags + LM prose).
Tier E - DoRA-done-right pass (lr 1e-3 + 0.1x magnitude group) once A-D settle.

Recommendation: bundle Tier A (free: factor 8 + alpha=dim) + the Tier B attn-mlp send into ONE new
baseline, A/B it by ear + CLAP vs the current crucible_avantasia adapter (factor -1, dim64/alpha128,
discrete, lr0.01, 150ep), THEN single-variable Prodigy (C) and captions (D). Continuous timestep is
NOT in this list - already tested, inconclusive. The capacity levers (factor + attn-mlp targets) are
the real untapped ones and were never changed across any run to date.

## 13c. Metric validity - CLAP is NOT trusted yet (2026-06-06)

User challenged CLAP fitness as a winner-picker; the challenge holds. Evidence from our own code +
the single stored run (`library/lora_train_history/crucible_nightwish_continuous_vs_discrete.json`):
- `analyze_server._clap_tags` returns only ranked TAG NAMES, discarding `scores = ae @ te.T`
  (analyze_server.py ~L110). Scoring is rank-only; the continuous magnitude is lost at the boundary.
- `_merge_labels` scores against a crowded ~40+ metal-subgenre vocab -> no fine artist discrimination.
- lora_eval = n=1 per (ckpt,scale) -> seed noise dominates.
- In that run, artist-defining tags (operatic/female/soprano/orchestral/choir) appeared in ONE take
  only = the GARBLED 0.5 take; generic progressive/gothic metal topped ALL 8 takes. So the score
  tracks "stayed metal vs drifted to pop" (over-strength garble guard), NOT artist fidelity.

CONSEQUENCE: continuous-timestep's "inconclusive" verdict was reached with this weak metric on a tiny
set -> not a settled negative (see §13b continuous bullet). Re-test fairly.

BEFORE any metric picks winners it must pass VALIDATION:
1. Monotonic ranking of known refs: real-artist > same-genre-other-artist > other-genre > noise.
2. Test-retest stability (same file twice ~= same score).
3. Agreement with the user's existing blind-A/B EAR verdicts on known pairs (e.g. the measured
   35-track-worse-than-6-track call). Ears are ground truth [[wait-for-feedback]].
Better metric candidates than zero-shot tags (validate the same way): CLAP centroid-distance to the
artist corpus (needs an analyze_server patch to expose the embedding) or FAD to the artist set. Until
a metric passes (1)-(3), it is an advisory pre-filter (catch obvious garble), never the judge.

## 13d. Metric validation RESULT - CLAP-centroid fails (measured 2026-06-06)

Ran the §13c harness on Avantasia (centroid 25 tracks, holdout 25) vs Battle Beast (same-genre),
Metallica (other-genre), generated noise. Report: library/lora_train_history/metric_validation.json.

Means (cosine to artist centroid): artist 0.835 / same-genre 0.824 / other-genre 0.787 / noise 0.079.
- (1) ORDERING: passes (right order) BUT artist-vs-same-genre is a TIE: AUC 0.538, Cohen d 0.11
  = chance. CANNOT distinguish Avantasia from other power metal. vs noise AUC 1.0 (d 8.5), vs
  other-genre AUC 0.65. => CLAP-centroid is a genre/garble guard, NOT an artist-fidelity meter.
- (2) TEST-RETEST: FAILS. Same file embedded twice -> self-cosine 0.88 / 0.58 / 0.84 (should ~1.0).
  Cause: laion_clap rand-truncates a ~10s window per call (enable_fusion=False) -> non-deterministic
  embedding. Every CLAP number we have ever computed sits on this unstable base.
- trustworthy_as_prefilter = False.

CONCLUSIONS:
- User's distrust of CLAP is vindicated with data. Do not use CLAP (tags or centroid) as an
  artist-fidelity judge. At most it flags "stopped being metal / garbled" (the AUC-1.0-vs-noise job).
- The harness itself WORKS and is model-agnostic - it caught both failures. Reuse it to vet any
  future metric: pass when ordering ok AND retest ~1.0 AND AUC(artist>same-genre) is high.
- Two fixes to consider: (a) make /embed deterministic (full-track windowed averaging or fusion) -
  needed for ANY CLAP use; (b) swap CLAP for a MUSIC embedding (MERT / MusicFM) which should have
  the artist/timbre resolution CLAP lacks, then re-run this harness. (a) is cheap; (b) is the real
  path to a believable automated metric.
- Until a metric passes the bar, judge LoRA runs by blind A/B EAR [[wait-for-feedback]].

## 13e. Metric retry RESULT - MERT works as a pre-filter (measured 2026-06-06)

Same harness (§13c), MERT-v1-95M run Mac-local (MPS, deterministic full-track windowed mean),
vs the CLAP failure in §13d. Report: library/lora_train_history/metric_validation_mert_centroid_cosine.json.

| check | CLAP | MERT |
|---|---|---|
| determinism (retest self-cosine) | 0.88/0.58/0.84 FAIL | 1.0/1.0/1.0 PASS |
| artist vs same-genre (Battle Beast) | AUC 0.538 (chance), d 0.11 | AUC 0.728, d 0.505 |
| artist vs other-genre (Metallica) | AUC 0.65 | AUC 0.928, d 1.83 |
| trustworthy_as_prefilter | False | True |

- MERT HAS the artist-level resolution CLAP lacked: Avantasia vs other power metal AUC 0.728 (CLAP
  was a coin flip). Deterministic windowing fixed stability (retest 1.0). Music-trained embedding
  beats text-audio CLAP for this task, as hypothesized.
- CAVEAT: MERT cosines sit in a narrow high cone (artist 0.979 vs same-genre 0.975, mean gap 0.004).
  Discrimination is real via tiny within-bucket std (AUC/Cohen-d are the signal, not the mean gap).
  AUC 0.728 = useful PRE-FILTER / triage, NOT a fine-grained judge of two good takes. Ears remain
  final, esp. on close calls [[wait-for-feedback]].
- REMAINING: criterion (3) not yet run - validate vs the user's EAR verdicts on real GENERATED pairs
  (ear_pairs hook is built). Validated on real artist tracks; generated LoRA output may differ.
- BOTTOM LINE (initial, TEMPERED below): MERT-centroid looked like a believable pre-filter on REAL
  tracks. But the applied test (§13f) shows it does NOT judge artist fidelity on GENERATED takes.

## 13f. Applied test on generated takes - MERT does NOT judge artist fidelity (2026-06-06)

Scored 9 generated takes vs a fresh Avantasia MERT centroid + anchors (held-out Avantasia / Battle
Beast / noise). Report: library/lora_train_history/mert_take_scores.json. Result is NEGATIVE - it
vindicates the user's distrust:
1. ANCHORS COLLAPSED: real Avantasia 0.9774 vs Battle Beast 0.9774 = IDENTICAL on this sample. The
   §13e AUC 0.728 (artist vs same-genre) was sample-fragile, not stable. With a different
   centroid-split + BB sample the artist/same-genre separation vanished.
2. CROSS-LORA CONTROL INVERTED: a nightwish-LoRA take scored the MOST "Avantasia" (0.9719), above
   EVERY avantasia-LoRA take (0.959-0.965). Metric is not tracking artist identity in generated output.
3. GENERATED TAKES CLUSTER 0.93-0.97, all BELOW both real-audio anchors (0.977). MERT mostly
   separates real-vs-generated audio, not artist-vs-artist. Cannot rank takes by Avantasia-ness.
CONCLUSION: MERT-centroid = a reliable music-vs-noise / on-genre GUARD only (0.62 vs 0.95+ solid).
It is NOT a usable artist-fidelity judge for our generated takes. Do NOT wire it in as the LoRA
artist A/B judge. EARS remain the judge ([[wait-for-feedback]]) for "did this carry the artist".
Confound we cannot resolve from this: the inverted control could also mean the LoRAs impart little
distinct artist character (consistent with the weak/perturbation-like adapter finding) - either way
the metric can't arbitrate it. Net: no automated artist-fidelity metric has earned trust; proceed
with the §13b capacity/target-module experiments judged BY EAR.

## 13g. Noisiness diagnosis + LoKr lever research (2026-06-07)

CONTEXT. First clean run on the new stack: crucible_nightwish_tarja (21 Tarja tracks, clean
lyrics/captions), LoKr factor 8 / dim 64 / alpha 64 (scale 1.0) / attn+mlp (q,k,v,o,gate,up,down_proj)
/ ~19.9M params / lr 0.01 AdamW / discrete timestep / 150 epochs / grad-ckpt. User by-ear: carries
MORE Nightwish character than the old tiny attention-only adapter, BUT noisier at strength 1.0, and
at strength 0.3 it loses most of the Nightwish character. => the USABLE STRENGTH WINDOW IS SQUEEZED
(1.0 noisy, 0.3 weak). That squeeze is itself the key signal: a well-trained adapter should either
stay clean at higher strength or hold character at lower strength. A squeezed window = over-geared /
fried weights => points at TRAINING-SIDE fixes, not just lowering inference strength.

WHAT CHANGED vs the old (cleaner-but-weaker) adapter, and which way each pushes NOISE@1.0:
| change                                   | style | noise@1.0 |
| factor -1 (min) -> 8 (high capacity)     | ++    | ++ bigger/finer deltas |
| attn-only -> attn+MLP (gate/up/down)     | ++    | ++ FFN writes hard into residual stream |
| alpha/dim 128/64 (scale 2.0) -> 64/64 (1.0) | -  | -  (the one change that REDUCED gain) |
| lr 0.01 held while params ~10x'd         | .     | ++ too hot for 20M params |
Net: we raised delta magnitude (capacity + FFN) and only partially compensated (alpha halved), kept
lr hot, and auditioned at max strength. More style AND more frying — exactly the symptom.

RANKED CAUSES (confidence; provenance = upstream unless tagged measured):
1. **lr 0.01 ~100x too hot for a 20M flow-matching LoRA** (HIGH). Flux/flow-matching LoRA consensus
   lr ~1e-4 (down to 1e-5 if noisy); 0.01 is the classic "fried adapter" recipe at this param count.
   Our 0.01 "safe default" [[engine-lokr-defaults]] was calibrated on the TINY factor=-1 attn-only
   adapter; it never scaled down when capacity ~10x'd. Sane range here 5e-5..2e-4, start 1e-4.
2. **Bigger adapter -> lower inference-strength sweet spot** (HIGH). strength linearly scales every
   delta; a high-capacity adapter's total perturbation at 1.0 is far larger than the old tiny one's,
   pushing activations off the base manifold = grain/noise. BUT user measured 0.3 = too weak, so the
   fix is not strength alone (see squeeze above). Test 0.5/0.6/0.7 for the current best.
3. **Overcooked: 21 tracks x 150 epochs** (HIGH; matches our MEASURED §13/§13.4: useful learning by
   ~ep30, ep104 ~tie with ep27, 200 was overshoot). Overfit on a tiny set = brittle spiky deltas =
   noise. Drop to 30-50 epochs (also makes every experiment ~3x cheaper). best(ep104) vs final(ep150)
   by-ear is a free check.
4. **No CFG/caption dropout + guidance 8 at inference** (MED). No trained null branch -> classifier-
   free guidance extrapolates into artifacts, worst at high strength. Add ~10% caption/CFG dropout.
5. **alpha/dim scale still high** (MED). 1.0 now; LyCORIS/kohya LoKr default is 0.5 (alpha 32/dim 64).
   Lower scale = more headroom before frying + lets a higher inference strength stay stable.
6. **MLP/FFN is the biggest single artifact contributor** (MED) AND the main reason we gained style.
   Keep it; tame it: lower alpha, dropout, OR asymmetric capacity (LyCORIS Flux preset uses attn
   factor 12 / FFN factor 6). If pruning, drop down_proj first (writes FFN output into residual).
7. **Discrete-8 timestep grid under-trains the continuous inference trajectory** (MED) -> smear/noise.
   Continuous logit-normal (mu=-0.4, sigma=1.0) matches the model's native sample_t_r. **The
   continuous run train_20260607-114629 tests exactly this** (discrete-vs-continuous A/B, task #1).
8. **Lossy MP3 codec artifacts** (LOW/2nd-order [[dataset-caption-sources]]). A higher-capacity
   adapter can memorize HF birdies/pre-echo the old small one couldn't. Mitigate only if 1-7 don't
   resolve: highest-bitrate sources, optional ~16-18 kHz low-pass on training audio.
9. **Inference shift** (LOW, FLAGGED CONFLICT). Side-Step doc says Base/SFT should generate at
   shift=1.0/50 steps and shift=3.0 is a Turbo setting; but OUR measured ACE recipe used shift 3 for
   xl-base and liked it (RESEARCH §8a). Treat as an inference-side thing to A/B, do NOT assume.

OPTIMIZER (user asked "better than AdamW now?"): AdamW is fine; lr=0.01 is the fault. Modern picks:
- **Prodigy** (auto-lr) or **Prodigy-Plus-Schedule-Free** (diffusion-LoRA-native, Adafactor-level
  memory; needs optimizer.train()/eval() + constant sched; set lr=1.0, d0~1e-7, d_coef~2 so d climbs
  on flow-matching) = removes lr guessing, the de-facto modern default. **CAME** = fixed-lr low-memory
  alt (lr ~0.5-0.9x AdamW). **ADOPT** = built for diffusion's noisy gradients, near drop-in.
- SKIP Muon/SOAP/Sophia/Adam-mini = LLM-pretraining tools, not for a 20M LoRA.
- Our v1 HTTP path = AdamW-only; engine training_v2 has prodigy/adafactor/adamw8bit -> optimizer swap
  needs an engine patch or routing through v2. lr/alpha/factor/dim/epochs/targets/timestep already
  free via the /api/lora/train body.

WHAT IS FREE-VIA-API vs NEEDS-PATCH:
- Free (existing /api/lora/train body): learning_rate, lokr_factor, lokr_linear_dim, lokr_linear_alpha,
  train_epochs, val_split, timestep_sampling_mode, target_modules, gradient_checkpointing.
- Needs an engine patch: lora_dropout for LoKr (NOT exposed on StartLoKRTrainingRequest today),
  CFG/caption dropout (v2 cfg_ratio), optimizer swap, asymmetric per-module-group alpha/factor,
  loss weighting (min-SNR).

EXPERIMENT PLAN (cheap -> expensive; epochs 50 makes a run ~1.5h not ~4.5h):
- TIER 0 (FREE, no training; needs engine free): strength sweep 0.5/0.6/0.7 (0.3 already = too weak,
  1.0 = noisy) + best(ep104) vs final(ep150) by ear, on BOTH discrete and continuous adapters.
- TIER 1 (CORRECTED 2026-06-07 after user Q: "does 50ep undertrain?"). YES it would, paired with a
  lower lr. lr and epochs are COUPLED (AdamW total movement ~ lr x epochs); the "learning done by
  ~ep30 / 200 was overshoot" finding was MEASURED AT lr 0.01 and does NOT transfer to a lower lr
  (lower lr converges slower -> needs MORE epochs, not fewer). The first-draft "lr 1e-4 + 50ep" cut
  BOTH coupled knobs -> ~1/300th the training = undertrained. FIX: change lr ONLY, keep the epoch
  budget. lr 0.01 -> 1e-3 (10x cooler, the prime fry fix; NOT the full 100x to 1e-4 which at 150ep
  risks undertraining), KEEP alpha 64, KEEP epochs 150, save_every_n_epochs 10 so intermediate
  checkpoints can be auditioned by ear (val is weak [[clap-scoring-unproven]] -> perceptual sweet spot
  may differ from val-best). One variable = lr. Body:
  {dataset:"crucible_nightwish_tarja", method:"lokr", lokr_factor:8, lokr_linear_dim:64,
   lokr_linear_alpha:64, lokr_weight_decompose:false, learning_rate:0.001, val_split:0.1,
   train_epochs:150, training_seed:42, gradient_checkpointing:true, save_every_n_epochs:10,
   timestep_sampling_mode:"discrete",
   target_modules:["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]}
  READ THE VAL CURVE: still falling steeply at ep150 -> undertrained, extend; flat by ~ep60 ->
  converged (noise was the hot lr). If 1e-3/150 still fries -> lr 1e-4 (accept more epochs needed, or
  use Prodigy which auto-tunes effective lr and decouples this). alpha 32 + epoch tuning = later
  single-variable follow-ons. (reuses clean tensors; fresh-boot engine + verify data gate before GPU.)
- TIER 2 (needs engine patch, only if Tier 1 still noisy): add lora_dropout 0.1 (patch
  StartLoKRTrainingRequest + plumb to LoKRConfig, like target_modules) and/or CFG/caption dropout 0.1,
  and/or swap optimizer to Prodigy. Optionally asymmetric FFN capacity or drop down_proj.
ALWAYS judge by ear (no automated metric trusted, §13f); verify the data gate (lyrics/bpm/key) before
any GPU [[lora-training-routine]]; serialize engine vs CLAP [[no-concurrent-clap-engine]].

## 13h. Task #1 ear verdict - discrete vs continuous (2026-06-07, user by ear)

Adapters compared (both factor 8 / dim 64 / alpha 64 / attn+mlp / lr 0.01 / 150ep, one variable =
timestep mode): nightwish_tarja_150ep_discrete vs nightwish_tarja_150ep_continuous. User ran the A/B
themselves across multiple Tarja songs/takes. Findings:
1. NO CLEAN WINNER discrete vs continuous - song-dependent (one wins on one song, the other on
   another take). Difference is slight. => continuous timestep remains NOT a clear win, consistent
   with §13b.4. Discrete stays the working default.
2. Continuous "BEST" checkpoint (val-best @ ep20, val 0.812) is a FALSE pick: little Nightwish
   character beyond what the prompt already supplies. The continuous FINAL (ep150) is better by ear.
   => the ep20 "plateau" was UNDERFIT, not converged. Re-confirms val-loss "best" is misleading at
   our scale [[clap-scoring-unproven]] - do NOT trust the engine's best/ checkpoint as the pick;
   audition finals + intermediates by ear (this is why Tier 1 sets save_every_n_epochs 10).
3. Discrete captures SLIGHTLY more of Tarja's vocal timbre (slight edge).
4. BOTH adapters break up a lot above strength 0.7. Usable ceiling ~0.7. The break-up is present on
   BOTH => it is a TRAINING-SIDE fault (over-geared / fried weights, §13g), not an inference-strength
   knob. Lowering strength alone cannot fix it (0.3 was already too weak in the original handoff).
DECISION: proceed to TIER 1 (§13g) - retrain discrete with lr 0.01 -> 1e-3, all else held
(alpha 64, 150ep, save_every 10). One variable = lr. Reuses the existing clean tensors.

## 13i. Tier 1 result - lr 0.01 -> 1e-3 (measured 2026-06-07)

Run: train_20260607-182943__lokr_150ep_discrete_lr1e-3_a64. SINGLE variable vs the discrete
baseline: learning_rate 0.01 -> 0.001. Trained on xl-base (gated at /v1/init, confirmed loaded
before train [[lora-train-on-xl-base]]), 21 Tarja tracks, factor 8 / dim 64 / alpha 64 / attn+mlp
(q,k,v,o,gate,up,down_proj) / discrete / 150ep / ~19.9M params / grad-ckpt. Engine echoed the full
config back (lr 0.001 confirmed, not the route's 0.01 default). Wall clock 4.39h.

VAL CURVE (untrusted as a judge [[clap-scoring-unproven]]; read ONLY for overfit dynamics):
- Drops ~0.98 -> ~0.75 by ep30-40 (per-epoch min 0.7502 @ ep40; engine running-best 0.7277 @ ep104),
  then RISES and oscillates noisily for the rest (mean val ep55-75 = 0.847, ep130+ = 0.823) to ep150.
- Shape = CONVERGED EARLY (~ep40) THEN OVERFIT / noisy. This is NOT the "still falling at ep150 =
  undertrained" case from §13g; it is the opposite. At lr 1e-3, 150ep is TOO MANY - useful learning
  is early. best val 0.7277 ~= the old lr-0.01 run's 0.733: cooling lr 10x barely moved val (expected,
  val is not the noise/fry metric).
- TEMPERS the §13g coupling claim ("lower lr needs MORE epochs"): a 10x-lower lr did NOT need
  proportionally more epochs to bottom val - it still bottomed ~ep40 (like lr 0.01's ~ep30). The
  slower-convergence assumption was not observed on val here. (Caveat: val untrusted; perceptual
  sweet spot may differ.)

CHECKPOINTS: box /adapters (helper :5080) exposes only best/ (ep104) + final/ (ep150) - BOTH sit in
the overfit/noisy region by val. save_every_n_epochs:10 WAS sent but per-epoch checkpoints are not
enumerated by the upload helper; whether they were saved to disk + are loadable is UNCONFIRMED.

NEXT (by EAR, user-driven [[wait-for-feedback]]): A/B the new lr-1e-3 adapter (best AND final) vs the
old lr-0.01 discrete at matched seed + strength 0.5/0.6/0.7. The real question val cannot answer: did
the lr cooldown REDUCE the frying at usable strength while keeping Tarja character?
- If it fries less + keeps character -> lr was a real lever; next cut epochs to ~50-60 (the overfit-
  after-ep40 shape) for a cheaper, less-overfit run.
- If it still fries -> lr alone is not enough -> Tier 2 (lora_dropout 0.1 + CFG/caption dropout, or
  Prodigy; needs engine patch) and/or alpha 32.
- If best+final are both overfit-noisy, making an early (~ep40) checkpoint loadable becomes worth it.

## 13j. v2 trainer over HTTP - wrapper STAGED (2026-06-08)

Decision (user): adopt training_v2 rather than keep hand-porting its features onto v1 one
patch at a time. v2 is more advanced (§13a Path B): optimizer choice incl. **Prodigy**,
scheduler choice, **CFG dropout (cfg_ratio)**, native continuous timestep, Fisher-info ranks.
Blocker was "v2 is CLI-only, no HTTP" - solved by an in-process wrapper.

Design (reverse-engineered from GitHub main; see patch README for the source reads):
- v2 `FixedLoRATrainer(model, adapter_cfg, train_cfg)` takes an ALREADY-LOADED model and
  `train()` is a generator yielding `TrainingUpdate(step, loss, msg, kind, epoch=)`. So the
  wrapper runs v2 IN-PROCESS reusing the engine's loaded decoder (same VRAM mgmt as v1
  start_lokr) - NO subprocess, NO double-VRAM (avoids the never-frees bug, §task18).
- v2 reads via the same `acestep.training.data_module` as v1 -> our v1 `tensors/` SHOULD
  feed it directly (the showstopper to verify = [V5] tensor-key compat).
- v2 LoKr saves the same `lokr_weights.safetensors` -> loads in our picker unchanged.

STAGED, NOT DEPLOYED: `patches/engine-2026-06-08/` (new route `/v1/training/start_lokr_v2`
+ a 2-line wiring edit to train_api_service.py) + Mac side (`acestep_train.train_lokr_v2`,
`POST /api/lora/train_v2`). Built vs GitHub main -> deploy is GATED on a cheap 2-epoch smoke
test that verifies [V1-V5] (import paths, config field names, trainer signature, tensor
compat) before any multi-hour run. Mac send-side needs a backend restart to activate (after
the current 250ep run). First Prodigy run: `optimizer_type:"prodigy"`, `scheduler_type:
"constant"`, `learning_rate:1.0` (Prodigy auto-estimates the real lr; §13g optimizer notes).
Judge by ear ([[clap-scoring-unproven]]); v2 may not emit v1-style val/best so the poller's
val curve may gap.

## 13k. 250ep result - more epochs at lr 1e-3 (measured 2026-06-08)

Run train_20260608-000153__lokr_250ep_discrete_lr1e-3_a64. SINGLE variable vs §13i: epochs
150 -> 250. All else held (lr 0.001, factor 8, dim/alpha 64, attn+mlp, discrete, xl-base).
Wall 7.31h. Final TRAIN loss 0.6096 (vs 150ep's 0.6964 - lower, as expected with more steps).
Motivation: user ear-judged 150ep final@1.0 = good Nightwish music but not enough bel canto;
wanted more training to deepen the style.

VAL CURVE (untrusted as judge [[clap-scoring-unproven]]; overfit dynamics only):
- Bottoms ~ep40 (0.751), engine running-best 0.7290 @ ep104 (~= 150ep's 0.7277 @ ep104 -
  basically identical best), then oscillates and TRENDS WORSE late: mean val ep30-50 0.822 ->
  ep150-170 0.864 -> ep230-250 0.856. min per-epoch 0.7374 @ ep160.
- Read: by val the extra 100 epochs did NOT deepen useful learning - best still ~ep104, late
  epochs noisier/worse = MORE overfitting. Lower TRAIN loss + higher VAL = textbook overfit.
- BUT val != style strength, and the user wanted MORE style. final@ep250 has the most weight
  movement (lowest train loss) -> potentially the strongest style imposition even as val
  "overfits". So the val verdict ("no gain, more overfit") cannot settle it.

DECISION = EARS. Audition 250ep FINAL at strength 0.8-1.0 for bel canto vs 150ep final
(train_20260607-182943, persists on box). If more bel canto -> more epochs helped (val lied
again); if garbled / no gain -> we've hit the ceiling of "more epochs at lr 1e-3" on this
21-track set -> pivot levers: the v2 path (Prodigy auto-lr + cfg_ratio, patch staged §13j) or
data (more Tarja tracks / caption rework §13b.4). All 4 nightwish_tarja runs (150 discrete,
150 continuous, 150 lr1e-3, 250 lr1e-3) persist on box with best+final for A/B.

EAR VERDICT (2026-06-08, user): 250ep final is MUFFLED - loss of clarity on the voice,
instrumentation less clear - but NO obvious frying. So more epochs at lr 1e-3 added OVERFIT
SMEAR/MUD, not bel canto (matches the val "overfit more / late epochs worse" reading). =>
"more epochs at lr 1e-3" is the WRONG direction; the 150ep lr-1e-3 final stays the better
CLEAN adapter. CONCLUSION: the bel-canto gap is NOT a training-duration problem. We have now
bracketed lr 1e-3: 150ep = clear but bel canto too weak; 250ep = muffled. And lr 0.01
(original) = more character but fried >0.7. So the sweet spot is a TRAINING-STRENGTH/cleanliness
problem, pointing to (a) FREE first: operatic vocal tag cluster at inference
[[operatic-vocal-tags]] (bel canto + classically trained + coloratura + rich operatic vibrato)
on the 150ep adapter - might close the gap with zero GPU; (b) a WARMER fixed lr ~3e-3 (between
the frying 0.01 and the weak 1e-3) at ~100-150ep; or (c) the v2/Prodigy path (auto-lr finds the
exact sweet spot + cfg_ratio for cleaner guidance, wrapper staged §13j). NOT more epochs, NOT
cooler lr.

## 13l. v2/Prodigy run IN FLIGHT (2026-06-08)

The v2 HTTP wrapper (§13j) is DEPLOYED + VERIFIED on the box. Bring-up caught two real issues
(both fixed): (1) `device="auto"` not resolved in-process -> torch.device threw; fixed by
passing `str(handler.device)`. (2) `prodigyopt` not installed -> v2 optim.py SILENTLY fell back
to AdamW (only a WARNING line; the smoke "succeeded" on AdamW@lr1.0 = garbage). Fixed via
`uv add prodigyopt>=1.1.2` from the engine dir. Prodigy engagement then CONFIRMED airtight:
`import prodigyopt` succeeds in the engine venv (run from the ENGINE dir, not parent) -> the
ImportError fallback is impossible -> Prodigy is used; corroborated by the `Using decoupled
weight decay` print (prodigyopt-only) + absence of the fallback warning.

Run: train_20260608-130123__lokrv2_150ep_prodigy_cfg0.1 (direct to engine, fresh xl-base).
Params: optimizer=prodigy, scheduler=constant, lr=1.0 (Prodigy auto-estimates effective lr),
cfg_ratio=0.1, factor8/dim64/alpha64, attn+mlp (q,k,v,o,gate,up,down_proj), 150ep, save_every10,
~19.9M params, ep1 ~115s -> ~5h ETA, ep1 loss 1.086. First real change of the OPTIMIZER lever
(prior runs all AdamW). Judge by EAR vs the 150ep lr-1e-3 AdamW adapter (does Prodigy auto-lr +
cfg_ratio carry more bel canto / stay cleaner at usable strength?). Fired direct to engine so the
Mac history poller is NOT capturing this run - read the curve via /v1/training/status loss_history
or check best/final on box when done.

RESULT (done ~5h, 2026-06-08): TRAIN-loss curve (no val - v2 emits none) was SMOOTH + MONOTONIC:
mean first-25% 0.890 -> mid 0.707 -> last-25% 0.552 (min 0.432, final ~0.45-0.54). Markedly lower
final train loss than the AdamW runs (150ep 0.70 / 250ep 0.61) = Prodigy found a higher effective
LR and fit harder. NOT a quality verdict (lower train loss could = richer capture OR more overfit;
no val to arbitrate, ears decide). Saved FINAL only (best_path=None - v2 has no val/best tracking;
save_every_10 intermediates not enumerated by the :5080 helper). Confound: bundle (Prodigy +
constant sched + cfg_ratio 0.1 + continuous timestep), so a win is not creditable to Prodigy alone.
EAR TEST PENDING: load final, judge on the FULL-ARTIST bar [[lora-goal-full-artist-sound]] (voice +
whole band/era, not just bel canto), strengths ~0.5-1.0, vs 150ep lr-1e-3 AdamW final
(train_20260607-182943).

EAR VERDICT (2026-06-08, user): BEST SO FAR. Usable window widened - 0.8 is about the max before it
gets messy (vs AdamW frying >0.7). Captures Nightwish STYLE elements well. Vocal identity is
inconsistent: sometimes loses the bel canto, a faster track at 0.6 read more Lizzy Hale than Tarja
(but not always). Overall = "you'd think INSPIRED BY Nightwish, not that it IS Nightwish." => the v2
recipe (Prodigy+constant+cfg0.1+continuous) is the new baseline; the gap now is STRENGTH/CONSISTENCY
of artist-specific capture (esp. the specific vocal timbre + whole-band cohesion), not cleanliness.
NEXT LEVERS (toward [[lora-goal-full-artist-sound]]): (1) MORE CAPACITY - dim 64->128 on the v2
recipe, cheap (reuse tensors), now affordable (~14GB used, ~10GB headroom); "inspired-by-not-is"
suggests undersized. (2) FISHER-INFO adaptive ranks (v2 estimate.py; needs wrapper plumbing) - the
most targeted "put capacity where THIS artist lives" lever. (3) cross-attn (attention_type) +
cfg_ratio tuning - cheap finer tuning, may sharpen vocal-identity binding. (4) more data / caption
rework (§13b.4) - bigger lift, strengthens specific identity. All judged by ear.

## 13m. Research: in-training overfit monitors (val-free), 2026-06-08 deep-research pass

Deep-research harness (105 agents, 2.5M tok, 23 sources, 25 claims adversarially verified -> 20
confirmed / 5 killed). Goal: detect overfit/overtrain DURING training without a trusted val signal,
cheap from adapter weights/grads (no generation). Full report:
tasks/wba2lymkv.output (this session's /private/tmp). RANKED shortlist:

IMPLEMENT FIRST (cheap, val-free, CPU-side from grads/weights):
1. **GSNR (gradient signal-to-noise ratio) - track its DECLINE.** Per-coordinate gradient mean^2 /
   variance over micro-batches; falling GSNR = overfit onset. (Liu ICLR2020 2001.07384; Wang
   2309.13681.) No val, no generation. Caveat: proven on classification, not flow-matching.
2. **Gradient disparity - L2 distance between gradients of two disjoint training micro-batches.**
   Rising = overfit. Built FOR limited-data early-stop (no val split). (Forouzesh ECML-PKDD2021
   2107.06665, public code.) Single-source but unanimous.
3. **Stable rank / spectral of the LoKr update.** stable_rank = ||W||_F^2 / ||W||_2^2; watch
   collapse + condition-number blowup. Trivially cheap (SVD of the tiny adapter factors). Complement
   with weight-norm growth + effective rank. (Sanyal ICLR2020 1906.04659.)

DIFFUSION WHERE-TO-LOOK (not standalone triggers):
4. Two timescales: generation-quality onset ~n-independent vs memorization onset grows with n -> the
   train/val BIFURCATION (not absolute val) is the signal, and the safe window WIDENS as n grows (so
   50+ tracks safer than 20). NOTE: the strong "memorization-onset LINEAR in n" form was REFUTED
   (0-3 / 1-2 votes) - keep the qualitative picture, NOT the constants. (Bonnaire NeurIPS2025
   2505.17638; Favero 2505.16959.)
5. Memorization concentrates at INTERMEDIATE timesteps/noise -> log per-timestep-bin train MSE, watch
   the medium band for earliest divergence. (2602.17846.) A refinement, caveat: mid-noise data has
   little overlap with inference trajectory.

AVOID / DEPRIORITIZE:
6. **Sharpness/flatness (SAM, Hessian trace/eigs, adaptive sharpness) - DO NOT use as primary.**
   Strong evidence it does NOT correlate with generalization in FINE-TUNING (closest analog to LoRA);
   tracks LR instead, sometimes negative corr. (Andriushchenko ICML2023 2302.07011.)
7. IGS / Fisher-trace - optional, expensive, secondary (ties our Fisher-rank interest).
8. Inference-time cond-vs-uncond memorization detector (AUC .96, 0.2s) - violates no-generation;
   cheapest spot-check IF generation ever acceptable.

CRITICAL CAVEAT (honest headline): EVERY method was validated in a DIFFERENT setting (classification
/ from-scratch full diffusion / large n) - NONE on MSE flow-matching LoRA at n=20-50. All evidence is
BY ANALOGY, no calibrated threshold for us. SAME discipline as CLAP/MERT (§13c-f): instrument cheaply
-> log across a KNOWN-overfit run (e.g. the 250ep AdamW the user ear-judged muffled) -> align curves
to the ear-judged overfit point. Trust only if they correlate with EAR verdicts; else discard. Don't
auto-early-stop on an uncalibrated monitor.

## 13n. Weight-space overfit metrics FAIL ear calibration (measured 2026-06-08)

Computed the §13m weight-only monitors (stable rank, effective rank, spectral & Frobenius norm,
condition number) on ALL saved LoKr checkpoints of 5 nightwish_tarja runs (148 ckpts, tool:
tools/lora_overfit_metrics.py, results library/lora_train_history/overfit_metrics.json). Lined up
vs EAR verdicts. NEGATIVE result:
| run | ear | stable_rk | eff_rk | spectral | fro |
| 150ep discrete lr0.01 | fried>0.7 | 8.71 | 39.1 | 27.3 | 1359 |
| 150ep continuous lr0.01 | wash | 14.16 | 46.4 | 27.1 | 1417 |
| 150ep lr1e-3 | clean-but-weak | 4.64 | 41.97 | 2.75 | 1240 |
| 250ep lr1e-3 | overfit/muffled | 6.06 | 43.28 | 3.75 | 1246 |
| v2 prodigy | BEST (usable 0.8) | 14.84 | 48.78 | 13.0 | 1316 |
- The BEST adapter (v2) has the HIGHEST stable+effective rank -> "high rank = overfit" is false here.
- spectral norm tracks LEARNING RATE (lr0.01~27, lr1e-3~3, Prodigy~13), not quality - matches the
  §13m research caveat that these track hyperparams not generalization.
- condition number numerically degenerate (~1e17, w2=w2_a@w2_b near-singular) - useless.
- No clean within-run knee at the ear-judged overfit onset; train-loss (from ckpt folder names)
  bounces 0.16-0.20 with no signal either.
CONCLUSION: cheap WEIGHT-ONLY metrics do NOT give a usable overfit detector for our setting (same
fate as CLAP §13d / MERT §13f). Caught in ~15 min + a file copy, not a long run. NOT YET TESTED: the
research-TOP-RANKED GRADIENT-based signals (GSNR-decline, gradient disparity) - these need GRADIENTS
during training, uncomputable from saved weights. Testing them = instrument the v2 wrapper to log
them + 1 run on a known-overfit config + check vs ears. That is the last cheap shot before concluding
no automated overfit signal exists for us and ears stay the only judge. Note (tooling): v2 saves
bf16 -> load via safetensors.torch not .numpy. Note (incidental): v2 adapter wraps cross-attn +
condition_embedder too (attention_type=both), more modules than the v1 attn+mlp filter.

## 13o. Capacity bump run IN FLIGHT (2026-06-08)

Run: train_20260608-220714__lokrv2_150ep_prodigy_dim128_cfg0.1 (direct to engine, fresh xl-base).
SINGLE variable vs the winning v2/Prodigy run (§13l): lokr_linear_dim 64->128, lokr_linear_alpha
64->128 (scale held at 1.0). All else identical: prodigy / constant / lr1.0 / cfg_ratio0.1 / factor8
/ attention_type both / target attn+mlp / 150ep / save_every10. Reuses existing tensors. ~109s/epoch
-> ~4.5h, ep1 loss 1.093. Tests the "inspired-by-not-is = undersized adapter" hypothesis - does
doubling capacity carry MORE of the specific artist (voice+band) per [[lora-goal-full-artist-sound]]?
EAR TEST (sparse-ladder, per the option-B decision in §13n - no automated overfit metric trusted):
audition ~ep40/80/120/final at strength 0.6-0.8 on the full-artist bar, vs the dim-64 v2/Prodigy
final. If more artist + still clean to ~0.8 -> capacity was the lever, push to 256 next; if it
overfits faster / muffles -> capacity isn't the gap, pivot to Fisher ranks / cross-attn / data.

RESULT (done ~4.5h, 2026-06-08): fit HARDER than dim-64 - train-loss last-quarter mean 0.40 vs 0.55
(min 0.256, final 0.431). More capacity -> lower train loss (expected); NOT a quality verdict (could
be richer capture OR more overfit - the lower loss raises overfit risk). Saved FINAL (loadable in
picker) + per-epoch checkpoints on box (helper enumerates final only; earlier ckpts loadable by path
via /v1/lora/load). EAR TEST PENDING: final-first - A/B dim-128 final vs dim-64 v2/Prodigy final at
0.6-0.8 on the full-artist bar. If final is muffled/overfit -> ladder back to ep40/80/120 (get
checkpoint folder names via a box `dir`, load each via /v1/lora/load). If final is more-artist +
clean -> capacity helped, push dim 256 next.

EAR VERDICT (2026-06-08, user): NOT better. Usable window moved DOWN - 0.7+ breaks up/artifacts (vs
dim-64 clean to 0.8), though 0.5 is fine. No clear artist gain. => uniform capacity (dim 64->128)
just made the deltas HOTTER (matches the harder fit / lower train loss) = more artifacts at lower
strength, NOT more artist. dim-64 v2/Prodigy REMAINS the champion; do NOT push dim 256 (would be even
hotter). KEY INFERENCE across all runs: scaling DELTA MAGNITUDE (capacity OR lr) does NOT add artist
fidelity - bigger just = artifacts. The "inspired-by-not-is" gap is about WHERE capacity goes
(targeting / Fisher-info ranks) or the DATA (more tracks / caption rework), not how much. Next lever
should be TARGETED capacity (Fisher) or DATA, picked research-led - NOT another magnitude knob.

## 13p. Research: capture-more-artist levers (cost-capped, 2026-06-08)

Cost-capped deep-research (54 agents, 1.24M tok - cap worked vs §13m's 2.5M). Full report:
tasks/w5ous7dxl.output. RANKED shortlist:
1. **FIM-LoRA-style gradient-variance calibration (TARGETED rank)** - ~8 calibration backward passes,
   compute gradient variance of each adapter matrix per module (cheap diagonal empirical Fisher),
   redistribute the SAME fixed rank budget per-module importance WITH a rank floor. MEASURES where THIS
   artist lives on OUR model vs guessing. (AdaLoRA 2303.10512, LoRA2 2603.21884, FIM-LoRA 2605.16800.)
   HONEST: evidence shows adaptive rank MATCHES uniform at equal budget (FIM-LoRA 88.6 vs 88.7) - wins
   on efficiency/low-budget robustness, NOT a big fidelity jump. So partly a DIAGNOSTIC (where to
   target), not a guaranteed "more artist."
2. **OFT/BOFT (orthogonal finetuning)** - multiplicative angle-preserving updates preserve hyperspherical
   energy -> retain base ability, reduce forgetting. Mechanistically DIFFERENT from magnitude; our
   artifact-when-pushed failure is consistent with base-corruption that OFT prevents. (OFT 2306.07280,
   BOFT 2311.06243.) CAVEATS: author-biased comparatives, ~3x slower, more params, anti-forgetting !=
   more identity, image-diffusion only, needs engine plumbing.
3. AdaLoRA SVD importance pruning - fallback.

REFUTED (0-2): "cross-attention carries identity capacity" - NO evidence. => layer targeting must be
MEASURED empirically on ACE-Step (-> Fisher calibration), NOT assumed from image-model priors.

DOMINANT CAVEAT: ALL evidence is image-diffusion / NLP - NONE on flow-matching, audio/music, LoKr, or
voice timbre. Pure extrapolation; ear-validate everything. Effect sizes MODEST (match-uniform), so no
silver bullet here. GAP: research found NO verified claims on lever #4 DATA (sample count 20->50->100,
augmentation, captions) or #5 loss-weighting (min-SNR/EMA) - those are UNRESEARCHED and, given our
magnitude failures + only 21 tracks, DATA is plausibly the bigger un-pulled lever (Tarja-era Nightwish
= ~40+ songs available, we used 21). NEXT-LEVER OPTIONS: (a) expand data to ~40 Tarja tracks + caption
rework (controllable, unresearched-but-plausible, needs re-upload/preprocess); (b) Fisher calibration
as diagnostic+lever (needs wrapper plumbing); (c) OFT/BOFT (bigger build, modest/biased evidence);
(d) a 2nd capped research pass on the data+recipe gap. All judged by ear [[lora-goal-full-artist-sound]].

## 13q. Battle Beast dim-128 / 100ep run IN FLIGHT (2026-06-09)

First DATA-lever test (§13p) + the user's overtraining hypothesis, combined. Run:
crucible_battlebeast/train_20260609-112643__lokrv2_100ep_prodigy_dim128_cfg0.1 (direct to engine,
fresh xl-base). Dataset: crucible_battlebeast = 40 tracks (vs nightwish_tarja's 21), data gate PASSED
(40/40 lyrics+bpm+key, clean LM-prose captions incl. "powerful high belting female vocals"=Noora,
tensors preprocessed/reused). Config = champion v2/Prodigy recipe at dim 128: prodigy / constant /
lr1.0 / cfg_ratio0.1 / factor8 / dim128 / alpha128 (scale 1.0) / attention_type both / attn+mlp /
100ep / save_every10. ~201s/epoch (40 tracks ~2x the 21-track ~110s) -> ~5.5h, ep3 loss 1.02.
HYPOTHESIS: dim-128 underperformed on 21 nightwish tracks (artifacts >0.7, no extra artist, §13o);
2x data + fewer epochs (100 vs 150) may give the higher capacity enough to fill WITHOUT the breakup.
Confounded (new artist + more data + dim128 + 100ep all at once) so not a clean ablation - judge by
EAR on the full-artist bar [[lora-goal-full-artist-sound]] vs prior battlebeast adapters; sparse-ladder
the ep checkpoints if final is hot.

RESULT (done 2026-06-09): final train loss 0.512 (min 0.465); curve first25% 0.87 -> mid 0.72 ->
last25% 0.59. Fit LESS HARD than the nightwish dim-128/150ep run (last-quarter 0.40) - expected from
2x data (40 vs 21) + fewer epochs (100 vs 150). PREDICTION: cooler deltas -> should break up LESS at
high strength than the nightwish dim-128 (artifacted >0.7, §13o). Saved FINAL (loadable) + per-epoch
ckpts on box. EAR TEST PENDING (full-artist bar [[lora-goal-full-artist-sound]]): load final, judge at
0.6-0.8 - (1) does it stay clean at 0.7-0.8 (the data+epochs hypothesis), (2) does it carry MORE of the
specific Battle Beast sound (Noora's vocals + band) than the 21-track nightwish adapters carried
Nightwish (the DATA lever)? Compare vs any prior battlebeast adapter on box.

## 13r. Beast in Black dim-64/100ep run IN FLIGHT (2026-06-09)

Biggest CLEAN dataset yet + the DATA lever on a male-fronted artist. Run:
crucible_beastinblack/train_20260609-212313__lokrv2_100ep_prodigy_dim64_cfg0.1. Champion recipe:
prodigy / constant / lr1.0 / cfg_ratio0.1 / factor8 / dim64 / alpha64 / attn+mlp / 100ep / save_every10.
~200s/epoch (39 tracks) -> ~5.5h, ep1 loss 1.14.
DATASET BUILD (full re-pipeline this session): 2 local Downloads folders (old 20 + new "All-songs" 40)
-> deduped to 40 (new is a superset) -> cleared+rebuilt crucible_beastinblack -> uploaded 40 (LRCLIB
lyrics) -> RESTART#1 -> scan -> autolabel xl-base+4B LM (merge SKIPPED = clean LM prose, no Last.fm
genre-soup; LORA_DIT_MODEL changed sft->base) -> CAPTION FIX: 8 LM "female lead" mislabels (Yannis's
high tenor) -> male + male reinforced, all 40 verified safe via full-field PUT (lyrics/bpm/key intact)
-> save -> preprocess -> 39 tensors (1 = "Battle Hymn" failed to encode + was silently skipped).
NOTE 2026-06-10: the "6.9min > length cap" reason was WRONG (unverified guess) -- there is NO hard
duration cap (Ghost Love Score ~10min trained fine; Battle Hymn encoded fine on the dim128 re-run's
preprocess -> 40 tensors). The engine SILENTLY SKIPS any track that fails to encode and still reports
"completed" with fewer tensors. Most likely cause of the one-off skip = transient VRAM/OOM on the
SINGLE LONGEST track (biggest latent, first to OOM when VRAM tight per the stickiness bug
[[engine-fresh-boot-for-lora]]); unproven retroactively (no engine log). -> RESTART#2 -> train.
EAR TEST PENDING: full-artist bar [[lora-goal-full-artist-sound]]
(Yannis's voice + the band), judge over MULTIPLE gens per setting (single-gen A/B unreliable
[[lora-scale-clean-seed-nondeterministic]]). If dim64 not enough, retry dim128.

RESULT (done ~5.5h, 2026-06-09): final train loss min 0.560; curve first25% 0.86 -> mid 0.75 ->
last25% 0.67 (less aggressive fit than the dim-128 runs ~0.40-0.59 = expected from dim64 + 39 tracks,
generally less overfit). Saved FINAL (loadable) + per-epoch ckpts on box. EAR TEST PENDING: load final,
judge at 0.5-0.8 over 2-3 gens/setting on the full-artist bar (Yannis's male tenor + the band/synth);
does the bigger CLEAN dataset (39 vs 21 nightwish) carry more of the SPECIFIC artist? If not enough,
dim128 next.

### §13s: Beast in Black dim-128 + band-name-trigger experiment (2026-06-10)
EAR VERDICT on §13r dim-64: messy at scale 0.8+, NO sign of Yannis's voice (full-artist bar FAILED).
-> Two-lever retry: (1) DIM 64->128 for capacity to carry the timbre, (2) NEW experiment = a SHARED
TRIGGER TOKEN. User's idea: line up generation to training by putting the band name in BOTH the
training captions AND the song prompt (we'd previously DROPPED band names from style captions per
[[lokr-style-fidelity]]; this deliberately reverses that to test if an explicit anchor helps).
Mechanics: prepended "Beast in Black. " to all 40 captions via engine PUT (sample_idx + all 9 fields
re-included, lyrics/bpm/key verified intact); since captions are baked into tensors, RE-PREPROCESSED
(no re-upload/scan/autolabel) on xl-base -> this time 40 tensors (Battle Hymn encoded; see
[[preprocess-silent-skip]] - last run's 39 was a silent encode skip, NOT a length cap). Heart of Winter
song tags rewritten to mirror captions + lead with "Beast in Black" trigger; dropped the "clean/ringing
head voice/controlled vibrato" that fought the gritty-tenor training distribution. Decision: train on
all 40 (more genuine artist data; run already multi-lever so +1 track is noise).
RUN (in flight, ~5.5h): train_20260610-094008__lokrv2_100ep_prodigy_dim128_cfg0.1. Champion v2 recipe
at dim128: prodigy / constant / lr1.0 / cfg_ratio0.1 / factor8 / dim128 / alpha128 / attn+mlp
(q,k,v,o,gate,up,down) / 100ep / save_every10. Gated on xl-base loaded (/health, not /v1/health).
10 steps/epoch, ep1 loss 1.177 (stable at lr1.0 = Prodigy engaged, AdamW would NaN).
RESULT (done, 2026-06-10): loss 0.855 -> 0.724 -> 0.564, min 0.404 (deeper fit than dim64's
0.86/0.75/0.67/0.560 = used the extra capacity). EAR VERDICT (user, on XL Base - user ALWAYS auditions
on base [[audition-lora-on-training-base]], so dim64 "no Yannis" was real not a base artifact):
"definitely captured more of Yannis' voice", still finding where it breaks up on the strength scale.
WIN = dim128 + band-name trigger both helped.

### §13t: alpha bump (more-capture lever) (2026-06-10)
Next lever toward [[lora-goal-full-artist-sound]]: alpha 128 -> 256 (effective SCALE 1.0 -> 2.0), single
variable off the §13s champion, reuses the same 40 tensors (no re-preprocess). Rationale: our earlier
AdamW-era A/B found scale 2.0 captured MORE than 1.0, and upstream high_quality.json uses alpha=2xdim;
caveat = Prodigy sets its own effective LR (D-adaptation) so the old gain may not transfer 1:1 - hence
test by ear. Tradeoff: stronger adapter likely breaks up at a LOWER slider (audition ~0.4-0.5).
RUN (in flight, ~6h @ ~220s/epoch): train_20260610-162556__lokrv2_100ep_prodigy_dim128_alpha256_cfg0.1.
All else identical (prodigy/constant/lr1.0/cfg0.1/factor8/dim128/100ep/save_every10/attn+mlp). Gated
xl-base. ep1 loss 1.177 (same as alpha128 ep1; divergence shows later). EAR TEST PENDING: more Yannis
than alpha128 at a usable scale? If yes, next swings = more epochs (200-300, audition mid ckpts) /
voice-specific captions / dim256 / DoRA. If it just breaks up sooner with no gain, revert to alpha128.
RESULT (done, 2026-06-10): loss curve NEARLY IDENTICAL to alpha128 - 0.857/0.739/0.567, min 0.405,
final 0.444 (vs alpha128 0.855/0.724/0.564/0.404). Doubling alpha barely moved training = PRODIGY
ABSORBED IT: its D-adaptation re-estimates effective LR, so a higher alpha (effective scale) gets
compensated -> near-identical dynamics. Confirms the earlier caveat (scale-2.0 AdamW-era gain does NOT
transfer under Prodigy). STRONG PRIOR that by ear alpha256 ~= alpha128 or just hotter/no-gain (= §13o
magnitude-knob dead-end again). EAR TEST PENDING: audition alpha256 FINAL on XL Base at LOWER strength
(~0.4-0.5, scale 2.0) vs the alpha128 sweet spot. If ~same/no gain -> magnitude levers are EXHAUSTED;
next PAID run = LoKr dropout 0.1 (regularizes the SHAPE of the deltas to widen the usable window, never
once tried; v2 route has native `dropout` field, default 0.0) per §13g Tier-2. Also free: ladder
alpha256 mid-checkpoints (ep40/60/80) by ear before any new run.
EAR VERDICT (2026-06-10, user): "seems very similar" to alpha128, and at higher strength "it feels
like it's trying to cram too much in at the same time and it gets messy". = CONFIRMS the prediction:
alpha (a magnitude knob) added nothing (Prodigy absorbed it); the high-strength break-up is over-geared
brittle co-adapted deltas. MAGNITUDE LEVERS NOW FULLY EXHAUSTED by ear (lr, dim, alpha all = heat not
artist, across nightwish + beastinblack). DECISION: champion stays dim128/alpha128 (094008); next paid
run = REGULARIZATION = LoKr dropout 0.1.

### §13u: LoKr dropout (regularization lever) (2026-06-10)
First-ever non-zero dropout run. SINGLE variable off the §13s champion (094008 dim128/alpha128):
add `dropout:0.1` (v2 route native field, was 0.0 every run to date). Rationale: the "crams too much in
-> messy at high strength" symptom = over-geared brittle deltas; dropout forces the adapter to learn a
redundant/distributed (not co-adapted-spiky) representation -> should WIDEN the usable strength window
WITHOUT cutting capacity (unlike lowering dim/alpha). Built for exactly this in the 2026-06-07 patch but
never used (pivoted to v2, then chased capacity/data/caption/alpha levers). Config: prodigy / constant /
lr1.0 / cfg_ratio0.1 / factor8 / dim128 / alpha128 / dropout0.1 / 100ep / save_every10 / attn+mlp,
reuse same 40 tensors. CAVEAT: dropout reduces overfit, so if the gap were UNDERfit it'd hurt - but our
symptom is over-gearing, the opposite. EAR TEST PENDING: does Yannis hold at HIGHER strength (window
widens) vs the champion? Judge same full-artist bar.
ATTEMPT 1 ABORTED (2026-06-10): fired dim128/alpha128/dropout0.1/200ep but caught via
[[verify-feature-engaged-not-just-ran]] that DROPOUT WAS A SILENT NO-OP - the v2 route
(`train_api_lokr_v2_start_route.py`) DECLARED the `dropout` field but `_build_v2_configs` never
passed it into `LoKRConfigV2(...)` (only cfg_ratio was plumbed). Proof: `/v1/training/status` config
block had NO dropout key + no "LoKr dropout config:" log line. STOPPED at ep1 (minimal waste). FIX
shipped (commit c9f23bc, patches/engine-2026-06-08 + README): pass `dropout=request.dropout` into
LoKRConfigV2 (field inherited from the 06-07-patched LoKRConfig; `inject_lokr_into_dit` already reads
it -> LyCORIS) + surface dropout in the status echo. REQUIRES box redeploy of the route file +
engine restart. ATTEMPT 2 PENDING after redeploy: re-fire + VERIFY dropout:0.1 shows in status config
AND the engine log prints "LoKr dropout config: dropout=0.1" BEFORE trusting the run.
ATTEMPT 2 ABORTED (2026-06-10): deployed the 06-07 dropout machinery (was NEVER deployed - HELD per
its README, not git-reverted; user confirmed no git pull) + the v2 route setattr fix. dropout=0.1 then
reached LyCORIS (log: "Use Dropout value: 0.1" + "LoKr dropout config: dropout=0.1") BUT the console
spammed `[WARN]LoHa/LoKr haven't implemented normal dropout yet.` x352 = NORMAL DROPOUT IS A NO-OP FOR
LoKr in LyCORIS. VERIFIED in lycoris/modules/lokr.py: `dropout` is stored+warned+ignored (~L200); only
`rank_dropout` (~L375, drops rank components/step) + `module_dropout` (~L544, skips modules) are
IMPLEMENTED for LoKr. Stopped at ep~10. FIX: v2 route now also plumbs rank_dropout + module_dropout
(only the route file needs re-copying; configs.py/lokr_utils.py already deployed + read all 3 via
getattr). ATTEMPT 3 = send **rank_dropout:0.1** (NOT dropout) - the real LoKr regularizer. Verify NO
"[WARN] normal dropout" spam + "LoKr dropout config: ... rank_dropout=0.1" before trusting.
LESSON: "the feature ran" needed THREE layers of verification - route set it (status echo) -> reached
LyCORIS (log line) -> LyCORIS actually IMPLEMENTS it (no warning). [[verify-feature-engaged-not-just-ran]]
ATTEMPT 3 ABORTED (2026-06-11): rank_dropout:0.1 CRASHED at step 0 - `Expected all tensors on same
device, cuda:0 and cpu`. VERIFIED in lycoris/modules/lokr.py:376 - rank_dropout builds `drop =
torch.rand(weight.size(0)).to(dtype)` = CPU tensor (only dtype cast, NOT device), then `weight *= drop`
(line 380) vs a cuda weight = device-mismatch BUG in this LyCORIS version's rank_dropout (fires even on
the disabled time_embed module since get_weight runs every forward). module_dropout is device-safe (its
check is a CPU-scalar compare; with rank_dropout=0 the buggy block is skipped). DECISION (user): run
module_dropout NOW, do the proper rank_dropout LyCORIS fix (1-line: `torch.rand(..., device=
weight.device)`) TOMORROW.
ATTEMPT 4 RUNNING (2026-06-11 ~00:07, ~12h @ 200ep): train_20260611-000703__lokrv2_200ep_prodigy_
dim128_a128_moduledrop0.1_cfg0.1. SINGLE variable off champion = module_dropout:0.1 (dropout/rank_drop
both 0). VERIFIED ENGAGED: config echo module_dropout=0.1 + cleared step 0 (where rank_dropout crashed)
+ ep1 step10 loss 1.177 no error. module_dropout drops whole adapter modules ~10%/step = coarser than
rank_dropout but trains the adapter to work when partly absent (~ graceful degradation across inference
strength). EAR TEST PENDING (full-artist bar): does the usable strength window WIDEN vs champion? TODO
TOMORROW: patch LyCORIS rank_dropout device bug -> A/B rank_dropout 0.1 vs this module_dropout run.
RESULT (done 2026-06-11, full 2000 steps): curve 0.819 -> 0.652 -> 0.559, min 0.398, final 0.518.
ENCOURAGING SIGNATURE: despite 2x epochs (200 vs champion 100) it did NOT overfit to a lower floor -
last25% 0.559 / min 0.398 ~= the 100ep champion (0.564/0.404), vs un-regularized dim128 that dove to
~0.40 at 100ep = module_dropout held the fit back from over-cramming (regularization working). Higher
final (0.518) vs min (0.398) = module_dropout per-step noise (random modules dropped each step). Saved
FINAL + per-epoch ckpts (save_every10). EAR TEST PENDING: audition FINAL on XL Base - does the usable
strength window WIDEN (Yannis holds past ~0.6, no cram/messy breakup) vs champion? If final over-noisy,
ladder ep~100/140/180. If window widens -> regularization IS the fix -> do proper rank_dropout tomorrow.
EAR VERDICT (2026-06-11, user): BIG WIN. module_dropout "definitely widened the usable window" + the
music is "more Beast in Black too" = regularization is confirmed the right lever (over-gearing was the
breakup cause, as diagnosed §13g). REMAINING GAP = Yannis's specific VOICE: "not bad but not quite
right" (band/feel/window all good now, timbre not nailed). So module_dropout 0.1 + dim128 is the new
champion. NEXT QUESTION (user): is more dim worth it? 64->128 was a big win on THIS 40-track set (unlike
the data-starved 21-track nightwish where dim128 just fried, §13o) so the capacity curve here has NOT
flattened -> dim256 worth a shot, esp. now dropout de-risks pushing capacity. Honest caveat: the voice
gap may instead be DATA/caption (captions say generic "gritty tenor", never pin HIS timbre) or a base-
model vocal-resolution ceiling ("inspired-by not is" across ALL artists).

### §13v: LyCORIS rank_dropout fix + dim256 capacity test (2026-06-11)
PROPER FIX first: patched the LyCORIS rank_dropout device bug (patches/engine-2026-06-11/lokr.py:376,
`torch.rand(..., device=weight.device)`; 1-line, diff-proven, box==upstream-main per traceback line #s;
site-packages file -> reverts on any lycoris reinstall). VERIFIED working: dim256 + rank_dropout:0.1
CLEARED step 0 (where it crashed pre-fix), ep1 step10 loss 1.078.
RUN IN FLIGHT (2026-06-11 ~22:07, ~12h @ 200ep): train_20260611-220741__lokrv2_200ep_prodigy_dim256_
a256_rankdrop0.1_cfg0.1. TWO changes vs the §13u module_dropout/dim128 winner (dim 128->256 AND
module_dropout->rank_dropout) - deliberate "best adapter" push, not a clean ablation: rank_dropout is
the FINER regularizer (drops rank components not whole modules) which pairs best with MORE capacity for
a complex target like voice. alpha=dim=256 (scale 1.0 held). EAR TEST PENDING: does the VOICE
specifically improve (his timbre, not just band/window) vs the module_dropout/dim128 champion? If yes ->
capacity still the lever (consider 384/512). If just hotter/same voice -> gap is NOT capacity -> pivot
to caption-pinning his timbre (re-preprocess) or Fisher-info targeted ranks (§13p).
RESULT (done 2026-06-11, full 2000 steps): curve 0.796 -> 0.549 -> 0.430, min 0.286, final 0.346 =
fit MUCH HARDER than the dim128/module_dropout champion (last25% 0.43 vs 0.56, min 0.286 vs 0.398).
Driven by BOTH changes: dim256 more capacity to fit + rank_dropout is a GENTLER regularizer than
module_dropout (drops ranks not whole modules) so holds back less. Harder fit is NOT a quality verdict
(more capture OR more overfit/hotter - back toward breakup). rank_dropout ran 2000 steps with zero
device crashes = LyCORIS fix fully validated. Saved FINAL + per-epoch ckpts. EAR TEST PENDING: audition
on XL Base, sweep 0.6/0.7/0.8 on Neon Crusader + Heart of Neon. KEY Qs: (1) does the VOICE improve vs
champion? (2) did the usable WINDOW hold (rank_dropout still regularizing enough at the higher fit) or
get hot again? If hotter, ladder mid ckpts and/or A/B rank_dropout vs module_dropout at dim256.
EAR VERDICT (2026-06-11, user): SLIGHT improvement, close; a little keener to break up over 0.8 (the
harder fit showing) but still produced a decent 0.8 take. CRUCIALLY: "doesn't really improve the voice
in most gens." => CAPACITY IS TAPPED OUT FOR THE VOICE: dim64->128 helped the voice (big), 128->256 did
NOT. Band/feel/window are good across the recent adapters; the residual gap is Yannis's SPECIFIC TIMBRE.
Same "inspired-by not is" voice verdict we got on EVERY artist (nightwish, battlebeast) = strong hint of
an ACE-Step vocal-RESOLUTION CEILING, not a tuning problem. DECISION: stop the capacity/dropout knobs
(they're polish now). LAST accessible LoRA lever for the VOICE = CAPTION-PINNING his specific timbre
(§13w): the 40 captions currently describe his voice GENERICALLY and INCONSISTENTLY (per-track LM prose
varies raspy/clean/operatic/scream) so the adapter learns a fuzzy CLASS of gritty tenor, not HIS
fingerprint. Pin a CONSISTENT specific timbre-core across all 40 (keep per-track delivery variation),
re-preprocess, retrain champion recipe. HONEST: if that doesn't crack it, the specific voice is beyond
ACE-Step+LoRA -> it's a MODEL-CHOICE problem (the LeVo path, RESEARCH.md) or accept "inspired-by".

### §13w: GENERALIZATION test - apply the developed method to Avantasia (2026-06-11)
User decision: rather than chase the Yannis voice ceiling further, prove the method generalizes to a
2nd artist. Avantasia chosen (old build "worked quite well sometimes"; clean before/after to hear).
KEY: Avantasia is a MULTI-SINGER metal opera (Tobias Sammet anchors, many guests incl. FEMALE - Floor
Jansen on Moonglow etc.) = a BAND/PRODUCTION-style target, NOT a single-voice one -> dodges the
single-timbre ceiling that walled Yannis/Tarja/Noora; plays to the method's proven strengths
(production/feel/window). Existing crucible_avantasia = 50 tracks but OLD captions (Last.fm genre-soup
prefix + LM prose, NO band trigger). APPLIED THE CURRENT METHOD: stripped the Last.fm soup
(caption.partition('. ')[2], verified 50/50 cleanly strippable) + prepended "Avantasia. " trigger, via
full-field engine PUT (sample_idx + all 9 fields; lyrics/bpm/key verified intact on sample 0 then all
50). KEPT the ~6 female/soprano guest refs in the prose (CORRECT for Avantasia - do NOT blind-flip like
the Yannis male-mislabel case). save -> preprocess on xl-base -> 50/50 tensors (no skips; Mac status
route choked on the LM-restore phase but engine + load_tensor_info confirmed done; USER verified tensor
files have current timestamps = definitively NEW, not the old build's). RUN IN FLIGHT (2026-06-12
~14:10, ~12-15h @ 200ep/50 tracks): train_20260612-141026__lokrv2_200ep_prodigy_dim256_a256_rankdrop0.1
_cfg0.1 (the §13v BiB recipe carried forward per user: dim256/alpha256/rank_dropout0.1/prodigy/constant/
lr1.0/cfg0.1/factor8/attn+mlp). Gated xl-base, config echo dim256+rank_dropout0.1. EAR TEST PENDING:
does the current full method beat the old avantasia build (factor-1/dim64/alpha128/discrete/lr0.01/
AdamW/noisy-captions)? Esp. production/feel/window; voice ceiling should bite less here (multi-singer).
RESULT (done 2026-06-12, full 2600 steps / 13 per-epoch x 200): curve 0.828 -> 0.642 -> 0.549, min
0.415, final 0.601 (final>min = rank_dropout per-step noise). Fit LESS hard than BiB's dim256
(0.43/0.286) = expected from 50 tracks (more data, less overfit) vs BiB's 40. Saved FINAL + per-epoch
ckpts. A/B READY (both in picker): NEW train_20260612-141026__...dim256_a256_rankdrop0.1 vs OLD
train_20260605-234720__lokr_150ep_discrete. EAR TEST PENDING: XL Base, sweep 0.6-0.8, 2-3 gens each
(ACE not seed-reproducible [[lora-scale-clean-seed-nondeterministic]]). Q: does the full developed
method clearly beat the old build on production/feel/window + Tobias/guest voices? = the generalization
verdict for everything built this session.

### §13x: Fisher / gradient-sensitivity - built, probed, SHELVED (2026-06-12)
Wired the engine's unused run_estimation into HTTP (patches/engine-2026-06-12: /v1/training/estimate
route + estimate.py MLP-coverage patch), deployed live via the box fs API. VRAM TRAP: run_estimation
sets requires_grad on the FULL BASE projection weights (q/k/v/o + the MLP gate/up/down I added) and does
a real backward -> GRADIENT STORAGE the size of those matrices (gigabytes), which gradient-checkpointing
does NOT reduce (it only saves activations). First runs maxed the 3090 (23GB VRAM + 10-15GB shared,
crawled). My configure_memory_features+offload_non_decoder fix did NOT help (wrong target: activations,
not grads) and the MLP addition made it WORSE (FFN up/down are the biggest matrices). Training stays at
~14GB ONLY because it grads the tiny LoKr adapter, not full base weights. A VRAM-safe estimate needs a
CHUNKED rewrite (grad a subset of modules per pass, accumulate) - real work.
DIAGNOSTIC obtained anyway (3-batch run completed in 124s via shared-RAM spill, then unloaded): for
crucible_beastinblack, sensitivity concentrates in CROSS-ATTENTION q_proj at mid-late layers (21,30,19,
17) + layer0 self_attn v_proj = the artist lives in the CONDITIONING paths, not the MLP. Noisy (n=3).
DECISION (user-aligned): SHELVE the targeted-train pursuit - §13p says targeted ~= uniform at equal
budget, and the voice gap is an ACE ceiling Fisher won't fix. Route left DORMANT (do NOT call without
the chunked rewrite - it maxes the box). estimate.py edits kept (harmless dormant). Revisit only if a
cheap chunked estimator is worth building.

## 14. Sources
RESEARCH §18 (+ its sources): ACE-Step-1.5 `docs/en/LoRA_Training_Tutorial.md`, `train.py`, `acestep/training_v2/cli/args.py`, `acestep/api/train_api_models.py`, training/lora route files, `scripts/lora_data_prepare/`, Side-Step toolkit. Live verification: `192.168.1.201:8001/openapi.json` + status probes (2026-05-27).

§§11–13a sources:
- **ACE-Step paper** [arxiv 2506.00045](https://arxiv.org/pdf/2506.00045) — training-time timestep sampling spec, flow-matching objective
- **Curriculum sampling** [arxiv 2603.12517](https://arxiv.org/pdf/2603.12517) — logit-normal vs uniform trade-offs (for §13a.3 follow-up reading)
- Engine repo files (verified 2026-05-30): [`acestep/training/configs.py`](https://github.com/ace-step/ACE-Step-1.5/blob/main/acestep/training/configs.py), [`acestep/training/trainer.py`](https://github.com/ace-step/ACE-Step-1.5/blob/main/acestep/training/trainer.py), [`acestep/training/lokr_utils.py`](https://github.com/ace-step/ACE-Step-1.5/blob/main/acestep/training/lokr_utils.py), [`acestep/training_v2/configs.py`](https://github.com/ace-step/ACE-Step-1.5/blob/main/acestep/training_v2/configs.py), [`acestep/training_v2/presets/recommended.json`](https://github.com/ace-step/ACE-Step-1.5/blob/main/acestep/training_v2/presets/recommended.json), [`acestep/training_v2/presets/high_quality.json`](https://github.com/ace-step/ACE-Step-1.5/blob/main/acestep/training_v2/presets/high_quality.json), [`acestep/api/train_api_models.py`](https://github.com/ace-step/ACE-Step-1.5/blob/main/acestep/api/train_api_models.py), [`acestep/api/train_api_lokr_start_route.py`](https://github.com/ace-step/ACE-Step-1.5/blob/main/acestep/api/train_api_lokr_start_route.py)
- **Side-Step community trainer** [github.com/koda-dernet/Side-Step](https://github.com/koda-dernet/Side-Step) — LoHA/OFT/Fisher info reference, Pinokio install path
- **HF LoRA collections** (cataloged): [woctordho/ACE-Step-v1-LoRA-collection](https://huggingface.co/woctordho/ACE-Step-v1-LoRA-collection), [David-A-Amoo/ACE-Step-1.5-Naija-Legacy-Rhythms-LoRA-v1](https://huggingface.co/David-A-Amoo/ACE-Step-1.5-Naija-Legacy-Rhythms-LoRA-v1) — both PEFT format targeting 2B, incompatible with our XL 4B without conversion
- **Tutorials** (for cross-reference): [ACE-Step LoRA Training Tutorial (official)](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/LoRA_Training_Tutorial.md), [Training a Custom LoRA — DeepWiki ACE-Step-1.5](https://deepwiki.com/ace-step/ACE-Step-1.5/10.3-training-a-custom-lora)
- §13 baseline stats **measured live on `192.168.1.201:8001` 2026-05-29/30** (200-ep / 35-track run, 8h 44m wall-clock, best at epoch 105)
- §13a.6 val-loss-is-misleading finding measured live 2026-05-30 (user A/B listening test)
