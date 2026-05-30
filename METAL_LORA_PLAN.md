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

Patched copies live in `patches/engine-2026-05-29/` — drop-in over the box's `<engine>\acestep\...` paths and restart `run_acestep_api.bat`. README in that folder has the file-to-destination map. **Don't upstream as PRs yet** — applying locally first to see if they actually help adapter quality (per [[dont-propagate-guesses]]: only PR what we've measured working).

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
  - **Win**: continuous-50ep best ckpt sounds at-least-as-good as discrete-200ep best by ear, AND CLAP fitness curve is comparable or higher. → Update [[engine-patches]] memory, flip continuous to default in the Mac route, run a proper 100-ep production run on the same data.
  - **Tie**: no audible difference, CLAP curves indistinguishable. → Keep the patch toggleable but don't flip default; flag for revisit on a future dataset (bigger or different subgenre) where divergence might emerge.
  - **Loss**: continuous is measurably worse. → Document the negative result, revert. Look at v2's `cfg_ratio` (CFG dropout) and Fisher-info adaptive ranks as the next lever; or shift to Side-Step trial.

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

## 14. Sources
RESEARCH §18 (+ its sources): ACE-Step-1.5 `docs/en/LoRA_Training_Tutorial.md`, `train.py`, `acestep/training_v2/cli/args.py`, `acestep/api/train_api_models.py`, training/lora route files, `scripts/lora_data_prepare/`, Side-Step toolkit. Live verification: `192.168.1.201:8001/openapi.json` + status probes (2026-05-27).

§§11–13 sources: ACE-Step paper [arxiv 2506.00045](https://arxiv.org/pdf/2506.00045) (training-time timestep sampling spec); engine repo files [`acestep/training/configs.py`](https://github.com/ace-step/ACE-Step-1.5/blob/main/acestep/training/configs.py), [`acestep/training_v2/configs.py`](https://github.com/ace-step/ACE-Step-1.5/blob/main/acestep/training_v2/configs.py), [`acestep/training_v2/presets/recommended.json`](https://github.com/ace-step/ACE-Step-1.5/blob/main/acestep/training_v2/presets/recommended.json) (verified 2026-05-30); [Side-Step trainer](https://github.com/koda-dernet/Side-Step) (LoHA/OFT/Fisher info reference); §13 stats measured live on `192.168.1.201:8001` 2026-05-30.
