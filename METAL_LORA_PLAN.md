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

## 11. Sources
RESEARCH §18 (+ its sources): ACE-Step-1.5 `docs/en/LoRA_Training_Tutorial.md`, `train.py`, `acestep/training_v2/cli/args.py`, `acestep/api/train_api_models.py`, training/lora route files, `scripts/lora_data_prepare/`, Side-Step toolkit. Live verification: `192.168.1.201:8001/openapi.json` + status probes (2026-05-27).
