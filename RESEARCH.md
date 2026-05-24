# MusicGen App — Model & Tooling Research

_Research current as of May 2026. Re-verify model versions/licenses before committing — this space moves fast._

## 1. What we're building

A music-generation app focused on **rock and metal subgenres** — heavy, power, symphonic, and folk metal. Two core modes:

- **Prompt → music**: generate a track from a text description.
- **Base track → restyle**: take an existing audio track and re-render it in a different genre/style from a prompt (e.g. a folk melody rendered as symphonic metal).

**Priority note:** death metal / harsh growls are *lower* priority. Power / symphonic / folk metal — which lean on clean & operatic vocals plus orchestral textures — are the main targets. This matters because growls are the single hardest thing for open models, and de-prioritizing them lets us pick the faster, cleaner engine.

**Vocals are produced as a separate step from the instrumental** (then mixed). This is a deliberate quality decision — see §5. It sidesteps the hardest open-model problem (intelligible vocals over dense distorted guitar) and unlocks dedicated singing-voice tools.

### Architecture

```
Mac (this repo)                          Windows PC + RTX 3090 (24GB)
─ UI + orchestration                     ─ ComfyUI (GPU backend)
─ job submit / library     ── HTTP/WS ──▶ ─ ACE-Step (primary model)
─ audio playback                         ─ stem separation / analysis tools
```

The Mac is a thin client. The Windows PC does all the GPU work and exposes an HTTP + WebSocket API that the Mac drives over the LAN.

## 2. The honest reality for metal

Every open model is weakest exactly where metal is hardest: sustained high-gain distorted guitar plus aggressive vocals. **None reliably one-shot a finished metal track.** The working model is **generate-then-curate** — produce batches, pick the good takes, and post-process (re-amp guitars, layer vocals). Power/symphonic/folk metal come out noticeably better than death/black/thrash, which is why de-prioritizing growls is a smart constraint.

## 3. Recommended model stack

| Role | Model | Why | License |
|---|---|---|---|
| **Primary engine** | **ACE-Step 1.5 XL (4B)** | <10s/song on a 3090, fits 24GB comfortably, full songs + lyrics, native ComfyUI support, built-in Cover/Audio2Audio restyle + stem separation. Strong on clean/operatic vocals + orchestral textures = ideal for power/symphonic/folk metal | Open weights (verify exact XL license file) |
| **Optional / watch** | **HeartMuLa** (oss-3B, Feb 2026) | Best-in-class lyric control, Apache 2.0 | Apache 2.0 |
| **Optional, only if growls ever matter** | **YuE** | Only open model that does genuine death growls/screams | Apache 2.0 |

**Demoted to optional:** YuE was the recommendation *only* for harsh growls. Since those are low priority and YuE is slow (~6 min compute per 30s audio on a 4090, worse on a 3090) and VRAM-heavy (short segments only on 24GB), it's not worth running day-one.

**Skip:**
- **MusicGen / AudioCraft** — no vocals, weights are CC-BY-NC (non-commercial), 2023-era quality.
- **Stable Audio Open** — no vocals/full songs, restrictive license. Still handy for short instrumental/SFX clips if ever needed.

### The one risk to validate first

ACE-Step's *older* Turbo/Base variants were explicitly criticized for failing on distorted-guitar content ("can't handle guitar music at all"). The new **XL 4B** (Apr 2026) claims better musicality but has no metal-specific listening reports yet. **Before building the full app, run a benchmark**: feed ACE-Step XL a set of power/symphonic/folk-metal prompts and confirm the distorted guitars and orchestration are usable. Everything downstream depends on this.

## 4. Base-track restyle pipeline

Best open path is **ACE-Step Cover / Audio2Audio**, using community-validated settings:

1. **Analyze the input** — Basic Pitch (melody → MIDI) + madmom (BPM, beat grid, key, chords) to auto-build an accurate prompt and lock tempo/key.
2. **(Optional) Stem-separate** — HT-Demucs `htdemucs_ft` for fast 4-stem; `htdemucs_6s` when you need a guitar stem; Mel-/BS-RoFormer for best vocal isolation. Lets you restyle stems independently and recombine.
3. **Restyle with ACE-Step Cover mode:**
   - **Cover Strength 30–50%** for big genre jumps (folk → metal); 50–70% moderate; 70–90% subtle.
   - **Explicit, instrument-rich prompts** beat vague tags: e.g. _"symphonic power metal, heavily distorted electric guitars, orchestral strings, double-bass drumming, soaring operatic male vocals, fast tempo, epic"_.
   - **Generate batches of 2–4** and pick the best (output is stochastic).

## 5. Vocal production (separate from the instrumental)

Generating vocals separately and mixing them in is the higher-quality path for metal. The key finding from research: **the genuinely excellent powerful/operatic/rock voices are mostly commercial** (Synthesizer V + HXVOC, ACE Studio) — the open-weight tools are good but skew toward soft pop/vocaloid timbre *unless* you steer them. The linchpin that fixes this is **voice conversion (RVC)**: power and emotion come from the *input performance and the target voice*, and RVC swaps the timbre onto it.

### Recommended open-weight vocal pipeline

1. **Get a vocal performance** with the right melody, timing, and energy. Two open options:
   - **ACE-Step `lyric2vocal` LoRA** — generates a vocal stem directly from lyrics, same ecosystem you're already running. Fastest; less precise pitch control.
   - **OpenUtau driving a DiffSinger voicebank** — a Windows GUI piano-roll where you author melody + lyrics with precise pitch/timing, and crank energy/tension params for power. More control; more setup.
2. **Convert the timbre with RVC** to a voice model trained on a powerful belted/operatic clean singer. This is what escapes "thin pop voice." RVC follows the input performance, so feed it an energetic take.
3. **Mix** the converted vocal stem against the ACE-Step instrumental with a standard metal vocal chain (compression, saturation/grit, doubling, reverb/delay). A lot of the perceived "power" comes from the mix, not just the model.

### Vocal tool options

| Tool | Role | License | Notes |
|---|---|---|---|
| **RVC** (Retrieval-based Voice Conversion) | Timbre conversion — the linchpin | MIT | One-click Windows installer, trains on 10–60 min of audio, runs easily on the 3090. Set this up first. |
| **OpenUtau + DiffSinger** | Lyrics+melody → controllable vocal | Apache 2.0 (banks vary) | Native Windows GUI; easy to run, custom voicebank training is the harder part. English banks exist but skew clean/pop. |
| **ACE-Step lyric2vocal LoRA** | Lyrics → vocal stem | Apache 2.0 | In-ecosystem, no extra install; prompt-driven style, less precise control. |
| **SoulX-Singer** (2026, zero-shot SVS) | Watch/evaluate | Verify | Prompt with a reference clip instead of training a voice; new, unproven on rock/operatic — test it. |
| **Synthesizer V + HXVOC / ACE Studio** | Commercial fallback for vocals | Commercial | If a fully-open constraint isn't required, by far the easiest route to genuinely powerful/operatic/rock vocals (HXVOC does belts, grit, even screams natively). |

**Caveat:** open-weight *and* powerful-operatic-out-of-the-box don't coexist well today. To hit it with open tools you must either train your own RVC/DiffSinger voice on a strong singer, or accept one commercial vocal tool. Note that individual voicebanks/voice models carry their own licenses even when the engine is permissive.

## 5b. AI Vocal Builder — compose-the-melody pipeline (2026 research)

Goal: instead of letting a model improvise the tune, **AI composes an explicit melody** from the song's structure + key, binds a note to each lyric syllable, then a singing engine sings it; RVC (or a zero-shot engine) sets the timbre; the mixer places it. Engine-agnostic by design (the app already does provider abstraction for LLMs and RVC drivers).

**Stage A — Melody composition (note-per-syllable, key/structure-aware).** Pluggable "composer brain":
- **Hybrid LLM + music-theory layer (chosen default):** Claude (when `ANTHROPIC_API_KEY` set) or local Gemma via `llm.py` proposes a melody as JSON (pitch + start + dur + syllable); a deterministic theory pass *repairs* it — snaps pitches to the key/scale, clamps range per section role (verse lower/narrower, chorus higher/hookier), quantizes timing to the BPM grid, enforces phrase cadences. A pure-algorithmic composer is the final fallback so output is guaranteed even with no LLM. Syllabification via `pyphen`; MIDI via `mido`.
- **Dedicated symbolic models (future engine options):** **SongComposer** (LLM specialized for lyric↔melody, reportedly > GPT-4 at lyric-to-melody; arXiv 2402.17645) and **MusicAIR** (algorithm-driven symbolic core that derives a full melodic score from lyrics with customizable key signatures; arXiv 2511.17323). Slot behind the same composer interface later.

**Stage B — Singing synthesis (sing the composed score).** Pluggable engine driver:
- **SoulX-Singer** (Soul AI Lab, Feb 2026, **Apache 2.0**; arXiv 2602.07803) — RECOMMENDED modern engine. Unified framework taking **score (MIDI notes)** *and* melody (F0) + lyrics; **zero-shot timbre cloning from a short reference clip** (no voicebank training); English/Mandarin/Cantonese; trained on 42k h; Python infer scripts (`example/infer.sh`, `infer_svc.sh`) + WebUI (`webui.py`/`webui_svc.py`) + HF Space; ships a MIDI editor. Zero-shot cloning can stand in for RVC. VRAM unspecified — verify it fits the 3090; runs on Windows next to ComfyUI. Drive via a thin API server (the `rvc_server.py` pattern).
- **DiffSinger / NNSVS** (Apache 2.0; `github.com/nnsvs/nnsvs`, `github.com/openvpi/DiffSinger`) — score + phonemes via **HTS labels derived from MusicXML or UST** (OpenUtau format). High control; needs per-voice **voicebanks** (English banks skew clean/pop) and more setup. Established and reliable.
- **Synth-guide + RVC (Mac-only, works today):** render the composed MIDI to a simple synthetic guide vocal (pitched tone w/ vibrato at the note times) on the Mac, then re-timbre with the existing RVC. No new GPU install — the fallback that makes the pipeline runnable immediately. Quality depends on how well RVC tracks a synthetic guide.
- **ACE-Step lyric2vocal** — in-ecosystem, but does not faithfully sing an exact composed melody (loose pitch control). Keep as a quick option.

**Stage C — Re-timbre (RVC, existing) + Stage D — Mix (existing).** SoulX's zero-shot cloning may let us skip RVC; keep RVC for the DiffSinger/synth-guide paths and for fine voice control.

**Pipeline:** Song Constructor song (lyrics + key + BPM + section roles) → Stage A melody → Stage B sing → Stage C/D timbre+mix. Stages A and the synth-guide+RVC path are Mac-only and verifiable now; SoulX-Singer & DiffSinger are code + Windows installer, user-activated.

**Host-engine API contract** (what a SoulX/DiffSinger Windows server must expose — `backend/voicegen.py` is the client; mirror the `rvc_server.py` pattern, run via the engine's own python next to ComfyUI):
- `GET /health` → 200 when ready (used for availability in the UI).
- `POST /synthesize` (multipart) — fields: `score` (JSON: bpm/key/duration/notes[{midi,start,dur,syllable,section}]), `lyrics` (text), `opts` (JSON), optional `reference` (WAV file, for SoulX zero-shot timbre) → returns `audio/wav` of the sung melody at the score's absolute times.
Config: set `soulx_host` / `diffsinger_host` in `app_config.json` (host:port) to light the engine up. TODO when installing: build each server against the real repo's inference entrypoints (SoulX `example/infer.sh`; DiffSinger via MusicXML/UST→HTS labels) and verify VRAM on the 3090.

## 5c. SoulX-Singer integration spec (VERIFIED from the repo, 2026-05-24)

Repo `github.com/Soul-AILab/SoulX-Singer` (Apache-2.0). Inference is reusable Python — `cli/inference.py` exposes `build_model(model_path, config, device, use_fp16)` and a `process()` loop; the core call is `model.infer(infer_data, auto_shift, pitch_shift, n_steps=config.infer.n_steps, cfg=config.infer.cfg, control, use_fp16)`. So a server loads the model once and calls `model.infer` per segment. **Output is 24 kHz WAV.**

**Inputs.** Two metadata dicts processed by `soulxsinger.utils.data_processor.DataProcessor(hop_size, sample_rate, phoneset_path, device).process(meta, wav_path)`:
- **prompt** = the reference voice: `prompt_wav_path` (mp3/wav) + prompt metadata (its own phoneme/duration/note/f0 transcription). Drives zero-shot timbre.
- **target** = the score to sing (our composed melody), `wav_path=None`.

**Target metadata schema** (list of segments; what we must generate). Per segment:
- `time`: `[start_ms, end_ms]` — positions the segment in the merged output.
- `duration`: space-separated **per-note seconds** (one value per token, incl. `<SP>` rests).
- `phoneme`: space-separated **per-note** tokens. English = `"en_" + "-".join(ARPABET phones)` for the word, e.g. `en_B-Y-UW1-T-AH0-F-AH0-L`; rests = `<SP>`. (G2P via `g2p_en`, an engine dep.) `DataProcessor.preprocess` splits each `en_…` token on `-`, prefixes `en_`, inserts `<BOW>/<EOW>/<SEP>` — so **one token = one whole word's phonemes** (multi-syllable words sit on a single note).
- `note_pitch`: space-separated **MIDI ints** per token (0 for `<SP>`).
- `note_type`: ints per token; `merge_phoneme` forces `<SP>`→1; sung notes carried through. Examples use mostly `2` with `3` at phrase ends — **use `2` for sung notes, `1` for rests** (refine after listening).
- `f0`: dense frame-level contour — **only for `control="melody"`; omit for `control="score"`** (we use score = MIDI).
- `text`: cosmetic — **not read by DataProcessor**; only phoneme/duration/note_pitch/note_type/(f0)/time matter.

**Score conversion (our melody → SoulX target).** Our composer is per-syllable but SoulX is **per-word**. So group a section's notes by word: pitch = the word's first note (or mean), duration = sum of its syllable durations; phonemize the word with `g2p_en`; insert `<SP>` (pitch 0, type 1) for inter-line gaps. Build one segment per Song section (or one per phrase), with `time` from the section's window. ⇒ Best done **server-side** (g2p_en + `phone_set.json` live in the SoulX env); the Mac sends the score with word grouping (add a `word` field to notes in `melody.py`).

**Reference voice.** Two options (support both — "more options"): (a) **bundle a default English prompt** (repo ships `example/audio/en_prompt.mp3` + `en_prompt.json`) so no input is needed; (b) **user reference clip** → run the repo's preprocess pipeline (`preprocess/pipeline.py` + `SoulX-Singer-Preprocess` models: BS-RoFormer separation, lyric+note transcription, f0) to make its metadata. RVC re-timbre stays available as a third path.

**Install / deps (heavy).** `pip install -r requirements.txt`: `torch==2.2.0`, `torchaudio==2.2.0`, `transformers==4.41.2`, `nemo_toolkit==2.6.1`, `funasr==1.3.0`, `g2p_en`, `g2pM`, `ToJyutping`, `sageattention`, `pyworld`, `librosa`, `numpy<2`, `gradio`, etc. CUDA. Models: `hf download Soul-AILab/SoulX-Singer --local-dir pretrained_models/SoulX-Singer` (+ `SoulX-Singer-Preprocess` for option (b)). **VRAM unconfirmed** — flow-matching + Llama-based; verify it fits the 3090's 24 GB at inference (and against ComfyUI's footprint).

**Server plan** (`backend/soulx_server.py`, run via SoulX's python like `rvc_server.py`): load model once; `GET /health`; `POST /synthesize` (score JSON + lyrics + opts + optional reference wav) → build target metadata (g2p + word grouping) → pick bundled/user prompt → `model.infer` per segment → merge → return 24 kHz WAV. `backend/voicegen.py` already points the `soulx` driver at `POST /synthesize`. **Open items to confirm on first run: `note_type` semantics, VRAM, English prompt quality on metal, and `auto_shift`/`pitch_shift` defaults.**

## 5d. DiffSinger integration spec (VERIFIED — via DiffSingerMiniEngine)

Don't drive raw `openvpi/DiffSinger`; use **`github.com/openvpi/DiffSingerMiniEngine`** — a small **HTTP inference server** (`server.py`, stdlib `http.server`, default port **9266**, 44.1 kHz out). It already does what we need, so our `diffsinger` driver speaks its **native API** (not our generic `/synthesize`):
- `GET /version`, `GET /models` (lists acoustic `.onnx` in `assets/acoustic`).
- `POST /rhythm` — `{notes:[{key:MIDI, duration:sec, slur:bool, phonemes:[ph…]}]}` (key 0 + `["SP"]` = rest) → `{phonemes:[{name,duration}]}` (phoneme-level durations from the rhythmizer ONNX).
- `POST /submit` — `{model, phonemes:[{name,duration}], f0:{timestep:0.01, values:[Hz…]}, speedup}` → `{token,status}` (async). **The acoustic model is MIDI-less: pitch comes from the explicit `f0` curve**, which we synthesize from our melody (piecewise-constant Hz per note at 10 ms timestep; optional glide/vibrato).
- `POST /query` `{token}` → status; `GET /download?token=…` → WAV.
Driver flow (in `voicegen.py`, Mac builds phonemes+f0): notes→`/rhythm`→f0 build→`/submit`→poll `/query`→`/download`.

**Phonemes are voicebank-specific.** MiniEngine maps via a **dictionary** (word→phones, e.g. `opencpop-strict.txt` for ZH). For English: install an **English DiffSinger acoustic ONNX voicebank + its English dictionary** (community `dsdict-en`), and G2P with `g2p_en` (ARPABET) → dictionary phones. Needs Mac-side `g2p_en` (light, also used for SoulX prep). **English banks skew clean/pop** — expect to train/pick a stronger one for metal.

**Install** (`DIFFSINGER-MINIENGINE_AUTO_INSTALL.bat`, runs on the 3090): clone MiniEngine; `pip install onnxruntime-gpu PyYAML soundfile`; download NSF-HiFiGAN ONNX vocoder (openvpi/vocoders `nsf-hifigan-v1`) → `assets/vocoder`, rhythmizer ONNX (openvpi/DiffSinger `v1.4.1`) → `assets/rhythmizer`, an English acoustic ONNX → `assets/acoustic`; set CUDA provider in `configs/default.yaml`; run `python server.py`. Set `diffsinger_host` in `app_config.json`. Lighter than SoulX (ONNX-runtime, no torch/nemo); **score control is native** (notes+f0). Trade-off vs SoulX: needs a voicebank (no zero-shot cloning) but cheaper to run.

## 6. Serving architecture

**Run ComfyUI natively on Windows as the GPU backend; drive it from the Mac over its HTTP + WebSocket API.** ComfyUI natively supports ACE-Step, has a built-in serial job queue (ideal for a single GPU), manages VRAM load/unload, and streams progress over WebSocket.

Job flow the Mac uses:
- `POST /prompt` — enqueue a workflow (returns a `prompt_id`)
- `GET /ws` — subscribe to live progress events
- `GET /history/{prompt_id}` — fetch result/status
- `GET /view?filename=...&type=output` — download the generated audio
- `POST /free` — unload models / free VRAM when switching

Later, optionally add a **thin FastAPI gateway** on the Mac to give a clean, stable API and to host extra models (YuE) behind a single-worker queue (Arq/asyncio, concurrency = 1). Not needed for v1.

---

## 7. Windows PC — install & run instructions

Everything below runs on the **Windows machine with the RTX 3090**. The goal is: get ComfyUI + ACE-Step running and reachable from the Mac.

> **Shortcut:** two auto-installers in this repo do most of this for you (verified May 2026):
> - `MUSICGEN-COMFYUI_AUTO_INSTALL.bat` — installs ComfyUI portable (v0.22.0) + ComfyUI-Manager, downloads ACE-Step 1.5 (Turbo AIO or XL), optionally Demucs, and creates a LAN launcher (`run_musicgen_lan.bat`, binds `0.0.0.0:8188`) so the Mac can reach it.
> - `RVC_AUTO_INSTALL.bat` — downloads + launches the bundled RVC voice-conversion package (own Python/ffmpeg/models, web UI on port 7897).
>
> The manual steps below are the fallback / explanation of what those scripts do.

### Prerequisites

1. **NVIDIA driver** — install the latest Game Ready / Studio driver from nvidia.com. (CUDA comes bundled with the PyTorch that ComfyUI installs; no separate CUDA toolkit needed.)
2. **Find the PC's LAN IP** — open Command Prompt and run `ipconfig`; note the IPv4 address (e.g. `192.168.1.50`). The Mac will connect to this.

### Step 1 — Install ComfyUI

**Easiest: ComfyUI Desktop**
1. Download the Windows installer from **https://www.comfy.org/download**.
2. Run it, accept defaults. It auto-detects the 3090 and installs the right PyTorch/CUDA build.
3. Launch it once to confirm it opens the web UI.

**Alternative: portable build** (no installer, fully self-contained)
1. Download the portable `.7z` from **https://github.com/comfyanonymous/ComfyUI/releases** — the NVIDIA asset is `ComfyUI_windows_portable_nvidia.7z` (latest verified release: **v0.22.0**, May 2026).
2. Extract it anywhere (e.g. `C:\ComfyUI`).
3. Start it with `run_nvidia_gpu.bat`.

### Step 2 — Make it reachable from the Mac

ComfyUI only listens on localhost by default. To let the Mac connect, start it bound to all interfaces:

- **Portable:** edit `run_nvidia_gpu.bat` and add `--listen 0.0.0.0` to the launch line, e.g.
  ```
  .\python_embeded\python.exe -s ComfyUI\main.py --listen 0.0.0.0
  ```
- **Desktop:** set the listen/host option in Settings → Server, or launch with the `--listen 0.0.0.0` argument.

Then from the Mac, open `http://<PC-LAN-IP>:8188` in a browser to confirm you can reach it. (Default port is `8188`.)

> **Security:** ComfyUI has no authentication. Only expose it on your trusted home LAN, not the public internet. If Windows Firewall prompts on first launch, allow it on private networks only.

### Step 3 — Get ACE-Step running

ACE-Step is built into current ComfyUI — no custom nodes needed. Model files (verified May 2026) live in the Comfy-Org repo **`Comfy-Org/ace_step_1.5_ComfyUI_files`** on Hugging Face. Two options:

- **Turbo AIO (simplest):** `checkpoints/ace_step_1.5_turbo_aio.safetensors` → drop in `models/checkpoints/`. Loads the "ACE-Step 1.5 Music Generation AIO" template.
- **XL (best quality, for the 3090):** three split files →
  - `split_files/diffusion_models/acestep_v1.5_xl_base_bf16.safetensors` → `models/diffusion_models/`
  - `split_files/text_encoders/qwen_4b_ace15.safetensors` → `models/text_encoders/`
  - `split_files/vae/ace_1.5_vae.safetensors` → `models/vae/`
  - Loads the "ACE-Step 1.5 Music Generation Workflow (split)" template.

Steps:
1. In the ComfyUI menu, open **Workflow → Browse Templates → Audio**, and pick the ACE-Step 1.5 template matching your model choice above (templates ship bundled with v0.22.0).
2. The first run will prompt to download any missing weights — accept, or pre-place the files above (this is what the auto-installer does).
3. Type a prompt + lyrics into the ACE-Step text node, hit **Queue Prompt**, and confirm audio generates (well under a minute on the 3090).

That's the minimum viable backend. The Mac app will later replicate the "Queue Prompt" step by POSTing the workflow JSON to `/prompt`.

### Step 4 (later, for restyle) — Stem separation & analysis tools

These are small Python tools. Install Python 3.10+ from python.org, then in a terminal:

```bat
pip install demucs        # stem separation (HT-Demucs)
pip install basic-pitch   # melody / audio-to-MIDI
```

Run examples:
```bat
demucs path\to\song.wav                          REM 4-stem split into .\separated\
demucs -n htdemucs_6s path\to\song.wav           REM 6-stem (adds guitar)
basic-pitch .\out\ path\to\song.wav              REM writes a MIDI transcription
```

`madmom` (beat/key/chord analysis) can be finicky to install on Windows — defer it until needed, and if it fights you, run it under WSL2 or pin an older Python.

### Step 5 — Vocal tools (separate vocal production)

**RVC (set this up — it's the vocal linchpin):**
1. Download the bundled Windows package (verified May 2026): **`RVC20240604Nvidia.7z`** (~5.6 GB) from Hugging Face — `https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/RVC20240604Nvidia.7z`. (GitHub releases are stale; use this. Use the `...50x0.7z` variant only for RTX 50-series — not the 3090.) It bundles its own Python, ffmpeg, and all models, so there's no pip/git step.
2. Extract and run `go-web.bat` — it opens the web UI in your browser on port **7897** (reachable from the Mac at `http://<PC-IP>:7897`).
3. To convert a vocal: load a trained voice model + your input vocal take, hit convert. To train a voice: feed 10–60 min of clean audio from a target singer and run the training tab (fast on the 3090).

**OpenUtau + DiffSinger (for authored melodies):**
1. Download OpenUtau from **https://github.com/stakira/OpenUtau** (Windows build) and unzip.
2. Add a DiffSinger English voicebank + the `nsf_hifigan` vocoder (links in the OpenUtau DiffSinger wiki).
3. Author notes/lyrics on the piano roll, render, then run the result through RVC for the target timbre.

**ACE-Step lyric2vocal LoRA:** already available wherever you run ACE-Step — no extra install. Use it for fast vocal stems.

### Step 6 (optional, only if growls become important) — YuE

YuE is Linux-first and heavy. If ever needed, the smoothest route on Windows is **WSL2** plus a quantized fork (`YuE-exllamav2` or `YuEGP`) to fit 24GB. Skip for now.

---

## 8. Verified ComfyUI integration (TESTED — works end to end)

Confirmed live against the Windows box at `192.168.1.x:8188` (ComfyUI v0.22.0, RTX 3090). The Mac submitted a job, ComfyUI generated, and the Mac pulled the audio back. ~40s of audio in ~50s of compute.

### Verified XL workflow (API format)

**Critical wiring gotcha:** ACE-Step 1.5 encoders CANNOT be loaded via a single `CLIPLoader` (vocab-size mismatch — ComfyUI issue #12278). XL needs a **`DualCLIPLoader`** with **`qwen_0.6b_ace15` in slot 1 and `qwen_4b_ace15` in slot 2** (NOT 1.7b — that pairing is for the non-XL split), type `ace`. A single 4b, or 4b twice, both fail with `NoneType ... shape` errors.

Node graph: `UNETLoader → ModelSamplingAuraFlow(shift 3) → KSampler.model`; `DualCLIPLoader → TextEncodeAceStepAudio1.5 → KSampler.positive` (and `→ ConditioningZeroOut → KSampler.negative`); `EmptyAceStep1.5LatentAudio → KSampler.latent_image`; `VAELoader(ace_1.5_vae) → VAEDecodeAudio ← KSampler`; `→ SaveAudioMP3`.

**Sampler settings per variant (from official templates):**

| Variant | DualCLIP encoders | steps | cfg | sampler | scheduler | shift |
|---|---|---|---|---|---|---|
| XL base | 0.6b + 4b | 50 | 6 | euler | simple | 3 |
| XL sft | 0.6b + 4b | 50 | 7 | euler | simple | 3 |
| XL turbo | 0.6b + 4b | 8 | 1 | euler | simple | 3 |
| Basic split (turbo) | 0.6b + 1.7b | 8 | 1 | euler | simple | 3 |

Use XL base/sft (50 steps) for final metal — turbo's 8 steps mushes distorted guitars and double-bass. Turbo for quick prompt iteration. A working API payload is in the git history of the test scripts; node input names verified via `/object_info`.

### Restyle (audio-to-audio) settings

Insert `LoadAudio → VAEEncodeAudio → KSampler.latent_image` (replacing the empty latent) and lower denoise. ACE-Step "cover strength" guidance: **0.3–0.5 for dramatic genre jumps (e.g. country/folk → metal)**, 0.5–0.7 moderate, 0.7–0.9 subtle; pair big jumps with guidance 9–10 so the prompt dominates. Multi-pass (lowering strength each pass, feeding output back) helps extreme shifts.

### Recommended integration architecture (from production-app research)

- **Don't hit ComfyUI directly from the Mac UI.** Put a thin **FastAPI gateway on the Windows box** next to ComfyUI (`localhost:8188`); the Mac talks only to that gateway. Gives auth (ComfyUI has none), a clean stable JSON contract, and isolates ComfyUI quirks.
- **Workflow templating:** export the graph in **API format** (Dev mode → Save API Format), name nodes semantically, and inject params by matching `_meta.title` — never hardcode numeric node IDs (they change when the graph is edited).
- **Progress loop:** connect the `/ws` socket with the SAME `client_id` you POST; completion = an `executing` event with `node == null` for your `prompt_id`. Guard against binary frames (`isinstance(msg,str)`). `/history/{id}` is `{}` until done.
- **Cancel:** queued jobs → `POST /queue {"delete":[id]}`; running job → `POST /interrupt`. Do both for a real Cancel button.
- **Latency:** run a warmup generation at gateway startup; keep the model resident; only `POST /free` when actually swapping models.
- Reference repo to study: `SaladTechnologies/comfyui-api` (MIT) — stateless wrapper over ComfyUI with warmup, storage, health probes.

## 8b. Compute placement (which machine runs what)

GPU-heavy Python tools must run where there's a GPU — they don't just run wherever the backend lives. Rule: **CUDA-locked heavy work → Windows 3090; light/CPU or MPS-capable work → Mac (in parallel, keeping the 3090 free for generation).**

| Tool | GPU need | Runs on | Notes |
|---|---|---|---|
| ACE-Step generation | heavy CUDA | **Windows (ComfyUI)** | the engine |
| RVC voice conversion | CUDA | **Windows** | one-click pkg is Windows-CUDA |
| Demucs stem separation | CUDA / **MPS** / CPU | **Mac (MPS)** | run in parallel → frees the 3090 |
| Basic Pitch (audio→MIDI) | tiny / CPU | **Mac** | negligible |
| madmom (beat/key/chord) | CPU | **Mac** | CPU-only lib |

**Decision (2026-05): Demucs stays on the Mac.** Benchmarked Mac MPS at ~7.7s / 35s (MPS barely beats CPU's 9.7s — Demucs doesn't accelerate well on Apple Silicon). A CUDA 3090 would be ~3–4× faster per job, but (a) it has no remote invocation channel (would need a new Windows endpoint), and (b) it would contend with generation on the single GPU. The Mac path is parallel, zero extra infra, and fast enough — chosen deliberately over the 3090.

The FastAPI backend runs on the Mac, so MPS/CPU tools (Demucs, Basic Pitch, madmom) it can invoke directly. For Windows-GPU tools beyond ComfyUI (e.g. RVC programmatically), drive them over the network (RVC's web UI / API at `:7897`) or add a small Windows-side worker — don't try to run them in the Mac backend.

## 8c. Alternative RVC drivers (researched — recommended migration)

Our current RVC integration drives the Gradio 3.14 WebUI via `/run` + smuggles audio through ComfyUI's input dir (server-side path hack). A cleaner option is verified:

**`rvc-python` (PyPI `rvc-python` v0.1.5, repo `daswer123/rvc-python`)** — RVC as a Python module, CLI, **and a proper HTTP API server**. Windows + CUDA supported; dynamic model load/unload; model-directory management.
- Install (Windows): `py -3.10 -m venv venv && venv\Scripts\activate && pip install rvc-python && pip install torch==2.1.1+cu118 torchaudio==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118`
- Run API server: `python -m rvc_python api -p 5050 -l`  (`-l` = listen on LAN)
- **Endpoints:** `POST /convert` with JSON `{"audio_data": "<base64 wav>"}` → returns converted WAV (real audio over HTTP — **no server-path hack**); `GET /models`; `POST /models/{model_name}` to load; params settable.
- Python module: `from rvc_python.infer import RVCInference; rvc=RVCInference(device="cuda:0"); rvc.load_model(path); rvc.infer_file(in,out)`.

**Why it looked good:** clean API (base64 `/convert`), removes the Gradio + ComfyUI-input-dir coupling.

**⚠️ BLOCKER (verified 2026-05): rvc-python does NOT install cleanly on Windows.** Its pinned deps are the problem, not omegaconf (which is a universal wheel — a red-herring in pip's error):
- `fairseq==0.12.2` has **no Windows wheel** (only macOS/Linux, only up to cp38) → must compile from source with MSVC C++ Build Tools; notoriously fails.
- `numpy<=1.23.5` has wheels only up to cp311.
So even forcing Python 3.10 (fixes numpy) still hits the fairseq build wall. **Not recommended on Windows** without committing to a source build.

**Reassessment of options:**
- **Keep the bundled RVC WebUI** (`RVC20240604Nvidia`) — it already ships a WORKING fairseq/torch env (prebuilt). Our Gradio driver works against it today (app auto-falls-back to it). The only wart is the ComfyUI-input-dir file bridge.
- **Custom thin API reusing the WebUI's env** — write our own small FastAPI server launched with the package's `runtime\python.exe` (which already has all deps working), exposing clean `/convert` + voice-upload endpoints. Achieves the migration goals (clean API, remote voice upload) with NO new install / no fairseq build. Best engineering path.
- **Applio** (`IAHispano/Applio`) — modern fork, but it's another Gradio app via `run-install.bat`; not a clean REST API; install may pull heavy deps. Not clearly better.
- **rvc-python** — only viable if willing to install VS C++ Build Tools and build fairseq.

UI/UX research & design brief: see **`UI_DESIGN.md`**.

## 8d. Ready-made RVC voices (downloadable — researched, web-verified)

Huge free community libraries exist; download a `.pth` (+ matching `.index`) instead of training.

**Hubs (priority order):**
- **voice-models.com** — largest searchable web catalog (tens of thousands), free, audio previews, download links out to HF/Drive zips. Only male/female facet (no singing/speaking filter — read titles). https://voice-models.com/
- **AI HUB Discord** `#voice-models` — biggest library + direct HF links + samples. ⚠️ Use verified invite `discord.gg/mmRR2TUJF5`; the `discord.gg/aihub` vanity was reportedly **hijacked Apr 2026** — avoid. https://docs.aihub.gg/essentials/voice-models/
- **Hugging Face** — where files physically live; good for backup/re-download.
- **rvc-models.com** — clean **Singers** category. https://rvc-models.com/c/singers/6
- ⚠️ **weights.gg** — reported degraded/closed ~Apr 2026; don't rely on it.

**Metal/rock vocalist models found (all RVC v2, free) — for clean/powerful targets:**
- Bruce Dickinson (Iron Maiden) — operatic/power-metal tenor: https://voice-models.com/model/1veXq1h08WI
- James Hetfield (Metallica) — era variants (gritty thrash).
- Dave Mustaine (Megadeth), Freddie Mercury (belted), Chester Bennington (gritty/screamed).
- Female symphonic-metal sopranos (Tarja / Floor Jansen / Simone Simons / Amy Lee) very likely present but URLs unverified — search voice-models.com / AI HUB directly.

**Compatibility:** v2 is standard (768-dim; v1=256, not interchangeable). Sample rate (32k/40k/48k) is baked into the `.pth`; WebUI auto-reads it, `rvc-python` needs `-v v2` set explicitly. **Always use the matching `.index`** for best timbre; never use the big `G_*/D_*` `logs/` checkpoints for inference (only the small extracted weight).

**Install:** RVC WebUI → `.pth` in `assets/weights/`, `.index` in `logs/<name>/`, click refresh. rvc-python → one subfolder per model in `rvc_models/` (`<name>.pth` + `<name>.index`).

**Licensing/ethics:** community licenses are inconsistent (many "other"/personal-use). Cloning real artists implicates right-of-publicity/voice rights — personal use low-risk; **sharing/monetizing imitations of a recognizable singer is risky**. For anything released, use your own/a consenting singer's voice. (See `PLAN.md` risks.)

## 9. Suggested next steps

1. ✅ **DONE — ACE-Step XL generates metal end-to-end** via the Mac→ComfyUI API. (Quality of distorted guitars/vocals still to be judged by ear and tuned.)
2. **Judge & tune metal quality** — listen to XL base output, sweep steps/cfg and prompt tags; compare XL base vs sft for distorted guitars + vocals.
3. **Test the vocal pipeline** — get a vocal stem (lyric2vocal or DiffSinger), run it through RVC (`192.168.1.x:7897`), confirm a usable powerful/clean metal vocal.
4. **Build the FastAPI gateway** on the Windows box (API-format template + `_meta.title` injection + `/ws` progress loop) — the contract is proven; now wrap it cleanly.
5. Prototype the **restyle pipeline** (Demucs + Basic Pitch + ACE-Step cover at 0.3–0.5 strength).
6. Pick the **Mac UI stack** and build a thin end-to-end slice: prompt → job → audio back → play.

## Key sources

- ACE-Step 1.5: https://github.com/ace-step/ACE-Step-1.5
- ACE-Step "Musician's Guide" (restyle settings): https://github.com/ace-step/ACE-Step-1.5/discussions/235
- ComfyUI ACE-Step support: https://docs.comfy.org/tutorials/audio/ace-step/ace-step-v1
- ComfyUI API routes: https://docs.comfy.org/development/comfyui-server/comms_routes
- ComfyUI download: https://www.comfy.org/download
- YuE: https://github.com/multimodal-art-projection/YuE
- HeartMuLa: https://github.com/HeartMuLa/heartlib
- Demucs: https://github.com/facebookresearch/demucs
- Basic Pitch: https://github.com/spotify/basic-pitch
- RVC (voice conversion): https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
- OpenUtau (+ DiffSinger): https://github.com/stakira/OpenUtau
- DiffSinger: https://github.com/openvpi/DiffSinger
- SoulX-Singer (zero-shot SVS): https://github.com/Soul-AILab/SoulX-Singer

---

## 10. MUSIC-QUALITY PUSH — research findings (2026-05-24)

_Scope: heavy/power/symphonic/folk **metal** AND **heavy rock** (Bon Jovi, Halestorm, Black Stone Cherry, AC/DC). Goal: make the generated audio itself better, especially the weak distorted-guitar tone. Four research tracks below; a prioritized experiment list closes the section. **Nothing here has been run on the GPU yet** — confirm before generating/installing._

### 10a. Post-processing & re-amp (Track B) — biggest near-term lever, runs on the Mac

**The honest caveat first.** True "re-amping" (NAM / amp-sim) expects a **clean DI** (un-amped) guitar signal. Our generated guitar — and any Demucs-separated guitar stem — is **already distorted and cabinet-coloured**. You cannot cleanly re-amp an already-amped signal. So on a generated/separated distorted stem the realistic, effective chain is **reshape**, not re-amp:

1. **Surgical EQ** — high-pass the sub-lows (~80–100 Hz) for tightness; narrow cuts on fizz/harshness (commonly ~3–6 kHz); a high shelf/low-pass above ~8–10 kHz to kill digital fizz. (EQ alone gets you most of the way; post-IR EQ is the standard fizz fix.)
2. **IR cab convolution ("re-cab")** — convolve the stem with a guitar **cabinet impulse response** to re-shape cab character and smooth fizz. ⚠️ Risk of "double-cab" mud since the stem already has a cab baked in — use a darker IR + high-cut, or treat IR as optional/parallel.
3. **Saturation / tightening** — gentle tape/tube saturation, light multiband, transient tightening for perceived power. A lot of "power" is mix, not model.
4. **Alternative to IR** — a synthesized "analog cab" (parametric EQ + a touch of reverb) is more editable than an IR and avoids double-cab artifacts.

**Tools (all Python, Mac, run in parallel to the 3090 like Demucs already does):**
- **Spotify `pedalboard`** (PyPI `pedalboard`; GPLv3 — verify, fine for local/personal use) — the core post-FX engine. Native blocks: `Convolution` (IR cab), `Distortion`, `LowpassFilter`/`HighpassFilter`/`PeakFilter`/`LadderFilter` (EQ), `Compressor`, `Gain`, `Limiter`, `Reverb`, `Chain`/`Pedalboard`. Also hosts **VST3/AU** via `load_plugin(path, parameter_values={...})`, params settable as attributes. macOS supported. → builds our guitar-stem tone chain and a master bus chain entirely in Python.
- **Neural Amp Modeler (`neural-amp-modeler`, MIT)** — thousands of free amp/pedal captures (`.nam`) at tone3000 / ToneHub. Two host paths: (a) load the **NAM VST3/AU plugin inside pedalboard** and point it at a `.nam`; ⚠️ **VERIFY** whether the model-file path is exposed as a settable plugin parameter (NAM loads models via its own file picker — may require a saved plugin **state/preset** rather than a param). (b) standalone. **But NAM needs a clean DI** → only useful if we ever produce/extract a clean DI (we don't today). **Lower priority for generated audio; IR + EQ + saturation is the practical win.** Revisit NAM if we add a DI-producing path (e.g. a clean-tone generation + re-amp workflow).
- **Matchering 2.0 (`matchering`, GPLv3)** — reference-based mastering: matches the target's RMS, frequency response, peak and stereo width to a **reference master** you supply (a Halestorm / AC-DC / Bon Jovi track you own). One `mg.process(target, reference, results=[...])` call. Mac-side. Great for (a) making a mix sit like a pro master and (b) consistency across a batch. Personal-use for copyrighted references.
- **Free cab IRs** — tone3000 and many CC/free packs (verify license per pack).

**Pipeline fit:** generate → **Demucs `htdemucs_6s`** (6-stem, *includes a guitar stem*) on the Mac → apply the **guitar-tone chain** (pedalboard) to the guitar stem → recombine via the existing in-app **mixer** → optional **Matchering** master on the final bounce. New module e.g. `backend/postfx.py` + endpoints; UI "Tone / Master" controls. (Note: current Demucs default is 2-/4-stem; need `htdemucs_6s` for a guitar stem.)

### 10b. Guitar / amp tone feature design (Track C)

Two complementary layers ("more options always better"):
- **Prompt-vocabulary layer** (free, only the normal gen cost) — curated **tone descriptors** injected into the tags: amp/era/cab/pickup/recording words (see 10d list). Steers what the model *renders*. Add as preset chips in `web/src/presets.ts`.
- **Post-process tone layer** (Mac, from 10a) — a **"Guitar Tone" stage** = a `pedalboard` chain on the *separated guitar stem*, exposed as named presets, each = `{EQ curve, IR cab choice, saturation amount, noise gate}`. Examples: **Modern High-Gain**, **AC/DC Crunch**, **80s Hot-Rodded Marshall**, **Doom Fuzz**, **Southern Overdrive**. **Primary engine = the user's licensed Line 6 Helix Native** hosted via `pedalboard` (cab/EQ/comp/FX preset with the amp block bypassed — see §10f); **NAM plugin slot** as the alternative; native `pedalboard` EQ+IR+saturation as the no-plugin fallback.

**Honest scope to set in the UI:** the prompt vocabulary steers the model's rendered tone; the post-FX stage *reshapes* the already-rendered tone (EQ / IR / saturation) — it does **not** fully re-amp from a clean DI. Frame it as "tone shaping," not "amp re-amping."

### 10c. ACE-Step / ComfyUI workflow tuning (Track A) — experiment matrix

- **Real negative prompts.** The standard XL workflow's `ConditioningZeroOut` is only the **empty-negative fallback** (confirmed on the official comfy.org XL Base workflow page). A genuine negative prompt = a **second `TextEncodeAceStepAudio1.5`** (negative tags) → `KSampler.negative` (replacing `ConditioningZeroOut`). The XL **SFT** community pipeline adds an **Adaptive Projected Guidance (APG)** node + first-class negative support. Candidate negatives to test: `"muddy, lo-fi, harsh fizz, digital clipping, thin weak guitars, out of tune, low quality, mono, distorted vocals"`. Plain negative conditioning works on **base**; APG needs **xl_sft** (not installed). **Currently we use neither — this is the cheapest GPU experiment.**
- **Sampler / scheduler.** We hardcode `euler`/`simple`. Community-reported best for XL: **SFT** → `euler` or `res_2s`, scheduler `normal`, ~46 steps, cfg ~7.3; **Turbo-SFT merges** → `er_sde` + `beta57`, ~22 steps. For **restyle/source-change**, `beta` / `normal` / `linear_quadratic` / `ddim_uniform` are reported stronger. Sweep: `{euler, res_2s, er_sde} × {simple, normal, beta, beta57}`.
- **AuraFlow `shift`** (fixed at 3.0). Sweep ~1–5; shifts the timestep distribution → detail vs. coherence trade-off.
- **cfg / steps.** Base cfg 6 / 50 steps; SFT cfg 7(.3) / 46–50. Try cfg 5–8 and steps 50–80 for guitar detail (diminishing returns + time cost on the single 3090).
- **Internal LM sampling params** (hardcoded in `_text_encode`: `cfg_scale 2.0, temperature 0.85, top_p 0.9`) — these drive the **audio-codes LM** (structure/coherence), separate from KSampler cfg. Expose + sweep.
- **Multi-pass / refiner.** Feed an output back through a **low-denoise** second pass (like restyle on its own output) to tighten/add detail; lower denoise each pass for extreme shifts (cf. §8 restyle).
- **Prereqs:** download **`xl_sft`** (APG + refined detail) and **`xl_turbo`** (fast iteration for sweeps) — ✅ xl_sft now installed (2026-05-25); turbo not.

**Community ACE-in-ComfyUI usage (2026-05-25, ongoing — capture what others actually run, not just the official workflow):**
- **Adaptive Projected Guidance (APG)** — native ComfyUI node (`APG`, category sampling/custom_sampling), a **model patch** (`model → APG → KSampler`); params `eta` (1 = plain CFG; ~1.05), `norm_threshold` (main knob, lower = more normalization; ~1.3 for ACE), `momentum` (0). **The recommended fix for ACE-Step's high-CFG oversaturation/muddiness** (ComfyUI issue #8026 "ACE-Step Support APG … to reduce vocal oversaturated"): lets you keep strong cfg (6–7.3) for prompt adherence *without* the mud, instead of just dropping cfg. Community ACE-XL-SFT recipe: euler (or res_2s), scheduler `normal`, ~46 steps, **cfg 7.3 + APG eta 1.05 / norm_thresh 1.3 / momentum 0**. ✅ **WIRED** into `comfy.py` `_apg_model` as an optional stage (`apg`+`apg_eta`/`apg_norm`/`apg_momentum`; Expert UI toggle) — GPU A/B pending.
- **Other guidance nodes present on the box** (worth trying / from community workflows): `CFGZeroStar`, `RescaleCFG`, `RenormCFG`, `CFGNorm`, `PerpNegGuider`, `PerturbedAttentionGuidance`, `TCFG`, `DualCFGGuider`, `SkipLayerGuidance*` — several target the same oversaturation/quality axis as APG.
- **Custom node pack:** **JK-AceStep-Nodes** (`github.com/jeankassio/JK-AceStep-Nodes`) — "advanced sampling nodes optimized for ACE-Step audio." **Guidance interval** (apply CFG only over a step range) is another community lever to reduce oversaturation (issue #8026). Community XL workflow w/ these: Civitai 2375403 (geoblocked here).
- _TODO (per user, 2026-05-25): keep mining how people run ACE in ComfyUI — samplers/guidance tricks, node packs, settings — and fold the good ones in. User is also relaying tips from a YouTube walkthrough._

### 10d. Prompt engineering per (sub)genre + models/LoRAs (Track D+E)

**Tag structure (official guide):** `genre + era → key instruments → mood/adjectives → tempo/BPM`; 3–7 tags is the sweet spot, but instrument-rich captions also work well. Lead with genre. Avoid contradictory tags (e.g. `ambient, metal`). Bracketed structure tags in lyrics (`[Verse]`, `[Chorus]` = "emotional peak, highest energy", etc.) steer dynamics.

**Verbatim official metal example:** `"heavy metal rock with heavily distorted electric guitars, aggressive double bass drumming, powerful screaming vocals, fast tempo, high energy, intense dark atmosphere"`.

**Heavy-rock vocabulary (new scope — drafts to A/B):**
- **AC/DC:** `"hard rock, crunchy overdriven Marshall guitars, swinging backbeat, gang-shout backing vocals, raspy male vocal, 4/4, mid tempo, live-room production"`
- **Bon Jovi (arena rock):** `"80s arena rock, anthemic, big layered guitars, huge hooky chorus, polished production, gated-reverb drums, soaring male vocal"`
- **Halestorm (modern hard rock):** `"modern hard rock, powerful female vocal, high-gain rhythm guitar, punchy modern production, radio rock, driving tempo"`
- **Black Stone Cherry (southern hard rock):** `"southern hard rock, bluesy thick overdriven guitar riffs, groovy, soulful gritty male vocal, mid tempo"`

**Modern power-metal bands (added to the genre registry 2026-05-24, web-researched).** _Bands are nested under a parent genre via the `parent` field (they stay in the flat registry so riff/solo lookup is unchanged, but the UI shows them under their genre — "Artist style" dropdown on the chips, `<optgroup>` in the Guitar picker — to keep the top-level genre list manageable). Battle Beast + Beast in Black → `power`; AC/DC → `hard_rock`._
- **Battle Beast** (Finnish) — power/heavy metal with strong **80s hard-rock swagger** + symphonic touches; **keytar/synthesizer** layers (Janne Björkroth); anthemic fist-pump choruses; fast/mid tempos, driving riffs; **powerful high belting female vocals** (Noora Louhimo, operatic range w/ shrieks). Registry id `battle_beast` (~155 BPM, E minor).
- **Beast in Black** (Finnish, Anton Kabanen ex-Battle Beast) — **power metal fused with 80s synthwave / Italo-disco**; bright retro synths over **tight palm-muted galloping** guitars; very **hooky/danceable**; **soaring high Halford-like male vocals** (Yannis Papadopoulos). Registry id `beast_in_black` (~165 BPM, E minor). Both are essentially '80s-tribute-leaning modern power metal — synth-forward, melodic leads, palm-muted gallops. (Band emulation = personal-use, see PLAN §7 risks.)

**Tone descriptors for the prompt-vocabulary layer (10b):** amp/era (`Marshall`, `Mesa Boogie`, `Orange`, `5150`, `Plexi`, `70s`, `80s`, `modern`), cab/mic (`4x12 cab`, `close-mic'd`, `SM57`), pickups (`humbucker`, `single-coil`), gain/feel (`high-gain`, `overdriven`, `crunchy`, `palm-muted`, `chugging`, `saturated`), production (`polished`, `live recording`, `bedroom`, `analog warmth`, `tight low end`).

**Reference-audio (Cover/Audio2Audio):** 0.3–0.5 strength for big jumps + guidance 9–10 so the prompt dominates; multi-pass for extreme shifts (cf. §8).

**Models:** ACE-Step 1.5 stays primary (fast, ComfyUI-native, restyle built-in). **YuE** is the only open model explicitly noted as *metal-resilient* (handles low vocal-to-accompaniment ratios) but is slow/VRAM-heavy → only worth it if we want harsh-vocal full songs, out of scope for guitar **tone**. **DiffRhythm** (fast latent-diffusion full songs) is an alternate engine to watch, not a tone fix. Conclusion: tone work belongs in ACE-Step tuning + post-FX, not a model swap.

**LoRAs (ACE-Step, Civitai/HF):** found **"Epic Music v1.0"** (symphonic/orchestral — useful for symphonic-metal beds), **"Psychedelic Rock/Funk v1.1"**, and HF **`ACE-Step-v1.5-acoustic-guitar`**. ⚠️ Civitai is **geoblocked** from this research environment (UK Online Safety Act) so **base-version compatibility (v1 vs v1.5 XL) is UNVERIFIED** — LoRA architecture differs by base model; verify before loading. **No dedicated metal / high-gain rhythm-guitar LoRA found.** ACE-Step supports easy LoRA training ("from a few songs," one-click in the Gradio UI) → **a custom heavy-rock / metal rhythm-guitar LoRA is the highest-ceiling (and highest-effort) lever.**

### 10f. Helix Native as the tone engine + the clean-DI ("de-amp") question

**The user owns a licensed Line 6 Helix Native** (pro amp/cab/effects modeler, VST3/AU/AAX on Mac). This is the **preferred tone engine** for the Track B/C guitar stage — it beats hand-rolled EQ/IR chains and NAM-capture wrangling, and it's already paid for.

**Hosting it from `pedalboard`:**
- `pedalboard.load_plugin("/path/Helix Native.vst3")` loads it; parameters are settable as Python attributes and via `raw_value` ∈ [0,1]. macOS AU/VST3 supported.
- ⚠️ **Preset loading is the rough edge.** `load_preset` only handles `.fxp`/`.vstpreset` and is plugin-dependent; Helix's own `.hlx` preset format and full internal state are **not** guaranteed to load cleanly, and pedalboard's save/restore of arbitrary plugin **state** is an open/under-developed area (issues #11, #187, #245). Robust plan: (a) dial tones in Helix's editor (pedalboard can open it via `plugin.show_editor()` on macOS) and/or (b) set the exposed parameters programmatically per named preset we define; verify state persistence on first integration. Treat "load my existing .hlx presets verbatim" as **unverified** until tested.
- **Block-level reality:** Helix **cab / IR / EQ / compressor / effects** blocks work fine on already-distorted audio (this is the realistic *reshape* path — build a "cab + EQ + comp" preset with the **amp block bypassed**). Helix **amp** blocks, like NAM, expect a **clean DI** and will sound wrong stacked on an already-amped stem.

**The clean-DI / "de-amp" path (worth the time — user asked to investigate).** To make amp blocks (Helix or NAM) truly usable on generated guitar, we'd need a *clean DI*. There's an active research line for exactly this — **guitar effect / distortion removal**:
- arXiv **2202.01664** (Sony, "Learning How to Recover the Clean Signal") — frames de-distortion as a source-separation/effect-modeling task; fast NN inference.
- arXiv **2407.16639** (DAFx 2024, "Distortion Recovery: A Two-Stage Method for Guitar Effect Removal") — Mel-spectrogram stage + neural vocoder (HiFi-GAN); trained on **EGDB** rendered with commercial VST FX; demos at y10ab1.github.io/guitar_effect_removal.
- **Honest reality check:** these work best where the clean signal is *superimposed* (overdrive/crunch) and are trained on relatively clean/single-instrument guitar (EGDB), **not** dense high-gain metal walls. So: plausibly useful for **AC/DC-style crunch** re-amp experiments; **unreliable/lossy for full high-gain metal**. No turnkey `pip` tool — would require adapting research code + checkpoints. **Experimental, medium–high effort, uncertain payoff for the hardest cases.**
- **If pursued, the pipeline:** generate → Demucs `htdemucs_6s` guitar stem → **de-amp model → estimated clean DI** → Helix/NAM **amp + cab** → recombine. Best built as an *optional* experimental stage, gated behind "this works better for crunch than for metal."

**Net recommendation:** make **Helix Native (cab/EQ/comp/FX preset, amp bypassed)** the default reshape tone engine; offer a **NAM plugin slot** as an alternative; build the **clean-DI de-amp stage as an experimental opt-in** so amp blocks become viable for lighter-gain (heavy-rock) material — with expectations set that high-gain metal may not recover cleanly.

### 10g. DI research batch — getting a clean DI so amp-sims become valid (2026-05-24)

**Why this is its own batch:** amp models (Helix amp blocks, NAM) only sound right on a **clean DI**. Guitar/amp FX must go **only on the guitar stem**, and amp-*sims* specifically need a de-amped signal. A generated/separated stem is already distorted, so today our Tone presets are *reshape only* (amp bypassed). To unlock true re-amping we need a clean DI by one of two routes:

**Route A — "reverse DI" / de-amp an existing distorted guitar (effect removal):**
- **RemFx** (`github.com/mhrice/RemFx`, **Apache 2.0**) — general-purpose audio-effect removal; **pretrained checkpoints on Zenodo** (`scripts/download_ckpts.sh`), `scripts/remfx_detect.sh in.wav -o dry.wav`; removes **distortion** (+ chorus/delay/compression/reverb), HF Space + Colab. **Most directly usable today.** Caveat: trained on GuitarSet-style (relatively clean) sources with effects added; the paper notes "examples with many effects remain challenging" → expect decent results on **crunch**, poor on dense **high-gain metal walls**.
- **Distortion Recovery** (DAFx 2024, arXiv 2407.16639; demos `y10ab1.github.io/guitar_effect_removal`) — Mel-spectrogram stage + neural vocoder, trained on EGDB w/ commercial VST FX. Research code; same clean-bias caveat.
- **Sony "recover the clean signal"** (arXiv 2202.01664) — earlier source-separation framing.
- Reality: **de-amping is reliable for overdrive/crunch, lossy/unreliable for high-gain.** Good enough for the heavy-rock end (AC/DC/BSC), weak for metal.

**Route B — generate/derive a clean DI instead of recovering one (often better):**
- **B1 — Generate CLEAN, then amp (recommended to try first).** Models render *clean* guitar far better than high-gain. Prompt ACE-Step for **"clean electric guitar, direct input, no distortion"** (or lightly-driven), separate that guitar stem, then apply a **Helix high-gain amp + cab** → full, controllable metal tone with a *real* DI-like source. Flips the pipeline: stop fighting the model's weak high-gain render; let it do the easy part and do the amping ourselves. Novel + low-risk; no new ML.
- **B2 — Audio→MIDI→render DI→re-amp.** Demucs guitar stem → **Basic Pitch / NeuralNote** (Spotify's open model; `basic-pitch` already planned, RESEARCH §4) → MIDI → render a **clean DI** with a sampled/virtual guitar → Helix amp. Caveat: polyphonic guitar transcription is **best on clean/lightly-distorted** input and **loses performance nuance** (it's a re-creation, not the original take); pitch-bend/vibrato capture varies. Commercial transcribers (Jam Origin **MIDI Guitar**, **Prism**) are stronger but paid. Useful as an *alternative render*, not a faithful DI.
- **B3 — MIDI-first guitar.** Compose/extract the riff as MIDI up front (we already have a melody/MIDI stack in `melody.py`) and render a clean DI to amp — most control, least "AI guitar," but most departure from text-to-music.

**The amp/cab side (forward direction) is solved:** Helix Native (owned, wired) + NAM (GuitarML **PedalNet/PedalNetRT** are the emulation side) + IRs. The missing piece is purely the **clean DI input** — Route B1 is the cheapest meaningful unlock; RemFx (A) is the quickest thing to *try* on crunch.

**Proposed DI experiments (add to §10e as #11–#13):** #11 RemFx de-amp on a separated guitar stem → Helix amp (crunch first); #12 generate-clean-then-amp (B1); #13 Basic Pitch → DI render → amp (B2). All Mac-side except the generation step. **Gate amp-sim presets behind "DI present"** so amps never land on an already-distorted stem.

### 10h. Other ways to generate guitar (beyond ACE-Step) — research batch (2026-05-24)

_Context: we ruled out getting a clean DI from ACE-Step (won't render clean on prompt) and from RemFx de-amp (too weak on high-gain — confirmed by ear). So: what other guitar-generation paths exist? Three families._

**Family 1 — Stem-output / multi-track audio models (native isolated guitar, no Demucs).** These generate an *isolated guitar stem* directly, avoiding Demucs separation artifacts (the "vague noises" problem), and some can generate a guitar **conditioned on existing drums/bass** (arrangement):
- **MSG-LD** (`github.com/karchkha/MSG-LD`) — latent-diffusion multi-track; joint separation+generation; stems = bass/drums/guitar/piano; can generate a guitar track given the others.
- **MusicGen-Stem** (arXiv 2501.01757) — first open-source multi-stem autoregressive model; good quality + coherent per-stem editing.
- **StemGen** (arXiv 2312.08723), **Multi-Track MusicLDM** (2409.02845), **Jen-1 Composer**, **MSDM** — related multi-stem approaches.
- *Pros:* clean isolated guitar (better raw material for the reshape chain; no separation artifacts), arrangement control. *Cons:* still **distorted audio** (not a DI — doesn't unlock amp-sims), unknown metal strength, training/setup, mostly research-grade. **Verdict:** a *quality* upgrade to the reshape pipeline, not a tone-control unlock.

**Family 2 — Symbolic / tablature generation → render a clean DI → amp (THE clean-DI solution for metal).** Generate the guitar as **GuitarPro/MIDI**, render it through a clean DI guitar instrument, then amp with Helix. This is the one path that gives metal-appropriate playing AND a true clean DI with full tone control:
- **DadaGP** (`github.com/dada-bots/dadaGP`, **code public**) — encoder/decoder GuitarPro↔token (gp3/4/5), the tokenizer for LM-based gen. 26k songs/739 genres (dataset by request).
- **ProgGP** (`github.com/otnemrasordep/ProgGP`) — 173 **progressive-metal** songs in DadaGP tokens + a fine-tuned Transformer that generates guitar/bass/drums/piano/orchestral. **Metal-specific.**
- **GTR-CTRL** (arXiv 2302.05393) — genre+instrument-conditioned guitar tab generation; **ShredGP** (2307.05324) — guitarist-style-conditioned. 
- **Render the MIDI to a clean DI:** free DI-recorded guitar instruments — **Shreddage 3 Stratus FREE** (recorded DI through the neck pickup, *designed to be amped*, runs in free Kontakt Player), **Cute Emily** (clean/dry SG), Spitfire **LABS Peel**. `pedalboard` can host an instrument VST3 + feed MIDI (≥0.9 supports instrument plugins), so the render can stay in our Python pipeline → then the existing Helix amp stage is finally valid (real clean DI in).
- *Pros:* true clean DI, metal-appropriate riffs, total tone control (Helix amp/cab), reproducible/editable (it's symbolic). *Cons:* different paradigm from text-to-music; the gen models are 2023-era (Transformer-XL) and need setup; MIDI→DI render via Kontakt/VST is fiddly; symbolic gen ≠ ACE's "vibe." **Verdict:** highest ceiling for *guitar tone* specifically; biggest build.

**Family 3 — Neural guitar synthesis (DDSP).** arXiv 2309.07658 — string-wise MIDI → guitar waveform via DDSP; demos at erl-j.github.io/neural-guitar-web-supplement. Research-grade, acoustic-leaning, not production-ready. Watch.

**Bottom line / recommendation.** Two tiers:
- **Near-term (improve the source, low effort):** stick with ACE-Step but make its output better — download **xl_sft** + run a sampler/scheduler/cfg/steps sweep (§10c), and consider a **Family-1 stem model** later to get a clean (artifact-free) guitar stem for the reshape chain.
- **High-ceiling (real tone control, big build):** the **Family-2 symbolic route** — generate metal guitar as MIDI (ProgGP/DadaGP/GTR-CTRL) → render a clean DI (Shreddage 3 Stratus FREE) → Helix amp. This is the only path that makes the owned Helix amp genuinely useful for metal, and gives reproducible, editable riffs. Worth prototyping once the near-term source work is done.

### 10e. Proposed prioritized experiments (confirm before any GPU run / install)

| # | Experiment | Track | Expected impact | Effort | GPU? |
|---|---|---|---|---|---|
| 1 | **Guitar post-FX chain** — `pedalboard` hosting **Helix Native** (cab/EQ/comp, amp bypassed) + IR/saturation on the `htdemucs_6s` guitar stem, recombine via mixer | B/C | **High** (direct fix for the weak guitar) | Med | No (Mac) |
| 2 | **Matchering master stage** — reference-master the final bounce | B | Med–High (cohesion/loudness) | Low | No (Mac) |
| 3 | **Real negative prompt on `xl_base`** — 2nd TextEncode → KSampler.negative | A | Med | Low | Yes (small) |
| 3✅ | _**WIRED 2026-05-24**: `comfy.py` `_negative_node` + `DEFAULT_NEGATIVE`; Generate UI Expert "Negative tags" field → `negative_tags`. Backward compatible (blank = ConditioningZeroOut). **GPU A/B test pending user OK.**_ | A | — | done | — |
| 10 | **Clean-DI "de-amp" stage** (experimental) — distortion-removal NN on the guitar stem → clean DI → Helix/NAM **amp** block. Better for crunch than high-gain metal (§10f) | B/C | Med (crunch) / Low (metal) | High | Maybe (Mac/GPU) |
| 11 | **RemFx de-amp** → Helix amp (Apache-2.0, pretrained ckpts; try on crunch first) — §10g Route A | B/C | Med (crunch) | Med | No (Mac) |
| 12 | **Generate-clean-then-amp** — prompt ACE-Step for clean/DI guitar, separate, apply Helix high-gain amp+cab — §10g Route B1 (recommended) | A/B/C | **High** | Med | Gen on GPU, amp on Mac |
| 13 | **Basic Pitch → DI render → amp** — guitar stem → MIDI → clean virtual-guitar DI → Helix — §10g Route B2 | B/C | Med | Med | No (Mac) |
| 14 | **Symbolic guitar → clean DI → amp** — ProgGP/DadaGP/GTR-CTRL generate metal MIDI → render Shreddage 3 Stratus FREE (clean DI) → Helix amp — §10h Family 2. The real tone-control path. | new gen | **High (tone control)** | High | Mostly Mac |
| 15 | **Stem-output model** (MSG-LD / MusicGen-Stem) for a native isolated guitar stem (no Demucs artifacts) feeding the reshape chain — §10h Family 1 | quality | Med | Med–High | GPU |
| 4 | **Tone-vocabulary presets** — amp/cab/era tags as chips | C/D | Med | Low | No (prompt only) |
| 5 | **Sampler/scheduler + shift + cfg/steps sweep** — structured A/B grid | A | Med | Med | Yes (batch) |
| 6 | **Download `xl_sft` (+ `xl_turbo`)** → APG negative + base-vs-sft quality compare | A | Med–High | Med | Yes + download |
| 7 | **Expose internal LM sampling params** (cfg_scale/temp/top_p) + sweep | A | Low–Med | Low | Yes |
| 8 | **Try Epic Music LoRA** (if XL-compatible) for symphonic beds | E | Med | Low–Med | Yes + download |
| 9 | **Train a custom metal/hard-rock rhythm-guitar LoRA** | E | **High ceiling** | High | GPU-heavy |

**Recommended order:** start with the **Mac-side, no-GPU-contention** wins (#1, #2, #4) — they're the biggest bang for the buck and can be built/verified without touching the 3090 — then the cheap GPU experiments (#3, #5, #7), then the downloads/bigger bets (#6, #8, #9).

**Music-quality sources (2026-05-24):**
- Spotify pedalboard: https://github.com/spotify/pedalboard · docs https://spotify.github.io/pedalboard/
- Neural Amp Modeler: https://github.com/sdatkinson/neural-amp-modeler · https://www.neuralampmodeler.com/ · captures https://www.tone3000.com/
- Matchering: https://github.com/sergree/matchering
- ACE-Step Musician's Guide: https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/ace_step_musicians_guide.md
- ACE-Step prompt guide (Ambience AI): https://www.ambienceai.com/tutorials/ace-step-music-prompting-guide
- ComfyUI XL Base workflow (confirms ConditioningZeroOut = empty-negative fallback): https://www.comfy.org/workflows/audio_ace_step1_5_xl_base-536dc32faee1/
- Community XL workflow w/ sampler/APG notes: https://civitai.com/models/2375403 (Civitai — geoblocked here)
- ACE-Step LoRAs: https://civitai.com/models/1962774 (Epic Music) · https://huggingface.co/DisturbingTheField/ACE-Step-v1.5-acoustic-guitar-and-a-merge-LoRA
- YuE: https://github.com/multimodal-art-projection/YuE · DiffRhythm: https://github.com/ASLP-lab/DiffRhythm
- Line 6 Helix Native (VST3/AU): https://line6.com/helix/helixnative.html
- pedalboard external plugins / preset limits: https://spotify.github.io/pedalboard/reference/pedalboard.html · issues #11/#187/#245
- Guitar effect / distortion removal (clean-DI recovery): https://arxiv.org/abs/2202.01664 · https://arxiv.org/abs/2407.16639 (demos: https://y10ab1.github.io/guitar_effect_removal/ , https://joimort.github.io/distortionremoval/ )
- RemFx general audio-effect removal (Apache-2.0, pretrained ckpts): https://github.com/mhrice/RemFx
- Audio→MIDI (clean-DI render route): https://github.com/spotify/basic-pitch · NeuralNote · Jam Origin MIDI Guitar https://www.jamorigin.com/ · Prism
- GuitarML (amp/pedal emulation, forward direction): https://github.com/GuitarML/PedalNetRT
- Stem-output music models: https://github.com/karchkha/MSG-LD · MusicGen-Stem https://arxiv.org/pdf/2501.01757 · StemGen https://arxiv.org/abs/2312.08723
- Symbolic guitar/tab gen: https://github.com/dada-bots/dadaGP · https://github.com/otnemrasordep/ProgGP · GTR-CTRL https://arxiv.org/abs/2302.05393 · ShredGP https://arxiv.org/html/2307.05324
- MIDI→clean DI render (free DI instruments): Shreddage 3 Stratus FREE https://impactsoundworks.com/product/shreddage-3-stratus-free-kp/ · DDSP guitar synth https://arxiv.org/abs/2309.07658
