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
