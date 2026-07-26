# Crucible — Handoff / Onboarding

_App name: **Crucible** (AI metal studio). Repo folder is still `MusicGen` and the backend module is `backend/` — only the user-facing name changed._


_Read this first when picking up the project in a fresh context. It's the index to everything and a snapshot of where we are._

Last updated: 2026-07-24 (the CURRENT STATE section below; everything under it is 2026-05-28 or older)

**Repository:** on GitHub at `git@github.com:apothis/Crucible.git` (public, `main`). `app_config.json` is gitignored (copy from `app_config.example.json`).

## CURRENT STATE (2026-07-24) - READ THIS FIRST

The "Current status" snapshot further down is from 2026-05-24 and covers the AUDIO app only.
Everything video (Characters / MV Studio / Shot Editor, built June 2026) postdates it. This
section is the current routing map; the sections below are history.

**Routing rule:** what runs where = the `app_config.json` VALUE times the code gate. Reading only
the gate's fallback default (e.g. `CFG.get("acestep_dcw_ok", False)`) gives the wrong answer -
this has been got wrong twice. Check the config file.

### Music -> the OFFICIAL ACE-Step engine (`acestep_host`, :8001)

`app_config.json` sets `acestep_dcw_ok: true`, so `use_engine` in `/api/generate` is always true here.

- `/api/generate` (Generate AND the Song builder, all models) -> engine `text2music`.
- `/api/cover` -> engine `cover`. Cover AND Restyle/Reimagine both ride this endpoint: the Restyle
  form branches on `cfg.acestep` and calls `api.cover` with `result_mode: "cover"|"restyle"`
  (`web/src/forms.tsx` ~L391), so `result_mode` only sets the library label.
- LoRA train / eval / adapters -> engine only (ACE LoRAs are PEFT; native ComfyUI cannot load them).
- `/api/repaint` -> ComfyUI `comfy.build_edit` (`acestep_repaint: false`; the engine's repaint
  silence-seeds the region and skips the LM).
- `/api/layer` (Add-a-Layer) -> ComfyUI `comfy.build_lego` (`acestep_lego: false`).
- COLD FALLBACKS, only reachable if `acestep_host` is cleared: `comfy.build_t2m`, `build_restyle`,
  `build_cover`, `build_extract`, the whole `/api/restyle` endpoint, and the ComfyUI VARIANTS model
  picker (the UI shows engine model ids `acestep-v15-xl-*` whenever `cfg.acestep`).

Engine specifics that bite (all in `backend/acestep_py.py` + `app.py`):
- `POST /release_task` does NOT load the requested model - it silently falls back to the loaded
  primary, so `_acestep_ensure_model()` has to `POST /v1/init` first (a ~9GB swap).
- The engine batches (generate 1, cover/repaint 2); extra takes are saved as their own library rows
  tagged `take: 2`, `take: 3`.
- Progress is scraped out of `progress_text` (`N/M` or `NN%`) and folded over `phases` (2 when
  `thinking` is on) into one monotonic bar.
- Every engine output lands as `<jid>.wav` then runs `postfx.tidy_ending`.
- Song builder: `compile` = blocks -> `[section - descriptor]` tagged lyrics + total duration -> ONE
  `/api/generate`; `stitch` = per-block generate at exact lengths -> `/api/stitch` crossfade concat.
  NOTE: on the engine path an INSTRUMENTAL song sends `lyrics: ""`, so the arrangement's section tags
  are dropped (the ComfyUI path kept them via `comfy._structure_only`). Unverified whether the
  engine's own `instrumental` flag compensates.

### Engine version + patch state (verified on the box 2026-07-26, read-only)

- Box checkout = commit **`dce6214`, dated 2026-05-18** (the v0.1.8 release day). Upstream `main` is
  **only 6 commits ahead** (the 2026-06-26 batch of cosmetic fixes, one of which is our own merged PR
  `fix(training): wire sample_labeled_callback through label_all_samples`). **Do not pull:** it buys
  nothing and reverts the DCW patch. No new checkpoints upstream since XL (2026-04-02); no ACE-Step 2.
- Patches confirmed LIVE: DCW off (`inference.py:148` = `dcw_enabled: bool = False`); LoKr `val_split`
  + `timestep_sampling_mode` on the v1 train request; `dropout` on the v2 request;
  `/v1/training/start_lokr_v2` present. Patch 2 (label_all absorber) is now redundant - upstream.
- **Retake + Flow-Edit are on the box but unreachable over HTTP** - see the CORRECTION under
  "Restyle/Cover QUALITY = WEAK" below and RESEARCH §10i's 2026-07-26 update. Candidate patch 8.
- The model we ACTUALLY generate on is **`acestep-v15-xl-base`**, not the code's `xl-sft` fallback
  (measured from `library.db`: the last 400 music jobs are dominated by xl-base at guidance 6 / 64
  steps with LoRAs attached; the persisted UI tuning draft overrides the code default). Matters for
  LoRA compatibility - our own LoKr adapters also train on xl-base, but community XL LoRAs are mostly
  trained on xl-turbo.
- Models are **lazy-loaded**; `/health.loaded_model` reports the CONFIGURED primary
  (`ACESTEP_CONFIG_PATH`, default `acestep-v15-turbo`), not something resident. Changing the
  launcher's primary does not save a load, it just moves it into the first request.

### Stills -> ComfyUI, `still_engine: "krea2"`

- `/api/video/still` -> `video.build_krea2_still` (verbatim port of the AItrepreneur Krea 2 Ultra
  workflow; `enhancer` + `seed_variance` default ON from config and need their custom nodes on the
  box). Z-Image Turbo remains available via `engine: "zimage"`.
- Reference-driven stills (character identity, wardrobe, band composites) -> `/api/video/char_still`
  = Qwen-Image-Edit-2511 fp8. Unaffected by the still-engine switch.
- GOTCHA: Krea2 runs at cfg 1 with `ConditioningZeroOut` as its negative and `build_krea2_still`
  never reads `p["negative"]`, so EVERY caller negative is discarded on the default engine. See
  docs/KREA2.md and Known issues below.

### Video -> ComfyUI, LTX-2.3 22B fp8 + distilled 8-step LoRA at cfg 1

Four graphs in `backend/video.py`:
- `build_ltx_msr` - the spine. Identity comes from REFERENCE IMAGES via the Licon MSR IC-LoRA (no
  keyframe anchor, so motion is prompt-driven). Native single-pass lip-sync = vocal trimmed to
  EXACTLY frames/fps -> `LTXVAudioVAEEncode` -> `SolidMask(0)` + `SetLatentNoiseMask` (preserve, do
  not denoise) -> concat into the AV latent. Optional two-stage: `mode:"hunt"` samples at half res,
  `mode:"finish"` re-runs the SAME stage-1 seed then latent-upscales + low-denoise refines, so the
  finish is provably an upscale of the draft you picked.
- `build_ltx_fflf` - stock `LTXVAddGuide` first/last anchors (each a still OR a clip tail/head),
  2-stage with decoupled seeds. `mode:"hunt"` submits 3 half-res drafts SERVER-side and returns
  `{base_seed, drafts[]}`.
- `build_ltx_keyframe` - faithful LTXDirector 2-stage keyframe port (`base_scale 0.5` so the x2
  upsampler nets back to target). No MSR IC-LoRA: LiconMSR prepends ~17 reference frames and
  corrupts LTXDirectorGuide's absolute `insert_frames`.
- `_build_ltx` (t2v/i2v) - B-roll camera moves, which only execute on the non-distilled dev model.

Post-processing: FlashVSR upscale (`/api/video/flashvsr`, auto `frame_chunk_size: 48` above 200
frames because the whole clip buffers in system RAM) and the ffmpeg grade looks applied per segment
at assemble (`musicvideo.GRADES`, ~20 looks).

Wired but NOT called by any UI: `/api/video/upscale` (SeedVR2, superseded by FlashVSR),
`/api/video/vace`, `/api/video/regrade` + `/api/mv/generate_lut` (the AI-grading scaffold, never
verified against the box). The legacy Video tab still drives Wan i2v, native S2V and InfiniteTalk
v2v directly.

Job plumbing: ComfyUI graph -> `_submit_video` (frees VRAM only when the model SIGNATURE changes,
so back-to-back LTX shots stay warm) -> prompt_id is the job id -> WS progress -> `on_complete*`
downloads into `library/<pid>.<ext>`. `reconcile_loop()` re-checks `/history` every 5s for
unfinished video jobs from memory AND the DB, so a render survives a missed WS event or a restart.

### MV pipeline (song -> finished video)

`/api/mv/script`: when `analyze_host` + an audio id are present it runs allin1 on the REAL audio and
builds a deterministic shot grid from segment boundaries + downbeats (`build_shot_grid`), then the
LLM only fills each fixed window's CONTENT (`build_grid_prompt`); without audio it falls back to
free timing (`build_prompt`). `parse_shots` hard-enforces the pipeline's rules (lip-sync /
performance / character-present shots can never be `keyframe`; wide lip-sync is pulled to medium).
Shots -> `Block`s (`web/src/mvmodel.ts`) -> MV Studio master timeline; each shot is authored in the
staged Shot Editor (Type -> Scene -> Cast -> Placement -> Video -> Result) -> `/api/mv/assemble`
(ffmpeg: per-slot scale/pad/fit, hard cut or xfade, grade, optional intro pre-roll, mux the song).

### Doc trust map

| Doc | Trust |
|---|---|
| VIDEO_PIPELINE_NOTES.md | AUTHORITATIVE - empirical ledger (what works / dead ends / mechanisms) |
| docs/SHOT_EDITOR_MODEL.md (v2 section) | AUTHORITATIVE - the per-shot staged flow that shipped |
| docs/KREA2.md | Accurate, box-verified |
| MUSIC_VIDEO_PLAN.md | PARTLY SUPERSEDED - see the banner at its top |
| docs/LTXDIRECTOR_PIPELINE_PLAN.md | Phases A + B shipped, C partial, D partial (retake shipped) |
| docs/SHOT_STUDIO_FFLF_PLAN.md | SUPERSEDED by SHOT_EDITOR_MODEL v2; its section 0 facts still hold |
| docs/MV_AI_GRADING_PLAN.md | Scaffold only, never verified on the box |
| This file below this section, README.md, PLAN.md, RESEARCH.md | Audio / LoRA history; predate all video work |

### Known issues (verified in code 2026-07-24, all open)

1. **`_write_media_retry()` has no caller.** The HEAD commit "library writes: retry transient EPERM"
   added the helper but `on_complete_media`, `on_complete` and `reconcile_video_job` still do a plain
   `open(path, "wb")`, so the render-loss it was written to fix is still unguarded.
2. **Krea2 discards negatives.** The MSR person-free background negatives in `MVStudio.genStill` and
   `ShotEditor.soloBgNeg`, and the anti-portrait negative in `Characters`, are inert on the default
   still engine - only their positive phrasing is doing the work.
3. **`build_ltx_msr` node-id collision.** The two-stage finish writes nodes 49-59 while the
   MSR-keyframe block below it starts at `kid = 50`. Unreachable today (ShotEditor sends `two_stage`
   without keyframes) but sending both would clobber the upscaler nodes.
4. **Orphaned UI code:** `web/src/ShotStudio.tsx`, `web/src/LtxDirectorEditor.tsx` and the vendored
   editor under `web/src/vendor/` are dead (MVStudio imports only `ShotEditor`; nothing imports
   `ShotStudio`). Delete once the staged Shot Editor is fully trusted.

## Guiding principle (applies to the WHOLE app)
**Research, take good ideas, and enhance.** For every feature, look at how others do it, borrow what's good, but don't be constrained by it — improve on it. We are building a power-user, metal-focused local studio, not a clone of any one tool.

## What this project is
A local, private music-generation studio focused on rock/metal (heavy, power, symphonic, folk). Generate from a prompt, restyle existing tracks, produce vocals separately and mix them, with deep guided tuning and a modern UI. Two machines:
- **Mac** (this repo): FastAPI backend + web UI + MPS/CPU audio tools.
- **Windows + RTX 3090** at `192.168.1.x`: ComfyUI `:8188` (ACE-Step generation), RVC `:7897` (voice conversion).

## Document map
- **`PLAN.md`** — the roadmap: phases 0–6, what's done, what's next, open decisions (D1–D6, all decided).
- **`RESEARCH.md`** — all technical research + verified facts: model survey, vocals, serving, the verified ACE-Step XL workflow & settings, restyle settings, compute-placement rule (§8b), ComfyUI integration patterns, alternative RVC drivers (§8c, `rvc-python`).
- **`UI_DESIGN.md`** — UI/UX research + design brief for the Phase 2 redesign (Suno/Udio/etc. teardown, patterns to steal, enhancements, proposed layout).
- **`HANDOFF.md`** (this file) — entry point + current status.
- **Memory** (`~/.claude/.../project_musicgen-app.md`) — condensed durable facts for the assistant.

Video / music-video docs (all newer than the audio docs above - see the doc trust map in CURRENT STATE):
- **`VIDEO_PIPELINE_NOTES.md`** - the empirical ledger for the video pipeline. Read before any video work.
- **`MUSIC_VIDEO_PLAN.md`** - the original 5-stage MV design (partly superseded; has a banner).
- **`docs/SHOT_EDITOR_MODEL.md`** - the per-shot staged Shot Editor model (v2 = what shipped).
- **`docs/LTXDIRECTOR_PIPELINE_PLAN.md`** - LTXDirector relay / keyframe-mode phased plan.
- **`docs/SHOT_STUDIO_FFLF_PLAN.md`** - FFLF lane + the old Shot Studio (superseded UI, valid FFLF facts).
- **`docs/KREA2.md`** - the Krea 2 Ultra still engine (current default).
- **`docs/MV_AI_GRADING_PLAN.md`** - AI grading design (scaffold only, not verified on the box).

## Current status (2026-05-24)
**Working end-to-end — the whole creative loop is built:**
- ComfyUI generation (ACE-Step 1.5 **XL base**) driven from the Mac; negative-prompt + sampler/scheduler/shift now exposed. Vocals done separately.
- **Crucible UI** (React) — **redesigned** (UI_DESIGN.md §7): grouped left **sidebar** (Create / Guitar / Vocals / Finish), working area with **inline results**, **Library drawer**. Tools: Generate · Song · Restyle · Voc.Builder · Vocals(RVC) · Voice Swap · Import · Stems · **Backing · Guitar · Tone · Master** · Mix. Waveform players, compare grid, Assistant dock (Gemma/Claude).
- **Full vocal pipeline:** CREATE (ACE-Step) → ISOLATE (Demucs/Mac) → RE-TIMBRE (RVC) → MIX. One-click Voice Swap chains it.
- **Guitar pipeline (new):** Backing (strip guitar) → Guitar (AI riff/solo, per-genre, → clean DI via Karplus-Strong/SoundFont/Shreddage → amp) → Tone (reshape) / Master (matchering). Unified genre registry (`backend/genres.py`). See the music-quality section below.

**Stack:** Python FastAPI backend + **React + Vite + TS + Tailwind v4** UI in `web/` (built to `web/dist`, served by FastAPI at `:8000`). The old vanilla `frontend/` is now just a fallback.

- **Song Constructor** (NEW) — a visual, draggable section-block arranger (Intro/Verse/Pre-Chorus/Chorus/Bridge/Solo/Breakdown/Outro). Add/remove/reorder (smooth drag via **@dnd-kit**), edit each block's length + optional per-block lyrics; shows total length. **Song templates** (card grid; `web/src/presets.ts` `SONG_TEMPLATES`) fill a ready-made arrangement + apply that template's own bespoke style (tags/BPM/key) in one click — Radio Anthem, Power Ballad, Prog Epic, Thrash Banger, Folk Singalong, Doom Dirge, Instrumental Showcase, Pirate Shanty. Replacing an edited arrangement asks for confirmation first. Subgenre tag bundles (`PRESETS`, used by the preset chips on Song/Generate/Restyle) now number 15 (Power/Symphonic/Folk/Heavy/Thrash/Doom + Black/Melodeath/Progressive/Gothic/Groove/Djent/Industrial/Viking/Pirate). ✅ Mode (a) verified end-to-end (instrumental 60s gen via the restarted backend). Two drive modes: **(a) Compile** — blocks → one ACE-Step structured-lyrics prompt + total duration → `/api/generate` (order/lyrics honored, section length approximate); **(b) Per-block + stitch** — generate each block to its exact length, then crossfade-concat via `POST /api/stitch` (exact lengths, per-block re-roll, lockable blocks). Stitched songs save to the library as mode `song`.

**Code layout:**
```
app_config.json          hosts/ports/paths (comfy_host, rvc_host, comfy_input_dir, server_port)
run.sh                   ./run.sh -> http://127.0.0.1:8000
backend/
  app.py                 FastAPI: endpoints, WS progress listener, SQLite library
  comfy.py               ComfyUI client + ACE-Step workflow builders (t2m + restyle)
  rvc.py                 RVC client — legacy Gradio 3.14 driver (+ ComfyUI-input-dir bridge)
  rvc_py.py              RVC client — clean REST API driver (base64 /convert)
  rvc_server.py          ⟵ runs ON the Windows RVC package (its runtime\python.exe);
                            our custom API reusing the package's working env. Speaks
                            the rvc-python dialect so rvc_py.py/voices.py are unchanged.
  voices.py              voice download/install helper (HF search + /upload_model to PC)
  stems.py               Demucs stem separation (runs on Mac MPS, in parallel w/ 3090)
  mix.py                  in-app mixer — layer tracks (gain/offset) -> bounce WAV
  requirements.txt
web/                     PRIMARY UI — Vite+React+TS+Tailwind v4. Built to web/dist and
                           served by FastAPI at :8000. Dev w/ hot-reload: cd web && npm run dev
                           -> http://localhost:5173 (proxies /api -> :8000).
                           src/: api.ts, ui.tsx, forms.tsx, WavePlayer.tsx, Assistant.tsx, presets.ts, App.tsx
frontend/                CLASSIC UI (vanilla JS) — now only a fallback if web/dist is absent.
.claude/launch.json      preview_start config (name "web") for the dev server
library/                 generated audio + library.db (SQLite)
test_output/             scratch audio from manual tests
*_AUTO_INSTALL.bat       Windows installers (ComfyUI+ACE-Step, RVC WebUI, RVC API server)
.venv/                   python 3.9 venv
```

**Run:** install deps (`python3 -m venv .venv && ./.venv/bin/pip install -r backend/requirements.txt`; `cd web && npm install && npm run build`), then `./run.sh` and open `http://127.0.0.1:8000` (FastAPI serves the React build). UI dev: `cd web && npm run dev` → `:5173`. Edit `app_config.json` for the Windows hosts.

## Key verified technical facts (details in RESEARCH.md)
- **ACE-Step 1.5 XL wiring:** `DualCLIPLoader(qwen_0.6b, qwen_4b, type="ace")` — slot order matters, 4b not 1.7b. XL base = 50 steps, cfg 6, euler/simple, ModelSamplingAuraFlow shift 3. Single CLIPLoader fails (issue #12278).
- **ComfyUI API:** `/run`? no — POST `/prompt` (API-format graph) → `/ws` progress (match client_id; done = executing w/ node==null) → `/history` → `/view`. Only `xl_base` diffusion model installed (sft/turbo not downloaded).
- **RVC integration (current, working):** we run **`backend/rvc_server.py`** INSIDE the bundled RVC WebUI package via its `runtime\python.exe` (reuses the working fairseq/torch env; Gradio supplies fastapi/uvicorn — no new install). It exposes a clean REST API (`GET /models`, `POST /models/{name}`, `POST /params`, `POST /convert` base64-in/WAV-out, `POST /upload_model`). Installed by `RVC-API_AUTO_INSTALL.bat` on `:5050`. The Mac's `rvc_py.py` client + `rvc_driver: auto` use it; `rvc.py` (Gradio) is a fallback. **`rvc-python` was abandoned** — its `fairseq==0.12.2` has no Windows wheel (see `RESEARCH.md §8c`). Starter voices installed: `FreddieMercury`, `james_hetfield`. Add more via the Vocals-tab helper.

## Research index (all in RESEARCH.md / UI_DESIGN.md)
- Model survey, vocals research, serving, verified ACE-Step XL wiring/settings, restyle settings → `RESEARCH.md` §1–8.
- Compute placement (Demucs on Mac MPS, decided) → §8b. Alternative RVC drivers + why rvc-python was dropped → §8c. Downloadable RVC voices → §8d.
- UI/UX teardown + design brief (incl. Song Constructor §4.10) → `UI_DESIGN.md`.
- _(Note: background sub-agents sometimes had web access denied; do web research in the main thread if a sub-agent is blocked.)_

## Operating preferences
- **Never auto-play audio** (no `open`/VLC) and don't kick off GPU generations without the user expecting it — ask first. Report file paths instead.

## Vocal Builder (NEW — AI melody → singing)
A new **Voc. Builder** tab (`web/src/VocalBuilder.tsx`) that **composes a vocal melody with AI** from a Song Constructor arrangement and sings it. Pipeline (RESEARCH.md §5b):
- **AI Melody Composer** (`backend/melody.py`): hybrid LLM (Claude/Gemma via `llm.py`, auto-pick) proposes the melody as scale **degrees** (in-key by construction, trivial to parse); a music-theory layer realizes them — degrees→MIDI in a vocal range snapped to the key's scale, verse-low / chorus-lift contour, notes laid into each section's time window, cadences at phrase ends. Pure-algorithmic fallback guarantees output. Syllables via `pyphen`, MIDI via `mido`. Endpoint `POST /api/melody/compose` → score; `POST /api/melody/midi` → .mid download.
- **Engine-agnostic synthesis** (`backend/voicegen.py`, `GET /api/vocal/engines`): `guide` (Mac synthetic 'ah' vocal — works now), `soulx` (SoulX-Singer zero-shot, sings words + clones timbre from a ref clip), `diffsinger` (voicebank). Host engines speak `POST /synthesize` + `GET /health`; configured via `soulx_host`/`diffsinger_host` in `app_config.json` (empty = shown as "needs host"). `POST /api/vocal/build` synthesizes → optional RVC re-timbre → library mode `vocal`.
- **UI**: pulls the live Song arrangement (lifted to App state, `SongDraft`), composes, shows an **SVG piano-roll** (notes coloured per section, syllables on notes, section bands), engine selector w/ availability, RVC re-timbre toggle, MIDI export.
- **Status**: Mac path (compose → guide → library) + MIDI export verified end-to-end (real local Gemma compose, in-key contour confirmed). RVC re-timbre wired but not run here (no-surprise-GPU rule).
- **Word-singing engines (code + installers built, PC-verification pending):**
  - **SoulX-Singer** — `backend/soulx_server.py` (FastAPI in the SoulX repo env: `/health` + `/synthesize`; builds SoulX score-control metadata from our melody via `g2p_en` + per-word grouping; bundled English prompt or posted reference clip; `model.infer` per segment; 24 kHz). Installer `SOULX-API_AUTO_INSTALL.bat` (clone + venv + deps + `hf download` model + run on :5060). Interface verified against the repo (RESEARCH.md §5c); confirm `note_type`/VRAM/prompt-quality on first run.
  - **DiffSinger** — driven via **openvpi DiffSingerMiniEngine** (native API). `backend/voicegen.py` `_diffsinger_synth` does `/models→/rhythm→/submit→/query→/download` with f0 built from the melody. Installer `DIFFSINGER-MINIENGINE_AUTO_INSTALL.bat` (:9266). Needs an English voicebank+dictionary (RESEARCH.md §5d). `g2p_en` is a Mac dep for its phonemes.
  - Set `soulx_host` / `diffsinger_host` in `app_config.json` to light them up (the Voc. Builder shows availability live).
  - **SoulX voices are zero-shot** (the reference clip *is* the voice). A **named voice library** (`web/src/VocalBuilder.tsx` picker + `POST /api/vocal/soulx/prep` → server preprocess) lets you prep a clip once and reuse it. The **Import tab** (`web/src/Import.tsx`) is the full song→voice pipeline: import a song → drag a region on the waveform → `POST /api/import/extract` (trim + Demucs vocal on the Mac) → preview → save as a SoulX voice. Don't need metal/solo refs — SoulX gives the performance/vibrato, RVC sets the final identity, and Demucs isolates a vocal from any mix. _Note: do NOT install `preprocess/requirements.txt` — it pins torch 2.10/numpy 2 and would clobber the working CUDA torch; its deps are already in the main env._
  - **Shared-GPU safety:** the SoulX server loads its model **on demand and unloads after each synth** (default; `MG_SOULX_KEEP_RESIDENT=1` to keep it loaded), and `backend/app.py` calls ComfyUI `POST /free` before any GPU vocal build (host engine or RVC re-timbre) so models don't collide on the 3090. Verify SoulX's actual footprint with `nvidia-smi` on first load.

## MUSIC-QUALITY PUSH — built this cycle (2026-05-24)

Scope includes **heavy rock** (Bon Jovi / Halestorm / Black Stone Cherry / AC/DC) alongside heavy/power/symphonic/folk metal. Full research in `RESEARCH.md §10`; UI redesign in `UI_DESIGN.md §7`. The whole **guitar-tone problem** got an alternative, controllable pipeline (since ACE's distorted guitars are the weak spot and can't be cleanly de-amped):

**The guitar pipeline (new tabs Backing / Guitar / Tone / Master):**
- **Backing** — 6-stem split (`htdemucs_6s`) → recombine WITHOUT the guitar = a guitar-less bed. (`backend/stems.py` 6-stem, `backend/postfx.py` recombine.)
- **Guitar** (`backend/guitar.py`) — generate a guitar part → **clean DI** → amp → optional mix onto a backing.
  - *Source:* Test riff · **Song arrangement** (per-section riffs from the Song Constructor) · MIDI upload.
  - *Riff brain:* Algorithmic (instant) or **AI** (local Gemma / Claude) — LLM writes a 2-bar pattern (or solo phrase) on a 16th grid.
  - *Part:* **Riff** (power chords) or **Solo** (single-note lead, high register), each genre-characteristic.
  - *DI render engine (pluggable):* **Karplus-Strong** (synth, no deps) · **SoundFont** (FreePats Fender, `soundfonts/eguitar_clean.sf2`, fluidsynth) · **Shreddage/Kontakt** (best — capture once via the setup button).
  - *Amp:* the tone presets / **Helix Native** (captured tones reused via `raw_state`).
- **Tone** (`backend/postfx.py`) — reshape an existing distorted guitar stem (EQ/cab/sat presets or Helix; delta-recombine, gain-matched + latency-aligned). Honest: this *reshapes*, it doesn't re-amp from clean.
- **Master** (`backend/master.py`, Matchering 2.0) — match a track to a reference master you own.

**Unified genre registry** — `backend/genres.py` is the SINGLE source of truth (22 genres incl. neoclassical), served via `/api/config` → drives both the generation preset chips AND the Guitar riff/solo genre pickers. Genre **suggests** bpm/key (chips set tags only; suggestions shown as hints) — never forces.

**Source tuning** — `comfy.py` exposes real **negative prompts** + **sampler/scheduler/shift** (Expert tuning). Only `xl_base` installed.

**Plugins (this Mac):** Helix Native VST3 (licensed, wired) at `/Library/Audio/Plug-Ins/VST3/Line 6/Helix Native.vst3`; Kontakt 7 + Shreddage 3 Stratus FREE (Kontakt state captured to `soundfonts/kontakt_guitar.state`). New `app_config.json` keys: `helix_vst3_path`, `helix_default_preset`, `guitar_ir_path`, `guitar_soundfont`, `kontakt_vst3_path`. `soundfonts/` + any `RemFx/` are gitignored.

**UI redesign** (`UI_DESIGN.md §7`): grouped left **sidebar** (Create / Guitar / Vocals / Finish) replaced 13 flat tabs; working area shows **results inline**; **Library is a collapsible right drawer**. Fixed WavePlayer silent playback (MediaElement backend).

## Open / next (for a fresh context)

### SESSION 2026-05-30 — Scaling experiment + research deep-dive + new plans + Nightwish control

**Headline finding:** **35 tracks made the LoRA WORSE, not better, despite lower val_loss.** The 6-track / 150-ep run from session 2026-05-29 remains our best-sounding adapter. User listening tests on 4 takes (best+final × 0.3+0.5 strengths, fixed seed+prompt) confirmed regression. **val_loss is misleading at our scale** — lower MSE on 5 val samples ≠ better perceptual quality.

**Pipeline rebuild + 200-ep overnight run (35 power-metal tracks):**
- New AcoustID-first artist resolution in `backend/lora_dataset.py:_resolve_artist_title_full` (commit `60896ce`) — priority chain is explicit args → ID3 tags → AcoustID fingerprint → filename parse. Promotes AcoustID above filename-parse so tag-less audio still feeds accurate artist into LRCLIB. ID3 wins over AcoustID per user preference.
- Mac-side training history capture (commit `9f6b613`) — spawned from `/api/lora/train`, polls `/v1/training/status` every 2s, parses Patch D's `🧪 / 🏆` status messages, persists per-dataset to `library/lora_train_history/<dataset>.json`. Engine never exposed `plot_best_step`; this fills the gap. New `GET /api/lora/train/best_history?dataset=X`. Caught the full 200-ep curve: 5 best updates, ep 1 → 105, val 0.808 → 0.692.
- Lyrics_source persisted in per-track .json meta (commit `e618691`) — UI badge survives reload. Future uploads benefit; existing crucible_metal + crucible_nightwish keep "none" badges since their JSONs predate the fix.
- 200-ep / 35-track overnight run completed cleanly: 8h 44m, 1600 steps, best at epoch 105 (val 0.6919). **But epoch 27 was statistically tied (val 0.697, Δ=0.005 inside noise band of 0.20)**. Useful learning concentrates in first ~30 epochs. **Future runs on similar-sized data should use 50 epochs not 200.**

**Research deep-dive + new plans written into METAL_LORA_PLAN (commit `0434ed6`):**
- **§11 Plan 1 — Improved evaluation (no VRAM risk)**: CPU-only weight introspection during training (norm growth, dora_scale, SV spectrum, train/val gap) + post-training per-checkpoint CLAP fitness scoring + centroid distance via embedding of training corpus. Architecturally avoids in-training generation (VRAM swap-in cost is the killer). Explicit checks #1A/#1B/#1C with stop conditions.
- **§12 Plan 2 — Continuous timestep sampling A/B (engine Patch 7)**: hypothesis that v1's discrete-8-timestep training distribution doesn't match xl-sft's 32-64-step continuous inference path, possibly explaining the smear. Pre-flight 2.0 captures reference takes; phase 2.1 patches the trainer with toggleable mode; phase 2.2 fires a calibrated 50-ep A/B; phase 2.3 runs Plan 1's post-eval. Concrete win/tie/loss decision criteria with actions for each.
- **§13 — Overnight 200-ep / 35-track stats** measured on our box.
- **§13a — Research catalog** (new, this session): the 3 training paths that exist (v1 HTTP API we use, v2 CLI in-tree, Side-Step community); existing HF LoRA collections + why they don't load on our XL 4B; adapter algorithm landscape (LoRA / DoRA / LoKr / LoHA / OFT / BOFT) with which paths support which; quality measurement landscape (7 approaches evaluated, why CLAP wins on ROI for us); val-loss-is-misleading empirical finding; DoRA discussion + Plan 3 deferral.

**Nightwish single-band 6-track control experiment** (task #12, in flight):
- Hypothesis test: does single-band, single-vocalist scope reduce the smear seen in 35-track mixed? Or is the training process itself the issue (Plan 2's hypothesis)?
- 6 Tarja-era Nightwish tracks: Sleeping Sun, Ghost Love Score, End Of All Hope, Wishmaster, She Is My Sin, Sacrament Of Wilderness. Spans tempo (99-172 BPM) + dynamics (intimate ballad to epic 10-min) + key (D/F/G/A minor) but holds vocal/production era constant.
- Dataset name `crucible_nightwish` (separate from `crucible_metal` 35-track) so prior artifacts stay intact for future comparison.
- Pipeline status as of this commit: ✅ uploaded, ✅ scanned, ✅ saved, ✅ autolabeled (6/6, 12 min). Merge + preprocess + engine restart + training next.
- Training config: SAME as session 2026-05-29's 6-track 150-ep run (plain LoKr, lr=0.01, val_split=0.1, seed=42, 150 ep). One variable changes vs that baseline: content (power-metal-mixed → Nightwish-Tarja).
- Decision criteria: clearly better → data scope is the dominant lever; about the same → Plan 1+2 right direction (training process is the issue); worse → Plan 2 (continuous sampling) gets priority.

**Tasks open at end-of-session:**
- #6 UI: DoRA + LR override + val_split (still pending from session 2026-05-29; would be useful once Plan 3 lands)
- #10 **Plan 1 — Improved evaluation** (the big infrastructure ship; needed for Plan 2 to be quantitative)
- #11 **Plan 2 — Continuous timestep sampling A/B** (depends on Plan 1)
- #12 **Nightwish single-band experiment** (in flight, training pending engine restart)
- _Implicit Plan 3 — DoRA re-enable A/B_ — to be authored after Plan 2 resolves

**Where to start the next session:**
1. **Read METAL_LORA_PLAN §11 §12 §13 §13a first** — that's the canonical reference for Plans 1+2 + research catalog. This block is the session summary; §§11-13a are the substance.
2. **Listen to the Nightwish training output** (assuming it ran while user was AFK). 4 takes in the Library, format "Crucible Nightwish 6-track 150ep — {BEST|FINAL} @ {0.3|0.5}".
3. **Decision triage on the Nightwish result** per §11.5 / §12.3 / §13a.8 criteria → which plan to ship first.
4. **Plan 1 ships first** regardless (it's the foundation for Plan 2 being quantitative). Patches A through D from session 2026-05-29 are live + we have the best-history poller — Plan 1 builds on those.

**Operating reminders that bit us this session and are worth keeping front-of-mind:**
- _val_loss is broken as a primary signal at our scale_ — quantitative comparisons need CLAP fitness via Plan 1 not just MSE
- _Engine doesn't expose `plot_best_step` / val curve over `/v1/training/status`_ — our Mac-side poller now captures it via `🧪 / 🏆` status string parsing. Without that we can't tune `train_epochs` rationally
- _More data ≠ better adapter without scope discipline_ — the 14-band 35-track mix produced a worse adapter than 6-track focused. Future runs should default to subgenre or single-band scope until evidence supports broader.
- _In-training generation is unsafe_ on our 22-24 GB VRAM working set — Plan 1 explicitly post-hoc-evaluates against on-disk checkpoints with fresh engine state

### SESSION 2026-05-29 — Engine memory-leak diagnosed + DoRA bug + 1st-and-2nd LoKr trained + 3 engine bugs found

Marathon training day. Pipeline VERIFIED end-to-end on the box (uploaded → captioned → preprocessed → trained → exported → loaded → generated), but adapter quality is held back by engine-level bugs. Findings deep enough that next session should patch the engine before another training run.

**🐛 Engine memory leak in `/v1/init` — diagnosed + fixed Mac-side**
- The engine's `/v1/init` handler reassigns `self.model/self.vae/self.text_encoder` when loading new models but only sets the OLD ones to `None` on FAILURE paths. On success it never calls `gc.collect()` or `torch.cuda.empty_cache()` (verified in `acestep/core/generation/handler/init_service_orchestrator.py:144-217`).
- Each `/v1/init` leaks ~4 GB of orphaned CUDA tensors. Over a multi-init session, leaks compound until VRAM saturates (24 GB) and training goes memory-bandwidth-bound: **99% GPU util + silent fan + ~7× slower steps** (315 s/epoch vs the smoke's 42 s/epoch).
- **Fix shipped (commit 7b2c427)**: `_ensure_training_ready` now calls `/v1/reinitialize` instead of `/v1/init`. The engine endpoint runs `gc.collect() + torch.cuda.empty_cache()` AND is documented in the engine's own `start_lokr` error message (`"Decoder not found. Please reload the model via /v1/reinitialize before training."`). The engine's `start_lokr` handler does its OWN component management (`mgr.unload_llm()` + decoder-to-GPU + VAE offload) — Mac doesn't need to drop the LM.
- `_ensure_labeling_ready` made idempotent: skips `/v1/init` when engine state already matches (xl-sft + 4B LM loaded). Dodges the leak on the no-op case.

**🐛 LoKr DoRA + lr 0.03 catastrophic combo — diagnosed + safer defaults shipped**
- Engine ships `lokr_weight_decompose=True` (DoRA) + `learning_rate=0.03` as defaults. **These are catastrophically incompatible.** DoRA's `dora_scale` magnitude parameter needs lr~1e-4..1e-3 (separate param group ideally), not 0.03.
- First 100-epoch run produced corrupted weights: dora_scale tensors mean=2.99..4.62, max=**19.18** (initialized at norm of base column ≈ 1-2). Loss "spike" at last step (0.211 → 0.852) wasn't divergence — it's an engine artifact where the trainer emits an aggregated loss at the final-save event. The weights are bad because DoRA's magnitude is blown up, not because of divergence.
- Garbled output at ANY strength because LyCORIS's `set_multiplier(0)` only zeros the delta term, but **`dora_scale` keeps scaling the base computation independently** (`y = (dora_scale/norm) * direction(x) + multiplier * delta(x)`). Toggle-off → garbled. Only full UNLOAD restored clean base.
- **Fix shipped (commit 7b2c427)**: `/api/lora/train` now defaults `lokr_weight_decompose=False` and `learning_rate=0.01`. Plain LoKr — no DoRA, no exploded magnitudes. Callers can opt in to DoRA explicitly when paired with a low lr.

**🐛 Three more engine bugs discovered (not yet patched — see task #19)**
1. **`lokr_config.target_modules` is silently IGNORED.** Engine's `inject_lokr_into_dit` calls `LycorisNetwork.apply_preset({"target_name": user_targets})` BEFORE `create_lycoris()`. But `create_lycoris` defaults `preset="full"` and calls `apply_preset(PRESET["full"])` AGAIN, overwriting. Whatever the user passes is discarded — they always get LyCORIS's `"full"` preset.
2. **Time-embed layers wrapped but learn nothing** — 104/360 `lokr_w2` tensors all-zero in the trained adapter. Structural: timestep input is random per-batch, no consistent gradient signal. ~29% of adapter capacity wasted. Engine should exclude time_embed-style layers from default wrapping.
3. **`val_split` hardcoded to 0.0** in `TrainingConfig`, not exposed via `StartLoKRTrainingRequest`. → No validation loss tracking → `checkpoints/best/` never written → we have no way to pick the best mid-training checkpoint, only "final" (which is post-aggregated-loss artifact).

**🐛 VRAM never physically releases** (task #18) — logically the engine reports LM unloaded (`loaded_lm_model: None`), but VRAM stays at 22+ GB. Even `/v1/reinitialize`'s `empty_cache()` only freed ~4 GB in our run, the rest stuck in PyTorch's caching allocator + vLLM's pool. Only OS restart truly releases. Investigation pending.

**Marathon training results (the 6-track power-metal corpus from previous smoke):**
- **Run 1 — broken (DoRA + lr 0.03)**: 100 epochs in 55 min on 3090. Output garbled at any strength even toggle-off. File inspection of safetensors confirmed dora_scale explosion (max 19.18). Adapter mounted but `discover_adapter_names()` returned `[]` (this is normal for LoKr — discovery only checks PEFT attrs).
- **Run 2 — plain LoKr (lr 0.01, no DoRA)**: 50 epochs in 38 min. File inspection confirmed clean weights (zero NaN/Inf, magnitudes 3-4× smaller than Run 1). Output produces real music, scale gradient works (0.3 less garbled than 0.75), but quality is "messy" — adapter learned more like "perturbation" than "coherent metal direction." We don't yet know how much of the quality limit is the 3 engine bugs above vs dataset size vs training config — we've only ever trained 6-track runs. **The METAL_LORA_PLAN §10.2 pyramid numbers (track counts, epoch sweet-spots) are cited from external upstream guides (Side-Step, engine tutorial) — NOT validated on Crucible's setup.** Treat them as starting hypotheses, not facts.

**Other UI/UX improvements shipped this session (commit 60ce396):**
- **Rich training progress block** — epoch %, step counter, derived rate (steps/s) and ETA (engine reports both as 0.0; UI derives from `loss_history.length + start_time + current_epoch`), loss sparkline (NaN-safe, handles engine's `{step,loss}` object shape), collapsible log tail, TensorBoard ↗ link.
- **TensorBoard URL rewrite** — engine reports `http://localhost:6006` (loopback ON THE BOX); Mac now rewrites to `http://<acestep_host>:6006` so the link works from any LAN browser.
- **Chain robustness** — Save+preprocess+train button now WAITS for preprocess to finish (was firing train immediately → empty tensor dir bail). Polls preprocess status, detects engine `"❌"` errors, verifies `num_tensors > 0`. Captures prior task_id to skip stale "completed" states from earlier runs.
- **chainPhase banner** — "1/3 Saving... 2/3 Preprocessing... 3/3 Starting training..." visible from the click moment (bridges the 12s gap before status poll catches up).
- **Train LoRA tab rehydrates** from engine `/samples` on mount → hard-refresh during a long upload no longer wipes back to Step 1.
- **Caption editor is a 5-row textarea** (merged captions run 700-900c).
- **Step ✓ badges are emerald** (was red/orange via `--color-accent`).
- **gradient_checkpointing default ON** (smoke calibration: ~2.5× faster on XL/4B LoKr).
- **Epoch progress reads `training.config.epochs`** (the engine's actual field name — was looking at `train_epochs`/`num_train_epochs` and falling back to React state which defaulted to 500 → bar showed 56/500 for a 100-epoch run).
- **Last.fm tag filter tightened** (count ≥ 10 on track-level too — was only artist + album; track-level was getting fan-folksonomy noise like "metalcore" tagged on Sabaton by ~3 users).

**Verified-working pipeline as of end-of-session:**
- Mac fixes have made it leak-resistant. Plain LoKr default avoids the DoRA blowup. Training produces real (if rough) adapters.
- Engine state hygiene is a real ongoing concern. After a marathon session of inits/loads/unloads, VRAM stays pegged at 22 GB and base inference quality degrades (we saw base-only generation become "garbled" by end of session; full LoRA unload restored it).
- For best results next session: **start with a fresh engine boot** (OS-level restart of `run_acestep_api.bat`) before any training, to get clean baseline VRAM and clean decoder state.

**Where to start next session (in order of recommended priority):**
1. **Patch the 3 engine bugs (task #19)** — they're all upstreamable. Local patches per `[[engine-patches]]` pattern: (a) pass user preset to `create_lycoris`, (b) exclude time_embed from default targeting, (c) expose `val_split` in `StartLoKRTrainingRequest`. Together they enable a meaningful retrain.
2. **Consider expanding the dataset** — but be honest that we don't know how big it needs to be. METAL_LORA_PLAN §10.2 suggests 20-40 for subgenre LoRAs based on UPSTREAM guides; we've never tested above 6. After the engine patches land, recommended approach is to retrain at the SAME 6-track size first (so the only variable changing is engine-bug-free vs not), then separately scale dataset to triangulate whether dataset size was actually limiting quality.
3. **Then retrain** with: tighter target_modules (just q/k/v/o on attention, skip MLP/time_embed/condition), `val_split=0.1`, lr=0.01 plain LoKr, 50 epochs first then dial up. Check `checkpoints/best/` exists at end.
4. **Ship UI controls** for DoRA toggle + lr override + val_split (#17). Once engine accepts val_split, expose in Advanced.
5. **Investigate VRAM-stickiness (#18)** — likely PyTorch caching allocator + vLLM pool not releasing. Worth a small engine patch with `torch.cuda.caching_allocator_delete` or explicit pool reset.
6. **Autolabel + chain UI cohesion (#14, #10)** — autolabel has no progress block today, and the orange chain button isn't disabled mid-autolabel. Both ship-tomorrow polish.

**Tasks open as of end-of-session:**
- #10 Cohesive chain UI (stitch save → preprocess → training as ONE visual flow)
- #14 Autolabel progress block + disable chain button during autolabel
- #15 Clean A/B: grad_ckpt OFF vs ON on a clean engine (the original smoke confounded)
- #16 Retrain LoKr (covered by recommendations above)
- #17 UI toggle for DoRA + LR override
- #18 Investigate why VRAM never released after engine logical LM unload
- #19 Engine patches: target_modules + time_embed exclusion + val_split

### SESSION 2026-05-28 — Metal LoRA pipeline + smoke-tested + UI polish (all shipped, on `main`)
The big LoRA-training push: full Mac→box pipeline built, calibrated on the 3090, one upstream PR opened, one engine source patch you must keep applied locally. **Plus a lot of unrelated UI polish that landed this session — listed first.**

**UI polish + small fixes shipped (not LoRA):**
- **Song names + version flow everywhere.** Download filenames now use `params.title` + `-vN`; cross-tab pickers (Master, Restyle, etc.) show `song: <title> · vN` via shared `trackLabel()` in `web/src/api.ts`. Backend `/api/export/{pid}` computes the version index from the jobs table.
- **Auto-master enhancement stages** (off by default per [[optional-additions]]) — `master.py` gains a Mid/Side **stereo widener** (with keep-bass-mono HP on the Side), a **warmth** (gentle tanh saturation) stage, and a UI `Enhancements` toggle in the Master tab that exposes width/bass-mono-freq/warmth. All before the limiter so peaks stay pinned to −1 dBFS.
- **Song Builder per-section style descriptor** — optional free-text per block compiles to `[Intro - spoken word]`-style ACE descriptors (verified in the ACE-Step 1.5 musician's guide); datalist of curated styles (`spoken word, whispered, anthemic, screamed, growled, gang vocals, …`) attached to every block input. Both Compile + Stitch drive modes pick up the descriptors.
- **Stitch cohesion + dead-air fix.** Added two stitch toggles (default ON): **Share one seed across sections** (cohesion) + **Beat-match section seams** (`mix.py` trims each clip's lead-in to its first onset AND trims trailing near-silence — fixes the "24s blocks only filling 20s" gap caused by ACE's natural prose-pad behavior). Both before the equal-power crossfade.
- **XL-SFT is now the default model** for engine-mode forms (was xl-base). Draft store doesn't persist defaults so this applies everywhere unless the user explicitly chose otherwise. Memory [[base-sft-default]] updated to reflect the concrete default.
- **Settings panel** — header **⚙** button opens a modal that edits the curated `app_config.json` keys grouped (Box services / Engine flags / API keys / Mac server). Backend `GET /api/settings` returns a self-documenting field list with hints; `PUT` whitelist-validates + preserves unrelated keys + reports `restart_required`. Secret inputs password-masked with show toggle.
- **LM bug-fix for "whispered intro carrying across the whole song":** added "sung" / "belted" / "full voice" / "powerful vocals" to the SECTION_STYLES datalist so the user can tag the first verse to push the LM back to normal singing after a `[Intro - whispered]`.

**Metal LoRA — the main push (METAL_LORA_PLAN.md is the live plan + working doc):**
- **Phase 0 — Research + box verification.** RESEARCH §18/§18a + METAL_LORA_PLAN. Engine exposes the WHOLE pipeline over HTTP (no Gradio needed): `/v1/dataset/{scan,auto_label[_async],save,preprocess[_async],samples,sample/{idx}}` + `/v1/training/{start,start_lokr,status,stop,export}` + `/v1/lora/{load,scale,toggle,unload,status}`. Live-probed on the box's installed engine build — all routes present.
- **Phase 1 — `backend/acestep_train.py`.** Mac client for every endpoint (dataset/training/lora) + async poller + `init`/`reinitialize` helpers. Verified against live engine.
- **Phase 2 — `backend/lora_dataset.py`.** Mac enrichment: librosa beat-track BPM + Krumhansl–Schmuckler key (→ valid `comfy.KEYS` string), faster-whisper lyrics (`asr.py`), and `bundle_for_track()` returning `[(audio), (name.json), (name.lyrics.txt)]` as `(filename, bytes)` ready for the box helper. Verified live (bpm 161 / E major / lyrics on a real track).
- **Phase 3 — Box upload helper.** `backend/lora_upload_server.py` + `LORA-UPLOAD_AUTO_INSTALL.bat` + `backend/lora_upload_py.py` Mac client + `lora_upload_host` config key. ⚠ The dataset root MUST live under the ACE engine's launch CWD (the engine's `acestep/training/path_safety.py` rejects anything outside). The installer warns + the engine launcher does `cd /d "%ENGINE%"` where `%ENGINE% = %DEST%\ACE-Step-1.5`, so the dataset root has to be e.g. `<install>\ACE-Step-1.5\lora_data`. Current live root on the box: `E:\AI\MusicGen\AceStep\ACE-Step-1.5\lora_data`. Helper also has `/dataset/delete` for full-folder cleanup (added late; needs a `run_lora_upload.bat` restart to take effect on the box).
- **Auto-caption layering — `backend/caption_fetch.py`.** Per track at upload time: **AcoustID + chromaprint/fpcalc** (identify untagged audio → artist+title, free key + `brew install chromaprint`) → **MusicBrainz** (recording tags + artist disambiguation + first inline release title — no key) → **Last.fm** (`track.getTopTags` + `artist.getTopTags` + `album.getTopTags` merged, popularity-filtered, free key) → **mutagen ALBUM tag** (local truth, usually the studio album rather than MB's first-release match) → **CLAP via the box analyze service** (audio-grounded subgenre tags, reuses §17a). Output merged, deduped case-insensitively, specific subgenres first via `_merge()`. Verified live on Nightwish – Wish I Had an Angel: sources `[acoustid, musicbrainz, lastfm, lastfm-album, clap]`, caption `finnish symphonic metal, gothic metal, symphonic metal, power metal, progressive metal, …`. **Both opt-in keys (`lastfm_key`, `acoustid_key`) are set in the live app_config.**
- **Phase 4 — Backend orchestration + Training tab UI.** `/api/lora/{status, dataset/add, dataset/scan, dataset/samples, dataset/sample/{idx} (PUT), dataset/autolabel(+merge), dataset/save, dataset/preprocess, dataset/preprocess/status, train, train/status, train/stop, export, load, scale, toggle, unload}` — Mac orchestrates the box via the engine's HTTP routes. New Lab → **Train LoRA** tab (`web/src/LoraTraining.tsx`): collapsible "What is a LoRA" explainer, live box-services preflight, 4 guided steps (file-picker add → load+review/edit table via engine `GET/PUT sample` → train w/ LoKr default + epoch hint scaled to track count + GPU warning + live epoch/step/loss/ETA → export+load). Self-contained; renders preflight cleanly when box services are down.
- **Phase 5 — Metal LoRA control on Generate.** `MetalLoraControl` in `web/src/forms.tsx` (visible when `cfg.lora_train`, not expert-gated): toggle + strength slider (range 0–1.5, commits scale on release) driving `/api/lora/{toggle,scale}` engine-globally. Shows the loaded adapter name; hint pointing to Lab → Train LoRA when none loaded. Verified: renders the "none loaded" state live.
- **Phase 6 — Smoke test on real power-metal data ✅ end-to-end.** 6 Sabaton/Avantasia/etc. tracks (`/Volumes/SSD1/Downloads/This-is-Power-Metal-Mini-ZIP-…`) → enrich+upload (with full caption layering, AcoustID resolved everything) → scan → save → preprocess (25 s on the 3090; healthy GPU work; engine auto-swapped LM out during preprocess and restored) → train. **Calibrated A/B for grad_checkpointing on real data:**
  - **gradient_checkpointing=True** → epoch 1 ~61 s (incl. setup), epochs 2–5 ~**41.5 s/epoch avg**, 5 epochs in 227 s.
  - **gradient_checkpointing=False** → epoch 1 ~127 s, epochs 2–5 ~**104 s/epoch avg**, 5 epochs in 544 s.
  - **Counter to textbook: grad_ckpt=ON is ~2.5× FASTER here.** Without checkpointing the 4B XL decoder backward saturates VRAM bandwidth → GPU stalls waiting on memory (bandwidth-bound). With checkpointing, less memory pressure → SMs stay compute-bound (audible: GPU fan spun up on Run A, near-silent on Run B). **Default `gradient_checkpointing=True` for XL/4B LoKr.**
  - **Real-world planning numbers on this rig:** a 50-epoch LoKr ≈ **~35 min**. A 500-epoch run ≈ **5.8 hr**. Per `METAL_LORA_PLAN §10.4a`.
- **Engine state safeguards built (and required) for training.** The FIRST attempted 50-epoch run was stuck CPU-bound for 5 minutes per step because training kicked off with the engine still in inference-state (LM + DiT both loaded) — Lightning Fabric setup wrestled saturated memory. Fix: `_ensure_training_ready()` in `app.py` calls `POST /v1/init {init_llm: false}` before any preprocess/train so the LM is dropped → DiT-decoder-only on GPU. Symmetric `_ensure_labeling_ready()` (init_llm=true) before autolabel so the LM is loaded when needed. Both endpoints `/api/lora/dataset/preprocess` and `/api/lora/train` and `/api/lora/dataset/autolabel` enforce the right state now.
- **🛠 Engine bug + upstream PR + LOCAL PATCH STILL ACTIVE.** The auto-label route in the engine (introduced by refactor commit `d19c2f3`, 2026-03-05) calls `builder.label_all_samples(chunk_size=..., batch_size=..., sample_labeled_callback=...)` but the method (`acestep/training/dataset_builder_modules/label_all.py`, last touched 2026-02-06) accepts none of those. Three-month-old regression nobody else hit because Gradio bypasses the API. Crucible's HTTP-driven flow exposed it. **PR submitted: https://github.com/ace-step/ACE-Step-1.5/pull/1226** (proper fix — wires `sample_labeled_callback` through to the method, drops the truly-unused `chunk_size`/`batch_size` from both sync + async routes). **Until the PR merges:** user manually patched `<engine>/acestep/training/dataset_builder_modules/label_all.py` on the box — added `**_ignored,` to the `label_all_samples` signature before the closing `)`. **Reverted by any engine `git pull` (re-running ACESTEP-ENGINE_AUTO_INSTALL.bat), so re-apply on every engine update.** Same caveat as the DCW patch. Cataloged in [[engine-patches]] memory + `METAL_LORA_PLAN §7a`.
- **LM caption format — verified by one-track probe.** Box LM emits **prose**, ~750 chars, multi-sentence, audio-grounded ("An explosive modern metal track driven by high-gain, heavily distorted dual guitars… around the two-and-a-half-minute mark… technical and melodic guitar solo filled with fast runs and expressive bends. The track culminates in a final, powerful chorus and an abrupt, impactful ending."). Captures things our metadata sources can't (production character, structural narrative, instrumentation specifics, feel). The LM is weak on categorical genre labels (says "modern metal" not "power metal") — exactly why merging with our enrichment is the right architecture.
- **Caption merge — prose-aware two-section + tokenizer cap research.** `caption_fetch.merge_seed_with_lm(seed, lm)` detects prose vs tag-list, picks the right strategy: prose → `"<seed tags>. <LM prose>"`; tag-list → dedupe via `_merge()`. **The actual caption cap is 256 *tokens* at the text-encoder tokenizer** (verified in `acestep/training/dataset_builder_modules/preprocess_text.py:18-21` — `padding="max_length", max_length=256`), not the 512 *chars* the RunComfy guide suggests. 256 tokens ≈ 1000–1500 chars of English; overflow is silently tokenizer-truncated, no error. We default `max_chars=2000` (effectively no-op for typical LM outputs) and only sentence-boundary-truncate as belt-and-braces. **Mac state cache `_AUTOLABEL_SEED[dataset]` captures seed captions BEFORE LM overwrites them so we can merge after, via `POST /api/lora/dataset/autolabel_merge`.** UI's Step-2 button "Auto-caption + merge" chains autolabel → poll → merge → reload samples.

**Where to start next session (the user hasn't decided yet — ask):**
1. **First REAL metal LoRA training** — likely a focused subgenre (METAL_LORA_PLAN §10.2 pyramid recommends subgenre as the sweet spot; ~25 tracks, LoKr, ~150 epochs ≈ ~1.5–2 hr on the 3090). User has a power-metal compilation handy. Smoke test proved the pipeline works end-to-end; no known blockers.
2. **Track upstream PR #1226** — when it merges, the local `**_ignored` patch becomes redundant but harmless.
3. **Drive a real Auto-caption + merge cycle in the UI** — the merge logic has been unit-verified on the real LM output (Sabaton track), but the full UI button chain hasn't been exercised live on a multi-track dataset yet. Worth a one-pass smoke when convenient.
4. **Smoke artifacts on the box** still live at `E:\AI\MusicGen\AceStep\ACE-Step-1.5\lora_data\smoke_power\` (data cleared, but `train_a/final` + `train_b/final` adapters + `tensors/` remain). User can delete the folder when convenient; the new `/dataset/delete` endpoint will work as one-call cleanup once the helper restarts to pick it up.
5. **Per-song image generator** (still open from the previous session) — cards are artwork-ready, just need `params.artwork_url` set by a future image-gen step.
6. **Helper restart for `/dataset/delete`** (low-priority).

### SESSION 2026-05-27 — workflow/UX + reference-analysis (all shipped, on `main`)
A big feature/UX pass on top of the settled engine outcome. Everything below is built, verified, committed (user-only attribution) and pushed.

- **Projects + per-page state persistence.** New `web/src/drafts.tsx` = a `DraftProvider` context + `useDrafts(ns).use(key,init)` (drop-in `useState`) lifted to the App root, so switching tabs no longer wipes a form's inputs. All 15 forms + `useTuning` + Vocal Builder converted (Files/fetched-lists stay ephemeral). On top: a **Projects** system — SQLite `projects` table + REST (`/api/projects` list/create/get(PUT save)/PATCH rename/DELETE) + a header **ProjectBar** (New/Open/Save/Save-As/Rename/Delete). A project serializes `{drafts, song, mode}`.
- **COVER FIX — the big one (corrects the old "cover is a ceiling" verdict).** Cover was weak because we never set **`cover_noise_strength`** (the engine's *melody-retention* knob; default 0 = pure style transfer). Now `/api/cover` sets it (default 0.2) + defaults the model to **xl-sft**, and Restyle exposes a "Melody retention" slider. **By-ear validated** (Baby One More Time → metal): cns 0.2 keeps the tune; cns 0 was garbage. See memory [[acestep-engine-outcome]].
- **Mastering — two new reference-free modes** (RESEARCH §16). `master.py` `master_auto()` = pedalboard tone-curve + glue comp + iterative loudness-drive to a target **LUFS** at −1 dBFS (pyloudnorm). Master tool now has **Auto** (no reference; tone presets + LUFS targets) · **Gold standard** (curated refs in `library/master_refs/`, Matchering) · **Reference** (original). New dep `pyloudnorm`.
- **Library redesign.** Wider (520px) panel, **type tabs** (replacing stacked accordions), responsive **card grid**, search + sort, **version-grouping** (same-titled songs collapse w/ v1/v2 chips), **artwork-ready cards** (16:9 header shows `params.artwork_url` when a future image-gen sets it). Cards have a ↻ **"Open in Song Builder"** for builder songs.
- **Song naming + re-import round-trip.** Song Builder "🎵 Name song" (LLM `names` task → pick a suggestion). Builder renders store the full recipe (`song_meta`: blocks/key/bpm/tags/instrumental/drive + `title`, `from_builder`) and save as mode `song`; the library card's ↻ repopulates the builder for a new version.
- **Reference-to-Song analysis (NEW — RESEARCH §17).** Song Builder **"📥 Analyze a reference track"** → detect BPM/key/structure(+optional lyrics) → fill the arrangement → generate a *similar* song via text2music (NOT cover). **P1** = Mac/librosa (always). **P2** = box GPU service (`backend/analyze_server.py`, `ANALYZE-API_AUTO_INSTALL.bat`, port 5075, `analyze_host` in app_config) running **allin1fix** (functional section labels) + **CLAP** zero-shot tags (curated metal vocab ∪ our genre registry) + librosa key. **WORKING end-to-end on the box** (engine=allin1, proper Intro/Verse/Chorus/Bridge labels + tags). The Mac always computes structure and the box's allin1 overrides it when present; box failure → graceful tags-only/Mac-structure. VRAM-coordinated (frees ComfyUI/RVC before; self-unloads after).
  - **Box install reality (hard-won; the installer encodes all of it):** needs **Python 3.10–3.12** (prebuilt NATTEN wheels are cp310/311/312 only) + **git** + **MS C++ Build Tools** (madmom compiles). Stack = **torch 2.6.0/cu126** + prebuilt **natten 0.17.5** wheel from `lldacing/NATTEN-windows` (no CUDA build) + **madmom from git** (`all-in-one-fix` imports it despite not declaring it) + **all-in-one-fix** + **`numpy<2`** (laion-clap needs it; re-pinned last). Fully self-contained: venv + pip/HF/torch caches + **TMP/TEMP** all under the install dir.
- **Engine model-picker fixes.** `/release_task` does NOT auto-load a model (silently falls back to the loaded one) — added `_acestep_ensure_model()` to generate/cover/repaint/lego so the picked model actually swaps. AND Song mode's tuning is now engine-aware (`useTuning(... , !!cfg.acestep)`) — it was sending the inert ComfyUI variant, so base/sft never changed. Header now has an **ACE chip** (polls `/api/acestep/info`, shows the loaded model) so swaps are visible.
- **Generate progress bar** folds the LM-thinking + DiT stages into one monotonic ramp (was resetting mid-run, looking like a 2nd take — it's one song).

### ENGINE MIGRATION — FINAL OUTCOME (2026-05-26): "Generate-only + LoRA-future"
After full migration + investigation, the verdict: **the official ACE-Step engine is worth it ONLY for text2music (Generate) + as the future metal-LoRA platform. All audio-INPUT tasks stay on ComfyUI.**
- **Generate (text2music) → ENGINE, LIVE.** xl-sft/base, the DCW-off fix + the documented recipe (ADG + guidance 8 + 64 steps + thinking/CoT). By ear ≈ ComfyUI, slight prompt-adherence edge. `acestep_dcw_ok:true`. Engine-aware Expert UI. ComfyUI `build_t2m` is the fallback (gated by `acestep_dcw_ok`).
- **Cover/Restyle → EXPERIMENTAL.** Engine cover *produces music* but **weak transforms** — doesn't track/resemble the source (user confirmed on a clean metal→metal run + extensive pop→metal tests). Kept with ComfyUI fallback; UI marked experimental.
- **Repaint → REVERTED to ComfyUI** (`acestep_repaint:false`). Engine repaint silence-seeds the region + skips the LM → quiet/garbled; aggressive mode re-renders the whole song. ComfyUI edit guider (LM in-graph) is the good path. Kept the new waveform region selector + beat-snap (engine-independent).
- **Lego (Add-a-Layer) → REVERTED to ComfyUI** (`acestep_lego:false`; engine code kept behind the flag). Engine lego garbles regions and returns an *isolated* layer (not a mix); needs `chunk_mask_mode=explicit` + `global_caption` (researched + wired) but still poor. ComfyUI `build_lego` (verified good earlier) is the path. Region/beat-snap selector added to the Layer tab too.
- **Extract → NOT migrated** (same audio-input family; left on ComfyUI/Demucs/RoFormer).
- **INVESTIGATION (why audio-input tasks fail) — settled with the box console log:** it is **NOT infrastructure.** Verified clean: (a) **Mac→Windows transfer** (engine receives + encodes the source correctly: right 30s/750-frame length, healthy output), (b) **CPU-offload** (VAE is loaded to *cuda* for encode/decode; offload only idles models between uses; the scary `RSS:0 MB` is a cosmetic logging bug; real VRAM 9.99GB alloc / 18GB peak), (c) **VAE encode/decode** (clean run, no errors/silencing/fallback). The failures are **model/task-level limitations**, not fixable by config. (Don't re-chase offload/transfer — disproven.)
- **LoRA reality (verified):** existing ACE-Step LoRAs are **PEFT format**, target the **2B base/turbo** (NOT our XL 4B), and the only metal-adjacent one is raspy-vocal + generic-instrumental (not true metal). **Native ComfyUI cannot load PEFT ACE-Step LoRAs** (key-format mismatch; ComfyUI #9753 open, #12638 key errors) — the **engine loads them natively**. So the engine's unique long-term value = **training a custom Crucible metal LoRA** (~17GB VRAM per the tutorial → fits the 3090; way down the line per user). No off-the-shelf LoRA helps our XL setup today.
- **Also shipped this session:** engine **progress bar** (poll parses the engine log line + time fallback → bar moves); **workspace WavePlayer upgraded** (taller, timeline, playhead, time readout, hover — shared across all tabs); **Export-MP3** on every library item; library cards render `params.note` (A/B labels visible).

### (historical migration notes below)
- **IN PROGRESS — migrate ComfyUI ACE touchpoints → the OFFICIAL ACE-Step engine (incremental, ComfyUI kept as fallback per step).** The official `acestep` API engine now runs on the box at **`192.168.1.201:8001`** (`acestep_host` in app_config). **Cover already migrated + works end-to-end** (`/api/cover` engine path: `backend/acestep_py.py` client → `/release_task` multipart submit → background poll `_acestep_cover_poll` → `/v1/audio` download → library; ComfyUI `build_cover` is the fallback when `acestep_host` empty). **Whisper lyric transcription** added for cover (`/api/transcribe`: Demucs **or** RoFormer vocal isolate → faster-whisper on Mac → fills Lyrics; copyright block on fetching commercial lyrics is why we transcribe the user's own audio).
  - **Generate (text2music) — BACKEND BUILT 2026-05-26, A/B test-gen PENDING.** `/api/generate` now has an official-engine branch (`backend/app.py`) mirroring cover: builds `text2music` fields → `acestep_py.submit` (no src_audio) → background `_acestep_poll(pid, task_id, "generate")` → library; ComfyUI `build_t2m` is the fallback when `acestep_host` empty. Engine **initialized + verified loaded: `acestep-v15-xl-sft` (default) + `acestep-5Hz-lm-4B`, `llm_initialized:true`** (xl-sft WAS already on the box). **Verified engine facts (this session):** (1) **xl-sft supports ALL tasks** — `text2music, repaint, cover, cover-nofsq, extract, lego, complete` (turbo only the first four) → **no turbo-forcing needed** for repaint/lego/extract on the engine, unlike ComfyUI. (2) **NO DiT-level negative prompt** exists — only `lm_negative_prompt` (5Hz-LM CFG); ComfyUI's `negative_tags` has no equivalent → **dropped** on the engine path. (3) **`use_random_seed` must be set `false`** alongside `seed` or the engine rolls its own (fixed in both generate + cover). (4) **status `2` = failed** (positive) per API.md — fixed `acestep_py.wait` (was hanging to timeout on failures). (5) `batch_size` default is **2** → set to **1** so each of the form's 1–4 `count` requests = one take/card. Param map: `cfg→guidance_scale` (base only), `steps→inference_steps` (base 32–64), keep `shift`/`infer_method`/`cfg_interval`; `instrumental` is a native bool; `thinking=true` (4B LM codes = ComfyUI `generate_audio_codes` parity) + `use_cot_caption/language=true` (user chose LM auto-expand). **GPU-TESTED 2026-05-26 — ROOT-CAUSE FOUND (DCW bug):** text2music on **xl-sft AND xl-base both produce garbled/incoherent audio** (user-confirmed by ear); **turbo produces coherent music** (user-confirmed, lower quality). Ruled out: LM (thinking on/off identical garble), steps (32 & 20 both garble), plumbing (raw `/query_result` returns the correct single 48kHz wav; engine reports success, normal timing LM~26s/DiT~6s), model-specific (base==sft). **Cause = DCW (Differential Correction in Wavelet domain), a sampler-side per-step SNR correction shipped ENABLED-BY-DEFAULT (`dcw_mode="double"`, `acestep/inference.py:~148 dcw_enabled=True`; PR #1120, ~2026-04-26).** Its error accumulates over the denoising trajectory → it destabilizes the **full from-noise XL text2music run**, but cover/repaint (source-anchored, short effective trajectory) and turbo (~8 steps) escape it. Matches upstream issues **#1206** (sft+DCW garbage on same RTX3090/CUDA12.8; closed) and **#1220/#1191/#1063**. **DCW CANNOT be disabled over the HTTP API** — the `/release_task` request builder (latest `main`, `acestep/api/http/release_task_request_builder.py`) forwards NO `dcw_*` field; no env var / launcher arg / gradio-api route exposes it either. **FIX APPLIED + CONFIRMED 2026-05-26:** box-side source patch — `dcw_enabled` default set to `False` in the engine's `acestep/inference.py` (~line 148), engine restarted, xl-sft+4B LM re-initialized. **xl-sft text2music now produces coherent music (user-confirmed by ear).** Metrics vs the garbled take: pulse-clarity 0.57→**0.855**, chroma-var 0.036→**0.060**, centroid 2301→**3673** (spectrogram `test_output/gen_dcw_fix.png`). **Generate is LIVE on the engine (xl-sft) end-to-end.** ⚠️ The patch edits third-party source → an engine update reverts it; if XL text2music garbles again, re-apply the patch (and the maintainers still don't expose `dcw_enabled` over the HTTP API — worth upstreaming). **SAFEGUARD:** `acestep_dcw_ok` config flag (default **false** in app_config.example) gates engine XL text2music — when false, Generate falls back to ComfyUI so it never silently emits DCW garbage; turbo + cover/repaint are never gated. **Currently `acestep_dcw_ok: true`** in the live app_config (box is patched).
  - **A/B vs tuned ComfyUI — SIGNED OFF 2026-05-26 (user by ear).** Engine driven the WRONG way first (CoT off + lowered guidance, no ADG) sounded muddy/fake/pop-like and slightly worse than ComfyUI. Driven the **documented best-quality way** — **`use_adg=true`, `guidance 8`, `inference_steps 64`, `shift 3`, thinking+CoT ON** — the engine **matches/beats ComfyUI** by ear: **xl-BASE = the quality pick** (clearer; docs agree base>sft for quality), xl-sft also good (denser/more harmonic). Mud metric (150–500Hz ratio): base-as-intended 0.125 < ComfyUI 0.14. Key lesson: **drive the reference engine as documented (LM CoT pipeline + ADG + high guidance + 64 steps), don't lower guidance to chase mud — ADG is what keeps high guidance clean.** Remaining ceiling-raiser for true metal character = a **metal LoRA** (RESEARCH §15; engine supports LoRAs).
  - **Engine-aware Generate Expert UI wired (2026-05-26):** `useTuning(engineMode)` ([web/src/forms.tsx]) shows engine levers when `cfg.acestep` — **Model (base/sft/turbo), Guidance, Steps, Inference (ode/sde), ADG toggle, CFG-interval start/end**, BPM/Key/Seed/Shift/Duration + thinking/CoT — and HIDES the inert ComfyUI controls (variant Model / Sampler / Scheduler / APG / Negative-tags) on the engine path. UI defaults to xl-base + ADG-on. Backend `/api/generate` engine branch passes all of these through (`use_adg` added). ComfyUI-path forms (Restyle/Repaint) keep the ComfyUI controls unchanged.
  - **Generate engine defaults = the A/B recipe** (when the field is left blank): guidance **8**, **64** steps (turbo→8), **ADG on** for base/sft. UI placeholders match.
- **Restyle — MIGRATED 2026-05-26 (functional; user tuning by ear).** Reimagine + Cover both ride the engine **`cover`** task: Cover uses the strength slider; **Reimagine maps `amount`→`audio_cover_strength = 1 − amount`** (more change = lower strength). `/api/cover` engine params aligned with generate (cfg→guidance, model/use_adg/infer_method/cfg_interval from the engine tuning UI) + `result_mode` so reimagine saves as `restyle`, cover as `cover`. ComfyUI `build_restyle`/`build_cover` kept as fallback when no engine. **Engine-aware + source-conditioned UI:** RestyleForm uses `useTuning(engineMode, sourceConditioned)` → shows engine levers but **hides Duration/BPM/Key** (the cover task derives length/tempo/key from the SOURCE audio; they aren't sent). **Verified end-to-end** on the box (pop "Baby One More Time" → metal, both modes ran, 211s source-locked output, correct library labels). **Tuning notes for testing:** cover defaults to **batch_size 2** (2 takes/request — engine default; not overridden); `cover_strength` **0.7–0.9 = stays close to original**, **0.3–0.5 = big genre change** (so low strength won't "sound like" the source); instrumental=true drops the vocal melody (use Transcribe to keep the sung line). **NEXT: Repaint** (`build_edit` turbo-forced → native `repaint` on xl-base/sft — engine supports repaint on all models, no turbo-forcing needed).
  - **Restyle/Cover QUALITY = WEAK (user by ear, 2026-05-26) — kept but marked EXPERIMENTAL.** Migration is mechanically correct (runs, strength works, labels right) but the engine's cover/remix output is poor: weird/incoherent screamed vocals, off timing, and it doesn't follow the source melody (pop→metal "you'd never know the original"). Ruled out: strength direction (0.4–0.5 for genre change, confirmed), forced lyrics (removed — still bad), **song length (tested at a 30s slice — still incoherent → it's the TASK, not length)**, engine-vs-ComfyUI (ComfyUI guider just different, not better). Matches maintainers calling remix "one of the most underdeveloped parts of the model" (#690) + uneven-syllable lyric drop (#391) + high-strength noise (#624). **No setting/our-code fix exists — it's an engine ceiling.** Revisit later via a metal LoRA (style) or the engine's in-progress remix work (#1143 raw-remix, #1156 flow-edit). UI now shows an "⚠ Experimental" note on Restyle. **Generate (text2music) remains the solid win.**
    - **CORRECTION 2026-07-26 — the "in-progress" flow-edit was ALREADY SHIPPED when this was written, and has been on our box the whole time.** Flow-Edit (and Retake) shipped upstream in **v0.1.8, dated 2026-05-18**, i.e. 8 days BEFORE this note; our box checkout IS that commit (`dce6214`). Verified live in the box source via `:5080 /fs/read`: `acestep/inference.py` L177-187 defines `flow_edit_morph` + `flow_edit_source_caption`/`_source_lyrics`/`_n_min`/`_n_max`/`_n_avg`, plumbed into the DiT call at L856-863; `retake_seed`/`retake_variance` at L170-171 with resolved seeds returned at L927-951. So this escape route was available from day one and we never rechecked (the note was written from the issue tracker, not from the installed source — a [[verify-feature-engaged-not-just-ran]] miss).
      **Why we still cannot use it:** the HTTP request model `acestep/api/http/release_task_models.py :: GenerateMusicRequest` has NO dcw / retake / flow_edit fields and NO `extra="allow"`, so anything we post is silently dropped. The blocker is the API surface, not the feature. Exposing it = a 2-file engine patch (add the fields + read them with `getattr` in `job_generation_setup.py`, which already does that for `repaint_mode`/`repaint_strength`) — logged as candidate patch 8 in memory `project_engine-patches`. GATE FIRST: try Flow-Edit in the engine's own Gradio UI before writing the patch; if its output is as weak as the cover task, don't build it.
      NOTE: `/openapi.json` cannot answer this — `/release_task` has an untyped request body, so its schema lists no fields at all. Read the pydantic model.
  - **Repaint — REVERTED TO COMFYUI 2026-05-26 (engine repaint is structurally weak; user by ear).** The official engine's repaint regenerates the region but it comes out **quiet/weak/incoherent**, regardless of model (base/sft/turbo all tried), `repaint_strength` (the codebase has CONTRADICTORY comments: `inference.py:167` says 0=aggressive but the authoritative `_resolve_repaint_config` says **higher=more aggressive/less preservation**), or `repaint_mode` (**aggressive mode broke the WHOLE song** — `do_wav_splice = mode != "aggressive"`, so aggressive skips the splice that preserves everything outside the window → full re-render). **Root cause (read from source):** repaint **silence-seeds the region latent** (`conditioning_masks.py`, issue #810), **skips the LM** so there's no audio-code plan (issue #963; docs say cover/repaint/extract ignore the LM), and the **splice does no loudness match** (`repaint_waveform_splice.py`) → quiet/weak regions. **Advanced path TESTED + FAILED:** feeding `audio_code_string` (from `extract_codes_only`, which encodes existing src audio) into repaint — full-song codes misaligned (silenced 40–60s), region-aligned codes collapsed output to 10s NaN. The code threads codes in but the repaint pipeline can't consume external codes coherently. **ComfyUI's edit guider runs LM codes in-graph (user-confirmed good earlier) → that's the repaint path.** Gated by `acestep_repaint` config (default **false** = ComfyUI; engine code kept behind the flag for a future upstream fix). **KEPT: the waveform region selector + beat-snap UI** works on the ComfyUI path too (engine-independent). EditForm keys its UI off `cfg.acestep_repaint`. **Repaint = "regenerate a section in-style"; adding a distinct part (e.g. a solo) is Add-a-Layer's job** ([[official-feature-guidance]]).
  - **(superseded) Repaint engine-migration build (2026-05-26):** `/api/repaint` engine branch (task_type=repaint, src_audio multipart, repainting_start/end, DiT params from the engine tuning UI) → `_acestep_poll`; **drops the ComfyUI turbo-forcing hack** (engine does repaint on base/sft). ComfyUI `build_edit` kept as fallback (+ `force_comfy`). `_source_bytes` helper for engine src_audio. EditForm uses engine-aware + source-conditioned tuning (hides Duration/BPM/Key + the ComfyUI edit-cfg on the engine path). **NEW waveform region selector** (`web/src/RegionSelector.tsx`, wavesurfer v7 + Regions plugin): drag-to-mark / drag-to-move / resize-edges, live start/end/length readout, **auto-detected beat markers + snap-to-beat (toggle, on by default)** via `GET /api/beats/{id}` + `POST /api/beats` (librosa beat_track, Mac CPU). Reusable for Layer later. **TODO: GPU-verify a repaint by ear (region changed, rest preserved, seam).**
  - **Export MP3 (2026-05-26):** `GET /api/export/{id}?fmt=mp3` (ffmpeg transcode for WAV, passthrough for MP3, friendly filename from note/tags) + a ⬇ button on every Library card.
  - **Library label fix (2026-05-26):** `libDesc()` (web/src/App.tsx) renders `params.note` when present → A/B/comparison takes are identifiable in the Library instead of all showing the same prompt. `_acestep_poll` set a `note` to tag takes. `/api/cover` gained `batch_size` passthrough + a `force_comfy` escape hatch (route to the ComfyUI guider for A/Bs even when the engine is set). **Always serialize cross-engine GPU runs (ACE + ComfyUI never simultaneously).**
  - **STILL ON COMFYUI → to migrate next, one at a time, GPU-verify + A/B each, keep ComfyUI fallback until signed off, retire ComfyUI last:** Generate (DONE pending A/B, above) · **Restyle** (Reimagine + Cover both collapse to the engine `cover` task, `audio_cover_strength`=the knob; reimagine≈low strength) · **Repaint** (`build_edit`, currently turbo-forced → native `repaint` on xl-base) · **Add-a-Layer** (`build_lego`→`lego`) · **Layer-isolate "ACE extract"** (`build_extract`→`extract`).
  - **Param mapping (ComfyUI→engine GenerationParams, see RESEARCH §13):** `cfg`→`guidance_scale`, `steps`→`inference_steps`, sampler→`infer_method` (ode/sde), keep `shift`; **APG → `cfg_interval_start/end`** (engine's native high-cfg control); **negative-prompt support on the engine is UNVERIFIED** — check, map or drop. Engine `/release_task` fields proven for cover: `task_type, prompt, lyrics, audio_cover_strength, guidance_scale, inference_steps, shift, cfg_interval_start/end, infer_method, audio_format, model, seed` (+ multipart `src_audio`/`ctx_audio`). Result is a JSON-array of takes → take first (or save all, see commit b93a8dc).
  - **Engine ops/quirks (hard-won):** loads models via `POST /v1/init {model, init_llm, lm_model_path}`; **xl-base + 4B LM currently loaded & default**; turbo also present. **NO unload/free API** (tested `/v1/init {model:null}` and `/v1/reinitialize` — both no-op; known issue #198 retains VRAM) → so the engine runs **CPU-offloaded** (`ACESTEP_OFFLOAD_TO_CPU=true` in `run_acestep_api.bat`) for low idle VRAM. To persist xl-base as default across restarts: `ACESTEP_CONFIG_PATH=acestep-v15-xl-base` in the launcher (inferred). Engine **auto-downloads models on first use to its repo `./checkpoints`** and does NOT recognize a hand-placed `huggingface-cli` copy for auto-download (but `/v1/init` DID load the installer's xl-base from disk). Installer = `ACESTEP-ENGINE_AUTO_INSTALL.bat`.
  - **Shared-3090 VRAM coordination (done):** `app.free_gpu(keep)` + `submit_comfy()` — each GPU op frees the others first (ComfyUI `/free`, RVC `/free` [added to `rvc_server.py`]; SoulX/RoFormer self-unload; ACE offloaded). 
  - **LAN serve (done):** `server_host` config = `0.0.0.0` → app reachable at `http://<mac-IP>:8000` (currently 192.168.1.87). API has no auth — trusted LAN only.
  - **After migration:** then Song Artwork via **Chroma** (RESEARCH §14) and metal LoRAs (RESEARCH §15; engine has `/v1/lora/*` + `/v1/training/*`).

- **PLANNED (queued 2026-05-25):** _Migrate to the official ACE-Step engine (RESEARCH §13), then Song Artwork (RESEARCH §14)._ Order: **(1)** stand up the official `acestep` API engine on the box (installer `ACESTEP-ENGINE_AUTO_INSTALL.bat` written; user installing — XL + 4B LM downloading), build Mac `acestep_py.py` client + `acestep_host` config, port **cover first** (test country→metal at `audio_cover_strength`~0.4), then text2music A/B vs ComfyUI, then repaint/lego/extract, then retire ComfyUI for ACE. **(2)** _Song artwork_ — **Chroma only** (uncensored FLUX-class, HF, ComfyUI-native, Apache; RESEARCH §14): `llm.py` artwork-prompt from tags/lyrics/title → `comfy.build_image()` (Chroma t2i) → `POST /api/artwork` → library mode `image`, shown with the track. Build AFTER cover works; needs the Chroma checkpoint installed on the box first.
- **DONE (2026-05-25):** _Clean-bed layer workflow + combine-other isolation (clean added-instrument stems)._ User QA found the isolated layer stem was bad. Diagnosis (RESEARCH §12c): (a) separators can't split two parts of the SAME instrument (rhythm vs lead guitar), and (b) the lego re-render misclassifies the distorted guitar into the `other` stem (measured: lego guitar lands in `other`≈0.13, `guitar`≈0.0). Fix: **`clean_bed`** param on `/api/layer` strips the layer's instrument from the backing FIRST (Demucs/RoFormer) → lego adds the part onto a bed lacking it → the added part is the only instance; **`combine_other`** param on `/api/layer/isolate` sums `target`+`other` to recover it whichever bucket it fell into. Verified: clean-bed guitar lead isolates at RMS 0.138 (vs 0.000 grabbing `guitar` alone). UI: Layer tab "Strip {track} from the backing first (clean bed)" toggle (+ bed-engine demucs/roformer) and the isolate auto-passes `combine_other`. **User by-ear confirm pending** (clean-bed lead in Library → Layer stems).
- **DONE — GPU-VERIFIED (2026-05-25):** _BS-Roformer SW as a selectable SOTA separator (any stem)._ Box service installed + running on `192.168.1.201:5070`; `roformer_host` set. Verified on the guitar layer mix vs the earlier stems: **BS-Roformer is by far the cleanest** — region RMS 0.019 (vs Demucs 0.085, ACE-extract 0.197) with far less broadband backing haze (spectrogram `test_output/stem_3way.png`), all correctly gated to 10–25s, and it ran on the 3090 in **~9 s** (no Mac crash). User by-ear confirm pending. Below = the build details.
- **BUILT (2026-05-25):** _BS-Roformer SW as a selectable SOTA separator (any stem)._ Best 2026 separation model (beats Demucs ~2 dB SDR); one model → 6 stems (vocals/bass/drums/guitar/piano/other). **⚠️ Runs on the 3090 ONLY — running it on the Mac's MPS HARD-CRASHED/rebooted the Mac (kernel panic from unified-memory pressure); Demucs stays the Mac engine.** Deployment = box-side service (RVC/SoulX pattern), not a ComfyUI node (none cleanly loads the SW config). New: `ROFORMER-API_AUTO_INSTALL.bat`, `backend/roformer_server.py` (box `/health`+`/separate`, runs the CLI per-request so VRAM frees after each), `backend/roformer_py.py` (Mac client). Engine-agnostic `_separate()` dispatches demucs(Mac)/roformer(box) and is wired into `/api/layer/isolate` (method `roformer`), `/api/stems/separate` (`engine`), `/api/backing/strip-guitar` (`engine`); Mac calls `C.free()` first to release the 3090. `/api/config` reports `roformer` (set `roformer_host` in `app_config.json`); UI shows the engine option only when present. **TO ACTIVATE:** run `ROFORMER-API_AUTO_INSTALL.bat` on the box → start `run_roformer_api.bat` → set `roformer_host: "<box-ip>:5070"`. Then GPU-verify guitar-stem cleanliness vs Demucs/ACE-extract + any-stem quality on a real mix. _(Lego stem-isolation is fundamentally leaky because lego re-renders the whole track — RESEARCH §11a/§12b; ACE can't cleanly generate an isolated single instrument either, it errored twice. A SOTA separator is the pragmatic fix; clean stems otherwise come from real, un-re-rendered mixes.)_
- **Disk hygiene (2026-05-25):** Mac system drive is small (~6 GB free). All project caches pinned to the SSD via `run.sh` (`TORCH_HOME`/`HF_HOME`/`PIP_CACHE_DIR`/`UV_CACHE_DIR` → `.caches/`); the unrelated 3.9 GB `~/.cache/huggingface` relocated to `/Volumes/SSD1/caches/huggingface` (symlink). `.gitignore`: `.venv-roformer/`, `.caches/`, `models/`. **Never run heavy models (RoFormer etc.) on the Mac — use the 3090.**
- **DONE — both routes GPU-verified (2026-05-25):** _Add-a-Layer — isolate the added part as a stem._ Layer form has an "Isolate the added part as a stem" toggle with two methods: **Demucs** (Mac-side `htdemucs_6s`, synchronous, 12 lego track-names mapped onto 6 demucs stems) and **native ACE extract** (`comfy.build_extract` → `ACEStep15NativeExtractGuider`, GPU, base/sft). `postfx.gate_region` trims the stem to the layer's time window (optional toggle). `POST /api/layer/isolate` (method demucs|extract); outputs toggle = keep mix+stem or stem-only (stem-only deletes the mix row). Subtraction was rejected (ACE re-decode phase-decorrelates even preserved regions, per the brief). Library mode `layerstem` ("Layer stems" section). Why a separator and not the guider: the lego node bakes the layer into a full mix (no native stem out). **Both verified on the same guitar layer mix, both correctly gated (silent outside 10–25s): Demucs = darker midband (centroid ~1.6 kHz, RMS 0.085); native extract = brighter/louder, lead-harmonic-focused (centroid ~3.4 kHz, RMS 0.20); mel-cosine 0.70 between them → genuinely different characters, pick by ear per use.** User by-ear A/B welcome.
- **DONE — GPU-VERIFIED (2026-05-25):** _Add-a-Layer (ACE-Step `lego` task)._ ACE Studio's flagship "Add a Layer" — generate a new named track (vocals/drums/bass/guitar/keyboard/strings/synth/fx/brass/woodwinds/backing_vocals) into a time region of an existing backing. `comfy.build_lego` drives `ACEStep15NativeLegoGuider` via `SamplerCustomAdvanced` (`ACEStep15TaskTextEncode(task_type="lego", track_name=X)`); **base/SFT only — defaults `xl_sft`, guider cfg 6** (lego/extract/complete can't use turbo, RESEARCH §10j). Optional **timbre reference** clip → `reference_latent` (item D folded in). `POST /api/layer` (mirrors `/api/repaint`, accepts a 2nd `timbre` upload); new **"Add Layer"** tab under Create ([forms.tsx](web/src/forms.tsx) `LayerForm`), library mode `layer`. **GPU run (xl_sft cfg6, guitar lead 10–25s over a generated rhythm backing) confirmed it works.** **Key unknown RESOLVED — unlike ACE Studio's isolated-stem layer (RESEARCH §11a), our lego guider returns a COMPLETE MIX with the new part baked into the region: mel-cosine 0.94–0.96 + RMS ratio ≈1.0 OUTSIDE the region (backing preserved) vs cosine 0.756 + RMS ratio 1.37 INSIDE (new guitar added on top). So NO separate `mix.py` step is needed — save-as-is is correct.** (User by-ear confirmation of musical quality still welcome.)
- **DONE (2026-05-25):** _ACE Studio comparative research (RESEARCH §11a)._ Verified the four flagged features from ACE Studio's own docs. Corrected two §11 mappings: **Music Enhancer = style-transfer/reimagine with a consistency slider (≈ our Restyle, NOT Matchering)**; **Inspire Me = text-to-music From-Idea/From-Lyrics (≈ our Generate + Song + lyric writer, NOT the Assistant dock)**. Their genuine lead = the **editable per-note piano-roll vocal editor** (6 pitch tools incl. vibrato w/ amplitude/phase/frequency + 6 expressive params Formant/Energy/Tension/Falsetto/Air/Breath); our Vocal Builder roll is read-only → biggest future opportunity. Cheap adoptables: rename Restyle's `restyle_amount` → "Consistency" + auto-analyze/pre-fill source tags.
- **DONE (2026-05-25):** _Repaint (works); Extend removed._ Vendored the MIT, self-contained `ACEStep15NativeEditGuider` (+ siblings) into `comfy_custom_nodes/ComfyUI-ACEStep-Repaint/` (installed on the box). `comfy.build_edit` drives it via `SamplerCustomAdvanced` for **Repaint** (regenerate a time range, rest preserved) — `POST /api/repaint`, **Repaint** tab under Create. ✅ GPU-verified + user-confirmed good. **Forces the turbo model** (the edit guider's silence-latent is turbo-trained; base/sft garble — `xl_turbo` now installed). **Extend was removed** — append-extend is *not* a native ACE task (official: text2music/remix/repaint/lego/extract/complete), so the community latent-padding hack had unfixable seam/beat/length artifacts (RESEARCH §10i/§10j). For longer songs: generate at a longer `duration` or use the Song Constructor. (Route 1 core-native masked-latent repaint was spiked and failed — §10i.)
- **DONE (2026-05-25):** _Adaptive Projected Guidance (APG) wired._ `comfy.py` `_apg_model` inserts the native ComfyUI `APG` node (`model → APG → KSampler`) as an **optional** stage (`apg` + `apg_eta`/`apg_norm`/`apg_momentum`; Expert UI toggle, defaults eta 1.05 / norm 1.3 / momentum 0). APG is the recommended fix for ACE's high-CFG oversaturation/mud — lets us keep strong cfg cleanly rather than only dropping it. Graph validated + GPU-verified; the cfg4-no-APG vs cfg6+APG A/B showed APG measurably fixes the high-cfg mud (more articulate, less bass-bloat, more dynamic). **Still optional/off by default — making APG on-by-default for `xl_sft` is a pending decision (user liked it).** See RESEARCH §10c "Community ACE-in-ComfyUI usage" (APG, CFGZeroStar/RescaleCFG/etc., JK-AceStep-Nodes, guidance interval — ongoing community-usage research the user asked for).
- **cfg sweep (2026-05-25):** SFT cfg4 vs cfg6 (seed 424242) — measured cfg4 brighter/more articulate/more dynamic, cfg6 bassier/darker. cfg5 take didn't land (stuck pending). User ear-verdict pending; leaning lower-cfg, but APG may make higher cfg viable.
- **DONE (2026-05-24):** _In-form AI lyric generation._ `backend/lyrics.py` + `POST /api/lyrics/song` writes **structure-aware** lyrics for a Song arrangement (distinct verses that advance the story, ONE repeated chorus hook, optional pre-chorus/bridge; intro/solo/breakdown/outro left wordless) and returns them keyed by block index. Song builder has a "✨ Write section lyrics" panel (theme + Local/Claude) that fills each block; Generate/Restyle have a "✨ Write lyrics" button that fills the Lyrics field. Runs on the Mac via local Gemma (Ollama @ :11434) — no 3090. Verified end-to-end (verses differ, choruses identical, instrumental sections wordless).
  - **Follow-ups (same day):** optional **sung intro/outro** (`extra_sung` param + a Song-panel checkbox; off by default); **"Send these lyrics + style → Generate"** (App-level `handoff` state → GenerateForm applies it via effect); **"Titles" task** in the Assistant dock (`names` system prompt in `llm.py`). All verified live.
- **xl_sft installed + base-vs-sft A/B DONE (2026-05-24):** both `xl_base` + `xl_sft` on the box. Verdict (by ear): base and **sft at cfg6** both good (keepers); **sft at cfg7.3 = muddier → don't raise cfg on SFT**. Base occasionally drops instruments briefly; both SFT takes had a clipping end-burst → fixed (see tidy below). Next GPU audition: the **sampler sweep** on SFT + the AI riffs/solos.
- **GPU A/B of negative prompts** (#3) — the *default* negative thinned the mix; needs by-ear tuning. NEVER run the 3090 (ComfyUI/RVC/SoulX) without asking; Mac-side Demucs/pedalboard/matchering/guitar-render are fine.
- **DONE (2026-05-24):** _Auto-fix ACE end-burst/clipping._ `postfx.tidy_ending` runs on every completed generation (`on_complete` in `app.py`) but **only rewrites when the take clips (peak ≥0.985) or has an end-burst** (peak in the last 20% & >1.25× the body) — clean takes (e.g. base) pass through untouched (no re-encode). When triggered: trims trailing near-silence, fades the last ~0.4 s, peak-limits to −1 dBFS, writes a WAV. Verified: base unchanged, both SFT takes de-clipped (1.017→0.891) + burst tamed.
- **Candidates:** LoRAs (Civitai geoblocked from Claude's web tools — user VPN-downloads + transfers); custom metal LoRA; verify SoulX/DiffSinger on the 3090; reproducibility "regenerate with tweak".
- **DONE (2026-05-24):** _Refined per-genre solo prompts._ All 22 `solo` strings in `backend/genres.py` rewritten to emphasize what the 16th-grid generator can actually render (density/speed, register height, contour, characteristic scale tones, phrasing/rests) rather than un-renderable expression (bends/vibrato/pinch). The algorithmic fallback `_algorithmic_solo` (`backend/guitar.py`) is now tempo-aware: density scales with BPM (doom ~3.75 → thrash ~11 notes/s), slow genres get sparse/sustained/repetitive lines, fast ones get busy scalar runs; uses each genre's scale + register.
- **DONE (2026-05-24):** _Solos in Song arrangements._ A `Solo` section in `generate_riff_arrangement` (`backend/guitar.py`) now renders a high-register **lead line** (`compose_riff` `part="solo"`, genre-aware, LLM or algorithmic) layered over a quieter power-chord **bed** (both summed into the one DI), instead of the prior power-chords-only. Other sections unchanged.
- **DONE (2026-05-24):** _Align guitar to a backing's real sections._ New `backend/sections.py` (librosa structural segmentation, Mac-side): `detect_segments` (MFCC+chroma → agglomerative into k segments + per-segment RMS energy); `align_blocks` re-times a Song arrangement's labeled blocks onto the backing's real boundaries (k=block count, 1:1, labels/order kept, total = backing length); `auto_blocks` builds an arrangement from the backing alone (roles inferred by energy: ends→intro/outro, loud middles→chorus). Wired into `POST /api/guitar/render-amp` via `align_backing` (with `backing_job_id`); `/api/tone/presets` reports `align_available`. Guitar-tab toggle "Align to the backing's real sections" (shown when a backing is picked). Replaces the old assumption that the arrangement matched the backing by section order/seconds. `librosa>=0.10` added to requirements.

_After UI source edits run `cd web && npm run build` for `:8000`; backend changes need `./run.sh` restart._
