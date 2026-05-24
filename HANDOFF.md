# Crucible — Handoff / Onboarding

_App name: **Crucible** (AI metal studio). Repo folder is still `MusicGen` and the backend module is `backend/` — only the user-facing name changed._


_Read this first when picking up the project in a fresh context. It's the index to everything and a snapshot of where we are._

Last updated: 2026-05-24

**Repository:** on GitHub at `git@github.com:apothis/Crucible.git` (public, `main`). `app_config.json` is gitignored (copy from `app_config.example.json`).

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
- **User-side (pending):** download **`xl_sft`** (~9.97 GB) → `…/ComfyUI/models/diffusion_models/acestep_v1.5_xl_sft_bf16.safetensors` (HF `Comfy-Org/ace_step_1.5_ComfyUI_files`); then base-vs-sft A/B. Audition the sampler-sweep takes + the AI riffs/solos.
- **GPU A/B of negative prompts** (#3) — the *default* negative thinned the mix; needs by-ear tuning. NEVER run the 3090 (ComfyUI/RVC/SoulX) without asking; Mac-side Demucs/pedalboard/matchering/guitar-render are fine.
- **Candidates:** solos in Song arrangements (a Solo section → lead, not riff); refine genre solo prompts; LoRAs (Civitai geoblocked from Claude's web tools — user VPN-downloads + transfers); custom metal LoRA; verify SoulX/DiffSinger on the 3090; reproducibility "regenerate with tweak".
- **DONE (2026-05-24):** _Align guitar to a backing's real sections._ New `backend/sections.py` (librosa structural segmentation, Mac-side): `detect_segments` (MFCC+chroma → agglomerative into k segments + per-segment RMS energy); `align_blocks` re-times a Song arrangement's labeled blocks onto the backing's real boundaries (k=block count, 1:1, labels/order kept, total = backing length); `auto_blocks` builds an arrangement from the backing alone (roles inferred by energy: ends→intro/outro, loud middles→chorus). Wired into `POST /api/guitar/render-amp` via `align_backing` (with `backing_job_id`); `/api/tone/presets` reports `align_available`. Guitar-tab toggle "Align to the backing's real sections" (shown when a backing is picked). Replaces the old assumption that the arrangement matched the backing by section order/seconds. `librosa>=0.10` added to requirements.

_After UI source edits run `cd web && npm run build` for `:8000`; backend changes need `./run.sh` restart._
