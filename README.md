# Crucible — local AI metal studio

A local, private studio for making **rock & metal** (power / symphonic / folk / heavy) and turning it into a **photoreal music video**. Generate instrumentals, restyle tracks, build songs from draggable section blocks, produce vocals with an **AI Vocal Builder**, then script, cast, shoot and assemble a full music video for the finished song - all driven from a Mac UI over a Windows RTX 3090 backend.

> The app is branded **Crucible**; the repo/module stays `MusicGen`.

## Architecture

- **Mac** (this repo): FastAPI backend + React/Vite/Tailwind web UI + MPS/CPU audio tools (Demucs, librosa, ffmpeg assembly).
- **Windows + RTX 3090** - several services, all reached over HTTP:
  - **ACE-Step 1.5 engine** `:8001` - music generation (the main path: Generate, Song, Cover/Restyle) + ACE LoRA training.
  - **ComfyUI** `:8188` - stills (Krea 2 Ultra / Z-Image Turbo, Qwen-Image-Edit) and video (LTX-2.3 22B), plus the ACE-Step graphs used for Repaint / Add-a-Layer.
  - **RVC** `:7897`/`:5050` (voice conversion), **BS-RoFormer** `:5070` (vocal isolation), **analyze** `:5075` (allin1 structure + CLAP), **fs/LoRA helper** `:5080`, and optionally SoulX-Singer / DiffSinger for the Vocal Builder.

Routing between the engines is config-driven (`app_config.json`) - see the CURRENT STATE section of [HANDOFF.md](HANDOFF.md) for the live map.

## Run

```bash
cp app_config.example.json app_config.json   # then edit it for your Windows host IPs/ports
python3 -m venv .venv && ./.venv/bin/pip install -r backend/requirements.txt
(cd web && npm install && npm run build)
./run.sh            # serves the built UI at http://127.0.0.1:8000
```

`app_config.json` (LAN IPs/paths) is gitignored — copy it from `app_config.example.json`; the backend falls back to the example if it's absent. UI development with hot-reload: `cd web && npm run dev` → http://localhost:5173 (proxies `/api` → `:8000`).

## Features

**Music:** Generate (batch + subgenre presets + Simple/Expert) · Song Constructor (draggable sections, templates, compile or per-block stitch) · **Vocal Builder** (AI melody → SoulX/DiffSinger/guide → optional RVC re-timbre) · Import (search the Internet Archive / paste a link / upload → extract a vocal → save as a SoulX voice) · Cover/Restyle · Repaint · Add-a-Layer · Stems (Demucs) · Voice Swap · Backing/Guitar/Tone/Master · Mix · grouped Library · LLM assistant (Ollama / Claude) · ACE-Step LoRA training.

**Music video:** **Characters** (build a reusable identity core + per-video wardrobes from generated reference stills) · **MV Studio** (LLM shot list cut to the song's real structure via allin1 segments + downbeats, master timeline, bulk render, ffmpeg assembly with grade / crossfades / intro pre-roll) · **Shot Editor** (per-shot staged flow: Scene background → Cast → Placement → 3 cheap video drafts → pick → full-res finish, with native single-pass lip-sync) · FlashVSR upscale.

## Docs

- **[HANDOFF.md](HANDOFF.md)** — entry point: **CURRENT STATE** (live routing map, doc trust map, known issues) then history.
- **[VIDEO_PIPELINE_NOTES.md](VIDEO_PIPELINE_NOTES.md)** — the video pipeline's empirical ledger: what works, dead ends, and why. Read before any video work.
- **[docs/](docs/)** — per-feature designs: Shot Editor model, LTXDirector plan, Krea 2 still engine, FFLF lane, AI grading.
- **[PLAN.md](PLAN.md)** — roadmap / phases / decisions (audio).
- **[RESEARCH.md](RESEARCH.md)** — verified technical research (models, ACE-Step wiring, vocal engines, install).
- **[UI_DESIGN.md](UI_DESIGN.md)** — UI/UX brief.

The Windows backends are installed via the `*_AUTO_INSTALL.bat` scripts. Voice-cloning / band-emulation features are **personal-use only** (see PLAN.md risks).
