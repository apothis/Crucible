# Crucible — local AI metal studio

A local, private music-generation studio focused on **rock & metal** (power / symphonic / folk / heavy). Generate instrumentals, restyle tracks, build songs from draggable section blocks, and produce vocals with an **AI Vocal Builder** (composes a melody from your song's structure + key, then sings it) — all driven from a Mac UI over a Windows RTX 3090 backend.

> The app is branded **Crucible**; the repo/module stays `MusicGen`.

## Architecture

- **Mac** (this repo): FastAPI backend + React/Vite/Tailwind web UI + MPS/CPU audio tools (Demucs).
- **Windows + RTX 3090**: ComfyUI (ACE-Step generation), RVC (voice conversion), and optionally SoulX-Singer / DiffSinger for the Vocal Builder.

## Run

```bash
cp app_config.example.json app_config.json   # then edit it for your Windows host IPs/ports
python3 -m venv .venv && ./.venv/bin/pip install -r backend/requirements.txt
(cd web && npm install && npm run build)
./run.sh            # serves the built UI at http://127.0.0.1:8000
```

`app_config.json` (LAN IPs/paths) is gitignored — copy it from `app_config.example.json`; the backend falls back to the example if it's absent. UI development with hot-reload: `cd web && npm run dev` → http://localhost:5173 (proxies `/api` → `:8000`).

## Features

Generate (batch + subgenre presets + Simple/Expert) · Song Constructor (draggable sections, templates, compile or per-block stitch) · **Vocal Builder** (AI melody → SoulX/DiffSinger/guide → optional RVC re-timbre) · Import (search the Internet Archive / paste a link / upload → extract a vocal → save as a SoulX voice) · Restyle · Stems (Demucs) · Voice Swap · Mix · grouped Library · LLM assistant (Ollama / Claude).

## Docs

- **[HANDOFF.md](HANDOFF.md)** — entry point: current status, code layout, how to run.
- **[PLAN.md](PLAN.md)** — roadmap / phases / decisions.
- **[RESEARCH.md](RESEARCH.md)** — verified technical research (models, ACE-Step wiring, vocal engines, install).
- **[UI_DESIGN.md](UI_DESIGN.md)** — UI/UX brief.

The Windows backends are installed via the `*_AUTO_INSTALL.bat` scripts. Voice-cloning / band-emulation features are **personal-use only** (see PLAN.md risks).
