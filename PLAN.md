# Crucible — Project Plan & Roadmap

_App name: **Crucible** — AI metal studio (where raw sound is melted down and reforged). Repo dir still `MusicGen`._


_Living document. Check items off as we go. Technical details and verified facts live in `RESEARCH.md`; this doc is the map of WHAT we're building and in what order._

Last updated: 2026-05-23

---

## 1. Vision

A local, private music-generation studio focused on **rock & metal** — heavy / power / symphonic / folk metal **and heavy rock** (e.g. Bon Jovi, Halestorm, Black Stone Cherry, AC/DC). It should:

- Generate full instrumental tracks from a text prompt.
- Restyle an existing track into a new genre.
- Produce vocals separately (clean / powerful / operatic) and mix them in.
- Give deep, **guided control** over the musical output (tuning is a first-class feature, not an afterthought).
- Use an LLM to help with the creative side (lyrics, style ideas, concepts).
- Have a **genuinely modern, polished UI**.
- Run across two machines: Mac (UI + orchestration + MPS-capable tools) and a Windows RTX 3090 (heavy generation).

Working principle for output quality: **generate-then-curate** — batch, audition, pick, refine, post-process. No model one-shots finished metal.

**Guiding principle (whole app): research, take good ideas, and enhance.** Look at how others solve each problem, borrow what's good, but improve on it — don't be limited to what existing tools do.

---

## 2. Architecture (current + target)

```
Mac (this repo)                                Windows + RTX 3090
─ FastAPI backend (orchestration)   ── HTTP/WS ──► ComfyUI :8188  (ACE-Step generation)
─ Web UI (browser)                  ── HTTP   ──► RVC :7897      (voice conversion)
─ MPS/CPU tools (Demucs, Basic Pitch, madmom — run in parallel)
─ SQLite library, audio files
```

**Compute placement rule** (see `RESEARCH.md §8b`): CUDA-locked heavy work → Windows (generation, RVC). Light/CPU or MPS-capable work → Mac, in parallel, to keep the 3090 free (Demucs, Basic Pitch, madmom).

**Verified working** (see `RESEARCH.md §8`): Mac→ComfyUI generation end-to-end; ACE-Step 1.5 XL wiring; RVC reachable at `:7897`.

---

## 3. Current status (Phase 0 — DONE)

- [x] Research: models, vocals, serving, ACE-Step workflows (`RESEARCH.md`)
- [x] Windows auto-installers: ComfyUI+ACE-Step, RVC (`*_AUTO_INSTALL.bat`)
- [x] Verified ACE-Step 1.5 XL generation via API (correct encoder wiring + settings)
- [x] FastAPI backend: ComfyUI client, workflow builders (t2m + restyle), WS progress, SQLite library
- [x] Web UI v1: generate + restyle, guided tuning controls, progress, playback, library
- [x] Confirmed end-to-end through the app (generate → download → library)

---

## 4. Feature roadmap

Phases are ordered by a suggested priority but can be reordered. Each phase = a coherent chunk of work.

### Phase 1 — Music quality & core workflow ★ (big, ongoing)
The "tuning the music is a lot of work" workstream. Goal: reliably get GOOD metal out.
- [x] **Batch generation** — Variations (1–4) on Generate → compare grid (done in Phase 2 UI).
- [x] **Prompt presets per subgenre** — `presets.ts` (Power/Symphonic/Folk/Heavy/Thrash/Doom).
- [x] **A/B compare** — workspace compare grid of takes (done in Phase 2 UI).
- [ ] **Model variants** — download `xl_sft` + `xl_turbo`; expose turbo as fast-preview, base/sft as final. (xl_base already installed; UI variant picker already filters by what's installed.)
- [ ] **Tuning presets** — save/load named setting bundles (variant + steps + cfg + shift + tags).
- [ ] **Parameter exposure** — add AuraFlow `shift`, sampler/scheduler choice, cfg_scale/temperature (currently fixed in `comfy.py`).
- [ ] **Reproducibility surfacing** — every track stores full params + seed (✅ stored); add "regenerate with tweak"/"branch from here" in the UI.
- [~] **Post-processing** — guitar **Tone** stage DONE (`backend/postfx.py` + "Tone" tab: pedalboard EQ/cab/saturation presets + optional Helix Native on the `htdemucs_6s` guitar stem → recombine, mode `tone`; verified end-to-end Mac-side). TODO: loudness normalization / `matchering` master (§10e #2).

### Phase 2 — UI/UX redesign ★ (modern look & feel) — _IN PROGRESS_
The classic UI is a functional vanilla-JS placeholder (still at `:8000`). New app being built in `web/`.
- [x] **Foundation** — Vite + React + TS + **Tailwind v4** app in `web/` (dev `:5173`, proxies `/api`→`:8000`). Dark "studio" theme, 3-column layout (controls / workspace / library), live status chips, library with players. `preview_start` config in `.claude/launch.json`.
- [x] **All flows ported** — Generate, Restyle, Vocals (+ Add-voices search/install), Voice Swap, Stems (inline stem players), Mix. Shared `ui.tsx` (Field/Slider/PrimaryButton/pollJob) + `forms.tsx`. Type-checks clean; verified rendering via preview.
- [x] **Waveform players** (wavesurfer.js, `WavePlayer.tsx`) on workspace results.
- [x] **Batch generation + compare grid** — Variations (1–4) on Generate; Workspace renders takes as a grid of result cards (progress → waveform). Unified results model in `ui.tsx` (Result/RunCtx/pollJob/runSync); all flows feed it (stems → one card per stem).
- [x] **Open-in-workspace** — click a library track's "↗ open" to load it (with waveform) into the workspace.
- [x] **Subgenre presets + Simple/Expert modes** — `presets.ts` (Power/Symphonic/Folk/Heavy/Thrash/Doom tag bundles + BPM/key) as clickable chips; Simple hides deep tuning, Expert reveals the full grid. On Generate + Restyle.
- [x] **LLM assistant dock (D4)** — `backend/llm.py` (Ollama + Claude providers) + `/api/llm` + `Assistant.tsx` bottom dock: Lyrics / Style tags / Ideas tasks, provider+model picker (Gemma local default; Claude if `ANTHROPIC_API_KEY` set), copy output. Verified with Gemma. (Future: one-click "send to prompt/lyrics field".)
- [x] **Production build** — `npm run build` → `web/dist`; FastAPI serves it at `:8000` (API same-origin), falls back to classic `frontend/` only if dist absent. The React app is now the primary UI. App named **Crucible**. `run.sh`/`HANDOFF.md` updated.
- [x] **Song Constructor** (flagship) — built. Song tab + `SongForm` (`web/src/forms.tsx`): draggable section-block lane (HTML5 DnD, no new deps), per-block length + optional lyrics, total length. Drive (a) compile → structured lyrics + total duration → `/api/generate`; drive (b) per-block generate + crossfade-stitch (`backend/mix.py:stitch` + `POST /api/stitch`), lockable blocks, mode `song`. Reorder via **@dnd-kit**. **Song templates** (`SONG_TEMPLATES` in `presets.ts`, card-grid selector) = ready-made arrangement layout + the template's own bespoke style, applied in one click (confirm before replacing an edited arrangement). `PRESETS` subgenre tag bundles expanded to 14. Mode (a) verified end-to-end. See Phase 6 + `UI_DESIGN.md` §4.10.
- [ ] **Polish (later)** — spectrogram view, keyboard shortcuts, motion/transitions, send-to-form handoffs (assistant→prompt, stem→vocals), optional native packaging (Tauri/Electron).

### Phase 3 — Vocals pipeline ★ — _IN PROGRESS_
Produce vocals separately, then mix. **The vocal pipeline has 4 stages:**
**CREATE** (ACE-Step sings lyrics — works now; or DiffSinger/Synth V to compose) → **ISOLATE** (Demucs pulls a clean vocal stem — Phase 4, the missing link) → **RE-TIMBRE** (RVC → a specific singer, optional — done) → **MIX** (in-app mixer — D3). RVC only does the RE-TIMBRE stage; vocal *creation* is ACE-Step. (See `RESEARCH.md §5`.)
- [x] **Drive RVC over the network** (`:7897`) — Gradio 3.14 `/run/<api>` sync endpoints; audio bridged to the Windows box via ComfyUI's input dir (both on same machine), output fetched via `/file=`. (`backend/rvc.py`)
- [x] **Convert** guide vocal → target timbre (Vocals tab: upload vocal, pick voice, transpose/pitch-method/index-rate/envelope/protect controls). Verified end-to-end.
- [x] **Voice listing & select** in UI (auto-matches feature index).
- [x] **Migrate off the Gradio hack → clean REST API.** (rvc-python abandoned — its `fairseq==0.12.2` has no Windows wheel, won't install; see `RESEARCH.md §8c`.) Instead: **`backend/rvc_server.py`** = small FastAPI server that runs INSIDE the existing RVC WebUI package using its `runtime\python.exe` (reuses the working fairseq/torch env + Gradio's bundled fastapi/uvicorn — no new install). It speaks the same API dialect, so `backend/rvc_py.py` + the voice installer work unchanged. **`RVC-API_AUTO_INSTALL.bat`** copies the server into the package, optionally downloads starter voices, and launches it on `:5050`. `rvc_driver: auto` auto-detects it. **✅ VERIFIED LIVE: server runs on the PC (:5050), Mac auto-switched to `rvc_python` driver, conversion ran end-to-end (Mac → API → WAV → library). Voices FreddieMercury + james_hetfield available.** NOTE: keep the RVC package (we reuse it); run `run_rvc_api.bat` instead of `go-web.bat`.
- [x] **In-app voice download/install helper** (`backend/voices.py` + Vocals-tab "Add voices"): HF API search (keyword-on-repo-id; not semantic) with sort, repo file listing (pairs .pth+.index), and **install-from-URL** (for voice-models.com finds). Installs land on the Windows PC via rvc-python's `/upload_model`. **Requires rvc-python running** (the install path). Search verified; install pending rvc-python.
- [x] **Starter metal voices** — `RVC-PYTHON_AUTO_INSTALL.bat` optionally downloads **James Hetfield** (thrash) + **Freddie Mercury** (powerful clean/operatic) into `rvc_models/` (verified HF direct downloads, ~450 MB). More via the in-app helper. ⚠️ real-artist clones = personal-use only.
- [~] **Guide vocal source (D2) — superseded by the AI Vocal Builder.** Instead of just `lyric2vocal`, built a **compose-the-melody** pipeline (RESEARCH.md §5b): AI Melody Composer (`backend/melody.py`, hybrid LLM scale-degrees + music-theory realization, in-key/verse-low/chorus-lift, syllable-aligned, MIDI export) → engine-agnostic singing synth (`backend/voicegen.py`: `guide` Mac-now, `soulx` zero-shot, `diffsinger`) → optional RVC re-timbre → mix. New **Voc. Builder** tab (`web/src/VocalBuilder.tsx`) with an SVG piano-roll, pulling the live Song arrangement. Mac path verified end-to-end; SoulX/DiffSinger pluggable + documented, Windows servers/installers TODO.
- [x] **Mixing (in-app mini-mixer — D3)** — `backend/mix.py` + Mix tab: layer N tracks (library items + stems) with per-track gain (dB) + start offset (s), auto-resample to 44.1k, normalize, bounce a stereo WAV into the library. ✅ verified. _Later: per-track FX, waveform alignment UI, stem export to DAW._
- [x] **One-click Voice Swap** (`/api/voiceswap` + Voice Swap tab) — pick a vocal song + a target voice → auto split (Demucs) → re-timbre vocal (RVC) → remix over its own instrumental. Stays in sync (all from one source). ✅ verified end-to-end (Freddie Mercury over its own instrumental). Optional transpose + per-stem gain.
- [ ] **Voice cloning from existing singers** — train an RVC (or SVC) voice on a target singer's vocals to clone their timbre/style, then convert guide vocals to it. (RVC is built for exactly this; ~10–60 min of clean source audio.) ⚠️ rights — see risks.
- [ ] (Later) SoulX-Singer / DiffSinger as alternative vocal engines; zero-shot voice cloning from a short reference clip.

### Phase 4 — Stem separation & advanced restyle
- [x] **Demucs on Mac MPS** (parallel to the 3090) — `backend/stems.py` + Stems tab. 2-stem (vocals/instrumental) or 4-stem split of an uploaded file or a library track; ~10s for 35s on MPS; no ffmpeg needed (soundfile decodes mp3); CPU fallback. ✅ verified end-to-end (incl. isolating a created vocal). _Next: a one-click "send vocal stem → Vocals tab" handoff._
- [ ] **Analysis** — Basic Pitch (melody→MIDI), madmom (BPM/key/chords) → auto-build accurate restyle prompts + lock tempo/key.
- [ ] **Per-stem restyle & recombine** — restyle drums/harmony separately, keep/replace melody.
- [ ] **Cover-strength UX** — map research guidance (0.3–0.5 big genre jumps) into clear controls; multi-pass option.

### Phase 5 — LLM-assisted creativity (mostly DONE via the Assistant dock)
- [x] **LLM provider abstraction (D4)** — `backend/llm.py`: local Ollama (`gemma4_4b`/`gemma4_2b`) + Claude (if `ANTHROPIC_API_KEY` set), selectable per request.
- [x] **Lyric writer** — Assistant "Lyrics" task ([Verse]/[Chorus] structure, metal themes).
- [x] **Style/idea assistant** — Assistant "Style tags" + "Ideas" tasks.
- [ ] **Naming** — track/album/concept titles (add as another assistant task).
- [ ] **Send-to-form handoff** — one click to push assistant output into the prompt/lyrics fields (currently copy-paste).

### Phase 6 — Advanced generation & polish
- [x] **Song Constructor (block builder)** — BUILT. Visual arrangement of draggable section blocks (Intro/Verse/Pre-Chorus/Chorus/Bridge/Solo/Breakdown/Outro) with editable per-section + total length, optional per-block lyrics. Drive (a) compile blocks → ACE-Step structured lyrics + total duration → `/api/generate` (order/lyrics exact, section length approximate); drive (b) generate each block to exact length + crossfade-stitch via `backend/mix.py:stitch` (`POST /api/stitch`) — exact lengths, per-block re-roll, lockable blocks, saved as mode `song`. Reorder via **@dnd-kit** (smooth animated drag). Also: `comfy.py` now preserves bracketed structure tags in lyrics for instrumental generations so arrangement is honored without vocals. Mode (a) verified end-to-end (instrumental 60s gen). (See `UI_DESIGN.md` §4.10.)
- [ ] **Artist / band style emulation** — generate songs that emulate a specific band's sound. Approaches, in increasing effort: (a) curated prompt presets describing a band's instrumentation/era/production; (b) reference-audio conditioning (restyle/cover from a representative track); (c) LoRA fine-tuning of ACE-Step on a band's catalogue for a true "style model". ⚠️ rights — see risks.
- [ ] **Repaint / edit** a section of a track (ACE-Step repaint).
- [ ] **Extend / continue** a track.
- [ ] **Export** — WAV/FLAC, stems, "send to DAW", project bundles.
- [ ] **Library upgrades** — favourites/rating, lineage (parent track), search/filter, projects/grouping.
- [ ] **Additional models** — YuE (harsh vocals, optional/Windows), HeartMuLa (watch).
- [ ] **Robustness** — model lifecycle/VRAM (`/free`) management, queue view, warmup, optional gateway-on-Windows + auth.

---

## 5. Cross-cutting concerns

- **Data model / library**: params, seed, lineage (restyle parent), ratings, tags, project grouping. (SQLite now; may grow.)
- **Reproducibility**: a track must always be reproducible from stored params+seed.
- **Performance / parallelism**: exploit Mac MPS + Windows CUDA simultaneously; keep the 3090 for generation.
- **Error handling**: surface ComfyUI node errors clearly; ws reconnect; cancel (interrupt + queue clear).
- **Config**: `app_config.json` for hosts/ports; per-machine paths.
- **Licensing**: keep to permissive/commercial-OK tools where possible (ACE-Step, Demucs, RVC, Basic Pitch all OK).

---

## 6. Open decisions (track these)

| # | Decision | Choice | Status |
|---|---|---|---|
| D1 | Frontend framework for the redesign | **React + Tailwind + shadcn/ui** | ✅ decided |
| D2 | Guide-vocal source | **ACE-Step lyric2vocal first**, OpenUtau+DiffSinger later | ✅ decided |
| D3 | Mixing approach | **In-app mini-mixer first**, stem export later | ✅ decided |
| D4 | LLM for lyrics/ideas | **Both** — local Ollama (gemma4_4b / gemma4_2b @ localhost:11434) + optional Claude API | ✅ decided |
| D5 | Native packaging eventually? | **Stay web for now**, revisit when UI matures | ✅ decided |
| D6 | Gateway-on-Windows + auth | **No for now** (single-user LAN) | ✅ decided |

_Mac Ollama also has embedding models (`bge-large`, `mxbai-embed-large`) — candidate for semantic library search later._

---

## 7. Known risks

- Metal is the hardest genre for these models; distorted-guitar + intelligible vocals remain weak — mitigated by separate vocals + post-processing + curation.
- Single 3090 = serial generation; batch/quality work is time-bound by GPU throughput.
- Open-weight + powerful/operatic vocals don't coexist out-of-box — RVC voice training or a commercial vocal tool may be needed (D2).
- Music tuning is open-ended; needs a disciplined benchmarking approach, not endless ad-hoc tweaking.
- **Rights / ethics of cloning real artists & bands**: cloning a real singer's voice or emulating a named band can infringe rights (likeness, copyright, "sound-alike" laws) — varies by jurisdiction and is generally fine for **private/personal use** but risky to **share or monetise**. Keep these features clearly personal-use; flag in-app; never train on material you don't have rights to if outputs leave your machine.

---

## 8. Status summary & immediate next actions

**Done:** Phase 0 (research + installers + verified generation); Phase 2 UI redesign (Crucible — React/Tailwind app served at `:8000`, all 6 flows, waveforms, batch+compare, presets, Simple/Expert, assistant dock, production build); Phase 3 vocals pipeline (RVC clean API + custom `rvc_server.py`, voice install helper, starter voices, in-app mixer, one-click Voice Swap); Phase 4 Demucs stem separation; Phase 5 LLM assistant.

**The whole creative loop works end-to-end:** generate (batch) → restyle / stems / voice-swap → mix → library, with an LLM assist dock.

**Done (2026-05-24):** ▶ **Song Constructor** (Phase 6 / Phase 2 flagship) — draggable section-block builder, both drive modes (compile + per-block stitch). Song tab + `SongForm`, `backend/mix.py:stitch`, `POST /api/stitch`.

**MUSIC QUALITY PUSH — built 2026-05-24** (research `RESEARCH.md §10`; UI `UI_DESIGN.md §7`; full detail in `HANDOFF.md`). Shipped a controllable **guitar pipeline** that sidesteps ACE's weak distorted guitars:
- **Backing** (strip guitar via 6-stem) → **Guitar** (AI/algorithmic riff *or* solo, per-genre, → clean DI [Karplus-Strong / SoundFont / Shreddage-Kontakt] → amp [tone presets / Helix Native]) → **Tone** (reshape) / **Master** (matchering reference-master).
- **Unified genre registry** (`backend/genres.py`, 22 genres incl. neoclassical) → drives generation chips + riff/solo pickers; genre **suggests** bpm/key, doesn't force.
- **Source tuning:** real negative prompts + sampler/scheduler/shift exposed (`comfy.py`).
- **UI redesign:** grouped sidebar + inline results + library drawer.
- Honest finding: a generated/separated guitar can't be cleanly de-amped (RemFx too weak; prompting-for-clean fails) — so amp-sims run on a *generated clean-DI* (symbolic→DI→amp) rather than on the ACE output.

**Next up (pending):** download `xl_sft` → base-vs-sft A/B; GPU A/B of negative prompts (default thinned the mix); refine genre solo prompts; LoRAs (Civitai via user VPN); custom metal LoRA.

**Done (2026-05-24):** ▶ **Align guitar to a backing's real sections** — `backend/sections.py` (librosa) detects the backing's actual section boundaries and either re-times a Song arrangement onto them (`align_blocks`, labels kept) or auto-builds one (`auto_blocks`, roles by energy). Wired to `/api/guitar/render-amp` (`align_backing`) + Guitar-tab toggle. Removes the prior "arrangement matches the backing by order/seconds" assumption.
**Done (2026-05-24):** ▶ **Solos in Song arrangements** — a `Solo` section now renders a genre-aware high-register lead line over a quieter power-chord bed (`generate_riff_arrangement` in `backend/guitar.py`), instead of power-chords-only.
**Done (2026-05-24):** ▶ **Refined per-genre solo prompts** — all 22 `solo` descriptions in `backend/genres.py` rewritten around renderable directives (density/register/contour/scale-tones/phrasing); `_algorithmic_solo` made tempo-aware (slow=sparse/sustained, fast=busy runs).

**Then (secondary):** verify SoulX/DiffSinger on the 3090; `xl_turbo`; reproducibility "regenerate with tweak"; voice cloning/training.

_Last updated: 2026-05-24._
