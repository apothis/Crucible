# MusicGen — UI/UX Design Brief

_Research + design direction for the Phase 2 UI redesign (React + Tailwind + shadcn/ui). Principle: **research, take good ideas, enhance** — learn from the best tools, then go beyond them for a power-user, metal-focused, local studio._

Last updated: 2026-05-23 (sources at bottom)

## 1. What the leading tools do (2026)

- **Suno** — natural-language prompt + genre selection + reference/sample import + lyric paste + export. v5.5 added voice cloning, custom model fine-tuning, and **"Suno Studio" (a full in-app DAW)**; stem export on higher tiers. Clean, approachable UI; full song in 30–90s.
- **Udio** — control & fidelity focus. **Inpainting** (fix a section without regenerating the whole song), **Extend** with a waveform Crop+Extend and "extension placement", **Manual Mode with Seed** for reproducibility, generation-quality-vs-speed toggle, strong instrument separation, precise key control. Widely cited as the best UX for power users.
- **ElevenMusic** — unified voice + music + SFX with a **built-in basic audio editor** so you mix without an export/import cycle.
- **Mureka** — **style matching from a reference track**, **stem separation**, DAW integration (Ableton), a **generation queue with concurrent jobs**, and a **"Music Agent Studio" that builds songs through dialogue** — adjust structure/sections conversationally without full regeneration.
- **Moises** — waveform **region-select → three-dot contextual actions**.
- **Research frontier** — masked-transformer inpainting, **bar-level / region-wise regeneration**, instrumentation & note-density control.

## 2. Patterns worth stealing
- Structured prompt building: genre/style pickers + reference upload + lyric field.
- **Seed / manual mode** for reproducibility (we have it — surface "reproduce" + "vary").
- **Variations**: generate several, audition, compare.
- **Waveform with region selection → contextual actions** (the single most valuable interaction).
- **Extend / continue** with waveform crop + placement.
- **Inpaint / repaint** a section.
- **In-app editor/mixer** (avoid export/import churn) — matches our D3 decision.
- **Stem view** for separation/isolation.
- **Generation queue / history** with status.
- **Conversational/agent editing** — adjust sections by talking to an assistant.

## 3. Where they're limited (our opportunities)
- They **hide the real knobs** (model, steps, cfg, sampler, denoise). We expose them — our edge for power users.
- They're **cloud, general-purpose, end-to-end**. We're **local, private, metal-focused**, and do **vocals as a separate pipeline** (instrumental → guide vocal → RVC timbre → mix) — closer to real metal production.
- No **parameter sweeps / experiment tracking** — power users tuning metal need this.
- Limited **stem-aware restyle** (restyle just the guitar stem).

## 4. Enhancements to build (beyond the others)
1. **Two-mode controls** — a clean "Simple" surface (prompt + subgenre preset + Generate) over an "Expert" drawer exposing every knob (variant, steps, cfg, shift, sampler, restyle denoise, RVC params). Best of both.
2. **Metal genre builder** — subgenre presets (power / symphonic / folk / heavy) that populate curated instrument + vocal tag bundles, suggested BPM/key. Tag **chips** you add/remove, not a raw text blob.
3. **Experiment / sweep mode** — "generate across cfg 4/6/8" or N seeds → a **compare grid** with A/B audition and one-click "promote to favourite". Built-in benchmarking — nobody else does this.
4. **Full reproducibility + lineage** — every track stores params+seed+parent; "regenerate with tweak", "branch from here".
5. **Waveform workspace** — region select → contextual menu: Extend, Repaint, **Restyle just this region**, separate stems.
6. **Separate-vocal pipeline as first-class** — a guided flow: pick/generate instrumental → write/generate guide vocal (lyric2vocal/DiffSinger) → convert via RVC voice → **in-app mixer** to align/level → bounce. 
7. **LLM assistant panel** (local Gemma or Claude, per D4) — write/iterate lyrics, suggest style tags & concepts, and **conversationally edit** ("make the chorus more symphonic") by translating to prompt/param changes.
8. **Stem-aware everything** (Demucs on Mac MPS, in parallel) — view, solo, restyle, recombine.
9. **Queue & dual-machine status** — show the single-GPU queue honestly, plus that Mac-side stem work runs in parallel.
10. **Song Constructor (block builder)** — a visual arrangement lane of **draggable section blocks** (Intro / Verse / Pre-Chorus / Chorus / Bridge / Solo / Breakdown / Outro), each with editable length and optional per-block lyrics/prompt. Set total song length and per-section length. Two drive modes: (a) **compile to one prompt** — blocks → ACE-Step structured lyrics (`[Verse]/[Chorus]/[Solo]`…) + total duration (order/lyrics honored, per-section length approximate); (b) **per-block generate + stitch** — render each block to its exact length and concatenate/crossfade via the mixer infra (exact lengths, per-block re-roll, lockable blocks). Nobody else offers a true visual song arranger — strong differentiator. Builds on shadcn dnd + our generate/repaint/mix pieces.

## 5. Proposed layout / IA
```
┌───────────────────────────────────────────────────────────────────────┐
│ Top bar: app · ComfyUI/RVC status · global generation queue/progress    │
├──────────────┬───────────────────────────────────────┬─────────────────┤
│ LEFT          │ CENTER — Workspace                     │ RIGHT            │
│ Control panel │ • Waveform player (region select →     │ Library/browser  │
│ • mode tabs   │   contextual actions)                  │ • search/filter  │
│   (Generate / │ • Variations / compare grid (A/B)      │ • favourites     │
│    Restyle /  │ • Mixer view (instrumental + vocal     │ • lineage tree   │
│    Vocals /   │   stems) when in vocal/mix flow        │ • params on hover│
│    Mix)       │                                        │                  │
│ • prompt      │                                        │                  │
│   builder     │                                        │                  │
│ • Simple/     │                                        │                  │
│   Expert knobs│                                        │                  │
│ • presets     │                                        │                  │
├──────────────┴───────────────────────────────────────┴─────────────────┤
│ Assistant (LLM) dock — lyrics, ideas, conversational edits (collapsible) │
└─────────────────────────────────────────────────────────────────────────┘
```
- **Dark "studio" aesthetic** (already our direction); tasteful motion; keyboard shortcuts.
- **wavesurfer.js** for waveforms + region selection; consider spectrogram toggle.
- Components via **shadcn/ui** (sliders, tabs, dialogs, command palette).

## 6. Pitfalls to avoid
- Don't bury the power features so deep they're unusable — but don't overwhelm the default view either (hence Simple/Expert).
- Don't fake real-time audio streaming — these are batch jobs; show honest progress.
- Keep the library reproducible and searchable as it grows; lineage matters for tuning work.
- Don't block the UI on slow GPU jobs; queue + async everywhere.

## Sources
- [How to Create Music with AI in 2026: Suno, Udio — Bertoproduction](https://bertoproduction.com/en/blog/how-to-create-music-with-ai-2026-suno-udio-guide.html)
- [Suno Hub — Create Music with AI](https://suno.com/hub/create-music-with-ai)
- [Udio — Best UX in AI Music (How-To Geek)](https://www.howtogeek.com/udio-offers-the-best-user-experience-in-ai-music-right-now-heres-how-to-use-it/)
- [Udio Guide](https://www.udio.com/guide)
- [ElevenLabs Music — capabilities](https://elevenlabs.io/docs/overview/capabilities/music)
- [ElevenMusic 2026 overview — AI Magicx](https://www.aimagicx.com/blog/elevenlabs-music-ai-audio-content-creators-2026)
- [Mureka AI](https://www.mureka.ai/)
- [Moises AI Studio](https://help.moises.ai/hc/en-us/articles/21745204066076-Moises-AI-Studio-Your-All-in-One-AI-Music-Creation-Platform)
- [Suno vs Udio 2026 — Neuronad](https://neuronad.com/suno-vs-udio/)
