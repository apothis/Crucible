# AI Mix Plan: mixing Suno stem downloads better

Status: PLAN, nothing built. Research pass 2026-10-11.

Provenance tags (per house rule, never cite a guess as fact):
- [V] verified on the cited upstream page during the research pass
- [U] unverified (third-party claim, forum, or not checked)
- [H] our hypothesis, to be tested
- [M] measured by us (none yet)

---

## 1. Goal and non-goals

Goal: take a Suno song (full mix + stem pack) and produce a mix that is better than
Suno's own, with AI doing the analysis and first-pass decisions and the USER keeping
control through editable, interpretable parameters (faders, EQ, comp, pan, sends).

Design rules (from house feedback):
- AI proposes, user adjusts. Output is a "mix doc" of parameters, not a black-box WAV.
- Opt-in. The existing Mix / Master / Stems paths stay exactly as they are.
- All ML inference on the box (3090). The Mac only does DSP rendering (pedalboard) and
  light analysis. Note: `backend/stems.py` still runs Demucs on Mac MPS; this feature
  must never route through it.
- Ears judge. Metrics are sanity checks only (we do not trust CLAP-style fitness).
- Every model or method choice gets an A/B listening gate before it becomes a default.

Non-goals (for now): generative re-performance of parts, re-amping Suno guitar stems
through Helix (they are already amped; the amp-sim rule needs a clean DI), cloud mixing
services (RoEx Tonn API is cloud [V] and stays out by the local-first rule).

---

## 2. What the research changed

### 2.1 Suno stems are regenerated, not separated (the big one)

- Suno's 2026-06-11 blog on the rebuilt stem splitter says it does not carve the
  finished song; it regenerates each stem from scratch with their latest model [V]
  https://suno.com/blog/stem-separation-updates
- Consequence [H]: stems will NOT sum back sample-exactly to the full mix. Phase and
  timing may drift from what is in the mixdown. Suno publishes nothing on summing.
- That matters because the cheapest high-quality trick, the delta method already in
  `postfx.recombine_delta` (`out = mix - stem + processed(stem)`), only works if the
  stem is coherent with what is in the mix. An incoherent stem subtracted from the mix
  leaves an anti-phase ghost.
- Modes available [V] https://help.suno.com/en/articles/13925185 :
  Auto Split (up to 12 stems, 50 credits), Split from Mix (one stem + complement),
  Advanced Split (~100 instruments, Premier). Formats: MP3, WAV, tempo-locked WAV, MIDI.
- 12-stem names (vocals, backing vocals, drums, bass, guitar, keys, strings, brass,
  woodwinds, percussion, synth, FX) [U, third-party guides].
- Sample rate / bit depth: 16-bit/48k for Get Stems, 32f/48k for Studio multitrack
  [U, third-party reviews]. ffprobe on a real download is the only ground truth.
- Suno Studio "Multitrack" export gives per-track WAVs where each clip was generated as
  its own track [V] https://help.suno.com/en/articles/13925249 . If the user has
  Studio access this is probably the cleanest source [H].
- Common complaints: bleed (reverb tails crossing stems), watery/phasey holes, low end
  cancelling when summed, hiss, metallic smear [U, blogs/forums]. Suno claims the new
  stems have no spill [U, vendor claim].

### 2.2 Mixing methods: what is real vs research-only

- No open, pretrained, general multitrack automixer exists that we can just run.
  Diff-MST is CC-BY-NC-SA with no weights [V]; automix-toolkit weights are drums-only
  [V]; MEGAMI (Sony, ICASSP 2026) and StemFX (ISMIR 2026) have no code [V].
- What IS buildable and proven:
  - Rule + measurement automixing (De Man & Reiss 2013; Hafezi & Reiss 2015 masking-
    aware multitrack EQ) [V]. RoEx documents its own product as exactly this: feature
    extraction in context of the other tracks + engineering rules, "not generative AI"
    [V] https://www.roexaudio.com/blog/how-roex-automix-works-under-the-hood
  - LLM as parameter planner: LLM2Fx (WASPAA 2025) shows LLMs map text intent to EQ /
    reverb params zero-shot, better when given DSP features + function code + few-shot
    examples [V] https://arxiv.org/abs/2505.20770 . MixAssist (COLM 2025) found LLMs
    have weak audio understanding [V]. So: feed Claude measurements, never ask it to
    "listen".
  - Reference matching by gradient-free search: ST-ITO (Steinmetz, ISMIR 2024,
    Apache-2.0) runs CMA-ES over ANY effect chain, including VSTs / pedalboard, guided
    by the AFx-Rep embedding [V] https://github.com/csteinmetz1/st-ito . AFx-Rep
    checkpoint host + license not stated [U].
  - Differentiable consoles: dasp-pytorch (Apache-2.0, gain/EQ/comp/reverb/pan/widener)
    [V] https://github.com/csteinmetz1/dasp-pytorch ; GRAFX (Apache-2.0, ships a
    mixing-console fit example) [V] https://github.com/sh-lee97/grafx
- Metrics: none trustworthy alone. KAD (`kadtk`) and Frechet distances over tonal
  balance / crest are the 2026 direction [V]; feature errors track loudness/dynamics/pan
  but not EQ/reverb well [V]. Ears decide.

### 2.3 Cleanup models (box, all roformer-family, ungated)

Most MVSEP leaderboard leaders are private (cannot download) [V]. The usable set is
community releases, all listed with exact filenames in python-audio-separator's
models.json [V]
https://raw.githubusercontent.com/nomadkaraoke/python-audio-separator/main/audio_separator/models.json
Most also run in ZFTurbo MSST, which our `bs-roformer-infer` server already uses [U].

| Job | First pick | Backup | Notes |
|---|---|---|---|
| Vocal bleed out of instrument stems | unwa-97chris MelBand Bleed Suppressor V1 (`mel_band_roformer_bleed_suppressor_v1.ckpt`) [V name, U license] | unwa BS-Roformer Revive 2 ("highest bleedless" per card) [V] https://huggingface.co/pcunwa/BS-Roformer-Revive ; Gabox Denoise-Debleed [V name] | Only model found whose stated purpose is debleed |
| Lead vs backing vocal | becruily BS-Roformer karaoke (`bs_roformer_karaoke_frazer_becruily.ckpt`) [V name] | anvuew karaoke_bs_roformer (GPL-3.0) [V] | No published comparison; order is judgement [U] |
| Drum split | aufr33-jarredou MDX23C DrumSep 6-stem (kick/snare/toms/hh/ride/crash; MVSEP kick 18.3 / snare 13.6 SDR) [V] | jarredou 5-stem DrumSep (ungated GitHub release) [V] https://github.com/jarredou/models/releases/download/DrumSep/drumsep_5stems_mdx23c_jarredou.ckpt | Expect drums-only input [V], which suits Suno's drum stem. hh/cymbals SDR is low (4-7), so process them gently |
| Dereverb | anvuew dereverb MelBand, less-aggressive variant for stacked vocals (GPL-3.0) [V] https://huggingface.co/anvuew/dereverb_mel_band_roformer | Sucial De-Reverb-Echo V2 [V name] | Mono variant is strongest but collapses to mono [V] |
| Denoise / hiss | aufr33 MelBand Denoise (normal, then aggressive only if needed) [V] | Gabox Denoise-Debleed | |
| Re-split muddy instrument stems | MVSep Mega 53 Stems (BS-Roformer, needs >=16 GB VRAM, MSST only) [V] | becruily mel-band-roformer-guitar [V exists, no card] | Author warns each stem is weaker than specialised models [V]. For when Suno lumps guitar+keys+synth |
| Codec / smeared top end | Apollo (CC BY-SA 4.0, music, 44.1k) [V] https://github.com/JusperLee/Apollo | FlashSR (one-step AudioSR) [U repo] | Apollo needs non-zero chunk padding or it seams [V] |
| Per-stem restoration (whole job) | Music Source Restoration challenge winner xlance-msr (chained roformers: sep+denoise+dereverb) [V] https://github.com/ModistAndrew/xlance-msr | | Percussion restores very poorly (0.29 dB avg) [V]. Most on-target research for this use case, worth an experiment |

Not adopted: SonicMaster (Apache-2.0, text-guided restore+master, but authors note the
VAE can make vocals robotic [V]); A2SB (NVIDIA, non-commercial [V]); Resemble Enhance /
VoiceFixer (speech-only [V]). No open model exists for AI-vocal "metallic" artifacts,
de-essing or breath control [V absence]; that stays DSP.

---

## 3. Architecture

```
 Suno full mix (WAV) ---+
                        |                          (box, GPU)
 Suno stem pack -------+--> [0 Ingest + Gate] --> [1 Cleanup chain] --+
 (or Studio multitrack) |        null test,          per-stem model   |
                        |        coherence           recipes          v
                        |                                       [2 Analysis] (Mac)
                        |                                        LUFS, crest, bands,
                        |                                        masking matrix, sections
                        |                                             |
                        |                                             v
                        |                                       [3 Mix engine]
                        |                                        rules -> Claude planner
                        |                                        -> (opt) reference search
                        |                                             |
                        |                                        mix doc (JSON params)
                        |                                             |  <-- user edits
                        |                                             v
                        +---------------------------------------> [4 Render] (Mac, pedalboard)
                                                                  Mode A delta / Mode B sum
                                                                       |
                                                                  [5 Master] existing master.py
                                                                       |
                                                                  [6 A/B] compare player, blind
```

### 3.1 Two mix modes, chosen by Gate 0

**Mode A: Corrective (base = Suno's full mix).**
`out = mix + sum_i (processed_i - source_i)` with latency-align + optional gain hold,
generalising `postfx.recombine_delta` to N stems. Untouched material comes from the clean
mixdown, so stem artifacts are never exposed. Covers: vocal forward/back, low-end
tightening, harshness taming, de-essing, width, per-stem reverb trims.

Mode A needs `source_i` to be COHERENT with the mix. Two ways to get that:
1. Suno stems, IF Gate 0 shows they null well against the mix [H: likely not, per 2.1].
2. Our own separation of the Suno full mix (BS-RoFormer SW, already on the box). Mask-
   based separation outputs sum close to the input by construction [H, measure it], so
   they are coherent with the mix even if they sound worse solo. The delta method hides
   that, because only the DIFFERENCE is added back.

**Mode B: Full remix (base = sum of cleaned stems).**
Use Suno's regenerated stems (which may be cleaner solo, being generated rather than
masked [H]) through the cleanup chain, then sum through the console. More freedom
(rebalance, re-pan, new space), more exposure to artifacts. A residual bus
(`mix - sum(stems)`) is offered as an optional fader only if Gate 0 shows it is small
and coherent; otherwise it would be cancellation garbage and stays off.

### 3.2 Stage 0: Ingest and Gate (the first thing built, cheap, no GPU)

For a given song:
- ffprobe every file: sample rate, bit depth, channels, length. Record [M] values.
- Resample to one working rate (48k if Suno is 48k).
- Align each stem to the mix (sliding xcorr; same idea as the MV intro-offset tool).
- Null test: residual = mix - sum(stems). Report residual LUFS relative to the mix,
  per band (sub/low/mid/high).
- Per-stem coherence: correlation of each stem with the mix in its dominant band, and
  the same for our own BS-RoFormer split of the mix, side by side.
- Output a short report in the UI: "Suno stems null to -X dB: Mode A on Suno stems
  OK / not OK; Mode A will use box re-separation instead."

**Gate 0 result #1 [M], 2026-10-11, "Dream Of Me" (Suno Auto Split, 12 stems + mix):**
scripts in `.mvwork/aimix/scripts/gate0.py`, `gate0b.py`; numbers in
`.mvwork/aimix/dream_of_me/gate0.json`.
- Format: every file 32-bit float WAV, 48 kHz stereo, 289.18 s, identical length.
- Stems do NOT null. `mix - sum(stems)` is +7.7 dB LOUDER than the mix (all bands
  +6..+9 dB). Sum of stems is +7.2 dB hotter than the mix (stems are delivered hot,
  roughly individually levelled [H]). Best least-squares gain per stem only reaches
  -0.04 dB, i.e. no gain setting makes them add up to the mix.
- Same music, same timing, different waveforms: sum-of-stems loudness envelope vs mix
  r = 0.90 (log-env 0.91) at lag 0, but waveform correlation only 0.23. Per stem
  waveform r is 0.0-0.17 while envelope r reaches 0.55-0.75 (Vocals, Drums, Bass, Synth,
  Guitar). Consistent with Suno's "regenerated" statement.
- Verdict: Mode A cannot use Suno stems (no coherence to subtract). Mode A must use a box
  re-separation of the full mix; Suno stems are a Mode B (full remix) source, gain-staged
  down first. Residual bus in Mode B is meaningless here and stays off.

Decision rule [H, thresholds to calibrate by ear]:
- residual <= -30 dB: Suno stems usable for Mode A directly.
- residual -30..-15 dB: Mode A via box re-separation; Suno stems for Mode B only.
- residual > -15 dB: stems are not the mix; Mode B only, and warn.

### 3.3 Stage 1: Cleanup chain (box)

Per-stem recipes, each step optional and toggled in the UI, defaults chosen by A/B:

| Stem | Default recipe [H] |
|---|---|
| vocals | karaoke split (lead/backing) if backing stem missing or polluted -> dereverb (less-aggressive) only if the user wants dry control -> denoise if hiss measured |
| backing vocals | debleed (remove lead leakage via karaoke model) |
| guitar / keys / synth / strings / brass / woodwinds | Bleed Suppressor (pull vocal bleed out, MOVE it to the vocal stem, never discard) -> optional Mega 53 re-split if stem is lumped |
| drums | DrumSep 6-stem -> kick/snare/toms/hh/ride/crash sub-stems, plus a "drum residual" so the drum sum still matches |
| bass | debleed (kick bleed is the usual problem [U]) |
| FX / percussion | passthrough |
| any | Apollo restore as a last optional step if the top end is smeared |

Conservation rule: every removal goes somewhere (moved to the right stem or into a
per-stem residual), so `sum(cleaned) == sum(raw)` up to model error. This keeps Mode B
honest and makes "undo" trivial.

### 3.4 Stage 2: Analysis (Mac, numpy/pyloudnorm/librosa)

Per stem and per section (sections from `backend/sections.py`):
- integrated + short-term LUFS, peak, crest factor
- band energies (e.g. 10 octave or 1/3-oct bands), spectral centroid, tilt
- stereo width / correlation, low-end mono-ness
- masking matrix: for each stem pair and band, how much A masks B (simplified
  Hafezi-style measure). The classic hot spots: vocal vs guitars/keys at 2-5 kHz,
  kick vs bass at 50-120 Hz.
- for the full mix: the same, plus the reference track if one is chosen

### 3.5 Stage 3: Mix engine (three layers, each opt-in, each produces a mix doc)

**Layer 1: Rules (deterministic, ships first).**
Role-keyed defaults from stem names (vocal, drums, bass, ...):
- gain staging to role-relative loudness targets (vocal ~ reference level, bass/kick
  balance, etc.)
- masking-aware cross-adaptive EQ: cut the masker in the band where the priority stem
  lives (priority order editable, vocal first by default); dynamic EQ when the masking
  is intermittent (sidechain from the priority stem)
- high-pass on non-bass stems, low-end mono below ~120 Hz
- role-based pan spread (doubled guitars wide, bass/kick/snare/lead centre)
- gentle per-role compression; one shared reverb + one delay send bus, send levels by role
Targets come from a genre profile table (metal / rock / country-parody, etc.), editable.

**Layer 2: Claude planner (intent).**
Input: analysis JSON + current mix doc + the console's parameter schema with ranges +
a few worked examples + user's free-text notes ("vocal more upfront, chorus guitars
bigger, less boomy"). Output: a mix doc diff, validated against the schema, clamped to
ranges, shown to the user as a diff before render. Uses the existing `claude_sub` path.
Claude never judges audio; it reasons from measurements (LLM2Fx-style).

**Layer 3: Reference match (optional, box).**
User picks a reference track they own. ST-ITO-style CMA-ES over the SAME pedalboard
parameter set (so no second DSP implementation), loss = loudness + band-balance + crest
+ width stats, plus AFx-Rep embedding distance once its license is confirmed. Runs on the
box as a job; result is a mix doc, same as the other layers. dasp-pytorch / GRAFX
gradient fitting is a later speed-up only, because it means keeping a torch twin of every
pedalboard effect in sync.

### 3.6 Mix doc (the contract between every layer, the UI, and the renderer)

```json
{
  "version": 1,
  "song": "<library job>",
  "mode": "corrective | remix",
  "source": "suno_stems | box_resep",
  "sample_rate": 48000,
  "stems": {
    "vocals": {
      "file": "...", "gain_db": 1.5, "pan": 0.0, "mute": false,
      "chain": [
        {"fx": "highpass", "hz": 90},
        {"fx": "peq", "bands": [{"hz": 3000, "q": 1.0, "db": 2.0}]},
        {"fx": "compressor", "threshold_db": -18, "ratio": 3, "attack_ms": 10, "release_ms": 120},
        {"fx": "deesser", "hz": 6500, "range_db": 4}
      ],
      "sends": {"verb": -18, "delay": -24},
      "sidechain_duck": []
    }
  },
  "buses": {"verb": {"fx": "reverb", "size": 0.4, "predelay_ms": 30}, "delay": {"fx": "delay", "ms": 375, "fb": 0.25}},
  "master": {"preset": "neutral", "target": "keep"},
  "provenance": [{"layer": "rules", "at": "..."}, {"layer": "claude", "notes": "..."}]
}
```

Every parameter maps 1:1 to a pedalboard plugin (or a small custom DSP block for
de-esser / dynamic EQ). Stored next to the job; re-render needs no AI.

### 3.7 Stage 4-6: Render, master, judge

- Render on the Mac with pedalboard (native plugins only; Helix optional per stem, off
  by default). Mode A = generalised delta recombine; Mode B = summing console.
- Master with existing `master.py` (Neutral open master / auto / reference match).
- Judge with the existing compare player in blind mode, loudness-matched, three takes:
  Suno original / Mode A / Mode B. KAD and feature stats shown as a side note only.

---

## 4. Box-side changes

- `roformer_server.py` is single-model today (CONFIG/CKPT from env). Needed: a model
  registry + `model=` form field, one subprocess per request (same free-the-GPU design),
  so the cleanup chain can call debleed / karaoke / drumsep / dereverb / denoise.
  Alternative: a python-audio-separator service, which already knows every model above
  by filename [V]. Decide in Phase 1 by checking which runner loads all the picks [U].
- Model downloads: one installer `.bat` in `ComfyUI_windows_portable\` style per the box
  convention (pure `curl -L -C -`, `%~dp0`-relative, ungated URLs only, no HF token).
  User runs it.
- Mega 53 needs >=16 GB VRAM [V], so free ComfyUI/engine first (existing `free_gpu`).
- Serialise with other GPU jobs (no concurrent engine + analyze work).

---

## 5. Phases (each ends in a listening gate; nothing fires without a go-ahead)

**Phase 0: Gate 0 tooling (Mac, no GPU).**
Ingest a real Suno stem pack (user supplies one song: full mix WAV + Auto Split WAVs,
plus a Studio multitrack export if available). ffprobe, align, null test, coherence vs a
box BS-RoFormer split. Deliverable: a measured [M] answer to "do Suno stems null?" and the
mode decision rule calibrated.

**Phase 1: Mode A corrective MVP.**
N-stem delta recombine + Layer 1 rules limited to safe moves (vocal level/EQ, low-end
tightening, harshness dynamic EQ, de-ess). Mix doc + render + compare. Gate: blind A/B
vs Suno original on 3 songs, user's ears.

**Phase 2: Cleanup chain on the box.**
Multi-model separation endpoint + installer. A/B each model in isolation on the same
stem (labelled files, one variable). Gate: per-model keep/drop.

**Phase 3: Mode B remix + full Layer 1.**
Summing console, drum sub-stems, sends/buses, masking EQ, pan. Gate: A/B vs Mode A and
Suno original.

**Phase 4: Claude planner + mix UI.**
Fader/EQ panel bound to the mix doc, notes box, diff preview, re-render. Gate: does the
planner's diff move the mix the way the note asked (ears).

**Phase 5: Reference match.**
ST-ITO-style search on the box over the pedalboard params. Confirm AFx-Rep license first.

**Later / experiments:** Apollo and xlance-msr restoration A/B; Mega 53 re-split;
dasp-pytorch gradient fit for speed.

---

## 6. Risks and open questions

- Suno stem summing is the biggest unknown; Phase 0 exists to measure it before we build
  on an assumption.
- Separation and dereverb models can make things worse (thin, phasey). Every model step
  is optional and A/B gated; the delta method limits damage in Mode A.
- hh/cymbal drum-split quality is low (SDR 4-7 [V]); keep cymbal processing gentle.
- Licences: many community models state none; anvuew models are GPL-3.0; becruily deux
  and LarsNet are CC-BY-NC [V]. Fine for personal local use; matters only for
  redistribution.
- Loading community `.ckpt` pickles: only from the named upstream repos, never from
  random mirrors.
- LLM planner overconfidence: schema-validate, clamp, show the diff, user approves.
- Metrics drift from ears: never auto-select by metric.

## 7. Watch list (no code yet, re-check periodically)

MEGAMI (Sony, ICASSP 2026) https://arxiv.org/abs/2511.08040 ; StemFX (ISMIR 2026)
https://arxiv.org/abs/2607.15634 ; sequential stem blending with flow matching
https://arxiv.org/abs/2608.05506 ; Fx-Encoder++ https://arxiv.org/abs/2507.02273 ;
private MVSEP models (MelBand DrumSep, BS 124-band) if ever released.
