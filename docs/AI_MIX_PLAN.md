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

### 3.0 Scope decision (user, 2026-10-11)

The product is an AI-driven mix BUILT FROM SUNO'S STEMS (Mode B below). That is the
brief. Mode A (corrective delta on a box re-separation of the full mix) is out of the
main path: it is a different product (polishing Suno's mixdown), parked as a possible
later option only. The box separation models stay in scope ONLY as cleanup tools applied
to Suno's own stems (debleed, drum split, lead/backing split, dereverb, denoise).

Consequences:
- Suno's full mix is used as a REFERENCE (A/B baseline and optional target for balance),
  never as audio in the output.
- The output is the sum of processed Suno stems through the console.
- Stems arrive hot (sum +7.2 dB over the mix [M]), so ingest gain-stages them first.

### 3.1 Two mix modes (Mode B is the product; Mode A parked)

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

**Gate 0 result #2 [M], 2026-10-11, same song, box BS-RoFormer SW split of the mix**
(app route `/api/stems/separate engine=roformer`, 101 s, output 48 kHz; script
`.mvwork/aimix/scripts/gate0c.py`):
- 6 stems (vocals/drums/bass/guitar/piano/other) sum to the mix at waveform r = 0.9995,
  lag 0, sum level -0.22 dB.
- Null: -22.5 dB overall; per band sub -33, low -32, lowmid -23, highmid -17, high -18 dB.
  Per second: median -29.8, p90 -24.6, worst -9.0 dB. Most of the loss sits in 2-6 kHz.
- Stem levels rel mix: vocals -4.1, other -6.1, drums -10.0, bass -11.0, piano -24.6,
  guitar -32.6 dB (the SW model put almost no guitar in "guitar"; that content lands in
  "other" for this song [M], so per-instrument control from this split is coarse).
- The extra "instrumental" output is NOT mix minus vocals here (vocals + instrumental is
  +3.3 dB over the mix); unexplained, unused.
- Verdict: Mode A via box re-separation is viable. In the delta method the output keeps
  the original mix untouched and adds only `processed_i - stem_i`, so the -22.5 dB
  residual never matters; what matters is that each stem is coherent with its slot in
  the mix, which r = 0.9995 supports. Finer per-instrument control in Mode A needs the
  Phase 2 models (karaoke, drumsep, Mega 53), run on the MIX for coherence.

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

Re-scoped 2026-10-11 to the brief: every phase mixes SUNO'S STEMS.

**Phase 0: Gate 0 tooling. DONE.**
Measured on "Dream Of Me" (results #1 and #2 in 3.2). Suno stems are the same arrangement
as the mix but different waveforms, delivered hot. They behave like a multitrack source (bleed quality still unjudged),
not a decomposition of the mixdown.

**Phase 1: Suno-stem console MVP (Mac, DSP only). BUILT 2026-10-11, awaiting the ears gate.**
Code: `backend/aimix.py`, routes `/api/aimix/run`, `/api/aimix/{sid}`, `/api/aimix/{sid}/render`
in `backend/app.py`, UI `web/src/AIMix.tsx` (Finish -> AI Mix). Session data in
`library/aimix/<sid>/` (extracted stems, session/analysis/mixdoc JSON).
As built:
- Suno balance = NNLS fit of per-stem gains to the mix's 10-band x 100 ms POWER envelopes
  (each band weighted equally). Dream Of Me: log-envelope fit r = 0.97 [M]; the plain sum
  lands within 1.4 dB of the Suno mix in every band [M]. Brass/Woodwinds fit to -30/-25 dB,
  i.e. Suno barely used them in its own mix.
- Rules: role HPF (capped below the stem's own 2%-energy point), masking cuts on maskers
  in 1-8 kHz where they cover the lead (score > 0.30, max -4 dB), role compression with
  threshold set from the stem's p95 100 ms level, width + low-mono below 120-200 Hz
  (zero-phase M/S split), verb + tempo-synced delay sends. All constants [H].
- Each stem is gain-landed on a target LUFS AFTER its chain, so compression never moves
  the balance; profile offsets (rock/vocal/suno) sit on top.
- Both renders mastered with master_auto tone=flat, glue off, to the Suno mix's LUFS,
  -1 dBTP, 24-bit: the A/B hears mix decisions, not loudness.
- Not yet: de-esser, dynamic (sidechain) EQ, per-section moves, editable panel (Phase 2).
Original scope:
- Ingest a Suno zip: map file names to roles (Vocals, Backing_Vocals, Drums, Bass, Guitar,
  Keyboard, Synth, Strings, Brass, Woodwinds, Percussion, FX), drop silent stems, gain-stage
  so the plain sum sits near the Suno mix's loudness.
- Analysis (3.4): per-stem LUFS, crest, bands, width, masking matrix, sections; the Suno
  full mix analysed as the baseline balance.
- Layer 1 rules (3.5) write a mix doc: role-relative levels, HPFs, masking-aware EQ cuts,
  low-end mono, pan spread, gentle comp, one verb + one delay bus.
- Render with pedalboard, master with existing master.py, file it in the library.
- Gate: blind A/B in the compare player, loudness-matched: Suno full mix vs plain stem sum
  vs rules mix, on Dream Of Me plus 2 more songs. User's ears.

**Phase 1b: Adaptive, measurement-led rules (user direction 2026-10-11). BUILT, presets building.**
User feedback on Phase 1: rules mix slightly better than Suno's; no added vocal reverb (Suno
vocals arrive wet); drums lack punch (in Suno's mix too). Direction: no fixed defaults, every
element tuned per song from measurements.
- Style presets = distributions of per-role measurements over reference songs (box split of the
  middle 2 min, numbers only): `backend/aimix_refs.py`, routes `/api/aimix/presets[/build]`.
  First three: symphonic (11 Nightwish + 4 Tarja), power (8 Battle Beast + 7 Beast in Black),
  hardrock (15 The Pretty Reckless). Per-song caches in `library/aimix/presets/_cache_<name>/`.
- Measurements (`backend/aimix_features.py`), per separator role (vocals, drums, bass, guitar,
  piano, other; Suno stems grouped onto them): share of mix (rel LUFS), third-octave LTAS shape,
  loudness range, short-term crest, punch (attack 0-15 ms over body 30-120 ms at low-band hits),
  sibilance (5-9 kHz bursts), width + low-end width, wetness (dip depth inside phrases),
  vocal presence over the rest in 1-5 kHz. Punch and wetness are proxies [H].
- Rules (`backend/aimix_adaptive.py`): target = clip(current, p25, p75); inside the band = no
  move. Joint level solve in power; EQ from the smoothed third-octave gap (<= 4 moves, +-3 dB);
  compression only on excess loudness range; drum punch and de-ess are closed loops (search the
  amount until the measure reaches the band); width only for gaps >= 2 dB; mono lows / reverb
  only when off-style. Masking cuts and content-capped HPF kept. Auto preset = nearest by whole-
  mix shape. Roles > 24 dB below the mix count as absent (separation residue) on both sides.
- Calibration [M], Dream Of Me separated vs its Suno stems: vocals/other offsets <= 1.6 dB;
  drums/guitar/piano offsets large but dominated by near-empty separated stems; one song cannot
  separate separation bias from regenerated-stem difference. NOT applied; needs more Suno packs.

Ears check #1 on Dream Of Me (2026-10-12): adaptive (symphonic, auto) had the opening vocal
"much quieter and boxed in" vs the preferred rules mix. Measured [M] at 0:15-0:50, vocal vs rest
broadband: preferred +5.2 dB, adaptive -1.0 dB; vocal air -2.5 dB. Causes, all three confirmed:
style-driven vocal cut (-2.1 dB), guitar compression keyed to whole-song range flattening the
arrangement build (lifted the intro guitar), and a -3 dB 12.5 kHz shelf on vocals. Fixes
(f65d673): lead vocal never cut below Suno placement + whole-song section guard (weakest 10 s
windows vs style p25, max +4 dB); no vocal top cuts unless measured sibilant; compression keyed
to within-phrase range (phrase_range_db). Adaptive v2 measured +5.3 dB at the opening, +0.9 dB
whole song (1.9 dB above preferred). A per-section ride was tried and rejected (spent its lift on
breath/bleed windows; opening +1.6 dB). The preferred mix's vocal openness also came from 5 dB of
vocal compression, which v2 lacks until references carry phrase_range_db (box re-run, stems now
kept as FLAC under _cache_<name>/stems/ so later re-measures stay on the Mac).

Ears check #2 (2026-10-12): v2 still worse than preferred - vocals quiet-ish and boxed. Measured
[M] the preferred mix vs v2 vs symphonic refs: preferred vocal is MORE compressed (within-phrase
range 1.7 vs 3.5 vs refs 6.7), wetter (dip 2.6 vs 4.7 vs 9.7), backing further under the lead
(8.5 vs 7.1 dB), while v2's vocal was actually LOUDER in 1-5 kHz. The user's vocal taste diverges
from the style references, so vocals now follow TASTE: star a version in Versions ->
backend/aimix_taste.py measures its vocal treatment from the render's mix-doc snapshot
(render_<jid>.json) into library/aimix/taste.json; adaptive closed loops then match backing level,
vocal compression, reverb send and (last) vocal level. Dream Of Me with 1 star: compress ~6 dB,
backing -1.4, no reverb needed after compression, level +2.6; vocal vs rest whole song -0.5 dB vs
preferred -1.0, air -17.2 vs -17.4 [M]. Manual console (channel strips + inspector) shipped
(086ab5b and later).

**Phase 2: Claude planner + mix panel.**
Analysis JSON + mix doc + schema + notes box -> Claude returns a mix-doc diff, shown before
render. Fader/EQ/send panel bound to the mix doc, re-render without AI. Gate: does the diff
move the mix the way the note asked (ears).

**Phase 3: Stem cleanup on the box (applied to Suno stems).**
Multi-model separation endpoint + installer. Per stem, optional: debleed (Bleed Suppressor /
Revive 2), lead/backing split, drum split into kick/snare/toms/cymbals, dereverb, denoise,
Apollo top-end restore. Conservation rule (3.3) keeps sums honest. Gate: A/B each model in
isolation on one stem, labelled files, one variable; keep/drop per model. Order vs Phase 2
can swap if bleed turns out to be the main audible problem in Phase 1.

**Phase 4: Reference match.**
ST-ITO-style search on the box over the same pedalboard params, target = a reference track
the user owns OR Suno's own full mix (as a balance target). Confirm AFx-Rep license first.

**Later / parked:** Mode A corrective polish of the mixdown via box re-separation (measured
viable, result #2, but not the brief); Mega 53 re-split; dasp-pytorch gradient fit for speed.

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
