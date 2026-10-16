# YUE2_REVIEW.md - a fresh look at how we use YuE2 (2026-10-16)

Why: a month of use, 25 library takes, never much good out of it, while others report good
results. This reviews everything (our graph, prompting, UI, the box install) against upstream
docs and community practice as of 2026-10-16. Training is out of scope until base use is good
(section 7 notes what exists).

Provenance: `[verified]` checked by us today, `[O]` official m-a-p docs/source/model card,
`[C]` ComfyUI core source/templates, `[Cm]` community report (anecdote unless it says measured),
`[hyp]` our inference, untested.

## 1. Things that are actually wrong on our side

### 1a. "Text guidance" (cfg_scale) has never been applied `[verified]`

- The box's `comfy_extras/nodes_yue2.py` is the launch-week file (mtime 2026-09-15). Its
  `YuE2GenerateMusic` has NO `cfg_scale` input (`/object_info` shows none, required or optional).
  CFG control was added upstream in Comfy PR #16373 on 2026-09-17 `[Cm/C]`.
- ComfyUI drops inputs a node does not declare, without error. Our backend sends `cfg_scale`
  when set (backend/yue2.py:250), so every "cfg 1.2" take ran at the default (1.0 = no guidance).
- Proof: library takes "Wonder A cfg1.0 seed7" and "Wonder B cfg1.2 seed7" null against each
  other to -91 dB (digital silence). They are the same audio; the 2026-09-23 A/B compared a
  render with itself. 11 library takes are labelled cfg 1.2 and are not.

### 1b. The box runs the first-cut YuE2 implementation `[verified]` + `[C]`

Not updated since 2026-09-15, so it is missing, besides the cfg input:
- SheetSage2 cover timing drift after ~50-90 s: fixed PR #16395 (2026-09-18).
- SheetSage2 sharps-only key/chord spelling (A# for Bb): fixed PR #16569 (2026-09-26).
- AR sampler rewrite / speed-up PR #16626 (2026-09-28).
- Caveat `[Cm]`: one user reports same seed + workflow sounding "several times worse" after a
  ComfyUI update (YuE issue #210, 2026-09-30, no root cause). So updating is necessary (it is
  the only way to get CFG) but is itself a variable: judge it by ear, not by assumption.

### 1c. Our style lines are the opposite of what the model was trained on `[verified]` + `[O]/[Cm]`

- Ours (library): 71 to 994 characters, median ~350. The longest ones narrate the arrangement
  section by section ("intro on a dark filtered synth pulse..., verses stripped to...,
  bridge dropping to piano...") and stack 2-4 "no X" negatives. Lint only warns above 600.
- Upstream examples: ~130-200 characters of comma-separated descriptors, e.g.
  `English, warm female vocal, contemporary pop, 96 BPM, piano, rounded electric bass,
  restrained drums, clear diction` `[O]`. Training captions were short attribute descriptions
  written by Qwen3-Omni (tech report App. C) `[O]`; the model never saw section-by-section
  arrangement prose, so it has no way to follow it `[hyp]`.
- Community, measured on 3090s (Ruach Studio, HF discussion #29) `[Cm]`: short SD-style tags beat
  prose; order = language + genre with era -> defining instruments + lead voice -> mood/production
  -> BPM last. "orchestral X" builds a whole orchestra and loses X. Some instruments are simply
  unknown (shofar, oboe, gong); they publish a 200-instrument cheat sheet with verdicts.
- Key in the style line: 0 effect in 18 tries `[Cm]`, and our own 8-take batch agrees (asked
  A major, got D#m every time) `[verified]`. Yet `song_to_style` appends the key
  (backend/yue2.py:120). Key is set by transposing the score, which works every time `[Cm]`.
- Per-section voice tags (`[Chorus - Female]`) do not work; the voice follows the planned
  melody's register `[O: issue #216]`.

### 1d. Every take was a single unranked take in `full` mode `[verified]` + `[O]`

- All 25 takes: plan mode `full`. `melody` and `off` never tried on a new song.
- Upstream's own numbers are cherry-picked: "standard YuE2" = the lower-PER of 2 takes; the
  headline = best-of-8 ranked by musicality, then prompt adherence, then PER `[O]`. Their expert
  listening test: best-of-8 ties Suno v5 and loses clearly to Suno v6 (31.6% vs 59.3%) `[O]`.
  A single take sits below both published figures by design.
- Our tab renders N takes but ranks nothing and has no A/B player.

### 1e. Smaller divergences `[verified]` vs `[O]`

| Ours | Upstream | Note |
|---|---|---|
| max_seconds default 300 | 360 (semantic max_tokens 9000 at 25 Hz) | Dream of Me hit 300 |
| max_abc_tokens 8192 | 4096 | Comfy default; runaway plans `[Cm]` |
| KSampler dpm_2 / sgm_uniform | midpoint ODE, uniform grid, 32 steps | both 2nd order, not identical |
| VAE: whichever is baked in the repack, dtype unknown (Comfy log shows fp16 `[C]`) | YuE2-Vae (listening) in FP32, tiled 1024 core / 16 halo | legacy VAE = benchmark decoder |
| Instrumental = "instrumental" + empty tags | Move Vocal notes into Ins in the score, empty lyrics `[O]` | still not guaranteed |
| Cover: plan mode not synced; chords kept | SheetSage2 melody, strip chords, `cot=melody` `[O]` | our "Canon" cover used full + chords |
| Agentic edit: rewrites the ABC only | edit ABC + revise style/lyrics, manifest, reviewer, note-level compare `[O]` | |
| Docs: weights non-commercial, music too | "Personal users, content creators, and musicians: free to use YuE2 and monetize generated outputs" `[O: README License]` | our YUE2_PLAN.md and tab note are stale |
| parse_header bar count | sums both voices + Z rests (143 "bars" for a 96-bar score) | display bug |

## 2. What people getting good results do `[Cm]` unless marked

1. **Guidance on.** cfg 1.4-1.6 steady, 2.0 more dynamics, 2.4 the stable max, ~2.6+ vocals
   break and tokens cut (Ruach, measured by ear + numbers on 3090s). Official: 1.2 "for stronger
   text guidance", "not a guaranteed quality improvement" `[O]`. Any cfg != 1.0 roughly doubles
   the AR stage time (a second, unconditional branch). One rock user credits CFG with fixing rock.
2. **Batch plans, not songs, then batch takes of the chosen plan, then pick.** Plans are seconds
   each; seed shapes form as much as lyrics. Reseed before rewording.
3. **Short tag-style style line** in the order above, vocal early, BPM last.
4. **Lyric length 180-290 words** for 3.5-4.5 min. Under ~150 the planner pads and can garble;
   over ~270 it can plan past 5 min (becausereasons LoRA cards). Tags on their own lines; only
   sung words in the lyrics; choruses written out in full; instrumental sections as empty tags.
   Safest tag names are the training labels (SongFormer: intro, verse, pre-chorus, chorus,
   bridge, inst, outro) `[O/hyp]`.
5. **Check the plan before rendering.** Good scores run ~95-125 lines; a section repeating or
   padding to the cap = "ghostly"/garbled take. Reseed the plan.
6. **Repetition penalty 1.0-1.05** on hook-heavy songs: 1.2 smears intentional repeats (two
   independent users). Keep 1.2 if sections run away.
7. **bf16 checkpoint, not the int8 template default** (we already do this).
8. **De-buzz.** A metallic buzz sits on the VAE's 1920-sample frame seam on every decoder
   (4-12 kHz envelope shows a 20% periodic pattern vs ~5% elsewhere). Dividing the highs by the
   measured per-frame profile at ~80% strength removes it (Ruach, measured). Plausibly part of
   the "rough guitars" / "AI sand" complaint `[hyp]`.
9. **For rock/metal with a belting female voice, they use LoRAs.** The base model is reported
   weak exactly there: rock drifts to pop/country (HF #10), fast metal vocals soften, the female
   voice is samey (HF #21, issue #220), belted raspy rock-soul is "not in the training data".
   The good rock results people post come with GRVL (raspy belting female rock-soul) or
   industrial-rock LoRAs. Operatic: zero community reports found.

Honest read: the genres we write (Halestorm-style hard rock, symphonic/power metal, operatic
female lead) are the base model's weakest area. Fixing sections 1 and 2 should lift every take;
it may not get those genres to "good" without a LoRA. That is the bridge to training.

## 3. Frontends and node packs

- **ComfyUI core** (what we use): fine once updated; no plan batch, no retake, LoRA loader
  only applies one of the two LoRA layouts (silently skips m-a-p layout files).
- **pytraveler/YuE2-ComfyUI** (84 stars): Plan Batch + Select Plan, score editor, **Edit Track**
  (retake chosen bars / cut / extend while keeping the rest sample-exact: the repaint YuE2
  otherwise lacks), both LoRA layouts, MIDI input. The most useful addition for us `[Cm]`.
- **YuE2-Studio** (timoncool, ~754 stars, Windows, yue2.cpp BF16/Q8): every sampling knob,
  built-in LoRA training (LoKr + Prodigy).
- Official runtime on Windows: NAR attention falls back to the slow MATH kernel (~6x, +7 GB)
  `[O: issue #209]`; Linux is the supported platform. Core ComfyUI is the right host for us.

## 4. Proposed redesign of the tab

Flow, top to bottom, one decision per step:

1. **Brief -> style + lyrics.** The style writer emits the tag order above, 120-220 chars, no
   arrangement narration, at most one negative, no key. A style meter (chars + tag count) and a
   lyric meter (words vs the 180-290 band, predicted length) replace the 600-char lint.
2. **Plans.** Batch 4-8 plans by default (cheap). Each card: key/BPM/sections/line count, a
   runaway flag (section repeated, line count out of band), MIDI preview. Transpose here to set
   the key. Pick one.
3. **Takes.** Render N takes (default 4) of the chosen plan. Settings collapsed into presets:
   "Upstream default" (cfg 1.0, rep 1.2) and "Guided" (cfg 1.5, rep 1.05) plus custom. A guard
   that refuses to send a setting the box's node does not declare (the failure in 1a).
4. **Pick.** Auto-rank takes by lyric intelligibility (PER via our existing Whisper/lyricalign
   path, which is what upstream's "standard" pick uses), shown as a hint, never as the verdict;
   an A/B player with loudness-matched switching. Ears decide.
5. **Fix.** Retake bars (needs pytraveler's Edit Track) or edit the score and re-render.
6. **Finish.** De-buzz + the existing master stage.

Cover mode: melody transcription -> strip chords -> plan mode forced to `melody`. Instrumental
mode: move Vocal notes to Ins in the score.

## 5. Proposed test ladder (one variable at a time, one song, ears judge)

Gate 0, box update (user step: `update\update_comfyui.bat`, then restart). Then confirm
`cfg_scale` appears in `/object_info/YuE2GenerateMusic` before anything else.

Fix the code bits first (cfg guard, drop key from style, max 360, rep penalty exposed), then on
one song with one fixed plan, 4 takes per condition:

| Step | Variable | Conditions |
|---|---|---|
| A | style line | our long prose line vs a short tag line (same words' intent) |
| B | guidance | cfg 1.0 vs 1.5 vs 2.0 (winner of A) |
| C | plan mode | full vs melody vs off |
| D | repetition penalty | 1.2 vs 1.05 |
| E | de-buzz | off vs on (post only, no render) |

~4 min per take with cfg on, so ~40 takes is ~2.5-3 h of box time, fired step by step with a
listen between steps.

## 6. Open items `[verify]`

- Which VAE the Comfy-Org repack bundles (YuE2-Vae or legacy) and at what dtype it decodes on
  the box; `--fp32-vae` is the A/B if it is fp16.
- Whether the post-09-28 AR sampler changes our sound (better or worse) for the same prompt.
- Does a short style line move the genre on rock/metal, or is a LoRA the only lever there.

## 7. Training (later, after base use is good) `[O]/[Cm]`

- No official training code; the audio->semantic tokenizer is unreleased `[O]`.
- Community made it possible: Mothersuperior real-audio tokenizer (v4) + NAR LoRA; FL YuE2 pack
  (training nodes, dataset prep, LoRA trainer; same pack family we chose for Music 3);
  Starnodes ComfyUI-YuE2-Trainer (NAR/timbre only); HOT-Step trainer inside YuE2-Studio.
- Recipes seen: Mothersuperior AR LoRA r64, 50/50 with YuE2-generated songs, ~1500 steps,
  14-18 GB; becausereasons GRVL r32, LR 5e-5, planner LR x0.4, planner KL 0.2, cot full with
  50% score dropout, 36-51 songs, stop ~20 passes (voice goes thin after).
- Ready-made LoRAs worth an audition first: becausereasons GRVL (raspy belting female rock),
  monsterovich industrial-rock (use NAR v2; AR ~0.5, NAR 0.3-1.0), steps-from-hell (epic
  orchestral/choir), ntc-ai sliders (Metal, Female, Male). Needs a loader that handles both
  layouts (pytraveler pack).

## Sources

- Repo + docs: github.com/multimodal-art-projection/YuE (docs/generation.md, covers.md,
  editing.md, benchmarks.md, skills/yue2-music/**, src/yue2/protocol.py); local copies in
  `.mvwork/yue2src/` and `.mvwork/yue2_upstream/`.
- Model card + discussions: huggingface.co/m-a-p/YuE2-3B (discussions #1, #7, #8, #10, #13,
  #21, #29, #30).
- ComfyUI: Comfy-Org/ComfyUI PRs #16250, #16373, #16395, #16569, #16626; docs.comfy.org
  tutorials/audio/yue2; workflow_templates audio_yue2_text2music.json.
- Community: github.com/pytraveler/YuE2-ComfyUI, github.com/timoncool/YuE2-Studio,
  huggingface.co/becausereasons/yue2-grvl-raspy-rock-soul,
  huggingface.co/monsterovich/yue2-industrial-rock-lora, Mothersuperior realaudio tokenizer v4.
- Not reachable: Reddit and the YuE Discord (seen only second-hand).

## 8. Ladder results

### Step A: style line length (2026-10-16) `[verified]`

Box on ComfyUI 0.41.0 (updated today; cfg_scale now declared). Bad Kind of Beautiful, the
library's fixed score (plan seed 1184589206, 4040 chars), same lyrics (310 words), seeds
1184589207-210, cfg default, full mode, max 360. Only the style line differed:
long = the 994-char library line, short = `English, modern hard rock, husky powerful female vocal
with a gritty belt, drop-tuned heavy guitars, dark synth bass, big tight rock drums, dark and
seductive, 118 BPM` (167 chars). Takes 302-312 s, none capped; short takes 0.7-1.6 dB louder in
3 of 4 pairs (a loudness bias to allow for).

User's ears: **short slightly better, not night and day.** Short lines carry forward.

### Side question: descriptive tags in the lyrics (`[Chorus - powerful belt]`) (2026-10-16)

Research only, nothing rendered. Training lyrics = SongFormer functional labels + ASR words, so
descriptors were never seen `[O: tech report App. C]`; the planner's section labels are a closed
vocabulary and can disagree with the lyric tags `[O/Cm]`. The only YuE2 tests of per-section
descriptors are negative: YuE issue #216 (voice tags, several methods, no effect) and
pytraveler's README (voice tags tried on 12 songs, none worked) `[Cm]`. Empty tags give
instrumental passages ~50% of the time; the reliable lever is the score (Vocal notes -> Ins).
Upstream and ComfyUI docs: only sung words + section tags in the lyrics. No YuE2 A/B of a
delivery descriptor exists; the "big effect" belief is most likely carried over from Suno /
ACE-Step. NOTE: our normalize_lyrics strips anything after `-`, `,` or `:` inside brackets, so
the app cannot currently send a descriptor even to test one.

### Step B: text guidance (2026-10-16) `[verified]`

Short style, same score + seeds as A, cfg_scale 1.0 (= step A short takes) vs 1.5 vs 2.0. Each
guided take differs from its 1.0 partner at music level (null residual about -13 dB), so the input
reached the model this time. User's ears: **2.0 better.** Now a per-render field defaulting to
2.0. Render wall on ComfyUI 0.41: ~125-155 s per ~310 s song; guidance adds ~5%, not the 2x
claimed in community posts.

### Tag test: descriptors in lyric tags (2026-10-16) - awaiting ears

4 takes with descriptors sent as written (opt-in toggle), same score / seeds / style / cfg 2.0 as
the step B cfg 2.0 takes. The takes differ from their bare-tag partners (residual -12.7 to -14.7
dB), but per pytraveler any text change reshuffles the performance, so only ears can tell whether
the change FOLLOWS the descriptors.

## 9. Guitar solos and per-section style: the levers (research 2026-10-16)

The model has ONE style line per song; nothing in training conditions text per section `[O]`.
audio.cpp's authors: "YuE2 continues what it has already sung far more than it follows its style
prompt, so changing the prompt part-way through a song does little" `[C]`.

Guitar solos, ranked:
1. **Write the solo into the score's Ins voice** (official recipe `[O]`): section `% interlude`,
   Vocal = chord symbols over rests, Ins = a monophonic lead line in guitar register; style line
   names the solo ("melodic electric guitar solo, the singer rests during the guitar solo"). The
   official sax-solo demo does exactly this. Works on core nodes today via the abc input; needs
   a solo writer/editor in our app. Ins-voice adherence unmeasured.
2. Let the planner write it: solo request in the style prose + plan batch + auto-pick plans
   whose solo bars have dense Ins notes and Vocal rests (official J-rock gallery, n=1).
3. pytraveler Edit Track `notes` retake of just the solo bars (needs the pack; it only edits
   songs it rendered itself).

Per-section style, ranked:
1. **The score**: vocal register per section (voice follows the melody's register, #216), vocal
   density, Ins activity, chords. Core nodes today.
2. Semantic-history splice (audio.cpp recipe; needs semantic_prefix, not in core ComfyUI).
3. Section-contrast phrases in the one style line ("intimate breathy verses, belted choruses");
   upstream's own demo captions do this; unmeasured on the base model. Cheap A/B.
4. GRVL v2 LoRA (captions describe verses and choruses separately; Halestorm-adjacent).
5. pytraveler per-region guidance/temperature/notes retakes (text stays fixed).
Not levers: lyric-tag descriptors (pending our test), dual guidance (global only).
