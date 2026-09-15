# YUE2_PLAN.md - YuE2-3B as a third music engine, with its own tab

Status: **research done, box provisioning in progress, no render yet.** Created 2026-09-15.

Provenance tags: `[cited]` upstream docs/source, `[measured]` seen on our box or in our library,
`[hyp]` hypothesis, `[verify]` to test before relying on it.

## 1. What YuE2 is

Released 2026-09-10 by the M-A-P research group. `[cited]`

- 3B parameters, one AR+NAR mixture-of-transformers backbone. Pipeline: **plan** (an editable
  ABC score: melody, optionally chords) -> **semantic tokens** (AR, 25 tokens/s) -> **acoustic
  latents** (flow matching, the KSampler stage) -> **VAE decode** to 48 kHz stereo. `[cited]`
- Context window 24576 tokens. Style + lyrics + ABC prefix + music tokens must fit, so a 360 s
  song (9000 music tokens) leaves ~15k for prefix + score. `[cited: comfy/text_encoders/yue2.py]`
- Weights CC BY-NC 4.0 (non-commercial, and the licence text extends to the music). Code
  Apache 2.0. This is the one strategic difference from ACE-Step / Music 3 / Suno. `[cited]`
- No audio reference, no repaint/extend/inpaint, no instrumental-only mode, no official LoRA.
  A community NAR-only LoRA trainer exists (Saganaki22/ComfyUI-YuE2-Trainer, Windows OK); its
  "vocal timbre transfers" claim is unverified. `[cited]`

### Benchmark (vendor's own, machine judges, WildSongBench 192 prompts) `[cited: HF model card]`

| System | SongBench avg | Musicality | PER |
|---|---|---|---|
| YuE2 best-of-8 | 6.96 | 6.27 | 9.8% |
| Mureka 9 | 6.94 | 6.05 | 11.7% |
| Suno v5 | 6.87 | 5.99 | 8.1% |
| YuE2 single | 6.73 | 5.91 | 8.4% |
| Suno v6 | 6.56 | 5.66 | 7.6% |
| MiniMax Music 3 | 6.28 | 5.35 | 6.3% |
| ACE-Step 1.5 | 6.01 | 5.16 | 7.5% |

Single-shot sits above both our local engines and under Suno v5. Lyric intelligibility (PER) is
worse than Music 3. Newest open upstream issue (#178, 2026-09-14) is "voice still robotic", no
reply. Ears decide, per CLAP-scoring-unproven.

### Speed / VRAM (other people's hardware) `[cited]`

- RTX 4090: 3.6 min song in 71 s, peak 14.1 GiB.
- 3090 estimate: 2 to 2.5 min per song `[hyp]`. Music 3 measured ~110 s per 60 s of audio.
- bf16 checkpoint 7.26 GB; must serialise with the ACE engine on the 3090 (no-concurrent rule).

## 2. How it runs on the box

ComfyUI **core** support merged 2026-09-11 (Comfy-Org/ComfyUI PR #16250), two days after the
v0.35.0 tag, so it needs a **nightly (master) pull**. The box was on master `v0.34.0-21` on
2026-09-15. `[measured]` Update path = `update\update_comfyui.bat` (git pull master + pip
requirements; bumps comfy-kitchen/comfy-aimdo/frontend). Manager's API route for the same thing
was blocked by the permission classifier, so it is a user step.

Models (Comfy-Org/YuE2 repack, ungated): `[cited]`

| File | Folder | Size |
|---|---|---|
| yue2_3b_bf16.safetensors | models/checkpoints | 7.26 GB |
| sheetsage2_bf16.safetensors | models/audio_encoders | 1.29 GB (cover mode only) |

Installer written to the box: `ComfyUI_windows_portable\download_yue2.bat` (pure curl, resumable).
The int8 checkpoint was skipped: reported to produce noise on some systems. `[cited]`

### Native nodes `[cited: comfy_extras/nodes_yue2.py]`

- `CheckpointLoaderSimple` -> MODEL (acoustic DiT), CLIP (the AR planner + semantic stage), VAE.
- `YuE2GenerateABC(clip, style, lyrics, seed, mode full|melody, max_abc_tokens 8192,
  temperature 0.7, top_p 0.9, top_k 30, repetition_penalty 1.005, penalty_window 100)` -> abc string.
- `YuE2GenerateMusic(clip, style, lyrics, abc, seed, mode, max_duration 360 (up to 900),
  temperature 1.0, top_p 0.95, top_k 100, repetition_penalty 1.2)` -> CONDITIONING, seconds.
  **Empty abc = "off" mode** (direct generation, no plan).
- `EmptyYuE2LatentAudio(seconds, batch)`; seconds must be LINKED from the generator.
- `KSampler` steps 32, cfg 1.0, dpm_2, sgm_uniform, denoise 1; positive and negative both wired
  to the same conditioning (reference workflow). `VAEDecodeAudio` untiled on 24 GB.
- `AudioEncoderLoader(sheetsage2_bf16)` + `SheetSage2AudioToABC(audio_encoder, audio, mode
  melody|full)` -> abc string, for covers.
- cfg_scale inside the semantic stage: 1.0 (full/melody), 1.01 (off). `[cited]`

Runner for the trial: `.mvwork/yue2/run_yue2.sh <prompt.json> [seed] [off|full|melody]`,
`ABC=score.abc` to render an edited/transcribed score; `.mvwork/yue2/fetch_yue2.sh` pulls the
FLAC plus the .abc from the job history.

### Prompt grammar `[cited: SKILL.md, chunks.md guide, reference workflow]`

- Prompt is assembled as `<instruction>\n[Tags]\n<style>\n[Lyrics]\n<lyrics>`.
- Style: compact. Language, ONE genre, instruments, vocal character and timbre, tempo/BPM, mood.
  "More adjectives do not automatically create more control." No living-artist names.
  This is the opposite of Music 3's 4000-char schema and closer to ACE-Step's tag list, as prose.
- Lyrics: bare lowercase section tags on their own line (`[verse]`, `[chorus]`, `[bridge]`,
  `[intro]`, `[solo]`, `[outro]` seen in use). Lyrics decide length: the model sings everything
  given plus intro/breaks/outro. Hitting the token cap is a HARD mid-phrase cut, so max_duration
  needs headroom (we use 300 s for ~4 min songs).

## 3. Trial (gate before any UI work)

Two songs that already have Music 3 and Suno renders in the library:

- **The Ones Who Stayed** (Halestorm-style hard rock power ballad, A major 84 bpm)
  -> `.mvwork/yue2/the_ones_who_stayed_prompt.json`
- **Dream of Me** (symphonic metal, operatic soprano, E minor 140 bpm)
  -> `.mvwork/yue2/dream_of_me_prompt.json`

4 seeds each in `full` mode, reference sampler settings, one variable (the engine). Listen for:
vocal naturalness (the "robotic" complaint), the guitar wall, operatic register, lyric adherence,
section structure, early endings/truncation. Record wall time per song on the 3090.

Second experiment once the first passes: **cover mode** on one of our Suno masters
(SheetSage2 melody-only -> re-render in a different arrangement). Tells us whether the
re-arrangement lever is real.

## 4. The tab (user direction 2026-09-15: own tab, all modes, audio input for covers)

Engine module `backend/yue2.py` as a sibling of `music3.py` (graph builder + prompt assembly +
availability check), routes under `/api/yue2/*`, opt-in and additive (Music 3 and ACE untouched).

Modes the tab must expose, each a native-node path:

1. **Song** (text -> song). Style + sectioned lyrics. Plan mode `off | melody | full`.
   Seed, count (best-of-N), max_duration, advanced sampler knobs collapsed.
2. **Score first** (plan -> inspect -> render). Run `YuE2GenerateABC` alone (fast, no acoustic
   stage), show the ABC (text + rendered notation/piano roll, and an audible MIDI preview via
   our soundfont path), let the user edit it or pick from a plan batch of N seeds, THEN render.
   Same seed + edited ABC = the controlled re-render. This is the feature no other engine has.
3. **Cover** (audio -> score -> song). Upload/pick a library take -> `SheetSage2AudioToABC`
   with `melody` (recommended for style changes: accompaniment is free) or `full` (keeps the
   harmony) -> show/edit the transcription -> new style + lyrics -> render. Not voice cloning;
   transcription errors carry into the cover, hence the edit step.
4. **Agentic edit** (later). Hand the ABC + style + lyrics to the app LLM (claude_sub) with a
   plain-language note ("lift the last chorus a tone", "longer solo") and re-render the edited
   score. Upstream frames this as its headline workflow.

Library/DB: mode `yue2`, params carry style, lyrics, plan mode, abc (the score is an artifact
worth keeping with the take), seed, sampler. Section timeline after the fact via the existing
Whisper alignment, as with Music 3 (YuE2 has no per-section timing either).

Song-object mapping: our tags -> compact style string (a writer like `music3_writer.py` but
shrinking rather than expanding), our per-section blocks -> bare lowercase tags + lyrics;
per-section style cues do NOT go in the tags (same lesson as Music 3).

## 5. Measured on the box (2026-09-15) `[measured]`

- ComfyUI updated to master 36da3ff (23 commits past the YuE2 merge), reports 0.35.0; both
  node classes registered after restart; bf16 checkpoint 7.80 GB, SheetSage2 1.39 GB on disk.
- First render, The Ones Who Stayed seed 7 full mode, reference sampler: **224 s wall for
  265 s of audio**, first run so it includes model load. Music 3 would take ~485 s for the same
  length. VRAM seen during the planner stage 6.5 GB (acoustic stage not sampled).
- Output FLAC 48 kHz stereo, -12.7 LUFS integrated, LRA 9.3 LU, true peak 0.2 dBFS (hot; it
  will need the master stage's limiter like everything else).
- The ABC score carries our section order as comments (% intro, % verse, % pre-chorus,
  % chorus), two voices (Vocal + Ins), chord symbols. **Key and tempo in the prompt were not
  honoured:** asked A major 84 BPM, planned D# minor Q=86. One take, so not yet a rule.

### Ears (user, 2026-09-15, take 1 only)

"Not bad, better than Music 3. Guitars a little rough. An odd loud humming just before the end.
Otherwise followed the brief pretty well. Not Suno quality, but it has a place." -> the trial
passes the gate; the tab build is on. Watch across the other seeds for: rough guitar texture
(acoustic-stage limitation or seed variance?) and the end-of-song hum (the outro region past
the last lyric; see whether a shorter max_duration or an explicit `[outro]` line tames it).

## 6. Open questions `[verify]`

- 3090 wall time and peak VRAM for a 4 min full-mode render.
- Whether `off` mode is faster and how much quality it loses (upstream: plan modes score higher).
- Does the score edit survive a re-render in recognisable form (melody fidelity of the NAR stage)?
- Robotic-voice issue: does it show on operatic material?
- Licence: NC weights vs the YouTube channel. User decision.
