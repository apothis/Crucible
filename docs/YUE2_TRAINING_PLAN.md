# YUE2_TRAINING_PLAN.md - band LoRAs for YuE2 (research 2026-10-17)

Goal (as for every artist LoRA): capture a band's COMPLETE sound - the singer's real voice and
the band/production - starting where the base model is weakest (female-fronted hard rock,
power/symphonic metal). Nothing below is installed or run yet.

Provenance: `[verified]` checked by us, `[src]` read in the trainer's source, `[auth]` measured
and published by the author, `[claim]` stated but unverified, `[hyp]` our inference.

## 1. What "training via Comfy" is

- ComfyUI CORE cannot train YuE2. Its generic `TrainLoraNode` (on the box) trains the diffusion
  half only, from latents + conditioning; YuE2's conditioning must be the real song's semantic
  (music) tokens, and core has no audio -> semantic encoder. `[verified: box /object_info]`
- What made training possible (from 2026-09-13): a community audio -> semantic tokenizer
  (Mothersuperior realaudio heads v4..v9, on MERT-v2) and trainers built on it. Ostris AI Toolkit
  added YuE2 on 2026-09-14 and saves LoRAs in ComfyUI's own key layout, so they load in a stock
  graph; that plus several node packs is the "training via Comfy" news. `[src]`

| Option | Trains | Loads in core LoraLoader | Evidence | Verdict |
|---|---|---|---|---|
| **Ostris AI Toolkit** (own app, not a node) | planner (AR) + decoder (NAR) jointly, one LoRA | yes (clip strength = planner, model strength = decoder) `[src]` | only trainer with a published same-songs A/B where the LoRA learned the singers' voices, not just the genre `[claim, becausereasons]` | **recommended** |
| speedyrulz/ComfyUI-YuE2-Trainer | the two halves separately, in ComfyUI | yes | 3 stars, no voice results; Metallica decoder LoRA barely moved the song `[auth]` | in-Comfy fallback |
| FS_Audio Suite | both | yes | codec-offset bug (#2), Windows GQA OOM (#4); learned genre not voice `[claim]` | avoid for voice |
| FL YuE2 | AR (+ acoustic companion) | no (own runtime/layout) | its own benchmark: no adapter beat base on held-out `[auth]` | avoid |
| Starnodes / Saganaki22 | decoder only, text-only conditioning | yes | "LoRA has no effect" issues; -0.004 CLAP `[auth]` | avoid |

Planner (AR) = composition, phrasing, diction, structure. Decoder (NAR) = most of the timbre,
production and likeness. Train both together.

## 2. AI Toolkit's data format `[src]`

- Whole songs (mp3/wav/flac..., resampled to 48 k stereo); trim to <= 320 s. The decoder trains
  on a random 60 s window, the planner from the song start up to `ar_max_tokens` (4000).
- One same-name `.txt` per song: the style line, then `[Lyrics]`, then sectioned lyrics.
  ```
  English, modern power metal, powerful high belting female vocal with rasp, ..., 140 BPM
  [Lyrics]
  [Verse]
  ...
  [Chorus]
  ...
  ```
- ABC scores: made automatically at cache time by SheetSage2 (`cot: full`, `abc_dropout 0.5`
  keeps score and no-score generation working). Semantic tokens: automatic (MERT-v2 layer 20 +
  Mothersuperior head, v4 default, v9 selectable). No forced alignment, no Demucs needed.
- Built-in captioner presets (Qwen2.5-Omni-7B / Qwen3-Omni) order tags genre -> mood -> vocal ->
  delivery -> instruments/production -> tempo -> era, no artist names.

## 3. What our ACE-Step pipeline gives us (audit 2026-10-17)

Reusable as-is: the audio (box `lora_data\crucible_*\data\` + Mac `Downloads/This-Is-*`, all
320 kbps MP3), artist/title lookup, LRCLIB/lyrics.ovh lyrics fetch (online only, per the
house rule), the box file API.

Reusable with adaptation:
- **Section tags**: fetched lyrics are plain stanzas. `backend/yue2.py` `section_lyrics()`
  already sections them (LLM + heuristic fallback); needs a batch driver and a filter for the
  source's singer-credit tags (`[Tobias Sammet:]`).
- **Style line**: ACE captions are noisy Last.fm tag soup + 550-980 chars of arrangement prose,
  the opposite of a YuE2 style line. Useful as INPUT to an LLM rewrite into the house format
  (language, genre, vocal, contrast clause, instruments, mood, BPM last), with a fixed
  per-artist voice vocabulary and a hand check of vocal gender.
- **BPM**: librosa whole-song values; halve doubled ballads.

Not applicable: the ACE engine stages (scan, LM autolabel, preprocess .pt tensors, LoKr train).

Datasets on hand: Avantasia 50, Battle Beast 40, Beast in Black 40, Metallica 30, Nightwish
(Tarja era) 21, Nightwish 8, mixed power metal 35 (box); plus unbuilt Mac folders: Alestorm,
Dio, Patty Gurdy, The Pretty Reckless, Tarja solo, more Nightwish/Power Metal.
Practice says leave out duets and guest leads: Avantasia (rotating guest singers) is a poor
voice-LoRA set; single-singer sets fit best.

## 4. Recipe that worked for others `[claim]`

rank 32 / alpha 32, adamw8bit, LR 5e-5, `ar_lr_multiplier` 0.4-0.6, `ar_kl_weight` 0.2,
`cot: full`, `abc_dropout` 0.5, `ar_max_tokens` 4000, window 1500, save every 50 steps. Keepers at
steps 400-650 and <= ~20 passes per song (past that: thin/tinny, burnt). 20-50 songs. Add a few
15-45 s whole excerpts of distinctive passages (full songs get cut by the memory cap, so the
planner never sees endings). "Anchor trick": 2 YuE2-generated songs captioned with the exact
prompt you will use. Inference: decoder strength 1.0 (0.8 if the top thins), planner 1.0 slow /
0.5 fast songs. Overfit signs: planner loss -> 0, runaway ABC (hundreds of intro/verse bars, no
ending), thin top, same riff everywhere. Judge a fixed-seed checkpoint ladder by ear.

## 5. Plan

1. **Dataset builder (Mac, no GPU)**: band folder -> per song: trimmed audio + `.txt` (house
   style line + `[Lyrics]` + sectioned lyrics), reusing lyrics_fetch + section_lyrics + an LLM
   style rewrite; written to a review folder for a hand check before upload.
2. **Install AI Toolkit on the box (user step)**: own folder + venv (Windows installer), its UI
   on its own port. Driver already CUDA 13 capable `[verified: box torch 2.11.0+cu130]`.
   Local patches as complete files: per-step `torch.cuda.empty_cache()` (PR #1069), K/V
   `repeat_interleave` instead of `enable_gqa=True` (Windows math-kernel fallback `[hyp]` for this
   trainer), `ar_cuda_graphs: false`. Models: point at the existing
   `models/checkpoints/yue2_3b_bf16.safetensors`; tokenizer head, MERT-v2 and SheetSage2
   auto-download.
3. **Train** one band, checkpoint every 50 steps; the 3090 to itself (no ComfyUI/ACE jobs).
4. **Audition in our tab**: add an opt-in core `LoraLoader` (planner + decoder strengths) to the
   YuE2 graph; fixed-seed ladder across checkpoints, ears decide.

Open user decisions: trainer (AI Toolkit vs in-Comfy speedyrulz), first band, caption source
(LLM rewrite of the ACE prose vs AI Toolkit's audio captioner on the box).

## 6. Battle Beast run log (2026-10-17)

Setup that is now built and remote-controlled from the Mac (nothing done at the box after one
helper restart): AI Toolkit at `E:\AI\MusicGen\ai-toolkit` (installed via the helper, torch
2.13+cu130, 11.5 GB); helper endpoints /aitk/install|run|status|log|stop + /helper/restart;
patches (complete files in tools/aitk/patches): K/V repeat instead of enable_gqa, per-step
empty_cache on Windows, @torch.no_grad on merge_nar_lora (PR #1068 - the v9 run died without it).

Dataset (tools/yue2_dataset.py): 40 Battle Beast songs, all LRCLIB lyrics, sectioned by the YuE2
sectioner + hand rules (Let It Roar chorus split out; sung "solo/interlude" -> Bridge; repeated
tags merged). Captions = fixed user-approved "English, modern power metal with 80s heavy metal
and synth influences, powerful raspy female rock belt with a soaring high range" + per-song parts.
Qwen2.5-Omni-7B captioner: run 1 copied the prompt's example phrases into ~15 captions and got
language/genre/voice wrong (German, melodic death metal, operatic, male guttural); run 2
(example-free) was correct but vague. Final per-song parts = the app LLM condensing the ACE 4B
LM's existing audio descriptions (section-by-section, instruments) + Qwen's tempo feel; 4
"operatic" leaks and one tempo contradiction fixed by hand. No BPM numbers (librosa unreliable).

Run 1 (v4 tokenizer, rank 32, LR 5e-5, planner x0.5 + KL 0.2, 800 steps): latent cache 13 min,
training 1 h 7 min at ~5 s/step, VRAM 14-22 GB (user's reading). ComfyUI's core LoraLoader applies
it (log: 112 patches on both halves, no key warnings; takes null to -11.3 dB vs no-LoRA).
Skyborn test (original song shaped like Wings of Light, one plan, seed 7101): **450 and 600 OK,
strong hints of Noora but the song doesn't push the voice where she shows; 800 good in much of it
with a little voice breakup.**

Run 2 (v9 tokenizer + merged nar_lora_joint_v9, otherwise identical): started 18:16. Generation
needs nar_lora_joint_v9_comfyui (loras/yue2/companions/) under the band LoRA - the tab's
"Tokenizer adapter" slot.

Not yet tried (AI Toolkit has them): LoKr, DoRA, dropout/rank/module dropout, rank 64, Prodigy,
EMA, warmup, regularisation songs (is_reg / anchor trick), output preservation, do_separation
(author: "doesn't work as expected"). One change per run on the Skyborn score + seed.

### Early end at 300 s + the score-length guard (2026-10-17)

Several LoRA takes (v4 600, v9 600/800, the F#m score tests) stopped at exactly 300.0 s (7500 music
tokens) mid-chorus on a 328.5 s score: the model itself emitted end-of-music (diagnostic: the music
stage's own seconds output = 300.0; no budget warning; no downstream cap). Base model and v9 450
played the full score. Writing (planner) strength 0 or 0.5 removed the cut, but 0.5 "loses a lot of
the Battle Beast and Noora sound" (user) - the character lives in the planner half (writing 0 vs base
differs only at -20 dB = timbre). Hypothesis (unverified): the planner LoRA was trained on the first
4000 music tokens only (ar_max_tokens), so past 160 s it drifts more as training goes on and falls
back to a "songs end at 5 min" habit. Fix in place: custom node YuE2GenerateMusicGuarded
(tools/comfy_nodes/ComfyUI-MusicGen-YuE2; raises core's own end-token floor) + tab option
"score-length guard" (min = score length - 8 s; score_seconds() within ~5 s of real renders).
Verified: v9-600 writing 1.0 + guard = 328.0 s with a natural fade; ~20% slower (160 s vs 133 s).
Root-cause fix to try in a retrain: planner on whole songs (ar_max_tokens up from 4000; VRAM?).

User notes: v9 450 full length and "pretty good"; voice not raspy enough for most of it - her rasp
comes from vocal EFFORT, not pitch (so the register-raising score tests aim at the wrong thing);
verse 1 should stay soft, pressure builds after. Effort tests (style wording / guidance 2.4 /
sound 1.2, all full length) rendered, awaiting ears.

Run 3 (2026-10-18, v9, planner on WHOLE songs: ar_max_tokens 0 = the trainer's own default; we had
copied 4000 from the community recipe, which calls it a memory cap): fits on the 3090, ~7.4 s/step vs
4-5 s (no spill to system RAM), 800 steps ~1 h 40. Tests the 300 s ending and whether the planner,
now seeing later choruses, learns her pushed voice. Awaiting the ladder.
Result (2026-10-18 02:17, measured): run 3's step 800 with NO guard plays the whole 328 s Skyborn
score and nulls to -91 dB against the same take with the guard on: the guard never had to act. The
old 4000-token LoRAs ended at exactly 300.0 s on the same score/seed. So the 300 s ending came from
the capped planner window (hypothesis now supported by the A/B), and whole-song planner training is
the fix. Ladder 450/600/800 (+guard) all full length; voice verdict = user's ears.

Round trip (tools/aitk/musicgen/yue2_roundtrip.py, helper /aitk/script): real song -> tokenizer head
-> YuE2 decoder, no planning; first 3 min of 3 BB + 2 BiB songs, v9 tokens through the base decoder,
the v9-600 LoRA and the run-3 LoRA, plus v4/v5/v8 heads on one song per band. Library titles "RT ...",
originals "RT original recording - ...". Sanity: Wings of Light round trip vs original chroma
correlation 0.84 (same song, not noise).

Why YuE2 runs are ~1 h and ACE LoKr ~10-12 h: same ~5 s/step, but ACE ran ~150 epochs (~7500
steps on 50 songs) vs 20 passes (800 steps) here. Consequence (hypothesis): the decoder sees one 60 s
window per step, so each part of a song reaches it only ~4-5 times per run, while the planner (an LM)
is what burns past ~20 passes. Queued next (written, not started): run 4 = run 3 with the planner
slowed to x0.2 over 2000 steps (same total planner LR as 800 at x0.5) -> 2.5x the decoder exposure;
checkpoint every 100 (yue2_battlebeast_v9_wholesong_plr0.2_2000_lora, ~4 h at 7.4 s/step).

Overnight 2026-10-18 (user asleep, full go-ahead):
- Beast in Black dataset: all 40 songs, LRCLIB lyrics sectioned by the app LLM, fixed lines DRAFTED BY
  CLAUDE (user to confirm): "English, 80s-inspired power metal with synthwave and disco-pop influences,
  soaring high male tenor belt with piercing falsetto screams and a gritty edge". Per-song parts: the ACE
  dataset had tag lists only, so ACE-Step's own captioner (acestep-captioner, describe-only script, fp8,
  18 s/song) -> app LLM condense; 4 genre-word leaks fixed by hand. Several parts carry vocal-effort
  words ("screamed choruses", "gritty strained vocals") per song.
- BiB run: v9, whole-song planner, 800 steps, ~7.5 s/step, 1 h 50. Test song "Midnight Machine"
  (original, project of that name); the LoRA writes its own score. 600 / 800 (two seeds) full length;
  450 ended at 300.0 s on a 307.5 s own score, and still did with the guard (floor = score - 8 s =
  299.5 s; the model ended 0.5 s after it). Own-score takes otherwise end 2-4 s before the score
  length, so a 5 s margin would be safe - not changed.
- Run 4 (BB, planner x0.2 over 2000 steps, whole song): 3 h 48; ladder 1200/1600/2000 on Skyborn with
  the guard, all full length.
- Captioner note: AI Toolkit's AceStepCaptioner job always runs the lyric transcriber too; both 7B on
  the 3090 hit OOM retries (6-9 min/song). Use tools/aitk/musicgen/acestep_describe.py.
