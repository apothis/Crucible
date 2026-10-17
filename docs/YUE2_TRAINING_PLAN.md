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
