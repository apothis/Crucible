# TARJA_OPERATIC_PLAN.md - LoRA to give ACE-Step an operatic soprano voice

Status: **PLAN. Not started.** Dataset (Tarja "My Winter Storm") being gathered.
Created: 2026-05-31.

Provenance tags: `[cited]` upstream, `[ours]` measured/established on our box,
`[hyp]` hypothesis, `[verify]` to test.

## 1. The goal + why this dataset

ACE-Step xl-sft gives a strong *contemporary* female rock/pop voice but **not a
classically trained operatic soprano** (Tarja's bel canto / aria delivery). Reasons:
- xl-sft drops speaker/timbre control in favor of text-prompt control. `[cited]`
- Opera/bel canto is rare in its training corpus, so "operatic soprano" has a weak
  anchor and it defaults to the common case (the "Evanescence not Nightwish" result). `[ours]` (heard 2026-05-31)
- Our existing `crucible_nightwish` LoRA can't fix it: full-song LoRA learns style
  more than isolated timbre, AND Tarja-era Nightwish has Marco Hietala's male vocals
  throughout, so it learned a Tarja+Marco *blend*. `[ours]`/`[cited]`

**Tarja's solo album "My Winter Storm" is the right dataset because it is
single-singer, operatic-forward, and has NO Marco dilution.** This is the cleanest
shot at teaching the operatic voice specifically. `[hyp]`

## 2. Dataset construction (the important part)

- Source: Tarja - My Winter Storm (her solo material). Single lead singer.
- **Curate vocal-forward, operatic-leaning tracks.** Prefer the symphonic/operatic
  songs where her trained soprano is prominent.
- **Exclude:** pure instrumentals / intros / interludes (no lead vocal to learn),
  and any heavy guest-duet tracks (we want her voice isolated, not blended - the
  exact mistake the Nightwish set made). `[ours]` lesson
- Target ~12-18 curated tracks (subgenre-LoRA range). Quality of curation matters
  more than count - keep it operatic and solo.
- Dataset name: `crucible_tarja` (separate from crucible_nightwish so both survive).

## 3. Captioning - bake the operatic trigger in

Beyond our normal enrichment (AcoustID/MusicBrainz/Last.fm/CLAP + LM prose), make
sure the captions consistently carry the operatic descriptors so they become a
learned trigger we can invoke at inference: e.g. "classically trained operatic
soprano, bel canto, operatic vibrato, coloratura, classical crossover, Tarja
Turunen". `[hyp]` (trigger-word idea from the 2026-05-31 research; modest but cheap)
- Use the autolabel + merge flow, then hand-edit the merged captions to ensure the
  operatic terms are present on every track before save/preprocess.

## 4. Training config (use our hard-won defaults)

- Plain **LoKr**, `lokr_weight_decompose=False`, **lr 0.01**, `val_split=0.1`,
  `timestep_sampling_mode=discrete` (default), `training_seed=42`,
  `gradient_checkpointing=True`. Per [[engine-lokr-defaults]] + the no-regression default.
- Epochs: start **150** with best-checkpoint + val tracking. Our BiB run showed
  FINAL (ep150) > BEST (ep62) on *vocals* specifically, so if by-ear says the voice
  is still improving, do a follow-up **250** run (single variable). `[ours]`
- One eval loop = train -> engine restart -> generate -> score -> repeat, never
  overlapping CLAP + engine. [[no-concurrent-clap-engine]]

## 5. Pipeline (the established, working flow)

1. **OS-level fresh engine boot** before training. [[engine-fresh-boot-for-lora]]
2. Upload curated tracks (Mac enrich + box upload helper).
3. Autolabel + merge -> hand-edit captions to ensure operatic trigger terms.
4. Save -> preprocess -> train (config above) -> export.
5. Load via the new **LoRA picker** (it now names the adapter on each take).

## 6. Evaluation

- **By ear is the judge** (val_loss is misleading at our scale [[dont-propagate-guesses]]).
  The question: does the vocal move toward *operatic technique* (trained vibrato,
  aria delivery), not just "female symphonic"?
- A/B on a fixed song + seed: base vs `crucible_tarja` at 0.4 / 0.6 / 0.8, with the
  aggressive operatic prompt (see "Dream of Me" tags). Compare also against
  `crucible_nightwish` to confirm Tarja-solo is cleaner than the Marco-blended set.
- Watch the strength tradeoff: 0.5-0.6 nudges, 0.8 tends to muddy (BiB lesson). `[ours]`

## 7. Honest ceiling

Even with a clean operatic dataset, ACE-Step may only partially deliver classical
technique - operatic vocals are a known weak spot for music-gen models, and xl-sft
gave up fine voice control by design. Tarja-solo is the **best shot, not a guarantee**. `[cited]`/`[hyp]`

## 8. Cross-refs

- If we later want cleaner inference off the engine: LoKr won't load in ComfyUI;
  we'd need a plain-LoRA retrain. See `COMFY_LORA_PLAN.md`.
- Pipeline + prior LoRA findings: `METAL_LORA_PLAN.md`.
