# Krea 2 Ultra — still-image engine (alt to Z-Image Turbo)

Krea 2 is Krea AI's own from-scratch **12 B dense DiT** image model (open weights, released
2026-06-23). It is NOT a FLUX/Qwen finetune — it's a bespoke DiT that *borrows components*:
the **Qwen-Image VAE** + a **Qwen3-VL-4B text encoder** (loaded via `CLIPLoader type "krea2"`).
Two checkpoints ship: **RAW** (undistilled, for finetuning/LoRA) and **Turbo** (fast generation —
what we use). Sources: krea.ai/blog/krea-2-technical-report, blog.comfy.org/p/krea-2-open-source,
docs.comfy.org/tutorials/image/krea/krea-2.

## How it's wired here
`backend/video.py` → `build_krea2_still(p)`, dispatched from `build_still(p)` when `engine=="krea2"`.
Faithful to the OFFICIAL ComfyUI template (`Comfy-Org/workflow_templates/image_krea2_turbo_t2i.json`)
and the AItrepreneur KREA2_ULTRA_WORKFLOW v2 the user is installing:

- UNETLoader `krea2_turbo_fp8_scaled.safetensors` (Ampere/3090 = fp8; 50-series = `_mxfp8`)
- CLIPLoader `qwen3vl_4b_fp8_scaled.safetensors`, **type `krea2`**
- VAELoader `qwen_image_vae.safetensors`
- CLIPTextEncode (positive) → **ConditioningZeroOut** as the negative (cfg 1 ⇒ no real negative)
- No ModelSamplingAuraFlow / shift node (unlike our Z-Image path)

All numbers are taken VERBATIM from the workflow (the `KREA2_*` constants in video.py). The builder
reproduces the workflow's TWO paths:

**1. Single pass — "TEXT TO IMAGE" (default).** Its Power Lora Loader is EMPTY, so **no LoRA**.
`EmptyLatentImage` + KSampler **er_sde / simple / 8 / cfg 1 / denoise 1**. (The official ComfyUI
template uses `euler`; the AItrepreneur workflow uses `er_sde` — we match the user's workflow.)

**2. Two pass — "TWO TIMES COMBO"** (`two_pass:true`, quality path, ~2× slower). The turbo LoRA
`krea2_turbo_lora_rank_64_bf16` @ **0.2** is enabled on **both** passes (Power Lora Loaders 174+180):
- pass 1: `EmptySD3LatentImage` + KSampler **er_sde / 8 / denoise 1**
- → VAEDecode → **ImageResize+** (lanczos, keep proportion, condition always) → VAEEncode
- pass 2: KSampler **euler / 4 / denoise 0.3** (refine), decoupled seed

`lora`/`lora_strength` add a trained character LoRA (model-only) on either path. `turbo_lora:true`
can also force the turbo LoRA onto the single pass.

## The two optional workflow nodes (now ported)
Both are CUSTOM nodes the workflow uses; ported as opt-in levers (off by default, controllable from
Settings → Engine flags; both need their node installed on the box):

- **Krea2T enhancer** — `ComfyUI-Krea2T-Enhancer` (capitan01R), a **MODEL→MODEL patch**, inputs
  `enabled / strength(0–2) / debug`; the workflow runs it **enabled, strength 1.0, debug false**. It
  scales Krea2's text-fusion *tap layers* (a baked profile with big gains on a few layers + a global
  multiplier) ⇒ **stronger prompt adherence + "unfilter"/quality-dilution bypass** (per the workflow
  note "remove the Safety Filter and improve prompting"). Wired UNET → [LoRAs] → **enhancer** →
  KSampler. App flag `still_krea2_enhancer` (request: `enhancer`, `enhancer_strength`). **Recommended
  ON** — it's what makes Krea2 worth using over Z-Image.
- **Seed variance** — `RBG_Smart_Seed_Variance` (RamonGuthrie), a **CONDITIONING→CONDITIONING** node
  that injects controlled noise into the text embedding. The workflow needs it because its KSampler
  seeds are **fixed** — variety comes from this node. OUR sampler seed already randomizes per call, so
  it's *extra* composition variety, not required. Workflow widget values (ported verbatim):
  preset `🌿 Balanced`, fine_tune 55, model_type `⚙️ Other`, fade `Instant`, noise `Beginning Steps`,
  protect `🚫 None`, direction `🚫 None`, shift_strength 129, schedule `constant`, cutoff_step 8,
  total_steps 20, cutoff_strength 0.0, seed randomized. Wired CLIPTextEncode → **seed-variance** →
  KSampler.positive. App flag `still_krea2_seed_variance` (request: `seed_variance`).

The img2img path (subgraph 116: euler/8/denoise 0.4) and the sharpen/film-grain nodes are NOT ported.

Scope: replaces ONLY the plain text→image still path (`/api/video/still` → genStill scene
backgrounds, character-identity stills, keyframe stills). The reference-driven `char_still`
(Qwen-Image-Edit, band composites) is unchanged.

## Switching it on
- Per-request: `POST /api/video/still {engine:"krea2", ...}`.
- Global default: **Settings → Engine flags → "Still image engine" = krea2** (writes
  `still_engine` to app_config.json; read per-request, so it applies WITHOUT a restart).
- Default stays `zimage` until Krea 2 is proven better (optional-additions rule).

## ⚠ Confirm on the box once the download finishes (models not down yet)
1. The exact UNET filename via `/object_info` UNETLoader list — `krea2_turbo_fp8_scaled.safetensors`
   is the official name, but AItrepreneur's download set may name the Ampere build differently.
   Fix the one constant `KREA2_UNET` if so.
2. `CLIPLoader` must offer **type `krea2`** (needs a recent ComfyUI; older builds won't have it).
3. The Krea2 prompt-enhancer / safety nodes (`ComfyUI-Krea2T-Enhancer`, `TextGenerate`) are NOT
   wired here — our graph is stock-node only. The Krea team's prompt-enhancer is optional polish;
   add later if wanted.

## Prompting Krea 2 (for genStill / our prompt builders)
Trained on short/medium/long **natural-language** prompts — **longer, detailed prose = best
quality**; no JSON/tag-soup needed. Practical rules:
- Write flowing sentences, ordered: subject → scene/background → shot type → camera/lens →
  style → lighting → medium → color palette.
- **Negatives are inert at cfg 1** — steer everything in the positive.
- Anti-"AI-plastic" (positive cues): name a real medium ("shot on 35mm / Kodak Portra 400"),
  "natural skin texture, visible pores, no retouching", candid/documentary framing, motivated
  soft lighting, shallow DoF + grain. AVOID "flawless / smooth / perfect / 8k / hyperdetailed /
  masterpiece" (they bring the gloss back). Keep quality-boosters minimal.
- Our existing person-free background prompts already read as prose, so they transfer directly;
  if Krea backgrounds look glossy, add medium + texture cues to genStill's scene phrasing.
