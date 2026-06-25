# Shot Studio + FFLF Lane — Implementation Plan

Status: **PLAN / not yet built.** No GPU runs fired. Provenance is marked throughout:
**[V]** = verified (from the workflow JSON, our source, or a read-only box check),
**[H]** = hypothesis / designed-correct but not yet run on our box.

This plan covers two linked workstreams agreed in the 2026-06-24 design pass:

1. **A new FFLF shot lane** based on foxydits' "LTX 2.3 FFLF Seed Hunter Multiroll" workflow
   (Civitai 2688482), which is **better than our LTXDirector path for 2-point shots** and is the
   route to **long / long-looking single takes** that dodge the long-clip RAM wall.
2. **A new "Shot Studio" page** — a universal per-shot deep editor — so seed-hunt, multiroll,
   extend/chain, anchors and retake all live off the cluttered main timeline.

Agreed decisions (2026-06-24):
- Shot Studio = **universal deep editor** (all shot types edit here; the timeline page becomes
  overview + sequencing).
- **Extend = both**: chain a continuous FFLF segment (primary) + a simple "lengthen this render"
  toggle (secondary).
- **This doc first**, then build.

Related: `docs/LTXDIRECTOR_PIPELINE_PLAN.md` (the existing relay-engine plan). The FFLF lane is a
deliberate move *off* LTXDirector for 2-point shots — see §2.

---

## 0. Validated this session + direction (2026-06-25)

**Proven on the box (corrects several earlier ASSUMPTIONS I'd asserted as facts):**
- **MSR identity + keyframe + lip-sync coexist in one graph** [V, user-confirmed "worked well"]. Added
  `first_keyframe`/`last_keyframe` to `build_ltx_msr` (endpoint `keyframe_first_id`/`keyframe_last_id`) —
  a stock `LTXVAddGuide` spliced after the MSR IC-LoRA guide. The old "MSR can't share a graph with
  keyframes" / "FFLF can't lip-sync" lines were unfounded. See [[project_flf-vs-msr]].
- **Two gotchas that make it "stop singing":** feed the **FULL SONG** (`isolate_vocal:false`), not an
  isolated/smeared vocal; and a last-frame keyframe must be a **singing pose (mic to mouth)** or it
  pulls the lips off the vocal as the shot converges.
- **Capability gain:** MSR + a keyframe gives a more **dynamic camera swing** *without* the MSR
  zoom-morph (the pinned endpoint tames it) — relaxes the old "keep MSR camera small/steady" rule.
- **Long continuous SINGING take via FFLF extend** [V mechanism]: extend an existing clip by feeding
  its **last 33 frames** (foxydits' value) as the FFLF first video-anchor; identity + motion carry
  from the tail, lip-sync from the continuation audio. Drive clip = the *result* clip you approved, not
  the input-source clip (got this wrong once). Audio for the extension is offset back by the overlap
  (33f) so the new content lands on the seam.
- **Seam needs work:** the regenerated overlap frames are *almost* pixel-identical → a slight flicker +
  an end-color pull at the join. Mitigations in play: **crossfade over the 33-frame overlap** at
  assembly; **lower `last_strength`** (~0.5) to stop the end yanking the color back.
- **Multiroll/seed-hunt is ESSENTIAL [user]:** going straight to single finishes wastes renders. The
  loop = 3 half-res hunt drafts → pick the cleanest seam/motion → finish (full res + lip-sync) → reroll
  if needed. NOTE: hunt drafts are **video-only** (no audio), so lip-sync is judged on the *finish*.

**UI direction (supersedes part of §6):**
- **Each segment in its OWN page view** (not just a panel) — the per-segment deep editor.
- **Revisit LTXDirector for the segment editor** specifically. We rejected Director for the *overall
  timeline* view, but its per-segment relay/timeline machinery may be the right fit for editing a
  single segment (prompt timeline, keyframes, retake regions) — re-evaluate, don't carry the blanket
  rejection. (FFLF stays stock-`LTXVAddGuide`; Director is for the editor scaffolding, not the 2-point
  shot graph.)

---

## 1. Why FFLF, and why a new page

- **FFLF teardown [V].** foxydits' v1.6 graph contains **zero `LTXDirector`/`LTXDirectorGuide`
  nodes**; it pins the first/last frame with **stock `LTXVAddGuide` ×2** (idx `0` and `-9`,
  per-end strength). The author states in his tutorial he no longer uses LTX Director. So this is a
  deliberate, field-preferred, stock-node FFLF design.
- **The long-shot path [V/H].** Each FF/LF end can be an image **or** a short *video* tail/head
  (`VHS_LoadVideo` with `frame_load_cap` + `skip_first_frames`). The author: *"sandwich a gen in
  between two videos."* Feeding the rendered **tail of clip A** as the next segment's first frame and
  a curated still/clip-head as its last frame lets short (~4s) segments **read as one continuous
  take** — sidestepping the long-clip RAM wall (which is driven by frame count, [V] per memory).
  Won't drift like SVI/temporal-tiling [H]: both ends of every segment are pinned to known pixels,
  so a segment can only interpolate between fixed anchors; identity/colour re-anchor at every seam.
- **The clutter problem [V].** `web/src/MVStudio.tsx` `Inspector` (≈ lines 591–930) already crams
  shot settings + the segments/keyframes timeline + retake into one panel. Multiroll adds a *growing
  strip of take variants per shot* — unworkable inline. Hence Shot Studio.

---

## 2. The FFLF recipe to replicate (faithful port)

Lesson [V, memory]: replicate the author's graph; our only sanctioned deviation is the fp8
`UNETLoader` (which he *already uses*, so even that matches). All node values below are **[V]** from
`ltx23FFLFSeedHunter_v162STAGEUPDATE.json`.

**Models (all already on the box / same family we run):**
- UNET `ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors` (pre-distilled — no
  separate distill LoRA needed).
- DualCLIP `gemma_3_12B_it_fp8` + `ltx-2.3_text_projection_bf16`, type `ltxv`.
- Video VAE + audio VAE; spatial upscaler `ltx-2.3-spatial-upscaler-x2-1.1`.
- SageAttention ships **disabled** (matches our "no sage for LTX" rule).

**Stage 1 — draft (×3 for seed hunt):**
- `EmptyLTXVLatentVideo` at **half target res** (`a/2`; res must satisfy `(target/2)/32` = whole
  number — e.g. 1280→640→20 ✓; 1080 silently becomes 1024).
- 2× `LTXVAddGuide`: Start (idx 0), End (idx -9); strengths from sliders, **default 0.7, best
  0.6–0.9**.
- Sampler: `SamplerCustomAdvanced` euler, `CFGGuider` cfg 1,
  `ManualSigmas = 1.0,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875,0.0` (**8 steps, denoise
  1.0**).
- Three identical instances seeded `a`, `a+1`, `a+2` from one `easy seed` button; previewed with
  **TAE** (`taeltx2_3.safetensors`) for cheap eyeballing.

**Selection:** a slider picks which of the 3 latents → an `ImpactSwitch` routes it to stage 2.

**Stage 2 — finish / multiroll:**
- `LTXVLatentUpsampler` ×2 → re-add Start/End guides → `LTXVConcatAVLatent`.
- Sampler euler, cfg 1, `ManualSigmas = 0.85,0.725,0.4219,0.0` (**3 steps, partial denoise from
  0.85**).
- **Its own separate seed** (the multiroll seed) — reroll for motion/detail variants without
  re-running the hunt.

**Prompt block:** split negatives — *video negatives* (`scene cut, scene transition, no movement,
still frame, frames…`) and *audio negatives* (`music, score, instruments, orchestra…`, to suppress
LTX's generated music since we mux our own). `LTX2_NAG` scale **50** (vs our 11), alpha 0.25, tau 2.5.

**Defaults:** 97 frames @ 24 fps ≈ 4s, target 1920×1024. Optional `OmniNFT-RL` LoRA via Power Lora
Loader (author's only LoRA, strength 2). Decode has a tiled fallback (`VAEDecodeTiled 512/128`).

> We port the **compute** graph only. The mxSlider/muter/bypasser/Anything-Everywhere nodes are
> manual-UI conveniences — our builder sets their values directly, so we don't need them installed.

---

## 3. Identity strategy (FFLF can't use MSR in-graph)

MSR identity and keyframe guides can't share a graph [V, memory]. But identity in FFLF is two levers:

- **Lever A — the anchors (primary).** FFLF identity *is* the FF/LF stills. We already generate
  on-model character stills with **`build_qwen_char_still`** (`backend/video.py:184`,
  Qwen-Image-Edit-2511, 1–3 refs, **no training**) [V]. Default identity path = generate anchors with
  it, then FFLF interpolates. **No character LoRA required for the baseline.**
- **Lever B — model-side reinforcement (optional).** A character LoRA on the LTX transformer via the
  Power Lora Loader slot, to keep the *middle* of long/fast segments on-model. Train **image LoRA
  before video LoRA** (cheaper; fixes the anchors, which is what FFLF consumes). An LTX-2.3 video LoRA
  is a separate training pipeline we have **not** stood up (our LoRA routine is ACE-Step audio) — defer
  until reference-still anchors prove insufficient.

**Reconciling MSR:** MSR/VACE can still be used **one step upstream** to produce on-model anchor
frames (render → grab a clean frame), then feed FFLF. "No MSR in the FFLF graph" ≠ "no MSR at all."

**Chain drift caveat [H].** Across a long chain, each segment's *rendered tail* becomes the next
anchor, so tiny per-segment drift can accumulate. Bound it by **re-anchoring** to a fresh curated
Qwen-char-still every K segments (accept a tiny seam to reset identity); a character LoRA raises K.

**Lane split (recommended):** singing/identity-critical → MSR single-take (≤ proven ~20s); B-roll /
camera-move / long takes → FFLF with reference-still anchors. FFLF + masked-vocal lip-sync (no MSR,
identity from anchors) is also viable — his graph already has the SolidMask+SetLatentNoiseMask path.

### 3a. Image anchors can't do entrances — use video anchors / MSR (2026-06-24)

**Image-only anchors are insufficient for on/off-screen motion [V, by construction].** FFLF pins
frame 0 and frame -9 to two stills and interpolates. If a character is off-screen at the start, the
FF anchor contains no character, so the model must *materialise* one partway through with identity
pinned only at the final frame (one view) — it fades/morphs them in rather than walking them in from
the edge, and identity drifts. A static endpoint cannot encode "moving in from off-screen." Stills
also give one angle / no parallax.

**Fix — video anchors (already in foxydits' design).** The FF/LF guide can be a short *clip*
(`VHS_LoadVideo`, N frames) that already shows the entrance motion; LTX continues the trajectory and
carries identity *through* the frames. This is the primary reason video anchors matter — not just
chaining.

**Source the entrance clip from MSR.** MSR is built for this: identity from references, **motion
prompt-driven, no keyframe anchor** ([memory] — "the walk comes from the prompt"). Pattern:
MSR renders the entrance (off-screen → on-screen) → **its tail becomes the FFLF video anchor** for the
held/continuous part that follows. The seam is clean because FFLF's FF guide *is* the MSR tail.

**Sharpened lane assignment:**
- Entrances / exits / free identity-driven motion → **MSR**.
- Locked A→B camera moves, B-roll, continuous-take chaining → **FFLF** (use *video* anchors, not
  stills, at any boundary that has motion; stills only where both ends are genuinely static, e.g. a
  slow push-in on a held pose).
- A single long take can **mix lanes**: MSR entrance → FFLF segments chained off its tail.

**Build consequence:** Shot Studio's Extend/chain must support **mixed-lane chains (MSR→FFLF)**, and
anchors must accept **video sources from prior takes**, not only library stills. See §6/§10.

---

## 4. Backend: `build_ltx_fflf` + endpoints

New builder in `backend/video.py`, beside `build_ltx_flf` / `build_ltx_keyframe`, on **stock
`LTXVAddGuide`** (not LTXDirector). Mirrors §2.

```
build_ltx_fflf(p, first_src, last_src, vocal_ref=None)
  first_src / last_src: {kind:"image"|"video", name, frames?, skip?}   # video end ⇒ chaining
  p: prompt, negative?, video_neg?, audio_neg?, width, height, frames(97), fps(24),
     first_strength(0.7), last_strength(0.7),
     stage1_seed, stage2_seed,            # DECOUPLED: hunt seed vs multiroll seed
     base_steps(8), refine_steps(3), refine_denoise(0.85),
     nag_scale(50), char_lora?(""), omni_lora?(off),
     half_res(True), draft(False)         # draft=True ⇒ stage-1 only, half res, TAE preview
  -> graph, resolved
```

Endpoint `POST /api/video/ltx_fflf` with a `mode` field (mirror the retake/keyframe endpoint shape
in `backend/app.py`):
- `mode:"hunt"` — render **3 half-res stage-1 drafts** (seeds `a/a+1/a+2`), return 3 job-ids /
  preview frames for a contact strip. Gates cheap before the full render (fits our
  "gate-before-render" habit [memory]).
- `mode:"finish"` — take the chosen draft latent + `stage2_seed` → upscale-refine → one clip.
  Reroll `stage2_seed` ⇒ another variant (multiroll).
- `mode:"extend"` — chain: build a new segment whose `first_src` = `{kind:"video", name:<prev
  clip>, skip:<len-N>, frames:N}` (the prev tail) and `last_src` = the next anchor.

Plus a thin orchestrator (can live in `app.py` or a helper) `fflf_chain(segments)` that walks a list,
feeding each rendered tail as the next FF guide, and concatenates the outputs.

Constants to reuse: `LTX_UNET_FP8`, `LTX_CLIP1/2`, `LTX_VAE_VIDEO/AUDIO`, `LTX_SPATIAL_UPSCALER`,
`_seed`, `_ltx_frames`. New constant: `LTX_TAE_PREVIEW` (optional).

api.ts: `videoLtxFflf: (p) => jpost("/api/video/ltx_fflf", p)`.

---

## 5. Data model changes (`web/src/mvmodel.ts`)

- `RenderMode = "msr" | "i2v" | "s2v" | "keyframe" | "fflf"`.
- **Takes with seeds (multiroll).** Today: `clipId`/`upscaledId` = selected take, `clipVariants?:
  string[]` = all job-ids [V]. Extend to carry seeds so multiroll variants are reproducible/labelled:
  ```
  type Take = { id: string; clipId: string; stage1Seed?: number; stage2Seed?: number;
                isDraft?: boolean; label?: string };
  // on Block:
  takes?: Take[];            // every hunt draft + finished variant
  selectedTakeId?: string;   // which take is the block's clip
  ```
  Keep `clipVariants` working (back-compat); `selectedTakeId`→`clipId` stays the timeline's source.
- **FFLF anchors on `Block` (or a per-segment piece):**
  ```
  type Anchor = { kind: "image" | "video"; id: string; frames?: number; skip?: number };
  firstAnchor?: Anchor; lastAnchor?: Anchor;
  firstStrength?: number; lastStrength?: number;   // 0.6–0.9
  charLora?: string;
  ```
- **Chain pieces.** Reuse `Seg[]` (it already has `keyframeStillId`/`isEndFrame`) extended with
  per-piece anchors/seeds, OR model a chain as linked Blocks. Decision in §9.
- `hydrateBlock` must default the new fields (back-compat with persisted projects).

---

## 6. Shot Studio page (universal deep editor)

New tab `case "segment"` in `App.tsx` Controls switch (navigation is `mode`/`goTo`, no router [V]).
New file `web/src/ShotStudio.tsx`.

**Navigation.** MVStudio shot-row gets an **Open** button → `goTo("segment")` with the block
selected (shared selection state / handoff). A **← Timeline** button returns. MVStudio slims to:
script generation, the readable shot list, sequencing, assemble. The heavy `Inspector` internals
(segments/keyframes, retake, and new FFLF/multiroll) **move into ShotStudio**.

**Layout (one shot in focus):**
- **Header:** Shot N, renderMode selector (msr/keyframe/fflf/i2v), timing (from timeline), ← Timeline.
- **Left — anchors & prompt:** mode-specific. FFLF: First/Last anchor (library still **or** video
  tail/head), strengths, prompt, split video/audio negatives, optional char-LoRA. Each anchor has a
  **"Generate with Qwen-char-still"** inline action (identity path, §3). MSR/keyframe: the existing
  controls, relocated.
- **Center — preview + takes strip:** current selected take + a gallery of every multiroll variant;
  click to select (→ becomes the block's clip). This is where multiroll lives.
- **Right — action rail:** Seed-hunt (3 drafts → contact strip → pick) · Finish/Multiroll (reroll
  stage-2 → new variant) · **Extend** (chain segment [primary] + "lengthen this render" toggle
  [secondary]) · Retake region (moved from Inspector).
- **Chain view** (FFLF multi-piece shots): the pieces of this shot, each with its own
  anchors/seed, assembled into one continuous clip; per-piece re-hunt/multiroll.

**Migration note:** "universal" means MSR/keyframe deep editing moves here too. Stage it (§8) so the
timeline keeps working while the Inspector is hollowed out.

---

## 7. Seed-hunt / multiroll UX

- **Seed-hunt:** one "New seeds" action renders 3 half-res drafts → contact strip (TAE previews if
  installed, else fast tiled decode). Pick one. Cheap; nothing full-res rendered yet.
- **Multiroll:** "Reroll" bumps `stage2_seed` only and finishes again → appends a labelled variant to
  the takes strip. The chosen stage-1 draft stays fixed. Variants are reproducible (seeds stored on
  the Take).
- All variants live in ShotStudio's strip — the timeline only ever shows the *selected* take.

---

## 8. Phasing

1. **Backend FFLF builder + endpoint** (hunt/finish), still/still anchors, no chaining. Validate the
   graph faithfully vs the JSON (py_compile + a single manual ComfyUI sanity run *with go-ahead*).
2. **Data model + api.ts** (RenderMode "fflf", Take[], anchors). Back-compat hydrate.
3. **ShotStudio page (FFLF first):** navigation, anchors UI, seed-hunt strip, multiroll, retake moved
   over. Wire to step-1 endpoints.
4. **Extend:** chain orchestrator + "lengthen" toggle; chain view.
5. **Universalize:** migrate MSR/keyframe deep editing from the Inspector into ShotStudio; hollow the
   Inspector; slim MVStudio to overview/sequencing.
6. **Identity polish:** inline Qwen-char-still anchor generation; optional char-LoRA slot.

Each phase: typecheck + "look in the app"; no preview-driving theater [memory].

---

## 9. Dependencies / install list

**Box compute nodes — all present [V]** (`/object_info`): `LTXVAddGuide`, `LTXVLatentUpsampler`,
`LatentUpscaleModelLoader`, `easy seed`, `SimpleCalculatorKJ`, `ImageResizeKJv2`, `LTXVPreprocess`,
`VHS_LoadVideo/VideoCombine`, `ImpactSwitch`, `LTX2SamplingPreviewOverride`, rgthree Any Switch +
Power Lora Loader. Models match what we run.

**Optional downloads (quality/speed):**
- `taeltx2_3.safetensors` (Kijai) — fast hunt previews.
- `LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors` (Kijai) — motion/fidelity.
- **INT8 spike (3090-relevant):** author claims **1.2–1.5× faster than fp8** via
  `ltx-2.3-22b-distilled-1.1-int8-ConvRot.safetensors` + `ComfyUI-INT8-Fast` loader. Separate spike.

**Not needed (UI-only, builder sets values directly):** mxToolkit `mxSlider`, rgthree Fast Groups
Muter/Bypasser, `Anything Everywhere`. (Only needed if opening foxydits' workflow in the ComfyUI UI,
which also needs "Nodes 2.0" disabled — irrelevant to our headless graphs.)

---

## 10. Open questions / risks

- **Chain unit (§5):** model a chain as `Seg[]` pieces inside one Block, or as linked Blocks on the
  timeline? Leaning `Seg[]`-pieces-in-a-Block (one timeline shot = one continuous take). Each piece
  must carry its own **lane** (msr/fflf) and **anchor source** (library still or a *prior take's*
  video tail/head), because a continuous take can mix lanes (§3a: MSR entrance → FFLF hold). **Decide
  before phase 4.**
- **Seam quality [H]:** untested on our footage. Phase-1 manual validation (one image→image + one
  video-tail→still, eyeball the seam on real Selene footage) should happen *before* phase 4 invests in
  chaining. Needs explicit go-ahead.
- **Half-res hunt fidelity:** does the half-res draft predict the finished look well enough to pick a
  seed? If not, raise hunt res or reduce the draft count.
- **NAG 50 vs our 11:** adopt his hotter NAG for FFLF? Test, don't assume.
- **Universal migration risk:** moving MSR/keyframe editing out of the Inspector touches working
  flows — stage carefully (phase 5 last), keep the timeline functional throughout.

---

## 11. Per-segment page design + LTXDirector-as-editor (2026-06-25)

This supersedes the panel sketch in §6: each segment gets its **own full-page view**, not an inline
panel. Reached from the timeline by **Open** on a shot row → `goTo("segment")` with that block
selected; **← Timeline** returns. The timeline page slims to overview + sequencing.

### 11.1 LTXDirector revisit — verdict: ADOPT it for the editor scaffolding (not the FFLF graph) [V from box I/O]

We rejected LTXDirector for the *overall timeline*, but its node I/O makes it the right backend for a
*single-segment* editor. `LTXDirector` exposes (verified via `/object_info`):
- **prompt timeline** — `global_prompt` + `local_prompts` (`|`-sep) + `segment_lengths` + `epsilon`
  (boundary softness).
- **keyframes** — `timeline_data` JSON (segments: type/start/length/imageFile/isEndFrame) +
  `guide_strength` CSV.
- **duration / framing** — start/end frame+second, `duration_*`, `frame_rate`, `display_mode`
  (seconds|frames), `custom_width/height`, `img_compression`.
- **audio** — `use_custom_audio` / `inpaint_audio` / `override_audio` (+ `audio_vae`, `combined_audio` out).
- `LTXDirectorGuide` adds **`retake_mode`** (re-render a region), **`ic_lora_name`/`ic_lora_strength`**
  (an identity/control IC-LoRA on the guide), and **`scale_by`** (the 2-stage refine).

So a single segment maps cleanly onto one Director graph: a prompt timeline + absolute keyframes +
retake regions + audio handling, optionally an IC-LoRA. **The editor's controls are essentially a UI
over Director's fields** — we already plumb most of them in `build_ltx_keyframe`/`build_ltx_msr`.

**Boundary:** LTXDirector backs **MSR / keyframe / timeline** segments. **FFLF (2-point + extend)
stays stock `LTXVAddGuide`** — proven better for 2-point and the only path we've validated for the
video-tail extend/chain. The editor hosts *both* backends; the segment's `renderMode` picks which
controls show.

### 11.2 Page layout (one segment in focus)

- **Header:** Shot N, `renderMode` selector (msr | keyframe | fflf | i2v), timing (from the timeline),
  ← Timeline.
- **Left — source & prompt (method-specific):**
  - *Director segments (msr/keyframe):* the **prompt timeline** (global + per-segment local prompts at
    segment_lengths, epsilon slider), keyframe list (still + start/length/isEndFrame/guide_strength),
    MSR refs + background, optional IC-LoRA / char-LoRA slot.
  - *FFLF:* First / Last anchors — each a **library still OR a clip tail/head** (video anchor =
    `frames`/`skip`, default **33**), strengths (0.6–0.9), prompt, split video/audio negatives.
  - **Lip-sync (any method):** vocal source + window, **full-song toggle (`isolate_vocal:false` default
    — decided)**; a singing keyframe pose if a last keyframe is used.
- **Center — preview + takes strip:** current take + every **multiroll** variant (each labelled with
  its seeds), click to select → becomes the block's clip. Variants live here, never on the timeline.
- **Right — action rail:**
  - **Seed-hunt** → 3 half-res drafts → contact strip → pick (drafts are **video-only**; lip-sync is
    judged on finish).
  - **Finish / Multiroll** → finish the picked draft at full res + lip-sync; reroll stage-2 seed → new
    variant.
  - **Extend** → append a continuous segment driven by **this clip's last 33 frames** (FFLF video
    anchor); supports the **crossfade-over-overlap** assembly (smooths the seam) + a `last_strength`
    control (≤0.5 to stop end-color pull). "Lengthen this render" stays as the simple secondary.
  - **Retake region** → Director `retake_mode` (or the existing retake lane) for a glitchy slice.
- **Chain view:** the ordered pieces of a continuous take (each with its own lane/anchor/seeds),
  assembled with crossfades; per-piece re-hunt/multiroll.

### 11.3 What's already built vs new

- **Built & proven:** `build_ltx_fflf` (hunt/finish, image+video anchors, lip-sync), `build_ltx_msr`
  (+ keyframe splice), `build_ltx_keyframe` (Director 2-stage), retake, `videoFreeModels` + auto-free,
  the seed-hunt→finish→crossfade-assemble loop (manual via curl this session).
- **New for the page:** the `ShotStudio.tsx` page + nav; the takes/multiroll strip with stored seeds;
  the extend **chain orchestrator** + **crossfade assemble** as a backend endpoint (we did it by hand
  with ffmpeg — needs an `/api/mv/assemble_chain` or extend-to-`Seg[]` flow); the audio-window math
  (continuation start, 33-frame overlap offset) done server-side so the UI never hand-computes it
  (note: got the overlap offset + a 15.0-vs-15.375 audio length wrong by hand this session — bake it in).
- **Data model:** `RenderMode += "fflf"`; `Take[]` with `stage1Seed`/`stage2Seed`; FFLF anchors;
  chain `Seg[]` pieces carrying lane + anchor source (§5/§10).

### 11.4 Phasing (revised)

1. `ShotStudio.tsx` page + nav + takes strip, wired to existing endpoints (FFLF hunt/finish first).
2. Server-side **chain orchestrator + crossfade assemble** (encapsulate the manual ffmpeg + audio math).
3. Director-backed segment controls (prompt timeline, keyframes, retake) — the LTXDirector editor.
4. Universalize: migrate MSR/keyframe editing off the Inspector into the page (§8 phase 5).

