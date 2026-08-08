# MiniMax H3 video backbone - integration plan

Status: **Phase 0 BUILT + GATE PASSED (2026-08-02).** `build_h3_t2v` + `/api/video/h3_t2v` are live;
four real renders measured on the box. Phases 1-5 still unbuilt. Written 2026-08-02 from a full read of the user's
reference workflow `MINIMAX_H3_ULTRA_WORKFLOW.json` (Aitrepreneur), the ComfyUI node source on the
box, and read-only `/object_info` + `/fs/read` probes.

Provenance tags: **[V]** = verified (node source, /object_info, or the reference workflow itself),
**[H]** = hypothesis / not yet run on our box. Per [[no-fabricated-causes]] nothing here is asserted
as measured unless tagged [V].

Goal: replace the LTX-2.3 / LTXDirector video path with MiniMax H3 as the music-video backbone,
copying the reference workflow's structure rather than inventing new graphs.

---

## 0a. MEASURED ON THE BOX (2026-08-02) - all [V]

Four renders, 1280x736 / 124 frames (5.17s) / seed-pinned, via `/api/video/h3_t2v`.

| Recipe | sampler / scheduler / steps | LoRA | Time |
|---|---|---|---|
| base | `res_multistep` / `simple` / 20 | - | **553s** |
| base + Spectrum | as above + SigmaShift(12,3) + SpectrumApply | - | **342s** (-38%) |
| **turbo** (author's ULTRA+TURBO) | `euler` / `beta` / **8** | 4-step @ **1.0** | **215s** (-61%) |

- **Memory: FITS.** Peak VRAM **18.7GB / 25.8GB**, ~7GB headroom, no spill, ~24s/it. Sampling begins
  ~51s in (the 27.1GB text encoder loads faster than feared). The 54GB-of-weights worry in section 6
  did NOT materialise - ComfyUI's sequential load handles it. The author's "usable on 8GB" claim is
  consistent with what we saw.
- **Native audio works**: every render came out h264 + AAC with real content (not silence).
- **CAMERA MOVES WORK - the headline win.** A `truck right ... past a foreground pillar` and an
  `arc shot to the left with large amplitude at slow speed` both executed with genuine parallax and
  NO warping, morphing or stepped framing. These are precisely the moves that failed on LTX
  ([[project_ltx-camera-lora]]), so the 3-move restriction in our script writer can be lifted.
- **USER VERDICT (2026-08-02): "turbo is fine for a seed hunt, but base is needed for final."**
  So: turbo = draft/hunt recipe, base = finish recipe. Both look good; Spectrum was slightly
  disliked vs off and is NOT in the default path.
- **Spectrum changes the picture, it is not free speed.** At a pinned seed it produced a materially
  DIFFERENT scene (different cathedral geometry/contrast), so it must be judged on quality. The turbo
  LoRA by contrast largely PRESERVED the seed's composition at the same resolution - which is what
  makes it viable as a hunt recipe.

### The hunt/finish design (from muse_minimax_h3_director_scout_v1.json)

That workflow implements the seed hunt with `MuseMinimaxDirector` (runs at **0.4MP**, emits
`candidate_1..4` images+audio plus a `compiled_prompt` STRING) -> each candidate through
`RTXVideoSuperResolution` (x3 ULTRA) -> `MuseMinimaxRefine` (runs at **0.9MP**, takes ALL four
candidates + a pick index + `ref_images` + the compiled prompt, at what looks like ~0.2 denoise).
So the finish is a LOW-DENOISE PASS CONDITIONED ON THE UPSCALED PICK - not a fresh generation at the
same seed. RTX VSR stands in for the latent upsampler H3 does not have. [V from node IO; the 0.2
semantics are [H], inferred from widget position.]

**BLOCKER: that pack is NOT installed.** `MuseMinimaxDirector`, `MuseMinimaxRefine`,
`RTXVideoSuperResolution`, `SolAttnPatch`, `MiniMaxH3MemoryEfficientSageAttentionPatch`,
`LayerUtility: PurgeVRAM`, `iToolsPreviewText` all absent from `/object_info` (only `EasyCache`,
`ModelPreviewOverrideKJ`, `ImageFromBatch` are present, and the `taeh3` preview AE is missing).
Source: github.com/muse-collective-26/MiniMaxH3-Director.

Two routes, DECISION OPEN:
- **A - install the Director pack.** Hands us the six-section prompt compilation (our Phase 3), the
  >15s auto-chunking, hybrid continuation, and the hunt/refine, as one node driven by a timeline JSON
  widget with a verifiable `compiled_prompt` output.
- **B - stock-only hunt.** N low-res turbo drafts via the server-side fan-out we already use in
  `/api/video/ltx_fflf` (`mode:"hunt"` -> `{base_seed, drafts[]}`), then finish the pick on the base
  recipe. Needs no installs, but the finish cannot be conditioned on the pick the way Refine does
  unless we feed the draft back as `ref_video_0` (REF2VA model, continuation/editing role).

### ANSWERED 2026-08-02: a pinned seed does NOT survive a resolution change [V]

Measured with matched-timestamp frames -> 128x72 grayscale -> Pearson r (`.mvwork/compose_corr.py`),
calibrated against pairs the user had already judged by eye:

| Pair (all seed-pinned, same prompt) | mean r | reading |
|---|---|---|
| same shot, RECIPE changed (base vs turbo, both 0.9MP) | **+0.946** | same shot, minor render differences |
| different scenes (Spectrum off vs on, both 0.9MP) | +0.765 | visibly different scene |
| **same recipe, TIER changed (turbo 0.4MP vs 0.9MP)** | **+0.461** | worse than "different scene" |
| unrelated scenes (moor vs cathedral) - floor | -0.098 | |

The 0.4 vs 0.9 pair also DEGRADES over the clip (+0.69 at t=0 -> +0.23 at t=5.1s): same concept
(standing stone, moor, storm) and the arc works in both, but different stone shape, different horizon
height, different foreground rocks/puddle, and the two arcs rotate at different rates. A low-res
draft predicts the CONCEPT, not the shot.
=> A cheap-tier hunt CANNOT predict its finish. Ruled out. Recipe transfer, by contrast, is strong,
which is what makes the chosen design work.
0.4MP timing for reference: 105s (vs 215s turbo / 553s base at 0.9MP).

### CHOSEN DESIGN (user decision 2026-08-02): hunt at FINISH resolution on the TURBO recipe

BUILT in `/api/video/h3_t2v`:
- `mode:"hunt"` -> N drafts (default 3, max 6) at `base_seed, +1, +2...`, forced onto the turbo
  recipe, at **the same width/height as the finish**. Returns `{mode, base_seed, drafts:[...]}`,
  mirroring `/api/video/ltx_fflf`'s server-side fan-out so the frontend pattern is unchanged.
- `mode:"finish"` -> one render of the picked seed on the BASE recipe, same resolution.
Cost: ~215s per draft + ~553s for the finish (3 drafts + finish ~= 20 min per approved shot).
Route A (install the Director pack) remains the alternative if we want the cheap 0.4MP tier back -
its Refine conditions the finish on the upscaled pick, so it does not depend on seed transfer at all.

### Corrections to earlier assumptions in this doc

- The turbo LoRA is NOT a drop-in: the author's ULTRA+TURBO workflow changes FOUR settings together
  (LoRA@1.0 + `euler` + `beta` + 8 steps). Diffed 2026-08-02. `video.py` applies them as one atomic
  `H3_TURBO` recipe so they cannot be half-applied.
- `beta` scheduler is now corroborated by BOTH authors (ULTRA+TURBO uses it; the scout note says
  "`beta` or `normal` tends to outperform `simple` for reference-heavy prompts"). Worth testing on
  the base recipe independently of the LoRA.
- The shipped `api_minimax_h3_*.json` ComfyUI templates are for the CLOUD node
  (`MinimaxHailuo03ReferenceNode`) and are NOT a guide for this local pipeline.

---

## 0. Why (what H3 gives us that LTX does not)

Capability deltas that matter for our pipeline, all [V] from the node schema + the author's guides:

- **Up to 9 reference images in one render**, each with an explicit role and retention verdict. Our
  MSR path caps at 4 subject refs and forces the background to be a SEPARATE person-free still.
- **Native reference AUDIO** (`ref_audios`, up to 3) with retention `fully_copy | partially_copy |
  reference | weak_reference`. This is the lip-sync lane without the SolidMask/SetLatentNoiseMask
  trick, and it can also act as a voice-timbre reference.
- **Native multi-shot in ONE generation** with cut timestamps (`[Shot 2] At 00:03.500, ...`).
- **A real camera vocabulary**: zoom in/out, push in, pull out, pan L/R, truck L/R, tilt U/D,
  pedestal U/D, arc, tracking, static, shake slight/strong, POV, roll CW/CCW, each with amplitude
  ("small/large") and speed ("slow/fast") modifiers. Our LTX path has THREE usable moves
  (static / slow push-in / slow pull-back) and lateral moves warp or no-op - see
  [[project_ltx-camera-lora]].
- **Native dialogue and singing** with stable speaker IDs `(S1)`, `<d>[English] ...</d>` tags,
  `<scenetrans>` for audio continuing across a cut, `<cutoff>` for interruption.
- **Reference VIDEO** (up to 3, each with an optional index-paired soundtrack) for motion/camera
  transfer, editing, and continuation.

What we LOSE (be explicit before committing):

- **N-keyframe interpolation.** `MiniMaxH3ImageToVideo` takes only `first_frame` / `last_frame`.
  Our `build_ltx_keyframe` places N stills at arbitrary frame positions; H3 has no equivalent. Any
  shot authored that way has no direct port.
- **Negative prompts.** The chain uses `BasicGuider` (no CFG), so there is no negative conditioning
  at all [V]. Every constraint must be phrased positively - same lesson as Krea2 (see docs/KREA2.md).
  `MVStudio.genStill`-style negatives and `ShotEditor`'s anti-zoom/axis-lock negatives are dead on
  this path.
- **Sub-5-second shots.** See the frame grid in section 3.

---

## 1. What is on the box (all [V], probed 2026-08-02)

Nodes registered (`/object_info`): `MiniMaxH3ImageToVideo`, `MiniMaxH3ReferenceToVideo`,
`MiniMaxH3SigmaShift`, `SpectrumApplyMiniMaxH3`, `ResolutionSelector`, `PathchSageAttentionKJ`,
`VHS_VideoCombine`, `BasicGuider`, `SamplerCustomAdvanced`, `VAEDecodeAudio`,
`ImageScaleToTotalPixels`. Core implementation is `ComfyUI/comfy_extras/nodes_minimax_h3.py`
(in-tree, not a custom node); the Spectrum speedup is `custom_nodes/ComfyUI-Spectrum-MiniMax-H3`.

Model files:

| File | Loader | Size |
|---|---|---|
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | UNETLoader, weight_dtype `default` | 21.0 GB |
| `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | UNETLoader, weight_dtype `default` | 21.0 GB |
| `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | CLIPLoader, type `minimax` | 27.1 GB |
| `minimax_h3_video_vae_fp16.safetensors` | VAELoader | 5.2 GB |
| `minimax_h3_audio_vae_fp32.safetensors` | VAELoader | 0.6 GB |

FL2VA serves the text + image lanes; REF2VA serves the reference lane.

**Sage**: the user re-added `--use-sage-attention` to `run_musicgen_lan.bat` (2026-08-02) because the
workflow author recommends it. Therefore we must **NOT** add `PathchSageAttentionKJ` to our graphs -
the author's own note says to patch in-graph only if the launcher lacks the argument. See the risk in
section 6 about what this means for the LTX path we still ship.

---

## 2. Canonical graph anatomy

Traced from the reference workflow. All three lanes share ONE chain after the conditioning node; the
only differences are which UNET is loaded and which conditioning node builds the latent.

Shared tail (identical in all three lanes) [V]:

```
UNETLoader(unet_name=<fl2va|ref2va>, weight_dtype="default")
  -> [Power Lora Loader]            (optional, empty by default in the workflow)
  -> [MiniMaxH3SigmaShift(shift_video=12, shift_audio=3)
      -> SpectrumApplyMiniMaxH3(...)]   (the "SPEEDUP" group; optional, see section 5)
  -> model

BasicGuider(model=model, conditioning=<H3node>.positive)
BasicScheduler(model=model, scheduler="simple", steps=20, denoise=1.0)   # NOTE: fed the PATCHED model
KSamplerSelect(sampler_name="res_multistep")
RandomNoise(noise_seed=<seed>)

SamplerCustomAdvanced(noise=RandomNoise, guider=BasicGuider,
                      sampler=KSamplerSelect, sigmas=BasicScheduler,
                      latent_image=<H3node>.LATENT)
  -> VAEDecode(vae=video_vae)        -> IMAGE
  -> VAEDecodeAudio(audio_vae=audio_vae) -> AUDIO

VHS_VideoCombine(images=IMAGE, audio=AUDIO, frame_rate=24,
                 format="video/h264-mp4", pix_fmt="yuv420p", crf=19,
                 filename_prefix="videogen/h3")
```

The joint AV latent is decoded TWICE off the same sampler output (video VAE for frames, audio VAE for
sound) - no separate-AV-latent node, unlike our LTX graphs [V].

### Lane A - TEXT / FIRST-LAST (FL2VA model)

```
MiniMaxH3ImageToVideo(clip, vae=video_vae, prompt, width, height, length,
                      [first_frame], [last_frame])  -> (positive, LATENT)
```

Four modes off one node [V]: no frames = text2video; `first_frame` = image2video;
both = first-last-frame; `last_frame` only = last-frame-to-video. The workflow's "LAST FRAME" group
feeds `last_frame` through `ImageScaleToTotalPixels(nearest-exact, 1.0, 32)` first.

Note this node takes only the VIDEO vae; audio still comes out of the shared tail's `VAEDecodeAudio`.

### Lane B - REFERENCES (REF2VA model)

```
MiniMaxH3ReferenceToVideo(clip, vae=video_vae, audio_vae=audio_vae,
                          prompt, width, height, length,
                          ref_image_size="match"|"max",
                          ref_image_0..8, ref_video_0..2,
                          ref_video_audio_0..2, ref_audio_0..2) -> (positive, LATENT)
```

The autogrow inputs are **flat, prefixed, zero-indexed keys** in API format - `ref_image_0`,
`ref_video_1`, `ref_video_audio_1`, `ref_audio_0` [V, from `comfy_api/latest/_io.py`
`Autogrow.TemplatePrefix`: `names = [f"{prefix}{i}" for i in range(max)]`]. The node collects them
into a dict and pairs a video with its soundtrack by index suffix
(`ref_video_audio_N` belongs to `ref_video_N`) [V, node source].

Maxima [V]: images 9, videos 3, video-audios 3, audios 3.

`ref_image_size` [V, tooltip]: `"match"` scales each ref DOWN only, keeping aspect, to the
generation's pixel area; `"max"` uses a 2048px short edge for best identity fidelity but "reference
tokens ride through every sampling step, so 'max' can be several times slower."

### What CANNOT be ported from the reference workflow

The workflow relies on UI-only constructs that have no API-format equivalent:

- `SetNode` / `GetNode` (model + VAE + CLIP distribution) -> inline the real links.
- rgthree `Any Switch` **feedback loops**: the subgraph computes the frame count and echoes
  width/height back INTO the H3 node's own `length`/`width`/`height` inputs. A cycle like that cannot
  exist in an API graph -> compute both in Python and pass literals.
- `ResolutionSelector` and `ComfyMathExpression` -> replaced by the two helpers in section 3.
- `Fast Groups Bypasser` / `Muter` (lane switching) -> we build one graph per endpoint.

---

## 3. Hard constraints (all [V])

**Frame grid: `frames % 17 == 5`, at 24 fps.** From `nodes_minimax_h3.py`:
`align_frame_count` does `while n % 17 != 5: n += 1` (snaps UP), and the node snaps internally, so an
off-grid `length` is accepted but silently changed. Mirror it on our side so our recorded
`frames`/`seconds` match reality:

```python
def _h3_frames(n):
    """MiniMax H3 wants frames % 17 == 5 at 24fps. Snap UP (mirrors align_frame_count)."""
    n = max(5, int(n))
    while n % 17 != 5:
        n += 1
    return n
```

Trained range is ~124-362 frames [V, `length` tooltip]:

| frames | 124 | 141 | 158 | 175 | **192** | 209 | 226 | 260 | 294 | 328 | 362 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| seconds @24fps | 5.17 | 5.88 | 6.58 | 7.29 | **8.00** | 8.71 | 9.42 | 10.83 | 12.25 | 13.67 | 15.08 |

So the practical shot window is **5.17s to 15.08s**. 192 frames is exactly 8.0s and is a good default.
The workflow's own default is 124 (~5s) with a duration slider.

**Resolution** comes from `ResolutionSelector(aspect, megapixels, multiple=32)`. The author's
reference table (0.9MP is his default; 0.4 = 480p tier; 2.0 = the top):

| MP | 16:9 output | note |
|---|---|---|
| 0.4 | 864 x 480 | cheap tier / gate tests |
| 0.5 | 960 x 544 | |
| 0.9 | **1280 x 736** | author default |
| 1.0 | 1376 x 768 | |
| 2.0 | 1920 x 1088 | heavy |

Our current MV resolution selector uses 1280x720; **736 is the H3-legal height at 0.9MP** (multiple of
32), so the MV canvas needs either 1280x736 or an explicit pad/scale at assemble. Worth deciding
early - `mv/assemble` already scales+pads every segment to the output canvas, so a 736 render into a
720 canvas is handled, but it will letterbox by 16px unless we set the canvas to 736.

**No negative prompt** (BasicGuider, no CFG).

---

## 4. Prompt format (the biggest work item)

H3 wants a structured script, not our current `"close-up shot. scene. wearing X. action. camera"`
string. Sections are literal keys in the prompt text [V, author guides + the 12k-char LLM system
prompt embedded in the workflow at node 2133].

**Text2video / image2video / first-last:**

```
integrated_multimodal_description: [Shot 1] <style, composition, characters, environment,
  lighting, action, camera movement, dialogue, diegetic sound>  [Shot 2] At 00:03.500, ...
overall_soundscape: <1-4 sentences: ambience, footsteps, impacts, wind, fabric, breathing>
non_diegetic_music: <audience-only music, or N/A>
```

Image2video must OPEN with this exact line, then a blank line [V]:

```
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.
```

First-last must open with [V]:

```
How the reference pictures align with the target video - Picture 1 (from Shot 1) aligns with the
0.00-second mark of the target video; Picture 2 (from Shot N) aligns with the S.SS-second mark.
```

**Reference mode uses six sections** [V]:

```
subject_definitions:   <Subject N> / <Picture N> / <Video N> / <Audio N>, each with its job
summary:               [reference generation | keyframe completion | video editing |
                        video continuation | audio reuse | audio reference]  (combine with +)
retention_analysis:    one line per reference, visual: fully_preserved | partially_preserved |
                        attribute_transfer | weak_reference
                        audio: fully_copy | partially_copy | reference | weak_reference
detailed_description:   style sentence, then chronological shots with labels used inline
overall_soundscape:
non_diegetic_music:
```

Rules that our builder must enforce mechanically:

- Shot 1 has NO timestamp; later shots have strictly increasing timestamps inside the duration.
- Dialogue: speaker ID outside the tag, only language + exact words inside `<d>[English] ...</d>`.
- Never repeat dialogue inside `overall_soundscape`.
- Visible on-screen text goes in English double quotes.
- Do not create a standalone `<Picture N>` definition when the image only defines a subject - mention
  the picture inside that `<Subject N>` definition.
- For a music video: `non_diegetic_music: N/A` always, because we mux our own song. Otherwise H3
  invents a score. (This replaces LTX's "discard the native audio" behaviour.)

---

## 5. Speed / quality levers

- `BasicScheduler(steps=20, denoise=1.0)`, sampler `res_multistep`, scheduler `simple` [V, workflow].
- **SPEEDUP group** = `MiniMaxH3SigmaShift(shift_video=12, shift_audio=3)` then
  `SpectrumApplyMiniMaxH3(enabled=true, blend_weight=0.5, degree=1, ridge_lambda=0.1, window_size=2,
  flex_window=0.75, warmup_steps=1, tail_actual_steps=1, max_history=8, debug=false,
  history_storage="system_ram", bootstrap_first_forecast=true)` [V, verbatim widget values].
  It is a MODEL->MODEL patch, so it also feeds `BasicScheduler`. Shipped bypassed in the workflow, so
  treat it as **opt-in** and A/B it once the base path works. NOTE `history_storage="system_ram"` is a
  RAM consumer and we are already RAM-constrained (section 6).
- `ref_image_size="max"` is the identity-fidelity lever and is "several times slower" [V].
- Resolution tier via the megapixel table.

---

## 6. Risks

1. **MEMORY - the main risk. [H] but grounded in file sizes [V].** A reference render touches
   ~54GB of weights (21 DiT + 27.1 text encoder + 5.2 + 0.6) on a 24GB card with 32GB system RAM.
   ComfyUI loads the encoder, frees it, then the DiT, so it is sequentially survivable, but the text
   encoder ALONE (27.1GB) exceeds VRAM and must spill. This is the same profile that gave us 150s/step
   SSD thrash on GGUF LTX (see MUSIC_VIDEO_PLAN section 11). **Gate everything on Phase 0.**
   Levers if it thrashes: 0.4MP tier, `ref_image_size="match"`, shorter `length`, Spectrum OFF
   (its `system_ram` history adds pressure), and possibly `--cache-none` / `--reserve-vram`.
2. **Sage now affects the LTX path we still ship. [H]** `--use-sage-attention` is back in the
   launcher. Our own note (2026-06-21, in the launcher comment) says sage silently drops attention
   masks and that this "killed the LTX PromptRelay per-segment-prompt scheduling". If that finding
   holds, LTX per-segment prompt timelines are silently degraded for as long as sage is on. Not
   verified now; check before trusting any LTX multi-segment shot. The launcher comment is
   self-contradictory (it says REMOVED while the flag is present) and should be corrected.
3. **No N-keyframe mode.** Shots authored with `brollMotion:"keyframe"` have no H3 equivalent.
   Either keep the LTX keyframe lane for those, or drop the feature.
4. **Sub-5.2s shots.** Our script writer emits 2-20s shots. Under H3 a 3s cut cannot be rendered
   natively; options are (a) render 124 frames and trim at assemble, (b) merge short cuts into one
   multi-shot render using native timestamps, (c) constrain the script writer to >=5.2s shots.
   This is the Phase 4 decision.
5. **Two 21GB models.** Switching between the image lane (FL2VA) and reference lane (REF2VA) inside
   one video is a 21GB model swap. Our `_submit_video` already frees only when the model signature
   changes, so batching by lane matters even more than it did for LTX.

---

## 7. Phases

Each phase is independently testable and gets its own commit. Do not start a phase before the
previous one is eyeballed by the user.

### Phase 0 - feasibility gate (no integration)
- `backend/video.py`: `build_h3_t2v(p)` - the shared tail + `MiniMaxH3ImageToVideo` with no frames.
- `backend/app.py`: `POST /api/video/h3_t2v`.
- Defaults for the gate: 864x480 (0.4MP), `length=124`, steps 20, Spectrum OFF, no LoRA, no sage
  patch. Prompt: a 3-section text2video prompt written to the author's format.
- **Measure:** s/it from the ComfyUI console, total wall time, and whether system RAM pins.
- **Decision point:** if it thrashes like GGUF LTX did, stop and report; H3 may simply not fit our
  box for reference-mode work.

### Phase 1 - image lane  [BUILT 2026-08-02, not yet rendered on the box]
- `build_h3_i2v(p, first_ref, last_ref=None)` + `POST /api/video/h3_i2v`. ONE endpoint covers three
  of the node's modes by which library ids you pass: `still_id` -> i2v; `still_id` + `last_id` ->
  first-last-frame; `last_id` alone -> last-frame-only. `resolved["lane"]` records which.
- Faithful details taken from the workflow's IMAGE TO VIDEO lane:
  - the FIRST frame is fed RAW; the LAST frame goes through
    `ImageScaleToTotalPixels(nearest-exact, 1.0, 32)` first (the author's asymmetry, not a slip);
  - output size is derived from the FIRST still via the "IMAGE SIZE" group
    (`ImageScaleToTotalPixels(megapixels)` -> `GetImageSize` -> width/height), which PRESERVES the
    still's aspect ratio instead of forcing 16:9. That chain is acyclic so it can be wired as real
    links, unlike the frame-count feedback loop. Explicit `width`+`height` selects the workflow's
    manual-resolution path instead.
- `_h3_dispatch()` in app.py now provides hunt/finish for EVERY H3 lane, so the image lane gets the
  seed hunt for free.
- Verified offline (no GPU): all four wiring modes, the raw-first/scaled-last asymmetry, the
  image-derived size chain, and turbo/base recipes applying to this lane.
- STILL TO DO on the box: render it, and re-test the camera moves LTX could not do (truck, pan, arc,
  pedestal) now that a still anchors the shot - the t2v tests proved the moves work WITHOUT an anchor,
  which is not the same thing.

### Phase 1 RENDER RESULTS + the TIMELAPSE/PACE investigation (2026-08-02, all [V])

**The image lane works on the box.** 2-draft hunt from a library still (person-free night clifftop,
1280x720), turbo, 124 frames:
- `mode:"hunt"` returned `{mode, base_seed, drafts:[2]}` with consecutive seeds - the same shape
  `/api/video/ltx_fflf` returns, so ShotEditor's existing pick-one-of-N pattern will slot in.
- **The anchor holds exactly**: source still vs frame 0 correlates at **r = +1.000** on both drafts.
  `first_frame` genuinely pins the opening frame.
- **Image-derived sizing works**: the 1280x720 still came out **1280x736** via the author's
  `ImageScaleToTotalPixels -> GetImageSize` chain (0.9MP on the /32 grid).
- Timings: 161s and 253s for the two turbo drafts.
- KNOWN GAP: in image-derived mode we record `width`/`height` as `None` in the library row (the
  dimensions are decided inside the graph, not in Python). Harmless but it makes the row less useful
  for later comparison; fill it from the finished file at reconcile time.

**USER VERDICT: parallax was fine but both drafts "looked like they were sped up timelapse shots."**
This is the same failure class we hit on LTX with scenic water/clouds (see VIDEO_PIPELINE_NOTES) so
it was worth decomposing properly rather than guessing.

Metric: mean absolute pixel change between matched-timestamp frames (128x72 grayscale, 0-255 scale) -
a motion-magnitude proxy. Composition r confirms each pair is the same shot, so the comparisons are
clean. FOUR conditions, varied ONE at a time from the failing draft:

| Condition | per-interval motion | mean | composition r vs draft1 |
|---|---|---|---|
| draft1: 3 motions incl. SKY, turbo | 8.99 9.24 10.45 11.22 | **9.98** | - |
| test1: SAME prompt, base recipe | 7.89 8.12 9.63 11.05 | **9.17** | +0.952 |
| test2: 1 motion + duration anchor, turbo | 7.54 7.04 6.86 6.38 | **6.96** | +0.980 |
| chapel: 1 motion + anchor, base, enclosed scene | 1.38 5.57 7.14 7.18 | **5.32** | (different shot) |

**Attribution, in order of magnitude:**
1. **PROMPT ~30%** (9.98 -> 6.96, prompt-only change). The dominant cause. Note the SHAPE flip too:
   the busy prompt ACCELERATES through the clip (8.99 -> 11.22) while the disciplined one
   DECELERATES (7.54 -> 6.38). Accelerating vs settling is the timelapse signature, and it came
   purely from wording.
2. **CONTENT ~ the residual** (6.96 -> 5.32). Open-sky night scenery carries more apparent motion
   than an enclosed stone interior even with identical prompt discipline.
3. **RECIPE ~8%** (9.98 -> 9.17, recipe-only change). Turbo is largely EXONERATED - which is good
   news for the hunt: turbo drafts do represent their base finals for pace, so the hunt economics in
   0a stand. (Contrast LTX, where the distill WAS the motion-speed culprit - do not carry that
   assumption across models.)

**TWO PROMPT RULES for the Phase 3 compiler (earned, not guessed):**
- **ONE described motion per shot, with a real-world duration anchor.** The wording that worked:
  "each stem's movement taking only a fraction of a second exactly as real grass does". Giving the
  model a physical referent to calibrate against is the lever. This matters because the chain is
  cfg-free so there is NO negative prompt - pace can only be steered POSITIVELY.
- **Pin the sky explicitly.** "The stars and the Milky Way hold fixed positions with no drift at all,
  and there is no cloud movement anywhere." Sky elements are the strongest timelapse cue and the
  model animates them if they are left unmentioned.
- Corollary constraint: prefer foreground-anchored framings for held shots; wide open-sky B-roll will
  always carry more motion. The author's own guide says the same thing in general terms ("do not
  overload short clips with too many actions", every movement "physically possible within the
  requested duration") - we now have the numbers for why.

**METHOD LESSON (recorded because it cost renders):** the first pace attempt changed THREE things at
once (recipe + content + prompt), produced a good clip, and was therefore worthless - user: "its
meaningless because you changed so many things, we have no idea which actually worked." Vary ONE
thing per render, hold the seed and the shot fixed, and carry unchanged sentences over VERBATIM
(assert it in code - the camera sentence was hash-checked identical between draft1 and test2).

### Phase 2 - reference lane (the MSR replacement)
- `build_h3_ref2v(p, ref_images, ref_videos, ref_video_audios, ref_audios)` on the REF2VA model,
  emitting flat `ref_image_N` / `ref_audio_N` keys.
- `POST /api/video/h3_ref2v`, mirroring `/api/video/ltx_msr`'s library-id -> upload plumbing
  (`_lib_image_path`, `C.upload_audio`, `_audio_name` content-hash naming, `_isolate_vocal_bytes`
  for the audio window). Reuse `_trim_audio_window` so the audio matches the clip length.
- Feed character sheet + wardrobe + background as separate refs; test whether the person-free
  background rule can be dropped (declare the background as its own `<Subject>`).

### Phase 3 - prompt builder
- `backend/musicvideo.py`: new `build_h3_prompt(shot, cast, mode)` producing the sectioned format,
  plus a rewrite of the script-writer schema so the LLM emits H3-shaped shots (subject definitions,
  retention verdicts, timestamped sub-shots, dialogue with speaker IDs, soundscape,
  `non_diegetic_music: N/A`).
- The author's embedded 12k-char LLM system prompt (workflow node 2133) is effectively the spec -
  adapt it for our writer rather than inventing our own phrasing.
- MUST enforce the two measured pace rules from the Phase 1 results above: exactly ONE described
  motion per shot carrying a real-world duration anchor, and an explicit "the sky holds still" clause
  whenever sky/cloud/stars are visible. These are the difference between a held shot and a timelapse
  on this model, and there is no negative prompt to fall back on.
- `web/src/mvmodel.ts`: `shotToBlock` stops building the LTX-style prompt string and carries the
  structured fields instead.

### Phase 4 - shot-model decision (architecture, needs a design pass)
Either:
- **(a) 1 block = 1 render** (minimal change): clamp shots to 5.17-15.08s, trim at assemble.
- **(b) 1 render = a multi-shot scene** (better model fit): one generation covers several scripted
  cuts via native timestamps. Changes what a "shot" is in MV Studio, and breaks per-shot
  retake/pick/upscale as they exist today.
Decide with the user; do not pick unilaterally.

### Phase 5 - retire LTX (only after 1-4 are proven)
Keep `build_ltx_*` until H3 is user-approved on identity, lip-sync and camera. Then remove the
LTXDirector paths, or keep `build_ltx_keyframe` alone for N-keyframe shots if we still want them.

---

## 8. Files to modify

- `backend/video.py` - `_h3_frames`, `_h3_resolution`, `build_h3_t2v` / `build_h3_i2v` /
  `build_h3_ref2v`, plus H3 model-file constants.
- `backend/app.py` - `/api/video/h3_t2v` / `h3_i2v` / `h3_ref2v`; a `video_h3` capability flag in
  `/api/config` gated on `C.has_node("MiniMaxH3ReferenceToVideo")`.
- `backend/musicvideo.py` - H3 prompt builder + script-writer schema (Phase 3).
- `web/src/api.ts` - `videoH3T2V` / `videoH3I2V` / `videoH3Ref2V`.
- `web/src/mvmodel.ts` - `RenderMode` gains H3 modes; payload builders.
- `web/src/ShotEditor.tsx` - dispatch to H3; drop the negatives (inert) and widen the camera list.
- `web/src/MVStudio.tsx` - resolution tier (736-legal), lane batching.

## 9. Verification per phase

`py_compile` + `npx tsc --noEmit` + `npm run build`, then ONE real render on the box per phase with
the user present, then the user's ears/eyes on the result before the next phase. Confirm the feature
actually ENGAGED, not merely that the job completed ([[verify-feature-engaged-not-just-ran]]) - e.g.
for Phase 2 confirm the refs were consumed (identity holds) rather than silently ignored.

## 10. Open decisions

1. Phase 4: 1-block-1-render vs multi-shot-per-render.
2. Do we keep `build_ltx_keyframe` for N-keyframe B-roll, or drop that capability?
3. MV canvas: move to 1280x736 or keep 720 and letterbox 16px at assemble?
4. Spectrum speedup on or off by default (RAM cost vs speed)?
5. Does the person-free-background rule survive? (test in Phase 2)
