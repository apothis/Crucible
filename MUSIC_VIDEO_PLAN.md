# Music Video Pipeline - design + phased build plan

_Crucible feature: turn a generated song into a full, character-consistent, beat-cut music video,
driven entirely from the app (ComfyUI headless). Research-led design, 2026-06-13. Builds on the
verified Phase-1 gate (Z-Image stills + Wan2.2 i2v + Wan2.2-S2V lip-sync all working). See
RESEARCH.md S19/S19a/S19b and memory [[video-pipeline]]._

Provenance tags: VERIFIED = confirmed by a source/tool/our own run; HYPOTHESIS = plausible, not yet
confirmed (per [[no-fabricated-causes]] - do not treat as fact); VERIFY-AT-BUILD = must test on our box.

## 1. Architecture (5 stages)

```
Song (library: audio + sections + lyrics + bpm)        [we already have this]
   |
1. SCRIPT      LLM -> ordered shot list (15-30 shots), each timed to a section/beat,
   (llm.py)    with: scene prompt, shot type (performance | narrative | b-roll),
               characters present, lip-sync? , motion cue.   [editable by the user]
   |
2. CAST        Define recurring characters once (lead singer, band members, narrative
               lead) + scenic shots with no fixed cast. Each character = a locked
               identity reused across all their shots. (consistency method = S3)
   |
3. SHOTS       Per shot: Z-Image keyframe still (with the character's identity if a
               character shot; scenic prompt otherwise) -> animate (Wan2.2 i2v) OR
               lip-sync (Wan2.2-S2V against that shot's audio segment).
   |
4. ASSEMBLE    Order shots -> cut to the beat (librosa /api/beats + Yvann nodes) ->
               frame-interpolate (RIFE/FILM) + upscale -> mux the full song -> 3-4 min MP4.
   |
5. UI          A "Music Video" builder: pick song -> gen/edit script -> manage cast ->
               generate shots (progress + preview) -> arrange timeline -> render final.
```

Everything reuses the existing job/WS/poll/library plumbing and the proven build_still/i2v/s2v
builders. The new work is orchestration (script, cast, shot graph, assembly) + UI.

## 2. Data model (new)

- **VideoProject** = { id, song_id, title, script[], cast[], settings }.
- **Shot** = { idx, section, t_start, t_end, type, scene_prompt, motion_prompt, character_ids[],
  lipsync: bool, still_job_id?, clip_job_id?, status }.
- **Character** = { id, name, role: singer|band|narrative, identity_prompt, ref_still_id,
  lora_name?, method }.
Persist alongside the existing `projects` table (or a new `video_projects` table). The library
already stores stills/clips by job id, so shots reference those ids.

## 3. Character consistency - the hard part (options + tradeoffs)

The make-or-break. No single verified silver bullet on a 24GB 3090; layered options:

**(A) Anchor-still reuse (Tier 1, works NOW, zero new infra).** Generate one canonical still per
character (Z-Image), the user approves it, and every shot featuring that character uses it as the
i2v/S2V START frame. Identity holds within a shot (VERIFIED: i2v/S2V preserve the start frame). Across
shots it is the SAME anchor, so framing-similar shots stay on-model. LIMITATION (HYPOTHESIS): new
poses/angles/scenes of that character need new stills that match the face, which Z-Image alone will
not reliably reproduce - so shot variety per character is limited without (B) or (C).

**(B) Z-Image character LoRA (Tier 2, best still-consistency, light training).** VERIFIED route:
9-25 images, rank 8-16, lr 1e-4, ~2-3k steps (Ostris adapter). Then generate unlimited consistent
stills of that character in any pose/scene. Cost: need an IMAGE-LoRA trainer on the box (ai-toolkit /
Ostris - NEW tool; our ACE-Step LoRA infra is audio-only) + a training image set per character
(bootstrap: one good still -> a few reference-consistent variations, or user-supplied photos). Best
for the lead singer + named band members.

**(C) Qwen-Image-Edit-2511 reference-driven (Tier 2 alt, NO training).** VERIFIED to do reference-
driven multi-person consistency; generates each shot's keyframe from character reference(s) with no
per-character LoRA. **GGUF fits 24GB (VERIFIED sizes):** Q5_K_M 14.9GB / Q6_K 16.8GB (Q8 21.8GB) via
the installed ComfyUI-GGUF node -> Q5/Q6 leaves room for text encoder + VAE. So this is PRACTICAL on
the 3090, not just theoretical. Best for the BAND (multi-person in one frame) and to avoid a training
pipeline. (unsloth/Qwen-Image-Edit-2511-GGUF, QuantStack/Qwen-Image-Edit-GGUF.)

**(D) Wan VACE / Phantom reference-to-video (Tier 3, video-side identity).** Native on our box
(WanVaceToVideo, WanPhantomSubjectToVideo). Pushes identity through motion from a reference image.
fp16/fp8 14B is heavy (~32GB cited) BUT **a GGUF exists (QuantStack Wan2.1_14B_VACE-GGUF)** -> 24GB
feasible via Q5/Q6, VERIFY-AT-BUILD. Otherwise fall back to anchor-still i2v (A). Zero-shot
PuLID/IPAdapter face-id FOR Z-IMAGE is UNVERIFIED (PuLID is FLUX-oriented) - do not assume; check first.

**GGUF is a standing VRAM lever ([[check-gguf-for-vram]]).** Every 14B Wan variant has a GGUF
(QuantStack/city96): S2V Q5 ~14.3GB (the cure for the gate's sampling-time shared-RAM spill), i2v/t2v
A14B Q5 ~10-11GB. So "too heavy for 24GB" should always be checked against a GGUF before being believed.

**Recommended layering:** start with (A) for an end-to-end pipeline, add (B) Z-Image LoRA for the lead
singer + key band members (strong varied-shot consistency), use (C) Qwen-Edit for multi-person band
shots, and treat (D) as an optional video-side boost pending the VRAM check. Scenic shots need no
consistency.

## 4. Script generation (stage 1, known approach)

LLM (backend/llm.py) prompt = song sections + lyrics + mood/genre tags + bpm + duration. Output a
JSON shot list: each shot { section, approx start/end (snapped to /api/beats markers), shot type,
scene description -> becomes the Z-Image prompt, which characters, lip-sync flag, motion cue }. The
user edits the list (add/remove/reorder/retime/reprompt shots) before generation. Performance/vocal
shots map to lyric-bearing sections (so S2V lip-syncs the right audio segment via the audio_start
offset we already built).

## 5. Assembly (stage 4, known building blocks)

Shots render to clips in the library. Assembly orders them by t_start, cuts on beat boundaries
(librosa beats we already expose, or Yvann Audio Peaks), runs RIFE/FILM interpolation + an upscale
pass (nodes installed), concatenates (VideoHelperSuite), and muxes the full song audio. Output one
MP4. HYPOTHESIS: a 3-4 min render is a long batch job (many shots x minutes each) - serialized vs the
music engine [[no-concurrent-clap-engine]], likely an overnight run.

## 6. Phased build

- **Phase A - foundation (no new models/training).** Data model + LLM script generation + a Music
  Video tab: song picker, editable shot list, simple cast (anchor-still per character, method A),
  per-shot generate (reusing build_still/i2v/s2v), preview each shot. Goal: drive a full shot list
  end to end from the app.
- **Phase B - consistency.** Add Z-Image character-LoRA training (image trainer on the box) and/or
  Qwen-Image-Edit reference-driven stills; let a character pick its method. Verify on the lead singer.
- **Phase C - assembly + polish.** Beat-cut ordering, RIFE/FILM interpolation, upscale, concat + mux
  to a final MP4; in-app timeline. Optional Wan VACE/Phantom video-side identity (after VRAM check).
- **Phase D - scale/quality.** Shot variations/reroll, S2V Extend for long performance takes, cloud
  hero-shot top-up (Sora/Kling) for marquee scenes, presets.

## 7. Open verifications (do before depending on them)
1. Z-Image image-LoRA training on the 3090: tool (ai-toolkit/Ostris), time, VRAM, and how to build a
   per-character training set.
2. Qwen-Image-Edit-2511 GGUF (Q5/Q6, fits 24GB - sizes VERIFIED): reference-consistency quality vs
   Z-Image+LoRA in practice.
3. Wan VACE/Phantom 14B GGUF (exists, QuantStack) actual 24GB fit + identity quality - or confirm
   anchor-still i2v is enough for video identity.
4. Any zero-shot face-id (PuLID/IPAdapter/InstantID) that actually works with Z-Image in ComfyUI.
5. (Carried from the gate) S2V GGUF Q5 (~14.3GB) as the spill cure if the driver setting isn't enough.

**Sources (S3/S4):** runcomfy (consistent-character workflows, Wan2.2 VACE Fun), apatero + nextdiffusion
(Z-Image character LoRA settings), HF Qwen-Image-Edit-2511, ComfyUI VACE docs, digen/onemoreshot AI
music-video guides (shot-list + beat-sync). Character-LoRA params + Qwen-Edit consistency VERIFIED via
multiple sources; VACE 24GB fit + Z-Image zero-shot face-id UNVERIFIED.

## 8. Character consistency - ALL FOUR methods built (2026-06-14)

Per "pre-build both alternatives", every consistency path is now wired into the Music Video tab
(method selector: auto / anchor / Qwen / VACE / LoRA). Each covers a different failure mode:

| Method | Code | Covers | Status |
|---|---|---|---|
| Anchor still | genShot anchor branch (i2v on the ref still) | quick, no extra models | WORKS (Phase A) |
| Qwen-Image-Edit | build_qwen_char_still + /api/video/char_still | place character in new scene (face fidelity) | built from spec, GGUF downloading |
| Wan VACE ref2v | build_vace_ref2v + /api/video/vace | identity held THROUGH motion (i2v drift) | built from spec, GGUF downloading |
| Z-Image char LoRA | build_still `lora` + /api/video/loras + cast picker | highest face fidelity, any pose | consuming side built; train externally |

`auto` resolves to: LoRA if the character has one, else Qwen if its GGUF is present, else anchor.
video_qwen / video_vace config flags auto-detect their GGUFs (so the options enable themselves once
download_video_models2.bat finishes + ComfyUI restarts).

### Z-Image character-LoRA training (external tool - Ostris AI Toolkit)
Crucible CONSUMES the LoRA (cast picker -> build_still applies LoraLoaderModelOnly); training is done
in **Ostris AI Toolkit** (github.com/ostris/ai-toolkit - 1-click Windows installer + its own web UI;
VERIFIED supports Z-Image Turbo LoRAs with 24GB configs). Per-character recipe (VERIFIED from multiple
2026 tutorials): 5-15 (up to 25) images of the character at 1024x1024, rank 8-16 (8 or 16), lr 1e-4
(down to 5e-5 for tighter identity), ~3000 steps, batch 1-2; ~1h on a 5090 (slower on the 3090).
Bootstrap the image set from one good Z-Image still -> a few Qwen-Edit/VACE variations, or user photos.
**Integration:** drop the trained `.safetensors` into ComfyUI/models/loras -> it appears in the cast
LoRA picker -> select it on a character -> that character's shots generate with the LoRA (consistent
identity in any scene). NOT shipping a from-scratch install bat (ai-toolkit's own installer + cu130
torch are better handled by its maintained setup; an untested heavy install would be fragile).

ALL untested-on-hardware paths (Qwen/VACE/LoRA graphs) need a first-fire check once models land +
a ComfyUI restart - watch for node-wiring errors and report them.

## 10. LTX-2.3 fast video backbone - BUILT 2026-06-14 (validation pending)
Migrated the i2v motion stage off the slow Wan path (native nodes + GGUF, ~126s/step, GPU pegged
on pytorch attention) onto LTX-2.3 (SageAttention runtime, distilled 8-step, native synced audio).
Box provisioned: SageAttention installed, LTX nodes installed, 8/8 model files present (incl. the
22.76GB ltx-2.3-22b-dev-Q8_0.gguf), ComfyUI restarted with --use-sage-attention.

Built (all node sigs verified live on box /object_info, values from reference/LTX-2-3_ULTRA_WORKFLOW-V2.json):
  - backend/video.py: build_ltx_t2v / build_ltx_i2v (shared _build_ltx). LTXDirector orchestrates
    (model+clip+prompt -> positive cond + video_latent + audio_latent + frame_rate + combined_audio);
    joint AV sampling = LTXVConcatAVLatent -> SamplerCustomAdvanced (euler + distilled 8-step sigmas,
    CFG 1) -> LTXVSeparateAVLatent -> spatio-temporal tiled video decode + audio VAE decode -> CreateVideo.
    i2v imprints the keyframe via LTXVImgToVideoInplace (strength 0.7) onto LTXDirector's video latent.
    Defaults: 768x512, 97 frames (~4s @24fps; coerced to valid 8k+1), distilled LoRA 0.5 + detailer 0.2.
  - app.py: /api/video/ltx_t2v + /api/video/ltx_i2v, video_ltx capability flag (detect via C.gguf_unets()).
  - web: Music Video tab "Motion engine" selector (LTX fast / Wan fallback), genShot animate stage
    routes through LTX i2v when video_ltx + selected. Lip-sync + still-gen unchanged.
  - NOT ported for v1 (kept simple for the first measurable clip): the 2nd-stage LTXVLatentUpsampler
    refine pass (extra sharpness), first-last-frame + extend/looping modes. Add after speed is confirmed.

LIP-SYNC DECISION (LTX native audio vs S2V): LTX native audio is GENERATED from text (invented
ambience/speech), NOT driven by our song vocal -> it does NOT lip-sync a singer to OUR track. So:
  - NON-lip-sync motion/B-roll shots -> LTX i2v (the speed win); native audio discarded, song muxed at assembly.
  - Lip-sync vocal shots -> stay on Wan2.2-S2V (audio-driven, proven). genShot already does exactly this.
  - FOLLOW-UP: LTX 2.3 ships audio-conditioning nodes (LTXVReferenceAudio, LTXVSetAudioRefTokens,
    LTXVAudioVideoMask) - evaluate whether they do true vocal-driven lip-sync to unify on the fast
    backbone. Do not block the speed validation on it.

VALIDATE NEXT (user presses Generate): render ONE shot with Motion engine = LTX, measure real s/step
from the ComfyUI log (expect a big drop from 126s). Fix any node-wiring error on first fire.

## 9. QUEUED (after LTX) - Z-Image ULTRA finishing pass (opt-in)
Source: reference/Z-IMAGE_TURBO_ULTRA_WORKFLOW-V2.json (AItrepreneur). Verified 2026-06-14 that our
build_still already matches the ULTRA CORE (ModelSamplingAuraFlow shift ~3, lumina2 CLIP, ae VAE,
euler/simple ~8-9 steps cfg 1.0, LoRA). This batch adds the ULTRA's FINISHING layers as opt-in
toggles in the Music Video tab; DEFAULT OFF, never changes current still behavior.

DO NOT run the ULTRA install bat on the box - it downloads a FRESH ComfyUI portable (v0.3.76) and
relaunches it (a second parallel ComfyUI). Cherry-pick into our EXISTING ComfyUI instead.

Already present on box (verified via /object_info): CLIPLoaderGGUF, UnetLoaderGGUF,
ModelSamplingAuraFlow, Power Lora Loader, UpscaleModelLoader, QwenImageDiffsynthControlnet,
InpaintModelConditioning, ModelPatchLoader, SamplerCustomAdvanced, easy clearCacheAll/cleanGpuUsed.

MISSING custom nodes to add (git clone into ComfyUI/custom_nodes, then pip -r requirements):
  - Detail-Daemon         github.com/Jonseed/ComfyUI-Detail-Daemon       -> DetailDaemonSamplerNode (HIGH: skin/texture realism)
  - vrgamedevgirl         github.com/vrgamegirl19/comfyui-vrgamedevgirl  -> FastFilmGrain, FastLaplacianSharpen (HIGH: photoreal finish)
  - wlsh_nodes            github.com/wallish77/wlsh_nodes                -> Upscale by Factor with Model (WLSH) (HIGH: 1080p to match LTX)
  - controlnet_aux        github.com/Fannovel16/comfyui_controlnet_aux   -> DepthAnythingV2/Canny/DWPose preproc (HIGH: cross-shot pose consistency)
  - ComfyUI_essentials    github.com/cubiq/ComfyUI_essentials            -> ImageResize+ (helper)
  - SeedVarianceEnhancer  github.com/ChangeTheConstants/SeedVarianceEnhancer (LOW: seed variety)

MISSING model files (from HF Aitrepreneur/FLX, ungated; download script or box-fs):
  - model_patches/Z-Image-Turbo-Fun-Controlnet-Union-fp8-e5m2.safetensors  (pose/depth control)
  - upscale_models/4x-ClearRealityV1.pth, RealESRGAN_x4plus_anime_6B.pth   (upscalers)
  - (OPTIONAL, we already have bf16 equivalents - skip unless VRAM-pinched:
     unet/z_image_turbo-Q8_0.gguf, text_encoders/Qwen3-4B-UD-Q6_K_XL.gguf)
  - controlnet_aux preproc weights (DepthAnythingV2, DWPose) auto-download on first use.

Backend work (mirror existing build_* pattern, additions OPTIONAL/gated OFF):
  - Extend build_still (or build_still2) with optional chain after KSampler:
      detail: swap KSampler -> SamplerCustomAdvanced + DetailDaemonSamplerNode
      finish: FastLaplacianSharpen + FastFilmGrain on the decoded image
      upscale: Upscale by Factor with Model (WLSH) -> 1080p+ keyframe (match LTX)
      pose:    ControlNet-Union + Depth/Canny/DWPose preproc from a reference pose image
  - app.py: new optional params on the still endpoint (detail/grain/sharpen/upscale/pose), each default off.
  - Music Video tab: small "Still finishing" toggles group; off = current behavior verbatim.
  - First-fire validation on box (user presses Generate); report any node-wiring errors.
