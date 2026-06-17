# Video Pipeline Notes (empirical ledger)

Running log of what we have tried in the photoreal music-video pipeline, with verdicts and
the REASON each thing failed, so we do not repeat dead ends. Companion to MUSIC_VIDEO_PLAN.md
(that is the plan; this is the lab notebook). Newest findings at the bottom of each section.

Test subject: "Selene" (Garden of Ashes). Box: Windows RTX 3090, 24GB VRAM, 32GB system RAM.
All renders driven from the app Video tab -> backend/video.py graph builders -> ComfyUI.

---

## WHAT WORKS (settled)

### 20s natural-speed walk -- SOLVED via MSR (2026-06-17)
- Route: `POST /api/video/ltx_msr` -> `build_ltx_msr` (backend/video.py).
- Identity comes from REFERENCE IMAGES (LTX Licon MSR IC-LoRA), NOT a keyframe imprint, so
  there is no pose anchor and motion is fully prompt-driven. This is what broke the all-day
  identity-vs-motion deadlock.
- Settled params (job 646a2cf0, user-approved):
  - 481 frames / 24fps / 832x480 / cfg = 1.0
  - subjects: 2 Selene refs -- 348645e0 (orb-free gown medium shot) + 7b5f66b7 (face close-up)
  - background: bf6c81d8 (garden scene, no character)
  - prompt: steady natural-pace walk + CAMERA TRACKS BACKWARD (so she has road for 20s)
  - distill LoRA: STOCK `ltx-2.3-22b-distilled-lora-384-1.1` (NOT ceil72 -- see dead ends)
- Result: identity holds the FULL 20s (no mid-clip morph), color/composition stable, camera
  tracks, motion plays at natural temporal rate (slow-mo FEEL gone), no VRAM thrash at cfg=1.
- Custom nodes on box: ComfyUI-Licon-MSR, kijai ComfyUI-PromptRelay, word2number (pure-file
  deploy into python_embeded site-packages, NO pip into the venv).

### InfiniteTalk v2v dub lane -- BUILT (2026-06-16)
- Route: `POST /api/video/infinitetalk` -> `build_infinitetalk_v2v`. Keeps an existing clip's
  motion/camera/bg, redrives ONLY the lips from audio. Wan2.1 i2v 14B + MultiTalk infra.
- Gotchas baked in: quantization `fp8_e4m3fn` (plain, NOT `_scaled`); frame count must SNAP to
  the MultiTalk window boundary (mf + k*(fws-mf)) or lips freeze at the end.
- Status: built + wired, not yet eyeballed for quality.

### /comfy restart endpoint -- BUILT
- `POST :5080 /comfy/restart` kills ONLY the python PID listening on :8188 and relaunches in a
  visible console. Solves GGUF memory residue without a full box reboot.

---

## DEAD ENDS (do NOT retry)

| Approach | Result | Why it failed |
|---|---|---|
| Pose-guided S2V (motion from pose video + lips from audio) | Treadmill: legs cycle in place, no travel | S2V LOCKS the subject to the reference-still framing. Proven by frames. |
| SVI2Pro chained (build_svi_i2v) at 20s | Identity morphs to a DIFFERENT woman by mid-clip; color/saturation swings; composition runaway | Autoregressive chaining drift accumulates across segments. Partly misuse (dolly prompt + non-identity-locked prompt) but chaining drift is real. |
| LTX temporal tiling (LTXVLoopingSampler, tiled=true) | Fixes cost (260s/it -> 5s/it) BUT diverges to a different woman + different scene after the first ~57-frame tile | Same autoregressive drift class as SVI. Coherence knobs untuned; not usable as-is. |
| LTX non-distilled + cfg ~2-2.5 (to make "slow motion" negative bite) | Motion got WORSE + 74min render (thrash) | cfg>1 runs a 2nd forward pass per step -> ~2x per-step VRAM -> shared-mem thrash. |
| LTX motion_fps decouple (condition at lower fps than playback) | Made slow-mo worse | Wrong theory; fps conditioning must match. |
| 48fps single render | Helped slow-mo only PARTIALLY | Doubles frames (961) -> VRAM thrash (19GB + 15GB shared); also fps mismatch with other clips downstream. User wants to AVOID 48fps. |
| ffmpeg retiming (speed up the clip) | Still looked slow at 1.3x AND shortens the clip | Not a real fix for the slow-mo FEEL. |
| ceil72 distill for MSR 20s walk | A step HITCH around 3-4s in one take | Higher motion ceiling MAY spike a jerk; stock distill (same seed) removed it. NOTE: may have been an ephemeral one-gen artifact, not definitively ceil72. |
| Vocal stems for lip-sync audio (Demucs / BS-RoFormer :5070) | Unusable | ACE-Step bakes vocals into the mix; isolation smears. Days spent confirming. Stem-free is the only path. |
| LongCat (alt to S2V) | Parked | quanto fp8 incompatible with our stack; bf16 too big for 32GB RAM. |

---

## KEY MECHANISM FACTS (the "why", so we reason instead of re-testing)

- **Slow-mo is frame-rate-CONDITIONING driven**, not the distill. The all-day "slow motion"
  problem was really the keyframe anchor forcing the model to interpolate tiny movements ->
  every limb/gown/hair motion read as time-stretched. Remove the keyframe anchor (MSR) and the
  temporal FEEL is natural regardless of how fast she actually travels.
- **cfg>1 and STG each add a 2nd (or perturbed) forward pass per step** -> ~2x per-step VRAM ->
  thrash at 481 frames (20s). So they are OFF-LIMITS as pace levers for long clips, even though
  they work fine on short 6s tests. The lesson: dial pace only with levers that survive 20s.
- **The "slow motion" NEGATIVE prompt is inert at distilled cfg=1** (needs cfg>1 to bite), so do
  not rely on it. Real 20s-safe pace levers: brisker PROMPT, and weight-merged motion LoRA
  (ceil72 / VBVR -- weight-space, no extra forward pass).
- **At 6s a brisk walk crosses the WHOLE path and arrives at camera.** For 20s the camera must
  track/dolly back with her (endless path) or the prompt must describe a long receding avenue,
  else she runs out of road.
- **24fps / 20s = 481 frames, which FITS VRAM with no thrash** at cfg=1. We do NOT need tiling
  for 24fps. Tiling was only ever a cost fix and it breaks continuity, so skip it.
- **A plain text2img still does NOT carry Selene's identity** (a name token is not a face). Use
  `/api/video/char_still` with the Selene anchor ref (cb81a415) to make on-model stills.

---

## OPEN THREADS / NEXT

- **LipDub** (`ltx-2.3-22b-ic-lora-lipdub-0.9`, on box): wire the in-LTX singing lip-sync on top
  of the MSR walk so she walks AND sings. RESEARCH FIRST: does it dub a finished clip (v2v) or
  generate jointly in one pass? Can MSR + LipDub IC-LoRAs coexist in one graph?
- Other LoRAs on box to evaluate: VBVR / OmniNFT / Motion-Track (motion), Crisp / Soft / Fantasy
  (look enhancers), plus a good single character LoRA for general use.
- Music-video assembly (grade + upscale via SeedVR2) -- see project_mv-grade-and-upscale memory.
