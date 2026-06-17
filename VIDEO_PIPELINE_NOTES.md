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

### Walk + SING in ONE on-model LTX pass -- SOLVED via MSR + native audio (2026-06-17)
- Route: `POST /api/video/ltx_msr` with `audio_id` -> `build_ltx_msr(..., vocal_ref=...)`.
- The singing lane. Lips are driven by the REAL song INSIDE the same MSR walk render - no keyframe
  anchor, no Wan/InfiniteTalk, no LatentSync, no second-model degradation. User verdict 2026-06-17
  (job 8f990ed4, 12s): "That works well... movement is good, all looks good." Identity + walk + clean
  LTX look all intact, lips track the vocal.
- THE RECIPE (the missing piece was the noise mask - traced from the official ia2v workflow
  video_ltx2_3_ia2v.json):
  1. RoFormer-isolate the vocal, trimmed to EXACTLY frames/fps (an over-long vocal misaligns the AV
     latent and leaks uncropped MSR reference frames into the first ~second).
  2. `LoadAudio -> LTXVAudioVAEEncode(audio_vae)` -> encode the vocal.
  3. `SolidMask(value=0, 1024x1024) -> SetLatentNoiseMask(samples=audio_latent, mask)` -- value 0 =
     noise mask of all zeros = "preserve, do NOT denoise" -> holds the vocal FIXED through sampling
     so the video is generated to MATCH it. THIS is what makes lips actually sync. WITHOUT it, the
     audio gets denoised away -> mouth moves but does NOT track the words (proven: job 357372f8).
  4. `LTXVConcatAVLatent(video, masked_audio)` -> sampler -> separate -> crop -> decode.
  5. Mux the driving vocal into CreateVideo.audio so playback has the song to judge sync.
- CFG stays 1 (the official ia2v is also cfg=1; CFG was NOT the lever). Base LTX-2.3 does the
  audio->lip conditioning natively - NO dedicated lip-sync IC-LoRA needed (the MSR IC-LoRA is the only
  one loaded). Minor: mouth movement reads slightly exaggerated at 832x480 where the face is tiny -
  expected to read better in closer framing / higher res; not a problem per user.

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
| Vocal stems for REMIX / re-timbre (Demucs / BS-RoFormer :5070) | Unusable AS STEMS | ACE bakes vocals into the mix; isolation smears at hi-fi/stem quality. NOTE: this is the STEM bar only -- RoFormer vocal IS adequate as a LIP-SYNC DRIVER (lower bar, see below). Do not over-apply. |
| LongCat (alt to S2V) | Parked | quanto fp8 incompatible with our stack; bf16 too big for 32GB RAM. |
| LTX LipDub IC-LoRA (`ltx-2.3-22b-ic-lora-lipdub-0.9`, on box) for SINGING | Wrong tool | Verified from the official workflow JSON (LTX-2.3_ICLoRA_Lipdub_Two_Stage_Distilled): NO LoadAudio node. It is TEXT-DRIVEN -- you type target DIALOGUE in the prompt and it GENERATES new audio+lips from that text. For spoken dubbing/translation only. It would invent a voice, not sync to our ACE-Step song. Do NOT use for walk+sing. |
| InfiniteTalk v2v (Wan2.1) to dub the LTX MSR walk (job 649030fa) | Terrible (user verdict) | Runs the LTX footage through a SECOND model (Wan2.1). It DROPPED the walk (she stops mid-clip), shifted colour/saturation, and went fake/plastic by the end. ROOT PROBLEM: any v2v dub re-generates every frame through a foreign model -> degrades the LTX look + can lose the motion. Do NOT dub finished LTX clips through Wan. The fix is to stay in ONE LTX pass (LTX is natively audio+video). |

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

- **Singing lane (walk + sing):** LipDub is OUT (text-driven, see dead ends). Use an AUDIO-driven
  tool with RoFormer-isolated vocal as the driver (ANSWERED 2026-06-17: RoFormer vocal is adequate
  for lip-sync driving even though it's too smeary for stems -- different, lower bar). Options:
  (1) InfiniteTalk v2v -- ALREADY BUILT (Wan2.1), /api/video/infinitetalk defaults to RoFormer
  isolation; (2) LTX LipSync -- native LTX-2.3, on-model with the MSR walk, but LoRA NOT downloaded
  yet. NEXT: test InfiniteTalk on the MSR walk + RoFormer vocal (what we have), then decide if the
  on-model LTX LipSync is worth downloading for final quality. GPU note: RoFormer (:5070) shares the
  3090 with ComfyUI -- serialize, do not run concurrently.
- Other LoRAs on box to evaluate: VBVR / OmniNFT / Motion-Track (motion), Crisp / Soft / Fantasy
  (look enhancers), plus a good single character LoRA for general use.
- Music-video assembly (grade + upscale via SeedVR2) -- see project_mv-grade-and-upscale memory.
