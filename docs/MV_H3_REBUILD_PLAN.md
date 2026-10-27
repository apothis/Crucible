# Music video pipeline - fresh rebuild on MiniMax H3

Status: **PROPOSAL 2026-10-27, awaiting the user's decisions; nothing deleted, nothing rendered.**
Built from a read-only inventory of the repo and box plus the upstream research pass in
`.mvwork/h3_research_2026-10.md` (every claim there carries a URL). Provenance tags: **[V]**
verified by us (source read, /object_info, measured), **[P]** upstream primary source, **[C]**
community report, **[M]** a number the cited author measured, **[U]** unverified, **[H]** hypothesis.

## 0. Brief (user, 2026-10-27)

- Start from fresh. Rip out the existing MV Studio and everything under it.
- MiniMax H3 is the ONLY video engine. Re-examine the model/quant we run, how we run it, newer
  workflows, LoRAs, Spectrum.
- Stills: OpenAI image generation. Claude writes the prompts; the user generates and hands back
  the images (links/files).
- Scripts: Claude writes them FROM THE SONG (the audio and its lyrics), not from a template.
- Cast: ONLY the female singer we already have an image of. No band members, ever.
- She lip-syncs where needed, not in every shot.
- The box is a stale snapshot (packs installed 2026-08-17 / 2026-09-01); it is NOT the reference.
  Upstream is.

## 1. What the box has today (snapshot, [V] 2026-10-27, read-only probes)

ComfyUI 0.41.0 / RTX 3090 25.8 GB / **34.2 GB system RAM**.

| Kind | Installed |
|---|---|
| UNET | `minimax_h3_fl2va_pruned_int8_convrot`, `minimax_h3_ref2va_pruned_int8_convrot` |
| LoRA | `minimax_h3_turbo_4step_ckpt500_comfyui_pruned`, `minimax_h3_turbo_v4_step600_ema_pruned_comfyui` |
| Text encoder | `qwen3vl_32b_minimax_h3_int8_convrot` |
| VAE | `minimax_h3_video_vae_fp16`, `minimax_h3_audio_vae_fp32`, `taeh3` |
| Packs | Aitrepreneur/ComfyUI-H3-Motion-Context-MultiRef-V3, seesee75/ComfyUI-MiniMaxH3-Director 0.2.2, LBH-123-AI latent upscaler, Fantastic PromptBuilder, xmarre Spectrum, NKD-Basic-Tools, MiniMaxH3-T1-Latent; core `MiniMaxH3FunControlNetApply` |

The one thing on the box we had never used and that changes the design: **MultiRef-V3's
"NEW - Music Video" workflow** (section 3). Everything else is to be re-decided from upstream.

## 2. The central mechanism: master-song latent masking [V from node source + architecture doc]

`MiniMaxH3SongMaskedAVContext` (MultiRef-V3):

1. takes the full master song once;
2. for clip N, computes the exact song slice from `clip_start_seconds` and the target frame count;
3. audio-VAE-encodes that slice straight INTO the clip's H3 audio latent;
4. sets the audio noise mask to 0 across the whole stream: **H3 never generates or re-renders the
   song**;
5. optionally copies the previous clip's video-latent tail (39 frames = 1.625 s = exactly 65 audio
   ticks) into the head of the new clip with video mask 0, so clip N+1 continues clip N's motion,
   camera and lighting;
6. `MiniMaxH3StreamLiveMusicVideoToVHS` decodes clip by clip, crossfades the 39-frame seams,
   streams to VHS and muxes the UNTOUCHED master song.

Why this matters to us: our measured failure mode ([[h3-audio-drift-seeds]]) was H3 re-rendering
the `<Audio 1>` reference and lip-syncing to its own drifted version (0.36-0.40 vs 0.87-1.00 on
clean takes). With the song protected in the latent there is nothing to drift. Also: no separate
ffmpeg assembly step inside a scene, no per-segment audio windows, no `H3_AUDIO_MIN` gate.

Timeline arithmetic [V]: `frames = 17k+5`; `advance = (frames - 39) / 24`;
`clip_N_start = (N-1) * advance`; `clips = 1 + ceil((song - frames/24) / advance)`.
Example globals in the upstream graph: 15 s raw -> 362 f, advance 13.458 s, 960x544, 8 steps on a
turbo LoRA, `res_multistep`/`simple`, SigmaShift 12/3.

Prompt shape per clip (upstream director prompt, [U]): six sections; `<Subject 1>` = the
performer from `<Picture 1>` (face) + `<Picture 2>` (full body/wardrobe); **no `<Audio 1>`**
(the song is in the latent, not a reference); lyrics only as `<Subject 1> (S1) sings ...:
<d>[English] exact words</d>`; continuation clips "continue first, then develop"; a generation
seam is not a cut; readable mouth framing on the vocal phrases that matter, wider elsewhere.

**Open until measured [H]:** how well mouth articulation tracks a PROTECTED song latent versus a
`fully_copy` audio reference. Upstream's example video claims it; we test it on one 2-clip chain
before building the UI around it (section 7, gate G1).

## 3. Design

### 3.1 Spine: scene chains, not independent shots

A video = an ordered list of **scenes**. A scene = one continuous H3 clip chain (song-masked,
39-frame continuation) covering a contiguous song window. Between scenes = a hard cut, where the
next scene starts fresh from its own opening still. Within a scene H3 may still write `[Shot 2] At
MM:SS.mmm` cuts in the prompt where the writer wants them.

- Per-scene chain length is the writer's choice from the song structure (a chorus = one chain, a
  verse = one chain, an instrumental = one chain of B-roll), capped by the 20-slot upstream limit
  if we reuse their graph, uncapped if we build the graph per chain (we will: one API graph per
  scene, N clips wired programmatically, exactly as `build_h3_ref2v` builds its lane today).
- Final assembly across scenes = ffmpeg concat of scene renders + the master song laid under at
  the scene offsets (each scene's audio is already the master slice, so a hard cut is sample-exact).
- Clip raw length: 10.125 s (243 f) default, 15.08 s (362 f) for long held performance chains.
  Shorter clips = more seams and more time lost to the 39-frame overlap.

### 3.2 Cast: one singer, two reference images per video

Canonical identity = the user's existing GPT reference (neutral studio portrait, black jumper).
Per video the user generates, from Claude-written prompts and THAT reference as the OpenAI
image input:

1. `<Picture 1>` face/identity portrait in the video's look (hair, makeup, light);
2. `<Picture 2>` full-body in the video's wardrobe, neutral pose, plain background;
3. one **opening still per scene** (the singer in the scene, composed as the first frame, 16:9,
   pre-cropped to the canvas aspect [V: first frame is stretched to canvas]);
4. optional person-free environment stills for B-roll scenes.

Claude writes all of these as one deliverable per video (the user's existing GPT-image filter
rules apply: one reveal per image, no brand names, [[cover-art-women-imagery]] default look).
No character library, no sheets, no costume/prop system: the per-video refs ARE the character.

### 3.3 Script writer: Claude, from the song

Inputs, all measured on the actual audio, none nominal:
- allin1 structure + downbeats from the box analyze service (:5075) [V, existing];
- `lyricalign.align_song` word/line timestamps with cover confidence [V, existing];
- lyrics text; song title/style; the user's one-line concept (optional).

Output (JSON, validated by code, timing never left to the LLM):
```
{ concept, look, wardrobe, picture_prompts: {face, body},
  scenes: [ { start, end, section, mood, setting,
              opening_still_prompt, lipsync: true|false,
              clips: [ { raw_seconds, prompt_sections: {summary, detailed_description,
                         overall_soundscape, non_diegetic_music}, lyrics_in_slice } ] } ] }
```
Code owns: the 17k+5 grid, clip starts, which lyric words fall in each slice (from the measured
alignment, with overlap), the fixed `subject_definitions` / `retention_analysis` boilerplate, the
sky-pin and one-motion pace clauses ([[minimax-h3-pace-control]]), and the no-`<Audio 1>` rule.
Claude owns: concept, scenes, where she sings to camera vs performs/walks/looks away vs B-roll,
camera language, the prose.

LLM path = the existing `claude_sub` route (Claude Max via CLI, [[claude-subscription-llm]]).

### 3.4 Model / recipe / speed (from the research pass; sources in `.mvwork/h3_research_2026-10.md`)

- **Engine: H3 stays.** No open weights newer than the 2026-07-28 release exist (no H3.1, no
  ref2va v2; H3 Max is hosted only) [P]. Nothing released since is credibly better for a singing
  performer on 24 GB: LTX-2.5 has no lip-sync path, LongCat-Video-Avatar 1.5 reports ~41 GB at
  480p and no ComfyUI support [C]. What changed is technique, not the model.
- **DiT quant: keep `pruned_int8_convrot`** for both fl2va and ref2va. Ampere has native int8
  matmuls and NO fp8 path (fp8_scaled is dequantised on load, no speed) [P: Comfy docs]; GGUF Q8
  is the same size with no measured quality win [U]. Pruned = AdaLN lookup table, "functionally
  equivalent" per Comfy [P].
- **Text encoder: A/B `qwen3vl_32b_minimax_h3_nvfp4_awq` (14.6 GiB) vs our int8 (25.3 GiB)** at a
  fixed seed. The Comfy-Org card says it does not need Blackwell [P]; on Ampere it stays packed
  with no emulation penalty [C]. This matters because the box has 34 GB RAM: int8 TE + int8 DiT
  = ~45 GiB on disk, and a 32 GB 3090 box was reported OOM-killed with default flags, fine with
  `--disable-pinned-memory` [C]. 64 GB RAM is the repeatedly named root fix.
- **Video VAE: add `minimax_h3_video_vae_int8_convrot`** (~2x decode on 0.36+) [P]. `taeh3`
  preview is already on the box.
- **Turbo LoRAs: ours are stale** (larryvrh v4, 2026-08-08). lightx2v now ships FL2VA 8-step
  v1.0/v1.1 768p and **Ref2VA 8-step v1.0/v1.1 768p** (repo 2026-10-20) [P]; 768p variants want
  shift 6/3. Field practice for SINGING cuts is base Ref2VA at 20 steps, turbo only where no
  mouth is read [C]; but that advice predates song-lock (it was about H3's generated audio being
  weak at 8 steps, which no longer matters when the audio is discarded). So: hunt on Ref2VA
  8-step v1.1, finish on base 20 steps, and test whether the 8-step finish is good enough on a
  singing close-up judged by watching (gate G1b).
- **Speed: try comfy-kitchen INT8 attention (`ModelAttentionBackend`, in core) before Spectrum.**
  Reported identity and lip correlation unchanged, 1.3-1.8x on 40/50-series [M]; no 3090 number
  [U]; needs a cu130 torch build or it silently disables (check the log for `int8_tensorwise`).
  **Spectrum stays off**: the author calls it an approximate accelerator, no version removes the
  composition change, and it is worse with turbo LoRAs [P]. Sage: in-graph only, and one Ampere
  report of corruption past ~92k tokens (1344x768x362) [C]; our 1280x736x362 is ~84k, close.
- **Ref2VA quality caveats to design around:** Ref2VA has lower base quality than FL2VA [C];
  faces distort on wide shots, close/medium are fine [C]. Hero singing = medium/close framing;
  wide shots are B-roll or back-to-camera.
- **Fun ControlNet Union 2.0** (core `MiniMaxH3FunControlNetApply`, int8 patch ~4.2 GB): pose /
  depth / canny driven video, works with ref2va [P]. Not in the first build; noted for driving
  her body from a reference performance clip later.
- **Character LoRA:** all four trainers support H3 now (musubi-tuner "teacher matching" for
  identity) [P], but nobody documents a 3090 + 34 GB run. Not needed if two GPT refs hold
  identity (gate G1 judges this); parked.

### 3.5 UI (minimal, after G1 passes)

One tab, "Music Video": pick song -> Write script (Claude) -> review scenes/clips (editable prose,
read-only timing) -> paste/upload the GPT stills into their slots -> Render scene (hunt N seeds on
the draft recipe at finish resolution, pick, finish) -> Assemble. State in `useDrafts("mv")`.
No timeline editor, no character tab, no per-block inspector.

## 4. Rip-out list (PROPOSED - needs the user's yes before any deletion)

### 4.1 Delete - frontend
`web/src/MVStudio.tsx`, `mvmodel.ts`, `mvui.tsx`, `MVTimeline.tsx`, `H3Studio.tsx`,
`H3SegTimeline.tsx`, `ShotStudio.tsx` (already orphaned), `ShotEditor.tsx`, `ShotTimeline.tsx`,
`LtxDirectorEditor.tsx`, `vendor/ltxdirector/` (596 K vendored GPL JS), `Characters.tsx`.
`App.tsx`: the MV Studio + Characters nav entries/routes (lines 8, 12, 46-47, 61, 201, 280, 284).
`api.ts`: the MV-only entries (64-84, 139-160 minus videoStill/videoI2V/videoLipsync/
videoInfinitetalk).
**Decision for the user:** the old "Video" tab (`forms.tsx` VideoForm 2420-2581 = Z-Image still /
Wan i2v / Wan S2V / InfiniteTalk gate). H3-only means it goes too; its still sub-tab is the only
piece anyone might miss (Z-Image/Krea2 stills stay reachable from the YouTube tab).

### 4.2 Delete - backend `app.py`
- `/api/comfy/{path}` proxy (880).
- `/api/video/`: free_models, char_still, loras, vace, ltx_t2v, h3_ref2v (rebuilt new), ltx_i2v,
  ltx_msr, ltx_flf, ltx_keyframe, ltx_fflf, crop_still, ltx_retake, svi_i2v, retime, upscale
  (SeedVR2), regrade, ltx_lipsync, s2v_wrapper, assemble_chain; and if the Video tab goes: i2v,
  lipsync, infinitetalk.
- all `/api/mv/*` (generate_lut, look_library, script, h3_script, h3_compile, h3_audio_check,
  h3_recompile, h3_snap_edges, h3_voicemap, h3_voicefix, grades, assemble) - rebuilt new.
- `/api/characters*` (4545-4783) and `/api/projects/{key}/video` (4797/4806) + helpers.
- config flags `_msr_available`, `_qwen_available`, `_vace_available`, `_ltx_quants_present`,
  `_ltx_available`; `_isolate_vocal_bytes`, `_probe_nframes/_probe_fps` if nothing else uses them.

### 4.3 Delete - backend `video.py`
`build_qwen_char_still`, `build_vace_ref2v`, `build_i2v`, `build_s2v`, `build_s2v_wrapper`,
`build_infinitetalk_v2v`, `_build_ltx` + all `build_ltx_*`, `build_seedvr2_upscale`,
`build_darkroom_grade`, `build_vcg_lut`, `build_vcg_apply`, `build_svi_i2v`, `build_h3_ref2v`
(rebuilt new); constants WAN, QWEN, VACE, LTX, DARKROOM/VCG, SVI.

### 4.4 Delete - backend `musicvideo.py`
Everything from line 658 to the end (LTX + H3 script writers, compile_h3_prompt, voice casting,
segment builders) plus GRADES/grade_names/audio_match/assemble (lines 25-253). The YouTube
helpers (`_ffmpeg`, `audio_duration`, `still_video`, `loop_tail_report`, `seam_loop`,
`trim_frames`, `seamless_loop`, `fullbleed_video`, `card_video`) stay in place.

### 4.5 Delete - docs, installers, scratch
- docs: `LTXDIRECTOR_PIPELINE_PLAN.md`, `SHOT_EDITOR_MODEL.md`, `SHOT_STUDIO_FFLF_PLAN.md`,
  `MV_AI_GRADING_PLAN.md`; root `MUSIC_VIDEO_PLAN.md`, `VIDEO_PIPELINE_NOTES.md`.
  `MINIMAX_H3_PLAN.md`: keep ONLY as history; its §11 same-still loop recipe is what the YouTube
  living cover depends on and moves to `docs/H3_LIVING_COVER.md` before the file is trimmed.
- installers: `download_ltx_models.bat/.py`, `install_ltx_nodes.bat`,
  `download_video_models2.bat/.py`, `download_infinitetalk_models.bat`.
- `.mvwork`: the h3_*/fflf_*/s20_msr*/shot6_* scratch (list on request; 9.5 GB folder, gitignored).
- `library.db` `characters` rows: Susan, Sally, Bassist, Guitarist, Bob (Selene kept until the new
  cast config exists, then the table is unused).

### 4.6 KEEP (shared with the YouTube tab or the box)
`/api/video/still` (Krea2 + Z-Image), `krea2_edit`, `h3_t2v`, `h3_i2v` (+ `_h3_dispatch`,
`_h3_tail`, `_h3_loaders`, `_crop_image_bytes_to_aspect`), `flashvsr` + `build_flashvsr_upscale`,
`_lib_video_path`, `_lib_image_path`, `_trim_audio_window`, `_audio_name`, `lyricalign.py`,
`analyze_py.py`, library modes `videostill`/`videoclip`/`musicvideo`, `install_video_nodes.bat`,
`KREA2-IDENTITY-EDIT_INSTALL.bat`, `SAGEATTENTION_INSTALL.bat`, `download_video_models.*`
(Z-Image). Loosh Woggins tooling is untouched (ffmpeg/Remotion + manual lyricalign).

## 5. New code (small, all new files)

- `backend/mv_h3.py`: graph builder for one scene chain (N x [ReferenceToVideo ->
  SongMaskedAVContext -> sampler] -> StreamLiveMusicVideoToVHS), hunt/finish, scene assembly.
- `backend/mv_writer.py`: Claude script writer (inputs 3.3, schema validation, slice-lyric table,
  prompt compiler with the fixed sections).
- `web/src/MusicVideo.tsx`: the one tab.
- `docs/MV_H3_REBUILD_PLAN.md` (this file) becomes the living spec; `docs/H3_LIVING_COVER.md`
  holds the YouTube loop recipe.

## 6. Which song-lock implementation, and why (research section 2.2)

The song-lock is now mainstream, with four implementations:

| | Where | API-driveable from our app? | Notes |
|---|---|---|---|
| Core per-token AV noise masks + `MiniMaxH3AddGuide` | ComfyUI 0.34+ (box has 0.41) [P] | yes, but we would re-implement the slice/grid arithmetic | the primitive everyone wraps |
| `MiniMaxH3SongMaskedAVContext` | seitanism MultiRef, **Update 9** 2026-09-05 [P]; box has Aitrepreneur's Update-6-era fork | **yes**: plain node inputs (latent, master audio, clip_start_seconds, context_length, source_latent) | exact slice arithmetic + 39f continuation + VHS streamer; GPL-3 |
| H3 Director v1.0.0 "lock (exact audio)" + Music-video template | seesee75, 2026-10-26 [P] | poorly: the node's `timeline_data` is a UI-managed JSON blob | measured 0.99 audio fidelity 5/5 vs 4/6 for prompt-level `fully_copy` [P][M]; Whisper lyric timing built in (we have lyricalign) |
| LongMedia `audio_mode=lip_sync` | vizart-vj 0.6.63 [P] | partly | needs image_1; adaptive low-VRAM routing |

**Decision: build our per-scene API graph on seitanism's `SongMaskedAVContext` + `StreamLive
MusicVideoToVHS` (Update 9), replacing the stale Aitrepreneur fork.** It is the only wrapper whose
inputs are plain node widgets, so `backend/mv_h3.py` can emit it the way `build_h3_ref2v` emits
its lane today, and the app stays the driver (the ComfyUI UI is never used, per the project
rule). The Director's measurement is the evidence the mechanism works; its UI is not our path.

**First test variable (G1): lock the FULL MIX vs the isolated LEAD-VOCAL stem** (box UVR at
:5060 splits it). Two independent field reports favour the stem for mouth accuracy, with the
master mixed back at the end [C]; nobody has published a controlled A/B [U]. Prompt practice
that travelled with it: sing only when the voice is audible, "hold one vowel shape on sustained
notes", close/relax in silence.

## 7. Gates (each needs the user's go before the GPU run - [[ask-before-firing]])

- **G0 box update (no GPU, ~25 GB of downloads via :5080):**
  - ComfyUI 0.41 -> 0.43 (optional; core already has the masks + AddGuide).
  - custom_nodes: replace `ComfyUI-H3-Motion-Context-MultiRef-V3` with seitanism Update 9.
  - models: `qwen3vl_32b_minimax_h3_nvfp4_awq` (14.6 GiB), `minimax_h3_video_vae_int8_convrot`
    (~3 GiB), lightx2v `ref2v_turbo_8step_v1.1_768p` + `fl2v_turbo_8step_v1.1_768p` (~2 GB each),
    all from Comfy-Org / lightx2v on HF (ungated).
  - `/object_info` confirms every node + file before anything renders.
- **G1 mechanism test (the gate that decides the design):** ONE 2-clip song-masked chain,
  singer singing a known chorus line to camera, medium close-up, from two GPT refs. Variables, one
  at a time, same seed: (a) full-mix lock vs vocal-stem lock; (b) Ref2VA 8-step v1.1 finish vs
  base 20 steps; (c) nvfp4 vs int8 text encoder. Judge: the user watching (mouth, seam, identity)
  plus the existing envelope check of the clip audio against the master slice as a sanity floor.
  If the mouth does not track a protected latent, fall back to `fully_copy` `<Audio 1>` per clip
  and keep the drift score.
- **G2 writer test:** one full script for one song, read by the user, no render.
- **G3 one full scene chain** (3-5 clips) rendered and assembled; then the UI.

## 8. G1 proposal (2026-10-27, awaiting the user's go - nothing fired)

**Song (proposed, user's call):** "Where the Tides Remember" (master "de-glitched"), the song
the August H3 lip-sync work was measured on, so we have a drift baseline to beat. Window = the
FIRST CHORUS, nominally 82-110 s ("Where the tides remember, where the evening never ends ...");
the real onset comes from `lyricalign` before anything renders, never the nominal arrangement.

**Clip plan (upstream arithmetic):** raw clip 243 f = 10.125 s; context 39 f; advance 204 f =
8.5 s. Two clips cover chorus_onset-1.0 s .. +17.6 s: clip 1 = the first two chorus lines sung to
camera, clip 2 = continuation across the seam into lines three and four. Canvas 960x544 (the
upstream example's), Ref2VA model, `res_multistep`/`simple`, shift 12/3, seed pinned.

**Variables, one at a time, same seed (each pair = 2 chains):**
- A. song lock: full master mix vs isolated lead-vocal stem (box UVR :5060 splits it; master
  muxed back either way). Decides what the writer feeds the latent.
- B. recipe: lightx2v Ref2VA 8-step v1.1 768p (shift 6/3) vs base 20 steps. Decides whether the
  finish can be turbo now that H3's own audio is discarded.
- C. text encoder: nvfp4_awq vs int8 (after `download_minimax_h3_v2.bat` has run). Decides the
  RAM budget.
Order A -> B -> C; each later pair runs on the winner of the earlier one.

**References (two GPT images from the canonical portrait, no opening still - the upstream chain
starts from ReferenceToVideo with refs only; scene opening stills come later via keyframes):**
- `<Picture 1>` face/identity: "Close three-quarter portrait of the woman in the reference
  photo, same face and long dark wavy hair, pale skin, dark smoky eye makeup, a thin silver chain
  at her throat, high-collared black lace blouse, soft cool window light from the left against a
  dark grey studio backdrop, calm direct gaze, mouth closed, photoreal, 85mm, shallow depth of
  field, 3:4."
- `<Picture 2>` full body / wardrobe: "Full-length studio photograph of the same woman from the
  reference photo, standing relaxed, arms at her sides, facing camera, in a fitted black velvet
  corset laced at the front over a white linen chemise slipped off one shoulder, long black
  skirt, bare feet, plain mid-grey seamless backdrop, even soft light, photoreal, 50mm, 2:3."
  (One reveal per image per the GPT filter notes; no brand names.)

**What has to be built first (no GPU):** `backend/mv_h3.py` = per-scene chain graph builder
(N x [ReferenceToVideo -> SongMaskedAVContext (+ AudioVAECompatibility) -> sampler] ->
StreamLiveMusicVideoToVHS, master audio uploaded once by content hash) + a `POST
/api/video/h3_chain` route using the existing `_submit_video`. GPT refs enter as library stills
through the existing still import.

**Cost estimate [H], extrapolated from our 1280x736 measurements:** ~4 min per turbo clip and
~10 min per base clip at 960x544x243 f, so variable A = ~16 min, B = ~28 min, C = ~16 min.

**Judge:** the user watches each chain (mouth on the words, the 10.1 s seam, identity against the
two refs). Floor check: envelope correlation of the clip's own decoded audio vs the master slice
should be ~1.0 by construction - if it is not, the lock did not take.
