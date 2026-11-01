# Music video pipeline - fresh rebuild on MiniMax H3

Status: **PROPOSAL 2026-10-27, awaiting the user's decisions; nothing deleted, nothing rendered.**
Built from a read-only inventory of the repo and box plus the upstream research pass in
`.mvwork/h3_research_2026-10.md` (every claim there carries a URL). Provenance tags: **[V]**
verified by us (source read, /object_info, measured), **[P]** upstream primary source, **[C]**
community report, **[M]** a number the cited author measured, **[U]** unverified, **[H]** hypothesis.

> The general rulebook distilled from this log is `docs/MV_H3_TECHNIQUES.md` (every rule with
> provenance, no song-specific content). This file stays the per-video record of what was fired
> and what it looked like.

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

## 8. G1 proposal (rewritten 2026-10-27 for the user's pick; awaiting the go - nothing fired)

**Song (user's choice):** "Never Mine" - the Suno take that went into the YouTube video (library:
Suno import 2026-10-15, take 1; 258 s; style "70 bpm, A minor, slow heavy metal ballad", verses
half-spoken and raspy, choruses soft clean confessional, final chorus a belt). Lyrics = the
project's Suno draft.

**Measured timing (lyricalign, Whisper small on the Mac, cover 0.97, 235/242 words placed):**
intro to 13.8 s; verse 1 13.8-42.0; verse 2 43.2-70.8; **chorus 1 70.78-89.72** (every line cover
1.00); verse 3 100.7-127.6; chorus 2 128.4-146.1; solo; final chorus 185.3-220.7; outro 223.6.
Chorus 1 lines: 70.78 "What I was, what I'd be" | 74.56 "Every piece decided for me" | 78.22 "And
they called it my life, called it fine" | 82.46-89.72 "But it never was mine, it never was mine".

**Window = chorus 1** (assumption, say if you want the belted final chorus instead): the soft
clean delivery is the HARDER lip-sync case (small mouth motion, long vowels), so a pass here means
more than a pass on the belt. The upstream director notes "hold one vowel shape on sustained notes"
applies directly to the "mine" sustains.

**Clip plan (the upstream example's own globals: 15 s raw = 362 f, 39 f context, 960x544):**
advance = 323 f = 13.458 s.
- clip 1: 69.80 -> 84.88 s. Lines 1-3 sung to camera, line 4 begins ("But it never was ...").
- clip 2: 83.26 -> 98.34 s. Continues through the seam INSIDE line 4 (a sustained phrase crossing
  the seam = the continuity test), the repeat "it never was mine" to 89.72, then 8.6 s of
  instrumental before verse 3: she stops singing, breath, holds the gaze, slow camera drift.
Seam = 83.26-84.88 s. Ref2VA model, `res_multistep`/`simple`, shift 12/3, seed pinned, 8 steps.

**Variables, one at a time, same seed (each pair = 2 chains):**
- A. song lock: full master mix vs isolated lead-vocal stem (box UVR :5060; master muxed back
  either way).
- B. recipe: lightx2v Ref2VA 8-step v1.1 768p (shift 6/3) vs base 20 steps.
- C. text encoder: nvfp4_awq vs int8 (after `download_minimax_h3_v2.bat` has run).
Order A -> B -> C; each later pair runs on the winner of the earlier one.

**Look for this song:** the lyric is a woman raised in a "house of rules" taking the hammer to it.
Setting for the chain: a dim, formal old parlour at night, dust in a single shaft of cold window
light, dark wood and faded wallpaper; she stands close to camera and sings the confession quietly.

**References (two GPT images from the canonical portrait; the user generates and hands them back):**
- `<Picture 1>` face/identity: "Close three-quarter portrait of the woman in the reference photo,
  same face and long dark wavy hair, pale skin, dark smoky eye makeup, lips slightly parted, a
  thin black velvet choker, cold window light from the left in a dim Victorian parlour, dark wood
  and faded wallpaper softly out of focus behind her, calm direct gaze, photoreal, 85mm, 3:4."
- `<Picture 2>` full body / wardrobe: "Full-length photograph of the same woman from the reference
  photo standing in a dim Victorian parlour, facing camera, arms relaxed, in a fitted black velvet
  corset laced low at the front over a thin white chemise slipped off one shoulder, long black
  skirt, bare feet on dark floorboards, one shaft of cold window light, photoreal, 50mm, 2:3."
  (One reveal per image per the GPT filter notes; no brand names.)

**What gets built first (no GPU):** `backend/mv_h3.py` chain builder + `POST /api/video/h3_chain`,
mirroring the Update 9 example's wiring exactly: UNETLoader(ref2va) -> ModelAttentionBackend ->
LoraLoaderModelOnly -> SigmaShift -> per clip [ReferenceToVideo -> SongMaskedAVContext(master
audio, clip_start, 39, source_latent = previous sampler) -> BasicGuider -> SamplerCustomAdvanced]
-> StreamLiveMusicVideoToVHS(all samplers, master audio) -> FinalizeVHSOutput; audio VAE routed
through AudioVAECompatibility; master song uploaded once by content hash. Variable C stays a
constant swap (H3_CLIP). The refs enter as library stills through the existing still import.

**Cost estimate [H], extrapolated from our 1280x736 measurements:** ~6 min per turbo clip and
~15 min per base clip at 960x544x362 f, so A = ~24 min, B = ~42 min, C = ~24 min.

**Judge:** the user watches each chain (mouth on the words, the seam inside line 4, identity
against the two refs). Floor check: the chain's own decoded audio vs the master slice should be
~1.0 by construction; if not, the lock did not take.

### 8.1 G1 A1 result (2026-10-27 20:09, full-mix lock, turbo 8-step v1.1 768p, seed 20261027) [V]
- Rendered 28.542 s = the planned 685 f exactly; 979 s wall for the pair (~8 min per 362 f
  turbo clip at 960x544; my 6 min estimate was low). Filed in the library as a video clip.
- **Lock: envelope r = 0.998 at 0.00 s lag** against the song window (August clean takes
  0.87-1.00, drifted 0.36-0.40). The song is the audio; nothing was re-rendered.
- Identity and wardrobe hold in every sampled frame (0.5 / 2 / 5.5 / 9 / 12.9 / 14.4 / 15.3 /
  16.6 / 19.5 / 23 / 27.5 s). Silent tail obeyed (lips closed, eyes down, gaze back at the end).
- Seam metric: frame-to-frame motion in the blend region mean 1.54 / max 2.64, inside the clip
  bodies' range (1.48-1.50); no single-frame jump at the join.
- Defects (stills; user to confirm by watching): (1) the "hand's width over fifteen seconds"
  push overshot - medium at 0.5 s, extreme close-up by 12.9-15.3 s (H3 pace, prompt-level);
  (2) clip 2 reset from that extreme close-up to a medium shot within ~1.5 s after the
  protected prefix, and the choker appears only in clip 2. Cause [H]: my clip 2 text asserted
  "the same medium close-up" before clip 1 existed, so text and prefix disagreed. Fix for the
  pipeline: render chains sequentially and write clip N+1 from clip N's actual last frame (or
  keep the camera static within a chain); never pre-assert the incoming framing.
- Lip-sync: not judged from stills - user watches
  `.mvwork/g1_refs/nevermine_G1_A1_fullmix-lock_turbo8_seed20261027.mp4`.
- Unverified: whether comfy-kitchen INT8 attention engaged (launcher writes no log).
- USER VERDICTS (watching, 2026-10-27): (a) the push-in was NOT too extreme - defect (1) above is
  withdrawn, a still-based misread; (b) a black choker APPEARS on her neck at the seam. Picture 1
  (face ref) wears the choker, picture 2 (body ref) does not; clip 1 followed picture 2, clip 2
  followed picture 1 and the retention line that named it. RULE: the references for one subject
  must agree on EVERY visible detail (jewellery, sleeves, hair parting); a disagreement is a coin
  flip per clip and the seam exposes it. Fix = regenerate one ref so both match, and never name
  in the prompt a detail that only one reference shows.
- USER VERDICT (watching, 2026-10-27): **"The mouth looked fine"** on A1 - full-mix lock, turbo
  8-step v1.1, 960x544. G1's core question is answered in the affirmative for the full-mix lock;
  A2 (vocal-stem lock) is now a refinement test, not a gate.
- A1b fired 2026-10-27 ~20:40: identical prompts/seed, picture 2 regenerated WITH the choker so
  both refs agree (library still "Never Mine - Selene body ref v2 (choker)"). Variable = refs only.

### 8.2 G1 A1b result (2026-10-27 20:48, same prompts/seed, refs now agree on the choker) [V]
- 28.542 s, lock envelope r = 0.998 @ 0 lag, 902 s wall for the pair.
- Choker present in every sampled frame from 0.5 s to 27.5 s: the seam disagreement is gone.
  The reference-consistency rule holds.
- With IDENTICAL text, clip 2 this time KEPT the close-up after the seam (16.6 / 19.5 s) instead
  of resetting to a medium shot as A1 did. So the A1 framing reset was not caused by the prompt
  text alone; it is per-render variance between clips (refs changed, seed held). The sequencing
  rule (write clip N+1 from clip N's real last frame) still stands as the way to make the text
  agree with the prefix, but it is not a guaranteed fix. Marked [H] for the cause.
- Unprompted motion at the end: a hand rises to her face (~27 s). Minor; "nothing else moves"
  does not fully pin the hands.
- Lip-sync and seam feel: user to watch
  `.mvwork/g1_refs/nevermine_G1_A1b_fullmix-lock_turbo8_refsmatch_seed20261027.mp4`.

## 9. G2 result (2026-10-27 21:08) - the writer works end to end [V]
- `POST /api/mv/write` on Never Mine: Opus (claude-opus-5-5) at medium effort via the CLI,
  386 s for pass 1 + ten pass-2 calls. 12 measured sections -> 10 scenes -> 22 clips, all
  prompts pass the mechanical rule check (after making the check performance-aware).
- The plan is a real video, not a lip-sync reel: environment intro (push toward the empty red
  armchair), verse in the chair, corridor B-roll on "the walls came up", chorus confession at the
  window, a cracking porcelain mask on the instrumental, the quarry with the hammer, chorus two in
  rain, the whole guitar solo spent carving her name, the belted final chorus back in the
  storm-wrecked parlour, an extreme-wide walk-away to the hard stop. 6 singing scenes / 4 B-roll.
- Per clip the model kept every rule: one camera move with a distance over the whole scene and
  a per-clip share, one main motion with seconds, hands and hair stated, sky pinned, lyrics
  split at clip edges mid-phrase, continuation openings on clip 2+, no <Audio N>.
- First-run bug fixed: the model returned `sections` as the string "[0, 0]"; parser now accepts
  strings and the schema shows an int array; raw answers are kept under
  .mvwork/lyricalign/writer_raw/.
- Script: `.mvwork/mv_scripts/never_mine_20261027_210758.md` (+ .json). USER TO READ.
- Gaps for G3 (render a scene): (a) the last scene's chain overshoots the song end
  (262.98 s on 258.0 s) and /api/video/h3_chain refuses it - pad the song window with silence for
  the final scene (a padded master is a real waveform to the node) and trim at assembly;
  (b) environment-only scenes (1, 5) still go through Ref2VA with the singer marked
  weak_reference - add an FL2VA (no refs) lane to mv_h3 for them; (c) the subject_definitions
  boilerplate repeats the refs' parlour sentence in quarry scenes - make `shared` refs-only.

## 10. G3 result (2026-10-27 21:50) - one scene renders as a 3-clip chain [V, motion pending]
- Scene 2 "Taught to Bow" (13.34-42.72 s) fired through `POST /api/mv/render_scene` with
  `.mvwork/g1_refs/g3_scene2_request.json`: Ref2VA, turbo v1.1 8-step shift 6/3, 960x544,
  seed 20261027 (+101 per clip), face + body(v2, choker) references, song window 13.34-55.34 s.
- Box time 1413 s for 3 x 362 f (about 7.8 min per clip, same as G1). Filed in the library as a
  `videoclip` row; copy at
  `.mvwork/g1_refs/nevermine_G3_scene2_taught-to-bow_3clip-chain_turbo8_seed20261027.mp4`.
- Measured: 1008 frames / 42.000 s at 24 fps = the planned span exactly (362 + 2 x 323);
  envelope r = 1.000 at 0 lag against the master at 13.34 s (floor check passes).
- Stills read only (contact sheet `nevermine_G3_scene2_contact_sheet.png`, 3 s steps plus both
  seams): identity and choker hold across all three clips; framing continuous at 13.46 s and
  26.92 s; push-in goes medium close-up -> close-up with eyes raised to the lens on the last
  line, as scripted. A thin diagonal white streak enters top-right in the last ~4 s [H: unprompted
  object]. Lip-sync, seam feel and pacing: USER TO WATCH.
- Note for assembly: the chain runs to 55.34 s song time; the scene ends at 42.72 s, so the
  final 12.6 s are overshoot to trim (or render the last clip at a shorter grid length).
- Decided with the user after G3 fired: per-scene OpenAI images are not needed for identity,
  but the full video needs the PLACE pinned across scenes. Next build = location references
  (one person-free GPT still per location: parlour, corridor, mantelpiece, quarry, wrecked
  parlour) passed as <Picture 3> "environment, fully_preserved" in the chain; first-frame
  keyframes (AddGuide) reserved for environment-only and composition-critical openings. The
  next render after this is a second parlour scene so the room match can be judged.

## 11. Location references (built 2026-10-27 22:30, unrendered)
- Why: identity is pinned by her two pictures; nothing pinned the PLACE, so each scene would
  reinvent the parlour and the quarry. Per-scene stills are not needed; per-LOCATION stills are.
- Writer: pass 1 now returns `locations` [{id, name, description, still_prompt}] and each scene a
  `location` id (SCENE_SYSTEM: few locations, returned to; a detail of a room is its own
  location). `normalize_locations` guarantees every scene points at a row (derived rows for
  strays). The clip pass is told the place is a referenced subject and to call it by name.
- Compiler: `compile_prompt(..., env=row)` writes `<Subject 2> is the location, <name>, shown in
  <Picture 3>` + a fully_preserved retention line (layout/furniture/materials/light direction
  kept; camera, weather, time of day, state changes free). Environment-only scenes put the
  location alone as <Subject 1>/<Picture 1> on the Ref2VA weights; with no location still they
  fall back to FL2VA with a no-subjects header (the old prompt wrongly cited pictures there).
  `compile_scene(script, scene, with_location)` recompiles from each clip's stored text.
- Route: `POST /api/mv/render_scene` takes `location_refs` {id: library still} or
  `location_ref_id`; uploads face, body, location in that order and recompiles the prompts to
  match; records location + still id in the job params. Chain builder unchanged (any picture list).
- Never Mine: `.mvwork/mv_scripts/never_mine_20261027_210758_locref.{json,md}` = the G2 script +
  five locations (parlour 1,2,4; corridor 3; mantelpiece 5; quarry 6,7,8,10; wrecked_parlour 9),
  refs.shared trimmed to her alone (no parlour sentence in quarry scenes), all 22 prompts
  recompiled, 0 rule problems. Still prompts: `.mvwork/g1_refs/g4_location_still_prompts.md`.
- Test next (G4): scene 4 "Confession at the Window" with the parlour still (a second parlour
  scene) and judge whether the room matches scene 2. [H] H3 honours an environment picture as a
  fully_preserved subject in the chain - supported by the reference guide and the August
  text-environment test, not yet measured in a chain.

## 12. Scene 2 verdict and the shorter-take rules (2026-10-27 23:00)
- User watched scene 2: (1) a thin white streak enters top-right at ~36 s (clip 3) and grows to
  the end - a real defect, cause unknown (re-render); (2) a single 42 s take with one slow push-in
  is too long for a verse. The background objects softening during the push-in is depth of field,
  not drift (user; my earlier read withdrawn).
- Writer rules changed (SCENE_SYSTEM): performance takes 8-15 s by default, never over 28 s;
  B-roll up to 45 s; a verse/chorus = two or three SETUPS in the same location, cut between
  (the location reference keeps the room); camera moves sized to the take (10-15 cm per 15 s).
- Scene bounds are now `from`/`to` seconds snapped to CUT POINTS = section edges + 0.5 s before
  each sung line (`cut_points`, listed in the section table the writer sees); `normalize_scenes`
  forces contiguity and still accepts the old `sections` ranges.
- New helpers: `write_scene_clips` (pass 2 for one scene, used by write_script) and
  `split_scene(script, index, at, second)` (cut a scene at a cut point, new fields for the second
  half, renumber).
- Never Mine: scene 2 split at 28.22 s -> "Taught to Bow (chair)" 13.34-28.22 (12 cm push-in,
  eyes down, one 362-f clip) + "Taught to Bow (profile)" 28.22-42.72 (static three-quarter from
  the lamp side, head turns to the lens on "wear the mask", one 362-f clip). Clip pass re-run on
  both (Opus medium, ~1 min), 0 problems. Script: `.mvwork/mv_scripts/never_mine_20261027_split.
  {json,md}` (11 scenes / 21 clips). The G2 scenes other than these two keep their long takes;
  a full re-run of the writer under the new rules is the clean fix for the rest.
- G4 (not fired): render both halves with the parlour still as <Picture 3>; judge room match
  across the cut and against scene 2's old render, seam-free single clips, and whether the streak
  recurs. Requests: `.mvwork/g1_refs/g4_scene1_chair_request.json`, `g4_scene2_profile_request.json`
  (location_ref_id to fill in once the parlour still is in the library).

## 13. Vocal stems (built 2026-10-27 23:10) and the G5 writer re-run
- Stems: `backend/mv_stems.py` + `POST /api/mv/stems {audio_id}` runs BS-RoFormer SW on the box
  (:5070, ComfyUI freed first) and caches `.mvwork/mv_stems/<id>.vocals.wav`. Uses: (1) lyric
  alignment on the stem (`/api/mv/write` align_on auto|vocals|mix; auto = stem when cached) -
  dense metal buries the words for Whisper, the stem does not; (2) the chain's `lock` option
  (mix [default] | vocals | vocal_forward = mix + stem at +6 dB): what the clip latents hear.
  The muxed master is ALWAYS the mix window (node 40); the lock audio is a second LoadAudio
  (node 41) into the song-masked context nodes. User's point kept as the default: the whole
  song in the latents may steer the rest of the imagery onto the beat; the stem locks are
  switches to test on dense songs, not a new default.
- G5 writer re-run under the shorter-take rules (520 s, Opus medium): 20 scenes / 24 clips,
  `.mvwork/mv_scripts/never_mine_20261027_231816.{json,md}`. 18 scenes are single clips of
  6-14.5 s; only the solo (38 s, 3 clips) and the outro (31 s, 3 clips) chain. Locations it
  chose: parlour (6 scenes), parlour_window (2), hallway (2), quarry (8), stone_detail (2); it
  dropped the storm-wrecked parlour and put the final chorus in the quarry. Stills on hand:
  parlour, corridor, mantelpiece, quarry, wrecked_parlour (library ids in
  `.mvwork/g1_refs/locations/location_refs.json`). Mapping for a render of this script:
  parlour_window -> parlour still, hallway -> corridor still, stone_detail -> quarry still
  (the location retention line leaves framing free); or generate the two new stills.
- Stems RUN 2026-10-27 23:20: POST /api/mv/stems on Never Mine = 81 s on the box (RoFormer,
  258 s stem). Alignment on the stem vs the mix (Whisper small, same lyrics): cover 0.967 vs
  0.971, 29 lines both; |start delta| median 0.08 s, p90 0.69 s, max 1.48 s; 10 line edges move
  by more than 0.3 s, mostly chorus openings later by 0.4-1.5 s and one held note ending 1.26 s
  later. Stem-level check inside the disputed gaps (sung reference -15 dBFS): "Till the only
  thing" (-36), "take the hammer" (-30) and the final "What I was" (-42) gaps are near-silent on
  the stem, so the MIX alignment started those lines early on an instrument sound and the stem is
  right; "never was mine" 146.1-147.4 is -14 on the stem, so the stem's later END is right (the
  mix cut a held note short); "What I feel" 200.0-200.7 is -13 on the stem (continuous phrase),
  ambiguous, sub-second. Verdict: the stem is the better alignment source even on this easy song
  (fewer early starts, held notes kept); align_on=auto keeps it the default whenever a stem is
  cached. The G5 script was aligned on the mix (written before the stem existed).
- G4 FIRED 2026-10-27 23:22: split-script scenes 1 (chair) and 2 (profile), each one 362-f clip,
  Ref2VA turbo, seed 20261027, pictures face + body + PARLOUR STILL (<Picture 3>), lock = mix.
  Box confirmed 3 LoadImage, 3 refs wired, <Picture 3> in the prompt. Jobs c4b296f0 / d523e14b.

## 14. G4 result (2026-10-27 23:37) - location still in the chain [V, motion pending]
- Both halves filed: chair c4b296f0 (13.34-28.42 s) and profile d523e14b (28.22-43.30 s), each
  362 f / 15.083 s, envelope r = 1.000 at 0 lag, ~7.8 min each. Copies
  `.mvwork/g1_refs/nevermine_G4_scene{1_chair_pushin,2_profile_static}_parlour-still_seed20261027.mp4`,
  sheet `nevermine_G4_contact_sheet.png` (both halves + old scene 2 + the parlour still).
- Stills read: the ROOM AGREES across the cut and with the still - same tall multi-pane sash
  window with bare trees behind her, same brass oil lamp on the side table at her right, same
  red wingback, dark panelled wall. The old scene 2 (no location still) had a different lamp
  set and window; the location picture did its job [H until the user watches].
  Geometry is not literal: the still has the chair across the room from the window, the renders
  put the window behind her; the model kept the furniture identity, not the floor plan.
- Profile half: static framing holds for all 15 s, three-quarter from the lamp side as scripted,
  identity consistent, eyes come to the lens near the end. No streak in the stills.
- Chair half: the "12 cm over 15 s" push-in was NOT honoured - it starts as a medium shot (whole
  chair, lamp, window) and ends as a tight close-up on her face, the same travel the 42 s take
  made. Pattern with scene 2: H3 drives a prompted push-in to a close-up regardless of the
  distance figure [H, two renders]. Mitigations to test: static takes for performance (the
  profile half shows the hold works), or anchor the END framing in words ("ends still a medium
  close-up, shoulders in frame") instead of a distance.
- USER TO WATCH both (lip-sync, the cut at 28.22 s, whether the chair push reads too fast).

## 15. Camera movement research (2026-10-28 00:05) - how H3 moves are actually controlled
Problem: two renders show a prompted "slow push-in of N cm over 15 s" always travelling to a tight
close-up, where the model invents background (streak, smeared wall). Static holds clean but a
locked-off singer video looks dead. What the sources say [P = primary, C = community]:

1. **Official amplitude grammar exists and we never used it** [P, MiniMaxAI/MiniMax-H3
   docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md 4.3]. Camera motion is one natural-English action
   inside the shot with two trained modifiers: amplitude "with small amplitude" / "with large
   amplitude" and speed "at slow speed" / "at fast speed" ("medium amplitude and normal speed are
   usually omitted"). Example: "The camera pushes in with small amplitude at slow speed toward the
   folded letter in her hands." Verbs: Zoom In/Out, Push In/Pull Out, Pan, Truck, Tilt, Pedestal,
   Arc Shot, Tracking Shot, Static Shot, Shake Slightly/Strongly, POV, Roll. Moves end on a subject
   clause, or "The camera holds a static shot as ...". No metric distances anywhere in the guides:
   our "12 cm" meant nothing to the model. The ref guide [P] says the same: "movement type,
   amplitude, and speed" in natural English; no bracket [Push in] commands in H3 (those are the
   hosted Hailuo Director API).
2. **Camera reference VIDEO is a first-class input** [P, ref guide; box node]. `<Video N>` "for
   camera movement, cuts, rhythm, or temporal structure", declared under reference generation,
   retention "<Video 1> (camera movement and pace): weak_reference". The box's
   MiniMaxH3ReferenceToVideo has ref_videos (up to 3, 2-15 s at 24 fps) next to ref_images.
   Vendor guides [C]: state that the video supplies ONLY pace/movement, not its location; a
   stable clean clip beats a busy one. No measured reports of how faithfully H3 copies a path.
   We can MANUFACTURE the reference with exact pace: ffmpeg zoompan over the location still
   (made: `.mvwork/g1_refs/camref_parlour_pushin_8pct_15s.mp4`, 8 % even push, 360 f).
3. **Keyframe-bounded moves** [P, box]: core `MiniMaxH3AddGuide` (image at any frame_idx,
   negative = from end) and `MiniMaxH3CustomKeyframes` pin exact start AND end compositions, so a
   push-in's extent is fixed by a cropped copy of the still as the last frame. Pins her pose too,
   so for ENVIRONMENT/B-roll scenes only (empty parlour, quarry, stone detail).
4. **Camera LoRAs** [P cards]: Jojocodex Camera Motion (Apache-2.0, trigger "camera motion" at
   prompt start, strength 0.8-1.0, "slow" softens, push-in its strongest move, T2V-trained, Turbo
   compatibility unstated) and Yunjing (push/orbit/tracking/handheld, stacks with Turbo per the
   wildminder table). They ADD motion vocabulary; they do not bound amplitude beyond words, and a
   third LoRA on top of Turbo + two identity pictures risks identity. Hold as a fallback for moves
   the base ignores (pan/truck/arc).
5. **Fun ControlNet Union** on the box has no camera-trajectory type (Canny/Depth/HED/MLSD/Pose +
   inpaint; v2 adds Scribble/Layout/Gray) [P/C]. Not a camera tool. Camera H3 path editor
   (NyckM) and imbutus Director are prompt compilers with no guarantee [C]; nothing new under them.
6. **Post move** (ours): render the performance take static (proved clean), FlashVSR x2, then an
   exact ffmpeg zoompan/pan of 4-8 %. No parallax, but on a seated singer a 2D push is
   indistinguishable from a dolly. Always available; the production fallback.

Recommendation and test (G6, two renders of the chair half, ~8 min each, parlour still kept):
- V1 prompt-only: official grammar + framing anchors: "The camera pushes in with small
  amplitude at slow speed toward her face, from a medium shot with the whole armchair and the oil
  lamp in frame to a medium close-up that keeps her shoulders and the lamp in frame; the lamp
  never leaves the frame. The camera holds as she finishes the line."
- V2 = V1 + `<Video 1>` camera reference (the 8 % zoompan clip) declared as pace/movement only.
- Judge: does the push stop at medium close-up with the lamp in frame; any right-side invention.
Writer change if V1 or V2 holds: camera field becomes {move, amplitude, speed, start_framing,
end_framing, keep_in_frame}; compiler writes the official sentence; environment scenes get
keyframe-bounded moves (3); performance takes default to small-amplitude slow moves with
anchors, never metric distances; static stays an option, not the default.

## 16. G6 result (2026-10-28 01:20) - the official camera grammar holds the push [V stills, motion pending]
- Three chair-half renders, same seed/turbo/pictures/window, only the camera wording differs
  (+ a reference clip in V2):
  | render | camera control | 0 s | 14.9 s | box time |
  |---|---|---|---|---|
  | G4 | "slow push-in ... about 12 cm over 15 s" (metric) | medium | tight close-up, lamp and window gone | 7.8 min |
  | G6 V1 | "pushes in with small amplitude at slow speed ... to a medium close-up that keeps her shoulders and the lamp in frame; the lamp never leaves the frame; holds a static shot as she finishes" | medium | medium close-up, lamp and window in frame | 8.5 min |
  | G6 V2 | V1 + 8 % zoompan clip as <Video 1> (pace only, weak_reference) | medium | medium close-up, lamp and window in frame | 25 min |
- V1 and V2 both stop where the words say. The reference clip added nothing visible in stills
  and tripled the render time (reference-video tokens ride through every step). Verdict: the
  model's own amplitude/speed grammar plus framing anchors is the control; the camera-reference
  lane stays available but is not the default. Background stayed stable in both (lamp, window,
  chair in place at every sample); no right-side invention in the stills.
- To check by eye: a dark cord/lace hangs down the chemise in V1/V2 from ~7 s that is not in the
  references [H wardrobe drift, stills only]; lip-sync; the push's feel.
- Misfiling found and fixed (5d48f28): with a LoadVideo node in the graph, ComfyUI history lists
  the reference clip under that node as type "input"; both collectors took the first media entry
  and filed the reference clip as the render. Collectors now skip type=input. The real V2
  (h3chain_00007-audio.mp4) was pulled from the box by hand over the misfiled copy. The backend
  still runs the old code until the next restart (deferred: the 768p test is on the box).
- Copies: `.mvwork/g1_refs/nevermine_G6_V1_chair_official-grammar_parlour-still_seed20261027.mp4`,
  `nevermine_G6_V2_chair_grammar-plus-camref-video_parlour-still_seed20261027.mp4`, sheet
  `nevermine_G6_contact_sheet.png`.
- G7 (768p timing test) fired 01:13:01: chair half, V1 prompt, 1344x768, same seed/turbo.
- CORRECTION 01:35 (user watched: "loads of weird artifacts"; 1 s frames confirm): V1 is NOT a
  dolly. Between 5 and 7 s it cross-dissolves from the medium shot to the medium close-up (a
  ghost double exposure at 6 s), and the wardrobe changes across the dissolve (laced corset gone
  from 7 s, a dark beaded cord appears on the chemise). V2 jumps medium -> medium close-up between
  8 and 9 s with the same drift (black lacing/cords over the chemise, corset gone, a candle appears
  by the lamp); the reference clip did not prevent it. [H] naming TWO framings ("from a medium shot
  to a medium close-up") reads as two shots joined by a hidden cut; the distilled sampler renders
  the cut as a dissolve, and the re-establish lets the wardrobe drift. G4's single-framing push was
  a true continuous move, only unbounded. Next (V3, not fired): one framing, the move bounded only
  by what must stay in frame, "one continuous shot with no cut and no dissolve", and a wardrobe
  hold in retention ("the laced corset stays visible throughout"). Frames:
  `.mvwork/g1_refs/G6_frames/dense/{V1,V2}_dense_sheet.png`. Section 16's "holds the push"
  verdict is withdrawn; the grammar bounded the END framing but via a cut.

## 17. G6 V3 and G7 768p results (2026-10-28 01:50) [V stills; user to watch]
Same chair half, seed 20261027, turbo, face + body + parlour still. 1 s frames in
`.mvwork/g1_refs/G6_frames/dense/{V3,P768}_dense_sheet.png`.
- **V3** (544p; one framing, "pushes in with small amplitude at slow speed toward her face
  throughout the take, one continuous shot with no cut and no dissolve", lamp/window/shoulders
  "stay in frame to the end", corset hold): a GENUINE continuous push, no ghost, no cut; corset
  held through 8 s; lamp and window in frame to 8 s. Then it keeps going: lamp out of frame by
  ~11 s, tight close-up by 13-14 s. So the one-framing wording removed the dissolve and the
  wardrobe drift, but the move is unbounded again (same extent as G4). "Small amplitude" and
  "stay in frame to the end" did not stop it at 544p. 8 min.
- **G7 768p** (1344x768; the V1 wording that dissolved at 544p): a gentle continuous push that
  STOPS - medium shot to a slightly tighter medium shot over 15 s, lamp, candle and window in
  frame throughout, corset present throughout, no dissolve, no wardrobe drift at 1 s spacing.
  Best chair take so far by stills. 23 min on the box (2.7x the 544p time), fits in memory with
  three pictures and the song lock. [H, one sample] the turbo LoRA is the 768p variant and the
  grammar is obeyed at its trained size; at 544p the same words degrade into a cut. Seed variance
  cannot be excluded from one render.
- Timing for a full video at 768p: 24 single clips x ~23 min + 2 chains ~ 10-11 h.
- Recommendation: judge performance takes at 768p from now on (it is the delivery size anyway);
  544p only for throwaway checks. Next test when awake: V3 wording at 768p (continuous + bounded?)
  and a second seed of G7 (variance). Not fired.
- Backend restarted 01:50 on the collector fix (type=input echoes skipped) and the writer's
  one-framing camera rules.

## 18. G8: V3 wording at 768p (2026-10-28 09:17) [V stills; user to watch]
Same chair half (13.34-28.42 s), seed 20261027, turbo, face + body + parlour still, mix lock,
1344x768, V3 one-framing wording ("pushes in with small amplitude at slow speed throughout the
take, one continuous shot, with the lamp and the window staying in frame"). 23 min on the box.
Filed as `.mvwork/g1_refs/nevermine_G8_chair_768p_1344x768_V3-prompt_seed20261027.mp4`;
1 s frames in `G6_frames/dense/P768V3_dense_sheet.png`. Floor check r = 1.000, 362 f.
- From stills: one continuous push, no dissolve seen, corset and chemise held to the end,
  hands move from folded to resting on the armrests around 8 s (prompt asked for still hands).
- Extent: pushes further than G7 (V1 wording at 768p). G7 settled at a loose medium close-up
  with lamp and window both in frame; G8 ends at a medium close-up with the window still in frame
  but the oil lamp at the right edge, partly out of frame by 12-14 s. So at 768p the V3 wording
  is continuous but less bounded than V1; the keep-in-frame sentence did not hold the lamp.
- Still one sample each; seed variance not excluded. Motion continuity is for the user to judge.
- Request: `g8_v3_768p_request.json` (V3 request minus megapixels, width 1344, height 768);
  hand-over script `g8_handover.sh <job> <tag> <name>` (probe, floor check, copy, 1 s sheet).
- **User verdict (09:40):** both 768p takes usable; prefers the G8 camera move (V3 wording),
  which is already the writer's default grammar (0d05c5c). No writer change needed.
- **Eyes:** user noted her eyes were closed at the start of earlier takes and open throughout
  now. Face crops of t=0.2/1/2 s for all six chair takes (`G6_frames/eyes/eyes_open_sheet.png`):
  G3 and 544p-V1 open lids-down and lift on the first line; G4, V3, G7, G8 open eyes-open into
  the lens. Every prompt has the same line ("head slightly lowered, eyes rest just below the
  lens"); nothing asks for closed eyes, so the lids-down opening was per-render luck, not
  wording (V1 gave both outcomes at 544p vs 768p). To get it reliably it must be written in
  explicitly (lids lowered at the start, gaze lifts into the lens on the first word). User's call.
- **Decision (09:45):** eyes left to the model; the chair prompt stays as written. No eye-direction rule added to the writer.

## 19. Reusable location library (built 2026-10-28 10:30) [V through the API, no GPU]
Why: every writer run invented fresh location names (G2: parlour/corridor/mantelpiece/quarry/
wrecked_parlour; G5: parlour/parlour_window/hallway/quarry/stone_detail), orphaning the stills
already generated and forcing a per-song name map. The user's rule (2026-10-28): every
experiment must land as a technique reusable across videos, never a per-song fix.

What: `backend/mv_locations.py`, store `.mvwork/mv_locations/locations.json` (rows: id, name,
description, still_prompt, still_id, tags, source, created/updated).
- Writer (`mv_writer.write_script(..., known_locations=rows)`): pass 1 gets a KNOWN LOCATIONS
  block (id, name, description) and the rule: reuse by exact id with `{"id", "reuse": true}`
  wherever a known place serves THIS video, invent only where none fits, never redecorate a
  known place. `normalize_locations(plan, scenes, known)` resolves a known id, or a new id
  carrying a known place's name, to the library row (description, still prompt and still id
  win; marked `known`; scene ids aliased). The markdown lists known rows as "still on file".
  The script records `known_offered`.
- `POST /api/mv/write` gains `locations`: "library" [default: every row with a still], "none",
  or a list of ids. Default is to offer the library: that is the point of the request; "none"
  reproduces the old behaviour.
- `POST /api/mv/render_scene`: with no `location_refs`/`location_ref_id` the scene's still comes
  from the script's row (a known location carries `still_id`) or the library by location id;
  `use_location_library: false` turns that off; response/meta carry `location_ref_from`
  (request | script | library). New `dry_run: true` returns everything that would be submitted
  (refs in upload order, location and its source, frames, prompts) with nothing queued: the
  gate before the GPU.
- Library routes: `GET /api/mv/locations` (rows + which are offered), `POST /api/mv/locations`
  (upsert by hand; still_id must exist), `POST /api/mv/locations/remove`, `POST
  /api/mv/locations/import {script, location_id, still_id}` (file a script's location with its
  imported still), `POST /api/mv/locations/import_still` (multipart: file + script +
  location_id; files the still AND the row in one step - the normal path for a new place).
- Seeded through the import route with the five Never Mine locations (tags never_mine,
  victorian). Verified: dry run of the chair scene with no map resolves the parlour from the
  library (`location_ref_from: library`), prompt carries <Picture 3>; a G5 hallway scene (not in
  the library) resolves no still; explicit map still wins; `use_location_library: false` gives
  her two pictures only; upsert rejects a missing still; remove works; box queue stayed 0.
- NOT yet exercised: a live writer run with the library offered (Claude call, ~9 min, no GPU).
  Offline: the prompt block renders, normalisation resolves reuse/alias/new/derived cases.
Next general pieces: techniques doc; shot-type regression set; second-song transfer test.

### 19.1 G9: live writer run with the library offered (2026-10-28 11:08) [V, no GPU]
Same request as G5 plus `locations: "library"`; alignment on the vocal stem (the default now;
G5 was on the mix). 478 s, Opus medium. `.mvwork/mv_scripts/never_mine_20261028_110755.{json,md}`.
- Reused ALL five known places by id, invented none: parlour (5 scenes), quarry (6), mantelpiece
  (3), wrecked_parlour (3), corridor (2). G5 without the library had invented parlour_window,
  hallway and stone_detail and dropped the wrecked parlour; those are gone.
- 19 scenes / 24 clips, 0 prompt problems; performance takes 6.6-19.4 s, B-roll to 38.4 s;
  camera lines all in the model's grammar with one framing and a keep-in-frame list.
- Known rows carry the library still id into the script: dry runs of scene 2 (singer, parlour)
  and scene 1 (environment-only, mantelpiece as <Picture 1>) resolve `location_ref_from:
  script` with no map in the request. Scene settings restate the library description (window
  left, fireplace + armchair right, lamp on the side table).
- Markdown lists the five as KNOWN, still on file; nothing for the user to generate.
- Caveat: one run cannot separate "fit" from "offered" (it also brought back the wrecked
  parlour G5 had dropped). The transfer test on a second, different song, run once with
  "none" and once with "library", is where that gets answered.

## 20. Second-song transfer test (set up 2026-10-28 12:00; song = user's pick; nothing fired)
Purpose: every rule in `docs/MV_H3_TECHNIQUES.md` was measured on Never Mine (slow, clean solo
vocal, one seated performer). This test asks whether the rules transfer, and settles the two open
hypotheses: (a) the location library's offer does not bias reuse beyond fit; (b) the camera
grammar holds on a shot that is not a seated push-in.

Song: the user's pick. It should DIFFER from Never Mine where the rules are weakest: faster or
denser (layered symphonic/power-metal vocals stress alignment and the lock), a vocal that moves
(stress take length and setups), and a world that is not a Victorian parlour (stress the library:
reuse should be near zero if the offer does not bias).

References (user's call, default A):
- A. reuse the Selene face + body pictures from Never Mine. Fast; but their descriptions and
  the body picture itself show the parlour, a bias toward reusing it. The runner's default refs
  file is `.mvwork/transfer/refs_selene_default.json`.
- B. two new pictures in the new video's look on a plain background (the design in s.3.2),
  from Claude prompts; descriptions free of any place. Cleaner test, costs two GPT images.

Steps (runner `.mvwork/transfer/transfer_test.sh <audio_id> <tag> <title> <style_file>
<lyrics_file> [refs_json]`, output under `.mvwork/transfer/<tag>/`; it stops before any render):
1. vocal stem on the box (`/api/mv/stems`, ~80 s); alignment then runs on the stem by default.
2. writer with the library WITHHELD (`locations: "none"`): the control; what it invents for
   this song with no suggestions.
3. writer with the library OFFERED (default): same request otherwise.
4. compare (`compare.md`): aligned_on + cover (alignment on a dense mix), scene/clip counts,
   singing vs B-roll, take-length range, prompt problems, the two location tables (known /
   new / derived), every camera line, and the still prompts for the library run's NEW places.
   Verdict (a): read both tables; a known place taken for a scene it does not suit = bias.
5. user generates the new-place stills (GPT) and files each with
   `POST /api/mv/locations/import_still` (file + script + location_id) - one step, the library
   grows. More stills are always fine (memory `more-stills-are-cheap`).
6. gated renders at 768p, one 362-f clip each, `dry_run: true` first, then the user's go per
   render (~23 min each): three SHOT TYPES, not three scenes of one kind:
   - sing-to-camera with a small-amplitude slow move (the proven case, new room);
   - a standing or walking performance/B-roll with a tracking or truck move (new case for (b));
   - an environment-only scene (location alone as <Picture 1>; later the keyframe-bounded lane).
   Judge: floor check r, 1 s frames for cuts/ghosts/wardrobe/extent, then the user watches.
   Same seed across the three; a second seed only on the shot that misbehaves.
7. record: techniques doc rules confirmed / amended with the new provenance; plan s.20.x.

Library songs with lyrics on file (read-only scan 2026-10-28): symphonic metal YuE2/Music 3 -
Dream Of Me (276 s), Nine Nights (294 s), Moonlit Requiem (273 s); power metal YuE2 - Skyborn
(227 s), Thunder in My Veins (265 s), Let the Stars Burn (~300 s). Suno imports carry no lyrics in
the library row; their lyrics come from the project song or the user. Lyrics go to the runner as a
tagged text file (section tags as for Never Mine); style as a one-paragraph file.

### 20.1 Song and references chosen (2026-10-28 12:40)
- Song: **She Sings Before the Storm**, the naturalized Suno version the YouTube tab used:
  library `2306adc5a44a43afb1e4743b86bd98b9` (252.4 s; md5 identical to the user's
  `/Volumes/SSD1/Downloads/She_Sings_Before_the_Storm_naturalized.wav`). 150 bpm A minor Celtic
  folk metal, mezzo belting to keening head voice, dual guitars, fiddle and whistle: faster,
  denser and a different world from Never Mine on every axis the test wants.
- Runner inputs under `.mvwork/transfer/ssbs/`: `lyrics.txt` (tagged, from the project song:
  2 verses, 2 pre-choruses, 3 choruses, solo, bridge, outro), `style.txt` (bpm/key/tags + the
  Suno house-format style), `refs.json` (place-free descriptions), `ref_prompts.md`.
- References: option B. Two new pictures on a plain slate-grey backdrop in the song's world
  (Irish coastal 19th-c. dress: green laced wool bodice, white linen shift off the left
  shoulder, charcoal skirt, dark shawl, bare feet, no jewellery), prompts in `ref_prompts.md`,
  recipe recorded as techniques 4a. Waiting on the user's two images; then
  `.mvwork/transfer/transfer_test.sh 2306adc5a44a43afb1e4743b86bd98b9 ssbs "She Sings Before the Storm" .mvwork/transfer/ssbs/style.txt .mvwork/transfer/ssbs/lyrics.txt .mvwork/transfer/ssbs/refs.json`
  (stems ~80 s on the box, two writer runs ~8 min each, no render).

### 20.2 Transfer test, steps 1-4 result (2026-10-28 12:15) [V, no GPU]
- Stem: RoFormer on the box, cached. Alignment on the stem: cover 0.894 (288/322 words), 14
  sections, both passes identical facts. Never Mine was 0.967; a dense Celtic mix costs ~7
  points and is still usable.
- CONTROL (library withheld, 721 s): 25 scenes / 30 clips, 14 singing / 11 B-roll, takes
  5.2-38.8 s, 0 problems. Five places, all invented, all in the song's world: cliff_road (chapel,
  gorse), cliff_wall (where she sings), harbour (quay, shutters, bollards), cottage_kitchen (her
  father by candlelight), tavern (the proud at supper). Script
  `.mvwork/mv_scripts/she_sings_before_the_storm_20261028_115922.{json,md}`.
- LIBRARY OFFERED (same request, 22 scenes / 27 clips, 0 problems): reused 3 of the 5 Never
  Mine places and invented 3. The parlour took the father's candlelight verse and the proud's
  supper (the control's cottage kitchen and tavern), the mantelpiece became a singing close-up
  twice, the wrecked parlour took "and by morning they are gone". Its "why" lines rationalise the
  fit after the fact ("the parlour where her father told the story waits by lamplight"); the
  control, with no suggestion, put those beats in a fisherman's cottage and a harbour tavern.
- **Verdict: the offer biases. Hypothesis (a) in s.20 is false.** The library's value is real
  for RE-RUNS of the same song (G9: all five reused, stills kept) and harmful across songs by
  default.
- **Rule change (built, backend restarted 12:20):** `/api/mv/write locations` default is now
  "auto" = offer only the library places that came from THIS song (title match on the row's
  source), so re-runs keep their stills and a new song starts clean; "library" = every place,
  an explicit opt-in for deliberate cross-song reuse; "none" and id lists unchanged. Verified
  offline: auto on Never Mine offers 5, auto on She Sings offers 0, library offers 5.
  `mv_locations.select(rows, want, title)`, `same_song`.
- Carry forward: the CONTROL script. Still prompts for its five places:
  `.mvwork/transfer/ssbs/location_still_prompts.md` (user generates; one-step filing with
  `/api/mv/locations/import_still`). Reference pictures filed: face 2c1c49e7, body cd98a0e4.
- Next: user's five stills, then step 6 (three shot types at 768p, dry run first, go per render).

### 20.3 Stills filed, three shot types dry-run (2026-10-28 13:50) [V through the API, no GPU]

- The user generated the five location stills from `location_still_prompts.md` (GPT, 1672x941,
  all person-free, all match their prompts on the named objects: chapel + belfry, lichen wall +
  gable, bollards + four boats, candle + oilskin + sea boots, fiddle + lamps + hearth).
- Filed with the one-step route, one call per still:
  `curl -F file=@stills/<id>.webp -F script=she_sings_before_the_storm_20261028_115922.json
  -F location_id=<id> /api/mv/locations/import_still`. Each call stored the still in the library
  and upserted the location row (description + still prompt verbatim from the script, source =
  this song). Library now 10 rows, five per song, every still on disk:
  cliff_road 7bb11d6c, cliff_wall 88d6056e, harbour 32de3cf9, cottage_kitchen 84bcecd9,
  tavern 0fc6c295. `.mvwork/transfer/ssbs/stills/` + `import_still_<id>.json`.
- Three dry runs at 1344x768, seed 20261027, turbo, refs face 2c1c49e7 + body cd98a0e4, NO
  `location_refs` map in the request: every scene resolved its still from the library
  (`location_ref_from: library`). Requests/responses + full prompts in
  `.mvwork/transfer/ssbs/dry/` (`dry_run_prompts.md`).

  | tag  | scene (0-based) | type | place | frames | notes |
  |------|-----------------|------|-------|--------|-------|
  | sing | 2 "A high clear line" 42.06-51.64 | sing_to_camera, medium close-up, push in small/slow | cliff_wall | 243 + 39 ctx, 1 clip | refs face+body+still; the G8 shot type on a new song |
  | walk | 1 "The voice that walks" 36.72-42.06 | no_singing_her, medium wide, tracking small/slow | cliff_wall | 141 + 39 ctx, 1 clip | "her one motion is a slow, steady walk of about four unhurried steps"; FIRST standing/walking take |
  | env  | 3 "Shutters barred, lines hauled" 51.64-57.20 | no_singing_environment, wide, truck right small/slow | harbour | 141 + 39 ctx, 1 clip | refs = the still only (no singer refs, as designed); FIRST environment-only take |

  All three prompts carry one action, one framing, keep-in-frame bounds, the no-cut/no-dissolve
  clause and a pinned sky. Scenes are shorter than 362 f, so each is one clip at its own length
  (cheaper than the s.20 estimate of ~23 min; expect roughly proportional).
- Box: queue empty, no backend jobs at 13:50. Nothing fired. Next = the user's go per render,
  recommended order sing (direct comparison with G8), walk, env; judge each by floor check,
  1 s frames, then the user watches.

### 20.4 Transfer test step 6: three shot types rendered at 768p (2026-10-28 14:17) [V floor check + stills; user to watch]

- Fired on the user's go ("go for 768", then "go for those too"), one request each, mirroring
  the G8 call with the She Sings refs and no `location_refs` map (stills from the library).
  Requests/responses/log: `.mvwork/transfer/ssbs/renders/`. Watcher + handover scripts there.

  | take | scene | frames | wall time | floor r | filed as |
  |------|-------|--------|-----------|---------|----------|
  | SING | idx 2 "A high clear line", cliff_wall, push in small/slow | 243 (10.1 s) | 13 min 16 s | 0.998 at lag 0 | `ssbs_sing_cliffwall_768p_pushin_seed20261027.mp4` |
  | WALK | idx 1 "The voice that walks", cliff_wall, tracking small/slow | 141 (5.9 s) | 5 min 52 s | 0.998 at lag 0 | `ssbs_walk_cliffwall_768p_tracking_seed20261027.mp4` |
  | ENV  | idx 3 "Shutters barred, lines hauled", harbour, truck right small/slow | 141 (5.9 s) | 5 min 09 s | 0.998 at lag 0 | `ssbs_env_harbour_768p_truckright_seed20261027.mp4` |

  Wall time scales with frames (about 3.3 s per frame at 768p incl. the fixed per-job cost),
  consistent with G8's 23 min for 362 f.
- Song lock: all three clips' audio follows the song window at r 0.998, lag 0 (clean band).
  The masked-song latent held on a dense mix with a 0.894 stem cover.
- From 1 s frames (`renders/frames/*_sheet.png`; stills only, motion is the user's call):
  - SING: identity + wardrobe hold through all 10 s (green laced bodice, shift off the left
    shoulder, shawl at the elbows); lichen wall, gorse, chapel gable and sea all present; the
    push tightens from medium close-up to close-up by t=9 with the wall still in frame; mouth
    moves throughout; eyes open; sky reads fixed. Same shot type as G8, new song, new place:
    the grammar transferred at the still level.
  - WALK: she walks barefoot along the wall in three-quarter view, one hand trailing on the
    stones, head to the sea, chapel behind; wardrobe holds; the near wall stones shift between
    frames (tracking present). First standing/walking take rendered under the grammar.
  - ENV: the harbour still reproduced (whitewashed cottages, barred shutters, bollards, four
    boats, harbour mouth), no people; boats change position between frames (swell); the near
    bollard moves little across the six frames, so the truck's extent is small; sky reads
    fixed. First environment-only take under the grammar, refs = the still only.
- Open until the user watches: pace of the walk (four unhurried steps in 5.9 s?), whether the
  track reads as a track or a drift, whether the truck is visible at all, lip-sync on SING.

### 20.5 Verdicts on the three takes; environment pace rule built (2026-10-28 14:45) [U + built; fix render pending]

- User watched all three: "1st is fine, 2nd the camera wobbles a little before tracking, looks
  like a hand cam, but its ok, 3rd one, the boats is massively sped up like a timelapse."
- SING passes: the camera grammar transferred to a second song, place and pair of reference
  pictures. WALK passes with a note (handheld wobble before the track settles; one sample;
  hypothesis for a fix in techniques s.7, no render spent). ENV fails on pace.
- Cause, found in the writer, not guessed: `CLIP_SYSTEM`'s motion rule read "One main physical
  motion for HER per clip", so an environment-only scene had no one-motion rule; the harbour prose
  stacked boats + ropes + shutters + water with "about once every two seconds". The August pace
  measurement (busy prompt accelerates) predicts exactly this.
- Built (no GPU): environment bullet in `SCENE_SYSTEM` + `action` schema; environment bullet in
  `CLIP_SYSTEM` (one moving element, counted over the whole clip, everything else still by name,
  no rhythm/frequency, sounds to the soundscape); `check_prompt` flags rhythm phrases in an
  environment description (flags the rendered harbour prompt, passes her shots and the clean
  form). New route `POST /api/mv/rewrite_scene {script, scene}`: re-runs pass 2 for one scene
  under current rules, saves `<script>_rw<scene>_<stamp>.json/.md`, original untouched. Backend
  restarted 14:40 (box idle, no jobs).
- Rewrite of scene 3 under the rule: 0 problems; "The one moving element is the four boats on the
  swell: over the whole 6 seconds the boats lift and settle once, slowly and heavily, together";
  shutters, ropes, cottages, bollards, sea band and sky all named still; creaks and rattles moved
  to the soundscape. Script `she_sings_before_the_storm_20261028_115922_rw3_20261028_144353.json`.
- Dry run of the rewritten scene: same still (32de3cf9, library), same refs, 141 f, 1 clip, prose
  differs. Gate passed; fired 14:44:20 on the user's "go ahead and fire when ready": seed
  20261027, turbo, 1344x768, ONE variable (the clip prose). `.mvwork/transfer/ssbs/renders/env2_*`.

### 20.6 Harbour re-render under the one-motion rule (2026-10-28 14:50) [V floor check + proxy; user to watch]

- Job finished 14:50:33 (6 min 13 s for 141 f). Floor check r 0.998 at lag 0. Filed as
  `ssbs_env2_harbour_768p_truckright_onemotion_seed20261027.mp4` beside the first take.
- One variable (the clip prose); seed, size, still, refs, frames identical (dry-run gate).
- Stills (`frames/ENV2_sheet.png`): place reproduced, no people; the boats sit in the same
  positions across all six 1 s frames (in the first take they shifted between frames).
- Motion-amount PROXY, not a verdict: mean inter-frame absolute difference (gray, 336x192):
  first take 3.30 (max 9.11), re-render 2.19 (max 3.65). The spikes are gone; the mean is
  a third lower. Whether the swell now reads at real speed, and whether the truck is visible,
  is the user's call from watching.
- **User verdict on the re-render (14:55): "still sped up".** The one-motion rule removed the
  spikes, not the timelapse. Rule kept (necessary), not marked passed. Waiting for the user's
  description of what moved too fast before choosing the next variable.
- **User description (14:58): "looks the same".** The prose change is invisible to the eye on
  this content: the proxy moved (3.30 to 2.19) and the take did not. MEASURED: on an
  environment-only open-water shot the prompt is not a pace lever at all. Note that the
  sing and walk takes at the cliff wall both had sea and sky in frame and did NOT timelapse:
  the person in frame is the real-time anchor. [H to test]

### 20.7 Environment pace: what is left after the prose (2026-10-28 15:10) [built; nothing fired]

- Levers ruled out for an open-water environment-only shot: the prose (two renders, "looks the
  same"); the turbo recipe (August, ~8 %); a negative prompt (none exists, cfg-free guider).
- Evidence for a different lever: the SING and WALK takes at the cliff wall had sea and sky in
  frame and ran at real speed. [U] Hypothesis A: a person in frame is the real-time anchor;
  environment-only open-water B-roll has none.
- Candidates, each one render of about 6 min, one variable against the same harbour scene:
  - B. Keyframe-bounded plate: the location still anchored at the first AND last frame of the
    clip through `MiniMaxH3AddGuide` (conditioning in/out; built as `keyframes:
    "still_first_last"`, opt-in, default off, also `still_first` / `still_last`; the chain
    anchors the LAST reference picture, which is always the location still). Known from the
    loop lane: the model eases out of the first anchor and settles onto the last (section 12),
    so the extent is pinned by construction; whether the mid-clip speed is real is the test.
    Caveat: AddGuide's resize of a 1.777 still onto a 1.75 canvas is undocumented (the I2V node
    stretches first, crops last); a 1.5 % mismatch, acceptable for the test.
  - A. The same place with her in it: script scene 16 "At the end of the quay" (sing_away,
    wide, pull out large/slow, 175 f, 7.3 s), dry-run clean, 3 refs. If the sea runs at real
    speed with her in frame, the writer rule becomes: open-water or open-sky B-roll is shot WITH
    her (no_singing_her) or as a plate; never environment-only by prompt.
  - C. Production fallback, already in the user's hands: the take slowed 2x and 3x with frame
    interpolation (`renders/*RETIMED*`), no GPU; an environment plate carries the master song in
    assembly, so retiming costs nothing but the look of interpolated water.
- Offline check of the keyframe graph: with the flag, two AddGuide nodes (frame 0 and -1) sit
  between ReferenceToVideo and BasicGuider, both fed the still's LoadImage and the video VAE;
  without the flag the graph is byte-identical to before (0 AddGuide nodes). Bad values are
  rejected. `render_scene` validates the flag, requires a location still, reports it in the
  dry run and the job meta.
- **User on the retimes (15:12): the 3x one "looks okish, just not great, the speed is about
  right but it doesnt really look natural"; 2x not mentioned as right.** Two facts from that:
  the model renders this open-water plate about THREE times too fast; and ffmpeg minterpolate
  on water is the unnatural part, not the speed. C stays a fallback; a box-side interpolator
  (RIFE/FILM through ComfyUI) would be the better version of C if A and B fail.
- **LoRA hypothesis (user, 15:25): "I wonder if its caused by the lora."** Checked against the
  record: the August exoneration of the turbo recipe (~8 % of motion) was measured with the
  AUTHOR's ULTRA+TURBO block (4-step LoRA, euler/beta, 8 steps, shift 12/3) on a grass shot. The
  current turbo block is DIFFERENT: lightx2v-style `minimax_h3_ref2v_turbo_8step_v1.1_768p` at
  1.0, res_multistep/simple, 8 steps, shift 6/3. Its effect on pace has never been isolated, and
  shift_video is known in DiT video models to trade motion amount against stability. So the
  hypothesis is open, not refuted. One-variable test D: the same harbour scene with `turbo:
  false` (base weights, 20 steps, no LoRA, shift 12/3), about 2.5x the wall time (~15 min for
  141 f). If D runs at real speed, a follow-up separates LoRA from shift (turbo LoRA with shift
  12/3). Research pass launched in parallel (upstream cards, guides, community reports).
- **Research pass back (15:40), full report `.mvwork/transfer/ssbs/research_h3_pace_2026-10-28.md`.**
  Decisive item: the publisher (lightx2v) states video shift 12 / audio 3 for the REF2V 8-step
  LoRA (release post #51) and says the shift-6 advice is for FL2V, not REF2V (post #57); our
  CHAIN_TURBO runs ref2v at 6/3 on a secondary wiki's word. Every community report on H3 turbo
  LoRAs is slower motion, not faster, and nobody reports water as timelapse; the acceleration
  signature ("start slow, speed up, settle") is a known H3 trait (Cadence-Retime exists for it).
  No node exposes fps or motion speed; the prompt guide has no pace vocabulary beyond the camera.
  New one-variable test E: turbo with shift_video 12 (publisher's value), ~6 min, ahead of test D.

### 20.8 Keyframe plate and her-at-the-quay rendered (2026-10-28 15:36) [V floor check + proxy; user to watch]

| take | change vs the env2 plate | frames | wall | floor r | proxy mean (per second) | filed as |
|------|--------------------------|--------|------|---------|-------------------------|----------|
| ENV3 | `keyframes: still_first_last` only (same seed, prose, still) | 141 | 5 min 21 s | 0.998 | 0.53, flat 0.51-0.64 then 0.22 at the tail (env2: 2.19, flat 2.1-2.5) | `ssbs_env3_harbour_768p_truckright_keyframes_still_first_last_seed20261027.mp4` |
| HERQUAY | script scene 16 "At the end of the quay" (sing_away, wide, pull out large/slow, her 2 refs + still) | 175 | 7 min 43 s | 0.998 | 3.1 rising to 6.5 across the clip | `ssbs_herquay_harbour_768p_pullout_with_her_seed20261027.mp4` |

- ENV3 stills: composition identical across all six 1 s frames (the anchors pinned it, camera
  truck included: no visible truck, as the research predicted for a same-still guide); whitecaps
  present in every frame. The proxy is a quarter of env2's; whether that is a real-speed swell or
  a near-frozen plate with texture flicker is the user's call. The tail drop is the settle onto
  the last anchor (loop-lane behaviour, section 12).
- HERQUAY stills: she stands at the quay end in three-quarter profile, wardrobe held, bare feet,
  rain streaks, waves breaking over the harbour mouth, the pull-out visible (a bollard enters the
  foreground by t=3). A wall lantern not in the still appears on the cottages (an added "state"
  element). The rising proxy is consistent with the pull-out revealing more frame, not by itself
  a timelapse signature. Whether the sea runs at real speed around her is the hypothesis-A test
  and is the user's call.
- Staged, not fired: `renders/env4_shift12_768p_request.json` (turbo, shift_video 12 = the
  publisher's REF2V value; one variable) and `renders/env5_base_768p_request.json` (turbo off).

### 20.9 Verdicts on ENV3 and HERQUAY; shift-12 and base plates fired (2026-10-28 15:53)

- User: "the 1st one worked, but is too static, the 2nd one worked well, but we need to be able
  to have b roll without having the singer in, go ahead."
- ENV3 (same-still keyframes): real speed, too static, camera pinned. A plate technique for
  near-still backgrounds, not for a sea. HERQUAY: hypothesis A CONFIRMED for this place: her in
  frame = real-time sea. Both in techniques s.8 and open item 3.
- Requirement restated by the user: environment-only B-roll without the singer must be possible.
  Fired on "go ahead", one after the other, same harbour scene, seed, prose, still:
  ENV4 = turbo with shift_video 12 (publisher's REF2V value; ONE variable vs env2);
  ENV5 = base recipe (20 steps, no LoRA, shift 12/3; three variables vs env2, one vs env4 = the
  LoRA + steps). 15:52:59, box running 1 pending 1. Expected ~6 min and ~15 min.
- Readout: if ENV4 runs at real speed, the shift was the lever and the turbo block gets shift 12
  for ref2v; if ENV4 is still fast and ENV5 is right, the LoRA (or 8 steps) is the cause and
  environment plates render on base; if both are still fast, pace is a content prior and the
  remaining levers are `still_last` keyframes, the Motion-Speed slider LoRA, or a box-side retime.

### 20.10 Shift-12 and base-recipe plates (2026-10-28 16:09) [V floor check + proxy; user to watch]

| take | change vs env2 (turbo, shift 6, free plate) | wall | floor r | proxy mean (per second) |
|------|---------------------------------------------|------|---------|-------------------------|
| ENV4 | shift_video 12 only | 5 min 02 s | 0.998 | 2.33 (2.47 2.35 2.39 2.28 2.26) |
| ENV5 | base: no LoRA, 20 steps, shift 12 | 10 min 58 s | 0.998 | 2.93 (2.92 2.74 3.07 2.96 2.95) |
| env2 | (reference) | 6 min 13 s | 0.998 | 2.19 (2.46 2.19 2.15 2.11 2.11) |

- By the proxy, the shift changes nothing visible and the base recipe moves MORE, not less. That
  is the direction the research predicted (distillation damps motion; nobody reports turbo as
  faster). Pending the user's eyes, the LoRA hypothesis is refuted and the shift is not a lever
  for pace. Filed: `ssbs_env4_harbour_768p_truckright_turbo_shift12_seed20261027.mp4`,
  `ssbs_env5_harbour_768p_truckright_base_recipe_noLoRA_20steps_seed20261027.mp4`.
- Base costs 1.8x turbo wall time here (11 vs 6 min for 141 f).
- If the user confirms: pace on an open-water environment-only plate is a CONTENT PRIOR of the
  model (open water + no person = about 3x fast) that neither prose, shift, nor the LoRA moves.
  Remaining levers, in order: (1) `keyframes: still_last` (free start, settle onto the still:
  between "free" and "too static"); (2) the Motion-Speed-MMH3 slider LoRA at a negative weight
  (Civitai 2985458, via VPN); (3) a box-side retime (RIFE/Cadence-Retime) of a free plate at
  ~3x, replacing ffmpeg minterpolate; (4) writer rule: open-water B-roll is shot with her, or
  the environment-only shot moves to an enclosed place.
- **User (16:15): "Both are still sped up."** LoRA and shift recorded as not pace levers
  (techniques s.8). Four variables exhausted on the one scene: prose, first+last keyframes,
  shift, LoRA. Next on the user's instruction: the last-frame-only keyframe plate (`still_last`).

### 20.11 Last-frame-only keyframe plate (2026-10-28 16:30) [V floor check + proxy; user to watch]

- ENV6 = env2 + `keyframes: still_last` (one variable). 5 min 00 s, floor r 0.998. Filed as
  `ssbs_env6_harbour_768p_truckright_keyframes_still_last_seed20261027.mp4`.
- Motion proxy 0.49, flat (0.46-0.54 per second): the SAME regime as the first+last plate (0.53),
  not the free plate (2.19). Anchoring only the end frame pins the whole clip as hard as anchoring
  both; there is no free-running start by the proxy. [M] Pending the user's eyes: expected to read
  like ENV3 (real speed, too static, no truck).
- Standing picture of the harbour plate after seven renders: free = ~3x fast whatever the prose,
  shift or LoRA; any same-still guide = near-static. Nothing in between from the chain's own
  controls. Remaining: a DIFFERENT end keyframe (a GPT still with the boats displaced, so the
  guide forces a known small motion over the clip), the Motion-Speed slider LoRA, a box-side
  retime of the free plate, or the writer rule (open water with her, environment-only in enclosed
  places).
- **User (16:40): "Its still too static, there must be some way to have it not go at timelapse
  speed. Perhaps have some random person at a distance in the video, maybe that will make the
  timing more real? Although that wont work for all b roll shots, worth a try."** ENV6 verdict:
  same as ENV3. Next test, the user's: a distant anonymous figure as the real-time anchor in an
  otherwise singer-free plate. Built as an opt-in per-scene field (`anchor_figure: true`) that the
  clip pass honours as the ONE exception to the no-other-people rule, so it is a writer technique,
  not a hand-edited prompt; `rewrite_scene` gains `set` to put such fields on a scene for a
  one-variable test.

### 20.12 Anchor-figure plate fired (2026-10-28 16:40) [built; render running]

- Rewrite of the harbour scene with `set: {anchor_figure: true}` under the new clip rule, 0
  problems: "Far down the quay near the harbour mouth, small in frame with his back to the camera,
  one fisherman in dark oilskins stands at the last bollard. He hauls a rope hand over hand at a
  natural human pace and makes four slow pulls over the 6 seconds, his face never seen." Boats
  lift and settle once, everything else named still. Script
  `..._rw3_20261028_144353_rw3_20261028_163956.json`.
- Dry run: same still, refs, frames, no keyframes; only the prose differs from the free plate
  (env2). Fired 16:40:28 on the user's "worth a try": ENV7, seed 20261027, turbo, 1344x768.
- Readout: real speed = a distant human at human pace anchors the whole shot (then the writer
  can offer `anchor_figure` on environment B-roll where a figure fits the world); still 3x = the
  anchor must be the referenced singer or the pace is not about people at all.
- ENV7 filed 16:45:40 (5 min 12 s), floor r 0.998,
  `ssbs_env7_harbour_768p_truckright_anchor_figure_seed20261027.mp4`. Motion proxy 2.38
  (2.44 2.31 2.34 2.42 2.47), the free-plate regime (env2 2.19), not the keyframed one (0.5).
  By the proxy the distant figure did not change the pace; whether the figure rendered and at
  what pace, and whether the sea reads real, is the user's call from watching.
- **User (16:50): "Still timelapse speed even with the figure."** Recorded (techniques s.8). The
  harbour plate after eight renders: free running ~3x fast under every prose, shift, recipe and
  prose-figure tried; any same-still guide near-static; the referenced singer in frame = real.

### 20.13 What is left for singer-free B-roll at real speed (2026-10-28 16:55; proposal, nothing fired)

Facts in hand: free plate ~3x fast under prose, shift, base recipe and a prose figure; same-still
guides near-static; the referenced singer in frame = real. The box has the Frame-Interpolation
pack (RIFE VFI, FILM VFI, GMFSS Fortuna VFI, ...) and `MiniMaxH3ImageToVideo` exposes an optional
`first_frame` that our environment lane never uses. Ranked by information per render:

- **A. Enclosed environment-only scene, no build.** Script scene idx 4 "The proud at supper"
  (tavern, push in toward the hearth, 5.2 s, env-only, still on file). Tests whether the 3x prior
  is specific to open water and sky. Real speed = the writer rule writes itself: environment-only
  B-roll lives in enclosed or foreground-anchored places, open water and sky are shot with her.
  Still fast = the prior is "no referenced person", whatever the place. ~5 min.
- **B. Harbour plate on the image-to-video mode with the still as the FIRST FRAME (small build).**
  A different model mode from reference-to-video: the place arrives as pixels to continue, not
  as a subject to retain. The living-cover lane (I2V/FLF from a still) erred on the slow and idle
  side, the opposite prior. Opt-in `lane: "i2v"` on env-only scenes: FL2VA weights + its turbo
  LoRA, `first_frame` = the still, no reference pictures. ~5 min after the build.
- **C. A referenced person as the anchor instead of a prose one (needs one GPT still).** A
  distant fisherman supplied as a reference picture (<Picture 2>, back to camera, small) tests
  whether the anchor is the reference-subject mode rather than a human shape. If yes, singer-free
  B-roll can carry an anonymous referenced extra where the world allows one. ~5 min.
- **D. Box-side retime of a free plate (production fallback).** Render free (the pace the user
  judged right at 3x), then RIFE VFI or FILM VFI x3 on the box instead of ffmpeg minterpolate,
  which the user found unnatural. Needs a small interpolation graph; GPU minutes. Works for any
  plate regardless of the model's prior; costs the look of interpolated water.
- **E. Motion-Speed-MMH3 slider LoRA at a negative weight** (Civitai 2985458, via VPN). Community,
  trained on character action, turbo compatibility unstated. Last.
- Floor, regardless: the writer rule from A, plus `anchor_figure` kept for story use only.

Recommended order: A, then B (both cheap, both general), D built in parallel as the fallback.

### 20.14 "Solo" clarified; pace research pass 2 (2026-10-28 17:10; nothing fired)
- User: the one-person rule was never the brief. "Solo" meant the one lead singer and no whole
  band; other characters in scenes are fine where they make sense. Writer changed: SCENE_SYSTEM
  fixed fact rewritten (one lead, no band, `others` per scene, null by default); scene schema
  gains `others`; CLIP_SYSTEM writes only the people a scene lists (or `anchor_figure`), anonymous,
  one slow action each, never a band; `rewrite_scene` `set` accepts `others`. Memory + techniques
  s.4 updated. Not yet exercised on a script (the next pass-1 run on any song will show it).
- User: "there must be guidance out there why this is happening with the sped up motion, and
  ways round it". The first research pass (s.20.7) was LoRA-centred. A second, wider pass is
  running (web + node sources), briefed with all eight renders and with two leads the box's node
  list exposes that we have never used: `ref_videos` on MiniMaxH3ReferenceToVideo (a real-speed
  reference video as a pace anchor?) and `frame_rate` / `use_custom_motion` on MiniMaxH3DirectorCS,
  plus MiniMaxH3FunControlNetApply (control video), the I2V first-frame lane, free-AV audio for
  env clips (model generates its own ambient sound; song replaces it at assembly), slider LoRAs,
  Cadence-Retime vs RIFE/FILM. Report lands at
  `.mvwork/transfer/ssbs/research_h3_pace_v2_2026-10-28.md`; findings will be folded into s.20.13's
  ranking before anything is fired.

#### 20.14a What the box's own H3 source says about time (read 2026-10-28 17:25; primary)
Files pulled read-only from the box into `.mvwork/transfer/ssbs/boxsrc/` (core
`comfy_extras/nodes_minimax_h3.py`, `comfy/ldm/minimax/model.py`, the Director pack).
- **The model has one shared time axis for audio and video, in audio-latent units (1/40 s).**
  `FRAME_RESCALE = 5/3` per pixel frame (24 fps x 5/3 = 40 = `AUDIO_LATENT_FPS`); video latent
  tokens cover 1,4,4,4,4 pixel frames (the 17k+5 grid) and are placed on that axis by their real
  duration. So the model always knows exactly how many seconds its frames span relative to the
  audio. The clock is structural; the timelapse is the CONTENT prior for person-free water, not a
  frame-rate confusion. [primary: model.py `_video_t_spans`, `_video_t_grid`, `_audio_grid`]
- **There is no fps, speed or motion input anywhere.** The Director pack's `frame_rate` is only a
  timeline editing rate ("Output is always 24 fps"); `use_custom_motion` just routes the editor's
  reference-video track into `<Video k>` references. Dead ends. [primary: minimax_director.py]
- **Reference videos sit on the same time axis ahead of the target, at their real duration.**
  `MiniMaxH3ReferenceToVideo` takes up to 3 reference videos (5+ frames, 24 fps, cut to 17k+5,
  <= the target length), VAE-encodes them as reference blocks AND shows Qwen the frames at 2 fps
  with timestamps; the position grid advances the cursor by the clip's real span. The model
  therefore reads a reference video's motion per unit of REAL time. The official template declares
  it as `<Video k>` with retention `weak_reference`. [primary: nodes_minimax_h3.py lines 243-360,
  model.py 354-440; template in docs/MINIMAX_H3_PLAN.md s.4]
  => A real-speed water clip as `<Video 1>` ("pace and movement of the water only, not its
  composition") is the one untested lever that is grounded in the model's structure rather than in
  prose. Our chain already carries this slot (`camera_ref` -> `<Video 1>`, built for camera pace;
  the writer's declaration text is push-in specific and would need a pace-of-water variant).
  Caveat [U 2026-10-28, camera test]: a manufactured push-in reference added nothing visible to
  the camera move and tripled render time at 544p. Water pace is a different ask (dynamics, not a
  camera path), so that result does not decide this one.
- **Keyframe guides (fl2va) are placed by frame index on the same axis** (`cursor + FRAME_RESCALE *
  frame_index`), which is why a same-still guide at 0 and -1 reads as "nothing changed in N
  seconds" and pins the camera too. [primary: model.py keyframes block] Consistent with ENV3/ENV6.
- **Fun ControlNet** (`MiniMaxH3FunControlNetApply`: control_video, mask, source_video, strength,
  start/end percent) exists in core, but NO H3 control or model-patch file is on the box
  (models/controlnet and models/model_patches hold nothing for H3; loras = the two v1.1 turbo
  files + two old 4-step files; diffusion_models = ref2va + fl2va int8). A control-video pace lane
  would start with a download. [checked 2026-10-28 17:30 via :5080 fs/find]

### 20.15 Pace research pass 2 folded in: the ranked proposal (2026-10-28 17:55; nothing fired)
Report: `.mvwork/transfer/ssbs/research_h3_pace_v2_2026-10-28.md` (13 mechanism findings, 10
candidates, sources). Three key sources re-read by me before ranking (AharaOoO repo, matlowai
MAINodes README, HF ref prompt guide). What changes versus s.20.13:

**New mechanism (community, measured on other content): the model's RoPE time extent is the only
clock, and motion per frame scales with how much world time the model is told a span covers.**
matlowai's Motion Lab / True Clock / DyRoPE pack measured that a span whose clock is wrong runs
its background movers at the wrong speed ("overcranks steady background movers"), and that
rewriting the time grid to the true duration "holds world speed". Nobody has applied a UNIFORM
clock change to a scenic shot; the two reports are about held/dilated regions. [community +
primary source]
- A ready node exists: AharaOoO `MiniMaxH3NativeFPSAVSyncAc` (repo ComfyUI-MiniMaxH3-FrameRate-
  Adjustment-Ac): model + positive + av_latent in, `model_fps` 24-120; keeps the frames, scales
  the video time coordinates by 40/fps in `position_ids`, trims/pads the audio latent to
  round(frames/fps x 40). Placement: after LoRA and sigma shift, before the guider. Its own risk
  table: 25-30 subtle, 31-36 "stronger retiming", 37-48 "high risk of flicker, color drift",
  49+ research only. Needs ComfyUI >= 0.31 (box: newer). NOT on the box. [primary: repo README]
- DIRECTION [inference, not stated by the author]: at model_fps 48 each frame is placed at 40/48
  units instead of 5/3, so the model believes 141 frames span 2.9 s and should move the boats
  half as far; saved at 24 fps that reads as 2x slower, generated frames, not interpolated.
  36 = 1.5x. 72 (our 3x) sits in the research-only band. The first render decides the direction
  and the ceiling; nothing else can.
- Song-lock: the node resizes the audio latent, so the song mask cannot ride along. For
  environment plates the song is laid in at assembly anyway (free AV, opt-in `free_av` chain
  flag: skip SongMaskedAVContext; the streamer still muxes the master). Singer shots: n/a.
- Also found: `ScaleROPE` never reaches H3 (model.py reads no rope options): do not spend a
  render. Civitai 2985458 is the LIMB action slider, wrong tool; the motion-speed slider is FoS
  2951265 (+8/-8, one "felt too fast, slider addressed that" comment, training unstated).

**Official text now names the reference-video pacing role.** HF ref guide s.2.3: `<Video N>` is
for "camera movement, cuts, rhythm, or temporal structure"; s.4.1 retention line format
`<Video 1> (cut and pacing structure): weak_reference`; s.2.1 example "whose walking motion comes
from <Video 1>". So a pace reference is the one lever the publisher documents. [primary] Our only
trial (G6 V2 camera push-in) added nothing visible at 3x wall time, on a performance shot.

**Ranked, one variable each, harbour scene 3, seed 20261027, 141 f, 1344x768, turbo:**
1. **Clock compression ladder (free AV): model_fps 36, then 48.** The only mechanism-level lever.
   Build: install the pack via :5080 fs/write + ComfyUI restart; chain opt-ins `clock_fps` (node
   between LoRA/shift and BasicGuider) and `free_av`; backend restart after queue + jobs check.
   Readout: proxy toward ~1.1 and the user's eyes; flicker/colour drift at 48 is the known risk.
   ~6 min per render.
2. **Pace reference video as `<Video 1>` with the official pacing line.** Zero install; the
   `camera_ref` slot already carries a video; the writer needs a `pace_video` declaration variant
   ("<Video 1> (pacing and motion speed of the water): fully_preserved - the water and boats move
   at exactly this real speed; nothing of its composition carries over"). Source clip = USER's
   call: phone footage of real water, free stock (a download, needs a yes), or a person-free crop
   of our HERQUAY take (the water ran at real speed there by the user's eyes; AI water as a pace
   exemplar, zero cost). Song-lock survives. ~3x wall time (ref tokens ride every step).
3. **RIFE x3 + V2V refresh (production fallback).** Interpolate the free env2 plate 3x, take the
   first 141 frames, re-denoise with `H3V2VGranularFractionalDenoise` at ~0.6 with the song slice
   as source_audio (audio_strength 0) so every frame is generated over the interpolated motion.
   All nodes on the box; new small graph builder. Answers whether a partial re-denoise makes the
   interpolated water natural without re-accelerating it. Song-lock survives.
4. Then the cheap writer-rule settlers from s.20.13: the enclosed tavern scene (A) and a
   referenced extra (C, one GPT still). I2V first-frame lane (B) stays queued; nothing found on
   its pace either way.
Not levers (do not spend renders): prose variants, sigma shift, the turbo LoRA, same-still
guides, ScaleROPE, Director frame_rate, the limb slider. Fun ControlNet: a real lever (frame-
locked to a control video) but needs a composition-matched control video, a download, and an
unknown VRAM footprint on 24 GB: last.

Recommended: 1 first (two renders, one install), 2 second with the HERQUAY crop unless the user
has real footage, 3 built in parallel as the fallback.

### 20.16 Clock-compression lane built; harbour plate dry-run at model_fps 36 (2026-10-28 18:30; NOT fired)
- Pack `AharaOoO/ComfyUI-MiniMaxH3-FrameRate-Adjustment-Ac` (v0.1.2, commit a5c8bb1, GPL-3.0, no pip
  deps) read in full on the Mac first (`.mvwork/transfer/ssbs/boxpack/`): it clones the model, wraps
  `extra_conds` to rewrite the PackedLayout's time axis (video rows at 40/fps per frame, refs and
  keyframes retimed on the same cursor), trims/pads the audio latent to round(frames/fps x 40),
  tags the conditioning with `minimax_frame_count`, and refuses a flat noise mask when the audio
  length changes. Deployed by :5080 fs/write (9 files), pycompile OK on the box, ComfyUI restarted
  (queue was empty), node `MiniMaxH3NativeFPSAVSyncAc` present in /object_info (model_fps 12-120).
- Chain opt-ins in `build_h3_chain`: `free_av` (no SongMaskedAVContext; the sampler takes the
  ref2v latent directly; single-clip only) and `clock_fps` (requires free_av; node id base+7 after
  the guides, before BasicGuider; model, positive and latent all pass through it). Baseline graph
  unchanged without the flags (offline test). `render_scene` validates: clock needs free_av,
  free_av needs an environment-only single-clip scene; dry_run and job meta carry both.
- Dry run (`.mvwork/transfer/ssbs/dry/env8_clock36_*.json`): scene 3 harbour, one-motion prose,
  location still from the library, 141 f, 1344x768, turbo, seed 20261027, free_av true,
  clock_fps 36 -> the model believes 3.92 s; saved at 24 fps = 5.88 s; expected 1.5x slow-down
  if the direction inference holds. The two rejection paths returned the right 400s.
- Two variables move against ENV2 (free AV and the clock). If 36 reads slower, a free_av-only
  control at 24 attributes the effect; if it reads the same, that control is unnecessary.
- Known unknowns for the run: the streamer receives a latent whose audio part is 157 ticks
  instead of 235 (it muxes the master, so this should be ignored, [H] untested); turbo LoRA under
  non-native coordinates (the author warns distilled LoRAs may interact).

### 20.17 ENV8 clock-36 plate filed (2026-10-28 18:52) [V floor + proxy + graph; user to watch]
- Job fd25fdbb; the running graph on the box was checked while it ran: node 107
  `MiniMaxH3NativeFPSAVSyncAc` model_fps 36 between ReferenceToVideo and BasicGuider, no song
  node, sampler fed from the clock node's latent. Execution 324 s (env2 ~465 s: the audio latent
  is 157 ticks instead of 235, so the attention sequence is shorter), 141 f at 24 fps, 5.875 s,
  the streamer muxed the master without complaint (the short-audio-latent unknown is settled).
- Floor check r 0.998 (the muxed master, as expected on free AV). Filed as
  `ssbs_env8_harbour_768p_truckright_clock36_freeAV_seed20261027.mp4`; frames
  `frames/ENV8_sheet.png`; proxy script now saved as `renders/motion_proxy.sh` (reproduces env2
  2.19, env7 2.38, env6 0.49 exactly).
- Motion proxy 2.57 vs env2 2.19: MORE inter-frame change, not less. The proxy cannot separate
  water pace from camera travel or flicker (the pack warns of flicker above 30), so this is not
  a verdict either way; the sheet shows a continuous truck right with composition held and no
  obvious colour drift at 1 s spacing. The user watches; two variables moved against env2
  (free AV and the clock), so if it reads slower a free-AV-only control at 24 attributes it.
- VERDICT [U 19:05]: "Both still fast, the 2nd one (env2, native clock) looks slightly better, but
  still like a timelapse." Clock compression at 36 is not a pace lever on this content; the
  proxy's rise was real (more change, not slower water). 48 and 72 are not worth firing: the
  author's own risk band starts at 37 and the direction gave nothing at 36. Lane stays in the
  chain, off by default. Day's tally on the harbour: nine renders, one positive (her in frame).

### 20.18 Image-to-video lane built; harbour plate dry-run on it (2026-10-28 19:35; NOT fired)
- User: "I cant believe that no-one else has this problem, this is a basic landscape scene." The
  answer: nobody shoots a landscape from a still the way our environment lane does. Everyone
  else runs IMAGE-TO-VIDEO (the still is frame 0 and the model continues it); our env lane hands
  the still to REFERENCE-to-video as a subject to retain, the mode built for her face and body.
  The research pass found no report of anyone pacing a seascape through the reference mode.
- Built, opt-in `lane: "i2v"` on render_scene (env-only scenes with a location still; no
  keyframes, no camera_ref): FL2VA weights + `minimax_h3_fl2v_turbo_8step_v1.1_768p` at shift 6/3
  (CHAIN_TURBO_FL2VA), `MiniMaxH3ImageToVideo` with `first_frame` = the location still fed raw
  (16:9 still onto 1.75 canvas: a 1.5 % stretch), NO reference pictures, song lock unchanged.
  Prompt = the official three-section FL2VA format behind the author's exact opening line
  (`compile_prompt_i2v`), with Claude's four fields reused verbatim so the lane is the only
  variable against ENV2. `_h3_chain` uploads `first_frame_id`; dry_run and meta carry `lane`.
- Dry run `.mvwork/transfer/ssbs/dry/env9_i2v_*.json`: scene 3, 141 f, 1344x768, seed 20261027,
  turbo, lane i2v, first frame = harbour still 32de3cf9. Three rejection paths return the right
  400s. (The dry run's `ref_ids` line listed the still although the chain gets none; fixed in
  code, live at the next backend restart; the submitted graph is unaffected.)
- Prior from our own history: the living-cover I2V/FLF lane erred SLOW and idle (s.12). If this
  plate runs at real speed the env lane switches to i2v by default for environment-only scenes
  (opt-out), and the ref2va env path stays for keyframed plates.

### 20.19 ENV9 image-to-video plate filed (2026-10-28 20:00) [V floor + proxy + graph; user to watch]
- Job e531385c; the running graph was read off the box queue: node 100 `MiniMaxH3ImageToVideo`
  with `first_frame` = the harbour still, fl2va weights + fl2v turbo LoRA, no ReferenceToVideo
  node, song node present, prompt opening with the author's exact line. Execution 381 s.
- Floor check r 0.998. Filed `ssbs_env9_harbour_768p_truckright_i2v_firstframe_fl2va_seed20261027.mp4`,
  frames `frames/ENV9_sheet.png`.
- Motion proxy 1.45 vs ENV2 2.19 (ENV8 2.57, keyframed plates ~0.5): the first drop of the day on
  a free-running plate, a third less change per frame, well above the frozen regime. The sheet:
  the clip keeps the still's own look (the window lamps, the choppier water of the photograph)
  since frame 0 IS the still; the truck right is gentle and continuous; no colour drift at 1 s
  spacing. Pace is not readable from stills; user to watch against ENV2.

### 20.20 ENV9 verdict: REAL SPEED. Image-to-video becomes the environment default (2026-10-28 20:05)
- User: "ENV9 runs at real speed." After nine failures on the reference lane, the lane itself was
  the variable: the reference mode retains the place as a subject (timelapse prior for scenery
  without a referenced person); image-to-video continues the still as pixels (real footage pace).
- Changed: `render_scene` lane default = `i2v` for environment-only scenes with a location still
  and no keyframes / camera_ref; `lane: "ref2va"` opts out; keyframes or a camera reference imply
  the reference lane. Routing checked through the real route on four cases (env default -> i2v,
  explicit ref2va with the still as <Picture 1>, keyframes -> ref2va, a singer scene -> ref2va
  with three pictures). Techniques s.8 rewritten; memory updated.
- Validation plate: scene idx 8 "Swell in the harbour" (83.84-93.02, 9.2 s, push in with LARGE
  amplitude toward the harbour mouth, spray over the outer wall; text predates the one-motion
  rule, i.e. the harder case) through the DEFAULT path with no lane parameter. Dry run
  `.mvwork/transfer/ssbs/dry/env10_scene8_default_*.json`.


### 20.21 ENV10 validation plate filed (2026-10-28 20:30) [V floor + graph + profile; user to watch]
- Scene idx 8 through the DEFAULT path (no lane sent): the running graph was ImageToVideo with the
  still on first_frame, fl2va weights, no reference node. Job cc6a30fc, 226 f = 9.42 s, exec 628 s
  (2.8 s/frame on fl2va at 768p). Floor check r 0.997. Filed
  `ssbs_env10_scene8_swell_768p_pushin_large_i2v_default_seed20261027.mp4`, frames
  `frames/ENV10_sheet.png` (9 frames).
- Motion proxy 9.04, four times anything today, so it was checked before handing over: per-second
  profile 5.2, 7.1, 8.7, then a flat 9.5-10.3 to the end, max pair 14.9 at frame 196; no spike,
  so no cut or dissolve. The sheet shows one continuous LARGE push toward the harbour mouth
  across the whole frame with breaking spray and whitecaps, so nearly every pixel moves every
  frame (camera travel + sea), where ENV9's camera barely moved and its water was calmer. The
  proxy is not comparable across scenes; pace is the user's call. Text was the pre-one-motion
  version (three swells with a rhythm) and the camera the large-amplitude kind: the hard case.
- VERDICT [U 20:40]: "Speed is fine." Camera note: the push ran along the quay path the still
  shows, not out into the sea; on the first-frame lane the move follows the image's geometry
  (techniques s.7/s.8). RULE VALIDATED on two environment scenes (3 and 8). Open item 1 of
  techniques s.14 (environment pace) is closed; the pace reference video (s.20.15 fix 3) and the
  retime+refresh fallback are no longer needed and stay documented only. The clock and free_av
  opt-ins remain in the chain, off.

### 20.22 Writer rerun on She Sings Before the Storm with the new rules (2026-10-28 20:50; no render)
- User: "Maybe think about rerunning the writer now we've got rid of the rule banning other people
  appearing?" Reran `/api/mv/write` with the ORIGINAL SSBS request (`write_library_request.json`,
  same style, lyrics, refs, Opus medium) and the default `locations: auto` (this song's five
  stills offered). 676 s. Output `mv_scripts/she_sings_before_the_storm_20261028_193924.{json,md}`.
- Result: 25 scenes, 30 clips, 0 prompt problems, aligned on vocals, cover 0.894. All five known
  places REUSED (cliff_road x7, cliff_wall x4, harbour x3, cottage_kitchen x3, tavern x3) plus ONE
  new place, `cliff_edge` (5 scenes, the chorus "she sings before the storm" setups on the
  headland lip with a huge sky): its still prompt is the user deliverable (in the .md).
- `others` used on 6 of 25 scenes, all where the lyric names people: the fisherman hauling the
  line (3, env-only), the proud at supper (4, env-only, three fishermen laughing once over the
  clip), the quay (12, her + one fisherman coiling a line), the hearth (13, her unheard by three
  diners), the dance (18, five or six couples in a slow reel in soft focus behind her), laughing
  seaward (19, env-only, four fishermen walking to the boats). Clip prose checked on 4 and 18:
  anonymous, faces unreadable, one counted action, never a band. `anchor_figure` unused (right:
  it is a story flag now). Environment one-motion rule held in the env clips read.
- Rendering note: env-only scenes 3, 4 and 19 now carry prose people AND default to the i2v
  lane; prose people on the first-frame lane are untested (on ref2va the ENV7 fisherman rendered
  fine). Scene 4 (tavern, static camera, 5.2 s) is the cheapest first check.

### 20.23 ENV11 tavern plate: prose people on the I2V lane (2026-10-28 21:25) [V floor + frames; user to watch]
- Scene 4 of the rerun script through the default path (job 8f15c8fb, i2v, tavern still as frame
  0, no references). Exec 326 s, floor r 0.998, proxy 0.36 (static room, small men). Filed
  `ssbs_env11_tavern_768p_static_three_fishermen_i2v_default_seed20261027.mp4`; frames
  `frames/ENV11_sheet.png`, arrival tile `frames/ENV11_arrival_0-29_every3.png`.
- The three fishermen RENDER, with the counted action as written (tankards up by ~2 s, heads
  back at ~3 s, settled by ~5 s), faces lit but small; the room, lamps and fire hold still.
- THE FLAW: frame 0 is the person-free still, the room stays empty for ~0.6 s, the men ghost in
  over ~0.4 s (frames 15-21) and are solid at ~1 s. The first-frame lane continues the still's
  pixels; people the still does not contain must dissolve in. Measured per-frame change: ~0.05
  for frames 2-12, rising through 0.2-0.6 over frames 13-24.
- General fixes (both opt-in, user to pick; nothing built):
  A. Per-scene opening still WITH the people for env-only scenes that carry `others` (the scene
     schema already has `opening_still`; the writer would add the listed people to it; the user
     generates it; render_scene takes it as the first frame instead of the location still).
     Frame 0 then holds the people and the location still stays person-free for reuse.
  B. Pre-roll: render such scenes one grid step longer (+17 f) with the song window starting
     ~0.7 s early, and drop the first 24 frames at assembly (the wind-intro offset mechanism).
     No new still; costs 17 frames per clip and makes assembly aware of per-scene trims.
  C. Reference lane (`lane: "ref2va"`) for env scenes with people: present from frame 0, but
     the timelapse prior for outdoor water/sky; untested in an enclosed room.

### 20.24 Fix A built + the unison rule (2026-10-28 21:50; nothing fired)
- User picked the opening still, and flagged that the three men "all do the same thing at the
  same time and look unnatural": the env one-motion rule leaked onto people. Writer changes:
  pass 1 `others` = each person's own business, never unison; `opening_still` must contain the
  listed people already in their opening poses (it IS frame 0 on the i2v lane); pass 2 people
  bullet rewritten (staggered, independent, idle motion, start pose matches the opening still,
  group rhythm in seconds, no choreography). `opening_still_prompt()` composes the frame-0
  prompt mechanically for scripts written before the rule (location still prompt minus "no
  people" plus the listed people, unposed, different poses, faces away).
- Routes: `POST /api/mv/opening_still {script, scene}` -> the prompt (+ source scene/composed);
  `POST /api/mv/scene_still` (multipart file, script, scene) files the generated picture as an
  opening still (kind opening_still, no location row); `render_scene` accepts `first_frame_id`
  (i2v lane only, a library still) which replaces the location still as frame 0; dry_run shows
  `first_frame_from` request|location.
- Scene 4 clip rewritten under the new people rule (pass 2 only, `rewrite_scene` with an empty
  set); the opening-still prompt for the tavern is the user deliverable. Render waits for the
  still.

### 20.25 ENV12 tavern plate with the opening still (2026-10-28 22:20) [V floor + ghost check + frames; user to watch]
- User generated the opening still from the composed prompt (1672x941, three men in three
  different poses at the far table); filed by `/api/mv/scene_still` as still 5efaeba1; scene 4
  of the rewritten script rendered on the i2v lane with `first_frame_id` = that still
  (`first_frame_from: request`, graph checked on the box: node 30 = the new still). Job ebab28d3,
  exec 322 s, floor r 0.998. Filed
  `ssbs_env12_tavern_768p_static_three_fishermen_openingstill_staggered_i2v_seed20261027.mp4`.
- GHOSTING GONE [M]: first-second per-frame change stays 0.04-0.33 with no rise (ENV11 rose
  0.05 -> 0.6 over frames 13-24); the arrival tile shows the men present from frame 0.
- INDEPENDENT MOVEMENT [frames]: t0 left man laughing with tankard up, middle at shoulder
  height, right leaning back; t2 middle man's tankard up; t4 left man lowered, right man
  drinking. Whole-clip proxy 0.16 = locked camera + small figures. Naturalness of the movement
  is the user's call.
- If the user passes it: the people mechanism is complete end to end (others -> opening still ->
  first_frame_id -> staggered clip) and generalises to scenes 3 and 19 (harbour, people) with
  one still each.
- VERDICT [U 22:30]: "The tavern passes." People mechanism VALIDATED end to end. Next: scenes 3
  and 19 (harbour, people) rewritten under the people rule, then their opening-still prompts.
- Scenes 3 and 19 rewritten under the people rule (19 s each, 0 problems; both clips now open
  with "At the start ..." poses). Final script:
  `she_sings_before_the_storm_20261028_193924_rw4_20261028_200608_rw3_20261028_202026_rw19_20261028_202045.json`.
  Composer tightened (pose extraction stops at motion sentences; "distance and shadow" instead
  of lamplight). Opening-still prompts delivered to the user:
  `.mvwork/transfer/ssbs/dry/scene3_opening_still_prompt.txt`, `scene19_opening_still_prompt.txt`.
  When the stills come back: `/api/mv/scene_still` (scene 3, scene 19) -> render with
  `first_frame_id` on the default lane.

### 20.26 Scenes 3 and 19 with opening stills (2026-10-28 23:05) [scene 3 filed; scene 19 rendering]
- User generated both stills from the composed prompts (1672x941 each); filed as b4c2d2ee
  (scene 3, one hauler) and 6f6e7b51 (scene 19, four walking). Both fired on the default lane
  with `first_frame_id`; graphs checked on the queue (node 30 = each scene's own still, lengths
  141 and 175).
- ENV13 (scene 3, job 3793ed2a): exec 376 s, floor r 0.998, proxy 0.64; first-second change
  flat at ~0.4 with a ~1.0 tick every fourth frame (the 17k+5 token cadence, also faintly visible
  on ENV12), no ghost rise. Frames: the hauler is in frame 0 bent to the line and leans back as he
  pulls; truck right continuous. Model-added element: the cottage windows light up amber within
  the first second (not in the still or the prompt; the same warm-window drift appeared on
  ENV9). Filed `ssbs_env13_scene3_harbour_768p_truckright_one_hauler_openingstill_i2v_seed20261027.mp4`.
  User to watch for pace and the hauling motion.
- ENV14 (scene 19, job f068bb2d): exec 456 s, 175 f, floor r 0.999, proxy 1.15. Filed
  `ssbs_env14_scene19_harbour_768p_pushin_four_walking_openingstill_i2v_seed20261027.mp4`. The men
  are in frame 0 and walk; BUT the frames show the WHOLE WORLD dissolving from the still's storm
  (slate sky, choppy sea, shutters barred) into a golden calm evening with open shutters inside
  the first second, plus a fifth man appearing in the nearest boat by ~4 s. Cause [M, mine]: the
  opening-still composer used the LOCATION still prompt's weather and light; scene 19's own
  setting is "a clear evening, pale gold sky, calm glassy harbour, shutters open" and the clip
  text says so, so frame 0 contradicted the clip and the model reconciled the two by dissolving.
  Scenes 3 and 4 matched their location's first-scene weather by luck.
  FIX (general): the composer now appends THIS SCENE'S STATE (setting + lighting) as an explicit
  override of the location still's weather, time of day, light and open/closed state. Scene 19
  prompt regenerated (`dry/scene19_opening_still_prompt_v2.txt`); the user regenerates the still.
  The writer's own `opening_still` field already takes the scene's light (schema), so new
  scripts are unaffected.
- VERDICT on ENV14 [U 23:15]: "nope, that ones no good, one of the men flies through the air onto
  the boat" and "They cant step down to the boat, its too far away". Two faults: (1) my
  opening-still composer carried the location still's storm light under a clear-evening scene
  (fixed above, state override); (2) the writer's action "one steps down into the nearest boat"
  is a large body transition for an extra AND impossible in the still's geometry (the boats are
  moored off the quay on ropes), so the model rendered a leap. Rule added to both writer passes:
  extras keep to small ordinary actions, never climbing/jumping/stepping into or out of a boat/
  mounting/falling, and nothing the still's geometry makes impossible (techniques s.8).
- Scene 19 rewritten with `set: {action: "... none of them boards ..."}` under the new rule
  (21 s, 0 problems; each man walks at his own pace, laughs staggered, no boarding). Final
  script: `..._rw19_20261028_205405.json`. Opening-still prompt v3 (scene state override + new
  poses): `.mvwork/transfer/ssbs/dry/scene19_opening_still_prompt_v3.txt`. The user regenerates
  the still; then render scene 19 again with it. Scene 3 (ENV13) awaits the user's verdict.
- VERDICT on ENV13 [U 23:25]: "Scene 3 passes." Second people scene passed on the i2v lane with
  an opening still (one hauler, truck right). Scene 19 awaits the regenerated still (prompt v3).

### 20.27 ENV15: scene 19 again with the golden-evening still (2026-10-28 23:55) [V floor + frames; user to watch]
- New still 41308b2d (clear evening, shutters open, four men walking, coiled rope on the last
  man's shoulder) filed; scene 19 of the final script rendered with it as frame 0 (job f40c39fd,
  175 f, exec 433 s). Floor r 0.999, proxy 0.97. Filed
  `ssbs_env15_scene19_harbour_768p_four_walking_noboarding_goldenstill_i2v_seed20261027.mp4`.
- The world HOLDS: golden evening, open shutters and glassy water through all seven 1 s frames;
  first-second change 0.3-2.3 with only the 4-frame token tick (ENV14 had the 2-3.5 dissolve
  spikes). The state override in the opening still fixed the dissolve [M]. No boarding, no leap
  in the frames.
- To watch for [frames]: by ~4 s the group looks turned toward the camera, walking back toward
  it, where the clip has them walking away toward the boats. If the user sees it: extras who
  walk away from camera may turn to face it mid-clip (one sample), a candidate rule for the
  people bullet ("they keep walking away; their backs stay to the camera to the end").
- VERDICT [U 2026-10-29 00:05]: "Scene 19 passes." ALL THREE PEOPLE SCENES VALIDATED (4, 3, 19).

### 20.28 Day's record, 2026-10-28 (what is now true)
- Environment-only B-roll runs at REAL SPEED on the image-to-video lane (location still as
  frame 0, FL2VA weights, no reference pictures); it is the default for environment-only scenes.
  Nine reference-lane levers (prose x3, shift, LoRA/steps, two same-still guides, a prose
  figure, RoPE clock compression) did nothing; the lane was the variable. Validated on scenes 3
  and 8 (plain) and 4, 3, 19 (with people). [U]
- "Solo singer" means one lead and no band; other people are allowed per scene via `others`,
  written anonymous, staggered, never in unison, small ordinary actions only, nothing the
  still's geometry forbids. They enter the picture through a per-scene OPENING STILL that
  already contains them in their stated opening poses and carries the scene's own weather and
  light (`/api/mv/opening_still` -> user generates -> `/api/mv/scene_still` -> `render_scene`
  `first_frame_id`). Without it people ghost in over the first second; with a wrong-weather
  still the whole world dissolves. [U/M]
- Writer rerun on the second song under the new rules: 25 scenes, five stills reused, one new
  place (cliff_edge, still prompt delivered), people on six scenes, 0 prompt problems.
- Chain opt-ins that stay, off by default: `free_av`, `clock_fps`, `keyframes`, `lane: ref2va`,
  `anchor_figure`. The AharaOoO frame-rate pack is installed on the box.
- Still open (techniques s.14): camera extent seeds at 768p; the tracking-shot wobble (second
  seed); final assembly of a whole song; the regression set; the cliff_edge still; the warm
  window lamps the model adds on the harbour at dusk (ENV9, ENV13; cosmetic, unaddressed).

### 20.29 cliff_edge still filed; the second song is fully provisioned (2026-10-29 00:30; nothing fired)
- User generated the cliff_edge still from the rerun's prompt (1672x941); filed by
  `/api/mv/locations/import_still` as 909f694e with the location row. The library now offers six
  places for this song. The five chorus scenes (5, 7, 16, 20, 22) dry-run on the reference lane
  with her two pictures + the cliff_edge still as <Picture 3>.
- Provisioning complete for a full render: six location stills, three opening stills (scenes 3,
  4, 19), her face + body. Sized at ~6 h of box time for 25 scenes / 30 clips at 768p turbo
  (s.20.28 table); three scenes already rendered on the final text. What does not exist yet: a
  whole-script batch runner and the assembly builder (both Mac-side, no GPU).

### 20.30 Whole-script batch + assembler built; the second song queued overnight (2026-10-28 23:20)
- `POST /api/mv/render_script {script, ref_ids, skip?, only?, existing?, dry_run?, knobs}`: every
  scene through `render_scene` (default lanes, each scene's recorded `opening_still_id`, library
  location stills), submitted with `free=False` after the first so the box is never freed under a
  running job (ComfyUI swaps fl2va/ref2va itself). Writes a manifest under `.mvwork/mv_batches/`
  (scene -> job_id; `existing` scenes carried in). `_h3_dispatch` now honours `free` (None auto,
  False never, True always).
- `POST /api/mv/scene_still` now records `opening_still_id` on the scene in the saved script;
  `POST /api/mv/scene_still/set` records an already-filed still; `render_scene` on the i2v lane
  takes request > scene record > location still as frame 0 (`first_frame_from`).
- `POST /api/mv/assemble {batch | script+jobs, partial?, crf?, fade?}`: each scene's render
  trimmed from frame 0 to its window (end - start), re-encoded 1344x768 24 fps x264, concat in
  script order, the song laid under from scene 0's start with a 1.5 s fade at the end. `partial`
  cuts what exists with each scene's own muxed song window (progress check). VERIFIED partial on
  the three finished scenes: 17.93 s vs 17.9 expected, 1344x768 24 fps. The full path (song mux)
  runs in the morning.
- Whole-script dry run: 22 dry + 3 existing, 0 errors; lanes as designed (env-only -> i2v with
  the location still or the scene's opening still; her scenes -> ref2va with three pictures).
- QUEUED 23:19: 22 jobs, 0 errors, box queue 1 running + 21 pending; scene 0 (three clips, 362 f
  each, ref2va) first. Manifest:
  `.mvwork/mv_batches/..._batch_20261028_231*.json` (path in `renders/batch_manifest_path.txt`).
  Estimate ~6 h. Watcher `renders/watch_batch.sh` logs to `renders/batch_watch.log` every 10 min;
  `renders/batch_status.sh` prints the per-scene table any time. Morning: `batch_status.sh`, then
  `POST /api/mv/assemble {"batch": <manifest>}`; a scene that failed gets re-queued with
  `render_script {only: [idx]}` and its job id patched into the assemble call's `jobs`.
- The backend (pid on :8000) must stay up overnight: it polls the box and files each finished
  render into the library. No backend restarts until the batch is done.

### 20.31 Scene 0 in the partial cut: "basically unusable" (2026-10-29 01:05) [U + frames]
- User on the first 30 s: the building and the person change and appear from nothing; candles
  and other scenes mix in and meld. Frame sheet at 1 s (`frames/scene0/scene0_1s_sheet.png`):
  (a) 0-3 s: the chapel changes shape (a white translucent building at 1 s), she fades in at the
  far end of the road at ~3 s; (b) 15-31 s: lit candles appear, then HER FACE PORTRAIT at
  close-up double-exposed over the landscape with candles, then back to the wide shot; (c) 31-42 s
  (clip 3): clean wide shot, she walks to the chapel. Per-second change 2.0-2.5 for 0-22 s,
  ~1.0 after 26 s.
- Regime: the first CHAINED take of this song (3 clips x 362 f, 42 s, extreme wide, she a tiny
  figure, no singing). Every validated plate today was ONE clip; her validated shots were
  medium close-up (sing) and a wide tracking walk (141 f).
- Hypotheses, ranked by how well they fit the frames [H, nothing measured yet]:
  H1 (fits best): the inserted image IS her reference portrait. Every clip of hers carries
  "fully_preserved - retain the referenced face, eyes, hair, skin ..." while the prose puts her
  at 20 px at the far end of a road: the retention cannot be met in the picture, so the model
  pays the reference debt by superimposing the portrait. The candles come from the LOOK sentence
  in every prompt ("candle and lamp interiors burn deep amber"), which names a prop the exterior
  cannot hold. Fix (general): distant/wide shots pass the BODY picture only with a
  partially_preserved retention (silhouette, hair, wardrobe; "her face is not readable at this
  distance"), and the style sentence names grade and camera only, never props.
  H2: the chain itself (clip 2's protected context + "continue" prose) weakens the scene prior;
  the insertion sits in clip 2. Less specific.
  H3: ref2va has no frame-0 pin, so the first ~2 s converge from noise (chapel morph, her
  fade-in); invisible on close shots, glaring on an extreme wide. Fix candidates: an opening
  still WITH her (GPT from her reference) anchored at frame 0 via the existing AddGuide path, or
  keep extreme wides to the I2V lane with her absent and bring her in on a closer setup.
- Same regime still queued: 17 (3 clips, extreme wide, ~60 min) and 24 (2 clips, extreme wide,
  ~40 min) at the end of the queue; single-clip extreme wides 6, 15, 21 (sing_away) are cheap
  data points either way.

### 20.32 Fix applied overnight (2026-10-29 01:35) [built; renders queued; verdict in the morning]
- User: "just fix it, and report back in the morning". Done under H1 (s.20.31), two moves at once
  because both are parts of the same leak:
  1. DISTANT SHOTS GET THE BODY PICTURE ONLY, with a partial retention. `mv_writer.is_distant`:
     extreme wide, or wide when she is not singing to camera (not medium wide). `compile_prompt`
     then declares her from <Picture 1> (body) with "partially_preserved - silhouette, hair, build,
     wardrobe; her face is not readable at this distance and is never shown closer; no close
     view of her is inserted", and the location still becomes <Picture 2>; `render_scene` uploads
     only the body still for such scenes (`distant` in dry_run and meta). Close shots unchanged.
  2. THE LOOK SENTENCE NAMES NO PROPS. Pass-1 schema rule (grade, light quality, contrast, colour,
     camera weight only; never candles, lamps, fires, windows). The final script's `look` patched
     in place to drop "candle and lamp interiors burn deep amber" (recorded in `rewrites`);
     interiors keep their amber through their own lighting fields.
  Frame-0 convergence on ref2va (chapel morph, her fade-in, H3) is NOT addressed tonight.
- Queue surgery: pulled the pending same-regime jobs 15, 17, 21, 24 and 12 (6 was already
  running on the old prompt; kept as a data point). Re-queued 0, 12, 15, 17, 21, 24 on the fix
  behind the batch (free=False). Scene 0's new job checked on the box: two pictures (body +
  cliff_road), partial retention in the prompt, three clips. Merged manifest (original batch
  with the six job ids replaced) is what `batch_status.sh` and the morning assemble read:
  `renders/batch_manifest_path.txt`.
- Morning: `batch_status.sh`; frames of the new scene 0 vs the old; then assemble. If scene 0
  still shows the portrait insertion, H1 is wrong and the candidates are the chain itself (H2)
  and frame-0 anchoring (H3).

### 20.33 Direction for tomorrow (user, 2026-10-29 01:45): first frames drive the video
- User: "This is why I said I think we need more first frame generations to drive the vid, rather
  than relying on adding character purely through reference. Gives better control."
- What today supports: every shot that started from a generated frame (the harbour plates, the
  tavern, the hauler, the walkers) held its world, its pace and its people from frame 0; the
  failures left are the reference lane converging from noise (scene 0: chapel morph, her
  fade-in) and reference debt paid mid-chain (the portrait insertion, fix under test).
- Design to build tomorrow (not started): a per-scene OPENING STILL WITH HER for scenes she is
  in (GPT from her reference pictures; the writer's `opening_still` already describes the first
  frame with her in the setting), used as frame 0:
  - environment-only scenes: already the I2V lane (still as first_frame);
  - her scenes: the reference lane keeps her face/body pictures for identity and lip-sync, and
    the opening still is anchored at frame 0 through the existing AddGuide path (`keyframes:
    still_first` today anchors the LOCATION still; make it take the scene's opening still). To
    test: one-variable on scene 0 (same prompt, fix included) with and without the frame-0
    anchor; and a sing_to_camera scene to confirm lip-sync survives the anchor.
  - fallback if the anchor pins the camera (as same-still first+last did): the I2V lane with the
    opening still for her non-singing scenes, where identity comes from the still itself.
- Morning first: the overnight batch results and the scene 0 verdict (s.20.32).

### 20.34 Morning after the overnight batch (2026-10-29 06:30) [frames; user to watch the cut]
- Batch: 25/25 scenes rendered by 06:21 (box queue empty). Audio floor check on every scene vs its
  song window: envelope r 0.996-1.000 at lag 0.00 s; no drift anywhere
  (`renders/batch_floor_check.txt`). Renders overshoot the scene window on the 17k+5 grid as
  expected (scene 24: 28.5 s rendered for a 17.9 s window), the assembler trims to the window.
- Scene 0 re-render on the H1 fix (body picture only + partial retention + props-free look), old
  df8568aa vs new d15c63bd, 1 s frame sheets in `renders/scene0_cmp/`:
  - GONE: the reference portrait and the candles superimposed mid-chain (old t=17-32). Nothing
    of the portrait or any interior appears in the new render. H1 holds for the CONTENT of the
    insertion: what was bleeding in was the face reference picture itself and the look
    sentence's candle clause.
  - STILL THERE (expected, not addressed): frame-0 convergence. She is absent at t=0 and fades in
    on the path by t=2; the chapel changes shape over the first seconds and then carries a dark
    net-like drape on its left side from t=6 to the end (a new artefact, not in the old render).
  - NEW PROBLEM: the shot does not stay distant. She walks toward the camera for the whole chain
    and is a medium shot by t=30-36 (inside the 36.72 s window), against "never shown closer".
- The other re-queued regime scenes (same fix), sheets in `renders/regime_cmp/`:
  - 12 Down the quay (1 clip, 5.6 s): holds. She mid-distance on the quay, the fisherman down the
    quay, boats and sea consistent.
  - 15 The road in the gale (1 clip, 7.2 s): holds. She small by the chapel, storm sky stable.
  - 21 Alone in the storm (1 clip, 7.5 s): holds. Night storm, arms spread, no insertion.
  - 24 Singing still (2 clips, 28.5 s rendered): holds for the whole chain. She stays small on
    the path, the house keeps its shape.
  - 17 The finest evening (3 clips, 42 s rendered): the long-chain failure again. First 12 s wide
    and clean (a second white building briefly doubles beside the chapel at t=4-5), then at
    t=25-31 a cut to a medium close shot of her (scene-consistent this time: same wardrobe, same
    sky, no portrait, no candles), a translucent crossfade back at t=31, wide again to the end.
  - 6 Between the dark and morning (1 clip, old prompt, face + body refs): holds. Data point:
    a 1-clip sing_away wide at this distance did not need the fix.
- Reading (hypothesis, from the frames above): the mid-chain cut to a closer shot is a property
  of the long chain (3 clips, 35-40 s), not of the face reference. H1 removed WHAT was inserted
  (portrait + candles) but a 3-clip chain still inserts a closer shot of her around 25-31 s,
  now drawn from the body picture and the scene. Every 1-clip and the one 2-clip distant scene
  held. Two candidate rules, both general, neither tested: (a) cap distant chains at 2 clips and
  let the writer split a long no_singing_her span into two scenes with their own opening stills;
  (b) the frame-0 anchor plus a keyframe at each clip boundary (s.20.33 design), which also
  answers the convergence. (b) is the direction the user set; (a) is the cheap fallback.
- Cut: `POST /api/mv/assemble {batch, name: ssbs_first_cut_20261029}` fired 06:30 (full path:
  trim to window, concat, song under, 1.5 s fade). Result recorded below when it lands.
- Cut landed 06:25: `.mvwork/mv_assembly/20261029_062450/ssbs_first_cut_20261029.mp4`, 252.44 s,
  25 scenes, 1344x768 24 fps, song under from 0.00 with the 1.5 s fade; whole-cut envelope r vs
  the song 0.979 at lag 0.00 (the full song-mux path works). Filed copy:
  `renders/ssbs_FIRST_FULL_CUT_25scenes_252s_1344x768_20261029.mp4`.
- Three-frame contact sheets of all 25 scenes (`renders/all_scenes_cmp/`), frames only, the user
  judges motion: suspect artefacts to look at are scene 22 (a large pale hand-like shape across
  the left of her face for the whole clip), scene 1 (a gold ornament on her head and a brown
  object in her hands that are in no prompt), scene 0 and 17 as above. The sing_to_camera scenes
  (2, 5, 7, 10, 11, 14, 16, 18, 20, 23) look like themselves frame to frame; the interiors keep
  their candles; the people scenes (3, 4, 13, 18, 19) hold their extras.
- Next (user's direction, s.20.33): the opening-still frame-0 anchor on the reference lane,
  one-variable on scene 0, then a sing_to_camera check. Nothing fired until the user says so.

### 20.35 User's notes on the first cut (2026-10-29 morning) [U = ground truth; frames confirm]
Frames at the reported times in `renders/user_flags_20261029/user_flags_sheet.png`.
1. 0:01 (scene 0, ref2va, 3 clips, body ref only): a ghost building appears behind the chapel at
   1 s, slides onto it and stays for the whole scene (the "drape" in s.20.34 is this ghost). U.
2. 0:32-0:45 (scene 1 Walking the wall, ref2va, face + body + location refs, sing_away, medium):
   a gold crown-like ornament on her head and a brown object in her left hand. Not in the prompt
   (checked the prompt node in the box history: "no jewellery" twice, "one hand trailing over the
   stones", no prop) and not in either reference picture (bare head, empty hands). U.
3. 2:30 (scene 17 The finest evening, t=4.7, ref2va, 3 clips, body ref only): a second ghost
   chapel appears to the right of the first and slides onto it; the background warps. U.
4. 2:46 (scene 17, t=21): her walk jumps to a run for a moment; after it the background warps
   and a cliff edge comes into view from nowhere (t=25-31, the segment s.20.34 called a cut to
   a closer shot). U.
5. 3:40 (scene 22 And by morning, ref2va, face + body + location refs, sing_to_camera, close): a
   pale claw-like shape hooked over the left side of her face for the whole clip. Not in the
   prompt ("her hands stay still and out of frame", "hair clings to her temple and cheek") and
   not in the references. U.
Reading (hypothesis): 1, 3 and 4 are the reference lane converging from noise over a long chain
(frame-0 has no picture to hold the building or her scale; the chain re-invents the world around
25-31 s). 2 and 5 are the model inventing objects on her at frame 0 of a close shot with nothing
pinning her first frame; both are single-clip close shots that would start from a generated
opening still under the s.20.33 design. All five are the same missing piece: a first frame that
already shows her, the building and her hands, which is the user's direction. Nothing fired.

### 20.36 Frame-0 anchor on the reference lane built (2026-10-29 07:20) [built; dry-run; nothing fired]
- What: `render_scene {anchor_first: true}` (or `anchor_still_id: <library still>`, which implies
  it) anchors the scene's OPENING still at frame 0 of the FIRST clip through MiniMaxH3AddGuide,
  on the reference lane, while the face/body/location pictures stay as references for identity
  and lip-sync. The still is pixel guidance (the node centre-resizes it to the canvas and VAE
  encodes it), not a <Picture n>, so the compiled prompt is byte-identical to the unanchored run:
  one variable. Clips 2+ continue from context frames and are not anchored. Opt-in, default off.
  - chain: `build_h3_chain(..., anchor_name=)`: LoadImage "29", AddGuide at clip-1 node base+8
    (after any keyframe guides, before the clock node and the guider); refused on lane i2v and
    with keyframes still_first / still_first_last (two guides on one frame); `resolved.anchor_first`.
  - submit: `_h3_chain` uploads `anchor_still_id` like `first_frame_id`.
  - render_scene: precedence request `anchor_still_id` > the scene's filed `opening_still_id`
    (from /api/mv/scene_still) > 400 "file it first". dry_run and meta carry `anchor_first`,
    `anchor_still_id`, `anchor_from`.
- Verified offline (package import): a 3-clip graph has AddGuide 108 {positive 100, latent 100,
  frame_idx 0, image 29} feeding guider 102; guiders 112 and 122 take the raw conditioning; both
  guard rails refuse. Backend restarted (box queue empty, no backend jobs).
- Dry run through the route, scene 0, same knobs as the batch (seed 20261027, turbo, 1344x768):
  the gate returns 400 until the scene's opening still is filed; with the cliff_road location
  still passed as a STAND-IN anchor (plumbing only) the dry run shows lane ref2va, distant True,
  refs [body, cliff_road], 3 clips, anchor_first True, anchor_from request, prompts unchanged.
  `dry/scene0_anchor_standin_dry.json`.
- The real test needs scene 0's opening still with her in it: prompt in
  `dry/scene0_opening_still_prompt.txt` (the writer's own `opening_still` for the scene, plus:
  far from camera, face not readable, nothing on her head or in her hands, same camera and fixed
  elements as the location still, THIS scene's pale sky and light rather than the still's storm
  front, no other people). User generates it from the location still + her body picture, 16:9;
  file with POST /api/mv/scene_still {script, scene 0}; then fire
  `render_scene {scene 0, anchor_first true}` = one variable against d15c63bd (same seed, same
  prompt, same refs). Expected if the anchor works: she and the chapel present at frame 0 with
  no fade-in and no ghost building in the first seconds. NOT expected to fix the 25-31 s
  mid-chain wander of a 3-clip chain (clips 2 and 3 are unanchored; open item 7).
- 2026-10-29 07:40 FIRED (user go): scene 0 opening still from the user filed as
  df88beb822f341d68285109e5e22f4e5 (1672x941; chapel left third, her tiny at the far end of the
  road, pale evening sky, nothing on her head or in her hands), recorded on the scene. Dry run
  took it from the scene (anchor_from scene), prompts identical to the overnight compile. Job
  2a25065c submitted; the running graph on the box carries LoadImage 29 = that still and
  AddGuide 108 at frame 0 on clip 1 (checked on :8188/queue). One variable against d15c63bd.
  Handover script `renders/handover_scene0_anchor.sh` (floor check, named copy, 1 s sheet, old
  vs new first 8 s at 0.5 s steps).
- 2026-10-29 10:33 LANDED: job 2a25065c, 42.0 s, 1344x768, 1008 frames, floor r 0.998 at
  lag 0.00. Filed `renders/ssbs_scene0_cliffroad_768p_3clips_bodyref_ANCHOR_openingstill_frame0_seed20261027.mp4`;
  sheets in `renders/scene0_anchor_cmp/` (1 s sheet; old vs new first 8 s at 0.5 s steps).
  Frames (user to watch):
  - ANCHOR TOOK. Frame 0 of the new render is the opening still: chapel on the left, the road,
    her tiny at the far end. No fade-in of her, no morph of the chapel over the first 2 s. The
    old render had neither building nor her settled until t=2.
  - GHOST BUILDING STILL ARRIVES, now as a second house: at 2.5 s a translucent building appears
    to the right of the anchored chapel and is solid by 4 s; it stays beside the chapel through
    clip 1. In the old render the ghost rose BEHIND the chapel at 1 s. Reading (hypothesis): the
    location reference picture <Picture 2> (cliff_road still 7bb11d6c) shows a different chapel
    than the opening still; with frame 0 pinned to one, the model retains the other as a second
    subject and draws both. Same mechanism as the old ghost, now visible as two buildings.
  - CHAIN WANDER WORSE: camera pushes to the chapel by 8 s, she is a medium shot by 13 s (old:
    t=30), the chapel grows a red door by 14 s, a translucent building overlay returns at 30-36 s.
    One variable, but the anchor changes the whole trajectory, so the earlier push cannot be
    separated from seed variance on this one take.
- Next one-variable test (proposed, not fired): keep the anchor and make the location REFERENCE
  the same pixels as the anchor (opening still as <Picture 2> in place of the cliff_road still),
  so the model holds one building, not two. Prompt structure unchanged. Alternative: drop the
  location picture entirely (body picture only + anchor), which also changes the prompt.
- 2026-10-29 10:50 USER VERDICT on the anchored take: "actually worse" than the unanchored one.
  Frame 0 holding the still does not pay for the second building and the earlier push-in.
- USER RULE (general, 2026-10-29): no shots where she walks toward camera from a distance. A face
  that starts tiny and grows as she approaches resolves badly all the way in ("the face being so
  small looks really messed up as it gets closer"). Her scale in a shot is fixed: distant shots
  stay distant for their whole length, close shots start close; a change of scale is a CUT to a
  new scene with its own opening still, never her approach inside one shot. Applies to the
  writer (no "walks toward the camera", no "grows larger in frame", push-ins on her bounded to
  small amplitude) and to the opening-still brief (she is already at the shot's scale at frame 0).

### 20.37 Fixed-scale rule written into the writer; scene 0 split (2026-10-29 11:30) [built; no GPU]
- SCENE_SYSTEM: any take SHE is in runs at most 30 s (measured: 3-clip chains re-invent the shot
  at 25-31 s; 1-2 clip scenes held); environment-only B-roll keeps its 45 s. New bullet HER SCALE
  IS FIXED INSIDE A SCENE: never walks toward camera, never grows in frame, distance at the end
  is the distance at the start (across, away or in place); a change of scale is a CUT to a new
  scene with its own framing; push-ins on her small amplitude only.
- CLIP_SYSTEM: the clip-level form (she never comes closer; wide / extreme wide stays small to
  the last frame; the next scene changes scale, not this clip).
- `split_scene` helper: the cut snaps to a cut point only within 1 s (an intro has none, the
  cut stays where asked); the second half starts without the first half's opening still.
  New route `POST /api/mv/split_scene {script, scene, at, first?, second}`: cuts the scene, runs
  pass 2 on both halves under the current rules, saves `<script>_sp<idx>_<stamp>.json` (original
  untouched), records the split in `rewrites`.
- Scene 0 under the rule (my proposal, creative fields are the user's to change): cut at 18.36 s.
  0a "The road at evening" 0.00-18.36, extreme wide, static camera, she stays tiny at the far
  end and walks ALONG the ridge away from the chapel (opening still df88beb8 kept).
  0b "At the chapel wall" 18.36-36.72, medium wide, truck right small/slow along the chapel
  wall, she stands at the wall corner in three-quarter profile, turns her head to the camera
  and back, no singing; needs its own opening still (prompt in the scene's `opening_still`).
  Request `rewrite/split_scene0_request.json`; writer running.
- 11:30 split landed in 57 s: new script `..._sp0_20261029_113024.json` (26 scenes, original
  untouched). 0 "The road at evening" 0.00-18.36 extreme wide static, 2 clips, 0 problems: she
  stands 3 s then walks ~10 steps right along the ridge away from the chapel, "never toward the
  camera", "the same tiny size until the last frame", stops short of the edge facing the sea;
  opening still df88beb8 kept. 1 "At the chapel wall" 18.36-36.72 medium wide, truck right
  small/slow, 2 clips, 0 problems: stands at the wall corner in three-quarter profile, turns her
  head to the lens over ~6 s, holds, turns back to the sea; "her distance from the camera and her
  size in frame do not change". No approach wording in either half beyond the negations. The
  second half has no opening still yet: prompt in `dry/scene1_chapel_wall_opening_still_prompt.txt`.
- Not fired. To render both halves on the split script: 0 as is (anchor off, the reference lane
  as the batch, or anchor on; one variable each), 1 after its still is filed. The later scenes
  are renumbered (+1) in the split script; the batch manifest's job ids still belong to the
  original numbering.

### 20.38 Split test fired: scene 0 + scene 1 of the split script (2026-10-29 13:14) [fired; results pending]
- User kept the split and generated the chapel wall opening still from
  `dry/scene1_chapel_wall_opening_still_prompt.txt`: filed as still 18adfe3c (1672x941) on scene 1
  "At the chapel wall" of `..._sp0_20261029_113024.json` via `/api/mv/scene_still`
  (`dry/scene1_opening_still_filed.json`). Working script pointer `rewrite/final_script_name.txt`
  now = the split script (pre-split name kept in `final_script_name_before_split.txt`).
- Dry runs `dry/split/scene{0,1}_dry.json`, batch knobs (1344x768, seed 20261027, turbo, kitchen
  attention, reference lane, no keyframes, ANCHOR OFF): scene 0 distant rule -> body + cliff_road
  location still, 2 clips, 362 frames; scene 1 face + body + location still, 2 clips, 362 frames.
  Both clip passes say her distance from the camera and her size in frame do not change.
  With the anchor off neither filed opening still enters the graph: this is the split / fixed-scale
  rewrite as ONE variable against the old 36.7 s three-clip scene 0 (d15c63bd).
- Fired 13:14 (user go): scene 0 job 98b2a984, scene 1 job 82380164 (queued behind it). Graphs
  verified on the box: no MiniMaxH3AddGuide, no node 29, LoadImage 30/31(/32) = the refs above,
  two ReferenceToVideo nodes each. Script `renders/fire_split_scene0_scene1.sh` waits, floor-checks
  (scene 1 at song offset 18.36), files `ssbs_scene0_roadatevening_768p_2clips_..._SPLIT_seed20261027.mp4`
  and `ssbs_scene1_chapelwall_768p_2clips_..._SPLIT_seed20261027.mp4`, 1 s sheets in `renders/split_cmp/`.
- Verdict = the user watching both takes; the frames cannot judge motion.
- 14:48 both takes landed (46-47 min each on the box, 685 frames = 28.5 s rendered per 18.36 s
  window, two clips of 362 frames with 39 context frames, clip 2 from ~13.5 s). Floor checks
  clean: scene 0 r = 0.999, scene 1 r = 0.998 at lag 0. Filed under `renders/` with the SPLIT names
  above; 1 s sheets + `split_cmp/scene1_end_16_to_19s.png`.
- Frame reading (NOT a motion verdict; the user watches):
  Scene 0: static extreme wide, she is tiny at the far end of the road and stays the same size in
  every frame, moving slowly right along the path. The fixed-scale rewrite held on the frames.
  BUT two buildings stand on the left from frame 0 (a chapel plus a larger rounded building beside
  it), held consistently through the take. Both reference pictures (location still 7bb11d6c and the
  filed opening still df88beb8, which did not enter this graph) show ONE chapel: the second building
  is invented by the model, the same pattern as the ghost building in the first cut, now present
  from frame 0 rather than arriving at 1 s.
  Scene 1: she stands fixed at the wall corner at the same size for the whole window, head turn
  to camera visible around 7-12 s, camera trucks right (the wall slides). A second chapel appears
  in the background right from ~7 s and stays. Then a DISSOLVE to a close-up of her (hand at her
  chin) begins at ~18.1 s and is clearly visible by 18.25 s, against the prompt's "one continuous
  shot with no cut and no dissolve": the chain re-invents the shot ~4.6 s into clip 2, inside the
  30 s cap. The scene window ends at 18.36 s, so the last ~0.3 s of the used window is
  contaminated; trimming the window to 18.0 s would clear it (assembly detail, user's call).
- Open: the invented second building is now the recurring defect on the reference lane for this
  location (hypothesis, unverified: the retention text asks to retain the layout of a picture
  whose chapel sits mid-left, and the model fills the left third with more masonry). Candidate
  test, not fired: the opening still as the location reference picture too.

### 20.39 User's notes on the split takes (2026-10-29 afternoon) [U = ground truth; frames located]
- U: scene 0 "actually has 2 of the woman at one point". Located: a second woman fades in beside
  her at ~23.5 s (ghost at 23.5 s, solid from 24 s) and both walk on to the end of the 28.5 s file
  (`split_cmp/scene0_tail_18_to_28s.png`). That is past the 18.36 s window the cut uses, ~10 s into
  clip 2, but it is the chain re-inventing the shot again, and earlier than the 25-31 s band.
- CORRECTION to 20.38's frame reading: she does NOT stay the same size in scene 0. At 0 s she is
  tiny by the chapel; by 18 s her face is readable (roughly double her height in frame, frame
  reading, `split_cmp/scene0_crop_chapel_0_to_18s.png`). The clip prose said "walks slowly to the
  right along the ridge, never toward the camera, the same tiny size until the last frame", but the
  road in the location still runs FROM the far chapel TOWARD the camera foreground, so "along the
  road to the right" is a walk toward the lens and the model followed the geometry, not the
  negation. Lesson for the writer (not yet written in): on a path that leads to the camera the only
  fixed-scale walk is AWAY (deeper into the frame, toward the far end) or across on the near
  ground; name the direction relative to the camera, never relative to the frame edge.
- U: scene 1 "isn't too bad, the fading could be considered artistic, but it's not ideal" (the
  dissolve to a close-up at 18.1 s, 20.38).
- U: "when there is a close up shot of her, and there is singing, she should be lipsyncing."
  On paper the split script already does this: every medium close-up and close-up scene is
  sing_to_camera (scenes 3, 6, 8, 11, 12, 15, 17, 19, 21, 23, 24) and every no_singing_her scene is
  extreme wide / medium wide / wide. So a silent close-up during a sung line in the cut came from
  the render, not the script: either a sing_to_camera take where the model kept her mouth shut, or
  a close-up the chain invented inside a wider non-singing or sing_away scene (as the 18.1 s
  dissolve here). Which take it was is still to be identified (user to say where in the cut).
  Proposed hard rule for SCENE_SYSTEM, not yet written in: a medium close-up or closer while a
  vocal line is audible is ALWAYS sing_to_camera; no_singing_her and sing_away are for medium and
  wider; CLIP_SYSTEM already requires the mouth readable and the exact words in <d>.

### 20.40 Three rules written into the writer (2026-10-29 16:00) [built; offline test; no GPU]
- U: at 20 s of the scene 1 file (song 38.4 s, first verse) her face is close enough to sing and
  she does not; and the writer should put her in a lip-sync shot BEFORE the first sung line.
  That 20 s frame is the chain's invented close-up past the 18.36 s window (20.38); the rule is
  general and now mechanical.
- mv_writer: SCENE_SYSTEM gains the close-up-under-words rule, the first-line setup rule (2 s
  lead, new FIRST-LINE SETUP cut point in the brief) and the camera-relative direction rule;
  CLIP_SYSTEM gains the direction rule; `FIRST_LINE_LEAD = 2.0`; `cut_points` adds the setup
  point; `first_sung_line`, `is_close`, `check_scene` (scene_problems via `_finish_scene`, printed
  as SCENE PROBLEMS in the markdown). Offline test on the SSBS sections: setup point 35.22 listed;
  scene 2 "Walking the wall" (medium, sing_away, starts 36.72) flagged on both rules; a silent
  close-up under words flagged; a compliant 35.22-44.76 MCU sing_to_camera scene clean.
- Not run: a rewrite of the split script under the new rules (scene 1 would end at 35.22 and
  scene 2 become an MCU sing_to_camera setup). Backend restarted so the routes carry the code.

### 20.41 Split script rewritten under the new rules: first-line setup (2026-10-29 15:47) [built; no GPU]
- New helper `move_cut` + `POST /api/mv/move_cut {script, scene, to, first?, second?}`: moves the
  boundary between a scene and the next to a cut point (within 1 s, the first-line setup point
  counts), overrides fields on either side, drops a filed still from a scene whose framing or
  performance changes, pass 2 on both, saves `<script>_mc<idx>_<stamp>.json`. Offline: boundary
  1|2 -> 35.22, scene 3 untouched, 30.0 s rejected as off-cut.
- Run (user go, 42 s): request `rewrite/move_cut_scene1_request.json`. Scene 2 fields are MY
  proposal (creative, user's to change): title "The voice at evening", medium close-up,
  sing_to_camera, static camera, keep the drystone wall and her shoulders, one hand on the stones,
  silent for the two seconds before the first word, then the verse to the lens; opening-still
  prompt in the scene (`dry/scene2_voice_at_evening_opening_still_prompt.txt`, attach cliff_wall
  still 88d6056e + face + body).
- Result `..._sp0_20261029_113024_mc1_20261029_154720.json` (working script pointer moved;
  previous name in `rewrite/final_script_name_before_move_cut.txt`), 26 scenes, scene problems
  across the whole script: none. Scene 1 "At the chapel wall" 18.36-35.22 (16.86 s, 2 clips, still
  18adfe3c kept): same shot, head turn to the lens, no singing; the earlier dissolve at 18.1 s
  would now fall past the window. Scene 2 35.22-44.76 (9.54 s, ONE clip of 10.12 s): "For the first
  two seconds she is silent and watchful, lips closed ... Then sings <d>There's a voice that walks
  at evening</d> ...", static, "her size in frame is the same at the end as at the start".
- Note for the user: scenes 2 and 3 are now back-to-back medium close-ups, sing_to_camera, same
  location (static, then push-in); a jump-cut risk. A change of angle on scene 3 is a one-field
  rewrite if wanted.
- Not fired. Scene 2 needs its opening still before its first-frame tests; the reference lane
  render needs nothing new.

### 20.42 First-line setup fired: scene 2 of the working script (2026-10-29 16:38) [fired; result pending]
- Scene 2 opening still (user, from the chat prompt: hand on the wall, face to the lens, chapel
  gable behind) filed as still c4bae7c1 on scene 2 of `..._mc1_20261029_154720.json`
  (`dry/scene2_opening_still_filed.json`). Dry run `dry/split/scene2_dry.json`: reference lane,
  face + body + cliff_wall still 88d6056e, ONE clip of 243 frames (10.12 s for a 9.54 s window),
  static, anchor OFF (the filed still is not in the graph), 0 problems.
- Fired 16:38 (user go): job 4ac3db3d, graph verified on the box (no AddGuide, no node 29,
  LoadImage 30/31/32 = the three refs, one ReferenceToVideo). `renders/fire_scene2_setup.sh`
  waits, floor-checks at song offset 35.22, files
  `ssbs_scene2_voiceatevening_768p_1clip_mcu_static_singtocamera_FIRSTLINE_SETUP_seed20261027.mp4`
  and a half-second sheet in `renders/split_cmp/`.
- What this tests: the first-line setup rule as ONE variable (silent two seconds, then the first
  verse lines to the lens, single clip, no chain hand-off). Verdict = the user watching: does she
  hold the lens silently before the first word, and does the mouth read on the words.
- 16:52 landed (13.7 min on the box, 243 frames = 10.125 s). Floor check r = 0.998 at lag 0. Filed
  `ssbs_scene2_voiceatevening_768p_1clip_mcu_static_singtocamera_FIRSTLINE_SETUP_seed20261027.mp4`;
  sheets `split_cmp/scene2_4ac3db3d_sheet_halfsec.png`, `split_cmp/scene2_mouth_0_to_3s.png`.
- Frame reading (user watches for the verdict): static medium close-up holds for the whole clip,
  same size, hand on the wall, chapel gable soft behind, no second building, no re-invention in
  10 s. BUT her mouth is open and moving from frame 0: open shapes at 0.00, 0.25, 0.50, 0.75 s
  and on, through the two seconds the prompt wrote as "silent and watchful, lips closed and
  relaxed". The silent lead-in did not hold on the picture. Hypothesis, unverified: in a
  sing_to_camera clip the model starts mouth motion at frame 0 regardless of where the vocal
  enters (the batch's sing_to_camera scenes all began on a line, so this was never visible);
  alternative: something vocal-like in the track at 35.22-37.22 (check the vocal stem's energy
  there before blaming the model).
- 2026-10-29 user verdict: the take is fine. The singing is mid-word when the clip starts, so
  the mouth moving from frame 0 is CORRECT. The "two silent seconds" never existed in the audio:
  the FIRST-LINE SETUP point was placed from the lyric timing in the script (first word at 37.22),
  which I never checked against the track; the vocal is already going at 35.22. Non-problem,
  hypothesis withdrawn, no rule. User, same message: we have spent the day making rules around
  small things and generating new stills when considerable chunks of last night's render were
  OK and only specific things were called out. Lesson filed as memory
  (feedback_fix-callouts-not-legislate): fix the called-out shots against the mostly-good render;
  verify a claimed problem against the audio or the user before proposing any rule.

### 20.43 Scene 0 clip text refreshed and re-fired (2026-10-29 18:35) [result: doubled again, inside the window]

- Before firing scene 0 "as-is" (user go), the dry run showed the clip prompts still carried the
  frame-relative wording "walk to the right along the ridge" (the move_cut rewrite re-ran pass 2 on
  scenes 1 and 2 only; scene 0's clips were the split-fire text). Only the scene summary had the
  new direction. Firing that text with the same seed would most likely reproduce the doubled take,
  so I stopped and said so.
- User: refresh scene 0's clip text, show the wording, fire. POST /api/mv/rewrite_scene {scene 0,
  set {}} = pass 2 on scene 0 only, no field changes, still df88beb8 kept, 0 problems ->
  `..._mc1_20261029_154720_rw0_20261029_183530.json` (now final_script_name.txt). New wording:
  "walks slowly across the frame on the far ground, away from the chapel and toward the cliff
  edge ... keeps the same distance from the camera and stays the same tiny size to the last frame.
  She never turns toward the camera and never comes closer." No "to the right" anywhere.
- Dry run identical to the split fire: ref2va, body ref cd98a0e4 + cliff_road still 7bb11d6c
  (distant => face ref dropped), 362/39, 2 clips, anchor OFF, seed 20261027.
- Fired 18:36: job d7d1d6ae, running on the box, queue otherwise empty.
  `renders/fire_scene0_rw0.sh` waits, floor-checks at offset 0, files
  `ssbs_scene0_roadatevening_768p_2clips_extremewide_static_bodyref_CAMERA_RELATIVE_WALK_seed20261027.mp4`
  and `split_cmp/scene0_rw0_d7d1d6ae_sheet.png`.
- ONE variable vs the SPLIT scene 0 take: the clip direction wording. Verdict = the user watching:
  one woman, same size throughout, in the 0-18.36 s window.
- 19:23 landed (46.7 min on the box, 685 frames = 28.54 s, window 18.36 s). Floor check r = 0.999
  at lag 0. Filed `..._CAMERA_RELATIVE_WALK_seed20261027.mp4`; sheets
  `split_cmp/scene0_rw0_d7d1d6ae_sheet.png`, `split_cmp/scene0_rw0_figure_crop_0_to_18s.png`.
- Frame reading (user watches for the verdict): static extreme wide holds, she walks away from the
  chapel along the ridge toward the cliff edge at a fixed size as written. BUT a SECOND woman in the
  same wardrobe fades in beside her at about 9 s and walks with her to the end of the window, i.e.
  INSIDE 0-18.36 s this time (the SPLIT take doubled her at 23.5 s, in the over-render tail). The
  direction wording was not the lever for the doubling: one seed, one variable, same result. Not
  a rule. Untested candidates, all hypotheses: seed variance (one other seed), the I2V lane with the
  filed opening still as frame 0 (the env B-roll lane), a shorter scene. User's call.

### 20.44 Scene 0 split into two closer shots (2026-10-29 19:29) [fired; scene 1 clean, scene 0 walks at the lens]

- User on the CAMERA_RELATIVE_WALK take: apart from the doubling, she is too far from the camera
  for too long. Wants her a bit closer for the first part, then another shot, closer again, from a
  different angle.
- POST /api/mv/split_scene {scene 0, at 9.0 (inside the intro, no cut point needed), first: wide
  static, full figure readable by the chapel then the walk away toward the cliff edge; second:
  "On the ridge", medium wide static, low on the road looking up the ridge, sea and sky behind her,
  three-quarter profile, slow head turn along the coast} -> `..._rw0_20261029_183530_sp0_20261029_192903.json`
  (now final_script_name.txt). 27 scenes, 31 clips. Scene 0 = 0-9.0 wide, scene 1 = 9.0-18.36
  medium wide, both ONE clip (no chain hand-off), 0 scene problems, 0 clip problems. Scenes
  renumbered: chapel wall = 2, first-line setup = 3.
- Scene 0 keeps the old extreme-wide still id df88beb8 on the record (the split route keeps the
  first half's still; the render path with the anchor OFF does not use it). Both halves have new
  opening_still prompts; given to the user in chat. Scene 1 has no still until one is filed.
- Note for the user: the writer's wide clip says she "stands small" with her face unreadable; if
  "a bit closer" means her face should read, the first half wants medium wide instead.
- 19:31 fired both on the user's go: scene 0 job a1c19da6 (body ref + road still 7bb11d6c, 226
  frames), scene 1 job 4f085048 (face + body + road still, 226 frames); graphs verified on the box
  (no AddGuide, no node 29). Landed 19:43 (686 s) and 19:55 (731 s). Floor checks r = 1.000 (offset
  0) and r = 0.997 (offset 9.0). Filed `..._wide_static_bodyref_SPLIT9_seed20261027.mp4` and
  `..._mediumwide_static_lowangle_facebodyref_SPLIT9_seed20261027.mp4`; half-second sheets in
  `split_cmp/sp0b_*`.
- Frame reading (user watches for the verdict). Scene 0: ONE woman, chapel held, static wide, but
  she walks DOWN the road toward the camera and roughly doubles in size over the 9 s: on this
  geometry the road runs from the chapel at the far end toward the lens, so "away from the chapel
  toward the cliff edge" IS the approach the no-approach rule forbids. The writer named her path by
  the landmark, not the camera. Sky rendered as a dark storm bank, sea rough. Scene 1: clean; she
  holds the ridge edge in three-quarter profile at one size, face readable, hair and shawl in the
  wind, the chapel small on the left; pale calm evening sky, warm light on the gorse. The cut
  between them therefore jumps from storm sky and rough sea to pale sky and calm sea: a look
  mismatch to judge by eye.
- User, after watching: her OUTFIT differs between the two SPLIT9 takes as well, "I feel we're
  going backwards". User is handing the build to a different model; handover written to
  docs/MV_H3_HANDOVER_20261029.md. Nothing more fires.

### 20.45 Scene 0 (0-9 s) redone as a standing shot on the image-to-video lane (2026-10-29 20:05) [built; dry-run; nothing fired]
- Handover read (docs/MV_H3_HANDOVER_20261029.md). User's picks: KEEP the SPLIT9 ridge take
  (scene 1, 9.0-18.36) as is; redo the 0-9 s wide as a STANDING shot (no walk) with an opening
  still as frame 0 on the image-to-video lane, so the sky, the outfit and one woman come from the
  still's pixels.
- Facts checked, not causes: the cliff_road location still 7bb11d6c is itself a dark storm bank
  over a rough sea, and the SPLIT9 wide (body ref + that still) reproduced it although its text said
  pale sky and calm sea; the ridge take used the same still and rendered pale, so the still alone
  does not decide the sky [H]. The body picture cd98a0e4 shows the shift off BOTH shoulders (puffed
  sleeves at the upper arms) while the script's wardrobe and refs text say "slipped off her left
  shoulder": text and picture disagree on a visible detail (s.4 rule) [H as a cause of the outfit
  drift; not changed].
- Writer: POST /api/mv/rewrite_scene {scene 0, set {action: stands still in front of the chapel the
  whole take, no step, one slow head turn along the coast; keep_in_frame: the chapel and her whole
  figure}} -> `..._sp0_20261029_192903_rw0_20261029_200343.json` (now final_script_name.txt;
  previous in final_script_name_before_standing.txt). 1 clip, 226 frames, 0 problems.
- Code (opt-in, defaults unchanged): render_scene accepts an explicit `lane: "i2v"` on a scene she
  is in when the performance is no_singing_her and her opening still is filed (or first_frame_id
  passed); never the person-free location still. compile_prompt_i2v writes "the woman" for
  <Subject 1> (S1), since that lane declares no subjects. Untested for her shots [H: identity and
  wardrobe held by the frame-0 pixels as on the people plates].
- Dry runs `transfer/ssbs/dry/standing/scene0_{i2v,ref2va}_dry.json`: i2v = no refs, 1 clip, 226
  frames, frame 0 = the scene's filed still, which is still the OLD extreme-wide df88beb8 (her tiny
  at the far end); the new still must be filed on scene 0 before firing.
- Next: the user generates the opening still (prompt given in chat), files it on scene 0, dry run
  shows it as first_frame_id, then fire on the user's go. Verdict = the user watching it cut against
  the ridge take.

### 20.46 Re-design from the general recipe, passed takes locked (2026-10-29 21:00) [built; writer run; no GPU]
- User: a whole day on scene 0 and no further forward; start again from the song and its stills
  and design a cool video with what we know, reproducible for any song and style. The design was
  written as the general recipe (TECHNIQUES s.15) and applied here; user: "ok".
- Built: `song_levels` (SSBS: every section loud except the bridge, mid -18 dB with a quiet
  188.5-191.5 s; the intro is loud from 2.5 s), SCENE_SYSTEM palette / motion / pace / section
  roles / arc / frame-0 rules, her scenes capped at one clip, free cuts inside lineless sections,
  `arc` and `matches_location_still` in the schema, locked takes through writer, batch and
  assembly. Offline: lock tiling (a model scene reaching into a lock is cut at it; an empty gap
  becomes a flagged placeholder), levels, still state.
- Locks for SSBS (24): ridge SPLIT9 4f085048 9.0-18.36; chapel wall 82380164 trimmed 18.36-25.0
  (before her head turn, so she is looking out to sea at the cut); first-line setup 4ac3db3d
  35.22-44.76; last night's unflagged scenes 2-16, 18-21, 23 at their windows; "Singing still"
  b4a68d7e trimmed 243.0-252.44. Gaps for the writer: 0-9.0 and 25.0-35.22 (intro), 145.3-184.14
  (solo), 218.5-225.16 (the flagged close-up), 234.52-243.0 (outro). Request
  `transfer/ssbs/redesign/write_locked_request.json`.
- Writer run (Claude, no GPU): `she_sings_before_the_storm_20261029_202712.json`, then pass 2 rerun
  on scene 23 (its clip carried "with each", flagged by check_prompt) ->
  `..._202712_rw23_20261029_202818.json` (now final_script_name.txt; previous in
  final_script_name_before_redesign.txt). 35 scenes: 24 locked, 11 new, 0 scene problems, 0 clip
  problems. Arc: slate evening with a pale band -> dusk and first rain -> black night storm (solo,
  last chorus) -> colourless dawn. Payoffs: the fiddle (lit tavern in the intro, abandoned at dawn
  in the outro), the boats (whole, pitching in the solo, gone on "and by morning they are gone"),
  father's coat on its peg in the storm night.
  New scenes: 0 The front comes in (cliff_edge plate, push to the horizon); 3 The old bar the
  shutters (harbour, one old woman barring a shutter, truck); 4 The fiddle waits (tavern plate,
  push to the fiddle); solo 21 The sea turns (cliff_edge night, fast push), 22 She holds the edge
  (her, medium wide, arms lifting, small arc; reference lane), 23 Boats on their ropes (harbour
  night, truck), 24 The coat stays on its peg (kitchen candle, truck), 25 Rain on her face (her
  close-up, silent in the solo, eyes open to the sea; reference lane), 26 The chapel in the gale
  (cliff_road night, pedestal up); 31 Empty moorings (harbour dawn, on "and by morning they are
  gone", replacing the flagged close-up); 33 The fiddle left behind (tavern dawn).
- Guard added: render_scene refuses an image-to-video scene the writer marked
  matches_location_still false until its own still is filed (would have rendered the location
  still's weather and dissolved). Markdown "Opening stills needed" lists plates only.
- Batch dry run: 5 new scenes ready (0, 4, 24 plates from their location stills; 22 and 25 hers on
  the reference lane), 6 waiting for stills (3, 21, 23, 26, 31, 33), 24 carried as existing.
- Look note for the user's eyes [frames, not a verdict]: the locked ridge and chapel wall takes
  have pale calm skies; the new intro plates around them follow the arc's slate evening.
- Partial assembly of the 24 locked takes through /api/mv/assemble (partial, 179.24 s, missing
  0, 3, 4, 21-26, 31, 33): `.mvwork/mv_assembly/20261029_202923/ssbs_redesign_LOCKED_TAKES_ONLY_partial.mp4`.
  Trimmed locks in sync: chapel wall (in-point 0) r 0.998, "Singing still" (in-point 8.48 s) r 1.000
  at lag 0, and the same segment against the untrimmed offset 0.212 (control: the seek is real).
  First-line take r 0.998. Note: `batch` must be an absolute path or a bare name under mv_batches.

### 20.47 The 11 new scenes rendered; first full cut (2026-10-29)

- Stills filed with /api/mv/scene_still for scenes 3, 21, 23, 26, 31, 33 (copies in
  .mvwork/transfer/ssbs/redesign/stills/, named by scene). Dry run: 11 ready, 24 existing.
- Batch fired on the user's go: manifest `..._rw23_20261029_202818_batch_20261029_214750.json`.
  All 11 done, no errors. Lanes engaged as planned (job params): 9 on i2v, each with its own
  first_frame_id (the filed stills; 0, 4, 24 from their location stills); 22 and 25 on ref2va.
- Mistake on the fire: the call passed only script + ref_ids, so render_scene defaults applied:
  960x544 and a random seed per scene, NOT the 1344x768 / seed 20261027 the earlier dry run had
  been given. The 24 locked takes are 1344x768; assemble scales all to 1344x768, so the 11 new
  ones are upscaled from 544p. Lesson: a render_script fire must repeat the dry run's width,
  height and seed explicitly (the dry run does not pin them for the fire).
- Floor check (envelope r at lag 0, song offset = scene start): 0 1.000, 3 0.998, 4 0.998,
  21 0.997, 22 0.999, 23 0.999, 24 0.997, 25 0.998, 26 0.999, 31 0.997, 33 0.998. All clean.
- Named copies: .mvwork/transfer/ssbs/redesign/takes_544p/.
- Full cut, 35 scenes, 252.44 s, 1344x768, song laid under:
  `.mvwork/mv_assembly/20261029_225322/ssbs_redesign_FULL_new_scenes_544p_upscaled.mp4`.
  Not yet watched by the user; motion and look are theirs to judge.

### 20.48 User callouts on the first full cut; gate before the GPU (2026-10-29)
- User: scene 0 "back to the timelapse effect"; the cut to her singing comes "a couple of seconds"
  after she starts; 2:01 she stands at the back of the tavern not lip-syncing, should be much
  closer; the 0:09-0:19 ridge stand is "too long for nothing happening".
- Scene 0 ran on i2v (box graph: MiniMaxH3ImageToVideo, fl2va weights), but on base 20 steps,
  shift 12, no fl2v turbo LoRA, 960x544, random seed: the fire omitted the settings (s.20.47).
  Its text also makes the cloud bank the one moving element. Locked scene 12 has the same kind of
  text on the proven settings and was passed, so the settings are the stronger candidate [H].
- Gate built (af69a07): a live batch fires only {from_dry, accept_flags}; the dry run shows each
  scene's resolved settings + pinned seed and flags settings off PROVEN (turbo, 1344x768) and pace
  problems (moving or unpinned sky); a changed prompt or flagged scene is refused. Refusal paths
  tested through the route with the box queue empty before and after.
- First line: the vocal stem enters at 32.25 s; Whisper missed "On the cliff road past the
  chapel / Where the gorse grows bent and low", so the verse aligned at 36.72 and the first-line
  setup was cut at 35.22. song_facts now times missed opening lines from the stem (only verse 1
  changed on SSBS); /api/mv/refresh_facts re-measures a saved script (30ad5db).
- Edits (writer passes, no GPU): first-line scene 30.25-44.76 (setup 2 s before the first word;
  passed take dropped, it cannot start before 35.22); old woman 25.0-27.75, fiddle 27.75-30.25;
  2:01 "Unheard by the hearth" rewritten sing_to_camera medium close-up, fishermen behind;
  ridge split at 13.5 (passed take trimmed to 9.0-13.5) + new "Wind on the ridge" medium
  close-up (wind lifts hair and shawl, one slow head turn sea -> chapel). Script:
  `..._facts_20261029_232452_mc4_..._mc3_..._rw17_..._sp1_20261029_232657.json`.
- Gated dry run: 14 to render at turbo 1344x768 seed 20261027, 22 passed takes, 102 s of video;
  flagged: scene 0 (sky moves / unpinned).

### 20.52 End picture for a camera move (2026-11-01) [built; one take]
- Why: on Never Yours every push inside the bar and the hall lost the room mid-move (scene 14
  corridor push, scenes 21/39 zoom into the wide bar). User: make the moves work, do not ban them.
- scene.end_still_id = the view the move must arrive at; i2v lane feeds it as the node's
  last_frame, the reference lane as MiniMaxH3AddGuide at frame -1. POST /api/mv/push_end_still
  crops the scene's first picture toward the aim point (zoom 1.05-4), stretched to the canvas
  aspect like the first frame [V, node source: first_frame stretches, last_frame cover-crops].
- Scene 14 take, seed 273520885, prompt unchanged ("large amplitude at fast speed"), x1.35 at
  (0.495, 0.43): sync 0.997, lag 0, last frame = the end picture. From frames (not motion):
  2-4 s the hall becomes a longer darker corridor with a ceiling lamp, then returns; streaky wall
  texture 5-7 s. [H] the prompt drives harder than the bounded distance, so the middle is free.

### 20.53 The rail: in-between views pinned along the move (2026-11-01) [built; one take, awaiting the user's eyes]
- Source facts [V, ComfyUI comfy_extras/nodes_minimax_h3.py + comfy/ldm/minimax/model.py]:
  MiniMaxH3AddGuide takes any pixel frame; each guide is extra conditioning tokens placed at
  time cursor + 5/3 * frame (the same list first_frame/last_frame use), cover-cropped to the
  canvas. Several guides chain on one conditioning.
- push_end_still {rail: seconds} also files crops every `rail` s (no pin in the last 0.5 s):
  a constant-rate zoom (scale zoom**(f/last)) toward the one fixed point the full picture and
  the end crop share, so every pin is the same geometry and the series lands on the end crop.
  Stored as scene.rail {end_still_id, every, stills[{still_id, frame, zoom}]}; used only with
  the end picture it was made for; single-clip scenes only. build_h3_chain(guide_frames=)
  adds LoadImage 900+k and AddGuide 950+k on clip 1; no rail = graph unchanged (checked offline).
- Tab: "pin the path every 1.5 s" on Where the move ends (default off). The reused-seed hold now
  compares seed AND end picture AND rail, so a same-seed comparison with a new rail is allowed.
- Scene 14 rail (x1.35 at the same point): pins at 36/72/108/144 (x1.06/1.12/1.18/1.25), new
  end picture equals the old one (max pixel diff 6). Tab check: seed 273520885, pinned at
  1.5/3.0/4.5/6.0 s, no warnings. Render waiting on the box (ComfyUI :8188 and :5080 refused
  connections 2026-11-01 10:40-11:00). One variable vs the 20.52 take: the rail.
- Rendered 11:09 (user go; graph verified on :8188: AddGuide 950-953 at 36/72/108/144, last_frame
  = end picture, seed 273520885). Take check: floor_r 0.997, lag 0, opening jump 0.91, no flag.
  Frames vs the 20.52 take at 0-7.9 s: the hallway stays the same room at every sampled time
  (no invented corridor or ceiling lamp at 2-4 s, no wall streaks at 5-7 s), scale grows
  steadily to the end picture. Motion (smoothness, any stepping at the pins) not judged: user
  watches. A/B copies: .mvwork/rail_s14/s14_A_end_picture_only_*.mp4, s14_B_rail_pins_*.mp4.
- Bar, reference lane (2026-11-01, user go): scenes 21 and 39 got opening stills WITH her (GPT,
  1672x941), end x1.30 at the last booth (0.55, 0.43), rail every 1.5 s. Crops cut from the
  still centre-cropped to the canvas (AddGuide cover-crops), and a rail on the reference lane pins
  frame 0 to the opening still (anchor_from "rail"); graph verified on :8188 (AddGuide 108 f0,
  109 f-1, 950-953). Seed 20261027 = the old free-move takes. Checks: 21 floor_r 0.997, 39
  0.995, lag 0, no flags. Frames: the room holds at every sampled time in both (the old takes
  rebuild the bar each second); both people stay in the booth with small head turns. 39's
  scripted reach to the keys in the first 4 s is not visible in any sampled frame [H: the pins
  at 1.5/3.0 s hold the opening pose]. Motion and identity are the user's to judge.
  A/B: .mvwork/rail_bar/s{21,39}_A_free_move_*.mp4 vs s{21,39}_B_rail_*.mp4.
- Scene 39 again with rail_after 4 (pins 108/144 only, same seed; graph verified on :8188):
  floor_r 0.995, lag 0, no flag. Her pose is nearly identical to the every-1.5-s take at every
  sampled time, and neither shows the reach. The old free-move take does reach (0-2 s). So the
  early pins were NOT what held her [measured]. [H] the late pins and the end picture are crops
  of the opening still, so they assert the PRE-action state (hand in lap, no keys on the table:
  the GPT prompt never put the keys in) at the end, which contradicts the scripted end state.
  A rail cut from one still fits scenes whose people end where they started (21: a head turn);
  a state change needs the late pins and the end cut from a still of the END state (an edit of
  the opening still, same framing). Not built. A/B: .mvwork/rail_bar/s39_C_rail_pins_from_4s_*.mp4.

### 20.54 After-the-action picture (2026-11-01) [built; offline-checked; not rendered]
- Why: 20.53's 39 result. User: build it only as something reusable, not for one scene.
- scene.after_still_id: filed through POST /api/mv/scene_still {slot: "after"} (paste/drop on the
  "after the action" block of Where the move ends) or /api/mv/scene_after_still {still_id|null}.
  push_end_still then cuts the END picture and every pin from it (same pre-crop as the opening
  still on the reference lane; shape must match the opening still within 2 %, resized to it);
  with a rail, rail_after ("action done by") is required and no pin goes before it. Frame 0 stays
  the opening still. Without an after picture nothing changes.
- The ChatGPT edit prompt is generic, built in the tab from scene.action: keep camera, framing,
  room, light and every person's place, clothes and identity; change only what has happened by
  the END of the moment, shown finished and held still.
- Offline check (real routes, scene 39): after = the opening still itself -> end picture and
  pins 108/144 pixel-identical to the previous rail (max diff 0); a rail with no done-by time is
  refused with the message. Stand-in cleared afterwards. Tab block renders with the prompt.
- User 2026-11-01: scene 39 is fine as is (the rail take; the reach is not needed in a wide).
  Placement rule for the after picture: pictures are filed in step 3 (Pictures), not on the render
  page, and the slot appears ONLY for scenes likely to need it. Done now: the render-page section
  is hidden unless a scene already has one. Not built: the writer marks scenes whose people end in
  a different state (a schema flag + its edit prompt, like opening_still), and step 3 shows the
  after slot for those scenes only. Backend unchanged.

### 20.55 Meridian re-shoot of a still singing take (2026-11-01) [one test, awaiting the user's eyes]
- Why: booth moves on her singing scenes rebuild the room (arcs 10/24/35 never chosen: the
  window, neon lettering, outside street and sconce change as the camera turns; pushes 34/37/31
  drift less). Still booth takes were all chosen. Route: render still, then re-camera the take.
- Box: Camera H3 node (NyckM/3d-Camera-control-H3-Minimax @36c4218, 36 runtime files via fs/write,
  sizes matched, pycompiled), Meridian INT8 pruned 21.06 GB, meridian_dmd_lora, Kijai taomate
  3-step LoRA, MoGe-2 ViT-L normal fp16, text_cond 124 (.safetensors) + fixed_embed_{73..243}.pt.
  ComfyUI restarted (queue empty); nodes and model dropdowns verified via /object_info.
- Graph = the author's v3 Quality workflow (Workflow/Minimax_bruxosdovfx_CameraMeridian_v3_Quality.json)
  with core LoadVideo+GetVideoComponents+ImageFromBatch(124) for their loader, no extra_reference
  (marked experimental by the node), no compare node: MoGeInference 3/0/4/true/true ->
  BruxosH3Camera (Motion Frame, Depth Warp, source aim, panel geometric, Meridian (H3)) ->
  BruxosMeridianReference(124, text_cond 124) -> UNET Meridian int8 + DMD LoRA 1 + taomate 1 +
  comfy kitchen attention x2 + SigmaShift 3/3, euler, ManualSigmas 1.0/0.857/0.6/0.0, seed
  20261027, VAEDecodeTiled 512/64/252/8, author's int8 VAE. File: .mvwork/booth/meridian_s20_arc_graph.json.
- Test: scene 20's chosen still take (first 124 frames), path azimuth 0 -> +8 deg, distance 1.
  318 s on the 3090, no OOM. Output video only; the take's audio muxed back.
  Frames [M]: the camera turns (window grows at left, lamp slides right), the booth, window frame,
  lamp and booth back stay the same room throughout; grey disocclusion behind her right side is
  filled plausibly. Mouth shapes match the source at 1/3/5 s (3 s a little less open). The neon
  lettering and the street outside the window are redrawn (loops, brighter street lights) though
  the warp shows the original sign [H: the window is a far, low-texture area the model re-renders].
  Face close to the source, small differences. Motion and lips are the user's to judge.
  Files: .mvwork/booth/s20_compare_source_meridian_warp.mp4 (source | Meridian | warp guide, with
  audio), s20_B_meridian_arc8_with_audio.mp4, s20_A_source_still_first5s.mp4.
- Open: scenes are 7-8 s, so 175/243 frames need their .pt converted to .safetensors
  (meridian_convert.py); not wired into the tab.
- Scene 10 (user pick: its arc warps toward the end), two runs:
  1. Still render: rewrite_scene {camera: "static"} -> new script copy
     never_yours_20261031_000950_rw10_20261101_132938.json (working script untouched); rendered with
     the old take's settings (refs face+body, turbo, 1344x768, seed 20261027), job e500adbe:
     floor_r 0.998, lag 0, same room 0-8.5 s.
  2. Meridian on it: text_cond 243 converted from fixed_embed_243.pt on the Mac (weights_only load;
     my conversion of 124 is tensor-equal to the hosted 124 file), all 209 frames at length 243
     (node holds the last frame), arc azimuth 0 -> +10 deg, trimmed to 209 frames, take audio
     muxed. 799 s on the 3090.
  Frames vs the old directly-rendered arc [M]: old take doubles the wall lamp at 4 s, grows a
  "GROCERY" sign and "heo" lettering in the window, shrinks the lamp; Meridian keeps one lamp, the
  same booth back and window frame while the camera turns toward the window (lamp leaves frame
  right). Still redrawn in Meridian: the neon shape (squiggle -> loops) and a brighter street.
  New artifact: white specks in the lower right from about 6 s, where the warp guide has a striped
  edge band and grey holes [H: the model fills the warp's edge streaks as sparkles].
  Files: .mvwork/booth/s10/s10_compare_old_left_meridian_right.mp4 (with audio),
  s10_A_old_take_free_arc.mp4, s10_B_still_render_source.mp4, s10_C_meridian_arc10_from_still.mp4.
- User: scene 10 Meridian lip-sync is out. Measured [M]: face-motion cross-correlation still
  source vs Meridian, 64-frame windows: lag 0 (r 0.86) at 0-2.7 s, then 3/-19/9/5 frames with r
  falling to 0.31 by 5.3-8 s: the lips drift and loosen after ~3 s. Not the warp: depth_warp.py
  frame_schedule maps Motion Frame output i -> source int(i * fps/24) = 1:1, and
  MeridianReference cuts <Video 1> 1:1. [H] Meridian regenerates the face from a 480-class
  <Video 1> with no song in the latent (our chains lip-lock through MiniMaxH3SongMaskedAVContext).
  Next test proposed: the song window in the Meridian latent via SongMaskedAVContext, all else equal.
  Fallback: InfiniteTalk v2v (redrives lips from audio, keeps camera) on the Meridian output.
- Song lock test (user go): same graph + LoadAudio(song window 74.12 s, 11.125 s, mix, as the
  take's lock) -> MiniMaxH3SongMaskedAVContext(clip_start 0, context 39, fps 24) on the Meridian
  latent, seed 20261027. Ran clean (Meridian accepts the song-masked latent). Face-motion lag vs
  the still source, 64-frame windows [M, crude proxy: the box also sees the camera move]:
  no song  0/0.86, 3/0.73, -19/0.76, 9/0.57, 5/0.31 (whole 4 frames, r 0.66)
  song     1/0.84, 1/0.78, -17/0.68, 6/0.35, 0/0.51 (whole 1 frame, r 0.72)
  = smaller overall lag, end window back in time, the 4-6.7 s window weaker. Not a clear fix by
  the numbers; the user's eyes decide. File: .mvwork/booth/s10/s10_D_meridian_arc10_songlock.mp4,
  compare s10_compare_C_left_no_song_D_right_song_locked.mp4.
- User: lips on the song-locked take good enough; white specks worse, neon change not liked.
  Specks [M]: they sit on the booth leather, where the guide is solid, not in its big grey holes;
  the guide's dotted/striped hole patterns (2 px splats at full 1344 shrunk to 832x480, a zebra
  band at the right edge) are the likely source [H]. Two runs queued, one change each, seed 20261027:
  E = song-locked + warp_long_side 832 (the node's tooltip: warp at the size the model receives);
  F = E + extra_reference = source frame 0 (the author's Quality workflow wires one).
  Results [M, crops of the lower right at 4/7 s]: D many bokeh specks; E almost none at 4 s, 2-3
  small at 7 s; F a few more than E. Crude bright-blob count 5-8.6 s (source 148 = leather
  highlights): D 236, E 209, F 212. Neon: the guide keeps the source squiggle at every frame, but
  D, E and F all redraw it as loops; the reference picture (F) did not hold it. So the neon is
  Meridian redrawing, not disocclusion. warp_long_side 832 = keep. Files: s10_E_warp832_songlock.mp4,
  s10_F_warp832_ref_songlock.mp4, s10_compare_D_left_E_right_warp832.mp4.
- Neon paste-back (user pick, no GPU; .mvwork/booth/s10/paste_sign.py): composite the guide's
  window glow onto E with a feathered union mask. Result [M]: the original squiggle returns, but
  from ~5 s a doubled window post shows and some mask edges read as hard lines. Cause [M]: the
  output's window does not follow the guide's geometry; edge correlation on the window area,
  output vs best-shifted guide, falls 0.46 (0 s) -> 0.36 (2 s) -> 0.25 (4 s) -> 0.17 (7 s) ->
  0.16 (8 s), with the best rigid shift drifting dx +2 -> -8 -> +2, dy +6 -> +12: Meridian draws
  a larger, re-shaped window as the camera turns, so a rigid paste cannot sit in it. Not usable;
  kept as s10_G_sign_pasted_REJECTED_doubled_post.mp4.

### 20.56 Meridian re-shoot in the app (2026-11-01) [built; tab checked; end-to-end fire not yet run]
- User: keep scene 10's version E; Meridian must be opt-in, only for scenes whose move failed
  (the previous video's moves were fine), never on by default: it is slow.
- How it engages (Scenes and takes):
  1. A scene whose direct move rebuilt the room: tick "still version, for a Meridian re-shoot"
     in the render card. The check first runs POST /api/mv/still_variant (the writer's clip pass
     for a copy of the scene with camera "static", no GPU, a few minutes), stored on the SAME
     script as scene.still_clips (+ still_of_camera); render_scene {variant: "still"} renders it
     (no end picture or rail). The take is labelled "still version".
  2. On any take of the scene, "re-shoot the move": POST /api/mv/meridian_take {script, scene,
     job_id, move?, amount?, seed?, dry_run?}. The move is guessed from the scene's camera line
     (mv_meridian.move_from_camera; flagged when the side is assumed) and can be changed:
     arc left/right, push in, pull out, truck left/right, pedestal up/down; small or large. The
     dry run shows frames, the Meridian length and minutes (3.3 s a frame measured); a take that
     already moves is flagged (Meridian keeps its warping). The fire uploads the take and the
     song window (mix, scene start, length/24 + 1 s), builds mv_meridian.build_meridian_graph
     (= version E: song lock + warp 832, author's sampler) and submits a normal library job;
     on download, mv_finish.trim_with_audio cuts it to the source's frames and lays the source
     take's audio under it. It lands as a take of the scene labelled "Meridian: <move>, <amount>".
- Box: all 8 text_cond lengths (73..243) converted (weights_only load, the 124 conversion
  tensor-equal to the hosted file) and installed. Scene length >243 frames (10.1 s) is refused.
- Filed: scene 10's still render (e500adbe, params script -> the working script, variant
  still, rendered_under kept) and version E as a Meridian take (b9aab3f9, 209 frames via
  trim_with_audio). Tab: both listed with their labels; the Meridian take measures sync 1.00 (the
  source take's audio: this score says nothing about the lips), opening 0.8x; the re-shoot dry
  run on the still take reads arc toward frame left / small / 209 at 243 / ~13 min, with the
  assumed-side flag.
- Mistake fixed: the hand-made rewrite copy (..._rw10_20261101_132938) had become the newest
  script, so the tab opened it instead of the working one (only scene 10's clips + camera
  differed; no user choice was in it). Moved to .mvwork/booth/s10/. still_variant stores on the
  working script, so this cannot recur.
- Signs [M]: azimuth + = toward frame left. lateral / height signs unmeasured [H].
- Not yet run through the app: still_variant (writer) + a variant still render, and a Meridian
  fire's download/trim/audio as a job.
- End-to-end through the tab, scene 24 (user go; the failed booth arc "toward the sconce side"):
  1. "still version" ticked -> check ran still_variant (writer, stored scene.still_clips on the
     working script; text "static, locked off"; scene.camera untouched) -> render variant still,
     seed 1175338985, job 80c19e62: floor_r 0.996, room the same 0-7.8 s.
  2. "re-shoot the move" on that take: guess arc_left with the assumed-side flag (camera line names
     no side); set arc_right in the panel (the sconce is frame right). Graph on the box verified:
     azimuth -10, warp_long_side 832, length 243, text_cond 243, SongMaskedAVContext, one SaveVideo.
     Job 89feb197 came back done as a take: 192 frames (trimmed from 243), aac audio, labelled
     Meridian; take check floor_r 0.996 (source audio), opening 0.91.
  Frames [M]: old direct arc rebuilds the room (a blue bottle appears, a car and a lit sign in the
  window, the panelling changes); Meridian keeps the booth, lamp and panelling while the camera
  swings toward the lamp and the window leaves frame left. New: a rippled, metallic-looking strip
  at the far right edge from ~6 s, where the arc reveals area beyond the source frame [H].
  Lip-timing proxy unusable on this take (her hand-to-hair moves dominate the face box: lags
  jump -20..+20, r 0.24-0.56); the user's eyes decide. Files: .mvwork/booth/s24/
  s24_compare_old_left_meridian_right.mp4 (+ A old, B still, C Meridian).

### 20.57 Scene 24 lips off again; depth parallax and 2D move tested (2026-11-01)
- User: scene 24's Meridian take (song-locked) has the lip-sync off again. So the song lock is
  not a reliable fix: Meridian redraws the mouth every frame from a 480-class copy of the source.
- Test (user go): the Camera H3 warp WITHOUT Meridian, at full resolution (warp_format
  "CrossView (LTX)": source size, any length, magenta + mask for gaps), same arc (az -10, 192
  frames, still take 80c19e62); 80 s on the box. Gaps filled with cv2.inpaint (Telea) and a fixed
  centre crop sized to hide the frame-edge gaps (.mvwork/booth/s24/parallax_fill.py).
  Result [M]: edge gaps up to L 75 / R 172 / T 19 / B 48 px -> 1.355x crop; filled share of the kept
  frame mean 1.3 %, max 4.0 %. Frames: a dark crack across her cheek and forehead by the end (depth
  tear at the hair/face edge), striped disocclusion beside her hair smeared by the fill. Lips exact
  (source pixels). Not usable at this move size.
- Comparison made with no GPU: a 2D move on the still take (scale 1.08, eased pan toward the lamp
  over the clip, ffmpeg). No parallax; nothing redrawn, lips and room exact.
  Files: s24_D_depth_parallax_arc_right.mp4, s24_E_2d_pan_toward_lamp.mp4,
  s24_compare_meridian_parallax_2dpan.mp4 (Meridian | depth parallax | 2D pan, with audio).
- User: depth parallax is useless (breaks as much as the original, differently); the 2D move will
  have to do. Asked why the lips held on scene 10 but not 24.
- Mouth-opening track [M] (OpenCV Haar face per frame, dark share of the lower-face patch,
  z-scored; source still take vs Meridian take; face found in every frame of all three):
  scene 10 song-locked:  whole lag 1 frame r 0.86; 2 s windows 1/0.94, 2/0.96, 1/0.91, -8/0.25
  scene 10 no song:      whole lag 4 r 0.51; 0/0.81, 4/0.84, 6/0.60, 8/-0.51 (drifts later)
  scene 24 song-locked:  whole lag -3 r 0.17; 8/0.22, 8/0.44, 3/0.61, -2/0.76
  = scene 24's mouth is about 8 frames (1/3 s) late and barely follows the source in its first
  4 s, the stretch where she laughs, belts and puts a hand to her hair; it recovers later.
  [H] Meridian redraws the mouth from a 480-class copy plus the song; a steady frontal face
  (scene 10) survives that, big fast performance (scene 24) does not. Not reliable for singing.
  opencv-python-headless pinned <5 in the venv (5.0 dropped CascadeClassifier).

### 20.58 2D move on a take, in the app (2026-11-01) [built; run through the tab on scene 24]
- User: "The 2d will have to do" for singing scenes whose direct move rebuilt the room.
- backend/mv_move2d.py: a moving crop of the take's own frames (cv2.warpAffine LANCZOS, cosine
  ease over the whole take, source audio copied, same size/frames/fps). Moves: push in, pull out,
  pan left/right, arc left/right (pan + push from 1+a/2 to 1+a), tilt up/down; amount = margin
  scaled in (small 8 %, large 16 %). Offline: no frame of any move/amount at 41 steps shows an
  edge. Guess from the camera line (trucks read as pans, pedestals as tilts; side assumed and
  flagged when the line names none). An 8 s take renders in ~4.5 s on the Mac.
- POST /api/mv/move2d_take {script, scene, job_id, move?, amount?, dry_run?} (shared take check
  _mv_take_of_scene with meridian_take) files the result as a done take (engine move2d,
  move2d_of = the source). Tab: "2D move" on each take -> panel (move, amount, make it); label
  "2D move: <move>, <amount>". The still-version toggle now reads "to add the move afterwards".
- Fixed: meridian_take's reply put the SOURCE id under job_id (overwrote the new job's id); both
  routes now return it as source_job_id.
- Through the tab, scene 24: panel guessed arc toward frame left (flagged), set to frame right;
  take 5950eac5 in seconds: 192 frames, audio, sync 0.996 lag 0, opening 0.55.
  Copy: .mvwork/booth/s24/s24_F_2d_arc_right_from_app.mp4.

### 20.59 Research: other ways to stop the warping (2026-11-01) [report in; nothing built]
- User asked for other options. Background research (subagent, read-only) ranked:
  1. Masked-latent room plate: the moving EMPTY-booth plate in the video latent, denoise mask 0 on
     the room, 1 on a dilated envelope of her, a 0.4-0.6 seam band; ref2va references and song
     lock as now. Room identical by construction where the mask is 0; she is generated as in the
     still takes (lips via the song latent). Risk [H]: her perspective may not match the plate
     move (she slides), visible seam. Needs a ~40-line node (H3 AV latent split/merge + mask).
  2. Composite then heal: matte her from the still singing take, composite over the moving plate
     (her layer moved more than the plate = real parallax), masked H3 pass with 0 on her and the
     room, ~0.3-0.5 only on the edge band / disocclusions / shadow. Lips exact (VAE round trip,
     or paste back). Good for pushes, trucks, arcs <= ~10 deg.
  3. Masked rail: the rail crops as background-only latent keyframes (mask 0 on the room only).
  4. MiniMax H3 Fun ControlNet Union 2.0 (core node, ref2va template, control off on audio):
     MLSD/depth of the warped plate; ~4.5 GB int8 download, VRAM unknown.
  5. Wan-family camera tools (ReCamMaster, Uni3C, Fun Camera...): no song lip-sync; second tier.
  6. Prompts / camera LoRAs / H3 camera reference video: already weak here.
- Verified by me [V]: comfy/ldm/minimax/model.py mask_row_values + the "masked rows run at their own
  strength: mask value m puts a row at sigma = m * sigma_stream ... clamped at the cond timestep for
  fully preserved rows" block (local copy .mvwork/comfy_src/minimax_model.py 77-85, 683-697):
  per-2x2-patch-row (32 px, 4-frame) denoise mask with its own timestep, amax-pooled.
  MiniMaxH3AddGuide has no mask input.

### 20.60 Room pinned to a moving plate, H3 draws her (option 1) - scene 24 (2026-11-01) [one take, user to watch]
- Plate: user's ChatGPT edit of scene 24 still frame 0 with her removed (1659x948, 1.75; 50/50 overlay
  with the frame: window, lamp, panelling, booth edges line up; rain and neon redrawn). Moved by the
  Camera H3 warp in CrossView format (full res, Freeze Frame, az 0 -> -10, 192 frames): pivot auto
  = 1.28 m (metric centre), average gaps 7.8 %. The still take moved along the same path with
  warp_pivot_depth 1.28.
- Her area: a difference against the ChatGPT plate caught redrawn rain/neon (52-56 % of the frame)
  -> replaced by SAM3 (core SAM3_VideoTrack, sam3.1_multiplex_fp16, text "woman", max 1 object) on the
  moved still take; held +-3 frames (H3's 4-frame latent steps), dilated ~30 px, + plate gaps,
  feathered outward. Generated share: mean 38 %, 24-48 %. Scripts: .mvwork/booth/s24/plate/
  (make_mask.py, sam3_graph.json, build_option1.py).
- New box node tools/box/comfyui_musicgen_h3plate (MusicGenH3RoomPlateLatent: VAE-encodes the plate
  into the VIDEO stream of the H3 AV latent, audio stream and noise mask kept), installed via
  fs/write, pycompiled, ComfyUI restarted (queue empty), visible in /object_info. Mask through the
  MultiRef pack's MiniMaxH3SetAVNoiseMask (keeps the song lock's audio mask).
- Graph = the app's own scene 24 chain (render_scene dry run: scene prompt with the arc, refs face/
  body/location, seed 1175338985, turbo, 1344x768, song lock) + plate node + mask node before the
  sampler. Accepted with no node errors; ran ~10 min; 192 frames + audio.
- Frames [M]: the camera visibly arcs toward the lamp (window narrows at left, lamp moves in) and
  the room is the plate throughout (no bottle, car or new sign, unlike the old direct arc); she sits
  in the booth and performs (hand to hair at 4 s); no visible seam around her at 3 and 6 s at full
  res. The table front at the bottom right goes soft late in the clip (the arc's gap area, generated).
  Lips: H3's own song-locked lip-sync, as in every normal take; nothing to measure against.
  Files: .mvwork/booth/s24/s24_G_room_pinned_her_drawn.mp4,
  s24_compare_2dmove_left_roomplate_right.mp4.
- User: "Looks good". Filed as a take of scene 24 (b680a3c0, engine roomplate, room_plate_of the
  still take): sync 0.996, lag 0, opening 0.66. Next: build into the tab (proposed, not started).

### 20.61 One "Add a camera move" panel: 2D move, room pinned, Meridian (2026-11-01) [built; room pinned run through the tab]
- User: build the room-pinned move into the tab, keep Meridian as an option, make the flow clear.
- Flow (Scenes and takes):
  0. A scene whose move broke the room: tick "still version, to add the move afterwards" in the
     render card, render (still text written first, no GPU).
  1. On the still take: "add a camera move" opens one panel: the move (arc left/right, push in, pull
     out, sideways left/right, rise, lower; small or large; the scene's camera line shown, an assumed
     side kept as a warning until changed) and HOW, three radio cards with what each does and costs:
     2D move (seconds, no GPU; lips and room exact; no depth), Room pinned (best for singing; real
     depth; the room is the empty-room picture held still, H3 draws the people with the song; ~12 min;
     needs one picture), Meridian (real depth, no picture; lips can lag on lively singing, windows and
     signs repainted; ~13 min). A take that already moves gets an amber note pointing to step 0.
  2. Room pinned shows its picture steps in place: download this take's first frame
     (GET /api/mv/take_frame/{id}), copy the generic ChatGPT prompt (mv_roomplate.EMPTY_ROOM_PROMPT,
     shown in full), paste or drop the result (scene_still slot "empty" -> scene.empty_room_id +
     empty_room_of = the take; a picture from another take is flagged).
  3. One button; the room-pinned job shows its stages live; every result is a labelled new take.
- Backend: backend/mv_roomplate.py (room move graph: Freeze Frame CrossView, orbit centre read from the
  camera node's info; people pass: the take moved around the same centre + SAM3 "person", max 4; plate
  and mask on the Mac), POST /api/mv/roomplate_take (dry run; live = background thread: upload, room
  move, people pass, mask, then render_scene with room_plate=[plate, mask] -> a normal take with
  engine roomplate), GET /api/mv/roomplate_job/{id}. render_scene: room_plate switches off the end
  picture and rail (whole-frame pins). mv_h3.build_h3_chain(room_plate=): the plate and mask nodes
  (offline: byte-identical to the hand-run scene 24 graph the user passed; without it unchanged).
- Take labels: still version / 2D move / room pinned / Meridian, each with the move and size.
- Run through the tab (scene 24 still take, push in, small): panel -> "render with the room pinned"
  (confirm) -> stages shown live -> take 2acf95de, engine roomplate, pivot 1.28, generated share
  mean 0.338, no end picture/rail, seed 683798812 (random); 192 frames + audio; sync 0.996 lag 0,
  opening 0.67. Frames [M]: a steady push in, the window, lamp, panelling and booth unchanged 0-7.8 s,
  she performs (hand to neck/hair). Copy: .mvwork/booth/s24/s24_H_room_pinned_push_in_from_app.mp4.
- User: the process needs to be explained more clearly (scene 33: opened the panel on the original
  take, the greyed button gave no reason; the still version was started via the render card's
  checkbox). Rebuilt as ONE guided card on every scene whose camera moves, above the render card:
  "Camera move broke the room?" with three numbered steps and their live state:
  1 Render a still version (one button: still text if needed, check, confirm with any warnings, send;
    then "rendering on the box" / "done"), 2 Add the camera move to the still version (the three ways
    in one line each; the button opens the panel on the still take and scrolls to it), 3 Watch the new
    take and choose it. The still-version checkbox is gone from the render card (one place to start);
    the panel's notes point back to the card. Checked in the tab on scene 33 mid-render.
- User: pressing "render with the room pinned" showed no progress. The job was running (scene 33,
  pull out: room move, people pass, plate+mask on the Mac, then the render sent as take 09379998),
  but the stage was one small grey line in the panel only. Now a RoomPinnedProgress bar ("step k of
  5: <stage>", then "sent: the new take is rendering below") shows in the panel, in the guided card
  and above the takes; the job id is remembered per scene in the browser (localStorage), so a reload
  or a closed panel keeps the bar; it clears once the take is in the list (its own bar takes over).
