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
