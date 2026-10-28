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
