# MV AI Colour-Grading Plan

Replace the music-video pipeline's ffmpeg "look" grading (the hand-tuned `eq`/`curves`/
`colorbalance` chains in `backend/musicvideo.py` `GRADES`, applied in `assemble()`) with
ML / colour-science grading that runs in ComfyUI on the GPU box. The ffmpeg looks are crude
global ops: they cannot do filmic tone-mapping (highlight roll-off), skin-tone-aware
adjustment, halation/bloom, or film-grain structure - which is the "cheap filter" feel we
want gone.

## Research verdict (2026-06-23)

Diffusion *regrade* (IC-Light / Light-A-Video) is the WRONG tool: it relights, flickers
temporally, drifts per-clip, and will not hold one cohesive palette across ~30 separate
clips. Skipped. The right tools are colour-science emulation + a diffusion **LUT generator**
(the diffusion runs once to produce a fixed lookup table, not to repaint pixels).

## Decided architecture

Grade **per clip, BEFORE assembly** (not the final stitched mp4). Reasons:
- Mirrors the existing per-block FlashVSR upscale pattern (`upscaleBlock` -> `videoFlashvsr`
  -> `upscaledId`). Add a `gradedId` the same way; `assemble()` already prefers
  `upscaledId || clipId`, so it just picks up the graded clip. `assemble()` then runs with
  `grade="none"` (raw stitch) since the look is already baked.
- Loading a full 3-4 min 720p video as one frame batch into ComfyUI would blow system RAM
  (same whole-clip-in-RAM wall noted for FlashVSR). Short per-clip batches are safe.
- Consistency is guaranteed by construction: the SAME film stock / SAME generated LUT on
  every clip. Nothing content-adaptive => no drift.

Three layers:

### 1. DEFAULT look library - ComfyUI-Darkroom (no reference needed)
`jeremieLouvaert/ComfyUI-Darkroom`. MIT, CPU (numpy/scipy), **no model downloads** (ships
its LUTs). ~196 named looks: 161 film stocks (Capture One curve data) + 35 spectral preset
LUTs, plus halation, grain, H&D tone curves, lift/gamma/gain, HSL-selective. Pick a look by
NAME - no reference image required. This is the drop-in replacement for the `GRADES` list and
the default grading path.

### 2. Reference-driven looks - Kijai VCG (the "neural LUT")
`kijai/ComfyUI-VideoColorGrading` - ComfyUI wrapper of ICCV 2025
"Video Color Grading via Look-Up Table Generation" (arxiv 2508.00548), CC-BY-4.0.
Two-stage diffusion **generates a 16^3 (4096-entry) 3D LUT from a reference still**, then
applies it. Because the output is a fixed LUT, generating ONE LUT and applying the SAME LUT
to every clip gives automatic temporal + cross-clip consistency. The 4GB model only loads
when *creating* a new look; applying the resulting LUT is cheap.
- Model: `vcg_combined_fp16.safetensors`, 4.12 GB, **ungated** (no token), at
  `https://huggingface.co/Kijai/VCG_comfy/tree/main/checkpoints`
  (contains CLIP ViT-B/32 + VAE + ReferenceNet + L-Diffuser).
- Nodes: `Load VCG Model`, `Generate Color LUT (VCG)`, `Apply 3D LUT (VCG)`.
- Reference material = curated public-domain / CC0 stills (starter library, see below) +
  bring-your-own frames. Generated looks save into a growing personal look library.

### 3. Fallback - KJNodes ColorMatch (already installed)
We already run KJNodes (`VAELoaderKJ`, `ImageResizeKJv2`), so `ColorMatch` is available now -
only needs `pip install color-matcher` in the Comfy env. Statistical reference transfer
(`mkl`/`hm`/`reinhard`/`mvgd`). Match all clips to ONE fixed reference for cohesion. The
zero-download lightweight path if VCG is overkill.

### Bridge
Any look (a Darkroom stock or a VCG-generated LUT) bakes to a `.cube`. The runtime apply step
is then a cheap LUT apply (no 4GB model resident) and could even run via ffmpeg `lut3d` on the
Mac with no box round-trip. Keep this in mind as a later optimisation.

## Box install (all ungated, no HF token)

1. `git clone https://github.com/jeremieLouvaert/ComfyUI-Darkroom` into `...\ComfyUI\custom_nodes\`, restart ComfyUI. (Optional `pip install POT` for its enhanced Color Match.)
2. `git clone https://github.com/kijai/ComfyUI-VideoColorGrading` into `custom_nodes\`, restart. Download `vcg_combined_fp16.safetensors` (4.12 GB) from `https://huggingface.co/Kijai/VCG_comfy/tree/main/checkpoints` into the folder the node expects (confirm from its README - likely `models/diffusion_models/` or a `VCG` subdir).
3. `python_embeded\python.exe -m pip install color-matcher` (enables the KJNodes ColorMatch fallback).

## MUST VERIFY ON BOX before first run (no GPU fired yet)

Per our "mirror the author's recipe / verify via /object_info" rule, these are placeholders in
the scaffolding and MUST be confirmed against `GET /object_info` after install:
- Darkroom film-stock node: exact `class_type` + input keys (stock name, grain, halation).
- VCG: exact `class_type`s + input/output keys for Load / Generate / Apply, and the model
  loader folder.
- VCG real VRAM + runtime (the docs give no figure; read the box load log - estimate is
  comfortably under 24GB / likely < 8GB, unconfirmed).
- Whether `Generate Color LUT` can run from the reference alone (or a fixed frame montage) so
  one shared LUT covers all clips; if it strictly needs per-clip source frames, generate the
  LUT once from a one-frame-per-clip montage and reuse it.

## Curated public-domain reference starter library (for VCG/ColorMatch)

Goal: a small bundled set of royalty-free reference stills, grouped by mood (warm/teal/noir/
desaturated/vibrant/amber...), shipped under `library/grade_refs/` with a `manifest.json`
(`{id, name, mood, source, license}`). Candidate sources (all PD / CC0, verify each file's
status before bundling): Wikimedia Commons PD, Library of Congress PD photo archives,
pre-1929 public-domain film frames, CC0 photography (e.g. museum open-access). User-supplied
frames drop in alongside. This curation is a separate follow-up task (involves fetching +
license-checking each image); not done yet.

## Backend integration (scaffolded, inert until called)

`backend/video.py` (new builders; node-specific class names/keys flagged for box verification):
- `build_darkroom_grade(p, video_ref, fps)` - `VHS_LoadVideo` -> Darkroom film-stock node ->
  optional grain/halation -> `CreateVideo` -> `SaveVideo` (prefix `videogen/regrade`).
- `build_vcg_lut(p, ref_image_ref, frames_ref=None)` - `LoadImage(ref)` -> `Load VCG Model`
  -> `Generate Color LUT (VCG)` -> save the LUT. Fired ONCE per look.
- `build_vcg_apply(p, video_ref, lut_ref, fps)` - `VHS_LoadVideo` -> `Apply 3D LUT (VCG)` ->
  `CreateVideo` -> `SaveVideo`. Cheap per-clip apply (no diffusion).

`backend/app.py` (new endpoints, mirror `video_flashvsr`):
- `POST /api/video/regrade` `{video_id, look_source: "darkroom"|"vcg"|"colormatch", film_stock?, grain?, halation?, lut_id?, ref_still_id?, cm_method?}` -> graded clip.
- `POST /api/mv/generate_lut` `{ref_still_id}` -> generates + stores a VCG LUT, returns `lut_id`.
- `GET /api/mv/look_library` -> `{darkroom_stocks: [...], luts: [...], refs: [...]}` to feed the picker (data-driven like `/api/mv/grades`).

## Frontend (`web/src/MVStudio.tsx` + `api.ts`) - wire after endpoints are live

- `api.ts`: `videoRegrade`, `mvGenerateLut`, `mvLookLibrary`.
- The assemble-time `Grade` `<select>` becomes a **look-source** control:
  - "Film stock" -> Darkroom stock dropdown (no reference).
  - "Match reference" -> pick a reference still (curated lib or upload) -> "Generate look"
    (one VCG call -> `lut_id`).
  - "Grade all blocks" loops blocks calling `videoRegrade` (same loop as `upscaleBlock`),
    storing `gradedId` per block.
- `assemble()` sends `clip_id: b.gradedId || b.upscaledId || b.clipId` and `grade: "none"`.
- Keep the ffmpeg-look dropdown as a labelled "fast / no-box" fallback.

## Status

**SCAFFOLD ONLY - nothing here is live (re-checked 2026-07-24).** The code exists but has never run:
`build_darkroom_grade` / `build_vcg_lut` / `build_vcg_apply` in `backend/video.py` and the
`/api/video/regrade`, `/api/mv/generate_lut`, `/api/mv/look_library` endpoints are written against
UNVERIFIED node class names, and **no frontend calls them** (`api.videoRegrade` / `api.mvGenerateLut`
are defined in `api.ts` but unused). `/api/mv/look_library` returns empty `darkroom_stocks`/`luts`.

The grading that ACTUALLY runs is the ffmpeg look chain applied per segment at assemble time
(`musicvideo.GRADES`, ~20 looks, exposed via `/api/mv/grades` and the MV Studio Grade picker).

- [x] Research + decision (this doc) - 2026-06-23
- [x] Backend builders + endpoints written (against UNVERIFIED node names - see above)
- [ ] Box install (Darkroom + VCG + color-matcher) - USER
- [ ] /object_info verification of node class names/keys + VCG VRAM
- [ ] Curated PD reference starter library (`library/grade_refs/manifest.json` absent)
- [ ] Frontend wiring (look-source picker, per-block grade loop, `gradedId` in assemble)
- [ ] Frontend look-source picker
