# Plan: Full script-writer-driven LTXDirector pipeline

## Context

We adopted **LTXDirector** (WhatDreamsCost) as the single relay engine for the music-video pipeline because it is far more feature-rich than the bare per-prompt encode we were using: it can schedule **per-segment prompts** across one shot, drive **keyframe-image guides**, do **retake regions** (re-render a frame-slice), and handle **audio/motion** tracks - all from one `timeline_data` JSON. Today our pipeline only uses the thin prompt-relay subset, and the **script writer only ever emits one flat prompt per shot**. The goal: make the **script writer author the full per-shot timeline**, so the LLM can direct shots like "walks in, looking down (0-4s) | lifts head and belts, arms rising (4-8s)" and designate keyframe stills/camera moves - driving LTXDirector's full capability.

Already done this session (committed): the box node updated to latest; MSR + i2v unified on `LTXDirector` (rewired `build_ltx_msr` node 9 off the standalone `PromptRelayEncode`, fixed the i2v `frame_rate` output index 5→6, gated `_msr_available` on `LTXDirector`); per-segment prompt scheduling verified (arms-down→arms-up in one MSR shot); the MV-Studio NLE timeline controls + export/import; sage dropped from the launcher.

## Key architectural constraint (resolved by exploration)

**Keyframe-image guides and MSR identity guides cannot coexist in one graph.** `LTXAddVideoICLoRAGuide`/`LiconMSR` (identity) prepend ~17 reference frames to the latent; `LTXDirectorGuide` (keyframes) inserts images at absolute `insert_frames[]` - the prepend corrupts those indices, and `LTXVCropGuides` would strip both. So we use **two render modes**, mirroring our existing MSR-vs-FLF split:

- **MSR mode** (`build_ltx_msr`, exists): identity + free motion + **per-segment PROMPT timeline** (text scheduling, no keyframe images). For walk/sing/performance shots.
- **Keyframe mode** (new `build_ltx_keyframe`): `LTXDirector` → `LTXDirectorGuide` (no MSR IC-LoRA) with **keyframe still segments + per-segment prompts**. For still-to-still transitions, camera moves, B-roll - replaces the old `build_ltx_flf`.
- **Retake** (later): `LTXDirectorGuide` `retake_mode` re-renders a frame-slice of an existing clip.

The **script writer chooses the mode per shot** (it already distinguishes performance/narrative/broll + lipsync).

## Phases (build incrementally, commit per phase)

### Phase A - Script-writer-driven prompt timelines (foundation, MSR shots)
The prompt-segment path is already plumbed end-to-end (`Block.segs[]`/`tlOn`/`global`/`epsilon` → `msrPayload` `local_prompts`/`segment_lengths` → `build_ltx_msr` → LTXDirector, verified). Only the **script writer** doesn't produce segments. Changes:
- `backend/musicvideo.py` `build_prompt()` (~L223-297): add an optional `segments: [{frames, prompt, keyframe?, note?}]` field to the requested shot JSON schema (~L262-269) + a "TIMELINE PROMPTER" guidance section (when to segment a shot: action/camera/emotion change mid-shot; place segments against the lyric lines it already sees via `_song_summary`).
- `backend/musicvideo.py` `parse_shots()` (~L318-341): safely extract `segments`.
- `web/src/mvmodel.ts` `ScriptShot` (~L135) + `shotToBlock()` (~L152-168): map `segments` → `Block.segs[]`, set `tlOn = true` when present.
- No UI change needed - the Timeline Prompter inspector (MVStudio.tsx ~L673-700) already renders `segs`.

### Phase B - Keyframe render mode (new builder, LTXDirector + LTXDirectorGuide)
- `backend/video.py`: add `build_ltx_keyframe(p, keyframes, ...)` - graph: model stack (distill+detailer, **no** MSR IC-LoRA) → `LTXDirector` (relay + `timeline_data` carrying image segments with `isEndFrame`) → `LTXDirectorGuide` (consumes `guide_data[9,4]`, `ic_lora_name="None"`, applies keyframe stills to the latent) → CFGGuider → sampler → decode. Build the `timeline_data` JSON from the segments (each `{type:"image", start, length, imageFile|imageB64, isEndFrame, guideStrength, prompt}`). Retire/redirect `build_ltx_flf` to this.
- `backend/app.py`: `/api/video/ltx_keyframe` endpoint (mirror `/api/video/ltx_msr`); upload each keyframe still to ComfyUI input and reference by `imageFile`.
- `web/src/mvmodel.ts`: add `"keyframe"` to `RenderMode`; extend `Seg` with optional `keyframeStillId` + `isEndFrame`; `msrPayload`-style `keyframePayload()` builder.
- `web/src/MVStudio.tsx` `genBlock()`: route `renderMode === "keyframe"` to the new endpoint.

### Phase C - Shot-internal timeline editor (the "use their editor" UI)
A per-block mini-timeline (new `web/src/ShotTimeline.tsx`) reusing MVTimeline's WaveSurfer + RegionsPlugin + NLE-keybindings + beat-snap, scoped to one block's audio window. One draggable region per segment; click to edit its prompt + pick start/end keyframe stills; add/delete/split segments. Emits updated `Block.segs[]` (and `timeline_data` for keyframe mode). Replaces the plain `{len,prompt}` list in the inspector.

### Phase D - Advanced, script-writer-driven (last)
- **Retake regions**: `build_ltx_retake` via `LTXDirectorGuide` `retake_mode` (+ `retakeStart`/`retakeLength`) to re-render only a glitchy slice of an existing clip; a "retake region" UI on the shot editor.
- **Audio/motion segments**: `audioSegments`/`motionSegments` in `timeline_data` (per-region vocal for lip-sync, IC-video motion drive). Extend `Seg` with `audioId`/`audioStart`/`motionPrompt`.
- Script-writer: teach it to emit keyframe designations, retake hints, and (optionally) list available background stills so it can reference them.

## Files to modify (representative)
- `backend/musicvideo.py` - script-writer schema + guidance + parse (Phase A, D)
- `backend/video.py` - `build_ltx_keyframe`, `build_ltx_retake` (Phase B, D)
- `backend/app.py` - `/api/video/ltx_keyframe` (+ retake) endpoints (Phase B, D)
- `web/src/mvmodel.ts` - `ScriptShot`/`Seg`/`Block` types, `shotToBlock`, payload builders (Phase A, B, D)
- `web/src/MVStudio.tsx` - `genBlock` routing, inspector wiring (Phase B, C)
- `web/src/ShotTimeline.tsx` (new) - shot-internal editor (Phase C)

## Verification
- **Phase A**: `/api/mv/script` on Garden of Ashes → confirm shots come back with `segments`; open a generated multi-segment block in MV Studio (Timeline Prompter shows the segs); render it via MSR and eyeball that the action follows the segment timeline (like the proven arms-down→arms-up test) at the right lyric moments.
- **Phase B**: render a keyframe shot (two library stills, start+end) → confirm a clean still-to-still transition with the per-segment prompt applied, no MSR frame-offset corruption.
- **Phase C**: in the preview server, drag a segment region, edit its prompt + set a keyframe still, render - confirm it round-trips.
- **Phase D**: retake a 2s slice of an existing clip → only that region regenerates.
- Each phase: `npx tsc --noEmit` + `npm run build` (frontend), `py_compile` (backend), backend restart, and a real render on the box. Commit per phase (user-only attribution).

## Notes / risks
- Keyframe stills source: the script writer designates intent; stills come from the existing background/still generators (we already generate per-block backgrounds) or library picks. Phase B starts with library-still keyframes; auto-generation can follow.
- `build_ltx_flf` (old 2-frame TTP path) is superseded by Phase B's N-keyframe `build_ltx_keyframe`; keep it until B is verified, then redirect.
- All new shot capabilities stay backward-compatible: a single-segment shot with no keyframes renders exactly as today.
