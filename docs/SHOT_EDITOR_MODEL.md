# Shot Editor model — "the editor IS the shot"

The target mental model after we simplify. One sentence: **MV Studio is the master
timeline of shots; each shot is authored and played in ONE LTX Director editor; the
shot's rendered clip drops into its slot on the master timeline.**

## Two levels (only two)

1. **MV Studio = master timeline.** A row of shots in order, each with a fixed time
   slot (from the song structure). You sequence shots and assemble the final video
   here. You do NOT author shot *content* here — just order, timing, the song.
2. **Shot editor (Shot Studio) = ONE shot.** The embedded LTX Director timeline editor
   is the whole surface. Everything for that one shot happens here.

There is no third system. (We are REMOVING the parallel FFLF "pieces" + chain-assembly.)

## Inside a shot (the editor)

- **Author:** drop keyframe stills at positions on MAIN + per-segment prompts (prompt
  relay) + optional audio. This is the generation *input*.
- **Generate (one button):** reads the editor timeline + the shot's settings and calls
  the right backend graph under the hood (the user does not pick graphs):
  - lip-sync / character singing → MSR graph (identity refs + masked audio)
  - scenic / camera move / still-to-still → keyframe (LTXDirector) graph
  - push-in B-roll → two keyframes (full + center-crop)
  The result is ONE clip.
- **Result → a video segment** on a dedicated **Result track** (separate row from the
  input stills, per the "separate tracks" choice), so the editor's **play button plays
  the rendered clip** on the canvas. Inputs (stills) and output (video) never overlap.
- **Extend (one concept):** LTX Director's native way — the result clip is re-fed as the
  starting video segment, you add keyframes/prompts after it, Generate again → a longer
  clip. No separate "pieces"/"chain assemble".
- **Variations:** "generate N, pick one" (what seed-hunt becomes), shown as pickable
  result takes — NOT a separate workflow.

## Feedback to the master timeline

The shot's chosen result clip = `block.clipId`. That is what plays in the shot's slot
on the MV Studio timeline and what the final assemble uses. "Use this take" = set
`clipId`. That's the only handoff between the two levels.

## What we are removing (the mess)

- The FFLF "pieces" card grid + chain-assembly endpoint usage as a *separate* system.
- The separate seed-hunt → finish → extend buttons as their own flow.
- The standalone FFLF anchor pickers (anchors are keyframe stills in the editor).
- The "Rendered segment" box / duplicate result players (the editor IS the player).

## What stays

- The backend graphs (MSR / keyframe / FFLF push-in / i2v) — they are the *engines*
  Generate dispatches to. The non-distilled option + steps stay as render settings.
- MV Studio master timeline + assemble.
- The LTX Director editor (now used for real: author + play + extend).

## Incremental refactor steps (each builds + is testable)

1. Result playback: load the rendered clip onto a dedicated **Result track** in the
   editor (separate from stills) so play works cleanly. Remove the top result box.
2. One **Generate** button in the editor area that dispatches by shot type and loads the
   result onto the Result track + sets `clipId`. Keep old paths working until proven.
3. Fold seed-hunt into "Generate ×N → pick" (result takes), remove the separate hunt UI.
4. Make **Extend** re-feed the result as a video segment (native), remove pieces/chain.
5. Delete the now-dead FFLF pieces/anchor-picker/assembly code + endpoints.

Status: agreed 2026-06-28. Step 1 (Edit inputs / View result toggle) done.

---

## v2 — full per-shot REDESIGN (agreed direction 2026-06-28)

Drop the embedded LTX Director editor as the shot surface. Replace Shot Studio with a
**guided staged flow** — each stage produces a CHEAP artifact you approve before paying
for the next (the "gate artifacts before render" rule). Much more usable; keeps every
working engine under the hood (MSR / FFLF push-in / keyframe / i2v + seed-hunt + extend).

### The stages (a shot is built in order)

1. **Scene** — the BACKGROUND still, person-free. Describe the setting → generate 3 →
   pick. This is the cheap gate + the MSR background (must contain NO person, per
   [[project_msr-background-must-be-personfree]]). For B-roll this IS the shot's image.
2. **Cast** — who's in it: ONE MSR-anchored lead (from the character library) + optional
   band-in-scene members composited into the background (named, instrument, side) + the
   mandatory drummer for stage shots (per [[project_band-shots-were-solo-msr]]). B-roll =
   no cast.
3. **Placement preview** — render a cheap STILL to approve the COMPOSITION before the
   video: toggle **background-only** vs **with the character(s) composited**. This is the
   "render with/without chars for initial placement" the user needs. Approve, then video.
4. **Video options** — action prompt + lip-sync toggle → "generate 3 options" (seed-hunt,
   half-res) → pick the best (plays).
5. **Result / Extend** — the chosen take plays; Extend continues it; "Use on timeline"
   sets `block.clipId` → the master timeline slot.

Shot type **Performance** vs **B-roll** chooses the engine + which stages show (B-roll
skips Cast + composite, uses keyframe/FFLF push-in; Performance uses MSR).

### Must-keep details (don't lose these in "simpler")
- Background still is FIRST-CLASS and person-free; it's reused as the MSR background.
- With/without-character still preview BEFORE the expensive video (cheap approval gate).
- Band-in-scene composite (other members named in the action; drummer always present).
- Seed-hunt = "generate N options"; extend; dev-model/steps under Advanced.
- Don't over-promise from mockups — build it real in the app and iterate on polish.

### Build order (incremental, each testable in the app)
1. Scene stage: background-still generate-3/pick (reuse genStill), person-free prompt.
2. Cast stage: lead + band-in-scene from the character library.
3. Placement preview: still render with/without character (reuse MSR still/composite).
4. Video options: wire seed-hunt → pick → result (reuse the working video render).
5. Result/Extend/Use-on-timeline; then delete the old Shot Studio internals.

Status: redesign agreed 2026-06-28; **BUILT clean 2026-06-28** as `web/src/ShotEditor.tsx`
(replaces ShotStudio in MV Studio's "✎ Edit segment"). All stages wired to the existing
engines; backend `/api/video/crop_still` added for the B-roll push-in's closing anchor.
The old `ShotStudio.tsx` + `LtxDirectorEditor.tsx` + vendored editor are left in the tree
but UNUSED — delete them once the new flow is proven in the app. Verified: tsc, vite build,
and the crop endpoint smoke-tested (real still → 924KB PNG served). NOT yet eyeballed by the
user against a live render.

Re-verified 2026-07-24: `MVStudio.tsx` imports ONLY `ShotEditor` (mounted as its "Edit segment"
view); nothing imports `ShotStudio`, and only `ShotStudio` imports `LtxDirectorEditor`. So those two
plus `web/src/vendor/` are confirmed dead code, safe to delete. The staged flow's actual dispatch:
Performance -> `ltx_msr` two-stage (`mode:"hunt"` x3 at half res -> pick -> `mode:"finish"` on the
same stage-1 seed, lip-sync added at finish with `isolate_vocal:false`); B-roll -> `brollMotion`
selects `fflf` push-in (closing anchor from `/api/video/crop_still`), `fflf` two-still,
`ltx_keyframe` (N stills at frame positions), or `ltx_i2v` (prompt camera move, dev model forced on).
