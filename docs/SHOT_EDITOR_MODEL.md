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

Status: agreed 2026-06-28. Step 1 next.
