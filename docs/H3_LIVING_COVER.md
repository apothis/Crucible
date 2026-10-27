# H3 living-cover loop recipe (YouTube tab)

Moved verbatim from docs/MINIMAX_H3_PLAN.md section 11 on 2026-10-27 when the MV pipeline was rebuilt (docs/MV_H3_REBUILD_PLAN.md). This is what POST /api/video/h3_i2v (crop_to_canvas / last_raw / overrender) and /api/youtube/livecover_check implement. Nothing below has changed.

## 11. Loop recipe - same-still first/last loops (MEASURED 2026-09-06, all [V] unless marked)

Viking Selene living cover, Seedream still as first+last frame, 1280x736, seed 1908767699, ONE
variable per render, judged by pixel measurement plus the user watching the 3x-concatenated loop.

| step | what | last frame vs still | wrap jump / median step | motion in the final second |
|---|---|---|---|---|
| baseline | 16:9 still on the 1280x736 canvas, old prompt | 12.0 (frame 0: 3.9) | 17.8x | ~0.02 (parked from f225) |
| 1 crop_to_canvas | still centre-cropped to 1280:736 first | 5.3 | 6.6x | parked from f239 |
| 2 prompt | constant-speed cyclical grammar, no "returns to start" | 5.3 | 5.1x | alive to ~9s, then parks |
| 3 overrender +34f | 277f, alignment mark moved to 11.54s | 5.4 | 1.1x at the lock cut (f261) | alive to 10.5s |
| 4 last_raw | last frame fed raw (no 1.0MP nearest-exact rescale) | 3.5 (= VAE floor) | 1.2x | unchanged |
| all four | one clip, cut at lock f260 = 10.83s | 3.5 | 1.1x | user: "the video is fine" |

Why each works:
- `MiniMaxH3ImageToVideo` STRETCHES `first_frame` to the canvas but COVER-CROPS `last_frame`
  (docs.comfy.org). A 1.778 still on a 1.739 canvas is two different anchors; the model spends
  the last ~1.5s morphing between them and parks. Crop first (backend `_crop_image_bytes_to_aspect`).
- The author's `ImageScaleToTotalPixels(nearest-exact, 1.0MP)` on the last frame is a second,
  smaller geometric difference (aliased edges: tail sharpness climbs 128 -> 160). `last_raw` skips it.
- Guide frames are hard latent replacements re-injected every step: NO strength/mask exists, and
  `MiniMaxH3AddGuide` at frame -1 is the same coordinates as `last_frame` [V, ComfyUI source +
  PR #15439]. The FL2VA model "decelerates while converging toward the supplied last frame, then
  locks onto the final image" [Herrgotts H3 continuation suite]. So the settle cannot be removed,
  only pushed past the cut: render 2 grid steps longer (`overrender`, +34f, header rewritten to
  the real duration) and let `loop_tail_report(overrender=34)` cut at the LOCK POINT (first frame
  after which mean motion stays under 25% of the clip median). The cut frame matched frame 0 at
  1.1x a normal step - trimming costs nothing on the seam.
- Prompt grammar: the official FLF guidance teaches "progressively narrowing differences ->
  last-frame state"; "holds her pose ... returns to the exact arrangement" is that convergence
  and it parks early. Use constant-speed cyclical verbs, "repeating with no beginning and no
  ending", never settles/returns/finally. Added atmosphere (fog/smoke/snow) NOT in the still must
  enter after the first second and clear before the last (Seedance's trick, user 2026-09-06) -
  `/api/youtube/livecover_prompt {atmosphere}`.
- Alternatives ruled out: `AddGuide` strength (none exists), crossfade loops (user: Seedance FLF
  lands as-is, no crossfade needed), LTX-2.3 `last_strength` (a different lane).

Wired into the YouTube living-cover lane (YouTube.tsx `lcAnimate`): `crop_to_canvas`, `last_raw`,
`overrender` always on for that lane; "native 720p, no upscale" toggle (1280x736 hunt, pick skips
FlashVSR; default stays 864x480 + FlashVSR); "add atmosphere" select feeds the prompt writer.
API flags are opt-in and default OFF for every other caller.

Still-generation side [user verdict]: local single-image identity arms Krea2 identity-edit,
Qwen-Image-Edit-2511 and InfiniteYou (Flux-dev fp8, official graph + face keypoints wired) all
failed the bar against Seedream; character stills come from Seedream/OpenArt for now, hard shots
(e.g. Jormungandr) ship to Seedance, the box carries ordinary loops.
