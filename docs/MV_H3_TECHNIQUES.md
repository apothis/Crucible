# MiniMax H3 music-video techniques

The rulebook. Every rule here is general: it applies to the next song, not to Never Mine.
The per-video log (what was fired, what it looked like, who watched it) is
`docs/MV_H3_REBUILD_PLAN.md`; this file is what that log has proved so far, stated once, with
where each rule came from and where it lives in code. A rule that is only a hypothesis says so.

Provenance tags, on every rule:

| tag | meaning |
|---|---|
| [M] | measured by us on the box (what, when, how many samples follow in brackets) |
| [U] | the user's verdict from watching the video; ground truth for motion and lip-sync |
| [P] | primary upstream source (MiniMax guides, node source, Comfy docs, model cards) |
| [C] | community report, not reproduced by us |
| [H] | hypothesis: consistent with what we saw, not established |

Code: `backend/mv_writer.py` (script writer, prompt compiler), `backend/mv_h3.py` (chain
graph), `backend/mv_stems.py` (vocal stems), `backend/mv_locations.py` (location library),
`backend/app.py` routes `/api/mv/*`. Where a rule is enforced by code the section says which
function; a rule only in prose is for the writer's system prompt or the person reading this.

---

## 1. Shape of a video

- **A video is an ordered list of scenes; a scene is one continuous H3 clip chain over a
  contiguous song window; between scenes is a hard cut.** [design, proved end to end: G3 chain,
  G4 cuts] Code: `mv_writer.normalize_scenes`, `mv_h3.chain_plan`.
- **Code owns timing, the model owns prose.** Clip grid (17k+5 frames), clip starts, which
  words fall in which clip, section boundaries and the fixed prompt boilerplate are computed;
  the LLM never chooses a number it could get wrong. [design; every writer run has produced
  valid timing since G2] Code: `plan_scene_clips`, `compile_prompt`.
- **Scene bounds fall on cut points: section edges and half a second before each sung line.**
  Cuts land where the words break instead of mid-phrase. [design after the user's G3 verdict;
  G5, G9 scripts honour it] Code: `cut_points`, `snap_cut`, `LINE_LEAD = 0.5`.
- **Mix performance with cinematic B-roll; the proportion follows the song, not a fixed ratio.**
  [U: the user's brief; G2 produced 6 singing / 4 B-roll scenes unprompted and the user read it as
  a real video] Prose rule in `SCENE_SYSTEM`.

## 2. The song lock

- **The song goes into the clip's audio latent with the noise mask at zero; H3 never generates
  or re-renders the audio.** This is `MiniMaxH3SongMaskedAVContext` (seitanism MultiRef Update
  9) plus `MiniMaxH3StreamLiveMusicVideoToVHS`, which muxes the untouched master. [P: node
  source and architecture doc] Code: `mv_h3.build_h3_chain`.
- **Why it matters: without the lock H3 lip-syncs to its own drifted rendering of the song on a
  bad seed.** [M: August 2026, about 90 takes; clean 0.87-1.00, drifted 0.36-0.40 on the envelope
  score] With the lock every chain since G1 scores r = 0.998-1.000 at zero lag. [M: G1, G3, G4,
  G6, G7, G8]
- **Floor-check every render anyway**: log-RMS envelope correlation between the clip's audio and
  the song window, best lag within 0.6 s. Below about 0.9 means the lock did not take or the mux
  is offset. [M] Code: `.mvwork/g1_refs/envelope_check.py`; `/api/mv/h3_audio_check`.
- **The mouth tracks a protected latent.** [U: "the mouth looked fine", G1 A1, full-mix lock,
  turbo 8-step, 544p] This was the gate question for the whole design and it is answered.
- **Lock source: the mix by default; the vocal stem and "vocal forward" (mix plus stem at +6 dB)
  are switches for dense songs, not a new default.** [U: the whole song in the latents may put
  the rest of the imagery on the beat; untested either way] The muxed master is always the mix
  window. Code: `mv_stems.lock_source`, chain `lock` option, node 41.
- **The window gets one extra second of real audio for the node's lookahead, and a final scene
  that overshoots the song is padded with silence and trimmed at assembly.** [M: G2 overshoot
  refusal, fixed] Code: `_h3_chain` in `app.py`.

## 3. Lyric alignment

- **Align on the isolated vocal stem when one exists.** On an easy song the stem already beat the
  mix: fewer lines started early on an instrument sound, held notes kept their true end; line
  start deltas median 0.08 s, max 1.48 s, all the disputed edges resolved in the stem's favour by
  stem-level energy. [M: Never Mine, 29 lines, Whisper small] Dense metal is where the mix fails
  outright. [H, from the August experience] Code: `/api/mv/write align_on=auto`.
- **Separation runs on the box (BS-RoFormer SW, :5070), 81 s for a 258 s song, ComfyUI freed
  first.** [M] Never on the Mac. Code: `mv_stems.separate_vocals`, `/api/mv/stems`.
- **Compare alignments by line position, never by matching line text**: repeated chorus lines
  match their first occurrence and produce 60-115 s phantom deltas. [M: my own error, 2026-10-27]

## 4. The singer: two reference pictures

- **Identity is pinned by two pictures per video: a face portrait (`<Picture 1>`) and a
  full-body wardrobe shot (`<Picture 2>`), both generated by the user from Claude's prompts.**
  No character LoRA was needed. [U: identity held across every chain the user watched]
- **The two references must agree on every visible detail: jewellery, sleeves, hair parting.** A
  detail only one of them shows is a coin flip per clip and the chain seam exposes it (the
  choker appeared at a seam when only the face picture wore it; regenerating the body picture
  with it fixed every frame). [M + U: G1 A1 vs A1b, same seed and prompts]
- **Never name in the prompt a detail that only one reference shows.** [same evidence]
- **Hero singing is medium close-up or close-up, face toward camera; wide shots are B-roll or
  back-to-camera.** Ref2VA distorts faces on wide shots. [C, consistent with our stills]
  Prose rule in `SCENE_SYSTEM`; `check_prompt` enforces readable-mouth framing on sung clips.
- **Describe wardrobe once for the video; weathering (rain, dust, wind) may be added, garments
  may not change.** [design] A retention line can hold a garment through a move: "the laced
  corset stays visible throughout" held the corset where earlier takes lost it. [M: G6 V3, G7,
  G8 vs V1/V2; stills]

## 5. Places: one person-free still per location

- **Nothing pins the place unless a picture does. A person-free still per location, handed to
  the chain as `<Subject N> is the location ... fully_preserved`, keeps the same room across
  cuts and matches the still.** [M + stills: G4, two scenes cut together; the earlier render of
  the same scene without the still had a different lamp and window] Code: `compile_prompt(env=)`,
  upload order face, body, location.
- **The model keeps the furniture identity, not the floor plan.** Camera, weather, time of day
  and state changes are free; layout is approximate. [M: G4, the chair moved relative to the
  window] Write the retention line that way; do not expect a literal plan.
- **A location is one physical place at one camera distance class; a detail of a room is its own
  location.** Few locations per video (3-6), each returned to. [design; every script since G5
  obeys it] Prose rule in `SCENE_SYSTEM`.
- **Environment-only scenes put the location alone as `<Picture 1>`; a scene with no still at
  all falls back to the no-reference lane with a "no referenced subjects" header.** [design,
  verified by dry run G9] Code: `compile_prompt`, `compile_scene(with_location)`.
- **Locations live in a library; the writer is offered the places that came from THIS song, so
  re-runs keep their stills.** With its own five places offered, a re-run reused all five and
  invented none; without them the previous run had invented three near-duplicates and orphaned
  the stills. [M: Never Mine G5 vs G9, same request] Code: `mv_locations.select` (default
  "auto" = same-song rows), `/api/mv/locations*`, `write_script(known_locations=)`,
  `render_scene` library fallback.
- **Offering ANOTHER song's places biases the writer: it takes them even when they do not fit.**
  Offered a Victorian parlour, mantelpiece and wrecked parlour, a Celtic banshee video put its
  father-by-candlelight verse and the proud's supper in the parlour and rationalised it; the
  control run with nothing offered invented a fisherman's cottage and a harbour tavern for the
  same beats. [M: She Sings Before the Storm, control vs library, same request, 2026-10-28]
  Cross-song reuse is therefore an explicit opt-in (`locations: "library"` or a list of ids),
  never the default.
- **More stills are always available.** A new place costs the user one image; never force reuse
  or drop a shot to save a still. Keyframes can be generated the same way when a shot needs
  them. [U, 2026-10-28; OpenAI adheres to prompts well]
- **Still prompts are written person-free, 16:9, naming camera position and height, every fixed
  element and where it sits in frame, materials, the first scene's light and weather, and the
  video's look.** [design; the five Never Mine stills came back usable first time]

### 4a. Reference-picture recipe (per video, two GPT images)

The user generates both from Claude's prompts with the canonical Selene portrait as the image
input. [design s.3.2; proved on Never Mine] The recipe, song-independent:
- **Plain seamless studio backdrop for both, nothing else in frame.** A place in the picture or
  in its description nudges the writer toward that place (the Never Mine body picture showed the
  parlour; its description named it). [H, flagged 2026-10-28; the reason for option B on the
  transfer test]
- **Picture 1: close three-quarter portrait, 85 mm, 3:4; picture 2: full length standing, facing
  camera, arms relaxed, 50 mm, 2:3.** Face in the song's makeup and key light; body in the song's
  one wardrobe. [design]
- **One wardrobe sentence, used verbatim three times**: in the body prompt, in the writer's
  `refs.body`, and (its neckline part) in the face prompt. Both pictures must agree on every
  visible detail; a detail only one shows is a coin flip per clip. [M + U: G1 choker]
- **Writer descriptions (`refs.face`, `refs.body`, `refs.shared`) are place-free**: identity,
  makeup, light, wardrobe; never a room or landscape. [same reason as the backdrop]
- **Identity constants for Selene**: long dark wavy hair, pale skin, grey-green eyes, dark brows.
  Everything else (makeup, jewellery, wardrobe) is per video. [the pictures on file]
- **Look: the first draft is the sensual version** (bodice laced low, a shoulder slipped, back
  arched), period-correct for the song's world, nudity excluded; **one reveal per image** and no
  cleavage/sheer/sensual vocabulary, which the GPT filter refuses. [U 2026-09-15; filter notes]
- **No brand names, no text, no watermark** in any prompt. [filter notes]

## 6. Take length and setups

- **Performance takes 8-15 s by default, never more than 28 s; B-roll up to 45 s; nothing under
  5 s.** A 42 s single take with one slow move is too long for a verse. [U: G3 scene 2] Prose
  rule in `SCENE_SYSTEM`; G5 and G9 scripts have 18 of 20 scenes as single clips.
- **A verse or chorus is two or three setups in the same location, cut between them** (chair
  front-on, then three-quarter from the lamp side; wide, then detail). The location still keeps
  the room across the cuts. [U + M: G4 cut reads as one room]
- **One continuous clip (362 f, 15.08 s) beats a chain where the take fits**: no seam, no
  overlap lost, no continuation prefix to disagree with the text. [M: G4 onward]

## 7. Camera

This is the most-tested area and still has an open end. Everything below was measured on one
seated performance shot, one seed, with only the camera wording or the resolution changed.

- **Write camera in the model's own grammar: one action, amplitude small or large, speed slow or
  fast, one framing, bounded only by what must stay in frame, "throughout the take, one
  continuous shot with no cut and no dissolve".** [P: MiniMax base prompt guide 4.3; M: G6-G8]
  Code: `SCENE_SYSTEM` camera bullet, `CLIP_SYSTEM`, schema `camera` + `keep_in_frame`.
- **Metric distances mean nothing.** "12 cm over 15 s" and "a hand's width" both ran to a tight
  close-up. [M: G1, G4, two renders]
- **Never name a start framing AND an end framing for one move.** "From a medium shot to a medium
  close-up" was rendered as a hidden cut dressed as a cross-dissolve, with a ghost frame and the
  wardrobe changing across it. [U: "loads of weird artifacts"; M: 1 s frames, G6 V1 and V2]
- **One framing plus the continuous-shot wording gives a genuine continuous move with the
  wardrobe held.** [M: G6 V3 at 544p, G8 at 768p; stills] Its EXTENT is the open problem: at
  544p the push ran unbounded; at 768p one wording stopped gently (G7, V1 words) and the other
  pushed further than asked, the lamp leaving frame (G8, V3 words). One sample each; seed
  variance not excluded. [H] The user watched both 768p takes and finds both usable, preferring
  G8's move. [U]
- **Performance takes use small amplitude at slow speed; large amplitude only on B-roll.**
  [P guide + design] Static is an option, never the default: a locked-off singer looks dead. [U]
- **The camera reference video lane (`<Video 1>`, weak_reference, pace only) added nothing
  visible and tripled render time.** [M: G6 V2, 8.5 to 25 min] Built and kept for moves words
  cannot express; not the default. Code: chain `ref_video_names`, route `camera_ref`.
- **Keyframe-bounded moves (`MiniMaxH3AddGuide`, `CustomKeyframes`) fix a move's end composition
  exactly but pin her pose, so B-roll and environment scenes only.** [P: node behaviour; not
  built in the MV lane yet] GPT keyframes are available on request. [U]
- **Camera LoRAs add vocabulary, not bounds, and a third LoRA over Turbo plus two identity
  pictures risks identity.** Fallback only. [P cards; untested here]
- **Production fallback that always works: render static, FlashVSR x2, ffmpeg zoompan of 4-8 %.**
  No parallax; on a seated singer a 2D push reads as a dolly. [design; the static profile take
  held clean for 15 s, M: G4]

## 8. Pace

- **One described motion per shot with a real-world duration anchor; pin the sky whenever it is
  visible.** Unmentioned sky gets animated and reads as timelapse; a busy prompt accelerates
  through the clip, a disciplined one settles. Prompt accounts for about 30 % of the motion
  magnitude, the turbo recipe about 8 %. [M: August 2026, one shot, one variable per render]
  Prose rules in `SCENE_SYSTEM` ("name any visible sky", one main motion) and `CLIP_SYSTEM`.
- **Pace can only be steered positively: the chain is cfg-free, there is no negative prompt.**
  [P: BasicGuider] Do not blame the turbo LoRA for pace; that was true of LTX, not H3. [M]
- **Open-sky wide B-roll is intrinsically more motion-prone; prefer a near-foreground anchor for
  held shots.** [M corollary]

## 9. Resolution and recipe

- **Judge performance takes at the turbo LoRA's trained size, 1344x768.** The same camera
  wording that degraded into a cut at 960x544 rendered as a gentle bounded move at 768p. [M:
  G6 V1 vs G7; one sample, H that it is the resolution] 768p is also the delivery size (YouTube
  serves it as 720p; 1080p native is outside the trained range).
- **Cost: about 8 min per 362-frame clip at 960x544, about 23 min at 1344x768 on the 3090**,
  with three pictures and the song lock in memory. A 24-clip video is 10-11 h at 768p. [M]
  544p is for throwaway checks only.
- **Recipe: Ref2VA pruned int8, lightx2v turbo 8-step v1.1 768p LoRA at shift 6/3, comfy
  kitchen attention, no Spectrum.** [P: Comfy docs, lightx2v card; M: every chain since G1]
  Sage goes in-graph per model, never on the launcher. [P + M: Krea2 black renders]
- **Stills handed to the chain's first/last frame slots must be pre-cropped to the canvas
  aspect**: the first frame is stretched to the canvas, the last is cover-cropped. [P + M:
  loop lane, 2026-09] Reference pictures (face, body, location) are not frame slots and need no
  crop.

## 10. Continuity inside a chain

- **Clip N+1 continues clip N through a 39-frame (1.625 s) protected prefix; the seam is a
  crossfade inside the streamer, not an edit.** [P] Seam motion stayed inside the clip bodies'
  range on every measured chain. [M: G1, G3]
- **Never pre-assert the incoming framing in clip N+1's text.** Text and prefix can disagree and
  the model re-establishes. [M: G1 A1 reset to a medium shot; A1b with identical text did not,
  so it is per-render variance, H as a cause] The safe form: "continue first, then develop".
  Prose rule in `CLIP_SYSTEM`.
- **A chain overshoots the scene end to the grid; trim at assembly or render the last clip at a
  shorter grid length.** [M: G3, 12.6 s overshoot] Shorter last clip not built yet.
- **Unprompted motion appears on long chains** (a hand to the face at 27 s, a white streak
  entering top-right in a third clip). [M + U] Short single-clip takes avoid the exposure.

## 11. What is luck and what is a rule

- **One render proves nothing about variance.** The same prompt line "eyes rest just below the
  lens" gave lids-down openings in two renders and eyes-open in four; the same wording gave both
  a cut and a clean move at different sizes. [M: six chair takes] Anything you want reliably
  must be written explicitly; anything not written is the model's choice per render. The user
  chose to leave the eyes to the model. [U]
- **Judge a rule over seeds, judge motion by watching.** Contact sheets and 1 s frames find
  cuts, ghosts, wardrobe drift and framing extent; they cannot judge lip-sync, pacing or the
  feel of a move, and have misread depth of field as drift and a fine push as "too extreme".
  [U corrections, twice] The user watches; stills gate what is worth watching.

## 12. Loops (same-still first and last frame)

For background loops and living covers, also usable as B-roll plates. Full recipe in memory
`h3-same-still-loop-recipe`; the rules that transfer:
- pre-crop the still to the canvas aspect; constant-speed ambient motion with no return-to-start
  language; over-render one grid step and cut at the lock point; the seam is a velocity dip, fixed
  by dropping the eased head and crossfading a moving tail into a moving head. [M, U: 2026-09]
- anything added that is not in the still must arrive after t = 0 and clear before the end.
  [U: 2026-09-06]

## 13. Process rules (how a test is run)

- **Gate before the GPU.** `render_scene` with `dry_run: true` returns the refs in upload order,
  the resolved location and where it came from, frames and the full prompts, with nothing queued.
  Read it before firing. Code: `/api/mv/render_scene dry_run`.
- **One variable per render, same seed, same pictures, same window**; descriptive filenames
  stating the condition; `ffmpeg -c copy` for the copies. [feedback rules]
- **Confirm a job is actually running on the box (`:8188/queue`) before reporting it running**,
  and floor-check the audio when it files.
- **Collectors skip `type: "input"` history entries**: a LoadVideo node echoes its input clip into
  history and a reference clip was filed as the render once. [M, fixed 5d48f28]
- **Iterate by changing the rule in the writer and re-running it through the normal path, not by
  hand-editing a script.** A hand-edited script is a single diagnostic; its finding goes back into
  the writer the same day. [U, 2026-10-28]
- **Validate on a second, different song before calling a technique proven.** Never Mine is the
  easy case: clean vocal, slow tempo, one singer. [U]

## 14. Open items, in order of what they would settle

1. Second-song transfer test, steps 5-6: the writer half is done (library bias measured and
   fixed; stem alignment on a dense Celtic mix gave cover 0.894 and a clean 25-scene script).
   Still open: the camera grammar on a standing or walking performance and an environment-only
   scene, three shot types at 768p.
2. Camera extent at 768p: a second seed of each wording; whether "keep in frame" bounds a push
   at all, or only amplitude does.
3. Keyframe-bounded B-roll moves (AddGuide with a GPT keyframe as the end frame): built in the
   chain graph, not in the MV lane.
4. Final assembly: concat at scene offsets, overshoot trim, shorter last clip per scene.
5. Shot-type regression set: a few canonical shots rendered at 768p whenever a writer rule
   changes, judged by type.
6. Stem-lock A/B on a dense song (mix vs vocals vs vocal forward): does the whole mix in the
   latents put B-roll on the beat, as the user expects.
