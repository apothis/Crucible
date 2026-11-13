# Nobody Stays Dead - music video plan

Per-video plan and log. General rules live in `docs/MV_H3_TECHNIQUES.md`; anything this video
proves goes there as a general rule, not here.

Status: PLANNING (started 2026-11-13). Nothing rendered.

## 1. Brief (user, 2026-11-13)

- "Funny, sexy, and with lots of action to suit the tempo of the song", needs careful planning.
- Concept: **graveyard party**. She claws out of her grave at midnight and turns the cemetery
  into a party; the dead try to dance and fall apart; the terrified living board up their
  windows; in the bridge the living give in and join.
- Era: **modern night** (a present-day town). This breaks the "nothing modern" look of the
  other Apotheon videos on purpose.
- People: **anonymous crowds**, no recurring comic foil. She is the only referenced identity.

## 2. Song facts

- AUDIO FOR THE VIDEO (user, 2026-11-13): the Mixea High Neutral master,
  `Downloads/Mixea_HighNeutral_Nobody Stays Dead.wav`, imported to the library as
  `67bb77e877a049a0b7cb426cdfd17ab0` ("Nobody Stays Dead (Mixea High Neutral)", samples identical
  to the file). Same arrangement and length as the earlier mix `e7413da4...` (229.93 s, aligned
  within 5 ms by cross-correlation at 10/100/200 s), so the beat grid below and T0 still hold.
- D minor.
- [M 2026-11-13, backend/mv_beats.py, CORRECTED] 155.00 bpm constant (fit residual 6 ms over 588
  beats), beat 0.387 s, bar 1.548 s, 4-bar phrase 6.19 s, first downbeat 0.089 s (downbeat phase
  by harmony change + kick, margin 3.26). The earlier "152 bpm, bar 1.58 s" was librosa's tempo
  readout, which is quantised to its tempogram bins; the line fit of the tracked beats is exact.
  T0 cut on the tracked beats themselves (beat 224 = a downbeat), so its cuts were on bar lines;
  only its "1.58 s" labels are 2% long.
- [M] Loudness per 4 bars against the song median: within about 1 dB the whole way, one dip of
  -4 dB at 31-37 s (verse 1 entry, estimate). Every section is "loud" to the writer.
- Section and line timings: NOT measured yet (the writer's alignment step does it on the vocal
  stem when the script is started in the tab).
- Structure (from the lyrics): Intro (Hey x4) / Verse 1 / Pre / Chorus / Verse 2 / Pre /
  Chorus / Guitar solo / Bridge (Hey living people) / Final chorus / Outro (Hey x4)
  / instrumental outro, hard stop.

## 3. Story outline (APPROVED by the user 2026-11-13)

User 2026-11-13: recurring characters are fine where they work better (the terrified guy in verse
2, the pizza-box man returning in the bridge stay). Section times below are the measured ones
after the bridge fix (s.7). "Key change" in the final chorus was an assumption, never measured:
removed.

| Section | Beat of the story | Comedy / action / sexy |
|---|---|---|
| Intro | A modern cemetery at night, fog, a sodium streetlight past the railings, a car parked at the gate. On each "Hey!" a hand punches up out of a different grave. | The couple in the parked car see the first hand. |
| Verse 1 | She claws out of a fresh grave and sings to camera; cutaways to what the lyric names (dirt, the moon, the clay). | She shakes dirt out of her hair and fixes smeared lipstick in a cracked compact mirror. |
| Pre-chorus 1 | "Lock your doors, board up the glass": the living panic in their homes. | A man nails a pizza box over his window; doorbell-camera fisheye shot of the dead on a porch. |
| Chorus 1 | The party starts: the dead rise all over the cemetery and dance; she leads, belting. | On "Hey! Hey!" a dancing zombie's jaw drops off; another's arm comes off and he waves it. |
| Verse 2 | "Queen with a crown of bone ... dance with me, darling": she flirts with a frozen, terrified living guy. | On "a little piece of your head" she eyes his head and licks her lips. |
| Pre-chorus 2 | The dead press against windows; the living pile furniture against doors. | A zombie licks the window glass. |
| Chorus 2 | The party grows: a hearse with its back open as a sound system, car headlights and phone torches as stage lights. | Zombies crowd-surf a coffin. |
| Guitar solo | Montage: the horde spills into the street, police lights, dancing between parked cars. | A zombie stuck in a cat flap; a zombie air-guitars a shovel. |
| Bridge | "Hey living people ... come and join the crowd": the living come out in pyjamas and join. | The pizza-box man joins the conga line. |
| Final chorus | Everyone, living and dead, dancing in the cemetery; she on a mausoleum roof belting. | Biggest shot of the video. |
| Outro | Dawn: the dead flop back into their graves. | She winks and pulls the dirt over herself like a duvet; hard stop on a slammed coffin lid. |

## 4. Look and wardrobe (DRAFT)

- Look: photoreal 35mm, high contrast, horror-comedy colour: cold green-blue moonlight and fog
  against warm practical lights (sodium streetlights, car headlights, police red-blue, string
  lights, phone torches). Present day, no period dress.
- Her (the only referenced identity): a zombie queen who is still glamorous. Crown of bone
  (lyric). A black satin slip dress torn at the hem, ripped fishnet tights, bare muddy feet or
  boots (decide), pale skin with a faint grey-green cast, dark smudged eye makeup, dark smeared
  lipstick, grave dirt on her shoulders and in her hair. Sexy first draft per the house rule;
  GPT filter rules apply (one reveal per image).
- Wardrobe sentence (verbatim in the body prompt, the face prompt's neckline part and the
  writer's `refs.body`), v3 = what the two passed pictures actually show: "a short black satin
  dress with a draped cowl neckline, its one thin strap slipped off her left shoulder and both
  shoulders bare, the hem frayed and torn at mid-thigh and smudged with grave dirt, black fishnet
  tights, scuffed black lace-up ankle boots, and a crown of small pale bones bound with black
  wire on her head". (v1 "short ... slip dress ... torn ragged ... ripped fishnet" was refused by
  the GPT filter on the body picture; v2 "hem just above the knee" passed but GPT drew it short.) Makeup: smudged black eyeliner, dark plum lipstick smeared at one
  corner, a faint grey pallor with soft shadows under the cheekbones, a smudge of earth on one
  cheek. Key light: cold blue-green from the left, warm amber rim from the right.
- The other dead: anonymous, comic, varied (a man in a burial suit, a bride, a jogger in
  running gear), decayed but cartoon-safe, not gore.

## 5. What the pipeline cannot do yet (the careful-planning part)

1. **Cutting speed.** The writer's fastest pace is 5-9 s scenes for a loud section, and cuts
   fall only on section edges and half a second before sung lines. This song wants 1-2 bar shots
   (1.6-3.2 s) in the choruses and the solo. Needs a general, opt-in "cut to the bar grid"
   option: beat tracking, cut points on bar lines, a short-shot palette for loud sections.
2. **Crowds dancing.** The rulebook allows extras only small ordinary actions; dancing is a large
   body action and is untested on H3, alone or in a crowd.
3. **Physical gags** (a jaw or arm falling off, stumbling, a cat flap). Untested.
4. **Her dancing while she sings.** Untested; lip-sync with body motion may break.
5. **Action by cutting.** One single-action shot per cut (cut-on-action montage) plays to what
   H3 does well; needs the short shots from item 1.

### Research pass (2026-11-13, web sources read by a research agent; nothing reproduced by us)

- H3 official prompt guides: no subject-motion intensity control (amplitude/speed is camera only);
  multi-person = speaker IDs for people who vocalise, crowds not covered; falls and state changes
  are in scope (their own example: a glass tips, falls, hits the floor). [P]
- The ref guide lets a subject's motion come from `<Video 1>` (we only used it for camera). [P,
  untested here] H3 is reported weak on large-scale action vs Seedance. [C]
- Pose/video-driven lanes said to fit 24 GB: Wan-Animate-2 (Aug 2026, many-to-many, int8 peak
  21.4 GB at 480x832 81 frames on a 4090, core nodes, Apache-2.0) [C]; SteadyDancer (multi-person
  example, Kijai wrapper, GGUF, Apache-2.0) [P]; SCAIL-2 [C]. UniAnimate-DiT, VACE pose and
  MimicMotion lose identity with 2+ people [P, MultiAnimate paper]. Not checked: what is installed
  on the box. A skeleton cannot express a falling jaw or arm, so gags are not a pose-lane job. [H]
- Her dancing with lip-sync: untested everywhere. Options: ref2va + dance prose; ref2va + a dance
  clip as `<Video 1>`; Wan2.2-S2V with a pose video [P]; dance render then InfiniteTalk v2v [P,
  expect damped dance]. Safe default: she dances at medium/wide NOT singing, intercut with her
  proven MCU/CU singing. [H]
- Cloud escape hatch: Seedance 2.x best on choreography [C]. Web reports say it blocks realistic
  face references [C], BUT the user's own use (2026-11-08) is that Seedance via Yapper accepted
  our generated characters and stills [U]; treat a refusal as per route. Never music or
  lip-sync. No model has a source on slapstick.
- Editing: bar = 38 frames at 24 fps; cut on action (start the action in one shot, finish it in
  the next), hold 6-12 frames after the key moment [C]. Proposed grid [H]: one bar per action
  shot, half bar (0.8 s) for impacts only, two bars per gag (setup, payoff + hold).

### Lanes per shot type (proposal, each to be settled by a test)

1. Her singing: ref2va MCU/CU, as proven.
2. Her dancing: medium/wide, not lip-synced, intercut with 1 (lane settled by T4/T5).
3. Crowds dancing: H3 i2v from an opening still if T1 passes, else a pose lane (T2).
4. Gags: H3 i2v, free prose vs a matched after-still as last frame (T3).
5. Living people (boarding up, pile-ups): H3 i2v from opening stills (proven method).
6. Cutting: bar-grid montage of single-action shots (needs the writer option in item 1 above).

## 6. Test renders (proposal; every GPU run needs the user's go-ahead)

One variable each; a moving shot that fails gets 3 seeds before it is judged.

- **T0, no GPU:** re-cut existing takes at half-bar, one-bar and two-bar shot lengths over this
  song; the user picks the shortest cut that still reads. Sets the writer's short-shot palette.
- **T1, H3 i2v crowd:** a GPT opening still of ~8 zombies in the cemetery; the only change from
  the proven people-plate recipe is the action class (each dances in place, staggered). Real speed
  and separate bodies = H3 can do crowds; melting or unison = pose lane.
- **T2, pose lane:** Wan-Animate-2 (if installed or installable) with one zombie still + a short
  licensed dance clip, then 3 targets. Measures VRAM, time, many-to-many hold.
- **T3, gag:** "his arm falls off" on H3 i2v, free prose vs the same with a matched after-still
  as the last frame.
- **T4, her dancing while singing:** the proven MCU-singing setup at medium framing, only change =
  dance prose. mv_take_check + the user watches; lip or face failure -> T5.
- **T5:** T4 with a dance clip as `<Video 1>` motion source, vs Wan S2V with a pose video.

## 7. Log

- 2026-11-13 T0 built (no GPU): `.mvwork/nsd_t0/build.py`. Song window 86.80-111.60 s
  (16 bars from beat 224, loud; section not measured), Second Skin assembled scene clips as
  stand-in footage (slow gothic material, so this judges shot length, not action), same shot
  order in all three. Half bar = 32 shots, one bar = 16, two bars = 8; every cut checked to land
  on its beat frame (scene detection, no misses). Copies in Downloads/Nobody Stays Dead tests.
  Waiting on the user's pick.
- 2026-11-13 T0 verdict [U]: one bar (1.58 s) is the SHORTEST shot, not the standard length:
  "we shouldn't always cut at that, that's just the shorter end". Writer consequence: cuts fall
  on bar lines, shot lengths vary from 1 bar upward (1, 2, 3, 4 bars...), with the short end
  kept for the loudest, busiest moments; half-bar cuts are out.
- 2026-11-13 stills back from GPT (copies in `.mvwork/nsd_stills/`): face and body references
  (agree on crown, strap off her left shoulder, earth smudge on her left cheek, lipstick), the
  cemetery location still, the eight-zombie crowd opening still (same place, eight separate
  bodies), gag before (arms up) and after (right arm off, held up in his left hand). GPT refused
  twice to draw the detached arm bare (jacket sleeve and shirt sleeve both refused), so the
  detached arm keeps its jacket sleeve; it reads as the sleeve tearing off with it.
- 2026-11-13 test setup: vocal stem made on the box from the master; alignment on the stem: cover
  0.77 (247/320 words). [M] The BRIDGE lines were not placed: 136.6-174.4 s comes out as one
  instrumental (solo + bridge), so the real script needs the bridge fixed first. Chorus 1 =
  61.84-81.20. The six stills are filed in the library. Test script builder
  `.mvwork/nsd_tests/build_script.py` (scene 0 = T4 at 61.84-67.62, scene 1 = T1 at 67.62-76.74,
  scene 2 = T3 at 140.0-146.0; clip text by the writer's own pass 2). BLOCKED: the app's claude
  CLI login expired ("OAuth session expired and could not be refreshed"), so the writer cannot
  run; the user must log in again. No render fired.
- 2026-11-13 18:40 test script `nobody_stays_dead_tests_20261113_183437.json` written (writer pass 2,
  no problems flagged). (Correction: the earlier "CLI login expired" block was the test runner
  missing `.env.local`, which run.sh loads; the app's writer was not blocked.) Fired 10 renders,
  base 20 steps, 1344x768, "comfy kitchen attention", same settings the tab sends: T1 seeds
  1101-1103; T3 text only seeds 1311/1312 and with the after picture as end still, same seeds;
  T4 seeds 1301-1303 (face + body + cemetery). Job list `.mvwork/nsd_tests/jobs.txt`. Confirmed
  running on the box (T1 1101 first, 9 queued).
- 2026-11-13 bar-grid writer option BUILT (opt-in, default off; plan s.5 item 1). Tab: "Write a new
  script" > "cut on the bar grid", shortest shot 1-4 bars (default 1), optional tempo hint; API
  `bar_grid: true | {min_bars, beats_per_bar, bpm}` on /api/mv/write, preview POST /api/mv/bar_grid.
  `backend/mv_beats.py` measures the grid per track: beat tracker, then the tempo that most beats
  agree with (circular coherence, robust to tracker slips); a drifting tempo gets a curve fit.
  Downbeat = the beat phase with the most harmony change + kick. The writer then lists the bar
  lines under each section, swaps the 5 s floor / 5-9 s pace / line-cut rule for the bar rules
  (varied whole bars, one bar = the short end, no more than three equal lengths in a row, main
  action in the first bar), snaps every boundary to a bar line and refuses scenes under the
  minimum; split and move-cut stay on the grid. [M] Grid on three songs: Nobody Stays Dead 155.00
  constant (6 ms); Second Skin 148.00 constant (13 ms; the tracker slips an extra beat near 67 s
  and 222 s, which beat counting would have turned into a bar-phase flip); Never Yours drifts
  129-132 bpm, curve fit 13 ms. Downbeats are measured, NOT verified by ear. Offline tests pass
  (brief, snapping, minimum, split, move, markdown; the off-grid brief is byte-identical). NOT yet
  run end to end: the backend restart that loads the endpoint waits for the test renders.
- 2026-11-13 21:15 all 10 test renders done. Take check [M]: every take holds the song (floor
  0.99-1.00, lag 0), no opening jump flagged. Copied as "Downloads/Nobody Stays Dead tests/NSD
  <test> seed <n>.mp4". From the contact sheets only (motion NOT judged; the user watches):
  - T1 crowd (3 seeds): the room and all eight undead hold; the bride spins in every seed and the
    old man bobs, but the other six mostly hold their opening poses with small sway, so on stills
    it reads as posing more than dancing. 1102/1103 grow fog on the left late in the take.
  - T3 arm gag (4 takes): the arm comes off and is waved in ALL FOUR, text only included, so the
    after picture is not needed for the gag to happen. In every take the gag starts late (about
    4.5 s into the 6.6 s take) and the frame where the arm leaves looks smeared (motion blur or a
    ghosted arm; to judge by watching).
  - T4 she sings (3 seeds): face, crown, slipped strap, satin dress and fishnets hold in all three;
    she sings with her hands on her hips in every seed; the written hip and shoulder roll shows only
    as a small sway on the sheets.
- 2026-11-13 bridge alignment FIXED (general, backend/lyricalign.py). Cause [M]: Whisper heard
  nothing over 136-169 s, then repeated verse 2 ("I don't need sleep...") with every word stamped
  at 174.9 s, so the bridge had nothing to match and the next line's stamps were 1.3 s late. Fix:
  a second pass for any lyric block with under half its sung words placed: that window alone
  (from the last placed line before it to the first placed line after it) is transcribed again
  with earlier text not carried forward, its words replace the first pass's in the window, and
  the song is matched again. Bracketed-only blocks (the intro "Hey!"s) are left out as before.
  Result on the stem [M]: cover 0.77 -> 0.87. Solo = instrumental 136.60-157.06; BRIDGE 157.06-
  170.58 (lines 157.56 / 160.44 / 163.48 / 166.58); a short instrumental 170.58-173.20 (the
  "Hey!" chant, unplaced); Final Chorus from 173.20 (first line 173.70, was 174.94). Chorus 2's
  last line moved 133.90 -> 134.66 (chorus 1's same line sits 14.90 s after its first: 134.60).
  Dry run over the six other songs with saved scripts: none triggers a second pass (unchanged).
  The backend needs a restart before the app's writer uses it.
- 2026-11-13 user verdicts: T4 her dancing [U] "actually good" (my sheet-based "small sway" read
  was wrong; the user's watch is ground truth). Rule [U]: when she dances in more than one scene,
  each scene gives her a DIFFERENT pose and move, never the same dance twice. Recurring
  characters OK [U]. Outline APPROVED [U] (s.3 as it stands).
