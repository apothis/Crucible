# Music Video tab: UI design (proposal, 2026-10-30)

## Status

- 2026-10-30 PHASE 1 BUILT (web/src/MusicVideo.tsx, backend/mv_takes.py, routes in app.py:
  GET /api/mv/scripts, GET /api/mv/script/{name}, GET /api/mv/takes, POST /api/mv/take_check,
  GET /api/mv/take_sheet/{id}, POST /api/mv/lock_take; takes now carry a `title` param so the
  Library groups them per scene). Browser-verified on the built dist (:8000): script list, the
  storyboard with stills and status, scene drawer, dry run through the real gate (ref2va, 1 clip,
  1344x768, seed pinned, no flags; Render enabled only then), take check from the tab (floor
  1.00, opening 2.1x, contact sheet), lock and unlock round trip written into the script file.
  NOT exercised: Render on the box (GPU, user-fired), still paste/upload (no new still to file).
  Bug found and fixed in the check: a loader effect that depended on a useDrafts setter re-ran
  every render (endless /api/mv/script requests until the browser ran out of sockets).
- 2026-10-30 PHASE 2 BUILT: Assemble stage in the tab (take per scene: lock, else opt-in newest
  take, else missing; output size; intro ident on/off with title, sub line and edge, preview;
  end card on/off with 10/15/20 s and the More-from text, live plate preview; async job with
  progress; result player; filed in the Library under Music videos). Backend: mv_finish.py
  (ident_render cached by props + Ident.tsx hash, ~5 s warm on the M4; finish_cut = one ffmpeg
  pass: ident concat + end plate overlay on the last N s), POST /api/mv/ident, GET
  /api/mv/ident_file/{hash}, /api/mv/assemble gained intro, end_card, file, async (+ GET
  /api/mv/assemble_result/{id}); the synchronous call and the curl scripts are unchanged.
  Verified: ident route (render + cache hit), finish_cut on a real take (12.46 s = 3.05 ident +
  9.42 take; plate over the last 3 s read on a frame tile), and from the tab a partial cut of the
  two scenes with takes (18.4 s, filed). NOT run: a full cut with intro + end card on a complete
  script (needs every scene to have a take; the newest SSBS script has 24 of 36 locked).
  Known: a partial cut is filed in the Library too (delete it there if it is only a check).
- 2026-10-30 CORRECTION + lineage: the "12 scenes without a take" claim was wrong. The other
  session had cut the FULL 36-scene video three times today (mv_assembly 091656, 101000, 131538,
  252.44 s each) from an explicit take map spanning six script generations; the tab only matched
  takes filed under the exact current script name. Fixed: /api/mv/takes matches takes rendered
  under any ANCESTOR script (names chain by suffix) whose song window equals the scene's, tagged
  from_script ("earlier script" pill), and always lists a scene's locked take by id (a lock may
  trim the window). Every assembly now writes assembly.json (request + result incl. the take map);
  GET /api/mv/assemblies?script lists a song's earlier cuts, served by GET /api/mv/cut/{stamp}/{file};
  the Assemble stage shows them with play and "use its takes" (the plan then follows that cut).
  Sidecars were written for today's three full cuts from the other session's request/result
  files. Verified in the browser on the newest script: 36 of 36 scenes with takes (52 takes), 21
  locked, full cut offered; "use its takes" on the 13:15 cut maps all 36. The full assemble with
  ident + end card was NOT clicked (a few minutes of Mac encoding; the user's call).


User ask (2026-10-30): "a UI for this entire video process, including adding the intro, and
adding the YouTube cards feature at the end, same way the YouTube tab does that part. Also
showing each scene from the video, its still etc, all needs to be user friendly ... including
the upscaling work."

## 1. State of play (facts, 2026-10-30)

- There is NO music-video UI. The June MV Studio (block timeline, character tab, per-block
  upscale) was deleted in the H3 rebuild on 2026-10-27 (commit eef724a). The "Video" group in
  web/src/App.tsx:56 holds only the YouTube tab. The plan's own UI section
  (docs/MV_H3_REBUILD_PLAN.md s.3.5) specifies one minimal "Music Video" tab after gate G3 and
  says "no timeline editor". The planned web/src/MusicVideo.tsx was never written.
- The whole H3 pipeline is backend routes under /api/mv/* (backend/app.py 1162-2000), driven
  only by curl scripts in .mvwork/transfer/ssbs/renders/. Every step a user needs exists as a
  route: write, rewrite_scene, split_scene, move_cut, refresh_facts, opening_still prompt,
  scene_still upload, render_scene (dry_run = the gate), render_script (batch from a dry run),
  assemble (partial or full), locations, stems.
- Missing routes: list saved scripts, read one script, list the takes of a scene, run the take
  check (mv_take_check.py) on demand, lock/unlock a take, render the ident, an MV end card.
- Takes land in the library as "Video clips" with params {script, scene_index, scene_title,
  first_frame_id} (app.py 1306-1314) but no title, so each is an anonymous card today. Opening
  stills are filed as videostill kind opening_still with {scene, script}.
- Locks: a passed take is recorded per scene as scene.locked = {job_id, take_start, script} and
  survives write/batch/assemble (app.py 1316-1350, 1493-1494, 1577). There is no route that sets
  it; the user has been passing lock lists by hand.
- YouTube tab end card: style "card" only; checkbox + 10/15/20 s + "More from" text + live
  preview (web/src/YouTube.tsx 885-920); wordmark.endcard_plate draws the plate; musicvideo
  .card_video overlays it for the last N seconds of the audio video (not a concat). The YouTube
  Studio "two videos + subscribe" template is laid over it by the user in Studio.
- Upscale: FlashVSR route /api/video/flashvsr (auto-chunks >200 frames), UI only in the living
  cover pick. New H3 latent upscale stage in mv_h3.py (upscale_width/height, single clip, commit
  197c1bc, from the other session), no UI. SeedVR2 is gone.
- Ident: Remotion composition tools/loosh_remotion (docs/APOTHEON_IDENT.md), 3.0 s, title as a
  prop, rendered on the Mac by `npx remotion render` (no GPU). No backend route yet.

## 2. Principles

1. One tab, "Music Video", in the Video group next to YouTube. State in useDrafts("mv") so the
   top-bar Save/Open carries it (lesson from the June build: no parallel save path).
2. The backend stays the source of truth: scripts are the JSON files in .mvwork/mv_scripts,
   takes are library rows, locks live in the script. The UI reads and calls; it holds only
   selection and form state.
3. The GPU fires only on a user click, and only from a dry run (the existing gate). The tab
   shows the dry run's flags and makes accept_flags an explicit tick.
4. Storyboard, not timeline. Scenes are cards in song order with their still, window, framing,
   performance and status. No draggable ruler (the plan's "no timeline editor" stands; the
   writer's cut points and split/move_cut are the editing tools).
5. Every scene card answers five questions at a glance: do I have a still, has it been dry-run,
   has it rendered, did the take pass, is it locked.
6. Nothing destructive from the tab without a confirm; rewrites save new scripts as today.

## 3. The tab, stage by stage

The tab is one page with a stage rail at the top (Song, Script, Scenes, Assemble, Publish).
Stages are not gates: the user can jump, and each shows what it is waiting on.

### 3.1 Song
- Pick the song (library audio picker, as the YouTube tab does). Shows duration, style line,
  whether the vocal stem exists (button "Make stem" -> /api/mv/stems).
- Script list for this song (new GET /api/mv/scripts?audio_id): name, written, scene count,
  clip count, locks count, last used. Open one, or "Write new script" -> /api/mv/write with the
  concept/look/wardrobe fields (the writer's inputs) and the lead's refs (face/body still pickers,
  defaulting to the last used).

### 3.2 Script
- The script header: title, concept, look, wardrobe, arc; the writer's design recipe output
  read-only; refresh facts button (/api/mv/refresh_facts).
- A scene strip: one chip per scene over a song ruler (read-only, proportional), coloured by
  status. Click selects the scene in 3.3.

### 3.3 Scenes (the storyboard)
Grid of scene cards in song order. Each card:
- still thumbnail (or an empty slot with "paste or drop the GPT still"; paste = clipboard image
  -> /api/mv/scene_still; "Copy still prompt" copies scene.opening_still, the paste-ready text)
- title, window (start-end, seconds), framing, performance, lip-sync badge, clip count
- status pills: still / dry run (ok or N flags) / rendering (job) / takes (n) / PASSED / LOCKED
- problems from check_scene (scene_problems) as a red count with a tooltip

Selecting a card opens the scene drawer (right side):
- Setting, action, camera, lighting, mood, others, why: editable fields; "Rewrite scene" sends
  the changed fields to /api/mv/rewrite_scene (empty set = refresh clip text only). The result is
  a new script; the tab switches to it and says so.
- Split at (seconds) / Move cut to (seconds) with the writer's cut points listed as choices
  (/api/mv/split_scene, /api/mv/move_cut).
- Clip prompts (pass 2) read-only, one block per clip, with the lines each clip covers.
- Render panel: seed, turbo, size (default the proven 1344x768), lane (ref2va / i2v for
  environment scenes), distant, anchor off by default, H3 latent upscale target (off by default,
  single-clip scenes only, from commit 197c1bc). "Dry run" fills a flags box; "Render" is enabled
  only after a dry run with unchanged prompts and either no flags or accept_flags ticked. Hunt =
  a seed list that queues N renders.
- Takes list: every library clip with this script+scene (new GET /api/mv/takes). Per take:
  thumbnail, seed, size, length, floor score (envelope check vs the song window; clean 0.87-1.00),
  opening_jump flag and the contact sheet from mv_take_check.py (new POST /api/mv/take_check, runs
  on demand on the Mac, no GPU), play inline, "Pass and lock" / "Unlock" (new POST
  /api/mv/lock_take writes scene.locked), "Upscale (FlashVSR 2x)" -> /api/video/flashvsr on that
  take, result filed and linked as the take's upscaled_id.

### 3.4 Assemble
- Shows the scene list with the take that will be used per scene: locked take, else the newest
  passed take, else "missing". Partial assembly allowed (existing partial mode) with the missing
  scenes listed.
- Options: output size (1344x768 render / 1080p / 4K), use upscaled takes when present,
  intro ident ON/OFF (title defaults to the song title, sub line "Official Video"; rendered via a
  new POST /api/mv/ident that runs Remotion on the Mac and caches per title), end card ON/OFF with
  the same three controls as the YouTube tab (duration 10/15/20 s, "More from" text, live preview
  from /api/youtube/endcard_preview), transitions = hard cuts (as today).
- "Assemble" -> /api/mv/assemble extended with intro + end card; output filed in the library under
  "Music videos" with the script name, and shown inline with its length.

### 3.5 Publish
- The assembled video: thumbnail candidates (/api/youtube/thumbnail from chosen frames or a
  still), metadata (title, description, tags via /api/youtube/metadata), the wordmark stamp if
  wanted, download. No upload automation (the user uploads in Studio and lays the card template).

## 4. Backend additions (small, all additive)

| Route | Does |
|---|---|
| GET /api/mv/scripts?audio_id | list .mvwork/mv_scripts JSONs for a song (name, counts, locks, mtime) |
| GET /api/mv/script/{name} | the script JSON (scenes with clips, locks, still ids) |
| GET /api/mv/takes?script&scene | library clips whose params match, with job state, upscaled_id, last check |
| POST /api/mv/take_check {job_id, script, scene} | envelope floor + opening_jump + contact sheet; cached next to the take |
| POST /api/mv/lock_take {script, scene, job_id, start?, end?} / unlock | writes scene.locked using _mv_resolve_locks; saves the script in place (locks are state, not a rewrite) |
| POST /api/mv/ident {title, sub, edge} | `npx remotion render` in tools/loosh_remotion, cached in .mvwork/ident/cache/<hash>.mp4 |
| /api/mv/assemble + {intro: {title, sub, edge}, end_card: {secs, text}} | ident concat at the head; the end plate overlaid on the last N s of the video (same mechanism as card_video) |
| render_scene params | upscale_width/height already exist; takes get a `title` param (scene title) so the library groups them |

## 5. Decisions for the user before building

1. End card on an MV: overlay the plate on the last 10-20 s of the video itself (the YouTube tab
   way; the song is still playing under it, the picture dims) or append a card after the last
   frame with the song's tail under it? Recommendation: overlay, same as the YouTube tab, since
   the Studio template needs the video to still be running.
2. Intro: the ident is 3.0 s with its own sound; the video's song starts after it, so every
   lyric timestamp shifts by 3 s. Fine for YouTube; say if chapter timestamps matter.
3. Upscale placement: per passed take before assembly (FlashVSR 2x on 1344x768) with assembly
   downsizing to 1080p/4K, or one pass on the assembled master? Recommendation: per take, so a
   re-rendered scene does not force a full re-upscale.
4. Scope of phase 1 (below).

## 6. Phases (each ends browser-verified on the Mac, GPU only on the user's clicks)

- P1 Storyboard: tab shell, Song (pick + script list), Scenes grid with stills, still paste,
  copy-prompt, scene drawer with clip prompts, dry run and Render, takes list with floor score,
  take check and lock. Backend: scripts list/get, takes, take_check, lock_take, take title param.
- P2 Assemble and publish: intro ident, end card, assemble with locked takes, library filing,
  thumbnail and metadata hooks. Backend: ident route, assemble extensions.
- P3 Editing and upscale: rewrite/split/move_cut from the drawer, write a new script from the
  tab, FlashVSR per take, H3 latent upscale option in the render panel.
