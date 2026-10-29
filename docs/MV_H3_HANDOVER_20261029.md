# Handover: She Sings Before the Storm music video on MiniMax H3 (2026-10-29, evening)

You are taking over an in-progress music video build. Read this, then read docs/MV_H3_TECHNIQUES.md (the rulebook) and the tail of docs/MV_H3_REBUILD_PLAN.md (the per-video log, sections 20.38 to 20.44 cover today). Do not fire any render until the user says go.

## What this is
- A local app (repo /Volumes/SSD1/Code/MusicGen). Mac runs the FastAPI backend on 127.0.0.1:8000 (started with ./run.sh). A Windows RTX 3090 box at 192.168.1.201 runs ComfyUI on :8188 and a file API on :5080. Video renders go to the box only; no ML compute on the Mac.
- The video engine is MiniMax Hailuo 3 (H3) via ComfyUI, reference-to-video lane: she is defined by two reference images (face 2c1c49e7c1a34a47878c4d416d6fb009, body/wardrobe cd98a0e41c1a4579944e5aff23239e25) plus a per-location still. The song audio is fed in as a masked latent so lip-sync and timing follow the track. One clip = up to 362 frames at 24 fps; longer scenes chain clips with 39 context frames. Renders land in library/<job_id>.mp4.
- Claude writes the script from the song (backend/mv_writer.py: pass 1 = scenes, pass 2 = per-clip H3 prompts). The user generates opening stills in OpenAI from prompts Claude gives IN CHAT, referring to the attached reference pictures. Never hand the user a prompt in a text file.
- Song: library/2306adc5a44a43afb1e4743b86bd98b9.wav. Lead: one woman, Selene (long dark wavy hair, dark green laced wool bodice over a white linen shift off one shoulder, charcoal wool skirt, dark grey shawl at her elbows, barefoot).

## Where things are
- Working script name is in .mvwork/transfer/ssbs/rewrite/final_script_name.txt; scripts live in .mvwork/mv_scripts/<name>.json and .md. Current: ..._mc1_20261029_154720_rw0_20261029_183530_sp0_20261029_192903.json (27 scenes, 31 clips).
- Takes: .mvwork/transfer/ssbs/renders/*.mp4, descriptively named. Frame sheets in renders/split_cmp/. Last night's full cut (25 scenes, 252 s) is renders/ssbs_FIRST_FULL_CUT_25scenes_252s_1344x768_20261029.mp4; the user said considerable chunks of it were OK.
- Fire scripts (bash + curl, mirror these, do not invent params): renders/fire_sp0b_scene0_scene1.sh is the latest; it POSTs /api/mv/render_scene, waits with renders/watch_file.sh, floor-checks with .mvwork/g1_refs/envelope_check.py <clip> <song.wav> <window start> (clean takes 0.87-1.00), files a copy, makes a sheet.
- Routes: POST /api/mv/render_scene {script, scene, ref_ids, seed 20261027, turbo, attention "comfy kitchen attention", width 1344, height 768, dry_run} (dry_run true returns the prompts and graph facts without firing); /api/mv/rewrite_scene {script, scene, set{fields}} re-runs pass 2 on one scene; /api/mv/split_scene {script, scene, at, first{}, second{}}; /api/mv/move_cut {script, scene, to, first{}, second{}}; /api/mv/scene_still (multipart: file, script, scene, label) files an opening still on a scene. Every rewrite saves a NEW script next to the old one.
- Dry-run JSON: 'clips' is an int; the prompts are in 'prompts'. Clip text in the script is a dict (summary, detailed_description, ...).

## The user's callouts on last night's cut (ground truth)
1. Scene 0 (the opening, extreme wide cliff road, 0-18.36 s): a second copy of her appears mid-take.
2. Scene 1 (chapel wall): a fade/dissolve mid-take, "could be artistic but not ideal".
3. Close-ups under singing must lip-sync, and the first sung line should land on a shot set up for lip-sync.

## What was done today and what it proved
- Rules 3 landed in the writer and are fine: a medium close-up or closer under audible words is always sing_to_camera; the scene holding the first sung line is a sing_to_camera MCU set up ahead of it (FIRST_LINE_LEAD); check_scene flags violations. The first-line setup take (scene "The voice at evening", 35.22-44.76 s, file ..._FIRSTLINE_SETUP_...) is GOOD per the user.
- A "walking direction relative to the camera" rule was added. It did NOT fix the doubling: the re-render (..._CAMERA_RELATIVE_WALK_...) doubled her again at 9 s. One seed only; cause unknown. Do not claim a cause.
- The chapel wall scene now ends at 35.22 (its dissolve at 18.1 s into the take falls outside the trimmed 16.86 s window, so the existing take covers it).
- Scene 0 was then split at 9 s into a wide by the chapel (0-9) and a medium wide on the ridge from low on the road (9-18.36), one clip each, because the user found her too far away for too long. Takes: ..._SPLIT9_... (two files). The ridge shot is clean and the user has not objected to it. The wide: one woman, but she walks down the road TOWARD the camera and doubles in size (the road runs from the chapel to the lens, so "away from the chapel" was the approach walk), the sky rendered as a storm bank while the ridge shot is a pale calm evening, and HER OUTFIT DIFFERS between the two takes (user's observation). The user's verdict on the day: "we're going backwards".
- Also known: on the reference lane the model sometimes invents a second building next to the chapel; the frame-0 anchor guide (opening still as first frame) was judged WORSE and is OFF; the I2V lane (still as frame 0) is used only for environment B-roll.

## Hard rules from the user (break none)
- Never fire a GPU/box render without an explicit go. Propose, then stop. Read-only checks are fine.
- One variable per render. Report only what is verified; mark hypotheses as hypotheses; the user's eyes and ears are ground truth; a frame sheet is not a verdict.
- Fix the shots the user called out against the mostly-good render. Do not invent new problems, do not add a writer rule per observation, do not re-split and re-still scenes the user did not complain about.
- Temp files under /Volumes/SSD1/Code/MusicGen/.mvwork, never /tmp. No em dashes in anything stored. Commits carry no AI attribution. Check :8188/queue before restarting the backend.
- Still prompts in chat, paste-ready, referring to the attached reference pictures.

## Open questions for the user before doing anything
- Which of the current scene 0 takes, if any, to keep (the SPLIT9 ridge shot looks keepable).
- Whether to make the opening a standing shot (no walk) to remove the approach.
- How to stop the outfit and sky drifting between adjacent takes: untested options are a shared opening still as a reference for both, or a wardrobe line tightened in the subject definition. Both are hypotheses.
