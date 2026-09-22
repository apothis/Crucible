# Loosh Woggins photo-story lyric video build

Copied from .mvwork after the Gym Membership build (2026-09-22). Paths inside the
scripts point at .mvwork; copy them back there (or edit the paths) to run.

Per song:
1. Stills from ChatGPT into .mvwork/loosh_<song>_stills/, named by shot number.
2. Lyric timing: backend/lyricalign.py align_song(wav, blocks) -> measured line times
   (cover >= 0.5 per line is the gate). Even splits were 4-12 s off on Gym Membership.
3. loosh_gym_cues.csv: sections -> still + camera move. Drops = sub-bass plateaus
   (bass-energy analysis), choruses = sung lines.
4. loosh_gym_captions.csv generated from the alignment (all lines, chorus = big).
5. loosh_gym_overlays.py renders captions + title card + bug + end card as PNGs
   (ffmpeg here has no drawtext) and the thumbnail.
6. loosh_gym_build.sh sections, then assemble. loosh_gym_short.sh for the vertical cut.

Gotchas that cost time:
- showwaves MUST use draw=full; the default draw=scale wrecks the red channel and the
  fill comes out teal under an ivory outline.
- ffmpeg inside a `while read` loop needs -nostdin or it eats the next csv line.
- macOS bash 3.2: no mapfile. zsh: $VAR:s and $VAR:r are history modifiers, so brace
  every variable that is followed by a colon in a filter string.
- Font: fonts/Rye-Regular.ttf (OFL, Google Fonts) lives in .mvwork/fonts.
