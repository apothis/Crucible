# Apotheon channel ident

Status 2026-10-30: picture mocked and accepted as a direction ("promising so far"); sound and
edge artwork under review by the user.

## Picture

Remotion composition `Ident` in tools/loosh_remotion/src/Ident.tsx (registered in Root.tsx).
1920x1080, 24 fps, 72 frames (3.0 s). Brand = the "modern" look in backend/wordmark.py:
Josefin Sans 300 tracked caps, flat ivory (236,231,220), ember accent (220,120,60).

- 0.00-0.50 s black, grain, low ember haze, embers drifting up (rumble + riser under it)
- 0.50 s (frame 12) the hit: 2-frame flash, wordmark slams in 108% -> 100%, red/cyan split
  collapsing over 8 frames, 4-frame shake, ember hairline draws out
- 0.90-2.50 s hold: one specular sheen pass, song title fades up, OFFICIAL VIDEO sub-line,
  word breathes to 101.5%
- 2.67 s hard cut to black, no fade (the video butts straight on)

Props: title, sub, audio (path under public/), edge (none | hairline | filigree | burnt).
The filigree option is a placeholder shape only; a real one needs an engraved ornament asset.

Render:

    cd tools/loosh_remotion && npx remotion render src/index.ts Ident out.mp4 \
      --props='{"title":"Song Title","sub":"Official Video","audio":"audio/ident_mix_A_whoosh_deep_impact_1143.wav","edge":"none"}'

## Sound

Suno Sounds was tried and dropped (user 2026-10-30): it does not follow the length request.
Stock instead: Mixkit, Sound Effects Free License (fetched 2026-10-30 from
mixkit.co/license/modal/sfxFree/, saved in .mvwork/ident/sfx_research/mixkit_sfx_license.txt):
usable in YouTube videos, music videos and commercial projects, no attribution, no sign-up;
not redistributable on its own. Files are plain GETs from assets.mixkit.co/active_storage/sfx/<id>/<id>.wav

Bed layers (tools/loosh_remotion/scripts/build_ident_mix.sh <label> <hit> <transient s> [gain dB]):
- hit: candidate aligned so its transient lands at 0.5 s (measured peak time, see table)
- rumble: Mixkit 1177 "Deep cinematic wind hum", 0.6 s slice, fade in, -6 dB
- riser: Mixkit 790 "Cinematic trailer riser" (2.58 s, peaks at its end) trimmed so it ends on the hit, -4 dB
- embers: Mixkit 1330 "Campfire crackles", from 0.6 s, -20 dB
- 150 ms fade ending at the 2.67 s cut, limiter at -1 dB

Hit candidates measured (mono, 48 k):

| id | title | file length | transient at | tail to -40 dB | sub (<120 Hz) share |
|---|---|---|---|---|---|
| 1143 | Cinematic whoosh deep impact | 4.08 s | 0.563 s | 1.73 s | 81% |
| 2908 | Movie trailer epic impact | 4.87 s | 0.925 s | 2.94 s | 84% |
| 1687 | Dramatic metal explosion impact | 4.00 s | 0.060 s | 3.29 s | 70% |
| 563 | Drum deep impact | 5.50 s | 0.168 s | 1.97 s | 71% |
| 498 | Deep heartbeat impact | 3.47 s | 0.100 s | 1.54 s | 68% |
| 2901 | Epic movie impact | 5.14 s | 1.406 s | 4.58 s | 49% |
| 2900 | Movie logo intro impact | 8.97 s | 4.424 s | 8.00 s | 69% (too long) |
| 788 | Big cinematic impact | 7.94 s | 2.184 s | 4.95 s | 78% |

Rendered for the user's ears (2026-10-30): .mvwork/ident/apotheon_ident_v2_sound_{A_1143,B_2908,C_1687,D_563}.mp4.
Picture is identical across the four; the ear decides.

## Edge artwork

Four stills on the hold frame in .mvwork/ident/ident_edge_options_A_to_D_frame44.png:
A none, B hairline window + ember corner brackets (the card plate's window border),
C filigree corners (placeholder), D burnt edges (too strong as mocked). Recommendation B.
Web research on what bands do found only stock-asset listings; the genre read (modern
metal = no frame or a minimal geometric one; ornate frames live in the symphonic end and
only work hand-drawn) is from the videos, not a source.
