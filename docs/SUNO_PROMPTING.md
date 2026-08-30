# Suno Prompting - Best Practices for Rock and Metal

Researched 2026-08-31 from community guides, tag references and Suno's own help pages
(v5/v5.5 era). Provenance: community-sourced practice unless marked [official]. Suno tags are
probabilistic hints, not commands - everything here raises hit-rate, nothing guarantees.

The purpose of this doc: our songs are WRITTEN as Music 3 captions (the master document);
this is the translation target when a song goes to Suno for a release-quality render.

## 1. The two control surfaces (Custom Mode)

- **Style of Music field** (max ~1000 chars): the WHOLE song's genre, sound, production,
  vocal character. Front-load it: the first 20-30 words carry the most weight.
- **Lyrics field**: the words, plus `[Square Bracket]` meta tags on their own lines that
  control structure and moments (sections, solos, vocal delivery, dynamics).
- **Exclude Styles field** (Advanced): the official negative control. Negatives written in
  the style field ("no synth") are UNRELIABLE; put them here instead. [official-ish,
  consistently reported]
- **Sliders**: Weirdness 40-60% for real songs (50 = normal); Style Influence HIGH (70%+)
  when the genre must hold exactly - for metal, hold it high or the output drifts soft.
  High weirdness + high style influence = adventurous but on-genre.

## 2. Style field recipe (metal)

Order and count matter:

1. Start with tempo + key + ONE precise subgenre. Precision beats piling: "melodic power
   metal" not "metal, power metal, heavy metal, rock".
2. Name 2-3 instruments WITH adjectives ("downtuned crushing rhythm guitars",
   "galloping double-kick drums"). Tuning words and drum-feel words (blast beats /
   double-kick / half-time stomp) are the levers that define metal subgenres.
3. One vocal descriptor phrase (see section 4).
4. Production words ("wall of sound", "heavily compressed", "raw", "polished arena mix").
5. TOTAL ~6-7 descriptors. Fewer = generic distorted rock; more = the mix turns to mush
   and Suno contradicts itself. This mirrors our ACE tag-bloat finding exactly.

Without clear extreme signals Suno defaults to the LEAST extreme reading of a prompt -
the pop-drift failure. The fix is subgenre precision + energy words ("aggressive",
"relentless", "crushing"), not more tags.

Example (Queen of the Hollow Stars compressed):
"140 bpm, E minor, symphonic metal, downtuned crushing rhythm guitars, galloping
double-kick drums, cinematic orchestra beneath the guitars, soaring operatic female
mezzo-soprano, dark reverent verses, monumental tragic choruses, wall of sound"
Exclude: "pop, EDM, synth-pop, country, acoustic"

## 3. Lyric meta tags

- Structure: `[Intro] [Verse] [Pre-Chorus] [Chorus] [Bridge] [Guitar Solo] [Breakdown]
  [Outro]` - our sheets already use this convention; they paste in unchanged.
- `[Guitar Solo]` in the LYRICS is what actually produces a solo; the style field alone
  is unreliable for it. Same for `[Breakdown]`.
- Performance tags stack on a section line: `[Chorus] [Belted]`, `[Verse] [Whispered]`,
  `[Female Vocal]`, `[Harmonies]`, `[Scream]`, `[Gang Vocals]`.
- Put a tag on its own line right BEFORE the lyrics it affects. Tags are not sung -
  but keep them simple; weird tag text can leak into the vocal (same failure Music 3 has
  with decorated section tags).
- Instrument-cue moderation: more than 3-4 instrument tags confuses it.
- A dynamics arc works via tags: `[Instrumental Intro]`, `[Build]`, `[Drop to whisper]`,
  `[Final Chorus - biggest]` - phrased simply.

## 4. Vocals (the big win over Music 3)

- Operatic vocabulary WORKS here: "bel canto", "operatic soprano", "dramatic vibrato",
  "classically trained" are real Suno style vocabulary (suno.com/style/bel-canto-technique
  exists). The exact words that derail Music 3's voice are the right words on Suno.
- Community formula for a Nightwish/Epica-type lead: "soaring operatic female
  soprano/mezzo-soprano, powerful and emotional, dramatic vibrato, transitions between
  operatic passages and chest-voice belting, ethereal backing choir".
- ARTIST NAMES ARE BLOCKED in prompts. Describe the voice instead (register + weight +
  grit + delivery) - our Vocal Gender & Timbre prose translates nearly verbatim.
- Extreme vocals get sanitized: growls/shrieks pull back toward a clean shout. Push with
  explicit subgenre ("death metal") + `[Harsh Vocals]`/`[Scream]` tags, accept variance.
- **Personas/Voices**: after a take nails the band's singer, save it as a Persona and
  reuse across every song = the consistent-vocalist feature we cannot get locally.
  A Persona carries voice + genre + energy + production ("an essence, not a lock"), so
  mint it from a take whose WHOLE sound is worth inheriting.

## 5. Mapping our Music 3 captions -> Suno

| Music 3 field                        | Suno destination                                  |
|--------------------------------------|---------------------------------------------------|
| Basic Attributes (bpm/key/genre)     | Style field, FIRST words                          |
| Vocal Gender & Timbre                | Style field vocal phrase (or skip if Persona)     |
| Sonics & Production                  | Style field production words (2-3 max)            |
| Primary (guitars)                    | Style field instruments w/ adjectives             |
| Secondary/Tertiary                   | ONE clause ("orchestra beneath the guitars")      |
| Global Emotional Progression         | 1-2 mood words + per-section lyric tags           |
| Groove per-section direction         | lyric tags ([Breakdown], [Build], [Half-time])    |
| Explicit negatives                   | Exclude Styles field                              |
| Vocal Style per-section              | stacked tags ([Chorus] [Belted], [Verse] [Whispered]) |
| Harmony/Backing                      | [Gang Vocals]/[Harmonies]/[Choir] tags + one style clause |
| Embellishments (solo)                | [Guitar Solo] tag at the right spot               |

Compression discipline: the caption is 4000+ chars; the style field wants ~400-700.
Keep the genre tail, the vocal phrase, the guitar phrase, the drum-feel phrase, the
production phrase, 1-2 mood words. Everything section-shaped becomes lyric tags.

## 6. Known limits / gotchas

- No per-section timing control (same as Music 3); structure follows lyric tags loosely.
- Not seed-deterministic; the workflow is roll-and-keep. Decide winners BY EAR in their
  player; download once (downloads are metered from Sept 2026: Pro 20/mo, Premier 60/mo,
  Premier+Studio unlimited).
- Artist names filtered; real-singer voice cloning not offered.
- Style field negatives unreliable -> Exclude field.
- Tags obeyed most of the time, not always; a run that ignores a tag = re-roll, or
  simplify the tag wording.
