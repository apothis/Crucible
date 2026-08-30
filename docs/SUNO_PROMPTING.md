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

## 3a. The proven tag vocabulary (community-tested; use only this)

Core structure tags (most reliable, verbatim): [Intro] [Verse] [Verse 1] [Verse 2]
[Pre-Chorus] [Chorus] [Hook] [Bridge] [Break] [Interlude] [Instrumental]
[Instrumental Break] [Build-Up] [Breakdown] [Drop] [Guitar Solo] [Outro] [End].

A core tag may take ONE qualifier - instrument/arrangement (Orchestral, Piano, Drum,
Riff, Acoustic, Choir, Chant, A Cappella, Half-time, Heavy, Final) or delivery
(Whispered, Spoken, Belted, Falsetto, Harmonized, Gang Vocal). Delivery can instead be
STACKED as a second tag on the line: "[Chorus] [Belted]" - the documented pattern.

Rules learned the hard way:
- Imagery/theme words in tags are noise ("[Ash Outro]" from our own enrichment - wrong;
  "[Fading Outro]" right). Session-musician-chart vocabulary only.
- Intro/outro tags should NAME the featured element ("[Drum Intro]", "[Orchestral
  Intro]"): metal styles fill vague instrumental intros with unrequested lead-guitar
  noodling, and a concrete element crowds it out.
- Always end with [Outro] (then [End]) or the song cuts off / fades awkwardly.
- Tags in the STYLE field are ignored; they only work in the lyrics field.

## 3b. The gentle-intro playbook (hard-won, 2026-09)

An epic-metal style field makes Suno inflate instrumental intros (brass fanfare, guitar
fills, early choir) no matter what the intro tag says - the style field is global and
dominates a lone tag. In order of reliability:
1. EXTEND TWO-STAGE (best): generate the intro as its OWN short piece with a no-metal
   style ("solo cello, soft strings, quiet cinematic film score, no drums, no guitars,
   no choir"), then Extend it with the FULL song style + lyrics. Each extension takes
   its own style prompt (this is why extends "drift" - here it is the feature). Keep the
   extension prompt full-strength, repeat the BPM.
2. ALIGN EVERYTHING + [Crescendo]: caption intro -> arc sentence -> intro tag all agree,
   an early whispered vocal denies Suno empty bars, and a [Crescendo] tag AFTER the
   quiet lines gives the growth a sanctioned place instead of the intro.
3. SALVAGE: Crop / Remove Section / Replace Section on an otherwise-good take.
4. EXPERIMENT: render locally on Music 3 (whose per-section obedience is measured) and
   Cover it on Suno - structure from Music 3, sound from Suno. Unverified.
Splice-in-Studio (two generations, crossfade, Full Mix export) remains the fully
deterministic fallback.

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
  Mechanics: "..." menu on ANY song (own or Explore-page public songs) -> Make a Persona;
  select via the Persona dropdown in Custom Mode; free to create. Public/private toggle
  defaults to PUBLIC - flip the band's Personas private. Other users' public Personas
  are usable too.

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

## 7. Paid-tier features worth exploiting (researched 2026-08-31)

- **Cover (Remix family)**: upload audio (up to 8 min) and transform its STYLE while
  keeping the melody. This is the true remix Music 3 could not do (our latent-injection
  band collapses below 0.85 strength): a local Music 3 take uploaded and covered = our
  composition + Suno's production/vocal quality. Available Pro+.
- **Replace Section / Song Editor**: regenerate one section from the waveform - the
  Repaint equivalent (fix a flubbed line without re-rolling the song). Extend = add
  sections/outro.
- **Custom Models (v5.5, Pro+)**: upload >=6 of YOUR OWN tracks -> a personal model that
  generates in that style (up to 3 models). The hosted equivalent of the band-LoRA plan:
  train an "Apotheon model" on the best local+Suno keepers for style-consistent output.
  You must own the rights to the uploads.
- **Personas/Voices**: persistent singer identity across songs (see section 4).
- **Suno Studio (Premier)**: generative DAW - up to 12 time-aligned WAV stems (32-bit/48k
  on Premier), works on UPLOADED external audio too (i.e. it can stem-split our local
  Music 3 renders at high fidelity - worth testing against our vocal-isolation dead end),
  MIDI export (10 credits per extraction), multitrack editing, automation. Studio usage
  on Premier = unlimited downloads.
- **My Taste**: passive personalization of default styles from your likes; no action
  needed, but it means the account's suggestions drift toward the band's sound over time.

## 8. Studio export mechanics (measured on our account, 2026-09)

- Export **Full Mix** = bounce to the SUNO LIBRARY with the master-bus FX chain BAKED IN.
  No local file.
- Export **Multitrack** = zip download of per-track audio, master bus BYPASSED (a
  single-track session yields one raw file). This is the download route exempt from the
  Sept 2026 download meter.
- **Mastered-download workaround** (user-discovered): master in Studio -> Export Full Mix
  (library, FX baked) -> open that bounced track in a NEW Studio session -> Export
  Multitrack -> the mastered audio downloads through the unmetered route.
- Keep BOTH: import the raw multitrack AND the mastered one as takes of the same title;
  the raw copy is the re-masterable archive.
- Suno Studio's mastering chain (EQ/compressor/limiter) is preferred by ear over our
  ffmpeg Master tab for Suno-rendered songs.
