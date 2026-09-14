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

REVISED 2026-09 after deeper research + our own failed prose-arc renders: the style
field is ONE comma-separated descriptor list, 30-45 words, NO periods ("commas separate
ideas; periods RESET context" - the prose arc sentences our first recipe used sat after
a period and were largely ignored, which matches our observed renders). The dynamic arc
goes at the END as comma journey phrases ("soft solo cello intro, gradually building,
colossal choir finale"). v5 rewards more description than v4, but as denser descriptors,
not longer sentences.

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

VERDICT after extended fighting: on epic symphonic metal, guitar/fanfare intro injection
could NOT be fully suppressed prompt-side - every lever below only lowers the odds. The
ACCEPTED WORKFLOW is post-edit in Studio: cut the intro, generate the replacement with
Studio's in-session generate box using a no-metal prompt ("quiet solo cello and soft
strings, cinematic, no drums, no guitars" - its own prompt, so it cannot grow guitars),
crossfade, Full Mix export, mastered-download chain.

An epic-metal style field makes Suno inflate instrumental intros (brass fanfare, guitar
fills, early choir) no matter what the intro tag says - the style field is global and
dominates a lone tag. In order of reliability:
1. EXTEND TWO-STAGE (best): generate the intro as its OWN short piece with a no-metal
   style ("solo cello, soft strings, quiet cinematic film score, no drums, no guitars,
   no choir"), then Extend it with the FULL song style + lyrics. Each extension takes
   its own style prompt (this is why extends "drift" - here it is the feature). Keep the
   extension prompt full-strength, repeat the BPM.
1a. WHY IT WORKS SOMETIMES (measured on our own songs, 2026-09): The Drowned Bell's
   whistle intro HELD while Queen of the Hollow Stars' quiet intro blew up, with
   identical prompt structure. Three separating factors: (a) genre-token prior - celtic
   folk metal's training data is full of quiet folk-instrument intros, symphonic metal's
   opens with orchestral bombast; the genre token sets the distribution, descriptors
   only steer inside it; (b) the opener instrument's ensemble membership - a tin whistle
   implicates nobody, "lone string melody" puts the blow-up-prone string section on
   stage at bar one (piano works for the same reason; cello does NOT, it is string
   section); (c) energy words inside the intro phrase ("soaring") prime bigness - use
   small dark words ("lone", "hushed", "over black water").
1b. THE CREATOR PATTERN (what the YouTube symphonic-metal channels do): the opening
   element goes EARLY in the style field, fused to the main instrumentation with a
   transition verb - "soft solo piano and strings intro building into downtuned guitars
   and double-kick" - not as an arc summary at the tail (the tail is the low-weight
   zone). Piano + strings is the community's most proven soft opener for this genre.
2. ALIGN EVERYTHING + [Crescendo]: caption intro -> arc sentence -> intro tag all agree,
   an early whispered vocal denies Suno empty bars, and a [Crescendo] tag AFTER the
   quiet lines gives the growth a sanctioned place instead of the intro.
3. SALVAGE: Crop / Remove Section / Replace Section on an otherwise-good take.
4. TESTED AND FAILED (2026-09): Music 3 render -> Suno Cover. Cover keeps the melody
   but RE-ARRANGES against the style prompt, so the metal style reclaims the intro
   (guitars injected again). Cover is not a structure-preservation tool.
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

## v6 (launched 2026-09-09) - what is known on day one

Provenance: OFFICIAL = suno.com/blog/introducing-v6 + release notes. REPORTED = hands-on
write-ups on launch day (weraveyou, note.com/cityedge). Nothing v6-specific exists yet on
help.suno.com or in the community tag guides (blakecrosley, hookgenius still say v5.5).

OFFICIAL
- Three models: v6 (flagship, "reliable, precise", paid), v6-wild ("built for
  exploration, less predictable and more varied", paid), v6-mini (fast, free tier).
- Trained from scratch on a new licensed data set (Warner, BMG, Believe). All previous
  models (v5.5 and earlier) are retired. Existing songs stay usable as cover/remix bases.
- Plain-language EDITING of an existing song: "change the chorus so it's sung by a
  gospel choir" keeps everything else. Single-lyric edits without regenerating.
- Mashups from several of your songs in one request ("vocals from x, drums from y").
- Sampling: "sample the riff at 0:45, isolate the guitar, build a beat around it".
- Vibe / reference prompting and multimodal input (text, audio, image, video).

MEASURED BY US (2026-09-09, Pro account, More Options panel after a page refresh)
- More Options now shows: Exclude styles, Vocal Gender (Male/Female), Duration
  (Custom/Auto), Max Mode (Off/On), Weirdness (default 50), Style Influence (default 50,
  was 71 on our old setting), Variety (a slider, default "Normal"), Personalize "My Taste"
  (Off/On). A stale tab shows the old v5.5 panel - refresh before judging.
- REPORTED, unverified by us: Max Mode = double credits, pitched at consistency across
  long songs and for Covers/Voices rather than raw quality; Variety rewrites the style
  prompt into a shorter generic one (one reviewer says keep it at the low end).
- Day-one impression from others (unverified): v6 ~ v5.5 quality with fewer stray
  vocal artifacts; v6-wild fails about half the time; v6-mini surprisingly competitive.
- Personas are now "Voices" (verified voice clones); Style Personas still exist inside
  Voices. Custom Models (10-60 own tracks) unchanged.

WHAT THIS MEANS FOR OUR WORKFLOW (to test, one variable at a time)
1. Vocal Gender (same control as before) on a fresh-from-scratch model: retest it on
   the female-lead songs that grew a male voice (Vanadis, Chooser of the Slain) - set
   Female, keep only the [Chorus] [Male Vocal] tag, see whether the chorus still gets him.
2. Plain-language edit = the new repair for a voice flip: "sing the second verse with
   the female vocalist, keep everything else" on a take we otherwise like.
3. Variety: leave at Normal for the first takes, then test the low end - if it really
   rewrites the style field, our voice/instrumentation wording is what it would drop.
4. Style Influence: our 70 was tuned on v5.5; retest 50 (new default) vs 70.
5. Everything measured on v5.5 (gothic+folk adds a male voice, [Key Change] no-op,
   intro-tag lottery, colon params ignored) is a HYPOTHESIS on v6 until re-measured -
   the model is new from scratch.
6. v6-wild is worth one roll on the deliberately silly songs (Yorkshire Thunder,
   Thirteenth Son of Odin); do not use it for the ones we need to land.

MEASURED 2026-09-09 (first v6 takes, Chooser of the Slain, defaults 50/50/Normal):
- Gender-tagged sections: NO better than v5.5 - the male voice still takes the tagged
  female verse and the chorus.
- Metal sound: WORSE on this song - muddier and more chaotic than the v5.5 takes.
  Caveat [H]: the panel reset Weirdness from our 28 to 50 and Style Influence from 71
  to 50 on the same run, so the defaults are a confound; retest at 28/71 before
  blaming the model.
- 2026-09-09: the model picker offers ONLY v6 / v6-wild / v6-mini. v5.5 is gone, so
  every fix now has to be found on v6.
- MEASURED 2026-09-09 (Winter in My Veins A/B, same lyrics + sliders): the user prefers
  our technical style field (bpm, key, subgenre, gear, vocal register) over the
  plain-language "no jargon" version that launch-week posts recommended for v6. Keep the
  established style-field form until there is real evidence for a change.

## Suno's own style expander (measured 2026-09-13, Queen of the Sun)

The SIMPLE-mode text box rewrites a one-line idea into a full style field before
generating. The user typed roughly "mid 1980s NWOBHM heavy metal, driving bass, analogue
production" and Suno expanded it to:

  mid-1980s NWOBHM heavy metal, galloping eighth-note rhythm guitars, twin harmonised
  lead guitars with melodic minor-key hooks, driving bass high in the mix, punchy live
  drums with roomy toms, anthemic [male -> user changed to mezzo soprano female] vocals,
  dramatic historical atmosphere, analog tape saturation, British valve-amp crunch,
  tight but organic studio mix, powerful chorus, guitar solo tradeoffs

That is the model's NATIVE dialect - the most training-aligned style text we can get.
What it looks like:
- comma-separated phrases, 12-13 of them, no sentences, no bpm, no key
- era + subgenre first, then guitars (rhythm, then leads), bass, drums, vocals,
  atmosphere, PRODUCTION (three phrases of it: tape, amp, mix), then arrangement
  ("powerful chorus", "guitar solo tradeoffs")
- concrete playing vocabulary ("galloping eighth-note", "roomy toms", "high in the mix")
  rather than adjectives
- it picked a MALE singer for NWOBHM on its own - gender must always be stated

How to use it:
1. For any new genre, type a one-line idea into simple mode, generate once, open the
   song and COPY the expanded style field. Edit the singer line. That is the template.
2. Our technical style fields still win the A/B (see above) - but fold in the expander's
   production triplet (tape / amp / mix) and its playing vocabulary; they cost nothing.
3. bpm and key are ours, not Suno's; keep them at the front, they have never hurt.

## v6 third-party tested notes (hookgenius "tested day one", read 2026-09-14)
- Field limits unchanged: style 1000 chars, lyrics 5000. The style prompt is stored
  byte-identical - the model does NOT rewrite it (so the expander only runs in simple mode).
- Exclude field: negatives typed into the STYLE box read as inclusions; keep them in Exclude.
- Style Influence default 50 = first suspect when a carefully specified prompt comes back
  generic. Variety = take-to-take divergence (lower for consistency, raise to explore).
  Vocal Gender / Weirdness / Max Mode listed but not tested by them.
- Bracketed per-section performance cues in the LYRICS are read and used by v6 (their test
  track's caption echoed "whispered", "Italian", "French" that lived only in section cues).
- Variance warning: the difference between the two takes of ONE generation can exceed the
  effect of a deliberate prompt change. Judge prompt edits over several generations.
- v6 returns longer songs on the same prompt (about 3:10 vs 2:45 on v5.5) and writes its
  own caption summary instead of echoing the style text.

## Reddit "Unofficial Suno v6 Prompting Guide" (r/SunoAI, OP Novel_Tumbleweed9989, read 2026-09-14)
Provenance: one experienced user's practice (pop-punk/metalcore/deathcore/darkwave), built
with ChatGPT, self-described as unofficial. 93 up / 67 down - contested; several
commenters say no prompt fixes v6's mix ("defaults to the generic v6 sound", "cannot do
proper guitars"), one EQ-analysed downloads and says v6 "does not know how to mix" and
must be spoon-fed mix instructions. Two commenters confirm the recipe fixed their
choruses/verses. Treat every claim as [H] until we measure it.

OP's core rule: "Tell it what the finished recording should actually sound like" -
direct the musicians AND the recording engineer.

1. Sliders for a NEW song: Variety 0, Weirdness under 50 (his sweet spot 30-50),
   Style Influence 80-95. Calibrate first, then loosen. No Max Mode during development.
   (Commenter: 33 / 85 also good. Our old 28 / 71 sits inside this band; the v6 reset to
   50 / 50 is OUTSIDE it - the confound we already suspected on Chooser of the Slain.)
2. Style field in MUSICAL ORDER with labels:
   Genre -> Vocals -> Drums -> Guitars -> Bass -> Other -> Arrangement/Energy ->
   Production Quality -> Ending Instruction (-> Sliders, which he types into the text;
   almost certainly a no-op, harmless).
3. PRODUCTION-QUALITY CLAUSE (his answer to muddy/distant/collapsing mixes - exactly our
   v6 complaint): "high-fidelity studio production, close-mic vocals, crisp transients,
   clear instrument separation, open low-mids, clear high-end air, stable loudness and
   tonal balance". Don't let it eat half the box.
4. Describe instrument ACTIVITY with verbs, not names: "continuous foreground guitar
   riffs, interlocking melodic leads, fast alternate picking, driving power chords";
   drums = what the drummer is doing (double bass, fills, ghost notes, ride bell).
   (Matches what fixed Say It Again's buried guitars for us.)
5. "Great chorus, dead verse": say "full-band energy throughout with foreground guitar
   riffs and active drums during verses" - v6 strips verses unless told not to.
6. Vocals: describe character, positively. His anti-operatic line for female vocals:
   "gritty chest voice, contemporary rock delivery, aggressive edge" (= our Noora fix).
   Prefer stating what you WANT over long negative lists.
7. Lyric tags = SHORT commands; long bracket paragraphs get partly ignored or sung.
   Per-section performance tags on their own line work: [Raspy Clean], [Building
   Scream], [Wounded Clean], [False Cord], [Layered Vocals] (his metalcore example).
   Parentheses = backing/echoes (same as ours).
8. Undefined intros invite random "grrr/whoa" noise: start lyrics immediately or write
   [Instrumental Intro] and describe it in the style box.
9. Songs ending mid-lyric (a v6 problem): finish the lyrics with
   [Final Chorus] ... [Instrumental Outro] [Hard Stop] [End]
   and add "complete every written lyric before a short instrumental outro and final
   hard stop" to the style box. The outro is a sacrificial buffer. Also stops outro drag.
10. Hybrid genres: "one main identity + two or three influences", not twelve.
11. Save working recipes verbatim; make "Recipe - v6" copies rather than rewriting.

Commenter alternatives (all [H]):
- Arqeph_: pipe-delimited labelled blocks in BOTH boxes, e.g. lyrics sections as
  "[Section: First Chorus | Mood: ... | Key Instruments: ... | Dynamic: ... | Spatial
  Effects: ...]" - says v6 treats the pipes as structure and can push lyrics off-beat
  when short repetitive lines are cut; fine for full lyrics.
- Auree_Gil: a working contralto rock/metal prompt in the OP's order (verses acoustic,
  choruses metal) - dynamics written as "restrained verses building into explosive
  choruses", ending "finish with a massive final metal chorus at maximum intensity".
- angryazneyes: ask an LLM to "rewrite this in musicological terms, under 500 chars".
- Osram_Serpentis: v6 quality degrades over a long song; extend with v6-mini instead.
- ART-ficial-Ignorance: the template blows the 1000-char limit and "insists on putting
  guitars in everything" (not a problem for us).

WHAT WE ADOPT (test on Chooser of the Slain, the muddy one, one change per run):
a. Sliders back to Weirdness 30 / Style Influence 85 / Variety 0 - first, alone.
b. Append the production-quality clause to the style field.
c. Add "full-band energy throughout, layered rhythm guitars remain present beneath the
   vocals in every verse" for songs where guitars vanish.
d. End every lyric sheet [Final Chorus] -> [Instrumental Outro] -> [Hard Stop] -> [End]
   (we already use [End]; add the buffer) + the ending sentence in the style box.
e. Keep our bpm/key/subgenre front-loading and tag grammar; the labelled-field order is
   compatible with it (genre first, vocals second, then the band).
