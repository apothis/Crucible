"""Unified genre registry — the SINGLE source of truth for genre/style info,
used everywhere it's needed:

- `tags`  → ACE-Step generation tag bundle (the Generate/Song/Restyle preset chips)
- `bpm` / `key` → SUGGESTED tempo + key (suggestions, never force-applied)
- `scale` → modal scale the symbolic riff/solo generator realises against
- `riff`  → LLM prompt feel for a rhythm riff
- `solo`  → LLM prompt feel for a lead solo
- `lead`  → riff renders single notes (shred) instead of power chords (optional)
- `reg`   → register offset in semitones for the riff/solo (optional)

Exposed to the frontend via /api/config ("genres") and consumed by guitar.py
for symbolic riff/solo generation. Add a genre here once → it appears in the
generation chips AND the Guitar riff/solo genre pickers.

NOTE on `solo` wording: the symbolic generator only places notes on a 16th-note
grid (it can't render bends/vibrato/pinch-harmonics/wah). So the solo prompts
lead with what the grid CAN express — note density/speed, how high to climb,
melodic contour (runs / leaps / pedal points / repeated motifs / sequences),
which characteristic scale tones to lean on, and phrasing (rests vs continuous).
"""

GENRES = [
    {"id": "power", "label": "Power", "bpm": 175, "key": "E minor", "scale": "minor",
     "tags": "power metal, galloping double-bass drums, fast palm-muted distorted guitars, twin lead guitar harmonies, soaring clean operatic male vocals, orchestral keyboards, triumphant, epic, fast tempo",
     "riff": "fast galloping, melodic and uplifting; mostly root chugs with soaring runs up to 5/8 and harmonised-sounding leaps",
     "solo": "fast triumphant legato runs that climb high (degrees to 12-15) and resolve onto held, singable peaks; mostly continuous with short breaths between phrases; melodic upward leaps, major-leaning"},
    {"id": "neoclassical", "label": "Neoclassical", "bpm": 168, "key": "A minor", "scale": "harmonic_minor",
     "lead": True, "reg": 24,
     "tags": "neoclassical metal, virtuoso shred lead guitar, fast sweep-picked arpeggios and harmonic-minor scalar runs, baroque classical influence, galloping double-bass drums, harpsichord and orchestral keyboards, soaring operatic male vocals, dramatic, Yngwie Malmsteen style",
     "riff": "virtuosic neoclassical shred (Yngwie/Becker): FAST continuous 16th SINGLE-NOTE harmonic-minor runs (raised-7th leading tone), pedal-point lines and diminished/sequence flurries; busy, few rests, ranging high (degrees to 12-15); a lead line, NOT power chords",
     "solo": "relentless 16th harmonic-minor runs, arpeggio-leap sweeps and pedal-point sequences climbing to the very top of the range (degrees 12-15); almost no rests; lean on the raised-7th leading tone and the b6 — pure Yngwie virtuosity"},
    {"id": "symphonic", "label": "Symphonic", "bpm": 150, "key": "D minor", "scale": "minor",
     "tags": "symphonic metal, lush orchestral strings, epic choir, heavily distorted guitars, double-bass drums, soaring operatic female vocals, cinematic, dramatic, grand",
     "riff": "grand, mid-fast power chords supporting an orchestra; melodic, dramatic, with melodic moves up the minor scale",
     "solo": "soaring cinematic melody — long sustained high notes answered by expressive medium runs; moderate density with dramatic rests; climbs then resolves, following the orchestra"},
    {"id": "folk_metal", "label": "Folk", "bpm": 160, "key": "E minor", "scale": "dorian",
     "tags": "folk metal, distorted electric guitars, double-bass drums, folk melody, violin, flute, accordion, anthemic gang vocals, festive, energetic",
     "riff": "lively, modal and melodic (dorian); bouncy mid-fast riff with folky melodic runs, major-ish",
     "solo": "jaunty dorian dance melody — bouncy mid-fast runs and ornaments built on a repeating singable motif; lean on the raised 6th; small rests for the lilt"},
    {"id": "heavy", "label": "Heavy", "bpm": 140, "key": "E minor", "scale": "minor",
     "tags": "heavy metal, crunchy distorted electric guitars, twin guitar harmonies, driving drums, powerful melodic male vocals, classic, high energy",
     "riff": "classic mid-tempo heavy-metal riffing; crunchy power chords with melodic moves and the occasional gallop",
     "solo": "classic melodic metal lead — singable medium-fast phrases mixing runs and held notes; balanced rests; climbs the scale and resolves to the root"},
    {"id": "nwobhm", "label": "NWOBHM", "bpm": 160, "key": "E minor", "scale": "minor",
     "tags": "NWOBHM, galloping twin-guitar harmonies, melodic distorted guitars, driving drums, soaring melodic male vocals, Iron Maiden style, heroic, energetic",
     "riff": "galloping, melodic and harmonised (Maiden-style); running 8th+16th gallops with melodic moves up the scale, heroic",
     "solo": "Maiden-style heroic lead — galloping melodic runs and a repeated harmonised-sounding motif climbing the scale; energetic, mostly continuous with clear phrase breaks"},
    {"id": "thrash", "label": "Thrash", "bpm": 180, "key": "E minor", "scale": "phrygian",
     "tags": "thrash metal, fast aggressive palm-muted riffs, rapid double-bass drumming, shredding lead solos, gritty aggressive male vocals, intense, dark",
     "riff": "fast, aggressive palm-muted 16th gallops on the low root with chromatic b2 and tritone (b5) accents; tight, relentless, downpicked",
     "solo": "fast, aggressive phrygian runs hammering the b2 and tritone (b5); relentless 16ths with very few rests, dark and angular, mid-high register", "lead": False},
    {"id": "melodeath", "label": "Melodeath", "bpm": 170, "key": "E minor", "scale": "minor",
     "tags": "melodic death metal, harmonized twin guitar leads, fast double-bass drums, growled vocals, melodic aggressive riffs, driving",
     "riff": "fast melodic tremolo + palm-muted riffing with harmonised lines climbing the minor scale; driving and aggressive",
     "solo": "fast emotive melodic runs climbing the minor scale around a repeated harmonised-sounding motif; driving, mostly continuous with brief phrase breaks"},
    {"id": "death_groove", "label": "Death / groove", "bpm": 140, "key": "E minor", "scale": "phrygian",
     "tags": "death metal, low downtuned palm-muted guitars, syncopated groove riffs, blast and double-bass drums, guttural growled vocals, brutal, heavy",
     "riff": "low, syncopated palm-muted chugs with off-beat rests and occasional b2/b5 pinch-accents; heavy and rhythmic, not busy",
     "solo": "short atonal/chromatic bursts around the b2 and b5 separated by rests; rhythmic and dissonant rather than fast; mid register, not busy"},
    {"id": "black", "label": "Black", "bpm": 180, "key": "F# minor", "scale": "harmonic_minor",
     "tags": "black metal, fast tremolo-picked guitars, relentless blast beats, shrieking raspy vocals, cold raw atmosphere, icy, lo-fi, dark",
     "riff": "fast, cold tremolo-picked single notes climbing the scale (steady 16ths, few rests), bleak and melodic", "lead": True, "reg": 12,
     "solo": "cold relentless tremolo melody — steady high 16th single notes weaving up and down harmonic minor with almost no rests; bleak, hypnotic, atmospheric"},
    {"id": "doom", "label": "Doom", "bpm": 80, "key": "C minor", "scale": "minor",
     "tags": "doom metal, slow heavy downtuned distorted guitars, crushing drums, mournful clean vocals, thick, dark, atmospheric, heavy",
     "riff": "very slow and heavy; sparse SUSTAINED power chords with lots of empty space (many rests), crushing and deliberate",
     "solo": "slow, mournful and spacious — a few sustained high notes with long rests between them; bluesy minor phrasing, lots of empty space, never busy"},
    {"id": "progressive", "label": "Progressive", "bpm": 140, "key": "C# minor", "scale": "dorian",
     "tags": "progressive metal, intricate technical riffs, shifting time signatures, atmospheric keyboards, dynamic builds, powerful clean vocals, expansive",
     "riff": "intricate, technical, syncopated riffing with odd accents and melodic moves; modal and dynamic",
     "solo": "technical and expressive — fast legato runs and wide interval leaps woven around a recurring melodic motif; shifting density with deliberate rests; ranges wide"},
    {"id": "gothic", "label": "Gothic", "bpm": 110, "key": "D minor", "scale": "minor",
     "tags": "gothic metal, melancholic brooding atmosphere, lush keyboards, downtuned guitars, deep clean vocals, female operatic vocals, dark romantic",
     "riff": "slow-to-mid, brooding power chords under lush keys; melancholic and spacious",
     "solo": "melancholic singing lead — slow expressive phrases of held notes and gentle runs with space between; dark and romantic, mid register"},
    {"id": "groove", "label": "Groove", "bpm": 125, "key": "C minor", "scale": "phrygian",
     "tags": "groove metal, heavy syncopated mid-tempo riffs, punchy downtuned guitars, pounding drums, aggressive shouted vocals, bouncy, powerful",
     "riff": "heavy syncopated mid-tempo chugs with bouncy groove and rests; punchy and rhythmic",
     "solo": "bluesy aggressive lead — punchy mid-tempo phrases around the low scale with groovy rests and a repeated lick; swaggering, not too busy"},
    {"id": "djent", "label": "Djent", "bpm": 120, "key": "F# minor", "scale": "phrygian",
     "tags": "djent, polyrhythmic palm-muted riffs, extended-range guitars, tight syncopated grooves, ambient clean sections, percussive, modern production",
     "riff": "tight low palm-muted polyrhythmic chugs with syncopated rests and the occasional high accent; percussive and modern",
     "solo": "ambient melodic lead floating over the groove — legato runs and wide melodic leaps with breathing space between phrases; modern, mid-high register"},
    {"id": "industrial", "label": "Industrial", "bpm": 130, "key": "E minor", "scale": "phrygian",
     "tags": "industrial metal, mechanical rhythms, distorted guitars, electronic beats, cold synth textures, harsh vocals, aggressive, machine-like",
     "riff": "mechanical, repetitive palm-muted chugs locked to a machine-like grid; cold and aggressive",
     "solo": "cold and repetitive — a short robotic motif repeated with dissonant accents, locked rigidly to the grid; sparse and mechanical"},
    {"id": "viking", "label": "Viking", "bpm": 150, "key": "E minor", "scale": "minor",
     "tags": "viking metal, epic folk melodies, choir chants, distorted guitars, double-bass drums, anthemic gang vocals, orchestral, heroic, battle atmosphere",
     "riff": "epic, anthemic mid-fast power chords with folk-melodic moves; heroic and chant-like",
     "solo": "epic anthemic lead — big folk-melodic runs and a chant-like repeated motif climbing to sustained high notes; heroic, moderate density"},
    {"id": "pirate", "label": "Pirate", "bpm": 175, "key": "E minor", "scale": "minor",
     "tags": "pirate metal, fast galloping power metal riffs, accordion and keytar leads, jaunty folk melodies, double-bass drums, gruff rowdy gang vocals, sea shanty energy, festive, anthemic, swashbuckling",
     "riff": "fast galloping power-metal riff with jaunty sea-shanty folk melodies; festive and rowdy",
     "solo": "jaunty fast folk-melodic lead — bouncy sea-shanty runs and a catchy repeated motif; festive and swashbuckling, mostly continuous with playful rests"},
    {"id": "hard_rock", "label": "Hard rock", "bpm": 120, "key": "A minor", "scale": "pentatonic_minor",
     "tags": "hard rock, crunchy overdriven guitars, driving backbeat drums, punchy bass, powerful rock vocals, energetic, anthemic",
     "riff": "mid-tempo driving power chords with swagger and space; pentatonic, punchy, not too busy",
     "solo": "tasteful pentatonic rock lead — bluesy medium-fast phrases and a repeated lick with plenty of space; swaggering, mid register, leave rests"},
    {"id": "accdc", "label": "AC/DC crunch", "bpm": 120, "key": "A major", "scale": "pentatonic_minor",
     "tags": "hard rock, AC/DC style, crunchy overdriven Marshall guitars, swinging backbeat, gang-shout backing vocals, raspy male vocal, mid tempo, live-room production",
     "riff": "mid-tempo big OPEN power chords with space and swing between hits; simple, pentatonic, swaggering — leave rests for groove",
     "solo": "simple swaggering pentatonic blues-rock licks — short punchy phrases with big rests and a repeated hook; attitude over speed, mid register"},
    {"id": "southern", "label": "Southern rock", "bpm": 110, "key": "A minor", "scale": "blues",
     "tags": "southern hard rock, bluesy thick overdriven guitar riffs, groovy drums, soulful gritty male vocal, mid tempo, Black Stone Cherry style",
     "riff": "bluesy, groovy pentatonic riffing with b5 passing notes; thick, mid-tempo, soulful",
     "solo": "soulful bluesy lead with b5 passing notes — expressive medium-paced pentatonic phrases and repeated licks with breathing space; groovy and gritty"},
    {"id": "arena", "label": "Arena rock", "bpm": 130, "key": "G major", "scale": "major",
     "tags": "80s arena rock, anthemic, big layered guitars, huge hooky chorus, polished production, gated-reverb drums, soaring male vocal, Bon Jovi style",
     "riff": "anthemic, simple and major-key; big ringing power chords, mid-tempo, hooky and uplifting",
     "solo": "melodic singable major-key lead — anthemic medium runs resolving onto held high notes; hooky and uplifting with balanced rests"},
]

BY_ID = {g["id"]: g for g in GENRES}
