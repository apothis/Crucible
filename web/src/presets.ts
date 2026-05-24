// Subgenre style presets now live in the UNIFIED backend registry
// (backend/genres.py), delivered via /api/config as `cfg.genres` and shown as
// the preset chips. This file only keeps the Song-Constructor templates, which
// carry their own bespoke arrangement + style.
export type Preset = { name: string; tags: string; bpm: number; key: string };

// Song templates = a ready-made arrangement (section layout + per-section
// lengths) plus a bespoke style (its OWN tags/BPM/key, tuned to that song
// type). Picking one fills the block lane AND applies the style — all still
// editable afterward. `instrumental` (optional) presets the vocals toggle.
export type SongTemplate = {
  name: string;
  description: string;
  tags: string;
  bpm: number;
  key: string;
  instrumental?: boolean;
  sections: { type: string; seconds: number }[];
};

const S = (type: string, seconds: number) => ({ type, seconds });

export const SONG_TEMPLATES: SongTemplate[] = [
  {
    name: "Radio Anthem",
    description: "Tight verse–chorus hook structure with a bridge lift.",
    tags: "power metal, huge catchy sing-along chorus, galloping double-bass drums, palm-muted distorted guitars, twin lead guitar harmonies, soaring clean male vocals, anthemic, polished radio production",
    bpm: 165, key: "E minor",
    sections: [S("Intro", 8), S("Verse", 24), S("Chorus", 24), S("Verse", 24),
               S("Chorus", 24), S("Bridge", 16), S("Chorus", 24), S("Outro", 12)],
  },
  {
    name: "Power Ballad",
    description: "Slow build through pre-choruses to a soaring final climax.",
    tags: "symphonic power ballad, emotional clean vocals, gentle piano and lush orchestral strings in verses, heavily distorted guitars in the chorus, building dynamics, soaring melodic guitar solo, cinematic, dramatic",
    bpm: 76, key: "D minor",
    sections: [S("Intro", 12), S("Verse", 24), S("Pre-Chorus", 12), S("Chorus", 24),
               S("Verse", 24), S("Pre-Chorus", 12), S("Chorus", 24), S("Solo", 20),
               S("Chorus", 28), S("Outro", 16)],
  },
  {
    name: "Prog Epic",
    description: "Long-form, multi-movement journey with twin solos.",
    tags: "progressive metal, intricate technical guitar work, shifting odd time signatures, dynamic builds and quiet passages, atmospheric keyboards, technical drumming, powerful clean vocals, expansive epic arrangement",
    bpm: 140, key: "C# minor",
    sections: [S("Intro", 16), S("Verse", 28), S("Chorus", 24), S("Bridge", 20),
               S("Solo", 24), S("Breakdown", 20), S("Verse", 24), S("Chorus", 28),
               S("Solo", 20), S("Outro", 20)],
  },
  {
    name: "Thrash Banger",
    description: "Fast, aggressive and riff-driven — in and out fast.",
    tags: "thrash metal, fast aggressive palm-muted riffs, rapid double-bass drumming, shredding lead solos, gritty aggressive shouted vocals, relentless, intense, raw energy",
    bpm: 185, key: "E minor",
    sections: [S("Intro", 6), S("Verse", 20), S("Chorus", 16), S("Verse", 20),
               S("Chorus", 16), S("Solo", 16), S("Breakdown", 16), S("Outro", 8)],
  },
  {
    name: "Folk Singalong",
    description: "Anthemic gang-vocal chorus you can shout along to.",
    tags: "folk metal, festive folk melodies, violin, flute and accordion, distorted electric guitars, double-bass drums, rowdy anthemic gang vocals, drinking-song energy, upbeat and celebratory",
    bpm: 155, key: "E minor",
    sections: [S("Intro", 10), S("Verse", 24), S("Chorus", 24), S("Verse", 24),
               S("Chorus", 24), S("Bridge", 16), S("Chorus", 24), S("Outro", 14)],
  },
  {
    name: "Doom Dirge",
    description: "Slow and crushing — a few long, heavy movements.",
    tags: "doom metal, extremely slow tempo, downtuned crushing distorted guitars, thunderous slow drums, mournful clean vocals, oppressive dark heavy atmosphere, monolithic",
    bpm: 65, key: "C minor",
    sections: [S("Intro", 20), S("Verse", 40), S("Chorus", 32), S("Breakdown", 28),
               S("Verse", 40), S("Outro", 24)],
  },
  {
    name: "Instrumental Showcase", instrumental: true,
    description: "Lead-guitar driven instrumental built around solos.",
    tags: "instrumental heavy metal, virtuoso lead guitar, melodic shredding solos, harmonized twin guitar leads, tight driving rhythm section, dynamic, no vocals, guitar showcase",
    bpm: 145, key: "A minor",
    sections: [S("Intro", 10), S("Verse", 24), S("Solo", 24), S("Breakdown", 20),
               S("Solo", 24), S("Outro", 14)],
  },
  {
    name: "Pirate Shanty",
    description: "Rowdy Alestorm-style romp with a shout-along chorus.",
    tags: "pirate metal, fast galloping power metal riffs, accordion and keytar leads, jaunty sea shanty melodies, double-bass drums, gruff rowdy gang vocals, festive, anthemic, swashbuckling, drinking-song energy",
    bpm: 175, key: "E minor",
    sections: [S("Intro", 8), S("Verse", 22), S("Chorus", 24), S("Verse", 22),
               S("Chorus", 24), S("Solo", 18), S("Chorus", 26), S("Outro", 12)],
  },
];
