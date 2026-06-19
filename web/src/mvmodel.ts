// MV Studio shared types + helpers (the LTX-MSR-native music-video editor).
// A music video = an ordered timeline of MSR blocks. Each block is one /api/video/ltx_msr
// render: up to 4 subject refs (resolved from characters x wardrobe, or picked manually),
// a background ref, a main prompt, the PromptRelayEncode timeline-prompter, and optional
// native single-pass lip-sync (audio window + vocal isolation).

import { rid } from "./ui";

// ---- character library (identity core + per-video wardrobe) ----
export type Identity = { faceRefId?: string; bodyRefId?: string; notes?: string };
export type Wardrobe = {
  id: string; name: string; outfitPrompt?: string;
  faceRefId?: string; bodyRefId?: string; sheetId?: string;   // the locked 2-image MSR ref pair
};
export type Character = {
  id: string; name: string; role?: string; kind?: string;
  appearance?: string;          // free-text look description (drives still generation + LLM enhance)
  identity?: Identity; wardrobes?: Wardrobe[];
  // legacy single-ref fields (pre-v2 characters); still resolved as a fallback
  refStillId?: string; refStillIds?: string[]; loraName?: string; method?: string; notes?: string;
};

// the up-to-2 hero characters a block features (each contributes its wardrobe's face+body refs)
export type BlockChar = { charId: string; wardrobeId?: string };
// one timeline-prompter segment: a frame count + the prompt active over it
export type Seg = { len: string; prompt: string };
export type RenderMode = "msr" | "i2v" | "s2v";

export type Block = {
  id: string; idx: number; start: number; end: number; kind: "msr";
  renderMode: RenderMode;
  prompt: string; negative: string;
  chars: BlockChar[];           // hero characters (resolve to subject refs)
  subjectIds: string[];         // manual / extra subject stills (merged with char refs, capped 4)
  backgroundId: string;
  // PromptRelayEncode timeline prompter
  tlOn: boolean; global: string; segs: Seg[]; epsilon: number;
  // native lip-sync
  lipsync: boolean; audioId: string; audioStart: number; isolateVocal: boolean;
  // render params
  width: number; height: number; frames: number; fps: number; seed: number;
  refFrames: number; msrStrength: number; guideStrength: number; steps: number; cfg: number;
  // result
  clipId?: string; upscaledId?: string;
};

export const MSR_REF_COMBOS = [17, 25, 33, 41];   // LiconMSR frame_count combo
export const DEFAULT_W = 832;
export const DEFAULT_H = 480;
export const DEFAULT_FPS = 24;

export function makeBlock(start: number, fps = DEFAULT_FPS): Block {
  return {
    id: rid(), idx: 0, start, end: start + 6, kind: "msr", renderMode: "msr",
    prompt: "", negative: "",
    chars: [], subjectIds: [], backgroundId: "",
    tlOn: false, global: "", segs: [], epsilon: 0.001,
    lipsync: false, audioId: "", audioStart: start, isolateVocal: true,
    width: DEFAULT_W, height: DEFAULT_H, frames: 145, fps, seed: 0,
    refFrames: 17, msrStrength: 1.0, guideStrength: 1.0, steps: 8, cfg: 1.0,
  };
}

// migrate a persisted (possibly partial) block to the full shape
export function hydrateBlock(b: Partial<Block>, i: number): Block {
  const base = makeBlock(b.start ?? 0);
  return { ...base, ...b, idx: i, id: b.id || base.id,
           segs: (b.segs || []).map((s) => ({ len: String(s.len ?? ""), prompt: s.prompt ?? "" })) };
}

// coerce a frame count to LTX's 8k+1 grid (mirrors backend _ltx_frames)
export function ltxFrames(n: number): number {
  const f = Math.max(9, Math.round(n));
  return Math.round((f - 1) / 8) * 8 + 1;
}

// the up-to-4 subject reference still-ids a character contributes for a given wardrobe
export function charRefIds(c: Character | undefined, wardrobeId?: string): string[] {
  if (!c) return [];
  const w = (c.wardrobes || []).find((x) => x.id === wardrobeId) || (c.wardrobes || [])[0];
  if (w && (w.faceRefId || w.bodyRefId)) return [w.faceRefId, w.bodyRefId].filter(Boolean) as string[];
  if (c.identity && (c.identity.faceRefId || c.identity.bodyRefId))
    return [c.identity.faceRefId, c.identity.bodyRefId].filter(Boolean) as string[];
  if (c.refStillIds?.length) return c.refStillIds;
  if (c.refStillId) return [c.refStillId];
  return [];
}

// the single still that best represents a character when composited into a SCENE (full-body
// preferred - e.g. a band member holding their instrument; falls back to face / legacy refs)
export function sceneRefOf(c: Character | undefined): string {
  if (!c) return "";
  const w = (c.wardrobes || [])[0];
  return c.identity?.bodyRefId || w?.bodyRefId || c.identity?.faceRefId || w?.faceRefId
    || (c.refStillIds && c.refStillIds[0]) || c.refStillId || "";
}

// final ordered subject_ids for a block: hero characters first, then manual extras, capped at 4
export function resolveSubjects(b: Block, libChars: Character[]): string[] {
  const fromChars = b.chars.flatMap((bc) =>
    charRefIds(libChars.find((c) => c.id === bc.charId), bc.wardrobeId));
  const all = [...fromChars, ...b.subjectIds].filter(Boolean);
  return Array.from(new Set(all)).slice(0, 4);
}

// build the /api/video/ltx_msr payload from a block (only sends what's set; omits defaults)
export function msrPayload(b: Block, subjectIds: string[], songAudioId: string): Record<string, unknown> {
  const p: Record<string, unknown> = {
    subject_ids: subjectIds, background_id: b.backgroundId, prompt: b.prompt,
    width: b.width, height: b.height, frames: b.frames, fps: b.fps,
    ref_frames: b.refFrames, msr_strength: b.msrStrength, guide_strength: b.guideStrength,
    steps: b.steps, cfg: b.cfg,
  };
  if (b.seed) p.seed = b.seed;
  if (b.negative.trim()) p.negative = b.negative.trim();
  if (b.tlOn) {
    p.global_prompt = b.global;
    p.local_prompts = b.segs.map((s) => s.prompt);
    p.segment_lengths = b.segs.map((s) => s.len.trim()).filter(Boolean).join(",");
    p.epsilon = b.epsilon;
  }
  if (b.lipsync) {
    p.audio_id = b.audioId || songAudioId;
    p.audio_start = b.audioStart;
    p.isolate_vocal = b.isolateVocal;
  }
  return p;
}

export const blockSeconds = (b: Block) => +(b.frames / b.fps).toFixed(2);

// a shot from /api/mv/script (the LLM shot list) - the source for "generate from song"
export type ScriptShot = {
  start?: number; end?: number; type?: string; scene?: string; action?: string;
  costume?: string; characters?: string[]; lipsync?: boolean; section?: string;
};

// map an LLM shot into an MSR block: scene + costume + action -> prompt, names -> hero
// characters (matched against the library, capped at 2), timing + lip-sync carried over.
export const LTX_MAX_SECONDS = 20;   // practical single-clip ceiling on the 3090

export function shotToBlock(s: ScriptShot, libChars: Character[], audioId: string): Block {
  const start = s.start || 0;
  const rawEnd = (s.end && s.end > start) ? s.end : start + 6;
  const dur = Math.min(LTX_MAX_SECONDS, Math.max(2, rawEnd - start));   // honor the script's length, clamped LTX-sane
  const end = start + dur;
  const base = makeBlock(start);
  const chars: BlockChar[] = (s.characters || [])
    .map((name) => libChars.find((c) => c.name.toLowerCase() === String(name).toLowerCase()))
    .filter((c): c is Character => !!c)
    .slice(0, 2)
    .map((c) => ({ charId: c.id, wardrobeId: c.wardrobes?.[0]?.id }));
  const prompt = [s.scene, s.costume ? `wearing ${s.costume}` : "", s.action].filter(Boolean).join(". ");
  return { ...base, id: rid(), start, end, frames: ltxFrames(dur * base.fps),
    prompt, lipsync: !!s.lipsync, chars, audioId, audioStart: start };
}
