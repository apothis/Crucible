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
  gender?: string;              // "female" | "male" | "non-binary" | "" - fed into band-composite + cast naming
  appearance?: string;          // free-text look description (drives still generation + LLM enhance)
  identity?: Identity; wardrobes?: Wardrobe[];
  // legacy single-ref fields (pre-v2 characters); still resolved as a fallback
  refStillId?: string; refStillIds?: string[]; loraName?: string; method?: string; notes?: string;
};

// the up-to-2 hero characters a block features (each contributes its wardrobe's face+body refs)
export type BlockChar = { charId: string; wardrobeId?: string };
// one timeline-prompter segment: a frame count + the prompt active over it. In KEYFRAME mode a
// segment may also pin a keyframe still: keyframeStillId places that library still at the segment's
// START (or its END when isEndFrame), and the model interpolates between the pinned keyframes.
export type Seg = { len: string; prompt: string; keyframeStillId?: string; isEndFrame?: boolean };
export type RenderMode = "msr" | "i2v" | "s2v" | "keyframe" | "fflf";

// ---- Shot Studio (per-segment editor) -------------------------------------------------------
// A rendered video for a piece: a half-res hunt draft (video-only) OR a finished full-res take.
export type Take = {
  id: string;
  clipId: string;                 // a job id; media at /api/media/<clipId>
  stage1Seed?: number;            // the hunt/base seed (what you picked)
  stage2Seed?: number;            // the refine/multiroll seed
  draft?: boolean;                // true = half-res hunt draft (no lip-sync), false = finished
  label?: string;
};
// The LTXDirector editor's authored output — feeds /api/video/ltx_keyframe (timeline_data passthrough).
export type DirectorPayload = {
  timeline_data?: string; local_prompts?: string; segment_lengths?: string;
  guide_strength?: string; global_prompt?: string;
};
// One piece of a continuous take. Piece 0 = the base shot; later pieces = FFLF extends off the
// previous piece's tail. Each piece keeps every take it rolled so you can pick / multiroll.
export type ChainPiece = {
  id: string;
  lane: "fflf" | "msr";
  label?: string;
  takes: Take[];
  selectedTakeId?: string;        // which take is this piece's chosen clip
  lastStillId?: string;           // fflf: the last-frame keyframe still
};

export type Block = {
  id: string; idx: number; start: number; end: number; kind: "msr";
  renderMode: RenderMode;
  prompt: string; negative: string;
  scene: string;                // SETTING/ENVIRONMENT only (person-free) - generates the MSR background
  framing: string;              // "close" | "medium" | "wide" - so the bg is framed to match the shot
  chars: BlockChar[];           // featured hero character(s) - anchored via MSR (resolve to subject refs)
  bandInScene: string[];        // other band members composited INTO the background (char ids; not MSR)
  subjectIds: string[];         // manual / extra subject stills (merged with char refs, capped 4)
  backgroundId: string;
  // PromptRelayEncode timeline prompter
  tlOn: boolean; global: string; segs: Seg[]; epsilon: number;
  // native lip-sync
  lipsync: boolean; audioId: string; audioStart: number; isolateVocal: boolean;
  // render params
  width: number; height: number; frames: number; fps: number; seed: number;
  refFrames: number; msrStrength: number; guideStrength: number; steps: number; cfg: number;
  // result (clipId/backgroundId = the SELECTED take; *Variants = every take rendered, to pick between)
  clipId?: string; upscaledId?: string;
  clipVariants?: string[];        // all rendered clip job-ids for this block
  bgVariants?: string[];          // all composed/generated background still-ids for this block
  // ---- Shot Studio per-segment editor: a continuous take = ordered chain pieces ----
  pieces?: ChainPiece[];          // piece 0 = base shot; later = FFLF extends off the prior tail
  // FFLF anchors come from the timeline editor's image keyframes (first image = opening, last = target);
  // no separate anchor fields. These two are just the AddGuide strengths.
  fflfFirstStrength?: number; fflfLastStrength?: number;
  charLora?: string;              // optional downloadable identity LoRA (instead of / with MSR)
  charStrength?: number;          // character-LoRA strength (default 1.0)
  assembledId?: string;           // the crossfade-assembled continuous take (set by the chain orchestrator)
  timelineData?: string;          // the LTXDirector editor's timeline_data JSON (segments/audio/keyframes)
  director?: DirectorPayload;      // full editor output (timeline_data + relay fields) for rendering
};

export const MSR_REF_COMBOS = [17, 25, 33, 41];   // LiconMSR frame_count combo
export const DEFAULT_W = 1280;
export const DEFAULT_H = 720;
export const DEFAULT_FPS = 24;

export function makeBlock(start: number, fps = DEFAULT_FPS): Block {
  return {
    id: rid(), idx: 0, start, end: start + 6, kind: "msr", renderMode: "msr",
    prompt: "", negative: "", scene: "", framing: "medium",
    chars: [], bandInScene: [], subjectIds: [], backgroundId: "",
    tlOn: false, global: "", segs: [], epsilon: 0.001,
    lipsync: false, audioId: "", audioStart: start, isolateVocal: false,
    width: DEFAULT_W, height: DEFAULT_H, frames: 145, fps, seed: 0,
    refFrames: 17, msrStrength: 1.0, guideStrength: 1.0, steps: 8, cfg: 1.0,
  };
}

// migrate a persisted (possibly partial) block to the full shape
export function hydrateBlock(b: Partial<Block>, i: number): Block {
  const base = makeBlock(b.start ?? 0);
  return { ...base, ...b, idx: i, id: b.id || base.id,
           segs: (b.segs || []).map((s) => ({ len: String(s.len ?? ""), prompt: s.prompt ?? "",
             ...(s.keyframeStillId ? { keyframeStillId: s.keyframeStillId } : {}),
             ...(s.isEndFrame ? { isEndFrame: true } : {}) })) };
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

// the per-segment frame windows of a block: timed lengths if every segment is timed, else an even
// split across b.frames (last window absorbs the remainder). Mirrors how the relay slices the clip.
export function segWindows(b: Block): { start: number; len: number }[] {
  const n = b.segs.length;
  if (!n) return [{ start: 0, len: b.frames }];
  const lens = b.segs.map((s) => parseInt(s.len, 10) || 0);
  const allTimed = lens.every((L) => L > 0);
  const out: { start: number; len: number }[] = [];
  if (allTimed) {
    let acc = 0;
    for (const L of lens) { out.push({ start: acc, len: L }); acc += L; }
  } else {
    const each = Math.max(1, Math.floor(b.frames / n));
    let acc = 0;
    b.segs.forEach((_, i) => {
      const L = i === n - 1 ? Math.max(1, b.frames - acc) : each;
      out.push({ start: acc, len: L }); acc += L;
    });
  }
  return out;
}

// the keyframe stills a KEYFRAME-mode block pins, in render order: every segment carrying a
// keyframeStillId becomes one keyframe placed at its segment window (start, or end if isEndFrame).
export function blockKeyframes(b: Block): { still_id: string; start: number; length: number; isEndFrame: boolean }[] {
  const win = segWindows(b);
  return b.segs
    .map((s, i) => ({ s, w: win[i] }))
    .filter((x) => x.s.keyframeStillId)
    .map((x) => ({ still_id: x.s.keyframeStillId as string, start: x.w.start, length: x.w.len,
                   isEndFrame: !!x.s.isEndFrame }));
}

// build the /api/video/ltx_keyframe payload from a KEYFRAME-mode block (the LTXDirector 2-stage
// keyframe-guide pipeline). Keyframe stills come from the segments (blockKeyframes); the per-segment
// PROMPT timeline is sent like MSR so a shot can both interpolate between stills AND schedule prompts.
// No lip-sync (keyframe / B-roll pins identity via the stills). Output res == width x height (the
// stage-1 base_scale 0.5 nets back the spatial-x2 refine); the backend exposes base_scale for 2x.
export function keyframePayload(b: Block): Record<string, unknown> {
  const p: Record<string, unknown> = {
    keyframes: blockKeyframes(b).map((k) => ({ ...k, guide_strength: b.guideStrength })),
    prompt: b.prompt, width: b.width, height: b.height, frames: b.frames, fps: b.fps, cfg: b.cfg,
  };
  if (b.seed) p.seed = b.seed;
  if (b.negative.trim()) p.negative = b.negative.trim();
  if (b.tlOn) {
    p.global_prompt = b.global;
    p.local_prompts = b.segs.map((s) => s.prompt);
    p.segment_lengths = b.segs.map((s) => s.len.trim()).filter(Boolean).join(",");
    p.epsilon = b.epsilon;
  }
  return p;
}

export const blockSeconds = (b: Block) => +(b.frames / b.fps).toFixed(2);

// a shot from /api/mv/script (the LLM shot list) - the source for "generate from song"
export type ScriptShot = {
  start?: number; end?: number; type?: string; scene?: string; action?: string;
  costume?: string; characters?: string[]; lipsync?: boolean; section?: string; camera?: string;
  render?: string;   // "msr" (default, performer / music-synced) | "keyframe" (pure scenic B-roll)
  framing?: string;  // "close" | "medium" | "wide" - kept out of scene so the background stays person-free
  band_in_scene?: string[];  // other band members composited into the background (NOT MSR-anchored)
  segments?: { seconds?: number; action?: string }[];   // per-shot prompt timeline (relay)
};

// the ONLY camera moves that render cleanly on our LTX MSR pipeline, carried as PROMPT TEXT (never a
// camera LoRA - those morph when stacked on the MSR IC-LoRA). Mirrors backend musicvideo.CAMERA_PHRASE.
const CAMERA_PHRASE: Record<string, string> = {
  "static": "the camera is locked off and still",
  "slow push-in": "the camera slowly and smoothly zooms in toward the subject",
  "slow pull-back": "the camera slowly and smoothly pulls back from the subject",
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
  // band members standing in the background (composited via Qwen, NOT MSR-anchored); exclude the featured
  const bandInScene = (s.band_in_scene || [])
    .map((name) => libChars.find((c) => c.name.toLowerCase() === String(name).toLowerCase()))
    .filter((c): c is Character => !!c)
    .map((c) => c.id)
    .filter((id) => !chars.some((bc) => bc.charId === id));
  const cam = CAMERA_PHRASE[s.camera || "static"] || CAMERA_PHRASE["static"];
  // scene = the SETTING/ENVIRONMENT only (person-free) - this is what generates the MSR background, so it
  // must NOT describe the singer. Framing is separate so the video can be close/medium without the
  // background ever containing a person. The full render prompt = framing + scene + costume + action.
  const scene = (s.scene || "").trim();
  const framePhrase = ({ close: "close-up shot", medium: "medium shot", wide: "wide shot" } as Record<string, string>)[s.framing || "medium"] || "medium shot";
  const prompt = [framePhrase, scene, s.costume ? `wearing ${s.costume}` : "", s.action, cam]
    .filter(Boolean).join(". ");
  // per-shot timeline: the constant look -> global, each slice's action -> a segment prompt. Lengths
  // are all-or-nothing (every segment timed, or all blank for an even split) so the relay's
  // local_prompts and segment_lengths stay aligned. Needs >=2 segments to be worth scheduling.
  const rawSegs = (s.segments || []).filter((sg) => sg && String(sg.action || "").trim());
  const allTimed = rawSegs.length > 0 && rawSegs.every((sg) => (sg.seconds || 0) > 0);
  const segs: Seg[] = rawSegs.map((sg) => ({
    len: allTimed ? String(Math.round((sg.seconds as number) * base.fps)) : "",
    prompt: String(sg.action).trim(),
  }));
  const tlOn = segs.length > 1;
  const look = [framePhrase, scene, s.costume ? `wearing ${s.costume}` : "", cam].filter(Boolean).join(". ");
  // RENDER MODE from the planner: keyframe ONLY for pure scenic B-roll (no performer / no music sync);
  // anything with lip-sync, a performance shot, or named characters present stays MSR (the only mode that
  // lip-syncs or syncs motion to the music). Mirrors the backend parse_shots enforcement.
  const keyframeOk = s.render === "keyframe" && !s.lipsync && s.type !== "performance" && chars.length === 0;
  const renderMode: RenderMode = keyframeOk ? "keyframe" : "msr";
  return { ...base, id: rid(), start, end, frames: ltxFrames(dur * base.fps), renderMode,
    prompt, scene, framing: s.framing || "medium", global: tlOn ? look : "", segs, tlOn,
    lipsync: !!s.lipsync && renderMode !== "keyframe", chars, bandInScene, audioId, audioStart: start };
}

// cumulative section-boundary times (seconds) from a song's ordered sections (each {seconds}).
export function sectionBoundaries(sections: { seconds?: number }[]): number[] {
  const out = [0]; let acc = 0;
  for (const s of sections || []) { acc += Math.max(0, s.seconds || 0); out.push(+acc.toFixed(2)); }
  return out;
}

// Snap shot cuts onto the song's structure: nudge each boundary between consecutive blocks to the
// nearest SECTION boundary (priority, so cuts land on verse/chorus changes), else the nearest BEAT
// (so they're musically tight). Keeps blocks contiguous, monotonic, and a sane minimum length;
// recomputes each block's frames + audioStart from its new window. The first start and last end are
// left as-is. Deterministic - more reliable than asking the LLM to hit beats by eye.
export function snapBlocksToStructure(blocks: Block[], beats: number[], sectionBounds: number[]): Block[] {
  if (blocks.length < 2) return blocks;
  const fps = blocks[0].fps || DEFAULT_FPS;
  const SECT_TOL = 1.5, BEAT_TOL = 0.35, MIN = 1.5;
  const nearest = (v: number, arr: number[]): { val: number; d: number } | null => {
    if (!arr.length) return null;
    let best = arr[0], bd = Math.abs(arr[0] - v);
    for (const x of arr) { const d = Math.abs(x - v); if (d < bd) { bd = d; best = x; } }
    return { val: best, d: bd };
  };
  const bounds = [blocks[0].start, ...blocks.map((b) => b.end)];
  for (let i = 1; i < bounds.length - 1; i++) {
    const v = bounds[i];
    const sec = nearest(v, sectionBounds);
    const beat = nearest(v, beats);
    let nv = v;
    if (sec && sec.d <= SECT_TOL) nv = sec.val;
    else if (beat && beat.d <= BEAT_TOL) nv = beat.val;
    if (nv > bounds[i - 1] + MIN && nv < bounds[i + 1] - MIN) bounds[i] = +nv.toFixed(2);
  }
  return blocks.map((b, i) => {
    const start = bounds[i], end = bounds[i + 1];
    return { ...b, start, end, audioStart: start, frames: ltxFrames(Math.max(2, end - start) * fps) };
  });
}
