import { useEffect, useState } from "react";
import { api, type Config, type LibItem, type SongDraft } from "./api";
import { Field, inp, PrimaryButton, GhostButton, rid, pollJob, type RunCtx } from "./ui";
import { useDrafts } from "./drafts";
import { MVTimeline } from "./MVTimeline";
import { ShotTimeline } from "./ShotTimeline";
import { Num, StillPick, stillLabel } from "./mvui";
import { CharacterLibrary } from "./Characters";
import {
  type Block, type Character, type RenderMode, type Seg, type ScriptShot,
  makeBlock, hydrateBlock, ltxFrames, resolveSubjects, charRefIds, sceneRefOf, msrPayload, shotToBlock,
  keyframePayload, blockKeyframes, blockSeconds, MSR_REF_COMBOS,
} from "./mvmodel";

// ---- readable block list: every shot at a glance (type / time / scene / needs / status) ----
const _fmt = (t: number) => `${Math.floor(t / 60)}:${String(Math.floor(Math.max(0, t) % 60)).padStart(2, "0")}`;
// a <video muted preload="metadata"> won't paint a frame on its own; a #t= media-fragment makes the
// browser seek to and render that frame, so the clip shows a real poster thumbnail instead of black.
const posterFrag = (url: string) => (url.includes("#") ? url : `${url}#t=0.5`);

function BlockList({ blocks, selId, onSelect, onRender, onPatch, onMove, libChars, library, busy }: {
  blocks: Block[]; selId: string; onSelect: (id: string) => void;
  onRender: (b: Block) => void; onPatch: (id: string, p: Partial<Block>) => void;
  onMove: (id: string, dir: -1 | 1) => void;
  libChars: Character[]; library: LibItem[]; busy: boolean;
}) {
  const typeOf = (b: Block) => {
    if (b.renderMode === "keyframe") return { label: "Keyframe", cls: "bg-sky-900/50 text-sky-200" };
    if (b.renderMode === "i2v") return { label: "B-roll", cls: "bg-slate-700/60 text-slate-300" };
    const names = b.chars.map((bc) => libChars.find((c) => c.id === bc.charId)?.name).filter(Boolean) as string[];
    const subj = resolveSubjects(b, libChars).length;
    if (subj > 1 || b.subjectIds.length > 1) return { label: names[0] ? `${names[0]} + band` : "Band", cls: "bg-amber-800/50 text-amber-200" };
    if (names.length) return { label: names.join(", "), cls: "bg-[#3a2a14] text-[var(--color-accent2)]" };
    if (b.chars.length || b.subjectIds.length) return { label: "Character", cls: "bg-[#3a2a14] text-[var(--color-accent2)]" };
    return { label: "shot", cls: "bg-slate-700/60 text-slate-300" };
  };
  const needsOf = (b: Block) => {
    const subj = resolveSubjects(b, libChars).length;
    if (b.renderMode === "msr" && !b.backgroundId) return { txt: "needs background", warn: true };
    if (b.renderMode === "msr" && !subj) return { txt: "needs refs", warn: true };
    if (b.renderMode === "i2v" && !b.backgroundId && !b.subjectIds.length) return { txt: "needs still", warn: true };
    if (b.renderMode === "keyframe" && !blockKeyframes(b).length) return { txt: "needs keyframes", warn: true };
    return { txt: "ready", warn: false };
  };
  const thumbOf = (b: Block) => {
    const clip = b.clipId ? library.find((i) => i.id === b.clipId) : undefined;
    if (clip?.media_url) return { url: clip.media_url, kind: "video" as const };
    const bg = b.backgroundId ? library.find((i) => i.id === b.backgroundId) : undefined;
    if (bg?.media_url) return { url: bg.media_url, kind: "img" as const };
    return null;
  };
  return (
    <div className="overflow-hidden rounded-lg border border-[var(--color-line)]">
      <div className="max-h-[460px] overflow-y-auto">
        <table className="w-full border-collapse text-left text-[11px]">
          <thead className="sticky top-0 z-10 bg-[var(--color-panel2)] text-[9px] uppercase tracking-wide text-[var(--color-muted)]">
            <tr>
              <th className="w-6 px-1 py-1.5"></th><th className="w-7 px-1 py-1.5">#</th>
              <th className="w-20 px-2 py-1.5">thumb</th><th className="w-28 px-2 py-1.5">time</th>
              <th className="w-32 px-2 py-1.5">shot</th><th className="px-2 py-1.5">scene (click to edit)</th>
              <th className="w-28 px-2 py-1.5">needs</th><th className="w-20 px-2 py-1.5">status</th><th className="w-16 px-2 py-1.5"></th>
            </tr>
          </thead>
          <tbody>
            {blocks.map((b, i) => {
              const t = typeOf(b); const n = needsOf(b); const th = thumbOf(b);
              const takes = b.clipVariants?.length || (b.clipId ? 1 : 0);
              const sel = b.id === selId;
              return (
                <tr key={b.id} onClick={() => onSelect(b.id)}
                  className={`cursor-pointer border-t border-[var(--color-line)] align-top ${sel ? "bg-[var(--color-accent)]/15" : "hover:bg-[var(--color-panel2)]"}`}>
                  <td className="px-0.5 py-1.5">
                    <div className="flex flex-col leading-none text-[var(--color-muted)]">
                      <button onClick={(e) => { e.stopPropagation(); onMove(b.id, -1); }} disabled={i === 0} title="move earlier" className="px-1 hover:text-[var(--color-ink)] disabled:opacity-25">▲</button>
                      <button onClick={(e) => { e.stopPropagation(); onMove(b.id, 1); }} disabled={i === blocks.length - 1} title="move later" className="px-1 hover:text-[var(--color-ink)] disabled:opacity-25">▼</button>
                    </div>
                  </td>
                  <td className="px-1 py-1.5 font-semibold text-[var(--color-muted)]">{b.idx + 1}</td>
                  <td className="px-2 py-1.5">
                    <div className="h-10 w-16 overflow-hidden rounded border border-[var(--color-line)] bg-[var(--color-bg)]">
                      {th ? (th.kind === "video"
                        ? <video src={posterFrag(th.url)} muted playsInline preload="metadata" className="h-full w-full object-cover" />
                        : <img src={th.url} className="h-full w-full object-cover" alt="" />)
                        : <span className="flex h-full w-full items-center justify-center text-[8px] text-[var(--color-muted)]">—</span>}
                    </div>
                  </td>
                  <td className="px-2 py-1.5 tabular-nums text-[var(--color-muted)]">{_fmt(b.start)}–{_fmt(b.end)}<span className="ml-1 opacity-60">{Math.round(b.end - b.start)}s</span></td>
                  <td className="px-2 py-1.5"><span className={`rounded px-1.5 py-0.5 ${t.cls}`}>{t.label}</span>{b.lipsync && <span title="lip-sync" className="ml-1 text-[var(--color-accent2)]">♪</span>}</td>
                  <td className="px-2 py-1.5">
                    <textarea value={b.prompt} rows={2} onClick={(e) => e.stopPropagation()}
                      onChange={(e) => onPatch(b.id, { prompt: e.target.value })}
                      placeholder="(no prompt)"
                      className="w-full min-w-[280px] resize-y rounded border border-transparent bg-transparent px-1 py-0.5 text-[11px] text-[var(--color-ink)]/85 hover:border-[var(--color-line)] focus:border-[var(--color-accent)] focus:bg-[var(--color-bg)] focus:outline-none" />
                  </td>
                  <td className="px-2 py-1.5"><span className={n.warn ? "text-amber-400" : "text-green-500"}>{n.txt}</span></td>
                  <td className="px-2 py-1.5">{takes ? <span className="text-green-400">{takes > 1 ? `${takes} takes` : "rendered"}</span> : <span className="text-[var(--color-muted)]">—</span>}</td>
                  <td className="px-2 py-1.5 text-right">
                    <button onClick={(e) => { e.stopPropagation(); onRender(b); }} disabled={busy}
                      className="rounded bg-[var(--color-accent)] px-2 py-0.5 text-[10px] font-semibold text-white disabled:opacity-50">{takes ? "Re-render" : "Render"}</button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ---- the editor ---------------------------------------------------------

export function MVStudioForm({ cfg, busy, library, song, goTo, ...ctx }:
  { cfg: Config; busy: boolean; library: LibItem[]; song: SongDraft | null; goTo: (m: string) => void } & RunCtx) {
  // The MV Studio timeline lives in this tab's drafts namespace ("mvstudio"), so it saves +
  // loads with the standard project Save/Open (top bar) exactly like every other tab - no
  // separate persistence. Characters are global (the shared /api/characters library).
  const d = useDrafts("mvstudio");
  const [audioId, setAudioId] = d.use("audioId", "");          // the song (drives lip-sync windows + assembly mux)
  const [grade, setGrade] = d.use("grade", "none");
  const [transition, setTransition] = d.use("transition", 0);  // crossfade seconds between blocks (0 = hard cut)
  const [introDur, setIntroDur] = d.use("introDur", 0);        // intro pre-roll: seconds the opening clip's own audio plays before the song (0 = off)
  const [introXfade, setIntroXfade] = d.use("introXfade", 1.5);// intro -> song crossfade seconds
  const [resolution, setResolution] = d.use("resolution", "1280x720"); // PROJECT render resolution (all shots + assembly)
  const [resW, resH] = resolution.split("x").map(Number);
  const [scriptModel, setScriptModel] = d.use("scriptModel", "claude-sonnet-4-6"); // LLM for the script writer
  const [grades, setGrades] = useState<string[]>(["none"]);
  const [scripting, setScripting] = useState(false);
  const [blocks, setBlocks] = d.use<Block[]>("blocks", []);
  const [selId, setSelId] = d.use("selId", "");
  const [libChars, setLibChars] = useState<Character[]>([]);
  const reloadChars = () => api.characters().then((r) => setLibChars(r as Character[])).catch(() => {});
  // which inspector sub-sections are expanded (per selected block is overkill; keep global)
  const [open, setOpen] = d.use<Record<string, boolean>>("openSecs", { refs: true, prompt: true });
  const toggle = (k: string) => setOpen({ ...open, [k]: !open[k] });

  const [beats, setBeats] = useState<number[]>([]);              // song beat times (for snap-to-beat)
  useEffect(() => { api.mvGrades().then((r) => setGrades((r as { grades: string[] }).grades)).catch(() => {}); }, []);
  useEffect(() => { reloadChars(); }, []);
  useEffect(() => {
    if (!audioId) { setBeats([]); return; }
    api.beats(audioId).then((r) => setBeats(((r as { beats?: number[] }).beats) || [])).catch(() => setBeats([]));
  }, [audioId]);

  const stills = library.filter((i) => i.mode === "videostill" && i.media_url);
  const audios = library.filter((i) => i.audio_url);
  const audioUrl = library.find((i) => i.id === audioId)?.audio_url || "";
  const sel = blocks.find((b) => b.id === selId);

  // ---- block ops (blocks are kept sorted by start time = the timeline + assembly order) ----
  const sortBlocks = (list: Block[]) =>
    [...list].sort((a, b) => (a.start - b.start) || (a.end - b.end)).map((b, i) => ({ ...b, idx: i }));
  function commit(list: Block[]) { setBlocks(sortBlocks(list)); }
  // FUNCTIONAL update: patch must read the LATEST blocks, not a stale render-closure snapshot - otherwise
  // rapid concurrent patches (e.g. "Render all stills" assigning 30 backgrounds in a loop) clobber each
  // other and only the last sticks. Using the updater form makes them compose.
  function patch(id: string, p: Partial<Block>) {
    setBlocks((prev) => sortBlocks(prev.map((b) => b.id === id ? { ...b, ...p } : b)));
  }
  function patchSel(p: Partial<Block>) { if (sel) patch(sel.id, p); }
  function addBlock() {
    const last = [...blocks].sort((a, b) => a.end - b.end)[blocks.length - 1];
    const start = last ? last.end : 0;
    const b = makeBlock(start);
    b.audioId = audioId;
    commit([...blocks, b]); setSelId(b.id);
  }
  function dupBlock(b: Block) {
    const copy: Block = { ...b, id: rid(), clipId: undefined, start: b.end, end: b.end + (b.end - b.start),
      segs: b.segs.map((s) => ({ ...s })), chars: b.chars.map((c) => ({ ...c })), subjectIds: [...b.subjectIds] };
    commit([...blocks, copy]); setSelId(copy.id);
  }
  // ---- export / import the timeline as a portable JSON shot list (move a timeline between projects) ----
  function exportTimeline() {
    const blob = new Blob([JSON.stringify({ version: 1, blocks }, null, 2)], { type: "application/json" });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
    a.download = `mv-timeline-${Date.now()}.json`; a.click(); URL.revokeObjectURL(a.href);
  }
  function importTimeline(file: File) {
    const r = new FileReader();
    r.onload = () => {
      try {
        const parsed = JSON.parse(String(r.result));
        const arr = Array.isArray(parsed) ? parsed : parsed.blocks;
        if (!Array.isArray(arr)) throw new Error("no blocks array in file");
        commit(arr.map((b: Partial<Block>, i: number) => ({ ...hydrateBlock(b, i), id: rid() })));
      } catch (e) { alert("Import failed: " + (e as Error).message); }
    };
    r.readAsText(file);
  }
  function delBlock(id: string) {
    const next = blocks.filter((b) => b.id !== id); commit(next);
    if (selId === id) setSelId(next[0]?.id || "");
  }
  // reorder: swap this block's CONTENT with its neighbour, keeping each time slot (the timeline is
  // time-locked to the song, so "reorder" = which shot plays in which window). Frames re-fit per slot.
  function move(id: string, dir: -1 | 1) {
    const i = blocks.findIndex((b) => b.id === id); const j = i + dir;
    if (i < 0 || j < 0 || j >= blocks.length) return;
    const a = blocks[i], b = blocks[j];
    // content (incl. any rendered clip) swaps slots; the time window + audioStart stay with the slot,
    // frames re-fit to the new slot. A lip-sync clip moved to a new window may need re-rendering.
    const slot = (x: Block) => ({ id: x.id, start: x.start, end: x.end, audioStart: x.start,
      frames: ltxFrames(Math.max(2, x.end - x.start) * x.fps) });
    const newA = { ...b, ...slot(a) }, newB = { ...a, ...slot(b) };
    commit(blocks.map((x) => x.id === a.id ? newA : x.id === b.id ? newB : x));
  }

  // ---- generate a starting timeline from the project's song arrangement ----
  const songTitle = library.find((i) => i.id === audioId)?.params?.title || "";
  const canScript = !!(song && song.blocks && song.blocks.length);
  async function generateScript() {
    if (!canScript || !song) { ctx.setResults([{ id: rid(), title: "No song arrangement", status: "error", pct: 0, err: "Open a project with a Song arrangement (the Song tab) to script from it." }]); return; }
    if (blocks.length && !window.confirm(`Replace the current ${blocks.length} blocks with a generated timeline?`)) return;
    setScripting(true);
    try {
      const payload = { title: String(songTitle || ""), tags: song.tags, bpm: song.bpm, keyscale: song.key,
        sections: song.blocks.map((b) => ({ type: b.type, seconds: b.seconds, lyrics: b.lyrics })) };
      // audio_id lets the backend snap the cuts onto the song's ACTUAL structure (allin1 segments +
      // downbeats from the rendered audio) - more accurate than the planned arrangement.
      const r = await api.mvScript({ song: payload, model: scriptModel, audio_id: audioId,
        cast: libChars.map((c) => ({ name: c.name, role: c.role || "", kind: c.kind || "musician" })) }) as { shots: ScriptShot[] };
      const next = (r.shots || []).map((s) => shotToBlock(s, libChars, audioId));
      commit(next); setSelId(next[0]?.id || "");
    } catch (e) {
      ctx.setResults([{ id: rid(), title: "Script generation failed", status: "error", pct: 0, err: (e as Error).message }]);
    } finally { setScripting(false); }
  }

  // ---- per-block SeedVR2 upscale of a rendered clip ----
  async function upscaleBlock(b: Block) {
    if (!b.clipId) return;
    const card = { id: rid(), title: `block ${b.idx + 1}: upscaling...`, status: "running" as const, pct: 5 };
    ctx.setResults([card]);
    try {
      const { job_id } = await api.videoFlashvsr({ video_id: b.clipId, scale: 2 }) as { job_id: string };
      patch(b.id, { upscaledId: job_id });
      pollJob(job_id, card.id, ctx);
    } catch (e) { ctx.patch(card.id, { status: "error", pct: 0, err: (e as Error).message }); }
  }

  // ---- compose a block's background by placing characters into a scene (Qwen char_still) ----
  // the band-member pipeline: composite 1-3 distant characters (e.g. guitarist/bassist holding
  // their instruments) into a backdrop, used as the MSR background while the singer drives MSR.
  async function composeBackground(blockId: string, charIds: string[], prompt: string) {
    const refs = charIds.map((id) => sceneRefOf(libChars.find((c) => c.id === id))).filter(Boolean).slice(0, 3);
    const fail = (err: string) => ctx.setResults([{ id: rid(), title: "Compose scene", status: "error", pct: 0, err }]);
    if (!refs.length) return fail("Pick at least one character that has a reference still.");
    if (!prompt.trim()) return fail("Describe the scene (where each character stands, the setting, the lighting).");
    const card = { id: rid(), title: "composing scene background...", status: "pending" as const, pct: 0 };
    ctx.setResults([card]);
    try {
      const { job_id } = await api.videoCharStill({ ref_ids: refs, prompt }) as { job_id: string };
      const blk = blocks.find((x) => x.id === blockId);
      patch(blockId, { backgroundId: job_id, bgVariants: [...(blk?.bgVariants || []), job_id] });
      ctx.patch(card.id, { status: "running", pct: 5 });
      pollJob(job_id, card.id, ctx);
    } catch (e) { ctx.patch(card.id, { status: "error", pct: 0, err: (e as Error).message }); }
  }

  // ---- render one block ----
  async function genBlock(b: Block) {
    const subjectIds = resolveSubjects(b, libChars);
    const fail = (err: string) => ctx.setResults([{ id: rid(), title: `block ${b.idx + 1}`, status: "error", pct: 0, err }]);
    if (b.renderMode === "msr") {
      if (!subjectIds.length) return fail("Pick at least one subject reference (a character with refs, or a still).");
      if (!b.backgroundId) return fail("Pick a background still for the MSR scene.");
    }
    if (b.renderMode === "keyframe" && !blockKeyframes(b).length)
      return fail("Keyframe mode needs at least one segment with a keyframe still (set a still on a segment).");
    const card = { id: rid(), title: `block ${b.idx + 1} (${b.renderMode})`, status: "pending" as const, pct: 0 };
    ctx.setResults([card]);
    try {
      let job_id: string;
      if (b.renderMode === "msr") {
        ({ job_id } = await api.videoLtxMsr({ ...msrPayload(b, subjectIds, audioId), width: resW, height: resH }) as { job_id: string });
      } else if (b.renderMode === "keyframe") {
        ({ job_id } = await api.videoLtxKeyframe({ ...keyframePayload(b), width: resW, height: resH }) as { job_id: string });
      } else if (b.renderMode === "s2v") {
        const still = subjectIds[0] || b.backgroundId;
        if (!still) return fail("S2V needs a reference still.");
        if (!(b.audioId || audioId)) return fail("S2V needs a song track for lip-sync.");
        ({ job_id } = await api.videoS2v({ still_id: still, audio_id: b.audioId || audioId, prompt: b.prompt, audio_start: b.audioStart }) as { job_id: string });
      } else {                                  // i2v (anchor): animate the first subject/background still
        const still = subjectIds[0] || b.backgroundId;
        if (!still) return fail("i2v needs a still to animate.");
        const frames = ltxFrames(Math.max(2, b.end - b.start) * b.fps);
        ({ job_id } = await (cfg.video_ltx
          ? api.videoLtxI2V({ still_id: still, prompt: b.prompt, frames, width: resW, height: resH })
          : api.videoI2V({ still_id: still, prompt: b.prompt, length: frames })) as { job_id: string });
      }
      patch(b.id, { clipId: job_id, clipVariants: [...(b.clipVariants || []), job_id] });
      ctx.patch(card.id, { status: "running", pct: 5 });
      pollJob(job_id, card.id, ctx);
    } catch (e) { ctx.patch(card.id, { status: "error", pct: 0, err: (e as Error).message }); }
  }

  // Retake a [start, start+length] second slice of this block's rendered clip (LTXDirectorGuide
  // retake_mode). Result is added as a new render take. For lip-sync blocks the clip's vocal window is
  // sent so the slice regenerates against the song. NOTE (known limits): the node re-decodes the whole
  // clip, and the slice can pop at the seam - kept for when a quick local fix is good enough.
  async function retakeRegion(b: Block, startS: number, lengthS: number, prompt: string, strength: number) {
    const fail = (err: string) => ctx.setResults([{ id: rid(), title: `block ${b.idx + 1} retake`, status: "error", pct: 0, err }]);
    if (!b.clipId) return fail("Render the block first - retake works on an existing clip.");
    if (!prompt.trim()) return fail("Describe what the retaken slice should show.");
    const card = { id: rid(), title: `block ${b.idx + 1} retake ${startS}-${startS + lengthS}s`, status: "pending" as const, pct: 0 };
    ctx.setResults([card]);
    try {
      const payload: Record<string, unknown> = {
        clip_id: b.clipId, retake_start: startS, retake_length: lengthS, prompt: prompt.trim(), retake_strength: strength,
      };
      if (b.lipsync) { payload.audio_id = b.audioId || audioId; payload.audio_start = b.audioStart; payload.isolate_vocal = b.isolateVocal; }
      const { job_id } = await api.videoLtxRetake(payload) as { job_id: string };
      patch(b.id, { clipId: job_id, clipVariants: [...(b.clipVariants || []), job_id] });
      ctx.patch(card.id, { status: "running", pct: 5 });
      pollJob(job_id, card.id, ctx);
    } catch (e) { ctx.patch(card.id, { status: "error", pct: 0, err: (e as Error).message }); }
  }

  // ---- bulk render: fire every renderable block (the box queues them). onlyMissing skips rendered. ----
  function blockReady(b: Block) {
    const subj = resolveSubjects(b, libChars).length;
    if (b.renderMode === "keyframe") return !!blockKeyframes(b).length;
    return b.renderMode === "i2v" ? !!(b.backgroundId || b.subjectIds.length) : !!(b.backgroundId && subj);
  }
  async function renderAll(onlyMissing: boolean) {
    const todo = blocks.filter((b) => blockReady(b) && (!onlyMissing || !b.clipId));
    if (!todo.length) { ctx.setResults([{ id: rid(), title: "Bulk render", status: "error", pct: 0, err: "No renderable blocks (generate the backgrounds/refs they need first)." }]); return; }
    for (const b of todo) await genBlock(b);   // genBlock submits + polls; the box processes them in order
  }

  // ---- generate a block's BACKGROUND scene still (Z-Image) from its prompt. For MSR blocks the
  // character is an MSR subject (added separately), so bias the still to an empty scene. ----
  async function genStill(b: Block) {
    // The MSR background must NOT contain the featured/MSR subject (a person there corrupts identity),
    // but a BAND shot DOES need the other band members composited in. Frame the bg to MATCH the shot
    // framing (a wide bg forces the composite wide, which breaks lip-sync close-ups).
    const bgFrame = ({ close: "a tight, close framing of", medium: "a medium framing of", wide: "a wide establishing shot of" } as Record<string, string>)[b.framing || "medium"] || "a medium framing of";
    const card = { id: rid(), title: `block ${b.idx + 1}: background still...`, status: "pending" as const, pct: 0 };
    const setBg = (job_id: string) => {
      const blk = blocks.find((x) => x.id === b.id);
      patch(b.id, { backgroundId: job_id, bgVariants: [...(blk?.bgVariants || []), job_id] });
      ctx.patch(card.id, { status: "running", pct: 5 });
      pollJob(job_id, card.id, ctx);
    };
    try {
      // BAND shot: composite the non-featured band members into the scene via Qwen (char_still). They are
      // a STATIC background image (only the featured singer is MSR-anchored).
      const bandRefs = b.renderMode === "msr"
        ? b.bandInScene.map((id) => sceneRefOf(libChars.find((c) => c.id === id))).filter(Boolean).slice(0, 3)
        : [];
      if (bandRefs.length) {
        ctx.setResults([card]);
        const prompt = `${(b.scene || b.prompt).trim()}. The band members standing together deep in the background of the scene (${bgFrame} the setting). Photoreal, no other people, no extra figures, no crowd.`;
        const { job_id } = await api.videoCharStill({ ref_ids: bandRefs, prompt }) as { job_id: string };
        setBg(job_id);
        return;
      }
      // SOLO / B-roll: person-free environment still.
      const scene = b.renderMode === "msr"
        ? `${bgFrame} the setting: ${(b.scene || b.prompt).trim()}. The empty environment only - absolutely no people, no person, no singer, no musician, no figure, no face, unpopulated.`
        : (b.prompt || "");
      if (!scene.trim()) return;
      ctx.setResults([card]);
      const neg = b.renderMode === "msr" ? "people, person, man, woman, singer, musician, performer, face, figure, silhouette, crowd, " + (b.negative || "") : (b.negative || undefined);
      const { job_id } = await api.videoStill({ prompt: scene, negative: neg, width: resW, height: resH }) as { job_id: string };
      setBg(job_id);
    } catch (e) { ctx.patch(card.id, { status: "error", pct: 0, err: (e as Error).message }); }
  }
  async function renderAllStills(onlyMissing: boolean) {
    const todo = blocks.filter((b) => (b.prompt || "").trim() && (!onlyMissing || !b.backgroundId));
    if (!todo.length) { ctx.setResults([{ id: rid(), title: "Render stills", status: "error", pct: 0, err: "Every block already has a background still." }]); return; }
    for (const b of todo) await genStill(b);   // submits + polls each; the box processes them in order
  }

  // ---- assemble ----
  async function assemble() {
    const ready = blocks.filter((b) => b.clipId);
    if (!ready.length) { ctx.setResults([{ id: rid(), title: "Nothing to assemble", status: "error", pct: 0, err: "Render some blocks first." }]); return; }
    const card = { id: rid(), title: `assembling ${ready.length} blocks...`, status: "running" as const, pct: 30 };
    ctx.setResults([card]);
    try {
      const introClip = blocks[0]?.upscaledId || blocks[0]?.clipId;   // the opening shot's render = the wind pre-roll
      const r = await api.mvAssemble({ shots: ready.map((b) => ({ clip_id: b.upscaledId || b.clipId, start: b.start, end: b.end })),
        audio_id: audioId, grade, transition, title: "music video", width: resW, height: resH,
        ...(introDur > 0 && introClip ? { intro_clip_id: introClip, intro_dur: introDur, intro_xfade: introXfade } : {}) }) as { media_url: string };
      ctx.patch(card.id, { status: "done", pct: 100, url: r.media_url + "?t=" + Date.now(), media: "video" });
      ctx.onDone();
    } catch (e) { ctx.patch(card.id, { status: "error", pct: 0, err: (e as Error).message }); }
  }

  if (!cfg.video_msr) {
    return <div className="rounded-xl border border-[var(--color-line)] bg-[var(--color-panel2)] p-4 text-xs text-[var(--color-muted)]">
      LTX MSR not detected on the box. MV Studio needs the LTX-2.3 backbone plus the Licon-MSR and PromptRelay custom nodes. Install them + restart ComfyUI, then restart the backend.
    </div>;
  }

  const span = Math.max(30, ...blocks.map((b) => b.end));
  const ready = blocks.filter((b) => b.clipId).length;

  return (
    <div className="space-y-4">
      <p className="text-[11px] text-[var(--color-muted)]">
        Build a music video as a timeline of <span className="text-[var(--color-ink)]">LTX MSR blocks</span> -
        each block holds a character's identity from reference stills and is fully prompt-driven (walk, sing,
        camera moves), with native single-pass lip-sync. This timeline is part of your project: it saves +
        loads with <span className="text-[var(--color-ink)]">Save</span> / <span className="text-[var(--color-ink)]">Open</span> in the project bar above.
      </p>

      {/* song audio (a library track ref - persists in the project drafts) */}
      <Field label="Song audio" hint="drives lip-sync windows + the final mux">
        <select className={inp} value={audioId} onChange={(e) => setAudioId(e.target.value)}>
          <option value="">- pick a track -</option>
          {audios.map((a) => <option key={a.id} value={a.id}>{(a.params?.title || a.params?.tags || a.mode || a.id).toString().slice(0, 40)}</option>)}
        </select>
      </Field>

      {/* quick character panel (same shared global cast as the Characters tab) */}
      <CharacterLibrary chars={libChars} setChars={setLibChars} reload={reloadChars}
        stills={stills} busy={busy} collapsible {...ctx} />
      <button onClick={() => goTo("characters")} className="text-[10px] text-[var(--color-muted)] hover:text-[var(--color-ink)] underline">
        Manage the full cast in the Characters tab →
      </button>

      <div className="flex flex-wrap items-center gap-2">
        <GhostButton onClick={addBlock}>+ Block</GhostButton>
        <GhostButton onClick={generateScript} disabled={scripting || !canScript}>
          {scripting ? "Scripting..." : "Generate from song"}
        </GhostButton>
        <select className={inp} style={{ width: "auto" }} value={scriptModel} onChange={(e) => setScriptModel(e.target.value)}
          title="Which Claude model writes the shot list" disabled={scripting}>
          <option value="claude-sonnet-4-6">Sonnet 4.6</option>
          <option value="opus">Opus</option>
        </select>
        {!canScript && <span className="text-[9px] text-[var(--color-muted)]">(open a project with a Song arrangement to script)</span>}
        {blocks.length > 0 && (
          <GhostButton onClick={() => renderAllStills(true)} disabled={busy}>
            Render all stills
          </GhostButton>
        )}
        {blocks.length > 0 && (
          <GhostButton onClick={() => renderAll(true)} disabled={busy}>
            Render all videos
          </GhostButton>
        )}
        {blocks.length > 0 && <GhostButton onClick={exportTimeline}>Export</GhostButton>}
        <label className="cursor-pointer rounded-md border border-[var(--color-line)] px-2.5 py-1 text-[11px] text-[var(--color-muted)] hover:text-[var(--color-ink)]" title="Import a timeline JSON shot list">
          Import
          <input type="file" accept="application/json" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) importTimeline(f); e.currentTarget.value = ""; }} />
        </label>
        <label className="ml-auto flex items-center gap-1.5 text-[11px] text-[var(--color-muted)]" title="Render resolution for ALL shots + the final video (project-wide)">
          Resolution
          <select className={inp} style={{ width: "auto" }} value={resolution} onChange={(e) => setResolution(e.target.value)}>
            <option value="832x480">832×480 (draft, fast)</option>
            <option value="1280x720">1280×720 (HD)</option>
            <option value="1920x1080">1920×1080 (full HD, slow)</option>
          </select>
        </label>
        <span className="text-[10px] text-[var(--color-muted)]">{blocks.length} blocks {"·"} {ready} rendered</span>
      </div>

      {/* the visual timeline: waveform + draggable blocks when a song is set; else a
          proportional click-to-select bar (no audio to scrub against) */}
      {blocks.length > 0 && audioUrl && (
        <MVTimeline url={audioUrl} beats={beats} blocks={blocks} selId={selId} fps={blocks[0]?.fps || 24}
          onSelect={setSelId} onChange={(id, s, e) => patch(id, { start: s, end: e })}
          onDelete={delBlock} onDuplicate={(id) => { const b = blocks.find((x) => x.id === id); if (b) dupBlock(b); }} />
      )}
      {blocks.length > 0 && !audioUrl && (
        <div className="relative h-16 w-full overflow-hidden rounded-lg border border-[var(--color-line)] bg-[var(--color-bg)]">
          {blocks.map((b) => {
            const left = (b.start / span) * 100, width = Math.max(2, ((b.end - b.start) / span) * 100);
            const isSel = b.id === selId;
            return (
              <button key={b.id} onClick={() => setSelId(b.id)} title={`block ${b.idx + 1}: ${b.start}-${b.end}s`}
                className={`absolute top-1 bottom-1 overflow-hidden rounded px-1 text-left text-[9px] leading-tight transition ${isSel ? "ring-2 ring-[var(--color-accent)] z-10" : "hover:brightness-125"} ${b.clipId ? "bg-green-900/60 text-green-200" : b.lipsync ? "bg-[#3a2a14] text-[var(--color-accent2)]" : "bg-[var(--color-panel2)] text-[var(--color-muted)]"}`}
                style={{ left: `${left}%`, width: `${width}%` }}>
                <span className="font-semibold">{b.idx + 1}</span>{b.lipsync && <span title="lip-sync"> {"♪"}</span>}
                <div className="truncate opacity-80">{b.prompt.slice(0, 24) || "(no prompt)"}</div>
              </button>
            );
          })}
        </div>
      )}

      {/* readable shot list - every block at a glance, click a row to edit it below */}
      {blocks.length > 0 && (
        <BlockList blocks={blocks} selId={selId} onSelect={setSelId} onRender={genBlock}
          onPatch={patch} onMove={move} libChars={libChars} library={library} busy={busy} />
      )}

      {blocks.length === 0 && (
        <div className="rounded-lg border border-dashed border-[var(--color-line)] p-6 text-center text-xs text-[var(--color-muted)]">
          No blocks yet. Click <span className="text-[var(--color-ink)]">+ Block</span> to start the timeline.
        </div>
      )}

      {/* inspector for the selected block */}
      {sel && <Inspector key={sel.id} b={sel} idx={sel.idx}
        cfg={cfg} busy={busy} stills={stills} audios={audios} library={library} libChars={libChars} songAudioId={audioId}
        url={audioUrl} beats={beats}
        open={open} toggle={toggle}
        patch={patchSel} gen={() => genBlock(sel)} dup={() => dupBlock(sel)}
        del={() => delBlock(sel.id)} upscale={() => upscaleBlock(sel)}
        retake={(s, l, p, st) => retakeRegion(sel, s, l, p, st)}
        compose={(charIds, prompt) => composeBackground(sel.id, charIds, prompt)} />}

      {/* intro pre-roll: the opening shot's own (LTX-generated) audio plays before the song, then crossfades out */}
      {blocks.length > 0 && (
        <div className="flex flex-wrap items-center gap-2 rounded-lg border border-[var(--color-line)] bg-[var(--color-bg)] p-2 text-[11px] text-[var(--color-muted)]">
          <span className="font-semibold text-[var(--color-ink)]">Intro pre-roll</span>
          <span className="opacity-80">opening shot's wind plays before the song, then crossfades out</span>
          <label className="ml-auto flex items-center gap-1" title="seconds the opening clip's own audio plays before the song starts (0 = off)">
            pre-roll
            <input type="number" min={0} max={10} step={0.5} value={introDur} onChange={(e) => setIntroDur(Number(e.target.value))} className={`${inp} w-16`} />s
          </label>
          <label className="flex items-center gap-1" title="wind -> song crossfade seconds">
            crossfade
            <input type="number" min={0.2} max={5} step={0.1} value={introXfade} onChange={(e) => setIntroXfade(Number(e.target.value))} className={`${inp} w-16`} />s
          </label>
          {introDur > 0 && !(blocks[0]?.clipId) && <span className="text-amber-400">render the opening shot first</span>}
        </div>
      )}

      {/* assemble */}
      {blocks.length > 0 && (
        <div className="flex items-center gap-2 border-t border-[var(--color-line)] pt-3">
          <PrimaryButton onClick={assemble} disabled={busy || ready === 0}>
            {`Assemble video (${ready}/${blocks.length} blocks)`}
          </PrimaryButton>
          <label className="ml-auto flex items-center gap-1.5 text-[11px] text-[var(--color-muted)]" title="Crossfade duration blended between consecutive blocks (0 = hard cut)">
            Transition
            <select className={inp} style={{ width: "auto" }} value={transition} onChange={(e) => setTransition(Number(e.target.value))}>
              <option value={0}>hard cut</option>
              <option value={0.3}>crossfade 0.3s</option>
              <option value={0.5}>crossfade 0.5s</option>
              <option value={1}>crossfade 1.0s</option>
            </select>
          </label>
          <label className="flex items-center gap-1.5 text-[11px] text-[var(--color-muted)]" title="Color grade applied to every block for a consistent look (GPU-free)">
            Grade
            <select className={inp} style={{ width: "auto" }} value={grade} onChange={(e) => setGrade(e.target.value)}>
              {grades.map((g) => <option key={g} value={g}>{g.replace(/_/g, " ")}</option>)}
            </select>
          </label>
        </div>
      )}
    </div>
  );
}

// ---- the per-block inspector -------------------------------------------

function Inspector({ b, idx, cfg, busy, stills, audios, library, libChars, songAudioId, url, beats, patch, gen, dup, del, upscale, retake, compose }: {
  b: Block; idx: number; cfg: Config; busy: boolean;
  stills: LibItem[]; audios: LibItem[]; library: LibItem[]; libChars: Character[]; songAudioId: string;
  url: string; beats: number[];
  open: Record<string, boolean>; toggle: (k: string) => void;
  patch: (p: Partial<Block>) => void; gen: () => void; dup: () => void; del: () => void; upscale: () => void;
  retake: (startS: number, lengthS: number, prompt: string, strength: number) => void;
  compose: (charIds: string[], prompt: string) => void;
}) {
  const [sceneOpen, setSceneOpen] = useState(false);
  const [selSeg, setSelSeg] = useState(0);
  const [rStart, setRStart] = useState(0);
  const [rLen, setRLen] = useState(1);
  const [rStrength, setRStrength] = useState(1);
  const [rPrompt, setRPrompt] = useState("");
  const [sceneChars, setSceneChars] = useState<string[]>([]);
  const [scenePrompt, setScenePrompt] = useState("");
  const toggleSceneChar = (id: string) => setSceneChars(
    sceneChars.includes(id) ? sceneChars.filter((x) => x !== id) : sceneChars.length < 3 ? [...sceneChars, id] : sceneChars);
  const subjects = resolveSubjects(b, libChars);
  const clip = stills.find((s) => s.id === b.clipId);    // (clip is a videoclip, not a still; thumb best-effort)
  const heroCount = b.chars.length;

  // segment editor helpers
  const setSeg = (i: number, p: Partial<Seg>) => patch({ segs: b.segs.map((s, j) => j === i ? { ...s, ...p } : s) });
  const addSeg = () => patch({ segs: [...b.segs, { len: "", prompt: "" }] });
  const delSeg = (i: number) => patch({ segs: b.segs.filter((_, j) => j !== i) });

  const setChar = (i: number, p: { charId?: string; wardrobeId?: string }) =>
    patch({ chars: b.chars.map((c, j) => j === i ? { ...c, ...p } : c) });
  const addChar = () => { if (heroCount < 2) patch({ chars: [...b.chars, { charId: "" }] }); };
  const delChar = (i: number) => patch({ chars: b.chars.filter((_, j) => j !== i) });

  const modes: { v: RenderMode; label: string }[] = [
    { v: "msr", label: "Performance (MSR)" },
    { v: "keyframe", label: "B-roll (keyframe)" },
    { v: "i2v", label: cfg.video_ltx ? "i2v" : "i2v - Wan" },
    { v: "s2v", label: "S2V" },
  ];
  return (
    <div className="mvi mvi-root">
      <style>{`
        .mvi.mvi-root{background:#0c0e13;border:1px solid rgba(255,255,255,.08);border-radius:16px;padding:16px;display:flex;flex-direction:column;gap:12px;}
        .mvi-hdr{display:flex;flex-wrap:wrap;align-items:center;gap:10px;}
        .mvi-title{font-size:16px;font-weight:500;color:#eef0f4;}
        .mvi-card{background:#16191f;border:1px solid rgba(255,255,255,.07);border-radius:14px;padding:14px;}
        .mvi-h{font-size:13px;font-weight:500;color:#e9ebef;margin:0 0 10px;display:flex;align-items:center;gap:8px;}
        .mvi-h .sub{font-weight:400;color:#9aa1ad;font-size:12px;}
        .mvi-sub{font-size:11px;font-weight:500;color:#8b92a0;letter-spacing:.04em;text-transform:uppercase;margin:0 0 6px;}
        .mvi-row{display:flex;flex-wrap:wrap;align-items:flex-end;gap:12px;}
        .mvi select,.mvi input:not([type=range]):not([type=checkbox]),.mvi textarea{background:#0c0e13;border:1px solid rgba(255,255,255,.10);border-radius:8px;color:#e7e8ec;font-size:13px;padding:8px 10px;}
        .mvi textarea:focus,.mvi select:focus,.mvi input:focus{outline:none;border-color:#EF9F27;}
        .mvi-tog{display:inline-flex;border:1px solid rgba(255,255,255,.12);border-radius:9px;overflow:hidden;}
        .mvi-tog button{border:0;padding:6px 12px;font-size:12px;background:transparent;color:#9aa1ad;cursor:pointer;}
        .mvi-tog button.on{background:#EF9F27;color:#3a2402;font-weight:500;}
        .mvi-row input[type=number]{width:104px;flex:0 0 auto;}
        .mvi-row select{flex:0 0 auto;}
        .mvi-row > label{flex:0 0 auto;}
        .mvi-btn{border:1px solid rgba(255,255,255,.14);background:transparent;color:#9aa1ad;border-radius:8px;padding:5px 10px;font-size:12px;cursor:pointer;}
        .mvi-btn:hover:not(:disabled){color:#e7e8ec;background:#1d212a;}
        .mvi-btn:disabled{opacity:.45;}
        .mvi-render{border:0;background:#D85A30;color:#fff;border-radius:8px;padding:7px 16px;font-size:13px;font-weight:500;cursor:pointer;}
        .mvi-render:disabled{opacity:.5;}
        .mvi-ok{font-size:11px;color:#5dca8b;}
        .mvi-acc{font-size:11px;color:#EF9F27;}
        .mvi-link{background:none;border:0;color:#EF9F27;font-size:11px;cursor:pointer;padding:0;}
        .mvi-muted{font-size:11px;color:#8b92a0;}
        .mvi-thumb{height:56px;width:96px;flex:none;overflow:hidden;border-radius:8px;border:1px solid rgba(255,255,255,.12);}
        .mvi-thumb.on{border-color:#EF9F27;box-shadow:0 0 0 1px #EF9F27;}
        .mvi-thumb img,.mvi-thumb video{height:100%;width:100%;object-fit:cover;}
        .mvi-divide{border-top:1px solid rgba(255,255,255,.07);margin-top:12px;padding-top:12px;}
      `}</style>

      {/* header */}
      <div className="mvi-hdr">
        <span className="mvi-title">Shot {idx + 1}</span>
        <div className="mvi-tog" title="render mode">
          {modes.map((m) => (
            <button key={m.v} className={b.renderMode === m.v ? "on" : ""}
              onClick={() => patch({ renderMode: m.v })}>{m.label}</button>
          ))}
        </div>
        <label className="mvi-muted" style={{ display: "flex", alignItems: "center", gap: 5 }} title="native single-pass lip-sync to the song vocal">
          <input type="checkbox" checked={b.lipsync} onChange={(e) => patch({ lipsync: e.target.checked })} /> lip-sync
        </label>
        {b.clipId && <span className="mvi-ok" title="rendered">rendered</span>}
        {b.upscaledId && <span className="mvi-acc" title="FlashVSR 2x upscaled">{"↑"}2x</span>}
        <span style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 6 }}>
          {b.clipId && <button className="mvi-btn" onClick={upscale} disabled={busy} title="FlashVSR 2x upscale (auto-chunks long clips)">{b.upscaledId ? "Re-upscale" : "Upscale"}</button>}
          <button className="mvi-btn" onClick={dup} title="Duplicate">{"⧉"}</button>
          <button className="mvi-btn" onClick={del} title="Delete">{"×"}</button>
          <button className="mvi-render" onClick={gen} disabled={busy}>{b.clipId ? "Re-render" : "Render"}</button>
        </span>
      </div>

      {/* timing */}
      <div className="mvi-card mvi-row">
        <Num label="start (s)" value={b.start} set={(n) => patch({ start: n })} />
        <Num label="end (s)" value={b.end} set={(n) => patch({ end: n })} />
        <Num label="frames" value={b.frames} set={(n) => patch({ frames: n })} w="w-24" title="LTX coerces to 8k+1" />
        <Num label="fps" value={b.fps} set={(n) => patch({ fps: n })} w="w-16" />
        <span className="mvi-muted" style={{ paddingBottom: 6 }}>
          {"≈"} {blockSeconds(b)}s render
          <button className="mvi-link" style={{ marginLeft: 6, textDecoration: "underline" }} title="set frames from the block length"
            onClick={() => patch({ frames: ltxFrames(Math.max(2, b.end - b.start) * b.fps) })}>fit to {Math.max(0, b.end - b.start)}s</button>
        </span>
      </div>

      {/* variant takes */}
      {(!!b.bgVariants?.length || !!b.clipVariants?.length) && (
        <div className="mvi-card">
          {!!b.bgVariants?.length && (
            <div>
              <p className="mvi-sub">Background takes ({b.bgVariants.length}) {"·"} click to use</p>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                {b.bgVariants.map((id) => {
                  const it = stills.find((s) => s.id === id); const on = b.backgroundId === id;
                  return <button key={id} onClick={() => patch({ backgroundId: id })} title={on ? "in use" : "use this background"}
                    className={`mvi-thumb ${on ? "on" : ""}`} style={{ opacity: on ? 1 : 0.7 }}>
                    {it?.media_url ? <img src={it.media_url} alt="" /> : <span className="mvi-muted">{id.slice(0, 6)}</span>}
                  </button>;
                })}
              </div>
            </div>
          )}
          {!!b.clipVariants?.length && (
            <div className={b.bgVariants?.length ? "mvi-divide" : ""}>
              <p className="mvi-sub">Render takes ({b.clipVariants.length}) {"·"} click to keep</p>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                {b.clipVariants.map((id) => {
                  const it = library.find((s) => s.id === id); const on = b.clipId === id;
                  return <button key={id} onClick={() => patch({ clipId: id })} title={on ? "kept" : "keep this take"}
                    className={`mvi-thumb ${on ? "on" : ""}`} style={{ opacity: on ? 1 : 0.7 }}>
                    {it?.media_url ? <video src={posterFrag(it.media_url)} muted playsInline preload="metadata" /> : <span className="mvi-muted">{id.slice(0, 6)}</span>}
                  </button>;
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* references + prompt row (horizontal cards; prompt goes full-width when no references) */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 12, alignItems: "start" }}>
      {b.renderMode === "msr" && (
        <div className="mvi-card">
          <div className="mvi-h">References <span className="sub">{subjects.length} / 4 slots</span>
            <button className="mvi-link" style={{ marginLeft: "auto" }} onClick={addChar} disabled={heroCount >= 2}>+ character</button>
          </div>
          <p className="mvi-sub">Hero characters ({heroCount}/2)</p>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {b.chars.map((bc, i) => {
              const c = libChars.find((x) => x.id === bc.charId);
              const refs = charRefIds(c, bc.wardrobeId);
              return (
                <div key={i} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <select className={inp} value={bc.charId} onChange={(e) => setChar(i, { charId: e.target.value, wardrobeId: undefined })}>
                    <option value="">- character -</option>
                    {libChars.map((x) => <option key={x.id} value={x.id}>{x.name}</option>)}
                  </select>
                  <select className={`${inp} w-40`} value={bc.wardrobeId || ""} title="wardrobe / look"
                    onChange={(e) => setChar(i, { wardrobeId: e.target.value || undefined })} disabled={!c?.wardrobes?.length}>
                    <option value="">{c?.wardrobes?.length ? "default look" : "(no wardrobes)"}</option>
                    {(c?.wardrobes || []).map((w) => <option key={w.id} value={w.id}>{w.name}</option>)}
                  </select>
                  <span className="mvi-muted" style={{ width: 48, flexShrink: 0 }} title="reference stills this character contributes">{refs.length} ref{refs.length === 1 ? "" : "s"}</span>
                  <button className="mvi-btn" onClick={() => delChar(i)} title="remove">{"×"}</button>
                </div>
              );
            })}
            {heroCount === 0 && <p className="mvi-muted">No characters set. Add a character (resolves its face + body refs), or pick subject stills manually below.</p>}
          </div>

          <div className="mvi-divide">
            <p className="mvi-sub">Subject stills (manual / extra)</p>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {[0, 1, 2, 3].map((i) => (
                <StillPick key={i} value={b.subjectIds[i] || ""} stills={stills}
                  placeholder={`- subject ${i + 1} -`}
                  set={(id) => { const a = [...b.subjectIds]; if (id) a[i] = id; else a.splice(i, 1); patch({ subjectIds: a.filter(Boolean) }); }} />
              ))}
            </div>
          </div>

          <div className="mvi-divide">
            <p className="mvi-muted" style={{ marginBottom: 6 }}>Active subject slots: {subjects.length ? subjects.map((id) => stillLabel(id, stills)).join(", ") : "none"} {subjects.length > 4 && "(capped at 4)"}</p>
            <p className="mvi-sub">Background (scene)</p>
            <StillPick value={b.backgroundId} stills={stills} set={(id) => patch({ backgroundId: id })} placeholder="- background still -" />
            <button className="mvi-link" style={{ marginTop: 6, display: "block" }} onClick={() => setSceneOpen(!sceneOpen)}>
              {sceneOpen ? "−" : "+"} compose background from characters
            </button>
            {sceneOpen && (
              <div style={{ marginTop: 6, display: "flex", flexDirection: "column", gap: 6, border: "1px solid rgba(255,255,255,.08)", borderRadius: 10, padding: 10 }}>
                <p className="mvi-muted">Place 1-3 characters into a scene (Qwen-Edit) and use it as this block's background - e.g. a band on stage behind the singer. Best for distant/secondary characters; the singer stays an MSR subject (not picked here).</p>
                {libChars.length === 0 && <p className="mvi-muted">No characters yet - build some in the Characters tab.</p>}
                <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                  {libChars.map((c) => {
                    const on = sceneChars.includes(c.id);
                    return (
                      <button key={c.id} onClick={() => toggleSceneChar(c.id)}
                        className="mvi-btn" style={on ? { background: "#EF9F27", color: "#3a2402", borderColor: "#EF9F27" } : undefined}>
                        {c.name}
                      </button>
                    );
                  })}
                </div>
                <textarea className={inp} rows={2} value={scenePrompt} placeholder="scene: where each character stands + setting + lighting (e.g. two musicians on a dark concert stage, guitarist left, bassist right, dramatic side light, haze)"
                  onChange={(e) => setScenePrompt(e.target.value)} />
                <button className="mvi-btn" onClick={() => compose(sceneChars, scenePrompt)} disabled={busy || !sceneChars.length} style={{ width: "100%" }}>
                  Generate background ({sceneChars.length}/3 characters)
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* prompt */}
      <div className="mvi-card">
        <div className="mvi-h">Prompt</div>
        <textarea className={inp} rows={3} value={b.prompt} placeholder="describe each reference's appearance + the action/camera move" style={{ width: "100%" }}
          onChange={(e) => patch({ prompt: e.target.value })} />
        <details style={{ marginTop: 8 }}>
          <summary className="mvi-muted" style={{ cursor: "pointer" }}>negative prompt</summary>
          <textarea className={inp} rows={2} value={b.negative} placeholder="(default: subtitles, watermark, worst quality, blurry, slow motion...)" style={{ width: "100%", marginTop: 6 }}
            onChange={(e) => patch({ negative: e.target.value })} />
        </details>
      </div>
      </div>

      {/* shot timeline (MSR + keyframe) */}
      {(b.renderMode === "msr" || b.renderMode === "keyframe") && (
        <div className="mvi-card">
          <div className="mvi-h">
            <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <input type="checkbox" checked={b.tlOn} onChange={(e) => patch({ tlOn: e.target.checked })} />
              {b.renderMode === "keyframe" ? "Segments & keyframes" : "Shot timeline"}
            </label>
            {b.tlOn && <span className="mvi-acc" style={{ fontSize: 11 }}>prompts on</span>}
          </div>
          <p className="mvi-muted" style={{ marginBottom: 8 }}>{b.renderMode === "keyframe"
            ? "Each segment is a time window. Pin a keyframe still to place it at the window start (or its end), and the model interpolates between pinned stills. Tick the box to ALSO schedule per-segment prompts."
            : "A global prompt held across the whole clip, plus ordered per-segment prompts placed along the frames. Tick the box to schedule per-segment prompts."}</p>
          <p className="mvi-sub">Global prompt</p>
          <textarea className={inp} rows={2} value={b.global} placeholder="constant scene/identity description held across the whole block" style={{ width: "100%" }} onChange={(e) => patch({ global: e.target.value })} />
          <div style={{ marginTop: 10 }}>
            {url ? (
              <ShotTimeline block={b} url={url} beats={beats} stills={stills} libChars={libChars}
                selSeg={selSeg} onSelSeg={setSelSeg} onPatch={patch} />
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                <p className="mvi-muted">Pick a song track above to open the waveform shot editor. Editing lengths/prompts directly below.</p>
                {b.segs.map((s, i) => (
                  <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 6 }}>
                    <input className={`${inp} w-20`} value={s.len} placeholder="frames" onChange={(e) => setSeg(i, { len: e.target.value })} />
                    <textarea className={inp} rows={1} value={s.prompt} placeholder={`segment ${i + 1} prompt`} style={{ flex: 1 }} onChange={(e) => setSeg(i, { prompt: e.target.value })} />
                    <button className="mvi-btn" onClick={() => delSeg(i)} title="remove">{"×"}</button>
                  </div>
                ))}
                <button className="mvi-link" onClick={addSeg}>+ segment</button>
              </div>
            )}
          </div>
          <label className="mvi-muted" style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 10 }}>
            epsilon
            <input type="range" min={0.001} max={0.6} step={0.001} value={b.epsilon} onChange={(e) => patch({ epsilon: Number(e.target.value) })} style={{ flex: 1 }} />
            <span style={{ width: 48, textAlign: "right" }}>{b.epsilon.toFixed(3)}</span>
          </label>
        </div>
      )}

      {/* audio + advanced row (horizontal cards) */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 12, alignItems: "start" }}>
      {(b.lipsync || b.renderMode === "s2v") && (
        <div className="mvi-card">
          <div className="mvi-h">Audio / lip-sync</div>
          <p className="mvi-sub">Vocal source</p>
          <select className={inp} value={b.audioId || ""} style={{ width: "100%" }} onChange={(e) => patch({ audioId: e.target.value })}>
            <option value="">{songAudioId ? "use the song track" : "- pick a track -"}</option>
            {audios.map((a) => <option key={a.id} value={a.id}>{(a.params?.title || a.params?.tags || a.id).toString().slice(0, 40)}</option>)}
          </select>
          <div className="mvi-row" style={{ marginTop: 10 }}>
            <Num label="audio start (s)" value={b.audioStart} set={(n) => patch({ audioStart: n })} step={0.1} w="w-24" title="offset into the track where this block's vocal begins" />
            <label className="mvi-muted" style={{ display: "flex", alignItems: "center", gap: 6, paddingBottom: 6 }} title="isolate the vocal from the mix (RoFormer) before driving lips">
              <input type="checkbox" checked={b.isolateVocal} onChange={(e) => patch({ isolateVocal: e.target.checked })} /> isolate vocal
            </label>
          </div>
        </div>
      )}

      {/* advanced render params */}
      {b.renderMode === "msr" && (
        <div className="mvi-card">
          <div className="mvi-h">Advanced render params <span className="sub">resolution is set by the project Resolution picker above</span></div>
          <div className="mvi-row">
            <label className="mvi-muted" style={{ display: "flex", flexDirection: "column", gap: 4 }} title="LiconMSR reference frame_count">
              ref frames
              <select className={`${inp} w-20`} value={b.refFrames} onChange={(e) => patch({ refFrames: Number(e.target.value) })}>
                {MSR_REF_COMBOS.map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </label>
            <Num label="seed" value={b.seed} set={(n) => patch({ seed: n })} w="w-28" title="0 = random" />
          </div>
          <div className="mvi-row" style={{ marginTop: 10 }}>
            <Num label="MSR strength" value={b.msrStrength} set={(n) => patch({ msrStrength: n })} step={0.05} w="w-24" title="identity IC-LoRA strength" />
            <Num label="guide strength" value={b.guideStrength} set={(n) => patch({ guideStrength: n })} step={0.05} w="w-24" />
            <Num label="steps" value={b.steps} set={(n) => patch({ steps: n })} w="w-16" />
            <Num label="cfg" value={b.cfg} set={(n) => patch({ cfg: n })} step={0.1} w="w-16" />
          </div>
        </div>
      )}
      </div>

      {/* retake region (re-render a slice of the rendered clip) */}
      {b.clipId && (
        <div className="mvi-card">
          <div className="mvi-h">Retake region <span className="sub">re-render a slice of the rendered clip{b.lipsync ? " (keeps lip-sync)" : ""}</span></div>
          <div className="mvi-row">
            <Num label="start (s)" value={rStart} set={setRStart} step={0.5} w="w-24" />
            <Num label="length (s)" value={rLen} set={setRLen} step={0.5} w="w-24" />
            <Num label="strength" value={rStrength} set={setRStrength} step={0.05} w="w-20" title="0-1; lower stays closer to the original" />
          </div>
          <textarea className={inp} rows={2} value={rPrompt} placeholder="what the retaken slice should show" style={{ width: "100%", marginTop: 8 }}
            onChange={(e) => setRPrompt(e.target.value)} />
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 8 }}>
            <button className="mvi-btn" onClick={() => retake(rStart, rLen, rPrompt, rStrength)} disabled={busy || !rPrompt.trim()}>Retake region</button>
            <span className="mvi-muted">re-decodes the whole clip and the slice can pop at the seam; result lands as a new render take</span>
          </div>
        </div>
      )}

      {/* rendered preview */}
      {clip?.media_url && (
        <video src={clip.media_url} controls loop style={{ width: "100%", borderRadius: 12 }} />
      )}
    </div>
  );
}
