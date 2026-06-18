import { useEffect, useState, type ReactNode } from "react";
import { api, type Config, type LibItem } from "./api";
import { Field, inp, PrimaryButton, GhostButton, rid, pollJob, type RunCtx } from "./ui";
import { useDrafts } from "./drafts";
import { openLightbox } from "./Lightbox";
import {
  type Block, type Character, type RenderMode, type Seg,
  makeBlock, hydrateBlock, ltxFrames, resolveSubjects, charRefIds, msrPayload,
  blockSeconds, MSR_REF_COMBOS,
} from "./mvmodel";

type Project = { id: string; name: string };

// ---- small UI helpers ---------------------------------------------------

function Collapse({ title, open, onToggle, children, accent }: {
  title: ReactNode; open: boolean; onToggle: () => void; children: ReactNode; accent?: boolean;
}) {
  return (
    <div className="rounded-lg border border-[var(--color-line)] bg-[var(--color-panel2)]">
      <button onClick={onToggle}
        className={`flex w-full items-center justify-between px-3 py-2 text-xs font-semibold ${accent ? "text-[var(--color-accent2)]" : "text-[var(--color-ink)]"}`}>
        <span>{title}</span>
        <span className="text-[var(--color-muted)]">{open ? "−" : "+"}</span>
      </button>
      {open && <div className="space-y-2 border-t border-[var(--color-line)] p-3">{children}</div>}
    </div>
  );
}

function Num({ label, value, set, step = 1, w = "w-20", title }: {
  label: string; value: number; set: (n: number) => void; step?: number; w?: string; title?: string;
}) {
  return (
    <label className="flex flex-col gap-0.5 text-[10px] text-[var(--color-muted)]" title={title}>
      {label}
      <input type="number" step={step} className={`${inp} ${w}`} value={value}
        onChange={(e) => set(Number(e.target.value))} />
    </label>
  );
}

// a still picker with a small clickable thumbnail of the current choice
function StillPick({ value, set, stills, placeholder = "— pick a still —", thumb = true }: {
  value: string; set: (id: string) => void; stills: LibItem[]; placeholder?: string; thumb?: boolean;
}) {
  const cur = stills.find((s) => s.id === value);
  return (
    <div className="flex items-center gap-2">
      {thumb && cur?.media_url && (
        <img src={cur.media_url} onClick={() => openLightbox(cur.media_url!)}
          className="h-10 w-10 shrink-0 cursor-zoom-in rounded object-cover" alt="" />
      )}
      <select className={inp} value={value} onChange={(e) => set(e.target.value)}>
        <option value="">{placeholder}</option>
        {stills.map((s) => (
          <option key={s.id} value={s.id}>
            {(s.params?.title || s.params?.prompt || s.id).toString().slice(0, 44)}
          </option>
        ))}
      </select>
    </div>
  );
}

const stillLabel = (id: string, stills: LibItem[]) => {
  const s = stills.find((x) => x.id === id);
  return s ? (s.params?.title || s.params?.prompt || id).toString().slice(0, 18) : id.slice(0, 8);
};

// ---- the editor ---------------------------------------------------------

export function MVStudioForm({ cfg, busy, library, ...ctx }:
  { cfg: Config; busy: boolean; library: LibItem[] } & RunCtx) {
  const d = useDrafts("mvstudio");
  const [projects, setProjects] = useState<Project[]>([]);
  const [projectId, setProjectId] = d.use("projectId", "");
  const [audioId, setAudioId] = d.use("audioId", "");          // the song (drives lip-sync windows + assembly mux)
  const [grade, setGrade] = d.use("grade", "none");
  const [grades, setGrades] = useState<string[]>(["none"]);
  const [blocks, setBlocks] = d.use<Block[]>("blocks", []);
  const [selId, setSelId] = d.use("selId", "");
  const [libChars, setLibChars] = useState<Character[]>([]);
  // which inspector sub-sections are expanded (per selected block is overkill; keep global)
  const [open, setOpen] = d.use<Record<string, boolean>>("openSecs", { refs: true, prompt: true });
  const toggle = (k: string) => setOpen({ ...open, [k]: !open[k] });

  useEffect(() => { api.projects().then(setProjects).catch(() => {}); }, []);
  useEffect(() => { api.mvGrades().then((r) => setGrades((r as { grades: string[] }).grades)).catch(() => {}); }, []);
  useEffect(() => { api.characters().then((r) => setLibChars(r as Character[])).catch(() => {}); }, []);

  const stills = library.filter((i) => i.mode === "videostill" && i.media_url);
  const audios = library.filter((i) => i.audio_url);
  const sel = blocks.find((b) => b.id === selId);

  // ---- block ops ----
  function reindex(list: Block[]): Block[] { return list.map((b, i) => ({ ...b, idx: i })); }
  function setAll(list: Block[]) { setBlocks(reindex(list)); }
  function patch(id: string, p: Partial<Block>) { setBlocks(blocks.map((b) => b.id === id ? { ...b, ...p } : b)); }
  function patchSel(p: Partial<Block>) { if (sel) patch(sel.id, p); }
  function addBlock() {
    const last = blocks[blocks.length - 1];
    const start = last ? last.end : 0;
    const b = makeBlock(start);
    b.audioId = audioId;
    setAll([...blocks, b]); setSelId(b.id);
  }
  function dupBlock(b: Block) {
    const copy: Block = { ...b, id: rid(), clipId: undefined, start: b.end, end: b.end + (b.end - b.start),
      segs: b.segs.map((s) => ({ ...s })), chars: b.chars.map((c) => ({ ...c })), subjectIds: [...b.subjectIds] };
    const at = blocks.findIndex((x) => x.id === b.id);
    const next = blocks.slice(); next.splice(at + 1, 0, copy); setAll(next); setSelId(copy.id);
  }
  function delBlock(id: string) {
    const next = blocks.filter((b) => b.id !== id); setAll(next);
    if (selId === id) setSelId(next[0]?.id || "");
  }
  function moveBlock(id: string, dir: -1 | 1) {
    const i = blocks.findIndex((b) => b.id === id); const j = i + dir;
    if (j < 0 || j >= blocks.length) return;
    const next = blocks.slice(); [next[i], next[j]] = [next[j], next[i]]; setAll(next);
  }

  // ---- persistence ----
  async function loadFromProject() {
    if (!projectId) return;
    try {
      const r = await api.projectVideoGet(projectId) as { video: { blocks?: Partial<Block>[]; audioId?: string; grade?: string } };
      const v = r.video;
      if (v.blocks?.length) { const hb = v.blocks.map(hydrateBlock); setAll(hb); setSelId(hb[0].id); }
      if (v.audioId) setAudioId(v.audioId);
      if (v.grade) setGrade(v.grade);
    } catch (e) { ctx.setResults([{ id: rid(), title: "Load failed", status: "error", pct: 0, err: (e as Error).message }]); }
  }
  async function saveToProject() {
    if (!projectId) { ctx.setResults([{ id: rid(), title: "Pick a project first", status: "error", pct: 0, err: "Select a song project to save the timeline into." }]); return; }
    try {
      await api.projectVideoSave(projectId, { blocks, audioId, grade });
      ctx.setResults([{ id: rid(), title: "Saved timeline to project", status: "done", pct: 100 }]);
    } catch (e) { ctx.setResults([{ id: rid(), title: "Save failed", status: "error", pct: 0, err: (e as Error).message }]); }
  }

  // ---- render one block ----
  async function genBlock(b: Block) {
    const subjectIds = resolveSubjects(b, libChars);
    const fail = (err: string) => ctx.setResults([{ id: rid(), title: `block ${b.idx + 1}`, status: "error", pct: 0, err }]);
    if (b.renderMode === "msr") {
      if (!subjectIds.length) return fail("Pick at least one subject reference (a character with refs, or a still).");
      if (!b.backgroundId) return fail("Pick a background still for the MSR scene.");
    }
    const card = { id: rid(), title: `block ${b.idx + 1} (${b.renderMode})`, status: "pending" as const, pct: 0 };
    ctx.setResults([card]);
    try {
      let job_id: string;
      if (b.renderMode === "msr") {
        ({ job_id } = await api.videoLtxMsr(msrPayload(b, subjectIds, audioId)) as { job_id: string });
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
          ? api.videoLtxI2V({ still_id: still, prompt: b.prompt, frames })
          : api.videoI2V({ still_id: still, prompt: b.prompt, length: frames })) as { job_id: string });
      }
      patch(b.id, { clipId: job_id });
      ctx.patch(card.id, { status: "running", pct: 5 });
      pollJob(job_id, card.id, ctx);
    } catch (e) { ctx.patch(card.id, { status: "error", pct: 0, err: (e as Error).message }); }
  }

  // ---- assemble ----
  async function assemble() {
    const ready = blocks.filter((b) => b.clipId);
    if (!ready.length) { ctx.setResults([{ id: rid(), title: "Nothing to assemble", status: "error", pct: 0, err: "Render some blocks first." }]); return; }
    const card = { id: rid(), title: `assembling ${ready.length} blocks...`, status: "running" as const, pct: 30 };
    ctx.setResults([card]);
    try {
      const r = await api.mvAssemble({ shots: ready.map((b) => ({ clip_id: b.clipId, start: b.start, end: b.end })),
        audio_id: audioId, grade, title: projects.find((p) => p.id === projectId)?.name || "music video" }) as { media_url: string };
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
        camera moves), with native single-pass lip-sync. Pick a block to edit it below.
      </p>

      {/* project + audio + grade */}
      <div className="grid grid-cols-2 gap-2">
        <Field label="Song project" hint="where the timeline is saved">
          <select className={inp} value={projectId} onChange={(e) => setProjectId(e.target.value)}>
            <option value="">- pick a project -</option>
            {projects.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
          </select>
        </Field>
        <Field label="Song audio" hint="drives lip-sync windows + final mux">
          <select className={inp} value={audioId} onChange={(e) => setAudioId(e.target.value)}>
            <option value="">- pick a track -</option>
            {audios.map((a) => <option key={a.id} value={a.id}>{(a.params?.title || a.params?.tags || a.mode || a.id).toString().slice(0, 40)}</option>)}
          </select>
        </Field>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <GhostButton onClick={loadFromProject}>Load saved</GhostButton>
        <GhostButton onClick={saveToProject}>Save to project</GhostButton>
        <GhostButton onClick={addBlock}>+ Block</GhostButton>
        <span className="ml-auto text-[10px] text-[var(--color-muted)]">{blocks.length} blocks {"·"} {ready} rendered</span>
      </div>

      {/* timeline overview (Phase 1: proportional, click-to-select; draggable ruler comes in Phase 3) */}
      {blocks.length > 0 && (
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

      {blocks.length === 0 && (
        <div className="rounded-lg border border-dashed border-[var(--color-line)] p-6 text-center text-xs text-[var(--color-muted)]">
          No blocks yet. Click <span className="text-[var(--color-ink)]">+ Block</span> to start the timeline.
        </div>
      )}

      {/* inspector for the selected block */}
      {sel && <Inspector key={sel.id} b={sel} idx={sel.idx} count={blocks.length}
        cfg={cfg} busy={busy} stills={stills} audios={audios} libChars={libChars} songAudioId={audioId}
        open={open} toggle={toggle}
        patch={patchSel} gen={() => genBlock(sel)} dup={() => dupBlock(sel)}
        del={() => delBlock(sel.id)} move={(dir) => moveBlock(sel.id, dir)} />}

      {/* assemble */}
      {blocks.length > 0 && (
        <div className="flex items-center gap-2 border-t border-[var(--color-line)] pt-3">
          <PrimaryButton onClick={assemble} disabled={busy || ready === 0}>
            {`Assemble video (${ready}/${blocks.length} blocks)`}
          </PrimaryButton>
          <label className="ml-auto flex items-center gap-1.5 text-[11px] text-[var(--color-muted)]" title="Color grade applied to every block for a consistent look (GPU-free)">
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

function Inspector({ b, idx, count, cfg, busy, stills, audios, libChars, songAudioId, open, toggle, patch, gen, dup, del, move }: {
  b: Block; idx: number; count: number; cfg: Config; busy: boolean;
  stills: LibItem[]; audios: LibItem[]; libChars: Character[]; songAudioId: string;
  open: Record<string, boolean>; toggle: (k: string) => void;
  patch: (p: Partial<Block>) => void; gen: () => void; dup: () => void; del: () => void; move: (dir: -1 | 1) => void;
}) {
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

  return (
    <div className="space-y-2 rounded-xl border border-[var(--color-accent)] bg-[var(--color-panel)] p-3">
      {/* header */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="rounded bg-[#2a1c19] px-2 py-0.5 text-xs font-semibold text-[var(--color-accent2)]">Block {idx + 1}</span>
        <select className={`${inp} w-auto text-[11px]`} value={b.renderMode} title="render mode"
          onChange={(e) => patch({ renderMode: e.target.value as RenderMode })}>
          <option value="msr">MSR (identity + free motion)</option>
          <option value="i2v">i2v (animate a still){cfg.video_ltx ? "" : " - Wan"}</option>
          <option value="s2v">S2V (anchor lip-sync)</option>
        </select>
        <label className="flex items-center gap-1 text-[10px] text-[var(--color-muted)]" title="native single-pass lip-sync to the song vocal">
          <input type="checkbox" checked={b.lipsync} onChange={(e) => patch({ lipsync: e.target.checked })} /> lip-sync
        </label>
        {b.clipId && <span className="text-[10px] text-green-400" title="rendered">rendered</span>}
        <span className="ml-auto flex items-center gap-1">
          <button onClick={() => move(-1)} disabled={idx === 0} className="px-1 text-[var(--color-muted)] hover:text-[var(--color-ink)] disabled:opacity-30" title="Move earlier">{"‹"}</button>
          <button onClick={() => move(1)} disabled={idx === count - 1} className="px-1 text-[var(--color-muted)] hover:text-[var(--color-ink)] disabled:opacity-30" title="Move later">{"›"}</button>
          <button onClick={dup} className="px-1 text-[var(--color-muted)] hover:text-[var(--color-ink)]" title="Duplicate">{"⧉"}</button>
          <button onClick={del} className="px-1 text-[var(--color-muted)] hover:text-red-400" title="Delete">{"×"}</button>
          <button onClick={gen} disabled={busy} className="rounded bg-[var(--color-accent)] px-3 py-1 text-[11px] font-semibold text-white disabled:opacity-50">{b.clipId ? "Re-render" : "Render"}</button>
        </span>
      </div>

      {/* timing */}
      <div className="flex flex-wrap items-end gap-2">
        <Num label="start (s)" value={b.start} set={(n) => patch({ start: n })} />
        <Num label="end (s)" value={b.end} set={(n) => patch({ end: n })} />
        <Num label="frames" value={b.frames} set={(n) => patch({ frames: n })} w="w-24" title="LTX coerces to 8k+1" />
        <Num label="fps" value={b.fps} set={(n) => patch({ fps: n })} w="w-16" />
        <span className="pb-1.5 text-[10px] text-[var(--color-muted)]">
          {"≈"} {blockSeconds(b)}s render
          <button className="ml-1 underline hover:text-[var(--color-ink)]" title="set frames from the block length"
            onClick={() => patch({ frames: ltxFrames(Math.max(2, b.end - b.start) * b.fps) })}>fit to {Math.max(0, b.end - b.start)}s</button>
        </span>
      </div>

      {b.renderMode === "msr" && (
        <Collapse title={`References (${subjects.length}/4 slots)`} open={!!open.refs} onToggle={() => toggle("refs")} accent>
          {/* hero characters */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-semibold uppercase tracking-wide text-[var(--color-muted)]">Hero characters ({heroCount}/2)</span>
              <button onClick={addChar} disabled={heroCount >= 2} className="text-[10px] text-[var(--color-accent2)] disabled:opacity-40">+ character</button>
            </div>
            {b.chars.map((bc, i) => {
              const c = libChars.find((x) => x.id === bc.charId);
              const refs = charRefIds(c, bc.wardrobeId);
              return (
                <div key={i} className="flex items-center gap-1.5">
                  <select className={inp} value={bc.charId} onChange={(e) => setChar(i, { charId: e.target.value, wardrobeId: undefined })}>
                    <option value="">- character -</option>
                    {libChars.map((x) => <option key={x.id} value={x.id}>{x.name}</option>)}
                  </select>
                  <select className={`${inp} w-40`} value={bc.wardrobeId || ""} title="wardrobe / look"
                    onChange={(e) => setChar(i, { wardrobeId: e.target.value || undefined })} disabled={!c?.wardrobes?.length}>
                    <option value="">{c?.wardrobes?.length ? "default look" : "(no wardrobes)"}</option>
                    {(c?.wardrobes || []).map((w) => <option key={w.id} value={w.id}>{w.name}</option>)}
                  </select>
                  <span className="w-12 shrink-0 text-[9px] text-[var(--color-muted)]" title="reference stills this character contributes">{refs.length} ref{refs.length === 1 ? "" : "s"}</span>
                  <button onClick={() => delChar(i)} className="px-1 text-[var(--color-muted)] hover:text-red-400" title="remove">{"×"}</button>
                </div>
              );
            })}
            {heroCount === 0 && <p className="text-[10px] text-[var(--color-muted)]">No characters set. Add a character (resolves its face + body refs), or pick subject stills manually below. Build characters in the Character library (coming in Phase 2) or the Music Video tab.</p>}
          </div>

          {/* manual subject stills */}
          <div className="space-y-1.5 border-t border-[var(--color-line)] pt-2">
            <span className="text-[10px] font-semibold uppercase tracking-wide text-[var(--color-muted)]">Subject stills (manual / extra)</span>
            {[0, 1, 2, 3].map((i) => (
              <StillPick key={i} value={b.subjectIds[i] || ""} stills={stills}
                placeholder={`- subject ${i + 1} -`}
                set={(id) => { const a = [...b.subjectIds]; if (id) a[i] = id; else a.splice(i, 1); patch({ subjectIds: a.filter(Boolean) }); }} />
            ))}
          </div>

          {/* resolved slots + background */}
          <div className="border-t border-[var(--color-line)] pt-2">
            <div className="mb-1 text-[10px] text-[var(--color-muted)]">Active subject slots: {subjects.length ? subjects.map((id) => stillLabel(id, stills)).join(", ") : "none"} {subjects.length > 4 && "(capped at 4)"}</div>
            <span className="text-[10px] font-semibold uppercase tracking-wide text-[var(--color-muted)]">Background (scene)</span>
            <StillPick value={b.backgroundId} stills={stills} set={(id) => patch({ backgroundId: id })} placeholder="- background still -" />
          </div>
        </Collapse>
      )}

      {/* prompt */}
      <Collapse title="Prompt" open={!!open.prompt} onToggle={() => toggle("prompt")} accent>
        <textarea className={inp} rows={3} value={b.prompt} placeholder="describe each reference's appearance + the action/camera move"
          onChange={(e) => patch({ prompt: e.target.value })} />
        <details>
          <summary className="cursor-pointer text-[10px] text-[var(--color-muted)]">negative prompt</summary>
          <textarea className={`${inp} mt-1`} rows={2} value={b.negative} placeholder="(default: subtitles, watermark, worst quality, blurry, slow motion...)"
            onChange={(e) => patch({ negative: e.target.value })} />
        </details>
      </Collapse>

      {/* timeline prompter */}
      {b.renderMode === "msr" && (
        <Collapse open={!!open.tl} onToggle={() => toggle("tl")}
          title={<label className="flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
            <input type="checkbox" checked={b.tlOn} onChange={(e) => patch({ tlOn: e.target.checked })} />
            Timeline prompter {b.tlOn && <span className="text-[9px] text-[var(--color-accent2)]">on</span>}
          </label>}>
          <p className="text-[10px] text-[var(--color-muted)]">A global prompt held across the whole clip, plus ordered per-segment prompts placed along the frames. Empty segment lengths = even split. Epsilon sets boundary softness (0.001 sharp cut, 0.5+ smooth for a continuous move).</p>
          <Field label="Global prompt" hint="held across the whole block">
            <textarea className={inp} rows={2} value={b.global} placeholder="constant scene/identity description" onChange={(e) => patch({ global: e.target.value })} />
          </Field>
          <div className="space-y-1.5">
            {b.segs.map((s, i) => (
              <div key={i} className="flex items-start gap-1.5">
                <input className={`${inp} w-20`} value={s.len} placeholder="frames" title="segment length in frames (empty = even)" onChange={(e) => setSeg(i, { len: e.target.value })} />
                <textarea className={inp} rows={1} value={s.prompt} placeholder={`segment ${i + 1} prompt`} onChange={(e) => setSeg(i, { prompt: e.target.value })} />
                <button onClick={() => delSeg(i)} className="px-1 pt-1.5 text-[var(--color-muted)] hover:text-red-400" title="remove">{"×"}</button>
              </div>
            ))}
            <button onClick={addSeg} className="text-[10px] text-[var(--color-accent2)]">+ segment</button>
          </div>
          <label className="flex items-center gap-2 text-[10px] text-[var(--color-muted)]">
            epsilon
            <input type="range" min={0.001} max={0.6} step={0.001} value={b.epsilon} onChange={(e) => patch({ epsilon: Number(e.target.value) })} className="flex-1" />
            <span className="w-12 text-right">{b.epsilon.toFixed(3)}</span>
          </label>
        </Collapse>
      )}

      {/* audio / lip-sync */}
      {(b.lipsync || b.renderMode === "s2v") && (
        <Collapse title="Audio / lip-sync" open={!!open.audio} onToggle={() => toggle("audio")}>
          <Field label="Vocal source" hint="defaults to the song track above; isolated to drive the lips">
            <select className={inp} value={b.audioId || ""} onChange={(e) => patch({ audioId: e.target.value })}>
              <option value="">{songAudioId ? "use the song track" : "- pick a track -"}</option>
              {audios.map((a) => <option key={a.id} value={a.id}>{(a.params?.title || a.params?.tags || a.id).toString().slice(0, 40)}</option>)}
            </select>
          </Field>
          <div className="flex items-end gap-3">
            <Num label="audio start (s)" value={b.audioStart} set={(n) => patch({ audioStart: n })} step={0.1} w="w-24" title="offset into the track where this block's vocal begins" />
            <label className="flex items-center gap-1.5 pb-1.5 text-[10px] text-[var(--color-muted)]" title="isolate the vocal from the mix (RoFormer) before driving lips">
              <input type="checkbox" checked={b.isolateVocal} onChange={(e) => patch({ isolateVocal: e.target.checked })} /> isolate vocal
            </label>
          </div>
        </Collapse>
      )}

      {/* advanced render params */}
      {b.renderMode === "msr" && (
        <Collapse title="Advanced render params" open={!!open.adv} onToggle={() => toggle("adv")}>
          <div className="flex flex-wrap items-end gap-3">
            <Num label="width" value={b.width} set={(n) => patch({ width: n })} step={32} w="w-20" />
            <Num label="height" value={b.height} set={(n) => patch({ height: n })} step={32} w="w-20" />
            <label className="flex flex-col gap-0.5 text-[10px] text-[var(--color-muted)]" title="LiconMSR reference frame_count">
              ref frames
              <select className={`${inp} w-20`} value={b.refFrames} onChange={(e) => patch({ refFrames: Number(e.target.value) })}>
                {MSR_REF_COMBOS.map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </label>
            <Num label="seed" value={b.seed} set={(n) => patch({ seed: n })} w="w-28" title="0 = random" />
          </div>
          <div className="flex flex-wrap items-end gap-3">
            <Num label="MSR strength" value={b.msrStrength} set={(n) => patch({ msrStrength: n })} step={0.05} w="w-24" title="identity IC-LoRA strength" />
            <Num label="guide strength" value={b.guideStrength} set={(n) => patch({ guideStrength: n })} step={0.05} w="w-24" />
            <Num label="steps" value={b.steps} set={(n) => patch({ steps: n })} w="w-16" />
            <Num label="cfg" value={b.cfg} set={(n) => patch({ cfg: n })} step={0.1} w="w-16" />
          </div>
        </Collapse>
      )}

      {/* rendered preview */}
      {clip?.media_url && (
        <video src={clip.media_url} controls loop className="w-full rounded-lg" />
      )}
    </div>
  );
}
