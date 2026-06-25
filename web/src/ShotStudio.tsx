import { useRef, useState } from "react";
import { api, type LibItem } from "./api";
import { Field, inp, GhostButton, PrimaryButton, rid } from "./ui";
import { StillPick, Num } from "./mvui";
import { LtxDirectorEditor, type LtxDirectorHandle } from "./LtxDirectorEditor";
import { type Block, type ChainPiece, type Take, type RenderMode } from "./mvmodel";

// ---------------------------------------------------------------------------------------------
// Shot Studio — the per-segment editor. One timeline shot opens here as a continuous take built
// from ORDERED chain pieces: piece 0 is the base shot; each Extend appends an FFLF piece driven by
// the previous piece's last 33 frames. Every piece keeps every take it rolled (seed-hunt drafts +
// finished multiroll variants) as cards, so you pick/reroll without losing work.
// FFLF graph = stock LTXVAddGuide (build_ltx_fflf); MSR/keyframe segments will gain the LTXDirector
// editor controls in a later phase. This phase wires the FFLF hunt -> finish -> extend loop + cards.
// ---------------------------------------------------------------------------------------------

const poster = (id: string) => `/api/media/${id}#t=0.5`;        // seek a frame so <video> paints a poster
const FFLF_TAIL = 33;                                            // foxydits' extend anchor length

// Poll a video job to its media url (ui.waitJob only resolves audio jobs).
function waitMedia(jobId: string, onPct?: (p: number) => void): Promise<string> {
  return new Promise((resolve, reject) => {
    const t = window.setInterval(async () => {
      const j = await api.job(jobId).catch(() => null);
      if (!j) return;
      if (j.status === "done" && j.media_url) { window.clearInterval(t); resolve(j.media_url); }
      else if (j.status === "error" || j.status === "failed") { window.clearInterval(t); reject(new Error(j.error || "render error")); }
      else if (onPct && j.max) onPct(Math.round((100 * (j.progress || 0)) / j.max));
    }, 1500);
  });
}

function ThumbVideo({ id, className = "" }: { id: string; className?: string }) {
  return <video src={poster(id)} muted playsInline preload="metadata"
    className={`h-full w-full object-cover ${className}`} />;
}

export function ShotStudio({ block: b, idx, patch, stills, audios, songAudioId, onClose }: {
  block: Block; idx: number; patch: (p: Partial<Block>) => void;
  stills: LibItem[]; audios: LibItem[]; songAudioId: string;
  onClose: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const pieces = b.pieces || [];
  // transient hunt state: the 3 half-res drafts for the piece currently being hunted
  const [hunt, setHunt] = useState<{ pieceLabel: string; forExtend: boolean; drafts: { jobId: string; seed: number; url?: string }[] } | null>(null);
  const [status, setStatus] = useState("");
  // "Add keyframe still" panel: generate (seed-hunt half-res -> full on pick) or pick from library,
  // then inject into the editor timeline as a keyframe via the editor's own add-image path.
  const edRef = useRef<LtxDirectorHandle>(null);
  const [kfPrompt, setKfPrompt] = useState("");
  const [kfFrame, setKfFrame] = useState(0);
  const [kfBusy, setKfBusy] = useState(false);
  const [kfDrafts, setKfDrafts] = useState<{ jobId: string; seed: number; url?: string }[]>([]);
  const [kfUseChar, setKfUseChar] = useState(false);   // generate from the shot's character refs (identity)
  const charRefs = (b.subjectIds || []).filter(Boolean).slice(0, 3);

  const setPieces = (next: ChainPiece[]) => patch({ pieces: next });
  const selectedTakeOf = (p: ChainPiece) => p.takes.find((t) => t.id === p.selectedTakeId) || p.takes[0];
  const lastPiece = pieces[pieces.length - 1];
  const lastClip = lastPiece && selectedTakeOf(lastPiece)?.clipId;

  // base-shot anchors live on the block; extends derive their first anchor from the prior tail.
  const fflfReady = !!b.fflfFirstId;

  function note(s: string) { setStatus(s); }

  // ---- seed-hunt: 3 half-res drafts for a new piece (base, or an extend off the last tail) ----
  async function runHunt(forExtend: boolean) {
    if (!forExtend && !b.fflfFirstId) { note("Pick a First anchor still for the base shot first."); return; }
    if (forExtend && !lastClip) { note("Render the base shot before extending."); return; }
    setBusy(true); note(forExtend ? "Hunting extension (3 drafts)…" : "Hunting base shot (3 drafts)…");
    try {
      const p: Record<string, unknown> = {
        mode: "hunt",
        last_id: forExtend ? (b.fflfFirstId || b.fflfLastId) : b.fflfLastId,   // extend lands on the opening; base = chosen last
        last_strength: b.fflfLastStrength ?? 0.5,
        prompt: b.prompt,
        width: b.width, height: b.height, frames: b.frames, fps: b.fps,
      };
      if (forExtend) { p.first_id = lastClip; p.first_kind = "video"; p.first_frames = FFLF_TAIL; p.first_skip = Math.max(0, b.frames - FFLF_TAIL); }
      else { p.first_id = b.fflfFirstId; }
      const r = await api.videoLtxFflf(p) as { base_seed: number; drafts: { job_id: string }[] };
      const drafts = r.drafts.map((d, i) => ({ jobId: d.job_id, seed: r.base_seed + i }));
      setHunt({ pieceLabel: forExtend ? `extend ${pieces.length + 1}` : "base", forExtend, drafts });
      // resolve each draft's poster as it finishes
      drafts.forEach((d) => waitMedia(d.jobId).then((u) =>
        setHunt((h) => h ? { ...h, drafts: h.drafts.map((x) => x.jobId === d.jobId ? { ...x, url: u } : x) } : h)).catch(() => {}));
      note("Drafts rendering — pick one to finish (drafts are video-only; lip-sync is added on finish).");
    } catch (e) { note("Hunt failed: " + (e as Error).message); }
    finally { setBusy(false); }
  }

  // ---- finish a chosen draft at full res (+ lip-sync) -> a Take on a (new) piece ----
  async function finishDraft(stage1Seed: number) {
    if (!hunt) return;
    const forExtend = hunt.forExtend;
    setBusy(true); note("Finishing at full res" + (b.lipsync ? " with lip-sync" : "") + "…");
    try {
      const p: Record<string, unknown> = {
        mode: "finish", stage1_seed: stage1Seed,
        last_id: forExtend ? (b.fflfFirstId || b.fflfLastId) : b.fflfLastId,
        last_strength: b.fflfLastStrength ?? 0.5,
        prompt: b.prompt, width: b.width, height: b.height, frames: b.frames, fps: b.fps,
      };
      if (forExtend) { p.first_id = lastClip; p.first_kind = "video"; p.first_frames = FFLF_TAIL; p.first_skip = Math.max(0, b.frames - FFLF_TAIL); }
      else { p.first_id = b.fflfFirstId; }
      if (b.lipsync && b.audioId) { p.audio_id = b.audioId; p.audio_start = b.audioStart; p.isolate_vocal = false; }   // full song (decided)
      const r = await api.videoLtxFflf(p) as { job_id: string };
      const clipId = await waitMedia(r.job_id, (pc) => note(`Finishing… ${pc}%`));
      const take: Take = { id: rid(), clipId, stage1Seed, draft: false, label: `seed ${stage1Seed}` };
      if (forExtend) {
        const piece: ChainPiece = { id: rid(), lane: "fflf", label: `Extend ${pieces.length + 1}`, takes: [take], selectedTakeId: take.id, lastStillId: b.fflfFirstId };
        setPieces([...pieces, piece]);
      } else if (pieces.length === 0) {
        setPieces([{ id: rid(), lane: "fflf", label: "Base shot", takes: [take], selectedTakeId: take.id, lastStillId: b.fflfLastId }]);
      } else {
        // re-finish the base piece (multiroll) -> add a take to piece 0
        setPieces(pieces.map((pc, i) => i === 0 ? { ...pc, takes: [...pc.takes, take], selectedTakeId: take.id } : pc));
      }
      setHunt(null); note("Finished. Added as a take — pick Extend to continue the take.");
    } catch (e) { note("Finish failed: " + (e as Error).message); }
    finally { setBusy(false); }
  }

  // ---- render the editor's authored timeline (timeline_data passthrough -> LTXDirector) ----
  async function renderTimeline() {
    if (!b.director?.timeline_data) { note("Build a timeline in the editor first (add a segment)."); return; }
    setBusy(true); note("Rendering the timeline…");
    try {
      const r = await api.videoLtxKeyframe({ ...b.director, width: b.width, height: b.height, frames: b.frames, fps: b.fps }) as { job_id: string };
      const clipId = await waitMedia(r.job_id, (pc) => note(`Rendering… ${pc}%`));
      const take: Take = { id: rid(), clipId, draft: false, label: "timeline" };
      if (pieces.length === 0) setPieces([{ id: rid(), lane: "fflf", label: "Base shot", takes: [take], selectedTakeId: take.id }]);
      else setPieces(pieces.map((pc, i) => i === 0 ? { ...pc, takes: [...pc.takes, take], selectedTakeId: take.id } : pc));
      note("Timeline rendered — added as a take.");
    } catch (e) { note("Render failed: " + (e as Error).message); }
    finally { setBusy(false); }
  }

  // ---- keyframe still: seed-hunt (half-res) -> pick -> full-res -> inject into the editor timeline ----
  const genStill = (seed: number, w: number, h: number) =>
    (kfUseChar && charRefs.length)
      ? api.videoCharStill({ ref_ids: charRefs, prompt: kfPrompt, width: w, height: h, seed })   // identity from refs
      : api.videoStill({ prompt: kfPrompt, width: w, height: h, seed });                          // prompt-only scene
  async function injectStill(mediaUrl: string) {
    const blob = await fetch(mediaUrl).then((r) => r.blob());
    const file = new File([blob], `kf_${Date.now()}.png`, { type: blob.type || "image/png" });
    if (!edRef.current) throw new Error("editor not ready");
    edRef.current.addImage(file, Math.max(0, Math.round(kfFrame)));
  }
  async function huntStills() {
    if (!kfPrompt.trim()) { setStatus("Enter a prompt for the keyframe still."); return; }
    setKfBusy(true); setStatus("Hunting 3 stills (full size)…");
    const base = Math.floor(Math.random() * 2_000_000_000);
    const hw = b.width, hh = b.height;   // full size — half-res previews drift too much from the final
    try {
      const drafts: { jobId: string; seed: number; url?: string }[] = [];
      for (let i = 0; i < 3; i++) {
        const r = await genStill(base + i, hw, hh) as { job_id: string };
        drafts.push({ jobId: r.job_id, seed: base + i });
      }
      setKfDrafts(drafts);
      drafts.forEach((d) => waitMedia(d.jobId).then((u) =>
        setKfDrafts((ds) => ds.map((x) => x.jobId === d.jobId ? { ...x, url: u } : x))).catch(() => {}));
      setStatus("Still drafts rendering — pick one to finalize at full res.");
    } catch (e) { setStatus("Still hunt failed: " + (e as Error).message); }
    finally { setKfBusy(false); }
  }
  async function useDraft(url: string) {
    setKfBusy(true); setStatus("Adding keyframe…");
    try {
      await injectStill(url);                 // drafts are already full-res, just inject the picked one
      setKfDrafts([]); setStatus(`Keyframe still added at frame ${kfFrame}.`);
    } catch (e) { setStatus("Add failed: " + (e as Error).message); }
    finally { setKfBusy(false); }
  }
  async function addLibraryStill(stillId: string) {
    if (!stillId) return;
    setKfBusy(true); setStatus("Adding library still as keyframe…");
    try { await injectStill(`/api/media/${stillId}`); setStatus(`Library still added at frame ${kfFrame}.`); }
    catch (e) { setStatus("Add failed: " + (e as Error).message); }
    finally { setKfBusy(false); }
  }

  function selectTake(pieceId: string, takeId: string) {
    setPieces(pieces.map((p) => p.id === pieceId ? { ...p, selectedTakeId: takeId } : p));
  }
  function removePiece(pieceId: string) { setPieces(pieces.filter((p) => p.id !== pieceId)); }

  const modes: { v: RenderMode; label: string }[] = [
    { v: "fflf", label: "FFLF (extendable)" }, { v: "msr", label: "MSR (identity+sing)" },
    { v: "keyframe", label: "Keyframe" }, { v: "i2v", label: "B-roll" },
  ];
  const totalSecs = pieces.length ? (b.frames / b.fps) + (pieces.length - 1) * ((b.frames - FFLF_TAIL) / b.fps) : 0;

  return (
    <div className="ss-root flex flex-col gap-4">
      {/* header */}
      <div className="flex items-center gap-3">
        <GhostButton onClick={onClose}>{"← Timeline"}</GhostButton>
        <span className="text-sm font-semibold text-[var(--color-ink)]">Shot {idx + 1}</span>
        <select className={inp} style={{ width: "auto" }} value={b.renderMode}
          onChange={(e) => patch({ renderMode: e.target.value as RenderMode })} title="How this shot is generated">
          {modes.map((m) => <option key={m.v} value={m.v}>{m.label}</option>)}
        </select>
        <span className="ml-auto text-[11px] text-[var(--color-muted)]">
          {pieces.length} piece{pieces.length === 1 ? "" : "s"} · ~{totalSecs.toFixed(1)}s take
        </span>
      </div>

      {/* ===================== TIMELINE EDITOR (vendored LTXDirector, GPL-3) ===================== */}
      {["fflf", "msr", "keyframe"].includes(b.renderMode) && (
        <div className="ss-card" style={{ padding: 8 }}>
          <LtxDirectorEditor ref={edRef} timelineData={b.director?.timeline_data} frames={b.frames} fps={b.fps}
            onChange={(payload) => patch({ director: payload, timelineData: payload.timeline_data })} />
        </div>
      )}

      {/* ===================== ADD KEYFRAME STILL (generate seed-hunt / library -> inject as keyframe) ===================== */}
      {["fflf", "msr", "keyframe"].includes(b.renderMode) && (
        <div className="ss-card flex flex-col gap-3">
          <div className="text-xs font-semibold text-[var(--color-ink)]">Add keyframe still</div>
          <label className="flex items-center gap-2 text-[11px] text-[var(--color-muted)]">
            <input type="checkbox" checked={kfUseChar} onChange={(e) => setKfUseChar(e.target.checked)} disabled={!charRefs.length} />
            Use this shot's character (identity){charRefs.length ? ` · ${charRefs.length} ref${charRefs.length > 1 ? "s" : ""}` : " — no refs on this shot"}
          </label>
          <div className="grid grid-cols-[1fr_auto_auto] items-end gap-3">
            <Field label="Prompt (generate a still)">
              <textarea className={inp} rows={2} value={kfPrompt} onChange={(e) => setKfPrompt(e.target.value)}
                placeholder="the singer on a ruined ashen stage, dramatic light…" />
            </Field>
            <Num label="At frame" value={kfFrame} set={setKfFrame} step={1} w="w-20" />
            <PrimaryButton onClick={huntStills} disabled={kfBusy || !kfPrompt.trim()}>Seed-hunt 3 stills</PrimaryButton>
          </div>
          {kfDrafts.length > 0 && (
            <div className="grid grid-cols-3 gap-3">
              {kfDrafts.map((d) => (
                <div key={d.jobId} className="ss-piece">
                  <div className="ss-thumb">{d.url ? <img src={d.url} alt="" className="h-full w-full object-cover" /> : <div className="ss-spin">rendering…</div>}</div>
                  <div className="flex items-center justify-between gap-2 px-2 py-1.5">
                    <span className="text-[11px] text-[var(--color-muted)]">seed {d.seed}</span>
                    <GhostButton onClick={() => useDraft(d.url!)} disabled={kfBusy || !d.url}>Use this</GhostButton>
                  </div>
                </div>
              ))}
            </div>
          )}
          <div className="flex items-center gap-2">
            <span className="text-[11px] text-[var(--color-muted)]">or add an existing still:</span>
            <div className="min-w-[220px]"><StillPick value="" set={addLibraryStill} stills={stills} placeholder="— pick from library —" /></div>
          </div>
        </div>
      )}

      {/* ===================== BUILD PANEL (anchors · lip-sync · render · actions) ===================== */}
      <div className="ss-card grid grid-cols-[1.3fr_1fr] gap-5">
        {/* left: source + prompt */}
        <div className="flex flex-col gap-3">
          {b.renderMode === "fflf" ? (
            <>
              <Field label="First frame (opening anchor)">
                <StillPick value={b.fflfFirstId || ""} set={(id) => patch({ fflfFirstId: id })} stills={stills} />
              </Field>
              <Field label="Last frame (keyframe target — use a singing pose for sung shots)">
                <StillPick value={b.fflfLastId || ""} set={(id) => patch({ fflfLastId: id })} stills={stills} />
              </Field>
              <div className="flex gap-3">
                <Num label="First strength" value={b.fflfFirstStrength ?? 0.7} set={(n) => patch({ fflfFirstStrength: n })} step={0.05} />
                <Num label="Last strength" value={b.fflfLastStrength ?? 0.5} set={(n) => patch({ fflfLastStrength: n })} step={0.05} />
              </div>
            </>
          ) : (
            <div className="rounded-md border border-dashed border-[var(--color-line)] p-3 text-[11px] text-[var(--color-muted)]">
              {b.renderMode.toUpperCase()} editor controls (LTXDirector prompt-timeline / keyframes / retake)
              land in the next phase. Switch to <b>FFLF</b> to use the seed-hunt / multiroll / extend loop now.
            </div>
          )}
          <Field label="Prompt (what happens between the anchors)">
            <textarea className={inp} rows={3} value={b.prompt} onChange={(e) => patch({ prompt: e.target.value })}
              placeholder="continuous fluid shot, the camera moving naturally around her…" />
          </Field>
        </div>
        {/* right: lip-sync + params + action rail */}
        <div className="flex flex-col gap-3">
          <label className="flex items-center gap-2 text-xs text-[var(--color-ink)]">
            <input type="checkbox" checked={b.lipsync} onChange={(e) => patch({ lipsync: e.target.checked })} />
            Lip-sync to the song (full song fed — no isolation)
          </label>
          {b.lipsync && (
            <div className="flex gap-3">
              <Field label="Vocal track"><StillPick value={b.audioId || songAudioId} set={(id) => patch({ audioId: id })} stills={audios} thumb={false} placeholder="— song —" /></Field>
              <Num label="Audio start (s)" value={b.audioStart} set={(n) => patch({ audioStart: n })} step={0.1} />
            </div>
          )}
          <div className="flex flex-wrap gap-2">
            <Num label="W" value={b.width} set={(n) => patch({ width: n })} step={32} w="w-20" />
            <Num label="H" value={b.height} set={(n) => patch({ height: n })} step={32} w="w-20" />
            <Num label="Frames" value={b.frames} set={(n) => patch({ frames: n })} step={8} w="w-20" />
            <Num label="FPS" value={b.fps} set={(n) => patch({ fps: n })} step={1} w="w-16" />
          </div>
          {/* action rail */}
          <div className="mt-1 flex flex-col gap-2 rounded-md border border-[var(--color-line)] p-3">
            <span className="text-[10px] uppercase tracking-wide text-[var(--color-muted)]">Build this shot</span>
            <PrimaryButton onClick={renderTimeline} disabled={busy || !b.director?.timeline_data}>
              Render timeline (from editor)
            </PrimaryButton>
            <PrimaryButton onClick={() => runHunt(false)} disabled={busy || b.renderMode !== "fflf" || !fflfReady}>
              {pieces.length ? "Re-hunt base (3 drafts)" : "Seed-hunt base (3 drafts)"}
            </PrimaryButton>
            <GhostButton onClick={() => runHunt(true)} disabled={busy || !lastClip}>
              + Extend off last piece's tail (3 drafts)
            </GhostButton>
            <GhostButton onClick={() => note("Crossfade-assemble runs server-side (next phase) — uses the 33-frame overlap + continuation audio.")} disabled={pieces.length < 2}>
              Assemble continuous take ({pieces.length} pieces)
            </GhostButton>
          </div>
          {status && <p className="text-[11px] text-[var(--color-accent2)]">{status}</p>}
        </div>
      </div>

      {/* ===================== HUNT DRAFTS (pick one) ===================== */}
      {hunt && (
        <div className="ss-card">
          <div className="mb-2 text-xs font-semibold text-[var(--color-ink)]">Pick a draft for “{hunt.pieceLabel}” (half-res, silent)</div>
          <div className="grid grid-cols-3 gap-3">
            {hunt.drafts.map((d) => (
              <div key={d.jobId} className="ss-piece">
                <div className="ss-thumb">{d.url ? <ThumbVideo id={d.url.split("/").pop()!.split("?")[0]} /> : <div className="ss-spin">rendering…</div>}</div>
                <div className="flex items-center justify-between gap-2 px-2 py-1.5">
                  <span className="text-[11px] text-[var(--color-muted)]">seed {d.seed}</span>
                  <GhostButton onClick={() => finishDraft(d.seed)} disabled={busy || !d.url}>Finish</GhostButton>
                </div>
              </div>
            ))}
          </div>
          <button className="mt-2 text-[10px] text-[var(--color-muted)] underline" onClick={() => setHunt(null)}>dismiss drafts</button>
        </div>
      )}

      {/* ===================== PIECES (card layout under the editor) ===================== */}
      <div>
        <div className="mb-2 text-xs font-semibold text-[var(--color-ink)]">Pieces of this take</div>
        {pieces.length === 0 ? (
          <div className="rounded-lg border border-dashed border-[var(--color-line)] p-6 text-center text-[11px] text-[var(--color-muted)]">
            No pieces yet. Set the anchors, then <b>Seed-hunt base</b> → pick a draft → it becomes piece 1. Then <b>Extend</b> to grow the take.
          </div>
        ) : (
          <div className="grid grid-cols-3 gap-3">
            {pieces.map((p, i) => {
              const sel = selectedTakeOf(p);
              return (
                <div key={p.id} className="ss-piece">
                  <div className="ss-thumb">{sel ? <ThumbVideo id={sel.clipId} /> : <div className="ss-spin">—</div>}</div>
                  <div className="px-2 py-1.5">
                    <div className="flex items-center justify-between">
                      <span className="text-[11px] font-semibold text-[var(--color-ink)]">{i + 1}. {p.label}</span>
                      <button className="text-[var(--color-muted)] hover:text-red-400" title="remove piece" onClick={() => removePiece(p.id)}>×</button>
                    </div>
                    {/* takes / multiroll variants */}
                    {p.takes.length > 1 && (
                      <div className="mt-1 flex gap-1">
                        {p.takes.map((t) => (
                          <button key={t.id} title={t.label} onClick={() => selectTake(p.id, t.id)}
                            className={`h-9 w-12 overflow-hidden rounded border ${t.id === p.selectedTakeId ? "border-[var(--color-accent)]" : "border-[var(--color-line)] opacity-70"}`}>
                            <ThumbVideo id={t.clipId} />
                          </button>
                        ))}
                      </div>
                    )}
                    <div className="mt-1 text-[10px] text-[var(--color-muted)]">{p.takes.length} take{p.takes.length === 1 ? "" : "s"} · {sel?.label}</div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      <style>{`
        .ss-card{border:1px solid var(--color-line);background:var(--color-panel);border-radius:12px;padding:16px;}
        .ss-piece{border:1px solid var(--color-line);background:var(--color-panel2);border-radius:10px;overflow:hidden;}
        .ss-thumb{aspect-ratio:16/9;background:#0d0f13;display:flex;align-items:center;justify-content:center;}
        .ss-spin{font-size:10px;color:var(--color-muted);}
      `}</style>
    </div>
  );
}
