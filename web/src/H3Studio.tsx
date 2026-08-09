import { useState } from "react";
import { api, type LibItem } from "./api";
import { inp, PrimaryButton, GhostButton, rid, pollJob, type RunCtx } from "./ui";
import { useDrafts } from "./drafts";
import { openLightbox } from "./Lightbox";
import { type Character } from "./mvmodel";

// ============================================================================
// MiniMax H3 segment pipeline (Phase 5 - docs/MINIMAX_H3_PLAN.md).
// The hybrid shot model: the writer LLM turns the song's audio structure into render
// SEGMENTS ("single" = one shot, always for lip-sync; "scene" = 2-4 timestamped cuts
// in one generation). Each segment renders as ONE /api/video/h3_ref2v call whose refs
// follow the compiled prompt's picture numbering: character sheets, then outfit sheets,
// then the segment's environment still. Lip-sync segments carry the song window as
// <Audio 1> (direct reuse - the soundtrack IS the song, so assembly lays the master
// underneath at zero offset).
// ============================================================================

type H3Shot = {
  type: string; framing: string; lipsync: boolean; camera: string;
  scene: string; action: string; costume: string; characters: string[];
};
export type H3Segment = {
  start: number; end: number; seconds: number; render_seconds: number; frames: number;
  section: string; cuts: { start: number; end: number }[];
  kind: "single" | "scene"; shots: H3Shot[]; lipsync: boolean; soundscape: string;
  prompt: string; picture_map: Record<string, number>; outfit_map: Record<string, number>;
  prop_map?: Record<string, number>;
  env_picture: number;
  // client-side render state
  envStillId?: string; clipId?: string; clipVariants?: string[];
  handEdited?: boolean;   // raw prompt overridden by hand (a recompile clears this)
};

const H3_CAMERAS = ["static", "push in", "pull back", "truck left", "truck right",
  "arc left", "arc right", "tilt up", "crane up"];

const fmt = (t: number) => `${Math.floor(t / 60)}:${String(Math.floor(Math.max(0, t) % 60)).padStart(2, "0")}`;

// environment still request: the verified rich-caption layout recipe (Krea2 needs a
// fully-populated caption; the ref's realism sets the scene's realism ceiling)
function envRequest(scene: string, seed: number) {
  const s = scene.replace(/\.\s*$/, "");
  return {
    engine: "krea2", two_pass: true, enhancer: true,
    width: 1920, height: 1088, seed,
    layout: {
      overview: `An ultra-photorealistic cinematic view of ${s}.`,
      background: `${s}. Rich physical detail: real materials, surface wear, depth layers, ` +
        `natural atmospheric haze where it fits, believable light falloff from every source.`,
      photo_style: "",
      aesthetics: "photorealistic, cinematic, true-to-life detail, natural film-like exposure, atmospheric",
      lighting: "natural motivated light exactly as the scene describes, deep soft shadows, no artificial studio light",
      medium: "photograph",
      regions: [{ desc: `${s}, rendered with true-to-life photographic detail.`, x: 0.03, y: 0.03, w: 0.94, h: 0.94 }],
    },
  };
}

export function H3Studio({ cast, audioId, songPayload, resW, resH, grade, library, busy, ...ctx }: {
  cast: Character[]; audioId: string; songPayload: unknown | null; resW: number; resH: number;
  grade: string; library: LibItem[]; busy: boolean;
} & RunCtx) {
  const d = useDrafts("mvstudio");
  const [segments, setSegments] = d.use<H3Segment[]>("h3segments", []);
  const [castCostume, setCastCostume] = d.use<Record<string, string>>("h3castCostume", {});
  const [castProp, setCastProp] = d.use<Record<string, string>>("h3castProp", {});   // "" = none
  const [writing, setWriting] = useState(false);
  const [genning, setGenning] = useState("");        // "env:<i>" | "seg:<i>" while submitting
  const [drafts, setDrafts] = useState<Record<number, { jobId: string; seed: number; url?: string; err?: boolean }[]>>({});
  const [editIdx, setEditIdx] = useState(-1);        // segment editor open for this row
  const [showRaw, setShowRaw] = useState(false);
  const [recompiling, setRecompiling] = useState(false);
  // which LLM writes the script (defaults to Claude; "local" = whatever local provider the backend picks)
  const [scriptLlm, setScriptLlm] = d.use("h3scriptLlm", "claude-sonnet-4-6");

  const h3cast = cast.filter((c) => c.style === "h3" && c.sheetId);
  const patchSeg = (i: number, p: Partial<H3Segment>) =>
    setSegments((prev) => (prev as H3Segment[]).map((s, j) => (j === i ? { ...s, ...p } : s)));

  // the cast payload the script writer + compiler see: identity look, sheet, chosen costume + prop
  function castPayload() {
    return h3cast.map((c) => {
      const co = (c.costumes || []).find((x) => x.id === (castCostume[c.id] ?? (c.costumes || [])[0]?.id));
      const pr = (c.props || []).find((x) => x.id === (castProp[c.id] ?? (c.props || [])[0]?.id));
      return {
        name: c.name, role: c.role, look: c.appearance || "",
        sheet_id: c.sheetId,
        // "" selection = "(sheet outfit)": no costume ref, the identity sheet's wardrobe is worn
        ...(co?.stillId && castCostume[c.id] !== "" ? { costume: { name: co.name, desc: co.desc, still_id: co.stillId } } : {}),
        ...(pr?.stillId && castProp[c.id] !== "" ? { prop: { name: pr.name, desc: pr.desc, still_id: pr.stillId } } : {}),
      };
    });
  }

  async function writeScript() {
    if (!audioId) { ctx.setResults([{ id: rid(), title: "Pick the song first", status: "error", pct: 0, err: "H3 segments are built from the real audio structure - set the song track above." }]); return; }
    if (!songPayload) { ctx.setResults([{ id: rid(), title: "No song arrangement", status: "error", pct: 0, err: "Open a project with a Song arrangement (the Song tab)." }]); return; }
    setWriting(true);
    const card = { id: rid(), title: "H3 script - analyzing audio structure…", status: "running" as const, pct: 3 };
    ctx.setResults([card]);
    // The call is one synchronous request (audio analysis ~1-2 min, then the writer LLM ~3-8 min),
    // so real progress isn't available - show STAGED progress with a live elapsed clock instead of
    // a frozen bar. Time-based pct creeps to 90 and holds until the response lands.
    const t0 = Date.now();
    const EXPECT = 7 * 60_000;                       // typical total; bar reaches 90% here
    const tick = window.setInterval(() => {
      const el = Date.now() - t0;
      const mm = Math.floor(el / 60_000), ss = String(Math.floor(el / 1000) % 60).padStart(2, "0");
      const stage = el < 100_000 ? "analyzing audio structure" : "writer LLM drafting segments";
      ctx.patch(card.id, {
        pct: Math.min(90, Math.round((el / EXPECT) * 90)),
        title: `H3 script - ${stage}… ${mm}:${ss} (usually 4-8 min total)`,
      });
    }, 1000);
    try {
      const llm = scriptLlm === "local" ? {} : { provider: "claude_sub", model: scriptLlm };
      const r = await api.mvH3Script({ song: songPayload, audio_id: audioId, cast: castPayload(), ...llm }) as
        { segments: H3Segment[]; singles: number; scenes: number };
      setSegments(r.segments.map((s) => ({ ...s })));
      ctx.patch(card.id, { status: "done", pct: 100, title: "H3 script written",
        err: `${r.segments.length} segments (${r.singles} single / ${r.scenes} scene)` });
    } catch (e) { ctx.patch(card.id, { status: "error", pct: 0, title: "H3 script failed", err: (e as Error).message }); }
    finally { window.clearInterval(tick); setWriting(false); }
  }

  // ---- per-segment editing: change the structured shot fields, then RECOMPILE server-side so
  // the enforced rules (duration anchors, sky-pin, subject-swap, doubled framing) stay intact ----
  const patchShot = (i: number, j: number, p: Partial<H3Shot>) =>
    setSegments((prev) => (prev as H3Segment[]).map((s, k) =>
      k === i ? { ...s, shots: s.shots.map((sh, m) => (m === j ? { ...sh, ...p } : sh)) } : s));
  async function recompile(i: number) {
    const seg = segments[i];
    setRecompiling(true);
    try {
      const r = await api.mvH3Compile({ segment: seg, cast: castPayload() }) as
        { prompt: string; picture_map: Record<string, number>; outfit_map: Record<string, number>;
          env_picture: number; lipsync: boolean; shots: H3Shot[]; kind: "single" | "scene"; cuts: { start: number; end: number }[] };
      patchSeg(i, { prompt: r.prompt, picture_map: r.picture_map, outfit_map: r.outfit_map,
        env_picture: r.env_picture, lipsync: r.lipsync, shots: r.shots, kind: r.kind, cuts: r.cuts,
        handEdited: false });
      ctx.setResults([{ id: rid(), title: `segment ${i + 1} recompiled`, status: "done", pct: 100 }]);
    } catch (e) { ctx.setResults([{ id: rid(), title: `segment ${i + 1} recompile failed`, status: "error", pct: 0, err: (e as Error).message }]); }
    finally { setRecompiling(false); }
  }

  // ---- environment stills (required: every compiled prompt references <Picture env>) ----
  async function genEnv(i: number) {
    const seg = segments[i];
    const scene = seg.shots[0]?.scene || "";
    if (!scene) { ctx.setResults([{ id: rid(), title: `segment ${i + 1}`, status: "error", pct: 0, err: "No scene description on this segment." }]); return; }
    setGenning(`env:${i}`);
    try {
      const r = await api.videoStill(envRequest(scene, Math.floor(Math.random() * 2_000_000_000))) as { job_id: string };
      patchSeg(i, { envStillId: r.job_id });
      const card = { id: rid(), title: `segment ${i + 1} environment`, status: "running" as const, pct: 5 };
      ctx.setResults([card]);
      pollJob(r.job_id, card.id, ctx);
    } catch (e) { ctx.setResults([{ id: rid(), title: `segment ${i + 1} environment`, status: "error", pct: 0, err: (e as Error).message }]); }
    finally { setGenning(""); }
  }
  async function genAllEnvs() {
    for (let i = 0; i < segments.length; i++) if (!segments[i].envStillId) await genEnv(i);
  }

  // refs in the compiled prompt's exact picture order: sheets, outfits, environment
  function refsFor(seg: H3Segment): string[] | string {
    const byName = Object.fromEntries(h3cast.map((c) => [c.name, c]));
    const sheets = Object.entries(seg.picture_map).sort((a, b) => a[1] - b[1])
      .map(([n]) => byName[n]?.sheetId).filter(Boolean) as string[];
    if (sheets.length !== Object.keys(seg.picture_map).length) return "a cast member in this segment has no identity sheet";
    const outfits = Object.entries(seg.outfit_map || {}).sort((a, b) => a[1] - b[1])
      .map(([n]) => {
        const c = byName[n];
        if (c && castCostume[c.id] === "") return undefined;   // (sheet outfit) selected after compile
        const co = (c?.costumes || []).find((x) => x.id === (castCostume[c!.id] ?? (c?.costumes || [])[0]?.id));
        return co?.stillId;
      }).filter(Boolean) as string[];
    if (outfits.length !== Object.keys(seg.outfit_map || {}).length) return "a costume still is missing";
    const props = Object.entries(seg.prop_map || {}).sort((a, b) => a[1] - b[1])
      .map(([n]) => {
        const c = byName[n];
        const pr = (c?.props || []).find((x) => x.id === (castProp[c!.id] ?? (c?.props || [])[0]?.id));
        return pr?.stillId;
      }).filter(Boolean) as string[];
    if (props.length !== Object.keys(seg.prop_map || {}).length) return "a prop still is missing";
    if (!seg.envStillId) return "generate the environment still first";
    return [...sheets, ...outfits, ...props, seg.envStillId];
  }

  async function renderSeg(i: number, mode: "hunt" | "finish", seed?: number) {
    const seg = segments[i];
    const refs = refsFor(seg);
    if (typeof refs === "string") { ctx.setResults([{ id: rid(), title: `segment ${i + 1}`, status: "error", pct: 0, err: refs }]); return; }
    setGenning(`seg:${i}`);
    const card = { id: rid(), title: `segment ${i + 1} ${seg.kind}${seg.lipsync ? " ♪" : ""} (${mode})`, status: "pending" as const, pct: 0 };
    ctx.setResults([card]);
    try {
      const body: Record<string, unknown> = {
        prompt: seg.prompt, ref_still_ids: refs,
        seconds: seg.render_seconds, mode,
        ...(seed ? { seed } : {}),
        ...(mode === "hunt" ? { drafts: 2 } : {}),
      };
      if (seg.lipsync && audioId)
        body.ref_audio_ids = [{ id: audioId, start: seg.start, seconds: seg.render_seconds }];
      const r = await api.videoH3Ref2V(body) as
        { mode?: string; drafts?: { job_id: string; seed: number }[]; job_id?: string };
      if (r.drafts) {
        setDrafts((s) => ({ ...s, [i]: r.drafts!.map((x) => ({ jobId: x.job_id, seed: x.seed })) }));
        ctx.patch(card.id, { status: "running", pct: 5 });
        r.drafts.forEach((x) => {
          const c2 = { id: rid(), title: `segment ${i + 1} draft (seed ${x.seed})`, status: "running" as const, pct: 2 };
          ctx.setResults([c2]);
          pollJob(x.job_id, c2.id, ctx);
        });
        ctx.patch(card.id, { status: "done", pct: 100, err: "2 drafts rendering - pick one, then Finish with its seed" });
      } else if (r.job_id) {
        patchSeg(i, { clipId: r.job_id, clipVariants: [...(seg.clipVariants || []), r.job_id] });
        ctx.patch(card.id, { status: "running", pct: 5 });
        pollJob(r.job_id, card.id, ctx);
      }
    } catch (e) { ctx.patch(card.id, { status: "error", pct: 0, err: (e as Error).message }); }
    finally { setGenning(""); }
  }

  async function assemble() {
    const ready = segments.filter((s) => s.clipId);
    if (!ready.length) { ctx.setResults([{ id: rid(), title: "Nothing to assemble", status: "error", pct: 0, err: "Render some segments first." }]); return; }
    const card = { id: rid(), title: `Assemble (${ready.length} segments)`, status: "pending" as const, pct: 0 };
    ctx.setResults([card]);
    try {
      const { job_id } = await api.mvAssemble({
        shots: ready.map((s) => ({ clip_id: s.clipId, start: s.start, end: s.end })),
        audio_id: audioId, grade, width: resW, height: resH,
      }) as { job_id: string };
      ctx.patch(card.id, { status: "running", pct: 5 });
      pollJob(job_id, card.id, ctx);
    } catch (e) { ctx.patch(card.id, { status: "error", pct: 0, err: (e as Error).message }); }
  }

  const clipUrl = (id?: string) => (id ? library.find((x) => x.id === id)?.media_url : undefined);

  return (
    <div className="space-y-3">
      <p className="text-[10px] text-[var(--color-muted)]">
        The MiniMax H3 pipeline: the writer cuts the song into <span className="text-[var(--color-ink)]">render segments</span> on
        its real structure ("single" = one shot, always for lip-sync; "scene" = several cuts in one render). Each segment references
        the cast's <span className="text-[var(--color-ink)]">identity sheets + costumes</span> and its own
        <span className="text-[var(--color-ink)]"> environment still</span>; lip-sync segments carry the song itself, so the
        performance is synced to the actual track.
      </p>

      {/* cast + costume selection */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[10px] uppercase tracking-wide text-[var(--color-muted)]">Cast</span>
        {h3cast.length === 0 && <span className="text-[10px] text-amber-400">no H3-style characters with identity sheets yet - build them in Characters</span>}
        {h3cast.map((c) => (
          <span key={c.id} className="flex items-center gap-1 rounded border border-[var(--color-line)] px-1.5 py-0.5 text-[10px]">
            {c.sheetId && <img src={`/api/media/${c.sheetId}`} className="h-5 w-9 rounded object-cover" alt="" />}
            {c.name}
            {(c.costumes || []).length > 0 && (
              <select className="bg-transparent text-[9px] text-[var(--color-muted)]" value={castCostume[c.id] ?? (c.costumes || [])[0]?.id}
                onChange={(e) => setCastCostume({ ...castCostume, [c.id]: e.target.value })} title="costume for this video ((sheet outfit) = wear what the identity sheet shows)">
                {(c.costumes || []).map((co) => <option key={co.id} value={co.id}>{co.name}</option>)}
                <option value="">(sheet outfit)</option>
              </select>
            )}
            {(c.props || []).length > 0 && (
              <select className="bg-transparent text-[9px] text-[var(--color-muted)]" value={castProp[c.id] ?? (c.props || [])[0]?.id}
                onChange={(e) => setCastProp({ ...castProp, [c.id]: e.target.value })} title="instrument/prop for this video">
                {(c.props || []).map((pr) => <option key={pr.id} value={pr.id}>{pr.name}</option>)}
                <option value="">(no prop)</option>
              </select>
            )}
          </span>
        ))}
        <span className="flex-1" />
        <select className="rounded border border-[var(--color-line)] bg-transparent px-1.5 py-0.5 text-[10px] text-[var(--color-muted)]"
          value={scriptLlm} onChange={(e) => setScriptLlm(e.target.value)} disabled={writing}
          title="Which LLM writes the script">
          <option value="claude-sonnet-4-6">Claude Sonnet 4.6</option>
          <option value="opus">Claude Opus</option>
          <option value="local">Local LLM</option>
        </select>
        <PrimaryButton onClick={writeScript} disabled={busy || writing || !audioId}>
          {writing ? "writing…" : segments.length ? "↻ rewrite script" : "Write H3 script"}
        </PrimaryButton>
      </div>

      {/* segment list */}
      {segments.length > 0 && (
        <>
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-[var(--color-muted)]">
              {segments.length} segments {"·"} {segments.filter((s) => s.kind === "single").length} single / {segments.filter((s) => s.kind === "scene").length} scene
              {"·"} {segments.filter((s) => s.clipId).length} rendered
            </span>
            <span className="flex-1" />
            <GhostButton onClick={genAllEnvs} disabled={busy || !!genning}>Gen missing environments</GhostButton>
            <PrimaryButton onClick={assemble} disabled={busy || !segments.some((s) => s.clipId)}>Assemble</PrimaryButton>
          </div>
          <div className="overflow-hidden rounded-lg border border-[var(--color-line)]">
            <div className="max-h-[520px] overflow-y-auto">
              <table className="w-full border-collapse text-left text-[11px]">
                <thead className="sticky top-0 z-10 bg-[var(--color-panel2)] text-[9px] uppercase tracking-wide text-[var(--color-muted)]">
                  <tr>
                    <th className="w-7 px-1 py-1.5">#</th><th className="w-24 px-2 py-1.5">time</th>
                    <th className="w-20 px-2 py-1.5">kind</th><th className="px-2 py-1.5">content</th>
                    <th className="w-24 px-2 py-1.5">environment</th><th className="w-24 px-2 py-1.5">clip</th>
                    <th className="w-40 px-2 py-1.5">render</th>
                  </tr>
                </thead>
                <tbody>
                  {segments.map((seg, i) => {
                    const url = clipUrl(seg.clipId);
                    const segDrafts = drafts[i] || [];
                    return (<>
                      <tr key={i} className="border-t border-[var(--color-line)] align-top">
                        <td className="px-1 py-1.5 font-semibold text-[var(--color-muted)]">{i + 1}</td>
                        <td className="px-2 py-1.5 text-[var(--color-muted)]">
                          {fmt(seg.start)}–{fmt(seg.end)}
                          <div className="text-[9px] opacity-70">{seg.section} {"·"} renders {seg.render_seconds}s</div>
                        </td>
                        <td className="px-2 py-1.5">
                          <span className={`rounded px-1.5 py-0.5 text-[9px] ${seg.lipsync ? "bg-[#3a2a14] text-[var(--color-accent2)]" : seg.kind === "scene" ? "bg-sky-900/50 text-sky-200" : "bg-slate-700/60 text-slate-300"}`}>
                            {seg.kind}{seg.lipsync ? " ♪" : ""}{seg.kind === "scene" ? ` ×${seg.shots.length}` : ""}
                          </span>
                        </td>
                        <td className="px-2 py-1.5">
                          {seg.shots.map((s, j) => (
                            <div key={j} className="truncate text-[10px] text-[var(--color-muted)]" title={`${s.scene} — ${s.action}`}>
                              <span className="text-[var(--color-ink)]">{s.characters.join("+") || s.type}</span> {s.framing} {"·"} {s.action.slice(0, 60)}
                            </div>
                          ))}
                        </td>
                        <td className="px-2 py-1.5">
                          {seg.envStillId
                            ? <img src={`/api/media/${seg.envStillId}`} onClick={() => openLightbox(`/api/media/${seg.envStillId}`)}
                                className="h-9 w-16 cursor-zoom-in rounded object-cover" alt="" title="environment still — click to enlarge" />
                            : <button onClick={() => genEnv(i)} disabled={busy || !!genning}
                                className="rounded border border-[var(--color-line)] px-1.5 py-0.5 text-[9px] text-[var(--color-muted)] hover:text-[var(--color-ink)] disabled:opacity-50">
                                {genning === `env:${i}` ? "…" : "gen env"}
                              </button>}
                        </td>
                        <td className="px-2 py-1.5">
                          {url
                            ? <video src={`${url}#t=0.5`} muted preload="metadata" onClick={() => openLightbox(url)}
                                className="h-9 w-16 cursor-zoom-in rounded object-cover" title="rendered clip — click to view" />
                            : <span className="text-[9px] text-[var(--color-muted)]">—</span>}
                        </td>
                        <td className="px-2 py-1.5">
                          <div className="flex flex-wrap gap-1">
                            <button onClick={() => { setEditIdx(editIdx === i ? -1 : i); setShowRaw(false); }}
                              title="edit this segment's shots + prompt"
                              className={`rounded border px-1.5 py-0.5 text-[9px] ${editIdx === i ? "border-[var(--color-accent2)] text-[var(--color-accent2)]" : "border-[var(--color-line)] text-[var(--color-muted)] hover:text-[var(--color-ink)]"}`}>
                              {editIdx === i ? "close" : "edit"}
                            </button>
                            <button onClick={() => renderSeg(i, "hunt")} disabled={busy || !!genning || !seg.envStillId}
                              title="2 fast turbo drafts to pick a seed"
                              className="rounded border border-[var(--color-line)] px-1.5 py-0.5 text-[9px] text-[var(--color-muted)] hover:text-[var(--color-ink)] disabled:opacity-50">hunt</button>
                            <button onClick={() => renderSeg(i, "finish")} disabled={busy || !!genning || !seg.envStillId}
                              title="one full-quality render (base recipe)"
                              className="rounded border border-[var(--color-line)] px-1.5 py-0.5 text-[9px] text-[var(--color-muted)] hover:text-[var(--color-ink)] disabled:opacity-50">finish</button>
                          </div>
                          {segDrafts.length > 0 && (
                            <div className="mt-1 flex gap-1">
                              {segDrafts.map((x) => (
                                <button key={x.jobId} onClick={() => {
                                    patchSeg(i, { clipId: x.jobId, clipVariants: [...(seg.clipVariants || []), x.jobId] });
                                    ctx.setResults([{ id: rid(), title: `✓ segment ${i + 1}: draft seed ${x.seed} selected`, status: "done", pct: 100 }]);
                                  }}
                                  title={`use draft seed ${x.seed} (or Finish with this seed for full quality)`}
                                  className={`rounded px-1 py-0.5 text-[8px] ${seg.clipId === x.jobId ? "bg-[var(--color-accent2)] text-black" : "border border-[var(--color-line)] text-[var(--color-muted)] hover:text-[var(--color-ink)]"}`}>
                                  s{String(x.seed).slice(-3)}
                                </button>
                              ))}
                              {seg.clipId && segDrafts.some((x) => x.jobId === seg.clipId) && (
                                <button onClick={() => renderSeg(i, "finish", segDrafts.find((x) => x.jobId === seg.clipId)?.seed)}
                                  disabled={busy || !!genning}
                                  title="re-render the picked draft's seed on the full-quality base recipe"
                                  className="rounded border border-[var(--color-accent2)] px-1 py-0.5 text-[8px] text-[var(--color-accent2)] disabled:opacity-50">
                                  finish seed
                                </button>
                              )}
                            </div>
                          )}
                        </td>
                      </tr>
                      {editIdx === i && (
                        <tr key={`edit-${i}`} className="border-t border-[var(--color-line)] bg-[var(--color-bg)]">
                          <td colSpan={7} className="px-3 py-2">
                            <div className="space-y-2">
                              {seg.shots.map((s, j) => (
                                <div key={j} className="rounded border border-[var(--color-line)] p-2 space-y-1.5">
                                  <div className="flex flex-wrap items-center gap-2 text-[9px] text-[var(--color-muted)]">
                                    <span className="font-semibold uppercase tracking-wide">{seg.kind === "scene" ? `cut ${j + 1}` : "shot"}</span>
                                    <label className="flex items-center gap-1">framing
                                      <select className={`${inp} !w-auto`} value={s.framing} onChange={(e) => patchShot(i, j, { framing: e.target.value })}>
                                        {["close", "medium", "wide"].map((f) => <option key={f} value={f}>{f}</option>)}
                                      </select>
                                    </label>
                                    <label className="flex items-center gap-1">camera
                                      <select className={`${inp} !w-auto`} value={s.camera} onChange={(e) => patchShot(i, j, { camera: e.target.value })}>
                                        {H3_CAMERAS.map((cm) => <option key={cm} value={cm}>{cm}</option>)}
                                      </select>
                                    </label>
                                    <label className="flex items-center gap-1" title="the singer performs the lyrics on camera in this shot (forces single-kind + close/medium)">
                                      <input type="checkbox" checked={s.lipsync} onChange={(e) => patchShot(i, j, { lipsync: e.target.checked })} /> lip-sync
                                    </label>
                                    <span className="flex-1" />
                                    <span title="named cast in this shot (from the script; edit via rewrite)">{s.characters.join(", ") || "no cast"}</span>
                                  </div>
                                  <label className="block text-[9px] text-[var(--color-muted)]">scene (environment - also drives the env still)
                                    <textarea className={inp} rows={2} value={s.scene} onChange={(e) => patchShot(i, j, { scene: e.target.value })} />
                                  </label>
                                  <label className="block text-[9px] text-[var(--color-muted)]">action (ONE continuous motion at real-world speed)
                                    <textarea className={inp} rows={2} value={s.action} onChange={(e) => patchShot(i, j, { action: e.target.value })} />
                                  </label>
                                </div>
                              ))}
                              <div className="flex items-center gap-2">
                                <PrimaryButton onClick={() => recompile(i)} disabled={recompiling || busy}>
                                  {recompiling ? "recompiling…" : "Recompile prompt"}
                                </PrimaryButton>
                                <span className="text-[9px] text-[var(--color-muted)]">
                                  recompiling re-applies the enforced rules (pace anchors, sky-pin, subject-swap, framing){seg.handEdited ? " and REPLACES your raw edit" : ""}
                                </span>
                                <span className="flex-1" />
                                <GhostButton onClick={() => setShowRaw(!showRaw)}>{showRaw ? "hide raw prompt" : "raw prompt"}</GhostButton>
                              </div>
                              {showRaw && (
                                <div className="space-y-1">
                                  <textarea className={`${inp} font-mono`} rows={14} value={seg.prompt}
                                    onChange={(e) => patchSeg(i, { prompt: e.target.value, handEdited: true })} />
                                  <p className="text-[9px] text-[var(--color-muted)]">
                                    {seg.handEdited
                                      ? "⚠ hand-edited - this exact text renders; Recompile discards it. Keep the six-section format and the <Picture/Subject/Audio N> tags."
                                      : "the compiled prompt - edit it directly for full control (renders verbatim)"}
                                  </p>
                                </div>
                              )}
                            </div>
                          </td>
                        </tr>
                      )}
                    </>);
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
      {segments.length === 0 && (
        <div className="rounded-lg border border-dashed border-[var(--color-line)] p-6 text-center text-xs text-[var(--color-muted)]">
          No segments yet. Set the song track, check the cast, then <span className="text-[var(--color-ink)]">Write H3 script</span>.
        </div>
      )}
    </div>
  );
}
