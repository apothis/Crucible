import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { api, type Config, type LibItem } from "./api";
import { useDraftCtx, useDrafts } from "./drafts";
import { Field, GhostButton, PrimaryButton, SectionTitle, inp, type RunCtx } from "./ui";
import { openLightbox } from "./Lightbox";

// Music Video tab, phase 1 (docs/MV_UI_DESIGN.md): the storyboard of a saved H3 script. Scenes
// in song order with their opening still and status; a drawer per scene with the writer's
// fields, the clip prompts, the gated render (dry run first, Render only on an unchanged dry
// run), and the takes with their checks and the lock. The backend stays the source of truth:
// scripts are the files in .mvwork/mv_scripts, takes are library rows, locks live in the script.

type Clip = { index: number; song_start: number; song_end: number; seconds: number; frames: number; lines?: string[]; vocal?: boolean; prompt?: string; problems?: string[] };
type Locked = { job_id: string; take_start: number; script: string };
type Scene = {
  index: number; title: string; start: number; end: number; location?: string; setting?: string; performance?: string;
  framing?: string; camera?: string; lighting?: string; mood?: string; action?: string; others?: string; why?: string;
  opening_still?: string; opening_still_id?: string; lipsync?: boolean; scene_problems?: string[]; clips?: Clip[];
  locked?: Locked; plan?: { frames: number; context_frames: number }; sections?: string[];
};
type Refs = { face?: string; body?: string; shared?: string };
type Script = { title: string; audio_id: string; duration: number; style?: string; concept?: string; look?: string; wardrobe?: string; arc?: string; refs?: Refs; scenes: Scene[] };
type ScriptRow = { name: string; title: string; audio_id: string; mtime: number; duration: number; scenes: number; clips: number; stills: number; locks: number; problems: number };
type Check = { floor_r: number | null; floor_lag: number | null; floor_ok: boolean; opening_jump: number | null; opening_flag: boolean; sheet_url: string; seconds: number | null; planned_seconds: number | null };
type Upscale = { id: string; status: string; progress: number; max: number; error?: string | null; media_url: string | null; engine?: string; scale?: number; fps?: number | null; fps_fixed?: boolean | null };
type Take = { id: string; created: number; status: string; progress: number; max: number; media_url: string | null; seconds: number | null; params: Record<string, any>; check: Check | null; from_script?: string | null; upscaled?: Upscale | null };
type Cut = { stamp: string; file: string; path: string; seconds: number | null; scenes: number | null; partial: boolean; script: string | null; jobs: Record<string, string>; intro: boolean; end_card: { secs: number; text: string } | null; library_id?: string | null; media_url?: string | null; mtime: number };
type Dry = { settings: Record<string, any>; flags: string[]; prompt_hash: string; prompts: string[]; clips: number; lane: string; ref_ids: string[]; first_frame_id?: string | null; sig: string };

const ATTENTION = "comfy kitchen attention";   // the proven chain setting (every SSBS fire script)
// FlashVSR+ Turbo 2x took 301 s for a 6 s take (144 frames) on the box, 2026-10-30, the one
// timing so far and it includes a model load: only a rough guide for the time shown before a fire
const UPSCALE_SECS_PER_SEC = 50;
const upscaleBusy = (u?: Upscale | null) => !!u && u.status !== "done" && u.status !== "error";

function fmt(t: number) {
  const m = Math.floor(t / 60), s = t - m * 60;
  return `${m}:${s.toFixed(2).padStart(5, "0")}`;
}
function ago(ts: number) {
  const d = (Date.now() / 1000 - ts) / 60;
  if (d < 60) return `${Math.max(1, Math.round(d))} min ago`;
  if (d < 60 * 24) return `${Math.round(d / 60)} h ago`;
  return `${Math.round(d / 60 / 24)} d ago`;
}
function audioLabel(it: LibItem): string {
  const p = it.params || {};
  return ((p.title || p.tags || p.source || it.mode || it.id).toString().slice(0, 56)) + (p.take ? ` · v${p.take}` : "");
}

function Pill({ tone, children, title }: { tone: "ok" | "warn" | "bad" | "muted" | "accent"; children: ReactNode; title?: string }) {
  const cls = tone === "ok" ? "text-emerald-500 border-emerald-800"
    : tone === "warn" ? "text-amber-400 border-amber-700"
    : tone === "bad" ? "text-red-400 border-red-800"
    : tone === "accent" ? "text-[var(--color-accent)] border-[var(--color-accent)]"
    : "text-[var(--color-muted)] border-[var(--color-line)]";
  return <span title={title} className={`rounded-full border px-2 py-0.5 text-[10px] whitespace-nowrap ${cls}`}>{children}</span>;
}

function Row({ k, v }: { k: string; v?: ReactNode }) {
  if (!v) return null;
  return (
    <div className="grid grid-cols-[92px_1fr] gap-2 text-xs">
      <div className="text-[var(--color-muted)]">{k}</div>
      <div className="whitespace-pre-wrap">{v}</div>
    </div>
  );
}

export function MusicVideoForm({ library, onDone }: { cfg: Config; busy: boolean; library: LibItem[] } & RunCtx) {
  const d = useDrafts("mv");
  const [audioId, setAudioId] = d.use("audioId", "");
  const [scriptName, setScriptName] = d.use("scriptName", "");
  const [sel, setSel] = d.use<number | null>("sel", null);
  // render settings: the proven recipe as defaults (turbo 1344x768, two reference stills)
  const [faceId, setFaceId] = d.use("faceId", "2c1c49e7c1a34a47878c4d416d6fb009");
  const [bodyId, setBodyId] = d.use("bodyId", "cd98a0e41c1a4579944e5aff23239e25");
  const [seed, setSeed] = d.use("seed", "20261027");
  const [turbo, setTurbo] = d.use("turbo", true);
  const [width, setWidth] = d.use("width", 1344);
  const [height, setHeight] = d.use("height", 768);
  const [acceptFlags, setAcceptFlags] = d.use("acceptFlags", false);

  const [scripts, setScripts] = useState<ScriptRow[]>([]);
  const [allScripts, setAllScripts] = useState(false);
  const [script, setScript] = useState<Script | null>(null);
  const [takes, setTakes] = useState<Record<string, Take[]>>({});
  const [dry, setDry] = useState<Record<number, Dry>>({});
  const [busyMsg, setBusyMsg] = useState("");
  const [err, setErr] = useState("");
  const [copied, setCopied] = useState(false);
  const [bumpTick, setBumpTick] = useState(0);
  const fileRef = useRef<HTMLInputElement | null>(null);

  const audios = library.filter((i) => i.audio_url);

  // ---------- loading ----------
  const loadScripts = useCallback(async () => {
    try { setScripts(await api.mvScripts(allScripts ? undefined : (audioId || undefined))); }
    catch (e) { setErr(String((e as Error).message || e)); }
  }, [audioId, allScripts]);
  useEffect(() => { loadScripts(); }, [loadScripts]);

  // NOTE: no drafts setter in these deps: useDrafts hands out a fresh setter every render, and a
  // loader that depends on one re-runs after every render (an endless request loop, found in the
  // first browser check 2026-10-30). The track is synced from the loaded script in its own effect.
  useEffect(() => {
    let alive = true;
    (async () => {
      if (!scriptName) { setScript(null); setTakes({}); return; }
      try {
        const r = await api.mvScript(scriptName);
        if (!alive) return;
        setScript(r.script as Script);
        const t = await api.mvTakes(scriptName);
        if (alive) setTakes(t.takes || {});
      } catch (e) { if (alive) setErr(String((e as Error).message || e)); }
    })();
    return () => { alive = false; };
  }, [scriptName, bumpTick]);
  useEffect(() => {
    if (script?.audio_id && !audioId) setAudioId(script.audio_id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [script?.audio_id]);

  const refreshTakes = useCallback(async () => {
    if (!scriptName) return;
    try { const t = await api.mvTakes(scriptName); setTakes(t.takes || {}); } catch { /* keep the last list */ }
  }, [scriptName]);

  // poll while any take is still rendering
  const rendering = useMemo(() => Object.values(takes).some((l) => l.some((t) => t.status !== "done" || upscaleBusy(t.upscaled))), [takes]);
  useEffect(() => {
    if (!rendering) return;
    const h = window.setInterval(refreshTakes, 5000);
    return () => window.clearInterval(h);
  }, [rendering, refreshTakes]);

  const scene = script && sel !== null ? script.scenes[sel] : null;
  const sceneTakes = (idx: number) => takes[String(idx)] || [];

  // ---------- status per scene ----------
  type Status = { still: boolean; dryOk: boolean | null; flags: number; takes: number; running: boolean; locked: boolean; problems: number };
  const statusOf = (sc: Scene): Status => {
    const dr = dry[sc.index];
    const tl = sceneTakes(sc.index);
    return {
      still: !!sc.opening_still_id,
      dryOk: dr ? dr.flags.length === 0 : null,
      flags: dr ? dr.flags.length : 0,
      takes: tl.filter((t) => t.status === "done").length,
      running: tl.some((t) => t.status !== "done"),
      locked: !!sc.locked,
      problems: (sc.scene_problems || []).length,
    };
  };
  const tone = (s: Status) => s.locked ? "bg-emerald-700" : s.running ? "bg-amber-500" : s.takes ? "bg-sky-700" : s.still ? "bg-zinc-500" : "bg-zinc-800";

  // ---------- actions ----------
  const sig = () => `${faceId}|${bodyId}|${seed}|${turbo}|${width}|${height}`;
  const renderBody = (idx: number, dryRun: boolean, pinSeed?: number) => {
    const body: Record<string, unknown> = {
      script: scriptName, scene: idx, ref_ids: [faceId, bodyId].map((x) => x.trim()).filter(Boolean),
      turbo, attention: ATTENTION, width: Number(width), height: Number(height), dry_run: dryRun,
    };
    const s = pinSeed ?? (seed.trim() ? Number(seed) : undefined);
    if (s !== undefined && !Number.isNaN(s)) body.seed = s;
    return body;
  };
  const doDry = async (idx: number) => {
    setErr(""); setBusyMsg("dry run...");
    try {
      const r = await api.mvRenderScene(renderBody(idx, true));
      setDry((p) => ({ ...p, [idx]: { ...(r as Dry), sig: sig() } }));
    } catch (e) { setErr(String((e as Error).message || e)); }
    finally { setBusyMsg(""); }
  };
  const canRender = (idx: number) => {
    const dr = dry[idx];
    return !!dr && dr.sig === sig() && (dr.flags.length === 0 || acceptFlags);
  };
  const doRender = async (idx: number) => {
    const dr = dry[idx];
    if (!dr || !canRender(idx)) return;
    if (!window.confirm(`Render scene ${idx} on the box now? (${dr.clips} clip${dr.clips === 1 ? "" : "s"}, ${dr.settings?.width}x${dr.settings?.height}, seed ${dr.settings?.seed})`)) return;
    setErr(""); setBusyMsg("submitting to the box (30-60 s)...");
    try {
      await api.mvRenderScene(renderBody(idx, false, Number(dr.settings?.seed)));
      await refreshTakes();
    } catch (e) { setErr(String((e as Error).message || e)); }
    finally { setBusyMsg(""); }
  };
  const doCheck = async (idx: number, jid: string) => {
    setErr(""); setBusyMsg("measuring the take (Mac, no GPU)...");
    try { await api.mvTakeCheck({ job_id: jid, script: scriptName, scene: idx }); await refreshTakes(); }
    catch (e) { setErr(String((e as Error).message || e)); }
    finally { setBusyMsg(""); }
  };
  const doLock = async (idx: number, jid: string | null) => {
    setErr("");
    try { await api.mvLockTake({ script: scriptName, scene: idx, job_id: jid }); setBumpTick((n) => n + 1); }
    catch (e) { setErr(String((e as Error).message || e)); }
  };
  const doUpscale = async (idx: number, t: Take) => {
    const mins = Math.max(1, Math.round(((t.seconds || 6) * UPSCALE_SECS_PER_SEC) / 60));
    if (!window.confirm(`Upscale this take 2x with FlashVSR+ on the box now? (${t.params.width}x${t.params.height} -> 2x, roughly ${mins} min)`)) return;
    setErr(""); setBusyMsg("sending the take to the box...");
    try {
      const r = await api.mvUpscaleTakes({ jobs: { [String(idx)]: t.id }, force: t.upscaled?.status === "error" || t.upscaled?.status === "done" });
      if (!r.queued?.length && r.skipped?.length) setErr(`not queued: ${r.skipped[0].why}`);
      await refreshTakes();
    } catch (e) { setErr(String((e as Error).message || e)); }
    finally { setBusyMsg(""); }
  };
  const fileStill = async (idx: number, file: File) => {
    setErr(""); setBusyMsg("filing the still...");
    try {
      const fd = new FormData();
      fd.append("file", file, file.name || "still.png");
      fd.append("script", scriptName);
      fd.append("scene", String(idx));
      await api.mvSceneStill(fd);
      setBumpTick((n) => n + 1);
    } catch (e) { setErr(String((e as Error).message || e)); }
    finally { setBusyMsg(""); }
  };
  const onPaste = (e: React.ClipboardEvent) => {
    if (!scene) return;
    const item = Array.from(e.clipboardData.items).find((it) => it.type.startsWith("image/"));
    const f = item?.getAsFile();
    if (f) { e.preventDefault(); fileStill(scene.index, f); }
  };
  const copyPrompt = async () => {
    if (!scene?.opening_still) return;
    try { await navigator.clipboard.writeText(scene.opening_still); setCopied(true); window.setTimeout(() => setCopied(false), 1500); }
    catch { setErr("clipboard blocked by the browser; select the prompt text and copy it"); }
  };

  return (
    <div className="space-y-5" onPaste={onPaste}>
      <p className="text-sm text-[var(--color-muted)]">
        The storyboard of a <b>music video script</b>: every scene with its opening still and status, the gated render
        (a dry run first, then Render), and the takes with their checks and the lock that carries a passed take into the cut.
      </p>

      {/* ---------- 1 Song and script ---------- */}
      <SectionTitle>1 · Song and script</SectionTitle>
      <div className="grid gap-3 md:grid-cols-[1fr_auto] md:items-end">
        <Field label="Track" hint="the song the script was written for">
          <select className={inp} value={audioId} onChange={(e) => { setAudioId(e.target.value); setScriptName(""); setSel(null); }}>
            <option value="">— pick a track —</option>
            {audios.map((a) => <option key={a.id} value={a.id}>{audioLabel(a)}</option>)}
          </select>
        </Field>
        <label className="flex items-center gap-2 text-xs text-[var(--color-muted)] pb-2">
          <input type="checkbox" checked={allScripts} onChange={(e) => setAllScripts(e.target.checked)} /> show scripts for every song
        </label>
      </div>
      <SongPanel audioId={audioId} onImported={(id) => { onDone(); setAudioId(id); setScriptName(""); setSel(null); }} />
      {audioId && (
        <WriterPanel audioId={audioId} loaded={script && script.audio_id === audioId ? { name: scriptName, script } : null}
                     onWritten={async (name) => { await loadScripts(); setScriptName(name); setSel(null); setDry({}); }} />
      )}
      {scripts.length === 0 ? (
        <div className="text-xs text-[var(--color-muted)]">No saved scripts{audioId && !allScripts ? " for this track" : ""} yet: write one above.</div>
      ) : (
        <div className="max-h-56 overflow-auto rounded border border-[var(--color-line)]">
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-[var(--color-panel)] text-[var(--color-muted)]">
              <tr><th className="px-2 py-1 text-left">script</th><th className="px-2 py-1 text-right">scenes</th><th className="px-2 py-1 text-right">clips</th><th className="px-2 py-1 text-right">stills</th><th className="px-2 py-1 text-right">locked</th><th className="px-2 py-1 text-right">problems</th><th className="px-2 py-1 text-right">saved</th></tr>
            </thead>
            <tbody>
              {scripts.map((s) => (
                <tr key={s.name} onClick={() => { setScriptName(s.name); setSel(null); setDry({}); }}
                    className={`cursor-pointer border-t border-[var(--color-line)] hover:bg-[var(--color-panel)] ${s.name === scriptName ? "bg-[var(--color-panel)] text-[var(--color-accent)]" : ""}`}>
                  <td className="px-2 py-1 font-mono" title={s.name}>{s.title || s.name}<span className="ml-2 text-[10px] text-[var(--color-muted)]">{s.name.replace(/\.json$/, "").slice(-40)}</span></td>
                  <td className="px-2 py-1 text-right">{s.scenes}</td>
                  <td className="px-2 py-1 text-right">{s.clips}</td>
                  <td className="px-2 py-1 text-right">{s.stills}</td>
                  <td className="px-2 py-1 text-right">{s.locks}</td>
                  <td className={`px-2 py-1 text-right ${s.problems ? "text-amber-400" : ""}`}>{s.problems}</td>
                  <td className="px-2 py-1 text-right text-[var(--color-muted)]">{ago(s.mtime)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {script && (
        <>
          {/* ---------- 2 Scenes ---------- */}
          <div className="flex flex-wrap items-baseline gap-3">
            <SectionTitle>2 · Scenes</SectionTitle>
            <span className="text-xs text-[var(--color-muted)]">
              {script.title} · {fmt(script.duration)} · {script.scenes.length} scenes · {script.scenes.filter((s) => s.locked).length} locked · {script.scenes.filter((s) => s.opening_still_id).length} stills
              {rendering ? " · rendering..." : ""}
            </span>
          </div>
          {/* the strip: one chip per scene over the song, proportional */}
          <div className="flex h-5 w-full overflow-hidden rounded border border-[var(--color-line)]">
            {script.scenes.map((sc) => {
              const st = statusOf(sc);
              const w = Math.max(0.5, (100 * (sc.end - sc.start)) / (script.duration || 1));
              return (
                <div key={sc.index} onClick={() => setSel(sc.index)} title={`${sc.index} ${sc.title} ${fmt(sc.start)}-${fmt(sc.end)}`}
                     style={{ width: `${w}%` }}
                     className={`cursor-pointer border-r border-black/40 ${tone(st)} ${sel === sc.index ? "outline outline-2 outline-[var(--color-accent)]" : ""}`} />
              );
            })}
          </div>
          <div className="flex flex-wrap gap-3 text-[10px] text-[var(--color-muted)]">
            <span><i className="inline-block h-2 w-3 bg-emerald-700" /> locked</span>
            <span><i className="inline-block h-2 w-3 bg-sky-700" /> has takes</span>
            <span><i className="inline-block h-2 w-3 bg-amber-500" /> rendering</span>
            <span><i className="inline-block h-2 w-3 bg-zinc-500" /> still filed</span>
            <span><i className="inline-block h-2 w-3 bg-zinc-800" /> nothing yet</span>
          </div>

          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {script.scenes.map((sc) => {
              const st = statusOf(sc);
              return (
                <div key={sc.index} onClick={() => setSel(sc.index)}
                     className={`cursor-pointer overflow-hidden rounded-lg border bg-[var(--color-panel)] ${sel === sc.index ? "border-[var(--color-accent)]" : "border-[var(--color-line)]"}`}>
                  <div className="relative aspect-video w-full bg-black">
                    {sc.opening_still_id
                      ? <img src={`/api/media/${sc.opening_still_id}`} alt="" className="h-full w-full object-cover" loading="lazy" />
                      : <div className="flex h-full w-full items-center justify-center text-[11px] text-[var(--color-muted)]">no opening still</div>}
                    <div className="absolute left-1 top-1 rounded bg-black/70 px-1.5 py-0.5 text-[10px] text-white">{sc.index}</div>
                    <div className="absolute right-1 top-1 rounded bg-black/70 px-1.5 py-0.5 text-[10px] text-white">{fmt(sc.start)} - {fmt(sc.end)}</div>
                  </div>
                  <div className="space-y-1.5 p-2">
                    <div className="truncate text-sm" title={sc.title}>{sc.title}</div>
                    <div className="truncate text-[11px] text-[var(--color-muted)]">{sc.framing}{sc.performance ? ` · ${sc.performance.replace(/_/g, " ")}` : ""}{sc.lipsync ? " · lip-sync" : ""}</div>
                    <div className="flex flex-wrap gap-1">
                      <Pill tone={st.still ? "ok" : "muted"}>{st.still ? "still" : "no still"}</Pill>
                      {st.dryOk === null ? <Pill tone="muted">no dry run</Pill> : st.dryOk ? <Pill tone="ok">dry run ok</Pill> : <Pill tone="warn">{st.flags} flag{st.flags === 1 ? "" : "s"}</Pill>}
                      {st.running ? <Pill tone="accent">rendering</Pill> : <Pill tone={st.takes ? "ok" : "muted"}>{st.takes} take{st.takes === 1 ? "" : "s"}</Pill>}
                      {st.locked ? <Pill tone="ok">LOCKED</Pill> : null}
                      {st.problems ? <Pill tone="bad" title={(sc.scene_problems || []).join("\n")}>{st.problems} problem{st.problems === 1 ? "" : "s"}</Pill> : null}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </>
      )}

      {/* ---------- 3 Scene drawer ---------- */}
      {script && scene && (
        <div className="rounded-lg border border-[var(--color-accent)] bg-[var(--color-panel)] p-4 space-y-4">
          <div className="flex flex-wrap items-center gap-3">
            <SectionTitle>Scene {scene.index} · {scene.title}</SectionTitle>
            <span className="text-xs text-[var(--color-muted)]">{fmt(scene.start)} - {fmt(scene.end)} ({(scene.end - scene.start).toFixed(2)} s) · {scene.clips?.length || 0} clip{(scene.clips?.length || 0) === 1 ? "" : "s"} · {scene.plan ? `${scene.plan.frames} frames` : ""}</span>
            <div className="ml-auto flex gap-2">
              <GhostButton onClick={() => setSel(Math.max(0, scene.index - 1))} disabled={scene.index === 0}>prev</GhostButton>
              <GhostButton onClick={() => setSel(Math.min(script.scenes.length - 1, scene.index + 1))} disabled={scene.index >= script.scenes.length - 1}>next</GhostButton>
              <GhostButton onClick={() => setSel(null)}>close</GhostButton>
            </div>
          </div>
          {(scene.scene_problems || []).length > 0 && (
            <div className="rounded border border-red-800 p-2 text-xs text-red-300">{(scene.scene_problems || []).map((p, i) => <div key={i}>{p}</div>)}</div>
          )}

          <div className="grid gap-4 lg:grid-cols-[1fr_1fr]">
            {/* left: still + fields + clip prompts */}
            <div className="space-y-3">
              <div className="relative aspect-video w-full overflow-hidden rounded bg-black">
                {scene.opening_still_id
                  ? <img src={`/api/media/${scene.opening_still_id}`} alt="" className="h-full w-full cursor-zoom-in object-contain" onClick={() => openLightbox(`/api/media/${scene.opening_still_id}`)} />
                  : <div className="flex h-full w-full flex-col items-center justify-center gap-2 border-2 border-dashed border-[var(--color-line)] text-xs text-[var(--color-muted)]">
                      <div>no opening still: paste the GPT image here (Cmd+V) or choose a file</div>
                    </div>}
              </div>
              <div className="flex flex-wrap gap-2">
                <GhostButton onClick={copyPrompt} disabled={!scene.opening_still}>{copied ? "copied" : "copy still prompt"}</GhostButton>
                <GhostButton onClick={() => fileRef.current?.click()}>{scene.opening_still_id ? "replace still" : "choose still file"}</GhostButton>
                <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) fileStill(scene.index, f); e.target.value = ""; }} />
                <span className="self-center text-[11px] text-[var(--color-muted)]">a pasted image anywhere on this panel files as this scene's still</span>
              </div>
              {scene.opening_still && (
                <details className="text-xs">
                  <summary className="cursor-pointer text-[var(--color-muted)]">still prompt (paste-ready)</summary>
                  <div className="mt-1 whitespace-pre-wrap rounded border border-[var(--color-line)] p-2">{scene.opening_still}</div>
                </details>
              )}
              <div className="space-y-1">
                <Row k="setting" v={scene.setting} />
                <Row k="performance" v={scene.performance} />
                <Row k="framing" v={scene.framing} />
                <Row k="camera" v={scene.camera} />
                <Row k="action" v={scene.action} />
                <Row k="others" v={scene.others} />
                <Row k="lighting" v={scene.lighting} />
                <Row k="mood" v={scene.mood} />
                <Row k="why" v={scene.why} />
                <Row k="location" v={scene.location} />
              </div>
              <details className="text-xs">
                <summary className="cursor-pointer text-[var(--color-muted)]">clip prompts ({scene.clips?.length || 0})</summary>
                <div className="mt-2 space-y-2">
                  {(scene.clips || []).map((c) => (
                    <div key={c.index} className="rounded border border-[var(--color-line)] p-2">
                      <div className="mb-1 text-[var(--color-muted)]">clip {c.index} · {fmt(c.song_start)} - {fmt(c.song_end)} · {c.frames} frames{c.vocal ? " · sung" : ""}{(c.lines || []).length ? ` · "${(c.lines || []).join(" / ")}"` : ""}</div>
                      <pre className="whitespace-pre-wrap font-sans text-[11px]">{c.prompt}</pre>
                      {(c.problems || []).length ? <div className="mt-1 text-red-300">{(c.problems || []).join("; ")}</div> : null}
                    </div>
                  ))}
                </div>
              </details>
            </div>

            {/* right: render + takes */}
            <div className="space-y-3">
              <div className="rounded border border-[var(--color-line)] p-3 space-y-2">
                <div className="text-xs font-semibold">Render</div>
                <div className="grid grid-cols-2 gap-2">
                  <Field label="face ref (still id)"><input className={inp} value={faceId} onChange={(e) => setFaceId(e.target.value)} /></Field>
                  <Field label="body ref (still id)"><input className={inp} value={bodyId} onChange={(e) => setBodyId(e.target.value)} /></Field>
                </div>
                <div className="flex gap-2">
                  {[faceId, bodyId].filter((x) => x.trim()).map((x) => <img key={x} src={`/api/media/${x.trim()}`} alt="" className="h-14 rounded border border-[var(--color-line)] object-cover" onError={(e) => { (e.target as HTMLImageElement).style.opacity = "0.2"; }} />)}
                </div>
                <div className="grid grid-cols-2 gap-2 xl:grid-cols-4">
                  <Field label="seed" hint="blank = drawn by the dry run"><input className={inp} value={seed} onChange={(e) => setSeed(e.target.value)} /></Field>
                  <Field label="width"><input className={inp} type="number" step={32} value={width} onChange={(e) => setWidth(Number(e.target.value))} /></Field>
                  <Field label="height"><input className={inp} type="number" step={32} value={height} onChange={(e) => setHeight(Number(e.target.value))} /></Field>
                  <label className="flex items-end gap-2 pb-2 text-xs"><input type="checkbox" checked={turbo} onChange={(e) => setTurbo(e.target.checked)} /> turbo</label>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <GhostButton onClick={() => doDry(scene.index)} disabled={!!busyMsg}>Dry run</GhostButton>
                  <PrimaryButton onClick={() => doRender(scene.index)} disabled={!!busyMsg || !canRender(scene.index)}
                                 title={canRender(scene.index) ? "queue this scene on the box" : "run a dry run with these settings first; flags need the accept tick"}>Render on the box</PrimaryButton>
                  <label className="flex items-center gap-1 text-[11px] text-[var(--color-muted)]"><input type="checkbox" checked={acceptFlags} onChange={(e) => setAcceptFlags(e.target.checked)} /> accept flags</label>
                  {busyMsg && <span className="text-[11px] text-[var(--color-muted)]">{busyMsg}</span>}
                </div>
                {dry[scene.index] && (() => {
                  const dr = dry[scene.index];
                  const stale = dr.sig !== sig();
                  return (
                    <div className={`rounded border p-2 text-[11px] ${stale ? "border-amber-700" : dr.flags.length ? "border-amber-700" : "border-emerald-800"}`}>
                      <div className="flex flex-wrap gap-x-3 gap-y-1">
                        <span>lane {dr.lane}</span><span>{dr.clips} clip{dr.clips === 1 ? "" : "s"}</span>
                        <span>{dr.settings?.width}x{dr.settings?.height}</span><span>seed {dr.settings?.seed}</span>
                        <span>{dr.settings?.turbo ? "turbo" : "base"}</span>
                        {dr.first_frame_id ? <span>frame 0 = still</span> : null}
                        <span>refs {dr.ref_ids?.length || 0}</span>
                        {stale ? <span className="text-amber-400">settings changed since this dry run</span> : null}
                      </div>
                      {dr.flags.length ? <ul className="mt-1 list-disc pl-4 text-amber-300">{dr.flags.map((f, i) => <li key={i}>{f}</li>)}</ul> : <div className="mt-1 text-emerald-500">no flags</div>}
                    </div>
                  );
                })()}
              </div>

              <div className="space-y-2">
                <div className="flex items-center gap-2 text-xs font-semibold">Takes <span className="font-normal text-[var(--color-muted)]">({sceneTakes(scene.index).length})</span>
                  <GhostButton onClick={refreshTakes} className="ml-auto">refresh</GhostButton>
                </div>
                {sceneTakes(scene.index).length === 0 && <div className="text-xs text-[var(--color-muted)]">no renders of this scene yet</div>}
                {sceneTakes(scene.index).map((t) => {
                  const isLocked = scene.locked?.job_id === t.id;
                  const c = t.check;
                  return (
                    <div key={t.id} className={`rounded border p-2 space-y-1.5 ${isLocked ? "border-emerald-700" : "border-[var(--color-line)]"}`}>
                      {t.status === "done" && t.media_url
                        ? <video src={t.media_url} controls muted preload="metadata" className="w-full rounded bg-black" />
                        : <div className="rounded bg-black/40 p-2 text-xs text-[var(--color-muted)]">{t.status}{t.max ? ` ${Math.round((100 * t.progress) / t.max)}%` : ""} on the box...</div>}
                      <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[11px] text-[var(--color-muted)]">
                        <span>seed {t.params.seed}</span><span>{t.params.width}x{t.params.height}</span>
                        <span>{t.params.lane}{t.params.turbo ? " turbo" : ""}</span>
                        {t.seconds !== null ? <span>{t.seconds.toFixed(2)} s{c?.planned_seconds ? ` / ${c.planned_seconds.toFixed(2)} planned` : ""}</span> : null}
                        {t.params.upscale_width ? <span>latent upscale {t.params.upscale_width}x{t.params.upscale_height}</span> : null}
                        <span>{t.created ? ago(t.created) : ""}</span>
                        {t.from_script ? <Pill tone="muted" title={`rendered under ${t.from_script}; same song window`}>earlier script</Pill> : null}
                      </div>
                      {c ? (
                        <div className="flex flex-wrap items-center gap-2">
                          <Pill tone={c.floor_r === null ? "muted" : c.floor_ok ? "ok" : "bad"} title="song-envelope floor vs the scene window; clean takes measure 0.87-1.00">floor {c.floor_r === null ? "n/a" : c.floor_r.toFixed(2)}{c.floor_lag ? ` @${c.floor_lag > 0 ? "+" : ""}${c.floor_lag}s` : ""}</Pill>
                          <Pill tone={c.opening_jump === null ? "muted" : c.opening_flag ? "bad" : "ok"} title="opening 2 s change over the take's own baseline; >= 4 = fades or morphs in">opening {c.opening_jump === null ? "n/a" : c.opening_jump.toFixed(1)}x</Pill>
                          <GhostButton onClick={() => openLightbox(c.sheet_url + "?t=" + Date.now())}>contact sheet</GhostButton>
                        </div>
                      ) : <div className="text-[11px] text-[var(--color-muted)]">not checked yet</div>}
                      {t.status === "done" && (
                        <div className="flex flex-wrap gap-2">
                          <GhostButton onClick={() => doCheck(scene.index, t.id)} disabled={!!busyMsg}>{c ? "re-check" : "check"}</GhostButton>
                          {isLocked
                            ? <GhostButton onClick={() => doLock(scene.index, null)}>unlock</GhostButton>
                            : <PrimaryButton onClick={() => doLock(scene.index, t.id)} title="pass this take: it goes into the cut and survives rewrites">pass and lock</PrimaryButton>}
                          {isLocked && <Pill tone="ok">LOCKED{scene.locked && scene.locked.script !== scriptName ? ` (from ${scene.locked.script.slice(-24)})` : ""}</Pill>}
                        </div>
                      )}
                      {t.status === "done" && (() => {
                        const u = t.upscaled;
                        return (
                          <div className="flex flex-wrap items-center gap-2 text-[11px]">
                            {u?.status === "done" && u.media_url
                              ? <><Pill tone="ok" title={`FlashVSR+ upscale${u.fps ? `, ${u.fps} fps` : ""}${u.fps_fixed ? " (rate corrected after download)" : ""}`}>upscaled {u.scale || 2}x</Pill>
                                  <GhostButton onClick={() => openLightbox(u.media_url!)}>play upscale</GhostButton></>
                              : upscaleBusy(u)
                                ? <Pill tone="warn">upscaling {u!.max ? `${Math.round((100 * u!.progress) / u!.max)}%` : u!.status}</Pill>
                                : <GhostButton onClick={() => doUpscale(scene.index, t)} disabled={!!busyMsg} title="FlashVSR+ Turbo 2x on the box (GPU)">upscale 2x (FlashVSR+)</GhostButton>}
                            {u?.status === "done" && <GhostButton onClick={() => doUpscale(scene.index, t)} disabled={!!busyMsg} title="queue another upscale of this take">redo</GhostButton>}
                            {u?.status === "error" && <span className="text-red-400" title={u.error || ""}>upscale failed{u.error ? `: ${u.error.slice(0, 80)}` : ""}</span>}
                          </div>
                        );
                      })()}
                    </div>
                  );
                })}
                {scene.locked && !sceneTakes(scene.index).some((t) => t.id === scene.locked!.job_id) && (
                  <div className="rounded border border-emerald-800 p-2 text-[11px]">
                    LOCKED to a take rendered under an earlier script name: <span className="font-mono">{scene.locked.job_id.slice(0, 8)}</span>
                    <video src={`/api/media/${scene.locked.job_id}`} controls muted preload="metadata" className="mt-1 w-full rounded bg-black" />
                    <div className="mt-1"><GhostButton onClick={() => doLock(scene.index, null)}>unlock</GhostButton></div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {script && <AssembleStage script={script} scriptName={scriptName} takes={takes} onUpscaleQueued={refreshTakes} />}

      {script && <UploadText script={script} />}

      {err && <div className="rounded border border-red-800 p-2 text-xs text-red-300">{err}</div>}
    </div>
  );
}

// ---------- 4 Assemble: locked takes -> the cut, with the ident in front and the end card ----------
type AsmResult = { output: string; seconds: number; scenes: number; partial?: boolean; missing?: number[]; intro?: { seconds: number; cached: boolean }; end_card?: { secs: number; text: string }; endcard_at?: number | null; library_id?: string; media_url?: string; upscaled?: Record<string, string>; not_upscaled?: number[] };

function AssembleStage({ script, scriptName, takes, onUpscaleQueued }: { script: Script; scriptName: string; takes: Record<string, Take[]>; onUpscaleQueued: () => Promise<void> }) {
  const d = useDrafts("mv");
  const [size, setSize] = d.use("asmSize", "1344x768");
  const [useUpscaled, setUseUpscaled] = d.use("asmUseUpscaled", false);
  const [upBusy, setUpBusy] = useState(false);
  const [upMsg, setUpMsg] = useState("");
  const [fillNewest, setFillNewest] = d.use("asmFillNewest", true);
  const [introOn, setIntroOn] = d.use("asmIntroOn", true);
  const [introTitle, setIntroTitle] = d.use("asmIntroTitle", "");
  const [introSub, setIntroSub] = d.use("asmIntroSub", "Official Video");
  const [introEdge, setIntroEdge] = d.use("asmIntroEdge", "none");
  const [ecOn, setEcOn] = d.use("asmEcOn", true);
  const [ecSecs, setEcSecs] = d.use("asmEcSecs", "15");
  const [ecText, setEcText] = d.use("asmEcText", "");
  const [ident, setIdent] = useState<{ url: string; seconds: number; cached: boolean } | null>(null);
  const [identBusy, setIdentBusy] = useState(false);
  const [job, setJob] = useState<{ id: string; status: string; progress: number; max: number } | null>(null);
  const [result, setResult] = useState<AsmResult | null>(null);
  const [err, setErr] = useState("");
  const [cuts, setCuts] = useState<Cut[]>([]);
  const [useCut, setUseCut] = useState<string>("");     // stamp of an earlier cut whose take map is reused
  const title = introTitle.trim() || script.title;
  useEffect(() => {
    let alive = true;
    api.mvAssemblies(scriptName).then((r) => { if (alive) setCuts(r as Cut[]); }).catch(() => { if (alive) setCuts([]); });
    return () => { alive = false; };
  }, [scriptName, result?.output]);
  const cutMap = (cuts.find((c) => c.stamp === useCut)?.jobs) || null;

  // the take per scene: an earlier cut's map when chosen, else the lock, else (opt-in) the newest
  // finished take (takes from earlier scripts with the same window count), else missing
  const plan = script.scenes.map((sc) => {
    const done = (takes[String(sc.index)] || []).filter((t) => t.status === "done");
    if (cutMap && cutMap[String(sc.index)]) return { sc, job: cutMap[String(sc.index)], how: "cut" as const };
    if (sc.locked) return { sc, job: sc.locked.job_id, how: "locked" as const };
    if (fillNewest && done[0]) return { sc, job: done[0].id, how: done[0].from_script ? ("earlier" as const) : ("newest" as const) };
    return { sc, job: null, how: "missing" as const };
  });
  const missing = plan.filter((p) => !p.job);
  const partial = missing.length > 0;
  const [w, h] = size.split("x").map(Number);

  // the upscale state of each planned take (every listed take, locked ones from earlier scripts too)
  const takeById = useMemo(() => {
    const m: Record<string, Take> = {};
    Object.values(takes).forEach((l) => l.forEach((t) => { m[t.id] = t; }));
    return m;
  }, [takes]);
  const upOf = (job: string | null) => (job ? takeById[job]?.upscaled || null : null);
  const planned = plan.filter((p) => p.job);
  const nUpDone = planned.filter((p) => upOf(p.job)?.status === "done").length;
  const nUpBusy = planned.filter((p) => upscaleBusy(upOf(p.job))).length;
  const toUpscale = planned.filter((p) => { const u = upOf(p.job); return !u || u.status === "error"; });
  const upscaleAll = async () => {
    if (!toUpscale.length) return;
    const secs = toUpscale.reduce((a, p) => a + (takeById[p.job!]?.seconds || (p.sc.end - p.sc.start)), 0);
    const mins = Math.max(1, Math.round((secs * UPSCALE_SECS_PER_SEC) / 60));
    if (!window.confirm(`Upscale ${toUpscale.length} take${toUpscale.length === 1 ? "" : "s"} 2x with FlashVSR+ on the box? They queue one after another: ${secs.toFixed(0)} s of video, roughly ${mins} min going by the one timing so far. Takes already upscaled or queued are left alone.`)) return;
    setErr(""); setUpMsg(""); setUpBusy(true);
    try {
      const jobs: Record<string, string> = {};
      toUpscale.forEach((p) => { jobs[String(p.sc.index)] = p.job!; });
      const r = await api.mvUpscaleTakes({ jobs });
      setUpMsg(`queued ${r.queued?.length || 0}${r.skipped?.length ? `, skipped ${r.skipped.length} (${r.skipped.map((s: { scene: number; why: string }) => `${s.scene}: ${s.why}`).join("; ")})` : ""}`);
      await onUpscaleQueued();
    } catch (e) { setErr(String((e as Error).message || e)); }
    finally { setUpBusy(false); }
  };

  const previewIdent = async () => {
    setErr(""); setIdentBusy(true);
    try { const r = await api.mvIdent({ title, sub: introSub, edge: introEdge }); setIdent({ url: r.url + "?t=" + Date.now(), seconds: r.seconds, cached: r.cached }); }
    catch (e) { setErr(String((e as Error).message || e)); }
    finally { setIdentBusy(false); }
  };
  const assemble = async () => {
    if (plan.every((p) => !p.job)) { setErr("no take to cut yet"); return; }
    const what = partial ? `a PARTIAL cut of ${plan.length - missing.length} of ${plan.length} scenes (no intro or end card on a partial)` : `the full cut of ${plan.length} scenes${introOn ? " with the ident in front" : ""}${ecOn ? ` and a ${ecSecs} s end card` : ""}`;
    const upNote = useUpscaled ? ` from the upscaled takes (${nUpDone} of ${planned.length} have one; the rest stay at render size)` : "";
    if (!window.confirm(`Assemble ${what} at ${size}${upNote}? This re-encodes on the Mac and can take a few minutes.`)) return;
    setErr(""); setResult(null);
    const jobs: Record<string, string> = {};
    plan.forEach((p) => { if (p.job && p.how !== "locked") jobs[String(p.sc.index)] = p.job; });
    try {
      const r = await api.mvAssemble({
        script: scriptName, jobs, partial, width: w, height: h, async: true, use_upscaled: useUpscaled,
        intro: introOn && !partial ? { title, sub: introSub, edge: introEdge } : null,
        end_card: ecOn && !partial ? { secs: Number(ecSecs), text: ecText.trim() } : null,
      });
      setJob({ id: r.job_id, status: "running", progress: 0, max: 1 });
    } catch (e) { setErr(String((e as Error).message || e)); }
  };
  useEffect(() => {
    if (!job || job.status !== "running") return;
    const h = window.setInterval(async () => {
      try {
        const r = await api.mvAssembleResult(job.id);
        if (r.status === "done") { setJob({ ...job, status: "done", progress: r.max, max: r.max }); setResult(r.result as AsmResult); window.clearInterval(h); }
        else if (r.status === "error") { setJob({ ...job, status: "error", progress: 0, max: 1 }); setErr(r.error || "assembly failed"); window.clearInterval(h); }
        else setJob((j) => (j ? { ...j, progress: r.progress, max: r.max } : j));
      } catch { /* keep polling */ }
    }, 3000);
    return () => window.clearInterval(h);
  }, [job?.id, job?.status]);   // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-baseline gap-3">
        <SectionTitle>4 · Assemble</SectionTitle>
        <span className="text-xs text-[var(--color-muted)]">{plan.length - missing.length} of {plan.length} scenes have a take{partial ? ` · ${missing.length} missing (partial cut only, no intro or end card)` : " · full cut"}</span>
      </div>
      <div className="max-h-48 overflow-auto rounded border border-[var(--color-line)]">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-[var(--color-panel)] text-[var(--color-muted)]"><tr><th className="px-2 py-1 text-left">#</th><th className="px-2 py-1 text-left">scene</th><th className="px-2 py-1 text-left">window</th><th className="px-2 py-1 text-left">take used</th><th className="px-2 py-1 text-left">upscale</th></tr></thead>
          <tbody>
            {plan.map((p) => (
              <tr key={p.sc.index} className="border-t border-[var(--color-line)]">
                <td className="px-2 py-1">{p.sc.index}</td>
                <td className="px-2 py-1">{p.sc.title}</td>
                <td className="px-2 py-1 text-[var(--color-muted)]">{fmt(p.sc.start)} - {fmt(p.sc.end)}</td>
                <td className="px-2 py-1">{p.how === "cut" ? <Pill tone="accent" title="the take this earlier cut used">cut {p.job!.slice(0, 8)}</Pill> : p.how === "locked" ? <Pill tone="ok">LOCKED {p.job!.slice(0, 8)}</Pill> : p.how === "newest" ? <Pill tone="warn" title="not passed: the newest finished take fills the gap">newest {p.job!.slice(0, 8)}</Pill> : p.how === "earlier" ? <Pill tone="warn" title="not passed: the newest take of this window, rendered under an earlier script">earlier {p.job!.slice(0, 8)}</Pill> : <Pill tone="bad">missing</Pill>}</td>
                <td className="px-2 py-1">{(() => { const u = upOf(p.job); return !p.job ? null : u?.status === "done" ? <Pill tone="ok">{u.scale || 2}x</Pill> : upscaleBusy(u) ? <Pill tone="warn">{u!.max ? `${Math.round((100 * u!.progress) / u!.max)}%` : u!.status}</Pill> : u?.status === "error" ? <Pill tone="bad" title={u.error || ""}>failed</Pill> : <span className="text-[var(--color-muted)]">-</span>; })()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {cuts.length > 0 && (
        <div className="rounded border border-[var(--color-line)] p-3 space-y-2">
          <div className="text-xs font-semibold">Earlier cuts of this song <span className="font-normal text-[var(--color-muted)]">({cuts.length}; a cut's take map can be reused as the starting point)</span></div>
          <div className="max-h-40 overflow-auto">
            <table className="w-full text-xs">
              <tbody>
                {cuts.map((c) => (
                  <tr key={c.stamp} className={`border-t border-[var(--color-line)] ${useCut === c.stamp ? "text-[var(--color-accent)]" : ""}`}>
                    <td className="px-2 py-1 font-mono text-[var(--color-muted)]">{c.stamp}</td>
                    <td className="px-2 py-1" title={c.path}>{c.file.replace(/\.mp4$/, "").slice(0, 60)}</td>
                    <td className="px-2 py-1 text-right">{c.seconds ? fmt(c.seconds) : ""}</td>
                    <td className="px-2 py-1 text-right">{c.scenes ?? "?"} sc{c.partial ? " (partial)" : ""}{c.intro ? " · ident" : ""}{c.end_card ? " · card" : ""}</td>
                    <td className="px-2 py-1 text-right whitespace-nowrap">
                      <GhostButton onClick={() => openLightbox(c.media_url || `/api/mv/cut/${c.stamp}/${encodeURIComponent(c.file)}`)}>play</GhostButton>{" "}
                      {Object.keys(c.jobs).length ? <GhostButton onClick={() => setUseCut(useCut === c.stamp ? "" : c.stamp)}>{useCut === c.stamp ? "using its takes" : "use its takes"}</GhostButton> : <span className="text-[10px] text-[var(--color-muted)]">no take map</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      <div className="grid gap-3 lg:grid-cols-3">
        <div className="rounded border border-[var(--color-line)] p-3 space-y-2">
          <div className="text-xs font-semibold">Cut</div>
          <Field label="output size">
            <select className={inp} value={size} onChange={(e) => setSize(e.target.value)}>
              <option value="1344x768">1344x768 (render size)</option>
              <option value="1920x1080">1920x1080</option>
              <option value="2560x1440">2560x1440</option>
              <option value="3840x2160">3840x2160</option>
            </select>
          </Field>
          <label className="flex items-center gap-2 text-xs"><input type="checkbox" checked={useUpscaled} onChange={(e) => setUseUpscaled(e.target.checked)} /> use the upscaled takes ({nUpDone} of {planned.length})</label>
          <div className="flex flex-wrap items-center gap-2">
            <GhostButton onClick={upscaleAll} disabled={upBusy || !toUpscale.length} title="FlashVSR+ Turbo 2x on the box (GPU) for every planned take without an upscale">
              {upBusy ? "queueing..." : toUpscale.length ? `upscale all ${toUpscale.length} remaining (FlashVSR+ 2x)` : "every planned take is upscaled or queued"}
            </GhostButton>
            {nUpBusy > 0 && <span className="text-[11px] text-[var(--color-muted)]">{nUpBusy} upscaling on the box</span>}
          </div>
          {upMsg && <div className="text-[11px] text-[var(--color-muted)]">{upMsg}</div>}
          <label className="flex items-center gap-2 text-xs"><input type="checkbox" checked={fillNewest} onChange={(e) => setFillNewest(e.target.checked)} /> fill unlocked scenes with their newest take</label>
        </div>
        <div className="rounded border border-[var(--color-line)] p-3 space-y-2">
          <label className="flex items-center gap-2 text-xs font-semibold"><input type="checkbox" checked={introOn} onChange={(e) => setIntroOn(e.target.checked)} /> Intro ident (3 s)</label>
          <Field label="title" hint="defaults to the script title"><input className={inp} value={introTitle} onChange={(e) => setIntroTitle(e.target.value)} placeholder={script.title} /></Field>
          <div className="grid grid-cols-2 gap-2">
            <Field label="sub line"><input className={inp} value={introSub} onChange={(e) => setIntroSub(e.target.value)} /></Field>
            <Field label="edge">
              <select className={inp} value={introEdge} onChange={(e) => setIntroEdge(e.target.value)}>
                <option value="none">none</option><option value="hairline">hairline</option><option value="burnt">burnt</option><option value="filigree">filigree (placeholder)</option>
              </select>
            </Field>
          </div>
          <div className="flex items-center gap-2">
            <GhostButton onClick={previewIdent} disabled={identBusy}>{identBusy ? "rendering..." : "preview ident"}</GhostButton>
            {ident && <span className="text-[11px] text-[var(--color-muted)]">{ident.seconds?.toFixed(2)} s{ident.cached ? " (cached)" : ""}</span>}
          </div>
          {ident && <video src={ident.url} controls preload="metadata" className="w-full rounded bg-black" />}
        </div>
        <div className="rounded border border-[var(--color-line)] p-3 space-y-2">
          <label className="flex items-center gap-2 text-xs font-semibold"><input type="checkbox" checked={ecOn} onChange={(e) => setEcOn(e.target.checked)} /> YouTube end card in the final</label>
          <div className="grid grid-cols-[auto_1fr] gap-2">
            <Field label="last">
              <select className={inp} value={ecSecs} onChange={(e) => setEcSecs(e.target.value)}><option value="10">10 s</option><option value="15">15 s</option><option value="20">20 s</option></select>
            </Field>
            <Field label="More from"><input className={inp} value={ecText} onChange={(e) => setEcText(e.target.value)} placeholder="More from Apotheon" /></Field>
          </div>
          <img src={`/api/youtube/endcard_preview?text=${encodeURIComponent(ecText)}&t=${encodeURIComponent(ecText.length)}`} alt="" className="w-full rounded border border-[var(--color-line)]" />
          <div className="text-[10px] text-[var(--color-muted)]">the plate dims the last seconds of the video; in Studio pick the "two videos + subscribe" end-screen template over it</div>
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-3">
        <PrimaryButton onClick={assemble} disabled={job?.status === "running" || plan.every((p) => !p.job)}>{partial ? "Assemble partial cut" : "Assemble the video"}</PrimaryButton>
        {job?.status === "running" && <span className="text-xs text-[var(--color-muted)]">cutting... {job.max ? Math.round((100 * job.progress) / job.max) : 0}%</span>}
      </div>
      {result && (
        <div className="rounded border border-emerald-800 p-3 space-y-2">
          <div className="text-xs">
            {result.partial ? "Partial cut" : "Final video"} · {result.seconds?.toFixed(1)} s · {result.scenes} scenes
            {result.intro ? ` · ident in front (${result.intro.seconds?.toFixed(1)} s)` : ""}{result.end_card ? ` · end card over the last ${result.end_card.secs} s` : ""}
            {result.upscaled ? ` · ${Object.keys(result.upscaled).length} upscaled takes${result.not_upscaled?.length ? `, ${result.not_upscaled.length} at render size (${result.not_upscaled.join(", ")})` : ""}` : ""}
            {result.library_id ? " · filed in the Library under Music videos" : ""}
          </div>
          {result.media_url ? <video src={result.media_url} controls preload="metadata" className="w-full rounded bg-black" /> : null}
          <div className="font-mono text-[10px] text-[var(--color-muted)]">{result.output}</div>
        </div>
      )}
      {err && <div className="rounded border border-red-800 p-2 text-xs text-red-300">{err}</div>}
    </div>
  );
}

// ---------- 1 Song: pick from the library or import from disk, its facts, the vocal stem ----------
type SongInfo = { audio_id: string; title: string; style: string; seconds: number | null; stem: boolean };

function SongPanel({ audioId, onImported }: { audioId: string; onImported: (id: string) => void }) {
  const [info, setInfo] = useState<SongInfo | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [fileTitle, setFileTitle] = useState("");
  const [busy, setBusy] = useState("");
  const [err, setErr] = useState("");
  const [tick, setTick] = useState(0);
  useEffect(() => {
    let alive = true;
    if (!audioId) { setInfo(null); return; }
    api.mvSongInfo(audioId).then((r) => { if (alive) setInfo(r as SongInfo); }).catch(() => { if (alive) setInfo(null); });
    return () => { alive = false; };
  }, [audioId, tick]);
  const importFile = async () => {
    if (!file) return;
    setErr(""); setBusy("importing...");
    try {
      const fd = new FormData();
      fd.append("file", file, file.name);
      if (fileTitle.trim()) fd.append("title", fileTitle.trim());
      const r = await api.importUpload(fd);
      setFile(null); setFileTitle("");
      onImported(r.job_id);
    } catch (e) { setErr(String((e as Error).message || e)); }
    finally { setBusy(""); }
  };
  const makeStem = async () => {
    if (!window.confirm("Separate the vocal stem on the box now? (BS-RoFormer; frees ComfyUI first, about a minute)")) return;
    setErr(""); setBusy("separating the vocals on the box...");
    try { await api.mvStems({ audio_id: audioId }); setTick((n) => n + 1); }
    catch (e) { setErr(String((e as Error).message || e)); }
    finally { setBusy(""); }
  };
  return (
    <div className="grid gap-3 lg:grid-cols-2">
      <div className="rounded border border-[var(--color-line)] p-3 space-y-2">
        <div className="text-xs font-semibold">Or import a song from disk</div>
        <div className="flex flex-wrap items-center gap-2">
          <input type="file" accept="audio/*,.wav,.flac,.mp3" className="text-[11px]" onChange={(e) => setFile(e.target.files?.[0] || null)} />
          <input className={`${inp} max-w-56`} value={fileTitle} onChange={(e) => setFileTitle(e.target.value)} placeholder="title (defaults to the file name)" />
          <GhostButton onClick={importFile} disabled={!file || !!busy}>import and use</GhostButton>
        </div>
        <div className="text-[10px] text-[var(--color-muted)]">wav, flac and mp3 are kept as they are; other formats become mp3. It lands in the Library as a source track.</div>
      </div>
      {info && (
        <div className="rounded border border-[var(--color-line)] p-3 space-y-1.5 text-xs">
          <div className="font-semibold">{info.title || "(untitled track)"} <span className="font-normal text-[var(--color-muted)]">{info.seconds ? fmt(info.seconds) : ""}</span></div>
          {info.style && <div className="line-clamp-2 text-[11px] text-[var(--color-muted)]" title={info.style}>{info.style}</div>}
          <div className="flex flex-wrap items-center gap-2">
            {info.stem ? <Pill tone="ok" title="lyric timing is measured on the isolated vocals">vocal stem cached</Pill>
              : <><Pill tone="warn" title="the writer times the lyrics on the full mix without it">no vocal stem</Pill>
                  <GhostButton onClick={makeStem} disabled={!!busy}>make vocal stem (box)</GhostButton></>}
          </div>
        </div>
      )}
      {(busy || err) && <div className={`text-[11px] lg:col-span-2 ${err ? "text-red-400" : "text-[var(--color-muted)]"}`}>{err || busy}</div>}
    </div>
  );
}

// ---------- the writer: Claude writes a whole script for the song (no GPU; the subscription CLI) ----------
function lyricsFromProject(get: (ns: string, key: string) => unknown): string {
  const suno = (get("suno", "lyrics") as string) || "";
  if (suno.trim()) return suno;
  const m3 = (get("music3", "lyrics") as string) || "";
  if (m3.trim()) return m3;
  const blocks = (get("song", "blocks") as { type?: string; lyrics?: string }[] | undefined) || [];
  return blocks.filter((b) => (b.lyrics || "").trim()).map((b) => `[${b.type || "Section"}]\n${(b.lyrics || "").trim()}`).join("\n\n");
}

function WriterPanel({ audioId, loaded, onWritten }: { audioId: string; loaded: { name: string; script: Script } | null; onWritten: (name: string) => Promise<void> }) {
  const d = useDrafts("mv");
  const drafts = useDraftCtx();
  const [open, setOpen] = useState(false);
  const [title, setTitle] = d.use("wTitle", "");
  const [style, setStyle] = d.use("wStyle", "");
  const [lyrics, setLyrics] = d.use("wLyrics", "");
  const [concept, setConcept] = d.use("wConcept", "");
  const [face, setFace] = d.use("wFace", "");
  const [body, setBody] = d.use("wBody", "");
  const [shared, setShared] = d.use("wShared", "");
  const [locations, setLocations] = d.use("wLocations", "auto");
  const [keepLocks, setKeepLocks] = d.use("wKeepLocks", true);
  const [forTrack, setForTrack] = d.use("wTrack", "");
  const [job, setJob] = useState<{ id: string; status: string; log: string[]; seconds: number; error?: string | null; script?: string | null } | null>(null);
  const [err, setErr] = useState("");

  // a different track: start from its own title and style (the refs describe the lead, so they stay)
  useEffect(() => {
    if (!audioId || forTrack === audioId) return;
    let alive = true;
    api.mvSongInfo(audioId).then((r) => {
      if (!alive) return;
      const i = r as SongInfo;
      setTitle(i.title || ""); setStyle(i.style || ""); setForTrack(audioId);
    }).catch(() => {});
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [audioId]);
  const fromScript = () => {
    if (!loaded) return;
    const r = loaded.script.refs || {};
    setFace(r.face || ""); setBody(r.body || ""); setShared(r.shared || "");
    if (!concept.trim() && loaded.script.concept) setConcept(loaded.script.concept);
  };
  const nLocks = loaded ? loaded.script.scenes.filter((s) => s.locked).length : 0;
  const ready = title.trim() && lyrics.trim() && face.trim() && body.trim() && shared.trim();

  const write = async () => {
    if (!ready) { setErr("title, lyrics and the three reference descriptions are needed"); return; }
    if (!window.confirm(`Write a new script for "${title}"? Claude (the subscription, no GPU) writes the whole video; it takes several minutes.${keepLocks && nLocks ? ` The ${nLocks} locked takes of the open script are kept.` : ""}`)) return;
    setErr("");
    try {
      const r = await api.mvWrite({
        async: true, audio_id: audioId, title: title.trim(), style: style.trim(), lyrics, concept: concept.trim() || undefined,
        refs: { face: face.trim(), body: body.trim(), shared: shared.trim() }, locations,
        keep_locks_from: keepLocks && nLocks && loaded ? loaded.name : undefined,
      });
      setJob({ id: r.job_id, status: "writing", log: [], seconds: 0 });
    } catch (e) { setErr(String((e as Error).message || e)); }
  };
  useEffect(() => {
    if (!job || job.status !== "writing") return;
    const h = window.setInterval(async () => {
      try {
        const r = await api.mvWriteJob(job.id);
        setJob({ id: job.id, status: r.status, log: r.log || [], seconds: r.seconds, error: r.error, script: r.script });
        if (r.status === "done" && r.script) { window.clearInterval(h); await onWritten(r.script); }
        if (r.status === "error") window.clearInterval(h);
      } catch { /* keep polling */ }
    }, 3000);
    return () => window.clearInterval(h);
  }, [job?.id, job?.status]);   // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="rounded border border-[var(--color-line)] p-3 space-y-2">
      <button className="flex w-full items-center gap-2 text-left text-xs font-semibold" onClick={() => setOpen(!open)}>
        <span>{open ? "v" : ">"}</span> Write a new script for this track
        {job?.status === "writing" && <span className="font-normal text-[var(--color-muted)]">writing... {Math.round(job.seconds)} s</span>}
      </button>
      {open && (
        <div className="space-y-2">
          <div className="grid gap-2 md:grid-cols-2">
            <Field label="title"><input className={inp} value={title} onChange={(e) => setTitle(e.target.value)} /></Field>
            <Field label="style" hint="the song's style line (genre, mood); filled from the track"><input className={inp} value={style} onChange={(e) => setStyle(e.target.value)} /></Field>
          </div>
          <Field label="lyrics" hint="section-tagged ([Verse], [Chorus] ...); the writer times them on this take">
            <textarea className={`${inp} h-40 font-mono text-[11px]`} value={lyrics} onChange={(e) => setLyrics(e.target.value)} placeholder="paste the lyrics, or fill them from the open project" />
          </Field>
          <div className="flex flex-wrap gap-2">
            <GhostButton onClick={() => { const t = lyricsFromProject(drafts.get); if (t.trim()) setLyrics(t); else setErr("the open project has no lyrics (Suno, Music 3 or Song builder)"); }}>lyrics from the open project</GhostButton>
            {loaded && <GhostButton onClick={fromScript} title="the lead's reference descriptions (and concept, if empty) of the open script">refs from the open script</GhostButton>}
          </div>
          <Field label="concept" hint="optional: your idea for the video; empty = Claude proposes one">
            <textarea className={`${inp} h-16`} value={concept} onChange={(e) => setConcept(e.target.value)} />
          </Field>
          <div className="grid gap-2 md:grid-cols-3">
            <Field label="face ref" hint="one line: what the face still shows"><textarea className={`${inp} h-20 text-[11px]`} value={face} onChange={(e) => setFace(e.target.value)} /></Field>
            <Field label="body ref" hint="one line: what the full-length still shows"><textarea className={`${inp} h-20 text-[11px]`} value={body} onChange={(e) => setBody(e.target.value)} /></Field>
            <Field label="shared" hint="what both stills show (the lead's constants)"><textarea className={`${inp} h-20 text-[11px]`} value={shared} onChange={(e) => setShared(e.target.value)} /></Field>
          </div>
          <div className="flex flex-wrap items-end gap-3">
            <Field label="places">
              <select className={inp} value={locations} onChange={(e) => setLocations(e.target.value)}>
                <option value="auto">this song's places (default)</option>
                <option value="library">every place with a still</option>
                <option value="none">invent all</option>
              </select>
            </Field>
            {nLocks > 0 && <label className="flex items-center gap-2 pb-2 text-xs"><input type="checkbox" checked={keepLocks} onChange={(e) => setKeepLocks(e.target.checked)} /> keep the open script's {nLocks} locked takes</label>}
            <PrimaryButton onClick={write} disabled={!ready || job?.status === "writing"}>{job?.status === "writing" ? "writing..." : "Write the script"}</PrimaryButton>
          </div>
          {job && (
            <div className={`rounded border p-2 text-[11px] ${job.status === "error" ? "border-red-800" : job.status === "done" ? "border-emerald-800" : "border-[var(--color-line)]"}`}>
              <div>{job.status === "writing" ? `writing, ${Math.round(job.seconds)} s` : job.status === "done" ? `written: ${job.script} (opened below)` : `failed: ${job.error}`}</div>
              {job.log.length > 0 && <pre className="mt-1 max-h-32 overflow-auto whitespace-pre-wrap text-[10px] text-[var(--color-muted)]">{job.log.slice(-12).join("\n")}</pre>}
            </div>
          )}
          {err && <div className="text-[11px] text-red-400">{err}</div>}
        </div>
      )}
    </div>
  );
}

// ---------- 5 YouTube upload text: the YouTube tab's writer, in its music-video form ----------
function UploadText({ script }: { script: Script }) {
  const d = useDrafts("mv");
  const drafts = useDraftCtx();
  const [notes, setNotes] = d.use("ytNotes", "");
  const [meta, setMeta] = d.use<{ video_title: string; description: string; tags: string[]; for?: string } | null>("ytMeta", null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const write = async () => {
    setErr(""); setBusy(true);
    try {
      const lyr = ((drafts.get("mv", "wLyrics") as string) || "").trim() || lyricsFromProject(drafts.get);
      const song = { title: script.title, tags: script.style || "", sections: lyr.trim() ? [{ lyrics: lyr }] : [] };
      const extra = [script.concept ? `The video: ${script.concept}` : "", notes.trim()].filter(Boolean).join("\n");
      const r = await api.ytMetadata({ title: script.title, song, notes: extra || undefined, kind: "music_video" }) as { video_title: string; description: string; tags: string[] };
      setMeta({ ...r, for: script.title });
    } catch (e) { setErr("Upload text failed: " + (e as Error).message); }
    finally { setBusy(false); }
  };
  return (
    <div className="space-y-3">
      <SectionTitle>5 · YouTube upload text</SectionTitle>
      <Field label="notes" hint="optional: anything the description should mention">
        <input className={inp} value={notes} onChange={(e) => setNotes(e.target.value)} />
      </Field>
      <GhostButton onClick={write} disabled={busy} title="Claude writes the title line, description (keywords + hashtags) and tags from the song, the lyrics and the video's concept">
        {busy ? "Writing..." : meta ? "Rewrite upload text" : "Write upload text"}
      </GhostButton>
      {meta && (
        <div className="space-y-2">
          {meta.for && meta.for !== script.title && <div className="text-[11px] text-amber-400">written for "{meta.for}"; rewrite it for this song</div>}
          {([["Video title", meta.video_title], ["Description", meta.description], ["Tags (comma-separated)", meta.tags.join(", ")]] as const).map(([label, value]) => (
            <div key={label} className="rounded-lg border border-[var(--color-line)] bg-[var(--color-panel2)] p-2">
              <div className="mb-1 flex items-center gap-2">
                <span className="text-[10px] font-medium uppercase tracking-wider text-[var(--color-muted)]">{label}</span>
                <button onClick={() => navigator.clipboard?.writeText(value)}
                  className="ml-auto rounded border border-[var(--color-line)] px-1.5 py-0.5 text-[10px] text-[var(--color-muted)] hover:text-[var(--color-ink)]">copy</button>
              </div>
              <pre className="whitespace-pre-wrap break-words text-[11px] leading-snug text-[var(--color-ink)]">{value}</pre>
            </div>
          ))}
          <p className="text-[10px] text-[var(--color-muted)]">
            Paste the description straight into YouTube's Description field (the hashtags on its last line are all you need; the first three show above the title).
            Tags go in the Tags box under "Show more". Remember to tick the altered/synthetic content disclosure.
          </p>
        </div>
      )}
      {err && <div className="rounded border border-red-800 p-2 text-xs text-red-300">{err}</div>}
    </div>
  );
}
