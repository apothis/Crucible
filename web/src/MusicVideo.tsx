import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { api, type Config, type LibItem } from "./api";
import { useDrafts } from "./drafts";
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
type Script = { title: string; audio_id: string; duration: number; concept?: string; look?: string; wardrobe?: string; arc?: string; scenes: Scene[] };
type ScriptRow = { name: string; title: string; audio_id: string; mtime: number; duration: number; scenes: number; clips: number; stills: number; locks: number; problems: number };
type Check = { floor_r: number | null; floor_lag: number | null; floor_ok: boolean; opening_jump: number | null; opening_flag: boolean; sheet_url: string; seconds: number | null; planned_seconds: number | null };
type Take = { id: string; created: number; status: string; progress: number; max: number; media_url: string | null; seconds: number | null; params: Record<string, any>; check: Check | null; from_script?: string | null };
type Cut = { stamp: string; file: string; path: string; seconds: number | null; scenes: number | null; partial: boolean; script: string | null; jobs: Record<string, string>; intro: boolean; end_card: { secs: number; text: string } | null; library_id?: string | null; media_url?: string | null; mtime: number };
type Dry = { settings: Record<string, any>; flags: string[]; prompt_hash: string; prompts: string[]; clips: number; lane: string; ref_ids: string[]; first_frame_id?: string | null; sig: string };

const ATTENTION = "comfy kitchen attention";   // the proven chain setting (every SSBS fire script)

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

export function MusicVideoForm({ library }: { cfg: Config; busy: boolean; library: LibItem[] } & RunCtx) {
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
  const rendering = useMemo(() => Object.values(takes).some((l) => l.some((t) => t.status !== "done")), [takes]);
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
      {scripts.length === 0 ? (
        <div className="text-xs text-[var(--color-muted)]">No saved scripts{audioId && !allScripts ? " for this track" : ""}. Scripts are written with /api/mv/write (the tab's writer lands in phase 3).</div>
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

      {script && <AssembleStage script={script} scriptName={scriptName} takes={takes} />}

      {err && <div className="rounded border border-red-800 p-2 text-xs text-red-300">{err}</div>}
    </div>
  );
}

// ---------- 4 Assemble: locked takes -> the cut, with the ident in front and the end card ----------
type AsmResult = { output: string; seconds: number; scenes: number; partial?: boolean; missing?: number[]; intro?: { seconds: number; cached: boolean }; end_card?: { secs: number; text: string }; endcard_at?: number | null; library_id?: string; media_url?: string };

function AssembleStage({ script, scriptName, takes }: { script: Script; scriptName: string; takes: Record<string, Take[]> }) {
  const d = useDrafts("mv");
  const [size, setSize] = d.use("asmSize", "1344x768");
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

  const previewIdent = async () => {
    setErr(""); setIdentBusy(true);
    try { const r = await api.mvIdent({ title, sub: introSub, edge: introEdge }); setIdent({ url: r.url + "?t=" + Date.now(), seconds: r.seconds, cached: r.cached }); }
    catch (e) { setErr(String((e as Error).message || e)); }
    finally { setIdentBusy(false); }
  };
  const assemble = async () => {
    if (plan.every((p) => !p.job)) { setErr("no take to cut yet"); return; }
    const what = partial ? `a PARTIAL cut of ${plan.length - missing.length} of ${plan.length} scenes (no intro or end card on a partial)` : `the full cut of ${plan.length} scenes${introOn ? " with the ident in front" : ""}${ecOn ? ` and a ${ecSecs} s end card` : ""}`;
    if (!window.confirm(`Assemble ${what} at ${size}? This re-encodes on the Mac and can take a few minutes.`)) return;
    setErr(""); setResult(null);
    const jobs: Record<string, string> = {};
    plan.forEach((p) => { if (p.job && p.how !== "locked") jobs[String(p.sc.index)] = p.job; });
    try {
      const r = await api.mvAssemble({
        script: scriptName, jobs, partial, width: w, height: h, async: true,
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
          <thead className="sticky top-0 bg-[var(--color-panel)] text-[var(--color-muted)]"><tr><th className="px-2 py-1 text-left">#</th><th className="px-2 py-1 text-left">scene</th><th className="px-2 py-1 text-left">window</th><th className="px-2 py-1 text-left">take used</th></tr></thead>
          <tbody>
            {plan.map((p) => (
              <tr key={p.sc.index} className="border-t border-[var(--color-line)]">
                <td className="px-2 py-1">{p.sc.index}</td>
                <td className="px-2 py-1">{p.sc.title}</td>
                <td className="px-2 py-1 text-[var(--color-muted)]">{fmt(p.sc.start)} - {fmt(p.sc.end)}</td>
                <td className="px-2 py-1">{p.how === "cut" ? <Pill tone="accent" title="the take this earlier cut used">cut {p.job!.slice(0, 8)}</Pill> : p.how === "locked" ? <Pill tone="ok">LOCKED {p.job!.slice(0, 8)}</Pill> : p.how === "newest" ? <Pill tone="warn" title="not passed: the newest finished take fills the gap">newest {p.job!.slice(0, 8)}</Pill> : p.how === "earlier" ? <Pill tone="warn" title="not passed: the newest take of this window, rendered under an earlier script">earlier {p.job!.slice(0, 8)}</Pill> : <Pill tone="bad">missing</Pill>}</td>
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
              <option value="3840x2160">3840x2160</option>
            </select>
          </Field>
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
