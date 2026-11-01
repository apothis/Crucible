import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { api, type Config, type LibItem } from "./api";
import { useDraftCtx, useDrafts } from "./drafts";
import { Field, GhostButton, inp, type RunCtx } from "./ui";
import { openLightbox } from "./Lightbox";

// Music Video tab (docs/MV_UI_DESIGN.md), laid out as six steps across the full width: the song,
// the script, the pictures (the singer's two reference stills and one opening still per scene),
// the takes (render each scene, choose the take that plays), the final video, the YouTube text.
// A "next" line under the steps always says what is left. The backend stays the source of truth:
// scripts are the files in .mvwork/mv_scripts, takes are library rows, the chosen take of a scene
// is its lock in the script (so a choice survives reloads and rewrites).

type Clip = { index: number; song_start: number; song_end: number; seconds: number; frames: number; lines?: string[]; vocal?: boolean; prompt?: string; problems?: string[] };
type Locked = { job_id: string; take_start: number; script: string };
type Scene = {
  index: number; title: string; start: number; end: number; location?: string; setting?: string; performance?: string;
  framing?: string; camera?: string; lighting?: string; mood?: string; action?: string; others?: string; why?: string;
  opening_still?: string; opening_still_id?: string; end_still_id?: string; rail?: { end_still_id: string; every: number; stills: { still_id: string; frame: number; zoom: number }[] }; lipsync?: boolean; scene_problems?: string[]; clips?: Clip[];
  locked?: Locked; plan?: { frames: number; context_frames: number }; sections?: string[]; matches_location_still?: boolean;
};
type Refs = { face?: string; body?: string; shared?: string };
type Place = { id: string; name?: string; description?: string; still_prompt?: string; still_id?: string };
type PlaceRow = Place & { still_present?: boolean };
type Script = { title: string; audio_id: string; duration: number; style?: string; concept?: string; look?: string; wardrobe?: string; arc?: string; refs?: Refs; ref_stills?: { face?: string; body?: string }; locations?: Place[]; scenes: Scene[] };
type ScriptRow = { name: string; title: string; audio_id: string; mtime: number; duration: number; scenes: number; clips: number; stills: number; locks: number; problems: number; videos?: number; last_video?: string | null };
type Check = { floor_r: number | null; floor_lag: number | null; floor_ok: boolean; opening_jump: number | null; opening_flag: boolean; sheet_url: string; seconds: number | null; planned_seconds: number | null };
type Upscale = { id: string; status: string; progress: number; max: number; error?: string | null; media_url: string | null; engine?: string; scale?: number; fps?: number | null; fps_fixed?: boolean | null };
type Take = { id: string; created: number; status: string; progress: number; max: number; media_url: string | null; seconds: number | null; params: Record<string, any>; check: Check | null; from_script?: string | null; upscaled?: Upscale | null };
type Cut = { stamp: string; file: string; path: string; seconds: number | null; scenes: number | null; partial: boolean; script: string | null; jobs: Record<string, string>; intro: boolean; end_card: { secs: number; text: string } | null; library_id?: string | null; media_url?: string | null; mtime: number };
type Dry = { settings: Record<string, any>; flags: string[]; prompt_hash: string; prompts: string[]; clips: number; lane: string; ref_ids: string[]; first_frame_id?: string | null; end_still_id?: string | null; rail_frames?: number[] | null; sig: string };
type Target = { kind: "face" | "body" } | { kind: "scene"; idx: number } | { kind: "place"; id: string } | null;

const ATTENTION = "comfy kitchen attention";   // the proven chain setting (every SSBS fire script)
// FlashVSR+ Turbo 2x: 554 s for the 15 s scene 6 take, 2026-10-30 (the whole SSBS batch ran at
// about that rate); only a rough guide for the time shown before a fire
const UPSCALE_SECS_PER_SEC = 37;
const upscaleBusy = (u?: Upscale | null) => !!u && u.status !== "done" && u.status !== "error";
// how a scene's opening picture is used (backend render_scene): a scene WITHOUT the singer renders
// image-to-video and starts on it (else on its place's picture; a scene the writer marked as
// differing from the place's picture must have its own); a scene WITH her renders from her two
// pictures and the place's picture, and its opening picture is not sent
const envOnly = (sc: Scene) => sc.performance === "no_singing_environment";
const needsPicture = (sc: Scene) => envOnly(sc) && sc.matches_location_still === false && !sc.opening_still_id;
const pictureUse = (sc: Scene) => envOnly(sc)
  ? (sc.matches_location_still === false ? "no singer: the render starts on this picture (required)" : "no singer: the render starts on this picture (else on the place's picture)")
  : "with the singer: she renders from her two pictures and the place's picture; this one is not used";

function fmt(t: number) {
  const m = Math.floor(t / 60), s = t - m * 60;
  return `${m}:${s.toFixed(2).padStart(5, "0")}`;
}
function mmss(t: number) {
  const m = Math.floor(t / 60), s = Math.round(t - m * 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
function when(ts: number) {
  const d = new Date(ts * 1000);
  return `${MONTHS[d.getMonth()]} ${d.getDate()} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}
function stampWhen(stamp: string) {   // 20261030_184008 -> Oct 30 18:40
  const m = /^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})/.exec(stamp);
  return m ? `${MONTHS[Number(m[2]) - 1]} ${Number(m[3])} ${m[4]}:${m[5]}` : stamp;
}
function audioLabel(it: LibItem): string {
  const p = it.params || {};
  return ((p.title || p.tags || p.source || it.mode || it.id).toString().slice(0, 70)) + (p.take ? ` · v${p.take}` : "");
}
const pct = (p: number, m: number) => (m ? `${Math.round((100 * p) / m)}%` : "");

function Pill({ tone, children, title }: { tone: "ok" | "warn" | "bad" | "muted" | "accent"; children: ReactNode; title?: string }) {
  const cls = tone === "ok" ? "text-emerald-400 border-emerald-800"
    : tone === "warn" ? "text-amber-400 border-amber-700"
    : tone === "bad" ? "text-red-400 border-red-800"
    : tone === "accent" ? "text-[var(--color-accent)] border-[var(--color-accent)]"
    : "text-[var(--color-muted)] border-[var(--color-line)]";
  return <span title={title} className={`rounded-full border px-2 py-0.5 text-[11px] whitespace-nowrap ${cls}`}>{children}</span>;
}

// an inline primary action (the shared PrimaryButton is a full-width form submit)
function Action({ onClick, disabled, children, title, big }: { onClick: () => void; disabled?: boolean; children: ReactNode; title?: string; big?: boolean }) {
  return (
    <button onClick={onClick} disabled={disabled} title={title}
      className={`rounded-lg bg-gradient-to-r from-[var(--color-accent)] to-[var(--color-accent2)] font-semibold text-white shadow transition hover:brightness-110 disabled:opacity-40 ${big ? "px-6 py-3 text-sm" : "px-3 py-2 text-xs"}`}>
      {children}
    </button>
  );
}

function Bar({ label, frac }: { label: ReactNode; frac: number }) {
  const w = Math.max(0, Math.min(1, frac)) * 100;
  return (
    <div>
      <div className="mb-0.5 flex justify-between text-[11px] text-[var(--color-muted)]"><span>{label}</span><span>{Math.round(w)}%</span></div>
      <div className="h-2 overflow-hidden rounded bg-[var(--color-line)]"><div className="h-full rounded bg-[var(--color-accent)] transition-all" style={{ width: `${w}%` }} /></div>
    </div>
  );
}

function Card({ title, sub, children, className = "" }: { title?: ReactNode; sub?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <div className={`rounded-xl border border-[var(--color-line)] bg-[var(--color-panel)] p-4 space-y-3 ${className}`}>
      {title && <div><div className="text-sm font-semibold">{title}</div>{sub && <div className="mt-0.5 text-xs text-[var(--color-muted)]">{sub}</div>}</div>}
      {children}
    </div>
  );
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

// A place for one picture: shows it, takes a dropped file, a chosen file, or Cmd+V once armed
// (clicked). The tab files pastes into the armed slot only, so a paste never lands by surprise.
function ImageSlot({ src, empty, armed, onArm, onFile, aspect = "aspect-video", children }: {
  src?: string | null; empty: ReactNode; armed: boolean; onArm: () => void; onFile: (f: File) => void; aspect?: string; children?: ReactNode;
}) {
  const ref = useRef<HTMLInputElement | null>(null);
  const [over, setOver] = useState(false);
  return (
    <div className="space-y-2">
      <div onClick={onArm}
           onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
           onDrop={(e) => { e.preventDefault(); setOver(false); const f = Array.from(e.dataTransfer.files).find((x) => x.type.startsWith("image/")); if (f) onFile(f); }}
           className={`relative ${aspect} w-full cursor-pointer overflow-hidden rounded-lg bg-black ${over ? "ring-2 ring-[var(--color-accent2)]" : armed ? "ring-2 ring-[var(--color-accent)]" : ""}`}>
        {src
          ? <img src={src} alt="" className="h-full w-full object-cover" loading="lazy" />
          : <div className={`flex h-full w-full flex-col items-center justify-center gap-1 border-2 border-dashed px-3 text-center text-xs text-[var(--color-muted)] ${over ? "border-[var(--color-accent2)]" : "border-[var(--color-line)]"}`}>{empty}</div>}
        {armed && <div className="absolute bottom-1 left-1 rounded bg-black/80 px-1.5 py-0.5 text-[10px] text-white">Cmd+V pastes here</div>}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <GhostButton onClick={() => ref.current?.click()}>{src ? "replace" : "choose file"}</GhostButton>
        {src && <GhostButton onClick={() => openLightbox(src)}>view</GhostButton>}
        {children}
        <input ref={ref} type="file" accept="image/*" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); e.target.value = ""; }} />
      </div>
    </div>
  );
}

const STEPS = ["Song", "Script", "Pictures", "Scenes and takes", "Final video", "YouTube text"] as const;

export function MusicVideoForm({ library, onDone }: { cfg: Config; busy: boolean; library: LibItem[] } & RunCtx) {
  const d = useDrafts("mv");
  const [step, setStep] = d.use("step", 1);
  const [audioId, setAudioId] = d.use("audioId", "");
  const [scriptName, setScriptName] = d.use("scriptName", "");
  const [sel, setSel] = d.use<number | null>("sel", null);
  // render settings: the proven recipe as defaults (turbo 1344x768, two reference stills)
  const [faceId, setFaceId] = d.use("faceId", "2c1c49e7c1a34a47878c4d416d6fb009");
  const [bodyId, setBodyId] = d.use("bodyId", "cd98a0e41c1a4579944e5aff23239e25");
  // blank = every check draws a new seed (pinned for the render it gates); the old "seed" draft
  // defaulted to 20261027, so every take of a scene repeated one seed
  const [seed, setSeed] = d.use("seedPin", "");
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
  const [bumpTick, setBumpTick] = useState(0);
  const [target, setTarget] = useState<Target>(null);
  const [placeRows, setPlaceRows] = useState<PlaceRow[]>([]);

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
  const reloadScript = () => setBumpTick((n) => n + 1);
  // the singer's pictures: the open script's own (each song keeps its look), else the shared setting
  const face = script?.ref_stills?.face || faceId;
  const body = script?.ref_stills?.body || bodyId;
  const setRef = async (kind: "face" | "body", id: string) => {
    if (!scriptName) { (kind === "face" ? setFaceId : setBodyId)(id); return; }
    setErr("");
    try { await api.mvScriptRefStills({ script: scriptName, [`${kind}_id`]: id }); reloadScript(); }
    catch (e) { setErr(String((e as Error).message || e)); }
  };
  const loadPlaces = useCallback(async () => {
    try { setPlaceRows(((await api.mvLocations()) as { locations: PlaceRow[] }).locations || []); } catch { /* keep the last list */ }
  }, []);
  useEffect(() => { loadPlaces(); }, [loadPlaces, scriptName]);

  // poll while any take or upscale is still running
  const rendering = useMemo(() => Object.values(takes).some((l) => l.some((t) => t.status !== "done" || upscaleBusy(t.upscaled))), [takes]);
  useEffect(() => {
    if (!rendering) return;
    const h = window.setInterval(refreshTakes, 5000);
    return () => window.clearInterval(h);
  }, [rendering, refreshTakes]);

  const sceneTakes = (idx: number) => takes[String(idx)] || [];
  const doneTakes = (idx: number) => sceneTakes(idx).filter((t) => t.status === "done");

  // ---------- filing pictures ----------
  const fileStill = async (idx: number, file: File) => {
    setErr(""); setBusyMsg(`filing the picture for scene ${idx}...`);
    try {
      const fd = new FormData();
      fd.append("file", file, file.name || "still.png");
      fd.append("script", scriptName);
      fd.append("scene", String(idx));
      await api.mvSceneStill(fd);
      reloadScript();
    } catch (e) { setErr(String((e as Error).message || e)); }
    finally { setBusyMsg(""); }
  };
  const fileRef = async (kind: "face" | "body", file: File) => {
    setErr(""); setBusyMsg(`filing the ${kind} reference...`);
    try {
      const fd = new FormData();
      fd.append("file", file, file.name || `${kind}.png`);
      fd.append("label", `${script?.title || "music video"} - lead ${kind} reference`);
      const r = await api.ytImportStill(fd) as { still_id: string };
      await setRef(kind, r.still_id);
    } catch (e) { setErr(String((e as Error).message || e)); }
    finally { setBusyMsg(""); }
  };
  const filePlace = async (id: string, file: File) => {
    setErr(""); setBusyMsg("filing the place picture...");
    try {
      const fd = new FormData();
      fd.append("file", file, file.name || `${id}.png`);
      fd.append("script", scriptName);
      fd.append("location_id", id);
      await api.mvLocationImportStill(fd);
      await loadPlaces();
    } catch (e) { setErr(String((e as Error).message || e)); }
    finally { setBusyMsg(""); }
  };
  const linkPlace = async (id: string, stillId: string) => {
    setErr("");
    try { await api.mvLocationImport({ script: scriptName, location_id: id, still_id: stillId }); await loadPlaces(); }
    catch (e) { setErr(String((e as Error).message || e)); }
  };
  const onPaste = (e: React.ClipboardEvent) => {
    if (!target || (e.target as HTMLElement).closest("input,textarea")) return;
    const item = Array.from(e.clipboardData.items).find((it) => it.type.startsWith("image/"));
    const f = item?.getAsFile();
    if (!f) return;
    e.preventDefault();
    if (target.kind === "scene") fileStill(target.idx, f);
    else if (target.kind === "place") filePlace(target.id, f);
    else fileRef(target.kind, f);
  };

  // ---------- what is left ----------
  const scenes = script?.scenes || [];
  const nStills = scenes.filter((s) => s.opening_still_id).length;
  // a place's picture, the way render_scene finds it: the script's own still_id, else the place library's
  const placePic = (pl: Place) => pl.still_id || placeRows.find((r) => r.id === pl.id && r.still_present)?.still_id || null;
  const places = (script?.locations || []).filter((pl) => scenes.some((sc) => sc.location === pl.id));
  const placesBare = places.filter((pl) => !placePic(pl));
  const nChosen = scenes.filter((s) => s.locked).length;
  const nRunning = scenes.filter((s) => sceneTakes(s.index).some((t) => t.status !== "done")).length;
  const notRendered = scenes.filter((s) => !s.locked && !doneTakes(s.index).length && !sceneTakes(s.index).some((t) => t.status !== "done"));
  const next: { step: number; text: string } = !audioId ? { step: 1, text: "Pick the song this video is for." }
    : !script ? { step: 2, text: scripts.length ? "Open one of this song's scripts, or write a new one." : "Write the script: Claude plans every scene of the video from the song." }
    : !face.trim() || !body.trim() ? { step: 3, text: "Add the singer's two reference pictures (face and full body): every scene is rendered with them." }
    : placesBare.length && notRendered.length ? { step: 3, text: `${placesBare.length} place${placesBare.length === 1 ? " has" : "s have"} no picture yet (${placesBare.map((pl) => pl.name || pl.id).join(", ")}). Every scene in a place renders against its picture.` }
    : scenes.some(needsPicture) ? { step: 3, text: `${scenes.filter(needsPicture).length} scene${scenes.filter(needsPicture).length === 1 ? " needs its" : "s need their"} own opening picture before rendering. Copy the scene's prompt, make the image in ChatGPT, drop it on the scene's card.` }
    : notRendered.length ? { step: 4, text: `${notRendered.length} scene${notRendered.length === 1 ? " is" : "s are"} not rendered yet.` }
    : nChosen < scenes.length ? { step: 4, text: `Choose the take that plays in ${scenes.length - nChosen} scene${scenes.length - nChosen === 1 ? "" : "s"} ("Use in the video"), or copy the choices of an earlier video in step 5.` }
    : { step: 5, text: "Every scene has its take. Make the video in step 5." };

  const stepNote = (i: number): string => {
    if (i === 1) return audioId ? (audios.find((a) => a.id === audioId) ? audioLabel(audios.find((a) => a.id === audioId)!).slice(0, 34) : "picked") : "pick one";
    if (i === 2) return script ? `${scenes.length} scenes` : "none open";
    if (i === 3) return script ? `${places.length - placesBare.length} of ${places.length} places · ${nStills} of ${scenes.length} scenes` : "";
    if (i === 4) return script ? `${nChosen} of ${scenes.length} chosen${nRunning ? ` · ${nRunning} rendering` : ""}` : "";
    if (i === 5) return script ? "make the video" : "";
    return "";
  };

  return (
    <div className="space-y-4" onPaste={onPaste}>
      {/* ---------- the steps ---------- */}
      <div className="grid grid-cols-2 gap-2 md:grid-cols-3 xl:grid-cols-6">
        {STEPS.map((label, i) => {
          const n = i + 1;
          const off = n > 2 && !script;
          return (
            <button key={label} onClick={() => !off && setStep(n)} disabled={off}
              className={`rounded-xl border px-3 py-2.5 text-left transition disabled:opacity-40 ${step === n ? "border-[var(--color-accent)] bg-[#2a1c19]" : "border-[var(--color-line)] bg-[var(--color-panel)] hover:border-[var(--color-muted)]"}`}>
              <div className="flex items-center gap-2 text-sm font-semibold">
                <span className={`flex h-5 w-5 items-center justify-center rounded-full text-[11px] ${step === n ? "bg-[var(--color-accent)] text-white" : "bg-[var(--color-panel2)] text-[var(--color-muted)]"}`}>{n}</span>
                {label}
                {next.step === n && <span className="ml-auto h-2 w-2 rounded-full bg-[var(--color-accent2)]" title="the next thing to do is here" />}
              </div>
              <div className="mt-0.5 truncate text-[11px] text-[var(--color-muted)]">{stepNote(n) || " "}</div>
            </button>
          );
        })}
      </div>
      <div className="flex flex-wrap items-center gap-3 rounded-xl border border-[var(--color-accent2)]/40 bg-[var(--color-panel2)] px-4 py-2.5 text-sm">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-[var(--color-accent2)]">Next</span>
        <span>{next.text}</span>
        {next.step !== step && <GhostButton onClick={() => setStep(next.step)} className="ml-auto">go to step {next.step}</GhostButton>}
      </div>
      {(busyMsg || err) && (
        <div className={`rounded-lg border p-2 text-xs ${err ? "border-red-800 text-red-300" : "border-[var(--color-line)] text-[var(--color-muted)]"}`}>
          {err || busyMsg}{err && <button className="ml-3 underline" onClick={() => setErr("")}>dismiss</button>}
        </div>
      )}

      {step === 1 && (
        <SongStep audioId={audioId} audios={audios}
          onPick={(id) => { setAudioId(id); setScriptName(""); setSel(null); setDry({}); }}
          onImported={(id) => { onDone(); setAudioId(id); setScriptName(""); setSel(null); }}
          onNext={() => setStep(2)} />
      )}

      {step === 2 && (
        <ScriptStep audioId={audioId} scripts={scripts} scriptName={scriptName} script={script} allScripts={allScripts} setAllScripts={setAllScripts}
          onOpen={(name) => { setScriptName(name); setSel(null); setDry({}); }}
          onWritten={async (name) => { await loadScripts(); setScriptName(name); setSel(null); setDry({}); }}
          onNext={() => setStep(3)} />
      )}

      {step === 3 && script && (
        <PicturesStep script={script} library={library} faceId={face} bodyId={body} setFaceId={(id) => setRef("face", id)} setBodyId={(id) => setRef("body", id)}
          target={target} setTarget={setTarget} fileStill={fileStill} fileRef={fileRef} setErr={setErr}
          places={places} placeRows={placeRows} placePic={placePic} filePlace={filePlace} linkPlace={linkPlace} />
      )}

      {step === 4 && script && (
        <TakesStep script={script} scriptName={scriptName} takes={takes} sel={sel} setSel={setSel}
          faceId={face} bodyId={body} seed={seed} setSeed={setSeed} turbo={turbo} setTurbo={setTurbo}
          width={width} setWidth={setWidth} height={height} setHeight={setHeight} acceptFlags={acceptFlags} setAcceptFlags={setAcceptFlags}
          dry={dry} setDry={setDry} busyMsg={busyMsg} setBusyMsg={setBusyMsg} setErr={setErr}
          refreshTakes={refreshTakes} reloadScript={reloadScript} target={target} setTarget={setTarget} fileStill={fileStill}
          goPictures={() => setStep(3)}
          startPicOf={(sc) => sc.opening_still_id || (envOnly(sc) ? (() => { const pl = (script.locations || []).find((x) => x.id === sc.location); return pl ? placePic(pl) : null; })() : null)} />
      )}

      {step === 5 && script && <FinalStep script={script} scriptName={scriptName} takes={takes} refreshTakes={refreshTakes} reloadScript={reloadScript} />}

      {step === 6 && script && <UploadText script={script} />}
    </div>
  );
}

// ---------- 1 Song: pick from the library or import from disk, its facts, the vocal stem ----------
type SongInfo = { audio_id: string; title: string; style: string; seconds: number | null; stem: boolean };

function SongStep({ audioId, audios, onPick, onImported, onNext }: { audioId: string; audios: LibItem[]; onPick: (id: string) => void; onImported: (id: string) => void; onNext: () => void }) {
  const [info, setInfo] = useState<SongInfo | null>(null);
  const [q, setQ] = useState("");
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
  const shown = useMemo(() => {
    const s = q.trim().toLowerCase();
    return (s ? audios.filter((a) => audioLabel(a).toLowerCase().includes(s)) : audios).slice(0, 200);
  }, [audios, q]);
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
    <div className="grid gap-4 xl:grid-cols-[2fr_1fr]">
      <Card title="Pick the song from the Library" sub="the finished track the video is cut to">
        <input className={inp} value={q} onChange={(e) => setQ(e.target.value)} placeholder="search the Library's tracks" />
        <div className="max-h-[50vh] overflow-auto rounded-lg border border-[var(--color-line)]">
          {shown.map((a) => (
            <button key={a.id} onClick={() => onPick(a.id)}
              className={`flex w-full items-center gap-3 border-b border-[var(--color-line)] px-3 py-2 text-left text-sm hover:bg-[var(--color-panel2)] ${a.id === audioId ? "bg-[#2a1c19] text-[var(--color-accent)]" : ""}`}>
              <span className="flex-1 truncate">{audioLabel(a)}</span>
              <span className="text-[11px] text-[var(--color-muted)]">{when(a.created)}</span>
            </button>
          ))}
          {!shown.length && <div className="p-3 text-xs text-[var(--color-muted)]">no track matches</div>}
        </div>
      </Card>
      <div className="space-y-4">
        {info && (
          <Card title={<>{info.title || "(untitled track)"} <span className="font-normal text-[var(--color-muted)]">{info.seconds ? mmss(info.seconds) : ""}</span></>} sub="the song picked">
            {info.style && <div className="text-xs text-[var(--color-muted)]">{info.style}</div>}
            <div className="flex flex-wrap items-center gap-2">
              {info.stem ? <Pill tone="ok" title="lyric timing is measured on the isolated vocals">vocal stem ready</Pill>
                : <><Pill tone="warn" title="the writer times the lyrics on the full mix without it">no vocal stem</Pill>
                    <GhostButton onClick={makeStem} disabled={!!busy}>make the vocal stem (box)</GhostButton></>}
            </div>
            <Action onClick={onNext}>Next: the script</Action>
          </Card>
        )}
        <Card title="Or import a song from disk" sub="wav, flac and mp3 are kept as they are; it lands in the Library as a source track">
          <input type="file" accept="audio/*,.wav,.flac,.mp3" className="text-xs" onChange={(e) => setFile(e.target.files?.[0] || null)} />
          <input className={inp} value={fileTitle} onChange={(e) => setFileTitle(e.target.value)} placeholder="title (defaults to the file name)" />
          <GhostButton onClick={importFile} disabled={!file || !!busy}>import and use it</GhostButton>
        </Card>
        {(busy || err) && <div className={`text-xs ${err ? "text-red-400" : "text-[var(--color-muted)]"}`}>{err || busy}</div>}
      </div>
    </div>
  );
}

// ---------- 2 Script: the song's scripts, open one or write a new one ----------
function ScriptStep({ audioId, scripts, scriptName, script, allScripts, setAllScripts, onOpen, onWritten, onNext }: {
  audioId: string; scripts: ScriptRow[]; scriptName: string; script: Script | null; allScripts: boolean; setAllScripts: (v: boolean) => void;
  onOpen: (name: string) => void; onWritten: (name: string) => Promise<void>; onNext: () => void;
}) {
  // revisions of one song share a title: the ones a video was cut from come first, newest video first
  const sorted = [...scripts].sort((a, b) => (b.last_video || "").localeCompare(a.last_video || "") || b.mtime - a.mtime);
  const latest = sorted[0]?.last_video ? sorted[0].name : null;
  return (
    <div className="grid items-start gap-4 xl:grid-cols-[3fr_2fr]">
      <Card title="This song's scripts" sub="a script is the plan of the whole video: every scene, its picture prompt and its clip prompts. Open the one you are working on.">
        <label className="flex items-center gap-2 text-xs text-[var(--color-muted)]">
          <input type="checkbox" checked={allScripts} onChange={(e) => setAllScripts(e.target.checked)} /> show the scripts of every song
        </label>
        {sorted.length === 0 ? <div className="text-sm text-[var(--color-muted)]">No scripts{audioId && !allScripts ? " for this song" : ""} yet: write one on the right.</div> : (
          <div className="max-h-[60vh] space-y-2 overflow-auto">
            {sorted.map((s) => (
              <div key={s.name} className={`flex flex-wrap items-center gap-3 rounded-lg border px-3 py-2 ${s.name === scriptName ? "border-[var(--color-accent)] bg-[#2a1c19]" : "border-[var(--color-line)]"}`}>
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2 text-sm font-semibold" title={s.name}>{s.title || "(untitled)"}
                    {s.name === latest && <Pill tone="accent">your latest video ({stampWhen(s.last_video!)}) was made from this</Pill>}
                    {s.name !== latest && !!s.videos && <Pill tone="muted">{s.videos} video{s.videos === 1 ? "" : "s"}, last {stampWhen(s.last_video!)}</Pill>}
                  </div>
                  <div className="text-[11px] text-[var(--color-muted)]">saved {when(s.mtime)} · {s.scenes} scenes · {s.stills} pictures · {s.locks} takes chosen{s.problems ? ` · ${s.problems} problems` : ""}</div>
                </div>
                {s.name === scriptName ? <Pill tone="accent">open</Pill> : <GhostButton onClick={() => onOpen(s.name)}>open</GhostButton>}
              </div>
            ))}
          </div>
        )}
        {script && <Action onClick={onNext}>Next: the pictures</Action>}
      </Card>
      {audioId && <WriterPanel audioId={audioId} startOpen={scripts.length === 0}
        loaded={script && script.audio_id === audioId ? { name: scriptName, script } : null} onWritten={onWritten} />}
    </div>
  );
}

// ---------- 3 Pictures: the singer's references and one opening picture per scene ----------
function StillPicker({ library, onPick, onClose, places }: { library: LibItem[]; onPick: (id: string) => void; onClose: () => void; places?: PlaceRow[] }) {
  const [q, setQ] = useState("");
  const stills = useMemo(() => {
    const s = q.trim().toLowerCase();
    return library.filter((i) => i.mode === "videostill")
      .filter((i) => !s || JSON.stringify(i.params || {}).toLowerCase().includes(s))
      .sort((a, b) => b.created - a.created).slice(0, 60);
  }, [library, q]);
  return (
    <div className="space-y-2 rounded-lg border border-[var(--color-accent)] p-3">
      <div className="flex items-center gap-2">
        <input className={inp} value={q} onChange={(e) => setQ(e.target.value)} placeholder="search the Library's pictures" />
        <GhostButton onClick={onClose}>close</GhostButton>
      </div>
      {places && places.length > 0 && (
        <>
          <div className="text-[11px] text-[var(--color-muted)]">pictures of places from earlier videos</div>
          <div className="grid max-h-56 grid-cols-3 gap-2 overflow-auto md:grid-cols-5">
            {places.map((r) => (
              <button key={r.id} onClick={() => onPick(r.still_id!)} title={r.description || ""} className="text-left">
                <img src={`/api/media/${r.still_id}`} alt="" className="aspect-video w-full rounded object-cover hover:ring-2 hover:ring-[var(--color-accent)]" loading="lazy" />
                <div className="truncate text-[10px] text-[var(--color-muted)]">{r.name || r.id}</div>
              </button>
            ))}
          </div>
          <div className="text-[11px] text-[var(--color-muted)]">every picture in the Library</div>
        </>
      )}
      <div className="grid max-h-80 grid-cols-4 gap-2 overflow-auto md:grid-cols-6">
        {stills.map((s) => (
          <button key={s.id} onClick={() => onPick(s.id)} title={String(s.params?.label || s.params?.prompt || "").slice(0, 200)}>
            <img src={s.media_url || `/api/media/${s.id}`} alt="" className="aspect-square w-full rounded object-cover hover:ring-2 hover:ring-[var(--color-accent)]" loading="lazy" />
          </button>
        ))}
      </div>
    </div>
  );
}

function PicturesStep({ script, library, faceId, bodyId, setFaceId, setBodyId, target, setTarget, fileStill, fileRef, setErr, places, placeRows, placePic, filePlace, linkPlace }: {
  script: Script; library: LibItem[]; faceId: string; bodyId: string; setFaceId: (v: string) => void; setBodyId: (v: string) => void;
  target: Target; setTarget: (t: Target) => void; fileStill: (idx: number, f: File) => void; fileRef: (k: "face" | "body", f: File) => void; setErr: (s: string) => void;
  places: Place[]; placeRows: PlaceRow[]; placePic: (pl: Place) => string | null; filePlace: (id: string, f: File) => void; linkPlace: (id: string, stillId: string) => void;
}) {
  const [picker, setPicker] = useState<"face" | "body" | null>(null);
  const [placePicker, setPlacePicker] = useState<string | null>(null);
  const [copiedPlace, setCopiedPlace] = useState<string | null>(null);
  const copyPlace = async (pl: Place) => {
    try { await navigator.clipboard.writeText(pl.still_prompt || ""); setCopiedPlace(pl.id); window.setTimeout(() => setCopiedPlace(null), 1500); }
    catch { setErr("the browser blocked the clipboard: open the prompt and copy it by hand"); }
  };
  const otherPlaces = placeRows.filter((r) => r.still_present && r.still_id && !places.some((pl) => pl.id === r.id));
  const [onlyMissing, setOnlyMissing] = useState(false);
  const [copied, setCopied] = useState<number | null>(null);
  const copy = async (sc: Scene) => {
    try { await navigator.clipboard.writeText(sc.opening_still || ""); setCopied(sc.index); window.setTimeout(() => setCopied(null), 1500); }
    catch { setErr("the browser blocked the clipboard: open the prompt and copy it by hand"); }
  };
  const isT = (t: Target) => JSON.stringify(t) === JSON.stringify(target);
  const shown = script.scenes.filter((s) => !onlyMissing || !s.opening_still_id);
  const missing = script.scenes.filter((s) => !s.opening_still_id).length;
  return (
    <div className="space-y-4">
      <Card title="The singer: two reference pictures" sub="used for EVERY scene so she looks the same throughout. Click a box and paste (Cmd+V), drop an image on it, choose a file, or pick one already in the Library.">
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {(["face", "body"] as const).map((k) => {
            const id = k === "face" ? faceId : bodyId;
            return (
              <div key={k} className="space-y-1">
                <div className="text-xs font-semibold">{k === "face" ? "Face (close-up portrait)" : "Full body (head to feet)"}</div>
                <ImageSlot src={id.trim() ? `/api/media/${id.trim()}` : null} aspect="aspect-[3/4]" armed={isT({ kind: k })} onArm={() => setTarget({ kind: k })}
                  onFile={(f) => fileRef(k, f)} empty={<><div>no {k} picture</div><div>click, then Cmd+V, or drop an image</div></>}>
                  <GhostButton onClick={() => setPicker(picker === k ? null : k)}>from the Library</GhostButton>
                </ImageSlot>
              </div>
            );
          })}
          <div className="text-xs text-[var(--color-muted)] xl:col-span-2 space-y-2">
            {script.refs?.face && <div><b className="text-[var(--color-ink)]">The script describes the face picture as:</b> {script.refs.face}</div>}
            {script.refs?.body && <div><b className="text-[var(--color-ink)]">and the full-body picture as:</b> {script.refs.body}</div>}
          </div>
        </div>
        {picker && <StillPicker library={library} onClose={() => setPicker(null)} onPick={(id) => { (picker === "face" ? setFaceId : setBodyId)(id); setPicker(null); }} />}
      </Card>

      {places.length > 0 && (
        <Card title={`The places (${places.filter((pl) => placePic(pl)).length} of ${places.length})`}
          sub="one picture per place, with NO people in it: every scene set there renders against it (scenes without the singer start on it). Copy the prompt, make the image in ChatGPT, then click the box and paste, or drop it. A place from an earlier video can be reused from the Library.">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-4">
            {places.map((pl) => {
              const pic = placePic(pl);
              const n = script.scenes.filter((sc) => sc.location === pl.id).length;
              return (
                <div key={pl.id} className="space-y-1.5">
                  <div className="flex items-baseline gap-2 text-xs">
                    <span className="font-semibold">{pl.name || pl.id}</span>
                    <span className="ml-auto text-[var(--color-muted)]">{n} scene{n === 1 ? "" : "s"}</span>
                  </div>
                  <ImageSlot src={pic ? `/api/media/${pic}` : null} armed={isT({ kind: "place", id: pl.id })}
                    onArm={() => setTarget({ kind: "place", id: pl.id })} onFile={(f) => filePlace(pl.id, f)}
                    empty={<><div className="font-semibold">no picture of this place yet</div><div>1. copy the prompt  2. make it in ChatGPT  3. click here + Cmd+V, or drop it</div></>}>
                    <GhostButton onClick={() => copyPlace(pl)} disabled={!pl.still_prompt}>{copiedPlace === pl.id ? "copied" : "copy prompt"}</GhostButton>
                    <GhostButton onClick={() => setPlacePicker(placePicker === pl.id ? null : pl.id)}>from the Library</GhostButton>
                  </ImageSlot>
                  {pl.still_prompt && (
                    <details className="text-[11px]">
                      <summary className="cursor-pointer text-[var(--color-muted)]">show the prompt</summary>
                      <div className="mt-1 whitespace-pre-wrap rounded border border-[var(--color-line)] p-2">{pl.still_prompt}</div>
                    </details>
                  )}
                </div>
              );
            })}
          </div>
          {placePicker && (
            <div className="space-y-1">
              <div className="text-xs font-semibold">Pick the picture for {places.find((pl) => pl.id === placePicker)?.name || placePicker}</div>
              <StillPicker library={library} places={otherPlaces} onClose={() => setPlacePicker(null)}
                onPick={(id) => { linkPlace(placePicker, id); setPlacePicker(null); }} />
            </div>
          )}
        </Card>
      )}

      <Card title={`Opening picture for each scene (${script.scenes.length - missing} of ${script.scenes.length})`}
        sub="scenes without the singer start on their picture; scenes with her render from her two pictures above. To add one: copy the prompt, make the image in ChatGPT, then click the scene's box and paste, or drop the image on it.">
        <label className="flex items-center gap-2 text-xs text-[var(--color-muted)]"><input type="checkbox" checked={onlyMissing} onChange={(e) => setOnlyMissing(e.target.checked)} /> only scenes without a picture ({missing})</label>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-4">
          {shown.map((sc) => (
            <div key={sc.index} className="space-y-1.5">
              <div className="flex items-baseline gap-2 text-xs">
                <span className="font-semibold">{String(sc.index).padStart(2, "0")} {sc.title}</span>
                <span className="ml-auto text-[var(--color-muted)]">{mmss(sc.start)}</span>
              </div>
              <div className={`text-[11px] ${needsPicture(sc) ? "text-amber-400" : "text-[var(--color-muted)]"}`}>{pictureUse(sc)}</div>
              <ImageSlot src={sc.opening_still_id ? `/api/media/${sc.opening_still_id}` : null} armed={isT({ kind: "scene", idx: sc.index })}
                onArm={() => setTarget({ kind: "scene", idx: sc.index })} onFile={(f) => fileStill(sc.index, f)}
                empty={<><div className="font-semibold">no picture yet</div><div>1. copy the prompt  2. make it in ChatGPT  3. click here + Cmd+V, or drop it</div></>}>
                <GhostButton onClick={() => copy(sc)} disabled={!sc.opening_still}>{copied === sc.index ? "copied" : "copy prompt"}</GhostButton>
              </ImageSlot>
              {sc.opening_still && (
                <details className="text-[11px]">
                  <summary className="cursor-pointer text-[var(--color-muted)]">show the prompt</summary>
                  <div className="mt-1 whitespace-pre-wrap rounded border border-[var(--color-line)] p-2">{sc.opening_still}</div>
                </details>
              )}
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

// ---------- the end of a camera move: a second view of the same picture the take must arrive at ----------
function EndOfMove({ scene, startPic, scriptName, width, height, busy, setErr, reload }: {
  scene: Scene; startPic: string | null; scriptName: string; width: number; height: number; busy: boolean;
  setErr: (s: string) => void; reload: () => void;
}) {
  const [zoomText, setZoomText] = useState("1.5");      // typed freely, clamped where it is used
  const zoom = Math.min(4, Math.max(1.05, Number(zoomText) || 1.5));
  const [c, setC] = useState({ x: 0.5, y: 0.5 });
  const [railOn, setRailOn] = useState(false);          // opt-in: pin the in-between views too
  const [working, setWorking] = useState(false);
  const rail = scene.rail && scene.rail.end_still_id === scene.end_still_id ? scene.rail.stills : [];
  useEffect(() => { setC({ x: 0.5, y: 0.5 }); }, [scene.index]);
  if (!startPic && !scene.end_still_id) return null;
  // the crop box as the backend clamps it, in fractions of the picture
  const fw = 1 / zoom, fh = 1 / zoom;
  const x0 = Math.min(Math.max(c.x - fw / 2, 0), 1 - fw), y0 = Math.min(Math.max(c.y - fh / 2, 0), 1 - fh);
  const make = async () => {
    setErr(""); setWorking(true);
    try { await api.mvPushEndStill({ script: scriptName, scene: scene.index, zoom, cx: x0 + fw / 2, cy: y0 + fh / 2, width, height, rail: railOn ? 1.5 : 0 }); reload(); }
    catch (e) { setErr(String((e as Error).message || e)); }
    finally { setWorking(false); }
  };
  const clear = async () => {
    setErr("");
    try { await api.mvSceneEndStill({ script: scriptName, scene: scene.index, still_id: null }); reload(); }
    catch (e) { setErr(String((e as Error).message || e)); }
  };
  return (
    <Card title="Where the move ends" sub="optional: the take starts on the first picture and must arrive at this one, so a push cannot run past it or lose the room. For a push-in: click the first picture where the push heads, set how far, make the end picture.">
      <div className="grid gap-3 md:grid-cols-2">
        {startPic ? (
          <div>
            <div className="mb-1 text-[11px] text-[var(--color-muted)]">first picture: click where the push heads</div>
            <div className="relative cursor-crosshair overflow-hidden rounded bg-black" onClick={(e) => {
              const r = (e.currentTarget as HTMLDivElement).getBoundingClientRect();
              setC({ x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height });
            }}>
              <img src={`/api/media/${startPic}`} alt="" className="block w-full" />
              <div className="pointer-events-none absolute border-2 border-[var(--color-accent)]"
                style={{ left: `${x0 * 100}%`, top: `${y0 * 100}%`, width: `${fw * 100}%`, height: `${fh * 100}%` }} />
            </div>
            <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
              <label className="flex items-center gap-1">push to <input type="number" min={1.05} max={4} step={0.05} value={zoomText} className={`${inp} w-20`}
                onChange={(e) => setZoomText(e.target.value)} />x closer</label>
              <label className="flex items-center gap-1" title="also pin a crop every 1.5 s along the push, so the camera's path is held all the way and not only at its two ends">
                <input type="checkbox" checked={railOn} onChange={(e) => setRailOn(e.target.checked)} />pin the path every 1.5 s</label>
              <GhostButton onClick={make} disabled={busy || working}>{working ? "making..." : scene.end_still_id ? "make it again" : "make the end picture"}</GhostButton>
            </div>
          </div>
        ) : <div className="text-xs text-[var(--color-muted)]">this scene renders from references, so it has no first picture to crop; an end picture made elsewhere still applies</div>}
        <div>
          <div className="mb-1 text-[11px] text-[var(--color-muted)]">end picture{scene.end_still_id ? "" : ": none, the move is free"}</div>
          {scene.end_still_id ? (
            <>
              <img src={`/api/media/${scene.end_still_id}`} alt="" className="block w-full rounded" />
              {rail.length ? (
                <div className="mt-2">
                  <div className="mb-1 text-[11px] text-[var(--color-muted)]">pinned on the way: {rail.length} pictures</div>
                  <div className="grid grid-cols-4 gap-1">
                    {rail.map((r) => (
                      <div key={r.still_id} className="text-[10px] text-[var(--color-muted)]">
                        <img src={`/api/media/${r.still_id}`} alt="" className="block w-full rounded" />
                        {(r.frame / 24).toFixed(1)} s, x{r.zoom.toFixed(2)}
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}
              <div className="mt-2"><GhostButton onClick={clear} disabled={busy}>remove the end picture</GhostButton></div>
            </>
          ) : <div className="flex aspect-video items-center justify-center rounded border border-dashed border-[var(--color-line)] text-xs text-[var(--color-muted)]">no end picture</div>}
        </div>
      </div>
    </Card>
  );
}

// ---------- 4 Scenes and takes: render each scene, choose the take that plays ----------
function TakesStep(p: {
  script: Script; scriptName: string; takes: Record<string, Take[]>; sel: number | null; setSel: (n: number | null) => void;
  faceId: string; bodyId: string; seed: string; setSeed: (v: string) => void; turbo: boolean; setTurbo: (v: boolean) => void;
  width: number; setWidth: (v: number) => void; height: number; setHeight: (v: number) => void; acceptFlags: boolean; setAcceptFlags: (v: boolean) => void;
  dry: Record<number, Dry>; setDry: (f: (d: Record<number, Dry>) => Record<number, Dry>) => void; busyMsg: string; setBusyMsg: (s: string) => void; setErr: (s: string) => void;
  refreshTakes: () => Promise<void>; reloadScript: () => void; target: Target; setTarget: (t: Target) => void; fileStill: (idx: number, f: File) => void; goPictures: () => void;
  startPicOf: (sc: Scene) => string | null;
}) {
  const { script, scriptName, takes, dry, busyMsg, setBusyMsg, setErr } = p;
  const [filter, setFilter] = useState<"all" | "todo">("all");
  const scene = script.scenes[p.sel ?? 0] || script.scenes[0];
  const list = (idx: number) => takes[String(idx)] || [];

  const sig = () => `${p.faceId}|${p.bodyId}|${p.seed}|${p.turbo}|${p.width}|${p.height}`;
  const renderBody = (idx: number, dryRun: boolean, pinSeed?: number) => {
    const body: Record<string, unknown> = {
      script: scriptName, scene: idx, ref_ids: [p.faceId, p.bodyId].map((x) => x.trim()).filter(Boolean),
      turbo: p.turbo, attention: ATTENTION, width: Number(p.width), height: Number(p.height), dry_run: dryRun,
    };
    const s = pinSeed ?? (p.seed.trim() ? Number(p.seed) : undefined);
    if (s !== undefined && !Number.isNaN(s)) body.seed = s;
    return body;
  };
  const doDry = async (idx: number) => {
    setErr(""); setBusyMsg("checking the render settings (no GPU)...");
    try {
      const r = await api.mvRenderScene(renderBody(idx, true));
      p.setDry((d) => ({ ...d, [idx]: { ...(r as Dry), sig: sig() } }));
    } catch (e) { setErr(String((e as Error).message || e)); }
    finally { setBusyMsg(""); }
  };
  // a seed that already made a take of this scene would most likely make the same take again
  // a take with this seed AND the same end picture and rail would come out the same; the same seed
  // with a different end picture or rail is a controlled comparison, not a repeat
  const seedUsed = (idx: number, sd: unknown, dr?: Dry) => sd != null && list(idx).some((t) => String(t.params?.seed) === String(sd)
    && (t.params?.end_still_id || null) === (dr?.end_still_id || null)
    && JSON.stringify(t.params?.rail_frames || null) === JSON.stringify(dr?.rail_frames || null));
  const canRender = (idx: number) => {
    const dr = dry[idx];
    const endNow = script.scenes.find((x) => x.index === idx)?.end_still_id || null;
    return !!dr && dr.sig === sig() && (dr.end_still_id || null) === endNow && (dr.flags.length === 0 || p.acceptFlags) && !seedUsed(idx, dr.settings?.seed, dr);
  };
  const doRender = async (idx: number) => {
    const dr = dry[idx];
    if (!dr || !canRender(idx)) return;
    if (!window.confirm(`Render scene ${idx} on the box now? (${dr.clips} clip${dr.clips === 1 ? "" : "s"}, ${dr.settings?.width}x${dr.settings?.height}, seed ${dr.settings?.seed})`)) return;
    setErr(""); setBusyMsg("sending the scene to the box (30-60 s)...");
    try {
      await api.mvRenderScene(renderBody(idx, false, Number(dr.settings?.seed)));
      p.setDry((d) => { const n = { ...d }; delete n[idx]; return n; });     // the next render checks again: a new seed
      await p.refreshTakes();
    }
    catch (e) { setErr(String((e as Error).message || e)); }
    finally { setBusyMsg(""); }
  };
  const doCheck = async (idx: number, jid: string) => {
    setErr(""); setBusyMsg("measuring the take (Mac, no GPU)...");
    try { await api.mvTakeCheck({ job_id: jid, script: scriptName, scene: idx }); await p.refreshTakes(); }
    catch (e) { setErr(String((e as Error).message || e)); }
    finally { setBusyMsg(""); }
  };
  const choose = async (idx: number, jid: string | null) => {
    setErr("");
    try { await api.mvLockTake({ script: scriptName, scene: idx, job_id: jid }); p.reloadScript(); }
    catch (e) { setErr(String((e as Error).message || e)); }
  };
  const doUpscale = async (idx: number, t: Take) => {
    const mins = Math.max(1, Math.round(((t.seconds || 6) * UPSCALE_SECS_PER_SEC) / 60));
    if (!window.confirm(`Upscale this take 2x with FlashVSR+ on the box now? (roughly ${mins} min)`)) return;
    setErr(""); setBusyMsg("sending the take to the box...");
    try {
      const r = await api.mvUpscaleTakes({ jobs: { [String(idx)]: t.id }, script: scriptName, force: t.upscaled?.status === "error" || t.upscaled?.status === "done" });
      if (!r.queued?.length && r.skipped?.length) setErr(`not queued: ${r.skipped[0].why}`);
      await p.refreshTakes();
    } catch (e) { setErr(String((e as Error).message || e)); }
    finally { setBusyMsg(""); }
  };

  const stateOf = (sc: Scene) => {
    const l = list(sc.index);
    const run = l.find((t) => t.status !== "done");
    const n = l.filter((t) => t.status === "done").length;
    if (sc.locked) return { tone: "ok" as const, text: "take chosen", todo: false };
    if (run) return { tone: "accent" as const, text: `rendering ${pct(run.progress, run.max)}`, todo: false };
    if (n) return { tone: "warn" as const, text: `${n} take${n === 1 ? "" : "s"}, none chosen`, todo: true };
    return { tone: "muted" as const, text: "not rendered", todo: true };
  };
  // the batch: every scene with no take yet and none chosen, checked together then sent in the background
  type BatchRow = { index: number; title?: string; status: string; error?: unknown; flags?: string[]; job_id?: string | null };
  type BatchDry = { manifest: string; counts: Record<string, number>; flagged: number[]; scenes: BatchRow[]; sig: string; only: number[] };
  type BatchJob = { id: string; status: string; error?: string; counts?: Record<string, number>; rows: BatchRow[]; total?: number };
  const [bDry, setBDry] = useState<BatchDry | null>(null);
  const [bAccept, setBAccept] = useState(false);
  const [bJob, setBJob] = useState<BatchJob | null>(null);
  const unrendered = script.scenes.filter((sc) => !sc.locked && list(sc.index).length === 0).map((sc) => sc.index);
  const bSig = () => `${sig()}|${unrendered.join(",")}`;
  const batchCommon = () => {
    const b: Record<string, unknown> = {
      script: scriptName, ref_ids: [p.faceId, p.bodyId].map((x) => x.trim()).filter(Boolean),
      turbo: p.turbo, attention: ATTENTION, width: Number(p.width), height: Number(p.height),
    };
    const sd = p.seed.trim() ? Number(p.seed) : undefined;
    if (sd !== undefined && !Number.isNaN(sd)) b.seed = sd;
    return b;
  };
  const doBatchDry = async () => {
    setErr(""); setBusyMsg(`checking the settings of ${unrendered.length} scenes (no GPU)...`);
    try {
      const r = await api.mvRenderScript({ ...batchCommon(), only: unrendered, dry_run: true });
      setBDry({ ...(r as Omit<BatchDry, "sig" | "only">), sig: bSig(), only: unrendered }); setBJob(null);
    } catch (e) { setErr(String((e as Error).message || e)); }
    finally { setBusyMsg(""); }
  };
  const bReady = bDry ? bDry.scenes.filter((r) => r.status === "dry") : [];
  const bFlagged = bReady.filter((r) => (r.flags || []).length);
  const bSendable = bReady.length - (bAccept ? 0 : bFlagged.length);
  const bFresh = !!bDry && bDry.sig === bSig();
  const bSending = bJob?.status === "sending";
  const doBatchFire = async () => {
    if (!bDry || !bFresh || bSendable === 0) return;
    if (!window.confirm(`Render ${bSendable} scene${bSendable === 1 ? "" : "s"} on the box now? They are sent one after another (30-60 s each) and render in the box queue.`)) return;
    setErr("");
    try {
      const r = await api.mvRenderScript({ from_dry: bDry.manifest, accept_flags: bAccept ? bFlagged.map((x) => x.index) : [], async: true });
      setBJob({ id: (r as { job_id: string }).job_id, status: "sending", rows: [], total: bSendable });
    } catch (e) { setErr(String((e as Error).message || e)); }
  };
  useEffect(() => {
    if (!bJob || bJob.status !== "sending") return;
    const t = window.setInterval(async () => {
      try {
        const j = (await api.mvRenderScriptJob(bJob.id)) as BatchJob;
        setBJob((b) => ({ ...j, total: b?.total }));
        await p.refreshTakes();
        if (j.status !== "sending") setBDry(null);
      } catch (e) { setErr(String((e as Error).message || e)); setBJob((b) => (b ? { ...b, status: "error" } : b)); }
    }, 4000);
    return () => window.clearInterval(t);
  }, [bJob?.id, bJob?.status]); // eslint-disable-line react-hooks/exhaustive-deps
  // progress: scenes sent (rows past the ones that skip straight through) and the box's renders of them
  const bTodo = (bJob?.rows || []).filter((r) => r.status !== "existing" && r.status !== "skipped");
  const bQueued = bTodo.filter((r) => r.status === "queued" && r.job_id);
  const sentFrac = bJob?.total ? Math.min(1, bTodo.length / bJob.total) : 0;
  const takeFrac = (r: BatchRow) => {
    const t = list(r.index).find((x) => x.id === r.job_id);
    if (!t) return 0;
    return t.status === "done" || t.status === "error" ? 1 : t.max ? Math.min(0.99, (t.progress || 0) / t.max) : 0;
  };
  const renderDone = bQueued.filter((r) => takeFrac(r) === 1).length;
  const renderFrac = bQueued.length ? bQueued.reduce((a, r) => a + takeFrac(r), 0) / bQueued.length : 0;
  const renderNow = bQueued.find((r) => { const f = takeFrac(r); return f > 0 && f < 1; });
  const errText = (e: unknown) => (typeof e === "string" ? e : JSON.stringify(e));

  const shown = script.scenes.filter((s) => filter === "all" || stateOf(s).todo);
  const tl = list(scene.index);
  const lockedElsewhere = scene.locked && !tl.some((t) => t.id === scene.locked!.job_id);
  const dr = dry[scene.index];

  return (
    <div className="grid gap-4 lg:grid-cols-[320px_1fr]">
      {/* top: render every scene that has no take yet, in one go */}
      <Card className="lg:col-span-2" title={`Render the rest: ${unrendered.length} scene${unrendered.length === 1 ? "" : "s"} with no take yet`}
        sub="one settings check for all of them (no GPU), then one button sends them all to the box. Scenes that already have a take are left alone; render those one at a time below.">
        <div className="flex flex-wrap items-center gap-2">
          <GhostButton onClick={doBatchDry} disabled={!!busyMsg || bSending || unrendered.length === 0}>1 · Check all {unrendered.length}</GhostButton>
          <Action onClick={doBatchFire} disabled={!!busyMsg || bSending || !bFresh || bSendable === 0}
            title={bFresh ? "send them to the box" : "check them first"}>2 · Render {bFresh ? bSendable : unrendered.length} on the box</Action>
          {bFresh && bFlagged.length > 0 && <label className="flex items-center gap-1 text-xs text-amber-400"><input type="checkbox" checked={bAccept} onChange={(e) => setBAccept(e.target.checked)} /> also render the {bFlagged.length} with warnings</label>}
        </div>
        {bDry && !bFresh && !bJob && <div className="text-xs text-amber-400">the settings or the scenes changed since the check: check again</div>}
        {bDry && bFresh && !bJob && (
          <div className={`rounded border p-2 text-[11px] ${bFlagged.length || bDry.scenes.some((r) => r.status === "error") ? "border-amber-700" : "border-emerald-800"}`}>
            <div className="text-[var(--color-muted)]">{bReady.length} ready{bFlagged.length ? `, ${bFlagged.length} with warnings` : ""}{bDry.counts?.error ? `, ${bDry.counts.error} cannot render` : ""} · {p.width}x{p.height} · {p.turbo ? "turbo" : "base"} · {p.seed.trim() ? `seed ${p.seed}` : "a random seed per scene (kept from this check)"}</div>
            {bDry.scenes.filter((r) => r.status === "error").map((r) => <div key={r.index} className="mt-1 text-red-300">scene {r.index} {r.title}: {errText(r.error)}</div>)}
            {bFlagged.map((r) => <div key={r.index} className="mt-1 text-amber-300">scene {r.index} {r.title}: {(r.flags || []).join("; ")}</div>)}
            {!bFlagged.length && !bDry.counts?.error && <div className="mt-1 text-emerald-500">no warnings: all ready to render</div>}
          </div>
        )}
        {bJob && (
          <div className="space-y-2 rounded border border-[var(--color-line)] p-2 text-[11px]">
            <Bar label={`sent to the box: ${bTodo.length} of ${bJob.total ?? "?"}`} frac={bJob.status === "done" ? 1 : sentFrac} />
            {bQueued.length > 0 && <Bar label={`rendered: ${renderDone} of ${bQueued.length}${renderNow ? ` · now scene ${renderNow.index} ${pct(Math.round(takeFrac(renderNow) * 100), 100)}` : ""}`} frac={renderFrac} />}
            <div className={bJob.status === "error" ? "text-red-300" : "text-[var(--color-muted)]"}>
              {bJob.status === "sending" ? `sending to the box: ${bJob.rows.filter((r) => r.status === "queued").length} sent so far (the box renders them in order; takes appear below as they finish)`
                : bJob.status === "done" ? `done: ${bJob.counts?.queued || 0} scenes queued on the box${bJob.counts?.blocked ? `, ${bJob.counts.blocked} held back` : ""}${bJob.counts?.error ? `, ${bJob.counts.error} failed` : ""}`
                : `stopped: ${bJob.error || "error"}`}
            </div>
            {bJob.rows.filter((r) => r.status === "blocked" || r.status === "error").map((r) => <div key={r.index} className="mt-1 text-amber-300">scene {r.index} {r.title}: {errText(r.error)}</div>)}
          </div>
        )}
      </Card>
      {/* left: every scene */}
      <div className="space-y-2">
        <div className="flex gap-2">
          <GhostButton onClick={() => setFilter("all")} className={filter === "all" ? "!text-[var(--color-ink)] border-[var(--color-accent)]" : ""}>all {script.scenes.length}</GhostButton>
          <GhostButton onClick={() => setFilter("todo")} className={filter === "todo" ? "!text-[var(--color-ink)] border-[var(--color-accent)]" : ""}>to do {script.scenes.filter((s) => stateOf(s).todo).length}</GhostButton>
        </div>
        <div className="max-h-[75vh] space-y-1 overflow-auto pr-1">
          {shown.map((sc) => {
            const st = stateOf(sc);
            return (
              <button key={sc.index} onClick={() => p.setSel(sc.index)}
                className={`flex w-full items-center gap-2 rounded-lg border p-1.5 text-left ${sc.index === scene.index ? "border-[var(--color-accent)] bg-[#2a1c19]" : "border-[var(--color-line)] bg-[var(--color-panel)] hover:border-[var(--color-muted)]"}`}>
                <div className="h-10 w-[72px] flex-none overflow-hidden rounded bg-black">
                  {sc.opening_still_id && <img src={`/api/media/${sc.opening_still_id}`} alt="" className="h-full w-full object-cover" loading="lazy" />}
                </div>
                <div className="min-w-0 flex-1">
                  <div className="truncate text-xs font-semibold">{String(sc.index).padStart(2, "0")} {sc.title}</div>
                  <div className="flex items-center gap-2 text-[10px] text-[var(--color-muted)]">{mmss(sc.start)} <Pill tone={st.tone}>{st.text}</Pill></div>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* right: the selected scene */}
      <div className="space-y-4">
        <div className="flex flex-wrap items-center gap-3">
          <div className="text-lg font-semibold">Scene {scene.index} · {scene.title}</div>
          <span className="text-xs text-[var(--color-muted)]">{fmt(scene.start)} - {fmt(scene.end)} ({(scene.end - scene.start).toFixed(1)} s) · {scene.clips?.length || 0} clip{(scene.clips?.length || 0) === 1 ? "" : "s"}{scene.lipsync ? " · lip-sync" : ""}</span>
          <div className="ml-auto flex gap-2">
            <GhostButton onClick={() => p.setSel(Math.max(0, scene.index - 1))} disabled={scene.index === 0}>previous scene</GhostButton>
            <GhostButton onClick={() => p.setSel(Math.min(script.scenes.length - 1, scene.index + 1))} disabled={scene.index >= script.scenes.length - 1}>next scene</GhostButton>
          </div>
        </div>
        {(scene.scene_problems || []).length > 0 && (
          <div className="rounded border border-red-800 p-2 text-xs text-red-300">{(scene.scene_problems || []).map((x, i) => <div key={i}>{x}</div>)}</div>
        )}

        <div className="grid gap-4 xl:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
          <Card title="What the scene is" sub={pictureUse(scene)}>
            <ImageSlot src={scene.opening_still_id ? `/api/media/${scene.opening_still_id}` : null}
              armed={JSON.stringify(p.target) === JSON.stringify({ kind: "scene", idx: scene.index })}
              onArm={() => p.setTarget({ kind: "scene", idx: scene.index })} onFile={(f) => p.fileStill(scene.index, f)}
              empty={<><div className="font-semibold">no opening picture</div><div>click + Cmd+V or drop one here, or add them all in step 3</div></>}>
              <GhostButton onClick={p.goPictures}>all pictures (step 3)</GhostButton>
            </ImageSlot>
            <div className="space-y-1">
              <Row k="action" v={scene.action} />
              <Row k="performance" v={scene.performance?.replace(/_/g, " ")} />
              <Row k="framing" v={scene.framing} />
              <Row k="camera" v={scene.camera} />
              <Row k="setting" v={scene.setting} />
              <Row k="others" v={scene.others} />
              <Row k="lighting" v={scene.lighting} />
              <Row k="mood" v={scene.mood} />
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
          </Card>

          <div className="space-y-4">
            <EndOfMove scene={scene} startPic={p.startPicOf(scene)} scriptName={scriptName} width={p.width} height={p.height}
              busy={!!busyMsg} setErr={setErr} reload={p.reloadScript} />
            <Card title="Render this scene on the box" sub="first check the settings (no GPU, shows any warnings), then render. Each render adds a take below.">
              <div className="flex flex-wrap items-center gap-2">
                <GhostButton onClick={() => doDry(scene.index)} disabled={!!busyMsg}>1 · Check the settings</GhostButton>
                <Action onClick={() => doRender(scene.index)} disabled={!!busyMsg || !canRender(scene.index)}
                  title={canRender(scene.index) ? "queue this scene on the box" : "check the settings first; warnings need the accept tick"}>2 · Render on the box</Action>
                {dr && dr.flags.length > 0 && <label className="flex items-center gap-1 text-xs text-amber-400"><input type="checkbox" checked={p.acceptFlags} onChange={(e) => p.setAcceptFlags(e.target.checked)} /> render despite the warnings</label>}
              </div>
              {dr && (
                <div className={`rounded border p-2 text-[11px] ${dr.sig !== sig() || dr.flags.length ? "border-amber-700" : "border-emerald-800"}`}>
                  <div className="flex flex-wrap gap-x-3 gap-y-1 text-[var(--color-muted)]">
                    <span>{dr.clips} clip{dr.clips === 1 ? "" : "s"}</span><span>{dr.settings?.width}x{dr.settings?.height}</span><span>seed {dr.settings?.seed}</span>
                    <span>{dr.settings?.turbo ? "turbo" : "base"}</span>{dr.lane === "i2v" ? <span>starts on {dr.first_frame_id ? "a picture" : "the place's picture"}</span> : <span>renders from the singer's and the place's pictures</span>}
                    {dr.end_still_id ? <span>ends on the end picture</span> : null}
                    {dr.rail_frames?.length ? <span>pinned on the way at {dr.rail_frames.map((f) => (f / 24).toFixed(1)).join(", ")} s</span> : null}
                    {dr.sig !== sig() || (dr.end_still_id || null) !== (scene.end_still_id || null) ? <span className="text-amber-400">settings changed since the check: check again</span> : null}
                  </div>
                  {seedUsed(scene.index, dr.settings?.seed, dr) && <div className="mt-1 text-amber-300">seed {dr.settings?.seed} already made a take of this scene with the same end picture, so it would most likely come out the same. Clear the seed in the render settings and check again for a new one.</div>}
                  {dr.flags.length ? <ul className="mt-1 list-disc pl-4 text-amber-300">{dr.flags.map((f, i) => <li key={i}>{f}</li>)}</ul> : <div className="mt-1 text-emerald-500">no warnings: ready to render</div>}
                </div>
              )}
              <details className="text-xs">
                <summary className="cursor-pointer text-[var(--color-muted)]">render settings (the proven defaults; the singer's pictures are set in step 3)</summary>
                <div className="mt-2 grid grid-cols-2 gap-2 md:grid-cols-4">
                  <Field label="seed" hint="blank = a new random seed on every check"><input className={inp} value={p.seed} onChange={(e) => p.setSeed(e.target.value)} /></Field>
                  <Field label="width"><input className={inp} type="number" step={32} value={p.width} onChange={(e) => p.setWidth(Number(e.target.value))} /></Field>
                  <Field label="height"><input className={inp} type="number" step={32} value={p.height} onChange={(e) => p.setHeight(Number(e.target.value))} /></Field>
                  <label className="flex items-end gap-2 pb-2 text-xs"><input type="checkbox" checked={p.turbo} onChange={(e) => p.setTurbo(e.target.checked)} /> turbo</label>
                </div>
              </details>
            </Card>

            <Card title={`Takes (${tl.length})`} sub='watch them and press "Use in the video" on the one that should play in this scene'>
              {tl.length === 0 && !lockedElsewhere && <div className="text-sm text-[var(--color-muted)]">no renders of this scene yet</div>}
              {lockedElsewhere && (
                <div className="space-y-2 rounded-lg border-2 border-emerald-700 p-2">
                  <video src={`/api/media/${scene.locked!.job_id}`} controls muted preload="metadata" className="w-full rounded bg-black" />
                  <div className="flex items-center gap-2"><Pill tone="ok">in the video</Pill><span className="text-[11px] text-[var(--color-muted)]">rendered under an earlier script</span>
                    <GhostButton onClick={() => choose(scene.index, null)} className="ml-auto">remove</GhostButton></div>
                </div>
              )}
              <div className="grid gap-3 2xl:grid-cols-2">
                {tl.map((t) => {
                  const chosen = scene.locked?.job_id === t.id;
                  const c = t.check;
                  const u = t.upscaled;
                  return (
                    <div key={t.id} className={`space-y-2 rounded-lg p-2 ${chosen ? "border-2 border-emerald-600" : "border border-[var(--color-line)]"}`}>
                      {t.status === "done" && t.media_url
                        ? <video src={t.media_url} controls muted preload="metadata" className="w-full rounded bg-black" />
                        : <div className="rounded bg-black/40 p-4"><Bar label={`${t.status} on the box`} frac={t.max ? (t.progress || 0) / t.max : 0} /></div>}
                      <div className="flex flex-wrap items-center gap-2">
                        {t.status === "done" && (chosen
                          ? <><Pill tone="ok">in the video</Pill><GhostButton onClick={() => choose(scene.index, null)}>remove</GhostButton></>
                          : <Action onClick={() => choose(scene.index, t.id)} title="this take plays in the scene; it replaces the scene's current choice">Use in the video</Action>)}
                        <span className="text-[11px] text-[var(--color-muted)]">{when(t.created)} · seed {t.params.seed}{t.seconds !== null ? ` · ${t.seconds.toFixed(1)} s` : ""}{t.from_script ? " · earlier script" : ""}</span>
                      </div>
                      {t.status === "done" && (
                        <div className="flex flex-wrap items-center gap-2 text-[11px]">
                          {c ? <>
                            <Pill tone={c.floor_r === null ? "muted" : c.floor_ok ? "ok" : "bad"} title="lip and music sync against the song; clean takes measure 0.87-1.00">sync {c.floor_r === null ? "n/a" : c.floor_r.toFixed(2)}</Pill>
                            <Pill tone={c.opening_jump === null ? "muted" : c.opening_flag ? "bad" : "ok"} title="a jump in the first 2 s (fade or morph in); 4 or more is flagged">opening {c.opening_jump === null ? "n/a" : c.opening_jump.toFixed(1)}x</Pill>
                            <GhostButton onClick={() => openLightbox(c.sheet_url + "?t=" + Date.now())}>frames</GhostButton>
                          </> : <GhostButton onClick={() => doCheck(scene.index, t.id)} disabled={!!busyMsg} title="measure sync and the opening (Mac, no GPU)">measure sync</GhostButton>}
                          {u?.status === "done" && u.media_url
                            ? <><Pill tone="ok">upscaled 2x</Pill><GhostButton onClick={() => openLightbox(u.media_url!)}>play upscale</GhostButton></>
                            : upscaleBusy(u) ? <Pill tone="warn">upscaling {pct(u!.progress, u!.max) || u!.status}</Pill>
                            : <GhostButton onClick={() => doUpscale(scene.index, t)} disabled={!!busyMsg} title="FlashVSR+ Turbo 2x on the box (GPU); step 5 can upscale every chosen take at once">upscale 2x</GhostButton>}
                          {u?.status === "error" && <span className="text-red-400" title={u.error || ""}>upscale failed</span>}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------- 5 Final video: the chosen takes -> the cut, with the ident in front and the end card ----------
type AsmResult = { output: string; seconds: number; scenes: number; partial?: boolean; missing?: number[]; intro?: { seconds: number; cached: boolean }; end_card?: { secs: number; text: string }; endcard_at?: number | null; library_id?: string; media_url?: string; upscaled?: Record<string, string>; not_upscaled?: number[] };

const QUALITY: { key: string; label: string; sub: string; size: string; up: boolean }[] = [
  { key: "draft", label: "Draft", sub: "the renders as they are (1344x768): a quick check of the edit", size: "1344x768", up: false },
  { key: "hd", label: "1080p from the renders", sub: "the renders resized to 1920x1080, no upscale", size: "1920x1080", up: false },
  { key: "up1440", label: "1440p, upscaled", sub: "every take upscaled 2x with FlashVSR+, cut at 2560x1440: the YouTube master", size: "2560x1440", up: true },
  { key: "up4k", label: "4K, upscaled", sub: "the 2x upscales stretched to 3840x2160", size: "3840x2160", up: true },
];

function FinalStep({ script, scriptName, takes, refreshTakes, reloadScript }: { script: Script; scriptName: string; takes: Record<string, Take[]>; refreshTakes: () => Promise<void>; reloadScript: () => void }) {
  const d = useDrafts("mv");
  const [quality, setQuality] = d.use("asmQuality", "draft");
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
  const [msg, setMsg] = useState("");
  const [busy, setBusy] = useState(false);
  const [cuts, setCuts] = useState<Cut[]>([]);
  const [copyFrom, setCopyFrom] = useState("");
  const title = introTitle.trim() || script.title;
  const q = QUALITY.find((x) => x.key === quality) || QUALITY[0];

  useEffect(() => {
    let alive = true;
    api.mvAssemblies(scriptName).then((r) => { if (alive) setCuts((r as Cut[]).sort((a, b) => b.stamp.localeCompare(a.stamp))); }).catch(() => { if (alive) setCuts([]); });
    return () => { alive = false; };
  }, [scriptName, result?.output]);

  const takeById = useMemo(() => {
    const m: Record<string, Take> = {};
    Object.values(takes).forEach((l) => l.forEach((t) => { m[t.id] = t; }));
    return m;
  }, [takes]);
  // the take per scene: the chosen one (the lock), else (opt-in) the newest finished take
  const plan = script.scenes.map((sc) => {
    const done = (takes[String(sc.index)] || []).filter((t) => t.status === "done");
    if (sc.locked) return { sc, job: sc.locked.job_id, how: "chosen" as const };
    if (fillNewest && done[0]) return { sc, job: done[0].id, how: "newest" as const };
    return { sc, job: null, how: "missing" as const };
  });
  const missing = plan.filter((x) => !x.job);
  const partial = missing.length > 0;
  const unchosen = plan.filter((x) => x.how === "newest").length;
  const upOf = (job: string | null) => (job ? takeById[job]?.upscaled || null : null);
  const planned = plan.filter((x) => x.job);
  const nUpDone = planned.filter((x) => upOf(x.job)?.status === "done").length;
  const nUpBusy = planned.filter((x) => upscaleBusy(upOf(x.job))).length;
  const toUpscale = planned.filter((x) => { const u = upOf(x.job); return !u || u.status === "error"; });

  const cut = cuts.find((c) => c.stamp === copyFrom) || null;
  const cutChanges = cut ? script.scenes.filter((sc) => cut.jobs[String(sc.index)] && cut.jobs[String(sc.index)] !== sc.locked?.job_id) : [];
  const adoptCut = async () => {
    if (!cut || !cutChanges.length) return;
    if (!window.confirm(`Make the takes of "${cut.file.replace(/\.mp4$/, "")}" (${stampWhen(cut.stamp)}) the chosen takes? ${cutChanges.length} scene${cutChanges.length === 1 ? "" : "s"} change: ${cutChanges.map((s) => s.index).join(", ")}.`)) return;
    setErr(""); setBusy(true);
    try {
      for (const sc of cutChanges) await api.mvLockTake({ script: scriptName, scene: sc.index, job_id: cut.jobs[String(sc.index)] });
      setMsg(`${cutChanges.length} scenes now use the takes of the ${stampWhen(cut.stamp)} video`);
      setCopyFrom("");
      reloadScript();
    } catch (e) { setErr(String((e as Error).message || e)); reloadScript(); }
    finally { setBusy(false); }
  };

  const upscaleAll = async () => {
    if (!toUpscale.length) return;
    const secs = toUpscale.reduce((a, x) => a + (takeById[x.job!]?.seconds || (x.sc.end - x.sc.start)), 0);
    const mins = Math.max(1, Math.round((secs * UPSCALE_SECS_PER_SEC) / 60));
    if (!window.confirm(`Upscale ${toUpscale.length} take${toUpscale.length === 1 ? "" : "s"} 2x with FlashVSR+ on the box? They run one after another: ${secs.toFixed(0)} s of video, roughly ${mins} min.`)) return;
    setErr(""); setMsg(""); setBusy(true);
    try {
      const jobs: Record<string, string> = {};
      toUpscale.forEach((x) => { jobs[String(x.sc.index)] = x.job!; });
      const r = await api.mvUpscaleTakes({ jobs, script: scriptName });
      setMsg(`queued ${r.queued?.length || 0} upscales${r.skipped?.length ? `, skipped ${r.skipped.length}` : ""}`);
      await refreshTakes();
    } catch (e) { setErr(String((e as Error).message || e)); }
    finally { setBusy(false); }
  };
  const previewIdent = async () => {
    setIdentBusy(true); setErr("");
    try { const r = await api.mvIdent({ title, sub: introSub, edge: introEdge }); setIdent({ url: r.url + "?t=" + Date.now(), seconds: r.seconds, cached: r.cached }); }
    catch (e) { setErr(String((e as Error).message || e)); }
    finally { setIdentBusy(false); }
  };
  const assemble = async () => {
    const [w, h] = q.size.split("x").map(Number);
    const what = partial ? `a PARTIAL video of ${planned.length} of ${plan.length} scenes (no intro or end card on a partial)` : `the full video, ${plan.length} scenes${introOn ? ", the intro in front" : ""}${ecOn ? `, a ${ecSecs} s end card` : ""}`;
    const upNote = q.up && nUpDone < planned.length ? ` ${planned.length - nUpDone} scene${planned.length - nUpDone === 1 ? " has" : "s have"} no upscale yet and will be stretched from the render.` : "";
    if (!window.confirm(`Make ${what} at ${q.label} (${q.size})?${upNote} It is cut on the Mac and filed in the Library under Music videos.`)) return;
    setErr(""); setResult(null);
    const jobs: Record<string, string> = {};
    plan.forEach((x) => { if (x.job && x.how !== "chosen") jobs[String(x.sc.index)] = x.job; });
    try {
      const r = await api.mvAssemble({
        script: scriptName, jobs, partial, width: w, height: h, async: true, use_upscaled: q.up,
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
        else if (r.status === "error") { setJob({ ...job, status: "error", progress: 0, max: 1 }); setErr(r.error || "the cut failed"); window.clearInterval(h); }
        else setJob((j) => (j ? { ...j, progress: r.progress, max: r.max } : j));
      } catch { /* keep polling */ }
    }, 3000);
    return () => window.clearInterval(h);
  }, [job?.id, job?.status]);   // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="space-y-4">
      <div className="grid gap-4 xl:grid-cols-[3fr_2fr]">
        <Card title={`Which take plays in each scene (${plan.filter((x) => x.how === "chosen").length} of ${plan.length} chosen)`}
          sub='choices are made in step 4 ("Use in the video"); to reuse an earlier video, copy its choices below'>
          <div className="flex flex-wrap items-end gap-2 rounded-lg bg-[var(--color-panel2)] p-2">
            <Field label="copy the choices of an earlier video">
              <select className={inp} value={copyFrom} onChange={(e) => setCopyFrom(e.target.value)}>
                <option value="">choose a video...</option>
                {cuts.filter((c) => Object.keys(c.jobs || {}).length).map((c) => (
                  <option key={c.stamp} value={c.stamp}>{stampWhen(c.stamp)} · {c.file.replace(/\.mp4$/, "")} · {c.scenes ?? "?"} scenes{c.partial ? " (partial)" : ""}</option>
                ))}
              </select>
            </Field>
            <GhostButton onClick={adoptCut} disabled={busy || !cut || !cutChanges.length}>{cut ? (cutChanges.length ? `use its takes (${cutChanges.length} scenes change)` : "already the same takes") : "use its takes"}</GhostButton>
          </div>
          <label className="flex items-center gap-2 text-xs"><input type="checkbox" checked={fillNewest} onChange={(e) => setFillNewest(e.target.checked)} /> scenes with no chosen take play their newest take{unchosen ? ` (${unchosen} scene${unchosen === 1 ? "" : "s"} now)` : ""}</label>
          <div className="max-h-[50vh] overflow-auto rounded-lg border border-[var(--color-line)]">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-[var(--color-panel)] text-[var(--color-muted)]"><tr><th className="px-2 py-1.5 text-left">scene</th><th className="px-2 py-1.5 text-left">at</th><th className="px-2 py-1.5 text-left">take</th><th className="px-2 py-1.5 text-left">upscaled</th></tr></thead>
              <tbody>
                {plan.map((x) => {
                  const t = x.job ? takeById[x.job] : null;
                  const u = upOf(x.job);
                  return (
                    <tr key={x.sc.index} className="border-t border-[var(--color-line)]">
                      <td className="px-2 py-1">{String(x.sc.index).padStart(2, "0")} {x.sc.title}</td>
                      <td className="px-2 py-1 text-[var(--color-muted)]">{mmss(x.sc.start)}</td>
                      <td className="px-2 py-1">{x.how === "chosen" ? <Pill tone="ok">chosen{t ? ` · ${when(t.created)}` : ""}</Pill>
                        : x.how === "newest" ? <Pill tone="warn" title="no take chosen: the newest finished take plays">not chosen · newest {t ? when(t.created) : ""}</Pill>
                        : <Pill tone="bad">no take</Pill>}</td>
                      <td className="px-2 py-1">{!x.job ? null : u?.status === "done" ? <Pill tone="ok">yes</Pill> : upscaleBusy(u) ? <Pill tone="warn">running {pct(u!.progress, u!.max)}</Pill> : u?.status === "error" ? <Pill tone="bad" title={u.error || ""}>failed</Pill> : <span className="text-[var(--color-muted)]">no</span>}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Card>

        <div className="space-y-4">
          <Card title="Quality">
            <div className="space-y-2">
              {QUALITY.map((o) => (
                <label key={o.key} className={`flex cursor-pointer gap-3 rounded-lg border p-2.5 ${quality === o.key ? "border-[var(--color-accent)] bg-[#2a1c19]" : "border-[var(--color-line)]"}`}>
                  <input type="radio" name="mvq" checked={quality === o.key} onChange={() => setQuality(o.key)} />
                  <div><div className="text-sm font-semibold">{o.label}</div><div className="text-[11px] text-[var(--color-muted)]">{o.sub}</div></div>
                </label>
              ))}
            </div>
            {q.up && (
              <div className={`rounded-lg border p-2.5 text-xs space-y-2 ${nUpDone === planned.length ? "border-emerald-800" : "border-amber-700"}`}>
                <div>{nUpDone} of {planned.length} takes upscaled{nUpBusy ? ` · ${nUpBusy} upscaling on the box now` : ""}</div>
                {toUpscale.length > 0 && <GhostButton onClick={upscaleAll} disabled={busy}>upscale the {toUpscale.length} missing on the box (GPU)</GhostButton>}
              </div>
            )}
          </Card>
          <Card title={<label className="flex items-center gap-2"><input type="checkbox" checked={introOn} onChange={(e) => setIntroOn(e.target.checked)} /> Intro (3 s channel ident)</label>}>
            {introOn && <>
              <div className="grid grid-cols-2 gap-2">
                <Field label="title"><input className={inp} value={introTitle} onChange={(e) => setIntroTitle(e.target.value)} placeholder={script.title} /></Field>
                <Field label="line under it"><input className={inp} value={introSub} onChange={(e) => setIntroSub(e.target.value)} /></Field>
              </div>
              <div className="flex items-end gap-2">
                <Field label="edge">
                  <select className={inp} value={introEdge} onChange={(e) => setIntroEdge(e.target.value)}>
                    <option value="none">none</option><option value="hairline">hairline</option><option value="burnt">burnt</option><option value="filigree">filigree (placeholder)</option>
                  </select>
                </Field>
                <GhostButton onClick={previewIdent} disabled={identBusy}>{identBusy ? "rendering..." : "preview"}</GhostButton>
              </div>
              {ident && <video src={ident.url} controls preload="metadata" className="w-full rounded bg-black" />}
            </>}
          </Card>
          <Card title={<label className="flex items-center gap-2"><input type="checkbox" checked={ecOn} onChange={(e) => setEcOn(e.target.checked)} /> YouTube end card</label>}>
            {ecOn && <>
              <div className="grid grid-cols-[auto_1fr] gap-2">
                <Field label="over the last">
                  <select className={inp} value={ecSecs} onChange={(e) => setEcSecs(e.target.value)}><option value="10">10 s</option><option value="15">15 s</option><option value="20">20 s</option></select>
                </Field>
                <Field label="text"><input className={inp} value={ecText} onChange={(e) => setEcText(e.target.value)} placeholder="More from Apotheon" /></Field>
              </div>
              <img src={`/api/youtube/endcard_preview?text=${encodeURIComponent(ecText)}&t=${encodeURIComponent(ecText.length)}`} alt="" className="w-full rounded border border-[var(--color-line)]" />
              <div className="text-[11px] text-[var(--color-muted)]">in YouTube Studio, put the "two videos + subscribe" end screen over it</div>
            </>}
          </Card>
        </div>
      </div>

      <Card>
        <div className="flex flex-wrap items-center gap-4">
          <Action big onClick={assemble} disabled={busy || job?.status === "running" || !planned.length}>{partial ? `Make a partial video (${planned.length} of ${plan.length} scenes)` : "Make the video"}</Action>
          <div className="text-sm text-[var(--color-muted)]">{q.label} · {q.size}{introOn && !partial ? " · intro" : ""}{ecOn && !partial ? ` · ${ecSecs} s end card` : ""}{q.up && nUpDone < planned.length ? ` · ${planned.length - nUpDone} scenes not upscaled` : ""}</div>
          {job?.status === "running" && <div className="text-sm">cutting... {pct(job.progress, job.max)}</div>}
        </div>
        {(msg || err) && <div className={`text-xs ${err ? "text-red-400" : "text-[var(--color-muted)]"}`}>{err || msg}</div>}
        {result && (
          <div className="space-y-2 rounded-lg border border-emerald-800 p-3">
            <div className="text-sm">
              {result.partial ? "Partial video" : "Video"} made · {mmss(result.seconds || 0)} · {result.scenes} scenes
              {result.upscaled ? ` · ${Object.keys(result.upscaled).length} upscaled${result.not_upscaled?.length ? `, ${result.not_upscaled.length} not (${result.not_upscaled.join(", ")})` : ""}` : ""}
              {result.library_id ? " · in the Library under Music videos" : ""}
            </div>
            {result.media_url ? <video src={result.media_url} controls preload="metadata" className="w-full max-w-5xl rounded bg-black" /> : null}
          </div>
        )}
      </Card>

      {cuts.length > 0 && (
        <Card title={`Videos made from this script (${cuts.length})`}>
          <div className="max-h-72 overflow-auto">
            {cuts.map((c) => (
              <div key={c.stamp} className="flex flex-wrap items-center gap-3 border-t border-[var(--color-line)] py-1.5 text-xs">
                <span className="w-24 text-[var(--color-muted)]">{stampWhen(c.stamp)}</span>
                <span className="flex-1 truncate" title={c.path}>{c.file.replace(/\.mp4$/, "")}</span>
                <span className="text-[var(--color-muted)]">{c.seconds ? mmss(c.seconds) : ""} · {c.scenes ?? "?"} scenes{c.partial ? " (partial)" : ""}{c.intro ? " · intro" : ""}{c.end_card ? " · end card" : ""}</span>
                <GhostButton onClick={() => openLightbox(c.media_url || `/api/mv/cut/${c.stamp}/${encodeURIComponent(c.file)}`)}>play</GhostButton>
              </div>
            ))}
          </div>
        </Card>
      )}
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

function WriterPanel({ audioId, loaded, onWritten, startOpen }: { audioId: string; loaded: { name: string; script: Script } | null; onWritten: (name: string) => Promise<void>; startOpen: boolean }) {
  const d = useDrafts("mv");
  const drafts = useDraftCtx();
  const [open, setOpen] = useState(startOpen);
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
    if (!ready) { setErr("title, lyrics and the three descriptions of the singer are needed"); return; }
    if (!window.confirm(`Write a new script for "${title}"? Claude (the subscription, no GPU) writes the whole video; it takes several minutes.${keepLocks && nLocks ? ` The ${nLocks} chosen takes of the open script are kept.` : ""}`)) return;
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
    <Card title={<button className="flex w-full items-center gap-2 text-left" onClick={() => setOpen(!open)}>
        <span className="text-[var(--color-muted)]">{open ? "▾" : "▸"}</span> Write a new script
        {job?.status === "writing" && <span className="font-normal text-[var(--color-muted)]">writing... {Math.round(job.seconds)} s</span>}
      </button>}
      sub="Claude writes every scene from the lyrics and your idea (no GPU, several minutes)">
      {open && (
        <div className="space-y-3">
          <div className="grid gap-2 md:grid-cols-2">
            <Field label="title"><input className={inp} value={title} onChange={(e) => setTitle(e.target.value)} /></Field>
            <Field label="style" hint="genre and mood, filled from the song"><input className={inp} value={style} onChange={(e) => setStyle(e.target.value)} /></Field>
          </div>
          <Field label="lyrics" hint="with section tags ([Verse], [Chorus] ...)">
            <textarea className={`${inp} h-40 font-mono text-[11px]`} value={lyrics} onChange={(e) => setLyrics(e.target.value)} placeholder="paste the lyrics, or take them from the open project" />
          </Field>
          <div className="flex flex-wrap gap-2">
            <GhostButton onClick={() => { const t = lyricsFromProject(drafts.get); if (t.trim()) setLyrics(t); else setErr("the open project has no lyrics (Suno, Music 3 or Song builder)"); }}>lyrics from the open project</GhostButton>
            {loaded && <GhostButton onClick={fromScript} title="the singer's descriptions (and the idea, if empty) of the open script">copy the singer from the open script</GhostButton>}
          </div>
          <Field label="your idea for the video" hint="optional; empty = Claude proposes one">
            <textarea className={`${inp} h-16`} value={concept} onChange={(e) => setConcept(e.target.value)} />
          </Field>
          <div className="text-xs font-semibold">The singer, in words (what her two reference pictures show)</div>
          <div className="grid gap-2 md:grid-cols-3">
            <Field label="face picture" hint="one line"><textarea className={`${inp} h-20 text-[11px]`} value={face} onChange={(e) => setFace(e.target.value)} /></Field>
            <Field label="full-body picture" hint="one line"><textarea className={`${inp} h-20 text-[11px]`} value={body} onChange={(e) => setBody(e.target.value)} /></Field>
            <Field label="in both" hint="what never changes"><textarea className={`${inp} h-20 text-[11px]`} value={shared} onChange={(e) => setShared(e.target.value)} /></Field>
          </div>
          <div className="flex flex-wrap items-end gap-3">
            <Field label="places">
              <select className={inp} value={locations} onChange={(e) => setLocations(e.target.value)}>
                <option value="auto">this song's places (default)</option>
                <option value="library">every place with a picture</option>
                <option value="none">invent them all</option>
              </select>
            </Field>
            {nLocks > 0 && <label className="flex items-center gap-2 pb-2 text-xs"><input type="checkbox" checked={keepLocks} onChange={(e) => setKeepLocks(e.target.checked)} /> keep the open script's {nLocks} chosen takes</label>}
            <Action onClick={write} disabled={!ready || job?.status === "writing"}>{job?.status === "writing" ? "writing..." : "Write the script"}</Action>
          </div>
          {job && (
            <div className={`rounded border p-2 text-[11px] ${job.status === "error" ? "border-red-800" : job.status === "done" ? "border-emerald-800" : "border-[var(--color-line)]"}`}>
              <div>{job.status === "writing" ? `writing, ${Math.round(job.seconds)} s` : job.status === "done" ? "written and opened" : `failed: ${job.error}`}</div>
              {job.log.length > 0 && <pre className="mt-1 max-h-32 overflow-auto whitespace-pre-wrap text-[10px] text-[var(--color-muted)]">{job.log.slice(-12).join("\n")}</pre>}
            </div>
          )}
          {err && <div className="text-[11px] text-red-400">{err}</div>}
        </div>
      )}
    </Card>
  );
}

// ---------- 6 YouTube upload text: the YouTube tab's writer, in its music-video form ----------
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
    <Card title="YouTube upload text" sub="Claude writes the title line, description (keywords and hashtags) and tags from the song, the lyrics and the video's idea">
      <div className="flex flex-wrap items-end gap-2">
        <div className="min-w-80 flex-1"><Field label="notes" hint="optional: anything the description should mention"><input className={inp} value={notes} onChange={(e) => setNotes(e.target.value)} /></Field></div>
        <Action onClick={write} disabled={busy}>{busy ? "Writing..." : meta ? "Write it again" : "Write the upload text"}</Action>
      </div>
      {meta && (
        <div className="space-y-2">
          {meta.for && meta.for !== script.title && <div className="text-[11px] text-amber-400">written for "{meta.for}"; write it again for this song</div>}
          {([["Video title", meta.video_title], ["Description", meta.description], ["Tags (comma-separated)", meta.tags.join(", ")]] as const).map(([label, value]) => (
            <div key={label} className="rounded-lg border border-[var(--color-line)] bg-[var(--color-panel2)] p-2">
              <div className="mb-1 flex items-center gap-2">
                <span className="text-[10px] font-medium uppercase tracking-wider text-[var(--color-muted)]">{label}</span>
                <button onClick={() => navigator.clipboard?.writeText(value)}
                  className="ml-auto rounded border border-[var(--color-line)] px-1.5 py-0.5 text-[10px] text-[var(--color-muted)] hover:text-[var(--color-ink)]">copy</button>
              </div>
              <pre className="whitespace-pre-wrap break-words text-xs leading-snug text-[var(--color-ink)]">{value}</pre>
            </div>
          ))}
          <p className="text-[11px] text-[var(--color-muted)]">
            Paste the description straight into YouTube's Description field (the first three hashtags show above the title).
            Tags go in the Tags box under "Show more". Tick the altered or synthetic content disclosure.
          </p>
        </div>
      )}
      {err && <div className="rounded border border-red-800 p-2 text-xs text-red-300">{err}</div>}
    </Card>
  );
}
