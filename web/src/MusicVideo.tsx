import { Fragment, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
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
type Locked = { job_id: string; take_start: number; script: string; chosen_at?: number };
type Scene = {
  takes_after?: number;
  index: number; title: string; start: number; end: number; location?: string; setting?: string; performance?: string; singer?: string;
  framing?: string; camera?: string; lighting?: string; mood?: string; action?: string; others?: string; why?: string;
  opening_still?: string; opening_still_id?: string; end_still_id?: string; after_still_id?: string; still_clips?: Clip[]; still_of_camera?: string; empty_room_id?: string; empty_room_of?: string; rail?: { end_still_id: string; every: number; after_still_id?: string | null; stills: { still_id: string; frame: number; zoom: number }[] }; lipsync?: boolean; scene_problems?: string[]; clips?: Clip[];
  locked?: Locked; plan?: { frames: number; context_frames: number }; sections?: string[]; matches_location_still?: boolean;
};
type Refs = { face?: string; body?: string; shared?: string };
type Place = { id: string; name?: string; description?: string; still_prompt?: string; still_id?: string };
type PlaceRow = Place & { still_present?: boolean };
type Script = { title: string; audio_id: string; duration: number; style?: string; concept?: string; look?: string; wardrobe?: string; arc?: string; refs?: Refs; ref_stills?: { face?: string; body?: string; him_face?: string; him_body?: string }; locations?: Place[]; scenes: Scene[];
  duet?: { refs2: Refs; voice?: string; how?: string; swap?: boolean; sung_seconds?: { her: number; him: number } }; wardrobe2?: string };
type RefKind = "face" | "body" | "him_face" | "him_body";
type VoiceLine = { start: number; end: number; text: string; voice: string; unsure?: boolean;
  answers?: { start: number; end: number; voice: string }[] };
type Voices = { how: string; swap: boolean; separation: number; sung_seconds: { her: number; him: number }; duration: number;
  sections: { type: string; start: number; end: number; lines: VoiceLine[] }[]; for?: string };
type ScriptRow = { name: string; title: string; audio_id: string; mtime: number; duration: number; scenes: number; clips: number; stills: number; locks: number; problems: number; videos?: number; last_video?: string | null };
type Check = { floor_r: number | null; floor_lag: number | null; floor_ok: boolean; opening_jump: number | null; opening_flag: boolean; sheet_url: string; seconds: number | null; planned_seconds: number | null };
type Upscale = { id: string; status: string; progress: number; max: number; error?: string | null; media_url: string | null; engine?: string; scale?: number; resolution?: number | null; fps?: number | null; fps_fixed?: boolean | null; temporal_compression?: number | null; newer?: Upscale | null };
type Take = { id: string; created: number; status: string; progress: number; max: number; media_url: string | null; seconds: number | null; params: Record<string, any>; check: Check | null; from_script?: string | null; stale?: boolean; finished?: number | null; upscaled?: Upscale | null; upscales?: Partial<Record<UpEngine, Upscale>> };
type Cut = { stamp: string; file: string; path: string; seconds: number | null; scenes: number | null; partial: boolean; script: string | null; jobs: Record<string, string>; intro: boolean; end_card: { secs: number; text: string } | null; library_id?: string | null; media_url?: string | null; mtime: number };
type Dry = { settings: Record<string, any>; flags: string[]; prompt_hash: string; prompts: string[]; clips: number; lane: string; ref_ids: string[]; first_frame_id?: string | null; anchor_first?: boolean; anchor_from?: string | null; end_still_id?: string | null; rail_frames?: number[] | null; variant?: string | null; sig: string };
type Target = { kind: RefKind } | { kind: "scene"; idx: number } | { kind: "after"; idx: number } | { kind: "empty"; idx: number; take: string } | { kind: "place"; id: string } | null;

const ATTENTION = "comfy kitchen attention";   // the proven chain setting (every SSBS fire script)
// FlashVSR+ Turbo 2x: 554 s for the 15 s scene 6 take, 2026-10-30 (the whole SSBS batch ran at
// about that rate); only a rough guide for the time shown before a fire
const UPSCALE_SECS_PER_SEC = 37;
// FlashVSR+ "fast motion" = temporal_compression 1 (the node README's setting for larger motion): the
// user judged it better on fast-moving scenes, 4 fine on still ones (SSBS scenes 8/32, 2026-11-11).
// Measured 1791 s for a 9.4 s take and 1356 s for 7.3 s, about 190 s a second
const FAST_SECS_PER_SEC = 190;
const isFast = (u?: Upscale | null) => u?.temporal_compression === 1;
// the two upscalers (backend mv_upscale_takes): FlashVSR+ Turbo 2x, and SeedVR2 7B fp8 to a 1440
// short side (brought back 2026-11-05 to compare again; no timing measured on the 7B yet)
type UpEngine = "flashvsr_plus" | "seedvr2" | "rtx_vsr";
const UP_ENGINES: Record<UpEngine, { name: string; button: string; done: string; tip: string; secsPerSec: number | null }> = {
  flashvsr_plus: { name: "FlashVSR+", button: "upscale 2x", done: "upscaled 2x", tip: "FlashVSR+ Turbo 2x on the box (GPU); step 5 can upscale every chosen take at once", secsPerSec: UPSCALE_SECS_PER_SEC },
  seedvr2: { name: "SeedVR2", button: "SeedVR2 1440p", done: "SeedVR2 1440p", tip: "SeedVR2 3B (fp16, the node author's video settings) to 1440p on the box (GPU), to compare with FlashVSR+; no timing measured yet", secsPerSec: null },
  // NVIDIA RTX Video Super Resolution (added 2026-11-11): non-generative, so no invented texture to crawl on moving areas
  rtx_vsr: { name: "RTX VSR", button: "RTX VSR 2x", done: "RTX VSR 2x", tip: "NVIDIA RTX Video Super Resolution 2x (HIGHBITRATE_ULTRA) on the box (GPU): sharpens without inventing detail, to compare with FlashVSR+ on moving areas; no timing measured yet", secsPerSec: null },
};
const upAsk = (eng: UpEngine) => eng === "seedvr2" ? "to 1440p with SeedVR2" : eng === "rtx_vsr" ? "2x with RTX Video Super Resolution (HIGHBITRATE_ULTRA)" : "2x with FlashVSR+";
const upMins = (eng: UpEngine, secs: number) => { const r = UP_ENGINES[eng].secsPerSec; return r ? `roughly ${Math.max(1, Math.round((secs * r) / 60))} min` : "time not measured yet"; };
const upscaleBusy = (u?: Upscale | null) => !!u && u.status !== "done" && u.status !== "error";
// a backend from before SeedVR2 sends only upscaled (the FlashVSR+ one)
const upsOf = (t?: Take | null): Partial<Record<UpEngine, Upscale>> => t?.upscales || (t?.upscaled ? { flashvsr_plus: t.upscaled } : {});
// how a scene's opening picture is used (backend render_scene): a scene WITHOUT the singer renders
// image-to-video and starts on it (else on its place's picture; a scene the writer marked as
// differing from the place's picture must have its own); a scene WITH her renders from her two
// pictures and the place's picture, and its opening picture is not sent unless the scene has a
// rail (s.20.53), which pins frame 0 to it
const envOnly = (sc: Scene) => sc.performance === "no_singing_environment";
function SingerBadge({ sc }: { sc: Scene }) {
  if (!sc.singer || sc.singer === "none") return null;
  return <span className={`rounded px-1 text-[10px] ${sc.singer === "him" ? "bg-sky-900 text-sky-200" : "bg-rose-900 text-rose-200"}`}>{sc.singer}</span>;
}
const VOICE_STYLE: Record<string, string> = { her: "bg-rose-900 text-rose-200", him: "bg-sky-900 text-sky-200", "her>him": "bg-violet-900 text-violet-200", "him>her": "bg-violet-900 text-violet-200", unknown: "bg-neutral-800 text-neutral-300" };
const VOICE_WORDS: Record<string, string> = { her: "her", him: "him", "her>him": "shared: shows her", "him>her": "shared: shows her", unknown: "not measured" };
const needsPicture = (sc: Scene) => envOnly(sc) && sc.matches_location_still === false && !sc.opening_still_id;
const pictureUse = (sc: Scene) => envOnly(sc)
  ? (sc.matches_location_still === false ? "no singer: the render starts on this picture (required)" : "no singer: the render starts on this picture (else on the place's picture)")
  : sc.rail && sc.rail.end_still_id === sc.end_still_id
    ? "with the singer: she renders from her two pictures and the place's picture; the move's rail pins the first frame to this one"
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

export function MusicVideoForm({ library, onDone, projectName }: { cfg: Config; busy: boolean; library: LibItem[]; projectName?: string } & RunCtx) {
  const d = useDrafts("mv");
  const drafts = useDraftCtx();
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
  // the recipe (turbo, steps, sampler, scheduler) is per scene: only the scenes changed from their
  // default are stored, keyed "<script>#<scene>"; the default is base on a camera move, turbo when static
  const [recipes, setRecipes] = d.use<Record<string, Partial<Recipe>>>("sceneRecipes", {});
  const [width, setWidth] = d.use("width", 1344);
  const [height, setHeight] = d.use("height", 768);
  const [acceptFlags, setAcceptFlags] = d.use("acceptFlags", false);
  const [soloLock, setSoloLock] = d.use("soloLock", false);

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
  // the open project's song and its tracks in the Library (by song name, through mastered/naturalized copies)
  const byId = useMemo(() => new Map(library.map((i) => [i.id, i])), [library]);
  const projSong = projectSong(drafts.get, projectName);
  const projKeys = [projSong.title, projectName || ""].map(songKey).filter(Boolean);
  const projTracks = projKeys.length ? audios.filter((a) => projKeys.includes(songKey(trackSongName(a, byId)))) : [];
  const originMode = (id: string) => { const it = byId.get(id); return it ? trackOrigin(it, byId).mode : undefined; };
  // a track picked from the project: the writer starts from the project's song, not the track's facts
  const pickFromProject = (id: string) => {
    const ps = projectSong(drafts.get, projectName, originMode(id));
    setAudioId(id); setScriptName(""); setSel(null); setDry({});
    if (ps.title) drafts.set("mv", "wTitle", ps.title);
    if (ps.style) drafts.set("mv", "wStyle", ps.style);
    if (ps.lyrics.trim()) drafts.set("mv", "wLyrics", ps.lyrics);
    drafts.set("mv", "wTrack", id);
  };
  const recipeKey = (idx: number) => `${scriptName}#${idx}`;
  const recipeOf = (sc: Scene): Recipe => ({ ...sceneDefaultRecipe(sc), ...(recipes[recipeKey(sc.index)] || {}) });
  const setRecipe = (sc: Scene, patch: Partial<Recipe> | null) => setRecipes((r) => {
    const n = { ...r };
    if (!patch) { delete n[recipeKey(sc.index)]; return n; }
    // a field set back to the scene's default is dropped, so the scene reads as default again
    const def = sceneDefaultRecipe(sc);
    const cur = { ...(n[recipeKey(sc.index)] || {}), ...patch };
    (Object.keys(cur) as (keyof Recipe)[]).forEach((k) => { if (cur[k] === def[k]) delete cur[k]; });
    if (Object.keys(cur).length) n[recipeKey(sc.index)] = cur; else delete n[recipeKey(sc.index)];
    return n;
  });

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
  const setRef = async (kind: RefKind, id: string) => {
    if (!scriptName && (kind === "him_face" || kind === "him_body")) { setErr("his pictures belong to a script: open one first"); return; }
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
  const rendering = useMemo(() => Object.values(takes).some((l) => l.some((t) => t.status !== "done" || Object.values(upsOf(t)).some((u) => upscaleBusy(u)))), [takes]);
  useEffect(() => {
    if (!rendering) return;
    const h = window.setInterval(refreshTakes, 5000);
    return () => window.clearInterval(h);
  }, [rendering, refreshTakes]);
  // takes and scripts made outside this tab (another window, the API, a scene rewrite) only appear
  // on a re-read: re-read both whenever the window comes back into focus, and every 30 s while visible
  useEffect(() => {
    const onBack = () => { if (document.visibilityState === "visible") { refreshTakes(); loadScripts(); } };
    window.addEventListener("focus", onBack);
    document.addEventListener("visibilitychange", onBack);
    const h = window.setInterval(onBack, 30000);
    return () => {
      window.removeEventListener("focus", onBack);
      document.removeEventListener("visibilitychange", onBack);
      window.clearInterval(h);
    };
  }, [refreshTakes, loadScripts]);

  const sceneTakes = (idx: number) => takes[String(idx)] || [];
  const doneTakes = (idx: number) => sceneTakes(idx).filter((t) => t.status === "done");

  // ---------- filing pictures ----------
  const fileStill = async (idx: number, file: File, slot: "opening" | "after" | "empty" = "opening", ofTake?: string) => {
    setErr(""); setBusyMsg(`filing the picture for scene ${idx}...`);
    try {
      const fd = new FormData();
      fd.append("file", file, file.name || "still.png");
      fd.append("script", scriptName);
      fd.append("scene", String(idx));
      fd.append("slot", slot);
      if (ofTake) fd.append("of_take", ofTake);
      await api.mvSceneStill(fd);
      reloadScript();
    } catch (e) { setErr(String((e as Error).message || e)); }
    finally { setBusyMsg(""); }
  };
  const fileRef = async (kind: RefKind, file: File) => {
    setErr(""); setBusyMsg(`filing the ${kind.replace("_", " ")} reference...`);
    try {
      const fd = new FormData();
      fd.append("file", file, file.name || `${kind}.png`);
      fd.append("label", `${script?.title || "music video"} - ${kind.startsWith("him_") ? `male singer ${kind.slice(4)}` : `lead ${kind}`} reference`);
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
    else if (target.kind === "after") fileStill(target.idx, f, "after");
    else if (target.kind === "empty") fileStill(target.idx, f, "empty", target.take);
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
  const next: { step: number; text: string } = !audioId ? { step: 1, text: projTracks.length ? `Pick the song this video is for: the open project "${projectName}" has ${projTracks.length} track${projTracks.length === 1 ? "" : "s"} of it (on the right).` : "Pick the song this video is for." }
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
          project={projectName || projSong.title ? { name: projectName || "", title: projSong.title, hasLyrics: !!projSong.lyrics.trim(), tracks: projTracks } : null}
          onPickProject={pickFromProject}
          onPick={(id) => { setAudioId(id); setScriptName(""); setSel(null); setDry({}); }}
          onImported={(id) => { onDone(); setAudioId(id); setScriptName(""); setSel(null); }}
          onNext={() => setStep(2)} />
      )}

      {step === 2 && (
        <ScriptStep audioId={audioId} scripts={scripts} scriptName={scriptName} script={script} allScripts={allScripts} setAllScripts={setAllScripts}
          onOpen={(name) => { setScriptName(name); setSel(null); setDry({}); }}
          onWritten={async (name) => { await loadScripts(); setScriptName(name); setSel(null); setDry({}); }}
          projectName={projectName} trackMode={audioId ? originMode(audioId) : undefined}
          onNext={() => setStep(3)} />
      )}

      {step === 3 && script && (
        <PicturesStep script={script} library={library} faceId={face} bodyId={body} setFaceId={(id) => setRef("face", id)} setBodyId={(id) => setRef("body", id)} setHim={(k, id) => setRef(k, id)}
          target={target} setTarget={setTarget} fileStill={fileStill} fileRef={fileRef} setErr={setErr}
          places={places} placeRows={placeRows} placePic={placePic} filePlace={filePlace} linkPlace={linkPlace} />
      )}

      {step === 4 && script && (
        <TakesStep script={script} scriptName={scriptName} takes={takes} sel={sel} setSel={setSel}
          faceId={face} bodyId={body} seed={seed} setSeed={setSeed} soloLock={soloLock} setSoloLock={setSoloLock}
          recipeOf={recipeOf} setRecipe={setRecipe} recipeChanged={(sc) => !!recipes[recipeKey(sc.index)]}
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

type ProjectTracks = { name: string; title: string; hasLyrics: boolean; tracks: LibItem[] };
function SongStep({ audioId, audios, project, onPickProject, onPick, onImported, onNext }: { audioId: string; audios: LibItem[]; project: ProjectTracks | null; onPickProject: (id: string) => void; onPick: (id: string) => void; onImported: (id: string) => void; onNext: () => void }) {
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
        {project && (
          <Card title={<>From the open project{project.name ? <span className="font-normal text-[var(--color-muted)]"> · {project.name}</span> : ""}</>}
            sub={`"${project.title || project.name}": picking a track here also fills the script writer with the project's title, style and lyrics${project.hasLyrics ? "" : " (it has no lyrics yet)"}`}>
            {project.tracks.length ? (
              <div className="max-h-56 overflow-auto rounded-lg border border-[var(--color-line)]">
                {[...project.tracks].sort((a, b) => b.created - a.created).map((a) => (
                  <button key={a.id} onClick={() => onPickProject(a.id)}
                    className={`flex w-full items-center gap-3 border-b border-[var(--color-line)] px-3 py-1.5 text-left text-xs hover:bg-[var(--color-panel2)] ${a.id === audioId ? "bg-[#2a1c19] text-[var(--color-accent)]" : ""}`}>
                    <span className="flex-1 truncate">{audioLabel(a)}</span>
                    <span className="text-[10px] text-[var(--color-muted)]">{a.mode} · {when(a.created)}</span>
                  </button>
                ))}
              </div>
            ) : <div className="text-xs text-[var(--color-muted)]">No Library track is named "{project.title || project.name}" yet: pick it on the left, or import it below (with that title).</div>}
          </Card>
        )}
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
function ScriptStep({ audioId, scripts, scriptName, script, allScripts, setAllScripts, onOpen, onWritten, onNext, projectName, trackMode }: {
  audioId: string; scripts: ScriptRow[]; scriptName: string; script: Script | null; allScripts: boolean; setAllScripts: (v: boolean) => void;
  onOpen: (name: string) => void; onWritten: (name: string) => Promise<void>; onNext: () => void; projectName?: string; trackMode?: string;
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
        loaded={script && script.audio_id === audioId ? { name: scriptName, script } : null} onWritten={onWritten}
        projectName={projectName} trackMode={trackMode} />}
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

function PicturesStep({ script, library, faceId, bodyId, setFaceId, setBodyId, setHim, target, setTarget, fileStill, fileRef, setErr, places, placeRows, placePic, filePlace, linkPlace }: {
  script: Script; library: LibItem[]; faceId: string; bodyId: string; setFaceId: (v: string) => void; setBodyId: (v: string) => void; setHim: (k: "him_face" | "him_body", id: string) => void;
  target: Target; setTarget: (t: Target) => void; fileStill: (idx: number, f: File) => void; fileRef: (k: RefKind, f: File) => void; setErr: (s: string) => void;
  places: Place[]; placeRows: PlaceRow[]; placePic: (pl: Place) => string | null; filePlace: (id: string, f: File) => void; linkPlace: (id: string, stillId: string) => void;
}) {
  const [picker, setPicker] = useState<RefKind | null>(null);
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
        {picker && (picker === "face" || picker === "body") && <StillPicker library={library} onClose={() => setPicker(null)} onPick={(id) => { (picker === "face" ? setFaceId : setBodyId)(id); setPicker(null); }} />}
      </Card>

      {script.duet && (
        <Card title="The male singer: two reference pictures" sub={`used for every scene that shows him (${script.scenes.filter((sc) => sc.singer === "him").length} scenes). Same as hers: click a box and paste, drop an image, or pick one from the Library.`}>
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            {(["him_face", "him_body"] as const).map((k) => {
              const id = (script.ref_stills || {})[k] || "";
              return (
                <div key={k} className="space-y-1">
                  <div className="text-xs font-semibold">{k === "him_face" ? "His face (close-up portrait)" : "His full body (head to feet)"}</div>
                  <ImageSlot src={id ? `/api/media/${id}` : null} aspect="aspect-[3/4]" armed={isT({ kind: k })} onArm={() => setTarget({ kind: k })}
                    onFile={(f) => fileRef(k, f)} empty={<><div>no {k === "him_face" ? "face" : "full-body"} picture of him</div><div>click, then Cmd+V, or drop an image</div></>}>
                    <GhostButton onClick={() => setPicker(picker === k ? null : k)}>from the Library</GhostButton>
                  </ImageSlot>
                </div>
              );
            })}
            <div className="text-xs text-[var(--color-muted)] xl:col-span-2 space-y-2">
              {script.duet.refs2?.face && <div><b className="text-[var(--color-ink)]">The script describes his face picture as:</b> {script.duet.refs2.face}</div>}
              {script.duet.refs2?.body && <div><b className="text-[var(--color-ink)]">and his full-body picture as:</b> {script.duet.refs2.body}</div>}
              {script.wardrobe2 && <div><b className="text-[var(--color-ink)]">His wardrobe:</b> {script.wardrobe2}</div>}
            </div>
          </div>
          {picker && (picker === "him_face" || picker === "him_body") && <StillPicker library={library} onClose={() => setPicker(null)} onPick={(id) => { setHim(picker, id); setPicker(null); }} />}
        </Card>
      )}

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
                {script.duet && <SingerBadge sc={sc} />}
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
// the ChatGPT edit that turns a scene's opening still into the state its action ends in (s.20.54):
// built from the scene's own action text, so it serves any scene where someone changes state
const afterPrompt = (sc: Scene) =>
  "Edit this picture. Keep the camera position, framing, lens, room, light, weather and every person's place, " +
  "clothes and identity exactly as they are. Change only what has happened by the END of this moment, shown " +
  `finished and held still, not in motion: ${(sc.action || "").trim()}`;

function EndOfMove({ scene, startPic, scriptName, width, height, busy, setErr, reload, armed, onArm, onFile }: {
  scene: Scene; startPic: string | null; scriptName: string; width: number; height: number; busy: boolean;
  setErr: (s: string) => void; reload: () => void; armed: boolean; onArm: () => void; onFile: (f: File) => void;
}) {
  const [zoomText, setZoomText] = useState("1.5");      // typed freely, clamped where it is used
  const zoom = Math.min(4, Math.max(1.05, Number(zoomText) || 1.5));
  const [c, setC] = useState({ x: 0.5, y: 0.5 });
  const [railOn, setRailOn] = useState(false);          // opt-in: pin the in-between views too
  const [railAfterText, setRailAfterText] = useState("0"); // no pin before this (an early action stays free)
  const [copiedAfter, setCopiedAfter] = useState(false);
  const hasAfter = !!scene.after_still_id;
  const railAfter = Math.max(0, Number(railAfterText) || 0);
  const [working, setWorking] = useState(false);
  const rail = scene.rail && scene.rail.end_still_id === scene.end_still_id ? scene.rail.stills : [];
  useEffect(() => { setC({ x: 0.5, y: 0.5 }); }, [scene.index]);
  if (!startPic && !scene.end_still_id) return null;
  // the crop box as the backend clamps it, in fractions of the picture
  const fw = 1 / zoom, fh = 1 / zoom;
  const x0 = Math.min(Math.max(c.x - fw / 2, 0), 1 - fw), y0 = Math.min(Math.max(c.y - fh / 2, 0), 1 - fh);
  const make = async () => {
    setErr(""); setWorking(true);
    try { await api.mvPushEndStill({ script: scriptName, scene: scene.index, zoom, cx: x0 + fw / 2, cy: y0 + fh / 2, width, height, rail: railOn ? 1.5 : 0,
      rail_after: railOn ? railAfter : 0 }); reload(); }
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
              {railOn && (
                <label className="flex items-center gap-1" title={hasAfter ? "the second the action is done by: no pin before it, and every pin after it comes from the after-the-action picture" : "no pin before this time, so an early action (a reach, a turn) is not held at the opening pose"}>
                  {hasAfter ? "action done by" : "from"} <input type="number" min={0} step={0.5} value={railAfterText} className={`${inp} w-16`}
                    onChange={(e) => setRailAfterText(e.target.value)} /> s</label>
              )}
              <GhostButton onClick={make} disabled={busy || working || (railOn && hasAfter && railAfter <= 0)}>{working ? "making..." : scene.end_still_id ? "make it again" : "make the end picture"}</GhostButton>
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
      {startPic && hasAfter && (   /* shown only once a scene has one; adding one moves to step 3 for scenes the writer marks (s.20.54) */
        <details className="mt-3 text-xs" open={hasAfter}>
          <summary className="cursor-pointer text-[var(--color-muted)]">after the action (optional): when someone ends somewhere other than where they start, the end and the pins come from this picture</summary>
          <div className="mt-2 grid gap-3 md:grid-cols-2">
            <ImageSlot src={scene.after_still_id ? `/api/media/${scene.after_still_id}` : null} armed={armed} onArm={onArm} onFile={onFile}
              empty={<><div className="font-semibold">no after picture</div><div>1. copy the edit prompt  2. in ChatGPT, edit this scene's opening picture with it  3. click here + Cmd+V, or drop it</div></>}>
              <div className="flex flex-wrap gap-2">
                <GhostButton onClick={async () => { try { await navigator.clipboard.writeText(afterPrompt(scene)); setCopiedAfter(true); window.setTimeout(() => setCopiedAfter(false), 1500); } catch { setErr("the browser blocked the clipboard: copy the prompt by hand"); } }}
                  disabled={!scene.action}>{copiedAfter ? "copied" : "copy edit prompt"}</GhostButton>
                {hasAfter && <GhostButton onClick={async () => { try { await api.mvSceneAfterStill({ script: scriptName, scene: scene.index, still_id: null }); reload(); } catch (e) { setErr(String((e as Error).message || e)); } }} disabled={busy}>remove</GhostButton>}
              </div>
            </ImageSlot>
            <div className="text-[11px] text-[var(--color-muted)]">
              <div className="mb-1 whitespace-pre-wrap rounded border border-[var(--color-line)] p-2">{afterPrompt(scene)}</div>
              {hasAfter ? "Make the end picture again to use it. With the path pinned, set when the action is done by: no pin goes before that, the pins after it and the end come from this picture, and the take still starts on the opening picture." : "Same framing as the opening picture (an edit of it), so the crops line up."}
            </div>
          </div>
        </details>
      )}
    </Card>
  );
}

// ---------- add a camera move to a take: three ways, one panel (s.20.58-20.61) ----------
// the panel's own move names; each way takes them in its own vocabulary
const CAM_MOVES: [string, string][] = [["arc_left", "arc toward frame left"], ["arc_right", "arc toward frame right"],
  ["push_in", "push in"], ["pull_out", "pull out"], ["side_left", "move sideways, left"], ["side_right", "move sideways, right"],
  ["rise", "rise"], ["lower", "lower"]];
const TO_2D: Record<string, string> = { side_left: "pan_left", side_right: "pan_right", rise: "tilt_up", lower: "tilt_down" };
const TO_3D: Record<string, string> = { side_left: "truck_left", side_right: "truck_right", rise: "pedestal_up", lower: "pedestal_down" };
const FROM_ANY: Record<string, string> = { pan_left: "side_left", pan_right: "side_right", truck_left: "side_left", truck_right: "side_right",
  tilt_up: "rise", tilt_down: "lower", pedestal_up: "rise", pedestal_down: "lower" };
const moveLabel = (m: unknown) => (CAM_MOVES.find((x) => x[0] === (FROM_ANY[String(m)] || String(m))) || ["", String(m || "")])[1];
type CamWay = "move2d" | "roomplate" | "meridian";
type CamDry = { move: string; amount: string; camera?: string; flags: string[]; minutes?: number; frames?: number; length?: number; seed?: number; empty_room_id?: string | null; empty_prompt?: string };

// a room-pinned job (s.20.61) before its take exists: 5 steps, named by the backend's stage text
type RpJob = { id: string; stage: string; status: string; error?: string | null; take_job_id?: string | null };
const RP_STEPS: [string, string][] = [["uploading", "uploading"], ["moving the empty room", "moving the empty room"],
  ["moving the take", "moving the take, finding the people"], ["building", "building the room plate and mask"],
  ["render", "rendering"]];
const rpStep = (stage: string) => {
  const i = RP_STEPS.findIndex(([k]) => (stage || "").includes(k));
  return i < 0 ? (stage.startsWith("sent") ? 5 : 1) : i + 1;
};
function RoomPinnedProgress({ job }: { job: RpJob | null | undefined }) {
  if (!job) return null;
  if (job.status === "error") return <div className="rounded border border-red-800 p-2 text-xs text-red-400">room-pinned move failed: {job.error}</div>;
  const k = rpStep(job.stage);
  const sent = job.status === "done";
  return (
    <div className="rounded border border-[var(--color-accent)] p-2 text-xs">
      <Bar label={sent ? "Room-pinned move sent: the new take is rendering below (about 10 min, it has its own bar)"
        : `Room-pinned move, step ${k} of 5: ${job.stage}`} frac={sent ? 1 : (k - 0.5) / 5} />
    </div>
  );
}

// the render sampler: the recipe's own first, then the alternatives the research supports for H3
// (docs/MV_H3_REBUILD_PLAN.md s.20.62); [value, label, what it does]
const DEFAULT_SAMPLER = "res_multistep";
const SAMPLERS: [string, string, string][] = [
  ["res_multistep", "res_multistep (current default)", "Comfy's official H3 template and our recipe, turbo and base"],
  ["euler", "euler", "the turbo LoRA authors' own sampler (ModelTC); try it with turbo on"],
  ["er_sde", "er_sde", "community pick for the turbo LoRA (Kijai); adds fresh noise each step"],
  ["sa_solver", "sa_solver", "community pick for the turbo LoRA (Kijai)"],
];
const DEFAULT_SCHEDULER = "simple";
const SCHEDULERS: [string, string][] = [
  ["simple", "simple (current default)"], ["beta", "beta"], ["normal", "normal"],
];
const stepsLabel = (turbo: boolean, steps: string) => steps.trim() || (turbo ? "8" : "20");
// one scene's recipe; steps "" = the recipe's own (8 turbo, 20 base). anchor (opt-in, off by
// default): a scene she is in starts on its own opening picture (backend anchor_first; the face
// and body pictures still carry her identity). Environment scenes already start on theirs.
type Recipe = { turbo: boolean; steps: string; sampler: string; scheduler: string; anchor: boolean };
// user 2026-11-01: turbo warped the room on camera moves (Never Yours booth scenes) and base cleared
// every one first try, so a camera move defaults to base and only a static scene to turbo
// (the backend's scene_default_turbo is the same rule)
const sceneMoves = (sc: Scene) => (sc.camera || "static").trim() !== "static";
const sceneDefaultRecipe = (sc: Scene): Recipe => ({ turbo: !sceneMoves(sc), steps: "", sampler: DEFAULT_SAMPLER, scheduler: DEFAULT_SCHEDULER, anchor: false });
const recipeText = (r: Recipe) => `${r.turbo ? "turbo" : "base"}, ${stepsLabel(r.turbo, r.steps)} steps, ${r.sampler}/${r.scheduler}${r.anchor ? ", starts on its picture" : ""}`;

const MERIDIAN_MAX_FRAMES = 243;   // backend/mv_meridian.py MERIDIAN_LENGTHS[-1]

function CameraMovePanel({ scriptName, scene, take, busy, setErr, setBusyMsg, refreshTakes, onClose, render, armed, onArm, onFile, roomJob, onRoomJob }: {
  scriptName: string; scene: Scene; take: Take; busy: boolean; setErr: (s: string) => void; setBusyMsg: (s: string) => void;
  refreshTakes: () => Promise<void>; onClose: () => void; render: Record<string, unknown>; armed: boolean; onArm: () => void; onFile: (f: File) => void;
  roomJob: RpJob | null; onRoomJob: (id: string) => void;
}) {
  const [way, setWay] = useState<CamWay>("roomplate");
  const [move, setMove] = useState("");
  const [amount, setAmount] = useState("small");
  const [dry, setDry] = useState<CamDry | null>(null);
  const [copied, setCopied] = useState(false);
  const [guess, setGuess] = useState("");        // the assumed side, kept until the move is chosen by hand
  const [checkErr, setCheckErr] = useState("");  // the check's refusal, shown in the panel (not only at the top of the tab)
  const fromStill = take.params.variant === "still" || (scene.camera || "static").trim() === "static";
  const wire = (m: string) => (way === "move2d" ? TO_2D[m] : TO_3D[m]) || m;
  const base = { script: scriptName, scene: scene.index, job_id: take.id };

  // the check: the chosen way's dry run (it also guesses the move from the scene's camera line)
  useEffect(() => {
    let live = true;
    (async () => {
      setErr(""); setCheckErr("");
      try {
        const q = { ...base, move: move ? wire(move) : undefined, amount, dry_run: true };
        const r = (way === "move2d" ? await api.mvMove2dTake(q)
          : way === "meridian" ? await api.mvMeridianTake(q)
          : await api.mvRoomplateTake({ ...render, ...q })) as CamDry;
        if (!live) return;
        setDry(r);
        if (!move) {
          setMove(FROM_ANY[r.move] || r.move);
          setGuess((r.flags || []).find((f) => f.includes("assumed")) || "");
        }
      } catch (e) { if (live) setCheckErr(String((e as Error).message || e)); }
    })();
    return () => { live = false; };
  }, [way, move, amount, take.id, scene.empty_room_id]);

  const make = async () => {
    if (!dry) return;
    const m = wire(move);
    setErr("");
    try {
      if (way === "move2d") {
        setBusyMsg("making the 2D move (Mac, seconds)...");
        await api.mvMove2dTake({ ...base, move: m, amount });
        await refreshTakes();
        onClose();
      } else if (way === "meridian") {
        if (!window.confirm(`Re-shoot this take with Meridian on the box now? (${moveLabel(move)}, ${amount}, about ${Math.round(dry.minutes || 13)} min)`)) return;
        setBusyMsg("sending the re-shoot to the box...");
        await api.mvMeridianTake({ ...base, move: m, amount, seed: dry.seed });
        await refreshTakes();
        onClose();
      } else {
        if (!window.confirm(`Render this scene with the room pinned on the box now? (${moveLabel(move)}, ${amount}, about ${Math.round(dry.minutes || 13)} min)`)) return;
        const { dry_run: _d, variant: _v, script: _s, scene: _c, ...settings } = render;
        const r = await api.mvRoomplateTake({ ...settings, ...base, move: m, amount }) as { job: string };
        onRoomJob(r.job);
      }
    } catch (e) { setErr(String((e as Error).message || e)); }
    finally { setBusyMsg(""); }
  };

  const WAYS: { id: CamWay; title: string; what: string; time: string }[] = [
    { id: "move2d", title: "2D move", time: "seconds, no GPU",
      what: "Moves a crop of this take. Her lips, the room and the song stay exactly as rendered. No depth: the background moves with her." },
    { id: "roomplate", title: "Room pinned (best for singing)", time: `about ${Math.round((way === "roomplate" && dry?.minutes) || 13)} min on the box`,
      what: "Real depth. The room is your empty-room picture, held still, so it cannot change; H3 draws the people again with the song, so lips are as good as a normal take. Needs one picture (below)." },
    { id: "meridian", title: "Meridian", time: "about 13 min on the box, takes up to 10 s only",
      what: "Real depth, no picture needed. Redraws the whole take from a small copy: fine when the face is still, but lips can lag on lively singing, and windows and signs get repainted." },
  ];
  const emptyFromOther = way === "roomplate" && !!scene.empty_room_id && !!scene.empty_room_of && scene.empty_room_of !== take.id;
  const needPicture = way === "roomplate" && (!scene.empty_room_id || emptyFromOther);
  const running = roomJob?.status === "running";
  // Meridian's text embeddings stop at 243 frames (10.1 s); longer takes cannot use it
  const takeFrames = Number(take.params.frames) || 0;
  const meridianTooLong = takeFrames > MERIDIAN_MAX_FRAMES;

  return (
    <div className="space-y-3 rounded border border-[var(--color-line)] p-3 text-[11px]">
      <div>
        <div className="text-sm font-semibold">Add a camera move</div>
        <div className="text-[var(--color-muted)]">makes a new take of this scene; this take stays as it is.</div>
      </div>
      {!fromStill && (
        <div className="rounded border border-amber-700 p-2 text-amber-300">
          This take already has the scene's own move, and a broken move stays broken. Use the "Camera move broke the room?" card
          above: step 1 renders the still version, step 2 opens this panel on it.
        </div>
      )}
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-semibold">1 · The move</span>
        <select className={`${inp} w-auto`} value={move} onChange={(e) => { setMove(e.target.value); setGuess(""); }}>
          {CAM_MOVES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
        </select>
        <select className={`${inp} w-auto`} value={amount} onChange={(e) => setAmount(e.target.value)}>
          <option value="small">small (a slow performance move)</option><option value="large">large</option>
        </select>
        {dry?.camera ? <span className="text-[var(--color-muted)]">the scene asks: {dry.camera}</span> : null}
      </div>
      {guess && <div className="text-amber-300">{guess}: check the side before you make it</div>}
      <div className="space-y-1">
        <div className="font-semibold">2 · How</div>
        {WAYS.map((w) => (
          <label key={w.id} className={`flex gap-2 rounded border p-2 ${w.id === "meridian" && meridianTooLong ? "cursor-not-allowed opacity-50" : "cursor-pointer"} ${way === w.id ? "border-[var(--color-accent)]" : "border-[var(--color-line)]"}`}>
            <input type="radio" name={`way_${take.id}`} checked={way === w.id} disabled={w.id === "meridian" && meridianTooLong}
              onChange={() => { setWay(w.id); setDry(null); setCheckErr(""); }} />
            <div><div className="font-semibold">{w.title} <span className="font-normal text-[var(--color-muted)]">· {w.time}</span></div>
              <div className="text-[var(--color-muted)]">{w.what}</div>
              {w.id === "meridian" && meridianTooLong
                ? <div className="text-amber-300">not for this take: it is {(takeFrames / 24).toFixed(1)} s, Meridian stops at 10.1 s. Use room pinned or the 2D move.</div> : null}</div>
          </label>
        ))}
      </div>
      {way === "roomplate" && (
        <div className="space-y-2 rounded border border-[var(--color-line)] p-2">
          <div className="font-semibold">3 · The empty-room picture{scene.empty_room_id && !emptyFromOther ? " (done)" : ""}</div>
          <div className="text-[var(--color-muted)]">this take's first frame with the people removed, made in ChatGPT:</div>
          <ol className="list-decimal space-y-1 pl-4">
            <li><a className="text-[var(--color-accent)] underline" href={`/api/mv/take_frame/${take.id}`} download>download this take's first frame</a></li>
            <li>in ChatGPT, give it the frame with this prompt{" "}
              <GhostButton onClick={async () => { try { await navigator.clipboard.writeText(dry?.empty_prompt || ""); setCopied(true); window.setTimeout(() => setCopied(false), 1500); } catch { setErr("the browser blocked the clipboard: copy the prompt by hand"); } }}
                disabled={!dry?.empty_prompt}>{copied ? "copied" : "copy the prompt"}</GhostButton>
              <div className="mt-1 whitespace-pre-wrap rounded border border-[var(--color-line)] p-2">{dry?.empty_prompt || "..."}</div></li>
            <li>paste the result here (click the box, then Cmd+V) or drop it on the box</li>
          </ol>
          <ImageSlot src={scene.empty_room_id && !emptyFromOther ? `/api/media/${scene.empty_room_id}` : null} armed={armed} onArm={onArm} onFile={onFile}
            empty={<><div className="font-semibold">{emptyFromOther ? "the empty room on file was made from another take" : "no empty-room picture yet"}</div><div>click here, then Cmd+V; or drop the picture</div></>} />
        </div>
      )}
      {dry?.flags.length ? <ul className="list-disc pl-4 text-amber-300">{dry.flags.map((f, i) => <li key={i}>{f}</li>)}</ul> : null}
      <div className="flex flex-wrap items-center gap-2">
        <Action onClick={make} disabled={busy || running || !dry || !move || !fromStill || needPicture}>
          {way === "move2d" ? "make the 2D move" : way === "meridian" ? "re-shoot with Meridian" : "render with the room pinned"}
        </Action>
        <GhostButton onClick={onClose}>close</GhostButton>
        {!fromStill
          ? <span className="text-amber-300">not on this take: open it on the "still version" take (step 2 of the card above)</span>
          : needPicture ? <span className="text-amber-300">add the empty-room picture first (step 3)</span>
          : checkErr ? <span className="text-red-400">can't make this: {checkErr}</span>
          : !dry ? <span className="text-[var(--color-muted)]">checking...</span> : null}
      </div>
      <RoomPinnedProgress job={roomJob} />
    </div>
  );
}

// ---------- 4 Scenes and takes: render each scene, choose the take that plays ----------
function TakesStep(p: {
  script: Script; scriptName: string; takes: Record<string, Take[]>; sel: number | null; setSel: (n: number | null) => void;
  faceId: string; bodyId: string; seed: string; setSeed: (v: string) => void;
  soloLock: boolean; setSoloLock: (v: boolean) => void;
  recipeOf: (sc: Scene) => Recipe; setRecipe: (sc: Scene, patch: Partial<Recipe> | null) => void; recipeChanged: (sc: Scene) => boolean;
  width: number; setWidth: (v: number) => void; height: number; setHeight: (v: number) => void; acceptFlags: boolean; setAcceptFlags: (v: boolean) => void;
  dry: Record<number, Dry>; setDry: (f: (d: Record<number, Dry>) => Record<number, Dry>) => void; busyMsg: string; setBusyMsg: (s: string) => void; setErr: (s: string) => void;
  refreshTakes: () => Promise<void>; reloadScript: () => void; target: Target; setTarget: (t: Target) => void; fileStill: (idx: number, f: File, slot?: "opening" | "after" | "empty", ofTake?: string) => void; goPictures: () => void;
  startPicOf: (sc: Scene) => string | null;
}) {
  const { script, scriptName, takes, dry, busyMsg, setBusyMsg, setErr } = p;
  const [filter, setFilter] = useState<"all" | "todo">("all");
  const scene = script.scenes[p.sel ?? 0] || script.scenes[0];
  const list = (idx: number) => takes[String(idx)] || [];
  const moving = sceneMoves(scene);
  const rc = p.recipeOf(scene);
  const recipeAt = (idx: number) => { const sc = script.scenes.find((x) => x.index === idx); return sc ? p.recipeOf(sc) : rc; };
  // the "add a camera move" panel, open on one take at a time
  const [camFor, setCamFor] = useState<string | null>(null);
  useEffect(() => { setCamFor(null); }, [scene.index]);
  // the scene's room-pinned job, remembered in the browser so a reload or a closed panel keeps it
  const rpKey = `mv_roomplate_${scriptName}_${scene.index}`;
  const [rpJob, setRpJob] = useState<RpJob | null>(null);
  useEffect(() => {
    let id: string | null = null;
    try { id = window.localStorage.getItem(rpKey); } catch { id = null; }
    setRpJob(id ? { id, stage: "checking", status: "running" } : null);
  }, [rpKey]);
  const startRpJob = (id: string) => {
    try { window.localStorage.setItem(rpKey, id); } catch { /* the bar still shows until a reload */ }
    setRpJob({ id, stage: "starting", status: "running" });
  };
  useEffect(() => {
    if (!rpJob || rpJob.status !== "running") return;
    const poll = async () => {
      try {
        const r = await api.mvRoomplateJob(rpJob.id) as RpJob;
        setRpJob({ ...r, id: rpJob.id });
        if (r.status !== "running") await p.refreshTakes();
      } catch {
        // the backend no longer knows it (restarted): forget it
        try { window.localStorage.removeItem(rpKey); } catch { /* nothing to clear */ }
        setRpJob(null);
      }
    };
    poll();
    const t = window.setInterval(poll, 4000);
    return () => window.clearInterval(t);
  }, [rpJob?.id, rpJob?.status]);
  useEffect(() => {
    // once its take is in the list, the take's own bar takes over
    if (rpJob?.status === "done" && rpJob.take_job_id && list(scene.index).some((t) => t.id === rpJob.take_job_id)) {
      try { window.localStorage.removeItem(rpKey); } catch { /* nothing to clear */ }
      setRpJob(null);
    }
  }, [takes, rpJob?.status, rpJob?.take_job_id]);

  const commonSig = () => `${p.faceId}|${p.bodyId}|${p.seed}|${p.width}|${p.height}|${p.soloLock && p.script.duet ? "solo" : ""}`;
  const sig = (idx: number = scene.index) => `${commonSig()}|${JSON.stringify(recipeAt(idx))}`;
  // turbo always goes out; steps, sampler and scheduler only when they differ from the recipe's own
  const recipeKnobs = (b: Record<string, unknown>, r: Recipe) => {
    b.turbo = r.turbo;
    const n = Number(r.steps);
    if (r.steps.trim() && Number.isInteger(n) && n > 0) b.steps = n;
    if (r.sampler && r.sampler !== DEFAULT_SAMPLER) b.sampler = r.sampler;
    if (r.scheduler && r.scheduler !== DEFAULT_SCHEDULER) b.scheduler = r.scheduler;
    if (r.anchor) b.anchor_first = true;
    return b;
  };
  const renderBody = (idx: number, dryRun: boolean, pinSeed?: number) => {
    const body: Record<string, unknown> = {
      script: scriptName, scene: idx, ref_ids: [p.faceId, p.bodyId].map((x) => x.trim()).filter(Boolean),
      attention: ATTENTION, width: Number(p.width), height: Number(p.height), dry_run: dryRun,
    };
    recipeKnobs(body, recipeAt(idx));
    if (p.soloLock && p.script.duet) body.solo_lock = true;
    const s = pinSeed ?? (p.seed.trim() ? Number(p.seed) : undefined);
    if (s !== undefined && !Number.isNaN(s)) body.seed = s;
    return body;
  };
  const doDry = async (idx: number) => {
    setErr(""); setBusyMsg("checking the render settings (no GPU)...");
    try {
      const r = await api.mvRenderScene(renderBody(idx, true));
      p.setDry((d) => ({ ...d, [idx]: { ...(r as Dry), sig: sig(idx) } }));
    } catch (e) { setErr(String((e as Error).message || e)); }
    finally { setBusyMsg(""); }
  };
  // step 1 of "fix a broken camera move": the scene again with a still camera (its still text is
  // written first when the scene has none), checked, confirmed and sent in one go
  const doStillRender = async (idx: number) => {
    setErr("");
    try {
      if (!script.scenes.find((x) => x.index === idx)?.still_clips?.length) {
        setBusyMsg("writing the still version of the scene (no GPU, a minute or two)...");
        await api.mvStillVariant({ script: scriptName, scene: idx });
        p.reloadScript();
      }
      setBusyMsg("checking the still version (no GPU)...");
      const dr = await api.mvRenderScene({ ...renderBody(idx, true), variant: "still" }) as Dry;
      setBusyMsg("");
      const warn = dr.flags.length ? `\n\nWarnings:\n- ${dr.flags.join("\n- ")}` : "";
      if (!window.confirm(`Render the still version of scene ${idx} on the box now? (about 10 min, seed ${dr.settings?.seed})${warn}`)) return;
      setBusyMsg("sending the still version to the box (30-60 s)...");
      await api.mvRenderScene({ ...renderBody(idx, false, Number(dr.settings?.seed)), variant: "still" });
      await p.refreshTakes();
    } catch (e) { setErr(String((e as Error).message || e)); }
    finally { setBusyMsg(""); }
  };
  const openMoveOn = (jid: string) => {
    setCamFor(jid);
    window.setTimeout(() => document.getElementById(`take-${jid}`)?.scrollIntoView({ behavior: "smooth", block: "start" }), 100);
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
    return !!dr && dr.sig === sig(idx) && (dr.end_still_id || null) === endNow && (dr.flags.length === 0 || p.acceptFlags) && !seedUsed(idx, dr.settings?.seed, dr);
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
  const doUpscale = async (idx: number, t: Take, eng: UpEngine = "flashvsr_plus", fast = false) => {
    const prev = upsOf(t)[eng];
    const ask = fast
      ? `Upscale this take 2x with FlashVSR+ for fast motion (temporal compression 1, about 5x slower than plain 2x) on the box now? (roughly ${Math.max(1, Math.round(((t.seconds || 6) * FAST_SECS_PER_SEC) / 60))} min)${prev?.status === "done" ? " It replaces the current FlashVSR+ upscale in the 1440p cut." : ""}`
      : `Upscale this take ${upAsk(eng)} on the box now? (${upMins(eng, t.seconds || 6)})`;
    if (!window.confirm(ask)) return;
    setErr(""); setBusyMsg("sending the take to the box...");
    try {
      const r = await api.mvUpscaleTakes({ jobs: { [String(idx)]: t.id }, script: scriptName, engine: eng, force: prev?.status === "error" || prev?.status === "done", ...(fast ? { temporal_compression: 1 } : {}) });
      if (!r.queued?.length && r.skipped?.length) setErr(`not queued: ${r.skipped[0].why}`);
      await p.refreshTakes();
    } catch (e) { setErr(String((e as Error).message || e)); }
    finally { setBusyMsg(""); }
  };

  // finished takes that landed after the scene's take was chosen: to review. The choice's moment is
  // locked.chosen_at (choosing again, or "keep this one", moves it); an older lock without it counts
  // from when its own take landed
  const newSinceChoice = (sc: Scene) => {
    if (!sc.locked) return [] as Take[];
    const l = list(sc.index);
    const base = sc.locked.chosen_at ?? l.find((t) => t.id === sc.locked!.job_id)?.finished ?? 0;
    return l.filter((t) => t.id !== sc.locked!.job_id && t.status === "done" && !t.stale && (t.finished ?? 0) > base);
  };
  const stateOf = (sc: Scene) => {
    const l = list(sc.index);
    const run = l.find((t) => t.status !== "done");
    const n = l.filter((t) => t.status === "done" && !t.stale).length;
    const old = l.filter((t) => t.status === "done" && t.stale).length;
    // an open render shows first, also on a scene whose take is already chosen (a new take on its way)
    if (run && !run.stale && run.status !== "error") {
      const what = run.status === "running" ? `rendering ${pct(run.progress, run.max)}`.trim() : "queued on the box";
      return { tone: "accent" as const, text: sc.locked ? `${what} · take chosen` : what, todo: false };
    }
    const fresh = newSinceChoice(sc).length;
    if (sc.locked && fresh) return { tone: "warn" as const, text: `${fresh} new take${fresh === 1 ? "" : "s"} to review · take chosen`, todo: true };
    if (sc.locked) return { tone: "ok" as const, text: "take chosen", todo: false };
    if (!n && (old || run)) return { tone: "warn" as const, text: "needs a new render (old shots only)", todo: true };
    if (n) return { tone: "warn" as const, text: `${n} take${n === 1 ? "" : "s"}, none chosen`, todo: true };
    return { tone: "muted" as const, text: "not rendered", todo: true };
  };
  // the batch: every scene with no take yet and none chosen, checked together then sent in the background
  type BatchRow = { index: number; title?: string; status: string; error?: unknown; flags?: string[]; job_id?: string | null;
    settings?: { turbo?: boolean; steps?: number; sampler?: string; scheduler?: string } };
  type BatchDry = { manifest: string; counts: Record<string, number>; flagged: number[]; scenes: BatchRow[]; sig: string; only: number[] };
  type BatchJob = { id: string; status: string; error?: string; counts?: Record<string, number>; rows: BatchRow[]; total?: number };
  const [bDry, setBDry] = useState<BatchDry | null>(null);
  const [bAccept, setBAccept] = useState(false);
  const [bJob, setBJob] = useState<BatchJob | null>(null);
  const unrendered = script.scenes.filter((sc) => !sc.locked && list(sc.index).filter((t) => !t.stale).length === 0).map((sc) => sc.index);
  const bSig = () => `${commonSig()}|${unrendered.map((i) => `${i}:${JSON.stringify(recipeAt(i))}`).join(",")}`;
  const batchCommon = () => {
    const b: Record<string, unknown> = {
      script: scriptName, ref_ids: [p.faceId, p.bodyId].map((x) => x.trim()).filter(Boolean),
      attention: ATTENTION, width: Number(p.width), height: Number(p.height),
      per_scene: Object.fromEntries(unrendered.map((i) => [String(i), recipeKnobs({}, recipeAt(i))])),
    };
    if (p.soloLock && p.script.duet) b.solo_lock = true;
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
  const queuedHere = Object.values(takes).flat().filter((t) => t.status === "pending").map((t) => t.id);
  const doCancelQueued = async () => {
    if (!window.confirm(`Take ${queuedHere.length} queued render${queuedHere.length === 1 ? "" : "s"} of this script off the box? The render running now carries on.`)) return;
    setErr(""); setBusyMsg("cancelling the queued renders...");
    try { await api.videoCancelQueued({ ids: queuedHere }); await p.refreshTakes(); }
    catch (e) { setErr(String((e as Error).message || e)); }
    finally { setBusyMsg(""); }
  };
  const doMark = async (idx: number, clear: boolean) => {
    setErr("");
    try { await api.mvMarkRerender({ script: scriptName, scenes: [idx], clear }); p.reloadScript(); await p.refreshTakes(); }
    catch (e) { setErr(String((e as Error).message || e)); }
  };
  // each scene's checked settings must be its own recipe: a backend from before per-scene settings
  // ignores per_scene (and reports no sampler), so its check is refused rather than fired
  const bWrong = bReady.filter((r) => {
    const s = r.settings || {}; const want = recipeAt(r.index);
    return s.sampler === undefined || !!s.turbo !== want.turbo || s.sampler !== want.sampler || s.scheduler !== want.scheduler
      || (want.steps.trim() !== "" && Number(s.steps) !== Number(want.steps));
  });
  const bBase = bReady.filter((r) => !recipeAt(r.index).turbo).length;
  const bFlagged = bReady.filter((r) => (r.flags || []).length);
  const bSendable = bWrong.length ? 0 : bReady.length - (bAccept ? 0 : bFlagged.length);
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
  // coming back to the step (or reloading): pick up this song's batch if it is still sending, any of
  // its renders is still unfinished on the box (a whole song renders for many hours), or it ended
  // in the last half hour, so its progress and errors show again
  useEffect(() => {
    let live = true;
    api.mvRenderScriptLatest(scriptName).then((r) => {
      const j = (r as { job: (BatchJob & { ended_ago?: number | null; open?: number }) | null }).job;
      if (live && j && (j.status === "sending" || (j.open ?? 0) > 0 || (j.ended_ago ?? Infinity) < 1800)) setBJob((b) => b ?? j);
    }).catch(() => {});
    return () => { live = false; };
  }, [scriptName]);
  useEffect(() => {
    if (!bJob || bJob.status !== "sending") return;
    const t = window.setInterval(async () => {
      try {
        const j = (await api.mvRenderScriptJob(bJob.id)) as BatchJob;
        setBJob((b) => ({ ...j, total: j.total ?? b?.total }));
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
  // every open render of this script that the batch bar does not cover (single takes, re-renders):
  // one bar for them, counted from when this page saw them open
  const inBatch = new Set((bJob?.rows || []).map((r) => r.job_id).filter(Boolean) as string[]);
  const openLoose = script.scenes.flatMap((sc) => list(sc.index).filter((t) => t.status !== "done" && t.status !== "error" && !t.stale && !inBatch.has(t.id)).map((t) => ({ sc, t })));
  const [looseSeen, setLooseSeen] = useState<Record<string, number>>({});
  useEffect(() => {
    if (!openLoose.length) return;
    setLooseSeen((m) => {
      const add = openLoose.filter(({ t }) => !(t.id in m));
      if (!add.length) return m;
      const n = { ...m }; add.forEach(({ sc, t }) => { n[t.id] = sc.index; }); return n;
    });
  }, [openLoose.map(({ t }) => t.id).join(",")]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { setLooseSeen({}); }, [scriptName]);
  const looseFrac = (id: string, idx: number) => {
    const t = list(idx).find((x) => x.id === id);
    if (!t || t.status === "done" || t.status === "error") return 1;
    return t.status === "running" && t.max ? Math.min(0.99, (t.progress || 0) / t.max) : 0;
  };
  const looseIds = Object.entries(looseSeen);
  const looseDone = looseIds.filter(([id, idx]) => looseFrac(id, idx) === 1).length;
  const looseNow = openLoose.find(({ t }) => t.status === "running");

  const shown = script.scenes.filter((s) => filter === "all" || stateOf(s).todo);
  const tl = list(scene.index);
  const freshHere = newSinceChoice(scene);
  const lockedElsewhere = scene.locked && !tl.some((t) => t.id === scene.locked!.job_id);
  const dr = dry[scene.index];

  return (
    <div className="grid gap-4 lg:grid-cols-[320px_1fr]">
      {/* top: render every scene that has no take yet, in one go */}
      <Card className="lg:col-span-2" title={`Render the rest: ${unrendered.length} scene${unrendered.length === 1 ? "" : "s"} with no take yet`}
        sub="one settings check for all of them (no GPU), then one button sends them all to the box. Scenes that already have a take are left alone; render those one at a time below.">
        <div className="flex flex-wrap items-center gap-2">
          <GhostButton onClick={doBatchDry} disabled={!!busyMsg || bSending || unrendered.length === 0}>1 · Check all {unrendered.length}</GhostButton>
          {queuedHere.length > 0 && <GhostButton onClick={doCancelQueued} disabled={!!busyMsg}
            title="takes the renders that have not started off the box queue; the one rendering now carries on">cancel {queuedHere.length} queued</GhostButton>}
          <Action onClick={doBatchFire} disabled={!!busyMsg || bSending || !bFresh || bSendable === 0}
            title={bFresh ? "send them to the box" : "check them first"}>2 · Render {bFresh ? bSendable : unrendered.length} on the box</Action>
          {bFresh && bFlagged.length > 0 && <label className="flex items-center gap-1 text-xs text-amber-400"><input type="checkbox" checked={bAccept} onChange={(e) => setBAccept(e.target.checked)} /> also render the {bFlagged.length} with warnings</label>}
        </div>
        {bDry && !bFresh && !bJob && <div className="text-xs text-amber-400">the settings or the scenes changed since the check: check again</div>}
        {bDry && bFresh && !bJob && (
          <div className={`rounded border p-2 text-[11px] ${bFlagged.length || bDry.scenes.some((r) => r.status === "error") ? "border-amber-700" : "border-emerald-800"}`}>
            <div className="text-[var(--color-muted)]">{bReady.length} ready{bFlagged.length ? `, ${bFlagged.length} with warnings` : ""}{bDry.counts?.error ? `, ${bDry.counts.error} cannot render` : ""} · {p.width}x{p.height} · {bBase} on base, {bReady.length - bBase} on turbo (each scene's own settings) · {p.seed.trim() ? `seed ${p.seed}` : "a random seed per scene (kept from this check)"}</div>
            {bWrong.length > 0 && <div className="mt-1 text-red-300">{bWrong.length} scene{bWrong.length === 1 ? "" : "s"} would not render on {bWrong.length === 1 ? "its" : "their"} own settings: the backend has not picked up per-scene settings yet and needs a restart. Nothing can be sent from this check.</div>}
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
        {looseIds.length > 0 && (
          <div className="space-y-2 rounded border border-[var(--color-line)] p-2 text-[11px]">
            <Bar label={`single renders on the box: ${looseDone} of ${looseIds.length} done${looseNow ? ` · now scene ${String(looseNow.sc.index).padStart(2, "0")} ${looseNow.sc.title || ""} ${pct(looseNow.t.progress || 0, looseNow.t.max || 0)}` : ""}`}
              frac={looseIds.reduce((a, [id, idx]) => a + looseFrac(id, idx), 0) / looseIds.length} />
            <div className="text-[var(--color-muted)]">
              {openLoose.length ? `${openLoose.length} to go: ${openLoose.map(({ sc }) => String(sc.index).padStart(2, "0")).join(", ")} (the box renders them in the order sent; each scene on the left shows its own state)`
                : "all done: the new takes are on their scenes below"}
            </div>
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
                  <div className="truncate text-xs font-semibold">{String(sc.index).padStart(2, "0")} {sc.title}{sc.singer === "him" ? " (him)" : ""}</div>
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
              {scene.singer && scene.singer !== "none" && <Row k="singer" v={scene.singer === "him" ? "him (his two pictures)" : "her"} />}
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
              busy={!!busyMsg} setErr={setErr} reload={p.reloadScript}
              armed={JSON.stringify(p.target) === JSON.stringify({ kind: "after", idx: scene.index })}
              onArm={() => p.setTarget({ kind: "after", idx: scene.index })} onFile={(f) => p.fileStill(scene.index, f, "after")} />
            {moving && (() => {
              const still = list(scene.index).find((t) => t.params.variant === "still");
              const stillDone = !!still && still.status === "done";
              const derived = list(scene.index).filter((t) => ["move2d", "meridian", "roomplate"].includes(String(t.params.engine)));
              const step = !still ? 1 : !stillDone ? 1 : derived.length ? 3 : 2;
              const n = (k: number, label: string) => (
                <span className={`mr-2 inline-flex h-5 w-5 items-center justify-center rounded-full text-[11px] font-semibold ${step === k ? "bg-[var(--color-accent)] text-white" : step > k ? "bg-emerald-700 text-white" : "bg-[var(--color-line)]"}`}>{step > k ? "✓" : label}</span>);
              return (
                <Card title="Camera move broke the room?"
                  sub="This scene's camera moves. If the walls, windows or signs change shape as it moves, don't re-render it as it is: render it with the camera still, then add the move to that clean take. Three steps:">
                  <div className="space-y-3 text-xs">
                    <div className="flex items-start">
                      {n(1, "1")}
                      <div className="flex-1">
                        <div className="font-semibold">Render a still version of this scene</div>
                        <div className="text-[var(--color-muted)]">the same scene and singing with the camera still, so the room cannot warp. About 10 min on the box. It appears in the takes below labelled "still version".</div>
                        <div className="mt-1">
                          {!still ? <Action onClick={() => doStillRender(scene.index)} disabled={!!busyMsg}>render the still version</Action>
                            : !stillDone ? <Pill tone="warn">still version rendering on the box</Pill>
                            : <span className="text-emerald-500">done ({when(still.created)})</span>}
                        </div>
                      </div>
                    </div>
                    <div className="flex items-start">
                      {n(2, "2")}
                      <div className="flex-1">
                        <div className="font-semibold">Add the camera move to the still version</div>
                        <div className="text-[var(--color-muted)]">opens a panel on the still version where you pick the move and one of three ways to add it:</div>
                        <ul className="ml-4 list-disc text-[var(--color-muted)]">
                          <li><b>Room pinned</b> (best for singing): real depth, the room cannot change, lips as good as a normal take. You make one picture in ChatGPT (the panel gives you the frame and the prompt). About 12 min.</li>
                          <li><b>2D move</b>: seconds, nothing redrawn, but flat (no depth).</li>
                          <li><b>Meridian</b>: real depth and no picture needed, but lips can lag when she sings energetically. About 13 min.</li>
                        </ul>
                        <div className="mt-1">
                          <Action onClick={() => still && openMoveOn(still.id)} disabled={!stillDone || !!busyMsg}>add the move to the still version</Action>
                          {!stillDone && <span className="ml-2 text-[var(--color-muted)]">after step 1</span>}
                        </div>
                        <div className="mt-2"><RoomPinnedProgress job={rpJob} /></div>
                      </div>
                    </div>
                    <div className="flex items-start">
                      {n(3, "3")}
                      <div className="flex-1">
                        <div className="font-semibold">Watch the new take and choose it</div>
                        <div className="text-[var(--color-muted)]">it appears in the takes below, labelled with the way and the move (for example "room pinned: pull out, small"). Press "Use in the video" on the one you want.</div>
                      </div>
                    </div>
                  </div>
                </Card>
              );
            })()}
            <Card title="Render this scene on the box" sub="first check the settings (no GPU, shows any warnings), then render. Each render adds a take below.">
              <div className="flex flex-wrap items-center gap-2">
                <GhostButton onClick={() => doDry(scene.index)} disabled={!!busyMsg}>1 · Check the settings</GhostButton>
                <Action onClick={() => doRender(scene.index)} disabled={!!busyMsg || !canRender(scene.index)}
                  title={canRender(scene.index) ? "queue this scene on the box" : "check the settings first; warnings need the accept tick"}>2 · Render on the box</Action>
                {dr && dr.flags.length > 0 && <label className="flex items-center gap-1 text-xs text-amber-400"><input type="checkbox" checked={p.acceptFlags} onChange={(e) => p.setAcceptFlags(e.target.checked)} /> render despite the warnings</label>}
                {scene.takes_after
                  ? <GhostButton onClick={() => doMark(scene.index, true)} disabled={!!busyMsg} title="its earlier takes count again">marked for a new render: undo</GhostButton>
                  : list(scene.index).length > 0 && <GhostButton onClick={() => doMark(scene.index, false)} disabled={!!busyMsg}
                      title="its takes so far become old shots, and the batch above renders it again">needs a new render</GhostButton>}
              </div>
              {dr && (
                <div className={`rounded border p-2 text-[11px] ${dr.sig !== sig(scene.index) || dr.flags.length ? "border-amber-700" : "border-emerald-800"}`}>
                  <div className="flex flex-wrap gap-x-3 gap-y-1 text-[var(--color-muted)]">
                    <span>{dr.clips} clip{dr.clips === 1 ? "" : "s"}</span><span>{dr.settings?.width}x{dr.settings?.height}</span><span>seed {dr.settings?.seed}</span>
                    <span>{dr.settings?.turbo ? "turbo" : "base"}, {dr.settings?.steps} steps, {dr.settings?.sampler || rc.sampler}/{dr.settings?.scheduler || rc.scheduler}</span>{dr.lane === "i2v" ? <span>starts on {dr.first_frame_id ? "a picture" : "the place's picture"}</span> : <span>renders from the singer's and the place's pictures{dr.anchor_first ? ", starting on this scene's picture" : ""}</span>}
                    {dr.variant === "still" ? <span>still version (camera static)</span> : null}
                    {dr.end_still_id ? <span>ends on the end picture</span> : null}
                    {dr.rail_frames?.length ? <span>pinned on the way at {dr.rail_frames.map((f) => (f / 24).toFixed(1)).join(", ")} s</span> : null}
                    {dr.sig !== sig(scene.index) || (dr.end_still_id || null) !== (scene.end_still_id || null) ? <span className="text-amber-400">settings changed since the check: check again</span> : null}
                  </div>
                  {seedUsed(scene.index, dr.settings?.seed, dr) && <div className="mt-1 text-amber-300">seed {dr.settings?.seed} already made a take of this scene with the same end picture, so it would most likely come out the same. Clear the seed in the render settings and check again for a new one.</div>}
                  {dr.flags.length ? <ul className="mt-1 list-disc pl-4 text-amber-300">{dr.flags.map((f, i) => <li key={i}>{f}</li>)}</ul> : <div className="mt-1 text-emerald-500">no warnings: ready to render</div>}
                </div>
              )}
              <details className="text-xs">
                <summary className="cursor-pointer text-[var(--color-muted)]">render settings · this scene: {recipeText(rc)}{p.recipeChanged(scene) ? " (changed)" : ` (default for a ${moving ? "camera move" : "static scene"})`}</summary>
                <div className="mt-2 text-[11px] text-[var(--color-muted)]">This scene only ({moving ? "camera move: base by default, turbo warped the room on moves" : "static camera: turbo by default"}):</div>
                <div className="mt-1 grid grid-cols-2 gap-2 md:grid-cols-4">
                  <label className="flex items-end gap-2 pb-2 text-xs"><input type="checkbox" checked={rc.turbo} onChange={(e) => p.setRecipe(scene, { turbo: e.target.checked })} /> turbo</label>
                  {!envOnly(scene) && (
                    <label className={`col-span-2 flex items-end gap-2 pb-2 text-xs md:col-span-3 ${scene.opening_still_id ? "" : "opacity-50"}`}
                      title="Off: her scenes render from her face, body and place pictures only. On: the clip starts on this scene's picture (step 3), and her face and body pictures still keep her identity. Untested on most scenes: judge the take.">
                      <input type="checkbox" checked={rc.anchor} disabled={!scene.opening_still_id && !rc.anchor} onChange={(e) => p.setRecipe(scene, { anchor: e.target.checked })} />
                      start on this scene's picture{scene.opening_still_id ? "" : " (add one in step 3 first)"}
                    </label>
                  )}
                  <Field label="steps" hint={`blank = ${rc.turbo ? "8 (the turbo recipe)" : "20 (the base recipe)"}`}>
                    <input className={inp} type="number" min={1} max={60} placeholder={rc.turbo ? "8" : "20"} value={rc.steps} onChange={(e) => p.setRecipe(scene, { steps: e.target.value })} /></Field>
                  <Field label="sampler" hint={SAMPLERS.find(([v]) => v === rc.sampler)?.[2] || ""}>
                    <select className={inp} value={rc.sampler} onChange={(e) => p.setRecipe(scene, { sampler: e.target.value })}>
                      {SAMPLERS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
                    </select></Field>
                  <Field label="scheduler" hint="Comfy's H3 template: beta or normal tends to beat simple when the pictures need to hold">
                    <select className={inp} value={rc.scheduler} onChange={(e) => p.setRecipe(scene, { scheduler: e.target.value })}>
                      {SCHEDULERS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
                    </select></Field>
                </div>
                {p.recipeChanged(scene) && <button className="mt-1 text-[11px] text-[var(--color-accent)] underline" onClick={() => p.setRecipe(scene, null)}>back to this scene's default ({recipeText(sceneDefaultRecipe(scene))})</button>}
                <div className="mt-3 text-[11px] text-[var(--color-muted)]">Every scene (the singer's pictures are set in step 3):</div>
                <div className="mt-1 grid grid-cols-2 gap-2 md:grid-cols-4">
                  <Field label="seed" hint="blank = a new random seed on every check"><input className={inp} value={p.seed} onChange={(e) => p.setSeed(e.target.value)} /></Field>
                  <Field label="width"><input className={inp} type="number" step={32} value={p.width} onChange={(e) => p.setWidth(Number(e.target.value))} /></Field>
                  <Field label="height"><input className={inp} type="number" step={32} value={p.height} onChange={(e) => p.setHeight(Number(e.target.value))} /></Field>
                </div>
                {p.script.duet && (
                  <label className="mt-2 flex items-center gap-2 text-xs" title="A lip-sync scene's render hears the song with the other singer's measured parts (his lines, her answers) taken out of the vocal, so the face on screen does not mouth them. The finished video keeps the full song.">
                    <input type="checkbox" checked={p.soloLock} onChange={(e) => p.setSoloLock(e.target.checked)} />
                    Two singers: lip-sync hears only the singer on screen (mutes the other singer's parts and answers)
                  </label>
                )}
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
              <RoomPinnedProgress job={rpJob} />
              <div className="grid gap-3 2xl:grid-cols-2">
                {tl.map((t) => {
                  const chosen = scene.locked?.job_id === t.id;
                  const c = t.check;
                  return (
                    <div key={t.id} id={`take-${t.id}`} className={`space-y-2 rounded-lg p-2 ${chosen ? "border-2 border-emerald-600" : "border border-[var(--color-line)]"}`}>
                      {t.status === "done" && t.media_url
                        ? <video src={t.media_url} controls muted preload="metadata" className="w-full rounded bg-black" />
                        : <div className="rounded bg-black/40 p-4"><Bar label={`${t.status} on the box`} frac={t.max ? (t.progress || 0) / t.max : 0} /></div>}
                      <div className="flex flex-wrap items-center gap-2">
                        {t.status === "done" && (chosen
                          ? <><Pill tone="ok">in the video</Pill><GhostButton onClick={() => choose(scene.index, null)}>remove</GhostButton>
                            {freshHere.length > 0 && <Action onClick={() => choose(scene.index, t.id)} title="you have looked at the new takes and this one stays: the scene stops asking for a review">keep this one</Action>}</>
                          : <Action onClick={() => choose(scene.index, t.id)} title="this take plays in the scene; it replaces the scene's current choice">Use in the video</Action>)}
                        {freshHere.some((x) => x.id === t.id) && <Pill tone="warn" title="finished after this scene's take was chosen">new, to review</Pill>}
                        <span className="text-[11px] text-[var(--color-muted)]">{when(t.created)} · seed {t.params.seed} · {t.params.turbo ? "turbo" : "base"}{t.params.steps ? ` ${t.params.steps} steps` : ""}{t.params.sampler && t.params.sampler !== DEFAULT_SAMPLER ? ` ${t.params.sampler}` : ""}{t.params.scheduler && t.params.scheduler !== DEFAULT_SCHEDULER ? `/${t.params.scheduler}` : ""}{t.seconds !== null ? ` · ${t.seconds.toFixed(1)} s` : ""}{t.from_script ? " · earlier script" : ""}</span>
                        {t.stale && <Pill tone="warn" title="rendered before this scene was rewritten or marked for a new render">old shot</Pill>}
                        {t.params.variant === "still" && <Pill tone="muted" title="rendered with a still camera: add the camera move on this take">still version</Pill>}
                        {t.params.engine === "roomplate" && <Pill tone="muted" title="rendered again with the room pinned to the empty-room picture; H3 drew the people">room pinned: {moveLabel(t.params.move)}, {t.params.amount}</Pill>}
                        {t.params.engine === "move2d" && <Pill tone="muted" title="a moving crop of another take of this scene: nothing redrawn">2D move: {moveLabel(t.params.move)}, {t.params.amount}</Pill>}
                        {t.params.engine === "meridian" && <Pill tone="muted" title="re-shot with Meridian from another take of this scene">Meridian: {moveLabel(t.params.move)}, {t.params.amount}</Pill>}
                      </div>
                      {t.status === "done" && (
                        <div className="flex flex-wrap items-center gap-2 text-[11px]">
                          {c ? <>
                            <Pill tone={c.floor_r === null ? "muted" : c.floor_ok ? "ok" : "bad"} title="lip and music sync against the song; clean takes measure 0.87-1.00">sync {c.floor_r === null ? "n/a" : c.floor_r.toFixed(2)}</Pill>
                            <Pill tone={c.opening_jump === null ? "muted" : c.opening_flag ? "bad" : "ok"} title="a jump in the first 2 s (fade or morph in); 4 or more is flagged">opening {c.opening_jump === null ? "n/a" : c.opening_jump.toFixed(1)}x</Pill>
                            <GhostButton onClick={() => openLightbox(c.sheet_url + "?t=" + Date.now())}>frames</GhostButton>
                          </> : <GhostButton onClick={() => doCheck(scene.index, t.id)} disabled={!!busyMsg} title="measure sync and the opening (Mac, no GPU)">measure sync</GhostButton>}
                          {(Object.keys(UP_ENGINES) as UpEngine[]).map((eng) => {
                            const u = upsOf(t)[eng];
                            const E = UP_ENGINES[eng];
                            return (
                              <Fragment key={eng}>
                                {u?.status === "done" && u.media_url
                                  ? <><Pill tone="ok">{E.done}{isFast(u) ? ", fast motion" : ""}</Pill><GhostButton onClick={() => openLightbox(u.media_url!)}>play {E.name}</GhostButton></>
                                  : upscaleBusy(u) ? <Pill tone="warn">{E.name}{isFast(u) ? " fast motion" : ""} {pct(u!.progress, u!.max) || u!.status}</Pill>
                                  : <GhostButton onClick={() => doUpscale(scene.index, t, eng)} disabled={!!busyMsg} title={E.tip}>{E.button}</GhostButton>}
                                {upscaleBusy(u?.newer) && <Pill tone="warn">{E.name}{isFast(u!.newer) ? " fast motion" : ""} {pct(u!.newer!.progress, u!.newer!.max) || u!.newer!.status}</Pill>}
                                {eng === "flashvsr_plus" && !upscaleBusy(u) && !upscaleBusy(u?.newer) && !(u?.status === "done" && isFast(u)) && (
                                  <GhostButton onClick={() => doUpscale(scene.index, t, eng, true)} disabled={!!busyMsg}
                                    title="FlashVSR+ Turbo 2x with temporal compression 1: better on fast camera moves and large moving areas, about 5x slower than plain 2x; still scenes are fine with plain 2x">upscale 2x, fast motion</GhostButton>
                                )}
                                {u?.status === "error" && <span className="text-red-400" title={u.error || ""}>{E.name} failed</span>}
                              </Fragment>
                            );
                          })}
                          {!["move2d", "meridian", "roomplate"].includes(String(t.params.engine)) && (
                            <GhostButton onClick={() => setCamFor(camFor === t.id ? null : t.id)} disabled={!!busyMsg}
                              title="add a camera move to this take: 2D move, room pinned or Meridian (the panel explains each)">{camFor === t.id ? "close camera move" : "add a camera move"}</GhostButton>
                          )}
                        </div>
                      )}
                      {camFor === t.id && (
                        <CameraMovePanel scriptName={scriptName} scene={scene} take={t} busy={!!busyMsg} setErr={setErr}
                          setBusyMsg={setBusyMsg} refreshTakes={p.refreshTakes} onClose={() => setCamFor(null)}
                          render={renderBody(scene.index, false)}
                          armed={JSON.stringify(p.target) === JSON.stringify({ kind: "empty", idx: scene.index, take: t.id })}
                          onArm={() => p.setTarget({ kind: "empty", idx: scene.index, take: t.id })}
                          onFile={(f) => p.fileStill(scene.index, f, "empty", t.id)}
                          roomJob={rpJob} onRoomJob={startRpJob} />
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

const QUALITY: { key: string; label: string; sub: string; size: string; up: boolean; engine?: UpEngine }[] = [
  { key: "draft", label: "Draft", sub: "the renders as they are (1344x768): a quick check of the edit", size: "1344x768", up: false },
  { key: "hd", label: "1080p from the renders", sub: "the renders resized to 1920x1080, no upscale", size: "1920x1080", up: false },
  { key: "up1440", label: "1440p, upscaled", sub: "every take upscaled 2x with FlashVSR+, cut at 2560x1440: the YouTube master", size: "2560x1440", up: true, engine: "flashvsr_plus" },
  { key: "up1440r", label: "1440p, upscaled with RTX VSR", sub: "every take upscaled 2x with NVIDIA RTX Video Super Resolution (HIGHBITRATE_ULTRA), cut at 2560x1440: to compare with FlashVSR+", size: "2560x1440", up: true, engine: "rtx_vsr" },
  { key: "up1440s", label: "1440p, upscaled with SeedVR2", sub: "every take upscaled to 1440p with SeedVR2 (the node author's video settings), cut at 2560x1440: to compare with FlashVSR+", size: "2560x1440", up: true, engine: "seedvr2" },
  { key: "up4k", label: "4K, upscaled", sub: "the FlashVSR+ 2x upscales stretched to 3840x2160", size: "3840x2160", up: true, engine: "flashvsr_plus" },
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
  const qEng: UpEngine = q.engine || "flashvsr_plus";

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
  const upOf = (job: string | null) => (job ? upsOf(takeById[job])[qEng] || null : null);
  const planned = plan.filter((x) => x.job);
  const nUpDone = planned.filter((x) => upOf(x.job)?.status === "done").length;
  const nUpBusy = planned.filter((x) => upscaleBusy(upOf(x.job))).length;
  const toUpscale = planned.filter((x) => { const u = upOf(x.job); return !u || u.status === "error"; });
  const nUpRunning = planned.filter((x) => { const s = upOf(x.job)?.status; return s === "running" || s === "finalizing"; }).length;
  const upLeftSecs = planned.filter((x) => upOf(x.job)?.status !== "done")
    .reduce((a, x) => a + (takeById[x.job!]?.seconds || (x.sc.end - x.sc.start)), 0);
  // "upscale all" uploads every take before it answers (about 8 s each, 6 min for a whole video):
  // meanwhile the takes are re-read so each one shows up as it reaches the box
  const [upSending, setUpSending] = useState<{ total: number; before: number } | null>(null);
  useEffect(() => {
    if (!upSending) return;
    const t = window.setInterval(() => { refreshTakes().catch(() => undefined); }, 4000);
    return () => window.clearInterval(t);
  }, [upSending]); // eslint-disable-line react-hooks/exhaustive-deps
  const upSent = upSending ? Math.min(upSending.total, Math.max(0, nUpBusy + nUpDone - upSending.before)) : 0;

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
    if (!window.confirm(`Upscale ${toUpscale.length} take${toUpscale.length === 1 ? "" : "s"} with ${UP_ENGINES[qEng].name} on the box? They run one after another: ${secs.toFixed(0)} s of video, ${upMins(qEng, secs)}.`)) return;
    setErr(""); setMsg(""); setBusy(true);
    setUpSending({ total: toUpscale.length, before: nUpBusy + nUpDone });
    try {
      const jobs: Record<string, string> = {};
      toUpscale.forEach((x) => { jobs[String(x.sc.index)] = x.job!; });
      const r = await api.mvUpscaleTakes({ jobs, script: scriptName, engine: qEng });
      setMsg(`queued ${r.queued?.length || 0} upscales${r.skipped?.length ? `, skipped ${r.skipped.length}` : ""}`);
      await refreshTakes();
    } catch (e) { setErr(String((e as Error).message || e)); }
    finally { setBusy(false); setUpSending(null); }
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
        script: scriptName, jobs, partial, width: w, height: h, async: true, use_upscaled: q.up, upscale_engine: qEng,
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
              <thead className="sticky top-0 bg-[var(--color-panel)] text-[var(--color-muted)]"><tr><th className="px-2 py-1.5 text-left">scene</th><th className="px-2 py-1.5 text-left">at</th><th className="px-2 py-1.5 text-left">take</th><th className="px-2 py-1.5 text-left">upscaled ({UP_ENGINES[qEng].name})</th></tr></thead>
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
                <div>{nUpDone} of {planned.length} takes upscaled with {UP_ENGINES[qEng].name}{nUpBusy ? ` · ${nUpRunning} upscaling now, ${nUpBusy - nUpRunning} waiting in the box queue` : ""}{nUpBusy && UP_ENGINES[qEng].secsPerSec ? ` · about ${Math.max(1, Math.round((upLeftSecs * UP_ENGINES[qEng].secsPerSec!) / 60))} min left` : ""}</div>
                {(nUpBusy > 0 || upSending) && (
                  <div className="h-1.5 w-full overflow-hidden rounded bg-[var(--color-line)]">
                    <div className="h-full bg-[var(--color-accent)] transition-all" style={{ width: `${planned.length ? (100 * nUpDone) / planned.length : 0}%` }} />
                  </div>
                )}
                {upSending && <div className="text-amber-400">sending the takes to the box: {upSent} of {upSending.total} sent (each one uploads in about 8 s; the box starts on the first straight away)</div>}
                {toUpscale.length > 0 && <GhostButton onClick={upscaleAll} disabled={busy}>upscale the {toUpscale.length} missing with {UP_ENGINES[qEng].name} on the box (GPU)</GhostButton>}
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
              <img src={`/api/youtube/endcard_preview?window=false&text=${encodeURIComponent(ecText)}&t=${encodeURIComponent(ecText.length)}`} alt="" className="w-full rounded border border-[var(--color-line)]" />
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
// the song engines whose drafts hold a song: the engine that made the picked track reads first
const SONG_ENGINES = ["suno", "music3", "yue2", "song"] as const;
function engineOrder(prefer?: string): string[] {
  return prefer && (SONG_ENGINES as readonly string[]).includes(prefer) ? [prefer, ...SONG_ENGINES.filter((e) => e !== prefer)] : [...SONG_ENGINES];
}
function lyricsFromProject(get: (ns: string, key: string) => unknown, prefer?: string): string {
  for (const ns of engineOrder(prefer)) {
    if (ns === "song") {
      const blocks = (get("song", "blocks") as { type?: string; lyrics?: string }[] | undefined) || [];
      const t = blocks.filter((b) => (b.lyrics || "").trim()).map((b) => `[${b.type || "Section"}]\n${(b.lyrics || "").trim()}`).join("\n\n");
      if (t.trim()) return t;
    } else {
      const t = (get(ns, "lyrics") as string) || "";
      if (t.trim()) return t;
    }
  }
  return "";
}
// the open project's song, from its live drafts (unsaved edits included): title, style, lyrics
function projectSong(get: (ns: string, key: string) => unknown, projectName?: string, prefer?: string): { title: string; style: string; lyrics: string } {
  const str = (ns: string, key: string) => ((get(ns, key) as string) || "").toString().trim();
  const m3Style = () => {
    // Music 3 keeps the genre at the end of its Basic Attributes field, the imagery in its own field
    const f = (get("music3", "fields") as Record<string, string> | undefined) || {};
    const tail = String(f["Basic Attributes"] || "").split(".").map((s) => s.trim()).filter(Boolean).pop() || "";
    const img = String(f["Application Scenarios & Imagery"] || "").trim();
    return [tail, img].filter(Boolean).join(". ");
  };
  let title = "", style = "";
  for (const ns of engineOrder(prefer)) {
    title = title || str(ns, "title");
    style = style || (ns === "suno" ? str("suno", "style") : ns === "yue2" ? str("yue2", "style") : ns === "song" ? str("song", "tags") : m3Style());
  }
  return { title: title || (projectName || "").trim(), style, lyrics: lyricsFromProject(get, prefer) };
}
// a track's song name: its title, else an import's name, walked back through params.source to the
// original take (a naturalized or mastered copy carries neither), with the copy suffix dropped
function trackOrigin(it: LibItem, byId: Map<string, LibItem>): LibItem {
  let cur = it;
  for (let i = 0; i < 12; i++) {
    const p = cur.params || {};
    const src = typeof p.source === "string" ? p.source : "";
    if ((p.title && String(p.title).trim()) || !/^[0-9a-f-]{32,36}$/.test(src)) return cur;
    const up = byId.get(src);
    if (!up) return cur;
    cur = up;
  }
  return cur;
}
function trackSongName(it: LibItem, byId: Map<string, LibItem>): string {
  const p = trackOrigin(it, byId).params || {};
  if (p.title && String(p.title).trim()) return String(p.title);
  return typeof p.source === "string" && !/^[0-9a-f-]{32,36}$/.test(p.source) ? p.source : "";
}
const songKey = (s: string) => s.toLowerCase().replace(/\s*\((naturalized|mastered|master|mixed|remaster(ed)?|imported)\)\s*$/i, "")
  .replace(/\s*\([^)]*\)\s*$/, "").replace(/[^a-z0-9]+/g, " ").trim();

function WriterPanel({ audioId, loaded, onWritten, startOpen, projectName, trackMode }: { audioId: string; loaded: { name: string; script: Script } | null; onWritten: (name: string) => Promise<void>; startOpen: boolean; projectName?: string; trackMode?: string }) {
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
  // opt-in bar-grid cutting (user 2026-11-13): cuts on the song's measured bar lines, one bar minimum, varied lengths
  const [barGrid, setBarGrid] = d.use("wBarGrid", false);
  const [minBars, setMinBars] = d.use("wMinBars", 1);
  const [bpmHint, setBpmHint] = d.use("wBpmHint", "");
  const [gridInfo, setGridInfo] = useState<{ for: string; how?: string; bar?: number; error?: string } | null>(null);
  useEffect(() => {
    if (!barGrid || !audioId) return;
    const key = `${audioId}|${bpmHint}`;
    if (gridInfo?.for === key) return;
    setGridInfo({ for: key, how: "measuring the tempo..." });
    api.mvBarGrid({ audio_id: audioId, bpm: Number(bpmHint) || undefined })
      .then((r) => setGridInfo({ for: key, how: (r as { how: string }).how, bar: (r as { bar: number }).bar }))
      .catch((e) => setGridInfo({ for: key, error: String((e as Error).message || e) }));
  }, [barGrid, audioId, bpmHint]);   // eslint-disable-line react-hooks/exhaustive-deps
  // the optional second (male) singer: off by default; who sings when is measured, not read from tags
  const [duet, setDuet] = d.use("wDuet", false);
  const [face2, setFace2] = d.use("wFace2", "");
  const [body2, setBody2] = d.use("wBody2", "");
  const [shared2, setShared2] = d.use("wShared2", "");
  const [voice2, setVoice2] = d.use("wVoice2", "");
  const [swap, setSwap] = d.use("wSwap", false);
  const [voices, setVoices] = d.use<Voices | null>("wVoices", null);
  const [measuring, setMeasuring] = useState(false);
  const measure = async (sw = swap) => {
    if (!lyrics.trim()) { setErr("the lyrics are needed first: the voices are read line by line"); return; }
    setErr(""); setMeasuring(true);
    try {
      const r = await api.mvVoices({ audio_id: audioId, lyrics, swap: sw }) as Voices;
      setVoices({ ...r, for: audioId });
    } catch (e) { setErr("Measuring the voices failed: " + String((e as Error).message || e)); }
    finally { setMeasuring(false); }
  };
  const [forTrack, setForTrack] = d.use("wTrack", "");
  const [job, setJob] = useState<{ id: string; status: string; log: string[]; seconds: number; error?: string | null; script?: string | null } | null>(null);
  const [jobId, setJobId] = d.use("wJobId", "");
  const [sending, setSending] = useState(false);
  const [err, setErr] = useState("");
  // a write started before the tab was left (or reloaded): show it again and keep polling it
  useEffect(() => {
    if (!jobId || job?.id === jobId) return;
    let alive = true;
    api.mvWriteJob(jobId).then((r) => { if (alive) setJob({ id: jobId, status: r.status, log: r.log || [], seconds: r.seconds, error: r.error, script: r.script }); })
      .catch(() => { if (alive) setJobId(""); });
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId]);

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
  const [latestFrom, setLatestFrom] = useState("");
  const fromLatest = async () => {
    setErr("");
    try {
      const rows = ((await api.mvScripts()) as ScriptRow[]).slice().sort((a, b) => b.mtime - a.mtime);
      for (const row of rows) {
        const r = ((await api.mvScript(row.name)).script as Script).refs || {};
        if (r.face && r.body && r.shared) {
          setFace(r.face); setBody(r.body); setShared(r.shared); setLatestFrom(row.title || row.name);
          return;
        }
      }
      setErr("no saved script has the singer's three descriptions");
    } catch (e) { setErr(String((e as Error).message || e)); }
  };
  // the open project's song: only what it has replaces a field, so a field it lacks keeps the track's
  const fromProject = () => {
    const ps = projectSong(drafts.get, projectName, trackMode);
    if (!ps.lyrics.trim() && !ps.style && !ps.title) { setErr("the open project has no song in it (Suno, Music 3, YuE2 or Song builder)"); return; }
    setErr("");
    if (ps.title) setTitle(ps.title);
    if (ps.style) setStyle(ps.style);
    if (ps.lyrics.trim()) setLyrics(ps.lyrics);
    else setErr("the open project has no lyrics: title and style filled");
  };
  const nLocks = loaded ? loaded.script.scenes.filter((s) => s.locked).length : 0;
  const missing = ([[title, "the title"], [lyrics, "the lyrics"], [face, "the face picture's description"], [body, "the full-body picture's description"], [shared, "what is in both pictures"],
    ...(duet ? [[face2, "his face picture's description"], [body2, "his full-body picture's description"], [shared2, "what is in both of his pictures"]] as const : [])] as const)
    .filter(([v]) => !v.trim()).map(([, label]) => label);
  const ready = !missing.length;

  const write = async () => {
    if (!ready) { setErr(`Still needed before Claude can write: ${missing.join(", ")}.`); return; }
    if (!window.confirm(`Write a new script for "${title}"? Claude (the subscription, no GPU) writes the whole video; it takes several minutes.${keepLocks && nLocks ? ` The ${nLocks} chosen takes of the open script are kept.` : ""}`)) return;
    setErr(""); setSending(true);
    try {
      const r = await api.mvWrite({
        async: true, audio_id: audioId, title: title.trim(), style: style.trim(), lyrics, concept: concept.trim() || undefined,
        refs: { face: face.trim(), body: body.trim(), shared: shared.trim() }, locations,
        keep_locks_from: keepLocks && nLocks && loaded ? loaded.name : undefined,
        duet: duet ? { refs2: { face: face2.trim(), body: body2.trim(), shared: shared2.trim() }, voice: voice2.trim() || undefined, swap } : undefined,
        bar_grid: barGrid ? { min_bars: minBars, bpm: Number(bpmHint) || undefined } : undefined,
      });
      setJob({ id: r.job_id, status: "writing", log: [], seconds: 0 }); setJobId(r.job_id);
    } catch (e) { setErr("The writer did not start: " + String((e as Error).message || e)); }
    finally { setSending(false); }
  };
  useEffect(() => {
    if (!job || job.status !== "writing") return;
    const h = window.setInterval(async () => {
      try {
        const r = await api.mvWriteJob(job.id);
        setJob({ id: job.id, status: r.status, log: r.log || [], seconds: r.seconds, error: r.error, script: r.script });
        if (r.status === "done" && r.script) { window.clearInterval(h); setJobId(""); await onWritten(r.script); }
        if (r.status === "error") { window.clearInterval(h); setJobId(""); }
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
            <textarea className={`${inp} h-40 font-mono text-[11px]`} value={lyrics} onChange={(e) => setLyrics(e.target.value)} placeholder="paste the lyrics, or fill them from the open project" />
          </Field>
          <div className="flex flex-wrap gap-2">
            <GhostButton onClick={fromProject} title={projectName ? `title, style and lyrics from the drafts of "${projectName}" (unsaved edits included)` : "title, style and lyrics from the drafts in the tabs"}>
              fill from the open project{projectName ? ` (${projectName})` : ""}</GhostButton>
            {loaded ? <GhostButton onClick={fromScript} title="the singer's descriptions (and the idea, if empty) of the open script">copy the singer from the open script</GhostButton>
              : <GhostButton onClick={fromLatest} title="the three descriptions of the singer from your newest script that has them (any song)">copy the singer from your latest script</GhostButton>}
            {latestFrom && <span className="self-center text-[11px] text-[var(--color-muted)]">singer copied from "{latestFrom}"</span>}
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
          <label className="flex items-center gap-2 text-xs font-semibold">
            <input type="checkbox" checked={duet} onChange={(e) => setDuet(e.target.checked)} /> Two singers: a male second voice as well as her
          </label>
          {duet && (
            <div className="space-y-2 rounded border border-[var(--color-line)] p-2">
              <div className="text-[11px] text-[var(--color-muted)]">Who sings when is measured from the recording (the vocal stem), not read from the lyric tags. Each scene then shows the singer whose voice is heard. His two pictures go in at step 3.</div>
              <div className="grid gap-2 md:grid-cols-3">
                <Field label="his face picture" hint="one line"><textarea className={`${inp} h-20 text-[11px]`} value={face2} onChange={(e) => setFace2(e.target.value)} /></Field>
                <Field label="his full-body picture" hint="one line"><textarea className={`${inp} h-20 text-[11px]`} value={body2} onChange={(e) => setBody2(e.target.value)} /></Field>
                <Field label="in both of his" hint="what never changes"><textarea className={`${inp} h-20 text-[11px]`} value={shared2} onChange={(e) => setShared2(e.target.value)} /></Field>
              </div>
              <Field label="how he sings" hint="optional, e.g. deep guttural growls"><input className={inp} value={voice2} onChange={(e) => setVoice2(e.target.value)} /></Field>
              <div className="flex flex-wrap items-center gap-3">
                <GhostButton onClick={() => measure()} title="aligns the lyrics on the vocal stem and fingerprints each line on the box (a few seconds; the first time also aligns the lyrics)">
                  {measuring ? "measuring..." : voices && voices.for === audioId ? "measure again" : "measure who sings when"}</GhostButton>
                <label className="flex items-center gap-2 text-xs" title="the measurement found two voices; this says which one is his">
                  <input type="checkbox" checked={swap} onChange={(e) => { setSwap(e.target.checked); if (voices && voices.for === audioId) measure(e.target.checked); }} /> swap who is who
                </label>
              </div>
              {voices && voices.for === audioId && (
                <div className="space-y-1 text-[11px]">
                  <div className="text-[var(--color-muted)]">her {voices.sung_seconds.her} s, him {voices.sung_seconds.him} s. Named by: {voices.how}. Listen at the hand-overs; if the two are the wrong way round, tick swap.</div>
                  <div className="max-h-72 space-y-1 overflow-auto">
                    {voices.sections.map((sec, i) => (
                      <div key={i}>
                        <div className="font-semibold">{sec.type} <span className="font-normal text-[var(--color-muted)]">{mmss(sec.start)}</span></div>
                        {sec.lines.map((l, j) => (
                          <div key={j} className="flex items-baseline gap-2 pl-2">
                            <span className="w-12 shrink-0 tabular-nums text-[var(--color-muted)]">{mmss(l.start)}</span>
                            <span className={`shrink-0 rounded px-1 text-[10px] ${VOICE_STYLE[l.voice] || VOICE_STYLE.unknown}`}>{VOICE_WORDS[l.voice] || l.voice}</span>
                            {(l.answers || []).map((a, k) => (
                              <span key={k} className="shrink-0 rounded bg-sky-900 px-1 text-[10px] text-sky-200" title="a short answer from the other voice inside this line: heard off screen, muted from the lip-sync with the two-singer option in step 4">answered by {a.voice === "him" ? "him" : "her"} {mmss(a.start)}</span>
                            ))}
                            {l.unsure && <span className="shrink-0 rounded bg-amber-900 px-1 text-[10px] text-amber-200" title="the measurement cannot place this line: the writer covers it with B-roll">unsure: B-roll</span>}
                            <span className="truncate">{l.text}</span>
                          </div>
                        ))}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
          <div className="flex flex-wrap items-end gap-3">
            <Field label="places">
              <select className={inp} value={locations} onChange={(e) => setLocations(e.target.value)}>
                <option value="auto">this song's places (default)</option>
                <option value="library">every place with a picture</option>
                <option value="none">invent them all</option>
              </select>
            </Field>
            {nLocks > 0 && <label className="flex items-center gap-2 pb-2 text-xs"><input type="checkbox" checked={keepLocks} onChange={(e) => setKeepLocks(e.target.checked)} /> keep the open script's {nLocks} chosen takes</label>}
            <label className="flex items-center gap-2 pb-2 text-xs" title="Scene cuts fall on the song's measured bar lines; the writer varies the lengths in whole bars, never shorter than the minimum. Off: cuts at section edges and before sung lines."><input type="checkbox" checked={barGrid} onChange={(e) => setBarGrid(e.target.checked)} /> cut on the bar grid</label>
            {barGrid && <>
              <Field label="shortest shot">
                <select className={inp} value={minBars} onChange={(e) => setMinBars(Number(e.target.value))}>
                  {[1, 2, 3, 4].map((n) => <option key={n} value={n}>{n} bar{n > 1 ? "s" : ""}</option>)}
                </select>
              </Field>
              <Field label="tempo hint (bpm, optional)">
                <input className={inp + " w-24"} value={bpmHint} placeholder="measured" onChange={(e) => setBpmHint(e.target.value.replace(/[^0-9.]/g, ""))} />
              </Field>
            </>}
            <Action onClick={write} disabled={sending || job?.status === "writing"}>{sending ? "starting..." : job?.status === "writing" ? "writing..." : "Write the script"}</Action>
          </div>
          {barGrid && gridInfo && <div className={`text-[11px] ${gridInfo.error ? "text-red-400" : "text-[var(--color-muted)]"}`}>
            {gridInfo.error ? `bar grid: ${gridInfo.error}` : `bar grid: ${gridInfo.how}${gridInfo.bar ? `; one bar = ${gridInfo.bar.toFixed(2)} s` : ""}`}
          </div>}
          {!ready && !job && <div className="text-[11px] text-amber-400">Still needed: {missing.join(", ")}.</div>}
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
