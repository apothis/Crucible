import { useEffect, useMemo, useState } from "react";
import { api, type Config, type LibItem, type Project, type SongDraft } from "./api";
import { GenerateForm, RestyleForm, RepaintForm, LayerForm, VocalsForm, SwapForm, StemsForm, ToneForm, BackingForm, GuitarForm, MasterForm, MixForm, SongForm } from "./forms";
import { VocalBuilderForm } from "./VocalBuilder";
import { ImportForm } from "./Import";
import { Assistant } from "./Assistant";
import { WavePlayer } from "./WavePlayer";
import { DraftProvider, useDraftCtx } from "./drafts";
import type { Result, RunCtx } from "./ui";

const MODES = [
  { id: "generate", label: "Generate" },
  { id: "song", label: "Song" },
  { id: "vocalbuilder", label: "Voc. Builder" },
  { id: "import", label: "Import" },
  { id: "restyle", label: "Restyle" },
  { id: "repaint", label: "Repaint" },
  { id: "layer", label: "Add Layer" },
  { id: "vocals", label: "Vocals" },
  { id: "swap", label: "Voice Swap" },
  { id: "stems", label: "Stems" },
  { id: "tone", label: "Tone" },
  { id: "backing", label: "Backing" },
  { id: "guitar", label: "Guitar" },
  { id: "master", label: "Master" },
  { id: "mix", label: "Mix" },
] as const;

// Grouped navigation (replaces the flat tab row) — by workflow stage.
const GROUPS: { name: string; modes: string[] }[] = [
  { name: "Create", modes: ["generate", "song", "restyle", "repaint", "layer"] },
  { name: "Guitar", modes: ["backing", "guitar", "tone"] },
  { name: "Vocals", modes: ["vocalbuilder", "vocals", "swap", "import"] },
  { name: "Finish", modes: ["stems", "mix", "master"] },
];
const LABELS: Record<string, string> = Object.fromEntries(MODES.map((m) => [m.id, m.label]));

export default function App() {
  return (
    <DraftProvider>
      <AppInner />
    </DraftProvider>
  );
}

function AppInner() {
  const drafts = useDraftCtx();
  const [cfg, setCfg] = useState<Config | null>(null);
  const [mode, setMode] = useState<string>("generate");
  const [library, setLibrary] = useState<LibItem[]>([]);
  const [results, setResults] = useState<Result[]>([]);
  const [song, setSong] = useState<SongDraft | null>(null);
  const [handoff, setHandoff] = useState<{ tags?: string; lyrics?: string } | null>(null);
  const [libOpen, setLibOpen] = useState(true);
  const [openGroups, setOpenGroups] = useState<string[]>(GROUPS.map((g) => g.name));
  // Projects — a saved bundle of every page's drafts + the song arrangement + tab.
  const [projects, setProjects] = useState<Project[]>([]);
  const [current, setCurrent] = useState<{ id: string; name: string } | null>(null);
  // ACE-Step engine status (availability + currently-loaded model), polled.
  const [ace, setAce] = useState<{ reachable: boolean; model?: string } | null>(null);

  const refreshLib = () => api.library().then(setLibrary).catch(() => {});
  const refreshProjects = () => api.projects().then(setProjects).catch(() => {});
  useEffect(() => { api.config().then(setCfg).catch(() => {}); refreshLib(); refreshProjects(); }, []);
  useEffect(() => { setResults([]); }, [mode]);
  // Poll the engine for availability + the loaded model (so a base/sft swap shows in the header).
  useEffect(() => {
    if (!cfg?.acestep) { setAce(null); return; }
    const tick = () => api.acestepInfo()
      .then((d: any) => setAce({ reachable: !!d?.reachable, model: d?.health?.data?.loaded_model || "" }))
      .catch(() => setAce({ reachable: false }));
    tick();
    const t = window.setInterval(tick, 12000);
    return () => window.clearInterval(t);
  }, [cfg?.acestep]);

  const busy = useMemo(() => results.some((r) => r.status === "pending" || r.status === "running"), [results]);
  const ctx: RunCtx = {
    setResults,
    patch: (id, p) => setResults((rs) => rs.map((r) => (r.id === id ? { ...r, ...p } : r))),
    onDone: refreshLib,
  };

  const snapshot = () => ({ drafts: drafts.store, song, mode });
  async function projectSave() {
    if (!current) return projectSaveAs();
    try { const r = await api.projectSave(current.id, { data: snapshot() }); setCurrent({ id: current.id, name: r.name }); refreshProjects(); }
    catch (e) { alert("Save failed: " + (e as Error).message); }
  }
  async function projectSaveAs() {
    const name = window.prompt("Save project as:", current?.name || "My song");
    if (!name) return;
    try { const r = await api.projectCreate({ name, data: snapshot() }); setCurrent({ id: r.id, name: r.name }); refreshProjects(); }
    catch (e) { alert("Save failed: " + (e as Error).message); }
  }
  async function projectOpen(id: string) {
    if (!id) return;
    try {
      const p = await api.projectGet(id);
      drafts.replaceAll(p.data?.drafts || {});
      setSong(p.data?.song ?? null);
      setHandoff(null);
      setResults([]);
      setMode(p.data?.mode || "generate");
      setCurrent({ id: p.id, name: p.name });
    } catch (e) { alert("Open failed: " + (e as Error).message); }
  }
  function projectNew() {
    drafts.reset();
    setSong(null);
    setHandoff(null);
    setResults([]);
    setCurrent(null);
    setMode("generate");
  }

  // Re-open a builder-made song in the Song Builder: repopulate its drafts (arrangement,
  // lyrics, tags, key/bpm, title) from the stored recipe, then switch to the Song tab.
  function loadSongIntoBuilder(it: LibItem) {
    const m = it.params?.song_meta;
    if (!m) return;
    const blocks = (m.blocks || []).map((b: any, i: number) => ({
      id: `lib${Date.now()}_${i}`, type: b.type, seconds: b.seconds, lyrics: b.lyrics || "", locked: false,
    }));
    drafts.set("song", "blocks", blocks);
    drafts.set("song", "tags", m.tags || "");
    drafts.set("song", "instrumental", !!m.instrumental);
    drafts.set("song", "drive", m.drive || "compile");
    drafts.set("song", "title", it.params?.title || "");
    drafts.set("song", "tpl", "");
    drafts.set("song", "dirty", false);
    drafts.set("song.tuning", "bpm", String(m.bpm ?? 120));
    drafts.set("song.tuning", "keyscale", m.key || "E minor");
    setMode("song");
  }
  async function projectRename() {
    if (!current) return;
    const name = window.prompt("Rename project:", current.name);
    if (!name) return;
    try { await api.projectRename(current.id, name); setCurrent({ ...current, name }); refreshProjects(); }
    catch (e) { alert("Rename failed: " + (e as Error).message); }
  }
  async function projectDelete() {
    if (!current) return;
    if (!window.confirm(`Delete project “${current.name}”? Your current workspace stays open.`)) return;
    try { await api.projectDelete(current.id); setCurrent(null); refreshProjects(); }
    catch (e) { alert("Delete failed: " + (e as Error).message); }
  }

  return (
    <div className="flex h-full flex-col">
      <Header cfg={cfg} ace={ace} libOpen={libOpen} toggleLib={() => setLibOpen((v) => !v)} />
      <ProjectBar projects={projects} current={current}
        onNew={projectNew} onOpen={projectOpen} onSave={projectSave} onSaveAs={projectSaveAs}
        onRename={projectRename} onDelete={projectDelete} />
      <main className="flex flex-1 min-h-0">
        <Sidebar mode={mode} setMode={setMode} openGroups={openGroups} setOpenGroups={setOpenGroups} />

        <section className="min-h-0 flex-1 overflow-y-auto p-6">
          <HowItWorks goTo={setMode} />
          <div className="max-w-2xl">
            {cfg ? <Controls mode={mode} cfg={cfg} busy={busy} song={song} setSong={setSong} goTo={setMode} handoff={handoff} setHandoff={setHandoff} {...ctx} />
                 : <p className="mt-6 text-sm text-[var(--color-muted)]">Connecting to backend…</p>}
          </div>
          {results.length > 0 && (
            <div className="mt-6 border-t border-[var(--color-line)] pt-5">
              <Workspace results={results} />
            </div>
          )}
        </section>

        {libOpen && (
          <aside className="w-[520px] flex-none min-h-0 overflow-y-auto border-l border-[var(--color-line)] bg-[var(--color-panel)] p-4">
            <Library items={library}
              onOpen={(it) => setResults([{ id: it.id, title: libDesc(it), status: "done", pct: 100, url: it.audio_url + "?t=" + Date.now() }])}
              onDelete={(id) => api.deleteLib(id).then(refreshLib).catch(() => {})}
              onBucket={(id, b) => api.setBucket(id, b).then(refreshLib).catch(() => {})}
              onOpenInBuilder={loadSongIntoBuilder} />
          </aside>
        )}
      </main>
      <Assistant />
    </div>
  );
}

function Sidebar({ mode, setMode, openGroups, setOpenGroups }: {
  mode: string; setMode: (m: string) => void; openGroups: string[]; setOpenGroups: (f: (g: string[]) => string[]) => void;
}) {
  const toggle = (n: string) => setOpenGroups((g) => (g.includes(n) ? g.filter((x) => x !== n) : [...g, n]));
  return (
    <nav className="w-52 flex-none min-h-0 overflow-y-auto border-r border-[var(--color-line)] bg-[var(--color-panel)] p-2">
      {GROUPS.map((grp) => (
        <div key={grp.name} className="mb-1.5">
          <button onClick={() => toggle(grp.name)}
            className="flex w-full items-center justify-between rounded px-2 py-1.5 text-[11px] font-semibold uppercase tracking-wide text-[var(--color-muted)] hover:text-[var(--color-ink)]">
            <span>{grp.name}</span><span className="text-[9px]">{openGroups.includes(grp.name) ? "▾" : "▸"}</span>
          </button>
          {openGroups.includes(grp.name) && grp.modes.map((id) => (
            <button key={id} onClick={() => setMode(id)}
              className={`mb-0.5 block w-full rounded-lg px-3 py-1.5 text-left text-sm transition ${
                mode === id ? "border border-[var(--color-accent)] bg-[#2a1c19] text-[var(--color-ink)]"
                            : "text-[var(--color-muted)] hover:bg-[var(--color-panel2)] hover:text-[var(--color-ink)]"}`}>
              {LABELS[id]}
            </button>
          ))}
        </div>
      ))}
    </nav>
  );
}

function Controls({ mode, cfg, busy, song, setSong, goTo, handoff, setHandoff, ...ctx }: { mode: string; cfg: Config; busy: boolean; song: SongDraft | null; setSong: (s: SongDraft) => void; goTo: (m: string) => void; handoff: { tags?: string; lyrics?: string } | null; setHandoff: (h: { tags?: string; lyrics?: string } | null) => void } & RunCtx) {
  const p = { cfg, busy, ...ctx };
  switch (mode) {
    case "generate": return <GenerateForm {...p} handoff={handoff} clearHandoff={() => setHandoff(null)} />;
    case "song": return <SongForm {...p} onSong={setSong} onSendToGenerate={(h: { tags?: string; lyrics?: string }) => { setHandoff(h); goTo("generate"); }} />;
    case "vocalbuilder": return <VocalBuilderForm {...p} song={song} />;
    case "import": return <ImportForm goTo={goTo} onImported={p.onDone} />;
    case "restyle": return <RestyleForm {...p} />;
    case "repaint": return <RepaintForm {...p} />;
    case "layer": return <LayerForm {...p} />;
    case "vocals": return <VocalsForm {...p} />;
    case "swap": return <SwapForm {...p} />;
    case "stems": return <StemsForm {...p} />;
    case "tone": return <ToneForm {...p} />;
    case "backing": return <BackingForm {...p} />;
    case "guitar": return <GuitarForm {...p} song={song} />;
    case "master": return <MasterForm {...p} />;
    case "mix": return <MixForm {...p} />;
    default: return null;
  }
}

function shortModel(m?: string): string {
  if (!m) return "";
  return m.replace(/^acestep-v15-/, "").replace(/^acestep-/, "");   // acestep-v15-xl-sft → xl-sft
}

function Header({ cfg, ace, libOpen, toggleLib }: { cfg: Config | null; ace: { reachable: boolean; model?: string } | null; libOpen: boolean; toggleLib: () => void }) {
  return (
    <header className="flex items-center justify-between border-b border-[var(--color-line)] px-5 py-3">
      <div className="flex items-baseline gap-2">
        <span className="bg-gradient-to-r from-[var(--color-accent)] to-[var(--color-accent2)] bg-clip-text text-lg font-bold tracking-tight text-transparent">Crucible</span>
        <span className="text-xs text-[var(--color-muted)]">AI metal studio</span>
      </div>
      <div className="flex items-center gap-2 text-[11px]">
        <Chip label={`ComfyUI ${cfg ? "· " + cfg.comfy_host : "…"}`} ok={!!cfg} />
        <Chip label={`RVC ${cfg ? "· " + cfg.rvc_driver : "…"}`} ok={!!cfg && cfg.rvc_driver !== "gradio"} />
        {cfg?.acestep && (
          <Chip ok={!!ace?.reachable}
            label={`ACE ${ace ? (ace.reachable ? "· " + (shortModel(ace.model) || "loading…") : "· off") : "…"}`} />
        )}
        <button onClick={toggleLib}
          className={`rounded-lg border px-2.5 py-1 transition ${libOpen ? "border-[var(--color-accent)] bg-[#2a1c19] text-[var(--color-ink)]" : "border-[var(--color-line)] bg-[var(--color-panel2)] text-[var(--color-muted)] hover:text-[var(--color-ink)]"}`}>
          {libOpen ? "Library ▸" : "◂ Library"}
        </button>
      </div>
    </header>
  );
}

function ProjectBar({ projects, current, onNew, onOpen, onSave, onSaveAs, onRename, onDelete }: {
  projects: Project[]; current: { id: string; name: string } | null;
  onNew: () => void; onOpen: (id: string) => void; onSave: () => void; onSaveAs: () => void;
  onRename: () => void; onDelete: () => void;
}) {
  const btn = "rounded-md border border-[var(--color-line)] bg-[var(--color-panel2)] px-2.5 py-1 text-[var(--color-muted)] transition hover:border-[var(--color-accent)] hover:text-[var(--color-ink)] disabled:opacity-40 disabled:hover:border-[var(--color-line)]";
  return (
    <div className="flex flex-wrap items-center gap-1.5 border-b border-[var(--color-line)] bg-[var(--color-panel)] px-5 py-2 text-[11px]">
      <span className="text-[var(--color-muted)]">Project</span>
      <span className="mr-1 rounded-md bg-[#2a1c19] px-2 py-1 font-medium text-[var(--color-accent2)]" title={current ? "current project" : "nothing saved yet — Save to name it"}>
        {current ? current.name : "Unsaved"}
      </span>
      <select className={`${btn} max-w-[180px]`} value="" title="Open a saved project"
        onChange={(e) => { onOpen(e.target.value); e.currentTarget.value = ""; }}>
        <option value="">Open…</option>
        {projects.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
      </select>
      <button className={btn} onClick={onNew} title="Clear every page and start fresh">New</button>
      <button className={btn} onClick={onSave} title="Save the current project">Save</button>
      <button className={btn} onClick={onSaveAs} title="Save as a new named project">Save As</button>
      <button className={btn} onClick={onRename} disabled={!current}>Rename</button>
      <button className={btn} onClick={onDelete} disabled={!current}>Delete</button>
    </div>
  );
}

function Chip({ label, ok }: { label: string; ok?: boolean }) {
  return (
    <span className="flex items-center gap-1.5 rounded-full border border-[var(--color-line)] bg-[var(--color-panel2)] px-2.5 py-1 text-[var(--color-muted)]">
      <span className={`h-1.5 w-1.5 rounded-full ${ok ? "bg-emerald-400" : "bg-neutral-500"}`} />{label}
    </span>
  );
}

function HowItWorks({ goTo }: { goTo: (m: string) => void }) {
  const [open, setOpen] = useState(false);
  const Step = ({ m, label, children }: { m: string; label: string; children: React.ReactNode }) => (
    <li>
      <button onClick={() => goTo(m)} className="font-semibold text-[var(--color-ink)] hover:text-[var(--color-accent2)]">{label}</button>
      <span className="text-[var(--color-muted)]"> — {children}</span>
    </li>
  );
  return (
    <div className="mb-4 rounded-xl border border-[var(--color-line)] bg-[var(--color-panel2)] p-2.5">
      <button onClick={() => setOpen(!open)} className="flex w-full items-center gap-1.5 text-xs font-medium text-[var(--color-accent2)]">
        <span className="text-[var(--color-muted)]">{open ? "▾" : "▸"}</span> How it all fits together
      </button>
      {open && (
        <ol className="mt-2 space-y-1.5 text-[11px] leading-relaxed">
          <Step m="song" label="1 · Song">arrange sections, lyrics, key &amp; BPM.</Step>
          <Step m="vocalbuilder" label="2 · Voc. Builder">AI composes a melody from that song and sings it (SoulX); optionally re-timbre with RVC.</Step>
          <Step m="import" label="3 · Import">turn any song into a custom SoulX voice for step 2.</Step>
          <Step m="mix" label="4 · Mix">layer the vocal over an instrumental into a finished track.</Step>
          <li className="text-[#5a5f6e]">Supporting: Generate / Restyle (instrumentals) · Stems (split a track) · Voice Swap (re-sing an existing track).</li>
        </ol>
      )}
    </div>
  );
}

function Workspace({ results }: { results: Result[] }) {
  if (results.length === 0) {
    return (
      <div className="flex h-full flex-col">
        <h2 className="mb-4 text-sm font-semibold text-[var(--color-muted)]">Workspace</h2>
        <div className="flex flex-1 items-center justify-center text-sm text-[var(--color-muted)]">Run something to see the result here.</div>
      </div>
    );
  }
  const grid = results.length > 1;
  return (
    <div>
      <h2 className="mb-4 text-sm font-semibold text-[var(--color-muted)]">
        Workspace {grid && <span className="text-[var(--color-muted)]">· {results.length} takes — audition & compare</span>}
      </h2>
      <div className={grid ? "grid grid-cols-1 gap-3 xl:grid-cols-2" : ""}>
        {results.map((r) => <ResultCard key={r.id} r={r} />)}
      </div>
    </div>
  );
}

function ResultCard({ r }: { r: Result }) {
  return (
    <div className="rounded-xl border border-[var(--color-line)] bg-[var(--color-panel2)] p-4">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-xs font-semibold text-[var(--color-accent2)]">{r.title}</span>
        <span className="text-[10px] text-[var(--color-muted)]">{r.status}</span>
      </div>
      {r.status === "done" && r.url ? (
        <WavePlayer url={r.url} />
      ) : r.status === "error" ? (
        <p className="text-xs text-red-400">{r.err}</p>
      ) : (
        <div className="h-2 overflow-hidden rounded-full bg-[var(--color-panel)]">
          <div className="h-full rounded-full bg-gradient-to-r from-[var(--color-accent)] to-[var(--color-accent2)] transition-all" style={{ width: `${r.pct}%` }} />
        </div>
      )}
    </div>
  );
}

function libDesc(it: LibItem): string {
  const p = it.params || {};
  // An explicit note (e.g. an A/B label) always wins — so tagged comparison takes
  // are identifiable in the library instead of all showing the same prompt.
  if (p.note) return String(p.note);
  if (p.source === "vocal-builder") {
    const o = p.opts || {};
    const parts = [
      (p.engine || "").toUpperCase(),
      o.n_steps ? `${o.n_steps} steps` : "",
      o.fp16 === false ? "fp32" : "fp16",
      o.cfg !== undefined ? `cfg ${o.cfg}` : "",
      o.auto_shift === false ? "no-shift" : "",
      p.voice ? `→ ${p.voice}` : "",
      p.key,
    ].filter(Boolean);
    return "Vocal Builder · " + parts.join(" · ");
  }
  if (it.mode === "stem") return `${p.kind || "stem"}${p.source ? " · from " + String(p.source).slice(0, 28) : ""}`;
  if (it.mode === "source") return String(p.source || p.title || "imported audio").slice(0, 40);
  if (it.mode === "tone") return `tone: ${p.preset || "?"}${p.source ? " · from " + String(p.source).slice(0, 24) : ""}`;
  if (it.mode === "master") return `mastered${p.source ? " · " + String(p.source).slice(0, 24) : ""}`;
  if (it.mode === "repaint") return `repaint: ${p.tags ? String(p.tags).slice(0, 28) : "?"} · [${p.repaint_start ?? "?"}–${p.repaint_end ?? "?"}s]`;
  if (it.mode === "cover") { const st = p.cover_strength != null ? " str" + p.cover_strength : (p.cover_cfg ? " cfg" + p.cover_cfg : ""); const tk = p.take ? " · take " + p.take : ""; return `cover${st}${tk}: ${p.tags ? String(p.tags).slice(0, 20) : "?"}${p.source ? " · " + String(p.source).slice(0, 12) : ""}`; }
  if (it.mode === "layer") return `layer: ${p.track_name || "?"}${p.tags ? " · " + String(p.tags).slice(0, 22) : ""}`;
  if (it.mode === "layerstem") return `${p.track_name || "?"} stem (${p.method || "?"})${p.source ? " · from " + String(p.source).slice(0, 18) : ""}`;
  if (it.mode === "extend") return `extend +${p.extend_left || 0}/${p.extend_right || 0}s · ${p.tags ? String(p.tags).slice(0, 24) : "?"}`;
  if (it.mode === "backing") return `guitar-less backing${p.source ? " · from " + String(p.source).slice(0, 22) : ""}`;
  if (it.mode === "guitar") return `amped guitar (${p.preset || "?"})${p.source ? " · " + String(p.source).slice(0, 22) : ""}`;
  if (it.mode === "guitardi") return `clean guitar DI${p.source ? " · " + String(p.source).slice(0, 24) : ""}`;
  const genlike = it.mode === "generate" || it.mode === "restyle";
  const variant = genlike && p.variant ? ` · ${p.variant}` : "";
  const cs = genlike && (p.cfg || p.steps) ? ` · cfg${p.cfg ?? "?"}/${p.steps ?? "?"}st` : "";
  const apg = genlike && p.apg ? " · APG" : "";
  const neg = genlike && (p.negative_tags || "").trim() ? " · NEG" : "";
  const smp = genlike && p.sampler_name ? ` · ${p.sampler_name}/${p.scheduler || "simple"}` : "";
  return (p.tags || p.voice || p.source || "—") + variant + smp + cs + apg + neg;
}

function hhmm(epoch?: number): string {
  if (!epoch) return "";
  return new Date(epoch * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

const LIB_SECTIONS = [
  { key: "song", label: "Songs" },
  { key: "generate", label: "Generated" },
  { key: "vocal", label: "Vocals" },
  { key: "voiceswap", label: "Voice swaps" },
  { key: "mix", label: "Mixes" },
  { key: "restyle", label: "Restyled" },
  { key: "cover", label: "Covers" },
  { key: "repaint", label: "Repainted" },
  { key: "layer", label: "Added layers" },
  { key: "layerstem", label: "Layer stems" },
  { key: "extend", label: "Extended" },
  { key: "tone", label: "Re-toned" },
  { key: "backing", label: "Backing (no guitar)" },
  { key: "guitar", label: "Amped guitar" },
  { key: "guitardi", label: "Guitar DI" },
  { key: "master", label: "Mastered" },
  { key: "stem", label: "Stems" },
  { key: "source", label: "Source audio" },
];

type LibActions = { onOpen: (it: LibItem) => void; onDelete: (id: string) => void; onBucket: (id: string, b: string) => void; onOpenInBuilder?: (it: LibItem) => void };

function libTitle(it: LibItem): string {
  const t = it.params?.title;
  return (t && String(t).trim()) || libDesc(it);
}

// Artwork header — shows a generated cover image when present (future image-gen),
// else a typed gradient placeholder. Kept compact so the grid stays dense.
function LibArtwork({ it }: { it: LibItem }) {
  const url = it.params?.artwork_url;
  if (url) return <img src={url} alt="" className="aspect-[16/9] w-full object-cover" />;
  return (
    <div className="flex aspect-[16/9] w-full items-center justify-center bg-gradient-to-br from-[#2a1c19] via-[var(--color-panel2)] to-[var(--color-panel)]">
      <span className="text-[10px] font-medium uppercase tracking-wider text-[var(--color-muted)]">{it.mode}</span>
    </div>
  );
}

// One card per "group" (a song + its versions, or a single track). Versions share a base
// title; chips switch which take the card shows/acts on (newest selected by default).
function LibCard({ group, inTests, onOpen, onDelete, onBucket, onOpenInBuilder }: { group: LibItem[]; inTests: boolean } & LibActions) {
  const [vi, setVi] = useState(group.length - 1);
  const it = group[Math.min(vi, group.length - 1)] || group[0];
  const multi = group.length > 1;
  const canBuild = !!(it.params?.from_builder && it.params?.song_meta && onOpenInBuilder);
  return (
    <div className="overflow-hidden rounded-xl border border-[var(--color-line)] bg-[var(--color-panel2)]">
      <LibArtwork it={it} />
      <div className="space-y-1.5 p-2.5">
        <div className="flex flex-wrap items-center gap-1.5 text-[10px]">
          <span className="rounded bg-[#2a1c19] px-1.5 py-0.5 text-[var(--color-accent2)]">{it.mode}</span>
          <span className="text-[var(--color-muted)]">{hhmm(it.created)}</span>
          {canBuild && <button onClick={() => onOpenInBuilder!(it)} className="ml-auto text-[var(--color-muted)] hover:text-[var(--color-accent2)]" title="Open in Song Builder (load arrangement, lyrics & tags)">↻</button>}
          <button onClick={() => onOpen(it)} className={`${canBuild ? "" : "ml-auto"} text-[var(--color-muted)] hover:text-[var(--color-accent2)]`} title="Open in workspace">↗</button>
          <a href={`/api/export/${it.id}?fmt=mp3`} download className="text-[var(--color-muted)] hover:text-[var(--color-accent2)]" title="Export as MP3 (320k)">⬇</a>
          <button onClick={() => onBucket(it.id, inTests ? "" : "tests")} className="text-[var(--color-muted)] hover:text-[var(--color-ink)]" title={inTests ? "Restore from Tests" : "Move to Tests"}>{inTests ? "↩" : "🧪"}</button>
          <button onClick={() => { if (confirm("Delete this track permanently?")) onDelete(it.id); }} className="text-[var(--color-muted)] hover:text-red-400" title="Delete">✕</button>
        </div>
        <p className="line-clamp-2 text-[11px] font-medium text-[var(--color-ink)]" title={libTitle(it)}>{libTitle(it)}</p>
        {multi && (
          <div className="flex flex-wrap gap-1">
            {group.map((g, i) => (
              <button key={g.id} onClick={() => setVi(i)} title={hhmm(g.created)}
                className={`rounded px-1.5 py-0.5 text-[10px] ${i === Math.min(vi, group.length - 1) ? "bg-[var(--color-accent)] text-white" : "bg-[var(--color-panel)] text-[var(--color-muted)] hover:text-[var(--color-ink)]"}`}>v{i + 1}</button>
            ))}
          </div>
        )}
        <audio className="h-8 w-full" controls src={it.audio_url} />
      </div>
    </div>
  );
}

function Library({ items, onOpenInBuilder, ...a }: { items: LibItem[] } & LibActions) {
  const [tab, setTab] = useState("");
  const [q, setQ] = useState("");
  const [sort, setSort] = useState<"new" | "old" | "name">("new");

  const tests = items.filter((i) => i.bucket === "tests");
  const live = items.filter((i) => i.bucket !== "tests");
  const sectionsWith = LIB_SECTIONS.filter((s) => live.some((i) => i.mode === s.key));
  const other = live.filter((i) => !LIB_SECTIONS.some((s) => s.key === i.mode));
  const tabs = [{ key: "__all", label: "All" }, ...sectionsWith.map((s) => ({ key: s.key, label: s.label })),
    ...(other.length ? [{ key: "__other", label: "Other" }] : []),
    ...(tests.length ? [{ key: "__tests", label: "Tests" }] : [])];
  const active = tabs.some((t) => t.key === tab) ? tab : (tabs[0]?.key || "__all");
  const inTests = active === "__tests";

  let pool = active === "__all" ? live : active === "__tests" ? tests : active === "__other" ? other : live.filter((i) => i.mode === active);
  if (q.trim()) {
    const qq = q.toLowerCase();
    pool = pool.filter((i) => (libTitle(i) + " " + (i.params?.tags || "")).toLowerCase().includes(qq));
  }
  pool = [...pool].sort((x, y) => sort === "name" ? libTitle(x).localeCompare(libTitle(y)) : sort === "old" ? x.created - y.created : y.created - x.created);

  // Group versions: items sharing a base title (+ same type) collapse into one card.
  const gmap = new Map<string, LibItem[]>();
  for (const it of pool) {
    const t = it.params?.title && String(it.params.title).trim();
    const key = t ? `t:${it.mode}:${t.toLowerCase()}` : `id:${it.id}`;
    (gmap.get(key) || gmap.set(key, []).get(key)!).push(it);
  }
  const groups = [...gmap.values()].map((g) => [...g].sort((m, n) => m.created - n.created));   // oldest→newest = v1..vN

  return (
    <div className="flex h-full flex-col">
      <div className="mb-2 flex items-center gap-2">
        <h2 className="text-sm font-semibold">Library</h2>
        <select value={sort} onChange={(e) => setSort(e.target.value as any)}
          className="ml-auto rounded-md border border-[var(--color-line)] bg-[var(--color-panel2)] px-1.5 py-1 text-[11px] text-[var(--color-muted)]">
          <option value="new">Newest</option><option value="old">Oldest</option><option value="name">Name</option>
        </select>
      </div>
      <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="search title / tags…"
        className="mb-2 w-full rounded-lg border border-[var(--color-line)] bg-[var(--color-panel2)] px-2.5 py-1.5 text-xs text-[var(--color-ink)] outline-none focus:border-[var(--color-accent)]" />
      <div className="-mx-1 mb-2 flex flex-none gap-1 overflow-x-auto px-1 pb-1">
        {tabs.map((t) => (
          <button key={t.key} onClick={() => setTab(t.key)}
            className={`whitespace-nowrap rounded-full border px-2.5 py-1 text-[11px] transition ${active === t.key ? "border-[var(--color-accent)] bg-[#2a1c19] text-[var(--color-ink)]" : "border-[var(--color-line)] bg-[var(--color-panel2)] text-[var(--color-muted)] hover:text-[var(--color-ink)]"}`}>{t.label}</button>
        ))}
      </div>
      {items.length === 0 ? (
        <p className="text-xs text-[var(--color-muted)]">No tracks yet.</p>
      ) : groups.length === 0 ? (
        <p className="text-xs text-[var(--color-muted)]">Nothing matches.</p>
      ) : (
        <div className="grid gap-3 [grid-template-columns:repeat(auto-fill,minmax(190px,1fr))]">
          {groups.map((g) => <LibCard key={g[g.length - 1].id} group={g} inTests={inTests} onOpenInBuilder={onOpenInBuilder} {...a} />)}
        </div>
      )}
    </div>
  );
}
