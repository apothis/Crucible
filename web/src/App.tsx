import { useEffect, useMemo, useState } from "react";
import { api, type Config, type LibItem, type SongDraft } from "./api";
import { GenerateForm, RestyleForm, VocalsForm, SwapForm, StemsForm, ToneForm, BackingForm, GuitarForm, MasterForm, MixForm, SongForm } from "./forms";
import { VocalBuilderForm } from "./VocalBuilder";
import { ImportForm } from "./Import";
import { Assistant } from "./Assistant";
import { WavePlayer } from "./WavePlayer";
import type { Result, RunCtx } from "./ui";

const MODES = [
  { id: "generate", label: "Generate" },
  { id: "song", label: "Song" },
  { id: "vocalbuilder", label: "Voc. Builder" },
  { id: "import", label: "Import" },
  { id: "restyle", label: "Restyle" },
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
  { name: "Create", modes: ["generate", "song", "restyle"] },
  { name: "Guitar", modes: ["backing", "guitar", "tone"] },
  { name: "Vocals", modes: ["vocalbuilder", "vocals", "swap", "import"] },
  { name: "Finish", modes: ["stems", "mix", "master"] },
];
const LABELS: Record<string, string> = Object.fromEntries(MODES.map((m) => [m.id, m.label]));

export default function App() {
  const [cfg, setCfg] = useState<Config | null>(null);
  const [mode, setMode] = useState<string>("generate");
  const [library, setLibrary] = useState<LibItem[]>([]);
  const [results, setResults] = useState<Result[]>([]);
  const [song, setSong] = useState<SongDraft | null>(null);
  const [handoff, setHandoff] = useState<{ tags?: string; lyrics?: string } | null>(null);
  const [libOpen, setLibOpen] = useState(true);
  const [openGroups, setOpenGroups] = useState<string[]>(GROUPS.map((g) => g.name));

  const refreshLib = () => api.library().then(setLibrary).catch(() => {});
  useEffect(() => { api.config().then(setCfg).catch(() => {}); refreshLib(); }, []);
  useEffect(() => { setResults([]); }, [mode]);

  const busy = useMemo(() => results.some((r) => r.status === "pending" || r.status === "running"), [results]);
  const ctx: RunCtx = {
    setResults,
    patch: (id, p) => setResults((rs) => rs.map((r) => (r.id === id ? { ...r, ...p } : r))),
    onDone: refreshLib,
  };

  return (
    <div className="flex h-full flex-col">
      <Header cfg={cfg} libOpen={libOpen} toggleLib={() => setLibOpen((v) => !v)} />
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
          <aside className="w-[360px] flex-none min-h-0 overflow-y-auto border-l border-[var(--color-line)] bg-[var(--color-panel)] p-4">
            <Library items={library}
              onOpen={(it) => setResults([{ id: it.id, title: libDesc(it), status: "done", pct: 100, url: it.audio_url + "?t=" + Date.now() }])}
              onDelete={(id) => api.deleteLib(id).then(refreshLib).catch(() => {})}
              onBucket={(id, b) => api.setBucket(id, b).then(refreshLib).catch(() => {})} />
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
    case "import": return <ImportForm goTo={goTo} />;
    case "restyle": return <RestyleForm {...p} />;
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

function Header({ cfg, libOpen, toggleLib }: { cfg: Config | null; libOpen: boolean; toggleLib: () => void }) {
  return (
    <header className="flex items-center justify-between border-b border-[var(--color-line)] px-5 py-3">
      <div className="flex items-baseline gap-2">
        <span className="bg-gradient-to-r from-[var(--color-accent)] to-[var(--color-accent2)] bg-clip-text text-lg font-bold tracking-tight text-transparent">Crucible</span>
        <span className="text-xs text-[var(--color-muted)]">AI metal studio</span>
      </div>
      <div className="flex items-center gap-2 text-[11px]">
        <Chip label={`ComfyUI ${cfg ? "· " + cfg.comfy_host : "…"}`} ok={!!cfg} />
        <Chip label={`RVC ${cfg ? "· " + cfg.rvc_driver : "…"}`} ok={!!cfg && cfg.rvc_driver !== "gradio"} />
        <button onClick={toggleLib}
          className={`rounded-lg border px-2.5 py-1 transition ${libOpen ? "border-[var(--color-accent)] bg-[#2a1c19] text-[var(--color-ink)]" : "border-[var(--color-line)] bg-[var(--color-panel2)] text-[var(--color-muted)] hover:text-[var(--color-ink)]"}`}>
          {libOpen ? "Library ▸" : "◂ Library"}
        </button>
      </div>
    </header>
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
  if (it.mode === "backing") return `guitar-less backing${p.source ? " · from " + String(p.source).slice(0, 22) : ""}`;
  if (it.mode === "guitar") return `amped guitar (${p.preset || "?"})${p.source ? " · " + String(p.source).slice(0, 22) : ""}`;
  if (it.mode === "guitardi") return `clean guitar DI${p.source ? " · " + String(p.source).slice(0, 24) : ""}`;
  const genlike = it.mode === "generate" || it.mode === "restyle";
  const variant = genlike && p.variant ? ` · ${p.variant}` : "";
  const cs = genlike && (p.cfg || p.steps) ? ` · cfg${p.cfg ?? "?"}/${p.steps ?? "?"}st` : "";
  const neg = genlike && (p.negative_tags || "").trim() ? " · NEG" : "";
  const smp = genlike && p.sampler_name ? ` · ${p.sampler_name}/${p.scheduler || "simple"}` : "";
  return (p.tags || p.voice || p.source || "—") + variant + smp + cs + neg;
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
  { key: "tone", label: "Re-toned" },
  { key: "backing", label: "Backing (no guitar)" },
  { key: "guitar", label: "Amped guitar" },
  { key: "guitardi", label: "Guitar DI" },
  { key: "master", label: "Mastered" },
  { key: "stem", label: "Stems" },
  { key: "source", label: "Source audio" },
];

type LibActions = { onOpen: (it: LibItem) => void; onDelete: (id: string) => void; onBucket: (id: string, b: string) => void };

function LibCard({ it, inTests, onOpen, onDelete, onBucket }: { it: LibItem; inTests: boolean } & LibActions) {
  return (
    <div className="rounded-xl border border-[var(--color-line)] bg-[var(--color-panel2)] p-3">
      <div className="mb-2 flex flex-wrap items-center gap-1.5 text-[10px]">
        <span className="rounded bg-[#2a1c19] px-1.5 py-0.5 text-[var(--color-accent2)]">{it.mode}</span>
        <span className="text-[var(--color-muted)]">{hhmm(it.created)}</span>
        <button onClick={() => onOpen(it)} className="ml-auto text-[var(--color-muted)] hover:text-[var(--color-accent2)]" title="Open in workspace">↗</button>
        <button onClick={() => onBucket(it.id, inTests ? "" : "tests")} className="text-[var(--color-muted)] hover:text-[var(--color-ink)]" title={inTests ? "Restore from Tests" : "Move to Tests"}>{inTests ? "↩" : "🧪"}</button>
        <button onClick={() => { if (confirm("Delete this track permanently?")) onDelete(it.id); }} className="text-[var(--color-muted)] hover:text-red-400" title="Delete">✕</button>
      </div>
      <p className="mb-2 line-clamp-2 text-[11px] text-[var(--color-muted)]">{libDesc(it)}</p>
      <audio className="h-8 w-full" controls src={it.audio_url} />
    </div>
  );
}

function LibSection({ label, items, defaultOpen, inTests, ...a }: { label: string; items: LibItem[]; defaultOpen: boolean; inTests?: boolean } & LibActions) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div>
      <button onClick={() => setOpen(!open)} className="flex w-full items-center gap-1.5 py-1.5 text-xs font-semibold text-[var(--color-ink)]">
        <span className="text-[var(--color-muted)]">{open ? "▾" : "▸"}</span>{label}
        <span className="text-[10px] font-normal text-[var(--color-muted)]">({items.length})</span>
      </button>
      {open && <div className="mb-2 space-y-2.5">{items.map((it) => <LibCard key={it.id} it={it} inTests={!!inTests} {...a} />)}</div>}
    </div>
  );
}

function Library({ items, ...a }: { items: LibItem[] } & LibActions) {
  const tests = items.filter((i) => i.bucket === "tests");
  const live = items.filter((i) => i.bucket !== "tests");
  const groups = LIB_SECTIONS.map((s) => ({ ...s, items: live.filter((i) => i.mode === s.key) })).filter((g) => g.items.length);
  const other = live.filter((i) => !LIB_SECTIONS.some((s) => s.key === i.mode));
  return (
    <div>
      <h2 className="mb-2 text-sm font-semibold">Library</h2>
      {items.length === 0 && <p className="text-xs text-[var(--color-muted)]">No tracks yet.</p>}
      {groups.map((g) => <LibSection key={g.key} label={g.label} items={g.items} defaultOpen {...a} />)}
      {other.length > 0 && <LibSection label="Other" items={other} defaultOpen={false} {...a} />}
      {tests.length > 0 && <LibSection label="Tests" items={tests} defaultOpen={false} inTests {...a} />}
    </div>
  );
}
