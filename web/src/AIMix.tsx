import { useEffect, useRef, useState } from "react";
import { api } from "./api";
import { useDrafts } from "./drafts";
import { inp, PrimaryButton, rid, type RunCtx } from "./ui";
import { Console } from "./AIMixPanel";
import { GROUPS, clone, colorOf, type MixDoc } from "./aimixModel";

// AI Mix (docs/AI_MIX_PLAN.md): a new mix built FROM a Suno stem download. The AI measures every
// stem group against a style preset made from real reference songs and moves each processor only
// as far as needed; the console then gives full manual control over the result. The full mix in
// the zip is only the reference (balance, loudness target, A/B baseline).

type Render = { job_id: string; label: string; audio_url: string };
type Session = { session: string; title: string; preset?: string; updated: number; renders: Render[] };
type PresetInfo = { name: string; label: string; n: number };
type Taste = { picks: { job_id: string; title?: string }[]; vocals: Record<string, number> };
type RunRes = { session: string; title: string; tracks: Render[];
  summary: { fit_r: number | null; bpm: number | null; silent: string[]; mix_lufs: number | null;
    preset: string | null; preset_scores: Record<string, number> | null; notes: string[] } };

const PROFILES = [
  { id: "rock", label: "Rock/metal: vocal + drums forward" },
  { id: "vocal", label: "Vocal-forward" },
  { id: "suno", label: "Suno balance (no level changes)" },
];

const card = "rounded-[14px] border border-[var(--color-line)] bg-[var(--color-panel)] p-4";
const h = "text-[15px] font-semibold text-[var(--color-ink)]";
const hint = "text-[11px] text-[var(--color-muted)]";

function Chip({ k, v }: { k: string; v: string }) {
  return <span className="rounded-full border border-[var(--color-line)] bg-[var(--color-panel2)] px-2.5 py-1 text-[11px]"><span className="text-[var(--color-muted)]">{k}</span> <b className="text-[var(--color-ink)]">{v}</b></span>;
}

export function AIMixForm({ busy, setCompare, goTo, ...ctx }: { busy: boolean; setCompare: (f: (ids: string[]) => string[]) => void; goTo: (m: string) => void } & RunCtx) {
  const d = useDrafts("aimix");
  const [path, setPath] = d.use("path", "");          // server-side path of the uploaded stems
  const [picked, setPicked] = d.use("picked", "");      // what the user chose, for display
  const fileRef = useRef<HTMLInputElement>(null);
  const [upPct, setUpPct] = useState<number | null>(null);
  const [profile, setProfile] = d.use("profile", "rock");
  const [preset, setPreset] = d.use("preset", "auto");
  const [presets, setPresets] = useState<PresetInfo[]>([]);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [sid, setSid] = useState("");
  const [doc, setDoc] = useState<MixDoc | null>(null);
  const [ai, setAi] = useState<MixDoc | null>(null);
  const [summary, setSummary] = useState<RunRes["summary"] | null>(null);
  const [running, setRunning] = useState(false);
  const [rendering, setRendering] = useState(false);
  const [label, setLabel] = useState("");
  const [cmp, setCmp] = useState<string[]>([]);
  const [showNotes, setShowNotes] = useState(false);
  const [taste, setTaste] = useState<Taste | null>(null);
  const [starring, setStarring] = useState("");

  const usePreset = presets.length > 0 && preset !== "";
  const loadSessions = () => api.aimixSessions().then((r) => setSessions((r as { sessions: Session[] }).sessions)).catch(() => setSessions([]));
  const openSession = (s: string, sum: RunRes["summary"] | null = null) => {
    setSid(s); setSummary(sum); setCmp([]);
    if (!s) { setDoc(null); setAi(null); return; }
    api.aimixGet(s).then((r) => { const x = r as { mixdoc: MixDoc; mixdoc_ai: MixDoc }; setDoc(x.mixdoc); setAi(x.mixdoc_ai); })
      .catch(() => setDoc(null));
  };
  useEffect(() => {
    api.aimixPresets().then((r) => setPresets((r as { presets: PresetInfo[] }).presets)).catch(() => setPresets([]));
    api.aimixTaste().then((r) => setTaste(r as Taste)).catch(() => setTaste(null));
    api.aimixSessions().then((r) => {
      const list = (r as { sessions: Session[] }).sessions;
      setSessions(list);
      if (list[0]) {
        setSid(list[0].session);
        api.aimixGet(list[0].session).then((x) => { const y = x as { mixdoc: MixDoc; mixdoc_ai: MixDoc }; setDoc(y.mixdoc); setAi(y.mixdoc_ai); }).catch(() => setDoc(null));
      }
    }).catch(() => setSessions([]));
  }, []);

  async function run() {
    if (!path.trim()) { ctx.setResults([{ id: rid(), title: "Choose the Suno stems first (the zip, or the stem WAVs).", status: "error", pct: 0 }]); return; }
    const id = rid();
    setRunning(true);
    ctx.setResults([{ id, title: "AI mix: measuring stems against the style + rendering (Mac, about a minute)…", status: "running", pct: 40 }]);
    try {
      const r = await api.aimixRun({ path: path.trim(), profile, preset: usePreset ? preset : "" }) as RunRes;
      ctx.setResults(r.tracks.map((t) => ({ id: rid(), title: t.label, status: "done" as const, pct: 100, url: t.audio_url })));
      ctx.onDone();
      await loadSessions();
      openSession(r.session, r.summary);
      if (/\/aimix\/uploads\/.*\.zip$/i.test(path)) { setPath(""); setPicked(""); }   // upload copy is removed once unpacked
      setCmp(r.tracks.map((t) => t.job_id));
    } catch (e) { ctx.patch(id, { status: "error", pct: 0, err: (e as Error).message }); }
    finally { setRunning(false); }
  }

  // Upload the chosen zip (or stem files) one file per request, streamed into the session area;
  // XHR for a real progress bar on big stem packs.
  function sendFile(f: File, batch: string, done: number, total: number): Promise<{ batch: string; path: string }> {
    return new Promise((resolve, reject) => {
      const x = new XMLHttpRequest();
      x.open("POST", `/api/aimix/upload?name=${encodeURIComponent(f.name)}&batch=${batch}`);
      x.upload.onprogress = (e) => { if (e.lengthComputable) setUpPct(Math.round(((done + e.loaded) / total) * 100)); };
      x.onload = () => x.status === 200 ? resolve(JSON.parse(x.responseText)) : reject(new Error(x.responseText || `upload failed (${x.status})`));
      x.onerror = () => reject(new Error("upload failed"));
      x.send(f);
    });
  }
  async function choose(files: FileList | null) {
    const list = Array.from(files || []).filter((f) => /\.(zip|wav|flac|mp3)$/i.test(f.name));
    if (!list.length) return;
    const zip = list.find((f) => /\.zip$/i.test(f.name));
    const send = zip ? [zip] : list;
    const total = send.reduce((a, f) => a + f.size, 0);
    setUpPct(0); setPath("");
    try {
      let batch = "", last = { batch: "", path: "" }, done = 0;
      for (const f of send) { last = await sendFile(f, batch, done, total); batch = last.batch; done += f.size; }
      setPath(last.path);
      setPicked(zip ? zip.name : send.length === 1 ? send[0].name : `${send.length} stem files`);
    } catch (e) { ctx.setResults([{ id: rid(), title: "Upload failed", status: "error", pct: 0, err: (e as Error).message }]); }
    finally { setUpPct(null); if (fileRef.current) fileRef.current.value = ""; }
  }

  async function render() {
    if (!doc || !sid) return;
    const lab = label.trim() || "manual edit";
    const id = rid();
    setRendering(true);
    ctx.setResults([{ id, title: `Rendering "${lab}"… (Mac, ~30 s)`, status: "running", pct: 40 }]);
    try {
      const r = await api.aimixRender(sid, { mixdoc: doc, label: lab }) as { track: Render };
      ctx.setResults([{ id: rid(), title: r.track.label, status: "done", pct: 100, url: r.track.audio_url }]);
      ctx.onDone();
      setCmp((c) => [...c, r.track.job_id]);
      loadSessions();
    } catch (e) { ctx.patch(id, { status: "error", pct: 0, err: (e as Error).message }); }
    finally { setRendering(false); }
  }

  async function star(jobId: string, on: boolean) {
    if (!sid) return;
    setStarring(jobId);
    try { setTaste(await api.aimixPrefer(sid, { job_id: jobId, remove: !on }) as Taste); }
    catch (e) { ctx.setResults([{ id: rid(), title: "Couldn't star that version", status: "error", pct: 0, err: (e as Error).message }]); }
    finally { setStarring(""); }
  }
  const starred = new Set((taste?.picks || []).map((p) => p.job_id));
  const tasteUsed = Array.isArray(doc?.provenance) && (doc?.provenance as { layer: string }[]).some((p) => p.layer === "taste");

  const cur = sessions.find((s) => s.session === sid);
  const notes = doc?.notes || summary?.notes || [];
  const byGroup = GROUPS.map((g) => ({ g, items: notes.filter((t) => t.startsWith(`${g.id}:`) || (g.id === "vocals" && /^(Vocals|Backing_Vocals):/.test(t))) }))
    .filter((x) => x.items.length);
  const dirty = !!(doc && ai && JSON.stringify(doc.stems) + JSON.stringify(doc.buses) !== JSON.stringify(ai.stems) + JSON.stringify(ai.buses));

  return (
    <div className="@container">
    <div className="grid grid-cols-1 items-start gap-4 @5xl:grid-cols-[320px_minmax(0,1fr)] @5xl:grid-rows-[auto_auto_1fr]">
        <div className={`${card} @5xl:col-start-1 @5xl:row-start-1`}>
          <h2 className={h}>New AI mix</h2>
          <p className={`${hint} mb-3`}>Choose a Suno stem download: the zip, or all the stem WAVs. The AI measures every stem group against a style made from real reference songs and only fixes what is outside that style.</p>
          <input ref={fileRef} type="file" multiple accept=".zip,.wav,.flac,.mp3" className="hidden" onChange={(e) => choose(e.target.files)} />
          <div className="mb-3">
            <span className="mb-1.5 block text-xs text-[var(--color-muted)]">Suno stems</span>
            <div className="flex items-center gap-2">
              <button onClick={() => fileRef.current?.click()} disabled={upPct !== null || running}
                className="flex-none rounded-lg border border-[var(--color-line)] bg-[var(--color-panel2)] px-3 py-2 text-xs text-[var(--color-ink)] transition hover:border-[var(--color-accent)] disabled:opacity-50">
                {picked ? "Change…" : "Choose stems…"}
              </button>
              <span className={`min-w-0 flex-1 truncate text-xs ${path ? "text-[var(--color-ink)]" : "text-[var(--color-muted)]"}`} title={picked}>
                {upPct !== null ? `Uploading… ${upPct}%` : path ? `✓ ${picked || path.split("/").pop()}` : "zip or stem WAVs"}
              </span>
              {path && upPct === null && !running && (
                <button onClick={() => { setPath(""); setPicked(""); }} title="Clear the chosen stems"
                  className="flex-none rounded-md px-1.5 text-[var(--color-muted)] hover:bg-[var(--color-panel2)] hover:text-red-400">✕</button>
              )}
            </div>
            {upPct !== null && (
              <div className="mt-2 h-1 overflow-hidden rounded-full bg-[#0e1015]">
                <div className="h-full bg-gradient-to-r from-[var(--color-accent)] to-[var(--color-accent2)] transition-all" style={{ width: `${upPct}%` }} />
              </div>
            )}
          </div>
          <label className="mb-1 block">
            <span className="mb-1.5 block text-xs text-[var(--color-muted)]">Style</span>
            <select className={inp} value={presets.length ? preset : ""} onChange={(e) => setPreset(e.target.value)}>
              {presets.length > 0 && <option value="auto">Auto (closest style to this song)</option>}
              {presets.map((p) => <option key={p.name} value={p.name}>{p.label} · {p.n} songs</option>)}
              <option value="">Fixed rules (no style)</option>
            </select>
          </label>
          {!usePreset && (
            <label className="mt-3 block">
              <span className="mb-1.5 block text-xs text-[var(--color-muted)]">Fixed-rules profile</span>
              <select className={inp} value={profile} onChange={(e) => setProfile(e.target.value)}>
                {PROFILES.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
              </select>
            </label>
          )}
          <PrimaryButton onClick={run} disabled={busy || running || upPct !== null || !path}>{running ? "Measuring + mixing…" : "Analyse + mix"}</PrimaryButton>
        </div>

      {/* right column: the mixer */}
      <div className={`${card} min-w-0 @5xl:col-start-2 @5xl:row-span-3 @5xl:row-start-1`}>
        <div className="mb-2 flex flex-wrap items-center gap-2">
          <h2 className={`${h} mr-1`}>Mixer</h2>
          <select className={`${inp} w-auto min-w-[200px] flex-1 py-1.5 text-xs`} value={sid} onChange={(e) => openSession(e.target.value)} title="Open a song you mixed before">
            <option value="">Open a song…</option>
            {sessions.map((s) => <option key={s.session} value={s.session}>{s.title} · {s.preset || "fixed rules"} · {new Date(s.updated * 1000).toLocaleDateString()}</option>)}
          </select>
        </div>
        {doc && (
          <div className="mb-3 flex flex-wrap items-center gap-2 rounded-xl bg-[var(--color-panel2)] px-3 py-2">
            <span className={`text-[11px] ${dirty ? "font-semibold text-sky-300" : "text-[var(--color-muted)]"}`}>{dirty ? "● Unrendered changes" : "✓ Matches the AI's mix"}</span>
            <div className="hidden flex-wrap items-center gap-x-2.5 text-[10px] text-[var(--color-muted)] @2xl:flex">
              {GROUPS.filter((g) => Object.values(doc.stems).some((s) => (s.group || "other") === g.id)).map((g) => (
                <span key={g.id} className="flex items-center gap-1"><span className="h-2 w-2 rounded-full" style={{ background: colorOf(g.id) }} />{g.label}</span>
              ))}
            </div>
            <div className="ml-auto flex min-w-0 flex-nowrap items-center gap-2">
              <input className={`${inp} w-40 min-w-0 shrink py-1.5 text-xs`} value={label} onChange={(e) => setLabel(e.target.value)} placeholder="Name this version" title="Shown in the library and Versions list" />
              {ai && dirty && <button onClick={() => setDoc(clone(ai))} className="whitespace-nowrap rounded-lg border border-[var(--color-line)] px-3 py-1.5 text-xs text-[var(--color-muted)] hover:text-[var(--color-ink)]" title="Discard every change and go back to the AI's mix">↺ Revert all</button>}
              <button onClick={render} disabled={rendering} title="Render exactly what is on screen as a new version"
                className="whitespace-nowrap rounded-lg bg-gradient-to-r from-[var(--color-accent)] to-[var(--color-accent2)] px-4 py-1.5 text-xs font-semibold text-white shadow transition hover:brightness-110 disabled:opacity-50">
                {rendering ? "Rendering…" : "Render version"}
              </button>
            </div>
          </div>
        )}
        {!doc ? (
          <div className="rounded-xl border border-dashed border-[var(--color-line)] p-10 text-center text-sm text-[var(--color-muted)]">
            Run <b className="text-[var(--color-ink)]">Analyse + mix</b> on a Suno stem download, or open an earlier song above.
          </div>
        ) : (
          <>
            <p className="mb-2 text-[11px] text-[var(--color-muted)] @xl:hidden">Tip: hide the Library (top right) to see more channels at once.</p>
            <Console key={sid} doc={doc} ai={ai} setDoc={(f) => setDoc((x) => (x ? f(x) : x))} />
          </>
        )}
      </div>
        {doc && (
          <div className={`${card} @5xl:col-start-1 @5xl:row-start-2`}>
            <div className="mb-2 flex items-baseline justify-between">
              <h2 className={h}>What the AI did</h2>
              {notes.length > 0 && <button onClick={() => setShowNotes((v) => !v)} className="text-[11px] text-[var(--color-accent2)] hover:underline">{showNotes ? "Hide" : `Show all ${notes.length}`}</button>}
            </div>
            <div className="mb-2 flex flex-wrap gap-1.5">
              <Chip k="Style" v={doc.preset ? `${doc.preset}${summary?.preset_scores ? " (auto)" : ""}` : "fixed rules"} />
              {summary?.fit_r != null && <Chip k="Balance fit" v={`r ${summary.fit_r.toFixed(2)}`} />}
              {summary?.bpm != null && <Chip k="Tempo" v={`${Math.round(summary.bpm)} BPM`} />}
              {summary?.mix_lufs != null && <Chip k="Loudness" v={`${summary.mix_lufs.toFixed(1)} LUFS`} />}
            </div>
            <p className={hint}>Each move is measured against the style's typical range; inside the range means no change.{tasteUsed ? " Vocal density, wetness and backing level follow your starred versions instead." : ""} Click a strip in the mixer to see its reasons.</p>
            {showNotes && (
              <div className="mt-3 space-y-2">
                {byGroup.map(({ g, items }) => (
                  <div key={g.id}>
                    <div className="mb-0.5 flex items-center gap-1.5 text-[11px] font-semibold" style={{ color: g.color }}><span className="h-2 w-2 rounded-full" style={{ background: g.color }} />{g.label}</div>
                    {items.map((t, i) => <div key={i} className="pl-3.5 text-[11px] leading-relaxed text-[var(--color-muted)]">{t.replace(/^[^:]+:\s*/, "")}</div>)}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {cur && cur.renders.length > 0 && (
          <div className={`${card} @5xl:col-start-1 @5xl:row-start-3`}>
            <h2 className={h}>Versions</h2>
            <p className={`${hint} mb-2`}>Every render of this song. Tick two or more to compare in sync (all loudness-matched). <b className="text-[var(--color-accent2)]">★</b> the ones you like: the AI measures their vocal treatment and aims for it on every song.</p>
            <div className="max-h-64 space-y-1 overflow-y-auto pr-1">
              {[...cur.renders].reverse().map((r) => (
                <label key={r.job_id} className={`flex cursor-pointer items-start gap-2 rounded-lg px-2 py-1.5 text-xs ${cmp.includes(r.job_id) ? "bg-[#2a1c19] text-[var(--color-ink)]" : "text-[var(--color-muted)] hover:bg-[var(--color-panel2)]"}`}>
                  <input type="checkbox" className="mt-0.5" style={{ accentColor: "var(--color-accent)" }} checked={cmp.includes(r.job_id)}
                    onChange={(e) => setCmp((c) => e.target.checked ? [...c, r.job_id] : c.filter((x) => x !== r.job_id))} />
                  <span className="flex-1">{r.label.replace(/^.*?\(AI mix: /, "").replace(/\)$/, "").replace(/^.*\((Suno mix)\)$/, "$1")}</span>
                  {!/Suno mix|stem sum/.test(r.label) && (
                    <button onClick={(e) => { e.preventDefault(); star(r.job_id, !starred.has(r.job_id)); }} disabled={starring === r.job_id}
                      title={starred.has(r.job_id) ? "Starred: part of your taste. Click to remove." : "Star: teach the AI your vocal taste from this version"}
                      className={`text-base leading-none transition ${starred.has(r.job_id) ? "text-[var(--color-accent2)]" : "text-[#4b5060] hover:text-[var(--color-accent2)]"} ${starring === r.job_id ? "animate-pulse" : ""}`}>★</button>
                  )}
                </label>
              ))}
            </div>
            <button onClick={() => { setCompare(() => cmp); goTo("compare"); }} disabled={cmp.length < 2}
              className="mt-3 w-full rounded-lg border border-[var(--color-accent)] bg-[#2a1c19] py-2 text-xs font-semibold text-[var(--color-ink)] transition hover:brightness-125 disabled:cursor-not-allowed disabled:border-[var(--color-line)] disabled:bg-transparent disabled:text-[var(--color-muted)]">
              Compare {cmp.length || ""} ticked
            </button>
          </div>
        )}
    </div>
    </div>
  );
}
