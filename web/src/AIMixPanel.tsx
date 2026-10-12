import { useEffect, useState } from "react";
import { api } from "./api";
import { GhostButton, PrimaryButton, inp, rid, type RunCtx } from "./ui";

// Manual control of an AI mix (docs/AI_MIX_PLAN.md, Phase 2): the mix doc the AI wrote is the
// starting point; every fader, send and processor here is editable, and Render files a new
// library track from exactly what is on screen. The AI's own doc is kept separately, so any
// stem (or the whole mix) can be reverted to the AI's version.

type EqMove = { type: string; hz: number; db: number; q?: number };
type Band = { hz: number; db: number; q?: number };
export type Fx = {
  fx: string; by?: string; why?: string;
  hz?: number; moves?: EqMove[]; bands?: Band[];
  threshold_db?: number; ratio?: number; attack_ms?: number; release_ms?: number;
  attack_db?: number; sustain_db?: number; range_db?: number;
};
type Stem = {
  role: string; group?: string; mute: boolean; target_lufs: number; trim_db: number;
  width: number; low_mono_hz: number; chain: Fx[]; sends: { verb?: number; delay?: number };
};
type Bus = { room_size?: number; damping?: number; predelay_ms?: number; ms?: number; feedback?: number; hpf?: number; lpf?: number };
export type MixDoc = {
  title: string; preset?: string; notes?: string[]; stems: Record<string, Stem>;
  buses: { verb: Bus; delay: Bus }; [k: string]: unknown;
};
type Render = { job_id: string; label: string; audio_url: string };
type Session = { session: string; title: string; preset?: string; updated: number; renders: Render[] };

const clone = <T,>(x: T): T => JSON.parse(JSON.stringify(x));
const FX_LABEL: Record<string, string> = {
  highpass: "High-pass", eq: "EQ", peq: "Unmask EQ", compressor: "Compressor", transient: "Punch", deess: "De-ess",
};
// Manual "add" starting values: only used when YOU add a processor; the AI sizes its own.
const NEW_FX: Record<string, Fx> = {
  highpass: { fx: "highpass", hz: 80 },
  eq: { fx: "eq", moves: [{ type: "bell", hz: 1000, db: 0, q: 1.4 }] },
  compressor: { fx: "compressor", threshold_db: -24, ratio: 3, attack_ms: 10, release_ms: 120 },
  transient: { fx: "transient", attack_db: 3, sustain_db: 0 },
  deess: { fx: "deess", range_db: 3 },
};

function Num({ v, set, step = 0.5, w = "w-16", title }: { v: number | undefined; set: (n: number) => void; step?: number; w?: string; title?: string }) {
  return <input type="number" title={title} step={step} className={`${inp} ${w} px-1 py-0.5 text-xs`} value={v ?? 0}
    onChange={(e) => set(parseFloat(e.target.value) || 0)} />;
}

function FxEditor({ f, set, remove }: { f: Fx; set: (p: Partial<Fx>) => void; remove: () => void }) {
  const lab = (t: string) => <span className="text-[var(--color-muted)]">{t}</span>;
  return (
    <div className="flex flex-wrap items-center gap-2 rounded border border-[var(--color-line)] px-2 py-1 text-xs">
      <b>{FX_LABEL[f.fx] || f.fx}</b>
      {f.by === "ai" && <span className="rounded bg-[var(--color-accent2)]/20 px-1 text-[10px]" title={f.why || "set by the AI from measurements"}>AI</span>}
      {f.by === "edited" && <span className="rounded bg-[var(--color-line)] px-1 text-[10px]">edited</span>}
      {f.fx === "highpass" && <>{lab("Hz")}<Num v={f.hz} step={5} set={(n) => set({ hz: n })} /></>}
      {f.fx === "compressor" && <>
        {lab("thr")}<Num v={f.threshold_db} set={(n) => set({ threshold_db: n })} />
        {lab("ratio")}<Num v={f.ratio} step={0.5} w="w-12" set={(n) => set({ ratio: n })} />
        {lab("att ms")}<Num v={f.attack_ms} step={1} w="w-12" set={(n) => set({ attack_ms: n })} />
        {lab("rel ms")}<Num v={f.release_ms} step={10} w="w-14" set={(n) => set({ release_ms: n })} />
      </>}
      {f.fx === "transient" && <>{lab("attack dB")}<Num v={f.attack_db} set={(n) => set({ attack_db: n })} />
        {lab("sustain dB")}<Num v={f.sustain_db} set={(n) => set({ sustain_db: n })} /></>}
      {f.fx === "deess" && <>{lab("range dB")}<Num v={f.range_db} set={(n) => set({ range_db: n })} /></>}
      {(f.fx === "eq" || f.fx === "peq") && (
        <div className="flex w-full flex-col gap-1">
          {(f.fx === "eq" ? f.moves || [] : f.bands || []).map((m, i) => (
            <div key={i} className="flex items-center gap-2">
              {f.fx === "eq" && (
                <select className={`${inp} w-24 px-1 py-0.5 text-xs`} value={(m as EqMove).type}
                  onChange={(e) => { const mv = clone(f.moves || []); mv[i].type = e.target.value; set({ moves: mv }); }}>
                  <option value="bell">bell</option><option value="lowshelf">low shelf</option><option value="highshelf">high shelf</option>
                </select>
              )}
              {lab("Hz")}<Num v={m.hz} step={10} w="w-20" set={(n) => { const k = f.fx === "eq" ? "moves" : "bands"; const a = clone((f[k] || []) as Band[]); a[i].hz = n; set({ [k]: a } as Partial<Fx>); }} />
              {lab("dB")}<Num v={m.db} set={(n) => { const k = f.fx === "eq" ? "moves" : "bands"; const a = clone((f[k] || []) as Band[]); a[i].db = n; set({ [k]: a } as Partial<Fx>); }} />
              {lab("Q")}<Num v={m.q ?? 1} step={0.1} w="w-12" set={(n) => { const k = f.fx === "eq" ? "moves" : "bands"; const a = clone((f[k] || []) as Band[]); a[i].q = n; set({ [k]: a } as Partial<Fx>); }} />
              <button className="text-[var(--color-muted)] hover:text-red-400" title="Remove band"
                onClick={() => { const k = f.fx === "eq" ? "moves" : "bands"; const a = clone((f[k] || []) as Band[]); a.splice(i, 1); set({ [k]: a } as Partial<Fx>); }}>✕</button>
            </div>
          ))}
          {f.fx === "eq" && <button className="self-start text-[var(--color-accent2)]" onClick={() => set({ moves: [...(f.moves || []), { type: "bell", hz: 1000, db: 0, q: 1.4 }] })}>+ band</button>}
        </div>
      )}
      {f.why && <span className="text-[var(--color-muted)]" title={f.why}>· {f.why}</span>}
      <button className="ml-auto text-[var(--color-muted)] hover:text-red-400" title="Remove processor" onClick={remove}>✕</button>
    </div>
  );
}

export function AIMixPanel({ sid: initialSid, ctx, setCompare, goTo }: { sid?: string; ctx: RunCtx; setCompare: (f: (ids: string[]) => string[]) => void; goTo: (m: string) => void }) {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [sid, setSid] = useState(initialSid || "");
  const [doc, setDoc] = useState<MixDoc | null>(null);
  const [ai, setAi] = useState<MixDoc | null>(null);
  const [open, setOpen] = useState<Record<string, boolean>>({});
  const [label, setLabel] = useState("manual edit");
  const [rendering, setRendering] = useState(false);
  const [cmp, setCmp] = useState<string[]>([]);

  const loadSessions = () => api.aimixSessions().then((r) => setSessions((r as { sessions: Session[] }).sessions)).catch(() => setSessions([]));
  const loadDoc = (s: string) => api.aimixGet(s)
    .then((r) => { const x = r as { mixdoc: MixDoc; mixdoc_ai: MixDoc }; setDoc(x.mixdoc); setAi(x.mixdoc_ai); })
    .catch(() => setDoc(null));
  const openSession = (s: string) => { setSid(s); setCmp([]); if (s) loadDoc(s); else setDoc(null); };
  // the parent remounts this panel (key) when a new run finishes, so mount = (re)load
  useEffect(() => { loadSessions(); if (initialSid) loadDoc(initialSid); }, [initialSid]);

  if (!sessions.length && !doc) return null;
  const cur = sessions.find((s) => s.session === sid);

  const setStem = (n: string, p: Partial<Stem>) => setDoc((d) => d && { ...d, stems: { ...d.stems, [n]: { ...d.stems[n], ...p } } });
  const setFx = (n: string, i: number, p: Partial<Fx>) => setDoc((d) => {
    if (!d) return d;
    const chain = clone(d.stems[n].chain);
    chain[i] = { ...chain[i], ...p, by: chain[i].by === "user" ? "user" : "edited" };
    return { ...d, stems: { ...d.stems, [n]: { ...d.stems[n], chain } } };
  });
  const delFx = (n: string, i: number) => setDoc((d) => {
    if (!d) return d;
    const chain = clone(d.stems[n].chain); chain.splice(i, 1);
    return { ...d, stems: { ...d.stems, [n]: { ...d.stems[n], chain } } };
  });
  const addFx = (n: string, kind: string) => setDoc((d) => {
    if (!d || !NEW_FX[kind]) return d;
    return { ...d, stems: { ...d.stems, [n]: { ...d.stems[n], chain: [...d.stems[n].chain, { ...clone(NEW_FX[kind]), by: "user" }] } } };
  });
  const setBus = (b: "verb" | "delay", p: Partial<Bus>) => setDoc((d) => d && { ...d, buses: { ...d.buses, [b]: { ...d.buses[b], ...p } } });
  const notesFor = (n: string, s: Stem) => (doc?.notes || []).filter((t) => t.startsWith(`${s.group || s.role}:`) || t.startsWith(`${n}:`));

  async function render() {
    if (!doc || !sid) return;
    const id = rid();
    setRendering(true);
    ctx.setResults([{ id, title: `rendering "${label}"… (Mac, ~30 s)`, status: "running", pct: 40 }]);
    try {
      const r = await api.aimixRender(sid, { mixdoc: doc, label }) as { track: Render };
      ctx.setResults([{ id: rid(), title: r.track.label, status: "done", pct: 100, url: r.track.audio_url }]);
      ctx.onDone();
      setCmp((c) => [...c, r.track.job_id]);
      loadSessions();
    } catch (e) { ctx.patch(id, { status: "error", pct: 0, err: (e as Error).message }); }
    finally { setRendering(false); }
  }

  return (
    <div className="space-y-3 border-t border-[var(--color-line)] pt-4">
      <div className="flex items-center gap-2">
        <b className="text-sm">Mixer</b>
        <select className={`${inp} flex-1 text-xs`} value={sid} onChange={(e) => openSession(e.target.value)}>
          <option value="">— open an AI mix session —</option>
          {sessions.map((s) => <option key={s.session} value={s.session}>{s.title} · {s.preset || "fixed rules"} · {new Date(s.updated * 1000).toLocaleString()}</option>)}
        </select>
      </div>
      {doc && (
        <>
          <p className="text-xs text-[var(--color-muted)]">
            Start from the AI's mix, change anything, then Render: each render is a new library track, so nothing is lost.
            Items marked <b>AI</b> were set from measurements (hover for why); <b>edited</b> = changed by you.
          </p>
          <div className="space-y-1">
            {Object.entries(doc.stems).map(([n, s]) => (
              <div key={n} className={`rounded border border-[var(--color-line)] ${s.mute ? "opacity-50" : ""}`}>
                <div className="flex flex-wrap items-center gap-2 px-2 py-1 text-xs">
                  <button className="w-4 text-[var(--color-muted)]" onClick={() => setOpen((o) => ({ ...o, [n]: !o[n] }))}>{open[n] ? "▾" : "▸"}</button>
                  <b className="w-28 truncate" title={`${s.role} (${s.group || s.role})`}>{n}</b>
                  <label className="flex items-center gap-1"><input type="checkbox" checked={s.mute} onChange={(e) => setStem(n, { mute: e.target.checked })} />mute</label>
                  <span className="text-[var(--color-muted)]">level</span>
                  <input type="range" min={-12} max={12} step={0.5} value={s.trim_db || 0} onChange={(e) => setStem(n, { trim_db: parseFloat(e.target.value) })} className="w-24" />
                  <span className="w-12 tabular-nums">{(s.trim_db || 0) > 0 ? "+" : ""}{(s.trim_db || 0).toFixed(1)}</span>
                  <span className="text-[var(--color-muted)]">width</span><Num v={s.width} step={0.05} w="w-14" set={(v) => setStem(n, { width: v })} />
                  <span className="text-[var(--color-muted)]">mono &lt;</span>
                  <select className={`${inp} w-20 px-1 py-0.5 text-xs`} value={s.low_mono_hz || 0} onChange={(e) => setStem(n, { low_mono_hz: parseInt(e.target.value) })}>
                    {[0, 100, 120, 150, 200].map((v) => <option key={v} value={v}>{v ? `${v} Hz` : "off"}</option>)}
                  </select>
                  {(["verb", "delay"] as const).map((b) => (
                    <span key={b} className="flex items-center gap-1">
                      <span className="text-[var(--color-muted)]">{b}</span>
                      <select className={`${inp} w-20 px-1 py-0.5 text-xs`} value={s.sends[b] ?? "off"}
                        onChange={(e) => { const sends = { ...s.sends }; if (e.target.value === "off") delete sends[b]; else sends[b] = parseFloat(e.target.value); setStem(n, { sends }); }}>
                        <option value="off">off</option>
                        {[-34, -30, -28, -26, -24, -22, -20, -18, -16, -14, -12].map((v) => <option key={v} value={v}>{v} dB</option>)}
                      </select>
                    </span>
                  ))}
                  {ai?.stems[n] && <button className="ml-auto text-[var(--color-muted)] hover:text-[var(--color-accent2)]" title="Put this stem back to what the AI chose"
                    onClick={() => setDoc((d) => d && { ...d, stems: { ...d.stems, [n]: clone(ai.stems[n]) } })}>↺ AI</button>}
                </div>
                {open[n] && (
                  <div className="space-y-1 border-t border-[var(--color-line)] px-2 py-2">
                    {notesFor(n, s).map((t, i) => <div key={i} className="text-[11px] text-[var(--color-muted)]">AI: {t}</div>)}
                    {s.chain.map((f, i) => <FxEditor key={i} f={f} set={(p) => setFx(n, i, p)} remove={() => delFx(n, i)} />)}
                    <select className={`${inp} w-44 text-xs`} value="" onChange={(e) => { addFx(n, e.target.value); }}>
                      <option value="">+ add processor…</option>
                      {Object.keys(NEW_FX).map((k) => <option key={k} value={k}>{FX_LABEL[k]}</option>)}
                    </select>
                  </div>
                )}
              </div>
            ))}
          </div>
          <div className="flex flex-wrap items-center gap-2 rounded border border-[var(--color-line)] px-2 py-1 text-xs">
            <b>Reverb bus</b>
            <span className="text-[var(--color-muted)]">size</span><Num v={doc.buses.verb.room_size} step={0.05} w="w-14" set={(v) => setBus("verb", { room_size: v })} />
            <span className="text-[var(--color-muted)]">damping</span><Num v={doc.buses.verb.damping} step={0.05} w="w-14" set={(v) => setBus("verb", { damping: v })} />
            <span className="text-[var(--color-muted)]">pre-delay ms</span><Num v={doc.buses.verb.predelay_ms} step={5} w="w-14" set={(v) => setBus("verb", { predelay_ms: v })} />
            <b className="ml-3">Delay bus</b>
            <span className="text-[var(--color-muted)]">ms</span><Num v={doc.buses.delay.ms} step={5} w="w-16" set={(v) => setBus("delay", { ms: v })} />
            <span className="text-[var(--color-muted)]">feedback</span><Num v={doc.buses.delay.feedback} step={0.05} w="w-14" set={(v) => setBus("delay", { feedback: v })} />
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <input className={`${inp} w-60 text-xs`} value={label} onChange={(e) => setLabel(e.target.value)} placeholder="label for this render" title="Shown in the library title" />
            <PrimaryButton onClick={render} disabled={rendering}>{rendering ? "Rendering…" : "Render"}</PrimaryButton>
            {ai && <GhostButton onClick={() => setDoc(clone(ai))} title="Discard all edits and go back to the AI's mix">↺ Revert all to AI</GhostButton>}
          </div>
          {cur && cur.renders.length > 0 && (
            <div className="space-y-1 text-xs">
              <div className="text-[var(--color-muted)]">Renders from this session: tick the ones to A/B, then open Compare.</div>
              {cur.renders.map((r) => (
                <label key={r.job_id} className="flex items-center gap-2">
                  <input type="checkbox" checked={cmp.includes(r.job_id)} onChange={(e) => setCmp((c) => e.target.checked ? [...c, r.job_id] : c.filter((x) => x !== r.job_id))} />
                  {r.label}
                </label>
              ))}
              <GhostButton onClick={() => { setCompare(() => cmp); goTo("compare"); }} disabled={cmp.length < 2}>Compare ticked ({cmp.length})</GhostButton>
            </div>
          )}
        </>
      )}
    </div>
  );
}
