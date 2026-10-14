import { useRef, useState } from "react";
import { inp } from "./ui";
import { GROUPS, clone, colorOf, groupOf, type EqMove, type Fx, type MixDoc, type Stem } from "./aimixModel";

// AI Mix console (docs/AI_MIX_PLAN.md, Phase 2): vertical channel strips bound to the mix doc,
// plus an inspector for the selected strip's processors. The AI's doc is the starting point;
// everything here is editable and Render files exactly what is on screen.

const FX_LABEL: Record<string, string> = {
  highpass: "High-pass", eq: "EQ", peq: "Unmask EQ", compressor: "Compressor", transient: "Punch", deess: "De-ess",
};
const FX_SHORT: Record<string, string> = { eq: "EQ", peq: "EQ", compressor: "COMP", transient: "PUNCH", deess: "DE-ESS" };
// Starting values only for processors YOU add; the AI sizes its own from measurements.
const NEW_FX: Record<string, Fx> = {
  highpass: { fx: "highpass", hz: 80 },
  eq: { fx: "eq", moves: [{ type: "bell", hz: 1000, db: 0, q: 1.4 }] },
  compressor: { fx: "compressor", threshold_db: -24, ratio: 3, attack_ms: 10, release_ms: 120 },
  transient: { fx: "transient", attack_db: 3, sustain_db: 0 },
  deess: { fx: "deess", range_db: 3 },
};
const SEND_OFF = -41;

// ---------------------------------------------------------------- controls

function Fader({ value, onChange, mark, min = -15, max = 15, height = 132 }: { value: number; onChange: (v: number) => void; mark?: number; min?: number; max?: number; height?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const pct = (value - min) / (max - min);
  const zero = (0 - min) / (max - min);
  const fromY = (clientY: number) => {
    const r = ref.current!.getBoundingClientRect();
    const p = 1 - Math.min(1, Math.max(0, (clientY - r.top) / r.height));
    return Math.round((min + p * (max - min)) * 2) / 2;
  };
  return (
    <div ref={ref} role="slider" tabIndex={0} aria-valuemin={min} aria-valuemax={max} aria-valuenow={value}
      title="Drag to set level · double-click resets to 0 dB · arrow keys nudge"
      className="relative mx-auto w-7 cursor-ns-resize rounded-full bg-[#0e1015] outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-accent)]"
      style={{ height }}
      onPointerDown={(e) => { dragging.current = true; e.currentTarget.setPointerCapture(e.pointerId); onChange(fromY(e.clientY)); }}
      onPointerMove={(e) => { if (dragging.current) onChange(fromY(e.clientY)); }}
      onPointerUp={() => { dragging.current = false; }}
      onPointerCancel={() => { dragging.current = false; }}
      onLostPointerCapture={() => { dragging.current = false; }}
      onDoubleClick={() => onChange(0)}
      onKeyDown={(e) => {
        if (e.key === "ArrowUp") { e.preventDefault(); onChange(Math.min(max, value + 0.5)); }
        if (e.key === "ArrowDown") { e.preventDefault(); onChange(Math.max(min, value - 0.5)); }
      }}>
      <div className="absolute left-1/2 w-1 -translate-x-1/2 rounded-full bg-[var(--color-accent2)]"
        style={{ bottom: `${Math.min(pct, zero) * 100}%`, height: `${Math.abs(pct - zero) * 100}%` }} />
      <div className="absolute left-0 right-0 h-px bg-[var(--color-line)]" style={{ bottom: `${zero * 100}%` }} />
      {mark !== undefined && Math.abs(mark) > 0.05 && (
        <div className="absolute -left-1 h-0.5 w-2 rounded bg-[var(--color-accent2)]" title={`AI level: ${mark > 0 ? "+" : ""}${mark.toFixed(1)} dB`}
          style={{ bottom: `${((Math.min(max, Math.max(min, mark)) - min) / (max - min)) * 100}%` }} />
      )}
      <div className="absolute left-1/2 h-3 w-7 -translate-x-1/2 translate-y-1/2 rounded-md border border-[#3a3f4d] bg-[#d9dbe3] shadow"
        style={{ bottom: `${pct * 100}%` }} />
    </div>
  );
}

function Mini({ label, value, min, max, step, onChange, fmt }: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; fmt: (v: number) => string }) {
  return (
    <label className="block w-full" title={`${label}: ${fmt(value)}`}>
      <div className="flex justify-between text-[10px] text-[var(--color-muted)]"><span>{label}</span><span className="tabular-nums text-[var(--color-ink)]">{fmt(value)}</span></div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(parseFloat(e.target.value))}
        className="h-1 w-full cursor-pointer" style={{ accentColor: "var(--color-accent2)" }} />
    </label>
  );
}

function Field({ label, unit, v, set, step = 0.5 }: { label: string; unit?: string; v: number | undefined; set: (n: number) => void; step?: number }) {
  return (
    <label className="block">
      <span className="mb-1 block text-[10px] uppercase tracking-wide text-[var(--color-muted)]">{label}{unit ? ` (${unit})` : ""}</span>
      <input type="number" step={step} value={v ?? 0} onChange={(e) => set(parseFloat(e.target.value) || 0)}
        className={`${inp} px-2 py-1 text-xs tabular-nums`} />
    </label>
  );
}

function Badge({ by, why }: { by?: string; why?: string }) {
  if (by === "ai") return <span className="rounded-full bg-[var(--color-accent2)]/15 px-2 py-0.5 text-[10px] font-semibold text-[var(--color-accent2)]" title={why || "Set by the AI from measurements"}>AI</span>;
  if (by === "edited") return <span className="rounded-full bg-sky-400/15 px-2 py-0.5 text-[10px] font-semibold text-sky-300">Edited</span>;
  if (by === "user") return <span className="rounded-full bg-sky-400/15 px-2 py-0.5 text-[10px] font-semibold text-sky-300">Added</span>;
  return null;
}

// ---------------------------------------------------------------- strip

function Strip({ name, s, selected, onSelect, set, edited }: { name: string; s: Stem; selected: boolean; onSelect: () => void; set: (p: Partial<Stem>) => void; edited: boolean }) {
  const color = colorOf(groupOf(s));
  const lit = new Set(s.chain.map((c) => FX_SHORT[c.fx]).filter(Boolean));
  const aiCount = s.chain.filter((c) => c.by === "ai").length;
  const aiDb = s.suno_lufs !== undefined ? Math.round((s.target_lufs - s.suno_lufs) * 10) / 10 : 0;   // AI's level vs Suno balance
  const send = (b: "verb" | "delay") => s.sends[b] ?? SEND_OFF;
  const setSend = (b: "verb" | "delay", v: number) => { const sends = { ...s.sends }; if (v <= SEND_OFF) delete sends[b]; else sends[b] = v; set({ sends }); };
  return (
    <div onClick={onSelect}
      className={`flex w-[92px] flex-none cursor-pointer flex-col gap-2 rounded-xl border p-2 transition ${selected ? "border-[var(--color-accent)] bg-[#221a19]" : "border-[var(--color-line)] bg-[var(--color-panel2)] hover:border-[#3a3f4d]"} ${s.mute ? "opacity-55" : ""}`}>
      <div className="h-1 rounded-full" style={{ background: color }} />
      <div className="min-h-[44px]">
        <div className="line-clamp-2 break-words text-xs font-semibold leading-tight text-[var(--color-ink)]" title={name}>{name.replace(/_/g, " ")}</div>
        <div className="flex items-center gap-1 text-[10px] text-[var(--color-muted)]">
          {aiCount > 0 && <span className="text-[var(--color-accent2)]" title={`${aiCount} AI-set processor(s)`}>AI·{aiCount}</span>}
          {edited && <span className="text-sky-300" title="You changed this strip">● edited</span>}
        </div>
      </div>
      <div className="flex flex-wrap gap-1">
        {["EQ", "COMP", "PUNCH", "DE-ESS"].map((k) => (
          <span key={k} className={`rounded px-1 py-px text-[9px] font-semibold ${lit.has(k) ? "bg-[var(--color-accent2)]/20 text-[var(--color-accent2)]" : "bg-[#0e1015] text-[#4b5060]"}`}>{k}</span>
        ))}
      </div>
      <div onClick={(e) => e.stopPropagation()} className="space-y-1.5">
        <Mini label="Width" value={s.width ?? 1} min={0} max={2} step={0.05} onChange={(v) => set({ width: v })} fmt={(v) => `${Math.round(v * 100)}%`} />
        <Mini label="Reverb" value={send("verb")} min={SEND_OFF} max={-10} step={1} onChange={(v) => setSend("verb", v)} fmt={(v) => v <= SEND_OFF ? "off" : `${v}`} />
        <Mini label="Delay" value={send("delay")} min={SEND_OFF} max={-10} step={1} onChange={(v) => setSend("delay", v)} fmt={(v) => v <= SEND_OFF ? "off" : `${v}`} />
        <button onClick={() => set({ low_mono_hz: s.low_mono_hz ? 0 : 150 })} title="Make everything below this frequency mono"
          className={`w-full whitespace-nowrap rounded-full border px-1 py-0.5 text-[10px] ${s.low_mono_hz ? "border-[var(--color-accent)] bg-[#2a1c19] text-[var(--color-ink)]" : "border-[var(--color-line)] text-[var(--color-muted)]"}`}>
          {s.low_mono_hz ? `Mono <${s.low_mono_hz} Hz` : "Mono lows"}
        </button>
      </div>
      <div onClick={(e) => e.stopPropagation()}>
        {/* 0 dB = Suno's own balance; the fader shows the AI's level change plus your trim, the
            orange tick marks where the AI put it. Dragging only changes your trim. */}
        <Fader value={aiDb + (s.trim_db || 0)} mark={aiDb} onChange={(v) => set({ trim_db: Math.round((v - aiDb) * 2) / 2 })} />
        <div className="mt-1 text-center text-[11px] tabular-nums text-[var(--color-ink)]" title={`AI ${aiDb > 0 ? "+" : ""}${aiDb.toFixed(1)} dB${s.trim_db ? `, your trim ${s.trim_db > 0 ? "+" : ""}${s.trim_db.toFixed(1)} dB` : ""} (0 = Suno's balance)`}>
          {aiDb + (s.trim_db || 0) > 0 ? "+" : ""}{(aiDb + (s.trim_db || 0)).toFixed(1)} dB
        </div>
      </div>
      <button onClick={(e) => { e.stopPropagation(); set({ mute: !s.mute }); }} title="Mute this stem in the next render"
        className={`rounded-md py-1 text-[11px] font-bold ${s.mute ? "bg-[var(--color-accent)] text-white" : "bg-[#0e1015] text-[var(--color-muted)] hover:text-[var(--color-ink)]"}`}>M</button>
    </div>
  );
}

// ---------------------------------------------------------------- inspector

function FxCard({ f, set, remove }: { f: Fx; set: (p: Partial<Fx>) => void; remove: () => void }) {
  const list = (f.fx === "eq" ? f.moves : f.bands) || [];
  const key = f.fx === "eq" ? "moves" : "bands";
  const setBand = (i: number, p: Partial<EqMove>) => { const a = clone(list) as EqMove[]; a[i] = { ...a[i], ...p }; set({ [key]: a } as Partial<Fx>); };
  return (
    <div className="rounded-xl border border-[var(--color-line)] bg-[var(--color-panel2)] p-3">
      <div className="mb-2 flex items-center gap-2">
        <b className="text-sm text-[var(--color-ink)]">{FX_LABEL[f.fx] || f.fx}</b>
        <Badge by={f.by} why={f.why} />
        <button onClick={remove} title="Remove this processor" className="ml-auto rounded-md px-2 text-[var(--color-muted)] hover:bg-[#0e1015] hover:text-red-400">✕</button>
      </div>
      {f.why && <p className="mb-2 text-[11px] text-[var(--color-muted)]">{f.why}</p>}
      {f.fx === "highpass" && <div className="grid grid-cols-2 gap-2 sm:grid-cols-4"><Field label="Cutoff" unit="Hz" step={5} v={f.hz} set={(n) => set({ hz: n })} /></div>}
      {f.fx === "compressor" && (
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Field label="Threshold" unit="dB" v={f.threshold_db} set={(n) => set({ threshold_db: n })} />
          <Field label="Ratio" unit=":1" v={f.ratio} set={(n) => set({ ratio: n })} />
          <Field label="Attack" unit="ms" step={1} v={f.attack_ms} set={(n) => set({ attack_ms: n })} />
          <Field label="Release" unit="ms" step={10} v={f.release_ms} set={(n) => set({ release_ms: n })} />
        </div>
      )}
      {f.fx === "transient" && (
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Field label="Attack boost" unit="dB" v={f.attack_db} set={(n) => set({ attack_db: n })} />
          <Field label="Sustain" unit="dB" v={f.sustain_db} set={(n) => set({ sustain_db: n })} />
        </div>
      )}
      {f.fx === "deess" && <div className="grid grid-cols-2 gap-2 sm:grid-cols-4"><Field label="Max reduction" unit="dB" v={f.range_db} set={(n) => set({ range_db: n })} /></div>}
      {(f.fx === "eq" || f.fx === "peq") && (
        <div className="space-y-2">
          {list.map((m, i) => (
            <div key={i} className="grid grid-cols-[1fr_1fr_1fr_1fr_auto] items-end gap-2">
              {f.fx === "eq" ? (
                <label className="block"><span className="mb-1 block text-[10px] uppercase tracking-wide text-[var(--color-muted)]">Type</span>
                  <select className={`${inp} px-2 py-1 text-xs`} value={(m as EqMove).type} onChange={(e) => setBand(i, { type: e.target.value })}>
                    <option value="bell">Bell</option><option value="lowshelf">Low shelf</option><option value="highshelf">High shelf</option>
                  </select></label>
              ) : <div className="pb-1.5 text-[11px] text-[var(--color-muted)]">Cut</div>}
              <Field label="Freq" unit="Hz" step={10} v={m.hz} set={(n) => setBand(i, { hz: n })} />
              <Field label="Gain" unit="dB" v={m.db} set={(n) => setBand(i, { db: n })} />
              <Field label="Q" step={0.1} v={m.q ?? 1} set={(n) => setBand(i, { q: n })} />
              <button title="Remove band" className="pb-1.5 text-[var(--color-muted)] hover:text-red-400"
                onClick={() => { const a = clone(list); a.splice(i, 1); set({ [key]: a } as Partial<Fx>); }}>✕</button>
            </div>
          ))}
          {f.fx === "eq" && <button className="text-xs text-[var(--color-accent2)] hover:underline" onClick={() => set({ moves: [...(f.moves || []), { type: "bell", hz: 1000, db: 0, q: 1.4 }] })}>+ Add band</button>}
        </div>
      )}
    </div>
  );
}

export function Console({ doc, ai, setDoc }: { doc: MixDoc; ai: MixDoc | null; setDoc: (f: (d: MixDoc) => MixDoc) => void }) {
  const names = Object.keys(doc.stems).sort((a, b) =>
    GROUPS.findIndex((g) => g.id === groupOf(doc.stems[a])) - GROUPS.findIndex((g) => g.id === groupOf(doc.stems[b])));
  const [sel, setSel] = useState<string>(names[0] || "");
  const s = doc.stems[sel];
  const setStem = (n: string, p: Partial<Stem>) => setDoc((d) => ({ ...d, stems: { ...d.stems, [n]: { ...d.stems[n], ...p } } }));
  const setFx = (i: number, p: Partial<Fx>) => setDoc((d) => {
    const chain = clone(d.stems[sel].chain);
    chain[i] = { ...chain[i], ...p, by: chain[i].by === "user" ? "user" : "edited" };
    return { ...d, stems: { ...d.stems, [sel]: { ...d.stems[sel], chain } } };
  });
  const delFx = (i: number) => setDoc((d) => { const chain = clone(d.stems[sel].chain); chain.splice(i, 1); return { ...d, stems: { ...d.stems, [sel]: { ...d.stems[sel], chain } } }; });
  const addFx = (k: string) => setDoc((d) => ({ ...d, stems: { ...d.stems, [sel]: { ...d.stems[sel], chain: [...d.stems[sel].chain, { ...clone(NEW_FX[k]), by: "user" }] } } }));
  const edited = (n: string) => !!ai?.stems[n] && JSON.stringify(ai.stems[n]) !== JSON.stringify(doc.stems[n]);
  const notes = s ? (doc.notes || []).filter((t) => t.startsWith(`${groupOf(s)}:`) || t.startsWith(`${sel}:`)) : [];

  return (
    <div className="space-y-4">
      <div className="overflow-x-auto pb-2">
        <div className="flex gap-2">
          {names.map((n, i) => {
            const g = groupOf(doc.stems[n]);
            const first = i === 0 || groupOf(doc.stems[names[i - 1]]) !== g;
            return (
              <div key={n} className={`flex flex-col ${first && i > 0 ? "ml-2" : ""}`}>
                <div className="mb-1 h-4 text-[10px] font-semibold uppercase tracking-wide" style={{ color: colorOf(g) }}>{first ? GROUPS.find((x) => x.id === g)?.label : ""}</div>
                <Strip name={n} s={doc.stems[n]} selected={n === sel} onSelect={() => setSel(n)} set={(p) => setStem(n, p)} edited={edited(n)} />
              </div>
            );
          })}
        </div>
      </div>

      {s && (
        <div className="rounded-[14px] border border-[var(--color-line)] bg-[var(--color-panel)] p-4">
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <span className="h-3 w-3 rounded-full" style={{ background: colorOf(groupOf(s)) }} />
            <h3 className="text-[15px] font-semibold text-[var(--color-ink)]">{sel.replace(/_/g, " ")}</h3>
            <span className="text-xs text-[var(--color-muted)]">processors run top to bottom, then the strip's level, width and sends</span>
            {ai?.stems[sel] && edited(sel) && (
              <button onClick={() => setDoc((d) => ({ ...d, stems: { ...d.stems, [sel]: clone(ai.stems[sel]) } }))}
                className="ml-auto rounded-full border border-[var(--color-line)] px-3 py-1 text-[11px] text-[var(--color-muted)] hover:text-[var(--color-ink)]">↺ Revert strip to AI</button>
            )}
          </div>
          {notes.length > 0 && (
            <div className="mb-3 rounded-lg bg-[#0e1015] p-2.5 text-[11px] leading-relaxed text-[var(--color-muted)]">
              <div className="mb-1 font-semibold text-[var(--color-accent2)]">Why the AI did this ({GROUPS.find((x) => x.id === groupOf(s))?.label} group)</div>
              {notes.map((t, i) => <div key={i}>· {t.replace(/^[^:]+:\s*/, "")}</div>)}
            </div>
          )}
          <div className="space-y-2">
            {s.chain.length === 0 && <p className="text-xs text-[var(--color-muted)]">No processors on this stem.</p>}
            {s.chain.map((f, i) => <FxCard key={i} f={f} set={(p) => setFx(i, p)} remove={() => delFx(i)} />)}
          </div>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <span className="text-[11px] text-[var(--color-muted)]">Add:</span>
            {Object.keys(NEW_FX).map((k) => (
              <button key={k} onClick={() => addFx(k)} className="se-pill hover:text-[var(--color-ink)]">+ {FX_LABEL[k]}</button>
            ))}
          </div>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-[14px] border border-[var(--color-line)] bg-[var(--color-panel)] px-4 py-3 text-xs">
        <b className="text-[var(--color-ink)]">Reverb bus</b>
        <div className="w-28"><Mini label="Size" value={doc.buses.verb.room_size ?? 0.35} min={0} max={1} step={0.05} onChange={(v) => setDoc((d) => ({ ...d, buses: { ...d.buses, verb: { ...d.buses.verb, room_size: v } } }))} fmt={(v) => v.toFixed(2)} /></div>
        <div className="w-28"><Mini label="Damping" value={doc.buses.verb.damping ?? 0.5} min={0} max={1} step={0.05} onChange={(v) => setDoc((d) => ({ ...d, buses: { ...d.buses, verb: { ...d.buses.verb, damping: v } } }))} fmt={(v) => v.toFixed(2)} /></div>
        <div className="w-28"><Mini label="Pre-delay" value={doc.buses.verb.predelay_ms ?? 30} min={0} max={120} step={5} onChange={(v) => setDoc((d) => ({ ...d, buses: { ...d.buses, verb: { ...d.buses.verb, predelay_ms: v } } }))} fmt={(v) => `${v} ms`} /></div>
        <span className="mx-1 h-6 w-px bg-[var(--color-line)]" />
        <b className="text-[var(--color-ink)]">Delay bus</b>
        <div className="w-28"><Mini label="Time" value={doc.buses.delay.ms ?? 500} min={50} max={1200} step={5} onChange={(v) => setDoc((d) => ({ ...d, buses: { ...d.buses, delay: { ...d.buses.delay, ms: v } } }))} fmt={(v) => `${Math.round(v)} ms`} /></div>
        <div className="w-28"><Mini label="Feedback" value={doc.buses.delay.feedback ?? 0.25} min={0} max={0.8} step={0.05} onChange={(v) => setDoc((d) => ({ ...d, buses: { ...d.buses, delay: { ...d.buses.delay, feedback: v } } }))} fmt={(v) => `${Math.round(v * 100)}%`} /></div>
        <span className="mx-1 h-6 w-px bg-[var(--color-line)]" />
        <label className="flex cursor-pointer items-center gap-2" title="Give the mix the same slow section-to-section loudness shape as Suno's own (mastered) mix. Needs the Suno full mix in the download; only changes over several seconds are touched.">
          <input type="checkbox" style={{ accentColor: "var(--color-accent)" }} checked={!!doc.master?.macro_match}
            onChange={(e) => setDoc((d) => ({ ...d, master: { ...(d.master || {}), macro_match: e.target.checked } }))} />
          <b className="text-[var(--color-ink)]">Match Suno's section levels</b>
        </label>
      </div>
    </div>
  );
}
