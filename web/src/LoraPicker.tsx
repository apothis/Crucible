import { useEffect, useState } from "react";
import { api, type Config } from "./api";
import { inp } from "./ui";

// One selected adapter: a box-side safetensors path + the strength to apply.
// `label` is display-only. The generation sends [{path, scale}] and the backend
// reconciles the engine to EXACTLY this set (verified) before the take.
export type LoraSel = { path: string; scale: number; label: string };
type Opt = { label: string; path: string };

// Short, human label for a training-run dir: the legacy `train/` is the
// continuous run; per-run dirs carry their config (e.g. "...__lokr_150ep_discrete").
function runShort(run: string): string {
  if (!run) return "";
  // run_label: train_YYYYMMDD-HHMMSS__lokr_<epochs>ep_<sampling>[_lr<x>_a<n>]
  // Show config (epochs/sampling/lr/alpha) AND the run time so every run is
  // distinguishable in the picker — two discrete runs no longer look identical.
  const ts = run.match(/(\d{8})-(\d{6})/);
  const when = ts ? `${ts[1].slice(4, 6)}-${ts[1].slice(6, 8)} ${ts[2].slice(0, 2)}:${ts[2].slice(2, 4)}` : "";
  const m = run.match(/__(.+)$/);
  let cfg = run === "train" ? "legacy" : (m ? m[1] : run);
  cfg = cfg.replace(/^lokr_/, "").replace(/^lora_/, "").replace(/_/g, " ");
  return [cfg, when].filter(Boolean).join(" · ");
}

// Build the `loras` field for a generate request. Only sent when the LoRA
// picker is applicable (engine + lora_train), so non-engine callers keep their
// legacy behavior. An empty array is intentional: it forces a clean base.
export function loraBody(cfg: Config, sel: LoraSel[]): { loras?: Array<{ path: string; scale: number; label: string }> } {
  if (!cfg.lora_train || !cfg.acestep) return {};
  return { loras: sel.map((l) => ({ path: l.path, scale: l.scale, label: l.label })) };
}

export function LoraPicker({ cfg, value, onChange }:
  { cfg: Config; value: LoraSel[]; onChange: (v: LoraSel[]) => void }) {
  const [opts, setOpts] = useState<Opt[]>([]);
  const [showCustom, setShowCustom] = useState(false);
  const [customPath, setCustomPath] = useState("");

  useEffect(() => {
    if (!cfg.lora_train) return;
    api.loraAdaptersAll().then((r: { adapters?: Array<{ dataset: string; run_label?: string; best_path?: string; final_path?: string }> }) => {
      const list: Opt[] = [];
      for (const a of (r?.adapters || [])) {
        const run = runShort(a.run_label || "");
        const ds = a.dataset.replace(/^crucible_/, "");
        const base = run ? `${ds} · ${run}` : ds;
        if (a.best_path) list.push({ label: `${base} · best`, path: a.best_path });
        if (a.final_path) list.push({ label: `${base} · final`, path: a.final_path });
      }
      setOpts(list);
    }).catch(() => {});
  }, [cfg.lora_train]);

  if (!cfg.lora_train || !cfg.acestep) return null;

  const add = (o: Opt) => onChange([...value, { path: o.path, scale: 0.5, label: o.label }]);
  const addCustom = () => {
    const p = customPath.trim();
    if (!p) return;
    onChange([...value, { path: p, scale: 0.5, label: "custom" }]);
    setCustomPath(""); setShowCustom(false);
  };
  const remove = (i: number) => onChange(value.filter((_, j) => j !== i));
  const setScale = (i: number, s: number) => onChange(value.map((v, j) => (j === i ? { ...v, scale: s } : v)));

  return (
    <div className="space-y-2 rounded-lg border border-[var(--color-accent)]/40 bg-[#2a1c19] p-2.5">
      <div className="flex items-center justify-between">
        <b className="text-xs text-[var(--color-ink)]">LoRA adapters</b>
        <span className="text-[10px] text-[var(--color-muted)]">{value.length ? `${value.length} selected` : "none · base model"}</span>
      </div>

      {value.map((v, i) => (
        <div key={i} className="rounded-md border border-[var(--color-line)] bg-[var(--color-panel2)] p-2">
          <div className="flex items-center justify-between text-[11px] text-[var(--color-ink)]">
            <span className="truncate" title={v.path}>{v.label}</span>
            <button type="button" className="ml-2 shrink-0 text-[var(--color-muted)] hover:text-[var(--color-accent)]" onClick={() => remove(i)}>remove</button>
          </div>
          <div className="mt-1 flex items-center gap-2">
            <span className="text-[10px] text-[var(--color-muted)]">strength</span>
            <input type="range" min={0} max={1.0} step={0.05} value={v.scale}
              onChange={(e) => setScale(i, parseFloat(e.target.value))}
              className="flex-1 accent-[var(--color-accent)]" />
            <span className="w-9 text-right text-[10px] tabular-nums text-[var(--color-muted)]">{v.scale.toFixed(2)}</span>
          </div>
        </div>
      ))}

      <div className="flex items-center gap-2">
        <select className={`${inp} text-xs`} value=""
          onChange={(e) => { const o = opts.find((x) => x.path === e.target.value); if (o) add(o); e.currentTarget.value = ""; }}>
          <option value="">+ add adapter…</option>
          {opts.map((o, i) => <option key={i} value={o.path}>{o.label}</option>)}
        </select>
        <button type="button" className="shrink-0 text-[10px] text-[var(--color-muted)] hover:text-[var(--color-accent)]" onClick={() => setShowCustom((c) => !c)}>custom path</button>
      </div>

      {showCustom && (
        <div className="flex items-center gap-2">
          <input className={`${inp} text-[11px]`} placeholder="E:\\AI\\...\\lokr_weights.safetensors"
            value={customPath} onChange={(e) => setCustomPath(e.target.value)} />
          <button type="button" className="shrink-0 text-[11px] text-[var(--color-accent)]" onClick={addCustom}>add</button>
        </div>
      )}

      <p className="text-[10px] text-[var(--color-muted)]">
        The engine is reconciled to exactly these adapters (verified) before each take — no residual state. 0.3–0.5 nudges style; 0.8+ tends to muddy. Empty = pure base.
      </p>
    </div>
  );
}
