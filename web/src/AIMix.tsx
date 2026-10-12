import { useEffect, useState } from "react";
import { api } from "./api";
import { useDrafts } from "./drafts";
import { Field, inp, PrimaryButton, GhostButton, rid, type RunCtx } from "./ui";
import { AIMixPanel } from "./AIMixPanel";

// AI Mix (docs/AI_MIX_PLAN.md, Phase 1): builds a new mix FROM a Suno stem download. The
// full mix in the zip is only the reference: its balance seeds the stem levels, its loudness
// is what both renders are mastered to, and it is filed so the three can be A/B'd blind.
type Track = { job_id: string; label: string; audio_url: string };
type StemDoc = { role: string; target_lufs: number; width: number; low_mono_hz: number;
  chain: { fx: string; hz?: number; bands?: { hz: number; db: number }[]; moves?: { hz: number; db: number }[];
    threshold_db?: number; ratio?: number; attack_db?: number; range_db?: number }[];
  sends: Record<string, number> };
type RunRes = { session: string; title: string; tracks: Track[];
  mixdoc: { profile: string; stems: Record<string, StemDoc> };
  summary: { fit_r: number | null; bpm: number | null; silent: string[]; mix_lufs: number | null;
    preset: string | null; preset_scores: Record<string, number> | null; notes: string[] } };
type PresetInfo = { name: string; label: string; n: number };

const PROFILES = [
  { id: "rock", label: "Rock/metal: vocal + drums forward" },
  { id: "vocal", label: "Vocal-forward" },
  { id: "suno", label: "Suno balance (no level changes)" },
];

function chainText(s: StemDoc): string {
  return s.chain.map((c) => {
    if (c.fx === "highpass") return `HPF ${c.hz}`;
    if (c.fx === "peq") return (c.bands || []).map((b) => `${b.db} dB @ ${b.hz}`).join(", ");
    if (c.fx === "eq") return "EQ " + (c.moves || []).map((m) => `${m.db > 0 ? "+" : ""}${m.db} @ ${Math.round(m.hz)}`).join(", ");
    if (c.fx === "transient") return `punch +${c.attack_db} dB`;
    if (c.fx === "deess") return `de-ess ${c.range_db} dB`;
    if (c.fx === "compressor") return `comp ${c.ratio}:1 @ ${c.threshold_db} dB`;
    return c.fx;
  }).join(" | ");
}

export function AIMixForm({ busy, setCompare, goTo, ...ctx }: { busy: boolean; setCompare: (f: (ids: string[]) => string[]) => void; goTo: (m: string) => void } & RunCtx) {
  const d = useDrafts("aimix");
  const [path, setPath] = d.use("path", "");
  const [profile, setProfile] = d.use("profile", "rock");
  const [preset, setPreset] = d.use("preset", "auto");
  const [presets, setPresets] = useState<PresetInfo[]>([]);
  useEffect(() => { api.aimixPresets().then((r) => setPresets((r as { presets: PresetInfo[] }).presets)).catch(() => setPresets([])); }, []);
  const usePreset = presets.length > 0 && preset !== "";
  const [res, setRes] = useState<RunRes | null>(null);
  const [running, setRunning] = useState(false);

  async function run() {
    if (!path.trim()) { ctx.setResults([{ id: rid(), title: "Give the path to a Suno stems zip or folder.", status: "error", pct: 0 }]); return; }
    const id = rid();
    setRunning(true); setRes(null);
    ctx.setResults([{ id, title: "AI mix: analysing stems + rendering (Mac, about a minute)…", status: "running", pct: 40 }]);
    try {
      const r = await api.aimixRun({ path: path.trim(), profile, preset: usePreset ? preset : "" }) as RunRes;
      setRes(r);
      ctx.setResults(r.tracks.map((t) => ({ id: rid(), title: t.label, status: "done" as const, pct: 100, url: t.audio_url })));
      ctx.onDone();
    } catch (e) { ctx.patch(id, { status: "error", pct: 0, err: (e as Error).message }); }
    finally { setRunning(false); }
  }

  return (
    <div className="space-y-4">
      <p className="text-xs text-[var(--color-muted)]">
        Builds a new mix from a <span className="text-[var(--color-accent2)]">Suno stem download</span>. The full mix in the zip is only the reference: its balance sets the starting stem levels. Then every stem group is measured (level, tone, dynamics, punch, sibilance, width, wetness) against a style preset built from real reference songs, and each processor moves only as far as needed to land inside the style's range, or stays off. You get three loudness-matched tracks to A/B: the Suno mix, a plain stem sum at Suno's balance, and the AI mix. Mac DSP only.
      </p>
      <Field label="Suno stems zip or folder" hint="full path on this Mac, e.g. /Volumes/SSD1/Downloads/Song.zip">
        <input className={inp} value={path} onChange={(e) => setPath(e.target.value)} placeholder="/Volumes/SSD1/Downloads/….zip" />
      </Field>
      <Field label="Style preset" hint="every move is measured against this style's reference songs and only closes the gap">
        <select className={inp} value={presets.length ? preset : ""} onChange={(e) => setPreset(e.target.value)}>
          {presets.length > 0 && <option value="auto">Auto (nearest style to this song)</option>}
          {presets.map((p) => <option key={p.name} value={p.name}>{p.label} · {p.n} songs</option>)}
          <option value="">Fixed rules (no preset)</option>
        </select>
      </Field>
      {!usePreset && (
        <Field label="Profile" hint="fixed rules only: level moves on top of Suno's balance">
          <select className={inp} value={profile} onChange={(e) => setProfile(e.target.value)}>
            {PROFILES.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
          </select>
        </Field>
      )}
      <PrimaryButton onClick={run} disabled={busy || running}>{running ? "Mixing…" : "Analyse + mix"}</PrimaryButton>
      {res && (
        <div className="space-y-3">
          <GhostButton onClick={() => { const ids = res.tracks.map((t) => t.job_id); setCompare(() => ids); goTo("compare"); }}>
            Send all three to Compare (use Blind)
          </GhostButton>
          <p className="text-xs text-[var(--color-muted)]">
            {res.title} · balance fit r {res.summary.fit_r?.toFixed(2) ?? "n/a"} · {res.summary.bpm ? `${Math.round(res.summary.bpm)} BPM` : "tempo n/a"}
            {res.summary.mix_lufs != null ? ` · mastered to ${res.summary.mix_lufs.toFixed(1)} LUFS (the Suno mix)` : ""}
            {res.summary.silent.length ? ` · silent, skipped: ${res.summary.silent.join(", ")}` : ""}
            {res.summary.preset ? ` · style: ${res.summary.preset}` : ""}
            {res.summary.preset_scores ? ` (distance ${Object.entries(res.summary.preset_scores).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(", ")})` : ""}
          </p>
          {res.summary.notes.length > 0 && (
            <div className="text-xs">
              <div className="mb-1 text-[var(--color-muted)]">What the AI changed, and the measurement behind it:</div>
              <ul className="list-disc space-y-0.5 pl-5">{res.summary.notes.map((n, i) => <li key={i}>{n}</li>)}</ul>
            </div>
          )}
          <table className="w-full text-xs">
            <thead className="text-[var(--color-muted)]"><tr><th className="text-left">Stem</th><th className="text-right">Level</th><th className="text-left pl-3">Chain</th><th className="text-left pl-3">Sends</th></tr></thead>
            <tbody>
              {Object.entries(res.mixdoc.stems).map(([n, s]) => (
                <tr key={n} className="border-t border-[var(--color-line)]">
                  <td className="py-1">{n}</td>
                  <td className="text-right tabular-nums">{s.target_lufs.toFixed(1)} LUFS</td>
                  <td className="pl-3">{chainText(s)}{s.width !== 1 ? ` | width ${s.width}` : ""}</td>
                  <td className="pl-3">{Object.entries(s.sends).map(([k, v]) => `${k} ${v}`).join(", ") || "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <AIMixPanel key={res?.session || "none"} sid={res?.session} ctx={ctx} setCompare={setCompare} goTo={goTo} />
    </div>
  );
}
