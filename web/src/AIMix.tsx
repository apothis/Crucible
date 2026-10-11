import { useState } from "react";
import { api } from "./api";
import { useDrafts } from "./drafts";
import { Field, inp, PrimaryButton, GhostButton, rid, type RunCtx } from "./ui";

// AI Mix (docs/AI_MIX_PLAN.md, Phase 1): builds a new mix FROM a Suno stem download. The
// full mix in the zip is only the reference: its balance seeds the stem levels, its loudness
// is what both renders are mastered to, and it is filed so the three can be A/B'd blind.
type Track = { job_id: string; label: string; audio_url: string };
type StemDoc = { role: string; target_lufs: number; width: number; low_mono_hz: number;
  chain: { fx: string; hz?: number; bands?: { hz: number; db: number }[]; threshold_db?: number; ratio?: number }[];
  sends: Record<string, number> };
type RunRes = { session: string; title: string; tracks: Track[];
  mixdoc: { profile: string; stems: Record<string, StemDoc> };
  summary: { fit_r: number | null; bpm: number | null; silent: string[]; mix_lufs: number | null } };

const PROFILES = [
  { id: "rock", label: "Rock/metal: vocal + drums forward" },
  { id: "vocal", label: "Vocal-forward" },
  { id: "suno", label: "Suno balance (no level changes)" },
];

function chainText(s: StemDoc): string {
  return s.chain.map((c) => {
    if (c.fx === "highpass") return `HPF ${c.hz}`;
    if (c.fx === "peq") return (c.bands || []).map((b) => `${b.db} dB @ ${b.hz}`).join(", ");
    if (c.fx === "compressor") return `comp ${c.ratio}:1 @ ${c.threshold_db} dB`;
    return c.fx;
  }).join(" | ");
}

export function AIMixForm({ busy, setCompare, goTo, ...ctx }: { busy: boolean; setCompare: (f: (ids: string[]) => string[]) => void; goTo: (m: string) => void } & RunCtx) {
  const d = useDrafts("aimix");
  const [path, setPath] = d.use("path", "");
  const [profile, setProfile] = d.use("profile", "rock");
  const [res, setRes] = useState<RunRes | null>(null);
  const [running, setRunning] = useState(false);

  async function run() {
    if (!path.trim()) { ctx.setResults([{ id: rid(), title: "Give the path to a Suno stems zip or folder.", status: "error", pct: 0 }]); return; }
    const id = rid();
    setRunning(true); setRes(null);
    ctx.setResults([{ id, title: "AI mix: analysing stems + rendering (Mac, about a minute)…", status: "running", pct: 40 }]);
    try {
      const r = await api.aimixRun({ path: path.trim(), profile }) as RunRes;
      setRes(r);
      ctx.setResults(r.tracks.map((t) => ({ id: rid(), title: t.label, status: "done" as const, pct: 100, url: t.audio_url })));
      ctx.onDone();
    } catch (e) { ctx.patch(id, { status: "error", pct: 0, err: (e as Error).message }); }
    finally { setRunning(false); }
  }

  return (
    <div className="space-y-4">
      <p className="text-xs text-[var(--color-muted)]">
        Builds a new mix from a <span className="text-[var(--color-accent2)]">Suno stem download</span>. The full mix in the zip is used only as the reference: its balance sets the starting stem levels, then rules high-pass, unmask the lead vocal, compress, set width and add reverb/delay sends. You get three loudness-matched tracks to A/B: the Suno mix, a plain stem sum at Suno's balance, and the rules mix. Mac DSP only.
      </p>
      <Field label="Suno stems zip or folder" hint="full path on this Mac, e.g. /Volumes/SSD1/Downloads/Song.zip">
        <input className={inp} value={path} onChange={(e) => setPath(e.target.value)} placeholder="/Volumes/SSD1/Downloads/….zip" />
      </Field>
      <Field label="Profile" hint="level moves on top of Suno's balance">
        <select className={inp} value={profile} onChange={(e) => setProfile(e.target.value)}>
          {PROFILES.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
        </select>
      </Field>
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
          </p>
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
    </div>
  );
}
