import { useEffect, useState } from "react";
import { api, type Config, type SongDraft, type Score, type VocalEngine } from "./api";
import { Field, inp, PrimaryButton, GhostButton, SectionTitle, runSync, type RunCtx } from "./ui";
import { useDrafts } from "./drafts";

type Props = { cfg: Config; busy: boolean; song: SongDraft | null } & RunCtx;

const SECTION_COLORS = ["#e07a3f", "#d4a13f", "#5fa9d4", "#8f7fd4", "#5fc7a0", "#d45f8f", "#9fc75f", "#c75f5f"];

export function VocalBuilderForm({ busy, song, ...ctx }: Props) {
  const d = useDrafts("vocalbuilder");
  const [provider, setProvider] = d.use("provider", "");
  const [score, setScore] = useState<Score | null>(null);
  const [composing, setComposing] = useState(false);
  const [engines, setEngines] = useState<VocalEngine[]>([]);
  const [engine, setEngine] = d.use("engine", "guide");
  const [voices, setVoices] = useState<string[]>([]);
  const [retimbre, setRetimbre] = d.use("retimbre", false);
  const [voice, setVoice] = d.use("voice", "");
  const [transpose, setTranspose] = d.use("transpose", "0");
  const [refs, setRefs] = useState<{ id: string; label: string }[]>([]);
  const [msg, setMsg] = useState("");
  // SoulX engine settings (expert)
  const [expert, setExpert] = d.use("expert", false);
  const [steps, setSteps] = d.use("steps", "200");
  const [cfg, setCfg] = d.use("cfg", "3");
  const [prec, setPrec] = d.use("prec", "fp32");
  const [autoShift, setAutoShift] = d.use("autoShift", true);
  const [pitchShift, setPitchShift] = d.use("pitchShift", "0");
  // SoulX reference-voice library (zero-shot timbre)
  const [refVoices, setRefVoices] = useState<{ name: string; ready: boolean }[]>([]);
  const [refVoice, setRefVoice] = d.use("refVoice", "");
  const [addOpen, setAddOpen] = useState(false);
  const [prepName, setPrepName] = useState("");
  const [prepJob, setPrepJob] = useState("");
  const [prepFile, setPrepFile] = useState<File | null>(null);
  const [prepping, setPrepping] = useState(false);
  const engineOpts = () => ({
    n_steps: parseInt(steps) || 200,
    cfg: parseFloat(cfg) || 3,
    fp16: prec === "fp16",
    auto_shift: autoShift,
    pitch_shift: parseInt(pitchShift) || 0,
    ref_voice: refVoice || undefined,
  });

  useEffect(() => {
    if (engine === "soulx") api.soulxVoices().then((d: any) => setRefVoices(d.voices || [])).catch(() => {});
  }, [engine]);

  async function prepVoice() {
    if (!prepName.trim()) return setMsg("Give the voice a name.");
    if (!prepFile && !prepJob) return setMsg("Pick a library track or upload a clip to prepare.");
    setPrepping(true); setMsg("preparing voice on the 3090 — separation + transcription, this is slow…");
    try {
      const fd = new FormData();
      fd.append("name", prepName.trim());
      if (prepFile) fd.append("file", prepFile); else fd.append("job_id", prepJob);
      const d = await api.soulxPrep(fd);
      const v = await api.soulxVoices();
      setRefVoices(v.voices || []); setRefVoice(d.name); setAddOpen(false);
      setMsg("✓ voice ready: " + d.name);
    } catch (e) { setMsg("✗ " + (e as Error).message); }
    finally { setPrepping(false); }
  }

  useEffect(() => { api.vocalEngines().then((d) => setEngines(d.engines)).catch(() => {}); }, []);
  useEffect(() => { api.rvcVoices().then((v: any) => setVoices(v.voices || [])).catch(() => {}); }, []);
  useEffect(() => { if (!voice && voices.length) setVoice(voices[0]); }, [voices]);
  useEffect(() => {
    api.library().then((l: any[]) => setRefs(l.filter((it) => ["vocal", "generate", "voiceswap"].includes(it.mode))
      .map((it) => ({ id: it.id, label: `${it.mode}: ${(it.params.tags || it.params.voice || it.params.source || it.id).slice(0, 28)}` })))).catch(() => {});
  }, [score]);

  const hasLyrics = !!song?.blocks.some((b) => (b.lyrics || "").trim());
  const eng = engines.find((e) => e.id === engine);

  async function compose() {
    if (!song) return setMsg("Build a song in the Song tab first.");
    if (!hasLyrics) return setMsg("Add lyrics to at least one section (Song tab).");
    setComposing(true); setMsg("composing melody…"); setScore(null);
    try {
      const { score } = await api.melodyCompose({ bpm: song.bpm, key: song.key, blocks: song.blocks, provider });
      setScore(score);
      setMsg(`composed ${score.notes.length} notes via ${score.provider}`);
    } catch (e) { setMsg("✗ " + (e as Error).message); }
    finally { setComposing(false); }
  }

  async function build() {
    if (!score) return setMsg("Compose a melody first.");
    if (eng && !eng.available) return setMsg(`${eng.label} isn't reachable — configure its host or pick another engine.`);
    if (retimbre && !voice) return setMsg("Pick an RVC voice or turn off re-timbre.");
    await runSync(`vocal · ${engine}${retimbre ? " → " + voice : ""}`,
      () => api.vocalBuild({
        score, engine, retimbre, voice, transpose: +transpose,
        opts: engine === "soulx" ? engineOpts() : undefined,
      }), ctx);
  }

  async function exportMidi() {
    if (!score) return;
    const blob = await api.melodyMidi({ score });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "melody.mid"; a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="space-y-4">
      <p className="text-xs text-[var(--color-muted)]">
        AI composes a vocal melody from your Song arrangement — a note per lyric syllable, in the song's key, lifting on choruses — then sings it. Hooks into the <span className="text-[var(--color-accent2)]">Song</span> tab's structure, key &amp; BPM.
      </p>

      {!song || !hasLyrics ? (
        <div className="rounded-xl border border-[var(--color-line)] bg-[var(--color-panel2)] p-4 text-xs text-[var(--color-muted)]">
          Go to the <span className="text-[var(--color-accent2)]">Song</span> tab, lay out sections and add lyrics to a Verse/Chorus, then come back. Detected: {song ? `${song.blocks.length} sections, key ${song.key}, ${song.bpm} BPM` : "no song yet"}.
        </div>
      ) : (
        <>
          <SectionTitle>1 · Compose melody</SectionTitle>
          <div className="text-[11px] text-[var(--color-muted)]">From: {song.blocks.length} sections · key {song.key} · {song.bpm} BPM</div>
          <Field label="Composer brain">
            <select className={inp} value={provider} onChange={(e) => setProvider(e.target.value)}>
              <option value="">Auto (Claude if available, else local)</option>
              <option value="claude">Claude</option>
              <option value="ollama">Local (Gemma)</option>
            </select>
          </Field>
          <GhostButton onClick={compose} className="w-full">{composing ? "Composing…" : score ? "Recompose melody" : "Compose melody"}</GhostButton>
        </>
      )}

      {score && (
        <>
          <PianoRoll score={score} />
          <div className="flex items-center justify-between text-[11px] text-[var(--color-muted)]">
            <span>{score.notes.length} notes · {score.duration}s · key {score.key}</span>
            <button onClick={exportMidi} className="text-[var(--color-accent2)] hover:underline">⤓ export MIDI</button>
          </div>

          <SectionTitle>2 · Sing it</SectionTitle>
          <div className="grid grid-cols-1 gap-1.5">
            {engines.map((e) => (
              <button key={e.id} onClick={() => setEngine(e.id)} disabled={!e.available}
                className={`rounded-lg border p-2.5 text-left transition disabled:opacity-45 ${engine === e.id ? "border-[var(--color-accent)] bg-[#2a1c19]" : "border-[var(--color-line)] bg-[var(--color-panel2)] hover:border-[var(--color-accent)]"}`}>
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-[var(--color-ink)]">{e.label}</span>
                  <span className={`text-[9px] ${e.available ? "text-emerald-400" : "text-[var(--color-muted)]"}`}>{e.available ? "ready" : "needs host"}</span>
                </div>
                <div className="mt-0.5 text-[10px] text-[var(--color-muted)]">{e.desc}</div>
              </button>
            ))}
          </div>

          {engine === "soulx" && (
            <>
              <Field label="SoulX voice" hint="zero-shot timbre — the reference the singing is cloned from">
                <select className={inp} value={refVoice} onChange={(e) => setRefVoice(e.target.value)}>
                  <option value="">Default (bundled English)</option>
                  {refVoices.filter((v) => v.ready).map((v) => <option key={v.name} value={v.name}>{v.name}</option>)}
                </select>
              </Field>
              <div className="rounded-xl border border-[var(--color-line)] bg-[var(--color-panel2)] p-3 text-xs">
                <button className="font-medium text-[var(--color-accent2)]" onClick={() => setAddOpen(!addOpen)}>＋ Add a reference voice</button>
                {addOpen && (
                  <div className="mt-3 space-y-2">
                    <p className="text-[var(--color-muted)]">A ~10–30s solo singing clip in the timbre/power you want. Preprocessing (vocal separation + transcription) runs once on the 3090 — can take ~30–60s.</p>
                    <input className={inp} placeholder="voice name (e.g. operatic_tenor)" value={prepName} onChange={(e) => setPrepName(e.target.value)} />
                    <select className={inp} value={prepJob} onChange={(e) => { setPrepJob(e.target.value); setPrepFile(null); }}>
                      <option value="">— from a library track —</option>
                      {refs.map((r) => <option key={r.id} value={r.id}>{r.label}</option>)}
                    </select>
                    <input className={inp} type="file" accept="audio/*" onChange={(e) => { setPrepFile(e.target.files?.[0] ?? null); setPrepJob(""); }} />
                    <PrimaryButton onClick={prepVoice} disabled={prepping}>{prepping ? "Preparing… (slow)" : "Prepare voice"}</PrimaryButton>
                  </div>
                )}
              </div>
            </>
          )}

          {engine === "soulx" && (
            <div className="rounded-xl border border-[var(--color-line)] bg-[var(--color-panel2)] p-3">
              <button className="text-xs font-medium text-[var(--color-accent2)]" onClick={() => setExpert(!expert)}>⚙ Engine settings (expert)</button>
              {expert && (
                <div className="mt-3 space-y-3">
                  <Field label="Steps" hint="diffusion steps · more = slightly cleaner but slower; quality plateaus ~150–200">
                    <input className={inp} type="number" value={steps} onChange={(e) => setSteps(e.target.value)} />
                  </Field>
                  <Field label="CFG" hint="guidance · higher follows the score more strictly; lower = more expressive vibrato (doesn't fix crackle)">
                    <input className={inp} type="number" step="0.5" value={cfg} onChange={(e) => setCfg(e.target.value)} />
                  </Field>
                  <Field label="Precision" hint="fp32 = cleaner, slower · fp16 = faster, slight artifacts">
                    <select className={inp} value={prec} onChange={(e) => setPrec(e.target.value)}><option>fp32</option><option>fp16</option></select>
                  </Field>
                  <label className="flex items-center gap-2 text-sm text-[var(--color-muted)]">
                    <input type="checkbox" checked={autoShift} onChange={(e) => setAutoShift(e.target.checked)} /> Auto-shift to the reference voice's range
                  </label>
                  <Field label="Pitch shift" hint="semitones · transpose the whole vocal up/down">
                    <input className={inp} type="number" value={pitchShift} onChange={(e) => setPitchShift(e.target.value)} />
                  </Field>
                </div>
              )}
            </div>
          )}

          {eng && !eng.sings_words && (
            <p className="text-[11px] text-[var(--color-muted)]">Note: this engine renders the melody wordlessly (vowel tone). Re-timbre it below for a real voice, or pick SoulX/DiffSinger to sing the actual lyrics.</p>
          )}

          <label className="flex items-center gap-2 text-sm text-[var(--color-muted)]">
            <input type="checkbox" checked={retimbre} onChange={(e) => setRetimbre(e.target.checked)} /> Re-timbre with RVC voice
          </label>
          {retimbre && (
            <div className="grid grid-cols-2 gap-3">
              <Field label="Voice">
                <select className={inp} value={voice} onChange={(e) => setVoice(e.target.value)}>{voices.map((v) => <option key={v}>{v}</option>)}</select>
              </Field>
              <Field label="Transpose" hint="semitones"><input className={inp} type="number" value={transpose} onChange={(e) => setTranspose(e.target.value)} /></Field>
            </div>
          )}

          <PrimaryButton onClick={build} disabled={busy}>{busy ? "Building…" : "Build vocal"}</PrimaryButton>
        </>
      )}

      {msg && <p className="text-xs text-[var(--color-muted)]">{msg}</p>}
    </div>
  );
}

function PianoRoll({ score }: { score: Score }) {
  const notes = score.notes;
  if (!notes.length) return null;
  const lo = Math.min(...notes.map((n) => n.midi)) - 1;
  const hi = Math.max(...notes.map((n) => n.midi)) + 1;
  const rows = hi - lo + 1;
  const rowH = 11, pps = Math.max(8, Math.min(24, 720 / (score.duration || 1)));
  const W = Math.max(320, score.duration * pps), H = rows * rowH;
  const y = (midi: number) => (hi - midi) * rowH;
  return (
    <div className="overflow-x-auto rounded-xl border border-[var(--color-line)] bg-[var(--color-panel)] p-2">
      <svg width={W} height={H + 16} className="block">
        {score.sections.map((s, i) => (
          <g key={i}>
            <rect x={s.start * pps} y={0} width={s.seconds * pps} height={H}
              fill={SECTION_COLORS[i % SECTION_COLORS.length]} opacity={0.06} />
            <line x1={s.start * pps} y1={0} x2={s.start * pps} y2={H} stroke="var(--color-line)" strokeWidth={1} />
            <text x={s.start * pps + 3} y={H + 12} fontSize={9} fill="var(--color-muted)">{s.role}</text>
          </g>
        ))}
        {notes.map((n, i) => {
          const w = Math.max(3, n.dur * pps - 1);
          return (
            <g key={i}>
              <rect x={n.start * pps} y={y(n.midi)} width={w} height={rowH - 2} rx={2}
                fill={SECTION_COLORS[n.section % SECTION_COLORS.length]} opacity={0.9} />
              {w > 16 && <text x={n.start * pps + 2} y={y(n.midi) + rowH - 4} fontSize={7} fill="#1a1410">{n.syllable}</text>}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
