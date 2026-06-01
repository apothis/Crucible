import { useEffect, useState } from "react";
import { api, trackLabel, type Config, type LibItem, type Score } from "./api";
import { Field, inp, PrimaryButton, GhostButton, SectionTitle, runSync, type RunCtx } from "./ui";
import { useDrafts } from "./drafts";
import { RegionSelector } from "./RegionSelector";
import { PianoRoll } from "./VocalBuilder";

type Props = { cfg: Config; busy: boolean } & RunCtx;

type SoloOptions = {
  amp_presets: { id: string; label: string; desc?: string }[];
  di_engines: { id: string; label: string }[];
  genres: { id: string; label: string }[];
  helix_available: boolean;
  pedalboard: boolean;
};

export function SoloBuilderForm({ cfg, busy, ...ctx }: Props) {
  const d = useDrafts("solo");
  const [job, setJob] = d.use("job", "");
  const [start, setStart] = d.use("start", "0");
  const [end, setEnd] = d.use("end", "");
  const [genre, setGenre] = d.use("genre", "");
  const [brain, setBrain] = d.use("brain", "algorithmic");   // algorithmic | local | claude
  const [diEngine, setDiEngine] = d.use("diEngine", "ks");
  const [ampPreset, setAmpPreset] = d.use("ampPreset", "");
  // mix controls
  const [autoMatch, setAutoMatch] = d.use("autoMatch", true);
  const [soloGain, setSoloGain] = d.use("soloGain", "0");
  const [duck, setDuck] = d.use("duck", "-4");
  const [fade, setFade] = d.use("fade", "120");
  const [hpf, setHpf] = d.use("hpf", "0");
  // detected/override key+bpm
  const [bpm, setBpm] = d.use("bpm", "");
  const [key, setKey] = d.use("key", "");

  const [tracks, setTracks] = useState<LibItem[]>([]);
  const [opts, setOpts] = useState<SoloOptions | null>(null);
  const [beats, setBeats] = useState<number[]>([]);
  const [srcUrl, setSrcUrl] = useState("");
  const [score, setScore] = useState<Score | null>(null);
  const [seed, setSeed] = useState(42);
  const [composing, setComposing] = useState(false);
  const [msg, setMsg] = useState("");

  useEffect(() => {
    api.library().then((l: LibItem[]) =>
      setTracks(l.filter((it) => ["generate", "song", "restyle", "cover", "mix", "source"].includes(it.mode)))
    ).catch(() => {});
    api.soloOptions().then(setOpts).catch(() => {});
  }, []);

  // load waveform + beats when the track changes
  useEffect(() => {
    if (!job) { setSrcUrl(""); setBeats([]); return; }
    setSrcUrl(`/api/audio/${job}`);
    let cancelled = false;
    api.beats(job).then((r: any) => { if (!cancelled) setBeats(Array.isArray(r?.beats) ? r.beats : []); })
      .catch(() => { if (!cancelled) setBeats([]); });
    return () => { cancelled = true; };
  }, [job]);

  const brainBody = () =>
    brain === "algorithmic" ? { brain: "algorithmic" }
      : { brain: "llm", provider: brain === "claude" ? "claude" : "ollama" };

  async function compose(reroll = false) {
    if (!job) return setMsg("Pick a track first.");
    const s = parseFloat(start) || 0, e = parseFloat(end) || 0;
    if (!(e > s)) return setMsg("Drag a region on the waveform (end after start).");
    const useSeed = reroll ? Math.floor(Math.random() * 1e6) : seed;
    setSeed(useSeed);
    setComposing(true); setMsg(reroll ? "re-rolling solo…" : "composing solo…");
    if (!reroll) setScore(null);
    try {
      const body: any = { job_id: job, start: s, end: e, genre, seed: useSeed, ...brainBody() };
      if (bpm) body.bpm = parseFloat(bpm);
      if (key) body.key = key;
      const r = await api.soloCompose(body);
      setScore(r.score);
      if (!bpm) setBpm(String(r.bpm));
      if (!key) setKey(r.key);
      setMsg(`composed ${r.score.notes.length} notes · ${r.bpm} BPM · ${r.key}`);
    } catch (err) { setMsg("✗ " + (err as Error).message); }
    finally { setComposing(false); }
  }

  async function render() {
    if (!score) return setMsg("Compose a solo first.");
    const s = parseFloat(start) || 0, e = parseFloat(end) || 0;
    const r = await runSync(`solo · ${genre || "lead"}${ampPreset ? " · " + ampPreset : ""}`,
      () => api.soloRender({
        job_id: job, start: s, end: e, score, di_engine: diEngine, amp_preset: ampPreset,
        genre, bpm: bpm ? parseFloat(bpm) : undefined, key: key || undefined,
        mix: {
          auto_match: autoMatch, solo_gain_db: parseFloat(soloGain) || 0,
          duck_db: parseFloat(duck) || 0, fade_ms: parseFloat(fade) || 120,
          highpass_hz: parseFloat(hpf) || 0,
        },
      }), ctx);
    if (r) setMsg("✓ solo added — saved to the library as a new version");
  }

  async function exportMidi() {
    if (!score) return;
    const blob = await api.melodyMidi({ score });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "solo.mid"; a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="space-y-4">
      <p className="text-xs text-[var(--color-muted)]">
        Compose a guitar <span className="text-[var(--color-accent2)]">solo</span> over a section of an existing track and mix it in — no separation needed.
        Pick a track, drag the region, and it picks up the section's BPM &amp; key, composes a lead, renders it through the DI + amp, and overlays it.
      </p>

      <SectionTitle>1 · Track &amp; region</SectionTitle>
      <Field label="Track">
        <select className={inp} value={job} onChange={(e) => { setJob(e.target.value); setScore(null); setBpm(""); setKey(""); }}>
          <option value="">— pick a library track —</option>
          {tracks.map((t) => <option key={t.id} value={t.id}>{trackLabel(t, tracks, false)}</option>)}
        </select>
      </Field>

      {srcUrl && (
        <RegionSelector url={srcUrl} beats={beats} start={parseFloat(start) || 0} end={parseFloat(end) || 0}
          onChange={(s, e) => { setStart(String(s)); setEnd(String(e)); setScore(null); }} />
      )}

      <SectionTitle>2 · Compose</SectionTitle>
      <div className="grid grid-cols-2 gap-3">
        <Field label="Genre / feel">
          <select className={inp} value={genre} onChange={(e) => setGenre(e.target.value)}>
            <option value="">Default (from key)</option>
            {opts?.genres.map((g) => <option key={g.id} value={g.id}>{g.label}</option>)}
          </select>
        </Field>
        <Field label="Composer brain">
          <select className={inp} value={brain} onChange={(e) => setBrain(e.target.value)}>
            <option value="algorithmic">Algorithmic (instant)</option>
            <option value="local">Local LLM (Gemma)</option>
            <option value="claude">Claude</option>
          </select>
        </Field>
        <Field label="BPM" hint="auto-detected; override if wrong">
          <input className={inp} value={bpm} placeholder="auto" onChange={(e) => setBpm(e.target.value)} />
        </Field>
        <Field label="Key" hint="auto-detected; override if wrong">
          <input className={inp} value={key} placeholder="auto e.g. E minor" onChange={(e) => setKey(e.target.value)} />
        </Field>
      </div>
      <div className="flex gap-2">
        <GhostButton onClick={() => compose(false)} className="flex-1">{composing ? "Composing…" : score ? "Recompose" : "Compose solo"}</GhostButton>
        {score && <GhostButton onClick={() => compose(true)} className="flex-1">🎲 Re-roll</GhostButton>}
      </div>

      {score && (
        <>
          <PianoRoll score={score} />
          <div className="flex items-center justify-between text-[11px] text-[var(--color-muted)]">
            <span>{score.notes.length} notes · {score.duration}s · {score.key} · seed {seed}</span>
            <button onClick={exportMidi} className="text-[var(--color-accent2)] hover:underline">⤓ export MIDI</button>
          </div>

          <SectionTitle>3 · Render &amp; mix</SectionTitle>
          <div className="grid grid-cols-2 gap-3">
            <Field label="DI engine">
              <select className={inp} value={diEngine} onChange={(e) => setDiEngine(e.target.value)}>
                {opts?.di_engines.map((d) => <option key={d.id} value={d.id}>{d.label}</option>)}
              </select>
            </Field>
            <Field label="Amp / tone">
              <select className={inp} value={ampPreset} onChange={(e) => setAmpPreset(e.target.value)}>
                <option value="">Clean DI (no amp)</option>
                {opts?.amp_presets.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
              </select>
            </Field>
          </div>

          <div className="rounded-xl border border-[var(--color-line)] bg-[var(--color-panel2)] p-3 space-y-3">
            <div className="text-xs font-medium text-[var(--color-ink)]">Mix controls</div>
            <label className="flex items-center gap-2 text-sm text-[var(--color-muted)]">
              <input type="checkbox" checked={autoMatch} onChange={(e) => setAutoMatch(e.target.checked)} /> Auto level-match to the section
            </label>
            <div className="grid grid-cols-2 gap-3">
              <Field label="Solo gain (dB)" hint="offset on top of auto-match">
                <input className={inp} type="number" step="0.5" value={soloGain} onChange={(e) => setSoloGain(e.target.value)} />
              </Field>
              <Field label="Duck backing (dB)" hint="dip the track under the solo · 0 = off">
                <input className={inp} type="number" step="1" value={duck} onChange={(e) => setDuck(e.target.value)} />
              </Field>
              <Field label="Edge fade (ms)" hint="solo + duck boundary fades">
                <input className={inp} type="number" step="10" value={fade} onChange={(e) => setFade(e.target.value)} />
              </Field>
              <Field label="High-pass (Hz)" hint="clean solo mud · 0 = off">
                <input className={inp} type="number" step="10" value={hpf} onChange={(e) => setHpf(e.target.value)} />
              </Field>
            </div>
          </div>

          <PrimaryButton onClick={render} disabled={busy}>{busy ? "Rendering…" : "Render & add solo"}</PrimaryButton>
        </>
      )}

      {msg && <p className="text-xs text-[var(--color-muted)]">{msg}</p>}
    </div>
  );
}
