import { useEffect, useState } from "react";
import { api, trackLabel, type Config, type LibItem, type Score } from "./api";
import { Field, inp, PrimaryButton, GhostButton, SectionTitle, type RunCtx } from "./ui";
import { useDrafts } from "./drafts";
import { RegionSelector } from "./RegionSelector";
import { PianoRoll } from "./VocalBuilder";
import { WavePlayer } from "./WavePlayer";
import { PluginRam } from "./PluginRam";

type Props = { cfg: Config; busy: boolean } & RunCtx;

type SoloOptions = {
  amp_presets: { id: string; label: string; desc?: string }[];
  di_engines: { id: string; label: string }[];
  genres: { id: string; label: string }[];
  helix_available: boolean;
  pedalboard: boolean;
  kontakt_available: boolean;
  kontakt_ready: boolean;
};

export function SoloBuilderForm(_props: Props) {
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
  const [soloGain, setSoloGain] = d.use("soloGain", "4");      // lead sits ABOVE the backing by default
  const [duck, setDuck] = d.use("duck", "-6");
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
  const [desc, setDesc] = useState("");          // ACE description, editable before composing
  const [listening, setListening] = useState(false);
  const [msg, setMsg] = useState("");
  // staged, individually-auditionable outputs (each cached so a later stage can be
  // re-run without redoing the earlier ones): a) clean DI  b) amped clip  c) final mix
  const [diJobId, setDiJobId] = useState("");
  const [diUrl, setDiUrl] = useState("");
  const [clipJobId, setClipJobId] = useState("");
  const [clipUrl, setClipUrl] = useState("");
  const [mixUrl, setMixUrl] = useState("");
  const [stage, setStage] = useState("");        // "di" | "clip" | "mix" while running
  const [helixName, setHelixName] = useState("");
  const [setupMsg, setSetupMsg] = useState("");
  const [plugKey, setPlugKey] = useState(0);     // bump to refresh the Plugin RAM row

  const clearStages = () => { setDiJobId(""); setDiUrl(""); setClipJobId(""); setClipUrl(""); setMixUrl(""); };
  const clearAmpDown = () => { setClipJobId(""); setClipUrl(""); setMixUrl(""); };   // DI still valid
  const refreshOpts = () => api.soloOptions().then(setOpts).catch(() => {});

  // Plugin setup, mirrored from the Guitar/Tone tabs so sounds can be picked without
  // leaving the solo flow. Both open the real plugin editor; close it to snapshot state.
  async function setupShreddage() {
    setSetupMsg(`Kontakt opening (~25s) — ${opts?.kontakt_ready ? "pick a different Shreddage patch" : "load Shreddage 3 Stratus FREE + pick a patch"}, then CLOSE the window…`);
    try {
      await api.kontaktCapture(); await refreshOpts();
      setDiEngine("kontakt"); clearStages();
      setSetupMsg("✓ Shreddage ready — selected as the DI engine.");
    } catch (e) { setSetupMsg("✗ setup failed: " + (e as Error).message); }
  }
  async function captureHelix() {
    if (!helixName.trim()) return setSetupMsg("Name the tone first.");
    setSetupMsg("Helix window opening — dial your tone, then CLOSE the window to save…");
    try {
      const d = await api.helixCapture(helixName.trim()); await refreshOpts();
      setAmpPreset(`helix:${d.saved}`); clearAmpDown(); setHelixName("");
      setSetupMsg(`✓ saved "${d.saved}" — now in the amp list`);
    } catch (e) { setSetupMsg("✗ capture failed: " + (e as Error).message); }
  }

  useEffect(() => {
    api.library().then((l: LibItem[]) =>
      setTracks(l.filter((it) => ["generate", "music3", "suno", "song", "restyle", "cover", "mix", "source"].includes(it.mode)))
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
      : brain === "listen" ? { brain: "listen", context: desc }   // the edited ACE description
        : { brain: "llm", provider: brain === "claude" ? "claude" : "ollama" };

  async function listen() {
    if (!job) return setMsg("Pick a track first.");
    const s = parseFloat(start) || 0, e = parseFloat(end) || 0;
    if (!(e > s)) return setMsg("Drag a region on the waveform (end after start).");
    setListening(true); setMsg("🎧 ACE is listening to the section… (~6s)");
    try {
      const r = await api.soloListen({ job_id: job, start: s, end: e });
      setDesc(r.heard || "");
      setMsg(r.heard ? "✓ heard it — review/edit the description below, then compose" : "ACE returned no description");
    } catch (err) { setMsg("✗ " + (err as Error).message); }
    finally { setListening(false); }
  }

  async function compose(reroll = false) {
    if (!job) return setMsg("Pick a track first.");
    const s = parseFloat(start) || 0, e = parseFloat(end) || 0;
    if (!(e > s)) return setMsg("Drag a region on the waveform (end after start).");
    if (brain === "listen" && !desc.trim()) return setMsg("Click 🎧 Listen first (or type a description for the note-writer).");
    const useSeed = reroll ? Math.floor(Math.random() * 1e6) : seed;
    setSeed(useSeed);
    setComposing(true);
    setMsg(brain === "listen" ? "composing a lead from the description…"
      : reroll ? "re-rolling solo…" : "composing solo…");
    if (!reroll) setScore(null);
    try {
      const body: any = { job_id: job, start: s, end: e, genre, seed: useSeed, ...brainBody() };
      if (bpm) body.bpm = parseFloat(bpm);
      if (key) body.key = key;
      const r = await api.soloCompose(body);
      setScore(r.score);
      clearStages();                       // new notes invalidate every downstream render
      if (!bpm) setBpm(String(r.bpm));
      if (!key) setKey(r.key);
      setMsg(`composed ${r.score.notes.length} notes · ${r.bpm} BPM · ${r.key}`);
    } catch (err) { setMsg("✗ " + (err as Error).message); }
    finally { setComposing(false); }
  }

  const region = () => ({ s: parseFloat(start) || 0, e: parseFloat(end) || 0 });

  // STAGE A — render the composed notes to a clean DI clip (no amp) and audition it.
  async function runDi() {
    if (!score) return setMsg("Compose a solo first.");
    const { s, e } = region();
    setStage("di");
    setMsg(diEngine === "kontakt"
      ? "rendering DI — first Kontakt render loads the plugin (~40s), then it's instant…"
      : "rendering the clean composed solo (DI, no amp)…");
    try {
      const r = await api.soloDi({ job_id: job, start: s, end: e, score, di_engine: diEngine, genre });
      setDiJobId(r.di_job_id); setDiUrl(r.di_url); clearAmpDown();
      setMsg("✓ DI rendered — hear the raw composed solo below, then amp it");
      if (diEngine === "kontakt") setPlugKey((k) => k + 1);   // Shreddage is now resident
    } catch (err) { setMsg("✗ " + (err as Error).message); }
    finally { setStage(""); }
  }

  // STAGE B — amp the cached DI into a dry, region-length solo clip and audition it.
  async function runClip() {
    if (!diJobId) return setMsg("Render the DI first.");
    const { s, e } = region();
    setStage("clip");
    setMsg(ampPreset ? "amping the solo clip…" : "preparing the dry clip…");
    try {
      const r = await api.soloClip({ job_id: job, start: s, end: e, di_job_id: diJobId, amp_preset: ampPreset });
      setClipJobId(r.clip_job_id); setClipUrl(r.clip_url); setMixUrl("");
      setMsg("✓ amped clip ready — audition it dry below, then mix it in");
    } catch (err) { setMsg("✗ " + (err as Error).message); }
    finally { setStage(""); }
  }

  // STAGE C — overlay the cached amped clip onto the track and audition the result.
  async function runMix() {
    if (!clipJobId) return setMsg("Render the amped clip first.");
    const { s, e } = region();
    setStage("mix");
    setMsg("mixing the solo into the track…");
    try {
      const r = await api.soloMix({
        job_id: job, start: s, end: e, clip_job_id: clipJobId,
        genre, bpm: bpm ? parseFloat(bpm) : undefined, key: key || undefined,
        mix: {
          auto_match: autoMatch, solo_gain_db: parseFloat(soloGain) || 0,
          duck_db: parseFloat(duck) || 0, fade_ms: parseFloat(fade) || 120,
          highpass_hz: parseFloat(hpf) || 0,
        },
      });
      setMixUrl(r.audio_url);
      setMsg("✓ solo mixed in — saved to the library as a new version");
    } catch (err) { setMsg("✗ " + (err as Error).message); }
    finally { setStage(""); }
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

      <PluginRam refreshKey={plugKey} />

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
        <Field label="Composer brain" hint={brain === "listen" ? "ACE LM hears the section (~6-10s), then an LLM writes a fitting lead" : undefined}>
          <select className={inp} value={brain} onChange={(e) => setBrain(e.target.value)}>
            <option value="algorithmic">Algorithmic (instant)</option>
            <option value="local">Local LLM (Gemma)</option>
            <option value="claude">Claude</option>
            <option value="listen">🎧 Listen to the section (ACE)</option>
          </select>
        </Field>
        <Field label="BPM" hint="auto-detected; override if wrong">
          <input className={inp} value={bpm} placeholder="auto" onChange={(e) => setBpm(e.target.value)} />
        </Field>
        <Field label="Key" hint="auto-detected; override if wrong">
          <input className={inp} value={key} placeholder="auto e.g. E minor" onChange={(e) => setKey(e.target.value)} />
        </Field>
      </div>
      {brain === "listen" && (
        <div className="rounded-xl border border-[var(--color-line)] bg-[var(--color-panel2)] p-3 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-[var(--color-ink)]">🎧 What ACE hears — fed to the note-writer (editable)</span>
            <GhostButton onClick={listen}>{listening ? "Listening…" : desc ? "Re-listen" : "Listen to section"}</GhostButton>
          </div>
          <textarea className={inp} rows={4} value={desc}
            placeholder="Click 'Listen to section' to have ACE describe it — then edit before composing (or type your own description). The note-writer uses this to fit the lead to the section."
            onChange={(e) => setDesc(e.target.value)} />
        </div>
      )}

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

          <SectionTitle>3a · Render DI — hear the raw composed solo</SectionTitle>
          <p className="text-[11px] text-[var(--color-muted)]">The clean composed notes, no amp. Re-roll above until the line is right, then move on.</p>
          <Field label="DI engine" hint="the guitar itself (Shreddage = real sampled tone)">
            <select className={inp} value={diEngine} onChange={(e) => { setDiEngine(e.target.value); clearStages(); }}>
              {opts?.di_engines.map((d) => <option key={d.id} value={d.id}>{d.label}</option>)}
            </select>
          </Field>
          {opts?.kontakt_available && (
            <div className="rounded-lg border border-[var(--color-line)] bg-[var(--color-panel2)] p-2.5 space-y-1.5">
              <div className="text-[11px] text-[var(--color-muted)]">{opts.kontakt_ready
                ? "Shreddage is set up. Re-pick to open Kontakt (~25s) and choose a different Stratus patch/articulation, then close the window."
                : "Kontakt is installed. Open it (~25s), load Shreddage 3 Stratus FREE + pick a patch, then close the window — saved for reuse."}</div>
              <GhostButton onClick={setupShreddage}>{opts.kontakt_ready ? "Re-pick Shreddage patch" : "Open Kontakt + capture Shreddage"}</GhostButton>
            </div>
          )}
          <GhostButton onClick={runDi} disabled={!!stage}>{stage === "di" ? "Rendering DI…" : diUrl ? "Re-render DI" : "Render DI"}</GhostButton>
          {diUrl && <WavePlayer url={diUrl} />}

          {diUrl && (
            <>
              <SectionTitle>3b · Amp — hear the solo clip dry</SectionTitle>
              <p className="text-[11px] text-[var(--color-muted)]">The amped solo on its own, gated to the region. Change the amp and re-amp without re-rendering the DI.</p>
              <Field label="Amp / tone" hint="the amplifier stacked on the DI">
                <select className={inp} value={ampPreset} onChange={(e) => { setAmpPreset(e.target.value); clearAmpDown(); }}>
                  <option value="">Clean DI (no amp)</option>
                  {opts?.amp_presets.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
                </select>
              </Field>
              {opts?.helix_available && (
                <div className="rounded-lg border border-[var(--color-line)] bg-[var(--color-panel2)] p-2.5 space-y-1.5">
                  <div className="text-[11px] text-[var(--color-muted)]">Capture a Helix tone: dial amp/cab/FX in Helix once and save it under a name — it joins the amp list here and everywhere.</div>
                  <div className="flex gap-1.5">
                    <input className={inp} placeholder="tone name (e.g. Modern Lead)" value={helixName} onChange={(e) => setHelixName(e.target.value)} />
                    <GhostButton onClick={captureHelix}>Open + capture</GhostButton>
                  </div>
                </div>
              )}
              <GhostButton onClick={runClip} disabled={!!stage}>{stage === "clip" ? "Amping…" : clipUrl ? "Re-amp clip" : "Apply amp"}</GhostButton>
              {clipUrl && <WavePlayer url={clipUrl} />}
            </>
          )}

          {clipUrl && (
            <>
              <SectionTitle>3c · Mix — hear it on the track</SectionTitle>
              <p className="text-[11px] text-[var(--color-muted)]">Overlay the clip onto the original. Tweak the mix and re-mix without redoing the DI or amp.</p>
              <div className="rounded-xl border border-[var(--color-line)] bg-[var(--color-panel2)] p-3 space-y-3">
                <div className="text-xs font-medium text-[var(--color-ink)]">Mix controls</div>
                <label className="flex items-center gap-2 text-sm text-[var(--color-muted)]">
                  <input type="checkbox" checked={autoMatch} onChange={(e) => { setAutoMatch(e.target.checked); setMixUrl(""); }} /> Auto level-match to the section
                </label>
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Solo gain (dB)" hint="lead boost above the section (auto-matched)">
                    <input className={inp} type="number" step="0.5" value={soloGain} onChange={(e) => { setSoloGain(e.target.value); setMixUrl(""); }} />
                  </Field>
                  <Field label="Duck backing (dB)" hint="dip the track under the solo · 0 = off">
                    <input className={inp} type="number" step="1" value={duck} onChange={(e) => { setDuck(e.target.value); setMixUrl(""); }} />
                  </Field>
                  <Field label="Edge fade (ms)" hint="solo + duck boundary fades">
                    <input className={inp} type="number" step="10" value={fade} onChange={(e) => { setFade(e.target.value); setMixUrl(""); }} />
                  </Field>
                  <Field label="High-pass (Hz)" hint="clean solo mud · 0 = off">
                    <input className={inp} type="number" step="10" value={hpf} onChange={(e) => { setHpf(e.target.value); setMixUrl(""); }} />
                  </Field>
                </div>
              </div>
              <PrimaryButton onClick={runMix} disabled={!!stage}>{stage === "mix" ? "Mixing…" : mixUrl ? "Re-mix into track" : "Mix into track"}</PrimaryButton>
              {mixUrl && <WavePlayer url={mixUrl} />}
            </>
          )}
        </>
      )}

      {setupMsg && <p className="text-[11px] text-[var(--color-accent2)]">{setupMsg}</p>}
      {msg && <p className="text-xs text-[var(--color-muted)]">{msg}</p>}
    </div>
  );
}
