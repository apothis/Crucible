import { useEffect, useRef, useState } from "react";
import { api, type Config, type LoraStatus, type LoraTrack } from "./api";
import { inp, Field, PrimaryButton, GhostButton } from "./ui";

// Small inline explainer block — used liberally so every step says what it does & why.
function Info({ children }: { children: React.ReactNode }) {
  return <p className="rounded-md border border-[var(--color-line)] bg-[var(--color-panel2)] px-2.5 py-1.5 text-[11px] leading-relaxed text-[var(--color-muted)]">{children}</p>;
}
function Warn({ children }: { children: React.ReactNode }) {
  return <p className="rounded-md border border-amber-500/30 bg-amber-500/5 px-2.5 py-1.5 text-[11px] leading-relaxed text-amber-300/90">{children}</p>;
}
function Step({ n, title, sub, done, disabled, children }: {
  n: number; title: string; sub?: string; done?: boolean; disabled?: boolean; children: React.ReactNode;
}) {
  return (
    <div className={`rounded-xl border p-3.5 ${disabled ? "border-[var(--color-line)] opacity-55" : "border-[var(--color-line)]"} bg-[var(--color-panel)]`}>
      <div className="mb-2 flex items-center gap-2">
        <span className={`flex h-6 w-6 flex-none items-center justify-center rounded-full text-[11px] font-bold ${done ? "bg-[var(--color-accent)] text-white" : "bg-[var(--color-panel2)] text-[var(--color-muted)]"}`}>{done ? "✓" : n}</span>
        <div>
          <div className="text-sm font-semibold text-[var(--color-ink)]">{title}</div>
          {sub && <div className="text-[11px] text-[var(--color-muted)]">{sub}</div>}
        </div>
      </div>
      <div className="space-y-2.5">{children}</div>
    </div>
  );
}
function srcBadge(src: string) {
  const map: Record<string, [string, string]> = {
    lrclib: ["LRCLIB", "bg-emerald-500/15 text-emerald-300"],
    "lyrics.ovh": ["lyrics.ovh", "bg-emerald-500/15 text-emerald-300"],
    whisper: ["whisper ⚠", "bg-amber-500/15 text-amber-300"],
    "": ["none", "bg-[var(--color-panel)] text-[var(--color-muted)]"],
  };
  const [label, cls] = map[src] || map[""];
  return <span className={`rounded px-1.5 py-0.5 text-[10px] ${cls}`}>{label}</span>;
}

export function LoraTrainingForm(_: { cfg: Config }) {
  const [status, setStatus] = useState<LoraStatus | null>(null);
  const [dataset, setDataset] = useState("crucible_metal");
  const [tracks, setTracks] = useState<LoraTrack[]>([]);
  const [instrumental, setInstrumental] = useState(false);
  const [online, setOnline] = useState(true);
  const [whisper, setWhisper] = useState("small");
  const [adding, setAdding] = useState("");
  const [scanned, setScanned] = useState(false);
  const [samples, setSamples] = useState<Record<string, any>[]>([]);
  const [savingIdx, setSavingIdx] = useState(-1);
  const [method, setMethod] = useState<"lokr" | "lora">("lokr");
  const [epochs, setEpochs] = useState(500);
  const [advanced, setAdvanced] = useState(false);
  const [rank, setRank] = useState(64);
  const [alpha, setAlpha] = useState(128);
  const [gradckpt, setGradckpt] = useState(false);
  const [msg, setMsg] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  const refresh = () => api.loraStatus().then(setStatus).catch(() => {});
  useEffect(() => { refresh(); }, []);
  const training = status?.training;
  const isTraining = !!training?.is_training;
  // poll: slow when idle, fast while training/working
  useEffect(() => {
    const ms = isTraining ? 4000 : 12000;
    const t = setInterval(refresh, ms);
    return () => clearInterval(t);
  }, [isTraining]);

  const engineOk = !!status?.engine && !status?.engine_error;
  const uploadOk = !!status?.upload && !status?.upload_error;
  const ready = engineOk && uploadOk;

  async function onFiles(files: FileList | null) {
    if (!files || !files.length) return;
    for (const f of Array.from(files)) {
      setAdding(f.name);
      try {
        const fd = new FormData();
        fd.append("file", f);
        fd.append("dataset", dataset);
        fd.append("instrumental", String(instrumental));
        fd.append("allow_online", String(online));
        fd.append("whisper_size", whisper);
        const info: LoraTrack = await api.loraDatasetAdd(fd);
        setTracks((t) => [...t.filter((x) => x.name !== info.name), info]);
      } catch (e) { setMsg(`✗ ${f.name}: ${(e as Error).message}`); }
    }
    setAdding("");
    if (fileRef.current) fileRef.current.value = "";
  }

  async function act(label: string, fn: () => Promise<unknown>) {
    setMsg(`${label}…`);
    try { await fn(); setMsg(`✓ ${label}`); refresh(); }
    catch (e) { setMsg(`✗ ${label}: ${(e as Error).message}`); }
  }

  async function loadSamples() {
    const d: any = await api.loraSamples();
    setSamples(Array.isArray(d) ? d : (d?.samples || d?.data || []));
  }
  const updSample = (i: number, patch: Record<string, any>) =>
    setSamples((ss) => ss.map((s, j) => (j === i ? { ...s, ...patch } : s)));
  async function saveSample(i: number) {
    const s = samples[i];
    const idx = s.sample_idx ?? s.index ?? i;
    setSavingIdx(idx);
    try {
      // engine UpdateSampleRequest fields (merge so we don't drop anything)
      const body: Record<string, any> = { sample_idx: idx };
      for (const k of ["caption", "genre", "prompt_override", "lyrics", "bpm", "keyscale", "timesignature", "language", "is_instrumental"])
        if (s[k] !== undefined) body[k] = s[k];
      await api.loraSamplePut(idx, body);
      setMsg(`✓ saved ${s.name || s.filename || "#" + idx}`);
    } catch (e) { setMsg(`✗ save: ${(e as Error).message}`); }
    finally { setSavingIdx(-1); }
  }
  const sampleName = (s: Record<string, any>, i: number) =>
    s.name || s.filename || (s.audio_path ? String(s.audio_path).split(/[\\/]/).pop() : "") || `#${s.sample_idx ?? i}`;

  const epochHint = tracks.length >= 80 ? "~500 for ~100 tracks" : tracks.length ? "~800 for 10–20 tracks" : "add tracks first";

  return (
    <div className="space-y-3">
      <div>
        <h2 className="text-base font-semibold text-[var(--color-ink)]">Train a Metal LoRA</h2>
        <p className="text-xs text-[var(--color-muted)]">Teach the ACE-Step engine your own metal sound — a small add-on (“adapter”) trained on tracks you own, then switchable on at generation time.</p>
      </div>

      <details className="rounded-xl border border-[var(--color-line)] bg-[var(--color-panel)] p-3 text-xs text-[var(--color-muted)]">
        <summary className="cursor-pointer font-medium text-[var(--color-ink)]">What is a LoRA, and how does this work? ▾</summary>
        <div className="mt-2 space-y-2 leading-relaxed">
          <p>A <b>LoRA</b> is a small file that nudges the frozen base model toward a style — it does <i>not</i> retrain the whole model. Trained on real metal, it teaches ACE-Step the high-gain rhythm walls + aggressive feel it otherwise struggles with.</p>
          <p>The pipeline runs on your Windows GPU box: <b>add tracks</b> (Crucible auto-detects BPM/key and pulls lyrics) → <b>upload</b> → <b>review/correct labels</b> → <b>preprocess</b> to tensors (GPU) → <b>train</b> → <b>export</b> → <b>use</b> with a strength slider on Generate.</p>
          <p><b>LoKr vs LoRA:</b> LoKr is ~10× faster (~5 min vs ~1 h) and great for iterating — start there. Use classic LoRA for a final, maximally-portable adapter. Both train from the same data.</p>
          <p><b>Data:</b> a few dozen tracks <b>you own</b> is enough. Whisper mis-hears screamed/accented vocals, so Crucible pulls lyrics from an online database (LRCLIB) for known songs and only falls back to transcription — but you should still eyeball them in the review step.</p>
        </div>
      </details>

      {/* Preflight */}
      <div className="rounded-xl border border-[var(--color-line)] bg-[var(--color-panel)] p-3">
        <div className="mb-1.5 text-xs font-semibold text-[var(--color-ink)]">Box services</div>
        <div className="space-y-1 text-[11px]">
          <div className="flex items-center gap-2">
            <span className={engineOk ? "text-emerald-400" : "text-red-400"}>{engineOk ? "●" : "○"}</span>
            <span className="text-[var(--color-muted)]">ACE-Step engine {engineOk ? `· ${status?.engine?.data?.loaded_model || "loaded"}` : "— needed for preprocess/train/use"}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className={uploadOk ? "text-emerald-400" : "text-red-400"}>{uploadOk ? "●" : "○"}</span>
            <span className="text-[var(--color-muted)]">Dataset upload helper {uploadOk ? `· ${status?.upload?.base_dir || "ok"}` : "— run LORA-UPLOAD_AUTO_INSTALL.bat on the box, then set lora_upload_host"}</span>
          </div>
        </div>
        {!status && <p className="mt-1 text-[11px] text-[var(--color-muted)]">checking…</p>}
        {status && !ready && <div className="mt-2"><Warn>Some box services aren’t reachable. Steps below stay disabled until they’re up — install/start them on the Windows box, then this updates automatically.</Warn></div>}
      </div>

      {/* Step 1 — add tracks */}
      <Step n={1} title="Add training tracks" sub="audio you own; Crucible auto-labels each" disabled={!uploadOk}>
        <Field label="Dataset name" hint="groups these tracks + the trained adapter (e.g. crucible_metal, thrash_rhythm)">
          <input className={inp} value={dataset} onChange={(e) => setDataset(e.target.value.trim() || "crucible_metal")} disabled={isTraining} />
        </Field>
        <div className="flex flex-wrap gap-3 text-[11px] text-[var(--color-muted)]">
          <label className="flex items-center gap-1.5"><input type="checkbox" checked={online} onChange={(e) => setOnline(e.target.checked)} /> Fetch lyrics online (LRCLIB) for known songs</label>
          <label className="flex items-center gap-1.5"><input type="checkbox" checked={instrumental} onChange={(e) => setInstrumental(e.target.checked)} /> Instrumental (skip lyrics)</label>
          <label className="flex items-center gap-1.5">Whisper fallback
            <select className="rounded border border-[var(--color-line)] bg-[var(--color-panel2)] px-1 py-0.5" value={whisper} onChange={(e) => setWhisper(e.target.value)}>
              {["tiny", "base", "small", "medium"].map((s) => <option key={s}>{s}</option>)}
            </select>
          </label>
        </div>
        <Info>Lyrics come from an online database first (accurate for known songs); whisper is the fallback and gets a ⚠ badge so you know to check it. Name files <b>“Artist - Title”</b> (or keep their tags) to help the online match.</Info>
        <input ref={fileRef} type="file" accept="audio/*" multiple className={inp} disabled={!uploadOk || isTraining} onChange={(e) => onFiles(e.target.files)} />
        {adding && <p className="text-[11px] text-[var(--color-accent2)]">labeling {adding}… (BPM/key + lyrics)</p>}
        {!!tracks.length && (
          <div className="overflow-hidden rounded-lg border border-[var(--color-line)]">
            <table className="w-full text-[11px]">
              <thead className="bg-[var(--color-panel2)] text-[var(--color-muted)]"><tr><th className="px-2 py-1 text-left">Track</th><th className="px-2 py-1">BPM</th><th className="px-2 py-1">Key</th><th className="px-2 py-1">Lyrics</th><th className="px-2 py-1 text-left">Caption seed</th></tr></thead>
              <tbody>
                {tracks.map((t) => (
                  <tr key={t.name} className="border-t border-[var(--color-line)]">
                    <td className="max-w-[140px] truncate px-2 py-1 text-[var(--color-ink)]" title={t.name}>{t.name}</td>
                    <td className="px-2 py-1 text-center">{t.bpm}</td>
                    <td className="px-2 py-1 text-center">{t.keyscale}</td>
                    <td className="px-2 py-1 text-center">{srcBadge(t.lyrics_source)}</td>
                    <td className="max-w-[260px] px-2 py-1">
                      <div className="truncate text-[var(--color-muted)]" title={t.caption || "(empty — edit in Step 2)"}>{t.caption || <span className="italic text-[var(--color-muted)]/60">no seed — edit in Step 2</span>}</div>
                      {!!(t.caption_sources && t.caption_sources.length) && (
                        <div className="mt-0.5 flex flex-wrap gap-1">{t.caption_sources!.map((s) => <span key={s} className="rounded bg-[var(--color-panel)] px-1 py-0.5 text-[9px] text-[var(--color-muted)]">{s}</span>)}</div>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <p className="text-[11px] text-[var(--color-muted)]">{tracks.length} track{tracks.length === 1 ? "" : "s"} uploaded to <b>{dataset}</b>. A few dozen is plenty.</p>
      </Step>

      {/* Step 2 — load + review */}
      <Step n={2} title="Load & review labels" sub="correct the auto-labels before training" done={scanned} disabled={!ready || !tracks.length}>
        <Info>Loads the dataset into the engine so you can fix anything the auto-labeler got wrong — <b>especially whisper-sourced lyrics</b> (the tutorial says correcting them is required for a good result). You can also have the box write style <b>captions</b> with its LM.</Info>
        <div className="flex flex-wrap gap-2">
          <GhostButton onClick={() => act("Load dataset", async () => { await api.loraScan({ dataset, instrumental }); setScanned(true); await loadSamples(); })}>Load dataset into engine</GhostButton>
          <GhostButton onClick={() => act("Auto-caption (box LM)", async () => { await api.loraAutolabel({ only_unlabeled: true }); })}>Auto-caption (box LM)</GhostButton>
          {scanned && <GhostButton onClick={() => act("Refresh", loadSamples)}>Refresh</GhostButton>}
        </div>
        {scanned && !samples.length && <Info>Loaded — no editable samples returned yet (auto-caption may still be running; hit Refresh).</Info>}
        {!!samples.length && (
          <div className="space-y-2">
            <Info>Fix anything wrong below — <b>especially whisper-flagged lyrics</b>. <b>Caption</b> is the style description the model learns from (e.g. “melodic death metal, blast beats, tremolo guitars, growled vocals”). BPM/key were measured locally and are usually right. Click <b>Save</b> on each track you edit.</Info>
            {samples.map((s, i) => {
              const idx = s.sample_idx ?? s.index ?? i;
              const nm = sampleName(s, i);
              const src = tracks.find((t) => nm.includes(t.name) || (t.name && nm.replace(/\.[^.]+$/, "") === t.name));
              return (
                <div key={idx} className="space-y-1.5 rounded-lg border border-[var(--color-line)] bg-[var(--color-panel2)] p-2.5">
                  <div className="flex items-center gap-2 text-[11px]">
                    <span className="flex-1 truncate text-[var(--color-ink)]" title={nm}>{nm}</span>
                    {src && srcBadge(src.lyrics_source)}
                    <span className="text-[var(--color-muted)]">{s.bpm ?? "?"} bpm · {s.keyscale ?? "?"}</span>
                  </div>
                  <input className={`${inp} text-xs`} value={s.caption || ""} placeholder="caption — style description (what the LoRA learns)"
                    onChange={(e) => updSample(i, { caption: e.target.value })} />
                  {!s.is_instrumental && (
                    <textarea className={`${inp} text-xs`} rows={3} value={s.lyrics || ""} placeholder="lyrics (correct any whisper errors)"
                      onChange={(e) => updSample(i, { lyrics: e.target.value })} />
                  )}
                  <button onClick={() => saveSample(i)} disabled={savingIdx === idx}
                    className="rounded border border-[var(--color-line)] px-2 py-0.5 text-[11px] text-[var(--color-muted)] hover:text-[var(--color-ink)] disabled:opacity-50">
                    {savingIdx === idx ? "saving…" : "Save"}
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </Step>

      {/* Step 3 — train */}
      <Step n={3} title="Train" sub="runs on the 3090 — GPU-exclusive" disabled={!ready || !tracks.length}>
        <Field label="Method">
          <div className="flex gap-1.5">
            {([["lokr", "LoKr — fast (recommended)"], ["lora", "LoRA — portable"]] as const).map(([m, label]) => (
              <button key={m} onClick={() => { setMethod(m); setEpochs(500); }} disabled={isTraining}
                className={`flex-1 rounded-lg border py-1.5 text-xs ${method === m ? "border-[var(--color-accent)] bg-[#2a1c19] text-[var(--color-ink)]" : "border-[var(--color-line)] bg-[var(--color-panel2)] text-[var(--color-muted)]"}`}>{label}</button>
            ))}
          </div>
        </Field>
        <Info>{method === "lokr" ? "LoKr trains in ~minutes — best for dialing in your dataset. DoRA is on by default." : "Classic LoRA — slower (~1 h class) but the most widely-compatible adapter format."}</Info>
        <Field label={`Epochs · ${epochHint}`} hint="how many passes over the data; more = stronger but slower / risk of overfit">
          <input className={inp} type="number" min={1} value={epochs} onChange={(e) => setEpochs(parseInt(e.target.value) || 0)} disabled={isTraining} />
        </Field>
        <button onClick={() => setAdvanced((a) => !a)} className="text-[11px] text-[var(--color-muted)] hover:text-[var(--color-ink)]">{advanced ? "▾" : "▸"} Advanced</button>
        {advanced && (
          <div className="space-y-2 rounded-lg border border-[var(--color-line)] bg-[var(--color-panel2)] p-2.5">
            {method === "lora" && <div className="grid grid-cols-2 gap-2">
              <Field label="Rank" hint="capacity (def 64)"><input className={inp} type="number" value={rank} onChange={(e) => setRank(parseInt(e.target.value) || 64)} /></Field>
              <Field label="Alpha" hint="scaling (def 128)"><input className={inp} type="number" value={alpha} onChange={(e) => setAlpha(parseInt(e.target.value) || 128)} /></Field>
            </div>}
            <label className="flex items-center gap-2 text-[11px] text-[var(--color-muted)]"><input type="checkbox" checked={gradckpt} onChange={(e) => setGradckpt(e.target.checked)} /> Gradient checkpointing (slower, lower VRAM — enable if it runs out of memory)</label>
          </div>
        )}
        <Warn>Training fully occupies the 3090 — other box GPU jobs (ComfyUI/RVC/etc.) are freed first and you can’t generate while it runs. First run on XL/4B: watch VRAM (24 GB should fit; LoKr + checkpointing help).</Warn>
        {!isTraining ? (
          <PrimaryButton onClick={() => act("Save + preprocess + train", async () => {
            await api.loraSave({ dataset });
            await api.loraPreprocess({ dataset });
            await api.loraTrain({ dataset, method, train_epochs: epochs, gradient_checkpointing: gradckpt,
              ...(method === "lora" ? { lora_rank: rank, lora_alpha: alpha } : {}) });
          })}>Save + preprocess + train</PrimaryButton>
        ) : (
          <GhostButton onClick={() => act("Stop training", () => api.loraTrainStop())}>Stop training</GhostButton>
        )}
        {training && (
          <div className="rounded-lg border border-[var(--color-line)] bg-[var(--color-panel2)] p-2.5 text-[11px] text-[var(--color-muted)]">
            <div className="flex flex-wrap gap-x-4 gap-y-1">
              <span>status: <b className="text-[var(--color-ink)]">{training.status || "—"}</b></span>
              <span>epoch: {training.current_epoch ?? 0}</span>
              <span>step: {training.current_step ?? 0}</span>
              <span>loss: {training.current_loss != null ? training.current_loss.toFixed(4) : "—"}</span>
              {!!training.estimated_time_remaining && <span>ETA: {Math.round(training.estimated_time_remaining / 60)}m</span>}
            </div>
          </div>
        )}
      </Step>

      {/* Step 4 — use */}
      <Step n={4} title="Export & use" sub="load the adapter, then mix it in on Generate" disabled={!ready}>
        <Info>Exports the trained adapter and loads it into the engine. After that a <b>Metal LoRA</b> toggle + <b>strength</b> slider appears on the Generate tab — low strength (0.2–0.5) nudges the style, higher pushes harder.</Info>
        <div className="flex flex-wrap gap-2">
          <GhostButton onClick={() => act("Export + load adapter", () => api.loraExport({ dataset, load: true }))}>Export + load adapter</GhostButton>
          {status?.lora?.lora_loaded && <GhostButton onClick={() => act("Unload adapter", () => api.loraUnload())}>Unload</GhostButton>}
        </div>
        {status?.lora?.lora_loaded && <Info>Loaded: <b>{(status.lora.adapters || []).join(", ") || "adapter"}</b> · scale {status.lora.lora_scale}. Toggle/scale it on the Generate tab.</Info>}
      </Step>

      {msg && <p className="text-[11px] text-[var(--color-muted)]">{msg}</p>}
    </div>
  );
}
