import { useEffect, useRef, useState, type ReactNode } from "react";
import WaveSurfer from "wavesurfer.js";
import RegionsPlugin from "wavesurfer.js/dist/plugins/regions.esm.js";
import { api } from "./api";
import { Field, inp, PrimaryButton, GhostButton } from "./ui";
import { WavePlayer } from "./WavePlayer";

type Sel = { start: number; end: number };

// Waveform with drag-to-select region (single region kept).
function RegionPicker({ url, onSelect }: { url: string; onSelect: (s: Sel | null) => void }) {
  const ref = useRef<HTMLDivElement>(null);
  const ws = useRef<WaveSurfer | null>(null);
  const [playing, setPlaying] = useState(false);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (!ref.current) return;
    const w = WaveSurfer.create({
      container: ref.current, url, height: 96,
      waveColor: "#3a3f4d", progressColor: "#e0512f", cursorColor: "#f0911d",
    });
    const regions = w.registerPlugin(RegionsPlugin.create());
    w.on("ready", () => setReady(true));
    w.on("play", () => setPlaying(true));
    w.on("pause", () => setPlaying(false));
    w.on("finish", () => setPlaying(false));
    w.on("decode", () => regions.enableDragSelection({ color: "rgba(224,81,47,0.25)" }));
    const keepOne = (region: { id: string; start: number; end: number }) => {
      regions.getRegions().forEach((r) => { if (r.id !== region.id) r.remove(); });
      onSelect({ start: region.start, end: region.end });
    };
    regions.on("region-created", keepOne);
    regions.on("region-updated", (r) => onSelect({ start: r.start, end: r.end }));
    ws.current = w;
    return () => { w.destroy(); ws.current = null; };
  }, [url]);

  return (
    <div className="space-y-2">
      <div ref={ref} className="rounded-lg bg-[var(--color-panel)] p-2" />
      <div className="flex items-center gap-2">
        <GhostButton onClick={() => ws.current?.playPause()}>{playing ? "❚❚ pause" : "▶ play"}</GhostButton>
        <span className="text-[11px] text-[var(--color-muted)]">{ready ? "drag across the waveform to select a section" : "loading…"}</span>
      </div>
    </div>
  );
}

function Step({ n, title, active, done, children }: { n: number; title: string; active: boolean; done: boolean; children: ReactNode }) {
  return (
    <div className={`rounded-xl border p-3 transition ${active ? "border-[var(--color-accent)] bg-[#1c1512]" : "border-[var(--color-line)] bg-[var(--color-panel2)]"} ${done && !active ? "opacity-70" : ""}`}>
      <div className="mb-2 flex items-center gap-2">
        <span className={`flex h-5 w-5 flex-none items-center justify-center rounded-full text-[10px] font-bold ${done ? "bg-emerald-500 text-white" : active ? "bg-[var(--color-accent)] text-white" : "bg-[var(--color-panel)] text-[var(--color-muted)]"}`}>{done ? "✓" : n}</span>
        <span className="text-xs font-semibold text-[var(--color-ink)]">{title}</span>
      </div>
      {children}
    </div>
  );
}

const mb = (n: number) => n ? `${(n / 1048576).toFixed(1)} MB` : "";

// Searchable Internet Archive browser → pick a track/quality → fetch it.
function ArchiveSearch({ onPick, busy }: { onPick: (url: string) => void; busy: boolean }) {
  const [q, setQ] = useState("");
  const [results, setResults] = useState<{ identifier: string; title: string; creator: string; year: string }[]>([]);
  const [openId, setOpenId] = useState("");
  const [tracks, setTracks] = useState<{ title: string; files: { format: string; size: number; url: string }[] }[]>([]);
  const [busyS, setBusyS] = useState(false);
  const [note, setNote] = useState("");

  async function search() {
    if (!q.trim()) return;
    setBusyS(true); setNote("searching…"); setResults([]); setOpenId("");
    try { const d = await api.archiveSearch(q.trim()); setResults(d.results); setNote(d.results.length ? "" : "no results"); }
    catch (e) { setNote("✗ " + (e as Error).message); }
    finally { setBusyS(false); }
  }
  async function openItem(id: string) {
    if (openId === id) { setOpenId(""); return; }
    setOpenId(id); setTracks([]); setNote("loading tracks…");
    try { const d = await api.archiveItem(id); setTracks(d.tracks); setNote(""); }
    catch (e) { setNote("✗ " + (e as Error).message); }
  }

  return (
    <div className="space-y-2">
      <div className="flex gap-2">
        <input className={inp} placeholder="search the Internet Archive (e.g. Nightwish, opera aria)" value={q}
          onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && search()} />
        <GhostButton onClick={search} className="flex-none">{busyS ? "…" : "Search"}</GhostButton>
      </div>
      {note && <p className="text-[10px] text-[var(--color-muted)]">{note}</p>}
      <div className="max-h-72 space-y-1 overflow-y-auto">
        {results.map((r) => (
          <div key={r.identifier} className="rounded-lg border border-[var(--color-line)] bg-[var(--color-panel)]">
            <button onClick={() => openItem(r.identifier)} className="flex w-full items-start gap-1.5 p-2 text-left">
              <span className="text-[var(--color-muted)]">{openId === r.identifier ? "▾" : "▸"}</span>
              <span className="text-[11px] text-[var(--color-ink)]">{r.title || r.identifier}
                {(r.creator || r.year) && <span className="text-[var(--color-muted)]"> — {[r.creator, r.year].filter(Boolean).join(" · ")}</span>}</span>
            </button>
            {openId === r.identifier && (
              <div className="space-y-1 px-2 pb-2">
                {tracks.length === 0 && <p className="text-[10px] text-[var(--color-muted)]">no audio files</p>}
                {tracks.map((t) => (
                  <div key={t.title} className="flex flex-wrap items-center gap-1.5 border-t border-[var(--color-line)] py-1">
                    <span className="mr-auto text-[10px] text-[var(--color-muted)]">{t.title}</span>
                    {t.files.map((f) => (
                      <button key={f.url} disabled={busy} onClick={() => onPick(f.url)}
                        className="rounded border border-[var(--color-line)] bg-[var(--color-panel2)] px-1.5 py-0.5 text-[9px] text-[var(--color-accent2)] hover:border-[var(--color-accent)] disabled:opacity-50"
                        title={`fetch ${f.format} ${mb(f.size)}`}>{f.format}{f.size ? ` · ${mb(f.size)}` : ""}</button>
                    ))}
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export function ImportForm({ goTo, onImported }: { goTo: (m: string) => void; onImported?: () => void }) {
  const [importId, setImportId] = useState("");
  const [importTitle, setImportTitle] = useState("");
  const [urlInput, setUrlInput] = useState("");
  const [fetching, setFetching] = useState(false);
  const [url, setUrl] = useState("");
  const [voiceMode, setVoiceMode] = useState(false);
  const [sel, setSel] = useState<Sel | null>(null);
  const [extracting, setExtracting] = useState(false);
  const [vocalUrl, setVocalUrl] = useState("");
  const [name, setName] = useState("");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [msg, setMsg] = useState("");
  const hasSource = !!importId;

  function reset() { setSel(null); setVocalUrl(""); setSaved(false); setMsg(""); setVoiceMode(false); }

  async function pick(f: File | null) {
    if (!f) return;
    reset(); setMsg("importing…");
    try {
      const fd = new FormData();
      fd.append("file", f);
      const d = await api.importUpload(fd);
      setImportId(d.import_id); setImportTitle(d.title || f.name);
      setUrl(d.audio_url + "?t=" + Date.now());
      setMsg("✓ imported to your library — usable in Cover, Restyle, Stems, Backing, etc.");
      onImported?.();
    } catch (e) { setMsg("✗ " + (e as Error).message); }
  }

  async function doFetch(u: string) {
    if (!u.trim()) return setMsg("Paste a link first.");
    setFetching(true); reset(); setMsg("downloading audio…");
    try {
      const d = await api.importFetch(u.trim());
      setImportId(d.import_id); setImportTitle(d.title || "audio");
      setUrl(d.audio_url + "?t=" + Date.now());
      setMsg("✓ imported to your library — usable in Cover, Restyle, Stems, Backing, etc.");
      onImported?.();
    } catch (e) { setMsg("✗ " + (e as Error).message); }
    finally { setFetching(false); }
  }
  const fetchUrl = () => doFetch(urlInput);

  async function extract() {
    if (!hasSource) return setMsg("Import a song first.");
    setExtracting(true); setVocalUrl(""); setMsg("extracting vocal on the Mac (Demucs)…");
    try {
      const fd = new FormData();
      fd.append("import_id", importId);
      fd.append("start", String(sel?.start ?? 0));
      fd.append("end", String(sel?.end ?? 0));
      const d = await api.importExtract(fd);
      setVocalUrl(d.vocal_url + "?t=" + Date.now());
      setMsg(`extracted ${d.duration}s — preview below, then save as a voice`);
    } catch (e) { setMsg("✗ " + (e as Error).message); }
    finally { setExtracting(false); }
  }

  async function save() {
    if (!vocalUrl) return setMsg("Extract a vocal first.");
    if (!name.trim()) return setMsg("Name the voice.");
    setSaving(true); setMsg("preparing SoulX voice on the 3090 (transcription)…");
    try {
      const fd = new FormData();
      fd.append("name", name.trim());
      fd.append("stem_src", vocalUrl.split("?")[0]);
      fd.append("vocal_sep", "false");
      const d = await api.soulxPrep(fd);
      setSaved(true);
      setMsg("✓ voice ready: " + d.name);
    } catch (e) { setMsg("✗ " + (e as Error).message); }
    finally { setSaving(false); }
  }

  const dur = sel ? sel.end - sel.start : 0;
  const selOk = !!sel && dur >= 8 && dur <= 40;
  return (
    <div className="space-y-3">
      <div>
        <h2 className="text-sm font-semibold text-[var(--color-ink)]">Import a song</h2>
        <p className="mt-1 text-xs text-[var(--color-muted)]">Bring any song into your <strong>library</strong> — then it's usable everywhere: <strong>Cover</strong>, <strong>Restyle</strong>, <strong>Stems</strong>, <strong>Backing</strong>, <strong>Master</strong>, layering, etc. (Making a singing voice from it is an optional extra below.)</p>
      </div>

      <Step n={1} title="Import a song (→ your library)" active={!hasSource} done={hasSource}>
        <p className="mb-2 text-[11px] text-[var(--color-muted)]">Three ways to bring audio in. Whichever you use, it's saved to your library as a reusable source track. Personal use only.</p>
        <p className="mb-1 text-[10px] font-semibold text-[var(--color-accent2)]">A · Upload a file</p>
        <p className="mb-1.5 text-[10px] text-[#5a5f6e]">Any audio file you already have (mp3/wav/flac/m4a…).</p>
        <input className={`${inp} mb-3`} type="file" accept="audio/*" onChange={(e) => pick(e.target.files?.[0] ?? null)} />
        <p className="mb-1 text-[10px] font-semibold text-[var(--color-accent2)]">B · Paste a link</p>
        <p className="mb-1.5 text-[10px] text-[#5a5f6e]">A YouTube (or similar) link — click Fetch and it pulls the audio in. Great for streaming-only tracks.</p>
        <div className="mb-3 flex gap-2">
          <input className={inp} placeholder="https://www.youtube.com/watch?v=…" value={urlInput} onChange={(e) => setUrlInput(e.target.value)} />
          <GhostButton onClick={fetchUrl} className="flex-none">{fetching ? "Fetching…" : "Fetch"}</GhostButton>
        </div>
        <p className="mb-1 text-[10px] font-semibold text-[var(--color-accent2)]">C · Search the Internet Archive</p>
        <p className="mb-1.5 text-[10px] text-[#5a5f6e]">Search archive.org, expand an item, and click a format/quality to pull it in. Reliable + huge catalogue (incl. public-domain opera).</p>
        <ArchiveSearch onPick={doFetch} busy={fetching} />
      </Step>

      {hasSource && url && (
        <Step n={2} title="Imported — preview" active done={false}>
          <p className="mb-2 text-[11px] text-[var(--color-muted)]"><strong>{importTitle || "your song"}</strong> is in the library now. Pick it as the source in Cover, Restyle, Stems, Backing, Master, or layering.</p>
          <WavePlayer url={url} />
          <div className="mt-2 flex flex-wrap gap-2">
            <GhostButton onClick={() => goTo("restyle")}>Use in Restyle / Cover →</GhostButton>
            <GhostButton onClick={() => goTo("stems")}>Split into stems →</GhostButton>
            {!voiceMode && <GhostButton onClick={() => setVoiceMode(true)}>Make a singing voice from it →</GhostButton>}
          </div>
        </Step>
      )}

      {voiceMode && url && (
        <Step n={3} title="Optional — select a section (for a singing voice)" active={hasSource && !vocalUrl} done={!!sel}>
          <p className="mb-2 text-[11px] text-[var(--color-muted)]">Drag across the waveform to select <strong>10–30s</strong> where the singing is clear and prominent — a chorus is ideal. Avoid instrumental-only parts.</p>
          <RegionPicker url={url} onSelect={setSel} />
          {sel && <div className={`mt-1 text-[11px] ${selOk ? "text-emerald-400" : "text-[var(--color-accent2)]"}`}>selection {sel.start.toFixed(1)}–{sel.end.toFixed(1)}s ({dur.toFixed(1)}s){selOk ? " ✓ good length" : " — aim for 10–30s"}</div>}
        </Step>
      )}

      {voiceMode && url && (
        <Step n={4} title="Optional — extract the vocal" active={!!sel && !vocalUrl} done={!!vocalUrl}>
          <p className="mb-2 text-[11px] text-[var(--color-muted)]">Pulls just the voice out (Demucs, on the Mac — keeps the GPU free). Then listen: if it's clean, continue; if it's muddy or has bleed, go back and pick a clearer section.</p>
          <PrimaryButton onClick={extract} disabled={extracting}>{extracting ? "Extracting… (~10s)" : sel ? "Extract vocal from selection" : "Select a section first (or extract whole song)"}</PrimaryButton>
          {vocalUrl && <div className="mt-2"><WavePlayer url={vocalUrl} /></div>}
        </Step>
      )}

      {vocalUrl && (
        <Step n={5} title="Optional — save as a SoulX voice" active={!saved} done={saved}>
          <p className="mb-2 text-[11px] text-[var(--color-muted)]">Name it and save. This prepares it for SoulX (transcribes the clip on the GPU — one time, ~30–60s). It then appears in the Voc. Builder's voice picker.</p>
          <Field label="Voice name"><input className={inp} value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. operatic_tenor" /></Field>
          <div className="mt-2"><PrimaryButton onClick={save} disabled={saving}>{saving ? "Preparing… (slow, ~30–60s)" : "Save as SoulX voice"}</PrimaryButton></div>
        </Step>
      )}

      {saved && (
        <Step n={6} title="Use your voice" active done={false}>
          <p className="mb-2 text-[11px] text-[var(--color-muted)]">In the Voc. Builder: compose a melody, pick <strong>SoulX</strong>, choose <strong>{name}</strong> as the voice (optionally turn on RVC re-timbre for a specific singer), then Build vocal.</p>
          <PrimaryButton onClick={() => goTo("vocalbuilder")}>Go to Voc. Builder →</PrimaryButton>
        </Step>
      )}

      {msg && <p className="text-xs text-[var(--color-muted)]">{msg}</p>}
    </div>
  );
}
