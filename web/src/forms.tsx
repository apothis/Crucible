import { useEffect, useState } from "react";
import { DndContext, closestCenter, PointerSensor, useSensor, useSensors, type DragEndEvent } from "@dnd-kit/core";
import { SortableContext, verticalListSortingStrategy, useSortable, arrayMove } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { api, type Config, type LibItem, type SongDraft } from "./api";
import { Field, inp, PrimaryButton, GhostButton, SectionTitle, Slider, pollJob, waitJob, runSync, rid, type RunCtx } from "./ui";
import { PRESETS, SONG_TEMPLATES, type Preset, type SongTemplate } from "./presets";

type FormProps = { cfg: Config; busy: boolean } & RunCtx;

const fail = (ctx: RunCtx, msg: string) =>
  ctx.setResults([{ id: rid(), title: "Can’t run", status: "error", pct: 0, err: msg }]);

// shared tuning controls for Generate + Restyle. `expert` reveals the full grid;
// otherwise only Duration shows and the rest use presets/defaults.
function useTuning(cfg: Config, expert: boolean, hideDuration = false) {
  const firstAvail = cfg.variants.find((v) => v.available);
  const [variant, setVariant] = useState(firstAvail?.id ?? "xl_base");
  const [steps, setSteps] = useState("");
  const [cfgScale, setCfgScale] = useState("");
  const [duration, setDuration] = useState("40");
  const [bpm, setBpm] = useState("170");
  const [keyscale, setKeyscale] = useState("E minor");
  const [seed, setSeed] = useState("");
  const v = cfg.variants.find((x) => x.id === variant);

  const node = expert ? (
    <>
      <SectionTitle>Guided tuning</SectionTitle>
      <div className="grid grid-cols-2 gap-3">
        <Field label="Model">
          <select className={inp} value={variant} onChange={(e) => setVariant(e.target.value)}>
            {cfg.variants.map((vv) => (
              <option key={vv.id} value={vv.id} disabled={!vv.available}>
                {vv.label}{vv.available ? "" : " — not installed"}
              </option>
            ))}
          </select>
        </Field>
        {!hideDuration && <Field label="Duration (s)"><input className={inp} type="number" value={duration} onChange={(e) => setDuration(e.target.value)} /></Field>}
        <Field label="Steps" hint={v ? `def ${v.steps}` : ""}><input className={inp} type="number" placeholder={String(v?.steps ?? "")} value={steps} onChange={(e) => setSteps(e.target.value)} /></Field>
        <Field label="CFG" hint={v ? `def ${v.cfg}` : ""}><input className={inp} type="number" step="0.5" placeholder={String(v?.cfg ?? "")} value={cfgScale} onChange={(e) => setCfgScale(e.target.value)} /></Field>
        <Field label="BPM"><input className={inp} type="number" value={bpm} onChange={(e) => setBpm(e.target.value)} /></Field>
        <Field label="Key">
          <select className={inp} value={keyscale} onChange={(e) => setKeyscale(e.target.value)}>
            {cfg.keys.map((k) => <option key={k}>{k}</option>)}
          </select>
        </Field>
        <Field label="Seed" hint="blank = random"><input className={inp} type="number" placeholder="random" value={seed} onChange={(e) => setSeed(e.target.value)} /></Field>
      </div>
    </>
  ) : hideDuration ? null : (
    <Field label="Duration (s)"><input className={inp} type="number" value={duration} onChange={(e) => setDuration(e.target.value)} /></Field>
  );

  const params = () => {
    const o: Record<string, unknown> = { variant, duration: parseFloat(duration) || 40, bpm: parseInt(bpm) || 120, keyscale };
    if (steps) o.steps = parseInt(steps);
    if (cfgScale) o.cfg = parseFloat(cfgScale);
    if (seed) o.seed = parseInt(seed);
    return o;
  };
  const applyPreset = (p: Preset) => { setBpm(String(p.bpm)); setKeyscale(p.key); };
  return { node, params, applyPreset, bpm: parseInt(bpm) || 120, keyscale };
}

function PresetBar({ onApply }: { onApply: (p: Preset) => void }) {
  return (
    <div>
      <span className="mb-1.5 block text-xs text-[var(--color-muted)]">Subgenre presets</span>
      <div className="flex flex-wrap gap-1.5">
        {PRESETS.map((p) => (
          <button key={p.name} onClick={() => onApply(p)}
            className="rounded-full border border-[var(--color-line)] bg-[var(--color-panel2)] px-3 py-1 text-xs text-[var(--color-muted)] transition hover:border-[var(--color-accent)] hover:text-[var(--color-ink)]">
            {p.name}
          </button>
        ))}
      </div>
    </div>
  );
}

function ModeToggle({ expert, setExpert }: { expert: boolean; setExpert: (b: boolean) => void }) {
  return (
    <div className="flex items-center justify-end gap-1 text-[11px]">
      {[["Simple", false], ["Expert", true]].map(([label, val]) => (
        <button key={label as string} onClick={() => setExpert(val as boolean)}
          className={`rounded-md px-2 py-1 ${expert === val ? "bg-[#2a1c19] text-[var(--color-accent2)]" : "text-[var(--color-muted)] hover:text-[var(--color-ink)]"}`}>
          {label}
        </button>
      ))}
    </div>
  );
}

function PromptFields({ tags, setTags, instrumental, setInstrumental, lyrics, setLyrics }: any) {
  return (
    <>
      <Field label="Style tags" hint="comma-separated; name instruments & tone">
        <textarea className={inp} rows={3} value={tags} onChange={(e) => setTags(e.target.value)} />
      </Field>
      <label className="flex items-center gap-2 text-sm text-[var(--color-muted)]">
        <input type="checkbox" checked={instrumental} onChange={(e) => setInstrumental(e.target.checked)} /> Instrumental (no vocals)
      </label>
      {!instrumental && (
        <Field label="Lyrics" hint="use [Verse] [Chorus] section tags">
          <textarea className={inp} rows={4} value={lyrics} onChange={(e) => setLyrics(e.target.value)} placeholder={"[Verse]\n...\n[Chorus]\n..."} />
        </Field>
      )}
    </>
  );
}

export function GenerateForm({ cfg, busy, ...ctx }: FormProps) {
  const [tags, setTags] = useState("symphonic power metal, heavily distorted electric guitars, double-bass drums, orchestral strings, fast tempo, heroic");
  const [instrumental, setInstrumental] = useState(true);
  const [lyrics, setLyrics] = useState("");
  const [count, setCount] = useState(1);
  const [expert, setExpert] = useState(false);
  const tuning = useTuning(cfg, expert);
  const applyPreset = (p: Preset) => { setTags(p.tags); tuning.applyPreset(p); };

  async function run() {
    if (!tags.trim()) return fail(ctx, "Add style tags first — an empty prompt produces noise.");
    const cards = Array.from({ length: count }, () => ({ id: rid(), title: "queued…", status: "pending" as const, pct: 0 }));
    ctx.setResults(cards);
    for (const c of cards) {
      try {
        const { job_id, seed } = await api.generate({ ...tuning.params(), tags, instrumental, lyrics });
        ctx.patch(c.id, { title: `seed ${seed}`, status: "running", pct: 5 });
        pollJob(job_id, c.id, ctx);
      } catch (e) { ctx.patch(c.id, { status: "error", pct: 0, err: (e as Error).message }); }
    }
  }

  return (
    <div className="space-y-4">
      <ModeToggle expert={expert} setExpert={setExpert} />
      <PresetBar onApply={applyPreset} />
      <PromptFields {...{ tags, setTags, instrumental, setInstrumental, lyrics, setLyrics }} />
      {tuning.node}
      <Field label="Variations" hint="generate several takes to compare">
        <div className="flex gap-1.5">
          {[1, 2, 3, 4].map((n) => (
            <button key={n} onClick={() => setCount(n)}
              className={`flex-1 rounded-lg border py-1.5 text-sm ${count === n ? "border-[var(--color-accent)] bg-[#2a1c19]" : "border-[var(--color-line)] bg-[var(--color-panel2)] text-[var(--color-muted)]"}`}>{n}</button>
          ))}
        </div>
      </Field>
      <PrimaryButton onClick={run} disabled={busy}>{busy ? "Generating…" : count > 1 ? `Generate ${count} takes` : "Generate"}</PrimaryButton>
    </div>
  );
}

export function RestyleForm({ cfg, busy, ...ctx }: FormProps) {
  const [file, setFile] = useState<File | null>(null);
  const [amount, setAmount] = useState(0.7);
  const [tags, setTags] = useState("");
  const [instrumental, setInstrumental] = useState(true);
  const [lyrics, setLyrics] = useState("");
  const [expert, setExpert] = useState(true);
  const tuning = useTuning(cfg, expert);
  const applyPreset = (p: Preset) => { setTags(p.tags); tuning.applyPreset(p); };

  async function run() {
    if (!file) return fail(ctx, "Choose a source track to restyle.");
    if (!tags.trim()) return fail(ctx, "Add style tags (target style).");
    const id = rid();
    ctx.setResults([{ id, title: "queued…", status: "pending", pct: 0 }]);
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("params", JSON.stringify({ ...tuning.params(), tags, instrumental, lyrics, restyle_amount: amount }));
      const { job_id } = await api.restyle(fd);
      ctx.patch(id, { status: "running", pct: 5 });
      pollJob(job_id, id, ctx);
    } catch (e) { ctx.patch(id, { status: "error", pct: 0, err: (e as Error).message }); }
  }

  return (
    <div className="space-y-4">
      <ModeToggle expert={expert} setExpert={setExpert} />
      <Field label="Source track"><input className={inp} type="file" accept="audio/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} /></Field>
      <Slider label="Restyle amount (higher = more change)" value={amount} set={setAmount} min={0.2} max={0.95} step={0.05} />
      <PresetBar onApply={applyPreset} />
      <PromptFields {...{ tags, setTags, instrumental, setInstrumental, lyrics, setLyrics }} />
      {tuning.node}
      <PrimaryButton onClick={run} disabled={busy}>{busy ? "Restyling…" : "Restyle"}</PrimaryButton>
    </div>
  );
}

function useVoices() {
  const [voices, setVoices] = useState<string[]>([]);
  const [status, setStatus] = useState("");
  const reload = () => api.rvcVoices().then((v: any) => {
    setVoices(v.voices || []);
    setStatus(v.available ? (v.voices.length ? "" : "no voices installed") : "RVC unreachable");
  }).catch(() => setStatus("RVC unreachable"));
  useEffect(() => { reload(); }, []);
  return { voices, status, reload };
}

function useLibrary(filter: (it: LibItem) => boolean) {
  const [items, setItems] = useState<LibItem[]>([]);
  useEffect(() => { api.library().then((l: LibItem[]) => setItems(l.filter(filter))).catch(() => {}); }, []);
  return items;
}

export function VocalsForm({ busy, ...ctx }: FormProps) {
  const { voices, status, reload } = useVoices();
  const [file, setFile] = useState<File | null>(null);
  const [voice, setVoice] = useState("");
  const [transpose, setTranspose] = useState("0");
  const [f0, setF0] = useState("rmvpe");
  const [indexRate, setIndexRate] = useState(0.75);
  const [rms, setRms] = useState(0.25);
  const [protect, setProtect] = useState(0.33);
  useEffect(() => { if (!voice && voices.length) setVoice(voices[0]); }, [voices]);

  async function run() {
    if (!file) return fail(ctx, "Choose a guide vocal.");
    if (!voice) return fail(ctx, "No target voice (is RVC running?).");
    const fd = new FormData();
    fd.append("file", file);
    fd.append("params", JSON.stringify({ voice, transpose: +transpose, f0_method: f0, index_rate: indexRate, rms_mix_rate: rms, protect }));
    await runSync(`converting → ${voice}`, () => api.rvcConvert(fd), ctx);
  }

  return (
    <div className="space-y-4">
      <Field label="Guide vocal" hint="a sung take to convert"><input className={inp} type="file" accept="audio/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} /></Field>
      <Field label="Target voice" hint={status}>
        <select className={inp} value={voice} onChange={(e) => setVoice(e.target.value)}>{voices.map((v) => <option key={v}>{v}</option>)}</select>
      </Field>
      <div className="grid grid-cols-2 gap-3">
        <Field label="Transpose" hint="semitones"><input className={inp} type="number" value={transpose} onChange={(e) => setTranspose(e.target.value)} /></Field>
        <Field label="Pitch method">
          <select className={inp} value={f0} onChange={(e) => setF0(e.target.value)}>
            <option>rmvpe</option><option>crepe</option><option>harvest</option><option>pm</option>
          </select>
        </Field>
      </div>
      <Slider label="Index rate" value={indexRate} set={setIndexRate} min={0} max={1} step={0.05} />
      <Slider label="Volume envelope" value={rms} set={setRms} min={0} max={1} step={0.05} />
      <Slider label="Protect" value={protect} set={setProtect} min={0} max={0.5} step={0.01} />
      <PrimaryButton onClick={run} disabled={busy}>{busy ? "Converting…" : "Convert voice"}</PrimaryButton>
      <AddVoices reload={reload} />
    </div>
  );
}

function AddVoices({ reload }: { reload: () => void }) {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  const [sort, setSort] = useState("likes");
  const [results, setResults] = useState<any[]>([]);
  const [url, setUrl] = useState("");
  const [msg, setMsg] = useState("");

  async function search() {
    setMsg("searching…");
    try { setResults((await api.voiceSearch(q, sort)).results || []); setMsg(""); } catch { setMsg("search failed"); }
  }
  async function install(body: any) {
    setMsg("installing to the PC…");
    try { const d = await api.voiceInstall(body); setMsg("✓ installed " + d.name); reload(); }
    catch (e) { setMsg("✗ " + (e as Error).message); }
  }

  return (
    <div className="rounded-xl border border-[var(--color-line)] bg-[var(--color-panel2)] p-3 text-xs">
      <button className="font-medium text-[var(--color-accent2)]" onClick={() => setOpen(!open)}>＋ Add / download voices</button>
      {open && (
        <div className="mt-3 space-y-3">
          <p className="text-[var(--color-muted)]">Installs land on the Windows PC; zips auto-unpack. Browse{" "}
            <a className="text-[var(--color-accent2)]" target="_blank" href="https://voice-models.com/">voice-models.com</a>,{" "}
            <a className="text-[var(--color-accent2)]" target="_blank" href="https://huggingface.co/models?other=rvc">Hugging Face</a>.
          </p>
          <div className="flex gap-2">
            <input className={inp} placeholder="search HF (e.g. metal, dickinson)" value={q} onChange={(e) => setQ(e.target.value)} />
            <select className="rounded-lg border border-[var(--color-line)] bg-[var(--color-panel)] px-2 text-xs" value={sort} onChange={(e) => setSort(e.target.value)}>
              <option value="likes">Top</option><option value="downloads">Downloads</option>
            </select>
            <GhostButton onClick={search}>Go</GhostButton>
          </div>
          {results.map((r) => <RepoRow key={r.id} repo={r} onInstall={install} />)}
          <div className="flex gap-2">
            <input className={inp} placeholder="install from URL (.zip/.pth)" value={url} onChange={(e) => setUrl(e.target.value)} />
            <GhostButton onClick={() => url && install({ url })}>Install</GhostButton>
          </div>
          {msg && <p className="text-[var(--color-muted)]">{msg}</p>}
        </div>
      )}
    </div>
  );
}

function RepoRow({ repo, onInstall }: { repo: any; onInstall: (b: any) => void }) {
  const [files, setFiles] = useState<any[] | null>(null);
  return (
    <div className="border-b border-[var(--color-line)] py-1.5">
      <button className="text-left text-[var(--color-ink)]" onClick={async () => setFiles(files ? null : (await api.voiceRepo(repo.id)).voices)}>
        {repo.id} <span className="text-[var(--color-muted)]">♥{repo.likes}</span>
      </button>
      {files?.map((v, i) => (
        <div key={i} className="flex items-center justify-between py-1 pl-3 text-[var(--color-muted)]">
          <span>{v.name}{v.zip ? " (zip)" : v.index ? " +index" : ""}</span>
          <GhostButton onClick={() => onInstall(v.zip ? { repo: repo.id, zip: v.zip, name: v.name } : { repo: repo.id, pth: v.pth, index: v.index, name: v.name })}>install</GhostButton>
        </div>
      ))}
    </div>
  );
}

export function SwapForm({ busy, ...ctx }: FormProps) {
  const { voices, status } = useVoices();
  const songs = useLibrary((it) => (it.mode === "generate" || it.mode === "restyle") && !it.params.instrumental);
  const [job, setJob] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [voice, setVoice] = useState("");
  const [transpose, setTranspose] = useState("0");
  const [vGain, setVGain] = useState("0");
  const [iGain, setIGain] = useState("0");
  useEffect(() => { if (!voice && voices.length) setVoice(voices[0]); }, [voices]);

  async function run() {
    if (!file && !job) return fail(ctx, "Choose a vocal song (library or upload).");
    if (!voice) return fail(ctx, "No target voice.");
    const fd = new FormData();
    fd.append("voice", voice); fd.append("transpose", transpose);
    fd.append("vocal_gain", vGain); fd.append("instr_gain", iGain);
    if (file) fd.append("file", file); else fd.append("job_id", job);
    await runSync(`swap → ${voice}`, () => api.voiceswap(fd), ctx);
  }

  return (
    <div className="space-y-4">
      <p className="text-xs text-[var(--color-muted)]">Pick a vocal song + a voice; it splits, re-timbres the vocal, and remixes over the original instrumental — in sync.</p>
      <Field label="Vocal song (library)">
        <select className={inp} value={job} onChange={(e) => setJob(e.target.value)}>
          <option value="">— choose —</option>
          {songs.map((s) => <option key={s.id} value={s.id}>{s.mode}: {(s.params.tags || "").slice(0, 36)}</option>)}
        </select>
      </Field>
      <Field label="…or upload a song"><input className={inp} type="file" accept="audio/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} /></Field>
      <Field label="Target voice" hint={status}>
        <select className={inp} value={voice} onChange={(e) => setVoice(e.target.value)}>{voices.map((v) => <option key={v}>{v}</option>)}</select>
      </Field>
      <div className="grid grid-cols-3 gap-3">
        <Field label="Transpose"><input className={inp} type="number" value={transpose} onChange={(e) => setTranspose(e.target.value)} /></Field>
        <Field label="Vocal dB"><input className={inp} type="number" value={vGain} onChange={(e) => setVGain(e.target.value)} /></Field>
        <Field label="Instr dB"><input className={inp} type="number" value={iGain} onChange={(e) => setIGain(e.target.value)} /></Field>
      </div>
      <PrimaryButton onClick={run} disabled={busy}>{busy ? "Working…" : "Swap voice"}</PrimaryButton>
    </div>
  );
}

export function StemsForm({ busy, ...ctx }: FormProps) {
  const tracks = useLibrary((it) => ["generate", "restyle", "voiceswap", "mix", "song"].includes(it.mode));
  const [job, setJob] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [mode, setMode] = useState("vocals");

  async function run() {
    if (!file && !job) return fail(ctx, "Choose a track to separate.");
    const id = rid();
    ctx.setResults([{ id, title: "separating on the Mac GPU…", status: "running", pct: 40 }]);
    try {
      const fd = new FormData();
      fd.append("mode", mode);
      if (file) fd.append("file", file); else fd.append("job_id", job);
      const d = await api.stems(fd);
      ctx.setResults(d.stems.map((s: any) => ({ id: rid(), title: s.name, status: "done", pct: 100, url: s.url })));
    } catch (e) { ctx.patch(id, { status: "error", pct: 0, err: (e as Error).message }); }
  }

  return (
    <div className="space-y-4">
      <p className="text-xs text-[var(--color-muted)]">Split a track on the Mac GPU (parallel with the 3090) — isolate a vocal or split fully.</p>
      <Field label="From a library track">
        <select className={inp} value={job} onChange={(e) => setJob(e.target.value)}>
          <option value="">— choose —</option>
          {tracks.map((t) => <option key={t.id} value={t.id}>{t.mode}: {(t.params.tags || t.params.voice || "").slice(0, 36)}</option>)}
        </select>
      </Field>
      <Field label="…or upload"><input className={inp} type="file" accept="audio/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} /></Field>
      <Field label="Split mode">
        <select className={inp} value={mode} onChange={(e) => setMode(e.target.value)}>
          <option value="vocals">Vocals + instrumental (2-stem)</option>
          <option value="all">Full split (vocals / drums / bass / other)</option>
        </select>
      </Field>
      <PrimaryButton onClick={run} disabled={busy}>{busy ? "Separating…" : "Separate"}</PrimaryButton>
    </div>
  );
}

export function MixForm({ busy, ...ctx }: FormProps) {
  const [sources, setSources] = useState<{ label: string; url: string }[]>([]);
  const [rows, setRows] = useState([{ src: "", gain: "0", offset: "0" }, { src: "", gain: "0", offset: "0" }]);
  const [norm, setNorm] = useState(true);
  useEffect(() => { api.sources().then(setSources).catch(() => {}); }, []);

  const upd = (i: number, k: string, val: string) => setRows((r) => r.map((row, j) => j === i ? { ...row, [k]: val } : row));

  async function run() {
    const tracks = rows.filter((r) => r.src).map((r) => ({ src: r.src, gain_db: parseFloat(r.gain) || 0, offset: parseFloat(r.offset) || 0 }));
    if (!tracks.length) return fail(ctx, "Choose at least one track.");
    await runSync("mixing down…", () => api.mix({ tracks, normalize: norm }), ctx);
  }

  return (
    <div className="space-y-3">
      <p className="text-xs text-[var(--color-muted)]">Layer tracks (instrumental + vocal) with level and start offset.</p>
      {rows.map((r, i) => (
        <div key={i} className="flex items-center gap-2">
          <select className={`${inp} flex-1`} value={r.src} onChange={(e) => upd(i, "src", e.target.value)}>
            <option value="">— track —</option>
            {sources.map((s) => <option key={s.url} value={s.url}>{s.label}</option>)}
          </select>
          <input className="w-14 rounded-lg border border-[var(--color-line)] bg-[var(--color-panel2)] px-2 py-2 text-xs" type="number" title="dB" value={r.gain} onChange={(e) => upd(i, "gain", e.target.value)} />
          <input className="w-14 rounded-lg border border-[var(--color-line)] bg-[var(--color-panel2)] px-2 py-2 text-xs" type="number" step="0.1" title="offset s" value={r.offset} onChange={(e) => upd(i, "offset", e.target.value)} />
          <GhostButton onClick={() => setRows((rs) => rs.filter((_, j) => j !== i))}>✕</GhostButton>
        </div>
      ))}
      <div className="flex items-center justify-between">
        <GhostButton onClick={() => setRows((r) => [...r, { src: "", gain: "0", offset: "0" }])}>+ add track</GhostButton>
        <label className="flex items-center gap-1.5 text-xs text-[var(--color-muted)]"><input type="checkbox" checked={norm} onChange={(e) => setNorm(e.target.checked)} /> normalize</label>
      </div>
      <PrimaryButton onClick={run} disabled={busy}>{busy ? "Mixing…" : "Mix down"}</PrimaryButton>
    </div>
  );
}

// ---------------- Song Constructor ----------------
// A visual, draggable song-structure builder. Two drive modes:
//  (a) compile — ordered blocks → one ACE-Step structured-lyrics prompt + total
//      duration, one /api/generate (order/lyrics honored, section length approx);
//  (b) stitch — generate each block to its exact length, then crossfade-concat
//      via /api/stitch (exact lengths, per-block re-roll, lockable blocks).
const SECTION_TYPES = ["Intro", "Verse", "Pre-Chorus", "Chorus", "Bridge", "Solo", "Breakdown", "Outro"] as const;
const DEFAULT_SECS: Record<string, number> = {
  Intro: 8, Verse: 24, "Pre-Chorus": 12, Chorus: 24, Bridge: 16, Solo: 20, Breakdown: 16, Outro: 12,
};
type Block = { id: string; type: string; seconds: number; lyrics: string; locked: boolean; url?: string };

const aceTag = (type: string) => `[${type.toLowerCase()}]`;
const blockTagged = (b: Block) => {
  const body = b.lyrics.trim();
  return body ? `${aceTag(b.type)}\n${body}` : aceTag(b.type);
};
const compileLyrics = (blocks: Block[]) => blocks.map(blockTagged).join("\n\n");

const newBlock = (type: string): Block =>
  ({ id: rid(), type, seconds: DEFAULT_SECS[type] ?? 16, lyrics: "", locked: false });

const SECTION_ABBR: Record<string, string> = {
  Intro: "In", Verse: "V", "Pre-Chorus": "PC", Chorus: "C", Bridge: "Br",
  Solo: "Solo", Breakdown: "Bd", Outro: "Out",
};
const templateSummary = (t: SongTemplate) => t.sections.map((s) => SECTION_ABBR[s.type] ?? s.type).join(" · ");
const templateLength = (t: SongTemplate) => t.sections.reduce((s, x) => s + x.seconds, 0);

function TemplateGrid({ active, onPick }: { active: string; onPick: (t: SongTemplate) => void }) {
  return (
    <div>
      <span className="mb-1.5 block text-xs text-[var(--color-muted)]">Song templates <span className="text-[#5a5f6e]">· layout + tuned style</span></span>
      <div className="grid grid-cols-2 gap-1.5">
        {SONG_TEMPLATES.map((t) => (
          <button key={t.name} onClick={() => onPick(t)} title={t.description}
            className={`rounded-lg border p-2 text-left transition ${
              active === t.name ? "border-[var(--color-accent)] bg-[#2a1c19]"
                                : "border-[var(--color-line)] bg-[var(--color-panel2)] hover:border-[var(--color-accent)]"}`}>
            <div className="flex items-baseline justify-between gap-1">
              <span className="text-xs font-medium text-[var(--color-ink)]">{t.name}</span>
              <span className="text-[9px] text-[var(--color-muted)]">{t.sections.length}· {templateLength(t)}s</span>
            </div>
            <div className="mt-0.5 truncate text-[10px] text-[var(--color-accent2)]" title={t.sections.map((s) => s.type).join(" · ")}>{templateSummary(t)}</div>
            <div className="mt-0.5 line-clamp-2 text-[10px] text-[var(--color-muted)]">{t.description}</div>
          </button>
        ))}
      </div>
    </div>
  );
}

function SortableBlock({ b, drive, instrumental, upd, remove }: {
  b: Block; drive: "compile" | "stitch"; instrumental: boolean;
  upd: (id: string, patch: Partial<Block>) => void; remove: (id: string) => void;
}) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({ id: b.id });
  const style = { transform: CSS.Transform.toString(transform), transition };
  return (
    <div ref={setNodeRef} style={style}
      className={`rounded-xl border bg-[var(--color-panel2)] p-2.5 ${isDragging ? "border-[var(--color-accent)] opacity-70 shadow-lg" : "border-[var(--color-line)]"}`}>
      <div className="flex items-center gap-2">
        <span {...attributes} {...listeners} className="cursor-grab touch-none select-none text-[var(--color-muted)] hover:text-[var(--color-ink)]" title="drag to reorder">⠿</span>
        <select className={`${inp} flex-1 py-1.5`} value={b.type} onChange={(e) => upd(b.id, { type: e.target.value })}>
          {SECTION_TYPES.map((t) => <option key={t}>{t}</option>)}
        </select>
        <input className="w-16 rounded-lg border border-[var(--color-line)] bg-[var(--color-panel)] px-2 py-1.5 text-xs" type="number" min={1} title="length (s)" value={b.seconds} onChange={(e) => upd(b.id, { seconds: parseInt(e.target.value) || 0 })} />
        <span className="text-[10px] text-[var(--color-muted)]">s</span>
        {drive === "stitch" && (
          <button title={b.locked ? "locked — reuse this take on re-run" : "lock this take"} onClick={() => upd(b.id, { locked: !b.locked })}
            className={`text-sm ${b.locked && b.url ? "text-[var(--color-accent2)]" : "text-[var(--color-muted)] hover:text-[var(--color-ink)]"}`}>{b.locked ? "🔒" : "🔓"}</button>
        )}
        <button onClick={() => remove(b.id)} className="text-[var(--color-muted)] hover:text-red-400" title="remove">✕</button>
      </div>
      {!instrumental && (
        <textarea className={`${inp} mt-2 text-xs`} rows={2} placeholder={`${b.type} lyrics (optional)`} value={b.lyrics} onChange={(e) => upd(b.id, { lyrics: e.target.value })} />
      )}
    </div>
  );
}

export function SongForm({ cfg, busy, onSong, ...ctx }: FormProps & { onSong?: (s: SongDraft) => void }) {
  const [blocks, setBlocks] = useState<Block[]>(() =>
    ["Intro", "Verse", "Chorus", "Verse", "Chorus", "Solo", "Chorus", "Outro"].map(newBlock));
  const [tags, setTags] = useState(PRESETS[0].tags);
  const [instrumental, setInstrumental] = useState(false);
  const [drive, setDrive] = useState<"compile" | "stitch">("compile");
  const [crossfade, setCrossfade] = useState(1);
  const [expert, setExpert] = useState(false);
  const [tpl, setTpl] = useState("");
  const [dirty, setDirty] = useState(false); // arrangement edited since last template/default
  const tuning = useTuning(cfg, expert, true); // duration is computed from blocks
  const applyPreset = (p: Preset) => { setTags(p.tags); tuning.applyPreset(p); };
  const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 5 } }));

  const applyTemplate = (t: SongTemplate) => {
    if (dirty && blocks.length &&
        !window.confirm(`Replace your current arrangement with the “${t.name}” template? Your section-block edits will be lost.`)) return;
    setBlocks(t.sections.map((s) => ({ ...newBlock(s.type), seconds: s.seconds })));
    if (t.instrumental !== undefined) setInstrumental(t.instrumental);
    setTags(t.tags);
    tuning.applyPreset({ name: t.name, tags: t.tags, bpm: t.bpm, key: t.key });
    setTpl(t.name);
    setDirty(false);
  };

  // publish the arrangement so the Vocal Builder can compose against it
  useEffect(() => {
    onSong?.({
      blocks: blocks.map((b) => ({ type: b.type, seconds: b.seconds, lyrics: b.lyrics })),
      key: tuning.keyscale, bpm: tuning.bpm, tags,
    });
  }, [blocks, tuning.bpm, tuning.keyscale, tags]);

  const total = blocks.reduce((s, b) => s + (b.seconds || 0), 0);
  const upd = (id: string, patch: Partial<Block>) => {
    setBlocks((bs) => bs.map((b) => (b.id === id ? { ...b, ...patch } : b)));
    setDirty(true);
  };
  const remove = (id: string) => { setBlocks((bs) => bs.filter((b) => b.id !== id)); setDirty(true); };
  const add = (type: string) => { setBlocks((bs) => [...bs, newBlock(type)]); setDirty(true); };

  const onDragEnd = (e: DragEndEvent) => {
    const { active, over } = e;
    if (!over || active.id === over.id) return;
    setBlocks((bs) => {
      const from = bs.findIndex((b) => b.id === active.id);
      const to = bs.findIndex((b) => b.id === over.id);
      return from < 0 || to < 0 ? bs : arrayMove(bs, from, to);
    });
    setDirty(true);
  };

  async function runCompile() {
    const lyrics = compileLyrics(blocks);
    const id = rid();
    ctx.setResults([{ id, title: `song · ${total}s · compiling`, status: "pending", pct: 0 }]);
    try {
      const { job_id, seed } = await api.generate({ ...tuning.params(), duration: total, tags, instrumental, lyrics });
      ctx.patch(id, { title: `song · ${total}s · seed ${seed}`, status: "running", pct: 5 });
      pollJob(job_id, id, ctx);
    } catch (e) { ctx.patch(id, { status: "error", pct: 0, err: (e as Error).message }); }
  }

  async function runStitch() {
    const cards = blocks.map((b, i) => ({ id: rid(), title: `${i + 1}. ${b.type} · ${b.seconds}s`, status: "pending" as const, pct: 0 }));
    const finalId = rid();
    ctx.setResults([...cards, { id: finalId, title: `stitched song · ${total}s`, status: "pending", pct: 0 }]);
    const urls: string[] = [];
    for (let i = 0; i < blocks.length; i++) {
      const b = blocks[i], card = cards[i];
      if (b.locked && b.url) { // reuse a locked block's existing take
        ctx.patch(card.id, { status: "done", pct: 100, url: b.url + "?t=" + Date.now() });
        urls.push(b.url);
        continue;
      }
      try {
        const { job_id } = await api.generate({ ...tuning.params(), duration: b.seconds, tags, instrumental, lyrics: blockTagged(b) });
        ctx.patch(card.id, { status: "running", pct: 5 });
        const url = await waitJob(job_id, card.id, ctx);
        urls.push(url);
        upd(b.id, { url });
      } catch (e) {
        ctx.patch(card.id, { status: "error", pct: 0, err: (e as Error).message });
        ctx.patch(finalId, { status: "error", pct: 0, err: "a block failed — fix it and re-run" });
        return;
      }
    }
    ctx.patch(finalId, { status: "running", pct: 60 });
    try {
      const sections = blocks.map((b) => b.type).join(" · ");
      const d = await api.stitch({ tracks: urls, crossfade_s: crossfade, tags, sections });
      ctx.patch(finalId, { status: "done", pct: 100, url: d.audio_url + "?t=" + Date.now() });
      ctx.onDone();
    } catch (e) { ctx.patch(finalId, { status: "error", pct: 0, err: (e as Error).message }); }
  }

  function run() {
    if (!tags.trim()) return fail(ctx, "Add style tags first — an empty prompt produces noise.");
    if (!blocks.length) return fail(ctx, "Add at least one section block.");
    return drive === "compile" ? runCompile() : runStitch();
  }

  return (
    <div className="space-y-4">
      <ModeToggle expert={expert} setExpert={setExpert} />
      <p className="text-xs text-[var(--color-muted)]">Start from a template, then arrange — drag to reorder, set each length, add optional per-section lyrics.</p>
      <TemplateGrid active={tpl} onPick={applyTemplate} />
      <PresetBar onApply={applyPreset} />
      <Field label="Style tags" hint="shared across the whole song">
        <textarea className={inp} rows={3} value={tags} onChange={(e) => setTags(e.target.value)} />
      </Field>
      <label className="flex items-center gap-2 text-sm text-[var(--color-muted)]">
        <input type="checkbox" checked={instrumental} onChange={(e) => setInstrumental(e.target.checked)} /> Instrumental (section tags still guide the arrangement)
      </label>

      <SectionTitle>Arrangement · {blocks.length} sections · {total}s total</SectionTitle>
      <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={onDragEnd}>
        <SortableContext items={blocks.map((b) => b.id)} strategy={verticalListSortingStrategy}>
          <div className="space-y-2">
            {blocks.map((b) => (
              <SortableBlock key={b.id} b={b} drive={drive} instrumental={instrumental} upd={upd} remove={remove} />
            ))}
          </div>
        </SortableContext>
      </DndContext>
      <div className="flex flex-wrap gap-1.5">
        {SECTION_TYPES.map((t) => (
          <button key={t} onClick={() => add(t)}
            className="rounded-full border border-[var(--color-line)] bg-[var(--color-panel2)] px-2.5 py-1 text-[11px] text-[var(--color-muted)] transition hover:border-[var(--color-accent)] hover:text-[var(--color-ink)]">+ {t}</button>
        ))}
      </div>

      <SectionTitle>Drive mode</SectionTitle>
      <div className="grid grid-cols-2 gap-1.5 text-xs">
        {([["compile", "Compile to one prompt", "Fast · one generation · section length approximate"],
           ["stitch", "Per-block + stitch", "Exact lengths · per-block re-roll · slow (serial GPU)"]] as const).map(([val, label, hint]) => (
          <button key={val} onClick={() => setDrive(val)}
            className={`rounded-lg border p-2.5 text-left transition ${drive === val ? "border-[var(--color-accent)] bg-[#2a1c19]" : "border-[var(--color-line)] bg-[var(--color-panel2)]"}`}>
            <div className="font-medium text-[var(--color-ink)]">{label}</div>
            <div className="mt-0.5 text-[10px] text-[var(--color-muted)]">{hint}</div>
          </button>
        ))}
      </div>
      {drive === "stitch" && <Slider label="Crossfade (s)" value={crossfade} set={setCrossfade} min={0} max={4} step={0.5} />}
      {tuning.node}

      <PrimaryButton onClick={run} disabled={busy}>
        {busy ? "Working…" : drive === "compile" ? `Generate song (${total}s)` : `Generate ${blocks.length} blocks + stitch`}
      </PrimaryButton>
    </div>
  );
}
