import { useEffect, useState } from "react";
import { DndContext, closestCenter, PointerSensor, useSensor, useSensors, type DragEndEvent } from "@dnd-kit/core";
import { SortableContext, verticalListSortingStrategy, useSortable, arrayMove } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { api, type Config, type Genre, type LibItem, type SongDraft } from "./api";
import { Field, inp, PrimaryButton, GhostButton, SectionTitle, Slider, pollJob, waitJob, runSync, rid, type RunCtx } from "./ui";
import { SONG_TEMPLATES, type Preset, type SongTemplate } from "./presets";

type FormProps = { cfg: Config; busy: boolean } & RunCtx;

const fail = (ctx: RunCtx, msg: string) =>
  ctx.setResults([{ id: rid(), title: "Can’t run", status: "error", pct: 0, err: msg }]);

// Is the Claude provider available (API key configured)? Used to offer it for AI lyrics.
function useClaudeAvail() {
  const [claude, setClaude] = useState(false);
  useEffect(() => { api.llmProviders().then((p: any) => setClaude(!!p.claude)).catch(() => {}); }, []);
  return claude;
}

const lyrProvOf = (v: string) => (v === "claude" ? "claude" : "ollama");

// shared tuning controls for Generate + Restyle. `expert` reveals the full grid;
// otherwise only Duration shows and the rest use presets/defaults.
const SAMPLERS = ["euler", "dpmpp_2m", "res_multistep", "er_sde", "dpmpp_2m_sde", "heun", "uni_pc"];
const SCHEDULERS = ["simple", "beta", "normal", "karras", "ddim_uniform"];

function useTuning(cfg: Config, expert: boolean, hideDuration = false) {
  const firstAvail = cfg.variants.find((v) => v.available);
  const [variant, setVariant] = useState(firstAvail?.id ?? "xl_base");
  const [steps, setSteps] = useState("");
  const [cfgScale, setCfgScale] = useState("");
  const [duration, setDuration] = useState("40");
  const [bpm, setBpm] = useState("170");
  const [keyscale, setKeyscale] = useState("E minor");
  const [seed, setSeed] = useState("");
  const [sampler, setSampler] = useState("euler");
  const [scheduler, setScheduler] = useState("simple");
  const [shift, setShift] = useState("");
  const [apg, setApg] = useState(false);
  const [apgNorm, setApgNorm] = useState("1.3");
  const [apgEta, setApgEta] = useState("1.05");
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
        <Field label="Sampler">
          <select className={inp} value={sampler} onChange={(e) => setSampler(e.target.value)}>
            {SAMPLERS.map((s) => <option key={s}>{s}</option>)}
          </select>
        </Field>
        <Field label="Scheduler">
          <select className={inp} value={scheduler} onChange={(e) => setScheduler(e.target.value)}>
            {SCHEDULERS.map((s) => <option key={s}>{s}</option>)}
          </select>
        </Field>
        <Field label="Shift" hint="AuraFlow; def 3"><input className={inp} type="number" step="0.5" placeholder="3" value={shift} onChange={(e) => setShift(e.target.value)} /></Field>
      </div>
      <label className="flex items-center gap-2 text-xs text-[var(--color-muted)]">
        <input type="checkbox" checked={apg} onChange={(e) => setApg(e.target.checked)} />
        Adaptive Projected Guidance (APG) <span className="text-[10px]">— cleaner high-CFG, less mud/oversaturation</span>
      </label>
      {apg && (
        <div className="grid grid-cols-2 gap-3">
          <Field label="APG norm" hint="main knob; lower = more (def 1.3)"><input className={inp} type="number" step="0.1" value={apgNorm} onChange={(e) => setApgNorm(e.target.value)} /></Field>
          <Field label="APG eta" hint="def 1.05 (1 = plain CFG)"><input className={inp} type="number" step="0.05" value={apgEta} onChange={(e) => setApgEta(e.target.value)} /></Field>
        </div>
      )}
    </>
  ) : hideDuration ? null : (
    <Field label="Duration (s)"><input className={inp} type="number" value={duration} onChange={(e) => setDuration(e.target.value)} /></Field>
  );

  const params = () => {
    const o: Record<string, unknown> = { variant, duration: parseFloat(duration) || 40, bpm: parseInt(bpm) || 120, keyscale };
    if (steps) o.steps = parseInt(steps);
    if (cfgScale) o.cfg = parseFloat(cfgScale);
    if (seed) o.seed = parseInt(seed);
    if (sampler !== "euler" || scheduler !== "simple") { o.sampler_name = sampler; o.scheduler = scheduler; }
    if (shift) o.shift = parseFloat(shift);
    if (apg) { o.apg = true; o.apg_norm = parseFloat(apgNorm) || 1.3; o.apg_eta = parseFloat(apgEta) || 1.05; }
    return o;
  };
  const applyPreset = (p: Preset) => { setBpm(String(p.bpm)); setKeyscale(p.key); };
  return { node, params, applyPreset, bpm: parseInt(bpm) || 120, keyscale };
}

function PresetBar({ genres, onApply }: { genres: Genre[]; onApply: (p: Preset) => void }) {
  const tops = genres.filter((g) => !g.parent);                       // top-level genres = the chips
  const bands = genres.filter((g) => g.parent);                       // band/artist styles = nested
  const labelOf = (id: string) => genres.find((g) => g.id === id)?.label || id;
  const apply = (g: Genre) => onApply({ name: g.label, tags: g.tags, bpm: g.bpm, key: g.key });
  // group bands under their parent genre for the dropdown
  const byParent: Record<string, Genre[]> = {};
  bands.forEach((b) => { (byParent[b.parent as string] ||= []).push(b); });
  return (
    <div>
      <span className="mb-1.5 block text-xs text-[var(--color-muted)]">Subgenre presets <span className="text-[10px]">(sets style tags · suggests {genres.length ? "BPM/key" : "…"})</span></span>
      <div className="flex flex-wrap items-center gap-1.5">
        {tops.map((g) => (
          <button key={g.id} title={`suggested: ${g.bpm} BPM · ${g.key}`}
            onClick={() => apply(g)}
            className="rounded-full border border-[var(--color-line)] bg-[var(--color-panel2)] px-3 py-1 text-xs text-[var(--color-muted)] transition hover:border-[var(--color-accent)] hover:text-[var(--color-ink)]">
            {g.label}
          </button>
        ))}
        {bands.length > 0 && (
          <select title="apply a specific band/artist style (grouped by genre)" value=""
            onChange={(e) => { const b = bands.find((x) => x.id === e.target.value); if (b) apply(b); }}
            className="rounded-full border border-[var(--color-line)] bg-[var(--color-panel2)] px-3 py-1 text-xs text-[var(--color-muted)] hover:border-[var(--color-accent)]">
            <option value="">Artist style…</option>
            {Object.keys(byParent).map((pid) => (
              <optgroup key={pid} label={labelOf(pid)}>
                {byParent[pid].map((b) => <option key={b.id} value={b.id}>{b.label}</option>)}
              </optgroup>
            ))}
          </select>
        )}
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
  const [theme, setTheme] = useState("");
  const [prov, setProv] = useState("local");
  const [lbusy, setLbusy] = useState(false);
  const claude = useClaudeAvail();
  async function writeLyrics() {
    setLbusy(true);
    try {
      const r: any = await api.llm({
        provider: lyrProvOf(prov), model: "", task: "lyrics",
        input: `Theme: ${theme || tags || "an epic metal song"}\nStyle: ${tags}`,
      });
      if (r.text) setLyrics(r.text);
    } catch (e) { alert("Lyric generation failed: " + (e as Error).message); }
    finally { setLbusy(false); }
  }
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
          <div className="mt-1.5 flex items-center gap-1.5">
            <input className={`${inp} flex-1 py-1.5 text-xs`} placeholder="theme for AI lyrics (blank = use style tags)" value={theme} onChange={(e) => setTheme(e.target.value)} />
            {claude && (
              <select className="rounded-lg border border-[var(--color-line)] bg-[var(--color-panel)] px-2 py-1.5 text-xs" value={prov} onChange={(e) => setProv(e.target.value)}>
                <option value="local">Gemma</option><option value="claude">Claude</option>
              </select>
            )}
            <button onClick={writeLyrics} disabled={lbusy}
              className="whitespace-nowrap rounded-lg border border-[var(--color-line)] bg-[var(--color-panel)] px-3 py-1.5 text-xs text-[var(--color-ink)] hover:border-[var(--color-accent)] disabled:opacity-50">
              {lbusy ? "Writing…" : "✨ Write lyrics"}
            </button>
          </div>
        </Field>
      )}
    </>
  );
}

export function GenerateForm({ cfg, busy, handoff, clearHandoff, ...ctx }: FormProps & { handoff?: { tags?: string; lyrics?: string } | null; clearHandoff?: () => void }) {
  const [tags, setTags] = useState("symphonic power metal, heavily distorted electric guitars, double-bass drums, orchestral strings, fast tempo, heroic");
  const [instrumental, setInstrumental] = useState(true);
  const [lyrics, setLyrics] = useState("");
  const [count, setCount] = useState(1);
  const [expert, setExpert] = useState(false);
  const [neg, setNeg] = useState("");
  const tuning = useTuning(cfg, expert);
  const applyPreset = (p: Preset) => setTags(p.tags);  // suggest style via tags; bpm/key stay the user's

  // accept a "send to Generate" handoff from the Song builder (tags + compiled lyrics)
  useEffect(() => {
    if (!handoff) return;
    if (handoff.tags) setTags(handoff.tags);
    if (handoff.lyrics !== undefined) { setLyrics(handoff.lyrics); setInstrumental(false); }
    clearHandoff?.();
  }, [handoff]);

  async function run() {
    if (!tags.trim()) return fail(ctx, "Add style tags first — an empty prompt produces noise.");
    const cards = Array.from({ length: count }, () => ({ id: rid(), title: "queued…", status: "pending" as const, pct: 0 }));
    ctx.setResults(cards);
    for (const c of cards) {
      try {
        const { job_id, seed } = await api.generate({ ...tuning.params(), tags, instrumental, lyrics, negative_tags: neg });
        ctx.patch(c.id, { title: `seed ${seed}`, status: "running", pct: 5 });
        pollJob(job_id, c.id, ctx);
      } catch (e) { ctx.patch(c.id, { status: "error", pct: 0, err: (e as Error).message }); }
    }
  }

  return (
    <div className="space-y-4">
      <ModeToggle expert={expert} setExpert={setExpert} />
      <PresetBar genres={cfg.genres} onApply={applyPreset} />
      <PromptFields {...{ tags, setTags, instrumental, setInstrumental, lyrics, setLyrics }} />
      {expert && (
        <Field label="Negative tags" hint="steer away from these — blank = off">
          <textarea className={inp} rows={2} value={neg} onChange={(e) => setNeg(e.target.value)}
            placeholder="muddy, lo-fi, harsh fizz, digital clipping, thin weak guitars, out of tune, low quality" />
        </Field>
      )}
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
  const applyPreset = (p: Preset) => setTags(p.tags);  // suggest style via tags; bpm/key stay the user's

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
      <PresetBar genres={cfg.genres} onApply={applyPreset} />
      <PromptFields {...{ tags, setTags, instrumental, setInstrumental, lyrics, setLyrics }} />
      {tuning.node}
      <PrimaryButton onClick={run} disabled={busy}>{busy ? "Restyling…" : "Restyle"}</PrimaryButton>
    </div>
  );
}

function EditForm({ cfg, busy, mode, ...ctx }: FormProps & { mode: "repaint" | "extend" }) {
  const tracks = useLibrary((it) => ["generate", "song", "restyle", "mix", "repaint", "extend", "voiceswap"].includes(it.mode));
  const [job, setJob] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [tags, setTags] = useState("");
  const [instrumental, setInstrumental] = useState(true);
  const [lyrics, setLyrics] = useState("");
  const [start, setStart] = useState("10");
  const [end, setEnd] = useState("20");
  const [left, setLeft] = useState("0");
  const [right, setRight] = useState("10");
  const [editCfg, setEditCfg] = useState("1");
  const [expert, setExpert] = useState(false);
  const tuning = useTuning(cfg, expert, true);   // duration derived from the source
  const applyPreset = (p: Preset) => setTags(p.tags);

  async function run() {
    if (!job && !file) return fail(ctx, "Choose a source track (library or upload).");
    if (!tags.trim()) return fail(ctx, "Add style tags describing the new content.");
    if (mode === "repaint" && !(parseFloat(end) > parseFloat(start))) return fail(ctx, "Repaint end must be after start.");
    if (mode === "extend" && !((parseFloat(left) || 0) > 0 || (parseFloat(right) || 0) > 0)) return fail(ctx, "Set seconds to add (before and/or after).");
    const id = rid();
    ctx.setResults([{ id, title: mode === "repaint" ? "repainting…" : "extending…", status: "pending", pct: 0 }]);
    try {
      const params: Record<string, unknown> = { ...tuning.params(), tags, instrumental, lyrics, edit_cfg: parseFloat(editCfg) || 3 };
      delete params.duration;                     // let the backend derive it from the source (+ extends)
      if (mode === "repaint") { params.repaint_start = parseFloat(start); params.repaint_end = parseFloat(end); }
      else { params.extend_left = parseFloat(left) || 0; params.extend_right = parseFloat(right) || 0; }
      const fd = new FormData();
      if (file) fd.append("file", file); else fd.append("job_id", job);
      fd.append("params", JSON.stringify(params));
      const { job_id } = await api.repaint(fd);   // extend removed (not a native ACE task)
      ctx.patch(id, { status: "running", pct: 5 });
      pollJob(job_id, id, ctx);
    } catch (e) { ctx.patch(id, { status: "error", pct: 0, err: (e as Error).message }); }
  }

  return (
    <div className="space-y-4">
      <ModeToggle expert={expert} setExpert={setExpert} />
      <p className="text-xs text-[var(--color-muted)]">
        {mode === "repaint"
          ? "Regenerate a time range of a track (new tags/lyrics) while keeping the rest — latent-level via ACEStep15NativeEditGuider."
          : "Lengthen a track by generating new content before and/or after it — the existing audio is preserved."}
        {" "}<em>Runs on the turbo model (what the edit guider needs); the Model picker below is ignored here.</em>
      </p>
      <Field label="Source track">
        <select className={inp} value={job} onChange={(e) => { setJob(e.target.value); if (e.target.value) setFile(null); }}>
          <option value="">— pick a library track —</option>
          {tracks.map((t) => <option key={t.id} value={t.id}>{(t.params.tags || t.params.source || t.id).slice(0, 40)}</option>)}
        </select>
      </Field>
      <Field label="…or upload" hint="overrides the picker"><input className={inp} type="file" accept="audio/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} /></Field>
      {mode === "repaint" ? (
        <div className="grid grid-cols-2 gap-3">
          <Field label="Repaint start (s)"><input className={inp} type="number" step="0.5" value={start} onChange={(e) => setStart(e.target.value)} /></Field>
          <Field label="Repaint end (s)"><input className={inp} type="number" step="0.5" value={end} onChange={(e) => setEnd(e.target.value)} /></Field>
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-3">
          <Field label="Add before (s)"><input className={inp} type="number" step="0.5" value={left} onChange={(e) => setLeft(e.target.value)} /></Field>
          <Field label="Add after (s)"><input className={inp} type="number" step="0.5" value={right} onChange={(e) => setRight(e.target.value)} /></Field>
        </div>
      )}
      <PresetBar genres={cfg.genres} onApply={applyPreset} />
      <PromptFields {...{ tags, setTags, instrumental, setInstrumental, lyrics, setLyrics }} />
      {expert && <Field label="Edit guidance (cfg)" hint="def 1 (canonical); >1 tends to garble"><input className={inp} type="number" step="0.5" value={editCfg} onChange={(e) => setEditCfg(e.target.value)} /></Field>}
      {tuning.node}
      <PrimaryButton onClick={run} disabled={busy}>{busy ? "Working…" : mode === "repaint" ? "Repaint region" : "Extend track"}</PrimaryButton>
    </div>
  );
}

export function RepaintForm(p: FormProps) { return <EditForm {...p} mode="repaint" />; }
// Extend (append) removed — not a native ACE task; see RESEARCH.md §10j. EditForm
// retains the "extend" branch in git history; RepaintForm is the only caller now.

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

export function ToneForm({ busy, ...ctx }: FormProps) {
  const tracks = useLibrary((it) => ["generate", "restyle", "song", "mix", "tone"].includes(it.mode));
  const [job, setJob] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [preset, setPreset] = useState("tighten_highgain");
  const [presets, setPresets] = useState<{ id: string; label: string; desc: string }[]>([]);
  const [helixOn, setHelixOn] = useState(false);
  const [capName, setCapName] = useState("");
  const [capStatus, setCapStatus] = useState("");
  const loadPresets = (selectFirst = false) => api.tonePresets().then((d) => {
    setPresets(d.presets); setHelixOn(d.helix_available);
    if (selectFirst && d.presets[0]) setPreset(d.presets[0].id);
  }).catch(() => {});
  useEffect(() => { loadPresets(true); }, []);
  const cur = presets.find((p) => p.id === preset);

  async function capture() {
    if (!capName.trim()) { setCapStatus("name the tone first"); return; }
    setCapStatus("Helix window opening — dial your tone, then CLOSE the window to save…");
    try {
      const d = await api.helixCapture(capName.trim());
      setCapStatus(`saved "${d.saved}" — now in the preset list`);
      setCapName("");
      await loadPresets();
      setPreset(`helix:${d.saved}`);
    } catch (e) { setCapStatus("capture failed: " + (e as Error).message); }
  }

  async function run() {
    if (!file && !job) return fail(ctx, "Choose a track to re-tone.");
    const id = rid();
    ctx.setResults([{ id, title: "splitting + re-toning the guitar… (Mac)", status: "running", pct: 35 }]);
    try {
      const fd = new FormData();
      fd.append("preset", preset);
      if (file) fd.append("file", file); else fd.append("job_id", job);
      const d = await api.tone(fd);
      ctx.setResults([
        { id: rid(), title: `re-toned (${preset})`, status: "done", pct: 100, url: d.audio_url },
        { id: rid(), title: "guitar stem (processed)", status: "done", pct: 100, url: d.guitar_url },
      ]);
    } catch (e) { ctx.patch(id, { status: "error", pct: 0, err: (e as Error).message }); }
  }

  return (
    <div className="space-y-4">
      <p className="text-xs text-[var(--color-muted)]">
        Reshape the distorted guitar: split off the guitar stem (Mac/Demucs 6-stem), run it through a tone chain, and recombine.
        This <em>reshapes</em> the existing tone (EQ / cab / saturation) — it doesn't re-amp from a clean DI.
      </p>
      <Field label="From a library track">
        <select className={inp} value={job} onChange={(e) => setJob(e.target.value)}>
          <option value="">— choose —</option>
          {tracks.map((t) => <option key={t.id} value={t.id}>{t.mode}: {(t.params.tags || t.params.preset || "").slice(0, 36)}</option>)}
        </select>
      </Field>
      <Field label="…or upload"><input className={inp} type="file" accept="audio/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} /></Field>
      <Field label="Tone preset" hint={helixOn ? "Helix Native detected" : "set helix_vst3_path to add Helix"}>
        <select className={inp} value={preset} onChange={(e) => setPreset(e.target.value)}>
          {presets.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
        </select>
      </Field>
      {cur && <p className="text-xs text-[var(--color-muted)]">{cur.desc}</p>}
      <PrimaryButton onClick={run} disabled={busy}>{busy ? "Working…" : "Re-tone guitar"}</PrimaryButton>
      {helixOn && (
        <div className="rounded-lg border border-[var(--color-line)] bg-[var(--color-panel2)] p-3 space-y-2">
          <div className="text-xs font-medium text-[var(--color-muted)]">Capture a Helix tone</div>
          <p className="text-[11px] text-[var(--color-muted)]">Dial any tone in Helix once and save it — reusable here + in the Guitar tab with no UI fiddling.</p>
          <div className="flex gap-1.5">
            <input className={inp} placeholder="tone name (e.g. Modern Rhythm)" value={capName} onChange={(e) => setCapName(e.target.value)} />
            <button onClick={capture} className="rounded-lg border border-[var(--color-line)] bg-[var(--color-panel)] px-3 text-sm text-[var(--color-ink)] hover:border-[var(--color-accent)]">Open + capture</button>
          </div>
          {capStatus && <p className="text-[11px] text-[var(--color-accent2)]">{capStatus}</p>}
        </div>
      )}
    </div>
  );
}

export function GuitarForm({ cfg, busy, song, ...ctx }: FormProps & { song?: SongDraft | null }) {
  const backings = useLibrary((it) => it.mode === "backing");
  const [source, setSource] = useState<"riff" | "song" | "midi">("riff");
  const [file, setFile] = useState<File | null>(null);
  const [key, setKey] = useState("E minor");
  const [bpm, setBpm] = useState("160");
  const [bars, setBars] = useState("8");
  const [style, setStyle] = useState("gallop");
  const [brain, setBrain] = useState("algorithmic");
  const [genre, setGenre] = useState(cfg.genres[0]?.id || "thrash");
  const [part, setPart] = useState("riff");
  const [preset, setPreset] = useState("helix");
  const [presets, setPresets] = useState<{ id: string; label: string }[]>([]);
  const [backing, setBacking] = useState("");
  const [align, setAlign] = useState(false);
  const [alignAvail, setAlignAvail] = useState(false);
  const [sfOn, setSfOn] = useState(false);
  const [diEngine, setDiEngine] = useState("ks");
  const [kontaktReady, setKontaktReady] = useState(false);
  const [kontaktAvail, setKontaktAvail] = useState(false);
  const [kStatus, setKStatus] = useState("");
  const genres = cfg.genres;                                 // unified registry from /api/config
  const sug = genres.find((g) => g.id === genre);            // selected genre → suggested bpm/scale
  const loadEngines = () => api.tonePresets().then((d) => {
    setPresets(d.presets); setSfOn(!!d.guitar_soundfont);
    setKontaktReady(!!d.kontakt_ready); setKontaktAvail(!!d.kontakt_available);
    setAlignAvail(!!d.align_available);
  }).catch(() => {});
  useEffect(() => { loadEngines(); }, []);

  async function setupKontakt() {
    setKStatus("Kontakt opening (~25s) — load Shreddage 3 Stratus FREE + pick a patch, then CLOSE the window…");
    try { await api.kontaktCapture(); await loadEngines(); setDiEngine("kontakt"); setKStatus("Shreddage/Kontakt ready — selected as the DI engine."); }
    catch (e) { setKStatus("setup failed: " + (e as Error).message); }
  }

  async function run() {
    if (source === "midi" && !file) return fail(ctx, "Upload a MIDI file.");
    if (source === "song" && !(song && song.blocks.length)) return fail(ctx, "Build a song arrangement in the Song tab first.");
    const id = rid();
    ctx.setResults([{ id, title: "rendering DI → amp…", status: "running", pct: 40 }]);
    try {
      const fd = new FormData();
      if (source === "riff") { fd.append("riff", "true"); fd.append("key", key); fd.append("bpm", bpm); fd.append("bars", bars); fd.append("style", style); }
      else if (source === "song") {
        fd.append("sections", JSON.stringify(song!.blocks.map((b) => ({ type: b.type, seconds: b.seconds }))));
        fd.append("key", song!.key); fd.append("bpm", String(song!.bpm));
      } else fd.append("midi", file as File);
      if (source !== "midi") { fd.append("riff_brain", brain); if (brain !== "algorithmic") fd.append("genre", genre); }
      if (source === "riff") fd.append("part", part);
      fd.append("preset", preset);
      fd.append("di_engine", diEngine);
      if (backing) fd.append("backing_job_id", backing);
      if (backing && align && source !== "midi") fd.append("align_backing", "true");
      const d = await api.guitarRender(fd);
      const cards = [
        { id: rid(), title: "clean DI", status: "done" as const, pct: 100, url: d.di_url },
        { id: rid(), title: `amped (${preset})`, status: "done" as const, pct: 100, url: d.amped_url },
      ];
      if (d.mix_url) cards.push({ id: rid(), title: "guitar + backing", status: "done" as const, pct: 100, url: d.mix_url });
      ctx.setResults(cards);
    } catch (e) { ctx.patch(id, { status: "error", pct: 0, err: (e as Error).message }); }
  }

  const tabBtn = (v: "riff" | "song" | "midi", label: string, disabled = false) => (
    <button disabled={disabled} onClick={() => setSource(v)}
      className={`flex-1 rounded-lg border py-1.5 ${source === v ? "border-[var(--color-accent)] bg-[#2a1c19]" : "border-[var(--color-line)] bg-[var(--color-panel2)] text-[var(--color-muted)]"} ${disabled ? "opacity-40" : ""}`}>{label}</button>
  );

  return (
    <div className="space-y-4">
      <p className="text-xs text-[var(--color-muted)]">
        Symbolic-guitar route: a generated/MIDI guitar part → <em>clean DI</em> (plucked-string synth) → your Helix/tone amp (valid on a clean DI!) → optionally mixed onto a guitar-less backing. Render quality is a prototype; a sampled/Shreddage DI can slot in later.
      </p>
      <div className="flex gap-1.5 text-sm">
        {tabBtn("riff", "Test riff")}
        {tabBtn("song", "Song arrangement", !(song && song.blocks.length))}
        {tabBtn("midi", "Upload MIDI")}
      </div>
      {source === "song" ? (
        <p className="text-xs text-[var(--color-muted)]">
          Per-section riffs from your Song arrangement ({song?.blocks.length ?? 0} sections, {song?.key}, {song?.bpm} BPM): verse chugs, chorus power-chords, breakdown drops an octave, intro/outro sparse, <em>solo sections get a lead line over a quieter bed</em>. Pick a matching backing below.
        </p>
      ) : source === "riff" ? (
        <>
          <div className="grid grid-cols-3 gap-3">
            <Field label="Key"><select className={inp} value={key} onChange={(e) => setKey(e.target.value)}>{cfg.keys.map((k) => <option key={k}>{k}</option>)}</select></Field>
            <Field label="BPM"><input className={inp} type="number" value={bpm} onChange={(e) => setBpm(e.target.value)} /></Field>
            <Field label="Bars" hint="ignored if mixing onto a backing"><input className={inp} type="number" value={bars} onChange={(e) => setBars(e.target.value)} /></Field>
          </div>
          <Field label="Riff style">
            <select className={inp} value={style} onChange={(e) => setStyle(e.target.value)}>
              <option value="gallop">Gallop (1/8+1/16+1/16)</option>
              <option value="chug">Chug (palm-mute)</option>
              <option value="powerchords">Power chords (held, i–VI–VII)</option>
              <option value="pedal">Pedal-tone</option>
            </select>
          </Field>
        </>
      ) : (
        <Field label="MIDI file"><input className={inp} type="file" accept=".mid,.midi" onChange={(e) => setFile(e.target.files?.[0] ?? null)} /></Field>
      )}
      {source === "riff" && (
        <Field label="Part" hint="rhythm riff or a lead solo line">
          <div className="flex gap-1.5">
            {[["riff", "Riff"], ["solo", "Solo (lead)"]].map(([v, l]) => (
              <button key={v} onClick={() => setPart(v)}
                className={`flex-1 rounded-lg border py-1.5 text-sm ${part === v ? "border-[var(--color-accent)] bg-[#2a1c19]" : "border-[var(--color-line)] bg-[var(--color-panel2)] text-[var(--color-muted)]"}`}>{l}</button>
            ))}
          </div>
        </Field>
      )}
      {source !== "midi" && (
        <Field label="Riff brain" hint="how the notes are chosen">
          <select className={inp} value={brain} onChange={(e) => setBrain(e.target.value)}>
            <option value="algorithmic">Algorithmic (instant, music-theory)</option>
            <option value="local">AI · local Gemma (musical, ~slower)</option>
            <option value="claude">AI · Claude (if API key set)</option>
          </select>
        </Field>
      )}
      {source !== "midi" && brain !== "algorithmic" && (
        <Field label="Genre feel" hint={sug ? `${sug.scale} · suggests ${sug.bpm} BPM` : "shapes the AI riff + modal scale"}>
          <select className={inp} value={genre} onChange={(e) => setGenre(e.target.value)}>
            {genres.filter((g) => !g.parent).map((g) => {
              const kids = genres.filter((k) => k.parent === g.id);
              return kids.length ? (
                <optgroup key={g.id} label={g.label}>
                  <option value={g.id}>{g.label} (genre)</option>
                  {kids.map((b) => <option key={b.id} value={b.id}>{b.label}</option>)}
                </optgroup>
              ) : <option key={g.id} value={g.id}>{g.label}</option>;
            })}
          </select>
          {sug && String(sug.bpm) !== bpm && (
            <button onClick={() => setBpm(String(sug.bpm))}
              className="mt-1 text-[11px] text-[var(--color-accent2)] hover:underline">use suggested {sug.bpm} BPM</button>
          )}
        </Field>
      )}
      <Field label="DI render engine" hint={kontaktReady ? "Shreddage = best" : sfOn ? "SoundFont = sampled" : "KS = synth"}>
        <select className={inp} value={diEngine} onChange={(e) => setDiEngine(e.target.value)}>
          <option value="ks">Karplus-Strong (synth, no deps)</option>
          {sfOn && <option value="soundfont">SoundFont (sampled, higher fidelity)</option>}
          {kontaktReady && <option value="kontakt">Shreddage / Kontakt (best, sampled)</option>}
        </select>
      </Field>
      {kontaktAvail && !kontaktReady && (
        <div className="rounded-lg border border-[var(--color-line)] bg-[var(--color-panel2)] p-3 space-y-2">
          <div className="text-xs font-medium text-[var(--color-muted)]">Set up Shreddage (one-time)</div>
          <p className="text-[11px] text-[var(--color-muted)]">Kontakt is installed. Click below — it opens Kontakt (~25s); load <em>Shreddage 3 Stratus FREE</em> + pick a patch, then close the window. Saved for reuse.</p>
          <button onClick={setupKontakt} className="rounded-lg border border-[var(--color-line)] bg-[var(--color-panel)] px-3 py-1 text-sm text-[var(--color-ink)] hover:border-[var(--color-accent)]">Open Kontakt + capture Shreddage</button>
        </div>
      )}
      {kStatus && <p className="text-[11px] text-[var(--color-accent2)]">{kStatus}</p>}
      <Field label="Amp / tone">
        <select className={inp} value={preset} onChange={(e) => setPreset(e.target.value)}>
          {presets.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
        </select>
      </Field>
      <Field label="Mix onto backing (optional)">
        <select className={inp} value={backing} onChange={(e) => setBacking(e.target.value)}>
          <option value="">— DI + amp only —</option>
          {backings.map((b) => <option key={b.id} value={b.id}>{(b.params.source || b.id).slice(0, 36)}</option>)}
        </select>
      </Field>
      {backing && alignAvail && source !== "midi" && (
        <label className="flex items-start gap-2 text-xs text-[var(--color-muted)]">
          <input type="checkbox" className="mt-0.5" checked={align} onChange={(e) => setAlign(e.target.checked)} />
          <span>
            <span className="text-[var(--color-ink)]">Align to the backing's real sections</span> — detect where the
            backing actually changes (librosa) and {source === "song"
              ? "re-time your arrangement's sections to those boundaries (labels kept)."
              : "auto-build a per-section guitar arrangement from them (roles inferred by energy)."}
          </span>
        </label>
      )}
      <PrimaryButton onClick={run} disabled={busy}>{busy ? "Rendering…" : "Render → amp"}</PrimaryButton>
    </div>
  );
}

export function BackingForm({ busy, ...ctx }: FormProps) {
  const tracks = useLibrary((it) => ["generate", "song", "restyle", "mix"].includes(it.mode));
  const [job, setJob] = useState("");
  const [file, setFile] = useState<File | null>(null);

  async function run() {
    if (!file && !job) return fail(ctx, "Choose a track to strip the guitar from.");
    const id = rid();
    ctx.setResults([{ id, title: "splitting + removing guitar… (Mac)", status: "running", pct: 40 }]);
    try {
      const fd = new FormData();
      if (file) fd.append("file", file); else fd.append("job_id", job);
      const d = await api.stripGuitar(fd);
      ctx.setResults([{ id: rid(), title: "guitar-less backing", status: "done", pct: 100, url: d.audio_url }]);
    } catch (e) { ctx.patch(id, { status: "error", pct: 0, err: (e as Error).message }); }
  }

  return (
    <div className="space-y-4">
      <p className="text-xs text-[var(--color-muted)]">
        Remove the guitar from a track (6-stem split on the Mac, recombine without the guitar stem) → a <em>guitar-less backing</em>.
        Foundation for the symbolic-guitar route: drop the model's distorted guitar so a clean-DI, properly-amped guitar can sit on top.
        Demucs isn't perfect — some guitar bleed may remain.
      </p>
      <Field label="From a library track">
        <select className={inp} value={job} onChange={(e) => setJob(e.target.value)}>
          <option value="">— choose —</option>
          {tracks.map((t) => <option key={t.id} value={t.id}>{t.mode}: {(t.params.tags || "").slice(0, 36)}</option>)}
        </select>
      </Field>
      <Field label="…or upload"><input className={inp} type="file" accept="audio/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} /></Field>
      <PrimaryButton onClick={run} disabled={busy}>{busy ? "Working…" : "Strip guitar → backing"}</PrimaryButton>
    </div>
  );
}

export function MasterForm({ busy, ...ctx }: FormProps) {
  const targets = useLibrary((it) => ["generate", "song", "mix", "tone", "restyle", "voiceswap"].includes(it.mode));
  const refs = useLibrary(() => true);  // any track can be a reference, esp. imported "source" songs
  const [job, setJob] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [refJob, setRefJob] = useState("");
  const [refFile, setRefFile] = useState<File | null>(null);

  async function run() {
    if (!file && !job) return fail(ctx, "Choose a track to master.");
    if (!refFile && !refJob) return fail(ctx, "Choose a reference master (a pro track whose sound you want to match).");
    const id = rid();
    ctx.setResults([{ id, title: "matching to reference… (Mac)", status: "running", pct: 45 }]);
    try {
      const fd = new FormData();
      if (file) fd.append("file", file); else fd.append("job_id", job);
      if (refFile) fd.append("ref_file", refFile); else fd.append("ref_job_id", refJob);
      const d = await api.master(fd);
      ctx.setResults([{ id: rid(), title: "mastered", status: "done", pct: 100, url: d.audio_url }]);
    } catch (e) { ctx.patch(id, { status: "error", pct: 0, err: (e as Error).message }); }
  }

  const opts = (list: any[]) => list.map((t) => <option key={t.id} value={t.id}>{t.mode}: {(t.params.tags || t.params.source || t.params.preset || "").slice(0, 32)}</option>);

  return (
    <div className="space-y-4">
      <p className="text-xs text-[var(--color-muted)]">
        Reference mastering (Matchering): match a track's loudness, EQ balance, peak & stereo width to a <em>reference</em> master you own. Mac-side. Personal use — the reference is only analysed.
      </p>
      <div className="text-xs font-medium text-[var(--color-muted)]">Target (the track to master)</div>
      <Field label="From a library track">
        <select className={inp} value={job} onChange={(e) => setJob(e.target.value)}>
          <option value="">— choose —</option>{opts(targets)}
        </select>
      </Field>
      <Field label="…or upload"><input className={inp} type="file" accept="audio/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} /></Field>
      <div className="text-xs font-medium text-[var(--color-muted)]">Reference (the sound to match — e.g. an imported pro track)</div>
      <Field label="From a library track">
        <select className={inp} value={refJob} onChange={(e) => setRefJob(e.target.value)}>
          <option value="">— choose —</option>{opts(refs)}
        </select>
      </Field>
      <Field label="…or upload"><input className={inp} type="file" accept="audio/*" onChange={(e) => setRefFile(e.target.files?.[0] ?? null)} /></Field>
      <PrimaryButton onClick={run} disabled={busy}>{busy ? "Mastering…" : "Master to reference"}</PrimaryButton>
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

export function SongForm({ cfg, busy, onSong, onSendToGenerate, ...ctx }: FormProps & { onSong?: (s: SongDraft) => void; onSendToGenerate?: (h: { tags: string; lyrics: string }) => void }) {
  const [blocks, setBlocks] = useState<Block[]>(() =>
    ["Intro", "Verse", "Chorus", "Verse", "Chorus", "Solo", "Chorus", "Outro"].map(newBlock));
  const [tags, setTags] = useState("");
  const [instrumental, setInstrumental] = useState(false);
  const [drive, setDrive] = useState<"compile" | "stitch">("compile");
  const [crossfade, setCrossfade] = useState(1);
  const [expert, setExpert] = useState(false);
  const [tpl, setTpl] = useState("");
  const [dirty, setDirty] = useState(false); // arrangement edited since last template/default
  const [lyrTheme, setLyrTheme] = useState("");
  const [lyrProv, setLyrProv] = useState("local");
  const [lyrBusy, setLyrBusy] = useState(false);
  const [sungEnds, setSungEnds] = useState(false);
  const claude = useClaudeAvail();
  const tuning = useTuning(cfg, expert, true); // duration is computed from blocks
  const applyPreset = (p: Preset) => setTags(p.tags);  // suggest style via tags; bpm/key stay the user's
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

  async function writeSectionLyrics() {
    setLyrBusy(true);
    try {
      const r: any = await api.songLyrics({
        blocks: blocks.map((b) => ({ type: b.type })),
        theme: lyrTheme || tags, style: tags, provider: lyrProvOf(lyrProv), model: "",
        extra_sung: sungEnds ? ["intro", "outro"] : [],
      });
      const byIdx: Record<number, string> = {};
      (r.sections || []).forEach((s: any) => { byIdx[s.index] = s.lyrics; });
      if (!Object.keys(byIdx).length) { alert("No sung sections to write (add Verse/Chorus blocks)."); return; }
      setBlocks((bs) => bs.map((b, i) => (byIdx[i] !== undefined ? { ...b, lyrics: byIdx[i] } : b)));
      setDirty(true);
    } catch (e) { alert("Section lyrics failed: " + (e as Error).message); }
    finally { setLyrBusy(false); }
  }

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
      <PresetBar genres={cfg.genres} onApply={applyPreset} />
      <Field label="Style tags" hint="shared across the whole song">
        <textarea className={inp} rows={3} value={tags} onChange={(e) => setTags(e.target.value)} />
      </Field>
      <label className="flex items-center gap-2 text-sm text-[var(--color-muted)]">
        <input type="checkbox" checked={instrumental} onChange={(e) => setInstrumental(e.target.checked)} /> Instrumental (section tags still guide the arrangement)
      </label>

      {!instrumental ? (
        <div className="rounded-lg border border-[var(--color-line)] bg-[var(--color-panel2)] p-3 space-y-2">
          <div className="text-xs font-medium text-[var(--color-ink)]">✨ AI lyrics for this arrangement</div>
          <div className="flex items-center gap-1.5">
            <input className={`${inp} flex-1 py-1.5 text-xs`} placeholder="theme (blank = use style tags)" value={lyrTheme} onChange={(e) => setLyrTheme(e.target.value)} />
            {claude && (
              <select className="rounded-lg border border-[var(--color-line)] bg-[var(--color-panel)] px-2 py-1.5 text-xs" value={lyrProv} onChange={(e) => setLyrProv(e.target.value)}>
                <option value="local">Gemma</option><option value="claude">Claude</option>
              </select>
            )}
            <button onClick={writeSectionLyrics} disabled={lyrBusy || !blocks.length}
              className="whitespace-nowrap rounded-lg border border-[var(--color-line)] bg-[var(--color-panel)] px-3 py-1.5 text-xs text-[var(--color-ink)] hover:border-[var(--color-accent)] disabled:opacity-50">
              {lyrBusy ? "Writing…" : "Write section lyrics"}
            </button>
          </div>
          <label className="flex items-center gap-2 text-[11px] text-[var(--color-muted)]">
            <input type="checkbox" checked={sungEnds} onChange={(e) => setSungEnds(e.target.checked)} /> Also write lyrics for intro &amp; outro (otherwise they stay wordless)
          </label>
          <p className="text-[11px] text-[var(--color-muted)]">Distinct verses that advance the story, one repeated chorus hook, plus pre-chorus/bridge; solo/breakdown stay wordless. Fills each block below.</p>
          {onSendToGenerate && (
            <button onClick={() => onSendToGenerate({ tags, lyrics: compileLyrics(blocks) })}
              className="text-[11px] text-[var(--color-accent2)] hover:underline">Send these lyrics + style → Generate tab →</button>
          )}
        </div>
      ) : (
        <p className="text-[11px] text-[var(--color-muted)]">Uncheck “Instrumental” to auto-write sung lyrics per section with AI.</p>
      )}

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
