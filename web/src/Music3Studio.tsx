import { useEffect, useMemo, useState } from "react";
import { api, type Config, type Project, type SongDraft } from "./api";
import { Field, inp, PrimaryButton, GhostButton, SectionTitle, pollJob, rid, type RunCtx } from "./ui";
import { useDrafts } from "./drafts";

// ---------------------------------------------------------------------------------------------
// MiniMax Music 3 studio.
//
// Separate from the ACE Song tab on purpose. The two engines want opposite prompts: ACE wants
// ~10-12 dense tag phrases and goes disjointed beyond that, Music 3 wants a 4000+ character
// caption in ITS OWN field schema. Measured on "Garden of Ashes" with seed and lyrics held fixed:
// freeform prose under the right headers produced cinematic orchestral pop with no metal in it;
// the same content in the canonical fields produced guitar-forward symphonic metal.
//
// The caption is edited as SEPARATE FIELDS rather than one textarea. That is not decoration - the
// field a sentence lands in changes the output. "Heavily distorted guitars" under Primary makes
// them the centre of the mix; the same words buried in a paragraph does not.
//
// Section tags stay BARE. Loading them with style cues is the one thing this UI refuses to let you
// do by accident, because it silently ruins takes: the model sings whatever follows the section
// name ("enthemic soaring in the garden of ashes", "half-time emotional when the night was deep").
// Per-section direction belongs in the progression fields, addressed by section name.
// ---------------------------------------------------------------------------------------------

type Props = { cfg: Config; busy: boolean; song: SongDraft | null } & RunCtx;
type SchemaField = { key: string; group: string; help: string };
type Schema = {
  fields: SchemaField[]; groups: string[]; tags: string[];
  defaults: Record<string, number | boolean | string>; max_seconds: number;
};

export function Music3StudioForm({ cfg: _cfg, busy, song, ...ctx }: Props) {
  const d = useDrafts("music3");
  const [schema, setSchema] = useState<Schema | null>(null);
  const [avail, setAvail] = useState<{ available: boolean; reason?: string } | null>(null);

  const [fields, setFields] = d.use<Record<string, string>>("fields", {});
  const [lyrics, setLyrics] = d.use("lyrics", "");
  const [title, setTitle] = d.use("title", "");
  const [seconds, setSeconds] = d.use("seconds", "210");
  const [seed, setSeed] = d.use("seed", "");
  const [count, setCount] = d.use("count", "1");
  const [cfgScale, setCfgScale] = d.use("cfgScale", "1.7");
  const [topK, setTopK] = d.use("topK", "50");
  const [steps, setSteps] = d.use("steps", "30");
  const [tiled, setTiled] = d.use("tiled", true);
  const [showAdv, setShowAdv] = useState(false);
  const [openHelp, setOpenHelp] = useState<string | null>(null);

  const [projects, setProjects] = useState<Project[]>([]);
  const [importFrom, setImportFrom] = useState("");
  const [note, setNote] = useState("");
  const [preview, setPreview] = useState<{ chars: number; approx_tokens: number; token_limit: number; over_limit: boolean } | null>(null);
  const [showCaption, setShowCaption] = useState(false);
  const [captionText, setCaptionText] = useState("");

  useEffect(() => {
    api.music3Schema().then(setSchema).catch(() => setSchema(null));
    api.music3Available().then(setAvail).catch(() => setAvail({ available: false, reason: "backend unreachable" }));
    api.projects().then(setProjects).catch(() => setProjects([]));
  }, []);

  // Size the caption as it is typed. The box hard-fails above 5000 tokens, and finding that out at
  // submit time costs a round trip and reads as an unexplained error.
  useEffect(() => {
    const t = window.setTimeout(() => {
      api.music3Preview({ fields, lyrics })
        .then((r) => { setPreview(r); setCaptionText(r.caption); })
        .catch(() => setPreview(null));
    }, 400);
    return () => window.clearTimeout(t);
  }, [fields, lyrics]);

  const set = (k: string, v: string) => setFields({ ...fields, [k]: v });
  const filled = useMemo(() => Object.values(fields).filter((v) => (v || "").trim()).length, [fields]);

  async function importSong(s: SongDraft | null, label: string) {
    if (!s || !(s.blocks || []).length) { setNote(`${label} has no song arrangement`); return; }
    try {
      const r = await api.music3FromSong(s);
      setFields({ ...fields, ...r.fields });
      setLyrics(r.lyrics);
      setNote(`imported ${label}: ${(s.blocks || []).length} sections. Per-section style cues went into the progression fields, not the tags.`);
    } catch (e) {
      setNote(`import failed: ${e}`);
    }
  }

  async function importProject(id: string) {
    setImportFrom(id);
    if (!id) return;
    try {
      const p = await api.projectGet(id);
      // A project stores the arrangement TWICE and neither copy is complete: `data.song` is the
      // canonical draft and the only one carrying key and bpm, while `data.drafts.song` is the
      // Song page's live state and the only one carrying the title. Reading either alone silently
      // drops half the caption - taking drafts.song on its own produced a Basic Attributes line
      // with the genre but no tempo and no key.
      const canon = (p?.data?.song || {}) as Partial<SongDraft>;
      const page = (p?.data?.drafts?.song || {}) as Partial<SongDraft> & { instrumental?: boolean };
      const merged = {
        ...canon, ...page,
        blocks: (page.blocks?.length ? page.blocks : canon.blocks) || [],
        key: canon.key, bpm: canon.bpm,
        tags: page.tags || canon.tags || "",
      } as SongDraft;
      await importSong(merged, p?.name || "project");
    } catch (e) {
      setNote(`could not read that project: ${e}`);
    }
  }

  function insertTag(tag: string) {
    const t = lyrics.trim();
    setLyrics((t ? t + "\n\n" : "") + `[${tag}]\n`);
  }

  async function run() {
    if (!(captionText || "").trim()) { setNote("write a caption first - Music 3 has nothing else to go on"); return; }
    const n = Math.max(1, Math.min(8, parseInt(count) || 1));
    const cards = Array.from({ length: n }, (_v, i) => ({
      id: rid(), status: "pending" as const, pct: 0,
      title: `${title || "Music 3"}${n > 1 ? ` (take ${i + 1}/${n})` : ""}`,
    }));
    // One setResults for all cards: it REPLACES the array, so setting it per-iteration inside the
    // loop wipes every card but the last, which is exactly how the seed-hunt progress bar broke.
    ctx.setResults(cards);
    for (let i = 0; i < n; i++) {
      try {
        const r = await api.music3Generate({
          fields, lyrics, title,
          seconds: parseFloat(seconds) || 210,
          // blank seed = a fresh random one per take, which is what a seed hunt wants
          seed: seed.trim() ? parseInt(seed) + i : undefined,
          cfg_scale: parseFloat(cfgScale) || 1.7,
          top_k: parseInt(topK) || 50,
          steps: parseInt(steps) || 30,
          tiled_decode: tiled,
        });
        ctx.patch(cards[i].id, { status: "running", pct: 5 });
        pollJob(r.job_id, cards[i].id, ctx);
      } catch (e) {
        ctx.patch(cards[i].id, { status: "error", pct: 0, err: String(e) });
      }
    }
  }

  if (!schema) return <div className="text-sm text-[var(--color-muted)]">loading Music 3 schema…</div>;

  const tagsInLyrics = (lyrics.match(/\[[^\]]+\]/g) || []);
  // A tag is "loaded" when it carries direction as well as a name. Match a comma, a colon, or a
  // SPACED hyphen - a bare hyphen would flag [Pre-Chorus], which is a documented tag name.
  const loadedTags = tagsInLyrics.filter((t) => /[,:]|\s-\s/.test(t.slice(1, -1)));
  // Parentheses in the lyrics field are LYRICS, not stage directions. Verified: the shipped
  // template's "(rain on the window)" came back in the transcript of our own render, sung. So
  // "(heavy guitar riff enters)" would be sung too - a natural thing to write, and quietly fatal.
  const stageDirections = (lyrics.match(/\([^)]*\)/g) || []).filter((p) =>
    /\b(guitar|drum|orchestr|solo|riff|choir|enters?|fade|swell|instrument|piano|strings|percussion|bass|tempo|build|chant)\b/i.test(p));

  return (
    <div className="space-y-4">
      {avail && !avail.available && (
        <div className="rounded border border-amber-600/50 bg-amber-950/30 p-2 text-xs text-amber-300">
          Music 3 is not ready on the box: {avail.reason}
        </div>
      )}

      {/* ---- start from an existing song ---- */}
      <div className="rounded border border-[var(--color-line)] p-3">
        <SectionTitle>Start from a song</SectionTitle>
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <GhostButton onClick={() => importSong(song, "the loaded song")} disabled={!song?.blocks?.length}
            title={song?.blocks?.length ? "fill the caption and lyrics from the song currently loaded" : "no song loaded"}>
            Import loaded song
          </GhostButton>
          <select className={inp} value={importFrom} onChange={(e) => importProject(e.target.value)}
            title="load any saved project's arrangement into this form">
            <option value="">or pick a project…</option>
            {projects.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
          </select>
          <span className="text-[10px] text-[var(--color-muted)]">
            sections and per-block style cues come across; seconds do not, because Music 3 has no
            per-section timing at all.
          </span>
        </div>
        {note && <div className="mt-2 text-[11px] text-[var(--color-accent2)]">{note}</div>}
      </div>

      {/* ---- caption fields ---- */}
      <div className="rounded border border-[var(--color-line)] p-3">
        <div className="flex items-center justify-between">
          <SectionTitle>Caption</SectionTitle>
          <div className="flex items-center gap-2 text-[10px] text-[var(--color-muted)]">
            <span>{filled}/{schema.fields.length} fields</span>
            {preview && (
              <span className={preview.over_limit ? "text-red-400" : ""}>
                {preview.chars} chars {"·"} ~{preview.approx_tokens}/{preview.token_limit} tokens
              </span>
            )}
            <GhostButton onClick={() => setShowCaption(!showCaption)}>
              {showCaption ? "hide" : "show"} assembled
            </GhostButton>
          </div>
        </div>

        {showCaption && (
          <pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap rounded bg-[#0e1015] p-2 text-[10px] text-[var(--color-muted)]">
            {captionText || "(empty)"}
          </pre>
        )}

        {schema.groups.map((group) => (
          <div key={group} className="mt-3">
            <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-[var(--color-accent2)]">{group}</div>
            {schema.fields.filter((f) => f.group === group).map((f) => (
              <div key={f.key} className="mb-2">
                <div className="flex items-baseline gap-2">
                  <label className="text-[11px] text-[var(--color-ink)]">{f.key}</label>
                  <button onClick={() => setOpenHelp(openHelp === f.key ? null : f.key)}
                    title="what belongs in this field"
                    className="text-[10px] text-[var(--color-muted)] hover:text-[var(--color-accent2)]">
                    {openHelp === f.key ? "hide help" : "?"}
                  </button>
                </div>
                {openHelp === f.key && (
                  <div className="my-1 rounded bg-[#0e1015] p-2 text-[10px] leading-relaxed text-[var(--color-muted)]">{f.help}</div>
                )}
                <textarea className={inp + " min-h-[52px] text-[11px]"} rows={2}
                  value={fields[f.key] || ""} onChange={(e) => set(f.key, e.target.value)} />
              </div>
            ))}
          </div>
        ))}
      </div>

      {/* ---- lyrics + tags ---- */}
      <div className="rounded border border-[var(--color-line)] p-3">
        <SectionTitle>Lyrics and sections</SectionTitle>
        <div className="mt-2 flex flex-wrap items-center gap-1">
          {schema.tags.map((t) => (
            <GhostButton key={t} onClick={() => insertTag(t)} title={`append a bare [${t}] section`}>
              +{t}
            </GhostButton>
          ))}
          <span className="ml-2 text-[10px] text-[var(--color-muted)]">
            {tagsInLyrics.length} sections
          </span>
        </div>
        {loadedTags.length > 0 && (
          <div className="mt-2 rounded border border-red-600/50 bg-red-950/30 p-2 text-[11px] text-red-300">
            These tags carry more than a section name: {loadedTags.join(" ")}
            <div className="mt-1 text-[10px] text-red-200/80">
              Measured: the model SINGS everything after the section name, and it wrecked the take
              ("driving fuller, a scar story carved in stone"). Keep tags bare and put the direction
              in Groove &amp; Foundation Progression instead, naming the section.
            </div>
          </div>
        )}
        {stageDirections.length > 0 && (
          <div className="mt-2 rounded border border-amber-600/50 bg-amber-950/30 p-2 text-[11px] text-amber-300">
            These look like stage directions: {stageDirections.slice(0, 3).join(" ")}
            {stageDirections.length > 3 && ` (+${stageDirections.length - 3} more)`}
            <div className="mt-1 text-[10px] text-amber-200/80">
              Parentheses here are LYRICS, not directions. Verified against our own render: the
              shipped template's "(rain on the window)" came back sung. Move instrument and
              arrangement notes into Embellishments or Groove &amp; Foundation Progression, and keep
              parentheses for words you actually want a backing vocal to sing.
            </div>
          </div>
        )}
        <textarea className={inp + " mt-2 min-h-[220px] font-mono text-[11px]"} rows={14}
          value={lyrics} onChange={(e) => setLyrics(e.target.value)}
          placeholder={"[Intro]\n\n[Verse]\nyour words here\n\n[Chorus]\n…"} />
        <div className="mt-1 text-[10px] text-[var(--color-muted)]">
          Undocumented but real: <code>{" ^ "}</code> (spaces either side) becomes a line break.
        </div>
      </div>

      {/* ---- render ---- */}
      <div className="rounded border border-[var(--color-line)] p-3">
        <SectionTitle>Render</SectionTitle>
        <div className="mt-2 grid grid-cols-2 gap-2 md:grid-cols-4">
          <Field label="Title"><input className={inp} value={title} onChange={(e) => setTitle(e.target.value)} /></Field>
          <Field label="Max seconds" hint={`ceiling only, up to ${schema.max_seconds}; it often stops short`}>
            <input className={inp} value={seconds} onChange={(e) => setSeconds(e.target.value)} />
          </Field>
          <Field label="Seed" hint="blank = random per take">
            <input className={inp} value={seed} onChange={(e) => setSeed(e.target.value)} placeholder="random" />
          </Field>
          <Field label="Takes"><input className={inp} value={count} onChange={(e) => setCount(e.target.value)} /></Field>
        </div>

        <div className="mt-2">
          <GhostButton onClick={() => setShowAdv(!showAdv)}>{showAdv ? "hide" : "show"} sampler settings</GhostButton>
        </div>
        {showAdv && (
          <div className="mt-2 grid grid-cols-2 gap-2 md:grid-cols-4">
            <Field label="cfg_scale" hint="template default 1.7; higher tightens caption adherence">
              <input className={inp} value={cfgScale} onChange={(e) => setCfgScale(e.target.value)} />
            </Field>
            <Field label="top_k" hint="template default 50">
              <input className={inp} value={topK} onChange={(e) => setTopK(e.target.value)} />
            </Field>
            <Field label="steps" hint="template default 30">
              <input className={inp} value={steps} onChange={(e) => setSteps(e.target.value)} />
            </Field>
            <Field label="Tiled decode" hint="lower VRAM on long songs; small risk of seams at tile boundaries, so turn it off if you hear one">
              <input type="checkbox" checked={tiled} onChange={(e) => setTiled(e.target.checked)} />
            </Field>
          </div>
        )}

        <div className="mt-3 flex items-center gap-2">
          <PrimaryButton onClick={run} disabled={busy || !!(avail && !avail.available) || !!preview?.over_limit}
            title={preview?.over_limit ? "caption is over the 5000-token limit" : "generate"}>
            Generate
          </PrimaryButton>
          <span className="text-[10px] text-[var(--color-muted)]">
            about 110s of GPU per 60s of audio. Masters are saved lossless; the library export
            button also offers 320k MP3.
          </span>
        </div>
      </div>
    </div>
  );
}
