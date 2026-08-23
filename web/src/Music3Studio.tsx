import { useEffect, useMemo, useState } from "react";
import { api, type Config, type Project, type SongDraft } from "./api";
import { Field, inp, PrimaryButton, GhostButton, SectionTitle, pollJob, rid, type RunCtx } from "./ui";
import { useDraftCtx, useDrafts } from "./drafts";

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

type Props = { cfg: Config; busy: boolean; song: SongDraft | null; projectName?: string; goTo?: (m: string) => void } & RunCtx;
type SchemaField = { key: string; group: string; help: string };
type Schema = {
  fields: SchemaField[]; groups: string[]; tags: string[];
  defaults: Record<string, number | boolean | string>; max_seconds: number;
};

export function Music3StudioForm({ cfg: _cfg, busy, song, projectName, goTo, ...ctx }: Props) {
  const d = useDrafts("music3");
  const draftCtx = useDraftCtx();   // imperative writes into the Song tab's drafts (send-to-song)
  // The Song page's draft carries the actual song title; the canonical SongDraft passed down as
  // `song` does not. Read it so an import can name the render after the song.
  const [songPageTitle] = useDrafts("song").use("title", "");
  const [schema, setSchema] = useState<Schema | null>(null);
  const [avail, setAvail] = useState<{ available: boolean; reason?: string } | null>(null);

  const [fields, setFields] = d.use<Record<string, string>>("fields", {});
  const [lyrics, setLyrics] = d.use("lyrics", "");
  const [title, setTitle] = d.use("title", "");
  const [seconds, setSeconds] = d.use("seconds", "360");
  const [seed, setSeed] = d.use("seed", "");
  const [count, setCount] = d.use("count", "1");
  const [cfgScale, setCfgScale] = d.use("cfgScale", "1.7");
  const [topK, setTopK] = d.use("topK", "50");
  const [steps, setSteps] = d.use("steps", "");
  const [tiled, setTiled] = d.use("tiled", false);
  // The seed picks the COMPOSITION (the AR token trajectory); the mix seed picks the RENDER of
  // it. Blank = follow the seed, which is the shipped-template behaviour. Holding the seed and
  // changing only the mix seed re-renders the same performance and skips the whole AR stage via
  // ComfyUI's node cache (AIPLAY measured 15s vs 50s).
  const [mixSeed, setMixSeed] = d.use("mixSeed", "");
  const [flowCfg, setFlowCfg] = d.use("flowCfg", "");
  // "template" = the shipped graph exactly (euler/simple, 30 steps). "shift5" = AIPLAY's
  // measured alternative (euler over shift-5 sigmas, 15 steps) - switch freely per render.
  const [schedule, setSchedule] = d.use<"template" | "shift5">("schedule", "template");
  const [showAdv, setShowAdv] = useState(false);
  const [openHelp, setOpenHelp] = useState<string | null>(null);

  const [projects, setProjects] = useState<Project[]>([]);
  const [importFrom, setImportFrom] = useState("");
  const [note, setNote] = useState("");
  const [preview, setPreview] = useState<{ chars: number; approx_tokens: number; token_limit: number; over_limit: boolean } | null>(null);
  const [showCaption, setShowCaption] = useState(false);
  const [captionText, setCaptionText] = useState("");

  // Writers. A rewrite is never applied straight over the fields: it lands here as a proposal and
  // is accepted field by field, so a good Primary is not lost to a worse Vocal FX.
  const [brief, setBrief] = d.use("brief", "");
  const [writing, setWriting] = useState("");
  const [ws, setWs] = useState<{ skill_installed: boolean; provider: string } | null>(null);
  const [prop, setProp] = useState<null | {
    writer: string; provider: string; fields: Record<string, string>;
    before: Record<string, string>; changed: string[];
    families?: string[]; templates?: string[]; auto_routed?: boolean;
    lyrics?: string;   // drafted only when the lyrics box was empty; never a rewrite
  }>(null);

  // Reference pinning. Auto-routing is the default, but it is NOT reproducible: the same brief
  // picked a different template trio on two consecutive runs, which makes an A/B meaningless.
  // Pinning is how you hold the references still while changing something else.
  type Card = { id: string; style: string; secondary: string; tempo_key: string; mood: string; vocal: string; palette: string; template: string };
  const [refsOpen, setRefsOpen] = useState(false);
  const [writerHelp, setWriterHelp] = useState(false);
  const [fams, setFams] = useState<{ file: string; label: string; cards: number }[]>([]);
  const [family, setFamily] = d.use("family", "");
  const [cards, setCards] = useState<Card[]>([]);
  const [pinned, setPinned] = d.use<string[]>("pinned", []);

  useEffect(() => {
    api.music3Schema().then(setSchema).catch(() => setSchema(null));
    api.music3Available().then(setAvail).catch(() => setAvail({ available: false, reason: "backend unreachable" }));
    api.projects().then(setProjects).catch(() => setProjects([]));
    api.music3WriterStatus().then(setWs).catch(() => setWs(null));
    api.music3References().then((r) => setFams(r.families || [])).catch(() => setFams([]));
  }, []);

  useEffect(() => {
    if (!family) { setCards([]); return; }
    api.music3References(family).then((r) => setCards(r.cards || [])).catch(() => setCards([]));
  }, [family]);

  const togglePin = (t: string) =>
    setPinned(pinned.includes(t) ? pinned.filter((x) => x !== t)
                                 : pinned.length >= 3 ? pinned : [...pinned, t]);

  async function runWriter(writer: "ours" | "skill") {
    setWriting(writer);
    setProp(null);
    try {
      const r = await api.music3Write({
        writer, brief, fields, lyrics,
        // only meaningful for the skill writer; harmless on ours
        family: family || undefined,
        templates: pinned.length ? pinned : undefined,
      });
      setProp(r);
      const who = writer === "skill" ? "MiniMax skill" : "our writer";
      const extra = r.lyrics ? " It also drafted lyrics (the box was empty) - accept or discard them below." : "";
      setNote(r.changed.length
        ? `${who} proposed ${r.changed.length} field change(s) - review below.${extra}`
        : `${who} returned nothing different.${extra}`);
    } catch (e) {
      setNote(`${writer} writer failed: ${e}`);
    } finally {
      setWriting("");
    }
  }

  const acceptField = (k: string) => {
    if (!prop) return;
    setFields({ ...fields, [k]: prop.fields[k] });
    setProp({ ...prop, changed: prop.changed.filter((c) => c !== k) });
  };
  const acceptAll = () => {
    if (!prop) return;
    const next = { ...fields };
    prop.changed.forEach((k) => { next[k] = prop.fields[k]; });
    setFields(next);
    // the guard repeats at accept time: if lyrics were typed while the writer ran, keep them
    if (prop.lyrics && !lyrics.trim()) setLyrics(prop.lyrics);
    setProp(null);
  };

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

  async function importSong(s: SongDraft | null, label: string, titleGuess?: string) {
    if (!s || !(s.blocks || []).length) { setNote(`${label} has no song arrangement`); return; }
    try {
      const r = await api.music3FromSong(s);
      // Merge only the NON-EMPTY imported values: song_to_fields returns every key, most blank,
      // and spreading them wholesale would wipe fields the user already wrote by hand.
      const incoming = Object.fromEntries(
        Object.entries(r.fields as Record<string, string>).filter(([, v]) => (v || "").trim()));
      const next = { ...fields, ...incoming };
      setFields(next);
      setLyrics(r.lyrics);
      if (titleGuess?.trim()) setTitle(titleGuess.trim());
      const got = schema ? Object.values(next).filter((v) => (v || "").trim()).length : 0;
      const total = schema?.fields.length ?? 13;
      setNote(`imported ${label}: ${(s.blocks || []).length} sections, ${got}/${total} caption fields seeded. `
        + `The empty fields are not optional padding - a sparse caption is what lost the genre in testing. `
        + `Run a writer (step 2, right) to fill them; style cues went into the progression fields, not the tags.`);
    } catch (e) {
      setNote(`import failed: ${e}`);
    }
  }

  // The REVERSE of importSong: this tab's caption + lyrics -> the Song tab (ACE-Step).
  // Lyrics move verbatim; the LLM compresses the caption into ACE's 10-12 dense tag
  // phrases plus per-section 1-3 word styles and durations. Then switch to the Song tab.
  const [sending, setSending] = useState(false);
  async function sendToSong() {
    if (!lyrics.trim()) { setNote("write lyrics with [Section] markers first - the structure comes from them"); return; }
    setSending(true);
    setNote("translating the caption into ACE-Step tags (a few seconds)...");
    try {
      const r = await api.music3ToSong({
        fields, lyrics, title: title || songPageTitle || projectName,
        seconds: parseFloat(seconds) || undefined,
      }) as { title: string; tags: string; bpm: number | null; keyscale: string | null; instrumental: boolean; blocks: { type: string; seconds: number; lyrics: string; style: string }[] };
      const blocks = r.blocks.map((b, i) => ({
        id: `m3_${Date.now()}_${i}`, type: b.type, seconds: b.seconds,
        lyrics: b.lyrics || "", style: b.style || "", locked: false,
      }));
      draftCtx.set("song", "blocks", blocks);
      draftCtx.set("song", "tags", r.tags);
      draftCtx.set("song", "instrumental", r.instrumental);
      draftCtx.set("song", "drive", "compile");
      draftCtx.set("song", "title", r.title || title || "");
      draftCtx.set("song", "tpl", "");
      draftCtx.set("song", "dirty", false);
      if (r.bpm) draftCtx.set("song.tuning", "bpm", String(r.bpm));
      if (r.keyscale) draftCtx.set("song.tuning", "keyscale", r.keyscale);
      setNote(`sent to the Song tab: ${blocks.length} sections, tags "${r.tags.slice(0, 60)}...". `
        + "Check the per-section styles and seconds there - ACE timing is explicit where Music 3's is not.");
      goTo?.("song");
    } catch (e) {
      setNote(`send to Song tab failed: ${e}`);
    }
    setSending(false);
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
      // the render title comes from the song's own title, falling back to the project name
      const t = ((page as { title?: string }).title || p?.name || "").trim();
      await importSong(merged, p?.name || "project", t);
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
    // With a mix seed set, takes vary the RENDER while the composition holds: the seed is
    // pinned (a random one is drawn once if blank) and mix_seed+i walks the re-rolls. Without
    // one, takes vary the seed as before.
    const mixRolls = mixSeed.trim() !== "";
    const baseSeed = seed.trim() ? parseInt(seed)
      : mixRolls ? Math.floor(Math.random() * 2 ** 31) : undefined;
    for (let i = 0; i < n; i++) {
      try {
        const r = await api.music3Generate({
          fields, lyrics, title,
          seconds: parseFloat(seconds) || 360,
          // blank seed = a fresh random one per take, which is what a seed hunt wants
          seed: baseSeed === undefined ? undefined : mixRolls ? baseSeed : baseSeed + i,
          ...(mixRolls ? { mix_seed: parseInt(mixSeed) + i } : {}),
          ...(flowCfg.trim() ? { flow_cfg: parseFloat(flowCfg) } : {}),
          schedule,
          cfg_scale: parseFloat(cfgScale) || 1.7,
          top_k: parseInt(topK) || 50,
          ...(steps.trim() ? { steps: parseInt(steps) } : {}),
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

  const allBrackets = (lyrics.match(/\[[^\]]+\]/g) || []);
  // A bracketed vocal attribution ([female vocal], [male vocal]) is a working casting lever, not
  // a section: keep it out of the section count and the loaded-tag warning.
  const tagsInLyrics = allBrackets.filter((t) => !/\bvocals?\b/i.test(t));
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
        <SectionTitle>1 {"·"} Song</SectionTitle>
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <GhostButton onClick={() => importSong(song, "the loaded song", songPageTitle || projectName)} disabled={!song?.blocks?.length}
            title={song?.blocks?.length ? "fill the caption and lyrics from the song currently loaded" : "no song loaded"}>
            Import loaded song
          </GhostButton>
          <GhostButton onClick={sendToSong} disabled={sending || !lyrics.trim()}
            title="the reverse: translate THIS caption + lyrics into a Song tab (ACE-Step) arrangement - lyrics verbatim, caption compressed to dense tags">
            {sending ? "Translating…" : "→ Song tab (ACE)"}
          </GhostButton>
          <select className={inp} value={importFrom} onChange={(e) => importProject(e.target.value)}
            title="copies that project's song arrangement into this caption - it does NOT switch projects; use the top-bar Open for that">
            <option value="">or copy a song from another project…</option>
            {projects.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
          </select>
          <span className="text-[10px] text-[var(--color-muted)]">
            This copies a song INTO the caption below - it never switches projects (the top-bar Open
            does that, and also restores a project's saved Music 3 caption). Sections and per-block
            style cues come across; seconds do not, because Music 3 has no per-section timing at all.
          </span>
        </div>
        {note && <div className="mt-2 text-[11px] text-[var(--color-accent2)]">{note}</div>}
      </div>

      {/* On a wide window the four working areas pair up: caption beside its writers,
          lyrics beside the render controls. The single-column stack only ever filled
          the left third of a large browser window. */}
      <div className="grid grid-cols-1 items-start gap-4 xl:grid-cols-2">
      <div className="space-y-4">
      {/* ---- caption fields ---- */}
      <div className="rounded border border-[var(--color-line)] p-3">
        <div className="flex items-center justify-between">
          <SectionTitle>2 {"·"} Caption</SectionTitle>
          <div className="flex items-center gap-2 text-[10px] text-[var(--color-muted)]">
            <span className={filled < schema.fields.length ? "text-amber-400" : ""}
              title={filled < schema.fields.length
                ? "empty fields weaken the caption - measured: a sparse caption lost the genre entirely. The writers can fill them."
                : "every field has content"}>
              {filled}/{schema.fields.length} fields</span>
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
                  value={fields[f.key] || ""} onChange={(e) => set(f.key, e.target.value)}
                  placeholder={(f.help.split(". ")[0] || "").slice(0, 110)} />
              </div>
            ))}
          </div>
        ))}
      </div>
      </div>
      <div className="space-y-4">
      {/* ---- writers ---- */}
      <div className="rounded border border-[var(--color-line)] p-3">
        <SectionTitle>2 {"·"} Caption writers</SectionTitle>
        <textarea className={inp + " mt-2 min-h-[64px] text-[11px]"} rows={3} value={brief}
          onChange={(e) => setBrief(e.target.value)}
          placeholder="Brief in plain words: what the song is, the singers, the feel, anything that must or must not be in it. Both writers work from this plus whatever fields are already filled. If the lyrics box is empty they also draft lyrics; populated lyrics are never touched." />
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <GhostButton onClick={() => runWriter("ours")} disabled={!!writing}
            title="our writer: one pass, carrying what we measured on this box (guitars as Primary, metal production language, per-section naming)">
            {writing === "ours" ? "writing…" : "Write with our writer"}
          </GhostButton>
          <GhostButton onClick={() => runWriter("skill")} disabled={!!writing || !ws?.skill_installed}
            title={ws?.skill_installed
              ? "MiniMax's own caption-rewriter: routes to a style family, compares cards, reads only the templates it picks (1000 available)"
              : "the vendored skill is missing from backend/skills/"}>
            {writing === "skill" ? "routing, picking, writing…" : "Rewrite with MiniMax skill"}
          </GhostButton>
          {ws && <span className="text-[10px] text-[var(--color-muted)]">via {ws.provider}</span>}
        </div>
        <div className="mt-2">
          <GhostButton onClick={() => setWriterHelp(!writerHelp)}>{writerHelp ? "hide" : "how the writers work"}</GhostButton>
        </div>
        {writerHelp && <div className="mt-2 rounded bg-[#0e1015] p-2 text-[10px] leading-relaxed text-[var(--color-muted)]">
          <b className="text-[var(--color-ink)]">Two writers, and you can mix them.</b> Run one, accept
          the fields you like, run the other, accept from that. Anything you accept becomes the
          "existing fields" the next writer is told to improve rather than replace, so you can build
          one caption out of both. <b className="text-[var(--color-ink)]">Our writer</b> knows what we
          measured on this box: guitars in Primary, metal production language, absolute negatives
          stated absolutely, and how to force a guitar solo.
          <b className="text-[var(--color-ink)]"> MiniMax's skill</b> knows their 1000-caption library
          and is stronger on genre vocabulary. Nothing is ever overwritten: changes arrive as a
          proposal with <i>use this</i> / <i>keep mine</i> per field.
        </div>}

        {/* ---- reference pinning ---- */}
        <div className="mt-2">
          <GhostButton onClick={() => setRefsOpen(!refsOpen)}
            title="choose which of the skill's 1000 reference captions it may look at">
            {refsOpen ? "hide" : "choose"} references{pinned.length ? ` (${pinned.length} pinned)` : family ? " (family pinned)" : " (auto)"}
          </GhostButton>
        </div>
        {refsOpen && (
          <div className="mt-2 rounded border border-[var(--color-line)] p-2">
            <div className="text-[10px] leading-relaxed text-[var(--color-muted)]">
              Left alone, the skill routes itself. That is convenient but <b>not reproducible</b>: the
              same brief picked a different template trio on two consecutive runs. Pin a family, or up
              to three specific references, to hold them still across an A/B.
            </div>
            <select className={inp + " mt-2"} value={family}
              onChange={(e) => { setFamily(e.target.value); setPinned([]); }}>
              <option value="">auto-route (let the skill choose)</option>
              {fams.map((f) => <option key={f.file} value={f.file}>{f.label} ({f.cards})</option>)}
            </select>
            {cards.length > 0 && (
              <div className="mt-2 max-h-56 overflow-auto">
                {cards.map((c) => {
                  const on = pinned.includes(c.template);
                  return (
                    <button key={c.template} onClick={() => togglePin(c.template)}
                      title={`${c.mood}\n\nPalette: ${c.palette}`}
                      className={`mb-1 block w-full rounded border p-1 text-left text-[10px] ${
                        on ? "border-[var(--color-accent2)] bg-[var(--color-accent2)]/10"
                           : "border-[var(--color-line)] hover:border-[var(--color-muted)]"}`}>
                      <div className="text-[var(--color-ink)]">{on ? "✓ " : ""}{c.style}</div>
                      <div className="text-[var(--color-muted)]">{c.tempo_key} {"·"} {c.vocal}</div>
                    </button>
                  );
                })}
              </div>
            )}
            {pinned.length > 0 && (
              <div className="mt-1 flex items-center gap-2 text-[10px] text-[var(--color-accent2)]">
                <span>{pinned.length}/3 pinned; routing and selection are both skipped</span>
                <GhostButton onClick={() => setPinned([])}>clear</GhostButton>
              </div>
            )}
          </div>
        )}

        {prop && (
          <div className="mt-3 rounded border border-[var(--color-accent2)]/40 bg-[#0e1015] p-2">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-[11px] text-[var(--color-accent2)]">
                {prop.writer === "skill" ? "MiniMax skill" : "our writer"} {"·"} {prop.changed.length} change(s) left
              </span>
              <span className="flex-1" />
              <GhostButton onClick={acceptAll} disabled={!prop.changed.length}>accept all</GhostButton>
              <GhostButton onClick={() => setProp(null)}>discard</GhostButton>
            </div>
            {/* What the skill based its answer on. SKILL.md says to keep this hidden unless asked;
                here it is always shown, because knowing which templates informed a caption is the
                difference between judging the result and guessing at it. */}
            {(prop.families?.length || prop.templates?.length) ? (
              <div className="mt-1 text-[10px] text-[var(--color-muted)]">
                {prop.auto_routed ? "auto-routed to" : "pinned"} {prop.families?.join(", ") || "(no family)"}
                {" · references "}{prop.templates?.join(", ") || "none"}
                {prop.auto_routed && <span className="text-amber-400"> {"·"} auto-routing varies run to run; pin these to reproduce this caption</span>}
              </div>
            ) : null}
            {prop.lyrics && !lyrics.trim() && (
              <div className="mt-2 border-t border-[var(--color-line)] pt-2">
                <div className="flex items-center gap-2">
                  <span className="text-[11px] font-semibold text-[var(--color-ink)]">Lyrics</span>
                  <span className="text-[10px] text-[var(--color-muted)]">drafted because the box was empty - existing lyrics are never rewritten</span>
                  <span className="flex-1" />
                  <GhostButton onClick={() => { setLyrics(prop.lyrics || ""); setProp({ ...prop, lyrics: undefined }); }}>use these</GhostButton>
                  <GhostButton onClick={() => setProp({ ...prop, lyrics: undefined })}>discard</GhostButton>
                </div>
                <pre className="mt-1 max-h-48 overflow-auto whitespace-pre-wrap rounded bg-emerald-950/20 p-1 text-[10px] leading-relaxed text-emerald-200/80">{prop.lyrics}</pre>
              </div>
            )}
            {prop.changed.length === 0 && !prop.lyrics && (
              <div className="mt-2 text-[10px] text-[var(--color-muted)]">Every proposed change has been dealt with.</div>
            )}
            {prop.changed.map((k) => (
              <div key={k} className="mt-2 border-t border-[var(--color-line)] pt-2">
                <div className="flex items-center gap-2">
                  <span className="text-[11px] font-semibold text-[var(--color-ink)]">{k}</span>
                  <span className="flex-1" />
                  <GhostButton onClick={() => acceptField(k)}>use this</GhostButton>
                  <GhostButton onClick={() => setProp({ ...prop, changed: prop.changed.filter((c) => c !== k) })}>keep mine</GhostButton>
                </div>
                {(prop.before[k] || "").trim() && (
                  <div className="mt-1 rounded bg-red-950/20 p-1 text-[10px] leading-relaxed text-red-200/70">
                    <span className="mr-1 opacity-60">now</span>{prop.before[k]}
                  </div>
                )}
                <div className="mt-1 rounded bg-emerald-950/20 p-1 text-[10px] leading-relaxed text-emerald-200/80">
                  <span className="mr-1 opacity-60">proposed</span>{prop.fields[k] || "(empty)"}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
      </div>
      </div>

      <div className="grid grid-cols-1 items-start gap-4 xl:grid-cols-2">
      <div className="space-y-4">
      {/* ---- lyrics + tags ---- */}
      <div className="rounded border border-[var(--color-line)] p-3">
        <SectionTitle>3 {"·"} Lyrics</SectionTitle>
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
              Parentheses are unreliable BOTH ways: sometimes obeyed as direction, sometimes sung
              (our render sang the shipped template's "(rain on the window)"). Only write words
              here that are acceptable to hear sung; instrument and arrangement notes belong in
              Embellishments or Groove &amp; Foundation Progression instead.
            </div>
          </div>
        )}
        <textarea className={inp + " mt-2 min-h-[220px] font-mono text-[11px]"} rows={14}
          value={lyrics} onChange={(e) => setLyrics(e.target.value)}
          placeholder={"[Intro]\n\n[Verse]\nyour words here\n\n[Chorus]\n…"} />
        <div className="mt-1 text-[10px] text-[var(--color-muted)]">
          Undocumented but real: <code>{" ^ "}</code> (spaces either side) becomes a line break.
          Casting lever: a one-or-two-word bracketed vocal attribution on its own line directly
          under a section tag works, e.g. <code>[Chorus]</code> then <code>[female vocal]</code>.
          Keep it that short; anything longer risks being sung.
        </div>
      </div>
      </div>
      <div className="space-y-4">
      {/* ---- render ---- */}
      <div className="rounded border border-[var(--color-line)] p-3">
        <SectionTitle>4 {"·"} Render</SectionTitle>
        <div className="mt-2 grid grid-cols-2 gap-2 md:grid-cols-4">
          <Field label="Title"><input className={inp} value={title} onChange={(e) => setTitle(e.target.value)} /></Field>
          <Field label="Max seconds" hint={`ceiling only, up to ${schema.max_seconds}; it often stops short`}>
            <input className={inp} value={seconds} onChange={(e) => setSeconds(e.target.value)} />
          </Field>
          <Field label="Seed" hint="blank = random per take">
            <input className={inp} value={seed} onChange={(e) => setSeed(e.target.value)} placeholder="random" />
          </Field>
          <Field label="Takes"><input className={inp} value={count} onChange={(e) => setCount(e.target.value)} /></Field>
          {/* Promoted out of the advanced panel: measured over 12 takes, cfg is the one knob that
              changed how reliably the song came out with the STRUCTURE the caption asked for.
              At 2.6 every seed produced an intro, where 1.7 collapsed one seed to no intro at all.
              It did not help guitar solos. Default stays at the template's 1.7. */}
          <Field label="cfg_scale" hint="1.7 is the template default. Higher tightens adherence to the caption: 2.6 gave a proper intro at every seed where 1.7 lost one. It did not improve solos.">
            <input className={inp} value={cfgScale} onChange={(e) => setCfgScale(e.target.value)} />
          </Field>
        </div>

        <div className="mt-2">
          <GhostButton onClick={() => setShowAdv(!showAdv)}>{showAdv ? "hide" : "show"} sampler settings</GhostButton>
        </div>
        {showAdv && (
          <>
          <div className="mt-2 grid grid-cols-2 gap-2 md:grid-cols-4">
            <Field label="Mix seed" hint="blank = follow Seed (the default). Set it to re-render the SAME composition with new sound: the Seed picks the performance, this picks the render. With Takes > 1, takes then re-roll the mix while the composition holds - and skipping the composition stage makes those re-rolls several times faster.">
              <input className={inp} value={mixSeed} onChange={(e) => setMixSeed(e.target.value)} placeholder="= seed" />
            </Field>
            <Field label="Render cfg" hint="blank = follow cfg_scale (the default). cfg_scale steers the composition; this steers the render. Splitting them samples off the diagonal - e.g. composition at 2.6 for structure with the render kept at 1.7.">
              <input className={inp} value={flowCfg} onChange={(e) => setFlowCfg(e.target.value)} placeholder="= cfg_scale" />
            </Field>
            <Field label="Schedule" hint="template = the shipped graph (euler/simple). shift-5 = AIPLAY Studio's measured alternative, ~2x closer to converged at half the time on their rig - unproven on ours until we A/B it. Switch freely per render.">
              <select className={inp} value={schedule} onChange={(e) => setSchedule(e.target.value as "template" | "shift5")}>
                <option value="template">template (default, 30 steps)</option>
                <option value="shift5">shift-5 (15 steps)</option>
              </select>
            </Field>
            <Field label="steps" hint={schedule === "shift5" ? "blank = 15 (the shift-5 default)" : "blank = 30 (the template default)"}>
              <input className={inp} value={steps} onChange={(e) => setSteps(e.target.value)}
                placeholder={schedule === "shift5" ? "15" : "30"} />
            </Field>
          </div>
          <div className="mt-2 grid grid-cols-2 gap-2 md:grid-cols-4">
            <Field label="top_k" hint="template default 50">
              <input className={inp} value={topK} onChange={(e) => setTopK(e.target.value)} />
            </Field>
            <Field label="Tiled decode" hint="lower VRAM on long songs; small risk of seams at tile boundaries, so turn it off if you hear one">
              <input type="checkbox" checked={tiled} onChange={(e) => setTiled(e.target.checked)} />
            </Field>
          </div>
          </>
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
      </div>
    </div>
  );
}
