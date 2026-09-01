import { useEffect, useState } from "react";
import { api, type Config, type LibItem } from "./api";
import { Field, GhostButton, PrimaryButton, SectionTitle, inp, rid, type RunCtx } from "./ui";
import { useDraftCtx, useDrafts } from "./drafts";
import { openLightbox } from "./Lightbox";

// ============================================================================
// YouTube tab: turn a finished song into an upload-ready static video.
// 1 · Cover art: the LLM proposes visual concepts from the song (title/tags/lyrics),
//     Krea2 renders N candidates with the TITLE typeset into the image (Ideogram4
//     text elements), the user picks one.
// 2 · Render: ffmpeg (Mac, GPU-free) loops the pick over the full track at 1 fps
//     (x264 stillimage + AAC 320k + faststart) into library/<id>.mp4; the library
//     card's download button hands back the friendly-named upload file.
// ============================================================================

type Concept = { name: string; overview: string; background: string; aesthetics: string; lighting: string; palette: string[]; title_style: string; features_woman?: boolean };
type Cand = { jobId: string; seed: number; url?: string; err?: string; pct?: number };
// The band wordmark + title lettering are GLOBAL app config (choose-once), not per-project drafts.
type Wordmark = { text: string; font: string; treatment: string; position: string; scale: number;
  title_font: string; title_treatment: string; title_position: string; title_scale: number;
  cover_model_still?: string };
const WM_SIZES: { label: string; scale: number }[] = [
  { label: "S", scale: 0.3 }, { label: "M", scale: 0.4 }, { label: "L", scale: 0.52 }];
const TITLE_SIZES: { label: string; scale: number }[] = [
  { label: "S", scale: 0.58 }, { label: "M", scale: 0.72 }, { label: "L", scale: 0.85 }];
const VIZ_SIZES: { label: string; scale: number }[] = [
  { label: "S", scale: 0.25 }, { label: "M", scale: 0.33 }, { label: "L", scale: 0.45 }];

const EMPTY_CONCEPT: Concept = { name: "", overview: "", background: "", aesthetics: "", lighting: "", palette: [], title_style: "" };

// 12-slot placement picker: a mini map of the cover mirroring backend/wordmark.py
// POSITIONS (4 vertical bands x 3 columns; bare band name = centered).
function PlacementGrid({ value, onPick }: { value: string; onPick: (slot: string) => void }) {
  const rows = ["header", "top", "middle", "bottom"];
  const cols = ["left", "", "right"];
  return (
    <div className="grid grid-cols-3 gap-[3px] rounded border border-[var(--color-line)] bg-[var(--color-panel)] p-[3px]"
      style={{ width: 92, aspectRatio: "16 / 9" }}>
      {rows.flatMap((r) => cols.map((c) => {
        const slot = c ? `${r}-${c}` : r;
        return (
          <button key={slot} onClick={() => onPick(slot)} title={slot}
            className={`rounded-[2px] transition ${value === slot ? "bg-[var(--color-accent)]" : "bg-[var(--color-line)] hover:bg-[var(--color-accent2)]"}`} />
        );
      }))}
    </div>
  );
}

function audioLabel(it: LibItem): string {
  const p = it.params || {};
  const take = p.take ? ` · v${p.take}` : "";
  return ((p.title || p.tags || p.source || it.mode || it.id).toString().slice(0, 48)) + take;
}

function waitMedia(jobId: string, onPct?: (p: number) => void): Promise<string> {
  return new Promise((resolve, reject) => {
    const t = window.setInterval(async () => {
      const j = await api.job(jobId).catch(() => null) as { status?: string; media_url?: string; error?: string; progress?: number; max?: number } | null;
      if (!j) return;
      if ((j.status === "running" || j.status === "finalizing") && onPct)
        onPct(j.max ? Math.max(2, Math.round((100 * (j.progress || 0)) / j.max)) : 2);
      if (j.status === "done" && j.media_url) { onPct?.(100); clearInterval(t); resolve(j.media_url + "?t=" + Date.now()); }
      else if (j.status === "error") { clearInterval(t); reject(new Error(j.error || "error")); }
    }, 1000);
  });
}

export function YouTubeForm({ busy, library, ...ctx }: { cfg: Config; busy: boolean; library: LibItem[] } & RunCtx) {
  const d = useDrafts("youtube");
  const drafts = useDraftCtx();
  const [title, setTitle] = d.use("title", "");
  const [audioId, setAudioId] = d.use("audioId", "");
  const [notes, setNotes] = d.use("notes", "");
  const [concepts, setConcepts] = d.use<Concept[]>("concepts", []);
  const [concept, setConcept] = d.use<Concept>("concept", EMPTY_CONCEPT);
  const [count, setCount] = d.use("count", 4);
  const [cands, setCands] = d.use<Cand[]>("cands", []);
  const [picked, setPicked] = d.use("picked", "");         // FINAL image id for the render (stamped copy when stamping)
  const [pickedSrc, setPickedSrc] = d.use("pickedSrc", ""); // which candidate it came from (grid highlight)
  const [pickedUrl, setPickedUrl] = d.use("pickedUrl", "");
  const [stampOn, setStampOn] = d.use("stampOn", true);
  const [res, setRes] = d.use("res", "1080p");
  // wordmark picker (global config; fetched, saved back on every change)
  const [wmFonts, setWmFonts] = useState<{ id: string; label: string }[]>([]);
  const [wmTreatments, setWmTreatments] = useState<string[]>([]);
  const [wmPositions, setWmPositions] = useState<string[]>([]);
  const [wm, setWm] = useState<Wordmark | null>(null);
  // collapsed by default (user request 2026-09-02): the font grids eat too much space to
  // live open; the per-cover placement grids + sizes stay visible above at all times
  const [wmOpen, setWmOpen] = useState(false);
  // per-cover placement overrides ("" = use the saved global) - placement depends on the
  // artwork (keep the wordmark off the subject), so it is chosen at pick time
  const [stampPos, setStampPos] = d.use("stampPos", "");
  const [stampScale, setStampScale] = d.use("stampScale", "");
  const [stampTitlePos, setStampTitlePos] = d.use("stampTitlePos", "");
  const [stampTitleScale, setStampTitleScale] = d.use("stampTitleScale", "");
  // audio visualiser baked into the video ("" = off, the classic static cover)
  const [vizStyle, setVizStyle] = d.use("vizStyle", "");
  const [vizPos, setVizPos] = d.use("vizPos", "bottom-right");
  const [vizScale, setVizScale] = d.use("vizScale", "0.33");
  // living cover: H3 FLF loop of the picked art (same image pinned at both ends)
  const [lcPrompt, setLcPrompt] = d.use("lcPrompt", "");
  const [lcTakes, setLcTakes] = d.use<Cand[]>("lcTakes", []);
  const [lcPicked, setLcPicked] = d.use("lcPicked", "");
  const [lcFinal, setLcFinal] = d.use("lcFinal", "");     // FlashVSR-upscaled loop
  const [lcBusy, setLcBusy] = useState("");
  // video style + card background
  const [style, setStyle] = d.use("style", "still");
  const [bgId, setBgId] = d.use("bgId", "");
  const [bgs, setBgs] = useState<{ id: string; label: string }[]>([]);
  const [bgPrompt, setBgPrompt] = d.use("bgPrompt", "");
  const [bgLabel, setBgLabel] = d.use("bgLabel", "");
  const [bgBusy, setBgBusy] = useState(false);
  // How the song TITLE gets onto the art. "stamped" (default) = deterministic typography,
  // always spelled right; "model" = Krea2 renders it into the image - beautiful but it
  // misspelled every take of a 5-word title, so it is the opt-in now.
  const [titleMode, setTitleMode] = d.use("titleMode", "stamped");
  // use the saved cover model (Krea2 Identity Edit) for concepts that feature a woman
  const [useModel, setUseModel] = d.use("useModel", true);
  const [meta, setMeta] = d.use<{ video_title: string; description: string; tags: string[] } | null>("meta", null);
  const [metaBusy, setMetaBusy] = useState(false);
  const [suggesting, setSuggesting] = useState(false);
  const [genning, setGenning] = useState(false);
  const [rendering, setRendering] = useState(false);
  const [err, setErr] = useState("");
  const [showAdv, setShowAdv] = useState(false);

  // Prefill the title from the Song page (or the Music 3 tab) once, if it is empty.
  useEffect(() => {
    if (title) return;
    const t = (drafts.get("song", "title") as string) || (drafts.get("music3", "title") as string) || "";
    if (t) setTitle(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    api.ytWordmarkOptions().then((o: { fonts: { id: string; label: string }[]; treatments: string[]; positions: string[]; current: Wordmark }) => {
      setWmFonts(o.fonts); setWmTreatments(o.treatments); setWmPositions(o.positions || []); setWm(o.current);
    }).catch(() => {});
    refreshBgs();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const refreshBgs = () =>
    api.ytBackgrounds().then((r: { backgrounds: { id: string; label: string }[] }) => setBgs(r.backgrounds)).catch(() => {});

  // ---- living cover: draft prompt -> hunt 2 FLF takes -> pick -> FlashVSR upscale ----
  async function lcDraftPrompt() {
    setErr(""); setLcBusy("prompt");
    try {
      const r = await api.ytLivecoverPrompt({ title, concept }) as { prompt: string };
      setLcPrompt(r.prompt);
    } catch (e) { setErr("Prompt draft failed: " + (e as Error).message); }
    setLcBusy("");
  }
  async function lcAnimate() {
    if (!pickedSrc) { setErr("Pick a cover candidate first."); return; }
    if (!lcPrompt.trim()) { setErr("Draft (or write) the motion prompt first."); return; }
    setErr(""); setLcBusy("animate");
    try {
      const r = await api.h3i2v({ still_id: pickedSrc, last_id: pickedSrc, prompt: lcPrompt,
                                  frames: 243, width: 864, height: 480,
                                  mode: "hunt", drafts: 2 }) as { drafts: { job_id: string; seed: number }[] };
      const fresh: Cand[] = r.drafts.map((t) => ({ jobId: t.job_id, seed: t.seed }));
      setLcTakes((p) => [...fresh, ...p]);
      fresh.forEach((c) => {
        waitMedia(c.jobId, (pct) => setLcTakes((p) => p.map((x) => x.jobId === c.jobId ? { ...x, pct } : x)))
          .then((url) => { setLcTakes((p) => p.map((x) => x.jobId === c.jobId ? { ...x, url } : x)); ctx.onDone(); })
          .catch(() => setLcTakes((p) => p.map((x) => x.jobId === c.jobId ? { ...x, err: "failed" } : x)));
      });
    } catch (e) { setErr("Animate failed: " + (e as Error).message); }
    setLcBusy("");
  }
  async function lcUpscale() {
    if (!lcPicked) return;
    setErr(""); setLcBusy("upscale");
    try {
      const r = await api.flashvsr({ video_id: lcPicked, scale: 2 }) as { job_id: string };
      const url = await waitMedia(r.job_id);
      void url;
      setLcFinal(r.job_id); ctx.onDone();
    } catch (e) { setErr("Upscale failed: " + (e as Error).message); }
    setLcBusy("");
  }

  // ---- background loops: generate (H3 t2v) -> loopify -> save to the named library ----
  async function bgGenerate() {
    if (!bgPrompt.trim() || !bgLabel.trim()) { setErr("Background needs a description and a name."); return; }
    setErr(""); setBgBusy(true);
    try {
      const prompt = "integrated_multimodal_description: [Shot 1] " + bgPrompt.trim() +
        " The camera is completely static, locked off for the entire 10 seconds, one continuous shot, " +
        "no cuts, and the scene stays uniformly dark and sparse throughout.\n" +
        "overall_soundscape: Quiet ambience.\nnon_diegetic_music: N/A";
      const r = await api.h3t2v({ prompt, frames: 243, width: 864, height: 480, turbo: true }) as { job_id: string };
      await waitMedia(r.job_id);
      const lp = await api.ytLoopify({ clip_id: r.job_id, label: bgLabel.trim() + " (loop)" }) as { job_id: string };
      await api.ytBackgroundsSave({ clip_id: lp.job_id, label: bgLabel.trim() });
      await refreshBgs();
      setBgId(lp.job_id); setBgPrompt(""); ctx.onDone();
    } catch (e) { setErr("Background generation failed: " + (e as Error).message); }
    setBgBusy(false);
  }

  const audios = library.filter((i) => i.audio_url);

  const setC = (k: keyof Concept, v: string | string[]) => setConcept((p) => ({ ...p, [k]: v } as Concept));
  const saveWm = (patch: Partial<Wordmark>) => {
    const next = { ...(wm as Wordmark), ...patch };
    setWm(next);
    api.ytWordmarkSave(next).catch((e) => setErr("Wordmark save failed: " + (e as Error).message));
  };
  const wmPreview = (font: string, treatment: string) =>
    `/api/youtube/wordmark_preview?text=${encodeURIComponent(wm?.text || "")}&font=${font}&treatment=${treatment}`;

  async function suggest() {
    if (!title.trim()) { setErr("Set a song title first."); return; }
    setErr(""); setSuggesting(true);
    try {
      // Lyrics come from the Song page's blocks; if that page is empty, fall back to the
      // Music 3 tab's plain lyrics. The Music 3 song brief rides along as extra direction.
      const blocks = (drafts.get("song", "blocks") as { lyrics?: string }[] | undefined) || [];
      let sections = blocks.map((b) => ({ lyrics: b.lyrics || "" })).filter((s) => s.lyrics.trim());
      if (!sections.length) {
        const m3 = (drafts.get("music3", "lyrics") as string) || "";
        if (m3.trim()) sections = [{ lyrics: m3 }];
      }
      const brief = (drafts.get("music3", "brief") as string) || "";
      const allNotes = [notes.trim(), brief.trim()].filter(Boolean).join("\n");
      const song = { title, tags: (drafts.get("song", "tags") as string) || "", sections };
      const r = await api.ytConcepts({ title, song, notes: allNotes || undefined, n: 4 }) as { concepts: Concept[] };
      setConcepts(r.concepts);
      if (r.concepts[0]) setConcept(r.concepts[0]);
    } catch (e) { setErr("Concepts failed: " + (e as Error).message); }
    setSuggesting(false);
  }

  async function generate() {
    if (!title.trim()) { setErr("Set a song title first (it is rendered into the artwork)."); return; }
    if (!concept.overview.trim()) { setErr("Describe the cover (or use Ideas from song)."); return; }
    setErr(""); setGenning(true);
    const base = Math.floor(Math.random() * 2 ** 30);
    const fresh: Cand[] = [];
    try {
      for (let i = 0; i < count; i++) {
        const r = await api.ytCover({ title, concept, seed: base + i * 101,
                                      omit_title: titleMode === "stamped",
                                      // same woman on every cover that features one: render via
                                      // Krea2 Identity Edit anchored on the saved cover model
                                      ref_still_id: (useModel && concept.features_woman && wm?.cover_model_still) || undefined }) as { job_id: string; seed: number };
        fresh.push({ jobId: r.job_id, seed: r.seed });
      }
    } catch (e) { setErr("Cover generation failed: " + (e as Error).message); }
    if (fresh.length) {
      setCands((p) => [...fresh, ...p]);
      fresh.forEach((c) => {
        waitMedia(c.jobId, (pct) => setCands((p) => p.map((x) => x.jobId === c.jobId ? { ...x, pct } : x)))
          .then((url) => { setCands((p) => p.map((x) => x.jobId === c.jobId ? { ...x, url } : x)); ctx.onDone(); })
          .catch(() => setCands((p) => p.map((x) => x.jobId === c.jobId ? { ...x, err: "failed" } : x)));
      });
    }
    setGenning(false);
  }

  // Picking a candidate stamps the band wordmark onto a COPY (deterministic Pillow text -
  // Krea2 misspelled the invented band name 5/6, so it is never model-rendered) and the
  // stamped copy becomes the render input. The un-stamped original stays in the library.
  async function stampCover(srcId: string, srcUrl?: string,
                            ov?: { pos?: string; scale?: string; tPos?: string; tScale?: string }) {
    setErr(""); setPickedSrc(srcId);
    // ov carries the just-clicked value (setState is async, the state read would be stale)
    const pos = ov?.pos ?? stampPos, scale = ov?.scale ?? stampScale;
    const tPos = ov?.tPos ?? stampTitlePos, tScale = ov?.tScale ?? stampTitleScale;
    if (stampOn && (wm?.text || "").trim()) {
      try {
        const r = await api.ytStamp({ image_id: srcId, position: pos || undefined,
                                      scale: scale ? Number(scale) : undefined,
                                      title_position: tPos || undefined,
                                      title_scale: tScale ? Number(tScale) : undefined,
                                      title: titleMode === "stamped" ? title.trim() : undefined }) as { job_id: string; media_url: string };
        setPicked(r.job_id); setPickedUrl(r.media_url + "?t=" + Date.now()); ctx.onDone();
        return;
      } catch (e) { setErr("Wordmark stamp failed (using the plain cover): " + (e as Error).message); }
    }
    setPicked(srcId); setPickedUrl(srcUrl || `/api/media/${srcId}`);
  }
  const pick = (c: Cand) => stampCover(c.jobId, c.url);

  async function writeMeta() {
    if (!title.trim()) { setErr("Set a song title first."); return; }
    setErr(""); setMetaBusy(true);
    try {
      const blocks = (drafts.get("song", "blocks") as { lyrics?: string }[] | undefined) || [];
      let sections = blocks.map((b) => ({ lyrics: b.lyrics || "" })).filter((s) => s.lyrics.trim());
      if (!sections.length) {
        const m3 = (drafts.get("music3", "lyrics") as string) || "";
        if (m3.trim()) sections = [{ lyrics: m3 }];
      }
      const song = { title, tags: (drafts.get("song", "tags") as string) || "", sections };
      const r = await api.ytMetadata({ title, song, notes: notes.trim() || undefined }) as { video_title: string; description: string; tags: string[] };
      setMeta(r);
    } catch (e) { setErr("Upload text failed: " + (e as Error).message); }
    setMetaBusy(false);
  }

  async function render() {
    if (!audioId) { setErr("Pick the song track to use."); return; }
    const loop = lcFinal || lcPicked;
    let body: Record<string, unknown>;
    if (style === "fullbleed") {
      if (!loop) { setErr("Animate the cover first (Living cover section)."); return; }
      body = { style, clip_id: loop, audio_id: audioId, title, res,
               position: stampPos || undefined, scale: stampScale ? Number(stampScale) : undefined,
               title_position: stampTitlePos || undefined,
               title_scale: stampTitleScale ? Number(stampTitleScale) : undefined };
    } else if (style === "card") {
      if (!bgId) { setErr("Pick a background loop."); return; }
      if (!loop && !pickedSrc) { setErr("Pick a cover candidate first."); return; }
      body = { style, bg_id: bgId, clip_id: loop || undefined,
               image_id: loop ? undefined : pickedSrc, audio_id: audioId, title, res };
    } else {
      if (!picked) { setErr("Pick a cover candidate first."); return; }
      body = { image_id: picked, audio_id: audioId, title, res,
               viz: vizStyle || undefined,
               viz_position: vizStyle ? vizPos : undefined,
               viz_scale: vizStyle ? Number(vizScale) : undefined };
    }
    setErr(""); setRendering(true);
    const animated = style !== "still" || !!vizStyle;
    const card = { id: rid(), title: `YouTube video · ${title || "untitled"}`, status: "running" as const, pct: animated ? 2 : 40 };
    ctx.setResults([card]);
    try {
      const r = await api.ytRender(body) as { job_id: string; media_url?: string; status: string };
      // static cover = sync (seconds); with a visualiser it is a background encode we poll
      const url = r.status === "done" && r.media_url
        ? r.media_url + "?t=" + Date.now()
        : await waitMedia(r.job_id, (pct) => ctx.patch(card.id, { pct }));
      ctx.patch(card.id, { status: "done", pct: 100, url, media: "video" });
      ctx.onDone();
    } catch (e) {
      ctx.patch(card.id, { status: "error", pct: 0, err: (e as Error).message });
    }
    setRendering(false);
  }

  return (
    <div className="space-y-4">
      <p className="text-xs text-[var(--color-muted)]">
        Turn a finished song into an <b>upload-ready YouTube video</b>: generate cover-art candidates with the
        title in the image, pick one, and it is muxed with the full track into a static MP4 (1 fps, AAC 320k).
      </p>

      <SectionTitle>1 · Song</SectionTitle>
      <Field label="Song title" hint="rendered INTO the artwork; the band name is stamped as a wordmark (step 2)">
        <input className={inp} value={title} onChange={(e) => setTitle(e.target.value)} placeholder="e.g. Ashes of the Dawn" />
      </Field>
      <Field label="Track" hint="the audio for the final video">
        <select className={inp} value={audioId} onChange={(e) => setAudioId(e.target.value)}>
          <option value="">— pick a track —</option>
          {audios.map((a) => <option key={a.id} value={a.id}>{audioLabel(a)}</option>)}
        </select>
      </Field>

      <SectionTitle>2 · Cover art</SectionTitle>
      <Field label="Visual notes" hint="optional — imagery, setting or symbols the art should show; Claude weighs these when proposing ideas">
        <textarea className={inp} rows={2} value={notes} onChange={(e) => setNotes(e.target.value)}
          placeholder="e.g. a burning longship, northern coastline, no people" />
      </Field>
      <div className="flex items-center gap-2">
        <GhostButton onClick={suggest} disabled={suggesting || busy} title="The LLM reads the song (title, tags, lyrics) and proposes 4 visual directions">
          {suggesting ? "Thinking…" : "✦ Ideas from song"}
        </GhostButton>
        {concepts.length > 0 && concepts.map((c, i) => (
          <button key={i} onClick={() => setConcept(c)}
            className={`rounded-full border px-2.5 py-1 text-[11px] transition ${concept.name === c.name ? "border-[var(--color-accent)] bg-[#2a1c19] text-[var(--color-ink)]" : "border-[var(--color-line)] bg-[var(--color-panel2)] text-[var(--color-muted)] hover:text-[var(--color-ink)]"}`}
            title={c.overview.slice(0, 200)}>{c.name || `Concept ${i + 1}`}</button>
        ))}
      </div>
      <Field label="Cover concept" hint="the whole image in prose — subject, scene, composition, atmosphere">
        <textarea className={inp} rows={4} value={concept.overview} onChange={(e) => setC("overview", e.target.value)}
          placeholder="e.g. A lone longship frozen mid-burn on a black glass sea at dusk, embers rising into a bruised violet sky, shot from low on the waterline with a 35mm lens, cinematic and desolate." />
      </Field>
      {titleMode === "model" && (
        <Field label="AI title style" hint="only used when the title is rendered in the artwork - describes the lettering to the model">
          <input className={inp} value={concept.title_style} onChange={(e) => setC("title_style", e.target.value)}
            placeholder="e.g. carved bone-white gothic serif lettering with charred edges" />
        </Field>
      )}
      <button onClick={() => setShowAdv((v) => !v)} className="text-[11px] text-[var(--color-muted)] hover:text-[var(--color-ink)]">
        {showAdv ? "▾" : "▸"} more fields (background · aesthetics · lighting · palette)
      </button>
      {showAdv && (
        <div className="space-y-3">
          <Field label="Background"><textarea className={inp} rows={2} value={concept.background} onChange={(e) => setC("background", e.target.value)} /></Field>
          <Field label="Aesthetics"><input className={inp} value={concept.aesthetics} onChange={(e) => setC("aesthetics", e.target.value)} /></Field>
          <Field label="Lighting"><input className={inp} value={concept.lighting} onChange={(e) => setC("lighting", e.target.value)} /></Field>
          <Field label="Palette" hint="comma-separated hex colors">
            <input className={inp} value={concept.palette.join(", ")}
              onChange={(e) => setC("palette", e.target.value.split(",").map((x) => x.trim()).filter(Boolean))} />
          </Field>
        </div>
      )}
      <div className="flex items-center gap-3">
        <Field label="Title lettering" hint="stamped = real typography, always spelled right">
          <select className={inp} value={titleMode} onChange={(e) => setTitleMode(e.target.value)}>
            <option value="stamped">Stamped (always correct)</option>
            <option value="model">In the artwork (AI - misspells long titles)</option>
          </select>
        </Field>
        <Field label="Candidates">
          <select className={inp} value={count} onChange={(e) => setCount(Number(e.target.value))}>
            {[2, 4, 6].map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </Field>
        <div className="flex-1 pt-5">
          <GhostButton onClick={generate} disabled={genning || busy} className="w-full text-center" title="Krea2 renders the candidates on the box (~1 min each)">
            {genning ? "Submitting…" : `Generate ${count} candidates`}
          </GhostButton>
        </div>
      </div>

      {cands.length > 0 && (
        <div className="space-y-1 rounded-lg border border-[var(--color-line)] bg-[var(--color-bg)] p-2">
          <div className="flex items-center justify-between px-0.5">
            <span className="text-[10px] text-[var(--color-muted)]">click to enlarge · "use this" to pick the cover</span>
            <button onClick={() => { setCands([]); setPicked(""); setPickedSrc(""); setPickedUrl(""); }} className="text-[11px] leading-none text-[var(--color-muted)] hover:text-red-400" title="clear candidates">×</button>
          </div>
          <div className="grid grid-cols-2 gap-1.5">
            {cands.map((c) => (
              <div key={c.jobId} className={`overflow-hidden rounded border ${picked === c.jobId ? "border-[var(--color-accent2)] ring-1 ring-[var(--color-accent2)]" : "border-[var(--color-line)]"}`}>
                <div className="relative aspect-video">
                  {c.err
                    ? <span className="flex h-full items-center justify-center text-[10px] text-red-400">failed</span>
                    : c.url
                    ? <img src={c.url} alt="" onClick={() => openLightbox(c.url!)} title={`seed ${c.seed} — click to enlarge`} className="h-full w-full cursor-zoom-in object-cover" />
                    : (
                      <span className="flex h-full flex-col items-center justify-center gap-1 text-[10px] text-[var(--color-muted)]">
                        <span>{c.pct ? `${c.pct}%` : "queued…"}</span>
                        <span className="h-0.5 w-3/4 overflow-hidden rounded bg-[var(--color-line)]">
                          <span className="block h-full bg-[var(--color-accent2)] transition-all" style={{ width: `${c.pct || 3}%` }} />
                        </span>
                      </span>
                    )}
                  {picked === c.jobId && <span className="absolute right-1 top-1 rounded bg-[var(--color-accent2)] px-1 text-[9px] text-black">cover</span>}
                </div>
                {c.url && (
                  <button onClick={() => pick(c)}
                    className={`w-full py-0.5 text-[10px] ${pickedSrc === c.jobId ? "bg-[var(--color-accent2)] text-black" : "text-[var(--color-muted)] hover:text-[var(--color-ink)]"}`}>
                    {pickedSrc === c.jobId ? "✓ this is the cover" : "use this"}
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      <SectionTitle>Cover model</SectionTitle>
      <div className="flex items-center gap-3 rounded-lg border border-[var(--color-line)] bg-[var(--color-panel2)] p-2.5">
        {wm?.cover_model_still ? (
          <img src={`/api/media/${wm.cover_model_still}`} alt="" onClick={() => openLightbox(`/api/media/${wm.cover_model_still}`)}
            className="h-16 cursor-zoom-in rounded border border-[var(--color-line)]" title="the cover model — click to enlarge" />
        ) : (
          <span className="text-[11px] text-[var(--color-muted)]">No cover model set — pick a candidate below, then</span>
        )}
        <GhostButton onClick={() => pickedSrc ? saveWm({ cover_model_still: pickedSrc }) : setErr("Pick a candidate first — the un-stamped pick becomes the reference.")}
          title="save the currently picked candidate (un-stamped) as the permanent cover-model reference">
          {wm?.cover_model_still ? "Replace with picked" : "Set from picked candidate"}
        </GhostButton>
        {wm?.cover_model_still && (
          <label className="flex items-center gap-2 text-[11px] text-[var(--color-muted)]">
            <input type="checkbox" checked={useModel} onChange={(e) => setUseModel(e.target.checked)} />
            use her on every woman-featuring cover (Identity Edit)
          </label>
        )}
      </div>

      <SectionTitle>Lettering · title &amp; band name</SectionTitle>
      <label className="flex items-center gap-2 text-xs text-[var(--color-muted)]">
        <input type="checkbox" checked={stampOn} onChange={(e) => setStampOn(e.target.checked)} />
        Stamp lettering on the picked cover
        <span className="text-[10px]">— real typography, always spelled right</span>
      </label>
      {wm && (
        <div className="space-y-2 rounded-lg border border-[var(--color-line)] bg-[var(--color-panel2)] p-2.5">
          <div className="flex items-center gap-2">
            <img src={wmPreview(wm.font, wm.treatment)} alt="" className="h-10 rounded border border-[var(--color-line)]" title="the current band wordmark" />
            <span className="text-[11px] text-[var(--color-muted)]">
              title: {wm.title_font} · {wm.title_treatment} · {wm.title_position} &nbsp;|&nbsp; band: {wm.font} · {wm.treatment} · saved globally
            </span>
            <GhostButton onClick={() => setWmOpen((v) => !v)} className="ml-auto">
              {wmOpen ? "▾ Hide fonts & placement" : "▸ Choose fonts & placement"}
            </GhostButton>
          </div>
          <div className="flex flex-wrap items-end gap-4">
            <Field label="Song title placement" hint="click a slot to move it off the subject's face">
              <div className="flex items-center gap-2">
                <PlacementGrid value={stampTitlePos || wm.title_position}
                  onPick={(slot) => { setStampTitlePos(slot); if (pickedSrc) stampCover(pickedSrc, undefined, { tPos: slot }); }} />
                <select className={inp} value={stampTitleScale}
                  onChange={(e) => { setStampTitleScale(e.target.value); if (pickedSrc) stampCover(pickedSrc, undefined, { tScale: e.target.value }); }}>
                  <option value="">size: default</option>
                  {TITLE_SIZES.map((s) => <option key={s.label} value={String(s.scale)}>size: {s.label}</option>)}
                </select>
              </div>
            </Field>
            <Field label="Band name placement">
              <div className="flex items-center gap-2">
                <PlacementGrid value={stampPos || wm.position}
                  onPick={(slot) => { setStampPos(slot); if (pickedSrc) stampCover(pickedSrc, undefined, { pos: slot }); }} />
                <select className={inp} value={stampScale}
                  onChange={(e) => { setStampScale(e.target.value); if (pickedSrc) stampCover(pickedSrc, undefined, { scale: e.target.value }); }}>
                  <option value="">size: default</option>
                  {WM_SIZES.map((s) => <option key={s.label} value={String(s.scale)}>size: {s.label}</option>)}
                </select>
              </div>
            </Field>
            {pickedSrc && (
              <GhostButton onClick={() => stampCover(pickedSrc)} title="re-stamp the picked cover with the placements above">
                ↻ Re-stamp
              </GhostButton>
            )}
          </div>
          {pickedSrc && <p className="text-[10px] text-[var(--color-muted)]">Clicking a slot re-stamps the picked cover instantly (from the clean original). Per-cover choice; the saved defaults are unchanged.</p>}
          {wmOpen && (
            <div className="space-y-2">
              <SectionTitle>Title lettering</SectionTitle>
              <p className="text-[10px] text-[var(--color-muted)]">The song title's stamped face - previews show the current title. Saved globally.</p>
              <div className="flex flex-wrap gap-1.5">
                {wmTreatments.map((t) => (
                  <button key={`t_${t}`} onClick={() => saveWm({ title_treatment: t })}
                    className={`rounded-full border px-2.5 py-1 text-[11px] transition ${wm.title_treatment === t ? "border-[var(--color-accent)] bg-[#2a1c19] text-[var(--color-ink)]" : "border-[var(--color-line)] bg-[var(--color-panel)] text-[var(--color-muted)] hover:text-[var(--color-ink)]"}`}>{t}</button>
                ))}
              </div>
              <div className="grid grid-cols-2 gap-1.5">
                {wmFonts.map((f) => (
                  <button key={`t_${f.id}`} onClick={() => saveWm({ title_font: f.id })} title={f.label}
                    className={`overflow-hidden rounded border ${wm.title_font === f.id ? "border-[var(--color-accent2)] ring-1 ring-[var(--color-accent2)]" : "border-[var(--color-line)]"}`}>
                    <img src={`/api/youtube/wordmark_preview?text=${encodeURIComponent(title.trim() || "Angel of the Shattered Sky")}&font=${f.id}&treatment=${wm.title_treatment}`} alt={f.label} className="w-full" loading="lazy" />
                  </button>
                ))}
              </div>
              <div className="flex items-end gap-3">
                <Field label="Title position">
                  <select className={inp} value={wm.title_position} onChange={(e) => saveWm({ title_position: e.target.value })}>
                    {wmPositions.map((p) => <option key={`t_${p}`} value={p}>{p}</option>)}
                  </select>
                </Field>
                <Field label="Title size">
                  <div className="flex gap-1.5">
                    {TITLE_SIZES.map((s) => (
                      <button key={`t_${s.label}`} onClick={() => saveWm({ title_scale: s.scale })}
                        className={`rounded border px-3 py-2 text-xs ${Math.abs(wm.title_scale - s.scale) < 0.01 ? "border-[var(--color-accent)] bg-[#2a1c19] text-[var(--color-ink)]" : "border-[var(--color-line)] bg-[var(--color-panel)] text-[var(--color-muted)]"}`}>{s.label}</button>
                    ))}
                  </div>
                </Field>
              </div>
              <SectionTitle>Band wordmark</SectionTitle>
              <Field label="Band name" hint="the exact letters stamped on every cover">
                <input className={inp} value={wm.text} onChange={(e) => saveWm({ text: e.target.value })} />
              </Field>
              <div className="flex flex-wrap gap-1.5">
                {wmTreatments.map((t) => (
                  <button key={t} onClick={() => saveWm({ treatment: t })}
                    className={`rounded-full border px-2.5 py-1 text-[11px] transition ${wm.treatment === t ? "border-[var(--color-accent)] bg-[#2a1c19] text-[var(--color-ink)]" : "border-[var(--color-line)] bg-[var(--color-panel)] text-[var(--color-muted)] hover:text-[var(--color-ink)]"}`}>{t}</button>
                ))}
              </div>
              <div className="grid grid-cols-2 gap-1.5">
                {wmFonts.map((f) => (
                  <button key={f.id} onClick={() => saveWm({ font: f.id })} title={f.label}
                    className={`overflow-hidden rounded border ${wm.font === f.id ? "border-[var(--color-accent2)] ring-1 ring-[var(--color-accent2)]" : "border-[var(--color-line)]"}`}>
                    <img src={wmPreview(f.id, wm.treatment)} alt={f.label} className="w-full" />
                  </button>
                ))}
              </div>
              <div className="flex items-end gap-3">
                <Field label="Default position">
                  <select className={inp} value={wm.position} onChange={(e) => saveWm({ position: e.target.value })}>
                    {(wmPositions.length ? wmPositions : ["bottom", "top"]).map((p) => <option key={p} value={p}>{p}</option>)}
                  </select>
                </Field>
                <Field label="Size">
                  <div className="flex gap-1.5">
                    {WM_SIZES.map((s) => (
                      <button key={s.label} onClick={() => saveWm({ scale: s.scale })}
                        className={`rounded border px-3 py-2 text-xs ${Math.abs(wm.scale - s.scale) < 0.01 ? "border-[var(--color-accent)] bg-[#2a1c19] text-[var(--color-ink)]" : "border-[var(--color-line)] bg-[var(--color-panel)] text-[var(--color-muted)]"}`}>{s.label}</button>
                    ))}
                  </div>
                </Field>
              </div>
              <p className="text-[10px] text-[var(--color-muted)]">Changes apply when you next pick a candidate (re-click "use this" to re-stamp).</p>
            </div>
          )}
        </div>
      )}

      <SectionTitle>Living cover · animate the picked art</SectionTitle>
      {!pickedSrc ? (
        <p className="text-[11px] text-[var(--color-muted)]">Pick a cover candidate first — the un-stamped art gets animated into a seamless 10s loop (H3, same frame pinned at both ends).</p>
      ) : (
        <div className="space-y-2 rounded-lg border border-[var(--color-line)] bg-[var(--color-panel2)] p-2.5">
          <div className="flex items-center gap-2">
            <GhostButton onClick={lcDraftPrompt} disabled={lcBusy !== ""}>
              {lcBusy === "prompt" ? "drafting…" : "✨ Draft motion prompt"}
            </GhostButton>
            <GhostButton onClick={lcAnimate} disabled={lcBusy !== "" || !lcPrompt.trim()}>
              {lcBusy === "animate" ? "submitting…" : "▶ Animate (2 takes, box GPU)"}
            </GhostButton>
            {lcPicked && (
              <GhostButton onClick={lcUpscale} disabled={lcBusy !== ""} title="FlashVSR 2x on the picked take — do this once you're happy with a take">
                {lcBusy === "upscale" ? "upscaling…" : lcFinal ? "✓ upscaled" : "⬆ Upscale pick"}
              </GhostButton>
            )}
          </div>
          {lcPrompt && (
            <textarea className={inp + " h-28 w-full font-mono text-[10px]"} value={lcPrompt}
              onChange={(e) => setLcPrompt(e.target.value)} />
          )}
          {lcTakes.length > 0 && (
            <div className="grid grid-cols-2 gap-1.5">
              {lcTakes.map((t) => (
                <div key={t.jobId} className={`overflow-hidden rounded border ${lcPicked === t.jobId ? "border-[var(--color-accent2)] ring-1 ring-[var(--color-accent2)]" : "border-[var(--color-line)]"}`}>
                  {t.url ? (
                    <video src={t.url} muted loop autoPlay playsInline className="w-full" />
                  ) : (
                    <div className="flex h-24 items-center justify-center text-[11px] text-[var(--color-muted)]">
                      {t.err || `rendering… ${t.pct || 0}%`}
                    </div>
                  )}
                  {t.url && (
                    <button onClick={() => { setLcPicked(t.jobId); setLcFinal(""); }}
                      className="w-full bg-[var(--color-panel)] py-1 text-[11px] text-[var(--color-muted)] hover:text-[var(--color-ink)]">
                      {lcPicked === t.jobId ? "✓ using this take" : "use this take"}
                    </button>
                  )}
                </div>
              ))}
            </div>
          )}
          <p className="text-[10px] text-[var(--color-muted)]">Watch each take loop — reject any where background figures visibly speed up to reach the wrap point (per-seed; re-Animate for fresh seeds).</p>
        </div>
      )}

      <SectionTitle>3 · Make the video</SectionTitle>
      {pickedUrl && (
        <img src={pickedUrl} alt="" onClick={() => openLightbox(pickedUrl)} title="the chosen cover — click to enlarge"
          className="max-h-40 cursor-zoom-in rounded-lg border border-[var(--color-line)]" />
      )}
      <Field label="Output" hint="1080p is YouTube's sweet spot; 4K upscales the art (lanczos) for the higher-res player codec">
        <select className={inp} value={res} onChange={(e) => setRes(e.target.value)}>
          <option value="1080p">1920 x 1080 (native)</option>
          <option value="4k">3840 x 2160 (upscaled)</option>
        </select>
      </Field>
      <Field label="Video style" hint="card = the layout you picked from the mocks: art in a card over an animated background, band + title beside it">
        <select className={inp} value={style} onChange={(e) => setStyle(e.target.value)}>
          <option value="still">Still cover (classic)</option>
          <option value="fullbleed">Living cover — full-bleed loop</option>
          <option value="card">Card visualizer — loop + background + lettering</option>
        </select>
      </Field>
      {style === "card" && (
        <div className="space-y-2 rounded-lg border border-[var(--color-line)] bg-[var(--color-panel2)] p-2.5">
          <Field label="Background loop">
            <select className={inp} value={bgId} onChange={(e) => setBgId(e.target.value)}>
              <option value="">choose…</option>
              {bgs.map((b) => <option key={b.id} value={b.id}>{b.label}</option>)}
            </select>
          </Field>
          <div className="flex items-end gap-2">
            <Field label="New background" hint="describe slow dark ambient motion (embers, mist, rain…) — generated on the box, looped, saved by name">
              <input className={inp} value={bgPrompt} onChange={(e) => setBgPrompt(e.target.value)}
                placeholder="slow drifting blue-grey mist over black…" />
            </Field>
            <Field label="Name">
              <input className={inp + " w-36"} value={bgLabel} onChange={(e) => setBgLabel(e.target.value)} placeholder="mist" />
            </Field>
            <GhostButton onClick={bgGenerate} disabled={bgBusy}>{bgBusy ? "generating…" : "＋ Generate"}</GhostButton>
          </div>
          <p className="text-[10px] text-[var(--color-muted)]">The card shows the living cover when one is picked, otherwise the still art.</p>
        </div>
      )}
      {style === "still" && (
        <Field label="Audio visualiser" hint="drawn from the actual samples (ffmpeg); animates the video, so the render takes minutes instead of seconds">
          <select className={inp} value={vizStyle} onChange={(e) => setVizStyle(e.target.value)}>
            <option value="">off — static cover (fast)</option>
            <option value="waves">Waveform</option>
            <option value="bars">Spectrum bars</option>
            <option value="spectro">Scrolling spectrogram</option>
          </select>
        </Field>
      )}
      {vizStyle && (
        <Field label="Visualiser placement" hint="same 12 slots as the lettering — keep it off the face and the title">
          <div className="flex items-center gap-2">
            <PlacementGrid value={vizPos} onPick={setVizPos} />
            <select className={inp} value={vizScale} onChange={(e) => setVizScale(e.target.value)}>
              {VIZ_SIZES.map((s) => <option key={s.label} value={String(s.scale)}>size: {s.label}</option>)}
            </select>
          </div>
        </Field>
      )}
      {err && <p className="text-xs text-red-400">{err}</p>}
      <PrimaryButton onClick={render} disabled={rendering || busy}
        title={(style !== "still" || vizStyle) ? "GPU-free ffmpeg on the Mac — an animated encode takes a few minutes" : "GPU-free ffmpeg on the Mac — a few seconds"}>
        {rendering ? "Rendering…" : "Create YouTube video"}
      </PrimaryButton>

      <SectionTitle>4 · Upload text</SectionTitle>
      <GhostButton onClick={writeMeta} disabled={metaBusy} title="Claude writes the title line, description (keywords + hashtags) and tags from the song">
        {metaBusy ? "Writing…" : meta ? "↻ Rewrite upload text" : "✦ Write upload text"}
      </GhostButton>
      {meta && (
        <div className="space-y-2">
          {([["Video title", meta.video_title], ["Description", meta.description], ["Tags (comma-separated)", meta.tags.join(", ")]] as const).map(([label, value]) => (
            <div key={label} className="rounded-lg border border-[var(--color-line)] bg-[var(--color-panel2)] p-2">
              <div className="mb-1 flex items-center gap-2">
                <span className="text-[10px] font-medium uppercase tracking-wider text-[var(--color-muted)]">{label}</span>
                <button onClick={() => navigator.clipboard?.writeText(value)}
                  className="ml-auto rounded border border-[var(--color-line)] px-1.5 py-0.5 text-[10px] text-[var(--color-muted)] hover:text-[var(--color-ink)]">copy</button>
              </div>
              <pre className="whitespace-pre-wrap break-words text-[11px] leading-snug text-[var(--color-ink)]">{value}</pre>
            </div>
          ))}
          <p className="text-[10px] text-[var(--color-muted)]">
            Paste the description block straight into YouTube's Description field (the hashtags on its last line are all you need — the first three show above the title).
            Tags go in the Tags box under "Show more". Remember to tick the altered/synthetic content disclosure.
          </p>
        </div>
      )}
      <p className="text-[11px] text-[var(--color-muted)]">
        The finished video lands in the Library under <b>YouTube videos</b> — its ⬇ button downloads the
        upload-ready MP4 named after the song.
      </p>
    </div>
  );
}
