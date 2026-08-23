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

type Concept = { name: string; overview: string; background: string; aesthetics: string; lighting: string; palette: string[]; title_style: string };
type Cand = { jobId: string; seed: number; url?: string; err?: string; pct?: number };

const EMPTY_CONCEPT: Concept = { name: "", overview: "", background: "", aesthetics: "", lighting: "", palette: [], title_style: "" };

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
  const [artist, setArtist] = d.use("artist", "Apotheon");   // the band (chosen 2026-08-23)
  const [audioId, setAudioId] = d.use("audioId", "");
  const [notes, setNotes] = d.use("notes", "");
  const [concepts, setConcepts] = d.use<Concept[]>("concepts", []);
  const [concept, setConcept] = d.use<Concept>("concept", EMPTY_CONCEPT);
  const [count, setCount] = d.use("count", 4);
  const [cands, setCands] = d.use<Cand[]>("cands", []);
  const [picked, setPicked] = d.use("picked", "");
  const [res, setRes] = d.use("res", "1080p");
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

  const audios = library.filter((i) => i.audio_url);
  const pickedCand = cands.find((c) => c.jobId === picked);

  const setC = (k: keyof Concept, v: string | string[]) => setConcept((p) => ({ ...p, [k]: v } as Concept));

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
        const r = await api.ytCover({ title, artist: artist || undefined, concept, seed: base + i * 101 }) as { job_id: string; seed: number };
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

  async function render() {
    if (!picked) { setErr("Pick a cover candidate first."); return; }
    if (!audioId) { setErr("Pick the song track to use."); return; }
    setErr(""); setRendering(true);
    const card = { id: rid(), title: `YouTube video · ${title || "untitled"}`, status: "running" as const, pct: 40 };
    ctx.setResults([card]);
    try {
      const r = await api.ytRender({ image_id: picked, audio_id: audioId, title, res }) as { media_url: string };
      ctx.patch(card.id, { status: "done", pct: 100, url: r.media_url + "?t=" + Date.now(), media: "video" });
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
      <div className="grid grid-cols-2 gap-3">
        <Field label="Song title" hint="rendered INTO the artwork">
          <input className={inp} value={title} onChange={(e) => setTitle(e.target.value)} placeholder="e.g. Ashes of the Dawn" />
        </Field>
        <Field label="Artist / band" hint="optional second line">
          <input className={inp} value={artist} onChange={(e) => setArtist(e.target.value)} placeholder="(none)" />
        </Field>
      </div>
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
      <Field label="Title lettering" hint="typography / material / color of the title text">
        <input className={inp} value={concept.title_style} onChange={(e) => setC("title_style", e.target.value)}
          placeholder="e.g. carved bone-white gothic serif lettering with charred edges" />
      </Field>
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
            <button onClick={() => { setCands([]); setPicked(""); }} className="text-[11px] leading-none text-[var(--color-muted)] hover:text-red-400" title="clear candidates">×</button>
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
                  <button onClick={() => setPicked(c.jobId)}
                    className={`w-full py-0.5 text-[10px] ${picked === c.jobId ? "bg-[var(--color-accent2)] text-black" : "text-[var(--color-muted)] hover:text-[var(--color-ink)]"}`}>
                    {picked === c.jobId ? "✓ this is the cover" : "use this"}
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      <SectionTitle>3 · Make the video</SectionTitle>
      {pickedCand?.url && (
        <img src={pickedCand.url} alt="" onClick={() => openLightbox(pickedCand.url!)} title="the chosen cover — click to enlarge"
          className="max-h-40 cursor-zoom-in rounded-lg border border-[var(--color-line)]" />
      )}
      <Field label="Output" hint="1080p is YouTube's sweet spot; 4K upscales the art (lanczos) for the higher-res player codec">
        <select className={inp} value={res} onChange={(e) => setRes(e.target.value)}>
          <option value="1080p">1920 x 1080 (native)</option>
          <option value="4k">3840 x 2160 (upscaled)</option>
        </select>
      </Field>
      {err && <p className="text-xs text-red-400">{err}</p>}
      <PrimaryButton onClick={render} disabled={rendering || busy} title="GPU-free ffmpeg on the Mac — a few seconds">
        {rendering ? "Rendering…" : "Create YouTube video"}
      </PrimaryButton>
      <p className="text-[11px] text-[var(--color-muted)]">
        The finished video lands in the Library under <b>YouTube videos</b> — its ⬇ button downloads the
        upload-ready MP4 named after the song.
      </p>
    </div>
  );
}
