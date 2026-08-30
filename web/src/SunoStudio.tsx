import { useRef, useState } from "react";
import { api } from "./api";
import { useDrafts } from "./drafts";
import { Field, inp, PrimaryButton, GhostButton, SectionTitle, rid, type RunCtx } from "./ui";

// The Suno lane: songs are WRITTEN here (or arrive compiled from the Music 3 tab's
// Export for Suno), generated on suno.com by hand, and the downloads come back through
// the import box below - landing in the library as versioned takes of the song, exactly
// like a local render. docs/SUNO_PROMPTING.md is the rulebook the writer follows.
export function SunoStudioForm({ busy, ...ctx }: { busy: boolean } & RunCtx) {
  const d = useDrafts("suno");
  const [title, setTitle] = d.use("title", "");
  const [style, setStyle] = d.use("style", "");
  const [exclude, setExclude] = d.use("exclude", "");
  const [lyrics, setLyrics] = d.use("lyrics", "");
  const [brief, setBrief] = d.use("brief", "");
  // Solo styles that demonstrably work as Suno tags (docs/SUNO_PROMPTING.md): one style
  // qualifier + "Guitar Solo". "" = let the writer pick from the song.
  const SOLO_STYLES = ["", "Guitar Solo", "Melodic Guitar Solo", "Shred Guitar Solo",
    "Blues Guitar Solo", "Harmonized Guitar Solo", "Emotional Guitar Solo",
    "Neoclassical Guitar Solo"] as const;
  const [soloStyle, setSoloStyle] = d.use("soloStyle", "Melodic Guitar Solo");
  // any existing solo tag in the lyrics follows the selector immediately (a fresh regex
  // per call: a shared /g regex is stateful across test/replace)
  function applySoloStyle(tag: string) {
    setSoloStyle(tag);
    if (tag) setLyrics(lyrics.replace(/^\[(?:solo|(?:\w+ )?(?:\w+ )?guitar solo)\]\s*$/gim, `[${tag}]`));
  }
  const [writing, setWriting] = useState(false);
  const [note, setNote] = useState("");
  const [copied, setCopied] = useState("");
  const [importing, setImporting] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  async function write() {
    setWriting(true); setNote("");
    try {
      const r = await api.sunoWrite({ brief, title, style, exclude, lyrics, solo_style: soloStyle }) as
        { title: string; style: string; exclude: string; lyrics: string };
      if (r.style) setStyle(r.style);
      if (r.exclude) setExclude(r.exclude);
      if (r.title && !title.trim()) setTitle(r.title);
      // same convention as the Music 3 writer: a populated lyrics box is never overwritten
      if (r.lyrics && !lyrics.trim()) { setLyrics(r.lyrics); setNote("lyrics drafted (the box was empty)"); }
      else if (lyrics.trim()) setNote("style/exclude updated - existing lyrics kept untouched");
    } catch (e) {
      ctx.setResults([{ id: rid(), title: "Suno writer failed", status: "error", pct: 0, err: String(e) }]);
    } finally { setWriting(false); }
  }

  async function importFile(f: File) {
    setImporting(true);
    const card = { id: rid(), title: `Importing "${f.name}"…`, status: "running" as const, pct: 50 };
    ctx.setResults([card]);
    try {
      const fd = new FormData();
      fd.append("file", f);
      if (title.trim()) fd.append("title", title.trim());
      if (style.trim()) fd.append("style", style.trim());
      if (exclude.trim()) fd.append("exclude", exclude.trim());
      const r = await api.sunoImport(fd) as { job_id: string; audio_url: string; title: string };
      ctx.patch(card.id, { status: "done", pct: 100, title: `Imported: ${r.title}`, url: r.audio_url });
      ctx.onDone();
    } catch (e) {
      ctx.patch(card.id, { status: "error", pct: 0, err: String(e) });
    } finally { setImporting(false); if (fileRef.current) fileRef.current.value = ""; }
  }

  const copyBtn = (label: string, text: string) => (
    <GhostButton onClick={() => { navigator.clipboard.writeText(text); setCopied(label); }}>
      {copied === label ? "copied ✓" : "copy"}
    </GhostButton>
  );

  return (
    <div className="space-y-4">
      <div className="rounded border border-[var(--color-line)] p-3">
        <SectionTitle>1 {"·"} Write</SectionTitle>
        <p className="mt-1 text-[10px] text-[var(--color-muted)]">
          Describe the song in plain words; the writer builds the Suno Custom Mode prompt by the
          rules in docs/SUNO_PROMPTING.md. A short brief on top of filled fields edits them
          ("more aggressive", "make the chorus a gang chant") rather than starting over. The
          Music 3 tab's Export for Suno also lands here.
        </p>
        <textarea className={inp + " mt-2"} rows={3} value={brief} onChange={(e) => setBrief(e.target.value)}
          placeholder="e.g. a fast symphonic metal anthem about a lighthouse keeper at the end of the world, operatic female lead, huge choir on the last chorus, fast melodic guitar solo" />
        <div className="mt-2 flex items-center gap-2">
          <PrimaryButton onClick={write} disabled={writing || busy}>{writing ? "Writing…" : "Write the prompt"}</PrimaryButton>
          {note && <span className="text-[10px] text-emerald-300">{note}</span>}
        </div>
      </div>

      <div className="rounded border border-[var(--color-line)] p-3">
        <SectionTitle>2 {"·"} The prompt {"·"} paste into suno.com Custom Mode</SectionTitle>
        <div className="mt-2 grid gap-2">
          <Field label="Title">
            <input className={inp} value={title} onChange={(e) => setTitle(e.target.value)} />
          </Field>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-[10px] uppercase tracking-wide text-[var(--color-muted)]">Style of Music</span>
              {copyBtn("style", style)}
              <span className="text-[10px] text-[var(--color-muted)]">{style.length}/1000 {style.length > 700 ? "· over the ~700 sweet spot" : ""}</span>
            </div>
            <textarea className={inp + " mt-1"} rows={3} value={style} onChange={(e) => setStyle(e.target.value)} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-[10px] uppercase tracking-wide text-[var(--color-muted)]">Exclude Styles</span>
              {copyBtn("exclude", exclude)}
            </div>
            <input className={inp + " mt-1"} value={exclude} onChange={(e) => setExclude(e.target.value)} />
          </div>
          <Field label="Guitar solo" hint="rewrites the solo tag in the lyrics and steers the writer; qualified tags are community-proven on Suno">
            <select className={inp} value={soloStyle} onChange={(e) => applySoloStyle(e.target.value)}>
              {SOLO_STYLES.map((t) => (
                <option key={t} value={t}>{t === "" ? "(writer's choice)" : t === "Guitar Solo" ? "Guitar Solo (plain)" : t.replace(" Guitar Solo", "")}</option>
              ))}
            </select>
          </Field>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-[10px] uppercase tracking-wide text-[var(--color-muted)]">Lyrics (Suno tags allowed: [Big Final Chorus], [Guitar Solo]…)</span>
              {copyBtn("lyrics", lyrics)}
            </div>
            <textarea className={inp + " mt-1 min-h-[220px] font-mono text-[11px]"} rows={12}
              value={lyrics} onChange={(e) => setLyrics(e.target.value)} />
          </div>
        </div>
        <div className="mt-2 text-[10px] text-[var(--color-muted)]">
          Sliders on suno.com: Weirdness 40-60%, Style Influence high (70%+) so metal stays metal.
          Pick the winning take BY EAR in their player - downloads are metered, download once.
        </div>
      </div>

      <div className="rounded border border-[var(--color-line)] p-3">
        <SectionTitle>3 {"·"} Bring the download home</SectionTitle>
        <p className="mt-1 text-[10px] text-[var(--color-muted)]">
          Import the downloaded file; it lands in the library under the Title above as a
          versioned take (with the style/exclude prompt saved as its params), usable by every
          downstream tool - MV writers, naturalize, stems, compare.
        </p>
        <div className="mt-2 flex items-center gap-2">
          <input ref={fileRef} type="file" accept=".mp3,.wav,.flac,.m4a,.ogg"
            className="text-[11px] text-[var(--color-muted)]"
            onChange={(e) => { const f = e.target.files?.[0]; if (f) importFile(f); }} disabled={importing} />
          {importing && <span className="text-[10px] text-[var(--color-muted)]">importing…</span>}
        </div>
      </div>
    </div>
  );
}
