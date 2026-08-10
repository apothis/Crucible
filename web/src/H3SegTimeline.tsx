import { useEffect, useRef, useState } from "react";

// ---------------------------------------------------------------------------------------------
// Per-segment audio timeline for the H3 segment editor.
//
// Every boundary in this pipeline is INFERRED: the cut times come from downbeat windows, and who
// sings when comes from the arrangement's section markers mapped onto the real audio. Both are
// close but not exact, and only ears can settle the last second. So this strip plays the segment's
// own audio window, draws which voice the song says is singing under each cut, and lets a cut
// boundary be dragged onto what you actually hear. Releasing a drag recompiles the segment, which
// re-runs voice-matched casting against the new boundaries.
//
// Deliberately not a waveform: decoding a four-minute track per segment strip costs far more than
// it tells you here, where the question is "does the singer change at the right instant".
// ---------------------------------------------------------------------------------------------

export type Cut = { start: number; end: number };
export type VoiceWin = { start: number; end: number; voice: string };

const VOICE_COLOR: Record<string, string> = {
  female: "#c05fd8", male: "#3f8fd0", duet: "#f0911d",
};
const fmt = (t: number) => `${t.toFixed(1)}s`;

export function H3SegTimeline({ url, start, end, cuts, voices, labels, onCutsChange, onCommit }: {
  url?: string;                       // the song's media url (the full track; we play a window of it)
  start: number; end: number;
  cuts: Cut[];
  voices: VoiceWin[];                 // song voice windows, real-audio timeline (any overlap)
  labels: string[];                   // one per cut: who is cast in that cut
  onCutsChange: (cuts: Cut[]) => void;   // live, while dragging
  onCommit: () => void;                  // drag released - recompile the segment
}) {
  const bar = useRef<HTMLDivElement>(null);
  const audio = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [head, setHead] = useState<number | null>(null);
  const [drag, setDrag] = useState<number | null>(null);   // index of the boundary being dragged
  const span = Math.max(0.01, end - start);
  const pct = (t: number) => `${Math.min(100, Math.max(0, ((t - start) / span) * 100))}%`;
  const MIN_CUT = 1.5;                // matches H3_MIN_CUT_S server-side

  // one audio element for the strip; playback is clamped to this segment's window
  useEffect(() => {
    if (!url) return;
    const a = new Audio(url);
    a.preload = "metadata";
    audio.current = a;
    const onTime = () => {
      if (a.currentTime >= end) { a.pause(); a.currentTime = start; setHead(null); setPlaying(false); return; }
      setHead(a.currentTime);
    };
    a.addEventListener("timeupdate", onTime);
    a.addEventListener("play", () => setPlaying(true));
    a.addEventListener("pause", () => setPlaying(false));
    return () => { a.pause(); a.removeEventListener("timeupdate", onTime); audio.current = null; };
  }, [url, start, end]);

  function toggle() {
    const a = audio.current;
    if (!a) return;
    if (a.paused) {
      if (a.currentTime < start || a.currentTime >= end) a.currentTime = start;
      a.play().catch(() => {});
    } else a.pause();
  }
  function seekTo(clientX: number) {
    const b = bar.current, a = audio.current;
    if (!b || !a) return;
    const r = b.getBoundingClientRect();
    a.currentTime = start + ((clientX - r.left) / r.width) * span;
    setHead(a.currentTime);
  }

  // drag a cut boundary. boundary k sits between cuts[k-1] and cuts[k], so 1..cuts.length-1.
  useEffect(() => {
    if (drag === null) return;
    const move = (ev: MouseEvent) => {
      const b = bar.current;
      if (!b) return;
      const r = b.getBoundingClientRect();
      const t = start + ((ev.clientX - r.left) / r.width) * span;
      const lo = cuts[drag - 1].start + MIN_CUT;
      const hi = cuts[drag].end - MIN_CUT;
      const clamped = Math.min(hi, Math.max(lo, t));
      if (!(clamped > lo - 0.001 && clamped < hi + 0.001)) return;   // no room to move here
      const next = cuts.map((c) => ({ ...c }));
      next[drag - 1].end = Math.round(clamped * 100) / 100;
      next[drag].start = next[drag - 1].end;
      onCutsChange(next);
    };
    const up = () => { setDrag(null); onCommit(); };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up, { once: true });
    return () => { window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [drag, cuts, start, span]);

  // the voice the song says is singing across a span (largest overlap), for the cut labels
  const voiceAt = (a: number, b: number) => {
    let best = "", ov = 0;
    for (const w of voices) {
      const o = Math.min(b, w.end) - Math.max(a, w.start);
      if (o > ov) { best = w.voice; ov = o; }
    }
    return best;
  };
  const segVoices = voices.filter((w) => w.end > start && w.start < end);
  // a voice change inside this segment - the thing worth checking by ear
  const handovers = segVoices.map((w) => w.start).filter((t) => t > start + 0.2 && t < end - 0.2);

  return (
    <div className="rounded border border-[var(--color-line)] bg-[#0e1015] p-2">
      <div className="mb-1 flex items-center gap-2">
        <button onClick={toggle} disabled={!url}
          title={url ? "play just this segment's audio" : "pick the song track above to hear this"}
          className="rounded border border-[var(--color-line)] px-2 py-0.5 text-[10px] text-[var(--color-muted)] hover:text-[var(--color-ink)] disabled:opacity-40">
          {playing ? "❚❚ pause" : "▶ play segment"}
        </button>
        <span className="text-[9px] text-[var(--color-muted)]">
          {fmt(start)}–{fmt(end)}
          {head !== null && <span className="text-[var(--color-accent2)]"> {"·"} {fmt(head)}</span>}
        </span>
        <span className="flex-1" />
        {Object.entries(VOICE_COLOR).map(([v, c]) => (
          <span key={v} className="flex items-center gap-1 text-[9px] text-[var(--color-muted)]">
            <i className="inline-block h-2 w-2 rounded-sm" style={{ background: c }} />{v}
          </span>
        ))}
      </div>

      {/* the strip: cut lanes on top, what the SONG says underneath */}
      <div ref={bar} className="relative h-14 w-full cursor-pointer select-none rounded bg-[#171a21]"
        onClick={(e) => { if (drag === null) seekTo(e.clientX); }}>
        {/* cut lanes */}
        {cuts.map((c, k) => (
          <div key={k} className="absolute top-0 h-8 overflow-hidden border-r border-[var(--color-line)] px-1 py-0.5 text-[9px] leading-tight text-[var(--color-ink)]"
            style={{ left: pct(c.start), width: `${((c.end - c.start) / span) * 100}%`,
                     background: k % 2 ? "#1e222b" : "#232833" }}
            title={`cut ${k + 1}: ${fmt(c.start)}–${fmt(c.end)} — ${labels[k] || "no cast"}`}>
            <div className="truncate">{labels[k] || "no cast"}</div>
            <div className="truncate text-[8px] text-[var(--color-muted)]">cut {k + 1} {"·"} {(c.end - c.start).toFixed(1)}s</div>
          </div>
        ))}
        {/* what the song says: the marked voice under each moment */}
        {segVoices.map((w, k) => (
          <div key={k} className="absolute bottom-0 h-5 opacity-80"
            style={{ left: pct(Math.max(w.start, start)),
                     width: `${((Math.min(w.end, end) - Math.max(w.start, start)) / span) * 100}%`,
                     background: VOICE_COLOR[w.voice] || "#555b68" }}
            title={`the song marks this ${w.voice} (${fmt(w.start)}–${fmt(w.end)})`} />
        ))}
        {/* where the voice changes inside this segment */}
        {handovers.map((t, k) => (
          <div key={`h${k}`} className="absolute bottom-0 top-0 w-0.5 bg-white/80" style={{ left: pct(t) }}
            title={`the song's voice changes here (${fmt(t)})`} />
        ))}
        {/* draggable cut boundaries */}
        {cuts.slice(1).map((c, k) => (
          <div key={`b${k}`} onMouseDown={(e) => { e.stopPropagation(); setDrag(k + 1); }}
            title={`drag to move this cut boundary (${fmt(c.start)}) - release to recompile`}
            className={`absolute top-0 h-8 w-2 -translate-x-1/2 cursor-col-resize ${drag === k + 1 ? "bg-[var(--color-accent2)]" : "bg-[var(--color-accent2)]/60 hover:bg-[var(--color-accent2)]"}`}
            style={{ left: pct(c.start) }} />
        ))}
        {/* playhead */}
        {head !== null && head >= start && head <= end && (
          <div className="pointer-events-none absolute top-0 h-full w-px bg-[var(--color-accent2)]" style={{ left: pct(head) }} />
        )}
      </div>

      <div className="mt-1 flex flex-wrap items-center gap-2 text-[9px] text-[var(--color-muted)]">
        <span>drag the amber handles to move a cut; the coloured strip is the voice the song says is singing.</span>
        <span className="text-[var(--color-muted)]">
          {cuts.map((c, k) => `cut ${k + 1}: ${voiceAt(c.start, c.end) || "instrumental"}`).join(" {·} ").replace(/\{·\}/g, "·")}
        </span>
        {handovers.length > 0 && (
          <span className="text-[var(--color-accent2)]">
            voice change at {handovers.map(fmt).join(", ")} - line a cut up with it if it sounds off.
          </span>
        )}
      </div>
    </div>
  );
}
