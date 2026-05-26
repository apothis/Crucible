import { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
// @ts-ignore - plugin ships ESM without bundled d.ts in some setups
import RegionsPlugin from "wavesurfer.js/dist/plugins/regions.esm.js";

/**
 * Waveform with a single draggable / resizable / movable selection region.
 * - drag on empty area  → mark a new selection (replaces the old one)
 * - drag the body       → move it;  drag an edge → resize/expand it
 * - beat markers (from props) are drawn; edges snap to the nearest beat when "snap" is on
 * Reports the selection [start,end] (seconds) via onChange.
 */
export function RegionSelector({ url, beats, start, end, onChange, height = 84 }: {
  url: string; beats: number[]; start: number; end: number;
  onChange: (s: number, e: number) => void; height?: number;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const ws = useRef<WaveSurfer | null>(null);
  const region = useRef<any>(null);
  const snapRef = useRef(true);
  const beatsRef = useRef<number[]>(beats);
  const guard = useRef(false);
  const [snap, setSnap] = useState(true);
  const [dur, setDur] = useState(0);
  const [sel, setSel] = useState<[number, number]>([start || 0, end || 0]);
  const [playing, setPlaying] = useState(false);
  const [ready, setReady] = useState(false);

  const round = (t: number) => Math.round(t * 100) / 100;
  useEffect(() => { snapRef.current = snap; }, [snap]);
  useEffect(() => { beatsRef.current = beats; }, [beats]);

  const nearestBeat = (t: number) => {
    const bs = beatsRef.current;
    if (!snapRef.current || !bs.length) return t;
    let best = bs[0], bd = Math.abs(bs[0] - t);
    for (const b of bs) { const d = Math.abs(b - t); if (d < bd) { bd = d; best = b; } }
    return best;
  };

  useEffect(() => {
    if (!ref.current) return;
    const audio = new Audio();
    audio.crossOrigin = "anonymous"; audio.preload = "auto"; audio.src = url;
    const reg = RegionsPlugin.create();
    const w = WaveSurfer.create({
      container: ref.current, media: audio, height,
      waveColor: "#3a3f4d", progressColor: "#e0512f", cursorColor: "#f0911d",
      barWidth: 2, barGap: 1, barRadius: 2, plugins: [reg],
    });
    ws.current = w;
    w.on("play", () => setPlaying(true));
    w.on("pause", () => setPlaying(false));
    w.on("finish", () => setPlaying(false));

    const apply = (r: any) => {
      if (guard.current) return;
      let s = r.start, e = r.end;
      const ss = nearestBeat(s), ee = nearestBeat(e);
      if ((ss !== s || ee !== e) && ee > ss) { guard.current = true; r.setOptions({ start: ss, end: ee }); guard.current = false; s = ss; e = ee; }
      setSel([s, e]); onChange(round(s), round(e));
    };

    w.on("ready", () => {
      const d = w.getDuration(); setDur(d); setReady(true);
      const s0 = Math.max(0, Math.min(start || 0, d));
      const e0 = (end && end > s0) ? Math.min(end, d) : Math.min(d, s0 + (d > 20 ? 10 : d * 0.3));
      const r = reg.addRegion({ start: s0, end: e0, color: "rgba(224,81,47,0.22)", drag: true, resize: true });
      region.current = r;
      setSel([r.start, r.end]); onChange(round(r.start), round(r.end));
    });
    reg.enableDragSelection({ color: "rgba(224,81,47,0.22)" });
    reg.on("region-created", (r: any) => {
      if (region.current && r !== region.current) region.current.remove();
      region.current = r;
    });
    reg.on("region-updated", apply);
    return () => { w.destroy(); ws.current = null; region.current = null; };
  }, [url]);

  const playSel = () => {
    const r = region.current, w = ws.current;
    if (!r || !w) return;
    w.setTime(r.start); w.play();
    const ms = Math.max(50, (r.end - r.start) * 1000);
    window.setTimeout(() => { try { w.pause(); } catch { /* */ } }, ms);
  };

  const btn = "flex h-7 w-7 items-center justify-center rounded-full bg-gradient-to-br from-[var(--color-accent)] to-[var(--color-accent2)] text-white text-[11px] shadow disabled:opacity-40";
  return (
    <div className="space-y-1.5">
      <div className="relative">
        <div ref={ref} />
        {dur > 0 && beats.map((t, i) => (
          <div key={i} style={{ position: "absolute", top: 0, bottom: 0, left: `${(t / dur) * 100}%`, width: 1, background: "rgba(240,145,29,0.30)", pointerEvents: "none" }} />
        ))}
      </div>
      <div className="flex items-center gap-2 text-[11px] text-[var(--color-muted)]">
        <button onClick={() => ws.current?.playPause()} disabled={!ready} className={btn}>{playing ? "❚❚" : "▶"}</button>
        <button onClick={playSel} disabled={!ready} className={btn} title="Play selection">⏵|</button>
        <span>start <b>{sel[0].toFixed(2)}</b>s · end <b>{sel[1].toFixed(2)}</b>s · length <b className="text-[var(--color-accent2)]">{Math.max(0, sel[1] - sel[0]).toFixed(2)}</b>s</span>
        <label className="ml-auto flex items-center gap-1 whitespace-nowrap">
          <input type="checkbox" checked={snap} onChange={(e) => setSnap(e.target.checked)} /> snap to beat{beats.length ? ` (${beats.length})` : ""}
        </label>
      </div>
    </div>
  );
}
