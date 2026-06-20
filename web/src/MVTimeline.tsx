import { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
// @ts-ignore - plugin ships ESM without bundled d.ts in some setups
import RegionsPlugin from "wavesurfer.js/dist/plugins/regions.esm.js";
// @ts-ignore
import TimelinePlugin from "wavesurfer.js/dist/plugins/timeline.esm.js";
import type { Block } from "./mvmodel";

const fmt = (t: number) => {
  if (!isFinite(t) || t < 0) t = 0;
  const m = Math.floor(t / 60), s = Math.floor(t % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
};

/**
 * The visual music-video timeline: the song waveform with one draggable / resizable
 * region per MSR block. Drag a block to move it in time, drag an edge to resize, click
 * to select (binds the inspector). Edges snap to the nearest beat when "snap" is on.
 * Reports a block's new [start,end] via onChange; the parent keeps blocks sorted by time.
 */
export function MVTimeline({ url, beats, blocks, selId, onSelect, onChange, height = 96 }: {
  url: string; beats: number[]; blocks: Block[]; selId: string;
  onSelect: (id: string) => void; onChange: (id: string, start: number, end: number) => void;
  height?: number;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const ws = useRef<WaveSurfer | null>(null);
  const regPlugin = useRef<any>(null);
  const regions = useRef<Map<string, any>>(new Map());
  const guard = useRef(false);
  const snapRef = useRef(true);
  const beatsRef = useRef<number[]>(beats);
  const blocksRef = useRef<Block[]>(blocks);
  const selRef = useRef(selId);
  const cbRef = useRef({ onSelect, onChange });
  const [ready, setReady] = useState(false);
  const [snap, setSnap] = useState(true);
  const [dur, setDur] = useState(0);
  const [cur, setCur] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [zoom, setZoom] = useState(14);   // px/sec - wide enough to read blocks; the waveform scrolls horizontally

  useEffect(() => { snapRef.current = snap; }, [snap]);
  useEffect(() => { beatsRef.current = beats; }, [beats]);
  useEffect(() => { blocksRef.current = blocks; }, [blocks]);
  useEffect(() => { selRef.current = selId; }, [selId]);
  useEffect(() => { cbRef.current = { onSelect, onChange }; });

  const round = (t: number) => Math.round(t * 100) / 100;
  const nearestBeat = (t: number) => {
    const bs = beatsRef.current;
    if (!snapRef.current || !bs.length) return t;
    let best = bs[0], bd = Math.abs(bs[0] - t);
    for (const b of bs) { const d = Math.abs(b - t); if (d < bd) { bd = d; best = b; } }
    return best;
  };
  const colorFor = (b: Block, selected: boolean) =>
    selected ? "rgba(224,81,47,0.42)" : b.clipId ? "rgba(34,160,80,0.30)"
      : b.lipsync ? "rgba(240,145,29,0.24)" : "rgba(120,130,150,0.20)";

  function syncRegion(r: any, b: Block) {
    // start/end (skip when unchanged so an active drag isn't disrupted)
    if (Math.abs(r.start - b.start) > 0.002 || Math.abs(r.end - b.end) > 0.002) {
      guard.current = true; r.setOptions({ start: b.start, end: b.end }); guard.current = false;
    }
    const col = colorFor(b, b.id === selRef.current);
    if (r._mvColor !== col) { r._mvColor = col; r.setOptions({ color: col }); }
    const label = `${b.idx + 1}${b.lipsync ? " ♪" : ""}`;
    if (r._mvLabel !== label) { r._mvLabel = label; try { r.setOptions({ content: label }); } catch { /* */ } }
  }

  // (re)build all regions from the current blocks (on ready / url change)
  function buildAll() {
    const reg = regPlugin.current; if (!reg) return;
    regions.current.forEach((r) => r.remove()); regions.current.clear();
    for (const b of blocksRef.current) {
      const r = reg.addRegion({ id: b.id, start: b.start, end: b.end, drag: true, resize: true,
        color: colorFor(b, b.id === selRef.current), content: `${b.idx + 1}${b.lipsync ? " ♪" : ""}` });
      r._mvColor = colorFor(b, b.id === selRef.current); r._mvLabel = `${b.idx + 1}${b.lipsync ? " ♪" : ""}`;
      regions.current.set(b.id, r);
    }
  }

  // init wavesurfer for the song
  useEffect(() => {
    if (!ref.current || !url) return;
    const audio = new Audio(); audio.crossOrigin = "anonymous"; audio.preload = "auto"; audio.src = url;
    const reg = RegionsPlugin.create();
    regPlugin.current = reg;
    const w = WaveSurfer.create({
      container: ref.current, media: audio, height,
      minPxPerSec: zoom, autoScroll: true, hideScrollbar: false,
      waveColor: "#3a404e", progressColor: "#7a4030", cursorColor: "#f7b733",
      cursorWidth: 2, barWidth: 2, barGap: 1, barRadius: 2,
      plugins: [reg, TimelinePlugin.create({ height: 14, insertPosition: "beforebegin", style: { fontSize: "9px", color: "#8b909a" } })],
    });
    ws.current = w;
    setReady(false);
    w.on("play", () => setPlaying(true));
    w.on("pause", () => setPlaying(false));
    w.on("finish", () => setPlaying(false));
    w.on("timeupdate", (t: number) => setCur(t));
    w.on("ready", () => { setDur(w.getDuration()); setReady(true); buildAll(); });
    reg.on("region-updated", (r: any) => {
      if (guard.current) return;
      let s = nearestBeat(r.start), e = nearestBeat(r.end);
      if (e <= s) e = s + 0.1;
      if (s !== r.start || e !== r.end) { guard.current = true; r.setOptions({ start: s, end: e }); guard.current = false; }
      cbRef.current.onChange(r.id, round(s), round(e));
    });
    reg.on("region-clicked", (r: any, e: any) => { e.stopPropagation(); cbRef.current.onSelect(r.id); });
    return () => { w.destroy(); ws.current = null; regPlugin.current = null; regions.current.clear(); };
  }, [url]);

  // reconcile regions <- blocks / selection
  useEffect(() => {
    if (!ready) return;
    const reg = regPlugin.current; const map = regions.current;
    const ids = new Set(blocks.map((b) => b.id));
    for (const [id, r] of map) { if (!ids.has(id)) { r.remove(); map.delete(id); } }
    for (const b of blocks) {
      let r = map.get(b.id);
      if (!r) {
        r = reg.addRegion({ id: b.id, start: b.start, end: b.end, drag: true, resize: true,
          color: colorFor(b, b.id === selId), content: `${b.idx + 1}` });
        map.set(b.id, r);
      }
      syncRegion(r, b);
    }
  }, [blocks, selId, ready]);

  // apply zoom (px/sec) when it changes - widens the waveform; the container scrolls horizontally
  useEffect(() => { if (ready && ws.current) try { ws.current.zoom(zoom); } catch { /* */ } }, [zoom, ready]);

  const pbtn = "flex h-7 w-7 flex-none items-center justify-center rounded-full bg-gradient-to-br from-[var(--color-accent)] to-[var(--color-accent2)] text-white text-[11px] shadow disabled:opacity-40";
  return (
    <div className="space-y-1.5 rounded-lg border border-[var(--color-line)] bg-[#0e1015] p-2">
      <div ref={ref} />
      <div className="flex items-center gap-2 text-[10px] text-[var(--color-muted)]">
        <button onClick={() => ws.current?.playPause()} disabled={!ready} className={pbtn}>{playing ? "❚❚" : "▶"}</button>
        <span className="tabular-nums">{fmt(cur)} / {fmt(dur)}</span>
        <span className="opacity-70">drag a block to move {"·"} drag an edge to resize {"·"} click to edit</span>
        <label className="ml-auto flex items-center gap-1 whitespace-nowrap" title="timeline zoom (px / second)">
          zoom
          <input type="range" min={4} max={60} step={1} value={zoom} onChange={(e) => setZoom(Number(e.target.value))} className="w-24" />
        </label>
        <label className="flex items-center gap-1 whitespace-nowrap">
          <input type="checkbox" checked={snap} onChange={(e) => setSnap(e.target.checked)} /> snap to beat{beats.length ? ` (${beats.length})` : ""}
        </label>
      </div>
    </div>
  );
}
