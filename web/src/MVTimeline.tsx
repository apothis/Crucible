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
 *
 * NLE controls (when the mouse is over the timeline and you're not typing in a field):
 *   Space play/pause · I/O set selected block's in/out to the playhead · L loop the selected
 *   shot · S toggle snap · Delete remove selected · Ctrl/Cmd+D duplicate · ←/→ nudge playhead.
 */
export function MVTimeline({ url, beats, blocks, selId, fps = 24, onSelect, onChange, onDelete, onDuplicate, height = 96 }: {
  url: string; beats: number[]; blocks: Block[]; selId: string; fps?: number;
  onSelect: (id: string) => void; onChange: (id: string, start: number, end: number) => void;
  onDelete?: (id: string) => void; onDuplicate?: (id: string) => void;
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
  const cbRef = useRef({ onSelect, onChange, onDelete, onDuplicate });
  const loopRef = useRef(false);
  const [ready, setReady] = useState(false);
  const [snap, setSnap] = useState(true);
  const [dur, setDur] = useState(0);
  const [cur, setCur] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [loop, setLoop] = useState(false);
  const [units, setUnits] = useState<"s" | "f">("s");        // selected-block readout units
  const [zoom, setZoom] = useState(14);   // px/sec - wide enough to read blocks; the waveform scrolls horizontally

  useEffect(() => { snapRef.current = snap; }, [snap]);
  useEffect(() => { beatsRef.current = beats; }, [beats]);
  useEffect(() => { blocksRef.current = blocks; }, [blocks]);
  useEffect(() => { selRef.current = selId; }, [selId]);
  useEffect(() => { cbRef.current = { onSelect, onChange, onDelete, onDuplicate }; });
  useEffect(() => { loopRef.current = loop; }, [loop]);

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

  const selBlock = () => blocksRef.current.find((b) => b.id === selRef.current);
  const seek = (t: number) => { const w = ws.current; if (w && w.getDuration()) w.setTime(Math.max(0, Math.min(t, w.getDuration()))); };

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
    w.on("timeupdate", (t: number) => {
      setCur(t);
      // loop the selected shot: when the playhead passes its end, jump back to its start
      if (loopRef.current) {
        const b = selBlock();
        if (b && t >= b.end - 0.02) { try { w.setTime(b.start); } catch { /* */ } }
      }
    });
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

  // ---- in/out + transport helpers ----
  const setIn = () => { const b = selBlock(); if (b) { const s = round(cur); cbRef.current.onChange(b.id, s, Math.max(s + 0.1, b.end)); } };
  const setOut = () => { const b = selBlock(); if (b) { const e = round(cur); cbRef.current.onChange(b.id, Math.min(b.start, e - 0.1), e); } };
  const zoomToFit = () => {
    const wEl = ref.current?.clientWidth || 0; const d = ws.current?.getDuration() || dur;
    if (wEl > 40 && d > 0) setZoom(Math.max(2, Math.min(60, (wEl - 24) / d)));
  };

  // ---- NLE keybindings (only when hovering the timeline + not typing in a field) ----
  const hoverRef = useRef(false);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!hoverRef.current) return;
      const t = e.target as HTMLElement;
      if (t && (/INPUT|TEXTAREA|SELECT/.test(t.tagName) || t.isContentEditable)) return;
      const w = ws.current; if (!w) return;
      const b = selBlock();
      const k = e.key.toLowerCase();
      if (e.key === " ") { e.preventDefault(); w.playPause(); }
      else if ((e.ctrlKey || e.metaKey) && k === "d") { e.preventDefault(); if (b) cbRef.current.onDuplicate?.(b.id); }
      else if (e.key === "Delete" || e.key === "Backspace") { if (b) { e.preventDefault(); cbRef.current.onDelete?.(b.id); } }
      else if (k === "i") { e.preventDefault(); setIn(); }
      else if (k === "o") { e.preventDefault(); setOut(); }
      else if (k === "l") { e.preventDefault(); setLoop((v) => !v); }
      else if (k === "s") { e.preventDefault(); setSnap((v) => !v); }
      else if (e.key === "ArrowLeft") { e.preventDefault(); seek((w.getCurrentTime?.() ?? cur) - (e.shiftKey ? 1 : 1 / fps)); }
      else if (e.key === "ArrowRight") { e.preventDefault(); seek((w.getCurrentTime?.() ?? cur) + (e.shiftKey ? 1 : 1 / fps)); }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [fps]);

  const sel = blocks.find((b) => b.id === selId);
  const u = (t: number) => units === "f" ? `${Math.round(t * fps)}f` : `${t.toFixed(2)}s`;
  const pbtn = "flex h-7 w-7 flex-none items-center justify-center rounded-full bg-gradient-to-br from-[var(--color-accent)] to-[var(--color-accent2)] text-white text-[11px] shadow disabled:opacity-40";
  const tbtn = "rounded border border-[var(--color-line)] px-1.5 py-0.5 text-[10px] text-[var(--color-muted)] hover:text-[var(--color-ink)] disabled:opacity-40";
  return (
    <div className="space-y-1.5 rounded-lg border border-[var(--color-line)] bg-[#0e1015] p-2"
      onMouseEnter={() => { hoverRef.current = true; }} onMouseLeave={() => { hoverRef.current = false; }}>
      <div ref={ref} />
      <div className="flex flex-wrap items-center gap-2 text-[10px] text-[var(--color-muted)]">
        <button onClick={() => ws.current?.playPause()} disabled={!ready} className={pbtn}>{playing ? "❚❚" : "▶"}</button>
        <span className="tabular-nums">{fmt(cur)} / {fmt(dur)}</span>
        <button onClick={setIn} disabled={!sel} className={tbtn} title="Set selected block IN point to playhead (I)">[ In</button>
        <button onClick={setOut} disabled={!sel} className={tbtn} title="Set selected block OUT point to playhead (O)">Out ]</button>
        <button onClick={() => setLoop((v) => !v)} disabled={!sel} className={`${tbtn} ${loop ? "text-[var(--color-accent2)] border-[var(--color-accent2)]" : ""}`} title="Loop the selected shot (L)">⟲ loop</button>
        {sel && <span className="tabular-nums text-[var(--color-muted)]" title="selected block timing">
          #{sel.idx + 1} · {u(sel.start)}–{u(sel.end)} · len {u(sel.end - sel.start)}
        </span>}
        <span className="ml-auto opacity-60" title="Space play · I/O in-out · L loop · S snap · Del delete · Ctrl+D dup · ←/→ nudge">⌨ shortcuts</span>
        <button onClick={() => setUnits((v) => v === "s" ? "f" : "s")} className={tbtn} title="toggle readout units">{units === "s" ? "sec" : "frames"}</button>
        <button onClick={zoomToFit} disabled={!ready} className={tbtn} title="Zoom to fit the whole timeline">fit</button>
        <label className="flex items-center gap-1 whitespace-nowrap" title="timeline zoom (px / second)">
          zoom
          <input type="range" min={4} max={60} step={1} value={zoom} onChange={(e) => setZoom(Number(e.target.value))} className="w-24" />
        </label>
        <label className="flex items-center gap-1 whitespace-nowrap">
          <input type="checkbox" checked={snap} onChange={(e) => setSnap(e.target.checked)} /> snap{beats.length ? ` (${beats.length})` : ""}
        </label>
      </div>
    </div>
  );
}
