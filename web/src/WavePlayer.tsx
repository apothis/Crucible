import { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
// @ts-ignore - plugins ship ESM without bundled d.ts in some setups
import TimelinePlugin from "wavesurfer.js/dist/plugins/timeline.esm.js";
// @ts-ignore
import HoverPlugin from "wavesurfer.js/dist/plugins/hover.esm.js";

const fmt = (t: number) => {
  if (!isFinite(t) || t < 0) t = 0;
  const m = Math.floor(t / 60), s = Math.floor(t % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
};

export function WavePlayer({ url, height = 96 }: { url: string; height?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const ws = useRef<WaveSurfer | null>(null);
  const [playing, setPlaying] = useState(false);
  const [ready, setReady] = useState(false);
  const [cur, setCur] = useState(0);
  const [dur, setDur] = useState(0);

  useEffect(() => {
    if (!ref.current) return;
    // Play through a real <audio> MediaElement (reliable) rather than wavesurfer's
    // default WebAudio backend, whose AudioContext can stay suspended → silent playback.
    const audio = new Audio();
    audio.crossOrigin = "anonymous"; audio.preload = "auto"; audio.src = url;
    const plugins: any[] = [TimelinePlugin.create({ height: 16, insertPosition: "beforebegin", style: { fontSize: "10px", color: "#8b909a" } })];
    try { plugins.push(HoverPlugin.create({ lineColor: "#f0911d", lineWidth: 1, labelBackground: "#11141a", labelColor: "#e7e9ee", labelSize: "10px", formatTimeCallback: fmt })); } catch { /* hover optional */ }
    const w = WaveSurfer.create({
      container: ref.current, media: audio, height,
      waveColor: "#454b59", progressColor: "#e0512f", cursorColor: "#f7b733",
      cursorWidth: 2, barWidth: 2, barGap: 1, barRadius: 2, plugins,
    });
    w.on("ready", () => { setReady(true); setDur(w.getDuration()); });
    w.on("timeupdate", (t: number) => setCur(t));
    w.on("play", () => setPlaying(true));
    w.on("pause", () => setPlaying(false));
    w.on("finish", () => setPlaying(false));
    ws.current = w;
    return () => { w.destroy(); ws.current = null; };
  }, [url]);

  return (
    <div className="rounded-lg bg-[#0e1015] p-2">
      <div ref={ref} className="min-w-0" />
      <div className="mt-1.5 flex items-center gap-3">
        <button
          onClick={() => ws.current?.playPause()}
          disabled={!ready}
          className="flex h-9 w-9 flex-none items-center justify-center rounded-full bg-gradient-to-br from-[var(--color-accent)] to-[var(--color-accent2)] text-white shadow disabled:opacity-40"
        >
          {playing ? "❚❚" : "▶"}
        </button>
        <span className="flex-none tabular-nums text-[11px] text-[var(--color-muted)]">{fmt(cur)} / {fmt(dur)}</span>
        <div className="flex-1" />
        <a href={url} download className="flex-none text-[11px] text-[var(--color-muted)] hover:text-[var(--color-ink)]" title="Download">↓</a>
      </div>
    </div>
  );
}
