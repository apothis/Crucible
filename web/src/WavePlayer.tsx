import { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";

export function WavePlayer({ url, height = 52 }: { url: string; height?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const ws = useRef<WaveSurfer | null>(null);
  const [playing, setPlaying] = useState(false);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (!ref.current) return;
    const w = WaveSurfer.create({
      container: ref.current,
      url,
      height,
      waveColor: "#3a3f4d",
      progressColor: "#e0512f",
      cursorColor: "#f0911d",
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
    });
    w.on("ready", () => setReady(true));
    w.on("play", () => setPlaying(true));
    w.on("pause", () => setPlaying(false));
    w.on("finish", () => setPlaying(false));
    ws.current = w;
    return () => { w.destroy(); ws.current = null; };
  }, [url]);

  return (
    <div className="flex items-center gap-3">
      <button
        onClick={() => ws.current?.playPause()}
        disabled={!ready}
        className="flex h-9 w-9 flex-none items-center justify-center rounded-full bg-gradient-to-br from-[var(--color-accent)] to-[var(--color-accent2)] text-white shadow disabled:opacity-40"
      >
        {playing ? "❚❚" : "▶"}
      </button>
      <div ref={ref} className="min-w-0 flex-1" />
      <a href={url} download className="flex-none text-[11px] text-[var(--color-muted)] hover:text-[var(--color-ink)]">↓</a>
    </div>
  );
}
