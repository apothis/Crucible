import { useEffect, useMemo, useRef, useState } from "react";

// Synchronous A/B(/C...) player: every track is loaded into its own <audio>, they all play in
// lock-step, and only the ACTIVE one is audible. Switching = swap which one is unmuted, so the
// change is instant and the position never moves. Keys: 1-9 pick a track, space play/pause,
// left/right seek 5 s. Blind mode shuffles the slots and hides the labels until revealed.

const fmt = (t: number) => {
  if (!isFinite(t) || t < 0) t = 0;
  const m = Math.floor(t / 60), s = Math.floor(t % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
};

export type ABTrack = { id: string; label: string; url: string };

export function ABPlayer({ tracks }: { tracks: ABTrack[] }) {
  const els = useRef<Map<string, HTMLAudioElement>>(new Map());
  const [active, setActive] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [cur, setCur] = useState(0);
  const [dur, setDur] = useState(0);
  const [blind, setBlind] = useState(false);
  const [revealed, setRevealed] = useState(false);
  const [seed, setSeed] = useState(0);
  const [ready, setReady] = useState(0);

  // Slot order: identity normally; a seeded shuffle in blind mode.
  const order = useMemo(() => {
    const idx = tracks.map((_, i) => i);
    if (!blind) return idx;
    let s = seed * 9301 + 49297;
    for (let i = idx.length - 1; i > 0; i--) { s = (s * 9301 + 49297) % 233280; const j = Math.floor((s / 233280) * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    return idx;
  }, [tracks, blind, seed]);

  // Build/destroy audio elements when the track list changes.
  useEffect(() => {
    const map = els.current;
    for (const t of tracks) {
      if (map.has(t.id)) continue;
      const a = new Audio(); a.preload = "auto"; a.src = t.url; a.volume = 0;
      a.addEventListener("loadedmetadata", () => { setDur((d) => Math.max(d, a.duration || 0)); setReady((r) => r + 1); });
      map.set(t.id, a);
    }
    for (const [id, a] of Array.from(map.entries())) {
      if (!tracks.some((t) => t.id === id)) { a.pause(); a.src = ""; map.delete(id); }
    }
    setActive((i) => Math.min(i, Math.max(0, tracks.length - 1)));
    return undefined;
  }, [tracks]);

  // Unmount: stop everything.
  useEffect(() => () => { for (const a of els.current.values()) { a.pause(); a.src = ""; } els.current.clear(); }, []);

  const audible = tracks[order[active]];
  // Only the active element is audible; others ride along silently.
  useEffect(() => {
    for (const t of tracks) { const a = els.current.get(t.id); if (a) a.volume = audible && t.id === audible.id ? 1 : 0; }
  }, [audible, tracks]);

  // Clock + drift guard: keep every element within 40 ms of the audible one.
  useEffect(() => {
    const iv = setInterval(() => {
      const ref = audible ? els.current.get(audible.id) : undefined;
      if (!ref) return;
      setCur(ref.currentTime);
      for (const t of tracks) {
        const a = els.current.get(t.id);
        if (!a || a === ref) continue;
        if (Math.abs(a.currentTime - ref.currentTime) > 0.04) a.currentTime = ref.currentTime;
      }
      if (ref.ended) { setPlaying(false); }
    }, 250);
    return () => clearInterval(iv);
  }, [audible, tracks]);

  const all = () => Array.from(els.current.values());
  const play = async () => { await Promise.all(all().map((a) => a.play().catch(() => {}))); setPlaying(true); };
  const pause = () => { all().forEach((a) => a.pause()); setPlaying(false); };
  const seek = (t: number) => { all().forEach((a) => { a.currentTime = t; }); setCur(t); };
  const pick = (slot: number) => { if (slot >= 0 && slot < tracks.length) setActive(slot); };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (e.key === " ") { e.preventDefault(); playing ? pause() : play(); }
      else if (e.key >= "1" && e.key <= "9") pick(parseInt(e.key, 10) - 1);
      else if (e.key === "ArrowLeft") seek(Math.max(0, cur - 5));
      else if (e.key === "ArrowRight") seek(Math.min(dur, cur + 5));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  if (tracks.length < 2) return null;
  const slotLabel = (slot: number) => {
    const t = tracks[order[slot]];
    return blind && !revealed ? `Track ${String.fromCharCode(65 + slot)}` : t.label;
  };

  return (
    <div className="mb-3 rounded-lg border border-[var(--color-accent)]/40 bg-[#0e1015] p-3">
      <div className="mb-2 flex items-center gap-2">
        <span className="text-xs font-semibold text-[var(--color-ink)]">A/B switch</span>
        <span className="text-[11px] text-[var(--color-muted)]">all tracks play together, only the chosen one is audible · keys 1-{Math.min(9, tracks.length)} switch · space play/pause · ←/→ seek 5 s</span>
        <div className="flex-1" />
        <label className="flex items-center gap-1 text-[11px] text-[var(--color-muted)]">
          <input type="checkbox" checked={blind} onChange={(e) => { setBlind(e.target.checked); setRevealed(false); setSeed((s) => s + 1); }} /> blind
        </label>
        {blind && <button onClick={() => setRevealed((r) => !r)} className="rounded border border-[var(--color-line)] px-2 py-0.5 text-[11px] text-[var(--color-muted)] hover:text-[var(--color-ink)]">{revealed ? "hide" : "reveal"}</button>}
        {blind && <button onClick={() => { setSeed((s) => s + 1); setRevealed(false); }} className="rounded border border-[var(--color-line)] px-2 py-0.5 text-[11px] text-[var(--color-muted)] hover:text-[var(--color-ink)]">reshuffle</button>}
      </div>
      <div className="mb-2 flex flex-wrap gap-1.5">
        {order.map((_, slot) => (
          <button key={slot} onClick={() => pick(slot)}
            className={`rounded-lg border px-3 py-1.5 text-sm ${active === slot ? "border-[var(--color-accent)] bg-[#2a1c19] text-[var(--color-ink)]" : "border-[var(--color-line)] bg-[var(--color-panel2)] text-[var(--color-muted)] hover:text-[var(--color-ink)]"}`}
            title={`key ${slot + 1}`}>
            <span className="mr-1.5 rounded bg-black/30 px-1 text-[10px]">{slot + 1}</span>{slotLabel(slot)}
          </button>
        ))}
      </div>
      <div className="flex items-center gap-3">
        <button onClick={() => (playing ? pause() : play())} disabled={ready < tracks.length}
          className="flex h-9 w-9 flex-none items-center justify-center rounded-full bg-gradient-to-br from-[var(--color-accent)] to-[var(--color-accent2)] text-white shadow disabled:opacity-40">
          {playing ? "❚❚" : "▶"}
        </button>
        <span className="flex-none tabular-nums text-[11px] text-[var(--color-muted)]">{fmt(cur)} / {fmt(dur)}</span>
        <input type="range" min={0} max={Math.max(1, dur)} step={0.1} value={Math.min(cur, dur || 0)} onChange={(e) => seek(parseFloat(e.target.value))} className="w-full accent-[var(--color-accent)]" />
        <span className="flex-none text-[11px] text-[var(--color-muted)]">{ready < tracks.length ? `loading ${ready}/${tracks.length}` : `audible: ${slotLabel(active)}`}</span>
      </div>
    </div>
  );
}
