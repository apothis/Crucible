import { useEffect, useMemo, useRef, useState } from "react";
import type { Block, Character, Seg } from "./mvmodel";
import { resolveSubjects } from "./mvmodel";
import type { LibItem } from "./api";

// Per-shot timeline editor (ports the approved mockup layout). Scoped to ONE block's audio window:
// a real waveform of just that window (decoded peaks) with a time ruler + beat ticks, contiguous
// draggable SEGMENT boundaries, and a mode-aware panel beside the selected-segment prompt. Always
// visible - with no segments it shows the whole shot as one region and a "+ segment" to split it.
// Emits updated Block.segs via onPatch. Dark, flat, matches the mockup's structure.

const A = "#EF9F27";   // amber - performance
const T = "#1D9E75";   // teal - keyframe / b-roll

function evenLens(total: number, n: number): number[] {
  if (n <= 0) return [];
  const each = Math.floor(total / n);
  const out = Array(n).fill(each);
  out[n - 1] = total - each * (n - 1);
  return out;
}
function segLens(segs: Seg[], total: number): number[] {
  if (!segs.length) return [];
  const lens = segs.map((s) => parseInt(s.len, 10) || 0);
  if (lens.every((l) => l > 0)) return lens;
  return evenLens(total, segs.length);
}

export function ShotTimeline({ block, url, beats, stills, libChars, selSeg, onSelSeg, onPatch }: {
  block: Block; url: string; beats: number[]; stills: LibItem[]; libChars: Character[];
  selSeg: number; onSelSeg: (i: number) => void; onPatch: (p: Partial<Block>) => void;
}) {
  const fps = block.fps || 24;
  const total = block.frames;
  const dur = total / fps;
  const winStart = block.audioStart || 0;
  const keyframe = block.renderMode === "keyframe";
  const acc = keyframe ? T : A;
  const laneRef = useRef<HTMLDivElement>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [peaks, setPeaks] = useState<number[]>([]);
  const [playing, setPlaying] = useState(false);
  const [head, setHead] = useState(0);
  const dragRef = useRef<{ i: number } | null>(null);

  const segs = block.segs;
  const hasSegs = segs.length > 0;
  const lens = useMemo(() => segLens(segs, total), [segs, total]);
  const bounds = useMemo(() => {
    if (!hasSegs) return [0, total];
    const b = [0]; let acc = 0;
    for (const l of lens) { acc += l; b.push(acc); }
    b[b.length - 1] = total;
    return b;
  }, [lens, total, hasSegs]);
  const winBeats = useMemo(() => beats
    .filter((t) => t >= winStart && t <= winStart + dur)
    .map((t) => (t - winStart) / dur), [beats, winStart, dur]);
  const ticks = useMemo(() => {
    const out: number[] = []; for (let s = 0; s <= Math.floor(dur); s++) out.push(s); return out;
  }, [dur]);

  useEffect(() => {
    let dead = false;
    if (!url) { setPeaks([]); return; }
    (async () => {
      try {
        const AC = (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext);
        const buf = await (await fetch(url)).arrayBuffer();
        const ctx = new AC();
        const audio = await ctx.decodeAudioData(buf);
        ctx.close();
        const ch = audio.getChannelData(0); const sr = audio.sampleRate;
        const s0 = Math.max(0, Math.floor(winStart * sr));
        const s1 = Math.min(ch.length, Math.floor((winStart + dur) * sr));
        const N = 150; const step = Math.max(1, Math.floor((s1 - s0) / N));
        const out: number[] = [];
        for (let i = s0; i < s1; i += step) {
          let m = 0; for (let j = i; j < i + step && j < s1; j++) { const v = Math.abs(ch[j]); if (v > m) m = v; }
          out.push(m);
        }
        const peak = Math.max(0.01, ...out);
        if (!dead) setPeaks(out.map((v) => v / peak));
      } catch { if (!dead) setPeaks([]); }
    })();
    return () => { dead = true; };
  }, [url, winStart, dur]);

  useEffect(() => {
    const a = audioRef.current; if (!a) return;
    const onTime = () => {
      const local = a.currentTime - winStart;
      if (local < 0) { a.currentTime = winStart; return; }
      if (local > dur) { a.pause(); a.currentTime = winStart; setHead(0); setPlaying(false); return; }
      setHead(local);
    };
    a.addEventListener("timeupdate", onTime);
    a.addEventListener("play", () => setPlaying(true));
    a.addEventListener("pause", () => setPlaying(false));
    return () => a.removeEventListener("timeupdate", onTime);
  }, [winStart, dur]);

  const playPause = () => {
    const a = audioRef.current; if (!a) return;
    if (a.paused) { if (a.currentTime < winStart || a.currentTime > winStart + dur) a.currentTime = winStart; a.play(); }
    else a.pause();
  };

  const snapFrac = (frac: number) => {
    if (!winBeats.length) return frac;
    let best = frac, bd = 1;
    for (const wb of winBeats) { const d = Math.abs(wb - frac); if (d < bd) { bd = d; best = wb; } }
    return bd < 0.04 ? best : frac;
  };

  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      const d = dragRef.current, lane = laneRef.current; if (!d || !lane) return;
      const rect = lane.getBoundingClientRect();
      let frac = (e.clientX - rect.left) / rect.width; frac = Math.max(0, Math.min(1, frac));
      frac = snapFrac(frac);
      let frame = Math.round(frac * total);
      const lo = bounds[d.i - 1] + 1, hi = bounds[d.i + 1] - 1;
      frame = Math.max(lo, Math.min(hi, frame));
      const next = segs.map((s) => ({ ...s }));
      next[d.i - 1].len = String(frame - bounds[d.i - 1]);
      next[d.i].len = String(bounds[d.i + 1] - frame);
      onPatch({ segs: next });
    };
    const onUp = () => { dragRef.current = null; };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => { window.removeEventListener("pointermove", onMove); window.removeEventListener("pointerup", onUp); };
  }, [segs, bounds, total]);

  const setSeg = (i: number, p: Partial<Seg>) => onPatch({ segs: segs.map((s, j) => j === i ? { ...s, ...p } : s) });
  const addSeg = () => {
    if (!hasSegs) { const h = Math.max(1, Math.floor(total / 2)); onPatch({ segs: [{ len: String(h), prompt: "" }, { len: String(total - h), prompt: "" }] }); onSelSeg(0); return; }
    onPatch({ segs: [...segs, { len: "", prompt: "" }] });
  };
  const splitSeg = (i: number) => {
    const half = Math.max(1, Math.floor(lens[i] / 2));
    onPatch({ segs: [...segs.slice(0, i), { ...segs[i], len: String(half) }, { ...segs[i], len: String(lens[i] - half) }, ...segs.slice(i + 1)] });
  };
  const delSeg = (i: number) => onPatch({ segs: segs.filter((_, j) => j !== i) });

  const sel = hasSegs && segs[selSeg] ? selSeg : 0;
  const charNames = block.chars.map((bc) => libChars.find((c) => c.id === bc.charId)?.name).filter(Boolean) as string[];
  const subjectCount = resolveSubjects(block, libChars).length;
  const stillLabel = (id?: string) => {
    if (!id) return "";
    const it = stills.find((s) => s.id === id);
    return (it?.params?.prompt as string || it?.params?.title as string || id).toString().slice(0, 26);
  };

  return (
    <div className="st-root">
      <style>{`
        .st-root{--bg:#0c0e13;--bg2:#16191f;--line:rgba(255,255,255,.09);--tx:#e7e8ec;--mut:#9aa1ad;font-size:12px;color:var(--tx);}
        .st-root *{box-sizing:border-box;}
        .st-ruler{position:relative;height:14px;margin-bottom:3px;font-size:10px;color:var(--mut);}
        .st-ruler span{position:absolute;transform:translateX(-50%);}
        .st-lane{position:relative;height:76px;border-radius:10px;overflow:hidden;background:var(--bg);border:1px solid var(--line);}
        .st-bars{position:absolute;inset:0;display:flex;align-items:center;gap:1px;padding:0 2px;}
        .st-bar{flex:1;border-radius:1px;background:#39404d;}
        .st-tick{position:absolute;top:0;bottom:0;width:1px;background:rgba(255,255,255,.10);}
        .st-seg{position:absolute;top:6px;bottom:6px;border-radius:5px;cursor:pointer;overflow:hidden;font-size:10px;padding:5px 7px;line-height:1.2;}
        .st-bnd{position:absolute;top:0;bottom:0;width:14px;margin-left:-7px;cursor:ew-resize;display:flex;justify-content:center;z-index:4;}
        .st-bnd::after{content:"";width:2px;height:100%;background:var(--tx);opacity:.6;}
        .st-pl{position:absolute;top:-1px;bottom:-1px;width:2px;background:#fff;z-index:5;pointer-events:none;}
        .st-ctl{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-top:9px;}
        .st-btn{border:1px solid var(--line);background:transparent;color:var(--mut);border-radius:8px;padding:5px 10px;font-size:12px;cursor:pointer;}
        .st-btn:hover:not(:disabled){color:var(--tx);background:var(--bg2);}
        .st-btn:disabled{opacity:.4;}
        .st-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px;}
        .st-pan{background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:11px;}
        .st-lbl{font-size:11px;color:var(--mut);margin-bottom:5px;}
        .st-ta{width:100%;background:var(--bg2);border:1px solid var(--line);border-radius:8px;color:var(--tx);font-size:13px;padding:8px 9px;resize:none;font-family:inherit;}
        .st-in{width:100%;background:var(--bg2);border:1px solid var(--line);border-radius:8px;color:var(--tx);font-size:12px;padding:6px 8px;}
        .st-note{font-size:11px;color:var(--mut);}
        .st-chip{display:inline-flex;align-items:center;justify-content:center;border-radius:8px;background:var(--bg2);border:1px solid var(--line);color:var(--mut);}
      `}</style>

      <div className="st-ruler">{ticks.map((s) => (
        <span key={s} style={{ left: `${(s / dur) * 100}%`, transform: s === 0 ? "none" : (s === ticks[ticks.length - 1] ? "translateX(-100%)" : "translateX(-50%)") }}>{s}s</span>
      ))}</div>

      <div ref={laneRef} className="st-lane">
        <div className="st-bars">
          {(peaks.length ? peaks : Array(110).fill(0.05)).map((v, i) => (
            <span key={i} className="st-bar" style={{ height: `${Math.max(4, v * 72)}%` }} />
          ))}
        </div>
        {winBeats.map((f, i) => <span key={i} className="st-tick" style={{ left: `${f * 100}%` }} />)}
        {hasSegs ? segs.map((s, i) => {
          const left = (bounds[i] / total) * 100, w = ((bounds[i + 1] - bounds[i]) / total) * 100;
          const on = i === sel;
          return (
            <div key={i} className="st-seg" onClick={() => onSelSeg(i)}
              style={{ left: `${left}%`, width: `${w}%`,
                background: on ? (keyframe ? "rgba(29,158,117,.32)" : "rgba(239,159,39,.32)") : (keyframe ? "rgba(29,158,117,.14)" : "rgba(239,159,39,.14)"),
                border: on ? `2px solid ${acc}` : `1px solid ${acc}55`, color: keyframe ? "#9fe1cb" : "#fac07a" }}>
              {s.keyframeStillId ? "◈ " : ""}{s.prompt ? s.prompt.slice(0, 44) : `segment ${i + 1}`}
            </div>
          );
        }) : (
          <div className="st-seg" style={{ left: 0, width: "100%", background: "rgba(255,255,255,.05)", border: "1px dashed var(--line)", color: "var(--mut)" }}>
            whole shot - no segments yet
          </div>
        )}
        {hasSegs && bounds.slice(1, -1).map((c, k) => (
          <div key={k} className="st-bnd" style={{ left: `${(c / total) * 100}%` }}
            onPointerDown={(e) => { e.preventDefault(); dragRef.current = { i: k + 1 }; }} title="drag to retime" />
        ))}
        <div className="st-pl" style={{ left: `${(head / dur) * 100}%` }} />
      </div>

      <div className="st-ctl">
        <button className="st-btn" onClick={playPause}>{playing ? "❚❚" : "▶"}</button>
        <span className="st-note" style={{ fontVariantNumeric: "tabular-nums" }}>{head.toFixed(1)}s / {dur.toFixed(1)}s</span>
        <button className="st-btn" onClick={addSeg}>+ segment</button>
        <button className="st-btn" onClick={() => splitSeg(sel)} disabled={!hasSegs}>split</button>
        <button className="st-btn" onClick={() => delSeg(sel)} disabled={segs.length < 2}>delete</button>
        <span className="st-note" style={{ marginLeft: "auto" }}>
          {winBeats.length ? `${winBeats.length} beats - drag a boundary (snaps to beat)` : "drag a boundary to retime"}
        </span>
      </div>

      <div className="st-grid">
        <div className="st-pan">
          {hasSegs ? (
            <>
              <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 8 }}>
                Segment {sel + 1} - {((bounds[sel] || 0) / fps).toFixed(1)}s to {((bounds[sel + 1] || 0) / fps).toFixed(1)}s
              </div>
              <textarea className="st-ta" rows={3} value={segs[sel]?.prompt || ""} placeholder={`segment ${sel + 1} prompt`}
                onChange={(e) => setSeg(sel, { prompt: e.target.value })} />
              {keyframe && (
                <div style={{ marginTop: 8 }}>
                  <div className="st-lbl">Keyframe still (placed at this segment)</div>
                  <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                    <select className="st-in" value={segs[sel]?.keyframeStillId || ""}
                      onChange={(e) => setSeg(sel, { keyframeStillId: e.target.value || undefined })}>
                      <option value="">- no still -</option>
                      {stills.map((s) => <option key={s.id} value={s.id}>{stillLabel(s.id)}</option>)}
                    </select>
                    <label style={{ display: "flex", gap: 4, alignItems: "center", color: "var(--mut)", whiteSpace: "nowrap" }}
                      title="place the still at the END of this segment instead of the start">
                      <input type="checkbox" checked={!!segs[sel]?.isEndFrame}
                        onChange={(e) => setSeg(sel, { isEndFrame: e.target.checked || undefined })} /> end
                    </label>
                  </div>
                </div>
              )}
            </>
          ) : (
            <>
              <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 8 }}>One continuous shot</div>
              <p className="st-note">This shot runs as one action ({dur.toFixed(1)}s). Add a segment to schedule the action/prompt over time{keyframe ? " or pin keyframe stills" : ""}.</p>
              <button className="st-btn" style={{ marginTop: 8 }} onClick={addSeg}>+ split into segments</button>
            </>
          )}
        </div>

        <div className="st-pan">
          {keyframe ? (
            <>
              <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 10 }}>B-roll shot</div>
              <div className="st-lbl">Keyframe stills (interpolate between)</div>
              <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                {[0, Math.max(0, segs.length - 1)].filter((v, i, a) => a.indexOf(v) === i).map((i) => (
                  <div key={i} style={{ textAlign: "center" }}>
                    <div className="st-chip" style={{ width: 64, height: 38 }}>{segs[i]?.keyframeStillId ? "◈" : "□"}</div>
                    <div style={{ fontSize: 10, color: "var(--mut)", marginTop: 3 }}>{i === 0 ? "start" : "end"}</div>
                  </div>
                ))}
              </div>
              <div className="st-note" style={{ marginTop: 10 }}>no identity anchor - no lip-sync (scenic / camera move)</div>
            </>
          ) : (
            <>
              <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 10 }}>Performance shot</div>
              <div className="st-lbl">Identity - anchored (MSR)</div>
              <div style={{ fontSize: 12, marginBottom: 12 }}>
                {charNames.length ? charNames.join(", ") : subjectCount ? `${subjectCount} subject still${subjectCount > 1 ? "s" : ""}` :
                  <span style={{ color: "var(--mut)" }}>set the character / subject stills above</span>}
              </div>
              <div className="st-lbl">Lip-sync</div>
              <div style={{ fontSize: 12 }}>
                {block.lipsync ? <span style={{ color: A }}>on - <span style={{ color: "var(--mut)" }}>offset {block.audioStart.toFixed(1)}s, LTX native masked-audio</span></span>
                  : <span style={{ color: "var(--mut)" }}>off (toggle in the shot header)</span>}
              </div>
            </>
          )}
        </div>
      </div>
      <audio ref={audioRef} src={url} preload="auto" crossOrigin="anonymous" style={{ display: "none" }} />
    </div>
  );
}
