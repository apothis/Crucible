import { useEffect, useMemo, useRef, useState } from "react";
import abcjs from "abcjs";
import "abcjs/abcjs-audio.css";
import { inp, GhostButton } from "./ui";
import type { AbcInfo } from "./Yue2Studio";

// ---------------------------------------------------------------------------------------------
// The YuE2 score editor. The score is ABC text (the thing YuE2 reads), and every edit here
// ends as a new ABC string handed back through setAbc. Three views of the same text:
//
//   notation   abcjs renderAbc (multi-voice staff, chord symbols). Clicking a note selects its
//              text in the ABC box, so notation is the map and the text is the edit surface.
//   roll       a piano roll drawn from abcjs's own audio sequence (pitch/start/duration per
//              voice), with the % section markers laid over it. Read-only in this version.
//   text       the raw ABC.
//
// Levers that hold: transpose (abcjs strTranspose rewrites the notes AND the chord symbols,
// measured correct on our first render), tempo (Q: rewrite; the planner ignored BPM in the
// style line on both seeds we tried, Q: is what the music stage reads), and MIDI export
// (abcjs getMidiFile). Playback is abcjs's synth: per-note GM samples fetched on demand,
// so the first play needs the network; program 27 = clean electric guitar for both voices.
// ---------------------------------------------------------------------------------------------

type Props = { abc: string; setAbc: (t: string) => void; info: AbcInfo | null; onClear: () => void };

// ---- pitch editing in the text ----
// A dragged roll note becomes a text edit: the note token at [startChar, endChar) is re-spelled at
// the new MIDI pitch WITH an explicit accidental (= ^ _), which is unambiguous for that note. An
// explicit accidental also carries to later notes of the same letter in the bar, so after the edit
// the score is re-parsed and any other note whose pitch moved gets an explicit accidental that
// restores it. Time/duration dragging is not implemented: durations are tied to L: and bar maths.
const NOTE_RE = /([_^=]*)([A-Ga-g])([,']*)/;
const SHARP_NAMES = ["C", "^C", "D", "^D", "E", "F", "^F", "G", "^G", "A", "^A", "B"];
const FLAT_NAMES = ["C", "_D", "D", "_E", "E", "F", "_G", "G", "_A", "A", "_B", "B"];
function keyPrefersFlats(k: string | undefined): boolean {
  const root = (k || "C").trim();
  return /^[A-G]b/.test(root) || /^F(?![#])/.test(root) || /^(D|G|C)m/.test(root) || /^Dm/.test(root);
}
function spell(midi: number, flats: boolean): string {
  const pc = ((midi % 12) + 12) % 12;
  const name = (flats ? FLAT_NAMES : SHARP_NAMES)[pc];
  const acc = name.length > 1 ? name[0] : "=";
  const letter = name[name.length - 1];
  const octave = Math.floor(midi / 12) - 1;         // 60 -> 4
  let body = octave >= 5 ? letter.toLowerCase() : letter;
  if (octave >= 6) body += "'".repeat(octave - 5);
  if (octave <= 3) body += ",".repeat(4 - octave);
  return acc + body;
}
function respellNote(abc: string, startChar: number, endChar: number, midi: number, flats: boolean): string {
  const tok = abc.slice(startChar, endChar);
  const m = NOTE_RE.exec(tok);
  if (!m || m.index == null) return abc;
  const a = startChar + m.index, b = a + m[0].length;
  return abc.slice(0, a) + spell(midi, flats) + abc.slice(b);
}
function notePitches(abc: string): { pitch: number; startChar: number; endChar: number }[] {
  const holder = document.createElement("div");
  const tunes = abcjs.renderAbc(holder, abc, { add_classes: false });
  const seq = tunes[0].setUpAudio({ chordsOff: true });
  const out: { pitch: number; startChar: number; endChar: number }[] = [];
  seq.tracks.forEach((track) => track.forEach((it) => { if (it.cmd === "note") out.push({ pitch: it.pitch, startChar: it.startChar, endChar: it.endChar }); }));
  return out.sort((x, y) => x.startChar - y.startChar);
}
// Move one note by `delta` semitones and keep every other note where it was.
function movePitch(abc: string, target: { startChar: number; endChar: number; pitch: number }, delta: number, flats: boolean): { abc: string; fixed: number } | null {
  const before = notePitches(abc);
  const idx = before.findIndex((n) => n.startChar === target.startChar);
  if (idx < 0) return null;
  let out = respellNote(abc, target.startChar, target.endChar, target.pitch + delta, flats);
  let fixed = 0;
  for (let pass = 0; pass < 3; pass++) {
    const after = notePitches(out);
    if (after.length !== before.length) return null;      // the edit broke the parse; bail
    let changed = false;
    // walk from the end so earlier char offsets stay valid while we splice
    for (let i = after.length - 1; i >= 0; i--) {
      if (i === idx) continue;
      if (after[i].pitch !== before[i].pitch) {
        out = respellNote(out, after[i].startChar, after[i].endChar, before[i].pitch, flats);
        fixed++; changed = true;
      }
    }
    if (!changed) break;
  }
  return { abc: out, fixed };
}
// ---- key picker ----
// The whole-score move is still abcjs strTranspose (notes, chord symbols and K: together); the picker
// only turns a target key into the semitone count, so a key change reads as a key, not as arithmetic.
const PITCH: Record<string, number> = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };
const MAJOR_NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"];
const MINOR_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"];
// transpose a score that is not on screen (the plan list): abcjs needs a parsed tune for strTranspose, so the
// score is laid out once in a detached element
export function transposeAbc(abc: string, steps: number): string {
  const tunes = abcjs.renderAbc(document.createElement("div"), abc, { staffwidth: 900 });
  return abcjs.strTranspose(abc, tunes, steps);
}
export const TONICS = ["C", "C#/Db", "D", "Eb", "E", "F", "F#", "G", "G#/Ab", "A", "Bb", "B"];
export function parseKey(k: string | undefined): { pc: number; minor: boolean; suffix: string } | null {
  const m = /^\s*([A-G])([#b]?)\s*(\S*)/.exec(k || "");
  if (!m) return null;
  const pc = (PITCH[m[1]] + (m[2] === "#" ? 1 : m[2] === "b" ? -1 : 0) + 12) % 12;
  const minor = /^m(in(or)?)?$/i.test(m[3]) || /^aeo/i.test(m[3]);
  return { pc, minor, suffix: minor ? "m" : m[3] && !/^maj/i.test(m[3]) ? m[3] : "" };
}
export function keyName(key: { pc: number; minor: boolean; suffix: string }, steps: number): string {
  const pc = (((key.pc + steps) % 12) + 12) % 12;
  return (key.minor ? MINOR_NAMES : MAJOR_NAMES)[pc] + key.suffix;
}

type View = "notation" | "roll" | "text";
type RollNote = { pitch: number; start: number; duration: number; voice: number; startChar: number; endChar: number };

const VOICE_COLORS = ["#f0911d", "#4fa3e0"];   // Vocal, Ins

export function ScoreEditor({ abc, setAbc, info, onClear }: Props) {
  const [view, setView] = useState<View>("notation");
  const paperRef = useRef<HTMLDivElement>(null);
  const audioRef = useRef<HTMLDivElement>(null);
  const textRef = useRef<HTMLTextAreaElement>(null);
  const tunesRef = useRef<abcjs.TuneObjectArray | null>(null);
  const synthRef = useRef<abcjs.SynthObjectController | null>(null);
  const [warn, setWarn] = useState("");
  const [bpm, setBpm] = useState("");
  const [roll, setRoll] = useState<{ notes: RollNote[]; total: number; lo: number; hi: number } | null>(null);
  const [synthReady, setSynthReady] = useState(false);

  useEffect(() => { setBpm(info?.bpm ? String(info.bpm) : ""); }, [info?.bpm]);

  // Render the notation whenever the text changes (debounced: abcjs is fast but a 140-bar
  // two-voice score is still ~50 ms, and typing should not fight it).
  useEffect(() => {
    if (!abc.trim() || !paperRef.current) { tunesRef.current = null; setRoll(null); return; }
    const t = window.setTimeout(() => {
      if (!paperRef.current) return;
      try {
        const tunes = abcjs.renderAbc(paperRef.current, abc, {
          responsive: "resize", add_classes: true, foregroundColor: "#e7e9ee",
          selectionColor: "#f0911d", staffwidth: 900, wrap: { minSpacing: 1.6, maxSpacing: 2.4, preferredMeasuresPerLine: 4 },
          clickListener: (el: abcjs.AbcElem) => {
            // a clicked note selects its ABC text; switch to the text view to edit it
            if (el.startChar == null || el.endChar == null || !textRef.current) return;
            setView("text");
            window.setTimeout(() => {
              const ta = textRef.current; if (!ta) return;
              ta.focus(); ta.setSelectionRange(el.startChar!, el.endChar!);
              // scroll the selection into view (textarea has no API for it; nudge by line)
              const line = abc.slice(0, el.startChar!).split("\n").length;
              ta.scrollTop = Math.max(0, (line - 4) * 15);
            }, 0);
          },
        });
        tunesRef.current = tunes;
        const w = (tunes[0] as unknown as { warnings?: string[] }).warnings;
        setWarn(w?.length ? w.slice(0, 3).join(" · ") : "");
        // piano-roll data from abcjs's own sequencing (no chord accompaniment track)
        try {
          const seq = tunes[0].setUpAudio({ chordsOff: true });
          const notes: RollNote[] = [];
          let total = 0, lo = 127, hi = 0;
          seq.tracks.forEach((track, vi) => {
            track.forEach((it) => {
              if (it.cmd !== "note") return;
              notes.push({ pitch: it.pitch, start: it.start, duration: it.duration, voice: Math.min(vi, 1), startChar: it.startChar, endChar: it.endChar });
              total = Math.max(total, it.start + it.duration);
              lo = Math.min(lo, it.pitch); hi = Math.max(hi, it.pitch);
            });
          });
          setRoll(notes.length ? { notes, total, lo: lo - 2, hi: hi + 2 } : null);
        } catch { setRoll(null); }
        // synth: reload the tune into the controller (no autoplay)
        if (synthRef.current && abcjs.synth.supportsAudio()) {
          synthRef.current.setTune(tunes[0], false, { chordsOff: true, program: 27 })
            .then(() => setSynthReady(true)).catch(() => setSynthReady(false));
        }
      } catch (e) {
        setWarn(`abcjs could not render this score: ${e}`);
        tunesRef.current = null;
      }
    }, 250);
    return () => window.clearTimeout(t);
  }, [abc]);

  // the synth controls widget mounts once
  useEffect(() => {
    if (!audioRef.current || synthRef.current || !abcjs.synth.supportsAudio()) return;
    const c = new abcjs.synth.SynthController();
    c.load(audioRef.current, null, { displayLoop: true, displayRestart: true, displayPlay: true, displayProgress: true, displayWarp: true });
    synthRef.current = c;
  }, []);

  function transpose(steps: number) {
    if (!tunesRef.current) return;
    try { setAbc(abcjs.strTranspose(abc, tunesRef.current, steps)); }
    catch (e) { setWarn(`transpose failed: ${e}`); }
  }
  function applyTempo() {
    const n = parseInt(bpm);
    if (!n) return;
    const out = abc.split("\n").map((l) => /^Q:/.test(l.trim()) ? l.replace(/=\s*\d+/, `=${n}`) : l).join("\n");
    setAbc(/^Q:/m.test(abc) ? out : abc.replace(/^(L:[^\n]*\n)/m, `$1Q:1/4=${n}\n`));
  }
  function downloadMidi() {
    try {
      const f = abcjs.synth.getMidiFile(abc, { midiOutputType: "binary", chordsOff: true, program: 27 }) as unknown as Uint8Array[] | Uint8Array;
      const bytes = Array.isArray(f) ? f[0] : f;
      const url = URL.createObjectURL(new Blob([bytes as BlobPart], { type: "audio/midi" }));
      const a = document.createElement("a"); a.href = url; a.download = "yue2_score.mid"; a.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 5000);
    } catch (e) { setWarn(`MIDI export failed: ${e}`); }
  }

  // section markers for the roll: the % comment's char offset -> first note at/after it
  const sectionMarks = useMemo(() => {
    if (!roll || !info?.sections?.length) return [] as { name: string; x: number }[];
    const lines = abc.split("\n");
    const offsets: number[] = []; let acc = 0;
    lines.forEach((l) => { offsets.push(acc); acc += l.length + 1; });
    const sorted = [...roll.notes].sort((a, b) => a.startChar - b.startChar);
    return info.sections.map((s) => {
      const off = offsets[s.line] ?? 0;
      const n = sorted.find((x) => x.startChar >= off);
      return { name: s.name, x: n ? n.start : roll.total };
    });
  }, [roll, info, abc]);

  const empty = !abc.trim();
  const curKey = parseKey(info?.K);
  return (
    <div className="mt-3 rounded border border-[var(--color-line)] bg-[#0e1015] p-2">
      <div className="flex flex-wrap items-center gap-1">
        {(["notation", "roll", "text"] as View[]).map((v) => (
          <button key={v} onClick={() => setView(v)}
            className={`rounded px-2 py-1 text-[11px] ${view === v ? "bg-[var(--color-accent2)]/20 text-[var(--color-ink)]" : "text-[var(--color-muted)] hover:text-[var(--color-ink)]"}`}>
            {v}
          </button>
        ))}
        <span className="flex-1" />
        {info && !empty && (
          <span className="text-[10px] text-[var(--color-muted)]">
            {info.K || "?"} {"·"} {info.bpm ?? "?"} bpm {"·"} {info.M} {"·"} {info.bars} bars {"·"} {info.voices.map((v) => v.id).join(" + ")}
            {info.problems.length > 0 && <span className="text-red-400"> {"·"} {info.problems.join("; ")}</span>}
          </span>
        )}
      </div>

      {/* levers */}
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <span className="text-[10px] text-[var(--color-muted)]">transpose</span>
        <GhostButton onClick={() => transpose(-12)} disabled={empty} title="down an octave">-12</GhostButton>
        <GhostButton onClick={() => transpose(-1)} disabled={empty} title="down a semitone (notes and chord symbols)">-1</GhostButton>
        <GhostButton onClick={() => transpose(1)} disabled={empty} title="up a semitone (notes and chord symbols)">+1</GhostButton>
        <GhostButton onClick={() => transpose(12)} disabled={empty} title="up an octave">+12</GhostButton>
        {curKey && (
          <select className={inp + " w-auto"} value="" disabled={empty}
            onChange={(e) => { const n = parseInt(e.target.value); if (n) transpose(n); }}
            title="move the whole score (both voices, chord symbols, K:) to another key; + is up, - is down, so pick the direction that suits the voice">
            <option value="">to key... (now {keyName(curKey, 0)})</option>
            {[6, 5, 4, 3, 2, 1, -1, -2, -3, -4, -5, -6].map((n) => (
              <option key={n} value={n}>{keyName(curKey, n)} ({n > 0 ? "+" : ""}{n})</option>
            ))}
          </select>
        )}
        <span className="ml-2 text-[10px] text-[var(--color-muted)]">tempo</span>
        <input className={inp} style={{ width: 72 }} value={bpm} onChange={(e) => setBpm(e.target.value)} placeholder="bpm" disabled={empty} />
        <GhostButton onClick={applyTempo} disabled={empty || !bpm.trim()} title="rewrite Q: in the header - the tempo lever the render honours">set</GhostButton>
        <span className="flex-1" />
        <GhostButton onClick={downloadMidi} disabled={empty} title="standard MIDI file of both voices">↓ MIDI</GhostButton>
        <GhostButton onClick={() => { navigator.clipboard.writeText(abc); }} disabled={empty}>copy ABC</GhostButton>
        <GhostButton onClick={onClear} disabled={empty} title="drop the score; the next render plans its own">clear</GhostButton>
      </div>
      {warn && <div className="mt-1 text-[10px] text-amber-400">{warn}</div>}

      {/* playback (abcjs synth controls) */}
      <div ref={audioRef} className={`abcjs-audio mt-2 ${empty || !synthReady ? "opacity-40" : ""}`} />

      {/* views: all three stay mounted; the paper div must exist for abcjs to render into */}
      {/* wrapper carries the hide: abcjs adds its own class + display rule to the paper div itself */}
      <div className={`mt-2 max-h-[480px] overflow-auto rounded bg-[#151821] p-2 ${view === "notation" && !empty ? "" : "hidden"}`}>
        <div ref={paperRef} />
      </div>
      {view === "notation" && empty && (
        <div className="mt-2 p-4 text-center text-[11px] text-[var(--color-muted)]">
          no score loaded: write plans above, transcribe a recording below, or paste ABC in the text view
        </div>
      )}
      {view === "roll" && (roll ? <PianoRoll roll={roll} marks={sectionMarks}
        onNote={(n) => {
          setView("text");
          window.setTimeout(() => { const ta = textRef.current; if (!ta) return; ta.focus(); ta.setSelectionRange(n.startChar, n.startChar + 2); }, 0);
        }}
        onDrag={(n, delta) => {
          if (!delta) return;
          const r = movePitch(abc, { startChar: n.startChar, endChar: n.endChar, pitch: n.pitch }, delta, keyPrefersFlats(info?.K));
          if (!r) { setWarn("could not move that note in the text (the token was not found or the edit broke the parse)"); return; }
          setAbc(r.abc);
          setWarn(r.fixed ? `moved ${delta > 0 ? "+" : ""}${delta} semitones; ${r.fixed} other note(s) in the bar got an explicit accidental to stay put` : "");
        }} /> : <div className="mt-2 p-4 text-center text-[11px] text-[var(--color-muted)]">no notes to show</div>)}
      {view === "roll" && roll && (
        <div className="mt-1 text-[10px] text-[var(--color-muted)]">
          Drag a note up or down to change its pitch (written into the ABC with an explicit accidental). Click a note to jump to it in the text. Time and length are edited in the text.
        </div>
      )}
      <textarea ref={textRef} className={inp + " mt-2 min-h-[320px] font-mono text-[11px] leading-[15px] " + (view === "text" ? "" : "hidden")}
        rows={20} value={abc} onChange={(e) => setAbc(e.target.value)} spellCheck={false}
        placeholder={"X:1\nT:\nM:4/4\nL:1/32\nQ:1/4=120\nV: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"\nV: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"\nK:Am\n% intro\nV: Vocal\n..."} />
      {view === "text" && !empty && (
        <div className="mt-1 text-[10px] text-[var(--color-muted)]">
          Two voices (Vocal, Ins) in alternating blocks, chords in quotes, sections as % comments, L:1/32 so a
          quarter note is 8. Click a note in the notation or the roll to jump to it here.
        </div>
      )}
    </div>
  );
}

function PianoRoll({ roll, marks, onNote, onDrag }: { roll: { notes: RollNote[]; total: number; lo: number; hi: number }; marks: { name: string; x: number }[]; onNote: (n: RollNote) => void; onDrag: (n: RollNote, deltaSemitones: number) => void }) {
  // drag state: which note, where the press started, and the live semitone offset for the ghost
  const [drag, setDrag] = useState<{ i: number; y0: number; dy: number } | null>(null);
  const pxPerUnit = 640 / Math.max(roll.total, 1);   // whole-note units -> px, fit ~640px then scroll
  const scale = Math.max(pxPerUnit, 160);            // at least 160 px per whole note so eighths are visible
  const width = Math.ceil(roll.total * scale) + 40;
  const rows = roll.hi - roll.lo + 1;
  const rowH = 7;
  const height = rows * rowH + 24;
  const y = (p: number) => (roll.hi - p) * rowH + 18;
  const isBlack = (p: number) => [1, 3, 6, 8, 10].includes(p % 12);
  return (
    <div className="mt-2 max-h-[480px] overflow-auto rounded bg-[#151821]">
      <svg width={width} height={height} className="block">
        {Array.from({ length: rows }, (_v, i) => roll.lo + i).map((p) => (
          <rect key={p} x={0} y={y(p)} width={width} height={rowH} fill={isBlack(p) ? "#11141a" : "#161a24"} />
        ))}
        {Array.from({ length: rows }, (_v, i) => roll.lo + i).filter((p) => p % 12 === 0).map((p) => (
          <text key={"c" + p} x={2} y={y(p) + rowH - 1} fontSize={7} fill="#8b909a">C{p / 12 - 1}</text>
        ))}
        {marks.map((m, i) => (
          <g key={i}>
            <line x1={m.x * scale + 30} y1={0} x2={m.x * scale + 30} y2={height} stroke="#3a3f4d" strokeDasharray="3 3" />
            <text x={m.x * scale + 33} y={10} fontSize={9} fill="#f7b733">{m.name}</text>
          </g>
        ))}
        {roll.notes.map((n, i) => {
          const off = drag && drag.i === i ? Math.round(-drag.dy / rowH) : 0;
          return (
            <rect key={i} x={n.start * scale + 30} y={y(n.pitch + off) + 1} width={Math.max(2, n.duration * scale - 1)} height={rowH - 2}
              rx={1} fill={VOICE_COLORS[n.voice]} opacity={drag && drag.i === i ? 1 : 0.9} stroke={drag && drag.i === i ? "#fff" : "none"}
              className="cursor-ns-resize"
              onMouseDown={(e) => { e.preventDefault(); setDrag({ i, y0: e.clientY, dy: 0 }); }}>
              <title>{`${n.voice === 0 ? "Vocal" : "Ins"} · midi ${n.pitch}${off ? ` -> ${n.pitch + off}` : ""}`}</title>
            </rect>
          );
        })}
        {drag && (
          // a full-size capture layer so the drag keeps tracking outside the note
          <rect x={0} y={0} width={width} height={height} fill="transparent"
            onMouseMove={(e) => setDrag({ ...drag, dy: e.clientY - drag.y0 })}
            onMouseUp={(e) => {
              const delta = Math.round(-(e.clientY - drag.y0) / rowH);
              const n = roll.notes[drag.i];
              setDrag(null);
              if (delta === 0) onNote(n); else onDrag(n, delta);
            }}
            onMouseLeave={() => setDrag(null)} />
        )}
      </svg>
    </div>
  );
}
