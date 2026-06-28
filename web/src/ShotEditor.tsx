import { useEffect, useMemo, useState } from "react";
import { api, type LibItem } from "./api";
import { Field, inp, GhostButton, PrimaryButton, rid } from "./ui";
import { Num, StillPick } from "./mvui";
import { type Block, type Character, type Take, charRefIds, sceneRefOf, resolveSubjects } from "./mvmodel";

// ---------------------------------------------------------------------------------------------
// Shot Editor — the clean, guided per-shot flow that REPLACES the old Shot Studio + LTXDirector
// embed. A shot is built in ordered STAGES, each producing a cheap artifact you approve before
// paying for the next (the "gate artifacts before render" rule):
//   Type → Scene (person-free background) → Cast → Placement preview → Video options → Result.
// It keeps every working engine under the hood: MSR (performance / lip-sync), char_still band
// composite, FFLF push-in (B-roll), seed-hunt, the dev model. No timeline editor to wrestle.
// ---------------------------------------------------------------------------------------------

const FFLF_TAIL = 33;
// A clip reference must be a BARE job id (the timeline/assembly + /api/media build the URL themselves).
const bareId = (s?: string) => (s ? s.split("/").pop()!.split("?")[0].split("#")[0] : "");
// FFLF "calm motion" (B-roll): LTX renders water/clouds as a fast timelapse by default + the stock FFLF
// negative pushes FOR motion, so explicit anti-timelapse cues tame the speed (with the dev model).
const CALM_NEG = "timelapse, time-lapse, hyperlapse, fast motion, sped-up footage, fast moving clouds, " +
  "fast moving water, racing waves, flickering, strobing, blurry, low quality, watermark, subtitles, music";
const CALM_POS = "slow gentle cinematic motion, calm, slow-moving water, slowly drifting clouds";

// Poll a video job to its media url; onPct reports (pct, pass). A multi-stage render emits a fresh
// 0→100 per stage; we detect the restart (big drop) and bump `pass` so the UI shows "pass 2 of 2".
function waitMedia(jobId: string, onPct?: (pct: number, pass: number) => void): Promise<string> {
  let lastPct = 0, pass = 1;
  return new Promise((resolve, reject) => {
    const t = window.setInterval(async () => {
      const j = await api.job(jobId).catch(() => null);
      if (!j) return;
      if (j.status === "done" && j.media_url) { window.clearInterval(t); resolve(j.media_url); }
      else if (j.status === "error" || j.status === "failed") { window.clearInterval(t); reject(new Error(j.error || "render error")); }
      else if (onPct && j.max) {
        const pct = Math.round((100 * (j.progress || 0)) / j.max);
        if (pct < lastPct - 20) pass += 1;
        lastPct = pct;
        onPct(pct, pass);
      }
    }, 1500);
  });
}

const vHalf = (n: number) => Math.max(256, Math.round(n / 2 / 32) * 32);
const FRAME_PHRASE: Record<string, string> = { close: "close-up shot", medium: "medium shot", wide: "wide shot" };
const BACKDROP: Record<string, string> = {
  close: "a close, soft, shallow-depth-of-field backdrop of",
  medium: "a medium-distance view of",
  wide: "a wide establishing shot of",
};

type Draft = { jobId: string; seed: number; url?: string; pct?: number; pass?: number };

function VideoTile({ src }: { src: string }) {
  return <video src={src} muted loop autoPlay playsInline controls preload="auto" className="h-full w-full object-contain" />;
}

export function ShotEditor({ block: b, idx, patch, stills, audios, songAudioId, libChars, onClose }: {
  block: Block; idx: number; patch: (p: Partial<Block>) => void;
  stills: LibItem[]; audios: LibItem[]; songAudioId: string; libChars: Character[];
  onClose: () => void;
}) {
  // Performance = a character performs (MSR identity / S2V lip-sync). Everything else (keyframe, fflf, i2v)
  // is scenic B-roll with no cast, authored through the FFLF push-in path.
  const perf = b.renderMode === "msr" || b.renderMode === "s2v";
  const STAGES: string[] = perf
    ? ["type", "scene", "cast", "placement", "video", "result"]
    : ["type", "scene", "video", "result"];

  const [step, setStep] = useState<string>("type");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const note = (s: string) => setStatus(s);

  // ---- shared render settings ----
  const nd = !!b.nonDistilled;
  const stepsVal = (b.steps && b.steps >= 16) ? b.steps : 35;
  const ndSteps = nd ? stepsVal : undefined;
  const lead = useMemo(() => libChars.find((c) => c.id === b.chars[0]?.charId), [libChars, b.chars]);
  const subjectIds = useMemo(() => resolveSubjects(b, libChars), [b, libChars]);
  const castChars = libChars;   // the cast picker upstream already scopes which characters exist

  // ---- transient per-stage state ----
  const [scenes, setScenes] = useState<Draft[]>([]);          // 3 background candidates
  const [placePreview, setPlacePreview] = useState<string>(""); // with-character composition preview (url)
  const [placeApproved, setPlaceApproved] = useState(false);
  const [vdrafts, setVdrafts] = useState<Draft[]>([]);         // 3 video option drafts
  const [pushKeep, setPushKeep] = useState(0.72);

  const resultClip = bareId(b.clipId);
  const charRefs = lead ? charRefIds(lead, b.chars[0]?.wardrobeId).slice(0, 3) : [];

  // auto-advance to the first incomplete stage on open
  useEffect(() => {
    if (!b.backgroundId) setStep("scene");
    else if (perf && !b.chars.length) setStep("cast");
    else if (!b.clipId) setStep("video");
    else setStep("result");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [b.id]);

  // ---------- STAGE: TYPE ----------
  function setType(t: "perf" | "broll") {
    patch({ renderMode: t === "perf" ? "msr" : "fflf", lipsync: t === "perf" ? b.lipsync : false });
  }

  // ---------- STAGE: SCENE (person-free background still) ----------
  function soloBgPrompt() {
    const backdrop = BACKDROP[b.framing] || BACKDROP.medium;
    return perf
      ? `${backdrop} the setting: ${(b.scene || b.prompt).trim()}. The empty environment only - absolutely no people, no person, no singer, no musician, no figure, no face, no silhouette, unpopulated.`
      : `${backdrop} ${(b.scene || b.prompt).trim()}. A clean scenic shot.`;
  }
  function soloBgNeg() {
    return (perf ? "people, person, man, woman, singer, musician, performer, face, figure, silhouette, crowd, " : "") + (b.negative || "");
  }
  async function genScenes() {
    if (!(b.scene || b.prompt).trim()) { note("Describe the setting first."); return; }
    setBusy(true); note("Generating 3 backgrounds…"); setScenes([]);
    const base = Math.floor(Math.random() * 2_000_000_000);
    try {
      const ds: Draft[] = [];
      for (let i = 0; i < 3; i++) {
        const r = await api.videoStill({ prompt: soloBgPrompt(), negative: soloBgNeg(), width: b.width, height: b.height, seed: base + i }) as { job_id: string };
        ds.push({ jobId: r.job_id, seed: base + i });
      }
      setScenes(ds);
      ds.forEach((d) => waitMedia(d.jobId, (pc, ps) => setScenes((s) => s.map((x) => x.jobId === d.jobId ? { ...x, pct: pc, pass: ps } : x)))
        .then((u) => setScenes((s) => s.map((x) => x.jobId === d.jobId ? { ...x, url: u } : x))).catch(() => {}));
      note("Backgrounds rendering — pick one (it becomes the MSR background, person-free).");
    } catch (e) { note("Background generation failed: " + (e as Error).message); }
    finally { setBusy(false); }
  }
  function pickScene(id: string) { patch({ backgroundId: bareId(id) }); note("Background set."); }

  // ---------- STAGE: CAST (lead + band-in-scene composite) ----------
  function setLead(charId: string) {
    const c = libChars.find((x) => x.id === charId);
    patch({ chars: charId ? [{ charId, wardrobeId: c?.wardrobes?.[0]?.id }] : [] });
  }
  function toggleBand(id: string) {
    const cur = b.bandInScene || [];
    patch({ bandInScene: cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id] });
  }
  // Band recipe (memory: band-shots-were-solo-msr): composite the lead (centre) + named members
  // (gender+role+side) + a mandatory drummer into the scene; that composite IS the MSR background.
  async function composeBand() {
    const bandChars = (b.bandInScene || []).map((id) => libChars.find((c) => c.id === id)).filter(Boolean) as Character[];
    const compChars = [...(lead ? [lead] : []), ...bandChars];
    const compRefs = compChars.map((c) => sceneRefOf(c)).filter(Boolean).slice(0, 3);
    if (!compRefs.length) { note("Pick a lead (and band members) with reference stills first."); return; }
    setBusy(true); note("Composing the band into the scene…");
    try {
      const sides = ["on the left", "on the right", "to one side"];
      const memberDesc = (c: Character) => `a ${[(c.gender || "").trim(), (c.role || "musician").trim()].filter(Boolean).join(" ")}${(c.appearance || "").trim() ? ` (${(c.appearance || "").trim()})` : ""}`;
      const bandShown = bandChars.slice(0, Math.max(0, compRefs.length - (lead ? 1 : 0)));
      const bandWho = bandShown.map((c, i) => `${memberDesc(c)} plays ${sides[i] || "to one side"}`).join("; ");
      const leadPhrase = lead ? `${memberDesc(lead)} stands centre at the microphone singing, the clear focus front and centre` : "the lead singer stands centre at the microphone, the clear focus";
      const bgFrame = ({ close: "a tight, close framing of", medium: "a medium framing of", wide: "a wide establishing shot of" } as Record<string, string>)[b.framing] || "a medium framing of";
      const prompt = `A live band performing together on a stage, ${bgFrame} the whole band: ${leadPhrase}; ${bandWho}; with a full drum kit at the back and a drummer seated behind it. Setting: ${(b.scene || "a stage").trim()}. Photoreal, cinematic. No audience, no crowd.`;
      const r = await api.videoCharStill({ ref_ids: compRefs, prompt, width: b.width, height: b.height }) as { job_id: string };
      const url = await waitMedia(r.job_id, (pc) => note(`Composing… ${pc}%`));
      patch({ backgroundId: bareId(url) });
      note("Band composited — set as the background. The lead is still MSR-anchored on top at render.");
    } catch (e) { note("Compose failed: " + (e as Error).message); }
    finally { setBusy(false); }
  }

  // ---------- STAGE: PLACEMENT (cheap composition gate, with/without character) ----------
  async function previewWithChar() {
    if (!charRefs.length) { note("This shot has no character refs to place."); return; }
    setBusy(true); note("Rendering a placement preview (cheap still)…"); setPlacePreview("");
    try {
      const prompt = `${FRAME_PHRASE[b.framing] || "medium shot"}. ${(b.scene || "").trim()}. The character standing in the scene${b.prompt ? `, ${b.prompt}` : ""}.`;
      const r = await api.videoCharStill({ ref_ids: charRefs, prompt, width: vHalf(b.width), height: vHalf(b.height) }) as { job_id: string };
      const url = await waitMedia(r.job_id, (pc) => note(`Placement preview… ${pc}%`));
      setPlacePreview(url); note("Composition preview ready — approve to render the video, or tweak the scene/cast.");
    } catch (e) { note("Preview failed: " + (e as Error).message); }
    finally { setBusy(false); }
  }

  // ---------- STAGE: VIDEO (seed-hunt → pick → finish) ----------
  function renderMsr(seed: number, w: number, h: number) {
    return api.videoLtxMsr({
      subject_ids: subjectIds, background_id: b.backgroundId,
      audio_id: b.lipsync ? (b.audioId || songAudioId) : undefined, audio_start: b.audioStart, isolate_vocal: false,
      prompt: b.prompt, width: w, height: h, frames: b.frames, fps: b.fps, ref_frames: b.refFrames, seed,
      nondistilled: nd || undefined, steps: ndSteps,
    }) as Promise<{ job_id: string }>;
  }
  // B-roll push-in: opening anchor = the chosen background; closing anchor = a center-crop of it.
  async function fflfAnchors() {
    const crop = await api.videoCropStill({ still_id: b.backgroundId, keep: pushKeep }) as { id: string };
    return { first_id: b.backgroundId, first_kind: "image", last_id: crop.id, last_kind: "image" };
  }
  async function genOptions() {
    if (!b.backgroundId) { note("Pick a background first."); return; }
    if (perf && !subjectIds.length) { note("Add a lead character (Cast) first."); return; }
    if (!b.prompt.trim()) { note("Describe the action (what happens in the shot)."); return; }
    setBusy(true); note("Generating 3 video options (half-res)…"); setVdrafts([]);
    try {
      let ds: Draft[] = [];
      if (perf) {
        const base = Math.floor(Math.random() * 2_000_000_000);
        const hw = vHalf(b.width), hh = vHalf(b.height);
        for (let i = 0; i < 3; i++) { const r = await renderMsr(base + i, hw, hh); ds.push({ jobId: r.job_id, seed: base + i }); }
      } else {
        const a = await fflfAnchors();
        const r = await api.videoLtxFflf({
          mode: "hunt", ...a,
          first_strength: b.fflfFirstStrength ?? 0.7, last_strength: b.fflfLastStrength ?? 0.5,
          prompt: nd ? `${CALM_POS}, ${b.prompt}` : b.prompt, negative: nd ? CALM_NEG : undefined,
          nondistilled: nd || undefined, steps: ndSteps, width: b.width, height: b.height, frames: b.frames, fps: b.fps,
        }) as { base_seed: number; drafts: { job_id: string }[] };
        ds = r.drafts.map((d, i) => ({ jobId: d.job_id, seed: r.base_seed + i }));
      }
      setVdrafts(ds);
      ds.forEach((d) => waitMedia(d.jobId, (pc, ps) => setVdrafts((s) => s.map((x) => x.jobId === d.jobId ? { ...x, pct: pc, pass: ps } : x)))
        .then((u) => setVdrafts((s) => s.map((x) => x.jobId === d.jobId ? { ...x, url: u } : x))).catch(() => {}));
      note("Options rendering — they play; pick the best to finish at full res.");
    } catch (e) { note("Generation failed: " + (e as Error).message); }
    finally { setBusy(false); }
  }
  async function finishOption(seed: number) {
    setBusy(true); note("Finishing at full res" + (perf && b.lipsync ? " with lip-sync" : "") + "…");
    try {
      let clipId: string;
      if (perf) {
        const r = await renderMsr(seed, b.width, b.height);
        clipId = bareId(await waitMedia(r.job_id, (pc, ps) => note(`Finishing… ${pc}%${ps > 1 ? ` · pass ${ps}` : ""}`)));
      } else {
        const a = await fflfAnchors();
        const p: Record<string, unknown> = {
          mode: "finish", stage1_seed: seed, ...a,
          first_strength: b.fflfFirstStrength ?? 0.7, last_strength: b.fflfLastStrength ?? 0.5,
          prompt: nd ? `${CALM_POS}, ${b.prompt}` : b.prompt, negative: nd ? CALM_NEG : undefined,
          nondistilled: nd || undefined, steps: ndSteps, width: b.width, height: b.height, frames: b.frames, fps: b.fps,
        };
        const r = await api.videoLtxFflf(p) as { job_id: string };
        clipId = bareId(await waitMedia(r.job_id, (pc, ps) => note(`Finishing… ${pc}%${ps > 1 ? ` · pass ${ps}` : ""}`)));
      }
      const take: Take = { id: rid(), clipId, stage1Seed: seed, draft: false, label: `seed ${seed}` };
      patch({ clipId, clipVariants: [...(b.clipVariants || []), clipId], pieces: [{ id: rid(), lane: perf ? "msr" : "fflf", label: "Base shot", takes: [take], selectedTakeId: take.id }] });
      setVdrafts([]); setStep("result"); note("Finished — this take is on the timeline. Press play.");
    } catch (e) { note("Finish failed: " + (e as Error).message); }
    finally { setBusy(false); }
  }

  // ---------- STAGE: RESULT (extend B-roll, use on timeline) ----------
  const pieces = b.pieces || [];
  async function extendBroll(seed?: number) {
    const lastClip = bareId((pieces[pieces.length - 1]?.takes.find((t) => t.id === pieces[pieces.length - 1]?.selectedTakeId) || pieces[pieces.length - 1]?.takes[0])?.clipId) || resultClip;
    if (!lastClip) { note("Render the shot before extending."); return; }
    setBusy(true); note(seed === undefined ? "Hunting an extension (3 drafts)…" : "Finishing extension…");
    try {
      const common = {
        first_id: lastClip, first_kind: "video", first_frames: FFLF_TAIL, first_skip: Math.max(0, b.frames - FFLF_TAIL),
        last_id: b.backgroundId, last_kind: "image",
        first_strength: b.fflfFirstStrength ?? 0.7, last_strength: b.fflfLastStrength ?? 0.5,
        prompt: nd ? `${CALM_POS}, ${b.prompt}` : b.prompt, negative: nd ? CALM_NEG : undefined,
        nondistilled: nd || undefined, steps: ndSteps, width: b.width, height: b.height, frames: b.frames, fps: b.fps,
      };
      if (seed === undefined) {
        const r = await api.videoLtxFflf({ mode: "hunt", ...common }) as { base_seed: number; drafts: { job_id: string }[] };
        const ds = r.drafts.map((d, i) => ({ jobId: d.job_id, seed: r.base_seed + i }));
        setVdrafts(ds);
        ds.forEach((d) => waitMedia(d.jobId, (pc, ps) => setVdrafts((s) => s.map((x) => x.jobId === d.jobId ? { ...x, pct: pc, pass: ps } : x)))
          .then((u) => setVdrafts((s) => s.map((x) => x.jobId === d.jobId ? { ...x, url: u } : x))).catch(() => {}));
        note("Extension drafts rendering — pick one to append.");
        return;
      }
      const r = await api.videoLtxFflf({ mode: "finish", stage1_seed: seed, ...common }) as { job_id: string };
      const clipId = bareId(await waitMedia(r.job_id, (pc, ps) => note(`Finishing… ${pc}%${ps > 1 ? ` · pass ${ps}` : ""}`)));
      const take: Take = { id: rid(), clipId, stage1Seed: seed, draft: false, label: `ext ${pieces.length + 1}` };
      const next = [...pieces, { id: rid(), lane: "fflf" as const, label: `Extend ${pieces.length + 1}`, takes: [take], selectedTakeId: take.id }];
      const clips = next.map((p) => bareId((p.takes.find((t) => t.id === p.selectedTakeId) || p.takes[0])?.clipId)).filter(Boolean) as string[];
      const asm = await api.videoAssembleChain({ clips, frames: b.frames, fps: b.fps, tail: FFLF_TAIL, width: b.width, height: b.height, transition: b.seamXfade || 0 }) as { id: string };
      patch({ pieces: next, assembledId: asm.id, clipId: asm.id });
      setVdrafts([]); note(`Extended — ${clips.length} pieces assembled into one continuous take.`);
    } catch (e) { note("Extend failed: " + (e as Error).message); }
    finally { setBusy(false); }
  }

  // ---- stage completion (for the rail ticks) ----
  const done: Record<string, boolean> = {
    type: true, scene: !!b.backgroundId, cast: !!b.chars.length,
    placement: placeApproved || !!b.clipId, video: !!b.clipId, result: !!b.clipId,
  };
  const stageLabel: Record<string, string> = {
    type: "Shot type", scene: "Scene", cast: "Cast", placement: "Placement", video: "Video", result: "Result",
  };

  return (
    <div className="se-root">
      <style>{`
        .se-root{display:grid;grid-template-columns:200px 1fr;gap:20px;}
        .se-rail{display:flex;flex-direction:column;gap:4px;}
        .se-step{display:flex;align-items:center;gap:9px;text-align:left;border:1px solid transparent;background:transparent;color:var(--color-muted);border-radius:9px;padding:9px 11px;font-size:13px;cursor:pointer;}
        .se-step:hover{background:var(--color-panel2);color:var(--color-ink);}
        .se-step.on{background:var(--color-panel2);border-color:var(--color-accent);color:var(--color-ink);font-weight:600;}
        .se-dot{display:flex;align-items:center;justify-content:center;width:20px;height:20px;border-radius:999px;font-size:11px;border:1px solid var(--color-line);flex:none;}
        .se-dot.ok{background:var(--color-accent2);border-color:var(--color-accent2);color:#0c0e13;}
        .se-card{border:1px solid var(--color-line);background:var(--color-panel);border-radius:14px;padding:18px;}
        .se-h{font-size:15px;font-weight:600;color:var(--color-ink);margin:0 0 3px;}
        .se-hint{font-size:11px;color:var(--color-muted);margin:0 0 14px;}
        .se-grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;}
        .se-tile{border:1px solid var(--color-line);background:var(--color-panel2);border-radius:10px;overflow:hidden;}
        .se-thumb{aspect-ratio:16/9;background:#0d0f13;display:flex;align-items:center;justify-content:center;}
        .se-thumb img{height:100%;width:100%;object-fit:cover;}
        .se-spin{font-size:10px;color:var(--color-muted);}
        .se-typebtn{flex:1;border:1px solid var(--color-line);background:var(--color-panel2);border-radius:12px;padding:16px;cursor:pointer;text-align:left;color:var(--color-muted);}
        .se-typebtn.on{border-color:var(--color-accent);color:var(--color-ink);background:#2a1c19;}
        .se-typebtn b{display:block;color:var(--color-ink);font-size:14px;margin-bottom:4px;}
        .se-foot{display:flex;align-items:center;gap:10px;margin-top:16px;}
      `}</style>

      {/* left rail */}
      <div>
        <GhostButton onClick={onClose}>{"← Timeline"}</GhostButton>
        <div className="mt-3 mb-2 text-sm font-semibold text-[var(--color-ink)]">Shot {idx + 1}</div>
        <div className="se-rail">
          {STAGES.map((s) => (
            <button key={s} className={`se-step ${step === s ? "on" : ""}`} onClick={() => setStep(s)}>
              <span className={`se-dot ${done[s] ? "ok" : ""}`}>{done[s] ? "✓" : STAGES.indexOf(s) + 1}</span>
              {stageLabel[s]}
            </button>
          ))}
        </div>
      </div>

      {/* active stage */}
      <div className="flex flex-col gap-3">
        {/* ---------------- TYPE ---------------- */}
        {step === "type" && (
          <div className="se-card">
            <div className="se-h">What kind of shot is this?</div>
            <p className="se-hint">Performance shots carry a character (identity + optional lip-sync via MSR). B-roll is scenic — a calm camera move with no people (FFLF push-in).</p>
            <div className="flex gap-3">
              <button className={`se-typebtn ${perf ? "on" : ""}`} onClick={() => setType("perf")}>
                <b>Performance</b>A character performs — singer/band, identity-locked, can lip-sync to the song.
              </button>
              <button className={`se-typebtn ${!perf ? "on" : ""}`} onClick={() => setType("broll")}>
                <b>B-roll (scenic)</b>A landscape / object / mood shot with a slow dolly-in. No people.
              </button>
            </div>
            <div className="se-foot"><PrimaryButton onClick={() => setStep("scene")}>Next: Scene →</PrimaryButton></div>
          </div>
        )}

        {/* ---------------- SCENE ---------------- */}
        {step === "scene" && (
          <div className="se-card flex flex-col gap-3">
            <div>
              <div className="se-h">Scene — the background</div>
              <p className="se-hint">{perf ? "A person-free environment still. It becomes the MSR background, so it must contain no people — the character is anchored on top at render." : "The scenic still this shot dollies into."}</p>
            </div>
            <Field label="Setting (environment only)">
              <textarea className={inp} rows={3} value={b.scene} onChange={(e) => patch({ scene: e.target.value })}
                placeholder="a ruined ashen cathedral at dusk, shafts of light through broken stained glass…" />
            </Field>
            <div className="flex items-end gap-3">
              <label className="text-[11px] text-[var(--color-muted)]">Framing
                <select className={inp} style={{ display: "block", marginTop: 4 }} value={b.framing} onChange={(e) => patch({ framing: e.target.value })}>
                  <option value="close">close</option><option value="medium">medium</option><option value="wide">wide</option>
                </select>
              </label>
              <PrimaryButton onClick={genScenes} disabled={busy}>Generate 3 backgrounds</PrimaryButton>
              <div className="flex-1" />
              <div className="min-w-[200px]">
                <span className="text-[11px] text-[var(--color-muted)]">or use a library still:</span>
                <StillPick value="" set={pickScene} stills={stills} placeholder="— pick from library —" />
              </div>
            </div>
            {scenes.length > 0 && (
              <div className="se-grid3">
                {scenes.map((d) => (
                  <div key={d.jobId} className="se-tile">
                    <div className="se-thumb">{d.url ? <img src={d.url} alt="" /> : <span className="se-spin">{d.pct ? `rendering… ${d.pct}%` : "queued…"}</span>}</div>
                    <div className="flex items-center justify-between px-2 py-1.5">
                      <span className="text-[11px] text-[var(--color-muted)]">seed {d.seed}</span>
                      <GhostButton onClick={() => pickScene(bareId(d.url!))} disabled={!d.url || busy}>Use this</GhostButton>
                    </div>
                  </div>
                ))}
              </div>
            )}
            {b.backgroundId && (
              <div className="flex items-center gap-3 rounded-lg border border-[var(--color-line)] p-2">
                <img src={`/api/media/${b.backgroundId}`} alt="" className="h-16 w-28 rounded object-cover" />
                <span className="text-[11px] text-[var(--color-accent2)]">✓ background set</span>
                <div className="flex-1" />
                <PrimaryButton onClick={() => setStep(perf ? "cast" : "video")}>Next: {perf ? "Cast" : "Video"} →</PrimaryButton>
              </div>
            )}
          </div>
        )}

        {/* ---------------- CAST ---------------- */}
        {step === "cast" && perf && (
          <div className="se-card flex flex-col gap-3">
            <div>
              <div className="se-h">Cast</div>
              <p className="se-hint">One lead is identity-locked via MSR. Add band members to composite them into the background (a drummer is always added for a band shot).</p>
            </div>
            <Field label="Lead (MSR-anchored)">
              <select className={inp} value={b.chars[0]?.charId || ""} onChange={(e) => setLead(e.target.value)}>
                <option value="">— pick the lead —</option>
                {castChars.map((c) => <option key={c.id} value={c.id}>{c.name}{c.role ? ` · ${c.role}` : ""}</option>)}
              </select>
            </Field>
            {lead && !charRefs.length && <span className="text-[10px] text-amber-400">This character has no reference stills — add them in the Characters tab, or identity won't hold.</span>}
            <div>
              <div className="mb-1 text-[11px] text-[var(--color-muted)]">Band in scene (composited into the background)</div>
              <div className="flex flex-wrap gap-1.5">
                {castChars.filter((c) => c.id !== b.chars[0]?.charId).map((c) => {
                  const on = (b.bandInScene || []).includes(c.id);
                  return <button key={c.id} onClick={() => toggleBand(c.id)}
                    className={`rounded-full border px-2 py-0.5 text-[11px] ${on ? "border-[var(--color-accent2)] bg-[#2a1c19] text-[var(--color-ink)]" : "border-[var(--color-line)] text-[var(--color-muted)]"}`}>
                    {c.name} {on ? "✓" : "+"}</button>;
                })}
              </div>
            </div>
            {(b.bandInScene || []).length > 0 && (
              <div className="flex items-center gap-2">
                <PrimaryButton onClick={composeBand} disabled={busy || !lead}>Compose band into background</PrimaryButton>
                <span className="text-[10px] text-[var(--color-muted)]">replaces the background with the band composite (lead centre + members + drummer)</span>
              </div>
            )}
            <div className="se-foot">
              <PrimaryButton onClick={() => setStep("placement")} disabled={!b.chars.length}>Next: Placement →</PrimaryButton>
              <span className="text-[11px] text-[var(--color-muted)]">{subjectIds.length} subject ref{subjectIds.length === 1 ? "" : "s"} resolved</span>
            </div>
          </div>
        )}

        {/* ---------------- PLACEMENT ---------------- */}
        {step === "placement" && perf && (
          <div className="se-card flex flex-col gap-3">
            <div>
              <div className="se-h">Placement preview</div>
              <p className="se-hint">Check the composition with a cheap still BEFORE spending GPU on the video. Background-only is the scene; with-character composites the lead in to verify scale/placement.</p>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="se-tile">
                <div className="se-thumb">{b.backgroundId ? <img src={`/api/media/${b.backgroundId}`} alt="" /> : <span className="se-spin">no background</span>}</div>
                <div className="px-2 py-1.5 text-[11px] text-[var(--color-muted)]">Background only</div>
              </div>
              <div className="se-tile">
                <div className="se-thumb">{placePreview ? <img src={placePreview} alt="" /> : <span className="se-spin">{busy ? "rendering…" : "— not rendered —"}</span>}</div>
                <div className="flex items-center justify-between px-2 py-1.5">
                  <span className="text-[11px] text-[var(--color-muted)]">With character</span>
                  <GhostButton onClick={previewWithChar} disabled={busy || !charRefs.length}>Render preview</GhostButton>
                </div>
              </div>
            </div>
            <div className="se-foot">
              <PrimaryButton onClick={() => { setPlaceApproved(true); setStep("video"); }}>Looks good — to Video →</PrimaryButton>
              <GhostButton onClick={() => setStep("scene")}>Back to Scene</GhostButton>
            </div>
          </div>
        )}

        {/* ---------------- VIDEO ---------------- */}
        {step === "video" && (
          <div className="se-card flex flex-col gap-3">
            <div>
              <div className="se-h">Video options</div>
              <p className="se-hint">Describe the action, then generate 3 cheap options and pick the best — it's finished at full res.</p>
            </div>
            <Field label="Action (what happens in the shot)">
              <textarea className={inp} rows={3} value={b.prompt} onChange={(e) => patch({ prompt: e.target.value })}
                placeholder={perf ? "she sings with intensity, the camera slowly pushing in…" : "the camera slowly dollies in across the still water…"} />
            </Field>
            <div className="flex flex-wrap items-center gap-4">
              {perf && (
                <label className="flex items-center gap-2 text-xs text-[var(--color-ink)]">
                  <input type="checkbox" checked={b.lipsync} onChange={(e) => patch({ lipsync: e.target.checked })} /> Lip-sync to the song
                </label>
              )}
              {perf && b.lipsync && (
                <div className="min-w-[200px]"><Field label="Vocal track"><StillPick value={b.audioId || songAudioId} set={(id) => patch({ audioId: id })} stills={audios} thumb={false} placeholder="— song —" /></Field></div>
              )}
              {!perf && <Num label="Push-in (keep)" value={pushKeep} set={setPushKeep} step={0.05} w="w-20" min={0.3} max={0.95} />}
            </div>
            {/* advanced render settings */}
            <details className="rounded-md border border-dashed border-[var(--color-line)] p-2">
              <summary className="cursor-pointer text-[11px] text-[var(--color-muted)]">Advanced render settings</summary>
              <div className="mt-2 space-y-2">
                <label className="flex items-start gap-2 text-[11px] text-[var(--color-muted)]">
                  <input type="checkbox" className="mt-0.5" checked={nd} onChange={(e) => patch({ nonDistilled: e.target.checked })} />
                  <span><span className="font-medium text-[var(--color-ink)]">Non-distilled (dev model)</span> — controllable, slower motion (no timelapse). Slower render; recommended for B-roll.</span>
                </label>
                {nd && <div className="flex items-center gap-2"><Num label="Steps" value={stepsVal} set={(n) => patch({ steps: Math.round(n) })} step={1} w="w-24" min={16} max={60} /><span className="text-[10px] text-[var(--color-muted)]">30–50 typical</span></div>}
                <div className="flex flex-wrap gap-2">
                  <Num label="W" value={b.width} set={(n) => patch({ width: n })} step={32} w="w-20" />
                  <Num label="H" value={b.height} set={(n) => patch({ height: n })} step={32} w="w-20" />
                  <Num label="Frames" value={b.frames} set={(n) => patch({ frames: n })} step={8} w="w-20" />
                  <Num label="FPS" value={b.fps} set={(n) => patch({ fps: n })} step={1} w="w-16" />
                </div>
              </div>
            </details>
            <div><PrimaryButton onClick={genOptions} disabled={busy}>{vdrafts.length ? "Re-generate 3 options" : "Generate 3 options"}</PrimaryButton></div>
            {vdrafts.length > 0 && (
              <div className="se-grid3">
                {vdrafts.map((d) => (
                  <div key={d.jobId} className="se-tile">
                    <div className="se-thumb">{d.url ? <VideoTile src={d.url} /> : <span className="se-spin">{d.pct ? `rendering… ${d.pct}% · pass ${Math.min(d.pass || 1, 2)} of 2` : "queued…"}</span>}</div>
                    <div className="flex items-center justify-between px-2 py-1.5">
                      <span className="text-[11px] text-[var(--color-muted)]">seed {d.seed}</span>
                      <GhostButton onClick={() => finishOption(d.seed)} disabled={busy || !d.url}>Finish</GhostButton>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ---------------- RESULT ---------------- */}
        {step === "result" && (
          <div className="se-card flex flex-col gap-3">
            <div className="se-h">Result</div>
            {resultClip ? (
              <video key={resultClip} src={`/api/media/${resultClip}`} controls autoPlay loop playsInline className="w-full rounded-lg bg-black" style={{ maxHeight: 440 }} />
            ) : (
              <p className="se-hint">No clip yet — generate one in the Video stage.</p>
            )}
            {resultClip && (
              <>
                <div className="se-foot">
                  <PrimaryButton onClick={onClose}>Use on timeline ✓</PrimaryButton>
                  <GhostButton onClick={() => { setVdrafts([]); setStep("video"); }}>Re-roll options</GhostButton>
                  {!perf && <GhostButton onClick={() => extendBroll()} disabled={busy}>+ Extend (3 drafts)</GhostButton>}
                  {pieces.length > 1 && <span className="text-[10px] text-[var(--color-accent2)]">✓ {pieces.length} pieces assembled</span>}
                </div>
                {!perf && vdrafts.length > 0 && (
                  <div className="se-grid3">
                    {vdrafts.map((d) => (
                      <div key={d.jobId} className="se-tile">
                        <div className="se-thumb">{d.url ? <VideoTile src={d.url} /> : <span className="se-spin">{d.pct ? `rendering… ${d.pct}%` : "queued…"}</span>}</div>
                        <div className="flex items-center justify-between px-2 py-1.5">
                          <span className="text-[11px] text-[var(--color-muted)]">seed {d.seed}</span>
                          <GhostButton onClick={() => extendBroll(d.seed)} disabled={busy || !d.url}>Append</GhostButton>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {status && <p className="text-[11px] text-[var(--color-accent2)]">{status}</p>}
      </div>
    </div>
  );
}
