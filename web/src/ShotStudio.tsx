import { useEffect, useRef, useState } from "react";
import { api, type LibItem } from "./api";
import { Field, inp, GhostButton, PrimaryButton, rid } from "./ui";
import { StillPick, Num } from "./mvui";
import { LtxDirectorEditor, type LtxDirectorHandle } from "./LtxDirectorEditor";
import { type Block, type ChainPiece, type Take, type RenderMode } from "./mvmodel";

// ---------------------------------------------------------------------------------------------
// Shot Studio — the per-segment editor. One timeline shot opens here as a continuous take built
// from ORDERED chain pieces: piece 0 is the base shot; each Extend appends an FFLF piece driven by
// the previous piece's last 33 frames. Every piece keeps every take it rolled (seed-hunt drafts +
// finished multiroll variants) as cards, so you pick/reroll without losing work.
// FFLF graph = stock LTXVAddGuide (build_ltx_fflf); MSR/keyframe segments will gain the LTXDirector
// editor controls in a later phase. This phase wires the FFLF hunt -> finish -> extend loop + cards.
// ---------------------------------------------------------------------------------------------

const poster = (id: string) => `/api/media/${id}#t=0.5`;        // seek a frame so <video> paints a poster
const FFLF_TAIL = 33;                                            // foxydits' extend anchor length
// A clip reference must be a BARE job id (the timeline/assembly + /api/media build the URL themselves).
// waitMedia resolves to a /api/media/<id> URL, and older takes stored that whole URL - strip it so both
// new and existing data yield a bare id (else srcs double up to /api/media//api/media/<id> = empty box).
const bareId = (s?: string) => (s ? s.split("/").pop()!.split("?")[0].split("#")[0] : "");
// FFLF "calm motion" (B-roll): LTX renders water/clouds as a fast timelapse by default, and the stock
// FFLF negative actually pushes FOR motion ("no movement, still frame"). NAG (scale 50) makes this
// negative strong, so an explicit anti-timelapse negative + slow positive cues tame the speed.
const CALM_NEG = "timelapse, time-lapse, hyperlapse, fast motion, sped-up footage, fast moving clouds, " +
  "fast moving water, racing waves, flickering, strobing, blurry, low quality, watermark, subtitles, music";
const CALM_POS = "slow gentle cinematic motion, calm, slow-moving water, slowly drifting clouds";

// Poll a video job to its media url (ui.waitJob only resolves audio jobs).
// onPct reports (pct, pass). A multi-stage render (sample -> refine/decode) emits a fresh 0->100 per
// stage; we detect the restart (big drop) and bump `pass` so the UI shows "… · pass 2" instead of a
// confusing reset to 0.
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
        if (pct < lastPct - 20) pass += 1;     // progress restarted -> next render pass
        lastPct = pct;
        onPct(pct, pass);
      }
    }, 1500);
  });
}

function ThumbVideo({ id, className = "" }: { id: string; className?: string }) {
  return <video src={poster(id)} muted playsInline preload="metadata"
    className={`h-full w-full object-cover ${className}`} />;
}

// A draft preview that actually PLAYS (looping, muted) so you can judge the motion when picking a
// seed-hunt draft - plus controls to scrub/pause. (ThumbVideo only paints a static poster frame.)
function PreviewVideo({ src, className = "" }: { src: string; className?: string }) {
  return <video src={src} muted loop autoPlay playsInline controls preload="auto"
    className={`h-full w-full object-contain ${className}`} />;
}

export function ShotStudio({ block: b, idx, patch, stills, audios, songAudioId, onClose }: {
  block: Block; idx: number; patch: (p: Partial<Block>) => void;
  stills: LibItem[]; audios: LibItem[]; songAudioId: string;
  onClose: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const [pushKeep, setPushKeep] = useState(0.72);   // FFLF push-in: fraction of the opening still kept in the end crop
  const [viewResult, setViewResult] = useState<boolean>(!!b.clipId);   // editor view: false = edit inputs (keyframes), true = play the rendered clip
  const pieces = b.pieces || [];
  // transient hunt state: the 3 half-res drafts for the piece currently being hunted
  const [hunt, setHunt] = useState<{ kind: "fflf" | "video"; pieceLabel: string; forExtend: boolean; drafts: { jobId: string; seed: number; url?: string; pct?: number; pass?: number }[] } | null>(null);
  const [status, setStatus] = useState("");
  // "Add keyframe still" panel: generate (seed-hunt half-res -> full on pick) or pick from library,
  // then inject into the editor timeline as a keyframe via the editor's own add-image path.
  const edRef = useRef<LtxDirectorHandle>(null);
  const [kfPrompt, setKfPrompt] = useState("");
  const [kfFrame, setKfFrame] = useState(0);
  const [kfBusy, setKfBusy] = useState(false);
  const [kfDrafts, setKfDrafts] = useState<{ jobId: string; seed: number; url?: string; pct?: number; pass?: number }[]>([]);
  const [kfUseChar, setKfUseChar] = useState(false);   // generate from the shot's character refs (identity)
  const charRefs = (b.subjectIds || []).filter(Boolean).slice(0, 3);
  const [loras, setLoras] = useState<string[]>([]);    // character/ID LoRAs on the box (identity instead of MSR)
  useEffect(() => { api.videoLoras().then((l) => setLoras(Array.isArray(l) ? l as string[] : [])).catch(() => {}); }, []);
  // Repair shots finished BEFORE the clip-writeback fix: if a single piece exists but the block's clip
  // doesn't point at its selected take, sync it so the timeline plays through. Runs once per shot open.
  useEffect(() => {
    const ps = b.pieces || [];
    if (ps.length === 1) {
      const t = ps[0].takes.find((x) => x.id === ps[0].selectedTakeId) || ps[0].takes[0];
      const cid = bareId(t?.clipId);
      if (cid && b.clipId !== cid) patch({ clipId: cid });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [b.id]);

  const setPieces = (next: ChainPiece[]) => {
    // Keep the BLOCK's playable clip in sync with the pieces so the MV Studio timeline + assembly use it
    // (a finished take was previously stored only in pieces, so the timeline never saw it). Single piece
    // -> the block clip IS the selected take. (A multi-piece chain needs assembly into one clip; handled
    // separately - don't clobber an existing assembled clipId here.)
    const upd: Partial<Block> = { pieces: next };
    if (next.length === 1) {
      const t = next[0].takes.find((x) => x.id === next[0].selectedTakeId) || next[0].takes[0];
      const cid = bareId(t?.clipId);
      if (cid) upd.clipId = cid;
    }
    patch(upd);
  };
  const selectedTakeOf = (p: ChainPiece) => p.takes.find((t) => t.id === p.selectedTakeId) || p.takes[0];
  const lastPiece = pieces[pieces.length - 1];
  const lastClip = lastPiece && bareId(selectedTakeOf(lastPiece)?.clipId);
  // the rendered output to preview at the top: the block's clip (single take or assembled chain), else
  // fall back to a single piece's selected take (covers the case where clipId hasn't synced yet).
  const resultClip = bareId(b.clipId) || (pieces.length === 1 ? bareId(selectedTakeOf(pieces[0])?.clipId) : "");

  // base-shot anchors live on the block; extends derive their first anchor from the prior tail.
  const fflfReady = !!parseKf(b.director?.timeline_data).firstName;   // FFLF anchors come from the timeline now

  function note(s: string) { setStatus(s); }
  // Non-distilled (dev) model = the only way to get controllable/slower LTX motion (distilled has none).
  // Works on every path now. `nd` = use the dev model; `cine` = also add B-roll slow cues (non-lip-sync).
  const nd = !!b.nonDistilled;
  // Non-distilled steps default = 35. Treat any sub-16 saved value as the default (8 is a distilled step
  // count; older builds could leave a stale 8 here). 30-50 is the real range.
  const stepsVal = (b.steps && b.steps >= 16) ? b.steps : 35;
  const ndSteps = nd ? stepsVal : undefined;
  const cine = nd && !b.lipsync;
  const fflfPrompt = () => cine ? `${CALM_POS}, ${b.prompt}` : b.prompt;
  const fflfNeg = () => cine ? CALM_NEG : undefined;                                  // undefined -> builder default

  // ---- FFLF push-in: center-crop the timeline's OPENING image and drop the crop back into the timeline
  // as the LAST keyframe. Both anchors then come from the timeline (no side fields), and the shot dollies
  // in between two person-free pinned frames (no hallucinated figures, controlled speed). Crop is done in
  // the browser from the segment's stored data URL - no GPU, nothing leaves the timeline. ----
  async function makePushIn() {
    const td = b.director?.timeline_data;
    let firstImg: { imageB64?: string; imageFile?: string } | undefined;
    try {
      firstImg = ((JSON.parse(td || "{}").segments || []) as { type?: string; imageB64?: string; imageFile?: string; start?: number }[])
        .filter((s) => s.type === "image" && (s.imageB64 || s.imageFile)).sort((a, c) => (a.start || 0) - (c.start || 0))[0];
    } catch { /* ignore */ }
    const src = firstImg?.imageB64 || (firstImg?.imageFile ? `/api/comfy/view?filename=${encodeURIComponent(firstImg.imageFile)}&type=input` : "");
    if (!src) { note("Add the opening still to the timeline first."); return; }
    if (!edRef.current) { note("Editor not ready."); return; }
    setBusy(true); note("Building push-in (cropping the opening still)…");
    try {
      const img = new Image();
      img.crossOrigin = "anonymous";
      await new Promise<void>((res, rej) => { img.onload = () => res(); img.onerror = () => rej(new Error("could not load the opening still")); img.src = src; });
      const keep = Math.max(0.3, Math.min(0.95, pushKeep));
      const cw = Math.round(img.naturalWidth * keep), ch = Math.round(img.naturalHeight * keep);
      const sx = Math.round((img.naturalWidth - cw) / 2), sy = Math.round((img.naturalHeight - ch) / 2);
      const cv = document.createElement("canvas");
      cv.width = img.naturalWidth; cv.height = img.naturalHeight;      // back to source res (matches the first anchor)
      cv.getContext("2d")!.drawImage(img, sx, sy, cw, ch, 0, 0, cv.width, cv.height);
      const blob: Blob = await new Promise((res) => cv.toBlob((b2) => res(b2!), "image/png"));
      const file = new File([blob], `pushin_${Date.now()}.png`, { type: "image/png" });
      edRef.current.addImage(file, Math.max(0, b.frames - 1));        // place as the LAST keyframe
      note("Push-in built — the crop is now the last keyframe in the timeline. Seed-hunt to render the dolly-in.");
    } catch (e) { note("Push-in build failed: " + (e as Error).message); }
    finally { setBusy(false); }
  }

  // ---- seed-hunt: 3 half-res drafts for a new piece (base, or an extend off the last tail) ----
  async function runHunt(forExtend: boolean) {
    const kf = parseKf(b.director?.timeline_data);   // anchors come from the timeline's image keyframes
    if (!forExtend && !kf.firstName) { note("Add the opening still to the timeline first (Generate / Add keyframe still)."); return; }
    if (forExtend && !lastClip) { note("Render the base shot before extending."); return; }
    setBusy(true); note(forExtend ? "Hunting extension (3 drafts)…" : "Hunting base shot (3 drafts)…");
    try {
      const p: Record<string, unknown> = {
        mode: "hunt",
        first_strength: b.fflfFirstStrength ?? 0.7, last_strength: b.fflfLastStrength ?? 0.5,
        prompt: fflfPrompt(), negative: fflfNeg(), nondistilled: nd || undefined, steps: ndSteps,
        width: b.width, height: b.height, frames: b.frames, fps: b.fps,
      };
      if (forExtend) {                                       // extend: video tail of the last clip -> back to the opening still
        p.first_id = lastClip; p.first_kind = "video"; p.first_frames = FFLF_TAIL; p.first_skip = Math.max(0, b.frames - FFLF_TAIL);
        p.last_name = kf.firstName;
      } else {                                               // base: first (+ optional last) image keyframe from the timeline
        p.first_name = kf.firstName;
        if (kf.lastName) p.last_name = kf.lastName;          // else FFLF defaults last==first (a slow static push)
      }
      const r = await api.videoLtxFflf(p) as { base_seed: number; drafts: { job_id: string }[] };
      const drafts = r.drafts.map((d, i) => ({ jobId: d.job_id, seed: r.base_seed + i }));
      setHunt({ kind: "fflf", pieceLabel: forExtend ? `extend ${pieces.length + 1}` : "base", forExtend, drafts });
      // resolve each draft's poster as it finishes
      drafts.forEach((d) => waitMedia(d.jobId, (pc, ps) =>
        setHunt((h) => h ? { ...h, drafts: h.drafts.map((x) => x.jobId === d.jobId ? { ...x, pct: pc, pass: ps } : x) } : h))
        .then((u) => setHunt((h) => h ? { ...h, drafts: h.drafts.map((x) => x.jobId === d.jobId ? { ...x, url: u } : x) } : h))
        .catch(() => {}));
      note("Drafts rendering — pick one to finish (drafts are video-only; lip-sync is added on finish).");
    } catch (e) { note("Hunt failed: " + (e as Error).message); }
    finally { setBusy(false); }
  }

  // can we seed-hunt video? fflf needs an anchor; msr/keyframe need the editor timeline.
  const canHuntVideo = b.renderMode === "fflf" ? fflfReady : !!b.director?.timeline_data;
  // pull the first/last keyframe image filenames out of the editor's timeline_data (for MSR keyframe pins)
  function parseKf(td?: string): { firstName?: string; lastName?: string } {
    try {
      const segs = ((JSON.parse(td || "{}").segments || []) as { type?: string; imageFile?: string; start?: number }[])
        .filter((s) => s.type === "image" && s.imageFile).sort((a, c) => (a.start || 0) - (c.start || 0));
      return { firstName: segs[0]?.imageFile, lastName: segs.length > 1 ? segs[segs.length - 1].imageFile : undefined };
    } catch { return {}; }
  }
  const vHalf = (n: number) => Math.max(256, Math.round(n / 2 / 32) * 32);
  // render ONE video take for this shot at (w,h)+seed, dispatched by mode — used at half-res for the
  // hunt drafts and at full-res on pick.
  function renderVideo(seed: number, w: number, h: number): Promise<{ job_id: string }> {
    if (b.renderMode === "msr") {
      const kf = parseKf(b.director!.timeline_data);   // MSR identity + lip-sync + the editor's keyframe(s)
      return api.videoLtxMsr({
        subject_ids: b.subjectIds, background_id: b.backgroundId,
        audio_id: b.lipsync ? (b.audioId || songAudioId) : undefined, audio_start: b.audioStart, isolate_vocal: false,
        keyframe_first_name: kf.firstName, keyframe_last_name: kf.lastName,
        prompt: b.prompt, width: w, height: h, frames: b.frames, fps: b.fps, ref_frames: b.refFrames, seed,
        nondistilled: nd || undefined, steps: ndSteps,
      }) as Promise<{ job_id: string }>;
    }
    return api.videoLtxKeyframe({ ...b.director, char_lora: b.charLora, char_strength: b.charStrength, width: w, height: h, frames: b.frames, fps: b.fps, seed, nondistilled: nd || undefined, steps: ndSteps }) as Promise<{ job_id: string }>;
  }
  // ---- video seed-hunt off the EDITOR TIMELINE: 3 HALF-RES drafts -> pick -> finish at full ----
  async function huntVideo() {
    if (b.renderMode === "fflf") return runHunt(false);   // fflf: its own anchor hunt
    if (!b.director?.timeline_data) { note("Build a timeline in the editor first (add a keyframe)."); return; }
    setBusy(true); note("Seed-hunting video (3 half-res drafts)…");
    const base = Math.floor(Math.random() * 2_000_000_000);
    const hw = vHalf(b.width), hh = vHalf(b.height);
    try {
      const drafts: { jobId: string; seed: number; url?: string }[] = [];
      for (let i = 0; i < 3; i++) {
        const r = await renderVideo(base + i, hw, hh);
        drafts.push({ jobId: r.job_id, seed: base + i });
      }
      setHunt({ kind: "video", pieceLabel: "video", forExtend: false, drafts });
      drafts.forEach((d) => waitMedia(d.jobId, (pc, ps) =>
        setHunt((h) => h ? { ...h, drafts: h.drafts.map((x) => x.jobId === d.jobId ? { ...x, pct: pc, pass: ps } : x) } : h))
        .then((u) => setHunt((h) => h ? { ...h, drafts: h.drafts.map((x) => x.jobId === d.jobId ? { ...x, url: u } : x) } : h))
        .catch(() => {}));
      note("Half-res drafts rendering — pick one to finish at full res.");
    } catch (e) { note("Video hunt failed: " + (e as Error).message); }
    finally { setBusy(false); }
  }

  // ---- finish a chosen draft at full res (+ lip-sync) -> a Take on a (new) piece ----
  async function finishDraft(stage1Seed: number) {
    if (!hunt) return;
    if (hunt.kind === "video") {   // drafts are half-res -> re-render the picked seed at full res
      setBusy(true); note("Finishing video at full res…");
      try {
        const r = await renderVideo(stage1Seed, b.width, b.height);
        const clipId = bareId(await waitMedia(r.job_id, (pc, ps) => note(`Finishing… ${pc}%${ps>1?` · pass ${ps}`:""}`)));
        const lane = b.renderMode === "msr" ? "msr" : "fflf";
        const take: Take = { id: rid(), clipId, stage1Seed, draft: false, label: `seed ${stage1Seed}` };
        if (pieces.length === 0) setPieces([{ id: rid(), lane, label: "Base shot", takes: [take], selectedTakeId: take.id }]);
        else setPieces(pieces.map((pc, i) => i === 0 ? { ...pc, takes: [...pc.takes, take], selectedTakeId: take.id } : pc));
        setHunt(null); setViewResult(true); note("Finished — added as a take. Showing the result; press play.");
      } catch (e) { note("Finish failed: " + (e as Error).message); }
      finally { setBusy(false); }
      return;
    }
    const forExtend = hunt.forExtend;
    const kf = parseKf(b.director?.timeline_data);
    setBusy(true); note("Finishing at full res" + (b.lipsync ? " with lip-sync" : "") + "…");
    try {
      const p: Record<string, unknown> = {
        mode: "finish", stage1_seed: stage1Seed,
        first_strength: b.fflfFirstStrength ?? 0.7, last_strength: b.fflfLastStrength ?? 0.5,
        prompt: fflfPrompt(), negative: fflfNeg(), nondistilled: nd || undefined, steps: ndSteps,
        width: b.width, height: b.height, frames: b.frames, fps: b.fps,
      };
      if (forExtend) {
        p.first_id = lastClip; p.first_kind = "video"; p.first_frames = FFLF_TAIL; p.first_skip = Math.max(0, b.frames - FFLF_TAIL);
        p.last_name = kf.firstName;
      } else {
        p.first_name = kf.firstName;
        if (kf.lastName) p.last_name = kf.lastName;
      }
      if (b.lipsync && b.audioId) { p.audio_id = b.audioId; p.audio_start = b.audioStart; p.isolate_vocal = false; }   // full song (decided)
      const r = await api.videoLtxFflf(p) as { job_id: string };
      const clipId = bareId(await waitMedia(r.job_id, (pc, ps) => note(`Finishing… ${pc}%${ps>1?` · pass ${ps}`:""}`)));
      const take: Take = { id: rid(), clipId, stage1Seed, draft: false, label: `seed ${stage1Seed}` };
      if (forExtend) {
        const piece: ChainPiece = { id: rid(), lane: "fflf", label: `Extend ${pieces.length + 1}`, takes: [take], selectedTakeId: take.id, lastStillId: kf.firstName };
        const next = [...pieces, piece];
        setPieces(next);
        setHunt(null);
        await assembleChain(next);   // stitch base+extends into one continuous clip -> onto the timeline
        return;
      } else if (pieces.length === 0) {
        setPieces([{ id: rid(), lane: "fflf", label: "Base shot", takes: [take], selectedTakeId: take.id, lastStillId: kf.lastName || kf.firstName }]);
      } else {
        // re-finish the base piece (multiroll) -> add a take to piece 0
        setPieces(pieces.map((pc, i) => i === 0 ? { ...pc, takes: [...pc.takes, take], selectedTakeId: take.id } : pc));
      }
      setHunt(null); setViewResult(true); note("Finished. Showing the result; press play. Pick Extend to continue.");
    } catch (e) { note("Finish failed: " + (e as Error).message); }
    finally { setBusy(false); }
  }

  // ---- render the editor's authored timeline (timeline_data passthrough -> LTXDirector) ----
  async function renderTimeline() {
    if (!b.director?.timeline_data) { note("Build a timeline in the editor first (add a segment)."); return; }
    setBusy(true); note("Rendering the timeline…");
    try {
      const r = await api.videoLtxKeyframe({ ...b.director, char_lora: b.charLora, char_strength: b.charStrength, width: b.width, height: b.height, frames: b.frames, fps: b.fps }) as { job_id: string };
      const clipId = bareId(await waitMedia(r.job_id, (pc, ps) => note(`Rendering… ${pc}%${ps>1?` · pass ${ps}`:""}`)));
      const take: Take = { id: rid(), clipId, draft: false, label: "timeline" };
      if (pieces.length === 0) setPieces([{ id: rid(), lane: "fflf", label: "Base shot", takes: [take], selectedTakeId: take.id }]);
      else setPieces(pieces.map((pc, i) => i === 0 ? { ...pc, takes: [...pc.takes, take], selectedTakeId: take.id } : pc));
      note("Timeline rendered — added as a take.");
    } catch (e) { note("Render failed: " + (e as Error).message); }
    finally { setBusy(false); }
  }

  // ---- keyframe still: seed-hunt (half-res) -> pick -> full-res -> inject into the editor timeline ----
  const genStill = (seed: number, w: number, h: number) =>
    (kfUseChar && charRefs.length)
      ? api.videoCharStill({ ref_ids: charRefs, prompt: kfPrompt, width: w, height: h, seed })   // identity from refs
      : api.videoStill({ prompt: kfPrompt, width: w, height: h, seed });                          // prompt-only scene
  async function injectStill(mediaUrl: string) {
    const blob = await fetch(mediaUrl).then((r) => r.blob());
    const file = new File([blob], `kf_${Date.now()}.png`, { type: blob.type || "image/png" });
    if (!edRef.current) throw new Error("editor not ready");
    edRef.current.addImage(file, Math.max(0, Math.round(kfFrame)));
  }
  // A read-only editor timeline that is JUST the rendered clip as one video segment from frame 0, so the
  // editor's play button plays through the result cleanly (separate "View result" view, not mixed with the
  // input stills). It's a view only — never written back, so it can't pollute the render inputs.
  const resultTimeline = () => JSON.stringify({
    global_prompt: "",
    segments: [{ id: "result", type: "video", start: 0, length: b.frames,
                 _blobUrl: `/api/media/${resultClip}`, imageFile: `${resultClip}.mp4` }],
    audioSegments: [],
  });
  async function huntStills() {
    if (!kfPrompt.trim()) { setStatus("Enter a prompt for the keyframe still."); return; }
    setKfBusy(true); setStatus("Hunting 3 stills (full size)…");
    const base = Math.floor(Math.random() * 2_000_000_000);
    const hw = b.width, hh = b.height;   // full size — half-res previews drift too much from the final
    try {
      const drafts: { jobId: string; seed: number; url?: string }[] = [];
      for (let i = 0; i < 3; i++) {
        const r = await genStill(base + i, hw, hh) as { job_id: string };
        drafts.push({ jobId: r.job_id, seed: base + i });
      }
      setKfDrafts(drafts);
      drafts.forEach((d) => waitMedia(d.jobId, (pc, ps) =>
        setKfDrafts((ds) => ds.map((x) => x.jobId === d.jobId ? { ...x, pct: pc, pass: ps } : x)))
        .then((u) => setKfDrafts((ds) => ds.map((x) => x.jobId === d.jobId ? { ...x, url: u } : x)))
        .catch(() => {}));
      setStatus("Still drafts rendering — pick one to finalize at full res.");
    } catch (e) { setStatus("Still hunt failed: " + (e as Error).message); }
    finally { setKfBusy(false); }
  }
  async function useDraft(url: string) {
    setKfBusy(true); setStatus("Adding keyframe…");
    try {
      await injectStill(url);                 // drafts are already full-res, just inject the picked one
      setKfDrafts([]); setStatus(`Keyframe still added at frame ${kfFrame}.`);
    } catch (e) { setStatus("Add failed: " + (e as Error).message); }
    finally { setKfBusy(false); }
  }
  async function addLibraryStill(stillId: string) {
    if (!stillId) return;
    setKfBusy(true); setStatus("Adding library still as keyframe…");
    try { await injectStill(`/api/media/${stillId}`); setStatus(`Library still added at frame ${kfFrame}.`); }
    catch (e) { setStatus("Add failed: " + (e as Error).message); }
    finally { setKfBusy(false); }
  }

  function selectTake(pieceId: string, takeId: string) {
    setPieces(pieces.map((p) => p.id === pieceId ? { ...p, selectedTakeId: takeId } : p));
  }
  function removePiece(pieceId: string) { setPieces(pieces.filter((p) => p.id !== pieceId)); }
  // Stitch a multi-piece chain (base + FFLF extends) into ONE continuous clip (ffmpeg, no GPU), trimming
  // each extend's overlapping tail frames, and set it as the block's playable clip so the timeline plays
  // the whole take. Pass the fresh pieces array when calling right after setPieces (state not yet flushed).
  async function assembleChain(ps: ChainPiece[] = (b.pieces || [])) {
    const clips = ps.map((p) => bareId((p.takes.find((t) => t.id === p.selectedTakeId) || p.takes[0])?.clipId)).filter(Boolean) as string[];
    if (clips.length < 2) return;
    setBusy(true); note("Assembling continuous take…");
    try {
      const r = await api.videoAssembleChain({ clips, frames: b.frames, fps: b.fps, tail: FFLF_TAIL, width: b.width, height: b.height, transition: b.seamXfade || 0 }) as { id: string };
      patch({ assembledId: r.id, clipId: r.id });
      note(`Assembled ${clips.length} pieces — the continuous take is on the timeline.`);
    } catch (e) { note("Assemble failed: " + (e as Error).message); }
    finally { setBusy(false); }
  }

  const modes: { v: RenderMode; label: string }[] = [
    { v: "fflf", label: "FFLF (extendable)" }, { v: "msr", label: "MSR (identity+sing)" },
    { v: "keyframe", label: "Keyframe" }, { v: "i2v", label: "B-roll" },
  ];
  const totalSecs = pieces.length ? (b.frames / b.fps) + (pieces.length - 1) * ((b.frames - FFLF_TAIL) / b.fps) : 0;

  return (
    <div className="ss-root flex flex-col gap-4">
      {/* header */}
      <div className="flex items-center gap-3">
        <GhostButton onClick={onClose}>{"← Timeline"}</GhostButton>
        <span className="text-sm font-semibold text-[var(--color-ink)]">Shot {idx + 1}</span>
        <select className={inp} style={{ width: "auto" }} value={b.renderMode}
          onChange={(e) => patch({ renderMode: e.target.value as RenderMode })} title="How this shot is generated">
          {modes.map((m) => <option key={m.v} value={m.v}>{m.label}</option>)}
        </select>
        <span className="ml-auto text-[11px] text-[var(--color-muted)]">
          {pieces.length} piece{pieces.length === 1 ? "" : "s"} · ~{totalSecs.toFixed(1)}s take
        </span>
      </div>


      {/* ===================== TIMELINE EDITOR (vendored LTXDirector, GPL-3) ===================== */}
      {["fflf", "msr", "keyframe"].includes(b.renderMode) && (
        <div className="ss-card" style={{ padding: 8 }}>
          {resultClip && (
            <div className="mb-2 flex items-center gap-2">
              <GhostButton onClick={() => setViewResult(false)} disabled={!viewResult}>Edit inputs</GhostButton>
              <GhostButton onClick={() => setViewResult(true)} disabled={viewResult}>{"View result ▶"}</GhostButton>
              <span className="text-[10px] text-[var(--color-muted)]">
                {viewResult ? "playing the rendered clip — press the editor's play button" : "authoring keyframes / prompts"}
              </span>
            </div>
          )}
          {viewResult && resultClip ? (
            // read-only playback view: just the rendered clip; edits here are ignored (onChange no-op)
            <LtxDirectorEditor key={`result:${resultClip}`} timelineData={resultTimeline()} frames={b.frames} fps={b.fps}
              onChange={() => { /* result view is read-only */ }} />
          ) : (
            <LtxDirectorEditor key={`edit:${b.id}`} ref={edRef} timelineData={b.director?.timeline_data} frames={b.frames} fps={b.fps}
              onChange={(payload) => patch({ director: payload, timelineData: payload.timeline_data })} />
          )}
        </div>
      )}

      {/* ===================== ADD KEYFRAME STILL (generate seed-hunt / library -> inject as keyframe) ===================== */}
      {["fflf", "msr", "keyframe"].includes(b.renderMode) && (
        <div className="ss-card flex flex-col gap-3">
          <div className="text-xs font-semibold text-[var(--color-ink)]">Add keyframe still</div>
          <label className="flex items-center gap-2 text-[11px] text-[var(--color-muted)]">
            <input type="checkbox" checked={kfUseChar} onChange={(e) => setKfUseChar(e.target.checked)} disabled={!charRefs.length} />
            Use this shot's character (identity){charRefs.length ? ` · ${charRefs.length} ref${charRefs.length > 1 ? "s" : ""}` : " — no refs on this shot"}
          </label>
          <div className="grid grid-cols-[1fr_auto_auto] items-end gap-3">
            <Field label="Prompt (generate a still)">
              <textarea className={inp} rows={2} value={kfPrompt} onChange={(e) => setKfPrompt(e.target.value)}
                placeholder="the singer on a ruined ashen stage, dramatic light…" />
            </Field>
            <Num label="At frame" value={kfFrame} set={setKfFrame} step={1} w="w-20" />
            <PrimaryButton onClick={huntStills} disabled={kfBusy || !kfPrompt.trim()}>Seed-hunt 3 stills</PrimaryButton>
          </div>
          {kfDrafts.length > 0 && (
            <div className="grid grid-cols-3 gap-3">
              {kfDrafts.map((d) => (
                <div key={d.jobId} className="ss-piece">
                  <div className="ss-thumb">{d.url ? <img src={d.url} alt="" className="h-full w-full object-cover" /> : <div className="ss-spin">{d.pct ? `rendering… ${d.pct}% · pass ${Math.min(d.pass || 1, 2)} of 2` : "queued…"}</div>}</div>
                  <div className="flex items-center justify-between gap-2 px-2 py-1.5">
                    <span className="text-[11px] text-[var(--color-muted)]">seed {d.seed}</span>
                    <GhostButton onClick={() => useDraft(d.url!)} disabled={kfBusy || !d.url}>Use this</GhostButton>
                  </div>
                </div>
              ))}
            </div>
          )}
          <div className="flex items-center gap-2">
            <span className="text-[11px] text-[var(--color-muted)]">or add an existing still:</span>
            <div className="min-w-[220px]"><StillPick value="" set={addLibraryStill} stills={stills} placeholder="— pick from library —" /></div>
          </div>
        </div>
      )}

      {/* ===================== BUILD PANEL (anchors · lip-sync · render · actions) ===================== */}
      <div className="ss-card grid grid-cols-[1.3fr_1fr] gap-5">
        {/* left: source + prompt */}
        <div className="flex flex-col gap-3">
          {b.renderMode === "fflf" ? (
            <>
              {/* Anchors come from the TIMELINE's image keyframes — first image = opening, last image =
                  target. Author them in the editor above (Generate / Add keyframe still), not here. */}
              {(() => {
                const kf = parseKf(b.director?.timeline_data);
                return (
                  <div className="rounded-md border border-[var(--color-line)] p-2 text-[11px]">
                    <div className="font-medium text-[var(--color-ink)]">Anchors (from the timeline)</div>
                    <div className="mt-1 text-[var(--color-muted)]">
                      Opening: {kf.firstName ? <span className="text-[var(--color-accent2)]">✓ set</span> : <span className="text-amber-400">— add the opening still to the timeline</span>}
                      {"  ·  "}
                      Last: {kf.lastName ? <span className="text-[var(--color-accent2)]">✓ set</span> : <span className="text-[var(--color-muted)]">none (slow static push) — use “Make push-in” or add a target still</span>}
                    </div>
                  </div>
                );
              })()}
              {/* B-roll push-in: crop the opening image into the LAST keyframe of the timeline. Both ends
                  pinned + person-free = a clean dolly-in, no hallucinated people, no super-speed. */}
              <div className="flex items-end gap-3 rounded-md border border-dashed border-[var(--color-line)] p-2">
                <div className="flex-1">
                  <div className="text-[11px] font-medium text-[var(--color-ink)]">Scenic push-in (B-roll)</div>
                  <div className="text-[10px] text-[var(--color-muted)]">Crops the opening image into a last keyframe in the timeline — a slow dolly-in with no stray people. Lower = stronger zoom.</div>
                </div>
                <Num label="Keep" value={pushKeep} set={setPushKeep} step={0.05} w="w-20" />
                <GhostButton onClick={makePushIn} disabled={busy || !parseKf(b.director?.timeline_data).firstName}>Make push-in</GhostButton>
              </div>
              <div className="flex gap-3">
                <Num label="First strength" value={b.fflfFirstStrength ?? 0.7} set={(n) => patch({ fflfFirstStrength: n })} step={0.05} />
                <Num label="Last strength" value={b.fflfLastStrength ?? 0.5} set={(n) => patch({ fflfLastStrength: n })} step={0.05} />
              </div>
            </>
          ) : (
            <div className="rounded-md border border-dashed border-[var(--color-line)] p-3 text-[11px] text-[var(--color-muted)]">
              {b.renderMode.toUpperCase()} editor controls (LTXDirector prompt-timeline / keyframes / retake)
              land in the next phase. Switch to <b>FFLF</b> to use the seed-hunt / multiroll / extend loop now.
            </div>
          )}
          <Field label="Prompt (what happens between the anchors)">
            <textarea className={inp} rows={3} value={b.prompt} onChange={(e) => patch({ prompt: e.target.value })}
              placeholder="continuous fluid shot, the camera moving naturally around her…" />
          </Field>
          {/* render model — applies to every path (fflf / msr / keyframe / b-roll) */}
          <div className="space-y-1.5 rounded-md border border-dashed border-[var(--color-line)] p-2">
            <label className="flex items-start gap-2 text-[11px] text-[var(--color-muted)]" title="Render on the non-distilled (dev) LTX model: much better, controllable motion (the distilled model has none — water/clouds timelapse). Real users run cfg 3 / 30-50 steps / euler / linear_quadratic. SLOWER render. Experimental on MSR (changes the lip-sync sampler).">
              <input type="checkbox" className="mt-0.5" checked={!!b.nonDistilled} onChange={(e) => patch({ nonDistilled: e.target.checked })} />
              <span><span className="font-medium text-[var(--color-ink)]">Non-distilled (dev model)</span> — better, controllable motion (no timelapse). Slower render; experimental on MSR.</span>
            </label>
            {b.nonDistilled && (
              <div className="flex items-center gap-2">
                <Num label="Steps" value={stepsVal} set={(n) => patch({ steps: Math.round(n) })} step={1} w="w-24" min={16} max={60} />
                <span className="text-[10px] text-[var(--color-muted)]">30–50 typical; more = cleaner but slower</span>
              </div>
            )}
          </div>
        </div>
        {/* right: lip-sync + params + action rail */}
        <div className="flex flex-col gap-3">
          <label className="flex items-center gap-2 text-xs text-[var(--color-ink)]">
            <input type="checkbox" checked={b.lipsync} onChange={(e) => patch({ lipsync: e.target.checked })} />
            Lip-sync to the song (full song fed — no isolation)
          </label>
          {b.lipsync && (
            <div className="flex gap-3">
              <Field label="Vocal track"><StillPick value={b.audioId || songAudioId} set={(id) => patch({ audioId: id })} stills={audios} thumb={false} placeholder="— song —" /></Field>
              <Num label="Audio start (s)" value={b.audioStart} set={(n) => patch({ audioStart: n })} step={0.1} />
            </div>
          )}
          <div className="flex flex-wrap gap-2">
            <Num label="W" value={b.width} set={(n) => patch({ width: n })} step={32} w="w-20" />
            <Num label="H" value={b.height} set={(n) => patch({ height: n })} step={32} w="w-20" />
            <Num label="Frames" value={b.frames} set={(n) => patch({ frames: n })} step={8} w="w-20" />
            <Num label="FPS" value={b.fps} set={(n) => patch({ fps: n })} step={1} w="w-16" />
          </div>
          {/* character / ID LoRA — identity from a downloadable LoRA instead of MSR */}
          <div className="flex items-end gap-2">
            <label className="flex-1 text-[11px] text-[var(--color-muted)]">
              Character LoRA <span className="opacity-70">(identity — instead of MSR)</span>
              <select className={inp} style={{ width: "100%" }} value={b.charLora || ""} onChange={(e) => patch({ charLora: e.target.value })}>
                <option value="">— none —</option>
                {loras.map((l) => <option key={l} value={l}>{l}</option>)}
              </select>
            </label>
            <Num label="Str" value={b.charStrength ?? 1.0} set={(n) => patch({ charStrength: n })} step={0.05} w="w-16" />
          </div>
          {!loras.length && <span className="text-[10px] text-[var(--color-muted)]">No character LoRAs on the box — drop one in ComfyUI <code>models/loras</code> + restart ComfyUI.</span>}
          {/* action rail */}
          <div className="mt-1 flex flex-col gap-2 rounded-md border border-[var(--color-line)] p-3">
            <span className="text-[10px] uppercase tracking-wide text-[var(--color-muted)]">Build this shot</span>
            <PrimaryButton onClick={renderTimeline} disabled={busy || !b.director?.timeline_data}>
              Render timeline (from editor)
            </PrimaryButton>
            <PrimaryButton onClick={huntVideo} disabled={busy || !canHuntVideo}>
              {pieces.length ? "Re-hunt video (3)" : "Seed-hunt video (3)"}
            </PrimaryButton>
            <GhostButton onClick={() => runHunt(true)} disabled={busy || !lastClip}>
              + Extend off last piece's tail (3 drafts)
            </GhostButton>
            <GhostButton onClick={() => note("Crossfade-assemble runs server-side (next phase) — uses the 33-frame overlap + continuation audio.")} disabled={pieces.length < 2}>
              Assemble continuous take ({pieces.length} pieces)
            </GhostButton>
          </div>
          {status && <p className="text-[11px] text-[var(--color-accent2)]">{status}</p>}
        </div>
      </div>

      {/* ===================== HUNT DRAFTS (pick one) ===================== */}
      {hunt && (
        <div className="ss-card">
          <div className="mb-2 text-xs font-semibold text-[var(--color-ink)]">Pick a draft for “{hunt.pieceLabel}” (half-res, silent)</div>
          <div className="grid grid-cols-3 gap-3">
            {hunt.drafts.map((d) => (
              <div key={d.jobId} className="ss-piece">
                <div className="ss-thumb">{d.url ? <PreviewVideo src={d.url} /> : <div className="ss-spin">{d.pct ? `rendering… ${d.pct}% · pass ${Math.min(d.pass || 1, 2)} of 2` : "queued…"}</div>}</div>
                <div className="flex items-center justify-between gap-2 px-2 py-1.5">
                  <span className="text-[11px] text-[var(--color-muted)]">seed {d.seed}</span>
                  <GhostButton onClick={() => finishDraft(d.seed)} disabled={busy || !d.url}>Finish</GhostButton>
                </div>
              </div>
            ))}
          </div>
          <button className="mt-2 text-[10px] text-[var(--color-muted)] underline" onClick={() => setHunt(null)}>dismiss drafts</button>
        </div>
      )}

      {/* ===================== PIECES (card layout under the editor) ===================== */}
      <div>
        <div className="mb-2 flex items-center gap-3">
          <span className="text-xs font-semibold text-[var(--color-ink)]">Pieces of this take</span>
          {pieces.length > 1 && (
            <GhostButton onClick={() => assembleChain()} disabled={busy}>{busy ? "Assembling…" : `Re-assemble ${pieces.length} pieces → timeline`}</GhostButton>
          )}
          {pieces.length > 1 && (
            <Num label="Seam crossfade (s)" value={b.seamXfade ?? 0} set={(n) => patch({ seamXfade: Math.max(0, n) })} step={0.1} w="w-20" min={0} max={2} />
          )}
          {pieces.length > 1 && b.assembledId && <span className="text-[10px] text-[var(--color-accent2)]">✓ continuous take on timeline</span>}
        </div>
        {pieces.length === 0 ? (
          <div className="rounded-lg border border-dashed border-[var(--color-line)] p-6 text-center text-[11px] text-[var(--color-muted)]">
            No pieces yet. Set the anchors, then <b>Seed-hunt base</b> → pick a draft → it becomes piece 1. Then <b>Extend</b> to grow the take.
          </div>
        ) : (
          <div className="grid grid-cols-3 gap-3">
            {pieces.map((p, i) => {
              const sel = selectedTakeOf(p);
              return (
                <div key={p.id} className="ss-piece">
                  <div className="ss-thumb">{sel?.clipId ? <PreviewVideo src={`/api/media/${bareId(sel.clipId)}`} /> : <div className="ss-spin">—</div>}</div>
                  <div className="px-2 py-1.5">
                    <div className="flex items-center justify-between">
                      <span className="text-[11px] font-semibold text-[var(--color-ink)]">{i + 1}. {p.label}</span>
                      <button className="text-[var(--color-muted)] hover:text-red-400" title="remove piece" onClick={() => removePiece(p.id)}>×</button>
                    </div>
                    {/* takes / multiroll variants */}
                    {p.takes.length > 1 && (
                      <div className="mt-1 flex gap-1">
                        {p.takes.map((t) => (
                          <button key={t.id} title={t.label} onClick={() => selectTake(p.id, t.id)}
                            className={`h-9 w-12 overflow-hidden rounded border ${t.id === p.selectedTakeId ? "border-[var(--color-accent)]" : "border-[var(--color-line)] opacity-70"}`}>
                            <ThumbVideo id={bareId(t.clipId)} />
                          </button>
                        ))}
                      </div>
                    )}
                    <div className="mt-1 text-[10px] text-[var(--color-muted)]">{p.takes.length} take{p.takes.length === 1 ? "" : "s"} · {sel?.label}</div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      <style>{`
        .ss-card{border:1px solid var(--color-line);background:var(--color-panel);border-radius:12px;padding:16px;}
        .ss-piece{border:1px solid var(--color-line);background:var(--color-panel2);border-radius:10px;overflow:hidden;}
        .ss-thumb{aspect-ratio:16/9;background:#0d0f13;display:flex;align-items:center;justify-content:center;}
        .ss-spin{font-size:10px;color:var(--color-muted);}
      `}</style>
    </div>
  );
}
