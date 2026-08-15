import { useState } from "react";
import { api, type LibItem } from "./api";
import { Field, inp, PrimaryButton, GhostButton, rid, pollJob, type RunCtx } from "./ui";
import { useDrafts } from "./drafts";
import { openLightbox } from "./Lightbox";
import { type Character } from "./mvmodel";
import { H3SegTimeline, H3_MIN_CUT_S, type VoiceWin } from "./H3SegTimeline";

// ============================================================================
// MiniMax H3 segment pipeline (Phase 5 - docs/MINIMAX_H3_PLAN.md).
// The hybrid shot model: the writer LLM turns the song's audio structure into render
// SEGMENTS ("single" = one shot, always for lip-sync; "scene" = 2-4 timestamped cuts
// in one generation). Each segment renders as ONE /api/video/h3_ref2v call whose refs
// follow the compiled prompt's picture numbering: character sheets, then outfit sheets,
// then the segment's environment still. Lip-sync segments carry the song window as
// <Audio 1> (direct reuse - the soundtrack IS the song, so assembly lays the master
// underneath at zero offset).
// ============================================================================

type H3Shot = {
  type: string; framing: string; lipsync: boolean; camera: string;
  location?: string; scene: string; action: string; costume: string; characters: string[];
  // Hand-cast by the user, so voice-matched casting must leave this shot alone. Who sings when is
  // INFERRED (nominal section markers mapped onto the real audio), and where that inference is
  // wrong the only correction available is a person's ear - so a manual swap has to outrank it.
  // Until this existed, recasting a shot appeared to work in the cast row and was then silently
  // reverted by the next recompile (segment 14: "103s Bob->Selene (female part)").
  cast_locked?: boolean;
  // Who performs the vocal, when that is not simply "everyone in frame with a singer role". Absent
  // = the role-based default. This is the only way to stage two singers together with one of them
  // silent, since both carry a singer role and were therefore both told to lip-sync.
  singers?: string[];
};
export type H3Segment = {
  start: number; end: number; seconds: number; render_seconds: number; frames: number;
  section: string; cuts: { start: number; end: number }[];
  kind: "single" | "scene"; shots: H3Shot[]; lipsync: boolean; soundscape: string;
  prompt: string; picture_map: Record<string, number>; outfit_map: Record<string, number>;
  prop_map?: Record<string, number>;
  env_map?: Record<string, number>;   // lowercased location -> env picture number ("" = unnamed)
  env_picture: number;
  // client-side render state
  envStillId?: string; clipId?: string; clipVariants?: string[];
  // FlashVSR 2x of the chosen take. Kept alongside clipId rather than replacing it, so the original
  // stays available and a re-render clears the upscale rather than silently assembling a stale one.
  upscaledId?: string;
  staleClip?: boolean;   // the segment's window moved after this was rendered - redo the take
  handEdited?: boolean;   // raw prompt overridden by hand (a recompile clears this)
  // A shot field changed but the prompt has not been recompiled, so the prompt (and any render
  // fired from it) still describes the OLD shots. Set by every structured edit, cleared by a
  // recompile. Without this the divergence is invisible: swapping a character in "in this shot"
  // updates the chip immediately while the compiled text below keeps the previous name, which
  // reads as "my edit was reverted".
  promptStale?: boolean;
};

// keep in sync with H3_SEG_MAX_S in backend/musicvideo.py. Past this the render collapses from the
// tail - the audio drifts away from the song and the lip-sync follows the drift, not the track.
// Segments cut before 2026-08-15 could exceed it, so this is shown rather than assumed away.
const H3_SEG_MAX_S = 10.5;

// keep in sync with H3_CAMERA_MOVES in backend/musicvideo.py (the compiler drops anything unknown
// back to "static"). Gentle tier first, then the assertive moves added for more dynamic camerawork.
const H3_CAMERAS = ["static", "push in", "pull back", "truck left", "truck right",
  "arc left", "arc right", "tilt up", "crane up",
  "push in strong", "pull back reveal", "orbit left", "orbit right", "handheld drift",
  "steadicam follow", "crane down", "tilt down", "rack focus"];
// Editor stages, in order - the same gate-then-pay progression as the LTX Shot Editor's rail.
const H3_STAGES = ["shots", "prompt", "env", "video", "result"];

// module scope on purpose: a random seed called straight from a component body trips the
// react-hooks purity rule, and these are fire-a-render callbacks, not render-time values
const randSeed = () => Math.floor(Math.random() * 2_000_000_000);

// Recasting a shot has to carry the PRONOUNS with the name, or the text says "Selene sings as he
// grips the rail" and the render gets a contradictory gender cue (the backend does the same for
// voice-matched casting). "her" is both object and possessive: followed by a word it is possessive
// ("her face" -> "his face"), otherwise object ("toward her" -> "him").
const PRONOUNS: Record<string, [RegExp, string][]> = {
  "male>female": [[/\bhimself\b/gi, "herself"], [/\bhis\b/gi, "her"], [/\bhim\b/gi, "her"], [/\bhe\b/gi, "she"]],
  "female>male": [[/\bherself\b/gi, "himself"], [/\bhers\b/gi, "his"],
    [/\bher\b(?=\s+[a-z])/gi, "his"], [/\bher\b/gi, "him"], [/\bshe\b/gi, "he"]],
};
function swapPronouns(text: string, from: string, to: string) {
  const rules = PRONOUNS[`${from}>${to}`];
  if (!rules) return text;                          // same gender, or a gender we do not know
  return rules.reduce((t, [re, rep]) =>
    t.replace(re, (m) => (m[0] === m[0].toUpperCase() ? rep[0].toUpperCase() + rep.slice(1) : rep)), text);
}

function VideoTile({ src }: { src: string }) {
  return <video src={src} muted loop autoPlay playsInline controls preload="auto" className="h-full w-full object-contain" />;
}

const fmt = (t: number) => `${Math.floor(t / 60)}:${String(Math.floor(Math.max(0, t) % 60)).padStart(2, "0")}`;

// environment still request: the verified rich-caption layout recipe (Krea2 needs a
// fully-populated caption; the ref's realism sets the scene's realism ceiling)
function envRequest(scene: string, seed: number) {
  const s = scene.replace(/\.\s*$/, "");
  return {
    engine: "krea2", two_pass: true, enhancer: true,
    width: 1920, height: 1088, seed,
    layout: {
      overview: `An ultra-photorealistic cinematic view of ${s}.`,
      background: `${s}. Rich physical detail: real materials, surface wear, depth layers, ` +
        `natural atmospheric haze where it fits, believable light falloff from every source.`,
      photo_style: "",
      aesthetics: "photorealistic, cinematic, true-to-life detail, natural film-like exposure, atmospheric",
      lighting: "natural motivated light exactly as the scene describes, deep soft shadows, no artificial studio light",
      medium: "photograph",
      regions: [{ desc: `${s}, rendered with true-to-life photographic detail.`, x: 0.03, y: 0.03, w: 0.94, h: 0.94 }],
    },
  };
}

export function H3Studio({ cast, audioId, songPayload, resW, resH, grade, library, busy, ...ctx }: {
  cast: Character[]; audioId: string; songPayload: unknown | null; resW: number; resH: number;
  grade: string; library: LibItem[]; busy: boolean;
} & RunCtx) {
  const d = useDrafts("mvstudio");
  const [segments, setSegments] = d.use<H3Segment[]>("h3segments", []);
  const [castCostume, setCastCostume] = d.use<Record<string, string>>("h3castCostume", {});
  const [castProp, setCastProp] = d.use<Record<string, string>>("h3castProp", {});   // "" = none
  // environment stills keyed by LOCATION (lowercased) - one still per named place, shared by
  // every segment/cut set there. Unnamed environments stay on the segment (envStillId).
  const [envByLoc, setEnvByLoc] = d.use<Record<string, string>>("h3envByLoc", {});
  // environment CANDIDATES keyed the same way (a location, or "seg<i>" for an unnamed one): the
  // gate-before-render rule wants several cheap stills to choose from, not one take you either
  // accept or re-roll away.
  const [envCands, setEnvCands] = d.use<Record<string, { jobId: string; seed: number }[]>>("h3envCands", {});
  const [writing, setWriting] = useState(false);
  const [genning, setGenning] = useState("");        // "env:<i>" | "seg:<i>" while submitting
  // seed-hunt drafts per segment index - persisted so closing the editor (or the project) does not
  // throw away renders you have already paid for
  const [drafts, setDrafts] = d.use<Record<number, { jobId: string; seed: number }[]>>("h3drafts", {});
  const [jobState, setJobState] = useState<Record<string, { pct: number; url?: string; err?: string }>>({});
  // where each voice sings, on the REAL audio timeline - drawn under the segment editor's timeline
  const [voiceWins, setVoiceWins] = useState<VoiceWin[]>([]);
  // clip id -> how closely that take's own audio follows the song (see /api/mv/h3_audio_check)
  const [audioScore, setAudioScore] = useState<Record<string, number | null>>({});
  const [editIdx, setEditIdx] = useState(-1);        // segment open in the full editor
  const [estep, setEstep] = useState("shots");       // which editor stage is showing
  const [huntN, setHuntN] = useState(2);
  const [recompiling, setRecompiling] = useState(false);
  // which LLM writes the script (defaults to Claude; "local" = whatever local provider the backend picks).
  // Key is versioned (h3scriptLlm2) so projects saved with the old Sonnet 4.6 default pick up Sonnet 5.
  const [scriptLlm, setScriptLlm] = d.use("h3scriptLlm2", "claude-sonnet-5");

  const h3cast = cast.filter((c) => c.style === "h3" && c.sheetId);
  const patchSeg = (i: number, p: Partial<H3Segment>) =>
    setSegments((prev) => (prev as H3Segment[]).map((s, j) => (j === i ? { ...s, ...p } : s)));

  // the cast payload the script writer + compiler see: identity look, sheet, chosen costume + prop
  function castPayload() {
    return h3cast.map((c) => {
      const co = (c.costumes || []).find((x) => x.id === (castCostume[c.id] ?? (c.costumes || [])[0]?.id));
      const pr = (c.props || []).find((x) => x.id === (castProp[c.id] ?? (c.props || [])[0]?.id));
      return {
        name: c.name, role: c.role, look: c.appearance || "",
        // the writer's cast formatter reads appearance + gender (voice-matched casting needs
        // the gender; without it the writer cannot know who sings the male/female parts)
        appearance: c.appearance || "", gender: c.gender || "",
        sheet_id: c.sheetId,
        // "" selection = "(sheet outfit)": no costume ref, the identity sheet's wardrobe is worn
        ...(co?.stillId && castCostume[c.id] !== "" ? { costume: { name: co.name, desc: co.desc, still_id: co.stillId } } : {}),
        ...(pr?.stillId && castProp[c.id] !== "" ? { prop: { name: pr.name, desc: pr.desc, still_id: pr.stillId } } : {}),
      };
    });
  }

  async function writeScript() {
    if (!audioId) { ctx.setResults([{ id: rid(), title: "Pick the song first", status: "error", pct: 0, err: "H3 segments are built from the real audio structure - set the song track above." }]); return; }
    if (!songPayload) { ctx.setResults([{ id: rid(), title: "No song arrangement", status: "error", pct: 0, err: "Open a project with a Song arrangement (the Song tab)." }]); return; }
    setWriting(true);
    const card = { id: rid(), title: "H3 script - analyzing audio structure…", status: "running" as const, pct: 3 };
    ctx.setResults([card]);
    // The call is one synchronous request (audio analysis ~1-2 min, then the writer LLM ~3-8 min),
    // so real progress isn't available - show STAGED progress with a live elapsed clock instead of
    // a frozen bar. Time-based pct creeps to 90 and holds until the response lands.
    const t0 = Date.now();
    const EXPECT = 7 * 60_000;                       // typical total; bar reaches 90% here
    const tick = window.setInterval(() => {
      const el = Date.now() - t0;
      const mm = Math.floor(el / 60_000), ss = String(Math.floor(el / 1000) % 60).padStart(2, "0");
      const stage = el < 100_000 ? "analyzing audio structure" : "writer LLM drafting segments";
      ctx.patch(card.id, {
        pct: Math.min(90, Math.round((el / EXPECT) * 90)),
        title: `H3 script - ${stage}… ${mm}:${ss} (usually 4-8 min total)`,
      });
    }, 1000);
    try {
      const llm = scriptLlm === "local" ? {} : { provider: "claude_sub", model: scriptLlm };
      const r = await api.mvH3Script({ song: songPayload, audio_id: audioId, cast: castPayload(), ...llm }) as
        { segments: H3Segment[]; singles: number; scenes: number };
      setSegments(r.segments.map((s) => ({ ...s })));
      ctx.patch(card.id, { status: "done", pct: 100, title: "H3 script written",
        err: `${r.segments.length} segments (${r.singles} single / ${r.scenes} scene)` });
    } catch (e) { ctx.patch(card.id, { status: "error", pct: 0, title: "H3 script failed", err: (e as Error).message }); }
    finally { window.clearInterval(tick); setWriting(false); }
  }

  // ---- per-segment editing: change the structured shot fields, then RECOMPILE server-side so
  // the enforced rules (duration anchors, sky-pin, subject-swap, doubled framing) stay intact ----
  const patchShot = (i: number, j: number, p: Partial<H3Shot>) =>
    setSegments((prev) => (prev as H3Segment[]).map((s, k) =>
      k === i ? { ...s, promptStale: true, shots: s.shots.map((sh, m) => (m === j ? { ...sh, ...p } : sh)) } : s));
  async function recompile(i: number) {
    const seg = segments[i];
    const missing = seg.shots.flatMap((s) => s.characters).filter((n) => !h3cast.some((c) => c.name === n));
    if (missing.length) {
      ctx.setResults([{ id: rid(), title: `segment ${i + 1} not recompiled - cast not loaded`, status: "error", pct: 0,
        err: `${[...new Set(missing)].join(", ")} missing from the loaded cast; recompiling would drop their identity references.` }]);
      return;
    }
    setRecompiling(true);
    try {
      // the song goes with it (its markers drive voice-matched casting) plus the whole video's
      // section grid, which is what maps the arrangement's nominal times onto the real audio
      const r = await api.mvH3Compile({ segment: seg, cast: castPayload(), song: songPayload,
        section_grid: segments.map((x) => ({ start: x.start, end: x.end, section: x.section })) }) as
        { prompt: string; picture_map: Record<string, number>; outfit_map: Record<string, number>;
          env_picture: number; lipsync: boolean; shots: H3Shot[]; kind: "single" | "scene";
          cuts: { start: number; end: number }[]; voice_fixes?: string[] };
      patchSeg(i, { prompt: r.prompt, picture_map: r.picture_map, outfit_map: r.outfit_map,
        env_picture: r.env_picture, lipsync: r.lipsync, shots: r.shots, kind: r.kind, cuts: r.cuts,
        handEdited: false, promptStale: false });
      // ALWAYS report a recast. Voice-matched casting can overrule who you put in a shot, and
      // swallowing this line is what made a hand swap look like it "changed itself back".
      ctx.setResults([{ id: rid(), title: `segment ${i + 1} recompiled`, status: "done", pct: 100,
        err: r.voice_fixes?.length
          ? `voice-matched casting overruled your cast: ${r.voice_fixes.join(", ")}. Swap the person again to lock the shot - a hand-cast shot is left alone from then on.`
          : undefined }]);
    } catch (e) { ctx.setResults([{ id: rid(), title: `segment ${i + 1} recompile failed`, status: "error", pct: 0, err: (e as Error).message }]); }
    finally { setRecompiling(false); }
  }

  // ---- environment stills (required: every compiled prompt references its env pictures).
  // LOCATION CONTINUITY: one still per named location (envByLoc), shared everywhere that
  // location appears - including as the second thread of an intercut scene. A segment may
  // reference up to TWO locations (its env_map); unnamed environments stay per-segment. ----
  const segLocs = (s: H3Segment): string[] =>
    s.env_map ? Object.keys(s.env_map).sort((a, b) => s.env_map![a] - s.env_map![b])
              : [(s.shots[0]?.location || "").trim().toLowerCase()];
  const locDisplay = (s: H3Segment, loc: string) =>
    s.shots.find((sh) => (sh.location || "").trim().toLowerCase() === loc)?.location || loc || "environment";
  const envOf = (s: H3Segment, loc: string) => (loc ? envByLoc[loc] : s.envStillId);
  async function genEnvLoc(i: number, loc: string, forceNew = false) {
    const seg = segments[i];
    const shot = seg.shots.find((sh) => (sh.location || "").trim().toLowerCase() === loc) || seg.shots[0];
    const scene = shot?.scene || "";
    if (!scene) { ctx.setResults([{ id: rid(), title: `segment ${i + 1}`, status: "error", pct: 0, err: "No scene description for this environment." }]); return; }
    if (!forceNew && envOf(seg, loc)) return;
    setGenning(`env:${i}:${loc}`);
    try {
      const r = await api.videoStill(envRequest(scene, randSeed())) as { job_id: string };
      if (loc) setEnvByLoc((prev) => ({ ...(prev as Record<string, string>), [loc]: r.job_id }));
      else patchSeg(i, { envStillId: r.job_id });
      const card = { id: rid(), title: `"${locDisplay(seg, loc)}" environment`, status: "running" as const, pct: 5 };
      ctx.setResults([card]);
      pollJob(r.job_id, card.id, ctx);
    } catch (e) { ctx.setResults([{ id: rid(), title: `environment render failed`, status: "error", pct: 0, err: (e as Error).message }]); }
    finally { setGenning(""); }
  }
  async function genAllEnvs() {
    for (let i = 0; i < segments.length; i++)
      for (const loc of segLocs(segments[i]))
        if (!envOf(segments[i], loc)) await genEnvLoc(i, loc);
  }

  // ---- job tracking for the editor's candidate tiles. A still's media is /api/media/<job_id>, so
  // all the tile needs is "is it done yet"; the poll stops itself on done/error. ----
  function track(jobId: string, onSettle?: (ok: boolean) => void) {
    if (jobState[jobId]?.url) { onSettle?.(true); return; }
    setJobState((s) => ({ ...s, [jobId]: { pct: s[jobId]?.pct ?? 2 } }));
    const t = window.setInterval(async () => {
      const j = await api.job(jobId).catch(() => null);
      if (!j) return;
      if (j.status === "done" && j.media_url) {
        window.clearInterval(t);
        setJobState((s) => ({ ...s, [jobId]: { pct: 100, url: j.media_url } }));
        onSettle?.(true);
      } else if (j.status === "error" || j.status === "failed") {
        window.clearInterval(t);
        setJobState((s) => ({ ...s, [jobId]: { pct: 0, err: j.error || "render error" } }));
        onSettle?.(false);
      } else {
        setJobState((s) => ({ ...s, [jobId]: { pct: j.max ? Math.round((100 * (j.progress || 0)) / j.max) : 5 } }));
      }
    }, 1500);
  }
  const envKey = (i: number, loc: string) => loc || `seg${i}`;
  const locUsedBy = (loc: string) =>
    loc ? segments.filter((s) => segLocs(s).includes(loc)).length : 1;

  // N environment candidates for one location - pick one, and it becomes the shared still for every
  // segment set there. Additive: the old single-still buttons in the table still work.
  async function genEnvCands(i: number, loc: string, n = 3) {
    const seg = segments[i];
    const shot = seg.shots.find((sh) => (sh.location || "").trim().toLowerCase() === loc) || seg.shots[0];
    const scene = shot?.scene || "";
    if (!scene) { ctx.setResults([{ id: rid(), title: `segment ${i + 1}`, status: "error", pct: 0, err: "No scene description for this environment - write one in Shots first." }]); return; }
    setGenning(`cand:${i}:${loc}`);
    try {
      const base = randSeed();
      const made: { jobId: string; seed: number }[] = [];
      for (let k = 0; k < n; k++) {
        const r = await api.videoStill(envRequest(scene, base + k)) as { job_id: string };
        made.push({ jobId: r.job_id, seed: base + k });
      }
      const key = envKey(i, loc);
      setEnvCands((prev) => ({ ...(prev as Record<string, { jobId: string; seed: number }[]>),
        [key]: [...made, ...((prev as Record<string, { jobId: string; seed: number }[]>)[key] || [])].slice(0, 9) }));
      // ONE card for the batch, driven to completion as the candidates settle. It must reach a
      // terminal status: App.tsx derives `busy` from "is any result still running", so a card left
      // running latches busy TRUE for the whole app and silently disables every render button -
      // which is exactly how "Generate 3 more" became a dead click (2026-08-10).
      const card = { id: rid(), title: `${n} "${locDisplay(seg, loc)}" candidates`, status: "running" as const, pct: 5 };
      ctx.setResults([card]);
      let settled = 0, ok = 0;
      made.forEach((m) => track(m.jobId, (good) => {
        settled += 1;
        ok += good ? 1 : 0;
        if (settled < made.length) { ctx.patch(card.id, { status: "running", pct: Math.round((100 * settled) / made.length) }); return; }
        ctx.patch(card.id, ok
          ? { status: "done", pct: 100, err: `${ok} of ${made.length} rendered - pick one` }
          : { status: "error", pct: 0, err: "every candidate failed to render" });
        ctx.onDone();
      }));
    } catch (e) { ctx.setResults([{ id: rid(), title: "environment render failed", status: "error", pct: 0, err: (e as Error).message }]); }
    finally { setGenning(""); }
  }
  function pickEnv(i: number, loc: string, jobId: string) {
    if (loc) setEnvByLoc((prev) => ({ ...(prev as Record<string, string>), [loc]: jobId }));
    else patchSeg(i, { envStillId: jobId });
    ctx.setResults([{ id: rid(), title: `✓ "${locDisplay(segments[i], loc)}" environment set`,
      status: "done", pct: 100, err: loc ? `used by ${locUsedBy(loc)} segment(s)` : undefined }]);
  }

  // refs in the compiled prompt's exact picture order: sheets, outfits, environment
  function refsFor(seg: H3Segment): string[] | string {
    const byName = Object.fromEntries(h3cast.map((c) => [c.name, c]));
    const sheets = Object.entries(seg.picture_map).sort((a, b) => a[1] - b[1])
      .map(([n]) => byName[n]?.sheetId).filter(Boolean) as string[];
    if (sheets.length !== Object.keys(seg.picture_map).length) return "a cast member in this segment has no identity sheet";
    const outfits = Object.entries(seg.outfit_map || {}).sort((a, b) => a[1] - b[1])
      .map(([n]) => {
        const c = byName[n];
        if (c && castCostume[c.id] === "") return undefined;   // (sheet outfit) selected after compile
        const co = (c?.costumes || []).find((x) => x.id === (castCostume[c!.id] ?? (c?.costumes || [])[0]?.id));
        return co?.stillId;
      }).filter(Boolean) as string[];
    if (outfits.length !== Object.keys(seg.outfit_map || {}).length) return "a costume still is missing";
    const props = Object.entries(seg.prop_map || {}).sort((a, b) => a[1] - b[1])
      .map(([n]) => {
        const c = byName[n];
        const pr = (c?.props || []).find((x) => x.id === (castProp[c!.id] ?? (c?.props || [])[0]?.id));
        return pr?.stillId;
      }).filter(Boolean) as string[];
    if (props.length !== Object.keys(seg.prop_map || {}).length) return "a prop still is missing";
    const envs = segLocs(seg).map((loc) => envOf(seg, loc));
    if (envs.some((e) => !e)) return "generate the environment still(s) first";
    return [...sheets, ...outfits, ...props, ...(envs as string[])];
  }

  async function renderSeg(i: number, mode: "hunt" | "finish", seed?: number) {
    const seg = segments[i];
    const refs = refsFor(seg);
    if (typeof refs === "string") { ctx.setResults([{ id: rid(), title: `segment ${i + 1}`, status: "error", pct: 0, err: refs }]); return; }
    setGenning(`seg:${i}`);
    const card = { id: rid(), title: `segment ${i + 1} ${seg.kind}${seg.lipsync ? " ♪" : ""} (${mode})`, status: "pending" as const, pct: 0 };
    ctx.setResults([card]);
    try {
      const body: Record<string, unknown> = {
        prompt: seg.prompt, ref_still_ids: refs,
        seconds: seg.render_seconds, mode,
        ...(seed ? { seed } : {}),
        ...(mode === "hunt" ? { drafts: huntN } : {}),
      };
      if (seg.lipsync && audioId)
        body.ref_audio_ids = [{ id: audioId, start: seg.start, seconds: seg.render_seconds }];
      const r = await api.videoH3Ref2V(body) as
        { mode?: string; drafts?: { job_id: string; seed: number }[]; job_id?: string };
      if (r.drafts && !r.drafts.length) {
        // an empty drafts array would otherwise blank the results panel and report nothing
        ctx.patch(card.id, { status: "error", pct: 0, err: "the hunt returned no drafts" });
      } else if (r.drafts) {
        // keep earlier hunts: a second hunt ADDS seeds to choose between instead of discarding the first
        setDrafts((s) => ({ ...(s as Record<number, { jobId: string; seed: number }[]>),
          [i]: [...r.drafts!.map((x) => ({ jobId: x.job_id, seed: x.seed })),
                ...((s as Record<number, { jobId: string; seed: number }[]>)[i] || [])].slice(0, 8) }));
        // ONE setResults for the WHOLE batch. ctx.setResults REPLACES the results array, so a card
        // per draft posted in a loop left only the last one alive - every earlier draft's card was
        // wiped, and its pollJob then patched an id that no longer existed, which is why a 2-take
        // hunt only ever showed the second take's bar moving (2026-08-12).
        const n = r.drafts.length;
        const cards = r.drafts.map((x, k) => ({
          id: rid(), title: `segment ${i + 1} draft ${k + 1}/${n} (seed ${x.seed})`,
          status: "running" as const, pct: 2,
        }));
        ctx.setResults(cards);
        // score each take's audio the moment it lands, so a drifted seed is visible before it is kept
        r.drafts.forEach((x, k) => { track(x.job_id, () => void checkTakeAudio(i)); pollJob(x.job_id, cards[k].id, ctx); });
      } else if (r.job_id) {
        track(r.job_id);
        patchSeg(i, { clipId: r.job_id, clipVariants: [...(seg.clipVariants || []), r.job_id],
                      upscaledId: undefined });   // 2x of the old take no longer applies
        ctx.patch(card.id, { status: "running", pct: 5 });
        pollJob(r.job_id, card.id, ctx);
      }
    } catch (e) { ctx.patch(card.id, { status: "error", pct: 0, err: (e as Error).message }); }
    finally { setGenning(""); }
  }

  // ---- per-shot cast editing. The shot's OWN cast is listed as named rows (a dropdown per
  // person, so picking another name is a 1-for-1 swap in place); everyone else is only reachable
  // through the "+ add" picker. An earlier version showed every cast member as a chip with the
  // ones in the shot merely highlighted, which read as "the video's cast" instead of this shot's.
  // A swap renames the person in the action/scene prose too - "Bob climbs the steps" has to become
  // "Selene climbs the steps", or the text contradicts the reference picture. ----
  function swapShotChar(i: number, j: number, from: string, to: string) {
    const s = segments[i].shots[j];
    if (!to || from === to) return;
    const chars = s.characters.map((n) => (n === from ? to : n));
    const p: Partial<H3Shot> = { characters: chars.filter((n, k) => chars.indexOf(n) === k), cast_locked: true };
    // pronouns only when this is the shot's only character - otherwise a "she" may be someone else
    const solo = s.characters.length === 1;
    const gOf = (n: string) => (h3cast.find((c) => c.name === n)?.gender || "").toLowerCase();
    const rewrite = (t: string) => {
      let out = t.split(from).join(to);
      if (solo) out = swapPronouns(out, gOf(from), gOf(to));
      return out;
    };
    if (s.action.includes(from) || solo) p.action = rewrite(s.action);
    if (s.scene.includes(from) || solo) p.scene = rewrite(s.scene);
    patchShot(i, j, p);
  }
  // ---- cuts. A "single" segment is one continuous take; a "scene" is 2-4 timestamped cuts rendered
  // as ONE clip. Turning one into the other used to need a rewrite, but the usual reason for wanting
  // a second shot is that the singer changes partway through a segment the writer cut as one take -
  // so it has to be an edit, not a regeneration. The compiler collapses any segment back to a single
  // shot when it has one cut, so kind and cuts are kept in step here. ----
  const H3_MAX_SHOTS = 4;      // mv_h3_compile keeps shots_in[:4]
  function addCut(i: number) {
    const seg = segments[i];
    if (seg.shots.length >= H3_MAX_SHOTS) {
      ctx.setResults([{ id: rid(), title: `segment ${i + 1}`, status: "error", pct: 0,
        err: `a segment renders at most ${H3_MAX_SHOTS} cuts in one clip.` }]);
      return;
    }
    const cuts = seg.cuts?.length ? seg.cuts : [{ start: seg.start, end: seg.end }];
    // split the LONGEST cut - the only one guaranteed to have room for two usable shots
    let k = 0;
    cuts.forEach((c, m) => { if (c.end - c.start > cuts[k].end - cuts[k].start) k = m; });
    const c = cuts[k];
    if (c.end - c.start < 2 * H3_MIN_CUT_S) {
      ctx.setResults([{ id: rid(), title: `segment ${i + 1}`, status: "error", pct: 0,
        err: `no room to split: the longest cut is ${(c.end - c.start).toFixed(1)}s and two cuts need ${2 * H3_MIN_CUT_S}s. Move this segment's edges first, or split a neighbour.` }]);
      return;
    }
    // prefer a vocal handover inside the cut: a singer change is nearly always why a second shot is
    // wanted, and landing the boundary there is what makes the lip-sync line up
    const inside = voiceWins.map((w) => w.start)
      .filter((t) => t >= c.start + H3_MIN_CUT_S && t <= c.end - H3_MIN_CUT_S)
      .sort((a, b) => a - b);
    const at = Math.round((inside.length ? inside[0] : (c.start + c.end) / 2) * 100) / 100;
    const nextCuts = [...cuts.slice(0, k), { start: c.start, end: at }, { start: at, end: c.end },
                      ...cuts.slice(k + 1)];
    // the new shot starts as a COPY of the one it splits, so its location, scene text and therefore
    // its environment still stay valid and only the singer/action need changing
    const src = seg.shots[k] || seg.shots[0];
    const nextShots = [...seg.shots.slice(0, k + 1), { ...src, cast_locked: false }, ...seg.shots.slice(k + 1)];
    patchSeg(i, { cuts: nextCuts, shots: nextShots, kind: "scene", promptStale: true });
  }
  function removeCut(i: number, j: number) {
    const seg = segments[i];
    if (seg.shots.length <= 1) return;
    const cuts = seg.cuts.map((c) => ({ ...c }));
    // the removed cut's time goes to a neighbour, so the segment still covers its whole window
    if (j === 0) cuts[1].start = cuts[0].start;
    else cuts[j - 1].end = cuts[j].end;
    cuts.splice(j, 1);
    const shots = seg.shots.filter((_, m) => m !== j);
    patchSeg(i, { cuts, shots, kind: cuts.length > 1 ? "scene" : "single", promptStale: true });
  }
  // Who carries the vocal in a shot. MIRRORS _vocalists() in musicvideo.py: an explicit pick wins,
  // else everyone whose role says singer, else the first person in the shot. Kept in step so the ♪
  // marks in the cast row show what the prompt will actually say.
  const shotSingers = (s: H3Shot): string[] => {
    const here = s.characters.filter((n) => h3cast.some((c) => c.name === n));
    const pick = (s.singers || []).filter((n) => here.includes(n));
    if (pick.length) return pick;
    const byRole = here.filter((n) => (h3cast.find((c) => c.name === n)?.role || "").toLowerCase().includes("singer"));
    return byRole.length ? byRole : here.slice(0, 1);
  };
  function toggleShotSinger(i: number, j: number, name: string) {
    const s = segments[i].shots[j];
    const cur = shotSingers(s);
    const next = cur.includes(name) ? cur.filter((n) => n !== name) : [...s.characters.filter((n) => cur.includes(n) || n === name)];
    // a lip-sync shot with nobody singing is a contradiction - turn lip-sync off for that instead
    if (!next.length) return;
    patchShot(i, j, { singers: next });
  }
  function addShotChar(i: number, j: number, name: string) {
    const s = segments[i].shots[j];
    if (!name || s.characters.includes(name)) return;
    patchShot(i, j, { characters: [...s.characters, name], cast_locked: true });
  }
  function removeShotChar(i: number, j: number, name: string) {
    const s = segments[i].shots[j];
    patchShot(i, j, { characters: s.characters.filter((n) => n !== name), cast_locked: true });
  }

  // Recast every lip-sync shot to the voice its song section calls for, across the whole existing
  // script, and recompile whatever changed. Deterministic and free - no writer run.
  async function fixVoices() {
    if (!songPayload) { ctx.setResults([{ id: rid(), title: "No song arrangement", status: "error", pct: 0, err: "The song's section markers are what say whose voice sings when." }]); return; }
    setRecompiling(true);
    try {
      const r = await api.mvH3VoiceFix({ segments, song: songPayload, cast: castPayload() }) as
        { segments: H3Segment[]; fixes: string[]; fixed_segments: number };
      setSegments(r.segments.map((s) => ({ ...s })));
      ctx.setResults([{ id: rid(), status: "done", pct: 100,
        title: r.fixes.length ? `✓ recast ${r.fixes.length} lip-sync shot(s) in ${r.fixed_segments} segment(s)`
                              : "✓ voice casting already matches the song",
        err: r.fixes.slice(0, 8).join(", ") || undefined }]);
    } catch (e) { ctx.setResults([{ id: rid(), title: "voice check failed", status: "error", pct: 0, err: (e as Error).message }]); }
    finally { setRecompiling(false); }
  }

  // Move segment boundaries onto vocal handovers that land too close to an edge to cut - the fix
  // for "the singer changes in the last half second and the shot keeps lip-syncing".
  async function snapEdges() {
    const miss = castMissing();
    if (miss.length || !songPayload) {
      ctx.setResults([{ id: rid(), title: "not snapping", status: "error", pct: 0,
        err: miss.length ? `${miss.join(", ")} missing from the loaded cast - wait for the character library.`
                         : "no song arrangement, so there is nothing that says where the voice changes." }]);
      return;
    }
    setRecompiling(true);
    try {
      const r = await api.mvH3SnapEdges({ segments, cast: castPayload(), song: songPayload,
        section_grid: segments.map((x) => ({ start: x.start, end: x.end, section: x.section })) }) as
        { segments: H3Segment[]; moved: string[]; recompiled: number[] };
      setSegments(r.segments.map((s) => ({ ...s })));
      const lost = r.segments.filter((s) => s.staleClip).length;
      ctx.setResults([{ id: rid(), status: "done", pct: 100,
        title: r.moved.length ? `✓ moved ${r.moved.length} boundary(s) onto the vocals` : "✓ every boundary already lines up",
        err: [r.moved.join(" {·} "), lost ? `${lost} rendered take(s) need redoing (the old take is still under "takes")` : ""]
          .filter(Boolean).join(" {·} ").replace(/\{·\}/g, "·") || undefined }]);
    } catch (e) { ctx.setResults([{ id: rid(), title: "snap failed", status: "error", pct: 0, err: (e as Error).message }]); }
    finally { setRecompiling(false); }
  }

  // Characters the script names but the loaded cast cannot supply. The character library arrives
  // async, so anything that recompiles must check first: with an empty cast the compiler happily
  // emits prompts with no identity references at all.
  function castMissing(): string[] {
    const have = new Set(h3cast.map((c) => c.name));
    const miss = new Set<string>();
    for (const s of segments) for (const sh of s.shots) for (const n of sh.characters) if (n && !have.has(n)) miss.add(n);
    return [...miss];
  }

  // Refresh every stored prompt the compiler would now emit differently - what you need after a
  // compiler rule changes (the no-singing clause, the reference budget, the band fill) without
  // paying for a rewrite. Hand-edited prompts are skipped server-side and reported.
  async function recompileStale() {
    const miss = castMissing();
    if (miss.length) {
      ctx.setResults([{ id: rid(), title: "not recompiling - cast not loaded", status: "error", pct: 0,
        err: `${miss.join(", ")} are in the script but not in the loaded cast. Recompiling now would strip their identity references - wait for the character library, then retry.` }]);
      return;
    }
    setRecompiling(true);
    try {
      const r = await api.mvH3Recompile({ segments, cast: castPayload(), song: songPayload,
        section_grid: segments.map((x) => ({ start: x.start, end: x.end, section: x.section })) }) as
        { segments: H3Segment[]; changed: number[]; skipped: number[]; voice_fixes: string[] };
      // every segment the server actually compiled now matches its shots, whether or not the text
      // came out different; only the hand-edited ones it refused to touch stay flagged
      setSegments(r.segments.map((s, k) => ({ ...s, promptStale: r.skipped.includes(k + 1) ? s.promptStale : false })));
      const bits = [
        r.changed.length ? `segments ${r.changed.join(", ")}` : "",
        r.voice_fixes.length ? `${r.voice_fixes.length} shot(s) recast` : "",
        r.skipped.length ? `skipped hand-edited ${r.skipped.join(", ")}` : "",
      ].filter(Boolean);
      ctx.setResults([{ id: rid(), status: "done", pct: 100,
        title: r.changed.length ? `✓ recompiled ${r.changed.length} out-of-date segment(s)`
                                : "✓ every prompt is already current",
        err: bits.join(" {·} ").replace(/\{·\}/g, "·") || undefined }]);
    } catch (e) { ctx.setResults([{ id: rid(), title: "recompile failed", status: "error", pct: 0, err: (e as Error).message }]); }
    finally { setRecompiling(false); }
  }

  // the voice map is per SONG, so fetch it when a segment is opened rather than per render
  async function loadVoiceMap() {
    if (!songPayload || !segments.length) { setVoiceWins([]); return; }
    try {
      const r = await api.mvH3VoiceMap({ song: songPayload,
        section_grid: segments.map((x) => ({ start: x.start, end: x.end, section: x.section })) }) as
        { windows: VoiceWin[] };
      setVoiceWins(r.windows || []);
    } catch { setVoiceWins([]); }
  }

  // Score every take of one segment against the song window it was rendered for. Only worth doing
  // for lip-sync segments - a shot with nobody singing has no lips to go out of step.
  async function checkTakeAudio(i: number) {
    const seg = segments[i];
    const ids = [...new Set([...(drafts[i] || []).map((d) => d.jobId), seg?.clipId,
                             ...(seg?.clipVariants || [])].filter(Boolean) as string[])];
    if (!seg?.lipsync || !audioId || !ids.length) return;
    const pending = ids.filter((id) => audioScore[id] === undefined);
    if (!pending.length) return;
    try {
      const r = await api.mvH3AudioCheck({ clip_ids: pending, audio_id: audioId,
        start: seg.start, seconds: seg.render_seconds }) as { scores: Record<string, number | null> };
      setAudioScore((prev) => ({ ...prev, ...r.scores }));
    } catch { /* a missing score just leaves the take unlabelled */ }
  }

  // open a segment in the editor at its first INCOMPLETE stage (the old editor's auto-advance):
  // no scene text yet -> Shots; no background -> Environment; no clip -> Video; else the Result.
  function openSeg(i: number) {
    if (i < 0 || i >= segments.length) return;
    const s = segments[i];
    setEditIdx(i);
    void loadVoiceMap();
    void checkTakeAudio(i);
    setEstep(!s.shots.every((x) => x.scene && x.action) ? "shots"
      : !segLocs(s).every((l) => !!envOf(s, l)) ? "env"
      : !s.clipId ? "video" : "result");
  }
  function keepDraft(i: number, jobId: string, seed: number) {
    const seg = segments[i];
    const r = audioScore[jobId];
    // a drifted take looks fine and only reveals itself once the whole video is assembled, so say
    // it plainly at the moment it would be locked in rather than leaving it to the score chip
    if (r !== undefined && r !== null && r < 0.7 &&
        !window.confirm(`Seed ${seed} scored ${r} for audio match (clean takes are 0.87-1.00).\n\n` +
          `H3 re-rendered the song differently on this seed and lip-synced to its own version, so ` +
          `once the master audio is laid over at assembly the mouths will not match the track.\n\n` +
          `Keep it anyway?`)) return;
    // the upscale belongs to the PREVIOUS take, so drop it rather than assemble a 2x of a clip
    // that is no longer the chosen one
    patchSeg(i, { clipId: jobId, clipVariants: [...(seg.clipVariants || []), jobId], upscaledId: undefined });
    ctx.setResults([{ id: rid(), title: `✓ segment ${i + 1}: draft seed ${seed} kept`, status: "done", pct: 100,
      err: r !== undefined && r !== null && r < 0.7 ? `kept despite audio drift (${r})` : undefined }]);
  }

  // ---- FlashVSR 2x upscale of a segment's chosen take (the same upscaler the LTX lane uses; it
  // auto-chunks long clips server-side). Runs on the box, so it is a real GPU job per segment. ----
  async function upscaleSeg(i: number) {
    const seg = segments[i];
    if (!seg?.clipId) return;
    const card = { id: rid(), title: `segment ${i + 1}: upscaling 2x`, status: "running" as const, pct: 5 };
    ctx.setResults([card]);
    try {
      const { job_id } = await api.videoFlashvsr({ video_id: seg.clipId, scale: 2 }) as { job_id: string };
      patchSeg(i, { upscaledId: job_id });
      track(job_id);
      pollJob(job_id, card.id, ctx);
    } catch (e) { ctx.patch(card.id, { status: "error", pct: 0, err: (e as Error).message }); }
  }
  async function upscaleAll() {
    const todo = segments.map((_, i) => i).filter((i) => segments[i].clipId && !segments[i].upscaledId);
    if (!todo.length) { ctx.setResults([{ id: rid(), title: "Nothing to upscale", status: "error", pct: 0, err: "Every segment with a take is already upscaled." }]); return; }
    for (const i of todo) await upscaleSeg(i);
  }

  async function assemble() {
    const ready = segments.filter((s) => s.clipId);
    if (!ready.length) { ctx.setResults([{ id: rid(), title: "Nothing to assemble", status: "error", pct: 0, err: "Render some segments first." }]); return; }
    const card = { id: rid(), title: `Assemble (${ready.length} segments)`, status: "pending" as const, pct: 0 };
    ctx.setResults([card]);
    try {
      // prefer the upscaled take where one exists, and lift the canvas so a 2x clip is not scaled
      // back down to the 720p default - mixing sizes is fine, assemble pads each clip to the canvas
      const up = ready.filter((s) => s.upscaledId).length;
      const { job_id } = await api.mvAssemble({
        shots: ready.map((s) => ({ clip_id: s.upscaledId || s.clipId, start: s.start, end: s.end })),
        audio_id: audioId, grade,
        width: up ? resW * 2 : resW, height: up ? resH * 2 : resH,
      }) as { job_id: string };
      ctx.patch(card.id, { status: "running", pct: 5 });
      pollJob(job_id, card.id, ctx);
    } catch (e) { ctx.patch(card.id, { status: "error", pct: 0, err: (e as Error).message }); }
  }

  const clipUrl = (id?: string) => (id ? library.find((x) => x.id === id)?.media_url : undefined);
  // The song's playable URL for the segment timeline. Library AUDIO items carry `audio_url` and leave
  // `media_url` null (only video/image items set it), so reading media_url alone gave the timeline no
  // source at all and its play button sat there disabled, doing nothing when clicked.
  const songUrl = (() => {
    const it = library.find((x) => x.id === audioId);
    return it?.audio_url || it?.media_url || undefined;
  })();

  // =====================================================================================
  // SEGMENT EDITOR - the selected segment opens FULL-PAGE, replacing the table, in the same
  // shape as the LTX Shot Editor it succeeds: a numbered stage rail on the left, one card at
  // a time on the right, ticks as each stage completes, and every expensive step gated behind
  // cheap candidates you pick from. Stages: Shots -> Prompt -> Environment -> Video -> Result.
  // =====================================================================================
  const eseg = editIdx >= 0 ? segments[editIdx] : undefined;
  if (eseg) {
    const i = editIdx;
    const refs = refsFor(eseg);
    const segDrafts = drafts[i] || [];
    const locs = segLocs(eseg);
    const done: Record<string, boolean> = {
      shots: eseg.shots.every((s) => s.scene && s.action),
      prompt: !!eseg.prompt,
      env: locs.every((l) => !!envOf(eseg, l)),
      video: !!eseg.clipId,
      result: !!eseg.clipId,
    };
    const stageLabel: Record<string, string> = {
      // "& cast" because the editor opens on the first INCOMPLETE stage, so the per-shot cast
      // controls are usually a click away and were not findable when this just said "Shots"
      shots: "Shots & cast", prompt: "Prompt", env: "Environment", video: "Video", result: "Result",
    };
    return (
      <div className="se-root">
        {/* left rail */}
        <div>
          <GhostButton onClick={() => setEditIdx(-1)}>{"← Segments"}</GhostButton>
          <div className="mt-3 mb-1 text-sm font-semibold text-[var(--color-ink)]">Segment {i + 1}</div>
          <div className="mb-2 text-[10px] leading-relaxed text-[var(--color-muted)]">
            {eseg.start.toFixed(2)}–{eseg.end.toFixed(2)}s ({eseg.seconds.toFixed(2)}s) {"·"} {eseg.section}<br />
            {eseg.kind}{eseg.lipsync ? " ♪ lip-sync" : ""}{eseg.kind === "scene" ? ` ×${eseg.shots.length} cuts` : ""}<br />
            <span title={`H3 renders on its frame grid (frames%17==5), so the render is snapped UP from the ${eseg.seconds.toFixed(2)}s window and the tail past ${eseg.end.toFixed(2)}s is trimmed at assembly - judge drafts on the window, not the tail`}>
              renders {eseg.render_seconds}s / {eseg.frames}f
              {eseg.render_seconds - eseg.seconds > 0.05 ? ` (trimmed to ${eseg.seconds.toFixed(2)}s)` : ""}
            </span>
            {eseg.render_seconds > H3_SEG_MAX_S && (
              <span className="mt-1 block text-amber-400"
                title={`Measured 2026-08-09: past ${H3_SEG_MAX_S}s a heavy-reference render collapses from the tail - the audio diverges from the song and the lip-sync follows the divergence. The identical payload at 10.1s is clean. This segment's window needs to come under 10.13s to render on the next frame step down.`}>
                ⚠ renders past the {H3_SEG_MAX_S}s measured ceiling — audio drift likely
              </span>
            )}
          </div>
          <div className="se-rail">
            {H3_STAGES.map((s) => (
              <button key={s} className={`se-step ${estep === s ? "on" : ""}`} onClick={() => setEstep(s)}>
                <span className={`se-dot ${done[s] ? "ok" : ""}`}>{done[s] ? "✓" : H3_STAGES.indexOf(s) + 1}</span>
                {stageLabel[s]}
              </button>
            ))}
          </div>
          <div className="mt-3 flex gap-1">
            <button onClick={() => openSeg(i - 1)} disabled={i === 0}
              className="flex-1 rounded border border-[var(--color-line)] px-1 py-0.5 text-[10px] text-[var(--color-muted)] hover:text-[var(--color-ink)] disabled:opacity-40">‹ prev</button>
            <button onClick={() => openSeg(i + 1)} disabled={i === segments.length - 1}
              className="flex-1 rounded border border-[var(--color-line)] px-1 py-0.5 text-[10px] text-[var(--color-muted)] hover:text-[var(--color-ink)] disabled:opacity-40">next ›</button>
          </div>
        </div>

        {/* active stage */}
        <div className="flex flex-col gap-3">
          {/* ---------------- SHOTS ---------------- */}
          {estep === "shots" && (
            <div className="se-card flex flex-col gap-3">
              <div>
                <div className="se-h">{eseg.kind === "scene" ? `Shots — ${eseg.shots.length} cuts in one render` : "Shot — one take across the segment"}</div>
                <p className="se-hint">
                  {eseg.kind === "scene"
                    ? "The cuts render as a SINGLE clip at the timestamps below, so keep them visually connected — an evolving viewpoint or a two-thread intercut."
                    : "One continuous shot spanning the whole segment."} The "scene" text is what renders the environment still; "action" is the motion inside it.
                </p>
              </div>
              {/* Editing a shot changes only the structured fields - the compiled prompt below is
                  what actually renders, and it does not follow until recompiled. Say so here, with
                  the fix in reach, rather than letting a swapped character silently fail to reach
                  the render. */}
              {eseg.promptStale && (
                <div className="flex flex-wrap items-center gap-2 rounded border border-amber-500/60 bg-amber-500/10 px-2 py-1.5 text-[10px] text-amber-300">
                  <span>⚠ These edits are not in the prompt yet — the render would still use the previous shots.</span>
                  <GhostButton onClick={() => recompile(i)} disabled={recompiling}>
                    {recompiling ? "Recompiling…" : "Recompile the prompt"}
                  </GhostButton>
                  {eseg.handEdited && <span>(the prompt was hand-edited — recompiling discards that edit)</span>}
                </div>
              )}
              {/* hear the segment and check the boundaries by ear: every cut time and voice handover
                  here is inferred, and a drag lands the cut on what you actually hear */}
              <H3SegTimeline
                url={songUrl}
                start={eseg.start} end={eseg.end} cuts={eseg.cuts} voices={voiceWins}
                labels={eseg.shots.map((s) => (s.characters.join(" + ") || "no cast") + (s.lipsync ? " ♪" : ""))}
                onCutsChange={(c) => patchSeg(i, { cuts: c, promptStale: true })}
                onCommit={() => recompile(i)} />
              {eseg.shots.map((s, j) => (
                <div key={j} className="flex flex-col gap-1.5 rounded-lg border border-[var(--color-line)] p-2">
                  <div className="flex flex-wrap items-center gap-2 text-[11px] text-[var(--color-muted)]">
                    <span className="font-semibold uppercase tracking-wide text-[var(--color-ink)]">
                      {eseg.kind === "scene" ? `cut ${j + 1}` : "shot"}
                    </span>
                    {eseg.cuts[j] && <span>{eseg.cuts[j].start.toFixed(1)}–{eseg.cuts[j].end.toFixed(1)}s</span>}
                    {eseg.shots.length > 1 && (
                      <button onClick={() => removeCut(i, j)}
                        title={`delete this cut - its ${((eseg.cuts[j]?.end ?? 0) - (eseg.cuts[j]?.start ?? 0)).toFixed(1)}s goes to the ${j === 0 ? "next" : "previous"} cut`}
                        className="text-[10px] text-[var(--color-muted)] hover:text-red-400">× delete cut</button>
                    )}
                    <label className="flex items-center gap-1">framing
                      <select className={`${inp} !w-auto`} value={s.framing} onChange={(e) => patchShot(i, j, { framing: e.target.value })}>
                        {["close", "medium", "wide"].map((f) => <option key={f} value={f}>{f}</option>)}
                      </select>
                    </label>
                    <label className="flex items-center gap-1">camera
                      <select className={`${inp} !w-auto`} value={s.camera} onChange={(e) => patchShot(i, j, { camera: e.target.value })}>
                        {H3_CAMERAS.map((cm) => <option key={cm} value={cm}>{cm}</option>)}
                      </select>
                    </label>
                    <label className="flex items-center gap-1" title="the singer performs the lyric on camera; recompiling attaches the song window as <Audio 1> and forces close/medium">
                      <input type="checkbox" checked={s.lipsync} onChange={(e) => patchShot(i, j, { lipsync: e.target.checked })} /> lip-sync
                    </label>
                  </div>
                  {/* who is IN this shot - one row per person, the dropdown swaps them for someone else */}
                  <div className="flex flex-wrap items-center gap-2 rounded border border-dashed border-[var(--color-line)] px-2 py-1.5">
                    <span className="text-[9px] uppercase tracking-wide text-[var(--color-muted)]">in this shot</span>
                    {s.characters.length === 0 && (
                      <span className="text-[10px] text-[var(--color-muted)]">nobody (scenic / b-roll)</span>
                    )}
                    {s.characters.map((n) => (
                      <span key={n} className="flex items-center gap-1 rounded-full border border-[var(--color-accent2)] bg-[#3a2a14] pl-1 pr-0.5 py-0.5">
                        <select className="bg-transparent text-[10px] text-[var(--color-accent2)] outline-none"
                          value={n} onChange={(e) => swapShotChar(i, j, n, e.target.value)}
                          title={`swap ${n} for someone else in this shot (renames them in the text too)`}>
                          {h3cast.filter((c) => c.name === n || !s.characters.includes(c.name))
                            .map((c) => <option key={c.id} value={c.name}>{c.name}</option>)}
                        </select>
                        {/* who actually performs the vocal. Role alone cannot say "both of them are
                            in frame but only one is singing this line" - both singers have a singer
                            role, so both lip-synced however the action was worded (segment 30). */}
                        {s.lipsync && (
                          <button onClick={() => toggleShotSinger(i, j, n)}
                            title={shotSingers(s).includes(n)
                              ? `${n} lip-syncs this line — click so ${n} is in shot but silent`
                              : `${n} is in shot but silent — click to have ${n} lip-sync this line too`}
                            className={`px-0.5 text-[10px] ${shotSingers(s).includes(n)
                              ? "text-[var(--color-accent2)]" : "text-[var(--color-muted)] opacity-50 hover:opacity-100"}`}>
                            ♪
                          </button>
                        )}
                        <button onClick={() => removeShotChar(i, j, n)} title={`take ${n} out of this shot`}
                          className="px-1 text-[10px] text-[var(--color-accent2)] hover:text-red-400">×</button>
                      </span>
                    ))}
                    {/* A hand-cast shot is exempt from voice-matched casting. Togglable BOTH ways on
                        purpose: swapping someone locks the shot automatically, but a shot already
                        holding the right person needs a way to say so without a pointless swap out
                        and back (which is the only thing that fires the dropdown's change event). */}
                    {s.lipsync && s.characters.length > 0 && (
                      <button onClick={() => patchShot(i, j, { cast_locked: !s.cast_locked })}
                        title={s.cast_locked
                          ? "hand-cast: the voice-matched casting check leaves this shot alone. Click to unlock it and let the check recast it again."
                          : "this shot can be recast automatically to match the voice the song's markers say is singing. Click to lock the cast as it is."}
                        className={`rounded border px-1 py-0.5 text-[9px] ${s.cast_locked
                          ? "border-amber-500/60 text-amber-300 hover:text-amber-100"
                          : "border-[var(--color-line)] text-[var(--color-muted)] hover:text-[var(--color-ink)]"}`}>
                        {s.cast_locked ? "🔒 hand-cast" : "🔓 lock cast"}
                      </button>
                    )}
                    {h3cast.some((c) => !s.characters.includes(c.name)) && (
                      <select className={`${inp} !w-auto !text-[10px]`} value=""
                        onChange={(e) => addShotChar(i, j, e.target.value)}
                        title="put another character in this shot as well">
                        <option value="">+ add</option>
                        {h3cast.filter((c) => !s.characters.includes(c.name))
                          .map((c) => <option key={c.id} value={c.name}>{c.name}</option>)}
                      </select>
                    )}
                  </div>
                  <Field label="Location (shared name — one environment still per name, everywhere it appears)">
                    <input className={inp} value={s.location || ""} onChange={(e) => patchShot(i, j, { location: e.target.value })} />
                  </Field>
                  <Field label="Scene — the environment only, no people (this renders the environment still)">
                    <textarea className={inp} rows={3} value={s.scene} onChange={(e) => patchShot(i, j, { scene: e.target.value })} />
                  </Field>
                  <Field label="Action — ONE continuous motion at real-world speed">
                    <textarea className={inp} rows={3} value={s.action} onChange={(e) => patchShot(i, j, { action: e.target.value })} />
                  </Field>
                  {/* The prose is not rewritten when someone is dropped from a shot (there is no safe
                      way to edit an arbitrary sentence), so it can end up asking for a person the
                      compiler then forbids: "Selene and Bob sing together ..." followed by "<Subject 2>
                      is NOT in this shot". H3 follows the prose, so name the clash and let it be fixed. */}
                  {(() => {
                    const stray = h3cast.map((c) => c.name).filter((n) =>
                      !s.characters.includes(n) &&
                      new RegExp(`\\b${n.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`).test(`${s.action} ${s.scene}`));
                    return stray.length ? (
                      <div className="rounded border border-amber-500/60 bg-amber-500/10 px-2 py-1 text-[10px] text-amber-300">
                        ⚠ the text still names {stray.join(", ")}, who {stray.length > 1 ? "are" : "is"} not in this cut.
                        The prompt would ask for {stray.length > 1 ? "them" : "them"} and forbid {stray.length > 1 ? "them" : "them"} in the same breath - reword the action, or add {stray.length > 1 ? "them" : "them"} back to the cut.
                      </div>
                    ) : null;
                  })()}
                </div>
              ))}
              {/* add a cut: the way to make a one-take segment into two shots, e.g. when the singer
                  changes partway through it. Splits on a vocal handover if there is one in range. */}
              <div className="flex flex-wrap items-center gap-2">
                <GhostButton onClick={() => addCut(i)} disabled={eseg.shots.length >= H3_MAX_SHOTS}>
                  + add a cut
                </GhostButton>
                <span className="text-[10px] text-[var(--color-muted)]">
                  {eseg.shots.length >= H3_MAX_SHOTS
                    ? `${H3_MAX_SHOTS} cuts is the most one clip can render.`
                    : `splits the longest cut${voiceWins.some((w) => w.start > eseg.start + H3_MIN_CUT_S && w.start < eseg.end - H3_MIN_CUT_S)
                        ? " on the vocal handover inside it" : " down the middle"}, copying that shot so only the singer and action need changing. Then recompile.`}
                </span>
              </div>
              <Field label="Soundscape (sits under the song on lip-sync segments)">
                <input className={inp} value={eseg.soundscape} onChange={(e) => patchSeg(i, { soundscape: e.target.value })} />
              </Field>
              <div className="se-foot">
                <PrimaryButton onClick={async () => { await recompile(i); setEstep("prompt"); }} disabled={recompiling}>
                  {recompiling ? "Recompiling…" : "Recompile prompt →"}
                </PrimaryButton>
                <span className="text-[10px] text-[var(--color-muted)]">re-applies the enforced rules: pace anchors, sky-pin, subject-swap, reference budget, band fill</span>
              </div>
            </div>
          )}

          {/* ---------------- PROMPT ---------------- */}
          {estep === "prompt" && (
            <div className="se-card flex flex-col gap-3">
              <div>
                <div className="se-h">Prompt</div>
                <p className="se-hint">
                  The compiled six-section prompt that renders. Edit it directly for full control — it then renders verbatim
                  and Recompile discards the edit.
                </p>
              </div>
              <textarea className={`${inp} font-mono`} rows={22} value={eseg.prompt}
                onChange={(e) => patchSeg(i, { prompt: e.target.value, handEdited: true })} />
              <div className="se-foot">
                {/* NOT gated on `busy`. Recompiling is a deterministic CPU-only call that never
                    touches the GPU box, and `busy` is app-wide: one unresolved render card left
                    anywhere (a timed-out download, a dismissed job) disabled this button silently,
                    so a character swap could not be written into the prompt and looked reverted. */}
                <GhostButton onClick={() => recompile(i)} disabled={recompiling}>
                  {recompiling ? "Recompiling…" : "Recompile from the shots"}
                </GhostButton>
                <PrimaryButton onClick={() => setEstep("env")}>Next: Environment →</PrimaryButton>
                {eseg.promptStale && <span className="text-[10px] text-amber-300">⚠ out of date — the shots changed since this was compiled</span>}
                {eseg.handEdited && <span className="text-[10px] text-amber-400">⚠ hand-edited — keep the six sections and the &lt;Picture/Subject/Audio N&gt; tags</span>}
              </div>
            </div>
          )}

          {/* ---------------- ENVIRONMENT ---------------- */}
          {estep === "env" && (
            <div className="se-card flex flex-col gap-4">
              <div>
                <div className="se-h">Environment — the background still</div>
                <p className="se-hint">
                  Person-free environment stills, one per named location. H3 swaps the people out and keeps the place,
                  so the still's realism sets the shot's realism. Generate candidates and pick one — the pick is shared
                  by every segment set in that location.
                </p>
              </div>
              {locs.map((loc) => {
                const chosen = envOf(eseg, loc);
                const cands = envCands[envKey(i, loc)] || [];
                return (
                  <div key={loc} className="flex flex-col gap-2">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-[13px] font-semibold text-[var(--color-ink)]">{locDisplay(eseg, loc)}</span>
                      {!!loc && <span className="text-[10px] text-[var(--color-muted)]">used by {locUsedBy(loc)} segment(s)</span>}
                      <div className="flex-1" />
                      <PrimaryButton onClick={() => genEnvCands(i, loc, 3)} disabled={busy || !!genning}>
                        {genning === `cand:${i}:${loc}` ? "Submitting…"
                          : busy ? "waiting for the current render…"
                            : cands.length ? "Generate 3 more" : "Generate 3 backgrounds"}
                      </PrimaryButton>
                      <GhostButton onClick={() => genEnvCands(i, loc, 1)} disabled={busy || !!genning}>+1</GhostButton>
                    </div>
                    {cands.length > 0 && (
                      <div className="se-grid3">
                        {cands.map((c) => {
                          const st = jobState[c.jobId];
                          const ready = !!st?.url || st === undefined;
                          const picked = chosen === c.jobId;
                          return (
                            <div key={c.jobId} className="se-tile">
                              <div className="se-thumb">
                                {st?.err ? <span className="se-spin text-red-300">failed</span>
                                  : ready ? <img src={`/api/media/${c.jobId}`} alt="" onClick={() => openLightbox(`/api/media/${c.jobId}`)} className="cursor-zoom-in" />
                                    : <span className="se-spin">{st?.pct ? `rendering… ${st.pct}%` : "queued…"}</span>}
                              </div>
                              <div className="flex items-center justify-between px-2 py-1.5">
                                <span className="text-[11px] text-[var(--color-muted)]">seed {c.seed}</span>
                                {picked
                                  ? <span className="text-[11px] text-[var(--color-accent2)]">✓ in use</span>
                                  : <GhostButton onClick={() => pickEnv(i, loc, c.jobId)} disabled={!ready || !!st?.err}>Use this</GhostButton>}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    )}
                    {chosen && !cands.some((c) => c.jobId === chosen) && (
                      <div className="flex items-center gap-3 rounded-lg border border-[var(--color-line)] p-2">
                        <img src={`/api/media/${chosen}`} alt="" onClick={() => openLightbox(`/api/media/${chosen}`)}
                          className="h-16 w-28 cursor-zoom-in rounded object-cover" />
                        <span className="text-[11px] text-[var(--color-accent2)]">✓ environment set</span>
                      </div>
                    )}
                    {!chosen && !cands.length && (
                      <p className="se-hint m-0">Nothing yet for this location.</p>
                    )}
                  </div>
                );
              })}
              <div className="se-foot">
                <PrimaryButton onClick={() => setEstep("video")} disabled={!done.env}>
                  {done.env ? "Backgrounds are good — to Video →" : "Pick a background for every location"}
                </PrimaryButton>
              </div>
            </div>
          )}

          {/* ---------------- VIDEO ---------------- */}
          {estep === "video" && (
            <div className="se-card flex flex-col gap-3">
              <div>
                <div className="se-h">Video options</div>
                <p className="se-hint">
                  Cheap turbo drafts first: pick the seed whose take works, then Finish renders that exact seed on the
                  full-quality base recipe. Drafts accumulate, so a second hunt adds seeds instead of replacing them.
                </p>
              </div>
              <div className="text-[11px] text-[var(--color-muted)]">
                {typeof refs === "string"
                  ? <span className="text-amber-400">⚠ {refs}</span>
                  : <>references in picture order: <span className="text-[var(--color-ink)]">{refs.length} pictures</span>
                    {eseg.lipsync && audioId ? " + the song window as <Audio 1> (the soundtrack IS the song)" : ""}
                    {refs.length >= 8 ? " · at the reference budget" : ""}</>}
              </div>
              <div className="flex flex-wrap items-center gap-3">
                <label className="flex items-center gap-2 text-[11px] text-[var(--color-muted)]">Drafts
                  <select className={`${inp} !w-auto`} value={huntN} onChange={(e) => setHuntN(Number(e.target.value))}>
                    {[2, 3, 4].map((n) => <option key={n} value={n}>{n}</option>)}
                  </select>
                </label>
                <PrimaryButton onClick={() => renderSeg(i, "hunt")} disabled={busy || !!genning || typeof refs === "string"}>
                  {segDrafts.length ? `Generate ${huntN} more options` : `Generate ${huntN} options`}
                </PrimaryButton>
                <GhostButton onClick={() => renderSeg(i, "finish")} disabled={busy || !!genning || typeof refs === "string"}>
                  Skip the hunt — finish now
                </GhostButton>
              </div>
              {segDrafts.length > 0 && (
                <div className="se-grid3">
                  {segDrafts.map((x) => {
                    const st = jobState[x.jobId];
                    const durl = st?.url || clipUrl(x.jobId);
                    const picked = eseg.clipId === x.jobId;
                    return (
                      <div key={x.jobId} className="se-tile">
                        <div className="se-thumb">
                          {st?.err ? <span className="se-spin text-red-300">failed</span>
                            : durl ? <VideoTile src={durl} />
                              : <span className="se-spin">{st?.pct ? `rendering… ${st.pct}%` : "queued…"}</span>}
                        </div>
                        <div className="flex items-center justify-between gap-1 px-2 py-1.5">
                          <span className="text-[11px] text-[var(--color-muted)]">seed {x.seed}
                            {/* H3 re-renders the audio rather than pasting the reference in, and a
                                seed that LOOKS right can have drifted away from the song with the
                                lip-sync following the drift - invisible until the whole video is
                                cut together. Score it here so a bad seed is rejectable now. */}
                            {audioScore[x.jobId] !== undefined && audioScore[x.jobId] !== null && (
                              <span className={audioScore[x.jobId]! < 0.7 ? "ml-1.5 text-red-400" : "ml-1.5 text-[var(--color-muted)]"}
                                title={audioScore[x.jobId]! < 0.7
                                  ? `AUDIO DRIFTED (${audioScore[x.jobId]}). H3 re-rendered the song differently on this seed and lip-synced to its own version, so the mouths will not match the real track once assembled. Clean takes score 0.87-1.00. Do not keep this one however good it looks.`
                                  : `audio follows the song (${audioScore[x.jobId]}) - the lip-sync will match the real track`}>
                                {audioScore[x.jobId]! < 0.7 ? `⚠ audio ${audioScore[x.jobId]}` : `♪ ${audioScore[x.jobId]}`}
                              </span>
                            )}
                          </span>
                          <div className="flex items-center gap-1">
                            {picked
                              ? <span className="text-[11px] text-[var(--color-accent2)]">✓ in use</span>
                              : <GhostButton onClick={() => keepDraft(i, x.jobId, x.seed)} disabled={!durl}>Keep</GhostButton>}
                            <GhostButton onClick={() => renderSeg(i, "finish", x.seed)} disabled={busy || !!genning || !durl}>Finish</GhostButton>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {/* ---------------- RESULT ---------------- */}
          {estep === "result" && (
            <div className="se-card flex flex-col gap-3">
              <div className="se-h">Result</div>
              {eseg.clipId
                ? <video key={eseg.clipId} src={`/api/media/${eseg.clipId}`} controls autoPlay loop muted playsInline
                    className="w-full rounded-lg bg-black" style={{ maxHeight: 440 }} />
                : <p className="se-hint">No clip yet — generate one in the Video stage.</p>}
              {eseg.clipId && (
                <>
                  <div className="se-foot">
                    <PrimaryButton onClick={() => setEditIdx(-1)}>Use in the video ✓</PrimaryButton>
                    <GhostButton onClick={() => setEstep("video")}>Re-roll options</GhostButton>
                    <GhostButton onClick={() => setEstep("env")}>Change the background</GhostButton>
                    {/* FlashVSR 2x, the same upscaler the LTX lane uses. Do it LAST: it is a GPU job
                        per segment and any re-render discards the result. */}
                    <GhostButton onClick={() => upscaleSeg(i)} disabled={busy || !!genning}
                      title="FlashVSR 2x upscale of this take (runs on the box; long clips are auto-chunked). Assembly uses the upscale when one exists.">
                      {eseg.upscaledId ? "Re-upscale 2x" : "Upscale 2x"}
                    </GhostButton>
                    {eseg.upscaledId && (
                      <span className="text-[10px] text-[var(--color-accent2)]"
                        title="assembly will use this instead of the base take">↑2x ready</span>
                    )}
                  </div>
                  {(eseg.clipVariants || []).length > 1 && (
                    <div className="flex flex-wrap items-center gap-1">
                      <span className="text-[10px] text-[var(--color-muted)]">takes:</span>
                      {(eseg.clipVariants || []).map((v, k) => (
                        <button key={v} onClick={() => patchSeg(i, { clipId: v })}
                          className={`se-pill ${eseg.clipId === v ? "on" : ""}`}>take {k + 1}</button>
                      ))}
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <p className="text-[10px] text-[var(--color-muted)]">
        The MiniMax H3 pipeline: the writer cuts the song into <span className="text-[var(--color-ink)]">render segments</span> on
        its real structure ("single" = one shot, always for lip-sync; "scene" = several cuts in one render). Each segment references
        the cast's <span className="text-[var(--color-ink)]">identity sheets + costumes</span> and its own
        <span className="text-[var(--color-ink)]"> environment still</span>; lip-sync segments carry the song itself, so the
        performance is synced to the actual track.
      </p>

      {/* cast + costume selection */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[10px] uppercase tracking-wide text-[var(--color-muted)]">Cast</span>
        {h3cast.length === 0 && <span className="text-[10px] text-amber-400">no H3-style characters with identity sheets yet - build them in Characters</span>}
        {h3cast.map((c) => (
          <span key={c.id} className="flex items-center gap-1 rounded border border-[var(--color-line)] px-1.5 py-0.5 text-[10px]">
            {c.sheetId && <img src={`/api/media/${c.sheetId}`} className="h-5 w-9 rounded object-cover" alt="" />}
            {c.name}
            {(c.costumes || []).length > 0 && (
              <select className="bg-transparent text-[9px] text-[var(--color-muted)]" value={castCostume[c.id] ?? (c.costumes || [])[0]?.id}
                onChange={(e) => setCastCostume({ ...castCostume, [c.id]: e.target.value })} title="costume for this video ((sheet outfit) = wear what the identity sheet shows)">
                {(c.costumes || []).map((co) => <option key={co.id} value={co.id}>{co.name}</option>)}
                <option value="">(sheet outfit)</option>
              </select>
            )}
            {(c.props || []).length > 0 && (
              <select className="bg-transparent text-[9px] text-[var(--color-muted)]" value={castProp[c.id] ?? (c.props || [])[0]?.id}
                onChange={(e) => setCastProp({ ...castProp, [c.id]: e.target.value })} title="instrument/prop for this video">
                {(c.props || []).map((pr) => <option key={pr.id} value={pr.id}>{pr.name}</option>)}
                <option value="">(no prop)</option>
              </select>
            )}
          </span>
        ))}
        <span className="flex-1" />
        <select className="rounded border border-[var(--color-line)] bg-transparent px-1.5 py-0.5 text-[10px] text-[var(--color-muted)]"
          value={scriptLlm} onChange={(e) => setScriptLlm(e.target.value)} disabled={writing}
          title="Which LLM writes the script">
          <option value="claude-sonnet-5">Claude Sonnet 5</option>
          <option value="claude-opus-5">Claude Opus 5</option>
          <option value="claude-sonnet-4-6">Claude Sonnet 4.6</option>
          <option value="local">Local LLM</option>
        </select>
        <PrimaryButton onClick={writeScript} disabled={busy || writing || !audioId}>
          {writing ? "writing…" : segments.length ? "↻ rewrite script" : "Write H3 script"}
        </PrimaryButton>
      </div>

      {/* segment list */}
      {segments.length > 0 && (
        <>
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-[var(--color-muted)]">
              {segments.length} segments {"·"} {segments.filter((s) => s.kind === "single").length} single / {segments.filter((s) => s.kind === "scene").length} scene
              {"·"} {segments.filter((s) => s.clipId).length} rendered
            </span>
            <span className="flex-1" />
            {/* the recompile family is CPU-only and deliberately NOT gated on `busy` - see the
                per-segment Recompile button for why that gate made edits look reverted */}
            <GhostButton onClick={fixVoices} disabled={recompiling || !songPayload}>
              {recompiling ? "checking…" : "Check voice casting"}
            </GhostButton>
            <GhostButton onClick={snapEdges} disabled={recompiling || !songPayload}
              >{recompiling ? "working…" : "Snap edges to vocals"}</GhostButton>
            <GhostButton onClick={recompileStale} disabled={recompiling}
              >{recompiling ? "recompiling…" : "Recompile out-of-date"}
              {segments.some((s) => s.promptStale) ? ` (${segments.filter((s) => s.promptStale).length})` : ""}</GhostButton>
            <GhostButton onClick={genAllEnvs} disabled={busy || !!genning}>Gen missing environments</GhostButton>
            {/* one GPU job per segment, so it names the count rather than starting quietly */}
            <GhostButton onClick={upscaleAll} disabled={busy || !!genning || !segments.some((s) => s.clipId && !s.upscaledId)}
              title="FlashVSR 2x every segment that has a take and is not upscaled yet. One GPU job each, run in sequence - do this once the takes are final, since a re-render discards the upscale.">
              Upscale all 2x{(() => { const n = segments.filter((s) => s.clipId && !s.upscaledId).length; return n ? ` (${n})` : ""; })()}
            </GhostButton>
            <PrimaryButton onClick={assemble} disabled={busy || !segments.some((s) => s.clipId)}
              title={segments.some((s) => s.upscaledId)
                ? `assembles at ${resW * 2}x${resH * 2}, using the 2x upscale for the ${segments.filter((s) => s.upscaledId).length} segment(s) that have one`
                : `assembles at ${resW}x${resH}`}>Assemble</PrimaryButton>
          </div>
          <div className="overflow-hidden rounded-lg border border-[var(--color-line)]">
            <div className="max-h-[520px] overflow-y-auto">
              <table className="w-full border-collapse text-left text-[11px]">
                <thead className="sticky top-0 z-10 bg-[var(--color-panel2)] text-[9px] uppercase tracking-wide text-[var(--color-muted)]">
                  <tr>
                    <th className="w-7 px-1 py-1.5">#</th><th className="w-24 px-2 py-1.5">time</th>
                    <th className="w-20 px-2 py-1.5">kind</th><th className="px-2 py-1.5">content</th>
                    <th className="w-24 px-2 py-1.5">environment</th><th className="w-24 px-2 py-1.5">clip</th>
                    <th className="w-40 px-2 py-1.5">render</th>
                  </tr>
                </thead>
                <tbody>
                  {segments.map((seg, i) => {
                    const url = clipUrl(seg.clipId);
                    const segDrafts = drafts[i] || [];
                    return (<>
                      <tr key={i} className="border-t border-[var(--color-line)] align-top">
                        <td className="px-1 py-1.5 font-semibold text-[var(--color-muted)]">{i + 1}
                          {seg.promptStale && <div className="text-[9px] font-normal text-amber-400"
                            title="the shots were edited but the prompt has not been recompiled - this would render the OLD shots. Use Recompile out-of-date, or open the segment and recompile it.">⚠</div>}</td>
                        <td className="px-2 py-1.5 text-[var(--color-muted)]">
                          {fmt(seg.start)}–{fmt(seg.end)}
                          <div className={`text-[9px] ${seg.render_seconds > H3_SEG_MAX_S ? "text-amber-400" : "opacity-70"}`}
                            title={seg.render_seconds > H3_SEG_MAX_S
                              ? `past the ${H3_SEG_MAX_S}s measured ceiling - the audio drifts from the song and the lip-sync follows it`
                              : undefined}>
                            {seg.section} {"·"} renders {seg.render_seconds}s{seg.render_seconds > H3_SEG_MAX_S ? " ⚠" : ""}
                          </div>
                        </td>
                        <td className="px-2 py-1.5">
                          <span className={`rounded px-1.5 py-0.5 text-[9px] ${seg.lipsync ? "bg-[#3a2a14] text-[var(--color-accent2)]" : seg.kind === "scene" ? "bg-sky-900/50 text-sky-200" : "bg-slate-700/60 text-slate-300"}`}>
                            {seg.kind}{seg.lipsync ? " ♪" : ""}{seg.kind === "scene" ? ` ×${seg.shots.length}` : ""}
                          </span>
                        </td>
                        <td className="px-2 py-1.5">
                          {seg.shots[0]?.location && (
                            <span className="mb-0.5 inline-block rounded bg-[var(--color-panel2)] px-1 text-[8px] uppercase tracking-wide text-[var(--color-muted)]"
                              title="named location - its environment still is shared by every segment set here">
                              {seg.shots[0].location}
                            </span>
                          )}
                          {seg.shots.map((s, j) => (
                            <div key={j} className="truncate text-[10px] text-[var(--color-muted)]" title={`${s.scene} — ${s.action}`}>
                              <span className="text-[var(--color-ink)]">{s.characters.join("+") || s.type}</span> {s.framing} {"·"} {s.action.slice(0, 60)}
                            </div>
                          ))}
                        </td>
                        <td className="px-2 py-1.5">
                          <div className="space-y-1">
                            {segLocs(seg).map((loc) => {
                              const still = envOf(seg, loc);
                              return still ? (
                                <div key={loc} className="space-y-0.5">
                                  <img src={`/api/media/${still}`} onClick={() => openLightbox(`/api/media/${still}`)}
                                    className="h-9 w-16 cursor-zoom-in rounded object-cover" alt=""
                                    title={`"${locDisplay(seg, loc)}" environment${loc ? " (shared)" : ""} — click to enlarge`} />
                                  <button onClick={() => genEnvLoc(i, loc, true)} disabled={busy || !!genning}
                                    title={loc ? `render a NEW "${locDisplay(seg, loc)}" environment (updates everywhere it appears)` : "render a new environment"}
                                    className="block text-[8px] text-[var(--color-muted)] hover:text-[var(--color-ink)] disabled:opacity-50">↻ {locDisplay(seg, loc).slice(0, 12)}</button>
                                </div>
                              ) : (
                                <button key={loc} onClick={() => genEnvLoc(i, loc)} disabled={busy || !!genning}
                                  className="block rounded border border-[var(--color-line)] px-1.5 py-0.5 text-[9px] text-[var(--color-muted)] hover:text-[var(--color-ink)] disabled:opacity-50">
                                  {genning === `env:${i}:${loc}` ? "…" : `gen ${locDisplay(seg, loc).slice(0, 14)}`}
                                </button>
                              );
                            })}
                          </div>
                        </td>
                        <td className="px-2 py-1.5">
                          {url
                            ? <>
                                <video src={`${url}#t=0.5`} muted preload="metadata" onClick={() => openLightbox(url)}
                                  className="h-9 w-16 cursor-zoom-in rounded object-cover" title="rendered clip — click to view" />
                                {seg.upscaledId
                                  ? <span className="block text-[8px] text-[var(--color-accent2)]" title="FlashVSR 2x - assembly will use this">↑2x</span>
                                  : <button onClick={() => upscaleSeg(i)} disabled={busy || !!genning}
                                      title="FlashVSR 2x upscale of this take"
                                      className="block text-[8px] text-[var(--color-muted)] hover:text-[var(--color-ink)] disabled:opacity-50">↑ upscale</button>}
                              </>
                            : seg.staleClip
                              ? <span className="text-[9px] text-amber-400" title="this segment's window moved, so the old take is the wrong length - re-render it (the take is still listed in the editor's Result stage)">⚠ re-render</span>
                              : <span className="text-[9px] text-[var(--color-muted)]">—</span>}
                        </td>
                        <td className="px-2 py-1.5">
                          <div className="flex flex-wrap gap-1">
                            <button onClick={() => openSeg(i)}
                              title="open this segment in the editor: shots, prompt, environment candidates, seed hunt, result"
                              className="rounded border border-[var(--color-line)] px-1.5 py-0.5 text-[9px] text-[var(--color-muted)] hover:text-[var(--color-ink)]">
                              open
                            </button>
                            <button onClick={() => renderSeg(i, "hunt")} disabled={busy || !!genning || segLocs(seg).some((l) => !envOf(seg, l))}
                              title="2 fast turbo drafts to pick a seed"
                              className="rounded border border-[var(--color-line)] px-1.5 py-0.5 text-[9px] text-[var(--color-muted)] hover:text-[var(--color-ink)] disabled:opacity-50">hunt</button>
                            <button onClick={() => renderSeg(i, "finish")} disabled={busy || !!genning || segLocs(seg).some((l) => !envOf(seg, l))}
                              title="one full-quality render (base recipe)"
                              className="rounded border border-[var(--color-line)] px-1.5 py-0.5 text-[9px] text-[var(--color-muted)] hover:text-[var(--color-ink)] disabled:opacity-50">finish</button>
                          </div>
                          {segDrafts.length > 0 && (
                            <div className="mt-1 flex gap-1">
                              {segDrafts.map((x) => (
                                <button key={x.jobId} onClick={() => {
                                    patchSeg(i, { clipId: x.jobId, clipVariants: [...(seg.clipVariants || []), x.jobId] });
                                    ctx.setResults([{ id: rid(), title: `✓ segment ${i + 1}: draft seed ${x.seed} selected`, status: "done", pct: 100 }]);
                                  }}
                                  title={`use draft seed ${x.seed} (or Finish with this seed for full quality)`}
                                  className={`rounded px-1 py-0.5 text-[8px] ${seg.clipId === x.jobId ? "bg-[var(--color-accent2)] text-black" : "border border-[var(--color-line)] text-[var(--color-muted)] hover:text-[var(--color-ink)]"}`}>
                                  s{String(x.seed).slice(-3)}
                                </button>
                              ))}
                              {seg.clipId && segDrafts.some((x) => x.jobId === seg.clipId) && (
                                <button onClick={() => renderSeg(i, "finish", segDrafts.find((x) => x.jobId === seg.clipId)?.seed)}
                                  disabled={busy || !!genning}
                                  title="re-render the picked draft's seed on the full-quality base recipe"
                                  className="rounded border border-[var(--color-accent2)] px-1 py-0.5 text-[8px] text-[var(--color-accent2)] disabled:opacity-50">
                                  finish seed
                                </button>
                              )}
                            </div>
                          )}
                        </td>
                      </tr>
                    </>);
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
      {segments.length === 0 && (
        <div className="rounded-lg border border-dashed border-[var(--color-line)] p-6 text-center text-xs text-[var(--color-muted)]">
          No segments yet. Set the song track, check the cast, then <span className="text-[var(--color-ink)]">Write H3 script</span>.
        </div>
      )}
    </div>
  );
}
