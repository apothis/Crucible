import { useState } from "react";
import { api, type LibItem } from "./api";
import { Field, inp, PrimaryButton, GhostButton, rid, pollJob, type RunCtx } from "./ui";
import { useDrafts } from "./drafts";
import { openLightbox } from "./Lightbox";
import { type Character } from "./mvmodel";

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
  handEdited?: boolean;   // raw prompt overridden by hand (a recompile clears this)
};

const H3_CAMERAS = ["static", "push in", "pull back", "truck left", "truck right",
  "arc left", "arc right", "tilt up", "crane up"];
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
      k === i ? { ...s, shots: s.shots.map((sh, m) => (m === j ? { ...sh, ...p } : sh)) } : s));
  async function recompile(i: number) {
    const seg = segments[i];
    setRecompiling(true);
    try {
      // the song goes with it: its section markers are what enforce voice-matched casting server-side
      const r = await api.mvH3Compile({ segment: seg, cast: castPayload(), song: songPayload }) as
        { prompt: string; picture_map: Record<string, number>; outfit_map: Record<string, number>;
          env_picture: number; lipsync: boolean; shots: H3Shot[]; kind: "single" | "scene"; cuts: { start: number; end: number }[] };
      patchSeg(i, { prompt: r.prompt, picture_map: r.picture_map, outfit_map: r.outfit_map,
        env_picture: r.env_picture, lipsync: r.lipsync, shots: r.shots, kind: r.kind, cuts: r.cuts,
        handEdited: false });
      ctx.setResults([{ id: rid(), title: `segment ${i + 1} recompiled`, status: "done", pct: 100 }]);
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
  function track(jobId: string) {
    if (jobState[jobId]?.url) return;
    setJobState((s) => ({ ...s, [jobId]: { pct: s[jobId]?.pct ?? 2 } }));
    const t = window.setInterval(async () => {
      const j = await api.job(jobId).catch(() => null);
      if (!j) return;
      if (j.status === "done" && j.media_url) {
        window.clearInterval(t);
        setJobState((s) => ({ ...s, [jobId]: { pct: 100, url: j.media_url } }));
      } else if (j.status === "error" || j.status === "failed") {
        window.clearInterval(t);
        setJobState((s) => ({ ...s, [jobId]: { pct: 0, err: j.error || "render error" } }));
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
        track(r.job_id);
      }
      const key = envKey(i, loc);
      setEnvCands((prev) => ({ ...(prev as Record<string, { jobId: string; seed: number }[]>),
        [key]: [...made, ...((prev as Record<string, { jobId: string; seed: number }[]>)[key] || [])].slice(0, 9) }));
      ctx.setResults([{ id: rid(), title: `${n} "${locDisplay(seg, loc)}" candidates rendering`,
        status: "running", pct: 5, err: "pick one when they land" }]);
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
      if (r.drafts) {
        // keep earlier hunts: a second hunt ADDS seeds to choose between instead of discarding the first
        setDrafts((s) => ({ ...(s as Record<number, { jobId: string; seed: number }[]>),
          [i]: [...r.drafts!.map((x) => ({ jobId: x.job_id, seed: x.seed })),
                ...((s as Record<number, { jobId: string; seed: number }[]>)[i] || [])].slice(0, 8) }));
        ctx.patch(card.id, { status: "running", pct: 5 });
        r.drafts.forEach((x) => {
          track(x.job_id);
          const c2 = { id: rid(), title: `segment ${i + 1} draft (seed ${x.seed})`, status: "running" as const, pct: 2 };
          ctx.setResults([c2]);
          pollJob(x.job_id, c2.id, ctx);
        });
        ctx.patch(card.id, { status: "done", pct: 100, err: `${r.drafts.length} drafts rendering - pick one, then Finish with its seed` });
      } else if (r.job_id) {
        track(r.job_id);
        patchSeg(i, { clipId: r.job_id, clipVariants: [...(seg.clipVariants || []), r.job_id] });
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
    const p: Partial<H3Shot> = { characters: chars.filter((n, k) => chars.indexOf(n) === k) };
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
  function addShotChar(i: number, j: number, name: string) {
    const s = segments[i].shots[j];
    if (!name || s.characters.includes(name)) return;
    patchShot(i, j, { characters: [...s.characters, name] });
  }
  function removeShotChar(i: number, j: number, name: string) {
    const s = segments[i].shots[j];
    patchShot(i, j, { characters: s.characters.filter((n) => n !== name) });
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

  // open a segment in the editor at its first INCOMPLETE stage (the old editor's auto-advance):
  // no scene text yet -> Shots; no background -> Environment; no clip -> Video; else the Result.
  function openSeg(i: number) {
    if (i < 0 || i >= segments.length) return;
    const s = segments[i];
    setEditIdx(i);
    setEstep(!s.shots.every((x) => x.scene && x.action) ? "shots"
      : !segLocs(s).every((l) => !!envOf(s, l)) ? "env"
      : !s.clipId ? "video" : "result");
  }
  function keepDraft(i: number, jobId: string, seed: number) {
    const seg = segments[i];
    patchSeg(i, { clipId: jobId, clipVariants: [...(seg.clipVariants || []), jobId] });
    ctx.setResults([{ id: rid(), title: `✓ segment ${i + 1}: draft seed ${seed} kept`, status: "done", pct: 100 }]);
  }

  async function assemble() {
    const ready = segments.filter((s) => s.clipId);
    if (!ready.length) { ctx.setResults([{ id: rid(), title: "Nothing to assemble", status: "error", pct: 0, err: "Render some segments first." }]); return; }
    const card = { id: rid(), title: `Assemble (${ready.length} segments)`, status: "pending" as const, pct: 0 };
    ctx.setResults([card]);
    try {
      const { job_id } = await api.mvAssemble({
        shots: ready.map((s) => ({ clip_id: s.clipId, start: s.start, end: s.end })),
        audio_id: audioId, grade, width: resW, height: resH,
      }) as { job_id: string };
      ctx.patch(card.id, { status: "running", pct: 5 });
      pollJob(job_id, card.id, ctx);
    } catch (e) { ctx.patch(card.id, { status: "error", pct: 0, err: (e as Error).message }); }
  }

  const clipUrl = (id?: string) => (id ? library.find((x) => x.id === id)?.media_url : undefined);

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
      shots: "Shots", prompt: "Prompt", env: "Environment", video: "Video", result: "Result",
    };
    return (
      <div className="se-root">
        {/* left rail */}
        <div>
          <GhostButton onClick={() => setEditIdx(-1)}>{"← Segments"}</GhostButton>
          <div className="mt-3 mb-1 text-sm font-semibold text-[var(--color-ink)]">Segment {i + 1}</div>
          <div className="mb-2 text-[10px] leading-relaxed text-[var(--color-muted)]">
            {fmt(eseg.start)}–{fmt(eseg.end)} {"·"} {eseg.section}<br />
            {eseg.kind}{eseg.lipsync ? " ♪ lip-sync" : ""}{eseg.kind === "scene" ? ` ×${eseg.shots.length} cuts` : ""}<br />
            renders {eseg.render_seconds}s / {eseg.frames}f
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
              {eseg.shots.map((s, j) => (
                <div key={j} className="flex flex-col gap-1.5 rounded-lg border border-[var(--color-line)] p-2">
                  <div className="flex flex-wrap items-center gap-2 text-[11px] text-[var(--color-muted)]">
                    <span className="font-semibold uppercase tracking-wide text-[var(--color-ink)]">
                      {eseg.kind === "scene" ? `cut ${j + 1}` : "shot"}
                    </span>
                    {eseg.cuts[j] && <span>{eseg.cuts[j].start.toFixed(1)}–{eseg.cuts[j].end.toFixed(1)}s</span>}
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
                        <button onClick={() => removeShotChar(i, j, n)} title={`take ${n} out of this shot`}
                          className="px-1 text-[10px] text-[var(--color-accent2)] hover:text-red-400">×</button>
                      </span>
                    ))}
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
                </div>
              ))}
              <Field label="Soundscape (sits under the song on lip-sync segments)">
                <input className={inp} value={eseg.soundscape} onChange={(e) => patchSeg(i, { soundscape: e.target.value })} />
              </Field>
              <div className="se-foot">
                <PrimaryButton onClick={async () => { await recompile(i); setEstep("prompt"); }} disabled={recompiling || busy}>
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
                <GhostButton onClick={() => recompile(i)} disabled={recompiling || busy}>
                  {recompiling ? "Recompiling…" : "Recompile from the shots"}
                </GhostButton>
                <PrimaryButton onClick={() => setEstep("env")}>Next: Environment →</PrimaryButton>
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
                        {genning === `cand:${i}:${loc}` ? "Submitting…" : cands.length ? "Generate 3 more" : "Generate 3 backgrounds"}
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
                          <span className="text-[11px] text-[var(--color-muted)]">seed {x.seed}</span>
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
            <GhostButton onClick={fixVoices} disabled={busy || recompiling || !songPayload}>
              {recompiling ? "checking…" : "Check voice casting"}
            </GhostButton>
            <GhostButton onClick={genAllEnvs} disabled={busy || !!genning}>Gen missing environments</GhostButton>
            <PrimaryButton onClick={assemble} disabled={busy || !segments.some((s) => s.clipId)}>Assemble</PrimaryButton>
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
                        <td className="px-1 py-1.5 font-semibold text-[var(--color-muted)]">{i + 1}</td>
                        <td className="px-2 py-1.5 text-[var(--color-muted)]">
                          {fmt(seg.start)}–{fmt(seg.end)}
                          <div className="text-[9px] opacity-70">{seg.section} {"·"} renders {seg.render_seconds}s</div>
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
                            ? <video src={`${url}#t=0.5`} muted preload="metadata" onClick={() => openLightbox(url)}
                                className="h-9 w-16 cursor-zoom-in rounded object-cover" title="rendered clip — click to view" />
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
