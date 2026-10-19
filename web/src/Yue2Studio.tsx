import { useEffect, useMemo, useState } from "react";
import { api, trackLabel, type Config, type LibItem, type Project, type SongDraft } from "./api";
import { Field, inp, PrimaryButton, GhostButton, SectionTitle, pollJob, rid, type RunCtx } from "./ui";
import { useDraftCtx, useDrafts } from "./drafts";
import { transposeAbc, parseKey, TONICS } from "./ScoreEditor";
import { ScoreEditor } from "./ScoreEditor";

// ---------------------------------------------------------------------------------------------
// YuE2 studio. A THIRD engine, with its own tab, because it wants a third kind of prompt:
//
//   ACE-Step   ~10-12 dense tag phrases            (Song / Generate tabs)
//   Music 3    a 4000+ character field-schema caption (Music 3 tab)
//   YuE2       ONE compact style line + lyrics with bare [Section] tags.
//
// What no other engine here has: the score. YuE2 writes an ABC plan (melody, and with "full" the
// chords too) BEFORE it renders audio, and then sings and plays that plan. So this tab has a
// score stage between the words and the render: write several plans in seconds and pick one,
// edit the one you like (key, tempo, notes), render it, and render it again with another seed
// while the tune holds. Cover mode goes the other way round: transcribe a real recording into a
// score (SheetSage2) and render that in a new style with new words.
//
// Measured on our 3090 2026-09-15: about 225 s of wall per 4-minute song (Music 3: ~485 s).
// The key/tempo written in the style line were ignored on both seeds we tried; Q: and K: in the
// score header are the levers that hold.
// ---------------------------------------------------------------------------------------------

type Props = { cfg: Config; busy: boolean; song: SongDraft | null; projectName?: string; library: LibItem[]; goTo?: (m: string) => void } & RunCtx;
type Schema = { tags: string[]; defaults: Record<string, number | boolean | string>; max_seconds: number; style_guide: { key: string; help: string }[] };
type Plan = { id: string; seed: number; status: "running" | "done" | "error"; abc: string; err?: string; info?: AbcInfo; base?: boolean; edited?: boolean };
export type AbcInfo = { K?: string; Q?: string; M?: string; L?: string; bpm: number | null; bars: number; voices: { id: string; name: string }[]; sections: { name: string; line: number }[]; problems: string[]; seconds?: number | null };
type Take = { id: string; seed: number; title: string; abc: string; mode: string };

const SOURCE_MODES = ["source", "generate", "music3", "yue2", "suno", "song", "restyle", "cover", "mix", "voiceswap", "master", "naturalize"];

// readable names for the band LoRA runs (folder under loras/yue2/ -> [name, final step]), oldest first; the picker lists
// them newest first. See docs/YUE2_TRAINING_PLAN.md.
// "bttlbst" runs were trained with that trigger word first in every caption, so their style line should start with it.
const LORA_RUNS: Record<string, [string, number]> = {
  yue2_beastinblack_v9_wholesong_lora: ["Beast in Black v9 whole-song", 800],
  yue2_battlebeast_lora: ["Battle Beast v4, first 160 s (oldest)", 800],
  yue2_battlebeast_v9_lora: ["Battle Beast v9, first 160 s", 800],
  yue2_battlebeast_v9_wholesong_lora: ["Battle Beast v9 whole-song", 800],
  "yue2_battlebeast_v9_wholesong_kl0.05_lora": ["Battle Beast whole-song, KL 0.05", 800],
  "yue2_battlebeast_v9_wholesong_plr0.2_2000_lora": ["Battle Beast run 4: slow planner, 2000 steps", 2000],
  yue2_battlebeast_effort_v9_wholesong_sep_lora: ["Battle Beast effort captions + vocal stem", 800],
  yue2_battlebeast_effort_v9_wholesong_plr2_800_sep_r64_trig_lora: ["Battle Beast run A: fast planner + stem, bttlbst (drifts off the score)", 800],
  yue2_battlebeast_effort_v9_wholesong_plr2_800_r64_trig_lora: ["Battle Beast run B: fast planner, bttlbst (loses its place)", 800],
  yue2_battlebeast_effort_v9_wholesong_sep_trig_lora: ["Battle Beast run C: effort + vocal stem, bttlbst", 800],
  yue2_battlebeast_effort_v9_wholesong_trig_lora: ["Battle Beast run D: effort, no stem, bttlbst", 800],
};
const loraRun = (l: string) => l.replace(/\\/g, "/").split("/").slice(-2, -1)[0] || "";
const loraStep = (l: string) => {
  const m = l.match(/_0*(\d+)\.safetensors$/);
  return m ? parseInt(m[1]) : (LORA_RUNS[loraRun(l)]?.[1] ?? 0);
};
const loraStepLabel = (l: string) => /_0*\d+\.safetensors$/.test(l) ? `step ${loraStep(l)}` : `step ${loraStep(l) || "?"} (final)`;

export function Yue2StudioForm({ cfg: _cfg, busy, song, projectName, library, goTo, ...ctx }: Props) {
  const d = useDrafts("yue2");
  const draftCtx = useDraftCtx();   // imperative writes into the other tabs' drafts (send-to-tab)
  // the other tabs' drafts, read for the pull buttons
  const sunoD = useDrafts("suno");
  const [sunoStyle] = sunoD.use("style", ""); const [sunoExclude] = sunoD.use("exclude", ""); const [sunoLyrics] = sunoD.use("lyrics", ""); const [sunoTitle] = sunoD.use("title", "");
  const m3D = useDrafts("music3");
  const [m3Fields] = m3D.use<Record<string, string>>("fields", {}); const [m3Lyrics] = m3D.use("lyrics", ""); const [m3Title] = m3D.use("title", "");
  const [converting, setConverting] = useState("");
  const [songPageTitle] = useDrafts("song").use("title", "");
  const [schema, setSchema] = useState<Schema | null>(null);
  const [avail, setAvail] = useState<{ available: boolean; reason?: string; cover?: boolean } | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [importFrom, setImportFrom] = useState("");
  const [note, setNote] = useState("");

  // ---- words ----
  const [title, setTitle] = d.use("title", "");
  const [style, setStyle] = d.use("style", "");
  const [lyrics, setLyrics] = d.use("lyrics", "");
  const [lint, setLint] = useState<string[]>([]);
  // style writer: a brief -> a PROPOSED line (before/after, accept or discard), never applied blind
  const [brief, setBrief] = d.use("brief", "");
  const [writing, setWriting] = useState(false);
  const [styleProp, setStyleProp] = useState<{ style: string; lyrics: string; provider: string } | null>(null);
  // agentic score edit: an instruction -> a PROPOSED score with a summary and structural checks
  const [instruction, setInstruction] = d.use("instruction", "");
  const [editing, setEditing] = useState(false);
  const [editProp, setEditProp] = useState<{ abc: string; summary: string; problems: string[]; before: { K?: string; bpm: number | null; bars: number; sections: string[] }; after: { K?: string; bpm: number | null; bars: number; sections: string[] } } | null>(null);

  // ---- the score ----
  // abc = the score that will be rendered (empty = let the render plan its own, or "off").
  // abcSource says where it came from so the library row and the card can say so.
  const [planMode, setPlanMode] = d.use<"off" | "melody" | "full">("planMode", "full");
  const [abc, setAbc] = d.use("abc", "");
  const [abcSource, setAbcSource] = d.use("abcSource", "");
  const [abcInfo, setAbcInfo] = useState<AbcInfo | null>(null);
  // the plan the loaded score came from: edits to the loaded score (transpose, tempo, text) are written back into
  // that plan, so switching plans with "use" keeps every plan's edits
  const [abcPlanId, setAbcPlanId] = d.use("abcPlanId", "");
  const [plans, setPlans] = d.use<Plan[]>("plans", []);
  const [planCount, setPlanCount] = d.use("planCount", "4");
  const [planSeed, setPlanSeed] = d.use("planSeed", "");
  // meter is the planner's choice, not the style text's: "in 6/8" gave 0/8 plans in 6/8 on a metal
  // ballad, "...lilting jig-like sway" 2/8 (2026-10-17), and a 6/8 plan rendered with the normal
  // style line was heard as 6/8. So: optional feel words used for PLANNING only, and keep planning
  // in rounds until enough plans come out in the wanted meter.
  const [wantMeter, setWantMeter] = d.use("wantMeter", "any");
  const [planFeel, setPlanFeel] = d.use("planFeel", "");
  const [planTries, setPlanTries] = d.use("planTries", "16");
  const [planBusy, setPlanBusy] = useState(false);
  // opt-in band LoRA (AI Toolkit, docs/YUE2_TRAINING_PLAN.md): decoder = sound/voice, planner = writing
  const [lora, setLora] = d.use("lora", "");
  const [loraModel, setLoraModel] = d.use("loraModel", "1.0");
  const [loraClip, setLoraClip] = d.use("loraClip", "1.0");
  // a v9-token LoRA needs the tokenizer's decoder adapter loaded under it (loras/yue2/companions/)
  const [loraBase, setLoraBase] = d.use("loraBase", "");
  // opt-in: with a loaded score, the model may not end the song before (score length - 8 s); band LoRAs
  // were seen ending at exactly 300 s mid-chorus on a 328 s score
  const [endGuard, setEndGuard] = d.use("endGuard", false);
  const [loras, setLoras] = useState<string[]>([]);
  const [loraNote, setLoraNote] = useState("");
  const isCompanion = (l: string) => /[\\/]companions[\\/]/i.test(l);
  // a band LoRA brings its tokenizer adapter: the run name carries the tokenizer version (yue2_<band>_v9_...);
  // v9 LoRAs were trained over nar_lora_joint_v9 and need it loaded under them, v4 (no version in the name) need none
  const pickLora = (l: string) => {
    setLora(l);
    if (!l) return;
    const v = l.replace(/\\/g, "/").split("/").slice(-2, -1)[0]?.match(/_v(\d+)_/)?.[1];
    if (!v || v === "4") { setLoraBase(""); setLoraNote("tokenizer adapter cleared: this LoRA was trained on v4 tokens"); return; }
    const c = loras.find((x) => isCompanion(x) && new RegExp(`nar_lora_joint_v${v}`, "i").test(x));
    if (c) { setLoraBase(c); setLoraNote(`tokenizer adapter set to ${c.replace(/^.*[\\/]/, "").replace(/\.safetensors$/, "")} (this LoRA was trained on v${v} tokens)`); }
    else setLoraNote(`this LoRA was trained on v${v} tokens but no nar_lora_joint_v${v} adapter is in loras/yue2/companions`);
  };
  const loraBody = {
    ...(lora ? { lora, lora_model: parseFloat(loraModel) || 1.0, lora_clip: parseFloat(loraClip) || 0 } : {}),
    ...(loraBase ? { lora_base: loraBase } : {}),
  };
  // opt-in: the base model writes the score (Plan and the plan check run with no LoRA), the render still
  // sings it through the LoRA; base-written scores carried her voice better on two songs (2026-10-18)
  const [baseWrites, setBaseWrites] = d.use("baseWrites", false);
  const baseWriting = baseWrites && !!lora;
  const planLora = baseWriting ? {} : loraBody;
  const byBase = baseWriting ? ", base model writing" : "";

  // ---- cover ----
  const [coverSrc, setCoverSrc] = d.use("coverSrc", "");
  const [coverFile, setCoverFile] = useState<File | null>(null);
  const [coverMode, setCoverMode] = d.use<"melody" | "full">("coverMode", "melody");
  const [coverBusy, setCoverBusy] = useState("");
  // online lyric fetch for the cover source (LRCLIB then lyrics.ovh), sectioned on arrival
  const [lyArtist, setLyArtist] = d.use("lyArtist", "");
  const [lyTitle, setLyTitle] = d.use("lyTitle", "");
  const [lyBusy, setLyBusy] = useState("");

  // ---- render ----
  const [seconds, setSeconds] = d.use("seconds", "360");
  const [seed, setSeed] = d.use("seed", "");
  const [count, setCount] = d.use("count", "1");
  // opt-in: with no score loaded, Generate plans each take on its own first and re-rolls any plan whose score
  // cannot fit Max seconds (a 705 s plan against a 360 s cap gave a crammed, cut take, 2026-10-18)
  const [planCheck, setPlanCheck] = d.use("planCheck", false);
  const [planCheckTries, setPlanCheckTries] = d.use("planCheckTries", "6");
  // with a score loaded from the plan list and more than one take, each take sings a different plan from the list
  // (take 1 = the loaded one, then the list order), each on its own plan seed; off = every take re-sings the loaded score
  const [perPlan, setPerPlan] = d.use("perPlan", true);
  const [showAdv, setShowAdv] = useState(false);
  const [steps, setSteps] = d.use("steps", "");
  const [cfgv, setCfgv] = d.use("cfg", "");
  const [temp, setTemp] = d.use("temp", "");
  const [topP, setTopP] = d.use("topP", "");
  const [topK, setTopK] = d.use("topK", "");
  const [rep, setRep] = d.use("rep", "");
  // semantic-stage CFG ("text guidance"), per render. 2.0 = the user's pick from ladder step B
  // (2026-10-16: 1.0 / 1.5 / 2.0 on one score + 4 seeds); blank = node default (1.0, off)
  const [guidance, setGuidance] = d.use("guidance", "2.0");
  // opt-in experiment: send "[Chorus - powerful belt]" as written instead of stripping to "[Chorus]"
  const [keepTagText, setKeepTagText] = d.use("keepTagText", false);
  const [tiled, setTiled] = d.use("tiled", false);
  const [takes, setTakes] = d.use<Take[]>("takes", []);

  useEffect(() => {
    api.yue2Schema().then(setSchema).catch(() => setSchema(null));
    api.yue2Available().then(setAvail).catch(() => setAvail({ available: false, reason: "backend unreachable" }));
    api.projects().then(setProjects).catch(() => setProjects([]));
    api.yue2Loras().then((r) => setLoras(r.loras || [])).catch(() => setLoras([]));
  }, []);

  // lint as you type (cheap, no GPU)
  useEffect(() => {
    const t = window.setTimeout(() => {
      api.yue2Lint({ style, lyrics, max_seconds: parseFloat(seconds) || 0, keep_tag_text: keepTagText })
        .then((r) => setLint(r.notes || [])).catch(() => setLint([]));
    }, 400);
    return () => window.clearTimeout(t);
  }, [style, lyrics, seconds, keepTagText]);

  // score info as it changes
  useEffect(() => {
    if (!abc.trim()) { setAbcInfo(null); return; }
    const t = window.setTimeout(() => {
      api.yue2AbcInfo(abc).then((info) => {
        setAbcInfo(info);
        if (abcPlanId) setPlans((ps) => ps.map((p) => p.id === abcPlanId && p.abc === abc ? { ...p, info } : p));
      }).catch(() => setAbcInfo(null));
    }, 300);
    return () => window.clearTimeout(t);
  }, [abc]);
  // edits to a score that came from the plan list go back into that plan
  const editScore = (t: string) => {
    setAbc(t);
    if (abcPlanId) setPlans((ps) => ps.map((p) => p.id === abcPlanId ? { ...p, abc: t, edited: true } : p));
  };

  const sections = useMemo(() => (lyrics.match(/\[[^\]]+\]/g) || []).length, [lyrics]);

  // ---------------- 1 · words from a song ----------------
  async function importSong(s: SongDraft | null, label: string, titleGuess?: string) {
    if (!s || !(s.blocks || []).length) { setNote(`${label} has no song arrangement`); return; }
    try {
      const r = await api.yue2FromSong(s);
      setStyle(r.style); setLyrics(r.lyrics);
      if (titleGuess?.trim()) setTitle(titleGuess.trim());
      setNote(`imported ${label}: ${(s.blocks || []).length} sections. The ACE tag list became the style line; `
        + "trim it to one genre plus the sound around it (upstream: more adjectives do not mean more control). "
        + "Key and BPM came along at the end; set them for real in the score header once a plan is written.");
    } catch (e) { setNote(`import failed: ${e}`); }
  }
  async function importProject(id: string) {
    setImportFrom(id);
    if (!id) return;
    try {
      const p = await api.projectGet(id);
      const canon = (p?.data?.song || {}) as Partial<SongDraft>;
      const page = (p?.data?.drafts?.song || {}) as Partial<SongDraft> & { title?: string };
      const m3 = (p?.data?.drafts?.music3 || {}) as { lyrics?: string; fields?: Record<string, string> };
      const y2 = (p?.data?.drafts?.yue2 || {}) as { style?: string; lyrics?: string; title?: string };
      if (y2.lyrics) {
        setStyle(y2.style || ""); setLyrics(y2.lyrics); setTitle(y2.title || p?.name || "");
        setNote(`copied ${p?.name}'s YuE2 draft`); return;
      }
      const blocks = (page.blocks?.length ? page.blocks : canon.blocks) || [];
      if (blocks.length) {
        await importSong({ ...canon, ...page, blocks, key: canon.key || "", bpm: canon.bpm || 0, tags: page.tags || canon.tags || "" } as SongDraft,
          p?.name || "project", (page.title || p?.name || "").trim());
        return;
      }
      if (m3.lyrics) {
        // Music 3 lyrics already carry bare tags; the caption's genre line seeds the style
        const basic = String(m3.fields?.["Basic Attributes"] || "");
        const genre = basic.split(".").map((s) => s.trim()).filter((s) => s && !/bpm|key/i.test(s)).pop() || "";
        const bpm = /bpm is (\d+)/.exec(basic)?.[1];
        setLyrics(m3.lyrics); setTitle(p?.name || "");
        setStyle(["English", genre, bpm ? `${bpm} BPM` : ""].filter(Boolean).join(", "));
        setNote(`copied ${p?.name}'s Music 3 lyrics; the style line is seeded from its genre only - describe the instruments and the voice`);
        return;
      }
      setNote("that project has no song, Music 3 or YuE2 draft to copy");
    } catch (e) { setNote(`could not read that project: ${e}`); }
  }
  function insertTag(tag: string) {
    const t = lyrics.trim();
    setLyrics((t ? t + "\n\n" : "") + `[${tag}]\n`);
  }

  // ---------------- the writers ----------------
  async function writeStyle() {
    if (!brief.trim() && !style.trim()) { setNote("give the writer a brief, or an existing line to improve"); return; }
    setWriting(true); setStyleProp(null);
    try {
      const r = await api.yue2WriteStyle({ brief, style, lyrics });
      setStyleProp(r);
      setNote(r.lyrics ? "the writer proposed a style line and drafted lyrics (the box was empty) - review below" : "the writer proposed a style line - review below");
    } catch (e) { setNote(`style writer failed: ${e}`); }
    setWriting(false);
  }
  async function askEditor() {
    if (!abc.trim()) { setNote("load or write a score first"); return; }
    if (!instruction.trim()) { setNote("say what to change in the score"); return; }
    setEditing(true); setEditProp(null);
    setNote("the editor is rewriting the score: the whole ABC comes back, which takes a few minutes for a full song (measured 4.5 min for 140 bars)...");
    try {
      const r = await api.yue2AgentEdit({ abc, style, lyrics, instruction });
      setEditProp(r);
      setNote(r.problems?.length ? `the editor's score has problems: ${r.problems.join("; ")} - review before accepting` : "the editor proposed a score - review below");
    } catch (e) { setNote(`score editor failed: ${e}`); }
    setEditing(false);
  }

  // ---------------- conversions to / from the other engines ----------------
  // Lyrics keep their words in every direction (only the bracket tags are mapped); the style
  // side is a translation because each engine wants a different prompt shape.
  async function fromSuno() {
    if (!sunoStyle.trim() && !sunoLyrics.trim()) { setNote("the Suno tab has no prompt to pull"); return; }
    try {
      const r = await api.convertSunoToYue2({ style: sunoStyle, exclude: sunoExclude, lyrics: sunoLyrics });
      setStyle(r.style); setLyrics(r.lyrics); if (sunoTitle.trim()) setTitle(sunoTitle);
      setNote("pulled the Suno prompt: its descriptors are the style line, its Exclude field became 'no ...' clauses, enriched tags collapsed to bare sections");
    } catch (e) { setNote(`pull from Suno failed: ${e}`); }
  }
  async function fromMusic3() {
    if (!Object.values(m3Fields).some((v) => (v || "").trim()) && !m3Lyrics.trim()) { setNote("the Music 3 tab has no caption to pull"); return; }
    setConverting("music3");
    try {
      const r = await api.convertMusic3ToYue2({ fields: m3Fields, lyrics: m3Lyrics });
      if (r.style) setStyle(r.style); if (r.lyrics) setLyrics(r.lyrics); if (m3Title.trim()) setTitle(m3Title);
      setNote(`pulled the Music 3 caption, compressed to one line (${r.source}); trim it further if it reads like a paragraph`);
    } catch (e) { setNote(`pull from Music 3 failed: ${e}`); }
    setConverting("");
  }
  async function toSuno() {
    if (!style.trim()) { setNote("write a style line first"); return; }
    setConverting("suno");
    try {
      // the loaded score's K:/Q: give the bpm + key the Suno rules want up front
      const r = await api.convertYue2ToSuno({ style, lyrics, title: title || songPageTitle || projectName, abc });
      draftCtx.set("suno", "title", title || songPageTitle || projectName || "");
      draftCtx.set("suno", "style", r.style); draftCtx.set("suno", "exclude", r.exclude); draftCtx.set("suno", "lyrics", r.lyrics);
      setNote(`sent to the Suno tab (${r.source}); the same lyrics with Suno's tag names, ready for a Suno cover of a YuE2 take`);
      goTo?.("suno");
    } catch (e) { setNote(`send to Suno failed: ${e}`); }
    setConverting("");
  }
  async function toMusic3() {
    if (!style.trim()) { setNote("write a style line first"); return; }
    setConverting("m3");
    setNote("expanding the style line into the 13 Music 3 caption fields (the Music 3 writer, ~20 s)...");
    try {
      const r = await api.convertYue2ToMusic3({ style, lyrics, title: title || songPageTitle || projectName });
      draftCtx.set("music3", "fields", r.fields); draftCtx.set("music3", "lyrics", r.lyrics);
      draftCtx.set("music3", "title", title || songPageTitle || projectName || "");
      setNote("sent to the Music 3 tab: caption written from the style line, lyrics with Music 3's tag names");
      goTo?.("music3");
    } catch (e) { setNote(`send to Music 3 failed: ${e}`); }
    setConverting("");
  }

  // ---------------- 3 · plans (score only, fast) ----------------
  // A poll that gives up. /api/yue2/job answers 404 "unknown job" once a job is gone (score-only
  // jobs live in memory, so a backend restart drops them); a loop that swallowed that spun forever.
  // Stop at once on "unknown job", and after 20 consecutive failures of any other kind.
  function pollGone(e: unknown, fails: { n: number }): string | null {
    const msg = String((e as Error)?.message ?? e);
    if (/unknown job/i.test(msg)) return "job no longer known to the backend (restarted?)";
    fails.n += 1;
    return fails.n >= 20 ? `lost contact with the job (${msg})` : null;
  }
  function pollText(jobId: string, onDone: (abc: string, seedOut: number) => void, onErr: (e: string) => void) {
    const fails = { n: 0 };
    const t = window.setInterval(async () => {
      let j;
      try { j = await api.yue2Job(jobId); fails.n = 0; }
      catch (e) { const gone = pollGone(e, fails); if (gone) { clearInterval(t); onErr(gone); } return; }
      if (j.status === "done") { clearInterval(t); onDone(j.abc || "", j.seed); }
      else if (j.status === "error") { clearInterval(t); onErr(j.error || "error"); }
    }, 1500);
  }
  function waitPlan(jobId: string): Promise<{ abc: string; seed: number }> {
    return new Promise((resolve, reject) => pollText(jobId, (abc, seedOut) => resolve({ abc, seed: seedOut }), reject));
  }
  // the planning-only style: feel words go right after language + genre (the house order puts the
  // genre second), so "English, slow metal power ballad, ..." -> "English, slow metal power ballad,
  // lilting jig-like sway, ..."
  const planStyle = (() => {
    // a LoRA trigger word means nothing to the base model, so base writing plans without it
    const st = baseWriting ? style.replace(/^\s*bttlbst\s*,\s*/i, "") : style;
    const feel = planFeel.trim();
    if (!feel) return st;
    const parts = st.split(",").map((x) => x.trim()).filter(Boolean);
    parts.splice(Math.min(2, parts.length), 0, feel);
    return parts.join(", ");
  })();
  const meterOf = (info?: AbcInfo, abcText?: string) =>
    (info?.M || (abcText || "").match(/^M:\s*(\S+)/m)?.[1] || "").trim();

  async function writePlans() {
    if (planMode === "off") { setNote("plan mode is off: pick melody or full to write a score"); return; }
    if (!style.trim() || !lyrics.trim()) { setNote("a style line and lyrics come first"); return; }
    const n = Math.max(1, Math.min(8, parseInt(planCount) || 1));
    const base = planSeed.trim() ? parseInt(planSeed) : Math.floor(Math.random() * 2 ** 31);
    if (wantMeter !== "any") return writePlansForMeter(n, base);
    const fresh: Plan[] = [];
    for (let i = 0; i < n; i++) {
      const s = base + i;
      try {
        const r = await api.yue2Plan({ style: planStyle, lyrics, mode: planMode, seed: s, abc_seed: s, max_seconds: parseFloat(seconds) || 360, ...planLora });
        const plan: Plan = { id: r.job_id, seed: r.seed, status: "running", abc: "", base: baseWriting };
        fresh.push(plan);
        setPlans((ps) => [plan, ...ps].slice(0, 16));
        pollText(r.job_id,
          async (text, seedOut) => {
            const info = await api.yue2AbcInfo(text).catch(() => undefined);
            setPlans((ps) => ps.map((p) => p.id === r.job_id ? { ...p, status: "done", abc: text, seed: seedOut ?? p.seed, info } : p));
          },
          (e) => setPlans((ps) => ps.map((p) => p.id === r.job_id ? { ...p, status: "error", err: e } : p)));
      } catch (e) { setNote(`plan ${i + 1} failed to queue: ${e}`); }
    }
    setNote(`${fresh.length} plan(s) queued (seeds ${base}..${base + fresh.length - 1}); each is seconds of GPU, not minutes`);
  }
  // rounds of 4 seeds until `want` plans come out in the wanted meter or the try cap is reached;
  // only the matching plans join the list, the note keeps the tally of what else came out
  async function writePlansForMeter(want: number, base: number) {
    const cap = Math.max(want, Math.min(64, parseInt(planTries) || 16));
    const tally: Record<string, number> = {};
    let tried = 0, hits = 0;
    setPlanBusy(true);
    try {
      while (hits < want && tried < cap) {
        const round = Math.min(4, cap - tried);
        const jobs: { id: string; seed: number }[] = [];
        for (let i = 0; i < round; i++) {
          const s = base + tried + i;
          try {
            const r = await api.yue2Plan({ style: planStyle, lyrics, mode: planMode, seed: s, abc_seed: s, max_seconds: parseFloat(seconds) || 360, ...planLora });
            jobs.push({ id: r.job_id, seed: r.seed });
          } catch (e) { setNote(`plan seed ${s} failed to queue: ${e}`); }
        }
        tried += round;
        for (const j of jobs) {
          let out;
          try { out = await waitPlan(j.id); } catch (e) { tally["error"] = (tally["error"] || 0) + 1; continue; }
          const info = await api.yue2AbcInfo(out.abc).catch(() => undefined);
          const m = meterOf(info, out.abc) || "?";
          tally[m] = (tally[m] || 0) + 1;
          if (m === wantMeter && hits < want) {
            hits += 1;
            const plan: Plan = { id: j.id, seed: out.seed ?? j.seed, status: "done", abc: out.abc, info, base: baseWriting };
            setPlans((ps) => [plan, ...ps].slice(0, 16));
          }
        }
        const seen = Object.entries(tally).map(([k, v]) => `${v}x ${k}`).join(", ");
        setNote(`${hits}/${want} plan(s) in ${wantMeter} after ${tried} tries (${seen})` +
          (hits < want && tried < cap ? "; planning more…" : ""));
      }
      if (hits === 0) setNote((nt) => `${nt}. None in ${wantMeter}: try feel words that imply it (6/8: "lilting jig-like sway"), or raise the try cap`);
    } finally { setPlanBusy(false); }
  }
  // every written plan to one key: each plan moves by its own interval from its own key (minor plans land on the
  // minor key, major on the major); nearest = at most 6 semitones either way
  const [allKeyDir, setAllKeyDir] = d.use<"nearest" | "up" | "down">("allKeyDir", "nearest");
  async function transposeAllPlans(targetPc: number) {
    const moved: Record<string, { abc: string; info?: AbcInfo }> = {};
    const skipped: string[] = [];
    for (const p of plans) {
      if (p.status !== "done") continue;
      const key = parseKey(p.info?.K || p.abc.match(/^K:\s*(.+)$/m)?.[1]);
      if (!key) { skipped.push(`seed ${p.seed}: no key`); continue; }
      let st = (((targetPc - key.pc) % 12) + 12) % 12;
      if (allKeyDir === "down") st = st ? st - 12 : 0;
      else if (allKeyDir === "nearest" && st > 6) st -= 12;
      if (!st) continue;
      try {
        const out = transposeAbc(p.abc, st);
        moved[p.id] = { abc: out, info: await api.yue2AbcInfo(out).catch(() => undefined) };
      } catch (e) { skipped.push(`seed ${p.seed}: ${e}`); }
    }
    setPlans((ps) => ps.map((p) => moved[p.id] ? { ...p, abc: moved[p.id].abc, info: moved[p.id].info ?? p.info, edited: true } : p));
    if (abcPlanId && moved[abcPlanId]) setAbc(moved[abcPlanId].abc);
    setNote(`${Object.keys(moved).length} plan(s) moved to ${TONICS[targetPc]}` + (skipped.length ? `; skipped ${skipped.join(", ")}` : ""));
  }
  // which plan each take sings when a plan-list score is loaded and "one plan per take" is on: the loaded plan
  // first, then the other finished plans in list order, cycling (with the next seed) if there are more takes
  function planPool(n: number): { plan: Plan; abc: string; seed: number; round: number }[] {
    const active = abc.trim() && abcPlanId ? plans.find((p) => p.id === abcPlanId) : undefined;
    if (!perPlan || !active || n < 2) return [];
    const pool = [active, ...plans.filter((p) => p.status === "done" && p.id !== active.id && !p.info?.problems?.length)];
    if (pool.length < 2) return [];
    const first = seed.trim() ? parseInt(seed) : active.seed;
    return Array.from({ length: n }, (_v, i) => {
      const k = i % pool.length, round = Math.floor(i / pool.length);
      return { plan: pool[k], abc: k === 0 ? abc : pool[k].abc, seed: (k === 0 ? first : pool[k].seed) + round, round };
    });
  }
  function usePlan(p: Plan) {
    const m = meterOf(p.info, p.abc);
    setAbc(p.abc); setAbcPlanId(p.id); setAbcSource(`plan seed ${p.seed} (${planMode}${p.base ? ", base model writing" : ""}${m ? `, ${m}` : ""}${planFeel.trim() ? `, planned with "${planFeel.trim()}"` : ""})`);
    if (!seed.trim()) setSeed(String(p.seed));
    setNote(`score from plan seed ${p.seed} loaded; edit it below or render it as is`);
  }

  // ---------------- 4 · cover: transcribe a recording ----------------
  async function transcribe() {
    if (!coverSrc && !coverFile) { setNote("pick a library track or upload a file to transcribe"); return; }
    setCoverBusy("uploading to the box…");
    try {
      let r: { job_id: string; mode: string };
      if (coverFile) {
        const fd = new FormData(); fd.append("file", coverFile); fd.append("mode", coverMode);
        r = await api.yue2TranscribeFile(fd);
      } else {
        r = await api.yue2Transcribe(coverSrc, coverMode);
      }
      setCoverBusy(`transcribing (${r.mode})…`);
      const label = coverFile ? coverFile.name : (library.find((t) => t.id === coverSrc) ? trackLabel(library.find((t) => t.id === coverSrc)!, library) : coverSrc);
      pollText(r.job_id,
        (text) => {
          setCoverBusy("");
          setAbc(text); setAbcPlanId(""); setAbcSource(`cover of ${label} (${r.mode})`);
          setNote(`transcribed ${label} (${r.mode}). Check the score: transcription errors carry into the cover. `
            + "Then give it the target style and the lyrics to sing (the original words, or new ones aligned to the same sections).");
        },
        (e) => { setCoverBusy(""); setNote(`transcription failed: ${e}`); });
    } catch (e) { setCoverBusy(""); setNote(`transcribe failed: ${e}`); }
  }

  async function fetchLyrics() {
    const explicit = lyArtist.trim() || lyTitle.trim();
    if (!explicit && !coverSrc) { setNote("give an artist and title, or pick a library track whose tags name the song"); return; }
    if (lyrics.trim() && !window.confirm("Replace the lyrics box with the fetched, sectioned lyrics?")) return;
    setLyBusy("fetching + sectioning…");
    try {
      const r = await api.yue2FetchLyrics({
        artist: lyArtist.trim() || undefined, title: lyTitle.trim() || undefined,
        src: !explicit && coverSrc ? coverSrc : undefined, abc: abc.trim() || undefined,
      });
      setLyrics(r.lyrics); setLyArtist(r.artist || lyArtist); setLyTitle(r.title || lyTitle);
      if (!title.trim() && r.title) setTitle(r.title);
      setNote(`lyrics loaded from ${r.source} (${r.artist} - ${r.title}), sectioned by ${r.tagged_by}`
        + (abc.trim() ? " following the loaded score's sections" : "")
        + ". Check the tags against the score before rendering.");
    } catch (e) { setNote(`lyric fetch failed: ${e}`); }
    setLyBusy("");
  }
  async function retagLyrics() {
    if (!lyrics.trim()) { setNote("nothing in the lyrics box to tag"); return; }
    setLyBusy("sectioning…");
    try {
      const r = await api.yue2SectionLyrics({ lyrics, abc: abc.trim() || undefined });
      setLyrics(r.lyrics);
      setNote(`lyrics re-tagged by ${r.tagged_by}${abc.trim() ? " following the loaded score's sections" : ""}`);
    } catch (e) { setNote(`re-tag failed: ${e}`); }
    setLyBusy("");
  }

  // ---------------- 5 · render ----------------
  async function run() {
    if (!style.trim()) { setNote("write a style line first"); return; }
    if (!lyrics.trim()) { setNote("write lyrics with [section] tags first; they decide the length"); return; }
    const n = Math.max(1, Math.min(8, parseInt(count) || 1));
    const withScore = !!abc.trim();
    // base writing needs each score written before the render (the render would otherwise plan through the LoRA)
    const checked = (planCheck || baseWriting) && !withScore && planMode !== "off";
    const pool = planPool(n);
    const fromPlans = pool.length > 0;
    const maxSec = parseFloat(seconds) || 360;
    const label = title || songPageTitle || projectName || "YuE2";
    const cards = Array.from({ length: n }, (_v, i) => ({
      id: rid(), status: "pending" as const, pct: 0,
      title: `${label}${n > 1 ? ` (take ${i + 1}/${n})` : ""}${fromPlans ? ` · plan seed ${pool[i].plan.seed}` : withScore ? " · scored" : planMode === "off" ? " · direct" : checked ? ` · ${baseWriting ? "base-written" : "checked"} ${planMode} plan` : ` · ${planMode} plan`}`,
    }));
    ctx.setResults(cards);
    const base = seed.trim() ? parseInt(seed) : undefined;
    // plan check: every take's score is written on its own first (seconds of GPU), measured, and re-rolled with
    // a new seed if it is broken, unreadable, longer than Max seconds or implausibly short; renders start after
    const planned: ({ abc: string; seed: number; source: string } | null)[] = fromPlans
      ? pool.map(({ plan: p, abc: a, seed: sd, round }) => ({
          abc: a, seed: sd,
          source: `plan seed ${p.seed} (${planMode}${p.base ? ", base model writing" : ""}${p.info?.K ? `, ${p.info.K}` : ""}` +
            `${p.edited ? ", edited" : ""}${round ? `, take ${round + 1} of this plan` : ""})`,
        }))
      : [];
    if (checked) {
      const tries = Math.max(1, Math.min(20, parseInt(planCheckTries) || 6));
      const notes: string[] = [];
      for (let i = 0; i < n; i++) {
        let got: { abc: string; seed: number; source: string } | null = null;
        const why: string[] = [];
        for (let t = 0; t < tries && !got; t++) {
          const sd = base === undefined ? Math.floor(Math.random() * 2 ** 31) : base + i + t * 1009;
          ctx.patch(cards[i].id, { status: "running", pct: 2, title: `${cards[i].title} · checking plan ${t + 1}/${tries}` });
          try {
            const r = await api.yue2Plan({ style: planStyle, lyrics, mode: planMode, seed: sd, abc_seed: sd, max_seconds: maxSec, ...planLora });
            const out = await waitPlan(r.job_id);
            const info: AbcInfo = await api.yue2AbcInfo(out.abc);
            const used = out.seed ?? sd;
            const secs = info.seconds;
            if (info.problems?.length) why.push(`seed ${used}: ${info.problems[0]}`);
            else if (secs == null) why.push(`seed ${used}: unreadable score`);
            else if (secs > maxSec - 5) why.push(`seed ${used}: ${Math.round(secs)} s score`);
            else if (secs < 60) why.push(`seed ${used}: ${Math.round(secs)} s score, too short`);
            else got = { abc: out.abc, seed: used, source: `plan seed ${used} (${planMode}${byBase}, checked: ${Math.round(secs)} s${t ? `, ${t} rejected` : ""})` };
          } catch (e) { why.push(`seed ${sd}: ${e}`); }
        }
        planned.push(got);
        if (why.length) notes.push(`take ${i + 1}: rejected ${why.join("; ")}`);
        if (got) ctx.patch(cards[i].id, { title: cards[i].title, pct: 4 });
        else ctx.patch(cards[i].id, { status: "error", pct: 0, err: `no plan fitted ${maxSec} s in ${tries} tries (${why.join("; ")})` });
      }
      setNote(notes.length ? `plan check: ${notes.join(" | ")}` : `plan check: all ${n} plan(s) fitted ${maxSec} s first time`);
    }
    for (let i = 0; i < n; i++) {
      const sc = checked || fromPlans ? planned[i] : null;
      if (checked && !sc) continue;
      try {
        const r = await api.yue2Generate({
          style, lyrics, title: label,
          mode: planMode, abc: sc ? sc.abc : withScore ? abc : "",
          source: sc ? sc.source : withScore ? abcSource : "",
          seed: sc ? sc.seed : base === undefined ? undefined : base + i,
          max_seconds: parseFloat(seconds) || 360,
          ...(steps.trim() ? { steps: parseInt(steps) } : {}),
          ...(cfgv.trim() ? { cfg: parseFloat(cfgv) } : {}),
          ...(temp.trim() ? { temperature: parseFloat(temp) } : {}),
          ...(topP.trim() ? { top_p: parseFloat(topP) } : {}),
          ...(topK.trim() ? { top_k: parseInt(topK) } : {}),
          ...(rep.trim() ? { repetition_penalty: parseFloat(rep) } : {}),
          ...(guidance.trim() ? { cfg_scale: parseFloat(guidance) } : {}),
          keep_tag_text: keepTagText,
          ...loraBody,
          ...(endGuard && (withScore || sc) ? { end_guard: true } : {}),
          tiled_decode: tiled,
        });
        ctx.patch(cards[i].id, { status: "running", pct: 5 });
        pollJob(r.job_id, cards[i].id, ctx);
        // remember the take so its generated score can be pulled back in afterwards
        const fails = { n: 0 };
        const t = window.setInterval(async () => {
          let j;
          try { j = await api.yue2Job(r.job_id); fails.n = 0; }
          catch (e) { if (pollGone(e, fails)) clearInterval(t); return; }
          if (j.status === "done" || j.status === "error") {
            clearInterval(t);
            if (j.status === "done") setTakes((ts) => [{ id: r.job_id, seed: r.seed, title: label, abc: j.abc || sc?.abc || abc, mode: fromPlans ? "plan list" : withScore ? "scored" : sc ? `checked ${planMode}` : planMode }, ...ts].slice(0, 12));
          }
        }, 3000);
      } catch (e) {
        ctx.patch(cards[i].id, { status: "error", pct: 0, err: String(e) });
      }
    }
  }

  if (!schema) return <div className="text-sm text-[var(--color-muted)]">loading YuE2 schema…</div>;
  const sources = library.filter((it) => SOURCE_MODES.includes(it.mode) && it.audio_url);
  const perSong = (parseFloat(seconds) || 360) * 0.45;   // ~135 s of wall per 310 s measured (ComfyUI 0.41, 2026-10-16)

  return (
    <div className="space-y-4">
      {avail && !avail.available && (
        <div className="rounded border border-amber-600/50 bg-amber-950/30 p-2 text-xs text-amber-300">
          YuE2 is not ready on the box: {avail.reason}
        </div>
      )}

      {/* ---- 1 · words from a song ---- */}
      <div className="rounded border border-[var(--color-line)] p-3">
        <SectionTitle>1 {"·"} Song</SectionTitle>
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <GhostButton onClick={() => importSong(song, "the loaded song", songPageTitle || projectName)} disabled={!song?.blocks?.length}
            title={song?.blocks?.length ? "style line + tagged lyrics from the song currently loaded" : "no song loaded"}>
            Import loaded song
          </GhostButton>
          <select className={inp} value={importFrom} onChange={(e) => importProject(e.target.value)}
            title="copies that project's words into this tab (its YuE2 draft, else its Song arrangement, else its Music 3 lyrics) - it does not switch projects">
            <option value="">or copy a song from another project…</option>
            {projects.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
          </select>
          <span className="text-[10px] text-[var(--color-muted)]">
            Lyrics decide the length: the model sings everything you give it plus an intro, breaks and an
            outro. Per-section timing does not exist here (same as Music 3); the seconds box is a ceiling.
          </span>
        </div>
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <span className="text-[10px] uppercase tracking-wide text-[var(--color-muted)]">convert</span>
          <GhostButton onClick={fromSuno} title="pull the Suno tab's prompt: style descriptors + exclude -> style line, tags -> bare sections (no LLM)">← from Suno</GhostButton>
          <GhostButton onClick={fromMusic3} disabled={converting === "music3"} title="pull the Music 3 tab's caption, compressed to one style line (LLM with a deterministic fallback)">{converting === "music3" ? "compressing…" : "← from Music 3"}</GhostButton>
          <GhostButton onClick={toSuno} disabled={converting === "suno" || !style.trim()} title="Suno Custom Mode prompt from this style line + lyrics, loaded into the Suno tab (the handoff for covering a YuE2 take on Suno)">{converting === "suno" ? "compiling…" : "→ Suno tab"}</GhostButton>
          <GhostButton onClick={toMusic3} disabled={converting === "m3" || !style.trim()} title="expand this style line into the 13 Music 3 caption fields with the Music 3 writer, loaded into the Music 3 tab">{converting === "m3" ? "writing…" : "→ Music 3 tab"}</GhostButton>
        </div>
        {note && <div className="mt-2 text-[11px] text-[var(--color-accent2)]">{note}</div>}
      </div>

      <div className="grid grid-cols-1 items-start gap-4 xl:grid-cols-2">
      <div className="space-y-4">
        {/* ---- 2 · style ---- */}
        <div className="rounded border border-[var(--color-line)] p-3">
          <div className="flex items-center justify-between">
            <SectionTitle>2 {"·"} Style line</SectionTitle>
            <span className="text-[10px] text-[var(--color-muted)]">{style.length} chars</span>
          </div>
          <textarea className={inp + " mt-2 min-h-[96px] text-[11px]"} rows={4} value={style}
            onChange={(e) => setStyle(e.target.value)}
            placeholder="English, symphonic metal power ballad, 140 BPM, E minor, distorted rhythm guitars panned wide, double-kick drums in the choruses, orchestra behind the guitars, female vocal, operatic soprano with bel canto vibrato, romantic and haunting, no growls" />
          <div className="mt-2 grid grid-cols-1 gap-1 md:grid-cols-2">
            {schema.style_guide.map((g) => (
              <div key={g.key} className="text-[10px] leading-snug text-[var(--color-muted)]">
                <span className="text-[var(--color-ink)]">{g.key}</span> {"·"} {g.help}
              </div>
            ))}
          </div>
          <div className="mt-3 border-t border-[var(--color-line)] pt-2">
            <div className="text-[10px] uppercase tracking-wide text-[var(--color-muted)]">style writer</div>
            <textarea className={inp + " mt-1 min-h-[52px] text-[11px]"} rows={2} value={brief} onChange={(e) => setBrief(e.target.value)}
              placeholder="Brief in plain words: the song, the singer, the feel, what must not be in it. With a line already written, the brief edits it ('heavier', 'add a choir on the last chorus'). Empty lyrics box = it drafts lyrics too." />
            <div className="mt-1 flex items-center gap-2">
              <GhostButton onClick={writeStyle} disabled={writing} title="one LLM pass by upstream's own style-line rules; lands as a proposal">{writing ? "writing…" : "Write style line"}</GhostButton>
            </div>
            {styleProp && (
              <div className="mt-2 rounded border border-[var(--color-accent2)]/40 bg-[#0e1015] p-2">
                <div className="flex items-center gap-2">
                  <span className="text-[11px] text-[var(--color-accent2)]">proposed {"·"} via {styleProp.provider}</span>
                  <span className="flex-1" />
                  <GhostButton onClick={() => { setStyle(styleProp.style); if (styleProp.lyrics && !lyrics.trim()) setLyrics(styleProp.lyrics); setStyleProp(null); }}>use this</GhostButton>
                  <GhostButton onClick={() => setStyleProp(null)}>discard</GhostButton>
                </div>
                {style.trim() && <div className="mt-1 rounded bg-red-950/20 p-1 text-[10px] leading-relaxed text-red-200/70"><span className="mr-1 opacity-60">now</span>{style}</div>}
                <div className="mt-1 rounded bg-emerald-950/20 p-1 text-[10px] leading-relaxed text-emerald-200/80"><span className="mr-1 opacity-60">proposed</span>{styleProp.style}</div>
                {styleProp.lyrics && !lyrics.trim() && <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap rounded bg-emerald-950/20 p-1 text-[10px] text-emerald-200/80">{styleProp.lyrics}</pre>}
              </div>
            )}
          </div>
          <div className="mt-2 text-[10px] text-[var(--color-muted)]">
            One compact line, strongest identity first. Living artist names are ignored; describe the audible
            qualities instead. BPM here is a soft prior (honoured about half the time upstream, two of our seeds
            ignored it) and a key is never honoured from text; K: and Q: in the score are the levers that hold.
          </div>
        </div>

        {/* ---- 2 · lyrics ---- */}
        <div className="rounded border border-[var(--color-line)] p-3">
          <SectionTitle>2 {"·"} Lyrics</SectionTitle>
          <div className="mt-2 flex flex-wrap items-center gap-1">
            {schema.tags.map((t) => (
              <GhostButton key={t} onClick={() => insertTag(t)} title={`append a bare [${t}] section`}>+{t}</GhostButton>
            ))}
            <span className="ml-2 text-[10px] text-[var(--color-muted)]">{sections} sections</span>
            <span className="flex-1" />
            <GhostButton onClick={retagLyrics} disabled={!!lyBusy || !lyrics.trim()}
              title="split untagged or badly tagged words into bare sections (repeats = chorus; a loaded score's markers set the form)">
              {lyBusy === "sectioning…" ? "sectioning…" : "Re-tag sections"}
            </GhostButton>
          </div>
          {lint.length > 0 && (
            <div className="mt-2 rounded border border-amber-600/50 bg-amber-950/30 p-2 text-[11px] text-amber-300">
              {lint.map((l, i) => <div key={i}>{l}</div>)}
            </div>
          )}
          <textarea className={inp + " mt-2 min-h-[260px] font-mono text-[11px]"} rows={16}
            value={lyrics} onChange={(e) => setLyrics(e.target.value)}
            placeholder={"[Intro]\n\n[Verse]\nyour words here\n\n[Chorus]\n…"} />
          <div className="mt-1 text-[10px] text-[var(--color-muted)]">
            Tags stay bare: [Verse], [Verse 2], [Pre-Chorus], [Guitar Solo] are all fine; anything else inside the
            brackets is stripped before sending. Empty bodies under every tag = instrumental. Words in parentheses may be sung.
          </div>
          <label className="mt-1 flex items-center gap-2 text-[11px]"
            title="sends tags exactly as written; untested on YuE2 (its training lyrics only had bare section names)">
            <input type="checkbox" checked={keepTagText} onChange={(e) => setKeepTagText(e.target.checked)} />
            send tag descriptors as written, e.g. [Chorus - powerful belt] (experimental)
          </label>
        </div>
      </div>

      <div className="space-y-4">
        {/* ---- 3 · score ---- */}
        <div className="rounded border border-[var(--color-line)] p-3">
          <div className="flex items-center justify-between">
            <SectionTitle>3 {"·"} Score</SectionTitle>
            <span className="text-[10px] text-[var(--color-muted)]">
              {abc.trim() ? `loaded: ${abcSource || "edited"}` : planMode === "off" ? "none (direct render)" : `the render will plan its own (${planMode})`}
            </span>
          </div>
          <div className="mt-2 grid grid-cols-2 gap-2 md:grid-cols-4">
            <Field label="Plan mode" hint="full = melody + chords (default); melody = melody only, accompaniment free (covers); off = no score">
              <select className={inp} value={planMode} onChange={(e) => setPlanMode(e.target.value as "off" | "melody" | "full")}>
                <option value="full">full</option>
                <option value="melody">melody</option>
                <option value="off">off</option>
              </select>
            </Field>
            <Field label="Plans to write" hint="seeds tried; seconds each">
              <input className={inp} value={planCount} onChange={(e) => setPlanCount(e.target.value)} />
            </Field>
            <Field label="First seed" hint="blank = random">
              <input className={inp} value={planSeed} onChange={(e) => setPlanSeed(e.target.value)} placeholder="random" />
            </Field>
            <div className="flex items-end">
              <GhostButton onClick={writePlans} disabled={busy || planBusy || planMode === "off" || !!(avail && !avail.available)} className="w-full"
                title="run only the planner: write N scores from N seeds, then pick one below">
                {planBusy ? "planning…" : "Write plans"}
              </GhostButton>
            </div>
            <Field label="Wanted meter" hint="keep planning until enough match">
              <select className={inp} value={wantMeter} onChange={(e) => setWantMeter(e.target.value)}>
                {["any", "4/4", "2/4", "3/4", "6/8"].map((m) => <option key={m} value={m}>{m}</option>)}
              </select>
            </Field>
            {wantMeter !== "any" && (
              <Field label="Try cap" hint="most plans to try">
                <input className={inp} value={planTries} onChange={(e) => setPlanTries(e.target.value)} placeholder="16" />
              </Field>
            )}
            <div className="col-span-2">
              <Field label="Planning-only feel words" hint={'plans only; the render keeps your style line. 6/8: "lilting jig-like sway"'}>
                <input className={inp} value={planFeel} onChange={(e) => setPlanFeel(e.target.value)} placeholder="optional, e.g. lilting jig-like sway" />
              </Field>
            </div>
          </div>
          {planFeel.trim() && (
            <div className="mt-1 text-[10px] text-[var(--color-muted)]">planning with: {planStyle}</div>
          )}
          {plans.filter((p) => p.status === "done").length > 1 && (
            <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px]">
              <span className="text-[var(--color-muted)]">all plans</span>
              <select className={inp + " w-auto"} value="" disabled={busy}
                onChange={(e) => { if (e.target.value !== "") transposeAllPlans(parseInt(e.target.value)); }}
                title="moves every written plan (notes, chord symbols, K:) to this key, each from its own key; minor plans go to the minor key. The loaded score follows if it is one of them.">
                <option value="">to key...</option>
                {TONICS.map((t, i) => <option key={t} value={i}>{t}</option>)}
              </select>
              <select className={inp + " w-auto"} value={allKeyDir} onChange={(e) => setAllKeyDir(e.target.value as "nearest" | "up" | "down")}
                title="nearest = at most 6 semitones either way">
                <option value="nearest">nearest</option>
                <option value="up">up</option>
                <option value="down">down</option>
              </select>
              <span className="text-[10px] text-[var(--color-muted)]">edits to a loaded plan (transpose, tempo, text) stay with that plan when you switch</span>
            </div>
          )}
          {plans.length > 0 && (
            <div className="mt-2 max-h-48 overflow-auto rounded border border-[var(--color-line)]">
              {plans.map((p) => (
                <div key={p.id} className={"flex items-center gap-2 border-b border-[var(--color-line)] px-2 py-1 text-[11px] last:border-0" + (p.id === abcPlanId && abc.trim() ? " bg-[var(--color-accent)]/10" : "")}>
                  <span className="w-24 text-[var(--color-ink)]">seed {p.seed}</span>
                  {p.status === "running" && <span className="text-[var(--color-muted)]">planning…</span>}
                  {p.status === "error" && <span className="text-red-400">{p.err}</span>}
                  {p.status === "done" && (
                    <span className="flex-1 text-[var(--color-muted)]">
                      {p.info ? `${meterOf(p.info, p.abc) || "?"} · ${p.info.K || "?"} · ${p.info.bpm ?? "?"} bpm · ${p.info.bars} bars · ${p.info.sections.map((s) => s.name).join(" ")}` : `${p.abc.length} chars`}
                    </span>
                  )}
                  <span className="flex-1" />
                  {p.edited && <span className="text-[10px] text-[var(--color-muted)]">edited</span>}
                  {p.id === abcPlanId && abc.trim() && <span className="text-[10px] text-[var(--color-accent)]">loaded</span>}
                  {p.status === "done" && <GhostButton onClick={() => usePlan(p)}>use</GhostButton>}
                  <GhostButton onClick={() => setPlans((ps) => ps.filter((x) => x.id !== p.id))}>×</GhostButton>
                </div>
              ))}
            </div>
          )}

          <ScoreEditor abc={abc} setAbc={(t) => { editScore(t); if (t.trim() && abcSource && !abcSource.endsWith("(edited)")) setAbcSource(abcSource + " (edited)"); }}
            info={abcInfo} onClear={() => { setAbc(""); setAbcSource(""); setAbcPlanId(""); }} />

          <div className="mt-3 border-t border-[var(--color-line)] pt-2">
            <div className="text-[10px] uppercase tracking-wide text-[var(--color-muted)]">ask the editor</div>
            <div className="mt-1 flex items-center gap-2">
              <input className={inp} value={instruction} onChange={(e) => setInstruction(e.target.value)} disabled={!abc.trim()}
                placeholder="e.g. lift the last chorus a tone, make the solo eight bars longer, reharmonise the bridge to the relative major, halve the intro" />
              <GhostButton onClick={askEditor} disabled={editing || !abc.trim()} title="the app LLM edits the ABC to the instruction; comes back as a proposal with before/after checks">{editing ? "editing…" : "Ask"}</GhostButton>
            </div>
            {editProp && (
              <div className="mt-2 rounded border border-[var(--color-accent2)]/40 bg-[#0e1015] p-2">
                <div className="flex items-center gap-2">
                  <span className="text-[11px] text-[var(--color-accent2)]">proposed edit</span>
                  <span className="flex-1" />
                  <GhostButton onClick={() => { editScore(editProp.abc); setAbcSource((abcSource || "score") + " (edited by the editor)"); setEditProp(null); }}>use this</GhostButton>
                  <GhostButton onClick={() => setEditProp(null)}>discard</GhostButton>
                </div>
                <div className="mt-1 text-[10px] leading-relaxed text-emerald-200/80">{editProp.summary || "(no summary)"}</div>
                <div className="mt-1 text-[10px] text-[var(--color-muted)]">
                  {editProp.before.K || "?"} {editProp.before.bpm ?? "?"} bpm {editProp.before.bars} bars [{editProp.before.sections.join(" ")}]
                  {" -> "}
                  {editProp.after.K || "?"} {editProp.after.bpm ?? "?"} bpm {editProp.after.bars} bars [{editProp.after.sections.join(" ")}]
                </div>
                {editProp.problems?.length > 0 && <div className="mt-1 text-[10px] text-red-400">{editProp.problems.join("; ")}</div>}
              </div>
            )}
          </div>

          {takes.length > 0 && (
            <div className="mt-2">
              <div className="text-[10px] uppercase tracking-wide text-[var(--color-muted)]">scores from recent takes</div>
              <div className="mt-1 flex flex-wrap gap-1">
                {takes.map((t) => (
                  <GhostButton key={t.id} onClick={() => { setAbc(t.abc); setAbcPlanId(""); setAbcSource(`take ${t.title} seed ${t.seed}`); if (!seed.trim()) setSeed(String(t.seed)); }}
                    title="load the score this take was rendered from, to edit and render again">
                    {t.title} {"·"} seed {t.seed} {"·"} {t.mode}
                  </GhostButton>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* ---- 4 · cover ---- */}
        <div className="rounded border border-[var(--color-line)] p-3">
          <SectionTitle>4 {"·"} Cover (score from a recording)</SectionTitle>
          <div className="mt-2 text-[10px] leading-relaxed text-[var(--color-muted)]">
            SheetSage2 transcribes a recording into a score; the score then renders with the style line and
            lyrics above. <b>melody</b> keeps the tune and frees the arrangement (recommended for a style
            change); <b>full</b> keeps the chords too. This is not voice cloning: the singer comes from the
            style line. Transcription errors carry into the cover, so read the score before rendering.
          </div>
          <div className="mt-2 grid grid-cols-1 gap-2 md:grid-cols-3">
            <Field label="Library track">
              <select className={inp} value={coverSrc} onChange={(e) => { setCoverSrc(e.target.value); if (e.target.value) setCoverFile(null); }}>
                <option value="">— pick a track —</option>
                {sources.map((t) => <option key={t.id} value={t.id}>{trackLabel(t, library)}</option>)}
              </select>
            </Field>
            <Field label="…or upload" hint="overrides the picker">
              <input className={inp} type="file" accept="audio/*" onChange={(e) => { setCoverFile(e.target.files?.[0] ?? null); if (e.target.files?.[0]) setCoverSrc(""); }} />
            </Field>
            <Field label="Transcribe as">
              <select className={inp} value={coverMode} onChange={(e) => setCoverMode(e.target.value as "melody" | "full")}>
                <option value="melody">melody only</option>
                <option value="full">melody + chords</option>
              </select>
            </Field>
          </div>
          <div className="mt-2 flex items-center gap-2">
            <GhostButton onClick={transcribe} disabled={!!coverBusy || !!(avail && (!avail.available || avail.cover === false))}
              title={avail?.cover === false ? "SheetSage2 weight missing on the box (download_yue2.bat)" : "transcribe on the box GPU, then load the score above"}>
              {coverBusy || "Transcribe → score"}
            </GhostButton>
            <span className="text-[10px] text-[var(--color-muted)]">
              then set the plan mode to match ({coverMode}) and press Generate.
            </span>
          </div>
          <div className="mt-3 border-t border-[var(--color-line)] pt-2">
            <div className="text-[10px] uppercase tracking-wide text-[var(--color-muted)]">the words</div>
            <div className="mt-1 grid grid-cols-1 gap-2 md:grid-cols-3">
              <Field label="Artist" hint="blank = from the track's tags"><input className={inp} value={lyArtist} onChange={(e) => setLyArtist(e.target.value)} /></Field>
              <Field label="Song title"><input className={inp} value={lyTitle} onChange={(e) => setLyTitle(e.target.value)} /></Field>
              <div className="flex items-end">
                <GhostButton onClick={fetchLyrics} disabled={!!lyBusy} className="w-full"
                  title="LRCLIB then lyrics.ovh; the words come untagged, so they are split into bare sections on arrival, following the transcribed score's markers when one is loaded">
                  {lyBusy === "fetching + sectioning…" ? lyBusy : "Fetch lyrics → sections"}
                </GhostButton>
              </div>
            </div>
            <div className="mt-1 text-[10px] text-[var(--color-muted)]">
              Transcribe first, then fetch: the score's section markers (intro, verse, chorus...) decide where the
              tags go. Online sources carry no tags; repeated stanzas become the chorus. Read the result against
              the score, the melody is fixed and the words must land in the same sections.
            </div>
          </div>
        </div>

        {/* ---- 5 · render ---- */}
        <div className="rounded border border-[var(--color-line)] p-3">
          <SectionTitle>5 {"·"} Render</SectionTitle>
          <div className="mt-2 grid grid-cols-2 gap-2 md:grid-cols-5">
            <Field label="Title"><input className={inp} value={title} onChange={(e) => setTitle(e.target.value)} /></Field>
            <Field label="Max seconds" hint={`ceiling, up to ${schema.max_seconds}; a hard cut if the words run past it`}>
              <input className={inp} value={seconds} onChange={(e) => setSeconds(e.target.value)} />
            </Field>
            <Field label="Seed" hint="blank = random per take; with a score loaded, takes re-perform the same tune">
              <input className={inp} value={seed} onChange={(e) => setSeed(e.target.value)} placeholder="random" />
            </Field>
            <Field label="Takes"><input className={inp} value={count} onChange={(e) => setCount(e.target.value)} /></Field>
            <Field label="Text guidance" hint="1.0 = off; 2.0 our pick; 2.4 the stable max">
              <input className={inp} value={guidance} onChange={(e) => setGuidance(e.target.value)} placeholder="1.0" />
            </Field>
          </div>
          <div className="mt-2 grid grid-cols-2 gap-2 md:grid-cols-6">
            <div className="col-span-2">
              <Field label="Band LoRA" hint="optional; trained with AI Toolkit, copied into ComfyUI by sync">
                <select className={inp} value={lora} onChange={(e) => pickLora(e.target.value)}>
                  <option value="">(none)</option>
                  {[...new Set(loras.filter((l) => !isCompanion(l)).map(loraRun))]
                    .sort((a, b) => (Object.keys(LORA_RUNS).indexOf(b) + 1 || -1) - (Object.keys(LORA_RUNS).indexOf(a) + 1 || -1)).map((run) => (
                    <optgroup key={run} label={LORA_RUNS[run]?.[0] || run || "loras/yue2"}>
                      {loras.filter((l) => !isCompanion(l) && loraRun(l) === run).sort((a, b) => loraStep(a) - loraStep(b))
                        .map((l) => <option key={l} value={l}>{LORA_RUNS[run] ? `${LORA_RUNS[run][0]} - ${loraStepLabel(l)}` : l.replace(/^yue2[\\/]/, "")}</option>)}
                    </optgroup>
                  ))}
                </select>
              </Field>
            </div>
            <Field label="Tokenizer adapter" hint="set automatically when you pick a band LoRA (v9 LoRAs need nar_lora_joint_v9, v4 none); change it here if needed">
              <select className={inp} value={loraBase} onChange={(e) => setLoraBase(e.target.value)}>
                <option value="">(none)</option>
                {loras.filter(isCompanion).map((l) => <option key={l} value={l}>{l.replace(/^.*[\\/]/, "").replace(/\.safetensors$/, "")}</option>)}
              </select>
            </Field>
            <Field label="Sound strength" hint="decoder: voice, timbre, production; 1.0 (0.8 if the top end thins)">
              <input className={inp} value={loraModel} onChange={(e) => setLoraModel(e.target.value)} placeholder="1.0" />
            </Field>
            <Field label="Writing strength" hint="planner: melody, phrasing; 1.0 slow songs, 0.5 fast ones">
              <input className={inp} value={loraClip} onChange={(e) => setLoraClip(e.target.value)} placeholder="1.0" />
            </Field>
            <div className="flex items-end">
              <GhostButton className="w-full" title="copy new checkpoints from AI Toolkit's output into ComfyUI's loras/yue2 folder"
                onClick={async () => {
                  setLoraNote("syncing…");
                  try { const r = await api.yue2LorasSync(); setLoras(r.loras || []); setLoraNote(`${r.copied.length} new, ${r.already_present} already there`); }
                  catch (e) { setLoraNote(`sync failed: ${e}`); }
                }}>sync LoRAs</GhostButton>
            </div>
          </div>
          {loraNote && <div className="mt-1 text-[10px] text-[var(--color-muted)]">{loraNote}</div>}
          <label className="mt-1 flex items-center gap-2 text-[11px]"
            title="needs a loaded score; uses the ComfyUI-MusicGen-YuE2 node on the box. Slower while the guard holds.">
            <input type="checkbox" checked={endGuard} onChange={(e) => setEndGuard(e.target.checked)} />
            score-length guard: don't let the song end before the loaded score does (recommended with band LoRAs)
          </label>
          <label className="mt-1 flex items-center gap-2 text-[11px]"
            title="Plan and the plan check run on the base model with no LoRA; the render then sings that score through the band LoRA. With no score loaded, Generate writes each take's score first (as the plan check does).">
            <input type="checkbox" checked={baseWrites} onChange={(e) => setBaseWrites(e.target.checked)} />
            base model writes the score, the LoRA sings it
            {baseWrites && !lora && <span className="text-[var(--color-muted)]">(no band LoRA picked, so this does nothing)</span>}
          </label>
          <div className="mt-2">
            <GhostButton onClick={() => setShowAdv(!showAdv)}>{showAdv ? "hide" : "show"} sampler settings</GhostButton>
          </div>
          {showAdv && (
            <div className="mt-2 grid grid-cols-2 gap-2 md:grid-cols-6">
              <Field label="steps" hint="32"><input className={inp} value={steps} onChange={(e) => setSteps(e.target.value)} placeholder="32" /></Field>
              <Field label="cfg" hint="1.0"><input className={inp} value={cfgv} onChange={(e) => setCfgv(e.target.value)} placeholder="1.0" /></Field>
              <Field label="temperature" hint="music stage, 1.0"><input className={inp} value={temp} onChange={(e) => setTemp(e.target.value)} placeholder="1.0" /></Field>
              <Field label="top_p" hint="0.95"><input className={inp} value={topP} onChange={(e) => setTopP(e.target.value)} placeholder="0.95" /></Field>
              <Field label="top_k" hint="100"><input className={inp} value={topK} onChange={(e) => setTopK(e.target.value)} placeholder="100" /></Field>
              <Field label="rep. penalty" hint="1.2; 1.0-1.05 keeps repeated hooks"><input className={inp} value={rep} onChange={(e) => setRep(e.target.value)} placeholder="1.2" /></Field>
              <Field label="Tiled decode" hint="lower VRAM; off on 24 GB">
                <input type="checkbox" checked={tiled} onChange={(e) => setTiled(e.target.checked)} />
              </Field>
            </div>
          )}
          <label className="mt-3 flex flex-wrap items-center gap-2 text-[11px]"
            title="only when no score is loaded: each take's score is written first (seconds of GPU), its length measured, and a broken or too-long plan is re-rolled with a new seed before any audio is rendered; the render then uses that score (and the score-length guard if ticked)">
            <input type="checkbox" checked={planCheck} onChange={(e) => setPlanCheck(e.target.checked)} />
            check the plan first: re-roll scores that won't fit Max seconds
            {planCheck && <><span className="text-[var(--color-muted)]">tries per take</span>
              <input className={inp + " w-14"} value={planCheckTries} onChange={(e) => setPlanCheckTries(e.target.value)} /></>}
            {planCheck && abc.trim() && <span className="text-[var(--color-muted)]">(a score is loaded, so this is skipped)</span>}
          </label>
          {abc.trim() && abcPlanId && parseInt(count) > 1 && (
            <label className="mt-1 flex flex-wrap items-center gap-2 text-[11px]"
              title="each take sings a different plan from the list: take 1 the loaded one, then the list order (each on its own plan seed); off = every take re-sings the loaded score">
              <input type="checkbox" checked={perPlan} onChange={(e) => setPerPlan(e.target.checked)} />
              one plan per take
              {(() => {
                const pp = planPool(Math.max(1, Math.min(8, parseInt(count) || 1)));
                return pp.length
                  ? <span className="text-[var(--color-muted)]">{pp.map((x, i) => `take ${i + 1}: seed ${x.plan.seed}${x.plan.info?.K ? ` ${x.plan.info.K}` : ""}${x.round ? ` (again, seed ${x.seed})` : ""}`).join(" · ")}</span>
                  : perPlan ? <span className="text-[var(--color-muted)]">(only one usable plan in the list, so every take sings it)</span> : null;
              })()}
            </label>
          )}
          <div className="mt-2 flex items-center gap-2">
            <PrimaryButton onClick={run} disabled={busy || !!(avail && !avail.available)}>
              Generate
            </PrimaryButton>
            <span className="text-[10px] text-[var(--color-muted)]">
              about {Math.round(perSong)} s of GPU per take at this length (measured ~135 s for 310 s of audio, guidance 2.0 included). 48 kHz
              stereo FLAC, saved with its score. Weights CC BY-NC 4.0 with a creator permission: personal creators may monetize outputs.
            </span>
          </div>
        </div>
      </div>
      </div>
    </div>
  );
}
