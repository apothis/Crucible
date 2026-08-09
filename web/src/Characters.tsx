import { useEffect, useState } from "react";
import { api, type LibItem } from "./api";
import { inp, rid, type RunCtx } from "./ui";
import { Collapse, StillPick } from "./mvui";
import { openLightbox } from "./Lightbox";
import { type Character, type Costume, type Identity, type Wardrobe } from "./mvmodel";

// ============================================================================
// H3-era character creator (docs/MINIMAX_H3_PLAN.md "Outfit-layering test").
// A character = a locked IDENTITY (face + body description) anchored by ONE canonical
// identity SHEET (the only face source, never re-rendered), plus COSTUMES - person-free
// three-view garment sheets layered onto the character at render time by MiniMax H3.
// The guided flow is three steps: 1 describe the identity, 2 pick the sheet, 3 add outfits.
// The legacy MSR reference UI survives below in a collapsed section (old pipeline).
// ============================================================================

const CHAR_REF_W = 1280, CHAR_REF_H = 720;
const PHOTO_POS = "candid photograph, photorealistic, natural realistic skin with visible pores and texture, sharp focus, 50mm";
const PHOTO_NEG = "anime, cartoon, illustration, painting, drawing, 3d render, cgi, video game, doll, " +
  "plastic skin, waxy skin, airbrushed, overly smooth skin, beauty filter, overly symmetrical face, " +
  "low quality, blurry, deformed, bad anatomy, extra fingers, watermark, text";
const FACE_VARY = [
  "softer rounder face shape, fuller cheeks, gentler features",
  "angular face, sharp defined cheekbones and strong jawline",
  "narrow oval face, slim refined features, straight nose",
  "broad strong face, wider-set eyes, heavier brow",
];
const BODY_VARY = ["slender lean build", "athletic toned build", "solid stocky build", "tall willowy build"];

// The guided identity fields, each with a concrete example as its placeholder. They compose
// (in this order) into the character's locked identity block.
const IDENTITY_FIELDS: { key: keyof Identity; label: string; hint: string }[] = [
  { key: "build", label: "Age & build", hint: "e.g. a woman in her mid-20s with a slender build" },
  { key: "face",  label: "Face",        hint: "e.g. an oval face with softly defined cheekbones" },
  { key: "eyes",  label: "Eyes & skin", hint: "e.g. dark brown eyes and pale porcelain skin" },
  { key: "hair",  label: "Hair",        hint: "e.g. long straight black center-parted hair falling past her shoulders" },
  { key: "marks", label: "Distinguishing details", hint: "e.g. a small mole below the left eye (optional)" },
];
const COSTUME_HINT = "e.g. a floor-length deep crimson velvet evening gown with a fitted bodice, " +
  "a sweetheart neckline, long fitted velvet sleeves and a full heavy skirt";

function composeIdentity(idn?: Identity): string {
  if (!idn) return "";
  return IDENTITY_FIELDS.map((f) => (idn[f.key] as string | undefined || "").trim())
    .filter(Boolean).join(", ");
}

type Draft = { jobId: string; seed: number; url?: string; err?: boolean; pct?: number };

function waitMedia(jobId: string, onPct?: (p: number) => void): Promise<string> {
  return new Promise((resolve, reject) => {
    const t = window.setInterval(async () => {
      const j = await api.job(jobId).catch(() => null) as { status?: string; media_url?: string; error?: string; progress?: number; max?: number } | null;
      if (!j) return;
      if ((j.status === "running" || j.status === "finalizing") && onPct)
        onPct(j.max ? Math.max(2, Math.round((100 * (j.progress || 0)) / j.max)) : 2);
      if (j.status === "done" && j.media_url) { onPct?.(100); clearInterval(t); resolve(j.media_url + "?t=" + Date.now()); }
      else if (j.status === "error") { clearInterval(t); reject(new Error(j.error || "error")); }
    }, 1000);
  });
}

// Candidate strip. `wide` renders 16:9 tiles (identity/costume SHEETS); default is square.
function DraftStrip({ drafts, picked, onPick, onReroll, onClose, busy, wide }:
  { drafts: Draft[]; picked?: string; onPick: (jobId: string) => void; onReroll: () => void; onClose: () => void; busy: boolean; wide?: boolean }) {
  if (!drafts.length) return null;
  return (
    <div className="space-y-1 rounded border border-[var(--color-line)] bg-[var(--color-bg)] p-1.5">
      <div className="flex items-center justify-between px-0.5">
        <span className="text-[9px] text-[var(--color-muted)]">click to enlarge · "use this" to keep</span>
        <button onClick={onClose} className="text-[10px] leading-none text-[var(--color-muted)] hover:text-red-400" title="close candidates">×</button>
      </div>
      <div className={`grid gap-1.5 ${wide ? "grid-cols-1 sm:grid-cols-2" : "grid-cols-2"}`}>
        {drafts.map((d) => (
          <div key={d.jobId}
            className={`overflow-hidden rounded border ${picked === d.jobId ? "border-[var(--color-accent2)] ring-1 ring-[var(--color-accent2)]" : "border-[var(--color-line)]"}`}>
            <div className={`relative ${wide ? "aspect-video" : "aspect-square"}`}>
              {d.err
                ? <span className="flex h-full items-center justify-center text-[9px] text-red-400">failed</span>
                : d.url
                ? <img src={d.url} alt="" onClick={() => openLightbox(d.url!)} title={`seed ${d.seed} — click to enlarge`} className="h-full w-full cursor-zoom-in object-cover" />
                : (
                  <span className="flex h-full flex-col items-center justify-center gap-1 text-[9px] text-[var(--color-muted)]">
                    <span>{d.pct ? `${d.pct}%` : "queued…"}</span>
                    <span className="h-0.5 w-3/4 overflow-hidden rounded bg-[var(--color-line)]">
                      <span className="block h-full bg-[var(--color-accent2)] transition-all" style={{ width: `${d.pct || 3}%` }} />
                    </span>
                  </span>
                )}
              {picked === d.jobId && <span className="absolute right-0.5 top-0.5 rounded bg-[var(--color-accent2)] px-1 text-[8px] text-black">using</span>}
            </div>
            {d.url && (
              <button onClick={() => onPick(d.jobId)}
                className={`w-full py-0.5 text-[9px] ${picked === d.jobId ? "bg-[var(--color-accent2)] text-black" : "text-[var(--color-muted)] hover:text-[var(--color-ink)]"}`}>
                {picked === d.jobId ? "✓ in use" : "use this"}
              </button>
            )}
          </div>
        ))}
      </div>
      <button onClick={onReroll} disabled={busy} className="w-full rounded border border-[var(--color-line)] py-0.5 text-[9px] text-[var(--color-muted)] hover:text-[var(--color-ink)] disabled:opacity-50">
        {busy ? "rerolling…" : "↻ reroll"}
      </button>
    </div>
  );
}

function StepTag({ n, done }: { n: number; done: boolean }) {
  return (
    <span className={`flex h-4 w-4 shrink-0 items-center justify-center rounded-full text-[9px] font-bold ${done ? "bg-[var(--color-accent2)] text-black" : "border border-[var(--color-line)] text-[var(--color-muted)]"}`}>
      {done ? "✓" : n}
    </span>
  );
}

export function CharacterLibrary({ chars, setChars, reload, stills, busy, collapsible = false, ...ctx }:
  { chars: Character[]; setChars: (c: Character[]) => void; reload: () => void; stills: LibItem[];
    busy: boolean; collapsible?: boolean } & RunCtx) {
  const [open, setOpen] = useState(false);
  const [editId, setEditId] = useState("");
  const [enhancing, setEnhancing] = useState("");
  const [drafts, setDrafts] = useState<Record<string, Draft[]>>({});
  const [hunting, setHunting] = useState("");
  const [matchFace, setMatchFace] = useState<Record<string, boolean>>({});
  const [vary, setVary] = useState<Record<string, boolean>>({});
  const [newCostume, setNewCostume] = useState<Record<string, { name: string; desc: string }>>({});
  const [llmProvider, setLlmProvider] = useState("ollama");
  useEffect(() => { api.llmProviders().then((p) => {
    const q = p as { claude?: boolean; claude_sub?: boolean };
    setLlmProvider(q.claude_sub ? "claude_sub" : q.claude ? "claude" : "ollama");
  }).catch(() => {}); }, []);

  function persist(c: Character) { api.characterSave(c).catch(() => {}); }
  function patch(c: Character, p: Partial<Character>) {
    const next = { ...c, ...p };
    setChars(chars.map((x) => x.id === c.id ? next : x));
    persist(next);
  }
  async function add() {
    const r = await api.characterSave({ name: "New character", kind: "musician", method: "auto", style: "h3" }) as Character;
    await reload(); setEditId(r.id);
  }
  async function del(id: string) { await api.characterDelete(id); await reload(); if (editId === id) setEditId(""); }

  const setIdentity = (c: Character, p: Partial<Identity>) => {
    const identity = { ...(c.identity || {}), ...p };
    // guided fields compose into the locked identity block (appearance) live
    const composed = composeIdentity(identity);
    patch(c, { identity, ...(composed ? { appearance: composed } : {}) });
  };
  const setWardrobe = (c: Character, wid: string, p: Partial<Wardrobe>) =>
    patch(c, { wardrobes: (c.wardrobes || []).map((w) => w.id === wid ? { ...w, ...p } : w) });
  const addWardrobe = (c: Character) =>
    patch(c, { wardrobes: [...(c.wardrobes || []), { id: rid(), name: "New look", outfitPrompt: "" }] });
  const delWardrobe = (c: Character, wid: string) =>
    patch(c, { wardrobes: (c.wardrobes || []).filter((w) => w.id !== wid) });
  const setCostume = (c: Character, coid: string, p: Partial<Costume>) =>
    patch(c, { costumes: (c.costumes || []).map((co) => co.id === coid ? { ...co, ...p } : co) });
  const delCostume = (c: Character, coid: string) =>
    patch(c, { costumes: (c.costumes || []).filter((co) => co.id !== coid) });

  function trackDrafts(key: string, ds: Draft[]) {
    setDrafts((d) => ({ ...d, [key]: ds }));
    const upd = (jobId: string, p: Partial<Draft>) =>
      setDrafts((s) => ({ ...s, [key]: (s[key] || []).map((x) => x.jobId === jobId ? { ...x, ...p } : x) }));
    ds.forEach((d) => waitMedia(d.jobId, (pct) => upd(d.jobId, { pct }))
      .then((u) => upd(d.jobId, { url: u }))
      .catch(() => upd(d.jobId, { err: true })));
  }
  async function hunt(key: string, gen: (seed: number, i: number) => Promise<unknown>) {
    setHunting(key);
    const base = Math.floor(Math.random() * 2_000_000_000);
    try {
      const ds: Draft[] = [];
      for (let i = 0; i < 4; i++) {
        const r = await gen(base + i, i) as { job_id: string };
        ds.push({ jobId: r.job_id, seed: base + i });
      }
      trackDrafts(key, ds);
    } catch (e) {
      ctx.setResults([{ id: rid(), title: "Generation failed", status: "error", pct: 0, err: (e as Error).message }]);
    } finally { setHunting(""); }
  }

  // ---- Step 2: canonical identity sheet (server-side verified Sheet-D recipe)
  async function genSheet(c: Character) {
    const key = `${c.id}:sheet`;
    setHunting(key);
    try {
      const r = await api.characterSheet(c.id, { drafts: 4 }) as { drafts: { job_id: string; seed: number }[] };
      trackDrafts(key, r.drafts.map((d) => ({ jobId: d.job_id, seed: d.seed })));
    } catch (e) {
      ctx.setResults([{ id: rid(), title: "Sheet generation failed", status: "error", pct: 0, err: (e as Error).message }]);
    } finally { setHunting(""); }
  }

  // ---- Step 3: costumes (server-side three-view garment sheet; person-free, zero identity risk)
  async function genCostume(c: Character) {
    const nc = newCostume[c.id];
    if (!nc?.desc?.trim()) {
      ctx.setResults([{ id: rid(), title: "Describe the outfit first", status: "error", pct: 0, err: "Write what the garment is - fabric, cut, neckline, sleeves, length - then generate its sheet." }]);
      return;
    }
    const key = `${c.id}:costume:new`;
    setHunting(key);
    try {
      const r = await api.characterCostume(c.id, { name: nc.name || "New outfit", desc: nc.desc, drafts: 2 }) as
        { costume: Costume; drafts: { job_id: string; seed: number }[] };
      await reload();
      setNewCostume((s) => ({ ...s, [c.id]: { name: "", desc: "" } }));
      trackDrafts(`${c.id}:costume:${r.costume.id}`, r.drafts.map((d) => ({ jobId: d.job_id, seed: d.seed })));
    } catch (e) {
      ctx.setResults([{ id: rid(), title: "Costume generation failed", status: "error", pct: 0, err: (e as Error).message }]);
    } finally { setHunting(""); }
  }
  async function rerollCostume(c: Character, co: Costume) {
    const key = `${c.id}:costume:${co.id}`;
    setHunting(key);
    try {
      const r = await api.characterCostume(c.id, { name: co.name, desc: co.desc, drafts: 2 }) as
        { costume: Costume; drafts: { job_id: string; seed: number }[] };
      // the endpoint appends a NEW entry; fold its drafts onto the existing costume and drop the dupe
      await api.characterSave({ ...c, costumes: (c.costumes || []).filter((x) => x.id !== r.costume.id) });
      await reload();
      trackDrafts(key, r.drafts.map((d) => ({ jobId: d.job_id, seed: d.seed })));
    } catch (e) {
      ctx.setResults([{ id: rid(), title: "Costume reroll failed", status: "error", pct: 0, err: (e as Error).message }]);
    } finally { setHunting(""); }
  }

  // expand a short description via the LLM
  async function enhance(c: Character) {
    const appearance = (c.appearance || "").trim();
    if (!appearance) { ctx.setResults([{ id: rid(), title: "Type a description first", status: "error", pct: 0, err: "Fill the guided identity fields (or the description), then Enhance expands them into a full identity block." }]); return; }
    setEnhancing(c.id);
    try {
      const r = await api.llm({ provider: llmProvider, model: "", task: "char_desc", input: appearance }) as { text?: string };
      if (r.text) patch(c, { appearance: r.text.trim() });
    } catch (e) { ctx.setResults([{ id: rid(), title: "Enhance failed", status: "error", pct: 0, err: (e as Error).message }]); }
    finally { setEnhancing(""); }
  }

  // ---- legacy MSR generation (old pipeline, kept intact below in a collapsed section)
  function genRef(c: Character, w: Wardrobe, slot: "face" | "body") {
    const face = c.identity?.faceRefId, body = c.identity?.bodyRefId;
    const bodyFace = w.faceRefId || face;
    const refs = (slot === "face"
      ? [face || body]
      : [body || bodyFace, ...(body && bodyFace ? [bodyFace] : [])]).filter(Boolean) as string[];
    if (!refs.length) { ctx.setResults([{ id: rid(), title: "Set an identity reference first", status: "error", pct: 0, err: "Pick a face/body still for the identity core, then generate the dressed look from it." }]); return; }
    const framing = slot === "face"
      ? "head-and-shoulders close-up portrait"
      : "Full-length full-body photograph of the person in image 1, standing upright and facing the camera, the entire body visible from head to toe including the legs and feet, the full figure centred in frame at a distance";
    const base = `${framing}, wearing ${w.outfitPrompt || "the same outfit"}, neutral studio background, ${PHOTO_POS}`;
    const neg = slot === "body"
      ? `${PHOTO_NEG}, close-up, portrait, headshot, head and shoulders, bust shot, cropped at the chest, face only, zoomed in`
      : PHOTO_NEG;
    const vy = !!vary[c.id], vset = slot === "face" ? FACE_VARY : BODY_VARY;
    hunt(`${c.id}:w${w.id}:${slot}`, (seed, i) =>
      api.videoCharStill({ ref_ids: refs, prompt: vy ? `${base}, ${vset[i % 4]}` : base, negative: neg, seed, width: CHAR_REF_W, height: CHAR_REF_H }));
  }
  function genIdentity(c: Character, slot: "face" | "body") {
    const appearance = (c.appearance || "").trim();
    if (!appearance) { ctx.setResults([{ id: rid(), title: "Describe the character first", status: "error", pct: 0, err: "Write an appearance description first." }]); return; }
    const faceRef = c.identity?.faceRefId;
    const useFace = slot === "body" && !!faceRef && matchFace[c.id] !== false;
    const bodyFraming = useFace
      ? "Full-length full-body photograph of the person in image 1, standing upright and facing the camera, the entire body visible from head to toe including the legs and feet, the full figure centred in frame at a distance, neutral mid-grey studio backdrop"
      : "full-length full-body photograph, standing upright and facing the camera, the entire body visible from head to toe including the legs and feet, the full figure centred in frame at a distance, neutral mid-grey studio backdrop";
    const framing = slot === "face"
      ? "head and shoulders close-up portrait, facing camera, neutral mid-grey studio backdrop, 85mm"
      : bodyFraming;
    const base = slot === "body" ? `${framing}. ${appearance}, ${PHOTO_POS}` : `${appearance}, ${framing}, ${PHOTO_POS}`;
    const neg = slot === "body"
      ? `${PHOTO_NEG}, close-up, portrait, headshot, head and shoulders, bust shot, cropped at the chest, face only, zoomed in`
      : PHOTO_NEG;
    const vy = !!vary[c.id], vset = slot === "face" ? FACE_VARY : BODY_VARY;
    hunt(`${c.id}:${slot}`, (seed, i) => {
      const prompt = vy ? `${base}, ${vset[i % 4]}` : base;
      return useFace
        ? api.videoCharStill({ ref_ids: [faceRef], prompt, negative: neg, seed, width: CHAR_REF_W, height: CHAR_REF_H })
        : api.videoStill({ prompt, negative: neg, seed, width: CHAR_REF_W, height: CHAR_REF_H });
    });
  }
  function pickIdentity(c: Character, slot: "face" | "body", jobId: string) {
    setIdentity(c, slot === "face" ? { faceRefId: jobId } : { bodyRefId: jobId });
  }
  function pickWardrobe(c: Character, w: Wardrobe, slot: "face" | "body", jobId: string) {
    setWardrobe(c, w.id, slot === "face" ? { faceRefId: jobId } : { bodyRefId: jobId });
  }
  function closeDrafts(key: string) {
    setDrafts((d) => { const n = { ...d }; delete n[key]; return n; });
  }

  const body = (
    <>
      <p className="text-[10px] text-[var(--color-muted)]">
        A character is a locked <span className="text-[var(--color-ink)]">identity</span> - face and body, anchored by one
        canonical <span className="text-[var(--color-ink)]">identity sheet</span> that is never re-rendered - plus a wardrobe of
        <span className="text-[var(--color-ink)]"> costumes</span> you can add at any time. Costume sheets contain no person,
        so new outfits never risk the face.
      </p>
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wide text-[var(--color-muted)]">{chars.length} character{chars.length === 1 ? "" : "s"}</span>
        <button onClick={add} className="text-[10px] text-[var(--color-accent2)]">+ character</button>
      </div>
      {chars.map((c) => {
        const editing = editId === c.id;
        const identityDone = !!(c.appearance || "").trim();
        const sheetDone = !!c.sheetId;
        const costumes = c.costumes || [];
        const nc = newCostume[c.id] || { name: "", desc: "" };
        return (
          <div key={c.id} className="rounded border border-[var(--color-line)] bg-[var(--color-bg)] p-2 space-y-2">
            <div className="flex items-center gap-1.5">
              <button onClick={() => setEditId(editing ? "" : c.id)} className="text-[var(--color-muted)] hover:text-[var(--color-ink)]" title="expand">{editing ? "▾" : "▸"}</button>
              {c.sheetId
                ? <img src={`/api/media/${c.sheetId}`} onClick={() => openLightbox(`/api/media/${c.sheetId}`)} title="canonical identity sheet — click to enlarge"
                    className="h-7 w-12 shrink-0 cursor-zoom-in rounded object-cover" alt="" />
                : <span className="flex h-7 w-12 shrink-0 items-center justify-center rounded border border-dashed border-[var(--color-line)] text-[8px] text-[var(--color-muted)]" title="no identity sheet yet">no sheet</span>}
              <input className={inp} value={c.name} onChange={(e) => patch(c, { name: e.target.value })} placeholder="name" />
              <select className={`${inp} w-28`} value={c.kind || "musician"} onChange={(e) => patch(c, { kind: e.target.value })} title="kind">
                <option value="musician">musician</option>
                <option value="actor">actor</option>
              </select>
              <select className={`${inp} w-28`} value={c.gender || ""} onChange={(e) => patch(c, { gender: e.target.value })} title="gender">
                <option value="">gender?</option>
                <option value="female">female</option>
                <option value="male">male</option>
                <option value="non-binary">non-binary</option>
              </select>
              <span className="w-16 shrink-0 text-[9px] text-[var(--color-muted)]">{costumes.length} outfit{costumes.length === 1 ? "" : "s"}</span>
              <button onClick={() => del(c.id)} className="px-1 text-[var(--color-muted)] hover:text-red-400" title="delete character">{"×"}</button>
            </div>
            {editing && (
              <div className="space-y-2.5 border-t border-[var(--color-line)] pt-2">
                <div className="flex items-center gap-1.5">
                  <span className="w-16 shrink-0 text-[10px] font-semibold uppercase tracking-wide text-[var(--color-muted)]">Role</span>
                  <input className={inp} value={c.role || ""} onChange={(e) => patch(c, { role: e.target.value })} placeholder="e.g. lead singer, lead guitarist, bassist" />
                </div>

                {/* ---- STEP 1: identity ---- */}
                <div className="space-y-1.5">
                  <div className="flex items-center gap-1.5">
                    <StepTag n={1} done={identityDone} />
                    <span className="text-[10px] font-semibold uppercase tracking-wide text-[var(--color-muted)]">Identity - who they are</span>
                    <span className="flex-1" />
                    <button onClick={() => enhance(c)} disabled={enhancing === c.id} className="text-[10px] text-[var(--color-accent2)] disabled:opacity-50"
                      title="expand the fields into a full photoreal identity block with the LLM">
                      {enhancing === c.id ? "enhancing..." : "✨ enhance"}
                    </button>
                  </div>
                  <p className="text-[9px] text-[var(--color-muted)]">Face and body only - no clothing here (outfits are step 3). This text is the character's permanent identity block; it goes into every render.</p>
                  <div className="grid grid-cols-1 gap-1 sm:grid-cols-2">
                    {IDENTITY_FIELDS.map((f) => (
                      <div key={f.key} className="flex items-center gap-1.5">
                        <span className="w-24 shrink-0 truncate text-[9px] text-[var(--color-muted)]" title={f.hint}>{f.label}</span>
                        <input className={inp} value={(c.identity?.[f.key] as string) || ""} placeholder={f.hint}
                          onChange={(e) => setIdentity(c, { [f.key]: e.target.value } as Partial<Identity>)} />
                      </div>
                    ))}
                  </div>
                  <textarea className={inp} rows={2} value={c.appearance || ""} placeholder="the composed identity block (edit freely, or fill the guided fields above)"
                    onChange={(e) => patch(c, { appearance: e.target.value })} />
                </div>

                {/* ---- STEP 2: canonical identity sheet ---- */}
                <div className="space-y-1.5 border-t border-[var(--color-line)] pt-2">
                  <div className="flex items-center gap-1.5">
                    <StepTag n={2} done={sheetDone} />
                    <span className="text-[10px] font-semibold uppercase tracking-wide text-[var(--color-muted)]">Identity sheet - lock the face</span>
                  </div>
                  <p className="text-[9px] text-[var(--color-muted)]">
                    One multi-view sheet (front, portrait, profile) in neutral base wear. The one you pick is the character's face
                    <span className="text-[var(--color-ink)]"> forever</span> - outfits never re-render it. Generate candidates until one feels right.
                  </p>
                  {c.sheetId && (
                    <img src={`/api/media/${c.sheetId}`} onClick={() => openLightbox(`/api/media/${c.sheetId}`)}
                      className="w-full cursor-zoom-in rounded border border-[var(--color-accent2)]" alt="canonical identity sheet" title="canonical identity sheet — click to enlarge" />
                  )}
                  <button onClick={() => genSheet(c)} disabled={busy || !!hunting || !identityDone}
                    title={identityDone ? "generate 4 sheet candidates" : "fill the identity first (step 1)"}
                    className="w-full rounded border border-[var(--color-line)] py-1 text-[10px] text-[var(--color-muted)] hover:text-[var(--color-ink)] disabled:opacity-50">
                    {hunting === `${c.id}:sheet` ? "generating 4 sheets…" : c.sheetId ? "↻ generate 4 new candidates" : "generate 4 sheet candidates"}
                  </button>
                  <DraftStrip wide drafts={drafts[`${c.id}:sheet`] || []} picked={c.sheetId} busy={hunting === `${c.id}:sheet`}
                    onPick={(id) => patch(c, { sheetId: id, style: "h3" })} onReroll={() => genSheet(c)} onClose={() => closeDrafts(`${c.id}:sheet`)} />
                </div>

                {/* ---- STEP 3: costumes ---- */}
                <div className="space-y-1.5 border-t border-[var(--color-line)] pt-2">
                  <div className="flex items-center gap-1.5">
                    <StepTag n={3} done={costumes.some((co) => co.stillId)} />
                    <span className="text-[10px] font-semibold uppercase tracking-wide text-[var(--color-muted)]">Costumes - the wardrobe</span>
                  </div>
                  <p className="text-[9px] text-[var(--color-muted)]">
                    Each costume is a three-view garment sheet (front / back / side on a dress form) with no person in it -
                    add as many as you like, the face is never touched. Describe fabric, cut, neckline, sleeves and length.
                  </p>
                  {costumes.map((co) => (
                    <div key={co.id} className="rounded border border-[var(--color-line)] p-1.5 space-y-1.5">
                      <div className="flex items-center gap-1.5">
                        {co.stillId
                          ? <img src={`/api/media/${co.stillId}`} onClick={() => openLightbox(`/api/media/${co.stillId}`)} title="garment sheet — click to enlarge"
                              className="h-8 w-14 shrink-0 cursor-zoom-in rounded object-cover" alt="" />
                          : <span className="flex h-8 w-14 shrink-0 items-center justify-center rounded border border-dashed border-[var(--color-line)] text-[8px] text-[var(--color-muted)]">rendering…</span>}
                        <input className={inp} value={co.name} onChange={(e) => setCostume(c, co.id, { name: e.target.value })} placeholder="outfit name" />
                        <button onClick={() => rerollCostume(c, co)} disabled={busy || !!hunting} className="shrink-0 text-[9px] text-[var(--color-muted)] hover:text-[var(--color-ink)] disabled:opacity-50" title="render new candidates of this outfit">↻</button>
                        <button onClick={() => delCostume(c, co.id)} className="px-1 text-[var(--color-muted)] hover:text-red-400" title="delete outfit">{"×"}</button>
                      </div>
                      <textarea className={inp} rows={2} value={co.desc} placeholder={COSTUME_HINT}
                        onChange={(e) => setCostume(c, co.id, { desc: e.target.value })} />
                      <DraftStrip wide drafts={drafts[`${c.id}:costume:${co.id}`] || []} picked={co.stillId} busy={hunting === `${c.id}:costume:${co.id}`}
                        onPick={(id) => setCostume(c, co.id, { stillId: id })} onReroll={() => rerollCostume(c, co)} onClose={() => closeDrafts(`${c.id}:costume:${co.id}`)} />
                    </div>
                  ))}
                  <div className="rounded border border-dashed border-[var(--color-line)] p-1.5 space-y-1.5">
                    <div className="flex items-center gap-1.5">
                      <span className="text-[9px] text-[var(--color-muted)]">new outfit</span>
                      <input className={inp} value={nc.name} placeholder="name (e.g. crimson velvet)"
                        onChange={(e) => setNewCostume((s) => ({ ...s, [c.id]: { ...nc, name: e.target.value } }))} />
                    </div>
                    <textarea className={inp} rows={2} value={nc.desc} placeholder={COSTUME_HINT}
                      onChange={(e) => setNewCostume((s) => ({ ...s, [c.id]: { ...nc, desc: e.target.value } }))} />
                    <button onClick={() => genCostume(c)} disabled={busy || !!hunting || !sheetDone}
                      title={sheetDone ? "render the garment sheet (2 candidates)" : "lock the identity sheet first (step 2)"}
                      className="w-full rounded border border-[var(--color-line)] py-1 text-[10px] text-[var(--color-muted)] hover:text-[var(--color-ink)] disabled:opacity-50">
                      {hunting === `${c.id}:costume:new` ? "rendering garment sheet…" : "+ add outfit (renders 2 candidates)"}
                    </button>
                  </div>
                </div>

                {/* ---- legacy MSR references (old pipeline) ---- */}
                <details className="border-t border-[var(--color-line)] pt-2 text-[9px] text-[var(--color-muted)]">
                  <summary className="cursor-pointer hover:text-[var(--color-ink)]">Legacy MSR references (old LTX pipeline)</summary>
                  <div className="mt-2 space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] font-semibold uppercase tracking-wide text-[var(--color-muted)]">Reference stills</span>
                      <label className="flex items-center gap-1 text-[9px] text-[var(--color-muted)]">
                        <input type="checkbox" checked={!!vary[c.id]} onChange={(e) => setVary((v) => ({ ...v, [c.id]: e.target.checked }))} />
                        vary candidates
                      </label>
                    </div>
                    <div className="grid grid-cols-2 gap-1.5">
                      <div className="space-y-1">
                        <span className="text-[9px] text-[var(--color-muted)]">face (close-up)</span>
                        <StillPick value={c.identity?.faceRefId || ""} stills={stills} set={(id) => setIdentity(c, { faceRefId: id })} placeholder="- face still -" />
                        <button onClick={() => genIdentity(c, "face")} disabled={busy || !!hunting} className="w-full rounded border border-[var(--color-line)] py-0.5 text-[9px] text-[var(--color-muted)] hover:text-[var(--color-ink)] disabled:opacity-50">{hunting === `${c.id}:face` ? "generating 4…" : "generate 4"}</button>
                        <DraftStrip drafts={drafts[`${c.id}:face`] || []} picked={c.identity?.faceRefId} busy={hunting === `${c.id}:face`}
                          onPick={(id) => pickIdentity(c, "face", id)} onReroll={() => genIdentity(c, "face")} onClose={() => closeDrafts(`${c.id}:face`)} />
                      </div>
                      <div className="space-y-1">
                        <span className="text-[9px] text-[var(--color-muted)]">body (full-length)</span>
                        <StillPick value={c.identity?.bodyRefId || ""} stills={stills} set={(id) => setIdentity(c, { bodyRefId: id })} placeholder="- body still -" />
                        <button onClick={() => genIdentity(c, "body")} disabled={busy || !!hunting} className="w-full rounded border border-[var(--color-line)] py-0.5 text-[9px] text-[var(--color-muted)] hover:text-[var(--color-ink)] disabled:opacity-50">{hunting === `${c.id}:body` ? "generating 4…" : "generate 4"}</button>
                        {c.identity?.faceRefId && (
                          <label className="flex items-center gap-1 text-[9px] text-[var(--color-muted)]">
                            <input type="checkbox" checked={matchFace[c.id] !== false} onChange={(e) => setMatchFace((m) => ({ ...m, [c.id]: e.target.checked }))} />
                            match the chosen face
                          </label>
                        )}
                        <DraftStrip drafts={drafts[`${c.id}:body`] || []} picked={c.identity?.bodyRefId} busy={hunting === `${c.id}:body`}
                          onPick={(id) => pickIdentity(c, "body", id)} onReroll={() => genIdentity(c, "body")} onClose={() => closeDrafts(`${c.id}:body`)} />
                      </div>
                    </div>
                    <div className="space-y-1.5">
                      <div className="flex items-center justify-between">
                        <span className="text-[10px] font-semibold uppercase tracking-wide text-[var(--color-muted)]">Wardrobes (MSR ref pairs)</span>
                        <button onClick={() => addWardrobe(c)} className="text-[10px] text-[var(--color-accent2)]">+ wardrobe</button>
                      </div>
                      {(c.wardrobes || []).map((w) => (
                        <div key={w.id} className="rounded border border-[var(--color-line)] p-1.5 space-y-1.5">
                          <div className="flex items-center gap-1.5">
                            <input className={inp} value={w.name} onChange={(e) => setWardrobe(c, w.id, { name: e.target.value })} placeholder="look name" />
                            <button onClick={() => delWardrobe(c, w.id)} className="px-1 text-[var(--color-muted)] hover:text-red-400" title="delete look">{"×"}</button>
                          </div>
                          <textarea className={inp} rows={2} value={w.outfitPrompt || ""} placeholder="outfit description" onChange={(e) => setWardrobe(c, w.id, { outfitPrompt: e.target.value })} />
                          <div className="grid grid-cols-2 gap-1.5">
                            <div className="space-y-1">
                              <StillPick value={w.faceRefId || ""} stills={stills} set={(id) => setWardrobe(c, w.id, { faceRefId: id })} placeholder="- face ref -" />
                              <button onClick={() => genRef(c, w, "face")} disabled={busy || !!hunting} className="w-full rounded border border-[var(--color-line)] py-0.5 text-[9px] text-[var(--color-muted)] hover:text-[var(--color-ink)] disabled:opacity-50">{hunting === `${c.id}:w${w.id}:face` ? "generating 4…" : "generate 4"}</button>
                              <DraftStrip drafts={drafts[`${c.id}:w${w.id}:face`] || []} picked={w.faceRefId} busy={hunting === `${c.id}:w${w.id}:face`}
                                onPick={(id) => pickWardrobe(c, w, "face", id)} onReroll={() => genRef(c, w, "face")} onClose={() => closeDrafts(`${c.id}:w${w.id}:face`)} />
                            </div>
                            <div className="space-y-1">
                              <StillPick value={w.bodyRefId || ""} stills={stills} set={(id) => setWardrobe(c, w.id, { bodyRefId: id })} placeholder="- body ref -" />
                              <button onClick={() => genRef(c, w, "body")} disabled={busy || !!hunting} className="w-full rounded border border-[var(--color-line)] py-0.5 text-[9px] text-[var(--color-muted)] hover:text-[var(--color-ink)] disabled:opacity-50">{hunting === `${c.id}:w${w.id}:body` ? "generating 4…" : "generate 4"}</button>
                              <DraftStrip drafts={drafts[`${c.id}:w${w.id}:body`] || []} picked={w.bodyRefId} busy={hunting === `${c.id}:w${w.id}:body`}
                                onPick={(id) => pickWardrobe(c, w, "body", id)} onReroll={() => genRef(c, w, "body")} onClose={() => closeDrafts(`${c.id}:w${w.id}:body`)} />
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </details>
              </div>
            )}
          </div>
        );
      })}
    </>
  );

  if (collapsible)
    return <Collapse title={`Character library (${chars.length})`} open={open} onToggle={() => setOpen(!open)} accent>{body}</Collapse>;
  return <div className="space-y-2">{body}</div>;
}

// The standalone Characters tab: loads + manages the shared cast on its own.
export function CharactersForm({ busy, library, ...ctx }: { busy: boolean; library: LibItem[] } & RunCtx) {
  const [chars, setChars] = useState<Character[]>([]);
  const reload = () => api.characters().then((r) => setChars(r as Character[])).catch(() => {});
  useEffect(() => { reload(); }, []);
  const stills = library.filter((i) => i.mode === "videostill" && i.media_url);
  return (
    <div className="space-y-4">
      <p className="text-[11px] text-[var(--color-muted)]">
        Your reusable cast, shared across every project. Build a character in three steps: describe the
        <span className="text-[var(--color-ink)]"> identity</span> (guided fields), lock their
        <span className="text-[var(--color-ink)]"> identity sheet</span> (the face, chosen once), then add
        <span className="text-[var(--color-ink)]"> costumes</span> - person-free garment sheets you can create at any time.
      </p>
      <CharacterLibrary chars={chars} setChars={setChars} reload={reload} stills={stills} busy={busy} {...ctx} />
    </div>
  );
}
