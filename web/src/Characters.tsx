import { useEffect, useState } from "react";
import { api, type LibItem } from "./api";
import { inp, rid, type RunCtx } from "./ui";
import { Collapse, StillPick } from "./mvui";
import { openLightbox } from "./Lightbox";
import { type Character, type Identity, type Wardrobe } from "./mvmodel";

// Push the Z-Image Turbo default look away from the smooth, anime-ish "AI face" it falls back to.
const PHOTO_POS = "candid photograph, photorealistic, natural realistic skin with visible pores and texture, sharp focus, 50mm";
const PHOTO_NEG = "anime, cartoon, illustration, painting, drawing, 3d render, cgi, video game, doll, " +
  "plastic skin, waxy skin, airbrushed, overly smooth skin, beauty filter, overly symmetrical face, " +
  "low quality, blurry, deformed, bad anatomy, extra fingers, watermark, text";

// "vary" mode: push each of the 4 candidates toward a distinct look so they're real alternatives to
// choose from (Z-Image Turbo barely varies across seeds alone). Indexed 0-3, one per tile.
const FACE_VARY = [
  "softer rounder face shape, fuller cheeks, gentler features",
  "angular face, sharp defined cheekbones and strong jawline",
  "narrow oval face, slim refined features, straight nose",
  "broad strong face, wider-set eyes, heavier brow",
];
const BODY_VARY = [
  "slender lean build", "athletic toned build", "solid stocky build", "tall willowy build",
];

// one generated draft (a still job we're waiting on, then can pick)
type Draft = { jobId: string; seed: number; url?: string; err?: boolean; pct?: number };

// poll a still job until it has a media URL (or errors), reporting progress along the way
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

// 4-up draft strip: 2x2 candidates. Click an image to ENLARGE it (lightbox); use the
// explicit "use this" button to lock one in. Picking does NOT clear the strip, so you can
// keep comparing, change your mind, reroll, or close it yourself.
function DraftStrip({ drafts, picked, onPick, onReroll, onClose, busy }:
  { drafts: Draft[]; picked?: string; onPick: (jobId: string) => void; onReroll: () => void; onClose: () => void; busy: boolean }) {
  if (!drafts.length) return null;
  return (
    <div className="space-y-1 rounded border border-[var(--color-line)] bg-[var(--color-bg)] p-1.5">
      <div className="flex items-center justify-between px-0.5">
        <span className="text-[9px] text-[var(--color-muted)]">click to enlarge · "use this" to keep</span>
        <button onClick={onClose} className="text-[10px] leading-none text-[var(--color-muted)] hover:text-red-400" title="close candidates">×</button>
      </div>
      <div className="grid grid-cols-2 gap-1.5">
        {drafts.map((d) => (
          <div key={d.jobId}
            className={`overflow-hidden rounded border ${picked === d.jobId ? "border-[var(--color-accent2)] ring-1 ring-[var(--color-accent2)]" : "border-[var(--color-line)]"}`}>
            <div className="relative aspect-square">
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
        {busy ? "rerolling…" : "↻ reroll 4"}
      </button>
    </div>
  );
}

// The reusable character library editor. `collapsible` embeds it as a Collapse (MV Studio's
// quick panel); otherwise it renders inline as a full page (the Characters tab). Both edit the
// SAME shared global /api/characters store - one source of truth, no divergence.
export function CharacterLibrary({ chars, setChars, reload, stills, busy, collapsible = false, ...ctx }:
  { chars: Character[]; setChars: (c: Character[]) => void; reload: () => void; stills: LibItem[];
    busy: boolean; collapsible?: boolean } & RunCtx) {
  const [open, setOpen] = useState(false);
  const [editId, setEditId] = useState("");
  const [enhancing, setEnhancing] = useState("");
  const [drafts, setDrafts] = useState<Record<string, Draft[]>>({});   // key `${id}:${slot}` -> 4 candidates
  const [hunting, setHunting] = useState("");                           // key currently generating
  const [matchFace, setMatchFace] = useState<Record<string, boolean>>({});   // per-character: derive body from picked face
  const [vary, setVary] = useState<Record<string, boolean>>({});             // per-character: perturb each candidate for variety
  const [llmProvider, setLlmProvider] = useState("ollama");   // prefer subscription -> API key -> local
  useEffect(() => { api.llmProviders().then((p) => {
    const q = p as { claude?: boolean; claude_sub?: boolean };
    setLlmProvider(q.claude_sub ? "claude_sub" : q.claude ? "claude" : "ollama");
  }).catch(() => {}); }, []);

  function persist(c: Character) { api.characterSave(c).catch(() => {}); }
  function patch(c: Character, p: Partial<Character>) {
    const next = { ...c, ...p };
    setChars(chars.map((x) => x.id === c.id ? next : x));     // optimistic (keeps input focus)
    persist(next);
  }
  async function add() {
    const r = await api.characterSave({ name: "New character", kind: "musician", method: "auto" }) as Character;
    await reload(); setEditId(r.id);
  }
  async function del(id: string) { await api.characterDelete(id); await reload(); if (editId === id) setEditId(""); }

  const setIdentity = (c: Character, p: Partial<Identity>) => patch(c, { identity: { ...(c.identity || {}), ...p } });
  const setWardrobe = (c: Character, wid: string, p: Partial<Wardrobe>) =>
    patch(c, { wardrobes: (c.wardrobes || []).map((w) => w.id === wid ? { ...w, ...p } : w) });
  const addWardrobe = (c: Character) =>
    patch(c, { wardrobes: [...(c.wardrobes || []), { id: rid(), name: "New look", outfitPrompt: "" }] });
  const delWardrobe = (c: Character, wid: string) =>
    patch(c, { wardrobes: (c.wardrobes || []).filter((w) => w.id !== wid) });

  // Fan out 4 candidate stills for one key, render them into the draft strip, and let the user pick.
  // `gen(seed)` makes ONE still job; nothing is locked in until the user clicks a candidate.
  async function hunt(key: string, gen: (seed: number, i: number) => Promise<unknown>) {
    setHunting(key);
    const base = Math.floor(Math.random() * 2_000_000_000);
    try {
      const ds: Draft[] = [];
      for (let i = 0; i < 4; i++) {
        const r = await gen(base + i, i) as { job_id: string };
        ds.push({ jobId: r.job_id, seed: base + i });
      }
      setDrafts((d) => ({ ...d, [key]: ds }));
      const upd = (jobId: string, p: Partial<Draft>) =>
        setDrafts((s) => ({ ...s, [key]: (s[key] || []).map((x) => x.jobId === jobId ? { ...x, ...p } : x) }));
      ds.forEach((d) => waitMedia(d.jobId, (pct) => upd(d.jobId, { pct }))
        .then((u) => upd(d.jobId, { url: u }))
        .catch(() => upd(d.jobId, { err: true })));
    } catch (e) {
      ctx.setResults([{ id: rid(), title: "Generation failed", status: "error", pct: 0, err: (e as Error).message }]);
    } finally { setHunting(""); }
  }

  // generate a dressed reference (Qwen char_still) from the identity core + the wardrobe's outfit text.
  // Face shot: anchored on the identity FACE. Body shot: anchored on the identity BODY (build, primary
  // ref) AND the FACE (identity, secondary ref) together so the dressed body keeps both - Qwen char_still
  // takes up to 3 refs, image1 = primary.
  function genRef(c: Character, w: Wardrobe, slot: "face" | "body") {
    const face = c.identity?.faceRefId, body = c.identity?.bodyRefId;
    const refs = (slot === "face"
      ? [face || body]
      : [body || face, ...(body && face ? [face] : [])]).filter(Boolean) as string[];
    if (!refs.length) { ctx.setResults([{ id: rid(), title: "Set an identity reference first", status: "error", pct: 0, err: "Pick a face/body still for the identity core, then generate the dressed look from it." }]); return; }
    const framing = slot === "face" ? "head-and-shoulders close-up portrait" : "full-body shot from head to toe";
    const base = `${framing}, wearing ${w.outfitPrompt || "the same outfit"}, neutral studio background, ${PHOTO_POS}`;
    const vy = !!vary[c.id], vset = slot === "face" ? FACE_VARY : BODY_VARY;
    hunt(`${c.id}:w${w.id}:${slot}`, (seed, i) =>
      api.videoCharStill({ ref_ids: refs, prompt: vy ? `${base}, ${vset[i % 4]}` : base, negative: PHOTO_NEG, seed }));
  }

  // generate identity reference candidates. Face: Z-Image t2i from the appearance text. Body: when a
  // face is already picked and "match face" is on, derive the body FROM that face (Qwen char_still) so
  // the body matches the face; otherwise an independent t2i from the same appearance text.
  function genIdentity(c: Character, slot: "face" | "body") {
    const appearance = (c.appearance || "").trim();
    if (!appearance) { ctx.setResults([{ id: rid(), title: "Describe the character first", status: "error", pct: 0, err: "Write an appearance description (optionally Enhance it), then generate the still from it." }]); return; }
    const framing = slot === "face"
      ? "head and shoulders close-up portrait, facing camera, neutral mid-grey studio backdrop, 85mm"
      : "full body shot from head to toe, standing, neutral mid-grey studio backdrop";
    const faceRef = c.identity?.faceRefId;
    const useFace = slot === "body" && !!faceRef && matchFace[c.id] !== false;
    const base = `${appearance}, ${framing}, ${PHOTO_POS}`;
    const vy = !!vary[c.id], vset = slot === "face" ? FACE_VARY : BODY_VARY;
    hunt(`${c.id}:${slot}`, (seed, i) => {
      const prompt = vy ? `${base}, ${vset[i % 4]}` : base;
      return useFace
        ? api.videoCharStill({ ref_ids: [faceRef], prompt, negative: PHOTO_NEG, seed })
        : api.videoStill({ prompt, negative: PHOTO_NEG, seed });
    });
  }

  // user picked a candidate -> lock it into the slot, but KEEP the strip so they can keep
  // comparing / change their mind / reroll. The strip clears only on reroll or explicit close.
  function pickIdentity(c: Character, slot: "face" | "body", jobId: string) {
    setIdentity(c, slot === "face" ? { faceRefId: jobId } : { bodyRefId: jobId });
  }
  function pickWardrobe(c: Character, w: Wardrobe, slot: "face" | "body", jobId: string) {
    setWardrobe(c, w.id, slot === "face" ? { faceRefId: jobId } : { bodyRefId: jobId });
  }
  function closeDrafts(key: string) {
    setDrafts((d) => { const n = { ...d }; delete n[key]; return n; });
  }

  // expand the user's short appearance text into a full photoreal prompt via the LLM
  async function enhance(c: Character) {
    const appearance = (c.appearance || "").trim();
    if (!appearance) { ctx.setResults([{ id: rid(), title: "Type a description first", status: "error", pct: 0, err: "Write a few words about the character, then Enhance expands them into a full prompt." }]); return; }
    setEnhancing(c.id);
    try {
      const r = await api.llm({ provider: llmProvider, model: "", task: "char_desc", input: appearance }) as { text?: string };
      if (r.text) patch(c, { appearance: r.text.trim() });
    } catch (e) { ctx.setResults([{ id: rid(), title: "Enhance failed", status: "error", pct: 0, err: (e as Error).message }]); }
    finally { setEnhancing(""); }
  }

  const body = (
    <>
      <p className="text-[10px] text-[var(--color-muted)]">
        Reusable across every video. A character = a clothing-agnostic <span className="text-[var(--color-ink)]">identity core</span> (face + body)
        plus one or more <span className="text-[var(--color-ink)]">wardrobes</span> - a per-video outfit that generates a dressed face + body pair (the 2-image MSR unit).
      </p>
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wide text-[var(--color-muted)]">{chars.length} character{chars.length === 1 ? "" : "s"}</span>
        <button onClick={add} className="text-[10px] text-[var(--color-accent2)]">+ character</button>
      </div>
      {chars.map((c) => {
        const editing = editId === c.id;
        return (
          <div key={c.id} className="rounded border border-[var(--color-line)] bg-[var(--color-bg)] p-2 space-y-2">
            <div className="flex items-center gap-1.5">
              <button onClick={() => setEditId(editing ? "" : c.id)} className="text-[var(--color-muted)] hover:text-[var(--color-ink)]" title="expand">{editing ? "▾" : "▸"}</button>
              {(() => {
                const ref = c.identity?.faceRefId || c.identity?.bodyRefId;
                return ref
                  ? <img src={`/api/media/${ref}`} onClick={() => openLightbox(`/api/media/${ref}`)} title="identity reference — click to enlarge"
                      className="h-7 w-7 shrink-0 cursor-zoom-in rounded object-cover" alt="" />
                  : <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded border border-dashed border-[var(--color-line)] text-[8px] text-[var(--color-muted)]" title="no reference still yet">?</span>;
              })()}
              <input className={inp} value={c.name} onChange={(e) => patch(c, { name: e.target.value })} placeholder="name" />
              <select className={`${inp} w-28`} value={c.kind || "musician"} onChange={(e) => patch(c, { kind: e.target.value })} title="kind">
                <option value="musician">musician</option>
                <option value="actor">actor</option>
              </select>
              <select className={`${inp} w-28`} value={c.gender || ""} onChange={(e) => patch(c, { gender: e.target.value })} title="gender (used to name the character in band/scene composites)">
                <option value="">gender?</option>
                <option value="female">female</option>
                <option value="male">male</option>
                <option value="non-binary">non-binary</option>
              </select>
              <span className="w-16 shrink-0 text-[9px] text-[var(--color-muted)]">{(c.wardrobes || []).length} look{(c.wardrobes || []).length === 1 ? "" : "s"}</span>
              <button onClick={() => del(c.id)} className="px-1 text-[var(--color-muted)] hover:text-red-400" title="delete character">{"×"}</button>
            </div>
            {editing && (
              <div className="space-y-2 border-t border-[var(--color-line)] pt-2">
                {/* role (e.g. "lead singer", "lead guitarist", "bassist") - used to name the character in composites */}
                <div className="flex items-center gap-1.5">
                  <span className="w-16 shrink-0 text-[10px] font-semibold uppercase tracking-wide text-[var(--color-muted)]">Role</span>
                  <input className={inp} value={c.role || ""} onChange={(e) => patch(c, { role: e.target.value })} placeholder="e.g. lead singer, lead guitarist, bassist" />
                </div>
                {/* appearance description (+ LLM enhance + example) */}
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-semibold uppercase tracking-wide text-[var(--color-muted)]">Appearance</span>
                    <button onClick={() => enhance(c)} disabled={enhancing === c.id} className="text-[10px] text-[var(--color-accent2)] disabled:opacity-50"
                      title="expand what you typed into a full photoreal prompt with the LLM">
                      {enhancing === c.id ? "enhancing..." : "✨ enhance"}
                    </button>
                  </div>
                  <textarea className={inp} rows={3} value={c.appearance || ""} placeholder="describe the character's look: age, build, face shape, hair, eyes, skin, distinctive features"
                    onChange={(e) => patch(c, { appearance: e.target.value })} />
                  <details className="text-[9px] text-[var(--color-muted)]">
                    <summary className="cursor-pointer hover:text-[var(--color-ink)]">example of a good description</summary>
                    <p className="mt-1 italic text-[var(--color-muted)]">"woman in her late 20s, slender athletic build, oval face with sharp cheekbones, fair porcelain skin, long straight jet-black hair with a centre part, intense pale-green eyes, small silver nose stud, detailed skin texture"</p>
                    <p className="mt-1">Name concrete, visible traits (age, build, face shape, hair, eyes, skin, marks) - not mood or backstory. For a background/distant character you can include their outfit + instrument here and generate just ONE full-body still.</p>
                  </details>
                </div>
                {/* reference stills - generate from the appearance, or pick from the library */}
                <div className="space-y-1.5 border-t border-[var(--color-line)] pt-2">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-semibold uppercase tracking-wide text-[var(--color-muted)]">Reference stills</span>
                    <label className="flex items-center gap-1 text-[9px] text-[var(--color-muted)]" title="Push each of the 4 candidates toward a distinct look (face shape / build) so they're real alternatives to explore. Off = 4 close variations of the same described look.">
                      <input type="checkbox" checked={!!vary[c.id]} onChange={(e) => setVary((v) => ({ ...v, [c.id]: e.target.checked }))} />
                      vary candidates
                    </label>
                  </div>
                  <p className="text-[9px] text-[var(--color-muted)]">One still is enough for a background/distant character; a hero seen in close-ups wants both a face and a full body.</p>
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
                        <label className="flex items-center gap-1 text-[9px] text-[var(--color-muted)]" title="Generate the body FROM the chosen face so they match (Qwen reference). Off = an independent body from the appearance text.">
                          <input type="checkbox" checked={matchFace[c.id] !== false} onChange={(e) => setMatchFace((m) => ({ ...m, [c.id]: e.target.checked }))} />
                          match the chosen face
                        </label>
                      )}
                      <DraftStrip drafts={drafts[`${c.id}:body`] || []} picked={c.identity?.bodyRefId} busy={hunting === `${c.id}:body`}
                        onPick={(id) => pickIdentity(c, "body", id)} onReroll={() => genIdentity(c, "body")} onClose={() => closeDrafts(`${c.id}:body`)} />
                    </div>
                  </div>
                </div>
                {/* wardrobes */}
                <div className="space-y-1.5 border-t border-[var(--color-line)] pt-2">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-semibold uppercase tracking-wide text-[var(--color-muted)]">Wardrobes</span>
                    <button onClick={() => addWardrobe(c)} className="text-[10px] text-[var(--color-accent2)]">+ wardrobe</button>
                  </div>
                  {(c.wardrobes || []).length === 0 && <p className="text-[9px] text-[var(--color-muted)]">No wardrobes yet. Add one, describe the outfit, then generate the dressed face + body refs from the identity core.</p>}
                  {(c.wardrobes || []).map((w) => (
                    <div key={w.id} className="rounded border border-[var(--color-line)] p-1.5 space-y-1.5">
                      <div className="flex items-center gap-1.5">
                        <input className={inp} value={w.name} onChange={(e) => setWardrobe(c, w.id, { name: e.target.value })} placeholder="look name" />
                        <button onClick={() => delWardrobe(c, w.id)} className="px-1 text-[var(--color-muted)] hover:text-red-400" title="delete look">{"×"}</button>
                      </div>
                      <textarea className={inp} rows={2} value={w.outfitPrompt || ""} placeholder="outfit description (e.g. covered-shoulder feathered black lace gown)" onChange={(e) => setWardrobe(c, w.id, { outfitPrompt: e.target.value })} />
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
        Your reusable cast - shared across every project and the MV Studio timeline. Describe a character,
        optionally let the LLM enhance it, then generate their reference stills right here (or pick existing
        ones). One still is enough for a background/distant character; add wardrobes for a hero who changes
        outfits per song.
      </p>
      <CharacterLibrary chars={chars} setChars={setChars} reload={reload} stills={stills} busy={busy} {...ctx} />
    </div>
  );
}
