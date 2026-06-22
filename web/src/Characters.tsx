import { useEffect, useState } from "react";
import { api, type LibItem } from "./api";
import { inp, rid, pollJob, type RunCtx } from "./ui";
import { Collapse, StillPick } from "./mvui";
import { type Character, type Identity, type Wardrobe } from "./mvmodel";

// The reusable character library editor. `collapsible` embeds it as a Collapse (MV Studio's
// quick panel); otherwise it renders inline as a full page (the Characters tab). Both edit the
// SAME shared global /api/characters store - one source of truth, no divergence.
export function CharacterLibrary({ chars, setChars, reload, stills, busy, collapsible = false, ...ctx }:
  { chars: Character[]; setChars: (c: Character[]) => void; reload: () => void; stills: LibItem[];
    busy: boolean; collapsible?: boolean } & RunCtx) {
  const [open, setOpen] = useState(false);
  const [editId, setEditId] = useState("");
  const [enhancing, setEnhancing] = useState("");
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

  // generate a dressed reference (Qwen char_still) from the identity core + the wardrobe's outfit text
  async function genRef(c: Character, w: Wardrobe, slot: "face" | "body") {
    const baseRef = slot === "face"
      ? (c.identity?.faceRefId || c.identity?.bodyRefId)
      : (c.identity?.bodyRefId || c.identity?.faceRefId);
    if (!baseRef) { ctx.setResults([{ id: rid(), title: "Set an identity reference first", status: "error", pct: 0, err: "Pick a face/body still for the identity core, then generate the dressed look from it." }]); return; }
    const framing = slot === "face" ? "head-and-shoulders close-up portrait" : "full-body shot from head to toe";
    const prompt = `${framing}, wearing ${w.outfitPrompt || "the same outfit"}, neutral studio background, photoreal, sharp focus`;
    const card = { id: rid(), title: `${c.name}: ${w.name} ${slot} ref`, status: "pending" as const, pct: 0 };
    ctx.setResults([card]);
    try {
      const { job_id } = await api.videoCharStill({ ref_ids: [baseRef], prompt }) as { job_id: string };
      setWardrobe(c, w.id, slot === "face" ? { faceRefId: job_id } : { bodyRefId: job_id });   // lock the produced still in
      ctx.patch(card.id, { status: "running", pct: 5 });
      pollJob(job_id, card.id, ctx);
    } catch (e) { ctx.patch(card.id, { status: "error", pct: 0, err: (e as Error).message }); }
  }

  // generate a reference still for the character from its appearance text (Z-Image t2i)
  async function genIdentity(c: Character, slot: "face" | "body") {
    const appearance = (c.appearance || "").trim();
    if (!appearance) { ctx.setResults([{ id: rid(), title: "Describe the character first", status: "error", pct: 0, err: "Write an appearance description (optionally Enhance it), then generate the still from it." }]); return; }
    const framing = slot === "face"
      ? "head and shoulders close-up portrait, facing camera, neutral mid-grey studio backdrop, 85mm"
      : "full body shot from head to toe, standing, neutral mid-grey studio backdrop";
    const card = { id: rid(), title: `${c.name}: ${slot} still`, status: "pending" as const, pct: 0 };
    ctx.setResults([card]);
    try {
      const { job_id } = await api.videoStill({ prompt: `${appearance}, ${framing}, photoreal, sharp focus` }) as { job_id: string };
      setIdentity(c, slot === "face" ? { faceRefId: job_id } : { bodyRefId: job_id });
      ctx.patch(card.id, { status: "running", pct: 5 });
      pollJob(job_id, card.id, ctx);
    } catch (e) { ctx.patch(card.id, { status: "error", pct: 0, err: (e as Error).message }); }
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
                  <span className="text-[10px] font-semibold uppercase tracking-wide text-[var(--color-muted)]">Reference stills</span>
                  <p className="text-[9px] text-[var(--color-muted)]">One still is enough for a background/distant character; a hero seen in close-ups wants both a face and a full body.</p>
                  <div className="grid grid-cols-2 gap-1.5">
                    <div className="space-y-1">
                      <span className="text-[9px] text-[var(--color-muted)]">face (close-up)</span>
                      <StillPick value={c.identity?.faceRefId || ""} stills={stills} set={(id) => setIdentity(c, { faceRefId: id })} placeholder="- face still -" />
                      <button onClick={() => genIdentity(c, "face")} disabled={busy} className="w-full rounded border border-[var(--color-line)] py-0.5 text-[9px] text-[var(--color-muted)] hover:text-[var(--color-ink)] disabled:opacity-50">generate face</button>
                    </div>
                    <div className="space-y-1">
                      <span className="text-[9px] text-[var(--color-muted)]">body (full-length)</span>
                      <StillPick value={c.identity?.bodyRefId || ""} stills={stills} set={(id) => setIdentity(c, { bodyRefId: id })} placeholder="- body still -" />
                      <button onClick={() => genIdentity(c, "body")} disabled={busy} className="w-full rounded border border-[var(--color-line)] py-0.5 text-[9px] text-[var(--color-muted)] hover:text-[var(--color-ink)] disabled:opacity-50">generate body</button>
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
                          <button onClick={() => genRef(c, w, "face")} disabled={busy} className="w-full rounded border border-[var(--color-line)] py-0.5 text-[9px] text-[var(--color-muted)] hover:text-[var(--color-ink)] disabled:opacity-50">generate face</button>
                        </div>
                        <div className="space-y-1">
                          <StillPick value={w.bodyRefId || ""} stills={stills} set={(id) => setWardrobe(c, w.id, { bodyRefId: id })} placeholder="- body ref -" />
                          <button onClick={() => genRef(c, w, "body")} disabled={busy} className="w-full rounded border border-[var(--color-line)] py-0.5 text-[9px] text-[var(--color-muted)] hover:text-[var(--color-ink)] disabled:opacity-50">generate body</button>
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
