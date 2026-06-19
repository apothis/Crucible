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
              <span className="w-16 shrink-0 text-[9px] text-[var(--color-muted)]">{(c.wardrobes || []).length} look{(c.wardrobes || []).length === 1 ? "" : "s"}</span>
              <button onClick={() => del(c.id)} className="px-1 text-[var(--color-muted)] hover:text-red-400" title="delete character">{"×"}</button>
            </div>
            {editing && (
              <div className="space-y-2 border-t border-[var(--color-line)] pt-2">
                {/* identity core */}
                <div className="space-y-1.5">
                  <span className="text-[10px] font-semibold uppercase tracking-wide text-[var(--color-muted)]">Identity core (clothing-agnostic)</span>
                  <div className="grid grid-cols-2 gap-1.5">
                    <label className="text-[9px] text-[var(--color-muted)]">face<StillPick value={c.identity?.faceRefId || ""} stills={stills} set={(id) => setIdentity(c, { faceRefId: id })} placeholder="- face still -" /></label>
                    <label className="text-[9px] text-[var(--color-muted)]">body<StillPick value={c.identity?.bodyRefId || ""} stills={stills} set={(id) => setIdentity(c, { bodyRefId: id })} placeholder="- body still -" /></label>
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
        Your reusable cast - shared across every project and the MV Studio timeline. Define each character's
        clothing-agnostic identity once, then add a wardrobe per song to dress them. Build face/body stills in
        the Video tab; pick them here as the identity core.
      </p>
      <CharacterLibrary chars={chars} setChars={setChars} reload={reload} stills={stills} busy={busy} {...ctx} />
    </div>
  );
}
