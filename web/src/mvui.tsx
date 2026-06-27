// Shared small UI primitives for the MV Studio + Characters tabs.
import { type ReactNode, useState } from "react";
import { inp } from "./ui";
import { type LibItem } from "./api";
import { openLightbox } from "./Lightbox";

export function Collapse({ title, open, onToggle, children, accent }: {
  title: ReactNode; open: boolean; onToggle: () => void; children: ReactNode; accent?: boolean;
}) {
  return (
    <div className="rounded-lg border border-[var(--color-line)] bg-[var(--color-panel2)]">
      <button onClick={onToggle}
        className={`flex w-full items-center justify-between px-3 py-2 text-xs font-semibold ${accent ? "text-[var(--color-accent2)]" : "text-[var(--color-ink)]"}`}>
        <span>{title}</span>
        <span className="text-[var(--color-muted)]">{open ? "−" : "+"}</span>
      </button>
      {open && <div className="space-y-2 border-t border-[var(--color-line)] p-3">{children}</div>}
    </div>
  );
}

export function Num({ label, value, set, step = 1, w = "w-20", title, min, max }: {
  label: string; value: number; set: (n: number) => void; step?: number; w?: string; title?: string;
  min?: number; max?: number;
}) {
  // Buffer keystrokes locally so the field can be cleared + retyped freely; clamp (min/max) only on
  // blur/Enter. Clamping on every keystroke made a min'd field un-clearable (empty -> snaps to min).
  const [draft, setDraft] = useState<string | null>(null);
  const commit = () => {
    if (draft === null) return;
    const n = Number(draft);
    if (draft.trim() === "" || Number.isNaN(n)) { setDraft(null); return; }   // revert to value
    let v = n;
    if (min !== undefined) v = Math.max(min, v);
    if (max !== undefined) v = Math.min(max, v);
    set(v); setDraft(null);
  };
  return (
    <label className="flex flex-col gap-0.5 text-[10px] text-[var(--color-muted)]" title={title}>
      {label}
      <input type="number" step={step} min={min} max={max} className={`${inp} ${w}`}
        value={draft ?? String(value)}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }} />
    </label>
  );
}

// a still picker with a small clickable thumbnail of the current choice
export function StillPick({ value, set, stills, placeholder = "— pick a still —", thumb = true }: {
  value: string; set: (id: string) => void; stills: LibItem[]; placeholder?: string; thumb?: boolean;
}) {
  const cur = stills.find((s) => s.id === value);
  // A freshly generated/picked still isn't in `stills` until the library refetches, so fall back to
  // its media endpoint by id - that way the picked ref is always visible immediately.
  const thumbUrl = cur?.media_url || (value ? `/api/media/${value}` : "");
  return (
    <div className="flex items-center gap-2">
      {thumb && thumbUrl && (
        <img src={thumbUrl} onClick={() => openLightbox(thumbUrl)}
          className="h-10 w-10 shrink-0 cursor-zoom-in rounded object-cover" alt="" />
      )}
      <select className={inp} value={value} onChange={(e) => set(e.target.value)}>
        <option value="">{placeholder}</option>
        {value && !cur && <option value={value}>{value.slice(0, 8)}… (generated)</option>}
        {stills.map((s) => (
          <option key={s.id} value={s.id}>
            {(s.params?.title || s.params?.prompt || s.id).toString().slice(0, 44)}
          </option>
        ))}
      </select>
    </div>
  );
}

export const stillLabel = (id: string, stills: LibItem[]) => {
  const s = stills.find((x) => x.id === id);
  return s ? (s.params?.title || s.params?.prompt || id).toString().slice(0, 18) : id.slice(0, 8);
};
