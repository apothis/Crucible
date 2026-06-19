// Shared small UI primitives for the MV Studio + Characters tabs.
import { type ReactNode } from "react";
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

export function Num({ label, value, set, step = 1, w = "w-20", title }: {
  label: string; value: number; set: (n: number) => void; step?: number; w?: string; title?: string;
}) {
  return (
    <label className="flex flex-col gap-0.5 text-[10px] text-[var(--color-muted)]" title={title}>
      {label}
      <input type="number" step={step} className={`${inp} ${w}`} value={value}
        onChange={(e) => set(Number(e.target.value))} />
    </label>
  );
}

// a still picker with a small clickable thumbnail of the current choice
export function StillPick({ value, set, stills, placeholder = "— pick a still —", thumb = true }: {
  value: string; set: (id: string) => void; stills: LibItem[]; placeholder?: string; thumb?: boolean;
}) {
  const cur = stills.find((s) => s.id === value);
  return (
    <div className="flex items-center gap-2">
      {thumb && cur?.media_url && (
        <img src={cur.media_url} onClick={() => openLightbox(cur.media_url!)}
          className="h-10 w-10 shrink-0 cursor-zoom-in rounded object-cover" alt="" />
      )}
      <select className={inp} value={value} onChange={(e) => set(e.target.value)}>
        <option value="">{placeholder}</option>
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
