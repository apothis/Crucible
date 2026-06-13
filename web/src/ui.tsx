import type React from "react";
import { api } from "./api";

export const inp =
  "w-full rounded-lg border border-[var(--color-line)] bg-[var(--color-panel2)] px-3 py-2 text-sm text-[var(--color-ink)] outline-none focus:border-[var(--color-accent)]";

export function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-xs text-[var(--color-muted)]">
        {label} {hint && <span className="text-[#5a5f6e]">· {hint}</span>}
      </span>
      {children}
    </label>
  );
}

export function PrimaryButton({ onClick, disabled, children }: { onClick: () => void; disabled?: boolean; children: React.ReactNode }) {
  return (
    <button onClick={onClick} disabled={disabled}
      className="mt-2 w-full rounded-xl bg-gradient-to-r from-[var(--color-accent)] to-[var(--color-accent2)] py-3 font-semibold text-white shadow-lg transition hover:brightness-110 disabled:opacity-50">
      {children}
    </button>
  );
}

export function GhostButton({ onClick, children, className = "", disabled }: { onClick: () => void; children: React.ReactNode; className?: string; disabled?: boolean }) {
  return (
    <button onClick={onClick} disabled={disabled}
      className={`rounded-lg border border-[var(--color-line)] bg-[var(--color-panel2)] px-3 py-2 text-xs text-[var(--color-muted)] transition hover:text-[var(--color-ink)] disabled:opacity-50 ${className}`}>
      {children}
    </button>
  );
}

export function SectionTitle({ children }: { children: React.ReactNode }) {
  return <div className="border-t border-[var(--color-line)] pt-3 text-xs font-semibold text-[var(--color-accent2)]">{children}</div>;
}

export function Slider({ label, value, set, min, max, step }: { label: string; value: number; set: (n: number) => void; min: number; max: number; step: number }) {
  return (
    <Field label={`${label} · ${value.toFixed(2)}`}>
      <input type="range" className="w-full" min={min} max={max} step={step} value={value}
        onChange={(e) => set(parseFloat(e.target.value))} />
    </Field>
  );
}

export type Result = {
  id: string;
  title: string;
  status: "pending" | "running" | "done" | "error";
  pct: number;
  url?: string;
  media?: "image" | "video";   // when set, url is an image/video (not audio)
  err?: string;
};

export type RunCtx = {
  setResults: (r: Result[]) => void;
  patch: (id: string, p: Partial<Result>) => void;
  onDone: () => void;
};

let _n = 0;
export const rid = () => `r${Date.now()}_${_n++}`;

// poll a ComfyUI-style async job into a specific result card
export function pollJob(jobId: string, resultId: string, ctx: RunCtx) {
  const t = window.setInterval(async () => {
    const j = await api.job(jobId).catch(() => null);
    if (!j) return;
    if (j.status === "running" || j.status === "finalizing") {
      ctx.patch(resultId, { status: "running", pct: j.max ? Math.round((100 * j.progress) / j.max) : 10 });
    } else if (j.status === "done" && (j.audio_url || j.media_url)) {
      if (j.media_url) {
        ctx.patch(resultId, { status: "done", pct: 100, url: j.media_url + "?t=" + Date.now(),
                              media: j.kind === "image" ? "image" : "video" });
      } else {
        ctx.patch(resultId, { status: "done", pct: 100, url: j.audio_url + "?t=" + Date.now() });
      }
      clearInterval(t);
      ctx.onDone();
    } else if (j.status === "error") {
      ctx.patch(resultId, { status: "error", pct: 0, err: j.error || "error" });
      clearInterval(t);
    }
  }, 1000);
}

// like pollJob, but returns a promise resolving to the audio URL when done
// (used by the Song Constructor stitch mode to chain block generations).
export function waitJob(jobId: string, resultId: string, ctx: RunCtx): Promise<string> {
  return new Promise((resolve, reject) => {
    const t = window.setInterval(async () => {
      const j = await api.job(jobId).catch(() => null);
      if (!j) return;
      if (j.status === "running" || j.status === "finalizing") {
        ctx.patch(resultId, { status: "running", pct: j.max ? Math.round((100 * j.progress) / j.max) : 10 });
      } else if (j.status === "done" && j.audio_url) {
        ctx.patch(resultId, { status: "done", pct: 100, url: j.audio_url + "?t=" + Date.now() });
        clearInterval(t);
        ctx.onDone();
        resolve(j.audio_url);
      } else if (j.status === "error") {
        ctx.patch(resultId, { status: "error", pct: 0, err: j.error || "error" });
        clearInterval(t);
        reject(new Error(j.error || "generation error"));
      }
    }, 1000);
  });
}

// run a synchronous endpoint into a single result card
export async function runSync<T extends { audio_url?: string }>(
  title: string, fn: () => Promise<T>, ctx: RunCtx
): Promise<T | null> {
  const id = rid();
  ctx.setResults([{ id, title, status: "running", pct: 40 }]);
  try {
    const d = await fn();
    ctx.patch(id, { status: "done", pct: 100, url: d.audio_url ? d.audio_url + "?t=" + Date.now() : undefined });
    ctx.onDone();
    return d;
  } catch (e) {
    ctx.patch(id, { status: "error", pct: 0, err: (e as Error).message });
    return null;
  }
}
