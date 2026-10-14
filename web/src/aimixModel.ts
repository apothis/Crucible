// Shared AI Mix model: mix-doc types, stem groups and helpers (docs/AI_MIX_PLAN.md).
export type EqMove = { type: string; hz: number; db: number; q?: number };
export type Band = { hz: number; db: number; q?: number };
export type Fx = {
  fx: string; by?: string; why?: string;
  hz?: number; moves?: EqMove[]; bands?: Band[];
  threshold_db?: number; ratio?: number; attack_ms?: number; release_ms?: number;
  attack_db?: number; sustain_db?: number; range_db?: number;
};
export type Stem = {
  role: string; group?: string; mute: boolean; target_lufs: number; suno_lufs?: number; trim_db: number;
  width: number; low_mono_hz: number; chain: Fx[]; sends: { verb?: number; delay?: number };
};
export type Bus = { room_size?: number; damping?: number; predelay_ms?: number; ms?: number; feedback?: number; hpf?: number; lpf?: number };
export type MixDoc = {
  title: string; preset?: string; preset_label?: string; notes?: string[]; stems: Record<string, Stem>;
  buses: { verb: Bus; delay: Bus };
  master?: { macro_match?: boolean; target_lufs?: number | null; [k: string]: unknown };
  [k: string]: unknown;
};

export const clone = <T,>(x: T): T => JSON.parse(JSON.stringify(x));

export const GROUPS: { id: string; label: string; color: string }[] = [
  { id: "vocals", label: "Vocals", color: "#f0911d" },
  { id: "drums", label: "Drums", color: "#e0512f" },
  { id: "bass", label: "Bass", color: "#a78bfa" },
  { id: "guitar", label: "Guitar", color: "#facc15" },
  { id: "piano", label: "Keys", color: "#2dd4bf" },
  { id: "other", label: "Other", color: "#60a5fa" },
];
export const groupOf = (s: Stem) => s.group || "other";
export const colorOf = (g: string) => GROUPS.find((x) => x.id === g)?.color || "#8a8fa0";
