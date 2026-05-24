async function jget(url: string) {
  const r = await fetch(url);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}
async function jpost(url: string, body: unknown) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.detail || r.statusText);
  return d;
}
async function jform(url: string, fd: FormData) {
  const r = await fetch(url, { method: "POST", body: fd });
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.detail || r.statusText);
  return d;
}

export const api = {
  config: () => jget("/api/config"),
  library: () => jget("/api/library"),
  deleteLib: async (id: string) => {
    const r = await fetch(`/api/library/${id}`, { method: "DELETE" });
    if (!r.ok) throw new Error("delete failed");
    return r.json();
  },
  setBucket: (id: string, bucket: string) => jpost(`/api/library/${id}/bucket`, { bucket }),
  job: (id: string) => jget(`/api/job/${id}`),
  generate: (p: unknown) => jpost("/api/generate", p),
  restyle: (fd: FormData) => jform("/api/restyle", fd),
  rvcVoices: () => jget("/api/rvc/voices"),
  rvcConvert: (fd: FormData) => jform("/api/rvc/convert", fd),
  voiceswap: (fd: FormData) => jform("/api/voiceswap", fd),
  stems: (fd: FormData) => jform("/api/stems/separate", fd),
  tonePresets: () => jget("/api/tone/presets"),
  tone: (fd: FormData) => jform("/api/tone/apply", fd),
  master: (fd: FormData) => jform("/api/master/apply", fd),
  stripGuitar: (fd: FormData) => jform("/api/backing/strip-guitar", fd),
  guitarRender: (fd: FormData) => jform("/api/guitar/render-amp", fd),
  helixCapture: (name: string) => jpost("/api/helix/capture", { name }),
  kontaktCapture: () => jpost("/api/guitar/kontakt/capture", {}),
  sources: () => jget("/api/sources"),
  mix: (p: unknown) => jpost("/api/mix", p),
  stitch: (p: unknown) => jpost("/api/stitch", p),
  voiceSearch: (q: string, sort: string) =>
    jget(`/api/voices/search?q=${encodeURIComponent(q)}&sort=${sort}`),
  voiceRepo: (id: string) => jget(`/api/voices/repo?id=${encodeURIComponent(id)}`),
  voiceInstall: (body: unknown) => jpost("/api/voices/install", body),
  llmProviders: () => jget("/api/llm/providers"),
  llm: (body: unknown) => jpost("/api/llm", body),
  songLyrics: (body: unknown) => jpost("/api/lyrics/song", body),
  vocalEngines: () => jget("/api/vocal/engines"),
  soulxVoices: () => jget("/api/vocal/soulx/voices"),
  soulxPrep: (fd: FormData) => jform("/api/vocal/soulx/prep", fd),
  importExtract: (fd: FormData) => jform("/api/import/extract", fd),
  importFetch: (url: string) => jpost("/api/import/fetch", { url }),
  archiveSearch: (q: string) => jget(`/api/archive/search?q=${encodeURIComponent(q)}`),
  archiveItem: (id: string) => jget(`/api/archive/item?id=${encodeURIComponent(id)}`),
  melodyCompose: (p: unknown) => jpost("/api/melody/compose", p),
  vocalBuild: (p: unknown) => jpost("/api/vocal/build", p),
  melodyMidi: async (p: unknown) => {
    const r = await fetch("/api/melody/midi", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(p),
    });
    if (!r.ok) throw new Error("midi export failed");
    return r.blob();
  },
  audioUrl: (id: string) => `/api/audio/${id}`,
  cancel: () => jpost("/api/cancel", {}),
};

export type Variant = { id: string; label: string; steps: number; cfg: number; available: boolean };
export type Genre = { id: string; label: string; tags: string; bpm: number; key: string; scale: string; lead: boolean; parent?: string | null };
export type Config = { comfy_host: string; variants: Variant[]; keys: string[]; rvc_driver: string; genres: Genre[] };
export type LibItem = { id: string; created: number; mode: string; params: Record<string, any>; audio_url: string; bucket?: string };
// Arrangement shared from the Song Constructor to the Vocal Builder.
export type SongDraft = { blocks: { type: string; seconds: number; lyrics: string }[]; key: string; bpm: number; tags: string };
export type Note = { midi: number; start: number; dur: number; syllable: string; section: number };
export type Section = { role: string; start: number; seconds: number; lyrics: string; notes: Note[] };
export type Score = { bpm: number; key: string; duration: number; provider: string; sections: Section[]; notes: Note[] };
export type VocalEngine = { id: string; label: string; desc: string; sings_words: boolean; available: boolean; host: string };
