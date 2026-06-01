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
async function jmethod(method: string, url: string, body: unknown) {
  const r = await fetch(url, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.detail || r.statusText);
  return d;
}

export const api = {
  config: () => jget("/api/config"),
  acestepInfo: () => jget("/api/acestep/info"),
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
  cover: (fd: FormData) => jform("/api/cover", fd),
  transcribe: (fd: FormData) => jform("/api/transcribe", fd),
  repaint: (fd: FormData) => jform("/api/repaint", fd),
  beats: (id: string) => jget(`/api/beats/${id}`),
  beatsUpload: (fd: FormData) => jform("/api/beats", fd),
  layer: (fd: FormData) => jform("/api/layer", fd),
  layerIsolate: (fd: FormData) => jform("/api/layer/isolate", fd),
  rvcVoices: () => jget("/api/rvc/voices"),
  rvcConvert: (fd: FormData) => jform("/api/rvc/convert", fd),
  voiceswap: (fd: FormData) => jform("/api/voiceswap", fd),
  stems: (fd: FormData) => jform("/api/stems/separate", fd),
  tonePresets: () => jget("/api/tone/presets"),
  tone: (fd: FormData) => jform("/api/tone/apply", fd),
  master: (fd: FormData) => jform("/api/master/apply", fd),
  masterOptions: () => jget("/api/master/options"),
  analyzeReference: (fd: FormData) => jform("/api/reference/analyze", fd),
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
  importUpload: (fd: FormData) => jform("/api/import/upload", fd),
  importFetch: (url: string) => jpost("/api/import/fetch", { url }),
  archiveSearch: (q: string) => jget(`/api/archive/search?q=${encodeURIComponent(q)}`),
  archiveItem: (id: string) => jget(`/api/archive/item?id=${encodeURIComponent(id)}`),
  melodyCompose: (p: unknown) => jpost("/api/melody/compose", p),
  vocalBuild: (p: unknown) => jpost("/api/vocal/build", p),
  soloOptions: () => jget("/api/solo/options"),
  soloCompose: (p: unknown) => jpost("/api/solo/compose", p),
  soloRender: (p: unknown) => jpost("/api/solo/render", p),
  melodyMidi: async (p: unknown) => {
    const r = await fetch("/api/melody/midi", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(p),
    });
    if (!r.ok) throw new Error("midi export failed");
    return r.blob();
  },
  audioUrl: (id: string) => `/api/audio/${id}`,
  cancel: () => jpost("/api/cancel", {}),
  // Projects — saved bundles of per-page state
  projects: () => jget("/api/projects"),
  projectGet: (id: string) => jget(`/api/projects/${id}`),
  projectCreate: (body: { name: string; data: unknown }) => jpost("/api/projects", body),
  projectSave: (id: string, body: { name?: string; data: unknown }) => jmethod("PUT", `/api/projects/${id}`, body),
  projectRename: (id: string, name: string) => jmethod("PATCH", `/api/projects/${id}`, { name }),
  projectDelete: async (id: string) => {
    const r = await fetch(`/api/projects/${id}`, { method: "DELETE" });
    if (!r.ok) throw new Error("delete failed");
    return r.json();
  },

  // Metal LoRA training (drives the box engine pipeline)
  loraStatus: (dataset?: string) => jget(`/api/lora/status${dataset ? `?dataset=${encodeURIComponent(dataset)}` : ""}`),
  loraDatasetAdd: (fd: FormData) => jform("/api/lora/dataset/add", fd),
  loraScan: (b: unknown) => jpost("/api/lora/dataset/scan", b),
  loraSamples: (dataset?: string) => jget(`/api/lora/dataset/samples${dataset ? `?dataset=${encodeURIComponent(dataset)}` : ""}`),
  loraSamplePut: (idx: number, b: unknown) => jmethod("PUT", `/api/lora/dataset/sample/${idx}`, b),
  loraAutolabel: (b: unknown) => jpost("/api/lora/dataset/autolabel", b),
  loraAutolabelStatus: () => jget("/api/lora/dataset/autolabel/status"),
  loraAutolabelMerge: (b: unknown) => jpost("/api/lora/dataset/autolabel_merge", b),
  loraSave: (b: unknown) => jpost("/api/lora/dataset/save", b),
  loraPreprocess: (b: unknown) => jpost("/api/lora/dataset/preprocess", b),
  loraPreprocessStatus: () => jget("/api/lora/dataset/preprocess/status"),
  loraTrain: (b: unknown) => jpost("/api/lora/train", b),
  loraTrainStatus: (dataset?: string) => jget(`/api/lora/train/status${dataset ? `?dataset=${encodeURIComponent(dataset)}` : ""}`),
  loraTrainStop: () => jpost("/api/lora/train/stop", {}),
  loraExport: (b: unknown) => jpost("/api/lora/export", b),
  loraAdaptersAll: () => jget("/api/lora/adapters/all"),
  loraLoad: (b: unknown) => jpost("/api/lora/load", b),
  loraScale: (b: unknown) => jpost("/api/lora/scale", b),
  loraToggle: (b: unknown) => jpost("/api/lora/toggle", b),
  loraUnload: () => jpost("/api/lora/unload", {}),

  // Settings panel — curated app_config.json editor
  settings: () => jget("/api/settings"),
  settingsSave: (b: Record<string, unknown>) => jmethod("PUT", "/api/settings", b),
};

export type SettingsField = { group: string; key: string; type: string; label: string;
                              hint: string; value: string | boolean };
export type SettingsResponse = { fields: SettingsField[]; config_path: string };

export type LoraStatus = {
  train_available: boolean; upload_available: boolean;
  engine?: { data?: { loaded_model?: string; loaded_lm_model?: string } }; engine_error?: string;
  upload?: { ok?: boolean; base_dir?: string }; upload_error?: string;
  training?: { is_training?: boolean; status?: string; current_epoch?: number; current_step?: number;
               current_loss?: number | null; steps_per_second?: number; estimated_time_remaining?: number;
               loss_history?: number[]; training_log?: string; tensorboard_url?: string | null;
               start_time?: number | null; error?: string | null; config?: Record<string, any> };
  preprocess?: { task_id?: string | null; status?: string; progress?: string;
                 current?: number; total?: number };
  lora?: { lora_loaded?: boolean; use_lora?: boolean; lora_scale?: number; adapters?: string[] };
};
export type LoraTrack = { name: string; bpm: number; keyscale: string; has_lyrics: boolean;
                          lyrics_source: string; lyrics?: string; caption?: string;
                          caption_sources?: string[]; uploaded?: number };

export type Variant = { id: string; label: string; steps: number; cfg: number; available: boolean };
export type Genre = { id: string; label: string; tags: string; bpm: number; key: string; scale: string; lead: boolean; parent?: string | null };
export type Config = { comfy_host: string; variants: Variant[]; keys: string[]; rvc_driver: string; roformer?: boolean; acestep?: boolean; acestep_repaint?: boolean; acestep_lego?: boolean; analyze?: boolean; lora_train?: boolean; lora_upload?: boolean; genres: Genre[] };
export type LibItem = { id: string; created: number; mode: string; params: Record<string, any>; audio_url: string; bucket?: string };
// Arrangement shared from the Song Constructor to the Vocal Builder.
export type SongDraft = { blocks: { type: string; seconds: number; lyrics: string }[]; key: string; bpm: number; tags: string };
export type Note = { midi: number; start: number; dur: number; syllable: string; section: number };
export type Section = { role: string; start: number; seconds: number; lyrics: string; notes: Note[] };
export type Score = { bpm: number; key: string; duration: number; provider: string; sections: Section[]; notes: Note[] };
export type VocalEngine = { id: string; label: string; desc: string; sings_words: boolean; available: boolean; host: string };
export type Project = { id: string; name: string; created: number; updated: number; data?: ProjectData };
export type ProjectData = { drafts?: Record<string, Record<string, unknown>>; song?: SongDraft | null; mode?: string };

// ---- Track naming / versioning (shared by the Library card + every track picker) ----

// A track's user-given song name, or "" if it was never named.
export function songTitle(it: LibItem): string {
  const t = it.params?.title;
  return (t && String(t).trim()) || "";
}

// Where a titled track sits among its siblings (same mode + same name), 1-based,
// oldest→newest = v1..vN. null when the track is unnamed or the only one of its name.
export function versionInfo(it: LibItem, all: LibItem[]): { v: number; total: number } | null {
  const t = songTitle(it).toLowerCase();
  if (!t) return null;
  const sibs = all
    .filter((x) => x.mode === it.mode && songTitle(x).toLowerCase() === t)
    .sort((a, b) => a.created - b.created);
  if (sibs.length < 2) return null;
  const v = sibs.findIndex((x) => x.id === it.id) + 1;
  return { v, total: sibs.length };
}

// Short, friendly label for a track in a picker: the song name when present (else
// tags/source/etc.), plus a "v2"-style suffix when it's one of several versions.
export function trackLabel(it: LibItem, all: LibItem[], withMode = true): string {
  const p = it.params || {};
  const name = songTitle(it) || String(p.tags || p.source || p.preset || p.voice || it.id);
  const vi = versionInfo(it, all);
  return `${withMode ? it.mode + ": " : ""}${name.slice(0, 40)}${vi ? ` · v${vi.v}` : ""}`;
}
