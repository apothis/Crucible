import { useEffect, useState } from "react";
import { api, type Config, type LibItem } from "./api";
import { Field, inp, PrimaryButton, GhostButton, rid, pollJob, type RunCtx } from "./ui";
import { useDrafts } from "./drafts";

type Shot = {
  idx: number; section: string; start: number; end: number;
  type: "performance" | "narrative" | "broll";
  scene: string; motion: string; characters: string[]; lipsync: boolean;
  clipId?: string;   // library id of the rendered clip for this shot
};
type Character = { name: string; role: string; refStillId: string; loraName?: string };
type Project = { id: string; name: string };
type Method = "auto" | "anchor" | "qwen" | "vace" | "lora";

// poll a job to completion (media-aware: stills/clips use media_url, not audio_url)
function waitDone(jobId: string): Promise<void> {
  return new Promise((res, rej) => {
    const t = window.setInterval(async () => {
      const j = await api.job(jobId).catch(() => null);
      if (!j) return;
      if (j.status === "done") { clearInterval(t); res(); }
      else if (j.status === "error") { clearInterval(t); rej(new Error(j.error || "error")); }
    }, 1500);
  });
}

export function MusicVideoForm({ cfg, busy, library, ...ctx }: { cfg: Config; busy: boolean; library: LibItem[] } & RunCtx) {
  const d = useDrafts("musicvideo");
  const [projects, setProjects] = useState<Project[]>([]);
  const [projectId, setProjectId] = d.use("projectId", "");
  const [audioId, setAudioId] = d.use("audioId", "");
  const [cast, setCast] = d.use<Character[]>("cast", []);
  const [shots, setShots] = d.use<Shot[]>("shots", []);
  const [method, setMethod] = d.use<Method>("method", "auto");
  const [loras, setLoras] = useState<string[]>([]);
  const [gen, setGen] = useState(false);

  useEffect(() => { api.projects().then(setProjects).catch(() => {}); }, []);
  useEffect(() => { api.videoLoras().then((r) => setLoras(r as string[])).catch(() => {}); }, []);

  // resolve which consistency method to actually use for a character shot
  function resolveMethod(char: Character): Method {
    if (method !== "auto") return method;
    if (char.loraName) return "lora";
    if (cfg.video_qwen) return "qwen";
    return "anchor";
  }
  const stills = library.filter((i) => i.mode === "videostill" && i.media_url);
  const audios = library.filter((i) => i.audio_url);

  async function generateScript() {
    if (!projectId) return ctx.setResults([{ id: rid(), title: "Pick a song project first", status: "error", pct: 0, err: "Choose a project whose Song arrangement to base the video on." }]);
    setGen(true);
    try {
      const r = await api.mvScript({ project: projectId, cast: cast.map((c) => ({ name: c.name, role: c.role })) }) as { shots: Shot[] };
      setShots(r.shots || []);
    } catch (e) {
      ctx.setResults([{ id: rid(), title: "Script generation failed", status: "error", pct: 0, err: (e as Error).message }]);
    } finally { setGen(false); }
  }

  function patchShot(i: number, p: Partial<Shot>) { setShots(shots.map((s, j) => j === i ? { ...s, ...p } : s)); }
  function recordClip(idx: number, clipId: string) { setShots(shots.map((s, j) => j === idx ? { ...s, clipId } : s)); }
  function reindex(list: Shot[]): Shot[] { return list.map((s, i) => ({ ...s, idx: i })); }
  function addShot() {
    const last = shots[shots.length - 1];
    const start = last ? last.end : 0;
    setShots(reindex([...shots, { idx: shots.length, section: "", start, end: start + 5, type: "broll", scene: "", motion: "", characters: [], lipsync: false }]));
  }
  function deleteShot(i: number) { setShots(reindex(shots.filter((_, j) => j !== i))); }
  function moveShot(i: number, dir: -1 | 1) {
    const j = i + dir; if (j < 0 || j >= shots.length) return;
    const next = shots.slice(); [next[i], next[j]] = [next[j], next[i]]; setShots(reindex(next));
  }
  async function loadFromProject() {
    if (!projectId) return;
    try {
      const r = await api.projectVideoGet(projectId) as { video: { cast: Character[]; shots: Shot[]; audioId?: string; method?: Method } };
      const v = r.video;
      if (v.shots?.length) setShots(reindex(v.shots));
      if (v.cast?.length) setCast(v.cast);
      if (v.audioId) setAudioId(v.audioId);
      if (v.method) setMethod(v.method);
    } catch (e) { ctx.setResults([{ id: rid(), title: "Load failed", status: "error", pct: 0, err: (e as Error).message }]); }
  }
  async function saveToProject() {
    if (!projectId) { ctx.setResults([{ id: rid(), title: "Pick a project first", status: "error", pct: 0, err: "Select a song project to save the script into." }]); return; }
    try {
      await api.projectVideoSave(projectId, { cast, shots, audioId, method });
      ctx.setResults([{ id: rid(), title: "Saved to project", status: "done", pct: 100 }]);
    } catch (e) { ctx.setResults([{ id: rid(), title: "Save failed", status: "error", pct: 0, err: (e as Error).message }]); }
  }

  async function assemble() {
    const ready = shots.filter((s) => s.clipId);
    if (!ready.length) { ctx.setResults([{ id: rid(), title: "Nothing to assemble", status: "error", pct: 0, err: "Generate some shots first - each rendered clip is stitched in order." }]); return; }
    const card = { id: rid(), title: `assembling ${ready.length} shots...`, status: "running" as const, pct: 30 };
    ctx.setResults([card]);
    try {
      const r = await api.mvAssemble({ shots: ready.map((s) => ({ clip_id: s.clipId, start: s.start, end: s.end })), audio_id: audioId, title: projects.find((p) => p.id === projectId)?.name || "music video" }) as { media_url: string };
      ctx.patch(card.id, { status: "done", pct: 100, url: r.media_url + "?t=" + Date.now(), media: "video" });
      ctx.onDone();
    } catch (e) { ctx.patch(card.id, { status: "error", pct: 0, err: (e as Error).message }); }
  }

  // generate one shot: lip-sync (if a named singer w/ ref still + audio) -> S2V; character
  // shot -> i2v anchored on the character's ref still; scenic -> fresh still then i2v.
  async function genShot(shot: Shot) {
    const char = cast.find((c) => shot.characters.includes(c.name) && c.refStillId);
    const card = { id: rid(), title: `shot ${shot.idx + 1} (${shot.type})`, status: "pending" as const, pct: 0 };
    ctx.setResults([card]);
    try {
      if (shot.lipsync && char && audioId) {
        const { job_id } = await api.videoLipsync({ still_id: char.refStillId, audio_id: audioId, prompt: shot.scene, audio_start: shot.start }) as { job_id: string };
        ctx.patch(card.id, { status: "running", pct: 5 }); recordClip(shot.idx, job_id); pollJob(job_id, card.id, ctx);
      } else if (char) {
        const m = resolveMethod(char);
        if (m === "vace" && cfg.video_vace) {
          // reference-to-video: animate the character directly (identity through motion)
          ctx.patch(card.id, { status: "running", pct: 5, title: `shot ${shot.idx + 1}: VACE ${char.name}...` });
          const { job_id } = await api.videoVace({ still_id: char.refStillId, prompt: `${shot.scene}. ${shot.motion}` }) as { job_id: string };
          recordClip(shot.idx, job_id); pollJob(job_id, card.id, ctx);
        } else if (m === "lora" && char.loraName) {
          // trained character LoRA -> consistent still in the new scene, then animate
          ctx.patch(card.id, { status: "running", pct: 3, title: `shot ${shot.idx + 1}: ${char.name} (LoRA)...` });
          const st = await api.videoStill({ prompt: shot.scene, lora: char.loraName }) as { job_id: string };
          await waitDone(st.job_id);
          ctx.patch(card.id, { pct: 50, title: `shot ${shot.idx + 1}: animating...` });
          const { job_id } = await api.videoI2V({ still_id: st.job_id, prompt: shot.motion || "subtle cinematic motion" }) as { job_id: string };
          recordClip(shot.idx, job_id); pollJob(job_id, card.id, ctx);
        } else if (m === "qwen" && cfg.video_qwen) {
          // place the character into THIS scene (Qwen-Edit, keeps identity), then animate
          ctx.patch(card.id, { status: "running", pct: 3, title: `shot ${shot.idx + 1}: placing ${char.name}...` });
          const cs = await api.videoCharStill({ ref_ids: [char.refStillId], prompt: shot.scene }) as { job_id: string };
          await waitDone(cs.job_id);
          ctx.patch(card.id, { pct: 50, title: `shot ${shot.idx + 1}: animating...` });
          const { job_id } = await api.videoI2V({ still_id: cs.job_id, prompt: shot.motion || "subtle cinematic motion" }) as { job_id: string };
          recordClip(shot.idx, job_id); pollJob(job_id, card.id, ctx);
        } else {
          // anchor: reuse the reference still as-is
          const { job_id } = await api.videoI2V({ still_id: char.refStillId, prompt: shot.motion || shot.scene }) as { job_id: string };
          ctx.patch(card.id, { status: "running", pct: 5 }); recordClip(shot.idx, job_id); pollJob(job_id, card.id, ctx);
        }
      } else {
        ctx.patch(card.id, { status: "running", pct: 3, title: `shot ${shot.idx + 1}: still...` });
        const still = await api.videoStill({ prompt: shot.scene }) as { job_id: string };
        await waitDone(still.job_id);
        ctx.patch(card.id, { pct: 50, title: `shot ${shot.idx + 1}: animating...` });
        const { job_id } = await api.videoI2V({ still_id: still.job_id, prompt: shot.motion || "subtle cinematic motion" }) as { job_id: string };
        recordClip(shot.idx, job_id); pollJob(job_id, card.id, ctx);
      }
    } catch (e) { ctx.patch(card.id, { status: "error", pct: 0, err: (e as Error).message }); }
  }

  if (!cfg.video) {
    return <div className="rounded-xl border border-[var(--color-line)] bg-[var(--color-panel2)] p-4 text-xs text-[var(--color-muted)]">
      Video models not detected on the box. Run the video model downloads + restart the backend.
    </div>;
  }

  return (
    <div className="space-y-4">
      <p className="text-[11px] text-[var(--color-muted)]">Turn a song into a beat-cut music video: generate a shot list, define the cast, then render each shot.</p>
      <Field label="Character consistency" hint="how to keep a character on-model across shots">
        <select className={inp} value={method} onChange={(e) => setMethod(e.target.value as Method)}>
          <option value="auto">Auto (LoRA if set, else Qwen, else anchor)</option>
          <option value="anchor">Anchor still (reuse the reference)</option>
          <option value="qwen" disabled={!cfg.video_qwen}>Qwen-Edit{cfg.video_qwen ? "" : " - not downloaded"}</option>
          <option value="vace" disabled={!cfg.video_vace}>Wan VACE reference-to-video{cfg.video_vace ? "" : " - not downloaded"}</option>
          <option value="lora">Character LoRA (per-character, trained)</option>
        </select>
      </Field>

      <div className="grid grid-cols-2 gap-2">
        <Field label="Song project" hint="its arrangement drives the script">
          <select className={inp} value={projectId} onChange={(e) => setProjectId(e.target.value)}>
            <option value="">- pick a project -</option>
            {projects.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
          </select>
        </Field>
        <Field label="Song audio" hint="track for lip-sync shots">
          <select className={inp} value={audioId} onChange={(e) => setAudioId(e.target.value)}>
            <option value="">- pick a track -</option>
            {audios.map((a) => <option key={a.id} value={a.id}>{(a.params?.title || a.params?.tags || a.mode || a.id).toString().slice(0, 40)}</option>)}
          </select>
        </Field>
      </div>

      {/* Cast */}
      <div className="rounded-lg border border-[var(--color-line)] bg-[var(--color-panel2)] p-3 space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-[var(--color-accent2)]">Cast</span>
          <GhostButton onClick={() => setCast([...cast, { name: "", role: "lead singer", refStillId: "" }])}>+ character</GhostButton>
        </div>
        {cast.length === 0 && <p className="text-[11px] text-[var(--color-muted)]">No fixed cast - the video will lean scenic. Add a character (e.g. the singer) and pick a reference still to keep them consistent.</p>}
        {cast.map((c, i) => (
          <div key={i} className="grid grid-cols-[1fr_1fr_1.3fr_1.3fr_auto] items-center gap-1.5">
            <input className={inp} placeholder="name" value={c.name} onChange={(e) => setCast(cast.map((x, j) => j === i ? { ...x, name: e.target.value } : x))} />
            <input className={inp} placeholder="role" value={c.role} onChange={(e) => setCast(cast.map((x, j) => j === i ? { ...x, role: e.target.value } : x))} />
            <select className={inp} value={c.refStillId} onChange={(e) => setCast(cast.map((x, j) => j === i ? { ...x, refStillId: e.target.value } : x))}>
              <option value="">- reference still -</option>
              {stills.map((s) => <option key={s.id} value={s.id}>{(s.params?.prompt || s.id).toString().slice(0, 36)}</option>)}
            </select>
            <select className={inp} value={c.loraName || ""} onChange={(e) => setCast(cast.map((x, j) => j === i ? { ...x, loraName: e.target.value } : x))} title="trained character LoRA (optional)">
              <option value="">- LoRA (optional) -</option>
              {loras.map((l) => <option key={l} value={l}>{l.replace(/\.safetensors$/, "").slice(0, 28)}</option>)}
            </select>
            <button onClick={() => setCast(cast.filter((_, j) => j !== i))} className="text-[var(--color-muted)] hover:text-red-400 px-1" title="Remove">x</button>
          </div>
        ))}
        <p className="text-[10px] text-[var(--color-muted)]">Make reference stills in the Video tab (Still), then pick them here.</p>
      </div>

      <PrimaryButton onClick={generateScript} disabled={gen || !projectId}>{gen ? "Writing script..." : shots.length ? "Regenerate script" : "Generate script"}</PrimaryButton>

      {/* Script toolbar */}
      <div className="flex flex-wrap items-center gap-2">
        <GhostButton onClick={loadFromProject}>Load saved</GhostButton>
        <GhostButton onClick={saveToProject}>Save to project</GhostButton>
        <GhostButton onClick={addShot}>+ Add shot</GhostButton>
        {shots.length > 0 && <span className="text-[10px] text-[var(--color-muted)]">{shots.length} shots</span>}
      </div>

      {/* Shot list (fully editable) */}
      {shots.length > 0 && (
        <div className="space-y-2">
          {shots.map((s, i) => (
            <div key={i} className="rounded-lg border border-[var(--color-line)] bg-[var(--color-panel2)] p-2.5 space-y-1.5">
              <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
                <span className="rounded bg-[#2a1c19] px-1.5 py-0.5 text-[var(--color-accent2)]">{i + 1}</span>
                <input type="number" className={`${inp} w-14`} value={s.start} title="start (s)" onChange={(e) => patchShot(i, { start: Number(e.target.value) })} />
                <span className="text-[var(--color-muted)]">-</span>
                <input type="number" className={`${inp} w-14`} value={s.end} title="end (s)" onChange={(e) => patchShot(i, { end: Number(e.target.value) })} />
                <select className={`${inp} w-28`} value={s.type} onChange={(e) => patchShot(i, { type: e.target.value as Shot["type"] })}>
                  <option value="performance">performance</option>
                  <option value="narrative">narrative</option>
                  <option value="broll">b-roll</option>
                </select>
                <label className="flex items-center gap-1 text-[10px] text-[var(--color-muted)]"><input type="checkbox" checked={s.lipsync} onChange={(e) => patchShot(i, { lipsync: e.target.checked })} />lip-sync</label>
                {s.clipId && <span className="text-[10px] text-green-400" title="rendered">rendered</span>}
                <span className="ml-auto flex items-center gap-1">
                  <button onClick={() => moveShot(i, -1)} disabled={i === 0} className="text-[var(--color-muted)] hover:text-[var(--color-ink)] disabled:opacity-30 px-1" title="Move up">^</button>
                  <button onClick={() => moveShot(i, 1)} disabled={i === shots.length - 1} className="text-[var(--color-muted)] hover:text-[var(--color-ink)] disabled:opacity-30 px-1" title="Move down">v</button>
                  <button onClick={() => deleteShot(i)} className="text-[var(--color-muted)] hover:text-red-400 px-1" title="Delete shot">x</button>
                  <button onClick={() => genShot(s)} disabled={busy} className="rounded bg-[var(--color-accent)] px-2 py-0.5 text-[11px] text-white disabled:opacity-50">{s.clipId ? "Re-gen" : "Generate"}</button>
                </span>
              </div>
              <textarea className={inp} rows={2} value={s.scene} placeholder="scene (photoreal prompt)" onChange={(e) => patchShot(i, { scene: e.target.value })} />
              <div className="grid grid-cols-2 gap-1.5">
                <input className={inp} value={s.motion} placeholder="motion" onChange={(e) => patchShot(i, { motion: e.target.value })} />
                <input className={inp} value={s.characters.join(", ")} placeholder="characters (comma-separated)" onChange={(e) => patchShot(i, { characters: e.target.value.split(",").map((x) => x.trim()).filter(Boolean) })} />
              </div>
            </div>
          ))}
          <div className="flex items-center gap-2 pt-1">
            <PrimaryButton onClick={assemble} disabled={busy || shots.filter((s) => s.clipId).length === 0}>
              {`Assemble video (${shots.filter((s) => s.clipId).length}/${shots.length} shots rendered)`}
            </PrimaryButton>
          </div>
          <p className="text-[10px] text-[var(--color-muted)]">Edit any shot's timing, type, scene, motion, or characters; add/delete/reorder freely. Save to project to keep edits (and to let me inject or revise shots). Assemble stitches the rendered clips, fitted to their timings, with the song.</p>
        </div>
      )}
    </div>
  );
}
