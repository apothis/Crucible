import { useEffect, useState } from "react";
import { api, type Config, type LibItem } from "./api";
import { Field, inp, PrimaryButton, GhostButton, rid, pollJob, type RunCtx } from "./ui";
import { useDrafts } from "./drafts";

type Shot = {
  idx: number; section: string; start: number; end: number;
  type: "performance" | "narrative" | "broll";
  scene: string; motion: string; characters: string[]; lipsync: boolean;
};
type Character = { name: string; role: string; refStillId: string };
type Project = { id: string; name: string };

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
  const [gen, setGen] = useState(false);

  useEffect(() => { api.projects().then(setProjects).catch(() => {}); }, []);
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

  // generate one shot: lip-sync (if a named singer w/ ref still + audio) -> S2V; character
  // shot -> i2v anchored on the character's ref still; scenic -> fresh still then i2v.
  async function genShot(shot: Shot) {
    const char = cast.find((c) => shot.characters.includes(c.name) && c.refStillId);
    const card = { id: rid(), title: `shot ${shot.idx + 1} (${shot.type})`, status: "pending" as const, pct: 0 };
    ctx.setResults([card]);
    try {
      if (shot.lipsync && char && audioId) {
        const { job_id } = await api.videoLipsync({ still_id: char.refStillId, audio_id: audioId, prompt: shot.scene, audio_start: shot.start }) as { job_id: string };
        ctx.patch(card.id, { status: "running", pct: 5 }); pollJob(job_id, card.id, ctx);
      } else if (char) {
        const { job_id } = await api.videoI2V({ still_id: char.refStillId, prompt: shot.motion || shot.scene }) as { job_id: string };
        ctx.patch(card.id, { status: "running", pct: 5 }); pollJob(job_id, card.id, ctx);
      } else {
        ctx.patch(card.id, { status: "running", pct: 3, title: `shot ${shot.idx + 1}: still...` });
        const still = await api.videoStill({ prompt: shot.scene }) as { job_id: string };
        await waitDone(still.job_id);
        ctx.patch(card.id, { pct: 50, title: `shot ${shot.idx + 1}: animating...` });
        const { job_id } = await api.videoI2V({ still_id: still.job_id, prompt: shot.motion || "subtle cinematic motion" }) as { job_id: string };
        pollJob(job_id, card.id, ctx);
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
      <p className="text-[11px] text-[var(--color-muted)]">Turn a song into a beat-cut music video: generate a shot list, define the cast, then render each shot. (Phase A: anchor-still consistency.)</p>

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
          <div key={i} className="grid grid-cols-[1fr_1fr_1.4fr_auto] items-center gap-1.5">
            <input className={inp} placeholder="name" value={c.name} onChange={(e) => setCast(cast.map((x, j) => j === i ? { ...x, name: e.target.value } : x))} />
            <input className={inp} placeholder="role" value={c.role} onChange={(e) => setCast(cast.map((x, j) => j === i ? { ...x, role: e.target.value } : x))} />
            <select className={inp} value={c.refStillId} onChange={(e) => setCast(cast.map((x, j) => j === i ? { ...x, refStillId: e.target.value } : x))}>
              <option value="">- reference still -</option>
              {stills.map((s) => <option key={s.id} value={s.id}>{(s.params?.prompt || s.id).toString().slice(0, 36)}</option>)}
            </select>
            <button onClick={() => setCast(cast.filter((_, j) => j !== i))} className="text-[var(--color-muted)] hover:text-red-400 px-1" title="Remove">x</button>
          </div>
        ))}
        <p className="text-[10px] text-[var(--color-muted)]">Make reference stills in the Video tab (Still), then pick them here.</p>
      </div>

      <PrimaryButton onClick={generateScript} disabled={gen || !projectId}>{gen ? "Writing script..." : shots.length ? "Regenerate script" : "Generate script"}</PrimaryButton>

      {/* Shot list */}
      {shots.length > 0 && (
        <div className="space-y-2">
          <div className="text-xs font-semibold text-[var(--color-accent2)]">{shots.length} shots</div>
          {shots.map((s, i) => (
            <div key={i} className="rounded-lg border border-[var(--color-line)] bg-[var(--color-panel2)] p-2.5 space-y-1.5">
              <div className="flex items-center gap-2 text-[11px]">
                <span className="rounded bg-[#2a1c19] px-1.5 py-0.5 text-[var(--color-accent2)]">{i + 1}</span>
                <span className="text-[var(--color-muted)]">{s.start}-{s.end}s</span>
                <span className="rounded bg-[var(--color-panel)] px-1.5 py-0.5">{s.section}</span>
                <span className="rounded bg-[var(--color-panel)] px-1.5 py-0.5">{s.type}</span>
                <label className="flex items-center gap-1 text-[10px] text-[var(--color-muted)]"><input type="checkbox" checked={s.lipsync} onChange={(e) => patchShot(i, { lipsync: e.target.checked })} />lip-sync</label>
                {s.characters.length > 0 && <span className="text-[10px] text-[var(--color-muted)]">{s.characters.join(", ")}</span>}
                <button onClick={() => genShot(s)} disabled={busy} className="ml-auto rounded bg-[var(--color-accent)] px-2 py-0.5 text-[11px] text-white disabled:opacity-50">Generate</button>
              </div>
              <textarea className={inp} rows={2} value={s.scene} onChange={(e) => patchShot(i, { scene: e.target.value })} />
              <input className={inp} value={s.motion} placeholder="motion" onChange={(e) => patchShot(i, { motion: e.target.value })} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
