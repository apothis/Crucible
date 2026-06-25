import { useEffect, useRef } from "react";
import { TimelineEditor } from "./vendor/ltxdirector/ltx_director.js";

// React wrapper that mounts the vendored (GPL-3) LTXDirector TimelineEditor standalone — no ComfyUI.
// The editor reads/writes everything through a shim "node" (widgets by name + a properties bag) and
// loads/uploads media through the box ComfyUI (window.__LTXD_BOX__). We surface ONE thing back to the
// app: the timeline_data JSON (same schema our LTXDirector builders consume), via onChange.

// The editor's file ops (/view, /upload/image, /ltx_director_*) go through our same-origin backend
// proxy (/api/comfy/<path> -> box ComfyUI), so the browser editor isn't CORS-blocked.
const BOX = "/api/comfy";

// every widget the TimelineEditor looks up by name (constructor + commitChanges)
const WIDGETS: Record<string, unknown> = {
  timeline_data: "", local_prompts: "", segment_lengths: "", guide_strength: "", audio_data: "",
  use_custom_audio: false, inpaint_audio: true, use_custom_motion: true, override_audio: false,
  start_frame: 0, start_second: 0, end_frame: 120, end_second: 5,
  duration_frames: 120, duration_seconds: 5, frame_rate: 24, display_mode: "frames",
};

export function LtxDirectorEditor({ timelineData, frames, fps, onChange }: {
  timelineData?: string; frames: number; fps: number; onChange: (json: string) => void;
}) {
  const ref = useRef<HTMLDivElement | null>(null);
  const edRef = useRef<TimelineEditor | null>(null);
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;

  useEffect(() => {
    (window as unknown as { __LTXD_BOX__: string }).__LTXD_BOX__ = BOX;
    const host = ref.current;
    if (!host) return;

    // build the shim node: widgets expose `value` get/set; setting timeline_data bubbles to the app
    const init = { ...WIDGETS, timeline_data: timelineData || "", end_frame: frames, end_second: frames / fps,
      duration_frames: frames, duration_seconds: frames / fps, frame_rate: fps };
    const widgets = Object.keys(init).map((name) => {
      const w: { name: string; _v: unknown; value?: unknown; callback?: (v: unknown) => void; options: Record<string, unknown> } =
        { name, _v: (init as Record<string, unknown>)[name], options: {} };
      Object.defineProperty(w, "value", {
        get() { return w._v; },
        set(v: unknown) { w._v = v; if (name === "timeline_data" && typeof v === "string") onChangeRef.current(v); },
      });
      return w;
    });
    const node = {
      widgets, properties: {} as Record<string, unknown>,
      setDirtyCanvas: () => {},
      addDOMWidget: () => ({ element: host }),
    };

    try {
      edRef.current = new TimelineEditor(node, host, { element: host });
    } catch (e) {
      console.error("[ShotStudio] LTXDirector editor failed to mount:", e);
      host.innerHTML = `<div style="padding:16px;color:#c66;font-size:12px">Timeline editor failed to mount — see console. (${(e as Error).message})</div>`;
    }
    return () => {
      try { edRef.current?.destroy?.(); } catch { /* editor has no destroy; just detach */ }
      if (host) host.innerHTML = "";
      edRef.current = null;
    };
    // mount once; the editor owns its own state after that
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return <div ref={ref} className="ltxd-editor" style={{ minHeight: 380, width: "100%" }} />;
}
