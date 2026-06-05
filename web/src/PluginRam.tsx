import { useEffect, useState } from "react";
import { api } from "./api";
import { GhostButton } from "./ui";

type Status = { kontakt_loaded: boolean; kontakt_rss_mb: number | null; idle_sec: number };

// Shared "Plugin RAM" row used by the Add Solo / Guitar / Tone tabs. Shows whether
// Shreddage/Kontakt is resident (the big hog, ~560 MB) + its footprint, with a manual
// "Unload plugins (free RAM)" button and an opt-in idle auto-unload toggle. Self-contained:
// refreshes on mount, polls lightly, on its own actions, and whenever `refreshKey` changes
// (parents bump it after a render so the status reflects a freshly-loaded plugin promptly).
export function PluginRam({ refreshKey = 0 }: { refreshKey?: number }) {
  const [plug, setPlug] = useState<Status | null>(null);
  const [unloading, setUnloading] = useState(false);
  const refresh = () => api.pluginsStatus().then(setPlug).catch(() => {});

  useEffect(() => { refresh(); }, [refreshKey]);
  useEffect(() => {
    const id = setInterval(refresh, 12000);     // catch idle auto-unload / other tabs
    return () => clearInterval(id);
  }, []);

  if (!plug) return null;

  async function unload() {
    setUnloading(true);
    try { await api.pluginsUnload(); } catch { /* ignore */ }
    await refresh(); setUnloading(false);
  }
  async function toggleIdle() {
    const on = (plug!.idle_sec || 0) > 0;
    try { await api.pluginsIdle(on ? 0 : 300); } catch { /* ignore */ }
    await refresh();
  }

  return (
    <div className="flex flex-wrap items-center gap-2 rounded-lg border border-[var(--color-line)] bg-[var(--color-panel2)] px-3 py-2 text-[11px] text-[var(--color-muted)]">
      <span>
        Plugin RAM: {plug.kontakt_loaded
          ? <span className="text-[var(--color-accent2)]">Shreddage loaded{plug.kontakt_rss_mb ? ` · ~${plug.kontakt_rss_mb} MB` : ""}</span>
          : "Shreddage not loaded"}
      </span>
      <span className="flex-1" />
      <label className="flex items-center gap-1.5" title="Free Shreddage after 5 min idle; reloads on next render">
        <input type="checkbox" checked={(plug.idle_sec || 0) > 0} onChange={toggleIdle} /> auto-unload when idle
      </label>
      <GhostButton onClick={unload} disabled={unloading}>
        {unloading ? "Freeing…" : "Unload plugins (free RAM)"}
      </GhostButton>
    </div>
  );
}
