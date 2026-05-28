import { useEffect, useState } from "react";
import { api, type SettingsField } from "./api";
import { inp, Field, PrimaryButton, GhostButton } from "./ui";

// Self-documenting Settings panel: renders form fields straight from the backend's
// curated field list (group + label + hint + type), so adding a new exposed setting
// is one backend change. Most edits need a ./run.sh restart to take effect.
export function SettingsModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [fields, setFields] = useState<SettingsField[]>([]);
  const [path, setPath] = useState("");
  const [show, setShow] = useState<Record<string, boolean>>({});
  const [msg, setMsg] = useState("");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!open) return;
    setMsg("");
    api.settings().then((d) => { setFields(d.fields); setPath(d.config_path); }).catch((e) => setMsg("Load failed: " + e.message));
  }, [open]);

  if (!open) return null;

  const groups = [...new Set(fields.map((f) => f.group))];
  const set = (k: string, v: string | boolean) => setFields((fs) => fs.map((f) => (f.key === k ? { ...f, value: v } : f)));

  async function save() {
    setSaving(true); setMsg("Saving…");
    try {
      const body: Record<string, unknown> = {};
      fields.forEach((f) => { body[f.key] = f.value; });
      const r = await api.settingsSave(body);
      setMsg(r.changes?.length ? `✓ Saved (${r.changes.length} change${r.changes.length === 1 ? "" : "s"}) — restart the backend (./run.sh) for changes to take effect.` : "✓ No changes.");
    } catch (e) { setMsg("✗ " + (e as Error).message); }
    finally { setSaving(false); }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" onClick={onClose}>
      <div className="flex max-h-[90vh] w-full max-w-2xl flex-col overflow-hidden rounded-xl border border-[var(--color-line)] bg-[var(--color-panel)] shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between border-b border-[var(--color-line)] px-4 py-2.5">
          <div>
            <div className="text-sm font-semibold text-[var(--color-ink)]">Settings</div>
            <div className="text-[10px] text-[var(--color-muted)]">{path}</div>
          </div>
          <button onClick={onClose} className="rounded text-[var(--color-muted)] hover:text-[var(--color-ink)]" title="Close">✕</button>
        </div>
        <div className="overflow-y-auto px-4 py-3 space-y-4 text-xs">
          <p className="rounded-md border border-amber-500/30 bg-amber-500/5 px-2.5 py-1.5 text-[11px] leading-relaxed text-amber-300/90">
            Most settings need a <b>./run.sh restart</b> to take effect (hosts, keys, flags are read at startup). Empty values disable the corresponding feature gracefully.
          </p>
          {groups.map((g) => (
            <div key={g} className="rounded-lg border border-[var(--color-line)] bg-[var(--color-panel2)] p-2.5">
              <div className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-[var(--color-muted)]">{g}</div>
              <div className="space-y-2.5">
                {fields.filter((f) => f.group === g).map((f) => (
                  <Field key={f.key} label={f.label} hint={f.hint}>
                    {f.type === "bool" ? (
                      <label className="flex items-center gap-2 text-[11px] text-[var(--color-muted)]">
                        <input type="checkbox" checked={!!f.value} onChange={(e) => set(f.key, e.target.checked)} />
                        {f.value ? "enabled" : "disabled"}
                      </label>
                    ) : f.type.startsWith("select:") ? (
                      <select className={inp} value={String(f.value || "")} onChange={(e) => set(f.key, e.target.value)}>
                        {f.type.split(":")[1].split(",").map((c) => <option key={c} value={c}>{c}</option>)}
                      </select>
                    ) : f.type === "secret" ? (
                      <div className="flex gap-1.5">
                        <input className={inp} type={show[f.key] ? "text" : "password"}
                          value={String(f.value || "")} onChange={(e) => set(f.key, e.target.value)}
                          placeholder="(empty = feature disabled)" autoComplete="off" />
                        <button onClick={() => setShow((s) => ({ ...s, [f.key]: !s[f.key] }))}
                          className="rounded border border-[var(--color-line)] px-2 text-[var(--color-muted)] hover:text-[var(--color-ink)]" title={show[f.key] ? "Hide" : "Show"}>
                          {show[f.key] ? "🙈" : "👁"}
                        </button>
                      </div>
                    ) : (
                      <input className={inp} value={String(f.value || "")} onChange={(e) => set(f.key, e.target.value)}
                        placeholder={f.key === "comfy_host" ? "WINDOWS_PC_IP:8188" : "<host>:<port>  (empty = disabled)"} autoComplete="off" />
                    )}
                  </Field>
                ))}
              </div>
            </div>
          ))}
        </div>
        <div className="flex items-center justify-between gap-2 border-t border-[var(--color-line)] px-4 py-2.5">
          <span className="flex-1 text-[11px] text-[var(--color-muted)]">{msg}</span>
          <GhostButton onClick={onClose}>Close</GhostButton>
          <PrimaryButton onClick={save} disabled={saving}>{saving ? "Saving…" : "Save"}</PrimaryButton>
        </div>
      </div>
    </div>
  );
}
