import { useEffect, useState } from "react";
import { api } from "./api";
import { inp, GhostButton } from "./ui";

const TASKS = [
  { id: "lyrics", label: "Lyrics" },
  { id: "tags", label: "Style tags" },
  { id: "ideas", label: "Ideas" },
];

export function Assistant() {
  const [open, setOpen] = useState(false);
  const [ollama, setOllama] = useState<string[]>([]);
  const [claude, setClaude] = useState(false);
  const [provider, setProvider] = useState("ollama");
  const [model, setModel] = useState("");
  const [task, setTask] = useState("ideas");
  const [input, setInput] = useState("");
  const [out, setOut] = useState("");
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!open) return;
    api.llmProviders().then((p: any) => {
      setOllama(p.ollama || []);
      setClaude(!!p.claude);
      if (p.ollama?.length) setModel(p.ollama[0]);
    }).catch(() => {});
  }, [open]);

  async function run() {
    if (!input.trim()) return;
    setBusy(true); setOut(""); setCopied(false);
    try {
      const d = await api.llm({ provider, model: provider === "ollama" ? model : "", task, input });
      setOut(d.text);
    } catch (e) { setOut("✗ " + (e as Error).message); }
    setBusy(false);
  }

  return (
    <div className="border-t border-[var(--color-line)] bg-[var(--color-panel)]">
      <button onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-2 px-5 py-2.5 text-left text-sm">
        <span className="bg-gradient-to-r from-[var(--color-accent)] to-[var(--color-accent2)] bg-clip-text font-semibold text-transparent">✦ Assistant</span>
        <span className="text-xs text-[var(--color-muted)]">lyrics · style tags · ideas {claude ? "· Gemma/Claude" : "· Gemma"}</span>
        <span className="ml-auto text-[var(--color-muted)]">{open ? "▾" : "▴"}</span>
      </button>

      {open && (
        <div className="grid gap-3 px-5 pb-4 lg:grid-cols-[1fr_1fr]">
          <div className="space-y-2">
            <div className="flex gap-2">
              <select className={`${inp} flex-1`} value={task} onChange={(e) => setTask(e.target.value)}>
                {TASKS.map((t) => <option key={t.id} value={t.id}>{t.label}</option>)}
              </select>
              <select className="rounded-lg border border-[var(--color-line)] bg-[var(--color-panel2)] px-2 text-xs" value={provider} onChange={(e) => setProvider(e.target.value)}>
                <option value="ollama">Local (Gemma)</option>
                <option value="claude" disabled={!claude}>Claude {claude ? "" : "(no key)"}</option>
              </select>
              {provider === "ollama" && (
                <select className="rounded-lg border border-[var(--color-line)] bg-[var(--color-panel2)] px-2 text-xs" value={model} onChange={(e) => setModel(e.target.value)}>
                  {ollama.map((m) => <option key={m} value={m}>{m.replace(":latest", "")}</option>)}
                </select>
              )}
            </div>
            <textarea className={inp} rows={4} value={input} onChange={(e) => setInput(e.target.value)}
              placeholder={task === "lyrics" ? "theme, e.g. a doomed voyage across a frozen sea" : task === "tags" ? "an idea, e.g. epic folk metal drinking song" : "what do you want ideas for?"} />
            <GhostButton onClick={run}>{busy ? "Thinking…" : "Generate"}</GhostButton>
          </div>
          <div className="rounded-lg border border-[var(--color-line)] bg-[var(--color-panel2)] p-3">
            <div className="mb-1.5 flex items-center justify-between">
              <span className="text-xs text-[var(--color-muted)]">Result</span>
              {out && <button className="text-xs text-[var(--color-accent2)]"
                onClick={() => { navigator.clipboard.writeText(out); setCopied(true); }}>{copied ? "copied ✓" : "copy"}</button>}
            </div>
            <pre className="max-h-40 overflow-y-auto whitespace-pre-wrap text-xs text-[var(--color-ink)]">{out || <span className="text-[var(--color-muted)]">Output appears here — copy it into the prompt or lyrics field.</span>}</pre>
          </div>
        </div>
      )}
    </div>
  );
}
