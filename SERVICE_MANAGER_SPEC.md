# Crucible Service Manager — build spec (Windows GPU box)

**Audience:** a coding agent (Claude) running **on the Windows PC** that hosts Crucible's GPU services. Build *and test* this on that machine — it manages local processes and must be verified against the real services.

## Why this exists
Crucible's Mac app drives several long-running services on this Windows box (ComfyUI + a few FastAPI servers), each currently started by hand via its own `.bat` in its own terminal. That's fragile — lots to start after a reboot, and nothing notices if one crashes. Build a small Windows app that **starts/stops/monitors** each service from one window and **auto-restarts a service if it crashes** (but not if the user stopped it deliberately).

## What to build (agreed requirements)
- **A simple always-available window** (a dashboard), not a tray-only app. One **row per service** showing:
  - a **status light** — Stopped / Starting / Running (healthy) / Crashed / Failed
  - **Start**, **Stop**, **Restart** buttons (enable/disable per state)
  - the port, and a way to **view the recent log** (tail) for that service
  - optional: uptime, last health-check time
- Plus **Start All** / **Stop All** buttons.
- **Config-driven service list** — services are defined in an editable `services.json` next to the app (schema below), so new services can be added/removed without code changes. Do NOT hardcode the service list.
- **Auto-restart ONLY on real crashes:** if a service's process dies (or stays unhealthy) while we expected it Running **and the user did not Stop it from the GUI**, restart it (with backoff + a max-attempts cap, then mark **Failed** and stop retrying). If the user clicked **Stop**, set a `user_stopped` flag and do **not** restart.
- Boot-autostart of the manager/services is **optional / nice-to-have**, not required.

## Tech
- **Python 3** (already on the box) with **`psutil`** (process trees) + **`requests`** (health checks). GUI: use a toolkit that's easy to ship on Windows — **Tkinter** (stdlib, zero extra install) is fine and recommended; PySide6/PySimpleGUI acceptable if you prefer. Keep dependencies minimal.
- Ship a `run_service_manager.bat` launcher and a `requirements.txt`.

## ⛔ HARD SAFETY RULE — read first (a violation already broke everything once)
**NEVER kill processes by image name or globally — only by the exact PID this manager launched, plus that PID's own child tree.** All managed services are `python.exe`, so a broad kill nukes every other service at once.
- **FORBIDDEN, anywhere — including while testing/developing the app:** `taskkill /F /IM python.exe`, `taskkill /IM pythonw.exe`, `Stop-Process -Name python*`, `pkill python`, `wmic process where name='python.exe' delete`, or any kill that targets a process the manager did not itself start.
- **ALLOWED:** terminate only `psutil.Process(<pid_we_started>)` and its `children(recursive=True)`, or `taskkill /PID <pid_we_started> /T /F`. Record each launched PID; only ever act on those.
- Before shipping/testing a Stop or Restart, assert the target PID is one the manager owns. When in doubt, do nothing and surface an error rather than kill broadly.
- (Real incident 2026-05-25: a "kill all python" issued during testing killed ComfyUI/RVC/SoulX/RoFormer and forced a full manual restart. Do not repeat.)

## Critical Windows gotchas (get these right)
1. **Killing a service kills its whole tree.** Several services are launched via a `.bat` that spawns `python.exe` (e.g. ComfyUI's `.bat` → `python_embeded\python.exe`). Terminating the `.bat`'s PID will **orphan the real python**, leaving the port in use. Launch each service with `subprocess.Popen([...], cwd=..., creationflags=CREATE_NEW_PROCESS_GROUP)` and on Stop, **kill the entire process tree** via `psutil.Process(pid).children(recursive=True)` (terminate children then parent), or `taskkill /PID <pid> /T /F`. Verify the port is actually freed after Stop.
2. **Crash vs. user-stop.** Maintain per-service state: `desired` (running/stopped) and `user_stopped`. The watchdog restarts only when `desired==running` and the process is gone/unhealthy and `user_stopped` is False.
3. **Startup grace period.** Models load slowly — ComfyUI and SoulX can take 10–60 s before their health endpoint answers. Treat "process alive but health not yet OK" as **Starting** for a configurable `startup_grace_sec` (default 90) before considering it Crashed.
4. **Restart backoff.** On crash: restart after an increasing delay (e.g. 3 s, 10 s, 30 s), cap at `max_restarts` (default 5) within a rolling window; then mark **Failed** and stop auto-retrying (user can still Start manually).
5. **Port-in-use / already-running detection.** On manager start, for each service do a health probe first; if it's already responding (started outside the manager), show it **Running** and attach (don't relaunch). If the port is in use but unhealthy, surface a clear error rather than spawning a duplicate.
6. **Logs.** Redirect each service's stdout/stderr to a per-service rolling log file (e.g. `logs/<name>.log`); the "view log" button tails the last ~200 lines. Don't let logs grow unbounded.

## Health model
Status per service = combine process-alive + HTTP health:
- process not started → **Stopped**
- process alive, within grace, health not OK yet → **Starting**
- process alive AND health OK → **Running**
- process died (or health failing past grace) while `desired==running` & not `user_stopped` → **Crashed** → trigger restart logic
- exceeded `max_restarts` → **Failed**
Poll health every `poll_sec` (default 5). A health check = HTTP GET the `health_url`, any 2xx within a short timeout = OK. (ComfyUI's `/system_stats` returns 200 with JSON; the FastAPI servers return 200.)

## `services.json` schema (the source of truth; user edits paths)
```json
{
  "poll_sec": 5,
  "startup_grace_sec": 90,
  "max_restarts": 5,
  "services": [
    {
      "name": "ComfyUI",
      "cwd": "C:\\AI\\ComfyUI_windows_portable",
      "command": ["run_musicgen_lan.bat"],
      "port": 8188,
      "health_url": "http://127.0.0.1:8188/system_stats",
      "autostart": true,
      "enabled": true
    },
    {
      "name": "RVC API",
      "cwd": "C:\\AI\\RVC...",
      "command": ["run_rvc_api.bat"],
      "port": 5050,
      "health_url": "http://127.0.0.1:5050/models",
      "autostart": true,
      "enabled": true
    },
    {
      "name": "SoulX API",
      "cwd": "C:\\AI\\SoulX...",
      "command": ["run_soulx_api.bat"],
      "port": 5060,
      "health_url": "http://127.0.0.1:5060/health",
      "autostart": false,
      "enabled": true
    },
    {
      "name": "RoFormer API",
      "cwd": "C:\\AI\\RoformerSep",
      "command": ["run_roformer_api.bat"],
      "port": 5070,
      "health_url": "http://127.0.0.1:5070/health",
      "autostart": true,
      "enabled": true
    }
  ]
}
```
Notes for whoever fills this in:
- `cwd` = the folder that contains the service's `.bat` (the user knows these from when they ran each `*_AUTO_INSTALL.bat`). The manager runs `command` with that `cwd`.
- Prefer launching the existing `.bat` (it sets the right env/venv). Tree-kill on stop handles the spawned python (gotcha #1).
- `autostart` here is a per-service hint for an optional "start on manager launch"; not required to implement, but harmless to support.
- `enabled:false` = show the row greyed/hidden; don't manage it.
- Real ports/health URLs are listed above; the FastAPI servers also serve their normal API on the same port. DiffSinger (`:9266`, `GET /models`) may be added later — keep it generic.

## Acceptance checklist (test on the box before declaring done)
1. Launch manager → it probes and shows each service's true state (already-running ones show Running without relaunch).
2. Click **Start** on a stopped service → goes Starting → Running once healthy; log tail shows output.
3. Externally kill the service's python (Task Manager) → manager detects within ~`poll_sec`, shows Crashed, **auto-restarts** back to Running.
4. Click **Stop** → process tree dies, **port is freed**, status stays **Stopped** and does **NOT** auto-restart.
5. Click **Restart** → clean stop then start.
6. **Start All** / **Stop All** work.
7. Restart the manager itself → it re-attaches to still-running services rather than duplicating them.
8. Kill a service repeatedly fast → backoff applies, eventually **Failed**, no tight restart loop.

## Out of scope
- No remote control, auth, or web UI — local Windows window only.
- VRAM management: these services share one RTX 3090; the manager only starts/stops/monitors. (The Mac app already calls ComfyUI `/free` before heavy GPU work; the manager doesn't need VRAM logic.)
