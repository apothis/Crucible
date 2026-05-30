# Box Service Orchestrator — Plan

_Standalone plan document. Designed to be picked up cold in a fresh session and executed end-to-end. Read this + HANDOFF.md and you have everything._

Last authored: 2026-05-30. Status: **planned, not yet started.** Tracked as task #13.

---

## 1. The problem

Today, every time we need to restart any service on the Windows GPU box (RTX 3090 at `192.168.1.201`), the Mac has to ask the user to physically close + relaunch a `.bat` file in a console window. This bites us repeatedly:

- **[[engine-fresh-boot-for-lora]]** — must restart `run_acestep_api.bat` before every training run. User does this manually each time.
- **[[no-concurrent-clap-engine]]** — must serialize analyze service and engine work. Mac can't enforce; relies on Claude remembering + user policing.
- **VRAM coordination** — on-demand services (RVC at ~3 GB, SoulX at ~5 GB, DiffSinger, RoFormer) ideally start only when needed and stop after, so the engine has full 24 GB. Today they sit idle holding VRAM.
- **State drift diagnosis** — when a service hangs or degrades, the only fix is "manually restart it". The Mac can't probe its process or force a kill.

Concrete pain we hit 2026-05-30: autolabel ran ~13× slower than baseline because engine state was drifting from a marathon session + my CLAP smoke test was concurrent. User had to look at the box's task manager, decide to restart, walk over to the console. Cost: ~5 min of explanation + ~30s of manual restart + ~10 min of stalled work.

## 2. The vision

A **box-side orchestrator service** that:

- Always-running Windows service (autostart on boot)
- Knows about all 9 box services and how to start/stop/health-check each
- HTTP API on a known port for Mac control
- Web UI on the same port for visual status
- Implements VRAM-coordination rules in one place
- Provides reliable process tree-kill (Python subprocesses are notorious for orphaning)

Mac integration:
- `backend/box_orchestrator.py` client with start/stop/restart/status/health calls
- `/api/box/*` routes on Mac
- A header chip in the Crucible UI showing live service state at a glance
- Existing routes (`_ensure_training_ready`, `_ensure_labeling_ready`, `free_gpu`) gain "via orchestrator" code paths that lift hard-coded service-name lists into the orchestrator's config

End state: instead of "user, please close + relaunch run_acestep_api.bat", the Mac calls `POST /orchestrator/services/acestep/restart` and we move on. Instead of "let me make sure SoulX is running before this vocal build", the Mac calls `POST /orchestrator/services/soulx/ensure_running` (idempotent).

## 3. Current service inventory (verified 2026-05-30)

From the project root's `*_AUTO_INSTALL.bat` set + `app_config.example.json` hosts:

| Service | Port | .bat installer | Generated launcher | Health endpoint | When needed | VRAM (rough) |
|---|---|---|---|---|---|---|
| **ACE-Step engine** | 8001 | `ACESTEP-ENGINE_AUTO_INSTALL.bat` | `run_acestep_api.bat` | `GET /v1/info` | always (generation + training) | 5-22 GB depending on state |
| **LoRA upload helper** | 5080 | `LORA-UPLOAD_AUTO_INSTALL.bat` | `run_lora_upload.bat` | `GET /health` | only when uploading dataset | <100 MB (CPU) |
| **Analyze service** | 5075 | `ANALYZE-API_AUTO_INSTALL.bat` | `run_analyze_api.bat` | `GET /health` | reference-track analysis + Plan 1 CLAP scoring | 3-4 GB (CLAP + allin1) |
| **RoFormer separator** | 5070 | `ROFORMER-API_AUTO_INSTALL.bat` | `run_roformer_api.bat` | `GET /health` | high-quality stem separation | 4-5 GB during run, idle releases |
| **RVC API** | 5050 | `RVC-API_AUTO_INSTALL.bat` | `run_rvc_api.bat` | `GET /models` | vocal re-timbring | 2-3 GB |
| **RVC WebUI (Gradio)** | 7897 | `RVC_AUTO_INSTALL.bat` | (WebUI launcher) | port probe | fallback for RVC | 2-3 GB |
| **SoulX-Singer** | 5060 | `SOULX-API_AUTO_INSTALL.bat` | `run_soulx_api.bat` | `GET /health` | word-singing vocal synthesis | 5 GB (on-demand load) |
| **DiffSinger MiniEngine** | 9266 | `DIFFSINGER-MINIENGINE_AUTO_INSTALL.bat` | (per the installer) | `GET /` or port probe | alternate vocal synthesis | 2-3 GB |
| **ComfyUI** | 8188 | `MUSICGEN-COMFYUI_AUTO_INSTALL.bat` | (ComfyUI's own launcher) | `GET /system_stats` | ACE-Step fallback + restyle/cover/repaint/lego/extract (per [[acestep-engine-outcome]]) | 6-10 GB |

Services that should be **always running** when the box is up: engine, ComfyUI, LoRA upload helper, analyze service (analyze is cheap-idle).

Services that should be **on-demand only** (start before use, stop after): RVC API, SoulX, DiffSinger, RoFormer. These eat VRAM that the engine or ComfyUI need.

## 4. Architecture

### 4.1 Orchestrator service (Python, FastAPI)

Single Python process, FastAPI, runs as a Windows service (or scheduled task) on box boot. Listens on a fixed port — **propose `:5099`**, chosen far from the service-port band.

Dependencies: `fastapi`, `uvicorn`, `psutil` (process tree management), `pynvml` (optional, for VRAM telemetry). Light; no torch / no GPU code. Single Python file ~400 LOC.

**Service registry** — config file `box_services.yaml` (next to the orchestrator script):

```yaml
services:
  acestep:
    label: "ACE-Step engine"
    port: 8001
    launcher: "E:\\AI\\MusicGen\\AceStep\\run_acestep_api.bat"
    cwd: "E:\\AI\\MusicGen\\AceStep"
    health_url: "http://127.0.0.1:8001/v1/info"
    startup_timeout_s: 180   # LM init takes a minute
    health_poll_interval_s: 5
    always_on: true          # default state on box boot
    vram_class: heavy        # for VRAM coordination
    auto_restart_on_crash: true
    log_file: "C:\\box_orchestrator\\logs\\acestep.log"
  analyze:
    label: "Analyze (CLAP + allin1)"
    port: 5075
    launcher: "E:\\AI\\MusicGen\\Analyze\\run_analyze_api.bat"
    ...
  # ... 8 more entries
```

### 4.2 Per-service state machine

Each service can be in one of: `idle | starting | running | stopping | crashed | unknown`.

Transitions:
- `idle → starting` on `POST /services/{name}/start` — spawn process via `subprocess.Popen`, capture PID
- `starting → running` when health probe succeeds within `startup_timeout_s`
- `starting → crashed` if timeout or process exits early
- `running → stopping` on `POST /services/{name}/stop` — psutil tree-kill (parent + all children)
- `stopping → idle` when PID gone and port free
- `running → crashed` if health probe fails N consecutive times (auto-restart kicks in if configured)
- `* → unknown` if orchestrator restarted while service was running — reconcile by port probe + PID lookup

### 4.3 Process management

Windows process management has known footguns:
- `.bat` launches Python which spawns workers — killing the .bat doesn't kill children
- Use **`psutil.Process(pid).children(recursive=True)`** + `terminate()` → fall back to `kill()` if still alive after 5s
- Use **Windows Job Objects** as second-line defense — every Popen'd child goes into a job, killing the job kills all descendants atomically
- Log every spawn / kill to `orchestrator.log`

VRAM cleanup after kill:
- For services known to leak GPU memory after process exit (notably the engine per task #18) — orchestrator could call `nvidia-smi --gpu-reset` or just wait + verify via pynvml. Most likely just process death is sufficient since Windows reclaims on process exit.

### 4.4 HTTP API

```
GET  /services                       → list with current state
GET  /services/{name}                → one service detail
POST /services/{name}/start          → idempotent — no-op if already running
POST /services/{name}/stop           → idempotent — no-op if not running
POST /services/{name}/restart        → stop + start
POST /services/{name}/ensure_running → start if not running, no-op otherwise
POST /services/{name}/ensure_stopped → stop if running, no-op otherwise
GET  /services/{name}/logs           → tail of N lines
GET  /services/{name}/health         → proxy the service's own health endpoint
GET  /vram                           → pynvml snapshot (free/used/total per GPU)
GET  /version                        → orchestrator version + git sha
POST /reload_config                  → re-read box_services.yaml without restart
```

Auth: **none** (trusted LAN, matches existing service pattern). Server binds `0.0.0.0` so the Mac at `192.168.1.x` can reach it.

### 4.5 Web UI

Single HTML page served by FastAPI at `/`. No build step — vanilla HTML + minimal JS (no React). Polls `GET /services` every 5s.

Rendering (~200 LOC of HTML+CSS+JS):

```
┌─────────────────────────────────────────────────────────────┐
│ Box Service Orchestrator              VRAM: 3.4 / 24.0 GB   │
├─────────────────────────────────────────────────────────────┤
│ ● ACE-Step engine             :8001  running    18m         │
│   memory 5.8 GB  vram 5.2 GB                  [Stop] [Restart] │
│ ● Analyze (CLAP + allin1)     :5075  running     6m         │
│   memory 1.2 GB                               [Stop] [Restart] │
│ ○ LoRA upload helper          :5080  idle                   │
│                                                  [Start]       │
│ ○ SoulX-Singer                :5060  idle                   │
│                                                  [Start]       │
│ ⚠ RVC API                     :5050  crashed                │
│   last error: connect timeout                 [Start] [Logs]   │
└─────────────────────────────────────────────────────────────┘
```

Colour states: green dot = running, grey = idle, amber = starting/stopping, red = crashed.

### 4.6 Mac integration

New `backend/box_orchestrator.py`:

```python
def list_services(host) -> list[dict]: ...
def get_service(host, name) -> dict: ...
def start(host, name, wait_healthy=True, timeout_s=180) -> dict: ...
def stop(host, name, wait_gone=True, timeout_s=30) -> dict: ...
def restart(host, name, wait_healthy=True) -> dict: ...
def ensure_running(host, name) -> dict: ...
def ensure_stopped(host, name) -> dict: ...
def vram(host) -> dict: ...
```

New Mac routes mirroring the orchestrator's:

```
GET  /api/box/services
POST /api/box/services/{name}/{action}
GET  /api/box/vram
```

Existing Mac code paths to migrate (one at a time, each gated on the orchestrator host being configured — fall back to current behavior when not):

- `_ensure_training_ready` (currently /v1/reinitialize) — now also accepts an `orchestrator_restart=True` mode that does a full engine restart via orchestrator. Slower (~90s) but actually clears VRAM-stickiness. Per [[engine-fresh-boot-for-lora]] this is what we always wanted but couldn't automate before.
- `_ensure_labeling_ready` — similar
- `free_gpu` (currently calls /free on ComfyUI + RVC) — now can additionally ensure_stopped on heavy on-demand services (SoulX, RoFormer)
- `voicegen.py` SoulX synth — pre-call `ensure_running("soulx")`, post-call `ensure_stopped("soulx")` if a flag is set
- Vocal Builder + Voice Swap — same pattern for SoulX
- Plan 1's post-training eval — uses `restart("acestep")` between training and eval (fresh inference state), per METAL_LORA_PLAN §11.3

New CFG key `orchestrator_host: "192.168.1.201:5099"` added to `app_config.json` curated keys in the Settings panel.

### 4.7 Crucible UI — Box state chip

Header chip next to the existing ACE chip + ProjectBar:

```
[🟢 box: 4 running ▾]
```

Click expands to a popover showing the same info as the orchestrator's web UI (filtered to a less-busy view — only running + on-demand-when-needed). Per-service Start/Stop/Restart actions. Polls `/api/box/services` every 10s when popover open, every 60s otherwise.

## 5. Phased plan (suggested execution order)

### Phase 1 — Box-side orchestrator MVP (~half day)
- [ ] Decide install location on box: propose `C:\box_orchestrator\` (separate from each service's install)
- [ ] Write `orchestrator.py` (~400 LOC)
- [ ] Write `box_services.yaml` with all 9 services from §3
- [ ] Build process management: `psutil` tree-kill + Windows Job Objects
- [ ] Build the 12 HTTP endpoints (§4.4)
- [ ] Health probe loop (background thread, configurable interval)
- [ ] Crash detection + auto-restart for `auto_restart_on_crash: true` services
- [ ] Logging to per-service rotating log files
- [ ] Write `INSTALL.bat` that:
  - Creates venv in `C:\box_orchestrator\venv\`
  - Installs `fastapi uvicorn psutil pynvml pyyaml requests`
  - Writes `run_orchestrator.bat`
  - Optionally registers as Windows service via `nssm` (preferred — more reliable than Task Scheduler)
- [ ] Manual test: each service start/stop/restart from `curl`, verify PID + port cleanup

### Phase 2 — Box-side UI (~2h)
- [ ] Single HTML page served from `/`
- [ ] Auto-refresh every 5s
- [ ] Action buttons posting to `POST /services/{name}/{action}`
- [ ] Logs popover (tail last 200 lines from per-service log)
- [ ] VRAM gauge at top

### Phase 3 — Mac client + routes (~2h)
- [ ] `backend/box_orchestrator.py` client (~80 LOC)
- [ ] `/api/box/services` routes in `app.py`
- [ ] `orchestrator_host` config key + Settings panel field
- [ ] Compatibility: when `orchestrator_host` empty, fall back to current behavior (per [[optional-additions]])

### Phase 4 — Crucible UI chip (~3h)
- [ ] Header chip component with status badge + popover
- [ ] Hook into App state for periodic polling
- [ ] Per-service action confirmations
- [ ] Loading + error states

### Phase 5 — Migrate existing code paths (~3h, one at a time with checks)
- [ ] `_ensure_training_ready` → support `orchestrator_restart` mode
- [ ] `_ensure_labeling_ready` → same
- [ ] `free_gpu` → optionally stop on-demand heavy services
- [ ] Vocal Builder + Voice Swap → wrap SoulX with ensure_running / ensure_stopped
- [ ] METAL_LORA_PLAN §11.3 post-training CLAP eval → uses `restart("acestep")` between training and eval

### Phase 6 — Always-on at box boot (~1h)
- [ ] Register orchestrator as Windows service (nssm) so it autostarts
- [ ] Orchestrator on-boot logic: start all `always_on: true` services
- [ ] Verify on a real reboot

## 6. Checks along the way (avoid regressions)

- **Check #1 — process kill actually kills children**: spawn a service, verify with Task Manager that python.exe children exist, hit `/services/{name}/stop`, verify all PIDs gone within 10s.
- **Check #2 — orchestrator-restart-recovery**: spawn a service, restart the orchestrator itself, verify it reconciles state (sees existing PID listening on its port, sets state=running without re-spawning).
- **Check #3 — VRAM actually clears**: kill the engine after a long training session, verify nvidia-smi shows VRAM dropping (engine has the documented "VRAM stickiness" bug per [[engine-fresh-boot-for-lora]] — this is the test that proves the orchestrator fix is real).
- **Check #4 — Mac fallback when orchestrator down**: blackhole the orchestrator port from Mac side, verify all existing flows still work (per [[optional-additions]] no regressions).
- **Check #5 — concurrent action safety**: hammer `POST /services/acestep/restart` 5× concurrently, verify no zombie processes, end state is single running process.

## 7. Risks + open questions

- **Windows process management quirks** — main risk. Mitigated by psutil + Job Objects + extensive Check #1 testing.
- **Orchestrator-as-service install** — nssm is the recommended tool but adds an external dep. Alternative is Task Scheduler; less reliable but more portable.
- **GPU process kill ≠ VRAM release** — per task #18, the engine's VRAM stickiness might survive even an orchestrator-driven kill. Worth testing as Check #3 explicitly. If kill alone doesn't release, orchestrator could add `nvidia-smi --gpu-reset` as an optional post-kill action.
- **What about ComfyUI?** It's the heaviest always-on service; its lifecycle should be handled by the orchestrator too. But ComfyUI has its own queue + websocket clients that could be mid-job during a restart. Decide whether to drain queue before stop.
- **Who installs the orchestrator initially?** Probably a `BOX_ORCHESTRATOR_AUTO_INSTALL.bat` in the project root, in the same convention as the 9 existing installers.
- **Logging volume** — services like ACE-Step training emit a lot. Rotation by size (e.g. 50 MB per file, 5 files retained) is essential.
- **Authentication** — none in MVP per existing pattern. If we ever expose the orchestrator beyond LAN, this changes.

## 8. Operating rules + memories worth touching

After Phase 5 ships:

- **[[engine-fresh-boot-for-lora]]** memory gets an update: "the orchestrator can now do this for you; the OS-restart pattern survives as the manual fallback when orchestrator is down."
- **[[no-concurrent-clap-engine]]** memory gets an update: "the orchestrator's free_gpu integration enforces this automatically when migrated routes call it; Claude should still respect manually."
- A new memory **[[box-orchestrator]]** documenting the orchestrator's existence, port, and the ensure_running / ensure_stopped pattern. So future Claude knows it's there.

After the orchestrator is reliable, the existing "user, please close + relaunch run_acestep_api.bat" pattern in the user-facing flow disappears entirely. That's the goal.

## 9. Sources / references

- `ACESTEP-ENGINE_AUTO_INSTALL.bat` etc — all 9 installers (paths + launcher conventions)
- `app_config.example.json` — current host config keys
- `backend/app.py free_gpu()` + `_ensure_training_ready()` + `_ensure_labeling_ready()` — existing GPU-coordination patterns to migrate
- HANDOFF.md SESSION 2026-05-30 — context on the autolabel slowness that triggered this work
- [[engine-fresh-boot-for-lora]], [[no-concurrent-clap-engine]] — operating constraints this plan automates
- psutil docs — `Process.children(recursive=True)`, `wait()`, `terminate()` vs `kill()`
- pynvml docs — VRAM telemetry
- nssm — recommended Windows-service wrapper

## 10. Out of scope for this plan

These are tempting but explicitly NOT included to keep the orchestrator focused:

- Cross-machine orchestration (Mac is the only client)
- Service installation / updates (the existing installers do that; orchestrator just runs them)
- Backup / restore of service state
- Metrics export to Prometheus / Grafana — overkill for our single-box single-user setup
- Authentication / RBAC — single-user trusted LAN
- A separate CLI tool — the web UI + Mac routes are enough
