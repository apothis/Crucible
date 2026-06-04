"""Persistent Kontakt (Shreddage) DI render daemon + its in-backend client.

WHY THIS EXISTS: pedalboard's `load_plugin` for the Kontakt VST3 (an instrument)
throws "Caught an unknown exception!" on any NON-main thread, and even when loaded
on the main thread it refuses to render from a worker unless reset=False. FastAPI
runs sync endpoints on a worker threadpool, so loading Kontakt inline always failed
-> the Kontakt/Shreddage DI engine 500'd every time (Helix, an effect, is unaffected).

Fix: run Kontakt in its OWN process where the render happens on THAT process's main
thread (clean reset per render, no threading restriction), and a Kontakt crash can't
take the backend down. Same subprocess precedent as `plugin_capture.py`.

Protocol (newline-delimited, sentinel-prefixed so stray plugin stdout can't corrupt it):
  daemon -> client:  "@@KREADY@@"                         once the plugin+state are loaded
  daemon -> client:  "@@KERR@@{json}"                     fatal load error (then exits)
  client -> daemon:  {"notes": [[p,start,dur,vel],...], "sr": 44100, "out": "/abs.wav"}\n
  daemon -> client:  "@@KRESP@@{\"ok\":true,\"out\":...}" per request (or ok:false+error)
The daemon writes the rendered WAV to `out` itself (no large audio over the pipe).
Closing the daemon's stdin (EOF) -> it exits cleanly, so a backend restart never
leaves an orphaned Kontakt process.
"""
import json
import sys

READY = "@@KREADY@@"
RESP = "@@KRESP@@"
LOADERR = "@@KERR@@"


# ----------------------------- daemon side -----------------------------
def _build_messages(notes, pb_range=2.0):
    """Build the mido message list, realizing the optional articulation (5th field)
    using Shreddage 3's NATIVE expression (verified against the S3 Stratus FREE manual,
    no patch-specific keyswitches needed):
      b bend   -> pitch-bend scoop a whole step up into the note (clamped to pb_range)
      s slide  -> pitch-bend glide from the previous pitch (or legato overlap if too far)
      v vibrato-> mod wheel (CC1) raised after the onset, released at note-off
      h hammer -> start a touch early so the note overlaps the previous (S3 auto-legato)
      ~ let-ring / . staccato -> handled by note duration upstream
    Pitch-bend/CC are global, which is fine for a monophonic lead line."""
    import mido

    def wheel(semi):
        return int(max(-8191, min(8191, round(8191 * semi / max(0.1, pb_range)))))

    msgs = [mido.Message("pitchwheel", pitch=0, time=0.0),
            mido.Message("control_change", control=1, value=0, time=0.0)]
    prev_pitch = None
    for n in notes:
        pitch = int(max(0, min(127, n[0])))
        st = float(n[1]); dur = float(max(0.05, n[2])); vel = int(max(1, min(127, n[3])))
        art = (n[4] if len(n) > 4 else "")
        on, off = st, st + dur
        glide = None                                       # (start_semitones, fraction_of_note)
        if art == "b":
            glide = (-2.0, 0.35)                            # scoop a whole step up into the note
        elif art == "s" and prev_pitch is not None:
            interval = prev_pitch - pitch
            if abs(interval) <= pb_range:
                glide = (float(interval), 0.30)            # slide in from the previous pitch
            else:
                on = max(0.0, st - 0.006)                  # too far for pitch-bend -> legato overlap
        elif art == "h":
            on = max(0.0, st - 0.006)                      # overlap previous -> S3 hammer-on/pull-off
        if glide is not None:
            semi0, frac = glide
            gl_end = on + dur * frac
            steps = 8
            for k in range(steps + 1):
                tt = on + (gl_end - on) * (k / steps)
                msgs.append(mido.Message("pitchwheel", pitch=wheel(semi0 * (1 - k / steps)), time=tt))
        else:
            msgs.append(mido.Message("pitchwheel", pitch=0, time=on))   # center (clears a prior bend)
        if art == "v":
            msgs.append(mido.Message("control_change", control=1, value=95,
                                     time=on + min(0.08, dur * 0.3)))
            msgs.append(mido.Message("control_change", control=1, value=0, time=off))
        msgs.append(mido.Message("note_on", note=pitch, velocity=vel, time=on))
        msgs.append(mido.Message("note_off", note=pitch, velocity=0, time=off))
        prev_pitch = pitch
    msgs.sort(key=lambda m: m.time)
    return msgs


def _render(plugin, notes, sr, out_path, pb_range=2.0):
    import numpy as np
    import soundfile as sf
    msgs = _build_messages(notes, pb_range=pb_range)
    total = (max(m.time for m in msgs) + 0.5) if msgs else 1.0
    # render on THIS process's main thread (default reset=True is fine + clean here)
    audio = np.asarray(plugin(msgs, duration=total, sample_rate=sr))   # (channels, samples)
    if audio.ndim == 2 and audio.shape[0] <= 2:
        audio = audio.T                                                 # -> (samples, channels)
    elif audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.99:
        audio = audio * (0.99 / peak)
    sf.write(out_path, audio.astype("float32"), sr, subtype="PCM_16")
    return float(peak)


def main():
    plugin_path, state_path = sys.argv[1], sys.argv[2]
    import os
    import threading
    import time
    try:
        import pedalboard as pb
        plugin = pb.load_plugin(plugin_path)            # ~25s; on this process's main thread
        if state_path and os.path.exists(state_path):
            with open(state_path, "rb") as f:
                plugin.raw_state = f.read()             # restore the captured Shreddage patch
    except Exception as e:                              # fatal: report + exit so client can fall back
        sys.stdout.write(LOADERR + json.dumps({"error": repr(e)}) + "\n")
        sys.stdout.flush()
        return
    # Optional idle auto-unload: free the (big) Shreddage samples after N idle seconds.
    # Default off; the client sets MG_KONTAKT_IDLE_SEC when the user opts in. The client
    # respawns the daemon (~25-40s) on the next render.
    idle = 0.0
    try:
        idle = float(os.environ.get("MG_KONTAKT_IDLE_SEC", "0") or 0)
    except ValueError:
        idle = 0.0
    last = [time.time()]
    if idle > 0:
        def _watch():
            while True:
                time.sleep(min(idle, 15.0))
                if time.time() - last[0] > idle:
                    os._exit(0)                         # exit -> OS frees all the RAM
        threading.Thread(target=_watch, daemon=True).start()
    sys.stdout.write(READY + "\n")
    sys.stdout.flush()
    for line in sys.stdin:                              # EOF (parent died) -> loop ends -> exit
        line = line.strip()
        if not line:
            continue
        last[0] = time.time()                           # mark activity for the idle watcher
        try:
            req = json.loads(line)
            peak = _render(plugin, req["notes"], int(req.get("sr", 44100)), req["out"],
                           pb_range=float(req.get("pb_range", 2.0)))
            out = json.dumps({"ok": True, "out": req["out"], "peak": peak})
        except Exception as e:
            out = json.dumps({"ok": False, "error": repr(e)})
        sys.stdout.write(RESP + out + "\n")
        sys.stdout.flush()


# ----------------------------- client side -----------------------------
class _KontaktClient:
    """Manages one long-lived daemon process, keyed by (plugin_path, state_path).
    Thread-safe: a lock serializes requests (the daemon renders one at a time)."""

    def __init__(self):
        import threading
        self._lock = threading.Lock()
        self._proc = None
        self._key = None

    def _alive(self):
        return self._proc is not None and self._proc.poll() is None

    def _spawn(self, plugin_path, state_path, load_timeout=90):
        import os
        import subprocess
        self._kill()
        env = dict(os.environ)
        env["MG_KONTAKT_IDLE_SEC"] = str(IDLE_SEC)     # opt-in idle auto-unload (0 = off)
        proc = subprocess.Popen(
            [sys.executable, "-m", "backend.kontakt_daemon", plugin_path, state_path or ""],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1, env=env)
        # wait for READY (or a load error) within the timeout
        import time
        deadline = time.time() + load_timeout
        while time.time() < deadline:
            line = proc.stdout.readline()
            if line == "" and proc.poll() is not None:
                raise RuntimeError("Kontakt daemon exited during load")
            line = line.strip()
            if line.startswith(READY):
                self._proc, self._key = proc, (plugin_path, state_path)
                return
            if line.startswith(LOADERR):
                try:
                    proc.kill()
                except Exception:
                    pass
                err = json.loads(line[len(LOADERR):]).get("error", "unknown")
                raise RuntimeError(f"Kontakt daemon load failed: {err}")
        try:
            proc.kill()
        except Exception:
            pass
        raise RuntimeError("Kontakt daemon load timed out")

    def _kill(self):
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()           # EOF -> daemon exits on its own
            except Exception:
                pass
            try:
                self._proc.terminate()
            except Exception:
                pass
            self._proc = None
            self._key = None

    def render(self, notes, out_path, plugin_path, state_path, sr=44100, pb_range=2.0):
        with self._lock:
            if not self._alive() or self._key != (plugin_path, state_path):
                self._spawn(plugin_path, state_path)
            req = json.dumps({"notes": [list(n) for n in notes], "sr": int(sr),
                              "out": out_path, "pb_range": float(pb_range)})
            try:
                self._proc.stdin.write(req + "\n")
                self._proc.stdin.flush()
            except Exception:
                self._spawn(plugin_path, state_path)   # respawn once on a dead pipe
                self._proc.stdin.write(req + "\n")
                self._proc.stdin.flush()
            while True:
                line = self._proc.stdout.readline()
                if line == "" and self._proc.poll() is not None:
                    raise RuntimeError("Kontakt daemon died during render")
                line = line.strip()
                if line.startswith(RESP):
                    resp = json.loads(line[len(RESP):])
                    if not resp.get("ok"):
                        raise RuntimeError(f"Kontakt render failed: {resp.get('error')}")
                    return out_path


_CLIENT = _KontaktClient()
IDLE_SEC = 0          # idle auto-unload seconds (0 = off); set via set_idle(), applied on next spawn


def is_loaded():
    """Is the Kontakt daemon currently resident (holding Shreddage in RAM)?"""
    return _CLIENT._alive()


def daemon_pid():
    p = _CLIENT._proc
    return p.pid if (p is not None and p.poll() is None) else None


def set_idle(seconds):
    """Set the idle auto-unload timeout (0 = off). Applies to the next daemon spawn;
    if one is running it is restarted so the new timeout takes effect."""
    global IDLE_SEC
    IDLE_SEC = max(0, int(seconds or 0))
    if _CLIENT._alive():
        _CLIENT._kill()              # next render respawns with the new idle setting
    return IDLE_SEC


def render(notes, out_path, plugin_path, state_path, sr=44100, pb_range=None):
    """Render notes -> Kontakt/Shreddage DI WAV at out_path via the daemon. pb_range
    = the patch's pitch-bend range in semitones (for bend/slide articulations);
    defaults to 2 (S3 default) or $MG_SHREDDAGE_PB_RANGE."""
    import os
    if pb_range is None:
        try:
            pb_range = float(os.environ.get("MG_SHREDDAGE_PB_RANGE", "2"))
        except ValueError:
            pb_range = 2.0
    return _CLIENT.render(notes, out_path, plugin_path, state_path, sr=sr, pb_range=pb_range)


def shutdown():
    _CLIENT._kill()


import atexit
atexit.register(shutdown)


if __name__ == "__main__":
    main()
