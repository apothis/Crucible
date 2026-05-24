"""rvc-python API client (clean replacement for the Gradio WebUI driver).

Talks to an `rvc-python` API server (`python -m rvc_python api -p 5050 -l`)
running on the Windows box. Accepts uploaded audio directly as base64 — no
Gradio, no ComfyUI-input-dir file-smuggling.

API: GET /models, POST /models/{name} (load), POST /params, POST /convert.
"""
import base64
import requests


class RVCPython:
    def __init__(self, host: str):
        self.base = f"http://{host}" if host else ""

    def reachable(self) -> bool:
        if not self.base:
            return False
        try:
            requests.get(f"{self.base}/models", timeout=4)
            return True
        except Exception:
            return False

    def voices(self):
        try:
            r = requests.get(f"{self.base}/models", timeout=10)
            r.raise_for_status()
            names = r.json()
            if isinstance(names, dict):  # tolerate {"models": [...]}
                names = names.get("models", [])
            return {"available": True, "voices": list(names), "indexes": []}
        except Exception as e:
            return {"available": False, "voices": [], "indexes": [], "error": str(e)}

    def convert(self, audio_bytes: bytes, filename: str, voice: str, *,
                transpose=0, f0_method="rmvpe", index_rate=0.75,
                filter_radius=3, rms_mix_rate=0.25, protect=0.33) -> bytes:
        # 1. load the requested voice model
        lr = requests.post(f"{self.base}/models/{voice}", timeout=120)
        lr.raise_for_status()
        # 2. set conversion parameters
        params = {"f0method": f0_method, "f0up_key": int(transpose),
                  "index_rate": index_rate, "filter_radius": int(filter_radius),
                  "resample_sr": 0, "rms_mix_rate": rms_mix_rate, "protect": protect}
        requests.post(f"{self.base}/params", json={"params": params}, timeout=30)
        # 3. convert (base64 in, WAV bytes out)
        b64 = base64.b64encode(audio_bytes).decode()
        cr = requests.post(f"{self.base}/convert", json={"audio_data": b64}, timeout=300)
        cr.raise_for_status()
        return cr.content
