"""RVC (Retrieval-based Voice Conversion) client.

RVC's WebUI is Gradio 3.14 on the Windows box. Its `/run/<api_name>` endpoints
respond synchronously (no queue websocket needed). The conversion input is a
SERVER-SIDE file path (a textbox), and RVC shares the filesystem with ComfyUI,
so we upload the audio via ComfyUI's /upload/image and hand RVC the absolute
path inside ComfyUI's input dir. Output is fetched via Gradio's /file= route.
"""
import os
import posixpath
import requests


class RVC:
    def __init__(self, host: str, comfy, comfy_input_dir: str):
        self.base = f"http://{host}"
        self.comfy = comfy
        self.input_dir = comfy_input_dir.rstrip("/\\")

    def reachable(self) -> bool:
        try:
            requests.get(f"{self.base}/config", timeout=4)
            return True
        except Exception:
            return False

    def _run(self, api_name: str, data: list, timeout=180):
        r = requests.post(f"{self.base}/run/{api_name}", json={"data": data}, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def voices(self):
        """Return available voice models + feature indexes."""
        try:
            j = self._run("infer_refresh", [], timeout=15)
            d = j.get("data", [])
            voices = d[0].get("choices", []) if len(d) > 0 else []
            indexes = d[1].get("choices", []) if len(d) > 1 else []
            return {"available": True, "voices": voices, "indexes": indexes}
        except Exception as e:
            return {"available": False, "voices": [], "indexes": [], "error": str(e)}

    def _match_index(self, voice: str, indexes: list) -> str:
        stem = os.path.splitext(os.path.basename(voice))[0].lower()
        for ix in indexes:
            if os.path.splitext(os.path.basename(ix))[0].lower().startswith(stem):
                return ix
        return ""

    def convert(self, audio_bytes: bytes, filename: str, voice: str, *,
                transpose=0, f0_method="rmvpe", index_rate=0.75,
                filter_radius=3, rms_mix_rate=0.25, protect=0.33) -> bytes:
        # 1. place the file on the Windows box via ComfyUI's input dir
        ref = self.comfy.upload_audio(audio_bytes, filename)  # "name" or "sub/name"
        abspath = posixpath.join(self.input_dir, ref)

        # 2. load the voice model
        self._run("infer_change_voice", [voice, protect, protect], timeout=60)

        # 3. pick a matching feature index and convert
        index = self._match_index(voice, self.voices().get("indexes", []))
        data = [0, abspath, transpose, None, f0_method, "", index,
                index_rate, filter_radius, 0, rms_mix_rate, protect]
        j = self._run("infer_convert", data, timeout=300)
        out = j["data"][1]
        info = j["data"][0]
        if not out or not out.get("name"):
            raise RuntimeError(f"RVC conversion produced no audio. Info: {info}")

        # 4. fetch the output file from Gradio
        r = requests.get(f"{self.base}/file={out['name']}", timeout=60)
        r.raise_for_status()
        return r.content
