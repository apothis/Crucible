"""MusicGen RVC API server — runs INSIDE the bundled RVC WebUI package.

Launch with the package's own interpreter from the package root, e.g.:
    cd <RVC package dir>
    set MG_PORT=5050
    runtime\\python.exe rvc_server.py

It reuses the package's already-working fairseq/torch env (no installs) and
Gradio's bundled fastapi+uvicorn. It exposes the SAME REST API as `rvc-python`
so the Mac backend (backend/rvc_py.py) and the voice installer work unchanged:

    GET  /models                 -> {"models": [names]}
    POST /models/{name}          -> load a voice
    POST /params  {"params":{}}  -> set conversion params
    POST /convert {"audio_data"} -> base64 wav in, WAV bytes out
    POST /upload_model (zip)     -> install a voice (.pth -> weights, .index -> index)
"""
import os
import sys
import io
import base64
import tempfile
import zipfile
import shutil

# RVC's Config parses sys.argv — keep it clean so our launch args don't break it.
sys.argv = [sys.argv[0]]

# RVC's bundled runtime is an EMBEDDABLE Python that does NOT auto-add the
# package root to sys.path (RVC's own infer-web.py appends cwd to compensate).
# Do the same so `configs` / `infer` import. Launcher cd's to the package root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

os.environ.setdefault("weight_root", "assets/weights")
os.environ.setdefault("index_root", "logs")
os.environ.setdefault("outside_index_root", "assets/indices")
os.environ.setdefault("rmvpe_root", "assets/rmvpe")
os.environ.setdefault("weight_uvr5_root", "assets/uvr5_weights")

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, Response
import uvicorn

from configs.config import Config
from infer.modules.vc.modules import VC

config = Config()
vc = VC(config)

WEIGHT_ROOT = os.environ["weight_root"]
INDEX_ROOTS = [os.environ.get("index_root", "logs"), os.environ.get("outside_index_root", "assets/indices")]

STATE = {"model": None, "index": "",
         "params": {"f0method": "rmvpe", "f0up_key": 0, "index_rate": 0.75,
                    "filter_radius": 3, "resample_sr": 0, "rms_mix_rate": 0.25, "protect": 0.33}}


def list_pths():
    if not os.path.isdir(WEIGHT_ROOT):
        return []
    return sorted(f[:-4] for f in os.listdir(WEIGHT_ROOT) if f.lower().endswith(".pth"))


def find_index(stem):
    for root in INDEX_ROOTS:
        if root and os.path.isdir(root):
            for r, _, files in os.walk(root):
                for f in files:
                    if f.lower().endswith(".index") and stem.lower() in f.lower():
                        return os.path.join(r, f)
    return ""


def wav_bytes(audio, sr):
    buf = io.BytesIO()
    try:
        import soundfile as sf
        sf.write(buf, audio, sr, format="WAV")
    except Exception:
        from scipy.io import wavfile
        import numpy as np
        a = audio
        if a.dtype.kind == "f":
            a = (a * 32767).clip(-32768, 32767).astype("int16")
        elif a.dtype != np.int16:
            a = a.astype("int16")
        wavfile.write(buf, sr, a)
    return buf.getvalue()


app = FastAPI(title="MusicGen RVC API")


@app.get("/models")
def models():
    return JSONResponse({"models": list_pths()})


@app.post("/models/{name}")
def load_model(name: str):
    pth = name if name.lower().endswith(".pth") else name + ".pth"
    p = STATE["params"]["protect"]
    vc.get_vc(pth, p, p)
    STATE["model"] = name
    STATE["index"] = find_index(os.path.splitext(os.path.basename(name))[0])
    return JSONResponse({"message": f"loaded {name}", "index": STATE["index"]})


@app.post("/params")
def set_params(body: dict):
    STATE["params"].update(body.get("params", {}))
    return JSONResponse({"message": "ok", "params": STATE["params"]})


@app.post("/convert")
def convert(body: dict):
    if not STATE["model"]:
        return JSONResponse({"error": "no model loaded"}, status_code=400)
    data = base64.b64decode(body["audio_data"])
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(data)
    tmp.close()
    p = STATE["params"]
    try:
        info, out = vc.vc_single(
            0, tmp.name, int(p["f0up_key"]), None, p["f0method"],
            STATE["index"], "", float(p["index_rate"]), int(p["filter_radius"]),
            int(p["resample_sr"]), float(p["rms_mix_rate"]), float(p["protect"]))
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
    if not out or out[0] is None:
        return JSONResponse({"error": info}, status_code=500)
    tgt_sr, audio = out
    return Response(content=wav_bytes(audio, tgt_sr), media_type="audio/wav")


@app.post("/upload_model")
async def upload_model(file: UploadFile = File(...)):
    data = await file.read()
    tmpd = tempfile.mkdtemp()
    try:
        zpath = os.path.join(tmpd, "m.zip")
        with open(zpath, "wb") as f:
            f.write(data)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(tmpd)
        os.makedirs(WEIGHT_ROOT, exist_ok=True)
        os.makedirs(INDEX_ROOTS[0], exist_ok=True)
        installed = []
        for r, _, files in os.walk(tmpd):
            for f in files:
                src = os.path.join(r, f)
                if f.lower().endswith(".pth"):
                    shutil.copy(src, os.path.join(WEIGHT_ROOT, f))
                    installed.append(f[:-4])
                elif f.lower().endswith(".index"):
                    shutil.copy(src, os.path.join(INDEX_ROOTS[0], f))
        return JSONResponse({"message": f"installed {installed}"})
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)


if __name__ == "__main__":
    port = int(os.environ.get("MG_PORT", "5050"))
    print(f"MusicGen RVC API on 0.0.0.0:{port}  (weights: {WEIGHT_ROOT})")
    uvicorn.run(app, host="0.0.0.0", port=port)
