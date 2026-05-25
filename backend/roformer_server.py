"""BS-Roformer separation API — runs ON the Windows GPU box (its own venv, created
by ROFORMER-API_AUTO_INSTALL.bat). Exposes the contract the Mac's
backend/roformer_py.py expects:

  GET  /health    -> {"ok": true, "model": <name>, "stems": [...]}
  POST /separate  (multipart) fields:
        audio  : the mix to separate (WAV/MP3/FLAC)
        stems  : optional comma-separated subset to return (e.g. "guitar,bass");
                 "all" / omitted = every stem the model produces
     -> JSON {"sr": <int>, "stems": {"<name>": "<base64 wav>", ...}}

Design for SHARED GPU: we run the proven `bs-roformer-infer` CLI as a subprocess
per request, so the model loads, separates, then the process EXITS and frees all
VRAM. The Mac frees ComfyUI's VRAM (POST /free) before calling us, so the 3090
isn't holding the ACE-Step models at the same time.

Default model = BS-Roformer SW (6 stems: vocals/bass/drums/guitar/piano/other).
Config + checkpoint paths come from env (set by the launcher):
  MG_ROFORMER_CONFIG, MG_ROFORMER_CKPT, MG_ROFORMER_DEVICE (default cuda),
  MG_ROFORMER_PORT (default 5070).
"""
import base64
import glob
import os
import shutil
import subprocess
import sys
import tempfile

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn

CONFIG = os.environ.get("MG_ROFORMER_CONFIG", "")
CKPT = os.environ.get("MG_ROFORMER_CKPT", "")
DEVICE = os.environ.get("MG_ROFORMER_DEVICE", "cuda")
PORT = int(os.environ.get("MG_ROFORMER_PORT", "5070"))

app = FastAPI()


def _cli():
    """Locate the bs-roformer-infer console script in this venv."""
    exe = shutil.which("bs-roformer-infer")
    if exe:
        return exe
    d = os.path.dirname(sys.executable)
    for c in ("bs-roformer-infer.exe", "bs-roformer-infer"):
        p = os.path.join(d, c)
        if os.path.exists(p):
            return p
    for c in ("Scripts/bs-roformer-infer.exe", "bin/bs-roformer-infer"):
        p = os.path.join(d, c)
        if os.path.exists(p):
            return p
    return "bs-roformer-infer"


def _stem_name(path):
    """Derive a stem name from an MSST output filename like 'song_guitar.wav'."""
    base = os.path.splitext(os.path.basename(path))[0].lower()
    for s in ("vocals", "bass", "drums", "guitar", "piano", "keyboards",
              "other", "instrumental"):
        if base.endswith("_" + s) or base == s:
            return "keyboard" if s == "keyboards" else s
    return base.rsplit("_", 1)[-1]


@app.get("/health")
def health():
    return {"ok": bool(CONFIG and CKPT and os.path.exists(CKPT)),
            "model": os.path.basename(CKPT) or "?", "device": DEVICE}


@app.post("/separate")
async def separate(audio: UploadFile = File(...), stems: str = Form("all")):
    if not (CONFIG and CKPT):
        return JSONResponse({"error": "model not configured (MG_ROFORMER_CONFIG/CKPT)"}, status_code=500)
    work = tempfile.mkdtemp(prefix="rofo_")
    in_dir = os.path.join(work, "in"); out_dir = os.path.join(work, "out")
    os.makedirs(in_dir); os.makedirs(out_dir)
    ext = os.path.splitext(audio.filename or "in.wav")[1] or ".wav"
    inp = os.path.join(in_dir, "mix" + ext)
    with open(inp, "wb") as f:
        f.write(await audio.read())
    try:
        cmd = [_cli(), "--config_path", CONFIG, "--model_path", CKPT,
               "--input_folder", in_dir, "--store_dir", out_dir, "--device", DEVICE]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return JSONResponse({"error": "inference failed", "stderr": r.stderr[-2000:]}, status_code=500)
        want = None if stems.strip() in ("", "all") else {s.strip().lower() for s in stems.split(",")}
        import soundfile as sf
        sr = None
        outs = {}
        for wav in sorted(glob.glob(os.path.join(out_dir, "**", "*.wav"), recursive=True)):
            name = _stem_name(wav)
            if want is not None and name not in want:
                continue
            with open(wav, "rb") as fh:
                data = fh.read()
            if sr is None:
                sr = sf.info(wav).samplerate
            outs[name] = base64.b64encode(data).decode("ascii")
        if not outs:
            return JSONResponse({"error": "no stems produced", "stderr": r.stderr[-1000:]}, status_code=500)
        return {"sr": sr or 44100, "stems": outs}
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    print(f"[roformer] model={os.path.basename(CKPT)} device={DEVICE} port={PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
