"""Box-side LoRA dataset upload helper (runs ON the Windows GPU box, tiny, no GPU).

The ACE-Step engine's training pipeline reads dataset/tensor/output folders from
BOX-side paths, but Crucible + its library live on the Mac. This service is the bridge:
the Mac uploads the enriched dataset (audio + {name}.lyrics.txt + {name}.json) and this
writes it into a folder the engine can `/v1/dataset/scan`. It also hands back the
sibling tensor/output paths so the Mac can drive preprocess/train/export.

Self-contained: everything lives under MG_LORA_DIR (default ./lora_data beside this
script) → `{base}/{dataset}/data` (audio+labels), `{base}/{dataset}/tensors`,
`{base}/{dataset}/adapter`. Delete the folder = clean uninstall.

Endpoints (all JSON unless noted):
  GET  /health                         -> {ok, base_dir}
  POST /dataset/new      {name}        -> {data_dir, tensor_dir, adapter_dir} (created)
  POST /dataset/upload   multipart     -> writes files into {dataset}/data; {written, data_dir}
                         (form `name`, repeated `files`)
  POST /dataset/clear    {name}        -> empties {dataset}/data
  GET  /dataset/paths?name=            -> {data_dir, tensor_dir, adapter_dir, exists}
  GET  /dataset/list?name=             -> {files: [...]}
"""
import os
import re
import shutil

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

PORT = int(os.environ.get("MG_LORA_PORT", "5080"))
BASE = os.environ.get("MG_LORA_DIR",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "lora_data"))
os.makedirs(BASE, exist_ok=True)

app = FastAPI()


def _safe(name: str) -> str:
    """Sanitize a dataset name into a single safe folder segment (no traversal)."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip()).strip("._-")
    return s[:64] or "dataset"


def _paths(name: str):
    root = os.path.join(BASE, _safe(name))
    return {
        "data_dir": os.path.join(root, "data"),
        "tensor_dir": os.path.join(root, "tensors"),
        "adapter_dir": os.path.join(root, "adapter"),
    }


@app.get("/health")
def health():
    return {"ok": True, "base_dir": BASE}


@app.post("/dataset/new")
def dataset_new(name: str = Form(...)):
    p = _paths(name)
    for d in p.values():
        os.makedirs(d, exist_ok=True)
    return p


@app.post("/dataset/upload")
async def dataset_upload(name: str = Form(...), files: list[UploadFile] = File(...)):
    p = _paths(name)
    os.makedirs(p["data_dir"], exist_ok=True)
    written = []
    for f in files:
        fn = os.path.basename(f.filename or "").strip()
        if not fn:
            continue
        dest = os.path.join(p["data_dir"], fn)
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        written.append(fn)
    return {"written": written, "count": len(written), "data_dir": p["data_dir"]}


@app.post("/dataset/clear")
def dataset_clear(name: str = Form(...)):
    p = _paths(name)
    if os.path.isdir(p["data_dir"]):
        shutil.rmtree(p["data_dir"], ignore_errors=True)
    os.makedirs(p["data_dir"], exist_ok=True)
    return {"ok": True, "data_dir": p["data_dir"]}


@app.get("/dataset/paths")
def dataset_paths(name: str):
    p = _paths(name)
    p["exists"] = os.path.isdir(p["data_dir"])
    return p


@app.get("/dataset/list")
def dataset_list(name: str):
    p = _paths(name)
    if not os.path.isdir(p["data_dir"]):
        return JSONResponse({"files": []})
    return {"files": sorted(os.listdir(p["data_dir"]))}


if __name__ == "__main__":
    import uvicorn
    print(f"[lora-upload] base dir: {BASE}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
