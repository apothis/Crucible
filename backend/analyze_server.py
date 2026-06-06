"""Box-side reference-analysis API (runs ON the Windows GPU box, in its own venv).

Crucible's P2 analyzer (RESEARCH.md §17): the Mac POSTs an audio file; this returns
high-quality musical analysis computed on the 3090:
  • allin1 (All-In-One Music Structure Analyzer) → BPM, beats, downbeats, and FUNCTIONAL
    section structure with labels (intro/verse/chorus/bridge/outro) — far better than the
    Mac's librosa energy-heuristic.
  • CLAP zero-shot → genre/mood/instrument style tags (scored against a label vocabulary).
  • librosa Krumhansl–Schmuckler → key/scale.

Self-contained: the launcher pins ALL caches/models (HF, torch hub, allin1 demix/spec)
inside the install dir, so nothing is written elsewhere on the box.

Endpoints: GET /health · POST /analyze (multipart `file`; optional form `labels` JSON,
`with_tags`, `with_key`). Models load lazily and stay resident (one box, one purpose).
"""
import json
import os
import tempfile

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

PORT = int(os.environ.get("MG_ANALYZE_PORT", "5075"))
DEVICE = os.environ.get("MG_ANALYZE_DEVICE", "cuda")
# Work dirs for allin1 byproducts (demixed stems / spectrograms) — kept inside the install
# dir via the launcher's MG_ANALYZE_WORK; cleaned per request.
WORK = os.environ.get("MG_ANALYZE_WORK", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".work"))
os.makedirs(WORK, exist_ok=True)

app = FastAPI()
_clap = None

# Pitch-class names matching the Mac app's KEYS (mixed sharps/flats) so the key is a valid
# Song-Builder option. Krumhansl–Schmuckler profiles.
PITCHES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

# Default style vocabulary for CLAP zero-shot (the Mac can override per request with its
# own genre registry). Kept metal/rock-leaning since that's Crucible's focus.
DEFAULT_LABELS = [
    "heavy metal", "power metal", "thrash metal", "death metal", "black metal", "doom metal",
    "symphonic metal", "progressive metal", "folk metal", "groove metal", "djent", "metalcore",
    "hard rock", "classic rock", "punk", "ballad", "orchestral", "electronic", "acoustic",
    "aggressive", "epic", "dark", "melodic", "fast", "slow", "heavy", "atmospheric",
    "distorted guitars", "clean guitars", "double-bass drums", "orchestral strings", "synths",
    "male vocals", "female vocals", "screamed vocals", "instrumental",
]


def _release_vram():
    """Free this service's VRAM after each analyze so it doesn't sit on the shared 3090
    between uses (mirrors SoulX/RoFormer). Set MG_ANALYZE_KEEP_RESIDENT=1 to keep CLAP
    loaded for faster repeat runs at the cost of held VRAM."""
    global _clap
    if os.environ.get("MG_ANALYZE_KEEP_RESIDENT", "") != "1":
        _clap = None
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _key_librosa(path):
    import librosa
    import numpy as np
    y, sr = librosa.load(path, sr=22050, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    if not np.isfinite(chroma).all() or chroma.sum() <= 0:
        return ""
    best = None
    for mode, prof in (("major", _MAJOR), ("minor", _MINOR)):
        p = np.array(prof)
        for i in range(12):
            c = np.corrcoef(chroma, np.roll(p, i))[0, 1]
            if np.isfinite(c) and (best is None or c > best[0]):
                best = (float(c), f"{PITCHES[i]} {mode}")
    return best[1] if best else ""


def _merge_labels(extra):
    """Union the curated DEFAULT_LABELS with caller-supplied labels (e.g. the Mac's genre
    registry), de-duped case-insensitively, defaults first. So tagging scores across BOTH
    the built-in metal/mood/instrument vocabulary AND the app's genres."""
    seen, merged = set(), []
    for x in list(DEFAULT_LABELS) + [str(v) for v in (extra or [])]:
        k = x.lower().strip()
        if k and k not in seen:
            seen.add(k); merged.append(x)
    return merged


def _ensure_clap():
    """Lazy-load the CLAP module once (shared by tagging + embedding)."""
    global _clap
    if _clap is None:
        import laion_clap
        _clap = laion_clap.CLAP_Module(enable_fusion=False)
        _clap.load_ckpt()                           # downloads into HF_HOME (pinned in-dir)
    return _clap


def _audio_embed(path):
    """L2-normalized CLAP audio embedding for one file (1D float list). This is the raw
    signal /analyze's tag-ranking throws away — exposed so the Mac can measure artist
    closeness via centroid cosine distance instead of lossy top-K tag presence."""
    import numpy as np
    clap = _ensure_clap()
    ae = clap.get_audio_embedding_from_filelist([path], use_tensor=False)
    ae = ae / (np.linalg.norm(ae, axis=1, keepdims=True) + 1e-9)
    return [float(x) for x in ae[0]]


def _clap_tags(path, labels, top=10, with_scores=False):
    try:
        import numpy as np
        clap = _ensure_clap()
        ae = clap.get_audio_embedding_from_filelist([path], use_tensor=False)
        te = clap.get_text_embedding(labels, use_tensor=False)
        ae = ae / (np.linalg.norm(ae, axis=1, keepdims=True) + 1e-9)
        te = te / (np.linalg.norm(te, axis=1, keepdims=True) + 1e-9)
        scores = (ae @ te.T)[0]
        order = list(np.argsort(scores)[::-1][:top])
        tags = [labels[i] for i in order]
        if with_scores:
            # keep the magnitude too — rank-only discards the only continuous signal
            return tags, {labels[i]: float(scores[i]) for i in order}
        return tags
    except Exception as e:
        return {"error": str(e)}


@app.get("/health")
def health():
    info = {"status": "ok", "service": "analyze", "clap_loaded": _clap is not None}
    try:
        try:
            import allin1fix  # noqa: F401   # the fork we actually use
        except Exception:
            import allin1     # noqa: F401   # legacy fallback
        info["allin1"] = True
    except Exception as e:
        info["allin1"] = False
        info["allin1_error"] = f"{type(e).__name__}: {e}"   # surface why import failed (natten/madmom/etc.)
    return info


@app.post("/embed")
async def embed(file: UploadFile = File(...)):
    """Return the L2-normalized CLAP audio embedding for one file. Enables the Mac's
    centroid-distance metric (cosine to the artist corpus mean) without re-uploading the
    whole corpus each time. Deterministic forward pass -> safe for test-retest validation."""
    tmp = tempfile.mkdtemp(dir=WORK)
    src = os.path.join(tmp, file.filename or "ref.wav")
    with open(src, "wb") as f:
        f.write(await file.read())
    try:
        vec = _audio_embed(src)
        return {"embedding": vec, "dim": len(vec)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"embed failed: {e}"})
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
        _release_vram()


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), labels: str = Form(""),
                  with_tags: str = Form("true"), with_key: str = Form("true"),
                  with_scores: str = Form("false")):
    tmp = tempfile.mkdtemp(dir=WORK)
    src = os.path.join(tmp, file.filename or "ref.wav")
    with open(src, "wb") as f:
        f.write(await file.read())
    out = {"bpm": 0, "beats": [], "downbeats": [], "segments": []}
    try:
        # allin1 (structure/beats) is OPTIONAL — needs madmom (a compiler on Windows). When
        # absent we still return tags + key, and the Mac uses its own (librosa) structure.
        try:
            try:
                import allin1fix as _allin1   # maintained fork (modern natten 0.17.5) — preferred
            except Exception:
                import allin1 as _allin1       # original (legacy natten 0.14.6)
            result = _allin1.analyze(
                src, device=DEVICE, keep_byproducts=False,
                demix_dir=os.path.join(WORK, "demix"), spec_dir=os.path.join(WORK, "spec"),
            )
            out["bpm"] = int(round(float(result.bpm))) if result.bpm else 0
            out["beats"] = [float(b) for b in (result.beats or [])]
            out["downbeats"] = [float(b) for b in (result.downbeats or [])]
            out["segments"] = [{"start": float(s.start), "end": float(s.end), "label": str(s.label)}
                               for s in (result.segments or [])]
        except Exception as e:
            out["structure_error"] = f"{type(e).__name__}: {e}"   # allin1 not installed → tags-only

        if str(with_key).lower() == "true":
            out["key"] = _key_librosa(src)
        if str(with_tags).lower() == "true":
            extra = []
            if labels:
                try:
                    parsed = json.loads(labels)
                    if isinstance(parsed, list):
                        extra = parsed
                except Exception:
                    pass
            if str(with_scores).lower() == "true":
                res = _clap_tags(src, _merge_labels(extra), with_scores=True)
                if isinstance(res, tuple):
                    out["tags"], out["tag_scores"] = res
                else:
                    out["tags"] = res            # {"error": ...}
            else:
                out["tags"] = _clap_tags(src, _merge_labels(extra))   # curated defaults ∪ caller's genres
        return out
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"analyze failed: {e}"})
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
        _release_vram()                                 # don't hog the shared 3090 between analyses


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
