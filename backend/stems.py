"""Stem separation via Demucs — runs LOCALLY on the Mac (MPS GPU, in parallel
with the Windows 3090). Used to isolate a vocal from an ACE-Step generation
(the CREATE→ISOLATE step of the vocal pipeline) and for the restyle pipeline.

Needs no ffmpeg: `soundfile` (libsndfile) decodes mp3/wav for torchaudio.
"""
import glob
import os
import shutil
import subprocess
import sys

PYTHON = sys.executable  # the venv interpreter running the backend
MODEL = "htdemucs"
MODEL_6S = "htdemucs_6s"  # 6-stem split: adds a guitar (+ piano) stem


def separate(input_path: str, out_root: str, mode: str = "vocals", device: str = "mps"):
    """Separate `input_path` into stems under out_root. mode: 'vocals' (2-stem),
    'all' (4-stem), or '6stem'/'guitar' (6-stem incl. a guitar stem, via
    htdemucs_6s). Returns the flattened list of produced wav paths."""
    os.makedirs(out_root, exist_ok=True)
    model = MODEL_6S if mode in ("6stem", "guitar") else MODEL

    def run(dev):
        args = [PYTHON, "-m", "demucs", "-n", model, "-o", out_root, "-d", dev]
        if mode == "vocals":
            args += ["--two-stems", "vocals"]
        args += [input_path]
        return subprocess.run(args, capture_output=True, text=True)

    r = run(device)
    if r.returncode != 0 and device != "cpu":
        r = run("cpu")  # fall back if MPS chokes on an op
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout or "demucs failed")[-600:])

    stem = os.path.splitext(os.path.basename(input_path))[0]
    outdir = os.path.join(out_root, model, stem)
    files = sorted(glob.glob(os.path.join(outdir, "*.wav")))
    # flatten into out_root so they serve as out_root/<name>.wav
    flat = []
    for f in files:
        dst = os.path.join(out_root, os.path.basename(f))
        shutil.move(f, dst)
        flat.append(dst)
    shutil.rmtree(os.path.join(out_root, MODEL), ignore_errors=True)
    return flat
