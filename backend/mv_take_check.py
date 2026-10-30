"""Picture checks on a rendered take, run before a take is handed over (s.20.49).

The song-sync check only proves the audio lines up; it says nothing about the picture. This module
measures the one picture fault that is cheap to catch by numbers: a take that does not START as
itself. H3 sometimes opens on the location picture and cross-fades the singer in over the first
second or two (SSBS scene 6, 2026-10-30), or morphs out of a different first frame. Either way the
opening two seconds change far more than any later two seconds of the same shot.

opening_jump: mean abs difference (grey, 160x90) between frame 0 and the frame 2 s later, over the
median of the same 2 s difference across the rest of the take. A steady shot sits near 1; a fade-in
or morph opening scores several times its own baseline. Camera moves raise every window alike, so
the ratio stays fair for them. Measured on the 36 SSBS takes: the faded-in scene 6 = 5.79, every
other take 0.07-2.73 (the top one a real lean-in over the first 2 s). One positive example so far,
so the 4.0 line is a first calibration, not a proven boundary.

It also writes a contact sheet (the first two seconds at 4 fps, then 1 fps) so each take is LOOKED
at before it is shown: duplicate faces, malformed hands, stray objects and flicker do not show in a
number. A sheet finds defects; it never proves a take good (motion is judged by watching).

usage: .venv/bin/python backend/mv_take_check.py <out_dir> <label>=<clip.mp4> [<label>=<clip.mp4> ...]
"""
import json
import os
import subprocess
import sys

import numpy as np

W, H = 160, 90
FPS = 8
JUMP_FLAG = 4.0          # opening 2 s vs the take's own median 2 s
WINDOW = 2               # seconds


def frames(path, fps=FPS, w=W, h=H):
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-vf", f"fps={fps},scale={w}:{h},format=gray",
           "-f", "rawvideo", "pipe:1"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, h, w).astype(np.float32)


def opening_jump(path):
    f = frames(path)
    step = WINDOW * FPS
    if len(f) < step + 2 * FPS:
        return None, None
    d = np.array([np.abs(f[i + step] - f[i]).mean() for i in range(len(f) - step)])
    base = float(np.median(d[2 * FPS:])) if len(d) > 2 * FPS else float(np.median(d))
    first = float(d[0])
    return first / max(base, 0.5), first


def sheet(path, out_png):
    # first 2 s at 4 fps (8 frames), then 1 fps to the end
    a = f"select='lt(t,2)*not(mod(n,6))+gte(t,2)*not(mod(n,24))',scale=384:-1,tile=6x4:padding=4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", path, "-vf", a, "-vsync", "vfr",
                    "-frames:v", "1", out_png], check=True)


def main(out_dir, pairs):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for pair in pairs:
        label, path = pair.split("=", 1)
        ratio, first = opening_jump(path)
        png = os.path.join(out_dir, f"{label}.png")
        sheet(path, png)
        flag = ratio is not None and ratio >= JUMP_FLAG
        rows.append({"label": label, "clip": path, "opening_jump": None if ratio is None else round(ratio, 2),
                     "opening_diff": None if first is None else round(first, 2), "flag": flag, "sheet": png})
        print(f"{label:48s} opening_jump {ratio if ratio is None else round(ratio, 2)!s:>6}"
              f"{'  <-- opening changes ' + str(round(ratio, 1)) + 'x its own baseline (fade-in or morph?)' if flag else ''}")
    with open(os.path.join(out_dir, "take_check.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    return rows


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:])
