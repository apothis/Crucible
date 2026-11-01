"""2D camera move on a finished take (opt-in, docs/MV_H3_REBUILD_PLAN.md s.20.58).

A moving crop of the take's own frames: nothing is redrawn, so her lips, the room and the song
stay exactly as rendered. No parallax (the background moves with her, not behind her). The user's
pick for singing scenes whose direct move rebuilt the room, after Meridian redrew the mouth late on
fast performance (scene 24, mouth match 0.17) and depth parallax tore the face (s.20.57). Best on a
still take (the scene's still version). Runs on the Mac (OpenCV + ffmpeg, no GPU, under a minute
for an 8 s take).

Each move eases in and out (cosine) over the whole take. `amount` is how much bigger than the frame
the picture is scaled (small 8 %, large 16 %): the margin the crop travels through, so the frame
never shows an edge."""
import re
import subprocess

import numpy as np

AMOUNTS = {"small": 0.08, "large": 0.16}
MOVES = ("push_in", "pull_out", "pan_left", "pan_right", "arc_left", "arc_right", "tilt_up", "tilt_down")
MOVE_LABELS = {"push_in": "push in", "pull_out": "pull out", "pan_left": "pan toward frame left",
               "pan_right": "pan toward frame right", "arc_left": "arc toward frame left (pan + slight push)",
               "arc_right": "arc toward frame right (pan + slight push)", "tilt_up": "tilt up", "tilt_down": "tilt down"}


def move_from_camera(camera):
    """Guess the 2D move from a scene's camera line (the writer's grammar). Returns (move or None,
    amount, guessed: True when the side had to be assumed). Trucks read as pans, pedestals as
    tilts (a crop cannot tell them apart)."""
    t = (camera or "").lower()
    amount = "large" if "large amplitude" in t else "small"
    if not t or t.strip() == "static":
        return None, amount, False
    side = "left" if re.search(r"\bleft\b", t) else "right" if re.search(r"\bright\b", t) else None
    if "push in" in t or "dolly in" in t:
        return "push_in", amount, False
    if "pull out" in t or "dolly out" in t:
        return "pull_out", amount, False
    if "arc" in t or "orbit" in t:
        return f"arc_{side or 'left'}", amount, side is None
    if "pan" in t or "truck" in t or "tracking" in t:
        return f"pan_{side or 'left'}", amount, side is None
    if "tilt" in t or "pedestal" in t:
        return ("tilt_down" if "down" in t else "tilt_up"), amount, False
    return None, amount, False


def transform(move, amount, s):
    """(zoom, dx, dy) at eased progress s in [0, 1]: zoom >= 1 scales the frame up about its centre,
    dx/dy shift the view in fractions of the frame (+ = toward frame right / bottom). Every value
    keeps the crop inside the scaled picture."""
    a = AMOUNTS[amount]
    if move == "push_in":
        return 1 + a * s, 0.0, 0.0
    if move == "pull_out":
        return 1 + a * (1 - s), 0.0, 0.0
    room = a / 2 / (1 + a)                     # how far the view can travel each way at zoom 1+a
    if move in ("pan_left", "pan_right"):
        sign = 1 if move == "pan_right" else -1
        return 1 + a, sign * room * (2 * s - 1), 0.0
    if move in ("arc_left", "arc_right"):
        # a pan with a slight push: starts at zoom 1+a/2, ends at 1+a, travelling across what the
        # zoom allows at each moment
        z = 1 + a / 2 + a / 2 * s
        r = (z - 1) / 2 / z
        sign = 1 if move == "arc_right" else -1
        return z, sign * r * (2 * s - 1), 0.0
    if move in ("tilt_up", "tilt_down"):
        sign = 1 if move == "tilt_down" else -1
        return 1 + a, 0.0, sign * room * (2 * s - 1)
    raise ValueError(f"move must be one of {', '.join(MOVES)}")


def render(src, dst, move, amount="small", crf=16):
    """The take at `src` -> the same take with the move at `dst` (same size, frame count and fps,
    the source audio copied). Returns {frames, width, height}."""
    import cv2
    if move not in MOVES or amount not in AMOUNTS:
        raise ValueError(f"move must be one of {', '.join(MOVES)}; amount small or large")
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                            "stream=width,height,nb_read_packets,r_frame_rate", "-of", "csv=p=0", src],
                           capture_output=True, text=True, timeout=60).stdout.strip().split(",")
    W, H, fps, n = int(probe[0]), int(probe[1]), probe[2], int(probe[3])
    dec = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-i", src, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                           stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", f"{W}x{H}", "-r", fps, "-i", "-", "-i", src, "-map", "0:v", "-map", "1:a?",
                            "-c:v", "libx264", "-crf", str(crf), "-preset", "slow", "-pix_fmt", "yuv420p",
                            "-c:a", "copy", "-movflags", "+faststart", "-shortest", dst], stdin=subprocess.PIPE)
    i = 0
    while True:
        b = dec.stdout.read(W * H * 3)
        if len(b) < W * H * 3:
            break
        f = np.frombuffer(b, np.uint8).reshape(H, W, 3)
        s = 0.5 - 0.5 * np.cos(np.pi * (i / max(1, n - 1)))
        z, dx, dy = transform(move, amount, s)
        # output pixel (u, v) samples the source at centre + ((u, v) - centre) / z + shift
        M = np.array([[z, 0, (1 - z) * W / 2 - dx * W * z], [0, z, (1 - z) * H / 2 - dy * H * z]], np.float32)
        out = cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
        enc.stdin.write(out.tobytes())
        i += 1
    enc.stdin.close()
    if enc.wait() != 0 or dec.wait() != 0:
        raise RuntimeError("the 2D move encode failed")
    return {"frames": i, "width": W, "height": H}
