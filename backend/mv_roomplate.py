"""Room-pinned camera move (opt-in, docs/MV_H3_REBUILD_PLAN.md s.20.60-20.61).

The scene renders again on its normal chain (same prompt with the move, references, song lock), but the
ROOM is pinned: an empty-room picture (the user's ChatGPT edit of the still take's first frame with the
people removed) is moved along the camera path by the Camera H3 depth warp and written into the video
latent; a mask (SAM3 "person" on the still take moved along the same path, held, widened, plus the
warp's edge gaps) marks where H3 generates. So the room is the plate by construction and H3 draws only
the people, lip-synced by the song lock as in every take. Measured on Never Yours scene 24 (s.20.60):
a real arc toward the lamp, the room unchanged, no visible seam; user: "Looks good".

Pure helpers here; app.py runs the box jobs (mv_roomplate_take)."""
import re
import subprocess

import numpy as np

from . import mv_meridian

EMPTY_ROOM_PROMPT = (
    "Edit this picture. Remove every person completely, so the place is empty. Keep everything else exactly "
    "as it is: the camera position, framing, lens, every piece of furniture, the walls, windows, signs and "
    "lamps, the light and the colours. Fill where the people were with what would be behind them, continuing "
    "the surfaces and textures naturally. No people, no new objects.")


def _camera_node(reference, trajectory, frames, frame_mode, pivot=0.0, moge="5"):
    return {"class_type": "BruxosH3Camera", "inputs": {
        "camera_trajectory": trajectory, "profile": "243 frames (~10.13s)", "interpolation": "smooth",
        "subject_framing": "medium shot", "minimax_format": "coordinate only", "reference_image": reference,
        "elevation_range": "+/-30", "orbit_direction": "invert H3 orbit", "subject_box": "",
        "frame_mode": frame_mode, "source_fps": 24.0, "ui_language": "English", "loop_closure": "auto",
        "depth_animation": "Depth Warp", "moge_geometry": [moge, 0], "warp_hfov": 50.0, "warp_depth_ratio": 6.0,
        "warp_invert_depth": False, "warp_smooth_depth": False, "warp_aim": "source aim",
        "warp_pivot_depth": float(pivot), "warp_direction": "panel (geometric)", "warp_length": int(frames),
        "warp_long_side": 0, "warp_offset_azimuth": 0.0, "warp_offset_elevation": 0.0,
        "warp_offset_distance": 1.0, "warp_hold_at": 0, "warp_hold_frames": 0,
        # full-resolution format: the source size, any length, magenta + mask where the move shows
        # what the source never saw (the Meridian format is 480-class)
        "warp_format": "CrossView (LTX)"}}


def _moge(image):
    return {"4": {"class_type": "LoadMoGeModel", "inputs": {"model_name": mv_meridian.MOGE_MODEL}},
            "5": {"class_type": "MoGeInference", "inputs": {"moge_model": ["4", 0], "image": image, "resolution_level": 3,
                                                            "fov_x_degrees": 0.0, "batch_size": 4, "force_projection": True,
                                                            "apply_mask": True, "refine_steps": 3}}}


def plate_warp_graph(empty_name, frames, move, amount, prefix):
    """The empty room (one picture) moved along the path; the camera node's info text (PreviewAny 21)
    reports the orbit centre it chose, which the people pass reuses."""
    g = {"20": {"class_type": "LoadImage", "inputs": {"image": empty_name}}}
    g.update(_moge(["20", 0]))
    g["6"] = _camera_node(["20", 0], mv_meridian.trajectory(move, amount), frames, "Freeze Frame")
    g["7"] = {"class_type": "CreateVideo", "inputs": {"images": ["6", 8], "fps": 24.0}}
    g["8"] = {"class_type": "SaveVideo", "inputs": {"video": ["7", 0], "filename_prefix": prefix, "format": "auto", "codec": "auto"}}
    g["21"] = {"class_type": "PreviewAny", "inputs": {"source": ["6", 3]}}
    return g


def people_warp_graph(take_name, frames, move, amount, pivot, prefix):
    """The still take moved along the same path around the same centre, and SAM3 tracking every
    person in that moved video (their mask in the moved view)."""
    g = {"1": {"class_type": "LoadVideo", "inputs": {"file": take_name}},
         "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
         "3": {"class_type": "ImageFromBatch", "inputs": {"image": ["2", 0], "batch_index": 0, "length": int(frames)}}}
    g.update(_moge(["3", 0]))
    g["6"] = _camera_node(["3", 0], mv_meridian.trajectory(move, amount), frames, "Motion Frame", pivot=pivot)
    g["7"] = {"class_type": "CreateVideo", "inputs": {"images": ["6", 8], "fps": 24.0}}
    g["8"] = {"class_type": "SaveVideo", "inputs": {"video": ["7", 0], "filename_prefix": prefix + "_take", "format": "auto", "codec": "auto"}}
    g["30"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sam3.1_multiplex_fp16.safetensors"}}
    g["31"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["30", 1], "text": "person"}}
    g["32"] = {"class_type": "SAM3_VideoTrack", "inputs": {"images": ["6", 8], "model": ["30", 0], "conditioning": ["31", 0],
                                                           "detection_threshold": 0.5, "max_objects": 4, "detect_interval": 1}}
    g["33"] = {"class_type": "SAM3_TrackToMask", "inputs": {"track_data": ["32", 0], "object_indices": ""}}
    g["34"] = {"class_type": "MaskToImage", "inputs": {"mask": ["33", 0]}}
    g["35"] = {"class_type": "CreateVideo", "inputs": {"images": ["34", 0], "fps": 24.0}}
    g["36"] = {"class_type": "SaveVideo", "inputs": {"video": ["35", 0], "filename_prefix": prefix + "_people", "format": "auto", "codec": "auto"}}
    return g


def pivot_from_info(text):
    """'pivot (0.00, 0.00, 1.28)' in the camera node's info -> 1.28 (the orbit centre's depth)."""
    m = re.search(r"pivot \(([-\d.]+), ([-\d.]+), ([-\d.]+)\)", text or "")
    if not m:
        raise ValueError("the camera node did not report its orbit centre")
    return float(m.group(3))


def _frames(path, w, h):
    p = subprocess.Popen(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path, "-vf", f"scale={w}:{h},format=bgr24",
                          "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    out = []
    while True:
        b = p.stdout.read(w * h * 3)
        if len(b) < w * h * 3:
            return np.stack(out)
        out.append(np.frombuffer(b, np.uint8).reshape(h, w, 3))


def _magenta(f):
    b, g, r = (f[..., i].astype(np.int16) for i in range(3))
    return ((r > 170) & (b > 170) & (g < 90)).astype(np.uint8)


def plate_and_mask(plate_warp, people_mask, out_plate, out_mask, frames, w=1344, h=768):
    """plate_warp (the moved empty room, magenta gaps) + people_mask (SAM3 on the moved take) ->
    out_plate (gaps filled neutrally: H3 regenerates them) and out_mask (white = H3 generates: the
    people held +-3 frames for H3's 4-frame latent steps and widened ~30 px, plus the gaps, feathered
    outward). Returns {frames, generated share mean/min/max}."""
    import cv2
    P = _frames(plate_warp, w, h)[:frames]
    M = _frames(people_mask, w, h)[:frames]
    n = min(len(P), len(M))
    if n < frames:
        raise ValueError(f"the moved room has {len(P)} frames and the people mask {len(M)}; {frames} needed")
    people = np.stack([(f[..., 1] > 127).astype(np.uint8) for f in M[:n]])
    held = np.stack([people[max(0, i - 3):i + 4].max(axis=0) for i in range(n)])

    def writer(path, gray):
        return subprocess.Popen(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt",
                                 "gray" if gray else "bgr24", "-s", f"{w}x{h}", "-r", "24", "-i", "-", "-c:v", "libx264",
                                 "-crf", "8", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    pw, mw = writer(out_plate, False), writer(out_mask, True)
    share = []
    for i in range(n):
        gaps = _magenta(P[i])
        m = cv2.dilate(held[i], np.ones((61, 61), np.uint8))
        m = np.maximum(m, cv2.dilate(gaps, np.ones((17, 17), np.uint8)))
        soft = np.maximum(cv2.GaussianBlur(m.astype(np.float32), (0, 0), 10), m.astype(np.float32))
        share.append(float(m.mean()))
        mw.stdin.write((soft.clip(0, 1) * 255).astype(np.uint8).tobytes())
        pw.stdin.write(cv2.inpaint(P[i], cv2.dilate(gaps, np.ones((5, 5), np.uint8)), 7, cv2.INPAINT_TELEA).tobytes())
    for wr in (pw, mw):
        wr.stdin.close()
        if wr.wait() != 0:
            raise RuntimeError("writing the plate or mask failed")
    return {"frames": n, "generated_mean": round(float(np.mean(share)), 3),
            "generated_min": round(float(np.min(share)), 3), "generated_max": round(float(np.max(share)), 3)}


def canvas_picture(src_path, dst_path, w=1344, h=768):
    """The user's empty-room picture at the canvas size (ChatGPT returns the frame's aspect at its own
    size; a different aspect is refused: the room would not line up)."""
    from PIL import Image
    im = Image.open(src_path).convert("RGB")
    a, b = im.size[0] / im.size[1], w / h
    if abs(a - b) / b > 0.02:
        raise ValueError(f"the empty-room picture is {im.size[0]}x{im.size[1]}; it must have the take's shape ({w}x{h})")
    im.resize((w, h), Image.LANCZOS).save(dst_path)
    return dst_path
