"""Meridian camera re-shoot (opt-in, docs/MV_H3_REBUILD_PLAN.md s.20.55-20.56).

Viggle Meridian re-cameras an EXISTING clip: MoGe reads the clip's depth, the Camera H3 node
(NyckM/3d-Camera-control-H3-Minimax) reprojects it along a new camera path into a guide video
(grey = never seen), and the Meridian H3 fine-tune redraws the clip along that path. The room is
the one already rendered, so a move cannot rebuild it. Use it on a STILL take (a still version of
a scene whose direct move fell apart): a take that already moves keeps its warping.

The graph mirrors the node author's v3 Quality workflow (Workflow/Minimax_bruxosdovfx_CameraMeridian_
v3_Quality.json) with three measured changes on Never Yours scene 10:
- the song window rides in the latent through MiniMaxH3SongMaskedAVContext (lips drifted ~3 s in
  without it; with it the user called them good enough),
- the guide is warped at 832 px (warp_long_side; at full width the 2 px splats left dotted holes
  that came back as white specks),
- no extra reference picture (it did not hold the redrawn neon and added specks).
Only the Meridian video is saved; the caller trims it to the source length and puts the source
take's audio back (on_complete_media, `audio_from` + `trim_frames`)."""
import json
import re

MERIDIAN_LENGTHS = (73, 90, 107, 124, 141, 158, 175, 243)   # the frozen text embeddings that exist
MERIDIAN_UNET = "H3Camera\\MeridianH3Camera_int8_pruned.safetensors"
MERIDIAN_LORAS = ("meridian_dmd_lora_comfyui.safetensors",
                  "minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors")
MERIDIAN_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"   # the author's
MERIDIAN_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
MERIDIAN_SIGMAS = "1.0, 0.8571428571428571, 0.6, 0.0"
MOGE_MODEL = "moge_2_vitl_normal_fp16.safetensors"
WARP_LONG_SIDE = 832
CONTEXT = 39

# move -> the end keyframe of the path (start is the source camera). Signs: azimuth + swings the
# view toward frame LEFT [M: scene 10, +10 opened the window at frame left; the author's -3.5 is
# compiled as "toward the camera's RIGHT"]. distance < 1 = closer. lateral and height signs are
# unmeasured [H]. Amounts: small = a slow performance move, large = B-roll scale.
MOVES = {
    "arc_left":      {"small": {"azimuth": 10}, "large": {"azimuth": 20}},
    "arc_right":     {"small": {"azimuth": -10}, "large": {"azimuth": -20}},
    "push_in":       {"small": {"distance": 0.85}, "large": {"distance": 0.7}},
    "pull_out":      {"small": {"distance": 1.15}, "large": {"distance": 1.3}},
    "truck_left":    {"small": {"lateral": -0.08}, "large": {"lateral": -0.16}},
    "truck_right":   {"small": {"lateral": 0.08}, "large": {"lateral": 0.16}},
    "pedestal_up":   {"small": {"height": 0.06}, "large": {"height": 0.12}},
    "pedestal_down": {"small": {"height": -0.06}, "large": {"height": -0.12}},
}
MOVE_LABELS = {"arc_left": "arc toward frame left", "arc_right": "arc toward frame right",
               "push_in": "push in", "pull_out": "pull out", "truck_left": "truck left",
               "truck_right": "truck right", "pedestal_up": "pedestal up", "pedestal_down": "pedestal down"}


def length_for(frames):
    """The Meridian length to render a source of `frames` at: the shortest embedding that covers
    it (the node holds the source's last frame for the rest, and the caller trims it off)."""
    for n in MERIDIAN_LENGTHS:
        if n >= int(frames):
            return n
    raise ValueError(f"a {frames}-frame take is longer than Meridian's longest length ({MERIDIAN_LENGTHS[-1]} frames, "
                     f"{MERIDIAN_LENGTHS[-1] / 24:.1f} s)")


def move_from_camera(camera):
    """Guess the re-shoot move from a scene's camera line (the writer's grammar). Returns
    (move or None, amount, guessed: True when the direction had to be assumed)."""
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
    if "truck" in t:
        return f"truck_{side or 'left'}", amount, side is None
    if "pedestal" in t:
        return ("pedestal_down" if "down" in t else "pedestal_up"), amount, False
    return None, amount, False


def trajectory(move, amount="small"):
    if move not in MOVES:
        raise ValueError(f"move must be one of {', '.join(MOVES)}")
    if amount not in ("small", "large"):
        raise ValueError("amount must be small or large")
    end = {"time": 1, "azimuth": 0, "elevation": 0, "distance": 1, "height": 0, "lateral": 0}
    end.update(MOVES[move][amount])
    start = {"time": 0, "azimuth": 0, "elevation": 0, "distance": 1, "height": 0, "lateral": 0}
    return json.dumps([start, end])


def build_meridian_graph(source_name, song_name, source_frames, move, amount="small", seed=0,
                         prefix="videogen/meridian"):
    """source_name: the take uploaded to ComfyUI's input folder (24 fps mp4). song_name: the song
    window (stereo WAV, starting at the scene's song start, length/24 + 1 s). Returns (graph,
    resolved)."""
    n = int(source_frames)
    length = length_for(n)
    profile = "124 frames (~5.17s)" if length <= 124 else "243 frames (~10.13s)"
    g = {
        "1": {"class_type": "LoadVideo", "inputs": {"file": source_name}},
        "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
        "3": {"class_type": "ImageFromBatch", "inputs": {"image": ["2", 0], "batch_index": 0, "length": n}},
        "4": {"class_type": "LoadMoGeModel", "inputs": {"model_name": MOGE_MODEL}},
        "5": {"class_type": "MoGeInference", "inputs": {"moge_model": ["4", 0], "image": ["3", 0], "resolution_level": 3,
                                                        "fov_x_degrees": 0.0, "batch_size": 4, "force_projection": True,
                                                        "apply_mask": True, "refine_steps": 3}},
        "6": {"class_type": "BruxosH3Camera", "inputs": {
            "camera_trajectory": trajectory(move, amount), "profile": profile, "interpolation": "smooth",
            "subject_framing": "medium shot", "minimax_format": "coordinate only", "reference_image": ["3", 0],
            "elevation_range": "+/-30", "orbit_direction": "invert H3 orbit", "subject_box": "",
            "frame_mode": "Motion Frame", "source_fps": 24.0, "ui_language": "English", "loop_closure": "auto",
            "depth_animation": "Depth Warp", "moge_geometry": ["5", 0], "warp_hfov": 50.0, "warp_depth_ratio": 6.0,
            "warp_invert_depth": False, "warp_smooth_depth": False, "warp_aim": "source aim", "warp_pivot_depth": 0.0,
            "warp_direction": "panel (geometric)", "warp_length": length, "warp_long_side": WARP_LONG_SIDE,
            "warp_offset_azimuth": 0.0, "warp_offset_elevation": 0.0, "warp_offset_distance": 1.0,
            "warp_hold_at": 0, "warp_hold_frames": 0, "warp_format": "Meridian (H3)"}},
        "10": {"class_type": "BruxosMeridianTextCond", "inputs": {"text_cond": f"meridian_fixed_embed_{length}.safetensors"}},
        "11": {"class_type": "VAELoader", "inputs": {"vae_name": MERIDIAN_VAE}},
        "12": {"class_type": "BruxosMeridianReference", "inputs": {"source_video": ["3", 0], "warp_video": ["6", 8],
                                                                   "vae": ["11", 0], "length": length, "text_cond": ["10", 0],
                                                                   "width": 0, "height": 0}},
        "13": {"class_type": "UNETLoader", "inputs": {"unet_name": MERIDIAN_UNET, "weight_dtype": "default"}},
        "14": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["13", 0], "lora_name": MERIDIAN_LORAS[0], "strength_model": 1.0}},
        "15": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["14", 0], "lora_name": MERIDIAN_LORAS[1], "strength_model": 1.0}},
        "16": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["15", 0], "attention": "comfy kitchen attention"}},
        "17": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["16", 0], "attention": "comfy kitchen attention"}},
        "18": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["17", 0], "shift_video": 3.0, "shift_audio": 3.0}},
        "19": {"class_type": "BasicGuider", "inputs": {"model": ["18", 0], "conditioning": ["12", 0]}},
        "20": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
        "21": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "22": {"class_type": "ManualSigmas", "inputs": {"sigmas": MERIDIAN_SIGMAS}},
        "23": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["20", 0], "guider": ["19", 0], "sampler": ["21", 0],
                                                                 "sigmas": ["22", 0], "latent_image": ["43", 0]}},
        "24": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["23", 0], "vae": ["11", 0], "tile_size": 512,
                                                          "overlap": 64, "temporal_size": 252, "temporal_overlap": 8}},
        "25": {"class_type": "CreateVideo", "inputs": {"images": ["24", 0], "fps": 24.0, "bit_depth": 8,
                                                       "color_space": "sRGB", "codec": "none"}},
        "26": {"class_type": "SaveVideo", "inputs": {"video": ["25", 0], "filename_prefix": prefix, "format": "auto", "codec": "auto"}},
        "40": {"class_type": "LoadAudio", "inputs": {"audio": song_name}},
        "41": {"class_type": "VAELoader", "inputs": {"vae_name": MERIDIAN_AUDIO_VAE}},
        "42": {"class_type": "MiniMaxH3AudioVAECompatibility", "inputs": {"audio_vae": ["41", 0]}},
        "43": {"class_type": "MiniMaxH3SongMaskedAVContext", "inputs": {"latent": ["12", 1], "audio_vae": ["42", 0],
                                                                        "master_audio": ["40", 0], "clip_start_seconds": 0.0,
                                                                        "context_length": CONTEXT, "source_fps": 24.0,
                                                                        "crop": "disabled"}},
    }
    resolved = {"engine": "meridian", "kind": "video", "lane": "meridian", "seed": int(seed),
                "frames": n, "meridian_length": length, "move": move, "amount": amount,
                "trajectory": trajectory(move, amount), "warp_long_side": WARP_LONG_SIDE, "song_lock": True,
                "model": MERIDIAN_UNET}
    return g, resolved
