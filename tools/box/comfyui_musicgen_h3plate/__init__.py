"""MusicGen H3 room plate (docs/MV_H3_REBUILD_PLAN.md s.20.60).

One node: write a decoded video (the moving EMPTY room plate) into the VIDEO stream of a MiniMax H3
AV latent, leaving the audio stream and any existing noise mask (the song lock's) as they are.
Pair it with ComfyUI-H3-Motion-Context-MultiRef's MiniMaxH3SetAVNoiseMask (video_mask: 1 = generate,
0 = keep the plate): H3's sampler re-injects the kept regions every step at the cond timestep
(comfy/model_base.py MiniMaxH3.scale_latent_inpaint), so the room is the plate by construction and
only the masked area (her, the gaps) is generated.
"""
import torch

import comfy.nested_tensor
import comfy.utils


def _streams(latent):
    samples = latent["samples"]
    parts = list(samples.unbind()) if hasattr(samples, "unbind") else list(samples)
    if len(parts) < 2 or parts[0].ndim != 5:
        raise ValueError("h3 room plate: expected a MiniMax H3 audio+video latent")
    return parts


def _pixel_frames(latent_t):
    # inverse of comfy_extras.nodes_minimax_h3.video_latent_t: 5 -> 2, then 17 frames per 5 latent steps
    return 5 if latent_t <= 2 else ((latent_t - 2) // 5) * 17 + 5


class MusicGenH3RoomPlateLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "latent": ("LATENT", {"tooltip": "H3 AV latent (after MiniMaxH3SongMaskedAVContext, so the song lock rides along)."}),
            "vae": ("VAE", {"tooltip": "MiniMax H3 video VAE."}),
            "plate": ("IMAGE", {"tooltip": "The moving room plate, at least as many frames as the latent covers; resized to the canvas."}),
        }}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = ("Write a decoded plate video into the video stream of an H3 AV latent (audio stream and noise mask kept). "
                   "Then set a video noise mask: 0 keeps the plate, 1 generates.")

    def apply(self, latent, vae, plate):
        video, audio = _streams(latent)[:2]
        _, _, t, h, w = video.shape
        n = _pixel_frames(int(t))
        if plate.shape[0] < n:
            raise ValueError(f"h3 room plate: the plate has {plate.shape[0]} frames, the latent needs {n}")
        px_h, px_w = h * 16, w * 16
        frames = plate[:n]
        if frames.shape[1] != px_h or frames.shape[2] != px_w:
            frames = comfy.utils.common_upscale(frames.movedim(-1, 1), px_w, px_h, "lanczos", "disabled").movedim(1, -1)
        enc = vae.encode(frames[..., :3])
        if tuple(enc.shape[2:]) != (t, h, w):
            raise ValueError(f"h3 room plate: encoded {tuple(enc.shape)} does not match the latent video {tuple(video.shape)}")
        out = latent.copy()
        out["samples"] = comfy.nested_tensor.NestedTensor((enc.to(device=video.device, dtype=video.dtype), audio))
        return (out,)


NODE_CLASS_MAPPINGS = {"MusicGenH3RoomPlateLatent": MusicGenH3RoomPlateLatent}
NODE_DISPLAY_NAME_MAPPINGS = {"MusicGenH3RoomPlateLatent": "MusicGen H3 Room Plate (video latent)"}
