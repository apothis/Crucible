"""MusicGen's YuE2 helper nodes for ComfyUI (deployed to custom_nodes/ComfyUI-MusicGen-YuE2 by the app).

YuE2 Generate Music (score-length guard)
    Same as core "YuE2 Generate Music", plus `min_seconds`: the music stage may not emit its
    end-of-music token before that many seconds. Uses core's own mechanism (the sampler already
    blocks the end token while step < min_tokens; core sets min_tokens=200) - this only raises that
    floor for the semantic phase of this one call. Why: band LoRAs trained with a capped planner
    window (ar_max_tokens 4000) were seen ending songs at exactly 300 s, mid-chorus, on a 328 s
    score (docs/YUE2_TRAINING_PLAN.md). The app sets min_seconds = the supplied score's length - 2 s.
    Cost: core only switches on CUDA-graph replay once step >= min_tokens, so the guarded stretch
    samples slower.

No core file is edited: the core text encoder's _generate is wrapped once at import, and the wrapper
does nothing unless this node set a floor for the current call.
"""
import contextvars
import logging

import comfy.text_encoders.yue2 as _te
from comfy.text_encoders.yue2 import FRAMES_PER_SECOND

_FLOOR = contextvars.ContextVar("musicgen_yue2_min_music_tokens", default=0)

if not getattr(_te.YuE2TEModel._generate, "_musicgen_guard", False):
    _core_generate = _te.YuE2TEModel._generate

    def _generate(self, prefix, seed, max_tokens, phase, dtype, **kwargs):
        floor = _FLOOR.get()
        if phase == "semantic" and floor > 0 and "min_tokens" in kwargs:
            raised = max(kwargs["min_tokens"], min(floor, max_tokens - 1))
            if raised != kwargs["min_tokens"]:
                logging.info("YuE2 score-length guard: end token blocked for the first %d music tokens (%.1f s)",
                             raised, raised / FRAMES_PER_SECOND)
            kwargs["min_tokens"] = raised
        return _core_generate(self, prefix, seed, max_tokens, phase, dtype, **kwargs)

    _generate._musicgen_guard = True
    _te.YuE2TEModel._generate = _generate


class YuE2GenerateMusicGuarded:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "style": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "lyrics": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "abc": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "mode": (["full", "melody"],),
                "max_duration": ("FLOAT", {"default": 360.0, "min": 0.04, "max": 900.0, "step": 0.04}),
                "min_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 900.0, "step": 0.04,
                                          "tooltip": "no end-of-music before this many seconds (0 = core behaviour)"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.01, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 100, "min": 1, "max": 32768}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 0.01, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "FLOAT")
    RETURN_NAMES = ("conditioning", "seconds")
    FUNCTION = "run"
    CATEGORY = "model/conditioning/yue2"

    def run(self, clip, style, lyrics, abc, seed, mode, max_duration, min_seconds, temperature, top_p,
            top_k, repetition_penalty, cfg_scale=None):
        # body mirrors core YuE2GenerateMusic.execute (comfy_extras/nodes_yue2.py, ComfyUI 0.41)
        if not abc.strip():
            mode = "off"
        if cfg_scale is None:
            cfg_scale = 1.01 if mode == "off" else 1.0
        token = _FLOOR.set(max(0, round(min_seconds * FRAMES_PER_SECOND)))
        try:
            tokens = clip.tokenize(style, lyrics=lyrics, cot=mode, seed=seed, abc=abc,
                                   max_tokens=max(1, round(max_duration * FRAMES_PER_SECOND)),
                                   temperature=temperature, top_p=top_p, top_k=top_k,
                                   repetition_penalty=repetition_penalty, cfg_scale=cfg_scale)
            conditioning = clip.encode_from_tokens_scheduled(tokens)
        finally:
            _FLOOR.reset(token)
        return (conditioning, conditioning[0][1]["yue2_frames"] / FRAMES_PER_SECOND)


NODE_CLASS_MAPPINGS = {"YuE2GenerateMusicGuarded": YuE2GenerateMusicGuarded}
NODE_DISPLAY_NAME_MAPPINGS = {"YuE2GenerateMusicGuarded": "YuE2 Generate Music (score-length guard)"}
