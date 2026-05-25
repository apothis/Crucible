"""ComfyUI client + ACE-Step 1.5 workflow builders.

Wiring is the VERIFIED-working setup (see RESEARCH.md §8):
  DualCLIPLoader(qwen_0.6b, qwen_4b, type="ace") -> TextEncodeAceStepAudio1.5
  UNETLoader -> ModelSamplingAuraFlow(shift=3) -> KSampler
  text-to-music: EmptyAceStep1.5LatentAudio -> KSampler.latent_image
  restyle:       LoadAudio -> VAEEncodeAudio -> KSampler.latent_image
"""
import json
import random
import requests

# Fixed encoder pairing for ACE-Step 1.5 XL (slot order matters: 0.6b then 4b).
CLIP1 = "qwen_0.6b_ace15.safetensors"
CLIP2 = "qwen_4b_ace15.safetensors"
VAE = "ace_1.5_vae.safetensors"

# Model variants: file on disk + sensible default sampler settings.
VARIANTS = {
    "xl_base":  {"file": "acestep_v1.5_xl_base_bf16.safetensors",  "steps": 50, "cfg": 6.0,
                 "label": "XL Base (best quality)"},
    "xl_sft":   {"file": "acestep_v1.5_xl_sft_bf16.safetensors",   "steps": 50, "cfg": 7.0,
                 "label": "XL SFT (refined)"},
    "xl_turbo": {"file": "acestep_v1.5_xl_turbo_bf16.safetensors", "steps": 8,  "cfg": 1.0,
                 "label": "XL Turbo (fast preview)"},
}

KEYS = [f"{n} {m}" for m in ("major", "minor")
        for n in ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")]

# Suggested starting negative prompt for rock/metal (steers away from the usual
# generated-guitar artifacts). Not applied automatically — pass p["negative_tags"]
# to use it (empty = the original ConditioningZeroOut behavior, fully backward
# compatible). See RESEARCH.md §10c.
DEFAULT_NEGATIVE = ("muddy, lo-fi, harsh fizz, digital clipping, thin weak guitars, "
                    "out of tune, low quality, mono, distorted vocals, noise")


class Comfy:
    def __init__(self, host: str):
        self.host = host
        self.base = f"http://{host}"

    # ---- introspection ----
    def models(self, folder: str):
        try:
            return requests.get(f"{self.base}/models/{folder}", timeout=8).json()
        except Exception:
            return []

    def available_variants(self):
        have = set(self.models("diffusion_models"))
        out = []
        for key, v in VARIANTS.items():
            out.append({"id": key, "label": v["label"], "steps": v["steps"],
                        "cfg": v["cfg"], "available": v["file"] in have})
        return out

    # ---- io ----
    def upload_audio(self, file_bytes: bytes, filename: str) -> str:
        files = {"image": (filename, file_bytes, "application/octet-stream")}
        r = requests.post(f"{self.base}/upload/image", files=files,
                          data={"overwrite": "true"}, timeout=30)
        r.raise_for_status()
        j = r.json()
        name = j["name"]
        sub = j.get("subfolder") or ""
        return f"{sub}/{name}" if sub else name

    def submit(self, graph: dict, client_id: str) -> dict:
        r = requests.post(f"{self.base}/prompt",
                          json={"prompt": graph, "client_id": client_id}, timeout=20)
        r.raise_for_status()
        return r.json()

    def history(self, prompt_id: str) -> dict:
        return requests.get(f"{self.base}/history/{prompt_id}", timeout=10).json()

    def view_bytes(self, filename: str, subfolder: str, ftype: str) -> bytes:
        r = requests.get(f"{self.base}/view",
                         params={"filename": filename, "subfolder": subfolder, "type": ftype},
                         timeout=60)
        r.raise_for_status()
        return r.content

    def interrupt(self):
        try:
            requests.post(f"{self.base}/interrupt", timeout=8)
        except Exception:
            pass

    def free(self, unload_models=True, free_memory=True):
        """Ask ComfyUI to unload models / free VRAM. Used before driving another
        GPU model on the shared 3090 (e.g. a SoulX vocal build)."""
        try:
            requests.post(f"{self.base}/free",
                          json={"unload_models": unload_models, "free_memory": free_memory},
                          timeout=10)
        except Exception:
            pass


# ---- shared node fragments ----
def _loaders(variant_file, shift=3.0):
    return {
        "4": {"class_type": "UNETLoader",
              "inputs": {"unet_name": variant_file, "weight_dtype": "default"}},
        "5": {"class_type": "DualCLIPLoader",
              "inputs": {"clip_name1": CLIP1, "clip_name2": CLIP2, "type": "ace"}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "7": {"class_type": "ModelSamplingAuraFlow",
              "inputs": {"model": ["4", 0], "shift": float(shift)}},
    }


def _structure_only(lyrics: str) -> str:
    """Keep only bracketed structure tags (e.g. [verse], [solo]); drop sung words.
    Lets an instrumental track still honor a section arrangement (Song Constructor)
    without the model singing lyrics."""
    keep = [ln for ln in lyrics.splitlines() if ln.strip().startswith("[") and ln.strip().endswith("]")]
    return "\n".join(keep)


def _text_encode(p, generate_audio_codes):
    lyrics = _structure_only(p.get("lyrics", "")) if p.get("instrumental") else p.get("lyrics", "")
    return {"class_type": "TextEncodeAceStepAudio1.5", "inputs": {
        "clip": ["5", 0],
        "tags": p.get("tags", ""),
        "lyrics": lyrics,
        "seed": p["seed"],
        "bpm": int(p.get("bpm", 120)),
        "duration": float(p.get("duration", 60.0)),
        "timesignature": str(p.get("timesignature", "4")),
        "language": p.get("language", "en"),
        "keyscale": p.get("keyscale", "E minor"),
        "generate_audio_codes": generate_audio_codes,
        "cfg_scale": 2.0, "temperature": 0.85, "top_p": 0.9, "top_k": 0, "min_p": 0.0,
    }}


def _neg_encode(p):
    """Real negative-prompt conditioning. The stock ACE-Step workflow uses
    ConditioningZeroOut (an empty negative); supplying actual negative tags can
    steer the sampler away from muddy/fizzy/low-quality artifacts. Lyrics are
    left empty — only the style tags act as the negative."""
    return {"class_type": "TextEncodeAceStepAudio1.5", "inputs": {
        "clip": ["5", 0],
        "tags": p.get("negative_tags", ""),
        "lyrics": "",
        "seed": p["seed"],
        "bpm": int(p.get("bpm", 120)),
        "duration": float(p.get("duration", 60.0)),
        "timesignature": str(p.get("timesignature", "4")),
        "language": p.get("language", "en"),
        "keyscale": p.get("keyscale", "E minor"),
        "generate_audio_codes": False,
        "cfg_scale": 2.0, "temperature": 0.85, "top_p": 0.9, "top_k": 0, "min_p": 0.0,
    }}


def _negative_node(p):
    """Node '9' fed to KSampler.negative: a real negative encode when
    negative_tags are given, else the original zero-out (backward compatible)."""
    if (p.get("negative_tags") or "").strip():
        return _neg_encode(p)
    return {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["8", 0]}}


def _apg_model(g, p):
    """Optional Adaptive Projected Guidance (ComfyUI native `APG` node). It patches
    the model so strong CFG stops oversaturating / muddying ACE-Step output — the
    recommended fix for ACE's high-guidance fuzziness (it lets us keep cfg ~6-7.3
    for prompt adherence without the mud, instead of just dropping cfg). Returns
    the model ref the KSampler should read. OFF unless p['apg'] — backward compatible.

    Defaults follow the community ACE-XL-SFT recommendation (eta 1.05, norm 1.3,
    momentum 0); `norm_threshold` is the main knob (lower = more normalization)."""
    if not p.get("apg"):
        return ["7", 0]
    g["16"] = {"class_type": "APG", "inputs": {
        "model": ["7", 0],
        "eta": float(p.get("apg_eta", 1.05)),
        "norm_threshold": float(p.get("apg_norm", 1.3)),
        "momentum": float(p.get("apg_momentum", 0.0)),
    }}
    return ["16", 0]


def _resolve(p):
    """Fill in seed + per-variant default steps/cfg if not overridden."""
    v = VARIANTS.get(p.get("variant", "xl_base"), VARIANTS["xl_base"])
    p = dict(p)
    if not p.get("seed"):
        p["seed"] = random.randint(1, 2**31 - 1)
    p["seed"] = int(p["seed"])
    p["_file"] = v["file"]
    p["_steps"] = int(p.get("steps") or v["steps"])
    p["_cfg"] = float(p.get("cfg") if p.get("cfg") not in (None, "") else v["cfg"])
    return p


def build_t2m(p):
    p = _resolve(p)
    g = _loaders(p["_file"], p.get("shift", 3.0))
    g["8"] = _text_encode(p, generate_audio_codes=True)
    g["9"] = _negative_node(p)
    g["10"] = {"class_type": "EmptyAceStep1.5LatentAudio",
               "inputs": {"seconds": float(p.get("duration", 60.0)), "batch_size": 1}}
    g["11"] = {"class_type": "KSampler", "inputs": {
        "model": _apg_model(g, p), "positive": ["8", 0], "negative": ["9", 0], "latent_image": ["10", 0],
        "seed": p["seed"], "steps": p["_steps"], "cfg": p["_cfg"],
        "sampler_name": p.get("sampler_name", "euler"),
        "scheduler": p.get("scheduler", "simple"), "denoise": 1.0}}
    g["12"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["11", 0], "vae": ["6", 0]}}
    g["13"] = {"class_type": "SaveAudioMP3",
               "inputs": {"audio": ["12", 0], "filename_prefix": "musicgen/t2m", "quality": "320k"}}
    return g, p


def build_restyle(p, audio_ref):
    p = _resolve(p)
    g = _loaders(p["_file"], p.get("shift", 3.0))
    # For restyle we give the model an audio reference, so audio codes are OFF.
    g["8"] = _text_encode(p, generate_audio_codes=False)
    g["9"] = _negative_node(p)
    g["14"] = {"class_type": "LoadAudio", "inputs": {"audio": audio_ref}}
    g["15"] = {"class_type": "VAEEncodeAudio", "inputs": {"audio": ["14", 0], "vae": ["6", 0]}}
    # denoise = "restyle amount": higher transforms more (further from source).
    denoise = float(p.get("restyle_amount", 0.7))
    g["11"] = {"class_type": "KSampler", "inputs": {
        "model": _apg_model(g, p), "positive": ["8", 0], "negative": ["9", 0], "latent_image": ["15", 0],
        "seed": p["seed"], "steps": p["_steps"], "cfg": p["_cfg"],
        "sampler_name": p.get("sampler_name", "euler"),
        "scheduler": p.get("scheduler", "simple"), "denoise": denoise}}
    g["12"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["11", 0], "vae": ["6", 0]}}
    g["13"] = {"class_type": "SaveAudioMP3",
               "inputs": {"audio": ["12", 0], "filename_prefix": "musicgen/restyle", "quality": "320k"}}
    return g, p


def build_edit(p, audio_ref):
    """Repaint and/or extend an existing track via the vendored
    `ACEStep15NativeEditGuider` custom node (comfy_custom_nodes/ComfyUI-ACEStep-Repaint,
    installed on the box) driven through `SamplerCustomAdvanced`.

    - **Repaint** a region: set `repaint_start`/`repaint_end` (seconds; the rest is
      preserved). - **Extend**: set `extend_left`/`extend_right` (seconds added).
    Either or both. New content follows `tags`/`lyrics`. The guider returns both the
    GUIDER and a correctly-sized `output_latent` (grown for extend) → we sample that.
    """
    p = _resolve(p)
    g = _loaders(p["_file"], p.get("shift", 3.0))
    # The 1.5 native edit guider MUST be fed task-aware conditioning: task_type
    # "repaint" turns ON the LM audio-code generation that gives the regenerated /
    # extended region its musical structure. Using the plain TextEncode (codes off,
    # the cover/restyle pattern) starves it of codes → garbled noise in the new region.
    g["8"] = {"class_type": "ACEStep15TaskTextEncode", "inputs": {
        "clip": ["5", 0],
        "text": p.get("tags", ""),
        "task_type": "repaint",
        "track_name": "None",
        "lyrics": "" if p.get("instrumental") else p.get("lyrics", ""),
        "bpm": int(p.get("bpm", 120)),
        "duration": float(p.get("duration", 60.0)),
        "keyscale": p.get("keyscale", "E minor"),
        "timesignature": str(p.get("timesignature", "4")),
        "language": p.get("language", "en"),
        "seed": p["seed"],
        "cfg_scale": 2.0, "temperature": 0.85, "top_p": 0.9, "top_k": 0, "min_p": 0.0,
    }}
    g["9"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["8", 0]}}
    g["14"] = {"class_type": "LoadAudio", "inputs": {"audio": audio_ref}}
    g["15"] = {"class_type": "VAEEncodeAudio", "inputs": {"audio": ["14", 0], "vae": ["6", 0]}}
    g["20"] = {"class_type": "ACEStep15NativeEditGuider", "inputs": {
        "model": ["7", 0], "positive": ["8", 0], "negative": ["9", 0],
        "source_latents": ["15", 0], "cfg": float(p.get("edit_cfg", 1.0)),   # canonical edit cfg=1 (DiT guidance off; content from LM codes). >1 over-guides → garbled
        "extend_left_seconds": float(p.get("extend_left", 0.0)),
        "extend_right_seconds": float(p.get("extend_right", 0.0)),
        "repaint_start_seconds": float(p.get("repaint_start", -1.0)),
        "repaint_end_seconds": float(p.get("repaint_end", -1.0))}}
    g["21"] = {"class_type": "BasicScheduler", "inputs": {
        "model": ["7", 0], "scheduler": p.get("scheduler", "simple"),
        "steps": p["_steps"], "denoise": 1.0}}
    g["22"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": p.get("sampler_name", "euler")}}
    g["23"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": p["seed"]}}
    g["24"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["23", 0], "guider": ["20", 0], "sampler": ["22", 0],
        "sigmas": ["21", 0], "latent_image": ["20", 1]}}   # guider's prepared (extend-sized) latent
    g["12"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["24", 0], "vae": ["6", 0]}}
    g["13"] = {"class_type": "SaveAudioMP3",
               "inputs": {"audio": ["12", 0], "filename_prefix": "musicgen/edit", "quality": "320k"}}
    return g, p
