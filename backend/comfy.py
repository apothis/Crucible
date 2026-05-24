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
def _loaders(variant_file):
    return {
        "4": {"class_type": "UNETLoader",
              "inputs": {"unet_name": variant_file, "weight_dtype": "default"}},
        "5": {"class_type": "DualCLIPLoader",
              "inputs": {"clip_name1": CLIP1, "clip_name2": CLIP2, "type": "ace"}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "7": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["4", 0], "shift": 3.0}},
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
    g = _loaders(p["_file"])
    g["8"] = _text_encode(p, generate_audio_codes=True)
    g["9"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["8", 0]}}
    g["10"] = {"class_type": "EmptyAceStep1.5LatentAudio",
               "inputs": {"seconds": float(p.get("duration", 60.0)), "batch_size": 1}}
    g["11"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["8", 0], "negative": ["9", 0], "latent_image": ["10", 0],
        "seed": p["seed"], "steps": p["_steps"], "cfg": p["_cfg"],
        "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}}
    g["12"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["11", 0], "vae": ["6", 0]}}
    g["13"] = {"class_type": "SaveAudioMP3",
               "inputs": {"audio": ["12", 0], "filename_prefix": "musicgen/t2m", "quality": "320k"}}
    return g, p


def build_restyle(p, audio_ref):
    p = _resolve(p)
    g = _loaders(p["_file"])
    # For restyle we give the model an audio reference, so audio codes are OFF.
    g["8"] = _text_encode(p, generate_audio_codes=False)
    g["9"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["8", 0]}}
    g["14"] = {"class_type": "LoadAudio", "inputs": {"audio": audio_ref}}
    g["15"] = {"class_type": "VAEEncodeAudio", "inputs": {"audio": ["14", 0], "vae": ["6", 0]}}
    # denoise = "restyle amount": higher transforms more (further from source).
    denoise = float(p.get("restyle_amount", 0.7))
    g["11"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["8", 0], "negative": ["9", 0], "latent_image": ["15", 0],
        "seed": p["seed"], "steps": p["_steps"], "cfg": p["_cfg"],
        "sampler_name": "euler", "scheduler": "simple", "denoise": denoise}}
    g["12"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["11", 0], "vae": ["6", 0]}}
    g["13"] = {"class_type": "SaveAudioMP3",
               "inputs": {"audio": ["12", 0], "filename_prefix": "musicgen/restyle", "quality": "320k"}}
    return g, p
