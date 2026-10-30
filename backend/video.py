"""ComfyUI graph builders for stills and H3 video, as used by the YouTube tab.

  - Z-Image Turbo still: UNETLoader(z_image_turbo) + CLIPLoader(qwen_3_4b, type=lumina2)
    + VAELoader(ae) + ModelSamplingAuraFlow(shift=3) + KSampler(8 steps, cfg 1,
    res_multistep/simple) + SaveImage.
  - Krea 2 still + Krea 2 identity edit (cover art, cover-model re-staging).
  - FlashVSR upscale of a library clip.
  - MiniMax H3 text/first-last-frame lanes (build_h3_t2v, build_h3_i2v): backgrounds and
    living-cover loops. Measured constraints are in docs/H3_LIVING_COVER.md and
    docs/MINIMAX_H3_PLAN.md (history).

The Wan/S2V/InfiniteTalk/LTX/MSR/ref2v builders were removed 2026-10-27; the rebuilt
music-video pipeline is docs/MV_H3_REBUILD_PLAN.md.

All builders return (graph_dict, resolved_params) like comfy.build_* do. Output node
filename_prefix routes the result into ComfyUI's output/ folder; app.on_complete fetches it.

Plain ASCII only.
"""
import json
import random

# ---- model files on the box (downloaded by download_video_models.py) ----
Z_IMAGE_UNET = "z_image_turbo_bf16.safetensors"
Z_IMAGE_CLIP = "qwen_3_4b.safetensors"           # CLIPLoader type "lumina2"
Z_IMAGE_VAE = "ae.safetensors"

# Krea 2 Ultra (AItrepreneur KREA2_ULTRA_WORKFLOW v2). A Qwen-Image-family turbo model: Qwen-Image
# VAE + a Qwen3-VL-4B text encoder (CLIPLoader type "krea2"), run at cfg 1 / ~8 steps / er_sde /
# simple (distilled "turbo"). Drop-in alternative to Z-Image Turbo for the photoreal still path.
# fp8 vs mxfp8: the AItrepreneur note says 5000-series GPUs use mxfp8, 4000-series-and-older use fp8 -
# our 3090 is Ampere, so the fp8 build. Filename VERIFIED on the box /object_info 2026-06-28 (the
# AItrepreneur download set ships it as krea2_turbo_fp8.safetensors, NOT the official _scaled name).
KREA2_UNET = "krea2_turbo_fp8.safetensors"            # 3090/Ampere fp8 (verified on box; 50-series = _mxfp8)
KREA2_CLIP = "qwen3vl_4b_fp8_scaled.safetensors"      # CLIPLoader type "krea2"
KREA2_VAE = "qwen_image_vae.safetensors"
# The turbo LoRA + every number below are taken VERBATIM from the AItrepreneur KREA2_ULTRA_WORKFLOW
# v2 (don't change them without re-reading the workflow). The single-pass "TEXT TO IMAGE" path runs
# the model with NO LoRA (its Power Lora Loader is empty); ONLY the "TWO TIMES COMBO" path enables
# the turbo LoRA, on BOTH passes at strength 0.2 (Power Lora Loaders 174 + 180).
KREA2_TURBO_LORA = "krea2_turbo_lora_rank_64_bf16.safetensors"
KREA2_TURBO_LORA_STRENGTH = 0.2
# TEXT TO IMAGE (subgraph 21ffbd86, KSampler 75): er_sde / simple / 8 steps / cfg 1 / denoise 1.
KREA2_T2I = {"sampler": "er_sde", "scheduler": "simple", "steps": 8, "cfg": 1.0, "denoise": 1.0}
# TWO TIMES COMBO: pass1 (KSampler 273) = er_sde/simple/8/cfg1/denoise1 on EmptySD3LatentImage;
# decode -> ImageResize+ (lanczos, keep proportion) -> VAEEncode; pass2 (KSampler 265) =
# euler/simple/4/cfg1/denoise 0.3. Both passes use the turbo LoRA @ 0.2.
KREA2_COMBO_P1 = {"sampler": "er_sde", "scheduler": "simple", "steps": 8, "cfg": 1.0, "denoise": 1.0}
# "IMAGE TO IMAGE" group, read off KREA2_ULTRA_WORKFLOW-V2.json (node 116 inside subgraph 06153499):
# LoadImage -> ImageResize+ (1024 long edge, lanczos, keep proportion) -> VAEEncode -> KSampler
# euler/simple/8/cfg1/DENOISE 0.40 -> VAEDecode. It is img2img on ONE image, not reference
# conditioning: it re-renders what it is given, so it polishes or restyles a still, it cannot place
# two referenced people in a new scene.
KREA2_I2I = {"sampler": "euler", "scheduler": "simple", "steps": 8, "cfg": 1.0, "denoise": 0.40,
             "resize": 1024}
KREA2_COMBO_P2 = {"sampler": "euler", "scheduler": "simple", "steps": 4, "cfg": 1.0, "denoise": 0.3}
# Photorealism emphasis appended to EVERY Krea2 prompt (user: "emphasize photorealism in all prompts").
# Tasteful, scene-safe cues per Krea2 prompting guidance (real medium + true-to-life detail) - NOT the
# gloss-inducing "8k/hyperdetailed/masterpiece" spam that reintroduces the AI look. Pass photoreal=False
# to skip for a specific render.
KREA2_PHOTOREAL = ("photorealistic, ultra-realistic, cinematic, true-to-life detail, natural lighting, "
                   "realistic textures, shot on a full-frame camera, no illustration, no CGI")

# A readable English negative that steers Wan/Z-Image away from the usual artifacts.
DEFAULT_NEG = ("low quality, blurry, distorted, deformed, bad anatomy, extra fingers, "
               "watermark, text, jpeg artifacts, overexposed, static, plastic skin, cartoon")


def _seed(p):
    s = p.get("seed")
    if s in (None, "", 0, "0"):
        return random.randint(0, 2**31 - 1)
    return int(s)


# ---------------------------------------------------------------- Z-Image still
def build_krea2_still(p, init_image=None):
    """Text-to-image photoreal still on Krea 2 Ultra (turbo). FAITHFUL port of AItrepreneur's
    KREA2_ULTRA_WORKFLOW v2 (numbers verbatim - see the KREA2_* constants above). Qwen3-VL-4B CLIP
    (type "krea2") + Qwen-Image VAE; NO ModelSamplingAuraFlow (unlike Z-Image); negative is a
    ConditioningZeroOut of the positive (cfg 1, so no real negative).

    Two modes (workflow's two paths):
      - default single pass ("TEXT TO IMAGE"): no LoRA, KSampler er_sde/8/cfg1/denoise1 on
        EmptyLatentImage.
      - two_pass=True ("TWO TIMES COMBO"): turbo LoRA @0.2 on BOTH passes; pass1 er_sde/8/denoise1
        on EmptySD3LatentImage -> VAEDecode -> ImageResize+ (lanczos, keep proportion) -> VAEEncode
        -> pass2 euler/4/denoise0.3. Quality path; ~2x slower.

    A third path, IMAGE TO IMAGE, runs when `init_image` (an uploaded image name on ComfyUI) is
    given: resize -> VAEEncode -> KSampler at denoise 0.40. Single image, img2img - it re-renders
    what it is handed, so it is for polishing/restyling a still, NOT for placing referenced people.

    p: {prompt, seed?, width?, height?, steps?(override pass-1 steps), cfg?, two_pass?(bool),
    turbo_lora?(bool, force the turbo LoRA on the single pass too), lora?/lora_strength? (a trained
    character LoRA, model-only)}. Output: SaveImage -> videogen/still."""
    seed = _seed(p)
    w = int(p.get("width", 1920))                   # workflow default canvas (EmptyLatentImage 76 = 1920x1080)
    h = int(p.get("height", 1080))
    cfg = float(p.get("cfg", KREA2_T2I["cfg"]))
    two_pass = bool(p.get("two_pass"))
    # Two optional workflow nodes (custom-node deps; off unless asked):
    #  - enhancer (ComfyUI-Krea2T-Enhancer): MODEL->MODEL patch that scales Krea2's text-fusion tap
    #    layers => stronger prompt adherence + "unfilter" (workflow ships it ENABLED, strength 1.0).
    #  - seed_variance (RBG_Smart_Seed_Variance): CONDITIONING->CONDITIONING controlled-noise inject
    #    for composition variety (the workflow's KSampler seeds are FIXED, so variety comes from here;
    #    OUR KSampler seed is already random, so this is extra variety, not required).
    enhancer = bool(p.get("enhancer"))
    enh_strength = float(p.get("enhancer_strength", 1.0))   # node range 0..2; workflow default 1.0
    seed_variance = bool(p.get("seed_variance"))
    prompt = (p.get("prompt") or "").strip()
    layout = p.get("layout")
    if prompt and not layout and p.get("photoreal", True):  # emphasize photorealism on every Krea2 still
        prompt = f"{prompt}. {KREA2_PHOTOREAL}"
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": KREA2_UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": KREA2_CLIP, "type": "krea2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": KREA2_VAE}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},   # negative = zeroed pos (cfg 1)
    }
    if layout:
        # REGIONAL layout prompting (the KREA2_ULTRA_WORKFLOW v2.1's Ideogram4PromptBuilderKJ path):
        # the KJ node assembles an Ideogram-4-style structured caption JSON - global fields plus
        # per-REGION descriptions on a normalized bbox grid (0-1000, [ymin,xmin,ymax,xmax]) - and its
        # STRING output feeds the ordinary CLIPTextEncode; Krea2's Qwen3-VL encoder understands the
        # format. Region x/y/w/h here are 0-1 FRACTIONS of the canvas (the node's own editor format,
        # matching the workflow's widgets_values); the node does the 0-1000 conversion itself.
        # `style` is a DynamicCombo: API format = {"style": "photo", "style.photo": "<camera text>"}
        # (dotted key, same _io.py machinery as the H3 autogrow inputs). When a layout is used, the
        # KREA2_PHOTOREAL suffix is NOT appended (it would corrupt the JSON caption) - put photoreal
        # emphasis in the layout's aesthetics/medium fields instead.
        elements = [{"type": r.get("type", "obj"), "text": r.get("text", ""),
                     "desc": (r.get("desc") or "").strip(),
                     "palette": r.get("palette", []),
                     "x": float(r["x"]), "y": float(r["y"]),
                     "w": float(r["w"]), "h": float(r["h"])}
                    for r in (layout.get("regions") or [])]
        g["50"] = {"class_type": "Ideogram4PromptBuilderKJ",
                   "inputs": {"width": w, "height": h,
                              "high_level_description": (layout.get("overview") or "").strip(),
                              "background": (layout.get("background") or "").strip(),
                              "style": "photo",
                              "style.photo": (layout.get("photo_style") or "").strip(),
                              "aesthetics": (layout.get("aesthetics") or "").strip(),
                              "lighting": (layout.get("lighting") or "").strip(),
                              "medium": layout.get("medium", "photograph"),
                              "style_palette_data": json.dumps(layout.get("palette", []), separators=(",", ":")),
                              "elements_data": json.dumps(elements, separators=(",", ":")),
                              "bg_brightness": 25, "import_mode": "when empty",
                              "output_format": "compact"}}
        g["5"]["inputs"]["text"] = ["50", 0]
    # positive conditioning: raw, or routed through the seed-variance node (faithful widget values)
    pos_src = ["5", 0]
    if seed_variance:
        g["41"] = {"class_type": "RBG_Smart_Seed_Variance",
                   "inputs": {"conditioning": ["5", 0],
                              "variance_preset": "\U0001f33f Balanced", "fine_tune_variance": 55,
                              "model_type": "⚙️ Other", "fade_curve": "Instant",
                              "noise_injection": "Beginning Steps", "protect_prompt": "\U0001f6ab None",
                              "direction_shift": "\U0001f6ab None", "shift_strength": 129,
                              "variance_schedule": "constant", "cutoff_step": 8, "total_steps": 20,
                              "cutoff_strength": 0.0, "seed": random.randint(0, 2**31 - 1)}}
        pos_src = ["41", 0]

    def _model_chain(use_turbo):
        """Build the model path off the UNET: [turbo LoRA @0.2] -> [optional character LoRA] ->
        [optional Krea2T enhancer]. Mirrors the workflow's UNET -> Lora -> Enhancer -> KSampler chain."""
        src = ["1", 0]; nid = 30
        if use_turbo:                               # combo path = turbo LoRA on (str 0.2, model-only)
            g[str(nid)] = {"class_type": "LoraLoaderModelOnly",
                           "inputs": {"model": src, "lora_name": KREA2_TURBO_LORA, "strength_model": KREA2_TURBO_LORA_STRENGTH}}
            src = [str(nid), 0]; nid += 1
        if p.get("lora"):                           # optional trained character LoRA -> identity
            g[str(nid)] = {"class_type": "LoraLoaderModelOnly",
                           "inputs": {"model": src, "lora_name": p["lora"], "strength_model": float(p.get("lora_strength", 1.0))}}
            src = [str(nid), 0]; nid += 1
        if enhancer:                                # ComfyUI-Krea2T-Enhancer (enabled, strength, debug)
            g["40"] = {"class_type": "ComfyUI-Krea2T-Enhancer",
                       "inputs": {"model": src, "enabled": True, "strength": enh_strength, "debug": False}}
            src = ["40", 0]
        return src

    # ---- IMAGE TO IMAGE: re-render an existing still through Krea2 (workflow's third path) ----
    if init_image:
        model_src = _model_chain(bool(p.get("turbo_lora")))
        s = KREA2_I2I
        steps = int(p.get("steps", s["steps"]))
        den = float(p.get("denoise", s["denoise"]))
        long_edge = int(p.get("resize", s["resize"]))
        g.update({
            "7": {"class_type": "LoadImage", "inputs": {"image": init_image}},
            # the workflow resizes BEFORE encoding (keep proportion, so the aspect survives)
            "11": {"class_type": "ImageResize+",
                   "inputs": {"image": ["7", 0], "width": long_edge, "height": long_edge,
                              "interpolation": "lanczos", "method": "keep proportion",
                              "condition": "always", "multiple_of": 0}},
            "12": {"class_type": "VAEEncode", "inputs": {"pixels": ["11", 0], "vae": ["3", 0]}},
            "8": {"class_type": "KSampler",
                  "inputs": {"model": model_src, "seed": seed, "steps": steps, "cfg": cfg,
                             "sampler_name": s["sampler"], "scheduler": s["scheduler"],
                             "positive": pos_src, "negative": ["6", 0],
                             "latent_image": ["12", 0], "denoise": den}},
            "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
            "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "videogen/still"}},
        })
        return g, {"seed": seed, "steps": steps, "cfg": cfg, "denoise": den, "prompt": prompt,
                   "lora": p.get("lora"), "engine": "krea2", "mode": "img2img",
                   "init_image": init_image, "resize": long_edge, "enhancer": enhancer,
                   "seed_variance": seed_variance, "kind": "image"}

    if not two_pass:
        model_src = _model_chain(bool(p.get("turbo_lora")))
        s = KREA2_T2I
        steps = int(p.get("steps", s["steps"]))
        g.update({
            # Single-pass t2i uses EmptyLatentImage (both the official template + workflow subgraph 76).
            "7": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
            "8": {"class_type": "KSampler",
                  "inputs": {"model": model_src, "seed": seed, "steps": steps, "cfg": cfg,
                             "sampler_name": s["sampler"], "scheduler": s["scheduler"],
                             "positive": pos_src, "negative": ["6", 0],
                             "latent_image": ["7", 0], "denoise": s["denoise"]}},
            "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
            "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "videogen/still"}},
        })
        return g, {"seed": seed, "width": w, "height": h, "steps": steps, "cfg": cfg,
                   "prompt": prompt, "lora": p.get("lora"), "engine": "krea2", "two_pass": False,
                   "enhancer": enhancer, "seed_variance": seed_variance,
                   "layout": layout or None, "kind": "image"}

    # ---- TWO TIMES COMBO (quality path): turbo LoRA on both passes ----
    model_src = _model_chain(True)                  # combo always enables the turbo LoRA
    p1, p2 = KREA2_COMBO_P1, KREA2_COMBO_P2
    p1_steps = int(p.get("steps", p1["steps"]))
    g.update({
        # pass 1 (workflow's combo uses EmptySD3LatentImage 266)
        "7": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "8": {"class_type": "KSampler",
              "inputs": {"model": model_src, "seed": seed, "steps": p1_steps, "cfg": cfg,
                         "sampler_name": p1["sampler"], "scheduler": p1["scheduler"],
                         "positive": pos_src, "negative": ["6", 0],
                         "latent_image": ["7", 0], "denoise": p1["denoise"]}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        # decode -> resize (ImageResize+ : lanczos, keep proportion, condition always) -> re-encode
        "11": {"class_type": "ImageResize+",
               "inputs": {"image": ["9", 0], "width": w, "height": h, "interpolation": "lanczos",
                          "method": "keep proportion", "condition": "always", "multiple_of": 0}},
        "12": {"class_type": "VAEEncode", "inputs": {"pixels": ["11", 0], "vae": ["3", 0]}},
        # pass 2 = refine at denoise 0.3 (euler / 4 steps), SAME conditioning, decoupled seed
        "13": {"class_type": "KSampler",
               "inputs": {"model": model_src, "seed": random.randint(0, 2**31 - 1), "steps": p2["steps"], "cfg": cfg,
                          "sampler_name": p2["sampler"], "scheduler": p2["scheduler"],
                          "positive": pos_src, "negative": ["6", 0],
                          "latent_image": ["12", 0], "denoise": p2["denoise"]}},
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["3", 0]}},
        "15": {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": "videogen/still"}},
    })
    return g, {"seed": seed, "width": w, "height": h, "steps": p1_steps, "cfg": cfg,
               "prompt": prompt, "lora": p.get("lora"), "engine": "krea2", "two_pass": True,
               "turbo_lora": KREA2_TURBO_LORA, "enhancer": enhancer, "seed_variance": seed_variance,
               "layout": layout or None, "kind": "image"}


# Krea 2 Identity Edit (community LoRA, civitai model 2761113 + lbouaraba/comfyui-krea2edit):
# instruction-based identity-PRESERVING re-staging on Krea2 - "move a person to new scenes
# while preserving their face". Numbers below are VERBATIM from the pack's shipped
# krea2_identity_edit.json workflow (euler/simple/10/cfg 1/denoise 1, LoRA @1.0 fixed,
# ref_boost 4 / ref_boost_a 1 / fit_mode "fit"). One deliberate deviation: grounding_px
# defaults to 1024 rather than the workflow's 768 - the author's own model page says
# "1024+ recommended for people" and our subject is always a person [upstream doc].
KREA2_EDIT_SETTINGS = {"steps": 10, "cfg": 1.0, "sampler": "euler", "scheduler": "simple",
                       "ref_boost": 4.0, "ref_boost_a": 1.0, "fit_mode": "fit",
                       "grounding_px": 1024, "lora_strength": 1.0}


def build_krea2_identity_still(p, ref_image, lora_name):
    """Re-stage the person in `ref_image` (uploaded name on ComfyUI) into a new scene per
    the instruction `prompt`, keeping their identity (Krea2 Identity Edit LoRA). p:
    {prompt, seed?, width?, height?, ref_boost?, grounding_px?, steps?, cfg?}. The author
    trained on <=2MP same-size pairs, so the default canvas is 1792x1008 (16:9 under 2MP)
    rather than the still path's 1920x1080. Output: SaveImage -> videogen/still."""
    s = KREA2_EDIT_SETTINGS
    seed = _seed(p)
    w = int(p.get("width", 1792))
    h = int(p.get("height", 1008))
    prompt = (p.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("an instruction prompt is required (e.g. 'Re-stage the woman into ...')")
    # EXPERIMENTAL flags, off by default (the author's shipped workflow is single-pass, no
    # enhancer; these graft our plain-still combo stages onto the identity path for A/B):
    #  - two_pass: decode -> resize -> re-encode -> 4-step euler refine at denoise 0.3
    #    (KREA2_COMBO_P2 verbatim), same patched model + conditioning.
    #  - enhancer: Krea2T enhancer patch between the identity LoRA and the edit patch.
    two_pass = bool(p.get("two_pass"))
    enhancer = bool(p.get("enhancer"))
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": KREA2_UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": KREA2_CLIP, "type": "krea2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": KREA2_VAE}},
        "4": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["1", 0], "lora_name": lora_name,
                         "strength_model": float(p.get("lora_strength", s["lora_strength"]))}},
        "7": {"class_type": "LoadImage", "inputs": {"image": ref_image}},
        "8": {"class_type": "VAEEncode", "inputs": {"pixels": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        # in-context patch: the reference's VAE tokens are prepended to the sampling context
        "10": {"class_type": "Krea2EditModelPatch",
               "inputs": {"model": ["40", 0] if enhancer else ["4", 0],
                          "source_latent": ["8", 0], "target_latent": ["9", 0],
                          "fit_mode": s["fit_mode"],
                          "ref_boost": float(p.get("ref_boost", s["ref_boost"])),
                          "ref_boost_a": s["ref_boost_a"]}},
        # image-grounded instruction encoding (Qwen3-VL sees the reference)
        "11": {"class_type": "Krea2EditGroundedEncode",
               "inputs": {"clip": ["2", 0], "prompt": prompt, "image": ["7", 0],
                          "grounding_px": int(p.get("grounding_px", s["grounding_px"]))}},
        "12": {"class_type": "Krea2EditGroundedEncode",
               "inputs": {"clip": ["2", 0], "prompt": "", "image": ["7", 0],
                          "grounding_px": int(p.get("grounding_px", s["grounding_px"]))}},
        "13": {"class_type": "KSampler",
               "inputs": {"model": ["10", 0], "seed": seed,
                          "steps": int(p.get("steps", s["steps"])), "cfg": float(p.get("cfg", s["cfg"])),
                          "sampler_name": s["sampler"], "scheduler": s["scheduler"],
                          "positive": ["11", 0], "negative": ["12", 0],
                          "latent_image": ["9", 0], "denoise": 1.0}},
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["3", 0]}},
    }
    if enhancer:
        g["40"] = {"class_type": "ComfyUI-Krea2T-Enhancer",
                   "inputs": {"model": ["4", 0], "enabled": True, "strength": 1.0, "debug": False}}
    if two_pass:
        r = KREA2_COMBO_P2
        g.update({
            "16": {"class_type": "ImageResize+",
                   "inputs": {"image": ["14", 0], "width": w, "height": h, "interpolation": "lanczos",
                              "method": "keep proportion", "condition": "always", "multiple_of": 0}},
            "17": {"class_type": "VAEEncode", "inputs": {"pixels": ["16", 0], "vae": ["3", 0]}},
            "18": {"class_type": "KSampler",
                   "inputs": {"model": ["10", 0], "seed": random.randint(0, 2**31 - 1),
                              "steps": r["steps"], "cfg": float(p.get("cfg", s["cfg"])),
                              "sampler_name": r["sampler"], "scheduler": r["scheduler"],
                              "positive": ["11", 0], "negative": ["12", 0],
                              "latent_image": ["17", 0], "denoise": r["denoise"]}},
            "19": {"class_type": "VAEDecode", "inputs": {"samples": ["18", 0], "vae": ["3", 0]}},
            "15": {"class_type": "SaveImage", "inputs": {"images": ["19", 0], "filename_prefix": "videogen/still"}},
        })
    else:
        g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": "videogen/still"}}
    return g, {"seed": seed, "width": w, "height": h, "prompt": prompt, "engine": "krea2_edit",
               "ref_boost": float(p.get("ref_boost", s["ref_boost"])),
               "grounding_px": int(p.get("grounding_px", s["grounding_px"])),
               "two_pass": two_pass, "enhancer": enhancer,
               "lora": lora_name, "kind": "image"}


def build_still(p, init_image=None):
    """Text-to-image photoreal still. Engine selectable: "zimage" (Z-Image Turbo, default) or
    "krea2" (Krea 2 Ultra). p: {prompt, negative?, seed?, width?, height?, steps?, cfg?, engine?}.
    `init_image` (krea2 only) runs the workflow's IMAGE TO IMAGE path over an uploaded image.
    Output: SaveImage -> videogen/still."""
    if (p.get("engine") or "").lower() == "krea2":
        return build_krea2_still(p, init_image=init_image)
    seed = _seed(p)
    w = int(p.get("width", 1024))
    h = int(p.get("height", 1024))
    steps = int(p.get("steps", 8))
    cfg = float(p.get("cfg", 1.0))
    lora = p.get("lora")                          # optional Z-Image character LoRA (filename)
    lora_strength = float(p.get("lora_strength", 1.0))
    prompt = (p.get("prompt") or "").strip()
    neg = p.get("negative")
    neg = DEFAULT_NEG if neg is None else neg
    model_src = ["1", 0]
    g = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": Z_IMAGE_UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": Z_IMAGE_CLIP, "type": "lumina2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": Z_IMAGE_VAE}},
    }
    if lora:                                       # trained character LoRA -> consistent identity
        g["16"] = {"class_type": "LoraLoaderModelOnly",
                   "inputs": {"model": ["1", 0], "lora_name": lora, "strength_model": lora_strength}}
        model_src = ["16", 0]
    g["4"] = {"class_type": "ModelSamplingAuraFlow",
              "inputs": {"model": model_src, "shift": 3.0}}
    g.update({
        "5": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["2", 0], "text": prompt}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["2", 0], "text": neg}},
        "7": {"class_type": "EmptySD3LatentImage",
              "inputs": {"width": w, "height": h, "batch_size": 1}},
        "8": {"class_type": "KSampler",
              "inputs": {"model": ["4", 0], "seed": seed, "steps": steps, "cfg": cfg,
                         "sampler_name": "res_multistep", "scheduler": "simple",
                         "positive": ["5", 0], "negative": ["6", 0],
                         "latent_image": ["7", 0], "denoise": 1.0}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveImage",
               "inputs": {"images": ["9", 0], "filename_prefix": "videogen/still"}},
    })
    return g, {"seed": seed, "width": w, "height": h, "steps": steps, "cfg": cfg,
               "prompt": prompt, "lora": lora, "kind": "image"}


def build_flashvsr_upscale(p, video_ref, fps):
    """Upscale a finished clip with FlashVSR (naxci1 "Stable" node, diffusion VSR). Alternative to
    SeedVR2. video_ref = uploaded video name on ComfyUI; fps = source fps. The single FlashVSRNode
    takes IMAGE frames -> upscaled IMAGE; scale is an INTEGER factor (2x/4x), NOT a target short-side.
    OOM safety nets (tiled_vae/tiled_dit/keep_models_on_cpu) stay ON so it degrades instead of
    crashing. p: {model?, mode? (tiny|tiny-long|full), vae_model?, scale? (2/4), tiled_vae?,
    tiled_dit?, unload_dit?, keep_models_on_cpu?, attention_mode?, frame_chunk_size?, resize_factor?,
    frame_cap?, seed?}. Models live in models/FlashVSR-v1.1 (see download_flashvsr_models.bat)."""
    seed = _seed(p)
    model = p.get("model") or "FlashVSR-v1.1"
    mode = p.get("mode") or "full"                         # tiny=fast/low-vram, full=best quality
    vae_model = p.get("vae_model") or "Wan2.1"
    scale = int(p.get("scale", 2))                         # integer factor: 2x (1280x704 -> 2560x1408)
    tiled_vae = bool(p.get("tiled_vae", True))            # OOM nets ON by default (24GB)
    tiled_dit = bool(p.get("tiled_dit", True))
    unload_dit = bool(p.get("unload_dit", True))
    keep_cpu = bool(p.get("keep_models_on_cpu", True))
    # sdpa is the always-available attention; sparse_sage_attention is faster but needs the bundled
    # sparse-sage kernels to build on the box - default to sdpa for reliability, override to try sage.
    attention = p.get("attention_mode") or "sdpa"
    chunk = int(p.get("frame_chunk_size", 0))              # 0 = all frames in one pass
    resize_factor = float(p.get("resize_factor", 1.0))
    cap = int(p.get("frame_cap", 0))                       # 0 = all frames
    load = {"1": {"class_type": "VHS_LoadVideo",
                  "inputs": {"video": video_ref, "force_rate": 0.0, "custom_width": 0,
                             "custom_height": 0, "frame_load_cap": cap, "skip_first_frames": 0,
                             "select_every_nth": 1}}}
    # VHS_LoadVideo outputs: images=[1,0], frame_count=[1,1], audio=[1,2], video_info=[1,3]
    # ADVANCED path is the DEFAULT (set simple=true to force the basic node): FlashVSRInitPipe ->
    # FlashVSRNodeAdv exposes tile_size/tile_overlap so we control how big each tile is. tile_size=512
    # is the measured sweet spot on the 24GB 3090 (peak ~14.4GB, ~164s/145f, 6 tiles): faster + fewer
    # seams than 256, while 768 OOM'd. Untiled OOMs at 2x/145f so tiling stays ON, but we keep models
    # resident (force_offload/keep_models_on_cpu False) for speed - a proven free win. Long clips eat
    # the same VRAM budget as big tiles, so set frame_chunk_size for >~10s shots (see endpoint).
    if not p.get("simple"):
        tile_size = int(p.get("tile_size", 512))           # measured sweet spot; 768 OOMs at 145f
        tile_overlap = int(p.get("tile_overlap", 32))
        color_fix = bool(p.get("color_fix", True))
        force_offload = bool(p.get("force_offload", False))   # False = model resident on GPU (faster)
        precision = p.get("precision") or "fp16"
        sparse_ratio = float(p.get("sparse_ratio", 2.0))
        kv_ratio = float(p.get("kv_ratio", 3.0))
        local_range = int(p.get("local_range", 11))
        adv_keep_cpu = bool(p.get("keep_models_on_cpu", False))  # resident by default in adv (speed)
        adv_unload = bool(p.get("unload_dit", False))
        g = dict(load)
        g["2"] = {"class_type": "FlashVSRInitPipe",
                  "inputs": {"model": model, "mode": mode, "vae_model": vae_model,
                             "force_offload": force_offload, "precision": precision,
                             "device": "auto", "attention_mode": attention}}
        g["3"] = {"class_type": "FlashVSRNodeAdv",
                  "inputs": {"pipe": ["2", 0], "frames": ["1", 0], "scale": scale,
                             "color_fix": color_fix, "tiled_vae": tiled_vae, "tiled_dit": tiled_dit,
                             "tile_size": tile_size, "tile_overlap": tile_overlap,
                             "unload_dit": adv_unload, "sparse_ratio": sparse_ratio,
                             "kv_ratio": kv_ratio, "local_range": local_range, "seed": seed,
                             "frame_chunk_size": chunk, "enable_debug": False,
                             "keep_models_on_cpu": adv_keep_cpu, "resize_factor": resize_factor}}
        # no audio passthrough: some source clips (no-lipsync MSR) have no audio track and VHS audio
        # extraction then fails; the final mux uses the song anyway, so upscaled clips stay silent.
        g["4"] = {"class_type": "CreateVideo",
                  "inputs": {"images": ["3", 0], "fps": float(fps)}}
        g["5"] = {"class_type": "SaveVideo",
                  "inputs": {"video": ["4", 0], "filename_prefix": "videogen/flashvsr",
                             "format": "auto", "codec": "auto"}}
        return g, {"seed": seed, "model": model, "mode": mode, "vae_model": vae_model, "scale": scale,
                   "advanced": True, "tile_size": tile_size, "tile_overlap": tile_overlap,
                   "tiled_vae": tiled_vae, "tiled_dit": tiled_dit, "force_offload": force_offload,
                   "attention_mode": attention, "fps": fps, "kind": "video"}
    g = dict(load)
    g["2"] = {"class_type": "FlashVSRNode",
              "inputs": {"frames": ["1", 0], "model": model, "mode": mode, "vae_model": vae_model,
                         "scale": scale, "tiled_vae": tiled_vae, "tiled_dit": tiled_dit,
                         "unload_dit": unload_dit, "seed": seed, "frame_chunk_size": chunk,
                         "attention_mode": attention, "enable_debug": False,
                         "keep_models_on_cpu": keep_cpu, "resize_factor": resize_factor}}
    g["3"] = {"class_type": "CreateVideo",
              "inputs": {"images": ["2", 0], "fps": float(fps)}}   # no audio passthrough (final mux uses the song)
    g["4"] = {"class_type": "SaveVideo",
              "inputs": {"video": ["3", 0], "filename_prefix": "videogen/flashvsr",
                         "format": "auto", "codec": "auto"}}
    return g, {"seed": seed, "model": model, "mode": mode, "vae_model": vae_model, "scale": scale,
               "tiled_vae": tiled_vae, "tiled_dit": tiled_dit, "attention_mode": attention,
               "fps": fps, "kind": "video"}


def build_flashvsr_plus_upscale(p, video_ref):
    """Upscale a finished clip with FlashVSR+ (kkget/ComfyUI-FlashVSR_Plus @ 23fef7c), the author's
    streaming example graph (example_workflows/streaming_video_api.json): LoadVideo ->
    FlashVSRPlusLoader -> FlashVSRPlusVideoUpscale -> SaveVideo, the settings of the 2026-10-30 SSBS
    A/B the user judged best (Turbo, scale 2, tiny decoder, color_fix off, crf 18, audio kept).
    Output sizes round up to multiples of 128 (1344x768 -> 2688x1536). The stock node encodes a
    24 fps input at 24.4167 fps (it reads the rate off a temp .mkv); the box copy is patched, and
    the caller re-checks the rate on the downloaded file anyway (mv_finish.fix_fps).
    p: {mode? (turbo), scale? (2.0), seed? (1), crf? (18), color_fix? (false)}."""
    mode = p.get("mode") or "turbo"
    scale = float(p.get("scale") or 2.0)
    seed = int(p.get("seed") if p.get("seed") is not None else 1)
    crf = int(p.get("crf") or 18)
    color_fix = bool(p.get("color_fix", False))
    g = {"1": {"class_type": "LoadVideo", "inputs": {"file": video_ref}},
         "2": {"class_type": "FlashVSRPlusLoader",
               "inputs": {"mode": mode, "profile": "auto", "attn_backend": "auto", "offload": "auto",
                          "decoder": "tiny", "auto_download": False, "model_directory": "", "reload_token": 0}},
         "3": {"class_type": "FlashVSRPlusVideoUpscale",
               "inputs": {"pipe": ["2", 0], "video": ["1", 0], "scale": scale, "height": 0, "width": 0,
                          "temporal_compression": 4, "tiled": "auto", "tile_size": 512, "tile_overlap": 128,
                          "color_fix": color_fix, "color_correction_levels": 5, "seed": seed,
                          "cache_offload": "auto", "decode_every": 4, "vae_tiled": True, "vae_tile_size": 256,
                          "vae_tile_overlap": 64, "oom_retry": True, "crf": crf, "keep_audio": True}},
         "4": {"class_type": "SaveVideo",
               "inputs": {"video": ["3", 0], "filename_prefix": "videogen/flashvsr_plus",
                          "format": "auto", "codec": "auto"}}}
    return g, {"engine": "flashvsr_plus", "mode": mode, "scale": scale, "seed": seed, "crf": crf,
               "color_fix": color_fix, "kind": "video"}


# ================================================================= MiniMax H3 (video + native audio)
# Ported from the user's reference workflow MINIMAX_H3_ULTRA_WORKFLOW.json (Aitrepreneur). Node
# signatures + model filenames VERIFIED live on the box /object_info 2026-08-02; the frame-count and
# autogrow-key rules VERIFIED from ComfyUI source (comfy_extras/nodes_minimax_h3.py and
# comfy_api/latest/_io.py). Plan + provenance: docs/MINIMAX_H3_PLAN.md.
#
# Two conditioning nodes cover all five modes, and BOTH emit (positive CONDITIONING, LATENT) which
# feed one shared tail:
#   MiniMaxH3ImageToVideo      - t2v (no frames) / i2v (first_frame) / FLF (both) / last-frame-only
#   MiniMaxH3ReferenceToVideo  - up to 9 ref images, 3 ref videos (+index-paired soundtracks), 3 audios
# The tail is cfg-FREE (BasicGuider, no CFG) => there is NO negative prompt on this path at all.
H3_FL2VA = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"   # UNETLoader (text + image lanes)
H3_REF2VA = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"  # UNETLoader (reference lane)
H3_CLIP = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"      # CLIPLoader, type "minimax"
H3_VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
H3_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
# The author's settings, verbatim from the workflow (all three lanes use the same numbers).
H3_FPS = 24
H3_STEPS = 20
H3_SAMPLER = "res_multistep"
H3_SCHEDULER = "simple"
# ResolutionSelector 16:9 reference table from the workflow's own note (multiple of 32). 0.9 is the
# author's default. We resolve this in PYTHON because the workflow feeds width/height back into the
# H3 node through rgthree Any Switch nodes - a cycle that cannot exist in an API-format graph.
H3_MEGAPIXELS = {0.2: (608, 352), 0.3: (736, 416), 0.4: (864, 480), 0.5: (960, 544),
                 0.6: (1056, 608), 0.7: (1152, 640), 0.8: (1216, 672), 0.9: (1280, 736),
                 0.98: (1344, 768), 1.0: (1376, 768), 1.2: (1504, 832), 1.5: (1664, 928),
                 1.8: (1824, 1024), 2.0: (1920, 1088)}
H3_MP_DEFAULT = 0.9
# The author ships a TURBO variant of the same workflow (MINIMAX_H3_ULTRA_+TURBO-LORA_WORKFLOW.json).
# Diffing it against the base ULTRA workflow shows FOUR settings change TOGETHER - the LoRA is not a
# drop-in on its own, so we apply them as ONE atomic recipe (`turbo: True`) to make half-applying it
# impossible. VERIFIED by diff 2026-08-02; the 4-step LoRA is present on the box.
H3_TURBO_LORA = "minimax_h3_turbo_4step_ckpt500_comfyui_pruned.safetensors"
H3_TURBO = {"lora": H3_TURBO_LORA, "lora_strength": 1.0,
            "sampler": "euler", "scheduler": "beta", "steps": 8}


def _h3_frames(n):
    """MiniMax H3 wants frame_count % 17 == 5 at 24fps, snapped UP. Mirrors align_frame_count() in
    comfy_extras/nodes_minimax_h3.py - the node snaps internally, so we mirror it to keep the
    frames/seconds we RECORD honest. Trained range is ~124-362 frames (5.17s - 15.08s)."""
    n = max(5, int(n))
    while n % 17 != 5:
        n += 1
    return n


def _h3_size(p):
    """Output size: explicit width/height win; else the megapixel tier (author default 0.9 = 1280x736).
    Both are snapped to /32 because the model's latent grid requires it."""
    if p.get("width") and p.get("height"):
        w, h = int(p["width"]), int(p["height"])
    else:
        mp = float(p.get("megapixels", H3_MP_DEFAULT))
        w, h = H3_MEGAPIXELS.get(mp) or H3_MEGAPIXELS[H3_MP_DEFAULT]
    return max(32, (w // 32) * 32), max(32, (h // 32) * 32)


def _h3_recipe(p):
    """Resolve sampler / scheduler / steps / LoRA. `turbo: True` applies the author's ULTRA+TURBO
    recipe VERBATIM (all four values at once); explicit per-request values still win over it."""
    r = {"sampler": H3_SAMPLER, "scheduler": H3_SCHEDULER, "steps": H3_STEPS,
         "lora": None, "lora_strength": 1.0}
    if p.get("turbo"):
        r.update(H3_TURBO)
    for k in ("sampler", "scheduler", "steps", "lora", "lora_strength"):
        if p.get(k) not in (None, ""):
            r[k] = p[k]
    r["steps"] = int(r["steps"])
    r["lora_strength"] = float(r["lora_strength"])
    return r


def _h3_tail(g, unet_name, cond_node, p, seed, frames, fps, prefix):
    """The shared chain every H3 lane uses, traced from the reference workflow's subgraphs:
      UNETLoader -> [SigmaShift -> SpectrumApply] -> BasicGuider
                                                 -> BasicScheduler (fed the PATCHED model)
      SamplerCustomAdvanced(RandomNoise, guider, KSamplerSelect, sigmas, latent)
        -> VAEDecode(video vae)     -> IMAGE
        -> VAEDecodeAudio(audio vae) -> AUDIO   (same joint AV latent, decoded twice)
      -> VHS_VideoCombine
    NOTE: PathchSageAttentionKJ ("auto") is applied IN-GRAPH, per the author's note to patch in-graph
    when the launcher argument is absent. --use-sage-attention came OFF the launcher 2026-08-09:
    launcher-wide sage silently drops attention masks, which NaN'd every short-prompt Krea2 render
    to black (and broke LTX PromptRelay back in June the same way). Sage is now H3-graph-only.
    `spectrum` (default OFF) adds the author's SPEEDUP group at its node defaults; it is a separate
    toggleable group in his workflow, so we keep it opt-in and A/B it rather than baking it in."""
    rec = _h3_recipe(p)
    steps = rec["steps"]
    g["1"] = {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "default"}}
    model = ["1", 0]
    # Sage attention for H3 only (author's node, his shipped setting "auto"); see the note above.
    g["8"] = {"class_type": "PathchSageAttentionKJ",
              "inputs": {"model": model, "sage_attention": "auto"}}
    model = ["8", 0]
    # LoRA sits immediately after the UNET, as in BOTH reference workflows (ULTRA puts an rgthree
    # Power Lora Loader there; the Director/scout workflow uses stock LoraLoaderModelOnly). We use the
    # stock node because the rgthree one carries a UI-shaped dict widget with no API-format equivalent.
    if rec["lora"]:
        g["4"] = {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"model": model, "lora_name": rec["lora"],
                             "strength_model": rec["lora_strength"]}}
        model = ["4", 0]
    if p.get("spectrum"):
        g["2"] = {"class_type": "MiniMaxH3SigmaShift",
                  "inputs": {"model": model,
                             "shift_video": float(p.get("shift_video", 12.0)),
                             "shift_audio": float(p.get("shift_audio", 3.0))}}
        g["3"] = {"class_type": "SpectrumApplyMiniMaxH3",
                  "inputs": {"model": ["2", 0], "enabled": True, "blend_weight": 0.5, "degree": 1,
                             "ridge_lambda": 0.1, "window_size": 2.0, "flex_window": 0.75,
                             "warmup_steps": 1, "tail_actual_steps": 1, "max_history": 8,
                             "debug": False, "history_storage": "system_ram",
                             "bootstrap_first_forecast": True}}
        model = ["3", 0]
    g["10"] = {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": [cond_node, 0]}}
    g["11"] = {"class_type": "BasicScheduler",
               "inputs": {"model": model, "scheduler": rec["scheduler"], "steps": steps, "denoise": 1.0}}
    g["12"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": rec["sampler"]}}
    g["13"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    g["14"] = {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["13", 0], "guider": ["10", 0], "sampler": ["12", 0],
                          "sigmas": ["11", 0], "latent_image": [cond_node, 1]}}
    g["15"] = {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["6", 0]}}
    g["16"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["14", 0], "vae": ["7", 0]}}
    g["17"] = {"class_type": "VHS_VideoCombine",
               "inputs": {"images": ["15", 0], "audio": ["16", 0], "frame_rate": fps,
                          "loop_count": 0, "filename_prefix": prefix,
                          "format": "video/h264-mp4", "pingpong": False, "save_output": True}}
    return steps


def _h3_loaders(g):
    """CLIP (type "minimax") + the two VAEs, shared by every lane."""
    g["5"] = {"class_type": "CLIPLoader",
              "inputs": {"clip_name": H3_CLIP, "type": "minimax", "device": "default"}}
    g["6"] = {"class_type": "VAELoader", "inputs": {"vae_name": H3_VIDEO_VAE}}
    g["7"] = {"class_type": "VAELoader", "inputs": {"vae_name": H3_AUDIO_VAE}}


def build_h3_t2v(p):
    """MiniMax H3 TEXT to video+audio (the reference workflow's TEXT TO VIDEO lane, FL2VA model).
    p: {prompt, seed?, width?/height? or megapixels?(0.9), frames?(124) or seconds?, steps?(20),
    spectrum?(False)}. The prompt must be the author's sectioned format - see docs/MINIMAX_H3_PLAN.md
    section 4 (integrated_multimodal_description / overall_soundscape / non_diegetic_music).
    There is NO negative prompt on this path (BasicGuider, cfg-free)."""
    seed = _seed(p)
    w, h = _h3_size(p)
    fps = int(p.get("fps", H3_FPS))
    frames = _h3_frames(p.get("frames") or round(float(p.get("seconds", 5.0)) * fps))
    prompt = (p.get("prompt") or "").strip()
    g = {}
    _h3_loaders(g)
    g["20"] = {"class_type": "MiniMaxH3ImageToVideo",
               "inputs": {"clip": ["5", 0], "vae": ["6", 0], "prompt": prompt,
                          "width": w, "height": h, "length": frames}}
    steps = _h3_tail(g, H3_FL2VA, "20", p, seed, frames, fps, "videogen/h3t2v")
    rec = _h3_recipe(p)
    return g, {"seed": seed, "width": w, "height": h, "frames": frames, "fps": fps,
               "seconds": round(frames / fps, 2), "steps": steps, "prompt": prompt,
               "engine": "minimax_h3", "lane": "t2v", "model": H3_FL2VA,
               "spectrum": bool(p.get("spectrum")), "turbo": bool(p.get("turbo")),
               "sampler": rec["sampler"], "scheduler": rec["scheduler"],
               "lora": rec["lora"], "lora_strength": rec["lora_strength"] if rec["lora"] else None,
               "kind": "video"}

def build_h3_i2v(p, first_ref, last_ref=None):
    """MiniMax H3 IMAGE to video+audio - a port of the reference workflow's IMAGE TO VIDEO lane
    (FL2VA model). Covers three of the node's modes depending on what you pass:
      first_ref only            -> image-to-video (animate a still)
      first_ref + last_ref      -> first-last-frame (interpolate between two stills)
      last_ref only             -> last-frame-only (converge onto a still) - pass first_ref=None
    Faithful details from the workflow:
      - the FIRST frame is fed RAW (LoadImage -> first_frame);
      - the LAST frame goes through ImageScaleToTotalPixels(nearest-exact, 1.0, 32) first;
      - output size comes from the FIRST image via its "IMAGE SIZE" group
        (ImageScaleToTotalPixels(megapixels) -> GetImageSize), which preserves the STILL's aspect
        ratio instead of forcing 16:9. Pass explicit width/height to override with the manual path
        (the workflow's "IMAGE MANUAL RESOLUTION" group).
    p: as build_h3_t2v, plus {img_megapixels? (sizing tier when deriving from the image)}.
    Prompt must be the sectioned format, and for i2v it must OPEN with the author's exact line:
      "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is
      fully referenced." followed by a blank line. See docs/MINIMAX_H3_PLAN.md section 4."""
    seed = _seed(p)
    fps = int(p.get("fps", H3_FPS))
    frames = _h3_frames(p.get("frames") or round(float(p.get("seconds", 5.0)) * fps))
    prompt = (p.get("prompt") or "").strip()
    manual = bool(p.get("width") and p.get("height"))
    w, h = _h3_size(p)
    g = {}
    _h3_loaders(g)
    node = {"clip": ["5", 0], "vae": ["6", 0], "prompt": prompt, "length": frames}
    if first_ref:
        g["30"] = {"class_type": "LoadImage", "inputs": {"image": first_ref}}
        node["first_frame"] = ["30", 0]
    if last_ref:
        g["31"] = {"class_type": "LoadImage", "inputs": {"image": last_ref}}
        if p.get("last_raw"):
            # opt-in: feed the last frame RAW like the first one (skips the author's nearest-exact
            # 1.0MP rescale). For same-still loops this keeps both anchors pixel-identical.
            node["last_frame"] = ["31", 0]
        else:
            g["32"] = {"class_type": "ImageScaleToTotalPixels",
                       "inputs": {"image": ["31", 0], "upscale_method": "nearest-exact",
                                  "megapixels": 1.0, "resolution_steps": 32}}
            node["last_frame"] = ["32", 0]
    if manual or not first_ref:
        node["width"], node["height"] = w, h          # manual path (or nothing to derive from)
    else:
        # the workflow's IMAGE SIZE group: scale the first still to the target megapixels on a /32
        # grid, then read its dimensions back. Keeps the still's aspect ratio.
        g["33"] = {"class_type": "ImageScaleToTotalPixels",
                   "inputs": {"image": ["30", 0], "upscale_method": "nearest-exact",
                              "megapixels": float(p.get("img_megapixels", H3_MP_DEFAULT)),
                              "resolution_steps": 32}}
        g["34"] = {"class_type": "GetImageSize", "inputs": {"image": ["33", 0]}}
        node["width"], node["height"] = ["34", 0], ["34", 1]
    g["20"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": node}
    steps = _h3_tail(g, H3_FL2VA, "20", p, seed, frames, fps, "videogen/h3i2v")
    rec = _h3_recipe(p)
    lane = ("flf" if (first_ref and last_ref) else "i2v" if first_ref else "lastframe")
    return g, {"seed": seed, "frames": frames, "fps": fps, "seconds": round(frames / fps, 2),
               "steps": steps, "prompt": prompt, "engine": "minimax_h3", "lane": lane,
               "model": H3_FL2VA, "spectrum": bool(p.get("spectrum")), "turbo": bool(p.get("turbo")),
               "sampler": rec["sampler"], "scheduler": rec["scheduler"],
               "lora": rec["lora"], "lora_strength": rec["lora_strength"] if rec["lora"] else None,
               "width": w if (manual or not first_ref) else None,
               "height": h if (manual or not first_ref) else None,
               "size_from_image": not (manual or not first_ref), "kind": "video"}
