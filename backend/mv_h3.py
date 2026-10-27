"""MiniMax H3 music-video SCENE CHAIN builder (docs/MV_H3_REBUILD_PLAN.md, sections 2-3 and 8).

One scene = one continuous chain of H3 Ref2VA clips over one window of the song:
  - the song is LOCKED in every clip's audio latent (written in, noise mask 0) so H3 never
    generates or re-renders it - the fix for the per-seed audio drift we measured in August;
  - clip N copies clip N-1's video-latent tail (39 frames = 1.625 s = exactly 65 audio ticks)
    as a protected prefix, so motion, camera and lighting continue across the join;
  - the streamer decodes clip by clip, crossfades the 39-frame seams, and muxes the untouched
    song window underneath.

The graph is a port of seitanism's "NEW - Music Video.json" (ComfyUI-H3-Motion-Context-MultiRef,
Update 9) - traced link by link on 2026-10-27, not invented:
  UNETLoader(ref2va) -> ModelAttentionBackend -> LoraLoaderModelOnly -> MiniMaxH3SigmaShift
  per clip:  MiniMaxH3ReferenceToVideo(clip, video vae, audio vae*, prompt, refs, w, h, length)
             -> MiniMaxH3SongMaskedAVContext(latent, audio vae*, master audio, clip_start,
                context 39, source_latent = previous clip's SAMPLER output)
             -> BasicGuider(model, positive) + SamplerCustomAdvanced(noise, guider, sampler,
                sigmas, latent)
  MiniMaxH3StreamLiveMusicVideoToVHS(video vae, master audio, every sampler output)
  -> MiniMaxH3FinalizeVHSOutput
  * the audio VAE is routed through MiniMaxH3AudioVAECompatibility once, as the example does.

Two deliberate differences from the example, both because the app drives the box headless:
  - no per-clip VHS previews (they would be extra VAE decodes, and on_complete_media files the
    FIRST media output it finds - the final movie must be the only one);
  - the streamer muxes its master audio from sample 0, so the caller hands this builder a master
    that is already CUT to the chain's song window; clip starts are then 0, advance, 2*advance
    inside it (the same arithmetic as the example, which starts at song time 0).

Plain ASCII only.
"""
import subprocess

from .video import (H3_AUDIO_VAE, H3_CLIP, H3_FPS, H3_REF2VA, H3_SAMPLER, H3_SCHEDULER, H3_STEPS,
                    H3_VIDEO_VAE, _h3_frames, _h3_size, _seed)

# The example's globals: 15 s raw clips (362 f), a 39-frame protected prefix, 0.5 MP 16:9.
CHAIN_FRAMES = 362
CHAIN_CONTEXT = 39
CHAIN_MP = 0.5                      # 960x544 in video.H3_MEGAPIXELS
CHAIN_SHIFT = (12.0, 3.0)           # SigmaShift video / audio, H3 defaults
CHAIN_CRF = 19
CHAIN_ATTENTION = "comfy kitchen attention"   # ModelAttentionBackend, as the example; torch is cu130
# Fast recipe for drafts/hunts. The example ships the lightx2v FL2VA 8-step v1.0 LoRA at 8 steps on
# res_multistep/simple; we stage the Ref2VA 8-step v1.1 768p sibling (research 5.1), whose 768p
# training wants sigma shift 6/3 instead of 12/3. All five values travel together.
CHAIN_TURBO = {"lora": "minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors",
               "lora_strength": 1.0, "sampler": "res_multistep", "scheduler": "simple", "steps": 8,
               "shift_video": 6.0, "shift_audio": 3.0}
# Hunt drafts take base_seed + 1, + 2 ... (app._h3_dispatch), so clip seeds inside one chain step
# by a larger stride to keep every (draft, clip) seed distinct.
CLIP_SEED_STRIDE = 101
MAX_CLIPS = 20                      # the streamer's clip_1 .. clip_20 sockets


def chain_plan(song_start, n_clips, frames=CHAIN_FRAMES, context=CHAIN_CONTEXT, fps=H3_FPS):
    """Exact clip windows for a chain. All arithmetic in FRAMES first (the node snaps the frame
    count to 17k+5; a continuation advances by frames - context), seconds derived from that, and
    the song time of every clip = song_start + its start inside the cut master window."""
    frames = _h3_frames(frames)
    n = max(1, min(int(n_clips), MAX_CLIPS))
    context = int(context)
    if not 0 < context < frames:
        raise ValueError(f"context {context} must be inside the clip ({frames} frames)")
    advance = frames - context
    clips = []
    for i in range(n):
        start_f = i * advance
        clips.append({"index": i, "start_frame": start_f,
                      "clip_start": round(start_f / fps, 6),
                      "song_start": round(song_start + start_f / fps, 3),
                      "song_end": round(song_start + (start_f + frames) / fps, 3)})
    span_f = frames + (n - 1) * advance
    return {"frames": frames, "seconds": round(frames / fps, 3), "context_frames": context,
            "advance_frames": advance, "advance_seconds": round(advance / fps, 6),
            "span_frames": span_f, "span_seconds": round(span_f / fps, 3),
            "song_start": round(float(song_start), 3),
            "song_end": round(float(song_start) + span_f / fps, 3), "clips": clips}


def song_window_wav(path, start, seconds):
    """The master song cut to [start, start + seconds] as 16-bit STEREO WAV at the source rate
    (ffmpeg, no GPU). Stereo on purpose: the song node encodes what it is given and the H3 audio
    VAE is stereo. The caller adds ~1 s of tail so the node's audio-grid lookahead never runs out
    of real audio (it refuses to pad with silence)."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{max(0.0, float(start)):.6f}",
           "-t", f"{float(seconds):.6f}", "-i", path, "-vn", "-ac", "2", "-c:a", "pcm_s16le",
           "-f", "wav", "pipe:1"]
    out = subprocess.run(cmd, capture_output=True, timeout=120)
    if out.returncode != 0 or len(out.stdout) < 1000:
        raise RuntimeError(f"ffmpeg song window failed: {out.stderr.decode(errors='replace')[:300]}")
    return out.stdout


def chain_recipe(p):
    """sampler / scheduler / steps / LoRA / sigma shift for a chain. `turbo: True` applies
    CHAIN_TURBO as one atomic block (the LoRA and its shift are not drop-ins on their own);
    explicit per-request values still win over it."""
    r = {"sampler": H3_SAMPLER, "scheduler": H3_SCHEDULER, "steps": H3_STEPS,
         "lora": None, "lora_strength": 1.0,
         "shift_video": CHAIN_SHIFT[0], "shift_audio": CHAIN_SHIFT[1]}
    if p.get("turbo"):
        r.update(CHAIN_TURBO)
    for k in ("sampler", "scheduler", "steps", "lora", "lora_strength", "shift_video", "shift_audio"):
        if p.get(k) not in (None, ""):
            r[k] = p[k]
    r["steps"] = int(r["steps"])
    r["lora_strength"] = float(r["lora_strength"])
    r["shift_video"] = float(r["shift_video"])
    r["shift_audio"] = float(r["shift_audio"])
    return r


def build_h3_chain(p, ref_names, master_name):
    """One scene chain as an API graph.

    p: {clips: [{prompt}, ...] (1-20, the six-section Ref2VA prompts, one per clip),
        song_start (seconds into the song where clip 1 begins - informational here, the caller
        has already cut `master_name` to start there), frames? (362) or seconds?, width?/height?
        or megapixels? (0.5 = 960x544), context_frames? (39), seed?, turbo?, sampler?, scheduler?,
        steps?, lora?, lora_strength?, shift_video?, shift_audio?, attention? ("comfy kitchen
        attention" | "pytorch attention" | ...), ref_image_size? ("match" | "max"), crf? (19),
        prefix?}
    ref_names: ComfyUI upload names of the <Picture 1..9> references, in prompt order.
    master_name: ComfyUI upload name of the song window (stereo WAV), starting at song_start.
    Returns (graph, resolved) like every builder in video.py."""
    clips = [c for c in (p.get("clips") or []) if (c.get("prompt") or "").strip()]
    if not clips:
        raise ValueError("at least one clip with a prompt is required")
    if len(clips) > MAX_CLIPS:
        raise ValueError(f"at most {MAX_CLIPS} clips per chain")
    if not ref_names:
        raise ValueError("at least one reference image is required (Ref2VA)")
    if len(ref_names) > 9:
        raise ValueError("H3 takes at most 9 reference images")
    seed = _seed(p)
    fps = int(p.get("fps", H3_FPS))
    frames = _h3_frames(p.get("frames") or round(float(p.get("seconds", CHAIN_FRAMES / fps)) * fps))
    plan = chain_plan(float(p.get("song_start") or 0.0), len(clips), frames,
                      int(p.get("context_frames") or CHAIN_CONTEXT), fps)
    q = dict(p)
    q.setdefault("megapixels", CHAIN_MP)
    w, h = _h3_size(q)
    rec = chain_recipe(p)
    attention = (p.get("attention") or CHAIN_ATTENTION).strip()
    ref_image_size = (p.get("ref_image_size") or "match").strip()
    prefix = p.get("prefix") or "videogen/h3chain"

    g = {}
    # --- shared loaders and the patched model (ids follow the example's roles)
    g["1"] = {"class_type": "UNETLoader", "inputs": {"unet_name": H3_REF2VA, "weight_dtype": "default"}}
    model = ["1", 0]
    if attention and attention != "none":
        g["2"] = {"class_type": "ModelAttentionBackend", "inputs": {"model": model, "attention": attention}}
        model = ["2", 0]
    if rec["lora"]:
        g["4"] = {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"model": model, "lora_name": rec["lora"], "strength_model": rec["lora_strength"]}}
        model = ["4", 0]
    g["3"] = {"class_type": "MiniMaxH3SigmaShift",
              "inputs": {"model": model, "shift_video": rec["shift_video"], "shift_audio": rec["shift_audio"]}}
    model = ["3", 0]
    g["5"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": H3_CLIP, "type": "minimax", "device": "default"}}
    g["6"] = {"class_type": "VAELoader", "inputs": {"vae_name": H3_VIDEO_VAE}}
    g["7"] = {"class_type": "VAELoader", "inputs": {"vae_name": H3_AUDIO_VAE}}
    g["9"] = {"class_type": "MiniMaxH3AudioVAECompatibility", "inputs": {"audio_vae": ["7", 0]}}
    g["11"] = {"class_type": "BasicScheduler",
               "inputs": {"model": model, "scheduler": rec["scheduler"], "steps": rec["steps"], "denoise": 1.0}}
    g["12"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": rec["sampler"]}}
    g["40"] = {"class_type": "LoadAudio", "inputs": {"audio": master_name}}
    for i, name in enumerate(ref_names):
        g[str(30 + i)] = {"class_type": "LoadImage", "inputs": {"image": name}}

    # --- the clips
    prev_sampler = None
    samplers = []
    seeds = []
    for i, (c, win) in enumerate(zip(clips, plan["clips"])):
        base = 100 + 10 * i
        r2v, song, guider, noise, sampler = (str(base), str(base + 1), str(base + 2), str(base + 3), str(base + 4))
        r2v_in = {"clip": ["5", 0], "vae": ["6", 0], "audio_vae": ["9", 0],
                  "prompt": c["prompt"].strip(), "width": w, "height": h, "length": frames,
                  "ref_image_size": ref_image_size}
        for k in range(len(ref_names)):
            r2v_in[f"ref_images.ref_image_{k}"] = [str(30 + k), 0]
        g[r2v] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": r2v_in}
        song_in = {"latent": [r2v, 1], "audio_vae": ["9", 0], "master_audio": ["40", 0],
                   "clip_start_seconds": float(win["clip_start"]),
                   "context_length": int(plan["context_frames"]), "source_fps": float(fps),
                   "crop": "disabled"}
        if prev_sampler is not None:
            song_in["source_latent"] = [prev_sampler, 0]
        g[song] = {"class_type": "MiniMaxH3SongMaskedAVContext", "inputs": song_in}
        g[guider] = {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": [r2v, 0]}}
        clip_seed = int(c.get("seed") if c.get("seed") not in (None, "") else seed + CLIP_SEED_STRIDE * i)
        g[noise] = {"class_type": "RandomNoise", "inputs": {"noise_seed": clip_seed}}
        g[sampler] = {"class_type": "SamplerCustomAdvanced",
                      "inputs": {"noise": [noise, 0], "guider": [guider, 0], "sampler": ["12", 0],
                                 "sigmas": ["11", 0], "latent_image": [song, 0]}}
        prev_sampler = sampler
        samplers.append(sampler)
        seeds.append(clip_seed)

    # --- final stream: every sampler latent in order, master song underneath
    # input_count = how many clip_N sockets the node exposes (a UI stepper in the editor; a plain
    # int here), active_clips = the cap it honours. Both = the chain length.
    stream_in = {"video_vae": ["6", 0], "master_audio": ["40", 0], "input_count": len(clips),
                 "active_clips": len(clips), "context_frames": int(plan["context_frames"]),
                 "video_overlap_frames": int(plan["context_frames"]),
                 "filename_prefix": prefix, "pix_fmt": "yuv420p", "crf": int(p.get("crf") or CHAIN_CRF),
                 "save_metadata": False, "trim_to_audio": False, "save_output": True}
    for i, s in enumerate(samplers):
        stream_in[f"clip_{i + 1}"] = [s, 0]
    g["50"] = {"class_type": "MiniMaxH3StreamLiveMusicVideoToVHS", "inputs": stream_in}
    g["51"] = {"class_type": "MiniMaxH3FinalizeVHSOutput", "inputs": {"filenames": ["50", 0]}}

    resolved = {"seed": seed, "clip_seeds": seeds, "clips": len(clips), "frames": frames, "fps": fps,
                "seconds": plan["seconds"], "span_seconds": plan["span_seconds"],
                "song_start": plan["song_start"], "song_end": plan["song_end"], "plan": plan,
                "width": w, "height": h, "steps": rec["steps"], "sampler": rec["sampler"],
                "scheduler": rec["scheduler"], "lora": rec["lora"],
                "lora_strength": rec["lora_strength"] if rec["lora"] else None,
                "shift_video": rec["shift_video"], "shift_audio": rec["shift_audio"],
                "attention": attention, "ref_image_size": ref_image_size, "ref_count": len(ref_names),
                "turbo": bool(p.get("turbo")), "prompts": [c["prompt"].strip() for c in clips],
                "engine": "minimax_h3", "lane": "chain", "model": H3_REF2VA, "kind": "video"}
    return g, resolved
