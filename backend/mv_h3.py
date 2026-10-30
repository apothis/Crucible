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

from .video import (H3_AUDIO_VAE, H3_CLIP, H3_FL2VA, H3_FPS, H3_REF2VA, H3_SAMPLER, H3_SCHEDULER,
                    H3_STEPS, H3_VIDEO_VAE, _h3_frames, _h3_size, _seed)

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
# The same recipe for the FL2VA (no-reference) lane: environment-only B-roll scenes carry no
# <Picture> at all, so they render on the fl2va weights with the matching lightx2v LoRA.
CHAIN_TURBO_FL2VA = dict(CHAIN_TURBO, lora="minimax_h3_fl2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors")
# Hunt drafts take base_seed + 1, + 2 ... (app._h3_dispatch), so clip seeds inside one chain step
# by a larger stride to keep every (draft, clip) seed distinct.
CLIP_SEED_STRIDE = 101
MAX_CLIPS = 20                      # the streamer's clip_1 .. clip_20 sockets
# The settings every passed take was rendered on (24 SSBS locks, ENV9/ENV10 real-speed plates,
# 2026-10-28/29): the turbo recipe at 1344x768. A batch dry run flags any scene that would render
# on something else (s.20.48: a fire that dropped them ran base 20-step 960x544 on every scene).
PROVEN = {"turbo": True, "width": 1344, "height": 768}
# Latent upscale second pass (opt-in, s.20.50), a port of LBH-123-AI's own example graphs
# (Comfyui_Minimax_h3_latent_Upscaler/workflow_templates, i2v "workflow2" and r2v "workflow3"):
# the finished take's AV latent -> LTXVSeparateAVLatent -> MinimaxH3LatentUpscaler3D (video only,
# target dimensions, align 32) -> LTXVConcatAVLatent -> a short re-sample on the example's manual
# sigmas with the euler sampler. The r2v example reuses the first pass's conditioning; the i2v
# example builds a second MiniMaxH3ImageToVideo at the target size. One addition of ours: the song
# is written back into the upscaled latent (MiniMaxH3SongMaskedAVContext) so the second pass keeps
# it locked like the first; the example has no song lock.
UPSCALE_MODEL = "minimax_h3_latent_upscaler_3d_fp16.safetensors"
UPSCALE_SIGMAS = "0.9035, 0.6316, 0.3158, 0.0000"
UPSCALE_SAMPLER = "euler"


def chain_settings(p, lane="ref2va"):
    """The recipe, size and seed build_h3_chain WILL use for request p on lane "ref2va" | "fl2va",
    without building a graph. A missing seed is drawn here so the caller can pin it."""
    q = dict(p)
    q.setdefault("megapixels", CHAIN_MP)
    w, h = _h3_size(q)
    rec = chain_recipe(p, lane)
    out = {"turbo": bool(p.get("turbo")), "steps": rec["steps"], "shift_video": rec["shift_video"],
           "lora": rec["lora"], "width": w, "height": h, "seed": _seed(p)}
    if p.get("upscale_width") or p.get("upscale_height"):
        out["upscale"] = [int(p.get("upscale_width") or 0), int(p.get("upscale_height") or 0)]
    return out


def settings_flags(s):
    """How settings `s` (chain_settings) differ from PROVEN, as readable flags; empty = proven."""
    out = []
    if bool(s.get("turbo")) != PROVEN["turbo"]:
        out.append(f"turbo {'on' if s.get('turbo') else 'off'} ({s.get('steps')} steps, shift {s.get('shift_video')}); "
                   f"passed takes used the turbo recipe")
    if (s.get("width"), s.get("height")) != (PROVEN["width"], PROVEN["height"]):
        out.append(f"{s.get('width')}x{s.get('height')}, not the proven {PROVEN['width']}x{PROVEN['height']}")
    return out


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
    of real audio (it refuses to pad with silence itself). When the window runs past the end of
    the file (a final scene whose chain overshoots the song), the output is padded with digital
    silence to the full `seconds`: to the node that is a real waveform, and the overshoot is
    trimmed at assembly anyway."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{max(0.0, float(start)):.6f}",
           "-t", f"{float(seconds):.6f}", "-i", path, "-vn", "-ac", "2",
           "-af", f"apad=whole_dur={float(seconds):.6f}", "-t", f"{float(seconds):.6f}",
           "-c:a", "pcm_s16le", "-f", "wav", "pipe:1"]
    out = subprocess.run(cmd, capture_output=True, timeout=120)
    if out.returncode != 0 or len(out.stdout) < 1000:
        raise RuntimeError(f"ffmpeg song window failed: {out.stderr.decode(errors='replace')[:300]}")
    return out.stdout


def chain_recipe(p, lane="ref2va"):
    """sampler / scheduler / steps / LoRA / sigma shift for a chain. `turbo: True` applies
    CHAIN_TURBO (or its FL2VA twin) as one atomic block (the LoRA and its shift are not drop-ins
    on their own); explicit per-request values still win over it."""
    r = {"sampler": H3_SAMPLER, "scheduler": H3_SCHEDULER, "steps": H3_STEPS,
         "lora": None, "lora_strength": 1.0,
         "shift_video": CHAIN_SHIFT[0], "shift_audio": CHAIN_SHIFT[1]}
    if p.get("turbo"):
        r.update(CHAIN_TURBO_FL2VA if lane == "fl2va" else CHAIN_TURBO)
    for k in ("sampler", "scheduler", "steps", "lora", "lora_strength", "shift_video", "shift_audio"):
        if p.get(k) not in (None, ""):
            r[k] = p[k]
    r["steps"] = int(r["steps"])
    r["lora_strength"] = float(r["lora_strength"])
    r["shift_video"] = float(r["shift_video"])
    r["shift_audio"] = float(r["shift_audio"])
    return r


def build_h3_chain(p, ref_names, master_name, lock_name=None, ref_video_names=None, first_frame_name=None,
                   anchor_name=None):
    """One scene chain as an API graph.

    p: {clips: [{prompt}, ...] (1-20, the six-section Ref2VA prompts, one per clip),
        song_start (seconds into the song where clip 1 begins - informational here, the caller
        has already cut `master_name` to start there), frames? (362) or seconds?, width?/height?
        or megapixels? (0.5 = 960x544), context_frames? (39), seed?, turbo?, sampler?, scheduler?,
        steps?, lora?, lora_strength?, shift_video?, shift_audio?, attention? ("comfy kitchen
        attention" | "pytorch attention" | ...), ref_image_size? ("match" | "max"), crf? (19),
        prefix?, keyframes? (None [default] | "still_first_last" | "still_first" | "still_last":
        anchor the last reference picture, the location still, at those frames of every clip
        via MiniMaxH3AddGuide; environment plates only, it pins whatever the still shows),
        free_av? (False [default]; True = no MiniMaxH3SongMaskedAVContext: the model generates its
        own audio and the streamer muxes the master over it; single-clip environment plates only,
        the song is laid in at assembly anyway), clock_fps? (None [default] = the model's native
        24; a value 12-120 inserts MiniMaxH3NativeFPSAVSyncAc (AharaOoO pack) after the guides and
        before the guider: same frames, video time coordinates scaled by 40/fps so the model
        believes the clip spans frames/fps seconds; saved at 24 fps a value above 24 reads as
        slow motion with generated frames [H, direction inferred from the source, s.20.15]; the
        node resizes the audio latent, so it requires free_av),
        lane? (None [default] = ref2va when any picture, else fl2va text-to-video; "i2v" = the
        image-to-video lane: FL2VA weights + their turbo LoRA, `first_frame_name` fed raw as the
        node's first_frame (the place arrives as pixels to continue, not as a subject to retain),
        NO reference pictures, prompts in the three-section FL2VA format; s.20.18),
        upscale_width? / upscale_height? (None [default] = off; both set = the latent upscale second
        pass at that size, multiples of 32, see UPSCALE_MODEL; single-clip chains only, s.20.50)}
    first_frame_name: upload name of the first-frame still for lane "i2v" (the location still).
    anchor_name: optional upload of the scene's OPENING still (her already in place) anchored at
        frame 0 of the FIRST clip only, through MiniMaxH3AddGuide, on the reference lane (opt-in,
        s.20.36). The references stay as they are for identity and lip-sync; this is pixel
        guidance for the first frame, not a <Picture n>, so the prompt is unchanged. Later clips
        continue from the previous clip's context frames and are not anchored. Not combinable
        with keyframes still_first / still_first_last (two guides on the same frame).
    ref_names: ComfyUI upload names of the <Picture 1..9> references, in prompt order.
        For a scripted scene that is face, body, then the person-free LOCATION still (the place
        as <Subject 2>/<Picture 3>); an environment-only scene passes the location still alone
        (<Picture 1>). Any list of pictures is fine here; the prompt decides what they mean.
    master_name: ComfyUI upload name of the song window (stereo WAV), starting at song_start.
    lock_name: optional upload of a DIFFERENT audio cut to the same window (vocal stem, or the mix
        with the vocals forward) to lock in the clip latents; the master is still what is muxed.
    ref_video_names: optional uploads (mp4 in the input folder) used as <Video 1..3> references,
        e.g. a manufactured camera-pace clip; Ref2VA lane only (the node has three video slots).
    Returns (graph, resolved) like every builder in video.py."""
    clips = [c for c in (p.get("clips") or []) if (c.get("prompt") or "").strip()]
    if not clips:
        raise ValueError("at least one clip with a prompt is required")
    if len(clips) > MAX_CLIPS:
        raise ValueError(f"at most {MAX_CLIPS} clips per chain")
    ref_names = list(ref_names or [])
    lane = "ref2va" if ref_names else "fl2va"     # no references at all -> environment B-roll on FL2VA
    i2v = (p.get("lane") or "").strip() == "i2v"
    if i2v:
        if ref_names:
            raise ValueError("lane i2v takes no reference pictures; the still is the first frame")
        if not first_frame_name:
            raise ValueError("lane i2v needs a first-frame still")
        lane = "fl2va"
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
    rec = chain_recipe(p, lane)
    unet = H3_REF2VA if lane == "ref2va" else H3_FL2VA
    attention = (p.get("attention") or CHAIN_ATTENTION).strip()
    ref_image_size = (p.get("ref_image_size") or "match").strip()
    prefix = p.get("prefix") or "videogen/h3chain"
    keyframes = (p.get("keyframes") or "").strip() or None
    keyframe_idx = {"still_first_last": (0, -1), "still_first": (0,), "still_last": (-1,)}.get(keyframes)
    if keyframes and keyframe_idx is None:
        raise ValueError("keyframes must be still_first_last, still_first or still_last")
    free_av = bool(p.get("free_av"))
    clock_fps = p.get("clock_fps")
    clock_fps = float(clock_fps) if clock_fps not in (None, "", 0, "0") else None
    if clock_fps is not None and not (12.0 <= clock_fps <= 120.0):
        raise ValueError("clock_fps must be between 12 and 120 (24 = native)")
    if clock_fps is not None and not free_av:
        raise ValueError("clock_fps resizes the audio latent, so the song cannot ride in it: set free_av true")
    if free_av and len(clips) > 1:
        raise ValueError("free_av is for single-clip plates (no song context to chain the clips on)")
    if anchor_name and (i2v or lane != "ref2va"):
        raise ValueError("the frame-0 anchor is a reference-lane guide; lane i2v pins the first frame by itself")
    if anchor_name and keyframes in ("still_first", "still_first_last"):
        raise ValueError("anchor and keyframes still_first both guide frame 0; choose one")
    up_w, up_h = int(p.get("upscale_width") or 0), int(p.get("upscale_height") or 0)
    upscale = bool(up_w or up_h)
    if upscale:
        if not (up_w and up_h) or up_w % 32 or up_h % 32:
            raise ValueError("upscale_width and upscale_height must both be set, multiples of 32")
        if len(clips) > 1:
            raise ValueError("the latent upscale pass is for single-clip chains")
        if free_av or clock_fps is not None:
            raise ValueError("the latent upscale pass keeps the song locked; it does not take free_av or clock_fps")

    g = {}
    # --- shared loaders and the patched model (ids follow the example's roles)
    g["1"] = {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}}
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
    lock_audio = ["40", 0]
    if lock_name and lock_name != master_name:
        g["41"] = {"class_type": "LoadAudio", "inputs": {"audio": lock_name}}
        lock_audio = ["41", 0]
    for i, name in enumerate(ref_names):
        g[str(30 + i)] = {"class_type": "LoadImage", "inputs": {"image": name}}
    if i2v:
        g["30"] = {"class_type": "LoadImage", "inputs": {"image": first_frame_name}}
    if anchor_name:
        g["29"] = {"class_type": "LoadImage", "inputs": {"image": anchor_name}}
    ref_video_names = list(ref_video_names or [])[:3]
    for k, name in enumerate(ref_video_names):
        g[str(60 + 2 * k)] = {"class_type": "LoadVideo", "inputs": {"file": name}}
        g[str(61 + 2 * k)] = {"class_type": "GetVideoComponents", "inputs": {"video": [str(60 + 2 * k), 0]}}

    # --- the clips
    prev_sampler = None
    samplers = []
    seeds = []
    for i, (c, win) in enumerate(zip(clips, plan["clips"])):
        base = 100 + 10 * i
        r2v, song, guider, noise, sampler = (str(base), str(base + 1), str(base + 2), str(base + 3), str(base + 4))
        if lane == "ref2va":
            r2v_in = {"clip": ["5", 0], "vae": ["6", 0], "audio_vae": ["9", 0],
                      "prompt": c["prompt"].strip(), "width": w, "height": h, "length": frames,
                      "ref_image_size": ref_image_size}
            for k in range(len(ref_names)):
                r2v_in[f"ref_images.ref_image_{k}"] = [str(30 + k), 0]
            for k in range(len(ref_video_names)):
                r2v_in[f"ref_videos.ref_video_{k}"] = [str(61 + 2 * k), 0]
            g[r2v] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": r2v_in}
        else:
            # text-to-video conditioning (no first frame): the same joint AV latent comes out of
            # output 1, which the song node then fills. [H] untested on the box until an
            # environment-only scene renders; the node family is the one build_h3_t2v uses.
            i2v_in = {"clip": ["5", 0], "vae": ["6", 0], "prompt": c["prompt"].strip(),
                      "width": w, "height": h, "length": frames}
            if i2v:
                i2v_in["first_frame"] = ["30", 0]     # fed raw, as build_h3_i2v does (stretched to the canvas)
            g[r2v] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": i2v_in}
        latent = [r2v, 1]
        if not free_av:
            song_in = {"latent": latent, "audio_vae": ["9", 0], "master_audio": lock_audio,
                       "clip_start_seconds": float(win["clip_start"]),
                       "context_length": int(plan["context_frames"]), "source_fps": float(fps),
                       "crop": "disabled"}
            if prev_sampler is not None:
                song_in["source_latent"] = [prev_sampler, 0]
            g[song] = {"class_type": "MiniMaxH3SongMaskedAVContext", "inputs": song_in}
            latent = [song, 0]
        positive = [r2v, 0]
        if keyframes and ref_names:
            # Keyframe-bounded plate (opt-in, environment B-roll): anchor the LAST reference
            # picture (the location still) at the first and/or last frame of every clip through
            # core MiniMaxH3AddGuide (conditioning in, conditioning out; the latent is only read).
            # Pins the composition at the anchored frames; whether it also holds the pace of an
            # open-water shot is the hypothesis under test (docs/MV_H3_REBUILD_PLAN.md s.20.7).
            still = [str(30 + len(ref_names) - 1), 0]
            for j, idx in enumerate(keyframe_idx):
                gid = str(base + 5 + j)
                g[gid] = {"class_type": "MiniMaxH3AddGuide",
                          "inputs": {"positive": positive, "latent": [r2v, 1], "frame_idx": int(idx),
                                     "vae": ["6", 0], "image": still}}
                positive = [gid, 0]
        if anchor_name and i == 0:
            # Frame-0 anchor (opt-in, s.20.36): the scene's opening still, with her and the place
            # as the first frame should look, guides frame 0 of clip 1 through the same AddGuide
            # the keyframe plates use (the node centre-resizes the still to the canvas and
            # VAE-encodes it). The reference pictures are untouched.
            gid = str(base + 8)
            g[gid] = {"class_type": "MiniMaxH3AddGuide",
                      "inputs": {"positive": positive, "latent": [r2v, 1], "frame_idx": 0,
                                 "vae": ["6", 0], "image": ["29", 0]}}
            positive = [gid, 0]
        clip_model = model
        if clock_fps is not None:
            # Clock compression (opt-in, free-AV plates): the pack's node retimes the packed
            # layout's time axis for this clip's conditioning, latent and model together. It must
            # see the final conditioning (after the guides) and sit right before the guider.
            cid = str(base + 7)
            g[cid] = {"class_type": "MiniMaxH3NativeFPSAVSyncAc",
                      "inputs": {"model": model, "positive": positive, "av_latent": latent, "model_fps": clock_fps}}
            clip_model, positive, latent = [cid, 0], [cid, 1], [cid, 2]
        g[guider] = {"class_type": "BasicGuider", "inputs": {"model": clip_model, "conditioning": positive}}
        clip_seed = int(c.get("seed") if c.get("seed") not in (None, "") else seed + CLIP_SEED_STRIDE * i)
        g[noise] = {"class_type": "RandomNoise", "inputs": {"noise_seed": clip_seed}}
        g[sampler] = {"class_type": "SamplerCustomAdvanced",
                      "inputs": {"noise": [noise, 0], "guider": [guider, 0], "sampler": ["12", 0],
                                 "sigmas": ["11", 0], "latent_image": latent}}
        if upscale:
            # second pass (UPSCALE_MODEL comment): ids 80-89, one clip only
            g["80"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": [sampler, 0]}}
            g["81"] = {"class_type": "MinimaxH3LatentUpscaler3D",
                       "inputs": {"latent": ["80", 0], "model_name": UPSCALE_MODEL, "mode": "target dimensions",
                                  "mode.width": up_w, "mode.height": up_h, "align": 32,
                                  "enable_temporal_chunking": True, "force_unload": True, "device": "cuda",
                                  "precision": "fp16"}}
            g["82"] = {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["81", 0], "audio_latent": ["80", 1]}}
            g["83"] = {"class_type": "MiniMaxH3SongMaskedAVContext",
                       "inputs": {"latent": ["82", 0], "audio_vae": ["9", 0], "master_audio": lock_audio,
                                  "clip_start_seconds": float(win["clip_start"]),
                                  "context_length": int(plan["context_frames"]), "source_fps": float(fps),
                                  "crop": "disabled"}}
            up_positive = positive
            if i2v:
                g["84"] = {"class_type": "MiniMaxH3ImageToVideo",
                           "inputs": dict(g[r2v]["inputs"], width=up_w, height=up_h)}
                up_positive = ["84", 0]
            g["85"] = {"class_type": "BasicGuider", "inputs": {"model": clip_model, "conditioning": up_positive}}
            g["86"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": UPSCALE_SAMPLER}}
            g["87"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": UPSCALE_SIGMAS}}
            g["88"] = {"class_type": "SamplerCustomAdvanced",
                       "inputs": {"noise": [noise, 0], "guider": ["85", 0], "sampler": ["86", 0],
                                  "sigmas": ["87", 0], "latent_image": ["83", 0]}}
            sampler = "88"
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
                "lock_separate": bool(lock_name and lock_name != master_name),
                "ref_video_count": len(ref_video_names) if lane == "ref2va" else 0,
                "keyframes": keyframes if (keyframes and ref_names) else None,
                "free_av": free_av, "clock_fps": clock_fps, "first_frame": bool(i2v),
                "anchor_first": bool(anchor_name),
                "upscale": [up_w, up_h] if upscale else None,
                "turbo": bool(p.get("turbo")), "prompts": [c["prompt"].strip() for c in clips],
                "engine": "minimax_h3", "lane": "chain_i2v" if i2v else f"chain_{lane}", "model": unet, "kind": "video"}
    return g, resolved
