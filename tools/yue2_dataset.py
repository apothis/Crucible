"""Build a YuE2 LoRA dataset for Ostris AI Toolkit from an existing band set on the box
(docs/YUE2_TRAINING_PLAN.md). Everything goes through the box helper (:5080); nothing is done by
hand on the box.

Per song the trainer reads <stem>.mp3 + <stem>.txt, where the .txt is
    <style line>
    [Lyrics]
    <lyrics with bare section tags>

Steps (run from the repo root with the backend up on :8000):
  python3 tools/yue2_dataset.py stage   <name>   copy audio box-side into ai-toolkit\\datasets\\yue2_<name>,
                                                section the online-fetched lyrics (YuE2 tab's sectioner,
                                                LLM with heuristic fallback) into .mvwork/yue2_train/<name>/
                                                lyrics/ for review
  python3 tools/yue2_dataset.py caption <name>   write the caption job (Qwen2.5-Omni hears each song and
                                                writes <stem>.parts: section contrast, instruments, mood,
                                                tempo feel; language/genre/voice are FIXED per band)
  python3 tools/yue2_dataset.py enrich  <name>   per-song parts rewritten by the app LLM from the ACE engine's
                                                existing audio description of each song (ACE 4B LM prose,
                                                far more specific than the captioner's), keeping the
                                                captioner's tempo feel; -> .mvwork/.../parts_final/
  python3 tools/yue2_dataset.py merge   <name>   style line + reviewed lyrics -> <stem>.txt on the box
                                                (copies kept in .mvwork/yue2_train/<name>/txt/)
  python3 tools/yue2_dataset.py train   <name> [v4|v9] [4000|whole] [steps] [planner LR x]   write the training job (one LoRA over planner +
                                                decoder); v9 = newer tokenizer + its merged decoder adapter
Lyrics are never transcribed from audio (house rule): they come from the online fetch the ACE
pipeline already did (<stem>.lyrics.txt beside the source audio).
"""
import json
import ntpath
import os
import re
import sys
import urllib.parse
import urllib.request

HELPER = "http://192.168.1.201:5080"
BACKEND = "http://127.0.0.1:8000"
AITK = r"E:\AI\MusicGen\ai-toolkit"
SOURCES = {   # dataset name -> the ACE dataset folder that already holds audio + fetched lyrics
    "battlebeast": r"E:\AI\MusicGen\AceStep\ACE-Step-1.5\lora_data\crucible_battlebeast\data",
    "beastinblack": r"E:\AI\MusicGen\AceStep\ACE-Step-1.5\lora_data\crucible_beastinblack\data",
}
LOCAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".mvwork", "yue2_train")

# Fixed per band (user-approved): the parts that carry the band's identity are the same words in
# every caption, so the LoRA ties them to this sound. Battle Beast 2026-10-17.
FIXED = {
    "battlebeast": {"language": "English",
                    "genre": "modern power metal with 80s heavy metal and synth influences",
                    "voice": "powerful raspy female rock belt with a soaring high range"},
    # 2026-10-18, drafted overnight under the user's go-ahead (their call to change): Yannis Papadopoulos
    "beastinblack": {"language": "English",
                     "genre": "80s-inspired power metal with synthwave and disco-pop influences",
                     "voice": "soaring high male tenor belt with piercing falsetto screams and a gritty edge"},
}

# The captioner hears each song and writes ONLY the per-song parts. No example phrases: the first run
# (with examples) copied them word for word into ~15 captions and got language/genre/voice wrong.
PARTS_PROMPT = (
    "Listen to this song. Write ONE line of short comma-separated descriptors covering, in this order: "
    "how the vocal delivery and the energy change between the song's sections (two or three short "
    "phrases naming the sections); the defining instruments and production; a two-word mood; the tempo "
    "feel (slow, mid-tempo or fast). Do not mention the language, the genre, or the singer's voice type "
    "or gender. Lowercase. No sentences, no names, no lyrics, no key, no numbers. Output only the line.")


def _get(endpoint, **q):
    return json.load(urllib.request.urlopen(HELPER + endpoint + "?" + urllib.parse.urlencode(q), timeout=120))


def _post(base, path, body, timeout=300):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def _paths(name):
    return SOURCES[name], AITK + r"\datasets\yue2_" + name, os.path.join(LOCAL, name)


def _stems(src):
    return sorted(e["name"][:-4] for e in _get("/fs/list", path=src)["entries"] if e["name"].lower().endswith(".mp3"))


def stage(name):
    src, dst, local = _paths(name)
    os.makedirs(os.path.join(local, "lyrics"), exist_ok=True)
    _post(HELPER, "/fs/mkdir", {"path": dst})
    have = {e["name"] for e in _get("/fs/list", path=dst)["entries"]}
    manifest = []
    for stem in _stems(src):
        if stem + ".mp3" not in have:
            _post(HELPER, "/fs/copy", {"src": src + "\\" + stem + ".mp3", "dst": dst + "\\" + stem + ".mp3"})
        out = os.path.join(local, "lyrics", stem + ".txt")
        if os.path.exists(out):
            tagged, how = open(out).read(), "kept (already reviewed)"
        else:
            raw = _get("/fs/read", path=src + "\\" + stem + ".lyrics.txt")["content"]
            r = _post(BACKEND, "/api/yue2/section_lyrics", {"lyrics": raw})
            tagged, how = r["lyrics"], r.get("tagged_by")
            open(out, "w").write(tagged)
        tags = re.findall(r"^\[([^\]]+)\]", tagged, re.M)
        manifest.append({"stem": stem, "tagged_by": how, "sections": tags})
        print(f"{stem:40s} {how:24s} {' '.join(tags)}", flush=True)
    json.dump(manifest, open(os.path.join(local, "manifest.json"), "w"), indent=1)
    print(f"\n{len(manifest)} songs staged in {dst}; lyrics for review in {os.path.join(local, 'lyrics')}")


def caption(name):
    _src, dst, _local = _paths(name)
    job = f"yue2_{name}_caption"
    cfg = {"job": "extension", "config": {"name": job, "process": [{
        "type": "Qwen25OmniCaptioner", "device": "cuda",
        "caption": {
            "model_name_or_path": "ai-toolkit/Qwen2.5-Omni-7B/qwen2_5_omni_7b_convrot8.safetensors",
            "dtype": "bf16", "extensions": ["mp3"], "path_to_caption": dst,
            "caption_extension": "parts", "recaption": False, "compile": False,
            "caption_prompt": PARTS_PROMPT, "max_new_tokens": 160, "batch_size": 1}}]}}
    # JSON is valid YAML, so the job file is written as JSON text with a .yaml name
    r = _post(HELPER, "/fs/write", {"path": AITK + r"\config" + "\\" + job + ".yaml", "content": json.dumps(cfg, indent=1)})
    print(f"wrote {r['path']}; run it with POST {HELPER}/aitk/run {{\"job\": \"{job}\"}}")


COMFY_MODELS = r"E:\AI\MusicGen\Comfyui\ComfyUI_windows_portable\ComfyUI\models"


HEADS = {   # community audio -> semantic tokenizers (Mothersuperior); v9 = the author's pick by listening,
            # co-trained on real rock/metal; it pairs with its own decoder adapter, merged for training and
            # loaded in ComfyUI as nar_lora_joint_v9_comfyui at generation
    "v4": None,
    "v9": {"semantic_head_path": "Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4/tokenizer_head_joint_v9.safetensors",
           "nar_lora_path": "Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4/nar_lora_joint_v9.safetensors",
           "merge_nar_lora": True},
}


def train(name, head="v4", planner="4000", steps="800", ar_mult="0.5", kl="0.2"):
    """Recipe from docs/YUE2_TRAINING_PLAN.md s4 (GRVL/CNZN/vdruts, AI Toolkit's yue2 defaults): rank 32,
    adamw8bit, LR 5e-5, planner LR x0.5 + KL 0.2 (the planner memorises small sets fast), cot full with
    50% score dropout, 60 s decoder windows, <= 20 passes (40 songs -> 800 steps), a checkpoint every 50
    steps for a by-ear ladder. bf16 base from ComfyUI's models folder (no download); in-training samples
    off (AR CUDA-graph crash on sampling) - audition in the YuE2 tab instead.
    planner: music tokens the planner trains on from each song's start - "4000" (the recipe's memory cap,
    160 s) or "whole" (trainer default 0 = whole song: the planner also learns later choruses and endings;
    LoRAs from the 4000 cap ended songs at 300 s, docs/YUE2_TRAINING_PLAN.md s6).
    steps / ar_mult: a longer run with a slower planner (e.g. 2000 at x0.2 = the planner's 800 at x0.5 in
    total LR) gives the decoder - one 60 s window per step - more of the voice without overcooking the
    planner. ~20 checkpoints per run are kept, so save_every grows with the step count.
    kl: the planner's pull back toward the base model (KL(base || LoRA) weight; recipe 0.2). Lower lets the
    planner move further toward the band's own token choices - where the singer's voice was found to be lost."""
    _src, dst, _local = _paths(name)
    steps, ar_mult, kl = int(steps), float(ar_mult), float(kl)
    ar_max_tokens = 0 if planner == "whole" else int(planner)
    job = f"yue2_{name}" + ("" if head == "v4" else f"_{head}")
    if planner != "4000":
        job += "_wholesong" if planner == "whole" else f"_ar{planner}"
    if ar_mult != 0.5 or steps != 800:
        job += f"_plr{ar_mult:g}_{steps}"
    if kl != 0.2:
        job += f"_kl{kl:g}"
    job += "_lora"
    save_every = max(50, round(steps / 20 / 50) * 50)
    cfg = {"job": "extension", "config": {"name": job, "process": [{
        "type": "diffusion_trainer", "training_folder": "output", "device": "cuda", "trigger_word": None,
        "network": {"type": "lora", "linear": 32, "linear_alpha": 32},
        "save": {"dtype": "bf16", "save_every": save_every, "max_step_saves_to_keep": 20},
        "datasets": [{"folder_path": dst, "caption_ext": "txt", "caption_dropout_rate": 0,
                      "cache_latents_to_disk": True, "resolution": [512], "num_repeats": 1}],
        "train": {"batch_size": 1, "steps": steps, "gradient_accumulation": 1, "train_unet": True,
                  "train_text_encoder": False, "gradient_checkpointing": True, "noise_scheduler": "flowmatch",
                  "timestep_type": "sigmoid", "optimizer": "adamw8bit", "optimizer_params": {"weight_decay": 1e-4},
                  "lr": 5e-5, "dtype": "bf16", "disable_sampling": True, "unload_text_encoder": False,
                  "cache_text_embeddings": False},
        "logging": {"log_every": 1, "use_ui_logger": False},
        "model": {"arch": "yue2", "name_or_path": COMFY_MODELS + r"\checkpoints\yue2_3b_bf16.safetensors",
                  "quantize": False, "low_vram": False,
                  "model_kwargs": {"cot": "full", "abc_dropout": 0.5, "ar_kl_weight": kl, "ar_lr_multiplier": ar_mult,
                                   "ar_max_tokens": ar_max_tokens, "train_window_frames": 1500, "ar_cuda_graphs": False,
                                   "sheetsage_path": COMFY_MODELS + r"\audio_encoders\sheetsage2_bf16.safetensors",
                                   **(HEADS[head] or {})}},
    }]}}
    r = _post(HELPER, "/fs/write", {"path": AITK + r"\config" + "\\" + job + ".yaml", "content": json.dumps(cfg, indent=1)})
    print(f"wrote {r['path']} ({steps} steps, checkpoint every {save_every}); run with POST {HELPER}/aitk/run {{\"job\": \"{job}\"}}")


def describe(name):
    """Audio descriptions from ACE-Step's own music captioner (AI Toolkit AceStepCaptioner: acestep-captioner
    for the description, acestep-transcriber alongside it, both Qwen2.5-Omni-7B, fp8 to fit the 3090), for
    bands whose ACE dataset has no LM prose (Beast in Black: tag lists only). Written beside each song as
    <stem>.acecap; enrich() condenses the description paragraph. The transcribed lyrics in that file are
    never used (lyrics come from the online fetch)."""
    _src, dst, _local = _paths(name)
    job = f"yue2_{name}_describe"
    cfg = {"job": "extension", "config": {"name": job, "process": [{
        "type": "AceStepCaptioner", "device": "cuda",
        "caption": {
            "model_name_or_path": "ACE-Step/acestep-transcriber", "model_name_or_path2": "ACE-Step/acestep-captioner",
            "caption_format": "yue2", "dtype": "bf16", "quantize": True, "qtype": "float8", "low_vram": False,
            "extensions": ["mp3"], "path_to_caption": dst, "caption_extension": "acecap",
            "recaption": False, "compile": False, "batch_size": 1}}]}}
    r = _post(HELPER, "/fs/write", {"path": AITK + r"\config" + "\\" + job + ".yaml", "content": json.dumps(cfg, indent=1)})
    print(f"wrote {r['path']}; run it with POST {HELPER}/aitk/run {{\"job\": \"{job}\"}}")


ENRICH_SYSTEM = (
    "You condense a detailed description of one song into a short training caption fragment for a "
    "music model. Output STRICT JSON ONLY: {\"parts\": \"...\"}. No prose, no markdown.")
ENRICH_RULES = """Write ONE lowercase line of short comma-separated descriptors, at most 45 words, in this order:
1. how the song moves through its sections: 2-4 short phrases naming sections and what happens in them
   (e.g. "atmospheric piano intro, belted choruses, shred guitar solo, chugging breakdown, piano outro");
2. the defining instruments and production (2-4 phrases);
3. a two-word mood;
4. the tempo feel: use the CAPTIONER TEMPO given below (slow, mid-tempo or fast) unless the description
   plainly contradicts it.
Use ONLY what the description states. Never mention genre or style names, the singer's voice type, gender,
register or vocal technique, the language, lyrics, artist/song names, keys or numbers."""


def enrich(name):
    src, _dst, local = _paths(name)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend import llm
    if name == "battlebeast":
        ds = json.loads(_get("/fs/read", path=ntpath.dirname(src) + "\\dataset.json")["content"])
        prose = {s["filename"].rsplit(".", 1)[0]: s.get("caption") or "" for s in ds["samples"]}
    else:   # no LM prose in the ACE dataset: ACE-Step captioner descriptions (describe())
        prose = {}
        for stem in _stems(src):
            text = _get("/fs/read", path=_paths(name)[1] + "\\" + stem + ".acecap")["content"]
            prose[stem] = "x. " + text.split("[Lyrics]")[0].strip()      # "x. ": the prefix drop below keeps it whole
    os.makedirs(os.path.join(local, "parts_final"), exist_ok=True)
    for stem in _stems(src):
        out = os.path.join(local, "parts_final", stem + ".txt")
        if os.path.exists(out):
            print(f"{stem:38s} kept: {open(out).read().strip()}")
            continue
        desc = prose.get(stem, "")
        desc = desc.split(". ", 1)[1] if ". " in desc else desc      # drop the curated tag prefix
        qpath = os.path.join(local, "parts", stem + ".txt")
        qwen = open(qpath).read().strip() if os.path.exists(qpath) else desc
        tempo = next((t for t in ("mid-tempo", "slow", "fast") if t in qwen.lower()), "mid-tempo")
        prompt = f"{ENRICH_RULES}\n\nCAPTIONER TEMPO: {tempo}\n\nDESCRIPTION:\n{desc}\n\nReturn the JSON now."
        text = llm.complete("claude_sub", "claude-sonnet-5", ENRICH_SYSTEM, prompt, "", timeout=180)
        m = re.search(r"\{.*\}", text or "", re.S)
        parts = " ".join(json.loads(m.group(0) if m else text)["parts"].split()).strip(" ,.").lower()
        open(out, "w").write(parts + "\n")
        print(f"{stem:38s} {parts}", flush=True)


def merge(name):
    _src, dst, local = _paths(name)
    os.makedirs(os.path.join(local, "txt"), exist_ok=True)
    done, missing = 0, []
    for stem in _stems(dst):
        final = os.path.join(local, "parts_final", stem + ".txt")
        if not os.path.exists(final):
            missing.append(stem)
            continue
        parts = " ".join(open(final).read().split()).strip(" ,.")
        f = FIXED[name]
        style = f"{f['language']}, {f['genre']}, {f['voice']}, {parts}"
        lyrics = open(os.path.join(local, "lyrics", stem + ".txt")).read().strip()
        text = f"{style}\n[Lyrics]\n{lyrics}\n"
        open(os.path.join(local, "txt", stem + ".txt"), "w").write(text)
        _post(HELPER, "/fs/write", {"path": dst + "\\" + stem + ".txt", "content": text})
        done += 1
    print(f"{done} .txt written to {dst}" + (f"; no per-song parts yet for {missing} (run enrich)" if missing else ""))


if __name__ == "__main__":
    cmds = {"stage": stage, "caption": caption, "describe": describe, "enrich": enrich, "merge": merge, "train": train}
    if len(sys.argv) not in range(3, 9) or sys.argv[1] not in cmds or sys.argv[2] not in SOURCES:
        sys.exit(__doc__)
    cmds[sys.argv[1]](*sys.argv[2:])      # train: [v4|v9] [4000|whole] [steps] [planner LR x] [KL weight]
