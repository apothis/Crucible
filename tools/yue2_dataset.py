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
                                                writes <stem>.style: the style line only, house order)
  python3 tools/yue2_dataset.py merge   <name>   style line + reviewed lyrics -> <stem>.txt on the box
                                                (copies kept in .mvwork/yue2_train/<name>/txt/)
Lyrics are never transcribed from audio (house rule): they come from the online fetch the ACE
pipeline already did (<stem>.lyrics.txt beside the source audio).
"""
import json
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
}
LOCAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".mvwork", "yue2_train")

# The captioner's prompt: the house style line (docs/YUE2_REVIEW.md) minus BPM and key, which it
# cannot hear reliably. No lyrics: those come from the online fetch, not from the audio.
STYLE_PROMPT = (
    "Listen to this song and describe how it sounds as ONE line of comma-separated descriptors, "
    "in exactly this order: the language of the lyrics (for example English); the genre with its era "
    "(for example modern power metal with 80s synth influence); the lead vocal: gender, register, timbre "
    "and delivery (for example powerful raspy female belt); how the vocal and energy change between "
    "sections, as two or three short phrases (for example restrained verses, soaring belted choruses, "
    "stripped-down bridge); the defining instruments and production; a two-word mood; the tempo feel "
    "(slow, mid-tempo or fast). Lowercase except the language. No sentences, no artist, band or song "
    "names, no lyrics, no key, no numbers. Output only the line.")


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
            "caption_extension": "style", "recaption": False, "compile": False,
            "caption_prompt": STYLE_PROMPT, "max_new_tokens": 160, "batch_size": 1}}]}}
    # JSON is valid YAML, so the job file is written as JSON text with a .yaml name
    r = _post(HELPER, "/fs/write", {"path": AITK + r"\config" + "\\" + job + ".yaml", "content": json.dumps(cfg, indent=1)})
    print(f"wrote {r['path']}; run it with POST {HELPER}/aitk/run {{\"job\": \"{job}\"}}")


def merge(name):
    _src, dst, local = _paths(name)
    os.makedirs(os.path.join(local, "txt"), exist_ok=True)
    done, missing = 0, []
    for stem in _stems(dst):
        st = _get("/fs/stat", path=dst + "\\" + stem + ".style")
        if not st.get("exists"):
            missing.append(stem)
            continue
        style = " ".join(_get("/fs/read", path=dst + "\\" + stem + ".style")["content"].split())
        lyrics = open(os.path.join(local, "lyrics", stem + ".txt")).read().strip()
        text = f"{style}\n[Lyrics]\n{lyrics}\n"
        open(os.path.join(local, "txt", stem + ".txt"), "w").write(text)
        _post(HELPER, "/fs/write", {"path": dst + "\\" + stem + ".txt", "content": text})
        done += 1
    print(f"{done} .txt written to {dst}" + (f"; no style line yet for {missing}" if missing else ""))


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("stage", "caption", "merge") or sys.argv[2] not in SOURCES:
        sys.exit(__doc__)
    {"stage": stage, "caption": caption, "merge": merge}[sys.argv[1]](sys.argv[2])
