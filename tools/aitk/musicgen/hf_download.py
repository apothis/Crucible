"""Pre-fetch Hugging Face repos into AI Toolkit's cache (HF_HOME = ai-toolkit\\.hf, set by the helper) so a
later job does not hold the GPU idle while it downloads. No GPU use: the helper runs it with free_gpu false.
Config (ai-toolkit\\config\\<name>.json): {"repos": ["org/name", ...]}"""
import json
import sys
import time

from huggingface_hub import snapshot_download

for repo in json.load(open(sys.argv[1], encoding="utf-8"))["repos"]:
    t0 = time.time()
    path = snapshot_download(repo)
    print(f"[{time.strftime('%H:%M:%S')}] {repo} -> {path} ({time.time() - t0:.0f} s)", flush=True)
print("DOWNLOAD DONE", flush=True)
