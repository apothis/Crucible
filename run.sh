#!/usr/bin/env bash
# MusicGen app launcher.
#   First time:
#     python3 -m venv .venv && ./.venv/bin/pip install -r backend/requirements.txt
#     (cd web && npm install && npm run build)     # build the React UI
#   Run:        ./run.sh   then open http://127.0.0.1:8000
#
# The backend serves the built React app from web/dist (falls back to the
# classic frontend/ if web/dist is absent). For UI development with hot-reload:
#     (cd web && npm run dev)   -> http://localhost:5173  (proxies /api -> :8000)
# Edit app_config.json to point at your ComfyUI/RVC hosts (Windows box).
cd "$(dirname "$0")"
# Keep all model/cache downloads on the SSD (the system drive is small). Scoped to
# this process so unrelated caches elsewhere are untouched.
export TORCH_HOME="$PWD/.caches/torch"
export HF_HOME="$PWD/.caches/hf"
export HUGGINGFACE_HUB_CACHE="$PWD/.caches/hf/hub"
export PIP_CACHE_DIR="$PWD/.caches/pip"
export UV_CACHE_DIR="$PWD/.caches/uv"
exec ./.venv/bin/python -m backend.app
