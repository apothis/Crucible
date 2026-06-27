"""LLM provider abstraction for the creative-assist features (D4).

Supports the user's LOCAL Ollama (gemma4_* @ localhost:11434, private/offline,
default), their Claude API account (set ANTHROPIC_API_KEY; model via config), and
their Claude SUBSCRIPTION (provider "claude_sub") via the installed `claude` CLI -
no per-token billing, drawn on the Max plan's Agent-SDK credit. The CLI is a
standalone Node binary, so it works under the 3.9 backend that the Python
claude-agent-sdk (needs 3.10+) cannot.
"""
import os
import shutil
import subprocess
import tempfile
import time

import requests

OLLAMA = "http://localhost:11434"

# Tools disabled for the subscription one-shot completion path - we want a plain
# text answer, never an agentic file/exec/web action.
_CC_NO_TOOLS = ["Bash", "Edit", "Write", "Read", "Glob", "Grep", "WebFetch",
                "WebSearch", "Task", "NotebookEdit", "TodoWrite"]


def _claude_cli():
    return shutil.which("claude") or (
        os.path.expanduser("~/.local/bin/claude")
        if os.path.exists(os.path.expanduser("~/.local/bin/claude")) else None)


def claude_code_available():
    """True when the `claude` CLI is installed (the subscription completion path)."""
    return bool(_claude_cli())


_AUTH = {"ok": None, "ts": 0.0}

def claude_code_authed(ttl: int = 300) -> bool:
    """True when the `claude` CLI can ACTUALLY authenticate headlessly - not merely
    installed. `claude_code_available()` only checks for the binary, which made the
    provider list advertise claude_sub even when the CLI's stored credential was
    expired (a 401 on every Enhance). This runs the same `claude -p` the completion
    path uses and caches the verdict for `ttl`s (a real probe costs a tiny call).
    Set CLAUDE_CODE_OAUTH_TOKEN (from `claude setup-token`) for a long-lived headless
    credential that survives the desktop app rotating the keychain token out."""
    cli = _claude_cli()
    if not cli:
        return False
    now = time.time()
    if _AUTH["ok"] is not None and (now - _AUTH["ts"]) < ttl:
        return _AUTH["ok"]
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    ok = False
    try:
        with tempfile.TemporaryDirectory() as cwd:
            r = subprocess.run([cli, "-p", "ok", "--output-format", "text", "--effort", "low",
                                "--disallowed-tools", *_CC_NO_TOOLS],
                               capture_output=True, text=True, env=env, cwd=cwd, timeout=45)
        ok = (r.returncode == 0) and bool((r.stdout or "").strip())
    except Exception:
        ok = False
    _AUTH.update(ok=ok, ts=now)
    return ok


def claude_code_chat(system: str, prompt: str, model: str = "", timeout: int = 240) -> str:
    """One-shot completion through the Claude Code CLI, drawing on the user's Claude
    SUBSCRIPTION (OAuth) instead of a per-token API key. Runs `claude -p` as a
    subprocess. ANTHROPIC_API_KEY is stripped from the child env so the CLI uses the
    subscription, not the key (a set key silently shadows the OAuth login). Runs in a
    neutral temp cwd so the project's CLAUDE.md / memory is not pulled into context;
    `--system-prompt` replaces the default agent prompt for a clean completion."""
    cli = _claude_cli()
    if not cli:
        raise RuntimeError("claude CLI not found - install Claude Code, then run `claude setup-token`")
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    # --effort low: these are one-shot text/JSON completions, NOT agentic coding. The Claude Code CLI
    # defaults to high/xhigh effort (deep extended thinking), which made a single script generation take
    # ~15 MINUTES. Low effort drops it to under a minute with no quality loss for generation tasks.
    cmd = [cli, "-p", prompt, "--system-prompt", system, "--output-format", "text",
           "--effort", "low", "--disallowed-tools", *_CC_NO_TOOLS]
    if model:
        cmd += ["--model", model]
    try:
        with tempfile.TemporaryDirectory() as cwd:
            r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"claude CLI timed out after {timeout}s (large generation - try a shorter song / "
                           f"fewer shots, or retry)")
    if r.returncode != 0:
        msg = (r.stderr or r.stdout or "").strip()[:300]
        raise RuntimeError(f"claude CLI failed (is it logged in? `claude setup-token`): {msg}")
    return (r.stdout or "").strip()

SYSTEM = {
    "lyrics": (
        "You are a metal lyricist. Write song lyrics for the user's theme. "
        "Use section tags like [Verse], [Pre-Chorus], [Chorus], [Bridge], [Outro]. "
        "Vivid, singable lines fitting metal (epic, dark, heroic, mythic as appropriate). "
        "Output ONLY the lyrics with section tags — no commentary."
    ),
    "tags": (
        "You write concise comma-separated style tags for an AI music generator "
        "(ACE-Step), focused on rock and metal. Given an idea, output ONE line of "
        "comma-separated tags naming: subgenre, key instruments (e.g. distorted guitars, "
        "double-bass drums), vocal style, tempo, and atmosphere. Output ONLY the tags."
    ),
    "ideas": (
        "You are a creative metal music collaborator. Brainstorm concise song concepts, "
        "themes, titles, or arrangement ideas for the user's prompt. Keep it tight and usable."
    ),
    "names": (
        "You name songs, albums, and bands for a rock/metal artist. Given a theme, style, "
        "or lyrics, output a short list of evocative, fitting titles (up to 8), one per line, "
        "with no numbering, quotes, or commentary."
    ),
    "char_desc": (
        "You write photoreal CHARACTER reference prompts for an image generator (Z-Image Turbo). "
        "Given the user's short description, expand it into ONE vivid, concrete prompt for a single "
        "person: approximate age, build/height, face shape, skin, hair (color, length, style), eyes, "
        "and any distinctive features; include wardrobe/props only if the user mentioned them; finish "
        "with photoreal cues (studio lighting, sharp focus, detailed skin texture). Do NOT specify the "
        "shot framing (close-up vs full-body) - that is added separately. Keep it a single flowing "
        "phrase-list, no sentences, no line breaks, no commentary. Output ONLY the prompt."
    ),
}


def ollama_models():
    try:
        r = requests.get(f"{OLLAMA}/api/tags", timeout=4)
        r.raise_for_status()
        skip = ("embed", "bge", "mxbai")
        return [m["name"] for m in r.json().get("models", []) if not any(s in m["name"].lower() for s in skip)]
    except Exception:
        return []


def ollama_chat(model: str, system: str, prompt: str, timeout: int = 180) -> str:
    r = requests.post(f"{OLLAMA}/api/chat", json={
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "stream": False,
    }, timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def claude_chat(model: str, system: str, prompt: str, timeout: int = 180, max_tokens: int = 1024) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set (needed for the Claude option)")
    r = requests.post("https://api.anthropic.com/v1/messages", headers={
        "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json",
    }, json={
        "model": model, "max_tokens": max_tokens, "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }, timeout=timeout)
    r.raise_for_status()
    return "".join(b.get("text", "") for b in r.json().get("content", [])).strip()


def chat(provider: str, model: str, task: str, user_input: str, claude_model: str) -> str:
    system = SYSTEM.get(task, "You are a helpful assistant for a metal musician.")
    return complete(provider, model, system, user_input, claude_model)


def complete(provider: str, model: str, system: str, prompt: str, claude_model: str, timeout: int = 240) -> str:
    """Run one system+user completion against the chosen provider (explicit system
    prompt — used by the melody composer and other structured-output features).
    `timeout` (seconds) lets big jobs (e.g. a long MV shot list) ask for more headroom."""
    if provider == "claude":
        # a long shot list needs a bigger token budget than the 1024 default, or it truncates.
        return claude_chat(model or claude_model, system, prompt, timeout=timeout, max_tokens=8192)
    if provider in ("claude_sub", "claude_code"):
        return claude_code_chat(system, prompt, model or "", timeout=timeout)   # subscription via CLI, no per-token billing
    return ollama_chat(model or "gemma4_4b:latest", system, prompt, timeout=timeout)


def best_provider() -> str:
    """Hybrid default: the Claude SUBSCRIPTION (no per-token) when the CLI is present,
    else the Claude API key when configured, else local Ollama."""
    if claude_code_available():
        return "claude_sub"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude"
    return "ollama"
