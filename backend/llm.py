"""LLM provider abstraction for the creative-assist features (D4).

Supports the user's LOCAL Ollama (gemma4_* @ localhost:11434, private/offline,
default) and their Claude account (set ANTHROPIC_API_KEY; model via config).
"""
import os
import requests

OLLAMA = "http://localhost:11434"

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
}


def ollama_models():
    try:
        r = requests.get(f"{OLLAMA}/api/tags", timeout=4)
        r.raise_for_status()
        skip = ("embed", "bge", "mxbai")
        return [m["name"] for m in r.json().get("models", []) if not any(s in m["name"].lower() for s in skip)]
    except Exception:
        return []


def ollama_chat(model: str, system: str, prompt: str) -> str:
    r = requests.post(f"{OLLAMA}/api/chat", json={
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "stream": False,
    }, timeout=180)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def claude_chat(model: str, system: str, prompt: str) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set (needed for the Claude option)")
    r = requests.post("https://api.anthropic.com/v1/messages", headers={
        "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json",
    }, json={
        "model": model, "max_tokens": 1024, "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }, timeout=180)
    r.raise_for_status()
    return "".join(b.get("text", "") for b in r.json().get("content", [])).strip()


def chat(provider: str, model: str, task: str, user_input: str, claude_model: str) -> str:
    system = SYSTEM.get(task, "You are a helpful assistant for a metal musician.")
    return complete(provider, model, system, user_input, claude_model)


def complete(provider: str, model: str, system: str, prompt: str, claude_model: str) -> str:
    """Run one system+user completion against the chosen provider (explicit system
    prompt — used by the melody composer and other structured-output features)."""
    if provider == "claude":
        return claude_chat(model or claude_model, system, prompt)
    return ollama_chat(model or "gemma4_4b:latest", system, prompt)


def best_provider() -> str:
    """Hybrid default: Claude when a key is configured, else local Ollama."""
    return "claude" if os.environ.get("ANTHROPIC_API_KEY") else "ollama"
