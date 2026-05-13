"""
Unified LLM client — switches between Ollama (local/free) and Groq (free tier).

Usage:
    from llm_client import chat, chat_json

    response = chat("Summarise Apple's 10-K risk factors.")
    data     = chat_json("Return JSON with keys: score, summary.")
"""

import json
import re
import requests
from config import (
    LLM_BACKEND,
    OLLAMA_BASE_URL, OLLAMA_MODEL,
    GROQ_API_KEY, GROQ_MODEL,
)


# ── Ollama ────────────────────────────────────────────────────────────────

def _ollama_chat(messages: list[dict], system: str = "", temperature: float = 0.1) -> str:
    """Call local Ollama server. Must have `ollama serve` running."""
    if system:
        messages = [{"role": "system", "content": system}] + messages

    payload = {
        "model":    OLLAMA_MODEL,
        "messages": messages,
        "stream":   False,
        "options":  {"temperature": temperature},
    }
    r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"]


# ── Groq ──────────────────────────────────────────────────────────────────

def _groq_chat(messages: list[dict], system: str = "", temperature: float = 0.1) -> str:
    """Call Groq cloud API (free tier). Requires GROQ_API_KEY in .env."""
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)

    if system:
        messages = [{"role": "system", "content": system}] + messages

    response = client.chat.completions.create(
        model       = GROQ_MODEL,
        messages    = messages,
        temperature = temperature,
        max_tokens  = 1500,
    )
    return response.choices[0].message.content


# ── Public API ────────────────────────────────────────────────────────────

def chat(
    user_message:  str,
    system:        str        = "",
    history:       list[dict] = None,
    temperature:   float      = 0.1,
) -> str:
    """
    Send a message to the configured LLM. Returns response string.
    history = [{"role": "user"|"assistant", "content": "..."}]
    """
    messages = list(history or [])
    messages.append({"role": "user", "content": user_message})

    if LLM_BACKEND == "groq":
        return _groq_chat(messages, system=system, temperature=temperature)
    return _ollama_chat(messages, system=system, temperature=temperature)


def chat_json(
    user_message: str,
    system:       str        = "",
    history:      list[dict] = None,
) -> dict:
    """
    Like chat() but instructs the model to return JSON and parses it.
    Falls back to empty dict on parse failure.
    """
    json_system = (system + "\n" if system else "") + \
        "IMPORTANT: Return ONLY valid JSON. No markdown, no explanation, no code fences."

    raw = chat(user_message, system=json_system, history=history, temperature=0.0)

    # Strip any accidental fences
    cleaned = re.sub(r"```json|```", "", raw).strip()
    # Find the first {...} block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"_raw": raw, "_error": "json_parse_failed"}