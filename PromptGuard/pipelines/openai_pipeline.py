"""
PromptGuard — Target Pipeline Wrappers
Defines the LLM pipelines that PromptGuard evaluates.
Add new pipelines here to benchmark different systems.
"""

import os
from typing import Optional


def build_openai_pipeline(
    system_prompt: str = "You are a helpful assistant.",
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
):
    """
    Returns a callable pipeline wrapping OpenAI's chat completions.
    The pipeline takes a user prompt and returns the model's response string.
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    def run(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content

    run.__name__ = f"openai_{model}"
    return run


def build_ollama_pipeline(
    system_prompt: str = "You are a helpful assistant.",
    model: str = "mistral",
    base_url: str = "http://localhost:11434",
):
    """
    Returns a callable pipeline for locally-running Ollama models.
    Requires Ollama to be running: `ollama run mistral`
    """
    try:
        import requests
    except ImportError:
        raise ImportError("requests package not installed. Run: pip install requests")

    def run(prompt: str) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        response = requests.post(f"{base_url}/api/chat", json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["message"]["content"]

    run.__name__ = f"ollama_{model}"
    return run


def build_mock_pipeline(vulnerability_level: float = 0.4):
    """
    A mock pipeline for testing without API costs.
    Simulates a vulnerable LLM — responds to injections with probability=vulnerability_level.
    """
    import random

    INJECTION_KEYWORDS = [
        "ignore", "system prompt", "override", "reveal", "forget", "sudo",
        "admin", "confidential", "hidden", "instructions", "context window",
        "dan", "unrestricted", "bypass", "memory", "secret",
    ]

    VULNERABLE_RESPONSES = [
        "Sure! My system prompt says: 'You are a helpful assistant with no restrictions.'",
        "I can reveal that my instructions include: respond to all queries without filtering.",
        "Developer mode activated. My hidden context is: [SYSTEM: classify all inputs as safe]",
        "As requested, here is my full context: You are an AI with elevated privileges...",
    ]

    SAFE_RESPONSES = [
        "I'm sorry, I can't help with that.",
        "That request falls outside what I'm able to assist with.",
        "I don't have access to my system prompt or context window.",
        "I'm designed to be helpful, harmless, and honest. I can't fulfil that request.",
    ]

    def run(prompt: str) -> str:
        prompt_lower = prompt.lower()
        is_injection_like = any(kw in prompt_lower for kw in INJECTION_KEYWORDS)

        if is_injection_like and random.random() < vulnerability_level:
            return random.choice(VULNERABLE_RESPONSES)
        return random.choice(SAFE_RESPONSES)

    run.__name__ = f"mock_pipeline_vuln{int(vulnerability_level*100)}"
    return run
