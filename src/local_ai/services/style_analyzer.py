"""Analyze writing samples to build a reusable writing style profile."""

import json
import logging
from pathlib import Path

import ollama as ollama_client

logger = logging.getLogger(__name__)

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


ANALYSIS_PROMPT = """You are an expert writing style analyst. Analyze the following writing samples from the same person and produce a detailed writing style profile.

Focus on these aspects:
1. **Tone**: formal/informal, direct/diplomatic, technical/accessible
2. **Sentence structure**: short and punchy, long and detailed, mixed
3. **Vocabulary**: simple/complex, technical jargon usage, favorite phrases or patterns
4. **Organization**: how they structure information (bullets, paragraphs, numbered lists)
5. **Voice**: active/passive, first person/third person, "we" vs "I" vs impersonal
6. **Formality level**: greeting style, sign-off patterns, use of abbreviations
7. **Language mixing**: do they mix German and English? Which terms stay in which language?
8. **Characteristic habits**: any recurring patterns, filler phrases, or distinctive choices

Produce the profile as a concise instruction paragraph (150-250 words) that could be given to another writer to mimic this person's style. Write it as direct instructions starting with "Write in the following style:".

Do NOT include the person's name or any identifying information in the profile. Focus purely on HOW they write, not WHO they are.

WRITING SAMPLES:
{samples}"""


def load_style_profile(path: Path) -> str | None:
    """Load an existing style profile from disk."""
    if path.exists():
        text = path.read_text(encoding="utf-8").strip()
        return text if text else None
    return None


def save_style_profile(path: Path, profile: str) -> None:
    """Save a style profile to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(profile, encoding="utf-8")
    logger.info("Style profile saved to %s (%d chars)", path, len(profile))


async def analyze_style_ollama(
    samples: list[str],
    ollama_base_url: str,
    model: str,
) -> str:
    """Analyze writing samples using Ollama."""
    combined = "\n\n---\n\n".join(f"Sample {i+1}:\n{s}" for i, s in enumerate(samples))
    prompt = ANALYSIS_PROMPT.format(samples=combined)

    client = ollama_client.AsyncClient(host=ollama_base_url)
    logger.info("Analyzing writing style via Ollama (model=%s, %d samples)...", model, len(samples))

    response = await client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.3},
    )
    return response["message"]["content"].strip()


async def analyze_style_openai(
    samples: list[str],
    base_url: str,
    api_key: str,
    model: str,
) -> str:
    """Analyze writing samples using OpenAI-compatible API."""
    if not HAS_HTTPX:
        raise RuntimeError("httpx is required for OpenAI-compatible backend")

    combined = "\n\n---\n\n".join(f"Sample {i+1}:\n{s}" for i, s in enumerate(samples))
    prompt = ANALYSIS_PROMPT.format(samples=combined)

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "none":
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }

    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()


async def analyze_style(
    samples: list[str],
    *,
    backend: str = "ollama",
    ollama_base_url: str = "",
    ollama_model: str = "",
    openai_base_url: str = "",
    openai_api_key: str = "",
    openai_model: str = "",
) -> str:
    """Analyze writing style using the configured backend."""
    if backend == "openai":
        return await analyze_style_openai(samples, openai_base_url, openai_api_key, openai_model)
    else:
        return await analyze_style_ollama(samples, ollama_base_url, ollama_model)
