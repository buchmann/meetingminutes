"""Text improvement service — corrects spelling, grammar, and clarity.

Uses the configured LLM (Granite via OpenAI-compatible API or Ollama)
to improve pasted text (emails, Slack messages, etc.) while preserving
the user's personal writing style from their style profile.
"""

import logging
import re
import time

from opentelemetry import trace

from transkriptor.tracing import get_llm_metrics

try:
    from openai import AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

logger = logging.getLogger(__name__)
_tracer = trace.get_tracer("transkriptor.text_improver")

# German stopwords for language detection
_DE_STOPWORDS = frozenset(
    "der die das ein eine und oder aber ich du er sie es wir ihr"
    " nicht mit von auf ist sind war haben wird werden kann"
    " auch noch schon wenn dann denn weil dass ob wie was wer"
    " mein dein sein unser euer ihr dem den des einem einen"
    " diese dieser dieses jeder keine mehr sehr viel".split()
)

# Max input chars — leaves room for prompt + output in Granite's 8k window
MAX_INPUT_CHARS = 6000

# Strip <think>...</think> blocks that Granite sometimes emits
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Common LLM preamble patterns to strip
_PREAMBLE_RE = re.compile(
    r"^(Here(?:'s| is) (?:the |a |your )?(?:corrected|improved|revised|updated|rewritten) (?:version|text|message)[:\s]*\n*)",
    re.IGNORECASE,
)


def detect_language(text: str) -> str:
    """Detect whether text is primarily German or English using stopword heuristic."""
    words = text.lower().split()
    if not words:
        return "en"
    de_count = sum(1 for w in words if w.strip(".,!?:;()\"'") in _DE_STOPWORDS)
    ratio = de_count / len(words)
    return "de" if ratio > 0.10 else "en"


def _build_prompt(text: str, lang: str, style_profile: str | None) -> str:
    """Build the correction prompt with optional style profile."""

    style_instruction = ""
    if style_profile:
        style_instruction = f"""
WRITING STYLE — match this person's style exactly:
{style_profile}

"""

    if lang == "de":
        return f"""Verbessere den folgenden Text: korrigiere Rechtschreibung, Grammatik und Zeichensetzung. Verbessere die Klarheit und Lesbarkeit wo noetig, aber aendere nicht die Bedeutung oder den Ton.
{style_instruction}
REGELN:
- Gib NUR den verbesserten Text zurueck, keine Erklaerungen
- Behalte die gleiche Sprache (Deutsch)
- Behalte die urspruengliche Formatierung (Absaetze, Aufzaehlungen)
- Wenn der Text bereits korrekt ist, gib ihn unveraendert zurueck
- Keine Anrede, keine Einleitung, keine Erklaerung — nur der korrigierte Text

TEXT:
{text}"""
    else:
        return f"""Improve the following text: fix spelling, grammar, and punctuation. Improve clarity and readability where needed, but do not change the meaning or tone.
{style_instruction}
RULES:
- Return ONLY the improved text, no explanations
- Keep the same language (English)
- Preserve original formatting (paragraphs, bullet points)
- If the text is already correct, return it unchanged
- No greeting, no introduction, no explanation — just the corrected text

TEXT:
{text}"""


def _clean_response(text: str) -> str:
    """Strip <think> blocks and LLM preambles from the response."""
    cleaned = _THINK_RE.sub("", text).strip()
    cleaned = _PREAMBLE_RE.sub("", cleaned).strip()
    # Remove wrapping quotes if the LLM wrapped the whole response
    if cleaned.startswith('"') and cleaned.endswith('"') and cleaned.count('"') == 2:
        cleaned = cleaned[1:-1].strip()
    return cleaned


async def _call_openai(
    prompt: str, base_url: str, api_key: str, model: str,
) -> str:
    """Call OpenAI-compatible API for text improvement."""
    if HAS_OPENAI:
        client = AsyncOpenAI(
            base_url=base_url.rstrip("/"),
            api_key=api_key if api_key and api_key != "none" else "not-needed",
            timeout=120.0,
            max_retries=2,
        )
        try:
            with _tracer.start_as_current_span(
                f"chat {model}",
                kind=trace.SpanKind.CLIENT,
                attributes={
                    "gen_ai.system": "openai",
                    "gen_ai.request.model": model,
                    "gen_ai.request.temperature": 0.3,
                    "llm.request.type": "chat",
                    "llm.request.model": model,
                },
            ) as span:
                t0 = time.monotonic()
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=2048,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                duration_ms = (time.monotonic() - t0) * 1000.0
                content = response.choices[0].message.content or ""
                usage = response.usage

                if usage:
                    prompt_tokens = usage.prompt_tokens or 0
                    completion_tokens = usage.completion_tokens or 0
                    total = prompt_tokens + completion_tokens
                    span.set_attribute("gen_ai.usage.prompt_tokens", prompt_tokens)
                    span.set_attribute("gen_ai.usage.completion_tokens", completion_tokens)
                    span.set_attribute("gen_ai.usage.total_tokens", total)

                    llm_metrics = get_llm_metrics()
                    if llm_metrics:
                        llm_metrics.record(
                            input_tokens=prompt_tokens,
                            output_tokens=completion_tokens,
                            total_tokens=total,
                            duration_ms=duration_ms,
                            service_name="transkriptor",
                            model_id=model,
                            ai_system="openai",
                        )

                logger.info(
                    "Text improve response: tokens=%s/%s, duration=%.1fs",
                    usage.prompt_tokens if usage else "?",
                    usage.completion_tokens if usage else "?",
                    duration_ms / 1000.0,
                )
                return content
        finally:
            await client.close()

    elif HAS_HTTPX:
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key and api_key != "none":
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 2048,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
    else:
        raise RuntimeError("Either 'openai' or 'httpx' package is required")


async def _call_ollama(
    prompt: str, ollama_base_url: str, model: str,
) -> str:
    """Call Ollama for text improvement."""
    import ollama as ollama_client
    client = ollama_client.AsyncClient(host=ollama_base_url)
    response = await client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.3},
    )
    return response["message"]["content"]


async def improve_text(
    text: str,
    *,
    style_profile: str | None = None,
    backend: str = "openai",
    openai_base_url: str = "",
    openai_api_key: str = "",
    openai_model: str = "",
    ollama_base_url: str = "",
    ollama_model: str = "",
) -> dict:
    """Improve text and return result dict with original, improved, and metadata.

    Returns:
        {
            "original": str,
            "improved": str,
            "language": "en" | "de",
            "had_style_profile": bool,
            "error": str | None,
        }
    """
    if not text or not text.strip():
        return {
            "original": text,
            "improved": text,
            "language": "en",
            "had_style_profile": False,
            "error": "No text provided.",
        }

    # Truncate if too long for Granite's context
    truncated = False
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS]
        truncated = True

    lang = detect_language(text)
    prompt = _build_prompt(text, lang, style_profile)

    logger.info(
        "Improving text: %d chars, lang=%s, style=%s, backend=%s",
        len(text), lang, "yes" if style_profile else "no", backend,
    )

    try:
        if backend == "openai":
            raw = await _call_openai(prompt, openai_base_url, openai_api_key, openai_model)
        else:
            raw = await _call_ollama(prompt, ollama_base_url, ollama_model)

        improved = _clean_response(raw)

        result = {
            "original": text,
            "improved": improved,
            "language": lang,
            "had_style_profile": style_profile is not None,
            "error": None,
        }
        if truncated:
            result["error"] = f"Text was truncated to {MAX_INPUT_CHARS} characters to fit the model's context window."
        return result

    except Exception as exc:
        logger.error("Text improvement failed: %s", exc, exc_info=True)
        return {
            "original": text,
            "improved": "",
            "language": lang,
            "had_style_profile": style_profile is not None,
            "error": f"LLM service error: {exc}",
        }
