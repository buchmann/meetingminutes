import json
import logging
import re
import time

import ollama as ollama_client
from opentelemetry import trace

from transkriptor.models import ActionItem, SubPoint, SummaryResult, TopicDetail, TranscriptResult
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
_tracer = trace.get_tracer("transkriptor.summarizer")

# Regex: keep Latin, digits, basic punctuation, German umlauts, common symbols
_CLEAN_RE = re.compile(r'[^\x00-\x7FÀ-ɏ‐-‧′-‷]+')


def _clean_str(s: str) -> str:
    """Remove garbled non-Latin characters (Arabic, Hebrew, CJK, Greek, etc.)
    that MoE models sometimes inject into structured output."""
    if not s:
        return s
    cleaned = _CLEAN_RE.sub('', s)
    # Collapse multiple spaces left by removed characters
    cleaned = re.sub(r' {2,}', ' ', cleaned).strip()
    return cleaned


def _clean_json_strings(obj):
    """Recursively clean all string values in a parsed JSON structure."""
    if isinstance(obj, str):
        return _clean_str(obj)
    elif isinstance(obj, dict):
        return {k: _clean_json_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_json_strings(item) for item in obj]
    return obj


def _is_garbage_subpoint(sp: dict) -> bool:
    """Detect sub-points that are just bare numbers or empty."""
    text = str(sp.get("text", "")).strip()
    detail = str(sp.get("detail", "")).strip() if sp.get("detail") else ""
    # Skip if text is just a number or empty
    if not text or text.isdigit():
        return True
    # Skip if both text and detail are very short numbers
    if len(text) <= 3 and text.replace(".", "").isdigit():
        return True
    return False


def _format_transcript_for_prompt(transcript: TranscriptResult) -> str:
    lines = []
    for seg in transcript.segments:
        ts = _format_timestamp(seg.start)
        speaker = seg.speaker or "UNKNOWN"
        lines.append(f"[{ts}] {speaker}: {seg.text}")
    return "\n".join(lines)


def _format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _detect_dominant_language(transcript: TranscriptResult) -> str:
    lang_count: dict[str, int] = {}
    for seg in transcript.segments:
        lang = seg.language or "en"
        lang_count[lang] = lang_count.get(lang, 0) + 1
    if not lang_count:
        return "en"
    return max(lang_count, key=lang_count.get)


# ── Duration-adaptive prompt tiers ──────────────────────────────────
# Short (<5 min): voice memo, quick check-in
# Medium (5-30 min): standard meeting
# Long (>30 min): workshop / deep-dive

_DURATION_TIERS = {
    "short": {"overall": "2-4 sentences", "topics": "2-3", "topic_detail": "2-3 sentences with context", "timeline": "2-4", "sub_points": "only if there are distinct sub-items"},
    "medium": {"overall": "a full paragraph (4-6 sentences)", "topics": "3-6", "topic_detail": "a short paragraph (3-5 sentences) explaining what was discussed, why it matters, and what was concluded", "timeline": "4-8", "sub_points": "include for each topic where there are distinct items, with meaningful detail per sub-point"},
    "long": {"overall": "1-2 paragraphs (5-8 sentences)", "topics": "5-10", "topic_detail": "a paragraph (4-6 sentences) covering what was discussed, the reasoning, specific concerns raised, and conclusions", "timeline": "6-15", "sub_points": "include for each topic, with detailed context per sub-point explaining the 'why' not just the 'what'"},
}


def _get_tier(duration_secs: float) -> dict:
    if duration_secs < 300:  # < 5 min
        return _DURATION_TIERS["short"]
    elif duration_secs < 1800:  # < 30 min
        return _DURATION_TIERS["medium"]
    else:
        return _DURATION_TIERS["long"]


def _build_prompt_en(tier: dict, duration_mins: int) -> str:
    return f"""You are a senior executive assistant writing professional meeting minutes.

This meeting is ~{duration_mins} minutes long. Produce a JSON object matching this EXACT structure:

{{
  "overall_summary": "{tier['overall']} identifying participants and their affiliations, the central theme, key outcomes, and agreed direction. Reference specific project names and terms in quotes.",
  "key_topics": [
    {{
      "name": "Descriptive Topic Title",
      "summary": "{tier['topic_detail']}. Explain context, current state, what was proposed, and what was concluded. Reference people and systems by name.",
      "sub_points": [
        {{"text": "Descriptive sub-item heading", "detail": "1-2 sentences with specifics about who, what, and why."}}
      ],
      "timestamp_start": "00:05:23",
      "speakers_involved": ["SPEAKER_00", "SPEAKER_01"],
      "status": "in_progress",
      "remaining": ["Concrete open item"]
    }}
  ],
  "action_items": [
    {{"description": "1-2 sentences: what needs to happen and expected outcome.", "assignee": "Person name or speaker label", "deadline": "mentioned timeframe or null"}}
  ],
  "key_decisions": ["Clear statement of what was decided with context and conditions."],
  "timeline": ["00:00-03:12: Opening with introductions and agenda review"],
  "participants": ["SPEAKER_00 (Name - Affiliation, role in meeting)"],
  "next_steps": ["Concrete action with owner and timeframe"],
  "open_questions": ["Unresolved item deferred for later"]
}}

Aim for {tier['topics']} topics and {tier['timeline']} timeline entries.

QUALITY RULES:
- Write substantive prose, not telegrams. Each topic summary should be a proper briefing paragraph.
- Reference specific names, systems, and domain terms (in quotes) from the transcript.
- Explain the "why" behind discussions, not just the "what".
- Use peoples names when identified, not just speaker labels.
- Professional third-person prose. NO greetings, NO letter format, NO filler words at the start of fields.
- ONLY valid JSON output. No markdown, no code fences, no explanation outside the JSON.
- Use only Latin characters. Timestamps: "HH:MM:SS". Do NOT invent facts.
- Every field in the schema must be present. Every array must have entries (not empty).

TRANSCRIPT:
{{transcript}}"""


def _build_prompt_en_compact(tier: dict, duration_mins: int) -> str:
    """Compact prompt for small-context models (Granite 8k). Simpler JSON, shorter instructions."""
    return f"""Write meeting minutes as JSON for this ~{duration_mins} min meeting.

Return ONLY this JSON structure (no markdown, no explanation):

{{
  "overall_summary": "4-6 sentence paragraph: who met, what was discussed, key outcomes and decisions. Use names and specific terms.",
  "key_topics": [
    {{
      "name": "Topic Title",
      "summary": "3-5 sentences: what was discussed, why it matters, what was decided or remains open. Use names."
    }}
  ],
  "action_items": [
    {{"description": "What needs to happen and why.", "assignee": "Person name", "deadline": "timeframe or null"}}
  ],
  "key_decisions": ["What was decided, with context."],
  "timeline": ["00:00-03:12: Brief description of this segment"],
  "participants": ["Name (Role/Affiliation)"],
  "next_steps": ["Action with owner and timeframe"],
  "open_questions": ["Unresolved question"]
}}

Aim for {tier['topics']} topics and {tier['timeline']} timeline entries.
Fill ALL fields with substantive content. Use names from the transcript, not just speaker labels.
ONLY valid JSON. No code fences.

TRANSCRIPT:
{{transcript}}"""


def _build_prompt_de_compact(tier: dict, duration_mins: int) -> str:
    """Compact German prompt for small-context models."""
    return f"""Erstelle ein Besprechungsprotokoll als JSON fuer dieses ~{duration_mins} Min. Meeting.

Gib NUR diese JSON-Struktur zurueck (kein Markdown, keine Erklaerung):

{{
  "overall_summary": "4-6 Saetze: Wer hat sich getroffen, was wurde besprochen, wichtigste Ergebnisse und Entscheidungen. Verwende Namen und konkrete Begriffe.",
  "key_topics": [
    {{
      "name": "Thema",
      "summary": "3-5 Saetze: Was wurde besprochen, warum wichtig, was wurde entschieden oder ist offen. Verwende Namen."
    }}
  ],
  "action_items": [
    {{"description": "Was muss geschehen und warum.", "assignee": "Name", "deadline": "Zeitrahmen oder null"}}
  ],
  "key_decisions": ["Was wurde entschieden, mit Kontext."],
  "timeline": ["00:00-03:12: Kurze Beschreibung dieses Abschnitts"],
  "participants": ["Name (Rolle/Organisation)"],
  "next_steps": ["Massnahme mit Verantwortlichem und Zeitrahmen"],
  "open_questions": ["Offene Frage"]
}}

Ziel: {tier['topics']} Themen und {tier['timeline']} Zeitachsen-Eintraege.
Fuellen Sie ALLE Felder mit inhaltlichen Angaben. Verwende Namen aus dem Transkript.
NUR gueltiges JSON. Keine Code-Bloecke.

TRANSKRIPT:
{{transcript}}"""


def _build_prompt_de(tier: dict, duration_mins: int) -> str:
    return f"""Du bist ein Senior Executive Assistant und schreibst professionelle Meeting-Protokolle.

Dieses Meeting dauert ca. {duration_mins} Minuten. Erstelle ein JSON-Objekt mit genau dieser Struktur:

{{
  "overall_summary": "{tier['overall']}. Identifiziere Teilnehmer und Zugehoerigkeiten, beschreibe das zentrale Thema, die wichtigsten Ergebnisse und die vereinbarte Richtung. Fachbegriffe in Anfuehrungszeichen.",
  "key_topics": [
    {{
      "name": "Beschreibender Thementitel",
      "summary": "{tier['topic_detail']}. Erklaere Kontext, aktuellen Stand, was vorgeschlagen wurde und was beschlossen wurde. Referenziere Personen und Systeme namentlich.",
      "sub_points": [
        {{"text": "Beschreibende Unterpunkt-Ueberschrift", "detail": "1-2 Saetze mit Details ueber wer, was und warum."}}
      ],
      "timestamp_start": "00:05:23",
      "speakers_involved": ["SPEAKER_00", "SPEAKER_01"],
      "status": "in_progress",
      "remaining": ["Konkreter offener Punkt"]
    }}
  ],
  "action_items": [
    {{"description": "1-2 Saetze: was muss passieren und erwartetes Ergebnis.", "assignee": "Name oder Speaker-Label", "deadline": "genannter Zeitrahmen oder null"}}
  ],
  "key_decisions": ["Klare Aussage was entschieden wurde mit Kontext und Bedingungen."],
  "timeline": ["00:00-03:12: Eroeffnung mit Vorstellungen und Agenda-Review"],
  "participants": ["SPEAKER_00 (Name - Zugehoerigkeit, Rolle im Meeting)"],
  "next_steps": ["Konkreter naechster Schritt mit Verantwortlichem und Zeitrahmen"],
  "open_questions": ["Ungeklaerter Punkt, verschoben auf spaeter"]
}}

Ziel: {tier['topics']} Themen und {tier['timeline']} Timeline-Eintraege.

QUALITAETSREGELN:
- Substantielle Prosa, keine Telegramme. Jede Themenzusammenfassung ist ein Briefing-Absatz.
- Spezifische Namen, Systeme und Fachbegriffe (in Anfuehrungszeichen) aus dem Transkript referenzieren.
- Das "Warum" hinter Diskussionen erklaeren, nicht nur das "Was".
- Namen der Personen verwenden wenn identifiziert, nicht nur Speaker-Labels.
- Professionelle dritte Person. KEINE Anreden, KEIN Briefformat, KEINE Fuellwoerter am Feldanfang.
- NUR valides JSON. Kein Markdown, keine Code-Bloecke, keine Erklaerung ausserhalb des JSON.
- Nur lateinische Zeichen plus Umlaute. Zeitstempel: "HH:MM:SS". KEINE Fakten erfinden.
- Jedes Feld im Schema muss vorhanden sein. Jedes Array muss Eintraege haben (nicht leer).

TRANSKRIPT:
{{transcript}}"""


async def _call_ollama(prompt: str, ollama_base_url: str, model: str) -> str:
    """Call Ollama API and return raw text response."""
    client = ollama_client.AsyncClient(host=ollama_base_url)
    response = await client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0.3},
    )
    return response["message"]["content"]


async def _call_openai_compatible(
    prompt: str, base_url: str, api_key: str, model: str,
    max_retries: int = 3,
) -> str:
    """Call an OpenAI-compatible API (DGX Spark, vLLM, etc.) and return raw text.

    Uses the ``openai`` SDK when available (auto-instrumented by traceloop for
    Instana GenAI monitoring).  Falls back to raw ``httpx`` otherwise.

    Retries up to *max_retries* times on connection errors or 5xx responses.
    """
    if HAS_OPENAI:
        return await _call_via_openai_sdk(prompt, base_url, api_key, model, max_retries)
    if HAS_HTTPX:
        return await _call_via_httpx(prompt, base_url, api_key, model, max_retries)
    raise RuntimeError(
        "Either 'openai' or 'httpx' package is required for OpenAI-compatible backend"
    )


# --------------- Model profiles ---------------
# Each profile defines the model's context window and how to budget it.
# Tokens are estimated at ~4 chars/token for English text.
_MODEL_PROFILES: dict[str, dict] = {
    "granite": {
        "context_window": 8192,
        "max_output_tokens": 2048,     # compact prompt needs less output
        "prompt_reserve_tokens": 800,  # compact prompt is shorter
        "chars_per_token": 4,
    },
    "gpt-oss-120b": {
        "context_window": 32768,
        "max_output_tokens": 16000,
        "prompt_reserve_tokens": 1500,
        "chars_per_token": 4,
    },
    "default": {
        "context_window": 16384,
        "max_output_tokens": 8000,
        "prompt_reserve_tokens": 1500,
        "chars_per_token": 4,
    },
}


def _get_model_profile(model: str, base_url: str) -> dict:
    """Match a model name / URL to its profile."""
    model_lower = model.lower()
    if "granite" in model_lower or ":8001" in base_url:
        return _MODEL_PROFILES["granite"]
    if "120b" in model_lower or "gpt-oss" in model_lower:
        return _MODEL_PROFILES["gpt-oss-120b"]
    return _MODEL_PROFILES["default"]


def _max_tokens_for_model(model: str, base_url: str) -> int:
    """Return a safe max_tokens limit based on the model's context window."""
    return _get_model_profile(model, base_url)["max_output_tokens"]


def _max_transcript_chars(model: str, base_url: str, prompt_template: str) -> int:
    """Calculate how many chars of transcript we can fit given the model profile."""
    profile = _get_model_profile(model, base_url)
    cpt = profile["chars_per_token"]
    # Budget: context_window = prompt_tokens + transcript_tokens + output_tokens
    prompt_tokens = len(prompt_template) // cpt + profile["prompt_reserve_tokens"]
    available_for_transcript = profile["context_window"] - prompt_tokens - profile["max_output_tokens"]
    max_chars = max(available_for_transcript * cpt, 2000)  # floor at 2000 chars
    logger.info(
        "Model profile: context=%d, output=%d, prompt_est=%d → transcript budget=%d tokens (%d chars)",
        profile["context_window"], profile["max_output_tokens"],
        prompt_tokens, available_for_transcript, max_chars,
    )
    return max_chars


async def _call_via_openai_sdk(
    prompt: str, base_url: str, api_key: str, model: str,
    max_retries: int = 3,
) -> str:
    """Call via openai SDK with manual gen_ai.* span attributes for Instana."""
    max_tokens = _max_tokens_for_model(model, base_url)
    client = AsyncOpenAI(
        base_url=base_url.rstrip("/"),
        api_key=api_key if api_key and api_key != "none" else "not-needed",
        timeout=600.0,
        max_retries=max_retries,
    )
    try:
        with _tracer.start_as_current_span(
            f"chat {model}",
            kind=trace.SpanKind.CLIENT,
            attributes={
                # gen_ai.* (OpenTelemetry GenAI semantic conventions)
                "gen_ai.system": "openai",
                "gen_ai.request.model": model,
                "gen_ai.request.max_tokens": max_tokens,
                "gen_ai.request.temperature": 0.1,
                "gen_ai.request.top_p": 0.9,
                # llm.* (Instana convention aliases)
                "llm.request.type": "chat",
                "llm.request.model": model,
            },
        ) as span:
            t0 = time.monotonic()
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                top_p=0.9,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                extra_body={
                    "repetition_penalty": 1.1,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            duration_ms = (time.monotonic() - t0) * 1000.0
            content = response.choices[0].message.content or ""
            finish = response.choices[0].finish_reason
            usage = response.usage

            # gen_ai.* attributes (OTEL standard)
            span.set_attribute("gen_ai.response.model", model)
            if finish:
                span.set_attribute("gen_ai.response.finish_reasons", [finish])

            prompt_tokens = 0
            completion_tokens = 0
            total = 0
            if usage:
                prompt_tokens = usage.prompt_tokens or 0
                completion_tokens = usage.completion_tokens or 0
                total = prompt_tokens + completion_tokens
                span.set_attribute("gen_ai.usage.prompt_tokens", prompt_tokens)
                span.set_attribute("gen_ai.usage.completion_tokens", completion_tokens)
                span.set_attribute("gen_ai.usage.total_tokens", total)
                # llm.* attributes (Instana aliases)
                span.set_attribute("llm.usage.input_tokens", prompt_tokens)
                span.set_attribute("llm.usage.output_tokens", completion_tokens)
                span.set_attribute("llm.usage.total_tokens", total)
            span.set_attribute("gen_ai.content.prompt", prompt[:4096])
            span.set_attribute("gen_ai.content.completion", content[:4096])

            # ── Record OTEL metrics (Instana GenAI dashboard) ─────
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
                "OpenAI response: finish_reason=%s, prompt_tokens=%s, completion_tokens=%s, "
                "content_len=%d, duration=%.1fs",
                finish,
                usage.prompt_tokens if usage else "?",
                usage.completion_tokens if usage else "?",
                len(content),
                duration_ms / 1000.0,
            )
            if finish == "length":
                logger.warning("Response was TRUNCATED (hit max_tokens). Summary may be incomplete!")
            return content
    finally:
        await client.close()


async def _call_via_httpx(
    prompt: str, base_url: str, api_key: str, model: str,
    max_retries: int = 3,
) -> str:
    """Fallback: raw httpx call (no traceloop LLM instrumentation)."""
    import asyncio as _aio

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "none":
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": 32000,
        "repetition_penalty": 1.1,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                choice = data["choices"][0]
                content = choice["message"]["content"]
                finish = choice.get("finish_reason", "unknown")
                usage = data.get("usage", {})
                logger.info(
                    "OpenAI response: finish_reason=%s, prompt_tokens=%s, completion_tokens=%s, content_len=%d",
                    finish,
                    usage.get("prompt_tokens", "?"),
                    usage.get("completion_tokens", "?"),
                    len(content),
                )
                if finish == "length":
                    logger.warning("Response was TRUNCATED (hit max_tokens). Summary may be incomplete!")
                return content
        except (httpx.ConnectError, httpx.ConnectTimeout, ConnectionError, OSError) as exc:
            last_exc = exc
            backoff = 5 * (2 ** (attempt - 1))
            logger.warning(
                "LLM connection failed (attempt %d/%d): %s — retrying in %ds",
                attempt, max_retries, exc, backoff,
            )
            await _aio.sleep(backoff)
        except httpx.HTTPStatusError as exc:
            last_exc = exc
            if exc.response.status_code >= 500:
                backoff = 5 * (2 ** (attempt - 1))
                logger.warning(
                    "LLM server error %d (attempt %d/%d) — retrying in %ds",
                    exc.response.status_code, attempt, max_retries, backoff,
                )
                await _aio.sleep(backoff)
            else:
                raise

    raise last_exc  # type: ignore[misc]


async def summarize(
    transcript: TranscriptResult,
    ollama_base_url: str,
    model: str,
    language: str = "auto",
    *,
    backend: str = "ollama",
    openai_base_url: str = "",
    openai_api_key: str = "",
    openai_model: str = "",
    style_profile: str | None = None,
) -> SummaryResult:
    dominant_lang = _detect_dominant_language(transcript)
    summary_lang = dominant_lang if language == "auto" else language

    # Pick prompt depth based on audio duration
    duration_secs = transcript.duration_secs or 0
    duration_mins = max(1, int(duration_secs / 60))
    tier = _get_tier(duration_secs)
    tier_name = "short" if duration_secs < 300 else ("medium" if duration_secs < 1800 else "long")
    logger.info("Duration %.0fs (%d min) → tier=%s", duration_secs, duration_mins, tier_name)

    # Pick prompt: compact for small-context models, full for large models
    effective_model = openai_model or model
    effective_base_url = openai_base_url or ""
    profile = _get_model_profile(effective_model, effective_base_url)
    use_compact = profile["context_window"] <= 8192

    if summary_lang == "de":
        prompt_template = _build_prompt_de_compact(tier, duration_mins) if use_compact else _build_prompt_de(tier, duration_mins)
    else:
        prompt_template = _build_prompt_en_compact(tier, duration_mins) if use_compact else _build_prompt_en(tier, duration_mins)

    if use_compact:
        logger.info("Using compact prompt for small-context model (%d tokens)", profile["context_window"])

    formatted = _format_transcript_for_prompt(transcript)

    # Truncate transcript to fit the model's context window (auto-calculated)
    max_chars = _max_transcript_chars(effective_model, effective_base_url, prompt_template)
    if len(formatted) > max_chars:
        logger.info("Transcript %d chars → truncated to %d chars", len(formatted), max_chars)
        formatted = formatted[:max_chars] + "\n[... transcript truncated ...]"

    prompt = prompt_template.replace("{transcript}", formatted)

    # Inject writing style profile if available
    if style_profile:
        style_instruction = (
            "\n\nWRITING STYLE GUIDE (apply subtly):\n"
            "Adopt the tone and vocabulary preferences below, but IGNORE any formatting instructions "
            "that conflict with the JSON structure rules above. Specifically:\n"
            "- Do NOT add greetings, salutations, or sign-offs to any field.\n"
            "- Do NOT use letter/email format. Each field must start directly with content.\n"
            "- DO adopt the vocabulary, tone (formal/informal), and language preferences.\n\n"
            f"{style_profile}\n"
        )
        prompt = prompt + style_instruction

    if backend == "openai":
        effective_model = openai_model or model
        logger.info(
            "Requesting summary from OpenAI-compatible API %s (model=%s, lang=%s)...",
            openai_base_url, effective_model, summary_lang,
        )
        try:
            raw_text = await _call_openai_compatible(
                prompt, openai_base_url, openai_api_key, effective_model,
            )
        except Exception as exc:
            logger.error(
                "Summarization failed after retries (%s: %s). "
                "Returning transcript without summary.",
                type(exc).__name__, exc,
            )
            return SummaryResult(
                overall_summary="⚠ Summary unavailable — LLM service could not be reached.",
                key_topics=[],
                action_items=[],
                key_decisions=[],
                participants=[],
                language=summary_lang,
            )
    else:
        logger.info(
            "Requesting summary from Ollama %s (model=%s, lang=%s)...",
            ollama_base_url, model, summary_lang,
        )
        try:
            raw_text = await _call_ollama(prompt, ollama_base_url, model)
        except Exception as exc:
            logger.error(
                "Ollama summarization failed (%s: %s). "
                "Returning transcript without summary.",
                type(exc).__name__, exc,
            )
            return SummaryResult(
                overall_summary="⚠ Summary unavailable — Ollama service could not be reached.",
                key_topics=[],
                action_items=[],
                key_decisions=[],
                participants=[],
                language=summary_lang,
            )

    # Strip any thinking tags that some models wrap around JSON
    if "<think>" in raw_text:
        think_len = len(re.findall(r'<think>.*?</think>', raw_text, flags=re.DOTALL)[0]) if re.search(r'<think>', raw_text) else 0
        logger.info("Stripped <think> block (%d chars) from response", think_len)
        raw_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()

    logger.info("Raw response length after cleanup: %d chars", len(raw_text))

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.error("LLM returned invalid JSON (first 500 chars): %s", raw_text[:500])
        logger.error("LLM returned invalid JSON (last 200 chars): %s", raw_text[-200:])
        data = {
            "overall_summary": raw_text[:500],
            "key_topics": [],
            "action_items": [],
            "key_decisions": [],
            "participants": [],
        }

    # Log which fields the model actually returned
    fields = {k: (len(v) if isinstance(v, list) else type(v).__name__) for k, v in data.items()}
    logger.info("Parsed JSON fields: %s", fields)

    # Clean garbled non-Latin characters from all string values
    data = _clean_json_strings(data)

    action_items = []
    for item in data.get("action_items", []) or []:
        if isinstance(item, dict):
            action_items.append(ActionItem(
                description=str(item.get("description", item)),
                assignee=str(item["assignee"]) if item.get("assignee") is not None else None,
                deadline=str(item["deadline"]) if item.get("deadline") is not None else None,
            ))
        else:
            action_items.append(ActionItem(description=str(item)))

    topics = []
    for t in data.get("key_topics", []):
        if isinstance(t, (str, int, float)):
            name = str(t).strip()
            if name and not name.isdigit():  # skip bare number topics
                topics.append(TopicDetail(name=name, summary=""))
        elif isinstance(t, dict):
            sub_points = []
            for sp in t.get("sub_points", []) or []:
                if isinstance(sp, dict):
                    if _is_garbage_subpoint(sp):
                        continue  # skip bare numbers like "452", "731"
                    sub_points.append(SubPoint(
                        text=str(sp.get("text", "")),
                        detail=str(sp["detail"]) if sp.get("detail") is not None else None,
                    ))
                else:
                    text = str(sp).strip()
                    if text and not text.isdigit():
                        sub_points.append(SubPoint(text=text))

            # Filter out empty remaining items
            remaining_raw = t.get("remaining") or []
            remaining = [str(r) for r in remaining_raw if str(r).strip() and not str(r).strip().isdigit()]

            topics.append(TopicDetail(
                name=str(t.get("name", t)),
                summary=str(t.get("summary", "")),
                sub_points=sub_points or None,
                timestamp_start=str(t["timestamp_start"]) if t.get("timestamp_start") is not None else None,
                speakers_involved=[str(s) for s in t["speakers_involved"]] if t.get("speakers_involved") else None,
                status=str(t["status"]) if t.get("status") else None,
                remaining=remaining or None,
            ))

    # Ensure all list-of-string fields contain strings
    def _str_list(val) -> list[str]:
        if not val or not isinstance(val, list):
            return []
        return [str(v) for v in val]

    return SummaryResult(
        overall_summary=str(data.get("overall_summary", "")),
        key_topics=topics,
        action_items=action_items,
        key_decisions=_str_list(data.get("key_decisions")),
        participants=_str_list(data.get("participants")),
        timeline=_str_list(data.get("timeline")) or None,
        next_steps=_str_list(data.get("next_steps")) or None,
        open_questions=_str_list(data.get("open_questions")) or None,
        language=summary_lang,
    )
