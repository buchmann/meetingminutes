import json
import logging
import re
import time

import ollama as ollama_client
from opentelemetry import trace

from local_ai.models import ActionItem, SubPoint, SummaryResult, TopicDetail, TranscriptResult
from local_ai.tracing import get_llm_metrics

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
_tracer = trace.get_tracer("local_ai.summarizer")

# ── Temperature ("hallucination" dial) ─────────────────────────────────────
# The UI exposes a 0..1 slider per feature. We clamp to a slightly wider band
# server-side so a hand-crafted request can still push a bit higher, but never
# into the unstable >1.5 range that produces garbage with these models.
TEMP_MIN = 0.0
TEMP_MAX = 1.5


def clamp_temperature(value) -> float | None:
    """Coerce a user-supplied temperature into a safe range.

    Returns a float in [TEMP_MIN, TEMP_MAX], or None when the input is missing
    or unparseable — in which case the caller falls back to its own default.
    """
    if value is None or value == "":
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return max(TEMP_MIN, min(TEMP_MAX, v))


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

# Detailed mode: roughly 2x the content of every section.
# Counts are doubled, prose length is doubled, and prompts ask for more depth.
_DURATION_TIERS_DETAILED = {
    "short": {"overall": "4-8 sentences with rich context", "topics": "4-6", "topic_detail": "4-6 sentences covering context, what was discussed, reasoning, and outcomes", "timeline": "4-8", "sub_points": "include sub-points for every topic, each with meaningful detail"},
    "medium": {"overall": "two full paragraphs (8-12 sentences) covering setup, central themes, key decisions, and outcomes", "topics": "6-12", "topic_detail": "a substantial paragraph (6-10 sentences) covering background, what was discussed, the reasoning, specific concerns or alternatives raised, and what was concluded", "timeline": "8-16", "sub_points": "include 3-5 sub-points for each topic with meaningful detail per sub-point"},
    "long": {"overall": "2-4 paragraphs (10-16 sentences) detailing setup, themes, deep-dive discussions, decisions and outcomes", "topics": "10-20", "topic_detail": "an extended paragraph (8-12 sentences) covering background, what was discussed, reasoning, specific concerns raised, alternatives considered, and conclusions", "timeline": "12-30", "sub_points": "include 4-8 detailed sub-points per topic explaining context, reasoning, and implications"},
}


# Extensive mode: roughly 4x the standard content (2x the "detailed" tier).
# The most exhaustive level — used together with dedicated product-feature and
# project-work sections for a maximally detailed record.
_DURATION_TIERS_EXTENSIVE = {
    "short": {"overall": "two rich paragraphs (8-14 sentences) covering setup, every theme raised, reasoning, and outcomes", "topics": "6-10", "topic_detail": "a full paragraph (8-12 sentences) covering context, what was discussed, the reasoning, concerns and alternatives, and outcomes", "timeline": "6-12", "sub_points": "include 3-6 detailed sub-points for every topic, each explaining context and reasoning"},
    "medium": {"overall": "three to four full paragraphs (16-24 sentences) covering setup, every central theme, the discussion arc, decisions, disagreements, and outcomes", "topics": "12-20", "topic_detail": "a thorough multi-sentence paragraph (10-16 sentences) covering background, the full discussion, reasoning, every concern or alternative raised, trade-offs, and conclusions", "timeline": "16-30", "sub_points": "include 5-9 detailed sub-points per topic, each a full thought explaining context, reasoning, and implications"},
    "long": {"overall": "four to six paragraphs (24-40 sentences) giving a comprehensive narrative of the entire session: setup, themes, every deep-dive, decisions, disagreements, and outcomes", "topics": "20-35", "topic_detail": "an exhaustive paragraph (14-22 sentences) covering background, the complete discussion, all reasoning, every concern, alternatives weighed, trade-offs, and conclusions", "timeline": "30-60", "sub_points": "include 6-12 detailed sub-points per topic, each a complete thought with context, reasoning, and implications"},
}


# Exhaustive mode: roughly 10x the standard content (~2.5x "extensive"). The
# most thorough level possible within the model context — every topic, every
# product feature with step-by-step sub-points, every project task, and every
# open question enumerated in maximum detail.
_DURATION_TIERS_EXHAUSTIVE = {
    "short": {"overall": "three to four rich paragraphs (16-26 sentences) covering setup, every theme, all reasoning, disagreements and outcomes", "topics": "12-18", "topic_detail": "an exhaustive paragraph (14-20 sentences) covering full context, the complete discussion, all reasoning, every concern and alternative, trade-offs, and outcomes", "timeline": "12-24", "sub_points": "include 6-12 detailed step-by-step sub-points for every topic"},
    "medium": {"overall": "five to seven full paragraphs (28-44 sentences) giving a complete narrative of the meeting: setup, every theme, the full discussion arc, all decisions, disagreements, and outcomes", "topics": "25-40", "topic_detail": "an exhaustive multi-paragraph treatment (18-28 sentences) covering background, the complete discussion verbatim in substance, all reasoning, every concern, alternative and trade-off, and conclusions", "timeline": "30-60", "sub_points": "include 10-18 detailed step-by-step sub-points per topic"},
    "long": {"overall": "eight to twelve paragraphs (48-80 sentences) — a comprehensive minute-by-minute narrative of the entire session", "topics": "40-70", "topic_detail": "an exhaustive treatment (24-36 sentences) capturing the complete discussion, all reasoning, every concern, alternative, trade-off and conclusion in full", "timeline": "60-120", "sub_points": "include 12-24 detailed step-by-step sub-points per topic"},
}


def _get_tier(duration_secs: float, detail_level: str = "standard") -> dict:
    """Return the tier dict for the given duration.
    detail_level: 'detailed' ≈2x, 'extensive' ≈4x, 'exhaustive' ≈10x content."""
    if detail_level == "exhaustive":
        pool = _DURATION_TIERS_EXHAUSTIVE
    elif detail_level == "extensive":
        pool = _DURATION_TIERS_EXTENSIVE
    elif detail_level == "detailed":
        pool = _DURATION_TIERS_DETAILED
    else:
        pool = _DURATION_TIERS
    if duration_secs < 300:  # < 5 min
        return pool["short"]
    elif duration_secs < 1800:  # < 30 min
        return pool["medium"]
    else:
        return pool["long"]


_EXTENSIVE_FIELDS_EN = """,
  "product_features": [
    {"name": "Feature or capability name", "summary": "As much detail as the transcript supports: what the feature does, the problem it solves, requirements and acceptance criteria mentioned, dependencies, edge cases, and any open design questions.", "sub_points": [{"text": "Specific requirement, behaviour, or sub-feature", "detail": "1-3 sentences with specifics."}], "status": "proposed | in_progress | done"}
  ],
  "project_work": [
    {"name": "Workstream, task, or deliverable", "summary": "As much detail as the transcript supports: scope, the work to be done, approach, blockers, risks, estimates, and dependencies on other work or people.", "sub_points": [{"text": "Concrete task or step", "detail": "1-3 sentences: who, what, status."}], "status": "not_started | in_progress | blocked | done", "remaining": ["Concrete remaining item"]}
  ]"""

_EXTENSIVE_RULES_EN = (
    "\n- product_features: exhaustively list EVERY product feature, capability, or "
    "requirement discussed, each described in maximum detail. If none were discussed, use an empty array.\n"
    "- project_work: exhaustively list EVERY project task, workstream, or deliverable "
    "discussed, each described in maximum detail (scope, approach, blockers, owners). "
    "If none were discussed, use an empty array."
)

_EXHAUSTIVE_RULES_EN = (
    "\n\nEXHAUSTIVE MODE — be as thorough as humanly possible (this is a maximal-detail record):\n"
    "- Do NOT summarise tersely anywhere. Capture EVERYTHING of substance from the transcript.\n"
    "- product_features AND project_work: for EACH item provide a long, step-by-step list of "
    "sub_points (the concrete steps, requirements, tasks and decisions), not just a couple.\n"
    "- open_questions: exhaustively enumerate EVERY unresolved point, ambiguity, risk, or "
    "question raised or implied — each as a full sentence WITH its context and why it is open. "
    "Aim for many entries, not a short list.\n"
    "- key_decisions and action_items: include every single one, with full context.\n"
    "- It is better to be exhaustive and long than concise."
)


def _build_prompt_en(tier: dict, duration_mins: int, extras: bool = False,
                     exhaustive: bool = False) -> str:
    extra_fields = _EXTENSIVE_FIELDS_EN if extras else ""
    extra_rules = (_EXTENSIVE_RULES_EN if extras else "") + (_EXHAUSTIVE_RULES_EN if exhaustive else "")
    return f"""You are a senior executive assistant writing professional meeting minutes.

This meeting is ~{duration_mins} minutes long. Produce a JSON object matching this EXACT structure:

{{
  "overall_summary": "{tier['overall']} identifying participants by the names actually used in the call (add a role or organisation ONLY if it was explicitly stated — never guess), the central theme, key outcomes, and agreed direction. Reference specific project names and terms in quotes.",
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
  "participants": ["SPEAKER_00 (Name) — append ', role' or ', organisation' ONLY if explicitly stated in the call, otherwise give just the name"],
  "next_steps": ["Concrete action with owner and timeframe"],
  "open_questions": ["Unresolved item deferred for later"]{extra_fields}
}}

Aim for {tier['topics']} topics and {tier['timeline']} timeline entries.

QUALITY RULES:{extra_rules}
- Write substantive prose, not telegrams. Each topic summary should be a proper briefing paragraph.
- Reference specific names, systems, and domain terms (in quotes) from the transcript.
- Explain the "why" behind discussions, not just the "what".
- Use peoples names when identified, not just speaker labels.
- Professional third-person prose. NO greetings, NO letter format, NO filler words at the start of fields.
- ONLY valid JSON output. No markdown, no code fences, no explanation outside the JSON.
- Use only Latin characters. Timestamps: "HH:MM:SS".
- CRITICAL — do NOT invent or guess any facts not stated in the transcript. In
  particular NEVER fabricate company names, organisations, job titles or roles.
  If a person's company/role was not stated, give only their name. Never use
  placeholder names like "XYZ Corp", "ACME", "ABC GmbH".
- Every field in the schema must be present. Arrays may be empty if the
  transcript genuinely contains nothing for them — do NOT pad with invented items.

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
  "participants": ["Name (add role/org ONLY if explicitly stated, never invent)"],
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
  "participants": ["Name (Rolle/Organisation NUR wenn genannt, niemals erfinden)"],
  "next_steps": ["Massnahme mit Verantwortlichem und Zeitrahmen"],
  "open_questions": ["Offene Frage"]
}}

Ziel: {tier['topics']} Themen und {tier['timeline']} Zeitachsen-Eintraege.
Fuellen Sie ALLE Felder mit inhaltlichen Angaben. Verwende Namen aus dem Transkript.
NUR gueltiges JSON. Keine Code-Bloecke.

TRANSKRIPT:
{{transcript}}"""


_EXTENSIVE_FIELDS_DE = """,
  "product_features": [
    {"name": "Name des Features oder der Funktion", "summary": "So detailliert wie das Transkript hergibt: was das Feature macht, welches Problem es loest, genannte Anforderungen und Akzeptanzkriterien, Abhaengigkeiten, Sonderfaelle und offene Design-Fragen.", "sub_points": [{"text": "Konkrete Anforderung, Verhalten oder Teil-Feature", "detail": "1-3 Saetze mit Details."}], "status": "geplant | in_arbeit | fertig"}
  ],
  "project_work": [
    {"name": "Arbeitspaket, Aufgabe oder Liefergegenstand", "summary": "So detailliert wie das Transkript hergibt: Umfang, zu erledigende Arbeit, Vorgehen, Blocker, Risiken, Schaetzungen und Abhaengigkeiten von anderen Arbeiten oder Personen.", "sub_points": [{"text": "Konkrete Aufgabe oder Schritt", "detail": "1-3 Saetze: wer, was, Status."}], "status": "offen | in_arbeit | blockiert | fertig", "remaining": ["Konkreter offener Punkt"]}
  ]"""

_EXTENSIVE_RULES_DE = (
    "\n- product_features: Liste LUECKENLOS JEDES besprochene Produkt-Feature, jede Funktion "
    "oder Anforderung auf, jeweils maximal detailliert. Falls keine besprochen wurden, leeres Array.\n"
    "- project_work: Liste LUECKENLOS JEDE besprochene Projektaufgabe, jeden Arbeitsstrang oder "
    "Liefergegenstand auf, jeweils maximal detailliert (Umfang, Vorgehen, Blocker, Verantwortliche). "
    "Falls keine besprochen wurden, leeres Array."
)

_EXHAUSTIVE_RULES_DE = (
    "\n\nEXHAUSTIVE-MODUS — so gruendlich wie nur moeglich (maximal detailliertes Protokoll):\n"
    "- NIRGENDS knapp zusammenfassen. Erfasse ALLES Inhaltliche aus dem Transkript.\n"
    "- product_features UND project_work: fuer JEDEN Eintrag eine lange, schrittweise Liste von "
    "sub_points (die konkreten Schritte, Anforderungen, Aufgaben und Entscheidungen), nicht nur ein paar.\n"
    "- open_questions: zaehle LUECKENLOS JEDEN offenen Punkt, jede Unklarheit, jedes Risiko und jede "
    "Frage auf — jeweils als vollstaendiger Satz MIT Kontext und warum es offen ist. Ziel: viele Eintraege.\n"
    "- key_decisions und action_items: jede einzelne mit vollem Kontext.\n"
    "- Lieber ausfuehrlich und lang als knapp."
)


def _build_prompt_de(tier: dict, duration_mins: int, extras: bool = False,
                     exhaustive: bool = False) -> str:
    extra_fields = _EXTENSIVE_FIELDS_DE if extras else ""
    extra_rules = (_EXTENSIVE_RULES_DE if extras else "") + (_EXHAUSTIVE_RULES_DE if exhaustive else "")
    return f"""Du bist ein Senior Executive Assistant und schreibst professionelle Meeting-Protokolle.

Dieses Meeting dauert ca. {duration_mins} Minuten. Erstelle ein JSON-Objekt mit genau dieser Struktur:

{{
  "overall_summary": "{tier['overall']}. Nenne die Teilnehmer mit den im Gespraech tatsaechlich verwendeten Namen (Rolle oder Organisation NUR wenn ausdruecklich genannt — niemals raten), beschreibe das zentrale Thema, die wichtigsten Ergebnisse und die vereinbarte Richtung. Fachbegriffe in Anfuehrungszeichen.",
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
  "participants": ["SPEAKER_00 (Name) — ', Rolle' oder ', Organisation' NUR anhaengen, wenn im Gespraech ausdruecklich genannt, sonst nur den Namen"],
  "next_steps": ["Konkreter naechster Schritt mit Verantwortlichem und Zeitrahmen"],
  "open_questions": ["Ungeklaerter Punkt, verschoben auf spaeter"]{extra_fields}
}}

Ziel: {tier['topics']} Themen und {tier['timeline']} Timeline-Eintraege.

QUALITAETSREGELN:{extra_rules}
- Substantielle Prosa, keine Telegramme. Jede Themenzusammenfassung ist ein Briefing-Absatz.
- Spezifische Namen, Systeme und Fachbegriffe (in Anfuehrungszeichen) aus dem Transkript referenzieren.
- Das "Warum" hinter Diskussionen erklaeren, nicht nur das "Was".
- Namen der Personen verwenden wenn identifiziert, nicht nur Speaker-Labels.
- Professionelle dritte Person. KEINE Anreden, KEIN Briefformat, KEINE Fuellwoerter am Feldanfang.
- NUR valides JSON. Kein Markdown, keine Code-Bloecke, keine Erklaerung ausserhalb des JSON.
- Nur lateinische Zeichen plus Umlaute. Zeitstempel: "HH:MM:SS".
- WICHTIG — KEINE Fakten erfinden oder raten, die nicht im Transkript stehen.
  Insbesondere NIEMALS Firmennamen, Organisationen, Jobtitel oder Rollen erfinden.
  Wenn Firma/Rolle einer Person nicht genannt wurde, nur den Namen angeben.
  Niemals Platzhalter-Namen wie "XYZ Corp", "ACME", "ABC GmbH" verwenden.
- Jedes Feld im Schema muss vorhanden sein. Arrays duerfen leer sein, wenn das
  Transkript dazu wirklich nichts enthaelt — NICHT mit erfundenen Eintraegen fuellen.

TRANSKRIPT:
{{transcript}}"""


async def _call_ollama(
    prompt: str, ollama_base_url: str, model: str,
    temperature: float | None = None,
) -> str:
    """Call Ollama API and return raw text response.

    ``temperature`` controls output randomness (the "hallucination" dial).
    None falls back to the historical default of 0.3.
    """
    temp = 0.3 if temperature is None else temperature
    client = ollama_client.AsyncClient(host=ollama_base_url)
    response = await client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": temp},
    )
    return response["message"]["content"]


async def _call_openai_compatible(
    prompt: str, base_url: str, api_key: str, model: str,
    max_retries: int = 3,
    detail_level: str = "standard",
    response_format: str = "json_object",
    temperature: float | None = None,
    max_tokens_override: int | None = None,
    reasoning_effort: str | None = None,
) -> str:
    """Call an OpenAI-compatible API (DGX Spark, vLLM, etc.) and return raw text.

    Uses the ``openai`` SDK when available (auto-instrumented by traceloop for
    Instana GenAI monitoring).  Falls back to raw ``httpx`` otherwise.

    Retries up to *max_retries* times on connection errors or 5xx responses.

    ``response_format`` controls the LLM's output mode:
      - ``"json_object"`` (default) — used by the summarizer, forces strict JSON
      - ``"text"`` — free-form text; used by translation and similar tasks

    ``temperature`` is the "hallucination" dial; None uses the historical
    default of 0.1 for these factual-output tasks.
    """
    if HAS_OPENAI:
        return await _call_via_openai_sdk(
            prompt, base_url, api_key, model, max_retries, detail_level,
            response_format, temperature, max_tokens_override, reasoning_effort,
        )
    if HAS_HTTPX:
        return await _call_via_httpx(
            prompt, base_url, api_key, model, max_retries, detail_level,
            response_format, temperature, max_tokens_override, reasoning_effort,
        )
    raise RuntimeError(
        "Either 'openai' or 'httpx' package is required for OpenAI-compatible backend"
    )


# --------------- Model profiles ---------------
# Each profile defines the model's context window and how to budget it.
# chars_per_token is conservative (German subword tokenization is ~3.0-3.2
# chars/token, English ~4). Using 3 ensures we don't overshoot.
# safety_margin_tokens absorbs off-by-one and prompt-formatting overhead.
_MODEL_PROFILES: dict[str, dict] = {
    "granite": {
        # Granite 4.0-H-Small unterstützt 128K nativ; der vLLM-Server läuft mit
        # --max-model-len 32768 → die App darf den größeren Kontext nutzen.
        "context_window": 32768,
        "max_output_tokens": 4096,     # mehr Spielraum für ausführliche Antworten
        "prompt_reserve_tokens": 800,
        "chars_per_token": 3,          # safe for German + English
        "safety_margin_tokens": 300,
    },
    "gpt-oss-120b": {
        "context_window": 32768,
        "max_output_tokens": 16000,
        "prompt_reserve_tokens": 1500,
        "chars_per_token": 3,
        "safety_margin_tokens": 500,
    },
    "default": {
        "context_window": 16384,
        "max_output_tokens": 8000,
        "prompt_reserve_tokens": 1500,
        "chars_per_token": 3,
        "safety_margin_tokens": 300,
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


def _max_tokens_for_model(model: str, base_url: str, detail_level: str = "standard") -> int:
    """Return a safe max_tokens limit. Scaled by detail level
    (detailed ≈2x, extensive ≈4x, exhaustive ≈10x the base output), but clamped
    so the input (transcript) still has reasonable room in the context window."""
    profile = _get_model_profile(model, base_url)
    base = profile["max_output_tokens"]
    factor = {"detailed": 2, "extensive": 4, "exhaustive": 10}.get(detail_level)
    if not factor:
        return base
    ctx = profile["context_window"]
    safety = profile.get("safety_margin_tokens", 100)
    desired = base * factor
    # Cap the output so the transcript still fits. For "exhaustive" the output
    # ceiling alone could otherwise consume ~75% of the context and truncate the
    # transcript so hard that quality drops — so reserve more room for input
    # (≈45%) while still allowing a very large summary.
    ceiling_frac = 0.55 if detail_level == "exhaustive" else 0.75
    max_allowed = int(ctx * ceiling_frac) - safety
    return min(desired, max_allowed)


def _max_transcript_chars(model: str, base_url: str, prompt_template: str) -> int:
    """Calculate how many chars of transcript we can fit given the model profile.
    Char-based fallback used only when accurate tokenization is unavailable."""
    profile = _get_model_profile(model, base_url)
    cpt = profile["chars_per_token"]
    safety = profile.get("safety_margin_tokens", 100)
    # Budget: context_window = prompt_tokens + transcript_tokens + output_tokens + safety
    prompt_tokens = len(prompt_template) // cpt + profile["prompt_reserve_tokens"]
    available_for_transcript = (
        profile["context_window"] - prompt_tokens - profile["max_output_tokens"] - safety
    )
    max_chars = max(available_for_transcript * cpt, 2000)  # floor at 2000 chars
    logger.info(
        "Model profile (char-fallback): context=%d, output=%d, safety=%d, prompt_est=%d → transcript budget=%d tokens (%d chars)",
        profile["context_window"], profile["max_output_tokens"], safety,
        prompt_tokens, available_for_transcript, max_chars,
    )
    return max_chars


async def count_tokens_vllm(text: str, base_url: str, model: str) -> int | None:
    """Get exact token count from vLLM's /tokenize endpoint. Returns None on failure.

    vLLM exposes /tokenize on the *server root* (e.g. http://host:8001/tokenize),
    while the chat completions live under /v1/chat/completions. base_url is
    typically configured as 'http://host:8001/v1', so we strip the /v1 suffix.
    """
    if not HAS_HTTPX:
        return None
    try:
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        url = f"{root}/tokenize"
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json={"model": model, "prompt": text})
            resp.raise_for_status()
            data = resp.json()
            return int(data.get("count", 0))
    except Exception as e:
        logger.warning("vLLM /tokenize failed (will fall back to char estimate): %s", e)
        return None


async def _truncate_transcript_to_fit(
    formatted: str,
    prompt_template: str,
    model: str,
    base_url: str,
    backend: str,
    detail_level: str = "standard",
) -> str:
    """Truncate the formatted transcript so prompt+transcript+output fits in the
    model's context window. Uses vLLM's /tokenize for exact counts when available,
    falling back to char-based estimation otherwise.

    Returns the (possibly truncated) transcript text. Iterates up to 3 times to
    converge if the first truncation still overshoots due to tokenizer drift.
    """
    profile = _get_model_profile(model, base_url)
    context = profile["context_window"]
    out_budget = _max_tokens_for_model(model, base_url, detail_level)
    safety = profile.get("safety_margin_tokens", 100)
    target_input = context - out_budget - safety  # max prompt+transcript tokens

    # Char-based fallback path (Ollama or vLLM unreachable)
    use_vllm_tokenize = backend == "openai" and bool(base_url)

    if not use_vllm_tokenize:
        max_chars = _max_transcript_chars(model, base_url, prompt_template)
        if len(formatted) > max_chars:
            logger.info("Transcript %d chars → truncated to %d chars (char-based)", len(formatted), max_chars)
            return formatted[:max_chars] + "\n[... transcript truncated ...]"
        return formatted

    # Accurate path: use vLLM /tokenize
    for attempt in range(3):
        candidate = prompt_template.replace("{transcript}", formatted)
        total = await count_tokens_vllm(candidate, base_url, model)
        if total is None:
            # vLLM tokenize failed mid-iteration → fall back to chars for this call
            logger.warning("Falling back to char-based truncation")
            max_chars = _max_transcript_chars(model, base_url, prompt_template)
            if len(formatted) > max_chars:
                return formatted[:max_chars] + "\n[... transcript truncated ...]"
            return formatted

        logger.info(
            "Token budget check (attempt %d): prompt+transcript=%d tokens, target≤%d, context=%d",
            attempt + 1, total, target_input, context,
        )

        if total <= target_input:
            return formatted  # fits

        # Compute how much transcript to keep based on the overshoot ratio.
        # Apply 5% extra cut to absorb non-linearity and converge faster.
        overshoot_ratio = total / target_input
        new_len = int(len(formatted) / overshoot_ratio * 0.95)
        new_len = max(new_len, 500)  # never cut below ~500 chars
        logger.info(
            "Transcript %d → %d chars (overshoot=%.2fx)",
            len(formatted), new_len, overshoot_ratio,
        )
        formatted = formatted[:new_len] + "\n[... transcript truncated ...]"

    logger.warning("Truncation did not converge after 3 attempts; returning best-effort cut")
    return formatted


def _effective_reasoning_effort(model: str, explicit: str | None) -> str | None:
    """Decide the reasoning_effort to send.

    gpt-oss is a reasoning model; with no bound it spends the whole output
    budget on the reasoning channel and returns empty content. For this app's
    structured/factual tasks (extraction, summaries, translation) "low" is both
    reliable and faster. An explicit value always wins. Non-gpt-oss models get
    nothing (the param is ignored by them anyway)."""
    if explicit:
        return explicit
    if "gpt-oss" in (model or "").lower():
        return "low"
    return None


async def _call_via_openai_sdk(
    prompt: str, base_url: str, api_key: str, model: str,
    max_retries: int = 3,
    detail_level: str = "standard",
    response_format: str = "json_object",
    temperature: float | None = None,
    max_tokens_override: int | None = None,
    reasoning_effort: str | None = None,
) -> str:
    """Call via openai SDK with manual gen_ai.* span attributes for Instana."""
    max_tokens = max_tokens_override or _max_tokens_for_model(model, base_url, detail_level)
    temp = 0.1 if temperature is None else temperature
    effort = _effective_reasoning_effort(model, reasoning_effort)
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
                "gen_ai.request.temperature": temp,
                "gen_ai.request.top_p": 0.9,
                # llm.* (Instana convention aliases)
                "llm.request.type": "chat",
                "llm.request.model": model,
            },
        ) as span:
            t0 = time.monotonic()
            sdk_kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temp,
                "top_p": 0.9,
                "max_tokens": max_tokens,
                "extra_body": {
                    "repetition_penalty": 1.1,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            }
            # gpt-oss is a reasoning model: without a bounded reasoning_effort it
            # can burn the whole max_tokens budget on the reasoning channel and
            # return EMPTY content (finish_reason=length, content_len=0) → JSON
            # parse failures. Inject reasoning_effort so the final channel is
            # actually produced.
            if effort:
                sdk_kwargs["extra_body"]["reasoning_effort"] = effort
            # Only enforce JSON when the caller asked for it. Free-form tasks
            # (translation, etc.) pass response_format="text".
            if response_format == "json_object":
                sdk_kwargs["response_format"] = {"type": "json_object"}
            response = await client.chat.completions.create(**sdk_kwargs)
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
                    service_name="local_ai",
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
    detail_level: str = "standard",
    response_format: str = "json_object",
    temperature: float | None = None,
    max_tokens_override: int | None = None,
    reasoning_effort: str | None = None,
) -> str:
    """Fallback: raw httpx call (no traceloop LLM instrumentation)."""
    import asyncio as _aio

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "none":
        headers["Authorization"] = f"Bearer {api_key}"

    max_tokens = max_tokens_override or _max_tokens_for_model(model, base_url, detail_level)
    temp = 0.1 if temperature is None else temperature
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temp,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "repetition_penalty": 1.1,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    effort = _effective_reasoning_effort(model, reasoning_effort)
    if effort:
        payload["reasoning_effort"] = effort
    if response_format == "json_object":
        payload["response_format"] = {"type": "json_object"}

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
    detail_level: str = "standard",
) -> SummaryResult:
    dominant_lang = _detect_dominant_language(transcript)
    summary_lang = dominant_lang if language == "auto" else language

    if detail_level not in ("standard", "detailed", "extensive", "exhaustive"):
        detail_level = "standard"

    # Pick prompt depth based on audio duration
    duration_secs = transcript.duration_secs or 0
    duration_mins = max(1, int(duration_secs / 60))
    tier = _get_tier(duration_secs, detail_level)
    tier_name = "short" if duration_secs < 300 else ("medium" if duration_secs < 1800 else "long")
    logger.info(
        "Duration %.0fs (%d min) → tier=%s, detail=%s",
        duration_secs, duration_mins, tier_name, detail_level,
    )

    # Pick prompt: compact for small-context models, full for large models
    effective_model = openai_model or model
    effective_base_url = openai_base_url or ""
    profile = _get_model_profile(effective_model, effective_base_url)
    use_compact = profile["context_window"] <= 8192
    # The dedicated product-feature / project-work breakdown is added in the
    # "extensive" and "exhaustive" levels; "exhaustive" also turns on the
    # maximal-thoroughness rules (step-by-step sub-points + exhaustive
    # questions). Only on the full prompt (small-context models use compact).
    extras = detail_level in ("extensive", "exhaustive") and not use_compact
    exhaustive = detail_level == "exhaustive" and not use_compact

    if summary_lang == "de":
        prompt_template = _build_prompt_de_compact(tier, duration_mins) if use_compact else _build_prompt_de(tier, duration_mins, extras, exhaustive)
    else:
        prompt_template = _build_prompt_en_compact(tier, duration_mins) if use_compact else _build_prompt_en(tier, duration_mins, extras, exhaustive)

    if use_compact:
        logger.info("Using compact prompt for small-context model (%d tokens)", profile["context_window"])

    formatted = _format_transcript_for_prompt(transcript)

    # Truncate transcript to fit the model's context window.
    # Uses vLLM's /tokenize endpoint for exact token counts when backend="openai",
    # falling back to char-based estimation otherwise (e.g. Ollama backend).
    formatted = await _truncate_transcript_to_fit(
        formatted, prompt_template, effective_model, effective_base_url, backend, detail_level,
    )

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
                detail_level=detail_level,
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

    # product_features / project_work share the topic shape (name, summary,
    # sub_points, status, remaining) — parse them the same way as key_topics.
    def _parse_topic_details(raw) -> list[TopicDetail]:
        out: list[TopicDetail] = []
        for t in raw or []:
            if isinstance(t, (str, int, float)):
                name = str(t).strip()
                if name and not name.isdigit():
                    out.append(TopicDetail(name=name, summary=""))
                continue
            if not isinstance(t, dict):
                continue
            sps = []
            for sp in t.get("sub_points", []) or []:
                if isinstance(sp, dict):
                    if _is_garbage_subpoint(sp):
                        continue
                    sps.append(SubPoint(
                        text=str(sp.get("text", "")),
                        detail=str(sp["detail"]) if sp.get("detail") is not None else None,
                    ))
                else:
                    txt = str(sp).strip()
                    if txt and not txt.isdigit():
                        sps.append(SubPoint(text=txt))
            remaining_raw = t.get("remaining") or []
            remaining = [str(r) for r in remaining_raw if str(r).strip() and not str(r).strip().isdigit()]
            name = str(t.get("name", "")).strip()
            summary = str(t.get("summary", "")).strip()
            if not name and not summary:
                continue
            out.append(TopicDetail(
                name=name or "(unnamed)",
                summary=summary,
                sub_points=sps or None,
                status=str(t["status"]) if t.get("status") else None,
                remaining=remaining or None,
            ))
        return out

    product_features = _parse_topic_details(data.get("product_features")) or None
    project_work = _parse_topic_details(data.get("project_work")) or None

    return SummaryResult(
        overall_summary=str(data.get("overall_summary", "")),
        key_topics=topics,
        action_items=action_items,
        key_decisions=_str_list(data.get("key_decisions")),
        participants=_str_list(data.get("participants")),
        timeline=_str_list(data.get("timeline")) or None,
        next_steps=_str_list(data.get("next_steps")) or None,
        open_questions=_str_list(data.get("open_questions")) or None,
        product_features=product_features,
        project_work=project_work,
        language=summary_lang,
    )


# ── Selection summarizer: summarize a chosen transcript excerpt ──────────
# Produces TWO variants of the same excerpt: one that keeps the concrete
# examples given, and one with all examples stripped (core points only).

_SELECTION_MAX_CHARS = 16000


_SELECTION_STRUCTURE = (
    "Structure each summary as a compact analyst brief in Markdown, using ONLY the "
    "sections that actually apply to this excerpt (omit empty ones):\n"
    "**Problem / Context** — what problem or situation is being discussed and why it matters.\n"
    "**Key points / Requirements** — bullet list; each bullet a complete, substantive statement "
    "(derived requirements, agreed behaviours, constraints).\n"
    "**Decisions** — what was explicitly decided, with conditions.\n"
    "**Open questions** — unresolved points, each with WHY it is open.\n"
    "**Implications** — what follows from this for the product/project (only if the excerpt supports it).\n"
    "Use '**Section**' bold headers and '- ' bullets. No deeper nesting."
)


def _build_selection_prompt(excerpt: str) -> str:
    return (
        "You are a senior analyst writing a structured brief from an excerpt of a "
        "meeting transcript. Analyze ONLY this excerpt — do not invent anything not "
        "present or clearly implied in it.\n\n"
        "Return a JSON object with EXACTLY these keys, written in the SAME language "
        "as the excerpt:\n"
        "{\n"
        '  "topic": "a short title (3-8 words) naming what this excerpt is about",\n'
        '  "with_examples": "Structured brief that PRESERVES the concrete examples, '
        'cases, tools, numbers and specifics mentioned (markdown string).",\n'
        '  "without_examples": "Structured brief of the SAME excerpt with every '
        'specific example, anecdote, illustrative case and concrete number REMOVED — '
        'only the core principles, requirements, decisions and questions (markdown string)."\n'
        "}\n\n"
        + _SELECTION_STRUCTURE + "\n\n"
        "QUALITY RULES for both briefs:\n"
        "- Do not just restate sentences — SYNTHESIZE: turn statements into derived "
        "requirements ('The orchestrator must...'), name trade-offs, and separate what "
        "was decided from what remains open.\n"
        "- Do NOT mention speaker labels (SPEAKER_00, SPEAKER_01, ...) and do NOT "
        "attribute statements to individual speakers. Write impersonal, topic-focused "
        'prose: "The discussion covers...", "It was decided...".\n'
        "- Real person names may be kept only when the content is ABOUT that person "
        "(e.g. an owner of an action item), not as the subject of who said what.\n"
        "- Inside the JSON string values use \\n for line breaks.\n\n"
        "ONLY valid JSON. No code fences, no text outside the JSON.\n\n"
        "EXCERPT:\n" + excerpt
    )


async def summarize_selection(
    text: str,
    *,
    backend: str = "openai",
    openai_base_url: str = "",
    openai_api_key: str = "",
    openai_model: str = "",
    ollama_base_url: str = "",
    ollama_model: str = "",
    temperature: float | None = None,
) -> dict:
    """Summarize a selected transcript excerpt into two variants.

    Returns {"topic": str, "with_examples": str, "without_examples": str}.
    """
    excerpt = (text or "").strip()[:_SELECTION_MAX_CHARS]
    prompt = _build_selection_prompt(excerpt)

    if backend == "openai":
        raw = await _call_openai_compatible(
            prompt, openai_base_url, openai_api_key, openai_model,
            temperature=temperature,
        )
    else:
        raw = await _call_ollama(prompt, ollama_base_url, ollama_model)

    if "<think>" in raw:
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("Selection summary: invalid JSON (first 300): %s", raw[:300])
        # Fall back: put the raw text into with_examples so the user still gets something
        data = {"topic": "Selection", "with_examples": raw.strip()[:2000], "without_examples": ""}

    data = _clean_json_strings(data)
    return {
        "topic": str(data.get("topic", "Selection")).strip() or "Selection",
        "with_examples": str(data.get("with_examples", "")).strip(),
        "without_examples": str(data.get("without_examples", "")).strip(),
    }
