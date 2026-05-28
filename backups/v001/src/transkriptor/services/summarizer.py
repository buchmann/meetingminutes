import json
import logging

import ollama as ollama_client

from transkriptor.models import ActionItem, SummaryResult, TopicDetail, TranscriptResult

logger = logging.getLogger(__name__)


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


SUMMARY_PROMPT_EN = """You are an expert meeting analyst. Analyze this transcript thoroughly and produce a detailed structured summary in JSON format.

The JSON must have exactly these fields:
- "overall_summary": A detailed summary of the entire meeting (8-15 sentences). Cover the main purpose, key discussion points, outcomes, and overall tone. Mention which speakers contributed what.
- "key_topics": Array of objects, each with:
  - "name": Short topic title
  - "summary": 2-4 sentence description of what was discussed about this topic
  - "timestamp_start": Approximate timestamp when this topic began (e.g. "00:05:23")
  - "speakers_involved": Array of speaker labels who discussed this topic
- "action_items": Array of objects with:
  - "description": Detailed description of what needs to be done
  - "assignee": Speaker label of the person responsible (or null if unclear)
  - "deadline": Any mentioned deadline or timeframe (or null)
- "key_decisions": Array of strings — be specific about what was decided, by whom, and why
- "timeline": Array of strings — chronological list of major moments in the meeting (e.g. "00:00 - Meeting opened by SPEAKER_00 with agenda overview", "00:05 - Discussion of Q2 results")
- "participants": Array of strings — the speaker labels found in the transcript, with a brief note on their apparent role if identifiable (e.g. "SPEAKER_00 (moderator)")

Be thorough and extract as much useful information as possible. Respond ONLY with valid JSON, no markdown, no explanation.

TRANSCRIPT:
{transcript}"""

SUMMARY_PROMPT_DE = """Du bist ein Experte fuer Meeting-Analyse. Analysiere dieses Transkript gruendlich und erstelle eine detaillierte strukturierte Zusammenfassung im JSON-Format.

Das JSON muss genau diese Felder haben:
- "overall_summary": Eine ausfuehrliche Zusammenfassung des gesamten Meetings (8-15 Saetze). Behandle den Hauptzweck, wichtige Diskussionspunkte, Ergebnisse und den allgemeinen Ton. Erwaehne, welche Sprecher was beigetragen haben.
- "key_topics": Array von Objekten, jeweils mit:
  - "name": Kurzer Thementitel
  - "summary": 2-4 Saetze Beschreibung, was zu diesem Thema besprochen wurde
  - "timestamp_start": Ungefaehrer Zeitstempel, wann dieses Thema begann (z.B. "00:05:23")
  - "speakers_involved": Array der Sprecherbezeichnungen, die an diesem Thema beteiligt waren
- "action_items": Array von Objekten mit:
  - "description": Detaillierte Beschreibung der Aufgabe
  - "assignee": Sprecherbezeichnung der verantwortlichen Person (oder null)
  - "deadline": Genannter Termin oder Zeitrahmen (oder null)
- "key_decisions": Array von Strings — sei spezifisch darueber, was entschieden wurde, von wem und warum
- "timeline": Array von Strings — chronologische Liste der wichtigsten Momente im Meeting (z.B. "00:00 - Meeting eroeffnet von SPEAKER_00 mit Agendauebersicht")
- "participants": Array von Strings — die im Transkript gefundenen Sprecherbezeichnungen, mit kurzer Bemerkung zur erkennbaren Rolle (z.B. "SPEAKER_00 (Moderator)")

Sei gruendlich und extrahiere so viele nuetzliche Informationen wie moeglich. Antworte NUR mit validem JSON, kein Markdown, keine Erklaerung.

TRANSKRIPT:
{transcript}"""


async def summarize(
    transcript: TranscriptResult,
    ollama_base_url: str,
    model: str,
    language: str = "auto",
) -> SummaryResult:
    dominant_lang = _detect_dominant_language(transcript)
    summary_lang = dominant_lang if language == "auto" else language

    prompt_template = SUMMARY_PROMPT_DE if summary_lang == "de" else SUMMARY_PROMPT_EN
    formatted = _format_transcript_for_prompt(transcript)

    max_chars = 60000
    if len(formatted) > max_chars:
        formatted = formatted[:max_chars] + "\n[... transcript truncated ...]"

    prompt = prompt_template.format(transcript=formatted)

    client = ollama_client.AsyncClient(host=ollama_base_url)
    logger.info("Requesting summary from %s (model=%s, lang=%s)...", ollama_base_url, model, summary_lang)

    response = await client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0.3},
    )

    raw_text = response["message"]["content"]

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.error("LLM returned invalid JSON: %s", raw_text[:200])
        data = {
            "overall_summary": raw_text[:500],
            "key_topics": [],
            "action_items": [],
            "key_decisions": [],
            "participants": [],
        }

    action_items = []
    for item in data.get("action_items", []):
        if isinstance(item, str):
            action_items.append(ActionItem(description=item))
        elif isinstance(item, dict):
            action_items.append(ActionItem(
                description=item.get("description", str(item)),
                assignee=item.get("assignee"),
                deadline=item.get("deadline"),
            ))

    topics = []
    for t in data.get("key_topics", []):
        if isinstance(t, str):
            topics.append(TopicDetail(name=t, summary=""))
        elif isinstance(t, dict):
            topics.append(TopicDetail(
                name=t.get("name", str(t)),
                summary=t.get("summary", ""),
                timestamp_start=t.get("timestamp_start"),
                speakers_involved=t.get("speakers_involved"),
            ))

    return SummaryResult(
        overall_summary=data.get("overall_summary", ""),
        key_topics=topics,
        action_items=action_items,
        key_decisions=data.get("key_decisions", []),
        participants=data.get("participants", []),
        timeline=data.get("timeline"),
        language=summary_lang,
    )
