"""Daily companion ("Liebling, wie war dein Tag?") — Phase 1.

A short conversational daily check-in. Two LLM steps:
  - chat(): a warm, multi-turn conversation, aware of the user's projects.
  - finalize(): turn the conversation into a STRUCTURED proposal of updates
    (per-project logbook entries, TODOs, optional new projects + a day summary).
The router then shows the proposal for confirmation and writes it on apply.
"""

from __future__ import annotations

import json
import logging
import re

try:
    import httpx
    HAS_HTTPX = True
except ImportError:  # pragma: no cover
    HAS_HTTPX = False

from local_ai.services.summarizer import _effective_reasoning_effort

logger = logging.getLogger(__name__)

# Keep the chat short and focused.
MAX_TURNS = 12

DAILY_PERSONA = (
    "Du bist ein warmherziger, aufmerksamer Projekt-Begleiter — wie ein guter "
    "Kollege, der sich abends kurz erkundigt: „Wie war dein Tag, woran hast du "
    "gearbeitet?\". Sprich Deutsch, locker und freundlich, aber nicht kitschig. "
    "Stelle EINE Frage pro Antwort, halte dich kurz. Frage gezielt nach: an "
    "welchem Projekt, was ist vorangekommen, welche Entscheidungen, welche "
    "Blocker, was sind die nächsten Schritte. Nach ein paar Wortwechseln (etwa "
    "3-6) biete an, den Tag abzuschließen ('Sollen wir den Tag festhalten?'). "
    "Erfinde nichts — frage nach, wenn etwas unklar ist."
)


def build_projects_context(projects: list[dict]) -> str:
    """Compact list of the user's projects so the assistant has context."""
    if not projects:
        return "Der Nutzer hat noch keine Projekte angelegt."
    lines = ["Bestehende Projekte des Nutzers (Name — Beschreibung):"]
    for p in projects:
        desc = (p.get("description") or "").strip().replace("\n", " ")
        lines.append(f"- {p['name']}" + (f" — {desc[:120]}" if desc else ""))
    return "\n".join(lines)


async def _chat_completion(messages: list[dict], settings, *,
                           response_format: str | None = None,
                           max_tokens: int = 1200) -> str:
    """Minimal messages-based chat call against the active vLLM model."""
    if not HAS_HTTPX:
        raise RuntimeError("httpx required for the daily companion")
    base = settings.openai_base_url.rstrip("/")
    model = settings.openai_model
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.5,
        "top_p": 0.9,
        "max_tokens": max_tokens,
    }
    effort = _effective_reasoning_effort(model, None)  # gpt-oss → "low"
    if effort:
        payload["reasoning_effort"] = effort
    if response_format == "json_object":
        payload["response_format"] = {"type": "json_object"}
        payload["temperature"] = 0.2
    headers = {"Content-Type": "application/json"}
    key = settings.openai_api_key
    if key and key != "none":
        headers["Authorization"] = f"Bearer {key}"
    async with httpx.AsyncClient(timeout=300.0) as client:
        r = await client.post(f"{base}/chat/completions", json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"].get("content") or ""


async def chat(history: list[dict], projects: list[dict], settings) -> str:
    """One conversational turn. *history* is a list of {role, content} (user/assistant)."""
    system = DAILY_PERSONA + "\n\n" + build_projects_context(projects)
    messages = [{"role": "system", "content": system}]
    messages += [{"role": m["role"], "content": m["content"]}
                 for m in history[-(2 * MAX_TURNS):] if m.get("content")]
    return (await _chat_completion(messages, settings, max_tokens=600)).strip()


_FINALIZE_INSTRUCTION = (
    "Fasse das folgende Tages-Gespräch in strukturierte Updates zusammen. "
    "Gib NUR ein JSON-Objekt zurück mit GENAU diesen Schlüsseln (deutsch):\n"
    "{\n"
    '  "daily_summary": "2-4 Sätze: was heute passiert ist (für das Tagebuch).",\n'
    '  "updates": [\n'
    '    {\n'
    '      "project": "exakter Name eines BESTEHENDEN Projekts oder null",\n'
    '      "new_project": "Name eines NEU anzulegenden Projekts oder null",\n'
    '      "new_project_description": "kurze Beschreibung oder null",\n'
    '      "logbook": "Logbuch-Eintrag für dieses Projekt (was passiert ist) oder null",\n'
    '      "todos": ["konkrete offene Aufgabe", "..."]\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "Regeln: Nutze NUR Inhalte aus dem Gespräch, erfinde nichts. Ordne Einträge "
    "dem genannten Projekt zu (exakter Name). Wenn ein klar neues Vorhaben "
    "genannt wurde, setze new_project. Leere Listen/null sind erlaubt. "
    "Kein Markdown, keine Code-Fences."
)


def _loads_json_lenient(raw: str) -> dict:
    s = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL).strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", s, flags=re.DOTALL)
    if m:
        s = m.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        i, j = s.find("{"), s.rfind("}")
        if i != -1 and j > i:
            return json.loads(s[i:j + 1])
        raise


async def finalize(history: list[dict], projects: list[dict], settings) -> dict:
    """Turn the conversation into a structured, not-yet-applied proposal."""
    convo = "\n".join(
        f"{'Du' if m['role']=='user' else 'Begleiter'}: {m['content']}"
        for m in history if m.get("content")
    )
    ctx = build_projects_context(projects)
    messages = [
        {"role": "system", "content": _FINALIZE_INSTRUCTION + "\n\n" + ctx},
        {"role": "user", "content": "GESPRÄCH:\n" + convo},
    ]
    raw = await _chat_completion(messages, settings, response_format="json_object", max_tokens=2000)
    data = _loads_json_lenient(raw)
    # normalise
    updates = []
    for u in (data.get("updates") or []):
        if not isinstance(u, dict):
            continue
        updates.append({
            "project": (u.get("project") or "").strip() or None,
            "new_project": (u.get("new_project") or "").strip() or None,
            "new_project_description": (u.get("new_project_description") or "").strip(),
            "logbook": (u.get("logbook") or "").strip(),
            "todos": [str(t).strip() for t in (u.get("todos") or []) if str(t).strip()],
        })
    return {
        "daily_summary": (data.get("daily_summary") or "").strip(),
        "updates": updates,
    }
