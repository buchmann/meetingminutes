"""Translator: high-fidelity bidirectional English <-> German translation
for both pasted text and uploaded documents.

The translator deliberately does NOT rewrite or improve the text — only
translate. Names, technical terms, code identifiers, URLs and numbers are
kept verbatim. Paragraph structure is preserved.

Long inputs are split at paragraph boundaries (reusing
:func:`document_checker._split_into_chunks`) and each chunk is translated
independently, then re-joined.

The "auto" direction does a lightweight character-frequency check to pick
the source language; the LLM is told the source and target explicitly so
mis-detection never silently produces a no-op translation.
"""

from __future__ import annotations

import logging
import re

from local_ai.services.document_checker import _split_into_chunks
from local_ai.services.summarizer import _call_ollama, _call_openai_compatible
from local_ai.services.text_improver import MAX_INPUT_CHARS

logger = logging.getLogger(__name__)


SUPPORTED_LANGS = ("en", "de")
LANG_NAMES = {"en": "English", "de": "German (Deutsch)"}


# ── Direction detection ───────────────────────────────────────────────────

_DE_HINT_CHARS = set("äöüÄÖÜß")

# Tokens that strongly indicate German when they appear as standalone words.
# Kept lower-case; matching happens on the lowercased input.
_DE_HINT_WORDS = {
    # articles + pronouns
    "der", "die", "das", "den", "des", "dem",
    "ein", "eine", "einer", "eines", "einem", "einen",
    "ich", "du", "er", "sie", "es", "wir", "ihr", "mir", "dir", "ihm",
    "uns", "euch", "ihnen", "mich", "dich", "sich", "mein", "dein",
    "sein", "unser", "euer", "ihre", "ihrer", "ihren", "ihres", "ihrem",
    "dieser", "diese", "dieses", "diesem", "diesen", "jeder", "jede",
    "jedes", "kein", "keine", "keinen", "keiner", "keines", "keinem",
    # connectors / particles
    "und", "oder", "aber", "doch", "denn", "weil", "wenn", "dass", "daß",
    "ob", "als", "wie", "was", "wer", "wo", "wann", "warum", "woher",
    "wohin", "wieso", "weshalb",
    # prepositions
    "mit", "ohne", "nach", "vor", "bei", "aus", "von", "zum", "zur",
    "im", "am", "auf", "in", "an", "zu", "für", "fuer", "über", "ueber",
    "unter", "neben", "zwischen", "gegen", "durch", "um", "seit", "bis",
    # verbs (common conjugations)
    "ist", "sind", "war", "waren", "wird", "werden", "wurde", "wurden",
    "habe", "haben", "hat", "hatte", "hatten", "hast",
    "kann", "können", "koennen", "konnte", "konnten",
    "muss", "müssen", "muessen", "musste", "mussten",
    "soll", "sollen", "sollte", "sollten",
    "möchte", "moechte", "möchten", "moechten",
    "geht", "gehen", "ging", "gegangen", "kommt", "kommen", "kam",
    "macht", "machen", "machte", "gemacht", "sagt", "sagen", "sagte",
    "gibt", "geben", "gab", "gegeben", "sieht", "sehen", "sah", "gesehen",
    "weiß", "weiss", "wissen", "wusste", "gewusst", "denke", "denken",
    "glaube", "glauben", "brauche", "brauchen",
    # adverbs / common short
    "auch", "noch", "schon", "nur", "sehr", "mehr", "weniger", "viel",
    "wenig", "gut", "schlecht", "groß", "gross", "klein", "alt", "neu",
    "ja", "nein", "nicht", "doch", "vielleicht", "natürlich", "natuerlich",
    "hier", "dort", "da", "heute", "morgen", "gestern", "jetzt", "dann",
    # greetings + small talk
    "hallo", "hi", "moin", "servus", "tschüss", "tschuess", "tschau",
    "guten", "morgen", "tag", "abend", "nacht", "danke", "bitte",
}

_EN_HINT_WORDS = {
    # articles + pronouns
    "the", "a", "an",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those",
    # connectors
    "and", "or", "but", "because", "if", "when", "while", "although",
    "though", "since", "unless", "until", "whether",
    # interrogatives
    "what", "where", "who", "whom", "why", "how", "which",
    # prepositions
    "of", "for", "on", "in", "at", "by", "from", "to", "with", "without",
    "about", "after", "before", "between", "during", "through", "under",
    "over", "above", "into", "onto", "upon",
    # verbs
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing", "done",
    "will", "would", "shall", "should", "can", "could", "may", "might",
    "must", "ought",
    "go", "goes", "went", "gone", "going",
    "come", "comes", "came", "coming",
    "say", "says", "said", "saying",
    "see", "sees", "saw", "seen",
    "know", "knows", "knew", "known",
    "think", "thinks", "thought",
    "want", "wants", "wanted", "need", "needs", "needed",
    # adverbs / common short
    "not", "no", "yes", "also", "very", "only", "still", "already",
    "just", "well", "more", "less", "much", "many", "few", "some",
    "any", "all", "every", "each", "here", "there", "now", "then",
    "today", "tomorrow", "yesterday", "soon", "later",
    # greetings + small talk
    "hello", "hi", "hey", "good", "morning", "evening", "night",
    "thanks", "please", "okay", "ok",
}


# Character n-gram patterns: bigrams/trigrams that are far more common in one
# language than the other. Used as a tiebreaker when word-list scoring is even
# (e.g. very short inputs with no listed words).
_DE_NGRAMS = ("sch", "chen", "lich", "keit", "ung", "ich", "ach", "och",
              "uch", "tz", "tsch", "pf", "stra", "zwi")
_EN_NGRAMS = ("th", "ing", "tion", "ed ", "ly ", "wh", "ough", "ould",
              "ight", "tch", "sh", "wo ", "you")


def _ngram_score(text: str, ngrams: tuple[str, ...]) -> int:
    return sum(text.count(n) for n in ngrams)


def detect_language(text: str) -> str:
    """Return 'en' or 'de' for the dominant language of *text*.

    Strategy (in order of confidence):
      1. Any of ä/ö/ü/ß → German (decisive).
      2. Score tokens against German vs English hint lists.
      3. Tiebreak with character n-gram counts.
      4. Default to English (longest-served pattern in the rest of the app).
    """
    if not text:
        return "en"
    sample = text[:4000]
    lower = sample.lower()

    # 1. Umlauts / sharp-s are decisive
    if any(ch in _DE_HINT_CHARS for ch in sample):
        return "de"

    # 2. Token-based scoring
    words = re.findall(r"[A-Za-zÄÖÜäöüß]+", lower)
    de = sum(1 for w in words if w in _DE_HINT_WORDS)
    en = sum(1 for w in words if w in _EN_HINT_WORDS)
    if de > en:
        return "de"
    if en > de:
        return "en"

    # 3. Tiebreak: character n-grams
    de_ng = _ngram_score(lower, _DE_NGRAMS)
    en_ng = _ngram_score(lower, _EN_NGRAMS)
    if de_ng > en_ng:
        return "de"
    if en_ng > de_ng:
        return "en"

    # 4. Fall through to English default
    return "en"


def resolve_direction(direction: str, text: str) -> tuple[str, str]:
    """Turn a user-supplied direction into (source_lang, target_lang).

    Accepts: 'en2de', 'de2en', 'auto2de', 'auto2en', 'auto'.
    'auto' picks source via :func:`detect_language` and flips to the other.
    """
    direction = (direction or "auto").lower().strip()
    if direction == "en2de":
        return ("en", "de")
    if direction == "de2en":
        return ("de", "en")
    if direction == "auto2de":
        return ("en" if detect_language(text) == "en" else "de", "de")
    if direction == "auto2en":
        return ("de" if detect_language(text) == "de" else "en", "en")
    # Plain "auto" — flip to the other language
    source = detect_language(text)
    target = "de" if source == "en" else "en"
    return (source, target)


# ── Prompt builders ───────────────────────────────────────────────────────


def _build_prompt(text: str, source_lang: str, target_lang: str,
                  style_profile: str | None) -> str:
    """Build the LLM prompt for one chunk of text."""
    src_name = LANG_NAMES.get(source_lang, source_lang)
    tgt_name = LANG_NAMES.get(target_lang, target_lang)

    if target_lang == "de":
        instructions = (
            "Du bist ein professioneller Uebersetzer. Uebersetze den unten "
            f"stehenden Text aus dem {src_name} ins Deutsche.\n\n"
            "REGELN\n"
            "- Gib AUSSCHLIESSLICH die Uebersetzung aus. Keine Vorrede, "
            "keine Erklaerung, keine Code-Fences, keine Anmerkungen.\n"
            "- Uebersetze inhaltsgetreu - nicht umformulieren, nicht "
            "zusammenfassen, nicht kommentieren.\n"
            "- Behalte die Absatzstruktur, Listen, Zeileneinrueckungen und "
            "Markdown-Auszeichnungen exakt bei.\n"
            "- Eigennamen, Produktnamen, Toolnamen, Personennamen, "
            "URLs, Dateinamen, Code-Bezeichner, Versionsnummern und "
            "Zahlen bleiben im Original (z. B. Ansible Automation "
            "Platform, BMC TrueSight, OPA, Kubernetes, GitHub).\n"
            "- Etablierte englische Fachbegriffe (Workflow, Container, "
            "Deployment, Policy, etc.) duerfen erhalten bleiben, wenn "
            "sie im deutschen Sprachgebrauch ueblich sind.\n"
            "- Verwende die deutsche Rechtschreibung mit Umlauten "
            "(ae/oe/ue/ss sind NICHT erwuenscht im endgueltigen Output).\n"
            "- Keine Erklaerungen in Klammern. Wenn ein Begriff nicht "
            "uebersetzt werden kann, lass ihn unveraendert.\n"
        )
    else:
        instructions = (
            "You are a professional translator. Translate the text below "
            f"from {src_name} into English.\n\n"
            "RULES\n"
            "- Output ONLY the translation. No preamble, no explanation, "
            "no code fences, no notes.\n"
            "- Translate faithfully - do not rephrase, do not summarize, "
            "do not comment.\n"
            "- Preserve paragraph structure, lists, indentation and "
            "Markdown formatting exactly.\n"
            "- Proper names, product names, tool names, personal names, "
            "URLs, filenames, code identifiers, version numbers and "
            "numerals stay in the original (e.g. Ansible Automation "
            "Platform, BMC TrueSight, OPA, Kubernetes, GitHub).\n"
            "- Use natural, idiomatic English. Do not over-literalise "
            "German compound nouns; restructure into normal English "
            "phrasing where appropriate.\n"
            "- No parenthetical explanations. If a term cannot be "
            "translated, leave it in the original.\n"
        )

    if style_profile:
        if target_lang == "de":
            style_block = (
                "\nSTILRICHTLINIE (nur auf die Wortwahl und den Ton anwenden, "
                "NICHT auf die Struktur):\n" + style_profile + "\n"
            )
        else:
            style_block = (
                "\nSTYLE GUIDE (apply only to word choice and tone, NOT to "
                "structure):\n" + style_profile + "\n"
            )
        instructions = instructions + style_block

    text_label = "TEXT" if target_lang == "en" else "TEXT"
    return (
        f"{instructions}\n"
        f"{text_label} ({src_name} -> {tgt_name}):\n"
        "<<<\n"
        f"{text}\n"
        ">>>\n\n"
        "Now output the translation only."
    )


# ── Translate ─────────────────────────────────────────────────────────────


def _clean_response(text: str) -> str:
    """Strip code fences and the <<<>>> markers if the LLM accidentally
    echoed them back, plus any obvious preamble."""
    t = (text or "").strip()
    if t.startswith("```"):
        first_nl = t.find("\n")
        if first_nl >= 0:
            t = t[first_nl + 1:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3].rstrip()
    # Strip <<< / >>> markers if the model echoed them
    t = re.sub(r"^<{3,}\s*\n?", "", t)
    t = re.sub(r"\n?>{3,}\s*$", "", t)
    # Strip common preambles ("Here is the translation:", "Hier ist die Uebersetzung:")
    t = re.sub(
        r"^(here(?:'s| is)|hier(?:\s+ist|\s+sind)|nachfolgend|im folgenden)[^\n]*:\s*\n",
        "",
        t,
        flags=re.IGNORECASE,
    )
    return t.strip()


async def translate_text(
    text: str,
    *,
    direction: str = "auto",
    style_profile: str | None = None,
    backend: str = "openai",
    openai_base_url: str = "",
    openai_api_key: str = "",
    openai_model: str = "",
    ollama_base_url: str = "",
    ollama_model: str = "",
    temperature: float | None = None,
) -> dict:
    """Translate *text* end-to-end.

    ``temperature`` is the "hallucination" dial; None uses the default 0.1
    (translation should stay faithful, so a low value is the sensible default).

    Returns ``{"original", "translated", "source_lang", "target_lang",
    "chunks", "error"}``.

    Long inputs are split at paragraph boundaries and translated chunk by
    chunk. The first non-trivial error from any chunk is returned in
    ``error`` while the rest of the translation is still produced (with the
    failing chunk kept as original to preserve structure).
    """
    if not text or not text.strip():
        return {
            "original": text or "",
            "translated": "",
            "source_lang": "en",
            "target_lang": "en",
            "chunks": 0,
            "error": "No text provided.",
        }

    source_lang, target_lang = resolve_direction(direction, text)
    if source_lang == target_lang:
        return {
            "original": text,
            "translated": text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "chunks": 0,
            "error": "Source and target language are the same — nothing to translate.",
        }

    chunk_budget = max(2000, MAX_INPUT_CHARS - 700)
    chunks = _split_into_chunks(text, chunk_budget)
    logger.info(
        "Translate %s->%s: %d chars in %d chunk(s)",
        source_lang, target_lang, len(text), len(chunks),
    )

    translated_parts: list[str] = []
    first_error: str | None = None

    for i, chunk in enumerate(chunks, start=1):
        prompt = _build_prompt(chunk, source_lang, target_lang, style_profile)
        logger.info("Translating chunk %d/%d (%d chars)", i, len(chunks), len(chunk))
        try:
            if backend == "openai":
                raw = await _call_openai_compatible(
                    prompt, openai_base_url, openai_api_key, openai_model,
                    response_format="text",  # plain prose; do NOT force JSON
                    temperature=temperature,
                )
            else:
                raw = await _call_ollama(
                    prompt, ollama_base_url, ollama_model, temperature,
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Translation chunk %d failed", i)
            translated_parts.append(chunk)  # keep original for this chunk
            if first_error is None:
                first_error = f"Chunk {i}: {type(exc).__name__}: {exc}"
            continue

        cleaned = _clean_response(raw)
        if cleaned:
            translated_parts.append(cleaned)
        else:
            translated_parts.append(chunk)
            if first_error is None:
                first_error = f"Chunk {i}: empty response from LLM"

    return {
        "original": text,
        "translated": "\n\n".join(translated_parts),
        "source_lang": source_lang,
        "target_lang": target_lang,
        "chunks": len(chunks),
        "error": first_error,
    }
