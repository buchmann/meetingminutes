"""Notes & Manuals knowledge base: keyword retrieval over the user's uploaded
documents + LLM summarisation/answer with source citations.

No embedding infra needed — chunks are scored by query-term overlap (TF with a
rarity weight), the top chunks are fed to the LLM, which answers strictly from
them and cites the source document. Good enough for "find it in my notes".
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter

import httpx

from local_ai.services.summarizer import _call_openai_compatible, _call_ollama

logger = logging.getLogger(__name__)

_CHUNK_CHARS = 900
_CHUNK_OVERLAP = 150
_TOP_K = 6


# ── Embeddings (vLLM OpenAI-compatible /v1/embeddings) ────────────────────

async def embed_texts(texts: list[str], base_url: str, model: str) -> list[list[float]] | None:
    """Return one embedding per input text, or None on failure."""
    if not texts or not base_url:
        return None
    url = base_url.rstrip("/") + "/embeddings"
    try:
        async with httpx.AsyncClient(timeout=60.0) as c:
            r = await c.post(url, json={"model": model, "input": texts})
            r.raise_for_status()
            data = r.json()
        # Preserve input order (OpenAI returns objects with "index").
        rows = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
        return [row["embedding"] for row in rows]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Embedding request failed: %s", exc)
        return None


def pack_vector(vec: list[float]) -> bytes:
    import numpy as np
    return np.asarray(vec, dtype=np.float32).tobytes()


async def embedding_available(base_url: str) -> bool:
    if not base_url:
        return False
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(base_url.rstrip("/") + "/models")
            return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


async def semantic_retrieve(query: str, chunks: list[dict], base_url: str,
                            model: str, top_k: int = _TOP_K) -> list[dict] | None:
    """Cosine similarity between the query and pre-embedded chunks.

    ``chunks`` items: {"doc_id","title","text","embedding"(float32 bytes)}.
    Returns top-k [{doc_id,title,text,score}] or None if unusable.
    """
    import numpy as np

    valid, vecs = [], []
    for ch in chunks:
        emb = ch.get("embedding")
        if emb:
            vecs.append(np.frombuffer(emb, dtype=np.float32))
            valid.append(ch)
    if not valid:
        return None
    qv = await embed_texts([query], base_url, model)
    if not qv:
        return None

    q = np.asarray(qv[0], dtype=np.float32)
    M = np.vstack(vecs)
    q = q / (np.linalg.norm(q) + 1e-8)
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
    sims = M @ q
    order = np.argsort(-sims)[:top_k]
    out = []
    for i in order:
        out.append({
            "doc_id": valid[int(i)]["doc_id"],
            "title": valid[int(i)]["title"],
            "text": valid[int(i)]["text"],
            "score": round(float(sims[int(i)]), 3),
        })
    return out or None

# Tiny multilingual stopword set (DE + EN) so scoring isn't dominated by glue words.
_STOP = set("""
der die das den dem des ein eine einer eines einem einen und oder aber nicht
ist sind war waren wird werden mit ohne auf für fuer von zu im am in an als wie
was wer wo wann auch noch nur sehr dass weil wenn the a an and or but not is are
was were will would with without for of to in on at as how what who where when
this that these those it its be been being have has had do does did i you he she
we they me my your our their
""".split())


def _tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[A-Za-zÄÖÜäöüß0-9]+", text.lower()) if len(w) > 1]


def chunk_text(text: str) -> list[str]:
    text = text.strip()
    if len(text) <= _CHUNK_CHARS:
        return [text] if text else []
    chunks, i = [], 0
    while i < len(text):
        chunk = text[i:i + _CHUNK_CHARS]
        chunks.append(chunk)
        i += _CHUNK_CHARS - _CHUNK_OVERLAP
    return chunks


def _retrieve(query: str, docs: list[dict], top_k: int = _TOP_K) -> list[dict]:
    """Return the top-scoring chunks across all docs for the query.

    Each returned item: {"doc_id", "title", "text", "score"}.
    """
    q_terms = [t for t in _tokenize(query) if t not in _STOP]
    if not q_terms:
        return []
    q_set = set(q_terms)

    # Build chunk list + document frequency for rarity weighting.
    all_chunks: list[dict] = []
    df: Counter = Counter()
    for d in docs:
        for ch in chunk_text(d.get("content") or ""):
            toks = set(_tokenize(ch))
            all_chunks.append({"doc_id": d["id"], "title": d["title"], "text": ch, "_toks": toks})
            for term in q_set & toks:
                df[term] += 1
    if not all_chunks:
        return []
    n_chunks = len(all_chunks)

    scored = []
    for ch in all_chunks:
        toks = ch["_toks"]
        score = 0.0
        for term in q_terms:
            if term in toks:
                # idf-ish weight: rarer query terms count more.
                idf = math.log((n_chunks + 1) / (df.get(term, 0) + 1)) + 1.0
                score += idf
        if score > 0:
            scored.append((score, ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for score, ch in scored[:top_k]:
        out.append({"doc_id": ch["doc_id"], "title": ch["title"],
                    "text": ch["text"], "score": round(score, 2)})
    return out


def _build_prompt(query: str, chunks: list[dict], language: str) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[{i}] (Quelle: {c['title']})\n{c['text']}")
    sources = "\n\n".join(blocks)
    if language == "de":
        return (
            "Beantworte die Frage des Nutzers AUSSCHLIESSLICH auf Basis der "
            "untenstehenden Auszüge aus seinen Notizen und Handbüchern.\n\n"
            "REGELN\n"
            "- Nutze NUR die Auszüge. Erfinde nichts.\n"
            "- Zitiere die Quelle inline mit [Nummer], z. B. [1], [3].\n"
            "- Wenn die Auszüge die Frage nicht beantworten, sage das klar.\n"
            "- Antworte präzise auf Deutsch, fasse zusammen, keine Wiederholung der Frage.\n\n"
            f"FRAGE: {query}\n\n"
            f"AUSZÜGE:\n{sources}\n\n"
            "Antwort (mit [n]-Zitaten):"
        )
    return (
        "Answer the user's question STRICTLY from the excerpts of their notes "
        "and manuals below.\n\n"
        "RULES\n"
        "- Use ONLY the excerpts. Invent nothing.\n"
        "- Cite the source inline with [number], e.g. [1], [3].\n"
        "- If the excerpts don't answer it, say so plainly.\n"
        "- Be concise, summarise, do not restate the question.\n\n"
        f"QUESTION: {query}\n\n"
        f"EXCERPTS:\n{sources}\n\n"
        "Answer (with [n] citations):"
    )


def _guess_lang(text: str) -> str:
    return "de" if any(c in "äöüÄÖÜß" for c in text) else "en"


async def search_notes(query: str, docs: list[dict], *, llm: dict,
                       language: str = "auto",
                       embedded_chunks: list[dict] | None = None,
                       embedding_base_url: str = "", embedding_model: str = "") -> dict:
    """Retrieve relevant chunks and produce a cited summary answer.

    Hybrid retrieval: if ``embedded_chunks`` and an embedding endpoint are
    available, use semantic (cosine) search; otherwise fall back to keyword
    retrieval over the full document text.

    Returns ``{"answer", "sources":[...], "matched", "mode", "error"}``.
    """
    query = (query or "").strip()
    if not query:
        return {"answer": "", "sources": [], "matched": 0, "error": "Keine Frage angegeben."}
    if not docs and not embedded_chunks:
        return {"answer": "", "sources": [], "matched": 0,
                "error": "Noch keine Dokumente hochgeladen."}

    chunks = None
    mode = "keyword"
    if embedded_chunks and embedding_base_url:
        chunks = await semantic_retrieve(query, embedded_chunks, embedding_base_url, embedding_model)
        if chunks:
            mode = "semantic"
    if not chunks:
        chunks = _retrieve(query, docs or [])   # keyword fallback
    if not chunks:
        return {"answer": "", "sources": [], "matched": 0,
                "error": "Nichts Passendes in den Dokumenten gefunden."}

    lang = language if language in ("de", "en") else _guess_lang(query)
    prompt = _build_prompt(query, chunks, lang)

    try:
        if llm["backend"] == "openai":
            raw = await _call_openai_compatible(
                prompt, llm["openai_base_url"], llm["openai_api_key"], llm["openai_model"],
                response_format="text", temperature=0.1,
            )
        else:
            raw = await _call_ollama(prompt, llm["ollama_base_url"], llm["ollama_model"], 0.1)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Notes search LLM failed")
        return {"answer": "", "sources": [], "matched": len(chunks),
                "error": f"LLM-Fehler: {type(exc).__name__}: {exc}"}

    sources = [{
        "n": i + 1,
        "title": c["title"],
        "snippet": " ".join(c["text"].split())[:240],
        "score": c["score"],
    } for i, c in enumerate(chunks)]

    return {"answer": (raw or "").strip(), "sources": sources,
            "matched": len(chunks), "mode": mode, "error": None}
