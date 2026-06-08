"""Router for Web Search: ad-free results via SearXNG, optional AI answer."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from local_ai.auth import require_user
from local_ai.services.summarizer import clamp_temperature
from local_ai.services.websearch import (
    ALLOWED_ENGINES,
    SEARCH_TYPES,
    SHOPPING_REGIONS,
    SearchError,
    TIME_RANGES,
    answer_with_sources,
    search,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/search")
async def search_page(request: Request, user: dict = Depends(require_user)):
    settings = request.app.state.settings
    resp = request.app.state.templates.TemplateResponse(
        request,
        "search.html",
        {
            "user": user,
            "search_enabled": bool(settings.searxng_url),
            "engines": ALLOWED_ENGINES,
        },
    )
    # Prevent stale-cache rendering of the page shell.
    resp.headers["Cache-Control"] = "no-store, must-revalidate"
    return resp


@router.post("/api/search")
async def api_search(request: Request, user: dict = Depends(require_user)):
    """Run a web search. Body: {query, mode: "links"|"answer", language, count}.

    Returns ``{results: [...], answer: str|None, error: str|None}``.
    """
    settings = request.app.state.settings

    body = await request.json()
    query = (body.get("query") or "").strip()
    mode = (body.get("mode") or "links").lower()       # "links" | "answer"
    search_type = (body.get("search_type") or "web").lower()
    if search_type not in SEARCH_TYPES:
        search_type = "web"
    shopping_region = (body.get("shopping_region") or "de").lower()
    if shopping_region not in SHOPPING_REGIONS:
        shopping_region = "de"
    # AI-answer synthesis only makes sense for textual web results.
    if search_type != "web":
        mode = "links"
    language = (body.get("language") or "auto").lower()
    count = int(body.get("count") or 10)
    # Image grids look better with more tiles.
    count = max(1, min(40 if search_type == "images" else 20, count))
    temperature = clamp_temperature(body.get("temperature"))

    # Time filter: "" / day / week / month / year
    time_range = (body.get("time_range") or "").lower()
    if time_range not in TIME_RANGES:
        time_range = ""
    # Engine filter: list of engine names (validated against the allow-list)
    raw_engines = body.get("engines") or []
    if isinstance(raw_engines, str):
        raw_engines = [e for e in raw_engines.split(",") if e]
    engines = [e for e in raw_engines if e in ALLOWED_ENGINES]

    if not query:
        return JSONResponse({"error": "No query provided."}, status_code=400)
    if not settings.searxng_url:
        return JSONResponse({"error": "Web search is not configured."}, status_code=503)

    try:
        results = await search(
            query,
            searxng_url=settings.searxng_url,
            count=count,
            language=language,
            time_range=time_range,
            engines=engines,
            search_type=search_type,
            shopping_region=shopping_region,
        )
    except SearchError as exc:
        return JSONResponse({"error": str(exc)}, status_code=502)

    answer = None
    answer_error = None
    if mode == "answer" and results:
        # Pick the answer language: explicit choice, else infer from query.
        ans_lang = language if language in ("en", "de") else _guess_lang(query)
        try:
            answer = await answer_with_sources(
                query,
                results,
                language=ans_lang,
                backend=settings.summary_backend,
                openai_base_url=settings.openai_base_url,
                openai_api_key=settings.openai_api_key,
                openai_model=settings.openai_model,
                ollama_base_url=settings.ollama_base_url,
                ollama_model=settings.ollama_model,
                temperature=temperature,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("AI answer generation failed")
            answer_error = f"AI answer failed: {type(exc).__name__}: {exc}"

    return JSONResponse({
        "query": query,
        "search_type": search_type,
        "results": results,
        "answer": answer,
        "error": answer_error,
    })


def _guess_lang(text: str) -> str:
    """Lightweight DE/EN guess for the answer language (umlauts → de)."""
    if any(ch in "äöüÄÖÜß" for ch in text):
        return "de"
    return "en"
