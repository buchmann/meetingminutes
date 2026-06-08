"""Web search via self-hosted SearXNG (ad-free metasearch with JSON API).

Two capabilities:
  * ``search()`` — return clean organic results (title, url, snippet, engine),
    no ads, no tracking.
  * ``answer_with_sources()`` — feed the top results to the configured LLM and
    return a synthesised answer that cites the sources by number.

SearXNG runs in-cluster (see ``k8s/searxng.yaml``); the app reaches it at
``settings.searxng_url`` (default ``http://searxng:8080``).
"""

from __future__ import annotations

import logging

import httpx

from local_ai.services.summarizer import _call_openai_compatible, _call_ollama

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(connect=8.0, read=25.0, write=10.0, pool=8.0)


class SearchError(RuntimeError):
    """Raised when the search backend is unreachable or returns an error."""


# Web engines we expose in the per-engine UI filter. Restricted to the ones
# that reliably return results from this SearXNG instance (verified live):
# DuckDuckGo/Brave block SearXNG scraping (0 results), Wikipedia only matches
# article titles — those are still used in the default "all engines" merge but
# are not offered as a standalone filter (which would surprise with 0 hits).
ALLOWED_ENGINES = ("google", "bing", "startpage")
TIME_RANGES = ("day", "week", "month", "year")

# Search types exposed in the UI.
SEARCH_TYPES = ("web", "images", "shopping")

# "Shopping" has no dedicated engine in this SearXNG build, so we run a normal
# web search restricted (via the site: operator on Google/Bing) to major
# shopping domains. Selectable per region.
SHOPPING_REGIONS = ("de", "eu", "intl")
SHOPPING_SITES_BY_REGION = {
    "de": (
        "amazon.de", "ebay.de", "otto.de", "mediamarkt.de", "saturn.de",
        "idealo.de", "geizhals.de", "kaufland.de", "conrad.de",
        "notebooksbilliger.de", "alternate.de", "zalando.de",
    ),
    "eu": (
        "amazon.de", "amazon.fr", "amazon.it", "amazon.es", "amazon.nl",
        "amazon.co.uk", "ebay.de", "ebay.fr", "ebay.it", "bol.com",
        "cdiscount.com", "fnac.com", "idealo.de", "zalando.de",
    ),
    "intl": (
        "amazon.com", "amazon.co.uk", "ebay.com", "walmart.com", "bestbuy.com",
        "target.com", "aliexpress.com", "etsy.com", "newegg.com", "alibaba.com",
    ),
}


async def search(
    query: str,
    *,
    searxng_url: str,
    count: int = 10,
    language: str = "auto",
    time_range: str = "",
    engines: list[str] | None = None,
    categories: str = "general",
    search_type: str = "web",
    shopping_region: str = "de",
) -> list[dict]:
    """Run a search through SearXNG and return a list of result dicts.

    ``search_type``     — "web" (default), "images", or "shopping".
    ``shopping_region`` — "de"/"eu"/"intl" (only used for shopping).
    ``time_range``      — one of "day"/"week"/"month"/"year" (else unrestricted).
    ``engines``         — restrict to these engine names (web only).

    Each result: ``{"title", "url", "content", "engine"}`` plus, for images,
    ``{"img_src", "thumbnail", "resolution", "source"}``.
    Raises :class:`SearchError` if SearXNG is not configured/reachable.
    """
    query = (query or "").strip()
    if not query:
        return []
    if not searxng_url:
        raise SearchError("Web search is not configured (no SearXNG URL).")
    if search_type not in SEARCH_TYPES:
        search_type = "web"

    params = {
        "q": query,
        "format": "json",
        "safesearch": "0",
    }
    if language and language != "auto":
        params["language"] = language
    if time_range in TIME_RANGES:
        params["time_range"] = time_range

    if search_type == "images":
        # Image search — engine filter / categories handled here.
        params["categories"] = "images"
    elif search_type == "shopping":
        # Restrict a normal web search to shopping domains via site: OR-filter.
        # Google/Bing honour site:; other engines ignore it (harmless).
        region = shopping_region if shopping_region in SHOPPING_REGIONS else "de"
        site_list = SHOPPING_SITES_BY_REGION[region]
        sites = " OR ".join(f"site:{s}" for s in site_list)
        params["q"] = f"{query} ({sites})"
        params["engines"] = "google,bing"
    else:
        # Plain web. Restrict to chosen engines if any (SearXNG UNIONs
        # categories with engines, so send ONLY engines when a selection
        # is made — otherwise it would query all general engines).
        chosen = [e for e in (engines or []) if e in ALLOWED_ENGINES]
        if chosen:
            params["engines"] = ",".join(chosen)
        else:
            params["categories"] = categories or "general"

    url = f"{searxng_url.rstrip('/')}/search"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        logger.warning("SearXNG request failed: %s", exc)
        raise SearchError(f"Search backend unreachable: {exc}") from exc

    is_images = search_type == "images"
    results: list[dict] = []
    seen: set[str] = set()
    for item in data.get("results", []):
        u = item.get("url") or ""
        img = item.get("img_src") or item.get("thumbnail_src") or ""
        # Dedup by image for image search (same page can yield many images),
        # else by page URL.
        key = img if is_images else u
        if not key or key in seen:
            continue
        if is_images and not img:
            continue
        if not is_images and not u:
            continue
        seen.add(key)
        row = {
            "title": (item.get("title") or u or "")[:300],
            "url": u,
            "content": (item.get("content") or "")[:600],
            "engine": item.get("engine") or "",
        }
        if is_images:
            row["img_src"] = img
            row["thumbnail"] = item.get("thumbnail_src") or item.get("thumbnail") or img
            row["resolution"] = item.get("resolution") or ""
            row["source"] = item.get("source") or ""
        results.append(row)
        if len(results) >= count:
            break
    return results


def _build_answer_prompt(query: str, results: list[dict], language: str) -> str:
    """Build the LLM prompt that turns search hits into a cited answer."""
    blocks = []
    for i, r in enumerate(results, start=1):
        blocks.append(
            f"[{i}] {r['title']}\nURL: {r['url']}\n{r['content']}"
        )
    sources = "\n\n".join(blocks)

    if language == "de":
        return (
            "Du beantwortest die Frage des Nutzers ausschliesslich auf Basis der "
            "untenstehenden Web-Suchergebnisse. Schreibe eine praezise, sachliche "
            "Antwort auf Deutsch.\n\n"
            "REGELN\n"
            "- Nutze NUR Informationen aus den Quellen. Erfinde nichts.\n"
            "- Zitiere die Quellen inline mit [Nummer], z. B. [1], [3].\n"
            "- Wenn die Quellen die Frage nicht beantworten, sage das offen.\n"
            "- Keine Vorrede, keine Wiederholung der Frage, kein Marketing.\n"
            "- 3-8 Saetze, danach ggf. eine kurze Stichpunktliste.\n\n"
            f"FRAGE: {query}\n\n"
            f"WEB-SUCHERGEBNISSE:\n{sources}\n\n"
            "Antwort (mit [n]-Zitaten):"
        )
    return (
        "You answer the user's question strictly from the web search results "
        "below. Write a concise, factual answer in English.\n\n"
        "RULES\n"
        "- Use ONLY information from the sources. Do not invent anything.\n"
        "- Cite sources inline with [number], e.g. [1], [3].\n"
        "- If the sources do not answer the question, say so plainly.\n"
        "- No preamble, no restating the question, no marketing.\n"
        "- 3-8 sentences, optionally followed by a short bullet list.\n\n"
        f"QUESTION: {query}\n\n"
        f"WEB SEARCH RESULTS:\n{sources}\n\n"
        "Answer (with [n] citations):"
    )


async def answer_with_sources(
    query: str,
    results: list[dict],
    *,
    language: str = "en",
    backend: str = "openai",
    openai_base_url: str = "",
    openai_api_key: str = "",
    openai_model: str = "",
    ollama_base_url: str = "",
    ollama_model: str = "",
    temperature: float | None = None,
    max_sources: int = 6,
) -> str:
    """Synthesise a cited answer from the top search results via the LLM."""
    top = results[:max_sources]
    if not top:
        return ""
    lang = "de" if language == "de" else "en"
    prompt = _build_answer_prompt(query, top, lang)
    temp = 0.2 if temperature is None else temperature
    if backend == "openai":
        return await _call_openai_compatible(
            prompt, openai_base_url, openai_api_key, openai_model,
            response_format="text", temperature=temp,
        )
    return await _call_ollama(prompt, ollama_base_url, ollama_model, temp)
