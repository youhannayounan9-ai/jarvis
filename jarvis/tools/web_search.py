"""
jarvis/tools/web_search.py
───────────────────────────
Tool: web_search

Performs a DuckDuckGo text search and returns the top N results as
structured, readable excerpts the model can synthesize from.

Why DuckDuckGo?
  - Free with no API key required.
  - The `ddgs` library (successor to duckduckgo-search) provides a clean Python interface.
  - Results include title, URL, and snippet — enough for the LLM to
    synthesize a useful answer.

Note on rate limiting:
  DuckDuckGo does not publish rate limits, but aggressive usage can result
  in temporary blocks. In v0.1 this is a non-issue (single user, low volume).
"""

import re
from ddgs import DDGS
from cachetools import TTLCache, cached

from jarvis.tools.base import BaseTool
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

_DEFAULT_MAX_RESULTS = 5
_MAX_SNIPPET_CHARS = 320
_search_cache = TTLCache(maxsize=100, ttl=3600)


class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "Search the live web via DuckDuckGo for current or specific factual "
        "information. Returns numbered result excerpts (title, URL, snippet). "
        "ONLY use when: (1) the user explicitly asks you to search the web, or "
        "(2) the answer needs information that may have changed after your "
        "training cutoff (news, prices, scores, recent events). "
        "After receiving results: synthesize a clear answer from the snippets — "
        "extract concrete facts, numbers, names, and dates. Do NOT dump a list "
        "of links. Prefer wikipedia_summary for encyclopedia-style topic overviews."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Focused search query. Include key entities, dates, or a "
                    "location when the answer is place-dependent (e.g. weather)."
                ),
            },
            "max_results": {
                "type": "integer",
                "description": (
                    f"Number of results to return "
                    f"(default: {_DEFAULT_MAX_RESULTS}, max: 10)."
                ),
            },
        },
        "required": ["query"],
    }

    def run(self, query: str, max_results: int = _DEFAULT_MAX_RESULTS, **kwargs) -> str:
        log.info("web_search", query=query, max_results=max_results)

        try:
            max_results = int(max_results)
        except (ValueError, TypeError):
            log.warning(
                "web_search_invalid_max_results",
                passed_value=max_results,
                default=_DEFAULT_MAX_RESULTS,
            )
            max_results = _DEFAULT_MAX_RESULTS

        max_results = min(max(1, max_results), 10)
        query = (query or "").strip()
        if not query:
            return "ERROR: Search query must not be empty."

        try:
            raw = self._do_search(query, max_results)
        except Exception as e:
            log.error("web_search_failed", query=query, error=str(e))
            return f"ERROR: Web search failed: {e}"

        results = _normalize_results(raw, limit=max_results)
        if not results:
            return (
                f"No useful results found for query: '{query}'. "
                "Try a more specific query, or use wikipedia_summary for "
                "well-known topics."
            )

        return _format_results(query, results)

    @staticmethod
    @cached(cache=_search_cache)
    def _do_search(query: str, max_results: int) -> list[dict]:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=min(max_results + 3, 12)))


def _normalize_results(raw: list[dict], limit: int) -> list[dict[str, str]]:
    """Deduplicate, clean, and keep the strongest snippets."""
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    cleaned: list[dict[str, str]] = []

    for item in raw:
        title = _clean_text(item.get("title") or "")
        url = (item.get("href") or item.get("link") or "").strip()
        snippet = _clean_text(item.get("body") or item.get("snippet") or "")

        if not title and not snippet:
            continue

        url_key = url.rstrip("/").lower()
        title_key = title.lower()
        if url_key and url_key in seen_urls:
            continue
        if title_key and title_key in seen_titles:
            continue

        if url_key:
            seen_urls.add(url_key)
        if title_key:
            seen_titles.add(title_key)

        if len(snippet) > _MAX_SNIPPET_CHARS:
            snippet = snippet[: _MAX_SNIPPET_CHARS - 1].rstrip() + "…"

        cleaned.append({
            "title": title or "Untitled",
            "url": url or "(no url)",
            "snippet": snippet or "(no snippet)",
        })
        if len(cleaned) >= limit:
            break

    return cleaned


def _clean_text(text: str) -> str:
    """Collapse whitespace and strip noisy control characters."""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _format_results(query: str, results: list[dict[str, str]]) -> str:
    """
    Present results as numbered prose excerpts.

    Readable blocks outperform raw JSON for smaller local models when
    they need to extract facts rather than echo links.
    """
    lines = [
        f'Web search results for: "{query}"',
        f"Returned {len(results)} result(s).",
        "",
        "Use the snippets below as evidence. Write a useful answer that "
        "extracts specific facts (names, numbers, dates, places). Mention "
        "sources briefly when helpful. Do NOT reply with only a list of URLs.",
        "",
    ]

    for i, r in enumerate(results, start=1):
        lines.append(f"[{i}] {r['title']}")
        lines.append(f"    URL: {r['url']}")
        lines.append(f"    Excerpt: {r['snippet']}")
        lines.append("")

    lines.append(
        "Answer guidance: Prefer concrete details from the excerpts. "
        "If sources conflict, note the disagreement. If results are thin, "
        "say what is missing instead of inventing facts."
    )
    return "\n".join(lines)
