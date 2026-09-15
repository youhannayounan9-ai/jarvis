"""
jarvis/tools/wikipedia_summary.py
──────────────────────────────────
Tool: wikipedia_summary

Fetches a short encyclopedia summary from Wikipedia's free public API.
No API key and no extra dependencies — stdlib urllib only.
"""

import json
import re
import urllib.error
import urllib.parse
import urllib.request

from jarvis.tools.base import BaseTool
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

_USER_AGENT = "JARVIS-LocalAssistant/0.1 (personal; contact: local-user)"
_OPENSEARCH_URL = "https://en.wikipedia.org/w/api.php"
_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
_TIMEOUT_SECONDS = 10
_MAX_SUMMARY_CHARS = 1200


class WikipediaSummaryTool(BaseTool):
    name = "wikipedia_summary"
    description = (
        "Get a clean, short encyclopedia summary from Wikipedia for a person, "
        "place, concept, or topic. Prefer this over web_search for general "
        "background knowledge and definitions. Do NOT use for breaking news "
        "or rapidly changing live data."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Topic to look up (e.g. 'Alan Turing', 'photosynthesis', "
                    "'Ollama (software)')."
                ),
            },
        },
        "required": ["query"],
    }

    def run(self, query: str, **kwargs) -> str:
        query = (query or "").strip()
        if not query:
            return "ERROR: Wikipedia query must not be empty."

        log.info("wikipedia_summary", query=query)

        try:
            title = _resolve_title(query)
            if not title:
                return (
                    f"No Wikipedia article found for '{query}'. "
                    "Try a different spelling or a broader topic name."
                )
            summary = _fetch_summary(title)
        except urllib.error.HTTPError as e:
            log.error("wikipedia_http_error", query=query, status=e.code)
            if e.code == 404:
                return f"No Wikipedia article found for '{query}'."
            return f"ERROR: Wikipedia request failed (HTTP {e.code})."
        except Exception as e:
            log.error("wikipedia_failed", query=query, error=str(e))
            return f"ERROR: Wikipedia lookup failed: {e}"

        return summary


def _resolve_title(query: str) -> str | None:
    """Resolve a free-text query to the best matching Wikipedia page title."""
    params = urllib.parse.urlencode({
        "action": "opensearch",
        "search": query,
        "limit": 1,
        "namespace": 0,
        "format": "json",
    })
    data = _http_get_json(f"{_OPENSEARCH_URL}?{params}")
    # OpenSearch format: [query, [titles], [descriptions], [urls]]
    if not isinstance(data, list) or len(data) < 2 or not data[1]:
        return None
    return data[1][0]


def _fetch_summary(title: str) -> str:
    """Fetch the REST summary for an exact page title."""
    encoded = urllib.parse.quote(title.replace(" ", "_"), safe="")
    data = _http_get_json(_SUMMARY_URL.format(title=encoded))

    page_type = data.get("type", "")
    if page_type == "disambiguation":
        return (
            f"Wikipedia page '{data.get('title', title)}' is a disambiguation "
            "page (multiple topics share this name). Ask the user to clarify, "
            "or retry with a more specific query."
        )

    extract = (data.get("extract") or "").strip()
    if not extract:
        return f"ERROR: Wikipedia returned an empty summary for '{title}'."

    extract = re.sub(r"\s+", " ", extract)
    if len(extract) > _MAX_SUMMARY_CHARS:
        extract = extract[: _MAX_SUMMARY_CHARS - 1].rstrip() + "…"

    page_title = data.get("title") or title
    page_url = (
        (data.get("content_urls") or {}).get("desktop", {}).get("page")
        or f"https://en.wikipedia.org/wiki/{urllib.parse.quote(page_title.replace(' ', '_'))}"
    )
    description = (data.get("description") or "").strip()

    lines = [
        f"Wikipedia summary: {page_title}",
    ]
    if description:
        lines.append(f"About: {description}")
    lines.extend([
        f"URL: {page_url}",
        "",
        extract,
        "",
        "Present this as a concise answer in your own words. "
        "Do not invent details beyond the summary.",
    ])
    return "\n".join(lines)


def _http_get_json(url: str) -> dict | list:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": "application/json",
        },
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw)
