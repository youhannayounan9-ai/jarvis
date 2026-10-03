"""
jarvis/browser/redaction.py
───────────────────────────
v0.28 Part 19 — credential/secret redaction for browser outputs.

Contract:
  - NOTHING that could be a credential reaches the model: cookies, session
    tokens, passwords, API keys, authorization headers. The browser tools
    never read cookies/storage in the first place; this filter is the
    DEFENSE-IN-DEPTH layer for free-text page content that happens to
    embed secrets (e.g. a page echoing a token in visible text).
  - ``redact(text)`` replaces matched secret-like substrings with
    ``[REDACTED]`` BEFORE any text is returned to the model, written to
    logs, or recorded as telemetry/evidence.
  - Deterministic, bounded cost (compiled regexes, single pass each).
  - Deliberately conservative on FALSE POSITIVES: long random-looking
    strings adjacent to secret-KEY words are redacted; ordinary prose and
    normal URLs are left alone.

The automation profile holds no credentials (Part 7), so the primary
exposure is page-embedded text — which this filter covers.
"""

from __future__ import annotations

import re

# URL query parameters that carry secrets (e.g. ?token=...&api_key=...).
_SECRET_QUERY_KEYS = (
    "token", "access_token", "refresh_token", "auth", "authorization",
    "api_key", "apikey", "key", "secret", "password", "passwd", "pwd",
    "session", "sessionid", "sid", "sig", "signature", "credential",
)
_QUERY_SECRET_RE = re.compile(
    r"(?i)([?&](?:" + "|".join(re.escape(k) for k in _SECRET_QUERY_KEYS) + r")=)"
    r"([^&#\s]{4,})"
)
# Authorization-style headers and bearer tokens in free text.
_BEARER_RE = re.compile(r"(?i)\b(bearer|basic|token)\s+([A-Za-z0-9._\-]{8,})")
# key: value pairs where the key names a secret (JSON-ish or YAML-ish).
_KV_SECRET_RE = re.compile(
    r"(?i)\b((?:api[_-]?key|secret|password|passwd|pwd|token|access[_-]?token|"
    r"client[_-]?secret|private[_-]?key|authorization)\w*)\s*[:=]\s*"
    r"([\"']?)([^\s\"']{6,})\2"
)
# Long base64/hex-ish blobs (≥24 chars of base64/hex alphabet) — likely
# tokens when they appear in page text; ordinary URLs survive because '/'
# and '=' pairs of URLs are excluded from the class... but query values
# were already handled above. Bounded length to avoid huge scans.
_BLOB_RE = re.compile(r"\b[A-Za-z0-9+/=_\-]{24,256}\b")

_REDACTED = "[REDACTED]"


def redact(text: str) -> str:
    """Return ``text`` with secret-like substrings replaced by [REDACTED]."""
    if not text:
        return text
    out = _QUERY_SECRET_RE.sub(r"\1" + _REDACTED, text)
    out = _BEARER_RE.sub(r"\1 " + _REDACTED, out)
    out = _KV_SECRET_RE.sub(r"\1=\2" + _REDACTED + r"\2", out)
    # Bare high-entropy-looking blobs: only redact when the LOCAL context
    # hints at a secret (within 40 chars) — plain long IDs in prose survive.
    def _blob_or_keep(m: re.Match[str]) -> str:
        start = max(0, m.start() - 40)
        context = out[start : m.end() + 10].lower()
        hints = ("token", "secret", "key", "password", "auth", "bearer", "credential")
        return _REDACTED if any(h in context for h in hints) else m.group(0)

    out = _BLOB_RE.sub(_blob_or_keep, out)
    return out


def redact_mapping(mapping: dict[str, object]) -> dict[str, object]:
    """Redact every string value in a shallow metadata dict (safe copy)."""
    return {
        k: redact(v) if isinstance(v, str) else v  # type: ignore[arg-type]
        for k, v in mapping.items()
    }
