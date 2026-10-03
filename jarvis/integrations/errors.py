"""
jarvis/integrations/errors.py
─────────────────────────────
v0.29 Part 15 — normalized provider errors.

Every provider adapter maps its native failures (timeouts, HTTP statuses,
auth rejections) into ONE of the categories below before anything reaches
the model or the ledger. Categories are honest: an ambiguous network
failure during a WRITE becomes AMBIGUOUS_OUTCOME (the side effect may or
may not have happened) and the ledger records UNKNOWN — never a blind
retry. Only deterministic, idempotent READS may be auto-retried (bounded),
and only for RATE_LIMITED / TRANSIENT_OUTAGE.

No error message ever carries credentials, tokens, or full raw provider
bodies — messages are bounded and pass through sanitization at the tool
boundary.
"""

from __future__ import annotations

from enum import Enum


class ProviderErrorCategory(str, Enum):
    TIMEOUT = "TIMEOUT"
    RATE_LIMITED = "RATE_LIMITED"
    AUTH_EXPIRED = "AUTH_EXPIRED"          # token expired / invalid
    AUTHORIZATION_DENIED = "AUTHORIZATION_DENIED"  # valid auth, insufficient scope/permission
    PROVIDER_OUTAGE = "PROVIDER_OUTAGE"    # 5xx / clearly down
    VALIDATION_ERROR = "VALIDATION_ERROR"  # provider rejected the payload
    DUPLICATE_REQUEST = "DUPLICATE_REQUEST"  # provider says already exists
    NOT_FOUND = "NOT_FOUND"
    AMBIGUOUS_OUTCOME = "AMBIGUOUS_OUTCOME"  # write possibly applied, result unknown
    PARTIAL_FAILURE = "PARTIAL_FAILURE"    # multi-item op, some succeeded
    PROVIDER_ERROR = "PROVIDER_ERROR"      # uncategorized; never blind-retried


# Categories where the side effect MIGHT have been applied despite the error.
# A write failing with one of these is recorded UNKNOWN, never FAILED — the
# distinction drives no-auto-retry semantics.
AMBIGUOUS_CATEGORIES: frozenset[ProviderErrorCategory] = frozenset(
    {ProviderErrorCategory.AMBIGUOUS_OUTCOME, ProviderErrorCategory.TIMEOUT}
)

# Categories for which a deterministic READ may be retried automatically
# (bounded by the caller). WRITES are never in this set.
READ_RETRYABLE_CATEGORIES: frozenset[ProviderErrorCategory] = frozenset(
    {ProviderErrorCategory.RATE_LIMITED, ProviderErrorCategory.PROVIDER_OUTAGE}
)

# HTTP status → category for adapters that speak HTTP.
HTTP_STATUS_MAP: dict[int, ProviderErrorCategory] = {
    400: ProviderErrorCategory.VALIDATION_ERROR,
    401: ProviderErrorCategory.AUTH_EXPIRED,
    403: ProviderErrorCategory.AUTHORIZATION_DENIED,
    404: ProviderErrorCategory.NOT_FOUND,
    408: ProviderErrorCategory.TIMEOUT,
    409: ProviderErrorCategory.DUPLICATE_REQUEST,
    429: ProviderErrorCategory.RATE_LIMITED,
    500: ProviderErrorCategory.PROVIDER_OUTAGE,
    502: ProviderErrorCategory.PROVIDER_OUTAGE,
    503: ProviderErrorCategory.PROVIDER_OUTAGE,
    504: ProviderErrorCategory.TIMEOUT,
}


class ProviderError(Exception):
    """
    A normalized provider failure. Safe, bounded message; the raw exception
    (which may embed URLs/headers) never reaches the model.
    """

    def __init__(self, category: ProviderErrorCategory, message: str) -> None:
        super().__init__(message)
        self.category = category
        self.message = message

    def to_tool_error(self) -> str:
        """Single-line, bounded, SANITIZED, category-prefixed tool result.

        Sanitization happens here (Part 15: never expose raw credentials or
        sensitive provider responses) so every caller is safe by default.
        """
        from jarvis.integrations.sanitize import sanitize

        return f"ERROR: PROVIDER_{self.category.value}: {sanitize(self.message)[:300]}"


def provider_error_from_exception(exc: Exception, operation: str) -> ProviderError:
    """
    Map an unexpected adapter exception to a normalized ProviderError.
    Deliberately conservative: anything unrecognized is PROVIDER_ERROR with
    only the exception TYPE in the message (str(exc) can embed secrets).
    """
    from socket import timeout as _socket_timeout

    if isinstance(exc, ProviderError):
        return exc
    if isinstance(exc, TimeoutError) or isinstance(exc, _socket_timeout):
        return ProviderError(
            ProviderErrorCategory.TIMEOUT,
            f"provider did not respond in time during {operation}",
        )
    return ProviderError(
        ProviderErrorCategory.PROVIDER_ERROR,
        f"{operation} failed ({type(exc).__name__})",
    )
