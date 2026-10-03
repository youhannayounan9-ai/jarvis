"""
jarvis/browser/url_policy.py
────────────────────────────
v0.28 Part 6 — deny-by-default URL safety policy.

Contract:
  - Scheme ALLOWLIST (http/https only by default). Everything else —
    ``javascript:``, ``data:``, ``file:``, ``vbscript:``, ``blob:``,
    ``chrome:``, ``about:``, ``ftp:``, custom app schemes — is DENIED.
    A missing scheme is never guessed: ``example.com`` is malformed input,
    not "obviously https".
  - Hostname validation: http(s) requires a syntactically valid hostname
    (or an IP literal). Spaces, backslashes, control characters, embedded
    credentials (``user:pass@host``), and ports outside 1..65535 deny.
  - Private-network policy (Part 6): loopback, link-local, private RFC1918,
    unique-local, and 0.0.0.0-style targets are denied unless the operator
    explicitly enables local access. A LOCAL TEST PAGE SERVES the live
    validation (Part 24) — the operator turn-on is the only way through,
    and hostname→IP resolution is performed so ``localtest.me``-style
    rebinding to 127.0.0.1 cannot sneak a private target through.
  - Length caps: URLs beyond ``max_url_chars`` are denied.
  - URLs that arrive FROM pages are untrusted data exactly like URLs typed
    by the model: the SAME validation applies (no provenance-based trust).

This module never performs I/O beyond hostname resolution when the private
check needs it — and resolution failures DENY (fail closed).
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlsplit

from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# Schemes JARVIS may ever navigate to. Deny-by-default: anything not listed
# here is refused, so future exotic schemes fail safe.
ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})

# Schemes explicitly named in the spec (kept for denial-reason clarity and
# deterministic tests; the allowlist already denies them).
DANGEROUS_SCHEMES: frozenset[str] = frozenset(
    {"javascript", "data", "file", "vbscript", "blob", "about", "chrome"}
)

_MAX_URL_CHARS_DEFAULT = 2048

# Hostname→IP resolver shape (deterministic tests may inject a stub; None
# means REAL socket resolution in production). Injecting a resolver can
# never WEAKEN the policy: unresolvable still denies (fail closed) and a
# resolver that reports a private/loopback address still denies.
Resolver = Callable[[str], list[str]]


@dataclass(frozen=True)
class URLPolicy:
    """
    Immutable configuration for URL validation.

    ``allow_local_network=False`` (default) denies loopback/private targets.
    Operators who run controlled LOCAL test pages (Part 24) may enable it
    for their deployment — the policy is deployment config, never something
    the model can influence.
    """

    allow_local_network: bool = False
    allowed_schemes: frozenset[str] = ALLOWED_SCHEMES
    max_url_chars: int = _MAX_URL_CHARS_DEFAULT
    resolver: Resolver | None = None


class URLPolicyError(ValueError):
    """Raised when a URL is denied. ``.reason`` is safe for the model/log."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"URL rejected by policy: {reason}")
        self.reason = reason


def _is_ip_denied(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _hostname_is_local(host: str, resolver: Resolver | None = None) -> bool:
    """
    True when ``host`` refers to the local machine or a private network —
    INCLUDING via DNS resolution (catches rebinding names to 127.0.0.1).
    Unresolvable hostnames FAIL CLOSED (True = treated as denied-category).
    Uses the injected ``resolver`` when provided (deterministic tests);
    otherwise real socket resolution.
    """
    try:
        if resolver is not None:
            addresses = [str(a) for a in resolver(host)]
        else:
            infos = socket.getaddrinfo(host, None)
            addresses = [info[4][0] for info in infos]
    except (socket.gaierror, OSError, UnicodeError):
        return True  # unresolvable → fail closed
    for addr in addresses:
        try:
            ip = ipaddress.ip_address(addr.split("%")[0])
        except ValueError:
            return True
        if _is_ip_denied(ip):
            return True
    return False


def validate_url(url: str, policy: URLPolicy | None = None) -> str:
    """
    Validate ``url`` against the policy. Returns the normalized URL string
    on success; raises :class:`URLPolicyError` on any violation.
    """
    policy = policy or URLPolicy()
    if not isinstance(url, str) or not url.strip():
        raise URLPolicyError("empty URL")
    if len(url) > policy.max_url_chars:
        raise URLPolicyError(f"URL exceeds {policy.max_url_chars} characters")

    try:
        parts = urlsplit(url.strip())
    except ValueError as e:
        raise URLPolicyError(f"malformed URL ({e.__class__.__name__})") from e

    scheme = (parts.scheme or "").lower()
    if scheme not in policy.allowed_schemes:
        if scheme in DANGEROUS_SCHEMES:
            raise URLPolicyError(f"scheme '{scheme}:' is denied")
        raise URLPolicyError(f"scheme '{scheme or '(none)'}' is not allowed")

    if not parts.hostname:
        raise URLPolicyError("missing hostname")

    host = parts.hostname
    # Embedded credentials are never needed for browsing and leak secrets.
    if parts.username or parts.password:
        raise URLPolicyError("embedded credentials are denied")
    # Whitespace/control characters anywhere in the netloc or path deny:
    # real URLs never need them and they enable parser-differential tricks.
    for label, segment in (("netloc", parts.netloc), ("path", parts.path)):
        if any(ch.isspace() or ord(ch) < 0x20 for ch in segment):
            raise URLPolicyError(f"invalid character in {label}")
    try:
        port = parts.port  # property raises ValueError for out-of-range
    except ValueError:
        raise URLPolicyError("invalid port") from None
    if port is not None and not (1 <= port <= 65535):
        raise URLPolicyError("invalid port")

    if not policy.allow_local_network:
        # IP literals are checkable without DNS: classify directly so a
        # deterministic resolver (tests) or a fresh DNS record can never
        # smuggle a private/loopback literal through as a "public name".
        try:
            literal_ip = ipaddress.ip_address(host)
        except ValueError:
            literal_ip = None
        if literal_ip is not None:
            if _is_ip_denied(literal_ip):
                raise URLPolicyError(
                    "local/private network targets are denied by policy"
                )
        elif _hostname_is_local(host, policy.resolver):
            raise URLPolicyError(
                "local/private network targets are denied by policy"
            )
    return url.strip()
