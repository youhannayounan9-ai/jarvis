"""
jarvis/core/result_cache.py
───────────────────────────
v0.24 cross-turn result cache (Parts B–G, J–M).

WHAT IT IS
    A bounded SQLite-backed cache of READ-ONLY retrieval results, reused
    across TURNS (v0.23's dispatch ledger is same-turn only — Part J keeps
    the mechanisms separate: the ledger suppresses an identical call inside
    one task; the cache serves a still-valid observation to a LATER turn).

WHAT IS NEVER CACHED
    Side-effect tools (write_file), state-coupled tools (get_current_datetime,
    recall_facts, remember_fact), and anything whose purpose is to cause an
    effect. A tool is cacheable ONLY by explicit declaration — a CachePolicy
    on the tool class. Tools without a policy are never cached (fail-safe
    default).

SAFETY BOUNDARY (Part F/Q)
    The cache sits AFTER PermissionGuard and confirmation parking and after
    the v0.23 duplicate ledger, BEFORE the registry dispatch. A cache lookup:
      - never runs for a blocked/unpermitted call (those return earlier),
      - never runs for arguments that fail the tool's OWN Pydantic schema
        (validation failure → normal miss; the registry then produces the
        canonical validation error),
      - never substitutes for a side effect (only policy-declared tools),
      - never authorizes anything (a hit is untrusted EVIDENCE, framed as
        data like every tool observation — Part Q).
    Failed (ERROR) results are never stored, so retry-after-failure always
    re-runs the real tool.

FRESHNESS (Part D/E)
    Three strategies, per policy:
      ttl                  — clock expiration (web: short; wikipedia: long;
                             calculator: effectively permanent, bounded by
                             the entry cap).
      source_stat          — read_file/list_directory: the argument path is
                             re-STAT-ed at lookup; missing/changed size or
                             mtime ⇒ stale. The stat is a change SIGNAL only;
                             the real tool still performs its full path
                             validation when re-run.
      knowledge_generation — search_knowledge: stale when the knowledge
                             DOCUMENT REGISTRY changes (count / chunk sum /
                             latest ingest). Registry rows are a handful of
                             SQLite rows — no Chroma scan per query (Part E).
    Source-modification invalidation (stat, generation) and time-based
    expiration (TTL) are different mechanisms and are kept distinct by
    design; a tool uses whichever (or none) matches its semantics.

PROVENANCE (Part D)
    A cache hit is returned with a short system-authored header (age + hint
    that it is cached). The model must not present a cached observation as
    live data; requests containing freshness words (latest/today/current…)
    bypass the cache entirely (Part L — small explicit policy, no NL
    classifier; see tool_policy.is_freshness_request).

KEYS (C1/M)
    cache_key = sha1(tool + "\\x1f" + normalized_args)[:16]
    The normalizer comes from the policy:
      generic               — v0.23 canonical JSON (sorted keys, collapsed
                              value whitespace) — search queries etc.
      verbatim              — sorted keys only; NO value rewriting (paths,
                              URLs, anything where spacing/punctuation is
                              meaning).
      calculator_expression — the calculator's existing AST parser proves
                              equivalence: 2+2 ≡ 2 + 2 ≡ (2+2). No new math
                              parser is invented (Part M); unparseable
                              expressions fall back to the generic form.

SCOPE (C3)
    "global"  — any session may reuse (public/deterministic content:
                calculator, web, wikipedia).
    "session" — only the originating session may reuse (user-private data:
                their files, their knowledge base).

STORAGE
    Lives in the SAME SQLite database via SessionStore (no new infrastructure).
    Bounded by RESULT_CACHE_MAX_ENTRIES: at store time expired entries are
    purged first; only if the cap is still exceeded are the OLDEST entries
    evicted. Maintenance cleanup is explicit and bounded (Part H).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

from jarvis.config import settings
from jarvis.tools.base import BaseTool, CachePolicy
from jarvis.tools.calculator import canonical_expression_form
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# Tools whose freshness strategy is source_stat stat the FIRST argument in
# this list. Only policy-declared source_stat tools are ever looked up here,
# and both such tools (read_file, list_directory) use "path".
_STAT_ARG_NAMES = ("path", "source", "url")

# Per-tool TTL overrides (Part C4). A policy with ttl_seconds=None resolves
# its TTL here at call time so configuration stays explicit and env-tunable.
_TTL_OVERRIDE = {
    "web_search": lambda: settings.RESULT_CACHE_WEB_TTL_SECONDS,
    "web_scrape": lambda: settings.RESULT_CACHE_WEB_TTL_SECONDS,
    "wikipedia_summary": lambda: settings.RESULT_CACHE_WIKI_TTL_SECONDS,
    "calculator": lambda: settings.RESULT_CACHE_CALC_TTL_SECONDS,
}


@dataclass
class CacheDecision:
    """Outcome of a cache lookup — never raises, never authorizes."""

    hit: bool
    result: str | None = None          # raw cached payload (without provenance)
    observation: str | None = None     # payload + provenance header (what the model sees)
    reason: str = "miss"               # miss | hit | stale | bypass | non_cacheable | disabled
    stale_reason: str | None = None    # ttl_expired | source_changed | kb_changed | missing_source
    age_seconds: float | None = None
    fingerprint: str | None = None
    scope: str | None = None


class ResultCache:
    """
    Cross-turn result cache. Holds NO state of its own beyond the store
    reference; all entries live in SQLite (crash-safe, multi-process).
    """

    def __init__(self, store: Any) -> None:
        self._store = store  # SessionStore

    # ── Policy resolution ─────────────────────────────────────────────────────

    @staticmethod
    def policy_for(tool: BaseTool | None) -> CachePolicy | None:
        """A tool is cacheable ONLY via an explicit class-level CachePolicy."""
        if tool is None:
            return None
        return getattr(tool, "cache_policy", None)

    @staticmethod
    def enabled() -> bool:
        return not bool(getattr(settings, "JARVIS_DISABLE_RESULT_CACHE", False))

    def _record_ops(
        self,
        *,
        hits: int = 0,
        misses: int = 0,
        stale: int = 0,
        bypass: int = 0,
        stores: int = 0,
        per_tool: dict[str, dict[str, int]] | None = None,
    ) -> None:
        """
        v0.25 (Part E): fold one operations delta into the store's daily
        metrics snapshot. Counts only — never tool arguments or results.
        Metrics recording must NEVER break a lookup (best-effort by design).
        """
        try:
            self._store.record_cache_metrics(
                hits=hits, misses=misses, stale=stale, bypass=bypass,
                stores=stores, per_tool=per_tool,
            )
        except Exception as e:  # noqa: BLE001 — telemetry never breaks serving
            log.warning("cache_metrics_record_failed", error=str(e))

    # ── Keys (C1/M) ───────────────────────────────────────────────────────────

    @staticmethod
    def normalized_args(policy: CachePolicy, tool_args_json: str) -> str:
        """Normalized argument form for the cache key — per-tool normalizer."""
        if policy.normalizer == "verbatim":
            try:
                raw = json.loads(tool_args_json) if tool_args_json and tool_args_json.strip() else {}
            except (json.JSONDecodeError, TypeError):
                return f"__unparsed__:{(tool_args_json or '').strip()}"
            return json.dumps(raw, sort_keys=True, separators=(",", ":"))
        if policy.normalizer == "calculator_expression":
            try:
                raw = json.loads(tool_args_json) if tool_args_json and tool_args_json.strip() else {}
            except (json.JSONDecodeError, TypeError):
                return f"__unparsed__:{(tool_args_json or '').strip()}"
            expr = raw.get("expression") if isinstance(raw, dict) else None
            if isinstance(expr, str):
                canonical = canonical_expression_form(expr)
                if canonical is not None:
                    # Only the expression participates in equivalence; any
                    # other (future) keys stay canonically sorted around it.
                    if set(raw) == {"expression"}:
                        return json.dumps({"expression": canonical}, separators=(",", ":"))
                    rest = {k: v for k, v in raw.items() if k != "expression"}
                    rest["expression"] = canonical
                    return json.dumps(rest, sort_keys=True, separators=(",", ":"))
            # Unparseable/unsafe expression → generic form (never crash).
            from jarvis.core.dispatch_guard import canonical_arguments

            return canonical_arguments("calculator", tool_args_json)
        # generic (default): reuse the v0.23 canonicalization.
        from jarvis.core.dispatch_guard import canonical_arguments

        return canonical_arguments("tool", tool_args_json)

    @classmethod
    def cache_key(cls, tool_name: str, policy: CachePolicy, tool_args_json: str) -> str:
        normalized = cls.normalized_args(policy, tool_args_json)
        digest = hashlib.sha1(f"{tool_name}\x1f{normalized}".encode("utf-8")).hexdigest()[:16]
        return digest

    # ── Freshness (D/E) ───────────────────────────────────────────────────────

    @staticmethod
    def _resolve_ttl(tool_name: str, policy: CachePolicy) -> float | None:
        if policy.ttl_seconds is not None:
            return float(policy.ttl_seconds)
        override = _TTL_OVERRIDE.get(tool_name)
        if override is not None:
            return float(override())
        return float(settings.RESULT_CACHE_DEFAULT_TTL_SECONDS)

    def _stat_signature(self, tool_args_json: str) -> str | None:
        """Cheap change signal for source_stat tools: size + mtime_ns."""
        import os

        try:
            raw = json.loads(tool_args_json) if tool_args_json and tool_args_json.strip() else {}
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(raw, dict):
            return None
        for name in _STAT_ARG_NAMES:
            path = raw.get(name)
            if isinstance(path, str) and path.strip():
                try:
                    st = os.stat(path)
                except OSError:
                    return "MISSING_SOURCE"
                return f"{st.st_size}:{st.st_mtime_ns}"
        return None

    def _kb_generation(self) -> str:
        """Knowledge-base generation: tiny registry aggregate, NOT a scan."""
        return self._store.knowledge_generation()

    # ── Lookup / store (F) ────────────────────────────────────────────────────

    def lookup(
        self,
        *,
        tool: BaseTool | None,
        tool_name: str,
        tool_args_json: str,
        session_id: str,
        bypass_reason: str | None = None,
    ) -> CacheDecision:
        """
        Look up a still-valid cached observation.

        Order of checks: kill switch → policy → bypass → key → row →
        freshness. NEVER runs permission checks itself (the caller has
        already passed PermissionGuard) and NEVER validates less than the
        registry would: arguments must parse and satisfy the tool's schema
        or the lookup is a plain miss.
        """
        if not self.enabled():
            return CacheDecision(hit=False, reason="disabled")
        policy = self.policy_for(tool)
        if policy is None or not policy.cacheable:
            return CacheDecision(hit=False, reason="non_cacheable")

        fingerprint = self.cache_key(tool_name, policy, tool_args_json)

        if bypass_reason:
            self._record_ops(bypass=1)
            log.info(
                "cache_bypass",
                tool=tool_name,
                fingerprint=fingerprint,
                reason=bypass_reason,
            )
            return CacheDecision(hit=False, reason="bypass", fingerprint=fingerprint, scope=policy.scope)

        # Validation boundary: identical to the registry's. A cache must not
        # serve results for arguments the tool itself would reject.
        try:
            raw_args: dict[str, Any] = (
                json.loads(tool_args_json) if tool_args_json and tool_args_json.strip() else {}
            )
            tool._args_model.model_validate(raw_args)
        except Exception:  # noqa: BLE001 — any validation failure = normal miss
            return CacheDecision(hit=False, reason="miss", fingerprint=fingerprint, scope=policy.scope)

        started = time.perf_counter()
        row = self._store.get_cached_result(
            cache_key=fingerprint,
            session_id=session_id,
        )
        lookup_ms = round((time.perf_counter() - started) * 1000, 2)

        if row is None:
            self._record_ops(misses=1, per_tool={tool_name: {"misses": 1}})
            log.info(
                "cache_miss", tool=tool_name, fingerprint=fingerprint,
                reason="first_use", lookup_ms=lookup_ms,
            )
            return CacheDecision(hit=False, reason="miss", fingerprint=fingerprint, scope=policy.scope)

        age = max(0.0, time.time() - _parse_ts(row["created_at"]))

        # Freshness strategy 1: TTL.
        if row["expires_at"] is not None:
            expires = _parse_ts(row["expires_at"])
            if time.time() >= expires:
                self._store.delete_cached_result(fingerprint)
                self._record_ops(stale=1)
                log.info(
                    "cache_stale", tool=tool_name, fingerprint=fingerprint,
                    reason="ttl_expired", age_s=round(age, 1), lookup_ms=lookup_ms,
                )
                return CacheDecision(
                    hit=False, reason="stale", stale_reason="ttl_expired",
                    fingerprint=fingerprint, scope=policy.scope, age_seconds=age,
                )

        # Freshness strategy 2: source stat.
        if policy.freshness == "source_stat":
            current = self._stat_signature(tool_args_json)
            if current == "MISSING_SOURCE":
                self._store.delete_cached_result(fingerprint)
                self._record_ops(stale=1)
                log.info(
                    "cache_stale", tool=tool_name, fingerprint=fingerprint,
                    reason="missing_source", lookup_ms=lookup_ms,
                )
                return CacheDecision(
                    hit=False, reason="stale", stale_reason="missing_source",
                    fingerprint=fingerprint, scope=policy.scope, age_seconds=age,
                )
            if current is not None and current != row["source_stat"]:
                self._store.delete_cached_result(fingerprint)
                self._record_ops(stale=1)
                log.info(
                    "cache_stale", tool=tool_name, fingerprint=fingerprint,
                    reason="source_changed", lookup_ms=lookup_ms,
                )
                return CacheDecision(
                    hit=False, reason="stale", stale_reason="source_changed",
                    fingerprint=fingerprint, scope=policy.scope, age_seconds=age,
                )

        # Freshness strategy 3: knowledge-base generation.
        if policy.freshness == "knowledge_generation":
            current_gen = self._kb_generation()
            if current_gen != (row["kb_generation"] or ""):
                self._store.delete_cached_result(fingerprint)
                self._record_ops(stale=1)
                log.info(
                    "cache_stale", tool=tool_name, fingerprint=fingerprint,
                    reason="kb_changed", lookup_ms=lookup_ms,
                )
                return CacheDecision(
                    hit=False, reason="stale", stale_reason="kb_changed",
                    fingerprint=fingerprint, scope=policy.scope, age_seconds=age,
                )

        # Valid hit: count it and return with provenance.
        self._store.record_cache_hit(fingerprint)
        self._record_ops(hits=1, per_tool={tool_name: {"hits": 1}})
        observation = _with_provenance(
            str(row["result"]), tool_name=tool_name, age_seconds=age, row=row
        )
        log.info(
            "cache_hit",
            tool=tool_name,
            fingerprint=fingerprint,
            age_s=round(age, 1),
            scope=row["scope"],
            lookup_ms=lookup_ms,
        )
        return CacheDecision(
            hit=True,
            result=str(row["result"]),
            observation=observation,
            reason="hit",
            age_seconds=age,
            fingerprint=fingerprint,
            scope=row["scope"],
        )

    def store_result(
        self,
        *,
        tool: BaseTool | None,
        tool_name: str,
        tool_args_json: str,
        session_id: str,
        result: str,
    ) -> bool:
        """
        Store a SUCCESSFUL read-only result. Errors are never cached
        (retry-after-failure must always reach the real tool).
        """
        if not self.enabled():
            return False
        policy = self.policy_for(tool)
        if policy is None or not policy.cacheable:
            return False

        fingerprint = self.cache_key(tool_name, policy, tool_args_json)
        ttl = self._resolve_ttl(tool_name, policy)
        expires_at: str | None
        if policy.freshness == "ttl" and ttl is not None:
            expires_at = _format_ts(time.time() + ttl)
        else:
            expires_at = None  # no time decay; freshness comes from source state

        stat_sig = (
            self._stat_signature(tool_args_json)
            if policy.freshness == "source_stat"
            else None
        )
        kb_gen = (
            self._kb_generation() if policy.freshness == "knowledge_generation" else None
        )

        self._store.put_cached_result(
            cache_key=fingerprint,
            tool_name=tool_name,
            args_json=self.normalized_args(policy, tool_args_json),
            result=result,
            scope=policy.scope,
            session_id=session_id if policy.scope == "session" else None,
            expires_at=expires_at,
            source_stat=stat_sig,
            kb_generation=kb_gen,
            max_entries=int(settings.RESULT_CACHE_MAX_ENTRIES),
        )
        self._record_ops(stores=1)
        log.info(
            "cache_store",
            tool=tool_name,
            fingerprint=fingerprint,
            ttl_s=ttl,
            scope=policy.scope,
            freshness=policy.freshness,
        )
        return True


# ── Timestamp + provenance helpers ────────────────────────────────────────────

def _parse_ts(value: Any) -> float:
    from jarvis.memory.session_store import _parse_ts as _parse_dt

    return _parse_dt(str(value)).timestamp()


def _format_ts(epoch: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _human_age(seconds: float) -> str:
    if seconds < 90:
        return f"{int(seconds)}s"
    if seconds < 5400:
        return f"{int(seconds // 60)}m"
    if seconds < 172800:
        return f"{int(seconds // 3600)}h"
    return f"{int(seconds // 86400)}d"


def _with_provenance(
    result: str,
    *,
    tool_name: str,
    age_seconds: float,
    row: dict[str, Any],
) -> str:
    expires_part = ""
    if row.get("expires_at"):
        remaining = _parse_ts(row["expires_at"]) - time.time()
        if remaining > 0:
            expires_part = f"; expires in {_human_age(remaining)}"
    header = (
        f"[cached result: retrieved {_human_age(age_seconds)} ago via {tool_name}"
        f"{expires_part} — not a live re-run; say 'latest' to force fresh retrieval]"
    )
    return f"{header}\n{result}"
