"""
jarvis/core/grounding.py
────────────────────────
v0.26 Grounding Guard & Answer Integrity.

v0.25 gave synthesis an AUTHORITATIVE TOOL EVIDENCE block and an instruction
to transcribe tool-derived values exactly. Live evidence (qwen2.5:7b) proved
that an instruction is not a guarantee: the model occasionally recomputed
arithmetic (33071 for 893 × 47 against tool evidence 41971).

This module turns the instruction into a SYSTEM guarantee: after synthesis,
a deterministic post-check inspects the final answer against the turn's
trusted evidence. Only high-confidence, tool-aware checks fire; anything the
parser cannot understand passes untouched (conservative by design).

Design invariants (all test-pinned):
  - The guard is DETERMINISTIC: same evidence + same answer ⇒ same verdict.
    It never calls the model, never fabricates a value, never guesses.
  - Evidence is DATA, never instructions: evidence text is scanned as data;
    nothing inside it can steer the guard or the runtime.
  - A cached hit is provenance-labeled evidence (v0.24) trusted at the same
    level as a live observation — appearing later in history never makes
    model prose more authoritative than a tool result.
  - A FAILED observation never enters the trusted set, so a later failure
    cannot overwrite an earlier success (the v0.25 ledger guarantees this;
    it is re-asserted here at the type boundary).
  - At most ONE bounded correction round (enforced by the caller); failure
    degrades to a truthful, provenance-labeled fallback that preserves the
    authoritative value — never silent acceptance, never infinite retry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# ── Trusted evidence (Part 2) ─────────────────────────────────────────────────


@dataclass
class TrustedEvidence:
    """
    One verifiable item of trusted answer evidence (v0.26 Part 2).

    Built ONLY from successful tool observations of the current turn — never
    from model prose, never from conversational history. Cached results enter
    through the same gate (provenance-labeled cache observations already flow
    into the v0.25 ledger via `_dispatch_with_permissions_async`).
    """

    step_number: int
    tool: str
    status: str                      # "ok" enforced by check_grounding — failed observations never govern
    result: str                      # clamped raw observation text
    source: str = "live"             # live | cache (provenance of the VALUE)
    cached_age: str | None = None    # humanized cache age when source == "cache"
    authoritative: bool = True       # structured enough for deterministic checking

    def render(self) -> str:
        """Stable one-line rendering for prompts and telemetry."""
        src = (
            self.source
            if self.source == "live"
            else f"cache ({self.cached_age or 'age unknown'})"
        )
        return (
            f"[step {self.step_number} | {self.tool} | status: {self.status} | "
            f"source: {src}] {self.result}"
        )


# Precedence order (Part 2). Index = rank; lower wins. Only the first two
# ranks are ELIGIBLE for deterministic checking; model prose and conversation
# history are never trusted enough to contradict evidence.
PRECEDENCE = (
    "authoritative_tool_observation",   # 1. this turn's successful tool result
    "validated_cached_result",          # 2. provenance-labeled cache hit
    "model_intermediate_reasoning",     # 3. never trusted against evidence
    "conversational_history",           # 4. never trusted against evidence
)

# Bound for the trusted set (mirrors the synthesis ledger's 16-item cap).
_MAX_TRUSTED_ITEMS = 16


# v0.24 cache provenance header that prefixes a served cache hit
# (``[cached result: retrieved 3m ago via calculator — not a live re-run…]``).
# Its presence in an observation is the single source of truth for the
# item's ``source``/``cached_age`` provenance (v0.26 Part 2).
_CACHED_HEADER_RE = re.compile(
    r"^\[cached result:\s*retrieved\s+(?P<age>[^\]]+?)\s+ago\s+via\s+"
    r"(?P<tool>[\w_]+)[^\]]*\]\s*",
    re.IGNORECASE,
)


def build_trusted_evidence(
    evidence: list[dict[str, Any]] | None,
) -> list[TrustedEvidence]:
    """
    Convert the v0.25 synthesis evidence ledger into trusted-evidence items.

    The v0.25 ledger already guarantees: successful observations only, each
    labeled with step/tool/result. Items additionally carry v0.26 provenance:
    an explicit ``source`` field when the caller provides one, else detection
    from the v0.24 cache provenance header; unknown provenance is treated as
    live (the ledger is per-turn and dispatch-fresh by construction).
    """
    if not evidence:
        return []
    items: list[TrustedEvidence] = []
    for raw in evidence[-_MAX_TRUSTED_ITEMS:]:
        result_text = str(raw.get("result") or "")
        header = _CACHED_HEADER_RE.match(result_text)
        if raw.get("source"):
            source, cached_age = str(raw["source"]), raw.get("cached_age")
        elif header:
            source, cached_age = "cache", header.group("age").strip()
        else:
            source, cached_age = "live", None
        items.append(
            TrustedEvidence(
                step_number=int(raw.get("step_number") or 0),
                tool=str(raw.get("tool") or "tool"),
                status=str(raw.get("status") or "ok"),
                result=result_text,
                source=source,
                cached_age=(str(cached_age) if cached_age else None),
                authoritative=bool(raw.get("authoritative", True)),
            )
        )
    return items


# ── Tool-aware grounding policies (Part 9) ────────────────────────────────────


class GroundingPolicy:
    """
    Interface for tool-aware grounding policies (Part 9).

    A policy decides whether a trusted evidence item is checkable, and if so,
    whether a final answer contradicts it. Policies are DETERMINISTIC and
    conservative: when in doubt, return ``None`` (no contradiction) — a
    missed catch is acceptable; a false rejection is not. Policies never
    raise for expected parsing trouble; they degrade to ``None``.
    """

    name: str = "base"

    def applies(self, item: TrustedEvidence) -> bool:
        """Whether this policy can deterministically check this item."""
        raise NotImplementedError

    def evidence_values(self, item: TrustedEvidence) -> set[float]:
        """
        Canonical value(s) this observation vouches for — used to reconcile
        multi-step evidence of the SAME tool (a number the answer took from
        ANY of this turn's observations is consistent; Part 7). Default:
        no numeric value (non-numeric policies handle multi-item groups by
        latest-only enforcement).
        """
        return set()

    def evidence_display(self, item: TrustedEvidence) -> str | None:
        """Human rendering of the evidence value, when numeric."""
        return None

    def contradiction(
        self, item: TrustedEvidence, answer: str
    ) -> dict[str, Any] | None:
        """
        Return a contradiction descriptor when ``answer`` is a HIGH-CONFIDENCE
        contradiction of ``item``; ``None`` when the answer is consistent or
        the check is inconclusive. The descriptor is safe metadata only
        (expected value rendering, evidence type, source) — never raw args.
        """
        raise NotImplementedError


class CalculatorGroundingPolicy(GroundingPolicy):
    """
    Numeric verification for calculator-style evidence (Parts 3+4).

    Fires only when the observation IS the calculator's exact single-line
    output format (``Result: <number>`` — the tool's documented contract;
    FULL-match anchoring so an injected ``Result:`` line inside a longer
    document can never impersonate calculator evidence). The answer is a
    contradiction ONLY when ALL of these hold:

      1. no rendering canonicalizing to the evidence value appears anywhere;
      2. a DIFFERENT canonical number appears near RESULT CUE wording
         ("result/answer/total/equals/=/multiplication/..." within 48 chars
         before the number) — prose-adjacent numbers, step narrations,
         latencies (``120 ms``), years and bare identifiers never trigger;
      3. the differing number is itself unambiguously canonicalizable.

    Conservative by design: a missed catch is acceptable; a false rejection
    is not (Part 3/4).
    """

    name = "calculator_numeric"

    # FULL-match: the WHOLE observation must be the calculator's output line
    # (an optional leading ``[cached result: …]`` provenance header from the
    # v0.24 cache is allowed — cache hits are evidence at the same trust
    # level). A ``Result:`` line inside a longer document can never
    # impersonate calculator evidence (Part 15).
    _RESULT_RE = re.compile(
        r"(?:\[cached result:[^\]]*\]\s*)?Result:\s*"
        r"(-?[\d,.\s'\u00a0\u202f\u2009]+?)\s*"
    )

    def applies(self, item: TrustedEvidence) -> bool:
        return (
            item.authoritative
            and item.tool == "calculator"
            and self._RESULT_RE.fullmatch(item.result.strip()) is not None
        )

    def evidence_values(self, item: TrustedEvidence) -> set[float]:
        """The canonical value(s) this observation vouches for (Part 7)."""
        m = self._RESULT_RE.fullmatch(item.result.strip())
        if not m:
            return set()
        cn = canonical_number(m.group(1))
        return {cn.value} if cn is not None else set()

    def evidence_display(self, item: TrustedEvidence) -> str | None:
        """Human rendering of the evidence value (correction/fallback text)."""
        m = self._RESULT_RE.fullmatch(item.result.strip())
        if not m:
            return None
        cn = canonical_number(m.group(1))
        return cn.display if cn is not None else None

    def contradiction(
        self, item: TrustedEvidence, answer: str
    ) -> dict[str, Any] | None:
        try:
            m = self._RESULT_RE.fullmatch(item.result.strip())
            if not m:
                return None
            expected = canonical_number(m.group(1))
            if expected is None:
                return None  # unparseable value — never guess, never reject
            if numeric_appearances(answer, expected.value):
                # A consistent rendering exists — not a contradiction.
                return None
            wrong: list[str] = []
            for cand in canonical_numbers(answer):
                if cand.value == expected.value:
                    continue
                if self._is_excluded_context(answer, cand):
                    continue
                wrong.append(cand.raw)
                if len(wrong) >= 5:
                    break
            if not wrong:
                return None
            return {
                "policy": self.name,
                "evidence_type": "calculator_numeric",
                "expected": expected.display,
                "source": item.source,
                "cached": item.source == "cache",
                "contradicting_values": wrong,
            }
        except Exception as e:  # noqa: BLE001 - degrade, never raise
            log.debug("grounding_policy_degraded", policy=self.name, error=str(e))
            return None

    # Result-cue wording that may directly precede a stated RESULT. Narrative
    # verbs ("gave", "returned") are deliberately absent: step narration
    # like "Step 1 gave 893" refers to operands, not the final result.
    _CUE_RE = re.compile(
        r"\b(?:results?|answers?|totals?|equals?|equal\s+to|product|sum|difference|"
        r"quotient|multiplication|multiplied|multiplying|addition|adding|"
        r"subtraction|subtracting|division|dividing|computed?|calculated?|"
        r"yields?|gives?|times)\b|[=:]",
        re.IGNORECASE,
    )
    _CUE_WINDOW = 48          # chars before the number scanned for a cue
    _POST_CUE_WINDOW = 24     # chars after: "33,071 is the result."
    _POST_CUE_RE = re.compile(r"\b(?:result|answer|total)\b", re.IGNORECASE)
    _STEP_ADJACENT_RE = re.compile(r"\bstep\s*#?\s*\d+\b", re.IGNORECASE)
    _LATENCY_RAW_RE = re.compile(r"\d\s?(?:ms|us|µs)$", re.IGNORECASE)
    _YEAR_RAW_RE = re.compile(r"\d{4}")

    def _is_excluded_context(self, answer: str, cand: Any) -> bool:
        """False-positive exclusion rules (Part 4). All are deterministic."""
        start = cand.span[0] if cand.span and cand.span[1] > cand.span[0] else 0
        window = answer[max(0, start - self._CUE_WINDOW):start]
        # (a) step narration: "Step 1 gave 893 ... step 2 gave 47"
        if self._STEP_ADJACENT_RE.search(window):
            return True
        # (b) latency/duration unit: "... completed in 120 ms"
        if self._LATENCY_RAW_RE.search(cand.raw.strip()):
            return True
        # (c) bare year: "Around 2026 ..."
        if self._YEAR_RAW_RE.fullmatch(cand.raw.strip()) and 1900 <= cand.value <= 2100:
            return True
        # (d) the deciding gate: a RESULT CUE must appear near the number —
        # before it ("the result is X") or in a narrow window after it
        # ("X is the result"). Comparatives ("25% higher than X") and bare
        # copulas ("It is X.") stay OUTSIDE the gate: conservative misses
        # are acceptable, false rejections are not.
        if self._CUE_RE.search(window):
            return False
        after = answer[start:min(len(answer), start + len(cand.raw) + self._POST_CUE_WINDOW)]
        if self._POST_CUE_RE.search(after):
            return False
        return True


class DatetimeGroundingPolicy(GroundingPolicy):
    """
    Date verification for the datetime tool's documented output format
    (``Current datetime: <Weekday>, <Month> <DD>, <YYYY> at <HH:MM:SS>``).

    A contradiction fires only when the answer asserts a DIFFERENT full date
    (weekday+month+day+year) — partial or differently-phrased dates pass.
    """

    name = "datetime_stated_date"

    _EVIDENCE_RE = re.compile(
        r"Current datetime:\s*(?P<weekday>[A-Za-z]+),\s*"
        r"(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2}),\s*(?P<year>\d{4})"
    )

    def applies(self, item: TrustedEvidence) -> bool:
        return item.authoritative and item.tool == "get_current_datetime"

    def contradiction(
        self, item: TrustedEvidence, answer: str
    ) -> dict[str, Any] | None:
        try:
            m = self._EVIDENCE_RE.search(item.result)
            if not m:
                return None
            expected_date = f"{m.group('month')} {m.group('day')}, {m.group('year')}"
            weekday = m.group("weekday")
            # Full stated dates in the answer with a different day/year.
            for am in re.finditer(
                r"\b([A-Za-z]+),?\s+([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})\b", answer
            ):
                aw_wd, aw_mo, aw_day, aw_yr = (
                    am.group(1),
                    am.group(2),
                    am.group(3),
                    am.group(4),
                )
                if aw_mo.lower() == m.group("month").lower() and aw_day == m.group(
                    "day"
                ) and aw_yr == m.group("year"):
                    continue  # consistent full date
                if _is_probable_other_date(aw_mo, aw_day, aw_yr):
                    return {
                        "policy": self.name,
                        "evidence_type": "datetime_stated_date",
                        "expected": f"{weekday}, {expected_date}",
                        "source": item.source,
                        "cached": item.source == "cache",
                        "contradicting_values": [f"{aw_wd} {aw_mo} {aw_day}, {aw_yr}"],
                    }
            return None
        except Exception as e:  # noqa: BLE001
            log.debug("grounding_policy_degraded", policy=self.name, error=str(e))
            return None


class FileResultGroundingPolicy(GroundingPolicy):
    """
    Exact-name verification for file/directory listings.

    Fires only for the documented listing formats (``Contents of 'path':`` /
    ``Contents of directory 'path':``). A contradiction fires when the answer
    claims a file is present/absent in a way that directly contradicts the
    listing (claims a filename the listing does not contain, or denies one it
    does). Conservative: only exact-basename claims are checked; partial
    mentions and paraphrases pass.
    """

    name = "file_listing_exact_names"

    _LISTING_RE = re.compile(
        r"Contents of (?:directory )?'(?P<path>[^']+)':", re.IGNORECASE
    )
    # "X is in the directory", "X is present", "the file X exists" …
    _CLAIM_PRESENT = re.compile(
        r"(?:^|\W)(?P<name>[A-Za-z0-9._\- ]{1,64}?)\s+(?:is\s+)?"
        r"(?:in\s+the\s+(?:directory|folder)|present|exists)\b"
        r"|(?P<dirclaim>directory|folder)\s+contains\s+"
        r"(?P<file>[A-Za-z0-9._\-]+\.[A-Za-z0-9]{1,5})\b",
        re.IGNORECASE,
    )
    _CLAIM_ABSENT = re.compile(
        r"(?:^|\W)(?P<name>[A-Za-z0-9._\- ]{1,64}?)\s+"
        r"(?:is\s+not\s+(?:in\s+the\s+(?:directory|folder)|present)|"
        r"does\s+not\s+exist)\b",
        re.IGNORECASE,
    )

    def applies(self, item: TrustedEvidence) -> bool:
        return item.authoritative and item.tool in (
            "read_file",
            "list_directory",
        )

    def _listed_names(self, item: TrustedEvidence) -> set[str] | None:
        m = self._LISTING_RE.search(item.result)
        if not m:
            return None
        body = item.result[m.end():]
        names: set[str] = set()
        for line in body.splitlines():
            name = line.strip().lstrip("📁📄").strip()
            if name:
                names.add(name.rstrip("/"))
        return names

    def contradiction(
        self, item: TrustedEvidence, answer: str
    ) -> dict[str, Any] | None:
        try:
            names = self._listed_names(item)
            if names is None:
                return None
            lowered = {n.lower(): n for n in names}
            for pattern, claimed_present in (
                (self._CLAIM_PRESENT, True),
                (self._CLAIM_ABSENT, False),
            ):
                for cm in pattern.finditer(answer):
                    raw_name = (
                        cm.group("file")
                        if cm.group("file")
                        else cm.group("name")
                    ).strip().strip("'\"")
                    key = raw_name.lower()
                    # Only check tokens that carry an extension (real file
                    # claims); bare words are too ambiguous.
                    if "." not in key:
                        continue
                    in_listing = key in lowered
                    if claimed_present and not in_listing:
                        return {
                            "policy": self.name,
                            "evidence_type": "file_listing_exact_names",
                            "expected": f"listing of {m.group('path') if (m := self._LISTING_RE.search(item.result)) else '?'}",
                            "source": item.source,
                            "cached": item.source == "cache",
                            "contradicting_values": [
                                f"claims present: {raw_name!r} (not in listing)"
                            ],
                        }
                    if not claimed_present and in_listing:
                        return {
                            "policy": self.name,
                            "evidence_type": "file_listing_exact_names",
                            "expected": f"listing contains {lowered[key]!r}",
                            "source": item.source,
                            "cached": item.source == "cache",
                            "contradicting_values": [
                                f"claims absent: {raw_name!r} (listed)"
                            ],
                        }
            return None
        except Exception as e:  # noqa: BLE001
            log.debug("grounding_policy_degraded", policy=self.name, error=str(e))
            return None


class StructuredFieldGroundingPolicy(GroundingPolicy):
    """
    Explicit 'field: value' verification for any tool whose observation
    exposes a labeled scalar (e.g. ``count: 12``, ``temperature: 21.5``).

    Fires only when BOTH the evidence and the answer state the SAME field
    label with different canonical numbers — same-label pairing keeps
    unrelated numbers out of the verdict.
    """

    name = "structured_field"

    _FIELD_RE = re.compile(
        r"^\s*(?P<label>[A-Za-z][A-Za-z0-9 _-]{1,30}?)\s*[:=]\s*"
        r"(?P<value>-?\d[\d,.'\u00a0\u202f ]*)",
        re.MULTILINE,
    )

    def applies(self, item: TrustedEvidence) -> bool:
        return item.authoritative and bool(
            self._FIELD_RE.search(item.result)
        )

    def contradiction(
        self, item: TrustedEvidence, answer: str
    ) -> dict[str, Any] | None:
        try:
            fields: dict[str, float] = {}
            for fm in self._FIELD_RE.finditer(item.result):
                value = canonical_number(fm.group("value"))
                if value is not None:
                    fields[fm.group("label").strip().lower()] = value.value
            if not fields:
                return None
            for am in self._FIELD_RE.finditer(answer):
                label = am.group("label").strip().lower()
                if label not in fields:
                    continue
                stated = canonical_number(am.group("value"))
                if stated is None:
                    continue
                if stated.value != fields[label]:
                    return {
                        "policy": self.name,
                        "evidence_type": "structured_field",
                        "expected": f"{label}: {_fmt_number(fields[label])}",
                        "source": item.source,
                        "cached": item.source == "cache",
                        "contradicting_values": [f"{label}: {stated.raw}"],
                    }
            return None
        except Exception as e:  # noqa: BLE001
            log.debug("grounding_policy_degraded", policy=self.name, error=str(e))
            return None


def _fmt_number(value: float) -> str:
    """Stable display rendering for telemetry/fallback text."""
    if float(value).is_integer() and abs(value) < 1e15:
        return f"{int(value):,}"
    return repr(value)


def _is_probable_other_date(month: str, day: str, year: str) -> bool:
    """True when (month, day, year) parses as a real calendar date."""
    try:
        datetime.strptime(f"{month} {day}, {year}", "%B %d, %Y")
        return True
    except ValueError:
        try:
            datetime.strptime(f"{month} {day}, {year}", "%b %d, %Y")
            return True
        except ValueError:
            return False


# ── v0.30 Part 20: labeled-field provider schedule verification ──────────────

_WEEKDAY_NAMES = (
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
)

# Small, EXPLICIT timezone vocabulary. An unknown city is never guessed — a
# missed catch is acceptable, a false rejection is not (v0.26 contract).
_ZONE_ALIASES: dict[str, str] = {
    "utc": "UTC", "gmt": "UTC", "cairo": "Africa/Cairo",
    "london": "Europe/London", "paris": "Europe/Paris",
    "berlin": "Europe/Berlin", "new york": "America/New_York",
    "newyork": "America/New_York", "los angeles": "America/Los_Angeles",
    "tokyo": "Asia/Tokyo", "dubai": "Asia/Dubai", "zurich": "Europe/Zurich",
}

_SCHEDULE_NOUNS = ("event", "meeting", "appointment", "calendar", "scheduled", "schedule")
_STATE_CHANGE_CLAIM_RE = re.compile(
    r"\b(moved|rescheduled|updated|changed|created|added|booked|deleted|cancelled|canceled|completed)\b",
    re.IGNORECASE,
)
# Observation sources that may be a READ pre-state of a same-turn write. A
# state-changing answer is exempt from schedule enforcement there (the write
# tool's own observation carries the authoritative post-state); a purely
# descriptive answer is always enforced.
_READ_PRE_STATE_TOOLS = frozenset({"calendar_list_events", "task_list"})

_CLOCK_RE = re.compile(r"\b(\d{1,2}):(\d{2})(?!\d|:)\s*([ap]\.?m\.?)?", re.IGNORECASE)
_HOUR_AMPM_RE = re.compile(r"\b(\d{1,2})\s*([ap])\.?m\.?", re.IGNORECASE)


def _canon_clock(hour: int, minute: int, ampm: str | None) -> int | None:
    """Minute-of-day for a clock time, or None when it cannot be one."""
    if minute > 59:
        return None
    if ampm:
        if not 1 <= hour <= 12:
            return None
        h = hour % 12
        if ampm[0].lower() == "p":
            h += 12
        return h * 60 + minute
    if 0 <= hour <= 23:
        return hour * 60 + minute
    return None


def _answer_clocks(answer: str) -> list[tuple[int, int, str]]:
    """Parsed (minute_of_day, position, raw) clock times stated in an answer."""
    out: list[tuple[int, int, str]] = []
    claimed: list[tuple[int, int]] = []
    for m in _CLOCK_RE.finditer(answer):
        value = _canon_clock(int(m.group(1)), int(m.group(2)), m.group(3))
        if value is None:
            continue
        out.append((value, m.start(), m.group(0).strip()))
        claimed.append(m.span())
    for m in _HOUR_AMPM_RE.finditer(answer):
        if any(s <= m.start() < e for s, e in claimed):
            continue
        value = _canon_clock(int(m.group(1)), 0, m.group(2) + "m")
        if value is not None:
            out.append((value, m.start(), m.group(0).strip()))
    return out


def _near_schedule_noun(answer: str, position: int, window: int = 60) -> bool:
    lo = max(0, position - window)
    hi = min(len(answer), position + window)
    return any(noun in answer[lo:hi].lower() for noun in _SCHEDULE_NOUNS)


def _zone_offset_minutes(zone_name: str, reference):
    """UTC offset of an IANA zone at a reference date; None when unknown."""
    from datetime import datetime as _dt, timedelta as _td
    from zoneinfo import ZoneInfo

    try:
        tz = ZoneInfo(str(zone_name))
    except Exception:  # noqa: BLE001 — unknown zone name → no verdict
        return None
    at = reference or _dt.now(tz=tz)
    try:
        off: _td = at.astimezone(tz).utcoffset() or _td(0)
    except Exception:  # noqa: BLE001
        return None
    return int(off.total_seconds() // 60)


def _answer_zones(answer: str) -> list[tuple[str, int | None]]:
    """Zone references in an answer as (label, explicit_offset_or_None)."""
    found: dict[str, int | None] = {}
    for m in re.finditer(r"\b([A-Za-z]+/[A-Za-z_\-0-9]+)\b", answer):
        found[m.group(1)] = None
    for m in re.finditer(r"\b(UTC|GMT)\s*([+-]\d{1,2})(?::?(\d{2}))?\b", answer, re.IGNORECASE):
        hours = abs(int(m.group(2)))
        minutes = int(m.group(3) or 0)
        sign = 1 if m.group(2).strip().startswith("+") else -1
        found[f"UTC{m.group(2)}"] = sign * (hours * 60 + minutes)
    for m in re.finditer(r"\b([A-Za-z]+(?:\s[A-Za-z]+)?)\s+time\b", answer, re.IGNORECASE):
        city = (m.group(1) or "").strip().lower()
        if city in _ZONE_ALIASES:
            found[_ZONE_ALIASES[city]] = None
            continue
        # "6:30 PM Tokyo time" — the marker before the city must not defeat
        # the match; fall back to the final word of the captured phrase.
        tail = city.split()[-1] if city else ""
        if tail in _ZONE_ALIASES:
            found[_ZONE_ALIASES[tail]] = None
    has_offset_form = any(k.startswith(("UTC+", "UTC-", "GMT+", "GMT-")) for k in found)
    if not has_offset_form and (
        re.search(r"\bUTC\b", answer, re.IGNORECASE)
        or re.search(r"\bGMT\b", answer, re.IGNORECASE)
    ):
        found.setdefault("UTC", 0)
    return list(found.items())


class ProviderScheduleGroundingPolicy(GroundingPolicy):
    """
    v0.30 Part 20: deterministic verification of LABELED provider schedule
    fields (the integration render emits ``date:`` / ``time:`` /
    ``timezone:`` / ``status:`` lines derived from real instants).

    Fires ONLY on observations that carry those exact labeled lines, and only
    on HIGH-CONFIDENCE claims near schedule wording:

      - a stated clock time that canonicalizes to a DIFFERENT time-of-day
        than every value in the evidence (12h/24h equivalence handled;
        3:30 PM == 15:30; a matching rendering anywhere passes — match-any);
      - a named weekday that differs from every date in the evidence;
      - a named/offset timezone whose UTC OFFSET differs from every evidence
        zone (UTC+2 == Africa/Cairo; unresolvable zones never verdict);
      - a task-status claim contradicting labeled status: ``done`` vs a
        claimed pending/open state, or (id-referenced only) ``open`` vs a
        claimed completed state.

    Conservative by design: durations, counts, years, IDs, and numbers
    without schedule wording never trigger; a read-only observation in a
    state-changing answer is exempt (the write's own observation carries the
    authoritative post-state); anything unparseable degrades to no verdict.
    """

    name = "provider_schedule"

    _DATE_RE = re.compile(r"^date:\s*(\d{4})-(\d{2})-(\d{2})\s*$", re.MULTILINE)
    _TIME_RE = re.compile(r"^time:\s*(\d{2}):(\d{2})\s*$", re.MULTILINE)
    _ZONE_RE = re.compile(r"^(?:timezone|tz):\s*([A-Za-z_+\-0-9/]+)\s*$", re.MULTILINE)
    _STATUS_RE = re.compile(r"^status:\s*(open|done)\s*$", re.MULTILINE)
    _KIND_RE = re.compile(r"^(event|task):\s*(\S+)\s*$", re.MULTILINE)

    def applies(self, item: TrustedEvidence) -> bool:
        if not item.authoritative:
            return False
        text = item.result
        if not text:
            return False
        has_zone = bool(self._ZONE_RE.search(text))
        has_date = bool(self._DATE_RE.search(text))
        has_time = bool(self._TIME_RE.search(text))
        has_status = bool(self._STATUS_RE.search(text)) and bool(self._KIND_RE.search(text))
        return (has_zone and (has_date or has_time)) or has_status

    # ── evidence side ────────────────────────────────────────────────────────

    def _evidence_facts(self, text: str) -> dict[str, Any]:
        from datetime import datetime as _dt, timezone as _tz

        times: set[int] = set()
        for hour, minute in self._TIME_RE.findall(text):
            value = _canon_clock(int(hour), int(minute), None)
            if value is not None:
                times.add(value)
        dates: list[Any] = []
        weekdays: set[str] = set()
        for y, mth, day in self._DATE_RE.findall(text):
            try:
                parsed = _dt(int(y), int(mth), int(day), tzinfo=_tz.utc)
            except ValueError:
                continue
            dates.append(parsed)
            weekdays.add(_WEEKDAY_NAMES[parsed.weekday()])
        zones = {z for z in self._ZONE_RE.findall(text) if z}
        status_by_id: dict[str, str] = {}
        current = None
        for line in text.splitlines():
            mk = self._KIND_RE.match(line.strip())
            if mk:
                current = mk.group(2)
                continue
            ms = self._STATUS_RE.match(line.strip())
            if ms and current:
                status_by_id[current] = ms.group(1)
        return {
            "times": times, "dates": dates, "weekdays": weekdays,
            "zones": zones, "status_by_id": status_by_id,
        }

    # ── detection ────────────────────────────────────────────────────────────

    def contradiction(self, item: TrustedEvidence, answer: str) -> dict[str, Any] | None:
        try:
            if not answer or not answer.strip():
                return None
            facts = self._evidence_facts(item.result)
            read_pre_state = item.tool in _READ_PRE_STATE_TOOLS
            state_change = bool(_STATE_CHANGE_CLAIM_RE.search(answer))

            def _finding(expected: str, stated: str) -> dict[str, Any]:
                return {
                    "policy": self.name,
                    "evidence_type": "provider_schedule",
                    "expected": expected,
                    "source": item.source,
                    "cached": item.source == "cache",
                    "contradicting_values": [stated],
                }

            if not (read_pre_state and state_change):
                # 1) clock time
                if facts["times"]:
                    clocks = _answer_clocks(answer)
                    if clocks and not any(v in facts["times"] for v, _pos, _raw in clocks):
                        for value, pos, raw in clocks:
                            if _near_schedule_noun(answer, pos):
                                expected = ", ".join(
                                    f"{v // 60:02d}:{v % 60:02d}" for v in sorted(facts["times"])
                                )
                                return _finding(f"time: {expected}", f"time {raw}")
                # 2) weekday
                if facts["weekdays"]:
                    named = {w for w in _WEEKDAY_NAMES if re.search(rf"\b{w}\b", answer, re.IGNORECASE)}
                    if named and not (named & facts["weekdays"]):
                        for weekday in sorted(named):
                            pos = answer.lower().find(weekday)
                            if pos >= 0 and _near_schedule_noun(answer, pos):
                                return _finding(
                                    "date: " + "/".join(sorted(facts["weekdays"])),
                                    f"weekday {weekday}",
                                )
                # 3) timezone (offset comparison: UTC+2 == Africa/Cairo)
                if facts["zones"]:
                    reference = facts["dates"][0] if facts["dates"] else None
                    evidence_offsets = {
                        off for off in (_zone_offset_minutes(z, reference) for z in facts["zones"])
                        if off is not None
                    }
                    if evidence_offsets:
                        for label, explicit in _answer_zones(answer):
                            offset = explicit if explicit is not None else _zone_offset_minutes(label, reference)
                            if offset is None:
                                continue
                            if offset not in evidence_offsets:
                                pos = answer.lower().find(label.split("/")[-1].lower())
                                if pos < 0 or _near_schedule_noun(answer, pos):
                                    return _finding(
                                        "timezone: " + "/".join(sorted(facts["zones"])),
                                        f"timezone {label}",
                                    )
            # 4) task status (always checked — the claim IS the state assertion)
            for resource_id, status in facts["status_by_id"].items():
                if status == "done" and re.search(
                    r"\b(still\s+)?(open|pending|unfinished)\b"
                    r"|\bnot\s+(yet\s+)?(done|completed|finished)\b",
                    answer, re.IGNORECASE,
                ):
                    return _finding("status: done", "claimed pending/open")
                if status == "open" and resource_id in answer and re.search(
                    r"\b(is\s+(now\s+)?(completed|done|finished))\b"
                    r"|\bhas\s+been\s+completed\b|\bmarked\s+(it\s+)?(as\s+)?done\b",
                    answer, re.IGNORECASE,
                ):
                    return _finding("status: open", "claimed completed")
            return None
        except Exception as e:  # noqa: BLE001 — policies degrade, never crash
            log.debug("grounding_policy_degraded", policy=self.name, error=str(e))
            return None


# Registry (Part 9): the ONLY wiring point — a new policy joins this list,
# nothing in the orchestrator changes.
POLICIES: tuple[GroundingPolicy, ...] = (
    CalculatorGroundingPolicy(),
    DatetimeGroundingPolicy(),
    FileResultGroundingPolicy(),
    StructuredFieldGroundingPolicy(),
    ProviderScheduleGroundingPolicy(),
)


# ── Numeric canonicalization (Part 4) ─────────────────────────────────────────

# Separator characters allowed INSIDE digit groups. Thin/narrow spaces and
# the apostrophe family are unambiguous group separators; the ASCII space is
# a separator only between strict 3-digit groups (enforced by the pattern).
_SEPARATORS = "\u0020\u00a0\u202f\u2009'\u2019"
_SEP_CLASS = re.escape(_SEPARATORS) + ","

# Strict grouping: 1-3 digits, then groups of EXACTLY 3 separated by one
# separator char. This rejects "12,34" (2-digit group) and "1,23" outright.
# NOTE: the dot is deliberately NOT in the separator class — dot-grouping is
# handled by the dedicated multi-dot shape (≥2 groups prove grouping).
_GROUPED = rf"\d{{1,3}}(?:[{_SEP_CLASS}]\d{{3}})+"
# ≥2 strict dot groups (1.234.567) — unambiguous dot-grouping locale.
_MULTI_DOT_GROUP = r"\d{1,3}(?:\.\d{3}){2,}"
# Optional decimal part: point or comma, 1+ digits. A COMMA decimal is only
# accepted on an UNGROUPED integer part (4,5 is European decimal; 41,971 is
# grouped integer — the strict grouping pattern decides, never a guess).
_UNGROUPED_DECIMAL = r"\d+\.\d+"
_COMMA_DECIMAL = r"\d+,\d+"
_PLAIN_INT = r"\d+"

# Number-core alternation shared by single-token canonicalization and
# answer-side extraction. Order matters: strict grouping, then multi-dot,
# then dot/comma decimals, then plain integer.
_NUMBER_ALT = (
    rf"{_GROUPED}(?:\.\d+)?|{_MULTI_DOT_GROUP}|"
    rf"{_UNGROUPED_DECIMAL}|{_COMMA_DECIMAL}|{_PLAIN_INT}"
)

# Full token with optional leading currency marks / trailing unit suffix.
_NUM_TOKEN = re.compile(
    rf"(?<![\w.])(?P<prefix>[$€£¥]\s*|(?i:usd|eur|gbp)\s*)?(?P<sign>-?)"
    rf"(?P<number>{_NUMBER_ALT})"
    rf"(?P<unit>\s?(?:%|usd|eur|gbp|kg|km|mi|cm|mm|ms))?",
    re.IGNORECASE,
)
# Timestamp/ID-shaped RAW tokens (all-digit, no separators) that must never
# be treated as candidate values: epoch seconds/ms/µs/ns and similar widths.
_TIMESTAMP_LENGTHS = (10, 12, 13, 16, 19)


@dataclass(frozen=True)
class CanonicalNumber:
    value: float
    raw: str
    has_decimal_point: bool
    had_separators: bool
    span: tuple[int, int] = (0, 0)   # position of the number group in the source text

    @property
    def display(self) -> str:
        """Preferred human rendering (grouped integer / plain decimal)."""
        if float(self.value).is_integer() and self.value < 1e15:
            return f"{int(self.value):,}"
        return repr(self.value)


# Optional currency/unit affixes around the numeric core. canonical_number
# accepts the SAME affix set as answer-side extraction (Part 4: "currency/
# percentage/unit suffixes when safely parseable"), so evidence tokens and
# answer tokens canonicalize identically.
_AFFIX_RE = re.compile(
    rf"(?P<prefix>[$€£¥]\s*|(?i:usd|eur|gbp)\s*)?(?P<sign>-?)"
    rf"(?P<number>{_NUMBER_ALT})"
    rf"(?P<unit>\s?(?:%|usd|eur|gbp|kg|km|mi|cm|mm|ms))?",
    re.IGNORECASE,
)


def canonical_number(text: str) -> CanonicalNumber | None:
    """
    Canonicalize ONE numeric token (strict; None when not cleanly parseable).

    Handles (Part 4): 41971, 41,971, 41 971, 41'971, ``$1,234`` / ``25%`` /
    ``41971 USD`` wrappers, and integer ≡ decimal equivalence (41,971.0 ≡
    41971). Interpretation rules — each decided by SHAPE, never a guess:

      - comma/space/apostrophe grouping requires STRICT 3-digit groups
        (41,971 ✓; 12,34 ✗ — a non-3-digit group is never a group separator);
      - a comma-decimal is accepted ONLY as ``<d>,<d>`` (4,5 ⇒ 4.5 — sloppy
        grouping for 45 is implausible); multi-digit comma pairs (12,34) are
        ambiguous between a European decimal and sloppy 1,234-style grouping
        ⇒ refused here, upgraded only by answer-side locale siblings;
      - a single strict dot-group (41.971 / 3.142) reads as a DECIMAL, which
        is what this project's own tools emit (Python ``str()`` convention);
        its GROUPED reading is used only by answer-side extraction when ≥2
        sibling dot-groups prove a dot-grouping locale;
      - ≥2 dot groups (1.234.567) are unambiguously dot-grouping.

    Ambiguous input returns None instead of guessing (Part 3: conservative).
    """
    if text is None:
        return None
    s = str(text).strip()
    m = _AFFIX_RE.fullmatch(s)
    if not m:
        return None
    sign = -1 if m.group("sign") else 1
    number = m.group("number")

    had_separators = any(c in number for c in _SEPARATORS + ",.")

    # 1) STRICT NON-DOT GROUPING (comma, space, apostrophe family): 1-3
    #    digits then groups of exactly 3. Any trailing dot-part is a decimal
    #    fraction (41,971.0 ≡ 41971; 41,971.25 ⇒ 41971.25).
    if re.fullmatch(rf"\d{{1,3}}(?:[{_SEP_CLASS}]\d{{3}})+(?:\.\d+)?", number):
        first_sep = re.search(rf"[{_SEP_CLASS}]", number).group(0)
        if first_sep == ".":
            # Strict DOT grouping is rule 2's business; rule 1 must not eat it.
            pass
        else:
            integer_part, _, frac_part = number.partition(".")
            digits = re.sub(rf"[{_SEP_CLASS}]", "", integer_part)
            value = float(f"{digits}.{frac_part}" if frac_part else digits)
            return CanonicalNumber(sign * value, s, bool(frac_part), True)

    # 2) MULTI-GROUP DOT FORM (1.234.567 — ≥2 dot groups) — unambiguous
    #    grouping. (A SINGLE dot-group falls through to rule 4 as a decimal:
    #    this project's tools emit Python str() output, whose dot is a
    #    decimal point. Answer-side extraction additionally refuses to treat
    #    a lone dot-group as a contradiction candidate.)
    if re.fullmatch(r"\d{1,3}(?:\.\d{3}){2,}", number):
        digits = number.replace(".", "")
        return CanonicalNumber(sign * float(digits), s, False, True)

    # 3) UNGROUPED COMMA-DECIMAL — only the single-digit pair (4,5 ⇒ 4.5):
    #    no plausible sloppy-grouping reading exists. Multi-digit pairs
    #    (12,34) stay ambiguous ⇒ refused (never guess).
    if re.fullmatch(r"\d,\d", number):
        return CanonicalNumber(sign * float(number.replace(",", ".")), s, True, False)

    # 4) Anything else with a comma or grouping separator inside is
    #    ambiguous — refuse. A single plain dot is a decimal point (rule 5),
    #    including the strict dot-group shape (41.971 ⇒ 41.971 decimal).
    if "," in number or number.count(".") > 1 or any(c in number for c in _SEPARATORS):
        return None

    # 5) Plain forms: integer or dot-decimal.
    value = float(number)
    return CanonicalNumber(sign * value, s, "." in number, False)


def canonical_numbers(text: str) -> list[CanonicalNumber]:
    """
    All canonicalizable numbers in free text (Part 4). Applies the sibling
    locale rule across the whole text and filters timestamp/ID-shaped tokens
    and standalone-ambiguous forms (lone dot-groups, multi-digit comma pairs).
    """
    return _extract_numbers(text)[0]


def contradictory_numbers(text: str, expected: float) -> list[CanonicalNumber]:
    """Numbers in ``text`` that canonically differ from ``expected``."""
    return [c for c in canonical_numbers(text) if c.value != expected]


def numeric_appearances(text: str, expected: float) -> list[CanonicalNumber]:
    """Numbers in ``text`` that canonically equal ``expected``."""
    return [c for c in canonical_numbers(text) if c.value == expected]


# Locale-shape probes used ONLY for SIBLING evidence across the whole text.
# A strict dot-group (\d{1,3}.\d{3}) or a short comma pair (\d{1,2},\d{1,2})
# is locally ambiguous — ONE such token proves nothing; TWO OR MORE siblings
# of the same shape prove a locale and upgrade all of them together.
_STRICT_DOT_GROUP = re.compile(r"\d{1,3}\.\d{3}\b")
_SHORT_COMMA_PAIR = re.compile(r"\d{1,2},\d{1,2}(?!\d)")


def _extract_numbers(text: str) -> tuple[list[CanonicalNumber], list[str]]:
    """
    Core extraction pass. Returns (canonical numbers, skipped-raw reasons)
    where skipped entries carry a reason among {"timestamp_like",
    "adjacent_digits", "ambiguous"}.

    Ambiguity policy (Part 4 — false positives are worse than misses):
      - a LONE strict dot-group (41.971) or a LONE multi-digit comma pair
        (12,34) is SKIPPED — its decimal/grouping reading cannot be decided;
      - ≥2 strict dot-group siblings upgrade ALL of them to dot-grouping
        ("1.234 and 5.678" ⇒ 1234, 5678);
      - ≥2 short comma-pair siblings upgrade all of them to comma-decimals
        ("4,5 and 2,5" ⇒ 4.5, 2.5);
      - strict 3-digit comma/space/apostrophe grouping (41,971) is always
        unambiguous;
      - epoch-like plain digit runs are never values (timestamps/IDs).
    """
    out: list[CanonicalNumber] = []
    skipped: list[str] = []
    raw_matches = list(_NUM_TOKEN.finditer(text))

    dot_locale = len(_STRICT_DOT_GROUP.findall(text)) >= 2
    comma_locale = len(_SHORT_COMMA_PAIR.findall(text)) >= 2

    for m in raw_matches:
        number = m.group("number")
        sign = -1 if m.group("sign") else 1
        raw = m.group(0).strip()
        span = (m.start("number"), m.end("number"))
        # Timestamp/ID guard: PLAIN digit runs (no separators) of epoch-like
        # width are identifiers, never values.
        if number.isdigit() and len(number) in _TIMESTAMP_LENGTHS:
            skipped.append("timestamp_like")
            continue
        # Adjacent-digit guard: token embedded in a longer digit run.
        start, end = span
        if start > 0 and text[start - 1].isdigit():
            skipped.append("adjacent_digits")
            continue
        if end < len(text) and text[end].isdigit():
            skipped.append("adjacent_digits")
            continue

        strict_dot_group = (
            re.fullmatch(r"\d{1,3}(?:\.\d{3})+", number) is not None
            and number.count(".") == 1
        )
        multi_dot = (
            re.fullmatch(r"\d{1,3}(?:\.\d{3}){2,}", number) is not None
        )
        strict_grouping = (
            re.fullmatch(rf"\d{{1,3}}(?:[{_SEP_CLASS}]\d{{3}})+", number) is not None
            and "." not in number
        )

        if multi_dot:
            # ≥2 dot groups — unambiguous grouping regardless of siblings.
            out.append(
                CanonicalNumber(sign * float(number.replace(".", "")), raw, False, True, span)
            )
            continue
        if strict_grouping:
            # 41,971 / 41 971 / 41'971 — comma/space/apostrophe groups.
            out.append(
                CanonicalNumber(
                    sign * float(re.sub(rf"[{_SEP_CLASS}]", "", number)),
                    raw, False, True, span,
                )
            )
            continue
        if strict_dot_group:
            if dot_locale:
                # ≥2 sibling strict dot-groups prove the locale: grouping.
                out.append(
                    CanonicalNumber(sign * float(number.replace(".", "")), raw, False, True, span)
                )
                continue
            # A LONE strict dot-group reads as a DECIMAL — the same decision
            # canonical_number() makes, so evidence and answer sides always
            # interpret identically and a faithful transcription can never
            # be judged a contradiction.
            canonical = canonical_number(number)
            if canonical is not None:
                out.append(
                    CanonicalNumber(
                        sign * canonical.value, raw,
                        canonical.has_decimal_point, canonical.had_separators, span,
                    )
                )
            else:
                skipped.append("ambiguous")
            continue
        if re.fullmatch(r"\d+,\d+", number):
            if comma_locale and _SHORT_COMMA_PAIR.fullmatch(number):
                # ≥2 sibling short comma-pairs prove comma-locale: decimals.
                out.append(
                    CanonicalNumber(
                        sign * float(number.replace(",", ".")), raw, True, False, span
                    )
                )
                continue
            if re.fullmatch(r"\d,\d", number):
                pass  # single-digit pair: canonical_number handles (4,5 ⇒ 4.5)
            else:
                skipped.append("ambiguous")
                continue

        canonical = canonical_number(number)
        if canonical is not None:
            out.append(
                CanonicalNumber(
                    sign * canonical.value,
                    raw,
                    canonical.has_decimal_point,
                    canonical.had_separators,
                    span,
                )
            )
        else:
            skipped.append("ambiguous")
    return out, skipped


# ── Guard (Part 3) ─────────────────────────────────────────────────────────────


@dataclass
class GroundingVerdict:
    """Outcome of the deterministic post-synthesis check (Part 3)."""

    checked: bool                 # any policy actually inspected the answer
    contradiction: bool           # high-confidence contradiction found
    details: list[dict[str, Any]] = field(default_factory=list)  # safe metadata

    @property
    def ok(self) -> bool:
        return not self.contradiction


def check_grounding(
    answer: str, evidence: list[dict[str, Any]] | None
) -> GroundingVerdict:
    """
    Deterministic post-synthesis validation (Part 3).

    Runs every registered policy over the trusted evidence (policies
    self-select via ``applies``) and reports HIGH-CONFIDENCE contradictions
    only. No model call, no mutation, no fabrication; low confidence passes.

    Per-tool recency (Part 7): when one tool produced SEVERAL observations,
    only the LAST one is enforced — the v0.25 evidence contract states "the
    LAST statement of a tool is definitive", so an earlier superseded value
    can never reject an answer that matches the newest authoritative result
    (and a failed retry never enters the ledger at all).
    """
    items = build_trusted_evidence(evidence)
    # Group applicable items per (tool, policy). Multi-step turns legitimately
    # run ONE tool several times with different inputs (e.g. two calculator
    # steps); the answer may transcribe ANY of this turn's genuine evidence
    # values (Part 7 match-any), while the NEWEST observation provides the
    # expected value for correction/fallback messaging. Failed observations
    # never govern (defense in depth — production ledgers never contain one).
    groups: dict[tuple[str, str], tuple[GroundingPolicy, list[TrustedEvidence]]] = {}
    for item in items:
        if item.status != "ok":
            continue
        for policy in POLICIES:
            if policy.applies(item):
                groups.setdefault((item.tool, policy.name), (policy, []))[1].append(item)
    details: list[dict[str, Any]] = []
    for (tool, _policy_name), (policy, group_items) in groups.items():
        all_values: set[float] = set()
        for it in group_items:
            all_values |= policy.evidence_values(it)
        latest = group_items[-1]
        finding = policy.contradiction(latest, answer)
        if not finding:
            continue
        # Reconcile: a "contradicting" number that matches ANOTHER genuine
        # evidence value of this turn is a legitimate transcription, not a
        # contradiction (conservative — misses beat false rejections).
        remaining: list[str] = []
        for raw in finding.get("contradicting_values", []):
            cn = canonical_number(raw)
            if cn is not None and cn.value in all_values:
                continue
            remaining.append(raw)
        if not remaining:
            continue
        finding["contradicting_values"] = remaining
        if len(all_values) > 1:
            displays = [d for d in (policy.evidence_display(it) for it in group_items) if d]
            if displays:
                finding["expected"] = " / ".join(displays)
        details.append({"tool": tool, **finding})
    return GroundingVerdict(
        checked=bool(groups),
        contradiction=bool(details),
        details=details,
    )


# ── Bounded correction round (Part 5) ─────────────────────────────────────────

CORRECTION_INSTRUCTION = (
    "The authoritative tool result is {expected}. Your previous answer "
    "contradicted the authoritative result ({stated}). Return the answer "
    "using the authoritative result exactly. Do not recompute the "
    "arithmetic. Do not introduce a different value."
)


def build_correction_prompt(details: list[dict[str, Any]]) -> str:
    """
    The one bounded correction-round user message (Part 5). Renders the
    authoritative value(s) and the contradicting statement, with an explicit
    do-not-recompute contract. Safe metadata only.
    """
    lines = ["GROUNDING CORRECTION REQUIRED."]
    for d in details[:4]:
        lines.append(
            CORRECTION_INSTRUCTION.format(
                expected=d.get("expected", "the authoritative value"),
                stated=", ".join(d.get("contradicting_values", []) or ["your stated value"]),
            )
        )
    lines.append(
        "Answer the original request now using only the authoritative "
        "result(s) above."
    )
    return "\n".join(lines)


def build_fallback_answer(details: list[dict[str, Any]]) -> str:
    """
    Honest fail-closed response (Part 6): preserves the authoritative value,
    states that the generated answer could not be verified, and never
    fabricates content. Used ONLY when the single correction round ALSO
    contradicts the evidence.
    """
    lines = [
        "I could not produce a verified answer for this request.",
        "The authoritative tool result(s) are:",
    ]
    for d in details[:4]:
        src = str(d.get("source") or "live")
        provenance = "" if src == "live" else f", {src}"
        lines.append(
            f"- {d.get('expected', 'unavailable')} (from {d.get('tool', 'tool')}"
            f"{provenance})"
        )
    lines.append(
        "My generated response contradicted these measured values, so it "
        "was withheld rather than presented."
    )
    return "\n".join(lines)
