"""
jarvis/integrations/validation.py
─────────────────────────────────
v0.29 Part 8 — no silent time guessing.

An external scheduling action must never be built from vague input:
"book something tomorrow" NEVER becomes a 09:00 default silently. The
contract:

  - explicit DATE (ISO YYYY-MM-DD), explicit START TIME (HH:MM 24h),
    explicit TIMEZONE (IANA), and a non-empty TITLE are REQUIRED for a
    calendar create/update that moves an event;
  - DURATION is required at creation (minutes, positive, bounded);
  - everything is validated BEFORE a confirmation preview exists — the
    user confirms a fully-specified action, never a guess;
  - normalization is explicit and returned (offset computed for the given
    timezone), never invented;
  - Attendee addresses are syntactically checked; free text is refused.

Deterministic; no LLM; no date math beyond timezone localization.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone as _timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

MAX_TITLE_CHARS = 200
MAX_NOTES_CHARS = 2000
MAX_ATTENDEES = 20
MAX_DURATION_MINUTES = 24 * 60


class EventValidationError(ValueError):
    """User-facing validation refusal (safe, specific, bounded message)."""


@dataclass(frozen=True)
class EventWindow:
    """A fully-specified, validated event time window."""

    date: str            # YYYY-MM-DD
    start_time: str      # HH:MM (24h)
    end_time: str        # HH:MM (24h, derived from duration)
    timezone_name: str   # IANA name as supplied
    duration_minutes: int
    start_utc: datetime  # explicit normalized instant (for previews/audit)
    end_utc: datetime
    start_local: datetime

    def preview_lines(self) -> list[str]:
        """Deterministic human preview (Part 9) — exactly what will be sent."""
        return [
            f"date: {self.date}",
            f"start: {self.start_time}",
            f"end: {self.end_time}",
            f"duration: {self.duration_minutes} min",
            f"tz: {self.timezone_name}",
            f"start_utc: {self.start_utc.strftime('%Y-%m-%dT%H:%MZ')}",
        ]


@dataclass(frozen=True)
class ValidatedEvent:
    """A fully-specified calendar event, ready for preview + provider."""

    title: str
    window: EventWindow
    notes: str = ""
    attendees: tuple[str, ...] = field(default_factory=tuple)

    def preview_lines(self) -> list[str]:
        lines = [f"title: {self.title}", *self.window.preview_lines()]
        if self.attendees:
            lines.append(f"attendees: {', '.join(self.attendees)}")
        if self.notes:
            lines.append(f"notes: {self.notes[:120]}")
        return lines


def _parse_timezone(name: str) -> ZoneInfo:
    tz_name = str(name or "").strip()
    if not tz_name:
        raise EventValidationError("timezone is required (IANA name, e.g. 'Europe/Berlin')")
    try:
        return ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError, TypeError):
        raise EventValidationError(
            f"unknown timezone {tz_name!r} — supply an IANA name like 'Europe/Berlin' "
            "or 'America/New_York' (never an offset guess)"
        ) from None


def build_event_window(
    *,
    date: str,
    start_time: str,
    timezone_name: str,
    duration_minutes: int,
) -> EventWindow:
    """
    Validate + normalize one explicit time window. Every component must be
    present and well-formed; nothing is defaulted silently.
    """
    date = (date or "").strip()
    if not _DATE_RE.match(date):
        raise EventValidationError(
            "date must be explicit and ISO-formatted (YYYY-MM-DD); "
            "vague dates are never guessed"
        )
    try:
        parsed_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise EventValidationError(f"date {date!r} is not a real calendar date") from None

    start = (start_time or "").strip()
    m = _TIME_RE.match(start)
    if not m:
        raise EventValidationError(
            "start time must be explicit 24-hour HH:MM; "
            "missing times are never guessed"
        )
    hour, minute = int(m.group(1)), int(m.group(2))
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise EventValidationError(f"start time {start!r} is not a valid time")

    try:
        duration = int(duration_minutes)
    except (TypeError, ValueError):
        raise EventValidationError("duration_minutes must be a whole number of minutes") from None
    if duration <= 0:
        raise EventValidationError("duration_minutes must be positive")
    if duration > MAX_DURATION_MINUTES:
        raise EventValidationError(
            f"duration_minutes is capped at {MAX_DURATION_MINUTES} (one day)"
        )

    tz = _parse_timezone(timezone_name)
    try:
        start_local = datetime(
            parsed_date.year, parsed_date.month, parsed_date.day, hour, minute, tzinfo=tz
        )
    except ValueError:
        raise EventValidationError(f"{date} {start} is not a valid local time") from None
    end_local = start_local + timedelta(minutes=duration)
    # Keep the tz NAME (str) as the portable field value — the ZoneInfo
    # object itself is never stored in fields that flow into provider
    # payloads, previews, or output rendering.
    tz_name = str(getattr(tz, "key", timezone_name)).strip() or str(timezone_name).strip()
    return EventWindow(
        date=date,
        start_time=f"{hour:02d}:{minute:02d}",
        end_time=end_local.strftime("%H:%M"),
        timezone_name=tz_name,
        duration_minutes=duration,
        start_utc=start_local.astimezone(_timezone.utc),
        end_utc=end_local.astimezone(_timezone.utc),
        start_local=start_local,
    )


def validate_event(
    *,
    title: str,
    date: str,
    start_time: str,
    timezone_name: str,
    duration_minutes: int,
    notes: str = "",
    attendees: list[str] | None = None,
) -> ValidatedEvent:
    """
    Fully validate an event for creation. Raises EventValidationError with
    the FIRST missing/invalid component — the caller surfaces it verbatim
    (this is the honest ask-for-details path, not an error to retry).
    """
    clean_title = (title or "").strip()
    if not clean_title:
        raise EventValidationError("title is required — never invent one")
    if len(clean_title) > MAX_TITLE_CHARS:
        raise EventValidationError(f"title must be at most {MAX_TITLE_CHARS} characters")
    clean_notes = (notes or "").strip()
    if len(clean_notes) > MAX_NOTES_CHARS:
        raise EventValidationError(f"notes must be at most {MAX_NOTES_CHARS} characters")

    clean_attendees: list[str] = []
    for raw in (attendees or [])[:MAX_ATTENDEES]:
        addr = str(raw).strip()
        if not addr:
            continue
        if not _EMAIL_RE.match(addr):
            raise EventValidationError(
                f"attendee {addr[:40]!r} is not a plain email address"
            )
        clean_attendees.append(addr)

    window = build_event_window(
        date=date,
        start_time=start_time,
        timezone_name=timezone_name,
        duration_minutes=duration_minutes,
    )
    return ValidatedEvent(
        title=clean_title,
        window=window,
        notes=clean_notes,
        attendees=tuple(clean_attendees),
    )
