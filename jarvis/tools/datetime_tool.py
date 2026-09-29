"""
jarvis/tools/datetime_tool.py
──────────────────────────────
Tool: get_current_datetime

Returns the current date, time, day of week, and timezone.
No external dependencies — stdlib only.
"""

from datetime import datetime, timezone

from jarvis.tools.base import BaseTool


class GetCurrentDatetimeTool(BaseTool):
    name = "get_current_datetime"
    description = (
        "Get the current local date, time, day of the week, and UTC offset. "
        "PURPOSE: anchor 'now' for any time-dependent reasoning. "
        "WHEN TO USE: the user asks for the current time/date, or a request "
        "depends on it ('today', 'tomorrow', 'next week', deadlines, ages, "
        "countdowns — pair with calculator for spans). "
        "WHEN NOT TO USE: purely historical or arithmetic date questions that "
        "give all dates explicitly and need no anchor. "
        "INPUT: none. OUTPUT: one line like 'Current datetime: Monday, "
        "January 05, 2026 at 14:03:11 (UTC+0100)'."
    )
    parameters = {
        "type": "object",
        "properties": {},  # No arguments needed
        "required": [],
    }

    def run(self, **kwargs) -> str:
        now = datetime.now(tz=timezone.utc).astimezone()  # local timezone
        return (
            f"Current datetime: {now.strftime('%A, %B %d, %Y at %H:%M:%S')} "
            f"(UTC{now.strftime('%z')})"
        )
