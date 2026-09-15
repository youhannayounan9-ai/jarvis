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
        "Returns the current local date, time, day of the week, and UTC offset. "
        "ONLY call this tool when the user explicitly asks what time or date it is. "
        "Do NOT call this for any other type of question."
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
