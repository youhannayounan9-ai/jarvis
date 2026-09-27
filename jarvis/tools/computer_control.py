"""
jarvis/tools/computer_control.py
────────────────────────────────
Tool: computer_control

PLACEHOLDER — real computer automation is currently disabled.

This module is intentionally kept so the tool class can be imported by
tests and as a structural placeholder for the future safe implementation
(e.g., a Docker-isolated or explicitly permission-gated executor).

DO NOT re-enable pyautogui or any desktop-control library here until a
proper sandboxing / human-approval workflow is in place.
"""

from typing import Any

from jarvis.tools.base import BaseTool
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

_DISABLED_MSG = (
    "ERROR: Computer control is currently disabled. "
    "The capability has been turned off because it directly manipulates "
    "the host machine's mouse and keyboard without a safe, isolated "
    "execution boundary. It will be re-enabled once a proper approval "
    "workflow and sandboxing strategy are implemented."
)


class ComputerControlTool(BaseTool):
    name = "computer_control"
    description = "CRITICAL: This tool is currently disabled. Do not attempt to use it."
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["move_mouse", "click", "type_text", "press_key", "scroll"],
                "description": "The type of automation action.",
            },
            "x": {
                "type": "number",
                "description": "X coordinate for mouse movement or click.",
            },
            "y": {
                "type": "number",
                "description": "Y coordinate for mouse movement or click.",
            },
            "text": {
                "type": "string",
                "description": "Text to type (for type_text).",
            },
            "key": {
                "type": "string",
                "description": "Key or shortcut to press (e.g., 'enter', 'ctrl+c').",
            },
            "amount": {
                "type": "number",
                "description": "Scroll amount (positive for up, negative for down).",
            },
        },
        "required": ["action"],
    }
    risk_level = "SYSTEM"
    timeout_seconds = 1.0

    def run(self, action: str, **kwargs: Any) -> str:
        log.warning("computer_control_disabled_attempt", action=action)
        return _DISABLED_MSG
