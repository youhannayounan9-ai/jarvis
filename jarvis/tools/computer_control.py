"""
jarvis/tools/computer_control.py
────────────────────────────────
Tool: computer_control

Uses PyAutoGUI to safely execute mouse and keyboard actions.
"""

import time
from typing import Any

from jarvis.tools.base import BaseTool
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class ComputerControlTool(BaseTool):
    name = "computer_control"
    description = (
        "CRITICAL: Use this to control the mouse and keyboard. "
        "Requires explicit action type and parameters."
    )
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
    timeout_seconds = 10.0

    def run(self, action: str, **kwargs: Any) -> str:
        try:
            import pyautogui
            # Failsafe: moving mouse to a corner aborts PyAutoGUI
            pyautogui.FAILSAFE = True
            
            x = kwargs.get("x")
            y = kwargs.get("y")
            text = kwargs.get("text")
            key = kwargs.get("key")
            amount = kwargs.get("amount")

            log.info("computer_control_start", action=action, kwargs=kwargs)

            # Execution logic
            if action == "move_mouse":
                if x is None or y is None:
                    return "ERROR: Missing x or y for move_mouse."
                pyautogui.moveTo(x, y, duration=0.5)
            
            elif action == "click":
                if x is not None and y is not None:
                    pyautogui.click(x=x, y=y)
                else:
                    pyautogui.click()
            
            elif action == "type_text":
                if not text:
                    return "ERROR: Missing text for type_text."
                pyautogui.write(text, interval=0.01)
            
            elif action == "press_key":
                if not key:
                    return "ERROR: Missing key for press_key."
                # Handle hotkeys like 'ctrl+c'
                if "+" in key:
                    keys = key.split("+")
                    pyautogui.hotkey(*keys)
                else:
                    pyautogui.press(key)
            
            elif action == "scroll":
                if amount is None:
                    return "ERROR: Missing amount for scroll."
                pyautogui.scroll(int(amount))
                
            else:
                return f"ERROR: Unknown action {action}."
            
            # Safety pause
            time.sleep(0.5)
            log.info("computer_control_success", action=action)
            return f"Successfully executed action: {action}"

        except Exception as e:
            log.error("computer_control_error", action=action, error=str(e))
            return f"ERROR: Computer control failed. {e}"
