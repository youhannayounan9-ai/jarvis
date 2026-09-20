"""
jarvis/tools/code_execution.py
──────────────────────────────
Tool: execute_python_code

Executes Python code in a restricted sandbox namespace.
"""

import builtins
import io
import math
import random
import threading
from datetime import datetime
from typing import Any, Dict

from jarvis.tools.base import BaseTool
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# Allowed builtins for the restricted sandbox
ALLOWED_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "ascii": ascii,
    "bin": bin,
    "bool": bool,
    "bytearray": bytearray,
    "bytes": bytes,
    "callable": callable,
    "chr": chr,
    "complex": complex,
    "dict": dict,
    "dir": dir,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "getattr": getattr,
    "hasattr": hasattr,
    "hash": hash,
    "hex": hex,
    "id": id,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "object": object,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,
    "property": property,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "setattr": setattr,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "super": super,
    "tuple": tuple,
    "type": type,
    "zip": zip,
}


def _run_with_timeout(func, timeout_sec: float) -> tuple[Any, Exception | None]:
    result = None
    exception = None

    def worker():
        nonlocal result, exception
        try:
            result = func()
        except Exception as e:
            exception = e

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout_sec)
    
    if t.is_alive():
        return None, TimeoutError(f"Execution timed out after {timeout_sec}s.")
    return result, exception


class CodeExecutionTool(BaseTool):
    name = "execute_python_code"
    description = (
        "CRITICAL: Use this to execute Python code in a safe sandbox. "
        "Only for calculation and data processing. Do not import unsafe modules."
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute. Standard output (print) is captured.",
            },
        },
        "required": ["code"],
    }
    risk_level = "SYSTEM"
    timeout_seconds = 10.0

    def run(self, code: str, **kwargs: Any) -> str:
        log.info("code_execution_start", length=len(code))
        
        # Static check for blatantly unsafe words
        unsafe_keywords = ["__import__", "eval", "exec", "open", "os.", "sys.", "subprocess."]
        for k in unsafe_keywords:
            if k in code:
                return f"ERROR: Security violation. Use of '{k}' is forbidden."

        output_buffer = io.StringIO()
        
        # Prepare execution environment
        def safe_print(*args, **kw):
            kw["file"] = output_buffer
            print(*args, **kw)
        
        safe_builtins = dict(ALLOWED_BUILTINS)
        safe_builtins["print"] = safe_print
        
        namespace: Dict[str, Any] = {
            "__builtins__": safe_builtins,
            "math": math,
            "random": random,
            "datetime": datetime,
        }

        def execute():
            # exec operates in the provided namespace
            exec(code, namespace)

        _, exc = _run_with_timeout(execute, timeout_sec=5.0)

        if exc:
            if isinstance(exc, (NameError, AttributeError, ImportError)):
                log.error("code_execution_security_violation", error=str(exc))
                return "ERROR: Security violation: Restricted function or module blocked."
            log.error("code_execution_failed", error=str(exc))
            if isinstance(exc, TimeoutError):
                return f"ERROR: {exc}"
            return f"ERROR: Exception during execution:\n{exc}"

        captured = output_buffer.getvalue()
        log.info("code_execution_success", output_length=len(captured))
        return captured or "(Execution finished with no output)"
