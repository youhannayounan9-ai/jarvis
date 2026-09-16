"""
jarvis/tools/write_file.py
─────────────────────────
Tool: write_file

Allows the LLM to write or append text to a file inside the allowed sandbox directory.
"""

from pathlib import Path

from jarvis.config import settings
from jarvis.tools.base import BaseTool
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class WriteFileTool(BaseTool):
    name = "write_file"
    description = (
        "CRITICAL: Use this to write or append text to a file. "
        "You MUST provide a valid filename and content."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The path to the file to write to, relative to the current directory.",
            },
            "content": {
                "type": "string",
                "description": "The content to write to the file.",
            },
            "append": {
                "type": "boolean",
                "description": "If true, append to the file instead of overwriting. Default is false.",
            },
        },
        "required": ["file_path", "content"],
    }
    risk_level = "FILE_WRITE"
    timeout_seconds = 10.0

    def run(self, file_path: str, content: str, append: bool = False, **kwargs) -> str:
        try:
            # Resolve absolute path and prevent traversal
            target = Path(file_path).resolve()
            allowed_dir = settings.file_reader_allowed_path

            if not target.is_relative_to(allowed_dir):
                log.warning("write_file_path_traversal_blocked", path=str(target))
                return f"ERROR: Access denied. You can only write to files inside '{allowed_dir}'."

            # Ensure parent directories exist
            target.parent.mkdir(parents=True, exist_ok=True)

            mode = "a" if append else "w"
            with open(target, mode, encoding="utf-8") as f:
                f.write(content)

            log.info("write_file_success", path=str(target), append=append, bytes=len(content))
            return f"Successfully wrote {len(content)} bytes to {target.name}."

        except Exception as e:
            log.error("write_file_error", error=str(e), path=file_path)
            return f"ERROR: Failed to write file. {e}"
