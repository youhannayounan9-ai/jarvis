"""
jarvis/tools/directory_lister.py
───────────────────────────────
Tool: list_directory

Lists the contents of a directory.

Security model — sandboxing:
  Like read_file, this tool is restricted to FILE_READER_ALLOWED_DIR.
  It ensures the resolved path sits within the allowed directory bounds.
"""

from pathlib import Path

from jarvis.config import settings
from jarvis.tools.base import BaseTool
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class ListDirectoryTool(BaseTool):
    name = "list_directory"
    description = (
        "List the contents (files and folders) of a specified directory on the local filesystem. "
        "The path must be relative to the allowed working directory. "
        "ONLY use this tool when the user asks to see what files or folders exist in a specific path."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Path to the directory to list. "
                    "Can be relative (e.g. '.') or absolute. "
                    "Must be within the allowed directory."
                ),
            },
        },
        "required": ["path"],
    }

    def run(self, path: str, **kwargs) -> str:
        allowed_dir = settings.file_reader_allowed_path

        try:
            requested = Path(path).resolve()
        except Exception as e:
            return f"ERROR: Invalid path '{path}': {e}"

        # ── Sandbox check ──────────────────────────────────────────────────────
        try:
            requested.relative_to(allowed_dir)
        except ValueError:
            log.warning(
                "dir_list_denied",
                path=str(requested),
                allowed_dir=str(allowed_dir),
            )
            return (
                f"ERROR: Access denied. '{path}' is outside the allowed directory "
                f"({allowed_dir}). Only directories within that path may be listed."
            )

        if not requested.exists():
            return f"ERROR: Directory not found: '{path}'"

        if not requested.is_dir():
            return f"ERROR: '{path}' is not a directory (it may be a file)."

        log.info("dir_list", path=str(requested))

        try:
            items = list(requested.iterdir())
        except Exception as e:
            log.error("dir_list_error", path=str(requested), error=str(e))
            return f"ERROR: Could not list directory '{path}': {e}"

        if not items:
            return f"Directory '{path}' is empty."

        # Sort: directories first, then files, both alphabetically
        dirs = sorted([item.name + "/" for item in items if item.is_dir()])
        files = sorted([item.name for item in items if item.is_file()])

        lines = [f"Contents of directory '{path}':\n"]
        for d in dirs:
            lines.append(f"📁 {d}")
        for f in files:
            lines.append(f"📄 {f}")

        return "\n".join(lines)
