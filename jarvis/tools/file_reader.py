"""
jarvis/tools/file_reader.py
────────────────────────────
Tool: read_file

Reads a text file and returns its contents.

Security model — sandboxing:
  The tool resolves the requested path to an absolute path and checks that
  it starts with the configured `FILE_READER_ALLOWED_DIR`.
  This prevents path traversal attacks like:
    - read_file("../../etc/passwd")
    - read_file("/absolute/secret/path")

  In v0.1 the allowed dir defaults to "." (the project directory).
  Users can configure a more restrictive path in .env.

File size guard:
  Reading a 500 MB log file into the LLM's context would be a bad time.
  We cap at MAX_BYTES and tell the LLM what happened so it can ask for
  a more specific query.
"""

from pathlib import Path

from jarvis.config import settings
from jarvis.tools.base import BaseTool
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

_MAX_BYTES = 32_000  # ~8k tokens; stays well within most context windows


class ReadFileTool(BaseTool):
    name = "read_file"
    description = (
        "Read the text contents of a specific file on the local filesystem. "
        "ONLY use this tool when the user explicitly names a file or path they "
        "want you to read, analyse, or summarise (e.g. 'read my config.py', "
        "'what does notes.txt say?'). "
        "Do NOT call this tool speculatively, to look up your own instructions, "
        "or when the user has not mentioned a specific file by name. "
        "NEVER use this tool to try to read 'conversation.log' or recover chat history."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Path to the file to read. "
                    "Can be relative (e.g. 'README.md') or absolute. "
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
        # Ensure the resolved path is inside the allowed directory.
        try:
            requested.relative_to(allowed_dir)
        except ValueError:
            log.warning(
                "file_read_denied",
                path=str(requested),
                allowed_dir=str(allowed_dir),
            )
            return (
                f"ERROR: Access denied. '{path}' is outside the allowed directory "
                f"({allowed_dir}). Only files within that directory may be read."
            )

        if not requested.exists():
            return f"ERROR: File not found: '{path}'"

        if not requested.is_file():
            return f"ERROR: '{path}' is not a file (it may be a directory)."

        log.info("file_read", path=str(requested))

        try:
            content = requested.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            log.error("file_read_error", path=str(requested), error=str(e))
            return f"ERROR: Could not read file '{path}': {e}"

        # ── Size guard ─────────────────────────────────────────────────────────
        if len(content.encode("utf-8")) > _MAX_BYTES:
            truncated = content.encode("utf-8")[:_MAX_BYTES].decode("utf-8", errors="replace")
            return (
                f"[FILE TRUNCATED — showing first {_MAX_BYTES} bytes of '{path}']\n\n"
                f"{truncated}\n\n"
                f"[... file continues. Ask for a specific section if you need more.]"
            )

        return f"Contents of '{path}':\n\n{content}"
