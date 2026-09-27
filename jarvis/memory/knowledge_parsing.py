"""
jarvis/memory/knowledge_parsing.py
──────────────────────────────────
v0.20 personal knowledge base: document parsing + deterministic chunking.

Parsers
───────
One parser per media type, all returning the same ``ParsedDocument``
shape: a list of ``Page`` units (page 0 for non-paginated formats) each
carrying its text and start line number. Supported, in dependency order:

  - .txt / .md / source code / generic text  → stdlib (always available)
  - .json                                    → stdlib json (pretty-printed)
  - .pdf                                     → pypdf (explicit dependency;
                                                page boundaries preserved)

Extraction failures never crash ingestion: they surface as ``ParserError``
and are reported per document.

Chunking
────────
Deterministic paragraph-first chunking: split on blank lines within a page,
pack paragraphs up to ``target_chars``, hard-split oversized paragraphs,
and carry ``overlap_chars`` from the previous chunk. Identical input +
identical settings ⇒ identical chunk sequence (stable chunk ids for
incremental ingestion). Every chunk carries traceable metadata:
document_id, source, filename, page, line range, chunk index, content
hash, and the section/title it belongs to when one is detectable.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# Bumped when parsing behavior changes in a way that should show in metadata.
PARSER_VERSION = "1"

# Text formats parsed with the stdlib (extension → media type).
_TEXT_SUFFIXES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".rst": "text/plain",
    ".py": "text/x-python",
    ".js": "text/javascript",
    ".ts": "text/typescript",
    ".tsx": "text/typescript",
    ".jsx": "text/javascript",
    ".java": "text/x-java",
    ".c": "text/x-c",
    ".h": "text/x-c",
    ".cpp": "text/x-c++",
    ".go": "text/x-go",
    ".rs": "text/x-rust",
    ".rb": "text/x-ruby",
    ".sh": "text/x-shellscript",
    ".toml": "text/plain",
    ".ini": "text/plain",
    ".cfg": "text/plain",
    ".csv": "text/csv",
    ".log": "text/plain",
    ".yml": "text/yaml",
    ".yaml": "text/yaml",
    ".html": "text/html",
    ".xml": "text/xml",
}

# Never ingest these, ever — even if an operator explicitly points at them.
# (Explicit refusal beats "allowed dir" heuristics for credential material.)
_FORBIDDEN_NAMES = {".env", ".env.local", ".env.production", ".env.development"}
_FORBIDDEN_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".keystore", ".kdbx")
_SECRET_HINTS = ("password", "passwd", "secret", "credential", "private_key")

_MAX_FILE_BYTES = 20_000_000  # refuse >20 MB inputs; parsing is in-memory


class ParserError(Exception):
    """A document could not be parsed (reported per document, never fatal)."""


@dataclass
class Page:
    """One parse unit: a PDF page, or a whole non-paginated text file."""

    number: int  # 1-based for PDFs; 0 = "no pagination"
    text: str
    start_line: int  # 1-based line where this page starts in the full text


@dataclass
class ParsedDocument:
    """Parser output: page units + safe metadata (no raw bytes retained)."""

    document_id: str
    source_path: str
    filename: str
    media_type: str
    size_bytes: int
    content_hash: str
    parser_version: str
    modified_at: str | None
    pages: list[Page] = field(default_factory=list)


def detect_media_type(filename: str) -> str:
    """Media type from the extension; '' when unsupported (caller decides)."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    if suffix == ".json":
        return "application/json"
    return _TEXT_SUFFIXES.get(suffix, "")


def is_supported_document(filename: str) -> bool:
    return detect_media_type(filename) != ""


def is_explicitly_forbidden(filename: str) -> bool:
    """Credential-material refusal: independent of any allowed-directory rule."""
    name = Path(filename).name.lower()
    if name in _FORBIDDEN_NAMES:
        return True
    if name.endswith(_FORBIDDEN_SUFFIXES):
        return True
    lowered = name
    return any(h in lowered for h in _SECRET_HINTS if h in lowered) and (
        name.endswith((".txt", ".md", ".json", ".yaml", ".yml", ".csv"))
    )


def compute_content_hash(data: bytes) -> str:
    """Stable sha256 of the raw bytes (dedup + change detection key)."""
    return hashlib.sha256(data).hexdigest()


def compute_document_id(source_path: str, content_hash: str) -> str:
    """Stable identity for a document: absolute path ⊕ content hash.

    Same path + same bytes ⇒ same id (dedup). Same path with new bytes ⇒ a
    NEW id, so the old version's chunks can be addressed for replacement.
    """
    raw = f"{Path(source_path).resolve().as_posix().lower()}::{content_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


# ── Parsers ───────────────────────────────────────────────────────────────────


def _parse_text(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _parse_pdf(data: bytes) -> list[Page]:
    """Page-preserving PDF extraction via pypdf (explicit dependency)."""
    import io

    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover - dependency is declared
        raise ParserError("PDF support requires the 'pypdf' package") from e
    try:
        reader = PdfReader(io.BytesIO(data))
        pages: list[Page] = []
        line_cursor = 1
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as e:  # a single broken page must not kill all
                log.warning("pdf_page_extract_failed", page=i, error=str(e))
                text = ""
            pages.append(Page(number=i, text=text, start_line=line_cursor))
            line_cursor += text.count("\n") + 1
        if not pages:
            raise ParserError("PDF contains no pages")
        return pages
    except ParserError:
        raise
    except Exception as e:
        raise ParserError(f"PDF extraction failed: {e}") from e


def _parse_json(data: bytes) -> list[Page]:
    """Parse JSON and pretty-print it so chunking sees stable structure."""
    raw = _parse_text(data)
    try:
        pretty = json.dumps(json.loads(raw), indent=2, ensure_ascii=False)
    except (json.JSONDecodeError, ValueError) as e:
        raise ParserError(f"invalid JSON: {e}") from e
    return [Page(number=0, text=pretty, start_line=1)]


def _parse_generic(data: bytes, media_type: str) -> list[Page]:
    text = _parse_text(data)
    return [Page(number=0, text=text, start_line=1)]


def parse_document(path: Path, allowed_root: Path) -> ParsedDocument:
    """
    Read + parse one file within the allowed root, enforcing the security
    rules (resolution, containment, refusal list, size cap) BEFORE parsing.
    Raises FileNotFoundError / ParserError / PermissionError as appropriate.
    """
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    if not path.is_file():
        raise ParserError(f"not a regular file: {path}")

    resolved = path.resolve()
    # Containment: the resolved real path must live inside the allowed root
    # (defeats traversal like docs/../../secrets/x.pdf and symlink escapes).
    if allowed_root not in resolved.parents and resolved != allowed_root:
        raise PermissionError(
            f"path is outside the allowed directory ({allowed_root}): {path}"
        )

    if is_explicitly_forbidden(resolved.name):
        raise PermissionError(
            f"refusing to ingest credential-like file: {resolved.name}"
        )

    media_type = detect_media_type(resolved.name)
    if not media_type:
        raise ParserError(
            f"unsupported document type '{resolved.suffix}' "
            f"(supported: txt, md, code, json, pdf)"
        )

    data = resolved.read_bytes()
    if len(data) > _MAX_FILE_BYTES:
        raise ParserError(
            f"file too large ({len(data)} bytes > {_MAX_FILE_BYTES})"
        )
    if not data:
        raise ParserError("file is empty")

    content_hash = compute_content_hash(data)
    document_id = compute_document_id(str(resolved), content_hash)

    if media_type == "application/pdf":
        pages = _parse_pdf(data)
    elif media_type == "application/json":
        pages = _parse_json(data)
    else:
        pages = _parse_generic(data, media_type)

    modified_at = None
    try:
        modified_at = datetime.fromtimestamp(
            resolved.stat().st_mtime, tz=timezone.utc
        ).isoformat()
    except OSError:  # pragma: no cover - stat failures degrade gracefully
        pass

    return ParsedDocument(
        document_id=document_id,
        source_path=str(resolved),
        filename=resolved.name,
        media_type=media_type,
        size_bytes=len(data),
        content_hash=content_hash,
        parser_version=PARSER_VERSION,
        modified_at=modified_at,
        pages=pages,
    )


# ── Chunking ──────────────────────────────────────────────────────────────────

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_MAX_PARAGRAPH_CHARS = 4000  # hard split guard for minified/one-paragraph files


@dataclass
class Chunk:
    """One embeddable unit with full trace-back metadata."""

    chunk_id: str
    document_id: str
    text: str
    page: int
    start_line: int
    end_line: int
    chunk_index: int
    section: str | None
    content_hash: str


def _detect_section(lines: list[str], line_index: int) -> str | None:
    """Nearest markdown-style heading at/above ``line_index`` (max 40 up)."""
    for i in range(line_index, max(-1, line_index - 40), -1):
        stripped = lines[i].strip() if 0 <= i < len(lines) else ""
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()[:120]
    return None


def chunk_document(
    parsed: ParsedDocument,
    target_chars: int = 1200,
    overlap_chars: int = 150,
) -> list[Chunk]:
    """Deterministic, LOSSLESS paragraph-first chunking over parsed pages.

    Algorithm (identical input + settings ⇒ identical chunk sequence):
      1. Split each page into paragraphs on blank lines.
      2. Hard-split any paragraph over ``_MAX_PARAGRAPH_CHARS`` into pieces.
      3. Pack units into chunks up to ``target_chars`` (units never span
         pages; a unit larger than the target becomes its own chunk —
         nothing is ever truncated or dropped).
      4. Carry ``overlap_chars`` of the previous chunk's tail into the next
         chunk of the same page for local continuity.
    """
    if target_chars < 200 or target_chars > 8000:
        raise ValueError(f"target_chars out of range [200, 8000]: {target_chars}")
    if overlap_chars < 0 or overlap_chars >= target_chars:
        raise ValueError(
            f"overlap_chars must be in [0, target_chars): {overlap_chars}"
        )

    chunks: list[Chunk] = []

    for page in parsed.pages:
        full_text = page.text
        lines = full_text.split("\n")
        # (start_ln, end_ln, text) units, page-local 1-based line numbers.
        units: list[tuple[int, int, str]] = []
        pos = 0
        for match in _PARAGRAPH_SPLIT.finditer(full_text):
            para = full_text[pos:match.start()]
            if para.strip():
                start_ln = full_text.count("\n", 0, pos) + 1
                end_ln = full_text.count("\n", 0, match.start()) + 1
                units.append((start_ln, end_ln, para))
            pos = match.end()
        tail = full_text[pos:]
        if tail.strip():
            units.append(
                (full_text.count("\n", 0, pos) + 1, len(lines), tail)
            )

        # Hard-split oversized units (deterministic, line-aware).
        pieces: list[tuple[int, int, str]] = []
        for start_ln, end_ln, text in units:
            if len(text) <= _MAX_PARAGRAPH_CHARS:
                pieces.append((start_ln, end_ln, text))
                continue
            stride = _MAX_PARAGRAPH_CHARS
            for i in range(0, len(text), stride):
                piece = text[i:i + stride]
                piece_start = start_ln + text[:i].count("\n")
                piece_end = start_ln + text[:i + stride].count("\n")
                pieces.append((piece_start, piece_end, piece))

        # Pack pieces into chunks (page-scoped).
        current: list[str] = []
        cur_start = cur_end = 0
        cur_section: str | None = None

        def flush() -> None:
            nonlocal current, cur_start, cur_end, cur_section
            if not current:
                return
            text = "\n\n".join(current)
            if chunks and overlap_chars and chunks[-1].page == page.number:
                text = f"{chunks[-1].text[-overlap_chars:]}\n{text}"
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            chunk = Chunk(
                chunk_id=f"{parsed.document_id}:{len(chunks)}:{content_hash}",
                document_id=parsed.document_id,
                text=text,
                page=page.number,
                start_line=cur_start,
                end_line=cur_end,
                chunk_index=len(chunks),
                section=cur_section,
                content_hash=content_hash,
            )
            chunks.append(chunk)
            current = []
            cur_section = None

        for piece_start, piece_end, piece in pieces:
            section = _detect_section(lines, piece_start - 1)
            if cur_section is None:
                cur_section = section
            projected = len(piece) if not current else (
                len("\n\n".join(current)) + 2 + len(piece)
            )
            if current and projected > target_chars:
                flush()
                cur_section = section
            if not current:
                cur_start = piece_start
            cur_end = piece_end
            current.append(piece)
        flush()

    return chunks
