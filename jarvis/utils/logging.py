"""
jarvis/utils/logging.py
───────────────────────
Structured logging setup using structlog.

Why structlog over Python's built-in logging?
  - Every log call produces a JSON-serialisable dict (key=value pairs).
  - Much easier to grep, filter, and pipe into future log aggregators.
  - Clean API: log.info("event", key=value) instead of string formatting.

Call `setup_logging()` once at application startup (in main.py).
Then use `get_logger(__name__)` in every module that needs a logger.
"""

import logging
import sys

import structlog


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure structlog and the standard library logging bridge.

    Args:
        log_level: One of DEBUG, INFO, WARNING, ERROR. Read from settings.
    """
    # Translate the string level to a logging constant.
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure the stdlib root logger (captures any library that uses it).
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=numeric_level,
    )

    structlog.configure(
        processors=[
            # Add log level to every event dict.
            structlog.stdlib.add_log_level,
            # Add timestamp.
            structlog.processors.TimeStamper(fmt="iso"),
            # Render as a clean key=value string for human-readable terminal output.
            # Swap to JSONRenderer for production / log aggregation.
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Return a structlog logger bound to the given module name.

    Usage:
        log = get_logger(__name__)
        log.info("tool_called", tool="web_search", query="OpenAI Ollama")
    """
    return structlog.get_logger(name)
