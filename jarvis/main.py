"""
jarvis/main.py
───────────────
JARVIS CLI entry point.

This is the top-level assembly point:
  1. Set up logging.
  2. Instantiate all components (store, tools, guard, orchestrator).
  3. Start the interactive read-eval-print loop (or voice mode).

Special CLI commands:
  /help     — show available commands
  /history  — print conversation history for this session
  /tools    — list registered tools
  /voice    — enter voice mode (Whisper STT + Edge TTS)
  /new      — start a new session (clears context)
  /quit     — exit

Flags:
  jarvis --voice  — start directly in voice mode
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from jarvis import __version__
from jarvis.config import settings
from jarvis.core.orchestrator import Orchestrator
from jarvis.core.permissions import PermissionGuard
from jarvis.memory.session_store import SessionStore
from jarvis.memory.vector_store import get_vector_store
from jarvis.tools import (
    CalculatorTool,
    GetCurrentDatetimeTool,
    ListDirectoryTool,
    ReadFileTool,
    RecallFactsTool,
    RememberFactTool,
    ToolRegistry,
    WebSearchTool,
    WikipediaSummaryTool,
    WriteFileTool,
)
from jarvis.utils.logging import get_logger, setup_logging

# ── Setup ──────────────────────────────────────────────────────────────────────
setup_logging(settings.log_level)
log = get_logger(__name__)
console = Console()

_TOOL_BLURBS = {
    "get_current_datetime": "Current local date and time",
    "web_search": "Live web search (DuckDuckGo)",
    "wikipedia_summary": "Short Wikipedia topic summary",
    "read_file": "Read a file inside the sandbox",
    "list_directory": "List files in a sandboxed path",
    "calculator": "Evaluate a math expression",
    "remember_fact": "Save a long-term memory fact",
    "recall_facts": "Search long-term memory",
    "write_file": "Write or append to a file",
}


def _build_orchestrator() -> tuple[Orchestrator, SessionStore, ToolRegistry]:
    """Assemble all components and return orchestrator, store, and registry."""
    store = SessionStore()

    # Warm the long-term memory singleton (creates local Chroma path if needed).
    get_vector_store()

    registry = ToolRegistry()
    registry.register(GetCurrentDatetimeTool())
    registry.register(WebSearchTool())
    registry.register(WikipediaSummaryTool())
    registry.register(ReadFileTool())
    registry.register(ListDirectoryTool())
    registry.register(CalculatorTool())
    registry.register(RememberFactTool())
    registry.register(RecallFactsTool())
    registry.register(WriteFileTool())

    guard = PermissionGuard()
    orchestrator = Orchestrator(store, registry, guard)
    return orchestrator, store, registry


def _short_session_id(session_id: str) -> str:
    return session_id[:8]


def _print_welcome(tool_count: int) -> None:
    title = Text()
    title.append("J.A.R.V.I.S", style="bold cyan")
    title.append(f"  v{__version__}", style="dim")

    body = Text()
    body.append("Local AI assistant", style="white")
    body.append("  ·  ", style="dim")
    body.append(f"model {settings.ollama_model}", style="dim cyan")
    body.append("\n")
    body.append(f"{tool_count} tools ready", style="dim")
    body.append("  ·  ", style="dim")
    body.append("type ", style="dim")
    body.append("/help", style="bold")
    body.append(" for commands", style="dim")
    body.append("  ·  ", style="dim")
    body.append("/voice", style="bold")
    body.append(" for speech", style="dim")

    console.print()
    console.print(Panel(
        body,
        title=title,
        title_align="left",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()


def _print_help() -> None:
    table = Table(
        show_header=True,
        header_style="bold",
        box=None,
        pad_edge=False,
        expand=False,
    )
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")

    table.add_row("/help", "Show this command reference")
    table.add_row("/tools", "List tools the assistant can call")
    table.add_row("/history", "Show this session's conversation")
    table.add_row("/voice", "Enter voice mode (speak with JARVIS)")
    table.add_row("/new", "Start a fresh session (clears context)")
    table.add_row("/quit", "Exit JARVIS")

    console.print()
    console.print(Panel(
        table,
        title="[bold]Commands[/bold]",
        border_style="dim",
        padding=(1, 2),
    ))
    console.print(
        "  [dim]Anything else is sent to the assistant.[/dim]\n"
        "  [dim]Tip: start with[/dim] [white]jarvis --voice[/white] "
        "[dim]to enter voice mode immediately.[/dim]"
    )
    console.print()


def _print_tools(registry: ToolRegistry) -> None:
    table = Table(
        show_header=True,
        header_style="bold",
        box=None,
        pad_edge=False,
    )
    table.add_column("Tool", style="cyan", no_wrap=True)
    table.add_column("Purpose", style="white")

    for name in registry.list_tools():
        table.add_row(name, _TOOL_BLURBS.get(name, "Registered tool"))

    console.print()
    console.print(Panel(
        table,
        title="[bold]Registered tools[/bold]",
        border_style="dim",
        padding=(1, 2),
    ))
    console.print()


def _print_history(store: SessionStore, session_id: str) -> None:
    history = store.load_history(session_id)
    if not history:
        console.print("\n[dim]No messages in this session yet.[/dim]\n")
        return

    console.print()
    console.print(Panel(
        f"[dim]Session {_short_session_id(session_id)}…[/dim]  "
        f"[dim]{len(history)} message(s) in context window[/dim]",
        border_style="dim",
        padding=(0, 1),
    ))

    for msg in history:
        role = msg.get("role", "?").upper()
        content = msg.get("content") or ""

        if msg.get("tool_calls"):
            names = [
                tc.get("function", {}).get("name", "?")
                for tc in msg["tool_calls"]
            ]
            content = f"→ calling {', '.join(names)}"

        if role == "TOOL" and len(content) > 160:
            content = content[:157].rstrip() + "…"

        style = {
            "USER": "green",
            "ASSISTANT": "cyan",
            "TOOL": "yellow",
            "SYSTEM": "dim",
        }.get(role, "white")

        console.print(f"[{style}]{role:<9}[/{style}] {content}")

    console.print()


def _print_session_banner(session_id: str, *, fresh: bool = False) -> None:
    label = "New session" if fresh else "Session"
    console.print(
        f"[dim]{label}[/dim]  "
        f"[cyan]{_short_session_id(session_id)}[/cyan][dim]…[/dim]"
    )
    console.print()


def _start_voice_mode(orchestrator: Orchestrator, session_id: str) -> None:
    """Lazily load voice stack and run a voice session."""
    console.print("\n[cyan]Starting voice mode…[/cyan] "
                  "[dim](loading Whisper — first run may take a moment)[/dim]\n")
    try:
        from jarvis.voice.interface import VoiceInterface
        from jarvis.voice.stt import SpeechToText
        from jarvis.voice.tts import TextToSpeech

        stt = SpeechToText()
        tts = TextToSpeech()
        voice = VoiceInterface(orchestrator, stt, tts)
        voice.run_voice_session(session_id)
    except Exception as e:
        log.error("voice_mode_failed", error=str(e))
        console.print(f"[red]Voice mode failed:[/red] {e}")
        console.print(
            "[dim]Need ffmpeg on PATH, a working microphone, and "
            "network access for Edge TTS. See README → Voice Mode.[/dim]\n"
        )
    else:
        console.print("\n[dim]Returned to text mode.[/dim]\n")


def _chat_loop(
    orchestrator: Orchestrator,
    store: SessionStore,
    registry: ToolRegistry,
) -> None:
    """The main read-eval-print loop."""
    session_id = store.create_session()
    get_vector_store().set_session(session_id)
    _print_session_banner(session_id)

    while True:
        try:
            user_input = Prompt.ask("[bold green]You[/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue

        command = user_input.lower()

        if command in ("/quit", "/exit", "/q"):
            break

        if command == "/help":
            _print_help()
            continue

        if command == "/tools":
            _print_tools(registry)
            continue

        if command == "/history":
            _print_history(store, session_id)
            continue

        if command == "/voice":
            _start_voice_mode(orchestrator, session_id)
            continue

        if command == "/new":
            session_id = store.create_session()
            get_vector_store().set_session(session_id)
            console.print()
            _print_session_banner(session_id, fresh=True)
            continue

        with console.status("[cyan]Thinking…[/cyan]", spinner="dots"):
            try:
                response = orchestrator.chat(session_id, user_input)
            except Exception as e:
                log.error("orchestrator_error", error=str(e))
                console.print(f"\n[red]Error:[/red] {e}")
                console.print(
                    "[dim]Is Ollama running? Try:[/dim] [white]ollama serve[/white]\n"
                )
                continue

        console.print()
        console.print(Panel(
            Markdown(response),
            title="[bold cyan]JARVIS[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        ))
        console.print()

    console.print("\n[dim]Goodbye.[/dim]")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="jarvis",
        description="JARVIS — local personal AI assistant",
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Start in voice mode (Whisper STT + Edge TTS)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point — called by `jarvis` command or `python -m jarvis.main`."""
    args = _parse_args(argv)

    try:
        orchestrator, store, registry = _build_orchestrator()
    except Exception as e:
        console.print(f"[red]Startup error:[/red] {e}")
        sys.exit(1)

    _print_welcome(len(registry))

    try:
        if args.voice:
            session_id = store.create_session()
            get_vector_store().set_session(session_id)
            _print_session_banner(session_id)
            _start_voice_mode(orchestrator, session_id)
        else:
            _chat_loop(orchestrator, store, registry)
    finally:
        store.close()


if __name__ == "__main__":
    main()
