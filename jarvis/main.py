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
  /stop     — HUMAN-ONLY browser emergency stop (interrupts browser actions)
  /resume   — clear the emergency stop (operator action)
  /browser  — show safe-browser status (posture only, never page content)
  /quit     — exit

Ctrl+X during a running turn (while JARVIS is processing) also triggers the
browser emergency stop (Windows; uses the stdlib msvcrt console reader —
no new dependency). At the input prompt, use /stop instead.

Flags:
  jarvis --voice  — start directly in voice mode
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

from jarvis.core.orchestrator import Orchestrator
from jarvis.memory.session_store import SessionStore
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from jarvis import __version__
from jarvis.browser.emergency import get_emergency_stop
from jarvis.browser.registry import get_browser_registry
from jarvis.config import settings
from jarvis.runtime import JarvisRuntime, build_runtime
from jarvis.tools.registry import ToolRegistry
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
    "vision_analyze": "Analyze an image using Vision LLM",
    "web_scrape": "Deep scrape a webpage via Playwright",
    # v0.29 integration tools (present only when ENABLE_INTEGRATIONS=true)
    "calendar_list_events": "List upcoming events on a connected calendar (read-only)",
    "calendar_get_event": "Get one event's details (read-only)",
    "calendar_create_event": "Create a calendar event (requires your confirmation)",
    "calendar_update_event": "Update/move a calendar event (requires confirmation)",
    "calendar_delete_event": "Delete/cancel a calendar event (HIGH risk; always confirmed)",
    "task_list": "List tasks on a connected account (read-only)",
    "task_create": "Create a task (requires your confirmation)",
    "task_complete": "Mark a task done (requires your confirmation)",
}


def _short_session_id(session_id: str) -> str:
    return session_id[:8]


def _print_browser_status() -> None:
    """v0.28: bounded, safe browser posture (never page content/URLs)."""
    from jarvis.browser.emergency import get_emergency_stop as _stop

    stop = _stop()
    registry = get_browser_registry()
    s = stop.status()
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("Aspect", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    table.add_row("Tool surface", "enabled" if settings.ENABLE_BROWSER_CONTROL else "disabled (ENABLE_BROWSER_CONTROL=false)")
    table.add_row("Driver", str(settings.BROWSER_DRIVER))
    table.add_row("Emergency stop", ("ACTIVE — " + str(s["reason"])) if s["active"] else "clear")
    table.add_row("Stop token", str(s["token"]))
    table.add_row("Open sessions", f"{registry.open_count()} / max {settings.BROWSER_MAX_SESSIONS}")
    console.print()
    console.print(Panel(table, title="[bold]Browser control[/bold]", border_style="dim", padding=(1, 2)))
    console.print(
        "  [dim]/stop triggers the emergency stop; /resume clears it. "
        "The model cannot trigger or clear it.[/dim]\n"
    )


def _print_integration_status(runtime: JarvisRuntime) -> None:
    """
    v0.29: bounded, safe integration posture — providers, connected
    accounts, scopes, auth state. NEVER credentials (they are structurally
    absent from the manager's public projection).
    """
    manager = getattr(runtime, "integration_manager", None)
    if manager is None:
        console.print(
            "[dim]Integrations are disabled "
            "(set ENABLE_INTEGRATIONS=true to enable).[/dim]"
        )
        return
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("Provider", style="cyan", no_wrap=True)
    table.add_column("Account", style="white")
    table.add_column("Account ID", style="dim")
    table.add_column("Auth", style="white")
    table.add_column("OAuth", style="white")
    table.add_column("Scopes", style="white")
    table.add_column("Last verified", style="dim")
    rows = 0
    for name, provider in manager.providers().items():
        accounts = manager.list_accounts(name)
        if not accounts:
            table.add_row(name, "(no accounts)", "—", "—", "—", "—", "—")
            rows += 1
        for a in accounts:
            table.add_row(
                name,
                a.display_label,
                a.account_id,
                a.auth_state.value,
                a.authorization_status or "—",
                ", ".join(sorted(a.scopes)) or "(none)",
                (a.last_verified_at or "never")[:19],
            )
            rows += 1
    if not rows:
        console.print("[dim]No integration providers registered.[/dim]")
        return
    console.print()
    console.print(Panel(table, title="[bold]Integrations[/bold]", border_style="dim", padding=(1, 2)))
    console.print(
        "  [dim]Connect: /integration-connect (OAuth) — Re-authenticate: "
        "/integration-reauthenticate — Disconnect: /integration-disconnect. "
        "Tokens and credentials are never displayed. External writes always "
        "require your explicit confirmation in chat.[/dim]\n"
    )


def _integration_connect_interactive(runtime: JarvisRuntime) -> None:
    """v0.29 (Part 19): explicit, USER-controlled connect flow.

    The credential is typed by the operator here — it NEVER passes through
    the chat model, is never stored in session history, and is never
    displayed afterward (Part 4)."""
    manager = getattr(runtime, "integration_manager", None)
    if manager is None:
        console.print("[dim]Integrations are disabled (ENABLE_INTEGRATIONS=false).[/dim]")
        return
    providers = manager.providers()
    if not providers:
        console.print("[dim]No integration providers registered.[/dim]")
        return
    console.print("Providers: " + ", ".join(sorted(providers)))
    provider = Prompt.ask("Provider", choices=sorted(providers)).strip()
    label = Prompt.ask("Account label (e.g. 'Personal')").strip()
    try:
        menu = sorted(manager.capabilities(provider).grantable_scopes)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        return
    console.print("Available scopes: " + ", ".join(menu))
    scopes_raw = Prompt.ask("Scopes to grant (comma-separated)").strip()
    scopes = {s.strip() for s in scopes_raw.split(",") if s.strip()}
    supports_oauth = bool(getattr(providers.get(provider), "supports_oauth", False))
    if supports_oauth and Prompt.ask(
        "Authorize with OAuth (recommended) instead of a local credential?",
        choices=["y", "n"],
        default="y",
    ) == "y":
        _integration_oauth_connect(runtime, provider, label, scopes)
        return
    credential = Prompt.ask(
        "Credential (blank = generate a local development token)",
        password=True,
        default="",
    ).strip() or None
    try:
        account = manager.connect(
            provider=provider,
            display_label=label,
            credential=credential,
            scopes=scopes,
        )
    except Exception as e:
        console.print(f"[red]Connect failed:[/red] {e}")
        return
    console.print(
        f"[green]Connected[/green] {account.provider}:{account.display_label} "
        f"({account.account_id}) — auth: {account.auth_state.value}, "
        f"scopes: {', '.join(sorted(account.scopes))}."
    )


def _integration_oauth_connect(
    runtime: JarvisRuntime,
    provider: str,
    label: str,
    scopes: set[str],
) -> None:
    """
    v0.30: user-controlled OAuth authorization from the CLI.

    The authorization URL is opened by the USER (it is never shown to the
    chat model). A real provider redirects the browser to the running API
    server's callback endpoint; for the bundled LOCAL simulated provider the
    operator can simulate consent inline. The CLI never prints codes, tokens,
    or the raw state value a second time — it polls the bounded flow status.
    """
    import uuid as _uuid

    manager = getattr(runtime, "integration_manager", None)
    if manager is None:
        console.print("[dim]Integrations are disabled (ENABLE_INTEGRATIONS=false).[/dim]")
        return
    session_id = f"cli-{_uuid.uuid4().hex[:8]}"
    try:
        start = manager.begin_authorization(
            provider=provider,
            session_id=session_id,
            display_label=label,
            scopes=scopes,
        )
    except Exception as e:  # noqa: BLE001 — bounded, safe message
        console.print(f"[red]Authorization start failed:[/red] {str(e)[:200]}")
        return
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_row("Provider", provider)
    table.add_row("Account label", start.get("display_label", label))
    table.add_row("Scopes", ", ".join(start.get("scopes", sorted(scopes))))
    table.add_row("Expires in", f"{start.get('expires_in', '?')}s")
    console.print()
    console.print(Panel(table, title="[bold]OAuth authorization[/bold]", border_style="dim", padding=(1, 2)))
    console.print("  [dim]Open this URL in a browser and approve:[/dim]")
    console.print(f"  {start.get('authorization_url', '')}\n")
    console.print(
        "  [dim]The provider must redirect back to the running API server's "
        "callback endpoint (see OAUTH_REDIRECT_BASE_URL).[/dim]\n"
    )
    if Prompt.ask(
        "Simulate the local development provider's consent now?",
        choices=["y", "n"],
        default="y",
    ) == "y":
        from jarvis.integrations.oauth import local_simulate_consent

        try:
            consent = local_simulate_consent(start["authorization_url"])
            if not consent.granted:
                console.print("[yellow]Consent denied (simulated) — no account created.[/yellow]")
                return
            account = manager.handle_callback(
                provider=provider,
                code=consent.code,
                state=consent.state,
                session_id=session_id,
                redirect_uri=consent.redirect_uri,
            )
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Callback failed:[/red] {str(e)[:200]}")
            return
        console.print(
            f"[green]Authorized[/green] {account.provider}:{account.display_label} "
            f"({account.account_id}) — status {account.authorization_status}, "
            f"scopes: {', '.join(sorted(account.scopes))}"
        )
        return
    console.print("[dim]Waiting for the callback (Ctrl+C to stop)…[/dim]")
    import time as _time

    for _ in range(60):
        _time.sleep(2)
        outcome = manager.authorization_flow_outcome(provider, session_id)
        if outcome.get("status") not in ("AUTHORIZING", "NO_FLOW"):
            console.print(f"Authorization result: {outcome.get('status')}")
            return
    console.print("[yellow]Timed out waiting for the authorization callback.[/yellow]")


def _integration_reauthenticate_interactive(runtime: JarvisRuntime) -> None:
    """v0.30: re-run authorization for an existing OAuth account (rotation)."""
    manager = getattr(runtime, "integration_manager", None)
    if manager is None:
        console.print("[dim]Integrations are disabled (ENABLE_INTEGRATIONS=false).[/dim]")
        return
    accounts = [a for a in manager.list_accounts() if a.is_oauth_account]
    if not accounts:
        console.print("[dim]No OAuth accounts to re-authenticate.[/dim]")
        return
    for a in accounts:
        console.print(f"  {a.account_id}  {a.provider}:{a.display_label} ({a.authorization_status})")
    account_id = Prompt.ask("Account ID to re-authenticate").strip()
    account = next((a for a in accounts if a.account_id == account_id), None)
    if account is None:
        console.print("[red]Unknown account ID.[/red]")
        return
    _integration_oauth_connect(
        runtime, account.provider, account.display_label, set(account.scopes)
    )


def _integration_disconnect_interactive(runtime: JarvisRuntime) -> None:
    """v0.29: explicit disconnect (removes the stored credential locally)."""
    manager = getattr(runtime, "integration_manager", None)
    if manager is None:
        console.print("[dim]Integrations are disabled (ENABLE_INTEGRATIONS=false).[/dim]")
        return
    accounts = manager.list_accounts()
    if not accounts:
        console.print("[dim]No connected accounts.[/dim]")
        return
    for a in accounts:
        console.print(f"  {a.account_id}  {a.provider}:{a.display_label} ({a.auth_state.value})")
    account_id = Prompt.ask("Account ID to disconnect").strip()
    if not any(a.account_id == account_id for a in accounts):
        console.print("[red]Unknown account ID.[/red]")
        return
    if not Prompt.ask("Disconnect and locally revoke this account?", choices=["y", "n"], default="n") == "y":
        console.print("[dim]Cancelled.[/dim]")
        return
    removed = manager.disconnect(account_id)
    console.print(
        "[green]Disconnected.[/green] Stored credentials were removed from "
        "the local database."
        if removed
        else "[yellow]Account was already absent.[/yellow]"
    )


def _start_estop_key_watcher() -> threading.Event:
    """
    v0.28 (Part 22): Ctrl+X during a RUNNING TURN triggers the browser
    emergency stop. Windows console via stdlib ``msvcrt`` — no new
    dependency. Human-only by construction: the model has no tool that can
    reach this code. The daemon thread exits with the process.
    """
    fired = threading.Event()

    def _watch() -> None:  # pragma: no cover - interactive console I/O
        try:
            import msvcrt
        except ImportError:
            return  # non-Windows: /stop remains the manual path
        while True:
            if msvcrt.kbhit():
                key = msvcrt.getwch()
                # Ctrl+X arrives as canonical '\x18' (or 'X' after a bare
                # Ctrl key press on some terminals — treat both as stop).
                if key in ("\x18",):
                    fired.set()
                    get_emergency_stop().trigger(
                        reason="Ctrl+X pressed during running turn"
                    )
                    console.print(
                        "\n[bold red]🛑 EMERGENCY STOP triggered (Ctrl+X).[/bold red] "
                        "Browser actions will be interrupted at the next gate. "
                        "Use /resume to clear."
                    )
            time.sleep(0.05)

    threading.Thread(target=_watch, name="jarvis-estop-key", daemon=True).start()
    return fired


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
    table.add_row(
        "/refresh",
        "Toggle refresh mode (re-fetch cached results this session)",
    )
    table.add_row("/confirm", "Confirm a pending high-risk action")
    table.add_row("/deny", "Deny a pending high-risk action")
    table.add_row("/stop", "EMERGENCY STOP: interrupt browser actions now (human-only)")
    table.add_row("/resume", "Clear the emergency stop (operator action)")
    table.add_row("/browser", "Show safe-browser status (posture only)")
    table.add_row("/integrations", "List integration providers + connected accounts")
    table.add_row("/integration-status", "Show integration auth states (same view)")
    table.add_row("/integration-connect", "Connect via OAuth (recommended) or a local credential — explicit, user-controlled")
    table.add_row("/integration-reauthenticate", "Re-run OAuth authorization for an existing account")
    table.add_row("/integration-disconnect", "Disconnect an account (provider revocation + local credential removal)")
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


def _start_voice_mode(orchestrator: Orchestrator, session_id: str, *, push_to_talk: bool = False, tts_enabled: bool = True) -> None:
    """Lazily load the voice stack and run a voice session (either mode)."""
    mode_label = "push-to-talk voice mode" if push_to_talk else "voice mode"
    console.print(f"\n[cyan]Starting {mode_label}…[/cyan] "
                  "[dim](loading Whisper — first run may take a moment)[/dim]\n")
    try:
        from jarvis.voice.interface import VoiceInterface
        from jarvis.voice.stt import SpeechToText
        from jarvis.voice.tts import TextToSpeech

        stt = SpeechToText()
        tts = TextToSpeech(enabled=tts_enabled)
        voice = VoiceInterface(orchestrator, stt, tts)
        if push_to_talk:
            voice.run_push_to_talk(session_id)
        else:
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


def _run_single_multimodal_turn(
    runtime: JarvisRuntime,
    *,
    audio_path: str | None,
    image_path: str | None,
    text: str,
    refresh: bool,
) -> None:
    """v0.27: one-shot multimodal turn from CLI (audio file and/or image)."""
    from pathlib import Path as _Path

    from jarvis.multimodal.models import (
        Attachment,
        MultimodalRequest,
        MultimodalValidationError,
    )
    from jarvis.multimodal.service import MultimodalService
    from jarvis.voice.stt import SpeechToText

    session_id = runtime.start_session()
    _print_session_banner(session_id)
    modality = "text"
    image = None
    text_final = text

    try:
        if audio_path:
            data = _Path(audio_path).read_bytes()
            from jarvis.multimodal.models import validate_audio_bytes

            mime = validate_audio_bytes(data)
            stt = SpeechToText()
            if not stt.is_ready:
                console.print("[red]Whisper failed to load — cannot transcribe audio.[/red]")
                return
            text_final = stt.transcribe_bytes(data, mime)
            if text_final.startswith("ERROR:") or not text_final:
                console.print(f"[red]{text_final or 'No speech detected.'}[/red]")
                return
            console.print(f"[dim]Transcribed:[/dim] {text_final}")
            modality = "audio"

        if image_path:
            data = _Path(image_path).read_bytes()
            image = Attachment.from_upload(data)   # content-sniffed, bounded
            modality = "image+text" if text_final else "image"
            if not text_final:
                text_final = "Describe this image in detail."
    except MultimodalValidationError as e:
        console.print(f"[red]Invalid upload:[/red] {e}")
        return
    except FileNotFoundError as e:
        console.print(f"[red]File not found:[/red] {e.filename}")
        return

    request = MultimodalRequest(
        text=text_final,
        session_id=session_id,
        modality=modality,
        image=image,
        refresh=refresh,
    )
    try:
        response = MultimodalService(runtime.orchestrator).run(request)
    except Exception as e:
        console.print(f"[red]Turn failed:[/red] {e}")
        return
    finally:
        if image is not None:
            image.cleanup()

    console.print("\n[JARVIS]")
    from rich.markdown import Markdown

    console.print(Markdown(response))


def _chat_loop(runtime: JarvisRuntime, refresh: bool = False) -> None:
    """The main read-eval-print loop.

    ``refresh`` (v0.25 Part D3) is the session-wide default for programmatic
    cache refresh: when True, every turn sets refresh=True on the orchestrator
    (bypasses ELIGIBLE cache entries for that turn; permissions, validation
    and confirmation are unaffected). Toggled per-session with /refresh.
    """
    session_id = runtime.start_session()
    _print_session_banner(session_id)
    # v0.28: Ctrl+X watcher — human-only emergency stop during running turns.
    _start_estop_key_watcher()
    if refresh:
        console.print(
            "[dim]Refresh mode ON for this session: cached results are "
            "re-fetched (use /refresh to toggle).[/dim]"
        )

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
            _print_tools(runtime.registry)
            continue

        if command == "/history":
            _print_history(runtime.store, session_id)
            continue

        if command == "/voice":
            _start_voice_mode(runtime.orchestrator, session_id)
            continue

        if command == "/new":
            session_id = runtime.start_session()
            console.print()
            _print_session_banner(session_id, fresh=True)
            continue

        if command == "/refresh":
            refresh = not refresh
            console.print(
                f"[dim]Refresh mode {'ON' if refresh else 'OFF'}: "
                + (
                    "cached results will be re-fetched this session."
                    if refresh
                    else "normal cache policy restored."
                )
                + "[/dim]"
            )
            continue
            
        if command == "/stop":
            stop = get_emergency_stop()
            token = stop.trigger(reason="CLI /stop command")
            closed = get_browser_registry().close_all()
            console.print(
                f"[bold red]🛑 EMERGENCY STOP #{token} active.[/bold red] "
                f"In-flight browser actions are interrupted at the next "
                f"runtime gate; {closed} browser session(s) closed. "
                "Use /resume to clear it."
            )
            continue

        if command == "/resume":
            was_active = get_emergency_stop().reset()
            console.print(
                "[dim]Emergency stop cleared.[/dim]"
                if was_active
                else "[dim]No emergency stop was active.[/dim]"
            )
            continue

        if command == "/browser":
            _print_browser_status()
            continue

        # ── v0.29: integration management (operator-side; the agent itself
        # still operates integrations only through the normal tool path) ──
        if command in ("/integrations", "/integration-status"):
            _print_integration_status(runtime)
            continue
        if command == "/integration-connect":
            _integration_connect_interactive(runtime)

        if command == "/integration-reauthenticate":
            _integration_reauthenticate_interactive(runtime)
            continue
        if command == "/integration-disconnect":
            _integration_disconnect_interactive(runtime)
            continue

        if command == "/confirm" or command == "/deny":
            with console.status("[cyan]Processing...[/cyan]", spinner="dots"):
                try:
                    response = runtime.handle_confirmation(session_id, command == "/confirm")
                except Exception as e:
                    console.print(f"\n[red]Error:[/red] {e}")
                    continue
            console.print()
            console.print(Panel(
                Markdown(response),
                title="[bold cyan]JARVIS[/bold cyan]",
                border_style="cyan",
                padding=(1, 2),
            ))
            console.print()
            continue

        with console.status("[cyan]Thinking…[/cyan]", spinner="dots"):
            try:
                response = runtime.chat(session_id, user_input, refresh=refresh)
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
    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Start with refresh mode ON: bypass eligible cache entries each "
            "turn (re-fetch fresh results; permissions unchanged). Toggle "
            "mid-session with /refresh."
        ),
    )
    # ── v0.27 multimodal options ────────────────────────────────────────
    parser.add_argument(
        "--voice-ptt",
        action="store_true",
        help=(
            "Start in push-to-talk voice mode (explicit ENTER-per-turn; no "
            "background listening). Speaks only the final grounded response."
        ),
    )
    parser.add_argument(
        "--audio-file",
        metavar="PATH",
        default=None,
        help="Transcribe an audio file (WAV/MP3) with local Whisper and send it as one turn.",
    )
    parser.add_argument(
        "--image",
        metavar="PATH",
        default=None,
        help="Attach an image (JPEG/PNG/WebP/GIF) to this turn ('What's in this image?').",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable spoken responses in voice modes (text only).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point — called by `jarvis` command or `python -m jarvis.main`."""
    args = _parse_args(argv)

    try:
        runtime = build_runtime()
    except Exception as e:
        console.print(f"[red]Startup error:[/red] {e}")
        sys.exit(1)

    _print_welcome(len(runtime.registry))

    try:
        if args.audio_file or args.image:
            # v0.27 one-shot multimodal turn (non-interactive).
            prompt = args.message if hasattr(args, "message") else ""
            _run_single_multimodal_turn(
                runtime,
                audio_path=args.audio_file,
                image_path=args.image,
                text=prompt or "",
                refresh=bool(args.refresh),
            )
        elif args.voice or args.voice_ptt:
            session_id = runtime.start_session()
            _print_session_banner(session_id)
            _start_voice_mode(
                runtime.orchestrator,
                session_id,
                push_to_talk=bool(args.voice_ptt),
                tts_enabled=not args.no_tts,
            )
        else:
            _chat_loop(runtime, refresh=bool(args.refresh))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
