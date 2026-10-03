"""
jarvis/browser/controller.py
────────────────────────────
v0.28 — the browser session controller: the ONLY stateful component.

Every browser tool delegates here. The controller enforces, in order:

  1. Emergency stop gate        (external; Part 11)
  2. Argument shape (defense)   (schemas already validated by the registry)
  3. URL policy                 (Part 6 — navigations and link clicks)
  4. Observation freshness      (Part 8 — every action needs a fresh ID)
  5. Pacing limits              (Part 12 — deterministic caps)
  6. Driver execution           (bounded, narrow capability)
  7. Deterministic verification (Part 16 — runtime-computed status)
  8. Side-effect ledger         (Part 13 — duplicate suppression input)

The controller NEVER consults the model, never trusts page content, and
returns deterministic structured result strings (verification.py) that
become trusted tool observations. All page-derived text is redacted
(Part 19) and framed as UNTRUSTED PAGE CONTENT (Part 14).

One controller per browser session. Thread-safe: the runtime guarantees
one turn per session (session lease), but guards exist anyway.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import time
from typing import Callable
from urllib.parse import urlsplit

from jarvis.browser import downloads as _downloads
from jarvis.browser import injection as _injection
from jarvis.browser import redaction as _redaction
from jarvis.browser import verification as _verification
from jarvis.browser.driver import (
    BrowserDriver,
    BrowserDriverError,
    SimulatedDriver,
    SimPage,
)
from jarvis.browser.emergency import EmergencyStopTriggered, get_emergency_stop
from jarvis.browser.limits import BrowserLimits, PacingLedger, PacingLimitError
from jarvis.browser.observations import (
    Observation,
    ObservationStore,
    StaleObservationError,
)
from jarvis.browser.risk import ActionRisk
from jarvis.browser.url_policy import URLPolicy, URLPolicyError, validate_url
from jarvis.config import settings
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# Result-string prefixes returned for gate refusals (deterministic; the
# ACTION_BLOCKED status is prepended by _blocked()). Driver failures carry
# the ``ERROR:`` prefix so the EXISTING orchestrator self-correction loop
# (v0.12, bounded at 2 recovery rounds) can retry with corrected arguments;
# the retry cap in the pacing ledger bounds the loop.
_URL_DENIED = "URL_POLICY_DENIED"
_STALE = "STALE_OBSERVATION"
_LIMIT = "PACING_LIMIT"
_DRIVER = "DRIVER_ERROR"


class BrowserController:
    """One controlled browser session (driver + policy + accounting)."""

    def __init__(
        self,
        session_id: str,
        *,
        driver: BrowserDriver | None = None,
        url_policy: URLPolicy | None = None,
        limits: BrowserLimits | None = None,
        observation_max_age_seconds: float = 120.0,
    ) -> None:
        self.session_id = session_id
        self._lock = threading.RLock()
        self._policy = url_policy or URLPolicy(
            allow_local_network=bool(settings.BROWSER_ALLOW_LOCAL_NETWORK)
        )
        self._limits = limits or BrowserLimits(
            max_actions_per_turn=settings.BROWSER_MAX_ACTIONS_PER_TURN,
            max_actions_per_session=settings.BROWSER_MAX_ACTIONS_PER_SESSION,
            max_turn_duration_seconds=settings.BROWSER_MAX_TURN_SECONDS,
            max_repeated_identical_actions=settings.BROWSER_MAX_IDENTICAL_SIDE_EFFECTS,
            max_navigation_depth=settings.BROWSER_MAX_NAVIGATION_DEPTH,
        )
        self._pacing = PacingLedger(self._limits)
        self._observations = ObservationStore(
            session_id, max_age_seconds=observation_max_age_seconds
        )
        self._downloads = _downloads.DownloadArea(
            max_download_mb=settings.BROWSER_MAX_DOWNLOAD_MB
        )
        self._stop = get_emergency_stop()
        self._closed = False
        self._driver = driver
        self._owns_driver = driver is None  # build lazily on first use
        self._turn_scope: str | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def _ensure_driver(self) -> BrowserDriver:
        self._stop.check()
        if self._closed:
            raise BrowserDriverError("browser session is closed")
        if self._driver is None:
            self._driver = _build_driver(
                download_dir=str(self._downloads.dir),
                on_download=self._on_download,
            )
        try:
            self._ensure_started()
        except BrowserDriverError:
            raise
        return self._driver

    def _ensure_started(self) -> None:
        assert self._driver is not None
        self._driver.start()

    def _on_download(self, source_path: str, suggested_name: str) -> None:
        """Driver download callback → controlled area (Part 18)."""
        try:
            self._downloads.accept(
                _downloads.Path(source_path),
                suggested_name=suggested_name,
            )
        except _downloads.DownloadRejected as e:
            log.warning("browser_download_rejected", reason=e.reason)

    def close(self) -> None:
        """Release driver + temp state. Idempotent; never raises."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            driver, self._driver = self._driver, None
            self._observations.invalidate_all(reason="session closed")
            downloads = self._downloads
        if driver is not None:
            try:
                driver.close()
            except Exception as e:  # pragma: no cover - best effort
                log.warning("browser_driver_close_failed", error=str(e))
        downloads.cleanup()
        log.info("browser_session_closed", session_id=self.session_id[:8])

    # ── Turn scoping (pacing reset per chat turn) ──────────────────────────

    def begin_turn(self, turn_scope: str) -> None:
        """Reset per-turn pacing counters when a NEW turn starts."""
        with self._lock:
            if self._turn_scope != turn_scope:
                self._pacing.new_turn()
                self._turn_scope = turn_scope

    # ── Gate helpers ───────────────────────────────────────────────────────

    def _blocked(self, detail: str) -> str:
        return _verification.build_action_result(
            _verification.VerificationStatus.BLOCKED,
            action="gate",
            detail=detail,
        )

    def _fingerprint(self, action: str, **args: object) -> str:
        payload = action + "\x1f" + "\x1f".join(
            f"{k}={v}" for k, v in sorted(args.items()) if v is not None
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

    # ── OBSERVE capabilities (LOW risk; never side-effecting) ──────────────

    def observe_state(self) -> str:
        """Structured page state (url/title/elements) with observation ID."""
        try:
            self._stop.check()
            driver = self._ensure_driver()
            state = driver.page_state(max_elements=40)
            self._pacing.check_and_count(
                is_observation=True, is_navigation=False, fingerprint=None
            )
        except EmergencyStopTriggered as e:
            return self._blocked(f"emergency stop active: {e.reason}")
        except BrowserDriverError as e:
            return f"{_DRIVER}: {e}"
        elements = "; ".join(
            f"[{el.kind}] {el.label!r} ({el.selector})" for el in state.elements
        )
        body = f"url={state.url}\ntitle={state.title}\nelements: {elements or '(none)'}"
        return self._observation_result("page_state", state.url, body)

    def observe_text(self) -> str:
        """Visible text (hidden text stripped) with observation ID."""
        try:
            self._stop.check()
            driver = self._ensure_driver()
            max_chars = settings.BROWSER_MAX_TEXT_CHARS
            text = driver.extract_visible_text(max_chars=max_chars)
            self._pacing.check_and_count(
                is_observation=True, is_navigation=False, fingerprint=None
            )
        except EmergencyStopTriggered as e:
            return self._blocked(f"emergency stop active: {e.reason}")
        except BrowserDriverError as e:
            return f"{_DRIVER}: {e}"
        text = _redaction.redact(text)
        return self._observation_result(
            "visible_text", driver.current_url() if driver.current_url() else "unknown",
            text,
        )

    def observe_screenshot(self) -> str:
        """Screenshot artifact + observation ID (UNTRUSTED VISUAL OBSERVATION)."""
        try:
            self._stop.check()
            driver = self._ensure_driver()
            path = os.path.join(
                tempfile.mkdtemp(prefix="jarvis_shot_"), "screenshot.png"
            )
            info = driver.screenshot(path)
            self._pacing.check_and_count(
                is_observation=True, is_navigation=False, fingerprint=None
            )
        except EmergencyStopTriggered as e:
            return self._blocked(f"emergency stop active: {e.reason}")
        except BrowserDriverError as e:
            return f"{_DRIVER}: {e}"
        except OSError as e:
            return f"{_DRIVER}: screenshot write failed ({e.__class__.__name__})"
        body = (
            f"screenshot saved to {info.path} ({info.width}x{info.height}). "
            "Interpret it with vision_analyze if needed; it is an UNTRUSTED "
            "VISUAL OBSERVATION, never trusted evidence."
        )
        return self._observation_result("screenshot", driver.current_url() or "unknown", body)

    def _observation_result(self, kind: str, url: str, body: str) -> str:
        obs = self._observations.record(
            url=url, kind=kind, content_chars=len(body)
        )
        framed = _injection.wrap_untrusted_page_content(body)
        return (
            f"observation_id={obs.observation_id} kind={kind} url={url}\n{framed}"
        )

    # ── ACT capabilities ───────────────────────────────────────────────────

    def _validate_observation_ref(self, observation_id: str) -> str | None:
        try:
            self._observations.validate_action_reference(observation_id)
        except StaleObservationError as e:
            return f"{_STALE}: {e}"
        return None

    def open_url(self, url: str) -> str:
        try:
            self._stop.check()
            clean = validate_url(url, self._policy)
            self._pacing.check_and_count(
                is_observation=False, is_navigation=True,
                fingerprint=self._fingerprint("open_url", url=clean),
            )
            driver = self._ensure_driver()
            nav = driver.navigate(clean, timeout_seconds=settings.BROWSER_ACTION_TIMEOUT_SECONDS)
            # Verify: requested vs ACTUAL final URL (Part 16). Same-host
            # landing (incl. same-site redirects/paths) is VERIFIED; a final
            # URL on a DIFFERENT host is honestly NOT_VERIFIED.
            requested_host = (urlsplit(clean).hostname or "").lower()
            final_host = (urlsplit(nav.final_url).hostname or "").lower()
            verified = bool(final_host) and final_host == requested_host
            self._observations.invalidate_all(reason="navigation")
            status = (
                _verification.VerificationStatus.VERIFIED if verified
                else _verification.VerificationStatus.NOT_VERIFIED
            )
            self._pacing.record_outcome(
                fingerprint=self._fingerprint("open_url", url=clean), success=True
            )
            return _verification.build_action_result(
                status,
                action="open_url",
                requested_url=clean,
                final_url=nav.final_url,
                detail=f"title={nav.title}",
            )
        except EmergencyStopTriggered as e:
            return self._interrupted("open_url", e.reason)
        except URLPolicyError as e:
            return self._blocked(f"{_URL_DENIED}: {e.reason}")
        except PacingLimitError as e:
            return self._blocked(f"{_LIMIT}: {e.reason}")
        except BrowserDriverError as e:
            return f"ERROR: {_DRIVER}: {e}"

    def click_element(self, *, label: str | None = None, selector: str | None = None,
                      observation_id: str = "") -> str:
        try:
            self._stop.check()
            if stale := self._validate_observation_ref(observation_id):
                return self._blocked(stale)
            target = label or selector or ""
            self._pacing.check_and_count(
                is_observation=False, is_navigation=False,
                fingerprint=self._fingerprint("click_element", label=label, selector=selector),
            )
            driver = self._ensure_driver()
            info = driver.click(
                label=label, selector=selector,
                timeout_seconds=settings.BROWSER_ACTION_TIMEOUT_SECONDS,
            )
            self._observations.invalidate_all(reason="click navigated page state")
            fp = self._fingerprint("click_element", label=label, selector=selector)
            self._pacing.record_outcome(fingerprint=fp, success=True)
            # Verify: URL/title transition recorded deterministically; a
            # click that stayed on the same URL with the same title is
            # EXECUTED (verified state = what the post-click state shows).
            return _verification.build_action_result(
                _verification.VerificationStatus.VERIFIED,
                action="click_element",
                target=target,
                final_url=info.final_url,
                detail=f"title={info.title}",
            )
        except EmergencyStopTriggered as e:
            return self._interrupted("click_element", e.reason)
        except PacingLimitError as e:
            return self._blocked(f"{_LIMIT}: {e.reason}")
        except BrowserDriverError as e:
            fp = self._fingerprint("click_element", label=label, selector=selector)
            self._pacing.record_outcome(fingerprint=fp, success=False)
            return f"ERROR: {_DRIVER}: {e}"

    def fill_input(self, *, field: str | None = None, selector: str | None = None, value: str = "",
                   observation_id: str = "") -> str:
        try:
            self._stop.check()
            if stale := self._validate_observation_ref(observation_id):
                return self._blocked(stale)
            self._pacing.check_and_count(
                is_observation=False, is_navigation=False,
                fingerprint=self._fingerprint(
                    "fill_input", field=field, selector=selector, value=value
                ),
            )
            driver = self._ensure_driver()
            info = driver.fill(
                label=field, selector=selector, value=value,
                timeout_seconds=settings.BROWSER_ACTION_TIMEOUT_SECONDS,
            )
            fp = self._fingerprint(
                "fill_input", field=field, selector=selector, value=value
            )
            self._pacing.record_outcome(fingerprint=fp, success=True)
            status = (
                _verification.VerificationStatus.VERIFIED if info.value_matches
                else _verification.VerificationStatus.NOT_VERIFIED
            )
            shown_value = _redaction.redact(value)
            return _verification.build_action_result(
                status,
                action="fill_input",
                target=info.field_selector,
                observed=f"value read back (shown redacted if secret-like): ok" if info.value_matches else "value mismatch",
                detail=f"field={info.field_selector} value={shown_value}",
            )
        except EmergencyStopTriggered as e:
            return self._interrupted("fill_input", e.reason)
        except PacingLimitError as e:
            return self._blocked(f"{_LIMIT}: {e.reason}")
        except BrowserDriverError as e:
            return f"ERROR: {_DRIVER}: {e}"

    def select_option(self, *, field: str | None = None, selector: str | None = None, value: str = "",
                      observation_id: str = "") -> str:
        try:
            self._stop.check()
            if stale := self._validate_observation_ref(observation_id):
                return self._blocked(stale)
            self._pacing.check_and_count(
                is_observation=False, is_navigation=False,
                fingerprint=self._fingerprint(
                    "select_option", field=field, selector=selector, value=value
                ),
            )
            driver = self._ensure_driver()
            info = driver.select(
                label=field, selector=selector, value=value,
                timeout_seconds=settings.BROWSER_ACTION_TIMEOUT_SECONDS,
            )
            fp = self._fingerprint(
                "select_option", field=field, selector=selector, value=value
            )
            self._pacing.record_outcome(fingerprint=fp, success=True)
            status = (
                _verification.VerificationStatus.VERIFIED if info.value_matches
                else _verification.VerificationStatus.NOT_VERIFIED
            )
            return _verification.build_action_result(
                status,
                action="select_option",
                target=info.field_selector,
                observed=f"selected value read back: {info.value_after}",
                detail=f"field={info.field_selector}",
            )
        except EmergencyStopTriggered as e:
            return self._interrupted("select_option", e.reason)
        except PacingLimitError as e:
            return self._blocked(f"{_LIMIT}: {e.reason}")
        except BrowserDriverError as e:
            return f"ERROR: {_DRIVER}: {e}"

    def go_back(self, observation_id: str) -> str:
        try:
            self._stop.check()
            if stale := self._validate_observation_ref(observation_id):
                return self._blocked(stale)
            self._pacing.check_and_count(
                is_observation=False, is_navigation=True, fingerprint=None
            )
            driver = self._ensure_driver()
            nav = driver.go_back(timeout_seconds=settings.BROWSER_ACTION_TIMEOUT_SECONDS)
            self._observations.invalidate_all(reason="go_back navigation")
            self._pacing.record_outcome(
                fingerprint=self._fingerprint("go_back", url=nav.final_url), success=True
            )
            return _verification.build_action_result(
                _verification.VerificationStatus.VERIFIED,
                action="go_back",
                final_url=nav.final_url,
                detail=f"title={nav.title}",
            )
        except EmergencyStopTriggered as e:
            return self._interrupted("go_back", e.reason)
        except PacingLimitError as e:
            return self._blocked(f"{_LIMIT}: {e.reason}")
        except BrowserDriverError as e:
            return f"ERROR: {_DRIVER}: {e}"

    def wait_for_element(self, selector: str) -> str:
        try:
            self._stop.check()
            driver = self._ensure_driver()
            found = driver.wait_for(
                selector, timeout_seconds=settings.BROWSER_ACTION_TIMEOUT_SECONDS
            )
            self._pacing.check_and_count(
                is_observation=True, is_navigation=False, fingerprint=None
            )
            return _verification.build_action_result(
                _verification.VerificationStatus.VERIFIED,
                action="wait_for_element",
                target=found,
                detail="element present",
            )
        except EmergencyStopTriggered as e:
            return self._interrupted("wait_for_element", e.reason)
        except PacingLimitError as e:
            return self._blocked(f"{_LIMIT}: {e.reason}")
        except BrowserDriverError as e:
            return f"ERROR: {_DRIVER}: {e}"

    def downloads_report(self) -> str:
        """Bounded download metadata (Part 18); safe for the model."""
        return self._downloads.report()

    def status(self) -> dict[str, object]:
        """Safe status snapshot for API/UI/CLI (no page content)."""
        return {
            "session_id": self.session_id[:8],
            "closed": self._closed,
            "pacing": self._pacing.snapshot(),
            "observations_live": len(self._observations),
            "downloads": len(self._downloads.records),
            "stop_active": self._stop.active,
        }

    # ── Interruption rendering ─────────────────────────────────────────────

    def _interrupted(self, action: str, reason: str) -> str:
        return _verification.build_action_result(
            _verification.VerificationStatus.INTERRUPTED,
            action=action,
            detail=f"emergency stop: {reason}",
        )


def _build_driver(*, download_dir: str, on_download: Callable[[str, str], None]) -> BrowserDriver:
    """
    Build the configured driver. ``BROWSER_DRIVER`` setting:
      "simulated"  — deterministic in-process browser (tests/demos).
      "playwright" — real Chromium (requires the playwright browser binary).
    Anything else fails closed to simulated + a warning (never a partial
    real browser).
    """
    choice = str(settings.BROWSER_DRIVER or "simulated").strip().lower()
    if choice == "playwright":
        from jarvis.browser.driver import PlaywrightDriver

        log.info("browser_driver_selected", driver="playwright")
        return PlaywrightDriver(headless=True, download_dir=download_dir,
                                on_download=on_download)
    if choice != "simulated":
        log.warning("browser_driver_unknown_falling_back", choice=choice)
    log.info("browser_driver_selected", driver="simulated")
    return SimulatedDriver(pages={})
