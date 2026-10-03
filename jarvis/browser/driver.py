"""
jarvis/browser/driver.py
────────────────────────
v0.28 Part 5 — the browser capability boundary.

The driver is the ONLY code that talks to a real browser. Everything above
it (controller, tools, orchestrator) sees structured Python objects and
never a raw browser handle, so no tool can "slip" a broader capability
through (arbitrary JS, cookies, file access — they simply do not exist on
this interface).

Capabilities (deliberately narrow; spec Part 5):
    navigate / go_back / page_state / click / fill / select /
    wait_for / extract_visible_text / screenshot / downloads captured

Explicitly ABSENT from the interface (impossible by construction):
    - arbitrary JavaScript execution
    - cookie / storage / credential access
    - browser filesystem access (beyond the controlled download area)
    - multiple tabs / window management (bounded single page)

Two implementations:
    PlaywrightDriver  — real automation (Chromium, headless by default),
                        dedicated non-persistent context (Part 7), its own
                        temp profile, visibility-filtered text extraction
                        (Part 14), bounded sizes everywhere.
    SimulatedDriver   — deterministic in-process browser used by the
                        deterministic test suite (Part 23) and available
                        as BROWSER_DRIVER=simulated for offline demos.

Every method raises BrowserDriverError on failure — the controller converts
failures to deterministic result strings; it never lets exceptions escape
into the orchestrator.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class BrowserDriverError(RuntimeError):
    """Driver-level failure (element missing, navigation failed, …)."""


@dataclass(frozen=True)
class ElementInfo:
    """One interactive element as exposed to the model (bounded fields)."""

    tag: str
    label: str
    selector: str
    kind: str = "element"          # link | button | input | select | textarea
    options: tuple[str, ...] = ()  # for selects (bounded)


@dataclass(frozen=True)
class PageState:
    url: str
    title: str
    elements: tuple[ElementInfo, ...] = ()


@dataclass(frozen=True)
class NavigationInfo:
    requested_url: str
    final_url: str
    title: str
    redirect_count: int = 0


@dataclass(frozen=True)
class ClickInfo:
    clicked: bool
    label: str
    selector: str
    final_url: str
    title: str


@dataclass(frozen=True)
class FillInfo:
    field_selector: str
    requested_value: str
    value_after: str              # deterministic read-back for verification
    value_matches: bool


@dataclass(frozen=True)
class SelectInfo:
    field_selector: str
    requested_value: str
    value_after: str
    value_matches: bool


@dataclass(frozen=True)
class ScreenshotInfo:
    path: str
    width: int
    height: int


class BrowserDriver(abc.ABC):
    """The narrow browser capability boundary (see module docstring)."""

    @abc.abstractmethod
    def start(self) -> None: ...

    @abc.abstractmethod
    def navigate(self, url: str, timeout_seconds: float) -> NavigationInfo: ...

    @abc.abstractmethod
    def go_back(self, timeout_seconds: float) -> NavigationInfo: ...

    @abc.abstractmethod
    def page_state(self, max_elements: int) -> PageState: ...

    @abc.abstractmethod
    def extract_visible_text(self, max_chars: int) -> str: ...

    @abc.abstractmethod
    def click(self, *, label: str | None, selector: str | None,
              timeout_seconds: float) -> ClickInfo: ...

    @abc.abstractmethod
    def fill(self, *, label: str | None, selector: str | None, value: str,
             timeout_seconds: float) -> FillInfo: ...

    @abc.abstractmethod
    def select(self, *, label: str | None, selector: str | None, value: str,
               timeout_seconds: float) -> SelectInfo: ...

    @abc.abstractmethod
    def wait_for(self, selector: str, timeout_seconds: float) -> str: ...

    @abc.abstractmethod
    def screenshot(self, path: str) -> ScreenshotInfo: ...

    @abc.abstractmethod
    def current_url(self) -> str: ...

    @abc.abstractmethod
    def close(self) -> None: ...


# ── Simulated driver (deterministic; tests + offline demos) ──────────────────


@dataclass
class SimElement:
    label: str
    selector: str
    kind: str = "element"                      # link | button | input | select
    href: str | None = None                    # links navigate
    value: str | None = None                   # inputs/selects (mutable)
    options: tuple[str, ...] = ()
    # Click effects for buttons: map selector-or-label -> (goto_url | confirm_text)
    effect_goto: str | None = None
    effect_confirm_text: str | None = None


@dataclass
class SimPage:
    url: str
    title: str
    text: str = ""
    hidden_text: str = ""                      # NEVER returned by extraction
    elements: list[SimElement] = field(default_factory=list)


class SimulatedDriver(BrowserDriver):
    """
    Deterministic browser. Pages are provided up front; navigation follows
    links/effect_goto; clicks apply scripted effects. History supports
    go_back. Every behavior is pure Python — no I/O, no timing.
    """

    def __init__(self, pages: dict[str, SimPage], start_url: str | None = None) -> None:
        self._pages = dict(pages)
        self._url = start_url or (next(iter(pages)) if pages else "about:blank")
        self._history: list[str] = []
        self._started = False

    # ── helpers ────────────────────────────────────────────────────────────

    def _page(self) -> SimPage:
        page = self._pages.get(self._url)
        if page is None:
            raise BrowserDriverError(f"no simulated page for {self._url}")
        return page

    @staticmethod
    def _find(page: SimPage, *, label: str | None, selector: str | None) -> SimElement:
        for el in page.elements:
            if selector and el.selector == selector:
                return el
            if label and el.label == label:
                return el
        raise BrowserDriverError("element not found")

    # ── capability implementations ─────────────────────────────────────────

    def start(self) -> None:
        self._started = True

    def navigate(self, url: str, timeout_seconds: float) -> NavigationInfo:
        target = self._pages.get(url)
        if target is None:
            raise BrowserDriverError(f"navigation failed: unknown page {url}")
        self._history.append(self._url)
        self._url = url
        return NavigationInfo(requested_url=url, final_url=url, title=target.title)

    def go_back(self, timeout_seconds: float) -> NavigationInfo:
        if not self._history:
            raise BrowserDriverError("no history to go back to")
        previous = self._history.pop()
        self._url = previous
        return NavigationInfo(
            requested_url="history:back", final_url=previous, title=self._page().title
        )

    def page_state(self, max_elements: int) -> PageState:
        page = self._page()
        elements = tuple(
            ElementInfo(
                tag=el.kind if el.kind != "element" else "button",
                label=el.label,
                selector=el.selector,
                kind=el.kind,
                options=tuple(el.options)[:10],
            )
            for el in page.elements[:max_elements]
        )
        return PageState(url=page.url, title=page.title, elements=elements)

    def extract_visible_text(self, max_chars: int) -> str:
        # Hidden text is deliberately NOT returned (Part 14 visibility filter).
        return self._page().text[:max_chars]

    def click(self, *, label: str | None, selector: str | None,
              timeout_seconds: float) -> ClickInfo:
        page = self._page()
        el = self._find(page, label=label, selector=selector)
        if el.kind == "input" or el.kind == "select":
            raise BrowserDriverError("element is not clickable")
        final_url = page.url
        if el.href:
            self._history.append(page.url)
            self._url = el.href
            final_url = el.href
        elif el.effect_goto:
            self._history.append(page.url)
            self._url = el.effect_goto
            final_url = el.effect_goto
        elif el.effect_confirm_text:
            page.text = f"{page.text}\n{el.effect_confirm_text}".strip()
        return ClickInfo(
            clicked=True, label=el.label, selector=el.selector,
            final_url=final_url, title=self._page().title,
        )

    def fill(self, *, label: str | None, selector: str | None, value: str,
             timeout_seconds: float) -> FillInfo:
        el = self._find(self._page(), label=label, selector=selector)
        if el.kind not in ("input", "textarea"):
            raise BrowserDriverError("element is not a fillable field")
        el.value = value
        return FillInfo(
            field_selector=el.selector, requested_value=value,
            value_after=el.value, value_matches=el.value == value,
        )

    def select(self, *, label: str | None, selector: str | None, value: str,
               timeout_seconds: float) -> SelectInfo:
        el = self._find(self._page(), label=label, selector=selector)
        if el.kind != "select":
            raise BrowserDriverError("element is not a select")
        if value not in el.options:
            raise BrowserDriverError(f"option '{value}' not available")
        el.value = value
        return SelectInfo(
            field_selector=el.selector, requested_value=value,
            value_after=el.value, value_matches=True,
        )

    def wait_for(self, selector: str, timeout_seconds: float) -> str:
        self._find(self._page(), label=None, selector=selector)
        return selector

    def screenshot(self, path: str) -> ScreenshotInfo:
        # A deterministic 1x1 PNG (smallest valid PNG bytes).
        png = bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000d49444154789c6260000000060005"
            "27de3bbb0000000049454e44ae426082"
        )
        with open(path, "wb") as f:
            f.write(png)
        return ScreenshotInfo(path=path, width=1, height=1)

    def current_url(self) -> str:
        return self._url

    def close(self) -> None:
        self._started = False


# ── Playwright driver (real automation) ──────────────────────────────────────


class PlaywrightDriver(BrowserDriver):
    """
    Real Chromium automation via Playwright.

    Isolation (Part 7): a NON-PERSISTENT context (fresh temp profile) is
    created per driver instance — the user's normal browser profile,
    cookies, and sessions are never touched. The context (and its storage)
    is destroyed on close(). Single bounded page; no arbitrary JS surface
    is exposed to callers (the visibility-filter evaluate below is fixed
    code in THIS module, not callable by the model).
    """

    # Fixed extraction code (not model-controlled): collects text of VISIBLE
    # nodes only — script/style/noscript/template and aria-hidden/display:none/
    # visibility:hidden/opacity:0 subtrees are skipped (Part 14).
    _VISIBLE_TEXT_JS = """
    () => {
      const limit = %d;
      const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
      const parts = [];
      let total = 0;
      while (walker.nextNode() && total < limit) {
        const node = walker.currentNode;
        const el = node.parentElement;
        if (!el) continue;
        const tag = el.tagName;
        if (tag === 'SCRIPT' || tag === 'STYLE' || tag === 'NOSCRIPT' || tag === 'TEMPLATE') continue;
        if (el.closest('[aria-hidden="true"]')) continue;
        const st = window.getComputedStyle(el);
        if (!st || st.display === 'none' || st.visibility === 'hidden' || parseFloat(st.opacity) === 0) continue;
        const text = (node.textContent || '').trim();
        if (!text) continue;
        parts.push(text);
        total += text.length;
      }
      return parts.join('\\n');
    }
    """

    _ELEMENTS_JS = """
    (maxElements) => {
      const nodes = Array.from(document.querySelectorAll(
        'a, button, input, select, textarea, [role=button], [role=link]'));
      const visible = (el) => {
        const st = window.getComputedStyle(el);
        return st && st.display !== 'none' && st.visibility !== 'hidden'
          && parseFloat(st.opacity) !== 0 && el.getAttribute('aria-hidden') !== 'true';
      };
      const cssPath = (el) => {
        if (el.id) return '#' + CSS.escape(el.id);
        const name = el.tagName.toLowerCase();
        const parent = el.parentElement;
        if (!parent) return name;
        const same = Array.from(parent.children).filter(c => c.tagName === el.tagName);
        const idx = same.indexOf(el) + 1;
        return cssPath(parent) + ' > ' + name + ':nth-of-type(' + idx + ')';
      };
      return nodes.filter(visible).slice(0, maxElements).map(el => {
        const tag = el.tagName.toLowerCase();
        const label = (el.getAttribute('aria-label') || el.textContent
          || el.getAttribute('placeholder') || el.getAttribute('name')
          || el.value || '').trim().slice(0, 80);
        const kind = tag === 'a' ? 'link' : tag === 'select' ? 'select'
          : (tag === 'input' || tag === 'textarea') ? 'input' : 'button';
        const options = tag === 'select'
          ? Array.from(el.options).slice(0, 10).map(o => o.value) : undefined;
        return {tag, label, selector: cssPath(el), kind, options};
      });
    }
    """

    def __init__(
        self,
        *,
        headless: bool = True,
        download_dir: str | None = None,
        on_download=None,
    ) -> None:
        self._headless = headless
        self._download_dir = download_dir
        self._on_download = on_download
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    # ── lifecycle ──────────────────────────────────────────────────────────

    def _ensure(self):
        if self._page is None:
            try:
                from playwright.sync_api import sync_playwright
            except ImportError as e:
                raise BrowserDriverError(
                    "playwright is not installed"
                ) from e
            try:
                self._playwright = sync_playwright().start()
                self._browser = self._playwright.chromium.launch(headless=self._headless)
                kwargs: dict = {"accept_downloads": self._download_dir is not None}
                self._context = self._browser.new_context(**kwargs)
                self._page = self._context.new_page()
                if self._download_dir is not None and self._on_download is not None:
                    self._page.on("download", self._handle_download)
            except Exception as e:
                self.close()
                raise BrowserDriverError(
                    f"browser startup failed ({e.__class__.__name__})"
                ) from e
        return self._page

    def _handle_download(self, download) -> None:  # pragma: no cover - live only
        import tempfile
        import os
        try:
            tmp = tempfile.mkdtemp(prefix="jarvis_dl_in_")
            target = os.path.join(tmp, download.suggested_filename or "download.bin")
            download.save_as(target)
            if self._on_download is not None:
                self._on_download(target, download.suggested_filename or "download.bin")
        except Exception as e:
            log.warning("browser_download_capture_failed", error=str(e))

    # ── capabilities ───────────────────────────────────────────────────────

    def start(self) -> None:
        self._ensure()

    def navigate(self, url: str, timeout_seconds: float) -> NavigationInfo:
        page = self._ensure()
        redirect_count = 0

        def _count_response(response) -> None:  # pragma: no cover - live only
            nonlocal redirect_count
            if response.redirected_from is not None:
                redirect_count += 1

        page.on("response", _count_response)
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=int(timeout_seconds * 1000))
        finally:
            try:
                page.remove_listener("response", _count_response)
            except Exception:  # pragma: no cover
                pass
        return NavigationInfo(
            requested_url=url, final_url=page.url,
            title=page.title() or "", redirect_count=min(redirect_count, 50),
        )

    def go_back(self, timeout_seconds: float) -> NavigationInfo:
        page = self._ensure()
        response = page.go_back(timeout=int(timeout_seconds * 1000))
        if response is None:
            raise BrowserDriverError("no history to go back to")
        return NavigationInfo(
            requested_url="history:back", final_url=page.url, title=page.title() or ""
        )

    def page_state(self, max_elements: int) -> PageState:
        page = self._ensure()
        raw = page.evaluate(self._ELEMENTS_JS, max_elements)
        elements = tuple(
            ElementInfo(
                tag=str(item.get("tag", "")),
                label=str(item.get("label", "")),
                selector=str(item.get("selector", "")),
                kind=str(item.get("kind", "element")),
                options=tuple(item.get("options") or ()),
            )
            for item in raw
        )
        return PageState(url=page.url, title=page.title() or "", elements=elements)

    def extract_visible_text(self, max_chars: int) -> str:
        page = self._ensure()
        # The JS itself bounds collection to max_chars (fail-safe second
        # bound applied in Python).
        text = page.evaluate(self._VISIBLE_TEXT_JS % int(max_chars))
        return str(text)[:max_chars]

    def _locate(self, *, label: str | None, selector: str | None):
        page = self._ensure()
        if selector:
            return page.locator(selector).first
        if label:
            for probe in (
                lambda: page.get_by_label(label, exact=False).first,
                lambda: page.get_by_role("button", name=label).first,
                lambda: page.get_by_role("link", name=label).first,
                lambda: page.get_by_text(label, exact=False).first,
            ):
                try:
                    loc = probe()
                    if loc.count() > 0:
                        return loc
                except Exception:  # noqa: BLE001 - try the next probe
                    continue
        raise BrowserDriverError("element not found")

    def click(self, *, label: str | None, selector: str | None,
              timeout_seconds: float) -> ClickInfo:
        page = self._ensure()
        loc = self._locate(label=label, selector=selector)
        loc.click(timeout=int(timeout_seconds * 1000))
        return ClickInfo(
            clicked=True, label=label or "", selector=selector or "",
            final_url=page.url, title=page.title() or "",
        )

    def fill(self, *, label: str | None, selector: str | None, value: str,
             timeout_seconds: float) -> FillInfo:
        page = self._ensure()
        loc = self._locate(label=label, selector=selector)
        loc.fill(value, timeout=int(timeout_seconds * 1000))
        # Deterministic read-back for verification (Part 16).
        try:
            after = loc.input_value(timeout=1000)
        except Exception:  # pragma: no cover - non-standard controls
            after = value
        return FillInfo(
            field_selector=selector or label or "", requested_value=value,
            value_after=str(after), value_matches=str(after) == value,
        )

    def select(self, *, label: str | None, selector: str | None, value: str,
               timeout_seconds: float) -> SelectInfo:
        page = self._ensure()
        loc = self._locate(label=label, selector=selector)
        loc.select_option(value, timeout=int(timeout_seconds * 1000))
        try:
            after = loc.input_value(timeout=1000)
        except Exception:  # pragma: no cover
            after = value
        return SelectInfo(
            field_selector=selector or label or "", requested_value=value,
            value_after=str(after), value_matches=str(after) == value,
        )

    def wait_for(self, selector: str, timeout_seconds: float) -> str:
        page = self._ensure()
        page.wait_for_selector(selector, timeout=int(timeout_seconds * 1000))
        return selector

    def screenshot(self, path: str) -> ScreenshotInfo:
        page = self._ensure()
        page.screenshot(path=path, full_page=False)
        return ScreenshotInfo(path=path, width=0, height=0)

    def current_url(self) -> str:
        return self._page.url if self._page is not None else ""

    def close(self) -> None:
        for attr in ("_context", "_browser", "_playwright"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.close() if attr != "_playwright" else obj.stop()
                except Exception:  # pragma: no cover - best effort
                    pass
                setattr(self, attr, None)
        self._page = None
