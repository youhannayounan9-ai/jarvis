"""
jarvis/tools/web_scrape.py
──────────────────────────
Tool: web_scrape

Uses Playwright to load a webpage fully (including JS) and extracts its text.
"""

from jarvis.tools.base import BaseTool
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class WebScrapeTool(BaseTool):
    name = "web_scrape"
    description = (
        "CRITICAL: Use this to read the full text content of a specific webpage URL. "
        "Better than web_search for deep reading."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL of the webpage to scrape.",
            },
        },
        "required": ["url"],
    }
    risk_level = "NETWORK"
    timeout_seconds = 30.0

    def run(self, url: str, **kwargs) -> str:
        try:
            from playwright.sync_api import sync_playwright
            
            log.info("web_scrape_start", url=url)
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, timeout=20000)
                text = page.inner_text("body")
                browser.close()
                
                # Truncate text to fit context window
                if len(text) > 4000:
                    text = text[:4000] + "\n...[TRUNCATED]"
                
                log.info("web_scrape_success", url=url, bytes=len(text))
                return text
        except Exception as e:
            log.error("web_scrape_error", error=str(e), url=url)
            return f"ERROR: Failed to scrape {url}. {e}"
