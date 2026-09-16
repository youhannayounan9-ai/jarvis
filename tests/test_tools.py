"""
tests/test_tools.py
────────────────────
Unit tests for all v0.1 tools.

These tests do NOT call Ollama or any external service.
Each tool's run() method is called directly.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Tool imports ───────────────────────────────────────────────────────────────
from jarvis.tools.calculator import CalculatorTool
from jarvis.tools.datetime_tool import GetCurrentDatetimeTool
from jarvis.tools.directory_lister import ListDirectoryTool
from jarvis.tools.file_reader import ReadFileTool
from jarvis.tools.registry import ToolRegistry
from jarvis.tools.web_search import WebSearchTool, _format_results, _normalize_results
from jarvis.tools.wikipedia_summary import WikipediaSummaryTool


# ══════════════════════════════════════════════════════════════════════════════
# get_current_datetime
# ══════════════════════════════════════════════════════════════════════════════

class TestGetCurrentDatetimeTool:
    def setup_method(self):
        self.tool = GetCurrentDatetimeTool()

    def test_name(self):
        assert self.tool.name == "get_current_datetime"

    def test_returns_string(self):
        result = self.tool.run()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_result_contains_datetime_keywords(self):
        result = self.tool.run()
        # Should contain something that looks like a time (colon between digits)
        assert ":" in result

    def test_schema_has_no_required_params(self):
        schema = self.tool.to_openai_schema()
        assert schema["function"]["name"] == "get_current_datetime"
        assert schema["function"]["parameters"]["required"] == []


# ══════════════════════════════════════════════════════════════════════════════
# read_file
# ══════════════════════════════════════════════════════════════════════════════

class TestReadFileTool:
    def setup_method(self):
        self.tool = ReadFileTool()

    def test_name(self):
        assert self.tool.name == "read_file"

    def test_reads_existing_file(self, sample_text_file: Path, temp_dir: Path, monkeypatch):
        """Tool should read file contents when path is within allowed dir."""
        monkeypatch.setenv("FILE_READER_ALLOWED_DIR", str(temp_dir))
        # Re-import settings to pick up the monkeypatched env
        import importlib
        import jarvis.config
        importlib.reload(jarvis.config)
        import jarvis.tools.file_reader
        importlib.reload(jarvis.tools.file_reader)
        from jarvis.tools.file_reader import ReadFileTool as FreshTool
        tool = FreshTool()
        result = tool.run(path=str(sample_text_file))
        assert "Hello from JARVIS test file!" in result
        assert "ERROR" not in result

    def test_blocks_path_traversal(self, tmp_path: Path, monkeypatch):
        """Tool must refuse paths outside allowed_dir."""
        monkeypatch.setenv("FILE_READER_ALLOWED_DIR", str(tmp_path))
        import importlib, jarvis.config, jarvis.tools.file_reader
        importlib.reload(jarvis.config)
        importlib.reload(jarvis.tools.file_reader)
        from jarvis.tools.file_reader import ReadFileTool as FreshTool
        tool = FreshTool()
        result = tool.run(path="/etc/passwd")
        assert "ERROR" in result
        assert "denied" in result.lower() or "outside" in result.lower()

    def test_file_not_found(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("FILE_READER_ALLOWED_DIR", str(tmp_path))
        import importlib, jarvis.config, jarvis.tools.file_reader
        importlib.reload(jarvis.config)
        importlib.reload(jarvis.tools.file_reader)
        from jarvis.tools.file_reader import ReadFileTool as FreshTool
        tool = FreshTool()
        result = tool.run(path=str(tmp_path / "nonexistent.txt"))
        assert "ERROR" in result
        assert "not found" in result.lower()


# ══════════════════════════════════════════════════════════════════════════════
# list_directory
# ══════════════════════════════════════════════════════════════════════════════

class TestListDirectoryTool:
    def setup_method(self):
        self.tool = ListDirectoryTool()

    def test_name(self):
        assert self.tool.name == "list_directory"

    def test_lists_existing_directory(self, temp_dir: Path, monkeypatch):
        """Tool should list contents when path is within allowed dir."""
        monkeypatch.setenv("FILE_READER_ALLOWED_DIR", str(temp_dir))
        import importlib, jarvis.config, jarvis.tools.directory_lister
        importlib.reload(jarvis.config)
        importlib.reload(jarvis.tools.directory_lister)
        from jarvis.tools.directory_lister import ListDirectoryTool as FreshTool
        
        # Create a test folder and file
        (temp_dir / "subfolder").mkdir()
        (temp_dir / "test_file.txt").write_text("hello")

        tool = FreshTool()
        result = tool.run(path=str(temp_dir))
        assert "subfolder/" in result
        assert "test_file.txt" in result
        assert "ERROR" not in result

    def test_blocks_path_traversal(self, tmp_path: Path, monkeypatch):
        """Tool must refuse paths outside allowed_dir."""
        monkeypatch.setenv("FILE_READER_ALLOWED_DIR", str(tmp_path))
        import importlib, jarvis.config, jarvis.tools.directory_lister
        importlib.reload(jarvis.config)
        importlib.reload(jarvis.tools.directory_lister)
        from jarvis.tools.directory_lister import ListDirectoryTool as FreshTool
        
        tool = FreshTool()
        result = tool.run(path="/etc")
        assert "ERROR" in result
        assert "denied" in result.lower() or "outside" in result.lower()

    def test_directory_not_found(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("FILE_READER_ALLOWED_DIR", str(tmp_path))
        import importlib, jarvis.config, jarvis.tools.directory_lister
        importlib.reload(jarvis.config)
        importlib.reload(jarvis.tools.directory_lister)
        from jarvis.tools.directory_lister import ListDirectoryTool as FreshTool
        
        tool = FreshTool()
        result = tool.run(path=str(tmp_path / "nonexistent_dir"))
        assert "ERROR" in result
        assert "not found" in result.lower()


# ══════════════════════════════════════════════════════════════════════════════
# calculator
# ══════════════════════════════════════════════════════════════════════════════

class TestCalculatorTool:
    def setup_method(self):
        self.tool = CalculatorTool()

    def test_name(self):
        assert self.tool.name == "calculator"

    def test_valid_expressions(self):
        assert "15" in self.tool.run(expression="10 + 5")
        assert "5" in self.tool.run(expression="10 - 5")
        assert "50" in self.tool.run(expression="10 * 5")
        assert "2" in self.tool.run(expression="10 / 5")
        assert "14" in self.tool.run(expression="2 + 3 * 4")
        assert "20" in self.tool.run(expression="(2 + 3) * 4")
        assert "100" in self.tool.run(expression="10 ** 2")
        assert "-5" in self.tool.run(expression="-5")

    def test_division_by_zero(self):
        result = self.tool.run(expression="10 / 0")
        assert "ERROR" in result
        assert "Division by zero" in result

    def test_huge_exponent(self):
        result = self.tool.run(expression="99 ** 9999")
        assert "ERROR" in result
        assert "too large" in result.lower()

    def test_invalid_syntax_and_code_execution(self):
        # Prevent code execution
        result1 = self.tool.run(expression="__import__('os').system('echo hi')")
        assert "ERROR" in result1
        
        # Prevent string manipulation
        result2 = self.tool.run(expression="'A' + 'B'")
        assert "ERROR" in result2

        # Invalid syntax
        result3 = self.tool.run(expression="10 + * 5")
        assert "ERROR" in result3


# ══════════════════════════════════════════════════════════════════════════════
# web_search (formatting — no network)
# ══════════════════════════════════════════════════════════════════════════════

class TestWebSearchTool:
    def setup_method(self):
        self.tool = WebSearchTool()

    def test_name(self):
        assert self.tool.name == "web_search"

    def test_empty_query(self):
        result = self.tool.run(query="   ")
        assert "ERROR" in result

    def test_normalize_dedupes_and_cleans(self):
        raw = [
            {"title": "Alpha", "href": "https://a.example/x", "body": "Fact one.  "},
            {"title": "Alpha", "href": "https://a.example/x/", "body": "duplicate"},
            {"title": "Beta", "href": "https://b.example", "body": "Fact two."},
            {"title": "", "href": "https://c.example", "body": ""},
        ]
        cleaned = _normalize_results(raw, limit=5)
        assert len(cleaned) == 2
        assert cleaned[0]["title"] == "Alpha"
        assert cleaned[1]["snippet"] == "Fact two."

    def test_format_encourages_synthesis(self):
        text = _format_results("local llms", [
            {"title": "News", "url": "https://example.com", "snippet": "Qwen released."},
        ])
        assert "Web search results for:" in text
        assert "[1] News" in text
        assert "Excerpt:" in text
        assert "Do NOT reply with only a list of URLs" in text

    def test_run_formats_ddgs_results(self):
        fake_rows = [
            {"title": "Ollama", "href": "https://ollama.com", "body": "Run LLMs locally."},
        ]
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__.return_value = mock_ddgs
        mock_ddgs.__exit__.return_value = False
        mock_ddgs.text.return_value = fake_rows

        with patch("jarvis.tools.web_search.DDGS", return_value=mock_ddgs):
            result = self.tool.run(query="ollama", max_results=3)

        assert "Ollama" in result
        assert "Run LLMs locally." in result
        assert "```json" not in result


# ══════════════════════════════════════════════════════════════════════════════
# wikipedia_summary (mocked HTTP — no network)
# ══════════════════════════════════════════════════════════════════════════════

class TestWikipediaSummaryTool:
    def setup_method(self):
        self.tool = WikipediaSummaryTool()

    def test_name(self):
        assert self.tool.name == "wikipedia_summary"

    def test_empty_query(self):
        assert "ERROR" in self.tool.run(query="")

    def test_successful_summary(self):
        with patch(
            "jarvis.tools.wikipedia_summary._resolve_title",
            return_value="Alan Turing",
        ), patch(
            "jarvis.tools.wikipedia_summary._fetch_summary",
            return_value=(
                "Wikipedia summary: Alan Turing\n"
                "About: English mathematician\n"
                "URL: https://en.wikipedia.org/wiki/Alan_Turing\n\n"
                "Alan Turing was a pioneer of computer science.\n"
            ),
        ):
            result = self.tool.run(query="Alan Turing")

        assert "Alan Turing" in result
        assert "computer science" in result

    def test_no_article(self):
        with patch(
            "jarvis.tools.wikipedia_summary._resolve_title",
            return_value=None,
        ):
            result = self.tool.run(query="zzzxnotatopic")
        assert "No Wikipedia article" in result


# ══════════════════════════════════════════════════════════════════════════════
# ToolRegistry
# ══════════════════════════════════════════════════════════════════════════════

class TestToolRegistry:
    def setup_method(self):
        self.registry = ToolRegistry()
        self.registry.register(GetCurrentDatetimeTool())

    def test_list_tools(self):
        assert "get_current_datetime" in self.registry.list_tools()

    def test_duplicate_registration_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            self.registry.register(GetCurrentDatetimeTool())

    def test_dispatch_known_tool(self):
        result = self.registry.dispatch("get_current_datetime", "{}")
        assert isinstance(result, str)
        assert "ERROR" not in result

    def test_dispatch_unknown_tool(self):
        result = self.registry.dispatch("nonexistent_tool", "{}")
        assert "ERROR" in result
        assert "Unknown tool" in result

    def test_dispatch_bad_json(self):
        result = self.registry.dispatch("get_current_datetime", "not-json{")
        assert "ERROR" in result

    def test_get_schemas_returns_list(self):
        schemas = self.registry.get_schemas()
        assert isinstance(schemas, list)
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"


# ══════════════════════════════════════════════════════════════════════════════
# vision_analyze (mocked — no network or large models)
# ══════════════════════════════════════════════════════════════════════════════

class TestVisionAnalyzeTool:
    def setup_method(self):
        from jarvis.tools.vision_analyze import VisionAnalyzeTool
        self.tool = VisionAnalyzeTool()

    def test_name(self):
        assert self.tool.name == "vision_analyze"

    def test_blocks_path_traversal(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("FILE_READER_ALLOWED_DIR", str(tmp_path))
        import importlib, jarvis.config, jarvis.tools.vision_analyze
        importlib.reload(jarvis.config)
        importlib.reload(jarvis.tools.vision_analyze)
        from jarvis.tools.vision_analyze import VisionAnalyzeTool as FreshTool
        tool = FreshTool()
        
        result = tool.run(image_path="/etc/passwd")
        assert "ERROR" in result
        assert "denied" in result.lower()

    def test_file_not_found(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("FILE_READER_ALLOWED_DIR", str(tmp_path))
        import importlib, jarvis.config, jarvis.tools.vision_analyze
        importlib.reload(jarvis.config)
        importlib.reload(jarvis.tools.vision_analyze)
        from jarvis.tools.vision_analyze import VisionAnalyzeTool as FreshTool
        tool = FreshTool()
        
        result = tool.run(image_path=str(tmp_path / "nonexistent.jpg"))
        assert "ERROR" in result
        assert "not found" in result.lower()


# ══════════════════════════════════════════════════════════════════════════════
# web_scrape (mocked playwright)
# ══════════════════════════════════════════════════════════════════════════════

class TestWebScrapeTool:
    def setup_method(self):
        from jarvis.tools.web_scrape import WebScrapeTool
        self.tool = WebScrapeTool()

    def test_name(self):
        assert self.tool.name == "web_scrape"

    @patch("playwright.sync_api.sync_playwright")
    def test_successful_scrape(self, mock_playwright):
        mock_p = MagicMock()
        mock_browser = MagicMock()
        mock_page = MagicMock()
        
        mock_playwright.return_value.__enter__.return_value = mock_p
        mock_p.chromium.launch.return_value = mock_browser
        mock_browser.new_page.return_value = mock_page
        mock_page.inner_text.return_value = "This is a mock webpage content."
        
        result = self.tool.run(url="https://example.com")
        assert "This is a mock webpage content." in result
        mock_page.goto.assert_called_with("https://example.com", timeout=20000)

