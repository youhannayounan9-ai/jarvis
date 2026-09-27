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

    def test_dispatch_validation_missing_required(self):
        from jarvis.tools.web_search import WebSearchTool
        self.registry.register(WebSearchTool())
        # WebSearchTool requires 'query'
        result = self.registry.dispatch("web_search", "{}")
        assert "ERROR: Invalid arguments for 'web_search'" in result
        assert "query" in result
        assert "Field required" in result or "required" in result

    def test_dispatch_validation_wrong_type(self):
        from jarvis.tools.web_search import WebSearchTool
        if "web_search" not in self.registry._tools:
            self.registry.register(WebSearchTool())
        # max_results should be an integer, but we pass an object
        result = self.registry.dispatch("web_search", '{"query": "test", "max_results": {}}')
        assert "ERROR: Invalid arguments" in result
        assert "max_results" in result

    def test_dispatch_validation_extra_args_forbidden(self):
        from jarvis.tools.web_search import WebSearchTool
        if "web_search" not in self.registry._tools:
            self.registry.register(WebSearchTool())
        result = self.registry.dispatch("web_search", '{"query": "test", "hallucinated_arg": "bad"}')
        assert "ERROR: Invalid arguments" in result
        assert "hallucinated_arg" in result
        assert "Extra inputs are not permitted" in result or "extra" in result.lower()

    def test_dispatch_validation_success_with_optional(self):
        from jarvis.tools.web_search import WebSearchTool
        if "web_search" not in self.registry._tools:
            self.registry.register(WebSearchTool())
        
        with patch("jarvis.tools.web_search.WebSearchTool.run", return_value="success") as mock_run:
            result = self.registry.dispatch("web_search", '{"query": "test"}')
            assert result == "success"
            mock_run.assert_called_once_with(query="test")

    def test_dispatch_validation_coercion(self):
        from jarvis.tools.web_search import WebSearchTool
        if "web_search" not in self.registry._tools:
            self.registry.register(WebSearchTool())
        
        with patch("jarvis.tools.web_search.WebSearchTool.run", return_value="success") as mock_run:
            # max_results is an integer field, but we pass a string "5"
            result = self.registry.dispatch("web_search", '{"query": "test", "max_results": "5"}')
            assert result == "success"
            # verify it was passed as the integer 5
            mock_run.assert_called_once_with(query="test", max_results=5)


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


# ══════════════════════════════════════════════════════════════════════════════
# computer_control (mocked pyautogui)
# ══════════════════════════════════════════════════════════════════════════════

class TestComputerControlTool:
    def setup_method(self):
        from jarvis.tools.computer_control import ComputerControlTool
        self.tool = ComputerControlTool()

    def test_name(self):
        assert self.tool.name == "computer_control"

    def test_move_mouse_disabled(self):
        result = self.tool.run(action="move_mouse", x=100, y=200)
        assert "Computer control is currently disabled" in result

    def test_type_text_disabled(self):
        result = self.tool.run(action="type_text", text="hello")
        assert "Computer control is currently disabled" in result

    def test_click_disabled(self):
        result = self.tool.run(action="click", x=50, y=50)
        assert "Computer control is currently disabled" in result

    def test_press_key_disabled(self):
        result = self.tool.run(action="press_key", key="enter")
        assert "Computer control is currently disabled" in result

    def test_scroll_disabled(self):
        result = self.tool.run(action="scroll", amount=3)
        assert "Computer control is currently disabled" in result

    def test_not_registered_in_main(self):
        """The standard runtime surface must NOT include computer_control."""
        from unittest.mock import patch as _patch
        with _patch("jarvis.runtime.get_vector_store"):
            from jarvis.runtime import build_runtime
            runtime = build_runtime()
        assert "computer_control" not in runtime.registry.list_tools()


# ══════════════════════════════════════════════════════════════════════════════
# code_execution (sandbox checks)
# ══════════════════════════════════════════════════════════════════════════════

class TestCodeExecutionTool:
    def setup_method(self):
        from jarvis.tools.code_execution import CodeExecutionTool
        self.tool = CodeExecutionTool()

    def test_name(self):
        assert self.tool.name == "execute_python_code"

    def test_basic_execution(self):
        result = self.tool.run(code="print(2 + 2)")
        assert "Code execution is currently disabled" in result

    def test_security_violation(self):
        result = self.tool.run(code="import os")
        assert "Code execution is currently disabled" in result

    def test_execution_timeout(self):
        result = self.tool.run(code="while True: pass")
        assert "Code execution is currently disabled" in result

    def test_not_registered_in_main(self):
        """The standard runtime surface must NOT include execute_python_code."""
        from unittest.mock import patch as _patch
        with _patch("jarvis.runtime.get_vector_store"):
            from jarvis.runtime import build_runtime
            runtime = build_runtime()
        assert "execute_python_code" not in runtime.registry.list_tools()


# ══════════════════════════════════════════════════════════════════════════════
# write_file (sandboxed)
# ══════════════════════════════════════════════════════════════════════════════

class TestWriteFileTool:
    def setup_method(self):
        from jarvis.tools.write_file import WriteFileTool
        self.tool = WriteFileTool()

    def test_name(self):
        assert self.tool.name == "write_file"

    def test_risk_level(self):
        assert self.tool.risk_level == "FILE_WRITE"

    def test_writes_file_inside_sandbox(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FILE_READER_ALLOWED_DIR", str(tmp_path))
        import importlib, jarvis.config, jarvis.tools.write_file
        importlib.reload(jarvis.config)
        importlib.reload(jarvis.tools.write_file)
        from jarvis.tools.write_file import WriteFileTool as FreshTool
        tool = FreshTool()
        result = tool.run(file_path=str(tmp_path / "out.txt"), content="hello")
        assert "Successfully wrote" in result
        assert (tmp_path / "out.txt").read_text() == "hello"

    def test_append_mode(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FILE_READER_ALLOWED_DIR", str(tmp_path))
        import importlib, jarvis.config, jarvis.tools.write_file
        importlib.reload(jarvis.config)
        importlib.reload(jarvis.tools.write_file)
        from jarvis.tools.write_file import WriteFileTool as FreshTool
        tool = FreshTool()
        target = tmp_path / "log.txt"
        target.write_text("line1\n")
        tool.run(file_path=str(target), content="line2\n", append=True)
        assert target.read_text() == "line1\nline2\n"

    def test_blocks_path_traversal(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FILE_READER_ALLOWED_DIR", str(tmp_path))
        import importlib, jarvis.config, jarvis.tools.write_file
        importlib.reload(jarvis.config)
        importlib.reload(jarvis.tools.write_file)
        from jarvis.tools.write_file import WriteFileTool as FreshTool
        tool = FreshTool()
        result = tool.run(
            file_path=str(tmp_path / "../../etc/crontab"),
            content="evil"
        )
        assert "ERROR" in result
        assert "Access denied" in result

    def test_blocks_absolute_path_outside_sandbox(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FILE_READER_ALLOWED_DIR", str(tmp_path))
        import importlib, jarvis.config, jarvis.tools.write_file
        importlib.reload(jarvis.config)
        importlib.reload(jarvis.tools.write_file)
        from jarvis.tools.write_file import WriteFileTool as FreshTool
        tool = FreshTool()
        # Attempt to overwrite a system-adjacent file
        result = tool.run(file_path="C:\\Windows\\evil.txt", content="evil")
        assert "ERROR" in result
        assert "Access denied" in result


# ══════════════════════════════════════════════════════════════════════════════
# permission confirmation flow (via Orchestrator)
# ══════════════════════════════════════════════════════════════════════════════

class TestPermissionConfirmationFlow:
    def test_confirmation_handling(self):
        from jarvis.core.orchestrator import Orchestrator
        from jarvis.core.permissions import PermissionGuard
        from jarvis.memory.session_store import SessionStore
        from jarvis.tools.registry import ToolRegistry
        
        # We just test the handle_confirmation logic
        registry = ToolRegistry()
        guard = PermissionGuard()
        store = MagicMock()
        orchestrator = Orchestrator(store, registry, guard)
        
        # Inject a pending confirmation
        pending_data = {
            "tool_name": "dummy",
            "tool_args": "{}",
            "tool_call_id": "call_123",
            "risk_level": "SYSTEM"
        }
        store.complete_pending_confirmation.return_value = pending_data
        
        # Deny
        res_deny = orchestrator.handle_confirmation("test_session", False)
        assert "denied" in res_deny.lower()
        
        # Confirm (since dummy isn't in registry, it'll error from registry, but that proves it tried to dispatch)
        pending_data["tool_call_id"] = "call_124"
        store.complete_pending_confirmation.return_value = pending_data
        res_confirm = orchestrator.handle_confirmation("test_session", True)
        assert "Unknown tool" in res_confirm or "Executed" in res_confirm


