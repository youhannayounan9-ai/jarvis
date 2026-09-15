"""
tests/test_memory_tools.py
───────────────────────────
Unit tests for remember_fact / recall_facts (mocked vector store — no Chroma).
"""

from unittest.mock import MagicMock, patch

from jarvis.tools.recall_facts import RecallFactsTool
from jarvis.tools.remember_fact import RememberFactTool


class TestRememberFactTool:
    def setup_method(self):
        self.tool = RememberFactTool()

    def test_name_and_risk(self):
        assert self.tool.name == "remember_fact"
        assert self.tool.risk_level == "SAFE"

    def test_run_delegates_to_vector_store(self):
        mock_store = MagicMock()
        mock_store.current_session_id = "sess-1"
        mock_store.add_fact.return_value = "Remembered: likes dark mode"

        with patch(
            "jarvis.tools.remember_fact.get_vector_store",
            return_value=mock_store,
        ):
            result = self.tool.run(fact="likes dark mode")

        mock_store.add_fact.assert_called_once_with(
            session_id="sess-1",
            fact="likes dark mode",
        )
        assert result == "Remembered: likes dark mode"


class TestRecallFactsTool:
    def setup_method(self):
        self.tool = RecallFactsTool()

    def test_name_and_risk(self):
        assert self.tool.name == "recall_facts"
        assert self.tool.risk_level == "SAFE"

    def test_run_delegates_to_vector_store(self):
        mock_store = MagicMock()
        mock_store.search_facts.return_value = "Long-term memory matches…"

        with patch(
            "jarvis.tools.recall_facts.get_vector_store",
            return_value=mock_store,
        ):
            result = self.tool.run(query="preferences")

        mock_store.search_facts.assert_called_once_with(query="preferences", limit=3)
        assert "Long-term memory" in result
