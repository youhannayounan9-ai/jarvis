# jarvis/tools/__init__.py
"""
Tools package.

All tool classes are importable from here for convenience.

Status legend:
  ACTIVE    — registered in main.py, available to the LLM.
  DISABLED  — placeholder only; NOT registered. run() returns an error message.
              Do not re-enable until a proper sandboxing/approval strategy exists.
"""

# ── Infrastructure ─────────────────────────────────────────────────────────────
from jarvis.tools.base import BaseTool
from jarvis.tools.registry import ToolRegistry

# ── ACTIVE tools ───────────────────────────────────────────────────────────────
from jarvis.tools.calculator import CalculatorTool
from jarvis.tools.datetime_tool import GetCurrentDatetimeTool
from jarvis.tools.directory_lister import ListDirectoryTool
from jarvis.tools.file_reader import ReadFileTool
from jarvis.tools.recall_facts import RecallFactsTool
from jarvis.tools.remember_fact import RememberFactTool
from jarvis.tools.search_knowledge import SearchKnowledgeTool
from jarvis.tools.vision_analyze import VisionAnalyzeTool
from jarvis.tools.web_scrape import WebScrapeTool
from jarvis.tools.web_search import WebSearchTool
from jarvis.tools.wikipedia_summary import WikipediaSummaryTool
from jarvis.tools.write_file import WriteFileTool

# ── DISABLED placeholders (not registered, kept for future implementation) ─────
from jarvis.tools.code_execution import CodeExecutionTool      # gated — Docker-isolated only
from jarvis.tools.computer_control import ComputerControlTool  # DISABLED — no OS automation

__all__ = [
    # Infrastructure
    "BaseTool",
    "ToolRegistry",
    # Active
    "CalculatorTool",
    "GetCurrentDatetimeTool",
    "ListDirectoryTool",
    "ReadFileTool",
    "RecallFactsTool",
    "RememberFactTool",
    "SearchKnowledgeTool",
    "VisionAnalyzeTool",
    "WebScrapeTool",
    "WebSearchTool",
    "WikipediaSummaryTool",
    "WriteFileTool",
    # Disabled placeholders
    "CodeExecutionTool",
    "ComputerControlTool",
]
