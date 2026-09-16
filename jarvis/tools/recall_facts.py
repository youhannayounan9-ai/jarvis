"""
jarvis/tools/recall_facts.py
─────────────────────────────
Tool: recall_facts

Searches local long-term vector memory (ChromaDB) for relevant user facts.
"""

from jarvis.memory.vector_store import get_vector_store
from jarvis.tools.base import BaseTool
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class RecallFactsTool(BaseTool):
    name = "recall_facts"
    description = (
        "CRITICAL: You MUST use this tool whenever the user asks a question "
        "about themselves, their preferences, or past conversations. "
        "Do not guess or hallucinate; search the database first."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query for the memory.",
            },
        },
        "required": ["query"],
    }
    risk_level = "SAFE"
    timeout_seconds = 60.0

    def run(self, query: str, **kwargs) -> str:
        store = get_vector_store()
        log.info("recall_facts_tool", query=query)
        return store.search_facts(query=query, limit=3)
