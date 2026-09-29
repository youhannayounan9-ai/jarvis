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
        "Search the user's personal long-term memory for facts about them. "
        "PURPOSE: recall what the user told you before (preferences, name, "
        "projects, plans). "
        "WHEN TO USE: before answering ANY question about the user personally "
        "('my favorite…', 'what is my name', 'what did I say about…') — "
        "never guess personal facts. "
        "WHEN NOT TO USE: content of the user's documents (search_knowledge), "
        "general world knowledge, or things already stated in this "
        "conversation. "
        "INPUT: short search query. OUTPUT: formatted memory matches, a "
        "no-results notice, or an ERROR string — report a miss honestly."
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
