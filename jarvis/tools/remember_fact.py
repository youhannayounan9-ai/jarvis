"""
jarvis/tools/remember_fact.py
──────────────────────────────
Tool: remember_fact

Persists an important user fact into local long-term vector memory (ChromaDB).
"""

from jarvis.memory.vector_store import get_vector_store
from jarvis.tools.base import BaseTool
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class RememberFactTool(BaseTool):
    name = "remember_fact"
    description = (
        "Saves an important, long-term fact about the user, their preferences, "
        "or ongoing projects. Use this when the user explicitly states something "
        "they want you to remember."
    )
    parameters = {
        "type": "object",
        "properties": {
            "fact": {
                "type": "string",
                "description": "The concise fact to remember.",
            },
        },
        "required": ["fact"],
    }
    risk_level = "SAFE"
    timeout_seconds = 60.0

    def run(self, fact: str, **kwargs) -> str:
        store = get_vector_store()
        session_id = store.current_session_id
        log.info("remember_fact_tool", session_id=session_id, chars=len(fact or ""))
        return store.add_fact(session_id=session_id, fact=fact)
