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
        "Persist one fact about the user to long-term memory. "
        "PURPOSE: make personal facts survive beyond this conversation. "
        "WHEN TO USE: the user states something about themselves worth keeping "
        "(name, preferences, projects, deadlines) — acknowledging in text does "
        "NOT store it. "
        "WHEN NOT TO USE: facts about the world, document contents, or "
        "transient chat context. "
        "INPUT: one concise fact. OUTPUT: a 'Remembered: …' confirmation or an "
        "ERROR string; never claim a fact was saved without this tool's result."
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
