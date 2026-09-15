# JARVIS v0.1 — Architecture Notes

## System Overview

```
User Input (CLI)
      │
      ▼
  main.py  ←── wires everything together
      │
      ▼
 Orchestrator  ←── core/orchestrator.py
      │
      ├─ SessionStore (memory/)   ←── SQLite conversation history
      ├─ PermissionGuard (core/)  ←── controls what tools may run
      └─ LLM Client (llm/)        ←── LiteLLM → Ollama
              │
              └─ ToolRegistry (tools/)
                      ├─ get_current_datetime
                      ├─ web_search
                      └─ read_file
```

## Request Lifecycle

1. User types a message at the CLI prompt.
2. `main.py` calls `orchestrator.chat(session_id, user_input)`.
3. The orchestrator saves the user message to SQLite.
4. It loads the full session history and prepends the system prompt.
5. It calls `llm/client.py → LiteLLM → Ollama` with messages + tool schemas.
6. **If the LLM returns tool_calls:**
   - The orchestrator checks each tool against `PermissionGuard`.
   - Dispatches the call through `ToolRegistry`.
   - Saves the tool result to SQLite and appends it to `messages`.
   - Calls the LLM again (another loop iteration).
7. **If the LLM returns plain text:** loop exits, text is returned.
8. `main.py` renders the text as Markdown in a Rich panel.

## Key Design Decisions

### LiteLLM abstraction
All LLM calls go through `jarvis/llm/client.py`. If we switch from Ollama to
OpenAI or Anthropic, we change the model string in `.env` — nothing else changes.

### Dependency injection in Orchestrator
The Orchestrator receives its dependencies (store, registry, guard) as constructor
arguments, never instantiates them itself. This makes it testable in isolation
without a running database or Ollama server.

### Tool sandboxing
`read_file` resolves all paths to absolute form and checks they are inside
`FILE_READER_ALLOWED_DIR` using `Path.relative_to()`. No regex, no string tricks.

### ReAct loop with safety cap
The tool-calling loop is capped at `MAX_TOOL_ROUNDS = 5`. If the LLM hasn't
produced a plain text answer by then, one final unconstrained call forces it.

### System prompt not stored
The system prompt is injected fresh at each call, not persisted. This means:
- It can be changed without a DB migration.
- It doesn't consume stored message slots.

## Adding a New Tool (Checklist)

1. Create `jarvis/tools/my_tool.py`.
2. Subclass `BaseTool`.
3. Define `name`, `description`, `parameters` (JSON Schema), and `run()`.
4. Import and register in `jarvis/main.py`:
   ```python
   from jarvis.tools.my_tool import MyTool
   registry.register(MyTool())
   ```
5. Add the export to `jarvis/tools/__init__.py`.
6. Write a test in `tests/test_tools.py`.

That's it. Nothing else in the system needs to change.
