# JARVIS Project Brain

## Project Overview
JARVIS is a local-first, multimodal, agentic AI assistant.

## Architecture Map
- `core/`: The brain of the system, orchestrator, and permission enforcement.
- `tools/`: The capabilities available to the agent (e.g., computer automation, web scraping).
- `memory/`: The state and session history (SQLite, ChromaDB).
- `llm/`: The wrapper for local LLM inference (LiteLLM/Ollama).
- `utils/`: Utilities like logging and system configuration.
- `voice/`: Whisper and TTS integration for voice mode.

## Strict Rules & Constraints
- Never add new dependencies without explicit instruction.
- Always run `uv run pytest -v` after modifying core logic.
- Treat the LLM as an untrusted reasoning engine; enforce PermissionGuard for SYSTEM/DESTRUCTIVE tools.
- Maintain dependency injection in the Orchestrator.

## Testing & Verification
- Run tests: `uv run pytest -v`
- Enforce linting/formatting according to project standards.

## Tech Stack
- Ollama
- LiteLLM
- ChromaDB
- SQLite
- Streamlit
- Playwright

## Intent Router Architecture & Trade-offs
- **Mechanism:** Heuristic keyword + length matching (zero latency).
- **Benefits:** Skips the heavy Plan-and-Execute loop for simple queries, saving ~2-3 LLM calls.
- **Known Limitations (Edge Cases):** False Positives (complex queries containing simple keywords) and False Negatives (simple queries missing keywords). This is an intentional trade-off prioritizing speed and reduced token cost over perfect classification accuracy.

## Tool-Selection Policy (v0.21)
- `jarvis/core/tool_policy.py` implements the tool-selection CONTRACT (system block + classifier + schema narrowing + unmet-capability honesty). It generalizes by capability class, not per-phrase special cases — extend it by class, never by quoting user phrases.
- The deterministic calculator fallback covers exactly ONE bounded class: single-intent arithmetic where the model emitted zero tool calls. Knowledge retrieval stays model-driven on purpose. Unavailable capabilities (code execution, computer control) are honest refusals — even when the request contains arithmetic like `print(2+2)`.
- Kill switch: `JARVIS_DISABLE_TOOL_POLICY=true` restores exact v0.20 behavior (test-pinned).
- Live evidence (qwen2.5:7b A/B, `live_tool_eval.py`, manual-only): correct-tool 0.688 → 0.812; fabrication 0.0 in both arms. `llava:latest` cannot call tools (Ollama rejects the tools parameter) — model compatibility varies; never assume it.

## Multi-Step Planning (v0.22)
- The Planner is grounded in a compact tool catalogue (name: description), NOT bare names; plans pass `jarvis/core/plan_validator.py` (STRUCTURAL checks only: shape, bounds, registry truth, duplicates, forward-reference rejection — never semantic judgment).
- Later steps receive an exact-evidence ledger (clamped raw tool results of earlier steps) via `_build_step_messages(observations=...)`; steps with required_tools get `min_rounds=1`. Keep both guarantees when touching the plan path.

## Repeat Semantics (v0.23)
- `jarvis/core/dispatch_guard.py` is the repeat CONTRACT: within one turn, a call whose (tool, canonical-arguments) fingerprint already SUCCEEDED is suppressed AFTER PermissionGuard/confirmation, BEFORE `registry.dispatch_async` — so suppression can never bypass security or pollute the registry probe.
- Preserve the exemption set EXACTLY `get_current_datetime`, `recall_facts`, `remember_fact` (state/turn-coupled only) and ledger properties when touching dispatch: success-only recording (retry-after-failure works), fresh ledger per turn (`chat()` re-creates it), fingerprint-only logging (never arguments).
- The plan loop skips an identical later step (`redundant_plan_step` log + `redundant_step` SSE) and logs `plan_step_satisfied`/`plan_completed`; `plan_ready` carries `quality=` from `jarvis/core/plan_quality.py` (deterministic, never authorization).
- `evaluation/plan_judge.py` is evaluation-ONLY (semantic plan scoring; conservative degradation). Never wire it into the production loop.
- Benchmarks: `evaluation/multistep_benchmark.py` (11 cases incl. repeat_semantics, pytest-wired), `live_multistep_eval.py` (manual-only; per-case suppression counts). Suppression is exact-argument: `"2+2"` vs `"2 + 2"` differ by fingerprint — do not add math special cases.
- Routing: compute+memory conjunctions and search+compare requests go to the planner; NEVER route on bare operator substrings (`-` matches 'plan-and-execute' — use the digit-operator pattern).
- A plan is a request, never authorization; live multi-step evidence and limits are in `live_multistep_eval.py` + developer manual §17/§26.
