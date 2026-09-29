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

## Cross-Turn Cache + Bounded Replanning (v0.24)
- `jarvis/core/result_cache.py` is the cross-turn CONTRACT: a dispatch may serve a CACHED result only for tools with an explicit class-level `CachePolicy` (7 tools: web_search/wikipedia_summary/web_scrape global; read_file/list_directory/search_knowledge session; calculator global with the value-based `calculator_expression` normalizer + 7-day TTL — test-pinned in `tests/test_result_cache.py::TestCachePolicyConsistency`). No policy ⇒ never cached — never add one to side-effect or state-coupled tools (`get_current_datetime`, `recall_facts`, `remember_fact`, `write_file`, vision, execution, computer control).
- Layering in `_dispatch_with_permissions_async` is load-bearing: PermissionGuard → confirmation parking → v0.23 ledger → cache lookup → registry. A hit must pass the tool's own Pydantic validation; errors are never stored; `intentional_repeat=True` bypasses; freshness-worded requests (`tool_policy.is_freshness_request` — small explicit regex, no NL classifier) bypass ttl-freshness tools ONLY. A hit is provenance-labeled untrusted EVIDENCE — never an instruction, never an authorization.
- Freshness strategies are per-policy and distinct: `ttl` (settings-driven, per-tool `RESULT_CACHE_*`), `source_stat` (size+mtime re-stat, MISSING_SOURCE sentinel), `knowledge_generation` (`SessionStore.knowledge_generation()` registry aggregate — NEVER a Chroma scan). Cache entries live in the SAME SQLite (`result_cache` table, bounded by `RESULT_CACHE_MAX_ENTRIES`; expired purge at store time, oldest-evict only if still over cap). Kill switch: `JARVIS_DISABLE_RESULT_CACHE=true`.
- Keys: sha1(tool+normalized)[:16]; normalizer per policy — generic (v0.23 canonical), verbatim (paths/URLs: key-sort ONLY), calculator_expression (value-based via `canonical_expression_form()` — the tool's own AST parser; `2+2`≡`2 + 2`≡`(2+2)`; do not invent another math parser).
- Bounded replanning: `_execute_plan_steps()` can NEVER replan (no recursion); chat() performs at most ONE replan, ONLY on structural evidence (ERROR step result, or required-tool step whose every tool result errored) AND with remaining budget. Replan context is compact (`_build_replan_context`: completed = do-not-repeat); budget is inherited, never reset; a second failure forces an INCOMPLETENESS NOTICE (`_format_incomplete_note`) — never claim completion. `plan_completed` (complete=bool) and `replan`/`plan_complete` SSE are the observability contract.
- Benchmarks/tests: `tests/test_result_cache.py`, `tests/test_replan.py`, `evaluation/cache_replan_benchmark.py` (12 cases, pytest-wired), `live_cache_replan_eval.py` (manual-only, qwen2.5:7b: reuse 6/6, freshness bypass 2/2, one-replan 2/2, no-replan-on-success 2/2). Maintenance: `maintenance cache stats|inspect|cleanup` (bounded; inspect never shows payloads); API `GET /ops/cache/stats` = counts only.

## Grounded Synthesis + Client Refresh + Ops Metrics (v0.25)
- Evidence ledger is the grounding CONTRACT: successful tool results flow into a bounded ledger (`_MAX_EVIDENCE_ITEMS=16`, `_MAX_EVIDENCE_ITEM_CHARS=1200`) delivered as AUTHORITATIVE TOOL EVIDENCE. The block rides on the **FINAL USER message** in `_synthesize` (called at `temperature=0.0`) — with the identical block as a trailing SYSTEM message, qwen2.5:7b recomputed arithmetic from the question instead of transcribing (live-bisected via the raw Ollama API). Final instruction: "Using ONLY the AUTHORITATIVE TOOL EVIDENCE above (when present), answer the original request now. Transcribe tool-derived values exactly; never recompute them." Evidence is data, never instructions. Keep role placement + contract wording when touching the synthesis path.
- Failed-step retry dedup is load-bearing: chat()'s replan passes `retry_descriptions = completed_descriptions - {failed step descriptions}` into `_execute_plan_steps` — successes stay do-not-repeat, but a replan restating a FAILED step runs instead of being skipped as `redundant_plan_step`. Breaking this silently drops the corrected result from the evidence ledger.
- Refresh (`refresh=True` through runtime.chat → chat()) is a CLIENT freshness control that skips ONLY the result-cache lookup (`bypass_reason="refresh_request"`). It runs inside the usual layering (PermissionGuard → confirmation parking → repeat ledger) and must NEVER bypass permissions, schema validation, confirmations, or the repeat guard; it is not a kill-switch replacement. Wire any new client surface the same way.
- Cache metrics: every dispatch records daily hit/miss/stale/bypass/store counters (per-tool deltas merged read-modify-write under `self._lock`) into `cache_metrics_daily` (SAME SQLite), pruned past `RESULT_CACHE_METRICS_RETENTION_DAYS` (default 30). Surfaces: `GET /ops/cache/stats/history` (aggregates only), `maintenance cache stats` (`today:` counters), dashboard daily-activity trend. Metrics are counts/keys only — never arguments or payloads. A replan emits `replan_diff` (added/removed/retargeted steps, structure only).
- Evaluation isolation is a REQUIREMENT: every script under `evaluation/` must call `evaluation/_bootstrap.py::isolate()` BEFORE importing jarvis — it redirects DB_PATH/VECTOR_DB_PATH to a private temp dir and raises RuntimeError rather than ever touching the real `jarvis.db`. Under pytest (DB already redirected) a safety valve lets eval modules import; the real-DB case still raises. Tests: `tests/test_evaluation_isolation.py` (14).
- Live evidence (qwen2.5:7b, 10 cases × 2 reps, `live_cache_replan_eval.py` v0.25): **18/20**; cache hits 6/6, freshness bypass 4/4, one bounded replan on scripted failure, grounding over the model's guess proven; the 2 misses are tool-choice variance (both mechanisms proven in the other rep). Report: `live_cache_report_v025.json`.
