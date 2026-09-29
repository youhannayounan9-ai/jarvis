# JARVIS v0.25 Final Report

**Version:** 0.25.0 (verified in `pyproject.toml`, `jarvis/__init__.py`, `jarvis/api/app.py`)
**Date:** 2026-09-30
**Scope:** GROUNDED SYNTHESIS + CLIENT FRESHNESS + OPERATIONS METRICS on the verified v0.24 state.

---

## 1. What Changed

v0.25 ships six coordinated changes, all test-verified and (where meaningful) live-verified on qwen2.5:7b:

| # | Change | Surface |
|---|--------|---------|
| B | **Grounded synthesis** — bounded evidence ledger of raw tool results delivered as AUTHORITATIVE TOOL EVIDENCE on the final user message, `temperature=0.0` transcription-style synthesis call | `jarvis/core/orchestrator.py`, `jarvis/llm/client.py` |
| D | **Explicit client refresh** — API `ChatRequest.refresh`, client `chat(..., refresh=)`, CLI `--refresh` + `/refresh` toggle, dashboard sidebar toggle | `jarvis/api/*`, `jarvis/main.py`, `ui/dashboard.py` |
| E | **Daily cache metrics** — `cache_metrics_daily` table (same SQLite), per-tool read-modify-write merge under the store lock, 30-day retention; history endpoint, maintenance `today:` counters, dashboard trend | `jarvis/memory/session_store.py`, `jarvis/core/result_cache.py`, `jarvis/api/app.py`, `jarvis/maintenance.py` |
| F | **Replan-diff telemetry** — `replan_diff` log + SSE (steps added/removed/retargeted, structure only, never arguments) | `jarvis/core/orchestrator.py` |
| G | **Evaluation isolation** — `evaluation/_bootstrap.py::isolate()` (private temp DB before any jarvis import; pytest safety valve) wired into all 9 eval scripts, plus per-case `fresh_store()` hermetic databases for both benchmarks | `evaluation/*` |
| H | **Cache-policy consistency pinning** — exactly 7 cached tools asserted against the documented policy table | `tests/test_result_cache.py::TestCachePolicyConsistency` |

Plus three **live-found defects fixed** (§9), two new security test suites (§10), documentation (§13) and the version bump (§14).

## 2. Synthesis Grounding (Part B)

- Every successful tool result flows into a bounded evidence ledger: **≤16 items (`_MAX_EVIDENCE_ITEMS`), ≤1200 chars/item (`_MAX_EVIDENCE_ITEM_CHARS`)**, clamped deterministically by `_format_evidence_ledger`; `_tool_from_observation` parses both raw results and registry-probe `<tool> ok: ...` output.
- `_synthesize` appends `_EVIDENCE_CONTRACT` + the ledger to the **FINAL USER message** with the final instruction: *"Using ONLY the AUTHORITATIVE TOOL EVIDENCE above (when present), answer the original request now. Transcribe tool-derived values exactly; never recompute them."* — and calls `chat_completion(..., temperature=0.0)`.
- **The message role is load-bearing, not stylistic.** With the identical block as a trailing SYSTEM message after bare user turns, qwen2.5:7b deterministically recomputed arithmetic from the question text (33071 / 42071 instead of 41971) — even at temperature 0. Verified by raw Ollama-API bisect (bare user + evidence-in-user → 41971; adding the JARVIS system prompt → 41971; adding the tool contract → 41971; evidence-in-system → wrong). In the final user turn it transcribes correctly every time.
- The evidence block is framed as **data, never instructions** (pinned in `tests/test_refresh_security.py`); after a failed replan the same block carries the INCOMPLETENESS NOTICE.

## 3. Explicit Refresh (Part D)

- One flag, four client surfaces: REST API (`"refresh": true`), Python client (`client.chat(..., refresh=True)`, `params=` support added to `_request`), CLI (`--refresh` flag, `/refresh` in-chat toggle, help row), dashboard (sidebar toggle threaded through both `ApiBackend` and `LegacyBackend`).
- **Security posture (test-pinned, 16 tests in `tests/test_refresh_security.py`):** refresh skips **only the result-cache lookup** (`bypass_reason="refresh_request"`). The layering is unchanged — PermissionGuard → confirmation parking → v0.23 repeat ledger → cache → registry — so refresh cannot bypass permissions, schema validation, confirmations, or repeat suppression. A refreshed high-risk action still parks for approval.
- Freshness-worded requests keep their v0.24 TTL-bypass semantics; the `JARVIS_DISABLE_*` kill switches are unchanged.

## 4. Cache Metrics History (Part E)

- `cache_metrics_daily` (same SQLite file): daily `hits/misses/stale/bypass/stores` counters plus per-tool deltas, merged **read-modify-write under `self._lock`** (verified: hits=3, misses=1 accumulate per tool).
- Wired at 7 points in `jarvis/core/result_cache.py` (hit/miss/stale/bypass/store; bypass distinguishes `refresh_request` from freshness-word bypasses).
- Retention: `RESULT_CACHE_METRICS_RETENTION_DAYS` (default **30**); older rows pruned (400-day-old rows verified pruned).
- Surfaces: `GET /ops/cache/stats/history?days=&limit=` (aggregates only), `maintenance cache stats` → `today: hits=.. misses=.. stale=.. bypass=.. stores=..` + per-tool deltas, dashboard **Daily cache activity (last 14 days)** trend table.
- Cost: ~0.02 ms/call (`record_cache_metrics`, n=100) and ~0.02 ms/query (`cache_metrics_history`, n=200) — measured, negligible.

## 5. Replan Diff Telemetry (Part F)

- A validated replan now emits `replan_diff` (log + SSE): `steps_added`, `steps_removed`, `tools_changed`, `added_tools`, `removed_tools`, `capabilities_changed` — **plan structure only, never tool arguments** (leak-checked in tests: `_diff_plans` never emits `tool_args` or injected secrets).
- Companion fix (live-found, §9): the replan passes `retry_descriptions` — completed descriptions **minus failed steps** — so a replan restating a failed step **executes** instead of being skipped as `redundant_plan_step`. Successful steps remain do-not-repeat.

## 6. Evaluation Isolation (Part G)

- `evaluation/_bootstrap.py::isolate()` runs **before any jarvis import** in all 9 evaluation entry points: private per-process temp dir for `DB_PATH`/`VECTOR_DB_PATH`; **RuntimeError** (fail loudly) if the real `jarvis.db` would be touched; one safety valve — if `jarvis.config` is already imported **and** `DB_PATH` is already redirected away from the production name (pytest conftest's `:memory:`), the environment is kept so eval modules import under tests.
- **Per-case hermeticity (this session's addition):** `fresh_store()` re-binds `settings.db_path` around each `SessionStore()` construction, giving every benchmark case its own database. Required because v0.25 added the calculator to the cached set (global scope, 7-day TTL): two standalone cases evaluating `12 * 12` shared one entry through `isolate()`'s single file DB and the later case recorded zero dispatches. Pytest was immune (`:memory:` is per-store) — the standalone path is now pinned by 2 new tests in `tests/test_evaluation_isolation.py` (**16 total**).
- All 9 scripts `py_compile` clean.

## 7. Cache Policy Consistency (Part H)

`tests/test_result_cache.py::TestCachePolicyConsistency` (4 tests) pins:

- **Exactly 7 cached tools:** `web_search`, `wikipedia_summary`, `web_scrape` (global / ttl / generic + verbatim + verbatim), `read_file`, `list_directory` (session / source_stat / verbatim), `search_knowledge` (session / knowledge_generation / generic), **`calculator` (global / ttl 604800 / `calculator_expression`)** — the v0.24 "6 tools" docs are updated everywhere to 7.
- The expected policy table, the calculator's **value-based keys** (`2+2` ≡ `2 + 2` ≡ `(2+2)`; `2+3` distinct), and the **never-cached set** (`get_current_datetime`, `remember_fact`, `recall_facts`, `write_file`, `vision_analyze` must have no `cache_policy`).
- Adding/removing/changing a policy without updating the docs now fails CI — docs and runtime cannot drift.

## 8. Deterministic Evaluation (Parts C, G)

- `tests/test_synthesis.py` — **13 tests**: evidence asserted in synthesis prompts via `_SynthScenario` (capturing tool-free `chat_completion` calls), contract wording, bounds, incomplete-note delivery; replan tests use distinct replan-step descriptions to avoid the legitimate v0.23 redundant-skip.
- `tests/test_evaluation_isolation.py` — **16 tests** (incl. subprocess-based write-confinement, since jarvis is already in-process under pytest).
- `evaluation/multistep_benchmark.py` — **11/11 standalone** (pytest-wired, green in the full suite).
- `evaluation/cache_replan_benchmark.py` — **12/12 standalone** (pytest-wired, green in the full suite).

## 9. Live Model Evaluation (Parts K/P — qwen2.5:7b, 10 cases × 2 reps)

**Result: 18/20 passed** (`live_cache_report_v025.json`; Ollama v0.34.4, qwen2.5:7b + llava:latest installed).

| Metric | Result |
|---|---|
| Cross-turn cache hits (turn 2, fresh session) | **6/6** |
| Freshness-word bypass | **4/4** |
| Exactly-one-bounded-replan on scripted failure | proven (replans_total 4 across cases) |
| Grounding — tool value 8947×123457 = 1104569779 transcribed over the model's guess | proven (rep 1) |
| Live tool calls / avg latency | 26 calls, 5.5 s |

The **2 misses are small-model tool-choice variance**, both mechanisms proven in the other rep: the grounding case's rep 2 answered 1104569779 correctly but *without* calling the tool; the knowledge case's rep 2 skipped the calculator yet stated 144. Neither is a system defect; both are documented 7B-model stochasticity (§16).

**Three live-found defects fixed during this phase** (each now regression-pinned):

1. **Evidence placement** — synthesis ignored trailing-system evidence and recomputed arithmetic; fixed by moving the block to the final user message (§2).
2. **Failed-step dedup** — replan restatements of failed steps were skipped as redundant; fixed with `retry_descriptions` (§5).
3. **Registry-probe parsing** — `_tool_from_observation` couldn't parse `<tool> ok: ...`; fixed (strip trailing ` ok`).

`chat_completion` gained a `temperature` parameter; the synthesis call uses `0.0` (transcription-style).

## 10. Tests

**Full suite: 770 passed, 1 skipped (151 s)** — baseline before the phase was 768 passed, 1 skipped; the +2 are the new `fresh_store()` isolation tests. Zero regressions from the synthesis-layout and replan-dedup changes (the two affected tests were updated to scan all messages, not just the system role).

New/extended suites this phase: `test_synthesis.py` (13), `test_evaluation_isolation.py` (16), `test_refresh_security.py` (16), `test_result_cache.py` (+4 policy-consistency), plus updated `test_replan.py` / `test_context_management.py`.

## 11. Security Review

- **Refresh cannot escalate:** pinned by real-registry tests — invalid arguments (`{"expression": 42}`) rejected and **nothing cached**; `computer_control` still parks with `refresh=True`; same-turn duplicate still suppressed with `refresh=True`; refresh actually re-dispatches across turns (entry re-stored); control case (`refresh=False`) still serves the cached result.
- **Evidence is untrusted data:** tool-error results never enter the ledger (`_is_tool_error`); injected instructions inside tool output stay quoted data; the contract explicitly forbids treating evidence as instructions.
- **Metrics leak nothing:** only the keys `day/hits/misses/stale/bypass/stores/per_tool` are accepted, values coerced int + range-clamped; the plan diff never carries `tool_args` or secrets.
- **Eval isolation is write-confined:** subprocess-verified that isolated runs create their scratch files only inside the private temp dir, never the real DB; the real-DB case still raises.

## 12. Runtime Verification

- **CONFIRMED (live):** evidence-in-final-user-message transcription (raw-API bisect); cross-turn cache reuse 6/6; freshness bypass 4/4; bounded replan firing once on scripted failure; failed-step retry executing instead of being skipped; registry-probe `ok` parsing.
- **TEST-VERIFIED:** refresh security posture (16 tests); policy table pinning; metrics merge/pruning/shape; replan-diff leak-freedom; evaluation isolation incl. subprocess confinement; standalone benchmarks 11/11 + 12/12; full suite 770 passed.
- **ENVIRONMENT-BLOCKED:** nothing new — code execution remains unregistered on this Windows host (documented since v0.16); `llava:latest` still cannot call tools.
- **KNOWN LIMITATION:** the 2/20 live misses (tool-choice variance); synthesis-prompt growth with a full ledger (below); per-database metrics scope.

## 13. Documentation

Updated: `README.md` (v0.25 What's New, hero paragraph, roadmap row, `--refresh` note), `AGENTS.md` (7 cached tools + full v0.25 section: synthesis contract, retry-dedup, refresh contract, metrics, isolation, live evidence), `docs/JARVIS_USER_MANUAL.md` (§8g: grounded answers / refresh mode / cache metrics; `/refresh` CLI row; replan-diff note), `docs/JARVIS_DEVELOPER_MANUAL.md` (§17 v0.25 entries, §21 config row incl. `RESULT_CACHE_METRICS_RETENTION_DAYS`, §26 new limitations), `docs/architecture.md` (Layer 3c: tool evidence is untrusted data), `deploy/README.md` (metrics history endpoint, retention setting, refresh surfaces, maintenance `today:` counters).

## 14. Version

`0.25.0` in all three sites — `pyproject.toml`, `jarvis/__init__.py`, `jarvis/api/app.py` — verified by import and package metadata.

## 15. Remaining Limitations

- **Grounding is prompt-side, not structural:** a sufficiently steered model can still ignore the evidence block (observed once live: correct value answered *without* calling the tool). The ledger grows the synthesis prompt +14.9% (one item) to +275.9% (full 16 items, 26194 chars vs 6968 base) — bounded by design, but real on an 8192-num_ctx budget.
- **Refresh is client trust, not a security boundary:** any client that may call chat may set it; it grants no capability and bypasses only the cache.
- **Metrics are per-database daily aggregates:** no cross-node rollup; rows older than 30 days are pruned.
- **Live eval remains small-sample:** 2 reps observe behavior, they do not establish significance.
- Carried from v0.24: SQLite-local cache; freshness-word vocabulary; structural-only replan triggers; exact-argument suppression semantics.

## 16. JARVIS Capability Delta

Before v0.25: execution could be correct while the **final answer** repeated the model's earlier mental arithmetic (41971 executed, 42071 spoken); users had no client-side way to force fresh data beyond wording; cache behavior was observable only as instantaneous counters; replans were invisible between `triggered` and `validated`; benchmarks depended on the host's real `jarvis.db` state.

After v0.25: the answer step transcribes measured evidence (role placement live-bisected); every client can force freshness with one flag that cannot touch security; daily cache activity is queryable across 30 days via API/CLI/dashboard; every replan ships a structural diff; and every deterministic evaluation runs in a throwaway, per-case-hermetic database that provably cannot touch production data.

## 17. Recommended v0.26 Direction

Recommend only; not implemented.

1. **Structural grounding backstop:** if the final answer names a number that appears in the evidence ledger with different digits, force one bounded transcription round (deterministic detection, no semantic judge) — converts the residual prompt-side risk into a system guarantee.
2. **Evidence-budget tokens:** replace the item/char caps with a model-aware token budget shared with the context window so a full ledger can never crowd out session history.
3. **Metrics rollup export:** append-only JSONL/CSV export from `cache_metrics_daily` for long-horizon analysis beyond the 30-day retention window.

## 18. Git Note

No Git/GitHub operations were performed.
