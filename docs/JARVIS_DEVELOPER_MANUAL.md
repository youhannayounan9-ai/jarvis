# JARVIS Developer Manual

> **For:** the project's owner/developer (AI engineer).
> **Repo state documented:** v0.16.0, branch `main`.
> **Companion:** [`JARVIS_USER_MANUAL.md`](JARVIS_USER_MANUAL.md) (operating JARVIS), [`architecture.md`](architecture.md) (safety-model detail), [`deploy/README.md`](../deploy/README.md) (deployment + sandbox verification).

---

## 1. Architecture Overview

```
   CLI (main.py) ─┐
   API (app.py)  ─┼─→ JarvisRuntime  (the single assembly seam)
   Dashboard     ─┘         │
                            ├── SessionStore (SQLite WAL)
                            ├── VectorStore  (ChromaDB long-term memory)
                            ├── ToolRegistry (11 tools + gated 12th)
                            ├── PermissionGuard
                            └── Orchestrator
                                   │
              ┌────────────────────┼─────────────────────────┐
              ▼                    ▼                         ▼
        route_intent()      Planner (complex only)      ContextManager
        simple | complex          │                          │
                            ReAct loop per step                │
                            (tools + permissions)              │
                                   │                           │
                            observation persisted ──→ context assembled
                                   │                           │
                                   └──────────→ LLM (LiteLLM → Ollama)
                                                               │
                                                        final synthesis
```

The seam rule: **interfaces never wire components themselves**. CLI, API, and dashboard all call `build_runtime()`; adding a tool or changing assembly happens in exactly one place (`jarvis/runtime.py::_TOOL_FACTORIES`).

---

## 2. Request Lifecycle (one real trace)

Example: *"Search for the population of Tokyo, then calculate what percentage of Japan's population that is."*

1. **Entry** — CLI `runtime.chat(session_id, text)` (or `POST /chat` → same). `JarvisRuntime.chat` acquires a **per-session lock** (non-blocking; `TimeoutError` → HTTP 409) and delegates.
2. **Persistence** — orchestrator saves the user message via `SessionStore.add_message` (SQLite WAL, full fidelity).
3. **Routing** — `route_intent()` (heuristic, zero-cost): multi-step signals ("then", " and then ", "write a", ≥3 sentences, …) → `complex`; short single-intent inputs → `simple`. Our example matches "then" + length → **complex**. Trade-off documented in AGENTS.md: FPs (complex queries with simple keywords) and FNs (simple queries missing keywords) are accepted for zero latency.
4. **Planner** — `Planner.generate_plan()` asks the planner model for a JSON array of steps; output is fenced-block-stripped, JSON-parsed, normalized, **unknown `required_tools` filtered against the real registry**, and capped at `MAX_PLAN_STEPS=5`. Parse failure → single-step fallback plan (the raw request).
5. **ReAct execution per step** — `_run_react()` loops: LLM call (with tool schemas + bounded context) → tool_calls? → parse args → validate → permission decision → dispatch → observation → persist → next round. Text with no tool calls ends the step.
6. **Argument validation** — each tool's JSON Schema was compiled to a Pydantic model at class definition (`__init_subclass__` → `_build_pydantic_model`, `extra="forbid"`); dispatch validates before execution and returns structured `ERROR:` strings on failure.
7. **Permissions** — `_dispatch_with_permissions_async()` checks `REQUIRE_CONFIRMATION_FOR_HIGH_RISK` + `guard.require_confirmation()` (SYSTEM/DESTRUCTIVE → park durable confirmation + `PAUSED_FOR_CONFIRMATION`), then `guard.is_allowed()` (SAFE/NETWORK/FILE_READ pass; FILE_WRITE → blocked string).
8. **Execution** — `ToolRegistry.dispatch_async()` runs `tool.run_async()` under `asyncio.wait_for(timeout=tool.timeout_seconds)`; result is a plain string ("ERROR: …" on failure — tools never raise through the loop).
9. **Observation** — the full tool result is persisted to SQLite (audit); the **prompt-side copy is clamped** by `ContextManager.clamp_tool_output` (6000 chars, head+tail kept).
10. **Budgets** — 5 tool rounds/request, 2/step, +2 self-correction rounds; exceeding them ends the step and the loop breaks.
11. **Synthesis** — `_synthesize()` makes one final **tool-free** LLM call over bounded context + clamped step results to produce the user answer.
12. **Persistence** — assistant message saved; response returned to the interface.

Simple path: skip steps 4 (planner) — a single ReAct step with the raw request, 2-round budget.

---

## 3. Execution Modes

| Mode | When | Flow |
|---|---|---|
| **simple (fast path)** | `route_intent()` says simple: short inputs with one simple-intent keyword | 1 ReAct step, budget 2 rounds, then straight to synthesis |
| **complex (planned)** | Multi-step signals, ≥3 sentences, or default | Planner → steps → ReAct per step (2/step) → synthesis |
| **ReAct loop** | Inside any step | LLM → tool_calls → observations → repeat, bounded |
| **Self-correction** | Tool returns an "ERROR: …" string | Up to `MAX_SELF_CORRECTION_ATTEMPTS=2` extra rounds with corrective feedback; persistent failure ends the step honestly |

Bounded autonomy is deliberate: a confused model cannot loop forever or burn the context window. Total LLM calls per complex turn are bounded by plan size (5) × rounds (2) + correction (2) + synthesis.

---

## 4. Orchestrator

- **Responsibilities:** routing, planning, ReAct execution, permission gatekeeping, durable pause/resume, synthesis, persistence, event emission (`on_event` seam for SSE).
- **State:** effectively stateless across turns except `_current_mode` (captured into pause contexts) and the injected store/registry/guard/planner/context. All *durable* state lives in SQLite.
- **Control flow:** `chat()` → persist user msg → route → complex: planner + per-step loop; simple: one step → `_run_react` → on `PAUSED_FOR_CONFIRMATION`: emit "paused" event and return the marker → else `_synthesize`.
- **Pause marker:** `PAUSED_FOR_CONFIRMATION` ("ACTION_REQUIRES_CONFIRMATION: …"). Interfaces surface it verbatim; the turn is *paused, not finished*.
- **Continuation:** see §8.
- **Budgets:** `MAX_TOOL_ROUNDS=5`, `MAX_TOOL_ROUNDS_PER_STEP=2`, `MAX_SELF_CORRECTION_ATTEMPTS=2`, `MAX_PLAN_STEPS=5`.
- **Error handling:** LLM exceptions bubble to interfaces (CLI catches, API → sanitized 503 with request-ID correlation); tool errors are strings handled in-loop via self-correction; observer exceptions are swallowed (`on_event` can never break execution).

---

## 5. Planner

- **Plan generation:** planner model (separate `PLANNER_MODEL` env, default same model) receives the system prompt + user request + context + the **real tool-name list**; must return a pure JSON array of `{step_number, description, required_tools}`.
- **Tool filtering:** hallucinated tool names are filtered against the actual registry when registry names are known. Empty `tool_names` means "planner used standalone without registry info" — plans pass through untouched (documented semantics).
- **Unknown-tool protection:** a step whose tools are all hallucinated degrades to a pure-reasoning step; guaranteed tool-error steps can't occur.
- **Step execution:** the orchestrator runs steps sequentially; each step description is self-contained (planner rule 6: references resolved into the step text because the executor re-reads descriptions cold).
- **Safety:** parse failure/invalid shape/empty → single-step fallback; >5 steps truncated + renumbered.

Regression tests: `tests/test_planner.py` (bounded plans, unknown-tool filtering, empty-names semantics, fence stripping, fallback).

---

## 6. Tool Architecture

- **BaseTool:** abstract base enforcing `name`, `description`, `parameters` (JSON Schema), `run()`. `__init_subclass__` validates risk levels/timeouts and **compiles the JSON Schema into a Pydantic model** (`extra="forbid"`).
- **Schemas:** `to_openai_schema()` renders OpenAI function-calling format; this list goes to the LLM with every call.
- **Registry:** manual registration (no auto-discovery — traceability over cleverness). `dispatch()` (sync, ThreadPoolExecutor + `future.result(timeout)`) and `dispatch_async()` (`asyncio.wait_for`) both enforce: unknown tool → error, JSON decode → error, **Pydantic validation → structured error**, timeout → error, exception → error string. No path bypasses validation.
- **Sync/async:** `run_async()` defaults to `asyncio.to_thread(self.run)`; tools may override for native async IO.
- **Error handling:** tools *return* "ERROR: …" strings rather than raising; the orchestrator's self-correction loop reads them; the LLM sees them as observations.

Concrete lifecycle — `calculator` with `expression="12 * (4 + 3) / 2.5"`:
1. Model emits `tool_calls: [{name: "calculator", arguments: "{\"expression\": \"...\"}"}]`.
2. Orchestrator → `_dispatch_with_permissions_async` → risk `SAFE` → allowed.
3. Registry validates args against the compiled model → runs `CalculatorTool.run`.
4. Restricted AST walker evaluates (no `eval()`; exponent size-guarded).
5. `"Result: 34"` returned → persisted + clamped → next ReAct round.

---

## 7. Permissions

- **Risk levels:** `SAFE, NETWORK, FILE_READ, FILE_WRITE, SYSTEM, DESTRUCTIVE` (Literal-validated at class creation).
- **PermissionGuard:** `require_confirmation()` returns True for SYSTEM/DESTRUCTIVE; `is_allowed()` returns True for SAFE/NETWORK/FILE_READ only — **FILE_WRITE returns False → blocked outright** (refusal string, not parked). This matches README's "file writes blocked" claim exactly.
- **Confirmation:** gated additionally by `settings.REQUIRE_CONFIRMATION_FOR_HIGH_RISK` (default True).
- **Durable state:** parking writes `pending_confirmations` (SQLite) with args, tool_call_id, risk level, `context_json` resume context, `expires_at` (TTL 10 min); `complete_pending_confirmation` atomically claims the row.

---

## 8. Confirmation State Machine

```
            ┌─────────────────────────────────────────────────────────┐
            │                    (restart any time)                   │
            ▼                                                         │
  requested → pending (SQLite: args, risk, context_json, expires_at) │
                 │                                                   │
        approve ─┴─ deny                                               
            │        │                                                
            ▼        ▼                                                
        execute     skip                                                
            │        │                                                
            ▼        ▼                                                
        persist result + outcome note                                  
            │                                                          
            ▼                                                          
        resume: restore paused plan/context, run remaining steps       
            │       (nested confirmation → re-park + return marker)    
            ▼                                                          
        synthesize final answer (failures visible to synthesis)        
```

Restart behavior: state is entirely in SQLite. Kill the process after "pending"; restart; `handle_confirmation` loads the row, and the resume path (`_resume_paused_turn`) restores `original_request`, `pending_plan`, `completed_steps`, `remaining_rounds`, and mode from `context_json`. Legacy rows without context (pre-v0.15) degrade to a plain reply; corrupt `context_json` degrades safely; expired rows (TTL 10 min) are refused. Covered by `tests/test_confirmation_continuation.py` + `test_confirmations.py` + `test_security_critical.py`.

### Execution ledger (v0.17) — at-most-once automatic dispatch

Every protected execution now carries a durable identity and state machine in SQLite (`action_executions`, paired 1:1 with its `pending_confirmations` row inside one transaction):

```
        action_id = uuid4 (server-generated, survives restart)
                          │
  confirmation parked ────┴──→ PENDING
                                 │
              approve ┌──────────┼───────────── deny/supersede ──→ FAILED (attempt stays 0)
                      ▼          ▼
                claim (atomic UPDATE on state='PENDING')
                      │ winner: PENDING→RUNNING, owner stamped
                      │ losers: "already_running" refusal / terminal report
                      ▼
                registry.dispatch  (the ONLY path that reaches the tool)
                      │
           ok ────────┴─────── error
           ▼                    ▼
       SUCCEEDED              FAILED  (result durably recorded)

  crash between dispatch and result record ──→ UNKNOWN (startup sweep RUNNING→UNKNOWN)
```

Semantics that matter:

- **PENDING** = eligible to execute. **RUNNING** = claimed and dispatching. **SUCCEEDED/FAILED** = outcome durably recorded (repeated approvals return the recorded result, never re-dispatch). **UNKNOWN** = the side effect may or may not have happened — *never* automatically re-executed; `handle_confirmation` returns an `ACTION_EXECUTION_STATE_UNKNOWN` report directing the user to resolve explicitly (e.g., re-issue the request as a new action).
- The claim is a single conditional `UPDATE ... WHERE state='PENDING'` judged by rowcount — two processes, two threads, two requests: exactly one winner. Duplicate approvals after a terminal state return the recorded result; empty pop returns a duplicate-resolution report built from the last ledger outcome.
- A crash after claim but before result cannot strand RUNNING silently: the next startup sweeps RUNNING→UNKNOWN (`recover_unknown_action_executions`). During runtime, a dispatch exception transitions RUNNING→FAILED (a known error is a known result).
- **Honesty rule:** the ledger makes automatic dispatch at-most-once. It does NOT make the external side effect exactly-once — the tool's effect is not transactionally coupled to SQLite. UNKNOWN exists precisely because that gap is real.
- Legacy rows (pre-v0.17, no ledger row) fall back to the historical direct dispatch with a warning.
- **Introspection (v0.18, read-only):** `list_action_executions(state, session_id, limit≤500, newer_than)`, `count_action_executions_by_state`, `get_last_action_execution`, `get_session_lease`, `list_session_leases` (computes `active`), `count_session_leases`, `redact_owner` (`host:pid:component:rand` → `component:rand`). Deterministic newest-first ordering; every query LIMIT-bounded.
- **Reissue (v0.18, the only mutation of a terminal state's aftermath):** `request_action_reissue(action_id, request_id, max_reissues=3)` — one SQLite transaction that (1) refuses anything not exactly UNKNOWN, (2) returns early when the same `(request_id, action_id)` pair already reissued (idempotency, enforced by the `idx_action_reissues_idempotency` UNIQUE index even across processes), (3) enforces the per-original ceiling, (4) refuses when the session has an **active** pending confirmation (one per session; completed/expired leftovers are cleaned and their stranded PENDING ledger rows closed, mirroring `save_pending_confirmation`'s replace semantics), then inserts the NEW PENDING ledger row, its durable `pending_confirmations` row (`confirmation_id = reissue:{request_id}:{original_action_id}`, standard 10-minute TTL), and the `action_reissues` audit row. The original row is never modified. The new action resolves through the ordinary `handle_confirmation` path — approval claims it at-most-once exactly like any parked action. Logging: `action_reissue_requested` (API layer), `action_reissue_created` (store), `action_reissue_duplicate_request` (replay).
- **Full-context recovery (v0.19):** every park now captures the pause context onto the LEDGER ROW itself (`action_executions.pause_context_json`, same transaction as the PENDING row; `ALTER TABLE` migration for existing DBs). Reissue copies that context verbatim onto the new confirmation — plus `recovered_from_action` / `recovered_request_id` lineage markers — so `handle_confirmation`'s existing resume path runs unchanged: the recovered step is labeled `, recovered action` in `completed_steps`, remaining plan steps execute under the ORIGINAL budget, and final synthesis completes the task. When a reissued action goes UNKNOWN again, the context is recovered **transitively**: the copy walks back through `action_reissues` (bounded by the ceiling) to the chain origin's ledger row. Malformed/absent context degrades to the pre-v0.19 raw-result reply (logged as `reissue_context_corrupt_degrades`); resolution of a recovered action logs `recovered_action_resolved`. Recovery cannot skip validation: the only execution path is the normal claim + PermissionGuard + dispatch. The operator decision surface is `get_action_recovery_preview` (has_context / recoverable / original_request truncated to 300 / pending steps / budget — never the raw context).
- **Session timeline (v0.19):** `get_session_timeline(session_id, limit≤500)` merges `messages` (kind only, no content), `pending_confirmations`, `action_executions`, `action_reissues`, and `session_leases` (owner redacted) into one (ts, seq)-ordered, LIMIT-bounded list — the shared data layer behind `GET /sessions/{id}/timeline`, `maintenance inspect --session`, and the dashboard. Safe metadata only.
- **Retention (v0.19):** `cleanup_operational_records(terminal_actions_days=30, reissues_days=90, leases_days=30, dry_run=False)` — one transaction; deletes only SUCCEEDED rows past retention (plus FAILED rows whose confirmation is completed), never PENDING/RUNNING/UNKNOWN or rows referenced by `action_reissues`; audit rows go only when BOTH linked actions are gone; expired/orphaned leases purge. `dry_run=True` is a verified no-mutation count pass.
- **Personal knowledge base (v0.20):** `jarvis/memory/knowledge_parsing.py` (parsers: TXT/MD/code/JSON via stdlib, PDF via `pypdf` with page preservation; deterministic lossless paragraph-first chunker, 1200/150 defaults, `_MAX_PARAGRAPH_CHARS` hard-split guard, stable chunk ids `{document_id}:{index}:{hash16}`) and `jarvis/memory/knowledge.py` (`KnowledgeService`: explicit-path ingestion inside `FILE_READER_ALLOWED_DIR` with unconditional credential-file refusal; content-hash dedup — unchanged → skip, same bytes new path → shared entry, changed → delete-then-replace with no stale chunks; bounded retrieval with `MAX_TOP_K`=20 / `MAX_EVIDENCE_CHARS`=6000 and metadata-only citations). Documents live in the dedicated `knowledge_base` Chroma collection — NEVER in `long_term_memory`; the `search_knowledge` tool returns evidence inside explicit untrusted-data delimiters. Tests: `tests/test_knowledge_rag.py` (32 deterministic cases: lifecycle, PDF pages, citations, adversarial framing, separation, path security, API lifecycle over real HTTP).

```
  UNKNOWN (crash ambiguity, never auto-retried)
      │  operator inspects: /actions, unknown-actions, GET /actions/{id},
      │                    dashboard Operations → UNKNOWN actions,
      │                    maintenance inspect --action <id>
      │  operator decides: POST /actions/{id}/reissue  or  maintenance reissue
      │                    (dashboard: ack checkbox + final confirmation)
      ▼
  NEW action_id (PENDING, attempt 0) ─── audit row: original → new
      │                                   context copied from original row
      │                                   (transitively across the chain)
      └── normal flow: approve → claim → dispatch once → resume remaining
                       plan steps under the original budget → synthesis
                       deny    → FAILED without any dispatch (denial recorded)
                       (original UNKNOWN row remains untouched for audit)
```

Tests: `tests/test_action_idempotency.py` (24 cases: atomic claims incl. an 8-thread barrier race, duplicate/triple approval, approval-after-failure, UNKNOWN never re-dispatched, denial attempts 0, restart between park and approval, migration backfill), the concurrent-approval cases in `tests/test_session_concurrency.py`, `tests/test_operator_introspection.py` (73 cases: introspection filters/pagination/safe-metadata, lease staleness/redaction, full reissue lifecycle incl. idempotent replay, ceiling, audit chain, active-confirmation refusal, unauthenticated reissue → 401, wrong state → 409, doctor scenarios, CLI output/exit codes), `tests/test_full_recovery.py` (27 cases: context capture/survival, full-context reissue with real-orchestrator approval/denial/nested/second-UNKNOWN/budget/malformed/missing-session, timeline ordering/causality/bounds/isolation/leak-proofing, retention protection + dry-run + audit integrity), and `tests/test_operator_experience.py` (11 cases: the new endpoints on a REAL uvicorn server via the REAL client incl. auth, and dashboard-backend parity).

---

## 9. Memory Architecture

| Store | Tech | What lives there |
|---|---|---|
| Session state | SQLite (WAL) via `SessionStore` | Sessions, messages (full fidelity incl. tool payloads), pending confirmations, execution ledger (`action_executions`), session leases (`session_leases`), reissue audit (`action_reissues`), knowledge registry (`knowledge_documents`, v0.20), rate-limit events (`rate_limit_events`, only when the durable limiter is enabled) |
| Long-term semantic | ChromaDB via `VectorStore` (`all-MiniLM-L6-v2` local embeddings, cosine HNSW) | **Two collections:** `long_term_memory` — facts written by `remember_fact`, searched by `recall_facts` (top-3); `knowledge_base` (v0.20) — ingested document chunks with trace-back metadata, searched by `search_knowledge` (bounded top-k) |

- **Session vs long-term:** session history = current conversation window; long-term = cross-session facts tied to `current_session_id` metadata. `VectorStore` is a process-wide singleton; `JarvisRuntime.start_session()` binds each new session to it.
- **SQLite as the single source of truth:** everything the model "remembers" from the conversation is rebuilt from SQLite every turn (bounded by ContextManager); nothing is kept only in process memory.

---

## 10. Context Management

Why bounded: 8192-token `num_ctx` cap (OOM guard for Qwen) + token cost predictability.

- **Windowing:** last `MAX_CONTEXT_MESSAGES=24` messages.
- **Anchor:** the session's first user message is re-injected as a system message when it slides out of the window (goal preservation).
- **Rolling summary:** dropped turns → deterministic extractive digest (user intents, tool names used, abridged assistant notes) — deliberately *not* an LLM call (free, fast, cannot hallucinate).
- **Two-layer tool-output policy:** full payload **stored** in SQLite (auditability); **clamped copy in prompts** (6000 chars, head+tail with `…[N characters omitted]…` marker). Synthesis clamps step results the same way. Idempotent: already-truncated payloads pass through.
- **Context floor:** `max(1, …)` prevents the classic `history[-0:]` full-history bug.

---

## 11. LLM Layer

- **Ollama** serves local models; **LiteLLM** abstracts the provider (one model-string change to swap providers).
- **Wrapper** (`jarvis/llm/client.py`): single call site injecting `api_base`, `max_tokens`, `num_ctx=8192`, `drop_params`, `tool_choice="auto"`. Sync `chat_completion`; the API layer bridges to async via threads.
- **Models:** main = `OLLAMA_MODEL` (default **qwen2.5:7b**), planner = `PLANNER_MODEL`, vision = hardcoded `ollama_chat/llava` in `vision_analyze.py`.
- **Untrusted by design:** every model output is treated as untrusted data — tool arguments are JSON-parsed + schema-validated before any execution, tool names are checked against the registry, and no model string ever reaches `exec`/`eval`/shell. The model can only *request* actions; the permission system decides.

---

## 12. Voice

- **STT:** `sounddevice` records `VOICE_RECORD_SECONDS` from the default mic → 16-bit PCM WAV tempfile → **local Whisper** (`WHISPER_MODEL`, `base` default) transcribes (`fp16=False`); ffmpeg must be on PATH (explicit check with a clear error). `is_ready=False` → voice mode aborts gracefully to text.
- **TTS:** text cleanup (strips code blocks/Markdown) → **Edge TTS** neural voice (**network required**) → mp3 → `ffplay` playback. Playback/synthesis failures are logged and swallowed so the voice loop survives.
- **Boundary:** voice is just another transport into `orchestrator.chat()` — identical permission/confirmation behavior; exit phrases handled locally ("exit", "quit", "stop", "goodbye").

---

## 13. Vision

`vision_analyze(image_path)` → resolve + containment check against `FILE_READER_ALLOWED_DIR` (traversal refused) → base64-encode → direct `litellm.completion(model="ollama_chat/llava", …)` with a text+image_url message → description returned as the tool result.

Notes: it calls llava **directly** (not via the main model), so no tool schemas/context go to llava; it works in every interface because the tool lives in the shared registry; requires `ollama pull llava`.

---

## 14. Web Security (prompt injection)

Trace: `web_search`/`web_scrape`/`wikipedia_summary` results → returned as **tool result strings** → persisted → clamped copy enters the next prompt as **observations**.

- **Untrusted data, not instructions:** the system prompt says so; but the *real* boundary is structural: no text in a tool result can register a tool, change risk tiers, execute code, or flip `REQUIRE_CONFIRMATION_FOR_HIGH_RISK`. Privileged actions pass only through PermissionGuard + durable confirmation.
- **Precise limitation:** prompt-level rules are **defense in depth, never the security boundary**. A crafted page can still try to steer a 7B model's behavior within the tools it may already call (e.g., trick it into scraping a URL). The live injection eval case passes, but that is not a guarantee.
- **No re-tooling risk:** the tool list is rebuilt server-side each call from the registry; model-mentioned tools that aren't registered cannot dispatch (registry returns "Unknown tool").

---

## 15. Docker Sandbox Architecture

Three timeout layers:

1. **Container workload timeout** — the entrypoint is `timeout <cap>s python3 script.py`; a runaway snippet is killed *inside* the container (exit 124). This is the primary enforcement point (the workload boundary).
2. **Host subprocess timeout** — `subprocess.run(..., timeout=cap+5)` backs the host out of a hung CLI/daemon call.
3. **Force removal** — if the host wait fires, `docker rm -f <container>` guarantees no orphan container keeps consuming resources. `ExecutionResult.timeout_layer` reports `container` / `host_kill` / none.

**Per-run container identity (v0.17):** every execution runs in a container named `jarvis-sbx-<12-hex-random>` (prefix `DockerCodeSandbox.CONTAINER_NAME_PREFIX`), generated server-side and reported on `ExecutionResult.container_name` plus the `sandbox_container_start` / `sandbox_container_done` log events. The v0.16 fixed name (`jarvis-sbx-exec`) serialized all executions on a host and made cleanup ambiguous under concurrency; force-removal now targets exactly the named container, so a timed-out run can never affect another run's container.

Every restriction and its reason:

| Restriction | Why |
|---|---|
| `--network none` | Code must not reach the host network or internet |
| `--read-only` rootfs | No persistence, no binary planting |
| `--cap-drop ALL` + `no-new-privileges` | No privilege escalation paths |
| `--user 65534:65534` | Non-root by construction (integration-verified uid) |
| `--memory 256m --memory-swap 256m` | RAM exhaustion can't OOM the host |
| `--cpus 5` (quota) | CPU burn can't starve the host |
| `--pids-limit 64` | Fork-bomb containment (integration-verified) |
| tmpfs `/tmp` 16 MB `noexec,nosuid,nodev` | Scratch space without execution/persistence |
| code mounted `:ro` | The script can't rewrite itself |
| `--rm` | One-shot; nothing lingers |
| per-run random name (`jarvis-sbx-*`, v0.17) | Concurrent executions isolated; cleanup targets exactly one container |

**Seccomp posture (v0.17, CONFIRMED against the live engine):** JARVIS does not weaken, replace, or opt out of Docker's seccomp filtering — no `--security-opt seccomp=unconfined`, no custom profile is passed. Containers therefore run under the daemon's **builtin default seccomp profile**, verified live: `docker info --format '{{json .SecurityOptions}}'` → `["name=seccomp,profile=builtin","name=cgroupns"]`. The builtin profile blocks ~44 of ~300 syscalls (including `keyctl`, `ptrace`, `kexec_load`, mount, and reboot families) and returns `EPERM` for ~26 more. This is default-profile containment on top of `--cap-drop ALL` + `no-new-privileges`; a custom minimized profile remains future work (see §26).
| fixed container name | Prevents parallel-run pileups on one host |

**Digest pinning:** `deploy/Dockerfile.sandbox` FROM-pins `python:3.12-slim@sha256:f77ac9e4…` — resolved by an actual `docker pull` against a real Linux daemon (v0.16) and re-verified by digest-ref pull + build. Never invent a digest; the build script refuses the placeholder.

**What v0.16 verified with a real Linux daemon (CONFIRMED):** 25 integration tests (`tests/test_sandbox_integration.py`) asserting *observed* behavior from inside containers: container identity, non-root uid, `setuid(0)`/`chown(0,0)` refused, read-only rootfs, read-only code mount, `/tmp` writable-but-noexec, no outbound network, container-local `/proc`, PIDs ceiling bounding forks, memory-cap OOM kill, workload-killing timeout (exit 124, no orphan), layer-C force removal, output caps, per-run tmpfs isolation, end-to-end `DockerCodeSandbox.execute()`.

**Windows behavior:** `is_available()` fails closed on `win32` by design — even with Docker Desktop's Linux engine running (observed in v0.16). The tool stays unregistered; run JARVIS under WSL2/Linux for code execution.

---

## 16. API Architecture

- **FastAPI** app (`jarvis/api/app.py`); runtime built lazily and swappable (`set_runtime`) for tests.
- **Sessions:** just IDs — all state in SQLite, so the API layer is stateless.
- **Auth:** optional key (`JARVIS_API_KEY`); Bearer or X-API-Key; exempt: `/health`, `/docs`, `/openapi.json`, `/redoc`; constant-time `hmac.compare_digest`; whitespace-only key → 503 fail-closed.
- **Rate limit:** sliding window (`api/ratelimit.py`), keyed by API key or client IP; 429 + Retry-After; /health exempt. Two interchangeable backends behind one class (v0.17): the default in-memory deque (per-process) and — opt-in via `make_durable_limiter(store)` — an SQLite-backed store where each check is one `BEGIN IMMEDIATE` write transaction, so multiple JARVIS processes on the same database enforce ONE limit per client (bounded growth: expired events deleted in-transaction). The deployment default remains in-memory; see deploy/README.md.
- **Request IDs:** middleware generates per-request UUID → `X-Request-ID` header, echoed in `/chat` bodies; errors are sanitized (generic message + correlation ID; details only in logs).
- **SSE:** `POST /chat/stream` runs the turn in a worker thread; lifecycle events flow through a queue to the client *as they happen*; 15s keepalive comments defeat proxy idle timeouts; in-band `error` events; `X-Accel-Buffering: no`.
- **Health:** `/health` = liveness + **deep DB read/write probe** (fail-closed 503 on persistence failure; probe rows self-clean).
- **Per-session serialization (v0.17: two layers):** one turn per session at a time; second concurrent turn → 409. Layer 1 is the runtime-local per-session mutex (fast path). Layer 2 is a **database-backed session lease** (`session_leases` table, TTL `SESSION_LEASE_TTL_SECONDS`=300 s, monotonically increasing fencing token bumped on stale takeover): cross-process exclusivity over the same SQLite file, no Redis required. A crashed owner's lease expires (no indefinite lock); releases/renewals are owner-checked; the documented residual race is a single turn that outlives the TTL near the boundary.
- **Client:** `jarvis/api/client.py` — stdlib-only (`urllib`), typed errors (`SessionConflict`, `RateLimitedError`), SSE iterator; used by the dashboard and scripts.

---

## 17. Evaluation Architecture

- **Cases:** 32, across 10 categories (tool_selection, refusal, instructions, ambiguity, continuity, multi_step, no_tool, recovery, robustness, injection), each defining a prompt + expected tools/content + grader set. Case identity = content digest (survives renames; excludes name).
- **Graders:** 12 deterministic/lexical functions; category breakdown per run. **LEXICAL EVALUATION IS NOT SEMANTIC PROOF** — no LLM judge exists; graders are a regression tripwire, documented as such everywhere.
- **Reports:** schema v2 JSON (`--json`): host/model info, per-case `{case_id, expected, tools_called, passed, grader_results, duration, failure_reason/failure_detail}`, category pass-rates, timeout count. Console output stays ASCII-safe (cp1252-safe; non-ASCII → escaped).
- **Failure classification:** `failure_reason` distinguishes grader failures (with expected-vs-actually-called tools) from budget exhaustion (no tools observed).
- **Timeout semantics (honest):** per-case timeout = daemon thread + `join(timeout)` — an **abandonment timeout**: the harness stops waiting and marks the case timed out, but the underlying model call is NOT terminated. Documented as such; acceptable for eval because the next case proceeds. (The sandbox, by contrast, truly terminates workloads.)
- **`--compare PRIOR_JSON`:** regression diff — newly failing / newly passing / unchanged failures, category changes, rename-surviving (digest-keyed). Proven on real v0.15→v0.16 runs.
- **Preflight:** Ollama unreachable → clean exit 2 before any case runs (no 32-failure cascade).

Live v0.16 result (qwen2.5:7b, real): **23/32 (71.9%), 0 timeouts**. Remaining failures classified: model variance/planning ×6, grader weakness ×2, (1 former fabrication defect fixed).

---

## 18. Testing Strategy

| Layer | Files | What it proves |
|---|---|---|
| Unit (mocked) | `test_tools.py`, `test_permissions.py`, `test_planner.py`, `test_context_manager.py`, `test_orchestrator.py`, `test_memory_tools.py`, … | Construction, validation, routing, fallbacks — fast, deterministic |
| Security-critical | `test_security_critical.py`, `test_confirmations.py`, `test_confirmation_continuation.py` | Fail-closed paths: unregistered tools, FILE_WRITE block, confirmation lifecycle, restart-safety |
| API | `test_api.py`, `test_api_security.py` | Endpoints, auth, rate limit, 409, sanitized errors, SSE events |
| Docker integration | `test_sandbox_integration.py` | **Observed** container behavior against a real Linux daemon (skips cleanly with reason when prerequisites are missing — a skip is never a pass) |
| Sandbox unit | `test_docker_sandbox.py`, `test_sandbox_timeout_layers.py` | CLI-arg construction, validation ladder, fail-closed |
| Eval harness | `test_eval_harness.py` | CLI, reports, `--compare`, cp1252 safety, grader audit |
| Voice | `test_voice.py` | TTS cleanup, voice-loop behavior (mocked audio) |
| Deploy readiness | `test_deploy_readiness.py` | Compose/Dockerfile/docs consistency |
| Full suite | `uv run pytest -q` | 346 passed + 1 honest skip (Windows-host availability gate) as of v0.16 |

Why both mocked and real: mocks give fast, deterministic, always-green construction evidence; the Docker integration suite gives **observed** boundary evidence that mocks cannot. Neither substitutes for the other.

---

## 19. CI/CD

`.github/workflows/ci.yml`:

- **PR evaluation environment:** GitHub-hosted ephemeral runners (tests matrix 3.11/3.12; api-image build; sandbox-image build + smoke run now that the digest is pinned).
- **Scheduled nightly eval:** `schedule`-only job on `[self-hosted, ollama]` runners — untrusted PRs never execute there (no `pull_request` trigger reaches it).
- **Permissions:** top-level and job-level `contents: read`; `persist-credentials: false` on checkouts.
- **CODEOWNERS:** required review for workflow/deploy/container changes (@youhannayounan9-ai).
- **Docker trust boundary:** the sandbox image build job never runs untrusted PR code on the self-hosted runner; sandbox runtime containers get no docker socket.

Remaining trust assumptions: the self-hosted runner machine itself and its network position; the Ollama models installed there; GitHub's control plane.

---

## 20. Deployment

- **Compose:** `api` service (non-root container, healthcheck, log rotation, 90s stop grace, `jarvis-data` volume for SQLite+Chroma) + optional `local-llm` profile Ollama service. Maintenance runs against the same volume: `docker compose run --rm api python -m jarvis.maintenance stats|cleanup --days 30`.
- **App image** (`Dockerfile`): python:3.12-slim + uv, non-root `jarvis` user, `/data` volume mount point, uvicorn CMD.
- **Sandbox image:** built separately via `deploy/build-sandbox-image.sh`; digest-pinned; must be pre-pulled on the daemon the API talks to (never pulled at runtime). Enabling docker.sock in the api container is a Linux-only, explicitly-accepted risk (documented in compose comments).
- **Single-replica assumption:** SQLite WAL + process-local locks/limits → run one API process (scale at the reverse proxy).

---

## 21. Configuration

All config flows through `jarvis/config.py` (pydantic-settings, `.env` in CWD, case-insensitive, extra vars ignored).

| Group | Keys (defaults) | Notes |
|---|---|---|
| LLM | `OLLAMA_BASE_URL` (localhost:11434), `OLLAMA_MODEL` (**qwen2.5:7b** code default; `.env.example` shows llama3.1:8b — keep consistent), `PLANNER_MODEL` (qwen2.5:7b), `MAX_TOKENS` (2048) | `num_ctx=8192` hardcoded-ish (config) OOM guard |
| Storage | `DB_PATH` (jarvis.db), `VECTOR_DB_PATH` (./jarvis_data/chroma_db), `MAX_CONTEXT_MESSAGES` (24), `MAX_TOOL_OUTPUT_CHARS` (6000) | |
| Files | `FILE_READER_ALLOWED_DIR` (`.`) | Sandbox boundary for read/list/vision |
| Voice | `WHISPER_MODEL` (base), `TTS_VOICE` (en-US-GuyNeural), `VOICE_RECORD_SECONDS` (5) | |
| Security | `REQUIRE_CONFIRMATION_FOR_HIGH_RISK` (true), `ENABLE_CODE_EXECUTION` (false), `SANDBOX_IMAGE` (ubuntu:24.04) | Sandbox needs a pinned ref; validator rejects mutable tags |
| API | `JARVIS_API_KEY` (""), `RATE_LIMIT_REQUESTS` (60), `RATE_LIMIT_WINDOW_SECONDS` (60) | Empty key = local trust; 0 disables limiting |
| Dashboard | `JARVIS_API_URL` (""), `JARVIS_CLIENT_API_KEY` ("") | Set URL → API-client mode |
| Logging | `LOG_LEVEL` (INFO) | structlog JSON |

Never print actual secrets; `.env` is git-ignored and docker-build-excluded.

---

## 22. How to Add a New Tool

1. **Implement** `jarvis/tools/my_tool.py`:

```python
from jarvis.tools.base import BaseTool

class MyTool(BaseTool):
    name = "my_tool"
    description = "What it does and WHEN to use it (the model reads this)."
    parameters = {
        "type": "object",
        "properties": {"input": {"type": "string", "description": "..."}},
        "required": ["input"],
    }
    risk_level = "SAFE"        # choose honestly: SAFE/NETWORK/FILE_READ/FILE_WRITE/SYSTEM/DESTRUCTIVE
    timeout_seconds = 15.0

    def run(self, input: str, **kwargs) -> str:
        return f"processed: {input}"   # return strings; "ERROR: ..." on failure — never raise
```

2. **Export** it in `jarvis/tools/__init__.py`.
3. **Register** in `jarvis/runtime.py::_TOOL_FACTORIES` (the single registration point — CLI, API, dashboard inherit it). If it needs a constructor arg, add a factory function instead.
4. **Permissions:** SAFE/NETWORK/FILE_READ auto-run. SYSTEM/DESTRUCTIVE get durable confirmations automatically; FILE_WRITE is blocked outright — only pick those deliberately.
5. **Tests:** happy path, validation error (bad args), unknown-risk/timeout behavior as applicable; see `tests/test_tools.py` for the pattern.
6. **Eval case:** add to `evaluation/run_evals.py` (see §24) so the live loop exercises it.
7. **CLI blurb** (optional): add to `_TOOL_BLURBS` in `jarvis/main.py` for `/tools`.

---

## 23. How to Modify the Planner Safely

The planner sits on the fail-safe path; changes need these regressions green:

- `tests/test_planner.py`: bounded plans (≤5 + renumbering), unknown-tool filtering (registry-known), empty-`tool_names` = pass-through (NOT "no tools allowed"), fence stripping, malformed JSON → single-step fallback.
- `tests/test_self_correction.py` + orchestrator tests: step execution over the planner output, budget interactions.
- `tests/test_confirmation_continuation.py`: planner changes must not break pause/resume (pending_plan is part of the durable context).

Rules of thumb: never trust planner JSON shape (fallback plan always); never let a plan step reference an unregistered tool; keep `MAX_PLAN_STEPS` small; keep step descriptions self-contained.

---

## 24. How to Add a New Evaluation Case

1. Add the case dict to `evaluation/run_evals.py` (`EVAL_CASES`): `name`, `category`, `prompt`, `expected_tools` (list), `expected_content` (strings), `graders` (from the existing 12).
2. Keep expectations **behavioral** (what a correct agent does), not lexical tics of today's model — graders are a tripwire, don't overfit.
3. Run `uv run python evaluation/run_evals.py --list` to confirm registration, then a filtered live run: `--filter "<name>"`.
4. Re-run the full suite live when you have Ollama available; record the report JSON; use `--compare` against the prior report.
5. If the new case exposes a real defect: fix the defect, not the grader (fixing the grader is only legitimate for grader bugs like the v0.16 context-budget fix).

---

## 25. How to Debug JARVIS

```
Is Ollama alive?                curl http://localhost:11434/api/version
        ↓ no → ollama serve; yes ↓
Is request entering JARVIS?     LOG_LEVEL=DEBUG; watch for llm_request / runtime_built
        ↓
Did routing happen?             DEBUG shows intent ("simple"/"complex")
        ↓
Did planner choose correctly?   DEBUG: plan_generated steps=N / planner_fallback_single_step
        ↓
Did tool validation pass?       tool_args_parse_error / tool_args_validation_error in logs
        ↓
Did permission allow it?        permission_allowed / permission_blocked / tool_requires_confirmation
        ↓
Did tool execute?               tool_dispatching → tool_success (elapsed) / tool_timeout / tool_exception
        ↓
Did context update?             add_message persisted; tool_output_clamped for big results
        ↓
Did synthesis work?             response_ready with duration_ms in logs
```

Extra levers: `python -m jarvis.maintenance doctor` (DB, Ollama + model availability, stale leases, UNKNOWN/PENDING/RUNNING action counts, sandbox posture, pending confirmations, voice deps — inspect/diagnose only, no silent repairs); `/history` in the CLI; API `X-Request-ID` correlation; `GET /sessions/{id}/history` for the exact persisted turns.

### Maintenance CLI (v0.18)

```
uv run python -m jarvis.maintenance actions --state UNKNOWN --json
uv run python -m jarvis.maintenance unknown-actions        # + why + reissue guidance
uv run python -m jarvis.maintenance sessions --expired
uv run python -m jarvis.maintenance reissue --action <id> --request-id <unique-id> [--yes]
uv run python -m jarvis.maintenance doctor
cleanup / expire-confirmations / stats   (unchanged)
```

Contract: `actions`, `unknown-actions`, `sessions` are **strictly read-only** (owner tokens redacted, `tool_args`/result bodies never printed, `--json` for scripts); `reissue` is the one mutating command (pre-checks state → refuses non-UNKNOWN with exit 1; interactive `yes` prompt unless `--yes`; reports the idempotent-replay note when the request id was already served); `doctor` reports failures with a non-zero exit for alerting and logs a structured `doctor_check` event. Exit codes: 0 ok, 1 error/refused.

---

## 26. Known Architectural Limitations (v0.20)

- **Knowledge base is single-user and local by design:** one Chroma `knowledge_base` collection per `VECTOR_DB_PATH`; no ACLs, no multi-tenant scoping, no sync (ingestion is explicit-path, never crawling).
- **PDF parsing is text-layer only:** scanned/image PDFs yield empty pages (no OCR); complex layouts can garble reading order. `pypdf` extraction failures degrade to empty pages, never to fabricated content.
- **Chunking is character-based:** paragraph-first with 1200/150 defaults; no token-aware splitting and no semantic boundaries beyond markdown headings (`#` lines feed `section` metadata).
- **Retrieval is embedding-only:** pure `all-MiniLM-L6-v2` cosine; no keyword/hybrid stage, no reranker. Distance threshold filtering exists but is off by default; irrelevant-but-ranked results are possible (the tool's honest `NO_RELEVANT_EVIDENCE` reply covers the true-empty case only).
- **Evidence is bounded, not tokenized:** `MAX_EVIDENCE_CHARS`=6000 caps the injected block deterministically; a token-exact budget is future work.
- **Prompt-injection defense is framing + tool discipline:** the `DOCUMENT EVIDENCE` delimiters and system-prompt rules tell the model to treat document text as data; the structural boundaries remain PermissionGuard, tool schema validation, and the execution ledger. A sufficiently steered 7B model reading malicious documents is a documented residual risk — never grant evidence paths to write/execute tools.
- **Test-isolation guard:** `tests/conftest.py` autouse `_offline_llm_guard` refuses `litellm.completion` in every test not marked `live_llm` — the Planner captures its client at construction, so name-patching `chat_completion` alone cannot stop a real network call (this caused an observed multi-minute hang when Ollama was wedged).
- **SQLite is the coordination substrate:** session leases, the execution ledger, the reissue audit trail, the knowledge registry, and the durable rate limiter are correct across processes sharing ONE database file, but SQLite writes serialize — high-write concurrency throughput is bounded. Not a distributed system: replicas on different files do not coordinate.
- **Lease TTL residual race:** a single turn that outlives `SESSION_LEASE_TTL_SECONDS` (300 s) can lose cross-process exclusivity near the TTL boundary (in-process mutex still holds). Turn durations are far below the TTL today.
- **UNKNOWN resolution is manual by design:** an action whose crash state is UNKNOWN is reported (`ACTION_EXECUTION_STATE_UNKNOWN`) and never auto-rerun. v0.18 adds inspection (API + CLI) and an explicit, idempotent, bounded reissue — still operator/user-driven, still no automatic recovery. No dashboard UI for curating UNKNOWN actions exists yet.

- **SQLite is the coordination substrate:** session leases, the execution ledger, the reissue audit trail, and the durable rate limiter are correct across processes sharing ONE database file, but SQLite writes serialize — high-write concurrency throughput is bounded. Not a distributed system: replicas on different files do not coordinate.
- **Lease TTL residual race:** a single turn that outlives `SESSION_LEASE_TTL_SECONDS` (300 s) can lose cross-process exclusivity near the TTL boundary (in-process mutex still holds). Turn durations are far below the TTL today.
- **UNKNOWN resolution is manual by design:** an action whose crash state is UNKNOWN is reported (`ACTION_EXECUTION_STATE_UNKNOWN`) and never auto-rerun. v0.18 adds inspection (API + CLI) and an explicit, idempotent, bounded reissue — still operator/user-driven, still no automatic recovery. No dashboard UI for curating UNKNOWN actions exists yet.
- **Durable rate limiter is opt-in:** the deployment default limiter stays in-memory; `make_durable_limiter(store)` exists for multi-process deployments (see deploy/README.md).
- **Lexical graders:** no semantic judge; pass ≠ correct.
- **Prompt-injection defense:** prompt rules are defense-in-depth only; the structural boundary is permissions/confirmation, and within allowed tools a 7B model can still be steered by crafted content.
- **Sandbox:** seccomp is Docker's builtin default profile (verified — no weakening, but also not minimized/custom); CPU limit is quota-based (not hard wall); container-name uniqueness is random 12-hex (collision-chance only, same as docker default naming); verification is single-machine.
- **Windows sandbox availability:** sandbox refuses win32 hosts even with a working Linux engine (verified); code execution requires WSL2/Linux.
- **FILE_WRITE blocked:** write_file registered but refused — writes need a deliberate design change (e.g., scoped workspace + confirmation), not just a flag.
- **Local-only evaluation signal:** nightly evals need the self-hosted Ollama runner; live results vary with model/hardware.
- **`.env.example` vs code default model mismatch:** llama3.1:8b (template) vs qwen2.5:7b (code default + evals) — see §21.
