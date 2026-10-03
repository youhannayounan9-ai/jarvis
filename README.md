# JARVIS

> A **Local-first, privacy-oriented, and free of paid model APIs by default** AI assistant — powered by Ollama. No cloud.

**JARVIS v0.30.0** — a **local-first agentic AI assistant with a closed, *inspectable*, *recoverable* reliability loop — grounded in your own documents, measurably better at choosing its tools, executing multi-step tasks in order, and knowing when to stop**. One agent runtime (Plan-and-Execute loop, 12 tools, permission tiers, *resumable* durable confirmations, context management, self-correction) behind a hardened **REST API (FastAPI)**: auth, rate limiting, per-session serialization, request-ID correlation, deep health checks, truly interleaved SSE streaming — deployed via docker-compose with WAL-backed persistence, a **runtime-verified** workload-timeout-enforced sandbox, and CI with a documented trust boundary. v0.20 adds a **personal knowledge base**: explicitly ingest your PDFs/Markdown/text/code/JSON into a dedicated Chroma collection (separate from personal memory), retrieve with `search_knowledge`, and get answers with real citations — with document text treated strictly as untrusted evidence, never as instructions. v0.21 adds **capability-aware tool selection** (deterministic contract + fast-path safety net + bounded calculator fallback; live correct-tool rate 0.688 → 0.812, zero fabrication). v0.22 adds **plan validation, a planner tool catalogue, exact-evidence flow between steps, and per-step tool enforcement** — live on qwen2.5:7b, "Calculate 893 × 47 and remember the result" now executes calculator → remember_fact in the exact order with the grounded result, every repetition. v0.23 adds **planning efficiency**: a per-turn duplicate-dispatch ledger that stops identical repeated tool calls while preserving legitimate repeats (different arguments, retry after failure, state-coupled reads), redundant plan-step detection, a deterministic plan-quality report, and an evaluation-only semantic plan judge. Computer control remains structurally disabled. v0.24 adds **cross-turn awareness + bounded recovery**: a result cache that reuses *read-only retrieval* across turns (explicit per-tool policies, TTL / file-mtime / knowledge-generation freshness, provenance headers, session isolation, freshness-word bypass), and **exactly ONE bounded replan** when a plan structurally fails — reusing completed work, staying inside the remaining tool budget, and telling the truth when something still could not be done. v0.25 closes the last grounding gap — the final answer is now synthesized **from the tool evidence itself**, never from the model's memory of it — and adds a client-controlled refresh, daily operations metrics, replan-diff telemetry, and fully isolated deterministic evaluation. v0.26 turns that instruction into a **system guarantee**: a deterministic **grounding guard** checks every final answer against the turn's trusted tool evidence; a high-confidence contradiction triggers exactly ONE transcription-only correction round, and if that still contradicts, the wrong answer is **withheld** and the authoritative value (with live/cache provenance) is reported instead — never silently accepted, never fabricated. v0.27 makes JARVIS **multimodal**: normalized text / voice / image requests converge on the SAME runtime (no separate agents, no second security model) — push-to-talk voice over LOCAL Whisper STT with an observable turn lifecycle, network-backed Edge TTS with a clean cancellation boundary, image input validated by content sniffing and routed through the existing `vision_analyze` tool (llava interprets; qwen reasons; vision output is an untrusted observation, never trusted evidence), all inside the same sessions, permissions, grounding and telemetry. v0.29 adds **personal integrations & workflow automation**: an opt-in, scope-enforced framework (calendar + tasks as deliberately small local-dev reference providers) where the model holds opaque account handles — never credentials — reads are safe and session-cached, and every external write runs EXACTLY ONCE through the existing confirmation parking, action ledger and read-back verification, with duplicate creation prevented by provider-side idempotency keys. v0.30 makes those integrations **really connectable**: a provider-neutral **OAuth 2.0 authorization-code + PKCE** flow with durable one-time `state`, a real callback endpoint, automatic **access-token refresh** with atomic rotation, honest `AUTHENTICATION_REQUIRED`/`REVOKED` states, provider-verified **account identity**, best-effort **revocation on disconnect**, and bounded **audit retention** — all inside the SAME integration boundary and single security model, with a new `provider_schedule` grounding policy that verifies event/task times, timezones, and statuses.

---

## ✨ What's New in v0.30.0

- 🔑 **Real OAuth, provider-neutral** — `jarvis/integrations/oauth.py` adds an authorization-code flow with **PKCE (S256)** on every attempt: the code verifier lives only server-side in a durable flow row, the challenge rides the authorization URL. `IntegrationProvider` is extended by an `OAuthIntegrationProvider` mixin; the orchestrator never sees provider-specific OAuth details (endpoints, scope strings, identity discovery stay in the adapter).
- 🎫 **Durable one-time state** — 256-bit cryptographically random `state` values stored **by SHA-256 hash** (`oauth_states` table; the raw value is never persisted, logged, or shown to the model), hard-bound to `(provider, session, display_label, scopes, redirect_uri)`, with expiry (`OAUTH_STATE_TTL_SECONDS`, default 600) and **atomic one-time consumption** — replay, expiry, and mismatch all fail closed.
- 🌐 **A real callback with no open redirects** — `GET /integrations/oauth/callback/{provider}` is deliberately **not** API-key-gated (it is visited by your browser): the one-time state IS the authenticator. The redirect URI is constructed from config (`OAUTH_REDIRECT_BASE_URL`), never from a provider-supplied parameter, and the response is a fixed local HTML page — never a redirect, never a token.
- ♻️ **Token lifecycle** — distinct access + refresh tokens with expiry; tokens are refreshed **before** an operation inside `OAUTH_REFRESH_MARGIN_SECONDS` (default 300) and rotated **atomically**, preserving account identity and scopes. A failed refresh becomes `AUTHENTICATION_REQUIRED`; a revoked grant becomes `REVOKED` and is **never retried**. A rotation that omits the refresh token preserves the stored one.
- 🪪 **Provider-verified identity** — after the exchange the account identity is discovered from the provider (introspection), never inferred from chat text; two accounts can never collide by label, and re-authorizing the same identity rotates the account **in place** (refusing to overwrite a legacy raw-credential account).
- 🧹 **Clean disconnect** — disconnect best-effort **revokes at the provider**, deletes the local row, purges that account's cached reads, and makes every later operation fail closed on the stale handle; reauthorization is the only way back.
- 🧾 **Bounded audit retention** — `INTEGRATION_AUDIT_RETENTION_DAYS` (default 90) plus a `maintenance integrations-audit` command (report-only by default; `--yes` deletes a bounded batch). Rows with state UNKNOWN/RUNNING are **always protected**; audit rows still carry only category/operation/op_id/idempotency-key/redacted summary.
- 🧮 **Schedule-aware grounding** — a new `provider_schedule` policy verifies an answer's **clock time, weekday, timezone offset, and task status** against labeled `date:`/`time:`/`timezone:`/`duration_minutes:`/`status:` evidence lines; a wrong time or status triggers the existing one-round correction/fallback. Read-pre-state observations are exempt so state-change answers are not false-rejected.
- 🖥️ **Every surface** — API `POST /integrations/{provider}/authorize`, `GET .../authorize/status`, `POST /integrations/accounts/{id}/refresh`, callback; CLI `/integration-connect` (OAuth path), `/integration-reauthenticate`; dashboard OAuth connect expander + expiry + Re-authenticate/Refresh; `GET /integrations` reports `supports_oauth`. **No execute endpoint** exists — the chat runtime remains the only execution path.
- 🔐 **Tokens are never leaked** — they do not appear in prompts, tool output, evidence, cache payloads, API responses, logs, audit rows, or exception traces; `repr`/`__eq__` exclude token fields and `public_metadata()` emits only safe `authorization_*` keys. Redaction is test-pinned.
- 🧪 **83 new deterministic tests** (`tests/test_oauth.py`, categories A–V/AA/AC: state lifecycle, replay, PKCE, redirect/provider/session mismatch, exchange failure, expiry/refresh/rotation, revoked grant, scope downgrade, identity, redaction, disconnect, injection, cross-account cache, audit retention, labeled-field grounding, API auth boundary) — full suite **1218 passed / 29 skipped**. Manual-only deterministic live eval (`live_oauth_eval.py`, no Ollama/network): **18/18 steps**, report `live_oauth_report_v030.json`. Security model: `docs/JARVIS_V030_SECURITY_MODEL.md`.

## ✨ What's New in v0.29.0

- 🔗 **Personal Integrations & Workflow Automation** — a secure, extensible integration framework (`jarvis/integrations/`) with calendar + tasks as deliberately small reference implementations: 8 narrow tools (`calendar_list_events`, `calendar_get_event`, `calendar_create_event`, `calendar_update_event`, `calendar_delete_event`, `task_list`, `task_create`, `task_complete`) — **opt-in** via `ENABLE_INTEGRATIONS=true` (default off).
- 🪪 **Explicit connected-account identity** — `IntegrationManager` is the ONLY path to a connected account, enforcing in order: provider registered → provider available → LIVE auth state (EXPIRED/REVOKED refuse) → exact scope → operation. The model holds opaque `account_id` handles, NEVER credentials; scopes are bound at CONNECT time by the user and the model has no tool to change them.
- 🔐 **Honest credential boundary** — credentials are XOR+base64-obfuscated at rest (per-install key file next to the DB) and documented as a local-dev privacy guard, not hard security; they never appear in prompts, tool output, logs, session memory, or action descriptions; every provider field is sanitized and framed as UNTRUSTED PROVIDER CONTENT (reusing the browser injection detector).
- ✍️ **Writes run EXACTLY ONCE** — fully-valid writes escalate to SYSTEM risk ⇒ the EXISTING durable confirmation parking with a full preview (title, date, time, timezone, duration, attendees, side effect); the action ledger claims at-most-once; ambiguity (TIMEOUT/AMBIGUOUS_OUTCOME) ⇒ UNKNOWN + audit row, never a silent re-execution; reissue reuses the existing confirmation machinery with a provider idempotency key so a replay returns the ORIGINAL resource.
- ✅ **Deterministic verification** — every create/update is verified by read-back comparison (UTC-instant normalization) and reported as `verification: VERIFIED` / `ACTION_NOT_VERIFIED` — an unverifiable write never reads as success.
- ♻️ **Cache- and grounding-aware** — the 3 read tools declare session-scoped CachePolicies (`RESULT_CACHE_INTEGRATION_TTL_SECONDS`, default 60 s, per-account keys); writes are NEVER cached; `refresh=True` bypasses via the existing v0.25 client control; list outputs render a line-anchored `count:` the v0.26 grounding guard can deterministically check.
- 🚫 **Honest capability boundaries** — email/messaging is DECLARED-NOT-IMPLEMENTED (the tool policy truthfully refuses email requests); there is NO arbitrary HTTP request tool and the model can never supply endpoints, headers, or auth values; deletes stay behind HIGH-risk confirmation; connect/disconnect is human-only via API/CLI/dashboard.
- 🖥️ **Surfaces mirror the browser pattern** — API `/integrations*` endpoints (account management ONLY — deliberately NO HTTP execute endpoint), CLI `/integration*` commands, dashboard Integrations section; providers ship local-dev only (`LocalCalendarProvider`, `LocalTasksProvider`, `production_like=false` surfaced by the API).
- 🧪 **103 new deterministic tests** (`tests/test_integrations.py`, categories A–AD: auth states, exact scope enforcement, injection-shaped provider data, idempotent reissue, UNKNOWN crash-window, grounding `count:`) — full suite **1135 passed / 29 skipped**. Manual-only live eval (`live_provider_eval.py`) against local dev providers. Security model: `docs/JARVIS_V029_SECURITY_MODEL.md`; final report: `docs/JARVIS_V029_FINAL_REPORT.md`.

## ✨ What's New in v0.28.0

- 🌐 **Safe Browser & Computer Interaction** — the computer-control placeholder becomes a real, policy-controlled browser layer: nine narrow tools (`open_url`, `get_page_state`, `extract_visible_text`, `take_screenshot`, `click_element`, `fill_input`, `select_option`, `go_back`, `wait_for_element`) behind a deny-by-default URL policy, observation-freshness gates, deterministic pacing limits, and OBSERVE→ACT→VERIFY outcome statuses — **opt-in** via `ENABLE_BROWSER_CONTROL=true` (default off).
- 🛡️ **The system, not the LLM, holds every boundary** — model-generated URLs/coordinates/commands are never auto-trusted: the driver interface has NO arbitrary JS, NO cookies/storage, NO tabs; page text is framed as UNTRUSTED PAGE CONTENT (hidden text is never extracted); submit/delete/password-worded actions escalate to SYSTEM risk dynamically and park in the SAME durable confirmation flow as before.
- 🧠 **Dynamic risk (args-dependent)** — `risk_for_args` + `registry.effective_risk_level`: a click on "Submit order" is NETWORK statically but SYSTEM for that call; escalation-only, fail-closed on any classifier error. No second permission system.
- 🛑 **Human-only emergency stop** — CLI `/stop` (or Ctrl+X mid-turn via stdlib `msvcrt`), API `POST /browser/emergency-stop`, dashboard button. The model has NO tool that can trigger or reset it. In-flight actions report `ACTION_INTERRUPTED`; `/resume` clears it.
- 🧾 **Honest verification** — every browser action returns a runtime-computed status (`ACTION_VERIFIED` / `ACTION_NOT_VERIFIED` / `ACTION_BLOCKED` / `ACTION_INTERRUPTED` / `ACTION_EXECUTED`): no executed-but-unverifiable action is ever reported as success, and duplicate side effects are suppressed exactly once per turn.
- 📥 **Bounded downloads** — captured to a per-session temp area (50 MB cap, executable suffixes never stored, random stored names, sha256 metadata, wiped on close) and reported as NEVER-executed artifacts.
- 🔒 **Credential redaction + session isolation** — page text is redacted (query secrets, bearer tokens, key-value secrets, high-entropy blobs near secret contexts) before the model ever sees it; the automation browser uses a fresh non-persistent profile (never the user's cookies); sessions are LRU-bounded with idle TTL.
- 🖥️ **Host boundary stays closed** — `jarvis/computer` keeps the restricted host abstraction with `DisabledHostBackend` as the ONLY backend; shell/registry/process/clipboard control remains structurally absent forever.
- 🧪 **124 new deterministic tests** (`tests/test_browser_control.py`, A–Z categories) — full suite **1032 passed**. Manual-only live eval (`live_browser_eval.py`) runs real Chromium against controlled LOCAL pages. Details: `docs/JARVIS_V028_SECURITY_MODEL.md`, final report in `docs/JARVIS_V028_FINAL_REPORT.md`.

## ✨ What's New in v0.27.0

- 🎙️ **Push-to-talk voice mode** (`--voice-ptt` / CLI option): explicit ENTER-per-turn capture — no wake word, no background listening, no persistent audio. Observable lifecycle (IDLE → LISTENING → TRANSCRIBING → THINKING → SPEAKING → ERROR); a failed turn never corrupts the session; speaks ONLY the final grounded response.
- 🗣️ **Honest provider boundaries** — STT is **LOCAL** (Whisper on-device; ffmpeg optional dependency); TTS is **NETWORK-BACKED** (Edge TTS endpoint) and clearly documented as such, with a disable switch (`TTS_ENABLED=false` / `--no-tts`), synthesis timeout, no silent retries, and a clean `stop()` cancellation boundary (half-duplex barge-in; no orphaned playback processes).
- 🖼️ **Image input as a first-class path** — attach an image to any request (API multipart, dashboard upload, CLI `--image`). Uploads are validated by CONTENT (magic-byte sniffing + pixel-bomb bounds; client filenames never trusted), stored under random names inside the sandbox, and cleaned after the turn. llava (verified: **no tools capability**) does visual interpretation only; tool reasoning stays with qwen2.5:7b.
- 👁️ **Visual observations are untrusted** — every vision result carries the VISUAL OBSERVATION contract (data, not instructions, not a trusted tool result); the v0.26 grounding guard excludes it from deterministic checking by construction.
- 🔌 **One multimodal API** — `POST /chat/multimodal` (multipart: text + optional image + optional audio) reuses the SAME auth, rate limiting, sessions, leases and grounding as `/chat`; `POST /chat` is untouched.
- 📊 **Multimodal telemetry** — voice_started/transcribed, tts_started/completed/cancelled/failed, vision_started/completed/failed, multimodal_request_started/completed events with bounded error categories; no raw audio/images ever logged.
- 🧪 **68 new deterministic tests** (validation, lifecycle, cancellation, API, injection, hygiene, leak checks) — full suite **936 passed**. Live validation: provider/model **5/5**, reasoning **2/2** (image reading, image→calculator tool use, session follow-up, grounded multimodal arithmetic). Report: `live_multimodal_report.json`.

## ✨ What's New in v0.26.0

- 🛡️ **Grounding Guard & Answer Integrity (system guarantee)** — every final answer is post-checked against the turn's trusted tool evidence by tool-aware deterministic policies (calculator numbers with locale-safe canonicalization, stated dates, exact file-listing names, labeled structured fields). High-confidence contradictions trigger exactly **one** bounded correction round; a still-contradicting answer is fail-closed into a truthful notice that preserves the authoritative value. Conservative by design: step narrations, years, timestamps/IDs, latencies, and percentages never trigger — a missed catch is acceptable, a false rejection is not.
- 🧮 **Locale-safe numeric canonicalization** — `41,971` ≡ `41 971` ≡ `41'971` ≡ `41971.0` ≡ `$41,971`; ambiguous forms (`12,34`, lone dot-groupings without locale siblings) are refused rather than guessed; `1.234.567` parses as 1,234,567.
- 🧾 **Trusted-evidence provenance** — evidence items carry live/cache provenance (detected from the v0.24 cache header); cached results are authoritative at the same level as live ones, a failed later attempt can never overwrite an earlier success, and multi-step turns accept any genuine evidence value (match-any, conservative).
- 🔁 **One bounded correction round** — the correction prompt restates the evidence and forbids recomputation; recursion is structurally impossible; a correction LLM failure degrades to the honest fallback, never an exception.
- 📊 **Grounding telemetry** — daily counters (checks / contradictions / corrections / fallbacks, per tool) in `grounding_metrics_daily`, exposed via `GET /ops/grounding/stats`, `maintenance cache stats`, and a dashboard **Answer grounding (last 14 days)** trend. Kill switch: `JARVIS_DISABLE_GROUNDING_GUARD=true`.
- 🧪 **17-case grounding benchmark (8 guard-layer + 9 orchestrator-layer, spec categories A–M): 17/17** — plus 97 new tests (numeric canonicalization, cue-gate false-positive rules, precedence, correction bound, fallback honesty, injection defenses). Live on qwen2.5:7b (2 reps): tool-choice 1.0, final-value 1.0, cache 2/2, refresh 2/2, conversational clean 2/2, zero fallbacks; guard overhead ~0.3–0.9 ms (<0.05% of synthesis). Report: `live_grounding_report_v026.json`.
- 🩹 **v0.25 gaps closed** — confirmation-resume turns now rebuild the evidence ledger (previously synthesized with none), and evidence attribution uses real dispatch-site tool names (the text heuristic is fallback only).

## ✨ What's New in v0.25.0

- 🧾 **Grounded synthesis with an evidence ledger** — the orchestrator now records every successful tool result into a bounded evidence ledger (clamped per item, at most 16 items, 1200 chars/item) and hands it to synthesis as **AUTHORITATIVE TOOL EVIDENCE**, framed as data ("transcribe tool-derived values exactly; never recompute them") — never as instructions. The live-verified detail: the block rides on the **final user message**, not a trailing system message (see the limitation note below for why that placement is load-bearing on qwen2.5:7b).
- 🔄 **Explicit client refresh** — every client surface can now force a fresh execution past the cache: the REST API (`ChatRequest.refresh`), the Python client (`client.chat(..., refresh=True)`), the CLI (`--refresh` flag or `/refresh` in-chat toggle), and a dashboard sidebar toggle. The dispatch layering is unchanged: refresh runs **after** PermissionGuard, confirmation parking, and the repeat ledger — it never bypasses permissions, schema validation, or confirmations, only the cache lookup (`bypass_reason="refresh_request"` in telemetry). Freshness-worded requests still bypass TTL-freshness tools automatically.
- 📊 **Daily cache metrics + history** — every dispatch records hit/miss/stale/bypass/store counters (per-tool deltas merged atomically) into a new `cache_metrics_daily` table in the same SQLite, pruned past `RESULT_CACHE_METRICS_RETENTION_DAYS` (default 30). Surface: `GET /ops/cache/stats/history?days=&limit=`, `maintenance cache stats` (now prints today's counters + per-tool deltas), and a dashboard **Daily cache activity (last 14 days)** trend table.
- 🧮 **Replan-diff telemetry** — a successful replan now logs/emits `replan_diff` (steps added, removed, retargeted — bounded, never arguments) next to `replan_triggered`/`replan_validated`, so "what changed between the plan that failed and the one that worked?" is answerable from logs or the SSE stream.
- 🛡️ **Deterministic evaluation, isolated** — all 9 evaluation scripts boot through `evaluation/_bootstrap.py::isolate()`, which redirects SQLite/Chroma to a private temp directory **before** any `jarvis` import (and fails loudly rather than ever touching the real `jarvis.db`); a pytest safety valve lets eval modules import under an already-redirected test DB. Regressions from live-eval fixes are pinned: synthesis receives its grounding block on the final user message; a replan may legitimately restate a failed step (`retry_descriptions`) without being skipped as redundant.
- 🧪 **Cache-policy consistency is test-pinned** — exactly **7 cached tools** (the 6 v0.24 retrieval tools plus `calculator`: global scope, `calculator_expression` value-based keys, 7-day TTL), with the expected-policy table and the never-cached set asserted in `tests/test_result_cache.py::TestCachePolicyConsistency`.
- 🤖 **Live verification (qwen2.5:7b, 10 cases × 2 reps): 18/20.** Cross-turn cache hits 6/6, freshness-word bypass 4/4, exactly-one-replan on scripted failure, grounding (tool value transcribed over the model's guess) proven; the 2 misses are small-model tool-choice variance, both mechanisms proven in the other rep. Synthesis-prompt overhead measured: +14.9% with one evidence item, +275.9% with the 16-item ledger (bounded by design).

---

## ✨ What's New in v0.24.0

- ♻️ **Cross-turn result cache** (`jarvis/core/result_cache.py`) — successful **read-only retrieval** is reused across turns: `web_search`, `wikipedia_summary`, `web_scrape` (global scope), `read_file`, `list_directory`, `search_knowledge` (session scope). Cacheability is an **explicit per-tool `CachePolicy`** (`jarvis/tools/base.py`) — tools without a policy are never cached (fail-safe), so side-effect tools (`write_file`), state-coupled tools (`get_current_datetime`, `recall_facts`, `remember_fact`), vision, code execution and computer control can never be replaced by a cached result.
- 🔑 **Safe normalized cache keys** — `sha1(tool + normalized_args)[:16]` with three per-policy normalizers: *generic* (v0.23 canonical JSON), *verbatim* (paths/URLs — key order only, never value rewriting), and *calculator_expression* (the calculator's existing AST parser proves equivalence: `2+2` ≡ `2 + 2` ≡ `(2+2)` ≡ `5-1` share one entry; unparseable input never collides).
- ⏱️ **Three freshness strategies, honestly separated** — *ttl* (web 300 s, Wikipedia 24 h, calculator 7 d — configurable via `RESULT_CACHE_*` settings), *source_stat* (file size+mtime re-checked at lookup; missing source ⇒ stale), and *knowledge_generation* (a tiny registry aggregate `docs:chunks:latest` — KB ingest invalidates retrieval without scanning Chroma). Source-modification invalidation and time expiration are distinct mechanisms by design.
- 🏷️ **Provenance, not pretense** — every cache hit is returned with a system-authored header: `[cached result: retrieved 5m ago via web_search; expires in 4m — not a live re-run; say 'latest' to force fresh retrieval]`. Requests containing freshness words (latest / today / current / breaking / …, `tool_policy.is_freshness_request`) bypass the cache for TTL-freshness tools — a small explicit policy, no NL classifier.
- 🔒 **Cache sits INSIDE the security boundary** — the lookup runs **after** PermissionGuard and confirmation parking and after the v0.23 duplicate ledger, and **before** the registry; arguments must pass the tool's own Pydantic schema or the lookup is a plain miss. Errors are never cached (retry-after-failure re-runs), `intentional_repeat` recovery rounds bypass it, session-scoped rows are served only to their session, and a hit is untrusted **evidence** framed as data — never an instruction, never an authorization. Kill switch: `JARVIS_DISABLE_RESULT_CACHE=true`.
- 🧭 **ONE bounded replan** — when structural execution evidence says a plan failed (a step's result is an ERROR, or a required tool's every attempt failed while the model papered over it — live evidence: confident wrong values), chat() runs exactly ONE validated replan within the **remaining** tool-round budget (never reset, never recursive — `_execute_plan_steps` cannot replan). The replan prompt is compact (original request, completed work as do-not-repeat, failed steps, remaining budget) and inherits completed work via shared state.
- 📣 **Truthful incomplete answers** — if even the replan fails, synthesis receives an `INCOMPLETENESS NOTICE` forcing it to name what remains unfinished — completion is never claimed because a replan *ended*. Honest telemetry: `plan_completed` logs planned/completed/failed/replans/complete; `plan_complete` and `replan` SSE events surface it live.
- 🧰 **Maintenance + observability** — `python -m jarvis.maintenance cache stats|inspect|cleanup` (bounded cleanup: expired rows first, oldest eviction only if still over the entry cap; `inspect` shows metadata only, never payloads), `GET /ops/cache/stats` (counts, no payloads), and a dashboard **Result cache** section (entries/hits/expired, per-tool counts, fingerprint-only listing).
- 🧪 **Evaluation** — `tests/test_result_cache.py` (42 tests: keys/normalization, TTL/stat/generation freshness, isolation, bypasses, never-cache, guard-before-cache, provenance, bounded maintenance), `tests/test_replan.py` (trigger-once, no-replan-on-success, recursion impossible, budget inheritance, compact replan context, truthful incompleteness), and `evaluation/cache_replan_benchmark.py` — **12/12 deterministic cases** (cache, normalization, replan) through the real orchestrator. **Live on qwen2.5:7b** (`live_cache_replan_eval.py`, 7 cases × 2 reps): cross-turn cache reuse **6/6** (wiki, web, calculator — zero re-retrieval on the repeat), freshness bypass **2/2**, exactly-one-replan-on-failure **2/2**, no-replan-on-success **2/2** (13/14 overall; the one miss was a model hallucinating a value before the fix landed mid-run — fixed and re-verified deterministically).

## ✨ What's New in v0.23.0

- 🔁 **Duplicate-dispatch ledger** (`jarvis/core/dispatch_guard.py`) — within one turn, a tool call whose (tool, canonical-arguments) fingerprint already **succeeded** is suppressed *before* dispatch (after PermissionGuard and confirmation parking) and the model is told to use the existing result. Legitimate repeats are preserved by construction: different arguments always run, failed calls are never recorded (retry works), genuinely state-coupled tools (`get_current_datetime`, `recall_facts`, `remember_fact`) are exempt, and the ledger is fresh every turn.
- 🧭 **Redundant plan-step skip** — the plan loop records completed step descriptions; an identical later step is logged (`redundant_plan_step`), surfaced as a `redundant_step` SSE event, and skipped instead of re-executed. Success/failure/completion of every plan is now logged (`plan_step_satisfied`, `plan_completed`).
- 🧮 **Plan-quality report** (`jarvis/core/plan_quality.py`) — every `plan_ready` log now carries a deterministic quality block (steps, toolless steps, unique/repeated tools, forward references, duplicate steps, estimated LLM calls, issues) so over-planning is visible without running the model.
- 🧪 **Semantic plan judge** (`evaluation/plan_judge.py`, evaluation-only) — asks the live model to score a plan (coverage / necessity / order / efficiency 1–5, unnecessary steps) with conservative error degradation. Not in the production loop; it never authorizes anything.
- 📊 **Dashboard plan status** — the Operations page now derives per-session plan status from the event timeline (tool executions/succeeded/failed, repeated-tool warning with the note that repeats may be legitimate or suppressed).
- 🧪 **Expanded evaluation** — `evaluation/multistep_benchmark.py` grows 7 → 11 cases with a `repeat_semantics` category (identical repeat suppressed, different-query repeat legit, calculator changed-numbers repeat, redundant duplicate plan step skipped) — **11/11 PASS**; `tests/test_dispatch_guard.py` adds 23 tests (fingerprints, ledger semantics, guard/confirmation interplay, fresh-ledger-per-turn). Live on qwen2.5:7b (11 cases × 2 reps): all-expected-tools **1.0**, argument-valid **1.0**, grounding **1.0**, fabrication **0.0**, completion **1.0** — and live suppression confirmed firing (identical `web_search` re-calls blocked pre-registry while both different-query searches dispatched).

## ✨ What's New in v0.22.0

- 🗺️ **Planner tool catalogue** — the planner now sees each tool's compact purpose (name: description), not bare names, so `required_tools` reflects what tools actually DO. Plus explicit planning rules: order steps so dependencies work, and name the matching tool for any capability needing current data.
- ✅ **Deterministic plan validation** (`jarvis/core/plan_validator.py`) — every generated plan is checked before execution: shape, ≤5 steps, registry truth (hallucinated tools removed, never executed), duplicate collapse, contiguous renumbering, and **forward-reference rejection** (a step depending on a later step's result is dropped unwinding-safely). A plan is a request, never authorization.
- 📎 **Exact-evidence flow between steps** — each step receives the *clamped raw tool results* of earlier steps ("Exact tool results from earlier steps"), so step 2 quotes the real calculator value instead of whatever step 1's prose happened to preserve. ContextManager remains the size boundary; resumed turns rebuild the ledger from persisted tool messages.
- 🔒 **Per-step tool enforcement** — a validated step that names required tools must attempt one tool round; the executor may not silently answer the step from memory when the capability exists.
- 🧭 **Routing fixes (Part L)** — "Calculate X and remember the result" and "search my documents and compare with the web" now go to the planner; a substring bug that sent ANY hyphenated prose ("plan-and-execute") to the fast path is fixed. Single-intent asks stay on the cheap path.
- 📊 **New multi-step evaluation** — `evaluation/multistep_benchmark.py` (7 deterministic cases: calc+memory, knowledge, web, comparison, scripted correction, no-tool control; metrics for plan structure, tool ORDER, argument validity, result propagation, grounding) plus 31 new tests in `tests/test_multistep.py`. Graders are formatting-normalized ("41,971" ≡ "41971") without weakening semantic expectations.
- 📡 **Plan lifecycle telemetry** — `plan_validated` (raw/steps/issues), `plan_step_failed`, `no_tool_direct_answer`, plus the existing `plan_ready`/`step_*` events answer "what did the agent plan, execute, and correct?" from logs.
- 🧪 **Live multi-step verification** (`live_multistep_eval.py`, manual): 9 cases × 2 reps on real qwen2.5:7b — plan creation 0.889, all-expected-tools **1.0**, exact tool order 0.556, argument-valid 0.944, grounding 0.944, fabrication **0.0**. The canonical calc+remember task: exact order + grounded in **4/4 reps**.

## ✨ What's New in v0.21.0

- 🎯 **Capability-aware tool policy** (`jarvis/core/tool_policy.py`) — a compact tool-selection contract (system message), a zero-latency capability classifier (no tool / specific capability / multi-step / knowledge / web / memory / file / vision / unavailable), and explicit *when-NOT-to-use* rules. The v0.16 absent-capability honesty rule is preserved and extended: requests for capabilities JARVIS lacks (code execution, computer control) are refused honestly, never pretended.
- 🛡️ **Fast-path safety net + deterministic calculator fallback** — single-intent arithmetic and knowledge requests force one tool round; if the model still answers arithmetic mentally (the v0.20 live-eval failure), JARVIS extracts the expression (symbol and word operators, date-literal safe), runs the REAL calculator through the normal PermissionGuard + schema-validation boundary, and forces one grounded phrasing round. Bounded: fires only when the model made zero tool attempts; knowledge stays model-driven.
- 🧭 **Routing disambiguation** — date literals (`2026-10-01`) route to datetime, never to the calculator (which would evaluate them as subtraction); unavailable-capability requests containing numbers (`run print(2+2)`) stay honest refusals, not calculator calls; planned steps now state that tool-backed capabilities require the tool.
- 📐 **All 12 tool descriptions rewritten capability-oriented** — PURPOSE / WHEN TO USE / WHEN NOT TO USE / inputs / output, tuned for a 7B model (the calculator description no longer discourages obvious arithmetic).
- 🧪 **+86 deterministic tests** — `tests/test_tool_policy.py` (policy, safety net, fallback, kill switch), `tests/test_tool_arguments.py` (schema-strictness regressions), `tests/test_tool_selection_benchmark.py`; plus `evaluation/tool_selection_benchmark.py`: a **27-case, 13-category deterministic benchmark** (tool dispatch, real Pydantic argument validation, hallucinated-tool blocking, grounding, no-tool false positives).
- 📊 **Live A/B evaluation** (`live_tool_eval.py`, NOT part of pytest) — faithful v0.20 prompt snapshot vs v0.21 on the real model: correct-tool rate **0.688 → 0.812**, tool-call rate **0.688 → 0.812**, argument-valid **1.0 → 1.0**, fabrication **0 → 0** on qwen2.5:7b. `llava:latest` was also evaluated and **cannot call tools at all** (Ollama rejects the tools parameter for it) — documented as a model-compatibility fact, not a system defect.
- 🔭 **New telemetry** — `tool_policy_applied`, `deterministic_tool_fallback`, `no_tool_direct_answer`, `forced_tool_round_unfulfilled` answer "why did/didn't JARVIS call X?" from structured logs without reading raw model output.
- Power users: `JARVIS_DISABLE_TOOL_POLICY=true` restores exact v0.20 behavior (covered by tests).

## ✨ What's New in v0.20.0

- 📚 **Personal knowledge base + RAG** — a clear architectural split: **personal memory** (facts about you, `remember_fact`/`recall_facts`, unchanged) vs **document knowledge** (files you explicitly ingest into a dedicated `knowledge_base` Chroma collection with its own SQLite registry).
- 🖹 **Safe ingestion** — explicit paths only (never crawling): TXT, Markdown, source code, JSON (stdlib) and **PDF with page-level metadata** (new `pypdf` dependency). Resolved paths must live inside `FILE_READER_ALLOWED_DIR`; credential-like files (`.env`, `*.pem/*.key`, secret-named text) are refused unconditionally; unsupported/oversized/empty files fail with clear reasons.
- 🧩 **Deterministic chunking** — paragraph-first packing (target 1200 chars, overlap 150), lossless hard-split for oversized paragraphs, stable chunk ids, and full trace-back metadata (document_id, source, filename, page, line range, chunk index, section, content hash).
- 🔁 **Incremental by content hash** — unchanged file → skipped (no re-embedding); identical bytes at a new path → one shared index entry; changed file → old chunks deleted and replaced in one pass (no stale chunks, no orphaned registry rows).
- 🎯 **`search_knowledge` tool** — bounded retrieval (top_k ≤ 20, evidence ≤ 6000 chars) with citation-ready results. Document evidence is returned inside explicit `DOCUMENT EVIDENCE START/END` framing marked as data, never instructions — prompt-injection text stays quoted, and the tool honestly reports `NO_RELEVANT_EVIDENCE` instead of inviting fabrication.
- 🧪 **32 deterministic RAG tests** (`tests/test_knowledge_rag.py`) — ingestion lifecycle incl. duplicate/reindex/delete, PDF page metadata, citation derivation from metadata only, adversarial-document framing, memory/knowledge separation, path-security refusals, API lifecycle over a real uvicorn server, and auth on all knowledge routes. Plus a test-isolation hardening: an autouse offline guard now fails fast (with a clear diagnosis) if any test tries to reach the real LLM.
- Full verification details: [`deploy/README.md`](deploy/README.md) and [`docs/architecture.md`](docs/architecture.md).

## ✨ What's New in v0.19.0

- 🖥️ **Operations dashboard** — the Streamlit UI gained an Operations view: recent actions (state/session filters), UNKNOWN actions with safe recovery guidance, session leases (redacted owners, staleness), and per-session causal timelines. Viewing never mutates; the ONLY mutation is reissue, behind an acknowledgement checkbox **plus** a final confirmation dialog. All API failures (unreachable, 401, 404, 409, 429) render as operator-level messages — the page never crashes.
- ♻️ **Full-context UNKNOWN recovery** — the resume context (original request, plan, completed steps, step number, remaining tool budget, mode) is now captured onto the ledger row at park time and copied onto a reissued action's confirmation (transitively across reissue chains). Approving a recovered action dispatches it exactly once, then **continues the original task** and synthesizes the final answer. Denial dispatches nothing and is recorded. Nested confirmations during recovery work and are themselves recoverable; a second UNKNOWN mid-recovery reissues safely.
- 📜 **Session timelines** — `GET /sessions/{id}/timeline`, `python -m jarvis.maintenance inspect --session <id> [--json]`, and a dashboard view merge messages, confirmation parks, action states, reissues, and leases into one bounded, chronological, safe-metadata-only stream (never tool args, result bodies, message content, or raw owner tokens).
- 🧹 **Retention + dry-run** — `cleanup --operational --dry-run` reports (without deleting) what would age out: terminal ledger rows past `--terminal-actions-days` (default 30), reissue audit rows whose BOTH linked actions are gone past `--reissues-days` (default 90), and expired/orphaned leases past `--leases-days` (default 30). PENDING/RUNNING/UNKNOWN actions, active confirmations, and any audit row still linking live actions are always protected.
- 🧪 **38 new tests** (`tests/test_full_recovery.py`, `tests/test_operator_experience.py`): the full recovery lifecycle against the real orchestrator (approve/deny/nested/second-UNKNOWN/budget/malformed context/missing session/duplicates/ceiling), timeline ordering/isolation/bounds/leak-proofing, retention protection rules incl. dry-run, the new endpoints served by a **real uvicorn server** consumed by the **real client**, and dashboard-backend parity. `reissue` now also accepts an optional `request_id` (server generates + echoes one otherwise).
- Full verification details: [`deploy/README.md`](deploy/README.md) and [`docs/architecture.md`](docs/architecture.md).

## ✨ What's New in v0.18.0

- 🔎 **Action ledger introspection** — the execution ledger is now queryable: `GET /actions` (filter by `state`/`session_id`, bounded `limit` ≤ 500) and `GET /actions/{action_id}` return safe metadata only (ids, tool name, risk level, state, timestamps, reissue depth) — never tool arguments or result bodies. Store-level: `list_action_executions`, `count_action_executions_by_state`, deterministic newest-first ordering.
- 📟 **Session-lease introspection** — `GET /sessions/leases` (and `python -m jarvis.maintenance sessions`) shows which sessions are leased, by a **redacted** owner (`component:rand8` — raw `host:pid:…` tokens never leave the store), until when, at which fencing token, and whether the lease is active or stale.
- ♻️ **Explicit UNKNOWN-action recovery (reissue)** — an UNKNOWN action can be deliberately re-issued via `POST /actions/{id}/reissue` or `python -m jarvis.maintenance reissue`. This is **not** a retry and never mutates the original: it mints a **new action id** + durable confirmation (so the new action passes the normal permission/confirmation flow), records an audit row (`action_reissues`), is **idempotent per request id** (same request twice → same new action), and is **bounded** (`MAX_REISSUES_PER_ACTION = 3` per original). Requires auth exactly like any mutating endpoint.
- 🩺 **Expanded `doctor` + maintenance CLI** — `python -m jarvis.maintenance` gained `actions`, `unknown-actions`, `sessions` (all read-only, `--json` available) and the `reissue` mutator; `doctor` now also checks the configured Ollama model's availability, stale leases, and UNKNOWN-action counts, with hints pointing at the matching inspection command. Inspection never mutates; only `reissue` does.
- 🧪 **73 new tests** (`tests/test_operator_introspection.py`): introspection filters/pagination/safe-metadata, lease staleness + redaction, the full reissue lifecycle (new identity, original stays UNKNOWN, approval dispatches exactly once, denial never executes, idempotent replay, ceiling, audit chain, unauthenticated reissue → 401, wrong state → 409), doctor scenarios, and CLI output/exit codes.
- Full verification details: [`deploy/README.md`](deploy/README.md) and [`docs/architecture.md`](docs/architecture.md).

## ✨ What's New in v0.17.0

- 🔁 **Durable action idempotency (execution ledger)** — every confirmed high-risk action now has a server-generated `action_id` and an explicit state machine in SQLite (`PENDING → RUNNING → SUCCEEDED/FAILED`, plus `UNKNOWN`). The execution claim is an atomic database conditional: duplicate approvals, retries, concurrent requests, and restarts can no longer double-dispatch a tool. A crash between dispatch and result recording surfaces as `UNKNOWN` — **never automatically re-executed** — with an explicit user-facing report. (At-most-once *automatic dispatch*; external side effects are honestly not claimed to be exactly-once.)
- 🗄️ **Multi-process session coordination without Redis** — per-session turns are now guarded by a two-layer scheme: the in-process mutex (fast) plus a database-backed **session lease** (TTL, owner-checked renew/release, fencing tokens, stale takeover after a crash). Two JARVIS processes sharing one SQLite file can no longer both process the same session; sessions recover automatically after a process dies.
- 🚦 **Database-backed rate limiting (opt-in)** — the sliding-window limiter gained an SQLite backend (`make_durable_limiter`): one `BEGIN IMMEDIATE` transaction per decision, so multiple processes on one database enforce a single limit per client. The in-memory default is unchanged.
- 🐳 **Per-run sandbox container identities** — every code execution runs in `jarvis-sbx-<random>` (previously one fixed name): concurrent executions no longer collide or serialize, timeout cleanup targets exactly the offending container, and `ExecutionResult.container_name` + structured logs make each run traceable. **Seccomp posture documented and verified**: containers run under Docker's builtin default seccomp profile (`name=seccomp,profile=builtin` — observed on the live engine), never weakened.
- 🧪 **Reliability regression suites** — 24 action-idempotency tests (including an 8-thread claim race and restart-between-park-and-approval), 10 session-concurrency tests (including true multi-process `ProcessPoolExecutor` lease tests), 13 rate-limiter tests (including cross-process limits), and new real-Docker concurrent-sandbox integration tests.
- Full verification details: [`deploy/README.md`](deploy/README.md) and [`docs/architecture.md`](docs/architecture.md).

## ✨ What's New in v0.16.0

- 🐳 **Sandbox base image digest genuinely pinned** — the `python:3.12-slim` digest in `deploy/Dockerfile.sandbox` was resolved by an actual `docker pull` against a Linux daemon and the production image built from it (previously a fail-closed placeholder). CI now really builds and smoke-runs the sandbox image.
- 🔬 **Real-Docker integration suite** — `tests/test_sandbox_integration.py` drives a real Linux engine and asserts *observed* behavior from inside containers: non-root uid, dropped capabilities, read-only rootfs, noexec tmpfs, no network, bounded PIDs, memory-cap kills, workload-killing timeouts (exit 124, no orphans), and the layer-C force-removal mechanism. Skips cleanly (with a reason) when Linux Docker is unavailable; a skip is never a pass.
- 🤖 **Live-model evaluation actually run** — the full 32-case suite executed against a real `qwen2.5:7b` via Ollama 0.34.1: **23/32 passed (71.9%), zero timeouts** on the final run (20/32 before the harness/grader fixes — the regression diff is exactly what `--compare` now reports). Failures were classified (agent defects vs grader gaps vs model variance); the real defects were fixed: an absent-tool fabrication hole in the system prompt (model silently computed `print(2+2)` instead of refusing) and an eval-harness mock that masked unregistered-tool calls.
- 📊 **Sharper evaluation signal** — machine-readable reports upgraded (schema v2: per-case content digest, expected-vs-actual tool detail, failure reasons/details, durations); `--compare PRIOR_JSON` regression diffing (newly failing / newly passing / unchanged, rename-surviving via content digests); the context-budget check now measures the prompt-side window it always claimed to check. Grader semantics audited; lexical-≠-semantic limitation documented.
- ⏱️ **Honest timeout semantics** — the per-case eval timeout is documented as a thread-join abandonment (the in-flight model call is NOT terminated); the code-execution sandbox remains the real workload terminator (exit 124, force-removal — both now observed against a live daemon).
- Full verification details: [`deploy/README.md`](deploy/README.md) and [`docs/architecture.md` § Verification status](docs/architecture.md).

## ✨ What's New in v0.15.0

- ⏱️ **Sandbox timeout enforced at the workload boundary** — the container entrypoint is `timeout <cap>s python3 …`, so a runaway snippet is killed *inside* the container (exit 124), not merely abandoned by the host. If the host-side wait times out first (hung CLI/daemon), the container is force-removed (`docker rm -f`). `ExecutionResult.timeout_layer` reports which layer fired: `container`, `host_kill`, or none. All isolation flags unchanged.
- 🔁 **Confirmation approval now resumes the original task** — approving a high-risk action executes it, persists the result, restores the paused plan from SQLite, executes the remaining steps, and synthesizes a final answer. Works after a process restart. Denials also continue the task (minus the denied action). Expired/legacy/corrupt pause states degrade safely.
- 🛡️ **CI trust boundary documented and enforced** — workflow `permissions: contents: read`, nightly eval job schedule-only with no persisted credentials, `CODEOWNERS` on workflows/deploy/container files; runner requirements (dedicated machine, unprivileged account, no docker.sock) written down in `deploy/README.md`.
- Full details of the resume architecture: [`docs/architecture.md`](docs/architecture.md).

## ✨ What's New in v0.14.0

- 🔬 **Live-model eval harness** — `evaluation/run_evals.py` is now practical: Ollama pre-flight, per-case timeouts (a runaway generation can't wedge the run), `--filter`/`--category` selection, category pass-rate breakdown, failure summaries, and `--json` machine-readable reports for trend tracking.
- 🤖 **Nightly CI evals** — scheduled job on self-hosted Ollama runners (non-blocking, report uploaded as an artifact).
- 🌊 **True SSE interleaving** — `/chat/stream` now runs the turn in a worker thread and streams lifecycle events *as they happen*, with 15-second keepalive comments so proxies don't close idle connections during long generations.
- 🧠 **Planner tool filtering** — hallucinated `required_tools` names are filtered against the real registry before execution (fully-hallucinated steps degrade to safe reasoning steps).
- 📏 **Strict-grounding prompt rules** — explicit anti-fabrication policy (say what's missing instead of guessing; never compute silently when a tool exists) and referent-echo rule for multi-turn answers.
- ⏱️ **Turn durations in logs** — `response_ready` now carries `duration_ms` for both the fast and complex paths; removed a duplicate schema-serialization call per complex request.
- 🧹 **Test dedup completed** — all fakes now come from `tests/fakes.py`.

- 🚦 **API rate limiting** — in-process sliding window (default 60 req/60s, `RATE_LIMIT_*` envs, `0` disables), keyed by API key or client IP; over-limit → `429` + `Retry-After`; `/health` exempt.
- 🔒 **Per-session serialization** — concurrent turns on one session now return `409 Conflict` instead of interleaving history or double-resolving confirmations; different sessions stay fully parallel.
- 🖥️ **API-first dashboard** — the Streamlit UI consumes the REST API via a new stdlib-only client (`jarvis/api/client.py`); with `JARVIS_API_URL` set it can run on a different machine than the agent runtime (legacy in-process mode still available).
- 📦 **Dedicated sandbox image** — `deploy/Dockerfile.sandbox`: minimal interpreter-only image with a dedicated unprivileged user; mutable tags (`latest`, bare names) now rejected by the sandbox validator — pin tags or digests. Full build/pin/pre-pull guide in `deploy/README.md`.
- 👁️ **Observability** — structured per-request logs (`api_request`), `tool_calls` lifecycle events in the SSE stream and `on_event` seam.
- 🧪 **Eval harness at 26 cases** — adds tool-failure honesty, three-turn context chains across topic detours, and comparative multi-source synthesis.

- 🐳 **Real Docker-isolated code execution** — one-shot containers with `--network none`, read-only rootfs, `--cap-drop ALL`, `no-new-privileges`, non-root user, memory/CPU/PID ceilings, and read-only code mounts. Off by default (`ENABLE_CODE_EXECUTION=false`); the tool joins the LLM's surface only when the config flag is set **and** Docker is verified usable. Fail-closed on every uncertain step.
- 🔑 **API authentication (opt-in)** — set `JARVIS_API_KEY` to require `Authorization: Bearer <key>` (or `X-API-Key`) on every endpoint except `/health`. Constant-time comparison.
- 🌊 **SSE streaming chat** — `POST /chat/stream` emits the agent's lifecycle (`intent` → `plan` → `step_start`/`step_done` → `synthesis` → `done`) as server-sent events, with in-band errors.
- 👁️ **Observability seam** — `Orchestrator.chat(..., on_event=...)` exposes the same lifecycle events the stream emits; observer failures can never break execution.
- 🧱 **Stability fixes** — context-window floor (a zero-length window can no longer return the entire history), session-history cache invalidated inside the store lock (no stale reads for concurrent API workers), plus regression tests for concurrency and edge cases.
- ⚠️ **Honest capability claims** — see [Current Limitations](#current-limitations).

---

## Features

- 💬 **Conversational AI** — full session memory backed by SQLite (thread-safe)
- 🛠️ **Tool calling** — Plan-and-Execute loop with a fast path for simple intents; concurrent dispatch within a tool round
- 🐍 **Code execution (opt-in)** — Docker-isolated one-shot containers; off unless explicitly enabled and verified (see [Safe Code Execution](#safe-code-execution-docker))
- 🔍 **Web search & Scraping** — DuckDuckGo and Playwright browser integration for **dynamic web browsing and extraction**
- 👁️ **Vision** — Image understanding via Llava
- 🕐 **Current time/date** — instant, no network
- 📄 **Read files** — sandboxed to your allowed directory (see [Current Limitations](#current-limitations) for writes)
- 🧠 **Long-term memory** — `remember_fact` / `recall_facts` backed by ChromaDB
- 🧩 **Context management** — windowing, task anchoring, rolling summary, tool-output clamping
- 🛡️ **Permission tiers & durable confirmations** — high-risk actions require explicit approval, persisted across restarts
- 🔒 **Fully local** — your data never leaves your machine, providing **local inference without remote API network latency**
- 🌐 **REST API** — FastAPI service layer with optional API-key auth and SSE streaming
- 🏗️ **Modular** — adding a new tool takes ~30 lines and one registration line

### 1. Python 3.11+

```bash
python --version   # must be 3.11 or higher
```

### 2. uv (fast Python package manager)

```bash
# Install uv
pip install uv
# or on macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
# or on Windows (PowerShell):
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. Ollama

Ollama runs LLMs locally on your machine.

**Install Ollama:**

| Platform | Command |
|----------|---------|
| macOS | `brew install ollama` or download from [ollama.com](https://ollama.com) |
| Linux | `curl -fsSL https://ollama.com/install.sh \| sh` |
| Windows | Download installer from [ollama.com/download](https://ollama.com/download) |

**Pull a model** (choose one):

```bash
# Recommended — capable, supports tool calling
ollama pull qwen2.5:7b

# Lighter alternative — fast, less capable (~2 GB)
ollama pull phi3:mini

# Verify it works
ollama run qwen2.5:7b "Hello, are you working?"
```

**Start the Ollama server** (if not already running):

```bash
ollama serve
```

> Ollama starts automatically on macOS and Windows after installation.
> On Linux you may need to run `ollama serve` manually in a terminal.

---

## Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd JARVIS

# 2. Create and activate a virtual environment with uv
uv venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows (PowerShell)

# 3. Install dependencies
uv pip install -e ".[dev]"

# 4. Copy the environment template
cp .env.example .env
```

The `.env` file comes pre-configured to use Ollama locally. No edits are needed
unless you want to change the model or sandbox directory.

---

## Running JARVIS

### Docker (recommended for a service)

```bash
cp .env.example .env          # set JARVIS_API_KEY at minimum
# external Ollama on the host:
OLLAMA_BASE_URL=http://host.docker.internal:11434 docker compose up -d api

# or fully self-contained (includes Ollama):
docker compose --profile local-llm up -d
docker compose exec ollama ollama pull qwen2.5:7b

curl http://localhost:8000/health
```

The `jarvis-data` volume persists `jarvis.db` and the vector store across restarts. Scheduled maintenance runs against the same volume:

```bash
docker compose run --rm api python -m jarvis.maintenance stats
docker compose run --rm api python -m jarvis.maintenance cleanup --days 30
```

See [`deploy/README.md`](deploy/README.md) for the sandbox image build and the full hardening checklist.

### Local CLI

Make sure Ollama is running in the background:

```bash
ollama serve
```

Then, you can start JARVIS in CLI mode:

```bash
jarvis
```

Or, if the script isn't on your PATH:

```bash
python -m jarvis.main
```

Pass `--refresh` (or toggle `/refresh` in-chat) to force fresh, non-cached tool runs — it skips only the result cache, never permissions or confirmations (v0.25).

### Voice mode

```bash
jarvis --voice
```

Or from the text REPL: type `/voice`.

### REST API server (v0.10)

Start the FastAPI service (same runtime as the CLI):

```bash
uv run uvicorn jarvis.api.app:app --port 8000
```

Quick tour:

```bash
# Health & introspection (version, model, tools, auth + sandbox posture)
curl http://localhost:8000/health

# Start a session
curl -X POST http://localhost:8000/sessions
# → {"session_id": "..."}

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<id>", "message": "What time is it?"}'

# Stream one turn as SSE events (intent → plan → steps → done)
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<id>", "message": "Research X and summarize it"}'

# Inspect history / resolve pending confirmations
curl http://localhost:8000/sessions/<id>/history
curl http://localhost:8000/sessions/<id>/confirm
curl -X POST http://localhost:8000/sessions/<id>/confirm \
  -H "Content-Type: application/json" -d '{"confirmed": true}'
```

Interactive OpenAPI docs are served at `http://localhost:8000/docs`.

### API authentication (opt-in)

By default the API trusts localhost. To require a key, set it in `.env` (or the environment):

```env
JARVIS_API_KEY=change-me-to-a-long-random-value
```

Every endpoint except `/health`, `/docs`, and `/openapi.json` then requires:

```bash
curl -H "Authorization: Bearer $JARVIS_API_KEY" ...      # preferred
curl -H "X-API-Key: $JARVIS_API_KEY" ...                 # alternative
```

Keys are compared in constant time; failed attempts log the client and return `401` with `WWW-Authenticate: Bearer`.

### Safe Code Execution (Docker)

Code execution ships **off**. To enable it:

```env
ENABLE_CODE_EXECUTION=true
SANDBOX_IMAGE=ubuntu:24.04          # tag-pinned; 'latest'/bare names are REJECTED
```

Production should build the dedicated minimal image instead (no shell tooling, no package managers, dedicated unprivileged user, **digest-pinned and pull-verified base**) — see [`deploy/README.md`](deploy/README.md):

```bash
docker build -t jarvis-sandbox:1.0.0 -f deploy/Dockerfile.sandbox deploy/
docker images --digests jarvis-sandbox   # then pin: SANDBOX_IMAGE=jarvis-sandbox:1.0.0@sha256:<digest>
```

The tool registers **only if** Docker verifies usable at startup (CLI + daemon + image present). Each execution runs in a one-shot container with: no network (`--network none`), a read-only root filesystem, all Linux capabilities dropped, `no-new-privileges`, a non-root user, hard memory/CPU/PID ceilings, and the code mounted read-only. Time limits are enforced **inside the container** (`timeout` wrapper, exit 124) with a host-side wait as backup and force-removal if the host wait fires first. Any failure mode is a *denial* — never host execution. Windows hosts are refused (use WSL2/Linux); the sandbox never pulls images at runtime.

> The code-execution tool is `SYSTEM` risk: even when enabled, a call parks a durable confirmation that must be approved (`/confirm` in the CLI, `POST /sessions/{id}/confirm` in the API) before the container runs.

## Security & Safety

JARVIS employs a strict permission model to ensure your machine stays secure.
- **Structurally absent tools:** `computer_control` is never registered — a pure placeholder with no OS automation code. `execute_python_code` registers only behind the double gate described above. The v0.28 **browser tools register only when `ENABLE_BROWSER_CONTROL=true`** (default off) and even then run through the deny-by-default URL policy, observation-freshness, pacing, dynamic-risk→confirmation, and deterministic verification gates — the model never receives raw coordinates, cookies, or arbitrary-JS capability, and the emergency stop is human-only (model has no tool for it). Details: [`docs/JARVIS_V028_SECURITY_MODEL.md`](docs/JARVIS_V028_SECURITY_MODEL.md).
- **Sandboxed File Reading:** `read_file` is strictly limited to paths relative to `FILE_READER_ALLOWED_DIR`.
- **Permission tiers:** `SAFE`/`NETWORK`/`FILE_READ` run automatically; `SYSTEM`/`DESTRUCTIVE` require explicit, durable user confirmation (`/confirm` in CLI, `POST /sessions/{id}/confirm` in the API, buttons in the dashboard); `FILE_WRITE` is currently blocked outright.
- **Hardened containers, not promises:** when code execution is enabled, every run is one-shot, network-less, capability-less, non-root, read-only-rootfs, and resource-capped. Failures deny — they never fall back to host execution.
- **No in-process `exec()`/`eval()`** on model-influenced strings — the v0.7-era in-process sandbox was removed. The calculator uses a restricted AST walker.
- **Bounded autonomy:** tool rounds are capped per request (5) and per step (2), with a self-correction sub-loop capped at 2 recovery attempts.
- **Optional API auth:** constant-time API-key enforcement whenever `JARVIS_API_KEY` is set.

Details: [`docs/architecture.md` § Safety Model](docs/architecture.md).

---

## Web UI Dashboard (v0.5+)

To launch the beautiful browser interface with thought process observability and image upload support, run:

```bash
streamlit run ui/dashboard.py
```

---

## Voice Mode (v0.4)

JARVIS can listen with **Whisper** (local STT) and speak with **Edge TTS** (free neural voices).

### Current Limitations

JARVIS advertises only what actually works today:

- **Code execution is opt-in and platform-gated** — disabled unless `ENABLE_CODE_EXECUTION=true` **and** Docker verifies usable; unavailable on Windows hosts (use WSL2/Linux), and the image must be pulled in advance. See [Safe Code Execution](#safe-code-execution-docker).
- **Computer control is disabled** — the tool is a pure placeholder with no OS automation code.
- **File writes are blocked** — `write_file` is registered but the permission guard blocks `FILE_WRITE` risk tier; only reads are permitted.
- **Voice TTS needs internet** — recognition is local (Whisper); speech output streams from Microsoft's free Edge TTS endpoint.
- **Auth is opt-in** — without `JARVIS_API_KEY` the API trusts localhost; set a key before exposing it beyond loopback.
- **Deterministic summaries** — context compaction is rule-based (no extra LLM call, zero cost, but less fluent than an LLM summary).
- **Lexical, not semantic, evaluation** — the 32 graders are deterministic string checks: great as a regression tripwire, but they do not prove semantic correctness (no LLM judge). See [`deploy/README.md` §7](deploy/README.md).

---

## Prerequisites

1. **ffmpeg** (required by Whisper)

| Platform | Install |
|----------|---------|
| Windows | `winget install FFmpeg` or download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to `PATH` |
| macOS | `brew install ffmpeg` |
| Linux | `sudo apt install ffmpeg` (Debian/Ubuntu) |

Verify:

```bash
ffmpeg -version
```

2. **Microphone** access allowed for your terminal / Python in OS privacy settings.

3. **Internet** for Edge TTS (speech synthesis is streamed from Microsoft Edge’s free TTS endpoint). Recognition itself is local via Whisper.

### Start voice mode

```bash
# Directly
jarvis --voice

# Or inside the text REPL
/voice
```

Say **“exit”** or **“quit”** (or press `Ctrl+C`) to leave voice mode.

### Configuration

```env
WHISPER_MODEL=base              # tiny | base | small | medium | large
TTS_VOICE=en-US-GuyNeural       # Edge neural voice
VOICE_RECORD_SECONDS=5          # Seconds of mic capture per turn
```

### Troubleshooting

| Symptom | Fix |
|---------|-----|
| Whisper fails to load / “ffmpeg not found” | Install ffmpeg and restart the terminal so `PATH` updates |
| No speech detected | Check mic permissions; speak during the “Listening…” window; raise `VOICE_RECORD_SECONDS` |
| TTS silent / network errors | Edge TTS needs internet; try another `TTS_VOICE` or check firewall |
| `playsound` errors on Windows | Install ffmpeg (`ffplay` is used as a fallback player) |

Text CLI mode remains fully available and is the default when you run `jarvis` without `--voice`.

---

## CLI Commands

| Command | Action |
|---------|--------|
| `/help` | Show all commands |
| `/tools` | List registered tools |
| `/history` | Print this session's messages |
| `/voice` | Enter voice mode |
| `/new` | Start a fresh session |
| `/quit` | Exit JARVIS |

---

## Configuration

Edit `.env` to customise behaviour:

```env
OLLAMA_MODEL=qwen2.5:7b        # Which model to use
OLLAMA_BASE_URL=http://localhost:11434
MAX_TOKENS=2048
FILE_READER_ALLOWED_DIR=.      # Sandbox for read_file tool
LOG_LEVEL=INFO                 # DEBUG | INFO | WARNING | ERROR

# Context management
MAX_CONTEXT_MESSAGES=24        # LLM-facing window size (older turns are summarized)
MAX_TOOL_OUTPUT_CHARS=6000     # Per-message cap for tool results (head + tail kept)

# Voice (v0.4)
WHISPER_MODEL=base
TTS_VOICE=en-US-GuyNeural
VOICE_RECORD_SECONDS=5
```

---

## Running Tests

```bash
pytest
```

Tests run fully offline — no Ollama or network required.

---

## Project Structure

```
JARVIS/
├── jarvis/
│   ├── runtime.py          ← Assembly seam: build_runtime() + JarvisRuntime
│   ├── main.py             ← CLI entry point (thin over the runtime)
│   ├── api/                ← FastAPI service layer (app.py, schemas.py)
│   ├── config.py           ← All config, loaded from .env
│   ├── core/
│   │   ├── orchestrator.py ← Route → Plan → Execute → Synthesize
│   │   ├── planner.py      ← JSON plan generation with fallback
│   │   ├── permissions.py  ← Risk tiers + confirmation policy
│   │   └── sandbox.py      ← CodeSandbox ABC + fail-closed implementations
│   ├── llm/
│   │   └── client.py       ← LiteLLM → Ollama wrapper
│   ├── memory/
│   │   ├── session_store.py   ← SQLite sessions/messages/confirmations (thread-safe)
│   │   ├── context_manager.py ← Windowing, anchoring, summarization, clamping
│   │   └── vector_store.py    ← ChromaDB long-term memory
│   ├── tools/
│   │   ├── base.py        ← Abstract tool interface + risk levels
│   │   ├── registry.py    ← Tool registry + dispatcher
│   │   ├── datetime_tool.py
│   │   ├── web_search.py
│   │   └── …
│   └── utils/
│       └── logging.py     ← structlog setup
├── evaluation/             ← Offline eval harnesses
├── tests/
├── docs/
│   ├── architecture.md     ← Layers, lifecycle, safety model
│   ├── JARVIS_USER_MANUAL.md     ← Owner's guide: capabilities, startup, troubleshooting
│   ├── JARVIS_DEVELOPER_MANUAL.md ← Architecture, internals, extension guides
│   └── PROJECT_OVERVIEW.md ← Handoff overview
├── ui/dashboard.py         ← Streamlit dashboard
├── .env.example
├── pyproject.toml
└── README.md
```

---

## Adding a New Tool

See [`docs/architecture.md` § Adding a New Tool](docs/architecture.md) for the full checklist.
In short: create a file in `jarvis/tools/`, subclass `BaseTool`, and add it to `_TOOL_FACTORIES` in `jarvis/runtime.py` — the single registration point that CLI, API, and dashboard all share.

---

## Roadmap & Versions

| Version | Highlights |
| :--- | :--- |
| **v0.27** | ✅ Multimodal experience: normalized text/voice/image requests over ONE runtime, push-to-talk voice with observable lifecycle, provider-hardened STT (local Whisper) + TTS (network Edge, cancellable), image input with content-sniffed validation + untrusted visual observations, `/chat/multimodal` API, multimodal CLI/dashboard, telemetry |
| **v0.26** | ✅ Grounding guard (deterministic post-synthesis answer integrity: trusted-evidence abstraction, tool-aware policies, ONE bounded correction round, fail-closed fallback), locale-safe numeric canonicalization, grounding telemetry + `/ops/grounding/stats` + dashboard trend, 17-case benchmark (17/17), resume-path evidence rebuild, dispatch-site tool attribution |
| **v0.25** | ✅ Grounded synthesis (bounded evidence ledger on the final user message), explicit client refresh (API/CLI/dashboard), daily cache metrics + history endpoint + trend, replan-diff telemetry, isolated deterministic evaluation, cache-policy consistency pinning |
| **v0.1** | ✅ Basic CLI, simple tools, SQLite memory |
| **v0.2** | ✅ Long-term memory (ChromaDB + embeddings) |
| **v0.3** | ✅ Plan-and-Execute multi-step agent loop |
| **v0.4** | ✅ Voice input (Whisper) + voice output (Edge TTS) |
| **v0.5** | ✅ Web UI (Streamlit) + write_file tool + synthesis polish |
| **v0.6** | ✅ Vision (image understanding) + Playwright web scrape |
| **v0.8** | ✅ Permission tiers, durable confirmations, caching, context management |
| **v0.9** | ✅ Service architecture: JarvisRuntime seam, FastAPI REST API, thread-safe store, sandbox contract |
| **v0.10** | ✅ Real Docker-isolated code execution (opt-in), API auth, SSE streaming, observability seam, stability fixes |
| **v0.11** | ✅ Deploy-ready service: rate limiting, per-session serialization, dedicated sandbox image, API-first dashboard |
| **v0.12** | ✅ Ship path: docker-compose, service image, CI, request IDs, maintenance CLI, dispatch dedup |
| **v0.13** | ✅ Operate: deep health (503 on broken persistence), error sanitization + correlation, WAL persistence, `doctor` command |
| **v0.14** | ✅ Live-model quality loop: practical eval harness, nightly CI evals, true SSE interleaving, planner grounding |
| **v0.15** | ✅ Reliability & security closure: workload-boundary sandbox timeout, resumable confirmations, CI trust hardening |

---

## Troubleshooting

- **LiteLLM / Ollama Error:** Ensure Ollama is running in the background (`ollama serve`). If you're missing a model, run `ollama pull <model-name>`.
- **Playwright errors:** Ensure the chromium binaries are installed via `playwright install chromium`.
- **Voice errors:** Ensure `ffmpeg` is on your system `PATH`.
- **API returns 503 on `/chat`:** Ollama is unreachable from the server process; check `ollama serve` and `OLLAMA_BASE_URL`.
- **API returns 401:** `JARVIS_API_KEY` is set and your request is missing/has a wrong key; send `Authorization: Bearer <key>`.
- **API returns 409 on `/chat`:** another turn is already running on that session; wait for it to finish (or use a different session).
- **Chat response is `ACTION_REQUIRES_CONFIRMATION…`:** the turn is *paused*, not finished — approve/deny via `/confirm`/`/deny` (CLI) or `POST /sessions/{id}/confirm` (API) and the original task continues automatically, including after a restart.
- **API returns 429:** you hit the rate limit (`RATE_LIMIT_REQUESTS` per `RATE_LIMIT_WINDOW_SECONDS`); honor the `Retry-After` header.
- **API returns 503 on `/health`:** the DB read/write probe failed — check volume mount and disk; `python -m jarvis.maintenance doctor` localizes it.
- **`/chat` returns a generic 503:** the real cause is in the server log under the response's `X-Request-ID` (also echoed in the JSON body as `request_id`); `python -m jarvis.maintenance doctor` checks Ollama reachability.
- **Code execution stays disabled:** with `ENABLE_CODE_EXECUTION=true` the tool still requires usable Docker (CLI, daemon, image pulled). Check the startup log for `code_execution_enabled_but_docker_unavailable`. Not supported on Windows hosts — use WSL2 or Linux. Also check `SANDBOX_IMAGE` is not `latest`/bare — mutable tags are rejected.


## License

MIT
