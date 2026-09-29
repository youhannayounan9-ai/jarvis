"""
jarvis/core/tool_policy.py
──────────────────────────
v0.21 Agent quality: capability-aware tool selection.

Deterministic, zero-LLM-cost routing helpers that sit between the intent
router and the ReAct loop. Three responsibilities:

  1. ``build_tool_policy_block()`` — a compact usage-policy system message
     (a structured, general contract — NOT per-phrase special cases) that
     teaches the model WHEN each capability applies and when NOT to call it.
  2. ``classify_intent()`` — a heuristic capability classifier that detects
     a single-obligation request (one clear tool whose absence would make
     the answer wrong, and no second task hiding in the text). The
     orchestrator uses this ONLY as a safety net on the fast path, to force
     one tool round when the router would otherwise have skipped tools.
  3. ``narrow_schemas_for_react()`` — capability-filtered tool schemas for
     the ReAct rounds of a PLANNED step, so the executing model sees a
     short, relevant surface instead of every tool at once.

Design rules (v0.21 contract):
  - Every mechanism here is advisory to the MODEL (policy text) or a
    SAFETY-NET for routing (fast-path override). It never authorizes a
    tool call: PermissionGuard remains the only authorization boundary,
    and dispatch-time Pydantic validation remains mandatory.
  - Hallucinated tool names stay impossible to execute: the registry
    rejects unknown tools, and planned required_tools are filtered by the
    planner. Nothing here widens that.
  - Unmet-capability honesty: when the user asks for something no
    registered tool can do, ``detect_unmet_capability()`` returns a short
    directive that is appended to the step brief / fast-path messages so
    the model refuses explicitly instead of pretending. This preserves and
    reinforces the v0.16 absent-tool rule.
  - Kill switch: ``JARVIS_DISABLE_TOOL_POLICY=true`` restores the exact
    v0.20 prompt/behavior (used by the A/B live benchmark).
"""

from __future__ import annotations

import re
from typing import Any

from jarvis.tools.registry import ToolRegistry
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


# ── Capability families (the vocabulary of the contract) ─────────────────────
# Capability → tool names. A tool may appear in one family only; the FIRST
# match wins for narrowing (order below = precedence for mixed asks).
CAPABILITY_TOOLS: dict[str, tuple[str, ...]] = {
    "knowledge": ("search_knowledge",),
    "memory": ("recall_facts", "remember_fact"),
    "files": ("read_file", "write_file", "list_directory"),
    "vision": ("vision_analyze",),
    "web": ("web_search", "web_scrape", "wikipedia_summary"),
    "compute": ("calculator", "get_current_datetime"),
}

# Tools that are obligation-triggers: if the request clearly needs this
# capability, calling the tool is REQUIRED, not optional. These are exactly
# the capabilities where silent mental substitution breaks grounding.
_OBLIGATION_TRIGGERS = ("calculator", "search_knowledge")

# Capability → heuristic keyword list. Deliberately SHORT and general
# (capability-oriented, not phrase-farms); unknown phrasings fall through to
# the normal full-schema ReAct path, which remains correct by construction.
_CAPABILITY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "knowledge": (
        "my documents", "my notes", "my roadmap", "my knowledge",
        "knowledge base", "documents i", "i ingested", "i uploaded",
        "my files say", "my pdf", "my markdown", "according to my",
    ),
    "memory": (
        "my favorite", "my name", "remember", "what do you remember",
        "my preferred", "i told you", "my project", "about me",
    ),
    "files": ("read the file", "open the file", "read ", "write a file", "list the files", "list files"),
    "vision": ("this image", "the screenshot", "analyze the image", "describe the image", "photo"),
    "web": ("search the web", "search for", "latest", "current", "news", "price of", "wikipedia", "who is", "what is the weather"),
    "compute": ("calculate", "compute", "how many", "sum of", "what day", "what date", "what time", "days until", "days between", "days from"),
}

# Word-boundary regexes for NARROWING planned-step descriptions. These are
# deliberately more general than the routing keywords above: a planner step
# like 'Search the knowledge base for X' never says 'my', so narrowing uses
# capability nouns instead. Narrowing is conservative (fallback = all tools).
_NARROW_PATTERNS: dict[str, re.Pattern[str]] = {
    "knowledge": re.compile(r"\b(knowledge base|documents?|ingested|roadmap|notes?)\b", re.IGNORECASE),
    "memory": re.compile(r"\b(memory|recall|remember|personal facts?)\b", re.IGNORECASE),
    "files": re.compile(r"\b(files?|directories|directories|folders?|directory)\b", re.IGNORECASE),
    "vision": re.compile(r"\b(images?|screenshots?|photos?|pictures?)\b", re.IGNORECASE),
    "web": re.compile(r"\b(web|search|wikipedia|urls?|news|latest|current)\b", re.IGNORECASE),
    "compute": re.compile(r"\b(calculate|calculation|compute|arithmetic|math|date|time|days?|weeks?|months?)\b", re.IGNORECASE),
}

# Explicit arithmetic always implies the compute capability, even when no
# keyword matches ('What is 893 * 47?' has no 'calculate').
_ARITHMETIC_PATTERN = re.compile(r"\d\s*[-+*/^]\s*\d")

# ISO date literals ('2026-10-01'). The calculator would evaluate one as
# numeric subtraction (2026-10-01 → 2015), so date literals route to the
# datetime anchor unless explicit calculation wording is present.
_DATE_LITERAL_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_CALCULATOR_WORDING = re.compile(
    r"\b(calculate|compute|sum of|multiply|divided by|product of|"
    r"how many days|days until|days between|days from)\b",
    re.IGNORECASE,
)

# Capabilities the runtime does not currently expose (e.g. code execution
# disabled → no execute_python_code tool). Matching user phrasings map to an
# honest-refusal directive instead of a pretended attempt.
_UNAVAILABLE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\b(run|execute)\b.*\b(python|code|script|snippet)\b", re.IGNORECASE),
        "code execution",
    ),
    (
        re.compile(r"\b(restart|shut ?down|lock)\b.*\b(computer|pc|machine|system)\b", re.IGNORECASE),
        "computer control",
    ),
)
# Word-boundary-safe regex cache (built once from _CAPABILITY_KEYWORDS).
_CAPABILITY_PATTERNS: dict[str, re.Pattern[str]] = {
    cap: re.compile(
        "|".join(re.escape(k) for k in keys),
        re.IGNORECASE,
    )
    for cap, keys in _CAPABILITY_KEYWORDS.items()
}

# Structured patterns the literal keyword list cannot express (interleaved
# words): 'my AI roadmap', 'my meeting notes', 'the PDF I ingested'.
_CAPABILITY_EXTRA_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "knowledge": (
        re.compile(
            r"\bmy\s+(?:\w+\s+){0,3}(roadmaps?|notes|documents|files|pdfs?|knowledge)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:the\s+)?(?:pdf|document|file)\s+i\s+(?:ingested|uploaded|gave|shared)\b",
            re.IGNORECASE,
        ),
    ),
}


def _capability_matches(cap: str, text: str) -> bool:
    if _CAPABILITY_PATTERNS[cap].search(text):
        return True
    return any(p.search(text) for p in _CAPABILITY_EXTRA_PATTERNS.get(cap, ()))


# ── 1. The tool-usage policy block ───────────────────────────────────────────

_TOOL_POLICY_BLOCK = (
    "## Tool Selection Contract\n"
    "Decide in this order:\n"
    "1. NO TOOL if the answer is already in the conversation or is general "
    "reasoning/knowledge you can explain (concepts, opinions, definitions, "
    "how-things-work). Do not call tools to decorate an answer you already have.\n"
    "2. A KNOWN CAPABILITY otherwise. Match the need to the capability, not "
    "to a memorized phrase:\n"
    "- EXACT arithmetic on numbers in the request (add, subtract, multiply, "
    "divide, powers, percentages, unit/day counts) → calculator. NEVER do "
    "this arithmetic in your head; an uncalculated number is a wrong answer. "
    "Conceptual math explanations need no tool.\n"
    "- The current date/time, or resolving 'today/tomorrow/next week' → "
    "get_current_datetime (then calculator for spans).\n"
    "- The user's OWN documents/notes ('my roadmap', 'my notes', 'my PDF', "
    "'my knowledge base') → search_knowledge. If its evidence does not "
    "cover the question, say so plainly — never invent a citation.\n"
    "- Facts about the USER (preferences, name, past statements) → "
    "recall_facts before answering; saving a user-stated fact → "
    "remember_fact. Never guess personal facts.\n"
    "- Current/external world info → web_search (recency) or "
    "wikipedia_summary (stable background); a specific URL → web_scrape.\n"
    "- A file the user named → read_file; creating/appending a file → "
    "write_file; directory contents → list_directory.\n"
    "- An image/screenshot the user provided → vision_analyze.\n"
    "3. If several independent needs exist, satisfy ALL of them (multiple "
    "tool calls or plan steps) — never drop one silently.\n"
    "4. A tool ABSENT from your list means the capability is absent: refuse "
    "honestly, name what is missing, offer the nearest safe alternative. "
    "Never simulate a result for a capability you do not have, and never "
    "claim you searched/read/calculated/remembered unless the tool result "
    "for it is in this conversation."
)


def build_tool_policy_block() -> str:
    """Return the tool-selection contract system block (stable text)."""
    return _TOOL_POLICY_BLOCK


# ── 2. Heuristic capability classifier ───────────────────────────────────────

def _count_substantive_clauses(text: str) -> int:
    """Rough count of separate asks: sentence-ish splits + coordination."""
    sentences = [s for s in re.split(r"[.!?\n]+", text) if s.strip()]
    clauses = 1 + sum(
        1
        for s in sentences
        for conn in (r"\band then\b", r"\bafter that\b", r"\balso\b", r"\bthen\b", r"\band\b")
        if re.search(conn, s, re.IGNORECASE)
    )
    return max(1, len(sentences)) + (clauses - max(1, len(sentences)))


def classify_intent(user_input: str, registry: ToolRegistry) -> dict[str, Any]:
    """
    Heuristic single-pass capability classifier (zero LLM cost).

    Returns:
        {
          "capability": str | None,   # detected capability family, if any
          "tools": list[str],         # candidate tools of that family that exist
          "tool": str | None,         # the single obligation tool, if unambiguous
          "single_intent": bool,      # one clear ask, no second task hiding
        }
    """
    text = (user_input or "").strip()
    registered = set(registry.list_tools())

    capability: str | None = None
    for cap in CAPABILITY_TOOLS:
        if _capability_matches(cap, text):
            capability = cap
            break
    if capability is None and _ARITHMETIC_PATTERN.search(text):
        capability = "compute"

    tools = [t for t in CAPABILITY_TOOLS.get(capability, ()) if t in registered]
    tool: str | None = None
    if len(tools) == 1:
        tool = tools[0]
    elif capability == "compute":
        # Distinguish exact-arithmetic obligations from datetime lookups.
        # Date literals must NOT route to the calculator (it would evaluate
        # '2026-10-01' as subtraction); they anchor via datetime instead.
        if _CALCULATOR_WORDING.search(text):
            tool = "calculator" if "calculator" in registered else None
        elif _DATE_LITERAL_PATTERN.search(text):
            tool = "get_current_datetime" if "get_current_datetime" in registered else None
        elif _ARITHMETIC_PATTERN.search(text):
            tool = "calculator" if "calculator" in registered else None
        else:
            tool = "get_current_datetime" if "get_current_datetime" in registered else None
    elif capability == "memory":
        if re.search(r"\bremember\b|\bsave\b|\bnote that\b", text, re.IGNORECASE):
            tool = "remember_fact" if "remember_fact" in registered else None
        else:
            tool = "recall_facts" if "recall_facts" in registered else None

    # Single intent = one detected capability AND no multi-task markers
    # (chained asks, coordination of different verbs, long compound text).
    multi_markers = re.search(
        r"\band then\b|\bafter that\b|\bthen\b|\balso\b|;|\band (write|save|search|remember|calculate|create)",
        text,
        re.IGNORECASE,
    )
    single_intent = capability is not None and multi_markers is None and len(text.split()) <= 40

    return {
        "capability": capability,
        "tools": tools,
        "tool": tool if tool in _OBLIGATION_TRIGGERS else tool,
        "single_intent": bool(single_intent),
    }


def extract_arithmetic(user_input: str) -> str | None:
    """
    v0.21 deterministic fallback (Part G): extract the arithmetic expression
    from a single-intent calculator obligation that the model refused to
    execute via the calculator tool.

    Handles both explicit operators ('893 * 47') and word operators
    ('144 divided by 12', '6 plus 7'). Returns None for anything else — the
    caller then keeps the model's own answer (policy degrades gracefully).
    """
    text = user_input.strip().rstrip("?").strip()
    # ISO date literals are dates, not subtraction: remove them BEFORE any
    # arithmetic matching so 'What day is 2026-10-01?' never yields '2026-10-01'.
    text = _DATE_LITERAL_PATTERN.sub(" ", text)

    # 1. Symbol operators: prefer the LONGEST match so '4 * (2 + 3)' wins
    #    over '4 * 2'. Numbers, operators, parentheses, decimals, commas.
    matches = re.findall(r"[0-9][0-9,]*(?:\.[0-9]+)?(?:\s*[-+*/^]\s*\(?\s*[0-9][0-9,]*(?:\.[0-9]+)?\s*\)?)+", text)
    if matches:
        expr = max(matches, key=len).replace(",", "").replace("^", "**")
        return expr

    # 2. Word operators: '<n> divided by <m>', '<n> plus/minus/times <m>'.
    word_match = re.search(
        r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*"
        r"(divided by|multiplied by|times|plus|minus)\s*"
        r"([0-9][0-9,]*(?:\.[0-9]+)?)",
        text,
        re.IGNORECASE,
    )
    if word_match:
        a, op, b = word_match.groups()
        sym = {
            "divided by": "/",
            "multiplied by": "*",
            "times": "*",
            "plus": "+",
            "minus": "-",
        }[op.lower()]
        return f"{a.replace(',', '')}{sym}{b.replace(',', '')}"

    return None


def is_single_intent_obligation(user_input: str, registry: ToolRegistry) -> str | None:
    """
    Fast-path safety net: return the ONE tool whose absence would make the
    answer wrong (calculator, search_knowledge) when the request is a clean
    single-intent ask for exactly that capability. Otherwise None.

    Used ONLY to force one tool round on the 'simple' path — never to skip
    tools, never to authorize anything.
    """
    result = classify_intent(user_input, registry)
    if result["single_intent"] and result["tool"] in _OBLIGATION_TRIGGERS:
        # A request for an UNAVAILABLE capability (e.g. 'run this python
        # snippet: print(2+2)') may contain arithmetic, but it is NOT a
        # calculator obligation — the honest-refusal note governs instead.
        # Also prevents the deterministic fallback from 'executing' the
        # snippet's constants as arithmetic.
        if detect_unmet_capability(user_input, registry):
            return None
        return str(result["tool"])
    return None


# ── 3. ReAct schema narrowing for planned steps ──────────────────────────────

def narrow_schemas_for_react(
    registry: ToolRegistry,
    step_description: str,
    planned_tools: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Capability-filtered tool schemas for a planned step's ReAct rounds.

    Order of preference:
      1. Planned tools that exist (planner intent, already hallucination-
         filtered by the planner) — unioned with (2).
      2. Capability families whose keywords appear in the step description.
      3. FALLBACK: if nothing matched (the common case — descriptions are
         prose), return ALL schemas. Narrowing must never hide a tool the
         step actually needs; a wrong guess is worse than no narrowing.
    """
    all_schemas = registry.get_schemas()
    wanted: set[str] = {t for t in (planned_tools or []) if t}

    matched_family = False
    for cap, pattern in _NARROW_PATTERNS.items():
        if pattern.search(step_description):
            matched_family = True
            wanted.update(t for t in CAPABILITY_TOOLS[cap] if registry.get(t))

    # A step that mentions files by extension/name almost always wants the
    # file family even if the description says only 'summarize the file'.
    if not matched_family and re.search(r"\b(file|\.md|\.txt|\.pdf|\.py)\b", step_description, re.IGNORECASE):
        wanted.update(t for t in CAPABILITY_TOOLS["files"] if registry.get(t))
        matched_family = True

    if not wanted or not matched_family:
        return all_schemas

    narrowed = [s for s in all_schemas if s["function"]["name"] in wanted]
    return narrowed if narrowed else all_schemas


# ── 4. Unmet-capability detection (honest-refusal reinforcement) ─────────────

# ── 5. Freshness requests (v0.24, Part L) ─────────────────────────

# Explicit freshness vocabulary. When a request contains one of these, the
# v0.24 cross-turn result cache is BYPASSED for freshness-sensitive tools —
# a small, auditable word-boundary policy, not a natural-language classifier
# (Part L: "use a small explicit policy where safe"). Applied only to tools
# whose policy declares time-based freshness; deterministic tools (calculator)
# and source-state tools (read_file) don't need it.
_FRESHNESS_PATTERN = re.compile(
    r"\b(latest|newest|current|currently|today|tonight|now|right now|real-?time|"
    r"up[- ]to[- ]date|fresh|breaking|this (?:week|month|year|morning|afternoon|evening)|"
    r"yesterday|live|so far)\b",
    re.IGNORECASE,
)


def is_freshness_request(user_input: str) -> bool:
    """
    True when the user's request explicitly asks for fresh/current
    information — such requests bypass cached retrieval (Part D: the user
    must be able to force fresh retrieval). Conservative by design: a miss
    just means the normal TTL/freshness machinery decides, never that stale
    data is served as current.
    """
    return bool(_FRESHNESS_PATTERN.search(user_input or ""))


def detect_unmet_capability(user_input: str, registry: ToolRegistry) -> str | None:
    """
    When the request names a capability NO registered tool provides, return
    a short directive to append to the model's instructions so it refuses
    explicitly instead of pretending. Otherwise None.
    """
    registered = set(registry.list_tools())
    for pattern, capability in _UNAVAILABLE_PATTERNS:
        if pattern.search(user_input) and not any(
            "code" in t or "computer" in t for t in registered
        ):
            return (
                f"CAPABILITY NOTE: the user's request needs {capability}, "
                "which is NOT in the tool list. Refuse that part honestly, "
                "name the missing capability, and offer the nearest safe "
                "alternative. Never simulate or claim its result."
            )
    return None
