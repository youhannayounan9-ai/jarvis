"""
tests/test_tool_selection_benchmark.py
──────────────────────────────────────
v0.21: the deterministic tool-selection benchmark itself must stay green.

Runs the full mocked-LLM benchmark suite (offline; no Ollama, no real tool
executions — the dispatch probe validates arguments only) and asserts the
headline metrics.
"""

import pytest

from evaluation import tool_selection_benchmark as tsb


def test_benchmark_has_all_required_categories():
    categories = {c["category"] for c in tsb.CASES}
    required = {
        "calculator", "datetime", "web", "wikipedia", "memory", "knowledge",
        "files", "vision", "no_tool", "multi_tool", "ambiguity",
        "unavailable_capability", "adversarial",
    }
    assert required <= categories


@pytest.fixture(scope="module")
def benchmark_results():
    registry = tsb._registry()
    return [tsb.run_case(case, registry) for case in tsb.CASES]


class TestBenchmarkCases:
    def test_all_cases_pass(self, benchmark_results):
        failures = [
            r for r in benchmark_results if not r["passed"]
        ]
        assert failures == [], f"benchmark failures: {failures}"

    def test_correct_tool_rate_perfect(self, benchmark_results):
        assert all(r["correct_tool"] for r in benchmark_results)

    def test_argument_validity_perfect(self, benchmark_results):
        assert all(r["valid_args"] for r in benchmark_results)

    def test_no_unavailable_tool_ever_executes(self, benchmark_results):
        """Hallucinated/unavailable names must be refused, never dispatched."""
        for r in benchmark_results:
            if r["fabricated_tool"]:
                assert r["tools_called"] == [], r["name"]

    def test_no_tool_cases_dispatch_nothing(self, benchmark_results):
        for case, r in zip(tsb.CASES, benchmark_results):
            if case.get("expect_none"):
                assert r["tools_called"] == [], case["name"]

    def test_knowledge_canonical_case_uses_search_knowledge(
        self, benchmark_results
    ):
        case = next(c for c in tsb.CASES if "canonical" in c["name"])
        r = next(r for r in benchmark_results if r["name"] == case["name"])
        assert r["tools_called"] == ["search_knowledge"]

    def test_adversarial_evidence_stays_unexecuted_data(self, benchmark_results):
        """The injection stub is returned as DATA; no side-effect tool runs."""
        case = next(c for c in tsb.CASES if c["category"] == "adversarial")
        r = next(r for r in benchmark_results if r["name"] == case["name"])
        assert r["tools_called"] == ["search_knowledge"]
        assert r["passed"]
