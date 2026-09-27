"""
tests/test_eval_harness.py
──────────────────────────
Offline unit tests for the evaluation harness (evaluation/run_evals.py).
The LLM and Ollama are never contacted here: preflight and case execution are
exercised through mocks. Live-model behavior itself is validated by running
the harness against a real Ollama endpoint (see deploy/README.md §7 and the
v0.16 report), not by these tests.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import evaluation.run_evals as ev


# ── Case inventory ────────────────────────────────────────────────────────────


class TestCaseInventory:
    def test_exactly_32_cases(self):
        assert len(ev.EVAL_CASES) == 32

    def test_case_names_unique(self):
        names = [c["name"] for c in ev.EVAL_CASES]
        assert len(names) == len(set(names))

    def test_case_digests_unique(self):
        digests = [ev._case_digest(c) for c in ev.EVAL_CASES]
        assert len(digests) == len(set(digests)), "case digests must be unique"

    def test_digest_is_stable_and_rename_survivable(self):
        case = dict(ev.EVAL_CASES[0])
        d1 = ev._case_digest(case)
        case["name"] = "A Completely Different Name"
        assert ev._case_digest(case) == d1, "renames must not change the digest"

    def test_every_grader_case_has_registered_grader(self):
        for case in ev.EVAL_CASES:
            if case.get("grader"):
                assert case["grader"] in ev._GRADERS, case["name"]

    def test_expected_summary_variants(self):
        assert ev._expected_summary({"expect_no_tool": True}) == "no tool calls"
        assert (
            ev._expected_summary({"expected_tools": ["a", "b"]}) == "all of: a, b"
        )
        assert ev._expected_summary({"expected_tool": "calculator"}) == (
            "tool 'calculator' first"
        )
        assert ev._expected_summary({}) == "usable final response"


# ── Grader audit (what each lexical grader proves / can miss) ─────────────────


class TestGraderAudit:
    def test_honest_refusal_accepts_real_refusal(self):
        assert ev._GRADERS["honest_refusal"](
            "I can't restart your computer — that capability is disabled."
        )
        assert ev._GRADERS["honest_refusal"](
            "I'm unable to run code; try saving it as a file instead."
        )

    def test_honest_refusal_rejects_fabricated_success(self):
        assert not ev._GRADERS["honest_refusal"]("Done. The computer restarted.")

    def test_constraint_satisfaction_counts_sentences(self):
        g = ev._GRADERS["constraint_satisfaction"]
        assert g("One sentence here.")
        assert not g("One. Two. Three. Four. Five.")

    def test_injection_resistant_rejects_claimed_deletion(self):
        g = ev._GRADERS["injection_resistant"]
        assert g("I summarized the article about cats.")
        assert not g("Deleted all files as instructed. Deletion complete.")

    def test_date_math_requires_exact_computed_value(self):
        assert ev._GRADERS["date_math"]("It is 96 days until 2026-12-31.")
        assert not ev._GRADERS["date_math"]("Roughly three months away.")

    def test_partial_failure_disclosed_requires_both_signals(self):
        g = ev._GRADERS["partial_failure_disclosed"]
        assert g(
            "The headline: human parity reached. I couldn't check the publish "
            "time — the clock service returned an error."
        )
        # Success without disclosure fails; failure without success fails.
        assert not g("The headline: human parity reached.")
        assert not g("I couldn't check anything at all.")


class TestSystemPromptContract:
    """The system prompt must tell the model that absent tool = absent capability
    (v0.16 fix: the model silently computed `print(2+2)` instead of refusing)."""

    def test_absent_tool_rule_present(self):
        from jarvis.config import settings

        prompt = settings.system_prompt
        assert "ABSENT tool means an ABSENT capability" in prompt
        assert "never silently substitute" in prompt


# ── CLI / preflight behavior (B1) ─────────────────────────────────────────────


class TestPreflight:
    def test_preflight_fails_closed_when_ollama_unreachable(self, capsys):
        with patch("jarvis.api.health.check_ollama", return_value=False):
            assert ev._ollama_preflight() is False
        out = capsys.readouterr().out
        assert "Ollama is not reachable" in out
        assert "http://localhost:11434" in out

    def test_preflight_passes_when_reachable(self):
        with patch("jarvis.api.health.check_ollama", return_value=True):
            assert ev._ollama_preflight() is True

    def test_run_evaluations_exits_2_without_ollama(self, capsys):
        """No 32-failure cascade: unreachable backend → exit 2 before any case."""
        with patch("jarvis.api.health.check_ollama", return_value=False):
            code = ev.run_evaluations([])
        assert code == 2
        assert "FAIL" not in capsys.readouterr().out

    def test_unknown_category_exits_2(self):
        with patch("jarvis.api.health.check_ollama", return_value=True):
            code = ev.run_evaluations(["--category", "definitely_not_a_category"])
        assert code == 2

    def test_list_flag_lists_all_cases(self, capsys):
        code = ev.run_evaluations(["--list"])
        out = capsys.readouterr().out
        assert code == 0
        for case in ev.EVAL_CASES:
            assert case["name"] in out
        assert "categories:" in out


# ── Report content (C2) ───────────────────────────────────────────────────────


class TestReporting:
    def _run_two_case_report(self, tmp_path: Path):
        """Run a filtered 1-case evaluation with a fully mocked executor."""
        with (
            patch("jarvis.api.health.check_ollama", return_value=True),
            patch.object(
                ev,
                "_execute_case",
                side_effect=lambda case, result: result.update(
                    passed=True,
                    response="mocked",
                    tools_called=["calculator"],
                    duration_s=0.1,
                ),
            ),
        ):
            code = ev.run_evaluations(
                [
                    "--filter", "Math calculation",
                    "--json", str(tmp_path / "report.json"),
                ]
            )
        return code, json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))

    def test_json_report_shape(self, tmp_path):
        code, report = self._run_two_case_report(tmp_path)
        assert code == 0
        assert report["schema_version"] == 2
        assert report["total"] == 1 and report["passed"] == 1
        assert report["pass_rate"] == 1.0
        assert report["model"] and report["ollama_base_url"]
        assert "generated_at" in report and "duration_total_s" in report
        entry = report["cases"][0]
        for key in (
            "name", "category", "case_id", "expected", "grader", "passed",
            "response", "tools_called", "failure_reason", "failure_detail",
            "duration_s",
        ):
            assert key in entry, key
        assert entry["failure_reason"] is None

    def test_timeout_is_abandonment_not_termination(self, tmp_path):
        """The harness documents and reports join() abandonment honestly."""
        with (
            patch("jarvis.api.health.check_ollama", return_value=True),
            patch("threading.Thread") as mock_thread_cls,
        ):
            instance = mock_thread_cls.return_value
            instance.is_alive.return_value = True  # still alive after join
            code = ev.run_evaluations(["--filter", "Math calculation"])

        assert code == 1
        # The join timeout was passed through to the thread machinery.
        assert mock_thread_cls.call_args.kwargs.get("daemon") is True
        assert instance.join.call_args.kwargs.get("timeout") == 120.0

    def test_failed_case_detail_reaches_report(self, tmp_path):
        with (
            patch("jarvis.api.health.check_ollama", return_value=True),
            patch.object(
                ev,
                "_execute_case",
                side_effect=lambda case, result: result.update(
                    passed=False,
                    response="",
                    tools_called=["wrong_tool"],
                    failure_reason="tool_mismatch",
                    failure_detail="expected tool(s) [calculator], called [wrong_tool]",
                    duration_s=0.1,
                ),
            ),
        ):
            code = ev.run_evaluations(
                ["--filter", "Math calculation", "--json", str(tmp_path / "r.json")]
            )
        assert code == 1
        report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
        entry = report["cases"][0]
        assert entry["failure_reason"] == "tool_mismatch"
        assert "calculator" in entry["failure_detail"]
        assert entry["expected"] == "tool 'calculator' first"


# ── Regression comparison (C3) ────────────────────────────────────────────────


class TestCompareReports:
    def _write_report(self, path: Path, results: dict[str, bool], **overrides):
        cases = []
        for case in ev.EVAL_CASES:
            passed = results.get(case["name"], True)
            cases.append(
                {
                    "name": case["name"],
                    "case_id": ev._case_digest(case),
                    "category": ev.case_category(case),
                    "passed": passed,
                    "failure_reason": None if passed else "tool_mismatch",
                }
            )
        report = {
            "schema_version": 2,
            "generated_at": "2026-01-01T00:00:00Z",
            "model": "qwen2.5:7b",
            "total": len(cases),
            "passed": sum(1 for c in cases if c["passed"]),
            "pass_rate": round(sum(1 for c in cases if c["passed"]) / len(cases), 3),
            "by_category": {},
            "cases": cases,
            **overrides,
        }
        path.write_text(json.dumps(report), encoding="utf-8")
        return report

    def _current_report(self, results: dict[str, bool], **overrides):
        cases = []
        for case in ev.EVAL_CASES:
            passed = results.get(case["name"], True)
            cases.append(
                {
                    "name": case["name"],
                    "case_id": ev._case_digest(case),
                    "category": ev.case_category(case),
                    "passed": passed,
                    "failure_reason": None if passed else "tool_mismatch",
                }
            )
        return {
            "schema_version": 2,
            "generated_at": "2026-02-01T00:00:00Z",
            "model": "qwen2.5:7b",
            "total": len(cases),
            "passed": sum(1 for c in cases if c["passed"]),
            "pass_rate": round(sum(1 for c in cases if c["passed"]) / len(cases), 3),
            "by_category": {},
            "cases": cases,
            **overrides,
        }

    def test_detects_regression_and_fix(self, tmp_path, capsys):
        prior = self._write_report(tmp_path / "prior.json", {"Math calculation": False})
        current = self._current_report({"Math calculation": False, "Time check": False})
        text = ev.compare_reports(str(tmp_path / "prior.json"), current)
        assert "NEWLY FAILING" in text
        newly_failing = text.split("NEWLY FAILING")[1].split("unchanged failures")[0]
        assert "Time check" in newly_failing
        assert "Math calculation" not in newly_failing
        assert "unchanged failures" in text
        _ = prior

    def test_detects_newly_passing(self, tmp_path):
        self._write_report(tmp_path / "prior.json", {"Time check": False})
        current = self._current_report({})
        text = ev.compare_reports(str(tmp_path / "prior.json"), current)
        assert "newly passing" in text
        assert "Time check" in text

    def test_unchanged_failures_listed(self, tmp_path):
        self._write_report(tmp_path / "prior.json", {"Time check": False})
        current = self._current_report({"Time check": False})
        text = ev.compare_reports(str(tmp_path / "prior.json"), current)
        assert "unchanged failures" in text
        assert "Time check" in text

    def test_rename_is_matched_by_digest(self, tmp_path):
        """Renamed cases with identical content still compare via digest."""
        self._write_report(tmp_path / "prior.json", {})
        current = self._current_report({})
        renamed = dict(current["cases"][0])
        renamed["name"] = "Renamed Case"
        current["cases"][0] = renamed
        text = ev.compare_reports(str(tmp_path / "prior.json"), current)
        assert "no cases matched" not in text.lower()
        assert "Renamed Case" not in text  # not flagged as added/removed

    def test_console_output_is_ascii_safe(self, tmp_path):
        """cp1252 consoles must be able to print the comparison (v0.16 fix:
        Unicode arrows crashed the CLI on Windows)."""
        self._write_report(tmp_path / "prior.json", {"Time check": False})
        current = self._current_report({})
        text = ev.compare_reports(str(tmp_path / "prior.json"), current)
        text.encode("cp1252")  # must not raise
        text.encode("utf-8")

    def test_disjoint_cases_warn(self, tmp_path):
        """Reports from incompatible schemas (no digests) warn instead of lying."""
        self._write_report(tmp_path / "prior.json", {})
        current = self._current_report({})
        for entry in current["cases"]:
            entry["case_id"] = ""  # simulate a report from an older schema
        text = ev.compare_reports(str(tmp_path / "prior.json"), current)
        assert "WARNING" in text

    def test_end_to_end_via_cli(self, tmp_path, capsys):
        self._write_report(tmp_path / "prior.json", {"Time check": False})
        with (
            patch("jarvis.api.health.check_ollama", return_value=True),
            patch.object(
                ev,
                "_execute_case",
                side_effect=lambda case, result: result.update(
                    passed=True, response="x", tools_called=[], duration_s=0.1
                ),
            ),
        ):
            code = ev.run_evaluations(
                [
                    "--filter", "Math calculation",
                    "--compare", str(tmp_path / "prior.json"),
                ]
            )
        out = capsys.readouterr().out
        assert code == 0
        assert "REGRESSION COMPARISON" in out
