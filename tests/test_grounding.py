"""
tests/test_grounding.py
───────────────────────
v0.26 Grounding Guard & Answer Integrity — unit tests for the deterministic
layer (jarvis/core/grounding.py).

Covers: numeric canonicalization (Part 4), tool-aware policies (Part 9),
the deterministic guard (Part 3), trusted-evidence construction and
provenance (Part 2), precedence guarantees (Part 7), and the correction/
fallback builders (Parts 5/6).
"""

import re

import pytest

from jarvis.core.grounding import (
    CORRECTION_INSTRUCTION,
    POLICIES,
    PRECEDENCE,
    CalculatorGroundingPolicy,
    CanonicalNumber,
    DatetimeGroundingPolicy,
    FileResultGroundingPolicy,
    GroundingPolicy,
    StructuredFieldGroundingPolicy,
    TrustedEvidence,
    build_correction_prompt,
    build_fallback_answer,
    build_trusted_evidence,
    canonical_number,
    canonical_numbers,
    check_grounding,
    contradictory_numbers,
    numeric_appearances,
)

CALC_EVID = [{"step_number": 1, "tool": "calculator", "status": "ok", "result": "Result: 41971"}]


# ── Part 4: numeric canonicalization ──────────────────────────────────────────

class TestCanonicalNumber:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("41971", 41971),
            ("41,971", 41971),
            ("41 971", 41971),
            ("41'971", 41971),
            ("41,971.0", 41971),
            ("41,971.25", 41971.25),
            ("1,234,567", 1234567),
            ("41971.0", 41971),
            ("3.14", 3.14),
            ("4,5", 4.5),               # single-digit comma pair: unambiguous decimal
            ("41971 USD", 41971),
            ("25%", 25),
            ("$1,234", 1234),
            ("-273.15", -273.15),
            ("893", 893),
            ("1.234.567", 1234567),     # ≥2 dot groups: unambiguous grouping
        ],
    )
    def test_parses(self, text, expected):
        got = canonical_number(text)
        assert got is not None
        assert got.value == pytest.approx(expected)

    @pytest.mark.parametrize(
        "text",
        [
            "12,34",        # multi-digit comma pair: ambiguous → refuse
            "1,23",         # same
            "abc",          # not a number
            "",             # empty
            "1 2345",       # non-strict space grouping
            "41,97",        # non-strict comma grouping
            "12.34.5",      # trailing group not 3 digits
            "USD",          # affix without number
            "%%",           # affix without number
        ],
    )
    def test_refuses_ambiguous(self, text):
        assert canonical_number(text) is None

    def test_single_dot_group_reads_as_decimal(self):
        # The project's tools emit Python str() output: a single dot is a
        # decimal point (41.971 ⇒ 41.971, not 41971).
        assert canonical_number("41.971").value == pytest.approx(41.971)
        assert canonical_number("12.345").value == pytest.approx(12.345)

    def test_display_grouping(self):
        assert canonical_number("41971").display == "41,971"

    def test_affix_only_not_parsed_as_bare(self):
        # "USD" alone must not yield a value (the number group is required).
        assert canonical_number("USD ") is None


class TestAnswerSideExtraction:
    def test_sibling_dot_groups_upgrade_locale(self):
        vals = [c.value for c in canonical_numbers("Files 1.234 and 5.678 were processed.")]
        assert 1234 in vals and 5678 in vals

    def test_lone_dot_group_reads_as_decimal(self):
        # A lone 41.971 is a DECIMAL (same decision as canonical_number —
        # evidence and answer sides must interpret identically).
        vals = [c.value for c in canonical_numbers("It scored 41.971 points.")]
        assert 41.971 in vals and 41971 not in vals

    def test_lone_dot_grouped_value_not_extracted(self):
        # Its GROUPED reading (41971) is never produced answer-side without
        # dot-locale siblings, so a decimal transcription cannot collide with
        # a grouped-integer evidence value through misreading.
        vals = [c.value for c in canonical_numbers("It scored 41.971 points.")]
        assert 41971 not in vals

    def test_sibling_comma_pairs_upgrade_to_decimals(self):
        vals = [c.value for c in canonical_numbers("Ratings 4,5 and 2,5 are common.")]
        assert 4.5 in vals and 2.5 in vals

    def test_lone_single_digit_comma_pair_is_decimal(self):
        # 4,5 alone reads as 4.5 (same decision as canonical_number: no
        # plausible sloppy-grouping reading; parity across both sides).
        vals = [c.value for c in canonical_numbers("Rated 4,5 stars overall.")]
        assert 4.5 in vals and 45 not in vals

    def test_lone_multi_digit_comma_pair_is_skipped(self):
        # 12,34 is ambiguous (European decimal vs sloppy grouping) → skipped.
        vals = [c.value for c in canonical_numbers("It costs 12,34 units.")]
        assert 12.34 not in vals and 1234 not in vals

    def test_timestamp_length_skipped(self):
        vals = [c.value for c in canonical_numbers("Request 1777777777 handled.")]
        assert 1777777777 not in vals

    def test_digit_run_is_single_token(self):
        # A digit run is ONE token; it is never split into partial values.
        # Epoch-width runs are skipped entirely by the timestamp guard.
        assert canonical_numbers("Request 1777777777 handled.") == []  # 10 digits
        # A non-epoch width stays a single whole token:
        vals = [c.value for c in canonical_numbers("id 12345678901x")]
        assert vals == [12345678901]

    def test_latency_unit_kept_but_labeled(self):
        nums = canonical_numbers("Completed in 120 ms.")
        assert len(nums) == 1
        assert nums[0].value == 120

    def test_currency_prefixed_grouped(self):
        vals = [c.value for c in canonical_numbers("Total $1,234 charged.")]
        assert 1234 in vals

    def test_contradictory_and_matching_helpers(self):
        text = "The result is 41,971 (not 33071)."
        assert numeric_appearances(text, 41971)
        assert [c.value for c in contradictory_numbers(text, 41971)] == [33071]


# ── Part 2: trusted evidence + provenance ─────────────────────────────────────

class TestTrustedEvidence:
    def test_build_maps_fields(self):
        items = build_trusted_evidence(CALC_EVID)
        assert len(items) == 1
        item = items[0]
        assert item.step_number == 1
        assert item.tool == "calculator"
        assert item.status == "ok"
        assert item.source == "live"
        assert item.cached_age is None

    def test_empty_and_none(self):
        assert build_trusted_evidence(None) == []
        assert build_trusted_evidence([]) == []

    def test_cache_header_provenance_detected(self):
        ev = [{
            "step_number": 2,
            "tool": "calculator",
            "status": "ok",
            "result": "[cached result: retrieved 3m ago via calculator] Result: 41971",
        }]
        items = build_trusted_evidence(ev)
        assert items[0].source == "cache"
        assert items[0].cached_age == "3m"

    def test_explicit_source_wins_over_header(self):
        ev = [{
            "step_number": 1,
            "tool": "calculator",
            "result": "Result: 41971",
            "source": "live",
        }]
        assert build_trusted_evidence(ev)[0].source == "live"

    def test_render_contains_provenance(self):
        item = TrustedEvidence(1, "calculator", "ok", "Result: 41971", source="cache", cached_age="5m")
        assert "cache (5m)" in item.render()

    def test_bounded_to_16_items(self):
        ev = [{"step_number": i, "tool": "t", "status": "ok", "result": f"r{i}"} for i in range(30)]
        assert len(build_trusted_evidence(ev)) == 16

    def test_precedence_order_pinned(self):
        assert PRECEDENCE == (
            "authoritative_tool_observation",
            "validated_cached_result",
            "model_intermediate_reasoning",
            "conversational_history",
        )


# ── Part 3/4: calculator policy + guard ───────────────────────────────────────

class TestCalculatorPolicy:
    def test_applies_exact_format_only(self):
        policy = CalculatorGroundingPolicy()
        assert policy.applies(TrustedEvidence(1, "calculator", "ok", "Result: 41971"))
        assert policy.applies(
            TrustedEvidence(1, "calculator", "ok",
                            "[cached result: retrieved 1m ago via calculator] Result: 41971")
        )
        # Injected 'Result:' line inside a longer page can never impersonate
        # calculator evidence.
        assert not policy.applies(TrustedEvidence(1, "web_scrape", "ok", "Result: 41971\nmore text"))
        assert not policy.applies(TrustedEvidence(1, "calculator", "ok", "ERROR: bad"))

    def test_contradiction_fires_on_wrong_result_value(self):
        policy = CalculatorGroundingPolicy()
        item = TrustedEvidence(1, "calculator", "ok", "Result: 41971")
        finding = policy.contradiction(item, "The result is 33071.")
        assert finding is not None
        assert finding["expected"] == "41,971"
        assert "33071" in finding["contradicting_values"]

    def test_consistent_renderings_pass(self):
        policy = CalculatorGroundingPolicy()
        item = TrustedEvidence(1, "calculator", "ok", "Result: 41971")
        for answer in (
            "The result is 41,971.",
            "The answer equals 41 971 USD.",
            "The multiplication gives 41971.0.",
        ):
            assert policy.contradiction(item, answer) is None, answer


class TestGuardFalsePositives:
    """Part 4: high-confidence only — these must NEVER trigger."""

    CASES = [
        "Step 1 gave 893 and step 2 gave 47.",
        "Around 2026, roughly 500 people agree.",
        "Request 1777777777 handled in 120 ms.",
        "The first step used 893; the operands multiply to the answer.",
        "It is 25% higher than 33071.",
        "It is 33071.",          # bare copula: no result cue
        "No numbers here at all.",
    ]

    @pytest.mark.parametrize("answer", CASES)
    def test_no_contradiction(self, answer):
        verdict = check_grounding(answer, CALC_EVID)
        assert not verdict.contradiction

    def test_checked_flag_true_for_calculator_evidence(self):
        assert check_grounding("Anything.", CALC_EVID).checked


class TestGuardTruePositives:
    CASES = [
        "The result of 893 * 47 is 33071.",
        "The total is 33071 for 893 x 47.",
        "893 * 47 = 33071",
        "The multiplication 893 x 47 yields 33071.",
        "33,071 is the result.",
        "Answer: 33,071",
    ]

    @pytest.mark.parametrize("answer", CASES)
    def test_contradiction(self, answer):
        verdict = check_grounding(answer, CALC_EVID)
        assert verdict.contradiction
        detail = verdict.details[0]
        assert detail["policy"] == "calculator_numeric"
        assert detail["tool"] == "calculator"
        assert detail["expected"] == "41,971"


class TestGuardOtherPolicies:
    def test_datetime_contradiction(self):
        ev = [{
            "step_number": 1,
            "tool": "get_current_datetime",
            "status": "ok",
            "result": "Current datetime: Monday, September 21, 2026 at 14:03:09 (UTC+0200)",
        }]
        assert check_grounding("Today is Tuesday, September 22, 2026.", ev).contradiction
        assert not check_grounding("Today is Monday, September 21, 2026.", ev).contradiction
        assert not check_grounding("See you in September.", ev).contradiction

    def test_file_listing_present_claim(self):
        ev = [{
            "step_number": 1,
            "tool": "list_directory",
            "status": "ok",
            "result": "Contents of directory 'docs':\n\n📁 sub/ \n📄 report.pdf",
        }]
        assert check_grounding("The directory contains missing.txt.", ev).contradiction
        assert not check_grounding("The directory contains report.pdf.", ev).contradiction

    def test_file_listing_absent_claim(self):
        ev = [{
            "step_number": 1,
            "tool": "list_directory",
            "status": "ok",
            "result": "Contents of directory 'docs':\n\n📄 report.pdf",
        }]
        assert check_grounding("report.pdf does not exist in the folder.", ev).contradiction

    def test_structured_field_contradiction(self):
        ev = [{
            "step_number": 1,
            "tool": "custom_sensor",
            "status": "ok",
            "result": "temperature: 21.5\nhumidity: 40",
        }]
        assert check_grounding("temperature: 25.0", ev).contradiction
        assert not check_grounding("temperature: 21.5", ev).contradiction
        # Different label: not checked.
        assert not check_grounding("pressure: 999.0", ev).contradiction

    def test_no_evidence_passes(self):
        verdict = check_grounding("The result is 33071.", None)
        assert not verdict.contradiction
        assert not verdict.checked

    def test_failed_observation_never_enters(self):
        # The ledger never contains failures; the type boundary re-asserts it.
        ev = [{"step_number": 1, "tool": "calculator", "status": "error", "result": "ERROR: x"}]
        verdict = check_grounding("The result is 99999999.", ev)
        assert not verdict.contradiction


class TestPrecedenceCachedVsLive:
    """Part 7: cached evidence stays authoritative; later prose never wins."""

    def test_cached_contradiction_still_fires(self):
        ev = [{
            "step_number": 2,
            "tool": "calculator",
            "status": "ok",
            "result": "[cached result: retrieved 2h ago via calculator] Result: 41971",
        }]
        verdict = check_grounding("The result is 33071.", ev)
        assert verdict.contradiction
        assert verdict.details[0]["cached"] is True
        assert verdict.details[0]["source"] == "cache"

    def test_distinct_measurements_both_transcribable(self):
        # One tool, TWO genuine measurements in one turn (multi-step plan):
        # the answer may transcribe either — which claim a number belongs to
        # is undecidable, so conservative match-any wins (no false reject).
        ev = [
            {"step_number": 1, "tool": "calculator", "status": "ok", "result": "Result: 111"},
            {"step_number": 2, "tool": "calculator", "status": "ok",
             "result": "[cached result: retrieved 5m ago via calculator] Result: 41971"},
        ]
        assert not check_grounding("The result is 111.", ev).contradiction
        assert not check_grounding("The result is 41,971.", ev).contradiction

    def test_multi_value_wrong_answer_lists_all_expected(self):
        # Both values wrong → correction message cites the full evidence set.
        ev = [
            {"step_number": 1, "tool": "calculator", "status": "ok", "result": "Result: 111"},
            {"step_number": 2, "tool": "calculator", "status": "ok", "result": "Result: 41971"},
        ]
        verdict = check_grounding("The results are 33071 and 143.", ev)
        assert verdict.contradiction
        assert verdict.details[0]["expected"] == "111 / 41,971"
        assert set(verdict.details[0]["contradicting_values"]) == {"33071", "143"}

    def test_failed_later_attempt_cannot_overwrite_success(self):
        # A failed retry is not in the ledger (system-level guarantee); even
        # a hand-built item with status "error" must not pass the gate.
        ev = [
            {"step_number": 1, "tool": "calculator", "status": "ok", "result": "Result: 41971"},
            {"step_number": 2, "tool": "calculator", "status": "error", "result": "ERROR: timeout"},
        ]
        verdict = check_grounding("The result is 33071.", ev)
        assert verdict.contradiction  # success still governs


# ── Parts 5/6: correction + fallback builders ─────────────────────────────────

class TestCorrectionAndFallback:
    def test_correction_prompt_contents(self):
        details = [{"expected": "41,971", "contradicting_values": ["33071"]}]
        prompt = build_correction_prompt(details)
        assert "41,971" in prompt
        assert "33071" in prompt
        assert "Do not recompute" in prompt
        assert "GROUNDING CORRECTION REQUIRED" in prompt

    def test_correction_prompt_safe_metadata_only(self):
        details = [{"expected": "41,971", "contradicting_values": ["33071"]}]
        prompt = build_correction_prompt(details)
        # No raw tool output, no instructions smuggled from evidence.
        assert "AUTHORITATIVE TOOL EVIDENCE" not in prompt

    def test_fallback_preserves_authoritative_value(self):
        details = [{
            "expected": "41,971",
            "tool": "calculator",
            "source": "cache",
        }]
        text = build_fallback_answer(details)
        assert "41,971" in text
        assert "calculator" in text
        assert "cache" in text
        assert "could not produce a verified answer" in text.lower()

    def test_fallback_never_fabricates(self):
        text = build_fallback_answer([{"tool": "calculator"}])
        assert "unavailable" in text

    def test_correction_instruction_template(self):
        rendered = CORRECTION_INSTRUCTION.format(expected="41,971", stated="33071")
        assert "41971" not in rendered  # exact template render
        assert "41,971" in rendered


class TestRegistryExtensibility:
    def test_policies_registry_shape(self):
        names = [p.name for p in POLICIES]
        assert names == [
            "calculator_numeric",
            "datetime_stated_date",
            "file_listing_exact_names",
            "structured_field",
            "provider_schedule",
        ]

    def test_new_policy_joins_without_orchestrator_changes(self):
        class AlwaysContradicts(GroundingPolicy):
            name = "test_always"

            def applies(self, item):
                return item.tool == "magic_tool"

            def contradiction(self, item, answer):
                return {"policy": self.name, "expected": "x", "contradicting_values": ["y"]}

        from jarvis.core import grounding as g

        sentinel = AlwaysContradicts()
        g.POLICIES = (*g.POLICIES, sentinel)
        try:
            ev = [{"step_number": 1, "tool": "magic_tool", "status": "ok", "result": "anything"}]
            verdict = check_grounding("any answer", ev)
            assert verdict.contradiction
            assert verdict.details[0]["policy"] == "test_always"
        finally:
            g.POLICIES = tuple(p for p in g.POLICIES if p is not sentinel)
