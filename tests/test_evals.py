from __future__ import annotations

import contextlib
import io
import os
import unittest
from dataclasses import replace
from unittest.mock import patch

from app.agent.core import AgentCore
from evals.metrics import (
    FAIL,
    NOT_APPLICABLE,
    NOT_EVALUATED,
    PASS,
    EvalOutcome,
    EvalReport,
    render_report,
)
from evals.runner import (
    VALID_CATEGORIES,
    build_parser,
    evaluate_case,
    load_cases,
    main,
    run_offline,
    run_live,
)


class EvalInfrastructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_cases()

    def test_dataset_is_valid_and_has_unique_ids(self) -> None:
        self.assertGreaterEqual(len(self.cases), 15)
        ids = [case.id for case in self.cases]
        self.assertEqual(len(ids), len(set(ids)))

    def test_dataset_covers_all_declared_categories(self) -> None:
        categories = {case.category for case in self.cases}
        self.assertTrue(VALID_CATEGORIES.issubset(categories))

    def test_required_and_forbidden_contracts_are_consistent(self) -> None:
        for case in self.cases:
            with self.subTest(case=case.id):
                self.assertFalse(
                    set(case.required_evidence_ids)
                    & set(case.forbidden_evidence_ids)
                )
                self.assertFalse(
                    set(case.top_evidence_ids) & set(case.forbidden_evidence_ids)
                )
                self.assertFalse(
                    {claim.casefold() for claim in case.required_facts}
                    & {claim.casefold() for claim in case.forbidden_claims}
                )
                if case.should_abstain:
                    self.assertFalse(case.required_evidence_ids)
                    self.assertFalse(case.top_evidence_ids)
                    self.assertFalse(case.required_facts)

    def test_offline_runner_is_reproducible_and_does_not_call_provider(self) -> None:
        with patch("app.llm.openai_provider.OpenAI") as openai_client:
            first = run_offline(self.cases)
            second = run_offline(self.cases)

        self.assertEqual(first, second)
        self.assertGreater(first.total, 0)
        self.assertEqual(
            first.total, first.passed + first.failed + first.not_evaluated
        )
        openai_client.assert_not_called()
        natural_mcp = next(
            outcome
            for outcome in first.outcomes
            if outcome.case_id == "natural-language-mcp"
        )
        self.assertEqual(natural_mcp.case_status, NOT_EVALUATED)
        self.assertEqual(natural_mcp.checks["expected_status"], PASS)
        self.assertEqual(natural_mcp.checks["evidence_package_non_empty"], PASS)
        self.assertEqual(natural_mcp.checks["required_evidence_ids"], PASS)
        self.assertEqual(natural_mcp.checks["top_evidence_ids"], PASS)
        self.assertEqual(
            natural_mcp.checks["forbidden_evidence_ids_absent"], NOT_APPLICABLE
        )
        self.assertEqual(natural_mcp.checks["required_facts_semantics"], NOT_EVALUATED)

    def test_followup_retrieval_is_observable_but_language_remains_pending(self) -> None:
        report = run_offline(self.cases)
        followup = next(
            outcome
            for outcome in report.outcomes
            if outcome.case_id == "conversation-followup"
        )
        self.assertEqual(followup.case_status, NOT_EVALUATED)
        self.assertEqual(followup.observed_status, "insufficient_evidence")
        self.assertEqual(followup.issue, "")
        self.assertEqual(followup.checks["expected_status"], PASS)
        self.assertEqual(followup.checks["evidence_package_non_empty"], PASS)
        self.assertEqual(
            followup.checks["forbidden_evidence_ids_absent"], NOT_APPLICABLE
        )
        self.assertIn("generated_answer", " ".join(followup.review_items))

    def test_live_mode_requires_explicit_confirmation_and_limit(self) -> None:
        args = build_parser().parse_args([])
        self.assertFalse(args.live)
        with self.assertRaises(SystemExit):
            main(["--live"])

    def test_default_runner_does_not_enter_live_mode(self) -> None:
        output = io.StringIO()
        with patch("evals.runner.run_live") as run_live:
            with contextlib.redirect_stdout(output):
                self.assertEqual(main([]), 0)
        run_live.assert_not_called()
        self.assertIn("TOTAL CASES:", output.getvalue())
        self.assertIn("CHECK COVERAGE:", output.getvalue())

    def test_results_do_not_contain_secrets(self) -> None:
        rendered = render_report(run_offline(self.cases))
        self.assertNotIn("OPENAI_API_KEY", rendered)
        configured_key = os.getenv("OPENAI_API_KEY", "").strip()
        if configured_key:
            self.assertNotIn(configured_key, rendered)

    def test_required_fact_impossible_is_not_reported_as_offline_pass(self) -> None:
        case = next(case for case in self.cases if case.id == "natural-language-mcp")
        mutated = replace(case, required_facts=("fact that is not in the profile",))
        outcome = evaluate_case(mutated, AgentCore())
        self.assertEqual(outcome.case_status, NOT_EVALUATED)
        self.assertEqual(outcome.checks["required_facts_semantics"], NOT_EVALUATED)
        self.assertNotEqual(outcome.checks["required_facts_semantics"], PASS)

    def test_incorrect_top_evidence_fails(self) -> None:
        case = next(case for case in self.cases if case.id == "natural-language-mcp")
        mutated = replace(case, top_evidence_ids=("mcp-analytics",), top_k=1)
        outcome = evaluate_case(mutated, AgentCore())
        self.assertEqual(outcome.case_status, FAIL)
        self.assertEqual(outcome.checks["top_evidence_ids"], FAIL)

    def test_forbidden_evidence_present_fails(self) -> None:
        case = next(case for case in self.cases if case.id == "natural-language-mcp")
        mutated = replace(case, forbidden_evidence_ids=("mcp",))
        outcome = evaluate_case(mutated, AgentCore())
        self.assertEqual(outcome.case_status, FAIL)
        self.assertEqual(outcome.checks["forbidden_evidence_ids_absent"], FAIL)

    def test_empty_forbidden_evidence_is_not_applicable(self) -> None:
        case = next(case for case in self.cases if case.id == "natural-language-mcp")
        outcome = evaluate_case(case, AgentCore())
        self.assertEqual(
            outcome.checks["forbidden_evidence_ids_absent"], NOT_APPLICABLE
        )

    def test_empty_expectations_are_not_applicable(self) -> None:
        case = next(case for case in self.cases if case.id == "conversation-followup")
        outcome = evaluate_case(case, AgentCore())
        self.assertEqual(outcome.checks["required_evidence_ids"], NOT_APPLICABLE)
        self.assertEqual(outcome.checks["top_evidence_ids"], NOT_APPLICABLE)
        self.assertEqual(outcome.checks["required_facts_semantics"], NOT_APPLICABLE)
        self.assertEqual(
            outcome.checks["forbidden_evidence_ids_absent"], NOT_APPLICABLE
        )
        self.assertEqual(
            outcome.checks["forbidden_claims_semantics"], NOT_EVALUATED
        )

    def test_live_only_and_manual_checks_are_not_offline_passes(self) -> None:
        case = next(case for case in self.cases if case.id == "factuality-mcp")
        outcome = evaluate_case(case, AgentCore())
        self.assertEqual(outcome.checks["required_facts_semantics"], NOT_EVALUATED)
        self.assertEqual(outcome.checks["forbidden_claims_semantics"], NOT_EVALUATED)
        self.assertEqual(outcome.checks["generated_answer_semantics"], NOT_EVALUATED)
        self.assertNotEqual(outcome.case_status, PASS)

    def test_cli_audits_partial_case_pass_na_and_not_evaluated_checks(self) -> None:
        case = next(case for case in self.cases if case.id == "natural-language-mcp")
        rendered = render_report(run_offline((case,)))
        self.assertIn("CASE: natural-language-mcp", rendered)
        self.assertIn("STATUS: NOT_EVALUATED", rendered)
        self.assertIn("PASS:\n  - expected_status", rendered)
        self.assertIn("  - required_evidence_ids", rendered)
        self.assertIn("NOT_EVALUATED:\n  - required_facts_semantics", rendered)
        self.assertIn("N/A:\n  - forbidden_evidence_ids_absent", rendered)

    def test_cli_audits_safe_followup_and_pending_semantics(self) -> None:
        case = next(case for case in self.cases if case.id == "conversation-followup")
        rendered = render_report(run_offline((case,)))
        self.assertIn("CASE: conversation-followup", rendered)
        self.assertIn("STATUS: NOT_EVALUATED", rendered)
        self.assertIn("PASS:\n  - expected_status", rendered)
        self.assertIn("NOT_EVALUATED:\n  - forbidden_claims_semantics", rendered)
        self.assertIn("NOT_EVALUATED:\n  - forbidden_claims_semantics", rendered)
        self.assertIn("N/A:\n  - required_evidence_ids", rendered)

    def test_metrics_exclude_not_evaluated_from_pass_rate(self) -> None:
        def outcome(case_id: str, status: str, checks: dict[str, str]) -> EvalOutcome:
            return EvalOutcome(
                case_id=case_id,
                category="factuality",
                input="q",
                expected_behavior="b",
                case_status=status,
                checks=checks,
                observed_status="ready",
                observed_evidence_ids=(),
            )

        report = EvalReport(
            (
                outcome("pass", PASS, {"retrieval": PASS, "optional": NOT_APPLICABLE}),
                outcome("fail", FAIL, {"retrieval": FAIL}),
                outcome(
                    "pending",
                    NOT_EVALUATED,
                    {"semantic": NOT_EVALUATED, "optional": NOT_APPLICABLE},
                ),
            )
        )
        self.assertEqual(report.passed, 1)
        self.assertEqual(report.failed, 1)
        self.assertEqual(report.not_evaluated, 1)
        self.assertEqual(report.executable_cases, 2)
        self.assertEqual(report.pass_rate, 0.5)
        self.assertEqual(report.checks_executed, 2)
        self.assertEqual(report.checks_declared, 5)
        self.assertEqual(report.checks_applicable, 3)
        self.assertEqual(report.checks_not_evaluated, 1)
        self.assertEqual(report.checks_not_applicable, 2)
        self.assertEqual(report.check_coverage, report.checks_executed / report.checks_applicable)
        self.assertAlmostEqual(report.check_coverage, 2 / 3)

    def test_live_limit_six_is_rejected_before_provider_use(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 1 and 5"):
            run_live(self.cases, 6)

    def test_live_without_api_key_fails_without_exposing_secret(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY is required") as error:
                run_live(self.cases, 1)
        self.assertNotIn("secret-value", str(error.exception))


if __name__ == "__main__":
    unittest.main()
