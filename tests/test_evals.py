from __future__ import annotations

import contextlib
import io
import os
import unittest
from unittest.mock import patch

from evals.metrics import render_report
from evals.runner import (
    VALID_CATEGORIES,
    build_parser,
    load_cases,
    main,
    run_offline,
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
                    set(case.expected_evidence_ids)
                    & set(case.forbidden_evidence_ids)
                )
                self.assertFalse(
                    {claim.casefold() for claim in case.required_facts}
                    & {claim.casefold() for claim in case.forbidden_claims}
                )
                if case.should_abstain:
                    self.assertFalse(case.expected_evidence_ids)
                    self.assertFalse(case.required_facts)

    def test_offline_runner_is_reproducible_and_does_not_call_provider(self) -> None:
        with patch("app.llm.openai_provider.OpenAI") as openai_client:
            first = run_offline(self.cases)
            second = run_offline(self.cases)

        self.assertEqual(first, second)
        self.assertGreater(first.total, 0)
        self.assertEqual(first.total, first.passed + first.failed)
        openai_client.assert_not_called()
        natural_mcp = next(
            outcome
            for outcome in first.outcomes
            if outcome.case_id == "natural-language-mcp"
        )
        self.assertTrue(natural_mcp.passed)

    def test_known_followup_limitation_is_reported_not_hidden(self) -> None:
        report = run_offline(self.cases)
        followup = next(
            outcome
            for outcome in report.outcomes
            if outcome.case_id == "conversation-followup"
        )
        self.assertFalse(followup.passed)
        self.assertIn("expected_status", followup.issue)

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
        self.assertIn("TOTAL:", output.getvalue())

    def test_results_do_not_contain_secrets(self) -> None:
        rendered = render_report(run_offline(self.cases))
        self.assertNotIn("OPENAI_API_KEY", rendered)
        configured_key = os.getenv("OPENAI_API_KEY", "").strip()
        if configured_key:
            self.assertNotIn(configured_key, rendered)


if __name__ == "__main__":
    unittest.main()
