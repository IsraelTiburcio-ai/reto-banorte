from __future__ import annotations

import contextlib
import io
import unittest
from unittest.mock import patch

from scripts.pre_banorte_smoke import (
    PROPERTY_DIMENSIONS,
    _deterministic_check,
    load_cases,
    run,
)


class PreBanorteSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = {case["id"]: case for case in load_cases()}

    def test_fixture_contains_thirty_unique_cases(self) -> None:
        self.assertEqual(len(self.cases), 30)
        self.assertTrue(any(case["multi_turn"] for case in self.cases.values()))
        self.assertTrue(any(case["stream"] for case in self.cases.values()))

    def test_fixture_properties_are_classified(self) -> None:
        for case in self.cases.values():
            properties = set(case["required_properties"]) | set(case["forbidden_properties"])
            self.assertTrue(properties <= PROPERTY_DIMENSIONS.keys(), case["id"])

    def test_offline_mode_does_not_fake_generated_answers(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = run(None, None)

        self.assertEqual(result, 0)
        self.assertIn("30/30", output.getvalue())
        self.assertIn("HTTP calls: 0", output.getvalue())
        self.assertIn("DETERMINISTIC: 0 cases; 0 PASS, 0 REVIEW, 0 FAIL", output.getvalue())
        self.assertIn("GENERATED: 30 cases NOT_RUN", output.getvalue())

    def test_semantic_checker_remains_separate_from_offline_execution(self) -> None:
        case = self.cases["meta-agent-identity"]
        self.assertEqual(
            _deterministic_check(case, "Soy el agente de CV de Israel Tiburcio."),
            "PASS",
        )
        self.assertNotEqual(_deterministic_check(case, "banana"), "PASS")

    def test_remote_execution_semantics_are_not_auto_passed(self) -> None:
        output = io.StringIO()
        with patch(
            "scripts.pre_banorte_smoke._post_case",
            return_value="generated answer",
        ), contextlib.redirect_stdout(output):
            result = run("https://example.invalid", None, limit=1)

        self.assertEqual(result, 0)
        self.assertIn("execution=NOT_OBSERVABLE", output.getvalue())


if __name__ == "__main__":
    unittest.main()
