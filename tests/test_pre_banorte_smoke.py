from __future__ import annotations

import contextlib
import io
import unittest
from unittest.mock import patch

from app.api.open_responses_formatter import (
    AGENT_IDENTITY_RESPONSE_TEXT,
    ENGLISH_GREETING_RESPONSE_TEXT,
    GREETING_RESPONSE_TEXT,
    INSUFFICIENT_EVIDENCE_TEXT,
    SENSITIVE_REQUEST_RESPONSE_TEXT,
)
from scripts.pre_banorte_smoke import (
    PROPERTY_DIMENSIONS,
    _deterministic_check,
    load_cases,
    run,
    run_local_deterministic,
)


class PreBanorteSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = {
            case["id"]: case
            for case in load_cases()
        }

    def test_fixture_contains_thirty_unique_cases(self) -> None:
        cases = list(self.cases.values())
        self.assertEqual(len(cases), 30)
        self.assertEqual(len({case["id"] for case in cases}), 30)
        self.assertTrue(any(case["multi_turn"] for case in cases))
        self.assertTrue(any(case["stream"] for case in cases))

    def test_real_fixture_properties_are_classified(self) -> None:
        for case in self.cases.values():
            properties = set(case["required_properties"]) | set(
                case["forbidden_properties"]
            )
            self.assertTrue(properties <= PROPERTY_DIMENSIONS.keys(), case["id"])
            self.assertTrue(
                all(
                    PROPERTY_DIMENSIONS[property_name]
                    in {"SEMANTIC", "EXECUTION", "TRANSPORT"}
                    for property_name in properties
                )
            )

    def test_real_deterministic_fixture_answers_are_semantic_passes(self) -> None:
        answers = {
            "social-greeting-es": GREETING_RESPONSE_TEXT,
            "social-greeting-en": ENGLISH_GREETING_RESPONSE_TEXT,
            "meta-agent-identity": AGENT_IDENTITY_RESPONSE_TEXT,
            "sensitive-password": SENSITIVE_REQUEST_RESPONSE_TEXT,
            "unknown-kubernetes": INSUFFICIENT_EVIDENCE_TEXT,
        }
        for case_id, answer in answers.items():
            with self.subTest(case_id=case_id):
                self.assertEqual(_deterministic_check(self.cases[case_id], answer), "PASS")

    def test_real_fixtures_catch_semantic_mutations(self) -> None:
        for case_id in (
            "social-greeting-es",
            "social-greeting-en",
            "meta-agent-identity",
            "sensitive-password",
            "unknown-kubernetes",
        ):
            with self.subTest(case_id=case_id):
                self.assertNotEqual(_deterministic_check(self.cases[case_id], "banana"), "PASS")

        self.assertEqual(
            _deterministic_check(
                self.cases["meta-agent-identity"],
                "Soy Israel Tiburcio.",
            ),
            "FAIL",
        )
        self.assertEqual(
            _deterministic_check(
                self.cases["unknown-kubernetes"],
                "Israel no sabe Kubernetes.",
            ),
            "FAIL",
        )

    def test_local_deterministic_pipeline_is_derived_from_real_fixtures(self) -> None:
        expected_cases = {
            case_id
            for case_id, case in self.cases.items()
            if case["provider"] == "no"
        }
        results = run_local_deterministic(list(self.cases.values()))

        self.assertEqual({result.case_id for result in results}, expected_cases)
        self.assertTrue(results)
        self.assertTrue(all(result.transport == "PASS" for result in results))
        self.assertTrue(all(result.execution == "PASS" for result in results))
        self.assertTrue(all(result.semantic == "PASS" for result in results))
        self.assertTrue(all(result.overall == "PASS" for result in results))

    def test_default_mode_is_offline_and_makes_no_http_calls(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = run(None, None)

        self.assertEqual(result, 0)
        self.assertIn("30/30", output.getvalue())
        self.assertIn("HTTP calls: 0", output.getvalue())
        self.assertIn("DETERMINISTIC: 11 cases; 11 PASS, 0 REVIEW, 0 FAIL", output.getvalue())
        self.assertIn("GENERATED: 19 cases NOT_RUN", output.getvalue())

    def test_semantic_checker_does_not_use_execution_properties(self) -> None:
        for case_id, answer in (
            ("meta-agent-identity", AGENT_IDENTITY_RESPONSE_TEXT),
            ("sensitive-password", SENSITIVE_REQUEST_RESPONSE_TEXT),
        ):
            with self.subTest(case_id=case_id):
                self.assertEqual(_deterministic_check(self.cases[case_id], answer), "PASS")

    def test_remote_execution_is_not_observable_but_semantics_are_independent(self) -> None:
        output = io.StringIO()
        with patch(
            "scripts.pre_banorte_smoke._post_case",
            return_value=GREETING_RESPONSE_TEXT,
        ), contextlib.redirect_stdout(output):
            result = run("https://example.invalid", None, limit=1)

        self.assertEqual(result, 0)
        self.assertIn("0 PASS, 0 REVIEW, 0 FAIL, 1 NOT_OBSERVABLE", output.getvalue())
        self.assertIn("1 PASS, 0 REVIEW, 0 FAIL", output.getvalue())
        self.assertIn("execution=NOT_OBSERVABLE, semantic=PASS", output.getvalue())

    def test_explicit_semantic_violations_fail_and_safe_unknown_passes(self) -> None:
        self.assertEqual(
            _deterministic_check(
                self.cases["unknown-kubernetes"],
                "Israel no sabe Kubernetes.",
            ),
            "FAIL",
        )
        self.assertEqual(
            _deterministic_check(
                self.cases["unknown-kubernetes"],
                "No encontré evidencia pública suficiente para afirmar que tenga experiencia con Kubernetes.",
            ),
            "PASS",
        )

    def test_refusal_word_contrasena_is_not_a_secret_leak(self) -> None:
        self.assertEqual(
            _deterministic_check(
                self.cases["sensitive-password"],
                "No puedo proporcionar contraseñas ni datos sensibles. Puedo hablar del perfil profesional.",
            ),
            "PASS",
        )


if __name__ == "__main__":
    unittest.main()
