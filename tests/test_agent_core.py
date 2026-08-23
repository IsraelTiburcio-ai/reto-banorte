from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

from app.agent.core import AgentCore, AgentInputError
from app.agent.policy import DEFAULT_AGENT_POLICY
from app.models.agent import AgentPolicy
from app.models.retrieval import SearchResult, VisibilityPolicy
from app.services.profile_service import ProfileService


def visibility_fixture() -> dict[str, object]:
    return {
        "metadata": {"visibility": "public", "schema_version": "1.0"},
        "identity": {"visibility": "public", "full_name": "Test Profile"},
        "professional_summary": {
            "visibility": "public",
            "profile": "Public professional summary",
        },
        "career_story": {"visibility": "public", "narrative": "Public career"},
        "experience": [
            {
                "id": "hidden-experience",
                "visibility": "do_not_expose",
                "name": "hidden relationship origin",
                "project_ids": ["public-project"],
            }
        ],
        "projects": [
            {
                "id": "public-project",
                "visibility": "public",
                "name": "Public project",
                "description": "public evidence",
                "nested": {"values": ["original"]},
                "internal_summary_details": {
                    "visibility": "internal_summary",
                    "text": "nested internal-only phrase",
                },
                "private_details": {
                    "visibility": "do_not_expose",
                    "text": "nested private-only phrase",
                },
            },
            {
                "id": "internal-project",
                "visibility": "internal_summary",
                "name": "internal-only root term",
            },
            {
                "id": "hidden-project",
                "visibility": "do_not_expose",
                "name": "hidden-only root term",
            },
        ],
        "education": {"formal": []},
        "training": [],
        "achievements": [],
        "hackathons": {"visibility": "public", "activities": []},
        "skills": [],
        "knowledge_areas": [],
        "working_style": {"visibility": "public", "principles": []},
        "agent_policies": [],
    }


class RecordingProfileService(ProfileService):
    """A real ProfileService that records calls without replacing retrieval."""

    def __init__(self, profile_path: Path) -> None:
        super().__init__(profile_path)
        self.calls: list[tuple[str, VisibilityPolicy]] = []

    def search(
        self, query: str, visibility: VisibilityPolicy = "public"
    ) -> list[SearchResult]:
        self.calls.append((query, visibility))
        return super().search(query, visibility)


class AgentCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.fixture_path = Path(self.temp_dir.name) / "profile.json"
        self.fixture_path.write_text(
            json.dumps(visibility_fixture()), encoding="utf-8"
        )
        self.profile_service = ProfileService(self.fixture_path)
        self.core = AgentCore(profile_service=self.profile_service)

    def test_prepare_packages_ranked_evidence(self) -> None:
        turn = AgentCore().prepare("MCP")

        self.assertEqual(turn.status, "ready")
        self.assertTrue(turn.evidence)
        scores = [item.score for item in turn.evidence]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(turn.query, "MCP")

    def test_natural_language_mcp_query_is_ready(self) -> None:
        turn = AgentCore().prepare("¿Qué experiencia tiene Israel con MCP?")

        self.assertEqual(turn.status, "ready")
        self.assertIn("mcp", {item.entity_id for item in turn.evidence})
        scores = [item.score for item in turn.evidence]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(turn.query, "¿Qué experiencia tiene Israel con MCP?")

    def test_core_keeps_retrieval_lexical_and_public_only(self) -> None:
        turn = AgentCore().prepare("¿Quién es Israel?")

        self.assertEqual(turn.status, "insufficient_evidence")
        self.assertEqual(turn.evidence, ())
        self.assertEqual(
            AgentCore().public_profile()["identity"]["visibility"], "public"
        )

    def test_prepare_enforces_public_visibility(self) -> None:
        profile_service = RecordingProfileService(self.fixture_path)
        turn = AgentCore(profile_service=profile_service).prepare("public")

        self.assertEqual(turn.status, "ready")
        self.assertEqual(profile_service.calls, [("public", "public")])

    def test_callers_cannot_request_internal_visibility(self) -> None:
        constructor_parameters = inspect.signature(AgentCore).parameters
        prepare_parameters = inspect.signature(AgentCore.prepare).parameters

        self.assertIn("profile_service", constructor_parameters)
        self.assertNotIn("retriever", constructor_parameters)
        self.assertNotIn("visibility", prepare_parameters)

    def test_agent_only_returns_public_profile_evidence(self) -> None:
        public_turn = self.core.prepare("public evidence")
        self.assertEqual(public_turn.status, "ready")
        self.assertEqual(
            [item.entity_id for item in public_turn.evidence], ["public-project"]
        )

        internal_results = self.profile_service.search(
            "internal-only root term", visibility="internal_summary"
        )
        self.assertTrue(internal_results)
        self.assertTrue(
            self.profile_service.search(
                "nested internal-only phrase", visibility="internal_summary"
            )
        )

        for query in (
            "internal-only root term",
            "hidden-only root term",
            "nested internal-only phrase",
            "nested private-only phrase",
            "hidden relationship origin",
        ):
            with self.subTest(query=query):
                turn = self.core.prepare(query)
                self.assertEqual(turn.status, "insufficient_evidence")
                self.assertEqual(turn.evidence, ())

    def test_empty_query_is_rejected(self) -> None:
        for query in ("", "   ", "\n\t"):
            with self.subTest(query=query):
                with self.assertRaises(AgentInputError):
                    self.core.prepare(query)

    def test_non_string_query_is_rejected_at_runtime(self) -> None:
        with self.assertRaises(AgentInputError):
            self.core.prepare(None)  # type: ignore[arg-type]

    def test_query_whitespace_is_normalized_before_retrieval(self) -> None:
        profile_service = RecordingProfileService(self.fixture_path)
        turn = AgentCore(profile_service=profile_service).prepare(
            "  public   evidence  "
        )

        self.assertEqual(turn.query, "public evidence")
        self.assertEqual(profile_service.calls[0][0], "public evidence")
        self.assertEqual(profile_service.calls[0][1], "public")

    def test_no_results_produce_insufficient_evidence_status(self) -> None:
        turn = self.core.prepare("unknown topic")
        self.assertEqual(turn.status, "insufficient_evidence")
        self.assertEqual(turn.evidence, ())

    def test_result_limit_is_applied_after_retrieval_ranking(self) -> None:
        expected = self.profile_service.search("public")[:2]
        turn = self.core.prepare("public", max_results=2)

        self.assertEqual(
            [item.entity_id for item in turn.evidence],
            [item.entity_id for item in expected],
        )

    def test_result_limit_has_conservative_bounds(self) -> None:
        for value in (0, -1, AgentCore.MAX_RESULTS + 1, True, 1.5):
            with self.subTest(value=value):
                with self.assertRaises(AgentInputError):
                    self.core.prepare("public", max_results=value)  # type: ignore[arg-type]

    def test_prepared_evidence_is_detached_from_retrieval_result(self) -> None:
        turn = self.core.prepare("public evidence")
        turn.evidence[0].data["nested"]["values"].append("mutated later")  # type: ignore[index,union-attr]

        fresh_project = self.profile_service.get_project("public-project")
        self.assertEqual(fresh_project["nested"]["values"], ["original"])  # type: ignore[index]

    def test_default_policy_contains_grounding_and_calibration_rules(self) -> None:
        rules = " ".join(DEFAULT_AGENT_POLICY.rules).casefold()
        self.assertTrue("ground" in rules or "sustent" in rules)
        self.assertTrue("insufficient" in rules or "insuficiente" in rules)
        self.assertTrue("skill levels" in rules or "niveles calibrados de habilidad" in rules)
        self.assertTrue("ownership" in rules or "autoría y participación" in rules)
        self.assertTrue("approximate metrics" in rules or "métricas aproximadas" in rules)
        self.assertTrue("sensitive information" in rules or "información sensible" in rules)

    def test_custom_policy_can_be_injected_without_changing_retrieval(self) -> None:
        policy = AgentPolicy(
            name="test-policy",
            objective="Test objective",
            rules=("Test rule",),
        )
        turn = AgentCore(
            profile_service=self.profile_service, policy=policy
        ).prepare("public")

        self.assertIs(turn.policy, policy)
        self.assertEqual(turn.status, "ready")


if __name__ == "__main__":
    unittest.main()
