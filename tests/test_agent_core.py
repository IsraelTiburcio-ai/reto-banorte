from __future__ import annotations

import unittest

from app.agent.core import AgentCore, AgentInputError
from app.agent.policy import DEFAULT_AGENT_POLICY
from app.models.agent import AgentPolicy
from app.models.retrieval import SearchResult, VisibilityPolicy


class FakeRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, VisibilityPolicy]] = []

    def search(
        self, query: str, visibility: VisibilityPolicy = "public"
    ) -> list[SearchResult]:
        self.calls.append((query, visibility))
        return self.results


def result(entity_id: str, score: float = 50.0) -> SearchResult:
    return SearchResult(
        entity_type="project",
        entity_id=entity_id,
        title=entity_id.replace("-", " ").title(),
        score=score,
        matched_fields=("name/title",),
        data={
            "id": entity_id,
            "visibility": "public",
            "nested": {"values": ["original"]},
        },
    )


class AgentCoreTests(unittest.TestCase):
    def test_prepare_packages_ranked_evidence(self) -> None:
        retriever = FakeRetriever([result("first", 90.0), result("second", 80.0)])
        turn = AgentCore(retriever=retriever).prepare("MCP")

        self.assertEqual(turn.status, "ready")
        self.assertEqual([item.entity_id for item in turn.evidence], ["first", "second"])
        self.assertEqual(turn.query, "MCP")

    def test_prepare_enforces_public_visibility(self) -> None:
        retriever = FakeRetriever([result("public-project")])
        AgentCore(retriever=retriever).prepare("project")
        self.assertEqual(retriever.calls, [("project", "public")])

    def test_callers_cannot_request_internal_visibility(self) -> None:
        self.assertNotIn("visibility", AgentCore.prepare.__annotations__)

    def test_empty_query_is_rejected(self) -> None:
        core = AgentCore(retriever=FakeRetriever([]))
        for query in ("", "   ", "\n\t"):
            with self.subTest(query=query):
                with self.assertRaises(AgentInputError):
                    core.prepare(query)

    def test_non_string_query_is_rejected_at_runtime(self) -> None:
        core = AgentCore(retriever=FakeRetriever([]))
        with self.assertRaises(AgentInputError):
            core.prepare(None)  # type: ignore[arg-type]

    def test_query_whitespace_is_normalized_before_retrieval(self) -> None:
        retriever = FakeRetriever([])
        turn = AgentCore(retriever=retriever).prepare("  experiencia   en   MCP  ")
        self.assertEqual(turn.query, "experiencia en MCP")
        self.assertEqual(retriever.calls[0][0], "experiencia en MCP")

    def test_no_results_produce_insufficient_evidence_status(self) -> None:
        turn = AgentCore(retriever=FakeRetriever([])).prepare("unknown topic")
        self.assertEqual(turn.status, "insufficient_evidence")
        self.assertEqual(turn.evidence, ())

    def test_result_limit_is_applied_after_retrieval_ranking(self) -> None:
        retriever = FakeRetriever([result(f"project-{index}", 100 - index) for index in range(5)])
        turn = AgentCore(retriever=retriever).prepare("project", max_results=2)
        self.assertEqual([item.entity_id for item in turn.evidence], ["project-0", "project-1"])

    def test_result_limit_has_conservative_bounds(self) -> None:
        core = AgentCore(retriever=FakeRetriever([]))
        for value in (0, -1, AgentCore.MAX_RESULTS + 1, True, 1.5):
            with self.subTest(value=value):
                with self.assertRaises(AgentInputError):
                    core.prepare("project", max_results=value)  # type: ignore[arg-type]

    def test_prepared_evidence_is_detached_from_retrieval_result(self) -> None:
        original = result("safe-project")
        turn = AgentCore(retriever=FakeRetriever([original])).prepare("safe")
        original.data["nested"]["values"].append("mutated later")  # type: ignore[index,union-attr]

        nested = turn.evidence[0].data["nested"]
        self.assertIsInstance(nested, dict)
        self.assertEqual(nested["values"], ["original"])

    def test_default_policy_contains_grounding_and_calibration_rules(self) -> None:
        rules = " ".join(DEFAULT_AGENT_POLICY.rules).casefold()
        self.assertIn("ground", rules)
        self.assertIn("insufficient", rules)
        self.assertIn("skill levels", rules)
        self.assertIn("ownership", rules)
        self.assertIn("approximate metrics", rules)
        self.assertIn("sensitive information", rules)

    def test_custom_policy_can_be_injected_without_changing_retrieval(self) -> None:
        policy = AgentPolicy(
            name="test-policy",
            objective="Test objective",
            rules=("Test rule",),
        )
        retriever = FakeRetriever([result("project")])
        turn = AgentCore(retriever=retriever, policy=policy).prepare("project")
        self.assertIs(turn.policy, policy)
        self.assertEqual(retriever.calls, [("project", "public")])


if __name__ == "__main__":
    unittest.main()
