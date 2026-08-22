from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app.agent.core import AgentCore
from app.api.main import create_app
from app.models.generation import TextGenerationRequest
from app.models.retrieval import SearchResult
from app.services.profile_service import ProfileService


class StubProfileService(ProfileService):
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, str]] = []

    def search(self, query: str, visibility: str = "public") -> list[SearchResult]:  # type: ignore[override]
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


class StubTextGenerator:
    def generate(self, request: TextGenerationRequest) -> str:
        return "stub grounded response"


class HttpApiTests(unittest.TestCase):
    def client_for(self, results: list[SearchResult]) -> tuple[TestClient, StubProfileService]:
        profile_service = StubProfileService(results)
        core = AgentCore(profile_service=profile_service)
        return TestClient(
            create_app(agent_core=core, text_generator=StubTextGenerator())
        ), profile_service

    def test_health_endpoint(self) -> None:
        client, _ = self.client_for([])
        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_prepare_endpoint_serializes_ready_turn(self) -> None:
        client, profile_service = self.client_for([result("mcp-project", 90.0)])
        response = client.post(
            "/agent/prepare",
            json={"query": "  experiencia   MCP  ", "max_results": 3},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["query"], "experiencia MCP")
        self.assertEqual(body["status"], "ready")
        self.assertEqual(body["evidence"][0]["entity_id"], "mcp-project")
        self.assertIn("name", body["policy"])
        self.assertEqual(profile_service.calls, [("experiencia MCP", "public")])

    def test_prepare_endpoint_serializes_insufficient_evidence(self) -> None:
        client, _ = self.client_for([])
        response = client.post("/agent/prepare", json={"query": "unknown"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "insufficient_evidence")
        self.assertEqual(response.json()["evidence"], [])

    def test_whitespace_only_query_maps_agent_error_to_422(self) -> None:
        client, _ = self.client_for([])
        response = client.post("/agent/prepare", json={"query": "   \n\t "})

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json(), {"detail": "query must not be empty"})

    def test_invalid_max_results_is_rejected_at_http_boundary(self) -> None:
        client, _ = self.client_for([])
        for value in (0, AgentCore.MAX_RESULTS + 1, True, 1.5):
            with self.subTest(value=value):
                response = client.post(
                    "/agent/prepare",
                    json={"query": "project", "max_results": value},
                )
                self.assertEqual(response.status_code, 422)

    def test_unknown_request_fields_are_rejected(self) -> None:
        client, _ = self.client_for([])
        response = client.post(
            "/agent/prepare",
            json={"query": "project", "visibility": "internal_summary"},
        )

        self.assertEqual(response.status_code, 422)

    def test_max_results_is_forwarded_without_changing_ranking(self) -> None:
        client, _ = self.client_for(
            [result("first", 100.0), result("second", 90.0), result("third", 80.0)]
        )
        response = client.post(
            "/agent/prepare",
            json={"query": "project", "max_results": 2},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            [item["entity_id"] for item in response.json()["evidence"]],
            ["first", "second"],
        )

    def test_open_responses_endpoint_is_available_in_phase_5(self) -> None:
        client, _ = self.client_for([])
        response = client.post(
            "/v1/responses",
            json={"model": "banorte-cv-agent", "input": "hello"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["object"], "response")


if __name__ == "__main__":
    unittest.main()
