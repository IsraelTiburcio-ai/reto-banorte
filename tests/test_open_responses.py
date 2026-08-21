from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.agent.core import AgentCore
from app.api.main import create_app
from app.api.open_responses_formatter import INSUFFICIENT_EVIDENCE_TEXT
from app.api.open_responses_schemas import (
    OpenResponsesErrorEnvelope,
    OpenResponsesResponse,
)
from app.models.retrieval import SearchResult, VisibilityPolicy
from app.services.profile_service import ProfileService


def profile_fixture() -> dict[str, object]:
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
                "name": "Public MCP project",
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
    """A real ProfileService that records the trusted boundary call."""

    def __init__(self, profile_path: Path) -> None:
        super().__init__(profile_path)
        self.calls: list[tuple[str, VisibilityPolicy]] = []

    def search(
        self, query: str, visibility: VisibilityPolicy = "public"
    ) -> list[SearchResult]:
        self.calls.append((query, visibility))
        return super().search(query, visibility)


class OpenResponsesApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.profile_path = Path(self.temp_dir.name) / "profile.json"
        self.profile_path.write_text(
            json.dumps(profile_fixture()), encoding="utf-8"
        )
        self.profile_service = RecordingProfileService(self.profile_path)
        self.client = TestClient(
            create_app(agent_core=AgentCore(profile_service=self.profile_service))
        )

    def post(self, payload: object):
        return self.client.post("/v1/responses", json=payload)

    def assert_error(
        self, response, code: str, param: str | None = None
    ) -> dict[str, object]:
        self.assertEqual(response.status_code, 400)
        body = response.json()
        parsed = OpenResponsesErrorEnvelope.model_validate(body)
        self.assertEqual(parsed.error.code, code)
        self.assertEqual(parsed.error.type, "invalid_request_error")
        if param is not None:
            self.assertEqual(parsed.error.param, param)
        return body

    def test_string_input_valid(self) -> None:
        response = self.post(
            {"model": "banorte-cv-agent", "input": "public evidence"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "banorte-cv-agent")

    def test_user_message_input_valid(self) -> None:
        response = self.post(
            {
                "model": "banorte-cv-agent",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": "public evidence",
                    }
                ],
            }
        )
        self.assertEqual(response.status_code, 200)

    def test_input_text_parts_are_supported(self) -> None:
        response = self.post(
            {
                "model": "banorte-cv-agent",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "public evidence"}
                        ],
                    }
                ],
            }
        )
        self.assertEqual(response.status_code, 200)

    def test_multiple_input_text_parts_preserve_order(self) -> None:
        response = self.post(
            {
                "model": "banorte-cv-agent",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "public"},
                            {"type": "input_text", "text": "evidence"},
                        ],
                    }
                ],
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.profile_service.calls[-1][0], "public evidence")

    def test_user_assistant_user_uses_last_user_message(self) -> None:
        response = self.post(
            {
                "model": "banorte-cv-agent",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": "first question",
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": "assistant history",
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": "public evidence",
                    },
                ],
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.profile_service.calls[-1][0], "public evidence")

    def test_multiple_assistant_messages_are_structural_history(self) -> None:
        response = self.post(
            {
                "model": "banorte-cv-agent",
                "input": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": "first assistant history",
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "second assistant history"}
                        ],
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": "public evidence",
                    },
                ],
            }
        )
        self.assertEqual(response.status_code, 200)

    def test_assistant_text_is_not_forwarded_as_instruction(self) -> None:
        response = self.post(
            {
                "model": "banorte-cv-agent",
                "input": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": "internal-only root term",
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": "unknown topic",
                    },
                ],
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.profile_service.calls[-1][0], "unknown topic")

    def test_current_user_is_normalized_by_agent_core(self) -> None:
        response = self.post(
            {"model": "banorte-cv-agent", "input": "  public   evidence  "}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.profile_service.calls[-1][0], "public evidence")

    def test_retrieval_remains_public_only(self) -> None:
        response = self.post(
            {"model": "banorte-cv-agent", "input": "public evidence"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.profile_service.calls[-1][1], "public")

    def test_internal_summary_is_not_exposed(self) -> None:
        response = self.post(
            {"model": "banorte-cv-agent", "input": "internal-only root term"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("internal-only root term", response.text)

    def test_do_not_expose_is_not_exposed(self) -> None:
        response = self.post(
            {"model": "banorte-cv-agent", "input": "hidden-only root term"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("hidden-only root term", response.text)

    def test_nested_restricted_content_is_not_exposed(self) -> None:
        for query in ("nested internal-only phrase", "nested private-only phrase"):
            with self.subTest(query=query):
                response = self.post({"model": "banorte-cv-agent", "input": query})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(
                    response.json()["output"][0]["content"][0]["text"],
                    INSUFFICIENT_EVIDENCE_TEXT,
                )

    def test_response_with_evidence_is_deterministic_and_grounded(self) -> None:
        response = self.post(
            {"model": "banorte-cv-agent", "input": "public evidence"}
        )
        body = response.json()
        text = body["output"][0]["content"][0]["text"]
        self.assertIn("public-project", text)
        self.assertNotIn("internal-only root term", text)
        self.assertNotIn("hidden-only root term", text)

    def test_response_without_evidence_uses_fixed_text(self) -> None:
        response = self.post(
            {"model": "banorte-cv-agent", "input": "unknown topic"}
        )
        self.assertEqual(
            response.json()["output"][0]["content"][0]["text"],
            INSUFFICIENT_EVIDENCE_TEXT,
        )

    def test_response_object_is_response(self) -> None:
        self.assertEqual(
            self.post({"model": "m", "input": "unknown"}).json()["object"],
            "response",
        )

    def test_response_status_is_completed(self) -> None:
        self.assertEqual(
            self.post({"model": "m", "input": "unknown"}).json()["status"],
            "completed",
        )

    def test_response_role_is_assistant(self) -> None:
        body = self.post({"model": "m", "input": "unknown"}).json()
        self.assertEqual(body["output"][0]["role"], "assistant")

    def test_response_content_is_output_text(self) -> None:
        body = self.post({"model": "m", "input": "unknown"}).json()
        content = body["output"][0]["content"][0]
        self.assertEqual(content["type"], "output_text")
        self.assertEqual(content["annotations"], [])

    def test_response_ids_have_opaque_prefixes(self) -> None:
        body = self.post({"model": "m", "input": "unknown"}).json()
        self.assertTrue(body["id"].startswith("resp_"))
        self.assertTrue(body["output"][0]["id"].startswith("msg_"))
        self.assertNotIn("Israel", body["id"])

    def test_response_ids_are_distinct_between_requests(self) -> None:
        first = self.post({"model": "m", "input": "unknown"}).json()
        second = self.post({"model": "m", "input": "unknown"}).json()
        self.assertNotEqual(first["id"], second["id"])
        self.assertNotEqual(first["output"][0]["id"], second["output"][0]["id"])

    def test_response_timestamps_are_unix_ints_and_ordered(self) -> None:
        body = self.post({"model": "m", "input": "unknown"}).json()
        self.assertIsInstance(body["created_at"], int)
        self.assertIsInstance(body["completed_at"], int)
        self.assertGreaterEqual(body["completed_at"], body["created_at"])

    def test_requested_model_is_preserved(self) -> None:
        body = self.post(
            {"model": "logical-model-name", "input": "unknown"}
        ).json()
        self.assertEqual(body["model"], "logical-model-name")

    def test_usage_is_null(self) -> None:
        body = self.post({"model": "m", "input": "unknown"}).json()
        self.assertIsNone(body["usage"])
        self.assertIsNone(body["error"])

    def test_metadata_is_preserved_as_transport_data(self) -> None:
        response = self.post(
            {
                "model": "m",
                "input": "unknown",
                "metadata": {"request_id": "abc"},
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["metadata"], {"request_id": "abc"})
        self.assertEqual(self.profile_service.calls[-1][0], "unknown")

    def test_absent_model_uses_local_default(self) -> None:
        response = self.post({"input": "unknown"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "banorte-cv-agent")

    def test_null_model_uses_local_default(self) -> None:
        response = self.post({"model": None, "input": "unknown"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "banorte-cv-agent")

    def test_invalid_model_values(self) -> None:
        for model in ("", "   ", 123):
            with self.subTest(model=model):
                self.assert_error(
                    self.post({"model": model, "input": "unknown"}),
                    "invalid_model",
                    "model",
                )

    def test_missing_input(self) -> None:
        self.assert_error(self.post({"model": "m"}), "missing_input", "input")

    def test_empty_input_is_rejected(self) -> None:
        for value in ("", []):
            with self.subTest(value=value):
                self.assert_error(
                    self.post({"model": "m", "input": value}),
                    "invalid_input",
                    "input",
                )

    def test_whitespace_only_input_is_rejected(self) -> None:
        self.assert_error(
            self.post({"model": "m", "input": " \n\t "}),
            "invalid_input",
            "input",
        )

    def test_stream_true_is_rejected(self) -> None:
        self.assert_error(
            self.post({"model": "m", "input": "unknown", "stream": True}),
            "streaming_not_supported",
            "stream",
        )

    def test_system_role_is_rejected(self) -> None:
        self.assert_error(
            self.post(
                {
                    "model": "m",
                    "input": [
                        {"type": "message", "role": "system", "content": "x"}
                    ],
                }
            ),
            "unsupported_feature",
        )

    def test_developer_role_is_rejected(self) -> None:
        self.assert_error(
            self.post(
                {
                    "model": "m",
                    "input": [
                        {"type": "message", "role": "developer", "content": "x"}
                    ],
                }
            ),
            "unsupported_feature",
        )

    def test_multimodal_content_is_rejected(self) -> None:
        for content_type in ("input_image", "input_file", "input_audio", "input_video"):
            with self.subTest(content_type=content_type):
                self.assert_error(
                    self.post(
                        {
                            "model": "m",
                            "input": [
                                {
                                    "type": "message",
                                    "role": "user",
                                    "content": [{"type": content_type}],
                                }
                            ],
                        }
                    ),
                    "unsupported_feature",
                )

    def test_function_calls_and_tools_are_rejected(self) -> None:
        for payload in (
            {
                "model": "m",
                "input": [{"type": "function_call", "name": "x"}],
            },
            {"model": "m", "input": "x", "tools": []},
        ):
            with self.subTest(payload=payload):
                self.assert_error(self.post(payload), "unsupported_feature")

    def test_unknown_input_items_are_rejected(self) -> None:
        self.assert_error(
            self.post(
                {
                    "model": "m",
                    "input": [{"type": "unknown_item", "role": "user"}],
                }
            ),
            "unsupported_input_type",
        )

    def test_previous_response_id_is_rejected(self) -> None:
        self.assert_error(
            self.post(
                {"model": "m", "input": "x", "previous_response_id": "resp_old"}
            ),
            "unsupported_feature",
            "previous_response_id",
        )

    def test_store_true_is_rejected(self) -> None:
        self.assert_error(
            self.post({"model": "m", "input": "x", "store": True}),
            "unsupported_feature",
            "store",
        )

    def test_background_is_rejected(self) -> None:
        self.assert_error(
            self.post({"model": "m", "input": "x", "background": True}),
            "unsupported_feature",
            "background",
        )

    def test_compaction_is_rejected(self) -> None:
        self.assert_error(
            self.post({"model": "m", "input": "x", "compaction": True}),
            "unsupported_feature",
            "compaction",
        )

    def test_visibility_is_rejected(self) -> None:
        self.assert_error(
            self.post({"model": "m", "input": "x", "visibility": "internal_summary"}),
            "unsupported_feature",
            "visibility",
        )

    def test_unknown_top_level_field_is_rejected(self) -> None:
        self.assert_error(
            self.post({"model": "m", "input": "x", "unknown": True}),
            "invalid_input",
            "unknown",
        )

    def test_error_envelope_has_local_schema(self) -> None:
        body = self.assert_error(
            self.post({"model": "", "input": "x"}), "invalid_model"
        )
        self.assertEqual(set(body), {"error"})
        self.assertIn("message", body["error"])
        self.assertIn("param", body["error"])

    def test_error_codes_are_stable(self) -> None:
        cases = (
            ({"model": "m"}, "missing_input"),
            ({"model": "m", "input": "x", "stream": True}, "streaming_not_supported"),
        )
        for payload, code in cases:
            with self.subTest(code=code):
                self.assert_error(self.post(payload), code)

    def test_health_endpoint_remains_intact(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_agent_prepare_endpoint_remains_intact(self) -> None:
        response = self.client.post(
            "/agent/prepare", json={"query": "public evidence"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["query"], "public evidence")

    def test_agent_prepare_keeps_its_existing_422_behavior(self) -> None:
        response = self.client.post("/agent/prepare", json={"query": "   "})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json(), {"detail": "query must not be empty"})

    def test_success_response_validates_against_local_schema(self) -> None:
        body = self.post({"model": "m", "input": "unknown"}).json()
        parsed = OpenResponsesResponse.model_validate(body)
        self.assertEqual(parsed.object, "response")

    def test_profile_json_is_valid(self) -> None:
        profile = json.loads(
            Path(__file__).resolve().parents[1]
            .joinpath("data", "profile.json")
            .read_text(encoding="utf-8")
        )
        self.assertIsInstance(profile, dict)


if __name__ == "__main__":
    unittest.main()
