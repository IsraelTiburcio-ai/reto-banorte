from __future__ import annotations

import json
import logging
import os
import re
import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from openai import APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError

from app.api.main import create_app
from app.api.open_responses_schemas import OpenResponsesErrorEnvelope
from app.core.limits import (
    MAX_CONTENT_PARTS,
    MAX_INPUT_TEXT_CHARS,
    MAX_REQUEST_BODY_BYTES,
    MAX_TRANSCRIPT_MESSAGES,
)
from app.core.observability import LOGGER_NAME, REQUEST_ID_HEADER
from app.llm.errors import TextGenerationError
from app.llm.openai_provider import OpenAITextGenerator
from app.models.generation import TextGenerationRequest


class StubGenerator:
    provider_model = "test-provider-model"

    def generate(self, request: TextGenerationRequest) -> str:
        return "safe generated response"


class ExplodingGenerator:
    def generate(self, request: TextGenerationRequest) -> str:
        raise RuntimeError("private provider detail")


class SafeProviderError(TextGenerationError):
    code = "provider_test_failure"
    status_code = 502
    public_message = "Safe provider error."


class SafeErrorGenerator:
    provider_model = "test-provider-model"

    def generate(self, request: TextGenerationRequest) -> str:
        raise SafeProviderError("private provider detail")


class ErrorResponses:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def create(self, **kwargs: object) -> object:
        raise self.error


class ErrorClient:
    def __init__(self, error: Exception) -> None:
        self.responses = ErrorResponses(error)


class LogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


class SecurityObservabilityTests(unittest.TestCase):
    def test_auth_is_optional_when_agent_key_is_unset(self) -> None:
        with patch.dict(os.environ, {"AGENT_API_KEY": ""}, clear=False):
            client = TestClient(create_app(text_generator=StubGenerator()))
            self.assertEqual(
                client.post("/agent/prepare", json={"query": "MCP"}).status_code,
                200,
            )
            self.assertEqual(
                client.post(
                    "/v1/responses", json={"input": "unknown topic"}
                ).status_code,
                200,
            )

    def test_configured_agent_key_protects_agent_routes_but_not_health(self) -> None:
        with patch.dict(os.environ, {"AGENT_API_KEY": "agent-key-placeholder"}, clear=False):
            client = TestClient(create_app(text_generator=StubGenerator()))
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertEqual(client.get("/ready").status_code, 200)
            for path, payload in (
                ("/agent/prepare", {"query": "MCP"}),
                ("/v1/responses", {"input": "unknown topic"}),
            ):
                with self.subTest(path=path):
                    response = client.post(path, json=payload)
                    self.assertEqual(response.status_code, 401)
                    self.assertEqual(
                        response.headers.get("WWW-Authenticate"), "Bearer"
                    )
            self.assertEqual(
                client.post(
                    "/agent/prepare",
                    json={"query": "MCP"},
                    headers={"Authorization": "Bearer wrong-key"},
                ).status_code,
                401,
            )
            for authorization in ("Basic wrong-key", "Bearer"):
                with self.subTest(authorization=authorization):
                    self.assertEqual(
                        client.post(
                            "/v1/responses",
                            json={"input": "unknown topic"},
                            headers={"Authorization": authorization},
                        ).status_code,
                        401,
                    )
            self.assertEqual(
                client.post(
                    "/v1/responses",
                    json={"input": "unknown topic"},
                    headers={"Authorization": "Bearer agent-key-placeholder"},
                ).status_code,
                200,
            )

    def test_request_id_is_server_generated_and_not_client_supplied(self) -> None:
        with patch.dict(os.environ, {"AGENT_API_KEY": ""}, clear=False):
            client = TestClient(create_app(text_generator=StubGenerator()))
            first = client.get(
                "/health", headers={REQUEST_ID_HEADER: "client-controlled-id"}
            )
            second = client.get("/health")

        first_id = first.headers[REQUEST_ID_HEADER]
        second_id = second.headers[REQUEST_ID_HEADER]
        self.assertRegex(first_id, re.compile(r"^[0-9a-f]{32}$"))
        self.assertRegex(second_id, re.compile(r"^[0-9a-f]{32}$"))
        self.assertNotEqual(first_id, "client-controlled-id")
        self.assertNotEqual(first_id, second_id)

    def test_logs_are_structured_and_exclude_payloads_headers_and_secrets(self) -> None:
        logger = logging.getLogger(LOGGER_NAME)
        capture = LogCapture()
        logger.addHandler(capture)
        self.addCleanup(logger.removeHandler, capture)
        with patch.dict(
            os.environ,
            {
                "AGENT_API_KEY": "agent-key-placeholder",
                "OPENAI_API_KEY": "provider-key-placeholder",
            },
            clear=False,
        ):
            client = TestClient(create_app(text_generator=StubGenerator()))
            response = client.post(
                "/v1/responses",
                json={"input": "secret-query-not-for-logs\nwith-control\u0000"},
                headers={"Authorization": "Bearer agent-key-placeholder"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(capture.messages)
        parsed = [json.loads(message) for message in capture.messages]
        self.assertTrue(
            {item["event"] for item in parsed}
            >= {"request_started", "generation_started", "generation_completed", "request_completed"}
        )
        request_ids = {item["request_id"] for item in parsed}
        self.assertEqual(len(request_ids), 1)
        serialized = "\n".join(capture.messages)
        self.assertTrue(all("\n" not in message for message in capture.messages))
        for forbidden in (
            "agent-key-placeholder",
            "provider-key-placeholder",
            "secret-query-not-for-logs",
            "Authorization",
            "Cookie",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_provider_errors_and_unexpected_errors_are_sanitized(self) -> None:
        for generator, expected_status, expected_code in (
            (SafeErrorGenerator(), 502, "provider_test_failure"),
            (ExplodingGenerator(), 500, "internal_error"),
        ):
            with self.subTest(expected_code=expected_code):
                client = TestClient(create_app(text_generator=generator))
                response = client.post("/v1/responses", json={"input": "MCP"})
                self.assertEqual(response.status_code, expected_status)
                body = OpenResponsesErrorEnvelope.model_validate(response.json())
                self.assertEqual(body.error.code, expected_code)
                self.assertNotIn("private provider detail", response.text)
                self.assertNotIn("Traceback", response.text)

    def test_openai_provider_errors_are_safe_at_http_boundary(self) -> None:
        http_request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        http_response = httpx.Response(401, request=http_request)
        cases = (
            (
                AuthenticationError("private auth detail", response=http_response, body=None),
                502,
                "provider_authentication_failed",
            ),
            (APITimeoutError(request=http_request), 504, "provider_timeout"),
            (
                RateLimitError("private rate detail", response=http_response, body=None),
                429,
                "provider_rate_limited",
            ),
            (
                APIConnectionError(request=http_request),
                503,
                "provider_unavailable",
            ),
        )
        with patch.dict(os.environ, {"OPENAI_API_KEY": "provider-key-placeholder"}, clear=False):
            for provider_error, expected_status, expected_code in cases:
                with self.subTest(expected_code=expected_code):
                    generator = OpenAITextGenerator(
                        client_factory=lambda *_args, error=provider_error: ErrorClient(error)
                    )
                    client = TestClient(create_app(text_generator=generator))
                    response = client.post("/v1/responses", json={"input": "MCP"})
                    self.assertEqual(response.status_code, expected_status)
                    self.assertEqual(response.json()["error"]["code"], expected_code)
                    self.assertNotIn("private", response.text)
                    self.assertNotIn("provider-key-placeholder", response.text)

    def test_input_limits_are_conservative_and_route_specific(self) -> None:
        client = TestClient(create_app(text_generator=StubGenerator()))
        oversized_text = "x" * (MAX_INPUT_TEXT_CHARS + 1)
        response = client.post("/v1/responses", json={"input": oversized_text})
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["error"]["code"], "input_too_large")

        too_many_messages = [
            {"type": "message", "role": "user", "content": "MCP"}
            for _ in range(MAX_TRANSCRIPT_MESSAGES + 1)
        ]
        response = client.post("/v1/responses", json={"input": too_many_messages})
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["error"]["code"], "input_too_large")

        too_many_parts = [
            {"type": "input_text", "text": "MCP"}
            for _ in range(MAX_CONTENT_PARTS + 1)
        ]
        response = client.post(
            "/v1/responses",
            json={
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": too_many_parts,
                    }
                ]
            },
        )
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["error"]["code"], "input_too_large")

        response = client.post(
            "/v1/responses", json={"input": "x" * (MAX_REQUEST_BODY_BYTES + 1)}
        )
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["error"]["code"], "request_too_large")

        response = client.post(
            "/agent/prepare", json={"query": oversized_text}
        )
        self.assertEqual(response.status_code, 422)

    def test_readiness_is_public_deterministic_and_provider_free(self) -> None:
        class FailingGenerator:
            def generate(self, request: TextGenerationRequest) -> str:
                raise AssertionError("readiness must not call a provider")

        client = TestClient(create_app(text_generator=FailingGenerator()))
        self.assertEqual(client.get("/ready").status_code, 200)
        self.assertEqual(client.get("/ready").json(), {"status": "ready"})


if __name__ == "__main__":
    unittest.main()
