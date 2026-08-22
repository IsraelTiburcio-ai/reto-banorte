from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
import unittest
from unittest.mock import patch

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from openai import APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError

from app.agent.core import AgentCore
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
from app.models.agent import AgentPolicy
from app.services.profile_service import ProfileService


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
    provider_request_attempted = True
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


class FailingProfileService(ProfileService):
    def search(self, query: str, visibility: str = "public") -> list[object]:
        raise RuntimeError("private profile path detail")


class LogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


class TrackingGenerator:
    provider_model = "tracked-provider-model"

    def __init__(self, text: str = "tracked generated response") -> None:
        self.text = text
        self.provider_request_attempted = False

    def generate(self, request: TextGenerationRequest) -> str:
        self.provider_request_attempted = True
        return self.text


async def call_asgi(
    app: object,
    path: str,
    body_chunks: list[bytes],
    headers: list[tuple[bytes, bytes]] | None = None,
    method: str = "POST",
) -> tuple[int, dict[str, str], bytes]:
    messages = [
        {
            "type": "http.request",
            "body": chunk,
            "more_body": index < len(body_chunks) - 1,
        }
        for index, chunk in enumerate(body_chunks)
    ]
    if not messages:
        messages = [{"type": "http.request", "body": b"", "more_body": False}]
    sent: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        return messages.pop(0)

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": headers or [],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "root_path": "",
    }
    await app(scope, receive, send)  # type: ignore[misc]
    start = next(message for message in sent if message["type"] == "http.response.start")
    response_body = b"".join(
        message.get("body", b"")
        for message in sent
        if message["type"] == "http.response.body"
    )
    response_headers = {
        key.decode("latin-1"): value.decode("latin-1")
        for key, value in start.get("headers", [])  # type: ignore[union-attr]
    }
    return int(start["status"]), response_headers, response_body  # type: ignore[index]


def run_asgi(
    app: object,
    path: str,
    body_chunks: list[bytes],
    headers: list[tuple[bytes, bytes]] | None = None,
    method: str = "POST",
) -> tuple[int, dict[str, str], bytes]:
    return asyncio.run(call_asgi(app, path, body_chunks, headers, method))


def padded_json(payload: object, size: int) -> bytes:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    if len(body) > size:
        raise AssertionError("payload is already larger than the requested size")
    return body + b" " * (size - len(body))


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
            stream_response = client.post(
                "/v1/responses",
                json={"input": "unknown topic", "stream": True},
                headers={"Authorization": "Bearer agent-key-placeholder"},
            )
            self.assertEqual(stream_response.status_code, 200)
            self.assertTrue(
                stream_response.headers["content-type"].startswith("text/event-stream")
            )
            self.assertEqual(
                client.post(
                    "/agent/prepare",
                    json={"query": "MCP"},
                    headers={"Authorization": "Bearer agent-key-placeholder"},
                ).status_code,
                200,
            )

    def test_authentication_precedes_body_validation(self) -> None:
        with patch.dict(os.environ, {"AGENT_API_KEY": "agent-key-placeholder"}, clear=False):
            client = TestClient(create_app(text_generator=StubGenerator()))
            requests = (
                ("/agent/prepare", None),
                ("/agent/prepare", {"content": b"{"}),
                ("/agent/prepare", {"json": {"query": 123}}),
                ("/agent/prepare", {"json": {"query": "MCP", "extra": True}}),
                ("/v1/responses", None),
                ("/v1/responses", {"content": b"{"}),
                ("/v1/responses", {"json": {"unexpected": True}}),
            )
            for path, kwargs in requests:
                with self.subTest(path=path, kwargs=kwargs):
                    response = client.post(path, **(kwargs or {}))
                    self.assertEqual(response.status_code, 401)
                    self.assertIn(REQUEST_ID_HEADER, response.headers)
                    self.assertEqual(
                        response.headers.get("WWW-Authenticate"), "Bearer"
                    )

            invalid_agent = client.post(
                "/agent/prepare",
                json={"query": 123},
                headers={"Authorization": "Bearer agent-key-placeholder"},
            )
            self.assertEqual(invalid_agent.status_code, 422)
            invalid_open_response = client.post(
                "/v1/responses",
                content=b"{",
                headers={
                    "Authorization": "Bearer agent-key-placeholder",
                    "Content-Type": "application/json",
                },
            )
            self.assertEqual(invalid_open_response.status_code, 400)

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
            "safe generated response",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_missing_provider_key_is_not_logged_as_an_outbound_attempt(self) -> None:
        logger = logging.getLogger(LOGGER_NAME)
        capture = LogCapture()
        logger.addHandler(capture)
        self.addCleanup(logger.removeHandler, capture)
        generator = OpenAITextGenerator(client_factory=lambda *_: self.fail("client called"))
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            response = TestClient(create_app(text_generator=generator)).post(
                "/v1/responses", json={"input": "MCP"}
            )
        self.assertEqual(response.status_code, 503)
        completed = [
            json.loads(message)
            for message in capture.messages
            if json.loads(message)["event"] == "request_completed"
        ][-1]
        self.assertFalse(completed["provider_invoked"])
        self.assertFalse(generator.provider_request_attempted)

    def test_real_generator_call_is_logged_as_an_outbound_attempt(self) -> None:
        logger = logging.getLogger(LOGGER_NAME)
        capture = LogCapture()
        logger.addHandler(capture)
        self.addCleanup(logger.removeHandler, capture)
        generator = TrackingGenerator()
        response = TestClient(create_app(text_generator=generator)).post(
            "/v1/responses", json={"input": "MCP"}
        )
        self.assertEqual(response.status_code, 200)
        completed = [
            json.loads(message)
            for message in capture.messages
            if json.loads(message)["event"] == "request_completed"
        ][-1]
        self.assertTrue(completed["provider_invoked"])
        self.assertTrue(generator.provider_request_attempted)

    def test_input_chars_include_full_transcript_text(self) -> None:
        logger = logging.getLogger(LOGGER_NAME)
        capture = LogCapture()
        logger.addHandler(capture)
        self.addCleanup(logger.removeHandler, capture)
        generator = TrackingGenerator()
        messages = [
            {"type": "message", "role": "user", "content": "a" * 1000},
            {"type": "message", "role": "assistant", "content": "b" * 1000},
            {"type": "message", "role": "user", "content": "MCP"},
        ]
        response = TestClient(create_app(text_generator=generator)).post(
            "/v1/responses", json={"input": messages}
        )
        self.assertEqual(response.status_code, 200)
        completed = [
            json.loads(message)
            for message in capture.messages
            if json.loads(message)["event"] == "request_completed"
        ][-1]
        self.assertEqual(completed["input_chars"], 2003)
        serialized = "\n".join(capture.messages)
        self.assertNotIn("a" * 1000, serialized)
        self.assertNotIn("b" * 1000, serialized)
        self.assertNotIn("tracked generated response", serialized)

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

    def test_readiness_returns_503_when_agent_core_is_unusable(self) -> None:
        app = create_app(text_generator=StubGenerator())
        app.state.agent_core._profile_service = None
        response = TestClient(app).get("/ready")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"status": "not_ready"})
        self.assertIn(REQUEST_ID_HEADER, response.headers)
        self.assertNotIn("profile", response.text.lower())

    def test_readiness_probe_fails_when_profile_search_is_unusable(self) -> None:
        app = create_app(
            agent_core=AgentCore(profile_service=FailingProfileService()),
            text_generator=StubGenerator(),
        )
        response = TestClient(app).get("/ready")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"status": "not_ready"})
        self.assertNotIn("private profile path detail", response.text)

    def test_readiness_rejects_missing_or_invalid_policy(self) -> None:
        for policy in (None, AgentPolicy(name="", objective="", rules=())):
            with self.subTest(policy=policy):
                app = create_app(text_generator=StubGenerator())
                app.state.agent_core._policy = policy
                response = TestClient(app).get("/ready")
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json(), {"status": "not_ready"})

    def test_readiness_rejects_missing_or_unusable_adapter(self) -> None:
        app = create_app(text_generator=StubGenerator())
        app.state.open_responses_adapter = None
        response = TestClient(app).get("/ready")
        self.assertEqual(response.status_code, 503)

        app = create_app(text_generator=StubGenerator())
        app.state.open_responses_adapter.create_response = None
        response = TestClient(app).get("/ready")
        self.assertEqual(response.status_code, 503)

    def test_readiness_rejects_core_adapter_mismatch(self) -> None:
        app = create_app(text_generator=StubGenerator())
        app.state.agent_core = AgentCore()
        response = TestClient(app).get("/ready")
        self.assertEqual(response.status_code, 503)

    def test_readiness_probe_does_not_mutate_local_components(self) -> None:
        app = create_app(text_generator=StubGenerator())
        core = app.state.agent_core
        profile = core._profile_service
        profile_before = copy.deepcopy(profile._profile)
        policy_before = core.policy
        response = TestClient(app).get("/ready")
        self.assertEqual(response.status_code, 200)
        self.assertIs(core.policy, policy_before)
        self.assertEqual(profile._profile, profile_before)

    def test_readiness_does_not_require_openai_key(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            response = TestClient(create_app()).get("/ready")
        self.assertEqual(response.status_code, 200)

    def test_body_limit_is_exact_and_applies_without_content_length(self) -> None:
        exact_body = padded_json({"input": "unknown topic"}, MAX_REQUEST_BODY_BYTES)
        over_body = exact_body + b"x"
        app = create_app(text_generator=StubGenerator())

        exact_status, exact_headers, exact_response = run_asgi(
            app, "/v1/responses", [exact_body]
        )
        self.assertEqual(len(exact_body), MAX_REQUEST_BODY_BYTES)
        self.assertEqual(exact_status, 200)
        self.assertRegex(exact_headers[REQUEST_ID_HEADER.lower()], r"^[0-9a-f]{32}$")
        self.assertEqual(json.loads(exact_response)["object"], "response")
        with patch.dict(os.environ, {"AGENT_API_KEY": ""}, clear=False):
            client_response = TestClient(app).post(
                "/v1/responses",
                content=exact_body,
                headers={"Content-Type": "application/json"},
            )
        self.assertEqual(client_response.status_code, 200)

        over_status, _, over_response = run_asgi(
            app, "/v1/responses", [over_body[:100], over_body[100:]]
        )
        self.assertEqual(over_status, 413)
        self.assertEqual(json.loads(over_response)["error"]["code"], "request_too_large")

        agent_status, _, agent_response = run_asgi(
            app, "/agent/prepare", [over_body[:10], over_body[10:]]
        )
        self.assertEqual(agent_status, 413)
        self.assertEqual(json.loads(agent_response)["detail"], "Request body is too large.")

    def test_body_limit_counts_bytes_not_unicode_characters(self) -> None:
        unicode_body = padded_json(
            {"input": "á" * 100}, MAX_REQUEST_BODY_BYTES
        )
        status, _, response = run_asgi(app=create_app(text_generator=StubGenerator()), path="/v1/responses", body_chunks=[unicode_body])
        self.assertEqual(len(unicode_body), MAX_REQUEST_BODY_BYTES)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(response)["object"], "response")

    def test_public_routes_keep_original_receive_and_skip_body_limit(self) -> None:
        app = create_app(text_generator=StubGenerator())

        async def public_echo(request: Request) -> JSONResponse:
            body = await request.body()
            return JSONResponse({"length": len(body)})

        app.add_api_route("/public-echo", public_echo, methods=["POST"])
        oversized_body = b"p" * (MAX_REQUEST_BODY_BYTES + 1)
        with patch.dict(os.environ, {"AGENT_API_KEY": "agent-key-placeholder"}, clear=False):
            status, _, response = run_asgi(app, "/public-echo", [oversized_body[:10], oversized_body[10:]])
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(response)["length"], len(oversized_body))

            health_status, _, _ = run_asgi(
                app, "/health", [oversized_body], method="GET"
            )
            ready_status, _, _ = run_asgi(
                app, "/ready", [oversized_body], method="GET"
            )
        self.assertEqual(health_status, 200)
        self.assertEqual(ready_status, 200)


if __name__ == "__main__":
    unittest.main()
