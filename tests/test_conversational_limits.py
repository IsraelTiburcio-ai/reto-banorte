from __future__ import annotations

import json
import unittest

from fastapi.testclient import TestClient

from app.agent.core import AgentCore
from app.api.main import create_app
from app.core.limits import (
    MAX_CONTENT_PARTS,
    MAX_GENERATION_HISTORY_CHARS,
    MAX_GENERATION_HISTORY_MESSAGES,
    MAX_MESSAGE_TEXT_CHARS,
    MAX_REQUEST_BODY_BYTES,
    MAX_TRANSCRIPT_MESSAGES,
)
from app.models.generation import TextGenerationRequest


class CountingGenerator:
    provider_model = "offline-conversational-limit-test"
    provider_request_attempted = False

    def __init__(self) -> None:
        self.requests: list[TextGenerationRequest] = []

    def generate(self, request: TextGenerationRequest) -> str:
        self.requests.append(request)
        return (
            "La evidencia pública describe experiencia profesional en datos, "
            "automatización y agentes. La respuesta de esta prueba mantiene "
            "el contexto recibido separado de la evidencia recuperada."
        )


def padded_json(payload: object, size: int) -> bytes:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    if len(body) > size:
        raise AssertionError("payload is already larger than the requested size")
    return body + b" " * (size - len(body))


def transcript(count: int) -> list[dict[str, str]]:
    messages = [
        {
            "role": "assistant",
            "type": "message",
            "content": (
                "Respuesta previa de contexto profesional: el perfil documenta "
                f"el intercambio histórico número {index}."
            ),
        }
        for index in range(count - 1)
    ]
    messages.append(
        {"role": "user", "type": "message", "content": "¿Qué experiencia tiene con MCP?"}
    )
    return messages


def content_parts(count: int) -> list[dict[str, object]]:
    return [
        {
            "role": "user",
            "type": "message",
            "content": [
                {"type": "input_text", "text": f"MCP parte textual {index}"}
                for index in range(count)
            ],
        }
    ]


class ConversationalLimitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = CountingGenerator()
        self.client = TestClient(
            create_app(
                agent_core=AgentCore(),
                text_generator=self.generator,
            )
        )

    def post_raw(self, body: bytes):
        return self.client.post(
            "/v1/responses",
            content=body,
            headers={"Content-Type": "application/json"},
        )

    def test_body_limit_exact_boundary_is_accepted(self) -> None:
        body = padded_json({"input": "MCP"}, MAX_REQUEST_BODY_BYTES)

        response = self.post_raw(body)

        self.assertEqual(len(response.request.content), MAX_REQUEST_BODY_BYTES)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.generator.requests), 1)

    def test_body_limit_one_byte_over_is_rejected_before_provider(self) -> None:
        body = padded_json({"input": "MCP"}, MAX_REQUEST_BODY_BYTES) + b" "

        response = self.post_raw(body)

        self.assertEqual(len(response.request.content), MAX_REQUEST_BODY_BYTES + 1)
        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.generator.requests, [])

    def test_body_over_64k_and_under_256k_is_accepted(self) -> None:
        body = padded_json({"input": "MCP"}, 64 * 1024 + 1)

        response = self.post_raw(body)

        self.assertGreater(len(response.request.content), 64 * 1024)
        self.assertLess(len(response.request.content), MAX_REQUEST_BODY_BYTES)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.generator.requests), 1)

    def test_individual_message_limit_is_preserved(self) -> None:
        response = self.client.post(
            "/v1/responses",
            json={"input": "MCP " + "x" * MAX_MESSAGE_TEXT_CHARS},
        )

        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.generator.requests, [])

    def test_received_message_boundary_is_128_and_not_generation_window(self) -> None:
        accepted = self.client.post(
            "/v1/responses", json={"input": transcript(MAX_TRANSCRIPT_MESSAGES)}
        )
        rejected = self.client.post(
            "/v1/responses",
            json={"input": transcript(MAX_TRANSCRIPT_MESSAGES + 1)},
        )

        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(rejected.status_code, 413)
        self.assertLessEqual(
            len(self.generator.requests[-1].transcript),
            MAX_GENERATION_HISTORY_MESSAGES,
        )
        self.assertLessEqual(
            sum(len(item.text) for item in self.generator.requests[-1].transcript),
            MAX_GENERATION_HISTORY_CHARS,
        )

    def test_content_part_boundary_is_32(self) -> None:
        accepted = self.client.post(
            "/v1/responses", json={"input": content_parts(MAX_CONTENT_PARTS)}
        )
        rejected = self.client.post(
            "/v1/responses",
            json={"input": content_parts(MAX_CONTENT_PARTS + 1)},
        )

        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(rejected.status_code, 413)

    def test_fifty_realistic_turns_remain_stateless_and_bounded(self) -> None:
        questions = tuple(
            f"En una revisión profesional, ¿qué experiencia con MCP respalda el "
            f"aspecto {index} del perfil de Israel?"
            for index in range(1, 51)
        )
        replay: list[dict[str, str]] = []
        request_sizes: dict[int, int] = {}
        transcript_sizes: dict[int, int] = {}

        for turn_number, question in enumerate(questions, start=1):
            replay.append({"role": "user", "type": "message", "content": question})
            response = self.client.post("/v1/responses", json={"input": replay})

            self.assertEqual(response.status_code, 200, question)
            if turn_number in (10, 20, 30, 50):
                request_sizes[turn_number] = len(response.request.content)
                transcript_sizes[turn_number] = len(replay)
            request = self.generator.requests[-1]
            self.assertEqual(request.query, question)
            self.assertNotIn(question, [message.text for message in request.transcript])
            self.assertLessEqual(
                len(request.transcript), MAX_GENERATION_HISTORY_MESSAGES
            )
            self.assertLessEqual(
                sum(len(message.text) for message in request.transcript),
                MAX_GENERATION_HISTORY_CHARS,
            )
            replay.append(
                {
                    "role": "assistant",
                    "type": "message",
                    "content": response.json()["output"][0]["content"][0]["text"],
                }
            )

        self.assertEqual(len(replay), 100)
        self.assertGreater(transcript_sizes[30], 32)
        self.assertLess(request_sizes[30], MAX_REQUEST_BODY_BYTES)
        self.assertLess(request_sizes[50], MAX_REQUEST_BODY_BYTES)
        self.assertEqual(len(self.generator.requests), 50)


if __name__ == "__main__":
    unittest.main()
