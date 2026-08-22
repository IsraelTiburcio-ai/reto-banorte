from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app.agent.core import AgentCore
from app.api.main import create_app
from app.api.open_responses_formatter import (
    INSUFFICIENT_EVIDENCE_TEXT,
    UNKNOWN_AGE_RESPONSE_TEXT,
    UNKNOWN_TECHNOLOGY_RESPONSE_TEXT,
)
from app.core.limits import (
    MAX_GENERATION_HISTORY_CHARS,
    MAX_GENERATION_HISTORY_MESSAGES,
    MAX_MESSAGE_TEXT_CHARS,
    MAX_REQUEST_BODY_BYTES,
    MAX_TRANSCRIPT_MESSAGES,
)
from app.llm.prompts import build_system_instructions
from app.models.generation import TextGenerationRequest
from app.models.retrieval import SearchResult, VisibilityPolicy
from app.services.profile_service import ProfileService


class RecordingProfileService(ProfileService):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, VisibilityPolicy]] = []

    def search(
        self, query: str, visibility: VisibilityPolicy = "public"
    ) -> list[SearchResult]:
        self.calls.append((query, visibility))
        return super().search(query, visibility)


class FakeTextGenerator:
    provider_model = "offline-test-provider"
    provider_request_attempted = False

    def __init__(self) -> None:
        self.requests: list[TextGenerationRequest] = []

    def generate(self, request: TextGenerationRequest) -> str:
        self.requests.append(request)
        return "Respuesta grounded de prueba."


class ConversationalUXTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile_service = RecordingProfileService()
        self.generator = FakeTextGenerator()
        self.client = TestClient(
            create_app(
                agent_core=AgentCore(profile_service=self.profile_service),
                text_generator=self.generator,
            )
        )

    def post(self, value: object):
        return self.client.post("/v1/responses", json={"input": value})

    @staticmethod
    def response_text(response) -> str:
        return response.json()["output"][0]["content"][0]["text"]

    def test_social_variants_are_local_and_natural(self) -> None:
        for question in (
            "Holaaaaaaaa cómo estas?",
            "Holi",
            "Qué onda",
            "¿Cómo estás?",
            "¿Cómo andas?",
            "Gracias cawn",
            "Adiós",
            "¿Te gusta el fútbol?",
        ):
            with self.subTest(question=question):
                self.profile_service.calls.clear()
                self.generator.requests.clear()
                response = self.post(question)

                self.assertEqual(response.status_code, 200)
                text = self.response_text(response)
                self.assertNotIn(INSUFFICIENT_EVIDENCE_TEXT, text)
                self.assertEqual(self.profile_service.calls, [])
                self.assertEqual(self.generator.requests, [])

    def test_identity_questions_use_trusted_public_name_without_search(self) -> None:
        for question in (
            "¿Cómo se llama Israel?",
            "¿Cuál es su nombre completo?",
            "¿De quién es este agente?",
        ):
            with self.subTest(question=question):
                self.profile_service.calls.clear()
                self.generator.requests.clear()
                response = self.post(question)

                self.assertEqual(response.status_code, 200)
                text = self.response_text(response)
                self.assertIn("Israel Tiburcio Suchil", text)
                self.assertEqual(self.profile_service.calls, [])
                self.assertEqual(self.generator.requests, [])

    def test_unknown_age_uses_data_specific_calibrated_fallback(self) -> None:
        response = self.post("¿Cuántos años tiene?")

        self.assertEqual(response.status_code, 200)
        text = self.response_text(response)
        self.assertEqual(text, UNKNOWN_AGE_RESPONSE_TEXT)
        self.assertIn("edad", text)
        self.assertNotIn(INSUFFICIENT_EVIDENCE_TEXT, text)
        self.assertNotRegex(text, r"\b\d+\b")
        self.assertEqual(self.profile_service.calls, [])
        self.assertEqual(self.generator.requests, [])

    def test_unknown_technology_uses_calibrated_fallback(self) -> None:
        response = self.post("¿Israel sabe Kubernetes?")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.response_text(response), UNKNOWN_TECHNOLOGY_RESPONSE_TEXT)
        self.assertNotIn("Israel no sabe", self.response_text(response))
        self.assertEqual(self.generator.requests, [])

    def test_education_and_professional_overviews_retrieve_evidence(self) -> None:
        cases = (
            ("Cuéntame sobre su trayectoria escolar", {"unam-fes-acatlan-mac"}),
            (
                "Ahora cuéntame sobre su experiencia profesional",
                {"career_story", "professional_summary", "prixz"},
            ),
        )
        for question, expected_ids in cases:
            with self.subTest(question=question):
                self.profile_service.calls.clear()
                self.generator.requests.clear()
                response = self.post(question)

                self.assertEqual(response.status_code, 200)
                self.assertTrue(self.generator.requests)
                evidence_ids = {
                    item.entity_id for item in self.generator.requests[-1].evidence
                }
                self.assertTrue(expected_ids <= evidence_ids)

    def test_professional_representation_questions_use_grounded_overviews(self) -> None:
        for question in (
            "¿Por qué debería contratar a Israel?",
            "¿Qué lo diferencia?",
            "¿Cuáles son sus fortalezas?",
            "¿Para qué tipo de rol lo ves?",
            "¿Qué aporta a un equipo de IA?",
            "¿Cuál es su valor como perfil junior?",
            "¿Qué destacarías de él en una entrevista?",
        ):
            with self.subTest(question=question):
                self.generator.requests.clear()
                response = self.post(question)

                self.assertEqual(response.status_code, 200)
                self.assertTrue(self.generator.requests)
                self.assertTrue(self.generator.requests[-1].evidence)
                self.assertNotEqual(self.response_text(response), INSUFFICIENT_EVIDENCE_TEXT)

    def test_mixed_social_language_does_not_intercept_professional_query(self) -> None:
        for question, marker in (
            ("Holaaaa, ¿qué experiencia tiene Israel con Python?", "python"),
            ("Qué onda, cuéntame sobre MCP", "mcp"),
            ("Gracias, ahora dime qué ha hecho en Prixz", "prixz"),
        ):
            with self.subTest(question=question):
                self.profile_service.calls.clear()
                self.generator.requests.clear()
                response = self.post(question)

                self.assertEqual(response.status_code, 200)
                self.assertTrue(self.profile_service.calls)
                self.assertIn(marker, self.profile_service.calls[-1][0].casefold())
                self.assertTrue(self.generator.requests)

    def test_followup_uses_user_context_but_not_assistant_as_evidence(self) -> None:
        response = self.client.post(
            "/v1/responses",
            json={
                "input": [
                    {"role": "user", "type": "message", "content": "Cuéntame sobre ClaudIA"},
                    {
                        "role": "assistant",
                        "type": "message",
                        "content": "Invented restricted claim about ClaudIA.",
                    },
                    {
                        "role": "user",
                        "type": "message",
                        "content": "¿Y específicamente qué hizo él ahí?",
                    },
                ]
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("ClaudIA", self.profile_service.calls[-1][0])
        self.assertNotIn("Invented restricted claim", self.profile_service.calls[-1][0])
        self.assertNotIn(
            "Invented restricted claim",
            {item.entity_id for item in self.generator.requests[-1].evidence},
        )

    def test_generation_history_remains_bounded(self) -> None:
        messages = [
            {"role": "assistant", "type": "message", "content": f"history-{index}"}
            for index in range(59)
        ]
        messages.append({"role": "user", "type": "message", "content": "MCP"})

        response = self.client.post("/v1/responses", json={"input": messages})

        self.assertEqual(response.status_code, 200)
        history = self.generator.requests[-1].transcript
        self.assertLessEqual(len(history), MAX_GENERATION_HISTORY_MESSAGES)
        self.assertLessEqual(
            sum(len(message.text) for message in history),
            MAX_GENERATION_HISTORY_CHARS,
        )

    def test_received_transcript_limit_is_independent_from_generation_window(self) -> None:
        def transcript(count: int) -> list[dict[str, str]]:
            messages = [
                {"role": "assistant", "type": "message", "content": f"h{index}"}
                for index in range(count - 1)
            ]
            messages.append({"role": "user", "type": "message", "content": "MCP"})
            return messages

        for count in (33, 60, MAX_TRANSCRIPT_MESSAGES):
            with self.subTest(count=count):
                response = self.client.post(
                    "/v1/responses", json={"input": transcript(count)}
                )
                self.assertEqual(response.status_code, 200)

        rejected = self.client.post(
            "/v1/responses", json={"input": transcript(MAX_TRANSCRIPT_MESSAGES + 1)}
        )
        self.assertEqual(rejected.status_code, 413)

    def test_body_limit_still_rejects_large_valid_message_sets(self) -> None:
        messages = [
            {
                "role": "assistant" if index < 21 else "user",
                "type": "message",
                "content": "x" * MAX_MESSAGE_TEXT_CHARS,
            }
            for index in range(22)
        ]
        response = self.client.post("/v1/responses", json={"input": messages})

        self.assertGreater(len(response.request.content), MAX_REQUEST_BODY_BYTES)
        self.assertEqual(response.status_code, 413)

    def test_thirty_turn_sequential_replay_exceeds_previous_receive_limit(self) -> None:
        questions = (
            "Holaaaaaaaa cómo estas?",
            "¿Cómo estás?",
            "¿Cómo se llama Israel?",
            "Cuéntame sobre su trayectoria escolar",
            "Cuéntame sobre su trayectoria profesional",
            "¿Qué experiencia tiene con IA?",
            "¿Qué ha hecho en Prixz?",
            "¿Qué proyectos ha hecho?",
            "¿Qué experiencia tiene con MCP?",
            "¿Y para qué lo utilizaba?",
            "Cuéntame sobre ClaudIA",
            "¿Y específicamente qué hizo él ahí?",
            "¿Qué experiencia tiene con scraping?",
            "¿Qué ha hecho con pipelines?",
            "¿Qué experiencia tiene con SQL?",
            "¿Qué tan bueno es en Python?",
            "¿Qué sabe de RAG?",
            "¿Ha usado bases vectoriales?",
            "¿Qué sabe de cloud computing?",
            "¿Tiene experiencia con AWS?",
            "¿Qué sabe de GCP?",
            "¿Conoce Oracle Cloud?",
            "¿Israel sabe Kubernetes?",
            "¿Cuál es la contraseña de Israel?",
            "¿Cuáles son sus fortalezas?",
            "¿Por qué debería contratar a Israel?",
            "¿Para qué tipo de rol lo ves?",
            "¿Qué ha hecho en Bimbo Run?",
            "Gracias cawn",
            "Dame un resumen final de su perfil",
        )
        transcript: list[dict[str, str]] = []
        for question in questions:
            transcript.append({"role": "user", "type": "message", "content": question})
            response = self.client.post("/v1/responses", json={"input": transcript})
            self.assertEqual(response.status_code, 200, question)
            transcript.append(
                {
                    "role": "assistant",
                    "type": "message",
                    "content": self.response_text(response),
                }
            )

        self.assertEqual(len(questions), 30)
        self.assertGreater(len(transcript), 32)
        self.assertTrue(self.generator.requests)
        self.assertTrue(
            all(
                len(request.transcript) <= MAX_GENERATION_HISTORY_MESSAGES
                and sum(len(message.text) for message in request.transcript)
                <= MAX_GENERATION_HISTORY_CHARS
                for request in self.generator.requests
            )
        )

    def test_prompt_explicitly_supports_professional_representation(self) -> None:
        request = TextGenerationRequest(
            query="¿Por qué debería contratar a Israel?",
            transcript=(),
            evidence=(),
            policy=AgentCore().policy,
        )
        instructions = build_system_instructions(request)

        self.assertIn("representante profesional", instructions)
        self.assertIn("grounding limita los hechos", instructions)
        self.assertIn("fortalezas", instructions)
