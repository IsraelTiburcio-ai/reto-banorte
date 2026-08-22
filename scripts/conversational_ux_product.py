"""Run the 30-turn conversational UX product check locally.

The default mode uses a realistic fake provider and never calls OpenAI. Pass
``--live`` only when a local ``OPENAI_API_KEY`` is already loaded in the
process. The transcript is sent afresh on every request; this runner adds no
conversation memory of its own.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from dataclasses import dataclass

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.core import AgentCore
from app.api.main import create_app
from app.models.generation import TextGenerationRequest
from app.models.retrieval import SearchResult, VisibilityPolicy
from app.llm.openai_provider import OpenAITextGenerator
from app.services.profile_service import ProfileService


QUESTIONS = (
    "holaaaaaaaaa como andas?",
    "tu eres el agente de isra o q?",
    "como se llama completo?",
    "y que estudia este wey?",
    "cuentame mas o menos su trayectoria escolar",
    "y profesionalmente que ha hecho?",
    "como termino metido en inteligencia artificial?",
    "osea si esta cabron en ia o apenas anda aprendiendo?",
    "por que lo contratarias?",
    "que tiene diferente a otro jr?",
    "que ha hecho en prixz que valga la pena?",
    "y de esos cual dirias que demuestra mas nivel tecnico?",
    "cuentame de ese mcp de pedidos",
    "y eso para q servia exactamente?",
    "el lo hizo solo?",
    "y claudia q pedo que era?",
    "y el especificamente que hizo ahi?",
    "cambiando de tema sabe scrapear o no?",
    "y pipelines?",
    "sql que tal lo maneja?",
    "python si le sabe bien o nomas poquito?",
    "rag embeddings y esas madres si sabe?",
    "entonces pgvector lo ha usado en produccion?",
    "y de nube q sabe?",
    "aws?",
    "gcp tambien?",
    "oracle?",
    "kubernetes?",
    "bueno y fuera de todo eso cual dirias que es su mayor fortaleza?",
    "va cawn gracias, vendemelo en corto como si yo fuera el reclutador",
)


class RecordingProfileService(ProfileService):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, VisibilityPolicy, list[SearchResult]]] = []

    def search(
        self, query: str, visibility: VisibilityPolicy = "public"
    ) -> list[SearchResult]:
        results = super().search(query, visibility)
        self.calls.append((query, visibility, results))
        return results


class OfflineProductGenerator:
    provider_model = "offline-product-test-provider"
    provider_request_attempted = False

    def __init__(self) -> None:
        self.requests: list[TextGenerationRequest] = []

    def generate(self, request: TextGenerationRequest) -> str:
        self.requests.append(request)
        titles = ", ".join(item.title for item in request.evidence[:3])
        return (
            "Con base en la evidencia pública recuperada para este turno, "
            f"el perfil se relaciona con: {titles or 'la pregunta requiere revisión'}. "
            "Esta respuesta offline se marca para revisión semántica humana."
        )


class RecordingGenerator:
    """Record provider calls without changing the provider contract."""

    def __init__(self, delegate: object) -> None:
        self._delegate = delegate
        self.requests: list[TextGenerationRequest] = []

    @property
    def provider_model(self) -> str:
        return str(getattr(self._delegate, "provider_model"))

    @property
    def provider_request_attempted(self) -> bool:
        return bool(getattr(self._delegate, "provider_request_attempted", False))

    def generate(self, request: TextGenerationRequest) -> str:
        self.requests.append(request)
        return self._delegate.generate(request)  # type: ignore[attr-defined]


@dataclass(frozen=True)
class TurnReport:
    turn: int
    question: str
    status: int
    provider_invoked: bool
    evidence_count: int
    semantic: str
    notes: str
    text: str
    body_bytes: int
    transcript_messages: int
    generation_history_messages: int
    generation_history_chars: int


def _semantic_label(provider_invoked: bool, text: str) -> tuple[str, str]:
    if not provider_invoked:
        return "PASS", "Deterministic handling; verify naturalness manually."
    if text.strip():
        return "REVIEW", "Generated prose requires human grounding/calibration review."
    return "FAIL", "Empty response text."


def run(*, live: bool) -> list[TurnReport]:
    profile = RecordingProfileService()
    offline_generator = OfflineProductGenerator()
    agent_core = AgentCore(profile_service=profile)
    live_generator = (
        RecordingGenerator(OpenAITextGenerator()) if live else offline_generator
    )
    app = create_app(
        agent_core=agent_core,
        text_generator=live_generator,
    )
    generator = app.state.open_responses_adapter._text_generator
    client = TestClient(app)
    replay: list[dict[str, str]] = []
    reports: list[TurnReport] = []
    previous_generator_requests = 0

    for turn_number, question in enumerate(QUESTIONS, start=1):
        replay.append({"type": "message", "role": "user", "content": question})
        response = client.post("/v1/responses", json={"input": replay})
        text = ""
        evidence_count = 0
        history_messages = 0
        history_chars = 0
        if response.status_code == 200:
            body = response.json()
            text = body["output"][0]["content"][0]["text"]
            if hasattr(generator, "requests"):
                current_requests = len(generator.requests)
                provider_invoked = current_requests > previous_generator_requests
                previous_generator_requests = current_requests
                if provider_invoked:
                    request = generator.requests[-1]
                    evidence_count = len(request.evidence)
                    history_messages = len(request.transcript)
                    history_chars = sum(len(item.text) for item in request.transcript)
            if not provider_invoked:
                latest_calls = profile.calls
                evidence_count = len(latest_calls[-1][2]) if latest_calls else 0
            semantic, notes = _semantic_label(provider_invoked, text)
            replay.append(
                {"type": "message", "role": "assistant", "content": text}
            )
        else:
            provider_invoked = False
            semantic = "FAIL"
            notes = "Transport failure; no retry was attempted."

        reports.append(
            TurnReport(
                turn=turn_number,
                question=question,
                status=response.status_code,
                provider_invoked=provider_invoked,
                evidence_count=evidence_count,
                semantic=semantic,
                notes=notes,
                text=text,
                body_bytes=len(response.request.content),
                transcript_messages=len(replay),
                generation_history_messages=history_messages,
                generation_history_chars=history_chars,
            )
        )
        if response.status_code != 200:
            break

    return reports


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use the configured OpenAI provider; never retries and never prints the key.",
    )
    args = parser.parse_args()
    if args.live and not os.getenv("OPENAI_API_KEY", "").strip():
        print("LIVE STATUS: NOT_RUN (OPENAI_API_KEY is not loaded)")
        return 2

    reports = run(live=args.live)
    provider_model = (
        "configured provider"
        if args.live
        else "offline-product-test-provider"
    )
    print(f"MODE: {'LIVE' if args.live else 'OFFLINE_FAKE'}")
    print(f"PROVIDER MODEL: {provider_model}")
    print("turn | question | HTTP | semantic | provider | evidence | body_bytes | notes")
    for report in reports:
        print(
            f"{report.turn} | {report.question} | {report.status} | "
            f"{report.semantic} | {report.provider_invoked} | "
            f"{report.evidence_count} | {report.body_bytes} | {report.notes}"
        )
        print(f"RESPONSE {report.turn}: {report.text}")
    for turn in (10, 20, 30):
        matching = [report for report in reports if report.turn == turn]
        if matching:
            report = matching[0]
            print(
                f"SIZE TURN {turn}: {report.body_bytes} bytes; "
                f"{report.transcript_messages} transcript messages"
            )
    print(f"CALLS: {len(reports)}")
    print("NO LLM-AS-A-JUDGE: semantic generated checks are REVIEW.")
    return 0 if all(report.status == 200 for report in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
