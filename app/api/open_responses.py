"""Adapter for the deliberately small synchronous Open Responses subset."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from pydantic import ValidationError

from app.agent.core import AgentCore, AgentInputError
from app.api.open_responses_formatter import INSUFFICIENT_EVIDENCE_TEXT
from app.api.open_responses_schemas import (
    OpenResponsesError,
    OpenResponsesErrorEnvelope,
    OpenResponsesRequest,
    OpenResponsesResponse,
    OpenResponsesOutputMessage,
    OpenResponsesOutputText,
    DEFAULT_MODEL_IDENTIFIER,
)
from app.llm.errors import EmptyProviderResponseError, TextGenerationError
from app.llm.openai_provider import OpenAITextGenerator
from app.core.limits import (
    MAX_CONTENT_PARTS,
    MAX_INPUT_TEXT_CHARS,
    MAX_TRANSCRIPT_MESSAGES,
)
from app.core.observability import log_event, set_request_fields
from app.models.agent import PreparedAgentTurn
from app.models.generation import ConversationMessage, TextGenerationRequest, TextGenerator


SUPPORTED_REQUEST_FIELDS = {"model", "input", "stream", "metadata"}
UNSUPPORTED_REQUEST_FIELDS = {
    "background",
    "compaction",
    "conversation",
    "frequency_penalty",
    "include",
    "instructions",
    "max_output_tokens",
    "max_tool_calls",
    "parallel_tool_calls",
    "previous_response_id",
    "prompt_cache_key",
    "reasoning",
    "safety_identifier",
    "service_tier",
    "store",
    "temperature",
    "text",
    "tools",
    "tool_choice",
    "top_logprobs",
    "top_p",
    "truncation",
    "visibility",
}
UNSUPPORTED_ITEM_TYPES = {
    "function_call",
    "function_call_output",
    "input_audio",
    "input_file",
    "input_image",
    "input_video",
    "tool_call",
    "tool_call_output",
}


@dataclass(frozen=True)
class TranscriptMessage:
    """Validated structural transcript data kept out of AgentCore instructions."""

    role: str
    text: str


class OpenResponsesRequestError(ValueError):
    """A request error that belongs only to the Open Responses endpoint."""

    def __init__(
        self,
        message: str,
        *,
        param: str,
        code: str,
        status_code: int = 400,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.param = param
        self.code = code
        self.status_code = status_code


class OpenResponsesAdapter:
    """Translate supported requests to ``AgentCore`` and back to local JSON."""

    def __init__(
        self,
        agent_core: AgentCore,
        text_generator: TextGenerator | None = None,
    ) -> None:
        self._agent_core = agent_core
        self._text_generator = text_generator or OpenAITextGenerator()

    def create_response(self, payload: object) -> tuple[dict[str, object], int]:
        try:
            request = self._validate_request(payload)
            current_user_query, transcript = self._extract_generation_context(
                request.input
            )
            turn = self._agent_core.prepare(current_user_query)
            response_text = self._generate_text(turn, transcript)
        except OpenResponsesRequestError as exc:
            set_request_fields(error_category=exc.code)
            return self.error_response(
                message=exc.message,
                param=exc.param,
                code=exc.code,
            ), exc.status_code
        except AgentInputError as exc:
            set_request_fields(error_category="invalid_request")
            return self.error_response(
                message=str(exc),
                param="input",
                code="invalid_input",
            ), 400
        except TextGenerationError as exc:
            set_request_fields(error_category=exc.code)
            return self.error_response(
                message=exc.public_message,
                param="input",
                code=exc.code,
            ), exc.status_code
        except Exception:
            set_request_fields(error_category="internal_error")
            log_event("request_failed", error_category="internal_error")
            return self.error_response(
                message="Internal server error.",
                param=None,
                code="internal_error",
            ), 500

        created_at = int(time.time())
        completed_at = max(created_at, int(time.time()))
        output_message = OpenResponsesOutputMessage(
            id=f"msg_{uuid.uuid4().hex}",
            content=[
                OpenResponsesOutputText(
                    text=response_text,
                )
            ],
        )
        response = OpenResponsesResponse(
            id=f"resp_{uuid.uuid4().hex}",
            created_at=created_at,
            completed_at=completed_at,
            model=request.model or DEFAULT_MODEL_IDENTIFIER,
            output=[output_message],
            metadata=request.metadata,
        )
        serialized = response.model_dump(mode="json", exclude_none=True)
        # The local subset makes the absence of model/token accounting explicit.
        serialized["error"] = None
        serialized["usage"] = None
        return serialized, 200

    def _generate_text(
        self,
        turn: PreparedAgentTurn,
        transcript: tuple[ConversationMessage, ...],
    ) -> str:
        provider_invoked = turn.status != "insufficient_evidence"
        provider_model = self._provider_model()
        set_request_fields(
            agent_status=turn.status,
            input_chars=len(turn.query),
            provider_invoked=provider_invoked,
            provider_model=provider_model,
        )
        generation_started = time.perf_counter()
        log_event(
            "generation_started",
            agent_status=turn.status,
            provider_invoked=provider_invoked,
            provider_model=provider_model,
        )
        if turn.status == "insufficient_evidence":
            text = INSUFFICIENT_EVIDENCE_TEXT
        else:
            try:
                text = self._text_generator.generate(
                    TextGenerationRequest(
                        query=turn.query,
                        transcript=transcript,
                        evidence=turn.evidence,
                        policy=turn.policy,
                    )
                )
            except TextGenerationError as exc:
                log_event(
                    "generation_failed",
                    agent_status=turn.status,
                    duration_ms=round(
                        (time.perf_counter() - generation_started) * 1000, 2
                    ),
                    error_category=exc.code,
                    provider_invoked=True,
                    provider_model=provider_model,
                )
                raise
            except Exception:
                log_event(
                    "generation_failed",
                    agent_status=turn.status,
                    duration_ms=round(
                        (time.perf_counter() - generation_started) * 1000, 2
                    ),
                    error_category="internal_error",
                    provider_invoked=True,
                    provider_model=provider_model,
                )
                raise

        if not isinstance(text, str) or not text.strip():
            error = EmptyProviderResponseError()
            log_event(
                "generation_failed",
                agent_status=turn.status,
                duration_ms=round(
                    (time.perf_counter() - generation_started) * 1000, 2
                ),
                error_category=error.code,
                provider_invoked=provider_invoked,
                provider_model=provider_model,
            )
            raise error
        log_event(
            "generation_completed",
            agent_status=turn.status,
            duration_ms=round((time.perf_counter() - generation_started) * 1000, 2),
            provider_invoked=provider_invoked,
            provider_model=provider_model,
        )
        return text

    def _provider_model(self) -> str | None:
        value = getattr(self._text_generator, "provider_model", None)
        return value if isinstance(value, str) and len(value) <= 128 else None

    @staticmethod
    def error_response(
        *, message: str, param: str | None, code: str
    ) -> dict[str, object]:
        return OpenResponsesErrorEnvelope(
            error=OpenResponsesError(message=message, param=param, code=code)
        ).model_dump(mode="json")

    @classmethod
    def _validate_request(cls, payload: object) -> OpenResponsesRequest:
        if not isinstance(payload, dict):
            raise OpenResponsesRequestError(
                "Request body must be a JSON object.",
                param="input",
                code="invalid_input",
            )

        unsupported_field = next(
            (field for field in payload if field in UNSUPPORTED_REQUEST_FIELDS),
            None,
        )
        if unsupported_field is not None:
            raise OpenResponsesRequestError(
                f"Request field '{unsupported_field}' is not supported by this subset.",
                param=unsupported_field,
                code="unsupported_feature",
            )

        unknown_fields = [
            field for field in payload if field not in SUPPORTED_REQUEST_FIELDS
        ]
        if unknown_fields:
            raise OpenResponsesRequestError(
                f"Request field '{unknown_fields[0]}' is not supported.",
                param=unknown_fields[0],
                code="invalid_input",
            )

        if "input" not in payload:
            raise OpenResponsesRequestError(
                "The input field is required.",
                param="input",
                code="missing_input",
            )

        raw_input = payload["input"]
        if not isinstance(raw_input, (str, list)):
            raise OpenResponsesRequestError(
                "Input must be a string or a list of messages.",
                param="input",
                code="unsupported_input_type",
            )
        if isinstance(raw_input, list) and any(
            not isinstance(item, dict) for item in raw_input
        ):
            raise OpenResponsesRequestError(
                "Every input list item must be a message object.",
                param="input",
                code="unsupported_input_type",
            )

        try:
            request = OpenResponsesRequest.model_validate(payload)
        except ValidationError as exc:
            locations = {
                str(error["loc"][0])
                for error in exc.errors()
                if error.get("loc")
            }
            if "model" in locations:
                raise OpenResponsesRequestError(
                    "Model must be a non-empty string.",
                    param="model",
                    code="invalid_model",
                ) from exc
            if "metadata" in locations:
                raise OpenResponsesRequestError(
                    "Metadata must be a compatible string map.",
                    param="metadata",
                    code="invalid_input",
                ) from exc
            raise OpenResponsesRequestError(
                "Request fields have invalid types.",
                param=next(iter(locations), "input"),
                code="invalid_input",
            ) from exc

        if request.stream:
            raise OpenResponsesRequestError(
                "Streaming is not supported in this synchronous subset.",
                param="stream",
                code="streaming_not_supported",
            )
        return request

    @classmethod
    def _extract_current_user_query(
        cls, input_value: str | list[dict[str, object]]
    ) -> str:
        query, _ = cls._extract_generation_context(input_value)
        return query

    @classmethod
    def _extract_generation_context(
        cls, input_value: str | list[dict[str, object]]
    ) -> tuple[str, tuple[ConversationMessage, ...]]:
        if isinstance(input_value, str):
            if not input_value.strip():
                raise OpenResponsesRequestError(
                    "Input must not be empty or whitespace-only.",
                    param="input",
                    code="invalid_input",
                )
            if len(input_value) > MAX_INPUT_TEXT_CHARS:
                raise OpenResponsesRequestError(
                    "Input text exceeds the maximum supported length.",
                    param="input",
                    code="input_too_large",
                    status_code=413,
                )
            return input_value, ()

        if not input_value:
            raise OpenResponsesRequestError(
                "Input message list must not be empty.",
                param="input",
                code="invalid_input",
            )
        if len(input_value) > MAX_TRANSCRIPT_MESSAGES:
            raise OpenResponsesRequestError(
                "Input contains too many transcript messages.",
                param="input",
                code="input_too_large",
                status_code=413,
            )

        transcript: list[TranscriptMessage] = []
        for index, item in enumerate(input_value):
            transcript.append(cls._parse_message(item, index))

        user_messages = [message.text for message in transcript if message.role == "user"]
        if not user_messages:
            raise OpenResponsesRequestError(
                "Input must contain at least one user message.",
                param="input",
                code="invalid_input",
            )
        if sum(len(message.text) for message in transcript) > MAX_INPUT_TEXT_CHARS:
            raise OpenResponsesRequestError(
                "Transcript text exceeds the maximum supported length.",
                param="input",
                code="input_too_large",
                status_code=413,
            )
        # Assistant history is validated structural data, never AgentCore input.
        return user_messages[-1], tuple(
            ConversationMessage(role=message.role, text=message.text)
            for message in transcript
        )

    @classmethod
    def _parse_message(
        cls, item: dict[str, object], index: int
    ) -> TranscriptMessage:
        item_param = f"input[{index}]"
        item_type = item.get("type")
        if item_type in UNSUPPORTED_ITEM_TYPES:
            raise OpenResponsesRequestError(
                f"Input item type '{item_type}' is not supported.",
                param=f"{item_param}.type",
                code="unsupported_feature",
            )
        if item_type != "message":
            raise OpenResponsesRequestError(
                "Only message input items are supported.",
                param=f"{item_param}.type",
                code="unsupported_input_type",
            )

        extra_fields = set(item) - {"type", "role", "content"}
        if extra_fields:
            field = sorted(extra_fields)[0]
            raise OpenResponsesRequestError(
                f"Message field '{field}' is not supported.",
                param=f"{item_param}.{field}",
                code="invalid_input",
            )

        role = item.get("role")
        if role in {"system", "developer"}:
            raise OpenResponsesRequestError(
                f"The {role} role is not supported in Phase 5.",
                param=f"{item_param}.role",
                code="unsupported_feature",
            )
        if role not in {"user", "assistant"}:
            raise OpenResponsesRequestError(
                "Only user and assistant message roles are supported.",
                param=f"{item_param}.role",
                code="unsupported_input_type",
            )

        text = cls._extract_message_text(item.get("content"), str(role), item_param)
        return TranscriptMessage(role=str(role), text=text)

    @classmethod
    def _extract_message_text(
        cls, content: object, role: str, item_param: str
    ) -> str:
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            if not content:
                raise OpenResponsesRequestError(
                    "Message content must not be empty.",
                    param=f"{item_param}.content",
                    code="invalid_input",
                )
            if len(content) > MAX_CONTENT_PARTS:
                raise OpenResponsesRequestError(
                    "Message contains too many content parts.",
                    param=f"{item_param}.content",
                    code="input_too_large",
                    status_code=413,
                )
            parts: list[str] = []
            expected_type = "input_text" if role == "user" else "output_text"
            for part_index, part in enumerate(content):
                part_param = f"{item_param}.content[{part_index}]"
                if not isinstance(part, dict):
                    raise OpenResponsesRequestError(
                        "Message content parts must be objects.",
                        param=part_param,
                        code="unsupported_input_type",
                    )
                extra_fields = set(part) - {"type", "text"}
                if extra_fields:
                    field = sorted(extra_fields)[0]
                    raise OpenResponsesRequestError(
                        f"Content field '{field}' is not supported.",
                        param=f"{part_param}.{field}",
                        code="invalid_input",
                    )
                part_type = part.get("type")
                if part_type in UNSUPPORTED_ITEM_TYPES:
                    raise OpenResponsesRequestError(
                        f"Content type '{part_type}' is not supported.",
                        param=f"{part_param}.type",
                        code="unsupported_feature",
                    )
                if part_type != expected_type:
                    raise OpenResponsesRequestError(
                        f"Expected {expected_type} content for {role} messages.",
                        param=f"{part_param}.type",
                        code="unsupported_input_type",
                    )
                part_text = part.get("text")
                if not isinstance(part_text, str):
                    raise OpenResponsesRequestError(
                        "Text content must be a string.",
                        param=f"{part_param}.text",
                        code="invalid_input",
                    )
                parts.append(part_text)
            text = " ".join(parts)
        else:
            raise OpenResponsesRequestError(
                "Message content must be a string or text parts.",
                param=f"{item_param}.content",
                code="unsupported_input_type",
            )

        if not text.strip():
            raise OpenResponsesRequestError(
                "Message content must not be empty or whitespace-only.",
                param=f"{item_param}.content",
                code="invalid_input",
            )
        return text
