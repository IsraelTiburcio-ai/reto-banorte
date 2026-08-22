"""Adapter for the deliberately small synchronous Open Responses subset."""

from __future__ import annotations

import json
import re
import time
import unicodedata
import uuid
from dataclasses import dataclass, replace

from pydantic import ValidationError

from app.agent.core import AgentCore, AgentInputError
from app.api.open_responses_formatter import (
    INSUFFICIENT_EVIDENCE_TEXT,
    deterministic_response_for,
)
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
    MAX_GENERATION_HISTORY_CHARS,
    MAX_GENERATION_HISTORY_MESSAGES,
    MAX_INPUT_TEXT_CHARS,
    MAX_MESSAGE_TEXT_CHARS,
    MAX_TRANSCRIPT_MESSAGES,
)
from app.core.observability import log_event, set_request_fields
from app.models.agent import PreparedAgentTurn
from app.models.generation import ConversationMessage, TextGenerationRequest, TextGenerator


SUPPORTED_REQUEST_FIELDS = {"model", "input", "stream", "store", "metadata"}
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


_FOLLOWUP_DEMONSTRATIVES = frozenset(
    {
        "ese",
        "esa",
        "eso",
        "esos",
        "esas",
        "ello",
        "ellos",
        "ellas",
        "estos",
        "estas",
        "aquel",
        "aquella",
        "aquellos",
        "aquellas",
    }
)
_FOLLOWUP_WHICH_TERMS = frozenset({"cual", "cuales"})
_FOLLOWUP_ANAPHORIC_VERBS = frozenset(
    {
        "usaba",
        "usabas",
        "utilizaba",
        "utilizabas",
        "empleaba",
        "empleabas",
        "gano",
        "ganaron",
        "fue",
        "fueron",
    }
)


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
    """Translate supported requests to ``AgentCore`` and local response transports."""

    def __init__(
        self,
        agent_core: AgentCore,
        text_generator: TextGenerator | None = None,
    ) -> None:
        self._agent_core = agent_core
        self._text_generator = text_generator or OpenAITextGenerator()

    @property
    def agent_core(self) -> AgentCore:
        """Expose the trusted core identity for local readiness checks."""

        return self._agent_core

    def create_response(self, payload: object) -> tuple[dict[str, object], int]:
        """Create the synchronous JSON response used by the local contract."""

        return self._create_response(payload, allow_stream=False)

    def create_stream_response(
        self, payload: object
    ) -> tuple[dict[str, object] | str, int]:
        """Create a complete response and serialize it as Open Responses SSE."""

        body, status_code = self._create_response(payload, allow_stream=True)
        if status_code != 200:
            return body, status_code
        return self._serialize_sse(body), status_code

    @staticmethod
    def is_stream_requested(payload: object) -> bool:
        """Select SSE only for the exact JSON boolean ``true``."""

        return isinstance(payload, dict) and payload.get("stream") is True

    def _create_response(
        self, payload: object, *, allow_stream: bool
    ) -> tuple[dict[str, object], int]:
        try:
            request = self._validate_request(payload, allow_stream=allow_stream)
            current_user_query, transcript, input_chars = self._extract_generation_context(
                request.input
            )
            deterministic_response = deterministic_response_for(current_user_query)
            if deterministic_response is not None:
                response_text, agent_status = deterministic_response
                set_request_fields(
                    agent_status=agent_status,
                    input_chars=input_chars,
                    provider_invoked=False,
                    provider_model=None,
                )
            else:
                turn = self._prepare_turn(current_user_query, transcript)
                response_text = self._generate_text(turn, transcript, input_chars)
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

    def _prepare_turn(
        self,
        current_user_query: str,
        transcript: tuple[ConversationMessage, ...],
    ) -> PreparedAgentTurn:
        """Use bounded prior user text only as retrieval context for follow-ups."""

        turn = self._agent_core.prepare(current_user_query)
        if not self._is_context_dependent_followup(current_user_query):
            return turn

        prior_user_text = [
            message.text
            for message in transcript
            if message.role == "user"
        ][-2:]
        if not prior_user_text:
            return turn

        contextual_query = " ".join((*prior_user_text, current_user_query))
        contextual_turn = self._agent_core.prepare(contextual_query)
        if contextual_turn.status != "ready":
            return turn
        # Keep the user-facing question current while using only prior user
        # text to recover public evidence for a stateless follow-up.
        return replace(contextual_turn, query=current_user_query)

    @staticmethod
    def _is_context_dependent_followup(query: str) -> bool:
        decomposed = unicodedata.normalize("NFKD", query)
        without_accents = "".join(
            character
            for character in decomposed
            if not unicodedata.combining(character)
        )
        normalized = re.sub(r"[^\w]+", " ", without_accents.casefold()).split()
        tokens = set(normalized)
        which_terms = tokens & _FOLLOWUP_WHICH_TERMS
        has_demonstrative = bool(tokens & _FOLLOWUP_DEMONSTRATIVES)
        has_anaphoric_verb = bool(tokens & _FOLLOWUP_ANAPHORIC_VERBS)
        has_explicit_topic = OpenResponsesAdapter._has_explicit_topic(query)
        if which_terms and has_explicit_topic:
            return False
        return bool(
            has_demonstrative
            or (which_terms and ("de" in tokens or has_anaphoric_verb))
            or (
                {"para", "que"} <= tokens
                and bool(tokens & {"lo", "la", "los", "las"})
                and has_anaphoric_verb
            )
        )

    @staticmethod
    def _has_explicit_topic(query: str) -> bool:
        """Recognize an explicit named topic without maintaining a topic list."""

        words = re.findall(r"[\w-]+", query, flags=re.UNICODE)
        for index, word in enumerate(words):
            if index == 0:
                continue
            if "-" in word or (word.isupper() and len(word) > 1):
                return True
            if word[:1].isupper() and word.casefold() not in {"cuál", "cual", "cuáles", "cuales"}:
                return True
        return False

    @staticmethod
    def _serialize_sse(response: dict[str, object]) -> str:
        output = response["output"]
        if not isinstance(output, list) or len(output) != 1:
            raise ValueError("The local response must contain one output message.")
        message = output[0]
        if not isinstance(message, dict):
            raise ValueError("The local response output message is invalid.")
        message_id = message.get("id")
        content = message.get("content")
        if (
            not isinstance(message_id, str)
            or not isinstance(content, list)
            or len(content) != 1
        ):
            raise ValueError("The local response output text is invalid.")
        output_text = content[0]
        if not isinstance(output_text, dict):
            raise ValueError("The local response output text is invalid.")
        text = output_text.get("text")
        if not isinstance(text, str):
            raise ValueError("The local response output text is invalid.")

        response_id = response.get("id")
        if not isinstance(response_id, str):
            raise ValueError("The local response id is invalid.")

        response_base = {
            key: value
            for key, value in response.items()
            if key not in {"completed_at", "status", "output"}
        }
        empty_message: list[dict[str, object]] = []
        message_in_progress = {
            "id": message_id,
            "type": "message",
            "status": "in_progress",
            "content": [],
            "role": "assistant",
        }
        empty_part = {
            "type": "output_text",
            "annotations": [],
            "text": "",
        }
        final_part = dict(output_text)
        final_message = {
            "id": message_id,
            "type": "message",
            "status": "completed",
            "content": [final_part],
            "role": "assistant",
        }

        def response_snapshot(
            status: str, response_output: list[dict[str, object]]
        ) -> dict[str, object]:
            snapshot = dict(response_base)
            snapshot["status"] = status
            snapshot["output"] = response_output
            return snapshot

        events: list[dict[str, object]] = []

        def add_event(event_type: str, **fields: object) -> None:
            events.append(
                {
                    "type": event_type,
                    "sequence_number": len(events),
                    **fields,
                }
            )

        add_event(
            "response.created",
            response=response_snapshot("queued", empty_message),
        )
        add_event(
            "response.in_progress",
            response=response_snapshot("in_progress", empty_message),
        )
        add_event(
            "response.output_item.added",
            output_index=0,
            item=message_in_progress,
        )
        add_event(
            "response.content_part.added",
            item_id=message_id,
            output_index=0,
            content_index=0,
            part=empty_part,
        )
        add_event(
            "response.output_text.delta",
            item_id=message_id,
            output_index=0,
            content_index=0,
            delta=text,
        )
        add_event(
            "response.output_text.done",
            item_id=message_id,
            output_index=0,
            content_index=0,
            text=text,
        )
        add_event(
            "response.content_part.done",
            item_id=message_id,
            output_index=0,
            content_index=0,
            part=final_part,
        )
        add_event(
            "response.output_item.done",
            output_index=0,
            item=final_message,
        )
        add_event("response.completed", response=response)

        frames = [
            "event: {event_type}\n"
            "data: {payload}\n\n".format(
                event_type=event["type"],
                payload=json.dumps(event, ensure_ascii=False, separators=(",", ":")),
            )
            for event in events
        ]
        frames.append("data: [DONE]\n\n")
        return "".join(frames)

    def _generate_text(
        self,
        turn: PreparedAgentTurn,
        transcript: tuple[ConversationMessage, ...],
        input_chars: int,
    ) -> str:
        provider_invoked = False
        provider_model = self._provider_model()
        set_request_fields(
            agent_status=turn.status,
            input_chars=input_chars,
            provider_invoked=provider_invoked,
            provider_model=provider_model,
        )
        generation_started = time.perf_counter()
        generation_fields: dict[str, object] = {
            "agent_status": turn.status,
            "provider_model": provider_model,
        }
        if turn.status == "insufficient_evidence":
            generation_fields["provider_invoked"] = False
        log_event("generation_started", **generation_fields)
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
                provider_invoked = self._provider_request_attempted()
                set_request_fields(provider_invoked=provider_invoked)
                log_event(
                    "generation_failed",
                    agent_status=turn.status,
                    duration_ms=round(
                        (time.perf_counter() - generation_started) * 1000, 2
                    ),
                    error_category=exc.code,
                    provider_invoked=provider_invoked,
                    provider_model=provider_model,
                )
                raise
            except Exception:
                provider_invoked = self._provider_request_attempted()
                set_request_fields(provider_invoked=provider_invoked)
                log_event(
                    "generation_failed",
                    agent_status=turn.status,
                    duration_ms=round(
                        (time.perf_counter() - generation_started) * 1000, 2
                    ),
                    error_category="internal_error",
                    provider_invoked=provider_invoked,
                    provider_model=provider_model,
                )
                raise

        if not isinstance(text, str) or not text.strip():
            error = EmptyProviderResponseError()
            provider_invoked = (
                False
                if turn.status == "insufficient_evidence"
                else self._provider_request_attempted()
            )
            set_request_fields(provider_invoked=provider_invoked)
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
        provider_invoked = (
            False
            if turn.status == "insufficient_evidence"
            else self._provider_request_attempted()
        )
        set_request_fields(provider_invoked=provider_invoked)
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

    def _provider_request_attempted(self) -> bool:
        """Read optional provider telemetry without coupling to provider secrets."""

        try:
            return (
                getattr(self._text_generator, "provider_request_attempted", False)
                is True
            )
        except Exception:
            return False

    @staticmethod
    def error_response(
        *, message: str, param: str | None, code: str
    ) -> dict[str, object]:
        return OpenResponsesErrorEnvelope(
            error=OpenResponsesError(message=message, param=param, code=code)
        ).model_dump(mode="json")

    @classmethod
    def _validate_request(
        cls, payload: object, *, allow_stream: bool = False
    ) -> OpenResponsesRequest:
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
            if "store" in locations:
                raise OpenResponsesRequestError(
                    "Store must be a boolean or null.",
                    param="store",
                    code="invalid_input",
                ) from exc
            raise OpenResponsesRequestError(
                "Request fields have invalid types.",
                param=next(iter(locations), "input"),
                code="invalid_input",
            ) from exc

        if request.stream and not allow_stream:
            raise OpenResponsesRequestError(
                "Streaming is not supported in this synchronous subset.",
                param="stream",
                code="streaming_not_supported",
            )
        if request.store is True:
            raise OpenResponsesRequestError(
                "Stateful storage is not supported; this agent is stateless. "
                "Use store=false or omit the field.",
                param="store",
                code="unsupported_feature",
            )
        return request

    @classmethod
    def _extract_current_user_query(
        cls, input_value: str | list[dict[str, object]]
    ) -> str:
        query, _, _ = cls._extract_generation_context(input_value)
        return query

    @classmethod
    def _extract_generation_context(
        cls, input_value: str | list[dict[str, object]]
    ) -> tuple[str, tuple[ConversationMessage, ...], int]:
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
            return input_value, (), len(input_value)

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
        input_chars = sum(len(message.text) for message in transcript)
        # The full validated transcript may exceed the old aggregate text
        # budget as long as the HTTP body and per-message limits still hold.
        # Keep only a bounded suffix for generation and contextual retrieval;
        # the current user question is carried separately and is never trimmed.
        current_user_query = user_messages[-1]
        prior_messages = transcript[:]
        for index in range(len(prior_messages) - 1, -1, -1):
            if prior_messages[index].role == "user" and prior_messages[index].text == current_user_query:
                prior_messages = prior_messages[:index]
                break
        return (
            current_user_query,
            cls._bound_history(
                tuple(
                    ConversationMessage(role=message.role, text=message.text)
                    for message in prior_messages
                )
            ),
            input_chars,
        )

    @staticmethod
    def _bound_history(
        transcript: tuple[ConversationMessage, ...],
    ) -> tuple[ConversationMessage, ...]:
        """Keep recent complete history within message and character budgets."""

        selected: list[ConversationMessage] = []
        used_chars = 0
        for message in reversed(transcript):
            if len(selected) >= MAX_GENERATION_HISTORY_MESSAGES:
                break
            if used_chars + len(message.text) > MAX_GENERATION_HISTORY_CHARS:
                continue
            selected.append(message)
            used_chars += len(message.text)
        selected.reverse()
        return tuple(selected)

    @classmethod
    def _parse_message(
        cls, item: dict[str, object], index: int
    ) -> TranscriptMessage:
        item_param = f"input[{index}]"
        item_type = item.get("type")
        if isinstance(item_type, str) and item_type in UNSUPPORTED_ITEM_TYPES:
            raise OpenResponsesRequestError(
                f"Input item type '{item_type}' is not supported.",
                param=f"{item_param}.type",
                code="unsupported_feature",
            )
        if item_type is not None and item_type != "message":
            raise OpenResponsesRequestError(
                "Only message input items are supported.",
                param=f"{item_param}.type",
                code="unsupported_input_type",
            )

        extra_fields = set(item) - {
            "id",
            "type",
            "role",
            "status",
            "content",
        }
        if extra_fields:
            field = sorted(extra_fields)[0]
            raise OpenResponsesRequestError(
                f"Message field '{field}' is not supported.",
                param=f"{item_param}.{field}",
                code="invalid_input",
            )

        item_id = item.get("id")
        if item_id is not None and (
            not isinstance(item_id, str)
            or not re.fullmatch(r"msg_[A-Za-z0-9_-]+", item_id)
        ):
            raise OpenResponsesRequestError(
                "Message id must be a valid msg_ identifier.",
                param=f"{item_param}.id",
                code="invalid_input",
            )

        status = item.get("status")
        if status is not None and (
            not isinstance(status, str)
            or status not in {"in_progress", "completed", "incomplete"}
        ):
            raise OpenResponsesRequestError(
                "Message status is not valid for transcript replay.",
                param=f"{item_param}.status",
                code="invalid_input",
            )

        role = item.get("role")
        if isinstance(role, str) and role in {"system", "developer"}:
            raise OpenResponsesRequestError(
                f"The {role} role is not supported in Phase 5.",
                param=f"{item_param}.role",
                code="unsupported_feature",
            )
        if not isinstance(role, str) or role not in {"user", "assistant"}:
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
                extra_fields = set(part) - {"type", "text", "annotations"}
                if extra_fields:
                    field = sorted(extra_fields)[0]
                    raise OpenResponsesRequestError(
                        f"Content field '{field}' is not supported.",
                        param=f"{part_param}.{field}",
                        code="invalid_input",
                    )
                part_type = part.get("type")
                if isinstance(part_type, str) and part_type in UNSUPPORTED_ITEM_TYPES:
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
                annotations = part.get("annotations")
                if annotations is not None and (
                    not isinstance(annotations, list)
                    or any(
                        not isinstance(annotation, dict)
                        for annotation in annotations
                    )
                ):
                    raise OpenResponsesRequestError(
                        "Annotations must be a list of objects.",
                        param=f"{part_param}.annotations",
                        code="invalid_input",
                    )
                part_text = part.get("text")
                if not isinstance(part_text, str):
                    raise OpenResponsesRequestError(
                        "Text content must be a string.",
                        param=f"{part_param}.text",
                        code="invalid_input",
                    )
                if len(part_text) > MAX_MESSAGE_TEXT_CHARS:
                    raise OpenResponsesRequestError(
                        "Message text exceeds the maximum supported length.",
                        param=part_param,
                        code="input_too_large",
                        status_code=413,
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
        if len(text) > MAX_MESSAGE_TEXT_CHARS:
            raise OpenResponsesRequestError(
                "Message text exceeds the maximum supported length.",
                param=f"{item_param}.content",
                code="input_too_large",
                status_code=413,
            )
        return text
