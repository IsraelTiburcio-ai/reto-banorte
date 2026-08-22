"""Strict local schemas for the supported Open Responses subset."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, StrictBool, StrictStr, field_validator

from app.api.schemas import StrictApiModel


DEFAULT_MODEL_IDENTIFIER = "banorte-cv-agent"


class OpenResponsesRequest(StrictApiModel):
    """Request envelope for the synchronous textual/SSE local subset."""

    model: StrictStr | None = None
    input: StrictStr | list[dict[str, object]]
    stream: StrictBool | None = False
    store: StrictBool | None = None
    metadata: dict[StrictStr, StrictStr] | None = None

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("model must not be empty")
        return value

    @field_validator("metadata")
    @classmethod
    def validate_metadata(
        cls, value: dict[str, str] | None
    ) -> dict[str, str] | None:
        if value is None:
            return None
        if len(value) > 16:
            raise ValueError("metadata must contain at most 16 entries")
        for key, item in value.items():
            if not key or len(key) > 64:
                raise ValueError("metadata keys must be 1 to 64 characters")
            if len(item) > 512:
                raise ValueError("metadata values must be at most 512 characters")
        return value


class OpenResponsesOutputText(StrictApiModel):
    """The only output content part emitted by Phase 5."""

    type: Literal["output_text"] = "output_text"
    text: StrictStr
    annotations: list[dict[str, object]] = Field(default_factory=list)


class OpenResponsesOutputMessage(StrictApiModel):
    """A completed assistant message containing deterministic text."""

    id: StrictStr
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    status: Literal["completed"] = "completed"
    content: list[OpenResponsesOutputText]


class OpenResponsesResponse(StrictApiModel):
    """Local response shape, intentionally smaller than upstream Response."""

    id: StrictStr
    object: Literal["response"] = "response"
    created_at: int
    completed_at: int
    status: Literal["completed"] = "completed"
    model: StrictStr
    output: list[OpenResponsesOutputMessage]
    error: None = None
    usage: None = None
    metadata: dict[StrictStr, StrictStr] | None = None


class OpenResponsesError(StrictApiModel):
    """Stable error detail for requests rejected by this endpoint."""

    message: StrictStr
    type: Literal["invalid_request_error"] = "invalid_request_error"
    param: StrictStr | None = None
    code: StrictStr


class OpenResponsesErrorEnvelope(StrictApiModel):
    """Error envelope scoped to ``POST /v1/responses``."""

    error: OpenResponsesError
