"""Request correlation and safe JSON logging for stdout-based runtimes."""

from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
import uuid
from typing import Any


REQUEST_ID_HEADER = "X-Request-ID"
LOGGER_NAME = "reto_banorte"

_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)
_request_fields: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "request_fields", default=None
)
_logger = logging.getLogger(LOGGER_NAME)


def configure_logging() -> logging.Logger:
    """Configure one machine-readable stdout handler without logging payloads."""

    if not _logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        _logger.addHandler(handler)
    configured_level = os.getenv("LOG_LEVEL", "INFO").upper()
    _logger.setLevel(getattr(logging, configured_level, logging.INFO))
    _logger.propagate = False
    return _logger


def start_request_context() -> tuple[str, contextvars.Token, contextvars.Token]:
    request_id = uuid.uuid4().hex
    id_token = _request_id.set(request_id)
    fields_token = _request_fields.set({})
    return request_id, id_token, fields_token


def finish_request_context(
    id_token: contextvars.Token, fields_token: contextvars.Token
) -> None:
    _request_id.reset(id_token)
    _request_fields.reset(fields_token)


def current_request_id() -> str | None:
    return _request_id.get()


def set_request_fields(**fields: Any) -> None:
    current = _request_fields.get()
    if current is not None:
        current.update(fields)


def request_fields() -> dict[str, Any]:
    current = _request_fields.get()
    return dict(current) if current is not None else {}


def log_event(event: str, **fields: Any) -> None:
    """Emit only explicitly selected, non-payload fields as one JSON object."""

    payload: dict[str, Any] = {
        "event": event,
        "request_id": current_request_id(),
    }
    payload.update(fields)
    configure_logging().info(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
