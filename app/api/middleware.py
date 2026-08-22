"""ASGI boundary for request authentication, limits, and safe telemetry."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from app.api.open_responses import OpenResponsesAdapter
from app.core.limits import MAX_REQUEST_BODY_BYTES
from app.core.observability import (
    REQUEST_ID_HEADER,
    finish_request_context,
    log_event,
    request_fields,
    set_request_fields,
    start_request_context,
)
from app.core.security import is_agent_authorized


ASGIMessage = dict[str, Any]
Receive = Callable[[], Awaitable[ASGIMessage]]
Send = Callable[[ASGIMessage], Awaitable[None]]


class RequestBodyTooLarge(Exception):
    """Raised internally when a protected request exceeds its byte limit."""


class SecurityObservabilityMiddleware:
    """Enforce inbound boundaries before downstream body parsing begins."""

    _PROTECTED_PATHS = {"/agent/prepare", "/v1/responses"}

    def __init__(self, app: Any, open_responses_adapter: OpenResponsesAdapter) -> None:
        self.app = app
        self._open_responses_adapter = open_responses_adapter

    @classmethod
    def _is_protected(cls, scope: dict[str, Any]) -> bool:
        return (
            scope.get("method") == "POST"
            and scope.get("path") in cls._PROTECTED_PATHS
        )

    async def __call__(self, scope: dict[str, Any], receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        request_id, id_token, fields_token = start_request_context()
        started = time.perf_counter()
        request = Request(scope)
        protected = self._is_protected(scope)
        response_status: int | None = None

        async def send_with_request_id(message: ASGIMessage) -> None:
            nonlocal response_status
            if message.get("type") == "http.response.start":
                response_status = int(message["status"])
                response_message = dict(message)
                headers = [
                    (key, value)
                    for key, value in message.get("headers", [])
                    if key.lower() != REQUEST_ID_HEADER.lower().encode("ascii")
                ]
                headers.append(
                    (REQUEST_ID_HEADER.lower().encode("ascii"), request_id.encode("ascii"))
                )
                response_message["headers"] = headers
                message = response_message
            await send(message)

        log_event(
            "request_started",
            method=scope.get("method"),
            path=scope.get("path"),
        )
        try:
            if protected and not is_agent_authorized(request):
                set_request_fields(error_category="unauthorized")
                await self._unauthorized_response(scope, receive, send_with_request_id)
                return

            if protected and self._declared_body_is_too_large(request):
                set_request_fields(error_category="request_too_large")
                await self._body_too_large_response(
                    scope, receive, send_with_request_id
                )
                return

            replay_receive = await self._read_and_replay_body(receive)
            await self.app(
                scope,
                replay_receive if protected else receive,
                send_with_request_id,
            )
        except RequestBodyTooLarge:
            if response_status is not None:
                raise
            set_request_fields(error_category="request_too_large")
            await self._body_too_large_response(scope, receive, send_with_request_id)
        except Exception:
            if response_status is not None:
                raise
            set_request_fields(error_category="internal_error")
            log_event("request_failed", error_category="internal_error")
            await self._internal_error_response(scope, receive, send_with_request_id)
        finally:
            log_event(
                "request_completed",
                method=scope.get("method"),
                path=scope.get("path"),
                status_code=response_status if response_status is not None else 500,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
                **request_fields(),
            )
            finish_request_context(id_token, fields_token)

    @staticmethod
    def _declared_body_is_too_large(request: Request) -> bool:
        value = request.headers.get("content-length")
        return value is not None and value.isdigit() and int(value) > MAX_REQUEST_BODY_BYTES

    @staticmethod
    async def _read_and_replay_body(receive: Receive) -> Receive:
        chunks: list[bytes] = []
        received_bytes = 0
        while True:
            message = await receive()
            if message.get("type") == "http.disconnect":
                chunks = []
                break
            if message.get("type") != "http.request":
                continue
            body = message.get("body", b"")
            received_bytes += len(body)
            if received_bytes > MAX_REQUEST_BODY_BYTES:
                raise RequestBodyTooLarge()
            chunks.append(body)
            if not message.get("more_body", False):
                break

        replayed = False

        async def replay_receive() -> ASGIMessage:
            nonlocal replayed
            if replayed:
                return {"type": "http.disconnect"}
            replayed = True
            return {
                "type": "http.request",
                "body": b"".join(chunks),
                "more_body": False,
            }

        return replay_receive

    async def _unauthorized_response(
        self, scope: dict[str, Any], receive: Receive, send: Send
    ) -> None:
        if scope.get("path") == "/v1/responses":
            content = self._open_responses_adapter.error_response(
                message="Authentication is required.",
                param="authorization",
                code="unauthorized",
            )
        else:
            content = {"detail": "Not authenticated."}
        response = JSONResponse(
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
            content=content,
        )
        await response(scope, receive, send)

    async def _body_too_large_response(
        self, scope: dict[str, Any], receive: Receive, send: Send
    ) -> None:
        if scope.get("path") == "/v1/responses":
            content = self._open_responses_adapter.error_response(
                message="Request body is too large.",
                param="input",
                code="request_too_large",
            )
        else:
            content = {"detail": "Request body is too large."}
        await JSONResponse(status_code=413, content=content)(scope, receive, send)

    async def _internal_error_response(
        self, scope: dict[str, Any], receive: Receive, send: Send
    ) -> None:
        if scope.get("path") == "/v1/responses":
            content = self._open_responses_adapter.error_response(
                message="Internal server error.",
                param=None,
                code="internal_error",
            )
        else:
            content = {"detail": "Internal server error."}
        await JSONResponse(status_code=500, content=content)(scope, receive, send)
