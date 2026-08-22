"""FastAPI application exposing the HTTP, security, and observability boundaries."""

from __future__ import annotations

import json
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.agent.core import AgentCore, AgentInputError
from app.api.open_responses import OpenResponsesAdapter
from app.api.schemas import (
    AgentPrepareRequest,
    AgentPrepareResponse,
    HealthResponse,
    ReadinessResponse,
)
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
from app.models.generation import TextGenerator


def create_app(
    agent_core: AgentCore | None = None,
    text_generator: TextGenerator | None = None,
) -> FastAPI:
    """Create the HTTP application around a trusted ``AgentCore`` instance."""

    app = FastAPI(
        title="Reto IA Banorte — CV Agent",
        version="0.6.0",
        description="Phase 6 HTTP API for grounded CV-agent generation.",
    )
    app.state.agent_core = agent_core if agent_core is not None else AgentCore()
    app.state.open_responses_adapter = OpenResponsesAdapter(
        app.state.agent_core,
        text_generator=text_generator,
    )

    @app.middleware("http")
    async def request_observability(request: Request, call_next):
        request_id, id_token, fields_token = start_request_context()
        started = time.perf_counter()
        log_event(
            "request_started",
            method=request.method,
            path=request.url.path,
        )
        response = None
        try:
            content_length = request.headers.get("content-length")
            body_too_large = (
                content_length is not None
                and content_length.isdigit()
                and int(content_length) > MAX_REQUEST_BODY_BYTES
            )
            if body_too_large:
                set_request_fields(error_category="request_too_large")
                if request.url.path == "/v1/responses":
                    response = JSONResponse(
                        status_code=413,
                        content=app.state.open_responses_adapter.error_response(
                            message="Request body is too large.",
                            param="input",
                            code="request_too_large",
                        ),
                    )
                else:
                    response = JSONResponse(
                        status_code=413,
                        content={"detail": "Request body is too large."},
                    )
            else:
                response = await call_next(request)
        except Exception:
            set_request_fields(error_category="internal_error")
            log_event("request_failed", error_category="internal_error")
            response = JSONResponse(
                status_code=500,
                content={"detail": "Internal server error."},
            )
        finally:
            if response is not None:
                response.headers[REQUEST_ID_HEADER] = request_id
            completed_fields = request_fields()
            log_event(
                "request_completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code if response is not None else 500,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
                **completed_fields,
            )
            finish_request_context(id_token, fields_token)
        return response

    @app.exception_handler(AgentInputError)
    async def handle_agent_input_error(
        _request: Request,
        exc: AgentInputError,
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/ready", response_model=ReadinessResponse)
    def ready() -> ReadinessResponse:
        # App construction already initialized AgentCore and ProfileService.
        # Readiness is deterministic and never calls the provider.
        return ReadinessResponse()

    @app.post("/agent/prepare", response_model=AgentPrepareResponse)
    def prepare_agent_turn(request: Request, payload: AgentPrepareRequest):
        if not is_agent_authorized(request):
            set_request_fields(error_category="unauthorized")
            return JSONResponse(
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
                content={"detail": "Not authenticated."},
            )
        core: AgentCore = app.state.agent_core
        turn = core.prepare(
            payload.query,
            max_results=payload.max_results,
        )
        return AgentPrepareResponse.from_turn(turn)

    @app.post("/v1/responses")
    async def create_open_response(request: Request) -> JSONResponse:
        adapter: OpenResponsesAdapter = app.state.open_responses_adapter
        if not is_agent_authorized(request):
            set_request_fields(error_category="unauthorized")
            return JSONResponse(
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
                content=adapter.error_response(
                    message="Authentication is required.",
                    param="authorization",
                    code="unauthorized",
                ),
            )
        try:
            raw_body = await request.body()
            if len(raw_body) > MAX_REQUEST_BODY_BYTES:
                set_request_fields(error_category="request_too_large")
                return JSONResponse(
                    status_code=413,
                    content=adapter.error_response(
                        message="Request body is too large.",
                        param="input",
                        code="request_too_large",
                    ),
                )
            payload = json.loads(raw_body)
        except ValueError:
            set_request_fields(error_category="invalid_request")
            return JSONResponse(
                status_code=400,
                content=adapter.error_response(
                    message="Request body must contain valid JSON.",
                    param="input",
                    code="invalid_input",
                ),
            )

        body, status_code = adapter.create_response(payload)
        return JSONResponse(status_code=status_code, content=body)

    return app


app = create_app()
