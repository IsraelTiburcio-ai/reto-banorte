"""FastAPI application exposing the Phase 4, 5, and 6 HTTP boundaries."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.agent.core import AgentCore, AgentInputError
from app.api.open_responses import OpenResponsesAdapter
from app.api.schemas import AgentPrepareRequest, AgentPrepareResponse, HealthResponse
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

    @app.exception_handler(AgentInputError)
    async def handle_agent_input_error(
        _request: Request,
        exc: AgentInputError,
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @app.post("/agent/prepare", response_model=AgentPrepareResponse)
    def prepare_agent_turn(payload: AgentPrepareRequest) -> AgentPrepareResponse:
        core: AgentCore = app.state.agent_core
        turn = core.prepare(
            payload.query,
            max_results=payload.max_results,
        )
        return AgentPrepareResponse.from_turn(turn)

    @app.post("/v1/responses")
    async def create_open_response(request: Request) -> JSONResponse:
        adapter: OpenResponsesAdapter = app.state.open_responses_adapter
        try:
            payload = await request.json()
        except ValueError:
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
